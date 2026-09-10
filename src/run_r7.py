"""
R-7: environment geometry x policy algebra, a 5 x 5 factorial.

    python run_r7.py --selftest
    python run_r7.py --seeds 3         # ~45 min on an RTX 5060
    python run_r7.py --analyse-only

Requires _r1core.py and _armlib.py.  No edits to main.py.

Motivation
----------
Every Task-1 result so far plants an SO(32) rotation in the environment and
finds that an so(32) policy wins.  R-4/R-6 controlled the algebra, R-5 the
learning rate, X-3 the particular rotation drawn, B-1 the retraction.  None of
them controlled the environment's geometry.  So the obvious question is
still open:

    Is this a result about compactness, or just about matching whatever
    structure the environment happens to have?

The two answers imply different claims:

  so(32) wins across geometries   -> the claim is about the compact algebra as
                                     an optimisation constraint.  Much stronger
                                     than the paper currently claims, and it is
                                     what the compactness theory predicts.
  so(32) wins only on SO          -> the claim is "match the environment's
                                     structure", which is unsurprising and
                                     narrows the scope of the claim.

Design
------
  environment geometry   M_env drawn as ...
    so     exp(skew)                 in SO(32)          <- as released
    sym    exp(symmetric)            SPD, not orthogonal
    gl     random Gaussian           invertible, generic
    diag   random positive diagonal  axis-aligned scaling
    id     identity                  no geometry to learn

  policy arm   baseline_ppo, so, sl, gl, sym       (as in R-4)
  seeds        0 .. n-1, shared across every cell so the design is paired

Changing M_env changes both the geometry reward target and the auxiliary-loss
target, because main.py's auxiliary loss regresses on env.M_env (this is the
S-7 finding).  That is the correct coupling for this experiment: the
environment's structure changes and the supervised target follows it.

What is and is not comparable
-----------------------------
AUC is not comparable across geometries: a symmetric or singular M_env makes a
different amount of geometry reward attainable, so the columns have different
ceilings.  Only the ranking of algebras within a geometry is meaningful, and
the interaction (does so(32)'s margin shrink off the SO geometry?) is tested
by pairing on seed.

Output   r7_cells.csv   geom, arm, seed, auc, r_task, r_geo, rho, menv_*, status
"""
from __future__ import annotations

import argparse
import io
import contextlib
import os
import sys

import numpy as np
import pandas as pd

from _armlib import (BASELINE, Sink, check_meta, clean, guard_outputs, paired,
                     report, run_one, write_meta)

CSV = "r7_cells.csv"
GEOMS = ("so", "sym", "gl", "diag", "id")
ARMS = (BASELINE, "so", "sl", "gl", "sym")
ALGS = tuple(a for a in ARMS if a != BASELINE)
W, HORIZON = 0.4, 20
MENV_SEED0 = 9000
COLS = ["geom", "arm", "seed", "auc", "r_task", "r_geo", "rho",
        "menv_orth", "menv_sym", "menv_cond", "recon_err", "status"]


# --------------------------------------------------------------------------- #
def draw_m_env(geom: str, k: int, seed: int):
    """Return (M, diagnostics) for one environment geometry, in float64.

    `so` reproduces main.py's own construction exactly (exp of a random skew
    matrix); the others are the same construction with the skew-symmetrisation
    replaced, so the four families are matched in scale and in RNG treatment
    and differ only in structure.
    """
    import torch
    from scipy.linalg import expm

    rng = np.random.RandomState(int(seed))
    S = rng.randn(k, k) * 0.5
    if geom == "so":
        M = expm(0.5 * (S - S.T))              # exactly main.py's M_env
    elif geom == "sym":
        M = expm(0.5 * (S + S.T))              # SPD, emphatically not a rotation
    elif geom == "gl":
        M = S.copy()
    elif geom == "diag":
        M = np.diag(np.exp(0.5 * np.diag(S)))  # positive axis-aligned scaling
    elif geom == "id":
        M = np.eye(k)
    else:
        raise ValueError(f"unknown geometry {geom!r}; expected one of {GEOMS}")

    M = np.asarray(M, dtype=np.float64)
    I = np.eye(k)
    diag = dict(
        menv_orth=float(np.abs(M.T @ M - I).max()),
        menv_sym=float(np.abs(M - M.T).max()),
        menv_cond=float(np.linalg.cond(M)),
    )
    # A singular M_env would make the geometry reward undefined on some
    # directions; refuse rather than produce rows that look fine.
    if not np.isfinite(diag["menv_cond"]) or diag["menv_cond"] > 1e8:
        raise RuntimeError(f"{geom}: M_env is numerically singular "
                           f"(cond={diag['menv_cond']:.3e}); redraw or widen "
                           f"the construction.")
    # Each family must actually be what it claims, or the factorial is a lie.
    if geom == "so" and diag["menv_orth"] > 1e-9:
        raise RuntimeError(f"so: M_env not orthogonal ({diag['menv_orth']:.2e})")
    if geom in ("sym", "diag") and diag["menv_sym"] > 1e-9:
        raise RuntimeError(f"{geom}: M_env not symmetric ({diag['menv_sym']:.2e})")
    if geom == "sym" and diag["menv_orth"] < 1e-3:
        raise RuntimeError("sym: M_env came out orthogonal; it would duplicate "
                           "the so cell instead of contrasting with it.")
    if geom == "gl" and (diag["menv_orth"] < 1e-3 or diag["menv_sym"] < 1e-3):
        raise RuntimeError("gl: M_env is orthogonal or symmetric; it would "
                           "duplicate another cell.")
    return torch.tensor(M, dtype=torch.float64), diag


def collect(n_seeds: int, overwrite: bool) -> pd.DataFrame:
    guard_outputs([CSV], overwrite)
    from main import GPT2EmbeddingProvider, MultiStepTextAlignmentEnv
    from _r1core import instrument

    emb = GPT2EmbeddingProvider(model_name="gpt2-medium", max_length=64)
    env = MultiStepTextAlignmentEnv(emb, n_prompts=16, n_actions=16,
                                    horizon=HORIZON, reward_noise=0.2,
                                    geo_weight=W, sparse_prob=0.7)
    instrument(env)
    sd, na, k = emb.hidden_dim, env.n_actions, int(env.k_transform)
    orig = env.M_env.clone()

    sink, rows = Sink(CSV, COLS), []
    write_meta(CSV, dict(geoms=list(GEOMS), arms=list(ARMS), w=W,
                         horizon=HORIZON, n_seeds=n_seeds,
                         menv_seed0=MENV_SEED0))
    total, done = len(GEOMS) * len(ARMS) * n_seeds, 0
    try:
        for gi, geom in enumerate(GEOMS):
            # One M_env per geometry, shared by every arm and seed in the
            # column, so the algebra contrast within a column is exact.
            M, diag = draw_m_env(geom, k, MENV_SEED0 + gi)
            env.M_env = M.to(device=orig.device, dtype=orig.dtype)
            print(f"\n--- geometry {geom}: |MtM-I|={diag['menv_orth']:.2e}  "
                  f"|M-Mt|={diag['menv_sym']:.2e}  "
                  f"cond={diag['menv_cond']:.3e}", flush=True)
            for seed in range(n_seeds):
                for arm in ARMS:
                    rec, _, _ = run_one(env, arm, seed, sd, na, k, W, HORIZON,
                                        extra=dict(geom=geom, **diag))
                    sink.add([{c: rec.get(c, np.nan) for c in COLS}])
                    rows.append(rec); done += 1
                    msg = (f"FAILED: {rec['status']}" if rec["status"] != "ok"
                           else f"AUC={rec['auc']:7.3f}  "
                                f"r_geo={rec['r_geo']:.4f}  "
                                f"rho={rec['rho']:.4g}")
                    print(f"  [{done:3d}/{total}] {geom:5s} s{seed} "
                          f"{arm:12s} {msg}", flush=True)
    finally:
        env.M_env = orig            # leave the env exactly as we found it
    df = pd.DataFrame(rows).reindex(columns=COLS)
    print(f"\nWrote {CSV} ({sink.n} rows)")
    return df


# --------------------------------------------------------------------------- #
def analyse(df: pd.DataFrame) -> None:
    e = pd.to_numeric(df.get("recon_err"), errors="coerce") if "recon_err" in df else None
    if e is not None and np.isfinite(e).any():
        print(f"\nchannel-reconstruction error: max = "
              f"{e[np.isfinite(e)].max():.3e}")
    bad = df[df.status != "ok"]
    if len(bad):
        print(f"\n{len(bad)} failed run(s):")
        for _, r in bad.iterrows():
            print(f"    {r.get('geom')} {r.arm} seed={r.seed}: {r.status}")
    d = df[df.status == "ok"]
    geoms = [g for g in GEOMS if g in set(d["geom"])]

    print("\n" + "=" * 100)
    print("R-7  MEAN AUC:  ENVIRONMENT GEOMETRY (rows) x POLICY ALGEBRA (cols)")
    print("  AUC is NOT comparable across ROWS -- each geometry makes a "
          "different amount of\n  geometry reward attainable.  Compare within "
          "a row only.")
    print("=" * 100)
    print(f"  {'geometry':>9s}" + "".join(f"{a:>15s}" for a in ARMS)
          + f"{'winner':>10s}")
    winner = {}
    for g in geoms:
        cells, means = [], {}
        for a in ARMS:
            v = d[(d.geom == g) & (d.arm == a)]["auc"].astype(float).dropna()
            cells.append(f"{v.mean():7.3f}+/-{v.std(ddof=1):5.3f}"
                         if len(v) > 1 else
                         (f"{v.mean():>15.3f}" if len(v) else f"{'n/a':>15s}"))
            if len(v):
                means[a] = v.mean()
        winner[g] = max(means, key=means.get) if means else "n/a"
        print(f"  {g:>9s}" + "".join(f"{c:>15s}" for c in cells)
              + f"{winner[g]:>10s}")

    # ---- within-geometry contrasts, paired on seed -------------------------
    print("\n" + "=" * 100)
    print("R-7  WITHIN EACH GEOMETRY:  so(32) minus each comparator, paired by "
          "seed")
    print("=" * 100)
    margins = {}
    for g in geoms:
        dg = clean(d[d.geom == g].copy(), "auc", "seed", ARMS)
        if not len(dg) or "so" not in set(dg.arm):
            print(f"\n  {g}: no complete paired block."); continue
        rows_ = [dict(label=f"[{g}] so - {a}",
                      d=paired(dg, "so", a, "auc", "seed")[0])
                 for a in ARMS if a != "so"]
        out = report(rows_, f"R-7  geometry = {g}",
                     "  Holm-corrected within this geometry.")
        # margin over the best NON-so algebra: the quantity that decides
        # whether so(32)'s advantage is specific to an SO environment
        others = [a for a in ALGS if a != "so"]
        per_seed = {}
        for s_ in sorted(dg[dg.arm == "so"]["seed"].unique()):
            row = dg[dg.seed == s_]
            so_v = row[row.arm == "so"]["auc"].astype(float)
            ot = row[row.arm.isin(others)]["auc"].astype(float)
            if len(so_v) == 1 and len(ot):
                per_seed[s_] = float(so_v.iloc[0]) - float(ot.max())
        margins[g] = per_seed
        if per_seed:
            v = np.array(list(per_seed.values()), dtype=float)
            print(f"\n  [{g}] so(32) minus the best other algebra: "
                  f"{v.mean():+.4f} +/- {v.std(ddof=1) if len(v) > 1 else float('nan'):.4f} "
                  f"(n={len(v)} seeds)")
        _ = out

    # ---- the interaction: is the margin specific to the SO geometry? -------
    print("\n" + "=" * 100)
    print("R-7  INTERACTION: is so(32)'s margin specific to an SO environment?")
    print("  Paired on seed: (so - best other) on the SO geometry, minus the "
          "same\n  quantity on each other geometry.  A large positive value "
          "means the\n  advantage does not transfer to that geometry.")
    print("=" * 100)
    inter = []
    if "so" in margins and margins["so"]:
        for g in geoms:
            if g == "so" or not margins.get(g):
                continue
            shared = sorted(set(margins["so"]) & set(margins[g]))
            if len(shared) < 3:
                print(f"  {g}: only {len(shared)} shared seed(s); skipped.")
                continue
            diff = np.array([margins["so"][s] - margins[g][s] for s in shared])
            inter.append(dict(label=f"margin(so-env) - margin({g}-env)", d=diff))
    if inter:
        report(inter, "R-7  INTERACTION (total AUC)",
               "  Holm-corrected over the geometries compared.")
    else:
        print("  not enough shared seeds to test the interaction.")

    # ---- does the matching algebra win its own geometry? ------------------
    print("\n" + "=" * 100)
    print("R-7  DOES THE MATCHING ALGEBRA WIN ITS OWN GEOMETRY?")
    print("=" * 100)
    for g in geoms:
        if g not in ALGS:
            print(f"  {g:>5s}: no matching algebra in the arm set "
                  f"(winner: {winner.get(g)})")
            continue
        mark = "YES" if winner.get(g) == g else "no"
        print(f"  {g:>5s}: winner is {winner.get(g):>12s}   matching? {mark}")

    # ---- the identity row is a positive control, not a test cell ----------
    #
    # With M_env = I the baseline is already optimal on the geometry channel:
    # it acts with M = I, which is exactly the target.  So the baseline should
    # top this row, and a Lie arm topping it would mean the geometry reward is
    # measuring something other than agreement with M_env.  Scoring `id` as a
    # test cell would push the verdict toward "MIXED" for a reason that has
    # nothing to do with compactness, so it is excluded from the
    # generalisation verdict and reported here instead.
    print("\n" + "=" * 100)
    print("R-7  POSITIVE CONTROL: the identity geometry")
    print("=" * 100)
    if "id" not in geoms:
        print("  the id row was not run; the harness check below is unavailable.")
    else:
        w_id = winner.get("id")
        print(f"  winner on M_env = I: {w_id}")
        if w_id == BASELINE:
            print("  -> as predicted.  With no geometry to learn the "
                  "unstructured control, which\n     acts with M = I, is "
                  "already optimal.  The harness is measuring what it\n     "
                  "claims to measure.")
        else:
            print("  -> !! UNEXPECTED.  With M_env = I the baseline acts with "
                  "exactly the target\n     transformation and should top this "
                  "row.  A Lie arm winning here means the\n     geometry "
                  "reward is not simply rewarding agreement with M_env, or the\n"
                  "     baseline is disadvantaged for some other reason.  "
                  "Resolve this before\n     reading any other row: it is a "
                  "property of the harness, not of the algebras.")

    # ---- verdict ----------------------------------------------------------
    print("\n" + "=" * 100)
    print("VERDICT")
    print("=" * 100)
    if not geoms:
        print("  no usable rows."); return
    # `id` is a control (see above), so the generalisation claim is judged on
    # the geometries that actually contain a transformation to learn.
    geoms = [g for g in geoms if g != "id"]
    if not geoms:
        print("  only the identity control was run; nothing to generalise "
              "from."); return
    non_so = [g for g in geoms if g != "so"]
    so_wins = [g for g in geoms if winner.get(g) == "so"]
    print(f"  (judged on {', '.join(geoms)}; the id control is excluded)")
    if len(so_wins) == len(geoms) and len(geoms) >= 3:
        print("  -> so(32) is the best policy algebra under every "
              "non-degenerate environment\n     geometry tested, including "
              "geometries it cannot represent.  The claim is\n     about the "
              "optimiser, not about matching the environment.\n     This is a "
              "STRONGER claim than the paper currently makes.")
    elif "so" in so_wins and not any(g in so_wins for g in non_so):
        print("  -> so(32) wins ONLY where the environment is itself SO(32).\n"
              "     The result is 'match the environment's structure', not "
              "'compactness helps'.\n     The claim holds for environments "
              "with a planted\n     compact-group structure.")
    else:
        won = ", ".join(so_wins) if so_wins else "none"
        print(f"  -> MIXED: so(32) is best under [{won}] but not under "
              f"[{', '.join(g for g in geoms if g not in so_wins) or 'none'}].\n"
              "     The claim holds only for the geometries where so(32) "
              "wins.")
    print("\n  CAVEAT: each geometry uses ONE draw of M_env, so a column "
          "reflects that draw\n  as well as its family.  X-3 established "
          "draw-robustness for the SO family only.\n  Treat a single "
          "off-diagonal column as indicative, not conclusive.")


# --------------------------------------------------------------------------- #
def selftest() -> None:
    ok = True

    def check(n, c, dd=""):
        nonlocal ok
        ok &= bool(c)
        print(f"  {'PASS' if c else 'FAIL'}  {n}{'  ' + dd if dd else ''}")

    print("(1) each geometry really is the family it claims")
    k = 32
    import torch
    M = {}
    for g in GEOMS:
        M[g], dg = draw_m_env(g, k, 123)
        M[g] = M[g].numpy()
        if g == "so":
            check("so: orthogonal", dg["menv_orth"] < 1e-9, f"{dg['menv_orth']:.1e}")
            check("so: det = +1", abs(np.linalg.det(M[g]) - 1) < 1e-9)
            check("so: not symmetric (it is a rotation)",
                  dg["menv_sym"] > 1e-3, f"|M-Mt|={dg['menv_sym']:.2e}")
        if g == "sym":
            check("sym: symmetric", dg["menv_sym"] < 1e-9)
            check("sym: not orthogonal", dg["menv_orth"] > 1e-3,
                  f"|MtM-I|={dg['menv_orth']:.2e}")
            check("sym: positive definite (it is exp of a symmetric matrix)",
                  np.linalg.eigvalsh(M[g]).min() > 0)
        if g == "gl":
            check("gl: neither orthogonal nor symmetric",
                  dg["menv_orth"] > 1e-3 and dg["menv_sym"] > 1e-3)
        if g == "diag":
            check("diag: diagonal",
                  np.abs(M[g] - np.diag(np.diag(M[g]))).max() < 1e-12)
            check("diag: positive entries", np.diag(M[g]).min() > 0)
        if g == "id":
            check("id: identity", np.abs(M[g] - np.eye(k)).max() < 1e-12)
        check(f"{g}: well conditioned", dg["menv_cond"] < 1e8,
              f"cond={dg['menv_cond']:.2e}")

    check("the five geometries are mutually distinct",
          len({M[g].tobytes() for g in GEOMS}) == len(GEOMS))
    check("so reproduces main.py's construction (exp of a random skew matrix)",
          np.abs(M["so"] @ M["so"].T - np.eye(k)).max() < 1e-9)
    for g in GEOMS:
        check(f"{g}: same draw twice is identical (reproducible)",
              torch.equal(draw_m_env(g, k, 7)[0], draw_m_env(g, k, 7)[0]))
    check("different seeds give different M (except id, which is constant)",
          not torch.equal(draw_m_env("so", k, 1)[0], draw_m_env("so", k, 2)[0])
          and torch.equal(draw_m_env("id", k, 1)[0], draw_m_env("id", k, 2)[0]))
    try:
        draw_m_env("nope", k, 1); bad = False
    except ValueError:
        bad = True
    check("an unknown geometry is refused, not defaulted", bad)

    print("(2) analysis and the three verdicts")
    rng = np.random.default_rng(0)

    def frame(table, n=3, noise=0.12):
        """table: {geom: {arm: true mean AUC}}"""
        rows = []
        for g, t in table.items():
            for s in range(n):
                se = rng.normal(0, .10)
                for a in ARMS:
                    rows.append(dict(geom=g, arm=a, seed=s,
                                     auc=t[a] + se + rng.normal(0, noise),
                                     r_task=.16, r_geo=.80,
                                     rho=np.nan if a == BASELINE else 1.0,
                                     menv_orth=0.0, menv_sym=1.0,
                                     menv_cond=1.0, recon_err=1e-15,
                                     status="ok"))
        return pd.DataFrame(rows)

    def run(df):
        b = io.StringIO()
        with contextlib.redirect_stdout(b):
            analyse(df)
        return b.getvalue()

    so_top = {BASELINE: 6.5, "so": 8.7, "sl": 8.1, "gl": 8.0, "sym": 6.6}
    sym_top = {BASELINE: 6.5, "so": 7.0, "sl": 7.4, "gl": 7.5, "sym": 8.6}

    # (a) so wins everywhere -> the strong claim
    everywhere = run(frame({g: so_top for g in GEOMS}, n=5, noise=0.08))
    check("so best in every geometry -> 'STRONGER claim'",
          "STRONGER claim" in everywhere)

    # (b) The case this experiment exists for: so wins only on the SO geometry
    matched = {"so": so_top, "sym": sym_top, "gl": sym_top,
               "diag": sym_top, "id": sym_top}
    only_so = run(frame(matched, n=5, noise=0.08))
    check("so best only on the SO geometry -> scope warning",
          "wins ONLY where the environment is itself SO(32)" in only_so)
    check("...and the interaction section reports it",
          "INTERACTION" in only_so)

    # (c) mixed
    mixed = {"so": so_top, "sym": sym_top, "gl": so_top,
             "diag": sym_top, "id": so_top}
    check("mixed pattern -> 'MIXED' and no generalisation",
          "MIXED" in run(frame(mixed, n=5, noise=0.08)))

    # (d) the matching-algebra diagnostic
    check("matching algebra detected when sym wins the sym geometry",
          "sym: winner is          sym   matching? YES" in only_so)

    # (e) The identity row is a control, not a test cell.
    #     With M_env = I the baseline acts with exactly the target, so it
    #     should top that row.  Scoring `id` as a test cell would turn a
    #     correct, predicted outcome into a "MIXED" verdict and understate a
    #     result that generalises.  These three checks pin that down.
    id_row = {BASELINE: 8.9, "so": 8.7, "sl": 8.1, "gl": 8.0, "sym": 6.6}
    realistic = run(frame({"so": so_top, "sym": so_top, "gl": so_top,
                           "diag": so_top, "id": id_row}, n=5, noise=0.08))
    check("baseline winning the id row does not downgrade the verdict",
          "STRONGER claim" in realistic and "MIXED" not in realistic)
    check("...and it is reported as the predicted control outcome",
          "as predicted" in realistic)
    check("...and the verdict says the id row was excluded",
          "the id control is excluded" in realistic)
    lie_wins_id = run(frame({"so": so_top, "sym": so_top, "gl": so_top,
                             "diag": so_top, "id": so_top}, n=5, noise=0.08))
    check("a Lie arm winning the id row raises a harness warning",
          "UNEXPECTED" in lie_wins_id)
    check("an id-only run does not generalise",
          "nothing to generalise" in run(frame({"id": id_row}, n=4)))

    # (f) the cross-row warning must always be present
    check("the 'not comparable across rows' warning is always printed",
          "NOT comparable across ROWS" in everywhere)
    check("the single-draw caveat is always printed",
          "each geometry uses ONE draw" in everywhere)

    # (g) degenerate inputs degrade loudly
    for name, dfx in [
        ("empty", frame({g: so_top for g in GEOMS}).iloc[0:0]),
        ("one seed", frame({g: so_top for g in GEOMS}, n=1)),
        ("one geometry", frame({"so": so_top}, n=4)),
        ("all so failed", frame({g: so_top for g in GEOMS}, n=4)
         .assign(status=lambda x: np.where(x.arm == "so", "err", "ok"))),
    ]:
        try:
            run(dfx); good = True
        except Exception as exc:                             # noqa: BLE001
            good = False
            print(f"        {name}: {type(exc).__name__}: {exc}")
        check(f"degenerate input '{name}' does not crash", good)

    print("\n" + ("SELFTEST PASS" if ok else "SELFTEST FAIL"))
    if not ok:
        sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--analyse-only", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest(); return
    if a.analyse_only:
        if not os.path.exists(CSV):
            sys.exit(f"{CSV} not found; run without --analyse-only first.")
        check_meta(CSV, dict(geoms=list(GEOMS), arms=list(ARMS), w=W,
                             horizon=HORIZON))
        df = pd.read_csv(CSV)
    else:
        df = collect(a.seeds, a.overwrite)
    analyse(df)


if __name__ == "__main__":
    main()
