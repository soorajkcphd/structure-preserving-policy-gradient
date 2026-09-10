"""
B-1-LR: does the expm-vs-Cayley gap survive per-MAP learning-rate tuning?

    python run_b1_lr.py --selftest
    python run_b1_lr.py --seeds 10        # ~20 min on an RTX 5060
    python run_b1_lr.py --analyse-only

Requires _r1core.py, _armlib.py and run_b1.py (for `cayley`/`matrix_map`).
No edits to main.py.

Why
---
B-1 found expm beating a Cayley retraction by +0.366 AUC (Holm p = 3.0e-07,
n = 20) with both maps producing proper rotations (rho = 1, |M^T M - I| < 3e-6).
Taken at face value that says the matrix exponential does work the SO(32)
constraint alone does not.

But B-1 ran both maps at theta_lr = 0.03, and that rate was chosen for the
exponential map.  The two maps are not interchangeable at a fixed rate:

    exp(A)      = I + A + A^2/2  + A^3/6   + ...
    cayley(A)   = I + A + A^2/2  + A^3/4   + ...

They agree to second order and diverge after, so the same step in the algebra
produces a different displacement in the group, and the Jacobian d(map)/dA
differs correspondingly.  A learning rate tuned for one is not neutral between
them.  This is exactly the confound R-5 removed for the algebra comparison,
and it is still present in the map comparison.

So B-1 as it stands is the shared-rate comparison the paper was criticised for.
This script repeats it with each map given its own rate, selected
leave-one-seed-out so selection and evaluation are disjoint.

Two outcomes
  gap survives  -> within one group, the exponential map optimises better than
                   a rational retraction.  A claim about the parameterisation,
                   not just the constraint.
  gap closes    -> any retraction onto SO(32) will do; the effect is the
                   constraint.  That is the stronger, more general claim, and
                   it is what the compactness theory actually predicts.
  neither       -> under-powered; report the interval and claim nothing.

Design
------
  maps       so_expm, so_cayley      (identical in every other respect)
  theta_lr   3e-4, 3e-3, 0.01, 0.03, 0.1     (same grid as R-5)
  seeds      0 .. n-1
  baseline   once per seed, at no theta_lr, for reference only

Output   b1lr_cells.csv   arm, theta_lr, seed, auc, r_task, r_geo, rho,
                          orth_err, recon_err, status
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys

import numpy as np
import pandas as pd

from _armlib import (BASELINE, Sink, check_meta, clean, equivalent,
                     guard_outputs, paired, report, run_one, write_meta)
from run_b1 import A_CAYLEY, A_EXPM, cayley, matrix_map, orthogonality_of_M

CSV = "b1lr_cells.csv"
MAPS = (A_EXPM, A_CAYLEY)
LR_GRID = (3e-4, 3e-3, 0.01, 0.03, 0.1)
MIN_CELL_N = 3          # a cell needs this many seeds to be eligible as "best"
PAPER_LR = 0.03
W, HORIZON = 0.4, 20
COLS = ["arm", "theta_lr", "seed", "auc", "r_task", "r_geo", "rho",
        "orth_err", "recon_err", "status"]

# Equivalence margin, pre-registered here rather than chosen after seeing the
# result: 25% of the expm-vs-baseline effect, the same rule B-1 used.  If the
# tuned CI lies entirely inside +/- this, the maps are interchangeable.
MARGIN_FRAC = 0.25
B1_EXPM_VS_BASELINE = 2.2338      # from B-1, 20 seeds


def _ctx(arm):
    """The Cayley arm runs inside the guarded global patch; expm runs bare."""
    return matrix_map(cayley) if arm == A_CAYLEY else contextlib.nullcontext()


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

    sink, rows = Sink(CSV, COLS), []
    write_meta(CSV, dict(maps=list(MAPS), lr_grid=list(LR_GRID),
                         w=W, horizon=HORIZON, n_seeds=n_seeds))
    plan = [(BASELINE, np.nan)] + [(m, lr) for m in MAPS for lr in LR_GRID]
    total, done = len(plan) * n_seeds, 0
    for seed in range(n_seeds):
        for arm, lr in plan:
            # The patch must be live for both training and the orthogonality
            # readout: orthogonality_of_M calls LieAlgebraOps.matrix_exp, so
            # outside the context it would measure the Cayley theta under expm.
            with _ctx(arm):
                rec, _, policy = run_one(
                    env, BASELINE if arm == BASELINE else "so", seed,
                    sd, na, k, W, HORIZON,
                    theta_lr=None if arm == BASELINE else lr,
                    extra={"theta_lr": lr})
                rec["orth_err"] = (orthogonality_of_M(policy)
                                   if policy is not None else np.nan)
            rec["arm"] = arm
            sink.add([{c: rec.get(c, np.nan) for c in COLS}])
            rows.append(rec); done += 1
            if rec["status"] != "ok":
                print(f"  [{done:3d}/{total}] {arm:10s} lr={lr:<7} s{seed:2d}  "
                      f"FAILED: {rec['status']}", flush=True)
            else:
                print(f"  [{done:3d}/{total}] {arm:10s} lr={lr:<7} s{seed:2d}  "
                      f"AUC={rec['auc']:7.3f}  rho={rec['rho']:.4g}  "
                      f"|MtM-I|={rec['orth_err']:.2e}", flush=True)
    df = pd.DataFrame(rows).reindex(columns=COLS)
    print(f"\nWrote {CSV} ({sink.n} rows)")
    return df


def _best_rates(d: pd.DataFrame) -> dict:
    """Whole-sample argmax per map, with a minimum cell size so one lucky seed
    on a partial CSV cannot win the argmax."""
    best = {}
    for m in MAPS:
        cell = {lr: d[(d.arm == m) & (d.theta_lr == lr)]["auc"].astype(float).dropna()
                for lr in LR_GRID}
        means = {lr: v.mean() for lr, v in cell.items()
                 if len(v) >= MIN_CELL_N and np.isfinite(v.mean())}
        best[m] = max(means, key=means.get) if means else np.nan
    return best


def analyse(df: pd.DataFrame) -> None:
    e = pd.to_numeric(df.get("recon_err"), errors="coerce") if "recon_err" in df else None
    if e is not None and np.isfinite(e).any():
        print(f"\nchannel-reconstruction error: max = "
              f"{e[np.isfinite(e)].max():.3e}")
    bad = df[df.status != "ok"]
    if len(bad):
        print(f"\n{len(bad)} failed run(s):")
        for _, r in bad.iterrows():
            print(f"    {r.arm} lr={r.theta_lr} seed={r.seed}: {r.status}")
    d = df[df.status == "ok"]

    # ---- validity gate: a row is only interpretable if M really is a rotation
    # `orth_err` may be absent entirely (an older CSV), in which case the gate
    # cannot run -- say so rather than skip it, because a missing
    # gate looks exactly like a passed one.
    if "orth_err" not in d.columns:
        print("\n  !! no 'orth_err' column: cannot verify that either map "
              "produced a rotation.\n     Re-collect with this version before "
              "citing anything below.")
        oel = pd.Series(dtype=float)
    else:
        oel = pd.to_numeric(d.loc[d.arm.isin(MAPS), "orth_err"], errors="coerce")
    if len(oel) and np.isfinite(oel).any() and np.nanmax(oel) > 1e-3:
        n_bad = int((oel > 1e-3).sum())
        print(f"\n  !! {n_bad} run(s) with |M^T M - I| > 1e-3: the map did NOT "
              f"produce a rotation.\n     Nothing in those rows is "
              f"interpretable.  Investigate before reading on.")

    print("\n" + "=" * 96)
    print("B-1-LR  AUC BY MAP x THETA LEARNING RATE  (mean +/- sd over seeds)")
    print("=" * 96)
    print(f"  {'map':>10s}" + "".join(f"{lr:>16.4g}" for lr in LR_GRID)
          + f"{'best lr':>10s}")
    best = _best_rates(d)
    for m in MAPS:
        cells = []
        for lr in LR_GRID:
            v = d[(d.arm == m) & (d.theta_lr == lr)]["auc"].astype(float).dropna()
            cells.append(f"{v.mean():8.3f}+/-{v.std(ddof=1):5.3f}"
                         if len(v) > 1 else f"{'n/a':>16s}")
        print(f"  {m:>10s}" + "".join(f"{c:>16s}" for c in cells)
              + f"{best[m]:>10.4g}")
    b = d[d.arm == BASELINE]["auc"].astype(float).dropna()
    if len(b):
        print(f"  {'baseline':>10s}  mean AUC {b.mean():.3f} +/- "
              f"{b.std(ddof=1):.3f} (no theta; reference only)")
    for m in MAPS:
        if "orth_err" not in d.columns:
            break
        o = pd.to_numeric(d[d.arm == m]["orth_err"], errors="coerce").dropna()
        r = pd.to_numeric(d[d.arm == m]["rho"], errors="coerce").dropna()
        if len(o) and len(r):
            print(f"  {m:>10s}: max|M^T M - I| = {o.max():.2e}, "
                  f"median rho = {r.median():.6g} over all rates")

    # ---- Q1: leave-one-seed-out tuning ------------------------------------
    #
    # Same argument as R-5: argmax over a 5-point grid of noisy cell means is
    # upward-biased, and the bias scales with each map's grid noise, so it is
    # not symmetric between the two arms.  Selecting on the other seeds makes
    # every evaluated point out-of-sample with respect to its own selection.
    seeds = sorted(d[d.arm.isin(MAPS)]["seed"].unique())
    loo_rows, chosen = [], {m: [] for m in MAPS}
    for s_ in seeds:
        rest = d[(d.seed != s_) & (d.arm.isin(MAPS))]
        pick, ok = {}, True
        for m in MAPS:
            cm = {lr: rest[(rest.arm == m) & (rest.theta_lr == lr)]["auc"]
                  .astype(float).dropna() for lr in LR_GRID}
            cm = {lr: v.mean() for lr, v in cm.items() if len(v) >= MIN_CELL_N}
            if not cm:
                ok = False; break
            pick[m] = max(cm, key=cm.get)
        if not ok:
            continue
        row, picked = {}, {}
        for m in MAPS:
            v = d[(d.arm == m) & (d.theta_lr == pick[m]) & (d.seed == s_)]["auc"]
            v = v.astype(float).dropna()
            if len(v) != 1:
                ok = False; break
            row[m] = float(v.iloc[0]); picked[m] = pick[m]
        if ok:
            # append to `chosen` only once the whole row survived, so the
            # printed rate tallies match the seeds actually analysed
            for m in MAPS:
                chosen[m].append(picked[m])
            loo_rows.append(dict(seed=s_, **row))

    print("\n" + "=" * 96)
    print("B-1-LR  Q1: EACH MAP AT ITS OWN BEST RATE  (leave-one-seed-out)")
    print("=" * 96)
    q1 = []
    if len(loo_rows) < 3:
        print("  not enough complete seeds for out-of-sample tuning.")
    else:
        L = pd.DataFrame(loo_rows)
        print(f"  n = {len(L)} seeds, each evaluated at the rate chosen from "
              f"the other {len(L) - 1}.")
        for m in MAPS:
            cnt = pd.Series(chosen[m]).value_counts().sort_index()
            print(f"    {m:>10s}: mean {L[m].mean():7.3f} +/- "
                  f"{L[m].std(ddof=1):5.3f}   rate chosen: "
                  + ", ".join(f"{lr:g}x{n}" for lr, n in cnt.items()))
        q1 = report([dict(label=f"{A_EXPM} - {A_CAYLEY}  (LOO-tuned)",
                          d=(L[A_EXPM] - L[A_CAYLEY]).to_numpy())],
                    "B-1-LR  TUNED MAP COMPARISON (total AUC), out-of-sample",
                    "  Selection and evaluation use disjoint seeds, so this "
                    "p-value is valid.\n  This is the number that decides "
                    "whether the map matters.")

    # ---- the in-sample version, as a contrast only ------------------------
    # If a map has no eligible cell (fewer than MIN_CELL_N seeds anywhere) its
    # best rate is NaN and no row would match, producing an empty
    # comparison that prints like a real one.  Skip it instead.
    if any(not np.isfinite(best[m]) for m in MAPS):
        print("\n  (in-sample tuned table skipped: at least one map has no "
              f"cell with {MIN_CELL_N}+ seeds)")
    else:
        tuned = pd.concat([d[(d.arm == m) & (d.theta_lr == best[m])] for m in MAPS])
        tuned = clean(tuned, "auc", "seed", MAPS)
        report([dict(label=f"{A_EXPM}@{best[A_EXPM]:g} - {A_CAYLEY}@{best[A_CAYLEY]:g}",
                     d=paired(tuned, A_EXPM, A_CAYLEY, "auc", "seed")[0])],
               "B-1-LR  IN-SAMPLE tuned comparison -- BIASED, for contrast only",
               "  The rate was chosen on the same seeds used for the test.  "
               "Shown only so the\n  size of the selection effect is visible "
               "against the out-of-sample table.")

    # ---- Q2: was the shared rate tilted toward expm? ----------------------
    print("\n" + "=" * 96)
    print(f"B-1-LR  Q2: WAS B-1's SHARED RATE ({PAPER_LR}) TILTED TOWARD expm?")
    print("=" * 96)
    for m in MAPS:
        v = {lr: d[(d.arm == m) & (d.theta_lr == lr)]["auc"].astype(float).mean()
             for lr in LR_GRID}
        v = {lr: x for lr, x in v.items() if np.isfinite(x)}
        if not v:
            continue
        at_paper = v.get(PAPER_LR, float("nan"))
        flag = "  <-- the shared rate IS its best" if best[m] == PAPER_LR else ""
        print(f"  {m:>10s}: best {max(v.values()):7.3f} at lr={best[m]:g}; "
              f"at lr={PAPER_LR:g} {at_paper:7.3f}  "
              f"(loses {max(v.values()) - at_paper:.3f}){flag}")
    print("\n  Read: if 0.03 is expm's best but not Cayley's, B-1's +0.366 was "
          "measured at a\n  setting that favours expm, and only the "
          "out-of-sample tuned table above is\n  admissible.")
    print("\n  CAVEAT, stated rather than hidden: this grid varies theta_lr "
          "alone, and the\n  remaining hyperparameters were calibrated for "
          "expm at 0.03.  'Each map at its\n  own best rate' means 'best rate "
          "holding those constants fixed'.")

    # ---- verdict ----------------------------------------------------------
    print("\n" + "=" * 96)
    print("VERDICT")
    print("=" * 96)
    if not q1:
        print("  Not enough paired seeds to decide."); return
    o = q1[0]
    margin = MARGIN_FRAC * B1_EXPM_VS_BASELINE
    if o.get("sig") and o.get("diff", 0) > 0:
        print(f"  -> The gap SURVIVES per-map tuning ({o['diff']:+.4f}, "
              f"95% CI [{o['ci'][0]:+.4f}, {o['ci'][1]:+.4f}],\n     "
              f"p_holm={o['p_holm']:.2e}).  Within SO(32), the exponential map "
              "optimises better\n     than a rational retraction at each map's "
              "own best rate.  The map matters, even though both\n     maps land "
              "in the same group and reach it to machine precision.")
    elif o.get("sig") and o.get("diff", 0) < 0:
        print(f"  -> Cayley BEATS expm once each map is tuned "
              f"({o['diff']:+.4f}, p_holm={o['p_holm']:.2e}).\n     B-1's "
              "ordering was a learning-rate artefact in the opposite "
              "direction.\n     The released scheme is not the best way to "
              "realise the constraint.")
    elif equivalent(o, margin):
        print(f"  -> EQUIVALENT within +/-{margin:.4f} AUC ({MARGIN_FRAC:.0%} "
              f"of B-1's expm-vs-baseline\n     effect): the whole 95% CI "
              f"[{o['ci'][0]:+.4f}, {o['ci'][1]:+.4f}] lies inside the "
              f"margin.\n     B-1's +0.366 was a shared-rate artefact.  The "
              "effect is a property of\n     constraining to SO(32), not of "
              "the matrix exponential -- the stronger and\n     more general "
              "claim, and the one the compactness theory predicts.")
    else:
        from sppg_defense.stats.tests import n_for_power_paired
        try:
            need = n_for_power_paired(abs(o["dz"]), 0.8)
        except Exception:                                    # noqa: BLE001
            need = None
        print(f"  -> UNDER-POWERED, not equivalent.  The tuned difference "
              f"({o['diff']:+.4f}) is not\n     significant, but the 95% CI "
              f"[{o['ci'][0]:+.4f}, {o['ci'][1]:+.4f}] is not inside the\n     "
              f"+/-{margin:.4f} equivalence margin either.  Claim neither "
              "direction."
              + (f"  About {need} seeds\n     would be needed for 80% power at "
                 f"the observed effect." if need else ""))


# --------------------------------------------------------------------------- #
def selftest() -> None:
    import torch
    ok = True

    def check(n, c, dd=""):
        nonlocal ok
        ok &= bool(c)
        print(f"  {'PASS' if c else 'FAIL'}  {n}{'  ' + dd if dd else ''}")

    print("(1) the two maps really are different, and both land in SO(32)")
    k = 32
    g = torch.Generator().manual_seed(11)
    X = torch.randn(k, k, generator=g, dtype=torch.float64)
    A = 0.5 * (X - X.T)
    I = torch.eye(k, dtype=torch.float64)
    C, E = cayley(A), torch.matrix_exp(A)
    check("cayley orthogonal", (C.T @ C - I).abs().max().item() < 1e-10)
    check("expm orthogonal", (E.T @ E - I).abs().max().item() < 1e-10)
    check("the two differ at this scale (a tuned comparison is meaningful)",
          (C - E).abs().max().item() > 1e-2,
          f"max|C-E|={(C - E).abs().max().item():.2e}")
    # the third-order claim made in the docstring, checked rather than asserted
    s = 1e-2 * A
    d2 = (cayley(s) - torch.matrix_exp(s)).abs().max().item()
    check("agree to 2nd order: |C-E| ~ O(|A|^3) for small A",
          d2 < 1e-5, f"max|C-E|={d2:.2e} at |A|~1e-2")

    print("(2) the global patch is live during training and the readout")
    try:
        from main import LieAlgebraOps
    except Exception as exc:                                 # noqa: BLE001
        print(f"  SKIP  main.py not importable here: {exc}")
    else:
        orig = LieAlgebraOps.__dict__["matrix_exp"]
        with _ctx(A_CAYLEY):
            check("cayley arm -> map is Cayley",
                  torch.allclose(LieAlgebraOps.matrix_exp(A), C))
        check("restored after the cayley arm",
              LieAlgebraOps.__dict__["matrix_exp"] is orig)
        with _ctx(A_EXPM):
            check("expm arm -> map is untouched expm",
                  torch.allclose(LieAlgebraOps.matrix_exp(A), E))
        with _ctx(BASELINE):
            check("baseline arm -> map is untouched expm",
                  torch.allclose(LieAlgebraOps.matrix_exp(A), E))
        try:
            with _ctx(A_CAYLEY):
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        check("restored even when the body raises",
              LieAlgebraOps.__dict__["matrix_exp"] is orig)

    print("(3) analysis, selection logic and the three verdicts")
    rng = np.random.default_rng(0)

    def frame(expm_by_lr, cay_by_lr, n=10, noise=0.15):
        """expm_by_lr / cay_by_lr: dict lr -> true mean AUC."""
        rows = []
        for s in range(n):
            se = rng.normal(0, .10)          # shared seed effect -> pairing
            rows.append(dict(arm=BASELINE, theta_lr=np.nan, seed=s,
                             auc=6.50 + se + rng.normal(0, noise),
                             r_task=.18, r_geo=.55, rho=np.nan,
                             orth_err=np.nan, recon_err=1e-15, status="ok"))
            for m, tbl in ((A_EXPM, expm_by_lr), (A_CAYLEY, cay_by_lr)):
                for lr in LR_GRID:
                    rows.append(dict(arm=m, theta_lr=lr, seed=s,
                                     auc=tbl[lr] + se + rng.normal(0, noise),
                                     r_task=.15, r_geo=.86, rho=1.0,
                                     orth_err=1e-7, recon_err=1e-15,
                                     status="ok"))
        return pd.DataFrame(rows)

    def run(df):
        b = io.StringIO()
        with contextlib.redirect_stdout(b):
            analyse(df)
        return b.getvalue()

    flat = {3e-4: 6.9, 3e-3: 8.0, 0.01: 8.4, 0.03: 8.66, 0.1: 8.6}

    # (a) the gap is real at every rate -> it must survive tuning
    real = run(frame(flat, {lr: v - 0.40 for lr, v in flat.items()}))
    check("a rate-independent gap -> 'gap SURVIVES per-map tuning'",
          "gap SURVIVES per-map tuning" in real)

    # (b) The case this script exists for: equal peaks at different rates.
    #     At the shared 0.03 expm leads by 0.66; at each map's own best they
    #     are equal.  A shared-rate analysis would call this a real effect.
    peak_shift = {3e-4: 6.9, 3e-3: 8.2, 0.01: 8.66, 0.03: 8.00, 0.1: 7.0}
    art = run(frame(flat, peak_shift, n=24, noise=0.06))
    check("equal peaks at different rates -> 'EQUIVALENT within'",
          "EQUIVALENT within" in art)
    check("...and Q2 flags that 0.03 is not Cayley's best",
          "the shared rate IS its best" in art.split("Q2:")[1].split("VERDICT")[0]
          and art.split("Q2:")[1].count("the shared rate IS its best") == 1)

    # (c) Cayley better once tuned
    rev = run(frame({lr: v - 0.40 for lr, v in flat.items()}, flat))
    check("Cayley ahead after tuning -> 'Cayley BEATS expm'",
          "Cayley BEATS expm" in rev)

    # (d) a small true gap at high noise must not be called equivalent
    und = run(frame(flat, {lr: v - 0.30 for lr, v in flat.items()},
                    n=6, noise=1.4))
    check("a small gap at high noise -> 'UNDER-POWERED, not equivalent'",
          "UNDER-POWERED, not equivalent" in und)

    # (e) the selection machinery itself
    df_e = frame(flat, peak_shift, n=24, noise=0.06)
    d_ok = df_e[df_e.status == "ok"]
    b_ = _best_rates(d_ok)
    check("whole-sample argmax finds each map's own peak",
          b_[A_EXPM] == 0.03 and b_[A_CAYLEY] == 0.01,
          f"expm={b_[A_EXPM]:g}, cayley={b_[A_CAYLEY]:g}")
    check("in-sample and out-of-sample tables are both printed, and labelled",
          "BIASED, for contrast only" in art and "out-of-sample" in art)

    # (f) the validity gate fires when the map stops producing a rotation
    dfb = frame(flat, flat, n=6)
    dfb.loc[(dfb.arm == A_CAYLEY) & (dfb.seed == 0), "orth_err"] = 0.5
    check("a non-orthogonal M raises the validity warning",
          "did NOT produce a rotation" in run(dfb))
    check("...and a clean run does not",
          "did NOT produce a rotation" not in real)

    # (g) an incomplete cell must not win the argmax
    dfc = frame(flat, flat, n=10)
    drop = (dfc.arm == A_CAYLEY) & (dfc.theta_lr == 0.1) & (dfc.seed > 0)
    dfc = dfc[~drop].copy()
    dfc.loc[(dfc.arm == A_CAYLEY) & (dfc.theta_lr == 0.1), "auc"] = 99.0
    b2 = _best_rates(dfc[dfc.status == "ok"])
    check("a 1-seed cell with a huge value cannot become the best rate",
          b2[A_CAYLEY] != 0.1, f"cayley best={b2[A_CAYLEY]:g}")

    # (h) failed rows are excluded, not averaged in
    dff = frame(flat, flat, n=6)
    dff.loc[(dff.arm == A_EXPM) & (dff.seed == 1), "status"] = "RuntimeError: x"
    check("failed runs are listed", "failed run(s)" in run(dff))

    print("\n" + ("SELFTEST PASS" if ok else "SELFTEST FAIL"))
    if not ok:
        sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--analyse-only", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest(); return
    if a.analyse_only:
        if not os.path.exists(CSV):
            sys.exit(f"{CSV} not found; run without --analyse-only first.")
        check_meta(CSV, dict(maps=list(MAPS), lr_grid=list(LR_GRID),
                             w=W, horizon=HORIZON))
        df = pd.read_csv(CSV)
    else:
        df = collect(a.seeds, a.overwrite)
    analyse(df)


if __name__ == "__main__":
    main()
