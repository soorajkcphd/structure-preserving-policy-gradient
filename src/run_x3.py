"""
X-3: does the algebra ordering hold across environment draws?

    python run_x3.py --selftest
    python run_x3.py --draws 20        # ~20 min on an RTX 5060
    python run_x3.py --analyse-only

Requires _r1core.py and _armlib.py.  No edits to main.py.

Why
---
Every Task-1 result in the paper -- R-1, R-1b, R-4, R-6, S-7 -- rests on one
planted rotation: M_env, built from numpy RandomState(42) at env construction.
The surviving claim after S-7 is an ordering of parameterisations, and an
ordering established on a single draw of the thing being matched is one draw
away from being an accident.

This redraws M_env 20 times and re-runs the comparison.  The unit of
replication is the environment draw, not the training seed: that is the
quantity the claim needs to generalise over.  One training seed per draw keeps
the design clean (each draw contributes one independent observation) and the
cost sane.

Mutating M_env is safe
----------------------
Nothing in MultiStepTextAlignmentEnv is precomputed from M_env: prompt_embs,
action_dirs, base_deltas and state_embs are all built before it and never
reference it.  It is read in exactly two places -- _geometry_reward (the
reward) and the trainer's auxiliary loss -- and unlike S-7 we change both here,
because a redrawn M_env is a different environment.  Each draw is
verified to be in SO(32) before use.

Output   x3_cells.csv   draw, arm, seed, auc, r_task, r_geo, rho, m_env_seed, status
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

from _armlib import (BASELINE, Sink, check_meta, clean, guard_outputs,
                     paired, report, run_one, write_meta)

CSV = "x3_cells.csv"
ARMS = (BASELINE, "so", "sl", "gl", "sym")
MENV_SEED0 = 900_000
W, HORIZON = 0.4, 20
COLS = ["draw", "m_env_seed", "arm", "seed", "auc", "r_task", "r_geo", "rho",
        "recon_err", "status"]
SEEDS_PER_DRAW = 3   # averaged within a draw, so seed luck does not ride along


def draw_m_env(k: int, seed: int, device, dtype):
    """A fresh planted rotation, built exactly as main.py builds M_env:
    exp of a random skew-symmetric matrix, hence in SO(k) by construction."""
    import torch
    from scipy.linalg import expm

    rng = np.random.RandomState(int(seed))
    S = rng.randn(k, k) * 0.5
    S = 0.5 * (S - S.T)
    M = torch.tensor(expm(S), dtype=torch.float64)
    err = (M.T @ M - torch.eye(k, dtype=torch.float64)).abs().max().item()
    det = float(torch.det(M))
    if not (err < 1e-9 and abs(det - 1.0) < 1e-9):
        raise RuntimeError(f"drawn M_env not in SO({k}): "
                           f"max|MtM-I|={err:.2e}, det={det:.6f}")
    return M.to(device=device, dtype=dtype)


def collect(n_draws: int, n_seeds: int, overwrite: bool) -> pd.DataFrame:
    guard_outputs([CSV], overwrite)
    import torch
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
    write_meta(CSV, dict(arms=list(ARMS), w=W, horizon=HORIZON,
                         n_draws=n_draws, seeds_per_draw=n_seeds,
                         menv_seed0=MENV_SEED0))
    total = n_draws * len(ARMS) * n_seeds
    done = 0
    try:
        for draw in range(n_draws):
            ms = MENV_SEED0 + draw
            env.M_env = draw_m_env(k, ms, orig.device, orig.dtype)
            for sd_i in range(n_seeds):
                for arm in ARMS:
                    rec, _, _ = run_one(env, arm, sd_i, sd, na, k, W, HORIZON,
                                        extra={"draw": draw, "m_env_seed": ms})
                    sink.add([{c: rec.get(c, np.nan) for c in COLS}])
                    rows.append(rec); done += 1
                    msg = (f"FAILED: {rec['status']}" if rec["status"] != "ok"
                           else f"AUC={rec['auc']:7.3f}  "
                                f"r_geo={rec['r_geo']:.4f}  rho={rec['rho']:.4g}")
                    print(f"  [{done:3d}/{total}] draw {draw:2d} s{sd_i} "
                          f"{arm:12s} {msg}", flush=True)
    finally:
        env.M_env = orig          # leave the env exactly as we found it
    df = pd.DataFrame(rows).reindex(columns=COLS)
    print(f"\nWrote {CSV} ({sink.n} rows)")
    return df


def analyse(df: pd.DataFrame) -> None:
    e = pd.to_numeric(df.get("recon_err"), errors="coerce") if "recon_err" in df else None
    if e is not None and np.isfinite(e).any():
        print(f"\nchannel-reconstruction error: max = "
              f"{e[np.isfinite(e)].max():.3e}")
    bad = df[df.status != "ok"]
    if len(bad):
        print(f"\n{len(bad)} failed run(s):")
        for _, r in bad.iterrows():
            print(f"    draw {r.draw} {r.arm}: {r.status}")

    # Average the training seeds within each draw first.  The replication unit
    # is the environment draw; averaging inside the block stops one lucky
    # initialisation from riding along into every draw, which is what a single
    # fixed seed would do.
    ok = df[df.status == "ok"]
    per_seed = ok.groupby(["draw", "arm", "seed"], as_index=False).first()
    d = (per_seed.groupby(["draw", "arm"], as_index=False)
         .agg(auc=("auc", "mean"), r_task=("r_task", "mean"),
              r_geo=("r_geo", "mean"), rho=("rho", "mean"),
              n_seed=("seed", "nunique")))
    d["status"] = "ok"
    ns = sorted(d["n_seed"].unique())
    print(f"\naveraging {ns} training seed(s) within each draw before pairing; "
          f"the draw is the replication unit")
    if max(ns) < 2:
        print("!! only one training seed per draw: the result is conditional on "
              "that seed,\n   and the verdict below is restricted accordingly.")
    d = clean(d, "auc", "draw", ARMS)
    n_draw = d["draw"].nunique()
    single_seed = max(ns) < 2
    print("\n" + "=" * 96)
    print(f"X-3  PER-ARM ACROSS {n_draw} ENVIRONMENT DRAWS  (mean +/- sd)")
    print("=" * 96)
    print(f"  {'arm':>12s} {'n':>3s} {'AUC':>18s} {'r_geo':>16s} {'rho':>10s}")
    for a in ARMS:
        v = d[d.arm == a]
        if not len(v):
            print(f"  {a:>12s}   (none)"); continue
        au = v["auc"].astype(float); rg = v["r_geo"].astype(float)
        rho = v["rho"].astype(float).dropna()
        print(f"  {a:>12s} {len(v):>3d} {au.mean():9.3f}+/-{au.std(ddof=1):6.3f} "
              f"{rg.mean():8.4f}+/-{rg.std(ddof=1):6.4f} "
              f"{(f'{rho.mean():10.4g}' if len(rho) else f'{chr(45):>10s}')}")

    rows = [dict(label=f"so - {a}", d=paired(d, "so", a, "auc", "draw")[0])
            for a in ARMS if a != "so"]
    out = report(rows, "X-3  TOTAL AUC, paired BY ENVIRONMENT DRAW",
                 "  The blocking variable is the draw, so these test whether "
                 "the ordering\n  generalises over planted rotations, not over "
                 "training seeds.")

    print("\n" + "=" * 96)
    print("PER-DRAW ORDERING")
    print("=" * 96)
    algs = [a for a in ARMS if a != BASELINE]
    wins = 0
    for dr in sorted(d["draw"].unique()):
        v = d[d.draw == dr].set_index("arm")["auc"].astype(float)
        order = v[algs].sort_values(ascending=False)
        top = order.index[0]
        wins += (top == "so")
        print(f"  draw {dr:>2d}: " + " > ".join(f"{a}({v[a]:.2f})" for a in order.index)
              + ("" if top == "so" else "   <-- so(32) is not top here"))
    print(f"\n  so(32) is the top algebra in {wins}/{len(d['draw'].unique())} draws.")

    print("\n" + "=" * 96)
    print("VERDICT")
    print("=" * 96)
    beat = [o for o in out if o["label"] != "so - baseline_ppo"]
    lost = [o for o in beat if o.get("sig") and o.get("diff", 0) < 0]
    won = [o for o in beat if o.get("sig") and o.get("diff", 0) > 0]
    if lost:
        print("  -> On some draws another algebra significantly beats so(32).  "
              "The ordering\n     does not generalise over planted rotations "
              "and is not a property\n     of the algebra.")
    elif len(won) == len(beat) and beat:
        tail = ("\n     NOTE: only one training seed per draw, so this holds "
                "at that seed;\n     it is not yet unconditional over "
                "initialisations." if single_seed else
                "\n     Training seeds are averaged within each draw, so the "
                "result is not\n     conditional on one initialisation.")
        print("  -> so(32) beats every other algebra across independently drawn "
              "environments.\n     The ordering is a property of the algebra, "
              "not of one planted rotation." + tail)
    else:
        print("  -> Not all comparisons reach significance across draws.  "
              "Draw-independence\n     is not established; see the intervals and the "
              "per-draw table.")


def selftest() -> None:
    import torch
    ok = True

    def check(n, c, dd=""):
        nonlocal ok
        ok &= bool(c)
        print(f"  {'PASS' if c else 'FAIL'}  {n}{'  ' + dd if dd else ''}")

    print("(1) drawn rotations really are in SO(32)")
    k = 32
    M = draw_m_env(k, MENV_SEED0, "cpu", torch.float64)
    check("orthogonal", (M.T @ M - torch.eye(k, dtype=torch.float64))
          .abs().max().item() < 1e-9)
    check("det = +1", abs(float(torch.det(M)) - 1.0) < 1e-9)
    g = torch.Generator().manual_seed(1)
    V = torch.randn(128, k, generator=g, dtype=torch.float64)
    check("isometry on every probe",
          ((V @ M.T).norm(dim=-1) - V.norm(dim=-1)).abs().max().item() < 1e-10)
    M2 = draw_m_env(k, MENV_SEED0 + 1, "cpu", torch.float64)
    check("different seed -> different rotation", not torch.allclose(M, M2))
    cos = float((torch.sum(M * M2) / (M.norm() * M2.norm())).item())
    check("independent draws are near-orthogonal", abs(cos) < 0.3, f"cos={cos:+.4f}")
    # not a hard-coded True: rebuild main.py's own M_env from seed 42 with this
    # function and require an exact match.
    from scipy.linalg import expm as _expm
    _r = np.random.RandomState(42); _S = _r.randn(k, k) * 0.5
    _S = 0.5 * (_S - _S.T)
    ref = torch.tensor(_expm(_S), dtype=torch.float64)
    mine = draw_m_env(k, 42, "cpu", torch.float64)
    check("reproduces main.py's own M_env exactly from seed 42",
          torch.allclose(ref, mine, atol=1e-12),
          f"max diff={float((ref - mine).abs().max()):.2e}")
    M32 = draw_m_env(k, MENV_SEED0, "cpu", torch.float32)
    check("float32 cast stays orthogonal to ~1e-6",
          (M32.T @ M32 - torch.eye(k)).abs().max().item() < 1e-4)

    print("(2) analysis blocks on the draw, not the seed")
    rng = np.random.default_rng(0)
    eff = {BASELINE: 6.50, "so": 8.73, "sl": 8.09, "gl": 8.05, "sym": 6.58}
    rows = []
    for dr in range(12):
        de = rng.normal(0, .25)                     # draw-level effect
        for a, base in eff.items():
            rows.append(dict(draw=dr, m_env_seed=MENV_SEED0 + dr, arm=a, seed=0,
                             auc=base + de + rng.normal(0, .12), r_task=.16,
                             r_geo=.86, rho=1.0, recon_err=1e-15, status="ok"))
    df = pd.DataFrame(rows)
    import contextlib, io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        analyse(df)
    out = buf.getvalue()
    check("pairs by draw", "paired BY ENVIRONMENT DRAW" in out)
    check("per-draw ordering reported", "PER-DRAW ORDERING" in out)
    check("verdict fires 'property of the algebra' when so wins everywhere",
          "not of one planted rotation" in out)

    df2 = df.copy()
    df2.loc[df2.arm == "sl", "auc"] += 1.2
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        analyse(df2)
    check("verdict fires 'does not generalise' when another algebra wins",
          "does not generalise over planted rotations" in buf.getvalue())

    df3 = df[~((df.arm == "sym") & (df.draw == 3))]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        analyse(df3)
    check("an incomplete draw is dropped, keeping the design paired",
          "incomplete block" in buf.getvalue())

    print("\n" + ("SELFTEST PASS" if ok else "SELFTEST FAIL"))
    if not ok:
        sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=20)
    ap.add_argument("--seeds-per-draw", type=int, default=SEEDS_PER_DRAW,
                    dest="seeds_per_draw",
                    help="training seeds averaged within each draw (the draw is "
                         "the replicate); 1 makes the result seed-conditional")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--analyse-only", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest(); return
    if a.analyse_only:
        if not os.path.exists(CSV):
            sys.exit(f"{CSV} not found; run without --analyse-only first.")
        check_meta(CSV, dict(arms=list(ARMS), w=W, horizon=HORIZON,
                             menv_seed0=MENV_SEED0))
        df = pd.read_csv(CSV)
    else:
        df = collect(a.draws, a.seeds_per_draw, a.overwrite)
    analyse(df)


if __name__ == "__main__":
    main()
