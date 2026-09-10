"""
R-5: per-algebra learning-rate sweep.

    python run_r5.py --selftest
    python run_r5.py --seeds 10        # ~35 min on an RTX 5060
    python run_r5.py --analyse-only

Requires _r1core.py and _armlib.py.  No edits to main.py.

Motivation
----------
After S-7, the surviving Task-1 claim is the algebra comparison of R-4/R-6:
constraining to the compact subalgebra so(32) optimises better than the
strictly larger sl(32) and gl(32), which both contain so(32) and could
represent M_env exactly.

That claim rests on a single shared theta learning rate.  main.py uses
theta_lr = 0.03 for every algebra, chosen (per its own comment) to make
non-compact spectral growth visible within 60 iterations, and the manuscript's
Limitations section notes the comparison is therefore "entangled with
optimizer stability at this shared rate".

The obvious objection is: sl(32) and gl(32) lose because 0.03 is wrong
for them, not because they are non-compact.  This sweep answers it by giving
every algebra its own best rate and re-running the comparison there.

Design
------
  algebras   so, sl, gl, sym            (all keep M and the geometry reward)
  theta_lr   3e-4, 3e-3, 0.01, 0.03, 0.1   (spans the feature-net rate to 3x
                                            the paper's)
  seeds      0 .. n-1
  plus baseline PPO once per seed, for reference only

Two questions, and they are different:
  1. At each algebra's own best rate, does so(32) still win?   <- the claim
  2. Is 0.03 the best rate for so(32)?  If so(32) is uniquely favoured by the
     shared rate, the published comparison was tilted.

Output   r5_cells.csv   arm, theta_lr, seed, auc, r_task, r_geo, rho, status
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

from _armlib import (BASELINE, Sink, check_meta, clean, guard_outputs,
                     paired, report, run_one, write_meta)

CSV = "r5_cells.csv"
ALGS = ("so", "sl", "gl", "sym")
LR_GRID = (3e-4, 3e-3, 0.01, 0.03, 0.1)
MIN_CELL_N = 3   # a cell needs this many seeds to be eligible as "best"
PAPER_LR = 0.03
W, HORIZON = 0.4, 20
COLS = ["arm", "theta_lr", "seed", "auc", "r_task", "r_geo", "rho",
        "recon_err", "status"]


def collect(n_seeds: int, overwrite: bool) -> pd.DataFrame:
    guard_outputs([CSV], overwrite)
    from main import (GPT2EmbeddingProvider, MultiStepTextAlignmentEnv,
                      _seed_all)
    from _r1core import instrument

    emb = GPT2EmbeddingProvider(model_name="gpt2-medium", max_length=64)
    env = MultiStepTextAlignmentEnv(emb, n_prompts=16, n_actions=16,
                                    horizon=HORIZON, reward_noise=0.2,
                                    geo_weight=W, sparse_prob=0.7)
    instrument(env)
    sd, na, k = emb.hidden_dim, env.n_actions, int(env.k_transform)

    sink, rows = Sink(CSV, COLS), []
    write_meta(CSV, dict(lr_grid=list(LR_GRID), algs=list(ALGS),
                         w=W, horizon=HORIZON, n_seeds=n_seeds))
    plan = [(BASELINE, np.nan)] + [(a, lr) for a in ALGS for lr in LR_GRID]
    total = len(plan) * n_seeds
    done = 0
    for seed in range(n_seeds):
        for arm, lr in plan:
            rec, _, _ = run_one(env, arm, seed, sd, na, k, W, HORIZON,
                                theta_lr=None if arm == BASELINE else lr,
                                extra={"theta_lr": lr})
            sink.add([{c: rec.get(c, np.nan) for c in COLS}])
            rows.append(rec); done += 1
            if rec["status"] != "ok":
                print(f"  [{done:3d}/{total}] {arm:5s} lr={lr:<7} s{seed:2d}  "
                      f"FAILED: {rec['status']}", flush=True)
            else:
                print(f"  [{done:3d}/{total}] {arm:5s} lr={lr:<7} s{seed:2d}  "
                      f"AUC={rec['auc']:7.3f}  rho={rec['rho']:.4g}", flush=True)
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
            print(f"    {r.arm} lr={r.theta_lr} seed={r.seed}: {r.status}")
    d = df[df.status == "ok"]

    print("\n" + "=" * 96)
    print("R-5  AUC BY ALGEBRA x THETA LEARNING RATE  (mean +/- sd over seeds)")
    print("=" * 96)
    print(f"  {'algebra':>8s}" + "".join(f"{lr:>16.4g}" for lr in LR_GRID)
          + f"{'best lr':>10s}")
    best = {}
    for a in ALGS:
        cells = []
        for lr in LR_GRID:
            v = d[(d.arm == a) & (d.theta_lr == lr)]["auc"].astype(float).dropna()
            cells.append(f"{v.mean():8.3f}+/-{v.std(ddof=1):5.3f}"
                         if len(v) > 1 else f"{'n/a':>16s}")
        cell = {lr: d[(d.arm == a) & (d.theta_lr == lr)]["auc"]
                .astype(float).dropna() for lr in LR_GRID}
        # A cell with one lucky seed must not win the argmax on a partial CSV.
        means = {lr: v.mean() for lr, v in cell.items()
                 if len(v) >= MIN_CELL_N and np.isfinite(v.mean())}
        best[a] = max(means, key=means.get) if means else np.nan
        print(f"  {a:>8s}" + "".join(f"{c:>16s}" for c in cells)
              + f"{best[a]:>10.4g}")
    b = d[d.arm == BASELINE]["auc"].astype(float).dropna()
    if len(b):
        print(f"  {'baseline':>8s}  mean AUC {b.mean():.3f} +/- {b.std(ddof=1):.3f} "
              f"(no theta; shown for reference only)")

    # ---- Q1: leave-one-seed-out tuning, so selection and test are disjoint -
    #
    # Choosing the best rate on the same seeds you then test on is selection
    # inference: max over a 5-point grid of noisy cell means is upward-biased,
    # the bias scales with each arm's grid noise (so it is not symmetric across
    # arms), and the resulting p-values and CIs have no valid interpretation.
    # Leave-one-seed-out fixes it at no extra compute: for each seed, pick the
    # rate using the other seeds, then evaluate that seed.  Every evaluated
    # point is then out-of-sample with respect to its own selection.
    seeds = sorted(d[d.arm.isin(ALGS)]["seed"].unique())
    loo_rows, chosen = [], {a: [] for a in ALGS}
    for s_ in seeds:
        rest = d[(d.seed != s_) & (d.arm.isin(ALGS))]
        ok = True
        pick = {}
        for a in ALGS:
            m = {lr: rest[(rest.arm == a) & (rest.theta_lr == lr)]["auc"]
                 .astype(float).dropna() for lr in LR_GRID}
            m = {lr: v.mean() for lr, v in m.items() if len(v) >= MIN_CELL_N}
            if not m:
                ok = False; break
            pick[a] = max(m, key=m.get)
        if not ok:
            continue
        row = {}
        for a in ALGS:
            v = d[(d.arm == a) & (d.theta_lr == pick[a]) & (d.seed == s_)]["auc"]
            v = v.astype(float).dropna()
            if len(v) != 1:
                ok = False; break
            row[a] = float(v.iloc[0])
            chosen[a].append(pick[a])
        if ok:
            loo_rows.append(dict(seed=s_, **row))

    print("\n" + "=" * 96)
    print("R-5  Q1: EACH ALGEBRA AT ITS OWN BEST RATE  "
          "(leave-one-seed-out selection)")
    print("=" * 96)
    if len(loo_rows) < 3:
        print("  not enough complete seeds for out-of-sample tuning.")
        q1 = []
    else:
        L = pd.DataFrame(loo_rows)
        print(f"  n = {len(L)} seeds, each evaluated at the rate chosen from "
              f"the other {len(L) - 1}.")
        for a in ALGS:
            cnt = pd.Series(chosen[a]).value_counts().sort_index()
            print(f"    {a:>5s}: mean {L[a].mean():7.3f} +/- {L[a].std(ddof=1):5.3f}"
                  f"   rate chosen: "
                  + ", ".join(f"{lr:g}x{n}" for lr, n in cnt.items()))
        rows = [dict(label=f"so - {a}  (LOO-tuned)",
                     d=(L["so"] - L[a]).to_numpy())
                for a in ALGS if a != "so"]
        q1 = report(rows, "R-5  TUNED COMPARISON (total AUC), out-of-sample",
                    "  Selection and evaluation use disjoint seeds, so these "
                    "p-values are valid.\n  If so(32) still wins here, the R-4 "
                    "ordering is not a learning-rate artefact.")

    # ---- the naive in-sample version, shown only as a contrast -------------
    tuned = pd.concat([d[(d.arm == a) & (d.theta_lr == best[a])] for a in ALGS])
    tuned = clean(tuned, "auc", "seed", ALGS)
    naive = [dict(label=f"so@{best['so']:g} - {a}@{best[a]:g}",
                  d=paired(tuned, "so", a, "auc", "seed")[0])
             for a in ALGS if a != "so"]
    report(naive, "R-5  IN-SAMPLE tuned comparison -- BIASED, for contrast only",
           "  The rate was chosen on the same seeds used for the test.  Shown "
           "only so the\n  size of the selection effect against the "
           "out-of-sample table is visible.")

    # ---- Q2: was the shared rate tilted toward so(32)? --------------------
    print("\n" + "=" * 96)
    print(f"R-5  Q2: WAS THE PAPER'S SHARED RATE ({PAPER_LR}) TILTED?")
    print("  best rate per algebra (whole-sample): "
          + ", ".join(f"{a}={best[a]:g}" for a in ALGS))
    print("=" * 96)
    for a in ALGS:
        v = {lr: d[(d.arm == a) & (d.theta_lr == lr)]["auc"].astype(float).mean()
             for lr in LR_GRID}
        v = {lr: m for lr, m in v.items() if np.isfinite(m)}
        if not v:
            continue
        at_paper = v.get(PAPER_LR, float("nan"))
        gap = max(v.values()) - at_paper
        flag = "  <-- the shared rate is its best" if best[a] == PAPER_LR else ""
        print(f"  {a:>5s}: best {max(v.values()):7.3f} at lr={best[a]:g}; "
              f"at lr={PAPER_LR:g} {at_paper:7.3f}  (loses {gap:.3f}){flag}")
    print("\n  Read: if the shared rate is so(32)'s best but not the others', "
          "the published\n  comparison was run at a setting that favours "
          "so(32), and only the\n  out-of-sample tuned table is admissible.")
    print("\n  CAVEAT, stated rather than hidden: this grid varies theta_lr "
          "alone.  main.py's\n  own comments say entropy_coef=0.03, "
          "geo_aux_coef=1.0 and reward_threshold were\n  each chosen to suit "
          "theta_lr=0.03, so the grid is not neutral between rates:\n  'each "
          "algebra at its own best rate' means 'best rate holding three "
          "co-tuned\n  constants fixed at values calibrated for 0.03'.")

    # ---- verdict ----------------------------------------------------------
    print("\n" + "=" * 96)
    print("VERDICT")
    print("=" * 96)
    wins = [o for o in q1 if o.get("sig") and o.get("diff", 0) > 0]
    losses = [o for o in q1 if o.get("sig") and o.get("diff", 0) < 0]
    if len(wins) == len(q1) and q1:
        print("  -> so(32) beats every other algebra at each algebra's own best "
              "rate.\n     The R-4 ordering is not a learning-rate artefact, and "
              "the shared-rate\n     confound noted in the Limitations section does "
              "not apply.")
    elif losses:
        print("  -> At its own best rate, at least one other algebra beats "
              "so(32).\n     The R-4 ordering is a learning-rate artefact.  The "
              "surviving Task-1\n     claim is not supported.")
    else:
        print("  -> Mixed or under-powered.  On this evidence the ordering is "
              "not established\n     as rate-independent; see the intervals.")


def selftest() -> None:
    ok = True

    def check(n, c, d=""):
        nonlocal ok
        ok &= bool(c)
        print(f"  {'PASS' if c else 'FAIL'}  {n}{'  ' + d if d else ''}")

    print("(1) grid and plumbing")
    check("lr grid spans the feature-net rate to 3x the paper's",
          min(LR_GRID) <= 3e-4 and max(LR_GRID) >= 0.1)
    check("the paper's rate is in the grid", PAPER_LR in LR_GRID)
    check("four algebras, all mapped", len(ALGS) == 4
          and all(a in __import__("_armlib").ALGEBRA_OF for a in ALGS))
    check("gl maps to main.py's 'full'",
          __import__("_armlib").ALGEBRA_OF["gl"] == "full")
    n = (len(ALGS) * len(LR_GRID) + 1)
    check(f"runs per seed = {n}", n == 21)

    print("(2) analysis picks the best rate per algebra and pairs correctly")
    rng = np.random.default_rng(0)
    peak = {"so": 0.03, "sl": 0.003, "gl": 0.01, "sym": 3e-4}
    top = {"so": 8.7, "sl": 8.5, "gl": 8.4, "sym": 7.9}
    rows = []
    for s in range(8):
        se = rng.normal(0, .08)
        rows.append(dict(arm=BASELINE, theta_lr=np.nan, seed=s, auc=6.5 + se,
                         r_task=.18, r_geo=.54, rho=np.nan, recon_err=1e-15,
                         status="ok"))
        for a in ALGS:
            for lr in LR_GRID:
                pen = 0.8 * abs(np.log10(lr) - np.log10(peak[a]))
                rows.append(dict(arm=a, theta_lr=lr, seed=s,
                                 auc=top[a] - pen + se + rng.normal(0, .05),
                                 r_task=.16, r_geo=.86, rho=1.0,
                                 recon_err=1e-15, status="ok"))
    df = pd.DataFrame(rows)
    import contextlib, io

    def run(x):
        b = io.StringIO()
        with contextlib.redirect_stdout(b):
            analyse(x)
        return b.getvalue()

    out = run(df)
    for a, lr in peak.items():
        check(f"best rate recovered for {a} ({lr:g})", f"{a}={lr:g}" in out)
    check("verdict fires the 'not an artefact' branch when so wins tuned",
          "not a learning-rate artefact" in out)
    check("selection and evaluation are disjoint (leave-one-seed-out)",
          "leave-one-seed-out selection" in out and "out-of-sample" in out)
    check("the biased in-sample table is shown but labelled contrast-only",
          "BIASED, for contrast only" in out)
    check("the co-tuning caveat is printed",
          "co-tuned constants fixed" in out or "not neutral between rates" in out)

    # now make sl beat so at its own best rate
    df2 = df.copy()
    df2.loc[(df2.arm == "sl") & (df2.theta_lr == 0.003), "auc"] += 1.5
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        analyse(df2)
    check("verdict fires the 'artefact' branch when another algebra wins",
          "is a learning-rate artefact" in buf.getvalue())

    print("(3) the min-n guard on the argmax")
    df_partial = df[~((df.arm == "so") & (df.theta_lr == 0.1) & (df.seed > 0))]
    # so@0.1 now has one seed; give it a huge lucky value
    df_partial = df_partial.copy()
    df_partial.loc[(df_partial.arm == "so") & (df_partial.theta_lr == 0.1),
                   "auc"] = 99.0
    check("a 1-seed cell cannot win the argmax", "so=0.1" not in run(df_partial))

    print("(4) degenerate differences are refused, not reported as significant")
    dfz = df[df.arm.isin([BASELINE] + list(ALGS))].copy()
    dfz.loc[dfz.arm == "sl", "auc"] = dfz.loc[dfz.arm == "so", "auc"].values - 1.0
    outz = run(dfz)
    check("zero-variance paired differences are marked UNTESTABLE",
          "UNTESTABLE" in outz and "zero variance" in outz)

    print("(5) failure handling")
    df3 = df.copy()
    df3.loc[(df3.arm == "sym") & (df3.seed == 0), "status"] = "RuntimeError: boom"
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        analyse(df3)
    check("failed runs are listed and their block dropped",
          "failed run(s)" in buf.getvalue())

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
        check_meta(CSV, dict(lr_grid=list(LR_GRID), algs=list(ALGS),
                             w=W, horizon=HORIZON))
        df = pd.read_csv(CSV)
    else:
        df = collect(a.seeds, a.overwrite)
    analyse(df)


if __name__ == "__main__":
    main()
