"""
R-1b: the (w, c_geo) trade-off surface, channel-decomposed.

    python run_r1b.py --selftest      # instant, no GPU, no model -- run first
    python run_r1b.py --seeds 10      # ~45 min on an RTX 5060
    python run_r1b.py --analyse-only

Requires _r1core.py alongside it.  Makes no edits to main.py.

Why
---
R-1 found that SP-PG forfeits task reward ($-0.338$ AUC) relative to a control
that carries no transformation.  R-4 localised that cost to the geometric
machinery rather than to the algebra.  Neither says how the cost depends on the
two knobs that govern it, both of which were fixed and never tuned:

    w      = env.geo_weight   (how much of the reward is geometric)   fixed 0.4
    c_geo  = cfg.geo_aux_coef (weight of the auxiliary loss on theta) fixed 1.0

The revised Limitations section states in print that this is unmeasured.  This
script measures it, and in doing so also replaces the seed-0 w-sweep whose
causal reading ("the geometry reward amplifies rather than creates the
advantage") had to be withdrawn for being n=1.

What is compared, and in which units
------------------------------------
AUC is not comparable across w: changing w changes what the reward means, so
the scale of the return changes with it.  The per-step channel means are
comparable, because r_task and r_geo each live in [0,1] whatever w is.  All
headline quantities here are therefore per-step channel means:

    task cost(w, c_geo) = mean r_task[so(32)]  -  mean r_task[baseline at same w]
    geo gain (w, c_geo) = mean r_geo [so(32)]  -  mean r_geo [baseline at same w]

paired by seed.  AUC is still recorded, but is only compared within a fixed w.

Design
------
  w      in {0.0, 0.2, 0.4, 0.6, 0.8}
  c_geo  in {0.0, 0.05, 0.25, 1.0}         (so(32) arm only).  The levels span
                                           two orders of magnitude on purpose:
                                           theta is optimised by Adam, which is
                                           approximately scale-invariant, so
                                           c_geo does not scale the update -- it
                                           only changes the ratio of auxiliary
                                           to surrogate gradient at theta.  A
                                           narrow grid could return a flat row
                                           meaning "Adam absorbed the knob"
                                           rather than "the trade-off is
                                           insensitive to it".
  plus one baseline-PPO arm per w          (c_geo is a no-op without theta:
                                            main.py guards on hasattr(policy,
                                            "theta"), so the baseline is run
                                            once per w, not once per cell)
  seeds  0..n-1
Total runs = |w| * (|c_geo| + 1) * seeds = 5 * 5 * n.

The env is built once and env.geo_weight is mutated between blocks; _r1core
reads geo_weight live, so the channel recovery stays exact.  Rebuilding the env
per w would re-run the GPT-2 embedding pre-computation five times for no gain,
and would also redraw M_env, which must stay fixed across the sweep.

The question it answers
-----------------------
At c_geo = 0 the auxiliary loss is off and theta receives gradient only through
the policy objective.  If the task cost vanishes there while a geometry gain
survives, the trade-off is a property of the auxiliary loss and is tunable
away.  If the cost persists at c_geo = 0, it is intrinsic to acting through M.

Outputs (written incrementally)
    r1b_cells.csv     w, c_geo, arm, seed, r_task, r_geo, auc, rho, status
    r1b_channels.csv  w, c_geo, arm, seed, iteration, r_task, r_geo
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import pandas as pd

from _r1core import channel_aucs, channels, instrument, instrument_trainer

CELL_CSV = "r1b_cells.csv"
CH_CSV = "r1b_channels.csv"

W_GRID = (0.0, 0.2, 0.4, 0.6, 0.8)
CGEO_GRID = (0.0, 0.05, 0.25, 1.0)
METHOD, BASELINE = "so(32)", "baseline_ppo"

CELL_COLS = ["w", "c_geo", "arm", "seed", "r_task", "r_geo", "auc",
             "rho", "recon_err", "status"]
CH_COLS = ["w", "c_geo", "arm", "seed", "iteration", "r_task", "r_geo"]


class _Sink:
    def __init__(self, path, cols, overwrite=False):
        self.path, self.cols, self.n = path, cols, 0
        if os.path.exists(path) and os.path.getsize(path) > 0 and not overwrite:
            sys.exit(f"{path} already exists and is non-empty.  Refusing to "
                     f"truncate a previous sweep.  Move it aside, or pass "
                     f"--overwrite.")
        pd.DataFrame(columns=cols).to_csv(path, index=False)

    def add(self, rows):
        if rows:
            pd.DataFrame(rows, columns=self.cols).to_csv(
                self.path, mode="a", header=False, index=False)
            self.n += len(rows)


def _guard_outputs(overwrite: bool) -> None:
    """Check before the expensive model load, not after."""
    for p in (CELL_CSV, CH_CSV):
        if os.path.exists(p) and os.path.getsize(p) > 0 and not overwrite:
            sys.exit(f"{p} already exists and is non-empty.  Refusing to "
                     f"truncate a previous sweep.\n"
                     f"Move it aside, or re-run with --overwrite.")


def collect(n_seeds: int, overwrite: bool = False) -> pd.DataFrame:
    _guard_outputs(overwrite)
    from main import (GPT2EmbeddingProvider, MultiStepTextAlignmentEnv,
                      LiePolicy, BaselinePolicy, ValueNet, LieStructuredPPO,
                      _default_cfg, _seed_all)

    embedder = GPT2EmbeddingProvider(model_name="gpt2-medium", max_length=64)
    env = MultiStepTextAlignmentEnv(
        embedder, n_prompts=16, n_actions=16, horizon=20,
        reward_noise=0.2, geo_weight=W_GRID[0], sparse_prob=0.7,
    )
    instrument(env)                       # reads geo_weight live
    horizon = int(env.horizon)
    state_dim, n_actions, k = embedder.hidden_dim, env.n_actions, int(env.k_transform)

    cells = _Sink(CELL_CSV, CELL_COLS, overwrite)
    chans = _Sink(CH_CSV, CH_COLS, overwrite)
    rows_all = []
    total = len(W_GRID) * (len(CGEO_GRID) + 1) * n_seeds
    done = 0

    for w in W_GRID:
        env.geo_weight = float(w)         # mutate, do not rebuild: M_env fixed
        plan = [(BASELINE, None)] + [(METHOD, c) for c in CGEO_GRID]

        for seed in range(n_seeds):
            for arm, c_geo in plan:
                _seed_all(seed)
                cfg = _default_cfg()
                if c_geo is not None:
                    cfg.geo_aux_coef = float(c_geo)

                try:
                    if arm == BASELINE:
                        policy = BaselinePolicy(state_dim, n_actions)
                        trainer = LieStructuredPPO(
                            env, policy, ValueNet(state_dim), cfg,
                            use_lie_projection=False)
                    else:
                        policy = LiePolicy(state_dim, n_actions, k=k,
                                           algebra="so")
                        trainer = LieStructuredPPO(
                            env, policy, ValueNet(state_dim), cfg,
                            use_lie_projection=True, algebra="so")
                    instrument_trainer(trainer, env, None, horizon, strict=True)
                    res = trainer.train(verbose=False)
                except BaseException as e:   # noqa: BLE001  incl. SystemExit
                    print(f"  w={w:.1f} c={c_geo} {arm:13s} s{seed:2d}  "
                          f"FAILED: {type(e).__name__}: {e}", flush=True)
                    row = {c: np.nan for c in CELL_COLS}
                    row.update(w=w, c_geo=(np.nan if c_geo is None else c_geo),
                               arm=arm, seed=seed,
                               status=f"{type(e).__name__}: {e}"[:120])
                    cells.add([row]); rows_all.append(row); done += 1
                    continue

                pairs = channels(res, trainer)
                cg = np.nan if c_geo is None else float(c_geo)
                chans.add([{"w": w, "c_geo": cg, "arm": arm, "seed": seed,
                            "iteration": i, "r_task": rt, "r_geo": rg}
                           for i, (rt, rg) in enumerate(pairs, start=1)])

                rt_m = float(np.mean([p[0] for p in pairs]))
                rg_m = float(np.mean([p[1] for p in pairs]))
                _, _, a_tot = channel_aucs(pairs, w, horizon)
                auc = float(res["auc"])
                spec = [s for s in res.get("spectral_radii", [])
                        if s is not None and math.isfinite(s)]
                recon = abs(a_tot - auc)
                finite = all(math.isfinite(x) for x in (rt_m, rg_m, auc, recon))
                row = {"w": w, "c_geo": cg, "arm": arm, "seed": seed,
                       "r_task": rt_m, "r_geo": rg_m, "auc": auc,
                       "rho": float(np.median(spec)) if spec else np.nan,
                       "recon_err": recon,
                       "status": "ok" if finite else "non-finite"}
                cells.add([row]); rows_all.append(row); done += 1

                print(f"  [{done:4d}/{total}] w={w:.1f} c_geo="
                      f"{'--  ' if c_geo is None else f'{c_geo:.2f}'} "
                      f"{arm:13s} s{seed:2d}  r_task={rt_m:.4f}  "
                      f"r_geo={rg_m:.4f}  AUC={auc:7.3f}", flush=True)

    df = pd.DataFrame(rows_all, columns=CELL_COLS)
    print(f"\nWrote {CELL_CSV} ({cells.n} rows) and {CH_CSV} ({chans.n} rows)")
    return df


# --------------------------------------------------------------------------- #
def _dupe_check(df: pd.DataFrame) -> None:
    key = ["w", "c_geo", "arm", "seed"]
    d = df.copy()
    d["c_geo"] = d["c_geo"].fillna(-1.0)          # baseline rows carry NaN
    n = d.duplicated(key).sum()
    if n:
        sys.exit(f"{n} duplicated (w, c_geo, arm, seed) rows in {CELL_CSV}. "
                 "Two partial sweeps were probably concatenated; the paired "
                 "join would expand them and inflate n.  Deduplicate first.")


def _paired(df, w, c_geo, col):
    """method - baseline at this (w, c_geo), matched on seed. Returns (d, n)."""
    m = df[(df.arm == METHOD) & (df.w == w) & (df.c_geo == c_geo)]
    b = df[(df.arm == BASELINE) & (df.w == w)]
    m = m.set_index("seed")[col].astype(float)
    b = b.set_index("seed")[col].astype(float)
    idx = m.index.intersection(b.index)
    d = (m.loc[idx] - b.loc[idx]).replace([np.inf, -np.inf], np.nan).dropna()
    return d.to_numpy(), len(d)


def _within(df, w, c_a, c_b, col):
    """method(c_a) - method(c_b) at this w, matched on seed.

    This is the contrast that answers the experiment's question.  Comparing two
    cells of the method-minus-baseline table is not a test: both share the same
    baseline term, so their errors are correlated.  Differencing within the
    method arm cancels the baseline entirely and is strictly more powerful.
    """
    a = df[(df.arm == METHOD) & (df.w == w) & (df.c_geo == c_a)]
    b = df[(df.arm == METHOD) & (df.w == w) & (df.c_geo == c_b)]
    a = a.set_index("seed")[col].astype(float)
    b = b.set_index("seed")[col].astype(float)
    idx = a.index.intersection(b.index)
    d = (a.loc[idx] - b.loc[idx]).replace([np.inf, -np.inf], np.nan).dropna()
    return d.to_numpy(), len(d)


def analyse(df: pd.DataFrame) -> None:
    from sppg_defense.stats.tests import holm, paired_report, tost_paired

    _dupe_check(df)

    if "recon_err" in df.columns:
        e = pd.to_numeric(df["recon_err"], errors="coerce")
        nonfinite = int((~np.isfinite(e)).sum())
        e = e[np.isfinite(e)]
        if len(e):
            print(f"\nchannel-reconstruction error vs main.py's AUC: max = "
                  f"{e.max():.3e}  (must be ~1e-12)"
                  + (f"   [{nonfinite} run(s) non-finite and excluded from "
                     f"this max -- see the failure list below]" if nonfinite else ""))

    bad = df[df.status != "ok"]
    if len(bad):
        print(f"\n{len(bad)} run(s) failed or produced non-finite values; "
              f"those cells lose those seeds:")
        for _, r in bad.iterrows():
            print(f"    w={r.w} c_geo={r.c_geo} {r.arm} seed={r.seed}: {r.status}")
    df = df[df.status == "ok"]
    n_max = int(df.groupby(["w", "arm"]).seed.nunique().max()) if len(df) else 0

    # ---- families of tests, Holm-corrected within each channel -------------
    for col, label in [("r_task", "TASK CHANNEL"), ("r_geo", "GEOMETRY CHANNEL")]:
        rows, pvals = [], []
        for w in W_GRID:
            for c in CGEO_GRID:
                d, n = _paired(df, w, c, col)
                if n < 3:
                    rows.append(dict(w=w, c_geo=c, n=n)); pvals.append(1.0)
                    continue
                r = paired_report(d, np.zeros_like(d))
                rows.append(dict(w=w, c_geo=c, n=n, diff=r.diff,
                                 sd=float(np.std(d, ddof=1)),
                                 ci=r.diff_ci, dz=r.cohen_dz,
                                 p=r.t_p, power=r.achieved_power))
                pvals.append(r.t_p)
        pvals = [p if np.isfinite(p) else 1.0 for p in pvals]
        hp = holm(pvals, 0.05)
        for row, ph, rj in zip(rows, hp["p_holm"], hp["reject"]):
            row["p_holm"], row["sig"] = ph, rj

        print("\n" + "=" * 96)
        print(f"R-1b  {label}:  so(32) minus baseline PPO (per-step mean), "
              f"paired by seed")
        print(f"      Holm-corrected over this family of {len(rows)} tests.  "
              f"* = significant after correction.")
        print("=" * 96)
        print(f"  {'w':>4s} {'c_geo':>6s} {'n':>3s} {'diff':>10s} {'sd':>8s} "
              f"{'95% CI':>20s} {'dz':>7s} {'p_holm':>10s} {'power':>7s}")
        for r in rows:
            if "diff" not in r:
                print(f"  {r['w']:>4.1f} {r['c_geo']:>6.2f} {r['n']:>3d}"
                      f"   (insufficient seeds)")
                continue
            flag = "*" if r["sig"] else " "
            warn = "  <-- n<max" if r["n"] < n_max else ""
            print(f"  {r['w']:>4.1f} {r['c_geo']:>6.2f} {r['n']:>3d} "
                  f"{r['diff']:>+10.4f}{flag} {r['sd']:>8.4f} "
                  f"[{r['ci'][0]:>+8.4f},{r['ci'][1]:>+8.4f}] "
                  f"{r['dz']:>+7.2f} {r['p_holm']:>10.2e} "
                  f"{(r['power'] if r['power'] is not None else float('nan')):>7.2f}{warn}")
        print("  power is the achieved power for the observed effect at this n; "
              "a null with\n  power well below 0.8 is uninformative, not "
              "evidence of no effect.")

    # ---- the decisive contrast, within the method arm ----------------------
    print("\n" + "=" * 96)
    print("IS THE TASK COST TUNABLE AWAY?   c_geo = 0 versus c_geo = "
          f"{CGEO_GRID[-1]:.2f}, within the so(32) arm")
    print("  The baseline cancels, so this is a direct test rather than a "
          "comparison of two\n  differences that share a baseline term.  "
          "Holm-corrected over the two channels x "
          f"{len(W_GRID)} w-values.")
    print("=" * 96)
    rows, pvals = [], []
    for col in ("r_task", "r_geo"):
        for w in W_GRID:
            d, n = _within(df, w, CGEO_GRID[0], CGEO_GRID[-1], col)
            if n < 3:
                rows.append(dict(col=col, w=w, n=n)); pvals.append(1.0); continue
            r = paired_report(d, np.zeros_like(d))
            rows.append(dict(col=col, w=w, n=n, diff=r.diff, ci=r.diff_ci,
                             dz=r.cohen_dz, p=r.t_p, power=r.achieved_power))
            pvals.append(r.t_p)
    pvals = [p if np.isfinite(p) else 1.0 for p in pvals]
    hp = holm(pvals, 0.05)
    for row, ph, rj in zip(rows, hp["p_holm"], hp["reject"]):
        row["p_holm"], row["sig"] = ph, rj
    print(f"  {'channel':>10s} {'w':>4s} {'n':>3s} {'c=0 minus c=max':>17s} "
          f"{'95% CI':>20s} {'dz':>7s} {'p_holm':>10s} {'power':>7s}")
    for r in rows:
        if "diff" not in r:
            print(f"  {r['col']:>10s} {r['w']:>4.1f} {r['n']:>3d}"
                  f"   (insufficient seeds)"); continue
        flag = "*" if r["sig"] else " "
        print(f"  {r['col']:>10s} {r['w']:>4.1f} {r['n']:>3d} "
              f"{r['diff']:>+17.4f}{flag} "
              f"[{r['ci'][0]:>+8.4f},{r['ci'][1]:>+8.4f}] {r['dz']:>+7.2f} "
              f"{r['p_holm']:>10.2e} "
              f"{(r['power'] if r['power'] is not None else float('nan')):>7.2f}")

    # ---- equivalence, so a null can be stated as a positive result ---------
    print("\n" + "=" * 96)
    print("EQUIVALENCE TEST at c_geo = 0 (TOST): is the task cost absent, "
          "or merely undetected?")
    print("  Margin = 25% of the task cost measured at c_geo = "
          f"{CGEO_GRID[-1]:.2f} for the same w.")
    print("=" * 96)
    for w in W_GRID:
        d0, n0 = _paired(df, w, CGEO_GRID[0], "r_task")
        d1, n1 = _paired(df, w, CGEO_GRID[-1], "r_task")
        if n0 < 3 or n1 < 3 or abs(np.mean(d1)) < 1e-12:
            print(f"  w={w:.1f}   (not evaluable)"); continue
        margin = 0.25 * abs(np.mean(d1))
        try:
            t = tost_paired(d0, np.zeros_like(d0), margin=margin)
        except ValueError as exc:
            print(f"  w={w:.1f}   TOST undefined: {exc}"); continue
        verdict = ("Equivalent to zero within the margin" if t["equivalent"]
                   else "Not shown equivalent -- the null is uninformative")
        print(f"  w={w:.1f}  cost at c=0 = {np.mean(d0):+.4f}  "
              f"margin = +/-{margin:.4f}  TOST p = {t['p_tost']:.4f}  "
              f"smallest margin that would hold = {t['smallest_equivalence_margin']:.4f}"
              f"   {verdict}")

    # ---- AUC within a fixed w (never across w) -----------------------------
    print("\n" + "=" * 96)
    print("AUC, so(32) minus baseline PPO -- compared within each w only")
    print("  (AUC is not comparable across w: changing w rescales the return.)")
    print("=" * 96)
    for w in W_GRID:
        cells = []
        for c in CGEO_GRID:
            d, n = _paired(df, w, c, "auc")
            cells.append(f"c={c:.2f}: {np.mean(d):+7.3f} (n={n})"
                         if n >= 2 else f"c={c:.2f}: n/a")
        print(f"  w={w:.1f}   " + " | ".join(cells))

    print("\nRead: a cell with task cost indistinguishable from zero and a "
          "positive geometry\ngain is a setting at which the geometric "
          "advantage is obtained for free.  c_geo = 0\nturns the auxiliary "
          "loss off entirely; if the cost persists there it is intrinsic to\n"
          "acting through M, not to the auxiliary objective.")
    print("\nCaveat: r_task and r_geo are derived from the same reward stream "
          "(r_task is\nrecovered by inverting the mixture), so the two "
          "families are mechanically coupled\nand are not 40 independent "
          "questions.")


# --------------------------------------------------------------------------- #
def selftest() -> None:
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")

    print("(1) channel recovery under a changing geo_weight")

    class StubEnv:
        horizon = 20

        def __init__(self):
            self.geo_weight = 0.4
            self.truth, self._t = [], 0

        def _geometry_reward(self, v, M):
            return 0.5 + 0.004 * (len(self.truth) % 20)

        def step(self, a, M=None):
            terminal = (self._t == self.horizon - 1)
            t = 0.9 - 0.05 * (len(self.truth) % 4)
            if not terminal and (len(self.truth) % 3 == 0):
                t = 0.0
            g = self._geometry_reward(None, M)
            self.truth.append((t, g, terminal))
            self._t = 0 if terminal else self._t + 1
            w = self.geo_weight
            return None, float((1 - w) * t + w * g), terminal

    class StubTrainer:
        def __init__(self, env, n_steps, n_iters):
            self.env, self.n_steps, self.n_iters = env, n_steps, n_iters

        def _gather_trajectory(self):
            cur, rets = 0.0, []
            for _ in range(self.n_steps):
                _, r, d = self.env.step(0)
                cur += r
                if d:
                    rets.append(cur); cur = 0.0
            return (None,) * 6 + (float(np.mean(rets)) if rets else 0.0,)

        def train(self):
            eps = [self._gather_trajectory()[6] for _ in range(self.n_iters)]
            from sppg_defense.rl.analysis import auc_trapezoid
            return {"auc": float(auc_trapezoid(eps))}

    for w in W_GRID:
        e = StubEnv()
        e.geo_weight = w
        instrument(e)
        tr = StubTrainer(e, 512, 6)
        instrument_trainer(tr, e, None, e.horizon, strict=True)
        res = tr.train()
        pairs = channels(res, tr)
        a_t, a_g, a_tot = channel_aucs(pairs, w, e.horizon)
        check(f"w={w:.1f}: decomposition == reported AUC",
              abs(a_tot - res["auc"]) < 1e-9,
              f"|diff|={abs(a_tot-res['auc']):.1e}")
        # Independent replay of the env's own log, using the same episode
        # accounting main.py uses, compared iteration by iteration.
        exp_t, exp_g = [], []
        cur_t = cur_g = 0.0
        eps_t, eps_g = [], []
        for j, (tt, gg, term) in enumerate(e.truth, start=1):
            cur_t += tt; cur_g += gg
            if term:
                eps_t.append(cur_t); eps_g.append(cur_g)
                cur_t = cur_g = 0.0
            if j % tr.n_steps == 0:                 # iteration boundary
                exp_t.append(np.mean(eps_t) / e.horizon)
                exp_g.append(np.mean(eps_g) / e.horizon)
                eps_t, eps_g = [], []
                cur_t = cur_g = 0.0
        err_t = max(abs(a - b) for (a, _), b in zip(pairs, exp_t))
        err_g = max(abs(a - b) for (_, a), b in zip(pairs, exp_g))
        check(f"w={w:.1f}: r_task/r_geo match an independent replay",
              max(err_t, err_g) < 1e-12,
              f"max err {max(err_t, err_g):.1e}")

    print("(2) a single env instance survives geo_weight being mutated")
    e = StubEnv(); instrument(e)
    seen = []
    for w in (0.0, 0.8, 0.2):
        e.geo_weight = w
        tr = StubTrainer(e, 512, 3)
        instrument_trainer(tr, e, None, e.horizon, strict=True)
        r = tr.train()
        p = channels(r, tr)
        seen.append(abs(channel_aucs(p, w, e.horizon)[2] - r["auc"]))
    check("exact at every w after mutation", max(seen) < 1e-9,
          f"max |diff|={max(seen):.1e}")

    print("(3) grid and plumbing")
    check("w grid excludes 1.0 (task channel identifiable)",
          all(0.0 <= w < 1.0 for w in W_GRID))
    check("c_geo grid includes 0.0 (auxiliary loss off)", 0.0 in CGEO_GRID)
    n = len(W_GRID) * (len(CGEO_GRID) + 1)
    check(f"runs per seed = {n}", n == 25)
    try:
        from sppg_defense.stats.tests import paired_report  # noqa: F401
        check("sppg_defense.stats importable", True)
    except ImportError as exc:
        check("sppg_defense.stats importable", False, str(exc))

    print("(4) analysis: Holm, pairing, dupes, NaN handling, within-contrast")
    rng = np.random.default_rng(0)
    rows = []
    for w in W_GRID:
        for s_ in range(8):
            rows.append(dict(w=w, c_geo=np.nan, arm=BASELINE, seed=s_,
                             r_task=0.18 + rng.normal(0, .002),
                             r_geo=0.54 + rng.normal(0, .002), auc=6.5,
                             rho=np.nan, recon_err=1e-15, status="ok"))
            for c in CGEO_GRID:
                cost = -0.03 * (c / max(CGEO_GRID))      # cost grows with c_geo
                rows.append(dict(w=w, c_geo=c, arm=METHOD, seed=s_,
                                 r_task=0.18 + cost + rng.normal(0, .002),
                                 r_geo=0.54 + 0.3 * (c > 0) + rng.normal(0, .002),
                                 auc=8.7, rho=1.0, recon_err=1e-15, status="ok"))
    d = pd.DataFrame(rows)
    dd, n = _paired(d, 0.4, max(CGEO_GRID), "r_task")
    check("pairing matches on seed and w", n == 8 and abs(dd.mean() + 0.03) < 0.005,
          f"n={n} mean={dd.mean():+.4f}")
    wd, wn = _within(d, 0.4, CGEO_GRID[0], CGEO_GRID[-1], "r_task")
    check("within-method contrast cancels the baseline",
          wn == 8 and abs(wd.mean() - 0.03) < 0.005, f"mean={wd.mean():+.4f}")

    d2 = d.copy()
    d2.loc[(d2.arm == BASELINE) & (d2.w == 0.4) & (d2.seed == 0), "status"] = "boom"
    _, n2 = _paired(d2[d2.status == "ok"], 0.4, max(CGEO_GRID), "r_task")
    check("a failed baseline drops that seed from every cell at that w", n2 == 7,
          f"n={n2}")

    d3 = pd.concat([d, d.iloc[:1]], ignore_index=True)
    try:
        _dupe_check(d3); check("duplicate (w,c_geo,arm,seed) detected", False)
    except SystemExit:
        check("duplicate (w,c_geo,arm,seed) detected", True)

    d4 = d.copy()
    d4.loc[(d4.arm == METHOD) & (d4.w == 0.4) & (d4.seed == 1), "r_task"] = np.inf
    dd4, n4 = _paired(d4, 0.4, max(CGEO_GRID), "r_task")
    check("inf dropped from the paired vector", n4 == 7 and np.isfinite(dd4).all())

    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        analyse(d)
    out = buf.getvalue()
    check("analysis runs end to end", "IS THE TASK COST TUNABLE AWAY?" in out
          and "EQUIVALENCE TEST" in out and "Holm-corrected" in out)
    check("no uncorrected star language", "p_holm" in out)

    print("(5) the strict guard rejects a NaN reconstruction")
    from _r1core import instrument_trainer as _it

    class BadEnv:
        geo_weight, horizon = 0.4, 20
        def _geometry_reward(self, v, M): return float("nan")
        def step(self, a, M=None): return None, float("nan"), True

    class BadTr:
        def __init__(self, env): self.env = env
        def _gather_trajectory(self):
            for _ in range(20): self.env.step(0)
            return (None,) * 6 + (1.0,)

    be = BadEnv(); instrument(be); bt = BadTr(be)
    _it(bt, be, None, be.horizon, strict=True)
    try:
        bt._gather_trajectory(); check("NaN reconstruction raises", False)
    except (ValueError, RuntimeError):
        check("NaN reconstruction raises", True)

    print("\n" + ("SELFTEST PASS" if ok else "SELFTEST FAIL"))
    if not ok:
        sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--analyse-only", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--overwrite", action="store_true",
                    help="allow truncating existing r1b_*.csv")
    a = ap.parse_args()

    if a.selftest:
        selftest(); return
    if a.analyse_only:
        if not os.path.exists(CELL_CSV):
            sys.exit(f"{CELL_CSV} not found; run without --analyse-only first.")
        df = pd.read_csv(CELL_CSV)
    else:
        df = collect(a.seeds, a.overwrite)
    analyse(df)


if __name__ == "__main__":
    main()
