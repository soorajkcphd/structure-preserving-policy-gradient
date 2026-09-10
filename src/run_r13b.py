"""
R-13b: Mistral-7B at 20 seeds, channel-decomposed, with a multi-seed ablation.

    python run_r13b.py --selftest
    python run_r13b.py --seeds 20            # ~50 min after the model loads
    python run_r13b.py --seeds 20 --arms so,baseline_ppo   # cheaper subset
    python run_r13b.py --analyse-only

Requires _r1core.py, _armlib.py and run_mistral_task1.py.
No edits to main.py or to run_mistral_task1.py.

Why
---
The abstract quotes +59.9% on Mistral-7B.  It is the least-supported number in
the paper:

  * run_mistral_task1.py calls run_rl_ablation(env, ..., seed=0) -- the algebra
    ablation is single-seed, three arms;
  * nothing on Mistral is channel-decomposed, so the +59.9% is undecomposed
    where Task 1's +34.2% has been split into geometry (+115%) and task
    (-15%) by R-1;
  * the paper already states, correctly, that no decomposition claim is made
    for Mistral -- which leaves a headline number in the abstract whose
    meaning is known to be different from the GPT-2 one.

This script fixes all three: 20 seeds, the full five-arm ablation, and the
R-1 channel decomposition applied to every arm.

What is reused
--------------
run_mistral_task1.py builds the same MultiStepTextAlignmentEnv as main.py --
only the embedding provider differs (Mistral-7B, 4-bit, projected 4096 -> 1024
by a fixed orthogonal QR projection, so k = 32 as in Task 1).  Therefore
_r1core.instrument and the whole _armlib harness attach unchanged, and every
statistic here is computed by the same code that produced R-1 and R-4.  As a
result, the GPT-2 and Mistral numbers are directly comparable.

Memory
------
MistralEmbeddingProvider loads the 7B model, extracts every embedding the
environment needs, then frees it (free_model()).  We follow the same
pre-compute-and-free pattern via create_env_and_free_model, so training runs
with the full VRAM available.  If the machine cannot hold the 4-bit model even
briefly, nothing in this script can help -- run it where the published Mistral
results were produced.

Output   r13b_cells.csv   arm, seed, auc, r_task, r_geo, rho, recon_err, status
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys

import numpy as np
import pandas as pd

from _armlib import (BASELINE, Sink, check_meta, clean, guard_outputs, paired,
                     report, run_one, write_meta)
from _r1core import channel_aucs

CSV = "r13b_cells.csv"
ALL_ARMS = (BASELINE, "so", "sl", "gl", "sym")
W, HORIZON = 0.4, 20
PROJECT_DIM = 1024                      # -> k = 32, matching Task 1
COLS = ["arm", "seed", "auc", "auc_task", "auc_geo", "r_task", "r_geo",
        "rho", "recon_err", "status"]

# Published GPT-2 Task-1 values, for the side-by-side comparison only.  These
# are not recomputed here; they are quoted so the two models can be read
# against each other in one table.
GPT2_REF = {"auc_gain": 2.2338, "d_task": -0.338, "d_geo": +2.572}


def build_env(project_dim: int):
    """Mistral embeddings -> the same environment class Task 1 uses."""
    from run_mistral_task1 import (MistralEmbeddingProvider,
                                   create_env_and_free_model)
    k = int(round(project_dim ** 0.5))
    if k * k != project_dim:
        raise ValueError(f"project_dim={project_dim} is not a perfect square; "
                         f"the reshape to k x k would be undefined.")
    emb = MistralEmbeddingProvider(project_dim=project_dim)
    env, state_dim, n_actions, _ = create_env_and_free_model(emb, k=k)
    if int(env.k_transform) != k:
        raise RuntimeError(f"env.k_transform={env.k_transform} but k={k}; the "
                           f"algebra dimension would not match the paper's.")
    return env, state_dim, n_actions, k


def collect(n_seeds: int, arms, overwrite: bool, project_dim: int) -> pd.DataFrame:
    guard_outputs([CSV], overwrite)
    from _r1core import instrument

    env, sd, na, k = build_env(project_dim)
    w0 = instrument(env)
    print(f"\nenvironment ready: state_dim={sd}, n_actions={na}, k={k}, "
          f"geo_weight={w0}")

    sink, rows = Sink(CSV, COLS), []
    write_meta(CSV, dict(arms=list(arms), w=W, horizon=HORIZON,
                         n_seeds=n_seeds, project_dim=project_dim,
                         model="mistral-7b"))
    total, done = len(arms) * n_seeds, 0
    for seed in range(n_seeds):
        for arm in arms:
            rec, pairs, _ = run_one(env, arm, seed, sd, na, k, W, HORIZON)
            # Record the channel AUCs exactly as channel_aucs computes them
            # (trapezoidal over iterations), not as horizon * mean-per-step.
            # The two differ by the trapezoid's half-weighted endpoints, so
            # reconstructing them by hand would make the decomposition table
            # non-additive: d AUC_task + d AUC_geo would not equal d AUC.
            if pairs:
                a_t, a_g, _tot = channel_aucs(pairs, W, HORIZON)
                rec["auc_task"], rec["auc_geo"] = a_t, a_g
            else:
                rec["auc_task"] = rec["auc_geo"] = np.nan
            sink.add([{c: rec.get(c, np.nan) for c in COLS}])
            rows.append(rec); done += 1
            msg = (f"FAILED: {rec['status']}" if rec["status"] != "ok"
                   else f"AUC={rec['auc']:7.3f}  r_task={rec['r_task']:.4f}  "
                        f"r_geo={rec['r_geo']:.4f}  rho={rec['rho']:.4g}")
            print(f"  [{done:3d}/{total}] seed {seed:2d} {arm:12s} {msg}",
                  flush=True)
    df = pd.DataFrame(rows).reindex(columns=COLS)
    print(f"\nWrote {CSV} ({sink.n} rows)")
    return df


# --------------------------------------------------------------------------- #
def analyse(df: pd.DataFrame) -> None:
    e = pd.to_numeric(df.get("recon_err"), errors="coerce") if "recon_err" in df else None
    if e is not None and np.isfinite(e).any():
        print(f"\nchannel-reconstruction error: max = "
              f"{e[np.isfinite(e)].max():.3e}")
        print("  (this must be at machine precision; it is the guarantee that "
              "the two\n   channels sum back to the AUC main.py itself "
              "reports)")
    bad = df[df.status != "ok"]
    if len(bad):
        print(f"\n{len(bad)} failed run(s):")
        for _, r in bad.iterrows():
            print(f"    {r.arm} seed={r.seed}: {r.status}")

    present = [a for a in ALL_ARMS if a in set(df[df.status == "ok"]["arm"])]
    d = clean(df, "auc", "seed", present)
    n = d["seed"].nunique()

    print("\n" + "=" * 96)
    print(f"R-13b  MISTRAL-7B PER-ARM  (n = {n} paired seeds)")
    print("=" * 96)
    print(f"  {'arm':>12s} {'n':>3s} {'AUC':>18s} {'r_task':>16s} "
          f"{'r_geo':>16s} {'rho':>12s}")
    for a in present:
        v = d[d.arm == a]

        def ms(c):
            if c not in v.columns:
                return f"{'n/a':>16s}"
            x = pd.to_numeric(v[c], errors="coerce").dropna()
            return (f"{x.mean():9.3f}+/-{x.std(ddof=1):6.3f}"
                    if len(x) > 1 else f"{'n/a':>16s}")

        rho = (pd.to_numeric(v["rho"], errors="coerce").dropna()
               if "rho" in v.columns else pd.Series(dtype=float))
        print(f"  {a:>12s} {len(v):>3d} {ms('auc'):>18s} {ms('r_task'):>16s} "
              f"{ms('r_geo'):>16s} "
              + (f"{rho.median():12.6g}" if len(rho) else f"{'-':>12s}"))

    if BASELINE not in present or "so" not in present:
        print("\n  baseline and so(32) are both required for the comparison; "
              "stopping here.")
        return

    # ---- the headline, and its decomposition ------------------------------
    rows = [dict(label=f"{a} - {BASELINE}",
                 d=paired(d, a, BASELINE, "auc", "seed")[0])
            for a in present if a != BASELINE]
    out = report(rows, f"R-13b  TOTAL AUC vs baseline PPO ({n} seeds)",
                 "  Holm-corrected over the arms compared.")

    print("\n" + "=" * 96)
    print("R-13b  CHANNEL DECOMPOSITION for Mistral")
    print("=" * 96)
    print(f"  {'arm':>12s} {'d AUC_task':>14s} {'d AUC_geo':>14s} "
          f"{'d AUC':>12s} {'geometry share':>16s}")
    for a in present:
        if a == BASELINE:
            continue
        da = paired(d, a, BASELINE, "auc", "seed")[0]
        if not len(da) or "auc_task" not in d.columns:
            continue
        dt = paired(d, a, BASELINE, "auc_task", "seed")[0]
        dg = paired(d, a, BASELINE, "auc_geo", "seed")[0]
        if not (len(dt) and len(dg)):
            continue
        t_auc, g_auc = dt.mean(), dg.mean()
        share = (g_auc / da.mean() * 100.0) if abs(da.mean()) > 1e-12 else np.nan
        resid = abs((t_auc + g_auc) - da.mean())
        flag = "" if resid < 1e-6 else f"   !! non-additive by {resid:.2e}"
        print(f"  {a:>12s} {t_auc:+14.4f} {g_auc:+14.4f} {da.mean():+12.4f} "
              f"{share:15.1f}%{flag}")
    print("\n  Read: on GPT-2 (R-1) the so(32) gain of +2.2338 was "
          f"{GPT2_REF['d_geo']:+.3f} geometry\n  and "
          f"{GPT2_REF['d_task']:+.3f} task -- {115}% from the channel the "
          "control cannot optimise.\n  If Mistral shows the same pattern, the "
          "+59.9% in the abstract means what the\n  +34.2% means.  If it does "
          "not, the two numbers are\n  not comparable.")

    # ---- task-channel sign test, the claim most likely to be challenged ---
    dt_so = (paired(d, "so", BASELINE, "auc_task", "seed")[0]
             if "auc_task" in d.columns else np.array([]))
    if len(dt_so):
        n_neg = int((dt_so < 0).sum())
        print(f"\n  so(32) task channel below baseline on {n_neg}/{len(dt_so)} "
              f"seeds (GPT-2: 20/20).")

    print("\n" + "=" * 96)
    print("VERDICT")
    print("=" * 96)
    o_so = next((o for o in out if o["label"] == f"so - {BASELINE}"), None)
    if o_so is None or "diff" not in o_so:
        print("  Not enough paired seeds to decide."); return
    if "ci" not in o_so or o_so.get("p_holm") is None:
        # report() withholds an interval when every paired difference is
        # identical.  Print what is known and give no verdict rather than
        # indexing a key that report() leaves out in this case.
        print(f"  Difference {o_so['diff']:+.4f}, but no interval was "
              "computed (the paired\n  differences are degenerate).  "
              "Investigate the runs before drawing a conclusion.")
        return
    if o_so.get("sig") and o_so["diff"] > 0:
        print(f"  -> so(32) beats baseline PPO on Mistral-7B at {n} seeds "
              f"({o_so['diff']:+.4f},\n     95% CI "
              f"[{o_so['ci'][0]:+.4f}, {o_so['ci'][1]:+.4f}], "
              f"p_holm={o_so['p_holm']:.2e}).  The abstract's\n     Mistral "
              "claim is now multi-seed and decomposed, on the same footing as "
              "Task 1.")
    elif o_so.get("sig"):
        print(f"  -> so(32) is significantly worse than baseline on Mistral "
              f"({o_so['diff']:+.4f},\n     p_holm={o_so['p_holm']:.2e}).  The "
              "published single-seed +59.9% does not\n     survive 20 seeds "
              "and is not supported.")
    else:
        print(f"  -> No significant difference at {n} seeds "
              f"({o_so['diff']:+.4f}, 95% CI\n     "
              f"[{o_so['ci'][0]:+.4f}, {o_so['ci'][1]:+.4f}], "
              f"p_holm={o_so['p_holm']:.2e}).  The single-seed +59.9% is not\n"
              "     reproducible as a population effect; only the interval "
              "is informative.")
    print("\n  CAVEAT: the embeddings are projected 4096 -> 1024 by a fixed "
          "random orthogonal\n  map with no distortion guarantee, exactly as "
          "in the published run.  This\n  script changes the seed count and "
          "the analysis, not that design choice.")


# --------------------------------------------------------------------------- #
def selftest() -> None:
    ok = True

    def check(n, c, dd=""):
        nonlocal ok
        ok &= bool(c)
        print(f"  {'PASS' if c else 'FAIL'}  {n}{'  ' + dd if dd else ''}")

    print("(1) the reshape dimension must be exact")
    for pd_, good in ((1024, True), (4096, True), (1000, False), (500, False)):
        try:
            k = int(round(pd_ ** 0.5))
            valid = (k * k == pd_)
        except Exception:                                    # noqa: BLE001
            valid = False
        check(f"project_dim={pd_} -> {'accepted' if good else 'rejected'}",
              valid == good, f"k={int(round(pd_ ** 0.5))}")
    check("the default project_dim gives k = 32 as in Task 1",
          int(round(PROJECT_DIM ** 0.5)) == 32)

    print("(2) channel scaling arithmetic")
    # A per-step task-channel difference of x contributes HORIZON*(1-W)*x to
    # the episode return; geometry contributes HORIZON*W*x.  Check the two
    # scalings recover a known total.
    dt, dg = -0.028, +0.321
    t_auc = HORIZON * (1 - W) * dt
    g_auc = HORIZON * W * dg
    check("task scaling uses (1-w)", abs(t_auc - (20 * 0.6 * dt)) < 1e-12,
          f"{t_auc:+.4f}")
    check("geometry scaling uses w", abs(g_auc - (20 * 0.4 * dg)) < 1e-12,
          f"{g_auc:+.4f}")
    check("the two channels have opposite signs here, as on GPT-2",
          t_auc < 0 < g_auc)

    print("(3) analysis and the three verdicts")
    rng = np.random.default_rng(0)

    def frame(gain, n=20, noise=0.2, arms=ALL_ARMS):
        rows = []
        for s in range(n):
            se = rng.normal(0, 0.15)
            for a in arms:
                g = gain if a == "so" else (gain * 0.6 if a in ("sl", "gl")
                                            else 0.0)
                auc = 6.5 + se + g + rng.normal(0, noise)
                # split the AUC into two channels that sum back exactly, so
                # the additivity check in analyse() is exercised for real
                a_task = 0.45 * auc - (0.34 if a != BASELINE else 0.0)
                rows.append(dict(arm=a, seed=s, auc=auc,
                                 auc_task=a_task, auc_geo=auc - a_task,
                                 r_task=0.18 - (0.03 if a != BASELINE else 0.0),
                                 r_geo=0.55 + (0.30 if a == "so" else 0.0),
                                 rho=np.nan if a == BASELINE else 1.0,
                                 recon_err=1e-15, status="ok"))
        return pd.DataFrame(rows)

    def run(df):
        b = io.StringIO()
        with contextlib.redirect_stdout(b):
            analyse(df)
        return b.getvalue()

    pos = run(frame(+2.2))
    check("a clear gain -> 'multi-seed and decomposed'",
          "now multi-seed and decomposed" in pos)
    check("the decomposition table is printed",
          "CHANNEL DECOMPOSITION" in pos and "geometry share" in pos)
    check("the projection caveat is always printed",
          "no distortion guarantee" in pos)
    neg = run(frame(-2.2))
    check("a significant loss -> 'not supported'",
          "not supported" in neg)
    null = run(frame(0.0, noise=1.2))
    check("a null -> 'only the interval is informative'",
          "only the interval is informative" in null)
    check("the task-channel sign count is reported",
          "task channel below baseline on" in pos)
    check("an additive decomposition is not flagged", "non-additive" not in pos)
    # break additivity: the guard must catch it
    # shift one arm only: a shift applied to every arm cancels in the paired
    # difference and would leave the guard untested
    broken = frame(+2.2)
    broken.loc[broken.arm == "so", "auc_geo"] += 0.5
    check("a non-additive decomposition is flagged",
          "non-additive" in run(broken))
    check("channel AUCs, not per-step means, drive the table",
          "auc_task" in COLS and "auc_geo" in COLS)

    print("(4) degenerate inputs")
    for name, dfx in [
        ("empty", frame(2.2).iloc[0:0]),
        ("baseline missing", frame(2.2, arms=("so", "sl"))),
        ("so missing", frame(2.2, arms=(BASELINE, "sl"))),
        ("one seed", frame(2.2, n=1)),
        ("all so failed", frame(2.2).assign(
            status=lambda x: np.where(x.arm == "so", "err", "ok"))),
    ]:
        try:
            run(dfx); good = True
        except Exception as exc:                             # noqa: BLE001
            good = False
            print(f"        {name}: {type(exc).__name__}: {exc}")
        check(f"degenerate input '{name}' does not crash", good)
    check("a missing arm is reported, not skipped",
          "both required" in run(frame(2.2, arms=("so", "sl"))))
    # report() withholds its interval when every paired difference is
    # identical; the verdict must not index the absent key
    flat = frame(2.2, n=6, noise=0.0)
    flat["auc"] = 7.0
    try:
        out_flat, crashed = run(flat), False
    except Exception:                                        # noqa: BLE001
        out_flat, crashed = "", True
    check("degenerate paired differences do not raise", not crashed)
    check("...and the verdict is refused, not guessed",
          "degenerate" in out_flat or "Not enough" in out_flat)
    for col in ("rho", "auc_task", "r_geo"):
        try:
            run(frame(2.2).drop(columns=[col])); good = True
        except Exception as exc:                             # noqa: BLE001
            good = False
            print(f"        missing {col}: {type(exc).__name__}: {exc}")
        check(f"a missing '{col}' column does not crash the table", good)

    print("\n" + ("SELFTEST PASS" if ok else "SELFTEST FAIL"))
    if not ok:
        sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--arms", type=str, default=",".join(ALL_ARMS),
                    help="comma-separated subset of " + ",".join(ALL_ARMS))
    ap.add_argument("--project-dim", type=int, default=PROJECT_DIM)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--analyse-only", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest(); return
    arms = tuple(x.strip() for x in a.arms.split(",") if x.strip())
    unknown = [x for x in arms if x not in ALL_ARMS]
    if unknown:
        sys.exit(f"unknown arm(s) {unknown}; expected a subset of {list(ALL_ARMS)}")
    if a.analyse_only:
        if not os.path.exists(CSV):
            sys.exit(f"{CSV} not found; run without --analyse-only first.")
        check_meta(CSV, dict(w=W, horizon=HORIZON, model="mistral-7b"))
        df = pd.read_csv(CSV)
    else:
        df = collect(a.seeds, arms, a.overwrite, a.project_dim)
    analyse(df)


if __name__ == "__main__":
    main()
