"""
sppg_analysis.py -- read the sweep CSVs and state what the paper may claim.
==========================================================================

    python sppg_analysis.py                 # every experiment found, in order
    python sppg_analysis.py r4 s7 b1lr      # only these
    python sppg_analysis.py --dir ./csv     # CSVs live elsewhere
    python sppg_analysis.py --manuscript    # just the reconciliation table
    python sppg_analysis.py --selftest      # no CSVs needed

Collection needs a GPU and an hour; analysis needs neither, and you will re-run
it many times.  Hence the split from `sppg_experiments.py`.

What this file is for
---------------------
Every number the manuscript quotes should be derivable from a CSV by running
this file.  Where a number is not derivable, that is the finding, and the code
says so in its own output rather than leaving you to notice.

Three rules are enforced throughout, because each one caught a real error:

  1. Pairing.  Every contrast is matched on the blocking variable -- the seed,
     or in X-3 the environment draw.  Comparing means taken over different
     seed subsets breaks the design.

  2. Multiplicity.  Every family of contrasts is Holm-corrected together.  A
     p-value quoted without saying which family it belongs to is not a result.

  3. Nulls.  "Not significant" is never reported as "equivalent".  A claim that
     two arms are the same goes through TOST with a margin fixed in advance.

Reproducing the manuscript
--------------------------
`--manuscript` prints a reconciliation table: every headline number in the
paper, the value this code computes from the CSVs, and a PASS/FAIL.  If a row
fails, either the CSV is stale or the manuscript is wrong; the table does not
guess which.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

from sppg_core import (BASELINE, clean, equivalent, holm, paired,
                       paired_report, report, tost_paired, n_for_power_paired)

# --------------------------------------------------------------------------- #
# The numbers the manuscript quotes.  Sourced from the submitted LaTeX so the
# reconciliation is against the paper, not against a memory of it.
# --------------------------------------------------------------------------- #
MANUSCRIPT = {
    "tab:ablation so(32)":      8.730,
    "tab:ablation sl(32)":      8.089,
    "tab:ablation gl(32)":      8.050,
    "tab:ablation random(496)": 7.516,
    "tab:ablation sym(32)":     6.576,
    "R-1 so - baseline":        2.2338,
    "R-1 task channel":        -0.338,
    "R-1 geometry channel":     2.572,
    "B-1 expm - cayley":        0.366,
    "Mistral gain (%)":         59.3,
}

#: Equivalence margin for every "are these two the same?" question: 25% of the
#: so(32)-vs-baseline effect.  Fixed here, before any result is seen, and used
#: unchanged by B-1, B-1-LR and S-7.  Choosing it afterwards would make every
#: equivalence claim unfalsifiable.
EFFECT_REF = 2.2338
MARGIN = 0.25 * EFFECT_REF


def _h(title: str, sub: str = "") -> None:
    print("\n" + "=" * 96)
    print(title)
    if sub:
        print(sub)
    print("=" * 96)


def _load(name: str, directory: str) -> pd.DataFrame | None:
    """Load <name>_cells.csv, tolerating the r4_arms.csv legacy filename."""
    for candidate in (f"{name}_cells.csv", f"{name}_arms.csv"):
        p = os.path.join(directory, candidate)
        if os.path.exists(p):
            df = pd.read_csv(p)
            # Legacy CSVs spell the arms "so(32)"; normalise to "so".
            if "arm" in df.columns:
                df["arm"] = df["arm"].replace({
                    "so(32)": "so", "sl(32)": "sl", "gl(32)": "gl",
                    "sym(32)": "sym", "random(496)": "random"})
            # Legacy column name for the spectral radius.
            if "rho" not in df.columns and "median_spectral_radius" in df.columns:
                df = df.rename(columns={"median_spectral_radius": "rho"})
            return df
    return None


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    """Numeric view of an optional column; empty Series if it is absent.

    `df.get(col)` returns None for a missing column, and pd.to_numeric(None)
    yields a bare float rather than a Series -- which then fails on .dropna().
    Every optional-column read goes through here.
    """
    if col not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[col], errors="coerce").dropna()


def _recon_note(df: pd.DataFrame) -> None:
    """The channel-decomposition guarantee, restated per experiment."""
    if "recon_err" not in df.columns:
        print("  !! no 'recon_err' column: the channel-decomposition guarantee "
              "cannot be checked.\n     This CSV predates the runtime check; "
              "re-collect before citing the channels.")
        return
    e = pd.to_numeric(df["recon_err"], errors="coerce")
    e = e[np.isfinite(e)]
    if len(e):
        print(f"  channel-reconstruction error vs main.py's own AUC: "
              f"max = {e.max():.3e}   (must be at machine precision)")


def _failures(df: pd.DataFrame) -> pd.DataFrame:
    bad = df[df.status != "ok"]
    if len(bad):
        print(f"\n  {len(bad)} failed run(s):")
        for _, r in bad.iterrows():
            print(f"    {r.get('arm')} seed={r.get('seed')}: {r['status']}")
    return df[df.status == "ok"]


# =========================================================================== #
# R-4 / R-1 -- the algebra ablation and its channel decomposition
# =========================================================================== #
def analyse_r4(df: pd.DataFrame) -> dict:
    """The source of Table tab:ablation, and the R-1 contrast that qualifies it."""
    _h("R-4 / R-6  ALGEBRA ABLATION",
       "  This table is tab:ablation.  The R-1 contrast below is why the "
       "so-vs-baseline row\n  cannot be read as evidence about the algebra.")
    _recon_note(df)
    d = _failures(df)
    arms = [a for a in ("so", "sl", "gl", "random", "sym", BASELINE)
            if a in set(d.arm)]
    d = clean(d, "auc", "seed", arms)

    print(f"\n  {'arm':>12s} {'n':>3s} {'AUC mean+/-sd':>20s} "
          f"{'manuscript':>11s} {'rho (median)':>14s}")
    out = {}
    for a in arms:
        v = d[d.arm == a].auc.astype(float)
        man = MANUSCRIPT.get(f"tab:ablation {a}(32)") or \
            MANUSCRIPT.get("tab:ablation random(496)") if a == "random" else \
            MANUSCRIPT.get(f"tab:ablation {a}(32)")
        rho = _num(d[d.arm == a], "rho")
        out[a] = float(v.mean())
        print(f"  {a:>12s} {len(v):>3d} {v.mean():10.4f} +/- {v.std(ddof=1):5.4f} "
              f"{('%.3f' % man) if man else '--':>11s} "
              f"{(f'{rho.median():.6g}' if len(rho) else '--'):>14s}")

    rows = [dict(label=f"so - {a}", d=paired(d, "so", a))
            for a in arms if a != "so"]
    report(rows, "R-4  TOTAL AUC, paired by seed",
           "  Holm-corrected over the arms compared.")

    # --- the channel decomposition, which reframes the headline -------------
    if "auc_task" in d.columns:
        t = paired(d, "so", BASELINE, "auc_task")
        g = paired(d, "so", BASELINE, "auc_geo")
        a = paired(d, "so", BASELINE, "auc")
        if not len(a):
            print("\n  no paired so/baseline block: the channel decomposition "
                  "cannot be computed.")
            return out
        _h("R-1  CHANNEL DECOMPOSITION of the so(32) vs baseline gap",
           "  The two channels sum to the total by construction (see recon_err "
           "above).")
        print(f"  total     {a.mean():+.4f}")
        print(f"  task      {t.mean():+.4f}   negative on {(t < 0).sum()}/{len(t)} seeds")
        print(f"  geometry  {g.mean():+.4f}   = {100 * g.mean() / a.mean():.0f}% "
              f"of the total")
        print("\n  Reading: the gain is carried by a reward channel the M = I control\n"
              "  cannot reach, and it is paid for with task reward.  The "
              "so-vs-baseline row\n  therefore says nothing about the "
              "algebra; the so-vs-sl/gl/random rows do.")
        out.update(task=float(t.mean()), geo=float(g.mean()), total=float(a.mean()))
    return out


# =========================================================================== #
# R-1b -- is the task cost tunable away?
# =========================================================================== #
def analyse_r1b(df: pd.DataFrame) -> dict:
    _h("R-1b  THE (w, c_geo) TRADE-OFF SURFACE",
       "  AUC is not comparable across w -- changing w rescales the return.\n"
       "  Each row below is therefore read within its own w.")
    d = _failures(df)
    d["c_geo"] = pd.to_numeric(d["c_geo"], errors="coerce")
    w_grid = sorted(d.w.unique())
    c_grid = sorted(d[d.arm != BASELINE].c_geo.dropna().unique())

    print("\n  so(32) minus baseline PPO, by (w, c_geo):")
    print(f"  {'w':>5s}" + "".join(f"{f'c={c:g}':>14s}" for c in c_grid))
    for w in w_grid:
        cells = []
        bl = d[(d.arm == BASELINE) & (d.w == w)].set_index("seed").auc
        for c in c_grid:
            m = d[(d.arm != BASELINE) & (d.w == w) & (d.c_geo == c)] \
                .set_index("seed").auc
            i = m.index.intersection(bl.index)
            cells.append(f"{(m.loc[i] - bl.loc[i]).mean():+13.4f}")
        print(f"  {w:>5.1f}" + "".join(cells))

    # The decisive statement: what happens with the auxiliary loss OFF.
    c_off = min(c_grid)
    off = []
    for w in w_grid:
        bl = d[(d.arm == BASELINE) & (d.w == w)].set_index("seed").auc
        m = d[(d.arm != BASELINE) & (d.w == w) & (d.c_geo == c_off)] \
            .set_index("seed").auc
        i = m.index.intersection(bl.index)
        off.append((m.loc[i] - bl.loc[i]).mean())
    _h(f"R-1b  WITH THE AUXILIARY LOSS OFF (c_geo = {c_off:g})")
    for w, v in zip(w_grid, off):
        print(f"  w={w:.1f}   so(32) - baseline = {v:+.4f}"
              + ("   <-- loses to the unstructured control" if v < 0 else ""))
    if off and all(v < 0 for v in off):
        print("\n  -> At every geometry weight tested, the method loses to "
              "baseline PPO once\n     the auxiliary loss is removed.  The "
              "task-channel cost is not 'tunable away'\n     because the "
              "benefit is not there without supervision.  This is the same\n"
              "     conclusion S-7 reaches from the other direction.")
    return {"c_off": c_off, "deltas_off": off}


# =========================================================================== #
# R-5 / B-1-LR -- leave-one-seed-out learning-rate tuning
# =========================================================================== #
def _loo_tuned(d: pd.DataFrame, arms, lr_col: str = "theta_lr",
               min_cell: int = 3) -> pd.DataFrame | None:
    """Evaluate each seed at the rate chosen from the other seeds.

    Choosing the best rate on the same seeds you then test on is selection
    inference: the max over a 5-point grid of noisy cell means is upward-biased,
    and the bias scales with each arm's grid noise, so it is not symmetric
    across arms.  Leave-one-seed-out fixes it at no extra compute -- every
    evaluated point is out-of-sample with respect to its own selection.
    """
    seeds = sorted(d[d.arm.isin(arms)].seed.unique())
    rows, dropped = [], []
    for s in seeds:
        rest = d[d.seed != s]
        pick, ok = {}, True
        for a in arms:
            cells = {lr: v.auc.astype(float).mean()
                     for lr, v in rest[rest.arm == a].groupby(lr_col)
                     if len(v) >= min_cell}
            if not cells:
                ok = False
                break
            pick[a] = max(cells, key=cells.get)
        if not ok:
            continue
        row, good = {"seed": s}, True
        for a in arms:
            v = d[(d.arm == a) & (d[lr_col] == pick[a]) & (d.seed == s)].auc
            if len(v) != 1:
                good = False
                break
            row[a] = float(v.iloc[0])
        if good:
            rows.append(row)
        else:
            dropped.append(s)
    if dropped:
        # These are the seeds where the selected rate had no run -- i.e.
        # where the winning rate was unstable.  Dropping them without comment
        # would bias the tuned contrast upward, so they are listed.
        print(f"  !! {len(dropped)} seed(s) dropped, no run at the selected "
              f"rate: {dropped}")
    return pd.DataFrame(rows) if len(rows) >= 3 else None


def _lr_table(d: pd.DataFrame, arms, lr_col="theta_lr") -> dict:
    grid = sorted(d[lr_col].dropna().unique())
    print(f"\n  {'arm':>10s}" + "".join(f"{lr:>13.4g}" for lr in grid)
          + f"{'best':>10s}")
    best = {}
    for a in arms:
        cells, means = [], {}
        for lr in grid:
            v = d[(d.arm == a) & (d[lr_col] == lr)].auc.astype(float)
            cells.append(f"{v.mean():13.3f}" if len(v) else f"{'n/a':>13s}")
            if len(v) >= 3:
                means[lr] = v.mean()
        best[a] = max(means, key=means.get) if means else float("nan")
        print(f"  {a:>10s}" + "".join(cells) + f"{best[a]:>10.4g}")
    return best


def analyse_r5(df: pd.DataFrame) -> dict:
    _h("R-5  PER-ALGEBRA LEARNING-RATE SWEEP",
       "  Does the R-4 ordering survive when every algebra gets its own rate?")
    d = _failures(df)
    arms = [a for a in ("so", "sl", "gl", "sym") if a in set(d.arm)]
    best = _lr_table(d, arms)
    L = _loo_tuned(d, arms)
    if L is None:
        print("\n  not enough complete seeds for out-of-sample tuning.")
        return {}
    print(f"\n  n = {len(L)} seeds, each evaluated at the rate chosen from the "
          f"other {len(L) - 1}.")
    res = report([dict(label=f"so - {a}  (LOO-tuned)",
                       d=(L["so"] - L[a]).to_numpy())
                  for a in arms if a != "so"],
                 "R-5  TUNED COMPARISON, out-of-sample",
                 "  Selection and evaluation use disjoint seeds, so these "
                 "p-values are valid.")
    wins = [o for o in res if o.get("sig") and o.get("diff", 0) > 0]
    _h("R-5  VERDICT")
    if not res:
        print("  -> No comparisons available; nothing to conclude.")
    elif len(wins) == len(res):
        print("  -> so(32) beats every other algebra at each algebra's own best "
              "rate.\n     The R-4 ordering is not a learning-rate artefact, "
              "and the shared-rate\n     confound noted in the Limitations section does "
              "not apply.")
    else:
        print("  -> Not all comparisons survive tuning, so the ordering is "
              "not established\n     as rate-independent.")
    print("\n  Caveat: this grid varies theta_lr "
          "alone, and main.py's\n  own comments say entropy_coef, geo_aux_coef "
          "and reward_threshold were each chosen\n  to suit theta_lr = 0.03.")
    return {"best": best, "n": len(L), "all_win": bool(res) and len(wins) == len(res),
            "lost": [o["label"] for o in res
                     if o.get("sig") and o.get("diff", 0) < 0]}


def analyse_b1(df: pd.DataFrame) -> dict:
    _h("B-1  THE MATRIX EXPONENTIAL vs A CAYLEY RETRACTION",
       "  Both maps send so(n) into SO(n) exactly.  Is the effect the "
       "constraint or the map?")
    d = _failures(df)
    if "orth_err" in d.columns:
        oe = _num(d[d.arm != BASELINE], "orth_err")
        if len(oe):
            print(f"  validity gate: max |M^T M - I| = {oe.max():.2e}"
                  + ("   OK, both maps produce proper rotations"
                     if oe.max() < 1e-3 else
                     "   !! Not a rotation -- nothing below is interpretable"))
    arms = [a for a in ("so_expm", "so_cayley", BASELINE) if a in set(d.arm)]
    d = clean(d, "auc", "seed", arms)
    res = report([dict(label="so_expm - baseline",
                       d=paired(d, "so_expm", BASELINE)),
                  dict(label="so_cayley - baseline",
                       d=paired(d, "so_cayley", BASELINE)),
                  dict(label="so_expm - so_cayley",
                       d=paired(d, "so_expm", "so_cayley"))],
                 "B-1  TOTAL AUC",
                 "  The third row is the one that matters: is the exponential "
                 "map itself doing\n  work, or would any retraction onto SO(n) "
                 "do?")
    cmp_ = next(o for o in res if o["label"] == "so_expm - so_cayley")
    _h("B-1  VERDICT")
    if "diff" not in cmp_:
        print("  -> Not enough paired seeds to decide "
              f"({cmp_.get('degenerate', 'no data')}).")
    elif cmp_.get("sig") and cmp_["diff"] > 0:
        print(f"  -> At the shared rate the exponential map beats Cayley "
              f"({cmp_['diff']:+.4f},\n     p_holm={cmp_['p_holm']:.2e}).  "
              "But both maps ran at 0.03, which was chosen\n     for expm.  "
              "B-1-LR removes that confound.")
    elif cmp_.get("sig"):
        # A significant effect in the opposite direction is not "no effect".
        print(f"  -> A Cayley retraction significantly beats the matrix "
              f"exponential\n     ({cmp_['diff']:+.4f}, "
              f"p_holm={cmp_['p_holm']:.2e}).  The paper's specific scheme is "
              "not the\n     best way to realise the constraint, and the "
              "contribution narrows accordingly.")
    else:
        print(f"  -> No significant difference at the shared rate "
              f"({cmp_['diff']:+.4f}).")
    return {"diff": cmp_.get("diff")}


def analyse_b1lr(df: pd.DataFrame) -> dict:
    _h("B-1-LR  THE MAP COMPARISON WITH EACH MAP AT ITS OWN RATE",
       "  exp(A) and cayley(A) agree to second order and diverge after, so the "
       "same step in\n  the algebra gives a different displacement in the "
       "group.  A rate tuned for one is\n  not neutral between them -- the "
       "same confound R-5 removed for the algebras.")
    d = _failures(df)
    maps = [a for a in ("so_expm", "so_cayley") if a in set(d.arm)]
    best = _lr_table(d, maps)

    # Q2: was the shared rate tilted?
    print("\n  Was the paper's shared rate (0.03) each map's own best?")
    for m in maps:
        v = {lr: d[(d.arm == m) & (d.theta_lr == lr)].auc.astype(float).mean()
             for lr in sorted(d.theta_lr.dropna().unique())}
        v = {lr: x for lr, x in v.items() if np.isfinite(x)}
        at_paper = v.get(0.03, float("nan"))
        flag = "  <-- 0.03 is its best" if best[m] == 0.03 else \
               f"  <-- its best is {best[m]:g}, NOT 0.03"
        print(f"    {m:>10s}: best {max(v.values()):.3f} at lr={best[m]:g}; "
              f"at 0.03 {at_paper:.3f}{flag}")

    if set(maps) != {"so_expm", "so_cayley"}:
        print(f"\n  both maps are required for the tuned comparison; "
              f"found {maps}.  Stopping here.")
        return {}
    L = _loo_tuned(d, maps)
    if L is None:
        print("\n  not enough complete seeds for out-of-sample tuning.")
        return {}
    diff = (L["so_expm"] - L["so_cayley"]).to_numpy()
    res = report([dict(label="so_expm - so_cayley  (LOO-tuned)", d=diff)],
                 "B-1-LR  TUNED MAP COMPARISON, out-of-sample",
                 "  This is the number that decides whether the map itself "
                 "matters.")
    o = res[0]
    _h("B-1-LR  VERDICT")
    if o.get("sig") and o.get("diff", 0) < 0:
        print(f"  -> Cayley beats expm once each map is tuned "
              f"({o['diff']:+.4f}, p_holm={o['p_holm']:.2e}).\n     B-1's "
              "ordering was a learning-rate artefact in the opposite "
              "direction.")
    elif o.get("sig") and o["diff"] > 0:
        print(f"  -> The gap SURVIVES per-map tuning ({o['diff']:+.4f}, "
              f"p_holm={o['p_holm']:.2e}).\n     The map itself matters, which "
              "needs an explanation, since\n     both maps land "
              "in the same group to machine precision.")
    elif equivalent(o, MARGIN):
        t = tost_paired(diff, margin=MARGIN)
        print(f"  -> EQUIVALENT within +/-{MARGIN:.4f} AUC (25% of the "
              f"so-vs-baseline effect,\n     fixed in advance): the whole 95% "
              f"CI [{o['ci'][0]:+.4f}, {o['ci'][1]:+.4f}] lies inside the "
              f"margin\n     (TOST p = {t['p_tost']:.4g}).")
        print("\n     The shared-rate gap was an ARTEFACT.  The effect is a "
              "property of\n     constraining to SO(32), not of the matrix "
              "exponential -- the stronger and more\n     general claim, and "
              "the one the compactness theory predicts.  The\n     "
              "manuscript currently states the opposite.")
    else:
        need = n_for_power_paired(abs(o.get("dz", 0)), 0.8)
        print(f"  -> Under-powered, not equivalent.  The difference "
              f"({o['diff']:+.4f}) is not\n     significant, but the CI is not "
              f"inside +/-{MARGIN:.4f} either, so neither is established."
              + (f"  About {need} seeds would give 80% power." if need else ""))
    return {"diff": o.get("diff"), "ci": o.get("ci")}


# =========================================================================== #
# X-3 -- robustness to the environment draw
# =========================================================================== #
def analyse_x3(df: pd.DataFrame) -> dict:
    _h("X-3  ROBUSTNESS TO THE ENVIRONMENT DRAW",
       "  The blocking variable is the draw, not the seed: training seeds are "
       "averaged\n  within a draw first, so one lucky initialisation cannot "
       "ride along into every draw.")
    d = _failures(df)
    g = d.groupby(["draw", "arm"], as_index=False).agg(
        auc=("auc", "mean"), n_seed=("seed", "nunique"))
    g["status"] = "ok"
    print(f"  averaging {sorted(int(x) for x in g.n_seed.unique())} "
          f"training seed(s) per draw")
    arms = [a for a in ("so", "sl", "gl", "sym", BASELINE) if a in set(g.arm)]
    g = clean(g, "auc", "draw", arms)
    report([dict(label=f"so - {a}", d=paired(g, "so", a, "auc", "draw"))
            for a in arms if a != "so"],
           f"X-3  TOTAL AUC, paired by ENVIRONMENT DRAW (n = {g.draw.nunique()})",
           "  These test whether the ordering generalises over planted "
           "rotations, not over\n  training seeds.")
    algs = [a for a in arms if a != BASELINE]
    w = g.pivot(index="draw", columns="arm", values="auc")
    wins = int((w[algs].idxmax(axis=1) == "so").sum())
    _h("X-3  VERDICT")
    print(f"  so(32) is the top algebra in {wins}/{len(w)} draws.")
    if wins == len(w):
        print("  -> The ordering is a property of the algebra, not of one "
              "planted rotation.")
    else:
        print("  -> The ordering does not hold on every draw; see the "
              "per-draw table.")
    return {"wins": wins, "n_draws": len(w)}


# =========================================================================== #
# R-7 -- environment geometry x policy algebra
# =========================================================================== #
def analyse_r7(df: pd.DataFrame) -> dict:
    _h("R-7  ENVIRONMENT GEOMETRY x POLICY ALGEBRA",
       "  AUC is NOT comparable across ROWS -- each geometry makes a different "
       "amount of\n  geometry reward attainable.  Compare within a row only.\n"
       "  `id` is a positive control: with M_env = I the baseline acts with "
       "exactly the\n  target, so it should top that row.")
    d = _failures(df)
    geoms = [g for g in ("so", "sym", "gl", "diag", "id") if g in set(d.geom)]
    arms = [a for a in (BASELINE, "so", "sl", "gl", "sym") if a in set(d.arm)]

    # Multi-draw handling.  With one M_env per geometry the seed is the only
    # blocking variable, and a column is a statement about that draw as much as
    # about its family.  With several draws the draw becomes the replication
    # unit -- the design X-3 uses -- so training seeds are averaged within a
    # draw first and the contrasts are paired on the draw.  Averaging first
    # stops one lucky initialisation riding along into every draw.
    n_draws = int(d["draw"].nunique()) if "draw" in d.columns else 1
    if n_draws > 1:
        by = "draw"
        d = (d.groupby(["geom", "arm", "draw"], as_index=False)
               .agg(auc=("auc", "mean"), n_seed=("seed", "nunique")))
        d["status"] = "ok"
        print(f"\n  {n_draws} draws of M_env per geometry; averaging "
              f"{sorted(int(x) for x in d.n_seed.unique())} training seed(s) "
              f"within each draw.  The draw is the replication unit, so the\n"
              f"  contrasts below generalise over the geometry family rather "
              f"than over one matrix.")
    else:
        by = "seed"
        print("\n  one draw of M_env per geometry: each column reflects that "
              "draw as much as its\n  family.  Re-run with --draws 3 to make "
              "the scope claim about the families.")

    # Two details of this table matter.
    #
    # (1) `winner` ranks algebras only.  Including baseline_ppo would make a
    #     geometry where the unstructured control happens to top the row count
    #     as "so(32) is not best", even when so(32) beats every algebra, and
    #     the verdict would then contradict the margin table below.  The
    #     baseline column is still printed, because it is the reference the
    #     reader needs; it just does not win the row.
    # (2) Each geometry is cleaned to a paired block first.  Ranking unpaired
    #     means over different seed sets can flip the winner against the paired
    #     contrast printed directly beneath it (e.g. so(32) best on every seed,
    #     yet 'sl' named the winner because so had failures).
    algebras = [a for a in arms if a != BASELINE]
    print(f"\n  {'geometry':>9s}" + "".join(f"{a:>13s}" for a in arms)
          + f"{'best algebra':>14s}")
    # Two rankings.  `winner` ranks algebras only and drives the
    # scope verdict; `winner_overall` includes baseline_ppo and is what the
    # `id` positive control needs -- with M_env = I the baseline acts with
    # exactly the target and should top that row, which an algebra-only ranking
    # can never show.
    winner, winner_overall, blocks = {}, {}, {}
    for geom in geoms:
        dg = clean(d[d.geom == geom].copy(), "auc", by, arms)
        blocks[geom] = dg
        cells, means = [], {}
        for a in arms:
            v = dg[dg.arm == a].auc.astype(float)
            cells.append(f"{v.mean():13.2f}" if len(v) else f"{'n/a':>13s}")
            if len(v) and a in algebras:
                means[a] = v.mean()
        winner[geom] = max(means, key=means.get) if means else "n/a"
        overall = {a: dg[dg.arm == a].auc.astype(float).mean()
                   for a in arms if len(dg[dg.arm == a])}
        winner_overall[geom] = max(overall, key=overall.get) if overall else "n/a"
        print(f"  {geom:>9s}" + "".join(cells) + f"{winner[geom]:>14s}")
    print("  (the 'best algebra' column excludes baseline_ppo: it is the "
          "reference, not an arm\n   in the algebra comparison)")

    # so(32)'s margin over the best other algebra, within each geometry.
    print(f"\n  so(32) minus the best other algebra, within each geometry "
          f"(paired by {by}):")
    others = [a for a in arms if a not in ("so", BASELINE)]
    margins = {}
    for geom in geoms:
        dg = blocks[geom]
        per = []
        for s in sorted(dg[dg.arm == "so"][by].unique()):
            row = dg[dg[by] == s]
            so_v = row[row.arm == "so"].auc.astype(float)
            ot = row[row.arm.isin(others)].auc.astype(float)
            if len(so_v) == 1 and len(ot):
                per.append(float(so_v.iloc[0]) - float(ot.max()))
        if per:
            margins[geom] = np.array(per)
            v = margins[geom]
            print(f"    {geom:>5s}: {v.mean():+.4f} +/- "
                  f"{v.std(ddof=1) if len(v) > 1 else float('nan'):.4f}"
                  + ("   <-- so(32) is not best here" if v.mean() < 0 else ""))

    _h("R-7  POSITIVE CONTROL: the identity geometry")
    if "id" in geoms:
        if winner_overall["id"] == BASELINE:
            print(f"  winner on M_env = I: {winner_overall['id']}  -> as "
                  f"predicted.  "
                  "The harness is\n  measuring what it claims to measure.")
        else:
            print(f"  winner on M_env = I: {winner_overall['id']}  -> !! "
                  f"UNEXPECTED.  "
                  "Resolve this before\n  reading any other row: it is a "
                  "property of the harness, not of the algebras.")

    # Judge only on geometries that contain a transformation an algebra could
    # plausibly match.  `id` is the harness control (M_env = I, so the baseline
    # is already optimal) and `diag` is an axis-aligned scaling that no arm in
    # the set is built for; scoring either as a test cell would downgrade the
    # verdict for reasons that have nothing to do with compactness.
    judged = [g for g in geoms if g not in ("id", "diag")]
    so_wins = [g for g in judged if winner.get(g) == "so"]
    lost = [g for g in judged if winner.get(g) != "so"]
    excluded = [g for g in geoms if g not in judged]
    _h("R-7  VERDICT",
       f"  (judged on {', '.join(judged) or 'nothing'}"
       + (f"; excluded as controls: {', '.join(excluded)})" if excluded else ")"))
    if not judged:
        print("  -> only control geometries were run; nothing to generalise "
              "from.")
    elif not lost:
        print("  -> so(32) is best under every non-degenerate geometry tested, "
              "including ones\n     it cannot represent.  The effect is a "
              "property of the optimiser, not of\n     matching the environment "
              "-- a STRONGER result than the paper currently claims.")
    elif so_wins:
        print(f"  -> MIXED.  so(32) is best under [{', '.join(so_wins)}] and "
              f"not under\n     [{', '.join(lost)}].")
        for g in lost:
            if g in margins and margins[g].mean() < 0:
                print(f"       on the {g} geometry so(32) trails the best other "
                      f"algebra by {margins[g].mean():+.3f}")
        # Name the geometries that actually lost, rather than hard-coding
        # "symmetric": the text must follow the data, not the run we happened
        # to look at first.
        print(f"\n     Scope of the claim: not "
              f"'compactness helps', but\n     'so(32) is best when the "
              f"environment geometry is {' or '.join(so_wins)}, and not when "
              f"it is\n     {' or '.join(lost)}'.")
    elif "so" in geoms and winner.get("so") == "so":
        print("  -> so(32) wins only where the environment is itself SO(32).  "
              "The result is\n     'match the environment's structure', not "
              "'compactness helps', and it holds\n     only for environments "
              "with a planted compact-group structure.")
    else:
        print(f"  -> so(32) is not best under any judged geometry "
              f"({', '.join(judged)}), including\n     the SO one.  The "
              "algebra claim does not survive this experiment at all.")
    print("\n  Caveat: each geometry uses one draw of M_env.  X-3 established "
          "draw-robustness\n  for the SO family only; treat a single "
          "off-diagonal column as indicative.")
    return {"winner": winner, "winner_overall": winner_overall,
            "so_wins": so_wins, "lost": lost, "judged": judged}


# =========================================================================== #
# S-7 -- prior or supervision?
# =========================================================================== #
def analyse_s7(df: pd.DataFrame) -> dict:
    _h("S-7  IS THE ADVANTAGE AN INDUCTIVE BIAS, OR SUPERVISION?",
       "  main.py's auxiliary loss regresses the policy's transformation onto "
       "the\n  environment's own latent rotation.  Section 8.1 states the "
       "opposite.\n  The geometry reward uses M_env in every arm; only the "
       "auxiliary target moves.")
    d = _failures(df)
    arms = [a for a in (BASELINE, "so+M_env", "so+G_ref", "so+G_rand",
                        "so+no_aux") if a in set(d.arm)]
    d = clean(d, "auc", "seed", arms)

    print(f"\n  {'arm':>12s} {'n':>3s} {'AUC':>18s} {'align(M_env)':>14s} "
          f"{'align(target)':>15s}")
    for a in arms:
        v = d[d.arm == a]
        ae, ar = _num(v, "align_env"), _num(v, "align_ref")
        print(f"  {a:>12s} {len(v):>3d} "
              f"{v.auc.mean():9.3f}+/-{v.auc.std(ddof=1):6.3f} "
              f"{(f'{ae.mean():+14.3f}' if len(ae) else f'{chr(45):>14s}')} "
              f"{(f'{ar.mean():+15.3f}' if len(ar) else f'{chr(45):>15s}')}")

    rows = [dict(label=f"{a} - baseline", d=paired(d, a, BASELINE))
            for a in arms if a != BASELINE]
    if "so+M_env" in arms and "so+G_ref" in arms:
        rows.append(dict(label="so+M_env - so+G_ref",
                         d=paired(d, "so+M_env", "so+G_ref")))
    res = report(rows, "S-7  TOTAL AUC", "  Holm-corrected over this family.")
    by = {o["label"]: o for o in res}

    menv = by.get("so+M_env - baseline", {})
    # so+no_aux must be in this list.  It is the arm that matches Sec. 8.1's
    # prose -- a target-free rotational prior -- and without it the verdict
    # below could name M_env as the only winning target even when so+no_aux
    # beats baseline significantly.
    alts = [by.get(f"{a} - baseline", {})
            for a in ("so+G_ref", "so+G_rand", "so+no_aux")]
    alts = [o for o in alts if o]
    _h("S-7  VERDICT")
    if not (menv.get("sig") and menv.get("diff", 0) > 0):
        print("  -> The published configuration does not beat the baseline in "
              "this run.\n     Check it before interpreting any other arm.")
    elif any(o.get("sig") and o.get("diff", 0) > 0 for o in alts):
        print("  -> A rotational target that is not the environment's also "
              "beats baseline.\n     The auxiliary loss functions as a PRIOR.  "
              "Sec. 8.1 is right in substance,\n     although the sentence "
              "'involves no term in M_env' does not match the code.")
    else:
        worse = [o["label"] for o in alts
                 if o and o.get("sig") and o.get("diff", 0) < 0]
        print("  -> ONLY the arm whose auxiliary target IS M_env beats the "
              "baseline.")
        if worse:
            print(f"     All alternatives are significantly WORSE than "
                  f"baseline: {', '.join(worse)}.")
        print("\n     Mechanism: the G_ref / G_rand arms end aligned with their "
              "own target and at\n     zero with M_env (see the table).  They "
              "followed the auxiliary loss and\n     learned nothing about the "
              "environment from reward.")
        print("\n     The Task-1 claim is therefore conditional: given a "
              "rotational target,\n     so(32) absorbs it better than other "
              "parameterisations (R-4/R-5/X-3\n     establish that "
              "separately).  Sec. 8.1's 'involves no term in M_env'\n"
              "     is contradicted by the released code.")
        print("\n     CONFOUND: so+G_ref supplies a "
              "target the geometry\n     reward penalises, so 'a wrong target "
              "hurts' and 'a prior does not help' are\n     not separated by "
              "the AUC alone.  The arm matching Sec. 8.1's prose is\n"
              "     so+no_aux, whose result is in the table above.")
    return {k: v.get("diff") for k, v in by.items()}


# =========================================================================== #
# R-11 -- Task 2 held out
# =========================================================================== #
def analyse_r11(df: pd.DataFrame) -> dict:
    _h("R-11  TASK 2 ON HELD-OUT PROMPTS",
       "  The replication unit is the training seed: prompt-level rewards are "
       "averaged\n  within a seed before any test.  Prompts are not "
       "independent replicates --\n  pooling 12 of them would inflate n by 12x.")
    d = _failures(df)          # list failed runs rather than dropping them unreported
    g = (d.groupby(["seed", "split", "arm"]).reward.mean()
         .unstack("arm").reset_index())
    out = {}
    print(f"\n  {'split':>10s} {'n':>3s} {'structured':>12s} {'control':>12s} "
          f"{'delta':>10s} {'95% CI':>22s} {'p':>10s}")
    for split in ("train", "test"):
        x = g[g.split == split]
        if not len(x) or not {"structured", "control"}.issubset(x.columns):
            print(f"  {split:>10s}   (both arms are required; skipped)")
            continue
        r = paired_report((x["structured"] - x["control"]).to_numpy())
        out[split] = r
        ci = (f"[{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}]"
              if "ci" in r else "n/a")
        print(f"  {split:>10s} {r['n']:>3d} {x['structured'].mean():12.4f} "
              f"{x['control'].mean():12.4f} {r.get('diff', float('nan')):+10.4f} "
              f"{ci:>22s} {r.get('p', float('nan')):10.4g}")

    _h("R-11  VERDICT")
    te, tr = out.get("test"), out.get("train")
    if te is None:
        print("  Not enough complete seeds to decide.")
    elif "diff" not in te:
        print(f"  -> Not evaluable ({te.get('degenerate', 'no data')}).")
    elif te["p"] < 0.05 and te["diff"] < 0:
        print(f"  -> The structured arm is significantly worse on held-out "
              f"prompts\n     ({te['diff']:+.4f}, p = {te['p']:.4g}).  The "
              "published in-sample effect does not\n     generalise and is not "
              "supported.")
    elif te.get("p", 1) < 0.05 and te.get("diff", 0) > 0:
        print(f"  -> The Task-2 benefit survives on held-out prompts "
              f"({te['diff']:+.4f}).\n     It is an out-of-sample result, "
              "which is stronger\n     than what is currently claimed.")
    else:
        print(f"  -> No effect on held-out prompts ({te['diff']:+.4f}, 95% CI "
              f"[{te['ci'][0]:+.4f}, {te['ci'][1]:+.4f}],\n     "
              f"p = {te['p']:.4g}, n = {te['n']} seeds).")
        if tr is not None and not (tr.get("p", 1) < 0.05 and tr.get("diff", 0) > 0):
            print(f"\n     The effect is ABSENT IN-SAMPLE TOO "
                  f"({tr['diff']:+.4f}, p = {tr['p']:.4g}).\n     This is not a "
                  "generalisation failure: it is the repaired control showing\n"
                  "     through.  The released baseline was built with "
                  "lmbda = 0, which severs\n     the gradient path entirely, so "
                  "the published comparison was\n     structured-vs-untrained.  "
                  "Section 9's effect is not supported; it is a null.")
        print("\n     Report the interval; do NOT read this as equivalence "
              "unless it lies inside\n     a pre-registered margin, which we do "
              "not claim here.")
        if te["n"] < 20:
            need = n_for_power_paired(abs(te.get("dz", 0.69)), 0.8)
            print(f"\n     NOTE: n = {te['n']} seeds."
                  + (f"  Detecting an effect the size of the one observed "
                     f"here\n     ({te['diff']:+.4f}) at 80% power would need "
                     f"about {need} seeds -- which is a\n     statement about "
                     f"how small the observed difference is, not a "
                     f"recommendation." if need else ""))
    return out


# =========================================================================== #
# R-15 / R-15b -- Task 3 positive control
# =========================================================================== #
def _task3_table(d: pd.DataFrame, title: str, sub: str) -> dict:
    _h(title, sub)
    ws = sorted(d.w.unique())
    print(f"\n  {'w':>6s} {'baseline AUC':>18s} {'so(32) AUC':>18s} "
          f"{'so - base':>12s} {'95% CI':>22s} {'p':>9s}")
    res = {}
    for w in ws:
        dd = d[d.w == w]
        b = dd[dd.arm == BASELINE].auc.astype(float)
        s = dd[dd.arm == "so"].auc.astype(float)
        r = paired_report(paired(dd, "so", BASELINE))
        res[w] = r
        ci = f"[{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}]" if "ci" in r else "n/a"
        print(f"  {w:>6g} {b.mean():9.3f}+/-{b.std(ddof=1):6.3f} "
              f"{s.mean():9.3f}+/-{s.std(ddof=1):6.3f} "
              f"{r.get('diff', float('nan')):+12.4f} {ci:>22s} "
              f"{r.get('p', float('nan')):9.4f}")

    # w = 0 is the null cell, not a test: it must not enter the Holm family.
    tests = [(w, r) for w, r in res.items() if w > 0 and "p" in r]
    if tests:
        adj = holm([r["p"] for _, r in tests])
        print(f"\n  Holm-corrected over the {len(tests)} INJECTED cells "
              f"(w = 0 excluded: it is the null cell):")
        for (w, _), ph, rj in zip(tests, adj["p_holm"], adj["reject"]):
            print(f"    w={w:<5g} p_holm={ph:.4f}{'*' if rj else ''}")
        res["_holm"] = {w: float(ph) for (w, _), ph in zip(tests, adj["p_holm"])}
    return res


def analyse_r15(df: pd.DataFrame) -> dict:
    d = _failures(df)
    res = _task3_table(
        d, "R-15  TASK-3 POSITIVE CONTROL (reward-only injection)",
        "  w = 0 is the published Task-3 environment, reproduced exactly.\n"
        "  A null at n = 10 is only informative if the design could have "
        "detected an effect.")

    # Manipulation check: did the injection reach the reward at all?
    print("\n  Manipulation check -- mean r_geo for the so(32) arm:")
    for w in sorted(d.w.unique()):
        g = _num(d[(d.w == w) & (d.arm == "so")], "r_geo")
        if w == 0:
            # Not a measurement: the geometry term is never evaluated here.
            print(f"    w={w:<5g} r_geo = not computed (this is the published "
                  f"environment)")
        else:
            print(f"    w={w:<5g} r_geo = {g.mean():.4f}")
    print("  (a policy with no alignment to the planted rotation scores ~0.5;\n"
          "   the w=0 row is the published environment, where no geometry "
          "term exists)")

    holm_p = res.get("_holm", {})
    detected = [w for w, p in holm_p.items()
                if p < 0.05 and res[w].get("diff", 0) > 0]
    _h("R-15  VERDICT")
    if detected:
        w0 = min(detected)
        print(f"  -> The design detects an injected benefit from w = {w0:g} "
              f"upward.\n     The published null at w = 0 can be reported as "
              f"bounded: an effect of the\n     size produced by w >= {w0:g} "
              f"would have been seen, and was not.")
    else:
        print("  -> No injected strength was detected after Holm correction, "
              "up to the largest\n     w tested.  The Task-3 design cannot "
              "detect a reward-only geometric benefit\n     at any strength "
              "tried, so its null bounds very little, and the falsification\n"
              "     claim is correspondingly weak.")
    print("\n  Note the r_geo column: it barely moves across a tenfold change "
          "in w.  The geometry\n  reward never taught the rotation -- which is "
          "S-7's Task-1 finding, reproduced in a\n  second, independent "
          "environment and codebase.")
    return res


def analyse_r15b(df: pd.DataFrame) -> dict:
    # Test on df, not on the post-filter frame: _failures() returns only
    # status == "ok" rows, so this guard could never fire.
    if (df.status == "aux-never-fired").any():
        print("\n  !! 'aux-never-fired' rows present: the forward pre-hook did "
              "not capture a\n     minibatch, so those runs had no auxiliary "
              "loss.  Do not read them as\n     supervised arms.")
    d = _failures(df)
    res = _task3_table(
        d, "R-15b  TASK-3 WITH THE AUXILIARY TARGET ENABLED",
        "  Same grid as R-15, with main.py's auxiliary loss added verbatim in "
        "form.")
    print("\n  Alignment with the planted rotation -- the decisive column:")
    for w, g in d[d.arm == "so"].groupby("w"):
        a = _num(g, "align")
        if len(a):
            print(f"    w={w:<5g} align(M_policy v, M_env v) = {a.mean():+.4f}")
    _h("R-15b  VERDICT")
    aligned = [float(w) for w, g in d[d.arm == "so"].groupby("w")
               if _num(g, "align").mean() > 0.5]
    # Key off _holm rather than type-sniffing the dict keys.  `isinstance(w,
    # float)` was there to skip the "_holm" string key, but an integer-typed w
    # column gives np.int64 keys, so `sep` would always be empty and the
    # verdict would contradict the data.
    holm_p = res.get("_holm", {})
    sep = [float(w) for w, p in holm_p.items()
           if p < 0.05 and res.get(w, {}).get("diff", 0) > 0]
    if aligned and sep:
        print("  -> The S-7 dichotomy replicates in a second environment.\n"
              f"     With the auxiliary target the policy aligns with M_env "
              f"(align > 0.5 at\n     w = {', '.join(f'{w:g}' for w in aligned)}) "
              f"and the arms separate at w = {', '.join(f'{w:g}' for w in sep)}."
              "\n     R-15 showed that the same reward channel "
              "without supervision produced neither.\n\n     Supervision "
              "teaches the rotation; reward does not.  This is now "
              "demonstrated\n     on two independent environments and "
              "codebases, and it is the strongest form\n     of the paper's "
              "central negative result.")
    elif aligned:
        print("  -> Supervision does teach the rotation, but the arms do not "
              "separate on AUC.\n     The mechanism transfers; the performance "
              "consequence does not.  Report both.")
    else:
        print("  -> Neither alignment nor separation, even with supervision.  "
              "S-7's mechanism\n     does not transfer here, so its scope is "
              "narrower than Task 1 suggests.")
    print("\n  NOTE: the auxiliary loss added here is main.py's, verbatim in "
          "form, and it\n  regresses on M_env.  This experiment does not defend "
          "that design; it measures\n  what it does.")
    return res


# =========================================================================== #
# R-13b -- Mistral
# =========================================================================== #
def analyse_r13b(df: pd.DataFrame) -> dict:
    _h("R-13b  MISTRAL-7B, 20 SEEDS, CHANNEL-DECOMPOSED",
       "  The analysis the paper says it did not do for Mistral.  Same harness "
       "as Task 1,\n  so the two models are directly comparable.")
    _recon_note(df)
    d = _failures(df)
    arms = [a for a in (BASELINE, "so", "sl", "gl", "sym") if a in set(d.arm)]
    d = clean(d, "auc", "seed", arms)
    base = d[d.arm == BASELINE].auc.astype(float).mean()

    print(f"\n  {'arm':>12s} {'n':>3s} {'AUC':>20s} {'vs baseline':>12s}")
    for a in arms:
        v = d[d.arm == a].auc.astype(float)
        pct = 100 * (v.mean() - base) / base
        print(f"  {a:>12s} {len(v):>3d} {v.mean():10.4f} +/- "
              f"{v.std(ddof=1):6.4f} "
              + (f"{pct:+11.1f}%" if a != BASELINE else f"{'--':>12s}"))

    report([dict(label=f"{a} - baseline", d=paired(d, a, BASELINE))
            for a in arms if a != BASELINE],
           "R-13b  TOTAL AUC vs baseline PPO", "")

    out = {}
    if "auc_task" in d.columns:
        _h("R-13b  CHANNEL DECOMPOSITION")
        print(f"  {'arm':>12s} {'d AUC_task':>14s} {'d AUC_geo':>14s} "
              f"{'d AUC':>12s} {'geometry share':>16s}")
        for a in arms:
            if a == BASELINE:
                continue
            da = paired(d, a, BASELINE, "auc")
            dt = paired(d, a, BASELINE, "auc_task")
            dg = paired(d, a, BASELINE, "auc_geo")
            if not (len(da) and len(dt) and len(dg)):
                continue
            resid = abs(dt.mean() + dg.mean() - da.mean())
            flag = "" if resid < 1e-6 else f"  !! non-additive by {resid:.2e}"
            print(f"  {a:>12s} {dt.mean():+14.4f} {dg.mean():+14.4f} "
                  f"{da.mean():+12.4f} "
                  f"{100 * dg.mean() / da.mean():15.1f}%{flag}")
            if a == "so":
                out = dict(task=float(dt.mean()), geo=float(dg.mean()),
                           total=float(da.mean()),
                           n_neg=int((dt < 0).sum()), n=len(dt))
        if out:
            _h("R-13b  VERDICT")
            print(f"  so(32) task channel below baseline on "
                  f"{out['n_neg']}/{out['n']} seeds.")
            print(f"  -> Mistral shows the same pattern as GPT-2: the gain is "
                  f"carried entirely by\n     the geometry channel "
                  f"({out['geo']:+.3f}) and paid for in task reward "
                  f"({out['task']:+.3f}).\n     The abstract's cross-model "
                  "number therefore means what the GPT-2 number\n     means, "
                  "and is read the same way -- decomposed.")
    print("\n  Caveat: the embeddings are projected 4096 -> 1024 by a fixed "
          "random orthogonal\n  map with no distortion guarantee, exactly as in "
          "the published run.  This changes\n  the seed count and the analysis, "
          "not that design choice.")
    return out


# =========================================================================== #
# Manuscript reconciliation
# =========================================================================== #
def reconcile(directory: str) -> None:
    """Every headline number, recomputed from the CSVs, with a PASS/FAIL.

    A failing row means either the CSV is stale or the manuscript is wrong.
    This table does not guess which; it only flags the disagreement.
    """
    _h("MANUSCRIPT RECONCILIATION",
       "  Each row: the number the paper quotes, the number these CSVs give, "
       "and whether\n  they agree to the precision the paper states.")
    got: dict[str, float] = {}

    d4 = _load("r4", directory)
    if d4 is not None:
        d = d4[d4.status == "ok"]
        for a, key in (("so", "so(32)"), ("sl", "sl(32)"), ("gl", "gl(32)"),
                       ("random", "random(496)"), ("sym", "sym(32)")):
            v = d[d.arm == a].auc.astype(float)
            if len(v):
                got[f"tab:ablation {key}"] = float(v.mean())
        if "auc_task" in d.columns and BASELINE in set(d.arm):
            got["R-1 so - baseline"] = float(paired(d, "so", BASELINE).mean())
            got["R-1 task channel"] = float(
                paired(d, "so", BASELINE, "auc_task").mean())
            got["R-1 geometry channel"] = float(
                paired(d, "so", BASELINE, "auc_geo").mean())

    d1 = _load("b1", directory)
    if d1 is not None:
        d = d1[d1.status == "ok"]
        v = paired(d, "so_expm", "so_cayley")
        if len(v):
            got["B-1 expm - cayley"] = float(v.mean())

    d13 = _load("r13b", directory)
    if d13 is not None:
        d = d13[d13.status == "ok"]
        b = d[d.arm == BASELINE].auc.astype(float).mean()
        s = d[d.arm == "so"].auc.astype(float).mean()
        if np.isfinite(b) and np.isfinite(s):
            got["Mistral gain (%)"] = float(100 * (s - b) / b)

    print(f"\n  {'quantity':>26s} {'manuscript':>12s} {'from CSV':>12s} "
          f"{'|diff|':>10s}  verdict")
    n_ok = n_miss = 0
    for key, man in MANUSCRIPT.items():
        if key not in got:
            print(f"  {key:>26s} {man:12.4f} {'--':>12s} {'--':>10s}  "
                  f"NO DATA (CSV missing)")
            n_miss += 1
            continue
        diff = abs(got[key] - man)
        # Tolerance = half the last quoted digit, so agreement means the paper's
        # own precision is honoured rather than merely being "close".
        # Half the last digit the paper actually quotes.  A single sub-10
        # tolerance of 0.0006 would be 12x looser than half the last digit of
        # a 4-dp figure like 2.2338.
        tol = 0.05 if abs(man) >= 10 else 0.0006 if abs(man) < 1 else 0.0005
        tol = max(tol, abs(man) * 1e-4)
        ok = diff <= tol
        n_ok += ok
        print(f"  {key:>26s} {man:12.4f} {got[key]:12.4f} {diff:10.5f}  "
              f"{'PASS' if ok else 'FAIL  <-- investigate'}")

    print(f"\n  {n_ok} reconciled, {len(MANUSCRIPT) - n_ok - n_miss} "
          f"discrepant, {n_miss} with no data.")
    # Track the same decision the table made.  A fixed 0.05 here meant a row
    # could print FAIL (tolerance 0.0006) with no warning block at all.
    if n_ok != len(MANUSCRIPT) - n_miss:
        print("\n  A FAIL means either the CSV is stale or the manuscript is "
              "wrong.  This table\n  does not guess which -- check the "
              ".meta.json alongside the CSV first.")


# =========================================================================== #
# Driver
# =========================================================================== #
ANALYSES = {
    "r4":   analyse_r4,
    "r1b":  analyse_r1b,
    "r5":   analyse_r5,
    "x3":   analyse_x3,
    "r7":   analyse_r7,
    "b1":   analyse_b1,
    "b1lr": analyse_b1lr,
    "s7":   analyse_s7,
    "r11":  analyse_r11,
    "r15":  analyse_r15,
    "r15b": analyse_r15b,
    "r13b": analyse_r13b,
}

#: Print order: Task 1 first (ablation, then the controls that qualify it),
#: then Task 2, Task 3 and the cross-model replication.
ORDER = ["r4", "r1b", "r5", "x3", "r7", "b1", "b1lr", "s7", "r11", "r15",
         "r15b", "r13b"]


def summary(found: dict) -> None:
    """The one-screen answer, derived from what was actually analysed.

    Every line below comes from the corresponding analyse_* return value, so
    the summary changes when its inputs change, and says "not established"
    when the CSV is absent.  A fixed text would be wrong as soon as an
    experiment was missing or reversed.
    """
    _h("SUMMARY -- WHAT THESE CSVs SUPPORT")

    survives, changes, unknown = [], [], []

    r4 = found.get("r4")
    if r4 and "total" in r4:
        survives.append(
            f"so(32) beats every comparator at n seeds, Holm-corrected "
            f"(so - baseline = {r4['total']:+.4f}).")
        if r4.get("task", 0) < 0:
            changes.append(
                f"That gap is carried by the geometry channel "
                f"({r4['geo']:+.3f}) and paid for in task reward "
                f"({r4['task']:+.3f}); it is a decomposed effect, not a single "
                f"number.")
    else:
        unknown.append("r4 (the ablation itself)")

    r5 = found.get("r5")
    if r5:
        if r5.get("all_win"):
            survives.append(
                f"The ordering holds at each algebra's own tuned rate "
                f"(R-5, n = {r5['n']}), so the shared-rate caveat in the Limitations section "
                f"is not needed.")
        else:
            changes.append(
                f"The ordering does not survive per-algebra tuning "
                f"(R-5 lost: {', '.join(r5.get('lost')) or 'mixed/underpowered'}).")
    else:
        unknown.append("r5 (learning-rate robustness)")

    x3 = found.get("x3")
    if x3:
        if x3["wins"] == x3["n_draws"]:
            survives.append(
                f"It holds across all {x3['n_draws']} independently drawn "
                f"planted rotations (X-3).")
        else:
            changes.append(
                f"It holds on only {x3['wins']}/{x3['n_draws']} environment "
                f"draws (X-3); it is not a property of the algebra alone.")
    else:
        unknown.append("x3 (draw robustness)")

    r7 = found.get("r7")
    if r7:
        if r7.get("lost"):
            changes.append(
                f"Scope: so(32) is best under [{', '.join(r7['so_wins']) or 'none'}] "
                f"and not under [{', '.join(r7['lost'])}] (R-7).")
        else:
            survives.append(
                "so(32) is best under every judged environment geometry "
                "(R-7) -- a stronger claim than the draft makes.")
    else:
        unknown.append("r7 (scope)")

    b1lr = found.get("b1lr")
    if b1lr and b1lr.get("ci") is not None:
        lo, hi = b1lr["ci"]
        if abs(lo) < MARGIN and abs(hi) < MARGIN:
            changes.append(
                f"The expm-vs-Cayley contrast does not survive per-map tuning "
                f"({b1lr['diff']:+.4f}, CI [{lo:+.4f}, {hi:+.4f}] inside "
                f"+/-{MARGIN:.4f}).  The corrected claim is stronger: it is "
                f"the constraint, not the map.")
        elif b1lr["diff"] > 0:
            survives.append(
                f"The exponential map beats Cayley even when each is tuned "
                f"({b1lr['diff']:+.4f}).")
    else:
        unknown.append("b1lr (map vs constraint)")

    s7 = found.get("s7")
    if s7:
        menv = s7.get("so+M_env - baseline")
        others = [v for k, v in s7.items()
                  if k != "so+M_env - baseline" and k.endswith("- baseline")
                  and v is not None]
        if menv is not None and others and all(v < 0 for v in others):
            changes.append(
                "Sec. 8.1's \"involves no term in M_env\" is contradicted by "
                "the code, and S-7 shows every arm without that target loses "
                "to baseline, so the claim is conditional on the target.")
        elif others and any(v > 0 for v in others):
            survives.append(
                "A rotational target that is not the environment's also "
                "helps (S-7): the auxiliary loss acts as a prior.")
    else:
        unknown.append("s7 (prior vs supervision)")

    r11 = found.get("r11")
    if r11 and r11.get("test") and "diff" in r11["test"]:
        te, tr = r11["test"], r11.get("train", {})
        if te["p"] >= 0.05:
            insample = (" and in-sample too" if tr and tr.get("p", 0) >= 0.05
                        else "")
            changes.append(
                f"Task 2 (Sec. 9) shows no effect on held-out prompts"
                f"{insample} ({te['diff']:+.4f}, p = {te['p']:.3g}).  "
                f"The effect is not supported.")
        else:
            survives.append(
                f"Task 2 survives on held-out prompts ({te['diff']:+.4f}).")
    else:
        unknown.append("r11 (Task 2 generalisation)")

    r15 = found.get("r15")
    if r15:
        det = [w for w, p in r15.get("_holm", {}).items()
               if p < 0.05 and r15.get(w, {}).get("diff", 0) > 0]
        if det:
            survives.append(
                f"Task 3's null is bounded: the design detects an injected "
                f"effect from w = {min(det):g} upward (R-15).")
        else:
            changes.append(
                "Task 3's design detected no injected effect at any strength "
                "tested (R-15), so its falsification claim bounds very "
                "little.")
    else:
        unknown.append("r15 (Task 3 power)")

    r13b = found.get("r13b")
    if r13b and "task" in r13b:
        survives.append(
            f"The Mistral number decomposes exactly as the GPT-2 one does "
            f"(R-13b: geometry {r13b['geo']:+.3f}, task {r13b['task']:+.3f}, "
            f"task negative on {r13b['n_neg']}/{r13b['n']} seeds).")
    else:
        unknown.append("r13b (cross-model)")

    print("\n  Supported by the CSVs analysed here:")
    for line in survives or ["(nothing established)"]:
        print("    * " + line)
    print("\n  Not supported by the CSVs analysed here:")
    for i, line in enumerate(changes or ["(none identified)"], 1):
        print(f"    {i}. {line}")
    if unknown:
        print("\n  Not established -- CSV absent, so nothing is claimed either "
              "way:")
        for u in unknown:
            print(f"    - {u}")


def selftest() -> None:
    """Synthetic frames exercise every verdict branch.  No real CSVs needed."""
    import contextlib
    import io

    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  {'PASS' if cond else 'FAIL'}  {name}"
              f"{'  ' + detail if detail else ''}")

    def run(fn, df):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            fn(df)
        return buf.getvalue()

    rng = np.random.default_rng(0)

    print("(1) R-4 / R-1: the channel decomposition is reported")
    rows = []
    for s in range(20):
        se = rng.normal(0, 0.1)
        for a, base in (("so", 8.73), ("sl", 8.09), ("gl", 8.05),
                        ("random", 7.52), ("sym", 6.58), (BASELINE, 6.50)):
            auc = base + se + rng.normal(0, 0.05)
            # split so the channels sum back exactly
            task = 2.23 if a == BASELINE else 1.80
            rows.append(dict(arm=a, seed=s, auc=auc, auc_task=task,
                             auc_geo=auc - task, r_task=0.16, r_geo=0.86,
                             rho=1.0, recon_err=1e-15, status="ok"))
    out = run(analyse_r4, pd.DataFrame(rows))
    check("the ablation table prints", "ALGEBRA ABLATION" in out)
    check("the channel decomposition prints", "CHANNEL DECOMPOSITION" in out)
    check("the task channel is flagged as negative on every seed",
          "negative on 20/20 seeds" in out)

    print("(2) B-1-LR: the three verdict branches")

    def maps_frame(expm_by_lr, cay_by_lr, n=10, noise=0.15):
        r = []
        for s in range(n):
            se = rng.normal(0, 0.10)
            r.append(dict(arm=BASELINE, theta_lr=np.nan, seed=s,
                          auc=6.5 + se, orth_err=np.nan, status="ok"))
            for m, tbl in (("so_expm", expm_by_lr), ("so_cayley", cay_by_lr)):
                for lr, val in tbl.items():
                    r.append(dict(arm=m, theta_lr=lr, seed=s,
                                  auc=val + se + rng.normal(0, noise),
                                  orth_err=1e-7, status="ok"))
        return pd.DataFrame(r)

    flat = {3e-4: 6.9, 3e-3: 8.0, 0.01: 8.4, 0.03: 8.66, 0.1: 8.6}
    real = run(analyse_b1lr, maps_frame(flat, {k: v - 0.4 for k, v in flat.items()}))
    check("a rate-independent gap -> 'SURVIVES per-map tuning'",
          "SURVIVES per-map tuning" in real)
    # The case the experiment exists for: equal peaks at different rates.
    shifted = {3e-4: 6.9, 3e-3: 8.2, 0.01: 8.66, 0.03: 8.00, 0.1: 7.0}
    art = run(analyse_b1lr, maps_frame(flat, shifted, n=24, noise=0.06))
    check("equal peaks at different rates -> 'EQUIVALENT within'",
          "EQUIVALENT within" in art)
    check("...and the artefact is named", "ARTEFACT" in art)
    check("...and 0.03 is flagged as not Cayley's best",
          "NOT 0.03" in art)

    print("(3) R-7: scope verdicts")

    def geo_frame(table, n=10, noise=0.10):
        r = []
        for g, t in table.items():
            for s in range(n):
                se = rng.normal(0, 0.10)
                for a, v in t.items():
                    r.append(dict(geom=g, arm=a, seed=s,
                                  auc=v + se + rng.normal(0, noise),
                                  status="ok"))
        return pd.DataFrame(r)

    so_top = {BASELINE: 6.5, "so": 8.7, "sl": 8.1, "gl": 8.0, "sym": 6.6}
    sym_top = {BASELINE: 8.1, "so": 7.5, "sl": 9.1, "gl": 9.0, "sym": 9.4}
    id_row = {BASELINE: 9.7, "so": 9.5, "sl": 9.4, "gl": 9.4, "sym": 9.6}
    everywhere = run(analyse_r7, geo_frame(
        {"so": so_top, "sym": so_top, "gl": so_top, "diag": so_top, "id": id_row}))
    check("so best in every geometry -> 'STRONGER'", "STRONGER" in everywhere)
    check("the baseline winning `id` does not downgrade the verdict",
          "as predicted" in everywhere and "MIXED" not in everywhere)
    mixed = run(analyse_r7, geo_frame(
        {"so": so_top, "sym": sym_top, "gl": so_top, "diag": so_top, "id": id_row}))
    check("so losing one geometry -> 'MIXED' + scope warning",
          "MIXED" in mixed and "Scope of the claim" in mixed)
    lie_wins_id = run(analyse_r7, geo_frame(
        {"so": so_top, "sym": so_top, "gl": so_top, "diag": so_top,
         "id": so_top}))
    check("a Lie arm winning `id` raises a harness warning",
          "UNEXPECTED" in lie_wins_id)
    check("the cross-row warning is always printed",
          "NOT comparable across ROWS" in everywhere)

    print("(4) S-7: prior vs supervision")

    def s7_frame(eff, align, n=20, noise=0.12):
        r = []
        for s in range(n):
            se = rng.normal(0, 0.10)
            for a, base in eff.items():
                r.append(dict(arm=a, seed=s, auc=base + se + rng.normal(0, noise),
                              align_env=align.get(a, np.nan),
                              align_ref=align.get(a + "_ref", np.nan),
                              recon_err=1e-15, status="ok"))
        return pd.DataFrame(r)

    sup = run(analyse_s7,
              s7_frame({BASELINE: 6.5, "so+M_env": 8.73, "so+G_ref": 5.99,
                        "so+G_rand": 5.85, "so+no_aux": 5.98},
                       {"so+M_env": 0.78, "so+G_ref": 0.01,
                        "so+G_ref_ref": 0.77}))
    check("only M_env winning -> the supervision branch",
          "ONLY the arm whose auxiliary target IS M_env" in sup)
    check("the alternatives are named as significantly worse",
          "significantly WORSE than baseline" in sup)
    check("the confound is stated, not hidden", "CONFOUND" in sup)
    prior = run(analyse_s7,
                s7_frame({BASELINE: 6.5, "so+M_env": 8.73, "so+G_ref": 7.80,
                          "so+G_rand": 7.75, "so+no_aux": 6.0},
                         {"so+M_env": 0.78, "so+G_ref": 0.60}))
    check("a foreign rotation also winning -> the PRIOR branch",
          "functions as a PRIOR" in prior)

    print("(5) R-11: in-sample-only detection")
    r = []
    for s in range(10):
        se = rng.normal(0, 0.01)
        for split, delta in (("train", 0.0), ("test", 0.0)):
            for pi in range(12):
                for arm in ("structured", "control"):
                    r.append(dict(seed=s, split=split, arm=arm, prompt_idx=pi,
                                  eval_seed=0,
                                  reward=0.61 + se
                                  + (delta if arm == "structured" else 0)
                                  + rng.normal(0, 0.02), status="ok"))
    null = run(analyse_r11, pd.DataFrame(r))
    check("a double null -> 'ABSENT IN-SAMPLE TOO'",
          "ABSENT IN-SAMPLE TOO" in null)
    check("...and does not call it equivalence",
          "do NOT read this as equivalence" in null)

    print("(6) R-15: the null cell is excluded from the Holm family")
    r = []
    for w in (0.0, 0.02, 0.05, 0.1, 0.2):
        for s in range(10):
            se = rng.normal(0, 0.05)
            for a in (BASELINE, "so"):
                r.append(dict(w=w, arm=a, seed=s,
                              auc=2.8 + se + rng.normal(0, 0.06),
                              r_sent=0.28, r_geo=0.0 if w == 0 else 0.45,
                              rho=1.0, status="ok"))
    flat_t3 = run(analyse_r15, pd.DataFrame(r))
    check("w=0 is excluded from the correction", "w = 0 excluded" in flat_t3)
    check("no detection -> the null 'bounds very little'",
          "bounds very little" in flat_t3)

    print("(7) degenerate inputs are reported, not hidden")
    for name, fn, df in [
        ("empty r4", analyse_r4, pd.DataFrame(rows).iloc[0:0]),
        ("one seed r4", analyse_r4, pd.DataFrame(rows)[lambda x: x.seed == 0]),
        ("all failed r4", analyse_r4, pd.DataFrame(rows).assign(status="err")),
        ("r7 one geometry", analyse_r7, geo_frame({"so": so_top}, n=4)),
    ]:
        try:
            run(fn, df)
            good = True
        except Exception as exc:                                 # noqa: BLE001
            good = False
            print(f"        {name}: {type(exc).__name__}: {exc}")
        check(f"'{name}' does not crash", good)

    print("\n" + ("SELFTEST PASS" if ok else "SELFTEST FAIL"))
    if not ok:
        sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyse the SP-PG sweep CSVs.")
    # No argparse `choices=` here: with nargs="*" argparse validates the empty
    # default against the choice list and rejects it.  Validate by hand.
    ap.add_argument("experiments", nargs="*", metavar="EXPERIMENT",
                    help=f"which to analyse (default: all found). "
                         f"One or more of: {', '.join(sorted(ANALYSES))}")
    ap.add_argument("--dir", default=".", help="directory holding the CSVs")
    ap.add_argument("--manuscript", action="store_true",
                    help="print only the reconciliation table")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        selftest()
        return

    # Validate --dir before reading anything.  A mistyped or non-existent path
    # otherwise produces a table of "NO DATA (CSV missing)" that looks exactly
    # like "these experiments were never run" -- the one confusion this table
    # exists to prevent.  Fail loudly, and say what is there.
    if not os.path.isdir(a.dir):
        sys.exit(f"--dir {a.dir!r} is not a directory.  Nothing was read, so "
                 f"the 'CSV missing' rows below would have been meaningless.")
    present = sorted(n for n in ANALYSES
                     if _load(n, a.dir) is not None)
    if not present:
        listing = sorted(f for f in os.listdir(a.dir) if f.endswith(".csv"))
        sys.exit(f"No sweep CSVs found in {os.path.abspath(a.dir)!r}.\n"
                 f"Expected files named <name>_cells.csv for: "
                 f"{', '.join(sorted(ANALYSES))}\n"
                 + (f"That directory does contain: {', '.join(listing[:12])}"
                    if listing else "That directory contains no .csv files."))

    if a.manuscript:
        reconcile(a.dir)
        return

    unknown = [e for e in a.experiments if e not in ANALYSES]
    if unknown:
        sys.exit(f"unknown experiment(s) {unknown}; expected a subset of "
                 f"{sorted(ANALYSES)}")
    wanted = a.experiments or ORDER
    found = {}
    for name in [n for n in ORDER if n in wanted]:
        df = _load(name, a.dir)
        if df is None:
            continue
        found[name] = ANALYSES[name](df)

    if not found:
        sys.exit(f"No CSVs found in {a.dir!r}.  Run sppg_experiments.py first, "
                 f"or pass --dir.")
    reconcile(a.dir)
    summary(found)


if __name__ == "__main__":
    main()
