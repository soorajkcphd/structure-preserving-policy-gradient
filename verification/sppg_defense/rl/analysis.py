"""
Analysis of RL logs: the experiments that do not need to re-run the training
loop, only to read what it logged.

  R-1  channel_decomposition -- splits the AUC gap by reward channel.
       Task 1's reward is r = (1-w) r_task + w r_geo with w = 0.4, and the
       baseline runs with M = I, so it has no gradient path to the geometry
       component.  The headline +34.2% therefore mixes (a) what the method adds
       on the part of the reward both arms can optimise with (b) a channel the
       baseline structurally cannot reach.  This function separates them.
       Log `r_task` and `r_geo` per step and everything else follows.

  R-4  ablation_report -- multi-seed, multi-arm comparison with an omnibus test,
       pre-registered orthogonal contrasts and Holm correction, replacing the
       seed-0 Table tab:ablation.

  AUC  auc_trapezoid -- Eq. (7.1) exactly as the manuscript defines it:
       trapezoidal rule with unit spacing and half-weight endpoints, divided by
       T.  Also provides the normalised score, which is what makes effect sizes
       comparable across tasks and removes the uninformative d ~ 13.7.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..stats.tests import (holm, iqm, iqm_ci, paired_report,
                           probability_of_improvement)

__all__ = ["auc_trapezoid", "normalised_score", "channel_decomposition",
           "ablation_report", "CONTRASTS"]


def auc_trapezoid(returns) -> float:
    """
    AUC per Eq. (7.1): (1/T) * trapezoid(returns, dx=1), i.e. unit spacing with
    half-weight endpoints, then divided by T.  Matches the manuscript exactly.
    """
    r = np.asarray(returns, float)
    T = len(r)
    trap = np.trapezoid(r, dx=1.0) if hasattr(np, "trapezoid") else np.trapz(r, dx=1.0)
    return float(trap / T)


def normalised_score(auc: float, auc_random: float, auc_oracle: float) -> float:
    """
    (auc - random) / (oracle - random).

    Report this instead of raw AUC wherever effect sizes are compared across
    tasks.  The manuscript quotes a random-policy AUC of ~5.2 for Task 1 but
    only an unreachable upper bound of 18.8, so an oracle (or best-observed)
    reference must be computed for the normalisation to be meaningful.
    """
    denom = auc_oracle - auc_random
    return float((auc - auc_random) / denom) if denom != 0 else np.nan


# --------------------------------------------------------------------------- #
# R-1
# --------------------------------------------------------------------------- #
def channel_decomposition(df: pd.DataFrame, w: float = 0.4,
                          method_arm: str = "so(32)",
                          control_arm: str = "baseline_ppo",
                          arm_col: str = "arm", seed_col: str = "seed",
                          iter_col: str = "iteration",
                          task_col: str = "r_task", geo_col: str = "r_geo",
                          steps_per_episode: int = 20) -> dict:
    """
    Split the AUC gap into the task channel and the geometry channel.

    Expects a tidy frame with one row per (arm, seed, iteration) and the mean
    per-step task and geometry rewards for that iteration.  Episodic
    contributions are reconstructed as

        contribution_task = (1 - w) * r_task * steps_per_episode
        contribution_geo  =      w  * r_geo  * steps_per_episode

    so that contribution_task + contribution_geo reproduces the episodic return
    the manuscript's AUC is built from.

    Interpretation rules, fixed before looking at the output
    --------------------------------------------------------
      dAUC_task > 0 significantly : the advantage is not confined to the
                                    geometry reward.
      dAUC_task ~ 0               : the contribution is representational
                                    (a structure class that flat
                                    parameterisations cannot access, at no cost
                                    to task reward), not faster optimisation.
      dAUC_task < 0               : the method trades task reward for geometry
                                    reward, as Section 9.3 already describes.
    """
    need = {arm_col, seed_col, iter_col, task_col, geo_col}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")

    d = df.copy()
    dup = d.duplicated([arm_col, seed_col, iter_col]).sum()
    if dup:
        raise ValueError(f"{dup} duplicated (arm, seed, iteration) rows; these "
                         "would lengthen the AUC series")
    lens = d.groupby([arm_col, seed_col]).size()
    if lens.nunique() != 1:
        raise ValueError("all (arm, seed) series must have the same number of "
                         f"iterations; found {sorted(lens.unique())}. AUC is "
                         "divided by T and is not comparable across different T")
    d["contrib_task"] = (1 - w) * d[task_col] * steps_per_episode
    d["contrib_geo"] = w * d[geo_col] * steps_per_episode
    d["contrib_total"] = d["contrib_task"] + d["contrib_geo"]

    per_seed = (d.sort_values(iter_col)
                  .groupby([arm_col, seed_col])
                  .agg(auc_task=("contrib_task", auc_trapezoid),
                       auc_geo=("contrib_geo", auc_trapezoid),
                       auc_total=("contrib_total", auc_trapezoid))
                  .reset_index())

    # Seed alignment.  Sorting each arm independently and zipping is not enough:
    # if the arms have equal counts but different seed sets, the paired test
    # runs on misaligned pairs without any error.  Pivot on the seed index instead and
    # require both arms to be present for every seed.
    seeds_m = set(per_seed.loc[per_seed[arm_col] == method_arm, seed_col])
    seeds_c = set(per_seed.loc[per_seed[arm_col] == control_arm, seed_col])
    if not seeds_m:
        raise ValueError(f"no rows for arm {method_arm!r}")
    if not seeds_c:
        raise ValueError(f"no rows for arm {control_arm!r}")
    if seeds_m != seeds_c:
        raise ValueError(
            "the two arms do not share the same seed set (paired analysis "
            f"impossible): only in {method_arm!r}: {sorted(seeds_m - seeds_c)}; "
            f"only in {control_arm!r}: {sorted(seeds_c - seeds_m)}")

    out = {"per_seed": per_seed, "w": w, "n_seeds": len(seeds_m)}
    for col in ("auc_total", "auc_task", "auc_geo"):
        piv = per_seed.pivot(index=seed_col, columns=arm_col, values=col)
        a = piv[method_arm].to_numpy()
        b = piv[control_arm].to_numpy()
        rep = paired_report(a, b)
        out[col] = dict(
            method_mean=float(a.mean()), control_mean=float(b.mean()),
            diff=rep.diff, ci=rep.diff_ci, dz=rep.cohen_dz,
            t_p=rep.t_p, wilcoxon_p=rep.nonparametric_p,
            prob_superiority=rep.prob_superiority,
            power=rep.achieved_power,
            iqm_method=iqm(a), iqm_control=iqm(b),
        )

    tot = out["auc_total"]["diff"]
    out["share_of_gap_from_task_channel"] = (
        out["auc_task"]["diff"] / tot if tot != 0 else np.nan)
    out["share_of_gap_from_geo_channel"] = (
        out["auc_geo"]["diff"] / tot if tot != 0 else np.nan)

    task_ci = out["auc_task"]["ci"]
    if not np.isfinite(task_ci).all():
        out["verdict"] = ("CI UNAVAILABLE (degenerate bootstrap) -- inspect the "
                          "per-seed values before drawing any conclusion.")
        return out
    if task_ci[0] > 0:
        verdict = ("TASK CHANNEL POSITIVE: the advantage is not confined to the "
                   "geometry reward; the claim holds with the decomposition.")
    elif task_ci[1] < 0:
        verdict = ("TASK CHANNEL NEGATIVE: the method trades task reward for "
                   "geometry reward: a characterised trade-off.")
    else:
        verdict = ("TASK CHANNEL INDISTINGUISHABLE FROM ZERO: the headline is a "
                   "representational claim, not an optimisation one.")
    out["verdict"] = verdict
    return out


# --------------------------------------------------------------------------- #
# R-4
# --------------------------------------------------------------------------- #
# Pre-registered orthogonal contrasts.  Declaring these in advance is what makes
# them confirmatory rather than exploratory, and it is far more powerful than
# running all pairwise comparisons.
CONTRASTS = {
    "constrained vs unconstrained": (["so", "sl", "sym"], ["gl"]),
    "compact vs non-compact Lie": (["so"], ["sl"]),
    "Lie vs non-Lie": (["so", "sl"], ["sym"]),
    "so vs dimension-matched random subspace": (["so"], ["random"]),
}


def ablation_report(df: pd.DataFrame, value_col: str = "auc",
                    arm_col: str = "arm", seed_col: str = "seed",
                    reference: str = "so",
                    contrasts: dict | None = None,
                    alpha: float = 0.05) -> dict:
    """
    Multi-seed ablation analysis replacing the single-seed Table tab:ablation.

    Returns per-arm summaries (mean, sd, IQM with bootstrap CI), an omnibus
    Kruskal-Wallis test, Holm-corrected paired comparisons against `reference`,
    and the pre-registered contrasts.
    """
    from scipy import stats as _st

    contrasts = CONTRASTS if contrasts is None else contrasts
    arms = sorted(df[arm_col].unique())
    if reference not in arms:
        raise ValueError(f"reference arm {reference!r} not in {arms}")

    dup = df.duplicated([seed_col, arm_col]).sum()
    if dup:
        raise ValueError(f"{dup} duplicated (seed, arm) rows; pivot_table would "
                         "average them and report the wrong n")
    wide = df.pivot(index=seed_col, columns=arm_col, values=value_col)
    if wide.isna().any().any():
        raise ValueError("every arm must have a value for every seed (paired design)")

    summary = []
    for a in arms:
        v = wide[a].to_numpy()
        lo, hi = iqm_ci(v)
        summary.append(dict(arm=a, n=len(v), mean=float(v.mean()),
                            sd=float(v.std(ddof=1)), iqm=iqm(v),
                            iqm_ci_low=lo, iqm_ci_high=hi))

    # Friedman, not Kruskal-Wallis: the design is explicitly paired (the check
    # above enforces one value per arm per seed), and Kruskal-Wallis discards
    # the blocking.  With a strong seed effect it is near-powerless -- on a
    # synthetic 4-arm design with a real +0.5 effect and seed SD 10, KW gives
    # p = 0.93 where Friedman gives p = 4e-4.
    chi2, p_omni = _st.friedmanchisquare(*[wide[a].to_numpy() for a in arms])

    ref = wide[reference].to_numpy()
    pair_rows, pvals = [], []
    for a in arms:
        if a == reference:
            continue
        rep = paired_report(ref, wide[a].to_numpy())
        pi = probability_of_improvement(ref, wide[a].to_numpy())
        pair_rows.append(dict(reference=reference, arm=a, diff=rep.diff,
                              ci_low=rep.diff_ci[0], ci_high=rep.diff_ci[1],
                              dz=rep.cohen_dz, t_p=rep.t_p,
                              wilcoxon_p=rep.nonparametric_p,
                              prob_improvement=pi["prob_improvement"],
                              power=rep.achieved_power))
        pvals.append(rep.t_p)
    hp = holm(pvals, alpha) if pvals else dict(p_holm=[], reject=[])
    for row, ph, rj in zip(pair_rows, hp["p_holm"], hp["reject"]):
        row["p_holm"] = ph
        row["significant_after_holm"] = rj

    def _match(names, arms_):
        """
        Exact arm matching (with an optional '(' or '_' suffix so that "so"
        matches "so(32)" but never "solora(32)").  Prefix matching by startswith
        would pull unrelated arms onto both sides of a contrast -- e.g.
        arms ["so", "so_conj"] with contrast (["so"], ["so_conj"]) would compare a
        set against a subset of itself and still report a difference.
        """
        out = []
        for a in arms_:
            for x in names:
                if a == x or a.startswith(x + "(") or a.startswith(x + "_"):
                    out.append(a)
                    break
        return out

    con_rows = []
    for label, (left, right) in contrasts.items():
        L = _match(left, arms)
        R = _match(right, arms)
        if not L or not R:
            continue
        overlap = set(L) & set(R)
        if overlap:
            raise ValueError(f"contrast {label!r} has arms on both sides: "
                             f"{sorted(overlap)}")
        lv = wide[L].mean(axis=1).to_numpy()
        rv = wide[R].mean(axis=1).to_numpy()
        rep = paired_report(lv, rv)
        con_rows.append(dict(contrast=label, left=L, right=R, diff=rep.diff,
                             ci_low=rep.diff_ci[0], ci_high=rep.diff_ci[1],
                             dz=rep.cohen_dz, p=rep.t_p, power=rep.achieved_power))
    cp = holm([r["p"] for r in con_rows], alpha) if con_rows else dict(p_holm=[], reject=[])
    for row, ph, rj in zip(con_rows, cp["p_holm"], cp["reject"]):
        row["p_holm"] = ph
        row["significant_after_holm"] = rj

    return dict(summary=pd.DataFrame(summary).sort_values("mean", ascending=False),
                omnibus=dict(test="Friedman (paired blocks = seeds)",
                             friedman_chi2=float(chi2), p=float(p_omni),
                             n_arms=len(arms), n_seeds=int(wide.shape[0])),
                pairwise=pd.DataFrame(pair_rows),
                contrasts=pd.DataFrame(con_rows),
                family_note=(f"Holm applied within two families: "
                             f"{len(pair_rows)} comparisons against {reference!r}, "
                             f"and {len(con_rows)} pre-registered contrasts."))
