"""
Statistical toolkit for the SP-PG revision.

  * Paired vs unpaired.  Section 7.1 states "SP-PG and the baseline are
    evaluated on the same prompts under the same seeds (paired design)", while
    Welch's t and Mann-Whitney U both assume independent samples.
    `paired_report` and `unpaired_report` are provided separately so the
    analysis can match the design.

  * Effect sizes when variance is tiny.  Cohen's d ~ 13.7 and ~ 47 are not
    calibrated magnitudes.  Every report therefore also returns the raw
    difference with a CI, the paired d_z, and the probability of superiority
    with an exact Clopper-Pearson interval.

  * Equivalence.  `tost_paired` / `tost_unpaired` implement two one-sided tests
    and also return the smallest margin at which equivalence would still be
    declared, which is the informative quantity for Task 3.

  * Multiplicity.  `holm` adjusts a family of p-values.

  * Power.  `n_for_power_paired` gives the seed budget for a target effect, and
    `power_paired` the achieved power.

  * Aggregation.  `iqm` with a bootstrap CI, and `probability_of_improvement`,
    following the rliable recommendations for reinforcement-learning reporting.
    Note: the resampling here is i.i.d. over seeds, not rliable's stratified
    (within-task) bootstrap; with a single task the two coincide.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from scipy import stats

__all__ = [
    "paired_report", "unpaired_report", "bootstrap_ci", "bca_ci",
    "tost_paired", "tost_unpaired", "holm",
    "power_paired", "n_for_power_paired",
    "iqm", "iqm_ci", "probability_of_improvement", "prob_superiority_paired",
]


# --------------------------------------------------------------------------- #
# confidence intervals
# --------------------------------------------------------------------------- #
def bca_ci(x, statistic=np.mean, alpha: float = 0.05, n_resamples: int = 20000,
           seed: int = 0) -> tuple[float, float]:
    """Bias-corrected and accelerated bootstrap CI for a one-sample statistic."""
    x = np.asarray(x, float)
    if x.std(ddof=1) == 0 if len(x) > 1 else True:
        # BCa's acceleration is 0/0 for constant data and returns (nan, nan);
        # a NaN interval downstream would otherwise become a scientific verdict.
        return float(statistic(x)), float(statistic(x))
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        try:
            res = stats.bootstrap((x,), statistic, confidence_level=1 - alpha,
                                  n_resamples=n_resamples, method="BCa",
                                  random_state=np.random.default_rng(seed))
            lo, hi = float(res.confidence_interval.low), float(res.confidence_interval.high)
            if np.isfinite(lo) and np.isfinite(hi):
                return lo, hi
        except Exception:
            pass
    return bootstrap_ci(x, statistic, alpha, n_resamples, seed)


def bootstrap_ci(x, statistic=np.mean, alpha: float = 0.05,
                 n_resamples: int = 20000, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI (the interval type used in the manuscript)."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    idx = rng.integers(0, len(x), size=(n_resamples, len(x)))
    vals = np.array([statistic(x[i]) for i in idx])
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


# --------------------------------------------------------------------------- #
# effect sizes
# --------------------------------------------------------------------------- #
def prob_superiority_paired(a, b, alpha: float = 0.05) -> dict:
    """
    P(a > b) from matched pairs, with an exact Clopper-Pearson interval,
    conditioned ON the discordant pairs (the sign-test convention).

    Ties are excluded from both the estimate and the interval, so the point
    estimate always lies inside its own CI.  (Splitting ties into
    the estimate, 0.5*ties, while the interval uses strict wins over the full n
    would, with 5 wins and 5 ties, put a point estimate of 0.75 inside a CI
    centred on 0.5.)  `p_superiority_tie_split` is also returned for the
    convention that counts ties as half a win, but the CI belongs to the
    conditional estimate.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    wins = int(np.sum(a > b))
    losses = int(np.sum(a < b))
    ties = int(np.sum(a == b))
    n = len(a)
    n_disc = wins + losses
    if n_disc == 0:
        return dict(p_superiority=float("nan"),
                    p_superiority_tie_split=0.5, wins=wins, losses=losses,
                    ties=ties, n=n, n_discordant=0,
                    ci_low=float("nan"), ci_high=float("nan"),
                    note="all pairs tied; no discordant pairs to condition on")
    ci = stats.binomtest(wins, n_disc).proportion_ci(1 - alpha, method="exact")
    return dict(p_superiority=wins / n_disc,
                p_superiority_tie_split=(wins + 0.5 * ties) / n,
                wins=wins, losses=losses, ties=ties, n=n, n_discordant=n_disc,
                ci_low=float(ci.low), ci_high=float(ci.high))


# --------------------------------------------------------------------------- #
# tests
# --------------------------------------------------------------------------- #
@dataclass
class TestReport:
    design: str
    n_a: int
    n_b: int
    mean_a: float
    mean_b: float
    sd_a: float
    sd_b: float
    diff: float
    diff_ci: tuple
    t_stat: float
    t_p: float
    nonparametric_stat: float
    nonparametric_p: float
    cohen_d_pooled: float
    cohen_dz: float | None
    prob_superiority: float | None
    achieved_power: float | None
    note: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def paired_report(a, b, alpha: float = 0.05, seed: int = 0) -> TestReport:
    """
    Paired analysis of matched runs (same seed / same prompts in both arms).
    `a` is the method, `b` the control; diff = mean(a - b).
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if a.shape != b.shape:
        raise ValueError("paired analysis needs equal-length, matched arrays")
    d = a - b
    n = len(d)
    degenerate = bool(d.std(ddof=1) == 0)
    if degenerate:
        # An exactly-null (or exactly-constant) difference is a legitimate
        # outcome -- it is exactly the R-1 result the channel decomposition
        # exists to detect -- so report it rather than aborting the analysis.
        t, p = (0.0, 1.0) if d.mean() == 0 else (np.inf * np.sign(d.mean()), 0.0)
    else:
        t, p = stats.ttest_rel(a, b)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            w, pw = stats.wilcoxon(a, b)
        except ValueError:                               # all differences zero
            w, pw = float("nan"), 1.0
    if not np.isfinite(pw):
        w, pw = float("nan"), 1.0
    dz = float(d.mean() / d.std(ddof=1)) if not degenerate else (
        np.inf if d.mean() != 0 else 0.0)
    pooled_sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    dpool = float((a.mean() - b.mean()) / pooled_sd) if pooled_sd > 0 else np.inf
    ps = prob_superiority_paired(a, b, alpha)
    return TestReport(
        design="paired", n_a=n, n_b=n,
        mean_a=float(a.mean()), mean_b=float(b.mean()),
        sd_a=float(a.std(ddof=1)), sd_b=float(b.std(ddof=1)),
        diff=float(d.mean()), diff_ci=bca_ci(d, alpha=alpha, seed=seed),
        t_stat=float(t), t_p=float(p),
        nonparametric_stat=float(w), nonparametric_p=float(pw),
        cohen_d_pooled=dpool, cohen_dz=dz,
        prob_superiority=ps["p_superiority"],
        achieved_power=power_paired(abs(dz), n, alpha) if np.isfinite(dz) else 1.0,
        note=("paired t is primary; Wilcoxon signed-rank is the nonparametric "
              "check" + ("  [DEGENERATE: all paired differences identical]"
                         if degenerate else "")),
    )


def unpaired_report(a, b, alpha: float = 0.05, seed: int = 0) -> TestReport:
    """Welch's t and Mann-Whitney U for independent samples."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    t, p = stats.ttest_ind(a, b, equal_var=False)
    u, pu = stats.mannwhitneyu(a, b, alternative="two-sided")
    pooled_sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    dpool = float((a.mean() - b.mean()) / pooled_sd) if pooled_sd > 0 else np.inf
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    df = se ** 4 / (
        (a.var(ddof=1) / len(a)) ** 2 / (len(a) - 1)
        + (b.var(ddof=1) / len(b)) ** 2 / (len(b) - 1))
    tc = stats.t.ppf(1 - alpha / 2, df)
    diff = float(a.mean() - b.mean())
    return TestReport(
        design="unpaired(Welch)", n_a=len(a), n_b=len(b),
        mean_a=float(a.mean()), mean_b=float(b.mean()),
        sd_a=float(a.std(ddof=1)), sd_b=float(b.std(ddof=1)),
        diff=diff, diff_ci=(diff - tc * se, diff + tc * se),
        t_stat=float(t), t_p=float(p),
        nonparametric_stat=float(u), nonparametric_p=float(pu),
        cohen_d_pooled=dpool, cohen_dz=None, prob_superiority=None,
        achieved_power=None,
        note="use only if the arms are independent; the manuscript "
             "describes a paired design, for which paired_report is correct",
    )


# --------------------------------------------------------------------------- #
# equivalence
# --------------------------------------------------------------------------- #
def tost_paired(a, b, margin: float, alpha: float = 0.05) -> dict:
    """
    Two one-sided tests for equivalence of paired samples within +/- margin.
    Also returns the smallest margin at which equivalence would still hold,
    which is the informative number for a negative control.
    """
    d = np.asarray(a, float) - np.asarray(b, float)
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n)
    if se == 0 or not np.isfinite(se):
        raise ValueError("degenerate standard error (zero-variance differences); "
                         "TOST is undefined here and must not be reported as "
                         "'equivalent, p = 0'")
    dfree = n - 1
    t1 = (d.mean() + margin) / se          # H01: mu <= -margin
    t2 = (d.mean() - margin) / se          # H02: mu >= +margin
    p1 = 1 - stats.t.cdf(t1, dfree)
    p2 = stats.t.cdf(t2, dfree)
    if not (np.isfinite(p1) and np.isfinite(p2)):
        raise ValueError("non-finite one-sided p-value in TOST")
    tc = stats.t.ppf(1 - alpha, dfree)     # 1-2*alpha CI is the TOST-compatible one
    lo, hi = d.mean() - tc * se, d.mean() + tc * se
    return dict(margin=margin, mean_diff=float(d.mean()),
                p_lower=float(p1), p_upper=float(p2), p_tost=float(max(p1, p2)),
                equivalent=bool(max(p1, p2) < alpha),
                ci_level=f"{100 * (1 - 2 * alpha):.0f}%",
                ci_low=float(lo), ci_high=float(hi),
                smallest_equivalence_margin=float(max(abs(lo), abs(hi))))


def tost_unpaired(a, b, margin: float, alpha: float = 0.05) -> dict:
    """Two one-sided Welch tests for equivalence of independent samples."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    if se == 0 or not np.isfinite(se):
        raise ValueError("degenerate standard error; TOST is undefined here")
    dfree = se ** 4 / (
        (a.var(ddof=1) / len(a)) ** 2 / (len(a) - 1)
        + (b.var(ddof=1) / len(b)) ** 2 / (len(b) - 1))
    diff = a.mean() - b.mean()
    p1 = 1 - stats.t.cdf((diff + margin) / se, dfree)
    p2 = stats.t.cdf((diff - margin) / se, dfree)
    if not (np.isfinite(p1) and np.isfinite(p2)):
        raise ValueError("non-finite one-sided p-value in TOST")
    tc = stats.t.ppf(1 - alpha, dfree)
    lo, hi = diff - tc * se, diff + tc * se
    return dict(margin=margin, mean_diff=float(diff),
                p_lower=float(p1), p_upper=float(p2), p_tost=float(max(p1, p2)),
                equivalent=bool(max(p1, p2) < alpha),
                ci_level=f"{100 * (1 - 2 * alpha):.0f}%",
                ci_low=float(lo), ci_high=float(hi),
                smallest_equivalence_margin=float(max(abs(lo), abs(hi))))


# --------------------------------------------------------------------------- #
# multiplicity
# --------------------------------------------------------------------------- #
def holm(pvals, alpha: float = 0.05) -> dict:
    """Holm-Bonferroni step-down adjustment for a family of p-values."""
    p = np.asarray(pvals, float)
    if not np.isfinite(p).all():
        raise ValueError("holm() received a non-finite p-value; a degenerate "
                         "upstream test must be handled explicitly, not "
                         "reported as 'not significant'")
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return dict(p_raw=p.tolist(), p_holm=adj.tolist(),
                reject=(adj <= alpha).tolist(), family_size=m, alpha=alpha)


# --------------------------------------------------------------------------- #
# power
# --------------------------------------------------------------------------- #
def power_paired(dz: float, n: int, alpha: float = 0.05) -> float:
    """Achieved power of a two-sided paired t-test at paired effect size dz."""
    if n < 2:
        return float("nan")
    df = n - 1
    ncp = dz * np.sqrt(n)
    crit = stats.t.ppf(1 - alpha / 2, df)
    return float(stats.nct.sf(crit, df, ncp) + stats.nct.cdf(-crit, df, ncp))


def n_for_power_paired(dz: float, power: float = 0.8, alpha: float = 0.05,
                       n_max: int = 10_000) -> int:
    """Smallest n giving at least `power` for a two-sided paired t-test."""
    for n in range(2, n_max + 1):
        if power_paired(dz, n, alpha) >= power:
            return n
    return n_max


# --------------------------------------------------------------------------- #
# aggregation (rliable-style)
# --------------------------------------------------------------------------- #
def iqm(x, method: str = "percentile") -> float:
    """Interquartile mean.  Two conventions exist and they do not agree.

    ``method="percentile"`` (default, and what the project's own code uses in
    ``main.py``/``sppg_core.py``): compute q25 and q75 by linear interpolation
    and average every value in the closed band ``[q25, q75]``.

    ``method="slice"`` (the index-slice convention used by some rliable-style
    implementations): sort, then average ``x[floor(n/4) : ceil(3n/4)]``.

    For n = 10 the two keep different numbers of seeds -- the percentile band
    typically retains 4 values, the slice always retains 6 -- so they give
    different answers on the same data.  Task-1's SPPG arm is a live example:
    percentile -> 8.6525 (the value Table tab:task1 reports as 8.652), slice
    -> 8.6867.  The default is "percentile" so that this package reconciles
    against the manuscript's own estimand rather than a different one.
    """
    x = np.asarray(x, float)
    if method == "slice":
        x = np.sort(x)
        n = len(x)
        lo, hi = int(np.floor(n * 0.25)), int(np.ceil(n * 0.75))
        return float(x[lo:hi].mean())
    if method != "percentile":
        raise ValueError(f"iqm: unknown method {method!r} "
                         "(expected 'percentile' or 'slice')")
    q25, q75 = np.percentile(x, [25, 75])
    trimmed = x[(x >= q25) & (x <= q75)]
    return float(trimmed.mean()) if len(trimmed) else float(x.mean())


def iqm_ci(x, alpha: float = 0.05, n_resamples: int = 20000, seed: int = 0):
    """Percentile bootstrap CI for the IQM (i.i.d. resampling over seeds)."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    idx = rng.integers(0, len(x), size=(n_resamples, len(x)))
    vals = np.array([iqm(x[i]) for i in idx])
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def probability_of_improvement(a, b, n_resamples: int = 20000,
                               alpha: float = 0.05, seed: int = 0) -> dict:
    """
    P(a > b) over all cross pairs, with a bootstrap CI.  This is the
    rliable-style aggregate that remains meaningful when the arms separate
    completely (where Cohen's d explodes to an uninformative number).
    """
    a = np.asarray(a, float); b = np.asarray(b, float)
    point = float((a[:, None] > b[None, :]).mean()
                  + 0.5 * (a[:, None] == b[None, :]).mean())
    rng = np.random.default_rng(seed)
    vals = np.empty(n_resamples)
    for i in range(n_resamples):
        aa = a[rng.integers(0, len(a), len(a))]
        bb = b[rng.integers(0, len(b), len(b))]
        # the same tie convention as the point estimate, or the CI is biased low
        # and can exclude its own point estimate
        vals[i] = ((aa[:, None] > bb[None, :]).mean()
                   + 0.5 * (aa[:, None] == bb[None, :]).mean())
    return dict(prob_improvement=point,
                ci_low=float(np.quantile(vals, alpha / 2)),
                ci_high=float(np.quantile(vals, 1 - alpha / 2)))
