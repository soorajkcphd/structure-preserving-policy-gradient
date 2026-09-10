"""
sppg_core.py -- shared infrastructure for the SP-PG revision experiments.
=====================================================================

This is the layer every experiment sits on.  It exists so that the statistics,
the reward-channel accounting and the arm construction are written once and
tested once, rather than copied into fifteen scripts where they can drift.

It replaces, and is behaviourally identical to, the pair `_r1core.py` +
`_armlib.py`, plus the parts of `sppg_defense.stats.tests` and
`sppg_defense.rl.analysis` that the run scripts actually import.

Contents
--------
    1. Statistics          paired_report, holm, tost_paired, power helpers
    2. Channel accounting  instrument / instrument_trainer / channel_aucs
    3. Arm construction    build / run_one, and the random-subspace control
    4. Bookkeeping         Sink, meta guards, clean / paired / report

The invariants, and exactly what each one proves
------------------------------------------------
`instrument_trainer(..., strict=True)` reconstructs main.py's own episode
return from the two channels every iteration and raises on disagreement.

What that check covers.  Because the task channel is recovered by inverting
the mixture, t = (reward - w*g)/(1-w), the reconstruction (1-w)*t + w*g is
algebraically identical to `reward` for any g.  So the strict check proves the
episode accounting is right -- which steps are summed, the 512-vs-25x20 orphan
handling, the live read of w -- and says nothing about whether `g` is the
environment's actual geometry reward.

The claim "auc_task + auc_geo == the reported AUC" is therefore true by
construction, not by measurement, and does not show that the channel split is
correct.

What does police the split is the second invariant: `geo_wrapper` counts its
own calls, and `step_wrapper` raises unless `_geometry_reward` fired exactly
once for this step, with both
recovered channels inside [0, 1].  If main.py ever renames that method, moves
it to a module-level helper, or skips it on a branch (e.g. M is None for the
baseline arm), `g` would become stale and `t` would absorb the error
with recon_err still at 1e-15.  The counter catches that; the reconstruction
cannot.  Do not set strict=False to make a run go through.

Why the episode accounting is fiddly
------------------------------------
main.py's `_gather_trajectory` runs `steps_per_iter` (512) steps but appends to
`ep_returns_all` only when an episode terminates, so the trailing partial
episode (512 - 25*20 = 12 steps) is discarded.  Those 12 steps are always
non-terminal, and non-terminal steps are the only ones subject to the 0.7
sparsity mask, so averaging over all 512 steps biases the task channel
downward.  We mirror the episode accounting exactly; the strict check is what
proves we got it right.

Every guard in here was added after a defect found in a real run; the
comments say which.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Constants shared across experiments
# --------------------------------------------------------------------------- #
BASELINE = "baseline_ppo"

#: main.py names the unconstrained algebra "full"; every script here says "gl",
#: because that is what the manuscript calls it.  One place to map them.
ALGEBRA_OF = {"so": "so", "sl": "sl", "gl": "full", "sym": "sym"}

#: Relative tolerance for the channel-reconstruction check (see module docstring).
RECON_TOL = 1e-6

#: The paper's operating point.  Both are read live from the env where possible;
#: these are only defaults for scripts that need to state them up front.
W_DEFAULT = 0.4
HORIZON_DEFAULT = 20


# =========================================================================== #
# 1. Statistics
# =========================================================================== #
class TestReport(dict):
    """A dict with attribute access, so `r.diff` and `r["diff"]` both work."""

    __getattr__ = dict.__getitem__


def paired_report(a, b=None, alpha: float = 0.05) -> TestReport:
    """Paired comparison of `a` against `b` (or against zero if b is None).

    Returns diff, its CI, Cohen's dz, the two-sided paired-t p-value and the
    achieved power at the observed effect.

    On `power`: it is a monotone restatement of p and says nothing about a null.
    It is reported for completeness; the appropriate instrument for a
    null is `tost_paired` below.

    An exact null (every difference zero) is reported, not raised, because
    R-1 exists to detect exactly that outcome.
    """
    a = np.asarray(a, dtype=float)
    d = a if b is None else a - np.asarray(b, dtype=float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 2:
        return TestReport(n=n, degenerate="n<2")

    m = float(d.mean())
    s = float(d.std(ddof=1))
    if s == 0.0:
        # Zero variance makes the t-statistic infinite and the p-value 0: a
        # maximally confident result from data carrying no information.
        return TestReport(n=n, diff=m, degenerate="zero variance")

    from scipy import stats as st

    t_stat, p = st.ttest_1samp(d, 0.0)
    if not np.isfinite(p):
        return TestReport(n=n, diff=m, degenerate="non-finite p")

    half = st.t.ppf(1 - alpha / 2, n - 1) * s / math.sqrt(n)
    dz = m / s
    return TestReport(
        n=n, diff=m, ci=(m - half, m + half), dz=dz, t=float(t_stat),
        p=float(p), power=power_paired(abs(dz), n, alpha),
    )


def holm(pvals, alpha: float = 0.05) -> dict:
    """Holm-Bonferroni step-down.

    Two details differ from the naive version:
      * rejection uses `<= alpha`, not `< alpha` (a p-value landing exactly on
        the threshold is a rejection);
      * a non-finite p-value raises instead of being treated as
        "not significant", which would turn a broken run into a published
        null.
    """
    p = np.asarray(pvals, dtype=float)
    if not np.isfinite(p).all():
        raise ValueError("holm() received a non-finite p-value; fix the "
                         "upstream test rather than reporting it as null.")
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m, dtype=float)
    running = 0.0
    for i, idx in enumerate(order):
        running = max(running, min(1.0, (m - i) * p[idx]))
        adj[idx] = running
    return {"p_holm": adj, "reject": adj <= alpha}


def tost_paired(d, margin: float, alpha: float = 0.05) -> dict:
    """Two one-sided tests: is the paired difference equivalent to zero?

    "Not significant" does not establish equivalence.  Any claim that two arms
    are the same must come through here, with a margin fixed before seeing the
    result.  Returns `equivalent` plus the smallest margin that would have held,
    which is the informative quantity to report when it does not.
    """
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 2:
        raise ValueError("TOST needs n >= 2")
    s = float(d.std(ddof=1))
    if s == 0.0:
        raise ValueError("TOST is undefined at zero variance; it would "
                         "certify equivalence at p = 0 from no information.")

    from scipy import stats as st

    se = s / math.sqrt(n)
    m = float(d.mean())
    p_lo = st.t.sf((m + margin) / se, n - 1)       # H0: diff <= -margin
    p_hi = st.t.cdf((m - margin) / se, n - 1)      # H0: diff >= +margin
    p_tost = max(p_lo, p_hi)
    half = st.t.ppf(1 - alpha, n - 1) * se         # 90% CI for TOST
    return {
        "p_tost": float(p_tost),
        "equivalent": bool(p_tost < alpha),
        "ci90": (m - half, m + half),
        "smallest_equivalence_margin": float(abs(m) + half),
    }


def equivalent(report, margin: float) -> bool:
    """True only if the whole confidence interval lies inside +/- margin."""
    if not report or "ci" not in report:
        return False
    lo, hi = report["ci"]
    return bool(abs(lo) < margin and abs(hi) < margin)


def power_paired(dz: float, n: int, alpha: float = 0.05) -> float:
    """Achieved power of a paired t-test at effect size dz and sample size n."""
    from scipy import stats as st

    if n < 2 or not np.isfinite(dz):
        return float("nan")
    ncp = dz * math.sqrt(n)
    crit = st.t.ppf(1 - alpha / 2, n - 1)
    p = float(st.nct.sf(crit, n - 1, ncp) + st.nct.cdf(-crit, n - 1, ncp))
    # scipy's nct can overflow to NaN at large |ncp| (observed at dz=3.63,
    # n=20, sitting next to a clean 1.00 for dz=3.04).  Power is monotone in
    # |ncp| and already ~1 well before that, so a NaN here is numerical, not
    # informative; report the saturated value rather than a hole in the table.
    if not math.isfinite(p):
        return 1.0 if abs(ncp) > crit else float("nan")
    return p


def n_for_power_paired(dz: float, power: float = 0.8, alpha: float = 0.05,
                       n_max: int = 10_000) -> int | None:
    """Smallest n reaching `power` at effect size dz."""
    if not np.isfinite(dz) or dz == 0:
        return None
    for n in range(2, n_max):
        if power_paired(abs(dz), n, alpha) >= power:
            return n
    return None


def iqm(x) -> float:
    """Interquartile mean -- the robust central tendency the RL literature uses."""
    x = np.sort(np.asarray(x, dtype=float))
    q25, q75 = np.percentile(x, [25, 75])
    trimmed = x[(x >= q25) & (x <= q75)]
    return float(trimmed.mean()) if len(trimmed) else float(x.mean())


def auc_trapezoid(returns) -> float:
    """main.py's own AUC: trapezoid over iterations, divided by the count.

    Must match main.py exactly or the channel decomposition is a different
    estimand from the number the paper reports.
    """
    r = np.asarray(returns, dtype=float)
    return float(np.trapezoid(r) / max(len(r), 1)) if hasattr(np, "trapezoid") \
        else float(np.trapz(r) / max(len(r), 1))


# =========================================================================== #
# 2. Reward-channel accounting
# =========================================================================== #
def instrument(env) -> float:
    """Record r_task / r_geo per step, with episode boundaries.  Idempotent.

    The environment's reward is  r = (1-w)*r_task + w*r_geo.  We wrap
    `_geometry_reward` to capture r_geo, then invert the mixture to recover
    r_task.  Neither the reward nor the RNG stream is touched.

    `w` is read live inside the wrapper rather than captured here, so a sweep
    (R-1b) can mutate env.geo_weight between blocks and the recovery stays
    exact.  Identical behaviour when w is constant.
    """
    if getattr(env, "_ch_instrumented", False):
        return float(env.geo_weight)

    w0 = float(env.geo_weight)
    if not (0.0 <= w0 < 1.0):
        raise ValueError(f"geo_weight={w0}: at w=1 the task channel is not "
                         f"identifiable (its coefficient is zero).")

    env._ch_ep_task, env._ch_ep_geo = [], []     # completed-episode sums
    env._ch_cur_task = env._ch_cur_geo = 0.0     # current episode
    env._ch_step_task = env._ch_step_geo = 0.0   # all steps (fallback only)
    env._ch_n = 0
    env._ch_geo_last = 0.0
    env._ch_geo_calls = 0            # witness: see the module docstring

    orig_geo, orig_step = env._geometry_reward, env.step

    def geo_wrapper(*a, **kw):
        g = orig_geo(*a, **kw)
        env._ch_geo_last = float(g)
        env._ch_geo_calls += 1
        return g

    def step_wrapper(*a, **kw):
        w = float(env.geo_weight)                # live read: see docstring
        if not (0.0 <= w < 1.0):
            raise ValueError(f"geo_weight={w}: task channel not identifiable.")
        calls_before = env._ch_geo_calls
        out = orig_step(*a, **kw)
        reward, done = float(out[1]), bool(out[2])
        # This is the check that polices the split.  The reconstruction in
        # instrument_trainer is an identity in g and cannot catch a stale
        # reading; this can.  If _geometry_reward did not fire exactly once
        # inside this step, `g` belongs to a previous step and the channel
        # decomposition is unmeasured, so raise rather than record it.
        fired = env._ch_geo_calls - calls_before
        if fired != 1:
            raise RuntimeError(
                f"_geometry_reward fired {fired} time(s) during this step, "
                f"expected exactly 1.  The reward channels cannot be split; "
                f"has main.py renamed or bypassed it?")
        g = env._ch_geo_last
        t = (reward - w * g) / (1.0 - w)         # invert the mixture
        # A range check, but a WARNING rather than a raise: the geometry reward
        # is cosine-derived and lives in [0, 1], whereas the task
        # reward passes through additive noise (reward_noise=0.2) and can leave
        # the unit interval on a single step without anything being wrong.
        # Raising here would abort valid sweeps; a once-per-env warning surfaces
        # a real mis-inversion without that cost.
        if not getattr(env, "_ch_range_warned", False) and not (
                -0.05 <= g <= 1.05):
            env._ch_range_warned = True
            print(f"  !! r_geo = {g!r} is outside [0, 1]; the geometry reward "
                  f"is cosine-derived and should not be.  Check that the "
                  f"mixture being inverted is the one the env applies.")
        env._ch_cur_task += t
        env._ch_cur_geo += g
        env._ch_step_task += t
        env._ch_step_geo += g
        env._ch_n += 1
        if done:                                 # mirrors ep_returns_all.append
            env._ch_ep_task.append(env._ch_cur_task)
            env._ch_ep_geo.append(env._ch_cur_geo)
            env._ch_cur_task = env._ch_cur_geo = 0.0
        return out

    env._geometry_reward = geo_wrapper
    env.step = step_wrapper
    env._ch_instrumented = True
    return w0


def _reset_channels(env) -> None:
    env._ch_ep_task, env._ch_ep_geo = [], []
    env._ch_cur_task = env._ch_cur_geo = 0.0
    env._ch_step_task = env._ch_step_geo = 0.0
    env._ch_n = 0
    env._ch_geo_last = 0.0           # never carry a reading across iterations


def instrument_trainer(trainer, env, w, horizon: int, strict: bool = True):
    """Per-iteration channel means land on `trainer._ch_hist` as (r_task, r_geo).

    `w` may be None, in which case env.geo_weight is read each iteration.

    Each entry satisfies, exactly,
        (1-w)*r_task*horizon + w*r_geo*horizon == main.py's own mean_ep
    and `strict` verifies it every iteration.  See the module docstring for
    what that check does and does not cover.
    """
    if getattr(trainer, "_ch_trainer_instrumented", False):
        # Wrapping twice would append two entries per iteration (verified: 8
        # entries for 4 iterations), doubling n in every channel
        # statistic.  Reset the history instead of re-wrapping.
        trainer._ch_hist, trainer._ch_recon_err = [], []
        return trainer
    trainer._ch_hist = []
    trainer._ch_recon_err = []
    trainer._ch_trainer_instrumented = True
    orig_gather = trainer._gather_trajectory

    def gather_wrapper(*a, **kw):
        _reset_channels(env)
        out = orig_gather(*a, **kw)

        if env._ch_ep_task:                          # completed episodes exist
            ep_t = float(np.mean(env._ch_ep_task))
            ep_g = float(np.mean(env._ch_ep_geo))
        else:                                        # main.py's own fallback:
            n = max(env._ch_n, 1)                    # per-step mean * horizon
            ep_t = env._ch_step_task / n * horizon
            ep_g = env._ch_step_geo / n * horizon

        ww = float(env.geo_weight) if w is None else float(w)
        recon = (1.0 - ww) * ep_t + ww * ep_g
        truth = float(out[6])                        # main.py's own mean_ep
        err = abs(recon - truth)
        trainer._ch_recon_err.append(err)
        tol = RECON_TOL * max(1.0, abs(truth))
        # not `err > tol`: a diverged run gives err = NaN, and `nan > tol` is
        # False, so that spelling would let a NaN row through as a good result.
        if strict and not (err <= tol):
            raise ValueError(
                f"channel reconstruction disagrees with main.py's episode "
                f"return: recon={recon!r} true={truth!r} err={err:.3e}. "
                f"Refusing to write numbers that do not decompose the "
                f"reported AUC.")

        trainer._ch_hist.append((ep_t / horizon, ep_g / horizon))
        return out

    trainer._gather_trajectory = gather_wrapper
    return trainer


def channels(res: dict, trainer) -> list:
    """Per-iteration (r_task, r_geo) pairs for a finished run."""
    if "r_task_per_iter" in res and "r_geo_per_iter" in res:
        return list(zip(res["r_task_per_iter"], res["r_geo_per_iter"]))
    hist = getattr(trainer, "_ch_hist", [])
    if not hist:
        raise RuntimeError("No channel data captured; the wrappers did not "
                           "fire.  Did you call instrument_trainer()?")
    return hist


def channel_aucs(pairs, w: float, horizon: int) -> tuple:
    """(auc_task, auc_geo, auc_total).

    Both channels go through the same trapezoid main.py uses, so the three
    numbers are additive: auc_task + auc_geo == auc_total to machine precision.
    Reconstructing them as horizon * mean-per-step instead would differ by the
    trapezoid's half-weighted endpoints and break additivity.
    """
    task = [(1.0 - w) * rt * horizon for rt, _ in pairs]
    geo = [w * rg * horizon for _, rg in pairs]
    total = [a + b for a, b in zip(task, geo)]
    return auc_trapezoid(task), auc_trapezoid(geo), auc_trapezoid(total)


# =========================================================================== #
# 3. Arm construction and training
# =========================================================================== #
def make_random_projector(k: int, dim: int, seed: int, device, dtype):
    """Frobenius-orthogonal projector onto a uniformly random `dim`-dim
    subspace of R^{k x k}.  This is the R-6 dimension-matched control.

    Returns a plain function of one argument, so attaching it to an object does
    not bind `self` -- main.py dispatches via
    `getattr(self.ops, f"project_{alg}")(X)`, which would pass self otherwise.
    Differentiable, because the auxiliary loss backpropagates through it.

    Caveat: a uniformly random 496-dim subspace of
    R^{32x32} generically has a large symmetric component, so exp(P_rand(theta))
    is not an isometry.  This arm controls for dimension while also varying
    compactness.  It is a dimension-matched control, not a dimension-only one.
    """
    import torch

    n = k * k
    if not 1 <= dim <= n:
        raise ValueError(f"dim must be in [1, {n}], got {dim}")

    g = torch.Generator(device="cpu").manual_seed(int(seed))
    A = torch.randn(n, dim, generator=g, dtype=torch.float64)
    Q, _ = torch.linalg.qr(A, mode="reduced")
    B = Q.to(device=device, dtype=dtype)

    # Verify orthonormality at the dtype that will actually run, not at the
    # float64 intermediate: float32 gives ~1e-6, and that is the number that
    # governs the projection error during training.
    eye = torch.eye(dim, device=B.device, dtype=B.dtype)
    err = float((B.T @ B - eye).abs().max().item())
    tol = 1e-4 if B.dtype == torch.float32 else 1e-9
    if not math.isfinite(err) or err > tol:
        raise RuntimeError(f"basis not orthonormal at {B.dtype}: "
                           f"max|BtB - I| = {err:.3e} > {tol:.0e}")

    def project_random(X):
        shape = X.shape
        flat = X.reshape(-1, n)
        return ((flat @ B) @ B.T).reshape(shape)

    return project_random, B, err


def attach_random_arm(policy, trainer, k: int, dim: int, seed: int) -> float:
    """Give this policy/trainer pair a private random-subspace projection.

    Attachment is per-object: policy.ops and trainer.ops are separate instances
    built in their own __init__, so LieAlgebraOps itself is never touched and
    no other arm can see this projector.
    """
    import torch

    proj, _, orth_err = make_random_projector(
        k, dim, seed, policy.theta.device, policy.theta.dtype)
    policy.ops.project_random = proj
    trainer.ops.project_random = proj
    policy.algebra = trainer.algebra = "random"
    # Start inside the subspace.  Provably a no-op for the trajectory (every
    # consumer projects anyway), but it makes iteration-0 diagnostics meaningful.
    with torch.no_grad():
        policy.theta.data.copy_(proj(policy.theta.data))
    return orth_err


def build(env, arm: str, seed: int, state_dim: int, n_actions: int, k: int,
          theta_lr=None, geo_aux_coef=None):
    """Construct (policy, trainer, cfg) for `arm`, seeded immediately beforehand.

    Re-seeding happens here, right before construction, so no arm inherits
    another's RNG state.  That inheritance is exactly why main.py's own seed-0
    so(32) score differs between its two runners (`_single_run` seeds once and
    trains the baseline first; `run_rl_ablation` does not), a 0.245 AUC swing
    on RNG ordering alone.
    """
    from main import (LiePolicy, BaselinePolicy, ValueNet, LieStructuredPPO,
                      _default_cfg, _seed_all)

    if theta_lr is not None and arm == BASELINE:
        raise ValueError(
            "theta_lr is meaningless for the baseline arm: it has no theta, "
            "and LieStructuredPPO drops cfg.theta_lr entirely on that branch. "
            "Passing it would add a grid dimension that has no effect.")

    _seed_all(seed)
    cfg = _default_cfg()
    if theta_lr is not None:
        cfg.theta_lr = float(theta_lr)
    if geo_aux_coef is not None:
        cfg.geo_aux_coef = float(geo_aux_coef)

    if arm == BASELINE:
        policy = BaselinePolicy(state_dim, n_actions)
        trainer = LieStructuredPPO(env, policy, ValueNet(state_dim), cfg,
                                   use_lie_projection=False)
    else:
        if arm not in ALGEBRA_OF and arm != "random":
            raise ValueError(f"unknown arm {arm!r}; expected {BASELINE}, "
                             f"'random', or one of {sorted(ALGEBRA_OF)}")
        # The random arm is initialised as so(32) and then re-projected, so its
        # initialisation distribution matches the arm it is controlling for.
        alg = "so" if arm == "random" else ALGEBRA_OF[arm]
        policy = LiePolicy(state_dim, n_actions, k=k, algebra=alg)
        trainer = LieStructuredPPO(env, policy, ValueNet(state_dim), cfg,
                                   use_lie_projection=True, algebra=alg)
        if arm == "random":
            attach_random_arm(policy, trainer, k, k * (k - 1) // 2,
                              seed=100_000 + seed)
    return policy, trainer, cfg


def run_one(env, arm, seed, state_dim, n_actions, k, w, horizon,
            theta_lr=None, geo_aux_coef=None, extra=None):
    """Train one arm and return (record, channel_pairs, policy).

    Never raises for a training failure -- the failure is recorded in
    `record["status"]` so the sweep continues and the other arms survive.
    (sym(32) can reach spectral radius > 1e8, at which point matrix_exp
    overflows and torch.linalg.eigvals aborts rather than returning NaN.)
    KeyboardInterrupt is not swallowed.
    """
    rec = dict(arm=arm, seed=seed, status="ok")
    if extra:
        rec.update(extra)
    try:
        policy, trainer, _ = build(env, arm, seed, state_dim, n_actions, k,
                                   theta_lr, geo_aux_coef)
        instrument_trainer(trainer, env, None, horizon, strict=True)
        res = trainer.train(verbose=False)
    except KeyboardInterrupt:
        raise
    except Exception as exc:                                 # noqa: BLE001
        rec.update(status=f"{type(exc).__name__}: {exc}"[:120])
        return rec, None, None

    pairs = channels(res, trainer)
    r_task = float(np.mean([p[0] for p in pairs]))
    r_geo = float(np.mean([p[1] for p in pairs]))
    a_task, a_geo, a_total = channel_aucs(pairs, w, horizon)
    auc = float(res["auc"])
    spec = [s for s in res.get("spectral_radii", [])
            if s is not None and math.isfinite(s)]
    finite = all(math.isfinite(x) for x in (r_task, r_geo, auc))

    rec.update(auc=auc, auc_task=a_task, auc_geo=a_geo,
               r_task=r_task, r_geo=r_geo,
               rho=float(np.median(spec)) if spec else np.nan,
               recon_err=abs(a_total - auc),
               final_return=float(res["returns"][-1]),
               status="ok" if finite else "non-finite")
    return rec, pairs, policy


# =========================================================================== #
# 4. Bookkeeping
# =========================================================================== #
class Sink:
    """Append-as-you-go CSV writer, so a crash never costs completed runs.

    Truncates unconditionally on construction.  `guard_outputs()` is what
    protects a previous sweep, and every caller must run it first, before the
    expensive model load.  There is no resume path by design: after a crash you
    re-run with --overwrite, which is why the guard exists.
    """

    def __init__(self, path: str, cols: list):
        self.path, self.cols, self.n = path, cols, 0
        pd.DataFrame(columns=cols).to_csv(path, index=False)

    def add(self, rows) -> None:
        if not rows:
            return
        pd.DataFrame(rows, columns=self.cols).to_csv(
            self.path, mode="a", header=False, index=False)
        self.n += len(rows)


def guard_outputs(paths, overwrite: bool) -> None:
    """Refuse to truncate a previous sweep.  Call before loading the model."""
    for p in paths:
        if os.path.exists(p) and os.path.getsize(p) > 0 and not overwrite:
            sys.exit(f"{p} exists and is non-empty.  Refusing to truncate a "
                     f"previous sweep.\nMove it aside, or re-run with "
                     f"--overwrite.")


def _main_hash() -> str:
    try:
        with open("main.py", "rb") as fh:
            return hashlib.md5(fh.read()).hexdigest()[:12]
    except OSError:
        return "unavailable"


def write_meta(csv_path: str, meta: dict) -> None:
    """Record the settings a later --analyse-only cannot otherwise know."""
    meta = dict(meta)
    meta["main_py_md5"] = _main_hash()
    with open(csv_path + ".meta.json", "w") as fh:
        json.dump(meta, fh, indent=2, default=str)


def check_meta(csv_path: str, expected: dict) -> None:
    """Refuse to analyse a CSV produced under different settings.

    Without this, `--analyse-only` on a stale file would report numbers
    from a different grid, a different geometry weight, or a different main.py.
    """
    p = csv_path + ".meta.json"
    if not os.path.exists(p):
        print(f"!! WARNING: no {p}; cannot verify that {csv_path} was produced "
              f"under the settings this analysis assumes.")
        return
    with open(p) as fh:
        got = json.load(fh)
    bad = {k: (got.get(k), v) for k, v in expected.items()
           if str(got.get(k)) != str(v)}
    if bad:
        lines = "\n".join(f"    {k}: file={a!r}  now={b!r}"
                          for k, (a, b) in bad.items())
        sys.exit(f"{csv_path} was produced under different settings:\n{lines}\n"
                 f"Re-run the sweep, or analyse it with the matching version.")
    cur = _main_hash()
    if got.get("main_py_md5") not in (cur, "unavailable") and cur != "unavailable":
        print(f"!! WARNING: main.py changed since {csv_path} was written "
              f"({got.get('main_py_md5')} -> {cur}).")


def clean(df: pd.DataFrame, col: str, by: str, arms) -> pd.DataFrame:
    """Drop non-finite rows and any block missing an arm, keeping the design
    paired.  `by` is the blocking variable (seed, or environment draw)."""
    d = df[df.status == "ok"].copy()
    # Restrict to the requested arms first.  Without this, a frame carrying an
    # extra arm makes nunique() exceed len(arms) for every block, so no block
    # is "full" and the function returns zero rows.
    extra = sorted(set(d["arm"]) - set(arms))
    if extra:
        print(f"  ignoring {len(extra)} arm(s) not in this comparison: {extra}")
        d = d[d["arm"].isin(list(arms))]
    dup = d.duplicated([by, "arm"]).sum()
    if dup:
        print(f"  !! {dup} duplicated ({by}, arm) row(s); keeping the last of "
              f"each.  Two partial sweeps were probably concatenated.")
        d = d.drop_duplicates([by, "arm"], keep="last")
    bad = ~np.isfinite(pd.to_numeric(d[col], errors="coerce"))
    if bad.any():
        print(f"  dropping {int(bad.sum())} non-finite '{col}' row(s)")
        d = d.loc[~bad]
    counts = d.groupby(by)["arm"].nunique()
    full = counts[counts == len(arms)].index
    lost = sorted(set(d[by]) - set(full))
    if lost:
        print(f"  dropping {len(lost)} incomplete block(s) to keep the design "
              f"paired: {lost}")
    return d[d[by].isin(full)]


def paired(df: pd.DataFrame, a: str, b: str, col: str = "auc",
           by: str = "seed") -> np.ndarray:
    """a - b, matched on the blocking variable.  Returns the difference vector."""
    xa, yb = df[df.arm == a], df[df.arm == b]
    # A duplicated blocking value would make the join cartesian and inflate n
    # (verified: 3 duplicated seeds gave 3 differences instead of 1).  Keep the
    # last row per block, matching clean()'s convention, and say so.
    for name, frame in (("a", xa), ("b", yb)):
        if frame.duplicated([by]).any():
            print(f"  !! duplicated '{by}' values in arm {a if name == 'a' else b!r}; "
                  f"keeping the last of each")
    x = xa.drop_duplicates([by], keep="last").set_index(by)[col].astype(float)
    y = yb.drop_duplicates([by], keep="last").set_index(by)[col].astype(float)
    idx = x.index.intersection(y.index)
    d = (x.loc[idx] - y.loc[idx]).replace([np.inf, -np.inf], np.nan).dropna()
    return d.to_numpy()


def report(rows, title: str, note: str = "") -> list:
    """Holm-correct a family of paired comparisons and print it.

    `rows` is a list of dicts each carrying a 'label' and a 'd' (difference
    vector).

    Degenerate comparisons (n < 2, zero variance, non-finite p) are printed as
    untestable and are excluded from the Holm family entirely.  Entering them
    with p = 1 would not be neutral: Holm's multiplier is the family size, so a
    single untestable row could flip a borderline pair from rejected to
    not-rejected.  A comparison that carries no information should not consume
    alpha from the ones that do.
    """
    out, pvals, notes = [], [], []
    testable = []                    # indices into `out` that enter the family
    for r in rows:
        d = np.asarray(r["d"], dtype=float)
        meta = {k: v for k, v in r.items() if k != "d"}
        rep = paired_report(d)
        if "degenerate" in rep:
            out.append(dict(**meta, **rep))
            if rep.get("degenerate") == "zero variance":
                notes.append(f"{meta.get('label', '?')}: all {rep['n']} paired "
                             f"differences are identical ({rep['diff']:+.4g}); "
                             f"no test is possible")
            continue
        testable.append(len(out))
        out.append(dict(**meta, **rep))
        pvals.append(rep["p"])

    if pvals:
        hp = holm(pvals, 0.05)
        for i, ph, rj in zip(testable, hp["p_holm"], hp["reject"]):
            out[i]["p_holm"], out[i]["sig"] = float(ph), bool(rj)

    print("\n" + "=" * 96)
    n_deg = len(out) - len(pvals)
    print(f"{title}   Holm-corrected over {len(pvals)} comparison(s)"
          + (f"  ({n_deg} untestable, excluded from the family)" if n_deg else ""))
    if note:
        print(note)
    print("=" * 96)
    for o in out:
        lab = o.get("label", "")
        if o.get("degenerate"):
            v = f"  diff={o['diff']:+.4f}" if "diff" in o else ""
            print(f"  {lab:<44s} n={o['n']}  UNTESTABLE ({o['degenerate']}){v}")
            continue
        pw = o["power"] if o["power"] is not None else float("nan")
        print(f"  {lab:<44s} n={o['n']:>2d}  {o['diff']:>+9.4f}"
              f"{'*' if o['sig'] else ' '} "
              f"[{o['ci'][0]:>+8.4f},{o['ci'][1]:>+8.4f}]  dz={o['dz']:>+6.2f}  "
              f"p_holm={o['p_holm']:>9.2e}  power={pw:.2f}")
    for nt in notes:
        print(f"  !! {nt}")
    print("  * = significant after Holm.  'power' is post-hoc power at the "
          "observed effect --\n  a restatement of p that says nothing about a null. "
          "For nulls read the CI and TOST.")
    return out


# =========================================================================== #
# Self-test
# =========================================================================== #
def selftest() -> None:
    """Verify the pieces that everything else depends on.  No GPU, no model."""
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  {'PASS' if cond else 'FAIL'}  {name}"
              f"{'  ' + detail if detail else ''}")

    print("(1) statistics")
    rng = np.random.default_rng(0)
    d = rng.normal(0.5, 1.0, 40)
    r = paired_report(d)
    check("paired_report returns a finite CI containing the mean",
          r["ci"][0] < r["diff"] < r["ci"][1])
    check("an exact null is reported, not raised",
          paired_report(np.zeros(10)).get("degenerate") == "zero variance")
    try:
        holm([0.01, float("nan")])
        check("a NaN p-value raises in holm()", False)
    except ValueError:
        check("a NaN p-value raises in holm()", True)
    hp = holm([0.01, 0.02, 0.04])
    check("Holm step-down is monotone", list(hp["p_holm"]) == sorted(hp["p_holm"]))
    check("Holm rejects at exactly p_holm == alpha",
          holm([0.05 / 1])["reject"][0])
    t = tost_paired(rng.normal(0.0, 0.05, 60), margin=0.5)
    check("a tight null is certified equivalent", t["equivalent"])
    t2 = tost_paired(rng.normal(0.0, 2.0, 8), margin=0.1)
    check("a wide null is not certified equivalent", not t2["equivalent"])
    try:
        tost_paired(np.zeros(10), margin=0.1)
        check("zero-variance TOST raises", False)
    except ValueError:
        check("zero-variance TOST raises", True)
    check("n_for_power at dz=0.69 is 19 seeds (matches Task 2)",
          n_for_power_paired(0.69, 0.8) == 19,
          f"got {n_for_power_paired(0.69, 0.8)}")
    check("auc_trapezoid of a constant-1 curve over 60 iters = 59/60",
          abs(auc_trapezoid([1.0] * 60) - 59 / 60) < 1e-12)

    print("(2) channel recovery, exactly, under a changing geo_weight")

    class StubEnv:
        """Mimics main.py's reward structure and episode accounting."""
        horizon = 20

        def __init__(self, w):
            self.geo_weight = w
            self.truth, self._t = [], 0

        def _geometry_reward(self, v, M):
            return 0.5 + 0.004 * (len(self.truth) % 20)

        def step(self, a, M=None):
            terminal = (self._t == self.horizon - 1)
            t = 0.9 - 0.05 * (len(self.truth) % 4)
            if not terminal and (len(self.truth) % 3 == 0):
                t = 0.0                                   # the sparsity mask
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
                _, r, done = self.env.step(0)
                cur += r
                if done:                                  # main.py's accounting
                    rets.append(cur)
                    cur = 0.0
            return (None,) * 6 + (float(np.mean(rets)) if rets else 0.0,)

        def train(self):
            eps = [self._gather_trajectory()[6] for _ in range(self.n_iters)]
            return {"auc": auc_trapezoid(eps)}

    for w in (0.0, 0.2, 0.4, 0.6, 0.8):
        e = StubEnv(w)
        instrument(e)
        tr = StubTrainer(e, 512, 6)                       # 512 = 25 eps + 12 orphans
        instrument_trainer(tr, e, None, e.horizon, strict=True)
        res = tr.train()
        pairs = channels(res, tr)
        _, _, total = channel_aucs(pairs, w, e.horizon)
        check(f"w={w:.1f}: auc_task + auc_geo == the reported AUC",
              abs(total - res["auc"]) < 1e-9,
              f"|diff|={abs(total - res['auc']):.1e}")

    e = StubEnv(0.4)
    instrument(e)
    seen = []
    for w in (0.0, 0.8, 0.2):                             # mutate between blocks
        e.geo_weight = w
        tr = StubTrainer(e, 512, 3)
        instrument_trainer(tr, e, None, e.horizon, strict=True)
        r = tr.train()
        seen.append(abs(channel_aucs(channels(r, tr), w, e.horizon)[2] - r["auc"]))
    check("exact at every w after mutating geo_weight in place",
          max(seen) < 1e-9, f"max |diff|={max(seen):.1e}")

    class BadEnv:
        geo_weight, horizon = 0.4, 20

        def _geometry_reward(self, v, M):
            return float("nan")

        def step(self, a, M=None):
            return None, float("nan"), True

    class BadTrainer:
        def __init__(self, env):
            self.env = env

        def _gather_trajectory(self):
            for _ in range(20):
                self.env.step(0)
            return (None,) * 6 + (1.0,)

    be = BadEnv()
    instrument(be)
    bt = BadTrainer(be)
    instrument_trainer(bt, be, None, be.horizon, strict=True)
    try:
        bt._gather_trajectory()
        check("a NaN reconstruction raises instead of passing", False)
    except (ValueError, RuntimeError):
        check("a NaN reconstruction raises instead of passing", True)

    print("(3) bookkeeping")
    df = pd.DataFrame([{"arm": a, "seed": s, "auc": 1.0 + s,
                        "status": "ok"}
                       for a in ("so", "sl", "gl") for s in range(5)])
    df.loc[(df.arm == "sl") & (df.seed == 3), "auc"] = np.inf
    cleaned = clean(df, "auc", "seed", ("so", "sl", "gl"))
    check("inf rows dropped and the design kept paired",
          len(cleaned) == 12 and cleaned.seed.nunique() == 4,
          f"{len(cleaned)} rows, {cleaned.seed.nunique()} seeds")
    check("paired() matches on the blocking variable",
          len(paired(cleaned, "so", "gl")) == 4)
    check("equivalent() needs the whole CI inside the margin",
          equivalent({"ci": (-0.1, 0.1)}, 0.2)
          and not equivalent({"ci": (-0.3, 0.1)}, 0.2))

    print("\n" + ("SELFTEST PASS" if ok else "SELFTEST FAIL"))
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    selftest()
