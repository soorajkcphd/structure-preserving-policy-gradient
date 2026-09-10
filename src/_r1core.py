"""
Shared channel instrumentation for R-1 / R-1b / R-4 / R-6.

main.py averages the return over completed episodes only: `_gather_trajectory`
runs `steps_per_iter` (512) steps but appends to `ep_returns_all` solely when an
episode terminates, so the trailing partial episode (512 - 25*20 = 12 steps) is
discarded.  Those 12 steps are always non-terminal, and non-terminal steps are
the only ones subject to the 0.7 sparsity mask, so including them would bias
the task channel downward and make auc_task + auc_geo a different estimand from
the AUC main.py reports.

This module mirrors main.py's episode accounting exactly and then verifies it at
runtime: every iteration it reconstructs main.py's own `mean_ep` from the two
channels and aborts if they disagree.  The decomposition is therefore checked
against ground truth 60 times per arm rather than argued for once.
"""
from __future__ import annotations

import numpy as np

RECON_TOL = 1e-6


def instrument(env) -> float:
    """Record r_task / r_geo per step, with episode boundaries. Idempotent.

    Returns the environment's geo_weight.  Changes no reward and consumes no
    randomness.
    """
    if getattr(env, "_r1_instrumented", False):
        return float(env.geo_weight)

    w0 = float(env.geo_weight)
    if not (0.0 <= w0 < 1.0):
        raise ValueError(
            f"geo_weight={w0}: the task channel is not identifiable.")

    env._r1_ep_task, env._r1_ep_geo = [], []      # completed-episode sums
    env._r1_cur_task = env._r1_cur_geo = 0.0      # current episode
    env._r1_step_task = env._r1_step_geo = 0.0    # all steps (fallback only)
    env._r1_n = 0
    env._r1_geo_last = 0.0

    orig_geo = env._geometry_reward
    orig_step = env.step

    def geo_wrapper(*a, **kw):
        g = orig_geo(*a, **kw)
        env._r1_geo_last = float(g)
        return g

    def step_wrapper(*a, **kw):
        # Read geo_weight before the step, so the value used for attribution is
        # provably the one step() will use even if a caller mutates it.  Live
        # read (rather than capture at instrument time) lets a sweep vary it
        # between blocks; identical when it is constant.
        w = float(env.geo_weight)
        if not (0.0 <= w < 1.0):
            raise ValueError(
                f"geo_weight={w}: the task channel is not identifiable.")
        out = orig_step(*a, **kw)
        reward, done = float(out[1]), bool(out[2])
        g = env._r1_geo_last
        t = (reward - w * g) / (1.0 - w)
        env._r1_cur_task += t
        env._r1_cur_geo += g
        env._r1_step_task += t
        env._r1_step_geo += g
        env._r1_n += 1
        if done:                       # mirrors main.py's ep_returns_all.append
            env._r1_ep_task.append(env._r1_cur_task)
            env._r1_ep_geo.append(env._r1_cur_geo)
            env._r1_cur_task = env._r1_cur_geo = 0.0
        return out

    env._geometry_reward = geo_wrapper
    env.step = step_wrapper
    env._r1_instrumented = True
    return w0


def _reset(env) -> None:
    env._r1_ep_task, env._r1_ep_geo = [], []
    env._r1_cur_task = env._r1_cur_geo = 0.0
    env._r1_step_task = env._r1_step_geo = 0.0
    env._r1_n = 0


def instrument_trainer(trainer, env, w, horizon: int, strict: bool = True):
    """`w` may be None, in which case env.geo_weight is read at each iteration."""
    """Per-iteration channel means land on trainer._r1_hist as (r_task, r_geo).

    Each entry satisfies, exactly:
        (1-w)*r_task*horizon + w*r_geo*horizon == main.py's mean_ep
    which is verified every iteration when `strict`.
    """
    trainer._r1_hist = []
    trainer._r1_recon_err = []
    orig_gather = trainer._gather_trajectory

    def gather_wrapper(*a, **kw):
        _reset(env)
        out = orig_gather(*a, **kw)

        if env._r1_ep_task:                       # completed episodes exist
            ep_t = float(np.mean(env._r1_ep_task))
            ep_g = float(np.mean(env._r1_ep_geo))
        else:                                     # main.py falls back to the
            n = max(env._r1_n, 1)                 # per-step mean over all steps
            ep_t = env._r1_step_task / n * horizon
            ep_g = env._r1_step_geo / n * horizon

        ww = float(env.geo_weight) if w is None else float(w)
        mean_ep_recon = (1.0 - ww) * ep_t + ww * ep_g
        mean_ep_true = float(out[6])              # main.py's own value
        err = abs(mean_ep_recon - mean_ep_true)
        trainer._r1_recon_err.append(err)
        tol = RECON_TOL * max(1.0, abs(mean_ep_true))
        # not `err > tol`: if the run diverged, err is NaN and `nan > tol` is
        # False, so the guard would pass and a NaN row would be
        # recorded as a good result.
        if strict and not (err <= tol):
            raise ValueError(
                "channel reconstruction disagrees with main.py's episode "
                f"return: recon={mean_ep_recon!r} true={mean_ep_true!r} "
                f"err={err:.3e}.  Refusing to write numbers that do not "
                "decompose the reported AUC."
            )

        trainer._r1_hist.append((ep_t / horizon, ep_g / horizon))
        return out

    trainer._gather_trajectory = gather_wrapper
    return trainer


def channels(res: dict, trainer) -> list:
    """Prefer patched-source columns; else the runtime wrappers."""
    if "r_task_per_iter" in res and "r_geo_per_iter" in res:
        return list(zip(res["r_task_per_iter"], res["r_geo_per_iter"]))
    hist = getattr(trainer, "_r1_hist", [])
    if not hist:
        raise RuntimeError("No channel data captured; the wrappers did not fire.")
    return hist


def channel_aucs(pairs, w: float, horizon: int) -> tuple:
    """auc_task, auc_geo, auc_total -- same construction channel_decomposition
    uses, so R-1 and R-4 are directly comparable."""
    from sppg_defense.rl.analysis import auc_trapezoid
    task = [(1.0 - w) * rt * horizon for rt, _ in pairs]
    geo = [w * rg * horizon for _, rg in pairs]
    tot = [a + b for a, b in zip(task, geo)]
    return (float(auc_trapezoid(task)), float(auc_trapezoid(geo)),
            float(auc_trapezoid(tot)))
