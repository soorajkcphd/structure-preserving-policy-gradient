"""
sppg_experiments.py -- every SP-PG revision experiment, in one place.
====================================================================

Thirteen experiments that together determine what the results support.  All of
them sit on `sppg_core`, so the statistics, the reward-channel accounting and
the arm construction are shared and tested once.  None of them edits main.py,
04_pap_CG.py or run_sentiment_task3.py.

    python sppg_experiments.py --list
    python sppg_experiments.py --selftest
    python sppg_experiments.py r4  --seeds 20
    python sppg_experiments.py b1lr --seeds 10

Each experiment writes `<name>_cells.csv` plus a `.meta.json` recording the
grid, the geometry weight and an MD5 of main.py, so a later analysis run can
refuse a stale file.  Analysis lives in `sppg_analysis.py`; this file only
collects.  The two are separate because collection needs a GPU and an hour,
analysis needs neither, and you will re-run the analysis many times.

--------------------------------------------------------------------------
The experiments, and what each one decides
--------------------------------------------------------------------------
Task 1 (GPT-2 medium, MultiStepTextAlignmentEnv):

  r1     Channel decomposition of the headline gap.  Splits the AUC into the
         task channel and the geometry channel.  Decides: how much of the
         +34.2% is a reward the M=I control cannot reach.

  r1b    The (w, c_geo) trade-off surface.  Decides: whether the task-channel
         cost is tunable away, and what happens with the auxiliary loss OFF.

  r4     Five-arm algebra ablation at 20 seeds, incl. the dimension-matched
         random(496) control.  This is the source of Table `tab:ablation`.
         Decides: the ordering of parameterisations.

  r5     Per-algebra learning-rate sweep, leave-one-seed-out tuned.
         Decides: whether that ordering is a shared-learning-rate artefact.

  x3     20 independent draws of the planted rotation M_env.
         Decides: whether the ordering generalises beyond one draw.

  r7     5 environment geometries x 5 policy algebras.
         Decides: whether the claim is about compactness or about matching
         whatever structure the environment happens to have.  It determines
         the scope of the result.

  b1     so(32) via the matrix exponential vs a Cayley retraction.
         Decides: whether the effect is the constraint or the specific map.

  b1lr   b1 again, with each map at its own leave-one-seed-out best rate.
         Decides: whether b1's gap was a shared-rate artefact.

  s7     Auxiliary target swapped: M_env / a fixed foreign rotation / a
         per-seed rotation / no auxiliary loss.
         Decides: prior, or supervision.

Task 2 (GPT-2, structure-informed REINFORCE):

  r11    Held-out prompts, plus an independent recomputation of the V2 check.
         Decides: whether Section 9's effect is in-sample only, and whether
         V2 = 0.99993 is a measurement or a tautology.

Task 3 (GPT-2 medium, sentiment steering, no engineered geometry):

  r15    Positive control: inject known geometry at strength w.
         Decides: what effect the Task-3 design could have detected.

  r15b   r15 with the auxiliary target enabled.
         Decides: whether s7's dichotomy replicates in a second environment.

Cross-model:

  r13b   Mistral-7B at 20 seeds, five arms, channel-decomposed.
         Decides: whether the abstract's +59.9% means what +34.2% means.
--------------------------------------------------------------------------

A note on what is and is not comparable
---------------------------------------
AUC is not comparable across geometry weights (r1b) or across environment
geometries (r7): changing either changes what the reward means and therefore
the ceiling of the return.  Per-step channel means are comparable, because
r_task and r_geo each live in [0,1] whatever w is.  Where a table crosses one
of those boundaries the code says so in its own output.
"""
from __future__ import annotations

import argparse
import contextlib
import math
import random
import sys

import numpy as np
import pandas as pd

from sppg_core import (BASELINE, Sink, W_DEFAULT, HORIZON_DEFAULT,
                       channel_aucs, channels, guard_outputs, instrument,
                       instrument_trainer, run_one, write_meta)

# --------------------------------------------------------------------------- #
# Shared grids and constants
# --------------------------------------------------------------------------- #
ALL_ARMS = (BASELINE, "so", "sl", "gl", "sym")
LR_GRID = (3e-4, 3e-3, 0.01, 0.03, 0.1)     # feature-net rate .. 3x the paper's
PAPER_LR = 0.03
MIN_CELL_N = 3          # a cell needs this many seeds to be eligible as "best"

BASE_COLS = ["arm", "seed", "auc", "auc_task", "auc_geo", "r_task", "r_geo",
             "rho", "recon_err", "status"]


def _task1_env(w: float = W_DEFAULT, horizon: int = HORIZON_DEFAULT):
    """The Task-1 environment, instrumented.  Built once per experiment: the
    GPT-2 embedding pre-computation is expensive, and rebuilding would also
    redraw M_env, which must stay fixed within a sweep."""
    from main import GPT2EmbeddingProvider, MultiStepTextAlignmentEnv

    emb = GPT2EmbeddingProvider(model_name="gpt2-medium", max_length=64)
    env = MultiStepTextAlignmentEnv(emb, n_prompts=16, n_actions=16,
                                    horizon=horizon, reward_noise=0.2,
                                    geo_weight=w, sparse_prob=0.7)
    w0 = instrument(env)
    return env, emb.hidden_dim, env.n_actions, int(env.k_transform), w0


def _progress(done, total, label, rec):
    msg = (f"FAILED: {rec['status']}" if rec["status"] != "ok"
           else f"AUC={rec['auc']:7.3f}  r_task={rec['r_task']:.4f}  "
                f"r_geo={rec['r_geo']:.4f}  rho={rec.get('rho', float('nan')):.4g}")
    print(f"  [{done:3d}/{total}] {label:28s} {msg}", flush=True)


# =========================================================================== #
# R-1 / R-4 / R-6 -- the algebra ablation and its channel decomposition
# =========================================================================== #
def run_r4(n_seeds: int = 20, overwrite: bool = False, **_) -> pd.DataFrame:
    """Five-arm ablation at n seeds, plus baseline PPO for the R-1 contrast.

    Arms:
        so(32)       compact Lie          dim  496
        sl(32)       non-compact Lie      dim 1023
        sym(32)      non-Lie subspace     dim  528
        gl(32)       unconstrained        dim 1024
        random(496)  random subspace      dim  496   <- dimension-matched
        baseline_ppo no M at all                     <- R-1 contrast

    The random arm redraws its subspace every seed, so its SD carries
    subspace-draw variance on top of training variance.  That is conservative
    for "so(32) beats random" and anti-conservative for any claim that the two
    are equivalent.  Quote the SD.

    Re-seeding per arm gives identical initialisation but not common random
    numbers: the number of np.random draws per step depends on whether
    task_r > 0, so the streams desynchronise once actions diverge.  This is a
    randomised block design, not a CRN design.
    """
    csv = "r4_cells.csv"
    guard_outputs([csv], overwrite)
    env, sd, na, k, w = _task1_env()
    arms = ["so", "sl", "sym", "gl", "random", BASELINE]

    sink, rows = Sink(csv, BASE_COLS), []
    write_meta(csv, dict(arms=arms, w=w, horizon=HORIZON_DEFAULT,
                         n_seeds=n_seeds))
    total, done = len(arms) * n_seeds, 0
    for seed in range(n_seeds):
        for arm in arms:
            rec, _, _ = run_one(env, arm, seed, sd, na, k, w, HORIZON_DEFAULT)
            sink.add([{c: rec.get(c, np.nan) for c in BASE_COLS}])
            rows.append(rec)
            done += 1
            _progress(done, total, f"seed {seed:2d} {arm}", rec)
    print(f"\nWrote {csv} ({sink.n} rows)")
    return pd.DataFrame(rows)


# =========================================================================== #
# R-1b -- the (w, c_geo) trade-off surface
# =========================================================================== #
def run_r1b(n_seeds: int = 10, overwrite: bool = False, **_) -> pd.DataFrame:
    """How the task-channel cost depends on the two knobs that govern it.

        w      = env.geo_weight   (how much of the reward is geometric)
        c_geo  = cfg.geo_aux_coef (weight of the auxiliary loss on theta)

    Both were fixed at 0.4 and 1.0 in the released code and never tuned.

    The question: at c_geo = 0 the auxiliary loss is off and theta receives
    gradient only through the policy objective.  If the task cost vanishes
    there while a geometry gain survives, the trade-off belongs to the
    auxiliary loss and is tunable away.  If it persists, it is intrinsic to
    acting through M.

    c_geo spans two orders of magnitude because theta is optimised by Adam,
    which is approximately scale-invariant, so c_geo does not scale the update
    -- it only changes the ratio of auxiliary to surrogate gradient.  A narrow
    grid could return a flat row meaning "Adam absorbed the knob" rather than
    "the trade-off is insensitive to it".

    The baseline is run once per w, not once per cell: c_geo is a no-op without
    theta (main.py guards on hasattr(policy, "theta")).
    """
    csv = "r1b_cells.csv"
    guard_outputs([csv], overwrite)
    w_grid = (0.0, 0.2, 0.4, 0.6, 0.8)
    c_grid = (0.0, 0.05, 0.25, 1.0)
    cols = ["w", "c_geo"] + BASE_COLS

    env, sd, na, k, _ = _task1_env(w=w_grid[0])
    sink, rows = Sink(csv, cols), []
    write_meta(csv, dict(w_grid=list(w_grid), c_grid=list(c_grid),
                         horizon=HORIZON_DEFAULT, n_seeds=n_seeds))
    total = len(w_grid) * (len(c_grid) + 1) * n_seeds
    done = 0
    for w in w_grid:
        env.geo_weight = float(w)   # mutate, do not rebuild: M_env stays fixed
        plan = [(BASELINE, None)] + [("so", c) for c in c_grid]
        for seed in range(n_seeds):
            for arm, c_geo in plan:
                rec, _, _ = run_one(env, arm, seed, sd, na, k, w,
                                    HORIZON_DEFAULT, geo_aux_coef=c_geo,
                                    extra=dict(w=w, c_geo=(np.nan if c_geo is None
                                                           else c_geo)))
                sink.add([{c: rec.get(c, np.nan) for c in cols}])
                rows.append(rec)
                done += 1
                _progress(done, total, f"w={w:.1f} c={c_geo} s{seed}", rec)
    print(f"\nWrote {csv} ({sink.n} rows)")
    return pd.DataFrame(rows)


# =========================================================================== #
# R-5 -- per-algebra learning-rate sweep
# =========================================================================== #
def run_r5(n_seeds: int = 10, overwrite: bool = False, **_) -> pd.DataFrame:
    """Give every algebra its own learning rate and re-run the comparison.

    main.py uses theta_lr = 0.03 for every algebra, chosen (per its own comment)
    to make non-compact spectral growth visible within 60 iterations.  The
    obvious objection is that sl(32) and gl(32) lose because 0.03 is
    wrong for them, not because they are non-compact.

    Caveat: this grid varies theta_lr alone, and
    main.py's own comments say entropy_coef, geo_aux_coef and reward_threshold
    were each chosen to suit theta_lr = 0.03.  "Each algebra at its own best
    rate" means "best rate holding three co-tuned constants fixed".
    """
    csv = "r5_cells.csv"
    guard_outputs([csv], overwrite)
    env, sd, na, k, w = _task1_env()
    cols = ["arm", "theta_lr"] + BASE_COLS[1:]

    sink, rows = Sink(csv, cols), []
    write_meta(csv, dict(lr_grid=list(LR_GRID), algs=["so", "sl", "gl", "sym"],
                         w=w, horizon=HORIZON_DEFAULT, n_seeds=n_seeds))
    plan = [(BASELINE, np.nan)] + [(a, lr) for a in ("so", "sl", "gl", "sym")
                                   for lr in LR_GRID]
    total, done = len(plan) * n_seeds, 0
    for seed in range(n_seeds):
        for arm, lr in plan:
            rec, _, _ = run_one(env, arm, seed, sd, na, k, w, HORIZON_DEFAULT,
                                theta_lr=None if arm == BASELINE else lr,
                                extra={"theta_lr": lr})
            sink.add([{c: rec.get(c, np.nan) for c in cols}])
            rows.append(rec)
            done += 1
            _progress(done, total, f"{arm} lr={lr:g} s{seed}", rec)
    print(f"\nWrote {csv} ({sink.n} rows)")
    return pd.DataFrame(rows)


# =========================================================================== #
# X-3 -- robustness to the environment draw
# =========================================================================== #
def draw_rotation(k: int, seed: int, device=None, dtype=None):
    """A planted rotation built exactly as main.py builds M_env: the matrix
    exponential of a random skew matrix, hence in SO(k) by construction.

    Verified rather than assumed: the caller gets an exception if the drawn
    matrix is not orthogonal with det = +1.
    """
    import torch
    from scipy.linalg import expm

    rng = np.random.RandomState(int(seed))
    S = rng.randn(k, k) * 0.5
    S = 0.5 * (S - S.T)
    M = torch.tensor(expm(S), dtype=torch.float64)
    err = float((M.T @ M - torch.eye(k, dtype=torch.float64)).abs().max())
    det = float(torch.det(M))
    if not (err < 1e-9 and abs(det - 1.0) < 1e-9):
        raise RuntimeError(f"drawn rotation not in SO({k}): "
                           f"max|MtM-I|={err:.2e}, det={det:.6f}")
    if device is not None:
        M = M.to(device=device, dtype=dtype)
    return M


def run_x3(n_draws: int = 20, seeds_per_draw: int = 3,
           overwrite: bool = False, **_) -> pd.DataFrame:
    """Redraw M_env `n_draws` times and re-run the algebra comparison.

    The unit of replication is the environment draw, not the training seed:
    that is the quantity the claim needs to generalise over.  Training seeds
    are averaged within a draw so one lucky initialisation cannot ride along
    into every draw.

    Mutating M_env in place is safe: nothing in MultiStepTextAlignmentEnv is
    precomputed from it (prompt_embs, action_dirs, base_deltas and state_embs
    are all built before it and never reference it).  It is read in exactly two
    places -- the geometry reward and the auxiliary loss -- and a redrawn M_env
    is a different environment, so we change both.
    """
    csv = "x3_cells.csv"
    guard_outputs([csv], overwrite)
    env, sd, na, k, w = _task1_env()
    cols = ["draw", "m_env_seed"] + BASE_COLS
    orig = env.M_env.clone()

    sink, rows = Sink(csv, cols), []
    write_meta(csv, dict(arms=list(ALL_ARMS), w=w, horizon=HORIZON_DEFAULT,
                         n_draws=n_draws, seeds_per_draw=seeds_per_draw))
    total = n_draws * len(ALL_ARMS) * seeds_per_draw
    done = 0
    try:
        for draw in range(n_draws):
            ms = 900_000 + draw
            env.M_env = draw_rotation(k, ms, orig.device, orig.dtype)
            for s in range(seeds_per_draw):
                for arm in ALL_ARMS:
                    rec, _, _ = run_one(env, arm, s, sd, na, k, w,
                                        HORIZON_DEFAULT,
                                        extra={"draw": draw, "m_env_seed": ms})
                    sink.add([{c: rec.get(c, np.nan) for c in cols}])
                    rows.append(rec)
                    done += 1
                    _progress(done, total, f"draw {draw:2d} s{s} {arm}", rec)
    finally:
        env.M_env = orig                # leave the env exactly as we found it
    print(f"\nWrote {csv} ({sink.n} rows)")
    return pd.DataFrame(rows)


# =========================================================================== #
# R-7 -- environment geometry x policy algebra
# =========================================================================== #
GEOMS = ("so", "sym", "gl", "diag", "id")


def menv_seed(gi: int, draw: int) -> int:
    """Seed for the M_env of geometry index `gi`, draw `draw`.

    Single source of truth: `run_r7` and `tools/check_r7_provenance.py` both
    call this, so the released CSVs and the provenance check cannot drift
    apart.

    Geometries are spaced by 1 and draws by 100, not the reverse.  Both layouts
    are collision-free (there are 5 geometries, far fewer than 100), but only
    this one gives 9000..9004 at draw 0 -- the seeds that produced the published
    Table tab:geomfactorial.  Spacing geometries by 100 instead would also be
    collision-free but would no longer reproduce the paper.
    """
    return 9000 + gi + 100 * draw


def draw_geometry(geom: str, k: int, seed: int):
    """(M, diagnostics) for one environment geometry, in float64.

    `so` reproduces main.py's own construction exactly; the others are the same
    construction with the skew-symmetrisation replaced, so the families are
    matched in scale and in RNG treatment and differ only in structure.

    Every family is checked to be what it claims, and checked not to duplicate
    another cell.  A factorial in which two columns are secretly the same
    matrix is not a factorial.
    """
    from scipy.linalg import expm

    rng = np.random.RandomState(int(seed))
    S = rng.randn(k, k) * 0.5
    if geom == "so":
        M = expm(0.5 * (S - S.T))               # exactly main.py's M_env
    elif geom == "sym":
        M = expm(0.5 * (S + S.T))               # SPD, emphatically not a rotation
    elif geom == "gl":
        M = S.copy()                            # generic invertible
    elif geom == "diag":
        M = np.diag(np.exp(0.5 * np.diag(S)))   # positive axis-aligned scaling
    elif geom == "id":
        M = np.eye(k)                           # no geometry to learn
    else:
        raise ValueError(f"unknown geometry {geom!r}; expected one of {GEOMS}")

    M = np.asarray(M, dtype=np.float64)
    I = np.eye(k)
    diag = dict(menv_orth=float(np.abs(M.T @ M - I).max()),
                menv_sym=float(np.abs(M - M.T).max()),
                menv_cond=float(np.linalg.cond(M)))
    if not np.isfinite(diag["menv_cond"]) or diag["menv_cond"] > 1e8:
        raise RuntimeError(f"{geom}: M_env is numerically singular "
                           f"(cond={diag['menv_cond']:.3e}).")
    if geom == "so" and diag["menv_orth"] > 1e-9:
        raise RuntimeError(f"so: not orthogonal ({diag['menv_orth']:.2e})")
    if geom in ("sym", "diag") and diag["menv_sym"] > 1e-9:
        raise RuntimeError(f"{geom}: not symmetric ({diag['menv_sym']:.2e})")
    if geom == "sym" and diag["menv_orth"] < 1e-3:
        raise RuntimeError("sym: came out orthogonal; it would duplicate the "
                           "so cell instead of contrasting with it.")
    if geom == "gl" and (diag["menv_orth"] < 1e-3 or diag["menv_sym"] < 1e-3):
        raise RuntimeError("gl: is orthogonal or symmetric; it would duplicate "
                           "another cell.")
    import torch
    return torch.tensor(M, dtype=torch.float64), diag


def run_r7(n_seeds: int = 10, n_draws: int = 1, overwrite: bool = False,
           **_) -> pd.DataFrame:
    """5 environment geometries x 5 policy algebras x n_draws x n_seeds.

    This experiment determines the scope of the result.  Every other Task-1
    result plants an SO(32) rotation and finds that an so(32) policy wins, which
    leaves the obvious question open: is this about compactness, or about
    matching whatever structure the environment happens to have?

    Changing M_env changes both the geometry-reward target and the
    auxiliary-loss target, because main.py's auxiliary loss regresses on
    env.M_env.  That is the correct coupling here: the environment's structure
    changes and the supervised target follows it.

    `id` is a positive control, not a test cell.  With M_env = I the baseline
    acts with exactly the target transformation and should top that row; a Lie
    arm winning there would mean the geometry reward is measuring something
    other than agreement with M_env.

    On n_draws.  At n_draws=1 every column reflects one particular draw of
    M_env as much as it reflects that draw's family, so the scope claim rests
    on five matrices.  With n_draws>1 the draw becomes the replication unit --
    the same design X-3 uses -- and the analysis averages seeds within a draw
    before testing across draws.  The identity geometry is constant by
    definition, so it is drawn once however large n_draws is; drawing it
    repeatedly would spend compute reproducing the same matrix.

    Cost is 5 * 5 * n_draws * n_seeds runs, minus the (n_draws-1) * 5 * n_seeds
    skipped for the identity.  n_draws=3, n_seeds=5 is the smallest design that
    lets a sign be called stable across draws.
    """
    csv = "r7_cells.csv"
    guard_outputs([csv], overwrite)
    env, sd, na, k, w = _task1_env()
    cols = (["geom", "draw", "menv_seed", "menv_orth", "menv_sym", "menv_cond"]
            + BASE_COLS)
    orig = env.M_env.clone()

    # The identity is a single matrix, not a family: drawing it n_draws times
    # would burn compute reproducing I.
    draws_for = lambda g: 1 if g == "id" else n_draws

    sink, rows = Sink(csv, cols), []
    write_meta(csv, dict(geoms=list(GEOMS), arms=list(ALL_ARMS), w=w,
                         horizon=HORIZON_DEFAULT, n_seeds=n_seeds,
                         n_draws=n_draws, menv_seed0=9000))
    total = sum(draws_for(g) for g in GEOMS) * len(ALL_ARMS) * n_seeds
    done = 0
    try:
        for gi, geom in enumerate(GEOMS):
            for draw in range(draws_for(geom)):
                ms = menv_seed(gi, draw)
                M, diag = draw_geometry(geom, k, ms)
                env.M_env = M.to(device=orig.device, dtype=orig.dtype)
                print(f"\n--- geometry {geom} draw {draw} (seed {ms}): "
                      f"|MtM-I|={diag['menv_orth']:.2e}  "
                      f"|M-Mt|={diag['menv_sym']:.2e}  "
                      f"cond={diag['menv_cond']:.3e}", flush=True)
                for seed in range(n_seeds):
                    for arm in ALL_ARMS:
                        rec, _, _ = run_one(
                            env, arm, seed, sd, na, k, w, HORIZON_DEFAULT,
                            extra=dict(geom=geom, draw=draw, menv_seed=ms,
                                       **diag))
                        sink.add([{c: rec.get(c, np.nan) for c in cols}])
                        rows.append(rec)
                        done += 1
                        _progress(done, total,
                                  f"{geom:5s} d{draw} s{seed} {arm}", rec)
    finally:
        env.M_env = orig
    print(f"\nWrote {csv} ({sink.n} rows)")
    return pd.DataFrame(rows)


# =========================================================================== #
# B-1 / B-1-LR -- the exponential map vs a Cayley retraction
# =========================================================================== #
A_EXPM, A_CAYLEY = "so_expm", "so_cayley"


def cayley(A):
    """(I - A/2)^-1 (I + A/2).  For skew A this lands exactly in SO(n).

    Differentiable (torch.linalg.solve has a backward), so the auxiliary loss
    trains through it.  Cannot reach rotations with an eigenvalue of -1, a
    measure-zero set, so nothing reachable is lost in practice.

    Relation to expm, which is what B-1-LR tests:
        exp(A)    = I + A + A^2/2 + A^3/6 + ...
        cayley(A) = I + A + A^2/2 + A^3/4 + ...
    They agree to second order and diverge after, so the same step in the
    algebra produces a different displacement in the group.  A learning rate
    tuned for one is not neutral between them.
    """
    import torch

    n = A.shape[-1]
    I = torch.eye(n, device=A.device, dtype=A.dtype)
    return torch.linalg.solve(I - 0.5 * A, I + 0.5 * A)


@contextlib.contextmanager
def matrix_map(fn):
    """Temporarily replace LieAlgebraOps.matrix_exp, and check that it is restored.

    main.py forms M in three places, all via this class attribute: the
    per-iteration M, the spectral-radius diagnostic, and the auxiliary loss.
    Patching it is global state, so the finally block restores it and the
    assert makes a leak impossible to miss -- if it ever leaked, every later arm
    would be measured with the wrong map.
    """
    from main import LieAlgebraOps

    original = LieAlgebraOps.__dict__["matrix_exp"]
    try:
        LieAlgebraOps.matrix_exp = staticmethod(fn)
        yield
    finally:
        LieAlgebraOps.matrix_exp = original
        assert LieAlgebraOps.__dict__["matrix_exp"] is original, \
            "FATAL: matrix_exp was not restored; later arms would be wrong"


def orthogonality_of_M(policy) -> float:
    """max |M^T M - I| for the arm's final M, whichever map produced it.

    Must be called inside the patch context, or it measures the Cayley theta
    under expm.  This is the validity gate: a row whose M is not a rotation is
    not interpretable, whatever its AUC says.
    """
    import torch
    from main import LieAlgebraOps

    if not hasattr(policy, "theta"):
        return float("nan")
    with torch.no_grad():
        M = LieAlgebraOps.matrix_exp(policy._proj(policy.theta.data)).double()
        I = torch.eye(M.shape[-1], dtype=M.dtype, device=M.device)
        return float((M.T @ M - I).abs().max().item())


def _run_map_sweep(csv: str, plan, n_seeds: int, overwrite: bool, meta: dict):
    """Shared driver for B-1 and B-1-LR.  `plan` is a list of (arm, theta_lr)."""
    guard_outputs([csv], overwrite)
    env, sd, na, k, w = _task1_env()
    cols = ["arm", "theta_lr"] + BASE_COLS[1:] + ["orth_err"]

    sink, rows = Sink(csv, cols), []
    write_meta(csv, meta)
    total, done = len(plan) * n_seeds, 0
    for seed in range(n_seeds):
        for arm, lr in plan:
            ctx = matrix_map(cayley) if arm == A_CAYLEY else contextlib.nullcontext()
            with ctx:
                rec, _, policy = run_one(
                    env, BASELINE if arm == BASELINE else "so", seed, sd, na, k,
                    w, HORIZON_DEFAULT,
                    theta_lr=None if arm == BASELINE else lr,
                    extra={"theta_lr": lr})
                rec["orth_err"] = (orthogonality_of_M(policy)
                                   if policy is not None else np.nan)
            rec["arm"] = arm                       # record the map, not "so"
            sink.add([{c: rec.get(c, np.nan) for c in cols}])
            rows.append(rec)
            done += 1
            _progress(done, total, f"{arm} lr={lr} s{seed}", rec)
    print(f"\nWrote {csv} ({sink.n} rows)")
    return pd.DataFrame(rows)


def run_b1(n_seeds: int = 20, overwrite: bool = False, **_) -> pd.DataFrame:
    """expm vs Cayley at the paper's shared theta_lr = 0.03.

    The surviving Task-1 claim is "constraining to the compact subalgebra
    optimises better".  Every comparator so far has been another of the paper's
    own parameterisations, which invites the question: better than what?  The standard way
    to optimise on SO(n) is a Cayley retraction, and the paper already cites the
    literature that uses it.
    """
    plan = [(BASELINE, np.nan), (A_EXPM, PAPER_LR), (A_CAYLEY, PAPER_LR)]
    return _run_map_sweep("b1_cells.csv", plan, n_seeds, overwrite,
                          dict(arms=[BASELINE, A_EXPM, A_CAYLEY], w=W_DEFAULT,
                               horizon=HORIZON_DEFAULT, n_seeds=n_seeds,
                               theta_lr=PAPER_LR))


def run_b1lr(n_seeds: int = 10, overwrite: bool = False, **_) -> pd.DataFrame:
    """B-1 again, with the full learning-rate grid for each map.

    B-1 ran both maps at 0.03, and that rate was chosen for the exponential
    map.  This is exactly the confound R-5 removed for the algebra comparison,
    and it is still present in the map comparison.  The analysis selects each
    map's rate leave-one-seed-out, so selection and evaluation are disjoint.
    """
    plan = [(BASELINE, np.nan)] + [(m, lr) for m in (A_EXPM, A_CAYLEY)
                                   for lr in LR_GRID]
    return _run_map_sweep("b1lr_cells.csv", plan, n_seeds, overwrite,
                          dict(maps=[A_EXPM, A_CAYLEY], lr_grid=list(LR_GRID),
                               w=W_DEFAULT, horizon=HORIZON_DEFAULT,
                               n_seeds=n_seeds))


# =========================================================================== #
# S-7 -- is the advantage an inductive bias, or supervision?
# =========================================================================== #
class _AuxTargetProxy:
    """Forwards every attribute to the real env except M_env.

    LieStructuredPPO reads only env.reset, env.step, env.k_transform and
    env.M_env.  The geometry reward is computed inside the env's own step() via
    `self.M_env`, so it is untouched by this proxy: only the auxiliary loss,
    which reads `self.env.M_env` from the trainer, sees the substitute.

    That asymmetry is the entire mechanism of this experiment.  Writing M_env
    through the proxy raises, because a write would also change the reward and
    invalidate the comparison.
    """

    __slots__ = ("_env", "_M_aux")

    def __init__(self, env, M_aux):
        object.__setattr__(self, "_env", env)
        object.__setattr__(self, "_M_aux", M_aux)

    @property
    def M_env(self):
        return object.__getattribute__(self, "_M_aux")

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_env"), name)

    def __setattr__(self, name, value):
        if name == "M_env":
            raise AttributeError("cannot write M_env through the proxy")
        setattr(object.__getattribute__(self, "_env"), name, value)


def haar_rotation(k: int, seed: int, device=None, dtype=None):
    """A uniformly random element of SO(k) (Haar, via QR with sign fixes)."""
    import torch

    g = torch.Generator(device="cpu").manual_seed(int(seed))
    A = torch.randn(k, k, generator=g, dtype=torch.float64)
    Q, R = torch.linalg.qr(A)
    Q = Q * torch.sign(torch.diagonal(R)).unsqueeze(0)    # Haar correction
    if torch.det(Q) < 0:                                  # force det = +1
        Q[:, 0] = -Q[:, 0]
    err = float((Q.T @ Q - torch.eye(k, dtype=torch.float64)).abs().max())
    if not (err < 1e-9 and abs(float(torch.det(Q)) - 1.0) < 1e-9):
        raise RuntimeError(f"G_ref not in SO({k}): max|QtQ-I|={err:.2e}")
    return Q.to(device=device, dtype=dtype) if device is not None else Q


def _alignment(M_pol, M_target, probes):
    """(mean cosine over data-manifold probes, Frobenius cosine).

    The probe version mirrors the reward's own form on the states the reward
    actually sees; the first k dims of GPT-2 embeddings are strongly
    anisotropic, so a policy can match M_env well where the reward looks and
    poorly elsewhere.  An isotropic probe set would just re-measure the
    Frobenius cosine with added noise.
    """
    import torch
    import torch.nn.functional as F

    if M_pol is None or M_target is None:
        return float("nan"), float("nan")
    A, B = M_pol.double(), M_target.double()
    fro = float((torch.sum(A * B) / (A.norm() * B.norm() + 1e-12)).item())
    if probes is None:
        return float("nan"), fro
    V = probes.to(A.device)
    cos = float(F.cosine_similarity(V @ A.T, V @ B.T, dim=-1).mean().item())
    return cos, fro


def run_s7(n_seeds: int = 20, overwrite: bool = False, **_) -> pd.DataFrame:
    """Swap the auxiliary target and see what the advantage was made of.

    main.py's auxiliary loss is
        L_geo = -c_geo * mean_v cos( exp(Proj_so(theta)) v ,  M_env v )
    i.e. it regresses the policy's transformation onto the environment's own
    latent rotation.  Section 8.1 of the manuscript states the opposite.

    Arms (the geometry reward uses M_env in all of them; only the target moves):
        baseline_ppo    no transformation at all
        so+M_env        target = M_env       <- as released
        so+G_ref        target = one fixed independent rotation
        so+G_rand       target redrawn per seed  (kills the one-draw objection)
        so+no_aux       c_geo = 0            <- what Sec. 8.1 describes

    Which arm is the paper's described configuration.
    Sec. 8.1 describes a target-free rotational inductive bias.  That bias is
    already supplied unconditionally in every non-baseline arm by the
    projection itself, which keeps theta in so(32) whatever c_geo is.  So
    `so+no_aux` is the described configuration.

    Confound: so+G_ref supplies a specific wrong target that the
    geometry reward actively penalises, at c_geo = 1.0.  If it loses, that is
    equally well explained by "we handed it a target that fights the reward" as
    by "a rotational prior does not help".  This design cannot separate those
    two.  The alignment columns are what distinguish them in practice: an arm
    that ends aligned with its own target and at zero with M_env followed the
    auxiliary loss and learned nothing from reward.
    """
    csv = "s7_cells.csv"
    guard_outputs([csv], overwrite)
    import torch
    import torch.nn.functional as F
    from main import LieAlgebraOps

    env, sd, na, k, w = _task1_env()
    M_env = env.M_env

    # Probes drawn from the states the reward is actually evaluated on.
    # Not `A or B`: bool() on a multi-element tensor raises
    # "Boolean value of Tensor with more than one value is ambiguous", and
    # MultiStepTextAlignmentEnv always has state_embs, so that spelling made
    # S-7 unrunnable -- after GPT-2 had already loaded.
    S = getattr(env, "state_embs", None)
    if S is None:
        S = getattr(env, "prompt_embs", None)
    probes = None
    if S is not None:
        V = S.reshape(-1, S.shape[-1])[:, :k]
        if V.shape[0] > 4096:
            g = torch.Generator(device="cpu").manual_seed(4242)
            V = V[torch.randperm(V.shape[0], generator=g)[:4096].to(V.device)]
        probes = F.normalize(V.double(), dim=-1)

    G_ref = haar_rotation(k, 777_000, M_env.device, M_env.dtype)
    a_cos, a_fro = _alignment(G_ref, M_env, probes)
    print(f"G_ref vs M_env: manifold cosine={a_cos:+.4f}  Frobenius={a_fro:+.4f}")
    print("  (near zero on both is what makes G_ref a fair non-environment "
          "target.)")
    if abs(a_fro) > 0.2 or (not math.isnan(a_cos) and abs(a_cos) > 0.3):
        sys.exit("G_ref is too close to M_env for the test to be clean.")

    arms = [BASELINE, "so+M_env", "so+G_ref", "so+G_rand", "so+no_aux"]
    cols = BASE_COLS + ["align_env", "align_ref"]
    sink, rows = Sink(csv, cols), []
    write_meta(csv, dict(arms=arms, w=w, horizon=HORIZON_DEFAULT,
                         n_seeds=n_seeds, gref_seed=777_000))
    total, done = len(arms) * n_seeds, 0
    for seed in range(n_seeds):
        G_rand = haar_rotation(k, 777_001 + seed, M_env.device, M_env.dtype)
        for arm in arms:
            target = {"so+G_ref": G_ref, "so+G_rand": G_rand}.get(arm)
            c_geo = 0.0 if arm == "so+no_aux" else None
            inner = BASELINE if arm == BASELINE else "so"

            from sppg_core import build
            try:
                policy, trainer, _ = build(env, inner, seed, sd, na, k,
                                           geo_aux_coef=c_geo)
                if target is not None:
                    # Only the auxiliary target changes; the reward does not.
                    trainer.env = _AuxTargetProxy(env, target)
                instrument_trainer(trainer, env, None, HORIZON_DEFAULT,
                                   strict=True)
                res = trainer.train(verbose=False)
                status = "ok"
            except KeyboardInterrupt:
                raise
            except Exception as exc:                         # noqa: BLE001
                rec = {c: np.nan for c in cols}
                rec.update(arm=arm, seed=seed,
                           status=f"{type(exc).__name__}: {exc}"[:120])
                sink.add([rec])
                rows.append(rec)
                done += 1
                continue

            pairs = channels(res, trainer)
            a_task, a_geo, a_tot = channel_aucs(pairs, w, HORIZON_DEFAULT)
            M_pol = None
            if hasattr(policy, "theta"):
                with torch.no_grad():
                    M_pol = LieAlgebraOps.matrix_exp(
                        policy._proj(policy.theta.data))
            al_env, _ = _alignment(M_pol, M_env, probes)
            # The arm's own auxiliary target: M_env for the reference arm (its
            # target is the environment's rotation), the substitute for the
            # swapped arms, and undefined for arms with no auxiliary loss.
            own = target if target is not None else (
                M_env if arm == "so+M_env" else None)
            al_ref, _ = _alignment(M_pol, own, probes)
            spec = [s for s in res.get("spectral_radii", [])
                    if s is not None and math.isfinite(s)]
            rec = dict(arm=arm, seed=seed, auc=float(res["auc"]),
                       auc_task=a_task, auc_geo=a_geo,
                       r_task=float(np.mean([p[0] for p in pairs])),
                       r_geo=float(np.mean([p[1] for p in pairs])),
                       rho=float(np.median(spec)) if spec else np.nan,
                       recon_err=abs(a_tot - float(res["auc"])),
                       align_env=al_env, align_ref=al_ref, status=status)
            sink.add([{c: rec.get(c, np.nan) for c in cols}])
            rows.append(rec)
            done += 1
            print(f"  [{done:3d}/{total}] seed {seed:2d} {arm:12s} "
                  f"AUC={rec['auc']:7.3f}  align(M_env)={al_env:+.3f}  "
                  f"align(target)={al_ref:+.3f}", flush=True)
    print(f"\nWrote {csv} ({sink.n} rows)")
    return pd.DataFrame(rows)


# =========================================================================== #
# R-11 -- Task 2 on held-out prompts, and a corrected V2
# =========================================================================== #
PUBLISHED_PROMPTS = [
    "Write about geometric structure in reinforcement learning.",
    "The story is about a robot that learns to paint.",
    "In the future, agents will learn from symmetry because",
    "Explain why group actions matter in optimization.",
    "A short story about a mathematician who loves groups.",
    "Summarize the role of invariances in optimization.",
]

ADDITIONAL_PROMPTS = [
    "Describe how rotations act on a vector space.",
    "The robot discovered that turning the canvas changed nothing important.",
    "Explain conservation laws to someone who has never studied physics.",
    "Write about a cartographer who could not decide on an orientation.",
    "Why does the order of two operations sometimes not matter?",
    "A brief note on why some transformations preserve distance.",
    "Tell a story about a dancer who thought in terms of symmetry.",
    "Explain what stays the same when everything else changes.",
    "Write about an engineer who trusted invariants over measurements.",
    "Describe the difference between a rotation and a stretch.",
    "In a world without preferred directions, navigation would",
    "Summarize why structure can make a search problem easier.",
    "A short story about a translator who preserved meaning exactly.",
    "Explain why physicists care about groups of transformations.",
    "Write about a sculptor who worked only with rigid motions.",
    "What does it mean for a system to have no preferred frame?",
    "Describe a machine that learns the shape of its own errors.",
    "Explain why averaging rotations is harder than it sounds.",
]


def build_prompt_splits(split_seed: int = 4242):
    """Disjoint (discovery+train, test) prompt lists.

    Section 9 as published uses the same six prompts for structure discovery,
    for REINFORCE training and for evaluation, so every number in it is
    in-sample.  Six prompts cannot be split, so we extend to twenty-four in the
    same register and split 12/12.

    The published six are forced into the train half, so the training
    distribution is a superset of the original and the test half is entirely
    unseen.  That makes the comparison strictly harder than the published one,
    never easier.
    """
    rng = random.Random(int(split_seed))
    extra = list(ADDITIONAL_PROMPTS)
    rng.shuffle(extra)
    train = list(PUBLISHED_PROMPTS) + extra[:len(PUBLISHED_PROMPTS)]
    test = extra[len(PUBLISHED_PROMPTS):]
    assert not (set(train) & set(test)), "train and test overlap"
    return train, test


def recompute_v2(trials: int = 200, n: int = 32, seed: int = 0) -> dict:
    """Independent recomputation of the V2 "projection == natural gradient" check.

    The released check (`validate_C2_proj_natgrad_equiv` in 04_pap_CG.py) builds
    both sides of the comparison from the same projection, so it compares an
    expression with itself and cannot fail; its 0.99993 carries no information.

    Here the "natural gradient" side is obtained without reference to the
    projection: a Euclidean gradient preconditioned by an explicitly
    constructed random SPD metric on the ambient n^2 coordinates.  If 0.99993
    is real this route reproduces it; if it was an artefact of self-comparison,
    it will not.
    """
    import torch

    g = torch.Generator().manual_seed(int(seed))
    cos_proj, cos_ambient = [], []
    for _ in range(trials):
        G = torch.randn(n, n, generator=g, dtype=torch.float64)
        G_proj = 0.5 * (G - G.T)                      # the SP-PG direction
        B = torch.randn(n * n, n * n, generator=g,
                        dtype=torch.float64) / math.sqrt(n * n)
        Fm = B @ B.T + 1e-2 * torch.eye(n * n, dtype=torch.float64)
        g_nat = torch.linalg.solve(Fm, G.reshape(-1)).reshape(n, n)
        g_nat_proj = 0.5 * (g_nat - g_nat.T)

        def cos(x, y):
            return float(torch.nn.functional.cosine_similarity(
                x.reshape(1, -1), y.reshape(1, -1)).item())

        cos_proj.append(cos(G_proj, g_nat_proj))
        cos_ambient.append(cos(G, g_nat))
    return {
        "n_trials": trials, "n": n,
        "cos_projected_mean": float(np.mean(cos_proj)),
        "cos_projected_std": float(np.std(cos_proj, ddof=1)),
        "cos_ambient_mean": float(np.mean(cos_ambient)),
        "cos_ambient_std": float(np.std(cos_ambient, ddof=1)),
        "published_V2": 0.99993,
    }


def run_r11(n_seeds: int = 10, overwrite: bool = False, **_) -> pd.DataFrame:
    """Task 2 with disjoint discovery+train and test prompt sets.

    Reuses 04_pap_CG.py's already-repaired control arm rather than
    reimplementing it: the released baseline was built with lmbda = 0, which
    severs the gradient path entirely (theta_raw and the feature heads receive
    exactly zero gradient), so the published comparison was
    structured-vs-untrained, not structured-vs-unstructured.  The correct
    control keeps the same capacity, the same lmbda, the same budget and the
    same seed, and removes only the compactness constraint.

    The replication unit is the training seed.  Prompt-level rewards are
    averaged within a seed before any test; pooling 12 prompts as independent
    replicates would inflate n by 12x and is the easiest way to manufacture
    significance here.
    """
    csv = "r11_cells.csv"
    guard_outputs([csv], overwrite)
    import torch
    from importlib.machinery import SourceFileLoader

    M = SourceFileLoader("pap_cg", "04_pap_CG.py").load_module()
    cfg = M.Cfg()
    M.ensure_dir(cfg.results_dir)
    base_model, tok = M.load_model(cfg)
    n = base_model.config.n_embd
    alg = M.LieAlgebra(cfg.algebra, n)

    train_prompts, test_prompts = build_prompt_splits()
    print(f"\nprompt split: {len(train_prompts)} discovery+train, "
          f"{len(test_prompts)} held-out (disjoint); the {len(PUBLISHED_PROMPTS)}"
          f" published prompts are all in train")

    # Structure discovery on the train half only.
    M.set_seed(0)
    V = torch.stack([M.repr_vec(base_model, tok, p, cfg.device)
                     for p in train_prompts], dim=0)
    k, eps = 4, 0.02
    R = torch.randn(k, n, n, device=cfg.device) * eps
    T = torch.linalg.matrix_exp(alg.project(R))
    disc = M.Discovery(alg, cfg.device, lr=cfg.discovery_lr,
                       steps=cfg.discovery_steps)
    X_star = disc.fit([T[i] for i in range(k)], V, steps=cfg.discovery_steps)

    cols = ["seed", "split", "arm", "prompt_idx", "eval_seed", "reward", "status"]
    sink, rows = Sink(csv, cols), []
    write_meta(csv, dict(n_seeds=n_seeds, split_seed=4242,
                         n_train=len(train_prompts), n_test=len(test_prompts)))

    for seed in range(n_seeds):
        print(f"\n[seed {seed}]", flush=True)
        M.set_seed(seed)
        pol = M.SIPolicy(base_model, tok, alg, X_star, cfg).to(cfg.device)
        tr = M.REINFORCE(pol, tok, cfg)
        for _ in range(cfg.rl_iters):
            tr.step([random.choice(train_prompts)
                     for _ in range(cfg.batch_prompts)])

        M.set_seed(seed)
        ctrl = M.SIPolicy(base_model, tok, M.LieAlgebra("gl", n), X_star,
                          cfg).to(cfg.device)
        # Same bias weight, not zero.  Copied by value: if SIPolicy.lmbda were
        # ever an nn.Parameter, a plain rebind would register the structured
        # policy's own parameter on the control, and training the control
        # (below) would then mutate the already-trained structured arm before
        # either is evaluated -- invisibly, since the movement check still
        # passes.
        import torch as _t
        ctrl.lmbda = (pol.lmbda.detach().clone()
                      if _t.is_tensor(pol.lmbda) else float(pol.lmbda))
        before = torch.cat([p.detach().reshape(-1).clone()
                            for p in ctrl.parameters() if p.requires_grad])
        ctr = M.REINFORCE(ctrl, tok, cfg)
        for _ in range(cfg.rl_iters):
            ctr.step([random.choice(train_prompts)
                      for _ in range(cfg.batch_prompts)])
        after = torch.cat([p.detach().reshape(-1).clone()
                           for p in ctrl.parameters() if p.requires_grad])
        if float((after - before).abs().max()) == 0.0:
            raise RuntimeError("control parameters did not move; the comparison "
                               "would again be trained-vs-untrained. Aborting.")

        batch = []
        with torch.no_grad():
            for split, plist in (("test", test_prompts), ("train", train_prompts)):
                for eval_seed in (0, 1, 2):
                    torch.manual_seed(eval_seed)
                    np.random.seed(eval_seed)
                    random.seed(eval_seed)
                    for pi, p in enumerate(plist):
                        for arm, model in (("structured", pol), ("control", ctrl)):
                            ids = tok(p, return_tensors="pt"
                                      ).to(cfg.device)["input_ids"]
                            out = model.generate(ids, cfg.max_new_tokens)
                            r = M.task_reward_from_ids(base_model, tok, ids,
                                                       out[:, ids.shape[1]:])
                            batch.append(dict(seed=seed, split=split, arm=arm,
                                              prompt_idx=pi, eval_seed=eval_seed,
                                              reward=float(r), status="ok"))
        sink.add(batch)
        rows.extend(batch)
        del pol, ctrl, tr, ctr
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\nWrote {csv} ({sink.n} rows)")
    return pd.DataFrame(rows)


# =========================================================================== #
# R-15 / R-15b -- Task 3 positive control
# =========================================================================== #
def _inject_geometry(base_env, M_env, w: float, k: int):
    """Wrap a SentimentSteeringEnv so its reward carries a geometry channel:

        r = (1 - w) * r_sentiment + w * r_geo,
        r_geo = (cos(M_policy v, M_env v) + 1) / 2,   v = state[:k]

    exactly as Task 1 does.  w = 0 reproduces the published environment bit
    for bit: the geometry term is not even computed.

    The planted rotation is stored as `M_env_r15`, not `M_env`.  main.py's
    trainer enables its auxiliary loss on `hasattr(env, "M_env")`; Task 3's
    trainer has no such path, but code ported from main.py would switch on a
    supervised regression and change what is being measured.  The distinct
    name prevents that.

    One asymmetry: Task 3 applies its sparsity mask (sparse_prob = 0.5)
    to the sentiment reward inside the original step(), before this wrapper sees
    it, whereas the injected geometry term is dense.  That is conservative for
    the conclusion we care about: if a dense geometric signal at w = 0.2 still
    produces no separation, a sparse one certainly would not.
    """
    import torch

    if not (0.0 <= w < 1.0):
        raise ValueError(f"w={w}: the sentiment channel is not identifiable "
                         f"at w >= 1.")
    env = base_env
    orig_step = env.step
    env.M_env_r15 = M_env
    env._r15_w, env._r15_k = float(w), int(k)
    env._r15_sent, env._r15_geo = [], []

    def step_wrapper(action, M=None):
        nxt, r_sent, done = orig_step(action, M)
        w_ = float(env._r15_w)
        if w_ == 0.0:
            env._r15_sent.append(float(r_sent))
            # NaN, not 0.0.  At w=0 the geometry term is never computed, and a
            # literal 0.0 in the manipulation-check column reads as "the
            # geometry channel collapsed" when the correct statement is "not
            # measured".  (An unaligned policy scores ~0.5, so 0.0 is not even
            # a plausible value.)
            env._r15_geo.append(float("nan"))
            return nxt, float(r_sent), done
        v = nxt[: env._r15_k].to(torch.float32)
        Mv = env.M_env_r15.to(v.dtype) @ v
        Pv = (M[: env._r15_k, : env._r15_k].to(v.dtype) @ v) if M is not None else v
        cos = torch.nn.functional.cosine_similarity(
            Pv.unsqueeze(0), Mv.unsqueeze(0)).item()
        r_geo = (cos + 1.0) / 2.0
        env._r15_sent.append(float(r_sent))
        env._r15_geo.append(float(r_geo))
        return nxt, (1.0 - w_) * float(r_sent) + w_ * r_geo, done

    env.step = step_wrapper
    return env


def _supervised_trainer(T3, aux_coef: float, k: int):
    """A LieStructuredPPO subclass carrying main.py's auxiliary loss.

    run_sentiment_task3.py has no auxiliary-loss code path, so R-15 can only
    test a reward-only injection.  Copying its training loop to insert one term
    would risk the copy drifting from the original, which is exactly what makes
    the comparison valid.  Instead we exploit the call order the original
    already has:

        self.pi_optim.zero_grad()
        dist_mb = Categorical(self.policy(obs[mb]))    <-- forward pre-hook fires
        ...
        pi_loss.backward()
        mb_gnorms.append(self._total_grad_norm(...))
        self._proj_grads()                            <-- we override THIS
        self.pi_optim.step()

    A forward pre-hook records the minibatch; the overridden _proj_grads adds
    the auxiliary gradient for exactly that minibatch and then defers to the
    parent.  The gradient lands at the same point in the update as main.py's
    L_geo.backward(), on the same states, and not one line of the original
    trainer is modified.

    Two diagnostics shift as a result, neither affecting training: grad_norms
    are recorded by the parent before this runs (so they exclude the auxiliary
    term), and projection_magnitudes now measure the combined gradient.  The
    optimiser sees exactly what main.py's optimiser sees.
    """
    import torch
    import torch.nn.functional as F

    class SupervisedPPO(T3.LieStructuredPPO):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self._aux_coef, self._k = float(aux_coef), int(k)
            self._last_obs, self._aux_fired = None, 0
            self._hook = self.policy.register_forward_pre_hook(
                lambda _m, inp: self._record(inp))

        def _record(self, inp):
            x = inp[0]
            self._last_obs = x if x.dim() == 2 else x.unsqueeze(0)

        def _proj_grads(self):
            M_env = getattr(self.env, "M_env_r15", None)
            if (self.use_lie and self._aux_coef > 0.0
                    and hasattr(self.policy, "theta")
                    and self._last_obs is not None and M_env is not None):
                v = self._last_obs[:, : self._k]
                M_pol = T3.LieAlgebraOps.matrix_exp(
                    self._apply_proj(self.policy.theta))
                actual = (M_pol @ v.unsqueeze(-1)).squeeze(-1)
                target = (M_env.to(v.dtype) @ v.unsqueeze(-1)).squeeze(-1)
                (-self._aux_coef *
                 F.cosine_similarity(actual, target, dim=-1).mean()).backward()
                self._aux_fired += 1
            super()._proj_grads()

        def alignment(self) -> float:
            """cos(M_policy v, M_env v) on the states actually visited."""
            M_env = getattr(self.env, "M_env_r15", None)
            if (M_env is None or not hasattr(self.policy, "theta")
                    or self._last_obs is None):
                return float("nan")
            with torch.no_grad():
                v = self._last_obs[:, : self._k]
                M_pol = T3.LieAlgebraOps.matrix_exp(
                    self._apply_proj(self.policy.theta.data))
                a = (M_pol @ v.unsqueeze(-1)).squeeze(-1)
                t = (M_env.to(v.dtype) @ v.unsqueeze(-1)).squeeze(-1)
                return float(F.cosine_similarity(a, t, dim=-1).mean().item())

        def close(self):
            if getattr(self, "_hook", None) is not None:
                self._hook.remove()
                self._hook = None

    return SupervisedPPO


def _run_task3(csv: str, w_grid, supervised: bool, n_seeds: int,
               overwrite: bool) -> pd.DataFrame:
    """Shared driver for R-15 (reward-only) and R-15b (with supervision)."""
    guard_outputs([csv], overwrite)
    import torch
    import run_sentiment_task3 as T3

    K = 32
    T3._seed_all(42)
    embedder = T3.GPT2EmbeddingProvider("gpt2-medium")
    sent_dir = T3.compute_sentiment_direction(embedder)
    Sup = _supervised_trainer(T3, 1.0, K) if supervised else None

    cols = ["w", "arm", "seed", "auc", "r_sent", "r_geo", "align", "rho", "status"]
    sink, rows = Sink(csv, cols), []
    write_meta(csv, dict(w_grid=list(w_grid), supervised=supervised,
                         n_seeds=n_seeds, menv_seed=7000, k=K))
    total = len(w_grid) * 2 * n_seeds
    done = 0
    for w in w_grid:
        for seed in range(n_seeds):
            for arm in ("baseline_ppo", "so"):
                # A fresh env per cell: the wrapper closes over w, and reusing
                # one env across w values would stack wrappers.
                T3._seed_all(seed)
                env = T3.SentimentSteeringEnv(
                    embedder, sentiment_direction=sent_dir, n_prompts=20,
                    n_actions=8, horizon=10, reward_noise=0.1, sparse_prob=0.5)
                M = draw_rotation(K, 7000, env.prompt_embs.device, torch.float32)
                env = _inject_geometry(env, M, w, K)

                cfg = T3.PPOConfig()
                sd = embedder.hidden_dim
                T3._seed_all(seed)
                Cls = Sup if supervised else T3.LieStructuredPPO
                if arm == "baseline_ppo":
                    # No theta, so the auxiliary term cannot fire: this arm is
                    # identical to the reward-only baseline.
                    tr = Cls(env, T3.BaselinePolicy(sd, env.n_actions),
                             T3.ValueNet(sd), cfg, use_lie_projection=False)
                else:
                    tr = Cls(env, T3.LiePolicy(sd, env.n_actions, k=K,
                                               algebra="so"),
                             T3.ValueNet(sd), cfg, use_lie_projection=True,
                             algebra="so")
                status, auc, rho, align = "ok", np.nan, np.nan, np.nan
                try:
                    res = tr.train(verbose=False)
                    auc = float(res["auc"])
                    # Guarded, and the median -- matching run_one() in
                    # sppg_core.  An unguarded res["spectral_radii"][-1] threw
                    # on an arm with no theta after auc was assigned, producing
                    # a row with a valid AUC but a non-ok status, which the
                    # analysis then drops, unbalancing the paired design.
                    sr = [x for x in (res.get("spectral_radii") or [])
                          if x is not None and math.isfinite(x)]
                    rho = float(np.median(sr)) if sr else np.nan
                    if supervised:
                        align = tr.alignment()
                        if arm == "so" and tr._aux_fired == 0:
                            status = "aux-never-fired"
                except KeyboardInterrupt:
                    raise
                except Exception as exc:                     # noqa: BLE001
                    status = f"{type(exc).__name__}: {exc}"[:110]
                finally:
                    if supervised:
                        tr.close()
                rec = dict(w=w, arm=arm, seed=seed, auc=auc,
                           r_sent=float(np.mean(env._r15_sent)) if env._r15_sent else np.nan,
                           # nanmean: the w=0 block records NaN by design
                           r_geo=(float(np.nanmean(env._r15_geo))
                                  if env._r15_geo and
                                  not np.all(np.isnan(env._r15_geo)) else np.nan),
                           align=align, rho=rho,
                           status=status if math.isfinite(auc) or status != "ok"
                           else "non-finite")
                sink.add([rec])
                rows.append(rec)
                done += 1
                print(f"  [{done:3d}/{total}] w={w:<5g} s{seed:2d} {arm:12s} "
                      f"AUC={auc:7.3f} r_geo={rec['r_geo']:.4f} "
                      f"align={align:+.4f}", flush=True)
    print(f"\nWrote {csv} ({sink.n} rows)")
    return pd.DataFrame(rows)


def run_r15(n_seeds: int = 10, overwrite: bool = False, **_):
    """Task-3 positive control, reward-only.

    Task 3 reports a null and the paper reads it as a falsification test
    passing.  A null at n = 10 says little about absence unless the design
    could have detected a present effect -- which was never measured.  This
    injects known geometry at increasing strength and finds the smallest
    injection the design detects, converting "we found no effect" into "we
    could have detected an effect of size >= X, and did not".

    w = 0 reproduces the published environment exactly and is the null cell,
    not a test; the analysis excludes it from the Holm family.
    """
    return _run_task3("r15_cells.csv", (0.0, 0.02, 0.05, 0.1, 0.2), False,
                      n_seeds, overwrite)


def run_r15b(n_seeds: int = 10, overwrite: bool = False, **_):
    """R-15 with the auxiliary target enabled.

    R-15 shows the geometry reward alone does not teach the rotation, which
    reproduces S-7's Task-1 finding in a second environment.  S-7 also showed
    the other half of the dichotomy -- that supervised regression onto M_env
    does teach it -- but Task 3 cannot show that half, because
    run_sentiment_task3.py has no auxiliary-loss path.  This adds one,
    identical in form to main.py's.  w = 0 is dropped: there is no geometry to
    supervise.
    """
    return _run_task3("r15b_cells.csv", (0.05, 0.1, 0.2), True, n_seeds,
                      overwrite)


# =========================================================================== #
# R-13b -- Mistral-7B, 20 seeds, channel-decomposed
# =========================================================================== #
def run_r13b(n_seeds: int = 20, overwrite: bool = False,
             project_dim: int = 1024, **_) -> pd.DataFrame:
    """The abstract's cross-model number, put on the same footing as Task 1.

    run_mistral_task1.py calls run_rl_ablation(..., seed=0): the algebra
    ablation is single-seed and three arms, and nothing on Mistral is
    channel-decomposed.  That leaves a headline number in the abstract whose
    meaning is known to differ from the GPT-2 one.

    run_mistral_task1.py builds the same MultiStepTextAlignmentEnv as main.py --
    only the embedding provider differs -- so the whole harness attaches
    unchanged and every statistic is computed by the code that produced R-1 and
    R-4, so the two models are directly comparable.

    Caveat: the embeddings are projected 4096 -> 1024 by a fixed random
    orthogonal map with no distortion guarantee, exactly as in the published
    run.  This changes the seed count and the analysis, not that design choice.
    """
    csv = "r13b_cells.csv"
    guard_outputs([csv], overwrite)
    from run_mistral_task1 import (MistralEmbeddingProvider,
                                   create_env_and_free_model)

    k = int(round(project_dim ** 0.5))
    if k * k != project_dim:
        raise ValueError(f"project_dim={project_dim} is not a perfect square; "
                         f"the reshape to k x k would be undefined.")
    emb = MistralEmbeddingProvider(project_dim=project_dim)
    env, sd, na, _ = create_env_and_free_model(emb, k=k)
    if int(env.k_transform) != k:
        raise RuntimeError(f"env.k_transform={env.k_transform} but k={k}")
    w = instrument(env)

    arms = [BASELINE, "so", "sl", "gl", "sym"]
    sink, rows = Sink(csv, BASE_COLS), []
    write_meta(csv, dict(arms=arms, w=w, horizon=HORIZON_DEFAULT,
                         n_seeds=n_seeds, project_dim=project_dim,
                         model="mistral-7b"))
    total, done = len(arms) * n_seeds, 0
    for seed in range(n_seeds):
        for arm in arms:
            rec, _, _ = run_one(env, arm, seed, sd, na, k, w, HORIZON_DEFAULT)
            sink.add([{c: rec.get(c, np.nan) for c in BASE_COLS}])
            rows.append(rec)
            done += 1
            _progress(done, total, f"seed {seed:2d} {arm}", rec)
    print(f"\nWrote {csv} ({sink.n} rows)")
    return pd.DataFrame(rows)


# =========================================================================== #
# Registry, self-test, CLI
# =========================================================================== #
# NOTE: there is no "r1" entry.  R-1's channel decomposition is
# produced by run_r4 (which includes the baseline arm), and an alias pointing
# at run_r4 would let `sppg_experiments.py r1 --overwrite` truncate
# an existing r4_cells.csv and its .meta.json.  Run r4; analyse r4.
EXPERIMENTS = {
    "r1b":  ("(w, c_geo) trade-off surface", run_r1b),
    "r4":   ("Five-arm algebra ablation + random(496) control", run_r4),
    "r5":   ("Per-algebra learning-rate sweep", run_r5),
    "x3":   ("20 environment draws", run_x3),
    "r7":   ("5 geometries x 5 algebras (x --draws M_env per geometry)", run_r7),
    "b1":   ("expm vs Cayley at the shared rate", run_b1),
    "b1lr": ("expm vs Cayley, each at its own tuned rate", run_b1lr),
    "s7":   ("Auxiliary target swap -- prior or supervision?", run_s7),
    "r11":  ("Task 2 on held-out prompts", run_r11),
    "r15":  ("Task-3 positive control, reward-only", run_r15),
    "r15b": ("Task-3 positive control, with supervision", run_r15b),
    "r13b": ("Mistral-7B at 20 seeds, decomposed", run_r13b),
}


def selftest() -> None:
    """Checks that need no GPU, no model and no CSV.  Run this first."""
    import torch
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  {'PASS' if cond else 'FAIL'}  {name}"
              f"{'  ' + detail if detail else ''}")

    print("(1) rotations really are rotations")
    k = 32
    M = draw_rotation(k, 7000)
    I = torch.eye(k, dtype=torch.float64)
    check("draw_rotation is orthogonal", (M.T @ M - I).abs().max() < 1e-9)
    check("draw_rotation has det = +1", abs(float(torch.det(M)) - 1) < 1e-9)
    check("draw_rotation is reproducible", torch.equal(draw_rotation(k, 7000), M))
    # Not hard-coded True: rebuild main.py's own M_env from seed 42 and match.
    from scipy.linalg import expm as _expm
    r = np.random.RandomState(42)
    S = r.randn(k, k) * 0.5
    ref = torch.tensor(_expm(0.5 * (S - S.T)), dtype=torch.float64)
    check("reproduces main.py's own M_env exactly from seed 42",
          torch.allclose(ref, draw_rotation(k, 42), atol=1e-12))
    H = haar_rotation(k, 777_000)
    check("haar_rotation is in SO(k)",
          (H.T @ H - I).abs().max() < 1e-9
          and abs(float(torch.det(H)) - 1) < 1e-9)
    check("two independent rotations are near-orthogonal",
          abs(float(torch.sum(M * H) / (M.norm() * H.norm()))) < 0.3)

    print("(2) each R-7 geometry is the family it claims")
    mats = {}
    for geom in GEOMS:
        Mg, dg = draw_geometry(geom, k, 123)
        mats[geom] = Mg.numpy()
        check(f"{geom}: well conditioned", dg["menv_cond"] < 1e8,
              f"cond={dg['menv_cond']:.2e}")
    check("so: orthogonal but not symmetric (a proper rotation)",
          np.abs(mats["so"].T @ mats["so"] - np.eye(k)).max() < 1e-9
          and np.abs(mats["so"] - mats["so"].T).max() > 1e-3)
    check("sym: symmetric, not orthogonal, positive definite",
          np.abs(mats["sym"] - mats["sym"].T).max() < 1e-9
          and np.abs(mats["sym"].T @ mats["sym"] - np.eye(k)).max() > 1e-3
          and np.linalg.eigvalsh(mats["sym"]).min() > 0)
    check("gl: neither orthogonal nor symmetric",
          np.abs(mats["gl"].T @ mats["gl"] - np.eye(k)).max() > 1e-3
          and np.abs(mats["gl"] - mats["gl"].T).max() > 1e-3)
    check("diag: diagonal with positive entries",
          np.abs(mats["diag"] - np.diag(np.diag(mats["diag"]))).max() < 1e-12
          and np.diag(mats["diag"]).min() > 0)
    check("id: identity", np.abs(mats["id"] - np.eye(k)).max() < 1e-12)
    check("the five geometries are mutually distinct",
          len({mats[g].tobytes() for g in GEOMS}) == len(GEOMS))
    try:
        draw_geometry("nope", k, 1)
        check("an unknown geometry is refused", False)
    except ValueError:
        check("an unknown geometry is refused", True)

    print("(3) the Cayley retraction")
    g = torch.Generator().manual_seed(11)
    X = torch.randn(k, k, generator=g, dtype=torch.float64)
    A = 0.5 * (X - X.T)
    C, E = cayley(A), torch.matrix_exp(A)
    check("cayley lands in SO(k)", (C.T @ C - I).abs().max() < 1e-10
          and abs(float(torch.det(C)) - 1) < 1e-9)
    check("cayley != expm at this scale (the comparison is meaningful)",
          (C - E).abs().max() > 1e-2, f"max|C-E|={(C - E).abs().max():.2e}")
    small = 1e-2 * A
    check("they agree to 2nd order for small A (docstring claim, checked)",
          (cayley(small) - torch.matrix_exp(small)).abs().max() < 1e-5)
    Z = torch.randn(k, k, generator=g, dtype=torch.float64, requires_grad=True)
    cayley(0.5 * (Z - Z.T)).pow(2).sum().backward()
    check("cayley is differentiable (the auxiliary loss trains through it)",
          Z.grad is not None and bool(torch.isfinite(Z.grad).all()))

    print("(4) the S-7 proxy swaps only the auxiliary target")

    class FakeEnv:
        def __init__(self):
            self.M_env = torch.eye(4)
            self.k_transform, self.horizon, self.geo_weight = 4, 20, 0.4
            self.calls = []

        def reset(self):
            self.calls.append("reset")
            return "S"

        def step(self, a, M=None):
            self.calls.append(("step", a))
            return "S", 1.0, True

    fe = FakeEnv()
    Gr = torch.full((4, 4), 9.0)
    px = _AuxTargetProxy(fe, Gr)
    check("proxy.M_env is the substitute", torch.equal(px.M_env, Gr))
    check("the real env keeps its own M_env (the reward is untouched)",
          torch.equal(fe.M_env, torch.eye(4)))
    check("reset/step/k_transform delegate",
          px.reset() == "S" and px.step(3)[1] == 1.0 and px.k_transform == 4)
    try:
        px.M_env = torch.zeros(4, 4)
        check("proxy blocks writes to M_env", False)
    except AttributeError:
        check("proxy blocks writes to M_env", True)

    print("(5) the R-11 prompt split")
    tr_p, te_p = build_prompt_splits()
    check("train and test are disjoint", not (set(tr_p) & set(te_p)))
    check("all six published prompts are in train",
          all(p in tr_p for p in PUBLISHED_PROMPTS))
    check("both halves are non-trivial", len(tr_p) >= 10 and len(te_p) >= 10,
          f"train={len(tr_p)}, test={len(te_p)}")
    check("the split is reproducible", build_prompt_splits() == (tr_p, te_p))
    check("a different split seed moves the test half",
          build_prompt_splits(999)[1] != te_p)
    check("...but the published six stay in train regardless",
          all(p in build_prompt_splits(999)[0] for p in PUBLISHED_PROMPTS))

    print("(6) the independent V2 recomputation")
    v2 = recompute_v2(trials=25, n=8, seed=1)
    check("returns both projected and ambient alignments",
          -1 <= v2["cos_projected_mean"] <= 1 and -1 <= v2["cos_ambient_mean"] <= 1,
          f"proj={v2['cos_projected_mean']:+.4f} amb={v2['cos_ambient_mean']:+.4f}")
    check("the projected alignment is not 1 by construction",
          abs(v2["cos_projected_mean"] - 1.0) > 1e-6,
          "if this fails the check is again comparing an expression with itself")

    print("(7) the Task-3 geometry injection")

    class StubT3Env:
        n_actions = 8

        def __init__(self):
            self.calls = 0

        def step(self, action, M=None):
            self.calls += 1
            v = torch.zeros(1024)
            v[:32] = torch.linspace(0.1, 1.0, 32)
            return v, 0.5, self.calls % 10 == 0

    Mf = draw_rotation(32, 7000).float()
    e0 = _inject_geometry(StubT3Env(), Mf, 0.0, 32)
    _, r0, _ = e0.step(0, torch.eye(1024))
    check("w=0 returns the untouched sentiment reward", abs(r0 - 0.5) < 1e-12)
    # NaN, not 0.0: at w=0 the geometry term is never computed, and recording a
    # literal zero would read as a measured collapse of the geometry channel.
    check("w=0 records geometry as not measured (NaN), not as zero",
          len(e0._r15_geo) == 1 and math.isnan(e0._r15_geo[0]))
    e1 = _inject_geometry(StubT3Env(), Mf, 0.2, 32)
    _, r1, _ = e1.step(0, torch.eye(1024))
    g1 = e1._r15_geo[-1]
    check("w>0 mixes the two channels exactly",
          abs(r1 - (0.8 * 0.5 + 0.2 * g1)) < 1e-6)
    check("r_geo lies in [0,1]", 0.0 <= g1 <= 1.0)
    big = torch.eye(1024)
    big[:32, :32] = Mf
    e2 = _inject_geometry(StubT3Env(), Mf, 0.5, 32)
    e2.step(0, big)
    check("acting with the true rotation gives near-perfect alignment",
          e2._r15_geo[-1] > 0.99, f"{e2._r15_geo[-1]:.6f}")
    check("the planted rotation is not exposed as `M_env`",
          not hasattr(e1, "M_env") and hasattr(e1, "M_env_r15"))
    try:
        _inject_geometry(StubT3Env(), Mf, 1.0, 32)
        check("w >= 1 is refused", False)
    except ValueError:
        check("w >= 1 is refused", True)

    print("\n" + ("SELFTEST PASS" if ok else "SELFTEST FAIL"))
    if not ok:
        sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="SP-PG revision experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Analysis lives in sppg_analysis.py; this file only collects.")
    ap.add_argument("experiment", nargs="?", choices=sorted(EXPERIMENTS),
                    help="which experiment to run")
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--draws", type=int, default=None,
                    help="X-3: number of environment draws (default 20).  "
                         "R-7: draws of M_env per geometry (default 1).")
    ap.add_argument("--project-dim", type=int, default=1024, help="R-13b only")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        selftest()
        return
    if a.list or not a.experiment:
        print("Experiments (run with: python sppg_experiments.py <name>)\n")
        for name, (desc, _) in EXPERIMENTS.items():
            print(f"  {name:6s}  {desc}")
        print("\nDefault seed counts match the manuscript: r4/s7/b1/r13b = 20, "
              "the rest = 10.")
        return

    desc, fn = EXPERIMENTS[a.experiment]
    print("=" * 78)
    print(f"{a.experiment.upper()}  --  {desc}")
    print("=" * 78)
    kwargs = dict(overwrite=a.overwrite, project_dim=a.project_dim)
    if a.experiment == "x3":
        kwargs["n_draws"] = a.draws if a.draws else 20
        kwargs["seeds_per_draw"] = a.seeds or 3
    elif a.experiment == "r7":
        kwargs["n_draws"] = a.draws if a.draws else 1
        if a.seeds is not None:
            kwargs["n_seeds"] = a.seeds
    elif a.seeds is not None:
        kwargs["n_seeds"] = a.seeds
    fn(**kwargs)


if __name__ == "__main__":
    main()
