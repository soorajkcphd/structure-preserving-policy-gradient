"""
S-7: is the Task-1 advantage an inductive bias, or supervision?

    python run_s7.py --selftest     # instant, no GPU, no model -- run first
    python run_s7.py --seeds 20     # ~15 min on an RTX 5060
    python run_s7.py --analyse-only

Requires _r1core.py.  Makes no edits to main.py.

The question
------------
main.py's auxiliary loss is

    L_geo = -c_geo * mean_v cos( exp(Proj_so(theta)) v ,  M_env v )

i.e. it regresses the policy's transformation onto the environment's own latent
rotation.  The manuscript states the opposite (Sec. 8.1):

    "the auxiliary loss supplies a rotational inductive bias toward
     SO-structured transformations and involves no term in M_env ... the policy
     can therefore learn the environment's rotation only through this reward
     signal, not through any direct regression on M_env."

R-1b showed that with the auxiliary loss off the method loses to baseline PPO at
every geometry weight tested.  So the whole advantage rests on this term.  The
question S-7 settles is whether the term works as a generic rotational prior
(the paper's claim) or only because it is handed the answer.

The design
----------
Five arms at the paper's setting (w = 0.4), paired by seed:

  baseline_ppo    no transformation at all
  so+M_env        auxiliary target = M_env      <- as released; reproduces the paper
  so+G_ref        auxiliary target = G_ref, one fixed independent rotation
  so+G_rand       auxiliary target redrawn per seed  <- guards against one draw
  so+no_aux       c_geo = 0, no auxiliary loss

The geometry reward uses M_env in every arm; only the auxiliary target changes.

Which arm is the paper's described configuration (read this before the result)
------------------------------------------------------------------------------
Sec. 8.1 describes a target-free rotational inductive bias.  That bias is
already supplied unconditionally, in every non-baseline arm, by the projection
itself: _apply_proj / _proj_grads / _proj_params keep theta in so(32), so
M_policy is in SO(32) whatever geo_aux_coef is.  It follows that

    so+no_aux is the configuration Sec. 8.1 describes,

and R-1b already measured it: it loses to baseline PPO at every geometry weight.

so+G_ref is therefore not a neutral prior.  It supplies a specific wrong target
which the geometry reward actively penalises, at c_geo = 1.0 -- whose own source
comment says the geo gradient dominates the action gradient roughly 3:1.  If
so+G_ref loses, that is equally well explained by "we handed it a target that
fights the reward" as by "a rotational prior does not help".  This experiment
cannot separate those two, and the verdict below says so.  so+G_rand redraws the
target per seed, which removes the "one unlucky draw" objection but not the
adversarial-target confound.

What the design can establish, and what the verdict rests on:
  * whether the published gain reproduces               (so+M_env vs baseline)
  * whether the gain requires the target to be M_env    (so+M_env vs so+G_ref,
                                                         so+M_env vs so+G_rand)
  * whether the paper's described configuration wins    (so+no_aux vs baseline)

How the target is swapped without editing main.py
-------------------------------------------------
LieStructuredPPO touches its env in exactly four places -- env.reset, env.step,
env.k_transform and env.M_env -- plus one hasattr(env, "M_env").  The auxiliary
loss reads self.env.M_env; the geometry reward reads self.M_env inside the env's
own step().  So giving the trainer a proxy whose .M_env is G_ref, and which
forwards everything else to the real env, changes the auxiliary target and
nothing else.  Verified in the selftest.

How to read the result
----------------------
Every statement below requires significance after Holm correction, not a sign.
The script does not draw a conclusion from a point estimate, and states
explicitly when a null is under-powered rather than treating it as evidence.

  so+M_env > baseline, and so+G_ref / so+G_rand are not
        -> the gain requires the auxiliary target to be the environment's own
           rotation.  Combined with so+no_aux < baseline (R-1b), the Task-1
           claim is conditional: given a rotational target, so(32)
           absorbs it better than other parameterisations, which R-4 / R-6 does
           establish.  Sec. 8.1's "involves no term in M_env" is contradicted
           by the released code in either case.

  so+G_ref > baseline significantly
        -> a rotational target that is not the environment's still helps, so the
           term functions as a prior rather than as supervision.  Sec. 8.1 is
           right in substance and wrong only in that one sentence.

  neither reaches significance
        -> report the interval, not a verdict.  Check the power column.

The alignment columns are the mechanism probe: align_env is how close the
learned M_policy ends up to M_env, align_ref how close to the arm's own target.
Note these are measured on the env's own state manifold, not an isotropic
sphere, so they are comparable with the reward.

Outputs (written incrementally)
    s7_cells.csv     arm, seed, r_task, r_geo, auc, rho, align_env, align_ref, ...
    s7_channels.csv  arm, seed, iteration, r_task, r_geo
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import pandas as pd

from _r1core import channel_aucs, channels, instrument, instrument_trainer

CELL_CSV = "s7_cells.csv"
CH_CSV = "s7_channels.csv"

W_DEFAULT = 0.4
GREF_SEED = 777_000          # independent of every training seed
BASELINE = "baseline_ppo"
A_MENV = "so+M_env"
A_GREF = "so+G_ref"
A_NOAUX = "so+no_aux"
A_GRAND = "so+G_rand"          # rotation redrawn per seed
ARMS = (BASELINE, A_MENV, A_GREF, A_GRAND, A_NOAUX)

CELL_COLS = ["w", "arm", "seed", "r_task", "r_geo", "auc", "rho",
             "align_env", "align_ref", "frob_env", "frob_ref",
             "gref_orth", "recon_err", "status"]
CH_COLS = ["w", "arm", "seed", "iteration", "r_task", "r_geo"]


# --------------------------------------------------------------------------- #
# G_ref, and the proxy that swaps only the auxiliary target
# --------------------------------------------------------------------------- #
def random_rotation(k: int, seed: int, device, dtype):
    """A uniformly random element of SO(k) (Haar, via QR with sign fixes)."""
    import torch

    g = torch.Generator(device="cpu").manual_seed(int(seed))
    A = torch.randn(k, k, generator=g, dtype=torch.float64)
    Q, R = torch.linalg.qr(A)
    Q = Q * torch.sign(torch.diagonal(R)).unsqueeze(0)     # Haar correction
    if torch.det(Q) < 0:                                   # force det = +1
        Q[:, 0] = -Q[:, 0]
    orth64 = (Q.T @ Q - torch.eye(k, dtype=torch.float64)).abs().max().item()
    det64 = float(torch.det(Q))
    if not (orth64 < 1e-9 and abs(det64 - 1.0) < 1e-9):
        raise RuntimeError(f"G_ref is not in SO({k}) at float64: "
                           f"max|QtQ-I|={orth64:.2e}, det={det64:.6f}")
    G = Q.to(device=device, dtype=dtype)
    # Verify the object that is actually used, at its dtype -- reporting the
    # float64 residual would certify a matrix the experiment never sees.
    eye = torch.eye(k, device=G.device, dtype=G.dtype)
    orth = float((G.T @ G - eye).abs().max().item())
    det = float(torch.det(G.double()))
    tol = 1e-4 if G.dtype == torch.float32 else 1e-9
    if not (orth < tol and abs(det - 1.0) < 1e-3):
        raise RuntimeError(f"G_ref left SO({k}) after the cast to {G.dtype}: "
                           f"max|GtG-I|={orth:.2e}, det={det:.6f}")
    return G, orth, det


class _AuxTargetProxy:
    """Forwards every attribute to the real env except M_env.

    LieStructuredPPO reads only env.reset, env.step, env.k_transform and
    env.M_env.  The geometry reward is computed inside the env's own step() via
    self.M_env, so it is untouched by this proxy: only the auxiliary loss, which
    reads self.env.M_env from the trainer, sees the substitute.
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
            raise AttributeError("refusing to write M_env through the proxy")
        setattr(object.__getattribute__(self, "_env"), name, value)


# --------------------------------------------------------------------------- #
def make_probes(env, k: int, n_max: int = 4096):
    """Probe vectors drawn from the states the reward is actually evaluated on.

    An isotropic probe set measures tr(M_pol^T M_a)/k, which the closed-form
    Frobenius cosine already gives exactly -- so it adds noise, not information.
    What is not redundant is the alignment restricted to the data manifold: the
    first k dims of GPT-2 embeddings are strongly anisotropic, and a policy can
    match M_env well where the reward looks and poorly elsewhere.  Using the
    env's own state table makes align_* comparable with the reward.
    """
    import torch
    import torch.nn.functional as F

    S = getattr(env, "state_embs", None)
    if S is None:
        S = getattr(env, "prompt_embs", None)
    if S is None:
        return None
    V = S.reshape(-1, S.shape[-1])[:, :k]
    if V.shape[0] > n_max:
        g = torch.Generator(device="cpu").manual_seed(4242)
        idx = torch.randperm(V.shape[0], generator=g)[:n_max].to(V.device)
        V = V[idx]
    return F.normalize(V.double(), dim=-1)


def alignment(M_pol, M_a, probes):
    """(mean cosine over data-manifold probes, Frobenius cosine).

    The first mirrors the reward's own form on the states the reward sees; the
    second, <M_pol, M_a>_F / (|M_pol|_F |M_a|_F), is the probe-free summary and
    for orthogonal matrices equals tr(M_pol^T M_a)/k exactly.
    """
    import torch
    import torch.nn.functional as F

    if M_pol is None or M_a is None:
        return float("nan"), float("nan")
    A = M_pol.double()
    B = M_a.double()
    fro = float((torch.sum(A * B) / (A.norm() * B.norm() + 1e-12)).item())
    if probes is None:
        return float("nan"), fro
    V = probes.to(A.device)
    cos = float(F.cosine_similarity(V @ A.T, V @ B.T, dim=-1).mean().item())
    return cos, fro


# --------------------------------------------------------------------------- #
class _Sink:
    def __init__(self, path, cols):
        self.path, self.cols, self.n = path, cols, 0
        pd.DataFrame(columns=cols).to_csv(path, index=False)

    def add(self, rows):
        if rows:
            pd.DataFrame(rows, columns=self.cols).to_csv(
                self.path, mode="a", header=False, index=False)
            self.n += len(rows)


def _guard(overwrite: bool) -> None:
    for p in (CELL_CSV, CH_CSV):
        if os.path.exists(p) and os.path.getsize(p) > 0 and not overwrite:
            sys.exit(f"{p} exists and is non-empty.  Refusing to truncate a "
                     f"previous run.\nMove it aside, or re-run with --overwrite.")


def collect(n_seeds: int, w: float, overwrite: bool) -> pd.DataFrame:
    _guard(overwrite)
    import torch
    from main import (GPT2EmbeddingProvider, MultiStepTextAlignmentEnv,
                      LiePolicy, BaselinePolicy, ValueNet, LieStructuredPPO,
                      LieAlgebraOps, _default_cfg, _seed_all)

    embedder = GPT2EmbeddingProvider(model_name="gpt2-medium", max_length=64)
    env = MultiStepTextAlignmentEnv(
        embedder, n_prompts=16, n_actions=16, horizon=20,
        reward_noise=0.2, geo_weight=w, sparse_prob=0.7,
    )
    instrument(env)
    horizon = int(env.horizon)
    state_dim, n_actions, k = embedder.hidden_dim, env.n_actions, int(env.k_transform)

    M_env = env.M_env
    probes = make_probes(env, k)
    print(f"\nalignment probes: {0 if probes is None else probes.shape[0]} "
          f"vectors from the env's own state table (not an isotropic sphere)")

    G_ref, orth, det = random_rotation(k, GREF_SEED, M_env.device, M_env.dtype)
    a_cos, a_fro = alignment(G_ref, M_env, probes)
    print(f"G_ref: seed={GREF_SEED}  max|GtG-I|={orth:.2e} (at {M_env.dtype})  "
          f"det={det:.6f}")
    print(f"G_ref vs M_env:  data-manifold cosine = {a_cos:+.4f}   "
          f"Frobenius cosine = {a_fro:+.4f}")
    print("  (near zero on both means G_ref carries no information about M_env,\n"
          "   which is what makes it a fair non-environment target.)\n")
    if abs(a_fro) > 0.2 or (not math.isnan(a_cos) and abs(a_cos) > 0.3):
        sys.exit("G_ref is too close to M_env for the test to be clean; "
                 "change GREF_SEED.")

    cells, chans = _Sink(CELL_CSV, CELL_COLS), _Sink(CH_CSV, CH_COLS)
    rows_all = []
    total = len(ARMS) * n_seeds
    done = 0

    for seed in range(n_seeds):
        # target for the per-seed arm, redrawn every seed
        G_rand, _, _ = random_rotation(k, GREF_SEED + 1 + seed,
                                       M_env.device, M_env.dtype)
        for arm in ARMS:
            _seed_all(seed)
            cfg = _default_cfg()
            policy = None
            target = None                      # the arm's own auxiliary target
            try:
                if arm == BASELINE:
                    policy = BaselinePolicy(state_dim, n_actions)
                    trainer = LieStructuredPPO(env, policy, ValueNet(state_dim),
                                               cfg, use_lie_projection=False)
                else:
                    if arm == A_NOAUX:
                        cfg.geo_aux_coef = 0.0
                    policy = LiePolicy(state_dim, n_actions, k=k, algebra="so")
                    trainer = LieStructuredPPO(env, policy, ValueNet(state_dim),
                                               cfg, use_lie_projection=True,
                                               algebra="so")
                    if arm == A_GREF:
                        target = G_ref
                    elif arm == A_GRAND:
                        target = G_rand
                    if target is not None:
                        # Only the auxiliary target changes; the reward does not.
                        trainer.env = _AuxTargetProxy(env, target)

                instrument_trainer(trainer, env, None, horizon, strict=True)
                res = trainer.train(verbose=False)
            except KeyboardInterrupt:
                print("\nInterrupted by user; partial results are already in "
                      f"{CELL_CSV}.")
                raise
            except Exception as e:                          # noqa: BLE001
                print(f"  seed {seed:2d}  {arm:13s}  FAILED: "
                      f"{type(e).__name__}: {e}", flush=True)
                row = {c: np.nan for c in CELL_COLS}
                row.update(w=w, arm=arm, seed=seed,
                           status=f"{type(e).__name__}: {e}"[:120])
                cells.add([row]); rows_all.append(row); done += 1
                continue

            pairs = channels(res, trainer)
            chans.add([{"w": w, "arm": arm, "seed": seed, "iteration": i,
                        "r_task": rt, "r_geo": rg}
                       for i, (rt, rg) in enumerate(pairs, start=1)])

            rt_m = float(np.mean([p[0] for p in pairs]))
            rg_m = float(np.mean([p[1] for p in pairs]))
            _, _, a_tot = channel_aucs(pairs, w, horizon)
            auc = float(res["auc"])

            M_pol = None
            if hasattr(policy, "theta"):
                with torch.no_grad():
                    M_pol = LieAlgebraOps.matrix_exp(policy._proj(policy.theta.data))
            al_env, fr_env = alignment(M_pol, M_env, probes)
            # align_ref is against this arm's own target, whatever it is
            al_ref, fr_ref = alignment(M_pol, target if target is not None else M_env,
                                       probes)
            if arm in (BASELINE, A_NOAUX):
                al_ref = fr_ref = float("nan")   # these arms have no target

            spec = [sp for sp in res.get("spectral_radii", [])
                    if sp is not None and math.isfinite(sp)]
            recon = abs(a_tot - auc)
            finite = all(math.isfinite(x) for x in (rt_m, rg_m, auc, recon))
            row = {"w": w, "arm": arm, "seed": seed, "r_task": rt_m,
                   "r_geo": rg_m, "auc": auc,
                   "rho": float(np.median(spec)) if spec else np.nan,
                   "align_env": al_env, "align_ref": al_ref,
                   "frob_env": fr_env, "frob_ref": fr_ref,
                   "gref_orth": orth, "recon_err": recon,
                   "status": "ok" if finite else "non-finite"}
            cells.add([row]); rows_all.append(row); done += 1
            print(f"  [{done:3d}/{total}] seed {seed:2d}  {arm:13s}  "
                  f"AUC={auc:7.3f}  r_task={rt_m:.4f}  r_geo={rg_m:.4f}  "
                  f"align(M_env)={al_env:+.3f}  align(target)={al_ref:+.3f}",
                  flush=True)

    df = pd.DataFrame(rows_all, columns=CELL_COLS)
    print(f"\nWrote {CELL_CSV} ({cells.n} rows) and {CH_CSV} ({chans.n} rows)")
    return df


# --------------------------------------------------------------------------- #
def _pair(df, arm_a, arm_b, col, seeds=None):
    """method - control, matched on seed. Returns (differences, n, seed_set)."""
    a = df[df.arm == arm_a].set_index("seed")[col].astype(float)
    b = df[df.arm == arm_b].set_index("seed")[col].astype(float)
    idx = a.index.intersection(b.index)
    if seeds is not None:
        idx = idx.intersection(pd.Index(sorted(seeds)))
    d = (a.loc[idx] - b.loc[idx]).replace([np.inf, -np.inf], np.nan).dropna()
    return d.to_numpy(), len(d), set(d.index)


COMPARISONS = [
    (A_MENV, BASELINE, "as released: does the published result reproduce?"),
    (A_GREF, BASELINE, "one fixed non-environment rotation as the target"),
    (A_GRAND, BASELINE, "target redrawn per seed (no single-draw artefact)"),
    (A_NOAUX, BASELINE, "the configuration Sec. 8.1 actually describes"),
    (A_MENV, A_GREF, "how much of the gain requires the target to be M_env?"),
]


def analyse(df: pd.DataFrame) -> None:
    from sppg_defense.stats.tests import holm, paired_report, tost_paired

    dup = df.duplicated(["w", "arm", "seed"]).sum()
    if dup:
        sys.exit(f"{dup} duplicated (w, arm, seed) rows in {CELL_CSV}.")

    if "recon_err" not in df.columns:
        print("\n!! WARNING: this CSV has no 'recon_err' column, so the "
              "channel-decomposition\n   guarantee cannot be checked.  It was "
              "probably produced by an older version.")
    else:
        e = pd.to_numeric(df["recon_err"], errors="coerce")
        nf = int((~np.isfinite(e)).sum())
        e = e[np.isfinite(e)]
        if len(e):
            print(f"\nchannel-reconstruction error: max = {e.max():.3e}"
                  + (f"   [{nf} non-finite run(s) excluded]" if nf else ""))

    bad = df[df.status != "ok"]
    if len(bad):
        print(f"\n{len(bad)} failed run(s):")
        for _, r in bad.iterrows():
            print(f"    {r.arm} seed={r.seed}: {r.status}")
    df = df[df.status == "ok"]

    # A common seed set across every arm: the design is paired, and comparing
    # means taken over different seed subsets would break that.
    per_arm = {a: set(df[df.arm == a].seed) for a in ARMS if (df.arm == a).any()}
    common = set.intersection(*per_arm.values()) if per_arm else set()
    dropped = sorted(set(df.seed) - common)
    if dropped:
        print(f"\nrestricting every comparison to the {len(common)} seed(s) "
              f"present in all arms; dropped {dropped}")

    print("\n" + "=" * 96)
    print(f"S-7  PER-ARM SUMMARY  (n = {len(common)} common seeds, mean +/- sd)")
    print("=" * 96)
    print(f"  {'arm':>14s} {'n':>3s} {'AUC':>16s} {'r_task':>15s} "
          f"{'r_geo':>15s} {'align(M_env)':>13s} {'align(target)':>14s}")
    for arm in ARMS:
        d = df[(df.arm == arm) & (df.seed.isin(common))]
        if not len(d):
            print(f"  {arm:>14s}   (no runs)"); continue

        def ms(c):
            v = d[c].astype(float).dropna()
            return (f"{v.mean():7.3f}+/-{v.std(ddof=1):5.3f}"
                    if len(v) > 1 else f"{'n/a':>13s}")

        def m1(c, wdt=13):
            v = d[c].astype(float).dropna()
            return f"{v.mean():+{wdt}.3f}" if len(v) else f"{'n/a':>{wdt}s}"

        print(f"  {arm:>14s} {len(d):>3d} {ms('auc'):>16s} {ms('r_task'):>15s} "
              f"{ms('r_geo'):>15s} {m1('align_env'):>13s} {m1('align_ref', 14):>14s}")

    # ---- all tests in one Holm family; AUC is primary ---------------------
    results = {}
    rows, pvals, keys = [], [], []
    for col in ("auc", "r_task", "r_geo"):
        for a, b, why in COMPARISONS:
            d, n, _ = _pair(df, a, b, col, common)
            if n < 3:
                rows.append(dict(col=col, a=a, b=b, why=why, n=n))
                pvals.append(1.0); keys.append((col, a, b)); continue
            r = paired_report(d, np.zeros_like(d))
            rows.append(dict(col=col, a=a, b=b, why=why, n=n, diff=r.diff,
                             ci=r.diff_ci, dz=r.cohen_dz, p=r.t_p,
                             power=r.achieved_power, d=d))
            pvals.append(r.t_p if np.isfinite(r.t_p) else 1.0)
            keys.append((col, a, b))
    hp = holm(pvals, 0.05)
    for row, key, ph, rj in zip(rows, keys, hp["p_holm"], hp["reject"]):
        row["p_holm"], row["sig"] = ph, rj
        results[key] = row

    for col, label in [("auc", "TOTAL AUC  (primary)"),
                       ("r_task", "TASK CHANNEL  (secondary)"),
                       ("r_geo", "GEOMETRY CHANNEL  (secondary)")]:
        print("\n" + "=" * 96)
        print(f"S-7  {label}   Holm-corrected over all "
              f"{len(rows)} tests in this run")
        print("=" * 96)
        for r in [x for x in rows if x["col"] == col]:
            head = f"  {r['a']:>12s} - {r['b']:<13s}"
            if "diff" not in r:
                print(f"{head} n={r['n']}  (insufficient seeds)"); continue
            print(f"{head} n={r['n']:>2d}  {r['diff']:>+9.4f}"
                  f"{'*' if r['sig'] else ' '} "
                  f"[{r['ci'][0]:>+8.4f},{r['ci'][1]:>+8.4f}]  "
                  f"dz={r['dz']:>+6.2f}  p_holm={r['p_holm']:>9.2e}  "
                  f"power={(r['power'] if r['power'] is not None else float('nan')):.2f}")
            print(f"       {r['why']}")
        print("  * = significant after Holm.  'power' is post-hoc power at the "
              "observed effect;\n  it is a restatement of p and says nothing about "
              "a null.  For nulls read the CI\n  and the equivalence test below.")

    # ---- equivalence, so a null can be stated rather than assumed ---------
    ref = results.get(("auc", A_MENV, BASELINE))
    print("\n" + "=" * 96)
    print("EQUIVALENCE (TOST): are the non-M_env arms equivalent to baseline?")
    if ref is None or "diff" not in ref:
        print("  cannot set a margin without the so+M_env comparison.")
    else:
        margin = 0.25 * abs(ref["diff"])
        print(f"  Margin = 25% of the published gain = {margin:.4f} AUC.")
        print("=" * 96)
        for arm in (A_GREF, A_GRAND, A_NOAUX):
            r = results.get(("auc", arm, BASELINE))
            if r is None or "diff" not in r:
                print(f"  {arm:>12s}   (not evaluable)"); continue
            try:
                t = tost_paired(r["d"], np.zeros_like(r["d"]), margin=margin)
            except ValueError as exc:
                print(f"  {arm:>12s}   TOST undefined: {exc}"); continue
            print(f"  {arm:>12s}  diff={r['diff']:+.4f}  TOST p={t['p_tost']:.4f}"
                  f"  smallest margin that would hold={t['smallest_equivalence_margin']:.4f}"
                  f"   {'equivalent to baseline' if t['equivalent'] else 'not shown equivalent'}")

    # ---- the verdict, which requires significance -------------------------
    print("\n" + "=" * 96)
    print("VERDICT")
    print("=" * 96)
    r_menv = results.get(("auc", A_MENV, BASELINE))
    r_gref = results.get(("auc", A_GREF, BASELINE))
    r_grand = results.get(("auc", A_GRAND, BASELINE))
    if any(r is None or "diff" not in r for r in (r_menv, r_gref, r_grand)):
        print("  Not enough complete seeds to decide.")
        return

    print(f"  so+M_env  - baseline : {r_menv['diff']:+.4f}  "
          f"p_holm={r_menv['p_holm']:.2e}  "
          f"{'significant' if r_menv['sig'] else 'not significant'}")
    for nm, r in (("so+G_ref ", r_gref), ("so+G_rand", r_grand)):
        print(f"  {nm} - baseline : {r['diff']:+.4f}  "
              f"p_holm={r['p_holm']:.2e}  "
              f"{'significant' if r['sig'] else 'not significant'}")

    menv_wins = r_menv["sig"] and r_menv["diff"] > 0
    alt_wins = [r for r in (r_gref, r_grand) if r["sig"] and r["diff"] > 0]

    if not menv_wins:
        print("\n  -> The published configuration does not significantly beat the\n"
              "     baseline in this run.  Nothing further can be concluded; "
              "check the\n     run before interpreting any other arm.")
    elif alt_wins:
        print("\n  -> A rotational target that is not the environment's rotation "
              "also\n     significantly beats the baseline.  The auxiliary loss "
              "functions as a\n     prior, not purely as supervision.  Sec. 8.1 "
              "is right in substance,\n     but the sentence 'involves no term in "
              "M_env' does not match\n     the released code.")
    else:
        share = (r_gref["diff"] / r_menv["diff"] * 100)
        print(f"\n  -> Only the arm whose auxiliary target is M_env beats the "
              f"baseline.\n     A non-environment rotation retains "
              f"{share:.0f}% of the published gain and\n     does not reach "
              f"significance.  On this evidence the Task-1 advantage\n"
              "     requires the auxiliary loss to be given the environment's own\n"
              "     rotation, and the claim is conditional: given a\n"
              "     rotational target, so(32) absorbs it better than other\n"
              "     parameterisations (R-4 / R-6 establishes that separately).")
        print("\n     CONFOUND, stated rather than hidden: so+G_ref supplies a "
              "target the\n     geometry reward penalises, so 'a wrong target "
              "hurts' and 'a prior does\n     not help' are not separated here.  "
              "The arm that matches Sec. 8.1's\n     prose is so+no_aux, whose "
              "result is reported above.")
    print("\n  Mechanism: compare align(M_env) against align(target) per arm in "
          "the summary.\n  An arm that ends aligned with its own target and not "
          "with M_env followed the\n  auxiliary loss and learned nothing about "
          "the environment from reward.")


# --------------------------------------------------------------------------- #
def selftest() -> None:
    import torch
    import torch.nn.functional as F
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")

    print("(1) G_ref is an independent element of SO(32)")
    k = 32
    G64, o64, d64 = random_rotation(k, GREF_SEED, "cpu", torch.float64)
    check("orthogonal at float64", o64 < 1e-9, f"max|GtG-I|={o64:.2e}")
    check("det = +1", abs(d64 - 1.0) < 1e-9, f"det={d64:.9f}")
    G32, o32, d32 = random_rotation(k, GREF_SEED, "cpu", torch.float32)
    check("orthogonality is verified at the dtype actually used",
          1e-12 < o32 < 1e-4, f"float32 max|GtG-I|={o32:.2e}")
    # isometry, properly: same probes, per-row norms
    g = torch.Generator().manual_seed(5)
    V = torch.randn(256, k, generator=g, dtype=torch.float64)
    err = ((V @ G64.T).norm(dim=-1) - V.norm(dim=-1)).abs().max().item()
    check("isometry: |Gv| == |v| for every probe", err < 1e-12, f"max err={err:.2e}")
    # a non-orthogonal matrix must FAIL that test (the check is not vacuous)
    B = torch.randn(k, k, generator=g, dtype=torch.float64)
    errb = ((V @ B.T).norm(dim=-1) - V.norm(dim=-1)).abs().max().item()
    check("the isometry check can fail", errb > 1.0, f"non-orthogonal err={errb:.2f}")
    G2, _, _ = random_rotation(k, GREF_SEED + 1, "cpu", torch.float64)
    check("different seed -> different rotation", not torch.allclose(G64, G2))
    probes = F.normalize(torch.randn(512, k, generator=g, dtype=torch.float64), dim=-1)
    c, f = alignment(G64, G2, probes)
    check("two independent rotations are near-orthogonal",
          abs(c) < 0.2 and abs(f) < 0.2, f"probe={c:+.4f} frob={f:+.4f}")
    c, f = alignment(G64, G64, probes)
    check("a rotation aligned with itself is 1",
          abs(c - 1) < 1e-9 and abs(f - 1) < 1e-9, f"{c:.6f} / {f:.6f}")

    print("(2) the proxy swaps only the auxiliary target")

    class FakeEnv:
        def __init__(self):
            self.M_env = torch.eye(4)
            self.k_transform, self.horizon, self.geo_weight = 4, 20, 0.4
            self.calls = []

        def reset(self):
            self.calls.append("reset"); return "S"

        def step(self, a, M=None):
            self.calls.append(("step", a)); return "S", 1.0, True

        def _geometry_reward(self, v, M):
            return float(self.M_env.sum().item())

    fe = FakeEnv(); Gr = torch.full((4, 4), 9.0)
    px = _AuxTargetProxy(fe, Gr)
    check("proxy.M_env is the substitute", torch.equal(px.M_env, Gr))
    check("real env keeps its own M_env", torch.equal(fe.M_env, torch.eye(4)))
    check("hasattr(proxy, 'M_env')", hasattr(px, "M_env"))
    check("k_transform / horizon / geo_weight delegate",
          px.k_transform == 4 and px.horizon == 20 and px.geo_weight == 0.4)
    check("reset delegates", px.reset() == "S" and "reset" in fe.calls)
    check("step delegates", px.step(3)[1] == 1.0 and ("step", 3) in fe.calls)
    check("the env's own geometry reward still uses its M_env",
          fe._geometry_reward(None, None) == 4.0)
    fe.step = lambda a, M=None: ("W", 2.0, False)
    check("instance-level wrappers are reached through the proxy",
          px.step(0)[1] == 2.0)
    try:
        px.M_env = torch.zeros(4, 4); check("proxy blocks writes to M_env", False)
    except AttributeError:
        check("proxy blocks writes to M_env", True)
    px.other = 5
    check("other writes reach the real env", getattr(fe, "other", None) == 5)
    try:
        px.definitely_missing
        check("unknown attribute raises AttributeError", False)
    except AttributeError:
        check("unknown attribute raises AttributeError", True)

    print("(3) alignment")
    A = torch.eye(8, dtype=torch.float64)
    P = F.normalize(torch.randn(64, 8, generator=g, dtype=torch.float64), dim=-1)
    c, f = alignment(A, A, P)
    check("identity vs identity = 1", abs(c - 1) < 1e-12 and abs(f - 1) < 1e-12)
    c, f = alignment(A, -A, P)
    check("A vs -A = -1", abs(c + 1) < 1e-12 and abs(f + 1) < 1e-12)
    c, f = alignment(None, A, P)
    check("None -> nan (arms with no theta)", math.isnan(c) and math.isnan(f))
    c, f = alignment(A, A, None)
    check("no probes -> frobenius still returned", math.isnan(c) and abs(f - 1) < 1e-12)

    class EnvWithStates:
        state_embs = torch.randn(3, 4, 5, 64)
    pr = make_probes(EnvWithStates(), 8)
    check("probes come from the env state table",
          pr is not None and pr.shape == (60, 8)
          and abs(pr.norm(dim=-1).mean().item() - 1) < 1e-9,
          f"shape={tuple(pr.shape)}")
    check("no state table -> None", make_probes(object(), 8) is None)

    print("(4) the verdict requires significance, not a sign")
    rng = np.random.default_rng(0)

    def frame(eff, n=12, noise=0.12):
        rows = []
        for s_ in range(n):
            se = rng.normal(0, .10)
            for arm, base in eff.items():
                rows.append(dict(w=0.4, arm=arm, seed=s_,
                                 r_task=0.18, r_geo=0.54,
                                 auc=base + se + rng.normal(0, noise), rho=1.0,
                                 align_env=0.0, align_ref=0.0, frob_env=0.0,
                                 frob_ref=0.0, gref_orth=1e-7, recon_err=1e-15,
                                 status="ok"))
        return pd.DataFrame(rows)

    import contextlib, io

    def run(df):
        b = io.StringIO()
        with contextlib.redirect_stdout(b):
            analyse(df)
        return b.getvalue()

    # (a) only M_env wins -> supervision branch
    out = run(frame({BASELINE: 6.50, A_MENV: 8.73, A_GREF: 6.52,
                     A_GRAND: 6.49, A_NOAUX: 5.95}))
    check("supervision branch when only M_env wins",
          "requires the auxiliary loss to be given" in out)
    check("the confound is stated, not hidden", "CONFOUND" in out)
    # (b) G_ref also wins -> prior branch
    out = run(frame({BASELINE: 6.50, A_MENV: 8.73, A_GREF: 7.80,
                     A_GRAND: 7.75, A_NOAUX: 5.95}))
    check("prior branch when a non-env rotation also wins",
          "functions as a\n     prior" in out or "functions as a" in out)
    # (c) a tiny positive G_ref effect swamped by noise must not claim a win
    out = run(frame({BASELINE: 6.50, A_MENV: 8.73, A_GREF: 6.53,
                     A_GRAND: 6.52, A_NOAUX: 5.95}, noise=1.2))
    check("a positive but non-significant G_ref does not trigger the prior branch",
          "functions as a" not in out)
    # (d) M_env itself failing to reproduce must stop the analysis
    out = run(frame({BASELINE: 6.50, A_MENV: 6.51, A_GREF: 6.50,
                     A_GRAND: 6.50, A_NOAUX: 6.49}, noise=0.8))
    check("a failed reproduction halts the verdict",
          "does not significantly beat the" in out)

    print("(5) bookkeeping")
    d = frame({BASELINE: 6.5, A_MENV: 8.7, A_GREF: 6.5, A_GRAND: 6.5, A_NOAUX: 6.0})
    d2 = d[~((d.arm == A_GREF) & (d.seed.isin([0, 1])))]
    out = run(d2)
    check("seeds missing from one arm are dropped from all comparisons",
          "restricting every comparison to the 10 seed(s)" in out)
    d3 = d.drop(columns=["recon_err"])
    check("a CSV without recon_err warns loudly",
          "no 'recon_err' column" in run(d3))
    d4 = pd.concat([d, d.iloc[:1]], ignore_index=True)
    try:
        run(d4); check("duplicate (w, arm, seed) detected", False)
    except SystemExit:
        check("duplicate (w, arm, seed) detected", True)
    check("equivalence test is reported", "EQUIVALENCE (TOST)" in run(d))

    print("\n" + ("SELFTEST PASS" if ok else "SELFTEST FAIL"))
    if not ok:
        sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--w", type=float, default=W_DEFAULT)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--analyse-only", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        selftest(); return
    if a.analyse_only:
        if not os.path.exists(CELL_CSV):
            sys.exit(f"{CELL_CSV} not found; run without --analyse-only first.")
        df = pd.read_csv(CELL_CSV)
    else:
        df = collect(a.seeds, a.w, a.overwrite)
    analyse(df)


if __name__ == "__main__":
    main()
