"""
R-4 + R-6: multi-seed algebra ablation with a dimension-matched random control.
Also re-runs the R-1 contrast for free, since baseline PPO is included.

    python run_r4.py --selftest        # instant, no GPU, no model -- run first
    python run_r4.py --seeds 20        # ~20-40 min
    python run_r4.py --analyse-only    # re-analyse existing csv files

Requires _r1core.py in the same directory.  Makes no edits to main.py.

Why
---
R-1 showed the SP-PG-vs-PPO headline is carried by a reward channel the M=I
control cannot reach.  That leaves this ablation as the only comparison in the
paper that speaks to the so(32) constraint itself: every arm here keeps M and
the geometry reward, and only the algebra differs.  It is currently reported at
seed 0, where the so(32) cell moves by 0.245 on RNG ordering alone.

Arms
    so(32)       compact Lie          dim  496
    sl(32)       non-compact Lie      dim 1023
    sym(32)      non-Lie subspace     dim  528
    gl(32)       unconstrained        dim 1024
    random(496)  random subspace      dim  496   <- R-6, dimension-matched
    baseline_ppo no M at all                     <- R-1 contrast; reported
                                                    separately, excluded from
                                                    the omnibus test

Caveats
  * A fresh random subspace is drawn per seed, so the random arm carries
    subspace-draw variance on top of training variance.  That is conservative
    for "so(32) beats random" and anti-conservative for any claim that they are
    equivalent.  The per-arm SD is printed; quote it.
  * A uniformly random 496-dim subspace of R^{32x32} generically has a large
    symmetric component, so exp(P_rand(theta)) is not an isometry.  The arm
    controls for dimension while also varying compactness; it is a
    dimension-matched control, not a dimension-ONLY control.
  * Re-seeding per arm gives identical initialisation but not common random
    numbers: the number of np.random draws per step depends on whether
    task_r > 0, so the noise stream desynchronises once actions diverge.  This
    is a randomised block design, not a CRN design.

Outputs (written incrementally, so a crash never costs completed arms)
    r4_channels.csv   arm, seed, iteration, r_task, r_geo
    r4_arms.csv       arm, seed, auc, auc_task, auc_geo, auc_recon,
                      recon_err, final_return, threshold_crossing,
                      median_spectral_radius, status
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import pandas as pd

from _r1core import channel_aucs, channels, instrument, instrument_trainer

CH_CSV = "r4_channels.csv"
ARM_CSV = "r4_arms.csv"

ARMS = [
    ("so(32)", "so"),
    ("sl(32)", "sl"),
    ("sym(32)", "sym"),
    ("gl(32)", "full"),
    ("random(496)", "random"),
]
REFERENCE = "so(32)"
BASELINE = "baseline_ppo"

ARM_COLS = ["arm", "seed", "auc", "auc_task", "auc_geo", "auc_recon",
            "recon_err", "final_return", "threshold_crossing",
            "median_spectral_radius", "status"]
CH_COLS = ["arm", "seed", "iteration", "r_task", "r_geo"]


# --------------------------------------------------------------------------- #
# R-6: dimension-matched random subspace
# --------------------------------------------------------------------------- #
def make_random_projector(k: int, dim: int, seed: int, device, dtype):
    """
    Frobenius-orthogonal projector onto a uniformly random `dim`-dimensional
    subspace of R^{k x k}.

    Returns a plain function of one argument, so attaching it to an object does
    not bind `self` -- which is how main.py's
    `getattr(self.ops, f"project_{alg}")(X)` dispatch calls it.  Differentiable,
    because the geometric auxiliary loss backpropagates through it.
    """
    import torch

    n = k * k
    if not 1 <= dim <= n:
        raise ValueError(f"dim must be in [1, {n}], got {dim}")

    g = torch.Generator(device="cpu").manual_seed(int(seed))
    A = torch.randn(n, dim, generator=g, dtype=torch.float64)
    Q, _ = torch.linalg.qr(A, mode="reduced")
    if Q.shape != (n, dim):
        raise RuntimeError(f"QR gave {tuple(Q.shape)}, expected {(n, dim)}")

    B = Q.to(device=device, dtype=dtype)

    # Check the object that will actually run, at its dtype -- not the float64
    # intermediate.  float32 orthonormality is ~1e-6, which is the number that
    # matters for the projection error during training.
    eye = torch.eye(dim, device=B.device, dtype=B.dtype)
    err = (B.T @ B - eye).abs().max().item()
    tol = 1e-4 if B.dtype == torch.float32 else 1e-9
    if not math.isfinite(err) or err > tol:
        raise RuntimeError(f"basis not orthonormal at {B.dtype}: "
                           f"max|BtB - I| = {err:.3e} > {tol:.0e}")

    def project_random(X):
        shape = X.shape
        flat = X.reshape(-1, n)
        return ((flat @ B) @ B.T).reshape(shape)

    return project_random, B, err


def attach_random_arm(policy, trainer, k: int, dim: int, seed: int):
    """Give this policy/trainer pair a private random-subspace projection."""
    import torch

    proj, B, orth_err = make_random_projector(
        k, dim, seed, policy.theta.device, policy.theta.dtype
    )
    # Per-OBJECT attachment: policy.ops and trainer.ops are separate instances
    # built in their own __init__, so nothing touches LieAlgebraOps itself.
    policy.ops.project_random = proj
    trainer.ops.project_random = proj
    policy.algebra = "random"
    trainer.algebra = "random"
    # Start inside the subspace.  Provably a no-op for the trajectory (every
    # consumer projects anyway, and _proj_params projects after step 1), but it
    # makes iteration-0 diagnostics meaningful.
    with torch.no_grad():
        policy.theta.data.copy_(proj(policy.theta.data))
    return orth_err


# --------------------------------------------------------------------------- #
# collection
# --------------------------------------------------------------------------- #
class _Sink:
    """Append-as-you-go CSV writer: a crash never costs completed arms."""

    def __init__(self, path, cols):
        self.path, self.cols, self.n = path, cols, 0
        pd.DataFrame(columns=cols).to_csv(path, index=False)

    def add(self, rows):
        if not rows:
            return
        pd.DataFrame(rows, columns=self.cols).to_csv(
            self.path, mode="a", header=False, index=False)
        self.n += len(rows)


def collect(n_seeds: int, with_baseline: bool) -> pd.DataFrame:
    from main import (GPT2EmbeddingProvider, MultiStepTextAlignmentEnv,
                      LiePolicy, BaselinePolicy, ValueNet, LieStructuredPPO,
                      _default_cfg, _seed_all)

    embedder = GPT2EmbeddingProvider(model_name="gpt2-medium", max_length=64)
    env = MultiStepTextAlignmentEnv(
        embedder, n_prompts=16, n_actions=16, horizon=20,
        reward_noise=0.2, geo_weight=0.4, sparse_prob=0.7,
    )
    # Hard exits, not asserts: `python -O` strips asserts and would let the
    # decomposition run with the wrong constants.
    w = instrument(env)
    horizon = int(env.horizon)
    if abs(w - 0.4) > 1e-12:
        print(f"NOTE: env.geo_weight = {w} (not 0.4). Using the live value.")
    if horizon != 20:
        print(f"NOTE: env.horizon = {horizon} (not 20). Using the live value.")

    state_dim = embedder.hidden_dim
    n_actions = env.n_actions
    k = int(env.k_transform)
    dim_so = k * (k - 1) // 2                      # 496 at k=32

    plan = list(ARMS) + ([(BASELINE, None)] if with_baseline else [])
    ch_sink, arm_sink = _Sink(CH_CSV, CH_COLS), _Sink(ARM_CSV, ARM_COLS)
    arm_rows_all = []

    for seed in range(n_seeds):
        for label, algebra in plan:
            # Re-seed immediately before every arm.  main.py's _single_run seeds
            # once and trains the baseline first, which is exactly why its
            # seed-0 so(32) score differs from run_rl_ablation's.
            _seed_all(seed)
            cfg = _default_cfg()                   # fresh per arm

            try:
                if algebra is None:
                    policy = BaselinePolicy(state_dim, n_actions)
                    trainer = LieStructuredPPO(
                        env, policy, ValueNet(state_dim), cfg,
                        use_lie_projection=False)
                else:
                    init_alg = "so" if algebra == "random" else algebra
                    policy = LiePolicy(state_dim, n_actions, k=k,
                                       algebra=init_alg)
                    trainer = LieStructuredPPO(
                        env, policy, ValueNet(state_dim), cfg,
                        use_lie_projection=True, algebra=init_alg)
                    if algebra == "random":
                        attach_random_arm(policy, trainer, k, dim_so,
                                          seed=100_000 + seed)

                instrument_trainer(trainer, env, w, horizon, strict=True)
                res = trainer.train(verbose=False)
                status = "ok"
            except Exception as e:                 # noqa: BLE001
                # sym(32) can reach spectral radius > 2400; if matrix_exp
                # overflows, torch.linalg.eigvals aborts rather than returning
                # NaN.  Record the failure and keep the other arms.
                print(f"  seed {seed:2d}  {label:13s}  FAILED: "
                      f"{type(e).__name__}: {e}", flush=True)
                row = {c: np.nan for c in ARM_COLS}
                row.update(arm=label, seed=seed,
                           status=f"{type(e).__name__}: {e}"[:120])
                arm_sink.add([row])
                arm_rows_all.append(row)
                continue

            pairs = channels(res, trainer)
            ch_sink.add([{"arm": label, "seed": seed, "iteration": i,
                          "r_task": rt, "r_geo": rg}
                         for i, (rt, rg) in enumerate(pairs, start=1)])

            a_task, a_geo, a_tot = channel_aucs(pairs, w, horizon)
            auc = float(res["auc"])
            recon_err = abs(a_tot - auc)
            spec = [s for s in res.get("spectral_radii", [])
                    if s is not None and math.isfinite(s)]
            tc = res.get("threshold_crossing")

            row = {
                "arm": label, "seed": seed, "auc": auc,
                "auc_task": a_task, "auc_geo": a_geo, "auc_recon": a_tot,
                "recon_err": recon_err,
                "final_return": float(res["returns"][-1]),
                "threshold_crossing": float(tc) if tc is not None else np.nan,
                "median_spectral_radius": (float(np.median(spec)) if spec
                                           else np.nan),
                "status": "ok" if math.isfinite(auc) else "non-finite auc",
            }
            arm_sink.add([row])
            arm_rows_all.append(row)

            flag = "" if recon_err < 1e-6 else f"  !! recon_err={recon_err:.2e}"
            print(f"  seed {seed:2d}  {label:13s}  AUC={auc:8.3f}  "
                  f"task={a_task:7.3f}  geo={a_geo:7.3f}  "
                  f"rho={row['median_spectral_radius']:.4g}{flag}", flush=True)

    print(f"\nWrote {CH_CSV} ({ch_sink.n} rows) and {ARM_CSV} ({arm_sink.n} rows)")
    return pd.DataFrame(arm_rows_all, columns=ARM_COLS)


# --------------------------------------------------------------------------- #
# analysis
# --------------------------------------------------------------------------- #
def _clean(arms: pd.DataFrame, col: str) -> pd.DataFrame:
    """Drop non-finite rows, and any seed that is then incomplete.

    ablation_report rejects NaN but not +/-inf, which would flow into the
    summary as inf while Friedman still returned a finite-looking p-value.
    """
    d = arms.copy()
    bad = ~np.isfinite(d[col].to_numpy(dtype=float))
    if bad.any():
        print(f"  dropping {int(bad.sum())} non-finite '{col}' rows: "
              f"{sorted(set(zip(d.loc[bad, 'arm'], d.loc[bad, 'seed'])))}")
        d = d.loc[~bad]
    counts = d.groupby("seed")["arm"].nunique()
    full = counts[counts == d["arm"].nunique()].index
    dropped = sorted(set(d["seed"]) - set(full))
    if dropped:
        print(f"  dropping {len(dropped)} incomplete seed(s) to keep the "
              f"design paired: {dropped}")
    return d[d["seed"].isin(full)]


def analyse(arms: pd.DataFrame) -> None:
    from sppg_defense.rl.analysis import ablation_report

    if "recon_err" in arms:
        e = pd.to_numeric(arms["recon_err"], errors="coerce").dropna()
        if len(e):
            print(f"\nchannel-reconstruction error vs main.py's own AUC: "
                  f"max = {e.max():.3e}  (must be ~1e-12; larger means the "
                  f"channels do not decompose the reported AUC)")

    if BASELINE in set(arms["arm"]):
        m = arms[arms["arm"] == REFERENCE].set_index("seed")["auc"]
        b = arms[arms["arm"] == BASELINE].set_index("seed")["auc"]
        common = m.index.intersection(b.index)
        d = (m.loc[common] - b.loc[common]).astype(float).dropna()
        if len(d):
            print("\n" + "=" * 72)
            print("R-1 CONTRAST (excluded from the omnibus test): "
                  "so(32) vs baseline PPO")
            print("=" * 72)
            print(f"  mean AUC gap = {d.mean():+.4f}   n = {len(d)}   "
                  f"so(32) wins {int((d > 0).sum())}/{len(d)} seeds")
            print("  Confounded by reward-channel access (R-1); reported for "
                  "continuity only,\n  not as evidence about the algebra.")

    abl = arms[arms["arm"] != BASELINE]
    if abl["arm"].nunique() < 3:
        print(f"\nOnly {abl['arm'].nunique()} ablation arms; Friedman needs "
              f">= 3. Skipping.")
        return

    for col, title in [("auc", "TOTAL AUC"),
                       ("auc_task", "TASK CHANNEL"),
                       ("auc_geo", "GEOMETRY CHANNEL")]:
        print("\n" + "=" * 72)
        print(f"R-4 / R-6 ABLATION  --  {title}   (reference: {REFERENCE})")
        print("=" * 72)
        d = _clean(abl, col)
        if d.empty or d["seed"].nunique() < 2:
            print("  not enough complete seeds after cleaning; skipped.")
            continue
        try:
            out = ablation_report(d, value_col=col, reference=REFERENCE)
        except Exception as e:                      # noqa: BLE001
            print(f"  !! ablation_report FAILED: {type(e).__name__}: {e}")
            continue
        _print_report(out)

    print("\nNote: random(496) redraws its subspace every seed, so its SD "
          "includes subspace variance.\nThat is conservative for "
          "'so(32) > random' and anti-conservative for 'so(32) == random'.")


def _print_report(out: dict) -> None:
    def rows(key):
        v = out.get(key)
        if v is None:
            return []
        return v.to_dict("records") if isinstance(v, pd.DataFrame) else list(v)

    for key, head in [("summary", "Per-arm"),
                      ("pairwise", "Pairwise vs reference"),
                      ("contrasts", "Pre-registered contrasts")]:
        r = rows(key)
        if r:
            print(f"\n  -- {head} --")
            print(pd.DataFrame(r).to_string(index=False, max_colwidth=34))

    for key, val in out.items():
        if key not in {"summary", "pairwise", "contrasts"}:
            print(f"\n  {key}: {val}")


# --------------------------------------------------------------------------- #
# selftest
# --------------------------------------------------------------------------- #
def selftest() -> None:
    import torch
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")

    print("(1) random projector, float64")
    k, dim = 32, 496
    proj, B, oe = make_random_projector(k, dim, 1, "cpu", torch.float64)
    X = torch.randn(k, k, dtype=torch.float64)
    Y = torch.randn(k, k, dtype=torch.float64)
    PX = proj(X)
    check("idempotent", (proj(PX) - PX).abs().max().item() < 1e-10,
          f"{(proj(PX)-PX).abs().max().item():.2e}")
    check("self-adjoint",
          abs((PX * Y).sum().item() - (X * proj(Y)).sum().item()) < 1e-10)
    check("residual orthogonal to image",
          abs(((X - PX) * PX).sum().item()) < 1e-10)
    check("rank == dim",
          round(float(torch.linalg.matrix_rank(B @ B.T).item())) == dim)
    check("dim(so(32)) == 496", k * (k - 1) // 2 == dim)

    print("(2) random projector, float32 (the dtype that actually runs)")
    proj32, B32, oe32 = make_random_projector(k, dim, 1, "cpu", torch.float32)
    X32 = torch.randn(k, k)
    P32 = proj32(X32)
    check("orthonormality at float32", oe32 < 1e-4, f"max|BtB-I|={oe32:.2e}")
    check("idempotent at float32",
          (proj32(P32) - P32).abs().max().item() < 1e-4,
          f"{(proj32(P32)-P32).abs().max().item():.2e}")

    print("(3) plumbing")
    Z = torch.randn(k, k, dtype=torch.float64, requires_grad=True)
    proj(Z).pow(2).sum().backward()
    check("differentiable", Z.grad is not None and bool(torch.isfinite(Z.grad).all()))
    _, B2, _ = make_random_projector(k, dim, 2, "cpu", torch.float64)
    check("fresh subspace per seed", not torch.allclose(B, B2))

    class Ops:
        pass
    o = Ops()
    o.project_random = proj
    getattr(o, "project_random")(X)
    check("attaches without self-binding", True)

    print("(4) channel recovery and exact AUC decomposition")

    class StubEnv:
        geo_weight, horizon = 0.4, 20

        def __init__(self):
            self.truth, self._t = [], 0

        def _geometry_reward(self, v, M):
            return 0.5 + 0.004 * (len(self.truth) % 20)

        def step(self, a, M=None):
            terminal = (self._t == self.horizon - 1)
            t = 0.9 - 0.05 * (len(self.truth) % 4)
            if not terminal and (len(self.truth) % 3 == 0):
                t = 0.0                                   # sparsity
            g = self._geometry_reward(None, M)
            self.truth.append((t, g, terminal))
            self._t = 0 if terminal else self._t + 1
            return None, float(0.6 * t + 0.4 * g), terminal

    class StubTrainer:
        def __init__(self, env, n_steps, n_iters):
            self.env, self.n_steps, self.n_iters = env, n_steps, n_iters
            self.ep = []

        def _gather_trajectory(self):
            # mirrors main.py exactly: accumulate, append only on `done`
            cur, done_returns = 0.0, []
            for _ in range(self.n_steps):
                _, r, d = self.env.step(0)
                cur += r
                if d:
                    done_returns.append(cur)
                    cur = 0.0
            mean_ep = float(np.mean(done_returns)) if done_returns else 0.0
            return (None, None, None, None, None, None, mean_ep)

        def train(self):
            eps = []
            for _ in range(self.n_iters):
                eps.append(self._gather_trajectory()[6])
            from sppg_defense.rl.analysis import auc_trapezoid
            return {"auc": float(auc_trapezoid(eps))}

    e = StubEnv()
    w = instrument(e)
    tr = StubTrainer(e, n_steps=512, n_iters=8)   # 512 = 25 eps + 12 orphans
    instrument_trainer(tr, e, w, e.horizon, strict=True)
    res = tr.train()
    pairs = channels(res, tr)
    a_t, a_g, a_tot = channel_aucs(pairs, w, e.horizon)
    check("iterations logged", len(pairs) == 8)
    check("auc_task + auc_geo == auc_total", abs(a_t + a_g - a_tot) < 1e-9)
    check("decomposition == the reported AUC (512 vs 500 step bug)",
          abs(a_tot - res["auc"]) < 1e-9, f"|diff|={abs(a_tot-res['auc']):.2e}")
    check("per-iteration recon check ran", len(tr._r1_recon_err) == 8)
    check("per-iteration recon error ~ 0", max(tr._r1_recon_err) < 1e-9,
          f"max={max(tr._r1_recon_err):.2e}")

    print("(5) analysis wiring")
    try:
        from sppg_defense.rl.analysis import CONTRASTS
        labels = [a for a, _ in ARMS]

        def match(n):
            return [a for a in labels if a == n or a.startswith(n + "(")
                    or a.startswith(n + "_")]

        unres = [lab for lab, (L, R) in CONTRASTS.items()
                 if not all(match(x) for x in L) or not all(match(x) for x in R)]
        check("all pre-registered contrasts resolvable", not unres, str(unres))
        check(f"reference {REFERENCE!r} present", REFERENCE in labels)
    except ImportError:
        print("  SKIP  sppg_defense not importable")

    df = pd.DataFrame([{"arm": a, "seed": s, "auc": (np.inf if (a, s) == ("sym(32)", 3) else 1.0 + s * 0.01)}
                       for a, _ in ARMS for s in range(5)])
    cleaned = _clean(df, "auc")
    check("inf rows dropped and design kept paired",
          len(cleaned) == 4 * len(ARMS) and np.isfinite(cleaned["auc"]).all(),
          f"{len(cleaned)} rows, {cleaned['seed'].nunique()} seeds")

    print("\n" + ("SELFTEST PASS" if ok else "SELFTEST FAIL"))
    if not ok:
        sys.exit(1)


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--no-baseline", action="store_true")
    ap.add_argument("--analyse-only", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        selftest()
        return

    if a.analyse_only:
        if not os.path.exists(ARM_CSV):
            sys.exit(f"{ARM_CSV} not found; run without --analyse-only first.")
        arms = pd.read_csv(ARM_CSV)
    else:
        arms = collect(a.seeds, with_baseline=not a.no_baseline)

    analyse(arms)


if __name__ == "__main__":
    main()
