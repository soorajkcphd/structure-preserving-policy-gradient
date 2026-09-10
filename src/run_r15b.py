"""
R-15b: Task 3 with the auxiliary target enabled -- does supervision separate
       the arms where reward alone did not?

    python run_r15b.py --selftest
    python run_r15b.py --seeds 10          # ~35 min on an RTX 5060
    python run_r15b.py --analyse-only

Requires run_sentiment_task3.py and run_r15.py in the same directory.
Edits neither.

Why
---
R-15 injected a geometric reward channel into Task 3 at strengths up to
w = 0.2 and found that alignment with the planted rotation never developed:
r_geo sat at 0.445-0.454 across a tenfold change in w, and the arms never
separated.  That reproduces S-7's Task-1 finding in a second, independent
environment: the geometry reward alone does not teach the rotation.

S-7 also showed the other half of the dichotomy on Task 1 -- that a supervised
regression onto M_env does teach it.  Task 3 cannot show that half, because
run_sentiment_task3.py has no auxiliary-loss code path at all.  This script
adds one, identical in form to main.py's:

    L_geo = -c_geo * cos( exp(Proj_so(theta)) v_t ,  M_env v_t )

and re-runs the same grid.  Two outcomes, both worth reporting:

  arms separate    -> the full S-7 dichotomy replicates in a second
                      environment: supervision teaches the rotation, reward
                      does not.  This is the strongest form of the paper's
                      central negative result.
  arms still flat  -> the mechanism is environment-specific and S-7's scope is
                      narrower than Task 1 suggests.

How the auxiliary loss is added without editing the trainer
-----------------------------------------------------------
Copying run_sentiment_task3.py's training loop to insert one term would risk
the copy drifting from the original, which is exactly what makes the
comparison valid.  Instead we exploit the call order the original already has:

    self.pi_optim.zero_grad()
    dist_mb = Categorical(self.policy(obs[mb]))    <-- forward pre-hook fires
    ...
    pi_loss.backward()
    mb_gnorms.append(self._total_grad_norm(...))
    self._proj_grads()                             <-- we override this
    self.pi_optim.step()

A forward pre-hook on the policy records the minibatch it was last called
with; the overridden _proj_grads adds the auxiliary gradient for exactly that
minibatch and then defers to the parent.  The gradient therefore lands at the
same point in the update as main.py's L_geo.backward(), on the same states,
and not one line of the original trainer is modified.

    grid    w in {0.05, 0.1, 0.2}   (w = 0 has no geometry to supervise)
    c_geo   1.0, as main.py uses
    arms    baseline_ppo, so
    seeds   0 .. n-1, paired

Output   r15b_cells.csv   w, arm, seed, auc, r_sent, r_geo, align, rho, status
"""
from __future__ import annotations

import argparse
import contextlib
import io
import math
import os
import sys

import numpy as np
import pandas as pd

from run_r15 import draw_m_env, make_injected_env

CSV = "r15b_cells.csv"
W_GRID = (0.05, 0.1, 0.2)
ARMS = ("baseline_ppo", "so")
GEO_AUX_COEF = 1.0
MENV_SEED = 7000            # the same planted rotation R-15 used
K_TRANSFORM = 32
COLS = ["w", "arm", "seed", "auc", "r_sent", "r_geo", "align", "rho", "status"]


# --------------------------------------------------------------------------- #
def make_supervised_trainer(T3, aux_coef: float, k: int):
    """Return a LieStructuredPPO subclass carrying main.py's auxiliary loss.

    The subclass overrides exactly one method (_proj_grads) and adds a forward
    pre-hook.  The parent's train(), trajectory gathering, GAE, PPO update and
    diagnostics are untouched.
    """
    import torch
    import torch.nn.functional as F

    class SupervisedPPO(T3.LieStructuredPPO):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self._aux_coef = float(aux_coef)
            self._k = int(k)
            self._last_obs = None
            self._aux_fired = 0
            self._hook = self.policy.register_forward_pre_hook(
                lambda _m, inp: self._record(inp))

        def _record(self, inp):
            x = inp[0]
            self._last_obs = x if x.dim() == 2 else x.unsqueeze(0)
            return None

        def _aux_target(self):
            """M_env as the environment's planted rotation, or None."""
            return getattr(self.env, "M_env_r15", None)

        def _proj_grads(self):
            M_env = self._aux_target()
            if (self.use_lie and self._aux_coef > 0.0
                    and hasattr(self.policy, "theta")
                    and self._last_obs is not None and M_env is not None):
                v = self._last_obs[:, : self._k]
                th = self._apply_proj(self.policy.theta)
                M_pol = T3.LieAlgebraOps.matrix_exp(th)
                actual = (M_pol @ v.unsqueeze(-1)).squeeze(-1)
                target = (M_env.to(v.dtype) @ v.unsqueeze(-1)).squeeze(-1)
                cos_geo = F.cosine_similarity(actual, target, dim=-1).mean()
                (-self._aux_coef * cos_geo).backward()
                self._aux_fired += 1
            # The parent projects the (now combined) gradient onto the
            # algebra and records the projection magnitude.  Two diagnostics
            # shift relative to main.py as a result, neither affecting
            # training:
            #   * grad_norms are recorded by the parent before this method
            #     runs, so they exclude the auxiliary contribution, whereas
            #     main.py records them after its L_geo.backward();
            #   * projection_magnitudes now measure the combined gradient
            #     rather than the PPO gradient alone.
            # The optimiser sees exactly what main.py's optimiser sees.
            super()._proj_grads()

        def alignment(self) -> float:
            """cos(M_policy v, M_env v) on the states actually visited."""
            M_env = self._aux_target()
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


# --------------------------------------------------------------------------- #
def collect(n_seeds: int, overwrite: bool) -> pd.DataFrame:
    import torch
    if os.path.exists(CSV) and not overwrite:
        sys.exit(f"{CSV} exists; pass --overwrite to replace it.")
    import run_sentiment_task3 as T3

    T3._seed_all(42)
    embedder = T3.GPT2EmbeddingProvider("gpt2-medium")
    sent_dir = T3.compute_sentiment_direction(embedder)
    Sup = make_supervised_trainer(T3, GEO_AUX_COEF, K_TRANSFORM)

    rows = []
    total, done = len(W_GRID) * len(ARMS) * n_seeds, 0
    for w in W_GRID:
        for seed in range(n_seeds):
            for arm in ARMS:
                T3._seed_all(seed)
                env = T3.SentimentSteeringEnv(
                    embedder, sentiment_direction=sent_dir, n_prompts=20,
                    n_actions=8, horizon=10, reward_noise=0.1, sparse_prob=0.5)
                M = draw_m_env(K_TRANSFORM, MENV_SEED,
                               env.prompt_embs.device, torch.float32)
                env = make_injected_env(env, M, w, K_TRANSFORM)

                cfg = T3.PPOConfig()
                sd = embedder.hidden_dim
                T3._seed_all(seed)
                tr = None
                if arm == "baseline_ppo":
                    # No theta, so the auxiliary term cannot fire: this arm is
                    # identical to R-15's baseline by construction.
                    pol = T3.BaselinePolicy(sd, env.n_actions)
                    tr = Sup(env, pol, T3.ValueNet(sd), cfg,
                             use_lie_projection=False)
                else:
                    pol = T3.LiePolicy(sd, env.n_actions, k=K_TRANSFORM,
                                       algebra="so")
                    tr = Sup(env, pol, T3.ValueNet(sd), cfg,
                             use_lie_projection=True, algebra="so")
                status, auc, rho, align = "ok", np.nan, np.nan, np.nan
                try:
                    res = tr.train(verbose=False)
                    auc = float(res["auc"])
                    r_ = res["spectral_radii"][-1]
                    rho = float(r_) if r_ is not None else np.nan
                    align = tr.alignment()
                    if arm == "so" and tr._aux_fired == 0:
                        status = "aux-never-fired"
                except KeyboardInterrupt:
                    raise
                except Exception as e:                       # noqa: BLE001
                    status = f"{type(e).__name__}: {e}"[:110]
                finally:
                    if tr is not None:
                        tr.close()
                rs = float(np.mean(env._r15_sent)) if env._r15_sent else np.nan
                rg = float(np.mean(env._r15_geo)) if env._r15_geo else np.nan
                if status == "ok" and not math.isfinite(auc):
                    status = "non-finite"
                rows.append(dict(w=w, arm=arm, seed=seed, auc=auc, r_sent=rs,
                                 r_geo=rg, align=align, rho=rho, status=status))
                pd.DataFrame(rows).reindex(columns=COLS).to_csv(CSV, index=False)
                done += 1
                msg = (f"FAILED: {status}" if status != "ok"
                       else f"AUC={auc:7.3f}  r_geo={rg:.4f}  "
                            f"align={align:+.4f}")
                print(f"  [{done:3d}/{total}] w={w:<5g} s{seed:2d} "
                      f"{arm:12s} {msg}", flush=True)
    print(f"\nWrote {CSV} ({len(rows)} rows)")
    return pd.DataFrame(rows).reindex(columns=COLS)


# --------------------------------------------------------------------------- #
def _paired(d, w, col="auc"):
    dd = d[d.w == w]
    x = dd[dd.arm == "so"].set_index("seed")[col].astype(float)
    y = dd[dd.arm == "baseline_ppo"].set_index("seed")[col].astype(float)
    idx = x.index.intersection(y.index)
    return (x.loc[idx] - y.loc[idx]).replace([np.inf, -np.inf], np.nan).dropna().to_numpy()


def _stats(v):
    from scipy import stats as st
    n = len(v)
    if n < 2 or np.isclose(float(np.std(v, ddof=1)) if n > 1 else 0.0, 0.0):
        return dict(n=n, mean=float(np.mean(v)) if n else np.nan,
                    lo=np.nan, hi=np.nan, dz=np.nan, p=np.nan)
    m, s = float(v.mean()), float(v.std(ddof=1))
    t, p = st.ttest_1samp(v, 0.0)
    h = st.t.ppf(0.975, n - 1) * s / math.sqrt(n)
    return dict(n=n, mean=m, lo=m - h, hi=m + h, dz=m / s, p=float(p))


def analyse(df: pd.DataFrame, reward_only: pd.DataFrame | None = None) -> None:
    bad = df[df.status != "ok"]
    if len(bad):
        print(f"\n{len(bad)} failed run(s):")
        for _, r in bad.iterrows():
            print(f"    w={r.w} {r.arm} seed={r.seed}: {r.status}")
        if (bad.status == "aux-never-fired").any():
            print("\n  !! 'aux-never-fired' means the forward pre-hook did not "
                  "capture a minibatch,\n     so those runs had NO auxiliary "
                  "loss and are not what this script claims\n     to measure.  "
                  "Do not read them as supervised arms.")
    d = df[df.status == "ok"].copy()
    ws = [w for w in W_GRID if w in set(d.w)]
    if not ws:
        print("\n  no usable rows: nothing to analyse."); return
    if "align" not in d.columns:
        # Alignment is the quantity this script exists to measure.  Without it
        # the AUC comparison alone cannot distinguish "learned the rotation"
        # from "gained AUC some other way", so say so rather than proceeding.
        print("\n  !! no 'align' column: this CSV predates the alignment "
              "readout.  The AUC\n     comparison below cannot tell whether "
              "the policy learned the rotation.\n     Re-collect before "
              "drawing the dichotomy conclusion.")
        d["align"] = np.nan

    print("\n" + "=" * 96)
    print("R-15b  TASK 3 WITH THE AUXILIARY TARGET ENABLED")
    print("=" * 96)
    print(f"  {'w':>6s} {'baseline AUC':>18s} {'so(32) AUC':>18s} "
          f"{'so - base':>12s} {'95% CI':>22s} {'p':>9s} {'align':>9s}")
    res = {}
    for w in ws:
        b = d[(d.w == w) & (d.arm == "baseline_ppo")]["auc"].astype(float).dropna()
        s = d[(d.w == w) & (d.arm == "so")]["auc"].astype(float).dropna()
        al = d[(d.w == w) & (d.arm == "so")]["align"].astype(float).dropna()
        st_ = _stats(_paired(d, w)); res[w] = st_
        ci = (f"[{st_['lo']:+.4f}, {st_['hi']:+.4f}]"
              if np.isfinite(st_["lo"]) else f"{'n/a':>20s}")
        pstr = f"{st_['p']:9.4f}" if np.isfinite(st_["p"]) else f"{'n/a':>9s}"
        print(f"  {w:>6g} {b.mean():9.3f}+/-{b.std(ddof=1):6.3f} "
              f"{s.mean():9.3f}+/-{s.std(ddof=1):6.3f} "
              f"{st_['mean']:+12.4f} {ci:>22s} {pstr} "
              + (f"{al.mean():+9.4f}" if len(al) else f"{'n/a':>9s}"))

    # ---- the decisive column: did supervision teach the rotation? ---------
    print("\n" + "=" * 96)
    print("R-15b  ALIGNMENT WITH THE PLANTED ROTATION")
    print("  cos(M_policy v, M_env v) at the end of training, on visited "
          "states.")
    print("=" * 96)
    for w in ws:
        al = d[(d.w == w) & (d.arm == "so")]["align"].astype(float).dropna()
        if not len(al):
            continue
        print(f"  w={w:<5g} supervised so(32): align = {al.mean():+.4f} "
              f"+/- {al.std(ddof=1) if len(al) > 1 else float('nan'):.4f}")
    if reward_only is not None and len(reward_only):
        ro = reward_only[reward_only.status == "ok"]
        ro = ro[(ro.arm == "so") & (ro.w > 0)]
        if len(ro) and "r_geo" in ro.columns:
            print(f"\n  R-15 reward-only, same grid: mean r_geo = "
                  f"{ro['r_geo'].astype(float).mean():.4f} "
                  f"(a policy with no alignment scores ~0.5)")
            print("  R-15 recorded no alignment column; r_geo is its proxy and "
                  "it never moved.")

    # ---- verdict ----------------------------------------------------------
    tests = [(w, res[w]) for w in ws if np.isfinite(res[w]["p"])]
    order = sorted(tests, key=lambda t: t[1]["p"])
    holm, running = {}, 0.0
    for i, (w, st_) in enumerate(order):
        running = max(running, min(1.0, (len(order) - i) * st_["p"]))
        holm[w] = running
    if holm:
        print("\n  Holm-corrected over the %d cells:" % len(holm))
        for w in ws:
            if w in holm:
                print(f"    w={w:<5g} p_holm={holm[w]:.4f}"
                      f"{'*' if holm[w] < 0.05 else ''}")

    aligned = [w for w in ws
               if len(d[(d.w == w) & (d.arm == "so")]["align"]
                      .astype(float).dropna())
               and d[(d.w == w) & (d.arm == "so")]["align"].astype(float).mean() > 0.5]
    wins = [w for w in ws if holm.get(w, 1.0) < 0.05 and res[w]["mean"] > 0]

    print("\n" + "=" * 96)
    print("VERDICT")
    print("=" * 96)
    if aligned and wins:
        print(f"  -> The S-7 dichotomy REPLICATES IN A SECOND ENVIRONMENT.\n"
              f"     With the auxiliary target the policy aligns with M_env "
              f"(align > 0.5 at\n     w = {aligned}) and the arms separate "
              f"(w = {wins}).  R-15 showed that\n     the same reward channel "
              "without supervision produced neither.  Supervision\n     "
              "teaches the rotation; reward does not.  This is now "
              "demonstrated on two\n     independent environments and "
              "codebases.")
    elif aligned:
        print(f"  -> Supervision does teach the rotation (align > 0.5 at "
              f"w = {aligned}), but the\n     arms do not separate "
              "significantly on AUC.  The mechanism transfers; the\n     "
              "performance consequence does not, at this seed count.  Report "
              "both.")
    elif wins:
        print(f"  -> The arms separate (w = {wins}) without the policy "
              "aligning with M_env.\n     Whatever the auxiliary loss is "
              "buying here, it is not the rotation.  Do\n     not describe "
              "this as learning the environment's transformation.")
    else:
        print("  -> Neither alignment nor separation, even with supervision.\n"
              "     S-7's mechanism does NOT transfer to this environment, so "
              "its scope is\n     narrower than Task 1 alone suggests.  State "
              "that: it bounds the claim\n     rather than refuting it.")
    print("\n  NOTE: the auxiliary loss added here is main.py's, verbatim in "
          "form, and it\n  regresses on M_env -- the same supervision the "
          "corrected Section 8.1 describes.\n  This script does not defend "
          "that design; it measures what it does.")


# --------------------------------------------------------------------------- #
def selftest() -> None:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    ok = True

    def check(nm, c, dd=""):
        nonlocal ok
        ok &= bool(c)
        print(f"  {'PASS' if c else 'FAIL'}  {nm}{'  ' + dd if dd else ''}")

    print("(1) the auxiliary gradient is real and points the right way")
    k = 8
    M_env = draw_m_env(k, 3, torch.device("cpu"), torch.float64).float()
    torch.manual_seed(0)
    theta = nn.Parameter(torch.randn(k, k) * 0.3)
    v = F.normalize(torch.randn(64, k), dim=-1)

    def aux(th):
        A = 0.5 * (th - th.T)
        Mp = torch.matrix_exp(A)
        a = (Mp @ v.unsqueeze(-1)).squeeze(-1)
        t = (M_env @ v.unsqueeze(-1)).squeeze(-1)
        return -F.cosine_similarity(a, t, dim=-1).mean()

    opt = torch.optim.Adam([theta], lr=0.05)
    c0 = -aux(theta).item()
    for _ in range(400):
        opt.zero_grad(); loss = aux(theta); loss.backward(); opt.step()
    c1 = -aux(theta).item()
    check("descending the auxiliary loss raises alignment", c1 > c0 + 0.2,
          f"{c0:+.4f} -> {c1:+.4f}")
    check("it reaches high alignment (the target is learnable this way)",
          c1 > 0.9, f"{c1:.4f}")

    print("(2) the hook-and-override mechanism")

    class FakeOps:
        @staticmethod
        def matrix_exp(X):
            return torch.matrix_exp(X)

    class FakeParentPPO:
        """Mimics run_sentiment_task3.LieStructuredPPO's relevant surface."""
        def __init__(self, env, policy, value_fn, cfg,
                     use_lie_projection=False, algebra="so"):
            self.env, self.policy, self.cfg = env, policy, cfg
            self.use_lie, self.algebra = use_lie_projection, algebra
            self.parent_calls = 0

        def _apply_proj(self, X):
            return 0.5 * (X - X.T)

        def _proj_grads(self):
            self.parent_calls += 1

    class FakeT3:
        LieStructuredPPO = FakeParentPPO
        LieAlgebraOps = FakeOps

    class Pol(nn.Module):
        def __init__(self):
            super().__init__()
            self.theta = nn.Parameter(torch.randn(k, k) * 0.3)
            self.lin = nn.Linear(64, 4)

        def forward(self, x):
            return F.softmax(self.lin(x), dim=-1)

    class Env:
        pass

    env = Env(); env.M_env_r15 = M_env
    Sup = make_supervised_trainer(FakeT3, 1.0, k)
    pol = Pol()
    tr = Sup(env, pol, None, None, use_lie_projection=True)

    obs = F.normalize(torch.randn(16, 64), dim=-1)
    pol(obs)                                    # the pre-hook should fire
    check("the pre-hook captured the minibatch",
          tr._last_obs is not None and tuple(tr._last_obs.shape) == (16, 64),
          f"{None if tr._last_obs is None else tuple(tr._last_obs.shape)}")
    pol.zero_grad()
    tr._proj_grads()
    check("the auxiliary gradient reached theta",
          pol.theta.grad is not None and pol.theta.grad.abs().max() > 0)
    check("the parent's _proj_grads still ran", tr.parent_calls == 1)
    check("the auxiliary term is counted", tr._aux_fired == 1)

    # a 1-D forward (trajectory gathering) must still be usable
    pol(F.normalize(torch.randn(64), dim=-1))
    check("a single-state forward is recorded as a batch of one",
          tuple(tr._last_obs.shape) == (1, 64))

    # baseline arm: no theta -> the term must not fire
    class BasePol(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(64, 4)

        def forward(self, x):
            return F.softmax(self.lin(x), dim=-1)

    tr_b = Sup(env, BasePol(), None, None, use_lie_projection=False)
    tr_b.policy(obs); tr_b._proj_grads()
    check("the baseline arm never fires the auxiliary term",
          tr_b._aux_fired == 0 and tr_b.parent_calls == 1)

    # no planted rotation -> the term must not fire
    tr_n = Sup(Env(), Pol(), None, None, use_lie_projection=True)
    tr_n.policy(obs); tr_n._proj_grads()
    check("without M_env_r15 the auxiliary term does not fire",
          tr_n._aux_fired == 0)

    # alignment readout
    a = tr.alignment()
    check("alignment() returns a cosine", -1.0 <= a <= 1.0, f"{a:+.4f}")
    tr.close(); tr_b.close(); tr_n.close()
    n_before = len(pol._forward_pre_hooks)
    check("close() removes the hook", n_before == 0, f"{n_before} left")

    print("(3) end-to-end: the auxiliary term inside a faithful update loop")
    # A replica of run_sentiment_task3's update order (forward -> backward ->
    # grad-norm -> _proj_grads -> step).  This is the only check that proves
    # the hook/override pair does what the docstring claims when embedded in a
    # real loop, rather than when poked by hand.
    D, A = 64, 4

    class Pol2(nn.Module):
        def __init__(self):
            super().__init__()
            self.theta = nn.Parameter(torch.randn(k, k) * 0.3)
            self.f = nn.Sequential(nn.Linear(D, 32), nn.Tanh(),
                                   nn.Linear(32, A * k * k))

        def forward(self, x):
            if x.dim() == 1:
                x = x.unsqueeze(0)
            B = x.shape[0]
            return F.softmax(torch.einsum(
                "bakl,kl->ba", self.f(x).view(B, A, k, k),
                0.5 * (self.theta - self.theta.T)), dim=-1)

    class Parent2:
        def __init__(self, env, policy, value_fn, cfg,
                     use_lie_projection=False, algebra="so"):
            self.env, self.policy, self.use_lie = env, policy, use_lie_projection
            self.opt = torch.optim.Adam([policy.theta], lr=0.02) \
                if use_lie_projection else None
            self.calls = 0

        def _apply_proj(self, X):
            return 0.5 * (X - X.T)

        def _proj_grads(self):
            self.calls += 1
            if self.policy.theta.grad is not None:
                with torch.no_grad():
                    self.policy.theta.grad.copy_(
                        self._apply_proj(self.policy.theta.grad))

        def loop(self, obs, n_mb=4):
            for i in range(n_mb):
                mb = obs[i * 8:(i + 1) * 8]
                self.opt.zero_grad()
                self.policy(mb).sum().backward()
                self._proj_grads()
                self.opt.step()

    class T3b:
        LieStructuredPPO = Parent2
        LieAlgebraOps = FakeOps

    torch.manual_seed(1)
    obs2 = F.normalize(torch.randn(32, D), dim=-1)
    env2 = Env(); env2.M_env_r15 = M_env
    Sup2 = make_supervised_trainer(T3b, 1.0, k)
    p2 = Pol2(); t2 = Sup2(env2, p2, None, None, use_lie_projection=True)
    p2(obs2[:8])
    a_before = t2.alignment()
    for _ in range(60):
        t2.loop(obs2)
    a_after = t2.alignment()
    check("inside a real loop, supervision drives alignment up",
          a_after > a_before + 0.3, f"{a_before:+.4f} -> {a_after:+.4f}")
    check("the auxiliary term fires exactly once per minibatch",
          t2._aux_fired == t2.calls, f"{t2._aux_fired} vs {t2.calls} calls")

    # the same loop with no planted rotation must not move alignment
    torch.manual_seed(1)
    p3 = Pol2(); t3 = Sup2(Env(), p3, None, None, use_lie_projection=True)
    p3(obs2[:8])

    def al(p):
        with torch.no_grad():
            Mp = torch.matrix_exp(0.5 * (p.theta.data - p.theta.data.T))
            v = obs2[:, :k]
            a = (Mp @ v.unsqueeze(-1)).squeeze(-1)
            t = (M_env @ v.unsqueeze(-1)).squeeze(-1)
            return float(F.cosine_similarity(a, t, dim=-1).mean())

    b_before = al(p3)
    for _ in range(60):
        t3.loop(obs2)
    b_after = al(p3)
    # The right assertion is not "alignment is unchanged" -- an arbitrary
    # objective moves theta and so moves alignment, in either direction, from
    # whatever the initialisation happened to give.  The claim being tested is
    # that without the target the loop does not arrive at the rotation.
    check("without the target, the same loop does not arrive at the rotation",
          b_after < 0.9, f"{b_before:+.4f} -> {b_after:+.4f}")
    check("...whereas with the target it does (contrast above)",
          a_after > 0.9 > b_after, f"with={a_after:+.4f}, without={b_after:+.4f}")
    check("...and fires the auxiliary term zero times", t3._aux_fired == 0)
    t2.close(); t3.close()

    print("(4) analysis and the four verdicts")
    rng = np.random.default_rng(0)

    def frame(sep, align, n=10, noise=0.03):
        rows = []
        for w in W_GRID:
            for s in range(n):
                se = rng.normal(0, 0.02)
                for arm in ARMS:
                    rows.append(dict(
                        w=w, arm=arm, seed=s,
                        auc=2.9 + se + (sep if arm == "so" else 0.0)
                        + rng.normal(0, noise),
                        r_sent=0.28, r_geo=0.45,
                        align=(align if arm == "so" else np.nan),
                        rho=1.0, status="ok"))
        return pd.DataFrame(rows)

    def run(df, ro=None):
        b = io.StringIO()
        with contextlib.redirect_stdout(b):
            analyse(df, ro)
        return b.getvalue()

    check("aligned AND separated -> 'REPLICATES IN A SECOND ENVIRONMENT'",
          "REPLICATES IN A SECOND ENVIRONMENT" in run(frame(0.25, 0.78)))
    check("aligned but not separated -> 'mechanism transfers'",
          "mechanism transfers" in run(frame(0.0, 0.78, noise=0.30)))
    check("separated but not aligned -> 'it is not the rotation'",
          "not the rotation" in run(frame(0.25, 0.02)))
    check("neither -> 'does NOT transfer'",
          "does NOT transfer" in run(frame(0.0, 0.02, noise=0.30)))
    check("the design note is always printed",
          "measures what it does" in run(frame(0.25, 0.78)))

    ro = pd.DataFrame([dict(w=0.1, arm="so", seed=s, auc=2.9, r_sent=0.28,
                            r_geo=0.4456, rho=1.0, status="ok")
                       for s in range(10)])
    check("the reward-only comparison is shown when R-15's CSV is present",
          "R-15 reward-only" in run(frame(0.25, 0.78), ro))

    fired = frame(0.25, 0.78)
    fired.loc[(fired.arm == "so") & (fired.seed == 0), "status"] = "aux-never-fired"
    check("an unfired auxiliary loss is called out loudly",
          "NO auxiliary" in run(fired))

    for name, dfx in [("empty", frame(0.25, 0.78).iloc[0:0]),
                      ("one seed", frame(0.25, 0.78, n=1)),
                      ("no align column", frame(0.25, 0.78).drop(columns=["align"])),
                      ("all failed", frame(0.25, 0.78).assign(status="err"))]:
        try:
            run(dfx); good = True
        except Exception as exc:                             # noqa: BLE001
            good = False
            print(f"        {name}: {type(exc).__name__}: {exc}")
        check(f"degenerate input '{name}' does not crash", good)

    print("\n" + ("SELFTEST PASS" if ok else "SELFTEST FAIL"))
    if not ok:
        sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--analyse-only", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest(); return
    if a.analyse_only:
        if not os.path.exists(CSV):
            sys.exit(f"{CSV} not found; run without --analyse-only first.")
        df = pd.read_csv(CSV)
    else:
        df = collect(a.seeds, a.overwrite)
    ro = pd.read_csv("r15_cells.csv") if os.path.exists("r15_cells.csv") else None
    analyse(df, ro)


if __name__ == "__main__":
    main()
