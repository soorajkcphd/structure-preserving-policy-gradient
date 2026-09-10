"""
R-15: Task-3 positive control -- what effect could this design have detected?

    python run_r15.py --selftest
    python run_r15.py --seeds 10          # ~35 min on an RTX 5060
    python run_r15.py --analyse-only

Requires run_sentiment_task3.py in the same directory.  No edits to it.

Why
---
Task 3 reports a null: with no engineered geometry, so(32) and baseline PPO
are indistinguishable (AUC 2.804 vs 2.802, p = 0.871, 10 seeds), and the
paper reads this as a falsification test passing.  Two problems:

  1. A null at n = 10 says nothing about absence unless the design could have
     detected a present effect.  That was never measured.
  2. Task 3 is easier than Task 1 -- sparse_prob 0.5 vs 0.7, reward_noise 0.1
     vs 0.2, horizon 10 vs 20, 8 actions vs 16 -- so an effect is not
     obviously harder to detect here, but nor is it obviously comparable.
     Without a positive control the reader cannot tell.

This script injects a known amount of geometric structure at increasing
strength and finds the smallest injection this design detects.  That converts
"we found no effect" into "we could have detected an effect of size >= X,
and did not" -- a bounded statement rather than an absence.

Design
------
The Task-3 environment is extended with a planted rotation M_env in SO(32)
acting on the first 32 state dimensions, and the reward becomes

    r = (1 - w) * r_sentiment + w * r_geo,
    r_geo = (cos(M_policy v, M_env v) + 1) / 2,   v = state[:32]

exactly as in Task 1 (main.py).  w = 0 reproduces the published Task-3
environment bit for bit; w > 0 injects structure the so(32) arm can exploit
and the baseline (which acts with M = I) cannot.

    w      0, 0.02, 0.05, 0.1, 0.2
    arms   baseline_ppo, so
    seeds  0 .. n-1, shared across cells so every contrast is paired

Note on the mechanism being tested
----------------------------------
run_sentiment_task3.py's trainer has no auxiliary-loss code path at all, so
the injected geometry is reward-only.  Given S-7 -- where the geometry reward
alone never taught the rotation -- the expected outcome is that even w = 0.2
produces little separation.  If so, that is the finding: the Task-3 design
cannot detect a reward-only geometric benefit at any injection strength
tested, and its null says nothing about whether such a benefit exists.  Either
way the section gains a quantitative bound where it currently has none.

One asymmetry
-------------
Task 3 applies its sparsity mask (sparse_prob = 0.5) to the sentiment reward
inside the original step(), before this wrapper sees it, whereas the injected
geometry term is dense.  The injected channel is therefore easier to learn
from than the sentiment channel at the same nominal weight.  That is
conservative for the conclusion we care about: if a dense geometric signal at
w = 0.2 still produces no detectable separation, a sparse one certainly would
not, and the Task-3 null is bounded by an even weaker statement than the grid
suggests.

Output   r15_cells.csv   w, arm, seed, auc, r_sent, r_geo, rho, status
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

CSV = "r15_cells.csv"
W_GRID = (0.0, 0.02, 0.05, 0.1, 0.2)
ARMS = ("baseline_ppo", "so")
MENV_SEED = 7000
K_TRANSFORM = 32
COLS = ["w", "arm", "seed", "auc", "r_sent", "r_geo", "rho", "status"]


# --------------------------------------------------------------------------- #
def draw_m_env(k: int, seed: int, device, dtype):
    """A planted rotation, built exactly as main.py builds M_env: the matrix
    exponential of a random skew matrix, hence in SO(k) by construction."""
    import torch
    from scipy.linalg import expm

    rng = np.random.RandomState(int(seed))
    S = rng.randn(k, k) * 0.5
    S = 0.5 * (S - S.T)
    M = torch.tensor(expm(S), dtype=torch.float64)
    err = (M.T @ M - torch.eye(k, dtype=torch.float64)).abs().max().item()
    det = float(torch.det(M))
    if not (err < 1e-9 and abs(det - 1.0) < 1e-9):
        raise RuntimeError(f"drawn M_env not in SO({k}): max|MtM-I|={err:.2e}, "
                           f"det={det:.6f}")
    return M.to(device=device, dtype=dtype)


def make_injected_env(base_env, M_env, w: float, k: int):
    """Wrap a SentimentSteeringEnv so its reward carries a geometry channel.

    Instrumentation is by instance attribute, never on the class: two cells
    in the same sweep must not see each other's w.  The original `step` is
    captured by closure, so the wrapper is idempotent under repeated calls
    only if applied to a fresh object -- which collect() guarantees.
    """
    import torch

    if not (0.0 <= w < 1.0):
        raise ValueError(f"w={w}: the sentiment channel is not identifiable "
                         f"at w >= 1.")
    env = base_env
    orig_step = env.step
    env.M_env_r15 = M_env          # named M_env_r15, not M_env: see note below
    env._r15_w = float(w)
    env._r15_k = int(k)
    env._r15_sent, env._r15_geo = [], []

    def step_wrapper(action, M=None):
        nxt, r_sent, done = orig_step(action, M)
        w_ = float(env._r15_w)
        if w_ == 0.0:
            # Exactly the published environment: no geometry term is even
            # computed, so the w=0 column reproduces Task 3 bit for bit.
            env._r15_sent.append(float(r_sent)); env._r15_geo.append(0.0)
            return nxt, float(r_sent), done
        v = nxt[: env._r15_k].to(torch.float32)
        Mv = (env.M_env_r15.to(v.dtype) @ v)
        Pv = (M[: env._r15_k, : env._r15_k].to(v.dtype) @ v) if M is not None else v
        cos = torch.nn.functional.cosine_similarity(
            Pv.unsqueeze(0), Mv.unsqueeze(0)).item()
        r_geo = (cos + 1.0) / 2.0
        r = (1.0 - w_) * float(r_sent) + w_ * r_geo
        env._r15_sent.append(float(r_sent)); env._r15_geo.append(float(r_geo))
        return nxt, float(r), done

    env.step = step_wrapper
    return env


# Note on the attribute name
# --------------------------
# The planted rotation is stored as `M_env_r15`, not `M_env`.  main.py's
# trainer enables its auxiliary loss on `hasattr(env, "M_env")`; Task 3's
# trainer has no such code path, but a future reader porting this file would
# switch on a supervised regression onto M_env and change what is
# being measured.  The distinct name makes that impossible by accident.


def collect(n_seeds: int, overwrite: bool) -> pd.DataFrame:
    import torch
    if os.path.exists(CSV) and not overwrite:
        sys.exit(f"{CSV} exists; pass --overwrite to replace it.")
    import run_sentiment_task3 as T3

    T3._seed_all(42)
    embedder = T3.GPT2EmbeddingProvider("gpt2-medium")
    sent_dir = T3.compute_sentiment_direction(embedder)

    rows = []
    total = len(W_GRID) * len(ARMS) * n_seeds
    done = 0
    for w in W_GRID:
        for seed in range(n_seeds):
            for arm in ARMS:
                # A fresh env per cell: the wrapper is a closure over w, and
                # reusing one env across w values would stack wrappers.
                T3._seed_all(seed)
                env = T3.SentimentSteeringEnv(
                    embedder, sentiment_direction=sent_dir, n_prompts=20,
                    n_actions=8, horizon=10, reward_noise=0.1, sparse_prob=0.5)
                M = draw_m_env(K_TRANSFORM, MENV_SEED,
                               env.prompt_embs.device, torch.float32)
                env = make_injected_env(env, M, w, K_TRANSFORM)

                cfg = T3.PPOConfig()
                state_dim = embedder.hidden_dim
                T3._seed_all(seed)
                if arm == "baseline_ppo":
                    policy = T3.BaselinePolicy(state_dim, env.n_actions)
                    trainer = T3.LieStructuredPPO(env, policy,
                                                  T3.ValueNet(state_dim), cfg,
                                                  use_lie_projection=False)
                else:
                    policy = T3.LiePolicy(state_dim, env.n_actions,
                                          k=K_TRANSFORM, algebra="so")
                    trainer = T3.LieStructuredPPO(env, policy,
                                                  T3.ValueNet(state_dim), cfg,
                                                  use_lie_projection=True,
                                                  algebra="so")
                status = "ok"
                try:
                    res = trainer.train(verbose=False)
                    auc = float(res["auc"])
                    rho = res["spectral_radii"][-1]
                    rho = float(rho) if rho is not None else np.nan
                except KeyboardInterrupt:
                    raise
                except Exception as e:                       # noqa: BLE001
                    auc, rho, status = np.nan, np.nan, f"{type(e).__name__}: {e}"[:110]
                rs = float(np.mean(env._r15_sent)) if env._r15_sent else np.nan
                rg = float(np.mean(env._r15_geo)) if env._r15_geo else np.nan
                if status == "ok" and not math.isfinite(auc):
                    status = "non-finite"
                rows.append(dict(w=w, arm=arm, seed=seed, auc=auc, r_sent=rs,
                                 r_geo=rg, rho=rho, status=status))
                pd.DataFrame(rows).reindex(columns=COLS).to_csv(CSV, index=False)
                done += 1
                msg = (f"FAILED: {status}" if status != "ok"
                       else f"AUC={auc:7.3f}  r_sent={rs:.4f}  r_geo={rg:.4f}")
                print(f"  [{done:3d}/{total}] w={w:<5g} s{seed:2d} "
                      f"{arm:12s} {msg}", flush=True)
    print(f"\nWrote {CSV} ({len(rows)} rows)")
    return pd.DataFrame(rows).reindex(columns=COLS)


# --------------------------------------------------------------------------- #
def _paired(d, w, col="auc"):
    """so - baseline, matched on seed, within one w."""
    dd = d[d.w == w]
    x = dd[dd.arm == "so"].set_index("seed")[col].astype(float)
    y = dd[dd.arm == "baseline_ppo"].set_index("seed")[col].astype(float)
    idx = x.index.intersection(y.index)
    v = (x.loc[idx] - y.loc[idx]).replace([np.inf, -np.inf], np.nan).dropna()
    return v.to_numpy()


def _stats(v):
    """mean, 95% CI, dz, two-sided p from a paired t-test."""
    from scipy import stats as st
    n = len(v)
    if n < 2 or np.allclose(v.std(ddof=1), 0.0):
        return dict(n=n, mean=float(np.mean(v)) if n else np.nan,
                    lo=np.nan, hi=np.nan, dz=np.nan, p=np.nan)
    m, s = float(v.mean()), float(v.std(ddof=1))
    t, p = st.ttest_1samp(v, 0.0)
    h = st.t.ppf(0.975, n - 1) * s / math.sqrt(n)
    return dict(n=n, mean=m, lo=m - h, hi=m + h, dz=m / s, p=float(p))


def analyse(df: pd.DataFrame) -> None:
    bad = df[df.status != "ok"]
    if len(bad):
        print(f"\n{len(bad)} failed run(s):")
        for _, r in bad.iterrows():
            print(f"    w={r.w} {r.arm} seed={r.seed}: {r.status}")
    d = df[df.status == "ok"].copy()
    ws = [w for w in W_GRID if w in set(d.w)]
    if not ws:
        print("\n  no usable rows: nothing to analyse.")
        return

    print("\n" + "=" * 92)
    print("R-15  TASK-3 POSITIVE CONTROL: injected geometry at strength w")
    print("  w = 0 is the published Task-3 environment, reproduced exactly.")
    print("=" * 92)
    print(f"  {'w':>6s} {'baseline AUC':>18s} {'so(32) AUC':>18s} "
          f"{'so - base':>12s} {'95% CI':>22s} {'p':>9s}")
    res = {}
    for w in ws:
        b = d[(d.w == w) & (d.arm == "baseline_ppo")]["auc"].astype(float).dropna()
        s = d[(d.w == w) & (d.arm == "so")]["auc"].astype(float).dropna()
        st_ = _stats(_paired(d, w))
        res[w] = st_
        ci = (f"[{st_['lo']:+.4f}, {st_['hi']:+.4f}]"
              if np.isfinite(st_["lo"]) else f"{'n/a':>20s}")
        print(f"  {w:>6g} {b.mean():9.3f}+/-{b.std(ddof=1):6.3f} "
              f"{s.mean():9.3f}+/-{s.std(ddof=1):6.3f} "
              f"{st_['mean']:+12.4f} {ci:>22s} "
              f"{st_['p']:9.4f}" if np.isfinite(st_["p"]) else
              f"  {w:>6g}  (insufficient data)")

    # ---- Holm correction across the injected cells (w = 0 is not a test) ---
    tests = [(w, res[w]) for w in ws if w > 0 and np.isfinite(res[w]["p"])]
    if tests:
        order = sorted(tests, key=lambda t: t[1]["p"])
        m = len(order)
        holm, running = {}, 0.0
        for i, (w, st_) in enumerate(order):
            running = max(running, min(1.0, (m - i) * st_["p"]))
            holm[w] = running
        print("\n  Holm-corrected over the %d injected cells "
              "(w = 0 excluded: it is the null cell, not a test):" % m)
        for w in ws:
            if w in holm:
                mark = "*" if holm[w] < 0.05 else " "
                print(f"    w={w:<5g} p_holm={holm[w]:.4f}{mark}")
    else:
        holm = {}

    # ---- the minimum detectable effect ------------------------------------
    print("\n" + "=" * 92)
    print("R-15  MINIMUM DETECTABLE INJECTION")
    print("=" * 92)
    detected = [w for w in ws if w > 0 and holm.get(w, 1.0) < 0.05
                and res[w]["mean"] > 0]
    if detected:
        w0 = min(detected)
        print(f"  -> The design detects an injected geometric benefit from "
              f"w = {w0:g} upward\n     ({res[w0]['mean']:+.4f} AUC, "
              f"95% CI [{res[w0]['lo']:+.4f}, {res[w0]['hi']:+.4f}], "
              f"p_holm={holm[w0]:.4f}).\n     The published null at w = 0 can "
              f"therefore be reported as bounded: an effect\n     of the size "
              f"produced by w >= {w0:g} would have been seen, and was not.")
    else:
        print("  -> No injected strength was detected, up to w = "
              f"{max(ws):g}.\n     The Task-3 design cannot detect a "
              "reward-only geometric benefit at any\n     strength tested, so "
              "its null is UNINFORMATIVE: it bounds nothing.  The\n     "
              "falsification claim is not supported unless the section is "
              "re-run with a\n     design that passes this control.")

    # ---- did the injection actually reach the reward? ---------------------
    print("\n" + "=" * 92)
    print("R-15  MANIPULATION CHECK: did the injection change what was "
          "rewarded?")
    print("=" * 92)
    for w in ws:
        g = d[(d.w == w) & (d.arm == "so")]["r_geo"].astype(float).dropna()
        s = d[(d.w == w) & (d.arm == "so")]["r_sent"].astype(float).dropna()
        print(f"  w={w:<5g} mean r_geo={g.mean():.4f}  mean r_sent={s.mean():.4f}")
    print("\n  r_geo must be ~0 at w=0 (never computed) and in [0,1] "
          "elsewhere.  If r_geo\n  does not move with w, the manipulation "
          "failed and nothing above is\n  interpretable.")
    g0 = d[(d.w == 0.0) & (d.arm == "so")]["r_geo"].astype(float).dropna()
    gmax = d[(d.w == max(ws)) & (d.arm == "so")]["r_geo"].astype(float).dropna()
    if len(g0) and len(gmax) and abs(gmax.mean() - g0.mean()) < 1e-6:
        print("\n  !! r_geo is identical at w=0 and w=%g: THE INJECTION DID "
              "NOT TAKE EFFECT.\n     Fix that before reading any result "
              "above." % max(ws))


# --------------------------------------------------------------------------- #
def selftest() -> None:
    import torch
    ok = True

    def check(n, c, dd=""):
        nonlocal ok
        ok &= bool(c)
        print(f"  {'PASS' if c else 'FAIL'}  {n}{'  ' + dd if dd else ''}")

    print("(1) the planted rotation is a rotation")
    M = draw_m_env(32, 7000, torch.device("cpu"), torch.float64)
    I = torch.eye(32, dtype=torch.float64)
    check("orthogonal", (M.T @ M - I).abs().max().item() < 1e-9)
    check("det = +1", abs(float(torch.det(M)) - 1) < 1e-9)
    check("reproducible", torch.equal(
        draw_m_env(32, 7000, torch.device("cpu"), torch.float64), M))

    print("(2) the reward wrapper")

    class FakeEnv:
        n_actions = 8

        def __init__(self):
            self.calls = 0

        def step(self, action, M=None):
            self.calls += 1
            v = torch.zeros(1024); v[: 32] = torch.linspace(0.1, 1.0, 32)
            return v, 0.5, self.calls % 10 == 0

    # w = 0 must reproduce the base environment exactly
    e0 = make_injected_env(FakeEnv(), M.float(), 0.0, 32)
    _, r0, _ = e0.step(0, torch.eye(1024))
    check("w=0 returns the untouched sentiment reward", abs(r0 - 0.5) < 1e-12,
          f"r={r0}")
    check("w=0 records no geometry", e0._r15_geo == [0.0])

    # w > 0 must mix, and the mix must be arithmetically right
    e1 = make_injected_env(FakeEnv(), M.float(), 0.2, 32)
    _, r1, _ = e1.step(0, torch.eye(1024))
    g1 = e1._r15_geo[-1]
    check("w>0 mixes the two channels exactly",
          abs(r1 - (0.8 * 0.5 + 0.2 * g1)) < 1e-6, f"r={r1:.6f}, r_geo={g1:.6f}")
    check("r_geo lies in [0,1]", 0.0 <= g1 <= 1.0, f"{g1:.6f}")

    # identity policy vs the true rotation: alignment must differ
    e2 = make_injected_env(FakeEnv(), M.float(), 0.5, 32)
    e2.step(0, torch.eye(1024))
    g_id = e2._r15_geo[-1]
    Mbig = torch.eye(1024); Mbig[:32, :32] = M.float()
    e3 = make_injected_env(FakeEnv(), M.float(), 0.5, 32)
    e3.step(0, Mbig)
    g_true = e3._r15_geo[-1]
    check("acting with the true rotation scores higher than the identity",
          g_true > g_id + 1e-3, f"true={g_true:.4f} vs identity={g_id:.4f}")
    check("acting with the true rotation is near-perfect alignment",
          g_true > 0.99, f"{g_true:.6f}")

    try:
        make_injected_env(FakeEnv(), M.float(), 1.0, 32); refused = False
    except ValueError:
        refused = True
    check("w >= 1 is refused (the sentiment channel is unidentifiable)", refused)

    check("the planted rotation is not exposed as `M_env`",
          not hasattr(e1, "M_env") and hasattr(e1, "M_env_r15"))

    print("(3) analysis: the two verdicts")
    rng = np.random.default_rng(0)

    def frame(effect_by_w, n=10, noise=0.06):
        rows = []
        for w, eff in effect_by_w.items():
            for s in range(n):
                se = rng.normal(0, 0.05)
                for arm in ARMS:
                    base = 2.80 + se + rng.normal(0, noise)
                    rows.append(dict(w=w, arm=arm, seed=s,
                                     auc=base + (eff if arm == "so" else 0.0),
                                     r_sent=0.55,
                                     r_geo=0.0 if w == 0 else 0.5 + 2 * w,
                                     rho=1.0, status="ok"))
        return pd.DataFrame(rows)

    def run(df):
        b = io.StringIO()
        with contextlib.redirect_stdout(b):
            analyse(df)
        return b.getvalue()

    graded = run(frame({0.0: 0.0, 0.02: 0.02, 0.05: 0.08, 0.1: 0.20, 0.2: 0.45}))
    check("a graded response -> a minimum detectable injection is reported",
          "detects an injected geometric benefit from w =" in graded)
    flat = run(frame({w: 0.0 for w in W_GRID}))
    check("no response at any w -> the null is called UNINFORMATIVE",
          "UNINFORMATIVE" in flat)
    check("...and the falsification claim is flagged as unsupported",
          "falsification claim is not supported" in flat)
    check("w=0 is excluded from the multiple-comparison correction",
          "w = 0 excluded" in graded)

    dead = frame({w: 0.0 for w in W_GRID})
    dead["r_geo"] = 0.0            # injection never took effect
    check("a failed manipulation is caught",
          "THE INJECTION DID NOT TAKE EFFECT" in run(dead))
    check("a working manipulation is not flagged",
          "DID NOT TAKE EFFECT" not in graded)

    for name, dfx in [("empty", frame({0.0: 0.0}).iloc[0:0]),
                      ("one seed", frame({w: 0.0 for w in W_GRID}, n=1)),
                      ("w=0 only", frame({0.0: 0.0}))]:
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
    analyse(df)


if __name__ == "__main__":
    main()
