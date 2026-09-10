"""
B-1: is the Frobenius-projection + expm scheme better than the standard
     retraction used for optimisation on SO(n)?

    python run_b1.py --selftest
    python run_b1.py --seeds 20        # ~12 min on an RTX 5060
    python run_b1.py --analyse-only

Requires _r1core.py and _armlib.py.  No edits to main.py.

Why
---
The surviving Task-1 claim is "constraining to the compact subalgebra optimises
better".  Every comparator so far has been another of the paper's own
parameterisations, which invites the obvious question: better than what?  The
standard way to optimise on SO(n) is a Cayley retraction, and the paper already
cites the literature that uses it (Absil et al.; Kiani et al.'s projUNN is a
low-rank Cayley).  If the matrix exponential loses to a Cayley retraction, the
claim narrows from "compact structure helps" to "this particular map helps".

ARMS (all at w = 0.4, paired by seed)
  baseline_ppo   no transformation
  so_expm        M = exp(Proj_so(theta))                 <- as released
  so_cayley      M = (I - A/2)^-1 (I + A/2),  A = Proj_so(theta)

Both maps send so(n) into SO(n) exactly, take the same theta, cost O(n^3), and
are differentiable, so the auxiliary loss trains through either.  They differ
only in the retraction.  Cayley cannot reach rotations with an eigenvalue of
-1, a measure-zero set, so nothing reachable is lost in practice.

How the map is swapped
----------------------
main.py forms M in three places, all via the class-level
LieAlgebraOps.matrix_exp: the per-iteration M (line ~882), the spectral-radius
diagnostic (~979), and the auxiliary loss (~1030).  Swapping requires patching
that class attribute, which is global -- so it is done inside a context manager
that restores the original in a finally block and then asserts the restoration.
The selftest checks that an exception mid-run still restores it.

Output   b1_cells.csv   arm, seed, auc, r_task, r_geo, rho, orth_err, status
"""
from __future__ import annotations

import argparse
import contextlib
import os
import sys

import numpy as np
import pandas as pd

from _armlib import (BASELINE, Sink, check_meta, clean, equivalent,
                     guard_outputs, paired, report, run_one, write_meta)

CSV = "b1_cells.csv"
A_EXPM, A_CAYLEY = "so_expm", "so_cayley"
ARMS = (BASELINE, A_EXPM, A_CAYLEY)
W, HORIZON = 0.4, 20
COLS = ["arm", "seed", "auc", "r_task", "r_geo", "rho", "orth_err",
        "recon_err", "status"]


def cayley(A):
    """(I - A/2)^-1 (I + A/2).  For skew A this is exactly in SO(n).

    Differentiable: torch.linalg.solve has a backward.  Kept in the input dtype
    so it is a drop-in for torch.matrix_exp.
    """
    import torch

    n = A.shape[-1]
    I = torch.eye(n, device=A.device, dtype=A.dtype)
    return torch.linalg.solve(I - 0.5 * A, I + 0.5 * A)


@contextlib.contextmanager
def matrix_map(fn):
    """Temporarily replace LieAlgebraOps.matrix_exp, and prove it is restored.

    This patches a class attribute, which is global state.  The finally block
    restores it and the assert makes a silent failure impossible -- if this
    ever leaked, every later arm would be measured with the wrong map.
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
    """max |M^T M - I| for the arm's final M, whichever map produced it."""
    import torch
    from main import LieAlgebraOps

    if not hasattr(policy, "theta"):
        return float("nan")
    with torch.no_grad():
        M = LieAlgebraOps.matrix_exp(policy._proj(policy.theta.data)).double()
        I = torch.eye(M.shape[-1], dtype=M.dtype, device=M.device)
        return float((M.T @ M - I).abs().max().item())


def collect(n_seeds: int, overwrite: bool) -> pd.DataFrame:
    guard_outputs([CSV], overwrite)
    from main import GPT2EmbeddingProvider, MultiStepTextAlignmentEnv
    from _r1core import instrument

    emb = GPT2EmbeddingProvider(model_name="gpt2-medium", max_length=64)
    env = MultiStepTextAlignmentEnv(emb, n_prompts=16, n_actions=16,
                                    horizon=HORIZON, reward_noise=0.2,
                                    geo_weight=W, sparse_prob=0.7)
    instrument(env)
    sd, na, k = emb.hidden_dim, env.n_actions, int(env.k_transform)

    sink, rows = Sink(CSV, COLS), []
    write_meta(CSV, dict(arms=list(ARMS), w=W, horizon=HORIZON,
                         n_seeds=n_seeds))
    total = len(ARMS) * n_seeds
    done = 0
    for seed in range(n_seeds):
        for arm in ARMS:
            inner = BASELINE if arm == BASELINE else "so"
            ctx = matrix_map(cayley) if arm == A_CAYLEY else contextlib.nullcontext()
            with ctx:
                rec, _, policy = run_one(env, inner, seed, sd, na, k, W, HORIZON)
                rec["orth_err"] = (orthogonality_of_M(policy)
                                   if policy is not None else np.nan)
            rec["arm"] = arm
            sink.add([{c: rec.get(c, np.nan) for c in COLS}])
            rows.append(rec); done += 1
            msg = (f"FAILED: {rec['status']}" if rec["status"] != "ok"
                   else f"AUC={rec['auc']:7.3f}  rho={rec['rho']:.6g}  "
                        f"|MtM-I|={rec['orth_err']:.2e}")
            print(f"  [{done:3d}/{total}] seed {seed:2d} {arm:12s} {msg}",
                  flush=True)
    df = pd.DataFrame(rows).reindex(columns=COLS)
    print(f"\nWrote {CSV} ({sink.n} rows)")
    return df


def analyse(df: pd.DataFrame) -> None:
    e = pd.to_numeric(df.get("recon_err"), errors="coerce") if "recon_err" in df else None
    if e is not None and np.isfinite(e).any():
        print(f"\nchannel-reconstruction error: max = "
              f"{e[np.isfinite(e)].max():.3e}")
    bad = df[df.status != "ok"]
    if len(bad):
        print(f"\n{len(bad)} failed run(s):")
        for _, r in bad.iterrows():
            print(f"    {r.arm} seed={r.seed}: {r.status}")

    d = clean(df, "auc", "seed", ARMS)
    print("\n" + "=" * 96)
    print(f"B-1  PER-ARM  (n = {d['seed'].nunique()} paired seeds)")
    print("=" * 96)
    print(f"  {'arm':>12s} {'n':>3s} {'AUC':>18s} {'r_task':>16s} "
          f"{'r_geo':>16s} {'rho':>10s} {'max|MtM-I|':>12s}")
    for a in ARMS:
        v = d[d.arm == a]
        if not len(v):
            print(f"  {a:>12s}   (none)"); continue

        def ms(c):
            x = v[c].astype(float).dropna()
            return (f"{x.mean():9.3f}+/-{x.std(ddof=1):6.3f}"
                    if len(x) > 1 else f"{'n/a':>16s}")

        rho = v["rho"].astype(float).dropna()
        oe = v["orth_err"].astype(float).dropna()
        print(f"  {a:>12s} {len(v):>3d} {ms('auc'):>18s} {ms('r_task'):>16s} "
              f"{ms('r_geo'):>16s} "
              f"{(f'{rho.mean():10.6g}' if len(rho) else f'{chr(45):>10s}')} "
              f"{(f'{oe.mean():12.2e}' if len(oe) else f'{chr(45):>12s}')}")
    print("\n  Both maps should show rho = 1 and |MtM-I| at machine precision; "
          "if either\n  does not, the retraction is not producing a rotation "
          "and nothing else\n  in that row is interpretable.")

    rows = [
        dict(label=f"{A_EXPM} - {BASELINE}", d=paired(d, A_EXPM, BASELINE, "auc", "seed")[0]),
        dict(label=f"{A_CAYLEY} - {BASELINE}", d=paired(d, A_CAYLEY, BASELINE, "auc", "seed")[0]),
        dict(label=f"{A_EXPM} - {A_CAYLEY}", d=paired(d, A_EXPM, A_CAYLEY, "auc", "seed")[0]),
    ]
    out = report(rows, "B-1  TOTAL AUC", "  The third row is the one that "
                 "matters: is the exponential map itself\n  doing work, or "
                 "would any retraction onto SO(n) do?")

    print("\n" + "=" * 96)
    print("VERDICT")
    print("=" * 96)
    cmp_ = next((o for o in out if o["label"] == f"{A_EXPM} - {A_CAYLEY}"), None)
    if cmp_ is None or "diff" not in cmp_:
        print("  Not enough paired seeds to decide."); return
    if not cmp_["sig"]:
        # "not significant" does not establish equivalence.  To claim the two
        # maps are interchangeable the whole CI must lie inside a
        # pre-registered margin; otherwise the correct answer is "under-powered".
        ref = next((o for o in out if o["label"] == f"{A_EXPM} - {BASELINE}"), None)
        margin = 0.25 * abs(ref["diff"]) if ref and "diff" in ref else float("nan")
        if ref and "diff" in ref and equivalent(cmp_, margin):
            print(f"  -> expm and Cayley are EQUIVALENT within +/-{margin:.4f} "
                  f"AUC (25% of the\n     expm-vs-baseline effect): the whole "
                  f"95% CI [{cmp_['ci'][0]:+.4f}, {cmp_['ci'][1]:+.4f}] lies "
                  f"inside the margin.\n     The result is a property of "
                  "CONSTRAINING TO SO(32), not of the matrix\n     exponential "
                  "specifically -- the stronger and more general claim.")
        else:
            from sppg_defense.stats.tests import n_for_power_paired
            try:
                need = n_for_power_paired(abs(cmp_["dz"]), 0.8)
            except Exception:                                # noqa: BLE001
                need = None
            print(f"  -> UNDER-POWERED, not equivalent.  The difference "
                  f"({cmp_['diff']:+.4f}, p_holm={cmp_['p_holm']:.2e}) is not "
                  f"significant, but the\n     95% CI "
                  f"[{cmp_['ci'][0]:+.4f}, {cmp_['ci'][1]:+.4f}] is not inside "
                  f"the +/-{margin:.4f} equivalence margin either.\n     You "
                  "cannot claim the two maps are interchangeable on this "
                  "evidence."
                  + (f"  About {need} seeds would be\n     needed for 80% power "
                     f"at the observed effect." if need else ""))
    elif cmp_["diff"] > 0:
        print(f"  -> The matrix exponential significantly beats a Cayley "
              f"retraction\n     ({cmp_['diff']:+.4f}, p_holm={cmp_['p_holm']:.2e}).  "
              "The choice of map matters.  Both maps\n     land in the same "
              "group, so the difference needs an explanation.")
    else:
        print(f"  -> A Cayley retraction significantly beats the matrix "
              f"exponential\n     ({cmp_['diff']:+.4f}, p_holm={cmp_['p_holm']:.2e}).  "
              "The paper's specific scheme is\n     not the best way to realise "
              "the constraint, and the contribution\n     narrows accordingly.")


def selftest() -> None:
    import torch
    ok = True

    def check(n, c, dd=""):
        nonlocal ok
        ok &= bool(c)
        print(f"  {'PASS' if c else 'FAIL'}  {n}{'  ' + dd if dd else ''}")

    print("(1) the Cayley map really lands in SO(32)")
    k = 32
    g = torch.Generator().manual_seed(3)
    X = torch.randn(k, k, generator=g, dtype=torch.float64)
    A = 0.5 * (X - X.T)                              # skew
    M = cayley(A)
    I = torch.eye(k, dtype=torch.float64)
    check("orthogonal", (M.T @ M - I).abs().max().item() < 1e-10,
          f"max|MtM-I|={(M.T@M-I).abs().max().item():.2e}")
    check("det = +1", abs(float(torch.det(M)) - 1.0) < 1e-9)
    ev = torch.linalg.eigvals(M).abs()
    check("spectral radius = 1", abs(ev.max().item() - 1.0) < 1e-10)
    check("A = 0 -> identity", torch.allclose(cayley(torch.zeros(k, k, dtype=torch.float64)), I))
    check("differs from expm for the same A (it is a different retraction)",
          not torch.allclose(M, torch.matrix_exp(A), atol=1e-3))
    check("agrees with expm to first order for small A",
          torch.allclose(cayley(1e-4 * A), torch.matrix_exp(1e-4 * A), atol=1e-8))
    Z = torch.randn(k, k, generator=g, dtype=torch.float64, requires_grad=True)
    cayley(0.5 * (Z - Z.T)).pow(2).sum().backward()
    check("differentiable (the auxiliary loss trains through it)",
          Z.grad is not None and bool(torch.isfinite(Z.grad).all()))
    Mf = cayley(A.float())
    check("float32 stays orthogonal to ~1e-5",
          (Mf.T @ Mf - torch.eye(k)).abs().max().item() < 1e-4,
          f"{(Mf.T@Mf-torch.eye(k)).abs().max().item():.2e}")

    print("(2) the global patch is always restored")
    try:
        from main import LieAlgebraOps
    except Exception as exc:                                 # noqa: BLE001
        print(f"  SKIP  main.py not importable here: {exc}")
    else:
        orig = LieAlgebraOps.__dict__["matrix_exp"]
        with matrix_map(cayley):
            patched = LieAlgebraOps.matrix_exp(A)
            check("inside the context, the map is Cayley",
                  torch.allclose(patched, M))
        check("after the context, the map is restored",
              LieAlgebraOps.__dict__["matrix_exp"] is orig)
        check("and behaves as expm again",
              torch.allclose(LieAlgebraOps.matrix_exp(A), torch.matrix_exp(A)))
        try:
            with matrix_map(cayley):
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        check("restored even when the body raises",
              LieAlgebraOps.__dict__["matrix_exp"] is orig)
        check("still expm after an exception",
              torch.allclose(LieAlgebraOps.matrix_exp(A), torch.matrix_exp(A)))

    print("(3) analysis and verdict")
    rng = np.random.default_rng(0)

    def frame(expm_auc, cay_auc, n=16, noise=0.15):
        rows = []
        for s in range(n):
            se = rng.normal(0, .10)
            for a, base in ((BASELINE, 6.50), (A_EXPM, expm_auc), (A_CAYLEY, cay_auc)):
                rows.append(dict(arm=a, seed=s, auc=base + se + rng.normal(0, noise),
                                 r_task=.16, r_geo=.86,
                                 rho=np.nan if a == BASELINE else 1.0,
                                 orth_err=np.nan if a == BASELINE else 1e-7,
                                 recon_err=1e-15, status="ok"))
        return pd.DataFrame(rows)

    import io

    def run(df):
        b = io.StringIO()
        with contextlib.redirect_stdout(b):
            analyse(df)
        return b.getvalue()

    check("equivalent maps -> 'property of constraining to SO(32)'",
          "property of CONSTRAINING TO SO(32)" in run(frame(8.73, 8.72)))
    check("expm clearly better -> 'choice of map matters'",
          "choice of map matters" in run(frame(8.73, 7.60)))
    check("Cayley clearly better -> 'contribution narrows'",
          "narrows accordingly" in run(frame(7.60, 8.73)))
    # a small true gap at high noise must not be called equivalent
    check("an under-powered null is reported as under-powered, not equivalent",
          "UNDER-POWERED, not equivalent" in run(frame(8.73, 8.40, n=8, noise=1.2)))
    check("a tight null is called equivalent",
          "EQUIVALENT within" in run(frame(8.73, 8.72, n=24, noise=0.06)))
    out = run(frame(8.73, 8.72))
    check("orthogonality of the realised M is reported", "max|MtM-I|" in out)

    print("\n" + ("SELFTEST PASS" if ok else "SELFTEST FAIL"))
    if not ok:
        sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--analyse-only", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest(); return
    if a.analyse_only:
        if not os.path.exists(CSV):
            sys.exit(f"{CSV} not found; run without --analyse-only first.")
        check_meta(CSV, dict(arms=list(ARMS), w=W, horizon=HORIZON))
        df = pd.read_csv(CSV)
    else:
        df = collect(a.seeds, a.overwrite)
    analyse(df)


if __name__ == "__main__":
    main()
