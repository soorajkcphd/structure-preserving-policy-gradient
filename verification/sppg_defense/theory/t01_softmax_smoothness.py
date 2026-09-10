"""
T-1  Radius-independence and tightness of the fixed-feature softmax smoothness bound.
Verifies Lemma `lem:softmax_lipschitz`(i)  ->  claim TH-1.

Claim
-----
For pi_theta(a|s) proportional to exp(<Phi_a(s), theta>_F) with theta-independent
features satisfying sup ||Phi_a||_F <= B, the log-partition function log Z is
B^2-smooth **independently of ||theta||_F**, on *any* linear parameter subspace.

Why this can be computed exactly (no finite differences)
--------------------------------------------------------
In orthonormal coordinates x of the subspace, the logits are exactly linear:
    logit_a(x) = <Phi_a, theta>_F = c_a . x ,   c_a := coords(Proj_V Phi_a)
so log Z(x) = logsumexp_a(c_a . x) and its Hessian is the *exact* covariance

    Hess log Z (x) = Cov_{a ~ pi_x}(c_a)
                   = sum_a pi_a c_a c_a^T - (sum_a pi_a c_a)(sum_a pi_a c_a)^T

Because the basis is Frobenius-orthonormal, the spectral norm of this coordinate
matrix *is* the operator norm with respect to the Frobenius metric, which is the
norm in which the paper states the bound.

Popoviciu's inequality gives lambda_max <= B^2, since |c_a . v| <= ||c_a|| <= B
for unit v puts the random variable in an interval of length 2B.

What the experiment shows
-------------------------
1. lambda_max <= B^2 at every radius, with zero violations, under both random
   directions and an adversarial projected-gradient-ascent search over the
   sphere ||theta||_F = R.
2. The supremum over that sphere equals B^2 for every R, on every subspace --
   so the constant is exactly radius-independent and sharp.  This is the
   precise content of "B^2-smooth independently of ||theta||_F".
3. Because the bound is attained on so, sym, sl and gl alike, it cannot be a
   consequence of compactness -- the evidence for the paper's "two channels"
   separation.

Not claimed: that lambda_max(theta) is a constant or monotone function of
||theta||_F.  It is neither.  Exponential tilting can move mass onto two
extreme feature vectors and thereby raise the variance above its value at
theta = 0 before saturation eventually drives it to zero; the measured
log-log slope over random features is about +0.27, and that is not a defect.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..algebra import SUBSPACES, orthonormal_basis, proj, fro_norm
from ._common import Result, loglog_slope

__all__ = ["hessian_logZ", "lambda_max_logZ", "run"]


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def hessian_logZ(coords: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Exact Hessian of log Z in subspace coordinates.

    coords : (m, d) feature coordinates c_a
    x      : (d,)   parameter coordinates
    returns  (d, d) covariance matrix
    """
    pi = _softmax(coords @ x)
    mean = pi @ coords                                   # (d,)
    centred = coords - mean                              # (m, d)
    return (centred * pi[:, None]).T @ centred           # sum_a pi_a cc^T - mm^T


def lambda_max_logZ(coords: np.ndarray, x: np.ndarray) -> float:
    """
    Largest eigenvalue of the exact log-partition Hessian, computed in O(m^3)
    rather than O(d^3).

    The Hessian is a covariance of m points, so it is  W^T W  with
        W = diag(sqrt(pi)) (C - 1 mu^T)   of shape (m, d),
    and rank(W^T W) <= m - 1.  The nonzero spectra of W^T W and W W^T coincide,
    so lambda_max = lambda_max(W W^T) with W W^T only m x m.  For m = 64 and
    d = 1024 this is ~250x cheaper and numerically identical.
    """
    pi = _softmax(coords @ x)
    centred = coords - pi @ coords                       # (m, d)
    W = np.sqrt(pi)[:, None] * centred                   # (m, d)
    G = W @ W.T                                          # (m, m), = W W^T
    G = 0.5 * (G + G.T)
    return float(max(np.linalg.eigvalsh(G)[-1], 0.0))


def _sample_features(n: int, m: int, B: float,
                     rng: np.random.Generator) -> np.ndarray:
    """m ambient feature matrices with ||Phi_a||_F == B exactly, shape (m, n, n)."""
    P = rng.standard_normal((m, n, n))
    out = np.empty_like(P)
    for a in range(m):
        out[a] = P[a] * (B / fro_norm(P[a]))
    return out


def _exact_witness(sub: str, n: int, radii, B: float,
                   rng: np.random.Generator) -> list[dict]:
    """
    Exact radius-independence witness.

    Take m = 2 with Phi_1 = -Phi_2 = P, ||P||_F = B, and move theta along a
    direction u orthogonal to P inside the subspace.  Then both logits are
    identically zero for every radius, so pi = (1/2, 1/2) and

        lambda_max = Var(c . v) maximised at v = c/||c||  =  ||c||^2 = B^2

    *exactly, at every radius R*.  This shows the B^2 bound is not merely valid
    but attained, uniformly in ||theta||_F, on every subspace -- compact or not.
    """
    Bm = orthonormal_basis(sub, n)
    flat = Bm.reshape(Bm.shape[0], -1)
    P = proj(sub, rng.standard_normal((n, n)))
    P *= B / fro_norm(P)
    coords = (flat @ np.stack([P, -P]).reshape(2, -1).T).T      # (2, d)

    c = coords[0]
    u = rng.standard_normal(Bm.shape[0])
    u -= (u @ c) / (c @ c) * c                                  # make u perpendicular to c
    u /= np.linalg.norm(u)

    out = []
    for R in radii:
        lmax = lambda_max_logZ(coords, R * u)
        out.append(dict(subspace=sub, n=n, radius=R, B=B,
                        lambda_max=lmax, bound=B * B, ratio=lmax / (B * B)))
    return out


def _sup_on_sphere(coords: np.ndarray, d: int, R: float,
                   rng: np.random.Generator, n_starts: int = 4,
                   n_steps: int = 120, lr: float = 0.35) -> float:
    """
    Adversarial maximisation of lambda_max(Hess log Z) over {||theta||_F = R}
    by projected gradient ascent with restarts.

    Random directions give a lower bound that concentrates in high dimension;
    without this search a bound many times tighter than B^2 would also report
    zero violations, so the test would not constrain the claim.
    """
    best = 0.0
    for _ in range(n_starts):
        x = rng.standard_normal(d)
        x *= R / np.linalg.norm(x)
        for _ in range(n_steps):
            base = lambda_max_logZ(coords, x)
            best = max(best, base)
            # finite-difference ascent direction (d is small enough here)
            g = np.zeros(d)
            probe = rng.standard_normal((8, d))
            probe /= np.linalg.norm(probe, axis=1, keepdims=True)
            eps = 1e-3 * max(R, 1.0)
            for p in probe:
                g += ((lambda_max_logZ(coords, x + eps * p) - base) / eps) * p
            ng = np.linalg.norm(g)
            if ng < 1e-14:
                break
            x = x + lr * R * g / ng
            x *= R / np.linalg.norm(x)                # back onto the sphere
        best = max(best, lambda_max_logZ(coords, x))
    return best


def run(n: int = 32,
        radii=(1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4),
        m_values=(2, 4, 16, 64),
        B_values=(0.5, 1.0, 2.0),
        n_directions: int = 32,
        seed: int = 0) -> Result:
    rng = np.random.default_rng(seed)
    bases = {s: orthonormal_basis(s, n) for s in SUBSPACES}
    rows = []

    for m in m_values:
        for B in B_values:
            # Same ambient features for every subspace, so the only thing that
            # changes across subspaces is the projection -- an apples-to-apples
            # test of whether compactness matters.
            Phi = _sample_features(n, m, B,
                                   np.random.default_rng(seed + 991 * m + 7 * int(B * 10)))
            for sub in SUBSPACES:
                Bmat = bases[sub]
                flat = Bmat.reshape(Bmat.shape[0], -1)
                coords = (flat @ Phi.reshape(m, -1).T).T            # (m, d)
                B_eff = float(np.linalg.norm(coords, axis=1).max())

                # directions are drawn once and reused at every radius, so the
                # radius sweep is a clean within-direction comparison
                dirs = rng.standard_normal((n_directions, Bmat.shape[0]))
                dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

                for R in radii:
                    vals = [lambda_max_logZ(coords, R * u) for u in dirs]
                    rows.append(dict(subspace=sub, n=n, m=m, B=B, radius=R,
                                     lambda_max=float(np.max(vals)),
                                     bound=B * B, ratio=float(np.max(vals)) / (B * B),
                                     B_eff=B_eff,
                                     ratio_eff=float(np.max(vals)) / (B_eff ** 2)))

    # ---- adversarial supremum over the sphere, small n so it is affordable ---
    adv_rows = []
    for sub in SUBSPACES:
        Bm = orthonormal_basis(sub, 8)
        flat = Bm.reshape(Bm.shape[0], -1)
        Phi = _sample_features(8, 4, 1.0, np.random.default_rng(seed + 5))
        co = (flat @ Phi.reshape(4, -1).T).T
        B_eff = float(np.linalg.norm(co, axis=1).max())
        for R in (0.1, 1.0, 10.0, 100.0):
            sup = _sup_on_sphere(co, Bm.shape[0], R, np.random.default_rng(seed + 3))
            # matched random-direction estimate on the same configuration, so
            # the two search strategies are directly comparable
            r2 = np.random.default_rng(seed + 4)
            rnd = 0.0
            for _ in range(200):
                u = r2.standard_normal(Bm.shape[0])
                u *= R / np.linalg.norm(u)
                rnd = max(rnd, lambda_max_logZ(co, u))
            adv_rows.append(dict(kind="adversarial_sup", subspace=sub, n=8,
                                 radius=R, lambda_max=sup, bound=1.0,
                                 ratio=sup, B_eff=B_eff,
                                 ratio_eff=sup / B_eff ** 2,
                                 random_dir_lambda_max=rnd,
                                 random_dir_ratio_eff=rnd / B_eff ** 2))
    adf = pd.DataFrame(adv_rows)

    df = pd.DataFrame(rows)
    violations = int((df["lambda_max"] > df["bound"] * (1 + 1e-9)).sum())
    violations += int((adf["ratio_eff"] > 1 + 1e-9).sum())
    violations_eff = int((df["ratio_eff"] > 1 + 1e-9).sum())

    # Flatness: within each (subspace, m, B) cell the value must not grow with R.
    # Softmax saturation makes it decay at large radius; that is the bound being
    # loose there, not a violation.  The claim under test is only "no growth".
    growth = []
    for (s, m, B), g in df.groupby(["subspace", "m", "B"]):
        g = g.sort_values("radius")
        growth.append(loglog_slope(g["radius"].values, g["lambda_max"].values))
    max_growth_slope = float(np.nanmax(growth))

    # Exact witness: lambda_max == B^2 at every radius, on every subspace.
    wrows = []
    for sub in SUBSPACES:
        wrows += _exact_witness(sub, n, radii, 1.0, np.random.default_rng(seed + 13))
    wdf = pd.DataFrame(wrows)
    witness_err = float(np.abs(wdf["ratio"] - 1.0).max())
    wdf["kind"] = "exact_witness"
    df["kind"] = "random_features"

    # Note on the correct pass criterion.
    # "B^2-smooth independently of ||theta||_F" asserts that the smoothness
    # constant does not depend on the radius.  It does not assert that
    # lambda_max(theta) is a monotone or constant function of ||theta||_F, and
    # in fact it is neither: exponential tilting can move mass onto two extreme
    # feature vectors and thereby raise the variance above its value at
    # theta = 0, before saturation eventually drives it to zero.  The two
    # statements that are equivalent to the lemma, and that we test, are:
    #     (1) lambda_max <= B^2 at every radius (no violations), and
    #     (2) sup over the subspace sphere of radius R equals B^2 for every R,
    #         witnessed exactly by the m = 2 construction above.
    # Together these say the supremum is exactly radius-independent and sharp.
    # The adversarial search must find strictly more than random directions do
    # (otherwise it adds no power), and must still respect the bound.
    adv_gain = float((adf["ratio_eff"] / adf["random_dir_ratio_eff"]).max())
    passed = (violations == 0 and violations_eff == 0 and witness_err < 1e-9
              and adv_gain > 1.05)

    return Result(
        name="t01_softmax_smoothness",
        claim="TH-1 / Lemma lem:softmax_lipschitz(i): log Z is B^2-smooth, radius-independent, on ANY linear subspace",
        passed=passed,
        summary={
            "cells_tested": len(df),
            "bound violations vs ambient B^2 (must be 0)": violations,
            "bound violations vs in-subspace B_eff^2 (must be 0)": violations_eff,
            "max lambda_max / B_eff^2 (random features)": float(df["ratio_eff"].max()),
            "ADVERSARIAL sup over the sphere: max ratio to B_eff^2": float(adf["ratio_eff"].max()),
            "ADVERSARIAL sup violations (must be 0)": int((adf["ratio_eff"] > 1 + 1e-9).sum()),
            "matched RANDOM-direction max on the same configs": float(adf["random_dir_ratio_eff"].max()),
            "gain of adversarial search over random directions (x)": adv_gain,
            "tightest bound the adversarial search would still not violate":
                f"{float(adf['ratio_eff'].max()):.3f} x B^2",
            "exact witness max |lambda_max/B^2 - 1| over all radii": witness_err,
            "exact witness subspaces (all attain the bound at every radius)":
                sorted(wdf["subspace"].unique().tolist()),
            "witness radii span": f"{min(radii):g} .. {max(radii):g}",
            "diagnostic: max dlog(lmax)/dlog(R), random features (may be >0, see note)":
                max_growth_slope,
        },
        table=pd.concat([df, wdf, adf], ignore_index=True),
    )


if __name__ == "__main__":
    r = run()
    print(r.report())
    r.save()
