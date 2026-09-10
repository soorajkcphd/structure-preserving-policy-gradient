"""
T-6  1-smoothness of the SO geometric auxiliary loss.
Verifies Lemma `lem:geo_smooth`  ->  claim TH-6.

Claim
-----
For ||v||_2 = 1, G_ref in SO(n) and theta in so(n),

        l_geo(theta; v) = - <exp(theta) v, G_ref v>

satisfies |D^2 l_geo(theta)[E1,E2]| <= ||E1||_F ||E2||_F, i.e. l_geo is
1-smooth on so(n); and c_geo * l_geo is |c_geo|-smooth.  The same constant holds
for the ambient parameterisation Theta -> l_geo(Proj_so(Theta); v), because the
orthogonal projector has operator norm one.

Both statements are checked, because Algorithm alg:spppo differentiates through
Proj_so(theta), i.e. it uses the ambient form.

Method
------
The exact gradient is available in closed form.  Writing w = G_ref v,

        l_geo(theta) = -<w v^T, exp(theta)>_F
        grad l_geo   = -Proj_so( Dexp(theta)^* [w v^T] )
                     = -Proj_so( expm_frechet(theta^T, w v^T) )

using the adjoint identity <Z, Dexp(A)[H]> = <Dexp(A^T)[Z], H>.  Hessian-vector
products are then central differences of this gradient, and lambda_max is
obtained by Lanczos on those products.

Why not random directions.  Maximising |Q(E)| over random unit E gives a
lower bound that concentrates near tr(H)/d, and at dim so(32) = 496 it
under-reports lambda_max by a factor of 10-100, so a violation could go
unseen.  Measured at n = 16, where the full
Hessian is assemblable: true lambda_max 0.274 vs random-direction estimate
0.044.  Lanczos recovers the true value to machine precision.

Two independent estimators are used and cross-checked:
  * n <= 16 : the full Hessian assembled by polarisation from second
              differences, giving lambda_max exactly;
  * all n   : Lanczos on analytic-gradient Hessian-vector products.
Agreement between them is reported, so neither is taken on trust.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.linalg import expm, expm_frechet, qr

from ..algebra import (from_coords, orthonormal_basis, proj, random_element,
                       to_coords)
from ._common import Result, lambda_max_abs
from .t09_pullback_bound import d2_directional

__all__ = ["geo_loss", "geo_grad", "random_SO", "run"]


def random_SO(n: int, rng: np.random.Generator) -> np.ndarray:
    """Haar-random element of SO(n) (QR with sign fix, then determinant fix)."""
    Q, R = qr(rng.standard_normal((n, n)))
    Q = Q * np.sign(np.diag(R))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1.0
    return Q


def geo_loss(theta: np.ndarray, v: np.ndarray, G_ref: np.ndarray,
             ambient: bool = False) -> float:
    """l_geo = -<exp(theta) v, G_ref v>; if ambient, project theta onto so(n) first."""
    th = proj("so", theta) if ambient else theta
    return float(-(expm(th) @ v) @ (G_ref @ v))


def geo_grad(theta: np.ndarray, v: np.ndarray, G_ref: np.ndarray,
             ambient: bool = False) -> np.ndarray:
    """
    Exact gradient of l_geo.  With w = G_ref v and Z = w v^T,

        l_geo(theta) = -<Z, exp(theta)>_F
        grad         = -Dexp(theta)^*[Z] = -expm_frechet(theta^T, Z)

    projected onto so(n) (algebra parameterisation) or onto so(n) then treated
    as an ambient gradient (ambient parameterisation -- the projector is
    self-adjoint, so the same expression serves, evaluated at Proj_so(theta)).
    """
    th = proj("so", theta) if ambient else theta
    Z = np.outer(G_ref @ v, v)
    G = -expm_frechet(th.T, Z, compute_expm=False)
    return proj("so", G)


def _lambda_max_lanczos(theta, v, G_ref, basis, rng, ambient=False,
                        h: float = 1e-5) -> float:
    """
    lambda_max of the Hessian in the given orthonormal basis, by Lanczos on
    Hessian-vector products formed from central differences of the exact
    gradient.  `basis` selects the parameterisation: an so(n) basis gives the
    algebra Hessian, a gl(n) basis the ambient one.
    """
    d = basis.shape[0]

    def hvp(c: np.ndarray) -> np.ndarray:
        E = from_coords(basis, c)
        gp = geo_grad(theta + h * E, v, G_ref, ambient)
        gm = geo_grad(theta - h * E, v, G_ref, ambient)
        return to_coords(basis, (gp - gm) / (2 * h))

    return lambda_max_abs(hvp, d, rng)


def _full_hessian(theta, v, G_ref, basis, ambient=False) -> np.ndarray:
    """Full Hessian by polarisation of second differences (small dim only)."""
    d = basis.shape[0]

    def Q(cvec: np.ndarray) -> float:
        E = from_coords(basis, cvec)
        return d2_directional(lambda r: geo_loss(theta + r * E, v, G_ref, ambient))

    e = np.eye(d)
    diag = np.array([Q(e[i]) for i in range(d)])
    H = np.empty((d, d))
    np.fill_diagonal(H, diag)
    for i in range(d):
        for j in range(i + 1, d):
            H[i, j] = H[j, i] = 0.5 * (Q(e[i] + e[j]) - diag[i] - diag[j])
    return H


def run(n_full=(4, 8, 16), n_lanczos=(4, 8, 16, 32, 64),
        radii=(0.0, 0.5, 2.0, 10.0, 100.0), tol: float = 1e-4,
        seed: int = 0) -> Result:
    rng = np.random.default_rng(seed)
    rows = []

    # ---- gradient sanity: the closed form must match finite differences ------
    grad_err = 0.0
    for n in (4, 8, 16):
        G_ref = random_SO(n, rng)
        v = rng.standard_normal(n); v /= np.linalg.norm(v)
        th = random_element("so", n, rng, fro=1.7)
        g = geo_grad(th, v, G_ref)
        B = orthonormal_basis("so", n)
        num = np.zeros(B.shape[0])
        for k in range(B.shape[0]):
            E = B[k]
            num[k] = (geo_loss(th + 1e-6 * E, v, G_ref)
                      - geo_loss(th - 1e-6 * E, v, G_ref)) / 2e-6
        grad_err = max(grad_err, float(np.abs(to_coords(B, g) - num).max()))

    # ---- the bound, on both parameterisations --------------------------------
    for n in sorted(set(n_full) | set(n_lanczos)):
        G_ref = random_SO(n, rng)
        v = rng.standard_normal(n); v /= np.linalg.norm(v)
        b_so = orthonormal_basis("so", n)
        b_gl = orthonormal_basis("gl", n)

        for R in radii:
            # Algebra parameterisation: theta and directions both in so(n)
            th = np.zeros((n, n)) if R == 0 else random_element("so", n, rng, fro=R)
            lam_lanczos = (_lambda_max_lanczos(th, v, G_ref, b_so, rng)
                           if n in n_lanczos else np.nan)
            lam_full = (float(np.abs(np.linalg.eigvalsh(
                _full_hessian(th, v, G_ref, b_so))).max())
                        if n in n_full else np.nan)
            rows.append(dict(param="algebra", n=n, dim=b_so.shape[0], radius=R,
                             lambda_lanczos=lam_lanczos, lambda_full=lam_full,
                             bound=1.0))

            # Ambient parameterisation: theta drawn from gl(n) and perturbed in
            # gl(n), so Proj_so is not the identity.  (If both were skew, proj
            # would be a bit-exact no-op and the "ambient" panel would
            # duplicate the algebra panel.)
            thA = np.zeros((n, n)) if R == 0 else random_element("gl", n, rng, fro=R)
            proj_is_identity = bool(np.array_equal(proj("so", thA), thA))
            lam_lanczosA = (_lambda_max_lanczos(thA, v, G_ref, b_gl, rng, ambient=True)
                            if n in n_lanczos else np.nan)
            lam_fullA = (float(np.abs(np.linalg.eigvalsh(
                _full_hessian(thA, v, G_ref, b_gl, ambient=True))).max())
                         if n in n_full and n <= 8 else np.nan)
            rows.append(dict(param="ambient", n=n, dim=b_gl.shape[0], radius=R,
                             lambda_lanczos=lam_lanczosA, lambda_full=lam_fullA,
                             bound=1.0, proj_was_identity=proj_is_identity))

    df = pd.DataFrame(rows)
    if len(df) == 0:
        raise RuntimeError("no configurations were evaluated")

    vals = pd.concat([df["lambda_lanczos"], df["lambda_full"]]).dropna()
    if len(vals) == 0:
        raise RuntimeError("no lambda_max values were computed")
    violations = int((vals > 1 + tol).sum())

    # cross-check the two independent estimators wherever both are available
    both = df.dropna(subset=["lambda_lanczos", "lambda_full"])
    agree = float((both["lambda_lanczos"] - both["lambda_full"]).abs().max()) \
        if len(both) else np.nan

    # the ambient panel must actually exercise the projector
    amb = df[df.param == "ambient"]
    # .eq(False) rather than ~(...).fillna(True): the column is object dtype
    # (algebra rows do not set it), and fillna on an object column triggers a
    # pandas downcasting FutureWarning.
    amb_nontrivial = bool(amb["proj_was_identity"].eq(False).any())

    # |c_geo|-smoothness: assemble the Hessian of c*l_geo independently and
    # compare its lambda_max with |c| times that of l_geo.  (The previous
    # version compared two finite-difference stencils, which is an identity of
    # the difference operator and tests nothing about l_geo.)
    n = 8
    B8 = orthonormal_basis("so", n)
    G8 = random_SO(n, rng)
    v8 = rng.standard_normal(n); v8 /= np.linalg.norm(v8)
    th8 = random_element("so", n, rng, fro=2.0)
    lam1 = float(np.abs(np.linalg.eigvalsh(_full_hessian(th8, v8, G8, B8))).max())
    scale_err = 0.0
    for c in (-3.0, -1.0, 0.1, 1.0, 3.0, 10.0):
        def hvp(cv, c=c):
            E = from_coords(B8, cv)
            gp = c * geo_grad(th8 + 1e-5 * E, v8, G8)
            gm = c * geo_grad(th8 - 1e-5 * E, v8, G8)
            return to_coords(B8, (gp - gm) / 2e-5)
        lam_c = lambda_max_abs(hvp, B8.shape[0], rng)
        scale_err = max(scale_err, abs(lam_c - abs(c) * lam1) / max(abs(c) * lam1, 1e-12))

    passed = (violations == 0
              and grad_err < 1e-6
              and (np.isnan(agree) or agree < 1e-3)
              and amb_nontrivial
              and scale_err < 1e-3)

    return Result(
        name="t06_geo_loss_smoothness",
        claim="TH-6 / Lemma lem:geo_smooth: l_geo is 1-smooth on so(n), also in the ambient Proj_so parameterisation",
        passed=passed,
        summary={
            "configurations tested": len(df),
            "lambda_max values computed": int(len(vals)),
            "violations of lambda_max <= 1 (must be 0)": violations,
            "max lambda_max over all configurations": float(vals.max()),
            "max lambda_max, ALGEBRA parameterisation":
                float(pd.concat([df[df.param == 'algebra']['lambda_lanczos'],
                                 df[df.param == 'algebra']['lambda_full']]).dropna().max()),
            "max lambda_max, AMBIENT Proj_so parameterisation":
                float(pd.concat([amb['lambda_lanczos'], amb['lambda_full']]).dropna().max()),
            "ambient panel genuinely exercises Proj_so (not a no-op)": amb_nontrivial,
            "closed-form gradient vs finite differences, max abs error": grad_err,
            "Lanczos vs full-Hessian cross-check, max abs difference": agree,
            "dims: full Hessian": list(n_full),
            "dims: Lanczos (dim so(32)=496, so(64)=2016, gl(64)=4096)": list(n_lanczos),
            "radii tested": list(radii),
            "|c_geo|-smoothness max rel error (c in +-{0.1,1,3,10})": scale_err,
            "tolerance on the bound (finite-difference Hessian accuracy)": tol,
        },
        table=df,
    )


if __name__ == "__main__":
    r = run()
    print(r.report())
    r.save()
