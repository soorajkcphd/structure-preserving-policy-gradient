"""
Structure selection with exact baselines.

Covers:
  T-13 / S-5  the SO fit has a closed-form global optimum (orthogonal Procrustes),
              so the paper's 300-step Adam fit is an upper bound of unknown
              looseness and the cross-algebra comparison in Table tab:structure
              is biased by an unknown, one-sided amount.
  S-1         the identity baseline.  Held-out losses of order 1e-7 on
              unit-norm embeddings are consistent with exp(X_hat) ~ I, i.e. with
              the fit describing noise around the identity map.  Reporting
              ||X_hat||_F, the rotation angles of exp(X_hat), and the residual of
              G = I settles that question.
  S-4         the exhaustive permutation test the paper's own appendix asks for
              (all m! pairings when m is small), rather than 128 Monte Carlo
              draws with replacement from only 120 distinct permutations.

The central fact
----------------
    min_{G in SO(k)}  sum_j || G V_j - T_j ||_F^2

is the orthogonal Procrustes problem.  Since G^T G = I leaves sum_j ||V_j||_F^2
invariant, minimising is equivalent to maximising  tr(G^T M)  with

    M = sum_j T_j V_j^T ,     M = U S W^T  (SVD)
    G* = U diag(1, ..., 1, det(U W^T)) W^T

The det correction keeps G* in SO(k) rather than O(k).  Every G in SO(k) has a
real skew-symmetric logarithm, so X* = logm(G*) lies in so(k) and the generator
parameterisation can represent the global optimum exactly.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd
from scipy.linalg import logm, expm

from ..algebra import fro_norm, proj

__all__ = [
    "procrustes_so", "linear_baseline", "residual", "rotation_angles",
    "orthogonality_error",
    "fit_generator_adam", "compare_fits", "exhaustive_permutation_test",
]


# --------------------------------------------------------------------------- #
# exact SO fit
# --------------------------------------------------------------------------- #
def orthogonality_error(G: np.ndarray) -> float:
    """||G^T G - I||_F, the quantity that certifies G is a rotation."""
    return fro_norm(G.T @ G - np.eye(G.shape[0]))


def procrustes_so(V: np.ndarray, T: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """
    Global minimiser of sum_j ||G V_j - T_j||_F^2 over SO(k).

    V, T : (m, k, k) source and target matrices.
    Returns G* in SO(k).

    The result is verified orthogonal before being returned.  Two hardening
    steps, both of which matter on real (badly conditioned, near-duplicate)
    embedding data:

      * the determinant sign comes from `slogdet`, not `det`.  `np.sign` of a
        det that underflows to exactly 0.0 returns 0.0, which would
        produce a rank-deficient G = U diag(1,...,1,0) W^T -- not a rotation,
        and not flagged anywhere downstream.
      * if the returned G still fails the orthogonality check, it is repaired
        by a polar projection and re-checked; an unrepairable G raises rather
        than propagating a non-rotation into the rotation-angle diagnostic.
    """
    if V.shape != T.shape:
        raise ValueError("V and T must have the same shape (m, k, k)")
    if not (np.isfinite(V).all() and np.isfinite(T).all()):
        raise ValueError("V and T must be finite")

    k = V.shape[-1]
    M = np.einsum("mik,mjk->ij", T, V)              # sum_j T_j V_j^T
    U, _, Wt = np.linalg.svd(M)

    sign, _ = np.linalg.slogdet(U @ Wt)
    if sign == 0.0:                                  # pragma: no cover
        sign = 1.0
    d = np.ones(k)
    d[-1] = sign
    G = U @ np.diag(d) @ Wt

    thresh = tol * np.sqrt(k)
    if orthogonality_error(G) > thresh:              # pragma: no cover
        Uu, _, Vv = np.linalg.svd(G)                 # polar repair
        G = Uu @ Vv
        s2, _ = np.linalg.slogdet(G)
        if s2 < 0:
            Uu[:, -1] *= -1.0
            G = Uu @ Vv
        if orthogonality_error(G) > thresh:
            raise RuntimeError(
                f"procrustes_so produced a non-orthogonal G "
                f"(||G^T G - I||_F = {orthogonality_error(G):.3e} > {thresh:.3e}). "
                f"cond(M) = {np.linalg.cond(M):.3e}. This should not happen; "
                f"please report the input.")
    return G


def linear_baseline(V: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    Unconstrained least-squares map W* = argmin_W sum_j ||W V_j - T_j||_F^2.
    This is the denominator of the paper's relative residual.
    """
    A = np.einsum("mik,mjk->ij", T, V)              # sum T_j V_j^T
    B = np.einsum("mik,mjk->ij", V, V)              # sum V_j V_j^T
    return A @ np.linalg.pinv(B)


def linear_baseline_flat(V: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    The baseline the manuscript's code actually uses: a d x d map on the
    flattened vectors, from torch.linalg.lstsq.

        A = V.view(m, d);  B = T.view(m, d);  W = lstsq(A, B).solution

    with d = k^2 = 1024.  That is a 1024 x 1024 operator -- 1,048,576 free
    parameters -- fitted from m = 3 training examples, so it is the minimum-norm
    solution of a massively underdetermined system and generalises terribly.

    Section 4.1 of the manuscript describes a different object: a k x k map,
    "the ordinary least-squares map from the reshaped source embeddings to the
    reshaped targets", and states twice that "no d x d linear operator is ever
    estimated" and "at no point is a d x d map formed".  Both statements are
    contradicted by the code.  This matters because W* is the denominator of
    every relative residual in Table tab:structure: an inflated denominator
    makes every constrained fit look good.
    """
    m = V.shape[0]
    A = V.reshape(m, -1)
    B = T.reshape(m, -1)
    W, *_ = np.linalg.lstsq(A, B, rcond=None)        # (d, d), minimum norm
    return W


def residual_flat(W: np.ndarray, V: np.ndarray, T: np.ndarray) -> float:
    """Sum of squared errors of the flattened map v -> v @ W."""
    m = V.shape[0]
    return float(np.sum((V.reshape(m, -1) @ W - T.reshape(m, -1)) ** 2))


def residual(G: np.ndarray, V: np.ndarray, T: np.ndarray) -> float:
    """Sum of squared Frobenius errors sum_j ||G V_j - T_j||_F^2."""
    return float(np.sum((np.einsum("ij,mjk->mik", G, V) - T) ** 2))


def rotation_angles(G: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    """
    Rotation angles of G in SO(k), in degrees: the arguments of its complex
    eigenvalues.  If these are all ~0 the map is essentially the identity and
    there is no rotational structure to speak of.

    Returns an empty array if G is not orthogonal.  "Rotation angle" has no
    meaning for a general matrix -- the eigenvalue arguments of the
    unconstrained least-squares map W*, for instance, are not rotations, and
    reporting them alongside the SO fit invites a false comparison.
    """
    k = G.shape[0]
    # tolerance scaled by sqrt(k) (the Frobenius norm of I_k), and loose enough
    # that a true rotation is never rejected for round-off: we only need to
    # know "is this a rotation", not to certify it to machine precision
    if orthogonality_error(G) > tol * np.sqrt(k):
        return np.array([])
    ev = np.linalg.eigvals(G)
    ang = np.degrees(np.abs(np.angle(ev)))
    return np.sort(ang)[::-1]


# --------------------------------------------------------------------------- #
# the paper's procedure, for comparison
# --------------------------------------------------------------------------- #
def fit_generator_adam(V: np.ndarray, T: np.ndarray, subspace: str,
                       iters: int = 300, lr: float = 5e-3,
                       seed: int = 0) -> np.ndarray:
    """
    Reproduces the manuscript's structure-selection fit: Adam on
        min_X sum_j || exp(Proj_V X) V_j - T_j ||_F^2
    for `iters` steps from X = 0, using the stated learning rate.

    Implemented in NumPy with the exact Frechet-derivative gradient, so it is
    faithful to the described procedure and independent of any autodiff version.
    """
    from scipy.linalg import expm_frechet

    k = V.shape[-1]
    X = np.zeros((k, k))
    m1 = np.zeros_like(X)
    m2 = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    del seed                                   # the fit is deterministic: X starts at 0

    for t in range(1, iters + 1):
        P = proj(subspace, X)
        E = expm(P)
        Rm = np.einsum("ij,mjk->mik", E, V) - T            # residual matrices
        # d/dE sum ||E V_j - T_j||^2 = 2 sum (E V_j - T_j) V_j^T
        dE = 2.0 * np.einsum("mik,mjk->ij", Rm, V)
        # pull back through exp: <dE, Dexp(P)[H]> = <Dexp(P)^*[dE], H>;
        # the adjoint of the Frechet derivative is Dexp(P^T)[.]
        dP = expm_frechet(P.T, dE, compute_expm=False)
        g = proj(subspace, dP)                             # project onto the subspace
        m1 = b1 * m1 + (1 - b1) * g
        m2 = b2 * m2 + (1 - b2) * g * g
        mh = m1 / (1 - b1 ** t)
        vh = m2 / (1 - b2 ** t)
        X = X - lr * mh / (np.sqrt(vh) + eps)

    return proj(subspace, X)


# --------------------------------------------------------------------------- #
# the full comparison
# --------------------------------------------------------------------------- #
def compare_fits(V_train: np.ndarray, T_train: np.ndarray,
                 V_test: np.ndarray, T_test: np.ndarray,
                 adam_iters: int = 300, adam_lr: float = 5e-3,
                 baseline: str = "matrix32") -> pd.DataFrame:
    """
    Fit every candidate on the training pairs and evaluate held-out residuals,
    including the identity and exact-Procrustes baselines.

    The reported `rel_residual` uses the manuscript's convention:
        sqrt( test_loss(model) / test_loss(unconstrained linear W*) )
    so values below 1.0 mean "better held-out fit than the linear baseline".
    """
    k = V_train.shape[-1]
    rows = []

    if baseline == "matrix32":                       # what Section 4.1 describes
        W = linear_baseline(V_train, T_train)
        denom = residual(W, V_test, T_test)
        base_name = "unconstrained linear W* k x k (paper Sec. 4.1)"
        base_params = k * k
    elif baseline == "flat1024":                     # what the code does
        Wf = linear_baseline_flat(V_train, T_train)
        denom = residual_flat(Wf, V_test, T_test)
        W = np.eye(k)                                # placeholder for reporting
        base_name = "unconstrained linear W* d x d FLAT (repository code)"
        base_params = (k * k) ** 2
    else:
        raise ValueError(f"baseline must be 'matrix32' or 'flat1024', got {baseline!r}")

    def add(name, G, X=None, note="", param=np.nan):
        loss = residual(G, V_test, T_test)
        ang = rotation_angles(G)
        rows.append(dict(
            model=name,
            fitted_scale=param,
            test_loss=loss,
            rel_residual=math.sqrt(loss / denom) if denom > 0 else np.nan,
            train_loss=residual(G, V_train, T_train),
            X_fro=(fro_norm(X) if X is not None else np.nan),
            G_minus_I_fro=fro_norm(G - np.eye(k)),
            orthogonality_err=orthogonality_error(G),
            is_orthogonal=bool(ang.size > 0),
            max_rotation_deg=float(ang[0]) if ang.size else np.nan,
            # median over the floor(k/2) distinct conjugate-pair angles, not
            # over all k eigenvalues: the trivial +1 eigenvalues would otherwise
            # drag the median to 0 for a single-plane rotation
            median_rotation_deg=(float(np.median(ang[: max(1, len(ang) // 2)]))
                                 if ang.size else np.nan),
            note=note,
        ))

    # --- the baselines the paper does not report ---
    add("identity G = I", np.eye(k), note="S-1: is the map essentially the identity?")
    s = float(np.einsum("mik,mik->", T_train, V_train) /
              max(np.einsum("mik,mik->", V_train, V_train), 1e-300))
    # the fitted c goes in its own column, not in the model name: a name that
    # changes per split fragments any groupby across leave-one-out splits
    add("scaled identity cI", s * np.eye(k), note="S-1 baseline", param=s)
    rows.append(dict(model=f"{base_name} (denominator)", fitted_scale=np.nan,
                     test_loss=denom, rel_residual=1.0,
                     train_loss=np.nan, X_fro=np.nan,
                     G_minus_I_fro=np.nan, orthogonality_err=np.nan,
                     is_orthogonal=False, max_rotation_deg=np.nan,
                     median_rotation_deg=np.nan,
                     note=f"{base_params:,} free parameters fitted from "
                          f"{V_train.shape[0]} training pairs"))

    # --- exact SO fit ---
    G_star = procrustes_so(V_train, T_train)
    try:
        X_star = np.real(logm(G_star))
        skew_err = fro_norm(X_star + X_star.T)
        # A pi-rotation block makes the principal logarithm purely imaginary;
        # np.real then returns 0, whose skew residual is ~1e-15 (the skewness
        # check passes) even though exp(X_star) != G_star.  Verify the
        # reconstruction, not just the skewness.
        recon_err = fro_norm(expm(X_star) - G_star)
        if recon_err > 1e-8:
            X_star = None
    except Exception:                                       # pragma: no cover
        X_star, skew_err, recon_err = None, np.nan, np.nan
    add("so(k) EXACT (orthogonal Procrustes)", G_star, X_star,
        note=(f"T-13 global optimum; ||X+X^T||_F={skew_err:.2e}; "
              f"||exp(X)-G||_F={recon_err:.2e}"
              + ("  [logm unreliable: X_fro suppressed]" if X_star is None else "")))

    # --- the paper's Adam fits ---
    for sub in ("so", "sl", "sym", "gl"):
        Xh = fit_generator_adam(V_train, T_train, sub, adam_iters, adam_lr)
        add(f"{sub}(k) Adam {adam_iters} iters", expm(Xh), Xh,
            note="manuscript procedure")

    df = pd.DataFrame(rows)

    # how far is the Adam SO fit from the exact optimum?
    tr_exact = float(df.loc[df.model.str.contains("EXACT"), "train_loss"].iloc[0])
    tr_adam = float(df.loc[df.model.str.startswith("so(k) Adam"), "train_loss"].iloc[0])
    df.attrs["so_adam_suboptimality_abs"] = tr_adam - tr_exact
    df.attrs["so_adam_suboptimality_rel"] = (
        (tr_adam - tr_exact) / tr_exact if tr_exact > 0 else np.nan)
    return df


# --------------------------------------------------------------------------- #
# exhaustive permutation test (S-4)
# --------------------------------------------------------------------------- #
def exhaustive_permutation_test(V: np.ndarray, T: np.ndarray,
                                max_exact: int = 8,
                                n_random: int = 10_000,
                                seed: int = 0) -> dict:
    """
    Permutation test on the source/target pairing, using the exact SO fit so
    that every permutation is refitted at its global optimum (which is what
    makes exhaustive enumeration cheap enough to do properly).

    For m <= max_exact all m! pairings are enumerated and
        p_exact = #{sigma : rho_sigma <= rho_orig} / m!
    with minimum attainable value 1/m!.  For larger m, n_random distinct random
    permutations are used instead.

    Note on the manuscript's protocol: with m = 5 there are only 5! = 120
    distinct permutations, so drawing B = 128 with replacement cannot resolve a
    p-value below 1/129 and necessarily duplicates draws.  Enumeration removes
    both problems.
    """
    m = V.shape[0]
    rho_orig = residual(procrustes_so(V, T), V, T)

    if m <= max_exact:
        perms = list(itertools.permutations(range(m)))
        exact = True
    else:
        # Sampled permutations must not be reported as n_le/B: the identity is
        # then included only by chance and the estimator can return p = 0, which
        # no valid permutation test can attain.  Use the Phipson-Smyth
        # estimator (1 + n_le)/(1 + B), whose minimum is 1/(1 + B).
        rng = np.random.default_rng(seed)
        seen, perms = set(), []
        while len(perms) < n_random:
            p = tuple(rng.permutation(m))
            if p not in seen:
                seen.add(p)
                perms.append(p)
        exact = False

    rhos = np.array([residual(procrustes_so(V, T[list(p)]), V, T[list(p)])
                     for p in perms])
    tol = abs(rho_orig) * 1e-12 + 1e-15               # relative, not absolute
    n_le = int((rhos <= rho_orig + tol).sum())
    if exact:
        p_value = n_le / len(perms)
        min_p = 1.0 / len(perms)
    else:
        p_value = (1 + n_le) / (1 + len(perms))
        min_p = 1.0 / (1 + len(perms))
    return dict(
        m=m, exact=exact, n_permutations=len(perms),
        rho_original=rho_orig,
        rho_permuted_mean=float(rhos.mean()),
        rho_permuted_min=float(rhos.min()),
        n_at_or_below_original=n_le,
        p_value=p_value,
        min_attainable_p=min_p,
        estimator=("exhaustive n_le/m!" if exact
                   else "Phipson-Smyth (1+n_le)/(1+B)"),
    )
