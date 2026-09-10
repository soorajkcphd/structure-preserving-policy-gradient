"""
Matrix Lie algebra utilities: Frobenius-orthogonal projections and orthonormal bases.

Conventions used everywhere in this package
-------------------------------------------
* Frobenius inner product:  <A,B>_F = tr(A^T B) = sum(A * B)
* Frobenius norm:           ||A||_F  = sqrt(<A,A>_F)
* A "basis" of a subspace V is returned as an array of shape (dim, n, n) whose
  elements are **Frobenius-orthonormal**:  <B_k, B_l>_F = delta_kl.

  This orthonormality is what makes the coordinate Hessian's spectral norm equal
  to the operator norm with respect to the Frobenius metric.  Every smoothness
  bound in the paper is stated in that norm, so getting this right is essential.

Subspaces
---------
    so(n)   = {X : X^T = -X}            dim = n(n-1)/2   (Lie algebra, compact)
    sym(n)  = {X : X^T =  X}            dim = n(n+1)/2   (Jordan algebra, not Lie)
    sl(n)   = {X : tr(X) = 0}           dim = n^2 - 1    (Lie algebra, non-compact)
    gl(n)   = R^{n x n}                 dim = n^2
    rand(d) = uniformly random d-dim linear subspace of R^{n x n}
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "SUBSPACES", "fro_ip", "fro_norm",
    "proj", "subspace_dim", "orthonormal_basis", "random_subspace_basis",
    "to_coords", "from_coords", "random_element", "check_basis",
]

SUBSPACES = ("so", "sym", "sl", "gl")


# --------------------------------------------------------------------------- #
# inner products
# --------------------------------------------------------------------------- #
def fro_ip(A: np.ndarray, B: np.ndarray) -> float:
    """Frobenius inner product <A,B>_F = tr(A^T B)."""
    return float(np.sum(A * B))


def fro_norm(A: np.ndarray) -> float:
    """Frobenius norm ||A||_F."""
    return float(np.sqrt(np.sum(A * A)))


# --------------------------------------------------------------------------- #
# projections (closed form, O(n^2), non-expansive)
# --------------------------------------------------------------------------- #
def proj(name: str, M: np.ndarray) -> np.ndarray:
    """Frobenius-orthogonal projection of M onto the named subspace."""
    if name == "so":
        return 0.5 * (M - M.T)
    if name == "sym":
        return 0.5 * (M + M.T)
    if name == "sl":
        n = M.shape[0]
        return M - (np.trace(M) / n) * np.eye(n)
    if name == "gl":
        return M.copy()
    raise ValueError(f"unknown subspace {name!r}; expected one of {SUBSPACES}")


def subspace_dim(name: str, n: int) -> int:
    """Dimension of the named subspace of R^{n x n}."""
    return {
        "so": n * (n - 1) // 2,
        "sym": n * (n + 1) // 2,
        "sl": n * n - 1,
        "gl": n * n,
    }[name]


# --------------------------------------------------------------------------- #
# orthonormal bases
# --------------------------------------------------------------------------- #
def orthonormal_basis(name: str, n: int) -> np.ndarray:
    """
    Frobenius-orthonormal basis of the named subspace, shape (dim, n, n).

    so, sym, gl use explicit constructions that are orthonormal by inspection.
    sl uses an SVD of the projected standard basis (exact to machine precision,
    just not written in closed form).
    """
    if name == "so":
        B = np.zeros((subspace_dim("so", n), n, n))
        k = 0
        for i in range(n):
            for j in range(i + 1, n):
                B[k, i, j] = 1.0 / np.sqrt(2.0)
                B[k, j, i] = -1.0 / np.sqrt(2.0)
                k += 1
        return B

    if name == "sym":
        B = np.zeros((subspace_dim("sym", n), n, n))
        k = 0
        for i in range(n):
            B[k, i, i] = 1.0
            k += 1
        for i in range(n):
            for j in range(i + 1, n):
                B[k, i, j] = 1.0 / np.sqrt(2.0)
                B[k, j, i] = 1.0 / np.sqrt(2.0)
                k += 1
        return B

    if name == "gl":
        B = np.zeros((n * n, n, n))
        for k in range(n * n):
            B[k, k // n, k % n] = 1.0
        return B

    if name == "sl":
        # Project the standard basis onto sl(n), then orthonormalise the span.
        #
        # Note: an un-pivoted QR does not in general place a basis of the range
        # in its first d columns when the matrix is rank deficient, so we use an
        # SVD, whose first d left singular vectors span the range by
        # construction.  (The QR happens to work here because the single
        # dependency among the projected standard basis involves the last
        # column, but relying on that is fragile.)
        d = subspace_dim("sl", n)
        if d == 0:                                         # sl(1) = {0}
            return np.zeros((0, n, n))
        E = np.eye(n * n).reshape(n * n, n, n)
        P = np.stack([proj("sl", e) for e in E])           # (n^2, n, n), rank n^2-1
        U, sv, _ = np.linalg.svd(P.reshape(n * n, -1).T, full_matrices=False)
        if sv[d - 1] <= 1e-10 * sv[0]:                     # pragma: no cover
            raise RuntimeError(f"sl({n}) basis is rank deficient: sv={sv[:d]}")
        return U[:, :d].T.reshape(d, n, n).copy()

    raise ValueError(f"unknown subspace {name!r}")


def random_subspace_basis(n: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    """
    Frobenius-orthonormal basis of a uniformly random `dim`-dimensional subspace
    of R^{n x n}.  This is the structure-free, dimension-matched control (R-6).
    """
    if not 1 <= dim <= n * n:
        raise ValueError(f"dim must lie in [1, {n * n}], got {dim}")
    G = rng.standard_normal((n * n, dim))
    Q, _ = np.linalg.qr(G)
    return Q.T.reshape(dim, n, n).copy()


# --------------------------------------------------------------------------- #
# coordinates
# --------------------------------------------------------------------------- #
def to_coords(basis: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Coordinates of X in an orthonormal basis: c_k = <B_k, X>_F."""
    return basis.reshape(basis.shape[0], -1) @ X.reshape(-1)


def from_coords(basis: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Matrix with the given coordinates: X = sum_k c_k B_k."""
    n = basis.shape[1]
    return (c @ basis.reshape(basis.shape[0], -1)).reshape(n, n)


def random_element(name: str, n: int, rng: np.random.Generator,
                   fro: float = 1.0) -> np.ndarray:
    """
    A random element of the named subspace with prescribed Frobenius norm.
    Sampling is uniform on the sphere of the subspace (isotropic Gaussian,
    projected, then rescaled), so no direction is favoured.
    """
    X = proj(name, rng.standard_normal((n, n)))
    nx = fro_norm(X)
    if nx == 0.0:                                  # measure zero, but be safe
        return X
    return X * (fro / nx)


# --------------------------------------------------------------------------- #
# self-check
# --------------------------------------------------------------------------- #
def check_basis(name: str, n: int, basis: np.ndarray, tol: float = 1e-10) -> dict:
    """
    Verify that `basis` is a Frobenius-orthonormal basis of the named subspace.
    Returns a dict of diagnostics; raises AssertionError on failure.
    """
    d_expected = subspace_dim(name, n)
    if d_expected == 0:                                    # e.g. so(1), sl(1)
        assert basis.shape == (0, n, n)
        return {"subspace": name, "n": n, "dim": 0, "gram_err": 0.0,
                "in_subspace_err": 0.0, "completeness_err": 0.0}
    assert basis.shape == (d_expected, n, n), (
        f"{name}({n}): basis shape {basis.shape} != {(d_expected, n, n)}")

    flat = basis.reshape(d_expected, -1)
    gram = flat @ flat.T
    gram_err = float(np.abs(gram - np.eye(d_expected)).max())
    assert gram_err < tol, f"{name}({n}): basis not orthonormal, err={gram_err:.3e}"

    # every basis element must already lie in the subspace
    in_sub = max(fro_norm(proj(name, B) - B) for B in basis)
    assert in_sub < tol, f"{name}({n}): basis leaves the subspace, err={in_sub:.3e}"

    # sum_k B_k <B_k, M> must equal proj(M) for arbitrary M (completeness)
    rng = np.random.default_rng(0)
    M = rng.standard_normal((n, n))
    recon = from_coords(basis, to_coords(basis, M))
    complete_err = fro_norm(recon - proj(name, M))
    assert complete_err < tol, (
        f"{name}({n}): basis incomplete, err={complete_err:.3e}")

    return {"subspace": name, "n": n, "dim": d_expected,
            "gram_err": gram_err, "in_subspace_err": in_sub,
            "completeness_err": complete_err}
