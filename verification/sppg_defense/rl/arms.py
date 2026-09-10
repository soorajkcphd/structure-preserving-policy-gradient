"""
Parameterisation arms for the RL comparisons (R-2, R-4, R-5, R-6, R-7, R-16).

These are drop-in replacements for the projection step in Algorithm alg:spppo:

        theta <- Proj_g( theta + alpha * g )

Each arm exposes the same interface, so a single training loop can run every
condition with nothing else changed -- which is what makes the comparison
matched.  Nothing here depends on the environment or the policy network; wire
`arm.project(theta)` in where `Proj_so(32)` currently is, and
`arm.diagnostics(theta)` into the per-iteration logger.

Arms
----
    AlgebraArm("so")      the method                       dim 496 at n=32
    AlgebraArm("sl")      non-compact Lie                  dim 1023
    AlgebraArm("sym")     non-Lie (Jordan) subspace        dim 528
    AlgebraArm("gl")      unconstrained                    dim 1024
    RandomSubspaceArm     R-6: dimension-matched, structure-free control.
                          It has exactly 496 dimensions and an
                          O(n^2)-equivalent non-expansive
                          projector, but no algebraic structure and no
                          compactness.  If it matches so(32), the effect is
                          dimensional regularisation, not geometry.
    ConjugatedAlgebraArm  P so(n) P^{-1} for a random non-orthogonal invertible
                          P: abstractly the same Lie algebra, a different
                          embedding in R^{n x n}, and a different projector.
                          (An orthogonally conjugated arm would be a null
                          control -- Q so(n) Q^T = so(n) identically; see
                          orthogonal_conjugation_is_a_noop.)
    LoRAArm               budget-matched low-rank control (R-16).
    LowRankSoArm          theta = U V^T - V U^T, the low-rank so(n) variant
                          suggested in the Limitations section (R-17).

Every arm reports the structure-preservation diagnostics of T-11:
skewness residual, orthogonality error of exp(theta), det, spectral radius --
because rho = 1 alone is not sufficient evidence for the geometric channel
(see t10_exponential_witness.py, nilpotent panel).
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from ..algebra import (SUBSPACES, fro_norm, orthonormal_basis, proj,
                       random_subspace_basis, subspace_dim)

__all__ = ["Arm", "AlgebraArm", "RandomSubspaceArm", "ConjugatedAlgebraArm",
           "LoRAArm", "LowRankSoArm", "make_arm", "structure_diagnostics",
           "orthogonal_conjugation_is_a_noop"]


def structure_diagnostics(theta: np.ndarray, compute_exp: bool = True) -> dict:
    """
    T-11 diagnostics.  Log these every iteration for every arm.

    The manuscript reports only rho(exp theta).  The orthogonality error is the
    quantity that actually certifies the geometric channel; rho = 1 also holds
    for nilpotent directions whose operator norm grows without bound.
    """
    n = theta.shape[0]
    out = {
        "theta_fro": fro_norm(theta),
        "skew_residual": fro_norm(theta + theta.T),      # 0 iff theta in so(n)
        "sym_residual": fro_norm(theta - theta.T),
        "trace": float(np.trace(theta)),
    }
    if compute_exp:
        E = expm(theta)
        out.update({
            "orthogonality_err": fro_norm(E.T @ E - np.eye(n)),
            "det_minus_one": float(np.linalg.det(E) - 1.0),
            "spectral_radius": float(np.max(np.abs(np.linalg.eigvals(E)))),
            "op_norm": float(np.linalg.norm(E, 2)),      # rho=1 does not imply this is 1
            "cond": float(np.linalg.cond(E)),
        })
    return out


class Arm:
    """Base interface: project a matrix onto the arm's parameter set."""

    name: str = "base"
    dim: int = 0

    def project(self, M: np.ndarray) -> np.ndarray:      # pragma: no cover
        raise NotImplementedError

    def diagnostics(self, theta: np.ndarray, compute_exp: bool = True) -> dict:
        d = structure_diagnostics(theta, compute_exp)
        d["arm"] = self.name
        d["dim"] = self.dim
        return d


class AlgebraArm(Arm):
    """Closed-form Frobenius projection onto so / sym / sl / gl."""

    def __init__(self, subspace: str, n: int):
        if subspace not in SUBSPACES:
            raise ValueError(f"subspace must be one of {SUBSPACES}")
        self.subspace = subspace
        self.n = n
        self.name = f"{subspace}({n})"
        self.dim = subspace_dim(subspace, n)

    def project(self, M):
        return proj(self.subspace, M)


class RandomSubspaceArm(Arm):
    """
    R-6.  Projection onto a uniformly random `dim`-dimensional subspace.

    Resample per seed (pass the seed's rng) so the result is not an artifact of
    one lucky subspace.  Default dim = dim so(n) for an exact dimension match.
    """

    def __init__(self, n: int, rng: np.random.Generator, dim: int | None = None):
        self.n = n
        self.dim = dim if dim is not None else subspace_dim("so", n)
        self.name = f"random-{self.dim}({n})"
        B = random_subspace_basis(n, self.dim, rng)
        self._flat = B.reshape(self.dim, -1)             # orthonormal rows

    def project(self, M):
        v = M.reshape(-1)
        return (self._flat.T @ (self._flat @ v)).reshape(self.n, self.n)


def orthogonal_conjugation_is_a_noop(n: int, rng: np.random.Generator,
                                     n_trials: int = 20) -> float:
    """
    Proof-by-computation that conjugating so(n) by an orthogonal Q changes
    nothing:

        Proj_so(Q^T M Q) = 1/2 (Q^T M Q - Q^T M^T Q) = Q^T Proj_so(M) Q
        =>  Q Proj_so(Q^T M Q) Q^T = Proj_so(M)   identically.

    So an "orthogonally conjugated so(n)" arm is bit-for-bit the plain so(n)
    arm and cannot isolate orientation from structure -- any invariance
    concluded from it would be vacuous.  This function returns the maximum
    observed discrepancy (machine epsilon), and exists so the invariance can be
    stated as a verified one-line fact instead of burned as an experiment arm.

    The frame-dependence that Section 4.1 actually worries about lives in the
    reshape phi (a permutation of the 1024 embedding coordinates changes what
    so(32) means on the representation), which is an environment-level
    intervention -- experiments S-3 and R-14 -- not a projection-level one.
    """
    worst = 0.0
    for _ in range(n_trials):
        A = rng.standard_normal((n, n))
        Q, R = np.linalg.qr(A)
        Q = Q * np.sign(np.diag(R))
        M = rng.standard_normal((n, n))
        worst = max(worst, fro_norm(Q @ proj("so", Q.T @ M @ Q) @ Q.T
                                    - proj("so", M)))
    return worst


class ConjugatedAlgebraArm(Arm):
    """
    P so(n) P^{-1} for a random invertible, non-orthogonal P.

    This is a different 496-dimensional subspace: abstractly
    isomorphic to so(n) as a Lie algebra, but a different embedding in
    R^{n x n}, and its Frobenius projector is not Proj_so.  It therefore tests
    whether the benefit attaches to the abstract algebra or to its particular
    embedding -- which the orthogonal version cannot do.

    `cond` controls how far P is from orthogonal (cond = 1 recovers the no-op).
    """

    def __init__(self, n: int, rng: np.random.Generator, cond: float = 4.0):
        self.n = n
        self.dim = subspace_dim("so", n)
        self.name = f"P-so({n})-Pinv(cond={cond:g})"
        Q1, _ = np.linalg.qr(rng.standard_normal((n, n)))
        Q2, _ = np.linalg.qr(rng.standard_normal((n, n)))
        sv = np.geomspace(1.0, cond, n)
        self.P = Q1 @ np.diag(sv) @ Q2
        self.Pinv = np.linalg.inv(self.P)
        # orthonormal basis of P so(n) P^{-1}
        Bso = orthonormal_basis("so", n)
        Braw = np.stack([self.P @ B @ self.Pinv for B in Bso])
        Q, _ = np.linalg.qr(Braw.reshape(self.dim, -1).T)
        self._flat = Q[:, :self.dim].T                 # (dim, n*n), orthonormal

    def project(self, M):
        v = M.reshape(-1)
        return (self._flat.T @ (self._flat @ v)).reshape(self.n, self.n)


class LoRAArm(Arm):
    """
    R-16.  Rank-r truncation, a budget-matched structural control.

    Parameter count is r * (2n - r); pick r so that this matches dim so(n) as
    closely as possible (`LoRAArm.rank_for_budget`).
    """

    def __init__(self, n: int, rank: int):
        self.n = n
        self.rank = rank
        # Two parameter counts, and the distinction matters for "matched budget":
        #   n_parameters = 2 n r        raw count of A (n x r) and B (r x n).
        #                               This is the manuscript's convention
        #                               ("LoRA r=8 uses 512 parameters" at n=32).
        #   dim          = r(2n - r)    dimension of the rank-r matrix manifold,
        #                               i.e. the count after removing the
        #                               r x r gauge redundancy A -> AG, B -> G^-1 B.
        # so(32) has 496 free parameters, so r = 8 matches on the first
        # convention (512) and r = 9 matches on the second (495).  State which
        # one was matched when comparing 496 against 512.
        self.dim = rank * (2 * n - rank)
        self.n_parameters = 2 * n * rank
        self.name = f"lora(r={rank},{n})"

    @staticmethod
    def rank_for_budget(n: int, budget: int, convention: str = "manifold") -> int:
        """
        Largest rank fitting a parameter budget.
        convention="manifold" uses r(2n-r); convention="raw" uses 2nr (the
        manuscript's convention).
        """
        best = 0
        for r in range(1, n + 1):
            cost = r * (2 * n - r) if convention == "manifold" else 2 * n * r
            if cost <= budget:
                best = r
        if best == 0:
            raise ValueError(f"no rank >= 1 fits a budget of {budget} at n={n} "
                             f"under the {convention!r} convention")
        return best

    def project(self, M):
        U, S, Vt = np.linalg.svd(M)
        S[self.rank:] = 0.0
        return (U * S) @ Vt


class LowRankSoArm(Arm):
    """
    R-17.  theta = U V^T - V U^T with U, V in R^{n x r}: skew by construction and
    of rank at most 2r.  This is the low-rank so(n) parameterisation the
    Limitations section proposes; the projection is the best rank-2r skew
    approximation, obtained from the SVD of the skew part.
    """

    def __init__(self, n: int, rank: int):
        if rank < 1:
            raise ValueError("rank must be >= 1")
        if 2 * rank > n:
            raise ValueError(
                f"2*rank ({2 * rank}) exceeds n ({n}); the parameterisation "
                "then degenerates to plain Proj_so and the dimension formula "
                "2nr - r(2r+1) is no longer the variety dimension")
        self.n = n
        self.rank = rank
        # dimension of the variety of skew matrices of rank <= 2r, valid for
        # 2r <= n; capped at dim so(n) for safety
        self.dim = min(2 * n * rank - rank * (2 * rank + 1), subspace_dim("so", n))
        self.name = f"lowrank-so(r={rank},{n})"

    def project(self, M):
        """
        Best rank-2r skew approximation via the real Schur (Youla) form.

        Truncating the SVD is not safe here: a skew matrix has paired singular
        values, and when two pairs coincide to within round-off the SVD mixes
        them, the truncation is no longer skew, and re-skewing afterwards
        returns a matrix of rank 2r+2.  The real Schur form is block-diagonal
        with 2x2 rotation blocks, so zeroing the smallest blocks is exactly
        skew and exactly rank 2r by construction.
        """
        from scipy.linalg import schur

        S = proj("so", M)
        Tq, Z = schur(S, output="real")
        n = self.n
        # magnitude of each 2x2 block on the sub-diagonal
        mags, idx = [], []
        i = 0
        while i < n - 1:
            if abs(Tq[i + 1, i]) > 0 or abs(Tq[i, i + 1]) > 0:
                mags.append(abs(Tq[i + 1, i]))
                idx.append(i)
                i += 2
            else:
                i += 1
        keep = set(np.argsort(mags)[::-1][:self.rank].tolist()) if mags else set()
        Tk = np.zeros_like(Tq)
        for b, i0 in enumerate(idx):
            if b in keep:
                Tk[i0:i0 + 2, i0:i0 + 2] = Tq[i0:i0 + 2, i0:i0 + 2]
        return proj("so", Z @ Tk @ Z.T)


def make_arm(spec: str, n: int, rng: np.random.Generator | None = None) -> Arm:
    """
    Build an arm from a short string, for config-driven sweeps.

        "so", "sym", "sl", "gl"      algebra arms
        "random"                     dimension-matched random subspace (R-6)
        "random:D"                   random subspace of dimension D
        "conj-so" / "conj-so:C"      P so(n) P^{-1}, non-orthogonal P (cond C)
        "lora:R"                     LoRA of rank R
        "lora:budget"                LoRA at the so(n) parameter budget
        "lowrank-so:R"               low-rank so(n)
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    if spec in SUBSPACES:
        return AlgebraArm(spec, n)
    if spec == "random":
        return RandomSubspaceArm(n, rng)
    if spec.startswith("random:"):
        return RandomSubspaceArm(n, rng, int(spec.split(":")[1]))
    if spec == "conj-so":
        return ConjugatedAlgebraArm(n, rng)
    if spec.startswith("conj-so:"):
        return ConjugatedAlgebraArm(n, rng, cond=float(spec.split(":")[1]))
    if spec == "lora:budget":
        return LoRAArm(n, LoRAArm.rank_for_budget(n, subspace_dim("so", n)))
    if spec.startswith("lora:"):
        return LoRAArm(n, int(spec.split(":")[1]))
    if spec.startswith("lowrank-so:"):
        return LowRankSoArm(n, int(spec.split(":")[1]))
    raise ValueError(f"unknown arm spec {spec!r}")
