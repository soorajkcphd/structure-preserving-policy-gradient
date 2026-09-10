#!/usr/bin/env python3
"""
verify_math.py -- independent numerical verification of every analytical claim
in the manuscript.

    python verify_math.py

Why this exists
---------------
`sppg_defense/` checks the theory against the *project's own* implementation of
the objects involved.  That is useful, but it shares helper code with the thing
being checked, so a defect in a shared helper can hide a defect in a theorem.
This file shares nothing with the project: it rebuilds every object from numpy and
scipy alone -- projectors, matrix exponentials, Frechet derivatives by finite
difference, Popoviciu variance bounds, Procrustes by SVD, permutation counts --
and compares the result against the constant the paper prints.

Every check reports PASS/FAIL and the observed value, so a reader can see how
much slack a bound has rather than only whether it held.  Two of the checks
found real defects while the paper was being revised:

  * V7b showed the geometric-loss constant 1 is valid but not attained; the
    sharp value is 1/2, attained at n=2, theta=0, E=J/sqrt2, G=I.
  * V5 confirmed exp(sl) has determinant exactly 1 and exp(gl(n,R)) cannot
    reach a negative determinant, which retired a claim that the larger
    algebras "represent any M_env exactly".

Runtime is a few seconds; no GPU, no model, no CSV.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm

RESULTS: list[tuple[bool, str, str]] = []


def check(ok: bool, name: str, observed) -> None:
    RESULTS.append((bool(ok), name, str(observed)))


# --------------------------------------------------------------------------
# projectors
# --------------------------------------------------------------------------
def P_so(X):  return 0.5 * (X - X.T)
def P_sym(X): return 0.5 * (X + X.T)
def P_sl(X):  return X - np.trace(X) / X.shape[0] * np.eye(X.shape[0])


def v1_projectors(n=32, trials=200, rng=None):
    """Idempotent, Frobenius-orthogonal complement, non-expansive, Pythagorean."""
    rng = rng or np.random.default_rng(0)
    e = []
    for _ in range(trials):
        X = rng.standard_normal((n, n))
        e.append(abs(np.linalg.norm(P_so(P_so(X)) - P_so(X))))          # idempotent
        e.append(abs(np.sum(P_so(X) * P_sym(X))))                        # orthogonal
        e.append(max(0.0, np.linalg.norm(P_so(X)) - np.linalg.norm(X)))  # non-expansive
        e.append(abs(np.linalg.norm(X) ** 2
                     - np.linalg.norm(P_so(X)) ** 2
                     - np.linalg.norm(P_sym(X)) ** 2))                   # Pythagoras
    check(max(e) < 1e-10, "V1  projector identities", f"max err {max(e):.2e}")


def v2_dimensions(n=32):
    d_so, d_sym, d_sl = n * (n - 1) // 2, n * (n + 1) // 2, n * n - 1
    ok = (d_so, d_sym, d_sl, d_so + d_sym) == (496, 528, 1023, n * n)
    check(ok, "V2  dim so/sym/sl = 496/528/1023 and 496+528=1024",
          f"{d_so}, {d_sym}, {d_sl}")


def v3_exp_so_in_SO(n=32, trials=50, rng=None):
    """exp of skew is orthogonal, det 1, spectral radius 1 -- at any radius."""
    rng = rng or np.random.default_rng(1)
    e = []
    for _ in range(trials):
        A = P_so(rng.standard_normal((n, n))) * rng.uniform(0.1, 50)
        E = expm(A)
        e.append(np.linalg.norm(E.T @ E - np.eye(n)))
        e.append(abs(np.linalg.det(E) - 1.0))
        e.append(abs(max(abs(np.linalg.eigvals(E))) - 1.0))
    check(max(e) < 1e-8, "V3  exp(so(n)) in SO(n), radii to 50", f"max err {max(e):.2e}")


def v4_sym_is_the_complement(n=32, trials=200, rng=None):
    """Proj_sym(Omega)=0 for Omega in so(n): the aligned-control claim."""
    rng = rng or np.random.default_rng(2)
    e = [abs(np.linalg.norm(P_sym(P_so(rng.standard_normal((n, n))))))
         for _ in range(trials)]
    check(max(e) < 1e-12, "V4  Proj_sym(Omega)=0 for Omega in so(32)", f"max {max(e):.2e}")


def v5_representability(n=8, trials=50, rng=None):
    """det exp X = e^{tr X}: exp(sl) has det 1; exp(gl) never has det<0."""
    rng = rng or np.random.default_rng(3)
    e = [abs(np.linalg.det(expm(X := rng.standard_normal((n, n))))
             - np.exp(np.trace(X))) for _ in range(trials)]
    sl = [X - np.trace(X) / n * np.eye(n) for X in
          (rng.standard_normal((n, n)) for _ in range(trials))]
    dev = max(abs(np.linalg.det(expm(X)) - 1.0) for X in sl)
    neg = sum(np.linalg.det(rng.standard_normal((n, n))) < 0 for _ in range(2000))
    check(max(e) < 1e-6 and dev < 1e-8 and 800 < neg < 1200,
          "V5  det exp X = e^{tr X}; exp(sl) det=1; ~half of Gaussians det<0",
          f"det err {max(e):.1e}, sl dev {dev:.1e}, det<0 in {neg}/2000")


# --------------------------------------------------------------------------
# derivative bounds
# --------------------------------------------------------------------------
def _dexp(A, E, h=1e-6):
    return (expm(A + h * E) - expm(A - h * E)) / (2 * h)


def _d2exp(A, E1, E2, h=1e-4):
    return (expm(A + h * E1 + h * E2) - expm(A + h * E1 - h * E2)
            - expm(A - h * E1 + h * E2) + expm(A - h * E1 - h * E2)) / (4 * h * h)


def v6_frechet(n=16, trials=30, rng=None):
    """||Dexp||<=1 and ||D2exp||<=1 on so(n) (Duhamel + orthogonal invariance)."""
    rng = rng or np.random.default_rng(4)
    r1, r2 = [], []
    for _ in range(trials):
        A = P_so(rng.standard_normal((n, n))) * rng.uniform(0.1, 8)
        E1 = P_so(rng.standard_normal((n, n))); E1 /= np.linalg.norm(E1)
        E2 = P_so(rng.standard_normal((n, n))); E2 /= np.linalg.norm(E2)
        r1.append(np.linalg.norm(_dexp(A, E1)))
        r2.append(np.linalg.norm(_d2exp(A, E1, E2)))
    check(max(r1) <= 1 + 1e-4, "V6a ||Dexp(A)[E]||_F <= ||E||_F on so(n)", f"max {max(r1):.6f}")
    check(max(r2) <= 1 + 1e-2, "V6b ||D2exp(A)[E1,E2]||_F <= 1", f"max {max(r2):.6f}")


def v7_geo_loss(n=16, trials=40, rng=None):
    """Geometric-loss smoothness: SO target, and the normalised general target.

    Also locates the sharp constant, which is 1/2, not the proved bound 1.
    """
    rng = rng or np.random.default_rng(5)

    def d2(theta, v, q, E, h=1e-4):
        f = lambda T: -float(expm(T) @ v @ q)
        return (f(theta + 2 * h * E) - 2 * f(theta) + f(theta - 2 * h * E)) / (4 * h * h)

    worst_so = worst_gen = 0.0
    for _ in range(trials):
        th = P_so(rng.standard_normal((n, n))) * rng.uniform(0.1, 6)
        v = rng.standard_normal(n); v /= np.linalg.norm(v)
        E = P_so(rng.standard_normal((n, n))); E /= np.linalg.norm(E)
        Q, R = np.linalg.qr(rng.standard_normal((n, n)))
        Q = Q @ np.diag(np.sign(np.diag(R)))
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1
        worst_so = max(worst_so, abs(d2(th, v, Q @ v, E)))
        M = rng.standard_normal((n, n))                  # arbitrary target
        q = M @ v / np.linalg.norm(M @ v)                # requires Mv != 0
        worst_gen = max(worst_gen, abs(d2(th, v, q, E)))
    check(worst_so <= 1 + 1e-2, "V7a lem:geo_smooth, SO target, bound 1", f"max {worst_so:.6f}")
    check(worst_gen <= 1 + 1e-2, "V7b same bound, normalised general target", f"max {worst_gen:.6f}")

    # the sharp constant, analytically: n=2, theta=0, E=J/sqrt2, G=I
    J = np.array([[0.0, 1.0], [-1.0, 0.0]]) / np.sqrt(2)
    sharp = abs(-float((J @ J @ np.array([1.0, 0.0])) @ np.array([1.0, 0.0])))
    check(abs(sharp - 0.5) < 1e-12,
          "V7c sharp constant is 1/2 (so the proved 1 is valid, not attained)",
          f"witness {sharp:.6f}")

    # and the fact underneath it: skew E has ||E||_op <= ||E||_F/sqrt2,
    # because eigenvalues come as +-i*sigma so every nonzero sv has even multiplicity
    worst = 0.0
    for m in (2, 4, 8, 16, 32):
        for _ in range(200):
            E = P_so(rng.standard_normal((m, m)))
            worst = max(worst, np.linalg.norm(E, 2) / np.linalg.norm(E, "fro"))
    check(worst <= 1 / np.sqrt(2) + 1e-9,
          "V7d ||E||_op <= ||E||_F/sqrt2 for skew E", f"sup {worst:.6f}")


def v8_softmax_hessian(n=16, B=1.0, rng=None):
    """Log-partition Hessian <= B^2, independent of ||theta||_F (Popoviciu)."""
    rng = rng or np.random.default_rng(6)
    A = [a / np.linalg.norm(a) * B for a in
         (rng.standard_normal((n, n)) for _ in range(8))]
    F = np.stack([P.ravel() for P in A])

    def top_eig(theta):
        z = np.array([np.sum(P * theta) for P in A]); z -= z.max()
        p = np.exp(z); p /= p.sum()
        C = (F * p[:, None]).T @ F - np.outer(F.T @ p, F.T @ p)
        return np.linalg.eigvalsh(C)[-1]

    worst = 0.0
    for R in (0, 1, 10, 100, 1e3, 1e4, 1e6):
        th = P_so(rng.standard_normal((n, n)))
        th = th / np.linalg.norm(th) * R if R > 0 else th * 0
        worst = max(worst, top_eig(th))
    check(worst <= B * B + 1e-6,
          "V8  log-partition Hessian <= B^2 for ||theta||_F up to 1e6", f"max {worst:.6f}")


# --------------------------------------------------------------------------
# discrete and statistical claims
# --------------------------------------------------------------------------
def v9_procrustes(k=8, trials=200, rng=None):
    """Determinant-corrected SVD solution is the global maximiser over SO(k)."""
    rng = rng or np.random.default_rng(7)
    err = []
    for _ in range(trials):
        Q, R = np.linalg.qr(rng.standard_normal((k, k)))
        Q = Q @ np.diag(np.sign(np.diag(R)))
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1
        V = [rng.standard_normal((k, k)) for _ in range(5)]
        T = [Q @ v for v in V]
        M = sum(t @ v.T for t, v in zip(T, V))
        U, _, Wt = np.linalg.svd(M)
        D = np.eye(k); D[-1, -1] = np.sign(np.linalg.det(U @ Wt))
        err.append(np.linalg.norm(U @ D @ Wt - Q))
    check(max(err) < 1e-8, "V9  Procrustes closed form recovers the planted rotation",
          f"max err {max(err):.2e}")


def v10_discrete():
    from itertools import permutations
    from math import factorial
    check(factorial(5) == 120, "V10a m=5 gives 120 orderings, min p = 1/120",
          f"{1/120:.4f}")
    r = list(range(5)); cnt = tot = 0
    for p in permutations(r):
        tot += 1
        d = sum((a - b) ** 2 for a, b in zip(r, p))
        if 1 - 6 * d / (5 * 24) <= -1 + 1e-12:
            cnt += 1
    check(abs(2 * cnt / tot - 0.0167) < 1e-3,
          "V10b Spearman rho=-1 at n=5: exact two-sided p", f"{2*cnt/tot:.4f}")
    check(abs(2 / 2 ** 10 - 0.00195) < 1e-5,
          "V10c Wilcoxon signed-rank n=10 all-positive", f"{2/2**10:.5f}")


def v11_reported_arithmetic():
    """Margins, ratios and differences the manuscript prints."""
    check(abs(0.05 * 2.804 - 0.1402) < 1e-3, "V11a Task-3 TOST margin = 5% of 2.804", f"{0.05*2.804:.4f}")
    check(abs(0.140 / 0.028 - 5.0) < 0.1,   "V11b interval sits a factor 5 inside it", f"{0.140/0.028:.2f}")
    check(abs(0.25 * 2.234 - 0.5585) < 1e-3, "V11c retraction margin = 25% of 2.234", f"{0.25*2.234:.4f}")
    check(abs(100 * 0.274 / 2.234 - 12.3) < 0.2, "V11d 0.274/2.234 = 12.3% (not 'at most 12%')",
          f"{100*0.274/2.234:.1f}%")
    # V11i-j: the minimum detectable effect at n=10 uses the noncentral t at
    # df=9, not the large-sample multiplier z_.975 + z_.80 = 2.802.  The normal
    # form gives 0.0054 / 0.0144, which deliver only ~70% power; these two
    # checks guard against using it.
    from scipy import optimize, stats as _st

    def _power(dz, n=10, alpha=0.05):
        df, nc, crit = n - 1, dz * np.sqrt(n), _st.t.ppf(1 - alpha / 2, n - 1)
        return _st.nct.sf(crit, df, nc) + _st.nct.cdf(-crit, df, nc)

    dz80 = optimize.brentq(lambda d: _power(d, 10) - 0.80, 0.1, 3.0)
    check(abs(dz80 - 0.996) < 0.01, "V11i 80% power at n=10 needs d_z = 1.00, not 0.886",
          f"d_z = {dz80:.4f}")
    for sd, printed, stale in ((0.0061, 0.0061, 0.0054), (0.0162, 0.0161, 0.0144)):
        check(abs(dz80 * sd - printed) < 5e-5 and _power(stale / sd, 10) < 0.75,
              f"V11j MDE at sd={sd} is {printed} (the normal-multiplier {stale} gives "
              f"{_power(stale/sd,10):.2f} power)", f"{dz80*sd:.5f}")
    L = 6 * 1.0 * 0.87 ** 2
    check(abs((1 / L) / 0.03 - 7.34) < 0.1, "V11e step-size margin at alpha=0.03", f"{(1/L)/0.03:.2f}x")
    check(abs((9.366 - 7.475) - 1.891) < 1e-3, "V11f sym geometry: sym-so", f"{9.366-7.475:.3f}")
    check(abs((9.673 - 9.455) - 0.218) < 1e-3, "V11g diagonal: baseline-so", f"{9.673-9.455:.3f}")
    check(abs((9.745 - 9.545) - 0.200) < 1e-3, "V11h identity: baseline-so", f"{9.745-9.545:.3f}")


def main() -> int:
    for fn in (v1_projectors, v2_dimensions, v3_exp_so_in_SO, v4_sym_is_the_complement,
               v5_representability, v6_frechet, v7_geo_loss, v8_softmax_hessian,
               v9_procrustes, v10_discrete, v11_reported_arithmetic):
        fn()
    width = max(len(n) for _, n, _ in RESULTS)
    print("\nINDEPENDENT NUMERICAL VERIFICATION")
    print("=" * (width + 26))
    for ok, name, obs in RESULTS:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<{width}}  {obs}")
    bad = sum(1 for ok, _, _ in RESULTS if not ok)
    print("=" * (width + 26))
    print(f"  {len(RESULTS) - bad}/{len(RESULTS)} checks passed")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
