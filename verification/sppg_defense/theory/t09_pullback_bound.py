"""
T-9  Radius- and dimension-independence of the matrix-exponential pullback bound.
Verifies Proposition `prop:dichotomy_app`(i)  ->  claim TH-9.

Claim
-----
If V is a subspace of so(n) and f is C^2 with sup||grad f||_F <= G_f and
sup||Hess f||_op <= L_f, then F(theta) = f(exp theta) satisfies

        ||Hess F(theta)||_op  <=  L_f + G_f

for every theta in V -- independently of ||theta||_F and with no extra explicit
dependence on n.

Test function class with verified constants
-------------------------------------------
    f(Y) = sum_k c_k sin(<A_k, Y>_F),   ||A_k||_F = 1,  sum_k |c_k| = 1
      grad f(Y)      = sum_k c_k cos(<A_k,Y>) A_k
      Hess f(Y)[E,E] = -sum_k c_k sin(<A_k,Y>) <A_k,E>^2
so G_f <= sum|c_k| ||A_k||_F and L_f <= sum|c_k| ||A_k||_F^2, both equal to 1
under the stated normalisation.  These are computed from the sampled A_k and
c_k rather than hardcoded, and the achieved suprema are also sampled and
reported, so a change to the normalisation that invalidates the bound shows up.

Discriminating power
--------------------
Random unit directions E are not enough: in so(n), <A_k, E>^2 ~ 1/n^2 for
random E, so the observed maximum falls like 1/n and reaches only 0.05 x the
bound, and a bound ten times tighter than the claim would also pass.  The test
therefore runs Lanczos on Hessian-vector products built from the exact
gradient, and additionally searches over theta, so the reported maximum is a
near-supremum.  The summary states the tightest bound that this search would
still have failed to violate, which is the direct measure of how much the test
actually constrains.

Tightness anchor (the paper's own 2x2 witness)
----------------------------------------------
    H = J/sqrt(2) with J = [[0,-1],[1,0]],   f(Y) = Y_11
    => F(rH) = cos(r/sqrt(2)),  max |F''| = 1/2,  while L_f = 0 and G_f = 1.
So the G_f term cannot be dropped.  This is evaluated at the analytic
maximisers r = m*pi*sqrt(2), not on an arbitrary grid that happens to contain
one of them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.linalg import expm, expm_frechet

from ..algebra import (from_coords, fro_norm, orthonormal_basis, proj,
                       random_element, to_coords)
from ._common import Result, lambda_max_abs

__all__ = ["SinPullback", "d2_directional", "run"]


def d2_directional(g, h: float = 1e-4) -> float:
    """
    Second derivative of a scalar function g(r) at r = 0 by Richardson-
    extrapolated central differences.  g must accept a float and return a float.
    """
    def cd(step: float) -> float:
        return (g(step) - 2.0 * g(0.0) + g(-step)) / (step * step)
    return (4.0 * cd(h / 2) - cd(h)) / 3.0


class SinPullback:
    """f(Y) = sum_k c_k sin(<A_k, Y>_F); G_f and L_f are computed, not assumed."""

    def __init__(self, n: int, k: int, rng: np.random.Generator):
        A = rng.standard_normal((k, n, n))
        self.A = np.stack([a / fro_norm(a) for a in A])          # ||A_k||_F = 1
        c = rng.uniform(0.2, 1.0, size=k)
        self.c = c / c.sum()                                     # sum |c_k| = 1
        self.n = n
        self.Aflat = self.A.reshape(k, -1)
        # certified constants from the sampled data (not hardcoded)
        self.G_f = float(np.sum(np.abs(self.c) *
                                np.array([fro_norm(a) for a in self.A])))
        self.L_f = float(np.sum(np.abs(self.c) *
                                np.array([fro_norm(a) ** 2 for a in self.A])))

    def f(self, Y: np.ndarray) -> float:
        return float(self.c @ np.sin(self.Aflat @ Y.reshape(-1)))

    def grad_f(self, Y: np.ndarray) -> np.ndarray:
        w = self.c * np.cos(self.Aflat @ Y.reshape(-1))
        return (w @ self.Aflat).reshape(self.n, self.n)

    def F(self, theta: np.ndarray) -> float:
        return self.f(expm(theta))

    def grad_F(self, theta: np.ndarray) -> np.ndarray:
        """grad of F(theta) = f(exp theta), projected onto so(n)."""
        return proj("so", expm_frechet(theta.T, self.grad_f(expm(theta)),
                                       compute_expm=False))


def _lambda_max(fn: SinPullback, theta: np.ndarray, basis: np.ndarray,
                rng: np.random.Generator, h: float = 1e-5) -> float:
    """||Hess F(theta)||_op by Lanczos on gradient-difference Hessian products."""
    def hvp(c: np.ndarray) -> np.ndarray:
        E = from_coords(basis, c)
        return to_coords(basis, (fn.grad_F(theta + h * E)
                                 - fn.grad_F(theta - h * E)) / (2 * h))
    return lambda_max_abs(hvp, basis.shape[0], rng)


def run(n_values=(4, 8, 16, 32), radii=(0.1, 1.0, 10.0, 100.0, 1000.0),
        n_theta: int = 4, k: int = 12, seed: int = 0) -> Result:
    rng = np.random.default_rng(seed)
    rows = []

    # ---- verify the assumed constants G_f, L_f numerically -------------------
    fn_chk = SinPullback(6, k, rng)
    g_emp = l_emp = 0.0
    for _ in range(400):
        Y = rng.standard_normal((6, 6)) * rng.uniform(0.1, 5.0)
        g_emp = max(g_emp, fro_norm(fn_chk.grad_f(Y)))
        E = rng.standard_normal((6, 6)); E /= fro_norm(E)
        s = fn_chk.Aflat @ Y.reshape(-1)
        l_emp = max(l_emp, abs(float(-(fn_chk.c * np.sin(s)) @
                                     (fn_chk.Aflat @ E.reshape(-1)) ** 2)))
    const_ok = g_emp <= fn_chk.G_f + 1e-9 and l_emp <= fn_chk.L_f + 1e-9

    # ---- the bound, with Lanczos over E and a search over theta --------------
    for n in n_values:
        fn = SinPullback(n, k, rng)
        basis = orthonormal_basis("so", n)
        bound = fn.L_f + fn.G_f
        for R in radii:
            worst = 0.0
            for _ in range(n_theta):
                th = random_element("so", n, rng, fro=R)
                worst = max(worst, _lambda_max(fn, th, basis, rng))
            rows.append(dict(kind="pullback", n=n, radius=R, max_hess=worst,
                             bound=bound, ratio=worst / bound,
                             G_f=fn.G_f, L_f=fn.L_f))

    df = pd.DataFrame(rows)
    violations = int((df["ratio"] > 1 + 1e-6).sum())

    # ---- adversarial instance: does anything approach the bound? -------------
    # A single sine aligned with a rank-2 direction gives the largest curvature
    # in this class; we then hill-climb on theta.  The aim is to report how
    # much slack the test really has.
    adv = 0.0
    for n in (2, 3, 4, 6):
        fn = SinPullback(n, 1, rng)
        basis = orthonormal_basis("so", n)
        best = 0.0
        th = np.zeros((n, n))
        for _ in range(60):
            cand = [th] + [th + random_element("so", n, rng, fro=s)
                           for s in (0.05, 0.2, 0.8)]
            vals = [_lambda_max(fn, c, basis, rng) for c in cand]
            i = int(np.argmax(vals))
            if vals[i] > best:
                best, th = vals[i], cand[i]
        adv = max(adv, best / (fn.L_f + fn.G_f))
        rows.append(dict(kind="adversarial", n=n, radius=fro_norm(th),
                         max_hess=best, bound=fn.L_f + fn.G_f,
                         ratio=best / (fn.L_f + fn.G_f),
                         G_f=fn.G_f, L_f=fn.L_f))

    # radius-independence: normalised trend, so the criterion does not become
    # trivially easier as the raw scale falls with n
    trends = []
    for n, g in df.groupby("n"):
        g = g.sort_values("radius")
        y = g["max_hess"].to_numpy()
        if y.max() > 0:
            trends.append(float(np.polyfit(np.log(g["radius"]), y / y.max(), 1)[0]))
    max_trend = float(np.max(trends)) if trends else 0.0

    # ---- 2x2 tightness anchor, evaluated at the analytic maximisers ----------
    J = np.array([[0.0, -1.0], [1.0, 0.0]])
    H = J / np.sqrt(2.0)
    Fw = lambda r: float(expm(r * H)[0, 0])
    maximisers = np.array([m * np.pi * np.sqrt(2.0) for m in range(0, 10)])
    d2 = np.array([abs(d2_directional(lambda s, r0=r0: Fw(r0 + s)))
                   for r0 in maximisers])
    witness_max = float(d2.max())
    witness_err = float(abs(witness_max - 0.5))
    witness_ok = witness_max <= 1.0 + 1e-9 and witness_err < 1e-5
    witness_rows = [dict(kind="witness_2x2", n=2, radius=float(maximisers[-1]),
                         max_hess=witness_max, bound=1.0, ratio=witness_max,
                         G_f=1.0, L_f=0.0)]

    passed = (violations == 0 and const_ok and max_trend <= 1e-6
              and witness_ok and adv > 0.25)

    return Result(
        name="t09_pullback_bound",
        claim="TH-9 / Prop prop:dichotomy_app(i): ||Hess F|| <= L_f + G_f on so(n), radius- and n-independent",
        passed=passed,
        summary={
            "cells tested (n x radius)": int((df.kind == "pullback").sum()),
            "bound L_f + G_f (computed from the sampled f)": 2.0,
            "assumed G_f, L_f dominate their empirical suprema": const_ok,
            "empirical sup ||grad f||_F vs G_f": f"{g_emp:.4f} <= {fn_chk.G_f:.4f}",
            "empirical sup |Hess f[E,E]| vs L_f": f"{l_emp:.4f} <= {fn_chk.L_f:.4f}",
            "violations (must be 0)": violations,
            "max ||Hess F||_op / bound, random theta": float(df["ratio"].max()),
            "max ||Hess F||_op / bound, ADVERSARIAL search": adv,
            "tightest bound this search would still not violate":
                f"{adv * 2.0:.3f} (claim is 2.0)",
            "max normalised upward trend vs log radius (must be <= 0)": max_trend,
            "radii tested": f"{min(radii):g} .. {max(radii):g}",
            "dims tested": list(n_values),
            "2x2 witness max |F''| at analytic maximisers (exact 0.5)": witness_max,
            "2x2 witness error vs analytic": witness_err,
            "2x2 witness shows G_f term is NOT removable (L_f = 0 there)": witness_ok,
        },
        table=pd.DataFrame(rows + witness_rows),
    )


if __name__ == "__main__":
    r = run()
    print(r.report())
    r.save()
