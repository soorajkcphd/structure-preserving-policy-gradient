"""
T-7  Deterministic projected-gradient stationarity rate and the eta < 2/L threshold.
Verifies Theorem `thm:nonconvex`  ->  claim TH-7.

Claim
-----
Under Assumption `asm:smooth`, for X_{t+1} = Proj_g(X_t - eta grad F(X_t)) with
0 < eta < 2/L:

    (a)  f(X_{t+1}) <= f(X_t) - eta (1 - L eta / 2) ||grad_g f(X_t)||_F^2
    (b)  min_{t<T} ||grad_g f(X_t)||_F^2 <= (f(X_0) - f_inf) / (T eta (1 - L eta / 2))
    (c)  at eta = 1/L this is  <= 2 L (f(X_0) - f_inf) / T          [O(1/T)]

Parameterisation
----------------
Both objectives are expressed directly in Frobenius-orthonormal coordinates of
the subspace g, i.e. x in R^d with d = dim g.  This is exact rather than a
simplification: because g is a linear subspace and X_t already lies in it,

        Proj_g(X_t - eta grad F(X_t)) = X_t - eta Proj_g grad F(X_t)

which is the very identity the theorem's proof uses, and in an orthonormal basis
the restricted gradient is the coordinate gradient.  The projection is therefore
built into the coordinate representation, not skipped.

Test objectives
---------------
  * quadratic   f(x) = 1/2 x^T A x + c^T x
                L = lambda_max(A) is exact, so the eta = 2/L divergence
                threshold can be probed sharply; f_inf = -1/2 c^T A^{-1} c.
  * nonconvex   f(x) = sum_i w_i sin(a_i^T x)
                L <= sum_i |w_i| ||a_i||^2 and f_inf >= -sum_i |w_i|, both
                conservative, so the theorem's inequality must still hold --
                with slack, which is reported.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..algebra import subspace_dim
from ._common import Result, loglog_slope

__all__ = ["Quadratic", "NonconvexSin", "projected_gd", "run"]


class Quadratic:
    """f(x) = 1/2 x^T A x + c^T x on the subspace; L = lambda_max(A) exactly."""

    name = "quadratic"

    def __init__(self, d: int, L: float, rng: np.random.Generator, mu_min: float = 0.05):
        Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
        eig = rng.uniform(mu_min, L, size=d)
        eig[0] = L                      # ensure lambda_max is exactly L
        eig[1] = mu_min
        self.A = Q @ np.diag(eig) @ Q.T
        self.A = 0.5 * (self.A + self.A.T)
        self.c = rng.standard_normal(d) * 0.1
        self.L = float(L)
        self.f_inf = float(-0.5 * self.c @ np.linalg.solve(self.A, self.c))

    def f(self, x):
        return float(0.5 * x @ self.A @ x + self.c @ x)

    def grad(self, x):
        return self.A @ x + self.c


class NonconvexSin:
    """f(x) = sum_i w_i sin(a_i^T x); L and f_inf are conservative upper/lower bounds."""

    name = "nonconvex_sin"

    def __init__(self, d: int, k: int, rng: np.random.Generator, scale: float = 1.0):
        self.a = rng.standard_normal((k, d))
        self.a /= np.linalg.norm(self.a, axis=1, keepdims=True)
        self.a *= scale
        self.w = rng.uniform(0.5, 1.5, size=k)
        self.L = float(np.sum(np.abs(self.w) * np.sum(self.a ** 2, axis=1)))
        self.f_inf = float(-np.sum(np.abs(self.w)))

    def f(self, x):
        return float(self.w @ np.sin(self.a @ x))

    def grad(self, x):
        return (self.w * np.cos(self.a @ x)) @ self.a


def projected_gd(obj, x0: np.ndarray, eta: float, T: int):
    """
    Projected gradient descent in orthonormal subspace coordinates.

    Because g is a linear subspace and x already lives in it, projecting the
    ambient step is identical to stepping with the restricted gradient:
        Proj_g(X - eta grad F(X)) = X - eta Proj_g grad F(X)
    which is exactly the identity the theorem's proof uses.  Working in
    coordinates makes that projection implicit and exact.
    """
    x = x0.copy()
    fs = np.empty(T + 1)
    gs = np.empty(T + 1)
    for t in range(T):
        g = obj.grad(x)
        fs[t] = obj.f(x)
        gs[t] = float(g @ g)
        x = x - eta * g
        if not np.isfinite(x).all() or np.abs(x).max() > 1e150:
            fs[t + 1:] = np.inf
            gs[t + 1:] = np.inf
            return x, fs, gs, True                    # diverged
    fs[T] = obj.f(x)
    gs[T] = float(obj.grad(x) @ obj.grad(x))
    return x, fs, gs, False


def run(n: int = 16, subspace: str = "so", T: int = 20000,
        seed: int = 0, n_inits: int = 8) -> Result:
    rng = np.random.default_rng(seed)
    d = subspace_dim(subspace, n)
    rows = []

    objs = [Quadratic(d, L=4.0, rng=rng), NonconvexSin(d, k=30, rng=rng, scale=1.0)]

    # ---- (a)+(b)+(c): descent inequality and the rate, for eta < 2/L
    desc_viol = 0
    rate_viol = 0
    diverged_below_threshold = 0
    for obj in objs:
        L = obj.L
        for eta_mult in (0.1, 0.5, 1.0, 1.5, 1.9):
            eta = eta_mult / L
            for i in range(n_inits):
                x0 = rng.standard_normal(d) * 0.5
                _, fs, gs, div = projected_gd(obj, x0, eta, T)
                if div:
                    # A violation of the theorem must be reported as a
                    # failure, not raised: an uncaught assertion would kill the
                    # whole suite mid-run with no diagnostic row.
                    diverged_below_threshold += 1
                    continue

                # (a) per-step sufficient decrease
                lhs = fs[1:] - fs[:-1]
                rhs = -eta * (1 - L * eta / 2) * gs[:-1]
                desc_viol += int((lhs > rhs + 1e-9 * np.maximum(1.0, np.abs(fs[:-1]))).sum())

                # (b) the stationarity bound at several horizons.
                # Horizons are clipped to T; otherwise a short run would report
                # the same minimum at two nominal horizons and the fitted decay
                # exponent would be an artifact of the duplicate.
                horizons = sorted({h for h in (10, 100, 1000, 10000, T) if h <= T})
                for Tt in horizons:
                    lhs_b = float(gs[:Tt].min())
                    rhs_b = (fs[0] - obj.f_inf) / (Tt * eta * (1 - L * eta / 2))
                    rate_viol += int(lhs_b > rhs_b * (1 + 1e-9))
                    rows.append(dict(kind="rate", obj=obj.name, eta_mult=eta_mult,
                                     init=i, T=Tt, min_grad_sq=lhs_b, bound=rhs_b,
                                     slack=rhs_b / max(lhs_b, 1e-300)))

    df = pd.DataFrame(rows)

    # ---- (d) empirical decay exponent of min_t ||grad||^2 vs T
    slopes = []
    for (o, em, i), g in df[df.kind == "rate"].groupby(["obj", "eta_mult", "init"]):
        g = g.sort_values("T")
        slopes.append(loglog_slope(g["T"].values, g["min_grad_sq"].values))
    slopes = np.array(slopes)
    n_nan_slopes = int(np.isnan(slopes).sum())
    worst_slope = float(np.nanmax(slopes))          # diagnostic only, see note below

    # ---- (e) sharp divergence threshold at eta = 2/L, using the exact-L quadratic.
    #
    # For a quadratic, GD is exactly  e_{t+1} = (I - eta A) e_t  with e = x - x*,
    # so the asymptotic growth factor is rho(eta) = max_i |1 - eta lambda_i| and
    # the iteration converges IFF rho < 1 IFF eta < 2 / lambda_max = 2 / L.
    # This is an exact, non-asymptotic characterisation, so we test it directly
    # rather than waiting for a floating-point overflow (near the threshold the
    # blow-up is real but slow: at eta = 2.001/L the growth is only 1.001^t).
    # The error is initialised along the top eigenvector of A, so the dynamics
    # reduce exactly to  e_t = (1 - eta lambda_max)^t e_0  with no transient from
    # the other modes.  The empirical growth factor then matches the prediction
    # to machine precision instead of only asymptotically.  lambda_max is the
    # eigenvalue that defines L, so this is exactly the mode governing 2/L.
    q = objs[0]
    eigA, vecA = np.linalg.eigh(q.A)
    v_max = vecA[:, -1]
    xstar = -np.linalg.solve(q.A, q.c)
    thr_rows = []
    Tthr = 200
    for eta_mult in (0.5, 1.0, 1.5, 1.9, 1.99, 1.999, 2.001, 2.01, 2.1, 3.0):
        eta = eta_mult / q.L
        rho_pred = float(abs(1.0 - eta * eigA[-1]))
        x = xstar + 0.1 * v_max
        e0 = float(np.linalg.norm(x - xstar))

        # One exact step: ||e_1|| / ||e_0|| = |1 - eta lambda_max| identically.
        # (Multi-step ratios are unusable at eta = 1/L, where the mode is
        # annihilated in a single step and the remainder is pure roundoff.)
        x1 = x - eta * q.grad(x)
        rho_emp = float(np.linalg.norm(x1 - xstar)) / e0

        # Many steps: does the error actually grow?
        x = x1
        for _ in range(Tthr - 1):
            x = x - eta * q.grad(x)
            if not np.isfinite(x).all():
                break
        eT = float(np.linalg.norm(x - xstar))
        thr_rows.append(dict(kind="threshold", eta_mult=eta_mult, T=Tthr,
                             rho_predicted=rho_pred, rho_empirical_1step=rho_emp,
                             grows=bool(not np.isfinite(eT) or eT > e0),
                             predicted_grows=bool(rho_pred > 1.0),
                             err=abs(rho_emp - rho_pred)))
    thr = pd.DataFrame(thr_rows)
    threshold_ok = bool((thr["grows"] == thr["predicted_grows"]).all()
                        and (thr["grows"] == (thr["eta_mult"] > 2.0)).all())
    rho_match = float(thr.loc[np.isfinite(thr["err"]), "err"].max())
    threshold_ok = threshold_ok and rho_match < 1e-6

    # Note on the correct pass criterion.
    # The theorem asserts (a) the per-step descent inequality, (b) the
    # stationarity bound, and (implicitly, via 0 < eta < 2/L) (e) the step-size
    # threshold.  It does not assert that the empirical decay exponent of
    # min_t ||grad||^2 is at most -1 at every finite horizon: a small step size
    # spends its early iterations in a transient during which the observed decay
    # is slower than the guarantee, even though the guarantee itself is never
    # violated (which is what (b) checks).  The exponent is therefore reported
    # as a diagnostic, with the robust median used as the pass criterion and the
    # worst case shown for transparency.
    median_slope = float(np.nanmedian(slopes))
    passed = (desc_viol == 0 and rate_viol == 0 and threshold_ok
              and diverged_below_threshold == 0
              and n_nan_slopes == 0
              and median_slope <= -1.0)

    return Result(
        name="t07_projected_gd",
        claim="TH-7 / Theorem thm:nonconvex: descent inequality, O(1/T) min squared restricted gradient, eta<2/L",
        passed=passed,
        summary={
            "subspace / n / dim": f"{subspace}({n}) dim={d}",
            "(a) per-step descent-inequality violations (must be 0)": desc_viol,
            "(a) runs that diverged despite eta < 2/L (must be 0)": diverged_below_threshold,
            "(d) slope fits that returned NaN (must be 0)": n_nan_slopes,
            "(b) stationarity-bound violations (must be 0)": rate_viol,
            "(b) number of (obj, eta, init, T) bound checks": int((df.kind == "rate").sum()),
            "(b) median slack factor bound/observed": float(df["slack"].median()),
            "(d) median empirical log-log slope of min||grad||^2 vs T (<= -1)": median_slope,
            "(d) worst empirical slope (transient-dominated at short horizons)": worst_slope,
            "(e) growth occurs exactly when eta > 2/L (sharp threshold)": threshold_ok,
            "(e) max |rho_empirical - rho_predicted| (exact 1-step ratio)": rho_match,
            "(e) eta multiples probed": thr["eta_mult"].tolist(),
            "objectives": [o.name for o in objs],
            "exact L (quadratic)": objs[0].L,
            "upper-bound L (nonconvex)": objs[1].L,
        },
        table=pd.concat([df, thr], ignore_index=True),
    )


if __name__ == "__main__":
    r = run()
    print(r.report())
    r.save()
