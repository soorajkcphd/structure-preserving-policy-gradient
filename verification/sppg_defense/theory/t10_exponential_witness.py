"""
T-10  The exponential curvature lower bound, and the nilpotent counterexample.
Verifies Proposition `prop:dichotomy_app`(ii) and the nilpotent remark
   ->  claims TH-10, TH-15.

Claim (ii)
----------
If the parameter subspace contains a direction H with ||H||_F = 1 and a real
eigenpair H u = mu u, mu > 0, ||u||_2 = 1, then there is a smooth f with
G_f <= 1 and L_f <= 1 such that F(theta) = f(exp theta) has, on the segment
{ r H : 0 <= r <= R },

        L_F  =  Omega( e^{2 mu R} ),   explicitly   L_F >= (mu^2 / 4) e^{2 mu R}
        whenever R >= log(4 pi) / mu.

The witness is fully constructive:

        f(Y) = sin(u^T Y u)
        u^T e^{rH} u = e^{mu r}                (u is an eigenvector of H)
        F(rH) = sin(e^{mu r})
        d^2/dr^2 F(rH) = mu^2 e^{mu r} cos(e^{mu r}) - mu^2 e^{2 mu r} sin(e^{mu r})

At  y_k := pi/2 + 2 pi k  (so sin = 1, cos = 0) with r_k = log(y_k)/mu, the
second derivative has magnitude exactly mu^2 y_k^2.

Claim (nilpotent remark)
------------------------
N = [[0,1],[0,0]] has rho(exp(tN)) = 1 for all t while ||exp(tN)||_op grows
linearly.  This is why part (ii) needs the positive-real-eigenpair
hypothesis -- and, for the paper, why spectral radius is the wrong diagnostic
for the geometric channel (cf. Fig. fig:spectral, which reports rho only).

Numerical care
--------------
sin(e^{mu r}) is meaningless in float64 once e^{mu r} exceeds ~1e15 (argument
reduction fails), and e^{2 mu R} overflows float64 past mu*R ~ 354.  Therefore:
  * the analytic second-derivative formula is validated against finite
    differences of the true matrix objective only where e^{mu r} <= 1e3;
  * the lower bound is evaluated with mpmath at precision scaled to mu*R
    (about mu*R/log(10) + 40 digits).  A fixed precision is not enough: at
    60 dps and mu*R = 759 the floor() selecting k is meaningless, sin(y_k) is
    not 1, and the ratio collapses to exactly 4.0, making the check vacuous.
    sin(y_k) = 1 is therefore verified explicitly on every row;
  * results are reported as log10(L_F) so no row becomes inf.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.linalg import expm

from ..algebra import fro_norm, random_element
from ._common import Result, loglog_slope

try:
    import mpmath as mp
    HAVE_MP = True
except ImportError:                                    # pragma: no cover
    HAVE_MP = False

__all__ = ["make_direction", "d2F_analytic", "L_F_lower", "run"]


def make_direction(n: int, mu: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Build H with ||H||_F = 1 and H u = mu u for u = e_1, 0 < mu <= 1.

    H = diag(mu, -sqrt(1 - mu^2), 0, ..., 0):  the second entry absorbs the
    remaining Frobenius mass without touching the e_1 eigenvector.
    """
    if not 0 < mu <= 1:
        raise ValueError("mu must lie in (0, 1] for a unit-Frobenius direction")
    if n < 2 and mu < 1:
        raise ValueError("n >= 2 required unless mu == 1")
    H = np.zeros((n, n))
    H[0, 0] = mu
    if n >= 2:
        H[1, 1] = -np.sqrt(max(0.0, 1.0 - mu * mu))
    u = np.zeros(n)
    u[0] = 1.0
    return H, u


def d2F_analytic(r: float, mu: float) -> float:
    """d^2/dr^2 sin(e^{mu r}) = mu^2 e^{mu r} cos(e^{mu r}) - mu^2 e^{2 mu r} sin(e^{mu r})."""
    y = np.exp(mu * r)
    return mu * mu * y * np.cos(y) - mu * mu * y * y * np.sin(y)


def L_F_lower(R: float, mu: float) -> dict:
    """
    Exact lower bound on L_F over [0, R], evaluated at the largest
    y_k = pi/2 + 2 pi k with y_k <= e^{mu R}.

    Returns log10 of the value and of the theoretical floor (so no field can
    overflow to inf), the ratio, the selected k, and an exactness flag.

    Precision.  mpmath is used at about mu*R/log(10) + 40 digits.  A fixed
    precision is not sufficient: at 60 dps and mu*R = 759 the floor() selecting
    k is meaningless, y_k is not of the form pi/2 + 2 pi k, sin(y_k) != 1, and
    the ratio collapses to exactly 4.0 -- which would make the check vacuous.
    The chosen k is therefore verified exactly at the working precision.
    """
    if not HAVE_MP:                                     # pragma: no cover
        raise RuntimeError("mpmath is required for T-10; pip install mpmath")

    dps = int(mu * R / np.log(10)) + 40
    with mp.workdps(dps):
        mu_ = mp.mpf(mu)
        emuR = mp.exp(mu_ * mp.mpf(R))
        nan = dict(k_log10=float("nan"), sin_yk_err=float("nan"), exact=False,
                   log10_L_F=float("nan"), log10_floor=float("nan"),
                   ratio=float("nan"))
        if emuR < mp.pi / 2:
            return nan

        k = int(mp.floor((emuR - mp.pi / 2) / (2 * mp.pi)))
        # make k maximal and admissible (mp.floor can land one off at the edge)
        while mp.pi / 2 + 2 * mp.pi * (k + 1) <= emuR:
            k += 1
        while k >= 0 and mp.pi / 2 + 2 * mp.pi * k > emuR:
            k -= 1
        if k < 0:
            return nan

        y_k = mp.pi / 2 + 2 * mp.pi * k
        # The property actually needed is sin(y_k) = 1 (so |d^2F/dr^2| = mu^2 y_k^2
        # exactly).  Testing that directly is the right check; comparing
        # (y_k - pi/2)/(2 pi) against k is dominated by the cancellation of two
        # numbers of size ~e^{mu R} and says nothing useful.
        exact = bool(abs(mp.sin(y_k) - 1) < mp.mpf(10) ** (-20)
                     and y_k <= emuR)
        val = mu_ ** 2 * y_k ** 2                       # |d^2F/dr^2| at r_k
        floor = mu_ ** 2 / 4 * emuR ** 2
        return dict(k_log10=float(mp.log10(k)) if k > 0 else 0.0,
                    sin_yk_err=float(abs(mp.sin(y_k) - 1)),
                    exact=exact,
                    log10_L_F=float(mp.log10(val)),
                    log10_floor=float(mp.log10(floor)),
                    ratio=float(val / floor))


def _check_f_constants(n: int = 8, seed: int = 0) -> dict:
    """
    Numerical check that f(Y) = sin(u^T Y u) has G_f <= 1 and L_f <= 1,
    by finite-differencing f itself -- not by re-evaluating the analytic
    formulas, which would be circular.

    The supremum of L_f is attained at E = u u^T, so that direction is included
    explicitly; sampling only random unit-Frobenius E in R^{n x n} gives
    (u^T E u)^2 ~ 1/n^2 and under-reports L_f by a factor of n^2 (at n = 8 the
    old version returned 0.119 against a true supremum of 1).
    """
    rng = np.random.default_rng(seed)
    u = np.zeros(n); u[0] = 1.0
    f = lambda Y: float(np.sin(u @ Y @ u))

    dirs = [np.outer(u, u)]                              # the extremal direction
    for _ in range(20):
        E = rng.standard_normal((n, n))
        dirs.append(E / fro_norm(E))

    gmax = lmax = 0.0
    h = 1e-5
    basis = [(i, j) for i in range(n) for j in range(n)]
    for _ in range(40):
        Y = rng.standard_normal((n, n)) * rng.uniform(0.1, 5.0)
        g = np.zeros((n, n))
        for (i, j) in basis:
            E = np.zeros((n, n)); E[i, j] = 1.0
            g[i, j] = (f(Y + h * E) - f(Y - h * E)) / (2 * h)
        gmax = max(gmax, fro_norm(g))
        for E in dirs:
            d2 = (f(Y + h * E) - 2 * f(Y) + f(Y - h * E)) / (h * h)
            lmax = max(lmax, abs(d2) / fro_norm(E) ** 2)
    return dict(G_f_empirical=gmax, L_f_empirical=lmax)


def run(mus=(0.1, 0.25, 0.5, 1.0),
        n_values=(2, 4, 8, 16, 32),
        R_multiples=(1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 40.0, 120.0, 300.0),
        seed: int = 0) -> Result:
    rows = []

    # ---- Part A: validate the analytic formula against the true matrix objective
    val_err = 0.0
    for mu in mus:
        H, u = make_direction(8, mu)
        # keep e^{mu r} <= 1e3 so that sin(e^{mu r}) is still meaningful in
        # float64 (argument reduction fails well before 1e15)
        for r in np.linspace(0.0, np.log(1e3) / mu, 40):
            F = lambda rr: np.sin(float(u @ expm(rr * H) @ u))
            h = 1e-5
            fd = (F(r + h) - 2 * F(r) + F(r - h)) / (h * h)
            val_err = max(val_err, abs(fd - d2F_analytic(r, mu)) /
                          max(1.0, abs(d2F_analytic(r, mu))))

    # also confirm the eigen-identity u^T e^{rH} u = e^{mu r} exactly
    eig_err = 0.0
    for mu in mus:
        for n in n_values:
            H, u = make_direction(n, mu)
            for r in (0.5, 2.0, 5.0):
                eig_err = max(eig_err, abs(float(u @ expm(r * H) @ u) - np.exp(mu * r)))

    # ---- Part B: the lower bound itself
    for mu in mus:
        R0 = np.log(4 * np.pi) / mu
        for m in R_multiples:
            R = m * R0
            d = L_F_lower(R, mu)
            rows.append(dict(kind="lower_bound", mu=mu, R0=R0, R=R, muR=mu * R,
                             k_log10=d["k_log10"], sin_yk_err=d["sin_yk_err"],
                             k_exact=d["exact"], log10_L_F=d["log10_L_F"],
                             log10_floor=d["log10_floor"], ratio=d["ratio"],
                             satisfied=bool(d["ratio"] >= 1.0)))

    dfb = pd.DataFrame(rows)
    bound_ok = bool(dfb["satisfied"].all())
    exact_ok = bool(dfb["k_exact"].all())
    min_ratio = float(dfb["ratio"].min())

    # ---- exponential growth rate, computed from L_F_lower.
    # Note 1.  L_F(R) = mu^2 y_k^2 with y_k the largest pi/2 + 2 pi k below
    # e^{mu R}, so as a function of R it is a staircase tracking e^{2 mu R}.
    # The sharp envelope 1 <= L_F/floor <= 4 is checked separately below.
    # Note 2.  Regressing log(mu^2 e^{2 mu Rk}) on Rk would be an algebraic
    # identity, returning 2*mu whether or not L_F_lower is correct, so the
    # regression uses L_F_lower's own output and a bug in it shows up here.
    slope_rows = []
    for mu in mus:
        R0 = np.log(4 * np.pi) / mu
        Rs = np.linspace(R0, 30 * R0, 60)
        y = np.array([L_F_lower(R, mu)["log10_L_F"] for R in Rs]) * np.log(10)
        good = np.isfinite(y)
        slope = float(np.polyfit(Rs[good], y[good], 1)[0])
        slope_rows.append(dict(kind="growth_rate", mu=mu, fitted_slope=slope,
                               expected=2 * mu,
                               rel_err=abs(slope - 2 * mu) / (2 * mu)))
    dfs = pd.DataFrame(slope_rows)
    max_slope_err = float(dfs["rel_err"].max())

    # (a) sharp envelope over a dense, arbitrary R grid
    env_lo, env_hi = np.inf, -np.inf
    for mu in mus:
        R0 = np.log(4 * np.pi) / mu
        for R in np.linspace(R0, 40 * R0, 400):
            ratio = L_F_lower(R, mu)["ratio"]
            env_lo = min(env_lo, ratio)
            env_hi = max(env_hi, ratio)
    envelope_ok = bool(env_lo >= 1.0 - 1e-12 and env_hi <= 4.0 + 1e-12)

    # ---- Part C: contrast on so(n) -- the same f gives a bounded Hessian
    rng = np.random.default_rng(seed)
    so_rows = []
    for n in (4, 8, 16, 32):
        u = np.zeros(n); u[0] = 1.0
        for R in (1.0, 10.0, 100.0, 1000.0):
            worst = 0.0
            for _ in range(20):
                Hs = random_element("so", n, rng, fro=1.0)
                h = 1e-4
                F = lambda rr: np.sin(float(u @ expm(rr * Hs) @ u))
                r = rng.uniform(0, R)
                worst = max(worst, abs((F(r + h) - 2 * F(r) + F(r - h)) / (h * h)))
            so_rows.append(dict(kind="so_contrast", n=n, R=R, max_abs_d2F=worst,
                                bound_Lf_plus_Gf=2.0))
    dfso = pd.DataFrame(so_rows)
    so_ok = bool((dfso["max_abs_d2F"] <= 2.0 + 1e-6).all())

    # ---- Part D: nilpotent counterexample (TH-15)
    N = np.array([[0.0, 1.0], [0.0, 0.0]])
    nil_rows = []
    for t in (1.0, 10.0, 100.0, 1000.0, 1e4, 1e5, 1e6):
        M = expm(t * N)                       # = I + tN exactly, since N^2 = 0
        rho = float(np.max(np.abs(np.linalg.eigvals(M))))
        opn = float(np.linalg.norm(M, 2))
        nil_rows.append(dict(kind="nilpotent", t=t, rho=rho, op_norm=opn,
                             op_norm_over_t=opn / t,
                             expm_equals_I_plus_tN=float(fro_norm(M - (np.eye(2) + t * N)))))
    dfn = pd.DataFrame(nil_rows)
    nil_rho_ok = bool(np.allclose(dfn["rho"], 1.0, atol=1e-9))
    # slope fitted on the asymptotic regime only; ||exp(tN)||_op / t -> 1
    big = dfn[dfn["t"] >= 100]
    nil_slope = loglog_slope(big["t"].values, big["op_norm"].values)
    nil_ratio = float(dfn["op_norm_over_t"].iloc[-1])
    nil_ok = nil_rho_ok and abs(nil_slope - 1.0) < 1e-3 and abs(nil_ratio - 1.0) < 1e-3

    fc = _check_f_constants(8, seed)

    # Note on the slope tolerance.  L_F(R) is a staircase (piecewise constant
    # between the jumps at r_k), so a least-squares fit of log L_F on an
    # arbitrary R grid recovers 2*mu only up to the quantisation -- observed
    # ~1e-3 relative.  A 1e-9 tolerance is only achievable when the quantity
    # being fitted is an algebraic identity rather than the output of
    # L_F_lower; 1e-2 is a non-trivial gate here.
    passed = (bound_ok and exact_ok and envelope_ok and max_slope_err < 1e-2
              and val_err < 1e-3 and eig_err < 1e-9 and so_ok and nil_ok
              and fc["G_f_empirical"] <= 1 + 1e-6
              and fc["L_f_empirical"] <= 1 + 1e-6
              # the extremal direction E = u u^T must actually be probed, or the
              # constant is under-reported by a factor of n^2
              and fc["L_f_empirical"] > 0.9)

    return Result(
        name="t10_exponential_witness",
        claim="TH-10 / Prop prop:dichotomy_app(ii): L_F >= (mu^2/4) e^{2 mu R}; and TH-15 nilpotent contrast",
        passed=passed,
        summary={
            "witness f: empirical sup ||grad f||_F by finite differences (<= 1)":
                fc["G_f_empirical"],
            "witness f: empirical sup |Hess f[E,E]| for unit E (<= 1, attained)":
                fc["L_f_empirical"],
            "identity u^T e^{rH} u = e^{mu r}, max abs error": eig_err,
            "analytic d2F/dr2 vs finite differences, max rel error": val_err,
            "lower bound L_F >= (mu^2/4)e^{2muR} satisfied at every (mu,R)": bound_ok,
            "selected k verified EXACT (|sin(y_k)-1| < 1e-20) at every (mu,R)": exact_ok,
            "max |sin(y_k) - 1| over all rows": float(dfb["sin_yk_err"].max()),
            "min ratio L_F / floor (must be >= 1)": min_ratio,
            "sharp envelope 1 <= L_F/floor <= 4 over dense R grid": envelope_ok,
            "envelope observed range": (float(env_lo), float(env_hi)),
            "growth rate d log L_F/dR from L_F_lower vs 2*mu, max rel error": max_slope_err,
            "mu values tested": list(mus),
            "largest mu*R reached": float(dfb["muR"].max()),
            "largest log10(L_F) reached": float(dfb["log10_L_F"].max()),
            "largest log10(k) (number of digits in the selected index)":
                float(dfb["k_log10"].max()),
            "CONTRAST so(n): max |d2F| <= L_f + G_f = 2 at all radii up to 1000": so_ok,
            "CONTRAST so(n): observed max |d2F|": float(dfso["max_abs_d2F"].max()),
            "NILPOTENT rho(exp(tN)) == 1 for all t": nil_rho_ok,
            "NILPOTENT log-log slope of ||exp(tN)||_op, t>=100 (linear => 1)": nil_slope,
            "NILPOTENT ||exp(tN)||_op / t at t=1e6 (-> 1)": nil_ratio,
            "mpmath available (needed for very large mu*R)": HAVE_MP,
        },
        table=pd.concat([dfb, dfs, dfso, dfn], ignore_index=True),
    )


if __name__ == "__main__":
    r = run()
    print(r.report())
    r.save()
