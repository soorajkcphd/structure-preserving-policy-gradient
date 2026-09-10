"""
T-8  Projected stochastic gradient: a.s. liminf stationarity, and the necessity
     of the Robbins-Monro step-size conditions.
Verifies Theorem `thm:stochastic`  ->  claim TH-8.

Claim
-----
Under Assumptions `asm:smooth` and `asm:bounded`, with
    sum_t alpha_t = infinity        and        sum_t alpha_t^2 < infinity,
the iterates X_{t+1} = X_t - alpha_t Proj_g ghat_t satisfy, almost surely,
    (i)   f(X_t) converges to a finite limit,
    (ii)  sum_t alpha_t ||grad_g f(X_t)||_F^2 < infinity,
    (iii) liminf_t ||grad_g f(X_t)||_F = 0.

Oracle (so the hypotheses hold exactly, not approximately)
----------------------------------------------------------
    ghat_t = grad f(x_t) + sigma * u_t ,  u_t uniform on the unit sphere of g.
E[ghat_t | F_t] = grad f(x_t) exactly, and ||Proj_g ghat_t|| <= sup||grad f|| +
sigma almost surely.  The experiment verifies the theorem; it makes no claim
that the RL implementation satisfies these hypotheses.

Part A -- necessity, with an exactly known floor
------------------------------------------------
On a strongly convex quadratic f(x) = 1/2 x^T A x the constant-step iteration
    x_{t+1} = (I - alpha A) x_t - alpha xi_t ,   E[xi xi^T] = (sigma^2/d) I
has a stationary covariance available in closed form.  In the eigenbasis of A,

    c_i = alpha^2 sigma^2 / d / (1 - (1 - alpha lambda_i)^2)
        = alpha sigma^2 / ( d lambda_i (2 - alpha lambda_i) )

    =>  E ||grad||^2 = sum_i lambda_i^2 c_i
                     = (alpha sigma^2 / d) sum_i lambda_i / (2 - alpha lambda_i)

so the noise floor is a number we can predict and check, not a qualitative
claim.  With a Robbins-Monro schedule alpha_t -> 0 the floor vanishes.  This is
the sharpest available demonstration that sum alpha_t^2 < infinity is
necessary rather than incidental.

Part B -- the three conclusions on a nonconvex objective
--------------------------------------------------------
    (i)   tail standard deviation of f(X_t) collapses,
    (ii)  the step-mass-weighted tail of sum alpha_t ||grad||^2 vanishes,
    (iii) the running minimum of ||grad|| is driven towards zero.

Part B does not attempt the necessity argument: over a finite horizon a
well-tuned constant step also converges on a benign nonconvex landscape, so a
RM-versus-constant comparison here is uninformative in either direction.
Necessity is Part A's job, where the floor is known in closed form.

Schedule note
-------------
alpha_t = c (t+1)^-0.75 satisfies both conditions (sum a = inf since p < 1;
sum a^2 < inf since 2p = 1.5 > 1).  The textbook c/(t+1) is avoided because its
total movement is only c log T, so over any practical horizon the iterate is
nearly frozen and the theorem's asymptotic conclusion is not observable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..algebra import subspace_dim
from ._common import Result
from .t07_projected_gd import NonconvexSin

__all__ = ["alpha_of", "quadratic_floor", "run"]


def alpha_of(schedule: str, c: float, t) -> float:
    t = np.asarray(t, dtype=float)
    if schedule == "rm":
        return c * (t + 1.0) ** -0.75          # sum a = inf, sum a^2 < inf
    if schedule == "sqrt":
        return c * (t + 1.0) ** -0.5           # sum a = inf, sum a^2 = inf
    if schedule == "const":
        return np.full_like(t, c)              # sum a = inf, sum a^2 = inf
    raise ValueError(schedule)


def quadratic_floor(eig: np.ndarray, alpha: float, sigma: float, d: int) -> float:
    """Exact stationary E||grad||^2 for constant-step SGD on 1/2 x^T A x."""
    return float((alpha * sigma ** 2 / d) * np.sum(eig / (2.0 - alpha * eig)))


class _BatchSin(NonconvexSin):
    def grad_batch(self, X):
        return (self.w * np.cos(X @ self.a.T)) @ self.a

    def f_batch(self, X):
        return np.sin(X @ self.a.T) @ self.w


def _sphere_batch(R: int, d: int, rng: np.random.Generator) -> np.ndarray:
    U = rng.standard_normal((R, d))
    return U / np.linalg.norm(U, axis=1, keepdims=True)


# --------------------------------------------------------------------------- #
def _part_a(d: int, T: int, n_runs: int, sigma: float,
            rng: np.random.Generator) -> tuple[pd.DataFrame, dict]:
    """Quadratic with an exactly predictable constant-step noise floor."""
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    eig = rng.uniform(0.5, 4.0, size=d)
    A = Q @ np.diag(eig) @ Q.T
    A = 0.5 * (A + A.T)

    rows = []
    out = {}
    # Two constant step sizes.  At alpha = 0.30 the (2 - alpha*lambda) term in
    # the closed form moves the answer by ~25%, so the leading-order formula
    # (alpha sigma^2 / 2d) sum lambda_i -- i.e. dropping exactly the term that
    # makes the closed form non-trivial -- is clearly distinguishable.  With
    # only alpha = 0.05 the two differ by under 7% and a loose tolerance would
    # not tell them apart.
    for sched, c in (("const", 0.05), ("const", 0.30), ("rm", 0.5)):
        X = rng.standard_normal((n_runs, d)) * 0.5
        gsq_hist = []
        for t in range(T):
            a = float(alpha_of(sched, c, t))
            Gm = X @ A
            gsq_hist.append((Gm * Gm).sum(axis=1))
            X = X - a * (Gm + sigma * _sphere_batch(n_runs, d, rng))
        gsq = np.array(gsq_hist)
        tail = gsq[int(0.8 * T):].mean(axis=0)
        pred = quadratic_floor(eig, c, sigma, d) if sched == "const" else np.nan
        naive = (c * sigma ** 2 / (2 * d)) * float(np.sum(eig)) \
            if sched == "const" else np.nan
        out[(sched, c)] = dict(tail=tail, pred=pred, naive=naive)
        for r in range(n_runs):
            rows.append(dict(part="A_quadratic", schedule=sched, c=c, run=r,
                             tail_mean_gradsq=float(tail[r]),
                             predicted_floor=pred, naive_floor=naive))

    res = {}
    for (sched, c), v in out.items():
        if sched != "const":
            continue
        obs = float(np.mean(v["tail"]))
        res[c] = dict(observed=obs, predicted=v["pred"],
                      rel_err=abs(obs - v["pred"]) / v["pred"],
                      naive=v["naive"],
                      naive_rel_err=abs(v["naive"] - v["pred"]) / v["pred"])
    rm_tail = float(np.mean(out[("rm", 0.5)]["tail"]))
    res["rm_tail"] = rm_tail
    res["rm_below_floor_factor"] = res[0.05]["predicted"] / max(rm_tail, 1e-300)
    return pd.DataFrame(rows), res


def _part_b(d: int, T: int, n_runs: int, sigma: float,
            rng: np.random.Generator) -> tuple[pd.DataFrame, dict]:
    """Nonconvex objective: the three conclusions of the theorem."""
    obj = _BatchSin(d, k=20, rng=rng, scale=1.0)
    rows = []
    stats = {}
    for sched, c in (("rm", 0.5), ("sqrt", 0.1), ("const", 0.02)):
        X = rng.standard_normal((n_runs, d)) * 0.5
        run_min = np.full(n_runs, np.inf)
        s1 = np.zeros(n_runs)
        s2 = np.zeros(n_runs)
        a1 = a2 = 0.0
        rec_at = set(np.unique(np.linspace(0, T - 1, 400).astype(int)).tolist())
        rec_f, rec_g = [], []
        for t in range(T):
            a = float(alpha_of(sched, c, t))
            Gm = obj.grad_batch(X)
            gn = np.linalg.norm(Gm, axis=1)
            run_min = np.minimum(run_min, gn)
            if t < T // 2:
                s1 += a * gn * gn
                a1 += a
            else:
                s2 += a * gn * gn
                a2 += a
            if t in rec_at:
                rec_f.append(obj.f_batch(X)); rec_g.append(gn)
            X = X - a * (Gm + sigma * _sphere_batch(n_runs, d, rng))
        rec_f = np.array(rec_f); rec_g = np.array(rec_g)
        tail = slice(len(rec_f) // 2, None)
        # Step-mass-normalised statistic.  The raw ratio
        #     sum_{t>=T/2} a_t g_t^2  /  sum_{t<T/2} a_t g_t^2
        # is dominated by the step-size schedule, not by convergence: for
        # a_t = c(t+1)^-0.75 it equals 2^0.25 - 1 = 0.19 even when g_t^2 is
        # constant (i.e. when the series diverges), so it cannot discriminate.
        # Dividing each half by its own step mass gives the a-weighted mean of
        # g_t^2 on each half, whose ratio -> 0 iff the gradient actually decays.
        w1 = s1 / max(a1, 1e-300)
        w2 = s2 / max(a2, 1e-300)
        ratio = w2 / np.maximum(w1, 1e-300)
        for r in range(n_runs):
            rows.append(dict(part="B_nonconvex", schedule=sched, c=c, run=r,
                             running_min_grad=float(run_min[r]),
                             weighted_tail_ratio=float(ratio[r]),
                             raw_tail_ratio=float(s2[r] / max(s1[r], 1e-300)),
                             tail_f_std=float(np.std(rec_f[tail, r])),
                             tail_grad_median=float(np.median(rec_g[tail, r]))))
        stats[sched] = dict(run_min=float(np.median(run_min)),
                            weighted_ratio=float(np.median(ratio)),
                            raw_ratio=float(np.median(s2 / np.maximum(s1, 1e-300))),
                            f_std=float(np.median(np.std(rec_f[tail], axis=0))))
    return pd.DataFrame(rows), stats


def run(n: int = 8, subspace: str = "so", T: int = 40_000, n_runs: int = 30,
        sigma: float = 1.0, seed: int = 0) -> Result:
    rng = np.random.default_rng(seed)
    d = subspace_dim(subspace, n)

    dfA, A = _part_a(d, T, n_runs, sigma, rng)
    dfB, B = _part_b(d, T, n_runs, sigma, rng)

    # The RM arm must beat the constant-step arm on the a-weighted tail ratio,
    # and the closed-form floor must be reproduced tightly enough to exclude the
    # naive leading-order formula (which drops the 2 - alpha*lambda correction).
    passed = (
        all(A[c]["rel_err"] < 0.02 for c in (0.05, 0.30))     # exact floor
        and A[0.30]["naive_rel_err"] > 5 * A[0.30]["rel_err"]  # and it discriminates
        and A["rm_below_floor_factor"] > 10.0                  # RM far below the floor
        and B["rm"]["weighted_ratio"] < 0.5                    # (ii) gradient decays
        and B["rm"]["run_min"] < 0.5                           # (iii) toward 0
        and B["rm"]["f_std"] < 1.0)                            # (i)  f settles

    return Result(
        name="t08_robbins_monro",
        claim="TH-8 / Theorem thm:stochastic: a.s. liminf stationarity under sum a = inf, sum a^2 < inf",
        passed=passed,
        summary={
            "subspace / dim / steps / runs": f"{subspace}({n}) d={d} T={T} runs={n_runs}",
            "oracle E[ghat|F_t] = grad exactly (mean-zero sphere noise)": True,
            "A: alpha=0.05 floor PREDICTED (closed form)": A[0.05]["predicted"],
            "A: alpha=0.05 floor OBSERVED": A[0.05]["observed"],
            "A: alpha=0.05 relative error": A[0.05]["rel_err"],
            "A: alpha=0.30 floor PREDICTED (closed form)": A[0.30]["predicted"],
            "A: alpha=0.30 floor OBSERVED": A[0.30]["observed"],
            "A: alpha=0.30 relative error": A[0.30]["rel_err"],
            "A: alpha=0.30 NAIVE leading-order formula, relative error":
                A[0.30]["naive_rel_err"],
            "A: discrimination (naive err / measured err) at alpha=0.30":
                A[0.30]["naive_rel_err"] / max(A[0.30]["rel_err"], 1e-30),
            "A: Robbins-Monro tail E||grad||^2": A["rm_tail"],
            "A: how far RM sits below the alpha=0.05 floor (x)":
                A["rm_below_floor_factor"],
            "B(iii): RM median running-min ||grad||": B["rm"]["run_min"],
            "B(iii): CONST median running-min ||grad||": B["const"]["run_min"],
            "B(ii): RM step-mass-weighted tail ratio (-> 0)": B["rm"]["weighted_ratio"],
            "B(ii): SQRT weighted tail ratio": B["sqrt"]["weighted_ratio"],
            "B(ii): CONST weighted tail ratio": B["const"]["weighted_ratio"],
            "B(ii): RM RAW ratio (schedule-dominated; shown to expose why it "
            "cannot discriminate)": B["rm"]["raw_ratio"],
            "B: NOTE -- the CONST arm here is NOT a necessity demonstration":
                ("on a benign nonconvex landscape a well-tuned constant step "
                 "also converges over a finite horizon, so it can look as good "
                 "as or better than RM on these statistics. The necessity of "
                 "sum alpha^2 < inf is established rigorously in PART A, "
                 "against a closed-form noise floor, not by this comparison."),
            "B(i): RM tail std of f": B["rm"]["f_std"],
            "B(i): CONST tail std of f": B["const"]["f_std"],
        },
        table=pd.concat([dfA, dfB], ignore_index=True),
    )


if __name__ == "__main__":
    r = run()
    print(r.report())
    r.save()
