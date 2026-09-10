"""
Mutation testing: break each claim and confirm that the suite fails.

    python -m sppg_defense.mutation_test

A verification suite that always passes proves nothing.  The most dangerous
class of bug is a test that passes vacuously -- comparing a quantity to
itself, maximising over random directions in a 496-dimensional
space (which under-reports the maximum by 10-100x), regressing an algebraic
identity, or evaluating an mpmath expression at a precision that makes the
selected index meaningless.

Each mutation below states a false version of a theorem and checks that the
corresponding measurement rejects it.  A mutation that is not caught means the
matching test cannot constrain the claim it advertises.

Exit code 0 means every mutation was caught.
"""
from __future__ import annotations

import sys

import numpy as np

CAUGHT: list[str] = []
MISSED: list[str] = []


def mutation(label: str, detected: bool, detail: str = "") -> None:
    tag = "CAUGHT " if detected else "MISSED "
    print(f"  [{tag}] {label}" + (f"   {detail}" if detail else ""))
    (CAUGHT if detected else MISSED).append(label)


# --------------------------------------------------------------------------- #
def mutate_t01() -> None:
    """False claim: log Z is (0.9 B^2)-smooth."""
    from .algebra import fro_norm, orthonormal_basis, proj
    from .theory.t01_softmax_smoothness import lambda_max_logZ

    print("\nT-1  softmax smoothness")
    rng = np.random.default_rng(0)
    worst = 0.0
    for sub in ("so", "sym", "sl", "gl"):
        B = orthonormal_basis(sub, 8)
        flat = B.reshape(B.shape[0], -1)
        P = proj(sub, rng.standard_normal((8, 8)))
        P /= fro_norm(P)
        co = (flat @ np.stack([P, -P]).reshape(2, -1).T).T
        c = co[0]
        u = rng.standard_normal(B.shape[0])
        u -= (u @ c) / (c @ c) * c
        u /= np.linalg.norm(u)
        worst = max(worst, lambda_max_logZ(co, 1e4 * u))
    mutation("a bound of 0.9*B^2 is rejected (the witness attains B^2 exactly)",
             worst > 0.9 + 1e-9, f"observed {worst:.6f} vs false bound 0.900")
    mutation("the true bound B^2 is not rejected", worst <= 1 + 1e-9,
             f"observed {worst:.6f} <= 1")


def mutate_t05() -> None:
    """False claim: ||D exp(theta)[E]||_F <= 0.9 ||E||_F on so(n)."""
    from .algebra import fro_norm, random_element
    from .theory.t05_frechet_bounds import d1_exp

    print("\nT-5  Frechet bounds")
    rng = np.random.default_rng(0)
    worst = 0.0
    for n in (2, 4, 8, 16, 32):
        for R in (0.1, 1.0, 10.0):
            for _ in range(20):
                th = random_element("so", n, rng, fro=R)
                E = random_element("so", n, rng, fro=1.0)
                worst = max(worst, fro_norm(d1_exp(th, E)) / fro_norm(E))
    mutation("a bound of 0.9 is rejected on so(n)", worst > 0.9 + 1e-9,
             f"observed {worst:.6f}")
    # and the bound must fail off so(n)
    off = 0.0
    for _ in range(20):
        th = random_element("sym", 16, rng, fro=10.0)
        E = random_element("sym", 16, rng, fro=1.0)
        off = max(off, fro_norm(d1_exp(th, E)) / fro_norm(E))
    mutation("the so(n) bound is rejected on sym(n) (so it is SO-specific)",
             off > 1.0 + 1e-6, f"sym(16) at radius 10 gives {off:.1f}")


def mutate_t06() -> None:
    """
    False claim: l_geo is 0.35-smooth.

    The mutation is checked against the number the module itself publishes
    (results/t06_geo_loss_smoothness.json, produced by run_theory), so it tests
    the shipped test rather than a narrower re-probe of it.
    """
    import json
    import os

    from .algebra import orthonormal_basis, random_element
    from .theory._common import RESULTS_DIR
    from .theory.t06_geo_loss_smoothness import (_lambda_max_lanczos, geo_loss,
                                                 random_SO, run as run_t06)
    from .theory.t09_pullback_bound import d2_directional

    print("\nT-6  geometric-loss smoothness")
    path = os.path.join(RESULTS_DIR, "t06_geo_loss_smoothness.json")
    if os.path.exists(path):
        with open(path) as fh:
            reported = float(json.load(fh)["summary"]
                             ["max lambda_max over all configurations"])
        src = "published result"
    else:                                                # pragma: no cover
        reported = float(run_t06(n_full=(8,), n_lanczos=(8, 32))
                         .summary["max lambda_max over all configurations"])
        src = "recomputed"

    mutation("a bound of 0.35 is rejected by the module's published maximum",
             reported > 0.35, f"{src}: {reported:.4f} vs false bound 0.350")

    # and show why Lanczos was necessary: a random-direction search on the same
    # problem lands an order of magnitude lower and would miss the mutation
    rng = np.random.default_rng(0)
    n = 32
    G = random_SO(n, rng)
    v = rng.standard_normal(n); v /= np.linalg.norm(v)
    b = orthonormal_basis("so", n)
    th = random_element("so", n, rng, fro=2.0)
    lanczos = _lambda_max_lanczos(th, v, G, b, rng)
    rnd = max(abs(d2_directional(
        lambda r, E=random_element("so", n, rng, fro=1.0): geo_loss(th + r * E, v, G)))
        for _ in range(200))
    mutation("random directions under-report lambda_max by >5x at dim 496 "
             "(why Lanczos replaced them)",
             lanczos > 5 * rnd, f"Lanczos {lanczos:.4f} vs random {rnd:.4f} "
                                f"({lanczos / max(rnd, 1e-12):.0f}x)")


def mutate_t07() -> None:
    """False claim: the descent inequality holds with L half its true value."""
    from .theory.t07_projected_gd import Quadratic, projected_gd

    print("\nT-7  projected-gradient descent")
    rng = np.random.default_rng(0)
    q = Quadratic(40, L=4.0, rng=rng)
    L_false = q.L / 2                       # understated smoothness constant
    eta = 1.9 / L_false                     # legal for L_false, illegal for L
    x0 = rng.standard_normal(40) * 0.5
    _, fs, gs, div = projected_gd(q, x0, eta, 400)
    with np.errstate(invalid="ignore"):
        lhs = fs[1:] - fs[:-1]
        rhs = -eta * (1 - L_false * eta / 2) * gs[:-1]
        viol = int(np.nansum(lhs > rhs + 1e-9))
    mutation("understating L by 2x is caught by the descent inequality",
             viol > 0 or div, f"{viol} violations, diverged={div}")
    # and the threshold test must localise 2/L exactly
    eigA = np.linalg.eigvalsh(q.A)
    grows_at = [em for em in (1.9, 1.99, 2.01, 2.1)
                if abs(1 - (em / q.L) * eigA[-1]) > 1]
    mutation("the divergence threshold sits exactly at eta = 2/L",
             grows_at == [2.01, 2.1], f"growth for eta*L in {grows_at}")


def mutate_t08() -> None:
    """False claim: the noise floor is the naive leading-order expression."""
    from .theory.t08_robbins_monro import quadratic_floor

    print("\nT-8  Robbins-Monro")
    rng = np.random.default_rng(0)
    eig = rng.uniform(0.5, 4.0, size=28)
    for alpha in (0.05, 0.30):
        exact = quadratic_floor(eig, alpha, 1.0, 28)
        naive = (alpha * 1.0 / (2 * 28)) * float(np.sum(eig))
        rel = abs(naive - exact) / exact
        mutation(f"naive leading-order floor is distinguishable at alpha={alpha}",
                 rel > 0.05, f"naive differs from exact by {rel:.1%}")


def mutate_t09() -> None:
    """False claim: ||Hess F|| <= 0.6 (the adversarial search reaches ~0.63)."""
    from .algebra import orthonormal_basis
    from .theory.t09_pullback_bound import SinPullback, _lambda_max

    print("\nT-9  pullback bound")
    rng = np.random.default_rng(0)
    best = 0.0
    for n in (2, 3, 4):
        fn = SinPullback(n, 1, rng)
        b = orthonormal_basis("so", n)
        th = np.zeros((n, n))
        for _ in range(40):
            from .algebra import random_element
            cand = [th] + [th + random_element("so", n, rng, fro=s)
                           for s in (0.05, 0.3, 1.0)]
            vals = [_lambda_max(fn, c, b, rng) for c in cand]
            i = int(np.argmax(vals))
            if vals[i] > best:
                best, th = vals[i], cand[i]
    mutation("a bound of 0.6 is rejected by the adversarial search",
             best > 0.6, f"adversarial max {best:.4f} vs false bound 0.600")
    mutation("the true bound L_f + G_f = 2 is not rejected", best <= 2.0,
             f"{best:.4f} <= 2")


def mutate_t10() -> None:
    """False setup: fixed 60-digit precision makes the index selection vacuous."""
    import mpmath as mp

    from .theory.t10_exponential_witness import L_F_lower

    print("\nT-10  exponential curvature witness")
    mu, R = 1.0, 300 * np.log(4 * np.pi)

    def bad_L_F(R, mu, dps=60):
        """The naive implementation: fixed precision."""
        with mp.workdps(dps):
            emuR = mp.e ** (mp.mpf(mu) * mp.mpf(R))
            k = mp.floor((emuR - mp.pi / 2) / (2 * mp.pi))
            y_k = mp.pi / 2 + 2 * mp.pi * k
            return float(abs(mp.sin(y_k) - 1))

    bad = bad_L_F(R, mu)
    good = L_F_lower(R, mu)
    mutation("fixed 60-dps precision is caught: sin(y_k) is not 1",
             bad > 1e-6, f"|sin(y_k)-1| = {bad:.3e} at 60 dps")
    mutation("scaled precision fixes it", good["exact"] and good["sin_yk_err"] < 1e-20,
             f"|sin(y_k)-1| = {good['sin_yk_err']:.2e} at scaled dps")
    # a false growth rate must be rejected
    Rs = np.linspace(np.log(4 * np.pi) / mu, 30 * np.log(4 * np.pi) / mu, 40)
    y = np.array([L_F_lower(r, mu)["log10_L_F"] for r in Rs]) * np.log(10)
    slope = float(np.polyfit(Rs, y, 1)[0])
    mutation("a claimed growth rate of mu (instead of 2*mu) is rejected",
             abs(slope - mu) / mu > 0.1, f"fitted slope {slope:.4f} vs 2*mu = {2 * mu}")


def mutate_stats() -> None:
    """The statistical guards must fire on degenerate input."""
    from .stats.tests import holm, paired_report, tost_paired

    print("\nSTATISTICS guards")
    try:
        tost_paired(np.full(10, 2.14), np.full(10, 2.0), margin=0.14)
        fired = False
    except ValueError:
        fired = True
    mutation("zero-variance TOST cannot certify equivalence at p = 0", fired)
    try:
        holm([0.01, float("nan")])
        fired = False
    except ValueError:
        fired = True
    mutation("a NaN p-value cannot be reported as 'not significant'", fired)
    r = paired_report(np.full(10, 3.0), np.full(10, 3.0))
    mutation("an exact null is reported, not raised",
             r.diff == 0.0 and "DEGENERATE" in r.note)


def mutate_rl() -> None:
    """The RL analysis guards must fire on misaligned or duplicated logs."""
    import pandas as pd

    from .rl.analysis import ablation_report, channel_decomposition

    print("\nRL analysis guards")
    rows = [dict(arm=a, seed=s, iteration=i, r_task=0.2, r_geo=0.5)
            for a, seeds in {"so(32)": [0, 1, 2, 3, 4],
                             "baseline_ppo": [0, 1, 2, 3, 9]}.items()
            for s in seeds for i in range(20)]
    try:
        channel_decomposition(pd.DataFrame(rows))
        fired = False
    except ValueError:
        fired = True
    mutation("misaligned seed sets cannot produce a paired test", fired)

    rng = np.random.default_rng(0)
    rows2 = [dict(arm=a, seed=s, auc=rng.normal())
             for a in ("so", "gl") for s in range(6)]
    rows2.append(dict(arm="so", seed=0, auc=99.0))
    try:
        ablation_report(pd.DataFrame(rows2))
        fired = False
    except ValueError:
        fired = True
    mutation("a duplicated (seed, arm) row cannot be averaged in", fired)

    from .rl.arms import orthogonal_conjugation_is_a_noop
    mutation("orthogonally-conjugated so(n) is confirmed to be a null control",
             orthogonal_conjugation_is_a_noop(16, rng) < 1e-10)


def main() -> int:
    print("=" * 78)
    print("SP-PG MUTATION TEST -- each mutation states a false claim and the")
    print("suite must reject it.  A missed mutation means the corresponding")
    print("test cannot constrain the claim it advertises.")
    print("=" * 78)
    for fn in (mutate_t01, mutate_t05, mutate_t06, mutate_t07, mutate_t08,
               mutate_t09, mutate_t10, mutate_stats, mutate_rl):
        fn()
    print("\n" + "=" * 78)
    print(f"{len(CAUGHT)} caught, {len(MISSED)} missed")
    if MISSED:
        for m in MISSED:
            print("   MISSED:", m)
        return 1
    print("ALL MUTATIONS CAUGHT -- every test constrains its claim")
    return 0


if __name__ == "__main__":
    sys.exit(main())
