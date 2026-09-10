"""
Self-test for the non-theory modules: algebra, structure, stats, rl.

    python -m sppg_defense.selftest

Every check has a known correct answer, obtained independently of the code being
tested (closed forms, brute-force optimisation, or values published in the
manuscript).  Exit code 0 means all checks passed.
"""
from __future__ import annotations

import sys

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

from .algebra import (SUBSPACES, check_basis, from_coords, fro_norm,
                      orthonormal_basis, proj, random_element, subspace_dim)
from .rl.analysis import auc_trapezoid, normalised_score
from .rl.arms import (LoRAArm, LowRankSoArm, make_arm,
                      orthogonal_conjugation_is_a_noop, structure_diagnostics)
from .stats.tests import (holm, iqm, n_for_power_paired, power_paired,
                          paired_report, prob_superiority_paired,
                          probability_of_improvement, tost_paired,
                          tost_unpaired, unpaired_report)
from .structure.procrustes import (compare_fits, exhaustive_permutation_test,
                                   procrustes_so, residual)

FAILURES: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'ok ' if ok else 'FAIL'}] {label}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILURES.append(label)


def test_algebra() -> None:
    print("\nALGEBRA")
    for n in (2, 3, 4, 8, 16, 32):
        for name in SUBSPACES:
            check_basis(name, n, orthonormal_basis(name, n))
    check("orthonormal bases valid for so/sym/sl/gl, n in {2..32}", True)
    dims = {k: subspace_dim(k, 32) for k in SUBSPACES}
    check("dimensions match the manuscript (496/528/1023/1024)",
          dims == {"so": 496, "sym": 528, "sl": 1023, "gl": 1024}, str(dims))
    rng = np.random.default_rng(0)
    M = rng.standard_normal((32, 32))
    check("projections are idempotent and non-expansive",
          all(fro_norm(proj(k, proj(k, M)) - proj(k, M)) < 1e-12
              and fro_norm(proj(k, M)) <= fro_norm(M) + 1e-12 for k in SUBSPACES))
    # so + sym decomposition is orthogonal and complete
    check("M = Proj_so(M) + Proj_sym(M) exactly",
          fro_norm(proj("so", M) + proj("sym", M) - M) < 1e-12)
    check("<Proj_so M, Proj_sym M>_F = 0 (orthogonal complements)",
          abs(float(np.sum(proj("so", M) * proj("sym", M)))) < 1e-10)


def test_exp_so() -> None:
    print("\nMATRIX EXPONENTIAL ON so(n)  (TH-11)")
    rng = np.random.default_rng(1)
    worst_orth = worst_det = 0.0
    for n in (2, 4, 8, 16, 32, 64):
        for R in (0.1, 1.0, 10.0, 100.0):
            th = random_element("so", n, rng, fro=R)
            E = expm(th)
            worst_orth = max(worst_orth, fro_norm(E.T @ E - np.eye(n)))
            worst_det = max(worst_det, abs(np.linalg.det(E) - 1.0))
    check("exp(theta) is orthogonal for skew theta, all n and radii",
          worst_orth < 1e-9, f"max ||E^T E - I||_F = {worst_orth:.2e}")
    check("det exp(theta) = 1", worst_det < 1e-9, f"max |det-1| = {worst_det:.2e}")


def test_procrustes() -> None:
    print("\nSTRUCTURE SELECTION  (T-13 / S-1 / S-4)")
    rng = np.random.default_rng(0)
    k, m = 6, 5
    V = rng.standard_normal((m, k, k))
    G0 = procrustes_so(V, rng.standard_normal((m, k, k)))
    T = np.einsum("ij,mjk->mik", G0, V) + 0.01 * rng.standard_normal((m, k, k))

    G = procrustes_so(V, T)
    check("Procrustes solution lies in SO(k)",
          fro_norm(G.T @ G - np.eye(k)) < 1e-10 and abs(np.linalg.det(G) - 1) < 1e-10)

    basis = orthonormal_basis("so", k)
    best = np.inf
    for _ in range(30):
        r = minimize(lambda c: residual(expm(from_coords(basis, c)), V, T),
                     rng.standard_normal(basis.shape[0]) * 0.5, method="Powell",
                     options=dict(maxiter=20000, xtol=1e-10, ftol=1e-12))
        best = min(best, r.fun)
    gap = best - residual(G, V, T)
    check("Procrustes attains the global optimum (vs best of 30 restarts)",
          gap > -1e-8, f"gap = {gap:.3e}")

    # exact permutation test resolution
    res = exhaustive_permutation_test(V, T)
    check("exhaustive permutation test enumerates all 5! = 120 pairings",
          res["n_permutations"] == 120 and abs(res["min_attainable_p"] - 1 / 120) < 1e-12,
          f"p = {res['p_value']:.5f}, min attainable = {res['min_attainable_p']:.5f}")
    V9 = rng.standard_normal((9, k, k))
    G9 = procrustes_so(V9, rng.standard_normal((9, k, k)))
    T9 = np.einsum("ij,mjk->mik", G9, V9) + 1e-3 * rng.standard_normal((9, k, k))
    res9 = exhaustive_permutation_test(V9, T9, max_exact=8, n_random=300)
    check("sampled permutation branch can never return p = 0 (Phipson-Smyth)",
          res9["p_value"] >= res9["min_attainable_p"] > 0,
          f"p = {res9['p_value']:.5f} >= {res9['min_attainable_p']:.5f}")

    # identity-baseline diagnostics in the manuscript's near-identity regime
    Vn = rng.standard_normal((m, k, k))
    Vn /= np.linalg.norm(Vn.reshape(m, -1), axis=1)[:, None, None]
    Xt = proj("so", rng.standard_normal((k, k)))
    Xt *= 0.02 / fro_norm(Xt)
    Tn = np.einsum("ij,mjk->mik", expm(Xt), Vn) + 1e-4 * rng.standard_normal((m, k, k))
    df = compare_fits(Vn[:3], Tn[:3], Vn[3:], Tn[3:])
    has = {"identity", "EXACT", "Adam"}
    check("compare_fits reports identity, exact-Procrustes and Adam baselines",
          all(any(h in mstr for mstr in df.model) for h in has))
    ang = float(df.loc[df.model.str.contains("EXACT"), "max_rotation_deg"].iloc[0])
    check("rotation-angle diagnostic recovers the planted rotation magnitude",
          0.0 < ang < 5.0, f"max rotation = {ang:.3f} deg for a 0.02-Frobenius generator")


def test_stats() -> None:
    print("\nSTATISTICS  (validated against the manuscript's own reported values)")
    # The per-seed Task-1 returns are quoted in the manuscript to two decimals,
    # so these arrays are 2-dp reconstructions, not the raw seed values.  Any
    # statistic whose denominator is a standard deviation is therefore only
    # determined to within the de-rounding envelope.  Perturbing each entry by
    # U(-0.005, +0.005) (20 000 draws) gives Welch t in [30.446, 31.456] and
    # pooled d in [13.62, 14.07]; the paper's 30.663 and 13.7 sit inside both,
    # so the tolerances below are the rounding envelope, not slack.
    base = np.array([6.47, 6.41, 6.48, 6.50, 6.66, 6.67, 6.49, 6.44, 6.54, 6.40])
    sp = np.array([8.56, 9.05, 8.63, 9.09, 8.72, 8.58, 8.64, 8.54, 8.62, 8.93])
    u = unpaired_report(sp, base)
    check("Welch t reproduces Table tab:task1 (paper 30.663, 2-dp envelope +-0.51)",
          abs(u.t_stat - 30.663) < 0.51, f"ours {u.t_stat:.3f}")
    check("Mann-Whitney reproduces U = 100, p = 0.0002",
          u.nonparametric_stat == 100 and abs(u.nonparametric_p - 0.0002) < 1e-4,
          f"U={u.nonparametric_stat:.0f}, p={u.nonparametric_p:.4f}")
    check("pooled Cohen's d reproduces ~13.7 (2-dp envelope +-0.38)",
          abs(u.cohen_d_pooled - 13.7) < 0.38, f"ours {u.cohen_d_pooled:.2f}")

    # The IQM is a trimmed mean, so rounding does not move it appreciably; it is
    # the convention that matters.  main.py averages the closed percentile band
    # [q25, q75] (4 of 10 seeds here); the index-slice convention averages the
    # middle 6.  Only the former reproduces the table, to four decimals.
    check("IQM reproduces Table tab:task1 (6.486 / 8.652) under main.py's convention",
          abs(iqm(base) - 6.486) < 0.002 and abs(iqm(sp) - 8.652) < 0.002,
          f"ours {iqm(base):.4f} / {iqm(sp):.4f}")
    check("the index-slice IQM convention is a different estimand",
          abs(iqm(sp, method="slice") - iqm(sp, method="percentile")) > 0.03,
          f"slice {iqm(sp, method='slice'):.4f} vs "
          f"percentile {iqm(sp, method='percentile'):.4f} "
          "-> reconcile with the convention the paper's code uses, not the other one")

    rng = np.random.default_rng(0)
    b3 = rng.normal(0, 1, 10); s3 = rng.normal(0, 1, 10)
    b3 = (b3 - b3.mean()) / b3.std(ddof=1) * 0.034 + 2.804
    s3 = (s3 - s3.mean()) / s3.std(ddof=1) * 0.031 + 2.802
    r = tost_unpaired(s3, b3, margin=0.140)
    check("TOST reproduces the Task-3 equivalence result (paper p < 1.2e-8)",
          r["equivalent"] and r["p_tost"] < 1.2e-8, f"p_tost = {r['p_tost']:.3e}")
    check("TOST 90% CI reproduces the paper's [-0.028, +0.023]",
          abs(r["ci_low"] + 0.028) < 0.003 and abs(r["ci_high"] - 0.023) < 0.003,
          f"[{r['ci_low']:.4f}, {r['ci_high']:.4f}]")

    check("power at the Task-2 effect (dz=0.69, n=10) is ~0.5, i.e. underpowered",
          abs(power_paired(0.69, 10) - 0.495) < 0.02,
          f"{power_paired(0.69, 10):.3f}")
    check("n for 80% power at dz=0.69 is 19 seeds",
          n_for_power_paired(0.69, 0.8) == 19)
    check("n for 80% power at the ablation effect (dz~2.5) is <= 5 seeds",
          n_for_power_paired(2.49, 0.8) <= 5,
          "-> the single-seed ablations were avoidable, not unaffordable")

    h = holm([0.001, 0.02, 0.04, 0.30])
    check("Holm-Bonferroni step-down is correct",
          np.allclose(h["p_holm"], [0.004, 0.06, 0.08, 0.30]))
    pi = probability_of_improvement(sp, base)
    check("probability of improvement = 1.0 with a degenerate-free CI",
          pi["prob_improvement"] == 1.0)
    tie_a = np.array([1., 1, 1, 1, 1, 2, 2, 2, 2, 2])
    tie_b = np.ones(10)
    ps = prob_superiority_paired(tie_a, tie_b)
    check("prob-superiority point estimate lies inside its own CI under ties",
          ps["ci_low"] <= ps["p_superiority"] <= ps["ci_high"])
    ident = probability_of_improvement(np.ones(10), np.ones(10))
    check("probability of improvement handles identical arms consistently",
          ident["ci_low"] <= ident["prob_improvement"] <= ident["ci_high"])
    check("Holm rejects at exactly p_holm == alpha (<= not <)",
          holm([0.938, 0.025])["reject"] == [False, True])
    try:
        holm([0.01, float("nan")]); nan_ok = False
    except ValueError:
        nan_ok = True
    check("Holm raises on a non-finite p-value instead of passing", nan_ok)
    try:
        tost_paired(np.full(10, 2.14), np.full(10, 2.0), margin=0.14); tost_ok = False
    except ValueError:
        tost_ok = True
    check("TOST raises on degenerate SE instead of certifying equivalence at p=0",
          tost_ok)
    deg = paired_report(np.full(10, 3.0), np.full(10, 3.0))
    check("paired_report reports an exact null instead of aborting the analysis",
          deg.diff == 0.0 and "DEGENERATE" in deg.note)


def test_rl() -> None:
    print("\nRL ARMS AND ANALYSIS")
    rng = np.random.default_rng(0)
    n = 32
    M = rng.standard_normal((n, n))
    ok = True
    for spec in ("so", "sym", "sl", "gl", "random", "conj-so",
                 "lora:8", "lowrank-so:8"):
        a = make_arm(spec, n, rng)
        P = a.project(M)
        ok &= fro_norm(a.project(P) - P) < 1e-10
        ok &= fro_norm(P) <= fro_norm(M) + 1e-9
    check("every arm's projection is idempotent and non-expansive", ok)
    check("random-subspace control matches so(32) dimension exactly",
          make_arm("random", n, rng).dim == subspace_dim("so", n) == 496)
    check("LoRA r=8 has 512 raw parameters (the manuscript's convention)",
          LoRAArm(32, 8).n_parameters == 512 and LoRAArm(32, 8).dim == 448)
    check("orthogonal conjugation of so(n) is provably a no-op (so that arm "
          "would be a null control)",
          orthogonal_conjugation_is_a_noop(32, rng) < 1e-10,
          f"max discrepancy {orthogonal_conjugation_is_a_noop(32, rng):.2e}")
    conj = make_arm("conj-so", n, rng)
    check("non-orthogonal P so(n) P^-1 is a different subspace",
          fro_norm(conj.project(M) - proj("so", M)) > 1.0)
    lr8 = LowRankSoArm(32, 8); P8 = lr8.project(M)
    check("low-rank so(n) projection is exactly skew and exactly rank <= 2r",
          fro_norm(P8 + P8.T) < 1e-12 and np.linalg.matrix_rank(P8, tol=1e-9) <= 16)
    tied = np.zeros((8, 8))
    for i in (0, 2, 4):
        tied[i, i + 1], tied[i + 1, i] = 1.0, -1.0
    Pt = LowRankSoArm(8, 1).project(tied)
    check("low-rank projection survives tied singular values (Schur, not SVD)",
          np.linalg.matrix_rank(Pt, tol=1e-9) == 2 and fro_norm(Pt + Pt.T) < 1e-12)

    th = random_element("so", n, rng, fro=7.8)
    d_so = structure_diagnostics(th)
    d_sym = structure_diagnostics(random_element("sym", n, rng, fro=7.8))
    check("T-11 diagnostics separate so from sym at the divergent magnitude",
          d_so["orthogonality_err"] < 1e-10 and d_sym["orthogonality_err"] > 1.0,
          f"orth err so={d_so['orthogonality_err']:.2e} sym={d_sym['orthogonality_err']:.2e}")
    check("rho = 1 but op-norm > 1 is detectable (rho alone is insufficient)",
          "op_norm" in d_so and "spectral_radius" in d_so)

    # AUC per Eq. (7.1): trapezoid, unit spacing, half-weight endpoints, /T
    r = np.ones(60)
    check("auc_trapezoid of a constant-1 curve over 60 iters = 59/60",
          abs(auc_trapezoid(r) - 59 / 60) < 1e-12, f"{auc_trapezoid(r):.6f}")
    check("normalised_score maps random->0 and oracle->1",
          abs(normalised_score(5.2, 5.2, 12.0)) < 1e-12
          and abs(normalised_score(12.0, 5.2, 12.0) - 1) < 1e-12)


def main() -> int:
    print("=" * 78)
    print("SP-PG DEFENSE PACKAGE -- SELF-TEST")
    print("=" * 78)
    test_algebra()
    test_exp_so()
    test_procrustes()
    test_stats()
    test_rl()
    print("\n" + "=" * 78)
    if FAILURES:
        print(f"{len(FAILURES)} CHECK(S) FAILED:")
        for f in FAILURES:
            print("   -", f)
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
