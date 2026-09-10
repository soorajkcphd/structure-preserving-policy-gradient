"""
Run the complete theory-verification suite.

    python -m sppg_defense.run_theory              # all modules
    python -m sppg_defense.run_theory t07 t10      # selected modules
    python -m sppg_defense.run_theory --quick      # reduced settings, ~1 min

Exit code is 0 only if every module passes, so this can be wired into CI or a
Makefile as the reproducibility gate.

Results are written to sppg_defense/results/ as one CSV and one JSON per module,
plus a combined summary; the LaTeX table for the appendix is emitted at the end.
"""
from __future__ import annotations

import argparse
import sys
import time

import pandas as pd

from .theory import (t01_softmax_smoothness, t05_frechet_bounds,
                     t06_geo_loss_smoothness, t07_projected_gd,
                     t08_robbins_monro, t09_pullback_bound,
                     t10_exponential_witness)
from .theory._common import RESULTS_DIR

MODULES = {
    "t01": (t01_softmax_smoothness, "TH-1  Lemma lem:softmax_lipschitz(i)"),
    "t05": (t05_frechet_bounds, "TH-5  Lemma lem:exp_derivative_bounds"),
    "t06": (t06_geo_loss_smoothness, "TH-6  Lemma lem:geo_smooth"),
    "t07": (t07_projected_gd, "TH-7  Theorem thm:nonconvex"),
    "t08": (t08_robbins_monro, "TH-8  Theorem thm:stochastic"),
    "t09": (t09_pullback_bound, "TH-9  Prop prop:dichotomy_app(i)"),
    "t10": (t10_exponential_witness, "TH-10/15 Prop prop:dichotomy_app(ii)"),
}

QUICK = {
    "t01": dict(radii=(1e-2, 1e0, 1e2, 1e4), m_values=(2, 16), B_values=(1.0,),
                n_directions=8),
    "t05": dict(n_values=(2, 8, 32), n_samples=2),
    "t06": dict(n_full=(4, 8), n_lanczos=(4, 8), radii=(0.0, 2.0, 100.0)),
    "t07": dict(T=2000, n_inits=3),
    "t08": dict(T=8000, n_runs=10),
    "t09": dict(n_values=(4, 16), n_theta=2, k=6),
    "t10": dict(),
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("modules", nargs="*", help="subset of: " + " ".join(MODULES))
    ap.add_argument("--quick", action="store_true",
                    help="reduced settings for a fast smoke test")
    args = ap.parse_args(argv)

    keys = args.modules or list(MODULES)
    unknown = [k for k in keys if k not in MODULES]
    if unknown:
        ap.error(f"unknown module(s): {unknown}; choose from {list(MODULES)}")

    print("=" * 78)
    print("SP-PG THEORY VERIFICATION SUITE")
    print("Every module verifies a stated theorem numerically.  A PASS means the")
    print("claim was reproduced to the stated tolerance, not merely asserted.")
    print("=" * 78)

    rows = []
    all_pass = True
    for k in keys:
        mod, claim = MODULES[k]
        kwargs = QUICK[k] if args.quick else {}
        t0 = time.time()
        res = mod.run(**kwargs)
        dt = time.time() - t0
        res.save()
        print()
        print(res.report())
        print(f"         {'elapsed (s)':38s} {dt:.1f}")
        rows.append(dict(module=res.name, claim=claim, passed=res.passed,
                         seconds=round(dt, 1)))
        all_pass &= res.passed

    df = pd.DataFrame(rows)
    df.to_csv(f"{RESULTS_DIR}/SUMMARY.csv", index=False)

    print()
    print("=" * 78)
    print(df.to_string(index=False))
    print("=" * 78)
    print(("ALL MODULES PASSED" if all_pass else "SOME MODULES FAILED")
          + f"   ({df.passed.sum()}/{len(df)})    results in {RESULTS_DIR}/")

    print()
    print("LaTeX table for the appendix:")
    print(r"\begin{tabular}{llc}")
    print(r"\toprule")
    print(r"Module & Verified statement & Outcome \\")
    print(r"\midrule")
    for r in rows:
        print(f"{r['module'].replace('_', chr(92) + '_')} & "
              f"{r['claim']} & {'PASS' if r['passed'] else 'FAIL'} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
