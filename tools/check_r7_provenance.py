#!/usr/bin/env python3
"""Check that an r7_cells.csv was produced by the current draw_geometry seeds.

Why this exists: r7 is the only sweep whose result depends on a matrix drawn
per geometry rather than per seed, and `run_r7` derives that matrix's seed from
the geometry's index.  If the seed formula is ever changed, an r7 CSV written
before the change still looks perfectly well-formed -- same columns, same row
count, same arms -- but its `sym`, `gl` and `diag` rows now describe different
environment matrices from the ones the current code would draw.  Nothing else
in the pipeline notices, and the scope verdict can flip.

For example, `data/r7_cells.csv` was written with consecutive seeds (9000,
9001, 9002, 9003).  Under a layout spaced 100 apart (9000, 9100, 9200, 9300)
`so` is unaffected (index 0 under both) and `id` is the identity for any seed,
so such a mismatch is easy to miss: two of the five rows still agree.

This script re-derives each geometry's matrix from the seed the current code
would use and compares the diagnostics against the ones stored in the CSV.

    python3 tools/check_r7_provenance.py                 # checks data/ and src/
    python3 tools/check_r7_provenance.py path/to.csv     # checks one file

Exit status 0 if every geometry matches the current code, 1 otherwise.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "src"))

TOL = 1e-6          # relative; the CSV stores full float64
SEARCH = range(9000, 9500)


def diagnostics(geom: str, seed: int, k: int) -> tuple[float, float, float]:
    """(menv_orth, menv_sym, menv_cond) for one geometry/seed, via the harness."""
    from sppg_experiments import draw_geometry          # noqa: PLC0415
    _, d = draw_geometry(geom, k, seed)
    return d["menv_orth"], d["menv_sym"], d["menv_cond"]


def close(a, b) -> bool:
    return all(abs(x - y) <= TOL * max(1.0, abs(y)) for x, y in zip(a, b))


def check(path: str) -> int:
    from sppg_experiments import GEOMS                  # noqa: PLC0415

    df = pd.read_csv(path)
    need = {"geom", "menv_orth", "menv_sym", "menv_cond"}
    if not need.issubset(df.columns):
        print(f"  {path}: not an r7 CSV (missing {sorted(need - set(df.columns))})")
        return 0

    from sppg_experiments import menv_seed              # noqa: PLC0415

    k = 32
    stored = df.groupby("geom")[["menv_orth", "menv_sym", "menv_cond"]].first()
    bad = []
    print(f"\n{path}")
    print(f"  {'geom':<6}{'seed now':>10}  {'verdict':<22}{'stored cond':>14}{'current cond':>14}")
    for gi, geom in enumerate(GEOMS):
        if geom not in stored.index:
            continue
        want = tuple(stored.loc[geom])
        seed_now = menv_seed(gi, 0)                     # the formula run_r7 uses
        got = diagnostics(geom, seed_now, k)
        if close(got, want):
            verdict = "matches current code"
        else:
            # Name the seed that would reproduce it, if one exists nearby.
            hit = next((s for s in SEARCH if close(diagnostics(geom, s, k), want)), None)
            verdict = f"MISMATCH (was seed {hit})" if hit else "MISMATCH (seed unknown)"
            bad.append(geom)
        print(f"  {geom:<6}{seed_now:>10}  {verdict:<22}{want[2]:>14.6f}{got[2]:>14.6f}")

    if bad:
        print(f"\n  {len(bad)} geometry/geometries stale: {', '.join(bad)}")
        print("  This CSV predates the current seed formula. Its rows describe")
        print("  different environment matrices from the ones the code draws now,")
        print("  so its scope verdict is not the one a re-run produces.")
        print("  Regenerate it:  ./check_all.sh --experiments-only --only r7 --overwrite")
        return 1
    print("\n  all geometries match the current code")
    return 0


def main() -> int:
    targets = sys.argv[1:] or [
        os.path.join(REPO, "data", "r7_cells.csv"),
        os.path.join(REPO, "src", "r7_cells.csv"),
    ]
    targets = [t for t in targets if os.path.exists(t)]
    if not targets:
        print("no r7_cells.csv found in data/ or src/ -- nothing to check")
        return 0
    return max(check(t) for t in targets)


if __name__ == "__main__":
    sys.exit(main())
