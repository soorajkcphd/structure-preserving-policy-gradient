"""
T-5  Frechet derivative bounds of the matrix exponential on so(n) -- and their
     failure off so(n).
Verifies Lemma `lem:exp_derivative_bounds`  ->  claims TH-5, TH-11.

Claim
-----
For theta, E1, E2 in so(n):
        ||D exp(theta)[E1]||_F          <= ||E1||_F
        ||D^2 exp(theta)[E1, E2]||_F    <= ||E1||_F ||E2||_F
both independently of ||theta||_F and of n.

The proof uses  D exp(theta)[E] = int_0^1 e^{(1-s)theta} E e^{s theta} ds  and
the fact that every exponential factor is orthogonal when theta is skew, so
Frobenius norm is preserved by the conjugation.

Why the contrast panel matters
------------------------------
The paper never shows that these bounds are specific to so(n).  They are: on
sym(n) and gl(n) the same ratios grow without bound with ||theta||_F, because
e^{theta} is no longer an isometry.  That contrast is the evidence that the
geometric channel is a property of skew-symmetry and not a generic fact about
matrix exponentials, so we compute it here explicitly.

Numerics
--------
First derivative: scipy.linalg.expm_frechet (exact to machine precision).
Second derivative: central difference of the first Frechet derivative along E2,
                   with Richardson extrapolation and a reported truncation-error
                   estimate, so the result is not an artifact of the step size.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.linalg import expm_frechet

from ..algebra import fro_norm, random_element
from ._common import Result

__all__ = ["d1_exp", "d2_exp", "run"]


def d1_exp(theta: np.ndarray, E: np.ndarray) -> np.ndarray:
    """First Frechet derivative D exp(theta)[E]."""
    return expm_frechet(theta, E, compute_expm=False)


def d2_exp(theta: np.ndarray, E1: np.ndarray, E2: np.ndarray,
           h: float = 1e-4) -> tuple[np.ndarray, float]:
    """
    Second Frechet derivative D^2 exp(theta)[E1, E2] by Richardson-extrapolated
    central differences of D exp(.)[E1] along E2.

    Returns (value, truncation_error_estimate).  The error estimate is the
    Frobenius distance between the h and h/2 central differences, which for a
    second-order scheme bounds the extrapolation error up to a constant.
    """
    def central(step: float) -> np.ndarray:
        return (d1_exp(theta + step * E2, E1) - d1_exp(theta - step * E2, E1)) / (2 * step)

    Dh = central(h)
    Dh2 = central(h / 2)
    rich = (4.0 * Dh2 - Dh) / 3.0                 # O(h^4) estimate
    return rich, fro_norm(Dh2 - Dh)


def run(subspaces=("so", "sym", "gl"),
        n_values=(2, 4, 8, 16, 32, 64, 128, 256),
        radii_so=(0.1, 1.0, 10.0, 100.0, 1000.0),
        radii_other=(0.1, 1.0, 3.0, 10.0),
        n_samples: int = 5,
        seed: int = 0) -> Result:
    rng = np.random.default_rng(seed)
    rows = []

    for sub in subspaces:
        radii = radii_so if sub == "so" else radii_other
        for n in n_values:
            # the second-derivative probe is O(n^3) with a large constant; keep
            # it to moderate n and rely on the first derivative for large n
            do_second = n <= 64
            for R in radii:
                for _ in range(n_samples):
                    th = random_element(sub, n, rng, fro=R)
                    E1 = random_element(sub, n, rng, fro=1.0)
                    E2 = random_element(sub, n, rng, fro=1.0)

                    r1 = fro_norm(d1_exp(th, E1)) / fro_norm(E1)
                    row = dict(subspace=sub, n=n, radius=R, ratio1=r1,
                               ratio2=np.nan, fd_err=np.nan)
                    if do_second:
                        D2, err = d2_exp(th, E1, E2)
                        row["ratio2"] = fro_norm(D2) / (fro_norm(E1) * fro_norm(E2))
                        row["fd_err"] = err
                    rows.append(row)

    df = pd.DataFrame(rows)
    so = df[df.subspace == "so"]
    off = df[df.subspace != "so"]
    if len(so) == 0 or len(off) == 0:
        raise RuntimeError("both so(n) and at least one contrast subspace are required")

    tol = 1e-8
    v1 = int((so["ratio1"] > 1 + tol).sum())
    v2 = int((so["ratio2"] > 1 + tol).sum())
    # NaN accounting.  ratio2 is only computed for n <= 64, and NaN > x is
    # False, so an unnoticed NaN would reduce the violation count to
    # zero.  Count the second-derivative rows explicitly and require that the
    # number of skipped rows is exactly the number expected.
    n_second = int(so["ratio2"].notna().sum())
    n_skipped = int(so["ratio2"].isna().sum())
    expected_skipped = int(sum(1 for _ in so.itertuples() if _.n > 64))
    nan_accounting_ok = (n_second > 0 and n_skipped == expected_skipped
                         and int(so["fd_err"].isna().sum()) == expected_skipped)

    # contrast: the same ratios must blow up off so(n)
    off_max1 = float(np.nanmax(off["ratio1"]))
    off_max2 = float(np.nanmax(off["ratio2"]))
    # growth with radius on sym/gl (median ratio at the largest vs smallest radius)
    grow = {}
    for sub in [s for s in subspaces if s != "so"]:
        g = off[off.subspace == sub]
        lo = float(g[g.radius == g.radius.min()]["ratio1"].median())
        hi = float(g[g.radius == g.radius.max()]["ratio1"].median())
        grow[sub] = hi / lo

    # step-size robustness: the second-derivative ratio must not depend on the
    # finite-difference step, or the "bound" would be an artifact of h
    # The same (theta, E1, E2) triples are reused at every h, so the spread is
    # purely the step-size sensitivity of the scheme and not resampling noise.
    h_rows = []
    rng2 = np.random.default_rng(seed + 7)
    triples = [(random_element("so", 16, rng2, fro=3.0),
                random_element("so", 16, rng2, fro=1.0),
                random_element("so", 16, rng2, fro=1.0)) for _ in range(6)]
    per_triple = {i: [] for i in range(len(triples))}
    for h in (1e-3, 3e-4, 1e-4, 3e-5, 1e-5):
        for i, (th, E1, E2) in enumerate(triples):
            D2, _ = d2_exp(th, E1, E2, h=h)
            val = fro_norm(D2) / (fro_norm(E1) * fro_norm(E2))
            per_triple[i].append(val)
            h_rows.append(dict(subspace="so", n=16, radius=3.0, h=h,
                               triple=i, ratio2=val))
    h_spread = float(max(max(v) - min(v) for v in per_triple.values()))

    passed = (v1 == 0 and v2 == 0 and off_max1 > 1.0 + 1e-6
              and nan_accounting_ok
              and h_spread < 1e-4
              and float(np.nanmax(so["fd_err"])) < 1e-4)

    return Result(
        name="t05_frechet_bounds",
        claim="TH-5 / Lemma lem:exp_derivative_bounds: ||Dexp||<=1 and ||D2exp||<=1 on so(n), uniformly in radius and n",
        passed=passed,
        summary={
            "so(n) rows (1st derivative)": int(len(so)),
            "so(n) rows with a 2nd derivative": n_second,
            "so(n) rows skipped (n > 64), accounted for exactly": nan_accounting_ok,
            "so(n) 1st-derivative violations (must be 0)": v1,
            "so(n) 2nd-derivative violations (must be 0)": v2,
            "so(n) max ratio1 (<= 1)": float(so["ratio1"].max()),
            "so(n) max ratio2 (<= 1)": float(np.nanmax(so["ratio2"])),
            "so(n) radii tested": f"{min(radii_so):g} .. {max(radii_so):g}",
            "so(n) dims tested": f"{min(n_values)} .. {max(n_values)}",
            "so(n) max finite-difference error estimate": float(np.nanmax(so["fd_err"])),
            "CONTRAST off so(n): max ratio1 (unbounded)": off_max1,
            "CONTRAST off so(n): max ratio2 (unbounded)": off_max2,
            "CONTRAST growth factor of ratio1, largest/smallest radius": grow,
            "2nd-derivative spread over FD steps h in [1e-5, 1e-3]": h_spread,
        },
        table=pd.concat([df, pd.DataFrame(h_rows)], ignore_index=True),
    )


if __name__ == "__main__":
    r = run()
    print(r.report())
    r.save()
