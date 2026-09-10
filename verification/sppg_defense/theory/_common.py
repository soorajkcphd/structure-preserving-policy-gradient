"""Shared helpers for the theory-verification suite."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd

RESULTS_DIR = os.environ.get(
    "SPPG_RESULTS", os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
)


@dataclass
class Result:
    """Outcome of one theory-verification experiment."""
    name: str
    claim: str                       # which TH-x / Lemma / Theorem it verifies
    passed: bool
    summary: dict = field(default_factory=dict)
    table: pd.DataFrame | None = None

    def save(self, outdir: str = RESULTS_DIR) -> None:
        os.makedirs(outdir, exist_ok=True)
        if self.table is not None:
            self.table.to_csv(os.path.join(outdir, f"{self.name}.csv"), index=False)
        with open(os.path.join(outdir, f"{self.name}.json"), "w") as fh:
            json.dump({k: v for k, v in asdict(self).items() if k != "table"},
                      fh, indent=2, default=_jsonable)

    def report(self) -> str:
        status = "PASS" if self.passed else "**FAIL**"
        lines = [f"[{status}] {self.name}  ({self.claim})"]
        for k, v in self.summary.items():
            lines.append(f"         {k:38s} {_fmt(v)}")
        return "\n".join(lines)


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _fmt(v) -> str:
    if isinstance(v, float):
        if v != 0 and (abs(v) < 1e-3 or abs(v) >= 1e5):
            return f"{v:.4e}"
        return f"{v:.6g}"
    return str(v)


def lambda_max_abs(hvp, d: int, rng: np.random.Generator,
                   n_iter: int = 300, n_restarts: int = 3,
                   tol: float = 1e-9) -> float:
    """
    Largest |eigenvalue| of a symmetric linear operator given only its
    matrix-vector product `hvp`.

    Uses Lanczos (scipy.sparse.linalg.eigsh on a LinearOperator), falling back
    to power iteration with restarts if Lanczos fails to converge.

    This replaces "maximise the quadratic form over random unit directions",
    which is a lower bound that concentrates near tr(H)/d in high dimension and
    therefore under-reports lambda_max by one to two orders of magnitude at
    d ~ 500 -- a test built on random directions cannot detect a violation.
    """
    from scipy.sparse.linalg import LinearOperator, eigsh, ArpackNoConvergence

    if d <= 8:
        # ARPACK requires k < N; for tiny operators just build the dense matrix.
        H = np.column_stack([hvp(e) for e in np.eye(d)])
        H = 0.5 * (H + H.T)
        return float(np.abs(np.linalg.eigvalsh(H)).max())

    op = LinearOperator((d, d), matvec=hvp, dtype=float)
    try:
        v0 = rng.standard_normal(d)
        vals = eigsh(op, k=1, which="LM", return_eigenvectors=False,
                     maxiter=n_iter * 10, tol=tol, v0=v0)
        return float(np.abs(vals).max())
    except (ArpackNoConvergence, ValueError):              # pragma: no cover
        best = 0.0
        for _ in range(n_restarts):
            v = rng.standard_normal(d)
            v /= np.linalg.norm(v)
            for _ in range(n_iter):
                w = hvp(v)
                nw = np.linalg.norm(w)
                if nw <= 1e-300:
                    break
                v = w / nw
            best = max(best, abs(float(v @ hvp(v))))
        return best


def loglog_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Least-squares slope of log y against log x (both strictly positive)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2:
        return float("nan")
    return float(np.polyfit(np.log(x[m]), np.log(y[m]), 1)[0])
