"""
Publication figures from the theory-verification results.

    python -m sppg_defense.figures.make_figures        # after run_theory

Produces four PDFs (vector, ready for elsarticle) in sppg_defense/figures/out/:

  fig_t01_radius_independence.pdf
      lambda_max / B^2 versus ||theta||_F for so / sym / sl / gl, with the exact
      witness showing the bound is attained at every radius on every subspace.
      This is the figure that evidences the "two channels" separation: the
      smoothness bound is identical on the compact and non-compact subspaces.

  fig_t05_frechet_contrast.pdf
      ||D exp(theta)[E]||_F / ||E||_F on so(n) (flat at <= 1) against sym(n) and
      gl(n) (unbounded).  This is what makes the geometric channel SO-specific.

  fig_t07_rate.pdf
      min_t ||grad||^2 versus T on log-log with the 2L(f0 - f_inf)/T envelope,
      plus the sharp eta = 2/L divergence threshold.

  fig_t10_exponential_barrier.pdf
      log L_F versus R with fitted slope 2 mu and the (mu^2/4) e^{2 mu R} floor,
      beside the nilpotent panel where rho = 1 but the operator norm grows
      linearly -- i.e. why spectral radius is the wrong diagnostic.

Styling is plain: no colour dependence for the key contrasts
(line style and marker carry the distinction), so the figures survive greyscale
printing, which some journals still use.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..theory._common import RESULTS_DIR

OUT = os.path.join(os.path.dirname(__file__), "out")

STYLE = {
    "so": dict(color="#2B6CB0", marker="o", ls="-", label=r"$\mathfrak{so}(n)$"),
    "sym": dict(color="#C05621", marker="s", ls="--", label=r"$\mathfrak{sym}(n)$"),
    "sl": dict(color="#2F855A", marker="^", ls="-.", label=r"$\mathfrak{sl}(n)$"),
    "gl": dict(color="#6B46C1", marker="d", ls=":", label=r"$\mathfrak{gl}(n)$"),
}


def _setup():
    os.makedirs(OUT, exist_ok=True)
    plt.rcParams.update({
        "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
        "figure.dpi": 150, "savefig.bbox": "tight", "pdf.fonttype": 42,
    })


def _load(name: str) -> pd.DataFrame:
    path = os.path.join(RESULTS_DIR, f"{name}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found -- run `python -m sppg_defense.run_theory` first")
    return pd.read_csv(path)


# --------------------------------------------------------------------------- #
def fig_t01():
    df = _load("t01_softmax_smoothness")
    rf = df[df.kind == "random_features"]
    wit = df[df.kind == "exact_witness"]

    fig, ax = plt.subplots(figsize=(5.0, 3.1))
    for sub, st in STYLE.items():
        g = rf[rf.subspace == sub].groupby("radius")["ratio_eff"].max()
        ax.plot(g.index, g.values, ms=3.5, lw=1.2, alpha=0.9, **st)
    w = wit.groupby("radius")["ratio"].max()
    ax.plot(w.index, w.values, color="k", lw=1.6, ls="-", marker="*", ms=7,
            label="exact witness (all four subspaces)")
    ax.axhline(1.0, color="k", lw=0.8, ls="--", alpha=0.6)
    ax.text(2e-2, 1.03, r"bound $B^2$", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\|\theta\|_F$")
    ax.set_ylabel(r"$\lambda_{\max}(\nabla^2\log Z)\,/\,B^2$")
    ax.set_ylim(0, 1.45)
    ax.legend(loc="lower center", ncol=3, framealpha=0.95, fontsize=7,
              columnspacing=1.0, handlelength=1.8)
    ax.set_title("Radius-independent smoothness holds on every linear subspace",
                 fontsize=9)
    fig.savefig(f"{OUT}/fig_t01_radius_independence.pdf")
    plt.close(fig)


def fig_t05():
    df = _load("t05_frechet_bounds")
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9))

    ax = axes[0]
    for sub in ("so", "sym", "gl"):
        g = df[df.subspace == sub].groupby("radius")["ratio1"].max()
        ax.plot(g.index, g.values, ms=3.5, lw=1.2, **STYLE[sub])
    ax.axhline(1.0, color="k", lw=0.8, ls="--", alpha=0.6)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$\|\theta\|_F$")
    ax.set_ylabel(r"$\|D\exp(\theta)[E]\|_F\,/\,\|E\|_F$")
    ax.set_title("First Frechet derivative", fontsize=9)
    ax.legend(loc="upper left")

    ax = axes[1]
    so = df[(df.subspace == "so") & df.ratio1.notna()]
    g = so.groupby("n")["ratio1"].max()
    ax.plot(g.index, g.values, color=STYLE["so"]["color"], marker="o", ms=4, lw=1.2,
            label=r"$\mathfrak{so}(n)$, max over radii $10^{-1}\!-\!10^{3}$")
    ax.axhline(1.0, color="k", lw=0.8, ls="--", alpha=0.6)
    ax.set_xscale("log", base=2)
    ax.set_ylim(0.0, 1.25)
    ax.set_xlabel(r"$n$")
    ax.set_ylabel("ratio")
    ax.set_title("Independent of dimension", fontsize=9)
    ax.legend(loc="lower left")
    fig.savefig(f"{OUT}/fig_t05_frechet_contrast.pdf")
    plt.close(fig)


def fig_t07():
    df = _load("t07_projected_gd")
    rate = df[df.kind == "rate"]
    thr = df[df.kind == "threshold"]

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9))
    ax = axes[0]
    for obj, mk in (("quadratic", "o"), ("nonconvex_sin", "s")):
        g = rate[(rate.obj == obj) & (rate.eta_mult == 1.0)]
        gg = g.groupby("T").agg(v=("min_grad_sq", "median"),
                                b=("bound", "median")).sort_index()
        ax.plot(gg.index, gg["v"], marker=mk, ms=3.5, lw=1.2, label=f"observed, {obj}")
        ax.plot(gg.index, gg["b"], lw=1.0, ls="--", alpha=0.8,
                label=r"bound $2L(f_0-f_{\inf})/T$" if obj == "quadratic" else None)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$T$")
    ax.set_ylabel(r"$\min_{t<T}\|\mathrm{grad}_{\mathfrak{g}}f(X_t)\|_F^2$")
    ax.set_title(r"Stationarity rate at $\eta=1/L$", fontsize=9)
    ax.legend(loc="lower left", fontsize=7)

    ax = axes[1]
    t = thr.sort_values("eta_mult")
    ax.plot(t["eta_mult"], t["rho_predicted"], "k-", lw=1.2,
            label=r"predicted $|1-\eta\lambda_{\max}|$")
    ax.plot(t["eta_mult"], t["rho_empirical_1step"], "o", ms=5, mfc="none",
            color="#2B6CB0", label="measured (one exact step)")
    ax.axhline(1.0, color="k", lw=0.8, ls="--", alpha=0.6)
    ax.axvline(2.0, color="#C05621", lw=1.0, ls=":")
    ax.text(2.03, 0.15, r"$\eta=2/L$", color="#C05621", fontsize=8)
    ax.set_xlabel(r"$\eta L$")
    ax.set_ylabel(r"growth factor $\rho$")
    ax.set_title("Sharp divergence threshold", fontsize=9)
    ax.legend(loc="upper left", fontsize=7)
    fig.savefig(f"{OUT}/fig_t07_rate.pdf")
    plt.close(fig)


def fig_t10():
    df = _load("t10_exponential_witness")
    lb = df[df.kind == "lower_bound"]
    nil = df[df.kind == "nilpotent"]

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9))
    ax = axes[0]
    for mu, mk in zip(sorted(lb["mu"].unique()), ["o", "s", "^", "d"]):
        g = lb[lb.mu == mu].sort_values("R")
        # log10(L_F) as reported by L_F_lower (mpmath at scaled precision),
        # not reconstructed from a closed form
        ax.plot(g["muR"], g["log10_L_F"], marker=mk, ms=3.5, lw=1.1,
                label=fr"$\mu={mu}$")
    ax.plot([], [], " ", label=r"slope $=2\mu$ exactly")
    ax.axhline(np.log10(2.0), color="k", ls="--", lw=1.0)
    ax.text(50, np.log10(2.0) + 12, r"$\mathfrak{so}(n)$ bound $L_f+G_f=2$",
            fontsize=8)
    ax.set_xlabel(r"$\mu R$")
    ax.set_ylabel(r"$\log_{10} L_F$")
    ax.set_title("Exponential curvature on a non-compact direction", fontsize=9)
    ax.legend(loc="upper left", fontsize=7)

    ax = axes[1]
    ax.plot(nil["t"], nil["rho"], "o-", ms=4, lw=1.2, color="#2B6CB0",
            label=r"$\rho(e^{tN})$")
    ax.plot(nil["t"], nil["op_norm"], "s--", ms=4, lw=1.2, color="#C05621",
            label=r"$\|e^{tN}\|_{\mathrm{op}}$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel("value")
    ax.set_title(r"Nilpotent $N$: $\rho=1$ while $\|\cdot\|_{\mathrm{op}}\sim t$",
                 fontsize=9)
    ax.legend(loc="upper left", fontsize=7)
    fig.savefig(f"{OUT}/fig_t10_exponential_barrier.pdf")
    plt.close(fig)


def main():
    _setup()
    made = []
    for fn in (fig_t01, fig_t05, fig_t07, fig_t10):
        fn()
        made.append(fn.__name__)
    print(f"wrote {len(made)} figures to {OUT}/")
    for f in sorted(os.listdir(OUT)):
        print("   ", f)


if __name__ == "__main__":
    main()
