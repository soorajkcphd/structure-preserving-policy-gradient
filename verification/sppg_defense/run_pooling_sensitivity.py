"""
Is the structure-selection result an artifact of the pooling convention?

    python -m sppg_defense.run_pooling_sensitivity --pairs pairs_actual.csv \\
        --model gpt2-medium --max-length 32

Why this exists
---------------
Section 4.1 says only "l2-normalized embeddings from GPT-2 Medium's final
layer".  That does not pin down how the token dimension is collapsed, and the
three natural readings give materially different answers:

  mean_padded   mean over all positions after padding to max_length.  This is
                what GPT2EmbeddingProvider.embed_sentence() in the manuscript's
                repository actually does:
                      padding="max_length";  hidden.mean(dim=1)
                For a 6-token sentence padded to 32, 26 of the 32 averaged
                positions are padding.  GPT-2 is causal with right padding, so
                those states are not noise -- each attends to the real tokens
                before it -- but averaging ~26 near-duplicate continuation
                states against ~6 content states compresses the differences
                between sentences, which is the most likely reason the reported
                held-out losses are of order 1e-7.

  mean_masked   mean over real tokens only.  The defensible version.

  last          final non-pad token.  The usual convention for a causal LM.

This script runs all three on the same pairs, with exhaustive leave-one-out at
every setting, and reports one table.  Whatever the outcome, the pooling rule
has to be stated in Section 4.1 -- and if the algebra ranking moves with it,
that is a result, not a detail.

Outputs (in --out)
------------------
  pooling_sensitivity.csv   LOO mean/sd per (pooling, class, model)
  pooling_sensitivity.tex   compact table: algebra x pooling, per class
  POOLING_VERDICT.txt       whether the ranking is robust
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

from .run_structure_diagnostic import embed
from .structure.procrustes import compare_fits

POOLINGS = ("mean_padded", "mean_masked", "last")

# the rows worth showing; everything else is near 1.0 by definition
KEEP = ["identity G = I", "so(k) EXACT (orthogonal Procrustes)",
        "sym(k) Adam 300 iters", "sl(k) Adam 300 iters", "gl(k) Adam 300 iters"]
# The denominator row is named after the chosen baseline and is excluded
# from KEEP because it is 1.0 by definition.
SHORT = {"identity G = I": "identity",
         "so(k) EXACT (orthogonal Procrustes)": "so(32) exact",
         "sym(k) Adam 300 iters": "sym(32)",
         "sl(k) Adam 300 iters": "sl(32)",
         "gl(k) Adam 300 iters": "gl(32)"}


def loo_table(V: np.ndarray, T: np.ndarray, adam_iters: int,
              adam_lr: float, baseline: str = "matrix32") -> pd.DataFrame:
    """Exhaustive leave-one-out; returns mean/sd of the relative residual."""
    m = V.shape[0]
    parts = []
    for h in range(m):
        tr = np.array([i for i in range(m) if i != h])
        c = compare_fits(V[tr], T[tr], V[[h]], T[[h]],
                         adam_iters=adam_iters, adam_lr=adam_lr,
                         baseline=baseline)
        c["held_out"] = h
        parts.append(c)
    allsp = pd.concat(parts, ignore_index=True)
    agg = (allsp.groupby("model")["rel_residual"]
           .agg(["mean", "std", "min", "max"]).reset_index())
    agg["n_splits"] = m
    return agg


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--model", default="gpt2-medium")
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=32,
                    help="must match self.max_length in the "
                         "GPT2EmbeddingProvider for mean_padded to reproduce it")
    ap.add_argument("--out", default="results/pooling")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--adam-iters", type=int, default=300)
    ap.add_argument("--adam-lr", type=float, default=5e-3)
    ap.add_argument("--baseline", default="matrix32",
                    choices=["matrix32", "flat1024"],
                    help="matrix32 = the k x k least-squares map Section 4.1 "
                         "describes; flat1024 = the d x d lstsq map the "
                         "repository actually uses as the denominator")
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    df = pd.read_csv(args.pairs)
    sents = df["source"].tolist() + df["target"].tolist()

    rows = []
    for pool in POOLINGS:
        print(f"\n{'=' * 70}\npooling = {pool}"
              f"{'  (reproduces the repository)' if pool == 'mean_padded' else ''}"
              f"\n{'=' * 70}")
        E = embed(sents, args.model, args.k, args.mock, args.seed,
                  pooling=pool, max_length=args.max_length)
        V_all, T_all = E[:len(df)], E[len(df):]

        # how much do source and target actually differ under this convention?
        sep = float(np.mean([np.linalg.norm(V_all[i] - T_all[i])
                             for i in range(len(df))]))
        print(f"mean ||phi(source) - phi(target)||_F = {sep:.6f}  "
              f"(both unit Frobenius norm; 0 = indistinguishable)")

        for cls, g in df.groupby("class", sort=False):
            ii = g.index.to_numpy()
            agg = loo_table(V_all[ii], T_all[ii], args.adam_iters, args.adam_lr,
                            args.baseline)
            agg = agg[agg.model.isin(KEEP)].copy()
            agg["model"] = agg["model"].map(SHORT)
            agg.insert(0, "class", cls)
            agg.insert(0, "pooling", pool)
            agg["pair_separation"] = sep
            rows.append(agg)
            best = agg.loc[agg["mean"].idxmin()]
            print(f"  {cls:16s} best = {best['model']:12s} "
                  f"({best['mean']:.3f} +- {best['std']:.3f})   "
                  f"so(32) = {float(agg.loc[agg.model == 'so(32) exact', 'mean'].iloc[0]):.3f}")

    out = pd.concat(rows, ignore_index=True)
    out.to_csv(f"{args.out}/pooling_sensitivity.csv", index=False)

    piv = out.pivot_table(index=["class", "model"], columns="pooling",
                          values="mean")[list(POOLINGS)]
    with open(f"{args.out}/pooling_sensitivity.tex", "w") as fh:
        fh.write("% Leave-one-out mean relative residual under three readings\n")
        fh.write("% of 'l2-normalized final-layer embeddings' (Section 4.1).\n")
        fh.write("% mean_padded reproduces the repository's embed_sentence().\n")
        fh.write(piv.to_latex(float_format="%.3f"))

    # ---- verdict -----------------------------------------------------------
    lines = ["POOLING SENSITIVITY", "=" * 62,
             f"pairs: {args.pairs}   model: {args.model}   "
             f"max_length: {args.max_length}",
             "*** MOCK -- pipeline check only ***" if args.mock else "", ""]

    winners = {}
    for (pool, cls), g in out.groupby(["pooling", "class"]):
        winners[(pool, cls)] = g.loc[g["mean"].idxmin(), "model"]
    classes = sorted(out["class"].unique())
    lines.append("Best-fitting algebra under each convention:")
    lines.append(f"  {'class':16s} " + " ".join(f"{p:>14s}" for p in POOLINGS))
    for c in classes:
        lines.append(f"  {c:16s} " +
                     " ".join(f"{winners[(p, c)]:>14s}" for p in POOLINGS))
    lines.append("")

    so_ever_wins = any(w == "so(32) exact" for w in winners.values())
    ranking_stable = all(len({winners[(p, c)] for p in POOLINGS}) == 1
                         for c in classes)

    so_rows = out[out.model == "so(32) exact"]
    id_rows = out[out.model == "identity"]
    merged = so_rows.merge(id_rows, on=["pooling", "class"],
                           suffixes=("_so", "_id"))
    so_beats_id = int((merged["mean_so"] < merged["mean_id"]).sum())

    lines += [
        f"so(32) is the best-fitting algebra in "
        f"{sum(w == 'so(32) exact' for w in winners.values())} of "
        f"{len(winners)} (convention, class) cells.",
        f"so(32) beats the identity map in {so_beats_id} of {len(merged)} cells.",
        f"The winning algebra is the same across all three conventions: "
        f"{ranking_stable}.",
        "",
        "READING", "-------",
    ]
    if not so_ever_wins:
        lines += [
            "so(32) is never the best-fitting algebra under any reading of",
            "Section 4.1's embedding description.  The structure-selection",
            "stage does not support the algebra the paper selects.",
            "",
            "This is consistent with the manuscript -- Section",
            "4.3 already states the ranking is 'not robust to sample size or",
            "reshape dimension' and that sym(32) typically wins at larger",
            "samples, and Section 7.5 finds the discriminant directions carry",
            "'substantial symmetric components outside so(32)'.",
            "",
            "Implication: structure selection is an exploratory diagnostic,",
            "and its result depends on the pooling convention shown above;",
            "the case for so(32) rests on the geometric argument and the RL",
            "results, as Section 4.3 says.",
            "S-7 checks that Task 1 is unaffected when G_ref is replaced by an",
            "arbitrary rotation.",
        ]
    elif not ranking_stable:
        lines += [
            "The winning algebra changes with the pooling convention.  The",
            "selection result therefore depends on an unstated preprocessing",
            "choice, and holds only together with the convention and this",
            "sensitivity table.",
        ]
    else:
        lines += [
            "The ranking is stable across all three conventions and so(32)",
            "wins in at least one class, so the selection result does not",
            "depend on the pooling convention.",
        ]

    txt = "\n".join(l for l in lines if l is not None)
    with open(f"{args.out}/POOLING_VERDICT.txt", "w") as fh:
        fh.write(txt + "\n")
    print("\n" + txt)
    print(f"\nwrote {args.out}/pooling_sensitivity.{{csv,tex}}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
