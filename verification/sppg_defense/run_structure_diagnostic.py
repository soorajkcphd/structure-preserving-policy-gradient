"""
S-1 / T-13 / S-4 on real embeddings: the corrected structure-selection table.

    # 1. write a template to fill in with sentence pairs
    python -m sppg_defense.run_structure_diagnostic --write-template pairs.csv

    # 2. dry run with synthetic embeddings, no model download
    python -m sppg_defense.run_structure_diagnostic --pairs pairs.csv --mock

    # 3. the real thing
    python -m sppg_defense.run_structure_diagnostic --pairs pairs.csv \\
        --model gpt2-medium --k 32 --out results/structure

What it answers
---------------
The manuscript reports held-out losses of order 1e-7 on unit-norm reshaped
embeddings and concludes that so(32) fits synonym transformations best.  This
script checks three things that conclusion depends on:

  S-1  Is the fitted map essentially the identity?  A relative error of 1e-3.5
       on normalised embeddings is what you would see if source ~ target and
       exp(X_hat) ~ I.  This script reports ||X_hat||_F, ||exp(X_hat) - I||_F,
       the rotation angles IN degrees, and the residual of the identity map
       itself as an extra baseline column.  If the angles are a fraction of a
       degree, there is no rotational structure to select.

  T-13 The SO fit has a closed-form global optimum (orthogonal Procrustes).
       The manuscript uses 300 Adam steps from X = 0 and declines to claim
       optimality, while comparing the result against sl/sym/gl fits obtained
       the same way.  This script reports both and the suboptimality gap, so
       the cross-algebra comparison is no longer biased by an unknown amount.

  S-4  The exhaustive permutation test over all m! pairings that the paper's own
       Appendix recommends, replacing 128 Monte-Carlo draws with replacement
       from a pool of only 120 distinct permutations.

Output
------
  structure_table.tex     Table tab:structure, recomputed
  structure_rows.csv      every number behind it
  permutation.csv         exact per-class permutation tests
  VERDICT.txt             a plain-language reading of the diagnostics

Embedding convention (matches Section 4.1)
------------------------------------------
final-layer hidden state of the last token, L2-normalised to the unit sphere,
then reshaped row-major to k x k with phi(v)_{ab} = v_{(a-1)k+b}.  Requires
d = k^2 exactly (1024 = 32^2 for GPT-2 Medium).
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

from .structure.procrustes import compare_fits, exhaustive_permutation_test

TEMPLATE = """class,source,target
synonym,The results were significant.,The findings were notable.
synonym,She began the project quickly.,She started the project rapidly.
synonym,The method is very effective.,The approach is highly efficient.
synonym,He purchased a new vehicle.,He bought a new car.
synonym,The film was quite entertaining.,The movie was rather amusing.
clause_reorder,Because it rained we stayed inside.,We stayed inside because it rained.
clause_reorder,After the meeting ended everyone left.,Everyone left after the meeting ended.
clause_reorder,If you finish early call me.,Call me if you finish early.
clause_reorder,When the sun set the temperature dropped.,The temperature dropped when the sun set.
clause_reorder,Although he tried he did not succeed.,He did not succeed although he tried.
active_passive,The committee approved the proposal.,The proposal was approved by the committee.
active_passive,Researchers published the study.,The study was published by researchers.
active_passive,The chef prepared the meal.,The meal was prepared by the chef.
active_passive,Students completed the assignment.,The assignment was completed by students.
active_passive,The engineer designed the bridge.,The bridge was designed by the engineer.
"""


# --------------------------------------------------------------------------- #
def embed(sentences: list[str], model_name: str, k: int,
          mock: bool = False, seed: int = 0,
          mock_angle_deg: float = 0.5,
          pooling: str = "mean_padded", max_length: int = 32) -> np.ndarray:
    """
    Final-layer hidden states, L2-normalised, reshaped to (m, k, k).

    Pooling (this is not a cosmetic choice -- see below):

      "mean_padded"  mean over all positions after padding to `max_length`.
                     This reproduces GPT2EmbeddingProvider.embed_sentence() in
                     the manuscript's own repository:
                         padding="max_length"
                         hidden.mean(dim=1)
                     For a 6-token sentence padded to 32, 26 of the 32 averaged
                     positions are padding.  GPT-2 is causal and padding is on
                     the right, so those positions are not noise -- each attends
                     to the real tokens before it -- but the average is heavily
                     dominated by repeated continuation states, which compresses
                     the differences between sentences.  That is very likely why
                     the manuscript sees held-out losses of order 1e-7.

      "mean_masked"  mean over real tokens only (attention-mask weighted).  The
                     defensible version of the same idea.

      "last"         final non-pad token.  The usual convention for a causal LM.

    Reporting all three is a sensitivity analysis: if the algebra ranking
    flips with the pooling rule, the ranking depends on it.

    The input list is sources followed by targets, in that order.

    With --mock, the targets are the sources rotated by a small skew generator
    (default 0.5 degrees) plus noise -- i.e. in the near-identity
    regime the manuscript's 1e-7 held-out losses suggest.  This exercises the
    whole pipeline without downloading a model and shows what the "near-identity"
    branch of the verdict looks like.  Mock numbers are
    labelled as such in the output and are not results.
    """
    d = k * k
    if mock:
        from scipy.linalg import expm

        from .algebra import fro_norm, proj

        rng = np.random.default_rng(seed)
        m = len(sentences) // 2
        V = rng.standard_normal((m, d))
        V /= np.linalg.norm(V, axis=1, keepdims=True)
        X = proj("so", rng.standard_normal((k, k)))
        X *= np.deg2rad(mock_angle_deg) / max(fro_norm(X), 1e-12)
        R = expm(X)
        Vm = V.reshape(m, k, k)
        Tm = np.einsum("ij,mjk->mik", R, Vm) + 1e-4 * rng.standard_normal((m, k, k))
        Tm /= np.linalg.norm(Tm.reshape(m, -1), axis=1)[:, None, None]
        return np.concatenate([Vm, Tm], axis=0)

    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModel.from_pretrained(model_name).eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(dev)

    hid = model.config.hidden_size
    if hid != d:
        raise ValueError(
            f"{model_name} has hidden size {hid} but k={k} needs {d}. "
            f"The reshape phi requires d = k^2 exactly; gpt2-medium (1024) "
            f"pairs with k=32. For other models, project to k^2 first and say "
            f"so in the paper.")

    if pooling not in ("mean_padded", "mean_masked", "last"):
        raise ValueError(f"unknown pooling {pooling!r}")

    pad_mode = "max_length" if pooling == "mean_padded" else True
    out = []
    with torch.inference_mode():
        for i in range(0, len(sentences), 8):
            batch = sentences[i:i + 8]
            enc = tok(batch, return_tensors="pt", padding=pad_mode,
                      truncation=True, max_length=max_length).to(dev)
            h = model(**enc).last_hidden_state            # (B, L, d)
            if pooling == "mean_padded":
                v = h.mean(dim=1)                        # includes padding
            elif pooling == "mean_masked":
                mask = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
                v = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
            else:                                        # "last"
                idx = enc["attention_mask"].sum(1) - 1
                v = h[torch.arange(h.shape[0]), idx]
            out.append(v.float().cpu().numpy())
    V = np.concatenate(out, 0)
    V /= np.linalg.norm(V, axis=1, keepdims=True)         # onto the unit sphere
    return V[:, :d].reshape(-1, k, k)                     # row-major phi


def split_train_test(m: int, frac: float = 0.7, seed: int = 0):
    """
    70/30 split.  Uses int() not round(), matching the repository:
        split = int(self.train_split * N)
    With m = 5 that is int(3.5) = 3, i.e. 3 train / 2 test -- not 4/1.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(m)
    n_tr = max(1, int(frac * m))
    n_tr = min(n_tr, m - 1)                              # always keep >= 1 test
    return np.sort(idx[:n_tr]), np.sort(idx[n_tr:])


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", help="CSV with columns: class, source, target")
    ap.add_argument("--write-template", metavar="PATH",
                    help="write a fill-in-the-blanks CSV and exit")
    ap.add_argument("--model", default="gpt2-medium")
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--out", default="results/structure")
    ap.add_argument("--pooling", default="mean_padded",
                    choices=["mean_padded", "mean_masked", "last"],
                    help="mean_padded reproduces the manuscript's own "
                         "GPT2EmbeddingProvider (mean over all positions after "
                         "padding to --max-length); mean_masked averages real "
                         "tokens only; last takes the final non-pad token")
    ap.add_argument("--max-length", type=int, default=32,
                    help="must match self.max_length in the "
                         "GPT2EmbeddingProvider for mean_padded to reproduce it")
    ap.add_argument("--mock", action="store_true",
                    help="synthetic embeddings; exercises the pipeline only")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--adam-iters", type=int, default=300)
    ap.add_argument("--adam-lr", type=float, default=5e-3)
    ap.add_argument("--baseline", default="matrix32",
                    choices=["matrix32", "flat1024"],
                    help="matrix32 = the k x k least-squares map Section 4.1 "
                         "describes; flat1024 = the d x d lstsq map the "
                         "repository actually uses as the denominator")
    ap.add_argument("--loo", action="store_true",
                    help="exhaustive leave-one-out over splits; reports the "
                         "spread of every residual across splits, which is the "
                         "direct evidence for or against overfitting at m=5")
    args = ap.parse_args(argv)

    if args.write_template:
        with open(args.write_template, "w") as fh:
            fh.write(TEMPLATE)
        print(f"wrote {args.write_template}")
        print("Replace these placeholders with five sentence pairs per class, then:")
        print(f"  python -m sppg_defense.run_structure_diagnostic "
              f"--pairs {args.write_template} --mock       # dry run")
        return 0

    if not args.pairs:
        ap.error("--pairs is required (or use --write-template)")

    os.makedirs(args.out, exist_ok=True)
    df = pd.read_csv(args.pairs)
    for col in ("class", "source", "target"):
        if col not in df.columns:
            ap.error(f"{args.pairs} is missing column {col!r}")

    print(f"{'MOCK ' if args.mock else ''}embedding {2 * len(df)} sentences "
          f"with {args.model} (k={args.k}, d={args.k ** 2}, "
          f"pooling={args.pooling}, max_length={args.max_length})")
    if args.pooling == "mean_padded" and not args.mock:
        print("  Note: mean_padded averages over padding positions too, "
              "reproducing the repo's embed_sentence(). Compare against "
              "--pooling mean_masked to see how much of the fit depends on it.")
    sents = df["source"].tolist() + df["target"].tolist()
    E = embed(sents, args.model, args.k, args.mock, args.seed,
              pooling=args.pooling, max_length=args.max_length)
    V_all, T_all = E[:len(df)], E[len(df):]

    rows, perm_rows, loo_rows, loo_pair = [], [], [], []
    for cls, g in df.groupby("class", sort=False):
        ii = g.index.to_numpy()
        V, T = V_all[ii], T_all[ii]
        m = len(ii)
        tr, te = split_train_test(m, seed=args.seed)
        print(f"\n=== {cls}: {m} pairs ({len(tr)} train / {len(te)} test) ===")

        if args.loo:
            # every leave-one-out split, exhaustively.  With m = 5 pairs and 496
            # free parameters the fit is wildly over-determined, so the question
            # is not "what is the residual" but "does it survive changing which
            # single pair is held out".
            per_split = []
            for h in range(m):
                trh = np.array([i for i in range(m) if i != h])
                c = compare_fits(V[trh], T[trh], V[[h]], T[[h]],
                                 adam_iters=args.adam_iters, adam_lr=args.adam_lr,
                                 baseline=args.baseline)
                c["held_out"] = h
                per_split.append(c)
            allsp = pd.concat(per_split, ignore_index=True)
            spread = (allsp.groupby("model")["rel_residual"]
                      .agg(["mean", "std", "min", "max"]).reset_index())
            spread.insert(0, "class", cls)
            print("  leave-one-out spread of the relative residual "
                  f"({m} splits):")
            print(spread.to_string(index=False,
                                   float_format=lambda x: f"{x:.4f}"))
            loo_rows.append(spread)

            # Paired comparison across splits: is the so(n) fit actually better
            # than doing nothing?  Same held-out pair on both sides, so this is
            # the right paired contrast even at m = 5.
            piv = allsp.pivot(index="held_out", columns="model",
                              values="rel_residual")
            so_col = [c for c in piv.columns if "EXACT" in c][0]
            id_col = [c for c in piv.columns if c == "identity G = I"][0]
            d = piv[so_col] - piv[id_col]
            print(f"  so(k) EXACT minus identity, per split: "
                  f"{np.array2string(d.to_numpy(), precision=4)}")
            print(f"     mean {d.mean():+.4f}  (negative = so(k) is better; "
                  f"{int((d < 0).sum())}/{m} splits favour so(k))")
            loo_pair.append(dict(**{"class": cls, "mean_so_minus_identity": float(d.mean()),
                                    "splits_favouring_so": int((d < 0).sum()),
                                    "n_splits": m,
                                    "so_mean": float(piv[so_col].mean()),
                                    "identity_mean": float(piv[id_col].mean()),
                                    "so_sd": float(piv[so_col].std()),
                                    "identity_sd": float(piv[id_col].std())}))

        cmp = compare_fits(V[tr], T[tr], V[te], T[te],
                           adam_iters=args.adam_iters, adam_lr=args.adam_lr,
                           baseline=args.baseline)
        cmp.insert(0, "class", cls)
        cmp["adam_suboptimality_rel"] = cmp.attrs.get("so_adam_suboptimality_rel",
                                                      np.nan)
        rows.append(cmp)
        print(cmp[["model", "test_loss", "rel_residual", "X_fro",
                   "orthogonality_err", "max_rotation_deg"]].to_string(
                       index=False, float_format=lambda x: f"{x:.6g}"))

        pt = exhaustive_permutation_test(V, T, seed=args.seed)
        pt["class"] = cls
        perm_rows.append(pt)
        print(f"  permutation ({pt['estimator']}): p = {pt['p_value']:.5f} "
              f"(min attainable {pt['min_attainable_p']:.5f})")

    out = pd.concat(rows, ignore_index=True)
    perm = pd.DataFrame(perm_rows)
    out.to_csv(f"{args.out}/structure_rows.csv", index=False)
    perm.to_csv(f"{args.out}/permutation.csv", index=False)
    if loo_rows:
        pd.concat(loo_rows, ignore_index=True).to_csv(
            f"{args.out}/leave_one_out.csv", index=False)
    if loo_pair:
        lp = pd.DataFrame(loo_pair)
        lp.to_csv(f"{args.out}/loo_so_vs_identity.csv", index=False)
        with open(f"{args.out}/loo_table.tex", "w") as fh:
            fh.write("% Leave-one-out stability of the relative residual.\n")
            fh.write("% Leave-one-out is more informative than a single split: at m = 5 the\n")
            fh.write("% residual depends strongly on which pair is held out.\n")
            fh.write(lp.to_latex(index=False, float_format="%.3f"))

    # ---- corrected Table 2 -------------------------------------------------
    keep = out[out.model.str.contains("identity G = I|EXACT|Adam", regex=True)]
    piv = keep.pivot_table(index="class", columns="model",
                           values="rel_residual", aggfunc="first")
    with open(f"{args.out}/structure_table.tex", "w") as fh:
        fh.write("% Corrected replacement for Table tab:structure.\n")
        fh.write("% Adds the identity baseline (S-1) and the exact orthogonal\n")
        fh.write("% Procrustes solution (T-13) alongside the Adam fits.\n")
        fh.write(piv.to_latex(float_format="%.3f", na_rep="---"))
    print(f"\nwrote {args.out}/structure_table.tex")

    # ---- verdict -----------------------------------------------------------
    so = out[out.model.str.contains("EXACT")]
    ident = out[out.model.str.contains("identity G = I")]
    max_ang = float(so["max_rotation_deg"].max())
    so_orth_err = float(so["orthogonality_err"].max())
    beats_identity = bool((so["test_loss"].to_numpy()
                           < ident["test_loss"].to_numpy()).all())
    sub = float(np.nanmax(out["adam_suboptimality_rel"]))

    lines = [
        "STRUCTURE-SELECTION DIAGNOSTIC",
        "=" * 60,
        f"source: {args.pairs}   model: {args.model}   k: {args.k}",
        "*** MOCK EMBEDDINGS -- pipeline check only, not a result ***"
        if args.mock else "",
        "",
        f"S-1  Largest rotation angle of the exact SO fit: {max_ang:.3f} degrees."
        + ("  [UNAVAILABLE: the exact fit failed its orthogonality check, "
           f"err = {so_orth_err:.3e} -- please report this]"
           if not np.isfinite(max_ang) else ""),
        f"     Exact SO fit beats the identity map on every class: {beats_identity}.",
        "",
        "T-13 Worst-case suboptimality of the 300-step Adam SO fit versus the",
        f"     closed-form Procrustes optimum: {sub:.2%} of the training loss.",
        "",
        "READING",
        "-------",
    ]
    # Which class does the manuscript single out?  Table tab:structure claims
    # so(32) = 0.155 on synonyms, the lowest residual in the table, and Sec. 4.3
    # rests the "secondary post-hoc validation" on exactly that cell.
    syn = out[(out["class"].str.contains("syn", case=False)) &
              (out.model.str.contains("EXACT"))]
    syn_id = out[(out["class"].str.contains("syn", case=False)) &
                 (out.model.str.contains("identity G = I"))]
    identity_wins_headline = (
        bool(float(syn_id["test_loss"].iloc[0]) < float(syn["test_loss"].iloc[0]))
        if len(syn) and len(syn_id) else False)

    if identity_wins_headline:
        lines += [
            "The identity map beats the exact so(32) fit on the synonym class --",
            "the very cell Table tab:structure singles out (residual 0.155, the",
            "lowest in the table) and on which Section 4.3 rests its 'secondary",
            "post-hoc validation'.  Note the fitted rotation is not small",
            f"({max_ang:.1f} degrees), so this is not 'exp(X) is approximately",
            "the identity' -- it is overfitting: 496 free parameters fitted to",
            "4 training pairs produce a substantial rotation that generalises",
            "worse than doing nothing.",
            "",
            "Implication: the structure-selection stage is exploratory, and",
            "G_ref can be replaced by a fixed reference rotation.  Section 4.3",
            "already says the case for so(32) rests on the geometric argument",
            "and the RL results, not on this fit.  S-7 (swap G_ref for an",
            "arbitrary rotation) checks that Task 1 is unaffected.",
            "",
            "--loo shows this directly: if the residual swings widely depending",
            "on which single pair is held out, the fit is overfitting.",
        ]
    elif max_ang < 1.0:
        lines += [
            "The fitted generator is essentially the identity (< 1 degree of",
            "rotation).  The residuals are then describing noise around a",
            "near-identity map, not rotational structure.  Implication: the",
            "structure-selection stage is exploratory and G_ref can be replaced",
            "by a fixed reference rotation.  Section 4.3 already states the case",
            "for so(32) rests on the geometric argument and the RL results.",
            "S-7 checks that the Task-1 results are unchanged when G_ref is",
            "replaced by an arbitrary rotation.",
        ]
    elif not beats_identity:
        lines += [
            "The exact SO fit does not beat the identity map on every class.",
            "The relative residual against the unconstrained linear baseline is",
            "therefore not sufficient on its own; the identity column shows",
            "the classes where SO actually wins.",
        ]
    else:
        lines += [
            "The fit is a non-trivial rotation and beats the identity baseline",
            "on every class, so the structure-selection stage is supported by",
            "the corrected table (exact Procrustes, with the identity column)",
            "and the exhaustive permutation p-values.  The SO column is a",
            "global optimum rather than a 300-step Adam approximation.  The",
            "sample-size curve (S-2) remains open.",
        ]
    if sub > 0.01:
        lines += ["",
                  f"Note: the Adam fit was {sub:.1%} suboptimal, so the published",
                  "so(32) residuals are upper bounds and the cross-algebra",
                  "comparison in Table tab:structure is biased; the exact",
                  "column is not."]

    txt = "\n".join(l for l in lines if l is not None)
    with open(f"{args.out}/VERDICT.txt", "w") as fh:
        fh.write(txt + "\n")
    print("\n" + txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
