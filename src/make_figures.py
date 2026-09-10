#!/usr/bin/env python3
"""Regenerate the five manuscript figures, and nothing else.

Why this exists: the sweep harness (`sppg_experiments.py`) writes CSVs only --
it never draws anything.  The figure code lives in the *original* scripts, in
`main.py`'s `main()` and in `task1_figures.py`, and running those end to end
also runs a lot of work the manuscript no longer uses (structure discovery, the
lambda ablation, the diagnostics panel, the overhead timing).  This driver
calls only the five paths that produce figures the paper includes.

It edits nothing.  It imports `main.py` and `task1_figures.py` and calls their
public functions with the same arguments `main()` uses.

    cd src
    python3 make_figures.py                  # all five, into plots/
    python3 make_figures.py --only returns,ablation
    python3 make_figures.py --outdir ../figs

Produces, with the exact filenames the manuscript's \\includegraphics expects:

    rl_returns.png           single-seed comparison, seed 0
    rl_ablation.png          algebra ablation, seed 0
    multiseed_auc.png        10-seed per-seed AUC
    geo_weight_ablation.png  geometry-weight sweep, seed 0
    spectral_comparison.png  spectral radius by arm, seed 0

Needs a GPU, torch and transformers, and downloads GPT-2 Medium on first use.
Rough cost on an 8 GB laptop GPU: returns+ablation ~10 min each, multiseed
~40 min (it is ten training runs), geo-weight ~25 min, spectral ~15 min.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

STAGES = ("returns", "ablation", "multiseed", "geoweight", "spectral")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default="plots",
                    help="where to write the PNGs (default: plots/)")
    ap.add_argument("--only", default="",
                    help=f"comma-separated subset of {','.join(STAGES)}")
    ap.add_argument("--seeds", type=int, default=10,
                    help="seeds for the multiseed figure (default 10, matching Table tab:task1)")
    args = ap.parse_args()

    want = [s.strip() for s in args.only.split(",") if s.strip()] or list(STAGES)
    unknown = [w for w in want if w not in STAGES]
    if unknown:
        print(f"make_figures.py: unknown stage(s) {unknown}; known: {', '.join(STAGES)}",
              file=sys.stderr)
        return 2

    if not os.path.exists("main.py"):
        print("make_figures.py: run this from src/ (main.py is not here).", file=sys.stderr)
        return 1

    import main as M                                          # noqa: PLC0415

    print("=" * 72)
    print(f" regenerating: {', '.join(want)}   ->  {args.outdir}/")
    print("=" * 72)

    # Built exactly as main() builds them, so the figures match the paper.
    embedder = M.GPT2EmbeddingProvider(model_name="gpt2-medium", max_length=64)
    state_dim = embedder.hidden_dim
    env = M.MultiStepTextAlignmentEnv(
        embedder, n_prompts=16, n_actions=16, horizon=20,
        reward_noise=0.2, geo_weight=0.4, sparse_prob=0.7,
    )
    n_actions = env.n_actions
    thr = M.PPOConfig.reward_threshold

    done, failed = [], []

    def stage(name, fn):
        if name not in want:
            return
        print(f"\n--- {name} " + "-" * (68 - len(name)))
        t0 = time.time()
        try:
            fn()
            print(f"--- {name}: done in {time.time() - t0:.0f}s")
            done.append(name)
        except Exception as e:                                # noqa: BLE001
            print(f"--- {name}: FAILED ({type(e).__name__}: {e})", file=sys.stderr)
            failed.append(name)

    # rl_returns.png and rl_ablation.png are two different runs, kept separate
    # so --only can regenerate one without paying for the other.
    def _returns():
        base_res, lie_res = M.run_rl_comparison(env, state_dim, n_actions)
        M.plot_rl_returns(base_res, lie_res, save_dir=args.outdir, threshold=thr)

    def _ablation():
        abl = M.run_rl_ablation(env, state_dim, n_actions, seed=0)
        M.plot_ablation(abl, save_dir=args.outdir, threshold=thr)

    def _multiseed():
        ms = M.run_multiseed(env, state_dim, n_actions, seeds=list(range(args.seeds)))
        M.plot_multiseed(ms, save_dir=args.outdir)

    def _geoweight():
        # note: this one builds its own envs per weight, so it takes the
        # embedder rather than the env above -- as in main().
        geo = M.run_geo_weight_ablation(embedder, state_dim, n_actions, seed=0)
        M.plot_geo_weight_ablation(geo, save_dir=args.outdir)

    def _spectral():
        import task1_figures as F                             # noqa: PLC0415
        res = F.run_spectral_comparison(env, state_dim, n_actions, seed=0)
        F.plot_spectral_comparison(res, save_dir=args.outdir)

    stage("returns", _returns)
    stage("ablation", _ablation)
    stage("multiseed", _multiseed)
    stage("geoweight", _geoweight)
    stage("spectral", _spectral)

    expected = {
        "returns":   ["rl_returns.png"],
        "ablation":  ["rl_ablation.png"],
        "multiseed": ["multiseed_auc.png"],
        "geoweight": ["geo_weight_ablation.png"],
        "spectral":  ["spectral_comparison.png"],
    }
    print("\n" + "=" * 72)
    print(" SUMMARY")
    print("=" * 72)
    missing = []
    for s in want:
        for f in expected[s]:
            path = os.path.join(args.outdir, f)
            if os.path.exists(path):
                print(f"  ok    {path}  ({os.path.getsize(path) // 1024} KB)")
            else:
                print(f"  MISS  {path}")
                missing.append(path)
    if failed:
        print(f"\n  {len(failed)} stage(s) failed: {', '.join(failed)}")
    if missing or failed:
        return 1
    print("\n  Figures written.  Check each against its caption:")
    print("    multiseed_auc        10 seeds, two fully separated clusters")
    print("    rl_returns           seed 0, SP-PG ahead from iteration 2")
    print("    rl_ablation          seed 0, so > gl > sl > sym ~ baseline")
    print("    spectral_comparison  rho = 1.000 for so(32) on every seed")
    print("    geo_weight_ablation  so(32) margin rising monotonically in w")
    return 0


if __name__ == "__main__":
    sys.exit(main())
