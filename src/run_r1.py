"""
R-1: channel decomposition of the Task-1 AUC gap.

    python run_r1.py --selftest        # instant, no GPU, no model
    python run_r1.py --seeds 20        # ~8 min
    python run_r1.py --analyse-only

Requires _r1core.py alongside it.  Makes no edits to main.py.

The two reward channels are averaged over completed episodes only, exactly
as main.py averages the episode return (512 = 25*20 + 12 discarded steps).
The 12 orphan steps are always non-terminal, and only non-terminal steps are
subject to the 0.7 sparsity mask, so averaging over all 512 steps would bias
the task channel downward (by ~0.009 AUC on the task-channel gap) and
auc_task + auc_geo would not equal the AUC main.py reports.  The
reconstruction is verified against main.py's own mean_ep every iteration.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

from _r1core import channel_aucs, channels, instrument, instrument_trainer

CSV = "r1_channels.csv"
METHOD, CONTROL = "so(32)", "baseline_ppo"


def collect(n_seeds: int) -> pd.DataFrame:
    from main import (GPT2EmbeddingProvider, MultiStepTextAlignmentEnv,
                      LiePolicy, BaselinePolicy, ValueNet, LieStructuredPPO,
                      _default_cfg, _seed_all)

    embedder = GPT2EmbeddingProvider(model_name="gpt2-medium", max_length=64)
    env = MultiStepTextAlignmentEnv(
        embedder, n_prompts=16, n_actions=16, horizon=20,
        reward_noise=0.2, geo_weight=0.4, sparse_prob=0.7,
    )
    w = instrument(env)
    horizon = int(env.horizon)
    state_dim, n_actions = embedder.hidden_dim, env.n_actions
    k = int(env.k_transform)

    pd.DataFrame(columns=["arm", "seed", "iteration", "r_task", "r_geo"]
                 ).to_csv(CSV, index=False)
    rows_all = []

    for seed in range(n_seeds):
        for arm in (CONTROL, METHOD):
            _seed_all(seed)
            cfg = _default_cfg()
            if arm == CONTROL:
                policy = BaselinePolicy(state_dim, n_actions)
                trainer = LieStructuredPPO(env, policy, ValueNet(state_dim),
                                           cfg, use_lie_projection=False)
            else:
                policy = LiePolicy(state_dim, n_actions, k=k, algebra="so")
                trainer = LieStructuredPPO(env, policy, ValueNet(state_dim),
                                           cfg, use_lie_projection=True,
                                           algebra="so")
            instrument_trainer(trainer, env, w, horizon, strict=True)
            res = trainer.train(verbose=False)

            pairs = channels(res, trainer)
            rows = [{"arm": arm, "seed": seed, "iteration": i,
                     "r_task": rt, "r_geo": rg}
                    for i, (rt, rg) in enumerate(pairs, start=1)]
            pd.DataFrame(rows).to_csv(CSV, mode="a", header=False, index=False)
            rows_all.extend(rows)

            a_t, a_g, a_tot = channel_aucs(pairs, w, horizon)
            err = abs(a_tot - float(res["auc"]))
            flag = "" if err < 1e-6 else f"  !! recon_err={err:.2e}"
            print(f"  seed {seed:2d}  {arm:13s}  AUC={res['auc']:7.3f}  "
                  f"task={a_t:6.3f}  geo={a_g:6.3f}  "
                  f"recon_err={err:.1e}{flag}", flush=True)

    df = pd.DataFrame(rows_all)
    print(f"\nWrote {CSV} ({len(df)} rows)")
    return df


def analyse(df: pd.DataFrame) -> None:
    from sppg_defense.rl.analysis import channel_decomposition
    out = channel_decomposition(df, w=0.4, method_arm=METHOD,
                                control_arm=CONTROL, steps_per_episode=20)
    print("\n" + "=" * 72)
    print("R-1  CHANNEL DECOMPOSITION")
    print("=" * 72)
    for k, v in out.items():
        print(f"  {k:32s} {v:+.4f}" if isinstance(v, float)
              else f"  {k:32s} {v}")


def selftest() -> None:
    sys.argv = ["run_r4.py", "--selftest"]
    import run_r4
    run_r4.selftest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--analyse-only", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        selftest()
        return
    if a.analyse_only:
        if not os.path.exists(CSV):
            sys.exit(f"{CSV} not found; run without --analyse-only first.")
        df = pd.read_csv(CSV)
    else:
        df = collect(a.seeds)
    analyse(df)


if __name__ == "__main__":
    main()
