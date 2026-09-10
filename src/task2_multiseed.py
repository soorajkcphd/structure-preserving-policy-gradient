"""
Task 2: 10-Seed Multi-Seed Experiment
=======================================

Wraps the original Task 2 code (04_pap_CG.py) to run 10 independent
training seeds and compute proper multi-seed statistics.

Pipeline per seed:
  1. Structure discovery (shared across seeds -- done once)
  2. Build fresh SIPolicy with discovered X*
  3. Train REINFORCE for 120 iterations (seed-specific)
  4. Evaluate: structured so(n) vs trained unconstrained gl(n) control
  5. Collect per-seed reward pair

After all seeds: paired bootstrap, Welch t-test, Mann-Whitney U.

Usage:
    python task2_10seed.py

    Requires 04_pap_CG.py in the same directory.

Estimated runtime: ~30 min on A6000 (original single seed ~= 3 min x 10 seeds x overhead).
"""

import os
import json
import time
import random
import logging
from typing import List

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -- Import everything from the original Task 2 code --------------------------
from importlib.machinery import SourceFileLoader

# Load 04_pap_CG.py as a module (handles the non-standard filename)
_mod = SourceFileLoader("pap_cg", "04_pap_CG.py").load_module()

Cfg              = _mod.Cfg
LieAlgebra       = _mod.LieAlgebra
Discovery        = _mod.Discovery
SIPolicy         = _mod.SIPolicy
REINFORCE        = _mod.REINFORCE
evaluate         = _mod.evaluate
bootstrap_paired_diff = _mod.bootstrap_paired_diff
task_reward_from_ids  = _mod.task_reward_from_ids
load_model       = _mod.load_model
repr_vec         = _mod.repr_vec
set_seed         = _mod.set_seed
ensure_dir       = _mod.ensure_dir
json_safe        = _mod.json_safe

try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

log = logging.getLogger("task2_10seed")
log.setLevel(logging.INFO)
if not log.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    log.addHandler(h)


# =============================================================================
# STATISTICS
# =============================================================================

def bootstrap_ci(data, n_boot=10_000, alpha=0.05, seed=42):
    rng = np.random.default_rng(seed)
    data = np.asarray(data)
    boots = np.array([
        rng.choice(data, size=len(data), replace=True).mean()
        for _ in range(n_boot)
    ])
    return float(np.percentile(boots, 100 * alpha / 2)), \
           float(np.percentile(boots, 100 * (1 - alpha / 2)))


def cohens_d(a, b):
    a, b = np.asarray(a), np.asarray(b)
    na, nb = len(a), len(b)
    pooled = np.sqrt(
        ((na - 1) * a.std(ddof=1)**2 + (nb - 1) * b.std(ddof=1)**2)
        / max(na + nb - 2, 1)
    )
    return float((a.mean() - b.mean()) / (pooled + 1e-12))


# =============================================================================
# SINGLE-SEED TRAIN + EVAL
# =============================================================================

def run_single_seed(
    base_model,
    tok,
    alg:     LieAlgebra,
    X_star:  torch.Tensor,
    prompts: List[str],
    cfg:     Cfg,
    seed:    int,
) -> dict:
    """
    Train one seed and evaluate structured vs plain.

    Returns dict with:
      structured_rewards: list of per-prompt rewards (structured, lam=1.5)
      plain_rewards:      list of per-prompt rewards (trained gl(n) control)
      mean_structured:    float
      mean_plain:         float
      delta:              float (structured - plain)
      eval_stats:         dict from evaluate()
      training_final_reward: float
    """
    log.info(f"\n{'='*60}")
    log.info(f"SEED {seed}: Training REINFORCE for {cfg.rl_iters} iterations")
    log.info(f"{'='*60}")

    # Seed everything
    set_seed(seed)

    # Build fresh policy
    pol = SIPolicy(base_model, tok, alg, X_star, cfg).to(cfg.device)

    # Train REINFORCE
    trainer = REINFORCE(pol, tok, cfg)
    last_stats = None
    for it in range(1, cfg.rl_iters + 1):
        batch = [random.choice(prompts) for _ in range(cfg.batch_prompts)]
        last_stats = trainer.step(batch)
        if it % 30 == 0:
            dbg = getattr(trainer, "_last_debug", {})
            log.info(
                f"  [Seed {seed}] it {it}/{cfg.rl_iters} | "
                f"reward={last_stats['reward']:.4f} | "
                f"baseline={last_stats['baseline']:.4f} | "
                f"|gradtheta|={dbg.get('theta_grad', 0.0):.3e}"
            )

    training_final_reward = float(last_stats["reward"]) if last_stats else 0.0

    # Evaluate: structured policy (as trained, lam=1.5)
    log.info(f"  [Seed {seed}] Evaluating structured policy...")
    eval_stats = evaluate(pol, tok, prompts, cfg)

    # ------------------------------------------------------------------
    # Control arm.
    #
    # The control must not use lmbda = 0.0.  Because the policy computes
    #     mixed = base_logits + lmbda * struct_bias(...),
    # setting lmbda = 0 severs the gradient path entirely: theta_raw and the
    # feature heads receive exactly zero gradient, so such a "baseline" is
    # frozen GPT-2 and cannot train at all, and the comparison becomes
    # structured-vs-untrained rather than structured-vs-unstructured.
    # (Symptom: a baseline reward near-identical across seeds -- 0.6038
    # recurring -- with ~14x smaller variance than the structured arm.)
    #
    # The correct control keeps the same capacity, the same lmbda and the same
    # trainable path, and removes only the compactness constraint: an
    # unconstrained gl(n) generator in place of so(n).  It is trained for the
    # same budget with the same optimizer, so any difference is attributable
    # to the constraint rather than to whether the arm was trained.
    # ------------------------------------------------------------------
    log.info(f"  [Seed {seed}] Training unconstrained gl({alg.n}) control...")
    set_seed(seed)                      # same seed as the structured arm
    # Matched control: identical in every respect except the projection.
    #   - same discovered generators X_star (not None): the control keeps the
    #     same feature-head inputs, so the comparison does not confound
    #     "compactness" with "having discovered generators at all";
    #   - same lmbda, same optimizer, same budget, same seed;
    #   - only difference: gl(n) (unconstrained) instead of so(n) (compact).
    # Any measured difference is therefore attributable to the compactness
    # constraint alone, which is the claim the paper needs to defend.
    alg_ctrl = LieAlgebra("gl", alg.n)
    plain = SIPolicy(base_model, tok, alg_ctrl, X_star, cfg).to(cfg.device)
    plain.lmbda = pol.lmbda             # same bias weight; not zero

    _before = torch.cat([p.detach().reshape(-1).clone()
                         for p in plain.parameters() if p.requires_grad])
    ctrl_trainer = REINFORCE(plain, tok, cfg)
    for it in range(1, cfg.rl_iters + 1):
        batch = [random.choice(prompts) for _ in range(cfg.batch_prompts)]
        ctrl_trainer.step(batch)
    _after = torch.cat([p.detach().reshape(-1).clone()
                        for p in plain.parameters() if p.requires_grad])
    _moved = float((_after - _before).abs().max().item())
    log.info(f"  [Seed {seed}] control trained {cfg.rl_iters} iters; "
             f"max |dparam| = {_moved:.3e}")
    if _moved == 0.0:
        raise RuntimeError(
            f"[Seed {seed}] Control parameters did not change during training. "
            "The comparison would again be trained-vs-untrained. Aborting "
            "rather than producing a misleading result.")

    log.info(f"  [Seed {seed}] Evaluating control...")

    # Collect paired rewards: structured vs plain, same prompts, same eval seed.
    # Use fixed eval seeds [0, 1, 2] matching original ablation_validate --
    # this isolates training randomness from eval randomness.
    structured_rewards = []
    plain_rewards = []

    with torch.no_grad():
        for eval_seed in [0, 1, 2]:
            torch.manual_seed(eval_seed)
            np.random.seed(eval_seed)
            random.seed(eval_seed)

            for p in prompts[:cfg.eval_episodes]:
                # Structured
                ids = tok(p, return_tensors="pt").to(cfg.device)["input_ids"]
                out = pol.generate(ids, cfg.max_new_tokens)
                cont = out[:, ids.shape[1]:]
                r_s = task_reward_from_ids(base_model, tok, ids, cont)
                structured_rewards.append(r_s)

                # Plain
                ids2 = tok(p, return_tensors="pt").to(cfg.device)["input_ids"]
                out2 = plain.generate(ids2, cfg.max_new_tokens)
                cont2 = out2[:, ids2.shape[1]:]
                r_p = task_reward_from_ids(base_model, tok, ids2, cont2)
                plain_rewards.append(r_p)

    mean_s = float(np.mean(structured_rewards))
    mean_p = float(np.mean(plain_rewards))
    delta = mean_s - mean_p

    log.info(f"  [Seed {seed}] Structured={mean_s:.4f}  Plain={mean_p:.4f}  Delta={delta:+.4f}")

    # Free policy memory before returning (base model is shared, not freed)
    del pol, plain, trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "seed": seed,
        "structured_rewards": structured_rewards,
        "plain_rewards": plain_rewards,
        "mean_structured": mean_s,
        "mean_plain": mean_p,
        "delta": delta,
        "eval_stats": eval_stats,
        "training_final_reward": training_final_reward,
    }


# =============================================================================
# 10-SEED RUNNER
# =============================================================================

def run_10seed(seeds=None):
    if seeds is None:
        seeds = list(range(10))

    # -- Phase 1: Load model (once) ----------------------------------------
    cfg = Cfg()
    ensure_dir(cfg.results_dir)

    base_model, tok = load_model(cfg)
    n = base_model.config.n_embd
    alg = LieAlgebra(cfg.algebra, n)

    # -- Phase 2: Structure discovery (once, seed=0) -----------------------
    log.info("Phase 2: Structure Discovery (shared across all seeds)...")
    set_seed(0)

    prompts = [
        "Write about geometric structure in reinforcement learning.",
        "The story is about a robot that learns to paint.",
        "In the future, agents will learn from symmetry because",
        "Explain why group actions matter in optimization.",
        "A short story about a mathematician who loves groups.",
        "Summarize the role of invariances in optimization."
    ]
    V = torch.stack([repr_vec(base_model, tok, p, cfg.device) for p in prompts], dim=0)
    k = 4; eps = 0.02
    R = torch.randn(k, n, n, device=cfg.device) * eps
    T = torch.linalg.matrix_exp(alg.project(R))
    T_mats = [T[i] for i in range(k)]

    discover = Discovery(alg, cfg.device, lr=cfg.discovery_lr, steps=cfg.discovery_steps)
    perm_stats = discover.perm_test(T_mats, V, B=cfg.perm_tests)
    X_star = discover.fit(T_mats, V, steps=cfg.discovery_steps)

    log.info(f"  residual={perm_stats['obs_residual']:.6f}  "
             f"perm_mean={perm_stats['perm_mean']:.6f}  "
             f"p={perm_stats['p_value']:.3f}  "
             f"d={perm_stats['effect_size_d']:+.2f}")

    # -- Phase 3: Run all seeds --------------------------------------------
    log.info(f"\n{'='*70}")
    log.info(f"TASK 2: {len(seeds)}-SEED EXPERIMENT")
    log.info(f"{'='*70}")

    all_results = []
    per_seed_structured = []
    per_seed_plain = []
    per_seed_delta = []

    for seed in seeds:
        res = run_single_seed(
            base_model, tok, alg, X_star, prompts, cfg, seed
        )
        all_results.append(res)
        per_seed_structured.append(res["mean_structured"])
        per_seed_plain.append(res["mean_plain"])
        per_seed_delta.append(res["delta"])

    # -- Phase 4: Aggregate statistics -------------------------------------
    s_arr = np.array(per_seed_structured)
    p_arr = np.array(per_seed_plain)
    d_arr = np.array(per_seed_delta)

    print("\n" + "=" * 70)
    print(f"TASK 2 MULTI-SEED SUMMARY ({len(seeds)} seeds)")
    print("=" * 70)

    # Per-seed table
    print(f"\n{'Seed':<6s} {'Structured':>12s} {'Plain':>12s} {'Delta':>12s}")
    print("-" * 44)
    for res in all_results:
        print(f"{res['seed']:<6d} {res['mean_structured']:>12.4f} "
              f"{res['mean_plain']:>12.4f} {res['delta']:>+12.4f}")
    print(f"{'Mean':<6s} {s_arr.mean():>12.4f} {p_arr.mean():>12.4f} "
          f"{d_arr.mean():>+12.4f}")
    print(f"{'Std':<6s} {s_arr.std(ddof=1):>12.4f} {p_arr.std(ddof=1):>12.4f} "
          f"{d_arr.std(ddof=1):>+12.4f}")

    # Paired bootstrap (matching original code's method)
    (ci_lo, ci_hi), p_boot, delta_mean = bootstrap_paired_diff(
        s_arr, p_arr, n_boot=10_000, alpha=0.05, seed=0
    )

    # Also bootstrap on the per-seed deltas directly
    ci_delta = bootstrap_ci(d_arr)

    print(f"\n-- Paired Analysis --")
    print(f"  Delta (Structured - Plain) = {delta_mean:.4f}")
    print(f"  95% Bootstrap CI (paired): [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"  95% Bootstrap CI (delta):  [{ci_delta[0]:.4f}, {ci_delta[1]:.4f}]")
    print(f"  Bootstrap p-value: {p_boot:.4f}")

    if p_boot < 0.001:
        sig_label = "*** (p < 0.001)"
    elif p_boot < 0.01:
        sig_label = "** (p < 0.01)"
    elif p_boot < 0.05:
        sig_label = "* (p < 0.05)"
    else:
        sig_label = "ns (not significant)"
    print(f"  Significance: {sig_label}")

    # Welch t-test and Mann-Whitney (if scipy available)
    if HAS_SCIPY:
        t_stat, p_welch = scipy_stats.ttest_rel(s_arr, p_arr)
        # Also unpaired for comparison
        t_unpaired, p_unpaired = scipy_stats.ttest_ind(s_arr, p_arr, equal_var=False)
        u_stat, p_mwu = scipy_stats.mannwhitneyu(s_arr, p_arr, alternative="two-sided")
        print(f"\n  Paired t-test: t={t_stat:.4f}, p={p_welch:.4f}")
        print(f"  Welch (unpaired): t={t_unpaired:.4f}, p={p_unpaired:.4f}")
        print(f"  Mann-Whitney U: U={u_stat:.0f}, p={p_mwu:.4f}")
    else:
        p_welch = None

    d = cohens_d(s_arr, p_arr)
    print(f"\n  Cohen's d: {d:.4f}")

    # C4 check from last seed
    last = all_results[-1]["eval_stats"]
    print(f"\n-- C4 (Non-trivial outputs, last seed) --")
    print(f"  Avg reward:     {last['average_reward']:.3f}  (expect > 0.5)")
    print(f"  Avg length:     {last['average_length']:.1f}  (expect > 10)")
    print(f"  Avg repetition: {last['average_repetition']:.3f}  (expect < 0.1)")

    # -- Phase 5: Save results ---------------------------------------------
    out = {
        "n_seeds": len(seeds),
        "seeds": seeds,
        "per_seed": [
            {
                "seed": r["seed"],
                "mean_structured": r["mean_structured"],
                "mean_plain": r["mean_plain"],
                "delta": r["delta"],
                "training_final_reward": r["training_final_reward"],
            }
            for r in all_results
        ],
        "aggregate": {
            "mean_structured": float(s_arr.mean()),
            "std_structured": float(s_arr.std(ddof=1)),
            "mean_plain": float(p_arr.mean()),
            "std_plain": float(p_arr.std(ddof=1)),
            "delta_mean": float(delta_mean),
            "delta_std": float(d_arr.std(ddof=1)),
            "CI95_paired": [ci_lo, ci_hi],
            "CI95_delta": list(ci_delta),
            "p_bootstrap": float(p_boot),
            "p_welch_paired": float(p_welch) if p_welch is not None else None,
            "cohens_d": d,
        },
        "structure_discovery": {
            "residual": perm_stats["obs_residual"],
            "perm_mean": perm_stats["perm_mean"],
            "p_value": perm_stats["p_value"],
            "effect_size_d": perm_stats["effect_size_d"],
        },
    }

    ts = int(time.time())
    path = os.path.join(cfg.results_dir, f"task2_10seed_{ts}.json")
    with open(path, "w") as f:
        json.dump(json_safe(out), f, indent=2)
    log.info(f"\nResults saved to: {path}")

    # -- Phase 6: Plot -----------------------------------------------------
    plot_results(all_results, s_arr, p_arr, d_arr, ci_delta, delta_mean)

    # -- Summary for paper -------------------------------------------------
    print(f"\n{'='*70}")
    print("VALUES FOR PAPER UPDATE")
    print(f"{'='*70}")
    print(f"""
C1 (Structure benefit, 10 seeds):
  Delta reward   = {delta_mean:.4f}
  95% CI     = [{ci_lo:.4f}, {ci_hi:.4f}]
  p-value    = {p_boot:.4f}
  Cohen's d  = {d:.4f}
  Seeds used = {len(seeds)}

Update in paper:
  Section 8.1 (Setup): "Seeds {{0,...,9}}" (was {{0,1,2}})
  Table 7 (C1 row): Delta={delta_mean:.4f}, CI=[{ci_lo:.4f},{ci_hi:.4f}], p={p_boot:.4f}
  Abstract: update CI and p if materially different from original
""")

    return out


# =============================================================================
# PLOTS
# =============================================================================

def plot_results(all_results, s_arr, p_arr, d_arr, ci_delta, delta_mean):
    os.makedirs("plots", exist_ok=True)

    n = len(all_results)
    seeds = [r["seed"] for r in all_results]
    x = np.arange(n)
    w = 0.35

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: Per-seed rewards
    axes[0].bar(x - w/2, p_arr, w,
                label="Plain (lam=0)", color="#4c72b0", alpha=0.85)
    axes[0].bar(x + w/2, s_arr, w,
                label="Structured (lam=1.5)", color="#c44e52", alpha=0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"S{s}" for s in seeds], fontsize=9)
    axes[0].set_ylabel("Mean Reward", fontsize=12)
    axes[0].set_title("Task 2: Per-Seed Reward", fontsize=13)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, linestyle="--", alpha=0.4, axis="y")

    # Panel 2: Per-seed delta
    colors = ["#c44e52" if d > 0 else "#4c72b0" for d in d_arr]
    axes[1].bar(x, d_arr, color=colors, alpha=0.85)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].axhline(delta_mean, color="#c44e52", linestyle="--",
                    linewidth=1.5, label=f"Mean Delta = {delta_mean:.4f}")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"S{s}" for s in seeds], fontsize=9)
    axes[1].set_ylabel("Delta (Structured - Plain)", fontsize=12)
    axes[1].set_title("Per-Seed Reward Difference", fontsize=13)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, linestyle="--", alpha=0.4, axis="y")

    # Panel 3: Aggregate with CI
    ci_s = bootstrap_ci(s_arr)
    ci_p = bootstrap_ci(p_arr)
    ms, mp = s_arr.mean(), p_arr.mean()
    axes[2].bar([0, 1], [mp, ms],
                yerr=[[mp - ci_p[0], ms - ci_s[0]],
                      [ci_p[1] - mp, ci_s[1] - ms]],
                color=["#4c72b0", "#c44e52"], alpha=0.85, capsize=8,
                error_kw={"linewidth": 2})
    axes[2].set_xticks([0, 1])
    axes[2].set_xticklabels(["Plain (lam=0)", "Structured (lam=1.5)"], fontsize=11)
    axes[2].set_ylabel("Mean Reward", fontsize=12)
    axes[2].set_title("Mean +/- 95% Bootstrap CI", fontsize=13)
    axes[2].grid(True, linestyle="--", alpha=0.4, axis="y")

    fig.suptitle("Task 2: Structure-Informed REINFORCE (10 Seeds)", fontsize=15, y=1.02)
    fig.tight_layout()
    fname = os.path.join("plots", "task2_10seed.png")
    fig.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fname}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    run_10seed(seeds=list(range(10)))
