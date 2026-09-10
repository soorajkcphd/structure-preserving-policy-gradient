#!/usr/bin/env python3
"""
Cross-Model Validation: SP-PG on Mistral-7B (Frozen, 4-bit Quantized)
======================================================================
Adapter script that reuses the exact infrastructure from main.py
(environment, policies, PPO trainer, statistics) but replaces
GPT-2 Medium embeddings with Mistral-7B embeddings.

Usage:
  # Place this file in the same directory as main.py
  python run_mistral_task1.py --seeds 0 1 2       # quick test
  python run_mistral_task1.py                       # full 10-seed run
  python run_mistral_task1.py --project_dim 1024    # project 4096->1024, k=32

Hardware: NVIDIA RTX 5060 Laptop (8 GB VRAM)
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# -- Import everything from main.py --------------------
# This ensures identical environment, policies, PPO, and statistics.
from main import (
    DEVICE,
    LieAlgebraOps,
    StructureDiscoveryExperiment,
    MultiStepTextAlignmentEnv,
    LiePolicy, BaselinePolicy, ValueNet,
    PPOConfig, LieStructuredPPO,
    build_semantic_pairs,
    stat_compare, auc_score, bootstrap_ci, iqm, cohens_d,
    _seed_all, _single_run, run_multiseed,
    run_structure_discovery,
    run_rl_comparison,
    run_rl_ablation,
    measure_overhead,
)


# =============================================================================
# MISTRAL EMBEDDING PROVIDER (same interface as GPT2EmbeddingProvider)
# =============================================================================

class MistralEmbeddingProvider:
    """
    Drop-in replacement for GPT2EmbeddingProvider using Mistral-7B (4-bit).

    Loads the model, extracts all needed embeddings, then frees the model
    from GPU memory so training can use the full 8 GB VRAM.

    If project_dim is set, applies a fixed random orthogonal projection
    from 4096 -> project_dim (e.g. 1024 for k=32 matching the paper).
    """

    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-v0.1",
        max_length: int = 64,
        project_dim: int = None,
    ):
        self.max_length = max_length
        self.model_name = model_name
        self.project_dim = project_dim
        self.native_dim = 4096  # Mistral-7B hidden size

        # Projection matrix (fixed, deterministic)
        self.P = None
        if project_dim is not None:
            assert int(project_dim ** 0.5) ** 2 == project_dim, \
                f"project_dim must be a perfect square, got {project_dim}"
            self.P = self._create_projection_matrix(self.native_dim, project_dim)
            self.hidden_dim = project_dim
            print(f"Projection: {self.native_dim} -> {project_dim} "
                  f"(k={int(project_dim**0.5)})")
        else:
            self.hidden_dim = self.native_dim

        # Load model, extract embeddings, free model
        self._model = None
        self._tokenizer = None
        self._load_model()

    def _create_projection_matrix(self, d_in, d_out, seed=42):
        """Fixed random orthogonal projection via QR decomposition."""
        rng = torch.Generator()
        rng.manual_seed(seed)
        G = torch.randn(d_in, d_out, generator=rng)
        Q, R = torch.linalg.qr(G)
        return Q.T  # (d_out, d_in)

    def _load_model(self):
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

        print(f"Loading {self.model_name} (4-bit quantized)...")
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quant_config,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        self._model.eval()
        print(f"Model loaded. Native hidden_dim={self.native_dim}")

    def _project(self, emb: torch.Tensor) -> torch.Tensor:
        """Apply projection if configured, then re-normalise."""
        if self.P is not None:
            if emb.dim() == 1:
                emb = (self.P.to(emb.device) @ emb)
            else:
                emb = emb @ self.P.T.to(emb.device)
            emb = F.normalize(emb.float(), dim=-1)
        return emb

    @torch.no_grad()
    def embed_sentence(self, text: str) -> torch.Tensor:
        """Embed a single sentence -- same interface as GPT2EmbeddingProvider."""
        enc = self._tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=self.max_length, padding="max_length",
        ).to(DEVICE)
        hidden = self._model(**enc, output_hidden_states=True).hidden_states[-1]
        emb = hidden.mean(dim=1).squeeze(0).float()
        emb = F.normalize(emb, dim=0)
        emb = self._project(emb)
        return emb.to(DEVICE)

    @torch.no_grad()
    def embed_pairs(self, pairs):
        """Embed sentence pairs -- same interface as GPT2EmbeddingProvider."""
        src_enc = self._tokenizer(
            [p[0] for p in pairs], return_tensors="pt", truncation=True,
            max_length=self.max_length, padding=True,
        ).to(DEVICE)
        tgt_enc = self._tokenizer(
            [p[1] for p in pairs], return_tensors="pt", truncation=True,
            max_length=self.max_length, padding=True,
        ).to(DEVICE)
        src_h = self._model(**src_enc, output_hidden_states=True).hidden_states[-1].mean(dim=1)
        tgt_h = self._model(**tgt_enc, output_hidden_states=True).hidden_states[-1].mean(dim=1)
        src_h = F.normalize(src_h.float(), dim=-1)
        tgt_h = F.normalize(tgt_h.float(), dim=-1)
        src_h = self._project(src_h)
        tgt_h = self._project(tgt_h)
        return src_h, tgt_h

    def free_model(self):
        """Free the LLM from GPU after all embeddings are extracted."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        torch.cuda.empty_cache()
        gc.collect()
        if torch.cuda.is_available():
            print(f"Model freed. GPU memory: {torch.cuda.memory_allocated()/1e6:.0f} MB")


# =============================================================================
# PRE-COMPUTE AND FREE PATTERN
# =============================================================================

def create_env_and_free_model(embedder, k=32):
    """
    Create the environment (which calls embedder.embed_sentence 16 times),
    run structure discovery (which calls embedder.embed_pairs),
    then free the LLM from GPU.
    """
    # Structure discovery (uses embed_pairs for synonym/clause/active-passive)
    print("\n" + "=" * 70)
    print("STRUCTURE DISCOVERY")
    print("=" * 70)
    sd_results = run_structure_discovery(embedder)

    # Create environment (uses embed_sentence for 16 prompts)
    print("\nCreating RL environment...")
    env = MultiStepTextAlignmentEnv(
        embedder,
        n_prompts=16,
        n_actions=16,
        horizon=20,
        reward_noise=0.2,
        geo_weight=0.4,
        sparse_prob=0.7,
        k_transform=k,
    )

    state_dim = embedder.hidden_dim
    n_actions = env.n_actions

    # Free the 7B model -- all embeddings are now in the env's tensors
    embedder.free_model()

    return env, state_dim, n_actions, sd_results


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="SP-PG cross-model validation on Mistral-7B")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Seeds (default: 0-9)")
    parser.add_argument("--project_dim", type=int, default=1024,
                        help="Project embeddings to this dim (default: 1024 for k=32)")
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-v0.1",
                        help="HuggingFace model name")
    parser.add_argument("--output_dir", type=str, default="results_mistral",
                        help="Output directory")
    parser.add_argument("--skip_multiseed", action="store_true",
                        help="Skip multi-seed, run seed-0 comparison only")
    args = parser.parse_args()

    seeds = args.seeds if args.seeds else list(range(10))
    k = int(args.project_dim ** 0.5) if args.project_dim else 64

    print("=" * 70)
    print("SP-PG Cross-Model Validation: Mistral-7B")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Projection: 4096 -> {args.project_dim} (k={k})")
    print(f"dim(so({k})) = {k*(k-1)//2}")
    print(f"Seeds: {seeds}")
    print(f"Device: {DEVICE}")
    print()

    _seed_all(42)

    # -- Step 1: Load Mistral, extract embeddings, create env, free model --
    t0 = time.time()
    embedder = MistralEmbeddingProvider(
        model_name=args.model,
        project_dim=args.project_dim,
    )

    env, state_dim, n_actions, sd_results = create_env_and_free_model(
        embedder, k=k
    )
    t_setup = time.time() - t0
    print(f"\nSetup time (model load + embeddings + env): {t_setup:.1f}s")
    print(f"state_dim={state_dim}, n_actions={n_actions}, k={k}")

    # -- Step 2: Single-seed comparison (seed=0) --------------------------
    print("\n" + "=" * 70)
    print("SINGLE-SEED COMPARISON (seed=0)")
    print("=" * 70)

    _seed_all(0)
    base_res, lie_res = _single_run(
        env, state_dim, n_actions, algebra="so", seed=0, verbose=True
    )

    print("\n-- Seed-0 Summary --")
    for name, res in [("Baseline PPO", base_res), ("SP-PG (SO)", lie_res)]:
        R = np.array(res["returns"])
        print(f"  {name}: mean={R.mean():.3f}  final={R[-1]:.3f}  "
              f"AUC={res['auc']:.3f}  "
              f"threshold_iter={res['threshold_crossing']}")

    # -- Step 3: Multi-seed (if not skipped) ------------------------------
    if not args.skip_multiseed:
        ms_results = run_multiseed(env, state_dim, n_actions, seeds=seeds)

        # Save results
        out_dir = Path(args.output_dir)
        out_dir.mkdir(exist_ok=True)
        out_path = out_dir / "results.json"

        save_data = {
            "model": args.model,
            "project_dim": args.project_dim,
            "k": k,
            "dim_so_k": k * (k - 1) // 2,
            "seeds": seeds,
            "setup_time_s": t_setup,
            "multiseed": {
                "base_aucs": ms_results["base_aucs"],
                "lie_aucs": ms_results["lie_aucs"],
                "base_finals": ms_results["base_finals"],
                "lie_finals": ms_results["lie_finals"],
                "base_crossings": ms_results["base_crossings"],
                "lie_crossings": ms_results["lie_crossings"],
            },
        }
        with open(out_path, "w") as f:
            json.dump(save_data, f, indent=2)
        print(f"\nResults saved to {out_path}")

        # LaTeX-ready output
        ba = np.array(ms_results["base_aucs"])
        la = np.array(ms_results["lie_aucs"])
        delta = (la.mean() - ba.mean()) / ba.mean() * 100
        print("\n-- For Paper (LaTeX) --")
        model_short = args.model.split("/")[-1]
        print(f"% {model_short} (projected to {args.project_dim}d, k={k})")
        print(f"% SP-PG: AUC ${la.mean():.3f} \\pm {la.std(ddof=1):.3f}$")
        print(f"% Baseline: AUC ${ba.mean():.3f} \\pm {ba.std(ddof=1):.3f}$")
        print(f"% Delta: ${delta:+.1f}\\%$")

    # -- Step 4: Algebra ablation (seed=0) --------------------------------
    print("\n" + "=" * 70)
    print("ALGEBRA ABLATION (seed=0)")
    print("=" * 70)
    run_rl_ablation(env, state_dim, n_actions, seed=0)

    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
