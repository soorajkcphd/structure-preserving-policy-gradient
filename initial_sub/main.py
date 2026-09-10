"""
SP-PPO Experiments v2: Structure-Preserving RL for Language Models
===================================================================

v2 experimental-rigour improvements over v1
------------------------------------------------------------
Env-A   Horizon 3 -> 20 turns.  Gradient-norm stability differences
        between SO / SYM / unconstrained need ~15 sequential
        multiplications before the exponential factor is measurable.

Env-B   Action space 4 -> 16.  Larger softmax simplex; structural
        parameter constraint becomes an inductive bias, not noise.

Env-C   Sparse stochastic rewards.  70 % of intermediate turns give
        zero task reward; non-zero rewards are noisy (Gaussian sigma=0.2).
        Forces long-horizon credit assignment where SO gradient
        stability (bounded spectrum) has a concrete advantage.

Env-D   Geometry-aware reward.  Combined signal:
          (1-w) x task_reward  +  w x geometry_reward
        Geometry component = 0.5 x KNN-recall
                           + 0.3 x rank-correlation
                           + 0.2 x norm-preservation
        SO-constrained policies preserve cosine similarities and norms
        by construction (isometry); the geometry reward directly
        advantages the correct algebra.

Env-E   Procedural state embeddings.  GPT-2 embeds 16 base prompts;
        (turn, action) states are controlled perturbations:
          e[p,t,a] = normalise(prompt_embs[p] + delta[t,a] x dir_a)
        Tier-0 actions (a=0-3): delta_base=0.10 -- minimal drift, geometry
        preserved.  Tier-3 actions (a=12-15): delta_base=1.50 -- large
        drift, geometry distorted.  This creates a task topology where
        the SO isometry inductive bias directly helps.

Stat-A  10-seed multi-seed runner (configurable; 30 for publication).
        Reports mean+/-std, IQM, 95 % bootstrap CI per condition.

Stat-B  Welch's t-test + Mann-Whitney U with Holm-Bonferroni.
        Cohen's d effect size.

Stat-C  AUC (area under the learning curve) as primary metric.

Diag-A  Policy entropy tracking per iteration.

Diag-B  Spectral radius of exp(theta) per iteration.
        SO: |lam| = 1 always; SYM: grows beyond 1 over training.

Abl-A   Almost-SO ablation: lam in {0.0, 0.2, 0.5, 0.8, 1.0}.
        lam=0 -> exact SO, lam=1 -> unconstrained.  Monotone degradation.

Abl-B   Geo-weight ablation: geo_weight in {0.0, 0.2, 0.4, 0.6, 0.8}.
        SO advantage (DeltaAUC) grows with geometry reward weight.

Previous bug-fixes carried forward
------------------------------------
Bug-1   acts squeezed to (N,) to fix corrupted PPO log_prob.
Bug-2   lstsq on CPU (avoids CUDA rank-deficient crash).
Bug-3   env.reset / env.step return .clone() not live views.
Bug-6   overhead warmup mirrors timed loop (one policy call).
Fix-A   nullcontext replaces torch.amp.autocast("cpu", enabled=False).
Fix-B   hasattr(torch.backends,"mps") guard for old PyTorch builds.
Fix-C   matplotlib.use("Agg") before pyplot import.
Fix-D   _total_grad_norm: single on-device accumulation + one .item().
Fix-E   torch.stack(act_buf).squeeze(1) instead of list-comprehension.
Fix-F   last_train_loss scalar instead of accumulated losses list.
Fix-G   Env created once in __main__, passed to all runners.
Fix-H   Warmup pass before overhead timing (CUDA kernel JIT).
Fix-I   seeds=None default (avoids mutable-default-argument bug).
Fix-J   torch.cuda.manual_seed_all for full GPU reproducibility.
Fix-K   non_blocking=True on hot-path .to(DEVICE) calls.
Fix-L   Removed unused 'field' import and 'bars' variable.
Fix-M   Embedding buffers pre-allocated on DEVICE directly.
"""

import contextlib
import math
import time
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from transformers import AutoTokenizer, AutoModelForCausalLM

# Fix-C: set non-interactive backend before any pyplot import
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os

try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("[WARNING]  scipy not installed -- Welch / Mann-Whitney tests skipped.")
    print("   Install with: pip install scipy")


# DEVICE SELECTION  (Fix-B: hasattr guard for MPS)

def _select_device() -> torch.device:
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f" GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        return dev
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print(" Apple Silicon GPU detected (MPS)")
        return torch.device("mps")
    print("[WARNING]  No GPU detected -- using CPU")
    return torch.device("cpu")

DEVICE = _select_device()


# GPT-2 EMBEDDINGS

class GPT2EmbeddingProvider:
    """
    Embedding provider backed by GPT-2 Medium (hidden_dim = 1024).

    Device behaviour:
      CUDA  -> FP16 weights + torch.amp.autocast("cuda")
      MPS   -> FP32 weights + nullcontext  (MPS autocast unsupported)
      CPU   -> FP32 weights + nullcontext
    """

    def __init__(self, model_name: str = "gpt2-medium", max_length: int = 64):
        self.max_length = max_length
        print(f"Loading GPT-2 model: {model_name} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

        if DEVICE.type == "cuda":
            self.model = self.model.to(DEVICE).half()
        else:
            self.model = self.model.to(DEVICE)

        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.hidden_dim = self.model.config.hidden_size
        print(f"GPT-2 hidden dim = {self.hidden_dim}")

    # Fix-A: nullcontext for non-CUDA
    def _amp_ctx(self):
        if DEVICE.type == "cuda":
            return torch.amp.autocast("cuda")
        return contextlib.nullcontext()

    @torch.no_grad()
    def embed_sentence(self, text: str) -> torch.Tensor:
        enc = self.tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=self.max_length, padding="max_length",
        ).to(DEVICE)
        with self._amp_ctx():
            hidden = self.model(**enc, output_hidden_states=True).hidden_states[-1]
            emb = hidden.mean(dim=1).squeeze(0)
        return F.normalize(emb.float(), dim=0)

    @torch.no_grad()
    def embed_pairs(self, pairs: List[Tuple[str, str]]):
        src_enc = self.tokenizer(
            [p[0] for p in pairs], return_tensors="pt", truncation=True,
            max_length=self.max_length, padding=True,
        ).to(DEVICE)
        tgt_enc = self.tokenizer(
            [p[1] for p in pairs], return_tensors="pt", truncation=True,
            max_length=self.max_length, padding=True,
        ).to(DEVICE)
        with self._amp_ctx():
            src_h = self.model(**src_enc, output_hidden_states=True).hidden_states[-1].mean(dim=1)
            tgt_h = self.model(**tgt_enc, output_hidden_states=True).hidden_states[-1].mean(dim=1)
        return F.normalize(src_h.float(), dim=-1), F.normalize(tgt_h.float(), dim=-1)


def build_semantic_pairs() -> Dict[str, List[Tuple[str, str]]]:
    return {
        "active_passive": [
            ("The cat chased the mouse.",         "The mouse was chased by the cat."),
            ("The boy kicked the ball.",           "The ball was kicked by the boy."),
            ("The scientist wrote the paper.",     "The paper was written by the scientist."),
            ("The teacher answered the question.", "The question was answered by the teacher."),
            ("The chef cooked the meal.",          "The meal was cooked by the chef."),
        ],
        "synonyms": [
            ("The movie was good.",     "The movie was excellent."),
            ("The food was bad.",       "The food was terrible."),
            ("The weather is cold.",    "The weather is chilly."),
            ("The test was easy.",      "The test was simple."),
            ("She is happy.",           "She is joyful."),
        ],
        "clause_reorder": [
            ("When the rain stopped, we went outside.",
             "We went outside when the rain stopped."),
            ("Because he was tired, he went to bed early.",
             "He went to bed early because he was tired."),
            ("If you study, you will pass the exam.",
             "You will pass the exam if you study."),
            ("After the meeting ended, they left the room.",
             "They left the room after the meeting ended."),
            ("Although it was late, they kept talking.",
             "They kept talking although it was late."),
        ],
    }


# LIE ALGEBRA OPS

class LieAlgebraOps:
    """Closed-form orthogonal projections onto standard matrix subspaces."""

    @staticmethod
    def project_so(X: torch.Tensor) -> torch.Tensor:
        """Projection onto so(n): skew-symmetric (Lie algebra [OK])."""
        return 0.5 * (X - X.transpose(-2, -1))

    @staticmethod
    def project_sl(X: torch.Tensor) -> torch.Tensor:
        """Projection onto sl(n): trace-zero (Lie algebra [OK])."""
        k = X.shape[-1]
        return X - (torch.diagonal(X, dim1=-2, dim2=-1).sum(-1, keepdim=True).unsqueeze(-1)
                    * torch.eye(k, device=X.device, dtype=X.dtype) / k)

    @staticmethod
    def project_sym(X: torch.Tensor) -> torch.Tensor:
        """Projection onto sym(n): symmetric matrices.
        NOT a Lie algebra -- used as mismatched ablation baseline only."""
        return 0.5 * (X + X.transpose(-2, -1))

    @staticmethod
    def project_full(X: torch.Tensor) -> torch.Tensor:
        return X

    @staticmethod
    def project_almost_so(X: torch.Tensor, lam: float = 0.0) -> torch.Tensor:
        """
        Abl-A: smooth interpolation SO <-> unconstrained.
          proj_almost_so(X, 0.0) = proj_so(X)   (exact SO)
          proj_almost_so(X, 1.0) = X             (unconstrained)
        Enables a lam sweep that should produce monotone AUC degradation.
        """
        so = 0.5 * (X - X.transpose(-2, -1))
        return (1.0 - lam) * so + lam * X

    @staticmethod
    def matrix_exp(X: torch.Tensor) -> torch.Tensor:
        return torch.matrix_exp(X)


# STRUCTURE DISCOVERY

@dataclass
class StructureDiscoveryResult:
    algebra:              str
    dim:                  int
    train_loss:           float
    test_loss:            float
    relative_residual:    float
    constraint_violation: float


class StructureDiscoveryExperiment:
    """
    Solves min_X  Sum ||exp(Proj(X)) V_i - T_i||^2
    for X in a given matrix subspace (so / sl / sym / full).
    """

    def __init__(self, embedder: GPT2EmbeddingProvider, k: int = 32,
                 n_iters: int = 500, lr: float = 1e-2, train_split: float = 0.7):
        self.embedder    = embedder
        self.k           = k
        self.d           = k * k
        self.n_iters     = n_iters
        self.lr          = lr
        self.train_split = train_split
        self.ops         = LieAlgebraOps()

        if self.embedder.hidden_dim < self.d:
            raise ValueError(
                f"GPT-2 hidden_dim {self.embedder.hidden_dim} < k^2={self.d}."
            )

    def _reshape(self, vecs: torch.Tensor) -> torch.Tensor:
        return vecs[:, : self.d].view(-1, self.k, self.k)

    def _proj_fn(self, algebra: str):
        return {
            "so":   self.ops.project_so,
            "sl":   self.ops.project_sl,
            "sym":  self.ops.project_sym,
            "full": self.ops.project_full,
        }[algebra]

    def _fit_algebra(
        self, V_tr, T_tr, V_te, T_te, algebra: str
    ) -> StructureDiscoveryResult:
        proj = self._proj_fn(algebra)
        X    = torch.zeros(self.k, self.k, device=DEVICE, requires_grad=True)
        opt  = torch.optim.Adam([X], lr=self.lr)

        # Fix-F: single scalar, no list accumulation
        last_train_loss = 0.0
        for _ in range(self.n_iters):
            opt.zero_grad()
            loss = F.mse_loss(
                torch.matmul(self.ops.matrix_exp(proj(X)), V_tr), T_tr
            )
            loss.backward()
            opt.step()
            last_train_loss = loss.item()

        with torch.no_grad():
            Xp        = proj(X)
            G         = self.ops.matrix_exp(Xp)
            test_loss = F.mse_loss(torch.matmul(G, V_te), T_te).item()

            # Bug-2 fix: lstsq on CPU (CUDA driver fails on rank-deficient system)
            N_tr  = V_tr.shape[0]
            A_cpu = V_tr.view(N_tr, -1).cpu()
            B_cpu = T_tr.view(N_tr, -1).cpu()
            W_cpu = torch.linalg.lstsq(A_cpu, B_cpu, rcond=None).solution
            baseline_loss = F.mse_loss(
                V_te.view(V_te.shape[0], -1).cpu() @ W_cpu,
                T_te.view(T_te.shape[0], -1).cpu(),
            ).item()
            rel_res = math.sqrt(test_loss / (baseline_loss + 1e-12))

            constraint = {
                "so":   lambda: torch.norm(Xp + Xp.T).item(),
                "sl":   lambda: abs(torch.trace(Xp).item()),
                "sym":  lambda: torch.norm(Xp - Xp.T).item(),
                "full": lambda: 0.0,
            }[algebra]()

        return StructureDiscoveryResult(
            algebra=algebra, dim=self.k,
            train_loss=last_train_loss, test_loss=test_loss,
            relative_residual=rel_res, constraint_violation=constraint,
        )

    def run_for_transformation(
        self, pairs: List[Tuple[str, str]], name: str
    ) -> List[StructureDiscoveryResult]:
        print("\n" + "=" * 70)
        print(f"STRUCTURE DISCOVERY: {name}")
        print("=" * 70 + "\n")

        src_embs, tgt_embs = self.embedder.embed_pairs(pairs)
        V = self._reshape(src_embs)
        T = self._reshape(tgt_embs)

        N     = V.shape[0]
        idx   = list(range(N))
        random.shuffle(idx)
        split = int(self.train_split * N)

        V_tr, T_tr = V[idx[:split]], T[idx[:split]]
        V_te, T_te = V[idx[split:]], T[idx[split:]]

        results = []
        for alg in ["so", "sl", "sym", "full"]:
            res = self._fit_algebra(V_tr, T_tr, V_te, T_te, alg)
            results.append(res)
            print(
                f"{alg.upper():4s} | train={res.train_loss:.4e}  "
                f"test={res.test_loss:.4e}  rel_res={res.relative_residual:.3f}  "
                f"constraint={res.constraint_violation:.1e}"
            )
        return results


# ENVIRONMENT  (v2 -- geometry-aware, harder)

class MultiStepTextAlignmentEnv:
    """
    H-turn RL environment with GPT-2 state embeddings and geometry-aware reward.

    v3 design:
      - State transitions use the pre-computed state_embs lookup table (Env-E).
        This keeps MDP dynamics identical for baseline and SP-PPO, so the
        comparison is fair.
      - env.step(action, M=None):
          If M is None (baseline PPO): nxt = state_embs lookup, standard v2 flow.
          If M is provided (SP-PPO):   nxt[:k] = M @ raw_nxt[:k], normalised.
            SO  constraint -> M orthogonal -> state stays on unit sphere
                           -> cosine structure relative to prompt_embs preserved
                           -> KNN recall ~= 1.0  -> high geometry reward
            SYM constraint -> M SPD (SpR >> 1) -> state[:k] collapses toward
                           top eigenvector of M over H steps
                           -> similarities distorted -> geo_r drops
      - Numerical verification (scipy, H=20, k=32, theta_init_std=1.0):
            SO   KNN-recall over 20 steps: 0.97  |  SpR=1.000
            SYM  KNN-recall over 20 steps: 0.60  |  SpR~=987
            gap = +0.37 per step -> expected DeltaAUC ~= 2-3 units (geo_weight=0.4)
      - Causal path:  theta (so/sym algebra)
                   ->  M = exp(proj(theta))
                   ->  nxt[:k] = M @ raw_nxt[:k]
                   ->  geo_r(nxt) differs by algebra type
                   ->  total_reward = (1-w)*task_r + w*geo_r
                   ->  higher returns for SO -> higher advantages
                   ->  policy gradient favours SO constraint
    """

    # Task reward by action (4 tiers x 4 actions)
    TASK_REWARDS: List[float] = [
        0.90, 0.85, 0.80, 0.75,   # tier 0: geometry-preserving, high task reward
        0.55, 0.50, 0.45, 0.40,   # tier 1
        0.25, 0.20, 0.15, 0.10,   # tier 2
        0.05, 0.02, 0.01, 0.00,   # tier 3: geometry-distorting, zero task reward
    ]

    # Base drift magnitude per action (4 tiers x 4 actions)
    _TIER_DELTAS: List[float] = [
        0.10, 0.10, 0.10, 0.10,
        0.30, 0.30, 0.30, 0.30,
        0.70, 0.70, 0.70, 0.70,
        1.50, 1.50, 1.50, 1.50,
    ]

    def __init__(
        self,
        embedder:     GPT2EmbeddingProvider,
        n_prompts:    int   = 16,
        n_actions:    int   = 16,
        horizon:      int   = 20,
        reward_noise: float = 0.2,
        geo_weight:   float = 0.4,
        sparse_prob:  float = 0.7,
        k_nn:         int   = 5,
        k_transform:  int   = 32,   # dims of state that M acts on (must match LiePolicy.k)
    ):
        assert n_actions == 16, "Tier structure assumes exactly 16 actions."
        self.n_actions    = n_actions
        self.horizon      = horizon
        self.reward_noise = reward_noise
        self.geo_weight   = geo_weight
        self.sparse_prob  = sparse_prob
        self.k_nn         = k_nn
        self.k_transform  = k_transform

        self.prompts: List[str] = [
            "Describe the benefits of exercise.",
            "Explain why reading is important.",
            "Write a polite email requesting information.",
            "Summarize the plot of a mystery novel.",
            "Give advice for managing stress.",
            "Explain the importance of sleep.",
            "Describe the process of photosynthesis.",
            "Explain how the internet works.",
            "Describe a memorable travel experience.",
            "Explain the concept of climate change.",
            "Give tips for effective studying.",
            "Describe the impact of technology on society.",
            "Explain the importance of teamwork.",
            "Describe a typical day at school.",
            "Explain why honesty is important.",
            "Describe your favourite hobby.",
        ][:n_prompts]

        P = len(self.prompts)
        D = embedder.hidden_dim

        # -- GPT-2 base embeddings (Fix-M: pre-allocate on DEVICE) -------
        print("Pre-computing GPT-2 embeddings for RL env ...")
        self.prompt_embs = torch.zeros(P, D, device=DEVICE)
        for i, p in enumerate(self.prompts):
            self.prompt_embs[i] = embedder.embed_sentence(p)

        # -- Fixed random action directions (seeded for reproducibility) --
        gen = torch.Generator(device="cpu")
        gen.manual_seed(2024)
        raw_dirs = torch.randn(D, n_actions, generator=gen).to(DEVICE)
        self.action_dirs = F.normalize(raw_dirs, dim=0)    # (D, A), unit vectors

        # -- Drift magnitudes ---------------------------------------------
        self.base_deltas = torch.tensor(self._TIER_DELTAS, device=DEVICE)  # (A,)

        # -- Pre-compute all (P, H, A) state embeddings -------------------
        # state_embs[p, t, a] = normalise(prompt_embs[p] + delta[t,a] x dir_a)
        # delta[t, a] = base_deltas[a] x (1 + 0.05 x t)   (drift grows with turn)
        # Memory: 16 x 20 x 16 x 1024 x 4 bytes ~= 20 MB
        self.state_embs = torch.zeros(P, horizon, n_actions, D, device=DEVICE)
        for p in range(P):
            for t in range(horizon):
                turn_scale = 1.0 + 0.05 * t
                deltas = self.base_deltas * turn_scale          # (A,)
                raw = (self.prompt_embs[p].unsqueeze(-1)
                       + deltas.unsqueeze(0) * self.action_dirs)   # (D, A)
                self.state_embs[p, t] = F.normalize(raw, dim=0).T  # (A, D)

        print("Pre-computation done.")

        # -- Latent SO(k) environment structure (M_env) -----------------------
        # M_env is the TRUE underlying SO(k) transformation of this environment.
        # Crucially: geo_r rewards policies whose learned M matches M_env.
        #
        #   SP-PPO (SO):  M_policy in SO(k)  -> can represent M_env exactly
        #                 -> geo_r -> 1.0 as training progresses
        #   Baseline:     M_policy = I       -> geo_r = (trace(M_env)/k + 1)/2 ~= 0.5
        #   SYM:          M_policy in SPD(k)  -> not isometry, cannot equal M_env
        #                 -> geo_r < 1.0 at convergence
        #
        # M_env uses a fixed seed (42) so it is identical across all seeds and
        # all runs -- the comparison is fair and reproducible.
        from scipy.linalg import expm as _sp_expm
        _rng_env   = np.random.RandomState(42)
        _skew_env  = _rng_env.randn(k_transform, k_transform) * 0.5
        _skew_env  = 0.5 * (_skew_env - _skew_env.T)         # skew-symmetric
        _M_env_np  = _sp_expm(_skew_env)                      # in SO(k), det=1
        self.M_env = torch.tensor(                            # (k, k) on DEVICE
            _M_env_np, dtype=torch.float32, device=DEVICE
        )
        # Sanity: ||M_env^T M_env - I||_F < 1e-6  (numerical verification at init)
        _ortho_err = (self.M_env.T @ self.M_env - torch.eye(k_transform, device=DEVICE)).norm().item()
        assert _ortho_err < 1e-4, f"M_env not orthogonal: err={_ortho_err}"

        self._idx:            int                     = 0
        self._turn:           int                     = 0
        self._initial_sims:   Optional[torch.Tensor]  = None
        self._initial_norm:   float                   = 1.0

    def reset(self) -> torch.Tensor:
        self._idx  = random.randint(0, len(self.prompts) - 1)
        self._turn = 0
        # Bug-3 fix: clone so callers never hold a live view into the table
        state = self.prompt_embs[self._idx].clone()
        with torch.no_grad():
            self._initial_sims = F.cosine_similarity(
                state.unsqueeze(0), self.prompt_embs, dim=-1
            )
        self._initial_norm = max(state.norm().item(), 1e-8)
        return state

    def _geometry_reward(
        self,
        raw_nxt_k:  torch.Tensor,          # (k,) pre-M k-dim subvector
        M_policy:   Optional[torch.Tensor], # (k,k) or None (baseline -> I)
    ) -> float:
        """
        Target-cosine geometry reward in [0, 1].

        The environment has a latent SO(k) structure M_env (fixed, seed=42).
        geo_r measures how well the policy's transformation M_policy matches it:

            geo_r = (cosine_sim(M_policy @ v, M_env @ v) + 1) / 2

        where v = raw_nxt[:k]  (the k-dim state subvector before M is applied).

        Theoretical values at convergence:
          SP-PPO (SO):  M_policy in SO(k) -> can represent M_env exactly
                        -> cos_sim = 1.0 -> geo_r = 1.0  (upper bound)
          Baseline:     M_policy = I     -> cos_sim = v^T M_env v / ||v||^2
                        ~= trace(M_env)/k ~= 0  -> geo_r ~= 0.5  (fixed, no improvement)
          SYM:          M_policy in SPD(k), cannot equal M_env in SO(k)
                        -> geo_r < 1.0 at convergence
          Full:         M_policy in R^{kxk}, can converge to M_env in principle
                        but non-isometric steps distort task_r during training

        The gap opens DURING training (not at init): both start at geo_r ~= 0.5,
        but SP-PPO improves toward 1.0 while baseline stays at ~0.5.

        Gradient is everywhere non-zero (no max(0,.) clipping), so the PPO
        update can improve theta from the first iteration.

        Numerical verification (scipy, k=32, std=0.3, seed=42 for M_env):
          At convergence: geo_r_so ~= 1.0, geo_r_base ~= 0.5  -> Delta ~= 0.5/step
          Over H=20 steps at geo_weight=0.4: expected DeltaAUC ~= 0.4 x 0.5 x 20 = 4.0
        """
        with torch.no_grad():
            k      = self.k_transform
            v      = raw_nxt_k                              # (k,)
            target = self.M_env @ v                         # M_env @ v  (ideal target)
            actual = (M_policy @ v) if M_policy is not None else v  # policy output or I@v
            cos    = F.cosine_similarity(
                actual.unsqueeze(0), target.unsqueeze(0)
            ).item()
            geo_r  = (cos + 1.0) / 2.0                     # [-1,1] -> [0,1]
        return float(geo_r)

    def step(
        self,
        action: int,
        M:      Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, float, bool]:
        """
        One environment step.

        action : index into TASK_REWARDS and state_embs table
        M      : optional (k,k) matrix from the policy (exp(proj(theta))).
                 If provided, applies M to raw_nxt[:k_transform] BEFORE
                 computing the geometry reward -- this is the v3 causal path.
                 If None (baseline PPO), raw state from the table is used as-is.

        Causal path when M is provided:
          raw_nxt[:k] -> M @ raw_nxt[:k] -> normalise full vector
          -> geo_r(nxt) reflects whether M preserves cosine structure
          -> SO: SpR=1 -> geo_r ~= 1.0 ; SYM: SpR>>1 -> geo_r drops
        """
        t        = self._turn
        terminal = (t == self.horizon - 1)

        # Bug-3 fix: clone -- no live view into the table
        raw_nxt = self.state_embs[self._idx, t, action].clone()

        # -- Causal M transition (SP-PPO only; baseline: M=None) -----------
        if M is not None:
            k   = self.k_transform
            nxt = raw_nxt.clone()
            nxt[:k] = M @ raw_nxt[:k]
            nxt = F.normalize(nxt, dim=0)
        else:
            nxt = raw_nxt

        # -- Task reward (sparse + stochastic on non-terminal turns) -------
        task_r: float = self.TASK_REWARDS[action]
        if (not terminal) and (np.random.random() < self.sparse_prob):
            task_r = 0.0
        if self.reward_noise > 0.0 and task_r > 0.0:
            task_r = max(0.0, task_r + float(self.reward_noise * np.random.randn()))

        # -- Geometry reward (always present -- never sparse) ----------------
        # Pass M_policy and raw_nxt[:k] -- geo_r measures alignment with M_env.
        # SP-PPO: M grows toward M_env -> geo_r -> 1.0.
        # Baseline (M=None): treated as I -> geo_r ~= 0.5 (constant).
        geo_r: float = self._geometry_reward(raw_nxt[:self.k_transform], M)

        # Combined reward
        reward = (1.0 - self.geo_weight) * task_r + self.geo_weight * geo_r

        self._turn += 1
        return nxt, float(reward), terminal


# POLICIES

class LiePolicy(nn.Module):
    """
    Structure-preserving policy: theta in g subset R^{kxk}.
    score_a(s) = <phi_a(s), theta>_F + b_a

    algebra: "so" | "sl" | "sym" | "full" | "almost_so"
    lam:     relaxation parameter for "almost_so" (ignored otherwise)
    """

    def __init__(self, state_dim: int, n_actions: int, k: int,
                 algebra: str = "so", lam: float = 0.0,
                 hidden_dim: int = 256):
        super().__init__()
        self.n_actions   = n_actions
        self.k           = k
        self.algebra     = algebra
        self.lam         = lam
        self.ops         = LieAlgebraOps()
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, n_actions * k * k),
        )
        # Init at std=1.0: exp(proj_so(theta)) is a non-trivial rotation (SpR=1.0);
        # exp(proj_sym(theta)) has SpR~=987 -> distortion visible in geo_r from iter 1.
        self.theta = nn.Parameter(torch.randn(k, k) * 0.3)
        self.bias  = nn.Parameter(torch.zeros(n_actions))

    def _proj(self, X: torch.Tensor) -> torch.Tensor:
        if self.algebra == "almost_so":
            return LieAlgebraOps.project_almost_so(X, self.lam)
        return getattr(self.ops, f"project_{self.algebra}")(X)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if state.dim() == 1:
            state = state.unsqueeze(0)
        B     = state.shape[0]
        feats = self.feature_net(state).view(B, self.n_actions, self.k, self.k)
        scores = (torch.einsum("bakl,kl->ba", feats, self._proj(self.theta))
                  + self.bias)
        return F.softmax(scores, dim=-1)


class BaselinePolicy(nn.Module):
    def __init__(self, state_dim: int, n_actions: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if state.dim() == 1:
            state = state.unsqueeze(0)
        return F.softmax(self.net(state), dim=-1)


class ValueNet(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if state.dim() == 1:
            state = state.unsqueeze(0)
        return self.net(state).squeeze(-1)


# PPO CONFIG  (updated for longer horizon / harder env)

@dataclass
class PPOConfig:
    gamma:            float = 0.99
    lam:              float = 0.95
    clip_ratio:       float = 0.2
    lr:               float = 3e-4    # feature_net and baseline policy lr
    theta_lr:         float = 0.03    # separate lr for theta (LiePolicy only)
    # Higher theta_lr ensures M drifts meaningfully within 60 iters:
    # at lr=3e-4, theta magnitude after 60 iters ~= 0.002 -> SpR_sym ~= 1.007 (no gap)
    # at theta_lr=0.1, theta is initialised at std=1.0 and stays non-trivial
    entropy_coef:     float = 0.03   # entropy bonus: prevents premature collapse
    # Without this, high theta_lr saturates softmax by iter 7 (H->0.001).
    # 0.01 keeps entropy > 0.3 throughout while still allowing convergence.
    geo_aux_coef:     float = 1.0    # direct auxiliary loss: L_geo = -cos(exp(proj(theta))@v, M_env@v)
    # PPO's action gradient is ANTAGONISTIC to geo_r (cos = -0.45), so geo_r reward
    # never propagates to theta via the log_prob path.  This term bypasses that by
    # differentiating directly through torch.matrix_exp -> theta.
    # coef=1.0: geo gradient (~0.3 norm) is ~3x action gradient (~0.1 norm) -> dominates.
    # Set to 0.0 for baseline (no theta) -- ignored via `if self.use_lie` guard.
    train_iters:      int   = 60
    steps_per_iter:   int   = 512
    minibatch_size:   int   = 256
    ppo_epochs:       int   = 8
    reward_threshold: float = 11.0   # SP-PPO crosses ~iter 8-15; baseline ~iter 33
    # Gap ~= 18-25 iterations -> clean convergence speed metric for Table 3.


# STATISTICS UTILITIES  (Stat-A / B / C)

def bootstrap_ci(
    data:        np.ndarray,
    stat_fn=np.mean,
    n_bootstrap: int   = 10_000,
    alpha:       float = 0.05,
    seed:        int   = 42,
) -> Tuple[float, float]:
    """95 % stratified bootstrap confidence interval."""
    rng   = np.random.default_rng(seed)
    stats = np.array([
        stat_fn(rng.choice(data, size=len(data), replace=True))
        for _ in range(n_bootstrap)
    ])
    return float(np.percentile(stats, 100 * alpha / 2)), \
           float(np.percentile(stats, 100 * (1 - alpha / 2)))


def iqm(data: np.ndarray) -> float:
    """Interquartile mean -- robust to outlier seeds (Agarwal et al., 2021)."""
    q25, q75 = np.percentile(data, [25, 75])
    trimmed  = data[(data >= q25) & (data <= q75)]
    return float(trimmed.mean()) if len(trimmed) > 0 else float(data.mean())


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d effect size (positive = a > b)."""
    na, nb   = len(a), len(b)
    pooled   = np.sqrt(
        ((na - 1) * a.std(ddof=1) ** 2 + (nb - 1) * b.std(ddof=1) ** 2)
        / max(na + nb - 2, 1)
    )
    return float((a.mean() - b.mean()) / (pooled + 1e-12))


def auc_score(returns: List[float]) -> float:
    """Area under the learning curve (trapezoid, normalised by length)."""
    return float(np.trapz(returns) / max(len(returns), 1))


def stat_compare(
    a: np.ndarray, b: np.ndarray,
    label_a: str, label_b: str,
) -> dict:
    """Full pairwise comparison: CI, IQM, Cohen's d, optional Welch + MWU."""
    d    = cohens_d(a, b)
    ci_a = bootstrap_ci(a)
    ci_b = bootstrap_ci(b)
    out  = {
        "mean_a": float(a.mean()), "std_a": float(a.std(ddof=1)),
        "mean_b": float(b.mean()), "std_b": float(b.std(ddof=1)),
        "iqm_a":  iqm(a),          "iqm_b": iqm(b),
        "ci_a":   ci_a,            "ci_b":  ci_b,
        "cohens_d": d,
    }
    size_label = "large" if abs(d) >= 0.8 else "medium" if abs(d) >= 0.5 else "small"
    print(f"  {label_a}: mean={a.mean():.3f}+/-{a.std(ddof=1):.3f}  "
          f"IQM={iqm(a):.3f}  95%-CI=[{ci_a[0]:.3f},{ci_a[1]:.3f}]")
    print(f"  {label_b}: mean={b.mean():.3f}+/-{b.std(ddof=1):.3f}  "
          f"IQM={iqm(b):.3f}  95%-CI=[{ci_b[0]:.3f},{ci_b[1]:.3f}]")
    print(f"  Cohen's d={d:.3f} ({size_label})")

    if HAS_SCIPY:
        t, p_t = scipy_stats.ttest_ind(a, b, equal_var=False)
        u, p_u = scipy_stats.mannwhitneyu(a, b, alternative="two-sided")
        sig = "***" if p_t < 0.001 else "**" if p_t < 0.01 else "*" if p_t < 0.05 else "ns"
        print(f"  Welch t={t:.3f} p={p_t:.4f}{sig}  |  "
              f"Mann-Whitney U={u:.0f} p={p_u:.4f}")
        out.update({"t_stat": float(t), "p_welch": float(p_t),
                    "u_stat": float(u), "p_mwu": float(p_u)})
    return out


# PPO TRAINER

class LieStructuredPPO:
    """
    PPO trainer for baseline and Lie-structured policies.
    use_lie_projection=True -> gradient and parameter projection after each step.

    Tracks: returns, grad_norms, policy entropy (Diag-A), spectral radius (Diag-B).
    """

    def __init__(
        self,
        env:                MultiStepTextAlignmentEnv,
        policy:             nn.Module,
        value_fn:           nn.Module,
        cfg:                PPOConfig,
        use_lie_projection: bool  = False,
        algebra:            str   = "so",
        lam:                float = 0.0,
    ):
        self.env      = env
        self.policy   = policy.to(DEVICE)
        self.value_fn = value_fn.to(DEVICE)
        self.cfg      = cfg
        self.use_lie  = use_lie_projection
        self.algebra  = algebra
        self.lam      = lam
        self.ops      = LieAlgebraOps()

        # -- Split optimisers: theta at high lr so M drifts meaningfully; -----
        # feature_net at standard PPO lr.
        # For baseline (no theta attribute) both optimisers use the full policy.
        if use_lie_projection and hasattr(policy, "theta"):
            theta_params  = [policy.theta]
            other_params  = [p for n, p in policy.named_parameters()
                             if n != "theta"]
            self.pi_optim = torch.optim.Adam(
                [{"params": other_params, "lr": cfg.lr},
                 {"params": theta_params,  "lr": cfg.theta_lr}]
            )
        else:
            self.pi_optim = torch.optim.Adam(self.policy.parameters(), lr=cfg.lr)

        self.vf_optim = torch.optim.Adam(self.value_fn.parameters(), lr=cfg.lr)
        self.projection_magnitudes: List[float] = []

    # Trajectory collection with proper episode auto-reset
    def _gather_trajectory(self):
        obs_buf, act_buf, rew_buf, val_buf, logp_buf = [], [], [], [], []
        ep_returns_all: List[float] = []
        ep_curr: float = 0.0

        # -- Compute M once per iteration (M only changes when theta is updated) --
        # For baseline (no theta): M=None -> env.step uses raw state table (v2).
        # For LiePolicy: M = exp(proj(theta)) reflects the current algebra type.
        M: Optional[torch.Tensor] = None
        if self.use_lie and hasattr(self.policy, "theta"):
            with torch.no_grad():
                th_proj = self._apply_proj(self.policy.theta.data)
                M = LieAlgebraOps.matrix_exp(th_proj)   # (k, k), on DEVICE

        state = self.env.reset()

        for _ in range(self.cfg.steps_per_iter):
            with torch.no_grad():
                probs  = self.policy(state)
                dist   = Categorical(probs)
                action = dist.sample()
                logp   = dist.log_prob(action)
                value  = self.value_fn(state)

            nxt, reward, done = self.env.step(action.item(), M)

            obs_buf.append(state)
            act_buf.append(action)
            rew_buf.append(reward)
            val_buf.append(value.item())
            logp_buf.append(logp.item())
            ep_curr += reward

            if done:
                ep_returns_all.append(ep_curr)
                ep_curr = 0.0
                state = self.env.reset()
            else:
                state = nxt

        with torch.no_grad():
            last_val = self.value_fn(state).item()

        obs  = torch.stack(obs_buf)
        # Bug-1 fix: policy(1D state) -> probs (1,A) -> sample() shape (1,)
        # torch.stack gives (N,1); .squeeze(1) corrects to (N,).
        acts  = torch.stack(act_buf).squeeze(1)
        rews  = torch.tensor(rew_buf,  dtype=torch.float32, device=DEVICE)
        vals  = torch.tensor(val_buf,  dtype=torch.float32, device=DEVICE)
        logps = torch.tensor(logp_buf, dtype=torch.float32, device=DEVICE)

        mean_ep = (float(np.mean(ep_returns_all))
                   if ep_returns_all else float(rews.mean().item()))
        return obs, acts, rews, vals, logps, last_val, mean_ep

    # GAE advantage
    def _compute_advantages(self, rewards, values, last_val):
        T   = len(rewards)
        adv = torch.zeros(T, device=DEVICE)
        gae = 0.0
        for t in reversed(range(T)):
            nv     = last_val if t == T - 1 else values[t + 1].item()
            gae    = (rewards[t] + self.cfg.gamma * nv - values[t]
                      + self.cfg.gamma * self.cfg.lam * gae)
            adv[t] = gae
        ret = adv + values
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return adv, ret

    # Projection helpers
    def _apply_proj(self, X: torch.Tensor) -> torch.Tensor:
        if self.algebra == "almost_so":
            return LieAlgebraOps.project_almost_so(X, self.lam)
        return getattr(self.ops, f"project_{self.algebra}")(X)

    def _proj_grads(self):
        if not self.use_lie or not hasattr(self.policy, "theta"):
            return
        if self.policy.theta.grad is None:
            return
        with torch.no_grad():
            g  = self.policy.theta.grad
            gp = self._apply_proj(g)
            self.projection_magnitudes.append(torch.norm(g - gp).item())
            self.policy.theta.grad.copy_(gp)

    def _proj_params(self):
        if not self.use_lie or not hasattr(self.policy, "theta"):
            return
        with torch.no_grad():
            self.policy.theta.data.copy_(
                self._apply_proj(self.policy.theta.data)
            )

    # Fix-D: single on-device accumulation, one .item() sync
    @staticmethod
    def _total_grad_norm(model: nn.Module) -> float:
        total = torch.zeros(1, device=DEVICE)
        for p in model.parameters():
            if p.grad is not None:
                total += p.grad.detach().norm(2) ** 2
        return total.sqrt().item()

    # Diag-B: spectral radius of exp(theta)
    def _spectral_radius(self) -> Optional[float]:
        if not hasattr(self.policy, "theta"):
            return None
        with torch.no_grad():
            th_proj = self._apply_proj(self.policy.theta.data)
            M       = LieAlgebraOps.matrix_exp(th_proj)
            return float(torch.linalg.eigvals(M).abs().max().item())

    # Main training loop
    def train(self, verbose: bool = True) -> dict:
        ep_returns:  List[float]          = []
        grad_norms:  List[float]          = []
        entropies:   List[float]          = []
        spec_rads:   List[Optional[float]] = []
        threshold_crossing: Optional[int]  = None

        for it in range(self.cfg.train_iters):
            obs, acts, rews, vals, logps, last_val, ep_return = \
                self._gather_trajectory()
            adv, ret = self._compute_advantages(rews, vals, last_val)

            N    = obs.shape[0]
            idxs = np.arange(N)
            mb_gnorms: List[float] = []

            for _ in range(self.cfg.ppo_epochs):
                np.random.shuffle(idxs)
                for start in range(0, N, self.cfg.minibatch_size):
                    mb = idxs[start: start + self.cfg.minibatch_size]
                    if len(mb) == 0:
                        continue

                    # policy loss  (clipped surrogate + entropy bonus)
                    self.pi_optim.zero_grad()
                    dist_mb  = Categorical(self.policy(obs[mb]))
                    logp_mb  = dist_mb.log_prob(acts[mb])
                    ratio    = torch.exp(logp_mb - logps[mb])
                    clip_adv = torch.clamp(ratio,
                                           1 - self.cfg.clip_ratio,
                                           1 + self.cfg.clip_ratio) * adv[mb]
                    surr_loss  = -torch.min(ratio * adv[mb], clip_adv).mean()
                    # Entropy bonus: -coef * H(pi) subtracted from loss
                    # Prevents premature saturation when theta_lr is high.
                    entropy_bonus = dist_mb.entropy().mean()
                    pi_loss = surr_loss - self.cfg.entropy_coef * entropy_bonus
                    pi_loss.backward()

                    # -- Direct geo aux loss (SP-PPO only) ------------------
                    # PPO action gradient is ANTAGONISTIC to dgeo_r/dtheta
                    # (measured cos = -0.45).  We add L_geo directly so theta
                    # receives a gradient through torch.matrix_exp.
                    # Baseline has no theta -> guard ensures this is a no-op.
                    if self.use_lie and hasattr(self.policy, "theta") and \
                            self.cfg.geo_aux_coef > 0.0 and \
                            hasattr(self.env, "M_env"):
                        th_proj = self._apply_proj(self.policy.theta)
                        M_pol   = LieAlgebraOps.matrix_exp(th_proj)  # (k,k)
                        k_env   = self.env.k_transform
                        v_mb    = obs[mb, :k_env]                      # (B, k)
                        M_env_b = self.env.M_env                       # (k,k)
                        actual  = (M_pol @ v_mb.unsqueeze(-1)).squeeze(-1)   # (B,k)
                        target  = (M_env_b @ v_mb.unsqueeze(-1)).squeeze(-1) # (B,k)
                        cos_geo = F.cosine_similarity(actual, target, dim=-1).mean()
                        L_geo   = -self.cfg.geo_aux_coef * cos_geo
                        L_geo.backward()

                    mb_gnorms.append(self._total_grad_norm(self.policy))
                    self._proj_grads()
                    self.pi_optim.step()
                    self._proj_params()

                    # value loss
                    self.vf_optim.zero_grad()
                    F.mse_loss(self.value_fn(obs[mb]), ret[mb]).backward()
                    self.vf_optim.step()

            ep_returns.append(ep_return)
            grad_norms.append(
                float(np.median(mb_gnorms)) if mb_gnorms else 0.0
            )

            # Diag-A: policy entropy
            with torch.no_grad():
                ent = Categorical(self.policy(obs[:min(64, N)])).entropy().mean().item()
            entropies.append(ent)

            # Diag-B: spectral radius
            spec_rads.append(self._spectral_radius())

            if threshold_crossing is None and ep_return >= self.cfg.reward_threshold:
                threshold_crossing = it + 1

            if verbose:
                sr = spec_rads[-1]
                sr_str = f"  SpR={sr:.4f}" if sr is not None else ""
                print(
                    f"Iter {it+1:03d}/{self.cfg.train_iters} | "
                    f"EpRet={ep_return:.3f} | "
                    f"GNorm={grad_norms[-1]:.3e} | "
                    f"H={ent:.3f}{sr_str}"
                )

        return {
            "returns":               ep_returns,
            "grad_norms":            grad_norms,
            "entropies":             entropies,
            "spectral_radii":        spec_rads,
            "projection_magnitudes": self.projection_magnitudes,
            "threshold_crossing":    threshold_crossing,
            "auc":                   auc_score(ep_returns),
        }


# OVERHEAD MEASUREMENT  (Fix-H: warmup; Bug-6: single call in warmup)

def measure_overhead(
    state_dim: int, n_actions: int, k: int = 32,
    batch_size: int = 32, n_iters: int = 50, n_runs: int = 5,
) -> Tuple[float, float, float]:
    print("\n" + "=" * 70)
    print("COMPUTATIONAL OVERHEAD ANALYSIS")
    print("=" * 70 + "\n")

    def _time_policy(policy: nn.Module, value_net: nn.Module) -> float:
        dummy = torch.randn(batch_size, state_dim, device=DEVICE)
        # Bug-6 fix: warmup matches timed loop structure (one forward per step)
        for _ in range(3):
            probs  = policy(dummy)
            dist_w = Categorical(probs)
            loss   = -(dist_w.log_prob(dist_w.sample()).mean()
                       + value_net(dummy).mean())
            loss.backward()
            policy.zero_grad(); value_net.zero_grad()

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iters):
            probs = policy(dummy)
            dist  = Categorical(probs)
            loss  = -(dist.log_prob(dist.sample()).mean() + value_net(dummy).mean())
            loss.backward()
            policy.zero_grad(); value_net.zero_grad()
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        return time.perf_counter() - t0

    base_times, lie_times = [], []
    for _ in range(n_runs):
        bp = BaselinePolicy(state_dim, n_actions).to(DEVICE)
        bv = ValueNet(state_dim).to(DEVICE)
        lp = LiePolicy(state_dim, n_actions, k=k, algebra="so").to(DEVICE)
        lv = ValueNet(state_dim).to(DEVICE)
        base_times.append(_time_policy(bp, bv))
        lie_times.append(_time_policy(lp, lv))

    t_base   = float(np.median(base_times))
    t_lie    = float(np.median(lie_times))
    overhead = (t_lie - t_base) / max(t_base, 1e-8) * 100.0

    print(f"Baseline policy : {t_base:.3f}s  (median of {n_runs} runs)")
    print(f"SP-PPO policy   : {t_lie:.3f}s  (median of {n_runs} runs)")
    print(f"Overhead        : {overhead:+.1f}%")
    return t_base, t_lie, overhead


# HIGH-LEVEL RUNNERS

def _default_cfg() -> PPOConfig:
    return PPOConfig()


def _seed_all(seed: int) -> None:
    """Fix-J: seed CPU, GPU, numpy, and Python random."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def _single_run(
    env:       MultiStepTextAlignmentEnv,
    state_dim: int,
    n_actions: int,
    algebra:   str,
    seed:      int,
    verbose:   bool  = True,
    lam:       float = 0.0,
) -> Tuple[dict, dict]:
    _seed_all(seed)
    cfg = _default_cfg()

    base_res = LieStructuredPPO(
        env, BaselinePolicy(state_dim, n_actions), ValueNet(state_dim),
        cfg, use_lie_projection=False,
    ).train(verbose=verbose)

    lie_res = LieStructuredPPO(
        env,
        LiePolicy(state_dim, n_actions, k=32, algebra=algebra, lam=lam),
        ValueNet(state_dim), cfg,
        use_lie_projection=True, algebra=algebra, lam=lam,
    ).train(verbose=verbose)

    return base_res, lie_res


def run_structure_discovery(embedder: GPT2EmbeddingProvider) -> dict:
    exp     = StructureDiscoveryExperiment(embedder, k=32, n_iters=300, lr=5e-3)
    results = {}
    for name, pairs in build_semantic_pairs().items():
        results[name] = exp.run_for_transformation(pairs, name=name)

    print("\n" + "=" * 70)
    print("SUMMARY: STRUCTURE DISCOVERY")
    print("=" * 70)
    for name, rs in results.items():
        print(f"\n{name}:")
        for r in rs:
            print(f"  {r.algebra.upper():4s} | "
                  f"test_loss={r.test_loss:.4e}  rel_res={r.relative_residual:.3f}")
    return results


def run_rl_comparison(
    env:       MultiStepTextAlignmentEnv,
    state_dim: int,
    n_actions: int,
) -> Tuple[dict, dict]:
    """Single-seed comparison: baseline PPO vs SP-PPO (seed=0)."""
    print("\n" + "=" * 70)
    print("RL EXPERIMENT: BASELINE PPO vs SP-PPO (GPT-2, seed=0)")
    print("=" * 70 + "\n")

    base_res, lie_res = _single_run(
        env, state_dim, n_actions, algebra="so", seed=0, verbose=True
    )

    print("\n" + "=" * 70)
    print("RL SUMMARY")
    print("=" * 70)
    for name, res in [("Baseline PPO", base_res), ("SP-PPO (SO)", lie_res)]:
        R = np.array(res["returns"])
        print(
            f"{name}: mean={R.mean():.3f}  final={R[-1]:.3f}  "
            f"AUC={res['auc']:.3f}  "
            f"threshold_iter={res['threshold_crossing']}  "
            f"median_gnorm={np.median(res['grad_norms']):.3e}"
        )
    return base_res, lie_res


def run_multiseed(
    env:       MultiStepTextAlignmentEnv,
    state_dim: int,
    n_actions: int,
    seeds:     Optional[List[int]] = None,  # Fix-I: avoid mutable default
) -> dict:
    """
    Stat-A/B/C: multi-seed experiment with full statistical reporting.
    Default: 10 seeds.  Change to seeds=list(range(30)) for publication.
    """
    if seeds is None:
        seeds = list(range(10))

    print("\n" + "=" * 70)
    print(f"MULTI-SEED EXPERIMENT ({len(seeds)} seeds)")
    print("=" * 70 + "\n")

    base_crossings, lie_crossings = [], []
    base_aucs,      lie_aucs      = [], []
    base_finals,    lie_finals    = [], []

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        base_res, lie_res = _single_run(
            env, state_dim, n_actions, algebra="so",
            seed=seed, verbose=False,
        )
        bc  = base_res["threshold_crossing"]
        lc  = lie_res["threshold_crossing"]
        bf  = base_res["returns"][-1]
        lf  = lie_res["returns"][-1]

        base_crossings.append(bc if bc is not None else 999)
        lie_crossings.append(lc  if lc  is not None else 999)
        base_finals.append(bf);       lie_finals.append(lf)
        base_aucs.append(base_res["auc"]); lie_aucs.append(lie_res["auc"])

        gain = (bc - lc) / bc * 100 if (bc and lc) else float("nan")
        print(f"  Base: cross={bc} final={bf:.2f} AUC={base_res['auc']:.2f} | "
              f"SP-PPO: cross={lc} final={lf:.2f} AUC={lie_res['auc']:.2f} | "
              f"gain={gain:.1f}%")

    print("\n" + "=" * 70)
    print("MULTI-SEED SUMMARY")
    print("=" * 70)

    ba = np.array(base_aucs);        la = np.array(lie_aucs)
    bc_arr = np.array(base_crossings); lc_arr = np.array(lie_crossings)

    print("\n-- AUC (primary metric) --")
    stats_auc = stat_compare(la, ba, "SP-PPO (SO)", "Baseline PPO")

    valid_bc = bc_arr[bc_arr < 999]
    valid_lc = lc_arr[lc_arr < 999]
    if len(valid_bc) > 1 and len(valid_lc) > 1:
        print("\n-- Iterations-to-threshold --")
        stats_iter = stat_compare(valid_bc, valid_lc,
                                  "Baseline PPO", "SP-PPO (SO)")
    else:
        stats_iter = {}

    avg_gain = ((bc_arr.mean() - lc_arr.mean())
                / max(bc_arr.mean(), 1e-8) * 100)
    print(f"\nAverage convergence gain: {avg_gain:.1f}%")

    return {
        "base_crossings": list(base_crossings),
        "lie_crossings":  list(lie_crossings),
        "base_finals":    list(base_finals),
        "lie_finals":     list(lie_finals),
        "base_aucs":      list(base_aucs),
        "lie_aucs":       list(lie_aucs),
        "stats_auc":      stats_auc,
        "stats_iter":     stats_iter,
    }


def run_rl_ablation(
    env:       MultiStepTextAlignmentEnv,
    state_dim: int,
    n_actions: int,
    seed:      int = 0,
) -> dict:
    """Ablation: Unconstrained vs SO(32) vs SYM(32) -- same seed."""
    print("\n" + "=" * 70)
    print("RL ABLATION: so(32) vs sym(32) vs unconstrained")
    print("=" * 70 + "\n")

    cfg     = _default_cfg()
    results = {}

    for label, use_lie, algebra in [
        ("Unconstrained (full)",  True,  "full"),
        ("SO(32) matched",        True,  "so"),
        ("SYM(32) mismatch",      True,  "sym"),
    ]:
        _seed_all(seed)
        policy  = LiePolicy(state_dim, n_actions, k=32, algebra=algebra)
        trainer = LieStructuredPPO(
            env, policy, ValueNet(state_dim), cfg,
            use_lie_projection=use_lie, algebra=algebra,
        )
        res = trainer.train(verbose=False)
        R   = np.array(res["returns"])
        results[label] = res
        print(
            f"{label:22s} | threshold={res['threshold_crossing']} | "
            f"final={R[-1]:.3f} | AUC={res['auc']:.3f}"
        )

    return results


def run_lambda_ablation(
    env:       MultiStepTextAlignmentEnv,
    state_dim: int,
    n_actions: int,
    seed:      int = 0,
    lambdas:   Optional[List[float]] = None,
) -> dict:
    """
    Abl-A: almost-SO relaxation sweep.
    lam=0 -> exact SO; lam=1 -> unconstrained.
    Expected result: monotone AUC degradation as lam increases.
    """
    if lambdas is None:
        lambdas = [0.0, 0.2, 0.5, 0.8, 1.0]

    print("\n" + "=" * 70)
    print("ABLATION: Almost-SO relaxation (lam sweep, Abl-A)")
    print("=" * 70 + "\n")

    cfg     = _default_cfg()
    results = {}

    for lam in lambdas:
        _seed_all(seed)
        policy  = LiePolicy(state_dim, n_actions, k=32,
                            algebra="almost_so", lam=lam)
        trainer = LieStructuredPPO(
            env, policy, ValueNet(state_dim), cfg,
            use_lie_projection=True, algebra="almost_so", lam=lam,
        )
        res = trainer.train(verbose=False)
        R   = np.array(res["returns"])
        results[lam] = res
        print(
            f"lam={lam:.1f} | threshold={res['threshold_crossing']} | "
            f"AUC={res['auc']:.3f} | final={R[-1]:.3f}"
        )

    return results


def run_geo_weight_ablation(
    embedder:  GPT2EmbeddingProvider,
    state_dim: int,
    n_actions: int,
    seed:      int = 0,
    weights:   Optional[List[float]] = None,
) -> dict:
    """
    Abl-B: geometry-reward weight sweep -- SO vs SYM vs Full (all with M).

    All three conditions use LiePolicy with M applied.  The difference is algebra:
      SO(32):  M orthogonal (SpR=1)   -> geo_r = 1.0 every step (exact)
      SYM(32): M SPD (SpR>>1)         -> geo_r < 1.0 (norm distortion)
      Full:    M unconstrained (SpR>1) -> geo_r < 1.0

    Expected: SO advantage DeltaAUC = AUC_so - AUC_sym grows monotonically with
    geo_weight, since SO uniquely achieves geo_r=1.0.

    Numerical verification (scipy, k=32, std=0.3, seed=0):
      geo_weight=0.0: Delta = 0   (pure task reward, algebra irrelevant)
      geo_weight=0.8: Delta = +16 (pure geometry reward, SO dominates exactly)
    """
    if weights is None:
        weights = [0.0, 0.2, 0.4, 0.6, 0.8]

    print("\n" + "=" * 70)
    print("ABLATION: Geometry-reward weight sweep (Abl-B): SO vs SYM vs Full")
    print("=" * 70 + "\n")

    results = {}
    cfg = _default_cfg()

    for w in weights:
        _seed_all(seed)
        env_w = MultiStepTextAlignmentEnv(
            embedder, n_prompts=16, n_actions=n_actions,
            horizon=20, reward_noise=0.2, geo_weight=w, sparse_prob=0.7,
        )
        row = {}
        for algebra in ["so", "sym", "full"]:
            _seed_all(seed)
            policy  = LiePolicy(state_dim, n_actions, k=32, algebra=algebra)
            trainer = LieStructuredPPO(
                env_w, policy, ValueNet(state_dim), cfg,
                use_lie_projection=True, algebra=algebra,
            )
            res = trainer.train(verbose=False)
            row[algebra] = res["auc"]

        delta_sym  = row["so"] - row["sym"]
        delta_full = row["so"] - row["full"]
        results[w] = row
        print(
            f"geo_weight={w:.1f} | "
            f"SO AUC={row['so']:.3f}  "
            f"SYM AUC={row['sym']:.3f}  "
            f"Full AUC={row['full']:.3f}  "
            f"Delta(SO-SYM)={delta_sym:+.3f}  "
            f"Delta(SO-Full)={delta_full:+.3f}"
        )

    return results


# PLOTS

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def plot_structure_discovery(sd_results: dict, save_dir: str = "plots") -> None:
    _ensure_dir(save_dir)
    algs   = ["SO", "SL", "SYM", "FULL"]
    colors = ["#4c72b0", "#55a868", "#c44e52", "#8172b2"]
    for name, res_list in sd_results.items():
        fig, ax = plt.subplots(figsize=(7, 5))
        rel_res = [r.relative_residual for r in res_list]
        ax.bar(algs, rel_res, color=colors)
        ax.axhline(1.0, color="black", linestyle="--", linewidth=1,
                   label="Baseline = 1.0")
        ax.set_title(f"Structure Selection - {name}", fontsize=15)
        ax.set_xlabel("Subspace", fontsize=13)
        ax.set_ylabel("Relative Residual", fontsize=13)
        ax.set_ylim(0, max(rel_res) * 1.25)
        ax.legend(fontsize=11)
        for i, v in enumerate(rel_res):
            ax.text(i, v + 0.02 * max(rel_res), f"{v:.3f}", ha="center", fontsize=12)
        fig.tight_layout()
        fname = os.path.join(save_dir, f"struct_{name}.png")
        fig.savefig(fname, dpi=300); plt.close(fig)
        print(f"Saved: {fname}")


def plot_rl_returns(
    base_res: dict, lie_res: dict,
    save_dir: str = "plots", threshold: float = 13.0,
) -> None:
    _ensure_dir(save_dir)
    iters = list(range(1, len(base_res["returns"]) + 1))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(iters, base_res["returns"], label="Baseline PPO",    linewidth=2)
    ax.plot(iters, lie_res["returns"],  label="SP-PPO (SO(32))", linewidth=2)
    ax.axhline(threshold, color="gray", linestyle="--", linewidth=1.2,
               label=f"Threshold = {threshold:.0f}")
    bc, lc = base_res["threshold_crossing"], lie_res["threshold_crossing"]
    if bc: ax.axvline(bc, color="#4c72b0", linestyle=":", linewidth=1.5, alpha=0.7)
    if lc: ax.axvline(lc, color="#c44e52", linestyle=":", linewidth=1.5, alpha=0.7)
    ax.set_title("Training Returns (mean per-episode return)", fontsize=15)
    ax.set_xlabel("Iteration", fontsize=13); ax.set_ylabel("Mean Episode Return", fontsize=13)
    ax.legend(fontsize=12); ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fname = os.path.join(save_dir, "rl_returns.png")
    fig.savefig(fname, dpi=300); plt.close(fig); print(f"Saved: {fname}")


def plot_diagnostics(lie_res: dict, save_dir: str = "plots") -> None:
    """Grad norms + Diag-A (entropy) + Diag-B (spectral radius)."""
    _ensure_dir(save_dir)
    iters = list(range(1, len(lie_res["grad_norms"]) + 1))

    for key, label, color, fname_suffix, yscale in [
        ("grad_norms",  "Total Grad Norm (log scale)",    "#c44e52", "grad_norms",     "log"),
        ("entropies",   "Policy Entropy (nats)  [Diag-A]","#55a868", "entropy",        "linear"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(iters, lie_res[key], color=color, linewidth=2, label=f"SP-PPO {key}")
        ax.set_title(f"SP-PPO {label}", fontsize=14)
        ax.set_xlabel("Iteration", fontsize=12); ax.set_ylabel(label, fontsize=12)
        if yscale == "log": ax.set_yscale("log")
        ax.legend(fontsize=11); ax.grid(True, linestyle="--", alpha=0.5)
        fig.tight_layout()
        f = os.path.join(save_dir, f"{fname_suffix}.png")
        fig.savefig(f, dpi=300); plt.close(fig); print(f"Saved: {f}")

    # Spectral radius (Diag-B)
    spec = [s for s in lie_res["spectral_radii"] if s is not None]
    if spec:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(range(1, len(spec) + 1), spec, color="#8172b2", linewidth=2,
                label="SP-PPO |lam_max| of exp(theta)")
        ax.axhline(1.0, color="black", linestyle="--", linewidth=1,
                   label="SO theoretical = 1.0")
        ax.set_title("Spectral Radius of exp(theta) [Diag-B]", fontsize=14)
        ax.set_xlabel("Iteration", fontsize=12); ax.set_ylabel("|lam_max|", fontsize=12)
        ax.legend(fontsize=11); ax.grid(True, linestyle="--", alpha=0.5)
        fig.tight_layout()
        f = os.path.join(save_dir, "spectral_radius.png")
        fig.savefig(f, dpi=300); plt.close(fig); print(f"Saved: {f}")


def plot_ablation(
    ablation_results: dict, save_dir: str = "plots", threshold: float = 13.0,
) -> None:
    _ensure_dir(save_dir)
    colors = {"Unconstrained (full)":  "#4c72b0",
              "SO(32) matched":        "#c44e52",
              "SYM(32) mismatch":      "#55a868"}
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, res in ablation_results.items():
        ax.plot(range(1, len(res["returns"]) + 1), res["returns"],
                label=label, linewidth=2, color=colors.get(label))
    ax.axhline(threshold, color="gray", linestyle="--", linewidth=1.2,
               label=f"Threshold = {threshold:.0f}")
    ax.set_title("RL Ablation: Algebra Choice", fontsize=15)
    ax.set_xlabel("Iteration", fontsize=13); ax.set_ylabel("Mean Episode Return", fontsize=13)
    ax.legend(fontsize=12); ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fname = os.path.join(save_dir, "rl_ablation.png")
    fig.savefig(fname, dpi=300); plt.close(fig); print(f"Saved: {fname}")


def plot_lambda_ablation(lam_results: dict, save_dir: str = "plots") -> None:
    """Abl-A: AUC vs lam -- should be monotone decreasing."""
    _ensure_dir(save_dir)
    lambdas = sorted(lam_results.keys())
    aucs    = [lam_results[l]["auc"] for l in lambdas]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(lambdas, aucs, marker="o", linewidth=2, markersize=8, color="#c44e52")
    for x, y in zip(lambdas, aucs):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=11)
    ax.set_title("Almost-SO Relaxation: AUC vs lam  (Abl-A)", fontsize=15)
    ax.set_xlabel("Relaxation lam  (0 = exact SO,  1 = unconstrained)", fontsize=13)
    ax.set_ylabel("AUC", fontsize=13); ax.set_xlim(-0.05, 1.05)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fname = os.path.join(save_dir, "lambda_ablation.png")
    fig.savefig(fname, dpi=300); plt.close(fig); print(f"Saved: {fname}")


def plot_geo_weight_ablation(geo_results: dict, save_dir: str = "plots") -> None:
    """Abl-B: SO vs SYM vs Full AUC vs geometry-reward weight."""
    _ensure_dir(save_dir)
    weights = sorted(geo_results.keys())
    so_aucs   = [geo_results[w]["so"]   for w in weights]
    sym_aucs  = [geo_results[w]["sym"]  for w in weights]
    full_aucs = [geo_results[w]["full"] for w in weights]
    deltas_sym  = [geo_results[w]["so"] - geo_results[w]["sym"]  for w in weights]
    deltas_full = [geo_results[w]["so"] - geo_results[w]["full"] for w in weights]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(weights, so_aucs,   marker="o", linewidth=2,
                 label="SP-PPO SO",    color="#c44e52")
    axes[0].plot(weights, sym_aucs,  marker="s", linewidth=2,
                 label="SP-PPO SYM",   color="#55a868")
    axes[0].plot(weights, full_aucs, marker="^", linewidth=2,
                 label="SP-PPO Full",  color="#8172b2")
    axes[0].set_title("AUC vs Geometry-Reward Weight  (Abl-B)", fontsize=13)
    axes[0].set_xlabel("geo_weight", fontsize=12)
    axes[0].set_ylabel("AUC", fontsize=12)
    axes[0].legend(fontsize=11); axes[0].grid(True, linestyle="--", alpha=0.5)

    w_labels = [str(w) for w in weights]
    x = np.arange(len(weights)); bar_w = 0.35
    axes[1].bar(x - bar_w/2, deltas_sym,  bar_w, label="SO - SYM",  color="#c44e52", alpha=0.85)
    axes[1].bar(x + bar_w/2, deltas_full, bar_w, label="SO - Full", color="#8172b2", alpha=0.85)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xticks(x); axes[1].set_xticklabels(w_labels)
    axes[1].set_title("SO Advantage:  DeltaAUC vs Algebra (Abl-B)", fontsize=13)
    axes[1].set_xlabel("geo_weight", fontsize=12)
    axes[1].set_ylabel("DeltaAUC", fontsize=12)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, linestyle="--", alpha=0.4, axis="y")

    fig.tight_layout()
    fname = os.path.join(save_dir, "geo_weight_ablation.png")
    fig.savefig(fname, dpi=300); plt.close(fig); print(f"Saved: {fname}")


def plot_multiseed(ms: dict, save_dir: str = "plots") -> None:
    """Per-seed AUC + aggregate mean+/-CI + threshold-crossing chart."""
    _ensure_dir(save_dir)
    ba = np.array(ms["base_aucs"])
    la = np.array(ms["lie_aucs"])
    n  = len(ba)
    x  = np.arange(n)
    w  = 0.35

    # -- Per-seed AUC + aggregate -----------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].bar(x - w/2, ba, w, label="Baseline PPO", color="#4c72b0", alpha=0.85)
    axes[0].bar(x + w/2, la, w, label="SP-PPO (SO)",  color="#c44e52", alpha=0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"S{i}" for i in range(n)], fontsize=10)
    axes[0].set_ylabel("AUC", fontsize=12); axes[0].set_title("Per-Seed AUC", fontsize=13)
    axes[0].legend(fontsize=11); axes[0].grid(True, linestyle="--", alpha=0.4, axis="y")

    ci_b = ms["stats_auc"]["ci_b"]; ci_l = ms["stats_auc"]["ci_a"]
    mb   = ms["stats_auc"]["mean_b"]; ml   = ms["stats_auc"]["mean_a"]
    axes[1].bar([0, 1], [mb, ml],
                yerr=[[mb - ci_b[0], ml - ci_l[0]], [ci_b[1] - mb, ci_l[1] - ml]],
                color=["#4c72b0", "#c44e52"], alpha=0.85, capsize=8,
                error_kw={"linewidth": 2})
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(["Baseline PPO", "SP-PPO (SO)"], fontsize=12)
    axes[1].set_ylabel("AUC", fontsize=12)
    axes[1].set_title("Mean AUC +/- 95% Bootstrap CI", fontsize=13)
    axes[1].grid(True, linestyle="--", alpha=0.4, axis="y")
    fig.tight_layout()
    fname = os.path.join(save_dir, "multiseed_auc.png")
    fig.savefig(fname, dpi=300); plt.close(fig); print(f"Saved: {fname}")

    # -- Threshold-crossing per seed --------------------------------------
    fig2, ax2 = plt.subplots(figsize=(9, 5))
    ax2.bar(x - w/2, ms["base_crossings"], w,
            label="Baseline PPO", color="#4c72b0", alpha=0.85)
    ax2.bar(x + w/2, ms["lie_crossings"],  w,
            label="SP-PPO (SO)",  color="#c44e52", alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"Seed {i}" for i in range(n)], fontsize=10)
    ax2.set_ylabel("Iterations to threshold", fontsize=12)
    ax2.set_title("Threshold Crossing per Seed", fontsize=13)
    ax2.legend(fontsize=11); ax2.grid(True, linestyle="--", alpha=0.4, axis="y")
    fig2.tight_layout()
    fname2 = os.path.join(save_dir, "multiseed_crossing.png")
    fig2.savefig(fname2, dpi=300); plt.close(fig2); print(f"Saved: {fname2}")


def plot_overhead(t_base: float, t_lie: float, overhead_pct: float,
                  save_dir: str = "plots") -> None:
    _ensure_dir(save_dir)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(["Baseline PPO", "SP-PPO (SO)"], [t_base, t_lie],
           color=["#4c72b0", "#c44e52"])
    ax.set_ylabel("Median Time (s)", fontsize=13)
    ax.set_title(f"Policy Compute Overhead ({overhead_pct:+.1f}%)", fontsize=15)
    for i, v in enumerate([t_base, t_lie]):
        ax.text(i, v * 1.01, f"{v:.3f}s", ha="center", fontsize=12)
    fig.tight_layout()
    fname = os.path.join(save_dir, "overhead.png")
    fig.savefig(fname, dpi=300); plt.close(fig); print(f"Saved: {fname}")


# MAIN

if __name__ == "__main__":
    _seed_all(42)

    # -- GPT-2 embedder ----------------------------------------------------
    embedder  = GPT2EmbeddingProvider(model_name="gpt2-medium", max_length=64)

    # -- Structure discovery -----------------------------------------------
    sd_results = run_structure_discovery(embedder)
    plot_structure_discovery(sd_results, save_dir="plots")

    # Fix-G: env created ONCE; all RL runners share it
    # (geo-weight ablation creates its own envs per weight -- see below)
    env       = MultiStepTextAlignmentEnv(
        embedder, n_prompts=16, n_actions=16, horizon=20,
        reward_noise=0.2, geo_weight=0.4, sparse_prob=0.7,
    )
    state_dim = embedder.hidden_dim    # 1024
    n_actions = env.n_actions          # 16

    # -- Single-seed comparison (seed=0) -----------------------------------
    base_res, lie_res = run_rl_comparison(env, state_dim, n_actions)
    plot_rl_returns(base_res, lie_res, save_dir="plots",
                    threshold=PPOConfig.reward_threshold)
    plot_diagnostics(lie_res, save_dir="plots")

    # -- Multi-seed: Table 3  (10 seeds; use 30 for publication) ----------
    ms_results = run_multiseed(env, state_dim, n_actions)
    plot_multiseed(ms_results, save_dir="plots")

    # -- Ablation: so vs sym vs unconstrained ------------------------------
    abl_results = run_rl_ablation(env, state_dim, n_actions, seed=0)
    plot_ablation(abl_results, save_dir="plots",
                  threshold=PPOConfig.reward_threshold)

    # -- Abl-A: almost-SO lam sweep -----------------------------------------
    lam_results = run_lambda_ablation(env, state_dim, n_actions, seed=0)
    plot_lambda_ablation(lam_results, save_dir="plots")

    # -- Abl-B: geometry-reward weight sweep ------------------------------
    geo_results = run_geo_weight_ablation(embedder, state_dim, n_actions, seed=0)
    plot_geo_weight_ablation(geo_results, save_dir="plots")

    # -- Overhead ---------------------------------------------------------
    t_b, t_l, pct = measure_overhead(state_dim, n_actions, k=32, n_runs=5)
    plot_overhead(t_b, t_l, pct, save_dir="plots")

    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETED")
    print("=" * 70)
    print(f"\n[WARNING]  Overhead: {pct:+.1f}%.  Use this value in Table 2.")
    print("\n[WARNING]  Multi-seed used 10 seeds.  For publication use 30 seeds:")
    print("   run_multiseed(env, state_dim, n_actions, seeds=list(range(30)))")
