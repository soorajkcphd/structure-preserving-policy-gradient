#!/usr/bin/env python3
"""
Task 3: Sentiment Steering -- Falsification Test (no engineered geometry).

Tests whether so(32) constraint helps when the environment has no latent
rotational structure. Reward = cosine similarity with a sentiment direction
derived from GPT-2 embeddings. M_policy is accepted but not used.

Usage:
  python run_sentiment_task3.py                  # full 10-seed run
  python run_sentiment_task3.py --seeds 0 1 2    # quick test
"""

import argparse
import contextlib
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from transformers import AutoTokenizer, AutoModelForCausalLM

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("[WARNING]  scipy not installed -- tests skipped. pip install scipy")


# DEVICE

def _select_device() -> torch.device:
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        return dev
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = _select_device()


# GPT-2 EMBEDDING PROVIDER (reused from main.py)

class GPT2EmbeddingProvider:
    def __init__(self, model_name: str = "gpt2-medium", max_length: int = 64):
        self.max_length = max_length
        print(f"Loading {model_name}...")
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
        print(f"Hidden dim = {self.hidden_dim}")

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


# SENTIMENT DIRECTION (derived from GPT-2 embeddings -- NO geometry)

def compute_sentiment_direction(embedder) -> torch.Tensor:
    """
    Fit a linear sentiment direction from GPT-2 embeddings.
    Returns a unit vector in R^d that points from negative to positive.
    """
    positive = [
        "This is wonderful and delightful, I am very happy.",
        "Excellent work, outstanding results, truly impressive.",
        "I love this, it makes me feel great and optimistic.",
        "The experience was fantastic and deeply satisfying.",
        "What a beautiful day, everything is going perfectly.",
        "I am thrilled with the outcome, it exceeded expectations.",
        "This product is amazing, best purchase I ever made.",
        "The team did a brilliant job, very proud of everyone.",
        "Absolutely phenomenal, I cannot recommend this enough.",
        "Life is beautiful and full of joy and wonder.",
    ]
    negative = [
        "This is terrible and awful, I am very upset.",
        "Horrible work, dreadful results, truly disappointing.",
        "I hate this, it makes me feel bad and pessimistic.",
        "The experience was awful and deeply frustrating.",
        "What a miserable day, everything is going wrong.",
        "I am disgusted with the outcome, it fell short completely.",
        "This product is garbage, worst purchase I ever made.",
        "The team did a terrible job, very disappointed in everyone.",
        "Absolutely dreadful, I would never recommend this.",
        "Life is painful and full of sadness and despair.",
    ]

    print("Computing sentiment direction from GPT-2 embeddings...")
    pos_embs = torch.stack([embedder.embed_sentence(s) for s in positive])
    neg_embs = torch.stack([embedder.embed_sentence(s) for s in negative])

    pos_mean = pos_embs.mean(dim=0)
    neg_mean = neg_embs.mean(dim=0)
    direction = F.normalize(pos_mean - neg_mean, dim=0)

    # Verify separation
    pos_scores = (pos_embs @ direction).cpu().numpy()
    neg_scores = (neg_embs @ direction).cpu().numpy()
    print(f"  Positive scores: {pos_scores.mean():.3f} +/- {pos_scores.std():.3f}")
    print(f"  Negative scores: {neg_scores.mean():.3f} +/- {neg_scores.std():.3f}")
    print(f"  Separation: {pos_scores.mean() - neg_scores.mean():.3f}")

    return direction


# SENTIMENT STEERING ENVIRONMENT (NO geometry, NO M_env, NO M_policy in reward)

class SentimentSteeringEnv:
    """
    Environment with no engineered geometric structure.

    No M_env, no geometry reward, M not used in transitions or rewards.
    Reward = cosine similarity with a fixed sentiment direction.
    Tests whether so(32) constraint helps without geometric structure.
    """

    # Per-action sentiment biases (8 actions):
    #   Actions 0-1: move toward positive sentiment
    #   Actions 2-3: neutral (random perturbation)
    #   Actions 4-5: move toward negative sentiment
    #   Actions 6-7: large random perturbation (high variance)
    _ACTION_SENTIMENT_BIAS: List[float] = [
        +0.15, +0.10,    # positive-leaning
         0.00,  0.00,    # neutral
        -0.10, -0.15,    # negative-leaning
         0.00,  0.00,    # high-variance neutral
    ]
    _ACTION_NOISE_SCALE: List[float] = [
        0.10, 0.10,
        0.15, 0.15,
        0.10, 0.10,
        0.40, 0.40,
    ]

    def __init__(
        self,
        embedder:            GPT2EmbeddingProvider,
        sentiment_direction: torch.Tensor,
        n_prompts:           int   = 20,
        n_actions:           int   = 8,
        horizon:             int   = 10,
        reward_noise:        float = 0.1,
        sparse_prob:         float = 0.5,
    ):
        assert n_actions == 8, "Sentiment env uses 8 actions."
        self.n_actions           = n_actions
        self.horizon             = horizon
        self.reward_noise        = reward_noise
        self.sparse_prob         = sparse_prob
        self.sentiment_direction = sentiment_direction.to(DEVICE)

        # -- Prompts with mixed sentiment potential ----------------------
        self.prompts = [
            "I went to the new restaurant and the food was",
            "The latest software update has made my computer",
            "After the long meeting, my manager told me that",
            "The weather today is making me feel quite",
            "My friend surprised me with a gift that was",
            "The book I just finished reading was absolutely",
            "When I arrived at the hotel, the room was",
            "The customer service representative was very",
            "I tried the new workout routine and it was",
            "The concert last night was an experience that",
            "My neighbor has been doing something that is",
            "The new policy at work has everyone feeling",
            "After tasting the dessert, I thought it was",
            "The documentary I watched was deeply",
            "The job interview went surprisingly",
            "My morning commute today was unusually",
            "The children at the park were being very",
            "The repair service was disappointingly",
            "After the vacation, I came back feeling",
            "The presentation my colleague gave was",
        ][:n_prompts]

        P = len(self.prompts)
        D = embedder.hidden_dim

        # -- Pre-compute prompt embeddings ------------------------------
        print(f"Pre-computing embeddings for {P} prompts...")
        self.prompt_embs = torch.zeros(P, D, device=DEVICE)
        for i, p in enumerate(self.prompts):
            self.prompt_embs[i] = embedder.embed_sentence(p)

        # -- Fixed random action directions (seeded) -------------------
        gen = torch.Generator(device="cpu")
        gen.manual_seed(2024)
        raw_dirs = torch.randn(D, n_actions, generator=gen).to(DEVICE)
        self.action_dirs = F.normalize(raw_dirs, dim=0)

        # -- Sentiment biases per action -------------------------------
        self.sentiment_bias = torch.tensor(
            self._ACTION_SENTIMENT_BIAS, device=DEVICE
        )
        self.noise_scale = torch.tensor(
            self._ACTION_NOISE_SCALE, device=DEVICE
        )

        # -- Pre-compute all (P, H, A) state embeddings ---------------
        # State perturbation: base + action_dir * scale + sentiment_bias * sent_dir
        self.state_embs = torch.zeros(P, horizon, n_actions, D, device=DEVICE)
        for p in range(P):
            for t in range(horizon):
                turn_scale = 1.0 + 0.03 * t   # mild growth with turn
                for a in range(n_actions):
                    # Perturbation = random direction + sentiment-biased component
                    pert = (self.noise_scale[a] * turn_scale * self.action_dirs[:, a]
                            + self.sentiment_bias[a] * self.sentiment_direction)
                    raw = self.prompt_embs[p] + pert
                    self.state_embs[p, t, a] = F.normalize(raw, dim=0)

        print("Pre-computation done.")

        # -- Episode state ---------------------------------------------
        self._idx:  int = 0
        self._turn: int = 0

    # NOTE: No M_env attribute -> geo_aux_loss is automatically disabled
    #       in LieStructuredPPO (the guard checks hasattr(env, "M_env"))

    def reset(self) -> torch.Tensor:
        self._idx  = random.randint(0, len(self.prompts) - 1)
        self._turn = 0
        return self.prompt_embs[self._idx].clone()

    def step(
        self,
        action: int,
        M: Optional[torch.Tensor] = None,  # accepted but not used
    ) -> Tuple[torch.Tensor, float, bool]:
        """
        One environment step.

        M is accepted for interface compatibility with LieStructuredPPO
        but is not used. Neither the state transition nor the
        reward depends on M_policy. Both methods face the same MDP.
        """
        t        = self._turn
        terminal = (t == self.horizon - 1)

        # State transition: look up pre-computed embedding
        # M is not applied to the state.
        nxt = self.state_embs[self._idx, t, action].clone()

        # -- Sentiment reward (external, M-independent) ----------------
        # Reward = how aligned the resulting state is with the sentiment
        # direction. This is a fixed linear function of the state --
        # no geometry, no M_env, no M_policy.
        sent_score = (nxt @ self.sentiment_direction).item()

        # Scale to [0, 1] range (cosine similarity is in [-1, 1])
        reward = (sent_score + 1.0) / 2.0

        # Sparse stochastic reward (matching Task 1 difficulty)
        if (not terminal) and (np.random.random() < self.sparse_prob):
            reward = 0.0

        # Reward noise
        if self.reward_noise > 0.0 and reward > 0.0:
            reward = max(0.0, reward + float(self.reward_noise * np.random.randn()))

        self._turn += 1
        return nxt, float(reward), terminal


# LIE ALGEBRA OPERATIONS (from main.py)

class LieAlgebraOps:
    @staticmethod
    def project_so(X: torch.Tensor) -> torch.Tensor:
        return 0.5 * (X - X.T)

    @staticmethod
    def project_sl(X: torch.Tensor) -> torch.Tensor:
        return X - (torch.trace(X) / X.shape[0]) * torch.eye(
            X.shape[0], device=X.device
        )

    @staticmethod
    def project_sym(X: torch.Tensor) -> torch.Tensor:
        return 0.5 * (X + X.T)

    @staticmethod
    def project_full(X: torch.Tensor) -> torch.Tensor:
        return X

    @staticmethod
    def matrix_exp(X: torch.Tensor) -> torch.Tensor:
        return torch.matrix_exp(X)


# POLICIES (from main.py -- identical)

class LiePolicy(nn.Module):
    def __init__(self, state_dim: int, n_actions: int, k: int,
                 algebra: str = "so", hidden_dim: int = 256):
        super().__init__()
        self.n_actions = n_actions
        self.k         = k
        self.algebra   = algebra
        self.ops       = LieAlgebraOps()
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, n_actions * k * k),
        )
        self.theta = nn.Parameter(torch.randn(k, k) * 0.3)
        self.bias  = nn.Parameter(torch.zeros(n_actions))

    def _proj(self, X):
        return getattr(self.ops, f"project_{self.algebra}")(X)

    def forward(self, state):
        if state.dim() == 1:
            state = state.unsqueeze(0)
        B = state.shape[0]
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

    def forward(self, state):
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

    def forward(self, state):
        if state.dim() == 1:
            state = state.unsqueeze(0)
        return self.net(state).squeeze(-1)


# PPO CONFIG (adapted: no geometry, moderate horizon)

@dataclass
class PPOConfig:
    gamma:            float = 0.99
    lam:              float = 0.95
    clip_ratio:       float = 0.2
    lr:               float = 3e-4
    theta_lr:         float = 0.01    # lower than Task 1 (no geo signal to drive theta)
    entropy_coef:     float = 0.03
    geo_aux_coef:     float = 0.0     # No geometric auxiliary loss
    train_iters:      int   = 80      # more iters (slower learning expected)
    steps_per_iter:   int   = 256
    minibatch_size:   int   = 128
    ppo_epochs:       int   = 6
    reward_threshold: float = 5.0


# PPO TRAINER (from main.py -- identical except geo_aux_loss auto-disabled)

class LieStructuredPPO:
    def __init__(self, env, policy, value_fn, cfg, use_lie_projection=False,
                 algebra="so"):
        self.env      = env
        self.policy   = policy.to(DEVICE)
        self.value_fn = value_fn.to(DEVICE)
        self.cfg      = cfg
        self.use_lie  = use_lie_projection
        self.algebra  = algebra
        self.ops      = LieAlgebraOps()

        if use_lie_projection and hasattr(policy, "theta"):
            theta_params = [policy.theta]
            other_params = [p for n, p in policy.named_parameters() if n != "theta"]
            self.pi_optim = torch.optim.Adam(
                [{"params": other_params, "lr": cfg.lr},
                 {"params": theta_params, "lr": cfg.theta_lr}]
            )
        else:
            self.pi_optim = torch.optim.Adam(self.policy.parameters(), lr=cfg.lr)

        self.vf_optim = torch.optim.Adam(self.value_fn.parameters(), lr=cfg.lr)
        self.projection_magnitudes: List[float] = []

    def _gather_trajectory(self):
        obs_buf, act_buf, rew_buf, val_buf, logp_buf = [], [], [], [], []
        done_buf: List[bool] = []   # needed for GAE terminal masking
        ep_returns_all: List[float] = []
        ep_curr: float = 0.0

        # M is computed but only used by env.step() interface --
        # SentimentSteeringEnv ignores it completely.
        M = None
        if self.use_lie and hasattr(self.policy, "theta"):
            with torch.no_grad():
                th_proj = self._apply_proj(self.policy.theta.data)
                M = LieAlgebraOps.matrix_exp(th_proj)

        state = self.env.reset()

        for _ in range(self.cfg.steps_per_iter):
            with torch.no_grad():
                probs  = self.policy(state)
                dist   = Categorical(probs)
                action = dist.sample()
                logp   = dist.log_prob(action)
                value  = self.value_fn(state)

            nxt, reward, done = self.env.step(action.item(), M)
            done_buf.append(bool(done))

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

        obs   = torch.stack(obs_buf)
        acts  = torch.stack(act_buf).squeeze(1)
        rews  = torch.tensor(rew_buf,  dtype=torch.float32, device=DEVICE)
        vals  = torch.tensor(val_buf,  dtype=torch.float32, device=DEVICE)
        logps = torch.tensor(logp_buf, dtype=torch.float32, device=DEVICE)

        dones = torch.tensor(done_buf, dtype=torch.bool, device=DEVICE)

        mean_ep = (float(np.mean(ep_returns_all))
                   if ep_returns_all else float(rews.mean().item()))
        return obs, acts, rews, vals, logps, last_val, mean_ep, dones

    def _compute_advantages(self, rewards, values, last_val, dones=None):
        """GAE with episode-boundary masking and per-episode normalization.

        Mirrors the corrected implementation in main.py: without terminal
        masks the recursion bootstraps across episode resets, and normalizing
        over the whole rollout rather than per episode breaks the
        (N-1)/sqrt(N) advantage bound quoted in the manuscript.
        """
        T   = len(rewards)
        adv = torch.zeros(T, device=DEVICE)
        if dones is None:
            dones = torch.zeros(T, dtype=torch.bool, device=DEVICE)
        gae = 0.0
        for t in reversed(range(T)):
            nonterminal = 0.0 if bool(dones[t]) else 1.0
            nv = last_val if t == T - 1 else values[t + 1].item()
            gae = (rewards[t] + self.cfg.gamma * nv * nonterminal - values[t]
                   + self.cfg.gamma * self.cfg.lam * gae * nonterminal)
            adv[t] = gae
        ret = adv + values

        mode = getattr(self.cfg, "adv_norm", "episode")
        if mode == "rollout":
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        elif mode == "episode":
            start = 0
            for t in range(T):
                if bool(dones[t]) or t == T - 1:
                    seg = adv[start:t + 1]
                    if seg.numel() > 1:
                        adv[start:t + 1] = (seg - seg.mean()) / (seg.std() + 1e-8)
                    else:
                        adv[start:t + 1] = 0.0
                    start = t + 1
        return adv, ret

    def _apply_proj(self, X):
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

    @staticmethod
    def _total_grad_norm(model):
        total = torch.zeros(1, device=DEVICE)
        for p in model.parameters():
            if p.grad is not None:
                total += p.grad.detach().norm(2) ** 2
        return total.sqrt().item()

    def _spectral_radius(self):
        if not hasattr(self.policy, "theta"):
            return None
        with torch.no_grad():
            th_proj = self._apply_proj(self.policy.theta.data)
            M       = LieAlgebraOps.matrix_exp(th_proj)
            return float(torch.linalg.eigvals(M).abs().max().item())

    def train(self, verbose=True):
        ep_returns  = []
        grad_norms  = []
        entropies   = []
        spec_rads   = []
        threshold_crossing = None

        for it in range(self.cfg.train_iters):
            obs, acts, rews, vals, logps, last_val, ep_return, dones = \
                self._gather_trajectory()
            adv, ret = self._compute_advantages(rews, vals, last_val, dones)

            N    = obs.shape[0]
            idxs = np.arange(N)
            mb_gnorms = []

            for _ in range(self.cfg.ppo_epochs):
                np.random.shuffle(idxs)
                for start in range(0, N, self.cfg.minibatch_size):
                    mb = idxs[start: start + self.cfg.minibatch_size]
                    if len(mb) == 0:
                        continue

                    self.pi_optim.zero_grad()
                    dist_mb  = Categorical(self.policy(obs[mb]))
                    logp_mb  = dist_mb.log_prob(acts[mb])
                    ratio    = torch.exp(logp_mb - logps[mb])
                    clip_adv = torch.clamp(ratio,
                                           1 - self.cfg.clip_ratio,
                                           1 + self.cfg.clip_ratio) * adv[mb]
                    surr_loss     = -torch.min(ratio * adv[mb], clip_adv).mean()
                    entropy_bonus = dist_mb.entropy().mean()
                    pi_loss       = surr_loss - self.cfg.entropy_coef * entropy_bonus
                    pi_loss.backward()

                    # No geo_aux_loss (env has no M_env attribute)

                    mb_gnorms.append(self._total_grad_norm(self.policy))
                    self._proj_grads()
                    self.pi_optim.step()
                    self._proj_params()

                    self.vf_optim.zero_grad()
                    F.mse_loss(self.value_fn(obs[mb]), ret[mb]).backward()
                    self.vf_optim.step()

            ep_returns.append(ep_return)
            grad_norms.append(
                float(np.median(mb_gnorms)) if mb_gnorms else 0.0)

            with torch.no_grad():
                ent = Categorical(self.policy(obs[:min(64, N)])).entropy().mean().item()
            entropies.append(ent)
            spec_rads.append(self._spectral_radius())

            if threshold_crossing is None and ep_return >= self.cfg.reward_threshold:
                threshold_crossing = it + 1

            if verbose:
                sr = spec_rads[-1]
                sr_str = f"  SpR={sr:.4f}" if sr is not None else ""
                print(f"Iter {it+1:03d}/{self.cfg.train_iters} | "
                      f"EpRet={ep_return:.3f} | "
                      f"GNorm={grad_norms[-1]:.3e} | "
                      f"H={ent:.3f}{sr_str}")

        auc = float(np.trapz(ep_returns) / max(len(ep_returns), 1))
        return {
            "returns":               ep_returns,
            "grad_norms":            grad_norms,
            "entropies":             entropies,
            "spectral_radii":        spec_rads,
            "projection_magnitudes": self.projection_magnitudes,
            "threshold_crossing":    threshold_crossing,
            "auc":                   auc,
        }


# STATISTICS (from main.py)

def bootstrap_ci(data, n_boot=10000, alpha=0.05):
    data = np.asarray(data)
    means = np.array([np.random.choice(data, len(data), replace=True).mean()
                      for _ in range(n_boot)])
    lo = np.percentile(means, 100 * alpha / 2)
    hi = np.percentile(means, 100 * (1 - alpha / 2))
    return float(lo), float(hi)

def iqm(data):
    data = np.sort(np.asarray(data))
    q25, q75 = np.percentile(data, [25, 75])
    trimmed = data[(data >= q25) & (data <= q75)]
    return float(trimmed.mean()) if len(trimmed) > 0 else float(data.mean())

def cohens_d(a, b):
    na, nb = len(a), len(b)
    pooled = np.sqrt(
        ((na - 1) * a.std(ddof=1)**2 + (nb - 1) * b.std(ddof=1)**2)
        / max(na + nb - 2, 1))
    return float((a.mean() - b.mean()) / (pooled + 1e-12))

def stat_compare(a, b, label_a, label_b):
    d    = cohens_d(a, b)
    ci_a = bootstrap_ci(a)
    ci_b = bootstrap_ci(b)
    size = "large" if abs(d) >= 0.8 else "medium" if abs(d) >= 0.5 else "small"
    print(f"  {label_a}: mean={a.mean():.3f}+/-{a.std(ddof=1):.3f}  "
          f"IQM={iqm(a):.3f}  95%-CI=[{ci_a[0]:.3f},{ci_a[1]:.3f}]")
    print(f"  {label_b}: mean={b.mean():.3f}+/-{b.std(ddof=1):.3f}  "
          f"IQM={iqm(b):.3f}  95%-CI=[{ci_b[0]:.3f},{ci_b[1]:.3f}]")
    print(f"  Cohen's d={d:.3f} ({size})")
    if HAS_SCIPY:
        t, p_t = scipy_stats.ttest_ind(a, b, equal_var=False)
        u, p_u = scipy_stats.mannwhitneyu(a, b, alternative="two-sided")
        sig = "***" if p_t < 0.001 else "**" if p_t < 0.01 else "*" if p_t < 0.05 else "ns"
        print(f"  Welch t={t:.3f} p={p_t:.4f}{sig}  |  "
              f"Mann-Whitney U={u:.0f} p={p_u:.4f}")
    return {"d": d, "ci_a": ci_a, "ci_b": ci_b}


# SEED UTILITY

def _seed_all(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


# MAIN EXPERIMENT

def main():
    parser = argparse.ArgumentParser(
        description="Task 3: Sentiment steering -- no engineered geometry")
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--output_dir", type=str, default="results_sentiment")
    args = parser.parse_args()

    seeds = args.seeds if args.seeds else list(range(10))

    print("=" * 70)
    print("TASK 3: SENTIMENT STEERING (NO GEOMETRY)")
    print("=" * 70)
    print(f"Seeds: {seeds}")
    print(f"Key design: no M_env, no geometry reward, M ignored in env")
    print()

    _seed_all(42)

    # -- Load GPT-2, compute sentiment direction, create env ----------
    embedder = GPT2EmbeddingProvider("gpt2-medium")
    sentiment_dir = compute_sentiment_direction(embedder)

    env = SentimentSteeringEnv(
        embedder,
        sentiment_direction=sentiment_dir,
        n_prompts=20,
        n_actions=8,
        horizon=10,
        reward_noise=0.1,
        sparse_prob=0.5,
    )

    state_dim = embedder.hidden_dim  # 1024
    n_actions = env.n_actions         # 8

    print(f"\nstate_dim={state_dim}, n_actions={n_actions}")
    print(f"Horizon=10, sparse_prob=0.5, reward_noise=0.1")
    print(f"geo_aux_coef=0.0 (no geometry loss)")
    print(f"hasattr(env, 'M_env') = {hasattr(env, 'M_env')}")
    print()

    # -- Multi-seed experiment ----------------------------------------
    cfg = PPOConfig()
    base_aucs, lie_aucs = [], []
    base_finals, lie_finals = [], []

    # Print parameter counts for transparency
    _tmp_lie  = LiePolicy(state_dim, n_actions, k=32, algebra="so")
    _tmp_base = BaselinePolicy(state_dim, n_actions)
    n_lie  = sum(p.numel() for p in _tmp_lie.parameters())
    n_base = sum(p.numel() for p in _tmp_base.parameters())
    print(f"Parameter counts:")
    print(f"  LiePolicy (so):  {n_lie:,}  (feature_net dominates; theta has 496 free)")
    print(f"  BaselinePolicy:  {n_base:,}")
    print(f"  Ratio: {n_lie/n_base:.1f}x")
    print(f"  [WARNING] Architecture differs -- algebra ablation (below) controls for this")
    del _tmp_lie, _tmp_base

    for seed in seeds:
        print(f"\n{'-'*40} Seed {seed} {'-'*40}")
        _seed_all(seed)

        # Baseline PPO
        print("\n[Baseline PPO]")
        base_res = LieStructuredPPO(
            env, BaselinePolicy(state_dim, n_actions), ValueNet(state_dim),
            cfg, use_lie_projection=False,
        ).train(verbose=True)

        # SP-PG (SO)
        print("\n[SP-PG (SO)]")
        _seed_all(seed)  # re-seed for fair comparison
        lie_res = LieStructuredPPO(
            env,
            LiePolicy(state_dim, n_actions, k=32, algebra="so"),
            ValueNet(state_dim), cfg,
            use_lie_projection=True, algebra="so",
        ).train(verbose=True)

        base_aucs.append(base_res["auc"])
        lie_aucs.append(lie_res["auc"])
        base_finals.append(base_res["returns"][-1])
        lie_finals.append(lie_res["returns"][-1])

        print(f"\n  Seed {seed}: Base AUC={base_res['auc']:.3f}  "
              f"SP-PG AUC={lie_res['auc']:.3f}  "
              f"Delta={(lie_res['auc']-base_res['auc'])/max(base_res['auc'],1e-8)*100:+.1f}%")

    # -- Summary statistics -------------------------------------------
    ba = np.array(base_aucs)
    la = np.array(lie_aucs)

    print("\n" + "=" * 70)
    print("TASK 3 SUMMARY: SENTIMENT STEERING (NO GEOMETRY)")
    print("=" * 70)
    print("\n-- AUC (primary metric) --")
    stat_compare(la, ba, "SP-PG (SO)", "Baseline PPO")

    delta = (la.mean() - ba.mean()) / max(ba.mean(), 1e-8) * 100
    print(f"\nDelta AUC: {delta:+.1f}%")

    if delta > 5:
        print("\n-> SP-PG helps even without engineered geometry.")
        print("  The compactness benefit generalises beyond geometry-aligned tasks.")
    elif delta < -5:
        print("\n-> SP-PG hurts without geometry reward.")
        print("  The constraint is beneficial only when the environment")
        print("  has geometric structure that so(32) can exploit.")
    else:
        print("\n-> SP-PG and baseline are comparable without geometry.")
        print("  The so(32) constraint neither helps nor hurts on this task;")
        print("  the benefit observed in Task 1 is specific to geometry-rich envs.")

    # -- Save results -------------------------------------------------
    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True)

    save_data = {
        "experiment": "Task 3: Sentiment steering (no geometry)",
        "design": {
            "M_env": "NONE",
            "geometry_reward": "NONE",
            "geo_aux_coef": 0.0,
            "M_ignored_in_env": True,
            "reward": "cosine similarity with sentiment direction",
        },
        "seeds": seeds,
        "base_aucs": list(base_aucs),
        "lie_aucs": list(lie_aucs),
        "delta_pct": delta,
    }
    with open(out_dir / "results_task3.json", "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {out_dir / 'results_task3.json'}")

    # -- Algebra ablation (seed 0) -- controls for architecture --------
    # LiePolicy(so) vs LiePolicy(full) vs LiePolicy(sym)
    # Same architecture, same param count -> isolates constraint effect
    print("\n" + "=" * 70)
    print("ALGEBRA ABLATION (seed 0) -- same architecture, different constraints")
    print("=" * 70)

    ablation_results = {}
    for label, algebra in [
        ("SO(32) -- compact",       "so"),
        ("Full (unconstrained)",   "full"),
        ("SYM(32) -- non-compact",  "sym"),
    ]:
        _seed_all(0)
        policy  = LiePolicy(state_dim, n_actions, k=32, algebra=algebra)
        trainer = LieStructuredPPO(
            env, policy, ValueNet(state_dim), cfg,
            use_lie_projection=True, algebra=algebra,
        )
        res = trainer.train(verbose=False)
        ablation_results[label] = res
        R = np.array(res["returns"])
        sr = res["spectral_radii"][-1] if res["spectral_radii"][-1] else "N/A"
        print(f"  {label:28s} | AUC={res['auc']:.3f} | final={R[-1]:.3f} | SpR={sr}")

    print("\nThis comparison controls for architecture (all use LiePolicy).")
    print("If so(32) > full, compactness helps even without geometry.")
    print("If so(32) ~= full, the constraint is neutral on this task.")

    # -- LaTeX-ready output -------------------------------------------
    print("\n-- For Paper (LaTeX) --")
    print(f"% Task 3: Sentiment steering (no geometry)")
    print(f"% SP-PG: AUC ${la.mean():.3f} \\pm {la.std(ddof=1):.3f}$")
    print(f"% Baseline: AUC ${ba.mean():.3f} \\pm {ba.std(ddof=1):.3f}$")
    print(f"% Delta: ${delta:+.1f}\\%$")
    print(f"% Design: no M_env, no geometry reward, M ignored in env")

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
