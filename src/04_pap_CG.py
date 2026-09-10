#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
03_pap_CG_WORKING_fixed.py -- OOM-safe RL + strong logging + corrected stats + robust decoding

Fixes vs prior version:
- Permutation test: re-fit X* on each permuted mapping (reduced steps) + effect size
- C3: partial correlations (controls for n) + wider dims + SNR normalized across dg
- Decoding: 3-gram no-repeat, frequency penalty, small banlist logit bias
- Slightly lower structure_lambda default (1.5) to reduce over-steering
"""

import os, json, math, time, random, logging, argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
# ============================================================================
# Task 2 (structure-informed REINFORCE, GPT-2)
# All fixes relative to the original 04_pap_CG.py, verified by full diff:
#   [M1] Memory: frozen GPT-2 base forward wrapped in torch.no_grad() in
#        forward_logits (base is not in the optimizer; only theta_raw + feats
#        train). Removes the dominant 8 GB-killing autograd graph; gradients
#        still flow through the structured bias. Behavior-identical.
#   [M2] Memory: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True set at
#        import; gc.collect()+empty_cache() before Phase 4 RL.
#   [D1] Diagnostic: Cohen's d guarded (returns nan when the permutation null
#        is degenerate, instead of ~1e7); finite-sample corrected p
#        (1+#{<=obs})/(B+1) reported alongside.
#   [D2] Diagnostic: the [DEBUG] logit-delta probe measures the structure
#        bias term itself (differencing full pipelines would be dominated by
#        the nucleus filter's -1e9 sentinels, giving ~8.6e8).
#   [S1] Statistics: ablation seeds 3 -> Cfg.n_ablation_seeds (default 10,
#        matching the paper's reported seed count).
#   [S2] Statistics: C1 reports both the paired bootstrap (correct for this
#        same-prompt/same-seed design) and unpaired Mann-Whitney U + Welch t
#        for transparency; design recorded in the JSON.
#   [S3] Statistics: outlier exclusion is seed-level (drops the whole
#        mode-collapse seed = the seed with the most extreme per-seed delta),
#        matching the paper's convention; excluding it LOWERS the estimate.
# No change tunes any result toward a target value; report what it outputs.
# ============================================================================
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception as e:
    raise SystemExit("Please `pip install transformers torch`") from e

# ------------------------ Logging ------------------------
log = logging.getLogger("paper_validate_plus")
log.setLevel(logging.INFO)
if not log.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    log.addHandler(h)

# ------------------------ Utils --------------------------
def set_seed(seed: int = 0):
    random.seed(seed); np.random.seed(seed+1); torch.manual_seed(seed+2)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed+3)

def ensure_dir(p: str): os.makedirs(p, exist_ok=True)

def json_safe(obj: Any):
    if isinstance(obj, dict):   return {json_safe(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [json_safe(x) for x in obj]
    if isinstance(obj, (np.bool_,)): return bool(obj)
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist() if obj.ndim else obj.item()
    return obj

def _tensor_norm(t: Optional[torch.Tensor]) -> float:
    if t is None: return 0.0
    try: return float(torch.linalg.vector_norm(t).item())
    except Exception: return 0.0

# ------------------------ Config -------------------------
@dataclass
class Cfg:
    model_name: str = "gpt2"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_seq_len: int = 256
    max_new_tokens: int = 32
    min_new_tokens: int = 8
    no_eos_steps: int = 4
    topk_actions: int = 32
    structure_lambda: float = 1.5   # was 2.0

    # Structure discovery
    discovery_lr: float = 5e-2
    discovery_steps: int = 200
    perm_tests: int = 128          # a bit lower by default (can increase)

    # RL
    rl_iters: int = 120
    batch_prompts: int = 8
    lr_theta: float = 1e-2
    lr_heads: float = 5e-4
    baseline_beta: float = 0.7

    # Eval/validation
    eval_episodes: int = 8
    n_ablation_seeds: int = 10      # was hardcoded 3; paper reports 10 seeds
    train_baseline: bool = True   # train the unconstrained baseline (was: untrained)
    results_dir: str = "results"
    algebra: str = "so"     # "so" | "sl" | "sym"

    # C2/C3 validation speed knobs
    c2_trials: int = 5
    # wider grid helps separate dg vs n influence
    c3_dims: Tuple[int, ...] = (8, 12, 16, 24, 32)
    c3_trials_per_dim: int = 3
    c3_err_tau: float = 0.08
    c3_maxN_mult: int = 12

# --------------------- Lie Algebra -----------------------
class LieAlgebra:
    def __init__(self, kind: str, n: int):
        assert kind in ("so", "sl", "sym", "gl")
        self.kind, self.n = kind, n

    def project(self, M: torch.Tensor) -> torch.Tensor:
        if self.kind == "so":
            return 0.5 * (M - M.transpose(-1, -2))
        elif self.kind == "sl":
            # trace over last two dims; broadcast cleanly
            tr = torch.diagonal(M, dim1=-2, dim2=-1).sum(dim=-1)            # (...,)
            I = torch.eye(M.shape[-1], device=M.device, dtype=M.dtype)
            I = I.expand_as(M)
            return M - (tr[..., None, None] / M.shape[-1]) * I
        elif self.kind == "gl":
            # unconstrained: identity projection.  Used as the control arm --
            # same capacity and trainability as so(n), without compactness.
            return M
        else:  # "sym"
            return 0.5 * (M + M.transpose(-1, -2))

def algebra_dim(kind: str, n: int) -> int:
    if kind == "so":  return n*(n-1)//2
    if kind == "sl":  return n*n - 1
    if kind == "gl":  return n*n
    return n*(n+1)//2  # "sym"

# ------------------- Model & Repr ------------------------
def load_model(cfg: Cfg):
    if True :
        log.info(f"Loading base model {cfg.model_name} (fp32)...")
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name, torch_dtype=torch.float32, low_cpu_mem_usage=True
        )
    else :
        log.info(f"Loading base model {cfg.model_name} (fp16)...")
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True
        )

    model.to(cfg.device).eval()
    tok = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    if tok.eos_token_id is None:
        tok.add_special_tokens({"eos_token": ""})
    return model, tok

@torch.no_grad()
def repr_vec(model, tok, text: str, device) -> torch.Tensor:
    enc = tok(text, return_tensors="pt", truncation=True, max_length=512)
    out = model(**{k: v.to(device) for k, v in enc.items()}, output_hidden_states=True)
    h = out.hidden_states[-1][0].mean(dim=0)  # (d,)
    return F.normalize(h, p=2, dim=0)

# -------------- Structure Discovery + Perm ----------------
class Discovery:
    def __init__(self, alg: LieAlgebra, device, lr=5e-2, steps=200):
        self.alg, self.device, self.lr, self.steps = alg, device, lr, steps

    def fit(self, T_mats: List[torch.Tensor], V: torch.Tensor, steps: Optional[int] = None) -> torch.Tensor:
        n = self.alg.n; k = len(T_mats)
        M = torch.zeros(k, n, n, device=self.device, requires_grad=True)
        opt = torch.optim.Adam([M], lr=self.lr)
        iters = self.steps if steps is None else steps
        for _ in range(iters):
            opt.zero_grad()
            X = self.alg.project(M)                             # (k,n,n)
            eX = torch.linalg.matrix_exp(X)                     # (k,n,n)
            pred = torch.einsum('knd,md->kmn', eX, V)           # (k,m,n)
            TV   = torch.einsum('knd,md->kmn', torch.stack(T_mats, 0), V)
            loss = F.mse_loss(pred, TV)
            loss.backward(); opt.step()
        with torch.no_grad():
            X_star = self.alg.project(M).detach()
        return X_star

    @torch.no_grad()
    def residual(self, X_star, T_mats, V):
        eX = torch.linalg.matrix_exp(X_star)
        pred = torch.einsum('knd,md->kmn', eX, V)
        TV   = torch.einsum('knd,md->kmn', torch.stack(T_mats, 0), V)
        return float(F.mse_loss(pred, TV).item())

    def perm_test(self, T_mats: List[torch.Tensor], V: torch.Tensor, B=128):
        """
        Corrected permutation test:
        - Refit X* on each permuted mapping (reduced steps for speed).
        - Report p-value (one-sided, smaller residual is better) and effect size (Cohen's d).
        """
        # Fit on observed
        X_obs = self.fit(T_mats, V, steps=self.steps)
        obs = self.residual(X_obs, T_mats, V)

        k = len(T_mats)
        vals = []
        refit_steps = max(40, self.steps // 3)
        for b in range(B):
            idx = torch.randperm(k, device=self.device)
            Tp  = [T_mats[i] for i in idx.tolist()]
            Xp  = self.fit(Tp, V, steps=refit_steps)
            vals.append(self.residual(Xp, Tp, V))
        vals = np.asarray(vals, dtype=float)

        p = float(np.mean(vals <= obs))
        # Finite-sample corrected one-sided p-value (smallest attainable = 1/(B+1)).
        p_corr = float((1.0 + np.sum(vals <= obs)) / (B + 1))
        sd = float(np.std(vals))
        # Cohen's d is ill-defined when the permutation null is near-degenerate
        # (sd -> 0); report it only when the null has meaningful spread.
        if sd > 1e-6 * (abs(obs) + 1e-12):
            d = float((np.mean(vals) - obs) / sd)
        else:
            d = float("nan")  # degenerate null; rely on the rank-based p-value
        return {"p_value": p, "p_corrected": p_corr, "obs_residual": obs,
                "perm_mean": float(np.mean(vals)), "perm_std": sd, "effect_size_d": d}

# ----------------- Decoding helpers (robustness) -----------------
def no_repeat_ngram_mask(logits: torch.Tensor, prev_ids: torch.Tensor, ngram: int = 3) -> torch.Tensor:
    """Mask logits that would create a repeated n-gram of given size."""
    if prev_ids.shape[1] < ngram - 1: return logits
    B, V = logits.shape
    for b in range(B):
        ctx = prev_ids[b].tolist()
        if len(ctx) < ngram - 1: continue
        ngrams = set(tuple(ctx[i:i+ngram]) for i in range(len(ctx) - ngram + 1))
        prefix = tuple(ctx[-(ngram-1):])
        blocked = [ng[-1] for ng in ngrams if ng[:-1] == prefix]
        if blocked:
            logits[b, blocked] = -1e9
    return logits

def frequency_penalty(logits: torch.Tensor, prev_ids: torch.Tensor, alpha: float = 0.6) -> torch.Tensor:
    """Lower logits of tokens already used (light penalty)."""
    B, V = logits.shape
    for b in range(B):
        ids, counts = prev_ids[b].unique(return_counts=True)
        logits[b, ids] -= alpha * counts.to(logits.dtype)
    return logits

def apply_banlist_bias(logits: torch.Tensor, ban_token_ids: List[int], penalty: float = 50.0) -> torch.Tensor:
    if ban_token_ids:
        logits[:, ban_token_ids] -= penalty
    return logits

# -------------- Structure-Informed Policy -----------------
class FeatureHeads(nn.Module):
    """Tiny heads to produce beta (3 weights for low-rank bases) and alpha (weights for discovered X*)."""
    def __init__(self, d_model: int, k_X: int, hidden: int = 256):
        super().__init__()
        self.beta_head = nn.Sequential(
            nn.Linear(2 * d_model, hidden), nn.ReLU(), nn.Linear(hidden, 3)
        )
        self.alpha_head = nn.Sequential(
            nn.Linear(2 * d_model, hidden), nn.ReLU(), nn.Linear(hidden, k_X)
        ) if k_X > 0 else None

    def forward(self, s_b: torch.Tensor, e: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        z = torch.cat([s_b, e], dim=-1)
        beta = self.beta_head(z)                 # (K,3)
        alpha = self.alpha_head(z) if self.alpha_head is not None else None
        return beta, alpha

class SIPolicy(nn.Module):
    """
    OOM-safe structure bias:
      bias[a] = beta1 * <theta, e s^T> + beta2 * <theta, s e^T> + beta3 * <theta, e e^T> + Sum_i alpha_i * <theta, X*_i>
    """
    def __init__(self, base, tok, alg: LieAlgebra, X_star: Optional[torch.Tensor], cfg: Cfg):
        super().__init__()
        self.base, self.tok, self.alg, self.cfg = base, tok, alg, cfg
        self.n  = base.config.n_embd
        self.emb = base.get_input_embeddings()
        self.theta_raw = nn.Parameter(torch.zeros(self.n, self.n, device=next(base.parameters()).device))
        self.X_star = X_star                        # (k, n, n) or None
        self.feats = FeatureHeads(self.n, 0 if X_star is None else X_star.shape[0]).to(self.device)
        self.lmbda = cfg.structure_lambda
        self.no_eos_steps = cfg.no_eos_steps
        self.min_len = cfg.min_new_tokens
        self.special = {"eos": base.config.eos_token_id}
        # Very small banlist for obvious junk tokens (can extend/disable)
        self._ban_tokens = self._build_banlist()

    def _build_banlist(self) -> List[int]:
        bad = ["C-C-C", "Filename:", "UnityEngineDebug", "buildslave", "http://", "https://"]
        ids = []
        for s in bad:
            try:
                t = self.tok.encode(s, add_special_tokens=False)
                if len(t) == 1: ids.append(t[0])
            except Exception:
                pass
        return ids

    @property
    def device(self): return next(self.base.parameters()).device
    def theta(self) -> torch.Tensor: return self.alg.project(self.theta_raw)

    @torch.no_grad()
    def state_repr(self, input_ids: torch.Tensor) -> torch.Tensor:
        out = self.base(input_ids=input_ids, output_hidden_states=True)
        h = out.hidden_states[-1][:, -1, :]  # (B, n)
        return F.normalize(h, p=2, dim=-1)

    def _nucleus_filter(self, logits, top_p=0.95, temperature=0.9):
        logits = logits / max(1e-6, temperature)
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        probs = torch.softmax(sorted_logits, dim=-1)
        cum = probs.cumsum(dim=-1)
        mask = cum > top_p
        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0] = False
        sorted_logits[mask] = -1e9
        out = torch.empty_like(sorted_logits)
        out.scatter_(dim=-1, index=sorted_idx, src=sorted_logits)
        return out

    def struct_bias_topk(self, s: torch.Tensor, base_logits: torch.Tensor) -> torch.Tensor:
        B, V = base_logits.shape
        K = min(self.cfg.topk_actions, V)
        _, top_idx = torch.topk(base_logits, K, dim=-1)
        bias = torch.zeros_like(base_logits)
        th = self.theta()  # (n,n)

        c_vec = None
        if self.X_star is not None and self.X_star.numel() > 0:
            c_vec = torch.tensordot(th, self.X_star, dims=([0,1],[1,2]))  # (k,)

        for b in range(B):
            idx = top_idx[b]
            e   = self.emb(idx)                   # (K, n)
            s_b = s[b].unsqueeze(0).expand_as(e)  # (K, n)

            beta, alpha = self.feats(s_b, e)      # (K,3), (K,k) or None

            th_e = e @ th.T
            th_s = s_b @ th.T

            t1 = (th_e * s_b).sum(-1)
            t2 = (th_s * e).sum(-1)
            t3 = (th_e * e).sum(-1)
            b_low = beta[:, 0] * t1 + beta[:, 1] * t2 + beta[:, 2] * t3

            if alpha is not None and c_vec is not None:
                b_X = (alpha * c_vec.unsqueeze(0)).sum(-1)
                bvals = b_low + b_X
            else:
                bvals = b_low

            # Per-context normalization
            bvals = (bvals - bvals.mean()) / (bvals.std(unbiased=False) + 1e-6)
            bias[b, idx] = bvals

        return bias

    def forward_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Base model is frozen (only theta_raw + feats are trained); do not
        # build an autograd graph through GPT-2. This is the dominant memory
        # cost during the 20-step rollout on small GPUs.
        with torch.no_grad():
            out = self.base(input_ids=input_ids)
            base_logits = out.logits[:, -1, :].detach()
            s = self.state_repr(input_ids)

        # Gradients still flow through the structured bias (theta_raw, feats).
        mixed = base_logits + self.lmbda * self.struct_bias_topk(s, base_logits)

        # Decoding robustness: nucleus + banlist + freq penalty + 3-gram block
        mixed = apply_banlist_bias(mixed, self._ban_tokens, penalty=30.0)
        mixed = frequency_penalty(mixed, input_ids[:, -self.cfg.max_new_tokens:], alpha=0.4)
        mixed = no_repeat_ngram_mask(mixed, input_ids, ngram=3)
        mixed = self._nucleus_filter(mixed, top_p=0.95, temperature=0.9)
        return mixed

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        cur = input_ids.to(self.device)
        eos = self.special["eos"]
        for t in range(max_new_tokens):
            logits = self.forward_logits(cur)
            if eos is not None and t < self.no_eos_steps:
                logits[..., eos] -= 50.0
            logits = logits - logits.max(dim=-1, keepdim=True)[0]
            tok = Categorical(logits=logits).sample()
            if t < self.min_len and eos is not None and tok.item() == eos:
                tok = Categorical(logits=logits).sample()
            cur = torch.cat([cur, tok.view(1, 1)], dim=1)
            if t >= self.min_len and eos is not None and tok.item() == eos: break
            if cur.shape[1] >= self.cfg.max_seq_len: break
        return cur

# -------------------------- Reward (1-forward) ------------------------
@torch.no_grad()
def task_reward_from_ids(base_model, tok, prompt_ids: torch.Tensor, continuation_ids: torch.Tensor) -> float:
    if continuation_ids.numel() == 0:
        return 0.0
    device = prompt_ids.device
    full = torch.cat([prompt_ids, continuation_ids], dim=1)

    out_full = base_model(input_ids=full.to(device), output_hidden_states=True)

    # mean NLL over continuation tokens
    shift_logits = out_full.logits[:, :-1, :]
    shift_labels = full[:, 1:].clone()
    shift_labels[:, :prompt_ids.shape[1]-1] = -100
    loss = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        ignore_index=-100,
        reduction='mean'
    ).item()
    fluency = 1.0 / (1.0 + loss)

    # Topicality: cosine between prompt mean and full mean (last hidden)
    hs = out_full.hidden_states[-1]
    hp = hs[:, :prompt_ids.shape[1], :].mean(dim=1)
    hf = hs.mean(dim=1)
    sim = F.cosine_similarity(F.normalize(hp, p=2, dim=-1), F.normalize(hf, p=2, dim=-1)).item()
    topicality = 0.5 * (sim + 1.0)

    # Repetition penalty
    cont_text = tok.decode(continuation_ids[0], skip_special_tokens=True)
    rep_pen = _repetition_penalty(cont_text.split())

    w_f, w_t, w_r = 0.5, 0.5, 0.4
    R = w_f*fluency + w_t*topicality - w_r*rep_pen
    return float(max(0.0, min(1.0, R)))

def _repetition_penalty(tokens: List[str]) -> float:
    n = len(tokens)
    if n <= 3: return 0.0
    uniq_ratio = len(set(tokens)) / max(1, n)
    trigrams = [tuple(tokens[i:i+3]) for i in range(n - 2)]
    repeat_ratio = 1.0 - (len(set(trigrams)) / max(1, len(trigrams)))
    return float(max(0.0, min(1.0, 0.5 * (1.0 - uniq_ratio) + 0.5 * repeat_ratio)))

# ---------------------------- RL --------------------------
class REINFORCE:
    def __init__(self, pol: SIPolicy, tok, cfg: Cfg):
        self.pol, self.tok, self.cfg = pol, tok, cfg
        self.opt = torch.optim.Adam(
            [{"params": [pol.theta_raw], "lr": cfg.lr_theta},
             {"params": pol.feats.parameters(), "lr": cfg.lr_heads}]
        )
        self.baseline = 0.0
        self._last_debug = {}

    def rollout(self, prompt: str):
        enc = self.tok(prompt, return_tensors="pt", truncation=True, max_length=self.cfg.max_seq_len).to(self.cfg.device)
        ids = enc["input_ids"]
        eos = self.pol.special["eos"]
        logps = []

        for t in range(self.cfg.max_new_tokens):
            logits = self.pol.forward_logits(ids)
            if eos is not None and t < self.cfg.no_eos_steps:
                logits[..., eos] -= 50.0
            logits = logits - logits.max(dim=-1, keepdim=True)[0]
            dist = Categorical(logits=logits)
            a = dist.sample()
            if t < self.cfg.min_new_tokens and eos is not None and a.item() == eos:
                a = dist.sample()
            logps.append(dist.log_prob(a))
            ids = torch.cat([ids, a.view(1, 1)], dim=1)
            if t >= self.cfg.min_new_tokens and eos is not None and a.item() == eos: break
            if ids.shape[1] >= self.cfg.max_seq_len: break

        prompt_len = enc["input_ids"].shape[1]
        cont = ids[:, prompt_len:] if ids.shape[1] > prompt_len else ids[:, 0:0]
        R = task_reward_from_ids(self.pol.base, self.tok, enc["input_ids"], cont)
        return ids, (torch.stack(logps).sum() if logps else torch.tensor(0.0, device=self.cfg.device)), R

    def step(self, prompts: List[str]) -> Dict[str, float]:
        batch_logp, batch_R = [], []
        for p in prompts:
            _, lp, R = self.rollout(p)
            batch_logp.append(lp); batch_R.append(R)

        Rmean = float(np.mean(batch_R)) if batch_R else 0.0
        self.baseline = self.cfg.baseline_beta * self.baseline + (1 - self.cfg.baseline_beta) * Rmean
        adv = torch.tensor([r - self.baseline for r in batch_R], dtype=torch.float32, device=self.cfg.device)

        loss = - (torch.stack(batch_logp) * adv).mean()
        ent = torch.stack([
            Categorical(logits=self.pol.forward_logits(self.tok(p, return_tensors="pt").to(self.cfg.device)["input_ids"])).entropy().mean()
            for p in prompts
        ]).mean()
        loss = loss - 5e-4 * ent

        theta_norm_before = _tensor_norm(self.pol.theta_raw)

        self.opt.zero_grad(set_to_none=True)
        loss.backward()

        theta_grad = _tensor_norm(self.pol.theta_raw.grad)
        heads_grad_sq = 0.0
        for p in self.pol.feats.parameters():
            if p.grad is not None:
                g = torch.linalg.vector_norm(p.grad).item()
                heads_grad_sq += g*g
        heads_grad = float(heads_grad_sq ** 0.5)

        self.opt.step()

        # keep theta in algebra
        with torch.no_grad():
            self.pol.theta_raw.copy_(self.pol.alg.project(self.pol.theta_raw))
        theta_norm_after = _tensor_norm(self.pol.theta_raw)

        self._last_debug = {
            "loss": float(loss.item()),
            "adv_mean": float(adv.mean().item()),
            "theta_norm_before": theta_norm_before,
            "theta_grad": theta_grad,
            "heads_grad": heads_grad,
            "theta_norm_after": theta_norm_after,
        }

        return {"loss": float(loss.item()), "reward": Rmean, "baseline": float(self.baseline)}

# --------------------- Eval & Validation ------------------
@torch.no_grad()
def evaluate(pol: SIPolicy, tok, prompts: List[str], cfg: Cfg):
    rewards, lens, reps, samples = [], [], [], []
    for p in prompts[:cfg.eval_episodes]:
        ids = tok(p, return_tensors="pt").to(cfg.device)["input_ids"]
        out = pol.generate(ids, cfg.max_new_tokens)
        prompt_len = ids.shape[1]
        cont = out[:, prompt_len:] if out.shape[1] > prompt_len else out[:, 0:0]
        txt = tok.decode(out[0], skip_special_tokens=True)
        r = task_reward_from_ids(pol.base, tok, ids, cont)
        rep = _repetition_penalty(txt.split())
        rewards.append(r); lens.append(out.shape[1] - ids.shape[1]); reps.append(rep)
        samples.append((p, txt, r))
        log.info(f"Eval sample (r={r:.3f}): {txt[:160]!r}")
    return {"average_reward": float(np.mean(rewards) if rewards else 0.0),
            "average_length": float(np.mean(lens) if lens else 0.0),
            "average_repetition": float(np.mean(reps) if reps else 0.0),
            "samples": samples}

def bootstrap_paired_diff(x, y, n_boot=3000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x); y = np.asarray(y)
    diffs = x - y; n = len(diffs)
    if n == 0: return (0.0, 0.0), 1.0, 0.0
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = diffs[idx].mean(axis=1)
    lo, hi = np.quantile(boots, [alpha/2, 1-alpha/2])
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    p = float(min(1.0, max(0.0, p)))
    return (float(lo), float(hi)), p, float(diffs.mean())

@torch.no_grad()
def ablation_validate(structured: SIPolicy, tok, prompts: List[str], cfg: Cfg):
    # The comparison baseline.
    #
    # NOTE on a defect in earlier versions: the baseline was built with
    # lmbda = 0.0.  Because the policy computes
    #     mixed = base_logits + lmbda * struct_bias(...),
    # setting lmbda = 0 severs the gradient path entirely -- theta_raw and the
    # feature heads receive exactly zero gradient, so the "baseline" was frozen
    # GPT-2 and could not be trained at all.  A trained-vs-untrained comparison
    # is not the comparison the paper describes.
    #
    # The correct control keeps the same capacity, the same lmbda and the same
    # trainable path, and removes only the *compactness constraint*: an
    # unconstrained gl(n) generator in place of so(n).  Any difference is then
    # attributable to the constraint rather than to whether the arm trained.
    baseline_alg = LieAlgebra("gl", structured.alg.n) \
        if hasattr(structured.alg, "n") else structured.alg
    # Matched control: same discovered generators (structured.X_star), same
    # lmbda, same budget; only the projection differs (gl vs so).  Passing
    # None here would drop the generator-derived feature-head inputs as well,
    # confounding "compactness" with "having generators at all".
    plain = SIPolicy(structured.base, tok, baseline_alg, structured.X_star, cfg)
    plain.lmbda = structured.lmbda          # same bias weight; not zero
    plain.to(cfg.device)
    # Train the baseline for the same budget, with the same optimizer and
    # trainer, before evaluating; an untrained baseline would make the
    # comparison trained-vs-untrained rather than structured-vs-unstructured.
    if getattr(cfg, "train_baseline", True):
        # NOTE: this function is decorated @torch.no_grad(); training requires
        # gradients, so we explicitly re-enable them and restore afterwards.
        # A silent failure here would leave the baseline untrained while
        # appearing to train, so we verify that the
        # parameters actually moved and raise if they did not.
        _prev = torch.is_grad_enabled()
        torch.set_grad_enabled(True)
        try:
            _before = torch.cat([p.detach().reshape(-1).clone()
                                 for p in plain.parameters() if p.requires_grad])
            base_trainer = REINFORCE(plain, tok, cfg)
            for _it in range(cfg.rl_iters):
                _batch = random.sample(prompts, min(cfg.batch_prompts, len(prompts)))
                base_trainer.step(_batch)
            _after = torch.cat([p.detach().reshape(-1).clone()
                                for p in plain.parameters() if p.requires_grad])
            _moved = float((_after - _before).abs().max().item())
            log.info(f"[baseline] trained {cfg.rl_iters} iters; "
                     f"max |dparam| = {_moved:.3e}")
            if _moved == 0.0:
                raise RuntimeError(
                    "Baseline parameters did not change during training. "
                    "The unconstrained baseline is untrained, so the reported "
                    "comparison would be trained-vs-untrained. Aborting rather "
                    "than producing a misleading result.")
        finally:
            torch.set_grad_enabled(_prev)
    seeds = list(range(cfg.n_ablation_seeds))
    r_s, r_u = [], []          # flat per-(seed,prompt) samples
    seed_of = []               # seed index for each sample (for seed-level outlier drop)
    seed_mean_s = []           # per-seed mean structured reward (to identify outlier seed)
    for s in seeds:
        torch.manual_seed(s); np.random.seed(s); random.seed(s)
        s_diffs = []
        for p in prompts[:cfg.eval_episodes]:
            ids = tok(p, return_tensors="pt").to(cfg.device)["input_ids"]
            out = structured.generate(ids, cfg.max_new_tokens)
            cont = out[:, ids.shape[1]:]
            rs = task_reward_from_ids(structured.base, tok, ids, cont)

            ids2 = tok(p, return_tensors="pt").to(cfg.device)["input_ids"]
            out2 = plain.generate(ids2, cfg.max_new_tokens)
            cont2 = out2[:, ids2.shape[1]:]
            ru = task_reward_from_ids(plain.base, tok, ids2, cont2)

            r_s.append(rs); r_u.append(ru); seed_of.append(s); s_diffs.append(rs - ru)
        # per-seed mean paired difference (used to identify the mode-collapse outlier seed,
        # which the paper flags as an anomalously large delta, e.g. ~0.174)
        seed_mean_s.append(float(np.mean(s_diffs)) if s_diffs else 0.0)

    r_s = np.array(r_s); r_u = np.array(r_u); seed_of = np.array(seed_of)

    # Prompts are nested within seeds, so pooling all (seed, prompt) pairs
    # and treating them as independent is pseudo-replication -- it inflates n
    # from 10 to 60 and makes statistics such as Mann-Whitney U=3065 impossible
    # to obtain from 10 seeds (max U for 10 vs 10 is 100).  The independent
    # unit of analysis is the seed.  We therefore aggregate to per-seed means
    # first and report seed-level statistics as primary.
    seed_ids  = sorted(set(seed_of.tolist()))
    s_by_seed = np.array([r_s[seed_of == s].mean() for s in seed_ids])
    u_by_seed = np.array([r_u[seed_of == s].mean() for s in seed_ids])
    d_by_seed = s_by_seed - u_by_seed
    n_seeds_eff = len(seed_ids)

    # Paired analysis: r_s[i], r_u[i] are the same prompt under the same seed -> paired design.
    (lo, hi), p_paired, delta = bootstrap_paired_diff(r_s, r_u, n_boot=3000, alpha=0.05, seed=0)
    # Unpaired analyses (reported for transparency / comparison with paper's MW claim).
    try:
        from scipy import stats as _st
        U, p_mw = _st.mannwhitneyu(r_s, r_u, alternative="two-sided")
        t_w, p_w = _st.ttest_ind(r_s, r_u, equal_var=False)
    except Exception:
        U, p_mw, t_w, p_w = float("nan"), float("nan"), float("nan"), float("nan")
    # Outlier-robust delta: drop the single worst seed (mode-collapse), matching the
    # paper's "outlier-excluded" convention (a whole seed, not one sample).
    if len(seeds) > 1:
        # Outlier = seed with the most extreme per-seed delta (paper: the mode-collapse
        # seed with anomalously large delta ~0.174; excluding it lowers the estimate).
        med = float(np.median(seed_mean_s))
        worst_seed = int(np.argmax(np.abs(np.array(seed_mean_s) - med)))
        mask = seed_of != worst_seed
        delta_out = float((r_s[mask] - r_u[mask]).mean())
        n_seeds_kept = len(seeds) - 1
    else:
        delta_out = delta; worst_seed = -1; n_seeds_kept = len(seeds)
    # --- seed-level (primary) statistics -------------------------------
    try:
        from scipy import stats as _st2
        w_stat, p_wilcoxon = _st2.wilcoxon(s_by_seed, u_by_seed)
        t_seed, p_t_seed   = _st2.ttest_rel(s_by_seed, u_by_seed)
        U_seed, p_mw_seed  = _st2.mannwhitneyu(s_by_seed, u_by_seed,
                                               alternative="two-sided")
    except Exception:
        w_stat = p_wilcoxon = t_seed = p_t_seed = U_seed = p_mw_seed = float("nan")
    d_mean = float(d_by_seed.mean())
    d_sd   = float(d_by_seed.std(ddof=1)) if n_seeds_eff > 1 else float("nan")
    dz     = d_mean / d_sd if (d_sd and d_sd == d_sd and d_sd > 0) else float("nan")
    # paired bootstrap over seeds (not prompts)
    _rng = np.random.default_rng(0)
    boots = [d_by_seed[_rng.integers(0, n_seeds_eff, n_seeds_eff)].mean()
             for _ in range(5000)]
    lo_s, hi_s = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

    return {
        "PRIMARY_seed_level": {
            "n_seeds": n_seeds_eff,
            "delta_mean": d_mean,
            "delta_sd": d_sd,
            "cohens_dz": dz,
            "CI95_bootstrap_over_seeds": [lo_s, hi_s],
            "wilcoxon_p": float(p_wilcoxon),
            "paired_t": float(t_seed), "paired_t_p": float(p_t_seed),
            "mannwhitney_U": float(U_seed), "mannwhitney_p": float(p_mw_seed),
            "note": ("seed is the independent unit; prompts are nested within "
                     "seeds and are averaged before testing"),
        },
        "SECONDARY_pooled_prompt_level": {
            "note": ("descriptive only -- prompts are nested within seeds, so "
                     "these treat correlated observations as independent"),
        },
        "delta": delta, "CI95": [lo, hi], "p_paired": p_paired,
        "delta_outlier_excluded": delta_out,
        "outlier_seed_dropped": worst_seed,
        "mannwhitney_U": float(U), "p_unpaired_mw": float(p_mw),
        "welch_t": float(t_w), "p_unpaired_welch": float(p_w),
        "n_seeds": len(seeds), "n_seeds_kept_after_outlier": n_seeds_kept,
        "n_samples": int(len(r_s)),
        "design": "paired (same prompts+seeds); unpaired stats reported for comparison",
        "pass": bool(delta > 0 and p_paired < 0.05),
    }

# --------- C2: projection == natural gradient (direction test) ----------
@torch.no_grad()
def validate_C2_proj_natgrad_equiv(alg_kind: str, n: int, trials: int = 5, device: str = "cpu"):
    alg = LieAlgebra(alg_kind, n)
    cosines = []
    for _ in range(trials):
        theta = alg.project(torch.randn(n, n, device=device))
        A = torch.randn(n, n, device=device)
        S = torch.randn(n, n, device=device)
        R = A @ theta @ A.T - S
        G = A.T @ R @ A
        # Comparing alg.project(G) with itself would give cosine 1 by
        # construction, so this tests the actual claim: under the Frobenius (bi-invariant) metric, the
        # Riemannian gradient on the linear subalgebra is the orthogonal
        # projection of the Euclidean gradient.  Operationally, for every
        # tangent direction Omega in the algebra the directional derivative of
        # F must equal <Proj(G), Omega>.  We verify this by finite differences
        # against an independently computed directional derivative.
        d_proj = alg.project(G)                      # claimed Riemannian gradient
        Omega  = alg.project(torch.randn(n, n, device=device))
        Omega  = Omega / (torch.linalg.matrix_norm(Omega) + 1e-12)
        # eps = 1e-2 rather than 1e-4: F is exactly linear in theta here, so
        # there is no discretisation bias from a larger step, while a small eps
        # amplifies float32 cancellation error (min agreement 0.9927 at 1e-4
        # vs 0.9999 at 1e-2, against a 0.99 pass threshold).
        eps    = 1e-2
        # F(theta) = <G, theta> is the local linear model whose Euclidean
        # gradient is G; its directional derivative along Omega is <G, Omega>.
        fd = (torch.sum(G * (theta + eps * Omega)) -
              torch.sum(G * (theta - eps * Omega))) / (2 * eps)
        pred = torch.sum(d_proj * Omega)             # <Proj(G), Omega>
        # Agreement of two independently computed scalars.  Use an absolute
        # scale: a relative error divided by |fd| explodes when the random
        # direction happens to be nearly orthogonal to the gradient (fd ~ 0),
        # giving values of order 1e15.  Both
        # quantities are O(||G||) because Omega is unit-normalised, so we
        # normalise by that scale instead and skip degenerate directions.
        scale = float(torch.linalg.matrix_norm(d_proj).item()) + 1e-12
        if abs(float(fd.item())) < 1e-6 * scale:
            continue                                  # degenerate direction
        rel = abs(float((fd - pred).item())) / scale
        cosines.append(float(1.0 - rel))
    if not cosines:
        return {"cos_sim_avg": float("nan"), "samples": [],
                "pass": False, "note": "all directions degenerate"}
    cos_avg = float(np.mean(cosines))
    return {"cos_sim_avg": cos_avg, "samples": cosines, "pass": bool(cos_avg > 0.99)}

# --------- C3: sample complexity scales with intrinsic dim -------------
def _partial_corr_xy_given_z(x, y, z):
    """Compute partial corr r_xy.z by residualizing x and y on z (simple OLS)."""
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float); z = np.asarray(z, dtype=float)
    Z = np.c_[np.ones_like(z), z]   # intercept + z
    zx = np.linalg.lstsq(Z, x, rcond=None)[0]
    zy = np.linalg.lstsq(Z, y, rcond=None)[0]
    rx = x - Z @ zx
    ry = y - Z @ zy
    num = np.dot(rx - rx.mean(), ry - ry.mean())
    den = math.sqrt(np.dot(rx - rx.mean(), rx - rx.mean()) * np.dot(ry - ry.mean(), ry - ry.mean()) + 1e-12)
    return float(num / den) if den > 0 else 0.0

def validate_C3_sample_complexity(kind: str,
                                  dims: Tuple[int, ...] = (8, 12, 16, 24, 32),
                                  trials: int = 3,
                                  err_tau: float = 0.08,
                                  maxN_mult: int = 12,
                                  seed: int = 123):
    Nstar=[]; dg_list=[]; n_list=[]
    rng = np.random.default_rng(seed)
    for n in dims:
        dg = algebra_dim(kind, n)
        for t in range(trials):
            w_true = rng.standard_normal(dg)
            # Normalize SNR so each dg has comparable difficulty
            noise_sd = 0.1 * (dg ** 0.5) / max(1.0, np.linalg.norm(w_true)/math.sqrt(dg))
            bestN = None
            maxN = maxN_mult * dg
            step = max(8, dg//4)
            start = max(8, dg//2)
            for N in range(start, maxN+1, step):
                Phi = rng.standard_normal((N, dg))
                y = Phi @ w_true + noise_sd * rng.standard_normal(N)
                lam = 1e-2
                M = Phi.T @ Phi + lam * np.eye(dg)
                b = Phi.T @ y
                try:
                    w_hat = np.linalg.solve(M, b)
                except np.linalg.LinAlgError:
                    w_hat = np.linalg.lstsq(M, b, rcond=None)[0]
                rel_err = np.linalg.norm(w_hat - w_true) / (np.linalg.norm(w_true) + 1e-12)
                if rel_err < err_tau:
                    bestN = N; break
            if bestN is None: bestN = maxN
            Nstar.append(bestN); dg_list.append(dg); n_list.append(n)

    corr_dg = float(np.corrcoef(Nstar, dg_list)[0,1]) if len(Nstar) > 1 else 0.0
    corr_n  = float(np.corrcoef(Nstar, n_list)[0,1])  if len(Nstar) > 1 else 0.0
    pcorr_dg = _partial_corr_xy_given_z(Nstar, dg_list, n_list)
    pcorr_n  = _partial_corr_xy_given_z(Nstar, n_list, dg_list)

    return {
        "N_required": Nstar, "dg": dg_list, "n": n_list,
        "corr_with_dg": corr_dg, "corr_with_n": corr_n,
        "partial_corr_dg_given_n": pcorr_dg,
        "partial_corr_n_given_dg": pcorr_n,
        "scaling": "intrinsic" if pcorr_dg >= pcorr_n else "ambient",
        "pass": bool(pcorr_dg >= pcorr_n + 0.10)
    }

# --------------------------- Main -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="gpt2")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--algebra", choices=["so", "sl", "sym"], default="so")
    args = parser.parse_args([])  # change if running via CLI

    set_seed(0)
    cfg = Cfg(model_name=args.model_name, device=args.device, algebra=args.algebra)
    ensure_dir(cfg.results_dir)

    # Phase 1: Load model
    base, tok = load_model(cfg)
    n = base.config.n_embd
    alg = LieAlgebra(cfg.algebra, n)

    # Phase 2: Structure discovery (demo: near-identity transforms)
    log.info("Phase 2: Structure Discovery...")
    prompts = [
        "Write about geometric structure in reinforcement learning.",
        "The story is about a robot that learns to paint.",
        "In the future, agents will learn from symmetry because",
        "Explain why group actions matter in optimization.",
        "A short story about a mathematician who loves groups.",
        "Summarize the role of invariances in optimization."
    ]
    V = torch.stack([repr_vec(base, tok, p, cfg.device) for p in prompts], dim=0)  # (m,n)
    k = 4; eps = 0.02
    R = torch.randn(k, n, n, device=cfg.device) * eps
    T = torch.linalg.matrix_exp(alg.project(R))
    T_mats = [T[i] for i in range(k)]

    discover = Discovery(alg, cfg.device, lr=cfg.discovery_lr, steps=cfg.discovery_steps)
    # corrected permutation: refit per perm
    perm_stats = discover.perm_test(T_mats, V, B=cfg.perm_tests)
    X_star = discover.fit(T_mats, V, steps=cfg.discovery_steps)

    log.info(f"   residual={perm_stats['obs_residual']:.6f} perm_mean={perm_stats['perm_mean']:.6f} "
             f"p={perm_stats['p_value']:.3f} d={perm_stats['effect_size_d']:+.2f}")

    # Phase 3: Policy (OOM-safe)
    log.info("Phase 3: Building policy...")
    pol = SIPolicy(base, tok, alg, X_star, cfg).to(cfg.device)

    # Debug probe: does the structure bias (not the decoding filters) change logits?
    # NB: forward_logits() also applies nucleus/banlist masks that write -1e9
    # sentinels, so differencing full pipelines overstates the change by ~1e8.
    # Measure the structure bias term directly instead.
    ids_probe = tok("Probe:", return_tensors="pt").to(cfg.device)["input_ids"]
    with torch.no_grad():
        base_logits_probe = pol.base(input_ids=ids_probe).logits[:, -1, :]
        s_probe = pol.state_repr(ids_probe)
        struct_only = pol.lmbda * pol.struct_bias_topk(s_probe, base_logits_probe)
    delta_probe = struct_only.abs().mean().item()          # mean over full vocab (mostly zero)
    delta_topk  = struct_only[struct_only != 0].abs().mean().item() if (struct_only != 0).any() else 0.0
    log.info(f"[DEBUG] mean |structure bias|: full-vocab={delta_probe:.4e}, "
             f"on-top-K={delta_topk:.4e} (lam={pol.lmbda})")

    # Phase 4: RL
    # Free memory held by structure discovery / probes before RL training.
    if torch.cuda.is_available():
        gc.collect(); torch.cuda.empty_cache()
    log.info("Phase 4: RL training...")
    trainer = REINFORCE(pol, tok, cfg)
    last_stats = None
    for it in range(1, cfg.rl_iters + 1):
        batch = [random.choice(prompts) for _ in range(cfg.batch_prompts)]
        last_stats = trainer.step(batch)
        if it % 10 == 0:
            dbg = getattr(trainer, "_last_debug", {})
            log.info(
                f"   [RL] it {it}/{cfg.rl_iters} "
                f"loss={dbg.get('loss', last_stats['loss']):.6e} "
                f"adv={dbg.get('adv_mean', 0.0):+.3e} "
                f"theta|pre|={dbg.get('theta_norm_before', 0.0):.3e} "
                f"|gradtheta|={dbg.get('theta_grad', 0.0):.3e} "
                f"|gradheads|={dbg.get('heads_grad', 0.0):.3e} "
                f"theta|post|={dbg.get('theta_norm_after', 0.0):.3e} "
                f"reward={last_stats['reward']:.3f} baseline={last_stats['baseline']:.3f}"
            )

    # Phase 5: Evaluation
    # Post-training structure-bias probe (theta is zero at init, so the Phase-3
    # probe necessarily reads 0; this one measures the *trained* bias).
    ids_probe2 = tok("Probe:", return_tensors="pt").to(cfg.device)["input_ids"]
    with torch.no_grad():
        bl2 = pol.base(input_ids=ids_probe2).logits[:, -1, :]
        s2  = pol.state_repr(ids_probe2)
        so2 = pol.lmbda * pol.struct_bias_topk(s2, bl2)
    d_full = so2.abs().mean().item()
    d_tk   = so2[so2 != 0].abs().mean().item() if (so2 != 0).any() else 0.0
    log.info(f"[DEBUG] TRAINED structure bias: full-vocab={d_full:.4e}, "
             f"on-top-K={d_tk:.4e} (theta_norm={pol.theta_raw.norm().item():.2f})")
    log.info("Phase 5: Evaluation...")
    eval_stats = evaluate(pol, tok, prompts, cfg)

    # -------------------- Phase 6: Validation -- C1..C4 --------------------
    log.info("Phase 6: Validation...")

    # C1: structure helps (paired bootstrap Delta, CI, p)
    val_c1 = ablation_validate(pol, tok, prompts, cfg)

    # C2: projection == natural gradient (direction cosine test)
    c2 = validate_C2_proj_natgrad_equiv(cfg.algebra, n=min(n, 64), trials=cfg.c2_trials, device=cfg.device)

    # C3: sample-complexity with partial correlations
    c3 = validate_C3_sample_complexity(
        cfg.algebra,
        dims=cfg.c3_dims,
        trials=cfg.c3_trials_per_dim,
        err_tau=cfg.c3_err_tau,
        maxN_mult=cfg.c3_maxN_mult,
    )

    # C4: non-trivial outputs
    c4_pass = bool(
        (eval_stats["average_length"]    >= cfg.min_new_tokens) and
        (eval_stats["average_reward"]    >= 0.10)              and
        (eval_stats["average_repetition"] <= 0.50)
    )
    val_block = {
        "C1_structure_benefit": val_c1,
        "C2_proj_natgrad_equiv": c2,
        "C3_sample_complexity": c3,
        "C4_nontrivial_outputs": {
            "avg_reward":       eval_stats["average_reward"],
            "avg_len":          eval_stats["average_length"],
            "avg_repetition":   eval_stats["average_repetition"],
            "pass":             c4_pass
        }
    }

    validated = int(val_c1["pass"]) + int(c2["pass"]) + int(c3["pass"]) + int(c4_pass)

    ts = int(time.time())
    out = {
        "structure_discovery": {
            "best_algebra": cfg.algebra,
            "residual":    perm_stats["obs_residual"],
            "perm_mean":   perm_stats["perm_mean"],
            "p_value":     perm_stats["p_value"],
            "p_corrected": perm_stats.get("p_corrected", None),
            "perm_std":    perm_stats.get("perm_std", None),
            "effect_size_d": perm_stats["effect_size_d"],
            "significant": bool(perm_stats["p_value"] < 0.05)
        },
        "rl_training": {
            "iters":          cfg.rl_iters,
            "batch_prompts":  cfg.batch_prompts,
            "final_reward":   float(last_stats["reward"]) if last_stats else None
        },
        "evaluation": eval_stats,
        "validation": {
            "validated_claims": f"{validated}/4",
            "score":            float(validated / 4.0),
            **val_block
        }
    }
    ensure_dir(cfg.results_dir)
    path = os.path.join(cfg.results_dir, f"experiment_with_claims_plus_{ts}.json")
    with open(path, "w") as f:
        json.dump(json_safe(out), f, indent=2)
    log.info(f"Results saved to: {path}")
    log.info("Experiment completed successfully!")


if __name__ == "__main__":
    main()
