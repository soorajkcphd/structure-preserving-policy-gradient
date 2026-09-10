"""
PRIORITY 3 -- LoRA Baseline for Task 1
======================================
SP-PPO so(32) vs LoRA-PPO at matched parameter budgets.
Uses same GeometricResponseEnv and per-action theta architecture as ablation_factorial.py.

Conditions:
  1. Baseline PPO       unconstrained  theta_a in R^{nxn}     params/action: n^2=1024
  2. LoRA PPO  r=1      low-rank       theta_a = A_a@B_a^T   params/action: 2n*1=64
  3. LoRA PPO  r=8      low-rank       theta_a = A_a@B_a^T   params/action: 2n*8=512
  4. LoRA PPO  r=16     low-rank       theta_a = A_a@B_a^T   params/action: 2n*16=1024
  5. SP-PPO so(32)      compact Lie    theta_a in so(n)        params/action: n(n-1)/2=496

Key comparison: so(32) [496/action] vs LoRA r=8 [512/action] -- equal budget.
If SP-PPO beats LoRA r=8: advantage is compactness (|lam|=1), not dimension.

Usage:
    python baseline_lora.py            # full: 10 seeds x 60 iters
    python baseline_lora.py --quick    # smoke: 3 seeds x 20 iters

Outputs:
    lora_comparison.csv / .json / _table.tex
"""

import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import csv
import time
from typing import List, Tuple, Dict, Optional
from scipy import stats

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -- Reproducibility -----------------------------------------------------------
def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def proj_so(M: torch.Tensor) -> torch.Tensor:
    return 0.5 * (M - M.T)


def random_SO_element(n: int, scale: float = 1.5,
                      gen: Optional[torch.Generator] = None) -> torch.Tensor:
    X = proj_so(torch.randn(n, n, generator=gen) * scale)
    return torch.matrix_exp(X)


def build_G_ref(G_list: List[torch.Tensor]) -> torch.Tensor:
    lie_dim = G_list[0].shape[0]
    X_sum   = torch.zeros(lie_dim, lie_dim)
    for G in G_list:
        X_sum += proj_so(G.cpu())
    X_mean = X_sum / len(G_list)
    X_star = X_mean / (X_mean.norm(p='fro').clamp(min=1e-8)) * 0.3
    return torch.matrix_exp(X_star).to(DEVICE)


# -- Environment (identical to ablation_factorial.py) -------------------------
class GeometricResponseEnv:
    """
    16 SO(32)-structured states. Correct action = alignment quartile.
    See ablation_factorial.py for full documentation.
    """
    def __init__(self, state_dim=1024, n_prompts=16, lie_dim=32, seed=42):
        self.n_prompts = n_prompts
        self.n_actions = 4
        self.T         = 3
        self.lie_dim   = lie_dim
        self.state_dim = state_dim

        gen = torch.Generator()
        gen.manual_seed(seed)

        B_raw    = torch.randn(lie_dim, lie_dim, generator=gen)
        self.B_0 = B_raw / B_raw.norm(p='fro').clamp(min=1e-8)

        self.G_list = [random_SO_element(lie_dim, scale=1.5, gen=gen)
                       for _ in range(n_prompts)]
        self.G_ref  = build_G_ref(self.G_list)

        embeddings = []
        for G_i in self.G_list:
            H_i  = G_i.cpu() @ self.B_0
            vec  = H_i.reshape(-1)
            reps = math.ceil(state_dim / vec.shape[0])
            emb  = vec.repeat(reps)[:state_dim]
            emb  = emb / emb.norm(p=2).clamp(min=1e-8)
            embeddings.append(emb)
        self.embeddings = torch.stack(embeddings).to(DEVICE)

        G_ref_cpu  = self.G_ref.cpu()
        alignments = torch.tensor([
            (G_i.cpu().T @ G_ref_cpu).trace().item() / lie_dim
            for G_i in self.G_list
        ])
        idx = torch.argsort(alignments)
        correct    = torch.zeros(n_prompts, dtype=torch.long)
        correct[idx[12:]]  = 1
        correct[idx[8:12]] = 3
        correct[idx[4:8]]  = 2
        correct[idx[:4]]   = 0
        self.correct_actions = correct.to(DEVICE)

    def get_states(self): return self.embeddings
    def reward(self, actions): return (actions == self.correct_actions).float()


# -- SP-PPO / Baseline Policy (per-action theta_a) -----------------------------
class GeometricPolicy(nn.Module):
    """
    Per-action theta_a matrices. score_a(s) = tr(H_s^T norm(theta_a)).
    use_projection=True  -> theta_a in so(lie_dim)   [SP-PPO]
    use_projection=False -> theta_a in R^{nxn}        [Baseline]
    """
    def __init__(self, state_dim=1024, n_actions=4, lie_dim=32,
                 use_projection=True):
        super().__init__()
        self.use_projection = use_projection
        self.lie_dim   = lie_dim
        self.n_actions = n_actions
        self.action_thetas = nn.Parameter(
            torch.randn(n_actions, lie_dim, lie_dim) * 0.01
        )

    def _theta(self, a: int) -> torch.Tensor:
        t = self.action_thetas[a]
        return proj_so(t) if self.use_projection else t

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        k = self.lie_dim
        H = state[:, :k * k].view(-1, k, k)
        logits = []
        for a in range(self.n_actions):
            theta_a = self._theta(a)
            theta_n = theta_a / theta_a.norm(p='fro').clamp(min=1e-8)
            score   = torch.sum(H * theta_n.unsqueeze(0), dim=(1, 2))
            logits.append(score)
        return torch.stack(logits, dim=1)

    def project_theta_(self):
        if self.use_projection:
            with torch.no_grad():
                for a in range(self.n_actions):
                    self.action_thetas[a].copy_(proj_so(self.action_thetas[a]))

    def n_constrained_params(self) -> int:
        if self.use_projection:
            return self.n_actions * self.lie_dim * (self.lie_dim - 1) // 2
        return self.n_actions * self.lie_dim * self.lie_dim

    def spectral_radius(self) -> float:
        radii = []
        with torch.no_grad():
            for a in range(self.n_actions):
                exp_t = torch.matrix_exp(self._theta(a))
                radii.append(torch.linalg.eigvals(exp_t).abs().max().item())
        return float(np.mean(radii))


# -- LoRA Policy (per-action A_a, B_a matrices) --------------------------------
class LoRAGeometricPolicy(nn.Module):
    """
    Per-action LoRA parameterisation.
    theta_a = (alpha/r) * A_a @ B_a^T,  A_a, B_a in R^{lie_dim x r}.
    score_a(s) = tr(H_s^T norm(theta_a))

    Parameter count per action: 2 * lie_dim * r.
    Scale-matched init: sigma chosen so E[||theta_a||_F] ~= ||so(32) theta_raw||_F.
    spectral_radius(): uses exp(theta_a) for consistent comparison with so(32).
    """
    def __init__(self, state_dim=1024, n_actions=4, lie_dim=32,
                 lora_rank=1, lora_alpha=1.0):
        super().__init__()
        self.lie_dim    = lie_dim
        self.lora_rank  = lora_rank
        self.n_actions  = n_actions
        self.scaling    = lora_alpha / lora_rank

        # Scale-matched init: E[||theta_a||_F] matches so(32) init norm
        target_fro = 0.01 * math.sqrt(lie_dim * (lie_dim - 1) / 2)
        sigma      = math.sqrt(target_fro * math.sqrt(lora_rank) / lie_dim)

        # Per-action LoRA matrices: (n_actions, lie_dim, lora_rank)
        self.lora_A = nn.Parameter(
            torch.randn(n_actions, lie_dim, lora_rank) * sigma
        )
        self.lora_B = nn.Parameter(
            torch.randn(n_actions, lie_dim, lora_rank) * sigma
        )

    def _theta(self, a: int) -> torch.Tensor:
        return self.scaling * (self.lora_A[a] @ self.lora_B[a].T)  # (lie_dim, lie_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        k = self.lie_dim
        H = state[:, :k * k].view(-1, k, k)
        logits = []
        for a in range(self.n_actions):
            theta_a = self._theta(a)
            theta_n = theta_a / theta_a.norm(p='fro').clamp(min=1e-8)
            score   = torch.sum(H * theta_n.unsqueeze(0), dim=(1, 2))
            logits.append(score)
        return torch.stack(logits, dim=1)

    def project_theta_(self):
        pass   # LoRA: no projection

    def n_constrained_params(self) -> int:
        return self.n_actions * 2 * self.lie_dim * self.lora_rank

    def spectral_radius(self) -> float:
        """Uses exp(theta_a) for direct comparison with so(32) (which gives 1.0)."""
        radii = []
        with torch.no_grad():
            for a in range(self.n_actions):
                exp_t = torch.matrix_exp(self._theta(a))
                radii.append(torch.linalg.eigvals(exp_t).abs().max().item())
        return float(np.mean(radii))


# -- Value network -------------------------------------------------------------
class GeometricValueNet(nn.Module):
    def __init__(self, lie_dim=32):
        super().__init__()
        self.lie_dim = lie_dim
        k = lie_dim
        self.net = nn.Sequential(
            nn.Linear(k * k, 64), nn.ReLU(),
            nn.Linear(64, 32),    nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        k = self.lie_dim
        return self.net(x[:, :k * k]).squeeze(-1)


# -- GAE -----------------------------------------------------------------------
def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    T        = len(rewards)
    adv      = torch.zeros(T, device=DEVICE)
    last_gae = 0.0
    for t in reversed(range(T)):
        nv       = values[t + 1].item() if t < T - 1 else 0.0
        delta    = rewards[t].item() + gamma * nv - values[t].item()
        last_gae = delta + gamma * lam * last_gae
        adv[t]   = last_gae
    return adv, adv + values


# -- Generic PPO update --------------------------------------------------------
def ppo_update_generic(
    policy, value_net, policy_opt, value_opt,
    states, actions, old_log_probs, advantages, returns,
    is_spppo=False,
    clip_eps=0.2, ppo_epochs=4, batch_size=128, entropy_coef=0.03,
) -> None:
    n = states.shape[0]
    for _ in range(ppo_epochs):
        idx = torch.randperm(n, device=DEVICE)
        for start in range(0, n, batch_size):
            mb     = idx[start:start + batch_size]
            logits = policy(states[mb])
            dist   = torch.distributions.Categorical(logits=logits)
            new_lp = dist.log_prob(actions[mb])
            ent    = dist.entropy().mean()

            ratio  = torch.exp(new_lp - old_log_probs[mb])
            s1     = ratio * advantages[mb]
            s2     = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages[mb]
            loss   = -torch.min(s1, s2).mean() - entropy_coef * ent

            policy_opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            policy_opt.step()
            if is_spppo:
                policy.project_theta_()

            vl = F.mse_loss(value_net(states[mb]), returns[mb])
            value_opt.zero_grad()
            vl.backward()
            value_opt.step()


# -- Single training run -------------------------------------------------------
def train_single(
    policy_type: str,
    seed: int,
    n_iterations: int = 60,
    state_dim: int    = 1024,
    lie_dim: int      = 32,
    lr: float         = 3e-4,
) -> Tuple[float, List[float], List[float]]:
    set_seed(seed)
    env = GeometricResponseEnv(state_dim=state_dim, n_prompts=16,
                               lie_dim=lie_dim, seed=42)

    is_spppo = False
    if policy_type == "baseline":
        policy = GeometricPolicy(state_dim, 4, lie_dim, use_projection=False).to(DEVICE)
    elif policy_type == "spppo":
        policy   = GeometricPolicy(state_dim, 4, lie_dim, use_projection=True).to(DEVICE)
        is_spppo = True
    elif policy_type == "lora_r1":
        policy = LoRAGeometricPolicy(state_dim, 4, lie_dim, lora_rank=1).to(DEVICE)
    elif policy_type == "lora_r8":
        policy = LoRAGeometricPolicy(state_dim, 4, lie_dim, lora_rank=8).to(DEVICE)
    elif policy_type == "lora_r16":
        policy = LoRAGeometricPolicy(state_dim, 4, lie_dim, lora_rank=16).to(DEVICE)
    else:
        raise ValueError(f"Unknown policy_type: {policy_type!r}")

    value_net = GeometricValueNet(lie_dim).to(DEVICE)
    p_opt     = torch.optim.Adam(policy.parameters(), lr=lr)
    v_opt     = torch.optim.Adam(value_net.parameters(), lr=lr)

    all_states    = env.get_states()
    T             = env.T
    auc_list: List[float] = []
    rad_list: List[float] = []

    for _ in range(n_iterations):
        ts, ta, tr, tlp, tv = [], [], [], [], []
        policy.eval()
        with torch.no_grad():
            for _step in range(T):
                logits = policy(all_states)
                dist   = torch.distributions.Categorical(logits=logits)
                acts   = dist.sample()
                lps    = dist.log_prob(acts)
                rews   = env.reward(acts)
                vals   = value_net(all_states)
                ts.append(all_states); ta.append(acts); tr.append(rews)
                tlp.append(lps);       tv.append(vals)

        s_f  = torch.cat(ts,  0); a_f  = torch.cat(ta,  0)
        r_f  = torch.cat(tr,  0); lp_f = torch.cat(tlp, 0)
        v_f  = torch.cat(tv,  0)

        adv, ret = compute_gae(r_f, v_f)
        adv      = (adv - adv.mean()) / (adv.std() + 1e-8)

        policy.train()
        ppo_update_generic(
            policy, value_net, p_opt, v_opt,
            s_f, a_f, lp_f.detach(), adv.detach(), ret.detach(),
            is_spppo=is_spppo,
        )
        auc_list.append(r_f.mean().item() * T)
        rad_list.append(policy.spectral_radius())

    return sum(auc_list), auc_list, rad_list


# -- Parameter analysis --------------------------------------------------------
def print_param_analysis():
    n          = 32
    target_fro = 0.01 * math.sqrt(n * (n - 1) / 2)
    print("\nParameter count (constrained theta, per action x n_actions=4):")
    print(f"  {'Method':<22} {'Type':<22} {'Params/act':>10}  {'Total':>7}  Spectral")
    print("  " + "-" * 72)
    print(f"  {'Baseline PPO':<22} {'Unconstrained':<22} {n*n:>10}  {4*n*n:>7}  None")
    for r in (1, 8, 16):
        sigma = math.sqrt(target_fro * math.sqrt(r) / n)
        print(f"  {'LoRA PPO r='+str(r):<22} {'Low-rank r='+str(r):<22}"
              f" {2*n*r:>10}  {4*2*n*r:>7}  None")
    print(f"  {'SP-PPO so(32)':<22} {'Compact Lie algebra':<22}"
          f" {n*(n-1)//2:>10}  {4*n*(n-1)//2:>7}  |lam(exp(theta_a))|=1")
    print()
    print(f"  Key: so(32) [496/action] ~= LoRA r=8 [512/action] -- equal per-action budget.")
    print(f"  If SP-PPO beats LoRA r=8 -> compactness (not dim reduction) drives the gain.")


# -- Full comparison run -------------------------------------------------------
def run_lora_comparison(
    seeds=list(range(10)), n_iterations=60,
) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
    conditions = {
        "Baseline PPO":    "baseline",
        "LoRA PPO (r=1)":  "lora_r1",
        "LoRA PPO (r=8)":  "lora_r8",
        "LoRA PPO (r=16)": "lora_r16",
        "SP-PPO so(32)":   "spppo",
    }
    results:     Dict[str, List[float]] = {k: [] for k in conditions}
    rad_summary: Dict[str, List[float]] = {k: [] for k in conditions}

    total = len(conditions) * len(seeds)
    run_n = 0
    for name, ptype in conditions.items():
        for seed in seeds:
            run_n += 1
            print(f"[{run_n:3d}/{total}] {name:22s} | seed={seed}", end=" ... ")
            t0 = time.time()
            auc, _, rads = train_single(ptype, seed, n_iterations)
            results[name].append(auc)
            rad_summary[name].append(float(np.mean(rads)))
            print(f"AUC={auc:.4f}  ({time.time()-t0:.1f}s)  rad={np.mean(rads):.3f}")

    return results, rad_summary


# -- Summary and outputs -------------------------------------------------------
def summarise_lora(results: Dict, rad_summary: Dict):
    print("\n" + "=" * 72 + "\nLORA vs SP-PPO COMPARISON\n" + "=" * 72)
    bm = np.mean(results["Baseline PPO"])
    summaries: Dict = {}

    for name, aucs in results.items():
        m  = np.mean(aucs); s = np.std(aucs, ddof=1)
        n  = len(aucs);     se = s / np.sqrt(n)
        tc = stats.t.ppf(0.975, df=n - 1)
        ci = (m - tc * se, m + tc * se)
        dp = 100 * (m - bm) / (bm + 1e-12)
        rm = np.mean(rad_summary[name])
        summaries[name] = {"mean":m,"std":s,"ci":ci,"delta_pct":dp,"aucs":aucs,"rad":rm}
        print(f"  {name:22s}: {m:.4f}+/-{s:.4f}  "
              f"[{ci[0]:.4f},{ci[1]:.4f}]  Delta={dp:+.1f}%  rad={rm:.3f}")

    print("\n  Pairwise Welch t vs SP-PPO so(32):")
    sp = results["SP-PPO so(32)"]
    for name, aucs in results.items():
        if name == "SP-PPO so(32)": continue
        t_s, p_w = stats.ttest_ind(sp, aucs, equal_var=False)
        print(f"    SP-PPO vs {name:22s}: t={t_s:+.3f}, p={p_w:.4f}")

    # CSV
    with open("lora_comparison.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "seed", "auc", "rad"])
        for name, aucs in results.items():
            for i, auc in enumerate(aucs):
                w.writerow([name, i, auc, rad_summary[name][i]])
    print("\nWritten: lora_comparison.csv")

    # JSON
    with open("lora_comparison.json", "w") as f:
        out = {}
        for name, d in summaries.items():
            out[name] = {k: (round(v,6) if isinstance(v,float) else v)
                         for k, v in d.items() if k != "aucs"}
            out[name]["aucs"] = [round(v, 6) for v in d["aucs"]]
        json.dump(out, f, indent=2)
    print("Written: lora_comparison.json")

    # LaTeX
    lie_dim = 32
    pc = {
        "Baseline PPO":    4 * lie_dim * lie_dim,
        "LoRA PPO (r=1)":  4 * 2 * lie_dim * 1,
        "LoRA PPO (r=8)":  4 * 2 * lie_dim * 8,
        "LoRA PPO (r=16)": 4 * 2 * lie_dim * 16,
        "SP-PPO so(32)":   4 * lie_dim * (lie_dim - 1) // 2,
    }
    sp_g = {k: "None" for k in pc}
    sp_g["SP-PPO so(32)"] = "$|\\lambda|{=}1$"
    tex_rows = []
    for name, d in summaries.items():
        ci = d["ci"]; dp = d["delta_pct"]
        ds = ("---" if abs(dp) < 0.01
              else f"$+{dp:.1f}\\%$" if dp > 0 else f"${dp:.1f}\\%$")
        tex_rows.append(
            f"  {name:25s} & {pc[name]:>6d} & {sp_g[name]:15s}"
            f" & ${d['mean']:.4f}\\pm{d['std']:.4f}$"
            f" & $[{ci[0]:.4f},\\,{ci[1]:.4f}]$ & {ds} \\\\"
        )
    tex = (
        "\\begin{table}[htbp]\n\\centering\n"
        "\\caption{SP-PPO vs LoRA-PPO (Task~1 geometric alignment, 10~seeds).\n"
        "  Each method uses 4 per-action theta matrices. "
        "$\\mathfrak{so}(32)$ [496/action] and LoRA $r{=}8$ [512/action] have equal budget.\n"
        "  Only SP-PPO provides $|\\lambda_{\\max}(\\exp(\\theta_a))|{=}1$ throughout.}\n"
        "\\label{tab:lora_comparison}\\scriptsize\n"
        "\\begin{tabular}{lrp{1.4cm}ccr}\\toprule\n"
        "\\textbf{Method} & \\textbf{Total params} & \\textbf{Spectral}"
        " & \\textbf{AUC mean$\\pm$std} & \\textbf{95\\%~CI} & $\\Delta$ \\\\\n"
        "\\midrule\n"
        + "\n".join(tex_rows)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )
    with open("lora_table.tex", "w") as f:
        f.write(tex)
    print("Written: lora_table.tex")


# -- Main ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="3 seeds x 20 iters (smoke test, ~3 min)")
    args   = parser.parse_args()
    seeds  = list(range(3)) if args.quick else list(range(10))
    n_iter = 20             if args.quick else 60

    print(f"Device: {DEVICE} | Seeds: {seeds} | Iterations: {n_iter}")
    print(f"Policy: per-action theta_a  [GeometricPolicy / LoRAGeometricPolicy]")
    print_param_analysis()
    print()

    results, rad_summary = run_lora_comparison(seeds=seeds, n_iterations=n_iter)
    summarise_lora(results, rad_summary)

    sp = np.mean(results["SP-PPO so(32)"])
    l8 = np.mean(results["LoRA PPO (r=8)"])
    print(f"\n  SP-PPO so(32) [4x496=1984 params]: AUC={sp:.4f}  rad~=1.000")
    print(f"  LoRA PPO r=8  [4x512=2048 params]: AUC={l8:.4f}  rad~={np.mean(rad_summary['LoRA PPO (r=8)']):.3f}")
    if sp > l8:
        print(f"  [OK] SP-PPO beats LoRA r=8 by {sp-l8:.4f} AUC -> compactness drives the gain.")
    else:
        print(f"  [WARNING] Run full 10-seed version to confirm trend.")
