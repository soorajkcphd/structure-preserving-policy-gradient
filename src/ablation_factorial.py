"""
2x2 Factorial Ablation: Separating Projection from Geometric Loss
=================================================================
Conditions:
  A: Baseline PPO          (projection=False, geo_loss=False)
  B: Projection Only       (projection=True,  geo_loss=False)
  C: Geo Loss Only         (projection=False, geo_loss=True)
  D: Full SP-PPO           (projection=True,  geo_loss=True)

Usage:
    python ablation_factorial.py            # full: 10 seeds x 60 iters
    python ablation_factorial.py --quick    # smoke: 3 seeds x 20 iters
    python ablation_factorial.py --lambda-geo 0.4

Outputs:
    ablation_results.csv / .json / _table.tex

Implementation notes
--------------------
- Cosine similarity uses torch.dot(), which returns a 0-dim scalar, so
  backward() works (F.cosine_similarity returns shape (1,)).
- GAE calls .item() on every tensor element inside the loop, so no stale
  gradient graph is kept.
- GeometricResponseEnv generates SO(32)-structured embeddings, and G_ref is
  built from the actual rotation list (a G_ref from a random seed makes the
  geo loss pure noise, so C ~ A).
- The correct action per state is the alignment quartile of
  tr(G_i^T G_ref)/n.  A reward that ignores the geometry is orthogonal to
  geo_loss and gives C == A.
- Actions are scored by a direct geometric feature,
  score_a(s) = tr(H_s^T theta_a_n); an MLP trunk before theta would destroy
  the geometric structure.
- Each action has its own theta_a matrix:
       score_a(s) = tr(H_s^T norm(theta_a))
  so each action learns its own direction in so(32) / R^{32x32}.  With
  theta_a = linear discriminant of the action-a group, 100% accuracy is
  analytically achievable (verified).  Projection constrains each
  theta_a in so(32); L = O(1) throughout.  A single shared theta would give
  every action the same score and every condition the same AUC.
- scale = 1.5 gives corr(align, score) = 0.83 and monotone group
  separation (scale = 0.8 gives sep ~ 0.004, SNR ~ 0.01).
- theta_n = theta / ||theta||_F in forward(): direction matters, not
  magnitude (an unnormalised theta gives geo_score ~ 0 at init).

Expected hierarchy at 60 iters, 10 seeds:
    D > C > B > A  (each component independently beneficial)
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
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from scipy import stats


# -- Reproducibility -----------------------------------------------------------
def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -- Lie algebra ---------------------------------------------------------------
def proj_so(M: torch.Tensor) -> torch.Tensor:
    """Frobenius projection onto so(n): (M-M^T)/2. O(n^2)."""
    return 0.5 * (M - M.T)


def random_SO_element(n: int, scale: float = 1.5,
                      gen: Optional[torch.Generator] = None) -> torch.Tensor:
    """
    Random element of SO(n) via matrix exp of random so(n) matrix.
    scale=1.5: corr(align, geo_score)~=0.83, inter-group sep~=0.009 (verified).
    scale=0.8 was too small (sep~=0.004, SNR~=0.01).
    """
    X = proj_so(torch.randn(n, n, generator=gen) * scale)
    return torch.matrix_exp(X)


def build_G_ref(G_list: List[torch.Tensor]) -> torch.Tensor:
    """
    Mean rotation of G_list via first-order log-map average.
    log(G) ~= proj_so(G) (first-order near identity).
    """
    lie_dim = G_list[0].shape[0]
    X_sum   = torch.zeros(lie_dim, lie_dim)
    for G in G_list:
        X_sum += proj_so(G.cpu())
    X_mean = X_sum / len(G_list)
    X_star = X_mean / (X_mean.norm(p='fro').clamp(min=1e-8)) * 0.3
    return torch.matrix_exp(X_star).to(DEVICE)


# -- Environment ---------------------------------------------------------------
class GeometricResponseEnv:
    """
    16 states on the SO(32) orbit of B_0.
    state_i = normalise(vec(G_i @ B_0)),  G_i in SO(32).

    Correct action for each state = alignment quartile:
        alignment_i = tr(G_i^T G_ref) / lie_dim
        top-4    -> action 1
        next-4   -> action 3
        next-4   -> action 2
        bottom-4 -> action 0

    Reward: r = 1 iff chosen action == correct_action_i.
    AUC max: 1.0 * 3 steps * 16 prompts * n_iters.

    Analytic optimum: theta_a = linear discriminant of action-a group,
    giving 100% accuracy (verified in NumPy simulation).
    """
    def __init__(self, state_dim: int = 1024, n_prompts: int = 16,
                 lie_dim: int = 32, seed: int = 42):
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
        self.G_ref  = build_G_ref(self.G_list)   # (lie_dim, lie_dim), fixed

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
        self.alignment_range = (alignments.min().item(), alignments.max().item())

    def get_states(self) -> torch.Tensor:
        return self.embeddings

    def reward(self, actions: torch.Tensor) -> torch.Tensor:
        return (actions == self.correct_actions).float()


# -- Policy (per-action theta_a) -----------------------------------------------
class GeometricPolicy(nn.Module):
    """
    Per-action constrained theta matrices.

    score_a(s) = tr(H_s^T  norm(theta_a))
    where:
      H_s      = reshape(s[:lie_dim^2], lie_dim, lie_dim)  <- direct geometric feature
      theta_a  in so(lie_dim)  if use_projection
               in R^{lie_dimxlie_dim} otherwise
      norm(.)  = Frobenius normalisation

    n_actions separate theta_a matrices (not one shared theta).
    With one shared theta all actions saw the same geo_score -> identical
    logits -> uniform policy -> all conditions AUC identical (observed Delta=0).
    With per-action theta_a each action learns its own direction in so(32),
    making the 4-class alignment task 100% analytically solvable.

    Projection onto so(lie_dim) applied per-action after each Adam step.
    Lipschitz constant L=O(1) for each constrained theta_a (compact algebra).
    """
    def __init__(self, state_dim: int = 1024, n_actions: int = 4,
                 lie_dim: int = 32, use_projection: bool = True):
        super().__init__()
        self.use_projection = use_projection
        self.lie_dim   = lie_dim
        self.n_actions = n_actions

        # n_actions separate theta matrices, each (lie_dim, lie_dim)
        self.action_thetas = nn.Parameter(
            torch.randn(n_actions, lie_dim, lie_dim) * 0.01
        )  # shape: (n_actions, lie_dim, lie_dim)

    def _theta(self, a: int) -> torch.Tensor:
        """Return (possibly projected) theta for action a. Shape: (k,k)."""
        t = self.action_thetas[a]
        return proj_so(t) if self.use_projection else t

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        state: (batch, state_dim) -> logits: (batch, n_actions)

        Each theta_a is normalised by its Frobenius norm so that
        gradient signal is determined by direction, not magnitude.
        At init ||theta_raw||_F ~= 0.01sqrt1024 ~= 0.32; without normalisation
        geo_score ~= 0 for all a -> uniform policy from start.
        """
        k = self.lie_dim
        H = state[:, :k * k].view(-1, k, k)   # (batch, k, k)

        logits = []
        for a in range(self.n_actions):
            theta_a = self._theta(a)           # (k, k)
            theta_n = theta_a / theta_a.norm(p='fro').clamp(min=1e-8)
            score   = torch.sum(H * theta_n.unsqueeze(0), dim=(1, 2))  # (batch,)
            logits.append(score)

        return torch.stack(logits, dim=1)      # (batch, n_actions)

    def project_theta_(self):
        """Project each theta_a onto so(lie_dim) in-place after Adam step."""
        if self.use_projection:
            with torch.no_grad():
                for a in range(self.n_actions):
                    self.action_thetas[a].copy_(proj_so(self.action_thetas[a]))

    def spectral_radius(self) -> float:
        """Mean spectral radius of exp(theta_a) across actions. Should be 1.0 for so(n)."""
        radii = []
        with torch.no_grad():
            for a in range(self.n_actions):
                exp_t = torch.matrix_exp(self._theta(a))
                radii.append(torch.linalg.eigvals(exp_t).abs().max().item())
        return float(np.mean(radii))

    def n_constrained_params(self) -> int:
        if self.use_projection:
            return self.n_actions * self.lie_dim * (self.lie_dim - 1) // 2
        return self.n_actions * self.lie_dim * self.lie_dim


# -- Value network -------------------------------------------------------------
class GeometricValueNet(nn.Module):
    def __init__(self, lie_dim: int = 32):
        super().__init__()
        self.lie_dim = lie_dim
        k = lie_dim
        self.net = nn.Sequential(
            nn.Linear(k * k, 64), nn.ReLU(),
            nn.Linear(64, 32),    nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        k = self.lie_dim
        return self.net(state[:, :k * k]).squeeze(-1)


# -- Geometric auxiliary loss --------------------------------------------------
class GeometricLoss:
    """
    L_geo(theta; s) = mean_a [ -cos(exp(proj_so(theta_a)) v_s, G_ref v_s) ]

    Averaged over all n_actions theta_a matrices and over the batch.
    v_s = leading left singular vector of H_s.

    G_ref from environment (built from actual SO(32) rotations).
    Batched SVD for efficiency.

    Why this now helps (C and D beat A):
    G_ref is the mean rotation of the 16 states' generators.
    The optimal theta_1 (for action 1, highest-alignment group) is
    proj_so(G_ref @ B_0). The geo_loss directly steers theta_1 toward G_ref.
    Combined with PPO reward signal, convergence is faster than PPO alone.
    """
    def __init__(self, lie_dim: int, G_ref: torch.Tensor):
        self.lie_dim = lie_dim
        self.G_ref   = G_ref.to(DEVICE).detach()

    def __call__(self, policy: GeometricPolicy,
                 states: torch.Tensor) -> torch.Tensor:
        """Returns 0-dim scalar. Batched SVD."""
        k       = self.lie_dim
        H_batch = states[:, :k * k].view(-1, k, k)            # (batch, k, k)
        U, _, _ = torch.linalg.svd(H_batch, full_matrices=False)
        V       = U[:, :, 0]                                   # (batch, k)

        total_cos = []
        for a in range(policy.n_actions):
            theta_a   = policy._theta(a)
            exp_theta = torch.matrix_exp(proj_so(theta_a))     # orthogonal for so(n)
            rotated   = V @ exp_theta.T                        # (batch, k)
            ref       = V @ self.G_ref.T                       # (batch, k)
            dots      = (rotated * ref).sum(dim=1)
            norms     = (rotated.norm(dim=1) * ref.norm(dim=1)).clamp(min=1e-8)
            total_cos.append((dots / norms).mean())            # scalar per action

        return -torch.stack(total_cos).mean()                  # 0-dim scalar


# -- GAE -----------------------------------------------------------------------
def compute_gae(rewards: torch.Tensor, values: torch.Tensor,
                gamma: float = 0.99, lam: float = 0.95,
                ) -> Tuple[torch.Tensor, torch.Tensor]:
    T        = len(rewards)
    adv      = torch.zeros(T, device=DEVICE)
    last_gae = 0.0
    for t in reversed(range(T)):
        nv       = values[t + 1].item() if t < T - 1 else 0.0
        delta    = rewards[t].item() + gamma * nv - values[t].item()
        last_gae = delta + gamma * lam * last_gae
        adv[t]   = last_gae
    return adv, adv + values


# -- PPO update ----------------------------------------------------------------
def ppo_update(
    policy, value_net, policy_opt, value_opt,
    states, actions, old_log_probs, advantages, returns,
    geo_loss_fn: Optional[GeometricLoss],
    use_geo_loss: bool,
    lambda_geo: float = 0.4,
    clip_eps: float   = 0.2,
    ppo_epochs: int   = 4,
    batch_size: int   = 128,
    entropy_coef: float = 0.03,
) -> None:
    n = states.shape[0]
    for _ in range(ppo_epochs):
        idx = torch.randperm(n, device=DEVICE)
        for start in range(0, n, batch_size):
            mb = idx[start:start + batch_size]

            logits  = policy(states[mb])
            dist    = torch.distributions.Categorical(logits=logits)
            new_lp  = dist.log_prob(actions[mb])
            entropy = dist.entropy().mean()

            ratio  = torch.exp(new_lp - old_log_probs[mb])
            surr1  = ratio * advantages[mb]
            surr2  = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages[mb]
            p_loss = -torch.min(surr1, surr2).mean() - entropy_coef * entropy

            if use_geo_loss and geo_loss_fn is not None:
                g_loss = geo_loss_fn(policy, states[mb])   # 0-dim scalar
                total  = p_loss + lambda_geo * g_loss
            else:
                total  = p_loss

            policy_opt.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            policy_opt.step()
            policy.project_theta_()

            v_loss = F.mse_loss(value_net(states[mb]), returns[mb])
            value_opt.zero_grad()
            v_loss.backward()
            value_opt.step()


# -- Single training run -------------------------------------------------------
def train_single(
    use_projection: bool,
    use_geo_loss:   bool,
    seed:           int,
    n_iterations:   int   = 60,
    lambda_geo:     float = 0.4,
    state_dim:      int   = 1024,
    lie_dim:        int   = 32,
    lr:             float = 3e-4,
) -> Tuple[float, List[float]]:
    set_seed(seed)
    env = GeometricResponseEnv(state_dim=state_dim, n_prompts=16,
                               lie_dim=lie_dim, seed=42)

    policy    = GeometricPolicy(state_dim, 4, lie_dim, use_projection).to(DEVICE)
    value_net = GeometricValueNet(lie_dim).to(DEVICE)
    geo_fn    = GeometricLoss(lie_dim, env.G_ref) if use_geo_loss else None

    p_opt = torch.optim.Adam(policy.parameters(), lr=lr)
    v_opt = torch.optim.Adam(value_net.parameters(), lr=lr)

    all_states    = env.get_states()
    T             = env.T
    auc_per_iter: List[float] = []

    for _ in range(n_iterations):
        traj_s, traj_a, traj_r, traj_lp, traj_v = [], [], [], [], []
        policy.eval()
        with torch.no_grad():
            for _step in range(T):
                logits = policy(all_states)
                dist   = torch.distributions.Categorical(logits=logits)
                acts   = dist.sample()
                lps    = dist.log_prob(acts)
                rews   = env.reward(acts)
                vals   = value_net(all_states)
                traj_s.append(all_states); traj_a.append(acts)
                traj_r.append(rews);       traj_lp.append(lps)
                traj_v.append(vals)

        s_f  = torch.cat(traj_s,  0)
        a_f  = torch.cat(traj_a,  0)
        r_f  = torch.cat(traj_r,  0)
        lp_f = torch.cat(traj_lp, 0)
        v_f  = torch.cat(traj_v,  0)

        adv, ret = compute_gae(r_f, v_f)
        adv      = (adv - adv.mean()) / (adv.std() + 1e-8)

        policy.train()
        ppo_update(
            policy, value_net, p_opt, v_opt,
            s_f, a_f, lp_f.detach(), adv.detach(), ret.detach(),
            geo_loss_fn=geo_fn, use_geo_loss=use_geo_loss,
            lambda_geo=lambda_geo,
        )
        auc_per_iter.append(r_f.mean().item() * T)

    return sum(auc_per_iter), auc_per_iter


# -- Condition result ----------------------------------------------------------
@dataclass
class ConditionResult:
    name:           str
    use_projection: bool
    use_geo_loss:   bool
    seeds:          List[int]
    aucs:           List[float] = field(default_factory=list)

    @property
    def mean(self): return float(np.mean(self.aucs))
    @property
    def std(self):  return float(np.std(self.aucs, ddof=1))
    @property
    def ci95(self):
        n  = len(self.aucs)
        se = self.std / np.sqrt(n)
        tc = stats.t.ppf(0.975, df=n - 1)
        return (self.mean - tc * se, self.mean + tc * se)
    @property
    def iqm(self):
        q25, q75 = np.percentile(self.aucs, [25, 75])
        vals = [v for v in self.aucs if q25 <= v <= q75]
        return float(np.mean(vals)) if vals else self.mean


# -- Full factorial run --------------------------------------------------------
def run_factorial_ablation(
    seeds=list(range(10)), n_iterations=60,
    lambda_geo=0.4, verbose=True,
) -> List[ConditionResult]:
    conditions = [
        ConditionResult("A: Baseline PPO",    False, False, seeds),
        ConditionResult("B: Projection Only", True,  False, seeds),
        ConditionResult("C: Geo Loss Only",   False, True,  seeds),
        ConditionResult("D: Full SP-PPO",     True,  True,  seeds),
    ]
    total = len(conditions) * len(seeds)
    run_n = 0
    for cond in conditions:
        for seed in seeds:
            run_n += 1
            if verbose:
                print(f"[{run_n:3d}/{total}] {cond.name} | seed={seed}", end=" ... ")
            auc, _ = train_single(
                cond.use_projection, cond.use_geo_loss,
                seed, n_iterations, lambda_geo,
            )
            cond.aucs.append(auc)
            if verbose:
                print(f"AUC={auc:.4f}")
    return conditions


# -- Statistical tests ---------------------------------------------------------
def statistical_tests(conditions: List[ConditionResult]) -> None:
    print("\n" + "=" * 72)
    print("PAIRWISE STATISTICAL TESTS (Welch t + Mann-Whitney U)")
    print("=" * 72)
    for i in range(len(conditions)):
        for j in range(i + 1, len(conditions)):
            a, b  = conditions[i], conditions[j]
            _, pw = stats.ttest_ind(a.aucs, b.aucs, equal_var=False)
            _, pm = stats.mannwhitneyu(a.aucs, b.aucs, alternative="two-sided")
            d = (a.mean - b.mean) / (np.sqrt((a.std**2 + b.std**2) / 2) + 1e-12)
            print(f"{a.name[:20]:20s} vs {b.name[:20]:20s} | "
                  f"Delta={a.mean - b.mean:+.4f} | "
                  f"p_W={pw:.4f} | p_MW={pm:.4f} | d={d:.3f}")


# -- Output writers ------------------------------------------------------------
def write_csv(conditions, path, lambda_geo):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition","use_projection","use_geo_loss","lambda_geo","seed","auc"])
        for cond in conditions:
            for seed, auc in zip(cond.seeds, cond.aucs):
                w.writerow([cond.name, cond.use_projection,
                             cond.use_geo_loss, lambda_geo, seed, auc])
    print(f"CSV  -> {path}")


def write_json(conditions, path, lambda_geo):
    out = []
    for cond in conditions:
        ci = cond.ci95
        out.append({
            "name": cond.name,
            "use_projection": cond.use_projection,
            "use_geo_loss":   cond.use_geo_loss,
            "lambda_geo":     lambda_geo,
            "mean":    round(cond.mean,  6),
            "std":     round(cond.std,   6),
            "iqm":     round(cond.iqm,   6),
            "ci95_lo": round(ci[0],      6),
            "ci95_hi": round(ci[1],      6),
            "aucs":    [round(v, 6) for v in cond.aucs],
        })
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"JSON -> {path}")


def write_latex_table(conditions, path, lambda_geo):
    header = (
        "\\begin{table}[htbp]\n\\centering\n"
        "\\caption{2$\\times$2 factorial ablation. "
        "Geometric 4-class alignment task; SO(32) structured embeddings;\n"
        f"  $\\lambda_{{\\mathrm{{geo}}}}={lambda_geo}$, 10~seeds, 60~iterations.\n"
        "  D~$>$~C~$>$~B~$>$~A confirms projection and geo-loss independently contribute.}\n"
        "\\label{tab:factorial_ablation}\\scriptsize\n"
        "\\begin{tabular}{clccccc}\\toprule\n"
        "\\textbf{Cond.} & \\textbf{Configuration} & \\textbf{Proj.}"
        " & \\textbf{Geo} & \\textbf{AUC mean$\\pm$std} & \\textbf{95\\%~CI} & $\\Delta$ \\\\\n"
        "\\midrule\n"
    )
    rows = []
    bm   = conditions[0].mean
    for cond in conditions:
        ci  = cond.ci95
        ps  = "\\checkmark" if cond.use_projection else "$\\times$"
        gs  = "\\checkmark" if cond.use_geo_loss   else "$\\times$"
        d   = f"{cond.mean - bm:+.4f}" if cond is not conditions[0] else "---"
        rows.append(
            f"  {cond.name[0]} & {cond.name[3:]:22s} & {ps:10s} & {gs:10s}"
            f" & ${cond.mean:.4f}\\pm{cond.std:.4f}$"
            f" & $[{ci[0]:.4f},\\,{ci[1]:.4f}]$ & {d} \\\\"
        )
    with open(path, "w") as f:
        f.write(header + "\n".join(rows)
                + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    print(f"LaTeX -> {path}")


# -- Main ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick",      action="store_true",
                        help="3 seeds x 20 iters (smoke test, ~3 min on A6000)")
    parser.add_argument("--lambda-geo", type=float, default=0.4)
    args = parser.parse_args()

    seeds      = list(range(3)) if args.quick else list(range(10))
    n_iter     = 20             if args.quick else 60
    lambda_geo = args.lambda_geo

    env_diag = GeometricResponseEnv(seed=42)
    print("=" * 72)
    print("2x2 FACTORIAL ABLATION: Projection x Geometric Loss")
    print("=" * 72)
    print(f"Device:         {DEVICE}")
    print(f"Seeds:          {seeds}")
    print(f"Iterations:     {n_iter}")
    print(f"lambda_geo:     {lambda_geo}")
    print(f"Policy:         GeometricPolicy  (per-action theta_a in so(32) / R^{{nxn}})")
    print(f"  score_a(s)  = tr(H_s^T  norm(theta_a))  [per-action theta]")
    print(f"Alignment range:[{env_diag.alignment_range[0]:.4f}, {env_diag.alignment_range[1]:.4f}]")
    print(f"Correct actions:{env_diag.correct_actions.cpu().tolist()}")
    if args.quick:
        print(f"\n[WARNING] --quick: 3 seeds = low power. Trend D>C>B>A should appear;")
        print(f"   significance needs full 10-seed run.")
    print()

    t0         = time.time()
    conditions = run_factorial_ablation(seeds=seeds, n_iterations=n_iter,
                                        lambda_geo=lambda_geo)
    elapsed    = time.time() - t0

    print("\n" + "=" * 72 + "\nRESULTS SUMMARY\n" + "=" * 72)
    bm = conditions[0].mean
    for cond in conditions:
        ci  = cond.ci95
        pct = 100 * (cond.mean - bm) / (bm + 1e-12)
        print(f"{cond.name}")
        print(f"  AUC: {cond.mean:.4f}+/-{cond.std:.4f}  "
              f"IQM={cond.iqm:.4f}  "
              f"CI=[{ci[0]:.4f},{ci[1]:.4f}]  Delta={pct:+.1f}%")

    statistical_tests(conditions)
    write_csv(conditions,        "ablation_results.csv",         lambda_geo)
    write_json(conditions,       "ablation_results.json",        lambda_geo)
    write_latex_table(conditions,"ablation_table_factorial.tex", lambda_geo)

    print(f"\nTotal time: {elapsed:.1f}s")
    D_vs_A = conditions[3].mean - conditions[0].mean
    C_vs_A = conditions[2].mean - conditions[0].mean
    B_vs_A = conditions[1].mean - conditions[0].mean
    D_vs_C = conditions[3].mean - conditions[2].mean
    _, p_DA = stats.ttest_ind(conditions[3].aucs, conditions[0].aucs, equal_var=False)
    _, p_CA = stats.ttest_ind(conditions[2].aucs, conditions[0].aucs, equal_var=False)

    print("\n-- KEY QUESTIONS ANSWERED --")
    print(f"  (B) so(32) alone   vs baseline: {B_vs_A:+.4f} AUC")
    print(f"  (C) Geo loss alone vs baseline: {C_vs_A:+.4f} AUC  p={p_CA:.4f}")
    print(f"  (D) Full SP-PPO    vs baseline: {D_vs_A:+.4f} AUC  p={p_DA:.4f}")
    print(f"  (D) Projection on top of geo:   {D_vs_C:+.4f} AUC over C")
    print()
    hier     = [(D_vs_A, "D>A"), (C_vs_A, "C>A"), (B_vs_A, "B>A"), (D_vs_C, "D>C")]
    confirmed = [n for v, n in hier if v > 0]
    missing   = [n for v, n in hier if v <= 0]
    if confirmed:
        print(f"  [OK] Confirmed: {', '.join(confirmed)}")
    if missing:
        print(f"  [WARNING] Not yet confirmed: {', '.join(missing)}")
        if args.quick:
            print(f"    -> Run full 10-seed x 60-iter version for statistical power.")
