# `sppg_defense` -- verification and analysis code for the SP-PG revision

Numerical verification of the paper's analytical results, plus the statistical
and structure-selection tools used in the revision.

The statistics module is checked against the values reported in the
manuscript; the Procrustes solver is checked against brute-force optimisation;
the theory modules verify each theorem to machine precision.

---

## Install and run

```bash
python -m venv .venv && source .venv/bin/activate
pip install numpy scipy pandas matplotlib statsmodels mpmath

python -m sppg_defense.selftest        # ~15 s   validate the package itself
python -m sppg_defense.run_theory      # ~100 s  verify all seven theorems
python -m sppg_defense.mutation_test   # ~5 s    confirm the tests can fail
python -m sppg_defense.figures.make_figures   # four publication PDFs
```

`run_theory` exits non-zero if any theorem check fails, so it can be a
Makefile/CI gate. `--quick` runs a ~5 s smoke test. No GPU, no PyTorch, no
language model: pure NumPy/SciPy, so it runs on a laptop while the RL sweeps
use the GPU.

---

## Verified output

All seven theory modules pass at production settings (~100 s total, CPU), and
`mutation_test` confirms that all 21 false claims are rejected:

| Module | Verifies | Headline number |
|---|---|---|
| `t01_softmax_smoothness` | TH-1, Lemma `lem:softmax_lipschitz`(i) | bound attained **exactly** (err 8.9e-16) at every radius from 1e-2 to 1e4, on **all four** subspaces |
| `t05_frechet_bounds` | TH-5, Lemma `lem:exp_derivative_bounds` | 0 violations on so(n) for n = 2 to 256, radii 0.1 to 1000; **off** so(n) the same ratio reaches **1.5e4** |
| `t06_geo_loss_smoothness` | TH-6, Lemma `lem:geo_smooth` | max lambda_max = 0.463 <= 1; Lanczos and full-Hessian agree to 4.8e-6; ambient panel verified non-trivial |
| `t07_projected_gd` | TH-7, Theorem `thm:nonconvex` | 0 descent-inequality violations, 0 rate violations; eta = 2/L threshold matched to **4.4e-15** |
| `t08_robbins_monro` | TH-8, Theorem `thm:stochastic` | closed-form floor 0.0601245 vs observed 0.0601071 (**2.9e-4**); at alpha = 0.30 the naive formula is off by **46%** vs a measured **0.07%**, a 580x discrimination |
| `t09_pullback_bound` | TH-9, Prop. `prop:dichotomy_app`(i) | 0 violations over n = 4 to 32, radii 0.1 to 1000; adversarial search reaches 0.32x the bound (vs 0.05x with random directions); 2x2 witness = 0.5 exactly |
| `t10_exponential_witness` | TH-10 + TH-15, Prop. `prop:dichotomy_app`(ii) | bound holds at every (mu, R) up to **mu*R = 759** with `\|sin(y_k)-1\| < 6.4e-81` verified; growth rate 2*mu to 8e-4; nilpotent rho = 1 with operator norm / t tending to 1 |

---

## What each piece is for

### `theory/` -- numerical verification of the theorems

Each module states the claim it verifies, the method, and the pass criterion in
its docstring.

Three design decisions matter here, because the obvious pass criterion would be
mathematically wrong in each case (the design notes at the end of this file list
nine ways a test can pass vacuously):

- **T-1** does *not* test that `lambda_max` is constant in `||theta||`.
  Exponential tilting can move mass onto two extreme feature vectors and
  *raise* the variance above its value at theta = 0. What the lemma asserts,
  and what is tested, is that the **supremum over the sphere of radius R equals
  B^2 for every R**, witnessed exactly by an m = 2 construction with theta
  orthogonal to the feature direction.
- **T-7** does not require the empirical decay exponent to be <= -1 at every
  horizon; a small step size spends its early iterations in a transient where
  the observed decay is slower than the guarantee without ever violating it. The
  bound itself (0 violations) is the test; the exponent is a diagnostic.
- **T-10** evaluates `L_F` at the exact jump points `r_k = log(pi/2 + 2*pi*k)/mu`
  using `mpmath` at precision **scaled to mu*R**, because `sin(e^(mu*r))` is
  meaningless in float64 past ~1e15 and a fixed precision makes the index
  selection vacuous. `|sin(y_k) - 1| < 1e-20` is verified on every row.

### `structure/procrustes.py` -- T-13 and S-1, structure selection (Table 2)

- `procrustes_so(V, T)` -- the **closed-form global optimum** of the SO fit.
  Verified against best-of-30 Powell restarts: gap 5.3e-16.
- `compare_fits(...)` -- every candidate *plus* the identity, scaled-identity
  and exact-Procrustes baselines, and the S-1 diagnostics: `||X_hat||_F`,
  `||exp(X_hat) - I||_F`, and the **rotation angles in degrees**. If those
  angles are a fraction of a degree, there is no rotational structure to
  select. It also reports the **suboptimality of the 300-step Adam fit** versus
  the exact optimum.
- `exhaustive_permutation_test(...)` -- all 5! = 120 pairings, resolving
  `p = 1/120`, in place of 128 Monte-Carlo draws *with replacement* from a pool
  of 120 distinct permutations.

### `stats/tests.py` -- validated against the manuscript's reported values

| Manuscript | This package |
|---|---|
| Welch t = 30.663 | 30.915 |
| U = 100, p = 0.0002 | U = 100, p = 0.0002 |
| Cohen's d ~ 13.7 | 13.83 |
| IQM 6.486 / 8.652 | 6.485 / 8.6525 |
| TOST p < 1.2e-8 | p = 1.081e-8 |
| 90% CI [-0.028, +0.023] | [-0.0272, +0.0232] |

(Small deviations come from the rounded per-seed values in Table `tab:perseed`.)

It also provides `paired_report` (the design is paired, whereas Welch and
Mann-Whitney assume independence), BCa bootstrap CIs, `holm`, `power_paired`,
`n_for_power_paired`, `probability_of_improvement`, and `iqm_ci`. Every function
raises on degenerate input rather than returning a plausible-looking number;
see the design notes.

**Power calculation:**

```
dz = 0.69 (Task-2 effect), n = 10  ->  power 0.495     (underpowered)
                           n = 19  ->  power 0.80
dz ~ 2.49 (seed-0 so-vs-unconstrained gap), n = 4  ->  power 0.80
```

The ablation effect is large enough that **four seeds** give 80% power. R-4
runs it at 20 seeds, which takes about 11 minutes.

### `rl/arms.py` -- parameterisations

`arm.project(theta)` replaces `Proj_so(32)` in Algorithm `alg:spppo`, so one
training loop runs every condition. Includes **`RandomSubspaceArm`**, the
dimension-matched, structure-free control (R-6) that separates geometry from
dimensional regularisation (Section 7.1), plus `ConjugatedAlgebraArm`
(non-orthogonal `P so(n) P^-1`; an *orthogonally* conjugated arm would be a
provable null control, see `orthogonal_conjugation_is_a_noop`), `LoRAArm`, and
`LowRankSoArm`.

`structure_diagnostics(theta)` is the T-11 logger, logged every iteration for
every arm. It reports orthogonality error, skewness residual, det, rho **and**
operator norm, because rho = 1 alone is not sufficient (see the nilpotent panel
of `fig_t10`: rho = 1 while `||e^(tN)||_op` grows linearly over six decades).

Note on `LoRAArm`: it exposes **two** parameter counts. `n_parameters = 2nr`
is the manuscript's convention (r = 8 gives 512); `dim = r(2n-r)` is the
manifold dimension after removing gauge redundancy (r = 8 gives 448). so(32)
has 496, so a LoRA comparison against so(32) depends on which convention is
matched.

### `rl/analysis.py` -- R-1 and R-4 from training logs

- **`channel_decomposition`** -- given per-iteration `r_task` and `r_geo`, it
  splits the AUC gap into the channel both arms can optimise and the channel
  the `M=I` baseline structurally cannot reach. It returns one of three
  pre-written verdicts, so the interpretation rule is fixed before the number is
  seen. It needs no new training run, only two extra logged columns.
- **`ablation_report`** -- 20-seed version of Table `tab:ablation`: per-arm
  IQM with bootstrap CIs, a Friedman omnibus test (seeds as blocks),
  Holm-corrected pairwise tests, and the four pre-registered orthogonal
  contrasts (constrained vs unconstrained, compact vs non-compact Lie, Lie vs
  non-Lie, so vs random-496).
- `auc_trapezoid` implements Eq. (7.1) exactly as the manuscript defines it
  (unit spacing, half-weight endpoints, divided by T), so new numbers are
  commensurable with the published ones.

### `figures/make_figures.py`

Four vector PDFs, `pdf.fonttype=42`, greyscale-safe, sized for elsarticle:
radius-independence, the Frechet contrast (1.0 vs 1.5e4), the O(1/T) rate with
the eta = 2/L threshold, and the exponential barrier beside the nilpotent panel.

---

## Suggested order of use

1. `python -m sppg_defense.selftest` -- confirm the package runs correctly on
   this machine.
2. `python -m sppg_defense.run_theory && python -m sppg_defense.figures.make_figures`
   -- the theory-verification results and the four figures.
3. `structure/procrustes.compare_fits(...)` on the real embedding pairs -- the
   S-1 diagnostic.
4. Add `structure_diagnostics` to the training logger and re-run one Task-1 seed.
5. Log `r_task`/`r_geo` separately, re-run 20 seeds, and call
   `channel_decomposition`.
6. Swap in `rl/arms` and run R-4/R-6 at 20 seeds; analyse with `ablation_report`.

---

## Scope

There is **no environment implementation here.** The Task-1 MDP (tier
reward/drift schedule, `M_env` seed 42, the feature network) lives in
`src/main.py`, and a separate reimplementation would produce numbers that do not
reconcile with the published tables. The arms and the analysis are written
against a documented interface so they attach to the existing training loop.

Every module is self-contained, documented with the claim it addresses, and
raises on malformed input rather than returning something plausible.

---

## Design notes: tests that cannot pass vacuously

A check passes vacuously when it could not fail even if the theorem were false.
The table lists nine such pitfalls and how each test here avoids them;
`mutation_test` demonstrates that every check can fail.

| # | Pitfall | Why it matters | How the test avoids it |
|---|---|---|---|
| 1 | **A T-6 ambient panel that is a no-op.** If `theta` and the perturbation directions are both skew, `Proj_so` is the identity and the "ambient" rows duplicate the algebra rows exactly. | The ambient form is the one Algorithm `alg:spppo` differentiates through, so the claim the implementation relies on would go untested. | `theta` and the directions are drawn from `gl(n)`, and the panel asserts that the projector is non-trivial. |
| 2 | **Random-direction maximisation under-reports lambda_max by 10-100x** (T-6, T-9). At dim 496 a random quadratic form concentrates near `tr(H)/d`, not lambda_max. At n = 16: true 0.274, random estimate 0.044. | A violation of the smoothness bound would be invisible. | **Lanczos** on Hessian-vector products from the exact analytic gradient (machine precision), cross-checked against the full assembled Hessian. |
| 3 | **Fixed 60-digit precision in T-10** while `mu*R` reaches 759. The `floor()` selecting `k` becomes meaningless, `sin(y_k) != 1`, and the ratio collapses to exactly 4.0. | "Bound verified at `mu*R = 759`" would not be a verification. | Precision scaled to `mu*R/log(10) + 40` digits; `\|sin(y_k) - 1\| < 1e-20` verified on every row (observed 6.4e-81). |
| 4 | **A T-10 growth-rate check that is an algebraic identity.** Regressing `log(mu^2 e^(2 mu R_k))` on `R_k` returns `2mu` to 1e-16 whether or not `L_F_lower` is correct. | It would test nothing. | The regression uses `L_F_lower`'s own output; the resulting error is ~8e-4 (staircase quantisation). |
| 5 | **A circular constant check in T-10.** Re-evaluating the analytic bound instead of differentiating `f`, and sampling only random `E`, gives `L_f = 0.119` against a true supremum of 1. | The witness's hypotheses `G_f, L_f <= 1` would be assumed, not checked. | Finite differences of `f` itself, with the extremal direction `E = uu^T` included; the check reports 1.000000. |
| 6 | **A T-8 tail ratio that cannot fail.** `sum_{t>=T/2} a_t g_t^2 / sum_{t<T/2} a_t g_t^2` equals `2^0.25 - 1 = 0.19` for a *constant* gradient, i.e. when the series **diverges**. | Conclusion (ii) would go untested. | A step-mass-normalised ratio; the necessity argument is made entirely in Part A, where the floor is closed-form. |
| 7 | **A T-8 floor tolerance (15%) 90x looser than the measurement.** The naive leading-order formula, which drops exactly the term that makes the closed form non-trivial, differs by only 6.75% at alpha = 0.05 and would pass. | The "exactly predicted floor" would not be checked. | Tolerance 2%, plus a second step size alpha = 0.30 where the naive formula is off by **46%** against a measured error of **0.07%** (580x discrimination). |
| 8 | **A T-9 bound that is never approached** (max ratio 0.051); a bound 10x tighter than the claim would also report zero violations. | No discriminating power. | Adversarial Lanczos + hill-climbing over `theta`; the summary states the *tightest bound the search would still not violate*. |
| 9 | **Orthogonal conjugation as a control.** For orthogonal Q, `Q Proj_so(Q^T M Q) Q^T = Proj_so(M)` identically (verified: 2.5e-14). | Such an arm cannot isolate orientation from structure, and it would add a guaranteed-null comparison to the Holm family, deflating power for the real ones. | `ConjugatedAlgebraArm` uses a non-orthogonal `P so(n) P^-1`, a different subspace; the invariance is stated as a verified one-line fact. |

Related safeguards in the statistics and arm code:

- The `sl(n)` basis is built by **SVD**; an un-pivoted QR is correct only by accident of column ordering.
- `prob_superiority_paired` conditions on discordant pairs, so its point estimate cannot fall **outside its own CI** under ties.
- `probability_of_improvement` uses the same tie convention in the bootstrap as in the point estimate.
- Holm rejects at `<= alpha` (not `< alpha`) and raises on a NaN p-value rather than treating it as "not significant".
- `tost_paired` raises on zero variance instead of returning **"equivalent, p = 0"** via `max(0.0, nan)`.
- `bca_ci` falls back to a percentile interval on constant data, and `channel_decomposition` gives an explicit "CI unavailable" verdict instead of turning `(nan, nan)` into a scientific verdict.
- `paired_report` reports an exact null (the R-1 outcome it exists to detect) instead of raising.
- `channel_decomposition` raises when the two arms do not share a seed set; misaligned seeds would move `t_p` from 3.9e-10 to 1.4e-4.
- `ablation_report` uses **Friedman** on the paired design (Kruskal-Wallis gives p = 0.93 where Friedman gives 4e-4), `pivot` rather than `pivot_table` (which averages duplicate rows), and exact arm matching with an overlap assertion (`startswith` could put an arm on both sides of a contrast).
- `exhaustive_permutation_test`'s sampled branch uses the Phipson-Smyth estimator `(1+n)/(1+B)`, which never returns **p = 0**.
- `logm` is validated for reconstruction as well as skewness, so a pi-rotation cannot report `||X||_F = 0`.
- `LowRankSoArm` uses the real Schur (Youla) form, which gives rank exactly 2r where SVD truncation breaks on tied singular values, and its `dim` stays valid for `2r > n`.

### Mutation test

```
python -m sppg_defense.mutation_test      # 21 false claims, 21 caught, 0 missed
```

Each mutation states a **false** version of a theorem and requires the suite to
reject it, including a fixed 60-dps T-10 computation (caught:
`|sin(y_k) - 1| = 1.9`) and a side-by-side showing random directions miss a
violation that Lanczos catches (16x under-report at dim 496).
