# Structure-Preserving Policy Gradient (SP-PG)

Code for the paper:

**"Structure-Preserving Policy Optimisation via Compact Lie Algebra Constraints"**
Sooraj K.C and Vivek Mishra, AIMS Mathematics, 2026.


## What this repo contains

SP-PG constrains reinforcement learning policy parameters to the compact
Lie algebra so(n) via a Frobenius-orthogonal projection after each gradient
step. The matrix exponential exp(theta) stays in SO(n) and preserves norms,
allowing the policy to align with latent rotational structure in frozen
language model representations.

This repo reproduces all experiments in the paper:

| Script | Paper section | What it does |
|---|---|---|
| main.py | Sec 4, 8 | Structure selection, Task 1 (PPO, GPT-2 Medium), four-way ablation, geo-weight sweep, LoRA comparison, overhead |
| task1_figures.py | Sec 8 | Spectral radius comparison figure |
| 04_pap_CG.py | Sec 9 | Task 2 single-seed engine (REINFORCE, GPT-2 text generation) |
| task2_multiseed.py | Sec 9 | Task 2 10-seed wrapper (Welch t, Mann-Whitney, bootstrap CI) |
| run_mistral_task1.py | Sec 10 | Cross-model validation on Mistral-7B (4-bit, 4096->1024 projection) |
| run_sentiment_task3.py | Sec 11 | Falsification test: sentiment steering with no geometric structure |
| run_all.py | -- | Orchestrator: runs all steps in order |


## Requirements

- Python 3.9 or later
- PyTorch 2.0 or later
- NVIDIA GPU with at least 8 GB VRAM (A6000 used for Tasks 1-2, RTX 5060 for Mistral)
- CUDA 11.8 or later

Install dependencies:

```
pip install -r requirements.txt
```


## Hardware used in the paper

| Experiment | GPU | Time |
|---|---|---|
| Task 1 (GPT-2 Medium, 10 seeds + ablations) | NVIDIA A6000 48 GB | ~15 min |
| Task 1 figures (spectral comparison) | NVIDIA A6000 48 GB | ~10 min |
| Task 2 single seed | NVIDIA A6000 48 GB | ~3 min |
| Task 2 10-seed | NVIDIA A6000 48 GB | ~30 min |
| Mistral-7B (10 seeds) | NVIDIA RTX 5060 8 GB | ~20 min |
| Task 3 sentiment (10 seeds) | NVIDIA A6000 48 GB | ~40 min |


## How to reproduce all results

### Option A: Run everything at once

```
python run_all.py
```

This runs all 6 steps in order and saves outputs to the outputs/ directory.
Total time: approximately 2 hours on an A6000.

To run a quick smoke test:

```
python run_all.py --quick
```

To skip or select specific steps:

```
python run_all.py --skip 5,6
python run_all.py --only 1,2
```


### Option B: Run experiments individually

Each script can be run standalone.

**Step 1: Task 1 + structure selection + ablation (Paper Sec 4, 8)**

```
python main.py
```

Produces:
- Table 1: Structure selection residuals
- Table 2: Hyperparameters
- Table 3: Task 1 main result (10 seeds)
- Table 4: Per-seed AUC values
- Table 5: Four-way ablation and geo-weight sweep
- Figures 1-4: Returns, multiseed, ablation, diagnostics
- LoRA comparison
- Overhead measurements


**Step 2: Spectral comparison figure (Paper Sec 8)**

```
python task1_figures.py
```

Produces:
- spectral_comparison.png (Figure 5 in paper)


**Step 3: Task 2 single seed (Paper Sec 9)**

```
python 04_pap_CG.py
```

Produces:
- Task 2 single-seed results and learning curves
- Structure discovery on text generation embeddings


**Step 4: Task 2 ten seeds (Paper Sec 9)**

Requires 04_pap_CG.py in the same directory.

```
python task2_multiseed.py
```

Produces:
- Table 7: Task 2 validation summary (Welch t, Mann-Whitney, bootstrap CI)
- 10-seed reward statistics


**Step 5: Mistral-7B cross-model validation (Paper Sec 10)**

Requires ~8 GB VRAM for 4-bit quantized Mistral-7B.

```
python run_mistral_task1.py
```

Produces:
- Table 8: Mistral-7B 10-seed results
- Algebra hierarchy comparison


**Step 6: Task 3 falsification (Paper Sec 11)**

```
python run_sentiment_task3.py
```

Or run a subset of seeds:

```
python run_sentiment_task3.py --seeds 0 1 2
```

Produces:
- Table 9: Task 3 results (SP-PG vs baseline, p=0.619)
- Welch t, Mann-Whitney U, Cohen's d, bootstrap CI


## Key results reproduced

| Experiment | Metric | Value |
|---|---|---|
| Task 1 (GPT-2 Medium) | AUC improvement over PPO | +30.2% (p < 0.0001) |
| Mistral-7B | AUC improvement over PPO | +59.9% (p < 0.0001) |
| Task 2 (text generation) | Reward difference | Delta = 0.036 (MW p = 0.0001) |
| Task 3 (falsification) | SP-PG vs baseline | No difference (p = 0.619) |
| Ablation hierarchy | so(32) > sl(32) > unconstrained > sym(32) | Consistent across lambda >= 0.2 |
| Overhead | Per-iteration cost | 9.6% at n=32 |


## Pretrained models used

- GPT-2 Medium: https://huggingface.co/gpt2-medium (Task 1)
- GPT-2: https://huggingface.co/gpt2 (Task 2, Task 3)
- Mistral-7B: https://huggingface.co/mistralai/Mistral-7B-v0.1 (Sec 10)

All models are loaded with frozen weights. No fine-tuning of transformer
parameters is performed. Only the policy head is trained.


## Seed control

All experiments use fixed random seeds for reproducibility.
Task 1, Task 2, Mistral, and Task 3 each use seeds 0-9 (10 seeds).
Ablation uses seed 0 only (single-seed, acknowledged as a limitation).

Seeds are set for Python, NumPy, PyTorch, and CUDA:

```python
import torch, numpy, random
random.seed(seed)
numpy.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
```


## No proprietary data

All data is generated synthetically as described in the paper.
No proprietary datasets are used.


## License

MIT License. See LICENSE file.


## Citation

```
@article{kc2026sppg,
  title={Structure-Preserving Policy Optimisation via Compact Lie Algebra Constraints},
  author={KC, Sooraj and Mishra, Vivek},
  journal={Neural Networks},
  year={2026},
  note={Under review}
}
```
