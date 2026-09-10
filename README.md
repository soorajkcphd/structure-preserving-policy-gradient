# Structure-Preserving Policy Gradient on Special Orthogonal Lie Algebras

Code, data and verification for the IJACM manuscript.

The paper constrains a policy's linear-head matrix parameter to a Lie algebra
and separates two things that are easy to conflate: a **smoothness** result
that follows from bounded features and holds on *any* linear matrix subspace,
and a **geometric** result that is specific to skew-symmetry. Empirically it
reports two matched studies whose answers differ, and the difference is the
result.

---

## One command

```bash
./check_all.sh
```

That runs all eleven stages below, each from the correct directory, in
about two minutes. It keeps going if a stage fails, writes a transcript to
`check_all.log`, and prints a PASS/FAIL summary. Run it from anywhere -- it
resolves its own location, so the "wrong directory" mistakes in
[When something raises](#when-something-raises) cannot happen.

**It installs nothing.** There is no `pip` call and no `conda` call in the
script. Stage 0 imports the packages it needs and reports what it found; if
something is missing it names it, prints the command *you* would run, and stops.
Nothing outside this repository is written.

**conda:** activate the env and run it. Stage 0 prints the interpreter path and
the env name first, so you can confirm it picked up the right one before
anything else happens. `--python /path/to/python` pins it explicitly, and
`$PYTHON` is honoured if set.

Prerequisites: `numpy scipy pandas mpmath`, plus `torch` (the CPU build is
enough) for stages 7 and 11. No GPU, no model downloads, no network.

### Running the experiments in the same one shot

```bash
nohup ./check_all.sh --experiments > run.out 2>&1 &
```

Verification first, then all twelve sweeps back to back. They run cheapest-first
(`b1` first, `r7` last), so a broken environment shows up in the first sweep
instead of after the first long one. `r15b` reads `r15`'s output, so it always
follows it. A failing sweep does not stop the others; the summary lists it at
the end. When everything finishes, the fresh CSVs in `src/` are re-analysed
automatically.

**Measured end to end: 15 h 55 m** on an RTX 5060 Laptop GPU (2026-08-14
12:09 UTC -> 2026-08-15 04:04 UTC), against a tabulated 10.5 h -- about 1.5x.
Treat the per-sweep numbers below as relative costs rather than wall-clock
promises, and detach the run rather than watching it.

Before starting, stage 0 additionally requires `torch` and `transformers` and a
visible CUDA device, and does not start without them rather than falling
back to a CPU run that would take days.

| Option | Effect |
|---|---|
| `--experiments` | Verification, then all 12 sweeps |
| `--experiments-only` | Skip verification, go straight to the sweeps |
| `--only r4,s7` | Just these, in the order given |
| `--overwrite` | Allow a sweep to replace an existing `<name>_cells.csv` |

Each sweep writes `src/<name>_cells.csv` and a `.meta.json` recording the grid,
the seed count and an MD5 of `main.py`. Without `--overwrite` a sweep will not
truncate a CSV that already exists -- that guard belongs to the harness and is
left in place.

| Stage | What it establishes |
|---|---|
| 0 | Prerequisites -- reports every package version, names anything missing |
| 1 | This checkout is current (the IQM convention fix is present) |
| 2 | 27 numerical checks of the analytical results, sharing no code with the project |
| 3 | `sppg_defense` selftest -- theory against the project's own implementation |
| 4 | Theory modules t01-t10 |
| 5 | 21-mutation suite -- confirms the checks are able to fail |
| 6-8 | Harness selftests: `sppg_core`, `sppg_experiments`, `sppg_analysis` |
| 9 | Every headline number in the paper reconciles against `data/` |
| 10 | Full analysis of all 13 shipped CSVs |
| 11 | `data/r7_cells.csv` was drawn with the seeds the code uses now |

Expected last lines:

```
  ALL 11 STAGES PASSED
```

Stage 11 guards the r7 environment seeds -- see
[The r7 M_env seeds](#the-r7-m_env-seeds-resolved).

| Option | Effect |
|---|---|
| `--quick` | Skip stages 5 and 10 (the two slowest) |
| `--no-log` | Terminal only, do not write `check_all.log` |
| `--python PATH` | Pin the interpreter (default: `$PYTHON`, else `python3`) |

Exit status is 0 only if every stage passed, so it drops into CI unchanged.

`verify.sh` is a strict subset -- five stages, stopping at the first failure.
Use it as a pre-push hook; use `check_all.sh` to see everything at once.

---

## Layout

```
src/           experiment code -- the released originals plus a consolidated harness
verification/  two independent layers of checks on the analytical results
data/          the result CSVs the manuscript tables are generated from
tools/         latexdiff repair, r13b preflight, r7 provenance check
check_all.sh   one shot: verification, and optionally all 12 sweeps
verify.sh      a subset of the same checks as a strict CI gate
```

`check_all.sh` is at the repository root. Do not confuse it with
`src/run_all.py`, which is one of the *original* drivers and launches the full
GPU experiment suite -- both take `--quick`, and they mean very different things.

`src/` is kept flat. The harness loads the original scripts by relative
path (`SourceFileLoader("pap_cg", "04_pap_CG.py")`), so they must share a
working directory. Splitting them into subpackages breaks the imports.

---

## Requirements

Two tiers, because most of this repository runs on a laptop.

| Tier | Packages | Enough for |
|---|---|---|
| **1** | `numpy scipy pandas mpmath` | `verify.sh`, `verify_math.py`, `sppg_analysis.py`, all of `sppg_defense` |
| **2** | `+ torch transformers matplotlib` | running eleven of the twelve sweeps (GPU, hours) |
| **2b** | `+ bitsandbytes accelerate` | `r13b` only -- Mistral-7B in 4-bit |

`r13b` is the only sweep with requirements past tier 2: `bitsandbytes` for
`load_in_4bit=True`, `accelerate` for `device_map="auto"`, and ~15 GB of
`mistralai/Mistral-7B-v0.1` downloaded on first use (the repo is public -- no
token needed). Since it runs last, check it before starting a long queue:

```bash
python3 tools/preflight_r13b.py      # seconds; safe to run while a sweep is going
```

---

## Running things

**Directories matter.** Each block below states where to stand.

### Verify the mathematics -- from the repo root

```bash
python3 verification/verify_math.py
```

27 checks, a few seconds, tier 1 only.

### Read the shipped results -- from `src/`

```bash
cd src
python3 sppg_analysis.py --dir ../data              # everything, in reading order
python3 sppg_analysis.py r4 s7 b1lr --dir ../data   # only these
python3 sppg_analysis.py --manuscript --dir ../data # the reconciliation table
```

### Check the theory against the project's own implementation -- from `verification/`

```bash
cd verification
python3 -m sppg_defense.selftest
python3 -m sppg_defense.run_theory
python3 -m sppg_defense.mutation_test
```

These are `python -m` module invocations, so they must be run from
`verification/`, not from the repository root.

### Regenerate the manuscript figures -- from `src/`, needs a GPU

The sweeps write CSVs only; they never draw anything. The figure code lives in
the original `main.py` and `task1_figures.py`, and running those end to end also
runs work the manuscript no longer uses. `make_figures.py` calls only the five
paths that produce figures the paper includes:

```bash
cd src
python3 make_figures.py                      # all five -> plots/
python3 make_figures.py --only spectral      # just one
python3 make_figures.py --outdir ../figs
```

| File | Figure |
|---|---|
| `rl_returns.png` | Single-seed comparison, seed 0 |
| `rl_ablation.png` | Algebra ablation, seed 0 |
| `multiseed_auc.png` | 10-seed per-seed AUC |
| `geo_weight_ablation.png` | Geometry-weight sweep, seed 0 |
| `spectral_comparison.png` | Spectral radius by arm, seed 0 |

It edits nothing -- it imports the originals and calls their public functions
with the arguments `main()` uses. Budget roughly 1.5 h for all five.

### Run experiments -- from `src/`, needs a GPU

```bash
cd src
python3 sppg_core.py                    # ~5 s   -> SELFTEST PASS
python3 sppg_experiments.py --selftest  # ~20 s  -> SELFTEST PASS
python3 sppg_analysis.py --selftest     # ~5 s   -> SELFTEST PASS
python3 sppg_experiments.py --list
python3 sppg_experiments.py r4          # ~30 min
```

The three selftests need no GPU, no model and no CSV (`sppg_experiments.py`
imports `torch`, but the CPU build is enough); run them before spending GPU time. If any fails, stop -- something in the environment is wrong and every
number downstream would inherit it.

---

## `src/` -- experiment code

### The originals (unmodified)

The scripts that produced the published runs. **Nothing here edits them**; the
harness imports them.

| File | Role |
|---|---|
| `main.py` | Task 1: environment, policy, PPO loop, projections |
| `04_pap_CG.py` | Task 2: text generation |
| `run_sentiment_task3.py` | Task 3: sentiment steering (falsification control) |
| `run_mistral_task1.py` | Cross-model replication on Mistral-7B |
| `task1_figures.py`, `task1_extra.py`, `task2_multiseed.py` | Figures and multi-seed runs |
| `ablation_factorial.py`, `baseline_lora.py` | Synthetic factorial; LoRA arm (withdrawn) |
| `_armlib.py`, `_r1core.py` | Shared helpers |
| `run_r*.py`, `run_b1*.py`, `run_s7.py`, `run_x3.py`, `run_all.py` | Original per-sweep drivers |

### The consolidated harness

Three files replacing the fifteen `run_*.py` drivers.

| File | Lines | Purpose | GPU |
|---|---:|---|---|
| `sppg_core.py` | 924 | Statistics, reward-channel accounting, arm construction | no |
| `sppg_experiments.py` | 1590 | Runs the twelve experiments, writes `<name>_cells.csv` | yes |
| `sppg_analysis.py` | 1490 | Reads the CSVs and states what may be claimed | no |

Collection needs a GPU and hours; analysis needs neither, and you will run it
many times.

Each experiment writes `<name>_cells.csv` and a `.meta.json` recording the
grid, the geometry weight, the seed count and **an MD5 of `main.py`**, so a
later analysis can refuse a stale file.

| Name | Time | Needs | Question it answers |
|---|---|---|---|
| `r4` | ~30 m | `main.py` | **The ablation.** Source of the algebra table |
| `r5` | ~35 m | `main.py` | Is the ordering a shared-learning-rate artefact? |
| `x3` | ~20 m | `main.py` | Does it survive redrawing the planted rotation? |
| `r7` | ~2.5 h | `main.py` | **Scope.** Compactness, or matching the environment? |
| `b1` | ~12 m | `main.py` | The constraint, or the matrix exponential? |
| `b1lr` | ~20 m | `main.py` | Was `b1`'s gap a shared-rate artefact? |
| `s7` | ~15 m | `main.py` | **Prior, or supervision?** |
| `r1b` | ~45 m | `main.py` | Is the task-channel cost tunable away? |
| `r11` | ~2.5 h | `04_pap_CG.py` | Is Task 2's effect in-sample only? |
| `r15` | ~35 m | `run_sentiment_task3.py` | What could Task 3 have detected? |
| `r15b` | ~35 m | `run_sentiment_task3.py` + `r15` | Does the s7 dichotomy replicate? |
| `r13b` | ~50 m | `run_mistral_task1.py` | Does Mistral's gain mean what GPT-2's means? |

There is no `r1` entry: R-1's decomposition is produced *by* `r4`, and an
alias would let `r1 --overwrite` truncate an existing R-4 sweep.

Timings are the ones the original runs were recorded with, on an 8 GB RTX 5060.
They are relative costs, not wall-clock promises: a full measured run on an
RTX 5060 Laptop came in at 15 h 55 m against a tabulated 10.5 h.

`r13b` is the only sweep needing anything beyond `torch` + `transformers` -- it
also wants `bitsandbytes` (4-bit) and `accelerate` (`device_map="auto"`), and it
downloads ~15 GB of Mistral-7B weights. Because it runs last, check it before
starting a long queue -- this is safe to run while a sweep is in progress, as it
never touches CUDA and downloads nothing:

```bash
python3 tools/preflight_r13b.py
```

### Three rules the analysis enforces

1. **Pairing.** Every contrast is matched on its blocking variable. Incomplete
   blocks are dropped and reported, never averaged in.
2. **Multiplicity.** Every family is Holm-corrected together and the header
   says how many comparisons are in it. Untestable comparisons are excluded
   rather than entered at p = 1, which would inflate the multiplier.
3. **Nulls.** "Not significant" is never printed as "equivalent". Equivalence
   goes through TOST against a margin fixed in advance, set at the top of
   `sppg_analysis.py` before any result is seen.

`--manuscript` compares every headline number the paper quotes against what the
CSVs contain, at half the last digit the paper states. A `FAIL` means *either*
the CSV is stale *or* the manuscript is wrong -- it does not guess which. Check
the `.meta.json` beside the CSV first.

---

## `verification/` -- checking the theory

Two independent layers.

### `verify_math.py` -- shares no code with the project

Rebuilds every object from numpy and scipy alone and compares against the
constant the paper prints: projector identities, `exp(so) subset of SO(n)` at radii to
50, `det exp X = e^{tr X}`, Frechet derivatives by finite difference,
Popoviciu's variance bound at `||theta||_F` up to 1e6, Procrustes by SVD, exact
permutation counts, and the reported margins.

`sppg_defense/` checks the theory against the *project's own* implementation of
these objects. That is useful, but it shares helper code with the thing being
checked, so a defect in a shared helper could hide a defect in a theorem. This
file cannot.

It prints the observed value beside each bound, so slack is visible rather than
only whether the bound held. Two checks caught real defects during revision:

- **V7c** -- the geometric-loss constant 1 is valid but *not attained*; the
  sharp value is 1/2, at n = 2, theta = 0, E = J/sqrt(2), G = I.
- **V5** -- `exp(sl)` has determinant exactly 1 and `exp(gl(n,R))` never has
  negative determinant, which retired a claim that the larger algebras
  "represent any M_env exactly".

### `sppg_defense/` -- checks against the project's implementation

One module per analytical result (`theory/t01`-`t10`), plus the Procrustes
solver, the statistical tests, and a **21-mutation suite** that
breaks each claim to confirm the corresponding test notices.

`sppg_defense/results/` holds the committed output of a passing run, so the
record travels with the code. Regenerate it with `run_theory` if you change
anything.

---

## `data/`

The result CSVs behind the manuscript tables. `sppg_analysis.py` reads them
directly and accepts the legacy `r4_arms.csv` filename and the `so(32)` /
`median_spectral_radius` spellings.

`.gitignore` excludes `*_cells.csv` so that new runs are not committed by
accident; the files in `data/` are tracked because they are the
ones the paper's tables come from.

Per-seed records, frozen problem objects, checkpoints and hash manifests are
**not** in this repository.

---

## `tools/fixdiff.py`

Repairs `latexdiff` output. `latexdiff` emits an unbalanced `\DIFdel{` run when
it meets a `\MBLOCKRIGHTBRACE` artefact, which makes the diff uncompilable.
This neutralises those runs, recolours additions red and deletions grey, and
removes pointers to objects the revision deleted.

```bash
latexdiff --type=CFONT --math-markup=off --disable-citation-markup \
  --config="PICTUREENV=(?:picture|DIFnomarkup|table|tabular|algorithm|figure|thebibliography)[\w\d*@]*" \
  submitted.tex revised.tex > raw.tex
python3 tools/fixdiff.py raw.tex tracked.tex
```

---

## When something raises

**`./check_all.sh: Permission denied`**
The executable bit was lost when the files were copied or downloaded.
`chmod +x check_all.sh verify.sh`, or just run `bash check_all.sh`.

**`this checkout is STALE -- this copy predates the IQM convention fix`**
Stage 1 found the old index-slice IQM convention, so its Task-1 IQM will
disagree with Table `tab:task1` by 0.035. See [The IQM convention](#the-iqm-convention-read-this-before-comparing-to-table-tabtask1).

**`_geometry_reward fired 0 time(s) during this step, expected exactly 1`**
The guard that actually polices the channel split. The task channel is
recovered by inverting the mixture, `t = (reward - w*g)/(1-w)`, so the
reconstruction `(1-w)*t + w*g` equals `reward` *identically, for any g* --
meaning `recon_err ~ 0` is **not** evidence that `g` is right. This counter is.
If it fires, `main.py` has renamed, moved or bypassed `_geometry_reward` and the
decomposition is unmeasured. Fix the wiring; do not pass `strict=False`.

**`<file> exists and is non-empty. Refusing to truncate a previous sweep.`**
Working as intended; there is no resume path by design. Move the file aside or
re-run with `--overwrite`.

**`<file> was produced under different settings`**
The `.meta.json` disagrees with what the analysis assumes -- a different grid,
geometry weight, or `main.py`. Re-run the sweep, or analyse with the matching
version.

**`control parameters did not move; the comparison would again be trained-vs-untrained`**
`r11` stops rather than produce the defect it exists to correct. The released Task-2
baseline was built with `lmbda = 0`, which severs the gradient path entirely.
Investigate rather than removing the check.

**`torch.OutOfMemoryError`**
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is already set by the scripts
that need it. If it still OOMs, lower `--seeds` and concatenate, or use a larger
card. `r13b` loads Mistral-7B in 4-bit, extracts every embedding, then frees the
model before training -- if the machine cannot hold it even briefly, nothing in
the script can help.

**`ModuleNotFoundError: No module named 'sppg_defense'`**
You are in the wrong directory. Those are `python -m` invocations and must be
run from `verification/`.

---

## Modifying things safely

- **Adding an experiment:** write `run_<name>()` returning a DataFrame and
  writing through `Sink`; add it to `EXPERIMENTS`; add `analyse_<name>()` and
  register it in `ANALYSES` and `ORDER`. Both selftests then cover the plumbing.
- **Changing the equivalence margin:** edit `MARGIN` at the top of
  `sppg_analysis.py` -- *before* looking at a result, or every equivalence claim
  becomes unfalsifiable.
- **Changing a verdict's wording:** the selftests assert on these strings. A
  failure after your edit is the test doing its job.

The comment beside each guard says which failure it prevents.

---

## The IQM convention (read this before comparing to Table `tab:task1`)

"Interquartile mean" names two different estimators, and on ten seeds they do
not agree.

- **Percentile band** -- take `q25`, `q75` by linear interpolation, average every
  value in the *closed* interval `[q25, q75]`. On Task-1's SPPG arm that keeps
  4 of 10 seeds. This is what `main.py` and `sppg_core.py` do, and it is the
  estimand the manuscript reports.
- **Index slice** -- sort and average `x[floor(n/4) : ceil(3n/4)]`, i.e. always
  the middle 6 of 10. Some rliable-style implementations do this.

On the Task-1 SPPG arm: percentile -> **8.6525**, slice -> **8.6867**. The paper
prints 8.652. The slice convention appears to contradict the table by 0.035
only because it measures a different quantity. `stats.tests.iqm` now defaults to `method="percentile"` and accepts
`method="slice"` explicitly, and the selftest asserts both that the percentile
value reproduces the table to within 0.002 and that the two conventions really
do differ -- so this cannot regress unnoticed.

One related caveat: the per-seed Task-1 returns are quoted to two decimals, so
the reconstructions in `selftest.py` fix any standard-deviation-denominated
statistic only to within the rounding envelope. De-rounding by
`U(-0.005, +0.005)` over 20 000 draws puts Welch *t* in [30.446, 31.456] and
pooled *d* in [13.62, 14.07]; the paper's 30.663 and 13.7 lie inside both. The
selftest tolerances are that envelope, not slack. The IQM is a trimmed mean and
is essentially unmoved by the rounding -- for it, only the convention matters.

## The r7 M_env seeds (resolved)

r7 is the one sweep whose result depends on a matrix drawn per *geometry*
rather than per seed, from a seed derived from the geometry's index. The
consecutive layout (`9000 + gi`) collides once `n_draws > 1`. Spacing the
seeds by geometry instead (`9000 + gi*100 + draw` -> 9000, 9100, 9200, 9300)
avoids the collision but no longer reproduces `data/r7_cells.csv`: `so` is
index 0 under both and `id` is the identity for any seed, so two of the five
rows still agree while `sym`, `gl` and `diag` describe *different*
environment matrices, and the scope verdict can move.

The layout used is `9000 + gi + 100*draw`, which is collision-free **and**
reproduces the published table: draw 0 gives 9000-9004, the seeds Table
`tab:geomfactorial` was generated with. Spacing draws instead of geometries
costs nothing -- there are five geometries, far fewer than 100.

`menv_seed()` in `sppg_experiments.py` is the single source of truth, and
`tools/check_r7_provenance.py` imports it rather than restating it, so the CSVs
and the check cannot drift apart. It runs as stage 11 of `check_all.sh`
and re-derives every geometry's matrix from the seed the code would use today,
comparing against the diagnostics stored in the CSV.

At the shipped `n_draws=1` each row rests on one matrix per family, as the
manuscript's *Limitation* paragraph states ("off-diagonal rows are indicative,
not established"). Running `python3 sppg_experiments.py r7 --draws 3 --seeds 5`
makes the draw the replication unit.

## Independent re-run of the experiments

The shipped CSVs in `data/` have been reproduced from scratch on different
hardware and a different software stack (Python 3.10.20, numpy 2.2.6,
scipy 1.15.3, pandas 2.3.3, torch 2.11.0+cu130, RTX 5060 Laptop) from the
environment they were originally produced in.

All twelve sweeps were re-run. Eleven reproduce `data/` exactly:

| Sweep | Cells | Agreement with `data/` |
|---|---:|---|
| `b1` | 60 | all cells, to the 3 dp the run prints |
| `s7` | 100 | all cells |
| `x3` | 300 | all cells |
| `b1lr` | 110 | all cells |
| `r4` | 120 | all cells |
| `r5` | 210 | all cells |
| `r15` | 100 | all cells |
| `r15b` | 60 | all cells |
| `r1b` | 250 | all cells |
| `r13b` | 100 | all cells |
| `r11` | 1440 | analysis output identical (no per-cell print) |
| `r7` | 250 | all cells, once the seed layout was restored |

1410 of 1410 per-cell comparisons across the ten sweeps that print cells, max
deviation 5.0e-4 -- entirely accounted for by the 3-decimal print precision.
`r11` prints no cells; its analysis output was identical. The 20-seed `r4` arm means land
within 8e-5 of the shipped values and within 5e-4 of Table `tab:ablation`.

r7 was the exception during that re-run, and it was not nondeterminism: the
seed layout had drifted, as described above. With `menv_seed()` restored, the
seeds that reproduce the shipped `sym`, `gl` and `diag` rows are exactly the
ones the code now uses.

Running the whole analysis on the fresh CSVs rather than `data/` reproduces
every section verbatim -- R-1, R-4, R-5, X-3, B-1, B-1-LR, S-7, R-11, R-15,
R-15b, R-13b -- with the only differences anywhere being the three r7 rows and
the scope verdict that follows from them.

The verification stages were byte-identical apart from two rounding digits in
`V3` and `V5` (8.66e-13 vs 8.67e-13), which is BLAS noise across scipy versions.

## Test coverage

`check_all.sh` runs the verification stages listed above; with the
prerequisites installed, all eleven pass in about two minutes. In
`--experiments` mode it checks for `torch`, `transformers` and a CUDA device
before starting, rejects unknown `--only` names, runs the sweeps cheapest-first,
and continues past a failing sweep, reporting it in the summary and exit status.

**Not covered by `check_all.sh`:** training itself, which needs `transformers`
and a CUDA GPU. The sweeps were run end to end on GPU as described in
[Independent re-run of the experiments](#independent-re-run-of-the-experiments).
In a new environment, run one short sweep (`s7`, ~15 min) before relying on the
collection path.
