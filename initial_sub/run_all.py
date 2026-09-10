"""
SP-PG Experiment Runner
========================
Maps every script to the corresponding paper section.
All experiments for the Neural Networks submission.

Paper sections covered:
  Sec 4   Structure selection (main.py)
  Sec 8   Task 1: SP-PG Response Selection (main.py)
  Sec 8   Task 1: Additional ablations (task1_extra.py)
  Sec 8   Task 1: Figures (task1_figures.py)
  Sec 9   Task 2: Single seed (04_pap_CG.py)
  Sec 9   Task 2: 10-seed (task2_multiseed.py)
  Sec 10  Mistral-7B cross-model (run_mistral_task1.py)
  Sec 11  Task 3: Sentiment falsification (run_sentiment_task3.py)

Usage:
    python run_all.py              # full run
    python run_all.py --quick      # smoke test (quick mode per script)
    python run_all.py --skip 1,2   # skip steps by number
    python run_all.py --only 4,5   # run only these steps

Outputs:
    outputs/   all csv, json, tex, png files
"""

import subprocess
import sys
import os
import shutil
import time
import argparse
from pathlib import Path

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Step definitions: maps paper section -> script
STEPS = {
    1: {
        "paper":   "Sec 4,8 -- Structure selection + Task 1 main result, multiseed, ablation, overhead",
        "tables":  "Tab 1 (structure), Tab 2 (hyperparams), Tab 3 (task1), Tab 4 (perseed), Tab 5 (ablation/sweep)",
        "figures": "Fig 1 (structure), Fig 2 (returns), Fig 3 (multiseed), Fig 4 (ablation)",
        "script":  "main.py",
        "quick":   None,
        "time":    "~15 min",
    },
    2: {
        "paper":   "Sec 8 -- Task 1 figures: spectral radius, fourway ablation",
        "tables":  "None (figures only)",
        "figures": "Fig spectral_comparison.png",
        "script":  "task1_figures.py",
        "quick":   None,
        "time":    "~10 min",
    },
    3: {
        "paper":   "Sec 9 -- Task 2 text generation, single seed (structure discovery + REINFORCE)",
        "tables":  "Tab task2 validation summary",
        "figures": "Task 2 learning curves",
        "script":  "04_pap_CG.py",
        "quick":   None,
        "time":    "~3 min",
    },
    4: {
        "paper":   "Sec 9 -- Task 2 text generation, 10-seed (Tab task2, CI, Welch, MW)",
        "tables":  "Tab task2 (primary) -- Welch t, Mann-Whitney, bootstrap CI",
        "figures": "None",
        "script":  "task2_multiseed.py",
        "quick":   None,
        "time":    "~30 min",
    },
    5: {
        "paper":   "Sec 10 -- Cross-model validation: Mistral-7B (4-bit, 4096->1024, 10 seeds)",
        "tables":  "Tab mistral",
        "figures": "None",
        "script":  "run_mistral_task1.py",
        "quick":   None,
        "time":    "~20 min",
    },
    6: {
        "paper":   "Sec 11 -- Falsification: Task 3 sentiment steering (no geometry, 10 seeds)",
        "tables":  "Tab task3",
        "figures": "None",
        "script":  "run_sentiment_task3.py",
        "quick":   None,
        "time":    "~40 min",
    },
}


# Output file patterns: collected to outputs/ after each step
OUTPUTS_BY_STEP = {
    1: ["*.png", "*.json", "*.csv"],
    2: ["*.png"],
    3: ["*.json", "*.png"],
    4: ["*.json", "*.png", "*.csv"],
    5: ["*.json", "*.csv"],
    6: ["*.json"],
}


# Helpers
def check_dependencies():
    print("Checking dependencies...")
    required = {
        "torch":        "torch",
        "numpy":        "numpy",
        "scipy":        "scipy",
        "transformers": "transformers",
        "matplotlib":   "matplotlib",
    }
    missing = []
    for module, package in required.items():
        try:
            __import__(module)
            print(f"  OK  {module}")
        except ImportError:
            print(f"  MISSING  {module}  ->  pip install {package}")
            missing.append(package)
    if missing:
        print(f"\nInstall missing packages:")
        print(f"  pip install {' '.join(missing)}")
        sys.exit(1)
    print("  All dependencies OK\n")


def check_scripts(run_steps):
    print("Checking scripts exist...")
    missing = []
    for step_num in run_steps:
        step = STEPS[step_num]
        script = Path(step["script"])
        if not script.exists():
            print(f"  MISSING  {script}  (needed for step {step_num})")
            missing.append(str(script))
        else:
            print(f"  OK  {script}")
        # Check extra required files (e.g. 04_pap_CG.py for task2_multiseed.py)
        for req in step.get("requires", []):
            if not Path(req).exists():
                print(f"  MISSING  {req}  (required by step {step_num}: {script})")
                missing.append(req)
            else:
                print(f"  OK  {req}  (required by step {step_num})")
    if missing:
        print(f"\nMissing files: {missing}")
        print("Make sure all scripts and their dependencies are in the same directory.")
        sys.exit(1)
    print()


def collect_outputs(step_num):
    """Copy output files matching patterns to outputs/.
    Searches both the current directory and the plots/ subdirectory,
    because main.py / task1_extra.py / task1_figures.py save PNG files
    to plots/ not to the root directory.
    """
    import glob
    patterns = OUTPUTS_BY_STEP.get(step_num, [])
    saved = []
    # Search root dir and plots/ subdir
    search_dirs = ["."]
    if Path("plots").is_dir():
        search_dirs.append("plots")

    for search_dir in search_dirs:
        for pattern in patterns:
            full_pattern = os.path.join(search_dir, pattern)
            for fpath in glob.glob(full_pattern):
                p = Path(fpath)
                # Skip files already inside outputs/
                if "outputs" in str(p.resolve()):
                    continue
                if p.is_file():
                    dst = OUTPUT_DIR / p.name
                    shutil.copy2(p, dst)
                    saved.append(p.name)
    if saved:
        print(f"  Saved to outputs/: {', '.join(sorted(set(saved)))}")


def print_step_header(step_num, quick):
    step = STEPS[step_num]
    mode = "(QUICK)" if quick and step["quick"] else ""
    print("=" * 70)
    print(f"STEP {step_num}  {mode}")
    print(f"  Paper:   {step['paper']}")
    print(f"  Tables:  {step['tables']}")
    print(f"  Figures: {step['figures']}")
    print(f"  Script:  {step['script']}")
    print(f"  Time:    {step['time']}")
    print("=" * 70)


def run_step(step_num, quick):
    """Run one step. Returns True on success."""
    step = STEPS[step_num]
    script = step["script"]

    cmd = [sys.executable, script]
    if quick and step["quick"]:
        cmd.append(step["quick"])

    print(f"  Running: {' '.join(cmd)}\n")
    t0 = time.time()
    result = subprocess.run(cmd, check=False)
    elapsed = time.time() - t0

    collect_outputs(step_num)

    if result.returncode != 0:
        print(f"\n  FAILED: step {step_num} returned code {result.returncode}")
        print(f"  Fix the error above, then re-run with --skip {','.join(str(i) for i in range(1, step_num))}")
        return False

    print(f"\n  Step {step_num} done in {elapsed/60:.1f} min")
    return True


def print_summary(results, quick, all_steps):
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for step_num in sorted(all_steps):
        if step_num in results:
            status = "PASS" if results[step_num] else "FAIL"
        else:
            status = "SKIP"
        print(f"  Step {step_num}: {status}  --  {STEPS[step_num]['paper']}")

    print("\nOutputs in ./outputs/")
    for f in sorted(OUTPUT_DIR.iterdir()):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name:50s}  {size_kb:6.1f} KB")

    print("\nPaper integration:")
    print("  outputs/*.json  ->  numerical results")
    print("  outputs/*.csv   ->  per-seed data")
    print("  outputs/*.png   ->  figures (copy to paper figs/ folder)")

    if quick:
        print("\nNOTE: run was in --quick mode. For publication results:")
        print("  python run_all.py")

    all_ok = all(results.values())
    if all_ok:
        print("\nAll steps passed.")
    else:
        failed = [str(k) for k, v in results.items() if not v]
        print(f"\nFailed steps: {', '.join(failed)}")


# Main
def main():
    parser = argparse.ArgumentParser(description="SP-PG Experiment Runner")
    parser.add_argument("--quick", action="store_true",
                        help="Run steps 4 and 5 in quick mode (3 seeds x 20 iters)")
    parser.add_argument("--skip", type=str, default="",
                        help="Comma-separated step numbers to skip, e.g. --skip 1,2,3")
    parser.add_argument("--only", type=str, default="",
                        help="Run only these steps, e.g. --only 4,5")
    args = parser.parse_args()

    skip_steps = {int(s.strip()) for s in args.skip.split(",") if s.strip()}
    only_steps = {int(s.strip()) for s in args.only.split(",") if s.strip()}

    if only_steps:
        run_steps = sorted(only_steps)
    else:
        run_steps = [s for s in sorted(STEPS.keys()) if s not in skip_steps]

    mode = "QUICK" if args.quick else "FULL"
    print("=" * 70)
    print(f"SP-PG EXPERIMENT RUNNER  [{mode}]")
    print(f"Steps: {run_steps}")
    print("=" * 70)
    print()

    check_dependencies()
    check_scripts(run_steps)

    results = {}
    total_t0 = time.time()

    for step_num in run_steps:
        print_step_header(step_num, args.quick)
        ok = run_step(step_num, args.quick)
        results[step_num] = ok
        if not ok:
            break
        print()

    total_elapsed = time.time() - total_t0
    print(f"Total time: {total_elapsed/60:.1f} min")

    print_summary(results, args.quick, run_steps)


if __name__ == "__main__":
    main()
