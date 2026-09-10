#!/usr/bin/env bash
# ===========================================================================
#  check_all.sh -- one-shot verification of everything in this repository that
#  needs no GPU and no model download.
#
#  Run it from anywhere:      ./check_all.sh
#  It resolves its own location, so you cannot get the "wrong directory"
#  errors that come from running the per-file commands by hand.
#
#  Unlike verify.sh (which stops at the first failure, for CI), this script
#  runs every stage even if one fails, then prints a summary table and exits
#  non-zero if anything failed.  A full transcript is written to
#  check_all.log next to this script.
#
#  This script installs nothing.  There is no pip call and no conda call
#  anywhere in it -- see the note at stage 0.
#  It runs your active interpreter on files inside this repository and touches
#  nothing else.  If a package is missing it names it and stops.
#
#  Options:
#     --quick         skip the two slowest verification stages
#     --no-log        print to the terminal only, do not write check_all.log
#     --python PATH   interpreter to use (default: $PYTHON, else python3, else
#                     python). Inside an activated conda env the default is
#                     already that env's interpreter.
#
#  Running the actual experiments (GPU, and they are long):
#     --experiments        verification first, then all 12 sweeps (long -- nohup it)
#     --experiments-only   skip verification, go straight to the sweeps
#     --only a,b,c         run just these sweeps, in the order given
#     --overwrite          let a sweep replace an existing <name>_cells.csv
#                          (without it a sweep will not truncate one)
#
#  A long run survives a dropped connection if you detach it:
#     nohup ./check_all.sh --experiments > run.out 2>&1 &
# ===========================================================================
# This script uses bash arrays, so re-exec under bash if invoked as `sh check_all.sh`.
if [ -z "${BASH_VERSION:-}" ]; then
  if command -v bash >/dev/null 2>&1; then exec bash "$0" "$@"; fi
  echo "check_all.sh needs bash. Install bash, or run the stages listed in README.md by hand." >&2
  exit 1
fi

set -u
set -o pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$HERE" || exit 1

QUICK=0; LOG="$HERE/check_all.log"; PY="${PYTHON:-}"
EXPERIMENTS=0; VERIFY=1; OVERWRITE=""; ONLY=""
while [ $# -gt 0 ]; do
  case "$1" in
    --quick)   QUICK=1 ;;
    --no-log)  LOG="" ;;
    --python)  shift; [ $# -gt 0 ] || { echo "check_all.sh: --python needs a path"; exit 2; }; PY="$1" ;;
    --python=*) PY="${1#--python=}" ;;
    --experiments)      EXPERIMENTS=1 ;;
    --experiments-only) EXPERIMENTS=1; VERIFY=0 ;;
    --overwrite)        OVERWRITE="--overwrite" ;;
    --only)    shift; [ $# -gt 0 ] || { echo "check_all.sh: --only needs a list"; exit 2; }; ONLY="$1"; EXPERIMENTS=1 ;;
    --only=*)  ONLY="${1#--only=}"; EXPERIMENTS=1 ;;
    # Accepted and ignored: this script has no install path, by design.
    --install) echo "check_all.sh: --install is gone; this script never installs anything." ;;
    -h|--help) sed -n '2,35p' "$0"; exit 0 ;;
    *) echo "check_all.sh: unknown option '$1' (try --help)"; exit 2 ;;
  esac
  shift
done
[ -n "$LOG" ] && : > "$LOG"

# Interpreter: whatever is active. Inside an activated conda env, python3 is
# already that env's python -- we do not go looking for another one.
if [ -z "$PY" ]; then
  if command -v python3 >/dev/null 2>&1; then PY=python3
  elif command -v python >/dev/null 2>&1; then PY=python
  else echo "check_all.sh: no python3 on PATH. Activate your environment first." >&2; exit 1; fi
fi
if ! command -v "$PY" >/dev/null 2>&1 && [ ! -x "$PY" ]; then
  echo "check_all.sh: interpreter '$PY' not found." >&2; exit 1
fi

if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
  G=$'\033[32m'; R=$'\033[31m'; Y=$'\033[33m'; B=$'\033[1m'; N=$'\033[0m'
else
  G=""; R=""; Y=""; B=""; N=""
fi

say() { if [ -n "$LOG" ]; then printf '%s\n' "$*" | tee -a "$LOG"; else printf '%s\n' "$*"; fi; }
rule(){ say "------------------------------------------------------------------------"; }

# Experiment sweeps, ordered cheapest first so a broken environment shows up in
# twelve minutes rather than after the first two-and-a-half-hour run. r15b reads
# r15's output, so it must follow it. Timings are for an 8 GB RTX 5060.
SWEEPS=(b1:12m s7:15m x3:20m b1lr:20m r4:30m r5:35m r15:35m r15b:35m \
        r1b:45m r13b:50m r11:2.5h r7:2.5h)

# Resolve --only now, before touching the environment, so a typo costs a second
# rather than surfacing after the prerequisite checks.
PLAN=()
if [ "$EXPERIMENTS" -eq 1 ]; then
  if [ -n "$ONLY" ]; then
    IFS=', ' read -r -a WANT <<< "$ONLY"
    for w in "${WANT[@]}"; do
      [ -z "$w" ] && continue
      hit=""
      for s in "${SWEEPS[@]}"; do [ "${s%%:*}" = "$w" ] && hit="$s"; done
      if [ -z "$hit" ]; then
        echo "check_all.sh: unknown sweep '$w'." >&2
        echo "  known: $(printf '%s ' "${SWEEPS[@]%%:*}")" >&2
        exit 2
      fi
      PLAN+=("$hit")
    done
    [ ${#PLAN[@]} -gt 0 ] || { echo "check_all.sh: --only matched no sweeps." >&2; exit 2; }
  else
    PLAN=("${SWEEPS[@]}")
  fi
fi

NAMES=(); CODES=()
STAGE=0
TOTAL=11; [ "$QUICK" -eq 1 ] && TOTAL=9
[ "$VERIFY" -eq 0 ] && TOTAL=0
TOTAL=$((TOTAL + ${#PLAN[@]}))

# run <label> <working-dir> <command...>
run() {
  local label="$1"; shift
  local dir="$1";   shift
  STAGE=$((STAGE + 1))
  say ""
  say "${B}== ${STAGE}/${TOTAL}  ${label}${N}"
  say "   \$ cd ${dir#$HERE/} && $*"
  rule
  local rc=0
  if [ -n "$LOG" ]; then
    ( cd "$dir" && "$@" ) 2>&1 | tee -a "$LOG"; rc=${PIPESTATUS[0]}
  else
    ( cd "$dir" && "$@" ); rc=$?
  fi
  rule
  if [ "$rc" -eq 0 ]; then say "   ${G}PASS${N}  $label"; else say "   ${R}FAIL${N}  $label  (exit $rc)"; fi
  NAMES+=("$label"); CODES+=("$rc")
  return 0
}

PYPATH=$("$PY" -c 'import sys;print(sys.executable)' 2>/dev/null || echo "$PY")
PYVER=$("$PY" -c 'import sys;print(sys.version.split()[0])' 2>/dev/null || echo UNKNOWN)
CONDA=0
if [ -n "${CONDA_PREFIX:-}" ] || case "$PYPATH" in *conda*|*mamba*) true ;; *) false ;; esac; then CONDA=1; fi

MODE="verification only"
[ "$EXPERIMENTS" -eq 1 ] && [ "$VERIFY" -eq 1 ] && MODE="verification, then experiments"
[ "$EXPERIMENTS" -eq 1 ] && [ "$VERIFY" -eq 0 ] && MODE="experiments only"

say "========================================================================"
say "${B} SPPG -- $MODE${N}"
say " repo:    $HERE"
say " date:    $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
say " python:  $PYVER  ($PYPATH)"
if [ "$CONDA" -eq 1 ]; then
  say " env:     conda${CONDA_DEFAULT_ENV:+ -- ${CONDA_DEFAULT_ENV}}"
else
  say " env:     ${VIRTUAL_ENV:-system}"
fi
say " install: ${G}none${N} -- this script does not install anything"
[ -n "$LOG" ] && say " log:     $LOG"
[ "$QUICK" -eq 1 ] && say " mode:    --quick (mutation suite and full analysis skipped)"
say "========================================================================"

# --- stage 0: prerequisites ------------------------------------------------
# No install path.  This stage only imports and reports: if something is
# missing it names it and stops, and you decide how to install it.  There is
# no pip step because this repository is normally used inside a conda env,
# where a stray pip install shadows conda-managed builds of numpy/scipy and
# breaks the env in ways that surface much later.
say ""
say "${B}== 0/${TOTAL}  prerequisites  (import-and-report only -- nothing is installed)${N}"
rule
MISSING=""
for p in numpy scipy pandas mpmath; do
  v=$("$PY" -c "import $p;print(getattr($p,'__version__','?'))" 2>/dev/null)
  if [ -n "$v" ]; then say "   present   $p $v"; else say "   ${R}MISSING   $p${N}"; MISSING="$MISSING $p"; fi
done
GPUMISS=""
for p in torch transformers; do
  v=$("$PY" -c "import $p;print(getattr($p,'__version__','?'))" 2>/dev/null)
  if [ -n "$v" ]; then say "   present   $p $v"
  else say "   ${Y}absent    $p${N}"; GPUMISS="$GPUMISS $p"; fi
done
if [ -n "$MISSING" ]; then
  say ""
  say "${R}   Required packages missing:$MISSING${N}"
  if [ "$CONDA" -eq 1 ]; then
    say "       conda install -c conda-forge$MISSING"
  else
    say "       $PY -m pip install$MISSING"
  fi
  say ""
  say "   Nothing was installed and nothing was changed."
  exit 1
fi
if [ "$EXPERIMENTS" -eq 1 ]; then
  if [ -n "$GPUMISS" ]; then
    say ""
    say "${R}   The experiments need:$GPUMISS${N}"
    say "   Verification does not. Drop --experiments, or install those first."
    say "   Nothing was installed and nothing was changed."
    exit 1
  fi
  CUDA=$("$PY" -c 'import torch;print("yes, "+torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO")' 2>/dev/null || echo "NO")
  say "   cuda      $CUDA"
  if [ "$CUDA" = "NO" ]; then
    say ""
    say "${Y}   torch reports no CUDA device. The sweeps will run on CPU and take${N}"
    say "${Y}   days rather than hours. Stopping instead of guessing.${N}"
    exit 1
  fi
fi
rule
say "   ${G}PASS${N}  prerequisites"

if [ "$VERIFY" -eq 1 ]; then

# --- stage 1: is this the updated checkout? --------------------------------
run "build check -- confirms the IQM fix is present in this checkout" "$HERE" \
    "$PY" -c "
import sys, pathlib
sys.path.insert(0, 'verification')
from sppg_defense.stats.tests import iqm
d = [8.56,9.05,8.63,9.09,8.72,8.58,8.64,8.54,8.62,8.93]
p, s = iqm(d), iqm(d, method='slice')
print('percentile-band IQM (main.py convention) : %.4f   <- Table tab:task1 prints 8.652' % p)
print('index-slice     IQM (other convention)   : %.4f' % s)
ok = abs(p - 8.652) < 2e-3 and abs(s - 8.6867) < 2e-3
print()
print('this checkout is', 'UP TO DATE (percentile-band IQM convention)' if ok
      else 'STALE -- this copy predates the IQM convention fix')
sys.exit(0 if ok else 1)
"

# --- stage 2: independent numerics ----------------------------------------
run "independent numerical verification (shares no code with the project)" "$HERE" \
    "$PY" verification/verify_math.py

# --- stage 3: defense selftest --------------------------------------------
run "sppg_defense selftest (theory vs the project's own implementation)" "$HERE/verification" \
    "$PY" -m sppg_defense.selftest

# --- stage 4: theory modules ----------------------------------------------
run "theory modules t01-t10" "$HERE/verification" \
    "$PY" -m sppg_defense.run_theory

# --- stage 5: mutation suite ----------------------------------------------
if [ "$QUICK" -eq 0 ]; then
  run "mutation suite (confirms the checks are able to fail)" "$HERE/verification" \
      "$PY" -m sppg_defense.mutation_test
fi

# --- stage 6: harness selftests -------------------------------------------
run "sppg_core selftest" "$HERE/src" "$PY" sppg_core.py
run "sppg_experiments selftest" "$HERE/src" "$PY" sppg_experiments.py --selftest
run "sppg_analysis selftest" "$HERE/src" "$PY" sppg_analysis.py --selftest

# --- stage 7: manuscript reconciliation -----------------------------------
run "manuscript reconciliation against data/" "$HERE/src" \
    "$PY" sppg_analysis.py --manuscript --dir ../data

# --- stage 8: full analysis -----------------------------------------------
if [ "$QUICK" -eq 0 ]; then
  run "full analysis of every shipped CSV" "$HERE/src" \
      "$PY" sppg_analysis.py --dir ../data
fi

# --- stage 9: r7 provenance ------------------------------------------------
# r7 is the one sweep whose result depends on a matrix drawn per geometry, from
# a seed derived from the geometry's index. A CSV written under an older seed
# formula still looks well-formed but describes different environments. See
# README, "The r7 provenance mismatch".
run "r7 provenance -- does data/r7_cells.csv match the current seeds?" "$HERE" \
    "$PY" tools/check_r7_provenance.py

fi   # end of verification block

# --- experiments -----------------------------------------------------------
# Long, GPU-bound, and they write: each sweep produces src/<name>_cells.csv and
# a .meta.json.  Without --overwrite a sweep will not truncate an existing
# file; that guard belongs to the harness and is left in place.
if [ "$EXPERIMENTS" -eq 1 ]; then
  say ""
  say "========================================================================"
  say "${B} EXPERIMENTS${N}   ${#PLAN[@]} sweep(s), cheapest first"
  say "========================================================================"
  for s in "${PLAN[@]}"; do say "   ${s%%:*}   ~${s##*:}"; done
  say ""
  say "   Each writes src/${B}<name>${N}_cells.csv plus a .meta.json recording the"
  say "   grid, the seed count and an MD5 of main.py."
  if [ -z "$OVERWRITE" ]; then
    say "   A sweep whose CSV already exists will not run. Pass --overwrite to"
    say "   replace them."
  else
    say "   ${Y}--overwrite is set: existing CSVs will be replaced.${N}"
  fi
  say "   Failures do not stop the run; the summary lists them at the end."

  for s in "${PLAN[@]}"; do
    name="${s%%:*}"; eta="${s##*:}"
    run "experiment $name  (~$eta)" "$HERE/src" \
        "$PY" sppg_experiments.py "$name" $OVERWRITE
  done

  # Re-analyse whatever was produced, so the run ends with readable results.
  if [ "$VERIFY" -eq 1 ]; then
    say ""
    say "   Re-analysing the freshly written CSVs (src/, not data/):"
    ( cd "$HERE/src" && "$PY" sppg_analysis.py --dir . ) 2>&1 \
      | (if [ -n "$LOG" ]; then tee -a "$LOG"; else cat; fi) || true
  fi
fi

# --- summary ---------------------------------------------------------------
say ""
say "========================================================================"
say "${B} SUMMARY${N}"
say "========================================================================"
FAILED=0
for i in "${!NAMES[@]}"; do
  if [ "${CODES[$i]}" -eq 0 ]; then
    say "  ${G}PASS${N}  ${NAMES[$i]}"
  else
    say "  ${R}FAIL${N}  ${NAMES[$i]}   (exit ${CODES[$i]})"
    FAILED=$((FAILED + 1))
  fi
done
say "------------------------------------------------------------------------"
if [ "$FAILED" -eq 0 ]; then
  say "  ${G}${B}ALL ${#NAMES[@]} STAGES PASSED${N}"
  [ "$QUICK" -eq 1 ] && say "  (--quick was used: mutation suite and full analysis were skipped)"
  say ""
  if [ "$EXPERIMENTS" -eq 1 ]; then
    say "  Fresh CSVs are in src/. To read them:"
    say "      cd src && $PY sppg_analysis.py --dir ."
    say "  To compare against the shipped run:  $PY sppg_analysis.py --dir ../data"
  else
    say "  What this does not cover: actually training anything. That needs a GPU"
    say "  plus torch and transformers. Either run one short sweep --"
    say "      cd src && $PY sppg_experiments.py s7"
    say "  -- or run all twelve in one go:"
    say "      ./check_all.sh --experiments        (long -- nohup it)"
  fi
  [ -n "$LOG" ] && say "  Transcript: $LOG"
  say "========================================================================"
  exit 0
else
  say "  ${R}${B}$FAILED of ${#NAMES[@]} STAGES FAILED${N}"
  [ -n "$LOG" ] && say "  See $LOG for the full traceback."
  say "========================================================================"
  exit 1
fi
