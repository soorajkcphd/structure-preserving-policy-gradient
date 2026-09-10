#!/usr/bin/env sh
# Run every check that needs no GPU and no model download.  Exits non-zero on
# the first failure, so it is safe to use in CI or as a pre-push hook.
set -e
here=$(cd "$(dirname "$0")" && pwd)

echo "== 1/5  independent numerical verification (shares no code with the project)"
python3 "$here/verification/verify_math.py"

echo "\n== 2/5  sppg_defense selftest"
cd "$here/verification" && python3 -m sppg_defense.selftest

echo "\n== 3/5  mutation suite (confirms the checks can fail)"
cd "$here/verification" && python3 -m sppg_defense.mutation_test

echo "\n== 4/5  harness selftests"
cd "$here/src" && python3 sppg_core.py && python3 sppg_analysis.py --selftest

echo "\n== 5/5  manuscript reconciliation against data/"
cd "$here/src" && python3 sppg_analysis.py --manuscript --dir ../data

echo "\nAll checks completed."
