#!/usr/bin/env bash
# Kindred CI gate. Runs every check, prints its findings, and exits non-zero
# when any check fails. Temporary tests are run separately.
set -uo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd -- "${REPO_ROOT}"

# Names vulture cannot see being used: Qt, pyqtgraph and QSyntaxHighlighter
# overrides, sympy printer hooks and signature parameters, pickle hooks,
# attributes read by openpyxl/zipfile/decimal/pyqtgraph.
VULTURE_IGNORE_NAMES="closeEvent,expandingDirections,minimumHeightForWidth,highlightBlock,showAxRect,\
_print_*,bold_name,persistent_id,persistent_load,\
sheet_state,create_system,external_attr,prec,border"

# Checks are only meaningful against the pinned environment, so a failed sync
# stops CI before any check runs.
echo "=== environment"
if ! python3 tools/audit/sync_env.py --group dev; then
  echo "CI FAILED: environment"
  exit 1
fi
echo

failed=()

run_check() {
  local name="$1"
  shift
  echo "=== ${name}"
  if "$@"; then
    echo "PASS ${name}"
  else
    echo "FAIL ${name}"
    failed+=("${name}")
  fi
  echo
}

run_check "ruff" ruff check . --no-cache --force-exclude
run_check "import graph" python3 tools/audit/import_graph.py
run_check "import sweep" python3 tools/audit/import_sweep.py
run_check "vulture" vulture kindred --min-confidence 60 --ignore-names "${VULTURE_IGNORE_NAMES}"

if [[ "${#failed[@]}" -gt 0 ]]; then
  echo "CI FAILED: ${failed[*]}"
  exit 1
fi
echo "CI PASSED"
