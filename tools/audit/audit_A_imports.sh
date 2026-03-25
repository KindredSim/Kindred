#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

REPORT_DIR="${1:?usage: audit_A_imports.sh <report_dir>}"
OUT_FILE="${REPORT_DIR}/A_imports_and_cycles.txt"
SMOKE_LOG="${REPORT_DIR}/A_import_smoke.log"

timestamp_utc="$(date -u +"%Y%m%dT%H%M%SZ")"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

hard_fail=0

{
  echo "Kindred Audit A: Imports / Layering / Cycles / Orphans"
  echo "Timestamp (UTC): ${timestamp_utc}"
  echo "Repo root: ${REPO_ROOT}"
  echo
  echo "Finding format:"
  echo "RULE-ID | SEVERITY | file:line | message | suggested-fix"
  echo
} > "${OUT_FILE}"

{
  echo "=== Import smoke test (A3) ==="
  echo "PYTHONPATH=${PYTHONPATH}"
  echo
  echo "\$ python3 -c \"import kindred\""
  python3 -c "import kindred"
  echo
  echo "\$ python3 -c \"import kindred.core.simulator.dsl, kindred.core.simulator.fast_eq, kindred.core.simulator.common, kindred.core.validation\""
  python3 -c "import kindred.core.simulator.dsl, kindred.core.simulator.fast_eq, kindred.core.simulator.common, kindred.core.validation"
} > "${SMOKE_LOG}" 2>&1 || hard_fail=1

if [[ "${hard_fail}" -ne 0 ]]; then
  echo "A3 ImportSmoke | FAIL | - | Import smoke test failed (see A_import_smoke.log) | Fix ImportError/SyntaxError at import time" >> "${OUT_FILE}"
else
  echo "A3 ImportSmoke | INFO | - | Import smoke test passed (see A_import_smoke.log) | -" >> "${OUT_FILE}"
fi

echo >> "${OUT_FILE}"
echo "=== AST import graph checks (A1/A2/A4/A5) ===" >> "${OUT_FILE}"
if python3 "${SCRIPT_DIR}/import_audit.py" --repo-root "${REPO_ROOT}" >> "${OUT_FILE}"; then
  :
else
  hard_fail=1
fi

exit "${hard_fail}"

