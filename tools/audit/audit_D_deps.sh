#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

REPORT_DIR="${1:?usage: audit_D_deps.sh <report_dir> <summary_file>}"
SUMMARY_FILE="${2:?usage: audit_D_deps.sh <report_dir> <summary_file>}"
OUT_FILE="${REPORT_DIR}/D_deps.txt"

timestamp_utc="$(date -u +"%Y%m%dT%H%M%SZ")"

get_count() {
  local counts_line key val
  counts_line="$1"
  key="$2"
  val="$(printf '%s\n' "${counts_line}" | sed -n "s/.*|${key}=\\([0-9][0-9]*\\).*/\\1/p")"
  if [[ -z "${val}" ]]; then
    val=0
  fi
  printf '%s' "${val}"
}

append_summary_line() {
  local status prod_undeclared test_undeclared tools_undeclared unused unguarded files_scanned
  status="$1"
  prod_undeclared="$2"
  test_undeclared="$3"
  tools_undeclared="$4"
  unused="$5"
  unguarded="$6"
  files_scanned="$7"
  echo "Audit D: ${status} (prod_undeclared=${prod_undeclared} test_undeclared=${test_undeclared} tools_undeclared=${tools_undeclared} unused_declared=${unused} unguarded_optional=${unguarded} files_scanned=${files_scanned})" >> "${SUMMARY_FILE}"
}

if ! command -v python3 >/dev/null 2>&1; then
  {
    echo "Kindred Audit D: Deps vs imports (report-only)"
    echo "Timestamp (UTC): ${timestamp_utc}"
    echo "Repo root: ${REPO_ROOT}"
    echo
    echo "D0 PythonMissing | SKIP | - | python3 not available; Audit D cannot run | Install python3"
    echo
    echo "COUNTS|files_scanned=0|parse_failures=0|third_party_modules=0|third_party_import_sites=0|undeclared_imports=0|undeclared_imports_prod=0|undeclared_imports_test=0|undeclared_imports_tools=0|unused_declared_deps=0|unguarded_optional_imports=0|pyproject_parse_warnings=1|stdlib_fallback=0"
  } > "${OUT_FILE}"
  append_summary_line "SKIP (python3 missing)" 0 0 0 0 0 0
  exit 0
fi

set +e
python3 "${SCRIPT_DIR}/deps_audit.py" --root "${REPO_ROOT}" --out "${OUT_FILE}" >/dev/null 2>&1
rc=$?
set -e

if [[ "${rc}" -ne 0 || ! -f "${OUT_FILE}" ]]; then
  {
    echo "Kindred Audit D: Deps vs imports (report-only)"
    echo "Timestamp (UTC): ${timestamp_utc}"
    echo "Repo root: ${REPO_ROOT}"
    echo
    echo "D0 AuditError | WARN | - | deps_audit.py failed (exit ${rc}) | Fix audit tooling"
    echo
    echo "COUNTS|files_scanned=0|parse_failures=0|third_party_modules=0|third_party_import_sites=0|undeclared_imports=0|undeclared_imports_prod=0|undeclared_imports_test=0|undeclared_imports_tools=0|unused_declared_deps=0|unguarded_optional_imports=0|pyproject_parse_warnings=1|stdlib_fallback=0"
  } > "${OUT_FILE}"
  append_summary_line "WARN (audit script error)" 0 0 0 0 0 0
  exit 0
fi

counts_line="$(grep -m1 '^COUNTS|' "${OUT_FILE}" || true)"
files_scanned="$(get_count "${counts_line}" "files_scanned")"
prod_undeclared="$(get_count "${counts_line}" "undeclared_imports_prod")"
test_undeclared="$(get_count "${counts_line}" "undeclared_imports_test")"
tools_undeclared="$(get_count "${counts_line}" "undeclared_imports_tools")"
unused="$(get_count "${counts_line}" "unused_declared_deps")"
unguarded="$(get_count "${counts_line}" "unguarded_optional_imports")"

status="PASS"
if [[ "${prod_undeclared}" -ne 0 ]]; then
  status="WARN"
fi

append_summary_line "${status}" "${prod_undeclared}" "${test_undeclared}" "${tools_undeclared}" "${unused}" "${unguarded}" "${files_scanned}"
exit 0
