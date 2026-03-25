#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

REPORT_DIR="${1:?usage: audit_L_windows_packaging.sh <report_dir> <summary_file>}"
SUMMARY_FILE="${2:?usage: audit_L_windows_packaging.sh <report_dir> <summary_file>}"
OUT_FILE="${REPORT_DIR}/L_windows_packaging.txt"

timestamp_utc="$(date -u +"%Y%m%dT%H%M%SZ")"

get_int() {
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
  local status likely_break case_conflicts scanned_py
  status="$1"
  likely_break="$2"
  case_conflicts="$3"
  scanned_py="$4"
  echo "Audit L: ${status} (likely_break=${likely_break} case_conflicts=${case_conflicts} scanned_py_files=${scanned_py})" >> "${SUMMARY_FILE}"
}

write_fallback_report() {
  local finding status counts_line
  finding="$1"
  status="$2"
  counts_line="$3"
  {
    echo "Kindred Audit L: Windows packaging readiness (report-only)"
    echo "Timestamp (UTC): ${timestamp_utc}"
    echo "Repo root: ${REPO_ROOT}"
    echo
    echo "${counts_line}"
    echo
    echo "${finding}"
  } > "${OUT_FILE}"
  append_summary_line "${status}" 0 0 0
}

if ! command -v python3 >/dev/null 2>&1; then
  write_fallback_report \
    "L0 PythonMissing | SKIP | - | python3 not available; Audit L cannot run | Install python3" \
    "SKIP" \
    "WINDOWS_PACKAGING_COUNTS|scanned_py_files=0|file_usage_total=0|file_usage_unguarded=0|posix_literal_hits=0|dynamic_import_hits=0|fs_resource_hits=0|case_conflicts_groups=0|likely_break_total=0"
  exit 0
fi

set +e
(cd -- "${REPO_ROOT}" && PYTHONDONTWRITEBYTECODE=1 python3 "${SCRIPT_DIR}/windows_packaging_audit.py" \
  --repo-root "${REPO_ROOT}" \
  --report-dir "${REPORT_DIR}" \
  --output "${OUT_FILE}") \
  >/dev/null 2>&1
rc=$?
set -e

if [[ "${rc}" -ne 0 || ! -f "${OUT_FILE}" ]]; then
  write_fallback_report \
    "L0 AuditError | WARN | - | windows_packaging_audit.py failed (exit ${rc}) | Fix audit tooling/environment" \
    "WARN" \
    "WINDOWS_PACKAGING_COUNTS|scanned_py_files=0|file_usage_total=0|file_usage_unguarded=0|posix_literal_hits=0|dynamic_import_hits=0|fs_resource_hits=0|case_conflicts_groups=0|likely_break_total=0"
  exit 0
fi

counts_line="$(grep -m1 '^WINDOWS_PACKAGING_COUNTS|' "${OUT_FILE}" || true)"
likely_break="$(get_int "${counts_line}" "likely_break_total")"
case_conflicts="$(get_int "${counts_line}" "case_conflicts_groups")"
scanned_py="$(get_int "${counts_line}" "scanned_py_files")"

status="PASS"
if [[ -z "${counts_line}" ]]; then
  status="WARN"
elif [[ "${likely_break}" -gt 0 || "${case_conflicts}" -gt 0 ]]; then
  status="WARN"
fi

append_summary_line "${status}" "${likely_break}" "${case_conflicts}" "${scanned_py}"
exit 0
