#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

REPORT_DIR="${1:?usage: audit_J_wheel.sh <report_dir> <summary_file>}"
SUMMARY_FILE="${2:?usage: audit_J_wheel.sh <report_dir> <summary_file>}"
OUT_FILE="${REPORT_DIR}/J_wheel.txt"

timestamp_utc="$(date -u +"%Y%m%dT%H%M%SZ")"
timeout_seconds="${AUDIT_J_WHEEL_TIMEOUT_SECONDS:-180.0}"

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

get_status() {
  local counts_line val
  counts_line="$1"
  val="$(printf '%s\n' "${counts_line}" | sed -n "s/.*|status=\\([A-Z][A-Z]*\\).*/\\1/p")"
  if [[ -z "${val}" ]]; then
    val="WARN"
  fi
  printf '%s' "${val}"
}

append_summary_line() {
  local status missing case_conflicts reserved invalid trailing pyc build_ok
  status="$1"
  missing="$2"
  case_conflicts="$3"
  reserved="$4"
  invalid="$5"
  trailing="$6"
  pyc="$7"
  build_ok="$8"
  echo "Audit J: ${status} (missing=${missing} case_conflicts=${case_conflicts} reserved=${reserved} invalid=${invalid} trailing=${trailing} pyc=${pyc} build_ok=${build_ok})" >> "${SUMMARY_FILE}"
}

write_fallback_report() {
  local finding status counts_line
  finding="$1"
  status="$2"
  counts_line="$3"
  {
    echo "Kindred Audit J: Wheel contents + Windows filename/case hygiene (report-only)"
    echo "Timestamp (UTC): ${timestamp_utc}"
    echo "Repo root: ${REPO_ROOT}"
    echo
    echo "${finding}"
    echo
    echo "${counts_line}"
  } > "${OUT_FILE}"
  append_summary_line "${status}" 0 0 0 0 0 0 0
}

if ! command -v python3 >/dev/null 2>&1; then
  write_fallback_report \
    "J0 PythonMissing | SKIP | - | python3 not available; Audit J cannot run | Install python3" \
    "SKIP" \
    "WHEEL_AUDIT_COUNTS|build_attempted=0|build_ok=0|wheel_files=0|missing_resources=0|case_conflicts=0|reserved_names=0|invalid_names=0|trailing_dot_space=0|long_paths=0|pyc_files=0|status=SKIP"
  exit 0
fi

set +e
PYTHONDONTWRITEBYTECODE=1 python3 "${SCRIPT_DIR}/wheel_audit.py" \
  --repo-root "${REPO_ROOT}" \
  --report-dir "${REPORT_DIR}" \
  --output "${OUT_FILE}" \
  --timeout-seconds "${timeout_seconds}" \
  --build-tail-lines 60 \
  >/dev/null 2>&1
rc=$?
set -e

if [[ "${rc}" -ne 0 || ! -f "${OUT_FILE}" ]]; then
  write_fallback_report \
    "J0 AuditError | WARN | - | wheel_audit.py failed (exit ${rc}) | Fix audit tooling/environment" \
    "WARN" \
    "WHEEL_AUDIT_COUNTS|build_attempted=0|build_ok=0|wheel_files=0|missing_resources=0|case_conflicts=0|reserved_names=0|invalid_names=0|trailing_dot_space=0|long_paths=0|pyc_files=0|status=WARN"
  exit 0
fi

counts_line="$(grep -m1 '^WHEEL_AUDIT_COUNTS|' "${OUT_FILE}" || true)"
status="$(get_status "${counts_line}")"
build_ok="$(get_int "${counts_line}" "build_ok")"
missing="$(get_int "${counts_line}" "missing_resources")"
case_conflicts="$(get_int "${counts_line}" "case_conflicts")"
reserved="$(get_int "${counts_line}" "reserved_names")"
invalid="$(get_int "${counts_line}" "invalid_names")"
trailing="$(get_int "${counts_line}" "trailing_dot_space")"
pyc="$(get_int "${counts_line}" "pyc_files")"

append_summary_line "${status}" "${missing}" "${case_conflicts}" "${reserved}" "${invalid}" "${trailing}" "${pyc}" "${build_ok}"
exit 0
