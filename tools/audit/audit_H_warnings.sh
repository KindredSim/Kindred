#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

REPORT_DIR="${1:?usage: audit_H_warnings.sh <report_dir> <summary_file>}"
SUMMARY_FILE="${2:?usage: audit_H_warnings.sh <report_dir> <summary_file>}"
OUT_FILE="${REPORT_DIR}/H_warnings.txt"

timestamp_utc="$(date -u +"%Y%m%dT%H%M%SZ")"
timeout_seconds="${AUDIT_H_TIMEOUT_SECONDS:-10.0}"

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
  local status note modules modules_with_warnings warnings unique failures timeouts
  status="$1"
  note="$2"
  modules="$3"
  modules_with_warnings="$4"
  warnings="$5"
  unique="$6"
  failures="$7"
  timeouts="$8"
  if [[ -n "${note}" ]]; then
    echo "Audit H: ${status} (reason=${note} modules=${modules} modules_with_warnings=${modules_with_warnings} warnings=${warnings} unique_warning_lines=${unique} failures=${failures} timeouts=${timeouts})" >> "${SUMMARY_FILE}"
  else
    echo "Audit H: ${status} (modules=${modules} modules_with_warnings=${modules_with_warnings} warnings=${warnings} unique_warning_lines=${unique} failures=${failures} timeouts=${timeouts})" >> "${SUMMARY_FILE}"
  fi
}

write_fallback_report() {
  local finding status note counts_line
  finding="$1"
  status="$2"
  note="$3"
  counts_line="$4"
  {
    echo "Kindred Audit H: Import-time Python warnings sweep (report-only)"
    echo "Timestamp (UTC): ${timestamp_utc}"
    echo "Repo root: ${REPO_ROOT}"
    echo
    echo "${finding}"
    echo
    echo "${counts_line}"
  } > "${OUT_FILE}"
  append_summary_line "${status}" "${note}" 0 0 0 0 0 0
}

if ! command -v python3 >/dev/null 2>&1; then
  write_fallback_report \
    "H0 PythonMissing | SKIP | - | python3 not available; Audit H cannot run | Install python3" \
    "SKIP" \
    "python3_missing" \
    "WARNINGS_SWEEP_COUNTS|modules=0|modules_with_warnings=0|warnings=0|unique_warning_lines=0|failures=0|timeouts=0"
  exit 0
fi

set +e
(
  cd "${REPO_ROOT}" &&
  python3 -m tools.audit.warnings_sweep_audit \
    --root "${REPO_ROOT}" \
    --out "${OUT_FILE}" \
    --timeout-seconds "${timeout_seconds}" \
    --max-tail-lines 25 \
    --max-tail-chars 4000
) >/dev/null 2>&1
rc=$?
set -e

if [[ "${rc}" -ne 0 || ! -f "${OUT_FILE}" ]]; then
  write_fallback_report \
    "H0 AuditError | WARN | - | warnings_sweep_audit.py failed (exit ${rc}) | Fix audit tooling/environment" \
    "WARN" \
    "audit_script_error" \
    "WARNINGS_SWEEP_COUNTS|modules=0|modules_with_warnings=0|warnings=0|unique_warning_lines=0|failures=0|timeouts=0"
  exit 0
fi

counts_line="$(grep -m1 '^WARNINGS_SWEEP_COUNTS|' "${OUT_FILE}" || true)"
modules="$(get_count "${counts_line}" "modules")"
modules_with_warnings="$(get_count "${counts_line}" "modules_with_warnings")"
warnings="$(get_count "${counts_line}" "warnings")"
unique="$(get_count "${counts_line}" "unique_warning_lines")"
failures="$(get_count "${counts_line}" "failures")"
timeouts="$(get_count "${counts_line}" "timeouts")"

status="PASS"
if [[ "${warnings}" -ne 0 || "${failures}" -ne 0 || "${timeouts}" -ne 0 ]]; then
  status="WARN"
fi

append_summary_line "${status}" "" "${modules}" "${modules_with_warnings}" "${warnings}" "${unique}" "${failures}" "${timeouts}"
exit 0
