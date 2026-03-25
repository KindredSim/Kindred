#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

REPORT_DIR="${1:?usage: audit_G_gui_audit.sh <report_dir> <summary_file>}"
SUMMARY_FILE="${2:?usage: audit_G_gui_audit.sh <report_dir> <summary_file>}"
OUT_FILE="${REPORT_DIR}/G_gui_audit.txt"

timestamp_utc="$(date -u +"%Y%m%dT%H%M%SZ")"
timeout_seconds="${AUDIT_G_TIMEOUT_SECONDS:-90}"

get_field() {
  local line key val
  line="$1"
  key="$2"
  val="$(printf '%s\n' "${line}" | sed -n "s/.*|${key}=\\([^|]*\\).*/\\1/p")"
  printf '%s' "${val}"
}

write_fallback_report() {
  local message exit_code timeout_flag seconds
  message="$1"
  exit_code="$2"
  timeout_flag="$3"
  seconds="$4"
  {
    echo "Kindred Audit G: GUI wiring/plumbing audit (report-only)"
    echo "Timestamp (UTC): ${timestamp_utc}"
    echo "Repo root: ${REPO_ROOT}"
    echo "Report dir: ${REPORT_DIR}"
    echo
    echo "G0 AuditError | WARN | - | ${message} | Fix audit tooling/environment"
    echo
    echo "GUI_AUDIT_COUNTS|exit=${exit_code}|timeout=${timeout_flag}|seconds=${seconds}"
  } > "${OUT_FILE}"
}

append_summary_line() {
  local status exit_code timeout_flag seconds
  status="$1"
  exit_code="$2"
  timeout_flag="$3"
  seconds="$4"
  echo "Audit G: ${status} (exit=${exit_code} timeout=${timeout_flag} seconds=${seconds})" >> "${SUMMARY_FILE}"
}

if ! command -v python3 >/dev/null 2>&1; then
  write_fallback_report "python3 not available; Audit G cannot run" 127 0 "0.000"
  append_summary_line "WARN" 127 0 "0.000"
  exit 0
fi

set +e
(cd -- "${REPO_ROOT}" && python3 "${SCRIPT_DIR}/gui_audit_audit.py" \
  --root "${REPO_ROOT}" \
  --report-dir "${REPORT_DIR}" \
  --out "${OUT_FILE}" \
  --timeout-seconds "${timeout_seconds}" \
  --max-stream-bytes 16384 \
  --max-tail-lines 80 \
  --max-tail-chars 8000) \
  >/dev/null 2>&1
rc=$?
set -e

if [[ "${rc}" -ne 0 || ! -f "${OUT_FILE}" ]]; then
  write_fallback_report "gui_audit_audit.py failed (exit ${rc})" "${rc}" 0 "0.000"
  append_summary_line "WARN" "${rc}" 0 "0.000"
  exit 0
fi

counts_line="$(grep -m1 '^GUI_AUDIT_COUNTS|' "${OUT_FILE}" || true)"
exit_code="$(get_field "${counts_line}" "exit")"
timeout_flag="$(get_field "${counts_line}" "timeout")"
seconds="$(get_field "${counts_line}" "seconds")"

if [[ -z "${exit_code}" ]]; then exit_code="${rc}"; fi
if [[ -z "${timeout_flag}" ]]; then timeout_flag=0; fi
if [[ -z "${seconds}" ]]; then seconds="0.000"; fi

status="PASS"
if [[ "${timeout_flag}" != "0" || "${exit_code}" != "0" ]]; then
  status="WARN"
fi

append_summary_line "${status}" "${exit_code}" "${timeout_flag}" "${seconds}"
exit 0
