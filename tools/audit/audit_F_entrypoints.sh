#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

REPORT_DIR="${1:?usage: audit_F_entrypoints.sh <report_dir> <summary_file>}"
SUMMARY_FILE="${2:?usage: audit_F_entrypoints.sh <report_dir> <summary_file>}"
OUT_FILE="${REPORT_DIR}/F_entrypoints.txt"

timestamp_utc="$(date -u +"%Y%m%dT%H%M%SZ")"
ALLOWLIST_FILE="${REPO_ROOT}/tools/audit/entrypoints_allowlist.txt"

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
  local status commands failures timeouts
  status="$1"
  commands="$2"
  failures="$3"
  timeouts="$4"
  echo "Audit F: ${status} (commands=${commands} failures=${failures} timeouts=${timeouts})" >> "${SUMMARY_FILE}"
}

if ! command -v python3 >/dev/null 2>&1; then
  {
    echo "Kindred Audit F: Runtime entrypoint sweep (report-only)"
    echo "Timestamp (UTC): ${timestamp_utc}"
    echo "Repo root: ${REPO_ROOT}"
    echo
    echo "F0 PythonMissing | WARN | - | python3 not available; Audit F cannot run | Install python3"
    echo
    echo "ENTRYPOINT_SWEEP_COUNTS|commands=0|failures=0|timeouts=0"
  } > "${OUT_FILE}"
  append_summary_line "WARN (python3 missing)" 0 0 0
  exit 0
fi

set +e
(cd -- "${REPO_ROOT}" && python3 "${SCRIPT_DIR}/entrypoint_sweep_audit.py" \
  --root "${REPO_ROOT}" \
  --out "${OUT_FILE}" \
  --allowlist "${ALLOWLIST_FILE}" \
  --timeout-seconds 10.0 \
  --max-stream-bytes 16384 \
  --max-tail-lines 25 \
  --max-tail-chars 4000) \
  >/dev/null 2>&1
rc=$?
set -e

if [[ "${rc}" -ne 0 || ! -f "${OUT_FILE}" ]]; then
  {
    echo "Kindred Audit F: Runtime entrypoint sweep (report-only)"
    echo "Timestamp (UTC): ${timestamp_utc}"
    echo "Repo root: ${REPO_ROOT}"
    echo
    echo "F0 AuditError | WARN | - | entrypoint_sweep_audit.py failed (exit ${rc}) | Fix audit tooling"
    echo
    echo "ENTRYPOINT_SWEEP_COUNTS|commands=0|failures=0|timeouts=0"
  } > "${OUT_FILE}"
  append_summary_line "WARN (audit script error)" 0 0 0
  exit 0
fi

counts_line="$(grep -m1 '^ENTRYPOINT_SWEEP_COUNTS|' "${OUT_FILE}" || true)"
commands="$(get_count "${counts_line}" "commands")"
failures="$(get_count "${counts_line}" "failures")"
timeouts="$(get_count "${counts_line}" "timeouts")"

status="PASS"
if [[ "${failures}" -ne 0 || "${timeouts}" -ne 0 ]]; then
  status="WARN"
fi

append_summary_line "${status}" "${commands}" "${failures}" "${timeouts}"
exit 0
