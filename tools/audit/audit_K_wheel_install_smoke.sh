#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

REPORT_DIR="${1:?usage: audit_K_wheel_install_smoke.sh <report_dir> <summary_file>}"
SUMMARY_FILE="${2:?usage: audit_K_wheel_install_smoke.sh <report_dir> <summary_file>}"
OUT_FILE="${REPORT_DIR}/K_wheel_install_smoke.txt"

timestamp_utc="$(date -u +"%Y%m%dT%H%M%SZ")"
timeout_seconds="${AUDIT_K_TIMEOUT_SECONDS:-240.0}"

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
  local status build_ok venv_ok install_ok failures timeouts case_conflicts resource_ok
  status="$1"
  build_ok="$2"
  venv_ok="$3"
  install_ok="$4"
  failures="$5"
  timeouts="$6"
  case_conflicts="$7"
  resource_ok="$8"
  echo "Audit K: ${status} (build_ok=${build_ok} venv_ok=${venv_ok} install_ok=${install_ok} smoke_failures=${failures} timeouts=${timeouts} case_conflicts=${case_conflicts} known_resource_on_disk_ok=${resource_ok})" >> "${SUMMARY_FILE}"
}

write_fallback_report() {
  local finding status counts_line
  finding="$1"
  status="$2"
  counts_line="$3"
  {
    echo "Kindred Audit K: Wheel build + install + smoke checks (report-only)"
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
    "K0 PythonMissing | SKIP | - | python3 not available; Audit K cannot run | Install python3" \
    "SKIP" \
    "WHEEL_INSTALL_SMOKE_COUNTS|build_attempted=0|build_ok=0|venv_ok=0|install_ok=0|smoke_checks=0|smoke_failures=0|timeouts=0|case_conflicts=0|known_resource_on_disk_ok=0|status=SKIP"
  exit 0
fi

set +e
PYTHONDONTWRITEBYTECODE=1 python3 "${SCRIPT_DIR}/wheel_install_smoke_audit.py" \
  --repo-root "${REPO_ROOT}" \
  --report-dir "${REPORT_DIR}" \
  --output "${OUT_FILE}" \
  --timeout-seconds "${timeout_seconds}" \
  >/dev/null 2>&1
rc=$?
set -e

if [[ "${rc}" -ne 0 || ! -f "${OUT_FILE}" ]]; then
  write_fallback_report \
    "K0 AuditError | WARN | - | wheel_install_smoke_audit.py failed (exit ${rc}) | Fix audit tooling/environment" \
    "WARN" \
    "WHEEL_INSTALL_SMOKE_COUNTS|build_attempted=0|build_ok=0|venv_ok=0|install_ok=0|smoke_checks=0|smoke_failures=0|timeouts=0|case_conflicts=0|known_resource_on_disk_ok=0|status=WARN"
  exit 0
fi

counts_line="$(grep -m1 '^WHEEL_INSTALL_SMOKE_COUNTS|' "${OUT_FILE}" || true)"
status="$(get_status "${counts_line}")"
build_ok="$(get_int "${counts_line}" "build_ok")"
venv_ok="$(get_int "${counts_line}" "venv_ok")"
install_ok="$(get_int "${counts_line}" "install_ok")"
failures="$(get_int "${counts_line}" "smoke_failures")"
timeouts="$(get_int "${counts_line}" "timeouts")"
case_conflicts="$(get_int "${counts_line}" "case_conflicts")"
resource_ok="$(get_int "${counts_line}" "known_resource_on_disk_ok")"

append_summary_line "${status}" "${build_ok}" "${venv_ok}" "${install_ok}" "${failures}" "${timeouts}" "${case_conflicts}" "${resource_ok}"
exit 0
