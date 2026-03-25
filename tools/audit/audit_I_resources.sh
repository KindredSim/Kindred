#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

REPORT_DIR="${1:?usage: audit_I_resources.sh <report_dir> <summary_file>}"
SUMMARY_FILE="${2:?usage: audit_I_resources.sh <report_dir> <summary_file>}"
OUT_FILE="${REPORT_DIR}/I_resources.txt"

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
  local status note missing case_conflicts packaging_risks posix_path_risks qt_refs scanned_py scanned_assets
  status="$1"
  note="$2"
  missing="$3"
  case_conflicts="$4"
  packaging_risks="$5"
  posix_path_risks="$6"
  qt_refs="$7"
  scanned_py="$8"
  scanned_assets="$9"
  if [[ -n "${note}" ]]; then
    echo "Audit I: ${status} (reason=${note} missing=${missing} case_conflicts=${case_conflicts} packaging_risks=${packaging_risks} posix_path_risks=${posix_path_risks} qt_refs=${qt_refs} scanned_py=${scanned_py} scanned_assets=${scanned_assets})" >> "${SUMMARY_FILE}"
  else
    echo "Audit I: ${status} (missing=${missing} case_conflicts=${case_conflicts} packaging_risks=${packaging_risks} posix_path_risks=${posix_path_risks} qt_refs=${qt_refs} scanned_py=${scanned_py} scanned_assets=${scanned_assets})" >> "${SUMMARY_FILE}"
  fi
}

write_fallback_report() {
  local finding status note counts_line
  finding="$1"
  status="$2"
  note="$3"
  counts_line="$4"
  {
    echo "Kindred Audit I: Resources + Windows packaging compatibility (report-only)"
    echo "Timestamp (UTC): ${timestamp_utc}"
    echo "Repo root: ${REPO_ROOT}"
    echo
    echo "${finding}"
    echo
    echo "${counts_line}"
  } > "${OUT_FILE}"
  append_summary_line "${status}" "${note}" 0 0 0 0 0 0 0
}

if ! command -v python3 >/dev/null 2>&1; then
  write_fallback_report \
    "I0 PythonMissing | WARN | - | python3 not available; Audit I cannot run | Install python3" \
    "WARN" \
    "python3_missing" \
    "I_RESOURCES_COUNTS|missing=0|case_conflicts=0|packaging_risks=0|posix_path_risks=0|qt_refs=0|scanned_py=0|scanned_assets=0"
  exit 0
fi

set +e
(cd -- "${REPO_ROOT}" && PYTHONDONTWRITEBYTECODE=1 python3 "${SCRIPT_DIR}/resources_audit.py" \
  --root "${REPO_ROOT}" \
  --report-dir "${REPORT_DIR}") \
  >/dev/null 2>&1
rc=$?
set -e

if [[ "${rc}" -ne 0 || ! -f "${OUT_FILE}" ]]; then
  write_fallback_report \
    "I0 AuditError | WARN | - | resources_audit.py failed (exit ${rc}) | Fix audit tooling/environment" \
    "WARN" \
    "audit_script_error" \
    "I_RESOURCES_COUNTS|missing=0|case_conflicts=0|packaging_risks=0|posix_path_risks=0|qt_refs=0|scanned_py=0|scanned_assets=0"
  exit 0
fi

counts_line="$(grep -m1 '^I_RESOURCES_COUNTS|' "${OUT_FILE}" || true)"
missing="$(get_count "${counts_line}" "missing")"
case_conflicts="$(get_count "${counts_line}" "case_conflicts")"
packaging_risks="$(get_count "${counts_line}" "packaging_risks")"
posix_path_risks="$(get_count "${counts_line}" "posix_path_risks")"
qt_refs="$(get_count "${counts_line}" "qt_refs")"
scanned_py="$(get_count "${counts_line}" "scanned_py")"
scanned_assets="$(get_count "${counts_line}" "scanned_assets")"

status="PASS"
if [[ "${missing}" -ne 0 || "${case_conflicts}" -ne 0 ]]; then
  status="WARN"
fi

append_summary_line "${status}" "" "${missing}" "${case_conflicts}" "${packaging_risks}" "${posix_path_risks}" "${qt_refs}" "${scanned_py}" "${scanned_assets}"
exit 0
