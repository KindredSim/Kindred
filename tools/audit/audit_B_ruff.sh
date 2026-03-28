#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

REPORT_DIR="${1:?usage: audit_B_ruff.sh <report_dir> <summary_file>}"
SUMMARY_FILE="${2:?usage: audit_B_ruff.sh <report_dir> <summary_file>}"
OUT_FILE="${REPORT_DIR}/B_ruff.txt"

timestamp_utc="$(date -u +"%Y%m%dT%H%M%SZ")"

{
  echo "Kindred Audit B: Ruff lint sweep"
  echo "Timestamp (UTC): ${timestamp_utc}"
  echo "Repo root: ${REPO_ROOT}"
  echo
} > "${OUT_FILE}"

if ! command -v ruff >/dev/null 2>&1; then
  {
    echo "B1 RuffMissing | SKIP | - | ruff not installed; skipping lint sweep | Install ruff or use project dev env"
    echo
    echo "Command (if available):"
    echo "  ruff check . --no-cache --force-exclude --extend-exclude _audit_reports"
  } >> "${OUT_FILE}"
  {
    echo "Audit B: SKIP (ruff not installed)"
    echo "Audit B report: B_ruff.txt"
  } >> "${SUMMARY_FILE}"
  exit 0
fi

ruff_version="$(ruff --version 2>/dev/null || echo "ruff (version unknown)")"

RUFF_CMD=(
  ruff
  check
  .
  --no-cache
  --force-exclude
  --extend-exclude
  _audit_reports
)

{
  echo "B0 RuffVersion | INFO | - | ${ruff_version} | -"
  echo
  printf "Command:"
  for arg in "${RUFF_CMD[@]}"; do
    printf " %q" "${arg}"
  done
  printf "\n\n"
  echo "Output:"
  echo
} >> "${OUT_FILE}"

set +e
(cd -- "${REPO_ROOT}" && "${RUFF_CMD[@]}") >> "${OUT_FILE}" 2>&1
ruff_rc=$?
set -e

if [[ "${ruff_rc}" -eq 0 ]]; then
  echo "B2 RuffViolations | INFO | - | ruff check clean (exit 0) | -" >> "${OUT_FILE}"
  {
    echo "Audit B: PASS"
    echo "Audit B report: B_ruff.txt"
  } >> "${SUMMARY_FILE}"
else
  echo "B2 RuffViolations | WARN | - | ruff check reported violations (exit ${ruff_rc}); see output above | Fix lint issues or update ruff config intentionally" >> "${OUT_FILE}"
  {
    echo "Audit B: WARN (ruff violations)"
    echo "Audit B report: B_ruff.txt"
  } >> "${SUMMARY_FILE}"
fi

exit 0
