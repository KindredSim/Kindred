#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

run_all_out="$(mktemp -t kindred_audit_run_all.XXXXXX)"
cleanup() {
  rm -f "${run_all_out}"
}
trap cleanup EXIT

set +e
bash "${SCRIPT_DIR}/run_all.sh" "$@" 2>&1 | tee "${run_all_out}"
run_all_rc="${PIPESTATUS[0]}"
set -e

report_dir="$(
  grep -aF 'AUDIT_REPORT_DIR|' "${run_all_out}" | tail -n 1 | sed 's/^AUDIT_REPORT_DIR|//'
)"

if [[ -z "${report_dir}" ]]; then
  echo "STRICT GATE FAILED: could not determine report dir from run_all output."
  echo "Expected a line like: AUDIT_REPORT_DIR|/abs/path/to/_audit_reports/<timestamp>"
  exit 2
fi

if [[ "${report_dir}" != /* ]]; then
  report_dir="${REPO_ROOT}/${report_dir}"
fi

summary="${report_dir}/SUMMARY.txt"
if [[ ! -f "${summary}" ]]; then
  echo "STRICT GATE FAILED: SUMMARY.txt missing."
  echo "Report dir: ${report_dir}"
  exit 2
fi

offending_lines="$(grep -nE '^Audit .*: (WARN|FAIL|TIMEOUT)($|[ (])' "${summary}" || true)"
if [[ -n "${offending_lines}" ]]; then
  echo "STRICT GATE FAILED: SUMMARY contains WARN/FAIL/TIMEOUT."
  echo "Report dir: ${report_dir}"
  echo "Offending lines:"
  printf '%s\n' "${offending_lines}" | head -n 50 | sed 's/^/  /'
  exit 2
fi

if [[ "${run_all_rc}" != "0" ]]; then
  echo "STRICT GATE FAILED: run_all.sh exited non-zero (${run_all_rc})."
  echo "Report dir: ${report_dir}"
  exit "${run_all_rc}"
fi

echo "STRICT GATE PASS: no WARN/FAIL/TIMEOUT in SUMMARY.txt."
echo "Report dir: ${report_dir}"
