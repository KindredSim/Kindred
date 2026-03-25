#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

strict_out="$(mktemp -t kindred_audit_ci_strict.XXXXXX)"
pytest_out="$(mktemp -t kindred_audit_ci_pytest.XXXXXX)"
cleanup() {
  rm -f "${strict_out}" "${pytest_out}"
}
trap cleanup EXIT

set +e
bash "${SCRIPT_DIR}/run_strict.sh" 2>&1 | tee "${strict_out}"
strict_rc="${PIPESTATUS[0]}"
set -e

report_dir="$(
  grep -aF 'AUDIT_REPORT_DIR|' "${strict_out}" | tail -n 1 | sed 's/^AUDIT_REPORT_DIR|//'
)"

if [[ -z "${report_dir}" ]]; then
  echo "CI RUNNER FAILED: could not determine report dir from strict audit output."
  echo "Expected a line like: AUDIT_REPORT_DIR|/abs/path/to/_audit_reports/<timestamp>"
  exit 2
fi

if [[ "${report_dir}" != /* ]]; then
  report_dir="${REPO_ROOT}/${report_dir}"
fi

summary="${report_dir}/SUMMARY.txt"
if [[ ! -f "${summary}" ]]; then
  echo "CI RUNNER FAILED: SUMMARY.txt missing."
  echo "Report dir: ${report_dir}"
  exit 2
fi

ci_pytest="${report_dir}/CI_pytest.txt"

if [[ "${strict_rc}" != "0" ]]; then
  {
    echo "Kindred CI pytest result"
    echo "command: pytest -q"
    echo "status: SKIPPED"
    echo "reason: strict_failed"
    echo "strict_exit: ${strict_rc}"
  } > "${ci_pytest}"
  echo "CI (pytest): SKIP (reason=strict_failed exit=${strict_rc})" >> "${summary}"
  exit "${strict_rc}"
fi

start_s="$(date +%s)"
set +e
(cd -- "${REPO_ROOT}" && pytest -q) 2>&1 | tee "${pytest_out}"
pytest_rc="${PIPESTATUS[0]}"
set -e
end_s="$(date +%s)"
duration_s="$((end_s - start_s))"

if [[ "${pytest_rc}" == "0" ]]; then
  {
    echo "Kindred CI pytest result"
    echo "command: pytest -q"
    echo "exit: ${pytest_rc}"
    echo "duration_s: ${duration_s}"
    echo
    echo "--- tail (last 30 lines) ---"
    tail -n 30 "${pytest_out}" || true
  } > "${ci_pytest}"
  echo "CI (pytest): PASS (exit=0 duration_s=${duration_s})" >> "${summary}"
  exit 0
fi

{
  echo "Kindred CI pytest result"
  echo "command: pytest -q"
  echo "exit: ${pytest_rc}"
  echo "duration_s: ${duration_s}"
  echo
  echo "RESULT: FAIL"
  echo
  echo "--- tail (last 200 lines) ---"
  tail -n 200 "${pytest_out}" || true
} > "${ci_pytest}"
echo "CI (pytest): FAIL (exit=${pytest_rc} duration_s=${duration_s})" >> "${summary}"
exit "${pytest_rc}"
