#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

timestamp_utc="$(date -u +"%Y%m%dT%H%M%SZ")"
report_root="${REPO_ROOT}/_audit_reports"
report_dir="${report_root}/${timestamp_utc}"

mkdir -p "${report_dir}"

summary="${report_dir}/SUMMARY.txt"
{
  echo "Kindred Audit Summary"
  echo "Timestamp (UTC): ${timestamp_utc}"
  echo "Repo root: ${REPO_ROOT}"
  echo "Report dir: ${report_dir}"
  echo
} > "${summary}"

hard_fail=0

echo "Running Audit A (imports/layering/cycles/orphans)..." >> "${summary}"
if bash "${SCRIPT_DIR}/audit_A_imports.sh" "${report_dir}"; then
  echo "Audit A: PASS" >> "${summary}"
else
  echo "Audit A: FAIL (hard failures present)" >> "${summary}"
  hard_fail=1
fi
echo "Audit A report: A_imports_and_cycles.txt" >> "${summary}"

echo >> "${summary}"
echo "Running Audit B (ruff lint sweep)..." >> "${summary}"
if bash "${SCRIPT_DIR}/audit_B_ruff.sh" "${report_dir}" "${summary}"; then
  :
else
  echo "Audit B: WARN (audit script error)" >> "${summary}"
  echo "Audit B report: B_ruff.txt" >> "${summary}"
fi

echo >> "${summary}"
echo "Running Audit C (dead-code candidates)..." >> "${summary}"
if bash "${SCRIPT_DIR}/audit_C_deadcode.sh" "${report_dir}" "${summary}"; then
  :
else
  echo "Audit C: WARN (audit script error)" >> "${summary}"
  echo "Audit C report: C_deadcode.txt" >> "${summary}"
fi

echo >> "${summary}"
echo "Running Audit D (deps vs imports)..." >> "${summary}"
if bash "${SCRIPT_DIR}/audit_D_deps.sh" "${report_dir}" "${summary}"; then
  :
else
  echo "Audit D: WARN (audit script error)" >> "${summary}"
  echo "Audit D report: D_deps.txt" >> "${summary}"
fi

echo >> "${summary}"
echo "Running Audit E (import sweep)..." >> "${summary}"
if bash "${SCRIPT_DIR}/audit_E_import_sweep.sh" "${report_dir}" "${summary}"; then
  :
else
  echo "Audit E: WARN (audit script error)" >> "${summary}"
  echo "Audit E report: E_import_sweep.txt" >> "${summary}"
fi

echo >> "${summary}"
echo "Running Audit F (runtime entrypoint sweep)..." >> "${summary}"
if bash "${SCRIPT_DIR}/audit_F_entrypoints.sh" "${report_dir}" "${summary}"; then
  :
else
  echo "Audit F: WARN (audit script error)" >> "${summary}"
  echo "Audit F report: F_entrypoints.txt" >> "${summary}"
fi

echo >> "${summary}"
echo "Running Audit G (GUI wiring/plumbing audit)..." >> "${summary}"
if bash "${SCRIPT_DIR}/audit_G_gui_audit.sh" "${report_dir}" "${summary}"; then
  :
else
  echo "Audit G: WARN (audit script error)" >> "${summary}"
  echo "Audit G report: G_gui_audit.txt" >> "${summary}"
fi

echo >> "${summary}"
echo "Running Audit H (import-time Python warnings sweep)..." >> "${summary}"
if bash "${SCRIPT_DIR}/audit_H_warnings.sh" "${report_dir}" "${summary}"; then
  :
else
  echo "Audit H: WARN (audit script error)" >> "${summary}"
  echo "Audit H report: H_warnings.txt" >> "${summary}"
fi

echo >> "${summary}"
echo "Running Audit I (resources + Windows packaging compatibility)..." >> "${summary}"
if bash "${SCRIPT_DIR}/audit_I_resources.sh" "${report_dir}" "${summary}"; then
  :
else
  echo "Audit I: WARN (audit script error)" >> "${summary}"
  echo "Audit I report: I_resources.txt" >> "${summary}"
fi

echo >> "${summary}"
echo "Running Audit J (wheel contents + Windows filename/case hygiene)..." >> "${summary}"
if bash "${SCRIPT_DIR}/audit_J_wheel.sh" "${report_dir}" "${summary}"; then
  :
else
  echo "Audit J: WARN (audit script error)" >> "${summary}"
  echo "Audit J report: J_wheel.txt" >> "${summary}"
fi

echo >> "${summary}"
echo "Running Audit K (wheel build + install + smoke checks)..." >> "${summary}"
if bash "${SCRIPT_DIR}/audit_K_wheel_install_smoke.sh" "${report_dir}" "${summary}"; then
  :
else
  echo "Audit K: WARN (audit script error)" >> "${summary}"
  echo "Audit K report: K_wheel_install_smoke.txt" >> "${summary}"
fi

echo >> "${summary}"
echo "Running Audit L (Windows packaging readiness)..." >> "${summary}"
if bash "${SCRIPT_DIR}/audit_L_windows_packaging.sh" "${report_dir}" "${summary}"; then
  :
else
  echo "Audit L: WARN (audit script error)" >> "${summary}"
  echo "Audit L report: L_windows_packaging.txt" >> "${summary}"
fi

echo "AUDIT_REPORT_DIR|${report_dir}"

exit "${hard_fail}"
