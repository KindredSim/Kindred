#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

REPORT_DIR="${1:?usage: audit_C_deadcode.sh <report_dir> <summary_file>}"
SUMMARY_FILE="${2:?usage: audit_C_deadcode.sh <report_dir> <summary_file>}"
OUT_LEGACY="${REPORT_DIR}/C_deadcode.txt"
OUT_LEGACY_FILTERED="${REPORT_DIR}/C_deadcode_filtered.txt"
OUT_PKG="${REPORT_DIR}/C_deadcode_package.txt"
OUT_PKG_FILTERED="${REPORT_DIR}/C_deadcode_package_filtered.txt"
OUT_PKG_EVIDENCE="${REPORT_DIR}/C_deadcode_package_evidence.txt"
OUT_PKG_EVIDENCE_FILTERED="${REPORT_DIR}/C_deadcode_package_evidence_filtered.txt"
OUT_PKG_BUCKETS="${REPORT_DIR}/C_deadcode_package_evidence_buckets.txt"
OUT_PKG_BUCKETS_FILTERED="${REPORT_DIR}/C_deadcode_package_evidence_buckets_filtered.txt"
OUT_PKG_HIGH="${REPORT_DIR}/C_deadcode_package_high_confidence.txt"
OUT_PKG_HIGH_FILTERED="${REPORT_DIR}/C_deadcode_package_high_confidence_filtered.txt"
OUT_PKG_BUCKETS_V25="${REPORT_DIR}/C_deadcode_package_evidence_buckets_v2_5.txt"
OUT_PKG_BUCKETS_V25_FILTERED="${REPORT_DIR}/C_deadcode_package_evidence_buckets_v2_5_filtered.txt"
OUT_PKG_NO_EVIDENCE_SHORTLIST="${REPORT_DIR}/C_deadcode_package_no_evidence_shortlist.txt"
OUT_PKG_NO_EVIDENCE_SHORTLIST_FILTERED="${REPORT_DIR}/C_deadcode_package_no_evidence_shortlist_filtered.txt"
OUT_PKG_PROD_BUCKETS_V26="${REPORT_DIR}/C_deadcode_package_prod_evidence_buckets.txt"
OUT_PKG_PROD_BUCKETS_V26_FILTERED="${REPORT_DIR}/C_deadcode_package_prod_evidence_buckets_filtered.txt"
OUT_PKG_NO_PROD_SHORTLIST_V26="${REPORT_DIR}/C_deadcode_package_no_prod_evidence_shortlist.txt"
OUT_PKG_NO_PROD_SHORTLIST_V26_FILTERED="${REPORT_DIR}/C_deadcode_package_no_prod_evidence_shortlist_filtered.txt"
OUT_PKG_NO_PROD_SHORTLIST_V26_KEEP_ACCEPTED="${REPORT_DIR}/C_deadcode_package_no_prod_evidence_shortlist_test_only_keep_accepted.txt"
OUT_PKG_NO_PROD_SHORTLIST_V26_KEEP_REMAINING="${REPORT_DIR}/C_deadcode_package_no_prod_evidence_shortlist_test_only_keep_remaining.txt"
OUT_TOOLS="${REPORT_DIR}/C_deadcode_tools.txt"
OUT_TOOLS_FILTERED="${REPORT_DIR}/C_deadcode_tools_filtered.txt"
OUT_TOOLS_INVENTORY="${REPORT_DIR}/C_deadcode_tools_inventory.txt"
OUT_ROOT="${REPORT_DIR}/C_deadcode_root.txt"
OUT_ROOT_FILTERED="${REPORT_DIR}/C_deadcode_root_filtered.txt"
OUT_ROOT_INVENTORY="${REPORT_DIR}/C_deadcode_root_inventory.txt"

timestamp_utc="$(date -u +"%Y%m%dT%H%M%SZ")"
ALLOWLIST_FILE="${REPO_ROOT}/tools/audit/deadcode_allowlist.txt"

if ! command -v python3 >/dev/null 2>&1; then
  report_stub="$(
    cat <<EOF
Kindred Audit C: Dead-code candidates (heuristic, report-only)
Timestamp (UTC): ${timestamp_utc}
Repo root: ${REPO_ROOT}

C0 PythonMissing | WARN | - | python3 not available; Audit C cannot run | Install python3
EOF
  )"
  {
    echo "${report_stub}"
  } > "${OUT_LEGACY}"
  cp -a "${OUT_LEGACY}" "${OUT_LEGACY_FILTERED}"
  cp -a "${OUT_LEGACY}" "${OUT_PKG}"
  cp -a "${OUT_LEGACY_FILTERED}" "${OUT_PKG_FILTERED}"
  cp -a "${OUT_LEGACY}" "${OUT_TOOLS}"
  cp -a "${OUT_LEGACY_FILTERED}" "${OUT_TOOLS_FILTERED}"
	  cp -a "${OUT_LEGACY}" "${OUT_TOOLS_INVENTORY}"
	  cp -a "${OUT_LEGACY}" "${OUT_ROOT}"
	  cp -a "${OUT_LEGACY_FILTERED}" "${OUT_ROOT_FILTERED}"
	  cp -a "${OUT_LEGACY}" "${OUT_ROOT_INVENTORY}"
	  cp -a "${OUT_LEGACY}" "${OUT_PKG_EVIDENCE}"
		  cp -a "${OUT_LEGACY_FILTERED}" "${OUT_PKG_EVIDENCE_FILTERED}"
		  cp -a "${OUT_LEGACY}" "${OUT_PKG_BUCKETS}"
		  cp -a "${OUT_LEGACY_FILTERED}" "${OUT_PKG_BUCKETS_FILTERED}"
		  cp -a "${OUT_LEGACY}" "${OUT_PKG_HIGH}"
		  cp -a "${OUT_LEGACY_FILTERED}" "${OUT_PKG_HIGH_FILTERED}"
		  cp -a "${OUT_LEGACY}" "${OUT_PKG_BUCKETS_V25}"
		  cp -a "${OUT_LEGACY_FILTERED}" "${OUT_PKG_BUCKETS_V25_FILTERED}"
		  cp -a "${OUT_LEGACY}" "${OUT_PKG_NO_EVIDENCE_SHORTLIST}"
		  cp -a "${OUT_LEGACY_FILTERED}" "${OUT_PKG_NO_EVIDENCE_SHORTLIST_FILTERED}"
		  cp -a "${OUT_LEGACY}" "${OUT_PKG_PROD_BUCKETS_V26}"
			  cp -a "${OUT_LEGACY_FILTERED}" "${OUT_PKG_PROD_BUCKETS_V26_FILTERED}"
			  cp -a "${OUT_LEGACY}" "${OUT_PKG_NO_PROD_SHORTLIST_V26}"
			  cp -a "${OUT_LEGACY_FILTERED}" "${OUT_PKG_NO_PROD_SHORTLIST_V26_FILTERED}"
			  cp -a "${OUT_LEGACY_FILTERED}" "${OUT_PKG_NO_PROD_SHORTLIST_V26_KEEP_ACCEPTED}"
			  cp -a "${OUT_LEGACY_FILTERED}" "${OUT_PKG_NO_PROD_SHORTLIST_V26_KEEP_REMAINING}"
			  {
			    echo "Audit C (package): WARN (python3 missing)"
			    echo "Audit C (tools): WARN (python3 missing)"
			    echo "Audit C (root): WARN (python3 missing)"
			    echo "Audit C (package evidence): WARN (python3 missing)"
			    echo "Audit C (package buckets): WARN (python3 missing)"
			    echo "Audit C (package high-confidence): WARN (python3 missing)"
			    echo "Audit C (package evidence v2.5): WARN (python3 missing)"
			    echo "Audit C (package prod evidence v2.6): WARN (python3 missing)"
			    echo "Audit C reports: C_deadcode*.txt"
			  } >> "${SUMMARY_FILE}"
			  exit 0
			fi

set +e
python_out="$(
  cd -- "${REPO_ROOT}" && python3 "${SCRIPT_DIR}/deadcode_audit.py" \
  --root "${REPO_ROOT}" \
  --report-dir "${REPORT_DIR}" \
  --allowlist "${ALLOWLIST_FILE}" 2>&1
)"
rc=$?
set -e

if [[ ! -f "${OUT_LEGACY}" ]]; then
  {
    echo "Kindred Audit C: Dead-code candidates (heuristic, report-only)"
    echo "Timestamp (UTC): ${timestamp_utc}"
    echo "Repo root: ${REPO_ROOT}"
    echo
    echo "C0 AuditError | WARN | - | deadcode_audit.py failed to produce report (exit ${rc}) | Fix audit tooling"
  } > "${OUT_LEGACY}"
  rc=99
fi

if [[ ! -f "${OUT_LEGACY_FILTERED}" ]]; then
  cp -a "${OUT_LEGACY}" "${OUT_LEGACY_FILTERED}"
fi

if [[ ! -f "${OUT_PKG}" ]]; then
  cp -a "${OUT_LEGACY}" "${OUT_PKG}"
fi
if [[ ! -f "${OUT_PKG_FILTERED}" ]]; then
  cp -a "${OUT_LEGACY_FILTERED}" "${OUT_PKG_FILTERED}"
fi
if [[ ! -f "${OUT_TOOLS}" ]]; then
  cp -a "${OUT_LEGACY}" "${OUT_TOOLS}"
fi
if [[ ! -f "${OUT_TOOLS_FILTERED}" ]]; then
  cp -a "${OUT_LEGACY_FILTERED}" "${OUT_TOOLS_FILTERED}"
fi
if [[ ! -f "${OUT_TOOLS_INVENTORY}" ]]; then
  cp -a "${OUT_LEGACY}" "${OUT_TOOLS_INVENTORY}"
fi
if [[ ! -f "${OUT_ROOT}" ]]; then
  cp -a "${OUT_LEGACY}" "${OUT_ROOT}"
fi
if [[ ! -f "${OUT_ROOT_FILTERED}" ]]; then
  cp -a "${OUT_LEGACY_FILTERED}" "${OUT_ROOT_FILTERED}"
fi
	if [[ ! -f "${OUT_ROOT_INVENTORY}" ]]; then
	  cp -a "${OUT_LEGACY}" "${OUT_ROOT_INVENTORY}"
	fi
	if [[ ! -f "${OUT_PKG_EVIDENCE}" ]]; then
	  {
	    echo "Kindred Audit C: Package dead-code candidates (usage evidence triage, heuristic, report-only)"
	    echo "Timestamp (UTC): ${timestamp_utc}"
	    echo "Repo root: ${REPO_ROOT}"
	    echo
	    echo "C0 AuditError | WARN | - | deadcode_audit.py failed to produce package evidence report (exit ${rc}) | Fix audit tooling"
	  } > "${OUT_PKG_EVIDENCE}"
	fi
	if [[ ! -f "${OUT_PKG_EVIDENCE_FILTERED}" ]]; then
	  {
	    echo "Kindred Audit C: Package dead-code candidates (usage evidence triage, filtered by allowlist)"
	    echo "Timestamp (UTC): ${timestamp_utc}"
	    echo "Repo root: ${REPO_ROOT}"
	    echo
	    echo "C0 AuditError | WARN | - | deadcode_audit.py failed to produce package evidence filtered report (exit ${rc}) | Fix audit tooling"
	  } > "${OUT_PKG_EVIDENCE_FILTERED}"
	fi
	if [[ ! -f "${OUT_PKG_BUCKETS}" ]]; then
	  {
	    echo "Kindred Audit C: Package dead-code candidates (evidence buckets, report-only)"
	    echo "Timestamp (UTC): ${timestamp_utc}"
	    echo "Repo root: ${REPO_ROOT}"
	    echo
	    echo "C0 AuditError | WARN | - | deadcode_audit.py failed to produce package buckets report (exit ${rc}) | Fix audit tooling"
	  } > "${OUT_PKG_BUCKETS}"
	fi
	if [[ ! -f "${OUT_PKG_BUCKETS_FILTERED}" ]]; then
	  {
	    echo "Kindred Audit C: Package dead-code candidates (evidence buckets, filtered by allowlist)"
	    echo "Timestamp (UTC): ${timestamp_utc}"
	    echo "Repo root: ${REPO_ROOT}"
	    echo
	    echo "C0 AuditError | WARN | - | deadcode_audit.py failed to produce package buckets filtered report (exit ${rc}) | Fix audit tooling"
	  } > "${OUT_PKG_BUCKETS_FILTERED}"
	fi
	if [[ ! -f "${OUT_PKG_HIGH}" ]]; then
	  {
	    echo "Kindred Audit C: Package dead-code candidates (high-confidence shortlist, report-only)"
	    echo "Timestamp (UTC): ${timestamp_utc}"
	    echo "Repo root: ${REPO_ROOT}"
	    echo
	    echo "C0 AuditError | WARN | - | deadcode_audit.py failed to produce package high-confidence report (exit ${rc}) | Fix audit tooling"
	  } > "${OUT_PKG_HIGH}"
	fi
		if [[ ! -f "${OUT_PKG_HIGH_FILTERED}" ]]; then
		  {
		    echo "Kindred Audit C: Package dead-code candidates (high-confidence shortlist, filtered by allowlist)"
		    echo "Timestamp (UTC): ${timestamp_utc}"
		    echo "Repo root: ${REPO_ROOT}"
		    echo
		    echo "C0 AuditError | WARN | - | deadcode_audit.py failed to produce package high-confidence filtered report (exit ${rc}) | Fix audit tooling"
		  } > "${OUT_PKG_HIGH_FILTERED}"
		fi
		if [[ ! -f "${OUT_PKG_BUCKETS_V25}" ]]; then
		  {
		    echo "Kindred Audit C: Package dead-code candidates (evidence buckets v2.5, report-only)"
		    echo "Timestamp (UTC): ${timestamp_utc}"
		    echo "Repo root: ${REPO_ROOT}"
		    echo
		    echo "C0 AuditError | WARN | - | deadcode_audit.py failed to produce package buckets v2.5 report (exit ${rc}) | Fix audit tooling"
		  } > "${OUT_PKG_BUCKETS_V25}"
		fi
		if [[ ! -f "${OUT_PKG_BUCKETS_V25_FILTERED}" ]]; then
		  {
		    echo "Kindred Audit C: Package dead-code candidates (evidence buckets v2.5, filtered by allowlist)"
		    echo "Timestamp (UTC): ${timestamp_utc}"
		    echo "Repo root: ${REPO_ROOT}"
		    echo
		    echo "C0 AuditError | WARN | - | deadcode_audit.py failed to produce package buckets v2.5 filtered report (exit ${rc}) | Fix audit tooling"
		  } > "${OUT_PKG_BUCKETS_V25_FILTERED}"
		fi
		if [[ ! -f "${OUT_PKG_NO_EVIDENCE_SHORTLIST}" ]]; then
		  {
		    echo "Kindred Audit C: Package dead-code candidates (NO_EVIDENCE shortlist v2.5, report-only)"
		    echo "Timestamp (UTC): ${timestamp_utc}"
		    echo "Repo root: ${REPO_ROOT}"
		    echo
		    echo "C0 AuditError | WARN | - | deadcode_audit.py failed to produce package NO_EVIDENCE shortlist v2.5 report (exit ${rc}) | Fix audit tooling"
		  } > "${OUT_PKG_NO_EVIDENCE_SHORTLIST}"
		fi
			if [[ ! -f "${OUT_PKG_NO_EVIDENCE_SHORTLIST_FILTERED}" ]]; then
			  {
			    echo "Kindred Audit C: Package dead-code candidates (NO_EVIDENCE shortlist v2.5, filtered by allowlist)"
			    echo "Timestamp (UTC): ${timestamp_utc}"
			    echo "Repo root: ${REPO_ROOT}"
			    echo
			    echo "C0 AuditError | WARN | - | deadcode_audit.py failed to produce package NO_EVIDENCE shortlist v2.5 filtered report (exit ${rc}) | Fix audit tooling"
			  } > "${OUT_PKG_NO_EVIDENCE_SHORTLIST_FILTERED}"
			fi
			if [[ ! -f "${OUT_PKG_PROD_BUCKETS_V26}" ]]; then
			  {
			    echo "Kindred Audit C: Package dead-code candidates (production-vs-tests evidence buckets v2.6, report-only)"
			    echo "Timestamp (UTC): ${timestamp_utc}"
			    echo "Repo root: ${REPO_ROOT}"
			    echo
			    echo "C0 AuditError | WARN | - | deadcode_audit.py failed to produce package PROD evidence buckets v2.6 report (exit ${rc}) | Fix audit tooling"
			  } > "${OUT_PKG_PROD_BUCKETS_V26}"
			fi
			if [[ ! -f "${OUT_PKG_PROD_BUCKETS_V26_FILTERED}" ]]; then
			  {
			    echo "Kindred Audit C: Package dead-code candidates (production-vs-tests evidence buckets v2.6, filtered by allowlist)"
			    echo "Timestamp (UTC): ${timestamp_utc}"
			    echo "Repo root: ${REPO_ROOT}"
			    echo
			    echo "C0 AuditError | WARN | - | deadcode_audit.py failed to produce package PROD evidence buckets v2.6 filtered report (exit ${rc}) | Fix audit tooling"
			  } > "${OUT_PKG_PROD_BUCKETS_V26_FILTERED}"
			fi
			if [[ ! -f "${OUT_PKG_NO_PROD_SHORTLIST_V26}" ]]; then
			  {
			    echo "Kindred Audit C: Package dead-code candidates (no-PROD-evidence shortlist v2.6, report-only)"
			    echo "Timestamp (UTC): ${timestamp_utc}"
			    echo "Repo root: ${REPO_ROOT}"
			    echo
			    echo "C0 AuditError | WARN | - | deadcode_audit.py failed to produce package no-PROD-evidence shortlist v2.6 report (exit ${rc}) | Fix audit tooling"
			  } > "${OUT_PKG_NO_PROD_SHORTLIST_V26}"
			fi
				if [[ ! -f "${OUT_PKG_NO_PROD_SHORTLIST_V26_FILTERED}" ]]; then
				  {
				    echo "Kindred Audit C: Package dead-code candidates (no-PROD-evidence shortlist v2.6, filtered by allowlist)"
				    echo "Timestamp (UTC): ${timestamp_utc}"
				    echo "Repo root: ${REPO_ROOT}"
				    echo
				    echo "C0 AuditError | WARN | - | deadcode_audit.py failed to produce package no-PROD-evidence shortlist v2.6 filtered report (exit ${rc}) | Fix audit tooling"
				  } > "${OUT_PKG_NO_PROD_SHORTLIST_V26_FILTERED}"
				fi
				if [[ ! -f "${OUT_PKG_NO_PROD_SHORTLIST_V26_KEEP_ACCEPTED}" ]]; then
				  {
				    echo "Kindred Audit C: Package dead-code candidates (no-PROD-evidence shortlist v2.6, accepted as TEST-only keep)"
				    echo "Timestamp (UTC): ${timestamp_utc}"
				    echo "Repo root: ${REPO_ROOT}"
				    echo
				    echo "C0 AuditError | WARN | - | deadcode_audit.py failed to produce package v2.6 TEST-only keep accepted report (exit ${rc}) | Fix audit tooling"
				  } > "${OUT_PKG_NO_PROD_SHORTLIST_V26_KEEP_ACCEPTED}"
				fi
				if [[ ! -f "${OUT_PKG_NO_PROD_SHORTLIST_V26_KEEP_REMAINING}" ]]; then
				  {
				    echo "Kindred Audit C: Package dead-code candidates (no-PROD-evidence shortlist v2.6, remaining after TEST-only keep allowlist)"
				    echo "Timestamp (UTC): ${timestamp_utc}"
				    echo "Repo root: ${REPO_ROOT}"
				    echo
				    echo "C0 AuditError | WARN | - | deadcode_audit.py failed to produce package v2.6 TEST-only keep remaining report (exit ${rc}) | Fix audit tooling"
				  } > "${OUT_PKG_NO_PROD_SHORTLIST_V26_KEEP_REMAINING}"
				fi

		pkg_raw=""
		pkg_filtered=""
		pkg_allowlisted=""
			pkg_parse_failures=""
	pkg_evidence_filtered_has=""
	pkg_evidence_filtered_none=""
	pkg_bucket_filtered_import=""
	pkg_bucket_filtered_py_runtime=""
	pkg_bucket_filtered_non_py=""
	pkg_bucket_filtered_no_evidence=""
	pkg_bucket_raw_import=""
	pkg_bucket_raw_py_runtime=""
	pkg_bucket_raw_non_py=""
		pkg_bucket_raw_no_evidence=""
		pkg_high_confidence_raw=""
		pkg_high_confidence_filtered=""
		pkg_v25_bucket_filtered_import=""
		pkg_v25_bucket_filtered_dynamic=""
		pkg_v25_bucket_filtered_reexport_only=""
			pkg_v25_bucket_filtered_no_evidence=""
			pkg_v25_shortlist_filtered=""
			pkg_v26_bucket_filtered_prod_import=""
			pkg_v26_bucket_filtered_prod_dynamic=""
				pkg_v26_bucket_filtered_prod_reexport_only=""
				pkg_v26_bucket_filtered_no_prod_evidence=""
				pkg_v26_shortlist_filtered=""
				pkg_v26_shortlist_filtered_keep_accepted=""
				pkg_v26_shortlist_filtered_keep_remaining=""
				pkg_v26_keep_allowlist_missing=""
				pkg_v26_keep_allowlist_entries=""
				tools_total_scripts=""
				tools_referenced_scripts=""
				tools_raw=""
		tools_filtered=""
tools_allowlisted=""
tools_parse_failures=""
root_total_scripts=""
root_referenced_scripts=""
root_raw=""
root_filtered=""
root_allowlisted=""
root_parse_failures=""
allowlist_missing=""

while IFS= read -r line; do
  if [[ "${line}" == DEADCODE_AUDIT_COUNTS_V2* ]]; then
    pkg_raw="$(sed -n 's/.* package_raw=\([0-9][0-9]*\).*/\1/p' <<<"${line}" | head -n 1)"
    pkg_filtered="$(sed -n 's/.* package_filtered=\([0-9][0-9]*\).*/\1/p' <<<"${line}" | head -n 1)"
	    pkg_allowlisted="$(sed -n 's/.* package_allowlisted=\([0-9][0-9]*\).*/\1/p' <<<"${line}" | head -n 1)"
	    pkg_parse_failures="$(sed -n 's/.* package_parse_failures=\([0-9][0-9]*\).*/\1/p' <<<"${line}" | head -n 1)"
	    pkg_evidence_filtered_has="$(sed -n 's/.* package_evidence_filtered_has=\([0-9][0-9]*\).*/\1/p' <<<"${line}" | head -n 1)"
	    pkg_evidence_filtered_none="$(sed -n 's/.* package_evidence_filtered_none=\([0-9][0-9]*\).*/\1/p' <<<"${line}" | head -n 1)"
	    pkg_bucket_raw_import="$(sed -n 's/.* package_bucket_raw_import=\([0-9][0-9]*\).*/\1/p' <<<"${line}" | head -n 1)"
	    pkg_bucket_raw_py_runtime="$(sed -n 's/.* package_bucket_raw_py_runtime=\([0-9][0-9]*\).*/\1/p' <<<"${line}" | head -n 1)"
	    pkg_bucket_raw_non_py="$(sed -n 's/.* package_bucket_raw_non_py=\([0-9][0-9]*\).*/\1/p' <<<"${line}" | head -n 1)"
	    pkg_bucket_raw_no_evidence="$(sed -n 's/.* package_bucket_raw_no_evidence=\([0-9][0-9]*\).*/\1/p' <<<"${line}" | head -n 1)"
	    pkg_bucket_filtered_import="$(sed -n 's/.* package_bucket_filtered_import=\([0-9][0-9]*\).*/\1/p' <<<"${line}" | head -n 1)"
	    pkg_bucket_filtered_py_runtime="$(sed -n 's/.* package_bucket_filtered_py_runtime=\([0-9][0-9]*\).*/\1/p' <<<"${line}" | head -n 1)"
	    pkg_bucket_filtered_non_py="$(sed -n 's/.* package_bucket_filtered_non_py=\([0-9][0-9]*\).*/\1/p' <<<"${line}" | head -n 1)"
		    pkg_bucket_filtered_no_evidence="$(sed -n 's/.* package_bucket_filtered_no_evidence=\([0-9][0-9]*\).*/\1/p' <<<"${line}" | head -n 1)"
		    pkg_high_confidence_raw="$(sed -n 's/.* package_high_confidence_raw=\([0-9][0-9]*\).*/\1/p' <<<"${line}" | head -n 1)"
		    pkg_high_confidence_filtered="$(sed -n 's/.* package_high_confidence_filtered=\([0-9][0-9]*\).*/\1/p' <<<"${line}" | head -n 1)"
		    pkg_v25_bucket_filtered_import="$(sed -n 's/.* package_v25_bucket_filtered_import=\([0-9][0-9]*\).*/\1/p' <<<"${line}" | head -n 1)"
		    pkg_v25_bucket_filtered_dynamic="$(sed -n 's/.* package_v25_bucket_filtered_dynamic=\([0-9][0-9]*\).*/\1/p' <<<"${line}" | head -n 1)"
		    pkg_v25_bucket_filtered_reexport_only="$(sed -n 's/.* package_v25_bucket_filtered_reexport_only=\([0-9][0-9]*\).*/\1/p' <<<"${line}" | head -n 1)"
		    pkg_v25_bucket_filtered_no_evidence="$(sed -n 's/.* package_v25_bucket_filtered_no_evidence=\([0-9][0-9]*\).*/\1/p' <<<"${line}" | head -n 1)"
		    pkg_v25_shortlist_filtered="$(sed -n 's/.* package_v25_shortlist_filtered=\([0-9][0-9]*\).*/\1/p' <<<"${line}" | head -n 1)"
		    pkg_v26_bucket_filtered_prod_import="$(sed -n 's/.* package_v26_bucket_filtered_prod_import=\([0-9][0-9]*\).*/\1/p' <<<"${line}" | head -n 1)"
		    pkg_v26_bucket_filtered_prod_dynamic="$(sed -n 's/.* package_v26_bucket_filtered_prod_dynamic=\([0-9][0-9]*\).*/\1/p' <<<"${line}" | head -n 1)"
			    pkg_v26_bucket_filtered_prod_reexport_only="$(sed -n 's/.* package_v26_bucket_filtered_prod_reexport_only=\([0-9][0-9]*\).*/\1/p' <<<"${line}" | head -n 1)"
			    pkg_v26_bucket_filtered_no_prod_evidence="$(sed -n 's/.* package_v26_bucket_filtered_no_prod_evidence=\([0-9][0-9]*\).*/\1/p' <<<"${line}" | head -n 1)"
			    pkg_v26_shortlist_filtered="$(sed -n 's/.* package_v26_shortlist_filtered=\([0-9][0-9]*\).*/\1/p' <<<"${line}" | head -n 1)"
			    pkg_v26_shortlist_filtered_keep_accepted="$(sed -n 's/.* package_v26_shortlist_filtered_keep_accepted=\([0-9][0-9]*\).*/\1/p' <<<"${line}" | head -n 1)"
			    pkg_v26_shortlist_filtered_keep_remaining="$(sed -n 's/.* package_v26_shortlist_filtered_keep_remaining=\([0-9][0-9]*\).*/\1/p' <<<"${line}" | head -n 1)"
			    pkg_v26_keep_allowlist_missing="$(sed -n 's/.* package_v26_keep_allowlist_missing=\([0-9][0-9]*\).*/\1/p' <<<"${line}" | head -n 1)"
			    pkg_v26_keep_allowlist_entries="$(sed -n 's/.* package_v26_keep_allowlist_entries=\([0-9][0-9]*\).*/\1/p' <<<"${line}" | head -n 1)"
			    tools_total_scripts="$(sed -n 's/.* tools_total_scripts=\([0-9][0-9]*\).*/\1/p' <<<"${line}" | head -n 1)"
			    tools_referenced_scripts="$(sed -n 's/.* tools_referenced_scripts=\([0-9][0-9]*\).*/\1/p' <<<"${line}" | head -n 1)"
			    tools_raw="$(sed -n 's/.* tools_raw=\([0-9][0-9]*\).*/\1/p' <<<"${line}" | head -n 1)"
    tools_filtered="$(sed -n 's/.* tools_filtered=\([0-9][0-9]*\).*/\1/p' <<<"${line}" | head -n 1)"
    tools_allowlisted="$(sed -n 's/.* tools_allowlisted=\([0-9][0-9]*\).*/\1/p' <<<"${line}" | head -n 1)"
    tools_parse_failures="$(sed -n 's/.* tools_parse_failures=\([0-9][0-9]*\).*/\1/p' <<<"${line}" | head -n 1)"
    root_total_scripts="$(sed -n 's/.* root_total_scripts=\([0-9][0-9]*\).*/\1/p' <<<"${line}" | head -n 1)"
    root_referenced_scripts="$(sed -n 's/.* root_referenced_scripts=\([0-9][0-9]*\).*/\1/p' <<<"${line}" | head -n 1)"
    root_raw="$(sed -n 's/.* root_raw=\([0-9][0-9]*\).*/\1/p' <<<"${line}" | head -n 1)"
    root_filtered="$(sed -n 's/.* root_filtered=\([0-9][0-9]*\).*/\1/p' <<<"${line}" | head -n 1)"
    root_allowlisted="$(sed -n 's/.* root_allowlisted=\([0-9][0-9]*\).*/\1/p' <<<"${line}" | head -n 1)"
    root_parse_failures="$(sed -n 's/.* root_parse_failures=\([0-9][0-9]*\).*/\1/p' <<<"${line}" | head -n 1)"
    allowlist_missing="$(sed -n 's/.* allowlist_missing=\([0-9][0-9]*\).*/\1/p' <<<"${line}" | head -n 1)"
    break
  fi
done <<<"${python_out}"

if [[ -z "${allowlist_missing}" ]]; then allowlist_missing="?" ; fi
if [[ -z "${pkg_raw}" ]]; then pkg_raw="?" ; fi
if [[ -z "${pkg_filtered}" ]]; then pkg_filtered="?" ; fi
	if [[ -z "${pkg_allowlisted}" ]]; then pkg_allowlisted="?" ; fi
	if [[ -z "${pkg_parse_failures}" ]]; then pkg_parse_failures="?" ; fi
	if [[ -z "${pkg_evidence_filtered_has}" ]]; then pkg_evidence_filtered_has="?" ; fi
	if [[ -z "${pkg_evidence_filtered_none}" ]]; then pkg_evidence_filtered_none="?" ; fi
	if [[ -z "${pkg_bucket_raw_import}" ]]; then pkg_bucket_raw_import="?" ; fi
	if [[ -z "${pkg_bucket_raw_py_runtime}" ]]; then pkg_bucket_raw_py_runtime="?" ; fi
	if [[ -z "${pkg_bucket_raw_non_py}" ]]; then pkg_bucket_raw_non_py="?" ; fi
	if [[ -z "${pkg_bucket_raw_no_evidence}" ]]; then pkg_bucket_raw_no_evidence="?" ; fi
	if [[ -z "${pkg_bucket_filtered_import}" ]]; then pkg_bucket_filtered_import="?" ; fi
	if [[ -z "${pkg_bucket_filtered_py_runtime}" ]]; then pkg_bucket_filtered_py_runtime="?" ; fi
	if [[ -z "${pkg_bucket_filtered_non_py}" ]]; then pkg_bucket_filtered_non_py="?" ; fi
	if [[ -z "${pkg_bucket_filtered_no_evidence}" ]]; then pkg_bucket_filtered_no_evidence="?" ; fi
		if [[ -z "${pkg_high_confidence_raw}" ]]; then pkg_high_confidence_raw="?" ; fi
		if [[ -z "${pkg_high_confidence_filtered}" ]]; then pkg_high_confidence_filtered="?" ; fi
		if [[ -z "${pkg_v25_bucket_filtered_import}" ]]; then pkg_v25_bucket_filtered_import="?" ; fi
		if [[ -z "${pkg_v25_bucket_filtered_dynamic}" ]]; then pkg_v25_bucket_filtered_dynamic="?" ; fi
			if [[ -z "${pkg_v25_bucket_filtered_reexport_only}" ]]; then pkg_v25_bucket_filtered_reexport_only="?" ; fi
			if [[ -z "${pkg_v25_bucket_filtered_no_evidence}" ]]; then pkg_v25_bucket_filtered_no_evidence="?" ; fi
			if [[ -z "${pkg_v25_shortlist_filtered}" ]]; then pkg_v25_shortlist_filtered="?" ; fi
			if [[ -z "${pkg_v26_bucket_filtered_prod_import}" ]]; then pkg_v26_bucket_filtered_prod_import="?" ; fi
			if [[ -z "${pkg_v26_bucket_filtered_prod_dynamic}" ]]; then pkg_v26_bucket_filtered_prod_dynamic="?" ; fi
			if [[ -z "${pkg_v26_bucket_filtered_prod_reexport_only}" ]]; then pkg_v26_bucket_filtered_prod_reexport_only="?" ; fi
			if [[ -z "${pkg_v26_bucket_filtered_no_prod_evidence}" ]]; then pkg_v26_bucket_filtered_no_prod_evidence="?" ; fi
			if [[ -z "${pkg_v26_shortlist_filtered}" ]]; then pkg_v26_shortlist_filtered="?" ; fi
			if [[ -z "${pkg_v26_shortlist_filtered_keep_accepted}" ]]; then pkg_v26_shortlist_filtered_keep_accepted="?" ; fi
			if [[ -z "${pkg_v26_shortlist_filtered_keep_remaining}" ]]; then pkg_v26_shortlist_filtered_keep_remaining="?" ; fi
			if [[ -z "${pkg_v26_keep_allowlist_missing}" ]]; then pkg_v26_keep_allowlist_missing="?" ; fi
			if [[ -z "${pkg_v26_keep_allowlist_entries}" ]]; then pkg_v26_keep_allowlist_entries="?" ; fi
	if [[ -z "${tools_raw}" ]]; then tools_raw="?" ; fi
	if [[ -z "${tools_filtered}" ]]; then tools_filtered="?" ; fi
			if [[ -z "${tools_allowlisted}" ]]; then tools_allowlisted="?" ; fi
	if [[ -z "${tools_parse_failures}" ]]; then tools_parse_failures="?" ; fi
if [[ -z "${tools_total_scripts}" ]]; then tools_total_scripts="?" ; fi
if [[ -z "${tools_referenced_scripts}" ]]; then tools_referenced_scripts="?" ; fi
if [[ -z "${root_raw}" ]]; then root_raw="?" ; fi
if [[ -z "${root_filtered}" ]]; then root_filtered="?" ; fi
if [[ -z "${root_allowlisted}" ]]; then root_allowlisted="?" ; fi
if [[ -z "${root_parse_failures}" ]]; then root_parse_failures="?" ; fi
if [[ -z "${root_total_scripts}" ]]; then root_total_scripts="?" ; fi
if [[ -z "${root_referenced_scripts}" ]]; then root_referenced_scripts="?" ; fi

	pkg_status="WARN"
	tools_status="WARN"
	root_status="WARN"
	pkg_high_status="WARN"
	if [[ "${pkg_v26_shortlist_filtered_keep_remaining}" != "?" ]]; then
	  if [[ "${pkg_v26_shortlist_filtered_keep_remaining}" == "0" && "${pkg_parse_failures}" == "0" ]]; then
	    pkg_status="PASS"
	  fi
	else
	  if [[ "${pkg_filtered}" == "0" && "${pkg_parse_failures}" == "0" ]]; then
	    pkg_status="PASS"
	  fi
	fi
	if [[ "${tools_filtered}" == "0" && "${tools_parse_failures}" == "0" ]]; then
	  tools_status="PASS"
	fi
	if [[ "${root_filtered}" == "0" && "${root_parse_failures}" == "0" ]]; then
	  root_status="PASS"
	fi
	if [[ "${pkg_high_confidence_filtered}" == "0" ]]; then
	  pkg_high_status="PASS"
	fi

	if [[ "${pkg_status}" == "PASS" ]]; then
	  {
		    echo "Audit C (package): PASS (raw=${pkg_raw}, filtered=${pkg_filtered}, allowlisted=${pkg_allowlisted}, allowlist_missing=${allowlist_missing}, parse_failures=${pkg_parse_failures})"
		    echo "Audit C (package evidence): INFO (filtered_has_evidence=${pkg_evidence_filtered_has}, filtered_no_evidence=${pkg_evidence_filtered_none})"
		    echo "Audit C (package buckets): INFO (raw: import=${pkg_bucket_raw_import}, py_runtime=${pkg_bucket_raw_py_runtime}, non_py=${pkg_bucket_raw_non_py}, no_evidence=${pkg_bucket_raw_no_evidence}; filtered: import=${pkg_bucket_filtered_import}, py_runtime=${pkg_bucket_filtered_py_runtime}, non_py=${pkg_bucket_filtered_non_py}, no_evidence=${pkg_bucket_filtered_no_evidence})"
			    echo "Audit C (package high-confidence): ${pkg_high_status} (raw=${pkg_high_confidence_raw}, filtered=${pkg_high_confidence_filtered})"
			    echo "Audit C (package evidence v2.5): INFO (filtered_shortlist=${pkg_v25_shortlist_filtered}; filtered_buckets: import=${pkg_v25_bucket_filtered_import}, dynamic=${pkg_v25_bucket_filtered_dynamic}, reexport_only=${pkg_v25_bucket_filtered_reexport_only}, no_evidence=${pkg_v25_bucket_filtered_no_evidence})"
			    echo "Audit C (package prod evidence v2.6): INFO (filtered_shortlist=${pkg_v26_shortlist_filtered}, keep_accepted=${pkg_v26_shortlist_filtered_keep_accepted}, remaining_after_keep=${pkg_v26_shortlist_filtered_keep_remaining}, keep_allowlist_missing=${pkg_v26_keep_allowlist_missing}, keep_allowlist_entries=${pkg_v26_keep_allowlist_entries}; filtered_buckets: prod_import=${pkg_v26_bucket_filtered_prod_import}, prod_dynamic=${pkg_v26_bucket_filtered_prod_dynamic}, prod_reexport_only=${pkg_v26_bucket_filtered_prod_reexport_only}, no_prod_evidence=${pkg_v26_bucket_filtered_no_prod_evidence})"
			    echo "Audit C (tools): ${tools_status} (scripts_total=${tools_total_scripts}, referenced=${tools_referenced_scripts}, raw=${tools_raw}, filtered=${tools_filtered}, allowlisted=${tools_allowlisted}, allowlist_missing=${allowlist_missing}, parse_failures=${tools_parse_failures})"
			    echo "Audit C (root): ${root_status} (scripts_total=${root_total_scripts}, referenced=${root_referenced_scripts}, raw=${root_raw}, filtered=${root_filtered}, allowlisted=${root_allowlisted}, allowlist_missing=${allowlist_missing}, parse_failures=${root_parse_failures})"
			    echo "Audit C reports: C_deadcode*.txt"
			  } >> "${SUMMARY_FILE}"
	else
	  {
		    echo "Audit C (package): WARN (raw=${pkg_raw}, filtered=${pkg_filtered}, allowlisted=${pkg_allowlisted}, allowlist_missing=${allowlist_missing}, parse_failures=${pkg_parse_failures})"
		    echo "Audit C (package evidence): INFO (filtered_has_evidence=${pkg_evidence_filtered_has}, filtered_no_evidence=${pkg_evidence_filtered_none})"
		    echo "Audit C (package buckets): INFO (raw: import=${pkg_bucket_raw_import}, py_runtime=${pkg_bucket_raw_py_runtime}, non_py=${pkg_bucket_raw_non_py}, no_evidence=${pkg_bucket_raw_no_evidence}; filtered: import=${pkg_bucket_filtered_import}, py_runtime=${pkg_bucket_filtered_py_runtime}, non_py=${pkg_bucket_filtered_non_py}, no_evidence=${pkg_bucket_filtered_no_evidence})"
			    echo "Audit C (package high-confidence): ${pkg_high_status} (raw=${pkg_high_confidence_raw}, filtered=${pkg_high_confidence_filtered})"
			    echo "Audit C (package evidence v2.5): INFO (filtered_shortlist=${pkg_v25_shortlist_filtered}; filtered_buckets: import=${pkg_v25_bucket_filtered_import}, dynamic=${pkg_v25_bucket_filtered_dynamic}, reexport_only=${pkg_v25_bucket_filtered_reexport_only}, no_evidence=${pkg_v25_bucket_filtered_no_evidence})"
			    echo "Audit C (package prod evidence v2.6): INFO (filtered_shortlist=${pkg_v26_shortlist_filtered}, keep_accepted=${pkg_v26_shortlist_filtered_keep_accepted}, remaining_after_keep=${pkg_v26_shortlist_filtered_keep_remaining}, keep_allowlist_missing=${pkg_v26_keep_allowlist_missing}, keep_allowlist_entries=${pkg_v26_keep_allowlist_entries}; filtered_buckets: prod_import=${pkg_v26_bucket_filtered_prod_import}, prod_dynamic=${pkg_v26_bucket_filtered_prod_dynamic}, prod_reexport_only=${pkg_v26_bucket_filtered_prod_reexport_only}, no_prod_evidence=${pkg_v26_bucket_filtered_no_prod_evidence})"
			    echo "Audit C (tools): ${tools_status} (scripts_total=${tools_total_scripts}, referenced=${tools_referenced_scripts}, raw=${tools_raw}, filtered=${tools_filtered}, allowlisted=${tools_allowlisted}, allowlist_missing=${allowlist_missing}, parse_failures=${tools_parse_failures})"
			    echo "Audit C (root): ${root_status} (scripts_total=${root_total_scripts}, referenced=${root_referenced_scripts}, raw=${root_raw}, filtered=${root_filtered}, allowlisted=${root_allowlisted}, allowlist_missing=${allowlist_missing}, parse_failures=${root_parse_failures})"
			    echo "Audit C reports: C_deadcode*.txt"
			  } >> "${SUMMARY_FILE}"
	fi

exit 0
