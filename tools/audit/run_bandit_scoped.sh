#!/usr/bin/env bash
set -euo pipefail

# Run Bandit only on scored, relevant Python sources.
#
# IMPORTANT:
#   which makes `bandit -r .` fail with: "Multiple .bandit files found".
# - Always pass `--ini` explicitly and target only the intended source roots.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [[ -x "${REPO_ROOT}/.venv/bin/bandit" ]]; then
  BANDIT_BIN="${REPO_ROOT}/.venv/bin/bandit"
else
  BANDIT_BIN="bandit"
fi

exec "${BANDIT_BIN}" \
  --ini "${REPO_ROOT}/kindred/.bandit" \
  -r \
  "${REPO_ROOT}/kindred" \
  "${REPO_ROOT}/tools/audit" \
  "${REPO_ROOT}/tests" \
  "$@"

