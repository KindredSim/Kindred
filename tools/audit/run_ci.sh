#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# CI is the strict audit gate. Temporary tests are run separately.
exec bash "${SCRIPT_DIR}/run_strict.sh"
