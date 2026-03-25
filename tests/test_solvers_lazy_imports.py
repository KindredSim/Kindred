from __future__ import annotations

import json
import os
import subprocess  # nosec B404 - test uses subprocess to spawn an isolated interpreter
import sys
from pathlib import Path

import pytest


pytestmark = [pytest.mark.unit]


def test_solvers_import_does_not_import_scipy_integrate() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    code = """
import json
import sys
import kindred.core.simulator.solvers
print(json.dumps({
    "scipy_integrate_imported": "scipy.integrate" in sys.modules,
}))
"""
    env = dict(os.environ)
    result = subprocess.run(  # nosec B603 - test invokes a local interpreter with controlled args
        [sys.executable, "-c", code],
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(result.stdout.strip())
    assert payload["scipy_integrate_imported"] is False
