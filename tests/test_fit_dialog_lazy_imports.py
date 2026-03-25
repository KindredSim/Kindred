from __future__ import annotations

import json
import os
import subprocess  # nosec B404 - controlled local interpreter invocation in test
import sys
from pathlib import Path

import pytest


pytestmark = [pytest.mark.unit]


def test_fitting_window_import_does_not_import_global_fitting() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    code = """
import json
import sys
import kindred.gui.fitting.window
print(json.dumps({
    "global_fitting_imported": "kindred.core.analysis.global_fitting" in sys.modules,
}))
"""
    env = dict(os.environ)
    result = subprocess.run(  # nosec B603 - controlled args, shell=False
        [sys.executable, "-c", code],
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(result.stdout.strip())
    assert payload["global_fitting_imported"] is False
