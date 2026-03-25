from __future__ import annotations

import json
import os
import subprocess  # nosec B404 - test uses subprocess to spawn an isolated interpreter
import sys
from pathlib import Path

import pytest


pytestmark = [pytest.mark.unit]


def test_main_window_import_does_not_touch_multiprocessing_modules() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    code = """
import json
import sys
import kindred.gui.main_window
print(json.dumps({
    "processpool_module": "concurrent.futures.process" in sys.modules,
    "has_executor_symbol": hasattr(kindred.gui.main_window, "ProcessPoolExecutor"),
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
    assert payload["processpool_module"] is False
    assert payload["has_executor_symbol"] is False
