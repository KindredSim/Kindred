import os
import subprocess  # nosec B404 - controlled local interpreter invocation in test
import sys
from pathlib import Path

import pytest


@pytest.mark.gui
def test_import_tutorial_manager_in_fresh_process():
    repo_root = Path(__file__).resolve().parents[1]

    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["QT_QPA_PLATFORM"] = "offscreen"
    env["MPLBACKEND"] = "Agg"

    proc = subprocess.run(  # nosec B603 - controlled args, shell=False
        [sys.executable, "-c", "import kindred.gui.tutorial_manager"],
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, (
        "Importing kindred.gui.tutorial_manager should succeed in a fresh subprocess.\n"
        f"stdout:\n{proc.stdout}\n"
        f"stderr:\n{proc.stderr}\n"
    )
