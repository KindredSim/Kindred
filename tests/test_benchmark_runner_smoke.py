from __future__ import annotations

import os
import subprocess  # nosec B404 - controlled local interpreter invocation in test
import sys
from pathlib import Path

import pytest


@pytest.mark.unit
def test_benchmark_runner_help_smoke() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["PYTHONNOUSERSITE"] = "1"

    result = subprocess.run(  # nosec B603 - controlled args, shell=False
        [sys.executable, str(repo_root / "benchmarks" / "run_benchmarks.py"), "--help"],
        cwd=str(repo_root),
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    assert "Run Kindred performance benchmarks" in result.stdout
