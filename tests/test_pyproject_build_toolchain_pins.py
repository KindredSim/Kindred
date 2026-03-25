from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.unit
def test_pyproject_pins_build_system_toolchain() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    content = (repo_root / "pyproject.toml").read_text(encoding="utf-8")

    assert 'requires = ["setuptools==75.1.0", "wheel==0.44.0"]' in content
