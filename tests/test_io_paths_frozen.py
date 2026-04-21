from __future__ import annotations

import sys
from pathlib import Path

from kindred.io import paths as kindred_paths
import pytest

pytestmark = pytest.mark.unit



def test_find_outputs_dir_respects_env_override(monkeypatch, tmp_path: Path) -> None:
    out_dir = tmp_path / "custom_outputs"
    monkeypatch.setenv("KINDRED_OUTPUT_DIR", str(out_dir))

    assert kindred_paths.find_outputs_dir() == str(out_dir.resolve())


def test_resolve_start_dir_does_not_create_outputs(monkeypatch, tmp_path: Path) -> None:
    base_dir = tmp_path / "base_dir"
    outputs_dir = base_dir / "outputs"

    monkeypatch.setenv("KINDRED_BASE_DIR", str(base_dir))
    monkeypatch.delenv("KINDRED_OUTPUT_DIR", raising=False)
    monkeypatch.setattr(sys, "frozen", True, raising=False)

    start_dir = Path(kindred_paths.resolve_start_dir(None))
    assert start_dir == outputs_dir.resolve()
    assert not outputs_dir.exists()
