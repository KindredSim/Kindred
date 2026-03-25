from __future__ import annotations

import ast
import importlib.resources
from pathlib import Path

import pytest


def _imports_module(source: str, *, module: str) -> bool:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name == module for alias in node.names):
                return True
        if isinstance(node, ast.ImportFrom):
            if node.module == module:
                return True
    return False


@pytest.mark.unit
def test_entrypoint_gui_wrapper_imports_pyside6() -> None:
    pkg = importlib.resources.files("kindred")
    source = pkg.joinpath("gui_entrypoint.py").read_text(encoding="utf-8")

    assert _imports_module(source, module="PySide6")


@pytest.mark.unit
def test_gui_entrypoint_returns_authoritative_exit_code(monkeypatch) -> None:
    import kindred.__main__ as launcher
    import kindred.gui_entrypoint as entrypoint

    monkeypatch.setattr(launcher, "main", lambda: 17)

    assert entrypoint.main() == 17


@pytest.mark.unit
def test_pyproject_maps_console_scripts_to_gui_wrapper() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    content = (repo_root / "pyproject.toml").read_text(encoding="utf-8")

    assert 'kindred = "kindred.gui_entrypoint:main"' in content
    assert 'kindred-gui = "kindred.gui_entrypoint:main"' in content
    assert 'kindred = "kindred.cli:main"' not in content
