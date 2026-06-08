from __future__ import annotations

from pathlib import Path
import pytest

pytestmark = pytest.mark.unit


try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _project_table() -> dict:
    pyproject_path = _repo_root() / "pyproject.toml"
    return tomllib.loads(pyproject_path.read_text(encoding="utf-8"))["project"]


def test_test_extra_is_self_sufficient_for_gui_suite() -> None:
    optional_deps = _project_table()["optional-dependencies"]
    test_extra = optional_deps["test"]

    assert set(test_extra) == {
        "pytest",
        "pytest-qt",
        "tomli; python_version < '3.11'",
    }


def test_requirements_txt_matches_runtime_dependencies_and_split_files_are_absent() -> None:
    root = _repo_root()
    requirements_path = root / "requirements.txt"
    lines = [
        line.strip()
        for line in requirements_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]

    assert lines == _project_table()["dependencies"]
    assert not (root / "requirements-core.txt").exists()
    assert not (root / "requirements-gui.txt").exists()
