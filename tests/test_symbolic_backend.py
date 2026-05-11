from __future__ import annotations

import pathlib
import tomllib

import pytest


pytestmark = pytest.mark.unit


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_sympy_is_exact_runtime_dependency_on_all_runtime_surfaces():
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project_deps = set(pyproject["project"]["dependencies"])
    requirements = {
        line.strip()
        for line in (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }

    assert "sympy==1.14.0" in project_deps
    assert "sympy==1.14.0" in requirements


def test_direct_sympy_imports_are_confined_to_symbolic_backend_package():
    violations: list[str] = []
    for root_name in ("kindred",):
        for path in (REPO_ROOT / root_name).rglob("*.py"):
            rel = path.relative_to(REPO_ROOT).as_posix()
            text = path.read_text(encoding="utf-8")
            if "import sympy" not in text and "from sympy" not in text:
                continue
            if rel.startswith("kindred/core/symbolic/"):
                continue
            violations.append(rel)

    assert violations == []


def test_symbolic_backend_metadata_reports_expected_sympy_version():
    from kindred.core.symbolic.backend import get_symbolic_backend_metadata

    metadata = get_symbolic_backend_metadata()

    assert metadata.backend_name == "sympy"
    assert metadata.backend_version == "1.14.0"
    assert metadata.profile_version


def test_symbolic_artifact_identity_has_stable_no_artifact_marker():
    from kindred.core.symbolic.artifacts import SymbolicArtifactIdentity
    from kindred.core.symbolic.backend import get_symbolic_backend_metadata

    metadata = get_symbolic_backend_metadata()
    marker = SymbolicArtifactIdentity.none(metadata)

    assert marker.kind == "none"
    assert marker.backend_name == "sympy"
    assert marker.backend_version == "1.14.0"
    assert marker.fingerprint
    assert marker.to_payload()["kind"] == "none"
