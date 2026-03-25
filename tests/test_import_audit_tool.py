import sys

import pytest

from tools.audit import import_audit

pytestmark = [pytest.mark.unit]


def test_strongly_connected_components_finds_cycles():
    graph = {
        "a": {"b"},
        "b": {"a"},
        "c": set(),
    }
    nodes = ["a", "b", "c"]
    sccs = import_audit.strongly_connected_components(graph, nodes)
    normalized = [tuple(sorted(comp)) for comp in sccs]
    assert ("a", "b") in normalized


def test_import_audit_main_smoke(tmp_path, monkeypatch, capsys):
    repo_root = tmp_path / "repo"
    (repo_root / "kindred").mkdir(parents=True, exist_ok=True)
    (repo_root / "kindred" / "__init__.py").write_text("", encoding="utf-8")
    (repo_root / "kindred" / "a.py").write_text("from kindred import b\n", encoding="utf-8")
    (repo_root / "kindred" / "b.py").write_text("VALUE = 1\n", encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "import_audit.py",
            "--repo-root",
            str(repo_root),
            "--package",
            "kindred",
        ],
    )

    rc = import_audit.main()
    assert rc == 0

    captured = capsys.readouterr().out
    assert "A1 CoreGUIImport" in captured
    assert "A2 ForbiddenDslFastEqCycle" in captured
