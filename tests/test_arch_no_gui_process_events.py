from __future__ import annotations

import ast
from pathlib import Path

import pytest

ALLOWED = {
    "kindred/gui/utils.py": {22},
    "kindred/gui/widgets/state_network_editor.py": {244},
    "kindred/gui/widgets/computational_mode_dialog.py": {599},
}


def _find_process_events_calls(source: str) -> list[int]:
    tree = ast.parse(source)
    hits: list[int] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "processEvents":
            hits.append(int(getattr(node, "lineno", 0) or 0))

    return sorted(set(hits))


@pytest.mark.unit
def test_arch_no_qt_process_events_in_gui_layer() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    target_paths = list((repo_root / "kindred" / "gui").rglob("*.py"))

    failures: list[str] = []
    for path in target_paths:
        source = path.read_text(encoding="utf-8")
        hits = _find_process_events_calls(source)
        rel = path.relative_to(repo_root).as_posix()
        allowed_lines = ALLOWED.get(rel, set())
        unapproved = [ln for ln in hits if ln not in allowed_lines]
        if unapproved:
            failures.append(f"{rel}:{unapproved}")

    assert failures == [], "Unapproved processEvents() call found: " + ", ".join(failures)
