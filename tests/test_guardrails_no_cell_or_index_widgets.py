from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import pytest


pytestmark = [pytest.mark.unit]


FORBIDDEN_ATTRS = {"setCellWidget", "setIndexWidget"}


@dataclass(frozen=True)
class _Hit:
    path: Path
    lineno: int
    attr: str
    line: str


def _iter_python_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.py") if p.is_file())


def _detect_forbidden_calls(source: str, *, filename: str) -> list[tuple[int, str]]:
    tree = ast.parse(source, filename=filename)
    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute):
            continue
        if func.attr not in FORBIDDEN_ATTRS:
            continue
        hits.append((node.lineno, func.attr))
    return hits


def test_guardrail_gui_no_cell_or_index_widgets() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    gui_root = repo_root / "kindred" / "gui"
    assert gui_root.is_dir(), f"Expected GUI root at {gui_root}"

    violations: list[_Hit] = []
    for path in _iter_python_files(gui_root):
        source = path.read_text(encoding="utf-8")
        for lineno, attr in _detect_forbidden_calls(source, filename=str(path)):
            line = source.splitlines()[lineno - 1].rstrip() if lineno <= source.count("\n") + 1 else ""
            violations.append(_Hit(path=path, lineno=lineno, attr=attr, line=line))

    if violations:
        rendered = "\n".join(
            f"{v.path.relative_to(repo_root)}:{v.lineno}: forbidden call to `{v.attr}` :: {v.line.strip()}"
            for v in violations
        )
        raise AssertionError(
            "Guardrail violated: do not use `QTableWidget.setCellWidget(...)` or "
            "`QAbstractItemView.setIndexWidget(...)` in `kindred/gui/**`.\n"
            "Prefer model/delegate patterns.\n\n"
            f"Violations:\n{rendered}"
        )


def test_guardrail_gui_no_cell_or_index_widgets_detector_selfcheck() -> None:
    good = """
def f(w):
    x = "setCellWidget("
    # w.setCellWidget(0, 0, None)
    return w
"""
    assert _detect_forbidden_calls(good, filename="<selfcheck-good>") == []

    bad = """
def f(w):
    w.setCellWidget(0, 0, None)
    w.setIndexWidget(None, None)
"""
    hits = _detect_forbidden_calls(bad, filename="<selfcheck-bad>")
    assert {attr for _, attr in hits} == {"setCellWidget", "setIndexWidget"}
