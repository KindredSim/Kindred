from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import pytest


pytestmark = [pytest.mark.unit]


TARGET_FUNCTIONS = {"_update_plot", "set_data", "_draw_overlay_series"}


@dataclass(frozen=True)
class _Hit:
    func_name: str
    lineno: int
    line: str


def _is_self_plotitem_clear_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if not isinstance(func, ast.Attribute):
        return False
    if func.attr != "clear":
        return False
    value = func.value
    if not isinstance(value, ast.Attribute):
        return False
    if value.attr != "_plot_item":
        return False
    base = value.value
    return isinstance(base, ast.Name) and base.id == "self"


def _scan_function_for_clear(fn: ast.FunctionDef) -> list[int]:
    return sorted({node.lineno for node in ast.walk(fn) if _is_self_plotitem_clear_call(node)})


def test_guardrail_pyqtgraph_no_plotitem_clear_in_update() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    target = repo_root / "kindred" / "gui" / "widgets" / "pyqtgraph_plot_panel_impl.py"
    assert target.is_file(), f"Expected file at {target}"

    source = target.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(target))

    functions: dict[str, ast.FunctionDef] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in TARGET_FUNCTIONS:
            functions[node.name] = node

    assert (
        "_update_plot" in functions
    ), "Guardrail expectation changed: `_update_plot` not found; update TARGET_FUNCTIONS to cover the routine update path."

    violations: list[_Hit] = []
    for name, fn in functions.items():
        for lineno in _scan_function_for_clear(fn):
            line = source.splitlines()[lineno - 1].rstrip() if lineno <= source.count("\n") + 1 else ""
            violations.append(_Hit(func_name=name, lineno=lineno, line=line))

    if violations:
        rendered = "\n".join(
            f"{target.relative_to(repo_root)}:{v.lineno}: `{v.func_name}` calls `self._plot_item.clear()` :: {v.line.strip()}"
            for v in violations
        )
        raise AssertionError(  # nosec B608 - this is a test message, not SQL construction
            "Guardrail violated: routine plot updates must not call `self._plot_item.clear()`.\n"  # nosec B608
            "Do not clear/recreate on each update; reuse plot items and update via `setData`.\n"
            "Allowed: explicit reset methods (e.g. `clear()`), not routine update path.\n\n"
            f"Violations:\n{rendered}"
        )


def test_guardrail_pyqtgraph_no_plotitem_clear_detector_selfcheck() -> None:
    ok = """
class W:
    def clear(self):
        self._plot_item.clear()
"""
    ok_tree = ast.parse(ok, filename="<selfcheck-ok>")
    ok_fns = {n.name: n for n in ast.walk(ok_tree) if isinstance(n, ast.FunctionDef)}
    assert _scan_function_for_clear(ok_fns["clear"]) == [4]

    bad = """
class W:
    def _update_plot(self):
        self._plot_item.clear()
"""
    bad_tree = ast.parse(bad, filename="<selfcheck-bad>")
    bad_fns = {n.name: n for n in ast.walk(bad_tree) if isinstance(n, ast.FunctionDef)}
    assert _scan_function_for_clear(bad_fns["_update_plot"]) == [4]
