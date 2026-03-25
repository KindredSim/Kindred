from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import pytest


pytestmark = [pytest.mark.unit]


@dataclass(frozen=True)
class _Hit:
    path: Path
    lineno: int
    kind: str
    detail: str
    line: str


def _iter_python_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.py") if p.is_file())


def _is_type_checking_test(expr: ast.expr) -> bool:
    # Allow `if TYPE_CHECKING:` and `if typing.TYPE_CHECKING:` blocks at module scope.
    if isinstance(expr, ast.Name) and expr.id == "TYPE_CHECKING":
        return True
    if isinstance(expr, ast.Attribute) and expr.attr == "TYPE_CHECKING":
        if isinstance(expr.value, ast.Name) and expr.value.id == "typing":
            return True
    return False


def _scan_module_scope_imports(source: str, *, filename: str) -> list[tuple[int, str, str]]:
    """
    Return (lineno, kind, detail) for forbidden module-scope imports.

    Rules:
    - Only consider module-scope statements (including module-scope if/try/with blocks).
    - Do not traverse into FunctionDef/ClassDef bodies.
    - Allow imports inside `if TYPE_CHECKING:` blocks only.
    """
    tree = ast.parse(source, filename=filename)
    hits: list[tuple[int, str, str]] = []

    def scan_statements(statements: list[ast.stmt], *, in_type_checking: bool) -> None:
        for stmt in statements:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue

            if isinstance(stmt, ast.Import):
                if not in_type_checking:
                    for alias in stmt.names:
                        if alias.name == "concurrent.futures.process":
                            hits.append((stmt.lineno, "import", "import concurrent.futures.process"))
                continue

            if isinstance(stmt, ast.ImportFrom):
                if in_type_checking:
                    continue
                module = stmt.module or ""
                imported_names = {a.name for a in stmt.names}
                if module == "concurrent.futures" and (
                    "ProcessPoolExecutor" in imported_names or "process" in imported_names
                ):
                    hits.append((stmt.lineno, "importfrom", f"from {module} import {sorted(imported_names)!r}"))
                elif module == "concurrent.futures.process":
                    hits.append((stmt.lineno, "importfrom", f"from {module} import {sorted(imported_names)!r}"))
                elif module == "multiprocessing" and "get_context" in imported_names:
                    hits.append((stmt.lineno, "importfrom", f"from {module} import {sorted(imported_names)!r}"))
                continue

            if isinstance(stmt, ast.If):
                if _is_type_checking_test(stmt.test):
                    scan_statements(stmt.body, in_type_checking=True)
                    scan_statements(stmt.orelse, in_type_checking=in_type_checking)
                else:
                    scan_statements(stmt.body, in_type_checking=in_type_checking)
                    scan_statements(stmt.orelse, in_type_checking=in_type_checking)
                continue

            if isinstance(stmt, ast.Try):
                scan_statements(stmt.body, in_type_checking=in_type_checking)
                for handler in stmt.handlers:
                    scan_statements(handler.body, in_type_checking=in_type_checking)
                scan_statements(stmt.orelse, in_type_checking=in_type_checking)
                scan_statements(stmt.finalbody, in_type_checking=in_type_checking)
                continue

            if isinstance(stmt, (ast.With, ast.For, ast.While)):
                scan_statements(stmt.body, in_type_checking=in_type_checking)
                scan_statements(stmt.orelse, in_type_checking=in_type_checking)
                continue

            if isinstance(stmt, ast.Match):
                for case in stmt.cases:
                    scan_statements(case.body, in_type_checking=in_type_checking)
                continue

    scan_statements(tree.body, in_type_checking=False)
    return hits


def test_guardrail_gui_no_processpool_imports_at_module_scope() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    gui_root = repo_root / "kindred" / "gui"
    assert gui_root.is_dir(), f"Expected GUI root at {gui_root}"

    violations: list[_Hit] = []
    for path in _iter_python_files(gui_root):
        source = path.read_text(encoding="utf-8")
        for lineno, kind, detail in _scan_module_scope_imports(source, filename=str(path)):
            line = source.splitlines()[lineno - 1].rstrip() if lineno <= source.count("\n") + 1 else ""
            violations.append(_Hit(path=path, lineno=lineno, kind=kind, detail=detail, line=line))

    if violations:
        rendered = "\n".join(
            f"{v.path.relative_to(repo_root)}:{v.lineno}: forbidden module-scope import ({v.kind}) "
            f"{v.detail} :: {v.line.strip()}"
            for v in violations
        )
        raise AssertionError(
            "Guardrail violated: `kindred/gui/**` must not import process-pool machinery at module scope.\n"
            "- Keep process pool imports lazy (inside functions), so GUI startup/import never touches "
            "`concurrent.futures.process`.\n"
            "- Allowed exception: imports inside `if TYPE_CHECKING:` blocks.\n\n"
            f"Violations:\n{rendered}"
        )


def test_guardrail_gui_no_processpool_imports_detector_selfcheck() -> None:
    bad1 = "from concurrent.futures import ProcessPoolExecutor\n"
    assert _scan_module_scope_imports(bad1, filename="<selfcheck-bad1>") != []

    bad2 = "import concurrent.futures.process as cfp\n"
    assert _scan_module_scope_imports(bad2, filename="<selfcheck-bad2>") != []

    bad3 = "from multiprocessing import get_context\n"
    assert _scan_module_scope_imports(bad3, filename="<selfcheck-bad3>") != []

    ok1 = """
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from concurrent.futures import ProcessPoolExecutor
"""
    assert _scan_module_scope_imports(ok1, filename="<selfcheck-ok1>") == []

    ok2 = """
def f():
    from concurrent.futures import ProcessPoolExecutor
    return ProcessPoolExecutor
"""
    assert _scan_module_scope_imports(ok2, filename="<selfcheck-ok2>") == []
