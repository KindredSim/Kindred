"""Fail on kindred.core -> kindred.gui imports and on internal import cycles.

Every runtime import counts, including imports inside functions; imports under
``if TYPE_CHECKING:`` do not.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _module_index() -> dict[str, tuple[Path, bool]]:
    """Map every module under kindred/ to its file and whether it is a package."""
    index: dict[str, tuple[Path, bool]] = {}
    for path in sorted((REPO_ROOT / "kindred").rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        parts = list(path.relative_to(REPO_ROOT).with_suffix("").parts)
        is_package = parts[-1] == "__init__"
        if is_package:
            parts.pop()
        index[".".join(parts)] = (path, is_package)
    return index


def _is_type_checking(test: ast.expr) -> bool:
    if isinstance(test, ast.Name):
        return test.id == "TYPE_CHECKING"
    if isinstance(test, ast.Attribute):
        return test.attr == "TYPE_CHECKING" and isinstance(test.value, ast.Name) and test.value.id == "typing"
    if isinstance(test, ast.BoolOp) and isinstance(test.op, ast.And):
        return any(_is_type_checking(value) for value in test.values)
    return False


def _runtime_imports(node: ast.AST):
    if isinstance(node, ast.If) and _is_type_checking(node.test):
        for child in node.orelse:
            yield from _runtime_imports(child)
        return
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        yield node
        return
    for child in ast.iter_child_nodes(node):
        yield from _runtime_imports(child)


def _targets(node: ast.Import | ast.ImportFrom, module: str, is_package: bool, index) -> list[str]:
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names if alias.name in index]
    if node.level:
        package = (module if is_package else module.rpartition(".")[0]).split(".")
        base_parts = package[: len(package) - (node.level - 1)]
        base = ".".join([*base_parts, node.module] if node.module else base_parts)
    else:
        base = node.module or ""
    found = []
    for alias in node.names:
        submodule = f"{base}.{alias.name}"
        if alias.name != "*" and submodule in index:
            found.append(submodule)
        elif base in index:
            found.append(base)
    return found


def _cycles(graph: dict[str, set[str]]) -> list[list[str]]:
    """Return strongly connected components with more than one module (Tarjan)."""
    order: dict[str, int] = {}
    low: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    components: list[list[str]] = []
    sys.setrecursionlimit(max(10_000, sys.getrecursionlimit()))

    def visit(node: str) -> None:
        order[node] = low[node] = len(order)
        stack.append(node)
        on_stack.add(node)
        for target in sorted(graph[node]):
            if target not in order:
                visit(target)
                low[node] = min(low[node], low[target])
            elif target in on_stack:
                low[node] = min(low[node], order[target])
        if low[node] == order[node]:
            component = []
            while True:
                member = stack.pop()
                on_stack.discard(member)
                component.append(member)
                if member == node:
                    break
            if len(component) > 1:
                components.append(sorted(component))

    for node in sorted(graph):
        if node not in order:
            visit(node)
    return components


def main() -> int:
    index = _module_index()
    graph: dict[str, set[str]] = {module: set() for module in index}
    problems: list[str] = []
    for module, (path, is_package) in index.items():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in _runtime_imports(tree):
            for target in _targets(node, module, is_package, index):
                if target == module:
                    continue
                graph[module].add(target)
                if module.startswith("kindred.core") and target.startswith("kindred.gui"):
                    problems.append(
                        f"{path.relative_to(REPO_ROOT)}:{node.lineno}: kindred.core must not import {target}"
                    )
    problems.extend(f"import cycle: {' <-> '.join(component)}" for component in _cycles(graph))
    for problem in problems:
        print(problem)
    print(f"import graph: {len(index)} modules, {len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
