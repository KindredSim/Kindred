from __future__ import annotations

import ast
import importlib.resources

import pytest


def _forbidden_imports_in_source(source: str, *, forbidden_module: str) -> list[str]:
    tree = ast.parse(source)
    hits: list[str] = []

    forbidden_parent: str | None = None
    forbidden_leaf: str | None = None
    if "." in forbidden_module and not forbidden_module.startswith("."):
        forbidden_parent, forbidden_leaf = forbidden_module.rsplit(".", 1)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == forbidden_module:
                    hits.append(f"import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if node.module == forbidden_module:
                hits.append(f"from {node.module} import ...")
                continue

            if (
                forbidden_parent
                and forbidden_leaf
                and node.module == forbidden_parent
                and any(alias.name == forbidden_leaf for alias in node.names)
            ):
                hits.append(f"from {node.module} import {forbidden_leaf}")
                continue

            if forbidden_module.startswith("."):
                local_name = forbidden_module.lstrip(".")
                if node.module == local_name and node.level == 1:
                    hits.append(f"from .{node.module} import ...")
                    continue

                if node.module is None and node.level == 1:
                    for alias in node.names:
                        if alias.name == local_name:
                            hits.append("from . import " + alias.name)

    return sorted(set(hits))


@pytest.mark.unit
def test_arch_no_import_edge_between_dsl_and_fast_eq() -> None:
    pkg = importlib.resources.files("kindred.core.simulator")

    dsl_source = pkg.joinpath("dsl.py").read_text(encoding="utf-8")
    fast_eq_source = pkg.joinpath("fast_eq.py").read_text(encoding="utf-8")

    dsl_forbidden = _forbidden_imports_in_source(
        dsl_source, forbidden_module=".fast_eq"
    ) + _forbidden_imports_in_source(
        dsl_source, forbidden_module="kindred.core.simulator.fast_eq"
    )
    fast_eq_forbidden = _forbidden_imports_in_source(
        fast_eq_source, forbidden_module=".dsl"
    ) + _forbidden_imports_in_source(
        fast_eq_source, forbidden_module="kindred.core.simulator.dsl"
    )

    assert dsl_forbidden == [], f"dsl.py must not import fast_eq.py: {dsl_forbidden}"
    assert fast_eq_forbidden == [], (
        f"fast_eq.py must not import dsl.py: {fast_eq_forbidden}"
    )
