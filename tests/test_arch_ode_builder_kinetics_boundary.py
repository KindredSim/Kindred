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


def _has_relative_import_from(
    source: str, *, module: str, level: int, required_names: set[str]
) -> bool:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.module != module or node.level != level:
            continue
        imported = {alias.name for alias in node.names}
        if required_names.issubset(imported):
            return True
    return False


@pytest.mark.unit
def test_arch_ode_builder_does_not_depend_on_simulator_kinetics() -> None:
    pkg = importlib.resources.files("kindred.core")
    ode_builder_source = pkg.joinpath("ode_builder.py").read_text(encoding="utf-8")

    forbidden = _forbidden_imports_in_source(
        ode_builder_source,
        forbidden_module="kindred.core.simulator.kinetics",
    ) + _forbidden_imports_in_source(
        ode_builder_source,
        forbidden_module=".simulator.kinetics",
    )

    assert forbidden == [], (
        "ode_builder.py must not import from kindred.core.simulator.kinetics; "
        f"use kindred.core.kinetics instead. Found: {forbidden}"
    )


@pytest.mark.unit
def test_arch_simulator_kinetics_reexports_core_formulas() -> None:
    pkg = importlib.resources.files("kindred.core.simulator")
    source = pkg.joinpath("kinetics.py").read_text(encoding="utf-8")

    assert _has_relative_import_from(
        source,
        module="kinetics",
        level=2,
        required_names={"arrhenius_rate", "eyring_rate", "K_from_deltaG_eq"},
    ), "simulator/kinetics.py must import shared formulas from kindred.core.kinetics"

    assert "def arrhenius_rate" not in source
    assert "def eyring_rate" not in source
    assert "def K_from_deltaG_eq" not in source

