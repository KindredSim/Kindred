from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest


pytestmark = [pytest.mark.unit]


def _protocol_block(source: str, protocol_name: str) -> str:
    pattern = re.compile(
        rf"^class {re.escape(protocol_name)}\(Protocol\):\n(?P<body>(?:^(?:    |\n).*\n?)*)",
        re.MULTILINE,
    )
    match = pattern.search(source)
    assert match, f"Expected protocol block for {protocol_name}"
    return match.group("body")


def test_simulation_ports_assign_mechanism_snapshots_to_mechanism_helpers_port() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    target = repo_root / "kindred" / "gui" / "ports.py"
    assert target.is_file(), f"Expected file at {target}"

    source = target.read_text(encoding="utf-8")
    slider_block = _protocol_block(source, "SimulationSliderPort")
    mechanism_helpers_block = _protocol_block(source, "SimulationMechanismHelpersPort")

    assert "def last_mechanism(" not in slider_block
    assert "def last_mechanism_context(" not in slider_block
    assert "def last_mechanism(" in mechanism_helpers_block
    assert "def last_mechanism_context(" in mechanism_helpers_block


def test_simulation_mechanism_port_exposes_explicit_run_readiness_query() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    target = repo_root / "kindred" / "gui" / "ports.py"
    assert target.is_file(), f"Expected file at {target}"

    source = target.read_text(encoding="utf-8")
    mechanism_block = _protocol_block(source, "SimulationMechanismPort")

    assert "def auto_lock_for_run(" in mechanism_block
    assert "def is_mechanism_ready_for_run(" in mechanism_block


def test_simulation_slider_port_exposes_preview_validity_query() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    target = repo_root / "kindred" / "gui" / "ports.py"
    assert target.is_file(), f"Expected file at {target}"

    source = target.read_text(encoding="utf-8")
    slider_block = _protocol_block(source, "SimulationSliderPort")

    assert "def is_mechanism_valid_for_preview(" in slider_block


def test_slider_preview_lifecycle_port_is_explicit_and_bounded() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    target = repo_root / "kindred" / "gui" / "ports.py"
    assert target.is_file(), f"Expected file at {target}"

    source = target.read_text(encoding="utf-8")
    lifecycle_block = _protocol_block(source, "SliderPreviewLifecyclePort")

    expected_methods = {
        "submit_slider_preview_replay_intent",
        "clear_pending_slider_preview_replay",
        "invalidate_slider_preview_work",
        "launch_pending_slider_preview_replay",
    }
    actual_methods = set(re.findall(r"def\s+([A-Za-z_][A-Za-z0-9_]*)\(", lifecycle_block))

    assert actual_methods == expected_methods
    assert "__getattr__" not in lifecycle_block


def test_build_simulation_plumbing_wires_truthful_explicit_owners() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    target = repo_root / "kindred" / "gui" / "app_wiring.py"
    assert target.is_file(), f"Expected file at {target}"

    source = target.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(target))

    build_fn = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "build_simulation_plumbing"
    )
    simulation_ui_ports_call = next(
        node
        for node in ast.walk(build_fn)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "SimulationUiPorts"
    )
    keyword_map = {kw.arg: kw.value for kw in simulation_ui_ports_call.keywords if kw.arg is not None}

    expected_attrs = {
        "slider": "_preview_session",
        "runtime": "_variable_runtime",
        "mechanism_helpers": "_mechanism_helpers",
    }
    for keyword, expected_attr in expected_attrs.items():
        value = keyword_map.get(keyword)
        assert isinstance(value, ast.Attribute), f"Expected `{keyword}` to be wired from a MainWindow attribute."
        assert isinstance(value.value, ast.Name) and value.value.id == "main_window", (
            f"Expected `{keyword}` to be sourced from `main_window.<owner>`."
        )
        assert value.attr == expected_attr, (
            f"Expected `{keyword}` to be wired to `main_window.{expected_attr}`, got `main_window.{value.attr}`."
        )


def test_simulation_ui_ports_do_not_provide_generic_cross_port_fallback() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    target = repo_root / "kindred" / "gui" / "ports.py"
    assert target.is_file(), f"Expected file at {target}"

    source = target.read_text(encoding="utf-8")

    assert "def __getattr__(" not in source
