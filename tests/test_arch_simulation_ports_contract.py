from __future__ import annotations

import ast
from dataclasses import fields
import re
from pathlib import Path
from types import SimpleNamespace

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
    assert "def show_preview_unavailable_for_dirty_state(" in slider_block


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


def test_pending_slider_replay_state_owns_explicit_freshness_identity() -> None:
    from kindred.gui.controllers.simulation_run_state import PendingSliderPreviewLaunchState

    field_names = {field.name for field in fields(PendingSliderPreviewLaunchState)}
    assert "request_id" in field_names
    assert "target_set_ids" in field_names
    assert "replay_generation" in field_names

    state = PendingSliderPreviewLaunchState(
        active=True,
        request_id="42",
        target_set_ids=["set-1", "set-1"],
        replay_generation="7",
    )

    assert state.request_id == 42
    assert state.target_set_ids == ("set-1",)
    assert state.replay_generation == 7


def test_preserved_pending_slider_replay_keeps_launch_identity_but_refreshes_intent() -> None:
    from kindred.gui.controllers.simulation_controller import SimulationController
    from kindred.gui.controllers.simulation_run_state import PendingSliderPreviewLaunchState

    controller = SimulationController.__new__(SimulationController)
    controller._run_state = SimpleNamespace(
        pending_slider_preview_launch=PendingSliderPreviewLaunchState(
            active=True,
            request_id=42,
            target_set_ids=("set-1",),
            handoff_queued=True,
            replay_generation=3,
        ),
        pending_slider_preview_replay_generation=3,
    )

    controller.queue_pending_slider_preview_replay(
        target_set_ids=["set-1"],
        preserve_existing_request=True,
    )

    pending = controller._pending_slider_preview_launch
    assert pending.active is True
    assert pending.request_id == 42
    assert pending.target_set_ids == ("set-1",)
    assert pending.handoff_queued is True
    assert pending.replay_generation == 4


def test_simulation_controller_has_no_pending_replay_setters_that_bypass_generation() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    target = repo_root / "kindred" / "gui" / "controllers" / "simulation_controller.py"
    assert target.is_file(), f"Expected file at {target}"

    tree = ast.parse(target.read_text(encoding="utf-8"), filename=str(target))
    controller_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "SimulationController"
    )
    forbidden_setters = {
        "_pending_slider_simulation",
        "_pending_slider_sim_request_id",
        "_pending_slider_target_set_ids",
        "_pending_slider_handoff_queued",
    }
    setter_names: set[str] = set()
    for node in controller_class.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Attribute) and decorator.attr == "setter":
                value = decorator.value
                if isinstance(value, ast.Name):
                    setter_names.add(value.id)

    assert forbidden_setters.isdisjoint(setter_names)


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


def _attribute_chain(node: ast.AST) -> tuple[str, ...] | None:
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        return None
    parts.append(current.id)
    return tuple(reversed(parts))


def _protocol_method_names(tree: ast.AST, protocol_names: set[str]) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or node.name not in protocol_names:
            continue
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                names.add(item.name)
    return names


def test_simulation_controller_surfaces_use_explicit_ui_subports() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    ports_target = repo_root / "kindred" / "gui" / "ports.py"
    ports_tree = ast.parse(ports_target.read_text(encoding="utf-8"), filename=str(ports_target))

    port_protocols = {
        "SimulationDialogsPort",
        "SimulationSettingsPort",
        "SimulationCacheControlsPort",
        "SimulationRunUiPort",
        "SimulationSliderPort",
        "SimulationBatchPort",
        "SimulationMechanismPort",
        "SimulationSolverPort",
        "SimulationRuntimePort",
        "SimulationResultsPort",
        "SimulationProvenancePort",
        "SimulationMechanismHelpersPort",
    }
    explicit_port_methods = _protocol_method_names(ports_tree, port_protocols)
    assert explicit_port_methods

    target_paths = [
        repo_root / "kindred" / "gui" / "controllers" / "simulation_controller.py",
        repo_root / "kindred" / "gui" / "controllers" / "simulation_run_preparation.py",
        repo_root / "kindred" / "gui" / "controllers" / "simulation_completion_callback.py",
        repo_root / "kindred" / "gui" / "controllers" / "simulation_completion_publication.py",
        repo_root / "kindred" / "gui" / "controllers" / "simulation_error_handling.py",
    ]
    flattened_hits: list[str] = []
    for target in target_paths:
        source = target.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(target))
        lines = source.splitlines()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Attribute):
                continue
            chain = _attribute_chain(node)
            if (
                chain is not None
                and len(chain) == 3
                and chain[:2] in {("self", "ui"), ("self", "_ui"), ("self", "_ports")}
                and chain[2] in explicit_port_methods
            ):
                flattened_hits.append(
                    f"{target.relative_to(repo_root)}:{node.lineno}: {lines[node.lineno - 1].strip()}"
                )

    assert flattened_hits == [], (
        "Simulation controller surfaces must route UI responsibilities through explicit "
        "`SimulationUiPorts` sub-ports, not flattened `self.ui.<method>`/`self._ui.<method>`/"
        "`self._ports.<method>` access.\n"
        + "\n".join(flattened_hits)
    )
