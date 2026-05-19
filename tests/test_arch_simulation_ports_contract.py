from __future__ import annotations

import ast
from dataclasses import fields
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
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


def test_explicit_cache_reconciliation_clear_removes_display_selection_truth() -> None:
    from kindred.core.batch_simulation_cache import BatchSimulationCache

    cache = BatchSimulationCache()
    cache.active_cache_key = "cache-key"
    cache.active_cache_preview_token = "preview-token"
    cache.active_cache_preview_scope_set_ids = ("set-a",)
    cache.active_cache_valid_set_ids = ("set-a",)
    cache.active_cache_invalidated_set_ids = ("set-b",)
    cache.active_preview_cache_key = "preview-key"
    cache.active_preview_scope_set_ids = ("set-a",)
    cache.last_display_selection = ["set-a"]
    cache.active_batch_set = "Set A"
    cache.active_batch_set_id = "set-a"

    cache.apply_explicit_cache_reconciliation(
        clear_active_selection_state=True,
        active_cache_key="new-cache",
        active_cache_preview_token="new-preview",
        active_cache_preview_scope_set_ids=("set-c",),
        active_cache_valid_set_ids=("set-c",),
        active_cache_invalidated_set_ids=(),
    )

    assert cache.active_cache_key is None
    assert cache.active_cache_preview_token is None
    assert cache.active_cache_preview_scope_set_ids is None
    assert cache.active_cache_valid_set_ids is None
    assert cache.active_cache_invalidated_set_ids is None
    assert cache.active_preview_cache_key is None
    assert cache.active_preview_scope_set_ids is None
    assert cache.last_display_selection == []
    assert cache.active_batch_set is None
    assert cache.active_batch_set_id is None


def test_explicit_completion_cache_publication_carries_provenance_for_non_primary_results() -> None:
    from kindred.gui.controllers.simulation_completion_publication import (
        CompletionCallbackState,
        CompletionResultState,
        SimulationCompletionPublicationDependencies,
        SimulationCompletionPublicationOwner,
    )

    captured: dict[str, object] = {}
    owner = SimulationCompletionPublicationOwner(
        ui=SimpleNamespace(),
        batch_context_owner=SimpleNamespace(
            include_mechanism_in_result_payload=lambda **_kwargs: False,
        ),
        batch_cache=SimpleNamespace(),
        cache_admin=SimpleNamespace(
            publish_completion_cache=lambda **kwargs: captured.update(kwargs),
        ),
        completion_policy=SimpleNamespace(),
        lifecycle_effect_owner=SimpleNamespace(),
        result_materialization_owner=SimpleNamespace(),
        dependencies=SimulationCompletionPublicationDependencies(
            apply_lifecycle_effects=lambda *_args, **_kwargs: None,
            record_nonfatal_exception=lambda *_args, **_kwargs: None,
            queue_slider_plot_update=lambda *_args, **_kwargs: None,
            finalize_explicit_batch_dirty_reset=lambda *_args, **_kwargs: {},
            flush_slider_plot_updates=lambda *_args, **_kwargs: None,
            show_scoped_batch_failure_summary=lambda *_args, **_kwargs: None,
            has_deferred_preview_replay_intent=lambda: False,
            start_next_batch_simulation=lambda: None,
        ),
    )
    t = np.asarray([0.0, 1.0], dtype=float)
    series = {"A": np.asarray([1.0, 0.4], dtype=float)}
    completion = CompletionResultState(
        t=t,
        Y=np.asarray([[1.0, 0.4]], dtype=float),
        species_names=["A"],
        algebra_scalars={},
        algebra_errors=[],
        solver_provenance={
            "launch_provenance": {
                "temperature_K": 298.15,
                "simulation_time": 1.0,
                "num_points_requested": 2,
            }
        },
        mechanism=None,
        base_species_count=None,
        mechanism_text="A -> B ; k=1.0",
        solver_config={"solver": "RK45", "rtol": 1e-6, "atol": 1e-12},
        warnings=[],
        fallback_occurred=False,
        fallback_message=None,
        series=series,
        is_primary=False,
        energy_mode=False,
        redraw_valid_set_ids=None,
        has_redraw_subset=False,
    )
    state = CompletionCallbackState(
        run_id=1,
        request_id=1,
        batch_set="Set B",
        batch_set_id="set-b",
        cache_key="cache-key",
        policy_context=None,
        ctx={},
        shutdown_requested=False,
        is_preview=False,
        slider_triggered=False,
        explicit_batch_coalescing=False,
    )

    owner.publish_cache_entry(completion, state)

    provenance = captured.get("completion_provenance")
    assert isinstance(provenance, dict)
    assert provenance["mechanism_text"] == "A -> B ; k=1.0"
    assert provenance["temperature_K"] == 298.15
    assert provenance["species_names"] == ["A"]
    assert captured["set_id"] == "set-b"


def test_completion_publication_has_no_multi_set_partial_direct_fallback() -> None:
    from kindred.gui.controllers.results_controller import ResultsController

    direct_completion_provenance = {"mechanism_text": "A -> B ; k=1.0"}

    assert (
        ResultsController._fresh_completion_cache_miss_can_publish_directly(
            reason="no_cached_results",
            batch_set_id="set-a",
            selected_sets=["set-a"],
            direct_completion_provenance=direct_completion_provenance,
        )
        is True
    )
    assert (
        ResultsController._fresh_completion_cache_miss_can_publish_directly(
            reason="no_cached_results",
            batch_set_id="set-a",
            selected_sets=["set-a", "set-b"],
            direct_completion_provenance=direct_completion_provenance,
        )
        is False
    )


def test_completed_run_display_coverage_is_typed_and_truthful() -> None:
    from kindred.gui.controllers.batch_run_context_owner import (
        BatchContextSeed,
        BatchCallbackContext,
        BatchRunContextOwner,
    )
    from kindred.gui.ports import CompletionDisplayEntry, CompletedRunDisplayIntent

    owner = BatchRunContextOwner()
    explicit_intent = CompletedRunDisplayIntent(
        set_ids=("set-a", "set-b"),
        labels_by_set_id={"set-a": "Intent A", "set-b": "Intent B"},
        primary_set_id="set-b",
        cache_key="cache-key",
        run_id=22,
        request_id=11,
    )
    owner.load_context(
        BatchContextSeed(
            active=True,
            request_id=11,
            run_id=22,
            cache_key="cache-key",
            queue_ids=("set-a", "set-b"),
            queue_names=("Raw A", "Raw B"),
            primary_set_id="set-a",
            total=2,
            completed_run_display_intent=explicit_intent,
        )
    )

    incomplete_context = owner.record_completion_display_entry(
        None,
        set_id="set-a",
        label="Set A",
        entry=CompletionDisplayEntry(
            set_id="set-a",
            label="Set A",
            t=np.asarray([0.0, 1.0], dtype=float),
            series={"A": np.asarray([1.0, 0.5], dtype=float)},
            algebra_scalars={},
            solver_provenance={},
            mechanism_text="A -> B ; k=1.0",
            solver_config={},
            warnings=(),
            completion_provenance={"mechanism_text": "A -> B ; k=1.0"},
        ),
    )
    incomplete = owner.completed_run_display_coverage(incomplete_context)

    assert isinstance(incomplete.intent, CompletedRunDisplayIntent)
    assert incomplete.intent == explicit_intent
    assert incomplete.intent.labels_by_set_id["set-a"] == "Intent A"
    assert incomplete.intent.primary_set_id == "set-b"
    assert incomplete.transaction is None
    assert incomplete.missing_set_ids == ("set-b",)

    complete_context = owner.record_completion_display_entry(
        incomplete_context,
        set_id="set-b",
        label="Set B",
        entry=CompletionDisplayEntry(
            set_id="set-b",
            label="Set B",
            t=np.asarray([0.0, 1.0], dtype=float),
            series={"A": np.asarray([0.5, 0.25], dtype=float)},
            algebra_scalars={},
            solver_provenance={},
            mechanism_text="A -> B ; k=1.0",
            solver_config={},
            warnings=(),
            completion_provenance={"mechanism_text": "A -> B ; k=1.0"},
        ),
    )
    complete = owner.completed_run_display_coverage(complete_context)

    assert complete.missing_set_ids == ()
    assert complete.transaction is not None
    assert complete.transaction.intent == incomplete.intent
    assert [
        entry.set_id for entry in complete.transaction.completion_entries
    ] == ["set-a", "set-b"]
    assert all(
        isinstance(entry, CompletionDisplayEntry)
        for entry in complete.transaction.completion_entries
    )
    assert "completed_run_display_intent" in owner.callback_context_snapshot()
    assert "completed_run_display_intent" in BatchCallbackContext(
        active=True,
        request_id=11,
        run_id=22,
        cache_key="cache-key",
        fast_mode=False,
        parallel=True,
        keep_lane_pool_alive=False,
        queue_ids=("set-a", "set-b"),
        queue_names=("Set A", "Set B"),
        total=2,
        primary_set_id="set-a",
        completed_run_display_intent=incomplete.intent,
    ).to_context()
    raw_context_without_typed_intent = {
        "active": True,
        "request_id": 11,
        "run_id": 22,
        "cache_key": "cache-key",
        "queue_ids": ["set-a", "set-b"],
        "queue_names": ["Set A", "Set B"],
        "primary_set_id": "set-a",
        "total": 2,
    }
    raw_coverage = owner.completed_run_display_coverage(raw_context_without_typed_intent)
    assert raw_coverage.intent is None
    assert raw_coverage.reason == "no_display_intent"
    raw_entry_context = dict(incomplete_context)
    raw_entry_context["completion_display_entries_by_set_id"] = {
        "set-a": {
            "set_id": "set-a",
            "label": "Set A",
            "entry": {
                "t": np.asarray([0.0, 1.0], dtype=float),
                "series": {"A": np.asarray([1.0, 0.5], dtype=float)},
                "completion_provenance": {"mechanism_text": "A -> B ; k=1.0"},
            },
        },
        "set-b": {
            "set_id": "set-b",
            "label": "Set B",
            "entry": {
                "t": np.asarray([0.0, 1.0], dtype=float),
                "series": {"A": np.asarray([0.5, 0.25], dtype=float)},
                "completion_provenance": {"mechanism_text": "A -> B ; k=1.0"},
            },
        },
    }
    raw_entry_coverage = owner.completed_run_display_coverage(raw_entry_context)
    assert raw_entry_coverage.transaction is None
    assert raw_entry_coverage.missing_set_ids == ("set-a", "set-b")


def test_simulation_results_port_exposes_completed_run_display_transaction_endpoint() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    target = repo_root / "kindred" / "gui" / "ports.py"
    assert target.is_file(), f"Expected file at {target}"

    source = target.read_text(encoding="utf-8")
    results_block = _protocol_block(source, "SimulationResultsPort")

    assert "def publish_completed_run_display_transaction(" in results_block


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


def test_main_plot_display_truth_mutation_routes_through_display_owner_boundaries() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    targets = [
        repo_root / "kindred" / "gui" / "main_window.py",
        repo_root / "kindred" / "gui" / "controllers" / "simulation_controller.py",
    ]
    forbidden_attrs = {
        "active_cache_valid_set_ids",
        "active_cache_invalidated_set_ids",
        "active_cache_preview_scope_set_ids",
        "active_cache_preview_token",
    }
    forbidden_calls = {
        "clear_active_preview_selection_state",
        "record_active_result_cache_staleness",
        "reset_runtime_state",
    }
    violations: list[str] = []

    for target in targets:
        source = target.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(target))
        lines = source.splitlines()
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                assignment_targets = list(node.targets)
            elif isinstance(node, ast.AnnAssign):
                assignment_targets = [node.target]
            elif isinstance(node, ast.AugAssign):
                assignment_targets = [node.target]
            else:
                assignment_targets = []
            for assignment_target in assignment_targets:
                for assignment_node in ast.walk(assignment_target):
                    if isinstance(assignment_node, ast.Attribute) and assignment_node.attr in forbidden_attrs:
                        violations.append(
                            f"{target.relative_to(repo_root)}:{assignment_node.lineno}: "
                            f"{lines[assignment_node.lineno - 1].strip()}"
                        )
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            receiver_chain = _attribute_chain(node.func.value)
            if (
                node.func.attr in forbidden_calls
                and receiver_chain is not None
                and receiver_chain[-1] in {"batch_cache", "_batch_cache"}
            ):
                violations.append(
                    f"{target.relative_to(repo_root)}:{node.lineno}: {lines[node.lineno - 1].strip()}"
                )

    assert violations == [], (
        "Main-plot display truth must be mutated through ResultsController, SimulationBatchOwner, "
        "or BatchSimulationCache owner methods, not directly from MainWindow or SimulationController.\n"
        + "\n".join(violations)
    )


def test_main_plot_display_transactions_are_not_owned_by_window_or_batch_adapter() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    targets = {
        repo_root / "kindred" / "gui" / "main_window.py": {
            "clear_main_plot_display_state",
            "restore_preserved_batch_display_outcome",
            "display_cached_batch_selection_outcome",
            "display_resolved_batch_selection_outcome",
            "display_transaction_snapshot",
            "restore_display_transaction_snapshot",
        },
        repo_root / "kindred" / "gui" / "simulation_batch_owner.py": {
            "clear_main_plot_display_state",
            "restore_preserved_batch_display_outcome",
            "display_cached_batch_selection_outcome",
            "display_resolved_batch_selection_outcome",
            "display_transaction_snapshot",
            "restore_display_transaction_snapshot",
        },
    }
    violations: list[str] = []

    for target, forbidden_calls in targets.items():
        source = target.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(target))
        lines = source.splitlines()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr not in forbidden_calls:
                continue
            violations.append(
                f"{target.relative_to(repo_root)}:{node.lineno}: {lines[node.lineno - 1].strip()}"
            )

    assert violations == [], (
        "MainWindow and SimulationBatchOwner must not apply or roll back main-plot display "
        "transactions directly. ResultsController owns display transaction application and "
        "rollback; batch-facing owners may resolve selection state and request publication only.\n"
        + "\n".join(violations)
    )


def test_main_window_does_not_expose_display_selection_forwarding_facades() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    target = repo_root / "kindred" / "gui" / "main_window.py"
    source = target.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(target))
    forbidden_methods = {
        "active_batch_selection",
        "set_active_batch_selection",
        "clear_display_selection_state",
    }
    violations = [
        f"{target.relative_to(repo_root)}:{node.lineno}: {node.name}"
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in forbidden_methods
    ]

    assert violations == [], (
        "MainWindow must not expose display-selection forwarding facades. Display selection "
        "state belongs to the batch/cache fact owner and display publication belongs to "
        "ResultsController; MainWindow may only wire events.\n"
        + "\n".join(violations)
    )


def test_batch_owner_does_not_depend_on_results_controller_display_owner() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    target = repo_root / "kindred" / "gui" / "simulation_batch_owner.py"
    source = target.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(target))
    lines = source.splitlines()
    violations: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            module = str(getattr(node, "module", "") or "")
            names = [alias.name for alias in getattr(node, "names", [])]
            if module == "kindred.gui.controllers.results_controller" or any(
                name == "kindred.gui.controllers.results_controller" for name in names
            ):
                violations.append(f"{target.relative_to(repo_root)}:{node.lineno}: {lines[node.lineno - 1].strip()}")
        if isinstance(node, ast.Attribute) and node.attr == "results_controller":
            violations.append(f"{target.relative_to(repo_root)}:{node.lineno}: {lines[node.lineno - 1].strip()}")

    assert violations == [], (
        "SimulationBatchOwner is a batch/cache/workspace fact source. It must not import or "
        "reach through ResultsController; display transaction policy belongs to ResultsController "
        "and shared fact/result value objects belong at a neutral port boundary.\n"
        + "\n".join(violations)
    )


def test_results_controller_does_not_sync_mechanism_controls() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    target = repo_root / "kindred" / "gui" / "controllers" / "results_controller.py"
    lines = target.read_text(encoding="utf-8").splitlines()
    violations = [
        f"{target.relative_to(repo_root)}:{index}: {line.strip()}"
        for index, line in enumerate(lines, start=1)
        if "sync_focused_mechanism_controls" in line
    ]

    assert violations == [], (
        "ResultsController may return display refresh outcomes, but non-display mechanism "
        "control synchronization must stay in the event wiring layer.\n"
        + "\n".join(violations)
    )
