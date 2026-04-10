from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import pytest


pytestmark = [pytest.mark.unit]


TARGET_METHODS = {
    "snapshot_datasets",
    "last_fit_metadata",
    "integrate_ctc",
    "set_last_simulation_ctc",
    "set_last_simulation_provenance",
}

BATCH_TARGET_METHODS = {
    "batch_set_ids_for_scope",
    "batch_current_row",
    "batch_set_id_for_row",
    "clear_display_selection_state",
    "display_cached_batch_selection",
    "set_active_batch_selection",
}

QUEUE_CONTEXT_BATCH_TARGET_METHODS = {
    "batch_cache_key",
    "batch_store_set_names",
    "batch_set_id_for_row",
    "batch_preferred_primary_set_id",
}

LAUNCH_VALIDATION_BATCH_TARGET_METHODS = {
    "batch_initials_for_row",
    "batch_set_name_for_id",
    "batch_model_validate_rows",
}

COMPLETION_RECONCILIATION_BATCH_TARGET_METHODS = {
    "batch_set_id_for_name",
    "batch_set_name_for_id",
    "sync_batch_species_columns",
}

RUN_ENTRY_BATCH_TARGET_METHODS = {
    "batch_rows_for_scope",
}

RUN_SIMULATION_INTERNAL_FINAL_BATCH_TARGET_METHODS = {
    "batch_rows_for_scope",
    "batch_store_row_count",
    "batch_model_validate_rows",
    "update_batch_row_controls_state",
    "batch_store_visible_species",
    "sync_batch_species_columns",
}

RUN_UI_ENTRY_PROGRESS_TARGET_METHODS_BY_METHOD = {
    "_run_simulation_from_slider": {
        "run_button_is_enabled",
        "set_run_button_enabled",
        "set_stop_button_enabled",
        "set_status_text",
        "set_sim_progress_value",
    },
    "_run_simulation": {
        "set_run_button_enabled",
        "set_stop_button_enabled",
        "set_status_text",
        "set_sim_progress_value",
    },
    "_cancel_active_run_for_restart": {
        "set_run_button_enabled",
        "set_stop_button_enabled",
    },
    "_start_parallel_batch_simulations": {
        "set_run_button_enabled",
        "set_stop_button_enabled",
        "set_status_text",
        "set_sim_progress_value",
    },
    "_start_next_batch_simulation": {
        "set_run_button_enabled",
        "set_stop_button_enabled",
        "set_status_text",
    },
    "_flush_progress_ui": {
        "set_sim_progress_value",
        "set_status_text",
        "repaint_simulation_widgets",
    },
}

RUN_UI_ENTRY_PROGRESS_TARGET_METHODS = set().union(*RUN_UI_ENTRY_PROGRESS_TARGET_METHODS_BY_METHOD.values())

RUN_UI_LIFECYCLE_TARGET_METHODS_BY_METHOD = {
    "_run_simulation_internal": {
        "set_run_button_enabled",
        "set_stop_button_enabled",
        "set_status_text",
    },
    "_on_simulation_complete": {
        "set_run_button_enabled",
        "set_stop_button_enabled",
        "set_status_text",
        "set_sim_progress_value",
        "repaint_simulation_widgets",
        "set_algebra_status_text",
    },
    "_on_simulation_error": {
        "set_run_button_enabled",
        "set_stop_button_enabled",
        "set_status_text",
        "set_sim_progress_value",
        "set_algebra_status_text",
    },
    "_stop_simulation": {
        "set_run_button_enabled",
        "set_stop_button_enabled",
        "set_status_text",
        "set_sim_progress_value",
    },
}

RUN_UI_LIFECYCLE_TARGET_METHODS = set().union(*RUN_UI_LIFECYCLE_TARGET_METHODS_BY_METHOD.values())

ALL_SIMULATION_CONTROLLER_RUN_UI_TARGET_METHODS = (
    RUN_UI_ENTRY_PROGRESS_TARGET_METHODS | RUN_UI_LIFECYCLE_TARGET_METHODS
)

ALL_SIMULATION_CONTROLLER_BATCH_TARGET_METHODS = (
    BATCH_TARGET_METHODS
    | QUEUE_CONTEXT_BATCH_TARGET_METHODS
    | LAUNCH_VALIDATION_BATCH_TARGET_METHODS
    | COMPLETION_RECONCILIATION_BATCH_TARGET_METHODS
    | RUN_ENTRY_BATCH_TARGET_METHODS
    | RUN_SIMULATION_INTERNAL_FINAL_BATCH_TARGET_METHODS
)

MECHANISM_HELPERS_SNAPSHOT_TARGET_METHODS = {
    "last_mechanism",
    "last_mechanism_context",
}

COMPLETION_MECHANISM_HELPERS_TARGET_METHODS = {
    "is_energy_mode_mechanism",
    "dsl_has_computational_mode_generated_block",
    "sync_energy_mode_temperature_from_mechanism",
    "set_temperature_override_state",
    "set_temperature_mode_indicator_text",
    "update_temperature_mode_indicator",
    "remember_last_mechanism",
    "apply_pending_init_migration",
    "arm_pending_init_result_invalidation_guard",
    "populate_energy_mode_variables_from_mechanism",
    "extract_and_populate_variables",
    "sync_mechanism_controls_to_focused_batch_set",
}

FAILURE_MECHANISM_HELPERS_TARGET_METHODS = {
    "invalidate_pending_init_preserved_results_after_failed_run",
}

COMPLETION_RESULTS_TARGET_METHODS = {
    "set_data",
    "main_plot",
    "set_results_table",
}

COMPLETION_SOLVER_TARGET_METHODS = {
    "temperature_spinbox_value",
    "sim_time_spinbox_text",
    "parse_sim_time_seconds",
    "initial_solver_name",
    "initial_rtol",
    "initial_atol",
    "dsl_global_temperature_K",
    "num_points_spinbox_value",
}

SETTINGS_TARGET_METHODS = {
    "settings_set_value",
    "settings_sync",
}

DIALOGS_TARGET_METHODS = {
    "message_box_warning",
    "message_box_critical",
}

REMAINING_SOLVER_TARGET_METHODS = {
    "parse_sim_time_seconds",
    "initial_solver_name",
    "initial_rtol",
    "initial_atol",
    "temperature_spinbox_value",
    "dsl_global_temperature_K",
    "num_points_spinbox_value",
    "use_sparse_jacobian",
    "wegscheider_cyclicity_enabled",
}

MECHANISM_TARGET_METHODS = {
    "slider_overrides",
    "apply_parameter_overrides_to_dsl",
    "auto_lock_for_run",
    "is_mechanism_ready_for_run",
    "mechanism_reactions_text_raw",
    "has_slider_overrides",
    "apply_overrides_to_text",
    "mechanism_state_network_dsl_raw",
    "apply_overrides_to_state_network_dsl",
    "mechanism_slider_points_value",
    "mechanism_slider_solver_value",
    "get_mechanism_text",
    "simulation_schema_id",
    "simulation_param_fingerprint",
}

RUNTIME_TARGET_METHODS = {
    "prepare_slider_runtime",
    "apply_slider_overrides_to_bindings",
    "set_slider_runtime_dirty",
}

SLIDER_TARGET_METHODS = {
    "stop_slider_release_commit_timer",
    "has_pending_slider_values",
    "finalize_slider_release_commit",
    "stop_variable_update_timer",
    "stop_species_slider_update_timer",
    "set_slider_triggered_simulation",
    "slider_triggered_simulation",
    "last_slider_change_name",
    "slider_drag_active",
    "suppress_slider_refresh",
    "preview_initials_for_row",
    "preview_batch_cache_token",
    "has_dirty_state_for_set",
    "dirty_state_generation",
    "reset_mechanism_workspaces",
    "discard_concentration_overlays_for_set_ids",
}

ALL_SIMULATION_CONTROLLER_SETTINGS_DIALOGS_SOLVER_TARGET_METHODS = (
    SETTINGS_TARGET_METHODS | DIALOGS_TARGET_METHODS | REMAINING_SOLVER_TARGET_METHODS
)

ALL_SIMULATION_CONTROLLER_MECHANISM_HELPERS_TARGET_METHODS = (
    COMPLETION_MECHANISM_HELPERS_TARGET_METHODS
    | FAILURE_MECHANISM_HELPERS_TARGET_METHODS
    | MECHANISM_HELPERS_SNAPSHOT_TARGET_METHODS
)


@dataclass(frozen=True)
class _CallHit:
    method: str
    lineno: int
    line: str


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


def _simulation_controller_method_node(tree: ast.AST, method_name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "SimulationController":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    return item
    raise AssertionError(f"Expected SimulationController.{method_name} to exist")


def _collect_port_usage(
    fn: ast.FunctionDef,
    lines: list[str],
    *,
    explicit_port: str,
    methods: set[str],
) -> tuple[list[_CallHit], set[str]]:
    flattened_hits: list[_CallHit] = []
    explicit_methods: set[str] = set()
    for node in ast.walk(fn):
        if not isinstance(node, ast.Attribute):
            continue
        chain = _attribute_chain(node)
        if not chain:
            continue
        if len(chain) == 3 and chain[:2] == ("self", "ui") and chain[2] in methods:
            flattened_hits.append(
                _CallHit(
                    method=chain[2],
                    lineno=node.lineno,
                    line=lines[node.lineno - 1].strip(),
                )
            )
            continue
        if len(chain) == 4 and chain[:3] == ("self", "ui", explicit_port) and chain[3] in methods:
            explicit_methods.add(chain[3])
    return flattened_hits, explicit_methods


def _statement_assigns_name(stmt: ast.stmt, target_name: str) -> bool:
    if isinstance(stmt, ast.Assign):
        return any(isinstance(target, ast.Name) and target.id == target_name for target in stmt.targets)
    if isinstance(stmt, ast.AnnAssign):
        return isinstance(stmt.target, ast.Name) and stmt.target.id == target_name
    return False


def _dict_constant_string_keys(node: ast.Dict) -> set[str]:
    return {key.value for key in node.keys if isinstance(key, ast.Constant) and isinstance(key.value, str)}


def _simulation_complete_solver_provenance_cluster(fn: ast.FunctionDef) -> list[ast.stmt]:
    required_keys = {
        "solver",
        "solver_label",
        "rtol",
        "atol",
        "temperature_source",
        "simulation_time",
        "num_points_requested",
        "datasets",
    }

    for node in ast.walk(fn):
        if not (isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == "is_primary"):
            continue

        provenance_idx: int | None = None
        for idx, stmt in enumerate(node.body):
            if not _statement_assigns_name(stmt, "provenance"):
                continue
            value = stmt.value if isinstance(stmt, (ast.Assign, ast.AnnAssign)) else None
            if isinstance(value, ast.Dict) and required_keys.issubset(_dict_constant_string_keys(value)):
                provenance_idx = idx
                break
        if provenance_idx is None:
            continue

        start_idx: int | None = None
        for idx, stmt in enumerate(node.body[: provenance_idx + 1]):
            if _statement_assigns_name(stmt, "temperature_used"):
                start_idx = idx
                break
        if start_idx is None:
            raise AssertionError("Expected solver provenance cluster to begin at `temperature_used` assignment.")

        return node.body[start_idx : provenance_idx + 1]

    raise AssertionError("Expected to find the structurally anchored solver provenance block in `_on_simulation_complete`.")


def test_simulation_complete_provenance_cluster_uses_explicit_provenance_port() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    target = repo_root / "kindred" / "gui" / "controllers" / "simulation_controller.py"
    assert target.is_file(), f"Expected file at {target}"

    source = target.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(target))
    fn = _simulation_controller_method_node(tree, "_on_simulation_complete")
    lines = source.splitlines()

    flattened_hits: list[_CallHit] = []
    explicit_methods: set[str] = set()
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        chain = _attribute_chain(node.func)
        if not chain:
            continue
        if len(chain) == 3 and chain[:2] == ("self", "ui") and chain[2] in TARGET_METHODS:
            flattened_hits.append(
                _CallHit(
                    method=chain[2],
                    lineno=node.lineno,
                    line=lines[node.lineno - 1].strip(),
                )
            )
            continue
        if len(chain) == 4 and chain[:3] == ("self", "ui", "provenance") and chain[3] in TARGET_METHODS:
            explicit_methods.add(chain[3])

    assert explicit_methods == TARGET_METHODS, (
        "Guardrail expectation changed: `_on_simulation_complete` must route the provenance cluster through "
        f"`self.ui.provenance`, but only found {sorted(explicit_methods)}."
    )

    assert flattened_hits == [], (
        "Guardrail violated: `_on_simulation_complete` must not use flattened `self.ui.<method>` access for "
        "provenance-port methods.\n"
        + "\n".join(
            f"{target.relative_to(repo_root)}:{hit.lineno}: `{hit.line}`"
            for hit in sorted(flattened_hits, key=lambda hit: (hit.lineno, hit.method))
        )
    )


def test_simulation_controller_init_uses_explicit_settings_port() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    target = repo_root / "kindred" / "gui" / "controllers" / "simulation_controller.py"
    assert target.is_file(), f"Expected file at {target}"

    source = target.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(target))
    fn = _simulation_controller_method_node(tree, "__init__")
    lines = source.splitlines()

    flattened_hits, explicit_methods = _collect_port_usage(
        fn,
        lines,
        explicit_port="settings",
        methods=SETTINGS_TARGET_METHODS,
    )

    assert explicit_methods == SETTINGS_TARGET_METHODS, (
        "Guardrail expectation changed: `SimulationController.__init__` must route the audited settings wiring "
        f"through `self.ui.settings`, but only found {sorted(explicit_methods)}."
    )

    assert flattened_hits == [], (
        "Guardrail violated: `SimulationController.__init__` must not use flattened `self.ui.<method>` access for "
        "SimulationSettingsPort methods.\n"
        + "\n".join(
            f"{target.relative_to(repo_root)}:{hit.lineno}: `{hit.line}`"
            for hit in sorted(flattened_hits, key=lambda hit: (hit.lineno, hit.method))
        )
    )


@pytest.mark.parametrize(
    ("method_name", "expected_methods"),
    (
        ("_run_simulation", {"message_box_warning"}),
        ("_start_parallel_batch_simulations", {"message_box_warning"}),
        ("_start_next_batch_simulation", {"message_box_warning"}),
        ("_run_simulation_internal", {"message_box_warning"}),
        ("_on_simulation_complete", DIALOGS_TARGET_METHODS),
        ("_on_simulation_error", {"message_box_critical"}),
    ),
)
def test_simulation_controller_dialog_clusters_use_explicit_dialogs_port(
    method_name: str, expected_methods: set[str]
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    target = repo_root / "kindred" / "gui" / "controllers" / "simulation_controller.py"
    assert target.is_file(), f"Expected file at {target}"

    source = target.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(target))
    fn = _simulation_controller_method_node(tree, method_name)
    lines = source.splitlines()

    flattened_hits, explicit_methods = _collect_port_usage(
        fn,
        lines,
        explicit_port="dialogs",
        methods=DIALOGS_TARGET_METHODS,
    )

    assert explicit_methods == expected_methods, (
        f"Guardrail expectation changed: `SimulationController.{method_name}` must route the audited dialogs calls "
        f"through `self.ui.dialogs`, but only found {sorted(explicit_methods)}."
    )

    assert flattened_hits == [], (
        f"Guardrail violated: `SimulationController.{method_name}` must not use flattened `self.ui.<method>` access "
        "for SimulationDialogsPort methods.\n"
        + "\n".join(
            f"{target.relative_to(repo_root)}:{hit.lineno}: `{hit.line}`"
            for hit in sorted(flattened_hits, key=lambda hit: (hit.lineno, hit.method))
        )
    )


@pytest.mark.parametrize(
    ("method_name", "expected_methods"),
    (
        ("_run_simulation", {"parse_sim_time_seconds"}),
        (
            "_run_simulation_internal",
            REMAINING_SOLVER_TARGET_METHODS,
        ),
    ),
)
def test_simulation_controller_non_completion_solver_clusters_use_explicit_solver_port(
    method_name: str, expected_methods: set[str]
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    target = repo_root / "kindred" / "gui" / "controllers" / "simulation_controller.py"
    assert target.is_file(), f"Expected file at {target}"

    source = target.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(target))
    fn = _simulation_controller_method_node(tree, method_name)
    lines = source.splitlines()

    flattened_hits, explicit_methods = _collect_port_usage(
        fn,
        lines,
        explicit_port="solver",
        methods=REMAINING_SOLVER_TARGET_METHODS,
    )

    assert explicit_methods == expected_methods, (
        f"Guardrail expectation changed: `SimulationController.{method_name}` must route the audited non-completion "
        f"solver calls through `self.ui.solver`, but only found {sorted(explicit_methods)}."
    )

    assert flattened_hits == [], (
        f"Guardrail violated: `SimulationController.{method_name}` must not use flattened `self.ui.<method>` access "
        "for the audited non-completion SimulationSolverPort methods.\n"
        + "\n".join(
            f"{target.relative_to(repo_root)}:{hit.lineno}: `{hit.line}`"
            for hit in sorted(flattened_hits, key=lambda hit: (hit.lineno, hit.method))
        )
    )


@pytest.mark.parametrize(
    ("method_name", "expected_methods"),
    (
        (
            "_start_next_batch_simulation",
            {
                "slider_overrides",
                "apply_parameter_overrides_to_dsl",
            },
        ),
        (
            "_run_simulation_internal",
            {
                "mechanism_reactions_text_raw",
                "has_slider_overrides",
                "apply_overrides_to_text",
                "mechanism_state_network_dsl_raw",
                "apply_overrides_to_state_network_dsl",
                "mechanism_slider_points_value",
                "mechanism_slider_solver_value",
            },
        ),
        (
            "_run_simulation",
            {
                "auto_lock_for_run",
                "is_mechanism_ready_for_run",
            },
        ),
        (
            "_on_simulation_complete",
            {
                "get_mechanism_text",
            },
        ),
    ),
)
def test_simulation_controller_mechanism_clusters_use_explicit_mechanism_port(
    method_name: str, expected_methods: set[str]
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    target = repo_root / "kindred" / "gui" / "controllers" / "simulation_controller.py"
    assert target.is_file(), f"Expected file at {target}"

    source = target.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(target))
    fn = _simulation_controller_method_node(tree, method_name)
    lines = source.splitlines()

    flattened_hits, explicit_methods = _collect_port_usage(
        fn,
        lines,
        explicit_port="mechanism",
        methods=MECHANISM_TARGET_METHODS,
    )

    assert explicit_methods == expected_methods, (
        f"Guardrail expectation changed: `SimulationController.{method_name}` must route the audited mechanism "
        f"calls through `self.ui.mechanism`, but only found {sorted(explicit_methods)}."
    )

    assert flattened_hits == [], (
        f"Guardrail violated: `SimulationController.{method_name}` must not use flattened `self.ui.<method>` access "
        "for the audited SimulationMechanismPort methods.\n"
        + "\n".join(
            f"{target.relative_to(repo_root)}:{hit.lineno}: `{hit.line}`"
            for hit in sorted(flattened_hits, key=lambda hit: (hit.lineno, hit.method))
        )
    )


@pytest.mark.parametrize(
    ("method_name", "expected_methods"),
    (
        (
            "_flush_slider_plot_updates",
            {
                "batch_set_ids_for_scope",
                "batch_current_row",
                "batch_set_id_for_row",
                "display_cached_batch_selection",
            },
        ),
        (
            "_on_simulation_complete",
            {
                "batch_set_ids_for_scope",
                "batch_current_row",
                "batch_set_id_for_row",
                "clear_display_selection_state",
                "display_cached_batch_selection",
                "set_active_batch_selection",
            },
        ),
    ),
)
def test_simulation_controller_cached_batch_selection_cluster_uses_explicit_batch_port(
    method_name: str, expected_methods: set[str]
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    target = repo_root / "kindred" / "gui" / "controllers" / "simulation_controller.py"
    assert target.is_file(), f"Expected file at {target}"

    source = target.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(target))
    fn = _simulation_controller_method_node(tree, method_name)
    lines = source.splitlines()

    flattened_hits: list[_CallHit] = []
    explicit_methods: set[str] = set()
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        chain = _attribute_chain(node.func)
        if not chain:
            continue
        if len(chain) == 3 and chain[:2] == ("self", "ui") and chain[2] in BATCH_TARGET_METHODS:
            flattened_hits.append(
                _CallHit(
                    method=chain[2],
                    lineno=node.lineno,
                    line=lines[node.lineno - 1].strip(),
                )
            )
            continue
        if len(chain) == 4 and chain[:3] == ("self", "ui", "batch") and chain[3] in BATCH_TARGET_METHODS:
            explicit_methods.add(chain[3])

    assert explicit_methods == expected_methods, (
        f"Guardrail expectation changed: `SimulationController.{method_name}` must route the cached batch-selection "
        f"cluster through `self.ui.batch`, but only found {sorted(explicit_methods)}."
    )

    assert flattened_hits == [], (
        f"Guardrail violated: `SimulationController.{method_name}` must not use flattened `self.ui.<method>` access "
        "for cached batch-selection methods.\n"
        + "\n".join(
            f"{target.relative_to(repo_root)}:{hit.lineno}: `{hit.line}`"
            for hit in sorted(flattened_hits, key=lambda hit: (hit.lineno, hit.method))
        )
    )


def test_run_simulation_internal_queue_context_cluster_uses_explicit_batch_port() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    target = repo_root / "kindred" / "gui" / "controllers" / "simulation_controller.py"
    assert target.is_file(), f"Expected file at {target}"

    source = target.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(target))
    fn = _simulation_controller_method_node(tree, "_run_simulation_internal")
    lines = source.splitlines()

    flattened_hits: list[_CallHit] = []
    explicit_methods: set[str] = set()
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        chain = _attribute_chain(node.func)
        if not chain:
            continue
        if len(chain) == 3 and chain[:2] == ("self", "ui") and chain[2] in QUEUE_CONTEXT_BATCH_TARGET_METHODS:
            flattened_hits.append(
                _CallHit(
                    method=chain[2],
                    lineno=node.lineno,
                    line=lines[node.lineno - 1].strip(),
                )
            )
            continue
        if (
            len(chain) == 4
            and chain[:3] == ("self", "ui", "batch")
            and chain[3] in QUEUE_CONTEXT_BATCH_TARGET_METHODS
        ):
            explicit_methods.add(chain[3])

    assert explicit_methods == QUEUE_CONTEXT_BATCH_TARGET_METHODS, (
        "Guardrail expectation changed: `SimulationController._run_simulation_internal` must route the queue/context "
        f"batch subcluster through `self.ui.batch`, but only found {sorted(explicit_methods)}."
    )

    assert flattened_hits == [], (
        "Guardrail violated: `SimulationController._run_simulation_internal` must not use flattened "
        "`self.ui.<method>` access for the queue/context batch subcluster.\n"
        + "\n".join(
            f"{target.relative_to(repo_root)}:{hit.lineno}: `{hit.line}`"
            for hit in sorted(flattened_hits, key=lambda hit: (hit.lineno, hit.method))
        )
    )


@pytest.mark.parametrize(
    ("method_name", "expected_methods"),
    (
        ("_start_parallel_batch_simulations", {"batch_initials_for_row"}),
        (
            "_start_next_batch_simulation",
            {
                "batch_set_name_for_id",
                "batch_initials_for_row",
                "batch_model_validate_rows",
            },
        ),
    ),
)
def test_simulation_controller_launch_validation_cluster_uses_explicit_batch_port(
    method_name: str, expected_methods: set[str]
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    target = repo_root / "kindred" / "gui" / "controllers" / "simulation_controller.py"
    assert target.is_file(), f"Expected file at {target}"

    source = target.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(target))
    fn = _simulation_controller_method_node(tree, method_name)
    lines = source.splitlines()

    flattened_hits: list[_CallHit] = []
    explicit_methods: set[str] = set()
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        chain = _attribute_chain(node.func)
        if not chain:
            continue
        if len(chain) == 3 and chain[:2] == ("self", "ui") and chain[2] in LAUNCH_VALIDATION_BATCH_TARGET_METHODS:
            flattened_hits.append(
                _CallHit(
                    method=chain[2],
                    lineno=node.lineno,
                    line=lines[node.lineno - 1].strip(),
                )
            )
            continue
        if (
            len(chain) == 4
            and chain[:3] == ("self", "ui", "batch")
            and chain[3] in LAUNCH_VALIDATION_BATCH_TARGET_METHODS
        ):
            explicit_methods.add(chain[3])

    assert explicit_methods == expected_methods, (
        f"Guardrail expectation changed: `SimulationController.{method_name}` must route the launch/validation "
        f"batch cluster through `self.ui.batch`, but only found {sorted(explicit_methods)}."
    )

    assert flattened_hits == [], (
        f"Guardrail violated: `SimulationController.{method_name}` must not use flattened `self.ui.<method>` access "
        "for launch/validation batch methods.\n"
        + "\n".join(
            f"{target.relative_to(repo_root)}:{hit.lineno}: `{hit.line}`"
            for hit in sorted(flattened_hits, key=lambda hit: (hit.lineno, hit.method))
        )
    )


def test_simulation_complete_completion_reconciliation_cluster_uses_explicit_batch_port() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    target = repo_root / "kindred" / "gui" / "controllers" / "simulation_controller.py"
    assert target.is_file(), f"Expected file at {target}"

    source = target.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(target))
    lines = source.splitlines()
    method_names = ("_on_simulation_complete", "_remember_primary_result_mechanism")

    flattened_hits: list[_CallHit] = []
    explicit_methods: set[str] = set()
    for method_name in method_names:
        fn = _simulation_controller_method_node(tree, method_name)
        for node in ast.walk(fn):
            if not isinstance(node, ast.Call):
                continue
            chain = _attribute_chain(node.func)
            if not chain:
                continue
            if len(chain) == 3 and chain[:2] == ("self", "ui") and chain[2] in COMPLETION_RECONCILIATION_BATCH_TARGET_METHODS:
                flattened_hits.append(
                    _CallHit(
                        method=chain[2],
                        lineno=node.lineno,
                        line=lines[node.lineno - 1].strip(),
                    )
                )
                continue
            if (
                len(chain) == 4
                and chain[:3] == ("self", "ui", "batch")
                and chain[3] in COMPLETION_RECONCILIATION_BATCH_TARGET_METHODS
            ):
                explicit_methods.add(chain[3])

    assert explicit_methods == COMPLETION_RECONCILIATION_BATCH_TARGET_METHODS, (
        "Guardrail expectation changed: the audited completion-reconciliation batch cluster must route through "
        f"`self.ui.batch`, but only found {sorted(explicit_methods)}."
    )

    assert flattened_hits == [], (
        "Guardrail violated: the audited completion-reconciliation batch cluster must not use flattened "
        "`self.ui.<method>` access.\n"
        + "\n".join(
            f"{target.relative_to(repo_root)}:{hit.lineno}: `{hit.line}`"
            for hit in sorted(flattened_hits, key=lambda hit: (hit.lineno, hit.method))
        )
    )


def test_simulation_complete_mechanism_helpers_cluster_uses_explicit_mechanism_helpers_port() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    target = repo_root / "kindred" / "gui" / "controllers" / "simulation_controller.py"
    assert target.is_file(), f"Expected file at {target}"

    source = target.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(target))
    lines = source.splitlines()
    method_names = (
        "_on_simulation_complete",
        "_update_primary_result_materialization_contract",
        "_remember_primary_result_mechanism",
        "_refresh_primary_result_controls",
    )

    flattened_hits: list[_CallHit] = []
    explicit_methods: set[str] = set()
    for method_name in method_names:
        fn = _simulation_controller_method_node(tree, method_name)
        for node in ast.walk(fn):
            if not isinstance(node, ast.Call):
                continue
            chain = _attribute_chain(node.func)
            if not chain:
                continue
            if len(chain) == 3 and chain[:2] == ("self", "ui") and chain[2] in COMPLETION_MECHANISM_HELPERS_TARGET_METHODS:
                flattened_hits.append(
                    _CallHit(
                        method=chain[2],
                        lineno=node.lineno,
                        line=lines[node.lineno - 1].strip(),
                    )
                )
                continue
            if (
                len(chain) == 4
                and chain[:3] == ("self", "ui", "mechanism_helpers")
                and chain[3] in COMPLETION_MECHANISM_HELPERS_TARGET_METHODS
            ):
                explicit_methods.add(chain[3])

    assert explicit_methods == COMPLETION_MECHANISM_HELPERS_TARGET_METHODS, (
        "Guardrail expectation changed: the audited mechanism-helper completion cluster must route through "
        "`self.ui.mechanism_helpers`, but only found "
        f"{sorted(explicit_methods)}."
    )

    assert flattened_hits == [], (
        "Guardrail violated: the audited mechanism-helper completion cluster must not use flattened "
        "`self.ui.<method>` access.\n"
        + "\n".join(
            f"{target.relative_to(repo_root)}:{hit.lineno}: `{hit.line}`"
            for hit in sorted(flattened_hits, key=lambda hit: (hit.lineno, hit.method))
        )
    )


def test_simulation_failure_mechanism_helpers_cluster_uses_explicit_mechanism_helpers_port() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    target = repo_root / "kindred" / "gui" / "controllers" / "simulation_controller.py"
    assert target.is_file(), f"Expected file at {target}"

    source = target.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(target))
    lines = source.splitlines()
    method_names = (
        "_invalidate_preserved_pending_init_results_after_failed_run",
    )

    flattened_hits: list[_CallHit] = []
    explicit_methods: set[str] = set()
    for method_name in method_names:
        fn = _simulation_controller_method_node(tree, method_name)
        for node in ast.walk(fn):
            if not isinstance(node, ast.Call):
                continue
            chain = _attribute_chain(node.func)
            if not chain:
                continue
            if len(chain) == 3 and chain[:2] == ("self", "ui") and chain[2] in FAILURE_MECHANISM_HELPERS_TARGET_METHODS:
                flattened_hits.append(
                    _CallHit(
                        method=chain[2],
                        lineno=node.lineno,
                        line=lines[node.lineno - 1].strip(),
                    )
                )
                continue
            if (
                len(chain) == 4
                and chain[:3] == ("self", "ui", "mechanism_helpers")
                and chain[3] in FAILURE_MECHANISM_HELPERS_TARGET_METHODS
            ):
                explicit_methods.add(chain[3])

    assert explicit_methods == FAILURE_MECHANISM_HELPERS_TARGET_METHODS, (
        "Guardrail expectation changed: the audited mechanism-helper failure cluster must route through "
        "`self.ui.mechanism_helpers`, but only found "
        f"{sorted(explicit_methods)}."
    )

    assert flattened_hits == [], (
        "Guardrail violated: the audited mechanism-helper failure cluster must not use flattened "
        "`self.ui.<method>` access.\n"
        + "\n".join(
            f"{target.relative_to(repo_root)}:{hit.lineno}: `{hit.line}`"
            for hit in sorted(flattened_hits, key=lambda hit: (hit.lineno, hit.method))
        )
    )


def test_run_simulation_internal_snapshot_reads_use_explicit_mechanism_helpers_port() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    target = repo_root / "kindred" / "gui" / "controllers" / "simulation_controller.py"
    assert target.is_file(), f"Expected file at {target}"

    source = target.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(target))
    fn = _simulation_controller_method_node(tree, "_run_simulation_internal")
    lines = source.splitlines()

    flattened_hits, explicit_methods = _collect_port_usage(
        fn,
        lines,
        explicit_port="mechanism_helpers",
        methods=MECHANISM_HELPERS_SNAPSHOT_TARGET_METHODS,
    )

    assert explicit_methods == MECHANISM_HELPERS_SNAPSHOT_TARGET_METHODS, (
        "Guardrail expectation changed: `SimulationController._run_simulation_internal` must route mechanism "
        f"snapshot reads through `self.ui.mechanism_helpers`, but only found {sorted(explicit_methods)}."
    )

    assert flattened_hits == [], (
        "Guardrail violated: `SimulationController._run_simulation_internal` must not use flattened "
        "`self.ui.<method>` access for mechanism snapshot reads.\n"
        + "\n".join(
            f"{target.relative_to(repo_root)}:{hit.lineno}: `{hit.line}`"
            for hit in sorted(flattened_hits, key=lambda hit: (hit.lineno, hit.method))
        )
    )


def test_run_simulation_internal_runtime_cluster_uses_explicit_runtime_port() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    target = repo_root / "kindred" / "gui" / "controllers" / "simulation_controller.py"
    assert target.is_file(), f"Expected file at {target}"

    source = target.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(target))
    fn = _simulation_controller_method_node(tree, "_run_simulation_internal")
    lines = source.splitlines()

    flattened_hits, explicit_methods = _collect_port_usage(
        fn,
        lines,
        explicit_port="runtime",
        methods=RUNTIME_TARGET_METHODS,
    )

    assert explicit_methods == RUNTIME_TARGET_METHODS, (
        "Guardrail expectation changed: `SimulationController._run_simulation_internal` must route preview-runtime "
        f"calls through `self.ui.runtime`, but only found {sorted(explicit_methods)}."
    )

    assert flattened_hits == [], (
        "Guardrail violated: `SimulationController._run_simulation_internal` must not use flattened "
        "`self.ui.<method>` access for preview-runtime methods.\n"
        + "\n".join(
            f"{target.relative_to(repo_root)}:{hit.lineno}: `{hit.line}`"
            for hit in sorted(flattened_hits, key=lambda hit: (hit.lineno, hit.method))
        )
    )


def test_simulation_controller_has_no_flattened_runtime_port_usage() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    target = repo_root / "kindred" / "gui" / "controllers" / "simulation_controller.py"
    assert target.is_file(), f"Expected file at {target}"

    source = target.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(target))
    lines = source.splitlines()

    flattened_hits: list[_CallHit] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or node.name != "SimulationController":
            continue
        for item in node.body:
            if not isinstance(item, ast.FunctionDef):
                continue
            method_hits, _ = _collect_port_usage(
                item,
                lines,
                explicit_port="runtime",
                methods=RUNTIME_TARGET_METHODS,
            )
            flattened_hits.extend(method_hits)

    assert flattened_hits == [], (
        "Guardrail violated: `SimulationController` must not use flattened `self.ui.<method>` access for any "
        "SimulationRuntimePort methods.\n"
        + "\n".join(
            f"{target.relative_to(repo_root)}:{hit.lineno}: `{hit.line}`"
            for hit in sorted(flattened_hits, key=lambda hit: (hit.lineno, hit.method))
        )
    )


def test_simulation_controller_has_no_flattened_mechanism_helpers_port_usage() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    target = repo_root / "kindred" / "gui" / "controllers" / "simulation_controller.py"
    assert target.is_file(), f"Expected file at {target}"

    source = target.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(target))
    lines = source.splitlines()

    flattened_hits: list[_CallHit] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or node.name != "SimulationController":
            continue
        for item in node.body:
            if not isinstance(item, ast.FunctionDef):
                continue
            method_hits, _ = _collect_port_usage(
                item,
                lines,
                explicit_port="mechanism_helpers",
                methods=ALL_SIMULATION_CONTROLLER_MECHANISM_HELPERS_TARGET_METHODS,
            )
            flattened_hits.extend(method_hits)

    assert flattened_hits == [], (
        "Guardrail violated: `SimulationController` must not use flattened `self.ui.<method>` access for any "
        "audited SimulationMechanismHelpersPort methods.\n"
        + "\n".join(
            f"{target.relative_to(repo_root)}:{hit.lineno}: `{hit.line}`"
            for hit in sorted(flattened_hits, key=lambda hit: (hit.lineno, hit.method))
        )
    )


def test_simulation_controller_slider_cluster_uses_explicit_slider_port() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    target = repo_root / "kindred" / "gui" / "controllers" / "simulation_controller.py"
    assert target.is_file(), f"Expected file at {target}"

    source = target.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(target))
    lines = source.splitlines()

    flattened_hits: list[_CallHit] = []
    explicit_methods: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or node.name != "SimulationController":
            continue
        for item in node.body:
            if not isinstance(item, ast.FunctionDef):
                continue
            method_hits, method_explicit = _collect_port_usage(
                item,
                lines,
                explicit_port="slider",
                methods=SLIDER_TARGET_METHODS,
            )
            flattened_hits.extend(method_hits)
            explicit_methods.update(method_explicit)

    assert explicit_methods == SLIDER_TARGET_METHODS, (
        "Guardrail expectation changed: `SimulationController` must route the audited slider cluster through "
        f"`self.ui.slider`, but only found {sorted(explicit_methods)}."
    )
    assert flattened_hits == [], (
        "Guardrail violated: `SimulationController` must not use flattened `self.ui.<method>` access for "
        "SimulationSliderPort methods.\n"
        + "\n".join(
            f"{target.relative_to(repo_root)}:{hit.lineno}: `{hit.line}`"
            for hit in sorted(flattened_hits, key=lambda hit: (hit.lineno, hit.method))
        )
    )


def test_simulation_complete_results_cluster_uses_explicit_results_port() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    target = repo_root / "kindred" / "gui" / "controllers" / "simulation_controller.py"
    assert target.is_file(), f"Expected file at {target}"

    source = target.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(target))
    fn = _simulation_controller_method_node(tree, "_on_simulation_complete")
    lines = source.splitlines()

    flattened_hits: list[_CallHit] = []
    explicit_methods: set[str] = set()
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        chain = _attribute_chain(node.func)
        if not chain:
            continue
        if len(chain) == 3 and chain[:2] == ("self", "ui") and chain[2] in COMPLETION_RESULTS_TARGET_METHODS:
            flattened_hits.append(
                _CallHit(
                    method=chain[2],
                    lineno=node.lineno,
                    line=lines[node.lineno - 1].strip(),
                )
            )
            continue
        if (
            len(chain) == 4
            and chain[:3] == ("self", "ui", "results")
            and chain[3] in COMPLETION_RESULTS_TARGET_METHODS
        ):
            explicit_methods.add(chain[3])

    assert explicit_methods == COMPLETION_RESULTS_TARGET_METHODS, (
        "Guardrail expectation changed: `SimulationController._on_simulation_complete` must route the audited "
        "results completion cluster through `self.ui.results`, but only found "
        f"{sorted(explicit_methods)}."
    )

    assert flattened_hits == [], (
        "Guardrail violated: `SimulationController._on_simulation_complete` must not use flattened "
        "`self.ui.<method>` access for the audited results completion cluster.\n"
        + "\n".join(
            f"{target.relative_to(repo_root)}:{hit.lineno}: `{hit.line}`"
            for hit in sorted(flattened_hits, key=lambda hit: (hit.lineno, hit.method))
        )
    )


def test_simulation_complete_solver_cluster_uses_explicit_solver_port() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    target = repo_root / "kindred" / "gui" / "controllers" / "simulation_controller.py"
    assert target.is_file(), f"Expected file at {target}"

    source = target.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(target))
    fn = _simulation_controller_method_node(tree, "_on_simulation_complete")
    lines = source.splitlines()
    cluster = _simulation_complete_solver_provenance_cluster(fn)

    flattened_hits: list[_CallHit] = []
    explicit_methods: set[str] = set()
    for stmt in cluster:
        for node in ast.walk(stmt):
            if not isinstance(node, ast.Call):
                continue
            chain = _attribute_chain(node.func)
            if not chain:
                continue
            if len(chain) == 3 and chain[:2] == ("self", "ui") and chain[2] in COMPLETION_SOLVER_TARGET_METHODS:
                flattened_hits.append(
                    _CallHit(
                        method=chain[2],
                        lineno=node.lineno,
                        line=lines[node.lineno - 1].strip(),
                    )
                )
                continue
            if (
                len(chain) == 4
                and chain[:3] == ("self", "ui", "solver")
                and chain[3] in COMPLETION_SOLVER_TARGET_METHODS
            ):
                explicit_methods.add(chain[3])

    assert explicit_methods == COMPLETION_SOLVER_TARGET_METHODS, (
        "Guardrail expectation changed: `SimulationController._on_simulation_complete` must route the audited "
        "solver completion cluster through `self.ui.solver`, but only found "
        f"{sorted(explicit_methods)}."
    )

    assert flattened_hits == [], (
        "Guardrail violated: `SimulationController._on_simulation_complete` must not use flattened "
        "`self.ui.<method>` access for the audited solver completion cluster.\n"
        + "\n".join(
            f"{target.relative_to(repo_root)}:{hit.lineno}: `{hit.line}`"
            for hit in sorted(flattened_hits, key=lambda hit: (hit.lineno, hit.method))
        )
    )


@pytest.mark.parametrize(
    "method_name",
    (
        "_run_simulation_from_slider",
        "_run_simulation",
    ),
)
def test_simulation_controller_run_entry_batch_cluster_uses_explicit_batch_port(method_name: str) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    target = repo_root / "kindred" / "gui" / "controllers" / "simulation_controller.py"
    assert target.is_file(), f"Expected file at {target}"

    source = target.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(target))
    fn = _simulation_controller_method_node(tree, method_name)
    lines = source.splitlines()

    flattened_hits: list[_CallHit] = []
    explicit_methods: set[str] = set()
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        chain = _attribute_chain(node.func)
        if not chain:
            continue
        if len(chain) == 3 and chain[:2] == ("self", "ui") and chain[2] in RUN_ENTRY_BATCH_TARGET_METHODS:
            flattened_hits.append(
                _CallHit(
                    method=chain[2],
                    lineno=node.lineno,
                    line=lines[node.lineno - 1].strip(),
                )
            )
            continue
        if len(chain) == 4 and chain[:3] == ("self", "ui", "batch") and chain[3] in RUN_ENTRY_BATCH_TARGET_METHODS:
            explicit_methods.add(chain[3])

    assert explicit_methods == RUN_ENTRY_BATCH_TARGET_METHODS, (
        f"Guardrail expectation changed: `SimulationController.{method_name}` must route the run-entry batch cluster "
        f"through `self.ui.batch`, but only found {sorted(explicit_methods)}."
    )

    assert flattened_hits == [], (
        f"Guardrail violated: `SimulationController.{method_name}` must not use flattened `self.ui.<method>` access "
        "for run-entry batch methods.\n"
        + "\n".join(
            f"{target.relative_to(repo_root)}:{hit.lineno}: `{hit.line}`"
            for hit in sorted(flattened_hits, key=lambda hit: (hit.lineno, hit.method))
        )
    )


@pytest.mark.parametrize(
    ("method_name", "expected_methods"),
    tuple(RUN_UI_ENTRY_PROGRESS_TARGET_METHODS_BY_METHOD.items()),
)
def test_simulation_controller_run_ui_entry_progress_clusters_use_explicit_run_ui_port(
    method_name: str, expected_methods: set[str]
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    target = repo_root / "kindred" / "gui" / "controllers" / "simulation_controller.py"
    assert target.is_file(), f"Expected file at {target}"

    source = target.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(target))
    fn = _simulation_controller_method_node(tree, method_name)
    lines = source.splitlines()

    flattened_hits, explicit_methods = _collect_port_usage(
        fn,
        lines,
        explicit_port="run_ui",
        methods=RUN_UI_ENTRY_PROGRESS_TARGET_METHODS,
    )

    assert explicit_methods == expected_methods, (
        f"Guardrail expectation changed: `SimulationController.{method_name}` must route the audited phase-1 "
        f"SimulationRunUiPort cluster through `self.ui.run_ui`, but only found {sorted(explicit_methods)}."
    )

    assert flattened_hits == [], (
        f"Guardrail violated: `SimulationController.{method_name}` must not use flattened `self.ui.<method>` access "
        "for the audited phase-1 SimulationRunUiPort methods.\n"
        + "\n".join(
            f"{target.relative_to(repo_root)}:{hit.lineno}: `{hit.line}`"
            for hit in sorted(flattened_hits, key=lambda hit: (hit.lineno, hit.method))
        )
    )


def test_simulation_controller_has_no_flattened_phase1_run_ui_usage() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    target = repo_root / "kindred" / "gui" / "controllers" / "simulation_controller.py"
    assert target.is_file(), f"Expected file at {target}"

    source = target.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(target))
    lines = source.splitlines()

    flattened_hits: list[_CallHit] = []
    for method_name in RUN_UI_ENTRY_PROGRESS_TARGET_METHODS_BY_METHOD:
        fn = _simulation_controller_method_node(tree, method_name)
        method_hits, _ = _collect_port_usage(
            fn,
            lines,
            explicit_port="run_ui",
            methods=RUN_UI_ENTRY_PROGRESS_TARGET_METHODS,
        )
        flattened_hits.extend(method_hits)

    assert flattened_hits == [], (
        "Guardrail violated: `SimulationController` must not use flattened `self.ui.<method>` access for the "
        "audited phase-1 SimulationRunUiPort entry/progress/start cluster.\n"
        + "\n".join(
            f"{target.relative_to(repo_root)}:{hit.lineno}: `{hit.line}`"
            for hit in sorted(flattened_hits, key=lambda hit: (hit.lineno, hit.method))
        )
    )


@pytest.mark.parametrize(
    ("method_name", "expected_methods"),
    tuple(RUN_UI_LIFECYCLE_TARGET_METHODS_BY_METHOD.items()),
)
def test_simulation_controller_run_ui_lifecycle_clusters_use_explicit_run_ui_port(
    method_name: str, expected_methods: set[str]
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    target = repo_root / "kindred" / "gui" / "controllers" / "simulation_controller.py"
    assert target.is_file(), f"Expected file at {target}"

    source = target.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(target))
    fn = _simulation_controller_method_node(tree, method_name)
    lines = source.splitlines()

    flattened_hits, explicit_methods = _collect_port_usage(
        fn,
        lines,
        explicit_port="run_ui",
        methods=RUN_UI_LIFECYCLE_TARGET_METHODS,
    )

    assert explicit_methods == expected_methods, (
        f"Guardrail expectation changed: `SimulationController.{method_name}` must route the audited phase-2 "
        f"SimulationRunUiPort lifecycle cluster through `self.ui.run_ui`, but only found {sorted(explicit_methods)}."
    )

    assert flattened_hits == [], (
        f"Guardrail violated: `SimulationController.{method_name}` must not use flattened `self.ui.<method>` access "
        "for the audited phase-2 SimulationRunUiPort methods.\n"
        + "\n".join(
            f"{target.relative_to(repo_root)}:{hit.lineno}: `{hit.line}`"
            for hit in sorted(flattened_hits, key=lambda hit: (hit.lineno, hit.method))
        )
    )


def test_simulation_controller_has_no_flattened_run_ui_port_usage() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    target = repo_root / "kindred" / "gui" / "controllers" / "simulation_controller.py"
    assert target.is_file(), f"Expected file at {target}"

    source = target.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(target))
    lines = source.splitlines()

    flattened_hits: list[_CallHit] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or node.name != "SimulationController":
            continue
        for item in node.body:
            if not isinstance(item, ast.FunctionDef):
                continue
            method_hits, _ = _collect_port_usage(
                item,
                lines,
                explicit_port="run_ui",
                methods=ALL_SIMULATION_CONTROLLER_RUN_UI_TARGET_METHODS,
            )
            flattened_hits.extend(method_hits)

    assert flattened_hits == [], (
        "Guardrail violated: `SimulationController` must not use flattened `self.ui.<method>` access for any "
        "SimulationRunUiPort methods.\n"
        + "\n".join(
            f"{target.relative_to(repo_root)}:{hit.lineno}: `{hit.line}`"
            for hit in sorted(flattened_hits, key=lambda hit: (hit.lineno, hit.method))
        )
    )


def test_run_simulation_internal_final_batch_residue_uses_explicit_batch_port() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    target = repo_root / "kindred" / "gui" / "controllers" / "simulation_controller.py"
    assert target.is_file(), f"Expected file at {target}"

    source = target.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(target))
    fn = _simulation_controller_method_node(tree, "_run_simulation_internal")
    lines = source.splitlines()

    flattened_hits: list[_CallHit] = []
    explicit_methods: set[str] = set()
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        chain = _attribute_chain(node.func)
        if not chain:
            continue
        if len(chain) == 3 and chain[:2] == ("self", "ui") and chain[2] in RUN_SIMULATION_INTERNAL_FINAL_BATCH_TARGET_METHODS:
            flattened_hits.append(
                _CallHit(
                    method=chain[2],
                    lineno=node.lineno,
                    line=lines[node.lineno - 1].strip(),
                )
            )
            continue
        if (
            len(chain) == 4
            and chain[:3] == ("self", "ui", "batch")
            and chain[3] in RUN_SIMULATION_INTERNAL_FINAL_BATCH_TARGET_METHODS
        ):
            explicit_methods.add(chain[3])

    assert explicit_methods == RUN_SIMULATION_INTERNAL_FINAL_BATCH_TARGET_METHODS, (
        "Guardrail expectation changed: `SimulationController._run_simulation_internal` must route the final "
        f"batch residue through `self.ui.batch`, but only found {sorted(explicit_methods)}."
    )

    assert flattened_hits == [], (
        "Guardrail violated: `SimulationController._run_simulation_internal` must not use flattened "
        "`self.ui.<method>` access for the final batch residue.\n"
        + "\n".join(
            f"{target.relative_to(repo_root)}:{hit.lineno}: `{hit.line}`"
            for hit in sorted(flattened_hits, key=lambda hit: (hit.lineno, hit.method))
        )
    )


def test_simulation_controller_has_no_flattened_batch_port_usage() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    target = repo_root / "kindred" / "gui" / "controllers" / "simulation_controller.py"
    assert target.is_file(), f"Expected file at {target}"

    source = target.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(target))
    lines = source.splitlines()

    flattened_hits: list[_CallHit] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or node.name != "SimulationController":
            continue
        for item in node.body:
            if not isinstance(item, ast.FunctionDef):
                continue
            for subnode in ast.walk(item):
                if not isinstance(subnode, ast.Call):
                    continue
                chain = _attribute_chain(subnode.func)
                if not chain:
                    continue
                if len(chain) == 3 and chain[:2] == ("self", "ui") and chain[2] in ALL_SIMULATION_CONTROLLER_BATCH_TARGET_METHODS:
                    flattened_hits.append(
                        _CallHit(
                            method=chain[2],
                            lineno=subnode.lineno,
                            line=lines[subnode.lineno - 1].strip(),
                        )
                    )

    assert flattened_hits == [], (
        "Guardrail violated: `SimulationController` must not use flattened `self.ui.<method>` access for any "
        "SimulationBatchPort methods.\n"
        + "\n".join(
            f"{target.relative_to(repo_root)}:{hit.lineno}: `{hit.line}`"
            for hit in sorted(flattened_hits, key=lambda hit: (hit.lineno, hit.method))
        )
    )


def test_simulation_controller_has_no_flattened_settings_dialogs_or_remaining_solver_usage() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    target = repo_root / "kindred" / "gui" / "controllers" / "simulation_controller.py"
    assert target.is_file(), f"Expected file at {target}"

    source = target.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(target))
    lines = source.splitlines()

    flattened_hits: list[_CallHit] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or node.name != "SimulationController":
            continue
        for item in node.body:
            if not isinstance(item, ast.FunctionDef):
                continue
            for subnode in ast.walk(item):
                if not isinstance(subnode, ast.Attribute):
                    continue
                chain = _attribute_chain(subnode)
                if not chain:
                    continue
                if (
                    len(chain) == 3
                    and chain[:2] == ("self", "ui")
                    and chain[2] in ALL_SIMULATION_CONTROLLER_SETTINGS_DIALOGS_SOLVER_TARGET_METHODS
                ):
                    flattened_hits.append(
                        _CallHit(
                            method=chain[2],
                            lineno=subnode.lineno,
                            line=lines[subnode.lineno - 1].strip(),
                        )
                    )

    assert flattened_hits == [], (
        "Guardrail violated: `SimulationController` must not use flattened `self.ui.<method>` access for the "
        "audited SimulationSettingsPort, SimulationDialogsPort, or remaining non-completion "
        "SimulationSolverPort methods.\n"
        + "\n".join(
            f"{target.relative_to(repo_root)}:{hit.lineno}: `{hit.line}`"
            for hit in sorted(flattened_hits, key=lambda hit: (hit.lineno, hit.method))
        )
    )


def test_simulation_controller_has_no_flattened_mechanism_port_usage() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    target = repo_root / "kindred" / "gui" / "controllers" / "simulation_controller.py"
    assert target.is_file(), f"Expected file at {target}"

    source = target.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(target))
    lines = source.splitlines()

    flattened_hits: list[_CallHit] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or node.name != "SimulationController":
            continue
        for item in node.body:
            if not isinstance(item, ast.FunctionDef):
                continue
            for subnode in ast.walk(item):
                if not isinstance(subnode, ast.Attribute):
                    continue
                chain = _attribute_chain(subnode)
                if not chain:
                    continue
                if len(chain) == 3 and chain[:2] == ("self", "ui") and chain[2] in MECHANISM_TARGET_METHODS:
                    flattened_hits.append(
                        _CallHit(
                            method=chain[2],
                            lineno=subnode.lineno,
                            line=lines[subnode.lineno - 1].strip(),
                        )
                    )

    assert flattened_hits == [], (
        "Guardrail violated: `SimulationController` must not use flattened `self.ui.<method>` access for any "
        "audited SimulationMechanismPort methods.\n"
        + "\n".join(
            f"{target.relative_to(repo_root)}:{hit.lineno}: `{hit.line}`"
            for hit in sorted(flattened_hits, key=lambda hit: (hit.lineno, hit.method))
        )
    )
