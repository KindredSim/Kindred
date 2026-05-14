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
        "set_status_text",
        "set_sim_progress_value",
    },
    "_handle_parallel_batch_runtime_waiting": {
        "set_run_button_enabled",
        "set_stop_button_enabled",
        "set_status_text",
        "set_sim_progress_value",
    },
    "_handle_parallel_batch_runtime_check_failed": {
        "set_run_button_enabled",
        "set_stop_button_enabled",
        "set_status_text",
        "set_sim_progress_value",
    },
    "_submit_parallel_batch_tasks": {
        "set_run_button_enabled",
        "set_stop_button_enabled",
    },
    "_finish_parallel_batch_with_no_active_requests": {
        "set_run_button_enabled",
        "set_stop_button_enabled",
    },
    "_start_next_batch_simulation": {
        "set_status_text",
    },
    "_abort_serial_batch_for_invalid_initials": {
        "set_run_button_enabled",
        "set_stop_button_enabled",
    },
    "_run_rows_or_abort": {
        "set_run_button_enabled",
        "set_stop_button_enabled",
    },
    "_run_mechanism_context_or_abort": set(),
    "_run_solver_context_or_abort": set(),
    "_build_run_dispatch_context_or_abort": set(),
    "_flush_progress_ui": {
        "set_sim_progress_value",
        "set_status_text",
        "repaint_simulation_widgets",
    },
}

RUN_UI_ENTRY_PROGRESS_TARGET_METHODS = set().union(*RUN_UI_ENTRY_PROGRESS_TARGET_METHODS_BY_METHOD.values())

RUN_UI_LIFECYCLE_TARGET_METHODS_BY_METHOD = {
    "_run_simulation_internal": set(),
    "_on_simulation_complete": set(),
    "_on_simulation_error": set(),
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
    "set_temperature_override_state",
    "set_temperature_mode_indicator_text",
    "update_temperature_mode_indicator",
    "remember_last_mechanism",
    "apply_pending_init_migration",
    "arm_pending_init_result_invalidation_guard",
    "sync_mechanism_controls_to_focused_batch_set",
}

FAILURE_MECHANISM_HELPERS_TARGET_METHODS = {
    "invalidate_pending_init_preserved_results_after_failed_run",
}

COMPLETION_RESULTS_TARGET_METHODS = {
    "main_plot",
    "publish_simulation_completion_result",
    "publish_completion_intervention_annotations",
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
    "message_box_question",
    "choose_wegscheider_resolution",
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
    "variable_slider_values",
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
    "apply_wegscheider_resolution_source_rewrite",
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
    return _class_method_node(tree, "SimulationController", method_name)


def _completion_publication_method_node(tree: ast.AST, method_name: str) -> ast.FunctionDef:
    return _class_method_node(tree, "SimulationCompletionPublicationOwner", method_name)


def _class_method_node(tree: ast.AST, class_name: str, method_name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    return item
    raise AssertionError(f"Expected {class_name}.{method_name} to exist")


def _class_node(tree: ast.AST, class_name: str) -> ast.ClassDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return node
    raise AssertionError(f"Expected {class_name} to exist")


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
        if (
            (
                len(chain) == 4
                and chain[:3] == ("self", "ui", explicit_port)
            )
            or (
                len(chain) == 4
                and chain[:3] == ("self", "_ui", explicit_port)
            )
            or (
                len(chain) == 4
                and chain[:3] == ("self", "_ports", explicit_port)
            )
        ) and chain[3] in methods:
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


def _repo_source_tree(relative_path: str) -> tuple[Path, str, ast.AST]:
    repo_root = Path(__file__).resolve().parents[1]
    target = repo_root / relative_path
    assert target.is_file(), f"Expected file at {target}"
    source = target.read_text(encoding="utf-8")
    return target, source, ast.parse(source, filename=str(target))


def _call_chains(fn: ast.FunctionDef) -> set[tuple[str, ...]]:
    chains: set[tuple[str, ...]] = set()
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        chain = _attribute_chain(node.func)
        if chain:
            chains.add(chain)
    return chains


def _all_attribute_chains(fn: ast.FunctionDef) -> set[tuple[str, ...]]:
    chains: set[tuple[str, ...]] = set()
    for node in ast.walk(fn):
        if not isinstance(node, ast.Attribute):
            continue
        chain = _attribute_chain(node)
        if chain:
            chains.add(chain)
    return chains


def test_parallel_batch_runtime_snapshot_status_is_owned_by_readiness_owner() -> None:
    _target, _source, tree = _repo_source_tree("kindred/gui/controllers/simulation_controller.py")
    fn = _simulation_controller_method_node(tree, "_parallel_batch_runtime_snapshot")

    chains = _call_chains(fn)

    forbidden = {
        ("self", "_batch_parallel", "has_ready_lane_pool"),
        ("self", "_batch_parallel", "runtime_snapshot"),
        ("self", "_parallel_batch_runtime_readiness_owner", "mark_ready"),
        ("self", "_parallel_batch_runtime_readiness_owner", "mark_not_ready"),
    }
    assert chains.isdisjoint(forbidden), (
        "`SimulationController._parallel_batch_runtime_snapshot` must not reconstruct batch readiness from "
        "`ParallelBatchExecutor` state or mutate readiness markers directly; status/snapshot authority belongs "
        "to `ParallelBatchRuntimeReadinessOwner`."
    )


def test_controller_no_longer_owns_duplicate_batch_lane_pool_factory() -> None:
    _target, _source, tree = _repo_source_tree("kindred/gui/controllers/simulation_controller.py")

    assert all(
        not (isinstance(node, ast.FunctionDef) and node.name == "_default_batch_lane_pool_factory")
        for node in ast.walk(tree)
    ), "`SimulationController` must not define a duplicate batch lane-pool factory; use the runtime adapter authority."


def test_callback_identity_has_no_full_batch_context_snapshot_escape_hatch() -> None:
    _target, _source, tree = _repo_source_tree("kindred/gui/controllers/simulation_callback_identity.py")
    class_node = next(
        node for node in ast.walk(tree) if isinstance(node, ast.ClassDef) and node.name == "SimulationCallbackIdentity"
    )

    annotated_fields = {
        stmt.target.id
        for stmt in class_node.body
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)
    }
    assert "context_snapshot" not in annotated_fields
    assert "callback_context" in annotated_fields


def test_batch_context_owner_has_no_raw_current_snapshot_compatibility_api() -> None:
    _target, _source, tree = _repo_source_tree("kindred/gui/controllers/batch_run_context_owner.py")
    class_node = next(
        node for node in ast.walk(tree) if isinstance(node, ast.ClassDef) and node.name == "BatchRunContextOwner"
    )
    methods = {node.name for node in class_node.body if isinstance(node, ast.FunctionDef)}

    assert "current_context_snapshot" not in methods
    assert "callback_context_snapshot" in methods


def test_production_code_does_not_call_raw_batch_context_snapshot_for_callbacks() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    production_paths = [
        repo_root / "kindred" / "gui" / "controllers" / "simulation_controller.py",
        repo_root / "kindred" / "gui" / "controllers" / "simulation_completion_callback.py",
        repo_root / "kindred" / "gui" / "controllers" / "simulation_error_handling.py",
        repo_root / "kindred" / "gui" / "controllers" / "parallel_batch_outcome.py",
    ]
    hits: list[str] = []
    for path in production_paths:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        lines = source.splitlines()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Attribute):
                continue
            chain = _attribute_chain(node)
            if chain and chain[-1] == "current_context_snapshot":
                hits.append(f"{path.relative_to(repo_root)}:{node.lineno}: {lines[node.lineno - 1].strip()}")

    assert hits == [], "Production callback flow must not call raw full batch context snapshots.\n" + "\n".join(hits)


def test_completion_publication_does_not_fallback_to_active_cache_key_for_callback_paths() -> None:
    _target, _source, tree = _repo_source_tree("kindred/gui/controllers/simulation_completion_publication.py")
    fn = _completion_publication_method_node(tree, "resolve_cache_key")
    chains = _all_attribute_chains(fn)

    forbidden = {
        ("self", "_batch_cache", "active_cache_key"),
        ("self", "_batch_cache", "active_preview_cache_key"),
    }
    assert chains.isdisjoint(forbidden), (
        "Callback-driven completion must not silently mask a missing callback cache key with active cache state; "
        "any fallback must be explicit, named, and tested."
    )


def test_completion_publication_does_not_rediscover_batch_identity_after_callback_resolution() -> None:
    _target, _source, tree = _repo_source_tree("kindred/gui/controllers/simulation_completion_publication.py")
    class_node = _class_node(tree, "SimulationCompletionPublicationOwner")
    method_names = {
        node.name
        for node in class_node.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    assert "resolve_batch_identity" not in method_names


def test_completion_publication_dependencies_exclude_false_owner_pass_throughs() -> None:
    _target, _source, tree = _repo_source_tree("kindred/gui/controllers/simulation_completion_publication.py")
    class_node = _class_node(tree, "SimulationCompletionPublicationDependencies")
    dependency_fields = {
        stmt.target.id
        for stmt in class_node.body
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)
    }

    forbidden = {
        "completion_policy_cache_state",
        "resolve_completion_mechanism",
        "update_primary_result_materialization_contract",
        "remember_primary_result_mechanism",
        "include_mechanism_in_result_payload",
        "refresh_primary_result_controls",
    }
    assert dependency_fields.isdisjoint(forbidden), (
        "Completion publication must own cache-state reads directly, use the result-materialization owner "
        "directly, and ask the batch-context owner for mechanism-payload policy instead of routing those "
        "responsibilities through controller pass-through callables."
    )


def test_controller_no_longer_owns_completion_publication_policy_helpers() -> None:
    _target, _source, tree = _repo_source_tree("kindred/gui/controllers/simulation_controller.py")
    class_node = _class_node(tree, "SimulationController")
    method_names = {
        node.name
        for node in class_node.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    assert "_completion_policy_cache_state" not in method_names
    assert "_include_mechanism_in_result_payload" not in method_names


def test_completion_publication_wiring_does_not_wrap_dependencies_in_lambdas() -> None:
    _target, _source, tree = _repo_source_tree("kindred/gui/controllers/simulation_controller.py")
    publication_dependency_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "SimulationCompletionPublicationDependencies"
    ]
    assert len(publication_dependency_calls) == 1

    lambda_fields = [
        keyword.arg
        for keyword in publication_dependency_calls[0].keywords
        if isinstance(keyword.value, ast.Lambda)
    ]
    assert lambda_fields == [], (
        "Completion publication wiring must pass bounded owner/controller callables directly; lambda wrappers "
        "hide pass-through scaffolding and make ownership harder to audit."
    )


def test_run_preparation_wiring_does_not_wrap_dependencies_in_lambdas() -> None:
    _target, _source, tree = _repo_source_tree("kindred/gui/controllers/simulation_controller.py")
    preparation_dependency_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "SimulationRunPreparationDependencies"
    ]
    assert len(preparation_dependency_calls) == 1

    lambda_fields = [
        keyword.arg
        for keyword in preparation_dependency_calls[0].keywords
        if isinstance(keyword.value, ast.Lambda)
    ]
    assert lambda_fields == [], (
        "Run-preparation wiring must not hide dependencies behind controller lambda wrappers; this guard is scoped "
        "to pass-through scaffolding and does not by itself prove final ownership for every callable dependency."
    )


def test_run_preparation_signature_authority_is_not_controller_dependency() -> None:
    _controller_target, _controller_source, controller_tree = _repo_source_tree(
        "kindred/gui/controllers/simulation_controller.py"
    )
    _prep_target, _prep_source, prep_tree = _repo_source_tree(
        "kindred/gui/controllers/simulation_run_preparation.py"
    )

    controller_imported_names = {
        alias.name
        for node in ast.walk(controller_tree)
        if isinstance(node, ast.ImportFrom) and node.module == "kindred.core.batch_parallel"
        for alias in node.names
    }
    controller_node = _class_node(controller_tree, "SimulationController")
    controller_method_names = {
        stmt.name for stmt in controller_node.body if isinstance(stmt, ast.FunctionDef)
    }
    preparation_dependency_calls = [
        node
        for node in ast.walk(controller_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "SimulationRunPreparationDependencies"
    ]
    assert len(preparation_dependency_calls) == 1
    preparation_keyword_fields = {keyword.arg for keyword in preparation_dependency_calls[0].keywords}

    dependency_node = _class_node(prep_tree, "SimulationRunPreparationDependencies")
    dependency_fields = {
        stmt.target.id
        for stmt in dependency_node.body
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)
    }
    prep_imported_names = {
        alias.name
        for node in ast.walk(prep_tree)
        if isinstance(node, ast.ImportFrom) and node.module == "kindred.core.batch_parallel"
        for alias in node.names
    }

    assert "batch_mechanism_signature" not in controller_imported_names
    assert "_run_preparation_batch_mechanism_signature" not in controller_method_names
    assert "batch_mechanism_signature" not in preparation_keyword_fields
    assert "batch_mechanism_signature" not in dependency_fields
    assert "batch_mechanism_signature" in prep_imported_names


def test_parallel_batch_outcome_uses_callback_owners_not_controller_dispatch_dependencies() -> None:
    _target, _source, tree = _repo_source_tree("kindred/gui/controllers/parallel_batch_outcome.py")
    dependency_node = _class_node(tree, "ParallelBatchOutcomeDependencies")
    dependency_fields = {
        stmt.target.id
        for stmt in dependency_node.body
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)
    }
    owner_init = _class_method_node(tree, "ParallelBatchOutcomeOwner", "__init__")
    init_args = {arg.arg for arg in owner_init.args.kwonlyargs}

    assert "dispatch_simulation_complete" not in dependency_fields
    assert "dispatch_simulation_error" not in dependency_fields
    assert "completion_callback_owner" in init_args
    assert "error_handling_owner" in init_args


def test_parallel_batch_outcome_wiring_does_not_wrap_dependencies_in_lambdas() -> None:
    _target, _source, tree = _repo_source_tree("kindred/gui/controllers/simulation_controller.py")
    outcome_dependency_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "ParallelBatchOutcomeDependencies"
    ]
    assert len(outcome_dependency_calls) == 1

    lambda_fields = [
        keyword.arg
        for keyword in outcome_dependency_calls[0].keywords
        if isinstance(keyword.value, ast.Lambda)
    ]
    assert lambda_fields == [], (
        "Parallel batch outcome wiring must not hide dependencies behind controller lambda wrappers; this syntax "
        "guard is not a claim that all remaining callback dependencies are final ownership boundaries."
    )


def test_slider_preview_launch_wiring_does_not_wrap_dependencies_in_lambdas() -> None:
    _target, _source, tree = _repo_source_tree("kindred/gui/controllers/simulation_controller.py")
    slider_dependency_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "SimulationSliderPreviewLaunchDependencies"
    ]
    assert len(slider_dependency_calls) == 1

    lambda_fields = [
        keyword.arg
        for keyword in slider_dependency_calls[0].keywords
        if isinstance(keyword.value, ast.Lambda)
    ]
    assert lambda_fields == [], (
        "Slider preview launch wiring must not hide runtime/readiness/replay dependencies behind controller lambda "
        "wrappers; broader owner migration is guarded by responsibility-specific tests."
    )


def test_slider_preview_launch_reads_preview_request_from_run_state_not_dependency() -> None:
    _target, _source, tree = _repo_source_tree("kindred/gui/controllers/simulation_slider_preview_launch.py")
    dependency_node = _class_node(tree, "SimulationSliderPreviewLaunchDependencies")
    dependency_fields = {
        stmt.target.id
        for stmt in dependency_node.body
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)
    }

    assert "preview_owner_request_id" not in dependency_fields
    assert "run_simulation_internal" not in dependency_fields


def test_contained_serial_worker_launch_wiring_does_not_wrap_dependencies_in_lambdas() -> None:
    _target, _source, tree = _repo_source_tree("kindred/gui/controllers/simulation_controller.py")
    launch_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "ContainedSerialWorkerLaunchOwner"
    ]
    assert len(launch_calls) == 1

    lambda_fields = [
        keyword.arg
        for keyword in launch_calls[0].keywords
        if isinstance(keyword.value, ast.Lambda)
    ]
    keyword_fields = {keyword.arg for keyword in launch_calls[0].keywords}
    assert lambda_fields == [], (
        "Contained serial worker launch must not hide runtime acquisition behind controller lambda wrappers."
    )
    assert "acquire_ready_owner_for_plan" not in keyword_fields
    assert "runtime_application" in keyword_fields


def test_removed_controller_scaffolding_methods_do_not_reappear() -> None:
    _target, _source, tree = _repo_source_tree("kindred/gui/controllers/simulation_controller.py")
    controller_node = _class_node(tree, "SimulationController")
    method_names = {stmt.name for stmt in controller_node.body if isinstance(stmt, ast.FunctionDef)}

    removed_pass_throughs = {
        "_slider_launch_run_simulation_internal",
        "_parallel_outcome_record_nonfatal_exception",
        "_slider_launch_supersede_parallel_batch_run_soft",
        "_slider_launch_ensure_parallel_batch_pool_eagerly_created",
        "_slider_launch_ensure_interactive_simulation_runtime_available_for_mode",
        "_contained_serial_acquire_ready_owner_for_plan",
        "_contained_serial_release_owner",
        "_run_preparation_batch_mechanism_signature",
    }

    assert method_names.isdisjoint(removed_pass_throughs), (
        "Demolished controller pass-through methods must not reappear as named wrappers; remaining controller "
        "dependencies need responsibility-specific justification."
    )


def test_simulation_complete_provenance_cluster_uses_explicit_provenance_port() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    target = repo_root / "kindred" / "gui" / "controllers" / "simulation_completion_publication.py"
    assert target.is_file(), f"Expected file at {target}"

    source = target.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(target))
    fn = _completion_publication_method_node(tree, "publish_annotations_and_provenance")
    lines = source.splitlines()

    flattened_hits: list[_CallHit] = []
    explicit_methods: set[str] = set()
    low_level_hits: list[_CallHit] = []
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        chain = _attribute_chain(node.func)
        if not chain:
            continue
        if len(chain) == 3 and chain[:2] in {("self", "ui"), ("self", "_ui")} and chain[2] in TARGET_METHODS:
            flattened_hits.append(
                _CallHit(
                    method=chain[2],
                    lineno=node.lineno,
                    line=lines[node.lineno - 1].strip(),
                )
            )
            continue
        if len(chain) == 4 and chain[:3] in {("self", "ui", "provenance"), ("self", "_ui", "provenance")} and chain[3] in TARGET_METHODS:
            low_level_hits.append(
                _CallHit(
                    method=chain[3],
                    lineno=node.lineno,
                    line=lines[node.lineno - 1].strip(),
                )
            )
            continue
        if (
            len(chain) == 4
            and chain[:3] in {("self", "ui", "provenance"), ("self", "_ui", "provenance")}
            and chain[3] == "publish_simulation_completion_provenance"
        ):
            explicit_methods.add(chain[3])

    assert explicit_methods == {"publish_simulation_completion_provenance"}, (
        "Guardrail expectation changed: `_on_simulation_complete` must route completion provenance through "
        "the provenance owner publication boundary, but found "
        f"{sorted(explicit_methods)}."
    )

    assert low_level_hits == [], (
        "Guardrail violated: `_on_simulation_complete` must not call low-level provenance/CTC methods directly.\n"
        + "\n".join(
            f"{target.relative_to(repo_root)}:{hit.lineno}: `{hit.line}`"
            for hit in sorted(low_level_hits, key=lambda hit: (hit.lineno, hit.method))
        )
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
        ("_resolve_wegscheider_cyclicity_for_run_or_abort", {"choose_wegscheider_resolution", "message_box_warning"}),
        ("_start_parallel_batch_simulations", set()),
        ("_submit_parallel_batch_tasks", {"message_box_warning"}),
        ("_start_next_batch_simulation", set()),
        ("_abort_serial_batch_for_invalid_initials", {"message_box_warning"}),
        ("_run_simulation_internal", set()),
        ("_run_rows_or_abort", {"message_box_warning"}),
        ("_run_mechanism_context_or_abort", set()),
        ("_run_solver_context_or_abort", set()),
        ("_build_run_dispatch_context_or_abort", set()),
        ("_on_simulation_complete", set()),
        ("_on_simulation_error", set()),
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
        ("_resolve_wegscheider_cyclicity_for_run_or_abort", {"wegscheider_cyclicity_enabled"}),
        (
            "_run_solver_context_or_abort",
            set(),
        ),
        ("_run_simulation_internal", set()),
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
    ("class_name", "method_name", "explicit_port", "target_methods", "expected_methods"),
    (
        (
            "SimulationRunMechanismPreparationOwner",
            "build_mechanism_context_or_abort",
            "dialogs",
            DIALOGS_TARGET_METHODS,
            {"message_box_warning"},
        ),
        (
            "SimulationRunSolverPreparationOwner",
            "build_solver_context_or_abort",
            "dialogs",
            DIALOGS_TARGET_METHODS,
            {"message_box_warning"},
        ),
        (
            "SimulationRunDispatchPreparationOwner",
            "build_dispatch_context_or_abort",
            "dialogs",
            DIALOGS_TARGET_METHODS,
            {"message_box_warning"},
        ),
        (
            "SimulationRunSolverPreparationOwner",
            "build_solver_context_or_abort",
            "solver",
            REMAINING_SOLVER_TARGET_METHODS,
            REMAINING_SOLVER_TARGET_METHODS,
        ),
        (
            "SimulationRunMechanismPreparationOwner",
            "build_mechanism_context_or_abort",
            "mechanism",
            MECHANISM_TARGET_METHODS,
            {
                "apply_overrides_to_state_network_dsl",
                "apply_overrides_to_text",
                "has_slider_overrides",
                "mechanism_reactions_text_raw",
                "mechanism_state_network_dsl_raw",
            },
        ),
        (
            "SimulationRunSolverPreparationOwner",
            "build_solver_context_or_abort",
            "mechanism",
            MECHANISM_TARGET_METHODS,
            {
                "mechanism_slider_points_value",
                "mechanism_slider_solver_value",
            },
        ),
    ),
)
def test_simulation_run_preparation_owners_use_explicit_ports(
    class_name: str,
    method_name: str,
    explicit_port: str,
    target_methods: set[str],
    expected_methods: set[str],
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    target = repo_root / "kindred" / "gui" / "controllers" / "simulation_run_preparation.py"
    assert target.is_file(), f"Expected file at {target}"

    source = target.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(target))
    fn = _class_method_node(tree, class_name, method_name)
    lines = source.splitlines()

    flattened_hits, explicit_methods = _collect_port_usage(
        fn,
        lines,
        explicit_port=explicit_port,
        methods=target_methods,
    )

    assert explicit_methods == expected_methods, (
        f"Guardrail expectation changed: `{class_name}.{method_name}` must route audited {explicit_port} calls "
        f"through `self._ports.{explicit_port}`, but only found {sorted(explicit_methods)}."
    )

    assert flattened_hits == [], (
        f"Guardrail violated: `{class_name}.{method_name}` must not use flattened `self.ui.<method>` access "
        f"for audited {explicit_port} port methods.\n"
        + "\n".join(
            f"{target.relative_to(repo_root)}:{hit.lineno}: `{hit.line}`"
            for hit in sorted(flattened_hits, key=lambda hit: (hit.lineno, hit.method))
        )
    )


@pytest.mark.parametrize(
    ("method_name", "expected_methods"),
    (
        (
            "_slider_execution_parameter_values",
            {
                "slider_overrides",
                "variable_slider_values",
            },
        ),
        (
            "_slider_runtime_parameter_names",
            {
                "slider_overrides",
                "variable_slider_values",
            },
        ),
        (
            "_serial_batch_dispatch_state",
            {
                "slider_overrides",
            },
        ),
        ("_start_next_batch_simulation", set()),
        (
            "_run_mechanism_context_or_abort",
            set(),
        ),
        (
            "_run_solver_context_or_abort",
            set(),
        ),
        ("_run_simulation_internal", set()),
        (
            "_run_simulation",
            {
                "auto_lock_for_run",
                "is_mechanism_ready_for_run",
            },
        ),
        (
            "_resolve_wegscheider_cyclicity_for_run_or_abort",
            {
                "apply_wegscheider_resolution_source_rewrite",
                "mechanism_reactions_text_raw",
            },
        ),
            ("_on_simulation_complete", set()),
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
            },
        ),
    ),
)
def test_simulation_controller_cached_batch_selection_cluster_uses_explicit_batch_port(
    method_name: str, expected_methods: set[str]
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    target = repo_root / "kindred" / "gui" / "controllers" / "simulation_controller.py"
    if method_name == "_on_simulation_complete":
        target = repo_root / "kindred" / "gui" / "controllers" / "simulation_completion_publication.py"
    assert target.is_file(), f"Expected file at {target}"

    source = target.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(target))
    lines = source.splitlines()

    flattened_hits: list[_CallHit] = []
    explicit_methods: set[str] = set()
    method_names = (
        ("publish_display",)
        if method_name == "_on_simulation_complete"
        else (method_name,)
    )
    for inspected_method in method_names:
        fn = (
            _completion_publication_method_node(tree, inspected_method)
            if method_name == "_on_simulation_complete"
            else _simulation_controller_method_node(tree, inspected_method)
        )
        for node in ast.walk(fn):
            if not isinstance(node, ast.Call):
                continue
            chain = _attribute_chain(node.func)
            if not chain:
                continue
            if len(chain) == 3 and chain[:2] in {("self", "ui"), ("self", "_ui")} and chain[2] in BATCH_TARGET_METHODS:
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
                and chain[:3] in {("self", "ui", "batch"), ("self", "_ui", "batch")}
                and chain[3] in BATCH_TARGET_METHODS
            ):
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
    lines = source.splitlines()

    flattened_hits: list[_CallHit] = []
    explicit_methods: set[str] = set()
    prep_target = repo_root / "kindred" / "gui" / "controllers" / "simulation_run_preparation.py"
    prep_source = prep_target.read_text(encoding="utf-8")
    prep_tree = ast.parse(prep_source, filename=str(prep_target))
    prep_lines = prep_source.splitlines()

    controller_methods = ("_run_simulation_internal",)
    prep_methods = (
        ("SimulationRunMechanismPreparationOwner", "build_mechanism_context_or_abort"),
        ("SimulationRunDispatchPreparationOwner", "build_dispatch_context_or_abort"),
    )
    for method_name in controller_methods:
        fn = _simulation_controller_method_node(tree, method_name)
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
    for class_name, method_name in prep_methods:
        fn = _class_method_node(prep_tree, class_name, method_name)
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
                        line=prep_lines[node.lineno - 1].strip(),
                    )
                )
                continue
            if (
                len(chain) == 4
                and chain[:3] in {
                    ("self", "ui", "batch"),
                    ("self", "_ports", "batch"),
                }
                and chain[3] in QUEUE_CONTEXT_BATCH_TARGET_METHODS
            ):
                explicit_methods.add(chain[3])

    assert explicit_methods == QUEUE_CONTEXT_BATCH_TARGET_METHODS, (
        "Guardrail expectation changed: run preparation must route the queue/context batch subcluster through "
        f"`self.ui.batch`, but only found {sorted(explicit_methods)}."
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
        ("_start_parallel_batch_simulations", set()),
        ("_submit_parallel_batch_tasks", set()),
        (
            "_start_next_batch_simulation",
            {
                "batch_set_name_for_id",
            },
        ),
        ("_abort_serial_batch_for_invalid_initials", {"batch_model_validate_rows"}),
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


def test_batch_dispatch_materialization_owner_owns_batch_initials_and_preview_overlay() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    target = repo_root / "kindred" / "gui" / "controllers" / "batch_dispatch_materialization.py"
    assert target.is_file(), f"Expected file at {target}"

    source = target.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(target))
    fn = _class_method_node(tree, "BatchDispatchMaterializationOwner", "materialize_initials")

    calls = {
        chain
        for node in ast.walk(fn)
        if isinstance(node, ast.Call) and (chain := _attribute_chain(node.func)) is not None
    }
    assert ("self", "_batch", "batch_initials_for_row") in calls
    assert ("self", "_slider", "preview_initials_for_row") in calls


def test_simulation_complete_completion_reconciliation_cluster_uses_explicit_batch_port() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    controller_target = repo_root / "kindred" / "gui" / "controllers" / "simulation_controller.py"
    owner_target = repo_root / "kindred" / "gui" / "controllers" / "simulation_completion_publication.py"
    materialization_target = repo_root / "kindred" / "gui" / "controllers" / "simulation_result_materialization.py"
    assert controller_target.is_file(), f"Expected file at {controller_target}"
    assert owner_target.is_file(), f"Expected file at {owner_target}"
    assert materialization_target.is_file(), f"Expected file at {materialization_target}"

    materialization_source = materialization_target.read_text(encoding="utf-8")
    materialization_tree = ast.parse(materialization_source, filename=str(materialization_target))
    method_specs = (
        (
            materialization_target,
            materialization_tree,
            materialization_source.splitlines(),
            _class_method_node(
                materialization_tree,
                "SimulationResultMaterializationOwner",
                "remember_primary_result_mechanism",
            ),
        ),
    )

    flattened_hits: list[_CallHit] = []
    explicit_methods: set[str] = set()
    for _target, _tree, lines, fn in method_specs:
        for node in ast.walk(fn):
            if not isinstance(node, ast.Call):
                continue
            chain = _attribute_chain(node.func)
            if not chain:
                continue
            if (
                len(chain) == 3
                and chain[:2] in {("self", "ui"), ("self", "_ui")}
                and chain[2] in COMPLETION_RECONCILIATION_BATCH_TARGET_METHODS
            ):
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
                and chain[:3] in {("self", "ui", "batch"), ("self", "_ui", "batch")}
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
            f"{controller_target.relative_to(repo_root)}:{hit.lineno}: `{hit.line}`"
            for hit in sorted(flattened_hits, key=lambda hit: (hit.lineno, hit.method))
        )
    )


def test_simulation_complete_mechanism_helpers_cluster_uses_explicit_mechanism_helpers_port() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    controller_target = repo_root / "kindred" / "gui" / "controllers" / "simulation_controller.py"
    owner_target = repo_root / "kindred" / "gui" / "controllers" / "simulation_completion_publication.py"
    materialization_target = repo_root / "kindred" / "gui" / "controllers" / "simulation_result_materialization.py"
    assert controller_target.is_file(), f"Expected file at {controller_target}"
    assert owner_target.is_file(), f"Expected file at {owner_target}"
    assert materialization_target.is_file(), f"Expected file at {materialization_target}"

    controller_source = controller_target.read_text(encoding="utf-8")
    controller_tree = ast.parse(controller_source, filename=str(controller_target))
    owner_source = owner_target.read_text(encoding="utf-8")
    owner_tree = ast.parse(owner_source, filename=str(owner_target))
    materialization_source = materialization_target.read_text(encoding="utf-8")
    materialization_tree = ast.parse(materialization_source, filename=str(materialization_target))
    method_specs = (
        (controller_target, controller_source.splitlines(), _simulation_controller_method_node(controller_tree, "_finalize_explicit_batch_dirty_reset")),
        (owner_target, owner_source.splitlines(), _completion_publication_method_node(owner_tree, "apply_pending_init")),
        (owner_target, owner_source.splitlines(), _completion_publication_method_node(owner_tree, "apply_pending_init_guard")),
        (materialization_target, materialization_source.splitlines(), _class_method_node(materialization_tree, "SimulationResultMaterializationOwner", "update_primary_result_materialization_contract")),
        (materialization_target, materialization_source.splitlines(), _class_method_node(materialization_tree, "SimulationResultMaterializationOwner", "remember_primary_result_mechanism")),
        (materialization_target, materialization_source.splitlines(), _class_method_node(materialization_tree, "SimulationResultMaterializationOwner", "refresh_primary_result_controls")),
    )

    flattened_hits: list[_CallHit] = []
    explicit_methods: set[str] = set()
    for _target, lines, fn in method_specs:
        for node in ast.walk(fn):
            if not isinstance(node, ast.Call):
                continue
            chain = _attribute_chain(node.func)
            if not chain:
                continue
            if (
                len(chain) == 3
                and chain[:2] in {("self", "ui"), ("self", "_ui")}
                and chain[2] in COMPLETION_MECHANISM_HELPERS_TARGET_METHODS
            ):
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
                and chain[:3] in {("self", "ui", "mechanism_helpers"), ("self", "_ui", "mechanism_helpers")}
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
            f"{controller_target.relative_to(repo_root)}:{hit.lineno}: `{hit.line}`"
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
    fn = _simulation_controller_method_node(tree, "_sync_batch_species_columns_for_run")
    lines = source.splitlines()

    flattened_hits, explicit_methods = _collect_port_usage(
        fn,
        lines,
        explicit_port="mechanism_helpers",
        methods=MECHANISM_HELPERS_SNAPSHOT_TARGET_METHODS,
    )

    assert explicit_methods == MECHANISM_HELPERS_SNAPSHOT_TARGET_METHODS, (
        "Guardrail expectation changed: `SimulationController._sync_batch_species_columns_for_run` must route mechanism "
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

    assert explicit_methods == set(), (
        "Guardrail expectation changed: `SimulationController._run_simulation_internal` should not perform "
        "GUI-thread preview-runtime preparation. Runtime readiness is now owned by the runtime/application "
        f"readiness path, but found runtime-port calls {sorted(explicit_methods)}."
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


def test_run_simulation_internal_delegates_plan_payload_construction() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    target = repo_root / "kindred" / "gui" / "controllers" / "simulation_controller.py"
    assert target.is_file(), f"Expected file at {target}"

    source = target.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(target))
    fn = _simulation_controller_method_node(tree, "_run_simulation_internal")
    lines = source.splitlines()

    forbidden_calls: list[_CallHit] = []
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        chain = _attribute_chain(node.func)
        name = chain[-1] if chain else ""
        if name in {"SimulationExecutionRequest", "_new_simulation_plan_payload"}:
            forbidden_calls.append(
                _CallHit(
                    method=name,
                    lineno=node.lineno,
                    line=lines[node.lineno - 1].strip(),
                )
            )

    assert forbidden_calls == [], (
        "`SimulationController._run_simulation_internal` must delegate per-set execution request and "
        "SimulationPlan payload construction to the batch dispatch plan boundary.\n"
        + "\n".join(
            f"{target.relative_to(repo_root)}:{hit.lineno}: `{hit.line}`"
            for hit in sorted(forbidden_calls, key=lambda hit: (hit.lineno, hit.method))
        )
    )


def test_runtime_readiness_builder_delegates_plan_payload_construction() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    target = repo_root / "kindred" / "gui" / "controllers" / "simulation_controller.py"
    assert target.is_file(), f"Expected file at {target}"

    source = target.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(target))
    fn = _simulation_controller_method_node(tree, "_build_runtime_readiness_plan_payloads")
    lines = source.splitlines()

    forbidden_calls: list[_CallHit] = []
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        chain = _attribute_chain(node.func)
        name = chain[-1] if chain else ""
        if name in {"SimulationExecutionRequest", "_new_simulation_plan_payload"}:
            forbidden_calls.append(
                _CallHit(
                    method=name,
                    lineno=node.lineno,
                    line=lines[node.lineno - 1].strip(),
                )
            )

    assert forbidden_calls == [], (
        "`SimulationController._build_runtime_readiness_plan_payloads` must share the batch dispatch plan "
        "boundary instead of preserving a second local SimulationPlan construction policy.\n"
        + "\n".join(
            f"{target.relative_to(repo_root)}:{hit.lineno}: `{hit.line}`"
            for hit in sorted(forbidden_calls, key=lambda hit: (hit.lineno, hit.method))
        )
    )


def test_campaign_a_runtime_controller_entrypoints_are_bounded_orchestrators() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    target = repo_root / "kindred" / "gui" / "controllers" / "simulation_controller.py"
    assert target.is_file(), f"Expected file at {target}"

    source = target.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(target))
    lines = source.splitlines()
    max_lines_by_method = {
        "_run_simulation_internal": 240,
        "_start_parallel_batch_simulations": 140,
        "_start_next_batch_simulation": 140,
    }

    hits: list[_CallHit] = []
    for method_name, max_lines in max_lines_by_method.items():
        fn = _simulation_controller_method_node(tree, method_name)
        line_count = int(fn.end_lineno or fn.lineno) - int(fn.lineno) + 1
        if line_count > max_lines:
            hits.append(
                _CallHit(
                    method=method_name,
                    lineno=fn.lineno,
                    line=f"{method_name} spans {line_count} lines (max {max_lines}): {lines[fn.lineno - 1].strip()}",
                )
            )

    assert hits == [], (
        "Campaign A runtime controller entrypoints must remain bounded orchestrators; large policy bodies need "
        "real owner/helper boundaries plus the responsibility-specific guards in this file.\n"
        + "\n".join(
            f"{target.relative_to(repo_root)}:{hit.lineno}: `{hit.line}`"
            for hit in sorted(hits, key=lambda hit: (hit.lineno, hit.method))
        )
    )


def test_start_run_context_and_dispatch_delegates_start_request_construction() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    target = repo_root / "kindred" / "gui" / "controllers" / "simulation_controller.py"
    assert target.is_file(), f"Expected file at {target}"

    source = target.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(target))
    lines = source.splitlines()
    fn = _simulation_controller_method_node(tree, "_start_run_context_and_dispatch")

    forbidden_calls: list[_CallHit] = []
    forbidden_names = {
        "BatchRunStartRequest",
        "batch_mechanism_signature",
        "coerce_simulation_identity",
    }
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        chain = _attribute_chain(node.func)
        name = chain[-1] if chain else ""
        if name in forbidden_names:
            forbidden_calls.append(
                _CallHit(
                    method=name,
                    lineno=node.lineno,
                    line=lines[node.lineno - 1].strip(),
                )
            )

    assert forbidden_calls == [], (
        "`SimulationController._start_run_context_and_dispatch` must compose around the run-preparation boundary; "
        "batch-run start request construction and primary-signature policy belong outside the controller.\n"
        + "\n".join(
            f"{target.relative_to(repo_root)}:{hit.lineno}: `{hit.line}`"
            for hit in sorted(forbidden_calls, key=lambda hit: (hit.lineno, hit.method))
        )
    )


def test_start_parallel_batch_simulations_delegates_runtime_readiness_decision() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    target = repo_root / "kindred" / "gui" / "controllers" / "simulation_controller.py"
    assert target.is_file(), f"Expected file at {target}"

    source = target.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(target))
    lines = source.splitlines()
    fn = _simulation_controller_method_node(tree, "_start_parallel_batch_simulations")

    forbidden_names = {
        "current_max_workers",
        "has_lane_pool",
        "has_ready_lane_pool",
        "ensure_lane_pool",
        "is_pool_stale",
    }
    delegated = False
    hits: list[_CallHit] = []
    for node in ast.walk(fn):
        if isinstance(node, ast.Call):
            chain = _attribute_chain(node.func)
            if chain == ("self", "_parallel_batch_runtime_readiness_owner", "run_start_availability"):
                delegated = True
            if chain and chain[0:2] == ("self", "_batch_parallel") and chain[-1] in forbidden_names:
                hits.append(
                    _CallHit(
                        method=chain[-1],
                        lineno=node.lineno,
                        line=lines[node.lineno - 1].strip(),
                    )
                )
        elif isinstance(node, ast.Attribute):
            chain = _attribute_chain(node)
            if chain and chain[0:2] == ("self", "_batch_parallel") and chain[-1] in forbidden_names:
                hits.append(
                    _CallHit(
                        method=chain[-1],
                        lineno=node.lineno,
                        line=lines[node.lineno - 1].strip(),
                    )
                )

    assert delegated is True, (
        "`SimulationController._start_parallel_batch_simulations` must ask "
        "`ParallelBatchRuntimeReadinessOwner` for run-path readiness instead of recomputing lane-pool truth inline."
    )
    assert hits == [], (
        "`SimulationController._start_parallel_batch_simulations` must not duplicate parallel batch runtime readiness "
        "truth from the batch executor; that decision belongs to `ParallelBatchRuntimeReadinessOwner`.\n"
        + "\n".join(
            f"{target.relative_to(repo_root)}:{hit.lineno}: `{hit.line}`"
            for hit in sorted(hits, key=lambda hit: (hit.lineno, hit.method))
        )
    )


def test_completion_materialization_policy_is_not_controller_local() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    controller_target = repo_root / "kindred" / "gui" / "controllers" / "simulation_controller.py"
    owner_target = repo_root / "kindred" / "gui" / "controllers" / "simulation_result_materialization.py"
    assert controller_target.is_file(), f"Expected file at {controller_target}"
    assert owner_target.is_file(), f"Expected file at {owner_target}"

    controller_source = controller_target.read_text(encoding="utf-8")
    controller_tree = ast.parse(controller_source, filename=str(controller_target))
    owner_source = owner_target.read_text(encoding="utf-8")
    owner_tree = ast.parse(owner_source, filename=str(owner_target))
    controller_methods = {
        node.name
        for node in ast.walk(controller_tree)
        if isinstance(node, ast.FunctionDef)
    }
    materialization_methods = {
        "resolve_completion_mechanism",
        "update_primary_result_materialization_contract",
        "remember_primary_result_mechanism",
        "refresh_primary_result_controls",
    }

    assert not (controller_methods & {f"_{name}" for name in materialization_methods}), (
        "Completion materialization policy must live in `SimulationResultMaterializationOwner`, not as "
        "controller-local callback dependencies."
    )
    _class_method_node(owner_tree, "SimulationResultMaterializationOwner", "resolve_completion_mechanism")
    _class_method_node(
        owner_tree,
        "SimulationResultMaterializationOwner",
        "update_primary_result_materialization_contract",
    )
    _class_method_node(owner_tree, "SimulationResultMaterializationOwner", "remember_primary_result_mechanism")
    _class_method_node(owner_tree, "SimulationResultMaterializationOwner", "refresh_primary_result_controls")


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
    prep_target = repo_root / "kindred" / "gui" / "controllers" / "simulation_run_preparation.py"
    callback_target = repo_root / "kindred" / "gui" / "controllers" / "simulation_completion_callback.py"
    materialization_target = repo_root / "kindred" / "gui" / "controllers" / "batch_dispatch_materialization.py"
    result_materialization_target = repo_root / "kindred" / "gui" / "controllers" / "simulation_result_materialization.py"
    assert target.is_file(), f"Expected file at {target}"
    assert prep_target.is_file(), f"Expected file at {prep_target}"
    assert callback_target.is_file(), f"Expected file at {callback_target}"
    assert materialization_target.is_file(), f"Expected file at {materialization_target}"
    assert result_materialization_target.is_file(), f"Expected file at {result_materialization_target}"

    source = target.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(target))
    lines = source.splitlines()
    prep_source = prep_target.read_text(encoding="utf-8")
    prep_tree = ast.parse(prep_source, filename=str(prep_target))
    prep_lines = prep_source.splitlines()
    callback_source = callback_target.read_text(encoding="utf-8")
    callback_tree = ast.parse(callback_source, filename=str(callback_target))
    callback_lines = callback_source.splitlines()
    materialization_source = materialization_target.read_text(encoding="utf-8")
    materialization_tree = ast.parse(materialization_source, filename=str(materialization_target))
    materialization_lines = materialization_source.splitlines()
    result_materialization_source = result_materialization_target.read_text(encoding="utf-8")
    result_materialization_tree = ast.parse(result_materialization_source, filename=str(result_materialization_target))
    result_materialization_lines = result_materialization_source.splitlines()

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
    for node in ast.walk(callback_tree):
        if not isinstance(node, ast.ClassDef) or node.name != "SimulationCompletionCallbackOwner":
            continue
        for item in node.body:
            if not isinstance(item, ast.FunctionDef):
                continue
            method_hits, method_explicit = _collect_port_usage(
                item,
                callback_lines,
                explicit_port="slider",
                methods=SLIDER_TARGET_METHODS,
            )
            flattened_hits.extend(method_hits)
            explicit_methods.update(method_explicit)
    for node in ast.walk(prep_tree):
        if not isinstance(node, ast.ClassDef) or node.name not in {
            "SimulationRunMechanismPreparationOwner",
            "SimulationRunSolverPreparationOwner",
            "SimulationRunDispatchPreparationOwner",
        }:
            continue
        for item in node.body:
            if not isinstance(item, ast.FunctionDef):
                continue
            method_hits, method_explicit = _collect_port_usage(
                item,
                prep_lines,
                explicit_port="slider",
                methods=SLIDER_TARGET_METHODS,
            )
            flattened_hits.extend(method_hits)
            explicit_methods.update(method_explicit)
    materialize_fn = _class_method_node(
        materialization_tree,
        "BatchDispatchMaterializationOwner",
        "materialize_initials",
    )
    for node in ast.walk(materialize_fn):
        if not isinstance(node, ast.Call):
            continue
        chain = _attribute_chain(node.func)
        if chain == ("self", "_slider", "preview_initials_for_row"):
            explicit_methods.add("preview_initials_for_row")
        elif chain == ("self", "ui", "preview_initials_for_row"):
            flattened_hits.append(
                _CallHit(
                    method="preview_initials_for_row",
                    lineno=node.lineno,
                    line=materialization_lines[node.lineno - 1].strip(),
                )
            )
    for node in ast.walk(result_materialization_tree):
        if not isinstance(node, ast.ClassDef) or node.name != "SimulationResultMaterializationOwner":
            continue
        for item in node.body:
            if not isinstance(item, ast.FunctionDef):
                continue
            method_hits, method_explicit = _collect_port_usage(
                item,
                result_materialization_lines,
                explicit_port="slider",
                methods=SLIDER_TARGET_METHODS,
            )
            flattened_hits.extend(method_hits)
            explicit_methods.update(method_explicit)

    assert explicit_methods == SLIDER_TARGET_METHODS, (
        "Guardrail expectation changed: runtime GUI owners must route the audited slider cluster through "
        f"`self.ui.slider`, but only found {sorted(explicit_methods)}."
    )
    assert flattened_hits == [], (
        "Guardrail violated: runtime GUI owners must not use flattened `self.ui.<method>` access for "
        "SimulationSliderPort methods.\n"
        + "\n".join(
            f"{target.relative_to(repo_root)}:{hit.lineno}: `{hit.line}`"
            for hit in sorted(flattened_hits, key=lambda hit: (hit.lineno, hit.method))
        )
    )


def test_simulation_complete_results_cluster_uses_explicit_results_port() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    target = repo_root / "kindred" / "gui" / "controllers" / "simulation_completion_publication.py"
    assert target.is_file(), f"Expected file at {target}"

    source = target.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(target))
    method_names = (
        "publish_display",
        "publish_annotations_and_provenance",
    )
    lines = source.splitlines()

    flattened_hits: list[_CallHit] = []
    explicit_methods: set[str] = set()
    for method_name in method_names:
        fn = _completion_publication_method_node(tree, method_name)
        for node in ast.walk(fn):
            if not isinstance(node, ast.Call):
                continue
            chain = _attribute_chain(node.func)
            if not chain:
                continue
            if (
                len(chain) == 3
                and chain[:2] in {("self", "ui"), ("self", "_ui")}
                and chain[2] in COMPLETION_RESULTS_TARGET_METHODS
            ):
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
                and chain[:3] in {("self", "ui", "results"), ("self", "_ui", "results")}
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
    controller_target = repo_root / "kindred" / "gui" / "controllers" / "simulation_controller.py"
    owner_target = repo_root / "kindred" / "gui" / "controllers" / "simulation_completion_publication.py"
    materialization_target = repo_root / "kindred" / "gui" / "controllers" / "simulation_result_materialization.py"
    assert controller_target.is_file(), f"Expected file at {controller_target}"
    assert owner_target.is_file(), f"Expected file at {owner_target}"
    assert materialization_target.is_file(), f"Expected file at {materialization_target}"

    owner_source = owner_target.read_text(encoding="utf-8")
    owner_tree = ast.parse(owner_source, filename=str(owner_target))
    materialization_source = materialization_target.read_text(encoding="utf-8")
    materialization_tree = ast.parse(materialization_source, filename=str(materialization_target))
    method_specs = (
        (materialization_target, materialization_source.splitlines(), _class_method_node(materialization_tree, "SimulationResultMaterializationOwner", "resolve_completion_mechanism")),
        (owner_target, owner_source.splitlines(), _completion_publication_method_node(owner_tree, "publish_annotations_and_provenance")),
    )
    flattened_hits: list[_CallHit] = []
    explicit_methods: set[str] = set()
    for _target, lines, fn in method_specs:
        for node in ast.walk(fn):
            if not isinstance(node, ast.Call):
                continue
            chain = _attribute_chain(node.func)
            if not chain:
                continue
            if (
                len(chain) == 3
                and chain[:2] in {("self", "ui"), ("self", "_ui")}
                and chain[2] in COMPLETION_SOLVER_TARGET_METHODS
            ):
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
                and chain[:3] in {("self", "ui", "solver"), ("self", "_ui", "solver")}
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
            f"{controller_target.relative_to(repo_root)}:{hit.lineno}: `{hit.line}`"
            for hit in sorted(flattened_hits, key=lambda hit: (hit.lineno, hit.method))
        )
    )


def test_simulation_complete_callback_does_not_own_completion_publication_policy() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    target = repo_root / "kindred" / "gui" / "controllers" / "simulation_controller.py"
    source = target.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(target))
    fn = _simulation_controller_method_node(tree, "_on_simulation_complete")
    lines = source.splitlines()

    forbidden_methods = {
        "record_preview_completion_cache_key",
        "apply_explicit_cache_reconciliation",
        "put_completion_entry",
        "display_cached_batch_selection",
        "set_intervention_annotations_from_provenance",
        "integrate_ctc",
        "set_last_simulation_ctc",
        "set_last_simulation_provenance",
    }
    hits: list[tuple[Path, _CallHit]] = []
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        chain = _attribute_chain(node.func)
        if chain and chain[-1] in forbidden_methods:
            hits.append(
                _CallHit(
                    method=chain[-1],
                    lineno=node.lineno,
                    line=lines[node.lineno - 1].strip(),
                )
            )

    assert hits == [], (
        "`SimulationController._on_simulation_complete` must orchestrate typed completion effects, not directly "
        "own cache/display/provenance publication policy.\n"
        + "\n".join(
            f"{target.relative_to(repo_root)}:{hit.lineno}: `{hit.line}`"
            for hit in sorted(hits, key=lambda hit: (hit.lineno, hit.method))
        )
    )


def test_simulation_complete_callback_remains_small_orchestration_boundary() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    target = repo_root / "kindred" / "gui" / "controllers" / "simulation_controller.py"
    source = target.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(target))
    fn = _simulation_controller_method_node(tree, "_on_simulation_complete")

    max_callback_lines = 120
    max_callback_statements = 50
    line_count = int(fn.end_lineno or fn.lineno) - int(fn.lineno) + 1
    statement_count = sum(1 for node in ast.walk(fn) if isinstance(node, ast.stmt))

    assert line_count <= max_callback_lines and statement_count <= max_callback_statements, (
        "`SimulationController._on_simulation_complete` must stay a small callback orchestrator; "
        "completion cache/display/provenance/runtime/batch policy belongs in named owner/effect boundaries. "
        f"Observed {line_count} lines and {statement_count} AST statements."
    )


def test_simulation_completion_owner_methods_do_not_hide_a_relocated_monolith() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    target = repo_root / "kindred" / "gui" / "controllers" / "simulation_completion_publication.py"
    source = target.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(target))

    max_owner_method_lines = 140
    hits: list[_CallHit] = []
    lines = source.splitlines()
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if not (
            node.name.startswith("build_")
            or node.name.startswith("publish_")
            or node.name.startswith("apply_")
            or node.name.startswith("advance_")
            or node.name.startswith("finalize_")
            or node.name.startswith("resolve_")
        ):
            continue
        line_count = int(node.end_lineno or node.lineno) - int(node.lineno) + 1
        if line_count > max_owner_method_lines:
            hits.append(
                _CallHit(
                    method=node.name,
                    lineno=node.lineno,
                    line=f"{node.name} spans {line_count} lines: {lines[node.lineno - 1].strip()}",
                )
            )

    assert hits == [], (
        "Completion owner/effect helpers must own bounded decisions, not relocate the old callback monolith.\n"
        + "\n".join(
            f"{target.relative_to(repo_root)}:{hit.lineno}: `{hit.line}`"
            for hit in sorted(hits, key=lambda hit: (hit.lineno, hit.method))
        )
    )


def test_simulation_completion_builders_do_not_publish_or_apply_effects() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    target = repo_root / "kindred" / "gui" / "controllers" / "simulation_completion_publication.py"
    source = target.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(target))
    lines = source.splitlines()
    fn = _completion_publication_method_node(tree, "build_result_state")

    forbidden_self_calls = {
        "apply_algebra_status",
        "publish_cache_truth",
        "publish_primary_materialization",
        "publish_cache_entry",
        "publish_display",
        "publish_annotations_and_provenance",
    }
    hits: list[_CallHit] = []
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        chain = _attribute_chain(node.func)
        if len(chain or ()) == 2 and chain[0] == "self" and chain[1] in forbidden_self_calls:
            hits.append(
                _CallHit(
                    method=chain[1],
                    lineno=node.lineno,
                    line=lines[node.lineno - 1].strip(),
                )
            )

    assert hits == [], (
        "Completion result builders must construct typed data only; publication and lifecycle effects belong "
        "to explicit publication/effect methods.\n"
        + "\n".join(
            f"{target.relative_to(repo_root)}:{hit.lineno}: `{hit.line}`"
            for hit in sorted(hits, key=lambda hit: (hit.lineno, hit.method))
        )
    )


def test_simulation_completion_variable_runtime_policy_uses_runtime_port_not_mechanism_helpers() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    target = repo_root / "kindred" / "gui" / "controllers" / "simulation_result_materialization.py"
    source = target.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(target))
    lines = source.splitlines()
    runtime_methods = {
        "is_energy_mode_mechanism",
        "dsl_has_computational_mode_generated_block",
        "sync_energy_mode_temperature_from_mechanism",
        "populate_energy_mode_variables_from_mechanism",
        "extract_and_populate_variables",
    }
    checked_methods = {
        "_update_primary_result_materialization_contract",
        "_refresh_primary_result_controls",
    }
    hits: list[_CallHit] = []
    for method_name in checked_methods:
        fn = _class_method_node(
            tree,
            "SimulationResultMaterializationOwner",
            method_name.lstrip("_"),
        )
        for node in ast.walk(fn):
            if not isinstance(node, ast.Call):
                continue
            chain = _attribute_chain(node.func)
            if len(chain or ()) == 4 and chain[:3] == ("self", "_ui", "mechanism_helpers") and chain[3] in runtime_methods:
                hits.append(
                    _CallHit(
                        method=f"{method_name}:{chain[3]}",
                        lineno=node.lineno,
                        line=lines[node.lineno - 1].strip(),
                    )
                )

    assert hits == [], (
        "Completion materialization must route variable-runtime policy through `ui.runtime`; "
        "`ui.mechanism_helpers` owns mechanism snapshots and pending-init helpers, not variable runtime forwarding.\n"
        + "\n".join(
            f"{target.relative_to(repo_root)}:{hit.lineno}: `{hit.line}`"
            for hit in sorted(hits, key=lambda hit: (hit.lineno, hit.method))
        )
    )


def test_simulation_completion_callback_owner_applies_callback_policy_outside_controller() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    target = repo_root / "kindred" / "gui" / "controllers" / "simulation_completion_callback.py"
    source = target.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(target))
    lines = source.splitlines()
    fn = _class_method_node(tree, "SimulationCompletionCallbackOwner", "handle_completion")

    hits: list[_CallHit] = []
    expected_owner_calls = {
        "active_batch_context_runtime_input_stale_for_set",
        "mark_stale_runtime_input_callback_consumed",
        "effective_preview_owner_epoch_for_callback",
        "missing_preview_owner_epoch_for_current_fast_owner",
        "preview_request_matches_current_owner_epoch",
        "apply_completion_policy_state_patch",
        "apply_lifecycle_effects",
        "publish_success",
    }
    observed_owner_calls: set[str] = set()
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        chain = _attribute_chain(node.func)
        if chain and chain[-1] in expected_owner_calls:
            observed_owner_calls.add(chain[-1])
        if chain and chain[0] == "self" and chain[-1] in {
            "_apply_simulation_lifecycle_effects",
            "_apply_completion_policy_state_patch",
            "_mark_stale_runtime_input_callback_consumed",
            "_active_batch_context_runtime_input_stale_for_set",
        }:
            hits.append(
                _CallHit(
                    method=chain[1],
                    lineno=node.lineno,
                    line=lines[node.lineno - 1].strip(),
                )
            )

    missing = expected_owner_calls - observed_owner_calls
    assert missing == set(), (
        "Completion callback policy must live in `SimulationCompletionCallbackOwner`, not in "
        f"`SimulationController._on_simulation_complete`. Missing expected owner calls: {sorted(missing)}"
    )
    assert hits == [], (
        "Completion callback owner must route state mutation through injected dependencies instead of "
        "controller-private callback policy methods.\n"
        + "\n".join(
            f"{target.relative_to(repo_root)}:{hit.lineno}: `{hit.line}`"
            for hit in sorted(hits, key=lambda hit: (hit.lineno, hit.method))
        )
    )


def test_simulation_completion_callback_does_not_resolve_policy_in_controller() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    target = repo_root / "kindred" / "gui" / "controllers" / "simulation_controller.py"
    source = target.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(target))
    lines = source.splitlines()
    fn = _simulation_controller_method_node(tree, "_on_simulation_complete")

    forbidden_calls = {
        "resolve_superseded_fast_completion",
        "_active_batch_context_runtime_input_stale_for_set",
        "_mark_stale_runtime_input_callback_consumed",
        "_effective_preview_owner_epoch_for_callback",
        "_missing_preview_owner_epoch_for_current_fast_owner",
        "_preview_request_matches_current_owner_epoch",
        "_apply_completion_policy_state_patch",
        "_apply_simulation_lifecycle_effects",
        "publish_success",
    }
    hits: list[_CallHit] = []
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        chain = _attribute_chain(node.func)
        if chain and chain[-1] in forbidden_calls:
            hits.append(
                _CallHit(
                    method=chain[-1],
                    lineno=node.lineno,
                    line=lines[node.lineno - 1].strip(),
                )
            )

    assert hits == [], (
        "`SimulationController._on_simulation_complete` must delegate callback policy to "
        "`SimulationCompletionCallbackOwner`; it must not resolve stale/publication policy inline.\n"
        + "\n".join(
            f"{target.relative_to(repo_root)}:{hit.lineno}: `{hit.line}`"
            for hit in sorted(hits, key=lambda hit: (hit.lineno, hit.method))
        )
    )


def test_simulation_completion_and_error_callbacks_do_not_own_lifecycle_policy() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    target = repo_root / "kindred" / "gui" / "controllers" / "simulation_controller.py"
    source = target.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(target))
    lines = source.splitlines()

    forbidden_methods = {
        "_release_current_simulation_worker",
        "_cleanup_parallel_batch_lane_pool_after_run",
        "_shutdown_batch_lane_pool",
        "_close_contained_simulation_owner",
        "_schedule_deferred_preview_replay_handoff_once",
    }
    forbidden_ui_methods = {
        "set_run_button_enabled",
        "set_stop_button_enabled",
        "set_status_text",
        "set_sim_progress_value",
        "set_algebra_status_text",
        "repaint_simulation_widgets",
        "set_slider_triggered_simulation",
    }
    callback_names = {
        "_on_simulation_complete",
        "_on_simulation_error",
        "_handle_current_preview_simulation_failure",
    }
    hits: list[tuple[Path, _CallHit]] = []
    for callback_name in callback_names:
        fn = _simulation_controller_method_node(tree, callback_name)
        for node in ast.walk(fn):
            if not isinstance(node, ast.Call):
                continue
            chain = _attribute_chain(node.func)
            if not chain:
                continue
            if chain[-1] in forbidden_methods or chain[-1] in forbidden_ui_methods:
                hits.append(
                    _CallHit(
                        method=f"{callback_name}:{chain[-1]}",
                        lineno=node.lineno,
                        line=lines[node.lineno - 1].strip(),
                    )
                )

    assert hits == [], (
        "Simulation completion/error callbacks must ask typed lifecycle owners for effects and apply them "
        "through one effect applier, not inline lifecycle cleanup or run-control policy.\n"
        + "\n".join(
            f"{target.relative_to(repo_root)}:{hit.lineno}: `{hit.line}`"
            for hit in sorted(hits, key=lambda hit: (hit.lineno, hit.method))
        )
    )


def test_simulation_error_callback_delegates_error_policy_to_owner() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    target = repo_root / "kindred" / "gui" / "controllers" / "simulation_controller.py"
    source = target.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(target))
    fn = _simulation_controller_method_node(tree, "_on_simulation_error")
    lines = source.splitlines()

    max_callback_lines = 35
    line_count = int(fn.end_lineno or fn.lineno) - int(fn.lineno) + 1
    forbidden_calls = {
        "coerce_simulation_failure",
        "simulation_failure_user_message",
        "simulation_failure_detail_text",
        "is_cancelled_failure",
        "resolve_superseded_fast_error",
        "terminal_error_effects",
    }
    hits: list[_CallHit] = []
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        chain = _attribute_chain(node.func)
        if chain and chain[-1] in forbidden_calls:
            hits.append(
                _CallHit(
                    method=chain[-1],
                    lineno=node.lineno,
                    line=lines[node.lineno - 1].strip(),
                )
            )

    assert line_count <= max_callback_lines, (
        "`SimulationController._on_simulation_error` must stay a small delegate to the error handling owner; "
        f"observed {line_count} lines."
    )
    assert hits == [], (
        "`SimulationController._on_simulation_error` must not own simulation-error classification or terminal "
        "lifecycle policy.\n"
        + "\n".join(
            f"{target.relative_to(repo_root)}:{hit.lineno}: `{hit.line}`"
            for hit in sorted(hits, key=lambda hit: (hit.lineno, hit.method))
        )
    )


def test_simulation_controller_has_no_completion_error_dispatch_scaffold() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    target = repo_root / "kindred" / "gui" / "controllers" / "simulation_controller.py"
    source = target.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(target))
    lines = source.splitlines()
    controller = _class_node(tree, "SimulationController")

    forbidden_names = {
        "_dispatch_simulation_complete",
        "_dispatch_simulation_error",
    }
    method_hits = [
        _CallHit(method=item.name, lineno=item.lineno, line=lines[item.lineno - 1].strip())
        for item in controller.body
        if isinstance(item, ast.FunctionDef) and item.name in forbidden_names
    ]
    call_hits: list[_CallHit] = []
    for node in ast.walk(controller):
        if not isinstance(node, ast.Call):
            continue
        chain = _attribute_chain(node.func)
        if chain and chain[-1] in forbidden_names:
            call_hits.append(
                _CallHit(
                    method=chain[-1],
                    lineno=node.lineno,
                    line=lines[node.lineno - 1].strip(),
                )
            )

    assert method_hits == [], (
        "Completion/error callback identity must flow through `_on_simulation_complete` and "
        "`_on_simulation_error`; extra dispatch adapters recapture or branch identity before owner policy.\n"
        + "\n".join(
            f"{target.relative_to(repo_root)}:{hit.lineno}: `{hit.line}`"
            for hit in sorted(method_hits, key=lambda hit: (hit.lineno, hit.method))
        )
    )
    assert call_hits == [], (
        "Production controller code must not call removed completion/error dispatch adapters.\n"
        + "\n".join(
            f"{target.relative_to(repo_root)}:{hit.lineno}: `{hit.line}`"
            for hit in sorted(call_hits, key=lambda hit: (hit.lineno, hit.method))
        )
    )


def test_batch_run_context_owner_has_no_raw_context_compatibility_api() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    target = repo_root / "kindred" / "gui" / "controllers" / "batch_run_context_owner.py"
    source = target.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(target))

    raw_api_names = {"current", "snapshot", "replace"}
    hits: list[_CallHit] = []
    lines = source.splitlines()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or node.name != "BatchRunContextOwner":
            continue
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name in raw_api_names:
                hits.append(
                    _CallHit(
                        method=item.name,
                        lineno=item.lineno,
                        line=lines[item.lineno - 1].strip(),
                    )
                )

    assert hits == [], (
        "`BatchRunContextOwner` must expose typed owner-facing APIs, not raw dict compatibility APIs.\n"
        + "\n".join(
            f"{target.relative_to(repo_root)}:{hit.lineno}: `{hit.line}`"
            for hit in sorted(hits, key=lambda hit: (hit.lineno, hit.method))
        )
    )


def test_batch_run_context_tests_do_not_seed_or_inspect_raw_context_dicts() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    targets = [
        repo_root / "tests" / "test_batch_run_context_owner.py",
        repo_root / "tests" / "test_simulation_controller.py",
        repo_root / "tests" / "test_gui_sliders.py",
        repo_root / "tests" / "test_slider_parallelism_regression.py",
        repo_root / "tests" / "test_slider_parallel_feels_serial_regression.py",
        repo_root / "tests" / "test_explicit_batch_ui_coalescing_regression.py",
        repo_root / "tests" / "test_simulation_cache_behavior.py",
    ]
    hits: list[tuple[Path, _CallHit]] = []
    for target in targets:
        source = target.read_text(encoding="utf-8")
        lines = source.splitlines()
        tree = ast.parse(source, filename=str(target))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            chain = _attribute_chain(node.func)
            if chain and chain[-1] == "_current_context":
                hits.append(
                    (
                        target,
                        _CallHit(
                            method="_current_context",
                            lineno=node.lineno,
                            line=lines[node.lineno - 1].strip(),
                        ),
                    )
                )
                continue
            if chain and chain[-1] == "load_context" and node.args and isinstance(node.args[0], ast.Dict):
                hits.append(
                    (
                        target,
                        _CallHit(
                            method="load_context(raw-dict)",
                            lineno=node.lineno,
                            line=lines[node.lineno - 1].strip(),
                        ),
                    )
                )

    assert hits == [], (
        "Batch context tests must use typed owner-facing seeds and queries, not raw context dictionaries.\n"
        + "\n".join(
            f"{target.relative_to(repo_root)}:{hit.lineno}: `{hit.line}`"
            for target, hit in sorted(hits, key=lambda item: (str(item[0]), item[1].lineno, item[1].method))
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
    owner_target = repo_root / "kindred" / "gui" / "controllers" / "simulation_slider_preview_launch.py"
    assert target.is_file(), f"Expected file at {target}"

    if method_name == "_run_simulation_from_slider":
        source = owner_target.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(owner_target))
        fn = _class_method_node(tree, "SimulationSliderPreviewLaunchOwner", "run_from_slider")
        inspected_target = owner_target
    else:
        source = target.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(target))
        fn = _simulation_controller_method_node(tree, method_name)
        inspected_target = target
    lines = source.splitlines()

    flattened_hits: list[_CallHit] = []
    explicit_methods: set[str] = set()
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        chain = _attribute_chain(node.func)
        if not chain:
            continue
        if len(chain) == 3 and chain[:2] in {("self", "ui"), ("self", "_ui")} and chain[2] in RUN_ENTRY_BATCH_TARGET_METHODS:
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
            and chain[:3] in {("self", "ui", "batch"), ("self", "_ui", "batch")}
            and chain[3] in RUN_ENTRY_BATCH_TARGET_METHODS
        ):
            explicit_methods.add(chain[3])

    assert explicit_methods == RUN_ENTRY_BATCH_TARGET_METHODS, (
        f"Guardrail expectation changed: `SimulationController.{method_name}` must route the run-entry batch cluster "
        f"through `self.ui.batch`, but only found {sorted(explicit_methods)}."
    )

    assert flattened_hits == [], (
        f"Guardrail violated: `SimulationController.{method_name}` must not use flattened `self.ui.<method>` access "
        "for run-entry batch methods.\n"
        + "\n".join(
            f"{inspected_target.relative_to(repo_root)}:{hit.lineno}: `{hit.line}`"
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
    owner_target = repo_root / "kindred" / "gui" / "controllers" / "simulation_slider_preview_launch.py"
    assert target.is_file(), f"Expected file at {target}"

    if method_name == "_run_simulation_from_slider":
        source = owner_target.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(owner_target))
        fn = _class_method_node(tree, "SimulationSliderPreviewLaunchOwner", "run_from_slider")
        inspected_target = owner_target
    else:
        source = target.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(target))
        fn = _simulation_controller_method_node(tree, method_name)
        inspected_target = target
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
            f"{inspected_target.relative_to(repo_root)}:{hit.lineno}: `{hit.line}`"
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
    prep_target = repo_root / "kindred" / "gui" / "controllers" / "simulation_run_preparation.py"
    assert target.is_file(), f"Expected file at {target}"
    assert prep_target.is_file(), f"Expected file at {prep_target}"

    source = target.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(target))
    lines = source.splitlines()
    prep_source = prep_target.read_text(encoding="utf-8")
    prep_tree = ast.parse(prep_source, filename=str(prep_target))
    prep_lines = prep_source.splitlines()

    flattened_hits: list[_CallHit] = []
    explicit_methods: set[str] = set()
    for method_name in (
        "_run_simulation_internal",
        "_run_rows_or_abort",
        "_run_solver_context_or_abort",
        "_run_mechanism_context_or_abort",
        "_sync_batch_species_columns_for_run",
    ):
        fn = _simulation_controller_method_node(tree, method_name)
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
    for class_name, method_name in (
        ("SimulationRunMechanismPreparationOwner", "build_mechanism_context_or_abort"),
        ("SimulationRunSolverPreparationOwner", "build_solver_context_or_abort"),
    ):
        fn = _class_method_node(prep_tree, class_name, method_name)
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
                        line=prep_lines[node.lineno - 1].strip(),
                    )
                )
                continue
            if (
                len(chain) == 4
                and chain[:3] in {
                    ("self", "ui", "batch"),
                    ("self", "_ports", "batch"),
                }
                and chain[3] in RUN_SIMULATION_INTERNAL_FINAL_BATCH_TARGET_METHODS
            ):
                explicit_methods.add(chain[3])

    assert explicit_methods == RUN_SIMULATION_INTERNAL_FINAL_BATCH_TARGET_METHODS, (
        "Guardrail expectation changed: run preparation must route the final "
        f"batch residue through `self.ui.batch`, but only found {sorted(explicit_methods)}."
    )

    assert flattened_hits == [], (
        "Guardrail violated: run preparation must not use flattened "
        "`self.ui.<method>` access for the final batch residue.\n"
        + "\n".join(
            f"{target.relative_to(repo_root)}:{hit.lineno}: `{hit.line}`"
            for hit in sorted(flattened_hits, key=lambda hit: (hit.lineno, hit.method))
        )
    )


def test_start_next_batch_simulation_dispatches_only_contained_plan_workers() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    target = repo_root / "kindred" / "gui" / "controllers" / "simulation_controller.py"
    owner_target = repo_root / "kindred" / "gui" / "controllers" / "serial_worker_launch.py"
    assert target.is_file(), f"Expected file at {target}"
    assert owner_target.is_file(), f"Expected file at {owner_target}"

    source = target.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(target))
    owner_source = owner_target.read_text(encoding="utf-8")
    owner_tree = ast.parse(owner_source, filename=str(owner_target))
    fn = _simulation_controller_method_node(tree, "_start_next_batch_simulation")
    helper_fn = _simulation_controller_method_node(tree, "_start_contained_serial_batch_worker")
    lines = source.splitlines()
    owner_lines = owner_source.splitlines()

    legacy_hits: list[_CallHit] = []
    contained_hits: list[_CallHit] = []
    helper_calls: list[_CallHit] = []
    owner_calls: list[_CallHit] = []
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        chain = _attribute_chain(node.func)
        if chain == ("self", "_start_contained_serial_batch_worker"):
            helper_calls.append(
                _CallHit(
                    method="_start_contained_serial_batch_worker",
                    lineno=node.lineno,
                    line=lines[node.lineno - 1].strip(),
                )
            )
        elif chain == ("ContainedSimulationWorker",):
            legacy_hits.append(
                _CallHit(
                    method="ContainedSimulationWorker",
                    lineno=node.lineno,
                    line=lines[node.lineno - 1].strip(),
                )
            )
        elif chain == ("SimulationWorker",):
            legacy_hits.append(
                _CallHit(
                    method="SimulationWorker",
                    lineno=node.lineno,
                    line=lines[node.lineno - 1].strip(),
                )
            )

    for node in ast.walk(helper_fn):
        if isinstance(node, ast.ImportFrom) and node.module == "kindred.gui.simulation_worker":
            imported_names = {alias.name for alias in node.names}
            if "SimulationWorker" in imported_names:
                legacy_hits.append(
                    _CallHit(
                        method="SimulationWorker",
                        lineno=node.lineno,
                        line=lines[node.lineno - 1].strip(),
                    )
                )
            if "ContainedSimulationWorker" in imported_names:
                contained_hits.append(
                    _CallHit(
                        method="ContainedSimulationWorker",
                        lineno=node.lineno,
                        line=lines[node.lineno - 1].strip(),
                    )
                )
            continue
        if isinstance(node, ast.Call):
            chain = _attribute_chain(node.func)
            if chain == ("SimulationWorker",):
                legacy_hits.append(
                    _CallHit(
                        method="SimulationWorker",
                        lineno=node.lineno,
                        line=lines[node.lineno - 1].strip(),
                    )
                )
            elif chain == ("ContainedSimulationWorker",):
                legacy_hits.append(
                    _CallHit(
                        method="ContainedSimulationWorker",
                        lineno=node.lineno,
                        line=lines[node.lineno - 1].strip(),
                    )
                )
            elif chain == ("self", "_contained_serial_worker_launch_owner", "create_worker"):
                owner_calls.append(
                    _CallHit(
                        method="_contained_serial_worker_launch_owner.create_worker",
                        lineno=node.lineno,
                        line=lines[node.lineno - 1].strip(),
                    )
                )

    owner_worker_class = _class_method_node(owner_tree, "ContainedSerialWorkerLaunchOwner", "_worker_class")
    owner_create_worker = _class_method_node(owner_tree, "ContainedSerialWorkerLaunchOwner", "create_worker")
    for node in ast.walk(owner_worker_class):
        if isinstance(node, ast.ImportFrom) and node.module == "kindred.gui.simulation_worker":
            imported_names = {alias.name for alias in node.names}
            if "ContainedSimulationWorker" in imported_names:
                contained_hits.append(
                    _CallHit(
                        method="ContainedSimulationWorker",
                        lineno=node.lineno,
                        line=owner_lines[node.lineno - 1].strip(),
                    )
                )
            if "SimulationWorker" in imported_names:
                legacy_hits.append(
                    _CallHit(
                        method="SimulationWorker",
                        lineno=node.lineno,
                        line=owner_lines[node.lineno - 1].strip(),
                    )
                )
    for node in ast.walk(owner_create_worker):
        if not isinstance(node, ast.Call):
            continue
        chain = _attribute_chain(node.func)
        if chain == ("SimulationWorker",):
            legacy_hits.append(
                _CallHit(
                    method="SimulationWorker",
                    lineno=node.lineno,
                    line=owner_lines[node.lineno - 1].strip(),
                )
            )

    assert helper_calls, (
        "Guardrail expectation changed: `SimulationController._start_next_batch_simulation` must delegate contained "
        "worker launch to `_start_contained_serial_batch_worker` instead of constructing the worker inline."
    )
    assert owner_calls, (
        "Guardrail expectation changed: `_start_contained_serial_batch_worker` must delegate contained worker "
        "materialization to `ContainedSerialWorkerLaunchOwner` instead of constructing the worker inline."
    )
    assert contained_hits, (
        "Guardrail expectation changed: serial batch launch must dispatch queued serial batch runs through the "
        "contained typed plan worker."
    )
    assert legacy_hits == [], (
        "Guardrail violated: serial batch launch must not retain direct worker construction in the orchestration "
        "method or a `SimulationWorker` fallback after typed plan normalization.\n"
        + "\n".join(
            f"{target.relative_to(repo_root)}:{hit.lineno}: `{hit.line}`"
            for hit in sorted(legacy_hits, key=lambda hit: (hit.lineno, hit.method))
        )
    )


def test_parallel_batch_outcome_policy_is_not_controller_local() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    target = repo_root / "kindred" / "gui" / "controllers" / "simulation_controller.py"
    owner_target = repo_root / "kindred" / "gui" / "controllers" / "parallel_batch_outcome.py"
    source = target.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(target))
    owner_source = owner_target.read_text(encoding="utf-8")
    owner_tree = ast.parse(owner_source, filename=str(owner_target))
    lines = source.splitlines()

    _class_method_node(owner_tree, "ParallelBatchOutcomeOwner", "handle_scoped_failure")
    _class_method_node(owner_tree, "ParallelBatchOutcomeOwner", "consume_outcome")

    forbidden_calls = {
        "record_scoped_failure",
        "record_explicit_scoped_failure_cache_state",
        "completion_summary",
        "resolve_parallel_batch_outcome",
        "active_request_metadata",
        "discard_request",
    }
    hits: list[_CallHit] = []
    for method_name in {"_try_handle_scoped_batch_failure", "_consume_parallel_batch_outcome"}:
        fn = _simulation_controller_method_node(tree, method_name)
        for node in ast.walk(fn):
            if not isinstance(node, ast.Call):
                continue
            chain = _attribute_chain(node.func)
            if chain and chain[-1] in forbidden_calls:
                hits.append(
                    _CallHit(
                        method=f"{method_name}:{chain[-1]}",
                        lineno=node.lineno,
                        line=lines[node.lineno - 1].strip(),
                    )
                )

    assert hits == [], (
        "Parallel batch lane outcome/scoped-failure policy must be owned by `ParallelBatchOutcomeOwner`; "
        "controller methods may only delegate for compatibility.\n"
        + "\n".join(
            f"{target.relative_to(repo_root)}:{hit.lineno}: `{hit.line}`"
            for hit in sorted(hits, key=lambda hit: (hit.lineno, hit.method))
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
