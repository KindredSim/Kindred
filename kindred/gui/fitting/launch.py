"""Launch-time orchestration for fitting sessions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import logging
import os
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np
from PySide6 import QtWidgets

from kindred.core.analysis.dataset_parameter_overrides import coerce_fit_dataset_parameter_overrides
from kindred.core.analysis.fit_dataset_payload import (
    FitDatasetPayloadResult,
    coerce_fit_dataset_specs,
    read_fit_dataset_payload,
)
from kindred.core.fitting_evaluation import SerialFittingEvaluator, prepare_fitting_execution_context
from kindred.core.simulator.solvers import normalize_solver_name
from kindred.gui.controllers.dataset_manager import DatasetManagerError
from kindred.gui.fitting.batch_mapping import (
    T0_SEED_TOL_S,
    apply_batch_mapping_to_settings,
    create_and_seed_batch_set,
    default_batch_set_name_for_dataset,
    pick_existing_batch_set,
    prompt_dataset_batch_mapping_choice,
    resolve_saved_batch_mapping,
    select_batch_set,
    unique_batch_set_name,
)
from kindred.gui.fitting.constants import FITTING_DEFAULT_SOLVER
from kindred.gui.fitting.run_stamp import build_global_fit_run_stamp, hash_global_fit_run_stamp
from kindred.gui.fitting.runtime_readiness import FittingRuntimeIdentity, PreparedFitEvaluator
from kindred.gui.project_schema import PROJECT_DEFAULTS

if TYPE_CHECKING:
    from kindred.gui.controllers.dataset_manager import DatasetFitSettings
    from kindred.gui.fitting.window import FittingWindow


logger = logging.getLogger(__name__)

__all__ = [
    "FittingLaunchDatasetSelection",
    "FittingLaunchPurpose",
    "FittingLaunchRejection",
    "FittingLaunchResult",
    "FittingLaunchSnapshot",
    "FittingLaunchIdentityOwner",
    "GlobalFitLaunchContext",
    "launch_global_fit_session",
    "validate_de_bounds",
]


def validate_de_bounds(
    config: Dict[str, Any],
    *,
    dataset_variable_params: Optional[Mapping[str, Mapping[str, Mapping[str, Any]]]] = None,
) -> Tuple[bool, List[str]]:
    errors = []

    method = str(config.get("method", "")).lower()
    if method not in {"differential_evolution", "de"}:
        return True, []

    parameters = config.get("parameters", {})
    bounds = config.get("bounds", {})

    for param_name in parameters.keys():
        if param_name not in bounds:
            errors.append(f"Parameter '{param_name}' has no bounds defined")
            continue

        bound_tuple = bounds[param_name]
        if not isinstance(bound_tuple, tuple) or len(bound_tuple) != 2:
            errors.append(f"Parameter '{param_name}' has invalid bound format")
            continue

        min_val, max_val = bound_tuple

        if not np.isfinite(min_val):
            errors.append(f"Parameter '{param_name}' has non-finite minimum bound: {min_val}")

        if not np.isfinite(max_val):
            errors.append(f"Parameter '{param_name}' has non-finite maximum bound: {max_val}")

        if np.isfinite(min_val) and np.isfinite(max_val) and min_val >= max_val:
            errors.append(
                f"Parameter '{param_name}' has invalid bounds: "
                f"min ({min_val:.6g}) >= max ({max_val:.6g})"
            )

    for dataset_id, spec_map in (dataset_variable_params or {}).items():
        if not isinstance(spec_map, Mapping):
            continue
        for param_name, spec in spec_map.items():
            if not isinstance(spec, Mapping):
                errors.append(f"Dataset '{dataset_id}' parameter '{param_name}' has invalid bound format")
                continue
            try:
                min_val = float(spec.get("min"))
                max_val = float(spec.get("max"))
            except (TypeError, ValueError):
                errors.append(f"Dataset '{dataset_id}' parameter '{param_name}' has non-numeric bounds")
                continue
            if not np.isfinite(min_val):
                errors.append(f"Dataset '{dataset_id}' parameter '{param_name}' has non-finite minimum bound: {min_val}")
            if not np.isfinite(max_val):
                errors.append(f"Dataset '{dataset_id}' parameter '{param_name}' has non-finite maximum bound: {max_val}")
            if np.isfinite(min_val) and np.isfinite(max_val) and min_val >= max_val:
                errors.append(
                    f"Dataset '{dataset_id}' parameter '{param_name}' has invalid bounds: "
                    f"min ({min_val:.6g}) >= max ({max_val:.6g})"
                )

    is_valid = len(errors) == 0
    return is_valid, errors


class FittingLaunchIdentityOwner:
    """Owns fitting launch identity collection for passive readiness and explicit Run Fit."""

    def __init__(self, window: "FittingWindow") -> None:
        self._window = window
        self._last_launch_result: Optional[FittingLaunchResult] = None

    def collect_dataset_selection(self) -> "FittingLaunchDatasetSelection":
        window = self._window
        rows = []
        included_ids: List[str] = []
        for entry in window._dataset_entries:
            dataset_id = str(entry.get("id") or "").strip()
            include = bool(entry.get("include", True))
            label = str(entry.get("label") or dataset_id)
            species = ", ".join(entry.get("selected_species", []))
            weight = window._dataset_weight_for_id(dataset_id)
            entry["weight"] = weight
            entry["include"] = include
            rows.append(
                {
                    "id": dataset_id,
                    "label": label,
                    "species": species,
                    "include": include,
                    "weight": weight,
                }
            )
            if include:
                included_ids.append(dataset_id)
        return FittingLaunchDatasetSelection(rows=tuple(rows), ids=tuple(included_ids))

    def build_current_launch_snapshot(
        self,
        *,
        integration_settings: Optional[tuple[str, float, float]] = None,
        purpose: "FittingLaunchPurpose" = None,
        refresh_current_mechanism: bool = True,
    ) -> Optional["FittingLaunchSnapshot"]:
        window = self._window
        launch_purpose = purpose or FittingLaunchPurpose.PASSIVE_READINESS
        explicit = launch_purpose is FittingLaunchPurpose.EXPLICIT_RUN
        if getattr(window, "_fit_window_state_refreshing", False):
            return None
        if refresh_current_mechanism and not window._refresh_fit_window_state_for_current_mechanism(
            show_errors=explicit
        ):
            return None
        collected_config_bundle = window._params_ics_tab.collect_parameter_config_bundle(
            show_errors=explicit
        )
        if collected_config_bundle is None:
            return None
        config, global_dataset_params, global_dataset_variable_params = collected_config_bundle
        dataset_selection = self.collect_dataset_selection()
        if explicit:
            window.fit_run_state_owner.set_active_dataset_ids(list(dataset_selection.ids))
        if not dataset_selection.ids:
            if explicit:
                QtWidgets.QMessageBox.warning(window, "No Datasets", "Select at least one dataset to include.")
            return None
        invalid = window._invalid_applied_used_dataset_ids_for_run()
        if invalid:
            if explicit:
                labels = [window._dataset_label_for_id(ds_id) for ds_id in invalid]
                QtWidgets.QMessageBox.warning(
                    window,
                    "Global Fit",
                    "Run Fit is disabled due to invalid applied settings for: "
                    + ", ".join(labels)
                    + ".",
                )
            elif hasattr(window, "_status_label"):
                window._status_label.setText("Fitting runtime not ready: invalid applied settings")
            return None
        if integration_settings is not None:
            integration = integration_settings
        elif explicit:
            integration = window._params_ics_tab.collect_integration_settings()
        else:
            collect_integration_settings = getattr(
                window._params_ics_tab,
                "collect_integration_settings_silent",
                window._params_ics_tab.collect_integration_settings,
            )
            integration = collect_integration_settings()
        if integration is None:
            return None
        return FittingLaunchSnapshot(
            config=dict(config or {}),
            global_dataset_params=dict(global_dataset_params or {}),
            global_dataset_variable_params=dict(global_dataset_variable_params or {}),
            dataset_selection=dataset_selection,
            integration=integration,
        )

    def build_current_launch_result(
        self,
        *,
        purpose: "FittingLaunchPurpose",
        integration_settings: Optional[tuple[str, float, float]] = None,
        refresh_current_mechanism: bool = True,
    ) -> "FittingLaunchResult":
        identity, rejection = self._build_current_fit_runtime_identity(
            purpose=purpose,
            integration_settings=integration_settings,
            refresh_current_mechanism=refresh_current_mechanism,
        )
        result = FittingLaunchResult(identity=identity, rejection=rejection)
        self._last_launch_result = result
        return result

    def current_launch_result(self) -> Optional["FittingLaunchResult"]:
        return self._last_launch_result

    def payloads_available_for_identity(self, identity: Optional[FittingRuntimeIdentity]) -> bool:
        if identity is None:
            return False
        for spec in identity.datasets:
            dataset_id = str(getattr(spec, "dataset_id", "") or "")
            result = self._window._global_payload_results.get(dataset_id)
            if isinstance(result, FitDatasetPayloadResult) and result.state == "invalid":
                return False
            if dataset_id not in self._window._global_payload_lookup:
                return False
        return True

    def build_current_fit_runtime_identity(
        self,
        *,
        integration_settings: Optional[tuple[str, float, float]] = None,
        refresh_current_mechanism: bool = True,
    ) -> Optional[FittingRuntimeIdentity]:
        result = self.build_current_launch_result(
            purpose=FittingLaunchPurpose.PASSIVE_READINESS,
            integration_settings=integration_settings,
            refresh_current_mechanism=refresh_current_mechanism,
        )
        return result.identity

    def render_launch_rejection(
        self,
        result: "FittingLaunchResult",
        *,
        purpose: "FittingLaunchPurpose",
    ) -> None:
        rejection = result.rejection
        if rejection is None:
            return
        window = self._window
        if purpose is FittingLaunchPurpose.EXPLICIT_RUN and rejection.title:
            QtWidgets.QMessageBox.warning(window, rejection.title, rejection.message)
        elif hasattr(window, "_status_label") and rejection.passive_status:
            window._status_label.setText(rejection.passive_status)

    def _build_current_fit_runtime_identity(
        self,
        *,
        purpose: "FittingLaunchPurpose",
        integration_settings: Optional[tuple[str, float, float]] = None,
        refresh_current_mechanism: bool = True,
    ) -> tuple[Optional[FittingRuntimeIdentity], Optional["FittingLaunchRejection"]]:
        window = self._window
        launch_snapshot = self.build_current_launch_snapshot(
            integration_settings=integration_settings,
            purpose=purpose,
            refresh_current_mechanism=refresh_current_mechanism,
        )
        if launch_snapshot is None:
            if purpose is FittingLaunchPurpose.EXPLICIT_RUN:
                return None, None
            if window._invalid_applied_used_dataset_ids_for_run():
                return None, FittingLaunchRejection(
                    title="",
                    message="Fitting launch inputs have invalid applied settings.",
                    passive_status="Fitting runtime not ready: invalid applied settings",
                )
            return None, FittingLaunchRejection(
                title="",
                message="Fitting launch inputs are not valid.",
                passive_status="Fitting runtime not ready",
            )
        config = launch_snapshot.config
        global_dataset_params = launch_snapshot.global_dataset_params
        global_dataset_variable_params = launch_snapshot.global_dataset_variable_params
        dataset_selection = launch_snapshot.dataset_selection
        solver, rtol, atol = launch_snapshot.integration
        requested_solver, _solver_warning = normalize_solver_name(str(solver or FITTING_DEFAULT_SOLVER))
        mechanism_text = window._safe_text_from_getter(getattr(window, "_mechanism_text_getter", None))
        evaluator_components = window._fitting_evaluator_components_for_runtime_identity(
            mechanism_text=mechanism_text,
            config=config,
            requested_solver=str(requested_solver),
            requested_rtol=float(rtol),
            requested_atol=float(atol),
        )
        if evaluator_components is None:
            return None, FittingLaunchRejection(
                title="Global Fit" if purpose is FittingLaunchPurpose.EXPLICIT_RUN else "",
                message="Failed to build fitting evaluator.",
                passive_status="Fitting runtime not ready",
            )
        simulation_func = evaluator_components.base_evaluator
        simulation_factory = evaluator_components.evaluator_factory
        prepared_simulation = evaluator_components.prepared_simulation
        readiness_required = evaluator_components.readiness_required

        datasets, rejection = self._dataset_payloads_for_launch(dataset_selection.ids)
        if datasets is None:
            return None, rejection
        try:
            dataset_specs = coerce_fit_dataset_specs(datasets)
        except Exception as exc:
            return None, FittingLaunchRejection(
                title="Global Fit" if purpose is FittingLaunchPurpose.EXPLICIT_RUN else "",
                message=str(exc) or "Dataset payloads are invalid.",
                passive_status="Fitting runtime not ready: invalid dataset payloads",
            )

        weights = window._weights_for_run(dataset_selection.as_mapping())
        staged_params = window._params_ics_tab.get_staged_dataset_params() or {}
        shared_param_keys = window._shared_param_keys_for_run(config)
        dataset_params_for_run = window._dataset_params_for_run(
            list(dataset_selection.ids),
            shared_param_keys,
            staged_params,
            global_dataset_params=global_dataset_params,
            evaluator=simulation_func,
        )
        variable_params = window._variable_params_for_run(
            list(dataset_selection.ids),
            shared_param_keys,
            staged_params,
            global_dataset_variable_params=global_dataset_variable_params,
            evaluator=simulation_func,
        )
        ok, errors = validate_de_bounds(
            config,
            dataset_variable_params=variable_params,
        )
        if not ok:
            return None, FittingLaunchRejection(
                title="Invalid Bounds" if purpose is FittingLaunchPurpose.EXPLICIT_RUN else "",
                message="\n".join(errors),
                passive_status="Fitting runtime not ready: invalid bounds",
            )
        dataset_overrides = coerce_fit_dataset_parameter_overrides(
            dataset_ids=list(dataset_selection.ids),
            dataset_params=dataset_params_for_run,
            dataset_variable_params=variable_params,
        )
        reactions_text = window._safe_text_from_getter(getattr(window, "_reactions_text_getter", None))
        applied_targets = dict(window._species_table.fit_targets_selection_applied or {})
        applied_target_weights = {
            str(ds_id): window._species_table.applied_target_weights_for_dataset(str(ds_id))
            for ds_id in applied_targets.keys()
        }
        stamp = build_global_fit_run_stamp(
            dataset_rows=list(dataset_selection.rows),
            included_ids=list(dataset_selection.ids),
            applied_fit_targets=applied_targets,
            applied_target_weights=applied_target_weights,
            weights_used=(dict(weights) if isinstance(weights, dict) else None),
            weight_mode=("equal" if weights is None else "custom"),
            fit_config=dict(config or {}),
            mechanism_text=mechanism_text,
            reactions_text=reactions_text,
            prepared_simulation=prepared_simulation,
            dataset_specs=list(dataset_specs),
            dataset_overrides=list(dataset_overrides),
        )
        if prepared_simulation is None:
            stamp["runtime_request"] = {
                "solver": str(requested_solver),
                "rtol": f"{float(rtol):.12g}",
                "atol": f"{float(atol):.12g}",
                "param_names": window._param_names_for_readiness_identity(
                    config=config,
                    prepared_simulation=None,
                    mechanism_text=mechanism_text,
                ),
            }
            stamp["runtime_request"].update(window._runtime_settings_for_identity())
        stamp_hash = hash_global_fit_run_stamp(stamp)
        stamp_short = str(stamp_hash)[:12]
        fixed_params = window._fixed_params_for_run(config)
        fit_evaluator = window._simulation_with_fixed_params(simulation_func, fixed_params) if simulation_func is not None else None
        fit_evaluator_factory = None
        if callable(simulation_factory):
            fixed_snapshot = dict(fixed_params)

            def fit_evaluator_factory():
                base_evaluator = simulation_factory()
                return PreparedFitEvaluator(
                    base_evaluator=base_evaluator,
                    fit_evaluator=window._simulation_with_fixed_params(base_evaluator, fixed_snapshot),
                )

        lane_budget = window._fit_runtime_lane_budget(len(dataset_specs))
        return FittingRuntimeIdentity(
            datasets=tuple(dataset_specs),
            config=config,
            dataset_overrides=tuple(dataset_overrides),
            weights=dict(weights) if weights is not None else None,
            requested_solver=str(requested_solver),
            requested_rtol=float(rtol),
            requested_atol=float(atol),
            fit_evaluator=fit_evaluator,
            stamp=stamp,
            stamp_hash=str(stamp_hash),
            stamp_short=str(stamp_short),
            lane_count=int(lane_budget),
            readiness_required=bool(readiness_required),
            fit_evaluator_factory=fit_evaluator_factory,
            base_evaluator=simulation_func,
        ), None

    def _dataset_payloads_for_launch(
        self,
        selected_ids: Sequence[str],
    ) -> tuple[Optional[list[dict[str, Any]]], Optional["FittingLaunchRejection"]]:
        window = self._window
        datasets: list[dict[str, Any]] = []
        for dataset_id in selected_ids:
            ds_id = str(dataset_id)
            result = window._global_payload_results.get(ds_id)
            if isinstance(result, FitDatasetPayloadResult) and result.state == "invalid":
                reason = str(result.error or "Dataset payload is invalid.")
                return None, FittingLaunchRejection(
                    title="Global Fit",
                    message=f"Dataset '{ds_id}' has invalid payload:\n{reason}",
                    passive_status="Fitting runtime not ready: invalid dataset payloads",
                )
            if ds_id not in window._global_payload_lookup:
                return None, FittingLaunchRejection(
                    title="Global Fit",
                    message=f"Dataset '{ds_id}' is missing payloads.",
                    passive_status="Fitting runtime not ready: missing dataset payloads",
                )
            datasets.append(dict(window._global_payload_lookup[ds_id]))
        return datasets, None


class FittingLaunchPurpose(Enum):
    PASSIVE_READINESS = "passive_readiness"
    EXPLICIT_RUN = "explicit_run"


@dataclass(frozen=True)
class FittingLaunchRejection:
    title: str
    message: str
    passive_status: str = "Fitting runtime not ready"


@dataclass(frozen=True)
class FittingLaunchResult:
    identity: Optional[FittingRuntimeIdentity]
    rejection: Optional[FittingLaunchRejection] = None


@dataclass(frozen=True)
class FittingLaunchDatasetSelection:
    rows: tuple[dict[str, Any], ...]
    ids: tuple[str, ...]

    def as_mapping(self) -> dict[str, Any]:
        return {
            "rows": [dict(row) for row in self.rows],
            "ids": list(self.ids),
        }


@dataclass(frozen=True)
class FittingLaunchSnapshot:
    config: dict[str, Any]
    global_dataset_params: dict[str, Any]
    global_dataset_variable_params: dict[str, Any]
    dataset_selection: FittingLaunchDatasetSelection
    integration: tuple[str, float, float]


@dataclass(frozen=True)
class GlobalFitLaunchContext:
    parent: QtWidgets.QWidget
    dataset_manager: Any
    data_manager_getter: Callable[[], Any]
    mechanism_text_getter: Callable[[], str]
    reactions_text_getter: Callable[[], str]
    reactions_text_setter: Callable[[str], None]
    extract_mechanism_initials: Callable[[str], Dict[str, float]]
    record_best_effort_failure: Callable[..., None]
    set_status: Callable[[str], None]
    sync_batch_species_columns: Callable[[List[str]], None]
    batch_initials_for_row: Callable[[int], Dict[str, float]]
    get_solver_settings: Callable[[], Dict[str, object]]
    temperature_getter: Callable[[], float]
    num_points_getter: Callable[[], int]
    register_fit_window: Callable[[QtWidgets.QWidget], None]
    write_fit_results_to_mechanism: Callable[[Dict[str, float]], None]
    apply_fit_results_to_project: Callable[[str, Dict[str, float], Dict[str, Dict[str, float]]], None]
    apply_dataset_initial_updates: Callable[[str, Dict[str, float]], None]
    load_fitting_defaults: Callable[[], Dict[str, object]]
    batch_store: Any = None
    batch_model: Any = None
    batch_table: Any = None
    window_factory: Optional[Callable[..., QtWidgets.QWidget]] = None


def _record_failure(
    context: GlobalFitLaunchContext,
    key: str,
    *,
    message: str,
    exc: Optional[Exception] = None,
) -> None:
    try:
        context.record_best_effort_failure(
            key,
            message=message,
            exc=exc,
        )
    except Exception:
        logger.debug("Failed to record fitting launch failure %s", key, exc_info=True)


def _coerce_dataset_payload(
    *,
    dataset_id: str,
    t_values: np.ndarray,
    species_map: Dict[str, np.ndarray],
) -> FitDatasetPayloadResult:
    if not species_map:
        return FitDatasetPayloadResult.absent()
    return read_fit_dataset_payload(
        dataset_id=str(dataset_id),
        t=np.asarray(t_values, dtype=float).reshape(-1),
        species_data={str(name): np.asarray(values, dtype=float).reshape(-1) for name, values in species_map.items()},
        selected_species=list(species_map.keys()),
    )


def _resolve_window_factory(context: GlobalFitLaunchContext) -> Callable[..., QtWidgets.QWidget]:
    if callable(context.window_factory):
        return context.window_factory
    from .window import FittingWindow

    return FittingWindow


def launch_global_fit_session(context: GlobalFitLaunchContext) -> Optional[QtWidgets.QWidget]:
    """Launch the global-fit window using the fitting package as the owner."""
    data_panel = context.data_manager_getter()
    if data_panel is None:
        QtWidgets.QMessageBox.warning(context.parent, "Global Fit", "Data manager unavailable in the current layout.")
        return None

    datasets_map_raw = data_panel.get_datasets()
    datasets_map = {str(k): dict(v or {}) for k, v in (datasets_map_raw or {}).items()}
    if not datasets_map:
        QtWidgets.QMessageBox.warning(context.parent, "Global Fit", "Load at least one dataset first.")
        return None

    has_any_species = False
    for payload in (datasets_map or {}).values():
        species_map = (payload or {}).get("species") or {}
        if isinstance(species_map, dict) and species_map:
            has_any_species = True
            break
    if not has_any_species:
        QtWidgets.QMessageBox.warning(context.parent, "Global Fit", "No species available for loaded datasets.")
        return None

    mechanism_text = str(context.mechanism_text_getter() or "")
    try:
        mechanism_initials = context.extract_mechanism_initials(mechanism_text)
    except Exception as exc:
        _record_failure(
            context,
            "global_fit.extract_mechanism_initials",
            message="Failed to extract mechanism initials while launching global fit; continuing with empty initials",
            exc=exc,
        )
        mechanism_initials = {}
    mechanism_species = list((mechanism_initials or {}).keys())
    selected_names = sorted(map(str, datasets_map.keys()))
    dataset_count = len(selected_names)
    dataset_label = f"{dataset_count} dataset{'s' if dataset_count != 1 else ''}"

    try:
        parameter_defs = context.dataset_manager.scan_mechanism_parameters(mechanism_text)
    except DatasetManagerError as exc:
        QtWidgets.QMessageBox.warning(context.parent, "Global Fit", str(exc))
        return None

    try:
        context.sync_batch_species_columns(mechanism_species)
    except Exception as exc:
        _record_failure(
            context,
            "global_fit.sync_batch_species_columns",
            message="Failed to sync batch species columns while launching global fit; continuing without mapping refresh",
            exc=exc,
        )

    batch_store = context.batch_store
    batch_model = context.batch_model
    batch_table = context.batch_table

    def _defaults_for_batch_set(set_name: str) -> Optional[Dict[str, float]]:
        if batch_store is None:
            return None
        row = batch_store.row_for_set(set_name)
        if row is None:
            return None
        try:
            return context.batch_initials_for_row(int(row))
        except Exception as exc:
            try:
                if batch_model is not None:
                    batch_model.validate_rows([int(row)])
            except Exception as inner_exc:
                _record_failure(
                    context,
                    "global_fit.batch_model.validate_rows",
                    message="Failed to revalidate batch rows after invalid initial conditions during global fit prep",
                    exc=inner_exc,
                )
            QtWidgets.QMessageBox.warning(
                context.parent,
                "Global Fit",
                f"Set '{set_name}' has invalid initial conditions:\n\n{exc}\n\n"
                "Fix the Initial Conditions table and retry Global Fit.",
            )
            return None

    defaults_by_dataset: Dict[str, Dict[str, float]] = {}
    batch_set_names = list(batch_store.set_names()) if batch_store is not None else []
    running_under_pytest = bool(os.environ.get("PYTEST_CURRENT_TEST"))

    for dataset_name in selected_names:
        settings = context.dataset_manager.get_fit_settings(dataset_name)
        base = default_batch_set_name_for_dataset(dataset_name) or str(dataset_name)

        resolved_mapping = resolve_saved_batch_mapping(settings, batch_store)
        target_set: Optional[str] = resolved_mapping.batch_set if resolved_mapping.status == "mapped" else None
        if target_set is None:
            create_set_name = unique_batch_set_name(batch_set_names, base)
            action = prompt_dataset_batch_mapping_choice(
                context.parent,
                dataset_name,
                create_set_name,
                title="Global Fit – Set Mapping",
                skip_label="Cancel",
                skip_description="Cancel global fit",
                running_under_pytest=running_under_pytest,
                pytest_default_action="create",
            )
            if action == "skip":
                context.set_status("Global fit cancelled")
                return None
            if action == "map":
                target_set = pick_existing_batch_set(
                    context.parent,
                    dataset_name,
                    batch_set_names,
                    title="Map Dataset to Set",
                    empty_message_title="Global Fit",
                    empty_message_text="No sets exist to map to. Create a set first.",
                )
                if not target_set:
                    context.set_status("Global fit cancelled")
                    return None
                apply_batch_mapping_to_settings(settings, batch_store, target_set)
            else:
                dataset_payload = datasets_map.get(dataset_name) or {}
                row_idx, created, seeded = create_and_seed_batch_set(
                    dataset_name=dataset_name,
                    dataset_payload=dataset_payload,
                    mechanism_species=mechanism_species,
                    batch_store=batch_store,
                    batch_model=batch_model,
                    set_name=create_set_name,
                    record_failure=lambda key, **kwargs: _record_failure(context, key, **kwargs),
                    failure_key_prefix="global_fit",
                    tol=T0_SEED_TOL_S,
                )
                target_set = create_set_name
                if created and not seeded and not mechanism_species:
                    context.set_status("Global fit cancelled")
                    return None
                apply_batch_mapping_to_settings(settings, batch_store, target_set)
                if created and batch_store is not None and not seeded:
                    try:
                        t_arr = np.asarray((dataset_payload or {}).get("t", []), dtype=float).reshape(-1)
                        t0 = float(t_arr[0]) if t_arr.size else float("nan")
                    except Exception:
                        t0 = float("nan")
                    if not (abs(t0) <= T0_SEED_TOL_S):
                        if running_under_pytest:
                            resp2 = QtWidgets.QMessageBox.StandardButton.Ok
                        else:
                            resp2 = QtWidgets.QMessageBox.warning(
                                context.parent,
                                "Global Fit – Initial Conditions",
                                (
                                    f"Dataset '{dataset_name}' does not start at t\u22480 "
                                    f"(t0={t0:.6g} s; tol={T0_SEED_TOL_S:.1e} s).\n\n"
                                    "OK: Create set with zeros and continue\n"
                                    "Cancel: Create set and edit manually (then restart Global Fit)"
                                ),
                                QtWidgets.QMessageBox.StandardButton.Ok | QtWidgets.QMessageBox.StandardButton.Cancel,
                                QtWidgets.QMessageBox.StandardButton.Ok,
                            )
                        if resp2 == QtWidgets.QMessageBox.StandardButton.Cancel:
                            select_batch_set(
                                batch_store,
                                batch_model,
                                batch_table,
                                target_set,
                                record_failure=lambda key, **kwargs: _record_failure(context, key, **kwargs),
                                failure_key_prefix="global_fit",
                            )
                            QtWidgets.QMessageBox.information(
                                context.parent,
                                "Global Fit",
                                (
                                    f"Set '{target_set}' was created.\n\n"
                                    "Edit its initial concentrations in the Initial Conditions table, "
                                    "then start Global Fit again."
                                ),
                            )
                            context.set_status("Global fit cancelled")
                            return None

        defaults = _defaults_for_batch_set(target_set)
        if defaults is None:
            context.set_status("Global fit cancelled")
            return None
        defaults_by_dataset[dataset_name] = defaults
        batch_set_names = list(batch_store.set_names()) if batch_store is not None else batch_set_names

    settings_map: Dict[str, "DatasetFitSettings"] = {}
    for dataset_name in selected_names:
        settings = context.dataset_manager.get_fit_settings(dataset_name)
        settings.ensure_species(mechanism_species, defaults_by_dataset.get(dataset_name, mechanism_initials))
        missing_initials = [
            species_name for species_name in (mechanism_species or [])
            if species_name not in (settings.initial_conditions or {})
        ]
        if missing_initials:
            QtWidgets.QMessageBox.warning(
                context.parent,
                "Global Fit",
                f"Dataset '{dataset_name}' requires initial concentrations for: {', '.join(missing_initials)}.",
            )
            context.set_status("Global fit cancelled")
            return None
        context.dataset_manager.update_fit_settings(dataset_name, settings)
        settings_map[dataset_name] = settings

    dataset_params: Dict[str, Dict[str, float]] = {}
    dataset_variable_params: Dict[str, Dict[str, Dict[str, float]]] = {}
    weights: Dict[str, float] = {}
    initial_prefix = "init:"
    for dataset_name in selected_names:
        settings = settings_map[dataset_name]
        weights[dataset_name] = settings.weight
        dataset_params[dataset_name] = {}
        var_specs: Dict[str, Dict[str, float]] = {}
        for species, init_value in settings.initial_conditions.items():
            key = f"{initial_prefix}{species}"
            if settings.fit_flags.get(species):
                bound_pair = settings.bounds.get(species, (0.0, max(10.0, init_value * 10 or 10.0)))
                var_specs[key] = {
                    "initial": init_value,
                    "min": bound_pair[0],
                    "max": bound_pair[1],
                    "log10": bool(settings.log10_flags.get(species, False)),
                }
            else:
                dataset_params[dataset_name][key] = init_value
        if var_specs:
            dataset_variable_params[dataset_name] = var_specs

    t_axes: List[np.ndarray] = []
    dataset_entries: List[Dict[str, object]] = []
    dataset_payloads: List[Dict[str, object]] = []
    dataset_payload_results: Dict[str, object] = {}
    for dataset_id, payload in datasets_map.items():
        t_values = np.asarray(payload.get("t", []), dtype=float).reshape(-1)
        species_map = payload.get("species") or {}
        series_map: Dict[str, np.ndarray] = {}
        if isinstance(species_map, dict):
            for name, values in species_map.items():
                try:
                    arr = np.asarray(values, dtype=float).reshape(-1)
                except Exception as exc:
                    _record_failure(
                        context,
                        "global_fit.dataset_entries.series",
                        message=f"Skipping invalid dataset series '{name}' while preparing global fit",
                        exc=exc,
                    )
                    continue
                if t_values.size and arr.size == t_values.size:
                    series_map[str(name)] = arr.copy()
        if t_values.size:
            t_axes.append(t_values)
        dataset_entries.append(
            {
                "id": str(dataset_id),
                "label": str(dataset_id),
                "t": t_values,
                "species_data": series_map,
                "selected_species": [],
                "weight": weights.get(str(dataset_id), 1.0),
                "include": True,
            }
        )
        fit_payload = _coerce_dataset_payload(
            dataset_id=str(dataset_id),
            t_values=t_values,
            species_map=series_map,
        )
        dataset_payload_results[str(dataset_id)] = fit_payload
        payload_dict = getattr(fit_payload, "payload", None)
        if isinstance(payload_dict, dict):
            dataset_payloads.append(dict(payload_dict))

    if t_axes:
        max_time = max(float(np.max(t_values)) for t_values in t_axes if t_values.size)
        if not np.isfinite(max_time):
            max_time = 1.0
        max_len = max(int(t_values.size) for t_values in t_axes if t_values.size)
    else:
        max_time = 1.0
        max_len = 2

    grid_points = max(2, int(context.num_points_getter()), int(max_len))
    solver_settings = dict(context.get_solver_settings() or {})
    def _build_simulation(
        mechanism_text_for_run: str,
        param_names_for_run: List[str],
        *,
        solver: Optional[str] = None,
        rtol: Optional[float] = None,
        atol: Optional[float] = None,
        temperature_K: Optional[float] = None,
        use_sparse_jacobian: Optional[bool] = None,
        wegscheider_cyclicity_enabled: Optional[bool] = None,
    ):
        current_solver_settings = solver_settings
        needs_live_solver_settings = (
            solver is None
            or rtol is None
            or atol is None
            or use_sparse_jacobian is None
            or wegscheider_cyclicity_enabled is None
        )
        if needs_live_solver_settings:
            try:
                current_solver_settings = dict(context.get_solver_settings() or {})
            except Exception:
                current_solver_settings = solver_settings

        current_solver_settings = dict(current_solver_settings or {})
        from kindred.core.simulator.solvers import normalize_solver_name

        current_solver_settings.setdefault("solver", FITTING_DEFAULT_SOLVER)
        current_solver_settings.setdefault("rtol", 1e-6)
        current_solver_settings.setdefault("atol", 1e-12)
        current_solver_settings.setdefault("use_sparse_jacobian", bool(PROJECT_DEFAULTS["use_sparse_jacobian"]))
        current_solver_settings.setdefault(
            "wegscheider_cyclicity_enabled",
            bool(PROJECT_DEFAULTS["wegscheider_cyclicity_enabled"]),
        )

        solver_label = str(solver or current_solver_settings.get("solver") or FITTING_DEFAULT_SOLVER).strip() or FITTING_DEFAULT_SOLVER
        solver_value, _solver_warning = normalize_solver_name(solver_label)
        rtol_value = float(rtol if rtol is not None else (current_solver_settings.get("rtol") or 1e-6))
        atol_value = float(atol if atol is not None else (current_solver_settings.get("atol") or 1e-12))
        temperature_value = float(temperature_K) if temperature_K is not None else float(context.temperature_getter())
        sparse_value = (
            bool(use_sparse_jacobian)
            if use_sparse_jacobian is not None
            else bool(current_solver_settings.get("use_sparse_jacobian"))
        )
        wegscheider_value = (
            bool(wegscheider_cyclicity_enabled)
            if wegscheider_cyclicity_enabled is not None
            else bool(current_solver_settings.get("wegscheider_cyclicity_enabled"))
        )
        fit_context = prepare_fitting_execution_context(
            mechanism_text=str(mechanism_text_for_run or ""),
            param_names=[str(x) for x in (param_names_for_run or []) if str(x)],
            t_end=max_time,
            num_points=grid_points,
            temperature_K=temperature_value,
            solver=solver_value,
            rtol=rtol_value,
            atol=atol_value,
            use_sparse_jacobian=sparse_value,
            wegscheider_cyclicity_enabled=wegscheider_value,
            initial_prefix=initial_prefix,
        )
        return SerialFittingEvaluator(fit_context)

    def _runtime_settings_for_fit_window() -> Dict[str, object]:
        current_solver_settings = dict(context.get_solver_settings() or {})
        return {
            "temperature_K": float(context.temperature_getter()),
            "use_sparse_jacobian": bool(
                current_solver_settings.get(
                    "use_sparse_jacobian",
                    PROJECT_DEFAULTS["use_sparse_jacobian"],
                )
            ),
            "wegscheider_cyclicity_enabled": bool(
                current_solver_settings.get(
                    "wegscheider_cyclicity_enabled",
                    PROJECT_DEFAULTS["wegscheider_cyclicity_enabled"],
                )
            ),
        }

    window_factory = _resolve_window_factory(context)
    window = window_factory(
        mode="global",
        parameter_defs=parameter_defs,
        dataset_entries=dataset_entries,
        dataset_manager=context.dataset_manager,
        simulation_func=None,
        mechanism_species=mechanism_species,
        mechanism_text_getter=context.mechanism_text_getter,
        reactions_text_getter=context.reactions_text_getter,
        reactions_text_setter=context.reactions_text_setter,
        simulation_builder=_build_simulation,
        runtime_settings_getter=_runtime_settings_for_fit_window,
        dataset_params=dataset_params,
        dataset_variable_params=dataset_variable_params,
        dataset_payloads=dataset_payloads,
        dataset_payload_results=dataset_payload_results,
        dataset_weights=weights,
        apply_callback=context.write_fit_results_to_mechanism,
        project_apply_callback=context.apply_fit_results_to_project,
        config_defaults=context.load_fitting_defaults(),
        dataset_settings_updater=context.apply_dataset_initial_updates,
        parent=context.parent,
    )
    window.setWindowTitle(f"Global Fit – {dataset_label}")
    context.set_status(f"Global fitting window open ({dataset_label})")
    context.register_fit_window(window)
    return window
