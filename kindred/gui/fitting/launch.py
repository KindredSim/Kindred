"""Launch-time orchestration for fitting sessions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import logging
import math
import os
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, TYPE_CHECKING

import numpy as np
from PySide6 import QtWidgets

from kindred.core.analysis.fit_dataset_payload import (
    FitDatasetPayloadResult,
    read_fit_dataset_payload,
)
from kindred.core.datasets.observation_payload import (
    copy_observations_map,
    dense_view_from_observations,
    observations_from_payload,
)
from kindred.core.fitting_evaluation import SerialFittingEvaluator, prepare_fitting_execution_context
from kindred.gui.controllers.dataset_errors import DatasetOwnerError
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
from kindred.gui.fitting.runtime_inputs import FittingEvaluatorRuntimeSettings, FittingRuntimeInputs
from kindred.gui.fitting.runtime_readiness import FittingRuntimeIdentity

if TYPE_CHECKING:
    from kindred.gui.controllers.dataset_fit_settings_store import DatasetFitSettings


logger = logging.getLogger(__name__)

__all__ = [
    "FittingLaunchDatasetSelection",
    "FittingLaunchPurpose",
    "FittingLaunchRejection",
    "FittingLaunchResult",
    "FittingLaunchSnapshot",
    "GlobalFitLaunchSettings",
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
class GlobalFitLaunchSettings:
    solver: str
    rtol: float
    atol: float
    runtime_inputs: FittingRuntimeInputs


@dataclass(frozen=True)
class GlobalFitLaunchContext:
    parent: QtWidgets.QWidget
    dataset_registry: Any
    dataset_fit_settings_store: Any
    dataset_view_publisher: Any
    mechanism_parameter_scan_owner: Any
    mechanism_text_getter: Callable[[], str]
    reactions_text_getter: Callable[[], str]
    reactions_text_setter: Callable[[str], None]
    extract_mechanism_initials: Callable[[str], Dict[str, float]]
    record_best_effort_failure: Callable[..., None]
    set_status: Callable[[str], None]
    sync_batch_species_columns: Callable[[List[str]], None]
    batch_initials_for_row: Callable[[int], Dict[str, float]]
    fitting_settings_getter: Callable[[], GlobalFitLaunchSettings]
    num_points_getter: Callable[[], int]
    register_fit_window: Callable[..., object]
    apply_fit_results_to_project: Callable[[str, Dict[str, float], Dict[str, Dict[str, float]]], None]
    load_fitting_defaults: Callable[[], Dict[str, object]]
    batch_store: Any = None
    batch_model: Any = None
    batch_table: Any = None
    window_factory: Optional[Callable[..., QtWidgets.QWidget]] = None


def _coerce_global_fit_launch_settings(raw_settings: object) -> GlobalFitLaunchSettings:
    if not isinstance(raw_settings, GlobalFitLaunchSettings):
        raise RuntimeError("Global Fit launch settings provider must return GlobalFitLaunchSettings.")
    return GlobalFitLaunchSettings(
        solver=_require_global_fit_launch_solver(raw_settings.solver),
        rtol=_require_global_fit_positive_finite_float(raw_settings.rtol, "rtol"),
        atol=_require_global_fit_positive_finite_float(raw_settings.atol, "atol"),
        runtime_inputs=_require_global_fit_runtime_inputs(raw_settings.runtime_inputs),
    )


def _require_global_fit_runtime_inputs(value: object) -> FittingRuntimeInputs:
    if not isinstance(value, FittingRuntimeInputs):
        raise RuntimeError("Global Fit launch settings require typed runtime inputs.")
    return value


def _require_global_fit_launch_solver(value: object) -> str:
    if not isinstance(value, str):
        raise RuntimeError("Global Fit launch settings require explicit solver.")
    solver_label = value.strip()
    if not solver_label:
        raise RuntimeError("Global Fit launch settings require explicit solver.")
    return solver_label


def _require_global_fit_positive_finite_float(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise RuntimeError(f"Global Fit launch settings require numeric {label}.")
    try:
        numeric = float(value)
    except Exception as exc:
        raise RuntimeError(f"Global Fit launch settings require numeric {label}.") from exc
    if not math.isfinite(numeric) or numeric <= 0.0:
        raise RuntimeError(f"Global Fit launch settings require positive finite {label}.")
    return numeric


def _read_global_fit_launch_settings(context: GlobalFitLaunchContext) -> GlobalFitLaunchSettings:
    getter = context.fitting_settings_getter
    try:
        raw_settings = getter()
    except Exception as exc:
        raise RuntimeError("Failed to read Global Fit launch settings.") from exc
    return _coerce_global_fit_launch_settings(raw_settings)


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
    observations: Mapping[str, Mapping[str, object]],
) -> FitDatasetPayloadResult:
    observations_map = copy_observations_map(observations)
    if not observations_map:
        return FitDatasetPayloadResult.absent()
    return read_fit_dataset_payload(
        dataset_id=str(dataset_id),
        observations=observations_map,
        selected_species=list(observations_map.keys()),
    )


def _resolve_window_factory(context: GlobalFitLaunchContext) -> Callable[..., QtWidgets.QWidget]:
    if callable(context.window_factory):
        return context.window_factory
    from .window import FittingWindow

    return FittingWindow


def launch_global_fit_session(context: GlobalFitLaunchContext) -> Optional[QtWidgets.QWidget]:
    """Launch the global-fit window using the fitting package as the owner."""
    if context.dataset_registry is None:
        QtWidgets.QMessageBox.warning(context.parent, "Global Fit", "Dataset source unavailable in the current layout.")
        return None

    records = tuple(context.dataset_registry.records())
    if not records:
        QtWidgets.QMessageBox.warning(context.parent, "Global Fit", "Load at least one dataset first.")
        return None
    selected_records = tuple(sorted(records, key=lambda record: str(record.display_name)))
    datasets_map = {
        str(record.dataset_id): dict(record.payload or {})
        for record in selected_records
    }
    display_name_by_id = {
        str(record.dataset_id): str(record.display_name)
        for record in selected_records
    }

    has_any_species = False
    for payload in (datasets_map or {}).values():
        if observations_from_payload(payload):
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
    selected_dataset_ids = [str(record.dataset_id) for record in selected_records]
    dataset_count = len(selected_dataset_ids)
    dataset_label = f"{dataset_count} dataset{'s' if dataset_count != 1 else ''}"

    try:
        parameter_defs = context.mechanism_parameter_scan_owner.scan_mechanism_parameters(mechanism_text)
    except DatasetOwnerError as exc:
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

    for dataset_id in selected_dataset_ids:
        display_name = display_name_by_id.get(str(dataset_id), str(dataset_id))
        settings = context.dataset_fit_settings_store.get_fit_settings(dataset_id)
        base = default_batch_set_name_for_dataset(display_name) or str(display_name)

        resolved_mapping = resolve_saved_batch_mapping(settings, batch_store)
        target_set: Optional[str] = resolved_mapping.batch_set if resolved_mapping.status == "mapped" else None
        if target_set is None:
            create_set_name = unique_batch_set_name(batch_set_names, base)
            action = prompt_dataset_batch_mapping_choice(
                context.parent,
                display_name,
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
                    display_name,
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
                dataset_payload = datasets_map.get(dataset_id) or {}
                row_idx, created, seeded = create_and_seed_batch_set(
                    dataset_name=display_name,
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
                        all_times = []
                        for spec in observations_from_payload(dataset_payload).values():
                            t_arr = np.asarray(spec.get("t", []), dtype=float).reshape(-1)
                            finite_times = t_arr[np.isfinite(t_arr)]
                            if finite_times.size:
                                all_times.extend(float(value) for value in finite_times)
                        t0 = float(min(all_times)) if all_times else float("nan")
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
                                    f"Dataset '{display_name}' does not start at t\u22480 "
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
        defaults_by_dataset[dataset_id] = defaults
        batch_set_names = list(batch_store.set_names()) if batch_store is not None else batch_set_names

    settings_map: Dict[str, "DatasetFitSettings"] = {}
    for dataset_id in selected_dataset_ids:
        display_name = display_name_by_id.get(str(dataset_id), str(dataset_id))
        settings = context.dataset_fit_settings_store.get_fit_settings(dataset_id)
        settings.ensure_species(mechanism_species, defaults_by_dataset.get(dataset_id, mechanism_initials))
        missing_initials = [
            species_name for species_name in (mechanism_species or [])
            if species_name not in (settings.initial_conditions or {})
        ]
        if missing_initials:
            QtWidgets.QMessageBox.warning(
                context.parent,
                "Global Fit",
                f"Dataset '{display_name}' requires initial concentrations for: {', '.join(missing_initials)}.",
            )
            context.set_status("Global fit cancelled")
            return None
        context.dataset_fit_settings_store.update_fit_settings(dataset_id, settings)
        settings_map[dataset_id] = settings

    dataset_params: Dict[str, Dict[str, float]] = {}
    dataset_variable_params: Dict[str, Dict[str, Dict[str, float]]] = {}
    weights: Dict[str, float] = {}
    initial_prefix = "init:"
    for dataset_id in selected_dataset_ids:
        settings = settings_map[dataset_id]
        weights[dataset_id] = settings.weight
        dataset_params[dataset_id] = {}
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
                dataset_params[dataset_id][key] = init_value
        if var_specs:
            dataset_variable_params[dataset_id] = var_specs

    t_axes: List[np.ndarray] = []
    dataset_entries: List[Dict[str, object]] = []
    dataset_payloads: List[Dict[str, object]] = []
    dataset_payload_results: Dict[str, object] = {}
    for dataset_id, payload in datasets_map.items():
        observations = observations_from_payload(payload)
        t_values, species_map = dense_view_from_observations(observations)
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
                "label": display_name_by_id.get(str(dataset_id), str(dataset_id)),
                "observations": copy_observations_map(observations),
                "t": t_values,
                "species_data": series_map,
                "selected_species": [],
                "weight": weights.get(str(dataset_id), 1.0),
                "include": True,
            }
        )
        fit_payload = _coerce_dataset_payload(
            dataset_id=str(dataset_id),
            observations=observations,
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
    launch_settings = _read_global_fit_launch_settings(context)

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
        from kindred.core.simulator.solvers import normalize_solver_name

        solver_label = _require_global_fit_launch_solver(
            solver if solver is not None else launch_settings.solver
        )
        solver_value, _solver_warning = normalize_solver_name(solver_label)
        rtol_value = _require_global_fit_positive_finite_float(
            rtol if rtol is not None else launch_settings.rtol,
            "rtol",
        )
        atol_value = _require_global_fit_positive_finite_float(
            atol if atol is not None else launch_settings.atol,
            "atol",
        )
        evaluator_inputs = launch_settings.runtime_inputs.evaluator
        runtime_settings = FittingEvaluatorRuntimeSettings(
            temperature_K=(
                temperature_K if temperature_K is not None else evaluator_inputs.temperature_K
            ),
            use_sparse_jacobian=(
                use_sparse_jacobian
                if use_sparse_jacobian is not None
                else evaluator_inputs.use_sparse_jacobian
            ),
            wegscheider_cyclicity_enabled=(
                wegscheider_cyclicity_enabled
                if wegscheider_cyclicity_enabled is not None
                else evaluator_inputs.wegscheider_cyclicity_enabled
            ),
        )
        fit_context = prepare_fitting_execution_context(
            mechanism_text=str(mechanism_text_for_run or ""),
            param_names=[str(x) for x in (param_names_for_run or []) if str(x)],
            t_end=max_time,
            num_points=grid_points,
            solver=solver_value,
            rtol=rtol_value,
            atol=atol_value,
            **runtime_settings.builder_kwargs(),
            initial_prefix=initial_prefix,
        )
        return SerialFittingEvaluator(fit_context)

    window_factory = _resolve_window_factory(context)
    window = window_factory(
        mode="global",
        parameter_defs=parameter_defs,
        dataset_entries=dataset_entries,
        dataset_fit_settings_store=context.dataset_fit_settings_store,
        dataset_view_publisher=context.dataset_view_publisher,
        mechanism_parameter_scan_owner=context.mechanism_parameter_scan_owner,
        simulation_func=None,
        mechanism_species=mechanism_species,
        mechanism_text_getter=context.mechanism_text_getter,
        reactions_text_getter=context.reactions_text_getter,
        reactions_text_setter=context.reactions_text_setter,
        simulation_builder=_build_simulation,
        runtime_inputs=launch_settings.runtime_inputs,
        dataset_params=dataset_params,
        dataset_variable_params=dataset_variable_params,
        dataset_payloads=dataset_payloads,
        dataset_payload_results=dataset_payload_results,
        dataset_weights=weights,
        project_apply_callback=context.apply_fit_results_to_project,
        config_defaults=context.load_fitting_defaults(),
        parent=context.parent,
    )
    window.setWindowTitle(f"Global Fit – {dataset_label}")
    context.set_status(f"Global fitting window open ({dataset_label})")
    context.register_fit_window(window, runtime_inputs=launch_settings.runtime_inputs)
    return window
