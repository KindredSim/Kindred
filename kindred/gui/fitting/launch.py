"""Launch-time orchestration for fitting sessions."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

import numpy as np
from PySide6 import QtWidgets

from kindred.core.analysis.fit_dataset_payload import FitDatasetPayloadResult, read_fit_dataset_payload
from kindred.core.fitting_evaluation import SerialFittingEvaluator, prepare_fitting_execution_context
from kindred.core.exceptions import FitSimulationError
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
from kindred.gui.project_schema import PROJECT_DEFAULTS

if TYPE_CHECKING:
    from kindred.gui.controllers.dataset_manager import DatasetFitSettings


logger = logging.getLogger(__name__)

__all__ = ["GlobalFitLaunchContext", "launch_global_fit_session"]


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
    wegscheider_enabled = bool(
        solver_settings.get(
            "wegscheider_cyclicity_enabled",
            PROJECT_DEFAULTS["wegscheider_cyclicity_enabled"],
        )
    )
    param_names = [str(entry.get("name")) for entry in (parameter_defs or []) if entry.get("name")]

    simulation_func = None
    try:
        fit_context = prepare_fitting_execution_context(
            mechanism_text=mechanism_text,
            param_names=param_names,
            t_end=max_time,
            num_points=grid_points,
            temperature_K=float(context.temperature_getter()),
            solver=str(solver_settings.get("solver") or FITTING_DEFAULT_SOLVER),
            rtol=float(solver_settings.get("rtol") or 1e-6),
            atol=float(solver_settings.get("atol") or 1e-12),
            use_sparse_jacobian=bool(
                solver_settings.get("use_sparse_jacobian", PROJECT_DEFAULTS["use_sparse_jacobian"])
            ),
            wegscheider_cyclicity_enabled=bool(wegscheider_enabled),
            initial_prefix=initial_prefix,
        )
    except FitSimulationError:
        logger.debug("Global-fit launch deferred fitting evaluator construction until run.", exc_info=True)
    else:
        simulation_func = SerialFittingEvaluator(fit_context)

    def _build_simulation(
        mechanism_text_for_run: str,
        param_names_for_run: List[str],
        *,
        solver: Optional[str] = None,
        rtol: Optional[float] = None,
        atol: Optional[float] = None,
    ):
        current_solver_settings = solver_settings
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
        fit_context = prepare_fitting_execution_context(
            mechanism_text=str(mechanism_text_for_run or ""),
            param_names=[str(x) for x in (param_names_for_run or []) if str(x)],
            t_end=max_time,
            num_points=grid_points,
            temperature_K=float(context.temperature_getter()),
            solver=solver_value,
            rtol=rtol_value,
            atol=atol_value,
            use_sparse_jacobian=bool(current_solver_settings.get("use_sparse_jacobian")),
            wegscheider_cyclicity_enabled=bool(current_solver_settings.get("wegscheider_cyclicity_enabled")),
            initial_prefix=initial_prefix,
        )
        return SerialFittingEvaluator(fit_context)

    window_factory = _resolve_window_factory(context)
    window = window_factory(
        mode="global",
        parameter_defs=parameter_defs,
        dataset_entries=dataset_entries,
        dataset_manager=context.dataset_manager,
        simulation_func=simulation_func,
        mechanism_species=mechanism_species,
        mechanism_text_getter=context.mechanism_text_getter,
        reactions_text_getter=context.reactions_text_getter,
        reactions_text_setter=context.reactions_text_setter,
        simulation_builder=_build_simulation,
        dataset_params=dataset_params,
        dataset_variable_params=dataset_variable_params,
        dataset_payloads=dataset_payloads,
        dataset_payload_results=dataset_payload_results,
        dataset_weights=weights,
        apply_callback=context.write_fit_results_to_mechanism,
        project_apply_callback=context.apply_fit_results_to_project,
        config_defaults=context.load_fitting_defaults(),
        shared_solver_settings_getter=context.get_solver_settings,
        dataset_settings_updater=context.apply_dataset_initial_updates,
        parent=context.parent,
    )
    window.setWindowTitle(f"Global Fit – {dataset_label}")
    context.set_status(f"Global fitting window open ({dataset_label})")
    context.register_fit_window(window)
    return window
