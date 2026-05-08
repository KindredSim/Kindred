"""
Fitting window and adjacent package-local UI helpers.

This module is the physical implementation root for the fitting subsystem UI.
"""

from __future__ import annotations

import hashlib
import logging
import math
import weakref
from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple, TYPE_CHECKING

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt

if TYPE_CHECKING:
    from kindred.core.api.fitting import GlobalFitResult

from kindred.core.api.fitting import fit_global
from kindred.core.api.simulation import (
    SimulationBuilder,
    SimulationBuilderContractError,
    coerce_simulation_builder,
)
from kindred.core.analysis.fit_dataset_payload import (
    FitDatasetPayloadResult,
    FitDatasetSpec,
    coerce_fit_dataset_payload_result,
    read_fit_dataset_payload,
)
from kindred.core.analysis.dataset_parameter_overrides import (
    FitDatasetParameterOverrides,
)
from kindred.core.fitting_evaluation import (
    FITTING_PARAM_ORIGIN_CONFIGURED_EVALUATOR,
    SerialFittingEvaluator,
    coerce_fitting_series_evaluator,
    evaluate_fitting_series,
)
from kindred.core.batch_parallel import compute_effective_batch_workers
from kindred.core.fitting_runtime_session import FittingRuntimeLedger, FittingRuntimeSession
from kindred.gui.project_schema import PROJECT_DEFAULTS
from kindred.core.simulator.solvers import normalize_solver_name
from kindred.core.simulation_preparation import PreparedSimulationMetadata
from kindred.gui.fitting.constants import FITTING_DEFAULT_SOLVER
from kindred.gui.fitting.launch import FittingLaunchIdentityOwner, validate_de_bounds
from kindred.core.simulation_failure import (
    coerce_simulation_failure,
    simulation_failure_detail_text,
    simulation_failure_user_message,
)
from kindred.gui.fitting.run_stamp import (
    build_global_fit_run_stamp,
    hash_global_fit_run_stamp,
)
from kindred.gui.fitting.runtime_readiness import (
    FittingRuntimeReadinessController,
    FittingRuntimeReadinessState,
)
from kindred.gui.fitting.data_tab import DataTab
from kindred.gui.fitting.data_targets_tab import DataTargetsTab
from kindred.gui.fitting.evaluator_state import FittingEvaluatorStateOwner
from kindred.gui.fitting.parameters_ics_tab import ParametersIcsTab
from kindred.gui.fitting.run_results_tab import RunResultsTab
from kindred.gui.fitting.run_command import FittingRunCommandOwner
from kindred.gui.fitting.run_state import FittingRunStateOwner
from kindred.gui.fitting.runtime_preparation import FittingRuntimePreparationOwner
from kindred.gui.fitting.unified_species_table import UnifiedSpeciesTable
from kindred.gui.fitting.worker_lifecycle import FitWorkerStopPolicy
from kindred.gui.fitting.worker_launch import FittingAcceptedLaunchWorkerOwner
from kindred.gui.fitting.worker import GlobalFitWorker
from kindred.core.analysis.dataset_sampling import compute_sampled_indices, compute_windowed_indices
from kindred.gui.fitting.constants import INITIAL_PREFIX, _SAMPLING_ALL_POINTS_SENTINEL, DEFAULT_PARALLEL_STARTS

logger = logging.getLogger(__name__)


class _EvaluateSeriesEvaluatorLike(Protocol):
    def evaluate_series(self, params: Mapping[str, float]) -> Dict[str, Any]:
        ...

_PROJECT_APPLY_SCOPE_PARAMETERS = "parameters"
_PROJECT_APPLY_SCOPE_INITIAL_CONDITIONS = "initial_conditions"
_PROJECT_APPLY_SCOPE_BOTH = "both"
_PROJECT_APPLY_OPTIONS: tuple[tuple[str, str], ...] = (
    ("Parameters only", _PROJECT_APPLY_SCOPE_PARAMETERS),
    ("Initial conditions only", _PROJECT_APPLY_SCOPE_INITIAL_CONDITIONS),
    ("Parameters and initial conditions", _PROJECT_APPLY_SCOPE_BOTH),
)


__all__ = [
    "DEFAULT_PARALLEL_STARTS",
    "FittingWindow",
    "GlobalFitWorker",
    "fit_global",
    "validate_de_bounds",
]


class _FitDialogWorkerRegistry(QtCore.QObject):
    """Owns detached fit-worker wrappers until they finish and can be deleted safely."""

    def __init__(self) -> None:
        super().__init__()
        self._retained_threads: List[QtCore.QThread] = []
        self._release_watchers: Dict[int, QtCore.QTimer] = {}

    @staticmethod
    def _is_running(worker: QtCore.QThread) -> bool:
        try:
            return bool(getattr(worker, "isRunning", lambda: False)())
        except Exception:
            return False

    def contains_thread(self, worker: QtCore.QThread) -> bool:
        return any(item is worker for item in self._retained_threads)

    def register_thread(self, worker: QtCore.QThread) -> None:
        if worker is None or self.contains_thread(worker):
            return
        self._retained_threads.append(worker)
        finished = getattr(worker, "finished", None)
        if finished is not None and hasattr(finished, "connect"):
            try:
                finished.connect(lambda *_args, w=worker: self.release_thread(w))
            except Exception as exc:
                logger.debug("Failed to connect fit-worker registry release hook: %s", exc, exc_info=True)
        self._start_release_watch(worker)
        if not self._is_running(worker):
            self.release_thread(worker)

    def release_thread(self, worker: QtCore.QThread) -> None:
        if worker is None:
            return
        self._stop_release_watch(worker)
        self._retained_threads = [item for item in self._retained_threads if item is not worker]
        self.schedule_cleanup(worker)

    def _start_release_watch(self, worker: QtCore.QThread) -> None:
        key = id(worker)
        watcher = self._release_watchers.get(key)
        if watcher is None:
            watcher = QtCore.QTimer(self)
            watcher.setInterval(50)
            watcher.setSingleShot(False)
            watcher.timeout.connect(lambda w=worker: self._poll_worker_release(w))
            self._release_watchers[key] = watcher
        if not watcher.isActive():
            watcher.start()

    def _stop_release_watch(self, worker: QtCore.QThread) -> None:
        watcher = self._release_watchers.pop(id(worker), None)
        if watcher is None:
            return
        if watcher.isActive():
            watcher.stop()
        watcher.deleteLater()

    def _poll_worker_release(self, worker: QtCore.QThread) -> None:
        if worker is None:
            return
        if self._is_running(worker):
            return
        self.release_thread(worker)

    def schedule_cleanup(self, worker: QtCore.QThread) -> None:
        if worker is None:
            return

        def _attempt_delete() -> None:
            if self._is_running(worker):
                QtCore.QTimer.singleShot(50, _attempt_delete)
                return
            try:
                worker.deleteLater()
            except Exception:
                return

        QtCore.QTimer.singleShot(0, _attempt_delete)


class _FitRuntimePreparationNotifier(QtCore.QObject):
    completed = QtCore.Signal()


# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

class _SimulationWithFixedParams:
    def __init__(self, base_evaluator: object, fixed_params: Dict[str, float]):
        self.base_evaluator = coerce_fitting_series_evaluator(base_evaluator)
        self.fixed_params = dict(fixed_params)

    @property
    def prepared_metadata(self) -> Optional[PreparedSimulationMetadata]:
        try:
            return getattr(self.base_evaluator, "prepared_metadata", None)
        except Exception:
            return None

    def __call__(self, params: Dict[str, float]):
        return self.evaluate_series(params)

    def evaluate_series(self, params: Dict[str, float]):
        origins = {
            str(name): FITTING_PARAM_ORIGIN_CONFIGURED_EVALUATOR
            for name in dict(params or {})
            if str(name).strip()
        }
        return self.evaluate_series_with_parameter_origins(params, origins)

    def evaluate_series_with_parameter_origins(
        self,
        params: Dict[str, float],
        origins: Optional[Dict[str, str]] = None,
        *,
        failed_params: Optional[Dict[str, float]] = None,
    ):
        merged = dict(self.fixed_params)
        merged_origins = {
            str(name): FITTING_PARAM_ORIGIN_CONFIGURED_EVALUATOR
            for name in merged
            if str(name).strip()
        }
        merged.update(dict(params or {}))
        for raw_name in dict(params or {}):
            name = str(raw_name)
            if not name.strip():
                continue
            merged_origins[name] = str((origins or {}).get(name, FITTING_PARAM_ORIGIN_CONFIGURED_EVALUATOR))
        return evaluate_fitting_series(
            self.base_evaluator,
            merged,
            origins=merged_origins,
            failed_params=failed_params,
        )


class FittingWindow(QtWidgets.QDialog):
    """
    Persistent fitting window that drives global fits.

    The window combines three workflow tabs:
        - Data and Targets (datasets, sampling, targets & weights, initial conditions)
        - Parameters (parameter table and integration settings)
        - Results (fit diagnostics and run stamp review)

    It owns the lifetime of GlobalFitWorker instances and updates plots +
    parameter tables after each run, enabling iterative workflows.
    """

    WINDOW_MODES = {"global"}

    def __init__(
        self,
        *,
        mode: str,
        parameter_defs: Sequence[Dict[str, Any]],
        dataset_entries: Sequence[Dict[str, Any]],
        dataset_manager: Optional[Any] = None,
        simulation_func: Optional[
            Callable[[Dict[str, float]], Dict[str, np.ndarray]] | _EvaluateSeriesEvaluatorLike
        ] = None,
        fit_func: Optional[Callable[..., "GlobalFitResult"]] = None,
        mechanism_species: Optional[Sequence[str]] = None,
        mechanism_text_getter: Optional[Callable[[], str]] = None,
        reactions_text_getter: Optional[Callable[[], str]] = None,
        reactions_text_setter: Optional[Callable[[str], None]] = None,
        simulation_builder: Optional[SimulationBuilder] = None,
        runtime_settings_getter: Optional[Callable[[], Mapping[str, Any]]] = None,
        dataset_params: Optional[Dict[str, Dict[str, float]]] = None,
        dataset_variable_params: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None,
        dataset_payloads: Optional[Sequence[Dict[str, Any]]] = None,
        dataset_payload_results: Optional[Dict[str, object]] = None,
        dataset_weights: Optional[Dict[str, float]] = None,
        apply_callback: Optional[Callable[[Dict[str, float]], None]] = None,
        project_apply_callback: Optional[Callable[[str, Dict[str, float], Dict[str, Dict[str, float]]], None]] = None,
        dataset_settings_updater: Optional[Callable[[str, Dict[str, float]], None]] = None,
        config_defaults: Optional[Dict[str, Any]] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        normalized_mode = str(mode or "global").lower()
        if normalized_mode not in self.WINDOW_MODES:
            raise ValueError(f"Unsupported fitting mode: {mode!r}")

        self._mode = normalized_mode
        self._dataset_manager = dataset_manager
        self._fit_func = fit_func
        self._mechanism_species = [str(x) for x in (mechanism_species or []) if str(x).strip()]
        self._mechanism_text_getter = mechanism_text_getter
        self._reactions_text_getter = reactions_text_getter
        self._reactions_text_setter = reactions_text_setter
        self._simulation_builder = (
            coerce_simulation_builder(simulation_builder)
            if callable(simulation_builder)
            else None
        )
        self._fit_evaluator_state = FittingEvaluatorStateOwner(
            base_evaluator=simulation_func,
            simulation_builder=self._simulation_builder,
        )
        self._fit_launch_identity_owner = FittingLaunchIdentityOwner(self)
        self._fit_run_command_owner = FittingRunCommandOwner(self)
        self._fit_worker_launch_owner = FittingAcceptedLaunchWorkerOwner(self)
        self._runtime_settings_getter = runtime_settings_getter
        self._apply_callback = apply_callback
        self._project_apply_callback = project_apply_callback
        self._dataset_settings_updater = dataset_settings_updater
        self._config_defaults = dict(config_defaults or {})

        self._global_dataset_params = dict(dataset_params or {})
        self._global_dataset_variable_params = dict(dataset_variable_params or {})
        self._global_payload_results: Dict[str, FitDatasetPayloadResult] = {
            str(payload["id"]): FitDatasetPayloadResult.valid(payload)
            for payload in (dataset_payloads or [])
            if "id" in payload
        }
        for ds_id, result in (dataset_payload_results or {}).items():
            self._global_payload_results[str(ds_id)] = coerce_fit_dataset_payload_result(result)
        self._global_payload_lookup = {
            ds_id: dict(result.payload)
            for ds_id, result in self._global_payload_results.items()
            if result.state == "valid" and isinstance(result.payload, dict)
        }
        self._global_weights = dict(dataset_weights or {})
        self._active_variable_specs: Dict[str, Dict[str, Dict[str, float]]] = {}

        self._fixed_shared_params: Dict[str, float] = {}
        self._prepared_param_names = [
            str(d.get("name") or "")
            for d in (parameter_defs or [])
            if isinstance(d, dict) and str(d.get("name") or "").strip()
        ]
        self._parameter_state = ParametersIcsTab._build_parameter_state(self, parameter_defs)
        self._initial_parameter_snapshot = [dict(row) for row in self._parameter_state]
        self._dataset_entries = self._normalize_dataset_entries(dataset_entries)
        if not self._dataset_entries:
            raise ValueError("At least one dataset entry is required for fitting.")

        self._load_dataset_pool_from_entries()

        # Per-dataset sampling/X-axis state (applied only; UI edits are pending until Apply).
        self._sampling_applied: Dict[str, Dict[str, float | int | str]] = {}

        self._init_fit_run_state()

        self.setWindowTitle("Fitting Window")
        self.resize(1280, 720)
        self.setWindowFlags(
            self.windowFlags()
            | Qt.WindowMinMaxButtonsHint
        )

        self._build_ui()
        # _apply_config_defaults and _populate_parameter_table are handled
        # internally by ParametersIcsTab during construction
        self._mark_fit_window_state_mechanism_current(
            self._safe_text_from_getter(self._mechanism_text_getter)
        )
        self._init_sampling_state()
        self._populate_dataset_table()
        self._species_table.on_tab_activated(
            seed_dataset_id=self._selected_data_table_dataset_id()
        )
        self._on_targets_validity_changed()
        self._refresh_sampling_validity_ui()
        ds_id = self._data_targets_tab.unified_list.selected_dataset_id()
        if ds_id:
            self._data_tab.select_dataset(ds_id)
        self._refresh_plot_baselines()
        self._fit_runtime_preparation_owner.schedule_refresh()


    def _load_dataset_pool_from_entries(self) -> None:
        # Snapshot of datasets available to this Global Fit session (already-loaded at window launch).
        # This enables in-window add/remove without mutating the project's dataset list.
        self._loaded_dataset_pool = {}
        self._loaded_dataset_order = []
        self._loaded_dataset_series_parse_failures = {}
        self._teardown_disable_failures = set()
        self._best_effort_failures = set()

        for entry in self._dataset_entries:
            ds_id = str(entry.get("id") or "").strip()
            if not ds_id or ds_id in self._loaded_dataset_pool:
                continue
            try:
                t_values = np.asarray(entry.get("t", []), dtype=float).reshape(-1).copy()
            except Exception:
                t_values = np.asarray([], dtype=float)
            raw_series = entry.get("species_data") or entry.get("species") or {}
            series_map: dict[str, np.ndarray] = {}
            series_failures: list[str] = []
            if isinstance(raw_series, dict):
                for name, values in raw_series.items():
                    key = str(name).strip()
                    if not key:
                        continue
                    try:
                        series_map[key] = np.asarray(values, dtype=float).reshape(-1).copy()
                    except Exception as exc:
                        series_failures.append(key)
                        if len(series_failures) <= 3:
                            logger.debug(
                                "Skipping invalid global-fit dataset '%s' series '%s' while loading: %s",
                                ds_id,
                                key,
                                exc,
                                exc_info=True,
                            )
                        continue
            if series_failures:
                self._loaded_dataset_series_parse_failures[ds_id] = list(series_failures)
            self._loaded_dataset_pool[ds_id] = {
                "id": ds_id,
                "label": str(entry.get("label") or ds_id),
                "t": t_values,
                "species_data": series_map,
            }
            self._loaded_dataset_order.append(ds_id)

    def _init_fit_run_state(self) -> None:
        self._worker = None
        self._results_rebuild_pending = False
        self._worker_registry = _FitDialogWorkerRegistry()
        self_ref = weakref.ref(self)
        self._worker_stop_policy = FitWorkerStopPolicy(
            record_failure=lambda key: (
                getattr(self_ref(), "_best_effort_failures", set()).add(str(key))
                if self_ref() is not None
                else None
            )
        )
        self._paused = False
        self._best_cost = None
        self._pre_run_parameter_state: Optional[List[Dict[str, Any]]] = None
        self._pre_run_staged_dataset_params: Optional[Dict[str, Dict[str, float]]] = None
        self._latest_model_series = {}
        self._latest_dataset_stats = {}
        self._latest_plot_model_series = {}
        self._latest_plot_model_x = {}
        self._last_result = None
        self._fit_run_state_owner = FittingRunStateOwner()
        self._fit_runtime_prepare_notifier = _FitRuntimePreparationNotifier(self)
        self._fit_runtime_readiness = FittingRuntimeReadinessController(
            session_factory=self._create_fit_runtime_session,
            finished_callback=self._fit_runtime_prepare_notifier.completed.emit,
        )
        self._fit_runtime_preparation_owner = FittingRuntimePreparationOwner(self)
        self._fit_window_state_refreshing = False
        self._fit_runtime_prepare_notifier.completed.connect(
            self._fit_runtime_preparation_owner.poll_preparation,
            QtCore.Qt.ConnectionType.QueuedConnection,
        )
        self._closing = False
        self._pending_best_payload = None
        self._pending_best_worker = None
        self._pending_best_timer = QtCore.QTimer(self)
        self._pending_best_timer.setSingleShot(True)
        self._pending_best_timer.setInterval(150)
        self._pending_best_timer.timeout.connect(self._apply_pending_best_update)

    @property
    def _is_fit_running(self) -> bool:
        return bool(self._worker and hasattr(self._worker, "isRunning") and self._worker.isRunning())

    @property
    def fit_launch_identity_owner(self) -> FittingLaunchIdentityOwner:
        return self._fit_launch_identity_owner

    @property
    def fit_run_command_owner(self) -> FittingRunCommandOwner:
        return self._fit_run_command_owner

    @property
    def fit_run_state_owner(self) -> FittingRunStateOwner:
        return self._fit_run_state_owner

    @property
    def fit_runtime_preparation_owner(self) -> FittingRuntimePreparationOwner:
        return self._fit_runtime_preparation_owner

    @property
    def fit_runtime_readiness(self) -> FittingRuntimeReadinessController:
        return self._fit_runtime_readiness

    @property
    def fit_worker_launch_owner(self) -> FittingAcceptedLaunchWorkerOwner:
        return self._fit_worker_launch_owner

    def _request_rebuild_subtabs(self) -> None:
        if self._is_fit_running:
            self._results_rebuild_pending = True
        else:
            self._run_results_tab.rebuild_subtabs(self._dataset_entries, self._results_fit_targets_by_dataset())

    def _active_integration_defaults_for_ui(self) -> Tuple[str, float, float]:
        """Read fitting integration defaults from the persisted fitting preferences."""
        _solver_raw = self._config_defaults.get("solver")
        solver_default, _warning = normalize_solver_name(
            _solver_raw if _solver_raw is not None else FITTING_DEFAULT_SOLVER
        )
        try:
            _rtol_raw = self._config_defaults.get("rtol")
            rtol_default = float(_rtol_raw) if _rtol_raw is not None else 1e-6
        except (TypeError, ValueError):
            rtol_default = 1e-6
        try:
            _atol_raw = self._config_defaults.get("atol")
            atol_default = float(_atol_raw) if _atol_raw is not None else 1e-12
        except (TypeError, ValueError):
            atol_default = 1e-12

        if not (np.isfinite(rtol_default) and rtol_default > 0.0):
            rtol_default = 1e-6
        if not (np.isfinite(atol_default) and atol_default > 0.0):
            atol_default = 1e-12
        return str(solver_default), float(rtol_default), float(atol_default)

    def _modeled_series_names_for_x_axis(self) -> set[str]:
        modeled = {str(x) for x in (self._params_ics_tab.get_mechanism_species() or []) if str(x).strip()}
        if callable(getattr(self, "_reactions_text_getter", None)):
            try:
                from kindred.core.algebra.observable_introspection import extract_observables_from_algebra_text
                from kindred.core.simulator.algebra_section import extract_algebra_section_text

                reactions_text = str(self._reactions_text_getter() or "")
                algebra_text = extract_algebra_section_text(reactions_text)
                observables = extract_observables_from_algebra_text(algebra_text)
                if isinstance(observables, dict):
                    modeled |= {str(k) for k in observables.keys() if str(k).strip()}
            except Exception:
                return modeled
        return modeled

    def _sampling_default_config_for_time_axis(self, t_values: np.ndarray) -> Dict[str, float | int | str]:
        t_axis = np.asarray(t_values, dtype=float).reshape(-1)
        if t_axis.size == 0:
            return {
                "t_min": 0.0,
                "t_max": 0.0,
                "n_points": int(_SAMPLING_ALL_POINTS_SENTINEL),
                "x_name": "t",
                "x_mapping_mode": "auto",
            }
        return {
            "t_min": float(np.min(t_axis)),
            "t_max": float(np.max(t_axis)),
            "n_points": int(_SAMPLING_ALL_POINTS_SENTINEL),
            "x_name": "t",
            "x_mapping_mode": "auto",
        }

    def _sampling_applied_config_for_dataset(self, dataset_id: str) -> Dict[str, float | int | str]:
        ds_id = str(dataset_id or "").strip()
        if not ds_id:
            return {
                "t_min": 0.0,
                "t_max": 0.0,
                "n_points": int(_SAMPLING_ALL_POINTS_SENTINEL),
                "x_name": "t",
                "x_mapping_mode": "auto",
            }
        cfg = self._sampling_applied.get(ds_id)
        if isinstance(cfg, dict) and cfg:
            return {
                "t_min": float(cfg.get("t_min", 0.0)),
                "t_max": float(cfg.get("t_max", 0.0)),
                "n_points": int(cfg.get("n_points", int(_SAMPLING_ALL_POINTS_SENTINEL))),
                "x_name": str(cfg.get("x_name") or "t").strip() or "t",
                "x_mapping_mode": str(cfg.get("x_mapping_mode") or "auto").strip() or "auto",
            }
        full_t = self._species_table.full_t_by_dataset.get(ds_id, np.asarray([]))
        return self._sampling_default_config_for_time_axis(full_t)

    def _init_sampling_state(self) -> None:
        full_t_map = self._species_table.full_t_by_dataset
        applied: Dict[str, Dict[str, float | int | str]] = {}
        for entry in self._dataset_entries or []:
            ds_id = str(entry.get("id") or "").strip()
            if not ds_id:
                continue
            full_t = full_t_map.get(ds_id, np.asarray(entry.get("t", [])))
            applied[ds_id] = self._sampling_default_config_for_time_axis(full_t)
        self._sampling_applied = applied
        self._refresh_dataset_entries_from_applied_fit_targets_and_sampling()
        self._rebuild_selected_payload_lookup()

    def _refresh_dataset_entries_from_applied_fit_targets_and_sampling(self) -> None:
        applied_sel = self._species_table.fit_targets_selection_applied
        full_t_map = self._species_table.full_t_by_dataset
        full_series_map = self._species_table.full_series_by_dataset
        for entry in self._dataset_entries or []:
            ds_id = str(entry.get("id") or "").strip()
            if not ds_id:
                continue
            selection = list(applied_sel.get(ds_id, []))
            entry["selected_species"] = list(selection)
            entry["target_weights"] = self._species_table.applied_target_weights_for_dataset(ds_id)

            full_t = full_t_map.get(ds_id, np.asarray(entry.get("t", []), dtype=float).reshape(-1))
            cfg = self._sampling_applied_config_for_dataset(ds_id)
            t_min = float(cfg.get("t_min", 0.0))
            t_max = float(cfg.get("t_max", 0.0))
            n_points_raw = int(cfg.get("n_points", int(_SAMPLING_ALL_POINTS_SENTINEL)))
            n_points = None if n_points_raw == int(_SAMPLING_ALL_POINTS_SENTINEL) else int(n_points_raw)
            try:
                idx = compute_sampled_indices(t=full_t, t_min=t_min, t_max=t_max, n_points=n_points)
            except Exception:
                idx = compute_windowed_indices(t=full_t, t_min=t_min, t_max=t_max)

            sampled_t = np.asarray(full_t[idx], dtype=float).reshape(-1) if idx.size else np.asarray([], dtype=float)
            entry["t"] = sampled_t

            full_series = full_series_map.get(ds_id, {})
            sampled_series: Dict[str, np.ndarray] = {}
            if selection and isinstance(full_series, dict) and idx.size:
                for name in selection:
                    if name not in full_series:
                        continue
                    series = np.asarray(full_series[name], dtype=float).reshape(-1)
                    if series.size != full_t.size:
                        continue
                    sampled_series[name] = np.asarray(series[idx], dtype=float).reshape(-1)
            entry["species_data"] = sampled_series

            x_name = str(cfg.get("x_name") or "t").strip() or "t"
            entry["x_name"] = x_name
            x_mode = str(cfg.get("x_mapping_mode") or "auto").strip().lower().replace("-", "_").replace(" ", "_") or "auto"
            if x_mode in ("monotone_only", "monotoneonly"):
                x_mode = "monotone"
            if x_mode not in ("auto", "monotone", "time_guided"):
                x_mode = "auto"
            entry["x_mapping_mode"] = x_mode
            if x_name != "t" and isinstance(full_series, dict) and idx.size and x_name in full_series:
                try:
                    x_series = np.asarray(full_series[x_name], dtype=float).reshape(-1)
                except Exception:
                    x_series = np.asarray([], dtype=float)
                if x_series.size == full_t.size:
                    entry["x_obs"] = np.asarray(x_series[idx], dtype=float).reshape(-1)
                else:
                    entry.pop("x_obs", None)
            else:
                entry.pop("x_obs", None)

    def _rebuild_selected_payload_lookup(self) -> None:
        rebuilt: Dict[str, Dict[str, Any]] = {}
        rebuilt_results: Dict[str, FitDatasetPayloadResult] = {}
        for ds_id, selection in (self._species_table.fit_targets_selection_applied or {}).items():
            entry = next((e for e in (self._dataset_entries or []) if str(e.get("id") or "").strip() == str(ds_id)), None)
            if not isinstance(entry, dict):
                rebuilt_results[str(ds_id)] = FitDatasetPayloadResult.absent()
                continue
            if not selection:
                existing = self._global_payload_results.get(str(ds_id))
                rebuilt_results[str(ds_id)] = (
                    existing if isinstance(existing, FitDatasetPayloadResult) and existing.state == "invalid"
                    else FitDatasetPayloadResult.absent()
                )
                continue
            series_map = entry.get("species_data") or {}
            t_values = entry.get("t", np.asarray([]))
            x_name = str(entry.get("x_name") or "t").strip() or "t"
            x_obs = entry.get("x_obs")
            x_mapping_mode = str(entry.get("x_mapping_mode") or "auto").strip() or "auto"
            result = read_fit_dataset_payload(
                dataset_id=ds_id,
                t=t_values,
                species_data=series_map,
                selected_species=selection,
                target_weights=self._species_table.applied_target_weights_for_dataset(ds_id),
                x_name=x_name,
                x_obs=x_obs,  # already sampled against the same indices as t/y
                x_mapping_mode=x_mapping_mode,
            )
            rebuilt_results[str(ds_id)] = result
            if result.state != "valid" or result.payload is None:
                continue
            rebuilt[ds_id] = dict(result.payload)
        self._global_payload_results = rebuilt_results
        self._global_payload_lookup = rebuilt

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------
    def _normalize_dataset_entries(self, entries: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        for entry in entries or []:
            dataset_id = str(entry.get("id", entry.get("label", "dataset")))
            t_values = np.asarray(entry.get("t", []), dtype=float).reshape(-1)
            species_data_raw = entry.get("species_data") or entry.get("species") or {}
            species_data: Dict[str, np.ndarray] = {}
            for species_name, values in species_data_raw.items():
                try:
                    species_data[str(species_name)] = np.asarray(values, dtype=float).reshape(-1)
                except Exception as exc:
                    logger.debug(
                        "Skipping invalid dataset entry series '%s' while normalizing global-fit data: %s",
                        species_name,
                        exc,
                        exc_info=True,
                    )
                    continue
            selected_species = entry.get("selected_species") or []
            entry_target_weights = dict(entry.get("target_weights")) if isinstance(entry.get("target_weights"), dict) else None
            seeded_payload = self._global_payload_lookup.get(dataset_id, {}) if isinstance(self._global_payload_lookup, dict) else {}
            payload_target_weights = (
                dict(seeded_payload.get("target_weights"))
                if isinstance(seeded_payload, dict) and isinstance(seeded_payload.get("target_weights"), dict)
                else {}
            )
            normalized.append(
                {
                    "id": dataset_id,
                    "label": str(entry.get("label", "") or "").strip() or dataset_id,
                    "t": t_values,
                    "species_data": species_data,
                    "selected_species": list(selected_species),
                    "target_weights": dict(entry_target_weights) if entry_target_weights is not None else dict(payload_target_weights),
                    "weight": float(entry.get("weight", 1.0)),
                    "include": bool(entry.get("include", True)),
                }
            )
        return normalized

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)
        # Use a weak reference to avoid reference cycles through callable getters
        # (FittingWindow -> DataTab -> getter -> FittingWindow). Without this,
        # Qt C++ destruction proceeds but the Python wrappers survive due to the
        # cycle, causing RuntimeError on double-close during test teardown.
        _w = weakref.ref(self)
        _empty_cfg = {"t_min": 0.0, "t_max": 0.0, "n_points": 0, "x_name": "t", "x_mapping_mode": "auto"}
        self._species_table = UnifiedSpeciesTable(
            dataset_entries=list(self._dataset_entries),
            mechanism_species=list(self.__dict__.get("_mechanism_species", [])),
            dataset_entries_getter=lambda: list(_w()._dataset_entries) if _w() is not None else [],
            included_dataset_ids_getter=lambda: _w()._included_dataset_ids() if _w() is not None else [],
            dataset_label_getter=lambda ds_id: _w()._dataset_label_for_id(ds_id) if _w() is not None else str(ds_id),
            dataset_weight_getter=lambda ds_id: _w()._dataset_weight_for_id(ds_id) if _w() is not None else 1.0,
            persist_dataset_weight_callback=lambda ds_id, w: _w()._persist_dataset_weight(ds_id, w) if _w() is not None else None,
            dataset_manager_getter=lambda: _w()._dataset_manager if _w() is not None else None,
            worker_running_getter=lambda: bool(_w() is not None and _w()._worker and hasattr(_w()._worker, "isRunning") and _w()._worker.isRunning()),
            modeled_series_getter=lambda: _w()._modeled_series_names_for_x_axis() if _w() is not None else set(),
            parent=self,
        )
        # Rewrite dataset_entries species_data to match applied fit-target selection
        # before result views are built, so they see only applied species.
        self._refresh_dataset_entries_from_applied_fit_targets_and_sampling()
        self._data_tab = DataTab(
            sampling_applied_config_getter=lambda ds_id: _w()._sampling_applied_config_for_dataset(ds_id) if _w() is not None else dict(_empty_cfg),
            sampling_default_config_getter=lambda t_values: _w()._sampling_default_config_for_time_axis(t_values) if _w() is not None else dict(_empty_cfg),
            fit_targets_full_t_getter=lambda ds_id: _w()._species_table.full_t_by_dataset.get(ds_id, np.asarray([])) if _w() is not None else np.asarray([]),
            fit_targets_available_getter=lambda ds_id: list(_w()._species_table.available_by_dataset.get(ds_id, [])) if _w() is not None else [],
            fit_targets_full_series_getter=lambda ds_id: _w()._species_table.full_series_by_dataset.get(ds_id, {}) if _w() is not None else {},
            fit_targets_selection_applied_getter=lambda ds_id: list(_w()._species_table.fit_targets_selection_applied.get(ds_id, [])) if _w() is not None else [],
            modeled_series_getter=lambda: _w()._modeled_series_names_for_x_axis() if _w() is not None else set(),
            worker_running_getter=lambda: bool(_w() is not None and _w()._worker and hasattr(_w()._worker, "isRunning") and _w()._worker.isRunning()),
            parent=self,
        )
        self._params_ics_tab = ParametersIcsTab(
            parameter_state=self.__dict__.pop("_parameter_state"),
            initial_parameter_snapshot=self.__dict__.pop("_initial_parameter_snapshot"),
            global_dataset_params=self.__dict__.pop("_global_dataset_params"),
            global_dataset_variable_params=self.__dict__.pop("_global_dataset_variable_params"),
            fixed_shared_params=self.__dict__.pop("_fixed_shared_params"),
            shared_param_definitions=self.__dict__.pop("_shared_param_definitions"),
            mechanism_species=self.__dict__.pop("_mechanism_species"),
            dataset_entries=list(self._dataset_entries),
            prepared_param_names=self.__dict__.pop("_prepared_param_names"),
            selected_dataset_ids_getter=lambda: _w()._selected_dataset_ids() if _w() is not None else [],
            dataset_entries_getter=lambda: list(_w()._dataset_entries) if _w() is not None else [],
            worker_running_getter=lambda: bool(_w() is not None and _w()._worker and hasattr(_w()._worker, "isRunning") and _w()._worker.isRunning()),
            dataset_manager_getter=lambda: _w()._dataset_manager if _w() is not None else None,
            reactions_text_getter=lambda: str(_w()._reactions_text_getter() or "") if _w() is not None and callable(getattr(_w(), "_reactions_text_getter", None)) else "",
            integration_defaults=self._active_integration_defaults_for_ui(),
            config_defaults=self._config_defaults,
            initial_parameter_defaults_getter=self._species_table.initial_parameter_defaults_for_species,
            parent=self,
        )
        self._params_ics_tab.addAlgebraicObservableRequested.connect(self._on_algebraic_observable_requested)
        self._run_results_tab = RunResultsTab(parent=self)
        self._run_results_tab.set_dark_mode(self._current_results_dark_mode())
        self._data_targets_tab = DataTargetsTab(
            data_tab=self._data_tab,
            species_table=self._species_table,
            parent=self,
        )

        self._tabs = QtWidgets.QTabBar(self)
        self._tabs.setObjectName("global_fit_top_tabs")
        self._tabs.setDocumentMode(True)
        self._tabs.setDrawBase(False)
        self._tabs.addTab("Data and Targets")
        self._tabs.addTab("Parameters")
        self._results_tab_index = self._tabs.addTab("Results")
        layout.addWidget(self._tabs, stretch=0)

        # Right: current tab content
        self._current_tab_stack = QtWidgets.QStackedWidget(self)
        self._current_tab_stack.setObjectName("global_fit_current_tab_stack")
        self._current_tab_stack.setMinimumWidth(320)
        self._current_tab_stack.addWidget(self._data_targets_tab)
        self._current_tab_stack.addWidget(self._params_ics_tab)
        self._current_tab_stack.addWidget(self._run_results_tab)
        layout.addWidget(self._current_tab_stack, stretch=1)

        self._tabs.currentChanged.connect(self._current_tab_stack.setCurrentIndex)
        self._tabs.currentChanged.connect(self._on_right_tabs_current_changed)
        self._current_tab_stack.setCurrentIndex(self._tabs.currentIndex())
        layout.addWidget(self._create_footer(), stretch=0)
        self._species_table.icApplied.connect(self._params_ics_tab._on_ic_applied)
        self._species_table.targetsApplied.connect(self._on_targets_applied)
        self._species_table.validityChanged.connect(self._on_targets_validity_changed)
        self._species_table.statusMessage.connect(self._on_targets_status_message)
        self._params_ics_tab.statusMessage.connect(self._status_label.setText)
        self._params_ics_tab.runtimeInputsChanged.connect(self._on_fit_runtime_inputs_changed)
        self._connect_fit_runtime_input_signals()
        self._run_results_tab.statusMessage.connect(self._status_label.setText)
        self._data_tab.statusMessage.connect(self._status_label.setText)
        self._data_targets_tab.unified_list.datasetIncludeChanged.connect(self._on_data_tab_include_changed)
        self._data_targets_tab.unified_list.addRequested.connect(self._open_add_datasets_dialog)
        self._data_targets_tab.unified_list.removeRequested.connect(self._remove_datasets_from_session)
        self._data_tab.samplingApplied.connect(self._on_data_tab_sampling_applied)
        self._run_results_tab.rebuild_subtabs(self._dataset_entries, self._results_fit_targets_by_dataset())
        self._refresh_project_apply_controls()

    def _connect_fit_runtime_input_signals(self) -> None:
        tab = self._params_ics_tab
        table = getattr(tab, "_param_table", None)
        if table is not None:
            table.itemChanged.connect(self._on_fit_runtime_inputs_changed)
        for signal_owner, signal_name in (
            (getattr(tab, "_method_combo", None), "currentTextChanged"),
            (getattr(tab, "_max_eval_spin", None), "valueChanged"),
            (getattr(tab, "_seed_spin", None), "valueChanged"),
            (getattr(tab, "_seed_check", None), "stateChanged"),
            (getattr(tab, "_ftol_edit", None), "editingFinished"),
            (getattr(tab, "_xtol_edit", None), "editingFinished"),
            (getattr(tab, "_integration_solver_combo", None), "currentTextChanged"),
            (getattr(tab, "_integration_rtol_edit", None), "textChanged"),
            (getattr(tab, "_integration_atol_edit", None), "textChanged"),
            (getattr(self._species_table, "_weight_mode_combo", None), "currentIndexChanged"),
        ):
            signal = getattr(signal_owner, signal_name, None) if signal_owner is not None else None
            if signal is not None:
                signal.connect(self._on_fit_runtime_inputs_changed)

    def _on_fit_runtime_inputs_changed(self, *_args) -> None:
        if getattr(self, "_closing", False):
            return
        worker = getattr(self, "_worker", None)
        if worker is not None and self._worker_is_running(worker):
            self._supersede_active_fit_worker_for_runtime_input_change()
        else:
            if worker is not None:
                self._fit_run_state_owner.mark_superseded()
                self._clear_pending_best_update_state()
                self._clear_worker_ref_if_stopped(worker)
                self._discard_fit_runtime_session(kill=True)
                self._set_running_state(False)
                if hasattr(self, "_status_label"):
                    self._status_label.setText("Fitting runtime inputs changed; fit cancelled")
            else:
                self._fit_run_state_owner.clear_superseded()
            self._fit_runtime_readiness.set_desired_identity(None)
        self._fit_run_state_owner.clear_active_run_stamp_hash()
        self._refresh_run_button_enabled_state()
        self._fit_runtime_preparation_owner.schedule_refresh()

    def handle_external_runtime_inputs_changed(self) -> None:
        self._on_fit_runtime_inputs_changed()

    def _on_algebraic_observable_requested(self, selection: dict) -> None:
        """Handle algebraic observable add request from ParametersIcsTab."""
        obs_name = str(selection.get("name") or "").strip()
        obs_expr = str(selection.get("expr") or "").strip()
        dataset_ids = selection.get("dataset_ids") or []
        scalar_scope = str(selection.get("scalar_scope") or "shared")
        persist = bool(selection.get("persist", False))
        if not obs_name or not obs_expr:
            return
        self._add_algebraic_observable(obs_name, obs_expr, dataset_ids, scalar_scope=scalar_scope, persist_observable=persist)
        self._params_ics_tab.repaint_parameter_table()
        self._on_fit_runtime_inputs_changed()

    def _create_footer(self) -> QtWidgets.QWidget:
        footer = QtWidgets.QWidget(self)
        footer.setObjectName("global_fit_footer")
        control_row = QtWidgets.QHBoxLayout(footer)
        control_row.setContentsMargins(0, 0, 0, 0)
        control_row.setSpacing(8)

        status_col_widget = QtWidgets.QWidget(footer)
        status_col = QtWidgets.QVBoxLayout(status_col_widget)
        status_col.setContentsMargins(0, 0, 0, 0)
        status_col.setSpacing(2)
        self._status_label = QtWidgets.QLabel("Ready")
        status_col.addWidget(self._status_label)
        self._run_block_reason_label = QtWidgets.QLabel(status_col_widget)
        self._run_block_reason_label.setObjectName("global_fit_run_block_reason_label")
        self._run_block_reason_label.setWordWrap(True)
        self._run_block_reason_label.setStyleSheet("font-size: 11px;")
        self._run_block_reason_label.hide()
        status_col.addWidget(self._run_block_reason_label)
        control_row.addWidget(status_col_widget, stretch=3)

        self._progress_bar = QtWidgets.QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        control_row.addWidget(self._progress_bar, stretch=2)

        self._run_button = QtWidgets.QPushButton("Run Fit")
        self._run_button.clicked.connect(self.run_fit)
        self._stop_button = QtWidgets.QPushButton("Stop")
        self._stop_button.setEnabled(False)
        self._stop_button.clicked.connect(self._cancel_fit)
        self._pause_button = QtWidgets.QPushButton("Pause")
        self._pause_button.setEnabled(False)
        self._pause_button.setToolTip(
            "Pause a running global fit. Takes effect at evaluation boundaries; "
            "the current simulation may finish before pausing."
        )
        self._pause_button.clicked.connect(self._pause_fit)
        self._resume_button = QtWidgets.QPushButton("Resume")
        self._resume_button.setEnabled(False)
        self._resume_button.setToolTip("Resume a paused global fit (continues the current run).")
        self._resume_button.clicked.connect(self._resume_fit)

        actions_row = QtWidgets.QHBoxLayout()
        actions_row.setContentsMargins(0, 0, 0, 0)
        actions_row.setSpacing(6)
        actions_row.addWidget(self._run_button)
        actions_row.addWidget(self._stop_button)
        actions_row.addWidget(self._pause_button)
        actions_row.addWidget(self._resume_button)

        self._results_summary_button = QtWidgets.QPushButton("Results Summary")
        self._results_summary_button.setObjectName("global_fit_results_summary_footer_button")
        self._results_summary_button.setEnabled(False)
        self._results_summary_button.clicked.connect(self._run_results_tab.open_results_summary_dialog)
        actions_row.addWidget(self._results_summary_button)

        self._apply_scope_combo = QtWidgets.QComboBox(footer)
        self._apply_scope_combo.setObjectName("global_fit_apply_scope_combo")
        for label, scope in _PROJECT_APPLY_OPTIONS:
            self._apply_scope_combo.addItem(label, userData=scope)
        self._apply_scope_combo.setEnabled(False)
        self._apply_to_project_button = QtWidgets.QPushButton("Apply to Project")
        self._apply_to_project_button.setObjectName("global_fit_apply_to_project_button")
        self._apply_to_project_button.setEnabled(False)
        self._apply_to_project_button.clicked.connect(self._apply_to_project)
        actions_row.addWidget(self._apply_scope_combo)
        actions_row.addWidget(self._apply_to_project_button)

        control_row.addLayout(actions_row)
        return footer

    def _selected_data_table_dataset_id(self) -> Optional[str]:
        return self._data_targets_tab.unified_list.selected_dataset_id()

    def _on_right_tabs_current_changed(self, index: int) -> None:
        # Targets panel is always visible in unified layout; activate on tab switch.
        if int(index) == 0:
            self._species_table.on_tab_activated(
                seed_dataset_id=self._selected_data_table_dataset_id()
            )

    def _results_fit_targets_by_dataset(self) -> Dict[str, List[str]]:
        if not hasattr(self, "_species_table"):
            return {}
        applied = dict(self._species_table.fit_targets_selection_applied or {})
        out: Dict[str, List[str]] = {}
        for entry in self._dataset_entries or []:
            ds_id = str(entry.get("id") or "").strip()
            if not ds_id:
                continue
            selected = applied.get(ds_id)
            if selected is None:
                selected = entry.get("selected_species") or []
            out[ds_id] = [str(name).strip() for name in selected if str(name).strip()]
        return out

    def _current_results_dark_mode(self) -> bool:
        for candidate in (self.parentWidget(), self.parent()):
            if candidate is not None and hasattr(candidate, "_dark_mode"):
                return bool(getattr(candidate, "_dark_mode"))
        app = QtWidgets.QApplication.instance()
        if app is None:
            return False
        window_color = app.palette().color(QtGui.QPalette.ColorRole.Window)
        return int(window_color.lightness()) < 128

    def changeEvent(self, event: QtCore.QEvent) -> None:
        super().changeEvent(event)
        if hasattr(self, "_run_results_tab") and event.type() in (
            QtCore.QEvent.Type.PaletteChange,
            QtCore.QEvent.Type.ApplicationPaletteChange,
            QtCore.QEvent.Type.StyleChange,
        ):
            self._run_results_tab.set_dark_mode(self._current_results_dark_mode())

    def _dataset_entry_for_id(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        ds_id = str(dataset_id or "").strip()
        if not ds_id:
            return None
        for entry in self._dataset_entries:
            if str(entry.get("id") or "").strip() == ds_id:
                return entry
        return None

    def _dataset_weight_for_id(self, dataset_id: str) -> float:
        entry = self._dataset_entry_for_id(dataset_id)
        if isinstance(entry, dict):
            try:
                value = float(entry.get("weight", 1.0))
            except Exception:
                value = 1.0
            if np.isfinite(value):
                return max(0.0, float(value))
        if self._dataset_manager is not None and hasattr(self._dataset_manager, "get_fit_settings"):
            try:
                settings = self._dataset_manager.get_fit_settings(str(dataset_id))
                persisted = float(getattr(settings, "weight", 1.0))
            except Exception:
                persisted = None
            if persisted is not None and np.isfinite(persisted):
                return max(0.0, float(persisted))
        try:
            fallback = float(self._global_weights.get(str(dataset_id), 1.0))
        except Exception:
            fallback = 1.0
        return max(0.0, fallback) if np.isfinite(fallback) else 1.0

    def _persist_dataset_weight(self, dataset_id: str, weight: float) -> None:
        ds_id = str(dataset_id or "").strip()
        if not ds_id:
            return
        normalized_weight = max(0.0, float(weight))
        previous_weight = self._dataset_weight_for_id(ds_id)
        entry = self._dataset_entry_for_id(ds_id)
        if isinstance(entry, dict):
            entry["weight"] = float(normalized_weight)
        self._global_weights[ds_id] = float(normalized_weight)
        if self._dataset_manager is not None and hasattr(self._dataset_manager, "get_fit_settings"):
            try:
                settings = self._dataset_manager.get_fit_settings(ds_id)
                setattr(settings, "weight", float(normalized_weight))
                if hasattr(self._dataset_manager, "update_fit_settings"):
                    self._dataset_manager.update_fit_settings(ds_id, settings)
            except Exception:
                pass
        if not math.isclose(float(previous_weight), float(normalized_weight), rel_tol=1e-12, abs_tol=1e-12):
            self._on_fit_runtime_inputs_changed()

    def _invalid_sampling_applied_used_dataset_ids(self) -> List[str]:
        used = set(self._included_dataset_ids())
        invalid: List[str] = []
        for ds_id in sorted(used):
            cfg = self._sampling_applied_config_for_dataset(ds_id)
            err = self._data_tab.sampling_validation_error(dataset_id=ds_id, config=cfg)
            if err:
                invalid.append(ds_id)
        return invalid

    def _refresh_sampling_validity_ui(self) -> None:
        if not hasattr(self, "_data_tab"):
            return
        invalid = self._invalid_sampling_applied_used_dataset_ids()
        if invalid:
            labels = [self._dataset_label_for_id(ds_id) for ds_id in invalid]
            joined = ", ".join(labels)
            message = (
                f"Run Fit disabled: {joined} has invalid applied sampling. Adjust sampling and Apply, or uncheck Use."
            )
            self._data_tab.set_sampling_secondary_error(message)
        else:
            self._data_tab.set_sampling_secondary_error(None)
        self._refresh_run_button_enabled_state()

    def _included_dataset_ids(self) -> List[str]:
        return [
            str(e.get("id") or "").strip()
            for e in (self._dataset_entries or [])
            if e.get("include", True) and str(e.get("id") or "").strip()
        ]

    def _dataset_label_for_id(self, dataset_id: str) -> str:
        ds_id = str(dataset_id or "").strip()
        for entry in self._dataset_entries or []:
            if str(entry.get("id") or "").strip() == ds_id:
                return str(entry.get("label", "") or "").strip() or str(dataset_id)
        return str(dataset_id)

    def _invalid_applied_used_dataset_ids_for_run(self) -> List[str]:
        invalid = set(self._species_table.invalid_applied_used_dataset_ids())
        try:
            invalid |= set(self._invalid_sampling_applied_used_dataset_ids())
        except Exception:
            invalid |= set()
        return sorted(invalid)

    def _refresh_run_button_enabled_state(self) -> None:
        if not hasattr(self, "_run_button"):
            return
        running = bool(self._worker and hasattr(self._worker, "isRunning") and self._worker.isRunning())
        readiness = self._fit_runtime_readiness.snapshot()
        preparing = readiness.state is FittingRuntimeReadinessState.PREPARING
        invalid = bool(self._invalid_applied_used_dataset_ids_for_run())
        ready = self._fit_runtime_ready_for_run_button()
        self._run_button.setEnabled((not running) and (not preparing) and not invalid and ready)

    def _fit_runtime_ready_for_run_button(self) -> bool:
        if self._fit_runtime_readiness.snapshot().state is not FittingRuntimeReadinessState.READY:
            return False
        try:
            identity = self._fit_launch_identity_owner.build_current_fit_runtime_identity()
        except RuntimeError:
            return False
        return self._fit_runtime_readiness.is_ready_for(identity)

    def _fitting_evaluator_components_for_runtime_identity(
        self,
        *,
        mechanism_text: str,
        config: Dict[str, Any],
        requested_solver: str,
        requested_rtol: float,
        requested_atol: float,
    ):
        return self._fit_evaluator_state.components_for_runtime_identity(
            mechanism_text=mechanism_text,
            config=config,
            requested_solver=requested_solver,
            requested_rtol=requested_rtol,
            requested_atol=requested_atol,
            runtime_settings=self._runtime_settings_for_identity(),
            param_names_for_readiness=self._param_names_for_readiness_identity,
        )

    def _runtime_settings_for_identity(self) -> dict[str, object]:
        settings: Mapping[str, Any] = {}
        getter = getattr(self, "_runtime_settings_getter", None)
        if callable(getter):
            try:
                raw_settings = getter()
                if isinstance(raw_settings, Mapping):
                    settings = raw_settings
            except Exception as exc:
                raise RuntimeError("Failed to read fitting runtime settings.") from exc
        try:
            temperature = float(settings.get("temperature_K", PROJECT_DEFAULTS["temperature_K"]))
        except Exception:
            temperature = float(PROJECT_DEFAULTS["temperature_K"])
        return {
            "temperature_K": f"{temperature:.12g}",
            "use_sparse_jacobian": bool(
                settings.get(
                    "use_sparse_jacobian",
                    PROJECT_DEFAULTS["use_sparse_jacobian"],
                )
            ),
            "wegscheider_cyclicity_enabled": bool(
                settings.get(
                    "wegscheider_cyclicity_enabled",
                    PROJECT_DEFAULTS["wegscheider_cyclicity_enabled"],
                )
            ),
        }

    def _prepared_simulation_matches_runtime_settings(
        self,
        prepared_simulation: Optional[PreparedSimulationMetadata],
    ) -> bool:
        if prepared_simulation is None:
            return False
        runtime_settings = self._runtime_settings_for_identity()
        try:
            prepared_temperature = float(prepared_simulation.temperature_K)
            requested_temperature = float(runtime_settings["temperature_K"])
        except Exception:
            return False
        return (
            np.isfinite(prepared_temperature)
            and np.isfinite(requested_temperature)
            and math.isclose(prepared_temperature, requested_temperature, rel_tol=1e-9, abs_tol=1e-12)
            and bool(prepared_simulation.use_sparse_jacobian) == bool(runtime_settings["use_sparse_jacobian"])
            and bool(prepared_simulation.wegscheider_cyclicity_enabled)
            == bool(runtime_settings["wegscheider_cyclicity_enabled"])
        )

    def _runtime_settings_for_builder_kwargs(self) -> dict[str, object]:
        runtime_settings = self._runtime_settings_for_identity()
        return {
            "temperature_K": float(runtime_settings["temperature_K"]),
            "use_sparse_jacobian": bool(runtime_settings["use_sparse_jacobian"]),
            "wegscheider_cyclicity_enabled": bool(runtime_settings["wegscheider_cyclicity_enabled"]),
        }

    def _param_names_for_readiness_identity(
        self,
        *,
        config: Dict[str, Any],
        prepared_simulation: Optional[PreparedSimulationMetadata],
        mechanism_text: str = "",
    ) -> list[str]:
        if prepared_simulation is not None and prepared_simulation.param_names:
            return [str(x) for x in prepared_simulation.param_names if str(x).strip()]
        scan_defs = getattr(self._params_ics_tab, "_scan_parameter_definitions_for_mechanism", None)
        if callable(scan_defs) and str(mechanism_text or "").strip():
            try:
                names = [str(d.get("name")) for d in (scan_defs(str(mechanism_text)) or []) if d.get("name")]
            except Exception:
                names = []
            if names:
                return [name for name in names if name.strip()]
        param_names = list(self._params_ics_tab.get_prepared_param_names() or [])
        if param_names:
            return [str(x) for x in param_names if str(x).strip()]
        names = set()
        names |= {str(k) for k in (config.get("parameters") or {}).keys() if str(k).strip()}
        names |= {str(k) for k in (config.get("fixed_params") or {}).keys() if str(k).strip()}
        return sorted(names)

    def _on_targets_validity_changed(self) -> None:
        if not hasattr(self, "_run_button"):
            return
        invalid_pending = set(self._species_table.invalid_pending_used_dataset_ids())
        invalid_pending_weights = set(self._species_table.invalid_pending_target_weight_dataset_ids())
        invalid_applied = set(self._species_table.invalid_applied_used_dataset_ids())

        # Row highlighting: applied-invalid is stronger than pending-invalid.
        for entry in self._dataset_entries:
            ds_id = str(entry.get("id") or "").strip()
            if not ds_id:
                continue
            if ds_id in invalid_applied:
                state = "invalid_applied"
            elif ds_id in invalid_pending or ds_id in invalid_pending_weights:
                state = "invalid_pending"
            else:
                state = ""
            self._data_targets_tab.unified_list.set_validation_state(ds_id, state)

        # Run Fit disabling while invalid applied.
        if invalid_applied:
            labels = [self._dataset_label_for_id(ds_id) for ds_id in sorted(invalid_applied)]
            joined = ", ".join(labels)
            message = (
                f"Run Fit disabled: {joined} has no applied fit targets. Select targets and Apply, or uncheck Use."
            )
            if hasattr(self, "_run_block_reason_label"):
                self._run_block_reason_label.setText(f"{message} Open Data and Targets to apply targets.")
                self._run_block_reason_label.show()
        else:
            if hasattr(self, "_run_block_reason_label"):
                self._run_block_reason_label.hide()

        self._refresh_run_button_enabled_state()

    def _on_targets_status_message(self, msg: str) -> None:
        self._status_label.setText(msg)

    def _on_targets_applied(self) -> None:
        self._on_fit_runtime_inputs_changed()
        self._refresh_dataset_entries_from_applied_fit_targets_and_sampling()
        self._rebuild_selected_payload_lookup()
        self._populate_dataset_table()
        self._on_targets_validity_changed()
        self._request_rebuild_subtabs()
        self._refresh_sampling_validity_ui()

    def _on_data_tab_include_changed(self, row: int, dataset_id: str, included: bool) -> None:
        self._on_fit_runtime_inputs_changed()
        entry = self._dataset_entry_for_id(dataset_id)
        if entry is not None:
            entry["include"] = included
        self._species_table.refresh_dataset_list()
        self._on_targets_validity_changed()
        self._refresh_sampling_validity_ui()
        self._request_rebuild_subtabs()

    def _on_data_tab_sampling_applied(self, dataset_id: str, config: dict) -> None:
        self._on_fit_runtime_inputs_changed()
        self._sampling_applied[str(dataset_id)] = dict(config)
        self._refresh_dataset_entries_from_applied_fit_targets_and_sampling()
        self._rebuild_selected_payload_lookup()
        self._request_rebuild_subtabs()
        self._refresh_sampling_validity_ui()
        self._status_label.setText("Sampling applied")

    def _open_add_datasets_dialog(self) -> None:
        present = {str(entry.get("id") or "").strip() for entry in (self._dataset_entries or []) if entry.get("id")}
        candidates = [ds_id for ds_id in (self._loaded_dataset_order or []) if ds_id and ds_id not in present]
        if not candidates:
            self._status_label.setText("No additional loaded datasets to add.")
            return

        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Add Datasets to Global Fit")
        dialog.setModal(True)
        layout = QtWidgets.QVBoxLayout(dialog)

        hint = QtWidgets.QLabel("Select loaded datasets to add to this Global Fit session.", dialog)
        hint.setWordWrap(True)
        hint.setStyleSheet("font-size: 11px;")
        layout.addWidget(hint)

        list_widget = QtWidgets.QListWidget(dialog)
        list_widget.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        for ds_id in candidates:
            pool_entry = self._loaded_dataset_pool.get(ds_id) or {}
            label = str(pool_entry.get("label") or ds_id)
            item = QtWidgets.QListWidgetItem(label, list_widget)
            item.setData(Qt.UserRole, ds_id)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            item.setCheckState(Qt.Unchecked)
        layout.addWidget(list_widget, stretch=1)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel, dialog)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        if dialog.exec() != QtWidgets.QDialog.Accepted:
            return

        chosen: List[str] = []
        for i in range(list_widget.count()):
            item = list_widget.item(i)
            if item is None:
                continue
            if item.checkState() != Qt.Checked:
                continue
            ds_id = str(item.data(Qt.UserRole) or "").strip()
            if ds_id:
                chosen.append(ds_id)
        self._add_datasets_to_session(chosen)

    def _add_datasets_to_session(self, dataset_ids: Sequence[str]) -> None:
        """Add datasets into this Global Fit session (from already-loaded pool only)."""
        present = {str(entry.get("id") or "").strip() for entry in (self._dataset_entries or []) if entry.get("id")}
        added = False
        for ds_id in [str(x).strip() for x in (dataset_ids or []) if str(x).strip()]:
            if ds_id in present:
                continue
            pool_entry = self._loaded_dataset_pool.get(ds_id)
            if not isinstance(pool_entry, dict):
                continue
            t_values = np.asarray(pool_entry.get("t", []), dtype=float).reshape(-1).copy()
            full_series = pool_entry.get("species_data") or {}
            series_map: Dict[str, np.ndarray] = {}
            series_failures: List[str] = []
            if isinstance(full_series, dict):
                for name, values in full_series.items():
                    key = str(name).strip()
                    if not key:
                        continue
                    try:
                        series_map[key] = np.asarray(values, dtype=float).reshape(-1).copy()
                    except Exception as exc:
                        series_failures.append(key)
                        if len(series_failures) <= 3:
                            logger.debug(
                                "Skipping invalid series '%s' for dataset '%s' when adding to global-fit session: %s",
                                key,
                                ds_id,
                                exc,
                                exc_info=True,
                            )
                        continue

            self._dataset_entries.append(
                {
                    "id": ds_id,
                    "label": str(pool_entry.get("label", "") or "").strip() or ds_id,
                    "t": t_values,
                    "species_data": {},  # applied selection starts empty
                    "selected_species": [],
                    "target_weights": {},
                    "weight": self._dataset_weight_for_id(ds_id),
                    "include": True,
                }
            )

            self._species_table.add_dataset_state(
                ds_id,
                full_series=dict(series_map),
                full_t=t_values,
                available=sorted(series_map.keys()),
            )
            self._sampling_applied[ds_id] = self._sampling_default_config_for_time_axis(t_values)

            # Seed per-dataset initial parameter maps from persisted fit settings (best-effort).
            self._params_ics_tab.seed_dataset_initial_params(ds_id)

            present.add(ds_id)
            added = True

        if not added:
            return
        self._sync_after_session_dataset_change()
        self._status_label.setText("Datasets added to session")

    def _remove_datasets_from_session(self, dataset_ids: Sequence[str]) -> None:
        """Remove datasets from this Global Fit session (does not delete from project)."""
        remove_set = {str(x).strip() for x in (dataset_ids or []) if str(x).strip()}
        if not remove_set:
            return

        self._dataset_entries = [entry for entry in self._dataset_entries if entry.get("id") not in remove_set]

        # Remove fit-target session state (keep loaded pool intact).
        self._species_table.remove_dataset_state(remove_set)
        for ds_id in list(remove_set):
            self._sampling_applied.pop(ds_id, None)
            self._global_payload_results.pop(ds_id, None)
            self._global_payload_lookup.pop(ds_id, None)
            self._active_variable_specs.pop(ds_id, None)

        # Delegate param/IC state cleanup and table repaint to the tab
        self._params_ics_tab.remove_dataset_parameter_rows(remove_set)

        self._sync_after_session_dataset_change()
        self._status_label.setText("Datasets removed from session")

    def _sync_after_session_dataset_change(self) -> None:
        self._populate_dataset_table()
        ds_id = self._data_targets_tab.unified_list.selected_dataset_id()
        self._data_tab.select_dataset(ds_id or "")
        self._species_table.refresh_dataset_list()
        self._species_table.refresh_dataset_entries(self._dataset_entries)
        self._params_ics_tab.refresh_dataset_entries(self._dataset_entries)
        self._on_targets_validity_changed()
        self._refresh_sampling_validity_ui()
        self._request_rebuild_subtabs()
        self._on_fit_runtime_inputs_changed()

    # ------------------------------------------------------------------
    # Table population
    # ------------------------------------------------------------------
    def _populate_dataset_table(self) -> None:
        self._data_tab.populate_table(self._dataset_entries)
        self._data_targets_tab.unified_list.populate(self._dataset_entries)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def _selected_dataset_ids(self) -> List[str]:
        selection = self._fit_launch_identity_owner.collect_dataset_selection()
        return list(selection.ids)

    def _add_algebraic_observable(
        self,
        name: str,
        expr: str,
        dataset_ids: Sequence[str],
        *,
        scalar_scope: str,
        persist_observable: bool,
    ) -> None:
        from kindred.core.algebra.observable_introspection import (
            analyze_observable_expression,
            detect_unknown_scalar_identifiers,
            extract_observables_from_algebra_text,
        )
        from kindred.core.algebra.symbols import SymbolTable
        from kindred.core.simulator.algebra_section import (
            extract_algebra_section_text,
            upsert_lines_into_algebra_section,
        )
        from kindred.core.validation import validate_name

        if not self._observable_dsl_edit_available():
            return

        normalized = self._normalize_observable_inputs(name, expr, validate_name=validate_name)
        if normalized is None:
            return
        obs_name, obs_expr = normalized

        mechanism_species = {str(x) for x in (self._params_ics_tab.get_mechanism_species() or []) if str(x).strip()}
        if not self._validate_observable_name_rules(obs_name, mechanism_species=mechanism_species, symbol_table=SymbolTable()):
            return

        reactions_text = str(self._reactions_text_getter() or "")
        algebra_text = extract_algebra_section_text(reactions_text)
        existing_obs_map = extract_observables_from_algebra_text(algebra_text)
        existing_observables = {str(x) for x in (existing_obs_map or {}).keys() if str(x).strip()}
        if not self._validate_observable_existence(obs_name, existing_observables=existing_observables, persist_observable=persist_observable):
            return

        try:
            analysis = analyze_observable_expression(obs_expr)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Add Observable", f"Invalid expression:\n\n{exc}")
            return
        if not self._validate_observable_expression_analysis(analysis, obs_expr=obs_expr, mechanism_species=mechanism_species):
            return

        known_identifiers = self._known_identifiers_for_observable(existing_observables=existing_observables)
        if obs_name in known_identifiers:
            QtWidgets.QMessageBox.warning(
                self,
                "Add Observable",
                f"Name '{obs_name}' is already used by a fit/solver parameter. Choose a different observable name.",
            )
            return
        known_identifiers |= {obs_name}

        missing_scalars = sorted(
            detect_unknown_scalar_identifiers(
                obs_expr,
                observable_name=obs_name,
                known_identifiers=known_identifiers,
                mechanism_species=mechanism_species,
            )
        )
        updated_reactions_text = self._persist_observable_updates(
            reactions_text=reactions_text,
            obs_name=obs_name,
            obs_expr=obs_expr,
            missing_scalars=missing_scalars,
            persist_observable=persist_observable,
            upsert_lines_into_algebra_section=upsert_lines_into_algebra_section,
        )
        if updated_reactions_text is None:
            return

        if not self._refresh_simulation_after_reactions_update():
            return

        self._params_ics_tab.add_missing_scalars_as_parameters(
            missing_scalars=missing_scalars,
            dataset_ids=dataset_ids,
            scalar_scope=scalar_scope,
        )

    def _observable_dsl_edit_available(self) -> bool:
        if not callable(getattr(self, "_reactions_text_getter", None)) or not callable(getattr(self, "_reactions_text_setter", None)):
            QtWidgets.QMessageBox.warning(
                self,
                "Add Observable",
                "Mechanism Reactions editor is unavailable in this window. Close and reopen Global Fit from the main window.",
            )
            return False
        if not callable(self._mechanism_text_getter) or not callable(self._simulation_builder) or self._dataset_manager is None:
            QtWidgets.QMessageBox.warning(
                self,
                "Add Observable",
                "Cannot refresh simulation/plumbing in this window. Close and reopen Global Fit from the main window.",
            )
            return False
        return True

    def _normalize_observable_inputs(
        self,
        name: str,
        expr: str,
        *,
        validate_name,
    ) -> Optional[tuple[str, str]]:
        import re

        try:
            obs_name = validate_name(str(name))
        except Exception:
            obs_name = str(name).strip()
        obs_expr = str(expr).strip()
        if not obs_name or not obs_expr:
            QtWidgets.QMessageBox.warning(self, "Add Observable", "Observable name and expression are required.")
            return None
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", obs_name):
            QtWidgets.QMessageBox.warning(
                self,
                "Add Observable",
                "Observable name must be a valid identifier: [A-Za-z_][A-Za-z0-9_]*",
            )
            return None
        return str(obs_name), str(obs_expr)

    def _validate_observable_name_rules(
        self,
        obs_name: str,
        *,
        mechanism_species: set[str],
        symbol_table,
    ) -> bool:
        import re

        if obs_name in symbol_table.protected_names() or obs_name in symbol_table.functions().keys():
            QtWidgets.QMessageBox.warning(
                self,
                "Add Observable",
                f"Observable name '{obs_name}' is reserved (built-in/protected). Choose a different name.",
            )
            return False
        if mechanism_species and obs_name in mechanism_species:
            QtWidgets.QMessageBox.warning(
                self,
                "Add Observable",
                f"Observable name '{obs_name}' conflicts with a mechanism species name.",
            )
            return False
        if re.fullmatch(r"(?i:(?:k|kf|kr|keq)\d+)", obs_name):
            QtWidgets.QMessageBox.warning(
                self,
                "Add Observable",
                f"'{obs_name}' looks like a mechanism parameter name. Choose a different observable name.",
            )
            return False
        return True

    def _validate_observable_existence(
        self,
        obs_name: str,
        *,
        existing_observables: set[str],
        persist_observable: bool,
    ) -> bool:
        if not persist_observable and obs_name not in existing_observables:
            QtWidgets.QMessageBox.warning(
                self,
                "Add Observable",
                f"Observable '{obs_name}' was not found in the Reactions text. Use ‘Define new…’ to add it.",
            )
            return False
        if persist_observable and obs_name in existing_observables:
            QtWidgets.QMessageBox.warning(
                self,
                "Add Observable",
                f"An algebraic observable named '{obs_name}' already exists.",
            )
            return False
        return True

    def _validate_observable_expression_analysis(
        self,
        analysis,
        *,
        obs_expr: str,
        mechanism_species: set[str],
    ) -> bool:
        if analysis.has_time_ref:
            QtWidgets.QMessageBox.warning(
                self,
                "Add Observable",
                "Algebra baseline references like [A](T0) are not supported for fitting (v1).",
            )
            return False
        bare_species = set(getattr(analysis, "identifiers", set()) or set()) & mechanism_species
        if bare_species:
            QtWidgets.QMessageBox.warning(
                self,
                "Add Observable",
                "Species must be referenced using brackets. Replace bare names with [name].\n\n"
                f"Bare species found: {', '.join(sorted(bare_species))}",
            )
            return False
        return True

    def _known_identifiers_for_observable(self, *, existing_observables: set[str]) -> set[str]:
        known: set[str] = set()
        known |= {str(k) for k in (self._params_ics_tab.get_shared_param_definitions() or {}).keys() if str(k).strip()}
        known |= {str(k) for k in (self._params_ics_tab.get_fixed_shared_params() or {}).keys() if str(k).strip()}
        known |= {
            str(entry.get("param_name") or "")
            for entry in (self._params_ics_tab.get_parameter_state() or [])
            if str(entry.get("param_name") or "").strip()
        }
        for _ds_id, specs in (self._params_ics_tab.get_global_dataset_variable_params() or {}).items():
            if not isinstance(specs, dict):
                continue
            known |= {str(k) for k in specs.keys() if str(k).strip()}
        known |= set(existing_observables)
        return known

    @staticmethod
    def _reactions_text_has_param_decl(reactions_text: str, name: str) -> bool:
        import re

        if not name:
            return False
        pat = rf"(?im)^\s*param\s+{re.escape(str(name))}\s*="
        return bool(re.search(pat, reactions_text or ""))

    def _persist_observable_updates(
        self,
        *,
        reactions_text: str,
        obs_name: str,
        obs_expr: str,
        missing_scalars: Sequence[str],
        persist_observable: bool,
        upsert_lines_into_algebra_section,
    ) -> Optional[str]:
        updated_reactions_text = reactions_text
        if missing_scalars or persist_observable:
            to_add: list[str] = []
            for scalar in missing_scalars:
                if self._reactions_text_has_param_decl(reactions_text, str(scalar)):
                    continue
                to_add.append(f"param {scalar} = 1.0")
            if persist_observable:
                to_add.append(f"let {obs_name} = {obs_expr}")
            if to_add:
                updated_reactions_text = upsert_lines_into_algebra_section(reactions_text, to_add)
        if updated_reactions_text != reactions_text:
            try:
                self._reactions_text_setter(updated_reactions_text)
            except Exception as exc:
                QtWidgets.QMessageBox.warning(self, "Add Observable", f"Failed to update Reactions DSL:\n\n{exc}")
                return None
        return str(updated_reactions_text)

    def _refresh_simulation_after_reactions_update(self) -> bool:
        try:
            mechanism_text = str(self._mechanism_text_getter() or "")
            param_names = self._refresh_parameter_definitions_for_mechanism(mechanism_text)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Add Observable", f"Failed to rescan mechanism parameters:\n\n{exc}")
            return False

        try:
            if not callable(self._simulation_builder):
                raise RuntimeError("Simulation builder unavailable.")
            integration = self._params_ics_tab.collect_integration_settings()
            if integration is None:
                solver, rtol, atol = (FITTING_DEFAULT_SOLVER, 1e-6, 1e-12)
            else:
                solver, rtol, atol = integration
            self._fit_evaluator_state.build_and_set(
                mechanism_text=mechanism_text,
                param_names=list(param_names),
                requested_solver=str(solver),
                requested_rtol=float(rtol),
                requested_atol=float(atol),
                runtime_settings=self._runtime_settings_for_identity(),
            )
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Add Observable", f"Failed to refresh simulation:\n\n{exc}")
            return False
        return True

    # ------------------------------------------------------------------
    # Fit lifecycle
    # ------------------------------------------------------------------
    def _reset_fit_run_cached_state(self) -> None:
        self._fit_run_state_owner.reset_for_new_run()
        self._pre_run_parameter_state = None
        self._pre_run_staged_dataset_params = None
        self._latest_model_series = {}
        self._latest_dataset_stats = {}
        self._latest_plot_model_series = {}
        self._latest_plot_model_x = {}
        self._best_cost = None
        self._best_effort_failures.clear()
        self._teardown_disable_failures.clear()

    def _mark_fit_window_state_mechanism_current(self, mechanism_text: str) -> None:
        text = str(mechanism_text or "")
        self._fit_window_state_mechanism_text_sha256 = self._mechanism_text_sha256(text)
        self._fit_window_state_mechanism_text_len = len(text)

    def _fit_window_state_matches_mechanism(self, mechanism_text: str) -> bool:
        expected_hash = str(getattr(self, "_fit_window_state_mechanism_text_sha256", "") or "")
        if not expected_hash:
            return False
        try:
            expected_len = int(getattr(self, "_fit_window_state_mechanism_text_len"))
        except Exception:
            return False
        text = str(mechanism_text or "")
        return expected_hash == self._mechanism_text_sha256(text) and expected_len == len(text)

    def _refresh_fit_window_state_for_current_mechanism(self, *, show_errors: bool = True) -> bool:
        mechanism_text = self._safe_text_from_getter(getattr(self, "_mechanism_text_getter", None))
        simulation_func = self._fit_evaluator_state.current_base_evaluator()
        prepared_simulation = self._prepared_simulation_meta(simulation_func)
        mechanism_matches = self._prepared_simulation_matches_mechanism(prepared_simulation, mechanism_text)
        table_matches = self._fit_window_state_matches_mechanism(mechanism_text)
        if not callable(getattr(self, "_simulation_builder", None)):
            return True
        if not callable(getattr(self, "_mechanism_text_getter", None)):
            return True
        if (prepared_simulation is None or mechanism_matches) and table_matches:
            return True
        if getattr(self, "_fit_window_state_refreshing", False):
            return False
        from kindred.gui.controllers.dataset_manager import DatasetManagerError

        self._fit_window_state_refreshing = True
        try:
            self._params_ics_tab.rebuild_for_mechanism(mechanism_text, list(self._dataset_entries))
            self._species_table.set_mechanism_species(list(self._params_ics_tab.get_mechanism_species()))
            self._species_table.refresh_dataset_entries(list(self._dataset_entries))
            self._capture_failed_fit_restore_baseline()
            self._mark_fit_window_state_mechanism_current(mechanism_text)
        except DatasetManagerError as exc:
            if show_errors:
                self._clear_failed_run_visual_state()
                QtWidgets.QMessageBox.warning(self, "Global Fit", str(exc))
            elif hasattr(self, "_status_label"):
                self._status_label.setText(f"Fitting runtime not ready: {exc}")
            return False
        except Exception as exc:
            logger.exception("Failed to refresh fit-window state before running fit.")
            if show_errors:
                self._clear_failed_run_visual_state()
                QtWidgets.QMessageBox.critical(
                    self,
                    "Simulation Error",
                    f"Failed to refresh fit-window state:\n{exc}",
                )
            elif hasattr(self, "_status_label"):
                self._status_label.setText(f"Fitting runtime not ready: {exc}")
            return False
        finally:
            self._fit_window_state_refreshing = False
        return True

    def run_fit(self) -> None:
        self._fit_run_command_owner.run_fit()

    def _datasets_payloads_for_run(self, selected_ids: Sequence[str]) -> Optional[list[dict[str, Any]]]:
        datasets: list[dict[str, Any]] = []
        for dataset_id in selected_ids:
            result = self._global_payload_results.get(str(dataset_id))
            if isinstance(result, FitDatasetPayloadResult) and result.state == "invalid":
                reason = str(result.error or "Dataset payload is invalid.")
                QtWidgets.QMessageBox.warning(
                    self,
                    "Global Fit",
                    f"Dataset '{dataset_id}' has invalid payload:\n{reason}",
                )
                return None
            if dataset_id not in self._global_payload_lookup:
                QtWidgets.QMessageBox.warning(self, "Global Fit", f"Dataset '{dataset_id}' is missing payloads.")
                return None
            datasets.append(dict(self._global_payload_lookup[dataset_id]))
        return datasets

    def _datasets_payloads_for_readiness(self, selected_ids: Sequence[str]) -> Optional[list[dict[str, Any]]]:
        datasets: list[dict[str, Any]] = []
        for dataset_id in selected_ids:
            result = self._global_payload_results.get(str(dataset_id))
            if isinstance(result, FitDatasetPayloadResult) and result.state == "invalid":
                return None
            if dataset_id not in self._global_payload_lookup:
                return None
            datasets.append(dict(self._global_payload_lookup[dataset_id]))
        return datasets

    @staticmethod
    def _shared_param_keys_for_run(config: Dict[str, Any]) -> set[str]:
        keys = {str(k) for k in (config.get("parameters") or {}).keys() if str(k).strip()}
        keys |= {str(k) for k in (config.get("fixed_params") or {}).keys() if str(k).strip()}
        return keys

    @staticmethod
    def _prepared_species_names_from_evaluator(evaluator: object) -> set[str]:
        candidates = [evaluator]
        try:
            base_evaluator = getattr(evaluator, "base_evaluator", None)
        except Exception:
            base_evaluator = None
        if base_evaluator is not None:
            candidates.append(base_evaluator)

        for candidate in candidates:
            try:
                context = getattr(candidate, "context", None)
            except Exception:
                context = None
            if context is None:
                try:
                    context = getattr(candidate, "_kindred_fitting_execution_context", None)
                except Exception:
                    context = None
            request = getattr(context, "execution_request", None)
            if request is None:
                continue
            payload = None
            to_payload = getattr(request, "to_payload", None)
            if callable(to_payload):
                try:
                    payload = to_payload()
                except Exception:
                    payload = None
            elif isinstance(request, Mapping):
                payload = request
            if not isinstance(payload, Mapping):
                continue
            prepared_payload = payload.get("prepared_payload")
            if not isinstance(prepared_payload, Mapping):
                continue
            species_names = {
                str(name)
                for name in (prepared_payload.get("species_names") or [])
                if str(name).strip()
            }
            if species_names:
                return species_names
        return set()

    def _consumable_dataset_param_names_for_run(self, evaluator: object | None = None) -> tuple[set[str], Optional[set[str]]]:
        evaluator = self._fit_evaluator_state.current_base_evaluator() if evaluator is None else evaluator
        species_names = self._prepared_species_names_from_evaluator(evaluator)
        if not species_names:
            species_names = {
                str(name)
                for name in (self._params_ics_tab.get_mechanism_species() or [])
                if str(name).strip()
            }
        prepared_simulation = self._prepared_simulation_meta(evaluator)
        if prepared_simulation is not None:
            prepared_param_names = {
                str(name)
                for name in (prepared_simulation.param_names or [])
                if str(name).strip()
            }
            return species_names, prepared_param_names
        return species_names, None

    @staticmethod
    def _dataset_param_consumable(
        name: str,
        *,
        mechanism_species: set[str],
        prepared_param_names: Optional[set[str]],
    ) -> bool:
        if name.startswith(INITIAL_PREFIX):
            species_name = name[len(INITIAL_PREFIX) :]
            return not mechanism_species or species_name in mechanism_species
        return prepared_param_names is None or name in prepared_param_names

    def _dataset_params_for_run(
        self,
        selected_ids: Sequence[str],
        shared_param_keys: set[str],
        staged_params: Dict[str, Dict[str, float]],
        *,
        global_dataset_params: Optional[Dict[str, Dict[str, float]]] = None,
        evaluator: object | None = None,
    ) -> dict[str, dict[str, float]]:
        global_ds_params = (
            {str(k): dict(v) for k, v in global_dataset_params.items()}
            if isinstance(global_dataset_params, dict)
            else self._params_ics_tab.get_global_dataset_params()
        )
        dataset_params_for_run: dict[str, dict[str, float]] = {}
        mechanism_species, prepared_param_names = self._consumable_dataset_param_names_for_run(evaluator)
        for ds_id in selected_ids:
            merged = dict(global_ds_params.get(ds_id, {}))
            stage_map = staged_params.get(ds_id)
            if isinstance(stage_map, dict):
                merged.update(stage_map)
            for key in shared_param_keys:
                merged.pop(key, None)
            merged = {
                str(key): value
                for key, value in merged.items()
                if self._dataset_param_consumable(
                    str(key),
                    mechanism_species=mechanism_species,
                    prepared_param_names=prepared_param_names,
                )
            }
            dataset_params_for_run[ds_id] = merged
        return dataset_params_for_run

    def _variable_params_for_run(
        self,
        selected_ids: Sequence[str],
        shared_param_keys: set[str],
        staged_params: Dict[str, Dict[str, float]],
        *,
        global_dataset_variable_params: Optional[Dict[str, Dict[str, Dict]]] = None,
        evaluator: object | None = None,
    ) -> dict[str, dict[str, dict[str, float]]]:
        global_ds_var_params = (
            {
                str(k): {str(kk): dict(vv) for kk, vv in v.items()} if isinstance(v, dict) else v
                for k, v in global_dataset_variable_params.items()
            }
            if isinstance(global_dataset_variable_params, dict)
            else self._params_ics_tab.get_global_dataset_variable_params()
        )
        variable_params: dict[str, dict[str, dict[str, float]]] = {}
        mechanism_species, prepared_param_names = self._consumable_dataset_param_names_for_run(evaluator)
        for ds_id in selected_ids:
            specs = global_ds_var_params.get(ds_id)
            if not isinstance(specs, dict) or not specs:
                continue
            stage_map = staged_params.get(ds_id) if isinstance(staged_params.get(ds_id), dict) else {}
            ds_specs: dict[str, dict[str, float]] = {}
            for param_name, spec in specs.items():
                if not isinstance(spec, dict):
                    continue
                if str(param_name) in shared_param_keys:
                    continue
                if not self._dataset_param_consumable(
                    str(param_name),
                    mechanism_species=mechanism_species,
                    prepared_param_names=prepared_param_names,
                ):
                    continue
                spec_copy = dict(spec)
                spec_copy.setdefault("log10", False)
                staged_value = stage_map.get(param_name) if isinstance(stage_map, dict) else None
                if staged_value is not None:
                    try:
                        staged_float = float(staged_value)
                    except (TypeError, ValueError):
                        staged_float = None
                    if staged_float is not None and np.isfinite(staged_float):
                        if bool(spec_copy.get("log10")):
                            if staged_float > 0.0:
                                spec_copy["initial"] = staged_float
                        else:
                            spec_copy["initial"] = staged_float
                ds_specs[str(param_name)] = spec_copy
            if ds_specs:
                variable_params[ds_id] = ds_specs
        return variable_params

    def _weights_for_run(self, dataset_selection: Dict[str, Any]) -> Optional[dict[str, float]]:
        if self._species_table.weight_mode_is_implicit():
            return None
        return {row["id"]: row["weight"] for row in dataset_selection.get("rows") or [] if row.get("include")}

    @staticmethod
    def _safe_text_from_getter(getter) -> str:
        if not callable(getter):
            return ""
        try:
            return str(getter() or "")
        except Exception:
            return ""

    @staticmethod
    def _prepared_simulation_meta(simulation_func) -> Optional[PreparedSimulationMetadata]:
        return FittingEvaluatorStateOwner.prepared_simulation_meta_for(simulation_func)

    @staticmethod
    def _simulation_with_fixed_params(simulation_func, fixed_params: Dict[str, float]):
        if simulation_func is None:
            return None
        normalized = coerce_fitting_series_evaluator(simulation_func)
        with_fixed_params = getattr(simulation_func, "with_fixed_params", None)
        if callable(with_fixed_params):
            return coerce_fitting_series_evaluator(with_fixed_params(fixed_params))
        return _SimulationWithFixedParams(normalized, fixed_params)

    @staticmethod
    def _prepared_solver_normalized(prepared_simulation: Optional[PreparedSimulationMetadata]) -> str:
        if prepared_simulation is None:
            return ""
        return str(prepared_simulation.solver_normalized).strip()

    @staticmethod
    def _mechanism_text_sha256(mechanism_text: str) -> str:
        return hashlib.sha256(str(mechanism_text or "").encode("utf-8")).hexdigest()

    @classmethod
    def _prepared_simulation_matches_mechanism(
        cls,
        prepared_simulation: Optional[PreparedSimulationMetadata],
        mechanism_text: str,
    ) -> bool:
        if prepared_simulation is None:
            return False
        expected_hash = str(getattr(prepared_simulation, "mechanism_text_sha256", "") or "")
        if not expected_hash:
            return False
        try:
            expected_len = int(getattr(prepared_simulation, "mechanism_text_len"))
        except Exception:
            return False
        text = str(mechanism_text or "")
        return expected_hash == cls._mechanism_text_sha256(text) and expected_len == len(text)

    def _refresh_parameter_definitions_for_mechanism(self, mechanism_text: str) -> list[str]:
        param_defs = self._params_ics_tab._scan_parameter_definitions_for_mechanism(mechanism_text)
        self._params_ics_tab.set_shared_param_definitions({str(d.get("name")): dict(d) for d in (param_defs or []) if d.get("name")})
        param_names = [str(d.get("name")) for d in (param_defs or []) if d.get("name")]
        self._params_ics_tab.set_prepared_param_names(list(param_names))
        return list(param_names)

    def _param_names_for_fit_run(self, prepared_simulation: Optional[PreparedSimulationMetadata]) -> list[str]:
        param_names = list(self._params_ics_tab.get_prepared_param_names() or [])
        if prepared_simulation is not None and prepared_simulation.param_names:
            try:
                param_names = [str(x) for x in prepared_simulation.param_names if str(x).strip()]
            except Exception:
                param_names = list(self._params_ics_tab.get_prepared_param_names() or [])
        return list(param_names)

    def _ensure_simulation_for_integration_settings(
        self,
        *,
        mechanism_text: str,
        param_names: Sequence[str],
        requested_solver: str,
        requested_rtol: float,
        requested_atol: float,
        prepared_simulation: Optional[PreparedSimulationMetadata],
    ) -> tuple[bool, Optional[PreparedSimulationMetadata]]:
        should_rebuild_sim = False
        if callable(getattr(self, "_simulation_builder", None)) and callable(getattr(self, "_mechanism_text_getter", None)):
            if prepared_simulation is None:
                should_rebuild_sim = True
            else:
                mechanism_matches = self._prepared_simulation_matches_mechanism(
                    prepared_simulation,
                    mechanism_text,
                )
                current_solver = self._prepared_solver_normalized(prepared_simulation)
                try:
                    current_rtol = float(prepared_simulation.rtol)
                except Exception:
                    current_rtol = float("nan")
                try:
                    current_atol = float(prepared_simulation.atol)
                except Exception:
                    current_atol = float("nan")
                should_rebuild_sim = not (
                    mechanism_matches
                    and
                    current_solver == requested_solver
                    and self._prepared_simulation_matches_runtime_settings(prepared_simulation)
                    and np.isfinite(current_rtol)
                    and np.isfinite(current_atol)
                    and math.isclose(float(current_rtol), float(requested_rtol), rel_tol=1e-9, abs_tol=1e-12)
                    and math.isclose(float(current_atol), float(requested_atol), rel_tol=1e-9, abs_tol=1e-12)
                )
        if not should_rebuild_sim:
            return True, prepared_simulation

        try:
            if not callable(self._simulation_builder):
                raise RuntimeError("Simulation builder unavailable.")
            current_param_names = list(self._params_ics_tab.get_prepared_param_names() or [])
            if not current_param_names:
                current_param_names = self._refresh_parameter_definitions_for_mechanism(mechanism_text)
            if not current_param_names:
                current_param_names = list(param_names)
            runtime_settings = self._runtime_settings_for_identity()
            _base_simulation, prepared_simulation = self._fit_evaluator_state.build_and_set(
                mechanism_text=mechanism_text,
                param_names=list(current_param_names),
                requested_solver=str(requested_solver),
                requested_rtol=float(requested_rtol),
                requested_atol=float(requested_atol),
                runtime_settings=runtime_settings,
            )
        except SimulationBuilderContractError as exc:
            logger.error("Simulation builder contract mismatch: %s", exc)
            QtWidgets.QMessageBox.critical(
                self,
                "Simulation Error",
                f"{exc}\n\n"
                "This is an internal integration error (the simulation builder must accept solver, rtol, atol).",
            )
            return False, prepared_simulation
        except Exception as exc:
            logger.exception("Failed to build simulation for fitting.")
            QtWidgets.QMessageBox.critical(
                self,
                "Simulation Error",
                f"Failed to build simulation for fitting:\n{exc}",
            )
            return False, prepared_simulation

        return True, prepared_simulation

    def _store_run_stamp_and_update_ui(
        self,
        *,
        dataset_selection: Dict[str, Any],
        included_ids: Sequence[str],
        weights: Optional[dict[str, float]],
        config: Dict[str, Any],
        mechanism_text: str,
        reactions_text: str,
        prepared_simulation: Optional[PreparedSimulationMetadata],
        dataset_overrides: Sequence[FitDatasetParameterOverrides],
        dataset_specs: Optional[Sequence[FitDatasetSpec]] = None,
    ) -> tuple[dict[str, Any], str, str]:
        weight_mode = "equal" if weights is None else "custom"
        applied_targets = dict(self._species_table.fit_targets_selection_applied or {})
        applied_target_weights = {
            str(ds_id): self._species_table.applied_target_weights_for_dataset(str(ds_id))
            for ds_id in applied_targets.keys()
        }
        stamp = build_global_fit_run_stamp(
            dataset_rows=list(dataset_selection.get("rows") or []),
            included_ids=list(included_ids),
            applied_fit_targets=applied_targets,
            applied_target_weights=applied_target_weights,
            weights_used=(dict(weights) if isinstance(weights, dict) else None),
            weight_mode=weight_mode,
            fit_config=dict(config or {}),
            mechanism_text=mechanism_text,
            reactions_text=reactions_text,
            prepared_simulation=prepared_simulation,
            dataset_specs=list(dataset_specs) if dataset_specs is not None else None,
            dataset_overrides=list(dataset_overrides),
        )
        stamp_hash = hash_global_fit_run_stamp(stamp)
        stamp_short = str(stamp_hash)[:12]
        self._run_results_tab.set_run_stamp(dict(stamp), str(stamp_hash), str(stamp_short))
        self._results_summary_button.setEnabled(True)
        return stamp, str(stamp_hash), str(stamp_short)

    @staticmethod
    def _fixed_params_for_run(config: Dict[str, Any]) -> dict[str, float]:
        fixed_params = config.get("fixed_params") or {}
        if not isinstance(fixed_params, dict):
            return {}
        out: dict[str, float] = {}
        for k, v in fixed_params.items():
            if not str(k).strip():
                continue
            try:
                out[str(k)] = float(v)
            except (TypeError, ValueError, OverflowError):
                continue
        return out

    def _fit_runtime_lane_budget(self, dataset_count: int) -> int:
        budget = None
        parent = self.parent()
        simulation_controller = None
        for owner in (parent, getattr(parent, "_sim_controller", None), getattr(parent, "simulation_controller", None)):
            if owner is None:
                continue
            if hasattr(owner, "batch_runtime_lane_budget"):
                simulation_controller = owner
                break
        if simulation_controller is not None:
            try:
                budget = int(getattr(simulation_controller, "batch_runtime_lane_budget"))
            except Exception:
                budget = None
        if budget is None:
            try:
                budget = int(PROJECT_DEFAULTS["batch_runtime_lane_budget"])
            except Exception:
                budget = 1
        return compute_effective_batch_workers(
            num_sets=max(1, int(dataset_count)),
            max_parallel_workers=max(1, int(budget)),
        )

    def _discard_fit_runtime_session(self, *, kill: bool = False) -> None:
        self._fit_runtime_readiness.cancel(kill=bool(kill))
        self._fit_runtime_readiness.close_ready_session(kill=bool(kill))

    def _set_fit_controls_locked(self, locked: bool) -> None:
        running = bool(locked)
        try:
            self._params_ics_tab.set_running_state(running)
            table = getattr(self._params_ics_tab, "_param_table", None)
            if table is not None:
                table.setEnabled(not running)
        except Exception as exc:
            self._best_effort_failures.add("fit_controls_locked.params_ics_tab")
            logger.debug("Failed to update Parameters/IC controls during fitting runtime preparation: %s", exc, exc_info=True)
        try:
            self._data_targets_tab.unified_list.set_running_state(running)
        except Exception as exc:
            self._best_effort_failures.add("fit_controls_locked.dataset_list")
            logger.debug("Failed to update dataset controls during fitting runtime preparation: %s", exc, exc_info=True)
        try:
            self._species_table.set_running_state(running)
        except Exception as exc:
            self._best_effort_failures.add("fit_controls_locked.species_table")
            logger.debug("Failed to update target controls during fitting runtime preparation: %s", exc, exc_info=True)
        self._refresh_project_apply_controls(running=running)

    def _create_fit_runtime_session(self, fit_evaluator, lane_count: int) -> Optional[FittingRuntimeSession]:
        if type(fit_evaluator) is not SerialFittingEvaluator:
            return None
        ledger = FittingRuntimeLedger()
        return FittingRuntimeSession.from_serial_evaluator(
            fit_evaluator,
            max_lanes=int(lane_count),
            ledger=ledger,
        )

    def _schedule_worker_cleanup(self, worker: QtCore.QThread) -> None:
        if worker is None:
            return
        self._worker_registry.schedule_cleanup(worker)

    def _hard_teardown_worker(self, *, reason: str, disable_ui: bool) -> None:
        worker = getattr(self, "_worker", None)
        if worker is None or not self._worker_is_running(worker):
            return

        self._disconnect_fit_worker_signals(worker)

        self._last_teardown_reason = str(reason)
        self._set_teardown_status_label(str(reason))
        self._disable_ui_for_worker_teardown(disable_ui=bool(disable_ui))

        self._cancel_worker_best_effort(worker)

        still_running = self._wait_for_worker_stop(worker, timeout_ms=2000, context="hard_teardown.wait")
        if still_running:
            still_running = self._terminate_worker_and_wait(worker, timeout_ms=2000)

        self._finalize_worker_after_teardown(worker, still_running=still_running)
        self._clear_worker_ref_if_stopped(worker)

    @staticmethod
    def _worker_is_running(worker: QtCore.QThread) -> bool:
        try:
            return bool(getattr(worker, "isRunning", lambda: False)())
        except Exception:
            return False

    def _active_fit_worker_running(self) -> bool:
        worker = getattr(self, "_worker", None)
        return bool(worker is not None and self._worker_is_running(worker))

    def _supersede_active_fit_worker_for_runtime_input_change(self) -> None:
        self._fit_run_state_owner.mark_superseded()
        self._clear_pending_best_update_state()
        self._hard_teardown_worker(
            reason="Fitting runtime inputs changed; fit cancelled",
            disable_ui=False,
        )
        self._discard_fit_runtime_session(kill=True)
        self._set_running_state(False)
        if hasattr(self, "_status_label"):
            self._status_label.setText("Fitting runtime inputs changed; fit cancelled")

    def _set_teardown_status_label(self, reason: str) -> None:
        status_label = getattr(self, "_status_label", None)
        if status_label is None:
            return
        try:
            status_label.setText(str(reason))
        except RuntimeError as exc:
            logger.debug("Failed to set teardown status label: %s", exc, exc_info=True)
            self._status_label = None

    def _disable_ui_for_worker_teardown(self, *, disable_ui: bool) -> None:
        if disable_ui:
            try:
                self.setEnabled(False)
                return
            except RuntimeError as exc:
                logger.debug("Failed to disable FitDialog during worker teardown: %s", exc, exc_info=True)

        for attr in ("_stop_button", "_pause_button", "_resume_button", "_run_button"):
            btn = getattr(self, attr, None)
            if btn is None:
                continue
            try:
                btn.setEnabled(False)
            except Exception as exc:
                self._teardown_disable_failures.add(attr)
                logger.debug("Failed to disable FitDialog button '%s' during worker teardown: %s", attr, exc, exc_info=True)

    def _cancel_worker_best_effort(self, worker: QtCore.QThread) -> None:
        self._worker_stop_policy.request_stop(worker, context="hard_teardown")

    def _wait_for_worker_stop(self, worker: QtCore.QThread, *, timeout_ms: int, context: str) -> bool:
        return self._worker_stop_policy.wait_for_stop(worker, timeout_ms=int(timeout_ms), context=context)

    def _terminate_worker_and_wait(self, worker: QtCore.QThread, *, timeout_ms: int) -> bool:
        return self._worker_stop_policy.terminate_if_needed(worker, timeout_ms=int(timeout_ms), context="hard_teardown")

    def _finalize_worker_after_teardown(self, worker: QtCore.QThread, *, still_running: bool) -> None:
        running_after = bool(still_running)
        if not running_after:
            self._worker_registry.schedule_cleanup(worker)
            return

        detached = False
        try:
            worker.setParent(None)
            detached = True
        except Exception as exc:
            logger.debug("Failed to detach worker parent during hard teardown: %s", exc, exc_info=True)
            detached = False
        if not detached:
            self._best_effort_failures.add("hard_teardown.detach_parent")
        self._worker_registry.register_thread(worker)

    def _clear_worker_ref_if_stopped(self, worker: QtCore.QThread) -> None:
        if getattr(self, "_worker", None) is not worker:
            return
        if not self._worker_is_running(worker):
            self._worker = None

    def _cancel_fit(self) -> None:
        readiness = self._fit_runtime_readiness.snapshot()
        if readiness.state is FittingRuntimeReadinessState.PREPARING:
            self._fit_runtime_preparation_owner.cancel_preparation(kill=True)
            try:
                self._stop_button.setEnabled(False)
                self._status_label.setText("Fitting runtime preparation cancelled")
                self._set_fit_controls_locked(False)
                self._fit_runtime_preparation_owner.refresh_pending = False
                self._refresh_run_button_enabled_state()
            except Exception as exc:
                self._best_effort_failures.add("cancel_fit_runtime_preparation.status_label")
                logger.debug("Failed to set fitting preparation cancellation status: %s", exc, exc_info=True)
            return
        worker = getattr(self, "_worker", None)
        if worker is None:
            return
        try:
            running = bool(getattr(worker, "isRunning", lambda: False)())
        except Exception:
            running = False
        if not running:
            return
        self._clear_pending_best_update_state()
        self._hard_teardown_worker(reason="Fit cancelled", disable_ui=False)
        self._discard_fit_runtime_session(kill=True)
        try:
            self._set_running_state(False)
            self._status_label.setText("Fit cancelled")
            self._fit_runtime_preparation_owner.schedule_refresh()
        except Exception as exc:
            self._best_effort_failures.add("cancel_fit.status_label")
            logger.debug("Failed to set cancellation status label: %s", exc, exc_info=True)

    def _pause_fit(self) -> None:
        if self._worker and self._worker.isRunning() and not self._paused:
            if hasattr(self._worker, "pause"):
                self._worker.pause()
                self._paused = True
                self._pause_button.setEnabled(False)
                self._resume_button.setEnabled(True)
                self._status_label.setText("Pause requested (after current evaluation)")

    def _resume_fit(self) -> None:
        if self._worker and self._worker.isRunning() and self._paused:
            if hasattr(self._worker, "resume"):
                self._worker.resume()
                self._paused = False
                self._pause_button.setEnabled(True)
                self._resume_button.setEnabled(False)
                self._status_label.setText("Resuming...")

    def _set_running_state(self, running: bool) -> None:
        invalid_applied = bool(self._invalid_applied_used_dataset_ids_for_run())
        preparing = self._fit_runtime_readiness.snapshot().state is FittingRuntimeReadinessState.PREPARING
        if running:
            self._run_button.setEnabled(False)
        else:
            self._run_button.setEnabled((not preparing) and not invalid_applied and self._fit_runtime_ready_for_run_button())
        self._stop_button.setEnabled(bool(running) or preparing)
        try:
            self._run_results_tab.set_view_autorange_locked(running)
        except Exception as exc:
            self._best_effort_failures.add("set_running_state.results_autorange_lock")
            logger.debug("Failed to update Results tab autorange lock state: %s", exc, exc_info=True)
        self._params_ics_tab.set_running_state(running)
        self._data_targets_tab.unified_list.set_running_state(running)
        self._species_table.set_running_state(running)
        self._paused = False
        self._pause_button.setEnabled(False)
        self._resume_button.setEnabled(False)
        if not running:
            self._worker = None
            self._progress_bar.setValue(0)
            if hasattr(self, "_pending_best_timer"):
                try:
                    self._pending_best_timer.stop()
                except Exception as exc:
                    self._best_effort_failures.add("set_running_state.pending_best_timer_stop")
                    logger.debug("Failed to stop pending-best timer: %s", exc, exc_info=True)
            self._pending_best_payload = None
            self._pending_best_worker = None
        self._species_table.refresh_validity_ui()
        self._refresh_sampling_validity_ui()
        if not running:
            self._refresh_run_button_enabled_state()
        self._refresh_project_apply_controls(running=running)

    # ------------------------------------------------------------------
    # Worker callbacks
    # ------------------------------------------------------------------
    def _fit_worker_sender(self) -> Optional[QtCore.QThread]:
        try:
            worker = self.sender()
        except Exception:
            return None
        return worker if worker is not None else None

    def _dispatch_fit_worker_target(self) -> Optional[QtCore.QThread]:
        worker = self._fit_worker_sender()
        if worker is not None:
            return worker
        current_worker = getattr(self, "_worker", None)
        return current_worker if current_worker is not None else None

    def _disconnect_fit_worker_signals(self, worker: Optional[QtCore.QThread]) -> None:
        if worker is None:
            return
        signal = getattr(worker, "progress", None)
        if signal is not None and hasattr(signal, "disconnect"):
            try:
                signal.disconnect(self._dispatch_fit_worker_progress)
            except (RuntimeError, TypeError):
                pass
        signal = getattr(worker, "bestUpdated", None)
        if signal is not None and hasattr(signal, "disconnect"):
            try:
                signal.disconnect(self._dispatch_fit_worker_best_update)
            except (RuntimeError, TypeError):
                pass
        signal = getattr(worker, "finished", None)
        if signal is not None and hasattr(signal, "disconnect"):
            try:
                signal.disconnect(self._dispatch_fit_worker_finished)
            except (RuntimeError, TypeError):
                pass
        signal = getattr(worker, "error", None)
        if signal is not None and hasattr(signal, "disconnect"):
            try:
                signal.disconnect(self._dispatch_fit_worker_error)
            except (RuntimeError, TypeError):
                pass

    def _stop_finished_worker(self, worker: Optional[QtCore.QThread]) -> None:
        if worker is None:
            return
        quit_worker = getattr(worker, "quit", None)
        if callable(quit_worker):
            try:
                quit_worker()
            except Exception:
                pass
        wait_for_worker = getattr(worker, "wait", None)
        if callable(wait_for_worker):
            try:
                wait_for_worker(2000)
            except Exception:
                pass

    def _clear_pending_best_update_state(self) -> None:
        if hasattr(self, "_pending_best_timer"):
            try:
                self._pending_best_timer.stop()
            except Exception as exc:
                self._best_effort_failures.add("clear_pending_best_update_state.pending_best_timer_stop")
                logger.debug("Failed to stop pending-best timer during worker completion cleanup: %s", exc, exc_info=True)
        self._pending_best_payload = None
        self._pending_best_worker = None

    @QtCore.Slot(int, str)
    def _dispatch_fit_worker_progress(self, percent: int, message: str) -> None:
        self._on_worker_progress(percent, message, worker=self._fit_worker_sender())

    @QtCore.Slot(dict)
    def _dispatch_fit_worker_best_update(self, payload: Dict[str, Any]) -> None:
        self._handle_global_best_update(payload, worker=self._fit_worker_sender())

    @QtCore.Slot(dict)
    def _dispatch_fit_worker_finished(self, payload: Dict[str, Any]) -> None:
        worker = self._dispatch_fit_worker_target()
        current_worker = getattr(self, "_worker", None)
        if worker is not None and current_worker is None:
            self._disconnect_fit_worker_signals(worker)
            self._stop_finished_worker(worker)
            self._schedule_worker_cleanup(worker)
            return
        if worker is not None and current_worker is not None and worker is not current_worker:
            self._disconnect_fit_worker_signals(worker)
            self._stop_finished_worker(worker)
            self._schedule_worker_cleanup(worker)
            return
        old_worker = current_worker if current_worker is not None else worker
        self._disconnect_fit_worker_signals(old_worker)
        if getattr(self, "_worker", None) is old_worker:
            self._worker = None
        self._stop_finished_worker(old_worker)
        try:
            self._handle_global_fit_complete(payload, worker=old_worker)
        finally:
            self._schedule_worker_cleanup(old_worker)

    @QtCore.Slot(object)
    def _dispatch_fit_worker_error(self, error: object) -> None:
        worker = self._dispatch_fit_worker_target()
        current_worker = getattr(self, "_worker", None)
        if worker is not None and current_worker is None:
            self._disconnect_fit_worker_signals(worker)
            self._stop_finished_worker(worker)
            self._schedule_worker_cleanup(worker)
            return
        if worker is not None and current_worker is not None and worker is not current_worker:
            self._disconnect_fit_worker_signals(worker)
            self._stop_finished_worker(worker)
            self._schedule_worker_cleanup(worker)
            return
        old_worker = current_worker if current_worker is not None else worker
        self._disconnect_fit_worker_signals(old_worker)
        if getattr(self, "_worker", None) is old_worker:
            self._worker = None
        self._stop_finished_worker(old_worker)
        try:
            self._on_worker_error(error, worker=old_worker)
        finally:
            self._schedule_worker_cleanup(old_worker)

    def _is_active_worker_callback(self, worker: Optional[QtCore.QThread]) -> bool:
        if self._fit_run_state_owner.active_run_superseded:
            return False
        if worker is None:
            return True
        return getattr(self, "_worker", None) is worker

    def _terminal_payload_matches_active_run(self, payload: object, *, worker: Optional[QtCore.QThread]) -> bool:
        if self._fit_run_state_owner.active_run_superseded:
            return False
        current_worker = getattr(self, "_worker", None)
        if worker is not None and current_worker is not None and current_worker is not worker:
            return False
        payload_hash = ""
        if isinstance(payload, Mapping):
            payload_hash = str(payload.get("run_stamp_hash") or "")
        active_hash = self._fit_run_state_owner.active_run_stamp_hash
        if payload_hash and active_hash and payload_hash != active_hash:
            return False
        return True

    def _on_worker_progress(self, percent: int, message: str, *, worker: Optional[QtCore.QThread] = None) -> None:
        if not self._is_active_worker_callback(worker):
            return
        if self._closing:
            return
        self._progress_bar.setValue(percent)
        self._status_label.setText(message)

    def _handle_global_fit_complete(
        self,
        payload: Dict[str, Any],
        *,
        worker: Optional[QtCore.QThread] = None,
    ) -> None:
        if not self._terminal_payload_matches_active_run(payload, worker=worker):
            return
        self._clear_pending_best_update_state()
        if self._closing:
            return
        result: GlobalFitResult = payload.get("result")
        if result is None:
            self._clear_failed_run_visual_state(None)
            self._status_label.setText("Global fit failed")
            return
        self._last_result = result
        self._best_cost = None
        severity, title, text = self._global_fit_completion_dialog_spec(result)
        if self._results_rebuild_pending:
            self._results_rebuild_pending = False
            if severity != "fail":
                self._run_results_tab.rebuild_subtabs(
                    self._dataset_entries,
                    self._results_fit_targets_by_dataset(),
                )
        if severity == "fail":
            self._clear_failed_run_visual_state(result)
            self._set_running_state(False)
        else:
            self._params_ics_tab.push_fit_results(
                dict(result.shared_params),
                {k: dict(v) for k, v in (result.dataset_params or {}).items()},
            )
        if severity == "fail":
            pass
        else:
            self._run_results_tab.push_final_result(result, self._dataset_entries)
            self._update_dataset_views_from_global(result)
        if severity == "fail":
            pass
        else:
            self._update_statistics(self._build_global_stats(result))
        self._refresh_project_apply_controls(prefer_broadest=True)
        if severity == "fail":
            pass
        else:
            self._latest_model_series = {k: dict(v) for k, v in (result.model_series or {}).items()}
            self._latest_dataset_stats = {
                info.dataset_id: {"chi_squared": float(info.chi_squared), "r_squared": float(info.r_squared)}
                for info in (result.dataset_info or [])
            }
            self._fit_run_state_owner.clear_active_dataset_ids()
            self._pre_run_parameter_state = None
            self._pre_run_staged_dataset_params = None
        completion = getattr(result, "completion", None)
        rendered_detail_sections: list[str] = []
        if completion is not None:
            for section in completion.detail_sections:
                detail_text = simulation_failure_detail_text(section.failure)
                if not detail_text:
                    detail_text = " ".join(
                        part.strip()
                        for part in str(simulation_failure_user_message(section.failure)).splitlines()
                        if part.strip()
                    )
                if not detail_text:
                    continue
                dataset_id = section.dataset_id
                if dataset_id is None:
                    rendered_detail_sections.append(detail_text)
                    continue
                label = self._dataset_label_for_id(str(dataset_id))
                rendered_detail_sections.append(f"{label}\n{detail_text}")
        combined_detail = "\n\n---\n\n".join(rendered_detail_sections)
        failure_message = str(result.message or "Unknown error")
        if severity == "ok":
            status = "Global fit complete"
        elif severity == "warn":
            status = "Global fit complete (warnings)"
            logger.warning("Global fit completed with warnings: %s", failure_message)
        else:
            status = (
                failure_message
                if failure_message.lower().startswith("global fit failed")
                else f"Global fit failed: {failure_message}"
            )
            logger.warning("%s", status)
        if combined_detail:
            logger.warning("%s", combined_detail)

        self._status_label.setText(status)
        if severity != "fail":
            self._set_running_state(False)
            self._fit_runtime_preparation_owner.schedule_refresh()
        if self.isVisible() and not self._closing:
            if severity == "ok":
                icon = QtWidgets.QMessageBox.Icon.Information
            else:
                icon = QtWidgets.QMessageBox.Icon.Warning
            dialog = QtWidgets.QMessageBox(self)
            dialog.setIcon(icon)
            dialog.setWindowTitle(title)
            dialog.setText(text)
            dialog.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Ok)
            dialog.setDetailedText(combined_detail)
            dialog.exec()

    @staticmethod
    def _failure_remediation_text(remediation: str) -> str:
        messages = {
            "x_axis_mapping": "Fix X axis / mapping and adjust t_min/t_max, then run again.",
            "nonfinite_metrics": "Inspect the fit objective and inputs for non-finite values, then run again.",
            "preparation": "Fix the preparation or parameter algebra error, then run again.",
            "generic_retry": "Fix the reported failure, then run again.",
        }
        return messages.get(str(remediation), messages["generic_retry"])

    def _capture_failed_fit_restore_baseline(self) -> None:
        self._pre_run_parameter_state = self._params_ics_tab.get_parameter_state()
        self._pre_run_staged_dataset_params = self._params_ics_tab.get_staged_dataset_params()

    def _clear_failed_fit_apply_state(self) -> None:
        self._params_ics_tab._restore_failed_fit_state(
            self._pre_run_parameter_state,
            self._pre_run_staged_dataset_params,
        )

    def _result_dataset_ids(self, result: Optional[GlobalFitResult]) -> List[str]:
        if result is None:
            return []
        ordered: List[str] = []
        seen: set[str] = set()

        def _add(dataset_id: object) -> None:
            ds_id = str(dataset_id or "").strip()
            if not ds_id or ds_id in seen:
                return
            seen.add(ds_id)
            ordered.append(ds_id)

        for info in getattr(result, "dataset_info", None) or []:
            _add(getattr(info, "dataset_id", None))
        for dataset_id in (getattr(result, "model_series", None) or {}).keys():
            _add(dataset_id)

        completion = getattr(result, "completion", None)
        if completion is not None:
            for dataset_id in getattr(completion, "dataset_failures", {}).keys():
                _add(dataset_id)
            for dataset_id in getattr(completion, "dataset_warnings", {}).keys():
                _add(dataset_id)
            for section in getattr(completion, "detail_sections", []) or []:
                _add(getattr(section, "dataset_id", None))

        return ordered

    def _active_or_result_dataset_ids(self, result: Optional[GlobalFitResult] = None) -> List[str]:
        active_dataset_ids = self._fit_run_state_owner.active_dataset_ids
        if active_dataset_ids:
            return list(active_dataset_ids)
        return self._result_dataset_ids(result)

    def _clear_failed_dataset_manager_fit_state(self, dataset_ids: Sequence[str]) -> None:
        if not self._dataset_manager:
            return
        normalized_ids = [str(dataset_id).strip() for dataset_id in dataset_ids if str(dataset_id).strip()]
        if not normalized_ids:
            return
        self._dataset_manager.sync_fit_result_views({}, dataset_stats={}, dataset_ids=normalized_ids)

    def _disable_results_summary_button(self) -> None:
        self._results_summary_button.setEnabled(False)

    def _clear_failed_run_visual_state(self, result: Optional[GlobalFitResult] = None) -> None:
        self._clear_failed_fit_apply_state()
        self._clear_failed_dataset_manager_fit_state(self._active_or_result_dataset_ids(result))
        self._run_results_tab._clear_failed_run_state(
            self._dataset_entries,
            self._results_fit_targets_by_dataset(),
        )
        self._disable_results_summary_button()
        self._refresh_project_apply_controls(prefer_broadest=True)
        self._latest_model_series = {}
        self._latest_dataset_stats = {}
        self._latest_plot_model_series = {}
        self._latest_plot_model_x = {}
        self._fit_run_state_owner.clear_active_dataset_ids()
        self._pre_run_parameter_state = None
        self._pre_run_staged_dataset_params = None
        self._set_running_state(False)

    def _failure_summary_lines(self, diagnostic: Any, *, label: str | None = None) -> list[str]:
        raw_message = simulation_failure_user_message(diagnostic.failure)
        compact_message = " ".join(part.strip() for part in str(raw_message).splitlines() if part.strip())
        summary_line = compact_message or "Global fit failed."
        prefix = f"- {label}: {summary_line}" if label else f"- {summary_line}"
        return [prefix, f"  {self._failure_remediation_text(diagnostic.remediation)}"]

    def _global_fit_completion_dialog_spec(self, result: GlobalFitResult) -> tuple[str, str, str]:
        """Return (severity, title, text) for the completion dialog."""
        chi_sq = float(getattr(result, "global_chi_squared", float("nan")))
        completion = getattr(result, "completion", None)
        if completion is None:
            return (
                "fail",
                "Global Fit Failed",
                "Global fit failed.\n\n- Missing completion metadata.",
            )
        status = completion.status
        dataset_failures = dict(completion.dataset_failures)
        warnings = dict(completion.dataset_warnings)
        optimizer_diagnostic = completion.optimizer_diagnostic
        nonfinite_chi = bool(completion.nonfinite_metrics)
        converged = bool(completion.optimizer_converged)

        if status == "fail":
            lines: List[str] = []
            if dataset_failures:
                for ds_id, diagnostic in sorted(dataset_failures.items(), key=lambda kv: str(kv[0])):
                    label = self._dataset_label_for_id(str(ds_id))
                    lines.extend(self._failure_summary_lines(diagnostic, label=label))
            elif optimizer_diagnostic is not None:
                lines.extend(self._failure_summary_lines(optimizer_diagnostic))
            else:
                lines.append("- Global fit failed before replay completed.")

            if nonfinite_chi:
                lines.append("")
                lines.append("Final χ² is non-finite; results are invalid.")

            return "fail", "Global Fit Failed", "Global fit failed.\n\n" + "\n".join(lines)

        param_section = self._format_params_for_completion_dialog(result)

        if status == "ok":
            text_parts = [f"Final Chi-Squared (χ²): {chi_sq:.6g}"]
            if param_section:
                text_parts.append("")
                text_parts.append(param_section)
            return "ok", "Optimization Complete", "\n".join(text_parts)

        warn_lines: List[str] = []
        if not converged:
            warn_lines.append("- Optimizer did not report convergence; results may be suboptimal.")
        if optimizer_diagnostic is not None:
            msg = simulation_failure_user_message(optimizer_diagnostic.failure)
            compact_message = " ".join(part.strip() for part in str(msg).splitlines() if part.strip())
            if compact_message:
                warn_lines.append(f"- {compact_message}")
        if warnings:
            for ds_id, msg in sorted(warnings.items(), key=lambda kv: str(kv[0])):
                label = self._dataset_label_for_id(str(ds_id))
                first_line = str(msg).strip().splitlines()[0] if str(msg).strip() else "Warning"
                warn_lines.append(f"- {label}: {first_line}")

        if warn_lines:
            text = [f"Final Chi-Squared (χ²): {chi_sq:.6g}"]
            if param_section:
                text.append("")
                text.append(param_section)
            text.extend(["", "Warnings:"])
            text.extend(warn_lines)
            return "warn", "Optimization Complete (Warnings)", "\n".join(text)
        return "warn", "Optimization Complete (Warnings)", f"Final Chi-Squared (χ²): {chi_sq:.6g}"

    def _format_params_for_completion_dialog(self, result: GlobalFitResult) -> str:
        """Format fitted parameters as compact text for the completion dialog."""
        shared = dict(getattr(result, "shared_params", None) or {})
        dataset_params = dict(getattr(result, "dataset_params", None) or {})
        all_items: list[tuple[str, float]] = sorted(
            shared.items(), key=lambda kv: str(kv[0])
        )
        for ds_id in sorted(dataset_params, key=str):
            ds_vals = dataset_params[ds_id]
            if not isinstance(ds_vals, dict) or not ds_vals:
                continue
            label = self._dataset_label_for_id(str(ds_id))
            for name, value in sorted(ds_vals.items(), key=lambda kv: str(kv[0])):
                all_items.append((f"{label}: {name}", float(value)))
        if not all_items:
            return ""
        max_shown = 10
        lines = ["Fitted Parameters:"]
        for name, value in all_items[:max_shown]:
            lines.append(f"  {name} = {value:.6g}")
        remaining = len(all_items) - max_shown
        if remaining > 0:
            lines.append(f"  ...and {remaining} more")
        return "\n".join(lines)

    def _handle_global_best_update(
        self,
        payload: Dict[str, Any],
        *,
        worker: Optional[QtCore.QThread] = None,
    ) -> None:
        """Live best-so-far updates during global fitting."""
        if self._closing:
            return
        if not self._is_active_worker_callback(worker):
            return
        try:
            cost = float(payload.get("cost"))
        except Exception:
            cost = None
        self._best_cost = cost

        tracker_panel = getattr(self._run_results_tab, "_tracker_panel", None)
        if tracker_panel is not None and any(key in payload for key in ("iteration", "cost", "shared_params")):
            tracker_panel.update_from_best(payload)

        shared_params = payload.get("shared_params") or {}
        dataset_params = payload.get("dataset_params") or {}
        if isinstance(shared_params, dict):
            ds_params = {str(ds_id): dict(pm) for ds_id, pm in dataset_params.items() if isinstance(pm, dict)} if isinstance(dataset_params, dict) else None
            self._params_ics_tab.push_best_update(
                {str(k): float(v) for k, v in shared_params.items()},
                ds_params,
            )

        model_series = payload.get("model_series")
        plot_model_series = payload.get("plot_model_series")
        plot_model_x = payload.get("plot_model_x")
        dataset_stats = payload.get("dataset_stats")
        has_plot_data = any(
            value is not None
            for value in (
                model_series,
                plot_model_series,
                plot_model_x,
                dataset_stats,
            )
        )
        running = bool(self._worker and hasattr(self._worker, "isRunning") and self._worker.isRunning())
        self._refresh_project_apply_controls(prefer_broadest=True, running=running)
        if not has_plot_data:
            if not running:
                self._params_ics_tab.repaint_parameter_table()
            return

        if isinstance(model_series, dict):
            self._latest_model_series = {k: dict(v) for k, v in model_series.items() if isinstance(v, dict)}
        else:
            self._latest_model_series = {}
        if isinstance(dataset_stats, dict):
            self._latest_dataset_stats = {
                str(ds_id): dict(stats) for ds_id, stats in dataset_stats.items() if isinstance(stats, dict)
            }
        else:
            self._latest_dataset_stats = {}
        if isinstance(plot_model_series, dict):
            self._latest_plot_model_series = {k: dict(v) for k, v in plot_model_series.items() if isinstance(v, dict)}
        else:
            self._latest_plot_model_series = {}
        if isinstance(plot_model_x, dict):
            self._latest_plot_model_x = {str(k): np.asarray(v, dtype=float).reshape(-1) for k, v in plot_model_x.items()}
        else:
            self._latest_plot_model_x = {}
        if not running:
            self._run_results_tab.push_live_update(payload, update_tracker=False)
            self._params_ics_tab.repaint_parameter_table()
            return
        self._pending_best_payload = dict(payload)
        self._pending_best_worker = worker
        if not self._pending_best_timer.isActive():
            self._pending_best_timer.start()

    def _apply_pending_best_update(self) -> None:
        if self._closing:
            self._pending_best_payload = None
            self._pending_best_worker = None
            return
        payload = self._pending_best_payload
        self._pending_best_payload = None
        worker = self._pending_best_worker
        self._pending_best_worker = None
        if not isinstance(payload, dict):
            return
        if not self._is_active_worker_callback(worker):
            return
        self._run_results_tab.push_live_update(payload, update_tracker=False)

    def _staged_initial_condition_parameters(self) -> Dict[str, Dict[str, float]]:
        staged: Dict[str, Dict[str, float]] = {}
        for dataset_id, param_map in (self._params_ics_tab.get_staged_dataset_params() or {}).items():
            if not isinstance(param_map, dict):
                continue
            updates: Dict[str, float] = {}
            for key, value in param_map.items():
                key_str = str(key)
                if not key_str.startswith(INITIAL_PREFIX):
                    continue
                try:
                    updates[key_str] = float(value)
                except (TypeError, ValueError):
                    continue
            if updates:
                staged[str(dataset_id)] = updates
        return staged

    def _available_project_apply_scopes(self) -> set[str]:
        scopes: set[str] = set()
        has_parameters = bool(self._params_ics_tab.get_last_fit_params())
        has_initial_conditions = bool(self._staged_initial_condition_parameters())
        if has_parameters:
            scopes.add(_PROJECT_APPLY_SCOPE_PARAMETERS)
        if has_initial_conditions:
            scopes.add(_PROJECT_APPLY_SCOPE_INITIAL_CONDITIONS)
        if has_parameters and has_initial_conditions:
            scopes.add(_PROJECT_APPLY_SCOPE_BOTH)
        return scopes

    @staticmethod
    def _preferred_project_apply_scope(scopes: set[str]) -> Optional[str]:
        for scope in (
            _PROJECT_APPLY_SCOPE_BOTH,
            _PROJECT_APPLY_SCOPE_PARAMETERS,
            _PROJECT_APPLY_SCOPE_INITIAL_CONDITIONS,
        ):
            if scope in scopes:
                return scope
        return None

    def _selected_project_apply_scope(self) -> Optional[str]:
        combo = getattr(self, "_apply_scope_combo", None)
        if combo is None:
            return None
        current_data = combo.currentData()
        if isinstance(current_data, str) and current_data:
            return current_data
        current_label = str(combo.currentText() or "")
        for label, scope in _PROJECT_APPLY_OPTIONS:
            if label == current_label:
                return scope
        return None

    def _refresh_project_apply_controls(
        self,
        *,
        prefer_broadest: bool = False,
        running: Optional[bool] = None,
    ) -> None:
        combo = getattr(self, "_apply_scope_combo", None)
        button = getattr(self, "_apply_to_project_button", None)
        if combo is None or button is None:
            return
        if running is None:
            worker = getattr(self, "_worker", None)
            running = bool(worker and hasattr(worker, "isRunning") and worker.isRunning())
        scopes = self._available_project_apply_scopes()
        model = combo.model()
        for index, (_label, scope) in enumerate(_PROJECT_APPLY_OPTIONS):
            item = model.item(index) if hasattr(model, "item") else None
            if item is not None:
                item.setEnabled(scope in scopes)
        selected_scope = self._selected_project_apply_scope()
        if prefer_broadest or selected_scope not in scopes:
            selected_scope = self._preferred_project_apply_scope(scopes)
            if selected_scope is not None:
                combo_index = combo.findData(selected_scope)
                if combo_index >= 0:
                    combo.setCurrentIndex(combo_index)
        can_dispatch = bool(self._project_apply_callback or self._apply_callback or self._dataset_settings_updater)
        combo.setEnabled(bool(scopes) and not bool(running))
        button.setEnabled(bool(scopes) and can_dispatch and not bool(running))

    def _mirror_staged_initial_condition_values(self) -> int:
        return self._params_ics_tab.mirror_staged_ic_values()

    def _apply_dataset_initials_via_updater(self) -> int:
        if not self._dataset_settings_updater:
            return 0
        self._mirror_staged_initial_condition_values()
        total_updates = 0
        for dataset_id, param_map in self._staged_initial_condition_parameters().items():
            updates: Dict[str, float] = {}
            for key, value in param_map.items():
                species = str(key)[len(INITIAL_PREFIX):]
                updates[species] = float(value)
            if updates:
                self._dataset_settings_updater(dataset_id, updates)
                total_updates += len(updates)
        return total_updates

    def _apply_to_project(self) -> None:
        scope = self._selected_project_apply_scope()
        if not scope or scope not in self._available_project_apply_scopes():
            return
        shared_params = {str(name): float(value) for name, value in (self._params_ics_tab.get_last_fit_params() or {}).items()}
        dataset_params = {
            str(dataset_id): dict(param_map)
            for dataset_id, param_map in (self._params_ics_tab.get_staged_dataset_params() or {}).items()
            if isinstance(param_map, dict)
        }
        apply_warning_text = ""
        try:
            if self._project_apply_callback:
                callback_result = self._project_apply_callback(scope, shared_params, dataset_params)
                if callback_result is False:
                    return
                if isinstance(callback_result, str):
                    apply_warning_text = str(callback_result).strip()
            else:
                if scope in {_PROJECT_APPLY_SCOPE_PARAMETERS, _PROJECT_APPLY_SCOPE_BOTH}:
                    if not (self._apply_callback and shared_params):
                        raise RuntimeError("No fitted parameter values are available to apply.")
                    self._apply_callback(shared_params)
                if scope in {_PROJECT_APPLY_SCOPE_INITIAL_CONDITIONS, _PROJECT_APPLY_SCOPE_BOTH}:
                    if not self._dataset_settings_updater:
                        raise RuntimeError("No dataset initial-condition updater is available.")
                    self._apply_dataset_initials_via_updater()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Apply to Project", str(exc))
            return
        if scope in {_PROJECT_APPLY_SCOPE_INITIAL_CONDITIONS, _PROJECT_APPLY_SCOPE_BOTH}:
            self._mirror_staged_initial_condition_values()
        if apply_warning_text:
            QtWidgets.QMessageBox.warning(self, "Apply to Project", apply_warning_text)
            return
        QtWidgets.QMessageBox.information(
            self,
            "Apply to Project",
            f"{self._apply_scope_combo.currentText() or 'Selected scope'} applied to project.",
        )

    def _update_global_plots_from_maps(
        self,
        model_series: Dict[str, Dict[str, np.ndarray]],
        dataset_stats: Dict[str, Dict[str, float]],
        model_x_by_dataset: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        self._run_results_tab.push_live_update(
            {
                "plot_model_series": model_series if model_x_by_dataset else {},
                "model_series": model_series if not model_x_by_dataset else {},
                "dataset_stats": dataset_stats,
                "plot_model_x": model_x_by_dataset or {},
            }
        )

    def _update_dataset_views_from_maps(
        self,
        model_series: Dict[str, Dict[str, np.ndarray]],
        dataset_stats: Dict[str, Dict[str, float]],
    ) -> None:
        if not self._dataset_manager:
            return
        self._dataset_manager.sync_fit_result_views(
            model_series,
            dataset_stats=dataset_stats,
        )

    def _update_dataset_views_from_global(self, result: GlobalFitResult) -> None:
        """Update dataset tabs/grid using global-fit output."""
        if not self._dataset_manager:
            return

        dataset_ids = self._result_dataset_ids(result)
        if not dataset_ids:
            return
        dataset_stats: Dict[str, Dict[str, float]] = {}
        info_map = {info.dataset_id: info for info in result.dataset_info}
        for dataset_id in dataset_ids:
            info = info_map.get(dataset_id)
            if info is not None:
                dataset_stats[dataset_id] = {
                    "chi_squared": info.chi_squared,
                    "r_squared": info.r_squared,
                }

        self._dataset_manager.sync_fit_result_views(
            result.model_series,
            dataset_stats=dataset_stats,
            dataset_ids=dataset_ids,
        )

    def _on_worker_error(self, error: object, *, worker: Optional[QtCore.QThread] = None) -> None:
        if not self._terminal_payload_matches_active_run(error, worker=worker):
            return
        self._clear_pending_best_update_state()
        if self._closing:
            return
        if self._results_rebuild_pending:
            self._results_rebuild_pending = False
        payload = coerce_simulation_failure(error)
        if str(payload.get("kind") or "") == "cancelled":
            self._set_running_state(False)
            self._status_label.setText("Fit cancelled")
            return
        self._clear_failed_run_visual_state()
        stack_trace = simulation_failure_detail_text(payload)
        has_stack_trace = bool(stack_trace)
        message = simulation_failure_user_message(payload)
        if message:
            logger.warning("Fitting worker reported error: %s", payload)
        if has_stack_trace:
            logger.warning("%s", stack_trace)
        if message and self.isVisible() and not self._closing:
            dialog = QtWidgets.QMessageBox(self)
            dialog.setIcon(QtWidgets.QMessageBox.Icon.Warning)
            dialog.setWindowTitle("Fitting")
            dialog.setText(message)
            dialog.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Ok)
            dialog.setDetailedText(stack_trace if has_stack_trace else "")
            dialog.exec()
        self._status_label.setText(message or "Fit error")

    # ------------------------------------------------------------------
    # Plot + stats helpers
    # ------------------------------------------------------------------
    def _refresh_plot_baselines(self) -> None:
        self._latest_model_series = {}
        self._latest_dataset_stats = {}
        self._latest_plot_model_series = {}
        self._latest_plot_model_x = {}
        self._request_rebuild_subtabs()

    def _update_global_plots(self, result: GlobalFitResult) -> None:
        self._run_results_tab.push_final_result(result, self._dataset_entries)

    def _build_global_stats(self, result: GlobalFitResult) -> Dict[str, float]:
        total_points = sum(info.n_points for info in result.dataset_info)
        series_count = sum(len(result.model_series.get(info.dataset_id, {})) for info in result.dataset_info)
        shared = len(result.shared_params)
        dataset_vars = sum(len(specs) for specs in self._active_variable_specs.values())
        params = shared + dataset_vars
        ssq = sum(np.sum(info.residuals**2) for info in result.dataset_info)
        weighted = float(np.sum(result.objective_residuals**2)) if result.objective_residuals is not None else ssq
        return {
            "Datasets": len(result.dataset_info),
            "Series": series_count or len(result.dataset_info),
            "Points": total_points,
            "Parameters": params,
            "DF": max(1, total_points - params),
            "SSQ": ssq,
            "Weighted SSQ": weighted,
            "-logL": 0.5 * weighted,
        }

    def _update_statistics(self, stats: Dict[str, Any]) -> None:
        self._run_results_tab.update_statistics(stats)

    # ------------------------------------------------------------------
    # Qt overrides
    # ------------------------------------------------------------------
    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # type: ignore[name-defined]
        self._closing = True
        if hasattr(self, "_pending_best_timer"):
            try:
                self._pending_best_timer.stop()
            except Exception as exc:
                self._best_effort_failures.add("closeEvent.pending_best_timer_stop")
                logger.debug("Failed to stop pending-best timer during closeEvent: %s", exc, exc_info=True)
        self._pending_best_payload = None
        self._hard_teardown_worker(reason="Cancelling...", disable_ui=True)
        if not self._fit_runtime_readiness.close(kill=True):
            self._fit_runtime_preparation_owner.request_close_after_prepare()
            try:
                self.setEnabled(False)
            except Exception:
                pass
            event.ignore()
            return
        event.accept()
        try:
            super().closeEvent(event)
        except Exception as exc:
            self._best_effort_failures.add("closeEvent.super_closeEvent")
            logger.exception("Error executing super().closeEvent in FittingWindow: %s", exc)
        try:
            self.deleteLater()
        except Exception as exc:
            self._best_effort_failures.add("closeEvent.deleteLater")
            logger.debug("Failed to schedule FittingWindow deleteLater during closeEvent: %s", exc, exc_info=True)
