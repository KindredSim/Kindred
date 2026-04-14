"""
Global fitting for chemical kinetics - fit single mechanism to multiple datasets.

This module provides functionality to fit a single reaction mechanism to multiple
experimental datasets simultaneously. Supports:
- Shared parameters (common across all datasets)
- Dataset-specific parameters (e.g., different initial conditions)
- Weighted residuals across datasets
- Comprehensive per-dataset and global statistics
"""

from __future__ import annotations

import logging
import math
import os
import pickle
import time
from concurrent.futures import FIRST_COMPLETED, Future, wait
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from kindred.core.analysis.parametric_alignment import (
    align_y_on_x_obs,
    align_y_on_x_obs_time_guided_penalized,
    is_non_monotone_in_sampled_window_error,
)
from kindred.core.analysis.fit_dataset_payload import (
    FitDatasetSpec,
    coerce_fit_dataset_specs,
)
from kindred.core.analysis.dataset_parameter_overrides import (
    coerce_fit_dataset_parameter_overrides,
    split_fit_dataset_parameter_overrides,
)
from kindred.core.exceptions import ErrorContext, FittingCancelled, FitSimulationError, SimulationCancelled
from kindred.core.fitting_optimization import fit_parameters
from kindred.core.fitting_evaluation import (
    FITTING_PARAM_ORIGIN_CONFIGURED_DATASET,
    FITTING_PARAM_ORIGIN_OPTIMIZER_DATASET,
    FITTING_PARAM_ORIGIN_OPTIMIZER_SHARED,
    SerialFittingEvaluator,
    coerce_fitting_series_evaluator,
    evaluate_fitting_series,
)
from kindred.core.fitting_process_pool import FittingProcessPool
from kindred.core.objective import ObjectiveContext, ObjectiveWrapper
from kindred.core.simulation_series_payload import coerce_simulation_series_payload
from kindred.core.analysis.x_mapping import normalize_x_mapping_mode

logger = logging.getLogger(__name__)

_MAX_PARALLEL_DATASET_LANES = 4

__all__ = [
    "GlobalFitResult",
    "DatasetFitInfo",
    "fit_global",
]


def _effective_fitting_process_workers(num_datasets: int) -> int:
    dataset_count = max(0, int(num_datasets))
    cpu = os.cpu_count()
    cpu_cap = max(1, int(cpu) - 1) if isinstance(cpu, int) and cpu > 0 else 1
    if dataset_count <= 0:
        return 1
    return int(min(dataset_count, cpu_cap, _MAX_PARALLEL_DATASET_LANES))

def _raise_if_fitting_cancelled(cancellation_check: Optional[Callable[[], bool]]) -> None:
    if cancellation_check is not None and cancellation_check():
        raise FittingCancelled()


def _raise_if_fitting_cancel_requested(cancellation_check: Optional[Callable[[], bool]]) -> None:
    if cancellation_check is None:
        return
    cancel_requested = getattr(cancellation_check, "_kindred_nonblocking_cancelled", cancellation_check)
    if bool(cancel_requested()):
        raise FittingCancelled()


class _DatasetLaneCancellation:
    def __init__(self, cancellation_check: Optional[Callable[[], bool]]) -> None:
        self._cancellation_check = cancellation_check
        self._internal_abort = False

    def request_internal_abort(self) -> None:
        self._internal_abort = True

    def internal_abort_requested(self) -> bool:
        return bool(self._internal_abort)

    def _kindred_nonblocking_cancelled(self) -> bool:
        if self._internal_abort:
            return True
        if self._cancellation_check is None:
            return False
        cancel_requested = getattr(
            self._cancellation_check,
            "_kindred_nonblocking_cancelled",
            self._cancellation_check,
        )
        return bool(cancel_requested())

    def __call__(self) -> bool:
        return self._kindred_nonblocking_cancelled()

    def raise_if_cancel_requested(self) -> None:
        if self._kindred_nonblocking_cancelled():
            raise FittingCancelled()

    def submission_paused_requested(self) -> bool:
        pause_requested = getattr(self._cancellation_check, "_kindred_nonblocking_paused", None)
        if not callable(pause_requested):
            self.raise_if_cancel_requested()
            return False
        self.raise_if_cancel_requested()
        return bool(pause_requested())

    def wait_if_submission_paused(self) -> None:
        if not self.submission_paused_requested():
            return
        wait_for_resume = getattr(self._cancellation_check, "_kindred_wait_for_resume", None)
        while self.submission_paused_requested():
            self.raise_if_cancel_requested()
            if callable(wait_for_resume):
                wait_for_resume(0.05)
            else:
                time.sleep(0.01)
        self.raise_if_cancel_requested()

@dataclass
class DatasetFitInfo:
    """Fit statistics for a single dataset within global fit."""
    dataset_id: str  # Identifier for this dataset
    r_squared: float  # Coefficient of determination
    chi_squared: float  # Reduced chi-squared
    rmse: float  # Root mean square error
    mae: float  # Mean absolute error
    residuals: np.ndarray  # Residuals for this dataset
    n_points: int  # Number of data points
    weight: float  # Weight used in global objective


@dataclass
class GlobalFitResult:
    """
    Results from global fitting across multiple datasets.

    Attributes
    ----------
    success : bool
        Whether optimization converged successfully
    shared_params : dict
        Optimized shared parameters {name: value}
    dataset_params : dict
        Dataset-specific parameters {dataset_id: {name: value}}
    uncertainties : dict, optional
        Parameter uncertainties {name: std_dev}
    global_chi_squared : float
        Weighted chi-squared across all datasets
    global_r_squared : float
        Overall R² (weighted by dataset sizes)
    dataset_info : list
        Per-dataset fit statistics (DatasetFitInfo objects)
    nfev : int
        Total number of function evaluations
    message : str
        Optimization termination message
    covariance : np.ndarray, optional
        Covariance matrix of all parameters
    """
    success: bool
    shared_params: Dict[str, float]
    dataset_params: Dict[str, Dict[str, float]]
    uncertainties: Optional[Dict[str, float]]
    global_chi_squared: float
    global_r_squared: float
    dataset_info: List[DatasetFitInfo]
    nfev: int
    message: str
    covariance: Optional[np.ndarray] = None
    objective_residuals: Optional[np.ndarray] = None
    model_series: Dict[str, Dict[str, np.ndarray]] = field(default_factory=dict)
    residual_series: Dict[str, Dict[str, np.ndarray]] = field(default_factory=dict)
    plot_model_x: Dict[str, np.ndarray] = field(default_factory=dict)
    plot_model_series: Dict[str, Dict[str, np.ndarray]] = field(default_factory=dict)
    dataset_errors: Dict[str, str] = field(default_factory=dict)
    dataset_warnings: Dict[str, str] = field(default_factory=dict)
    alignment_report: Dict[str, Dict[str, float]] = field(default_factory=dict)


def _normalize_input_datasets(datasets: List[object]) -> List[object]:
    return list(datasets)


def _build_dataset_payloads(datasets: List[object]) -> List[FitDatasetSpec]:
    return coerce_fit_dataset_specs(datasets)


def _normalize_weights(payloads: List[FitDatasetSpec], weights: Optional[Dict[str, float]]) -> Dict[str, float]:
    payload_ids = [str(payload.dataset_id) for payload in payloads]
    payload_id_set = set(payload_ids)
    provided_weights = {} if weights is None else {str(key): float(value) for key, value in dict(weights).items()}

    unknown_ids = sorted(set(provided_weights) - payload_id_set)
    if unknown_ids:
        raise ValueError(f"Unknown dataset weight IDs: {', '.join(unknown_ids)}")

    normalized: Dict[str, float] = {}
    total_weight = 0.0
    for payload in payloads:
        ds_id = str(payload.dataset_id)
        weight = provided_weights.get(ds_id, 1.0 / max(1, payload.point_count))
        if not np.isfinite(weight) or weight <= 0.0:
            raise ValueError(f"Dataset weight for '{ds_id}' must be finite and positive.")
        normalized[ds_id] = float(weight)
        total_weight += float(weight)

    if not np.isfinite(total_weight) or total_weight <= 0.0:
        raise ValueError("Sum of dataset weights must be positive.")
    n = max(1, len(payloads))
    return {ds_id: float(weight) * float(n) / total_weight for ds_id, weight in normalized.items()}


def _normalized_target_weight_multipliers(
    *,
    species_list: Sequence[str],
    target_weights: Optional[Dict[str, float]],
) -> Dict[str, float]:
    names = [str(name) for name in (species_list or []) if str(name).strip()]
    if not names:
        return {}

    raw_weights = dict(target_weights or {}) if isinstance(target_weights, dict) else {}
    cleaned: List[Tuple[str, float]] = []
    total_weight = 0.0
    for name in names:
        try:
            value = float(raw_weights.get(name, 1.0))
        except Exception:
            value = 1.0
        if not np.isfinite(value) or value <= 0.0:
            value = 1.0
        cleaned.append((name, float(value)))
        total_weight += float(value)

    if not np.isfinite(total_weight) or total_weight <= 0.0:
        return {name: 1.0 for name, _value in cleaned}

    count = float(len(cleaned))
    return {
        name: float(np.sqrt(count * float(value) / total_weight))
        for name, value in cleaned
    }


def _robust_span(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    try:
        lo = float(np.percentile(arr, 5))
        hi = float(np.percentile(arr, 95))
        span = hi - lo
    except Exception:
        span = float(np.max(arr) - np.min(arr))
    if not np.isfinite(span) or span < 0.0:
        return 0.0
    return span


def _extract_simulation_payload(simulation: object) -> Tuple[Optional[np.ndarray], Dict[str, np.ndarray]]:
    """Return (time array, species map) from simulation output."""
    payload = coerce_simulation_series_payload(simulation)
    t_sim = np.asarray(payload.t, dtype=float).reshape(-1)
    return (t_sim if t_sim.size else None), dict(payload.species)


def _align_series(series: np.ndarray, sim_time: Optional[np.ndarray], target_time: np.ndarray) -> np.ndarray:
    """Return model series aligned to the dataset time grid."""
    values = np.asarray(series, dtype=float).reshape(-1)
    if values.size == target_time.size:
        return values
    if sim_time is not None and values.size == sim_time.size:
        return np.interp(target_time, sim_time, values)
    raise ValueError(
        f"Simulation output has {values.size} points but dataset uses {target_time.size}.",
    )


@dataclass(frozen=True)
class _FitParameterLayout:
    param_names: List[str]
    shared_log10: Dict[str, bool]
    dataset_var_order: List[Tuple[str, str]]
    dataset_var_index: Dict[str, Dict[str, int]]
    dataset_var_log10: Dict[Tuple[str, str], bool]
    x0: np.ndarray
    lower: np.ndarray
    upper: np.ndarray

    def opt_param_keys(self) -> List[str]:
        keys: List[str] = list(self.param_names)
        keys.extend([f"{ds_id}::{param_name}" for (ds_id, param_name) in self.dataset_var_order])
        return keys

    def initial_params(self) -> Dict[str, float]:
        keys = self.opt_param_keys()
        return {key: float(self.x0[i]) for i, key in enumerate(keys)}

    def bounds_dict(self) -> Dict[str, Tuple[float, float]]:
        keys = self.opt_param_keys()
        return {key: (float(self.lower[i]), float(self.upper[i])) for i, key in enumerate(keys)}

    def shared_param_dict_from_vector(self, vector: np.ndarray) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for idx, name in enumerate(self.param_names):
            raw_val = float(vector[idx])
            if self.shared_log10.get(name):
                out[name] = float(10.0 ** raw_val)
            else:
                out[name] = raw_val
        return out

    def dataset_var_params_for_dataset(self, ds_id: str, vector: np.ndarray) -> Dict[str, float]:
        out: Dict[str, float] = {}
        var_map = self.dataset_var_index.get(ds_id)
        if not var_map:
            return out
        for var_name, idx in var_map.items():
            raw_val = float(vector[idx])
            if self.dataset_var_log10.get((ds_id, var_name)):
                out[var_name] = float(10.0 ** raw_val)
            else:
                out[var_name] = raw_val
        return out


def _build_parameter_layout(
    *,
    payloads: List[FitDatasetSpec],
    shared_params: Dict[str, float],
    dataset_variable_params: Dict[str, Dict[str, Dict[str, float]]],
    bounds: Optional[Dict[str, Tuple[float, float]]],
    log10_params: Optional[Dict[str, bool]],
) -> _FitParameterLayout:
    param_names = list(shared_params.keys())
    log10_params_norm = {str(k): bool(v) for k, v in (log10_params or {}).items()}

    shared_initials: List[float] = []
    shared_log10: Dict[str, bool] = {}
    for name in param_names:
        flag = bool(log10_params_norm.get(name, False))
        shared_log10[name] = flag
        value = float(shared_params[name])
        if flag:
            if not (value > 0.0):
                raise ValueError(f"log10 fitting requires positive initial value for '{name}'.")
            shared_initials.append(math.log10(value))
        else:
            shared_initials.append(value)

    dataset_var_order: List[Tuple[str, str]] = []
    dataset_var_index: Dict[str, Dict[str, int]] = {}
    dataset_var_inits: List[float] = []
    dataset_var_bounds: List[Tuple[float, float]] = []
    dataset_var_log10: Dict[Tuple[str, str], bool] = {}

    for payload in payloads:
        ds_id = payload.dataset_id
        specs = dataset_variable_params.get(ds_id, {})
        if not specs:
            continue
        dataset_var_index.setdefault(ds_id, {})
        for param_name, spec in specs.items():
            log10_flag = bool(spec.get("log10", False))
            dataset_var_order.append((ds_id, param_name))
            dataset_var_index[ds_id][param_name] = len(param_names) + len(dataset_var_inits)
            init_val = float(spec.get("initial", 0.0))
            min_val = float(spec.get("min", -np.inf))
            max_val = float(spec.get("max", np.inf))
            if log10_flag:
                if not (init_val > 0.0):
                    raise ValueError(f"log10 fitting requires positive initial value for '{ds_id}:{param_name}'.")
                if not (min_val > 0.0 and max_val > 0.0):
                    raise ValueError(f"log10 fitting requires positive bounds for '{ds_id}:{param_name}'.")
                if not (np.isfinite(min_val) and np.isfinite(max_val)):
                    raise ValueError(f"log10 fitting requires finite bounds for '{ds_id}:{param_name}'.")
                dataset_var_inits.append(math.log10(init_val))
                dataset_var_bounds.append((math.log10(min_val), math.log10(max_val)))
            else:
                dataset_var_inits.append(init_val)
                dataset_var_bounds.append((min_val, max_val))
            dataset_var_log10[(ds_id, param_name)] = log10_flag

    x0 = np.array(shared_initials + dataset_var_inits, dtype=float)

    shared_lower = []
    shared_upper = []
    for name in param_names:
        if bounds is not None:
            b = bounds.get(name, (-np.inf, np.inf))
        else:
            b = (-np.inf, np.inf)
        low = float(b[0])
        high = float(b[1])
        if shared_log10.get(name):
            if not (low > 0.0 and high > 0.0):
                raise ValueError(f"log10 fitting requires positive bounds for '{name}'.")
            if not (np.isfinite(low) and np.isfinite(high)):
                raise ValueError(f"log10 fitting requires finite bounds for '{name}'.")
            shared_lower.append(math.log10(low))
            shared_upper.append(math.log10(high))
        else:
            shared_lower.append(low)
            shared_upper.append(high)

    lower = np.array(shared_lower + [b[0] for b in dataset_var_bounds], dtype=float)
    upper = np.array(shared_upper + [b[1] for b in dataset_var_bounds], dtype=float)

    return _FitParameterLayout(
        param_names=param_names,
        shared_log10=shared_log10,
        dataset_var_order=dataset_var_order,
        dataset_var_index=dataset_var_index,
        dataset_var_log10=dataset_var_log10,
        x0=x0,
        lower=lower,
        upper=upper,
    )


class _ParametricXAligner:
    def __init__(
        self,
        *,
        mode: str,
        t_obs: np.ndarray,
        x_obs: np.ndarray,
        t_sim: Optional[np.ndarray],
        x_model: np.ndarray,
        dataset_label: str,
        x_name: str,
    ) -> None:
        mode_norm = normalize_x_mapping_mode(mode)
        if mode_norm not in {"auto", "monotone", "time_guided"}:
            raise FitSimulationError(
                f"Dataset '{dataset_label}': invalid x_mapping_mode '{mode}'. Expected auto, monotone, or time_guided."
            )
        self._mode = mode_norm
        self._use_penalized_mapping = bool(mode_norm == "time_guided")
        self._t_obs = t_obs
        self._x_obs = x_obs
        self._t_sim = t_sim
        self._x_model = np.asarray(x_model, dtype=float)
        self._dataset_label = str(dataset_label)
        self._x_name = str(x_name)
        self._mapping_t_star: Optional[np.ndarray] = None
        self._mapping_dx: Optional[np.ndarray] = None
        self._penalized_out = None

    @property
    def mapping_dx(self) -> Optional[np.ndarray]:
        if self._mapping_dx is not None:
            return self._mapping_dx
        if self._penalized_out is not None:
            try:
                return np.asarray(self._penalized_out.dx, dtype=float).reshape(-1)
            except Exception:
                return None
        return None

    @property
    def penalized_out(self):
        return self._penalized_out

    def align(self, *, y_model: np.ndarray, y_name: str, need_dx_penalty: bool) -> np.ndarray:
        if self._mode == "monotone":
            return align_y_on_x_obs(
                t_obs=self._t_obs,
                x_obs=self._x_obs,
                t_sim=self._t_sim,
                x_model=self._x_model,
                y_model=np.asarray(y_model, dtype=float),
                dataset_label=self._dataset_label,
                x_name=self._x_name,
                y_name=y_name,
            )

        if self._use_penalized_mapping:
            if self._mapping_t_star is None or self._mapping_dx is None:
                out = align_y_on_x_obs_time_guided_penalized(
                    t_obs=self._t_obs,
                    x_obs=self._x_obs,
                    t_sim=self._t_sim,
                    x_model=self._x_model,
                    y_model=np.asarray(y_model, dtype=float),
                    dataset_label=self._dataset_label,
                    x_name=self._x_name,
                    y_name=y_name,
                )
                self._penalized_out = out
                self._mapping_t_star = np.asarray(out.t_star, dtype=float).reshape(-1)
                self._mapping_dx = np.asarray(out.dx, dtype=float).reshape(-1)
                return np.asarray(out.y_aligned, dtype=float).reshape(-1)
            return np.interp(
                np.asarray(self._mapping_t_star, dtype=float).reshape(-1),
                np.asarray(self._t_sim, dtype=float).reshape(-1),
                np.asarray(y_model, dtype=float).reshape(-1),
            )

        # Auto: try monotone inversion; if non-monotone/out-of-range, fall back to penalized mapping.
        try:
            y_sim = align_y_on_x_obs(
                t_obs=self._t_obs,
                x_obs=self._x_obs,
                t_sim=self._t_sim,
                x_model=self._x_model,
                y_model=np.asarray(y_model, dtype=float),
                dataset_label=self._dataset_label,
                x_name=self._x_name,
                y_name=y_name,
            )
            if need_dx_penalty and self._mapping_dx is None:
                self._mapping_dx = np.zeros_like(self._x_obs, dtype=float).reshape(-1)
            return y_sim
        except FitSimulationError as exc:
            msg = str(exc).lower()
            if is_non_monotone_in_sampled_window_error(exc) or ("fall outside model range" in msg):
                self._use_penalized_mapping = True
                out = align_y_on_x_obs_time_guided_penalized(
                    t_obs=self._t_obs,
                    x_obs=self._x_obs,
                    t_sim=self._t_sim,
                    x_model=self._x_model,
                    y_model=np.asarray(y_model, dtype=float),
                    dataset_label=self._dataset_label,
                    x_name=self._x_name,
                    y_name=y_name,
                )
                self._penalized_out = out
                self._mapping_t_star = np.asarray(out.t_star, dtype=float).reshape(-1)
                self._mapping_dx = np.asarray(out.dx, dtype=float).reshape(-1)
                return np.asarray(out.y_aligned, dtype=float).reshape(-1)
            raise


@dataclass(frozen=True)
class _ObjectiveDatasetInput:
    index: int
    payload: FitDatasetSpec
    full_params: Dict[str, float]
    parameter_origins: Dict[str, str]
    failed_param_snapshot: Dict[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "full_params", dict(self.full_params))
        object.__setattr__(self, "parameter_origins", dict(self.parameter_origins))
        object.__setattr__(self, "failed_param_snapshot", dict(self.failed_param_snapshot))


@dataclass(frozen=True)
class _DatasetSimulationEvaluation:
    index: int
    sim_time: Optional[np.ndarray]
    sim_species: Dict[str, np.ndarray]
    error: Optional[BaseException] = None
    error_provenance: Optional[Dict[str, Any]] = None
    final_error_message: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "sim_species", dict(self.sim_species))
        if self.error_provenance is not None:
            object.__setattr__(self, "error_provenance", dict(self.error_provenance))


def _dataset_evaluation_is_fatal(result: _DatasetSimulationEvaluation) -> bool:
    return isinstance(result.error, FitSimulationError) and bool(
        getattr(result.error, "details", {}).get("fatal")
    )


def _evaluate_dataset_simulation(
    fit_evaluator,
    item: _ObjectiveDatasetInput,
    *,
    cancellation_check: Optional[Callable[[], bool]] = None,
    lane_cancellation: Optional[_DatasetLaneCancellation] = None,
) -> _DatasetSimulationEvaluation:
    ds_id = item.payload.dataset_id
    try:
        _raise_if_fitting_cancel_requested(cancellation_check)
        sim_result = evaluate_fitting_series(
            fit_evaluator,
            item.full_params,
            origins=item.parameter_origins,
            failed_params=item.failed_param_snapshot,
        )
        sim_time, sim_species = _extract_simulation_payload(sim_result)
        _raise_if_fitting_cancel_requested(cancellation_check)
        return _DatasetSimulationEvaluation(
            index=int(item.index),
            sim_time=sim_time,
            sim_species=sim_species,
        )
    except FitSimulationError as exc:
        return _DatasetSimulationEvaluation(
            index=int(item.index),
            sim_time=None,
            sim_species={},
            error=exc,
            error_provenance={"dataset": ds_id, "provenance": getattr(exc, "provenance", None)},
            final_error_message=str(exc),
        )
    except Exception as exc:
        if isinstance(exc, (FittingCancelled, SimulationCancelled)):
            raise FittingCancelled() from exc
        return _DatasetSimulationEvaluation(
            index=int(item.index),
            sim_time=None,
            sim_species={},
            error=FitSimulationError(
                f"Simulation failed for dataset '{ds_id}': {exc}",
                failed_params=item.failed_param_snapshot,
            ),
            error_provenance={"dataset": ds_id},
            final_error_message=str(exc),
        )


def _error_context_from_process_payload(payload: Optional[Dict[str, Any]]) -> Optional[ErrorContext]:
    if not isinstance(payload, dict):
        return None
    context = ErrorContext(
        line=payload.get("line"),
        col=payload.get("col"),
        line_text=payload.get("line_text"),
        file_path=payload.get("file_path"),
        stack_trace=payload.get("stack_trace"),
    )
    if (
        context.line is None
        and context.col is None
        and context.line_text is None
        and context.file_path is None
        and context.stack_trace is None
    ):
        return None
    return context


def _error_from_process_payload(
    payload: Optional[Dict[str, Any]],
    *,
    failed_params: Optional[Dict[str, float]],
) -> BaseException:
    if not isinstance(payload, dict):
        return RuntimeError("Missing process worker error payload.")
    kind = str(payload.get("kind") or "generic")
    message = str(payload.get("message") or "Fitting worker failed.")
    code = payload.get("code")
    details = dict(payload.get("details") or {})
    context = _error_context_from_process_payload(payload.get("context"))
    if kind == "fit_simulation":
        marshalled_failed_params = dict(payload.get("failed_params") or {})
        return FitSimulationError(
            message,
            failed_params=marshalled_failed_params or dict(failed_params or {}) or None,
            code=code,
            details=details,
            context=context,
        )
    if kind == "fitting_cancelled":
        return FittingCancelled(message, code=code, details=details, context=context)
    if kind == "simulation_cancelled":
        return SimulationCancelled(message, code=code, details=details, context=context)
    return RuntimeError(message)


def _dataset_evaluation_from_process_payload(
    payload: Dict[str, Any],
    item: _ObjectiveDatasetInput,
) -> _DatasetSimulationEvaluation:
    if bool(payload.get("ok")):
        sim_time, sim_species = _extract_simulation_payload(payload.get("series_payload"))
        return _DatasetSimulationEvaluation(
            index=int(payload.get("index", item.index)),
            sim_time=sim_time,
            sim_species=sim_species,
        )
    error = _error_from_process_payload(
        payload.get("error"),
        failed_params=item.failed_param_snapshot,
    )
    return _DatasetSimulationEvaluation(
        index=int(payload.get("index", item.index)),
        sim_time=None,
        sim_species={},
        error=error,
        error_provenance=dict(payload.get("error_provenance") or {"dataset": item.payload.dataset_id}),
        final_error_message=str(payload.get("final_error_message") or error),
    )


def _cancel_pending_dataset_evaluations(futures: Sequence[Future]) -> None:
    for future in futures:
        future.cancel()


def _run_parallel_dataset_lane_batch(
    items: Sequence[_ObjectiveDatasetInput],
    *,
    max_workers: int,
    lane_cancellation: _DatasetLaneCancellation,
    stop_on_fatal: bool,
    submit_item: Callable[[int, _ObjectiveDatasetInput], Future],
    resolve_result: Callable[[Future, _ObjectiveDatasetInput], _DatasetSimulationEvaluation],
    abort_backend: Callable[[], None],
    wait_timeout: Optional[float] = None,
    should_suppress_exception: Optional[Callable[[Exception], bool]] = None,
) -> List[_DatasetSimulationEvaluation]:
    results: List[Optional[_DatasetSimulationEvaluation]] = [None] * len(items)
    future_to_index: Dict[Future, int] = {}
    future_to_slot: Dict[Future, int] = {}
    available_slots: List[int] = list(range(max_workers))
    next_pos = 0
    stop_submitting = False
    batch_aborted = False

    def abort_active_batch() -> None:
        nonlocal batch_aborted
        if batch_aborted:
            return
        batch_aborted = True
        lane_cancellation.request_internal_abort()
        _cancel_pending_dataset_evaluations(tuple(future_to_index))
        abort_backend()

    def submit_next_dataset_lane(slot: int) -> None:
        nonlocal next_pos
        lane_cancellation.wait_if_submission_paused()
        pos = next_pos
        future = submit_item(int(slot), items[pos])
        future_to_index[future] = pos
        future_to_slot[future] = int(slot)
        next_pos += 1

    try:
        while next_pos < len(items) and available_slots:
            submit_next_dataset_lane(available_slots.pop(0))

        while future_to_index:
            lane_cancellation.raise_if_cancel_requested()
            if wait_timeout is None:
                done, _pending = wait(tuple(future_to_index), return_when=FIRST_COMPLETED)
            else:
                done, _pending = wait(tuple(future_to_index), timeout=wait_timeout, return_when=FIRST_COMPLETED)
            if not done:
                continue
            for future in done:
                pos = future_to_index.pop(future)
                slot = future_to_slot.pop(future)
                available_slots.append(int(slot))
                if future.cancelled():
                    continue
                try:
                    result = resolve_result(future, items[pos])
                    results[pos] = result
                    lane_cancellation.raise_if_cancel_requested()
                except Exception as exc:
                    if should_suppress_exception is not None and should_suppress_exception(exc):
                        continue
                    abort_active_batch()
                    raise
                if stop_on_fatal and _dataset_evaluation_is_fatal(result):
                    stop_submitting = True
                    abort_active_batch()
                    future_to_index.clear()
                    future_to_slot.clear()
                    break
            available_slots.sort()
            while not stop_submitting and next_pos < len(items) and available_slots:
                if lane_cancellation.submission_paused_requested() and future_to_index:
                    break
                submit_next_dataset_lane(available_slots.pop(0))
    except FittingCancelled:
        abort_active_batch()
        raise
    except Exception:
        abort_active_batch()
        raise

    return [result for result in results if result is not None]


def _evaluate_dataset_simulations(
    fit_evaluator,
    items: Sequence[_ObjectiveDatasetInput],
    *,
    cancellation_check: Optional[Callable[[], bool]] = None,
    stop_on_fatal: bool = True,
    process_pool: Optional[FittingProcessPool] = None,
) -> List[_DatasetSimulationEvaluation]:
    if not items:
        return []
    if process_pool is None or len(items) == 1:
        return _evaluate_dataset_simulations_serial(
            fit_evaluator,
            items,
            cancellation_check=cancellation_check,
            stop_on_fatal=stop_on_fatal,
        )

    lane_cancellation = _DatasetLaneCancellation(cancellation_check)
    return _run_parallel_dataset_lane_batch(
        items,
        max_workers=min(len(items), process_pool.max_workers),
        lane_cancellation=lane_cancellation,
        stop_on_fatal=stop_on_fatal,
        submit_item=lambda _slot, item: process_pool.submit(item),
        resolve_result=lambda future, item: _dataset_evaluation_from_process_payload(future.result(), item),
        abort_backend=process_pool.cancel,
        wait_timeout=0.05,
    )


def _evaluate_dataset_simulations_serial(
    fit_evaluator,
    items: Sequence[_ObjectiveDatasetInput],
    *,
    cancellation_check: Optional[Callable[[], bool]],
    stop_on_fatal: bool,
) -> List[_DatasetSimulationEvaluation]:
    results = []
    for item in items:
        _raise_if_fitting_cancelled(cancellation_check)
        result = _evaluate_dataset_simulation(
            fit_evaluator,
            item,
            cancellation_check=cancellation_check,
        )
        results.append(result)
        if stop_on_fatal and _dataset_evaluation_is_fatal(result):
            break
    return results


class _GlobalFitObjective:
    def __init__(
        self,
        *,
        fit_evaluator,
        payloads: List[FitDatasetSpec],
        shared_params: Dict[str, float],
        dataset_params: Dict[str, Dict[str, float]],
        weights: Dict[str, float],
        layout: _FitParameterLayout,
        penalty_value: float,
        ctx: ObjectiveContext,
        progress_callback: Optional[Callable[[int, float, Dict[str, float]], None]],
        cancellation_check: Optional[Callable[[], bool]],
        process_pool: Optional[FittingProcessPool] = None,
    ) -> None:
        self._fit_evaluator = fit_evaluator
        self._payloads = payloads
        self._shared_params = shared_params
        self._dataset_params = dataset_params
        self._weights = weights
        self._layout = layout
        self._penalty_value = float(penalty_value)
        self._ctx = ctx
        self._progress_callback = progress_callback
        self._cancellation_check = cancellation_check
        self._process_pool = process_pool

        self._iteration = 0
        self._best_cost = float("inf")
        self._warned_objective_keys: set[tuple[str, str, str]] = set()

    @staticmethod
    def _format_params(params_map: Dict[str, float]) -> Dict[str, float]:
        def _try_float(value: object) -> Optional[float]:
            try:
                return float(value)  # type: ignore[arg-type]
            except (TypeError, ValueError, OverflowError):
                return None

        formatted: Dict[str, float] = {}
        for key, value in params_map.items():
            coerced = _try_float(value)
            if coerced is None:
                continue
            formatted[str(key)] = coerced
        if formatted:
            return formatted
        return {str(key): float(value) for key, value in params_map.items()}

    @staticmethod
    def _build_failed_param_snapshot(
        *,
        ds_id: str,
        shared_params: Dict[str, float],
        full_params: Dict[str, float],
    ) -> Dict[str, float]:
        snapshot = _GlobalFitObjective._format_params(shared_params)
        shared_keys = set(snapshot)
        for name, value in _GlobalFitObjective._format_params(full_params).items():
            if name in shared_keys or "::" in name:
                snapshot[name] = float(value)
            else:
                snapshot[f"{ds_id}::{name}"] = float(value)
        return snapshot

    def __call__(self, params: np.ndarray) -> np.ndarray:
        _raise_if_fitting_cancelled(self._cancellation_check)
        self._ctx.clear()

        param_dict = self._layout.shared_param_dict_from_vector(params)
        all_residuals: List[float] = []
        dataset_inputs: List[_ObjectiveDatasetInput] = []

        for index, payload in enumerate(self._payloads):
            _raise_if_fitting_cancel_requested(self._cancellation_check)

            ds_id = payload.dataset_id
            optimizer_dataset_params = self._layout.dataset_var_params_for_dataset(ds_id, params)
            raw_full_params: Dict[str, float] = {}
            parameter_origins: Dict[str, str] = {}
            for values, origin in (
                (param_dict, FITTING_PARAM_ORIGIN_OPTIMIZER_SHARED),
                (self._dataset_params.get(ds_id, {}), FITTING_PARAM_ORIGIN_CONFIGURED_DATASET),
                (optimizer_dataset_params, FITTING_PARAM_ORIGIN_OPTIMIZER_DATASET),
            ):
                for raw_name, raw_value in dict(values or {}).items():
                    name = str(raw_name or "").strip()
                    if not name:
                        continue
                    raw_full_params[name] = float(raw_value)
                    parameter_origins[name] = origin
            full_params = self._format_params(raw_full_params)
            parameter_origins = {
                name: parameter_origins[name]
                for name in full_params
                if name in parameter_origins
            }
            failed_param_snapshot = self._build_failed_param_snapshot(
                ds_id=ds_id,
                shared_params=param_dict,
                full_params=full_params,
            )
            dataset_inputs.append(
                _ObjectiveDatasetInput(
                    index=int(index),
                    payload=payload,
                    full_params=full_params,
                    parameter_origins=parameter_origins,
                    failed_param_snapshot=failed_param_snapshot,
                )
            )

        simulation_evaluations = {
            int(result.index): result
            for result in _evaluate_dataset_simulations(
                self._fit_evaluator,
                dataset_inputs,
                cancellation_check=self._cancellation_check,
                process_pool=self._process_pool,
            )
        }

        for evaluation in simulation_evaluations.values():
            if not _dataset_evaluation_is_fatal(evaluation):
                continue
            exc = evaluation.error
            if isinstance(exc, FitSimulationError):
                self._ctx.set_error(exc, evaluation.error_provenance)
                raise exc

        for item in dataset_inputs:
            _raise_if_fitting_cancel_requested(self._cancellation_check)

            payload = item.payload
            ds_id = payload.dataset_id
            species_list = payload.species_list
            y_matrix = payload.y_matrix
            t_exp = payload.t_exp
            x_name = payload.x_name
            x_obs = payload.x_obs if x_name != "t" else None
            x_mode = payload.x_mode
            weight = self._weights.get(ds_id, 1.0)
            target_weights = dict(getattr(payload, "target_weights", {}) or {})
            target_multipliers = _normalized_target_weight_multipliers(
                species_list=species_list,
                target_weights=target_weights,
            )
            failed_param_snapshot = item.failed_param_snapshot
            evaluation = simulation_evaluations.get(int(item.index))
            if evaluation is None:
                raise RuntimeError(f"Missing fitting simulation result for dataset '{ds_id}'.")

            if evaluation.error is not None:
                exc = evaluation.error
                if not isinstance(exc, FitSimulationError):
                    if isinstance(exc, (FittingCancelled, SimulationCancelled)):
                        raise FittingCancelled() from exc
                    exc = FitSimulationError(
                        f"Simulation failed for dataset '{ds_id}': {exc}",
                        failed_params=failed_param_snapshot,
                    )
                self._ctx.set_error(exc, evaluation.error_provenance)
                if bool(getattr(exc, "details", {}).get("fatal")):
                    raise exc
                key = (str(ds_id), "__simulation__", str(exc).splitlines()[0])
                if key not in self._warned_objective_keys:
                    self._warned_objective_keys.add(key)
                    logger.warning("Simulation failed for %s: %s", ds_id, exc)
                for idx in range(int(y_matrix.shape[0])):
                    species_name = str(species_list[idx])
                    target_weight = float(target_multipliers.get(species_name, 1.0))
                    y_exp = np.asarray(y_matrix[idx], dtype=float).reshape(-1)
                    all_residuals.extend(
                        (float(weight) * float(target_weight) * self._penalty_value) * np.ones_like(y_exp, dtype=float)
                    )
                if x_name != "t" and x_mode in ("auto", "time_guided"):
                    all_residuals.extend(
                        (float(weight) * self._penalty_value)
                        * np.ones_like(np.asarray(t_exp, dtype=float).reshape(-1), dtype=float)
                    )
                continue

            sim_time = evaluation.sim_time
            sim_species = evaluation.sim_species

            need_dx_penalty = bool(x_name != "t" and x_mode in ("auto", "time_guided"))
            dx_penalty_scale = 1.0
            if need_dx_penalty and x_obs is not None:
                try:
                    y_span = _robust_span(y_matrix)
                    x_span = _robust_span(x_obs)
                    if y_span > 0.0:
                        dx_penalty_scale = float(y_span / max(x_span, 1e-12))
                except Exception:
                    dx_penalty_scale = 1.0

            aligner: Optional[_ParametricXAligner] = None
            for idx, species_name in enumerate(species_list):
                _raise_if_fitting_cancelled(self._cancellation_check)
                y_exp = y_matrix[idx]
                target_weight = float(target_multipliers.get(str(species_name), 1.0))
                effective_weight = float(weight) * float(target_weight)
                model_series = sim_species.get(species_name)
                if model_series is None:
                    err = FitSimulationError(
                        f"Species '{species_name}' missing in simulation result for dataset '{ds_id}'",
                        failed_params=failed_param_snapshot,
                    )
                    self._ctx.set_error(err, {"dataset": ds_id})
                    key = (str(ds_id), str(species_name), "missing_species")
                    if key not in self._warned_objective_keys:
                        self._warned_objective_keys.add(key)
                        logger.warning("%s", err)
                    y_exp_arr = np.asarray(y_exp, dtype=float).reshape(-1)
                    all_residuals.extend((effective_weight * self._penalty_value) * np.ones_like(y_exp_arr, dtype=float))
                    continue

                try:
                    if x_name == "t":
                        y_sim = _align_series(model_series, sim_time, t_exp)
                    else:
                        x_model_series = sim_species.get(x_name)
                        if x_model_series is None:
                            raise FitSimulationError(
                                f"X series '{x_name}' missing in simulation result for dataset '{ds_id}'.",
                                failed_params=failed_param_snapshot,
                            )
                        if x_obs is None:
                            raise FitSimulationError(
                                f"Dataset '{ds_id}' is missing x_obs for X='{x_name}'.",
                                failed_params=failed_param_snapshot,
                            )
                        if aligner is None:
                            aligner = _ParametricXAligner(
                                mode=x_mode,
                                t_obs=t_exp,
                                x_obs=x_obs,
                                t_sim=sim_time,
                                x_model=np.asarray(x_model_series, dtype=float),
                                dataset_label=ds_id,
                                x_name=x_name,
                            )
                        y_sim = aligner.align(
                            y_model=np.asarray(model_series, dtype=float),
                            y_name=species_name,
                            need_dx_penalty=need_dx_penalty,
                        )
                except FitSimulationError as exc:
                    self._ctx.set_error(
                        FitSimulationError(str(exc), failed_params=failed_param_snapshot),
                        {"dataset": ds_id},
                    )
                    key = (str(ds_id), str(species_name), str(exc).splitlines()[0])
                    if key not in self._warned_objective_keys:
                        self._warned_objective_keys.add(key)
                        logger.warning("Alignment failed for %s:%s: %s", ds_id, species_name, exc)
                    y_exp_arr = np.asarray(y_exp, dtype=float).reshape(-1)
                    all_residuals.extend((effective_weight * self._penalty_value) * np.ones_like(y_exp_arr, dtype=float))
                    continue
                except Exception as exc:
                    if isinstance(exc, (FittingCancelled, SimulationCancelled)):
                        raise FittingCancelled() from exc
                    err = FitSimulationError(
                        f"Failed to align species '{species_name}' for dataset '{ds_id}': {exc}",
                        failed_params=failed_param_snapshot,
                    )
                    self._ctx.set_error(err, {"dataset": ds_id})
                    key = (str(ds_id), str(species_name), str(err).splitlines()[0])
                    if key not in self._warned_objective_keys:
                        self._warned_objective_keys.add(key)
                        logger.warning("%s", err)
                    y_exp_arr = np.asarray(y_exp, dtype=float).reshape(-1)
                    all_residuals.extend((effective_weight * self._penalty_value) * np.ones_like(y_exp_arr, dtype=float))
                    continue

                if not np.all(np.isfinite(y_sim)):
                    err = FitSimulationError(
                        f"Simulation produced non-finite values for dataset '{ds_id}', species '{species_name}'.",
                        failed_params=failed_param_snapshot,
                    )
                    self._ctx.set_error(err, {"dataset": ds_id})
                    key = (str(ds_id), str(species_name), "nonfinite_model")
                    if key not in self._warned_objective_keys:
                        self._warned_objective_keys.add(key)
                        logger.warning("%s", err)
                    y_exp_arr = np.asarray(y_exp, dtype=float).reshape(-1)
                    all_residuals.extend((effective_weight * self._penalty_value) * np.ones_like(y_exp_arr, dtype=float))
                    continue

                residual_vector = effective_weight * (
                    np.asarray(y_sim, dtype=float).reshape(-1) - np.asarray(y_exp, dtype=float).reshape(-1)
                )
                if not np.all(np.isfinite(residual_vector)):
                    err = FitSimulationError(
                        f"Residuals contained non-finite values for dataset '{ds_id}'.",
                        failed_params=failed_param_snapshot,
                    )
                    self._ctx.set_error(err, {"dataset": ds_id})
                    key = (str(ds_id), str(species_name), "nonfinite_residual")
                    if key not in self._warned_objective_keys:
                        self._warned_objective_keys.add(key)
                        logger.warning("%s", err)
                    y_exp_arr = np.asarray(y_exp, dtype=float).reshape(-1)
                    all_residuals.extend((effective_weight * self._penalty_value) * np.ones_like(y_exp_arr, dtype=float))
                    continue
                all_residuals.extend(np.asarray(residual_vector, dtype=float).reshape(-1))

            if need_dx_penalty:
                dx_block = None
                if aligner is not None and aligner.mapping_dx is not None:
                    dx_block = np.asarray(aligner.mapping_dx, dtype=float).reshape(-1)
                if (
                    dx_block is None
                    or dx_block.size != np.asarray(t_exp, dtype=float).reshape(-1).size
                    or not np.all(np.isfinite(dx_block))
                ):
                    all_residuals.extend(
                        (float(weight) * self._penalty_value)
                        * np.ones_like(np.asarray(t_exp, dtype=float).reshape(-1), dtype=float)
                    )
                else:
                    lam = 1.0
                    dx_resid = float(weight) * (lam * float(dx_penalty_scale) * dx_block)
                    if not np.all(np.isfinite(dx_resid)):
                        all_residuals.extend(
                            (float(weight) * self._penalty_value)
                            * np.ones_like(np.asarray(t_exp, dtype=float).reshape(-1), dtype=float)
                        )
                    else:
                        all_residuals.extend(np.asarray(dx_resid, dtype=float).reshape(-1))

        residuals = np.asarray(all_residuals, dtype=float).ravel()

        self._iteration += 1
        if self._progress_callback is not None:
            cost = float(np.sum(residuals**2))
            improved = cost < self._best_cost
            if improved:
                self._best_cost = cost
            if improved or self._iteration % 10 == 0:
                callback_params = dict(param_dict)
                if self._layout.dataset_var_index:
                    for ds_id, var_map in self._layout.dataset_var_index.items():
                        for var_name, idx in var_map.items():
                            raw_val = float(params[idx])
                            if self._layout.dataset_var_log10.get((ds_id, var_name)):
                                callback_params[f"{ds_id}::{var_name}"] = float(10.0 ** raw_val)
                            else:
                                callback_params[f"{ds_id}::{var_name}"] = raw_val
                self._progress_callback(self._iteration, cost, callback_params)
        _raise_if_fitting_cancelled(self._cancellation_check)

        return residuals


def _assemble_global_fit_result(
    *,
    fit_evaluator,
    payloads: List[FitDatasetSpec],
    layout: _FitParameterLayout,
    fitted_params: Dict[str, float],
    combined_dataset_params: Dict[str, Dict[str, float]],
    weights: Dict[str, float],
    penalty_value: float,
    cancellation_check: Optional[Callable[[], bool]],
    success: bool,
    message: str,
    nfev: int,
    covariance: Optional[np.ndarray],
    objective_residuals: Optional[np.ndarray],
    uncertainties: Optional[Dict[str, float]],
    process_pool: Optional[FittingProcessPool] = None,
) -> GlobalFitResult:
    dataset_info = []
    total_ss_res = 0.0
    total_ss_tot = 0.0
    total_points = 0
    model_series_map: Dict[str, Dict[str, np.ndarray]] = {}
    residual_series_map: Dict[str, Dict[str, np.ndarray]] = {}
    plot_model_x_map: Dict[str, np.ndarray] = {}
    plot_model_series_map: Dict[str, Dict[str, np.ndarray]] = {}
    final_dataset_errors: Dict[str, str] = {}
    final_dataset_warnings: Dict[str, str] = {}
    alignment_report: Dict[str, Dict[str, float]] = {}

    dataset_inputs: List[_ObjectiveDatasetInput] = []
    for index, payload in enumerate(payloads):
        ds_id = payload.dataset_id
        _raise_if_fitting_cancelled(cancellation_check)

        full_params = dict(fitted_params)
        full_params.update(combined_dataset_params.get(ds_id, {}))
        parameter_origins = {
            name: FITTING_PARAM_ORIGIN_OPTIMIZER_SHARED
            for name in fitted_params
        }
        for name in combined_dataset_params.get(ds_id, {}):
            if name in layout.dataset_var_index.get(ds_id, {}):
                parameter_origins[name] = FITTING_PARAM_ORIGIN_OPTIMIZER_DATASET
            else:
                parameter_origins[name] = FITTING_PARAM_ORIGIN_CONFIGURED_DATASET
        failed_param_snapshot = _GlobalFitObjective._build_failed_param_snapshot(
            ds_id=ds_id,
            shared_params=fitted_params,
            full_params=full_params,
        )
        dataset_inputs.append(
            _ObjectiveDatasetInput(
                index=int(index),
                payload=payload,
                full_params=full_params,
                parameter_origins=parameter_origins,
                failed_param_snapshot=failed_param_snapshot,
            )
        )

    simulation_evaluations = {
        int(result.index): result
        for result in _evaluate_dataset_simulations(
            fit_evaluator,
            dataset_inputs,
            cancellation_check=cancellation_check,
            stop_on_fatal=False,
            process_pool=process_pool,
        )
    }

    for item in dataset_inputs:
        payload = item.payload
        ds_id = payload.dataset_id
        species_list = payload.species_list
        y_matrix = payload.y_matrix
        t_exp = payload.t_exp
        x_name = payload.x_name
        x_obs = payload.x_obs if x_name != "t" else None
        x_mode = payload.x_mode
        failed_param_snapshot = item.failed_param_snapshot
        _raise_if_fitting_cancelled(cancellation_check)

        sim_time: Optional[np.ndarray] = None
        sim_species: Dict[str, np.ndarray] = {}
        evaluation = simulation_evaluations.get(int(item.index))
        if evaluation is None:
            raise RuntimeError(f"Missing fitting simulation result for dataset '{ds_id}'.")
        if evaluation.error is not None:
            exc = evaluation.error
            if ds_id not in final_dataset_errors:
                final_dataset_errors[ds_id] = str(evaluation.final_error_message or exc)
        else:
            sim_time = evaluation.sim_time
            sim_species = evaluation.sim_species

        plot_mask: Optional[np.ndarray] = None
        if x_name == "t":
            plot_x = np.asarray(t_exp, dtype=float).reshape(-1)
        else:
            plot_x = np.asarray([], dtype=float)
            if sim_time is not None and isinstance(sim_species, dict) and x_name in sim_species:
                t0 = float(np.min(np.asarray(t_exp, dtype=float)))
                t1 = float(np.max(np.asarray(t_exp, dtype=float)))
                if t0 > t1:
                    t0, t1 = t1, t0
                t_scale = max(1.0, abs(t0), abs(t1))
                t_pad = 1e-12 * t_scale
                plot_mask = (sim_time >= (t0 - t_pad)) & (sim_time <= (t1 + t_pad))
                if np.any(plot_mask):
                    try:
                        plot_x = np.asarray(sim_species.get(x_name), dtype=float).reshape(-1)[plot_mask]
                    except Exception:
                        plot_x = np.asarray([], dtype=float)
        plot_model_x_map[ds_id] = np.asarray(plot_x, dtype=float).reshape(-1)

        residual_blocks: List[np.ndarray] = []
        exp_blocks: List[np.ndarray] = []
        aligner: Optional[_ParametricXAligner] = None

        for idx, species_name in enumerate(species_list):
            y_exp = np.asarray(y_matrix[idx], dtype=float).reshape(-1)
            exp_blocks.append(y_exp)
            penalty_block = np.full_like(y_exp, penalty_value, dtype=float)

            if not (isinstance(sim_species, dict) and species_name in sim_species):
                if ds_id not in final_dataset_errors:
                    final_dataset_errors[ds_id] = f"Species '{species_name}' missing in simulation result."
                residual_blocks.append(penalty_block)
                residual_series_map.setdefault(ds_id, {})[species_name] = penalty_block
                continue

            try:
                y_sim_time = _align_series(sim_species[species_name], sim_time, t_exp)
                model_series_map.setdefault(ds_id, {})[species_name] = y_sim_time
            except Exception as exc:
                if ds_id not in final_dataset_errors:
                    final_dataset_errors[ds_id] = str(exc)
                residual_blocks.append(penalty_block)
                residual_series_map.setdefault(ds_id, {})[species_name] = penalty_block
                continue

            try:
                if x_name == "t":
                    y_sim_resid = y_sim_time
                else:
                    x_model_series = sim_species.get(x_name)
                    if x_model_series is None:
                        raise FitSimulationError(
                            f"X series '{x_name}' missing in simulation result for dataset '{ds_id}'.",
                            failed_params=failed_param_snapshot,
                        )
                    if x_obs is None:
                        raise FitSimulationError(
                            f"Dataset '{ds_id}' is missing x_obs for X='{x_name}'.",
                            failed_params=failed_param_snapshot,
                        )
                    if aligner is None:
                        aligner = _ParametricXAligner(
                            mode=x_mode,
                            t_obs=t_exp,
                            x_obs=x_obs,
                            t_sim=sim_time,
                            x_model=np.asarray(x_model_series, dtype=float),
                            dataset_label=ds_id,
                            x_name=x_name,
                        )
                    y_sim_resid = aligner.align(
                        y_model=np.asarray(sim_species[species_name], dtype=float),
                        y_name=species_name,
                        need_dx_penalty=False,
                    )
            except Exception as exc:
                if ds_id not in final_dataset_errors:
                    final_dataset_errors[ds_id] = str(exc)
                residual_blocks.append(penalty_block)
                residual_series_map.setdefault(ds_id, {})[species_name] = penalty_block
                continue

            if not np.all(np.isfinite(y_sim_resid)):
                if ds_id not in final_dataset_errors:
                    final_dataset_errors[ds_id] = (
                        f"Non-finite aligned series for dataset '{ds_id}', species '{species_name}'."
                    )
                residual_blocks.append(penalty_block)
                residual_series_map.setdefault(ds_id, {})[species_name] = penalty_block
            else:
                raw_residual = np.asarray(y_sim_resid, dtype=float) - y_exp
                residual_blocks.append(raw_residual)
                residual_series_map.setdefault(ds_id, {})[species_name] = raw_residual

            if x_name == "t":
                plot_model_series_map.setdefault(ds_id, {})[species_name] = y_sim_time
            else:
                if plot_mask is not None and isinstance(sim_species, dict) and species_name in sim_species:
                    values = np.asarray(sim_species[species_name], dtype=float).reshape(-1)
                    try:
                        sliced = np.asarray(values[plot_mask], dtype=float).reshape(-1)
                    except (IndexError, TypeError, ValueError):
                        sliced = None
                    if sliced is not None:
                        plot_model_series_map.setdefault(ds_id, {})[species_name] = sliced

        if aligner is not None and aligner.penalized_out is not None and x_name != "t" and x_obs is not None:
            try:
                dx = np.asarray(aligner.penalized_out.dx, dtype=float).reshape(-1)
                exact = np.asarray(aligner.penalized_out.exact, dtype=bool).reshape(-1)
            except Exception:
                dx = np.asarray([], dtype=float)
                exact = np.asarray([], dtype=bool)
            abs_dx = np.abs(dx[np.isfinite(dx)])
            max_abs_dx = float(np.max(abs_dx)) if abs_dx.size else float("inf")
            med_abs_dx = float(np.median(abs_dx)) if abs_dx.size else float("inf")
            n_points = float(dx.size)
            n_exact = float(np.sum(exact)) if exact.size else 0.0
            alignment_report[ds_id] = {
                "n_points": n_points,
                "n_exact": n_exact,
                "max_abs_dx": max_abs_dx,
                "median_abs_dx": med_abs_dx,
            }
            if bool(np.any(~exact)):
                x_span = _robust_span(x_obs)
                rel = (max_abs_dx / max(x_span, 1e-12)) if np.isfinite(max_abs_dx) else float("inf")
                if ds_id not in final_dataset_warnings:
                    final_dataset_warnings[ds_id] = (
                        f"Parametric-X mapping used approximations for X='{x_name}': "
                        f"exact={int(n_exact)}/{int(n_points)} points, max |ΔX|={max_abs_dx:.6g} "
                        f"(~{rel:.3g} of X span)."
                    )

        residuals = np.concatenate(residual_blocks) if residual_blocks else np.asarray([], dtype=float)
        y_exp_all = np.concatenate(exp_blocks) if exp_blocks else np.asarray([], dtype=float)

        ss_res = (
            float(np.sum(residuals**2)) if residuals.size else float((penalty_value**2) * float(y_exp_all.size))
        )
        ss_tot = float(np.sum((y_exp_all - float(np.mean(y_exp_all)))**2)) if y_exp_all.size else 0.0

        r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        chi_squared = ss_res / residuals.size if residuals.size else float(penalty_value**2)
        rmse = np.sqrt(np.mean(residuals**2)) if residuals.size else float(penalty_value)
        mae = np.mean(np.abs(residuals)) if residuals.size else float(penalty_value)

        dataset_info.append(
            DatasetFitInfo(
                dataset_id=ds_id,
                r_squared=r_squared,
                chi_squared=chi_squared,
                rmse=rmse,
                mae=mae,
                residuals=residuals,
                n_points=int(residuals.size),
                weight=weights.get(ds_id, 1.0),
            )
        )

        total_ss_res += ss_res
        total_ss_tot += ss_tot
        total_points += int(residuals.size)

    global_ss_res = total_ss_res
    global_points = total_points
    if objective_residuals is not None:
        objective_vector = np.asarray(objective_residuals, dtype=float).reshape(-1)
        if objective_vector.size:
            global_ss_res = float(np.sum(objective_vector**2))
            global_points = int(objective_vector.size)

    global_r_squared = 1.0 - (global_ss_res / total_ss_tot) if total_ss_tot > 0 else 0.0
    global_chi_squared = global_ss_res / global_points if global_points > 0 else float(penalty_value**2)

    metrics_finite = bool(
        np.isfinite(global_ss_res) and np.isfinite(global_r_squared) and np.isfinite(global_chi_squared)
    )
    if final_dataset_errors or not metrics_finite:
        success = False
        if final_dataset_errors:
            failed_labels = ", ".join(sorted(final_dataset_errors.keys()))
            message = f"{message} | alignment_failed={failed_labels}"
        if not metrics_finite:
            message = f"{message} | nonfinite_metrics"

    if success:
        logger.info("Global fit complete: R²=%.4f, χ²=%.4e", global_r_squared, global_chi_squared)
    else:
        logger.warning(
            "Global fit failed: χ²=%.4e datasets_failed=%s",
            global_chi_squared,
            ",".join(sorted(final_dataset_errors.keys())),
        )
    for info in dataset_info:
        logger.debug("  %s: R²=%.4f, RMSE=%.4e", info.dataset_id, info.r_squared, info.rmse)

    return GlobalFitResult(
        success=success,
        shared_params=fitted_params,
        dataset_params=combined_dataset_params,
        uncertainties=uncertainties,
        global_chi_squared=global_chi_squared,
        global_r_squared=global_r_squared,
        dataset_info=dataset_info,
        nfev=nfev,
        message=message,
        covariance=covariance,
        objective_residuals=objective_residuals.copy() if objective_residuals is not None else None,
        model_series=model_series_map,
        residual_series=residual_series_map,
        plot_model_x=plot_model_x_map,
        plot_model_series=plot_model_series_map,
        dataset_errors=dict(final_dataset_errors),
        dataset_warnings=dict(final_dataset_warnings),
        alignment_report=dict(alignment_report),
    )


def fit_global(
    fit_evaluator,
    datasets: List[object],
    shared_params: Dict[str, float],
    dataset_params: Optional[Dict[str, Dict[str, float]]] = None,
    dataset_variable_params: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None,
    bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    weights: Optional[Dict[str, float]] = None,
    method: str = "trf",
    max_nfev: int = 1000,
    ftol: float = 1e-10,
    xtol: float = 1e-10,
    seed: Optional[int] = None,
    log10_params: Optional[Dict[str, bool]] = None,
    progress_callback: Optional[Callable[[int, float, Dict[str, float]], None]] = None,
    cancellation_check: Optional[Callable[[], bool]] = None,
    dataset_overrides: Optional[List[object]] = None,
    process_pool_callback: Optional[Callable[[Optional[FittingProcessPool]], None]] = None,
) -> GlobalFitResult:
    """
    Fit single mechanism to multiple experimental datasets simultaneously.

    This function optimizes a single set of shared parameters (e.g., rate constants)
    to fit multiple datasets, while allowing dataset-specific parameters (e.g.,
    initial conditions). The objective is a weighted sum of residuals across all
    datasets.

    Parameters
    ----------
    fit_evaluator : object
        Shared fitting evaluation boundary. Must expose
        `evaluate_series(params: Dict[str, float]) -> SimulationSeriesPayload`,
        or be a callable that returns a compatible simulation-series payload.
    datasets : list of dict
        List of experimental datasets. Each dict must contain:
        - 't': np.ndarray of time points
        - 'y': np.ndarray of experimental values
        - 'species': str, target species name
        - 'id': str, unique identifier (optional, defaults to index)
    shared_params : dict
        Initial values for shared parameters {name: value}
    dataset_params : dict of dict, optional
        Fixed dataset-specific parameters {dataset_id: {name: value}}
        (e.g., required initial concentrations)
    dataset_variable_params : dict of dict, optional
        Dataset parameters to be optimized {dataset_id: {name: {"initial": float, "min": float, "max": float}}}
        Useful for fitting per-dataset initial conditions with bounds.
    bounds : dict, optional
        Parameter bounds {name: (min, max)} for shared parameters only
    weights : dict, optional
        Dataset weights {dataset_id: weight}. Defaults to 1/n_points for each dataset
    method : str
        Optimization method ('trf', 'dogbox', 'lm', 'de')
    max_nfev : int
        Maximum number of function evaluations
    seed : int, optional
        Random seed for differential evolution (reproducibility).
    log10_params : dict, optional
        Per-parameter toggle for log10-space fitting of shared parameters.
        Keys are shared parameter names; values are True/False. When enabled,
        bounds/initial values must be strictly positive and are transformed to
        log10-space internally.
    progress_callback : callable, optional
        Called with (iteration, cost, params_dict) periodically and on best-so-far
        improvements. When dataset-specific variable parameters are being optimized,
        they are included in `params_dict` with a unique key of the form
        `"{dataset_id}::{param_name}"` to avoid collisions across datasets.
    cancellation_check : callable, optional
        Function returning True when fitting should abort early.

    Returns
    -------
    GlobalFitResult
        Comprehensive fitting results including per-dataset statistics

    Examples
    --------
    >>> # Fit rate constant to two datasets with different initial conditions
    >>> def sim(params):
    ...     k = params['k']
    ...     # Run simulation with k...
    ...     return {'A': concentrations}
    >>>
    >>> datasets = [
    ...     {'id': 'exp1', 't': t1, 'y': y1, 'species': 'A'},
    ...     {'id': 'exp2', 't': t2, 'y': y2, 'species': 'A'},
    ... ]
    >>> shared = {'k': 0.1}  # Initial guess
    >>> result = fit_global(sim, datasets, shared)
    >>> print(f"Optimal k: {result.shared_params['k']:.3f}")
    >>> print(f"Global R²: {result.global_r_squared:.3f}")

    Notes
    -----
    - Datasets may be single-species (`species: str`, `y: (N,)`) or multi-species
      (`species: list[str]`, `y: (S,N)` where rows align to `species` order).
    - Weights default to inverse of number of points (equal contribution per dataset).
    - Dataset-specific params are fixed (not optimized) - useful for IC variations.
    - Returns comprehensive per-dataset statistics for validation.
    """
    datasets = _normalize_input_datasets(datasets)
    payloads = _build_dataset_payloads(datasets)

    overrides = coerce_fit_dataset_parameter_overrides(
        dataset_ids=[payload.dataset_id for payload in payloads],
        dataset_overrides=dataset_overrides,
        dataset_params=dataset_params,
        dataset_variable_params=dataset_variable_params,
    )
    dataset_params_map_raw, dataset_variable_params_map_raw = split_fit_dataset_parameter_overrides(overrides)
    dataset_params_map: Dict[str, Dict[str, float]] = {
        ds_id: {name: float(value) for name, value in param_map.items()}
        for ds_id, param_map in dataset_params_map_raw.items()
    }
    dataset_variable_params_map: Dict[str, Dict[str, Dict[str, float]]] = {
        ds_id: {
            name: {
                "initial": float(spec["initial"]),
                "min": float(spec["min"]),
                "max": float(spec["max"]),
                "log10": bool(spec.get("log10", False)),
            }
            for name, spec in param_map.items()
        }
        for ds_id, param_map in dataset_variable_params_map_raw.items()
    }

    weights_norm = _normalize_weights(payloads, weights)

    logger.info("Global fitting: %d datasets, %d shared parameters", len(payloads), len(shared_params))
    for payload in payloads:
        species_label = ", ".join(payload.species_list)
        logger.debug(
            "  %s: %d points, weight=%.3f, species=%s",
            payload.dataset_id,
            payload.point_count,
            weights_norm.get(payload.dataset_id, 1.0),
            species_label,
        )

    penalty_value = 1e6
    objective_total_points = int(sum(p.point_count for p in payloads))

    layout = _build_parameter_layout(
        payloads=payloads,
        shared_params=shared_params,
        dataset_variable_params=dataset_variable_params_map,
        bounds=bounds,
        log10_params=log10_params,
    )

    method_key = str(method or "trf").strip().lower()
    if method_key in {"differential_evolution", "de"}:
        method_key = "de"
    elif method_key not in {"lm", "trf", "dogbox"}:
        raise ValueError(f"Unknown optimization method: {method}")

    ctx = ObjectiveContext()
    fit_evaluator = coerce_fitting_series_evaluator(fit_evaluator)
    process_pool: Optional[FittingProcessPool] = None
    process_payload = None
    try:
        if type(fit_evaluator) is SerialFittingEvaluator and len(payloads) > 1:
            try:
                process_payload = fit_evaluator.to_process_payload()
            except (pickle.PicklingError, TypeError) as exc:
                logger.warning(
                    "Fitting evaluator payload is not process-picklable; using serial evaluation: %s",
                    exc,
                )
            if process_payload is not None:
                process_pool = FittingProcessPool(
                    process_payload,
                    max_workers=_effective_fitting_process_workers(len(payloads)),
                    limit_blas_threads=True,
                    publish_callback=process_pool_callback,
                )
    except Exception:
        if process_pool is not None:
            process_pool.shutdown(force_terminate=True)
        raise

    objective_impl = _GlobalFitObjective(
        fit_evaluator=fit_evaluator,
        payloads=payloads,
        shared_params=shared_params,
        dataset_params=dataset_params_map,
        weights=weights_norm,
        layout=layout,
        penalty_value=penalty_value,
        ctx=ctx,
        progress_callback=progress_callback,
        cancellation_check=cancellation_check,
        process_pool=process_pool,
    )
    objective_wrapper = ObjectiveWrapper(objective_impl, ctx)

    def _failed_result(message: str, *, failed_params: Optional[Dict[str, float]] = None) -> GlobalFitResult:
        shared_snapshot = dict(shared_params)
        dataset_snapshot: Dict[str, Dict[str, float]] = {
            payload.dataset_id: dict(dataset_params_map.get(payload.dataset_id, {})) for payload in payloads
        }
        if failed_params:
            formatted_failed_params = _GlobalFitObjective._format_params(failed_params)
            dataset_param_owners: Dict[str, set[str]] = {}
            for ds_id, params_map in dataset_snapshot.items():
                for param_name in params_map:
                    dataset_param_owners.setdefault(param_name, set()).add(ds_id)
            for ds_id, var_map in layout.dataset_var_index.items():
                for var_name in var_map:
                    dataset_param_owners.setdefault(var_name, set()).add(ds_id)

            for name, value in formatted_failed_params.items():
                if "::" in name:
                    failed_ds_id, param_name = name.split("::", 1)
                    dataset_snapshot.setdefault(failed_ds_id, {})
                    dataset_snapshot[failed_ds_id][param_name] = float(value)
                    continue
                if name in shared_snapshot or name in layout.param_names:
                    shared_snapshot[name] = float(value)
                    continue
                owners = dataset_param_owners.get(name, set())
                if len(owners) == 1:
                    owner_ds_id = next(iter(owners))
                    dataset_snapshot.setdefault(owner_ds_id, {})
                    dataset_snapshot[owner_ds_id][name] = float(value)

        return GlobalFitResult(
            success=False,
            shared_params=shared_snapshot,
            dataset_params=dataset_snapshot,
            uncertainties=None,
            global_chi_squared=np.inf,
            global_r_squared=0.0,
            dataset_info=[],
            nfev=int(getattr(objective_impl, "_iteration", 0)),
            message=message,
            covariance=None,
            objective_residuals=None,
            model_series={},
            residual_series={},
        )

    objective_residuals: Optional[np.ndarray] = None

    def _shutdown_process_pool(*, force_terminate: bool) -> None:
        if process_pool is None:
            return
        process_pool.shutdown(force_terminate=force_terminate)

    try:
        de_penalty = None
        if method_key == "de":
            de_penalty = float((float(penalty_value) ** 2) * float(max(1, objective_total_points)))

        fit_result = fit_parameters(
            objective_wrapper,
            layout.initial_params(),
            bounds=layout.bounds_dict(),
            method=method_key,
            progress_callback=None,
            max_nfev=max_nfev,
            seed=seed,
            ftol=ftol,
            xtol=xtol,
            cancellation_check=cancellation_check,
            de_penalty=de_penalty,
        )

        opt_param_keys = layout.opt_param_keys()
        x_opt = np.asarray([fit_result.parameters[key] for key in opt_param_keys], dtype=float)
        success = bool(fit_result.success)
        message = str(fit_result.message)
        nfev = int(fit_result.nfev)
        covariance = fit_result.covariance
        objective_residuals = np.asarray(fit_result.residuals, dtype=float).reshape(-1)

        if getattr(objective_wrapper, "last_error", None) is not None:
            ds_tag = None
            prov = getattr(objective_wrapper, "last_error_provenance", None)
            if isinstance(prov, dict):
                ds_tag = prov.get("dataset")
            message = f"{message} | last_error_dataset={ds_tag}: {objective_wrapper.last_error}"

    except FittingCancelled:
        _shutdown_process_pool(force_terminate=True)
        raise
    except FitSimulationError as exc:
        logger.error("Global fitting failed: %s", exc, exc_info=False)
        _shutdown_process_pool(force_terminate=True)
        return _failed_result(str(exc), failed_params=exc.failed_params)
    except Exception as exc:
        if isinstance(exc, (FittingCancelled, SimulationCancelled)) or "cancelled" in str(exc).lower():
            _shutdown_process_pool(force_terminate=True)
            raise FittingCancelled() from exc
        logger.error("Global fitting failed: %s", exc, exc_info=True)
        _shutdown_process_pool(force_terminate=True)
        return _failed_result(f"Fitting failed: {exc}")

    try:
        fitted_params = layout.shared_param_dict_from_vector(x_opt)

        combined_dataset_params: Dict[str, Dict[str, float]] = {
            payload.dataset_id: dict(dataset_params_map.get(payload.dataset_id, {})) for payload in payloads
        }
        for (ds_id, param_name) in layout.dataset_var_order:
            combined_dataset_params.setdefault(ds_id, {})
            idx = layout.dataset_var_index[ds_id][param_name]
            val = float(x_opt[idx])
            if layout.dataset_var_log10.get((ds_id, param_name)):
                combined_dataset_params[ds_id][param_name] = float(10.0 ** val)
            else:
                combined_dataset_params[ds_id][param_name] = val

        uncertainties = None
        if covariance is not None:
            try:
                std_devs = np.sqrt(np.diag(covariance))
                uncertainties = {}
                for i, name in enumerate(layout.param_names):
                    if i >= int(std_devs.size):
                        break
                    std = float(std_devs[i])
                    if layout.shared_log10.get(name):
                        x_val = float(fitted_params.get(name, float("nan")))
                        uncertainties[name] = float(np.log(10.0) * x_val * std)
                    else:
                        uncertainties[name] = std
            except Exception as exc:
                logger.debug("Failed to calculate uncertainties: %s", exc)

        return _assemble_global_fit_result(
            fit_evaluator=fit_evaluator,
            payloads=payloads,
            layout=layout,
            fitted_params=fitted_params,
            combined_dataset_params=combined_dataset_params,
            weights=weights_norm,
            penalty_value=penalty_value,
            cancellation_check=cancellation_check,
            success=success,
            message=message,
            nfev=nfev,
            covariance=covariance,
            objective_residuals=objective_residuals,
            uncertainties=uncertainties,
            process_pool=process_pool,
        )
    except Exception:
        _shutdown_process_pool(force_terminate=True)
        raise
    finally:
        _shutdown_process_pool(force_terminate=False)
        if process_pool_callback is not None:
            with suppress(Exception):
                process_pool_callback(None)
