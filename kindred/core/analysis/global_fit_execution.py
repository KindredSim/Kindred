"""
Global fitting execution boundary.

This module owns multi-dataset fitting candidate evaluation, final replay,
runtime dispatch, diagnostics, and result/completion assembly. Public API
composition remains in ``global_fitting.fit_global``.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from kindred.core.analysis.parametric_alignment import (
    align_y_on_x_obs,
    align_y_on_x_obs_time_guided_penalized,
    is_non_monotone_in_sampled_window_error,
)
from kindred.core.analysis.fit_dataset_payload import FitDatasetSpec, coerce_fit_dataset_specs
from kindred.core.exceptions import FittingCancelled, FitSimulationError, SimulationCancelled
from kindred.core.fitting_completion import FitDiagnostic, FitDetailSection, GlobalFitCompletion
from kindred.core.fitting_evaluation import (
    FITTING_PARAM_ORIGIN_CONFIGURED_DATASET,
    FITTING_PARAM_ORIGIN_OPTIMIZER_DATASET,
    FITTING_PARAM_ORIGIN_OPTIMIZER_SHARED,
    evaluate_fitting_series,
)
from kindred.core.fitting_runtime_session import FittingRuntimeRequest
from kindred.core.objective import ObjectiveContext
from kindred.core.simulation_failure import (
    SimulationFailure,
    build_simulation_failure,
    coerce_simulation_failure,
    simulation_failure_detail_text,
    simulation_failure_from_exception,
    simulation_failure_user_message,
)
from kindred.core.simulation_series_payload import coerce_simulation_series_payload
from kindred.core.analysis.x_mapping import normalize_x_mapping_mode

logger = logging.getLogger(__name__)

__all__ = [
    "DatasetFitInfo",
    "GlobalFitResult",
    "FitParameterLayout",
    "ObjectiveDatasetInput",
    "DatasetSimulationEvaluation",
    "GlobalFitObjective",
    "build_dataset_payloads",
    "build_parameter_layout",
    "normalize_input_datasets",
    "normalize_weights",
    "evaluate_dataset_simulation",
    "evaluate_dataset_simulations",
    "dataset_evaluation_is_fatal",
    "assemble_global_fit_result",
]

def _raise_if_fitting_cancelled(cancellation_check: Optional[Callable[[], bool]]) -> None:
    if cancellation_check is not None and cancellation_check():
        raise FittingCancelled()


def _raise_if_fitting_cancel_requested(cancellation_check: Optional[Callable[[], bool]]) -> None:
    if cancellation_check is None:
        return
    cancel_requested = getattr(cancellation_check, "_kindred_nonblocking_cancelled", cancellation_check)
    if bool(cancel_requested()):
        raise FittingCancelled()

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
    t_obs: Optional[np.ndarray] = None  # Observed time values used for fitting
    x_name: str = "t"  # Observed X-axis name for render projection
    x_obs: Optional[np.ndarray] = None  # Observed non-time X values when x_name != "t"


@dataclass
class GlobalFitResult:
    """
    Results from global fitting across multiple datasets.

    Attributes
    ----------
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
    shared_params: Dict[str, float]
    dataset_params: Dict[str, Dict[str, float]]
    uncertainties: Optional[Dict[str, float]]
    global_chi_squared: float
    global_r_squared: float
    dataset_info: List[DatasetFitInfo]
    nfev: int
    message: str
    completion: GlobalFitCompletion
    covariance: Optional[np.ndarray] = None
    objective_residuals: Optional[np.ndarray] = None
    model_series: Dict[str, Dict[str, np.ndarray]] = field(default_factory=dict)
    residual_series: Dict[str, Dict[str, np.ndarray]] = field(default_factory=dict)
    plot_observed_series: Dict[str, Dict[str, np.ndarray]] = field(default_factory=dict)
    plot_model_x: Dict[str, np.ndarray] = field(default_factory=dict)
    plot_model_series: Dict[str, Dict[str, np.ndarray]] = field(default_factory=dict)
    alignment_report: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.completion, GlobalFitCompletion):
            raise TypeError("GlobalFitResult.completion must be a GlobalFitCompletion")


def _coerce_fit_failure_payload(value: object) -> SimulationFailure:
    payload = dict(coerce_simulation_failure(value))
    details = dict(payload.get("details") or {})
    details.pop("parameters", None)
    payload["details"] = details
    return payload


def _coerce_parameter_snapshot(snapshot: Optional[Dict[str, float]]) -> Optional[Dict[str, float]]:
    if not snapshot:
        return None
    return {str(name): float(value) for name, value in dict(snapshot).items()}


def _compact_failure_message(failure: SimulationFailure) -> str:
    raw_message = simulation_failure_user_message(failure)
    return " ".join(part.strip() for part in str(raw_message).splitlines() if part.strip())


def build_fit_diagnostic_from_failure(
    failure: object,
    *,
    phase: str,
    dataset_id: Optional[str] = None,
    parameter_snapshot: Optional[Dict[str, float]] = None,
) -> FitDiagnostic:
    coerced_failure = _coerce_fit_failure_payload(failure)
    return FitDiagnostic(
        phase=phase,
        dataset_id=dataset_id,
        failure=coerced_failure,
        parameter_snapshot=_coerce_parameter_snapshot(parameter_snapshot),
    )


def _embedded_fit_failure_payload(exc: BaseException) -> Optional[SimulationFailure]:
    details = getattr(exc, "details", None)
    embedded = details.get("failure") if isinstance(details, Mapping) else None
    if not (isinstance(embedded, Mapping) and "kind" in embedded and "message" in embedded):
        return None
    return _coerce_fit_failure_payload(embedded)


def build_fit_diagnostic_from_exception(
    exc: BaseException,
    *,
    phase: str,
    dataset_id: Optional[str] = None,
    parameter_snapshot: Optional[Dict[str, float]] = None,
    message_override: Optional[str] = None,
) -> FitDiagnostic:
    chosen_snapshot = getattr(exc, "failed_params", None)
    if not chosen_snapshot:
        chosen_snapshot = parameter_snapshot
    failure = _embedded_fit_failure_payload(exc)
    if failure is None:
        failure = _coerce_fit_failure_payload(simulation_failure_from_exception(exc))
    if message_override is not None and str(message_override).strip():
        failure["message"] = str(message_override)
    return FitDiagnostic(
        phase=phase,
        dataset_id=dataset_id,
        failure=failure,
        parameter_snapshot=_coerce_parameter_snapshot(chosen_snapshot),
    )


def build_completion_detail_sections(
    *,
    status: str,
    optimizer_diagnostic: Optional[FitDiagnostic],
    dataset_failures: Dict[str, FitDiagnostic],
) -> List[FitDetailSection]:
    if status == "ok":
        return []

    sections: List[FitDetailSection] = []
    if optimizer_diagnostic is not None:
        optimizer_detail = simulation_failure_detail_text(optimizer_diagnostic.failure)
        optimizer_summary = _compact_failure_message(optimizer_diagnostic.failure)
        matching_detail = ""
        matching_summary = ""
        if optimizer_diagnostic.dataset_id is not None:
            matching = dataset_failures.get(str(optimizer_diagnostic.dataset_id))
            if matching is not None:
                matching_detail = simulation_failure_detail_text(matching.failure)
                matching_summary = _compact_failure_message(matching.failure)
        if (optimizer_detail and optimizer_detail != matching_detail) or (
            (not optimizer_detail) and optimizer_summary and optimizer_summary != matching_summary
        ):
            sections.append(
                FitDetailSection(
                    dataset_id=optimizer_diagnostic.dataset_id,
                    failure=optimizer_diagnostic.failure,
                )
            )

    for ds_id, diagnostic in sorted(dataset_failures.items(), key=lambda kv: str(kv[0])):
        if simulation_failure_detail_text(diagnostic.failure) or _compact_failure_message(diagnostic.failure):
            sections.append(
                FitDetailSection(
                    dataset_id=str(ds_id),
                    failure=diagnostic.failure,
                )
            )
    return sections


def completion_result_message(
    *,
    base_message: str,
    completion: GlobalFitCompletion,
    optimizer_diagnostic: Optional[FitDiagnostic],
    dataset_failures: Dict[str, FitDiagnostic],
    dataset_warnings: Dict[str, str],
) -> str:
    if completion.status == "ok":
        return str(base_message)
    if completion.status == "warn":
        warning_parts: list[str] = []
        if not completion.optimizer_converged:
            fallback = str(base_message or "").strip()
            if fallback and "success" not in fallback.lower():
                warning_parts.append(fallback)
            else:
                warning_parts.append("optimizer did not report convergence")
        if optimizer_diagnostic is not None:
            detail = _compact_failure_message(optimizer_diagnostic.failure)
            if detail:
                warning_parts.append(detail)
        for ds_id, message in sorted(dataset_warnings.items(), key=lambda kv: str(kv[0])):
            compact_message = " ".join(part.strip() for part in str(message).splitlines() if part.strip())
            if compact_message:
                warning_parts.append(f"dataset '{ds_id}': {compact_message}")
        if warning_parts:
            return "Global fit completed with warnings: " + "; ".join(warning_parts)
        fallback = str(base_message or "").strip()
        if fallback and "success" not in fallback.lower():
            return fallback
        return "Global fit completed with warnings."
    if dataset_failures:
        dataset_ids = sorted(str(ds_id) for ds_id in dataset_failures)
        if len(dataset_ids) == 1:
            dataset_id = dataset_ids[0]
            detail = _compact_failure_message(dataset_failures[dataset_id].failure)
            if detail:
                return f"Global fit failed for dataset '{dataset_id}': {detail}"
            return f"Global fit failed for dataset '{dataset_id}'."
        return f"Global fit failed for datasets: {', '.join(dataset_ids)}."
    if completion.nonfinite_metrics:
        return "Final χ² is non-finite; results are invalid."
    if optimizer_diagnostic is not None:
        detail = _compact_failure_message(optimizer_diagnostic.failure)
        if detail:
            return f"Global fit failed: {detail}"
    fallback = str(base_message or "").strip()
    if fallback:
        return fallback
    return "Global fit failed."


def ensure_failure_diagnostic(
    diagnostic: Optional[FitDiagnostic],
    *,
    message: str,
) -> FitDiagnostic:
    if diagnostic is not None:
        return diagnostic
    return build_fit_diagnostic_from_failure(
        build_simulation_failure(
            kind="simulation_error",
            message=message,
        ),
        phase="fatal",
    )


def normalize_input_datasets(datasets: List[object]) -> List[object]:
    return list(datasets)


def build_dataset_payloads(datasets: List[object]) -> List[FitDatasetSpec]:
    return coerce_fit_dataset_specs(datasets)


def normalize_weights(payloads: List[FitDatasetSpec], weights: Optional[Dict[str, float]]) -> Dict[str, float]:
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
class FitParameterLayout:
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


def build_parameter_layout(
    *,
    payloads: List[FitDatasetSpec],
    shared_params: Dict[str, float],
    dataset_variable_params: Dict[str, Dict[str, Dict[str, float]]],
    bounds: Optional[Dict[str, Tuple[float, float]]],
    log10_params: Optional[Dict[str, bool]],
) -> FitParameterLayout:
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

    return FitParameterLayout(
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
class ObjectiveDatasetInput:
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
class DatasetSimulationEvaluation:
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


def dataset_evaluation_is_fatal(result: DatasetSimulationEvaluation) -> bool:
    return isinstance(result.error, FitSimulationError) and bool(
        getattr(result.error, "details", {}).get("fatal")
    )


def evaluate_dataset_simulation(
    fit_evaluator,
    item: ObjectiveDatasetInput,
    *,
    cancellation_check: Optional[Callable[[], bool]] = None,
) -> DatasetSimulationEvaluation:
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
        return DatasetSimulationEvaluation(
            index=int(item.index),
            sim_time=sim_time,
            sim_species=sim_species,
        )
    except FitSimulationError as exc:
        return DatasetSimulationEvaluation(
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
        preferred_failed_params = getattr(exc, "failed_params", None)
        if not preferred_failed_params:
            preferred_failed_params = item.failed_param_snapshot
        error_details = dict(getattr(exc, "details", None) or {})
        if preferred_failed_params:
            error_details.pop("parameters", None)
        return DatasetSimulationEvaluation(
            index=int(item.index),
            sim_time=None,
            sim_species={},
            error=FitSimulationError(
                f"Simulation failed for dataset '{ds_id}': {exc}",
                failed_params=preferred_failed_params,
                details=error_details,
                context=getattr(exc, "context", None),
            ),
            error_provenance={"dataset": ds_id},
            final_error_message=str(exc),
        )

def evaluate_dataset_simulations(
    fit_evaluator,
    items: Sequence[ObjectiveDatasetInput],
    *,
    cancellation_check: Optional[Callable[[], bool]] = None,
    stop_on_fatal: bool = True,
) -> List[DatasetSimulationEvaluation]:
    if not items:
        return []
    runtime_batch = getattr(fit_evaluator, "evaluate_fitting_runtime_batch", None)
    if callable(runtime_batch):
        return evaluate_dataset_simulations_runtime_batch(
            fit_evaluator,
            items,
            cancellation_check=cancellation_check,
        )
    return evaluate_dataset_simulations_serial(
        fit_evaluator,
        items,
        cancellation_check=cancellation_check,
        stop_on_fatal=stop_on_fatal,
    )


def _evaluation_from_runtime_value(
    item: ObjectiveDatasetInput,
    value: object,
    *,
    cancellation_check: Optional[Callable[[], bool]],
) -> DatasetSimulationEvaluation:
    if isinstance(value, BaseException):
        exc = value
        if isinstance(exc, (FittingCancelled, SimulationCancelled)):
            raise FittingCancelled() from exc
        if isinstance(exc, FitSimulationError):
            return DatasetSimulationEvaluation(
                index=int(item.index),
                sim_time=None,
                sim_species={},
                error=exc,
                error_provenance={"dataset": item.payload.dataset_id, "provenance": getattr(exc, "provenance", None)},
                final_error_message=str(exc),
            )
        if exc.__class__.__name__ == "FittingLaneProtocolError":
            fatal = FitSimulationError(
                str(exc),
                failed_params=item.failed_param_snapshot,
                details={
                    "fatal": True,
                    "failure": build_simulation_failure(
                        "fitting_containment_protocol",
                        str(exc),
                        exc_type=exc.__class__.__name__,
                    ),
                },
                context=getattr(exc, "context", None),
            )
            return DatasetSimulationEvaluation(
                index=int(item.index),
                sim_time=None,
                sim_species={},
                error=fatal,
                error_provenance={"dataset": item.payload.dataset_id},
                final_error_message=str(exc),
            )
        preferred_failed_params = getattr(exc, "failed_params", None) or item.failed_param_snapshot
        error_details = dict(getattr(exc, "details", None) or {})
        if preferred_failed_params:
            error_details.pop("parameters", None)
        return DatasetSimulationEvaluation(
            index=int(item.index),
            sim_time=None,
            sim_species={},
            error=FitSimulationError(
                f"Simulation failed for dataset '{item.payload.dataset_id}': {exc}",
                failed_params=preferred_failed_params,
                details=error_details,
                context=getattr(exc, "context", None),
            ),
            error_provenance={"dataset": item.payload.dataset_id},
            final_error_message=str(exc),
        )

    _raise_if_fitting_cancel_requested(cancellation_check)
    sim_time, sim_species = _extract_simulation_payload(value)
    _raise_if_fitting_cancel_requested(cancellation_check)
    return DatasetSimulationEvaluation(
        index=int(item.index),
        sim_time=sim_time,
        sim_species=sim_species,
    )


def _runtime_protocol_failure_evaluations(
    exc: BaseException,
    items: Sequence[ObjectiveDatasetInput],
) -> List[DatasetSimulationEvaluation]:
    fatal_failure = build_simulation_failure(
        "fitting_containment_protocol",
        str(exc),
        exc_type=exc.__class__.__name__,
    )
    results: List[DatasetSimulationEvaluation] = []
    for item in items:
        fatal = FitSimulationError(
            str(exc),
            failed_params=item.failed_param_snapshot,
            details={
                "fatal": True,
                "failure": dict(fatal_failure),
            },
            context=getattr(exc, "context", None),
        )
        results.append(
            DatasetSimulationEvaluation(
                index=int(item.index),
                sim_time=None,
                sim_species={},
                error=fatal,
                error_provenance={"dataset": item.payload.dataset_id},
                final_error_message=str(exc),
            )
        )
    return results


def _fit_simulation_error_failure_kind(exc: FitSimulationError) -> str:
    details = getattr(exc, "details", None)
    if not isinstance(details, Mapping):
        return ""
    failure = details.get("failure")
    if not isinstance(failure, Mapping):
        return ""
    return str(failure.get("kind") or "")


def _fit_simulation_error_is_runtime_fatal(exc: FitSimulationError) -> bool:
    details = getattr(exc, "details", None)
    return (
        isinstance(details, Mapping)
        and bool(details.get("fatal"))
        and _fit_simulation_error_failure_kind(exc).startswith("fitting_containment_")
    )


def _runtime_fit_failure_evaluations(
    exc: FitSimulationError,
    items: Sequence[ObjectiveDatasetInput],
) -> List[DatasetSimulationEvaluation]:
    return [
        DatasetSimulationEvaluation(
            index=int(item.index),
            sim_time=None,
            sim_species={},
            error=exc,
            error_provenance={"dataset": item.payload.dataset_id, "provenance": getattr(exc, "provenance", None)},
            final_error_message=str(exc),
        )
        for item in items
    ]


def evaluate_dataset_simulations_runtime_batch(
    fit_evaluator,
    items: Sequence[ObjectiveDatasetInput],
    *,
    cancellation_check: Optional[Callable[[], bool]],
) -> List[DatasetSimulationEvaluation]:
    requests = [
        FittingRuntimeRequest(
            params=item.full_params,
            origins=item.parameter_origins,
            failed_params=item.failed_param_snapshot,
        )
        for item in items
    ]
    try:
        values = fit_evaluator.evaluate_fitting_runtime_batch(
            requests,
            cancellation_check=cancellation_check,
        )
    except Exception as exc:
        if isinstance(exc, (FittingCancelled, SimulationCancelled)):
            raise FittingCancelled() from exc
        if isinstance(exc, FitSimulationError) and _fit_simulation_error_is_runtime_fatal(exc):
            return _runtime_fit_failure_evaluations(exc, items)
        if exc.__class__.__name__ == "FittingLaneProtocolError":
            return _runtime_protocol_failure_evaluations(exc, items)
        raise
    if len(values) != len(items):
        raise RuntimeError("Fitting runtime session returned the wrong number of dataset evaluations.")
    return [
        _evaluation_from_runtime_value(
            item,
            value,
            cancellation_check=cancellation_check,
        )
        for item, value in zip(items, values)
    ]


def evaluate_dataset_simulations_serial(
    fit_evaluator,
    items: Sequence[ObjectiveDatasetInput],
    *,
    cancellation_check: Optional[Callable[[], bool]],
    stop_on_fatal: bool,
) -> List[DatasetSimulationEvaluation]:
    results = []
    for item in items:
        _raise_if_fitting_cancelled(cancellation_check)
        result = evaluate_dataset_simulation(
            fit_evaluator,
            item,
            cancellation_check=cancellation_check,
        )
        results.append(result)
        if stop_on_fatal and dataset_evaluation_is_fatal(result):
            break
    return results


class GlobalFitObjective:
    def __init__(
        self,
        *,
        fit_evaluator,
        payloads: List[FitDatasetSpec],
        shared_params: Dict[str, float],
        dataset_params: Dict[str, Dict[str, float]],
        weights: Dict[str, float],
        layout: FitParameterLayout,
        penalty_value: float,
        ctx: ObjectiveContext,
        progress_callback: Optional[Callable[[int, float, Dict[str, float]], None]],
        cancellation_check: Optional[Callable[[], bool]],
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

        self._iteration = 0
        self._best_cost = float("inf")
        self._warned_objective_keys: set[tuple[str, str, str]] = set()

    @staticmethod
    def format_params(params_map: Dict[str, float]) -> Dict[str, float]:
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
    def build_failed_param_snapshot(
        *,
        ds_id: str,
        shared_params: Dict[str, float],
        full_params: Dict[str, float],
    ) -> Dict[str, float]:
        snapshot = GlobalFitObjective.format_params(shared_params)
        shared_keys = set(snapshot)
        for name, value in GlobalFitObjective.format_params(full_params).items():
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
        dataset_inputs: List[ObjectiveDatasetInput] = []

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
            full_params = self.format_params(raw_full_params)
            parameter_origins = {
                name: parameter_origins[name]
                for name in full_params
                if name in parameter_origins
            }
            failed_param_snapshot = self.build_failed_param_snapshot(
                ds_id=ds_id,
                shared_params=param_dict,
                full_params=full_params,
            )
            dataset_inputs.append(
                ObjectiveDatasetInput(
                    index=int(index),
                    payload=payload,
                    full_params=full_params,
                    parameter_origins=parameter_origins,
                    failed_param_snapshot=failed_param_snapshot,
                )
            )

        simulation_evaluations = {
            int(result.index): result
            for result in evaluate_dataset_simulations(
                self._fit_evaluator,
                dataset_inputs,
                cancellation_check=self._cancellation_check,
            )
        }

        for evaluation in simulation_evaluations.values():
            if not dataset_evaluation_is_fatal(evaluation):
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
                        details=dict(getattr(exc, "details", None) or {}),
                        context=getattr(exc, "context", None),
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
                        FitSimulationError(
                            str(exc),
                            failed_params=failed_param_snapshot,
                            details=dict(getattr(exc, "details", None) or {}),
                            context=getattr(exc, "context", None),
                        ),
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
                        details=dict(getattr(exc, "details", None) or {}),
                        context=getattr(exc, "context", None),
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


def assemble_global_fit_result(
    *,
    fit_evaluator,
    payloads: List[FitDatasetSpec],
    layout: FitParameterLayout,
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
    optimizer_diagnostic: Optional[FitDiagnostic] = None,
) -> GlobalFitResult:
    dataset_info = []
    total_ss_res = 0.0
    total_ss_tot = 0.0
    total_points = 0
    model_series_map: Dict[str, Dict[str, np.ndarray]] = {}
    residual_series_map: Dict[str, Dict[str, np.ndarray]] = {}
    plot_observed_series_map: Dict[str, Dict[str, np.ndarray]] = {}
    plot_model_x_map: Dict[str, np.ndarray] = {}
    plot_model_series_map: Dict[str, Dict[str, np.ndarray]] = {}
    final_dataset_failures: Dict[str, FitDiagnostic] = {}
    final_dataset_warnings: Dict[str, str] = {}
    alignment_report: Dict[str, Dict[str, float]] = {}

    dataset_inputs: List[ObjectiveDatasetInput] = []
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
        failed_param_snapshot = GlobalFitObjective.build_failed_param_snapshot(
            ds_id=ds_id,
            shared_params=fitted_params,
            full_params=full_params,
        )
        dataset_inputs.append(
            ObjectiveDatasetInput(
                index=int(index),
                payload=payload,
                full_params=full_params,
                parameter_origins=parameter_origins,
                failed_param_snapshot=failed_param_snapshot,
            )
        )

    simulation_evaluations = {
        int(result.index): result
        for result in evaluate_dataset_simulations(
            fit_evaluator,
            dataset_inputs,
            cancellation_check=cancellation_check,
            stop_on_fatal=False,
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
            if ds_id not in final_dataset_failures:
                if evaluation.error is not None:
                    final_error_message = str(evaluation.final_error_message or "")
                    diagnostic = build_fit_diagnostic_from_exception(
                        evaluation.error,
                        phase="final_replay",
                        dataset_id=ds_id,
                        parameter_snapshot=failed_param_snapshot,
                        message_override=(
                            final_error_message
                            if final_error_message.strip() and final_error_message != str(exc)
                            else None
                        ),
                    )
                else:
                    diagnostic = build_fit_diagnostic_from_failure(
                        build_simulation_failure(
                            kind="simulation_error",
                            message=str(evaluation.final_error_message or ""),
                        ),
                        phase="final_replay",
                        dataset_id=ds_id,
                        parameter_snapshot=failed_param_snapshot,
                    )
                final_dataset_failures[ds_id] = diagnostic
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
            plot_observed_series_map.setdefault(ds_id, {})[species_name] = y_exp.copy()
            penalty_block = np.full_like(y_exp, penalty_value, dtype=float)

            if not (isinstance(sim_species, dict) and species_name in sim_species):
                if ds_id not in final_dataset_failures:
                    final_dataset_failures[ds_id] = build_fit_diagnostic_from_failure(
                        build_simulation_failure(
                            kind="simulation_error",
                            message=f"Species '{species_name}' missing in simulation result.",
                        ),
                        phase="final_replay",
                        dataset_id=ds_id,
                        parameter_snapshot=failed_param_snapshot,
                    )
                residual_blocks.append(penalty_block)
                residual_series_map.setdefault(ds_id, {})[species_name] = penalty_block
                continue

            try:
                y_sim_time = _align_series(sim_species[species_name], sim_time, t_exp)
                model_series_map.setdefault(ds_id, {})[species_name] = y_sim_time
            except Exception as exc:
                if ds_id not in final_dataset_failures:
                    final_dataset_failures[ds_id] = build_fit_diagnostic_from_exception(
                        exc,
                        phase="final_replay",
                        dataset_id=ds_id,
                        parameter_snapshot=failed_param_snapshot,
                    )
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
                if ds_id not in final_dataset_failures:
                    final_dataset_failures[ds_id] = build_fit_diagnostic_from_exception(
                        exc,
                        phase="final_replay",
                        dataset_id=ds_id,
                        parameter_snapshot=failed_param_snapshot,
                    )
                residual_blocks.append(penalty_block)
                residual_series_map.setdefault(ds_id, {})[species_name] = penalty_block
                continue

            if not np.all(np.isfinite(y_sim_resid)):
                if ds_id not in final_dataset_failures:
                    final_dataset_failures[ds_id] = build_fit_diagnostic_from_failure(
                        build_simulation_failure(
                            kind="simulation_error",
                            message=f"Non-finite aligned series for dataset '{ds_id}', species '{species_name}'.",
                        ),
                        phase="final_replay",
                        dataset_id=ds_id,
                        parameter_snapshot=failed_param_snapshot,
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
                t_obs=np.asarray(t_exp, dtype=float).reshape(-1),
                x_name=x_name,
                x_obs=None if x_obs is None else np.asarray(x_obs, dtype=float).reshape(-1),
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
    optimizer_converged = bool(success)
    completion_optimizer_diagnostic = optimizer_diagnostic
    if (
        optimizer_converged
        and metrics_finite
        and not final_dataset_failures
    ):
        completion_optimizer_diagnostic = None
    if final_dataset_failures or not metrics_finite:
        completion_status = "fail"
    elif (not optimizer_converged) or final_dataset_warnings or completion_optimizer_diagnostic is not None:
        completion_status = "warn"
    else:
        completion_status = "ok"

    if completion_status == "ok":
        completion_optimizer_diagnostic = None
        completion_detail_sections: List[FitDetailSection] = []
    else:
        if completion_status == "fail" and (not metrics_finite) and not final_dataset_failures:
            completion_optimizer_diagnostic = ensure_failure_diagnostic(
                completion_optimizer_diagnostic,
                message="Final χ² is non-finite; results are invalid.",
            )
        completion_detail_sections = build_completion_detail_sections(
            status=completion_status,
            optimizer_diagnostic=completion_optimizer_diagnostic,
            dataset_failures=dict(final_dataset_failures),
        )

    completion = GlobalFitCompletion(
        status=completion_status,
        optimizer_converged=optimizer_converged,
        nonfinite_metrics=not metrics_finite,
        optimizer_diagnostic=completion_optimizer_diagnostic,
        dataset_failures=dict(final_dataset_failures),
        dataset_warnings=dict(final_dataset_warnings),
        detail_sections=completion_detail_sections,
    )
    completion_message = completion_result_message(
        base_message=message,
        completion=completion,
        optimizer_diagnostic=completion_optimizer_diagnostic,
        dataset_failures=dict(final_dataset_failures),
        dataset_warnings=dict(final_dataset_warnings),
    )

    if completion.status == "ok":
        logger.info("Global fit complete: R²=%.4f, χ²=%.4e", global_r_squared, global_chi_squared)
    elif completion.status == "warn":
        logger.warning("Global fit complete with warnings: %s", completion_message)
    else:
        logger.warning(
            "Global fit failed: χ²=%.4e datasets_failed=%s",
            global_chi_squared,
            ",".join(sorted(final_dataset_failures.keys())),
        )
    for info in dataset_info:
        logger.debug("  %s: R²=%.4f, RMSE=%.4e", info.dataset_id, info.r_squared, info.rmse)

    return GlobalFitResult(
        shared_params=fitted_params,
        dataset_params=combined_dataset_params,
        uncertainties=uncertainties,
        global_chi_squared=global_chi_squared,
        global_r_squared=global_r_squared,
        dataset_info=dataset_info,
        nfev=nfev,
        message=completion_message,
        completion=completion,
        covariance=covariance,
        objective_residuals=objective_residuals.copy() if objective_residuals is not None else None,
        model_series=model_series_map,
        residual_series=residual_series_map,
        plot_observed_series=plot_observed_series_map,
        plot_model_x=plot_model_x_map,
        plot_model_series=plot_model_series_map,
        alignment_report=dict(alignment_report),
    )
