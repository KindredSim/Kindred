"""
Global fitting execution boundary.

This module owns serial multi-dataset candidate evaluation, final replay,
diagnostics, and result/completion assembly. Public API composition remains in
``global_fitting.fit_global``.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from kindred.core.analysis.parametric_alignment import (
    align_y_on_x_obs,
    align_y_on_x_obs_time_guided,
)
from kindred.core.analysis.dataset_parameter_overrides import (
    FitCoupledVariableParamSpec,
    FitDatasetParameterOverrides,
)
from kindred.core.analysis.fit_dataset_payload import FitDatasetSpec
from kindred.core.exceptions import FittingCancelled, FitSimulationError, SimulationCancelled
from kindred.core.fitting_completion import FitDiagnostic, FitDetailSection, GlobalFitCompletion
from kindred.core.fitting_evaluation import (
    FITTING_PARAM_ORIGIN_CONFIGURED_DATASET,
    FITTING_PARAM_ORIGIN_OPTIMIZER_DATASET,
    FITTING_PARAM_ORIGIN_OPTIMIZER_SHARED,
    FittingSeriesEvaluation,
    evaluate_fitting_series,
)
from kindred.core.objective import ObjectiveContext
from kindred.core.simulation_failure import (
    SimulationFailure,
    build_simulation_failure,
    coerce_simulation_failure,
    simulation_failure_detail_text,
    simulation_failure_from_exception,
    simulation_failure_user_message,
)

logger = logging.getLogger(__name__)

__all__ = [
    "DatasetFitInfo",
    "GlobalFitResult",
    "FitParameterLayout",
    "ObjectiveDatasetInput",
    "DatasetSimulationEvaluation",
    "GlobalFitObjective",
    "build_parameter_layout",
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
    residual_mse: float  # Raw residual mean square
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
    shared_params : dict
        Optimized shared parameters {name: value}
    dataset_params : dict
        Dataset-specific parameters {dataset_id: {name: value}}
    uncertainties : dict, optional
        Parameter uncertainties {name: std_dev}
    global_residual_mse : float
        Raw residual mean square across all datasets
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
    global_residual_mse: float
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
    plot_observed_x: Dict[str, Dict[str, np.ndarray]] = field(default_factory=dict)
    plot_observed_source_rows: Dict[str, Dict[str, tuple[int, ...]]] = field(
        default_factory=dict
    )
    plot_observed_raw_ids: Dict[str, Dict[str, str]] = field(
        default_factory=dict
    )
    plot_model_x_by_species: Dict[str, Dict[str, np.ndarray]] = field(default_factory=dict)
    plot_model_series: Dict[str, Dict[str, np.ndarray]] = field(default_factory=dict)
    plot_model_segment_ids: Dict[str, Dict[str, np.ndarray]] = field(
        default_factory=dict
    )
    plot_model_segment_keys: Dict[
        str,
        Dict[str, tuple[str, ...]],
    ] = field(default_factory=dict)

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
        return "Final residual MSE is non-finite; results are invalid."
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
        value = float(raw_weights.get(name, 1.0))
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(
                f"Target weight for {name!r} must be finite and positive."
            )
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
    lo = float(np.percentile(arr, 5))
    hi = float(np.percentile(arr, 95))
    span = hi - lo
    if not np.isfinite(span) or span < 0.0:
        return 0.0
    return span


def _extract_simulation_payload(
    simulation: FittingSeriesEvaluation,
) -> Tuple[
    Optional[np.ndarray],
    Dict[str, np.ndarray],
    np.ndarray,
    tuple[str, ...],
    np.ndarray,
]:
    """Return time, series, and discontinuity segment identity."""
    payload = simulation.series
    t_sim = np.asarray(payload.t, dtype=float).reshape(-1)
    return (
        t_sim if t_sim.size else None,
        dict(payload.species),
        np.asarray(simulation.segment_ids, dtype=np.int64).reshape(-1),
        tuple(simulation.segment_keys),
        np.asarray(
            simulation.pre_boundary_sample_mask,
            dtype=bool,
        ).reshape(-1),
    )


def _align_series(
    series: np.ndarray,
    sim_time: Optional[np.ndarray],
    target_time: np.ndarray,
    segment_ids: np.ndarray,
    segment_keys: Sequence[str],
    pre_boundary_sample_mask: np.ndarray,
    target_source_rows: Sequence[int],
    pre_boundary_segment_keys_by_source_row: Mapping[int, str],
) -> np.ndarray:
    """Align time-fit values without interpolating across a discontinuity."""
    values = np.asarray(series, dtype=float).reshape(-1)
    if sim_time is not None and values.size == sim_time.size:
        sim_t = np.asarray(sim_time, dtype=float).reshape(-1)
        target_t = np.asarray(target_time, dtype=float).reshape(-1)
        source_rows = tuple(int(value) for value in target_source_rows)
        requested_pre = {
            int(row): str(key or "").strip()
            for row, key
            in pre_boundary_segment_keys_by_source_row.items()
        }
        if (
            len(source_rows) != target_t.size
            or any(not key for key in requested_pre.values())
            or any(row not in source_rows for row in requested_pre)
        ):
            raise FitSimulationError(
                "Explicit pre-boundary Fit identity is missing or "
                "misaligned with captured source rows.",
                details={"fatal": True, "stage": "fit_capture"},
            )
        segments = np.asarray(segment_ids, dtype=np.int64).reshape(-1)
        if segments.size != sim_t.size:
            raise FitSimulationError(
                "Discontinuity segment identity must align with simulation time.",
                details={
                    "fatal": True,
                    "stage": "intervention_execution",
                },
            )
        keys = tuple(str(value or "").strip() for value in segment_keys)
        if len(keys) != sim_t.size or any(not value for value in keys):
            raise FitSimulationError(
                "Exact segment keys must align with simulation time.",
                details={
                    "fatal": True,
                    "stage": "intervention_execution",
                },
            )
        pre_boundary = np.asarray(
            pre_boundary_sample_mask,
            dtype=bool,
        ).reshape(-1)
        if pre_boundary.size != sim_t.size:
            raise FitSimulationError(
                "Pre-boundary sample identity must align with simulation time.",
                details={
                    "fatal": True,
                    "stage": "intervention_execution",
                },
            )
        if (
            not requested_pre
            and sim_t.size == target_t.size
            and np.array_equal(sim_t, target_t)
        ):
            return values
        if sim_t.size < 2 or np.any(np.diff(sim_t) < 0.0):
            raise ValueError(
                "Simulation time must be ordered for fitting alignment."
            )
        aligned = np.empty(target_t.size, dtype=float)
        for index, time_value in enumerate(target_t):
            exact = np.flatnonzero(sim_t == time_value)
            requested_key = requested_pre.get(source_rows[index])
            if exact.size:
                chosen = (
                    exact[-1:]
                    if requested_key is None
                    else np.asarray(
                        [
                            candidate
                            for candidate in exact
                            if (
                                pre_boundary[int(candidate)]
                                and keys[int(candidate)] == requested_key
                            )
                        ],
                        dtype=np.int64,
                    )
                )
                if chosen.size != 1:
                    raise FitSimulationError(
                        "Explicit pre-boundary Fit segment key is missing "
                        "or ambiguous in the exact simulation result.",
                        details={
                            "fatal": True,
                            "stage": "intervention_execution",
                        },
                    )
                aligned[index] = values[int(chosen[0])]
                continue
            if requested_key is not None:
                raise FitSimulationError(
                    "Explicit pre-boundary Fit identity requires an exact "
                    "boundary-time simulation sample.",
                    details={
                        "fatal": True,
                        "stage": "intervention_execution",
                    },
                )
            right = int(np.searchsorted(sim_t, time_value, side="right"))
            left = right - 1
            if left < 0 or right >= sim_t.size:
                raise ValueError(
                    "Dataset time lies outside the simulation time span."
                )
            if (
                segments[left] != segments[right]
                or keys[left] != keys[right]
            ):
                raise FitSimulationError(
                    "Time-fit interpolation cannot cross an intervention "
                    "discontinuity or integration-segment boundary."
                )
            width = float(sim_t[right] - sim_t[left])
            if width <= 0.0:
                raise ValueError(
                    "Simulation time must increase inside each fitting segment."
                )
            fraction = float((time_value - sim_t[left]) / width)
            aligned[index] = float(
                values[left] + fraction * (values[right] - values[left])
            )
        return aligned
    if sim_time is None:
        raise ValueError("Simulation output must include time coordinates for fitting alignment.")
    raise ValueError(
        f"Simulation output has {values.size} values but {np.asarray(sim_time).size} time points.",
    )


def _align_segment_ids(
    sim_time: np.ndarray,
    target_time: np.ndarray,
    segment_ids: np.ndarray,
) -> np.ndarray:
    """Project exact model segment identity to aligned observation times."""

    sim_t = np.asarray(sim_time, dtype=float).reshape(-1)
    target_t = np.asarray(target_time, dtype=float).reshape(-1)
    segments = np.asarray(segment_ids, dtype=np.int64).reshape(-1)
    if sim_t.size != segments.size:
        raise ValueError(
            "Discontinuity segment identity must align with simulation time."
        )
    aligned = np.empty(target_t.size, dtype=np.int64)
    for index, time_value in enumerate(target_t):
        exact = np.flatnonzero(sim_t == time_value)
        if exact.size:
            aligned[index] = segments[int(exact[-1])]
            continue
        right = int(np.searchsorted(sim_t, time_value, side="right"))
        left = right - 1
        if left < 0 or right >= sim_t.size or segments[left] != segments[right]:
            raise FitSimulationError(
                "Time-fit segment projection cannot cross an intervention "
                "discontinuity."
            )
        aligned[index] = segments[left]
    aligned.setflags(write=False)
    return aligned


def _align_segment_keys(
    sim_time: np.ndarray,
    target_time: np.ndarray,
    segment_ids: np.ndarray,
    segment_keys: Sequence[str],
) -> tuple[str, ...]:
    """Project exact opaque segment identity to aligned observation times."""

    sim_t = np.asarray(sim_time, dtype=float).reshape(-1)
    target_t = np.asarray(target_time, dtype=float).reshape(-1)
    segments = np.asarray(segment_ids, dtype=np.int64).reshape(-1)
    keys = tuple(str(value or "").strip() for value in segment_keys)
    if (
        sim_t.size != segments.size
        or len(keys) != sim_t.size
        or any(not value for value in keys)
    ):
        raise ValueError(
            "Exact segment identity must align with simulation time."
        )
    aligned: list[str] = []
    for time_value in target_t:
        exact = np.flatnonzero(sim_t == time_value)
        if exact.size:
            aligned.append(keys[int(exact[-1])])
            continue
        right = int(np.searchsorted(sim_t, time_value, side="right"))
        left = right - 1
        if (
            left < 0
            or right >= sim_t.size
            or segments[left] != segments[right]
            or keys[left] != keys[right]
        ):
            raise FitSimulationError(
                "Time-fit segment projection cannot cross an intervention "
                "discontinuity or integration-segment boundary."
            )
        aligned.append(keys[left])
    return tuple(aligned)


def _objective_dataset_weight_multiplier(weight: float) -> float:
    value = float(weight)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("Objective dataset weight must be finite and positive.")
    return float(np.sqrt(value))


def _payload_observations(
    payload: FitDatasetSpec,
) -> Dict[str, Dict[str, Any]]:
    return {
        str(name): {
            "t": np.asarray(series.t, dtype=float).reshape(-1),
            "y": np.asarray(series.y, dtype=float).reshape(-1),
            "source_rows": np.asarray(
                series.source_rows,
                dtype=np.int64,
            ).reshape(-1),
            "pre_boundary_segment_keys_by_source_row": dict(
                series.pre_boundary_segment_keys_by_source_row
            ),
        }
        for name, series in payload.observations.items()
    }


def _payload_x_obs_by_species(payload: FitDatasetSpec) -> Dict[str, np.ndarray]:
    return {
        str(name): np.asarray(values, dtype=float).reshape(-1)
        for name, values in payload.x_obs_by_species.items()
        if str(name).strip()
    }


@dataclass(frozen=True)
class FitParameterLayout:
    param_names: List[str]
    shared_log10: Dict[str, bool]
    dataset_var_order: List[Tuple[str, str]]
    dataset_var_index: Dict[str, Dict[str, int]]
    dataset_var_log10: Dict[Tuple[str, str], bool]
    coupled_var_order: List[FitCoupledVariableParamSpec]
    coupled_var_index: Dict[str, int]
    x0: np.ndarray
    lower: np.ndarray
    upper: np.ndarray

    def opt_param_keys(self) -> List[str]:
        keys: List[str] = list(self.param_names)
        keys.extend([f"{ds_id}::{param_name}" for (ds_id, param_name) in self.dataset_var_order])
        keys.extend(item.variable_id for item in self.coupled_var_order)
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

    def coupled_params_for_dataset(
        self,
        ds_id: str,
        vector: np.ndarray,
    ) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for spec in self.coupled_var_order:
            if ds_id not in spec.dataset_ids:
                continue
            index = self.coupled_var_index[spec.variable_id]
            raw_value = float(vector[index])
            out[spec.parameter_name] = (
                float(10.0**raw_value) if spec.log10 else raw_value
            )
        return out


def build_parameter_layout(
    *,
    payloads: List[FitDatasetSpec],
    shared_params: Dict[str, float],
    dataset_overrides: Sequence[FitDatasetParameterOverrides],
    coupled_variables: Sequence[FitCoupledVariableParamSpec],
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

    overrides_by_id = {item.dataset_id: item for item in dataset_overrides}
    for payload in payloads:
        ds_id = payload.dataset_id
        specs = overrides_by_id.get(
            ds_id,
            FitDatasetParameterOverrides(dataset_id=ds_id),
        ).variable_params
        if not specs:
            continue
        dataset_var_index.setdefault(ds_id, {})
        for spec in specs:
            param_name = spec.name
            log10_flag = bool(spec.log10)
            dataset_var_order.append((ds_id, param_name))
            dataset_var_index[ds_id][param_name] = len(param_names) + len(dataset_var_inits)
            init_val = float(spec.initial)
            min_val = float(spec.minimum)
            max_val = float(spec.maximum)
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

    coupled_order = list(coupled_variables)
    coupled_ids = [item.variable_id for item in coupled_order]
    if len(coupled_ids) != len(set(coupled_ids)):
        raise ValueError("Coupled Fit variable IDs must be unique.")
    known_dataset_ids = {item.dataset_id for item in payloads}
    for spec in coupled_order:
        unknown = set(spec.dataset_ids) - known_dataset_ids
        if unknown:
            raise ValueError(
                f"Coupled Fit variable {spec.variable_id!r} references unknown "
                f"datasets: {', '.join(sorted(unknown))}."
            )
    coupled_initials = [
        math.log10(item.initial) if item.log10 else float(item.initial)
        for item in coupled_order
    ]
    x0 = np.array(
        shared_initials + dataset_var_inits + coupled_initials,
        dtype=float,
    )

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

    coupled_bounds = [
        (
            math.log10(item.minimum) if item.log10 else float(item.minimum),
            math.log10(item.maximum) if item.log10 else float(item.maximum),
        )
        for item in coupled_order
    ]
    lower = np.array(
        shared_lower
        + [b[0] for b in dataset_var_bounds]
        + [b[0] for b in coupled_bounds],
        dtype=float,
    )
    upper = np.array(
        shared_upper
        + [b[1] for b in dataset_var_bounds]
        + [b[1] for b in coupled_bounds],
        dtype=float,
    )
    coupled_start = len(param_names) + len(dataset_var_inits)
    coupled_index = {
        item.variable_id: coupled_start + index
        for index, item in enumerate(coupled_order)
    }

    return FitParameterLayout(
        param_names=param_names,
        shared_log10=shared_log10,
        dataset_var_order=dataset_var_order,
        dataset_var_index=dataset_var_index,
        dataset_var_log10=dataset_var_log10,
        coupled_var_order=coupled_order,
        coupled_var_index=coupled_index,
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
        segment_ids: np.ndarray,
        segment_keys: Sequence[str],
        dataset_label: str,
        x_name: str,
    ) -> None:
        mode_norm = str(mode or "").strip().lower()
        if mode_norm not in {"monotone", "time_guided"}:
            raise FitSimulationError(
                f"Dataset '{dataset_label}': Fit requires a locked parametric-X "
                f"mode of 'monotone' or 'time_guided'; got {mode!r}."
            )
        self._mode = mode_norm
        self._t_obs = t_obs
        self._x_obs = x_obs
        self._t_sim = t_sim
        self._x_model = np.asarray(x_model, dtype=float)
        self._segment_ids = np.asarray(segment_ids, dtype=np.int64)
        self._segment_keys = tuple(
            str(value or "").strip() for value in segment_keys
        )
        self._dataset_label = str(dataset_label)
        self._x_name = str(x_name)

    def align(self, *, y_model: np.ndarray, y_name: str) -> np.ndarray:
        if self._mode == "monotone":
            return align_y_on_x_obs(
                t_obs=self._t_obs,
                x_obs=self._x_obs,
                t_sim=self._t_sim,
                x_model=self._x_model,
                y_model=np.asarray(y_model, dtype=float),
                segment_ids=self._segment_ids,
                segment_keys=self._segment_keys,
                dataset_label=self._dataset_label,
                x_name=self._x_name,
                y_name=y_name,
            )
        return align_y_on_x_obs_time_guided(
            t_obs=self._t_obs,
            x_obs=self._x_obs,
            t_sim=self._t_sim,
            x_model=self._x_model,
            y_model=np.asarray(y_model, dtype=float),
            segment_ids=self._segment_ids,
            segment_keys=self._segment_keys,
            dataset_label=self._dataset_label,
            x_name=self._x_name,
            y_name=y_name,
        )


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
    segment_ids: Optional[np.ndarray] = None
    segment_keys: tuple[str, ...] | None = None
    pre_boundary_sample_mask: Optional[np.ndarray] = None
    error: Optional[BaseException] = None
    error_provenance: Optional[Dict[str, Any]] = None
    final_error_message: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "sim_species", dict(self.sim_species))
        if self.segment_ids is not None:
            segments = np.asarray(
                self.segment_ids,
                dtype=np.int64,
            ).reshape(-1)
            keys = tuple(
                str(value or "").strip()
                for value in (self.segment_keys or ())
            )
            expected_size = (
                0
                if self.sim_time is None
                else np.asarray(self.sim_time).reshape(-1).size
            )
            if (
                segments.size != expected_size
                or len(keys) != expected_size
                or any(not value for value in keys)
            ):
                raise ValueError(
                    "Fit simulation segment identity must align with time."
                )
            pre_boundary = np.asarray(
                self.pre_boundary_sample_mask,
                dtype=bool,
            ).reshape(-1)
            if pre_boundary.size != expected_size:
                raise ValueError(
                    "Fit pre-boundary identity must align with time."
                )
            object.__setattr__(self, "segment_ids", segments)
            object.__setattr__(self, "segment_keys", keys)
            object.__setattr__(
                self,
                "pre_boundary_sample_mask",
                pre_boundary,
            )
        elif (
            self.segment_keys is not None
            or self.pre_boundary_sample_mask is not None
        ):
            raise ValueError(
                "Fit simulation segment/pre-boundary identity requires "
                "discontinuity identity."
            )
        if self.error_provenance is not None:
            object.__setattr__(self, "error_provenance", dict(self.error_provenance))


def dataset_evaluation_is_fatal(result: DatasetSimulationEvaluation) -> bool:
    return isinstance(result.error, FitSimulationError) and bool(
        getattr(result.error, "details", {}).get("fatal")
    )


def evaluate_dataset_simulation(
    fit_evaluators: Mapping[str, object],
    item: ObjectiveDatasetInput,
    *,
    cancellation_check: Optional[Callable[[], bool]] = None,
) -> DatasetSimulationEvaluation:
    ds_id = item.payload.dataset_id
    try:
        _raise_if_fitting_cancel_requested(cancellation_check)
        evaluator = fit_evaluators.get(ds_id)
        if evaluator is None:
            raise FitSimulationError(
                f"No fitting evaluator exists for dataset {ds_id!r}.",
                details={"fatal": True, "stage": "fit_capture"},
            )
        sim_result = evaluate_fitting_series(
            evaluator,
            item.full_params,
            origins=item.parameter_origins,
            failed_params=item.failed_param_snapshot,
        )
        (
            sim_time,
            sim_species,
            segment_ids,
            segment_keys,
            pre_boundary_sample_mask,
        ) = _extract_simulation_payload(sim_result)
        if item.payload.x_name != "t":
            keep = ~pre_boundary_sample_mask
            sim_time = None if sim_time is None else sim_time[keep]
            sim_species = {
                name: np.asarray(values, dtype=float).reshape(-1)[keep]
                for name, values in sim_species.items()
            }
            segment_ids = segment_ids[keep]
            segment_keys = tuple(
                key
                for key, retain in zip(
                    segment_keys,
                    keep,
                    strict=True,
                )
                if retain
            )
            pre_boundary_sample_mask = pre_boundary_sample_mask[keep]
        _raise_if_fitting_cancel_requested(cancellation_check)
        return DatasetSimulationEvaluation(
            index=int(item.index),
            sim_time=sim_time,
            sim_species=sim_species,
            segment_ids=segment_ids,
            segment_keys=segment_keys,
            pre_boundary_sample_mask=pre_boundary_sample_mask,
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
    fit_evaluators: Mapping[str, object],
    items: Sequence[ObjectiveDatasetInput],
    *,
    cancellation_check: Optional[Callable[[], bool]] = None,
    stop_on_fatal: bool = True,
) -> List[DatasetSimulationEvaluation]:
    if not items:
        return []
    return evaluate_dataset_simulations_serial(
        fit_evaluators,
        items,
        cancellation_check=cancellation_check,
        stop_on_fatal=stop_on_fatal,
    )


def evaluate_dataset_simulations_serial(
    fit_evaluators: Mapping[str, object],
    items: Sequence[ObjectiveDatasetInput],
    *,
    cancellation_check: Optional[Callable[[], bool]],
    stop_on_fatal: bool,
) -> List[DatasetSimulationEvaluation]:
    results = []
    for item in items:
        _raise_if_fitting_cancelled(cancellation_check)
        result = evaluate_dataset_simulation(
            fit_evaluators,
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
        fit_evaluators: Mapping[str, object],
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
        self._fit_evaluators = dict(fit_evaluators)
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
            coupled_dataset_params = self._layout.coupled_params_for_dataset(
                ds_id,
                params,
            )
            raw_full_params: Dict[str, float] = {}
            parameter_origins: Dict[str, str] = {}
            for values, origin in (
                (param_dict, FITTING_PARAM_ORIGIN_OPTIMIZER_SHARED),
                (self._dataset_params.get(ds_id, {}), FITTING_PARAM_ORIGIN_CONFIGURED_DATASET),
                (optimizer_dataset_params, FITTING_PARAM_ORIGIN_OPTIMIZER_DATASET),
                (coupled_dataset_params, FITTING_PARAM_ORIGIN_OPTIMIZER_DATASET),
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
                self._fit_evaluators,
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
            observations = _payload_observations(payload)
            x_name = payload.x_name
            x_obs_by_species = _payload_x_obs_by_species(payload) if x_name != "t" else {}
            x_mode = payload.x_mode
            weight = self._weights.get(ds_id, 1.0)
            target_weights = dict(payload.target_weights)
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
                for species_name in species_list:
                    target_weight = float(target_multipliers.get(species_name, 1.0))
                    dataset_weight = _objective_dataset_weight_multiplier(weight)
                    y_exp = np.asarray(observations[species_name]["y"], dtype=float).reshape(-1)
                    all_residuals.extend(
                        (float(dataset_weight) * float(target_weight) * self._penalty_value)
                        * np.ones_like(y_exp, dtype=float)
                    )
                continue

            sim_time = evaluation.sim_time
            sim_species = evaluation.sim_species
            segment_ids = evaluation.segment_ids
            segment_keys = evaluation.segment_keys
            pre_boundary_sample_mask = (
                evaluation.pre_boundary_sample_mask
            )
            if (
                segment_ids is None
                or segment_keys is None
                or pre_boundary_sample_mask is None
            ):
                raise FitSimulationError(
                    f"Dataset {ds_id!r}: Fit simulation lost exact segment "
                    "identity.",
                    failed_params=failed_param_snapshot,
                )

            aligners: Dict[str, _ParametricXAligner] = {}
            for species_name in species_list:
                _raise_if_fitting_cancelled(self._cancellation_check)
                species_obs = observations[species_name]
                obs_t = np.asarray(species_obs["t"], dtype=float).reshape(-1)
                y_exp = np.asarray(species_obs["y"], dtype=float).reshape(-1)
                species_x_obs = (
                    np.asarray(x_obs_by_species.get(species_name), dtype=float).reshape(-1)
                    if species_name in x_obs_by_species
                    else None
                )
                target_weight = float(target_multipliers.get(str(species_name), 1.0))
                effective_weight = _objective_dataset_weight_multiplier(weight) * float(target_weight)
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
                        y_sim = _align_series(
                            model_series,
                            sim_time,
                            obs_t,
                            np.asarray(segment_ids, dtype=np.int64),
                            segment_keys,
                            pre_boundary_sample_mask,
                            species_obs["source_rows"],
                            species_obs[
                                "pre_boundary_segment_keys_by_source_row"
                            ],
                        )
                    else:
                        x_model_series = sim_species.get(x_name)
                        if x_model_series is None:
                            raise FitSimulationError(
                                f"X series '{x_name}' missing in simulation result for dataset '{ds_id}'.",
                                failed_params=failed_param_snapshot,
                            )
                        if species_x_obs is None:
                            raise FitSimulationError(
                                f"Dataset '{ds_id}' species '{species_name}' is missing x observations for X='{x_name}'.",
                                failed_params=failed_param_snapshot,
                            )
                        aligner = aligners.get(species_name)
                        if aligner is None:
                            aligner = _ParametricXAligner(
                                mode=x_mode,
                                t_obs=obs_t,
                                x_obs=species_x_obs,
                                t_sim=sim_time,
                                x_model=np.asarray(x_model_series, dtype=float),
                                segment_ids=np.asarray(segment_ids, dtype=np.int64),
                                segment_keys=segment_keys,
                                dataset_label=ds_id,
                                x_name=x_name,
                            )
                            aligners[species_name] = aligner
                        y_sim = aligner.align(
                            y_model=np.asarray(model_series, dtype=float),
                            y_name=species_name,
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
                for spec in self._layout.coupled_var_order:
                    raw_val = float(params[self._layout.coupled_var_index[spec.variable_id]])
                    callback_params[spec.variable_id] = (
                        float(10.0**raw_val) if spec.log10 else raw_val
                    )
                self._progress_callback(self._iteration, cost, callback_params)
        _raise_if_fitting_cancelled(self._cancellation_check)

        return residuals


def assemble_global_fit_result(
    *,
    fit_evaluators: Mapping[str, object],
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
    plot_observed_x_map: Dict[str, Dict[str, np.ndarray]] = {}
    plot_observed_source_rows_map: Dict[
        str, Dict[str, tuple[int, ...]]
    ] = {}
    plot_observed_raw_ids_map: Dict[str, Dict[str, str]] = {}
    plot_model_x_by_species_map: Dict[str, Dict[str, np.ndarray]] = {}
    plot_model_series_map: Dict[str, Dict[str, np.ndarray]] = {}
    plot_model_segment_ids_map: Dict[str, Dict[str, np.ndarray]] = {}
    plot_model_segment_keys_map: Dict[
        str,
        Dict[str, tuple[str, ...]],
    ] = {}
    final_dataset_failures: Dict[str, FitDiagnostic] = {}
    final_dataset_warnings: Dict[str, str] = {}

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
            coupled_names = {
                item.parameter_name
                for item in layout.coupled_var_order
                if ds_id in item.dataset_ids
            }
            if name in layout.dataset_var_index.get(ds_id, {}) or name in coupled_names:
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
            fit_evaluators,
            dataset_inputs,
            cancellation_check=cancellation_check,
            stop_on_fatal=False,
        )
    }

    for item in dataset_inputs:
        payload = item.payload
        ds_id = payload.dataset_id
        species_list = payload.species_list
        observations = _payload_observations(payload)
        observed_times = np.concatenate(
            [
                np.asarray(observations[name]["t"], dtype=float).reshape(-1)
                for name in species_list
            ]
        )
        x_name = payload.x_name
        x_obs_by_species = _payload_x_obs_by_species(payload) if x_name != "t" else {}
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
        segment_ids = evaluation.segment_ids
        segment_keys = evaluation.segment_keys
        pre_boundary_sample_mask = evaluation.pre_boundary_sample_mask

        plot_mask: Optional[np.ndarray] = None
        plot_x = np.asarray([], dtype=float)
        if x_name != "t" and sim_time is not None and isinstance(sim_species, dict) and x_name in sim_species:
            t0 = float(np.min(observed_times))
            t1 = float(np.max(observed_times))
            if t0 > t1:
                t0, t1 = t1, t0
            t_scale = max(1.0, abs(t0), abs(t1))
            t_pad = 1e-12 * t_scale
            plot_mask = (sim_time >= (t0 - t_pad)) & (sim_time <= (t1 + t_pad))
            if np.any(plot_mask):
                plot_x = np.asarray(
                    sim_species[x_name],
                    dtype=float,
                ).reshape(-1)[plot_mask]

        residual_blocks: List[np.ndarray] = []
        exp_blocks: List[np.ndarray] = []
        aligners: Dict[str, _ParametricXAligner] = {}

        for species_name in species_list:
            species_obs = observations[species_name]
            obs_t = np.asarray(species_obs["t"], dtype=float).reshape(-1)
            y_exp = np.asarray(species_obs["y"], dtype=float).reshape(-1)
            species_x_obs = (
                np.asarray(x_obs_by_species.get(species_name), dtype=float).reshape(-1)
                if species_name in x_obs_by_species
                else None
            )
            exp_blocks.append(y_exp)
            plot_observed_series_map.setdefault(ds_id, {})[species_name] = y_exp.copy()
            plot_observed_x_map.setdefault(ds_id, {})[species_name] = (
                obs_t.copy()
                if x_name == "t"
                else np.asarray(species_x_obs, dtype=float).reshape(-1).copy()
            )
            captured_observation = payload.observations[species_name]
            plot_observed_source_rows_map.setdefault(ds_id, {})[
                species_name
            ] = tuple(captured_observation.source_rows)
            plot_observed_raw_ids_map.setdefault(ds_id, {})[
                species_name
            ] = captured_observation.raw_observation_id
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
                y_sim_time = _align_series(
                    sim_species[species_name],
                    sim_time,
                    obs_t,
                    np.asarray(segment_ids, dtype=np.int64),
                    tuple(segment_keys or ()),
                    np.asarray(
                        pre_boundary_sample_mask,
                        dtype=bool,
                    ),
                    species_obs["source_rows"],
                    species_obs[
                        "pre_boundary_segment_keys_by_source_row"
                    ],
                )
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
                    if species_x_obs is None:
                        raise FitSimulationError(
                            f"Dataset '{ds_id}' species '{species_name}' is missing x observations for X='{x_name}'.",
                            failed_params=failed_param_snapshot,
                        )
                    aligner = aligners.get(species_name)
                    if aligner is None:
                        aligner = _ParametricXAligner(
                            mode=x_mode,
                            t_obs=obs_t,
                            x_obs=species_x_obs,
                            t_sim=sim_time,
                            x_model=np.asarray(x_model_series, dtype=float),
                            segment_ids=np.asarray(segment_ids, dtype=np.int64),
                            segment_keys=tuple(segment_keys or ()),
                            dataset_label=ds_id,
                            x_name=x_name,
                        )
                        aligners[species_name] = aligner
                    y_sim_resid = aligner.align(
                        y_model=np.asarray(sim_species[species_name], dtype=float),
                        y_name=species_name,
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
                plot_model_series_map.setdefault(ds_id, {})[species_name] = np.asarray(y_sim_resid, dtype=float).reshape(-1)
                plot_model_x_by_species_map.setdefault(ds_id, {})[species_name] = obs_t.copy()
                plot_model_segment_ids_map.setdefault(ds_id, {})[
                    species_name
                ] = _align_segment_ids(
                    np.asarray(sim_time, dtype=float),
                    obs_t,
                    np.asarray(segment_ids, dtype=np.int64),
                )
                plot_model_segment_keys_map.setdefault(ds_id, {})[
                    species_name
                ] = _align_segment_keys(
                    np.asarray(sim_time, dtype=float),
                    obs_t,
                    np.asarray(segment_ids, dtype=np.int64),
                    tuple(segment_keys or ()),
                )
            else:
                sliced = None
                if plot_mask is not None and isinstance(sim_species, dict) and species_name in sim_species:
                    values = np.asarray(sim_species[species_name], dtype=float).reshape(-1)
                    try:
                        sliced = np.asarray(values[plot_mask], dtype=float).reshape(-1)
                    except (IndexError, TypeError, ValueError):
                        sliced = None
                if sliced is not None and plot_x.size == sliced.size:
                    plot_model_series_map.setdefault(ds_id, {})[species_name] = sliced
                    plot_model_x_by_species_map.setdefault(ds_id, {})[species_name] = plot_x.copy()
                    plot_model_segment_ids_map.setdefault(ds_id, {})[
                        species_name
                    ] = np.asarray(
                        segment_ids,
                        dtype=np.int64,
                    ).reshape(-1)[plot_mask].copy()
                    plot_model_segment_keys_map.setdefault(ds_id, {})[
                        species_name
                    ] = tuple(
                        key
                        for key, selected in zip(
                            tuple(segment_keys or ()),
                            np.asarray(plot_mask, dtype=bool),
                            strict=True,
                        )
                        if selected
                    )
                else:
                    full_model_y = np.asarray(
                        sim_species[species_name],
                        dtype=float,
                    ).reshape(-1)
                    full_model_x = np.asarray(
                        sim_species[x_name],
                        dtype=float,
                    ).reshape(-1)
                    full_segments = np.asarray(
                        segment_ids,
                        dtype=np.int64,
                    ).reshape(-1)
                    if not (
                        full_model_x.size
                        == full_model_y.size
                        == full_segments.size
                    ):
                        raise FitSimulationError(
                            f"Dataset {ds_id!r}: terminal Parametric model "
                            "axes lost segment-aligned identity."
                        )
                    plot_model_series_map.setdefault(ds_id, {})[
                        species_name
                    ] = full_model_y.copy()
                    plot_model_x_by_species_map.setdefault(ds_id, {})[
                        species_name
                    ] = full_model_x.copy()
                    plot_model_segment_ids_map.setdefault(ds_id, {})[
                        species_name
                    ] = full_segments.copy()
                    plot_model_segment_keys_map.setdefault(ds_id, {})[
                        species_name
                    ] = tuple(segment_keys or ())

        residuals = np.concatenate(residual_blocks) if residual_blocks else np.asarray([], dtype=float)
        y_exp_all = np.concatenate(exp_blocks) if exp_blocks else np.asarray([], dtype=float)

        ss_res = (
            float(np.sum(residuals**2)) if residuals.size else float((penalty_value**2) * float(y_exp_all.size))
        )
        ss_tot = float(np.sum((y_exp_all - float(np.mean(y_exp_all)))**2)) if y_exp_all.size else 0.0

        r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        residual_mse = ss_res / residuals.size if residuals.size else float(penalty_value**2)
        rmse = np.sqrt(np.mean(residuals**2)) if residuals.size else float(penalty_value)
        mae = np.mean(np.abs(residuals)) if residuals.size else float(penalty_value)

        dataset_info.append(
            DatasetFitInfo(
                dataset_id=ds_id,
                r_squared=r_squared,
                residual_mse=residual_mse,
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

    global_r_squared = 1.0 - (total_ss_res / total_ss_tot) if total_ss_tot > 0 else 0.0
    global_residual_mse = total_ss_res / total_points if total_points > 0 else float(penalty_value**2)

    metrics_finite = bool(
        np.isfinite(total_ss_res) and np.isfinite(global_r_squared) and np.isfinite(global_residual_mse)
    )
    optimizer_converged = bool(success)
    completion_optimizer_diagnostic = optimizer_diagnostic
    if (
        optimizer_converged
        and metrics_finite
        and not final_dataset_failures
        and completion_optimizer_diagnostic is not None
        and not bool(
            (
                completion_optimizer_diagnostic.failure.get("details", {})
                if isinstance(completion_optimizer_diagnostic.failure, Mapping)
                else {}
            ).get("final_report_warning")
        )
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
                message="Final residual MSE is non-finite; results are invalid.",
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
        logger.info("Global fit complete: R²=%.4f, residual MSE=%.4e", global_r_squared, global_residual_mse)
    elif completion.status == "warn":
        logger.warning("Global fit complete with warnings: %s", completion_message)
    else:
        logger.warning(
            "Global fit failed: residual MSE=%.4e datasets_failed=%s",
            global_residual_mse,
            ",".join(sorted(final_dataset_failures.keys())),
        )
    for info in dataset_info:
        logger.debug("  %s: R²=%.4f, RMSE=%.4e", info.dataset_id, info.r_squared, info.rmse)

    return GlobalFitResult(
        shared_params=fitted_params,
        dataset_params=combined_dataset_params,
        uncertainties=uncertainties,
        global_residual_mse=global_residual_mse,
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
        plot_observed_x=plot_observed_x_map,
        plot_observed_source_rows=plot_observed_source_rows_map,
        plot_observed_raw_ids=plot_observed_raw_ids_map,
        plot_model_x_by_species=plot_model_x_by_species_map,
        plot_model_series=plot_model_series_map,
        plot_model_segment_ids=plot_model_segment_ids_map,
        plot_model_segment_keys=plot_model_segment_keys_map,
    )
