"""
Global fitting execution boundary.

This module owns serial multi-dataset candidate evaluation, final replay,
diagnostics, and result/completion assembly. Public API composition remains in
``global_fitting.fit_global``.
"""

from __future__ import annotations

import copy
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from kindred.core.analysis.parametric_alignment import (
    DeterministicParametricResolutionError,
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
    "select_fit_rows",
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


def _is_expected_candidate_domain_failure(exc: BaseException) -> bool:
    """Accept penalties only for the evaluator's explicit candidate domain."""

    return bool(
        isinstance(exc, FitSimulationError)
        and isinstance(exc.details, Mapping)
        and exc.details.get("candidate_domain") is True
        and exc.details.get("fatal") is False
    )


def _candidate_trajectory_failure(
    exc: FitSimulationError,
    *,
    stage: str,
) -> FitSimulationError:
    """Classify one proven optimizer-trajectory outcome, never configuration."""

    return FitSimulationError(
        exc.message,
        failed_params=exc.failed_params,
        details={
            "fatal": False,
            "candidate_domain": True,
            "stage": str(stage),
        },
        context=exc.context,
    )

@dataclass
class DatasetFitInfo:
    """Fit statistics for a single dataset within global fit."""
    dataset_id: str  # Identifier for this dataset
    r_squared_by_series: Dict[str, float]  # Ordinary, unweighted R² by series
    weighted_residual_sse: float  # Exact contribution to the objective
    weighted_residual_mse: float  # Exact objective contribution per observation
    weighted_rmse: float
    weighted_mae: float
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
    weighted_residual_sse : float
        Exact sum of squares minimized by the weighted objective
    weighted_residual_mse : float
        Exact weighted objective sum of squares per observation
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
    weighted_residual_sse: float
    weighted_residual_mse: float
    dataset_info: List[DatasetFitInfo]
    nfev: int
    message: str
    completion: GlobalFitCompletion
    covariance: Optional[np.ndarray] = None
    objective_residuals: Optional[np.ndarray] = None
    model_series: Dict[str, Dict[str, np.ndarray]] = field(default_factory=dict)
    model_x_by_species: Dict[str, Dict[str, np.ndarray]] = field(default_factory=dict)
    model_segment_ids: Dict[str, Dict[str, np.ndarray]] = field(default_factory=dict)
    model_segment_keys: Dict[str, Dict[str, tuple[str, ...]]] = field(default_factory=dict)
    model_discontinuity_ids: Dict[str, Dict[str, np.ndarray]] = field(default_factory=dict)
    model_selection_provenance: Dict[str, Dict[str, tuple[str, ...]]] = field(default_factory=dict)
    model_observation_ids: Dict[str, Dict[str, tuple[str, ...]]] = field(default_factory=dict)
    model_raw_columns: Dict[str, Dict[str, tuple[str, ...]]] = field(default_factory=dict)
    model_source_rows: Dict[str, Dict[str, tuple[int, ...]]] = field(default_factory=dict)
    solver_provenance_by_dataset: Dict[str, Dict[str, Any]] = field(
        default_factory=dict
    )
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
    plot_model_discontinuity_ids: Dict[str, Dict[str, np.ndarray]] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if not isinstance(self.completion, GlobalFitCompletion):
            raise TypeError("GlobalFitResult.completion must be a GlobalFitCompletion")
        solver_provenance: Dict[str, Dict[str, Any]] = {}
        for raw_dataset_id, provenance in self.solver_provenance_by_dataset.items():
            dataset_id = str(raw_dataset_id or "").strip()
            if not dataset_id or dataset_id in solver_provenance:
                raise ValueError(
                    "Fit solver provenance requires distinct stable dataset IDs."
                )
            if not isinstance(provenance, Mapping):
                raise TypeError("Fit solver provenance entries must be mappings.")
            solver_provenance[dataset_id] = copy.deepcopy(dict(provenance))
        object.__setattr__(
            self,
            "solver_provenance_by_dataset",
            solver_provenance,
        )
        identity_maps = (
            self.model_x_by_species,
            self.model_segment_ids,
            self.model_segment_keys,
            self.model_discontinuity_ids,
            self.model_selection_provenance,
            self.model_observation_ids,
            self.model_raw_columns,
            self.model_source_rows,
        )
        if self.model_series and any(
            set(mapping) != set(self.model_series) for mapping in identity_maps
        ):
            raise ValueError("Fit objective model identity must be complete by dataset.")
        for dataset_id, series_by_name in self.model_series.items():
            expected_names = set(series_by_name)
            if any(
                set(mapping[dataset_id]) != expected_names
                for mapping in identity_maps
            ):
                raise ValueError(
                    "Fit objective model identity must be complete by series."
                )
            stable_raw_ids: set[str] = set()
            for series_name, values in series_by_name.items():
                expected_size = np.asarray(values, dtype=float).reshape(-1).size
                aligned = (
                    self.model_x_by_species[dataset_id][series_name],
                    self.model_segment_ids[dataset_id][series_name],
                    self.model_segment_keys[dataset_id][series_name],
                    self.model_discontinuity_ids[dataset_id][series_name],
                    self.model_selection_provenance[dataset_id][series_name],
                    self.model_observation_ids[dataset_id][series_name],
                    self.model_raw_columns[dataset_id][series_name],
                    self.model_source_rows[dataset_id][series_name],
                )
                if expected_size == 0 or any(
                    len(item) != expected_size for item in aligned
                ):
                    raise ValueError(
                        "Fit objective model rows require complete paired identity."
                    )
                selection = self.model_selection_provenance[dataset_id][series_name]
                observation_ids = self.model_observation_ids[
                    dataset_id
                ][series_name]
                if (
                    not np.all(
                        np.isfinite(
                            np.asarray(values, dtype=float).reshape(-1)
                        )
                    )
                    or not np.all(
                        np.isfinite(
                            np.asarray(
                                self.model_x_by_species[dataset_id][series_name],
                                dtype=float,
                            ).reshape(-1)
                        )
                    )
                    or np.any(
                        np.asarray(
                            self.model_segment_ids[dataset_id][series_name],
                            dtype=np.int64,
                        )
                        < 0
                    )
                    or any(
                        not str(value or "").strip()
                        for value in self.model_segment_keys[dataset_id][series_name]
                    )
                    or np.any(
                        np.asarray(
                            self.model_discontinuity_ids[dataset_id][series_name],
                            dtype=np.int64,
                        )
                        < 0
                    )
                    or any(
                        not str(value or "").strip()
                        for value in observation_ids
                    )
                    or len(set(observation_ids)) != 1
                    or any(
                        not str(value or "").strip()
                        for value in self.model_raw_columns[dataset_id][series_name]
                    )
                    or any(
                        int(value) <= 0
                        for value in self.model_source_rows[dataset_id][series_name]
                    )
                    or any(value not in {"direct", "interpolated"} for value in selection)
                ):
                    raise ValueError("Fit objective row selection provenance is invalid.")
                stable_raw_ids.add(observation_ids[0])
            if len(stable_raw_ids) != len(series_by_name):
                raise ValueError(
                    "Fit objective series must retain unique dataset-local "
                    "raw observation IDs."
                )


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
        return "Final weighted residual MSE is non-finite; results are invalid."
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
    np.ndarray,
    Dict[str, np.ndarray],
    np.ndarray,
    tuple[str, ...],
    np.ndarray,
    Dict[str, Any],
]:
    """Return time, series, and direct canonical row identity."""
    payload = simulation.series
    t_sim = np.asarray(payload.t, dtype=float).reshape(-1)
    return (
        t_sim,
        dict(payload.species),
        np.asarray(simulation.segment_ids, dtype=np.int64).reshape(-1),
        tuple(simulation.segment_keys),
        np.asarray(simulation.discontinuity_ids, dtype=np.int64).reshape(-1),
        dict(simulation.solver_provenance),
    )


def _align_series(
    series: np.ndarray,
    sim_time: np.ndarray,
    target_time: np.ndarray,
    segment_ids: np.ndarray,
    segment_keys: Sequence[str],
    discontinuity_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], np.ndarray, tuple[str, ...]]:
    """Align values and their direct canonical row identity together."""
    values = np.asarray(series, dtype=float).reshape(-1)
    sim_t = np.asarray(sim_time, dtype=float).reshape(-1)
    target_t = np.asarray(target_time, dtype=float).reshape(-1)
    if values.size != sim_t.size:
        raise ValueError(
            f"Simulation output has {values.size} values but {sim_t.size} time points."
        )
    segments = np.asarray(segment_ids, dtype=np.int64).reshape(-1)
    keys = tuple(str(value or "").strip() for value in segment_keys)
    discontinuities = np.asarray(discontinuity_ids, dtype=np.int64).reshape(-1)
    if (
        sim_t.size < 2
        or segments.size != sim_t.size
        or len(keys) != sim_t.size
        or discontinuities.size != sim_t.size
        or any(not value for value in keys)
        or np.any(segments < 0)
        or np.any(discontinuities < 0)
    ):
        raise FitSimulationError(
            "Fit simulation row identity must align with simulation time.",
            details={"fatal": True, "stage": "intervention_execution"},
        )
    if np.any(np.diff(sim_t) <= 0.0):
        raise ValueError("Simulation time must be strictly increasing for fitting alignment.")

    aligned = np.empty(target_t.size, dtype=float)
    aligned_segments = np.empty(target_t.size, dtype=np.int64)
    aligned_discontinuities = np.empty(target_t.size, dtype=np.int64)
    aligned_keys: list[str] = []
    selection: list[str] = []
    for index, time_value in enumerate(target_t):
        exact = np.flatnonzero(sim_t == time_value)
        if exact.size:
            if exact.size != 1:
                raise FitSimulationError(
                    "Exact Fit time must select one public simulation row.",
                    details={"fatal": True, "stage": "fit_identity"},
                )
            chosen = int(exact[0])
            aligned[index] = values[chosen]
            aligned_segments[index] = segments[chosen]
            aligned_discontinuities[index] = discontinuities[chosen]
            aligned_keys.append(keys[chosen])
            selection.append("direct")
            continue
        right_candidates = np.flatnonzero(sim_t > time_value)
        right = int(right_candidates[0]) if right_candidates.size else sim_t.size
        left = right - 1
        if left < 0 or right >= sim_t.size:
            raise ValueError("Dataset time lies outside the simulation time span.")
        if (
            segments[left] != segments[right]
            or keys[left] != keys[right]
            or discontinuities[left] != discontinuities[right]
        ):
            raise FitSimulationError(
                "Time-fit interpolation cannot cross an executor-issued region."
            )
        width = float(sim_t[right] - sim_t[left])
        fraction = float((time_value - sim_t[left]) / width)
        aligned[index] = float(
            values[left] + fraction * (values[right] - values[left])
        )
        aligned_segments[index] = segments[left]
        aligned_discontinuities[index] = discontinuities[left]
        aligned_keys.append(keys[left])
        selection.append("interpolated")
    return (
        aligned,
        aligned_segments,
        tuple(aligned_keys),
        aligned_discontinuities,
        tuple(selection),
    )


def select_fit_rows(
    *,
    model_y: np.ndarray,
    model_x: np.ndarray | None,
    x_name: str,
    x_mode: str,
    sim_time: np.ndarray,
    observation_time: np.ndarray,
    observation_x: np.ndarray,
    observation_id: str,
    raw_column: str,
    source_rows: Sequence[int],
    segment_ids: np.ndarray,
    segment_keys: Sequence[str],
    discontinuity_ids: np.ndarray,
    dataset_label: str,
    y_name: str,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[str, ...],
    np.ndarray,
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[int, ...],
]:
    """Select model X/Y and complete model/observation identity together."""

    obs_t = np.asarray(observation_time, dtype=float).reshape(-1)
    obs_x = np.asarray(observation_x, dtype=float).reshape(-1)
    rows = tuple(int(value) for value in source_rows)
    raw_id = str(observation_id or "").strip()
    raw_name = str(raw_column or "").strip()
    if (
        not raw_id
        or not raw_name
        or obs_t.size == 0
        or obs_t.size != obs_x.size
        or len(rows) != obs_t.size
        or len(rows) != len(set(rows))
        or any(row <= 0 for row in rows)
    ):
        raise FitSimulationError(
            "Fit selector requires complete row-aligned observation identity.",
            details={"fatal": True, "stage": "fit_capture"},
        )
    if str(x_name) == "t":
        selected = _align_series(
            model_y,
            sim_time,
            obs_t,
            segment_ids,
            segment_keys,
            discontinuity_ids,
        )
        selected_x = obs_t.copy()
    else:
        if model_x is None:
            raise FitSimulationError(
                f"X series {x_name!r} is missing from the simulation result.",
                details={"fatal": True, "stage": "fit_identity"},
            )
        mode = str(x_mode or "").strip().lower()
        if mode not in {"monotone", "time_guided"}:
            raise FitSimulationError(
                f"Dataset {dataset_label!r}: Fit requires a locked parametric-X "
                f"mode; got {x_mode!r}.",
                details={"fatal": True, "stage": "fit_configuration"},
            )
        sim_t = np.asarray(sim_time, dtype=float).reshape(-1)
        sim_y = np.asarray(model_y, dtype=float).reshape(-1)
        sim_x = np.asarray(model_x, dtype=float).reshape(-1)
        segments = np.asarray(segment_ids, dtype=np.int64).reshape(-1)
        keys = tuple(str(value or "").strip() for value in segment_keys)
        discontinuities = np.asarray(
            discontinuity_ids,
            dtype=np.int64,
        ).reshape(-1)
        if (
            obs_t.size == 0
            or obs_t.size != obs_x.size
            or not np.all(np.isfinite(obs_t))
            or not np.all(np.isfinite(obs_x))
            or sim_t.size < 2
            or sim_y.size != sim_t.size
            or sim_x.size != sim_t.size
            or not np.all(np.isfinite(sim_t))
            or not np.all(np.isfinite(sim_y))
            or not np.all(np.isfinite(sim_x))
            or segments.size != sim_t.size
            or len(keys) != sim_t.size
            or discontinuities.size != sim_t.size
            or any(not value for value in keys)
            or np.any(segments < 0)
            or np.any(np.diff(segments) < 0)
            or np.any(discontinuities < 0)
            or np.any(np.diff(discontinuities) < 0)
            or np.any(np.diff(sim_t) <= 0.0)
        ):
            raise FitSimulationError(
                "Parametric Fit received malformed observation or simulation "
                "row identity.",
                details={"fatal": True, "stage": "fit_identity"},
            )
        aligner = _ParametricXAligner(
            mode=mode,
            t_obs=obs_t,
            x_obs=obs_x,
            t_sim=sim_t,
            x_model=sim_x,
            segment_ids=segments,
            segment_keys=keys,
            discontinuity_ids=discontinuities,
            dataset_label=dataset_label,
            x_name=x_name,
        )
        try:
            parametric = aligner.align(
                y_model=sim_y,
                y_name=y_name,
            )
        except DeterministicParametricResolutionError as exc:
            raise _candidate_trajectory_failure(
                exc,
                stage="parametric_alignment",
            ) from exc
        except FitSimulationError as exc:
            if mode != "time_guided":
                raise
            # With the complete typed inputs above, the time-guided aligner can
            # now fail only because this candidate trajectory has no admissible
            # crossing or produces a non-finite interpolated trajectory value.
            raise _candidate_trajectory_failure(
                exc,
                stage="parametric_alignment",
            ) from exc
        selected_x = np.asarray(parametric[0], dtype=float)
        selected = parametric[1:]
    return (
        selected_x,
        np.asarray(selected[0], dtype=float),
        np.asarray(selected[1], dtype=np.int64),
        tuple(selected[2]),
        np.asarray(selected[3], dtype=np.int64),
        tuple(selected[4]),
        tuple(raw_id for _row in rows),
        tuple(raw_name for _row in rows),
        rows,
    )


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
            "raw_observation_id": series.raw_observation_id,
            "raw_column": series.raw_column,
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
        discontinuity_ids: np.ndarray,
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
        self._discontinuity_ids = np.asarray(
            discontinuity_ids,
            dtype=np.int64,
        )
        self._dataset_label = str(dataset_label)
        self._x_name = str(x_name)

    def align(
        self,
        *,
        y_model: np.ndarray,
        y_name: str,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        tuple[str, ...],
        np.ndarray,
        tuple[str, ...],
    ]:
        if self._mode == "monotone":
            return align_y_on_x_obs(
                t_obs=self._t_obs,
                x_obs=self._x_obs,
                t_sim=self._t_sim,
                x_model=self._x_model,
                y_model=np.asarray(y_model, dtype=float),
                segment_ids=self._segment_ids,
                segment_keys=self._segment_keys,
                discontinuity_ids=self._discontinuity_ids,
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
            discontinuity_ids=self._discontinuity_ids,
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
    discontinuity_ids: Optional[np.ndarray] = None
    solver_provenance: Optional[Dict[str, Any]] = None
    error: Optional[BaseException] = None
    error_provenance: Optional[Dict[str, Any]] = None
    final_error_message: Optional[str] = None

    def __post_init__(self) -> None:
        species = dict(self.sim_species)
        object.__setattr__(self, "sim_species", species)
        if self.error is not None:
            if (
                self.sim_time is not None
                or species
                or self.segment_ids is not None
                or self.segment_keys is not None
                or self.discontinuity_ids is not None
                or self.solver_provenance is not None
            ):
                raise ValueError(
                    "Failed Fit evaluations cannot publish a partial result."
                )
        elif self.segment_ids is None or self.sim_time is None:
            raise ValueError(
                "Successful Fit simulation results require complete row identity."
            )
        else:
            times = np.asarray(self.sim_time, dtype=float).reshape(-1).copy()
            if (
                times.size == 0
                or np.any(~np.isfinite(times))
                or np.any(np.diff(times) <= 0.0)
                or not species
            ):
                raise ValueError(
                    "Successful Fit simulation results require finite ordered rows."
                )
            normalized_species: Dict[str, np.ndarray] = {}
            for raw_name, raw_values in species.items():
                name = str(raw_name or "").strip()
                values = np.asarray(raw_values, dtype=float).reshape(-1).copy()
                if (
                    not name
                    or values.size != times.size
                    or np.any(~np.isfinite(values))
                ):
                    raise ValueError(
                        "Fit simulation species must align with finite row time."
                    )
                values.setflags(write=False)
                normalized_species[name] = values
            times.setflags(write=False)
            object.__setattr__(self, "sim_time", times)
            object.__setattr__(self, "sim_species", normalized_species)
        if self.segment_ids is not None:
            segments = np.asarray(
                self.segment_ids,
                dtype=np.int64,
            ).reshape(-1).copy()
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
                or np.any(segments < 0)
                or np.any(np.diff(segments) < 0)
            ):
                raise ValueError(
                    "Fit simulation segment identity must align with time."
                )
            discontinuities = np.asarray(
                self.discontinuity_ids,
                dtype=np.int64,
            ).reshape(-1).copy()
            if (
                discontinuities.size != expected_size
                or np.any(discontinuities < 0)
                or np.any(np.diff(discontinuities) < 0)
            ):
                raise ValueError(
                    "Fit direct row identity must align with time."
                )
            object.__setattr__(self, "segment_ids", segments)
            object.__setattr__(self, "segment_keys", keys)
            segments.setflags(write=False)
            discontinuities.setflags(write=False)
            object.__setattr__(self, "discontinuity_ids", discontinuities)
        elif (
            self.segment_keys is not None
            or self.discontinuity_ids is not None
        ):
            raise ValueError(
                "Fit simulation row identity requires "
                "discontinuity identity."
            )
        if self.error_provenance is not None:
            object.__setattr__(self, "error_provenance", dict(self.error_provenance))
        if self.solver_provenance is not None:
            if not isinstance(self.solver_provenance, Mapping):
                raise TypeError("Fit solver provenance must be a mapping.")
            object.__setattr__(
                self,
                "solver_provenance",
                copy.deepcopy(dict(self.solver_provenance)),
            )


def dataset_evaluation_is_fatal(result: DatasetSimulationEvaluation) -> bool:
    error = result.error
    return error is not None and not _is_expected_candidate_domain_failure(error)


def evaluate_dataset_simulation(
    fit_evaluators: Mapping[str, object],
    item: ObjectiveDatasetInput,
    *,
    cancellation_check: Optional[Callable[[], bool]] = None,
    retain_solver_provenance: bool = False,
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
            discontinuity_ids,
            solver_provenance,
        ) = _extract_simulation_payload(sim_result)
        _raise_if_fitting_cancel_requested(cancellation_check)
        return DatasetSimulationEvaluation(
            index=int(item.index),
            sim_time=sim_time,
            sim_species=sim_species,
            segment_ids=segment_ids,
            segment_keys=segment_keys,
            discontinuity_ids=discontinuity_ids,
            solver_provenance=(
                solver_provenance if retain_solver_provenance else None
            ),
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
    retain_solver_provenance: bool = False,
) -> List[DatasetSimulationEvaluation]:
    if not items:
        return []
    return evaluate_dataset_simulations_serial(
        fit_evaluators,
        items,
        cancellation_check=cancellation_check,
        stop_on_fatal=stop_on_fatal,
        retain_solver_provenance=retain_solver_provenance,
    )


def evaluate_dataset_simulations_serial(
    fit_evaluators: Mapping[str, object],
    items: Sequence[ObjectiveDatasetInput],
    *,
    cancellation_check: Optional[Callable[[], bool]],
    stop_on_fatal: bool,
    retain_solver_provenance: bool,
) -> List[DatasetSimulationEvaluation]:
    results = []
    for item in items:
        _raise_if_fitting_cancelled(cancellation_check)
        result = evaluate_dataset_simulation(
            fit_evaluators,
            item,
            cancellation_check=cancellation_check,
            retain_solver_provenance=retain_solver_provenance,
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
                (
                    self._dataset_params.get(ds_id, {}),
                    FITTING_PARAM_ORIGIN_CONFIGURED_DATASET,
                ),
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
                if not _is_expected_candidate_domain_failure(exc):
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
            discontinuity_ids = evaluation.discontinuity_ids
            if (
                segment_ids is None
                or segment_keys is None
                or discontinuity_ids is None
            ):
                raise FitSimulationError(
                    f"Dataset {ds_id!r}: Fit simulation lost exact segment "
                    "identity.",
                    failed_params=failed_param_snapshot,
                )

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
                    raise err

                try:
                    if x_name != "t" and species_x_obs is None:
                        raise FitSimulationError(
                            f"Dataset '{ds_id}' species '{species_name}' is missing x observations for X='{x_name}'.",
                            failed_params=failed_param_snapshot,
                        )
                    selected_rows = select_fit_rows(
                        model_y=np.asarray(model_series, dtype=float),
                        model_x=(
                            None
                            if x_name == "t"
                            else sim_species.get(x_name)
                        ),
                        x_name=x_name,
                        x_mode=x_mode,
                        sim_time=np.asarray(sim_time, dtype=float),
                        observation_time=obs_t,
                        observation_x=(
                            obs_t
                            if x_name == "t"
                            else np.asarray(species_x_obs, dtype=float)
                        ),
                        observation_id=species_obs["raw_observation_id"],
                        raw_column=species_obs["raw_column"],
                        source_rows=species_obs["source_rows"],
                        segment_ids=np.asarray(segment_ids, dtype=np.int64),
                        segment_keys=segment_keys,
                        discontinuity_ids=np.asarray(
                            discontinuity_ids,
                            dtype=np.int64,
                        ),
                        dataset_label=ds_id,
                        y_name=species_name,
                    )
                    y_sim = selected_rows[1]
                except FitSimulationError as exc:
                    err = FitSimulationError(
                        str(exc),
                        failed_params=failed_param_snapshot,
                        details=dict(getattr(exc, "details", None) or {}),
                        context=getattr(exc, "context", None),
                    )
                    self._ctx.set_error(err, {"dataset": ds_id})
                    if not _is_expected_candidate_domain_failure(err):
                        raise err
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
                    raise err from exc

                if not np.all(np.isfinite(y_sim)):
                    err = FitSimulationError(
                        f"Simulation produced non-finite values for dataset '{ds_id}', species '{species_name}'.",
                        failed_params=failed_param_snapshot,
                    )
                    self._ctx.set_error(err, {"dataset": ds_id})
                    raise err

                residual_vector = effective_weight * (
                    np.asarray(y_sim, dtype=float).reshape(-1) - np.asarray(y_exp, dtype=float).reshape(-1)
                )
                if not np.all(np.isfinite(residual_vector)):
                    err = FitSimulationError(
                        f"Residuals contained non-finite values for dataset '{ds_id}'.",
                        failed_params=failed_param_snapshot,
                    )
                    self._ctx.set_error(err, {"dataset": ds_id})
                    raise err
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
    fixed_dataset_params: Dict[str, Dict[str, float]],
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
    total_weighted_sse = 0.0
    total_points = 0
    model_series_map: Dict[str, Dict[str, np.ndarray]] = {}
    model_x_by_species_map: Dict[str, Dict[str, np.ndarray]] = {}
    model_segment_ids_map: Dict[str, Dict[str, np.ndarray]] = {}
    model_segment_keys_map: Dict[str, Dict[str, tuple[str, ...]]] = {}
    model_discontinuity_ids_map: Dict[str, Dict[str, np.ndarray]] = {}
    model_selection_provenance_map: Dict[str, Dict[str, tuple[str, ...]]] = {}
    model_observation_ids_map: Dict[str, Dict[str, tuple[str, ...]]] = {}
    model_raw_columns_map: Dict[str, Dict[str, tuple[str, ...]]] = {}
    model_source_rows_map: Dict[str, Dict[str, tuple[int, ...]]] = {}
    solver_provenance_by_dataset: Dict[str, Dict[str, Any]] = {}
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
    plot_model_discontinuity_ids_map: Dict[str, Dict[str, np.ndarray]] = {}
    final_dataset_failures: Dict[str, FitDiagnostic] = {}
    final_dataset_warnings: Dict[str, str] = {}

    dataset_inputs: List[ObjectiveDatasetInput] = []
    for index, payload in enumerate(payloads):
        ds_id = payload.dataset_id
        _raise_if_fitting_cancelled(cancellation_check)

        full_params = dict(fitted_params)
        full_params.update(fixed_dataset_params.get(ds_id, {}))
        full_params.update(combined_dataset_params.get(ds_id, {}))
        parameter_origins = {
            name: FITTING_PARAM_ORIGIN_OPTIMIZER_SHARED
            for name in fitted_params
        }
        for name in fixed_dataset_params.get(ds_id, {}):
            parameter_origins[name] = FITTING_PARAM_ORIGIN_CONFIGURED_DATASET
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
            retain_solver_provenance=True,
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
            if evaluation.solver_provenance is None:
                raise RuntimeError(
                    f"Final Fit replay omitted solver provenance for dataset {ds_id!r}."
                )
            solver_provenance_by_dataset[ds_id] = dict(
                evaluation.solver_provenance
            )
        segment_ids = evaluation.segment_ids
        segment_keys = evaluation.segment_keys
        discontinuity_ids = evaluation.discontinuity_ids

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
        weighted_residual_blocks: List[np.ndarray] = []
        r_squared_by_series: Dict[str, float] = {}
        for species_name in species_list:
            species_obs = observations[species_name]
            obs_t = np.asarray(species_obs["t"], dtype=float).reshape(-1)
            y_exp = np.asarray(species_obs["y"], dtype=float).reshape(-1)
            species_x_obs = (
                np.asarray(x_obs_by_species.get(species_name), dtype=float).reshape(-1)
                if species_name in x_obs_by_species
                else None
            )
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
                continue

            try:
                if x_name != "t" and species_x_obs is None:
                    raise FitSimulationError(
                        f"Dataset '{ds_id}' species '{species_name}' is missing x observations for X='{x_name}'.",
                        failed_params=failed_param_snapshot,
                    )
                selected_rows = select_fit_rows(
                    model_y=np.asarray(sim_species[species_name], dtype=float),
                    model_x=(
                        None if x_name == "t" else sim_species.get(x_name)
                    ),
                    x_name=x_name,
                    x_mode=x_mode,
                    sim_time=np.asarray(sim_time, dtype=float),
                    observation_time=obs_t,
                    observation_x=(
                        obs_t
                        if x_name == "t"
                        else np.asarray(species_x_obs, dtype=float)
                    ),
                    observation_id=species_obs["raw_observation_id"],
                    raw_column=species_obs["raw_column"],
                    source_rows=species_obs["source_rows"],
                    segment_ids=np.asarray(segment_ids, dtype=np.int64),
                    segment_keys=tuple(segment_keys or ()),
                    discontinuity_ids=np.asarray(
                        discontinuity_ids,
                        dtype=np.int64,
                    ),
                    dataset_label=ds_id,
                    y_name=species_name,
                )
                y_sim_resid = selected_rows[1]
                model_series_map.setdefault(ds_id, {})[species_name] = np.asarray(
                    selected_rows[1], dtype=float
                )
                model_x_by_species_map.setdefault(ds_id, {})[species_name] = np.asarray(
                    selected_rows[0], dtype=float
                )
                model_segment_ids_map.setdefault(ds_id, {})[species_name] = np.asarray(
                    selected_rows[2], dtype=np.int64
                )
                model_segment_keys_map.setdefault(ds_id, {})[species_name] = tuple(
                    selected_rows[3]
                )
                model_discontinuity_ids_map.setdefault(ds_id, {})[species_name] = np.asarray(
                    selected_rows[4], dtype=np.int64
                )
                model_selection_provenance_map.setdefault(ds_id, {})[species_name] = tuple(
                    selected_rows[5]
                )
                model_observation_ids_map.setdefault(ds_id, {})[species_name] = tuple(
                    selected_rows[6]
                )
                model_raw_columns_map.setdefault(ds_id, {})[species_name] = tuple(
                    selected_rows[7]
                )
                model_source_rows_map.setdefault(ds_id, {})[species_name] = tuple(
                    selected_rows[8]
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
            else:
                raw_residual = np.asarray(y_sim_resid, dtype=float) - y_exp
                residual_blocks.append(raw_residual)
                ss_res = float(np.sum(raw_residual**2))
                ss_tot = float(
                    np.sum((y_exp - float(np.mean(y_exp))) ** 2)
                )
                r_squared_by_series[str(species_name)] = (
                    1.0 - (ss_res / ss_tot) if ss_tot > 0.0 else float("nan")
                )
                target_multiplier = float(
                    _normalized_target_weight_multipliers(
                        species_list=species_list,
                        target_weights=dict(payload.target_weights),
                    )[species_name]
                )
                weighted_residual_blocks.append(
                    _objective_dataset_weight_multiplier(
                        weights.get(ds_id, 1.0)
                    )
                    * target_multiplier
                    * raw_residual
                )

            if x_name == "t":
                plot_model_series_map.setdefault(ds_id, {})[species_name] = np.asarray(
                    sim_species[species_name],
                    dtype=float,
                ).reshape(-1).copy()
                plot_model_x_by_species_map.setdefault(ds_id, {})[species_name] = np.asarray(
                    sim_time,
                    dtype=float,
                ).reshape(-1).copy()
                plot_model_segment_ids_map.setdefault(ds_id, {})[
                    species_name
                ] = np.asarray(segment_ids, dtype=np.int64).reshape(-1).copy()
                plot_model_segment_keys_map.setdefault(ds_id, {})[
                    species_name
                ] = tuple(segment_keys or ())
                plot_model_discontinuity_ids_map.setdefault(ds_id, {})[
                    species_name
                ] = np.asarray(discontinuity_ids, dtype=np.int64).reshape(-1).copy()
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
                    plot_model_discontinuity_ids_map.setdefault(ds_id, {})[
                        species_name
                    ] = np.asarray(discontinuity_ids, dtype=np.int64).reshape(-1)[plot_mask].copy()
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
                    plot_model_discontinuity_ids_map.setdefault(ds_id, {})[
                        species_name
                    ] = np.asarray(discontinuity_ids, dtype=np.int64).reshape(-1).copy()

        residuals = np.concatenate(residual_blocks) if residual_blocks else np.asarray([], dtype=float)
        weighted_residuals = (
            np.concatenate(weighted_residual_blocks)
            if weighted_residual_blocks
            else np.asarray([], dtype=float)
        )
        dataset_failed = ds_id in final_dataset_failures
        if dataset_failed or weighted_residuals.size != residuals.size:
            weighted_sse = float("nan")
            weighted_mse = float("nan")
            weighted_rmse = float("nan")
            weighted_mae = float("nan")
            r_squared_by_series = {
                str(species_name): float("nan")
                for species_name in species_list
            }
        else:
            weighted_sse = float(np.sum(weighted_residuals**2))
            weighted_mse = weighted_sse / float(weighted_residuals.size)
            weighted_rmse = float(np.sqrt(weighted_mse))
            weighted_mae = float(np.mean(np.abs(weighted_residuals)))

        dataset_info.append(
            DatasetFitInfo(
                dataset_id=ds_id,
                r_squared_by_series=dict(r_squared_by_series),
                weighted_residual_sse=weighted_sse,
                weighted_residual_mse=weighted_mse,
                weighted_rmse=weighted_rmse,
                weighted_mae=weighted_mae,
                residuals=residuals,
                n_points=int(residuals.size),
                weight=weights.get(ds_id, 1.0),
            )
        )

        if not dataset_failed:
            total_weighted_sse += weighted_sse
        total_points += int(residuals.size)

    weighted_residual_mse = (
        total_weighted_sse / total_points
        if total_points > 0 and not final_dataset_failures
        else float("nan")
    )

    metrics_finite = bool(
        np.isfinite(total_weighted_sse) and np.isfinite(weighted_residual_mse)
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
                message="Final weighted residual MSE is non-finite; results are invalid.",
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
        logger.info(
            "Global fit complete: weighted residual SSE=%.4e, weighted residual MSE=%.4e",
            total_weighted_sse,
            weighted_residual_mse,
        )
    elif completion.status == "warn":
        logger.warning("Global fit complete with warnings: %s", completion_message)
    else:
        logger.warning(
            "Global fit failed: weighted residual MSE=%.4e datasets_failed=%s",
            weighted_residual_mse,
            ",".join(sorted(final_dataset_failures.keys())),
        )
    for info in dataset_info:
        logger.debug(
            "  %s: weighted SSE=%.4e, weighted RMSE=%.4e",
            info.dataset_id,
            info.weighted_residual_sse,
            info.weighted_rmse,
        )

    return GlobalFitResult(
        shared_params=fitted_params,
        dataset_params=combined_dataset_params,
        uncertainties=uncertainties,
        weighted_residual_sse=total_weighted_sse,
        weighted_residual_mse=weighted_residual_mse,
        dataset_info=dataset_info,
        nfev=nfev,
        message=completion_message,
        completion=completion,
        covariance=covariance,
        objective_residuals=objective_residuals.copy() if objective_residuals is not None else None,
        model_series=model_series_map,
        model_x_by_species=model_x_by_species_map,
        model_segment_ids=model_segment_ids_map,
        model_segment_keys=model_segment_keys_map,
        model_discontinuity_ids=model_discontinuity_ids_map,
        model_selection_provenance=model_selection_provenance_map,
        model_observation_ids=model_observation_ids_map,
        model_raw_columns=model_raw_columns_map,
        model_source_rows=model_source_rows_map,
        solver_provenance_by_dataset=solver_provenance_by_dataset,
        plot_observed_x=plot_observed_x_map,
        plot_observed_source_rows=plot_observed_source_rows_map,
        plot_observed_raw_ids=plot_observed_raw_ids_map,
        plot_model_x_by_species=plot_model_x_by_species_map,
        plot_model_series=plot_model_series_map,
        plot_model_segment_ids=plot_model_segment_ids_map,
        plot_model_segment_keys=plot_model_segment_keys_map,
        plot_model_discontinuity_ids=plot_model_discontinuity_ids_map,
    )
