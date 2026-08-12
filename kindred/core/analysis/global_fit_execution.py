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
from types import MappingProxyType
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
from kindred.core.analysis.fit_evaluation_outcome import FitEvaluationOutcome
from kindred.core.analysis.fit_statistics import (
    FitMetric,
    finite_positive_mean_normalize,
    finite_sum_of_squares,
    fit_error_metrics,
    r_squared_metric,
)
from kindred.core.exceptions import (
    FitCandidateDomainError,
    FitFinalReplayError,
    FittingCancelled,
    FitSimulationError,
    SimulationCancelled,
)
from kindred.core.fitting_completion import (
    FitDiagnostic,
    FitTerminalExtension,
    GlobalFitCompletion,
)
from kindred.core.fitting_evaluation import (
    FITTING_PARAM_ORIGIN_CONFIGURED_DATASET,
    FITTING_PARAM_ORIGIN_OPTIMIZER_DATASET,
    FITTING_PARAM_ORIGIN_OPTIMIZER_SHARED,
    evaluate_fitting_series,
    is_candidate_domain_failure,
)
from kindred.core.objective import ObjectiveContext
from kindred.core.result_repository import SimulationResultValue
from kindred.core.simulator.solver_types import SimulationCancellationControl
from kindred.core.simulation_failure import (
    SimulationFailure,
    coerce_simulation_failure,
    simulation_failure_from_exception,
    simulation_failure_user_message,
)

logger = logging.getLogger(__name__)


class _ObjectiveTransport(np.ndarray):
    """Optimizer-compatible vector carrying its same-call typed outcome."""

    def __new__(
        cls,
        values: object,
        outcome: FitEvaluationOutcome,
    ) -> "_ObjectiveTransport":
        array = np.asarray(values, dtype=float).reshape(-1).view(cls)
        array.outcome = outcome  # type: ignore[attr-defined]
        return array

__all__ = [
    "FitProgress",
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
    "fit_result_target_values",
    "select_fit_rows",
    "dataset_evaluation_is_fatal",
    "assemble_global_fit_result",
    "attach_terminal_extensions",
]


def _decode_log10_physical(raw_value: object, *, coordinate: str) -> float:
    """Decode one optimizer Log10 coordinate without clipping or fabrication."""

    try:
        value = float(raw_value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as exc:
        raise FitSimulationError(
            f"Invalid Log10 optimizer coordinate for {coordinate!r}.",
            details={
                "fatal": False,
                "candidate_domain": True,
                "stage": "log10_decode",
            },
        ) from exc
    if not math.isfinite(value):
        raise FitSimulationError(
            f"Non-finite Log10 optimizer coordinate for {coordinate!r}.",
            details={
                "fatal": False,
                "candidate_domain": True,
                "stage": "log10_decode",
            },
        )
    try:
        decoded = math.pow(10.0, value)
    except (OverflowError, ValueError) as exc:
        raise FitSimulationError(
            f"Log10 optimizer coordinate for {coordinate!r} overflowed.",
            details={
                "fatal": False,
                "candidate_domain": True,
                "stage": "log10_decode",
            },
        ) from exc
    if not math.isfinite(decoded):
        raise FitSimulationError(
            f"Log10 optimizer coordinate for {coordinate!r} decoded non-finitely.",
            details={
                "fatal": False,
                "candidate_domain": True,
                "stage": "log10_decode",
            },
        )
    return float(decoded)

def _raise_if_fitting_cancelled(cancellation_check: Optional[Callable[[], bool]]) -> None:
    if cancellation_check is not None and cancellation_check():
        raise FittingCancelled()


def _raise_if_fitting_cancel_requested(
    cancellation_control: SimulationCancellationControl | None,
) -> None:
    if cancellation_control is None:
        return
    if cancellation_control.cancel_requested():
        raise FittingCancelled()


@dataclass(frozen=True, slots=True)
class FitProgress:
    """Immutable Global Fit progress with its complete best-result payload."""

    objective_calls: int
    cost: float
    parameters: Mapping[str, float]
    improved: bool = False
    best_simulation_results_by_dataset: Mapping[str, SimulationResultValue] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "objective_calls", int(self.objective_calls))
        object.__setattr__(self, "cost", float(self.cost))
        object.__setattr__(
            self,
            "parameters",
            MappingProxyType(
                {str(name): float(value) for name, value in self.parameters.items()}
            ),
        )
        object.__setattr__(self, "improved", bool(self.improved))
        best_results = {
            str(dataset_id): value
            for dataset_id, value in self.best_simulation_results_by_dataset.items()
        }
        if any(
            not isinstance(value, SimulationResultValue)
            for value in best_results.values()
        ):
            raise TypeError(
                "Fit progress best simulations require SimulationResultValue values."
            )
        object.__setattr__(
            self,
            "best_simulation_results_by_dataset",
            MappingProxyType(best_results),
        )


@dataclass
class DatasetFitInfo:
    """Fit statistics for a single dataset within global fit."""
    dataset_id: str  # Identifier for this dataset
    r_squared_by_series: Dict[str, FitMetric]  # Ordinary, unweighted R² by series
    weighted_residual_sse: FitMetric  # Exact contribution to the objective
    weighted_residual_mse: FitMetric  # Exact objective contribution per observation
    weighted_rmse: FitMetric
    weighted_mae: FitMetric
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
    weighted_residual_sse : FitMetric
        Exact sum of squares minimized by the weighted objective, or an
        explicit unavailable reason.
    weighted_residual_mse : FitMetric
        Exact weighted objective sum of squares per observation, or an
        explicit unavailable reason.
    dataset_info : list
        Per-dataset fit statistics (DatasetFitInfo objects)
    nfev : int
        Total number of function evaluations
    message : str
        Optimization termination message
    """
    shared_params: Dict[str, float]
    dataset_params: Dict[str, Dict[str, float]]
    weighted_residual_sse: FitMetric
    weighted_residual_mse: FitMetric
    dataset_info: List[DatasetFitInfo]
    nfev: int
    message: str
    completion: GlobalFitCompletion
    scipy_nfev: int | None = None
    terminal_replay_count: int = 0
    objective_call_count: int | None = None
    shared_refit_provenance: Mapping[str, object] = field(default_factory=dict)
    model_series: Dict[str, Dict[str, np.ndarray]] = field(default_factory=dict)
    model_x_by_species: Dict[str, Dict[str, np.ndarray]] = field(default_factory=dict)
    model_segment_ids: Dict[str, Dict[str, np.ndarray]] = field(default_factory=dict)
    model_segment_keys: Dict[str, Dict[str, tuple[str, ...]]] = field(default_factory=dict)
    model_discontinuity_ids: Dict[str, Dict[str, np.ndarray]] = field(default_factory=dict)
    model_selection_provenance: Dict[str, Dict[str, tuple[str, ...]]] = field(default_factory=dict)
    model_observation_ids: Dict[str, Dict[str, tuple[str, ...]]] = field(default_factory=dict)
    model_raw_columns: Dict[str, Dict[str, tuple[str, ...]]] = field(default_factory=dict)
    model_source_rows: Dict[str, Dict[str, tuple[int, ...]]] = field(default_factory=dict)
    simulation_results_by_dataset: Dict[str, SimulationResultValue] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if not isinstance(self.completion, GlobalFitCompletion):
            raise TypeError("GlobalFitResult.completion must be a GlobalFitCompletion")
        if not (
            isinstance(self.weighted_residual_sse, FitMetric)
            and isinstance(self.weighted_residual_mse, FitMetric)
        ):
            raise TypeError(
                "GlobalFitResult aggregate metrics require typed FitMetric values."
            )
        shared_params = {
            str(name): float(value)
            for name, value in self.shared_params.items()
        }
        dataset_params = {
            str(dataset_id): {
                str(name): float(value)
                for name, value in values.items()
            }
            for dataset_id, values in self.dataset_params.items()
        }
        if any(not np.isfinite(value) for value in shared_params.values()) or any(
            not np.isfinite(value)
            for values in dataset_params.values()
            for value in values.values()
        ):
            raise ValueError(
                "GlobalFitResult requires finite completed parameter values."
            )
        self.shared_params = shared_params
        self.dataset_params = dataset_params
        self.nfev = int(self.nfev)
        self.scipy_nfev = self.nfev if self.scipy_nfev is None else int(self.scipy_nfev)
        self.terminal_replay_count = int(self.terminal_replay_count)
        if self.terminal_replay_count < 0:
            raise ValueError("GlobalFitResult terminal replay count cannot be negative.")
        if self.objective_call_count is not None:
            self.objective_call_count = int(self.objective_call_count)
            if self.objective_call_count < 0:
                raise ValueError("GlobalFitResult objective-call count cannot be negative.")
        self.shared_refit_provenance = MappingProxyType(
            dict(self.shared_refit_provenance or {})
        )
        simulation_results: Dict[str, SimulationResultValue] = {}
        for raw_dataset_id, simulation in self.simulation_results_by_dataset.items():
            dataset_id = str(raw_dataset_id or "").strip()
            if not dataset_id or dataset_id in simulation_results:
                raise ValueError(
                    "Fit simulation results require distinct stable dataset IDs."
                )
            if not isinstance(simulation, SimulationResultValue):
                raise TypeError(
                    "Fit simulation result entries must be SimulationResultValue values."
                )
            simulation_results[dataset_id] = simulation
        object.__setattr__(
            self,
            "simulation_results_by_dataset",
            simulation_results,
        )
        info_ids = [str(info.dataset_id or "").strip() for info in self.dataset_info]
        included_dataset_ids = set(dataset_params)
        if (
            not included_dataset_ids
            or any(not dataset_id for dataset_id in info_ids)
            or len(info_ids) != len(set(info_ids))
            or set(info_ids) != included_dataset_ids
            or set(simulation_results) != included_dataset_ids
            or set(self.model_series) != included_dataset_ids
        ):
            raise ValueError(
                "GlobalFitResult rosters must exactly match completed included datasets."
            )
        for info in self.dataset_info:
            if (
                int(info.n_points) <= 0
                or not all(
                    isinstance(value, FitMetric)
                    for value in (
                        info.weighted_residual_sse,
                        info.weighted_residual_mse,
                        info.weighted_rmse,
                        info.weighted_mae,
                    )
                )
                or not all(
                    isinstance(value, FitMetric)
                    for value in info.r_squared_by_series.values()
                )
                or not np.isfinite(float(info.weight))
                or float(info.weight) <= 0.0
            ):
                raise ValueError(
                    "GlobalFitResult requires typed per-dataset metrics and finite weights."
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


def _compact_failure_message(failure: SimulationFailure) -> str:
    raw_message = simulation_failure_user_message(failure)
    return " ".join(part.strip() for part in str(raw_message).splitlines() if part.strip())


def build_fit_diagnostic_from_failure(
    failure: object,
    *,
    phase: str,
    dataset_id: Optional[str] = None,
) -> FitDiagnostic:
    coerced_failure = _coerce_fit_failure_payload(failure)
    return FitDiagnostic(
        phase=phase,
        dataset_id=dataset_id,
        failure=coerced_failure,
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
) -> FitDiagnostic:
    failure = _embedded_fit_failure_payload(exc)
    if failure is None:
        failure = _coerce_fit_failure_payload(simulation_failure_from_exception(exc))
    return FitDiagnostic(
        phase=phase,
        dataset_id=dataset_id,
        failure=failure,
    )


def completion_result_message(
    *,
    base_message: str,
    completion: GlobalFitCompletion,
    optimizer_diagnostic: Optional[FitDiagnostic],
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
        if warning_parts:
            return "Global fit completed with warnings: " + "; ".join(warning_parts)
        fallback = str(base_message or "").strip()
        if fallback and "success" not in fallback.lower():
            return fallback
        return "Global fit completed with warnings."
    raise AssertionError(f"Unsupported completed Fit status: {completion.status!r}")


def attach_terminal_extensions(
    result: GlobalFitResult,
    extensions: Sequence[FitTerminalExtension],
) -> GlobalFitResult:
    """Attach method-owned terminal truth to the canonical completed result.

    Uncertainty and hierarchical engines run only after the terminal candidate
    has been replayed.  This is therefore the sole post-replay attachment seam;
    it does not alter optimizer convergence or create another result type.
    """

    if not isinstance(result, GlobalFitResult):
        raise TypeError("Terminal extensions require a GlobalFitResult.")
    additions = tuple(extensions)
    if any(not isinstance(item, FitTerminalExtension) for item in additions):
        raise TypeError("Terminal extensions must be FitTerminalExtension values.")
    existing = tuple(result.completion.terminal_extensions)
    result.completion = GlobalFitCompletion(
        status=result.completion.status,
        optimizer_converged=result.completion.optimizer_converged,
        optimizer_diagnostic=result.completion.optimizer_diagnostic,
        terminal_extensions=existing + additions,
    )
    return result


def normalize_weights(payloads: List[FitDatasetSpec], weights: Optional[Dict[str, float]]) -> Dict[str, float]:
    payload_ids = [str(payload.dataset_id) for payload in payloads]
    payload_id_set = set(payload_ids)
    provided_weights = {} if weights is None else {str(key): float(value) for key, value in dict(weights).items()}

    unknown_ids = sorted(set(provided_weights) - payload_id_set)
    if unknown_ids:
        raise ValueError(f"Unknown dataset weight IDs: {', '.join(unknown_ids)}")

    admitted: Dict[str, float] = {}
    for payload in payloads:
        ds_id = str(payload.dataset_id)
        weight = provided_weights.get(ds_id, 1.0 / max(1, payload.point_count))
        if not np.isfinite(weight) or weight <= 0.0:
            raise ValueError(f"Dataset weight for '{ds_id}' must be finite and positive.")
        admitted[ds_id] = float(weight)
    return finite_positive_mean_normalize(admitted)


def _normalized_target_weight_multipliers(
    *,
    species_list: Sequence[str],
    target_weights: Optional[Dict[str, float]],
) -> Dict[str, float]:
    names = [str(name) for name in (species_list or []) if str(name).strip()]
    if not names:
        return {}

    raw_weights = dict(target_weights or {}) if isinstance(target_weights, dict) else {}
    cleaned: Dict[str, float] = {}
    for name in names:
        value = float(raw_weights.get(name, 1.0))
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(
                f"Target weight for {name!r} must be finite and positive."
            )
        cleaned[name] = float(value)
    normalized = finite_positive_mean_normalize(cleaned)
    return {
        name: float(np.sqrt(value))
        for name, value in normalized.items()
    }


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
        parametric = aligner.align(
            y_model=sim_y,
            y_name=y_name,
        )
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


def fit_result_target_values(
    result: SimulationResultValue,
    name: str,
) -> np.ndarray:
    """Resolve a Fit target from the canonical scalar/series result split.

    The retained Algebra DAG and evaluator decide whether a let is scalar or
    row-varying.  This boundary consumes that typed result directly; it never
    infers type from static parser tables or sampled values.
    """

    if not isinstance(result, SimulationResultValue):
        raise TypeError("Fit target lookup requires a SimulationResultValue.")
    target_name = str(name)
    time_size = np.asarray(result.t, dtype=float).reshape(-1).size
    if target_name in result.series:
        values = np.asarray(result.series[target_name], dtype=float).reshape(-1)
    elif target_name in result.scalars:
        value = float(result.scalars[target_name])
        if not np.isfinite(value):
            raise FitSimulationError(
                f"Fit scalar target {target_name!r} is non-finite.",
                details={"fatal": True, "stage": "fit_target_lookup"},
            )
        values = np.full(time_size, value, dtype=float)
    else:
        raise FitSimulationError(
            f"Fit target {target_name!r} is absent from the canonical "
            "simulation scalar/series outputs.",
            details={"fatal": True, "stage": "fit_target_lookup"},
        )
    if values.size != time_size or not np.all(np.isfinite(values)):
        raise FitSimulationError(
            f"Fit target {target_name!r} is not a finite row-aligned output.",
            details={"fatal": True, "stage": "fit_target_lookup"},
        )
    return values


@dataclass(frozen=True)
class FitParameterLayout:
    param_names: tuple[str, ...]
    shared_log10: Mapping[str, bool]
    dataset_var_order: tuple[tuple[str, str], ...]
    dataset_var_index: Mapping[str, Mapping[str, int]]
    dataset_var_log10: Mapping[Tuple[str, str], bool]
    coupled_var_order: tuple[FitCoupledVariableParamSpec, ...]
    coupled_var_index: Mapping[str, int]
    x0: np.ndarray
    lower: np.ndarray
    upper: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "param_names",
            tuple(str(name) for name in self.param_names),
        )
        object.__setattr__(
            self,
            "shared_log10",
            MappingProxyType(
                {str(name): bool(value) for name, value in self.shared_log10.items()}
            ),
        )
        object.__setattr__(
            self,
            "dataset_var_order",
            tuple(
                (str(dataset_id), str(parameter_name))
                for dataset_id, parameter_name in self.dataset_var_order
            ),
        )
        object.__setattr__(
            self,
            "dataset_var_index",
            MappingProxyType(
                {
                    str(dataset_id): MappingProxyType(
                        {
                            str(parameter_name): int(index)
                            for parameter_name, index in values.items()
                        }
                    )
                    for dataset_id, values in self.dataset_var_index.items()
                }
            ),
        )
        object.__setattr__(
            self,
            "dataset_var_log10",
            MappingProxyType(
                {
                    (str(dataset_id), str(parameter_name)): bool(value)
                    for (dataset_id, parameter_name), value in self.dataset_var_log10.items()
                }
            ),
        )
        object.__setattr__(self, "coupled_var_order", tuple(self.coupled_var_order))
        object.__setattr__(
            self,
            "coupled_var_index",
            MappingProxyType(
                {str(variable_id): int(index) for variable_id, index in self.coupled_var_index.items()}
            ),
        )
        for name in ("x0", "lower", "upper"):
            values = np.asarray(getattr(self, name), dtype=float).reshape(-1).copy()
            values.setflags(write=False)
            object.__setattr__(self, name, values)

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

    def coordinate_index(self, coordinate: str) -> int:
        name = str(coordinate)
        try:
            return self.opt_param_keys().index(name)
        except ValueError as exc:
            raise KeyError(name) from exc

    def coordinate_log10(self, coordinate: str) -> bool:
        name = str(coordinate)
        if name in self.param_names:
            return bool(self.shared_log10.get(name, False))
        if "::" in name:
            dataset_id, parameter_name = name.split("::", 1)
            if parameter_name in self.dataset_var_index.get(dataset_id, {}):
                return bool(
                    self.dataset_var_log10.get((dataset_id, parameter_name), False)
                )
        for item in self.coupled_var_order:
            if item.variable_id == name:
                return bool(item.log10)
        raise KeyError(name)

    def decode_coordinate(self, coordinate: str, optimizer_value: float) -> float:
        name = str(coordinate)
        value = float(optimizer_value)
        self.coordinate_index(name)
        return (
            _decode_log10_physical(value, coordinate=name)
            if self.coordinate_log10(name)
            else value
        )

    def encode_coordinate(self, coordinate: str, physical_value: float) -> float:
        name = str(coordinate)
        value = float(physical_value)
        self.coordinate_index(name)
        if self.coordinate_log10(name):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"Log10 Fit coordinate {name!r} requires a positive finite value."
                )
            return math.log10(value)
        if not math.isfinite(value):
            raise ValueError(f"Fit coordinate {name!r} requires a finite value.")
        return value

    def shared_param_dict_from_vector(self, vector: np.ndarray) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for idx, name in enumerate(self.param_names):
            raw_val = float(vector[idx])
            if self.shared_log10.get(name):
                out[name] = _decode_log10_physical(
                    raw_val,
                    coordinate=name,
                )
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
                out[var_name] = _decode_log10_physical(
                    raw_val,
                    coordinate=f"{ds_id}::{var_name}",
                )
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
                _decode_log10_physical(
                    raw_value,
                    coordinate=spec.variable_id,
                )
                if spec.log10
                else raw_value
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
        if bounds is None or name not in bounds:
            raise ValueError(
                f"Fit coordinate {name!r} requires finite configured bounds."
            )
        b = bounds[name]
        low = float(b[0])
        high = float(b[1])
        if not np.isfinite(low) or not np.isfinite(high) or low >= high:
            raise ValueError(
                f"Fit coordinate {name!r} requires finite ordered bounds."
            )
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
    result: Optional[SimulationResultValue] = None
    error: Optional[BaseException] = None
    error_provenance: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if self.error is not None:
            if self.result is not None:
                raise ValueError(
                    "Failed Fit evaluations cannot publish a partial result."
                )
        elif self.result is None:
            raise ValueError(
                "Successful Fit evaluations require a canonical simulation result."
            )
        elif not isinstance(self.result, SimulationResultValue):
            raise TypeError(
                "Successful Fit evaluations require a SimulationResultValue."
            )
        if self.error_provenance is not None:
            object.__setattr__(self, "error_provenance", dict(self.error_provenance))


def dataset_evaluation_is_fatal(result: DatasetSimulationEvaluation) -> bool:
    error = result.error
    return error is not None and not is_candidate_domain_failure(error)


def evaluate_dataset_simulation(
    fit_evaluators: Mapping[str, object],
    item: ObjectiveDatasetInput,
    *,
    cancellation_control: SimulationCancellationControl | None = None,
) -> DatasetSimulationEvaluation:
    ds_id = item.payload.dataset_id
    try:
        _raise_if_fitting_cancel_requested(cancellation_control)
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
        _raise_if_fitting_cancel_requested(cancellation_control)
        return DatasetSimulationEvaluation(
            index=int(item.index),
            result=sim_result,
        )
    except FitSimulationError as exc:
        return DatasetSimulationEvaluation(
            index=int(item.index),
            error=exc,
            error_provenance={"dataset": ds_id, "provenance": getattr(exc, "provenance", None)},
        )
    except Exception as exc:
        if isinstance(exc, (FittingCancelled, SimulationCancelled)):
            raise FittingCancelled() from exc
        return DatasetSimulationEvaluation(
            index=int(item.index),
            error=exc,
            error_provenance={"dataset": ds_id},
        )

def evaluate_dataset_simulations(
    fit_evaluators: Mapping[str, object],
    items: Sequence[ObjectiveDatasetInput],
    *,
    cancellation_check: Optional[Callable[[], bool]] = None,
    cancellation_control: SimulationCancellationControl | None = None,
    stop_on_fatal: bool = True,
) -> List[DatasetSimulationEvaluation]:
    if not items:
        return []
    results = []
    for item in items:
        _raise_if_fitting_cancelled(cancellation_check)
        _raise_if_fitting_cancel_requested(cancellation_control)
        result = evaluate_dataset_simulation(
            fit_evaluators,
            item,
            cancellation_control=cancellation_control,
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
        progress_callback: Optional[Callable[[FitProgress], None]],
        cancellation_check: Optional[Callable[[], bool]],
        cancellation_control: SimulationCancellationControl | None = None,
    ) -> None:
        self._fit_evaluators = dict(fit_evaluators)
        self._payloads = payloads
        self._dataset_params = dataset_params
        self._weights = weights
        self._layout = layout
        self._penalty_value = float(penalty_value)
        self._ctx = ctx
        self._progress_callback = progress_callback
        self._cancellation_check = cancellation_check
        self._cancellation_control = cancellation_control

        self._objective_calls = 0
        self._best_cost = float("inf")
        self._warned_objective_keys: set[tuple[str, str, str]] = set()

    @property
    def objective_calls(self) -> int:
        """Number of distinct candidate objective calls made by this owner."""

        return int(self._objective_calls)

    def evaluate_outcome(self, params: np.ndarray) -> FitEvaluationOutcome:
        """Evaluate one candidate and return scientific status, never a penalty."""

        transport = self(np.asarray(params, dtype=float))
        outcome = getattr(transport, "outcome", None)
        if not isinstance(outcome, FitEvaluationOutcome):
            raise RuntimeError("Fit objective omitted its typed evaluation outcome.")
        return outcome

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

        def candidate_penalty_residuals() -> np.ndarray:
            blocks: list[np.ndarray] = []
            for payload in self._payloads:
                target_multipliers = _normalized_target_weight_multipliers(
                    species_list=payload.species_list,
                    target_weights=dict(payload.target_weights),
                )
                dataset_weight = _objective_dataset_weight_multiplier(
                    self._weights.get(payload.dataset_id, 1.0)
                )
                observations = _payload_observations(payload)
                for species_name in payload.species_list:
                    y_exp = np.asarray(
                        observations[species_name]["y"], dtype=float
                    ).reshape(-1)
                    blocks.append(
                        np.full_like(
                            y_exp,
                            float(dataset_weight)
                            * float(target_multipliers.get(species_name, 1.0))
                            * self._penalty_value,
                        )
                    )
            return (
                np.concatenate(blocks).astype(float, copy=False)
                if blocks
                else np.asarray([], dtype=float)
            )

        def finalize_candidate(
            *,
            parameters: Mapping[str, float],
            cost: float,
            improved: bool,
            best_results: Mapping[str, SimulationResultValue],
            publish_every_call: bool = False,
        ) -> None:
            self._objective_calls += 1
            if improved:
                self._best_cost = float(cost)
            if self._progress_callback is None:
                return
            if not (publish_every_call or improved or self._objective_calls % 10 == 0):
                return
            self._progress_callback(
                FitProgress(
                    objective_calls=self._objective_calls,
                    cost=float(cost),
                    parameters=parameters,
                    improved=bool(improved),
                    best_simulation_results_by_dataset=best_results,
                )
            )

        def reject_candidate(
            exc: BaseException,
            provenance: object,
            *,
            parameters: Mapping[str, float] | None = None,
        ) -> np.ndarray:
            penalty = candidate_penalty_residuals()
            if isinstance(exc, FitSimulationError) and is_candidate_domain_failure(exc):
                outcome = FitEvaluationOutcome.candidate_rejected(
                    exc,
                    transport_residuals=penalty,
                    provenance=dict(provenance) if isinstance(provenance, Mapping) else {},
                )
            elif isinstance(exc, (FittingCancelled, SimulationCancelled)):
                outcome = FitEvaluationOutcome.cancelled(exc)
            else:
                outcome = FitEvaluationOutcome.fatal(
                    exc,
                    provenance=dict(provenance) if isinstance(provenance, Mapping) else {},
                )
            self._ctx.set_error(exc, provenance)
            key = ("__candidate__", "__simulation__", str(exc).splitlines()[0])
            if key not in self._warned_objective_keys:
                self._warned_objective_keys.add(key)
                logger.warning("Fit candidate rejected: %s", exc)
            penalty_cost = finite_sum_of_squares(penalty)
            if penalty_cost is None:
                raise RuntimeError("Configured Fit candidate penalty overflowed.")
            finalize_candidate(
                parameters=dict(parameters or {}),
                cost=float(penalty_cost),
                improved=False,
                best_results={},
                publish_every_call=True,
            )
            return _ObjectiveTransport(penalty, outcome)

        try:
            param_dict = self._layout.shared_param_dict_from_vector(params)
        except FitSimulationError as exc:
            return reject_candidate(exc, {"stage": "log10_decode"})
        all_residuals: List[float] = []
        dataset_inputs: List[ObjectiveDatasetInput] = []

        for index, payload in enumerate(self._payloads):
            _raise_if_fitting_cancel_requested(self._cancellation_control)

            ds_id = payload.dataset_id
            try:
                optimizer_dataset_params = (
                    self._layout.dataset_var_params_for_dataset(ds_id, params)
                )
                coupled_dataset_params = self._layout.coupled_params_for_dataset(ds_id, params)
            except FitSimulationError as exc:
                return reject_candidate(exc, {"dataset": ds_id}, parameters=param_dict)
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
                cancellation_control=self._cancellation_control,
            )
        }

        for evaluation in simulation_evaluations.values():
            if not dataset_evaluation_is_fatal(evaluation):
                continue
            exc = evaluation.error
            if exc is None:
                raise RuntimeError("Fatal Fit evaluation omitted its exception.")
            self._ctx.set_error(exc, evaluation.error_provenance)
            raise exc

        for evaluation in simulation_evaluations.values():
            exc = evaluation.error
            if exc is not None and is_candidate_domain_failure(exc):
                return reject_candidate(
                    exc,
                    evaluation.error_provenance,
                    parameters=param_dict,
                )

        for item in dataset_inputs:
            _raise_if_fitting_cancel_requested(self._cancellation_control)

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
                self._ctx.set_error(exc, evaluation.error_provenance)
                if not is_candidate_domain_failure(exc):
                    raise exc
                return reject_candidate(
                    exc,
                    evaluation.error_provenance,
                    parameters=param_dict,
                )

            simulation = evaluation.result
            if simulation is None:
                raise FitSimulationError(
                    f"Dataset {ds_id!r}: Fit simulation lost its canonical result.",
                    failed_params=failed_param_snapshot,
                )
            sim_time = simulation.t
            segment_ids = simulation.segment_ids
            segment_keys = simulation.segment_keys
            discontinuity_ids = simulation.discontinuity_ids

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
                try:
                    model_series = fit_result_target_values(
                        simulation,
                        species_name,
                    )
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
                            else fit_result_target_values(simulation, x_name)
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
                    self._ctx.set_error(exc, {"dataset": ds_id})
                    if not is_candidate_domain_failure(exc):
                        raise
                    return reject_candidate(
                        exc,
                        {"dataset": ds_id},
                        parameters=param_dict,
                    )
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
                    err = FitCandidateDomainError(
                        f"Simulation produced non-finite values for dataset '{ds_id}', species '{species_name}'.",
                        stage="objective_nonfinite",
                        failed_params=failed_param_snapshot,
                    )
                    self._ctx.set_error(err, {"dataset": ds_id})
                    return reject_candidate(
                        err,
                        {"dataset": ds_id},
                        parameters=param_dict,
                    )

                with np.errstate(over="ignore", invalid="ignore"):
                    residual_vector = effective_weight * (
                        np.asarray(y_sim, dtype=float).reshape(-1)
                        - np.asarray(y_exp, dtype=float).reshape(-1)
                    )
                if not np.all(np.isfinite(residual_vector)):
                    err = FitCandidateDomainError(
                        f"Residuals contained non-finite values for dataset '{ds_id}'.",
                        stage="objective_residual_arithmetic",
                        failed_params=failed_param_snapshot,
                    )
                    self._ctx.set_error(err, {"dataset": ds_id})
                    return reject_candidate(
                        err,
                        {"dataset": ds_id},
                        parameters=param_dict,
                    )
                if finite_sum_of_squares(residual_vector) is None:
                    err = FitCandidateDomainError(
                        f"Residual square arithmetic overflowed for dataset '{ds_id}'.",
                        stage="objective_residual_arithmetic",
                        failed_params=failed_param_snapshot,
                    )
                    self._ctx.set_error(err, {"dataset": ds_id})
                    return reject_candidate(
                        err,
                        {"dataset": ds_id},
                        parameters=param_dict,
                    )
                all_residuals.extend(np.asarray(residual_vector, dtype=float).reshape(-1))

        residuals = np.asarray(all_residuals, dtype=float).ravel()

        finite_cost = finite_sum_of_squares(residuals)
        if finite_cost is None:
            err = FitCandidateDomainError(
                "Combined Fit residual square arithmetic overflowed.",
                stage="objective_residual_arithmetic",
            )
            return reject_candidate(err, {"stage": "objective"}, parameters=param_dict)
        cost = float(finite_cost)
        callback_params = dict(param_dict)
        for ds_id, var_map in self._layout.dataset_var_index.items():
            for var_name, idx in var_map.items():
                raw_val = float(params[idx])
                callback_params[f"{ds_id}::{var_name}"] = (
                    _decode_log10_physical(
                        raw_val,
                        coordinate=f"{ds_id}::{var_name}",
                    )
                    if self._layout.dataset_var_log10.get((ds_id, var_name))
                    else raw_val
                )
        for spec in self._layout.coupled_var_order:
            raw_val = float(params[self._layout.coupled_var_index[spec.variable_id]])
            callback_params[spec.variable_id] = (
                _decode_log10_physical(raw_val, coordinate=spec.variable_id)
                if spec.log10
                else raw_val
            )
        candidate_evaluations = {
            item.payload.dataset_id: evaluation.result
            for item in dataset_inputs
            if (
                (evaluation := simulation_evaluations.get(int(item.index)))
                is not None
                and evaluation.error is None
                and evaluation.result is not None
            )
        }
        complete_evaluations = len(candidate_evaluations) == len(dataset_inputs)
        improved = complete_evaluations and cost < self._best_cost
        finalize_candidate(
            parameters=callback_params,
            cost=cost,
            improved=improved,
            best_results=candidate_evaluations if improved else {},
        )
        _raise_if_fitting_cancelled(self._cancellation_check)

        outcome = FitEvaluationOutcome.converged(
            residuals,
            provenance={"stage": "objective", "objective_calls": self._objective_calls},
        )

        return _ObjectiveTransport(residuals, outcome)


def _raise_final_replay_candidate_error(
    exc: FitSimulationError,
    *,
    stage: str,
    failed_params: Optional[Dict[str, float]] = None,
) -> None:
    """Convert one nonfatal candidate outcome into typed terminal fatality."""

    raise FitFinalReplayError(
        "Final Fit replay rejected the optimizer's terminal candidate.",
        typed_cause=exc,
        stage=stage,
        failed_params=failed_params,
    ) from exc


def assemble_global_fit_result(
    *,
    fit_evaluators: Mapping[str, object],
    payloads: List[FitDatasetSpec],
    layout: FitParameterLayout,
    fitted_params: Dict[str, float],
    fixed_dataset_params: Dict[str, Dict[str, float]],
    combined_dataset_params: Dict[str, Dict[str, float]],
    weights: Dict[str, float],
    cancellation_check: Optional[Callable[[], bool]],
    success: bool,
    message: str,
    nfev: int,
    optimizer_diagnostic: Optional[FitDiagnostic] = None,
    cancellation_control: SimulationCancellationControl | None = None,
) -> GlobalFitResult:
    dataset_info = []
    total_points = 0
    global_weighted_residual_blocks: List[np.ndarray] = []
    model_series_map: Dict[str, Dict[str, np.ndarray]] = {}
    model_x_by_species_map: Dict[str, Dict[str, np.ndarray]] = {}
    model_segment_ids_map: Dict[str, Dict[str, np.ndarray]] = {}
    model_segment_keys_map: Dict[str, Dict[str, tuple[str, ...]]] = {}
    model_discontinuity_ids_map: Dict[str, Dict[str, np.ndarray]] = {}
    model_selection_provenance_map: Dict[str, Dict[str, tuple[str, ...]]] = {}
    model_observation_ids_map: Dict[str, Dict[str, tuple[str, ...]]] = {}
    model_raw_columns_map: Dict[str, Dict[str, tuple[str, ...]]] = {}
    model_source_rows_map: Dict[str, Dict[str, tuple[int, ...]]] = {}
    simulation_results_by_dataset: Dict[str, SimulationResultValue] = {}

    dataset_inputs: List[ObjectiveDatasetInput] = []
    for index, payload in enumerate(payloads):
        ds_id = payload.dataset_id
        _raise_if_fitting_cancelled(cancellation_check)
        _raise_if_fitting_cancel_requested(cancellation_control)

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
            cancellation_control=cancellation_control,
        )
    }
    for evaluation in simulation_evaluations.values():
        if evaluation.error is not None:
            if is_candidate_domain_failure(evaluation.error):
                _raise_final_replay_candidate_error(
                    evaluation.error,
                    stage="final_replay_simulation",
                )
            raise evaluation.error
    expected_evaluation_indexes = set(range(len(dataset_inputs)))
    if set(simulation_evaluations) != expected_evaluation_indexes:
        raise RuntimeError(
            "Final Fit replay did not evaluate every included dataset."
        )

    for item in dataset_inputs:
        payload = item.payload
        ds_id = payload.dataset_id
        species_list = payload.species_list
        observations = _payload_observations(payload)
        x_name = payload.x_name
        x_obs_by_species = _payload_x_obs_by_species(payload) if x_name != "t" else {}
        x_mode = payload.x_mode
        failed_param_snapshot = item.failed_param_snapshot
        _raise_if_fitting_cancelled(cancellation_check)
        _raise_if_fitting_cancel_requested(cancellation_control)

        evaluation = simulation_evaluations.get(int(item.index))
        if evaluation is None:
            raise RuntimeError(f"Missing fitting simulation result for dataset '{ds_id}'.")
        simulation = evaluation.result
        if simulation is None:
            raise RuntimeError(
                f"Final Fit replay omitted the canonical result for dataset {ds_id!r}."
            )
        simulation_results_by_dataset[ds_id] = simulation
        sim_time = simulation.t
        segment_ids = simulation.segment_ids
        segment_keys = simulation.segment_keys
        discontinuity_ids = simulation.discontinuity_ids

        residual_blocks: List[np.ndarray] = []
        weighted_residual_blocks: List[np.ndarray] = []
        r_squared_by_series: Dict[str, FitMetric] = {}
        for species_name in species_list:
            species_obs = observations[species_name]
            obs_t = np.asarray(species_obs["t"], dtype=float).reshape(-1)
            y_exp = np.asarray(species_obs["y"], dtype=float).reshape(-1)
            species_x_obs = (
                np.asarray(x_obs_by_species.get(species_name), dtype=float).reshape(-1)
                if species_name in x_obs_by_species
                else None
            )
            model_series = fit_result_target_values(
                simulation,
                species_name,
            )
            if x_name != "t" and species_x_obs is None:
                raise FitSimulationError(
                    f"Dataset '{ds_id}' species '{species_name}' is missing x observations for X='{x_name}'.",
                    failed_params=failed_param_snapshot,
                    details={"fatal": True, "stage": "final_replay_alignment"},
                )
            try:
                selected_rows = select_fit_rows(
                    model_y=np.asarray(model_series, dtype=float),
                    model_x=(
                        None
                        if x_name == "t"
                        else fit_result_target_values(simulation, x_name)
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
                    segment_keys=tuple(segment_keys),
                    discontinuity_ids=np.asarray(
                        discontinuity_ids,
                        dtype=np.int64,
                    ),
                    dataset_label=ds_id,
                    y_name=species_name,
                )
            except FitSimulationError as exc:
                if is_candidate_domain_failure(exc):
                    _raise_final_replay_candidate_error(
                        exc,
                        stage="final_replay_alignment",
                        failed_params=failed_param_snapshot,
                    )
                raise
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

            if not np.all(np.isfinite(y_sim_resid)):
                cause = FitCandidateDomainError(
                    f"Non-finite aligned series for dataset '{ds_id}', species '{species_name}'.",
                    stage="parametric_x_alignment",
                    failed_params=failed_param_snapshot,
                )
                _raise_final_replay_candidate_error(
                    cause,
                    stage="final_replay_alignment",
                    failed_params=failed_param_snapshot,
                )
            with np.errstate(over="ignore", invalid="ignore"):
                raw_residual = np.asarray(y_sim_resid, dtype=float) - y_exp
            if not np.all(np.isfinite(raw_residual)):
                cause = FitCandidateDomainError(
                    f"Non-finite final residuals for dataset '{ds_id}', species '{species_name}'.",
                    stage="objective_residual_arithmetic",
                    failed_params=failed_param_snapshot,
                )
                _raise_final_replay_candidate_error(
                    cause,
                    stage="final_replay_metrics",
                    failed_params=failed_param_snapshot,
                )
            residual_blocks.append(raw_residual)
            r_squared_by_series[str(species_name)] = r_squared_metric(
                y_exp,
                raw_residual,
            )
            target_multiplier = float(
                _normalized_target_weight_multipliers(
                    species_list=species_list,
                    target_weights=dict(payload.target_weights),
                )[species_name]
            )
            with np.errstate(over="ignore", invalid="ignore"):
                weighted_residual = (
                    _objective_dataset_weight_multiplier(
                        weights.get(ds_id, 1.0)
                    )
                    * target_multiplier
                    * raw_residual
                )
            if not np.all(np.isfinite(weighted_residual)):
                cause = FitCandidateDomainError(
                    f"Weighted residual arithmetic overflowed for dataset '{ds_id}'.",
                    stage="objective_residual_arithmetic",
                    failed_params=failed_param_snapshot,
                )
                _raise_final_replay_candidate_error(
                    cause,
                    stage="final_replay_metrics",
                    failed_params=failed_param_snapshot,
                )
            weighted_residual_blocks.append(weighted_residual)

        residuals = np.concatenate(residual_blocks) if residual_blocks else np.asarray([], dtype=float)
        weighted_residuals = (
            np.concatenate(weighted_residual_blocks)
            if weighted_residual_blocks
            else np.asarray([], dtype=float)
        )
        if residuals.size == 0 or weighted_residuals.size != residuals.size:
            raise FitSimulationError(
                f"Final Fit replay produced incomplete residuals for dataset '{ds_id}'.",
                failed_params=failed_param_snapshot,
                details={"fatal": True, "stage": "final_replay_metrics"},
            )
        metrics = fit_error_metrics(weighted_residuals)

        dataset_info.append(
            DatasetFitInfo(
                dataset_id=ds_id,
                r_squared_by_series=dict(r_squared_by_series),
                weighted_residual_sse=metrics.weighted_sse,
                weighted_residual_mse=metrics.weighted_mse,
                weighted_rmse=metrics.weighted_rmse,
                weighted_mae=metrics.weighted_mae,
                n_points=int(residuals.size),
                weight=weights.get(ds_id, 1.0),
            )
        )

        global_weighted_residual_blocks.append(weighted_residuals)
        total_points += int(residuals.size)

    if total_points <= 0:
        raise FitSimulationError(
            "Final Fit replay produced no residual observations.",
            details={"fatal": True, "stage": "final_replay_metrics"},
        )
    aggregate_metrics = fit_error_metrics(
        np.concatenate(global_weighted_residual_blocks)
    )

    included_dataset_ids = {payload.dataset_id for payload in payloads}
    if (
        set(combined_dataset_params) != included_dataset_ids
        or set(simulation_results_by_dataset) != included_dataset_ids
        or {info.dataset_id for info in dataset_info} != included_dataset_ids
    ):
        raise RuntimeError(
            "Final Fit result rosters must exactly match the included datasets."
        )
    optimizer_converged = bool(success)
    completion_optimizer_diagnostic = optimizer_diagnostic
    if (
        optimizer_converged
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
    if (not optimizer_converged) or completion_optimizer_diagnostic is not None:
        completion_status = "warn"
    else:
        completion_status = "ok"

    if completion_status == "ok":
        completion_optimizer_diagnostic = None

    completion = GlobalFitCompletion(
        status=completion_status,
        optimizer_converged=optimizer_converged,
        optimizer_diagnostic=completion_optimizer_diagnostic,
    )
    completion_message = completion_result_message(
        base_message=message,
        completion=completion,
        optimizer_diagnostic=completion_optimizer_diagnostic,
    )

    if completion.status == "ok":
        logger.info("Global fit complete with typed weighted metrics.")
    elif completion.status == "warn":
        logger.warning("Global fit complete with warnings: %s", completion_message)
    for info in dataset_info:
        logger.debug(
            "  %s: weighted SSE=%s, weighted RMSE=%s",
            info.dataset_id,
            info.weighted_residual_sse.value,
            info.weighted_rmse.value,
        )

    return GlobalFitResult(
        shared_params=fitted_params,
        dataset_params=combined_dataset_params,
        weighted_residual_sse=aggregate_metrics.weighted_sse,
        weighted_residual_mse=aggregate_metrics.weighted_mse,
        dataset_info=dataset_info,
        nfev=nfev,
        message=completion_message,
        scipy_nfev=nfev,
        terminal_replay_count=1,
        objective_call_count=None,
        completion=completion,
        model_series=model_series_map,
        model_x_by_species=model_x_by_species_map,
        model_segment_ids=model_segment_ids_map,
        model_segment_keys=model_segment_keys_map,
        model_discontinuity_ids=model_discontinuity_ids_map,
        model_selection_provenance=model_selection_provenance_map,
        model_observation_ids=model_observation_ids_map,
        model_raw_columns=model_raw_columns_map,
        model_source_rows=model_source_rows_map,
        simulation_results_by_dataset=simulation_results_by_dataset,
    )
