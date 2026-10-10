"""
Global fitting execution boundary.

This module owns concurrent canonical-order multi-dataset candidate evaluation, final replay,
diagnostics, and canonical result assembly. Public API composition remains in
``global_fitting.fit_global``.
"""

from __future__ import annotations

import logging
import math
import time
from concurrent.futures import Executor
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np

from kindred.core.analysis.parametric_alignment import (
    CandidateParametricAlignmentError,
    ParametricProjection,
    align_parametric,
    project_parametric_y,
)
from kindred.core.analysis.dataset_parameter_overrides import (
    FitCoupledVariableParamSpec,
    FitDatasetParameterOverrides,
)
from kindred.core.analysis.fit_dataset_payload import FitDatasetSpec
from kindred.core.analysis.fit_evaluation_outcome import ScientificUnavailable
from kindred.core.analysis.fit_statistics import (
    FitMetric,
    finite_positive_mean_normalize,
    finite_sum_of_squares,
    fit_error_metrics,
    r_squared_metric,
    root_mean_square,
)
from kindred.core.exceptions import (
    FitCandidateDomainError,
    FittingCancelled,
    FitSimulationError,
    SimulationCancelled,
)
from kindred.core.fitting_evaluation import (
    FITTING_PARAM_ORIGIN_CONFIGURED_DATASET,
    FITTING_PARAM_ORIGIN_OPTIMIZER_DATASET,
    FITTING_PARAM_ORIGIN_OPTIMIZER_SHARED,
)
from kindred.core.project_presentation_state import SimulationSeriesAxisSelection
from kindred.core.result_curve_projection import alignment_rows
from kindred.core.result_repository import SimulationResultValue
from kindred.core.simulator.solver_types import SimulationCancellationControl
from kindred.core.simulation_failure import (
    FactualCause,
    factual_cause_from_exception,
)

if TYPE_CHECKING:
    from kindred.core.analysis.fit_uncertainty import (
        ProfileLikelihoodResult,
        ResidualBootstrapResult,
    )

logger = logging.getLogger(__name__)
_DATASET_DEADLINE_SECONDS = 10.0


__all__ = [
    "FitCoordinate",
    "FitProgress",
    "DatasetFitInfo",
    "GlobalFitResult",
    "FitParameterLayout",
    "ObjectiveDatasetInput",
    "GlobalFitObjective",
    "build_parameter_layout_for_dataset_ids",
    "normalize_weights",
    "evaluate_dataset_simulation",
    "fit_result_target_values",
    "parameter_group_variable_id",
]


@dataclass(frozen=True, slots=True)
class FitCoordinate:
    """Collision-proof internal optimizer coordinate identity.

    ``display_name`` is a report/UI serialization only.  Optimizer maps retain
    the typed identity so a shared ``D::p`` cannot alias dataset ``D``
    parameter ``p``.  A ``group`` coordinate is the public uncertainty target
    of one Per-dataset parameter group, identified by its parameter and origin
    dataset; the optimizer moves it as the coupled variable named by
    :func:`parameter_group_variable_id`.
    """

    scope: Literal["shared", "dataset", "coupled", "group"]
    name: str
    dataset_id: str | None = None

    # Built from a Fit setup scope literal, an admitted parameter name and,
    # for dataset and group coordinates, a captured dataset identity.

    @property
    def display_name(self) -> str:
        if self.scope == "dataset":
            return f"{self.dataset_id}::{self.name}"
        if self.scope == "group":
            return f"group {self.dataset_id}::{self.name}"
        return self.name

    def __str__(self) -> str:
        return self.display_name


def parameter_group_variable_id(parameter_name: str, origin_dataset_id: str) -> str:
    """Stable coupled-variable ID of one Per-dataset parameter group."""

    return f"parameter:{parameter_name}:{origin_dataset_id}"


def _decode_log10_physical(raw_value: object, *, coordinate: str) -> float:
    """Decode one optimizer Log10 coordinate without clipping or fabrication."""

    try:
        value = float(raw_value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as exc:
        raise FitCandidateDomainError(
            f"Invalid Log10 optimizer coordinate for {coordinate!r}.",
            stage="log10_decode",
        ) from exc
    if not math.isfinite(value):
        raise FitCandidateDomainError(
            f"Nonfinite Log10 optimizer coordinate for {coordinate!r}.",
            stage="log10_decode",
        )
    try:
        decoded = math.pow(10.0, value)
    except (OverflowError, ValueError) as exc:
        raise FitCandidateDomainError(
            f"Log10 optimizer coordinate for {coordinate!r} overflowed.",
            stage="log10_decode",
        ) from exc
    if not math.isfinite(decoded):
        raise FitCandidateDomainError(
            f"Log10 optimizer coordinate for {coordinate!r} decoded non-finitely.",
            stage="log10_decode",
        )
    return float(decoded)


def _decode_admitted_physical(
    raw_value: float,
    *,
    encoded: tuple[float, float, float],
    physical: tuple[float, float, float],
    coordinate: str,
) -> float:
    """Decode one Log10 value; admitted starts and bounds decode exactly.

    ``encoded``/``physical`` hold (start, lower, upper).  Inside the optimizer
    box the decoded value is clamped to the admitted bounds, which only ever
    corrects floating-point rounding; values outside stay for rejection.
    """

    for encoded_value, physical_value in zip(encoded, physical):
        if raw_value == encoded_value:
            return float(physical_value)
    decoded = _decode_log10_physical(raw_value, coordinate=coordinate)
    if encoded[1] <= raw_value <= encoded[2]:
        return min(max(decoded, float(physical[1])), float(physical[2]))
    return decoded


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
    """Compact optimizer progress; simulation curves are callback-transient only."""

    objective_calls: int
    cost: float
    optimizer_vector: np.ndarray
    improved: bool = False
    transient_simulation_results_by_dataset: Mapping[str, SimulationResultValue] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        # The objective reports its own finite cost, the candidate vector it
        # admitted and its canonical simulation results.
        vector = np.asarray(self.optimizer_vector, dtype=float).copy()
        vector.setflags(write=False)
        object.__setattr__(self, "optimizer_vector", vector)
        object.__setattr__(
            self,
            "transient_simulation_results_by_dataset",
            MappingProxyType(dict(self.transient_simulation_results_by_dataset)),
        )


@dataclass(frozen=True, slots=True)
class DatasetFitInfo:
    """Complete direct scientific evaluation for one included dataset.

    The direct dataset operation returns this natural owner.  It carries the
    canonical simulation together with the aligned model rows and residual
    blocks that objective and terminal replay consume immediately; no opaque
    positional fact bundle is needed between those owners.
    """
    dataset_id: str  # Identifier for this dataset
    r_squared_by_series: Dict[str, FitMetric]  # Ordinary, unweighted R² by series
    # Unweighted RMS residual by series in its scientific unit; the X block
    # of a measured-X dataset is keyed by its X quantity.
    raw_rms_by_series: Dict[str, FitMetric]
    weighted_residual_sse: FitMetric  # Exact contribution to the objective
    weighted_residual_mse: FitMetric  # Exact objective contribution per observation
    weighted_rmse: FitMetric
    weighted_mae: FitMetric
    weight: float  # Weight used in global objective
    simulation_result: SimulationResultValue
    residuals: np.ndarray
    weighted_residuals: np.ndarray
    model_series: Mapping[str, np.ndarray]
    x_projection: ParametricProjection | None

    def __post_init__(self) -> None:
        # The evaluator owns admission of these typed values.  Keep the
        # arrays immutable without copying them and shallow-freeze the
        # mappings so DatasetFitInfo remains an immutable terminal owner.
        self.residuals.setflags(write=False)
        self.weighted_residuals.setflags(write=False)
        object.__setattr__(
            self,
            "r_squared_by_series",
            MappingProxyType(dict(self.r_squared_by_series)),
        )
        object.__setattr__(
            self,
            "raw_rms_by_series",
            MappingProxyType(dict(self.raw_rms_by_series)),
        )
        object.__setattr__(
            self, "model_series", MappingProxyType(dict(self.model_series)),
        )


@dataclass(frozen=True, slots=True)
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
    message : str
        Optimization termination message
    """
    shared_params: Dict[str, float]
    dataset_params: Dict[str, Dict[str, float]]
    weighted_residual_sse: FitMetric
    weighted_residual_mse: FitMetric
    dataset_info: List[DatasetFitInfo]
    message: str
    optimizer_vector: np.ndarray | None
    status: Literal["ok", "warn"]
    optimizer_converged: bool
    optimizer_diagnostic: FactualCause | None = None
    profile_likelihood_result: ProfileLikelihoodResult | None = None
    residual_bootstrap_result: ResidualBootstrapResult | None = None

    def __post_init__(self) -> None:
        # Its sole producer publishes the settled replay state: a literal
        # status with a diagnostic exactly for "warn", the admitted finite
        # optimizer vector, its decoded finite parameters, and one
        # DatasetFitInfo per included dataset in roster order.  This carrier
        # freezes them.
        optimizer_vector = np.asarray(self.optimizer_vector, dtype=float).copy()
        optimizer_vector.setflags(write=False)
        object.__setattr__(self, "optimizer_vector", optimizer_vector)
        object.__setattr__(
            self,
            "shared_params",
            MappingProxyType({name: float(value) for name, value in self.shared_params.items()}),
        )
        object.__setattr__(
            self,
            "dataset_params",
            MappingProxyType(
                {
                    dataset_id: MappingProxyType(
                        {name: float(value) for name, value in values.items()}
                    )
                    for dataset_id, values in self.dataset_params.items()
                }
            ),
        )
        object.__setattr__(self, "dataset_info", tuple(self.dataset_info))


@dataclass(frozen=True, slots=True)
class _FitReplayState:
    """Private replay evidence held until the sole publishable result build."""

    shared_params: Mapping[str, float]
    dataset_params: Mapping[str, Mapping[str, float]]
    weighted_residual_sse: FitMetric
    weighted_residual_mse: FitMetric
    dataset_info: tuple[DatasetFitInfo, ...]
    message: str
    status: Literal["ok", "warn"]
    optimizer_converged: bool
    optimizer_diagnostic: FactualCause | None
    optimizer_vector: np.ndarray
    weighted_residuals: np.ndarray
    profile_likelihood_result: ProfileLikelihoodResult | None = None
    residual_bootstrap_result: ResidualBootstrapResult | None = None

    def __post_init__(self) -> None:
        # The final replay admitted this vector as a candidate and its
        # residuals through each dataset's finite residual check.
        vector = np.asarray(self.optimizer_vector, dtype=float).copy()
        vector.setflags(write=False)
        object.__setattr__(self, "optimizer_vector", vector)
        residuals = np.asarray(self.weighted_residuals, dtype=float).reshape(-1).copy()
        residuals.setflags(write=False)
        object.__setattr__(self, "weighted_residuals", residuals)


def _construct_global_fit_result(
    state: _FitReplayState,
) -> GlobalFitResult:
    """Construct the sole immutable publishable result after settlement."""

    return GlobalFitResult(
        shared_params=dict(state.shared_params),
        dataset_params={
            str(dataset_id): dict(values)
            for dataset_id, values in state.dataset_params.items()
        },
        weighted_residual_sse=state.weighted_residual_sse,
        weighted_residual_mse=state.weighted_residual_mse,
        dataset_info=list(state.dataset_info),
        message=state.message,
        optimizer_vector=state.optimizer_vector,
        status=state.status,
        optimizer_converged=state.optimizer_converged,
        optimizer_diagnostic=state.optimizer_diagnostic,
        profile_likelihood_result=state.profile_likelihood_result,
        residual_bootstrap_result=state.residual_bootstrap_result,
    )


def _compact_failure_message(failure: FactualCause) -> str:
    raw_message = failure.message
    return " ".join(part.strip() for part in str(raw_message).splitlines() if part.strip())


def fit_result_message(
    *,
    base_message: str,
    status: Literal["ok", "warn"],
    optimizer_converged: bool,
    optimizer_diagnostic: FactualCause | None,
) -> str:
    if status == "ok":
        if not optimizer_converged:
            stopped = str(base_message or "").strip()
            if not stopped:
                stopped = "the optimizer stopped without a termination message"
            return f"Fit completed. Optimizer stopped: {stopped}"
        return str(base_message)
    if status == "warn":
        # A Stop's own message already says the fit stopped.
        if optimizer_diagnostic is not None and optimizer_diagnostic.type == "FittingStopped":
            return str(optimizer_diagnostic.message)
        warning_parts: list[str] = []
        # A converged optimizer whose endpoint was replaced did not stop early.
        replaced_after_convergence = (
            optimizer_diagnostic is not None
            and optimizer_diagnostic.type == "FittingUnavailableEndpointReplaced"
            and optimizer_diagnostic.details.get("optimizer_converged") is True
        )
        if not optimizer_converged:
            fallback = str(base_message or "").strip()
            if fallback and replaced_after_convergence:
                warning_parts.append(fallback)
            elif fallback:
                warning_parts.append(f"Optimizer stopped: {fallback}")
            else:
                warning_parts.append("Optimizer stopped without a termination message")
        if optimizer_diagnostic is not None:
            detail = _compact_failure_message(optimizer_diagnostic)
            if detail and detail != str(base_message or "").strip():
                warning_parts.append(detail)
        if warning_parts:
            return "Global fit completed with warnings: " + " ".join(warning_parts)
        fallback = str(base_message or "").strip()
        if fallback and "success" not in fallback.lower():
            return fallback
        return "Global fit completed with warnings."
    raise AssertionError(f"Unsupported completed Fit status: {status!r}")


def normalize_weights(payloads: List[FitDatasetSpec], weights: Optional[Dict[str, float]]) -> Dict[str, float]:
    # User dataset weights are the admitted positive weights of the included
    # datasets; implicit weights are positive by construction.
    provided_weights = weights or {}
    return finite_positive_mean_normalize({
        payload.dataset_id: float(
            provided_weights.get(payload.dataset_id, 1.0 / max(1, payload.point_count))
        )
        for payload in payloads
    })


def _normalized_target_weight_multipliers(
    *,
    species_list: Sequence[str],
    target_weights: Optional[Dict[str, float]],
) -> Dict[str, float]:
    # Target weights are the admitted positive weights of the captured roster.
    normalized = finite_positive_mean_normalize(
        {name: float(target_weights[name]) for name in species_list}
    )
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
    """Align values and their direct canonical row identity together.

    The canonical result supplies one value and one segment, key and
    discontinuity identity per strictly increasing row.
    """
    values = np.asarray(series, dtype=float).reshape(-1)
    sim_t = np.asarray(sim_time, dtype=float).reshape(-1)
    target_t = np.asarray(target_time, dtype=float).reshape(-1)
    segments = np.asarray(segment_ids, dtype=np.int64).reshape(-1)
    keys = tuple(segment_keys)
    discontinuities = np.asarray(discontinuity_ids, dtype=np.int64).reshape(-1)

    aligned = np.empty(target_t.size, dtype=float)
    aligned_segments = np.empty(target_t.size, dtype=np.int64)
    aligned_discontinuities = np.empty(target_t.size, dtype=np.int64)
    aligned_keys: list[str] = []
    selection: list[str] = []
    insertion_positions = np.searchsorted(sim_t, target_t, side="left")
    for index, time_value in enumerate(target_t):
        right = int(insertion_positions[index])
        if right < sim_t.size and sim_t[right] == time_value:
            aligned[index] = values[right]
            aligned_segments[index] = segments[right]
            aligned_discontinuities[index] = discontinuities[right]
            aligned_keys.append(keys[right])
            selection.append("direct")
            continue
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


def _select_fit_rows(
    *,
    model_y: np.ndarray,
    sim_time: np.ndarray,
    observation_time: np.ndarray,
    segment_ids: np.ndarray,
    segment_keys: Sequence[str],
    discontinuity_ids: np.ndarray,
) -> np.ndarray:
    """Align one time-based Y series to its admitted observation times."""

    obs_t = np.asarray(observation_time, dtype=float).reshape(-1)
    selected = _align_series(
        np.asarray(model_y, dtype=float),
        np.asarray(sim_time, dtype=float),
        obs_t,
        np.asarray(segment_ids, dtype=np.int64),
        segment_keys,
        np.asarray(discontinuity_ids, dtype=np.int64),
    )
    return np.asarray(selected[0], dtype=float)


def _objective_dataset_weight_multiplier(weight: float) -> float:
    # Normalized dataset weights are finite and positive.
    return float(np.sqrt(weight))


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


def _payload_x_observations(payload: FitDatasetSpec) -> Dict[str, object]:
    """Return one canonical X block plus its dataset row lookup."""

    if payload.x_name == "t" or payload.x_observations is None:
        return {}
    source = payload.x_observations
    rows = payload.x_source_rows
    by_row = {
        int(row): float(value)
        for row, value in zip(source.source_rows, source.y, strict=True)
    }
    return {
        "x": np.asarray([by_row[row] for row in rows], dtype=float),
        "source_rows": rows,
        "raw_observation_id": source.raw_observation_id,
        "raw_column": source.raw_column,
    }


def fit_result_target_values(
    result: SimulationResultValue,
    name: str,
    *,
    dataset_id: str,
) -> np.ndarray:
    """Resolve a Fit target from the canonical scalar/series result split.

    The retained Algebra DAG and evaluator decide whether a let is scalar or
    row-varying.  This boundary consumes that typed result directly; it never
    infers type from static parser tables or sampled values.
    """

    target_name = str(name)
    details = {"fatal": True, "stage": "fit_target_lookup", "dataset": dataset_id, "target": target_name}
    time_size = np.asarray(result.t, dtype=float).reshape(-1).size
    if target_name in result.series:
        values = np.asarray(result.series[target_name], dtype=float).reshape(-1)
    elif target_name in result.scalars:
        value = float(result.scalars[target_name])
        if not np.isfinite(value):
            raise FitSimulationError(
                f"Fit scalar target {target_name!r} is nonfinite.",
                details=details,
            )
        values = np.full(time_size, value, dtype=float)
    else:
        algebra_cause = next(
            (cause for cause in result.algebra_errors if cause.details.get("name") == target_name), None,
        )
        error = FitSimulationError(
            f"Fit target {target_name!r} could not be calculated." if algebra_cause is not None else
            f"Fit target {target_name!r} is absent from the canonical simulation scalar/series outputs.",
            details=details,
        )
        if algebra_cause is not None:
            error.factual_cause = replace(
                factual_cause_from_exception(error, domain="fit", origin="fit_target_lookup"),
                cause=algebra_cause,
            )
        raise error
    if values.size != time_size or not np.all(np.isfinite(values)):
        raise FitSimulationError(
            f"Fit target {target_name!r} is not a finite row-aligned output.",
            details=details,
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
    # Admitted physical start and bounds per optimizer index, so admitted
    # values decode exactly rather than through 10**log10(value).
    physical_x0: np.ndarray
    physical_lower: np.ndarray
    physical_upper: np.ndarray

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
        for name in ("x0", "lower", "upper", "physical_x0", "physical_lower", "physical_upper"):
            values = np.asarray(getattr(self, name), dtype=float).reshape(-1).copy()
            values.setflags(write=False)
            object.__setattr__(self, name, values)

    def _coordinate_keys(self) -> tuple[FitCoordinate, ...]:
        keys = tuple(FitCoordinate("shared", name) for name in self.param_names)
        keys += tuple(
            FitCoordinate("dataset", parameter_name, ds_id)
            for ds_id, parameter_name in self.dataset_var_order
        )
        keys += tuple(
            FitCoordinate("coupled", item.variable_id)
            for item in self.coupled_var_order
        )
        return keys

    def opt_param_keys(self) -> tuple[FitCoordinate, ...]:
        return self._coordinate_keys()

    def initial_params(self) -> Dict[FitCoordinate, float]:
        keys = self.opt_param_keys()
        return {key: float(self.x0[i]) for i, key in enumerate(keys)}

    def bounds_dict(self) -> Dict[FitCoordinate, Tuple[float, float]]:
        keys = self.opt_param_keys()
        return {key: (float(self.lower[i]), float(self.upper[i])) for i, key in enumerate(keys)}

    def decode_coordinate(self, coordinate: FitCoordinate, optimizer_value: float) -> float:
        index = self.opt_param_keys().index(coordinate)
        if coordinate.scope == "shared":
            log10 = bool(self.shared_log10.get(coordinate.name, False))
        elif coordinate.scope == "dataset":
            log10 = bool(self.dataset_var_log10.get((coordinate.dataset_id, coordinate.name), False))
        else:
            log10 = next(item.log10 for item in self.coupled_var_order if item.variable_id == coordinate.name)
        return self._decode_index(index, float(optimizer_value), log10=log10, coordinate=coordinate.display_name)

    def _decode_index(
        self, index: int, raw_value: float, *, log10: bool, coordinate: str,
    ) -> float:
        if not log10:
            return float(raw_value)
        return _decode_admitted_physical(
            float(raw_value),
            encoded=(float(self.x0[index]), float(self.lower[index]), float(self.upper[index])),
            physical=(
                float(self.physical_x0[index]),
                float(self.physical_lower[index]),
                float(self.physical_upper[index]),
            ),
            coordinate=coordinate,
        )

    def validate_optimizer_vector(self, vector: np.ndarray) -> None:
        """Validate exact optimizer shape and finiteness."""

        values = np.asarray(vector, dtype=float)
        if (
            values.ndim != 1
            or values.size != self.x0.size
            or not np.all(np.isfinite(values))
        ):
            raise FitCandidateDomainError(
                "Fit candidate has a nonfinite or incomplete optimizer vector.",
                stage="optimizer_vector_domain",
            )

    def shared_param_dict_from_vector(self, vector: np.ndarray) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for idx, name in enumerate(self.param_names):
            out[name] = self._decode_index(
                idx, float(vector[idx]),
                log10=bool(self.shared_log10.get(name)), coordinate=name,
            )
        return out

    def dataset_var_params_for_dataset(self, ds_id: str, vector: np.ndarray) -> Dict[str, float]:
        out: Dict[str, float] = {}
        var_map = self.dataset_var_index.get(ds_id)
        if not var_map:
            return out
        for var_name, idx in var_map.items():
            out[var_name] = self._decode_index(
                idx, float(vector[idx]),
                log10=bool(self.dataset_var_log10.get((ds_id, var_name))),
                coordinate=f"{ds_id}::{var_name}",
            )
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
            out[spec.parameter_name] = self._decode_index(
                index, float(vector[index]),
                log10=spec.log10, coordinate=spec.variable_id,
            )
        return out


def build_parameter_layout_for_dataset_ids(
    *,
    dataset_ids: Sequence[str],
    shared_params: Dict[str, float],
    dataset_overrides: Sequence[FitDatasetParameterOverrides],
    coupled_variables: Sequence[FitCoupledVariableParamSpec],
    bounds: Dict[str, Tuple[float, float]],
    log10_params: Optional[Dict[str, bool]],
) -> FitParameterLayout:
    """Build the optimizer layout from the admitted included-dataset roster.

    Coordinates follow the roster's order (the project's dataset order) and
    authored names, never the values of opaque dataset or set identities.
    """

    # The included roster, the coupled members and the shared bounds come from
    # one compiled Fit setup, whose dataset and variable identities are unique.
    normalized_dataset_ids = tuple(dataset_ids)
    param_names = sorted(str(name) for name in shared_params.keys())
    log10_params_norm = {str(k): bool(v) for k, v in (log10_params or {}).items()}

    shared_initials: List[float] = []
    shared_log10: Dict[str, bool] = {}
    # (start, lower, upper) admitted physical values per optimizer index.
    shared_physical: List[tuple[float, float, float]] = []
    for name in param_names:
        flag = bool(log10_params_norm.get(name, False))
        shared_log10[name] = flag
        value = float(shared_params[name])
        if flag:
            shared_initials.append(math.log10(value))
        else:
            shared_initials.append(value)

    dataset_var_order: List[Tuple[str, str]] = []
    dataset_var_index: Dict[str, Dict[str, int]] = {}
    dataset_var_inits: List[float] = []
    dataset_var_bounds: List[Tuple[float, float]] = []
    dataset_var_log10: Dict[Tuple[str, str], bool] = {}
    dataset_var_physical: List[tuple[float, float, float]] = []

    overrides_by_id = {item.dataset_id: item for item in dataset_overrides}
    for ds_id in normalized_dataset_ids:
        specs = overrides_by_id[ds_id].variable_params
        if not specs:
            continue
        dataset_var_index.setdefault(ds_id, {})
        for spec in specs:
            param_name = spec.name
            log10_flag = bool(spec.log10)
            dataset_var_order.append((ds_id, param_name))
            dataset_var_index[ds_id][param_name] = (
                len(param_names) + len(dataset_var_inits)
            )
            init_val = float(spec.initial)
            min_val = float(spec.minimum)
            max_val = float(spec.maximum)
            if log10_flag:
                dataset_var_inits.append(math.log10(init_val))
                dataset_var_bounds.append((math.log10(min_val), math.log10(max_val)))
            else:
                dataset_var_inits.append(init_val)
                dataset_var_bounds.append((min_val, max_val))
            dataset_var_physical.append((init_val, min_val, max_val))
            dataset_var_log10[(ds_id, param_name)] = log10_flag

    # Variables sharing a name have disjoint members, so the first member's
    # roster position and the name order every coupled variable uniquely.
    roster_position = {ds_id: index for index, ds_id in enumerate(normalized_dataset_ids)}
    coupled_order = sorted(
        coupled_variables,
        key=lambda item: (
            min(roster_position[ds_id] for ds_id in item.dataset_ids),
            item.parameter_name,
        ),
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
        # Values, bounds and Log10 positivity arrive admitted by the Fit setup carriers.
        low, high = bounds[name]
        if shared_log10.get(name):
            shared_lower.append(math.log10(low))
            shared_upper.append(math.log10(high))
        else:
            shared_lower.append(low)
            shared_upper.append(high)
        shared_physical.append((float(shared_params[name]), low, high))

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
    physical = np.array(
        shared_physical
        + dataset_var_physical
        + [(float(item.initial), float(item.minimum), float(item.maximum)) for item in coupled_order],
        dtype=float,
    ).reshape(-1, 3)
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
        physical_x0=physical[:, 0],
        physical_lower=physical[:, 1],
        physical_upper=physical[:, 2],
    )


@dataclass(frozen=True)
class ObjectiveDatasetInput:
    payload: FitDatasetSpec
    full_params: Dict[str, float]
    parameter_origins: Dict[str, str]
    failed_param_snapshot: Dict[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "full_params", dict(self.full_params))
        object.__setattr__(self, "parameter_origins", dict(self.parameter_origins))
        object.__setattr__(self, "failed_param_snapshot", dict(self.failed_param_snapshot))


class _DatasetFitUnavailable(Exception):
    """Call-local bridge from the direct evaluator to its caller."""

    def __init__(self, result: ScientificUnavailable) -> None:
        self.result = result
        super().__init__(result.stage)


def _is_dataset_deadline_unavailable(
    result: ScientificUnavailable,
) -> bool:
    """Classify the direct evaluator's cooperative deadline result."""

    return bool(dict(result.details).get("deadline")) or result.stage == "dataset_deadline"


def _dataset_deadline_cause(
    dataset_ids: Sequence[str],
) -> FactualCause:
    """Build the fit-owned factual cause for one or more timed-out datasets."""

    roster = tuple(sorted({str(dataset_id) for dataset_id in dataset_ids}))
    cause_details: dict[str, object] = {
        "datasets": roster,
        "deadline": True,
        "deadline_seconds": _DATASET_DEADLINE_SECONDS,
    }
    if len(roster) == 1:
        cause_details["dataset"] = roster[0]
    return FactualCause(
        domain="fit",
        origin="dataset_deadline",
        code=None,
        type="DatasetEvaluationDeadline",
        message="Fit dataset evaluation exceeded its deadline.",
        details=cause_details,
    )


def _aggregate_dataset_deadline_unavailability(
    items: Sequence[ObjectiveDatasetInput],
    evaluations: Sequence[DatasetFitInfo | ScientificUnavailable],
) -> ScientificUnavailable | None:
    """Aggregate timed-out dataset facts in canonical sorted order."""

    timed_out_dataset_ids = tuple(
        sorted(
            {
                item.payload.dataset_id
                for item, evaluation in zip(items, evaluations, strict=True)
                if isinstance(evaluation, ScientificUnavailable)
                and _is_dataset_deadline_unavailable(evaluation)
            }
        )
    )
    if not timed_out_dataset_ids:
        return None
    return ScientificUnavailable(
        producer="fit_dataset_deadline",
        stage="dataset_deadline",
        details={
            "deadline": True,
            "deadline_seconds": _DATASET_DEADLINE_SECONDS,
            "datasets": timed_out_dataset_ids,
        },
        factual_cause=_dataset_deadline_cause(timed_out_dataset_ids),
    )


def _build_dataset_fit_info(
    item: ObjectiveDatasetInput,
    simulation: SimulationResultValue,
    *,
    weights: Mapping[str, float],
    cancellation_control: SimulationCancellationControl | None = None,
) -> DatasetFitInfo:
    """Project one simulation and build direct weighted residual facts.

    The returned ``DatasetFitInfo`` is the natural owner of every per-dataset
    scientific fact consumed by objective and terminal replay.
    """

    _raise_if_fitting_cancel_requested(cancellation_control)
    payload = item.payload
    dataset_id = payload.dataset_id
    observations = _payload_observations(payload)
    x_name = payload.x_name
    x_projection: ParametricProjection | None = None
    sim_time = np.asarray(simulation.t, dtype=float)
    segment_ids = np.asarray(simulation.segment_ids, dtype=np.int64)
    segment_keys = tuple(simulation.segment_keys)
    discontinuity_ids = np.asarray(simulation.discontinuity_ids, dtype=np.int64)
    target_multipliers = _normalized_target_weight_multipliers(
        species_list=payload.target_roster,
        target_weights=dict(payload.target_weights),
    )
    alignment_y: dict[str, np.ndarray] = {}
    if x_name != "t":
        x_observations = _payload_x_observations(payload)
        # Fail closed on unavailable targets before matching.
        for name in (x_name, *payload.species_list):
            fit_result_target_values(simulation, name, dataset_id=dataset_id)
        # Match along the same curve the Fit plot draws: known pre-boundary
        # endpoints included, instantaneous jumps never crossed.
        alignment_time, alignment_x, alignment_y, alignment_regions = alignment_rows(
            simulation,
            SimulationSeriesAxisSelection(x_name),
            tuple((name, SimulationSeriesAxisSelection(name)) for name in payload.species_list),
        )
        try:
            x_projection = align_parametric(
                observed_x=x_observations["x"],
                source_rows=x_observations["source_rows"],
                model_time=alignment_time,
                model_x=alignment_x,
                segment_ids=alignment_regions,
                segment_keys=tuple(str(region) for region in alignment_regions),
                discontinuity_ids=np.zeros_like(alignment_regions),
                observations=payload.observations,
                model_y=alignment_y,
                multipliers=target_multipliers,
                dataset_label=dataset_id,
                x_name=x_name,
                cancellation_control=cancellation_control,
            )
        except CandidateParametricAlignmentError as exc:
            raise _DatasetFitUnavailable(ScientificUnavailable(
                producer="parametric_alignment",
                stage="parametric_x_alignment",
                details={"dataset": dataset_id, **dict(getattr(exc, "details", None) or {})},
                factual_cause=factual_cause_from_exception(
                    exc, domain="fit", origin="parametric_x_alignment"
                ),
            )) from exc
    _raise_if_fitting_cancel_requested(cancellation_control)

    dataset_weight = _objective_dataset_weight_multiplier(weights.get(dataset_id, 1.0))
    raw_blocks: list[np.ndarray] = []
    weighted_blocks: list[np.ndarray] = []
    model_series: dict[str, np.ndarray] = {}
    r_squared: dict[str, FitMetric] = {}
    raw_rms: dict[str, FitMetric] = {}
    for species_name in sorted(payload.species_list):
        _raise_if_fitting_cancel_requested(cancellation_control)
        species_obs = observations[species_name]
        obs_t = np.asarray(species_obs["t"], dtype=float).reshape(-1)
        y_exp = np.asarray(species_obs["y"], dtype=float).reshape(-1)
        model_values = fit_result_target_values(simulation, species_name, dataset_id=dataset_id)
        if x_name == "t":
            y_sim = _select_fit_rows(
                model_y=np.asarray(model_values, dtype=float),
                sim_time=sim_time,
                observation_time=obs_t,
                segment_ids=segment_ids,
                segment_keys=segment_keys,
                discontinuity_ids=discontinuity_ids,
            )
        else:
            assert x_projection is not None
            y_projection = x_projection.subset(species_obs["source_rows"])
            y_sim = project_parametric_y(
                projection=y_projection,
                model_y=alignment_y[species_name],
            )
        y_sim = np.asarray(y_sim, dtype=float).reshape(-1)
        if y_sim.size != y_exp.size or not np.all(np.isfinite(y_sim)):
            raise _DatasetFitUnavailable(ScientificUnavailable(
                producer="global_fit_objective",
                stage="objective_residual_arithmetic",
                details={"dataset": dataset_id, "species": species_name},
            ))
        raw_residual = y_sim - y_exp
        multiplier = dataset_weight * float(target_multipliers.get(species_name, 1.0))
        weighted = multiplier * raw_residual
        if not np.all(np.isfinite(raw_residual)) or not np.all(np.isfinite(weighted)):
            raise _DatasetFitUnavailable(ScientificUnavailable(
                producer="global_fit_objective",
                stage="objective_residual_arithmetic",
                details={"dataset": dataset_id, "species": species_name},
            ))
        raw_blocks.append(raw_residual)
        weighted_blocks.append(weighted)
        model_series[species_name] = y_sim
        r_squared[str(species_name)] = r_squared_metric(y_exp, raw_residual)
        raw_rms[str(species_name)] = root_mean_square(raw_residual)

    if x_projection is not None:
        x_multiplier = dataset_weight * float(target_multipliers[payload.x_name])
        x_raw = np.asarray(x_projection.delta_x, dtype=float).reshape(-1)
        x_weighted = x_multiplier * x_raw
        if not np.all(np.isfinite(x_weighted)) or finite_sum_of_squares(x_weighted) is None:
            raise _DatasetFitUnavailable(ScientificUnavailable(
                producer="global_fit_objective",
                stage="objective_residual_arithmetic",
                details={"dataset": dataset_id, "target": payload.x_name},
            ))
        raw_blocks.append(x_raw)
        weighted_blocks.append(x_weighted)
        raw_rms[str(payload.x_name)] = root_mean_square(x_raw)
    _raise_if_fitting_cancel_requested(cancellation_control)
    residuals = np.concatenate(raw_blocks)
    weighted_residuals = np.concatenate(weighted_blocks)
    residuals.setflags(write=False)
    weighted_residuals.setflags(write=False)
    for array in model_series.values():
        array.setflags(write=False)
    dataset_metrics = fit_error_metrics(weighted_residuals)
    return DatasetFitInfo(
        dataset_id=dataset_id,
        r_squared_by_series=r_squared,
        raw_rms_by_series=raw_rms,
        weighted_residual_sse=dataset_metrics.weighted_sse,
        weighted_residual_mse=dataset_metrics.weighted_mse,
        weighted_rmse=dataset_metrics.weighted_rmse,
        weighted_mae=dataset_metrics.weighted_mae,
        weight=weights.get(dataset_id, 1.0),
        simulation_result=simulation,
        residuals=residuals,
        weighted_residuals=weighted_residuals,
        model_series=model_series,
        x_projection=x_projection,
    )


def evaluate_dataset_simulation(
    fit_evaluators: Mapping[str, object],
    item: ObjectiveDatasetInput,
    *,
    cancellation_control: SimulationCancellationControl | None = None,
    weights: Mapping[str, float] | None = None,
) -> DatasetFitInfo | ScientificUnavailable:
    """Bind, simulate, align, weight, and evaluate one dataset directly.

    A successful call returns its complete ``DatasetFitInfo`` owner.  Candidate
    scientific unavailability is returned directly; timeout is represented by
    ``ScientificUnavailable`` and is settled into a penalty block only by the
    optimizer objective.  Every other exception remains a factual cancellation
    or fatal route.
    """
    ds_id = item.payload.dataset_id
    started = time.monotonic()
    deadline = started + _DATASET_DEADLINE_SECONDS
    worker_cancel = cancellation_control
    task_control = SimulationCancellationControl(
        lambda: (
            worker_cancel is not None
            and worker_cancel.cancel_requested()
        )
        or time.monotonic() >= deadline
    )

    def deadline_unavailable() -> ScientificUnavailable:
        # Recheck whole-worker cancellation immediately before settling a
        # timeout so Stop/Close wins a simultaneous deadline race.
        _raise_if_fitting_cancel_requested(worker_cancel)
        return ScientificUnavailable(
            producer="fit_dataset_deadline",
            stage="dataset_deadline",
            details={
                "dataset": ds_id,
                "deadline": True,
                "deadline_seconds": _DATASET_DEADLINE_SECONDS,
            },
            factual_cause=_dataset_deadline_cause((ds_id,)),
        )

    try:
        _raise_if_fitting_cancel_requested(worker_cancel)
        # The dispatch compiled one evaluator per included dataset.
        evaluator = fit_evaluators[ds_id]
        setter = getattr(evaluator, "set_cancellation_control", None)
        if callable(setter):
            setter(task_control)
        series_result = evaluator.evaluate_series_with_parameter_origins(
            item.full_params,
            origins=item.parameter_origins,
            failed_params=item.failed_param_snapshot,
        )
        _raise_if_fitting_cancel_requested(worker_cancel)
        if time.monotonic() >= deadline:
            return deadline_unavailable()
        if isinstance(series_result, ScientificUnavailable):
            return series_result
        fit_info = _build_dataset_fit_info(
            item,
            series_result,
            weights=weights or {},
            cancellation_control=task_control,
        )
        _raise_if_fitting_cancel_requested(worker_cancel)
        if time.monotonic() >= deadline:
            return deadline_unavailable()
        return fit_info
    except _DatasetFitUnavailable as unavailable:
        _raise_if_fitting_cancel_requested(worker_cancel)
        if time.monotonic() >= deadline:
            return deadline_unavailable()
        return unavailable.result
    except FitSimulationError:
        _raise_if_fitting_cancel_requested(worker_cancel)
        if time.monotonic() >= deadline:
            return deadline_unavailable()
        raise
    except Exception as exc:
        if isinstance(exc, (FittingCancelled, SimulationCancelled)):
            _raise_if_fitting_cancel_requested(worker_cancel)
            if time.monotonic() >= deadline:
                return deadline_unavailable()
            raise FittingCancelled() from exc
        _raise_if_fitting_cancel_requested(worker_cancel)
        if time.monotonic() >= deadline:
            return deadline_unavailable()
        raise


def evaluate_dataset_simulations(
    fit_evaluators: Mapping[str, object],
    items: Sequence[ObjectiveDatasetInput],
    *,
    cancellation_check: Optional[Callable[[], bool]] = None,
    cancellation_control: SimulationCancellationControl | None = None,
    executor: Executor,
    weights: Mapping[str, float] | None = None,
) -> list[DatasetFitInfo | ScientificUnavailable]:
    if not items:
        return []
    ordered = tuple(items)
    _raise_if_fitting_cancelled(cancellation_check)
    _raise_if_fitting_cancel_requested(cancellation_control)
    futures = [
        executor.submit(
            evaluate_dataset_simulation,
            fit_evaluators,
            item,
            cancellation_control=cancellation_control,
            weights=weights,
        )
        for item in ordered
    ]
    results: list[DatasetFitInfo | ScientificUnavailable] = []
    first_exception: BaseException | None = None
    for future in futures:
        try:
            result = future.result()
        except BaseException as exc:
            # Every submitted task is joined before propagating cancellation or
            # a worker exception; the owning executor must never abandon native
            # solver work behind an early return.
            if first_exception is None:
                first_exception = exc
            continue
        results.append(result)
    if first_exception is not None:
        raise first_exception
    return results


class GlobalFitObjective:
    def __init__(
        self,
        *,
        fit_evaluators: Mapping[str, object],
        payloads: List[FitDatasetSpec],
        dataset_params: Dict[str, Dict[str, float]],
        weights: Dict[str, float],
        layout: FitParameterLayout,
        penalty_value: float,
        progress_callback: Optional[Callable[[FitProgress], None]],
        cancellation_check: Optional[Callable[[], bool]],
        cancellation_control: SimulationCancellationControl | None = None,
        executor: Executor,
    ) -> None:
        # One compiled Fit setup supplies the nonempty unique included
        # roster, its evaluators and the layout built from that roster.
        self._fit_evaluators = dict(fit_evaluators)
        payload_roster = tuple(payloads)
        self._payloads = payload_roster
        self._dataset_params = dataset_params
        self._weights = weights
        self._layout = layout
        self._penalty_value = float(penalty_value)
        self._last_error: FitSimulationError | None = None
        self._progress_callback = progress_callback
        self._cancellation_check = cancellation_check
        self._cancellation_control = cancellation_control
        self._executor = executor

        self._objective_calls = 0
        self._best_cost = float("inf")
        self._best_vector: np.ndarray | None = None
        self.best_complete_facts: tuple[DatasetFitInfo, ...] = ()
        self._warned_objective_keys: set[tuple[str, str, str]] = set()

    def evaluate_scientific_residuals(
        self,
        params: np.ndarray,
    ) -> np.ndarray | None:
        """Evaluate one candidate and refuse penalty residuals as evidence.

        The optimizer-facing call necessarily returns finite penalty residuals
        for scientifically unavailable candidates.  Profile likelihood uses
        this call-local route so it can distinguish that settlement from a
        scientific residual vector without introducing a second result store.
        """

        residuals = self(np.asarray(params, dtype=float))
        if self._last_error is not None:
            return None
        values = np.asarray(residuals, dtype=float).reshape(-1)
        return values if values.size and np.all(np.isfinite(values)) else None

    @property
    def best_complete_vector(self) -> np.ndarray | None:
        """Return the sole retained complete candidate vector, if any."""

        if self._best_vector is None:
            return None
        return np.asarray(self._best_vector, dtype=float).copy()

    def retain_complete_candidate(
        self, vector: np.ndarray, facts: tuple[DatasetFitInfo, ...]
    ) -> None:
        """Retain matching scientific facts and coordinates on strict improvement."""
        if tuple(info.dataset_id for info in facts) != tuple(
            payload.dataset_id for payload in self._payloads
        ):
            return
        cost = finite_sum_of_squares(
            np.concatenate(tuple(info.weighted_residuals for info in facts))
        )
        if cost is not None and cost < self._best_cost:
            self._best_cost = float(cost)
            self._best_vector = np.asarray(vector, dtype=float).copy()
            self.best_complete_facts = facts

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
            if name in shared_keys:
                snapshot[name] = float(value)
            else:
                snapshot[f"{ds_id}::{name}"] = float(value)
        return snapshot

    def __call__(self, params: np.ndarray) -> np.ndarray:
        _raise_if_fitting_cancelled(self._cancellation_check)
        self._last_error = None

        def candidate_penalty_residuals() -> np.ndarray:
            blocks: list[np.ndarray] = []
            for payload in self._payloads:
                target_multipliers = _normalized_target_weight_multipliers(
                    species_list=payload.target_roster,
                    target_weights=dict(payload.target_weights),
                )
                dataset_weight = _objective_dataset_weight_multiplier(
                    self._weights.get(payload.dataset_id, 1.0)
                )
                observations = _payload_observations(payload)
                for species_name in sorted(payload.species_list):
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
                if payload.x_name != "t":
                    blocks.append(
                        np.full(
                            len(payload.x_source_rows),
                            float(dataset_weight)
                            * float(target_multipliers[payload.x_name])
                            * self._penalty_value,
                            dtype=float,
                        )
                    )
            return (
                np.concatenate(blocks).astype(float, copy=False)
                if blocks
                else np.asarray([], dtype=float)
            )

        def dataset_penalty_residuals(payload: FitDatasetSpec) -> np.ndarray:
            target_multipliers = _normalized_target_weight_multipliers(
                species_list=payload.target_roster,
                target_weights=dict(payload.target_weights),
            )
            dataset_weight = _objective_dataset_weight_multiplier(
                self._weights.get(payload.dataset_id, 1.0)
            )
            observations = _payload_observations(payload)
            blocks = [
                np.full(
                    np.asarray(observations[name]["y"], dtype=float).size,
                    dataset_weight
                    * float(target_multipliers.get(name, 1.0))
                    * self._penalty_value,
                    dtype=float,
                )
                for name in sorted(payload.species_list)
            ]
            if payload.x_name != "t":
                blocks.append(
                    np.full(
                        len(payload.x_source_rows),
                        dataset_weight
                        * float(target_multipliers[payload.x_name])
                        * self._penalty_value,
                        dtype=float,
                    )
                )
            return np.concatenate(blocks) if blocks else np.asarray([], dtype=float)

        def settle_unavailable_residuals(
            residual_blocks: Sequence[np.ndarray],
            artificial_block_indexes: set[int],
        ) -> tuple[np.ndarray, float] | None:
            """Return one finite objective vector for an unavailable candidate.

            Completed scientific residuals retain their original values.  Only
            artificial penalty blocks may be enlarged, and only after a
            scientifically available complete candidate has established a
            finite ordering reference.
            The factor is derived from the separately stable block SSEs and is
            nudged by representable floating-point steps until the returned
            vector scores strictly above the retained scientifically available
            best.  Blocks stay
            in canonical dataset order so residual-row identity never changes
            between optimizer calls.
            """

            ordered_blocks = tuple(
                np.asarray(block, dtype=float).reshape(-1)
                for block in residual_blocks
            )
            completed = (
                np.concatenate(
                    tuple(
                        block
                        for index, block in enumerate(ordered_blocks)
                        if index not in artificial_block_indexes
                    )
                ).astype(float, copy=False)
                if len(artificial_block_indexes) < len(ordered_blocks)
                else np.asarray([], dtype=float)
            )
            artificial = (
                np.concatenate(
                    tuple(
                        block
                        for index, block in enumerate(ordered_blocks)
                        if index in artificial_block_indexes
                    )
                ).astype(
                    float,
                    copy=False,
                )
                if artificial_block_indexes
                else np.asarray([], dtype=float)
            )
            completed_sse = finite_sum_of_squares(completed)
            artificial_sse = finite_sum_of_squares(artificial)
            if completed_sse is None or artificial_sse is None:
                return None

            residuals = np.concatenate(ordered_blocks)
            cost = finite_sum_of_squares(residuals)
            if cost is None:
                return None

            best_cost = float(self._best_cost)
            if (
                not math.isfinite(best_cost)
                or not artificial_block_indexes
                or artificial_sse <= 0.0
                or cost > best_cost
            ):
                return residuals, float(cost)

            next_cost = math.nextafter(best_cost, math.inf)
            if not math.isfinite(next_cost):
                return residuals, float(cost)

            required_penalty_sse = next_cost - completed_sse
            if not math.isfinite(required_penalty_sse) or required_penalty_sse <= 0.0:
                return residuals, float(cost)
            factor = math.sqrt(required_penalty_sse / artificial_sse)
            if not math.isfinite(factor) or factor <= 1.0:
                factor = 1.0

            maximum_penalty = float(np.max(np.abs(artificial)))
            if maximum_penalty > 0.0:
                safe_factor = np.finfo(float).max / maximum_penalty
                if math.isfinite(safe_factor):
                    factor = min(factor, math.nextafter(safe_factor, 0.0))

            def scaled_settlement(
                scale: float,
            ) -> tuple[np.ndarray, float] | None:
                settled = np.concatenate(
                    tuple(
                        block * float(scale)
                        if index in artificial_block_indexes
                        else block
                        for index, block in enumerate(ordered_blocks)
                    )
                )
                if not np.all(np.isfinite(settled)):
                    return None
                settled_cost = finite_sum_of_squares(settled)
                if settled_cost is None:
                    return None
                return settled, float(settled_cost)

            settled = scaled_settlement(factor)
            if settled is None:
                return residuals, float(cost)
            _, settled_cost = settled

            # Bracket the first finite scale that clears the best cost.  The
            # square-root estimate is based on separate SSEs, while the final
            # comparison is made against the exact returned residual vector;
            # geometric expansion handles rounding plateaus in the combined
            # stable-SSE calculation without touching completed blocks.
            lower_factor = 1.0
            if settled_cost <= best_cost:
                lower_factor = factor
                upper_factor = factor
                for _ in range(1024):
                    expanded = upper_factor * 2.0
                    if not math.isfinite(expanded):
                        expanded = np.finfo(float).max
                    if maximum_penalty > 0.0:
                        safe_factor = np.finfo(float).max / maximum_penalty
                        if math.isfinite(safe_factor):
                            expanded = min(
                                expanded,
                                math.nextafter(safe_factor, 0.0),
                            )
                    if expanded <= upper_factor:
                        break
                    upper_factor = expanded
                    expanded_settlement = scaled_settlement(upper_factor)
                    if expanded_settlement is None:
                        break
                    _, settled_cost = expanded_settlement
                    if settled_cost > best_cost:
                        break
                else:
                    return residuals, float(cost)
                if settled_cost <= best_cost:
                    return residuals, float(cost)
            else:
                upper_factor = factor

            # Binary search over representable factors for the smallest scale
            # that scores strictly above the retained scientifically available
            # candidate.
            for _ in range(128):
                next_lower = math.nextafter(lower_factor, upper_factor)
                if next_lower >= upper_factor:
                    break
                midpoint = lower_factor + (upper_factor - lower_factor) / 2.0
                if midpoint <= lower_factor or midpoint >= upper_factor:
                    break
                midpoint_settlement = scaled_settlement(midpoint)
                if midpoint_settlement is None:
                    upper_factor = midpoint
                    continue
                if midpoint_settlement[1] > best_cost:
                    upper_factor = midpoint
                    _, settled_cost = midpoint_settlement
                else:
                    lower_factor = midpoint

            final_settlement = scaled_settlement(upper_factor)
            if final_settlement is None or final_settlement[1] <= best_cost:
                return residuals, float(cost)
            return final_settlement

        def finalize_candidate(
            *,
            optimizer_vector: np.ndarray,
            cost: float,
            improved: bool,
            best_results: Mapping[str, SimulationResultValue],
            publish_every_call: bool = False,
        ) -> None:
            self._objective_calls += 1
            if self._progress_callback is None:
                return
            progress_vector = np.asarray(optimizer_vector, dtype=float).reshape(-1)
            if (
                progress_vector.size != self._layout.x0.size
                or not np.all(np.isfinite(progress_vector))
            ):
                return
            if not (publish_every_call or improved or self._objective_calls % 10 == 0):
                return
            self._progress_callback(
                FitProgress(
                    objective_calls=self._objective_calls,
                    cost=float(cost),
                    optimizer_vector=progress_vector,
                    improved=bool(improved),
                    transient_simulation_results_by_dataset=best_results,
                )
            )

        def record_candidate_unavailability(
            unavailability: ScientificUnavailable,
        ) -> None:
            context_error = FitSimulationError(
                "Fit candidate scientific result was unavailable.",
                details={
                    **dict(unavailability.details),
                    "stage": unavailability.stage,
                    "producer": unavailability.producer,
                },
            )
            if unavailability.factual_cause is not None:
                context_error.factual_cause = unavailability.factual_cause
            self._last_error = context_error
            key = ("__candidate__", unavailability.producer, unavailability.stage)
            if key not in self._warned_objective_keys:
                self._warned_objective_keys.add(key)
                logger.warning(
                    "Fit candidate scientific result unavailable: %s/%s",
                    unavailability.producer,
                    unavailability.stage,
                )

        def reject_candidate(
            unavailability: ScientificUnavailable,
            *,
            optimizer_vector: np.ndarray,
        ) -> np.ndarray:
            penalty = candidate_penalty_residuals()
            record_candidate_unavailability(unavailability)
            settled_penalty = settle_unavailable_residuals((penalty,), {0})
            if settled_penalty is None:
                raise RuntimeError("Configured Fit candidate penalty overflowed.")
            penalty, penalty_cost = settled_penalty
            finalize_candidate(
                optimizer_vector=optimizer_vector,
                cost=float(penalty_cost),
                improved=False,
                best_results={},
                publish_every_call=True,
            )
            return penalty

        try:
            self._layout.validate_optimizer_vector(params)
            param_dict = self._layout.shared_param_dict_from_vector(params)
        except FitSimulationError as exc:
            return reject_candidate(
                ScientificUnavailable(
                    producer="global_fit_objective",
                    stage="optimizer_vector_domain",
                    details={"error_type": type(exc).__name__},
                ),
                optimizer_vector=np.asarray(params, dtype=float).reshape(-1),
            )
        dataset_inputs: List[ObjectiveDatasetInput] = []

        for payload in self._payloads:
            _raise_if_fitting_cancel_requested(self._cancellation_control)

            ds_id = payload.dataset_id
            try:
                optimizer_dataset_params = (
                    self._layout.dataset_var_params_for_dataset(ds_id, params)
                )
                coupled_dataset_params = self._layout.coupled_params_for_dataset(ds_id, params)
            except FitSimulationError as exc:
                return reject_candidate(
                    ScientificUnavailable(
                        producer="global_fit_objective",
                        stage="dataset_parameter_decode",
                        details={"dataset": ds_id, "error_type": type(exc).__name__},
                    ),
                    optimizer_vector=params,
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
                    payload=payload,
                    full_params=full_params,
                    parameter_origins=parameter_origins,
                    failed_param_snapshot=failed_param_snapshot,
                )
            )

        simulation_evaluations = tuple(
            evaluate_dataset_simulations(
                self._fit_evaluators,
                dataset_inputs,
                cancellation_check=self._cancellation_check,
                cancellation_control=self._cancellation_control,
                executor=self._executor,
                weights=self._weights,
            )
        )

        timeout_unavailability = _aggregate_dataset_deadline_unavailability(
            dataset_inputs,
            simulation_evaluations,
        )
        timed_out_dataset_roster = (
            tuple(timeout_unavailability.details["datasets"])
            if timeout_unavailability is not None
            else ()
        )
        timed_out_dataset_ids = set(timed_out_dataset_roster)
        if timeout_unavailability is not None:
            timeout_error = FitSimulationError(
                "Fit candidate dataset evaluation exceeded its deadline.",
                details={
                    "scientific_unavailable": True,
                    "stage": "dataset_deadline",
                    "datasets": timed_out_dataset_roster,
                },
            )
            timeout_error.factual_cause = timeout_unavailability.factual_cause
            self._last_error = timeout_error

        residual_blocks: list[np.ndarray] = []
        artificial_block_indexes: set[int] = set()
        nondeadline_unavailability_recorded = False
        for item, evaluation in zip(dataset_inputs, simulation_evaluations, strict=True):
            if isinstance(evaluation, ScientificUnavailable):
                if (
                    item.payload.dataset_id not in timed_out_dataset_ids
                    and not nondeadline_unavailability_recorded
                ):
                    record_candidate_unavailability(evaluation)
                    nondeadline_unavailability_recorded = True
                artificial_block_indexes.add(len(residual_blocks))
                residual_blocks.append(
                    dataset_penalty_residuals(item.payload)
                )
                continue
            residual_blocks.append(
                np.asarray(evaluation.weighted_residuals, dtype=float)
            )
            continue

        if artificial_block_indexes:
            settled_unavailable = settle_unavailable_residuals(
                residual_blocks,
                artificial_block_indexes,
            )
            if settled_unavailable is None:
                return reject_candidate(
                    ScientificUnavailable(
                        producer="global_fit_objective",
                        stage="objective_residual_arithmetic",
                        details={"scope": "combined"},
                    ),
                    optimizer_vector=params,
                )
            residuals, cost = settled_unavailable
        else:
            residuals = np.concatenate(tuple(residual_blocks)).astype(
                float, copy=False
            )
            finite_cost = finite_sum_of_squares(residuals)
            if finite_cost is None:
                return reject_candidate(
                    ScientificUnavailable(
                        producer="global_fit_objective",
                        stage="objective_residual_arithmetic",
                        details={"scope": "combined"},
                    ),
                    optimizer_vector=params,
                )
            cost = float(finite_cost)
        candidate_evaluations = {
            evaluation.dataset_id: evaluation.simulation_result
            for evaluation in simulation_evaluations
            if isinstance(evaluation, DatasetFitInfo)
        }
        complete_evaluations = len(candidate_evaluations) == len(dataset_inputs)
        improved = complete_evaluations and cost < self._best_cost
        if improved:
            self.retain_complete_candidate(params, tuple(simulation_evaluations))
        finalize_candidate(
            optimizer_vector=params,
            cost=cost,
            improved=improved,
            best_results=candidate_evaluations if improved else {},
            publish_every_call=nondeadline_unavailability_recorded,
        )
        _raise_if_fitting_cancelled(self._cancellation_check)

        return residuals


def _raise_final_replay_candidate_error(
    exc: BaseException | ScientificUnavailable,
    *,
    stage: str,
    failed_params: Optional[Dict[str, float]] = None,
) -> None:
    """Raise one factual final-replay error.

    Scientific-unavailability remains tagged as such at the replay boundary.
    """

    if isinstance(exc, ScientificUnavailable):
        details = dict(exc.details)
        details.update(
            {
                "scientific_unavailable": True,
                "stage": str(stage),
            }
        )
        terminal_error = FitSimulationError(
            "Final Fit replay rejected the terminal candidate of the optimizer.",
            failed_params=failed_params,
            details=details,
        )
        if exc.factual_cause is not None:
            terminal_error.factual_cause = exc.factual_cause
        raise terminal_error
    factual_cause = getattr(exc, "factual_cause", None)
    if not isinstance(factual_cause, FactualCause):
        exc_details = getattr(exc, "details", None)
        if isinstance(exc_details, Mapping) and str(
            exc_details.get("producer", "")
        ) == "parametric_alignment":
            factual_cause = factual_cause_from_exception(
                exc,
                domain="fit",
                origin="parametric_x_alignment",
            )
            factual_cause = replace(
                factual_cause,
                details={
                    **dict(factual_cause.details),
                    "candidate_parameter_facts": {
                        "failed_params": dict(failed_params or {}),
                    },
                },
            )
    terminal_error = FitSimulationError(
        "Final Fit replay rejected the terminal candidate of the optimizer.",
        failed_params=failed_params,
        details={"fatal": True, "stage": str(stage)},
    )
    if isinstance(factual_cause, FactualCause):
        terminal_error.factual_cause = factual_cause
    raise terminal_error from exc


def _assemble_fit_replay_state(
    *,
    fit_evaluators: Mapping[str, object],
    payloads: List[FitDatasetSpec],
    layout: FitParameterLayout,
    candidate_vector: np.ndarray,
    fixed_dataset_params: Dict[str, Dict[str, float]],
    weights: Dict[str, float],
    cancellation_check: Optional[Callable[[], bool]],
    success: bool,
    message: str,
    optimizer_diagnostic: FactualCause | None = None,
    cancellation_control: SimulationCancellationControl | None = None,
    executor: Executor,
) -> _FitReplayState:
    layout.validate_optimizer_vector(candidate_vector)
    candidate_vector = np.asarray(candidate_vector, dtype=float).reshape(-1).copy()
    candidate_vector.setflags(write=False)
    fitted_params = layout.shared_param_dict_from_vector(candidate_vector)
    combined_dataset_params = {
        payload.dataset_id: {
            **layout.dataset_var_params_for_dataset(payload.dataset_id, candidate_vector),
            **layout.coupled_params_for_dataset(payload.dataset_id, candidate_vector),
        }
        for payload in payloads
    }

    dataset_inputs: List[ObjectiveDatasetInput] = []
    for payload in payloads:
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
            if (
                name in layout.dataset_var_index.get(ds_id, {})
                or name in coupled_names
            ):
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
                payload=payload,
                full_params=full_params,
                parameter_origins=parameter_origins,
                failed_param_snapshot=failed_param_snapshot,
            )
        )

    simulation_evaluations = tuple(
        evaluate_dataset_simulations(
            fit_evaluators,
            dataset_inputs,
            cancellation_check=cancellation_check,
            cancellation_control=cancellation_control,
            executor=executor,
            weights=weights,
        )
    )
    timeout_unavailability = _aggregate_dataset_deadline_unavailability(
        dataset_inputs,
        simulation_evaluations,
    )
    if timeout_unavailability is not None:
        _raise_final_replay_candidate_error(
            timeout_unavailability,
            stage="final_replay_simulation",
        )
    for evaluation in simulation_evaluations:
        if isinstance(evaluation, ScientificUnavailable):
            _raise_final_replay_candidate_error(
                evaluation,
                stage="final_replay_simulation",
            )

    return _assemble_fit_result_facts(
        payloads=payloads,
        layout=layout,
        candidate_vector=candidate_vector,
        facts=tuple(simulation_evaluations),
        cancellation_check=cancellation_check,
        cancellation_control=cancellation_control,
        success=success,
        message=message,
        optimizer_diagnostic=optimizer_diagnostic,
    )


def _assemble_fit_result_facts(
    *,
    payloads: Sequence[FitDatasetSpec],
    layout: FitParameterLayout,
    candidate_vector: np.ndarray,
    facts: tuple[DatasetFitInfo, ...],
    cancellation_check: Optional[Callable[[], bool]],
    cancellation_control: SimulationCancellationControl | None,
    success: bool,
    message: str,
    optimizer_diagnostic: FactualCause | None,
) -> _FitReplayState:
    """Construct terminal metrics and values from complete scientific facts.

    The replay admitted the candidate vector and found every dataset's facts
    available, in roster order.
    """
    fitted_params = layout.shared_param_dict_from_vector(candidate_vector)
    combined_dataset_params = {
        payload.dataset_id: {
            **layout.dataset_var_params_for_dataset(payload.dataset_id, candidate_vector),
            **layout.coupled_params_for_dataset(payload.dataset_id, candidate_vector),
        }
        for payload in payloads
    }
    dataset_info = []
    global_weighted_residual_blocks: List[np.ndarray] = []
    for evaluation in facts:
        _raise_if_fitting_cancelled(cancellation_check)
        _raise_if_fitting_cancel_requested(cancellation_control)

        dataset_info.append(evaluation)
        global_weighted_residual_blocks.append(
            np.asarray(evaluation.weighted_residuals, dtype=float)
        )

    aggregate_metrics = fit_error_metrics(
        np.concatenate(global_weighted_residual_blocks)
    )

    optimizer_converged = bool(success)
    factual_optimizer_diagnostic = optimizer_diagnostic
    if (
        optimizer_converged
        and factual_optimizer_diagnostic is not None
        and not bool(factual_optimizer_diagnostic.details.get("final_report_warning"))
    ):
        factual_optimizer_diagnostic = None
    if factual_optimizer_diagnostic is not None:
        result_status: Literal["ok", "warn"] = "warn"
    else:
        result_status = "ok"

    result_message = fit_result_message(
        base_message=message,
        status=result_status,
        optimizer_converged=optimizer_converged,
        optimizer_diagnostic=factual_optimizer_diagnostic,
    )

    if result_status == "ok":
        logger.info("Global fit complete with typed weighted metrics.")
    elif result_status == "warn":
        logger.warning("Global fit complete with warnings: %s", result_message)
    for info in dataset_info:
        logger.debug(
            "  %s: weighted SSE=%s, weighted RMSE=%s",
            info.dataset_id,
            info.weighted_residual_sse.value,
            info.weighted_rmse.value,
        )

    return _FitReplayState(
        shared_params=fitted_params,
        dataset_params=combined_dataset_params,
        weighted_residual_sse=aggregate_metrics.weighted_sse,
        weighted_residual_mse=aggregate_metrics.weighted_mse,
        dataset_info=tuple(dataset_info),
        message=result_message,
        status=result_status,
        optimizer_converged=optimizer_converged,
        optimizer_diagnostic=factual_optimizer_diagnostic,
        optimizer_vector=candidate_vector,
        weighted_residuals=np.concatenate(global_weighted_residual_blocks),
        profile_likelihood_result=None,
        residual_bootstrap_result=None,
    )
