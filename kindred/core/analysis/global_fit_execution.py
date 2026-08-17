"""
Global fitting execution boundary.

This module owns serial multi-dataset candidate evaluation, final replay,
diagnostics, and result/completion assembly. Public API composition remains in
``global_fitting.fit_global``.
"""

from __future__ import annotations

import logging
import math
import hashlib
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Callable, Dict, List, Literal, Mapping, Optional, Sequence, Tuple

import numpy as np

from kindred.core.analysis.parametric_alignment import (
    CandidateParametricAlignmentError,
    ParametricXProjection,
    align_parametric_x,
    project_parametric_y,
)
from kindred.core.analysis.dataset_parameter_overrides import (
    FIT_SCOPE_CENTRED_FIXED_EFFECTS,
    FitCentredFixedEffectsParamSpec,
    FitCoupledVariableParamSpec,
    FitDatasetParameterOverrides,
)
from kindred.core.analysis.fit_dataset_payload import FitDatasetSpec
from kindred.core.analysis.fit_evaluation_outcome import (
    FitEvaluationOutcome,
    ScientificUnavailable,
)
from kindred.core.analysis.fit_statistics import (
    FitMetric,
    finite_positive_mean_normalize,
    finite_sum_of_squares,
    fit_error_metrics,
    r_squared_metric,
)
from kindred.core.exceptions import (
    FitCandidateDomainError,
    FittingCancelled,
    FitSimulationError,
    SimulationCancelled,
)
from kindred.core.fitting_completion import (
    FitDiagnostic,
    FitTerminalExtension,
    FitPrimaryOutcome,
    GlobalFitCompletion,
)
from kindred.core.fitting_evaluation import (
    FITTING_PARAM_ORIGIN_CONFIGURED_DATASET,
    FITTING_PARAM_ORIGIN_OPTIMIZER_DATASET,
    FITTING_PARAM_ORIGIN_OPTIMIZER_SHARED,
    evaluate_fitting_series,
)
from kindred.core.objective import ObjectiveContext
from kindred.core.result_repository import SimulationResultValue
from kindred.core.simulator.solver_types import SimulationCancellationControl
from kindred.core.simulation_failure import (
    FactualCause,
    factual_cause_from_exception,
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
    "FitCoordinate",
    "CentredFixedEffectsDeviation",
    "CentredFixedEffectsSettlement",
    "FitCandidateCompleteness",
    "FitProgress",
    "DatasetFitInfo",
    "GlobalFitResult",
    "FitParameterLayout",
    "ObjectiveDatasetInput",
    "DatasetSimulationEvaluation",
    "GlobalFitObjective",
    "build_parameter_layout_for_dataset_ids",
    "normalize_weights",
    "evaluate_dataset_simulation",
    "evaluate_dataset_simulations",
    "fit_result_target_values",
    "dataset_evaluation_is_fatal",
]


@dataclass(frozen=True, slots=True)
class FitCoordinate:
    """Collision-proof internal optimizer coordinate identity.

    ``display_name`` is a report/UI serialization only.  Optimizer maps retain
    the typed identity so a shared ``D::p`` cannot alias dataset ``D``
    parameter ``p``.
    """

    scope: Literal[
        "shared",
        "dataset",
        "coupled",
        "centred_overall",
        "centred_deviation",
    ]
    name: str
    dataset_id: str | None = None

    def __post_init__(self) -> None:
        scope = str(self.scope)
        name = str(self.name or "").strip()
        dataset_id = None if self.dataset_id is None else str(self.dataset_id).strip()
        if not name:
            raise ValueError("Fit coordinate name must be nonblank.")
        if scope in {
            "dataset",
            "centred_deviation",
        } and not dataset_id:
            raise ValueError(f"{scope} coordinates require a dataset identity.")
        if scope not in {
            "shared",
            "dataset",
            "coupled",
            "centred_overall",
            "centred_deviation",
        }:
            raise ValueError(f"Unsupported Fit coordinate scope: {scope!r}")
        object.__setattr__(self, "scope", scope)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "dataset_id", dataset_id)

    @property
    def display_name(self) -> str:
        if self.scope in {"dataset", "centred_deviation"}:
            return f"{self.dataset_id}::{self.name}"
        return self.name

    def to_machine_payload(self) -> dict[str, object]:
        """Return the canonical machine-readable coordinate atom."""

        if self.scope == "centred_overall":
            display_name = f"Overall · {self.name}"
        elif self.scope == "centred_deviation":
            display_name = f"Deviation · {self.dataset_id}::{self.name}"
        else:
            display_name = self.display_name
        return {
            "scope": self.scope,
            "name": self.name,
            "dataset_id": self.dataset_id,
            "display_name": display_name,
            "optimizer_display_name": self.display_name,
        }

    def to_machine_key(self) -> str:
        """Return the canonical collision-proof machine mapping key."""

        return f"{self.scope}|{self.dataset_id or ''}|{self.name}"

    def __str__(self) -> str:
        return self.display_name


@dataclass(frozen=True, slots=True)
class CentredFixedEffectsDeviation:
    """One typed dataset deviation in a centred terminal settlement."""

    coordinate: FitCoordinate
    optimizer_value: float
    decoded_value: float

    def __post_init__(self) -> None:
        if self.coordinate.scope != "centred_deviation":
            raise ValueError(
                "Centred settlement deviations require a centred_deviation coordinate."
            )
        optimizer_value = float(self.optimizer_value)
        decoded_value = float(self.decoded_value)
        if not math.isfinite(optimizer_value) or not math.isfinite(decoded_value):
            raise ValueError("Centred settlement deviations must be finite.")
        object.__setattr__(self, "optimizer_value", optimizer_value)
        object.__setattr__(self, "decoded_value", decoded_value)


@dataclass(frozen=True, slots=True)
class CentredFixedEffectsSettlement:
    """Immutable terminal authority for one centred fixed-effects parameter."""

    parameter_name: str
    dataset_ids: tuple[str, ...]
    reference_dataset_id: str
    overall_coordinate: FitCoordinate
    overall_optimizer_value: float
    overall_decoded_value: float
    deviations: tuple[CentredFixedEffectsDeviation, ...]
    dataset_values: Mapping[str, float]

    def __post_init__(self) -> None:
        name = str(self.parameter_name or "").strip()
        roster = tuple(str(value or "").strip() for value in self.dataset_ids)
        if (
            not name
            or not roster
            or any(not value for value in roster)
            or len(roster) != len(set(roster))
            or roster != tuple(sorted(roster))
        ):
            raise ValueError("Centred settlement requires a sorted unique dataset roster.")
        reference = str(self.reference_dataset_id or "").strip()
        if reference != roster[0]:
            raise ValueError("Centred settlement reference must be the first sorted dataset.")
        expected_overall = FitCoordinate("centred_overall", name)
        if self.overall_coordinate != expected_overall:
            raise ValueError("Centred settlement overall identity is not canonical.")
        deviations = tuple(self.deviations)
        if any(not isinstance(item, CentredFixedEffectsDeviation) for item in deviations):
            raise TypeError("Centred settlement deviations require typed rows.")
        expected_deviation_coordinates = tuple(
            FitCoordinate("centred_deviation", name, dataset_id)
            for dataset_id in roster
        )
        if tuple(item.coordinate for item in deviations) != expected_deviation_coordinates:
            raise ValueError(
                "Centred settlement deviations require the exact typed dataset identities, "
                "including the derived reference."
            )
        values = {str(dataset_id): float(value) for dataset_id, value in self.dataset_values.items()}
        if tuple(values) != roster or any(not math.isfinite(value) for value in values.values()):
            raise ValueError("Centred settlement decoded dataset values must match its roster.")
        overall_optimizer = float(self.overall_optimizer_value)
        overall_decoded = float(self.overall_decoded_value)
        if not math.isfinite(overall_optimizer) or not math.isfinite(overall_decoded):
            raise ValueError("Centred settlement overall values must be finite.")
        object.__setattr__(self, "parameter_name", name)
        object.__setattr__(self, "dataset_ids", roster)
        object.__setattr__(self, "reference_dataset_id", reference)
        object.__setattr__(self, "overall_optimizer_value", overall_optimizer)
        object.__setattr__(self, "overall_decoded_value", overall_decoded)
        object.__setattr__(self, "deviations", deviations)
        object.__setattr__(self, "dataset_values", MappingProxyType(values))


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
            f"Non-finite Log10 optimizer coordinate for {coordinate!r}.",
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
class FitCandidateCompleteness:
    """Typed proof that a compact candidate has one valid result per dataset."""

    expected_dataset_ids: tuple[str, ...]
    evaluated_dataset_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        expected = tuple(str(value or "").strip() for value in self.expected_dataset_ids)
        evaluated = tuple(str(value or "").strip() for value in self.evaluated_dataset_ids)
        if (
            not expected
            or any(not value for value in expected)
            or len(expected) != len(set(expected))
            or any(not value for value in evaluated)
            or len(evaluated) != len(set(evaluated))
            or not set(evaluated).issubset(expected)
        ):
            raise ValueError("Fit candidate completeness proof has an invalid roster.")
        object.__setattr__(self, "expected_dataset_ids", expected)
        object.__setattr__(self, "evaluated_dataset_ids", evaluated)

    @property
    def is_complete(self) -> bool:
        return set(self.expected_dataset_ids) == set(self.evaluated_dataset_ids)


@dataclass(frozen=True, slots=True)
class FitProgress:
    """Compact optimizer progress; simulation curves are callback-transient only."""

    objective_calls: int
    cost: float
    optimizer_vector: np.ndarray
    improved: bool = False
    completeness: FitCandidateCompleteness | None = None
    transient_simulation_results_by_dataset: Mapping[str, SimulationResultValue] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "objective_calls", int(self.objective_calls))
        cost = float(self.cost)
        if not math.isfinite(cost):
            raise ValueError("Fit progress cost must be finite.")
        object.__setattr__(self, "cost", cost)
        raw_vector = np.asarray(self.optimizer_vector, dtype=float)
        if raw_vector.ndim != 1:
            raise ValueError("Fit progress optimizer vector must be one-dimensional.")
        vector = raw_vector.copy()
        if vector.size == 0 or not np.all(np.isfinite(vector)):
            raise ValueError(
                "Fit progress requires a nonempty finite optimizer vector."
            )
        vector.setflags(write=False)
        object.__setattr__(self, "optimizer_vector", vector)
        object.__setattr__(self, "improved", bool(self.improved))
        transient = {
            str(dataset_id): value
            for dataset_id, value in self.transient_simulation_results_by_dataset.items()
        }
        if any(
            not isinstance(value, SimulationResultValue)
            for value in transient.values()
        ):
            raise TypeError(
                "Transient Fit progress simulations require SimulationResultValue values."
            )
        object.__setattr__(
            self,
            "transient_simulation_results_by_dataset",
            MappingProxyType(transient),
        )
        if self.completeness is not None and not isinstance(
            self.completeness, FitCandidateCompleteness
        ):
            raise TypeError("Fit progress completeness must be typed evidence.")

    def compact(self) -> "FitProgress":
        """Drop callback-transient curves before retaining best-candidate custody."""

        return replace(self, transient_simulation_results_by_dataset={})


@dataclass(frozen=True, slots=True)
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
    x_point_count: int = 0
    x_weighted_residual_sse: FitMetric | None = None
    x_weighted_residual_mse: FitMetric | None = None
    x_weighted_rmse: FitMetric | None = None
    x_weighted_mae: FitMetric | None = None
    x_delta_x: tuple[float, ...] = ()
    x_match_kinds: tuple[str, ...] = ()
    x_sample_kinds: tuple[str, ...] = ()
    x_source_rows: tuple[int, ...] = ()
    x_identity_hash: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_id", str(self.dataset_id or "").strip())
        object.__setattr__(
            self,
            "r_squared_by_series",
            MappingProxyType(
                {
                    str(name): value
                    for name, value in self.r_squared_by_series.items()
                }
            ),
        )
        object.__setattr__(self, "n_points", int(self.n_points))
        object.__setattr__(self, "weight", float(self.weight))
        object.__setattr__(self, "x_point_count", int(self.x_point_count))
        if self.x_point_count < 0:
            raise ValueError("Dataset X point count cannot be negative.")
        x_metrics = (
            self.x_weighted_residual_sse,
            self.x_weighted_residual_mse,
            self.x_weighted_rmse,
            self.x_weighted_mae,
        )
        if any(item is not None and not isinstance(item, FitMetric) for item in x_metrics):
            raise TypeError("Dataset X metrics require typed FitMetric values.")
        object.__setattr__(self, "x_delta_x", tuple(float(item) for item in self.x_delta_x))
        object.__setattr__(self, "x_match_kinds", tuple(str(item) for item in self.x_match_kinds))
        object.__setattr__(self, "x_sample_kinds", tuple(str(item) for item in self.x_sample_kinds))
        object.__setattr__(self, "x_source_rows", tuple(int(item) for item in self.x_source_rows))
        if self.x_identity_hash is not None:
            identity = str(self.x_identity_hash).strip()
            if not identity:
                raise ValueError("Dataset X identity hash cannot be blank.")
            object.__setattr__(self, "x_identity_hash", identity)


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
    completion: GlobalFitCompletion
    optimizer_vector: np.ndarray | None
    centred_fixed_effects_settlements: tuple[CentredFixedEffectsSettlement, ...] = ()
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
    model_x_projection_by_dataset: Dict[str, ParametricXProjection] = field(
        default_factory=dict
    )
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
        if self.optimizer_vector is None:
            raise ValueError("GlobalFitResult requires its exact optimizer vector.")
        raw_optimizer_vector = np.asarray(self.optimizer_vector, dtype=float)
        if raw_optimizer_vector.ndim != 1:
            raise ValueError("GlobalFitResult optimizer vector must be one-dimensional.")
        optimizer_vector = raw_optimizer_vector.copy()
        if optimizer_vector.size == 0 or not np.all(np.isfinite(optimizer_vector)):
            raise ValueError(
                "GlobalFitResult requires a nonempty finite optimizer vector."
            )
        optimizer_vector.setflags(write=False)
        object.__setattr__(self, "optimizer_vector", optimizer_vector)
        settlements = tuple(self.centred_fixed_effects_settlements or ())
        if any(not isinstance(item, CentredFixedEffectsSettlement) for item in settlements):
            raise TypeError("GlobalFitResult centred settlements require typed DTOs.")
        expected_primary_model = (
            "centred_fixed_effects" if settlements else "ordinary"
        )
        if self.completion.primary_model != expected_primary_model:
            raise ValueError(
                "GlobalFitResult primary model must match its centred settlement roster."
            )
        object.__setattr__(self, "centred_fixed_effects_settlements", settlements)
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
        object.__setattr__(self, "shared_params", MappingProxyType(shared_params))
        object.__setattr__(
            self,
            "dataset_params",
            MappingProxyType(
                {
                    dataset_id: MappingProxyType(values)
                    for dataset_id, values in dataset_params.items()
                }
            ),
        )
        object.__setattr__(self, "dataset_info", tuple(self.dataset_info))
        if self.scipy_nfev is None:
            raise ValueError("GlobalFitResult requires the exact SciPy nfev count.")
        scipy_nfev = int(self.scipy_nfev)
        terminal_replay_count = int(self.terminal_replay_count)
        if terminal_replay_count < 0:
            raise ValueError("GlobalFitResult terminal replay count cannot be negative.")
        object.__setattr__(self, "scipy_nfev", scipy_nfev)
        object.__setattr__(self, "terminal_replay_count", terminal_replay_count)
        if self.objective_call_count is not None:
            objective_call_count = int(self.objective_call_count)
            if objective_call_count < 0:
                raise ValueError("GlobalFitResult objective-call count cannot be negative.")
            object.__setattr__(self, "objective_call_count", objective_call_count)
        def freeze_provenance(value: object) -> object:
            if isinstance(value, Mapping):
                return MappingProxyType(
                    {
                        str(name): freeze_provenance(item)
                        for name, item in value.items()
                    }
                )
            if isinstance(value, (list, tuple, set, frozenset)):
                return tuple(freeze_provenance(item) for item in value)
            if isinstance(value, np.ndarray):
                array = np.asarray(value).copy()
                array.setflags(write=False)
                return array
            return value

        object.__setattr__(
            self,
            "shared_refit_provenance",
            freeze_provenance(dict(self.shared_refit_provenance or {})),
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
            MappingProxyType(simulation_results),
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
        def freeze_array(value: object, *, dtype: object = float) -> np.ndarray:
            array = np.asarray(value, dtype=dtype).reshape(-1).copy()
            array.setflags(write=False)
            return array

        def freeze_nested_mapping(value: Mapping[str, Mapping[str, object]], *, dtype: object = float) -> Mapping[str, Mapping[str, object]]:
            return MappingProxyType(
                {
                    str(dataset_id): MappingProxyType(
                        {
                            str(name): freeze_array(item, dtype=dtype)
                            if isinstance(item, np.ndarray) or dtype is not object
                            else item
                            for name, item in values.items()
                        }
                    )
                    for dataset_id, values in value.items()
                }
            )

        object.__setattr__(self, "model_series", freeze_nested_mapping(self.model_series))
        object.__setattr__(self, "model_x_by_species", freeze_nested_mapping(self.model_x_by_species))
        object.__setattr__(self, "model_segment_ids", freeze_nested_mapping(self.model_segment_ids, dtype=np.int64))
        object.__setattr__(self, "model_segment_keys", freeze_nested_mapping(self.model_segment_keys, dtype=object))
        object.__setattr__(self, "model_discontinuity_ids", freeze_nested_mapping(self.model_discontinuity_ids, dtype=np.int64))
        object.__setattr__(self, "model_selection_provenance", freeze_nested_mapping(self.model_selection_provenance, dtype=object))
        object.__setattr__(self, "model_observation_ids", freeze_nested_mapping(self.model_observation_ids, dtype=object))
        object.__setattr__(self, "model_raw_columns", freeze_nested_mapping(self.model_raw_columns, dtype=object))
        object.__setattr__(self, "model_source_rows", freeze_nested_mapping(self.model_source_rows, dtype=np.int64))
        projections = {
            str(dataset_id): projection
            for dataset_id, projection in self.model_x_projection_by_dataset.items()
        }
        if any(not isinstance(item, ParametricXProjection) for item in projections.values()):
            raise TypeError("Fit X projections require immutable ParametricXProjection values.")
        object.__setattr__(
            self,
            "model_x_projection_by_dataset",
            MappingProxyType(projections),
        )


@dataclass(frozen=True, slots=True)
class _FitReplayState:
    """Private replay evidence held until the sole publishable result build."""

    shared_params: Mapping[str, float]
    dataset_params: Mapping[str, Mapping[str, float]]
    weighted_residual_sse: FitMetric
    weighted_residual_mse: FitMetric
    dataset_info: tuple[DatasetFitInfo, ...]
    message: str
    completion: GlobalFitCompletion
    optimizer_vector: np.ndarray
    scipy_nfev: int
    terminal_replay_count: int
    objective_call_count: int | None
    model_series: Mapping[str, Mapping[str, np.ndarray]]
    model_x_by_species: Mapping[str, Mapping[str, np.ndarray]]
    model_segment_ids: Mapping[str, Mapping[str, np.ndarray]]
    model_segment_keys: Mapping[str, Mapping[str, tuple[str, ...]]]
    model_discontinuity_ids: Mapping[str, Mapping[str, np.ndarray]]
    model_selection_provenance: Mapping[str, Mapping[str, tuple[str, ...]]]
    model_observation_ids: Mapping[str, Mapping[str, tuple[str, ...]]]
    model_raw_columns: Mapping[str, Mapping[str, tuple[str, ...]]]
    model_source_rows: Mapping[str, Mapping[str, tuple[int, ...]]]
    model_x_projection_by_dataset: Mapping[str, ParametricXProjection]
    simulation_results_by_dataset: Mapping[str, SimulationResultValue]
    centred_fixed_effects_settlements: tuple[CentredFixedEffectsSettlement, ...] = ()

    def __post_init__(self) -> None:
        raw_vector = np.asarray(self.optimizer_vector, dtype=float)
        if raw_vector.ndim != 1:
            raise ValueError("Fit replay optimizer vector must be one-dimensional.")
        vector = raw_vector.copy()
        if vector.size == 0 or not np.all(np.isfinite(vector)):
            raise ValueError("Fit replay state requires a nonempty finite optimizer vector.")
        vector.setflags(write=False)
        object.__setattr__(self, "optimizer_vector", vector)


def _construct_global_fit_result(
    state: _FitReplayState,
    *,
    terminal_extensions: Sequence[FitTerminalExtension] = (),
    shared_refit_provenance: Mapping[str, object] | None = None,
    primary_outcome: FitPrimaryOutcome | None = None,
) -> GlobalFitResult:
    """Construct the sole immutable publishable result after settlement."""

    if not isinstance(state, _FitReplayState):
        raise TypeError("Global Fit publication requires private replay state.")
    base = state.completion
    settlements = tuple(state.centred_fixed_effects_settlements or ())
    if settlements:
        primary_model = "centred_fixed_effects"
        primary_model_details: Mapping[str, object] = {
            "scope": FIT_SCOPE_CENTRED_FIXED_EFFECTS,
            "parameter_coordinates": tuple(
                item.overall_coordinate for item in settlements
            ),
            "dataset_rosters": tuple(item.dataset_ids for item in settlements),
            "reference_dataset_ids": tuple(
                item.reference_dataset_id for item in settlements
            ),
        }
    else:
        primary_model = "ordinary"
        primary_model_details = {}
    completion = GlobalFitCompletion(
        status=base.status,
        optimizer_converged=base.optimizer_converged,
        optimizer_diagnostic=base.optimizer_diagnostic,
        terminal_extensions=tuple(terminal_extensions),
        primary_model=primary_model,
        primary_model_details=primary_model_details,
        primary_outcome=primary_outcome,
    )
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
        completion=completion,
        optimizer_vector=state.optimizer_vector,
        centred_fixed_effects_settlements=settlements,
        scipy_nfev=int(state.scipy_nfev),
        terminal_replay_count=int(state.terminal_replay_count),
        objective_call_count=state.objective_call_count,
        shared_refit_provenance=dict(shared_refit_provenance or {}),
        model_series={
            str(dataset_id): dict(values)
            for dataset_id, values in state.model_series.items()
        },
        model_x_by_species={
            str(dataset_id): dict(values)
            for dataset_id, values in state.model_x_by_species.items()
        },
        model_segment_ids={
            str(dataset_id): dict(values)
            for dataset_id, values in state.model_segment_ids.items()
        },
        model_segment_keys={
            str(dataset_id): dict(values)
            for dataset_id, values in state.model_segment_keys.items()
        },
        model_discontinuity_ids={
            str(dataset_id): dict(values)
            for dataset_id, values in state.model_discontinuity_ids.items()
        },
        model_selection_provenance={
            str(dataset_id): dict(values)
            for dataset_id, values in state.model_selection_provenance.items()
        },
        model_observation_ids={
            str(dataset_id): dict(values)
            for dataset_id, values in state.model_observation_ids.items()
        },
        model_raw_columns={
            str(dataset_id): dict(values)
            for dataset_id, values in state.model_raw_columns.items()
        },
        model_source_rows={
            str(dataset_id): dict(values)
            for dataset_id, values in state.model_source_rows.items()
        },
        model_x_projection_by_dataset={
            str(dataset_id): projection
            for dataset_id, projection in state.model_x_projection_by_dataset.items()
        },
        simulation_results_by_dataset=dict(state.simulation_results_by_dataset),
    )


def _compact_failure_message(failure: FactualCause) -> str:
    if not isinstance(failure, FactualCause):
        raise TypeError("Fit diagnostics require a FactualCause.")
    raw_message = failure.message
    return " ".join(part.strip() for part in str(raw_message).splitlines() if part.strip())


def build_fit_diagnostic_from_failure(
    failure: object,
    *,
    phase: str,
    dataset_id: Optional[str] = None,
) -> FitDiagnostic:
    if not isinstance(failure, FactualCause):
        raise TypeError("Fit diagnostic failure must be a FactualCause.")
    return FitDiagnostic(
        phase=phase,
        dataset_id=dataset_id,
        failure=failure,
    )


def build_fit_diagnostic_from_exception(
    exc: BaseException,
    *,
    phase: str,
    dataset_id: Optional[str] = None,
) -> FitDiagnostic:
    failure = factual_cause_from_exception(
        exc,
        domain="fit",
        origin=f"diagnostic_{phase}",
    )
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


def _select_fit_rows(
    *,
    model_y: np.ndarray,
    sim_time: np.ndarray,
    observation_time: np.ndarray,
    observation_id: str,
    raw_column: str,
    source_rows: Sequence[int],
    segment_ids: np.ndarray,
    segment_keys: Sequence[str],
    discontinuity_ids: np.ndarray,
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
    """Align one time-based Y series and preserve its row identity."""

    obs_t = np.asarray(observation_time, dtype=float).reshape(-1)
    rows = tuple(int(value) for value in source_rows)
    raw_id = str(observation_id or "").strip()
    raw_name = str(raw_column or "").strip()
    if (
        not raw_id
        or not raw_name
        or obs_t.size == 0
        or len(rows) != obs_t.size
        or len(rows) != len(set(rows))
        or any(row <= 0 for row in rows)
    ):
        raise FitSimulationError(
            "Fit selector requires complete row-aligned observation identity.",
            details={"fatal": True, "stage": "fit_capture"},
        )
    selected = _align_series(
        np.asarray(model_y, dtype=float),
        np.asarray(sim_time, dtype=float),
        obs_t,
        np.asarray(segment_ids, dtype=np.int64),
        segment_keys,
        np.asarray(discontinuity_ids, dtype=np.int64),
    )
    selected_x = obs_t.copy()
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


def _payload_x_observations(payload: FitDatasetSpec) -> Dict[str, object]:
    """Return one canonical X block plus its physical-row lookup."""

    if payload.x_name == "t" or payload.x_observations is None:
        return {}
    source = payload.x_observations
    rows = payload.x_source_rows
    by_row = {
        int(row): (float(time), float(value))
        for row, time, value in zip(source.source_rows, source.t, source.y, strict=True)
    }
    return {
        "t": np.asarray([by_row[row][0] for row in rows], dtype=float),
        "x": np.asarray([by_row[row][1] for row in rows], dtype=float),
        "source_rows": rows,
        "raw_observation_id": source.raw_observation_id,
        "raw_column": source.raw_column,
    }


def _x_target_multiplier(payload: FitDatasetSpec) -> float:
    if payload.x_name == "t":
        raise ValueError("Time-X datasets do not have an X objective target.")
    multipliers = _normalized_target_weight_multipliers(
        species_list=payload.target_roster,
        target_weights=dict(payload.target_weights),
    )
    return float(multipliers[payload.x_name])


def _x_identity_hash(
    payload: FitDatasetSpec,
    projection: ParametricXProjection | None,
    *,
    dataset_weight: float,
) -> str | None:
    """Hash the canonical X roster and terminal projection exactly once."""

    if projection is None or payload.x_observations is None:
        return None
    digest = hashlib.sha256()
    digest.update(str(payload.dataset_id).encode("utf-8"))
    digest.update(str(payload.x_name).encode("utf-8"))
    digest.update(repr(tuple(payload.x_source_rows)).encode("ascii"))
    digest.update(np.asarray(payload.x_values_for_rows(), dtype=np.float64).tobytes())
    digest.update(repr(float(dataset_weight)).encode("ascii"))
    digest.update(repr(float(_x_target_multiplier(payload))).encode("ascii"))
    digest.update(repr(int(projection.size)).encode("ascii"))
    digest.update(np.asarray(projection.selected_model_x, dtype=np.float64).tobytes())
    digest.update(np.asarray(projection.delta_x, dtype=np.float64).tobytes())
    digest.update(repr(projection.match_kinds).encode("utf-8"))
    digest.update(repr(projection.sample_kinds).encode("utf-8"))
    return digest.hexdigest()


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
    centred_var_order: tuple[FitCentredFixedEffectsParamSpec, ...] = ()
    centred_overall_index: Mapping[str, int] = field(default_factory=dict)
    centred_deviation_index: Mapping[tuple[str, str], int] = field(default_factory=dict)
    scope: str = "ordinary"

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
        centred_order = tuple(self.centred_var_order)
        if any(not isinstance(item, FitCentredFixedEffectsParamSpec) for item in centred_order):
            raise TypeError("Centred layout entries require typed parameter specs.")
        centred_names = tuple(item.parameter_name for item in centred_order)
        if len(centred_names) != len(set(centred_names)):
            raise ValueError("Centred layout parameter names must be unique.")
        object.__setattr__(self, "centred_var_order", centred_order)
        object.__setattr__(
            self,
            "centred_overall_index",
            MappingProxyType(
                {str(name): int(index) for name, index in self.centred_overall_index.items()}
            ),
        )
        object.__setattr__(
            self,
            "centred_deviation_index",
            MappingProxyType(
                {
                    (str(name), str(dataset_id)): int(index)
                    for (name, dataset_id), index in self.centred_deviation_index.items()
                }
            ),
        )
        for name in ("x0", "lower", "upper"):
            values = np.asarray(getattr(self, name), dtype=float).reshape(-1).copy()
            values.setflags(write=False)
            object.__setattr__(self, name, values)
        scope = str(self.scope or "ordinary")
        if scope == "ordinary" and centred_order:
            scope = FIT_SCOPE_CENTRED_FIXED_EFFECTS
        if scope not in {"ordinary", FIT_SCOPE_CENTRED_FIXED_EFFECTS}:
            raise ValueError(f"Unsupported Fit layout scope: {scope!r}")
        if scope == FIT_SCOPE_CENTRED_FIXED_EFFECTS and not centred_order:
            raise ValueError("Centred Fit layout scope requires centred coordinates.")
        object.__setattr__(self, "scope", scope)

    def _coordinate_keys(self) -> tuple[FitCoordinate, ...]:
        keys = tuple(FitCoordinate("shared", name) for name in self.param_names)
        keys += tuple(
            coordinate
            for item in self.centred_var_order
            for coordinate in (
                FitCoordinate("centred_overall", item.parameter_name),
                *tuple(
                    FitCoordinate("centred_deviation", item.parameter_name, dataset_id)
                    for dataset_id in item.dataset_ids[1:]
                ),
            )
        )
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

    def coordinate_identity(self, coordinate: FitCoordinate) -> FitCoordinate:
        keys = self.opt_param_keys()
        if not isinstance(coordinate, FitCoordinate):
            raise TypeError("Fit coordinate identity must be a typed FitCoordinate.")
        if coordinate not in keys:
            raise KeyError(coordinate)
        return coordinate

    def coordinate_index(self, coordinate: FitCoordinate) -> int:
        identity = self.coordinate_identity(coordinate)
        return self.opt_param_keys().index(identity)

    def coordinate_log10(self, coordinate: FitCoordinate) -> bool:
        identity = self.coordinate_identity(coordinate)
        if identity.scope == "shared":
            return bool(self.shared_log10.get(identity.name, False))
        if identity.scope == "dataset":
            return bool(self.dataset_var_log10.get((identity.dataset_id, identity.name), False))
        if identity.scope == "coupled":
            return next(item.log10 for item in self.coupled_var_order if item.variable_id == identity.name)
        if identity.scope == "centred_overall":
            return next(item.log10 for item in self.centred_var_order if item.parameter_name == identity.name)
        if identity.scope == "centred_deviation":
            return next(item.log10 for item in self.centred_var_order if item.parameter_name == identity.name)
        raise KeyError(identity)

    def decode_coordinate(self, coordinate: FitCoordinate, optimizer_value: float) -> float:
        identity = self.coordinate_identity(coordinate)
        value = float(optimizer_value)
        self.coordinate_index(identity)
        if identity.scope == "centred_deviation":
            # Deviations live on the admitted raw scale.  Only a complete
            # derived dataset value is physically decoded for log10 effects.
            if not math.isfinite(value):
                raise FitCandidateDomainError(
                    "Centred deviation coordinate must be finite.",
                    stage="centred_fixed_effects_domain",
                )
            return value
        return (
            _decode_log10_physical(value, coordinate=identity.display_name)
            if self.coordinate_log10(identity)
            else value
        )

    def _centred_raw_values_for_spec(
        self,
        spec: FitCentredFixedEffectsParamSpec,
        vector: np.ndarray,
    ) -> tuple[float, dict[str, float]]:
        overall_index = self.centred_overall_index[spec.parameter_name]
        overall_raw = float(vector[overall_index])
        deltas: dict[str, float] = {}
        for dataset_id in spec.dataset_ids[1:]:
            index = self.centred_deviation_index[(spec.parameter_name, dataset_id)]
            deltas[dataset_id] = float(vector[index])
        reference = spec.dataset_ids[0]
        deltas[reference] = -float(sum(deltas.values()))
        raw_values = {
            dataset_id: overall_raw + deltas[dataset_id]
            for dataset_id in spec.dataset_ids
        }
        return overall_raw, raw_values

    def validate_optimizer_vector(self, vector: np.ndarray) -> None:
        """Validate exact optimizer shape/finiteness and centred-domain custody."""

        values = np.asarray(vector, dtype=float)
        if (
            values.ndim != 1
            or values.size != self.x0.size
            or not np.all(np.isfinite(values))
        ):
            raise FitCandidateDomainError(
                "Fit candidate has a non-finite or incomplete optimizer vector.",
                stage="optimizer_vector_domain",
            )
        for spec in self.centred_var_order:
            _overall_raw, raw_values = self._centred_raw_values_for_spec(spec, values)
            low = math.log10(spec.minimum) if spec.log10 else spec.minimum
            high = math.log10(spec.maximum) if spec.log10 else spec.maximum
            invalid = tuple(
                dataset_id
                for dataset_id, raw_value in raw_values.items()
                if not math.isfinite(raw_value) or raw_value < low or raw_value > high
            )
            if invalid:
                raise FitCandidateDomainError(
                    f"Centred parameter {spec.parameter_name!r} leaves its configured domain "
                    f"for dataset(s): {', '.join(invalid)}.",
                    stage="centred_fixed_effects_domain",
                    details={
                        "parameter": spec.parameter_name,
                        "invalid_dataset_ids": invalid,
                        "bounds": (low, high),
                    },
                )

    def centred_params_for_dataset(
        self,
        dataset_id: str,
        vector: np.ndarray,
    ) -> dict[str, float]:
        self.validate_optimizer_vector(vector)
        dataset_id = str(dataset_id or "").strip()
        values: dict[str, float] = {}
        for spec in self.centred_var_order:
            if dataset_id not in spec.dataset_ids:
                continue
            _overall_raw, raw_values = self._centred_raw_values_for_spec(spec, vector)
            raw_value = raw_values[dataset_id]
            values[spec.parameter_name] = (
                _decode_log10_physical(raw_value, coordinate=spec.parameter_name)
                if spec.log10
                else float(raw_value)
            )
        return values

    def centred_settlements_for_vector(
        self,
        vector: np.ndarray,
    ) -> tuple[CentredFixedEffectsSettlement, ...]:
        self.validate_optimizer_vector(vector)
        values = np.asarray(vector, dtype=float).reshape(-1)
        settlements: list[CentredFixedEffectsSettlement] = []
        for spec in self.centred_var_order:
            overall_raw, raw_values = self._centred_raw_values_for_spec(spec, values)
            overall_decoded = (
                _decode_log10_physical(overall_raw, coordinate=spec.parameter_name)
                if spec.log10
                else float(overall_raw)
            )
            deviations = tuple(
                CentredFixedEffectsDeviation(
                    coordinate=FitCoordinate(
                        "centred_deviation", spec.parameter_name, dataset_id
                    ),
                    optimizer_value=float(raw_values[dataset_id] - overall_raw),
                    decoded_value=float(raw_values[dataset_id] - overall_raw),
                )
                for dataset_id in spec.dataset_ids
            )
            decoded_dataset_values = {
                dataset_id: (
                    _decode_log10_physical(raw_value, coordinate=spec.parameter_name)
                    if spec.log10
                    else float(raw_value)
                )
                for dataset_id, raw_value in raw_values.items()
            }
            settlements.append(
                CentredFixedEffectsSettlement(
                    parameter_name=spec.parameter_name,
                    dataset_ids=tuple(sorted(spec.dataset_ids)),
                    reference_dataset_id=tuple(sorted(spec.dataset_ids))[0],
                    overall_coordinate=FitCoordinate(
                        "centred_overall", spec.parameter_name
                    ),
                    overall_optimizer_value=overall_raw,
                    overall_decoded_value=overall_decoded,
                    deviations=deviations,
                    dataset_values=decoded_dataset_values,
                )
            )
        return tuple(settlements)

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



def build_parameter_layout_for_dataset_ids(
    *,
    dataset_ids: Sequence[str],
    shared_params: Dict[str, float],
    dataset_overrides: Sequence[FitDatasetParameterOverrides],
    coupled_variables: Sequence[FitCoupledVariableParamSpec],
    bounds: Optional[Dict[str, Tuple[float, float]]],
    log10_params: Optional[Dict[str, bool]],
    centred_variables: Sequence[FitCentredFixedEffectsParamSpec] = (),
) -> FitParameterLayout:
    """Build the optimizer layout from the admitted included-dataset roster."""

    normalized_dataset_ids = tuple(sorted(str(value or "").strip() for value in dataset_ids))
    if (
        not normalized_dataset_ids
        or any(not value for value in normalized_dataset_ids)
        or len(normalized_dataset_ids) != len(set(normalized_dataset_ids))
    ):
        raise ValueError("Fit parameter layout requires unique included dataset IDs.")
    param_names = sorted(str(name) for name in shared_params.keys())
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

    raw_centred_order = tuple(centred_variables)
    if any(not isinstance(item, FitCentredFixedEffectsParamSpec) for item in raw_centred_order):
        raise TypeError("Centred Fit variables must use typed parameter specifications.")
    centred_order = tuple(
        sorted(raw_centred_order, key=lambda item: item.parameter_name)
    )
    centred_names = tuple(item.parameter_name for item in centred_order)
    if len(centred_names) != len(set(centred_names)):
        raise ValueError("Centred Fit parameter names must be unique.")
    overlap = set(param_names).intersection(centred_names)
    if overlap:
        raise ValueError(
            "A Fit parameter cannot be both Shared and centred: "
            + ", ".join(sorted(overlap))
        )
    if set(centred_names).intersection(
        name
        for override in dataset_overrides
        for name in (
            *(item.name for item in override.variable_params),
            *(item[0] for item in override.fixed_params),
        )
    ):
        raise ValueError("A centred Fit parameter cannot have a dataset override route.")
    if set(centred_names).intersection(item.parameter_name for item in coupled_variables):
        raise ValueError("A centred Fit parameter cannot have a coupled route.")

    centred_initials: list[float] = []
    centred_bounds: list[tuple[float, float]] = []
    centred_overall_index: dict[str, int] = {}
    centred_deviation_index: dict[tuple[str, str], int] = {}
    for spec in centred_order:
        if tuple(spec.dataset_ids) != normalized_dataset_ids:
            raise ValueError(
                f"Centred parameter {spec.parameter_name!r} must use the exact included dataset roster."
            )
        raw_starts = np.asarray(
            [
                math.log10(value) if spec.log10 else float(value)
                for _dataset_id, value in spec.initial_values
            ],
            dtype=float,
        )
        overall_raw = float(np.mean(raw_starts))
        centred_initials.append(overall_raw)
        low = math.log10(spec.minimum) if spec.log10 else float(spec.minimum)
        high = math.log10(spec.maximum) if spec.log10 else float(spec.maximum)
        centred_bounds.append((low, high))
        overall_index = len(param_names) + len(centred_initials) - 1
        centred_overall_index[spec.parameter_name] = overall_index
        for dataset_id, raw_start in zip(spec.dataset_ids[1:], raw_starts[1:], strict=True):
            delta = float(raw_start - overall_raw)
            centred_initials.append(delta)
            centred_bounds.append((low - high, high - low))
            centred_deviation_index[(spec.parameter_name, dataset_id)] = (
                len(param_names) + len(centred_initials) - 1
            )

    dataset_var_order: List[Tuple[str, str]] = []
    dataset_var_index: Dict[str, Dict[str, int]] = {}
    dataset_var_inits: List[float] = []
    dataset_var_bounds: List[Tuple[float, float]] = []
    dataset_var_log10: Dict[Tuple[str, str], bool] = {}

    overrides_by_id = {item.dataset_id: item for item in dataset_overrides}
    for ds_id in normalized_dataset_ids:
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
            dataset_var_index[ds_id][param_name] = (
                len(param_names) + len(centred_initials) + len(dataset_var_inits)
            )
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

    coupled_order = sorted(coupled_variables, key=lambda item: item.variable_id)
    coupled_ids = [item.variable_id for item in coupled_order]
    if len(coupled_ids) != len(set(coupled_ids)):
        raise ValueError("Coupled Fit variable IDs must be unique.")
    known_dataset_ids = set(normalized_dataset_ids)
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
        shared_initials + centred_initials + dataset_var_inits + coupled_initials,
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
        + [b[0] for b in centred_bounds]
        + [b[0] for b in dataset_var_bounds]
        + [b[0] for b in coupled_bounds],
        dtype=float,
    )
    upper = np.array(
        shared_upper
        + [b[1] for b in centred_bounds]
        + [b[1] for b in dataset_var_bounds]
        + [b[1] for b in coupled_bounds],
        dtype=float,
    )
    coupled_start = len(param_names) + len(centred_initials) + len(dataset_var_inits)
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
        centred_var_order=centred_order,
        centred_overall_index=centred_overall_index,
        centred_deviation_index=centred_deviation_index,
        scope=(FIT_SCOPE_CENTRED_FIXED_EFFECTS if centred_order else "ordinary"),
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
    status: Literal["available", "scientific_unavailable", "cancelled", "fatal"]
    result: Optional[SimulationResultValue] = None
    error: Optional[BaseException] = None
    unavailability: Optional[ScientificUnavailable] = None
    error_provenance: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        status = str(self.status or "")
        if status not in {"available", "scientific_unavailable", "cancelled", "fatal"}:
            raise ValueError(f"Unsupported dataset evaluation status: {status!r}")
        if status == "available" and self.result is None:
            raise ValueError(
                "Successful Fit evaluations require a canonical simulation result."
            )
        if status != "available" and self.result is not None:
            raise ValueError("Failed Fit evaluations cannot publish a partial result.")
        if status == "scientific_unavailable":
            if self.unavailability is None or self.error is not None:
                raise ValueError(
                    "Scientific-unavailability evaluations require their typed result."
                )
        elif self.unavailability is not None:
            raise ValueError("Only unavailable evaluations may carry unavailability.")
        if status == "fatal" and self.error is None:
            raise ValueError("Fatal Fit evaluations require their exception.")
        if status != "fatal" and self.error is not None:
            raise ValueError("Only fatal evaluations may carry an exception.")
        if status == "available" and not isinstance(self.result, SimulationResultValue):
            raise TypeError(
                "Successful Fit evaluations require a SimulationResultValue."
            )
        object.__setattr__(self, "status", status)
        if self.error_provenance is not None:
            object.__setattr__(self, "error_provenance", dict(self.error_provenance))


def dataset_evaluation_is_fatal(result: DatasetSimulationEvaluation) -> bool:
    return result.status == "fatal"


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
        series_outcome = evaluate_fitting_series(
            evaluator,
            item.full_params,
            origins=item.parameter_origins,
            failed_params=item.failed_param_snapshot,
        )
        _raise_if_fitting_cancel_requested(cancellation_control)
        if series_outcome.status == "available":
            if not isinstance(series_outcome.result, SimulationResultValue):
                raise TypeError("Available fitting series outcome omitted its result.")
            return DatasetSimulationEvaluation(
                index=int(item.index),
                status="available",
                result=series_outcome.result,
            )
        if series_outcome.status == "scientific_unavailable":
            return DatasetSimulationEvaluation(
                index=int(item.index),
                status="scientific_unavailable",
                unavailability=series_outcome.unavailability,
                error_provenance={
                    "dataset": ds_id,
                    "provenance": dict(series_outcome.provenance),
                },
            )
        if series_outcome.status == "cancelled":
            return DatasetSimulationEvaluation(
                index=int(item.index),
                status="cancelled",
                error_provenance={"dataset": ds_id},
            )
        return DatasetSimulationEvaluation(
            index=int(item.index),
            status="fatal",
            error=series_outcome.cause,
            error_provenance={
                "dataset": ds_id,
                "provenance": dict(series_outcome.provenance),
            },
        )
    except FitSimulationError as exc:
        return DatasetSimulationEvaluation(
            index=int(item.index),
            status="fatal",
            error=exc,
            error_provenance={"dataset": ds_id, "provenance": getattr(exc, "provenance", None)},
        )
    except Exception as exc:
        if isinstance(exc, (FittingCancelled, SimulationCancelled)):
            raise FittingCancelled() from exc
        return DatasetSimulationEvaluation(
            index=int(item.index),
            status="fatal",
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
        if result.status == "cancelled":
            raise FittingCancelled()
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
        payload_roster = tuple(payloads)
        if any(not isinstance(item, FitDatasetSpec) for item in payload_roster):
            raise TypeError("Fit objective payloads must contain FitDatasetSpec values.")
        dataset_ids = tuple(item.dataset_id for item in payload_roster)
        if (
            not dataset_ids
            or any(not str(dataset_id).strip() for dataset_id in dataset_ids)
            or len(dataset_ids) != len(set(dataset_ids))
        ):
            raise ValueError(
                "Fit objective payloads require a nonempty unique dataset roster."
            )
        included_dataset_ids = set(dataset_ids)
        layout_dataset_ids = {
            dataset_id
            for dataset_id, _parameter_name in layout.dataset_var_order
        }
        if not layout_dataset_ids.issubset(included_dataset_ids):
            raise ValueError(
                "Fit objective layout references datasets absent from its payload roster."
            )
        for spec in layout.centred_var_order:
            if tuple(spec.dataset_ids) != tuple(sorted(dataset_ids)):
                raise ValueError(
                    "Fit objective centred layout must use the exact sorted payload roster."
                )
        self._payloads = tuple(
            sorted(payload_roster, key=lambda item: item.dataset_id)
        )
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
            if name in shared_keys:
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

        def finalize_candidate(
            *,
            optimizer_vector: np.ndarray,
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
            progress_vector = np.asarray(optimizer_vector, dtype=float).reshape(-1)
            if (
                progress_vector.size != self._layout.x0.size
                or not np.all(np.isfinite(progress_vector))
            ):
                return
            if not (publish_every_call or improved or self._objective_calls % 10 == 0):
                return
            expected_ids = tuple(item.dataset_id for item in self._payloads)
            evaluated_ids = tuple(sorted(best_results)) if improved else ()
            self._progress_callback(
                FitProgress(
                    objective_calls=self._objective_calls,
                    cost=float(cost),
                    optimizer_vector=progress_vector,
                    improved=bool(improved),
                    completeness=FitCandidateCompleteness(
                        expected_dataset_ids=expected_ids,
                        evaluated_dataset_ids=evaluated_ids,
                    ),
                    transient_simulation_results_by_dataset=best_results,
                )
            )

        def reject_candidate(
            unavailability: ScientificUnavailable,
            provenance: object,
            *,
            optimizer_vector: np.ndarray,
        ) -> np.ndarray:
            penalty = candidate_penalty_residuals()
            outcome = FitEvaluationOutcome.scientific_unavailable(
                unavailability,
                transport_residuals=penalty,
                provenance=dict(provenance) if isinstance(provenance, Mapping) else {},
            )
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
            self._ctx.set_error(context_error, provenance)
            key = ("__candidate__", unavailability.producer, unavailability.stage)
            if key not in self._warned_objective_keys:
                self._warned_objective_keys.add(key)
                logger.warning(
                    "Fit candidate scientific result unavailable: %s/%s",
                    unavailability.producer,
                    unavailability.stage,
                )
            penalty_cost = finite_sum_of_squares(penalty)
            if penalty_cost is None:
                raise RuntimeError("Configured Fit candidate penalty overflowed.")
            finalize_candidate(
                optimizer_vector=optimizer_vector,
                cost=float(penalty_cost),
                improved=False,
                best_results={},
                publish_every_call=True,
            )
            return _ObjectiveTransport(penalty, outcome)

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
                {"stage": "optimizer_vector_domain"},
                optimizer_vector=np.asarray(params, dtype=float).reshape(-1),
            )
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
                return reject_candidate(
                    ScientificUnavailable(
                        producer="global_fit_objective",
                        stage="dataset_parameter_decode",
                        details={"dataset": ds_id, "error_type": type(exc).__name__},
                    ),
                    {"dataset": ds_id},
                    optimizer_vector=params,
                )
            raw_full_params: Dict[str, float] = {}
            parameter_origins: Dict[str, str] = {}
            for values, origin in (
                (param_dict, FITTING_PARAM_ORIGIN_OPTIMIZER_SHARED),
                (
                    self._layout.centred_params_for_dataset(ds_id, params),
                    FITTING_PARAM_ORIGIN_OPTIMIZER_DATASET,
                ),
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
            if evaluation.status == "scientific_unavailable":
                if evaluation.unavailability is None:
                    raise RuntimeError("Unavailable Fit evaluation omitted its result.")
                return reject_candidate(
                    evaluation.unavailability,
                    evaluation.error_provenance,
                    optimizer_vector=params,
                )

        for item in dataset_inputs:
            _raise_if_fitting_cancel_requested(self._cancellation_control)

            payload = item.payload
            ds_id = payload.dataset_id
            species_list = payload.species_list
            observations = _payload_observations(payload)
            x_name = payload.x_name
            x_observations = _payload_x_observations(payload)
            x_mode = payload.x_mode
            weight = self._weights.get(ds_id, 1.0)
            target_weights = dict(payload.target_weights)
            target_multipliers = _normalized_target_weight_multipliers(
                species_list=payload.target_roster,
                target_weights=target_weights,
            )
            failed_param_snapshot = item.failed_param_snapshot
            evaluation = simulation_evaluations.get(int(item.index))
            if evaluation is None:
                raise RuntimeError(f"Missing fitting simulation result for dataset '{ds_id}'.")

            if evaluation.error is not None:
                exc = evaluation.error
                self._ctx.set_error(exc, evaluation.error_provenance)
                raise exc

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

            x_projection: ParametricXProjection | None = None
            if x_name != "t":
                try:
                    x_projection = align_parametric_x(
                        observation_time=x_observations["t"],
                        observed_x=x_observations["x"],
                        source_rows=x_observations["source_rows"],
                        model_time=np.asarray(sim_time, dtype=float),
                        model_x=fit_result_target_values(simulation, x_name),
                        segment_ids=np.asarray(segment_ids, dtype=np.int64),
                        segment_keys=tuple(segment_keys),
                        discontinuity_ids=np.asarray(discontinuity_ids, dtype=np.int64),
                        mode=x_mode,
                        dataset_label=ds_id,
                        x_name=x_name,
                    )
                except CandidateParametricAlignmentError as exc:
                    details = dict(getattr(exc, "details", None) or {})
                    details.update({"dataset": ds_id, "stage": "parametric_x_alignment"})
                    return reject_candidate(
                        ScientificUnavailable(
                            producer="parametric_alignment",
                            stage="parametric_x_alignment",
                            details=details,
                            factual_cause=factual_cause_from_exception(
                                exc,
                                domain="fit",
                                origin="parametric_x_alignment",
                            ),
                        ),
                        {"dataset": ds_id, "stage": "parametric_x_alignment"},
                        optimizer_vector=params,
                    )
                if x_projection.size != len(payload.x_source_rows):
                    raise RuntimeError("Parametric X projection roster changed during evaluation.")

            for species_name in sorted(species_list):
                _raise_if_fitting_cancelled(self._cancellation_check)
                species_obs = observations[species_name]
                obs_t = np.asarray(species_obs["t"], dtype=float).reshape(-1)
                y_exp = np.asarray(species_obs["y"], dtype=float).reshape(-1)
                target_weight = float(target_multipliers.get(str(species_name), 1.0))
                effective_weight = _objective_dataset_weight_multiplier(weight) * float(target_weight)
                try:
                    model_series = fit_result_target_values(
                        simulation,
                        species_name,
                    )
                    if x_name == "t":
                        selected_rows = _select_fit_rows(
                            model_y=np.asarray(model_series, dtype=float),
                            sim_time=np.asarray(sim_time, dtype=float),
                            observation_time=obs_t,
                            observation_id=species_obs["raw_observation_id"],
                            raw_column=species_obs["raw_column"],
                            source_rows=species_obs["source_rows"],
                            segment_ids=np.asarray(segment_ids, dtype=np.int64),
                            segment_keys=segment_keys,
                            discontinuity_ids=np.asarray(discontinuity_ids, dtype=np.int64),
                        )
                    else:
                        assert x_projection is not None
                        y_projection = x_projection.subset(species_obs["source_rows"])
                        y_sim = project_parametric_y(
                            projection=y_projection,
                            model_time=np.asarray(sim_time, dtype=float),
                            model_y=np.asarray(model_series, dtype=float),
                            segment_ids=np.asarray(segment_ids, dtype=np.int64),
                            segment_keys=tuple(segment_keys),
                            discontinuity_ids=np.asarray(discontinuity_ids, dtype=np.int64),
                        )
                        selected_rows = (
                            y_projection.selected_model_x,
                            y_sim,
                            y_projection.segment_ids,
                            y_projection.segment_keys,
                            y_projection.discontinuity_ids,
                            y_projection.sample_kinds,
                            tuple(species_obs["raw_observation_id"] for _ in species_obs["source_rows"]),
                            tuple(species_obs["raw_column"] for _ in species_obs["source_rows"]),
                            tuple(int(row) for row in species_obs["source_rows"]),
                        )
                    if x_name == "t":
                        y_sim = selected_rows[1]
                except CandidateParametricAlignmentError as exc:
                    # Parametric-X is a candidate result predicate, but its
                    # settlement still belongs to this optimizer boundary:
                    # only exact optimizer-origin names may receive penalty.
                    details = dict(getattr(exc, "details", None) or {})
                    optimizer_names = frozenset(
                        str(name).strip()
                        for name, origin in parameter_origins.items()
                        if str(name).strip()
                        and origin
                        in {
                            FITTING_PARAM_ORIGIN_OPTIMIZER_SHARED,
                            FITTING_PARAM_ORIGIN_OPTIMIZER_DATASET,
                        }
                    )
                    dependencies = frozenset(
                        str(name).strip()
                        for name in (
                            details.get("value_dependencies", ())
                            or details.get("parameter_names", ())
                            or ()
                        )
                        if str(name).strip()
                    )
                    candidate_dependencies = dependencies.intersection(optimizer_names)
                    candidate_facts = {
                        "values": dict(full_params),
                        "origins": dict(parameter_origins),
                        "failed_params": dict(failed_param_snapshot),
                    }
                    details.update(
                        {
                            "producer": "parametric_alignment",
                            "stage": "parametric_x_alignment",
                            "dataset": ds_id,
                            "candidate_parameter_facts": candidate_facts,
                        }
                    )
                    if dependencies:
                        details.update(
                            {
                                "value_dependencies": tuple(sorted(dependencies)),
                                "optimizer_dependencies": tuple(
                                    sorted(candidate_dependencies)
                                ),
                            }
                        )
                    factual_cause = factual_cause_from_exception(
                        exc,
                        domain="fit",
                        origin="parametric_x_alignment",
                    )
                    factual_cause = replace(
                        factual_cause,
                        details={**dict(factual_cause.details), **details},
                    )
                    return reject_candidate(
                        ScientificUnavailable(
                            producer="parametric_alignment",
                            stage="parametric_x_alignment",
                            details=details,
                            factual_cause=factual_cause,
                        ),
                        {"dataset": ds_id, "stage": "parametric_x_alignment"},
                        optimizer_vector=params,
                    )
                except FitSimulationError:
                    raise
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
                    return reject_candidate(
                        ScientificUnavailable(
                            producer="global_fit_objective",
                            stage="objective_nonfinite",
                            details={"dataset": ds_id, "species": species_name},
                        ),
                        {"dataset": ds_id},
                        optimizer_vector=params,
                    )

                with np.errstate(over="ignore", invalid="ignore"):
                    residual_vector = effective_weight * (
                        np.asarray(y_sim, dtype=float).reshape(-1)
                        - np.asarray(y_exp, dtype=float).reshape(-1)
                    )
                if not np.all(np.isfinite(residual_vector)):
                    return reject_candidate(
                        ScientificUnavailable(
                            producer="global_fit_objective",
                            stage="objective_residual_arithmetic",
                            details={"dataset": ds_id, "species": species_name},
                        ),
                        {"dataset": ds_id},
                        optimizer_vector=params,
                    )
                if finite_sum_of_squares(residual_vector) is None:
                    return reject_candidate(
                        ScientificUnavailable(
                            producer="global_fit_objective",
                            stage="objective_residual_arithmetic",
                            details={"dataset": ds_id, "species": species_name},
                        ),
                        {"dataset": ds_id},
                        optimizer_vector=params,
                    )
                all_residuals.extend(np.asarray(residual_vector, dtype=float).reshape(-1))

            if x_projection is not None:
                x_weight = (
                    _objective_dataset_weight_multiplier(weight)
                    * float(target_multipliers[payload.x_name])
                )
                with np.errstate(over="ignore", invalid="ignore"):
                    x_residual = x_weight * np.asarray(x_projection.delta_x, dtype=float)
                if (
                    not np.all(np.isfinite(x_residual))
                    or finite_sum_of_squares(x_residual) is None
                ):
                    return reject_candidate(
                        ScientificUnavailable(
                            producer="global_fit_objective",
                            stage="objective_residual_arithmetic",
                            details={
                                "dataset": ds_id,
                                "target": payload.x_name,
                                "point_count": int(x_projection.size),
                            },
                        ),
                        {"dataset": ds_id, "target": payload.x_name},
                        optimizer_vector=params,
                    )
                all_residuals.extend(np.asarray(x_residual, dtype=float).reshape(-1))

        residuals = np.asarray(all_residuals, dtype=float).ravel()

        finite_cost = finite_sum_of_squares(residuals)
        if finite_cost is None:
            return reject_candidate(
                ScientificUnavailable(
                    producer="global_fit_objective",
                    stage="objective_residual_arithmetic",
                    details={"scope": "combined"},
                ),
                {"stage": "objective"},
                optimizer_vector=params,
            )
        cost = float(finite_cost)
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
            optimizer_vector=params,
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
    exc: BaseException | ScientificUnavailable,
    *,
    stage: str,
    failed_params: Optional[Dict[str, float]] = None,
) -> None:
    """Raise one factual final-replay error.

    Scientific-unavailability remains tagged as such; the composition
    boundary settles it into a typed FitEvaluationOutcome.
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
            "Final Fit replay rejected the optimizer's terminal candidate.",
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
        "Final Fit replay rejected the optimizer's terminal candidate.",
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
    scipy_nfev: int,
    terminal_replay_count: int = 1,
    objective_call_count: int | None = None,
    optimizer_diagnostic: Optional[FitDiagnostic] = None,
    cancellation_control: SimulationCancellationControl | None = None,
) -> _FitReplayState:
    layout.validate_optimizer_vector(candidate_vector)
    candidate_vector = np.asarray(candidate_vector, dtype=float).reshape(-1).copy()
    candidate_vector.setflags(write=False)
    fitted_params = layout.shared_param_dict_from_vector(candidate_vector)
    combined_dataset_params = {
        payload.dataset_id: {
            **layout.centred_params_for_dataset(payload.dataset_id, candidate_vector),
            **layout.dataset_var_params_for_dataset(payload.dataset_id, candidate_vector),
            **layout.coupled_params_for_dataset(payload.dataset_id, candidate_vector),
        }
        for payload in payloads
    }
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
    model_x_projection_map: Dict[str, ParametricXProjection] = {}
    simulation_results_by_dataset: Dict[str, SimulationResultValue] = {}

    dataset_inputs: List[ObjectiveDatasetInput] = []
    for index, payload in enumerate(payloads):
        ds_id = payload.dataset_id
        _raise_if_fitting_cancelled(cancellation_check)
        _raise_if_fitting_cancel_requested(cancellation_control)

        full_params = dict(fitted_params)
        full_params.update(layout.centred_params_for_dataset(ds_id, candidate_vector))
        full_params.update(fixed_dataset_params.get(ds_id, {}))
        full_params.update(combined_dataset_params.get(ds_id, {}))
        parameter_origins = {
            name: FITTING_PARAM_ORIGIN_OPTIMIZER_SHARED
            for name in fitted_params
        }
        for name in layout.centred_params_for_dataset(ds_id, candidate_vector).keys():
            parameter_origins[name] = FITTING_PARAM_ORIGIN_OPTIMIZER_DATASET
        for name in fixed_dataset_params.get(ds_id, {}):
            parameter_origins[name] = FITTING_PARAM_ORIGIN_CONFIGURED_DATASET
        for name in combined_dataset_params.get(ds_id, {}):
            coupled_names = {
                item.parameter_name
                for item in layout.coupled_var_order
                if ds_id in item.dataset_ids
            }
            centred_names = {
                item.parameter_name
                for item in layout.centred_var_order
                if ds_id in item.dataset_ids
            }
            if (
                name in layout.dataset_var_index.get(ds_id, {})
                or name in coupled_names
                or name in centred_names
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
        if evaluation.status == "scientific_unavailable":
            if evaluation.unavailability is None:
                raise RuntimeError("Unavailable final replay omitted its result.")
            _raise_final_replay_candidate_error(
                evaluation.unavailability,
                stage="final_replay_simulation",
            )
        if evaluation.error is not None:
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
        x_observations = _payload_x_observations(payload)
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

        x_projection: ParametricXProjection | None = None
        if x_name != "t":
            try:
                x_projection = align_parametric_x(
                    observation_time=x_observations["t"],
                    observed_x=x_observations["x"],
                    source_rows=x_observations["source_rows"],
                    model_time=np.asarray(sim_time, dtype=float),
                    model_x=fit_result_target_values(simulation, x_name),
                    segment_ids=np.asarray(segment_ids, dtype=np.int64),
                    segment_keys=tuple(segment_keys),
                    discontinuity_ids=np.asarray(discontinuity_ids, dtype=np.int64),
                    mode=x_mode,
                    dataset_label=ds_id,
                    x_name=x_name,
                )
            except CandidateParametricAlignmentError as exc:
                raise FitSimulationError(
                    "Final Fit replay could not settle parametric-X alignment.",
                    failed_params=failed_param_snapshot,
                    details={
                        "stage": "final_replay_alignment",
                        "scientific_unavailable": True,
                        **dict(getattr(exc, "details", None) or {}),
                    },
                ) from exc
            model_x_projection_map[ds_id] = x_projection

        residual_blocks: List[np.ndarray] = []
        weighted_residual_blocks: List[np.ndarray] = []
        x_residual_blocks: List[np.ndarray] = []
        weighted_x_residual_blocks: List[np.ndarray] = []
        r_squared_by_series: Dict[str, FitMetric] = {}
        for species_name in species_list:
            species_obs = observations[species_name]
            obs_t = np.asarray(species_obs["t"], dtype=float).reshape(-1)
            y_exp = np.asarray(species_obs["y"], dtype=float).reshape(-1)
            model_series = fit_result_target_values(
                simulation,
                species_name,
            )
            try:
                if x_name == "t":
                    selected_rows = _select_fit_rows(
                        model_y=np.asarray(model_series, dtype=float),
                        sim_time=np.asarray(sim_time, dtype=float),
                        observation_time=obs_t,
                        observation_id=species_obs["raw_observation_id"],
                        raw_column=species_obs["raw_column"],
                        source_rows=species_obs["source_rows"],
                        segment_ids=np.asarray(segment_ids, dtype=np.int64),
                        segment_keys=tuple(segment_keys),
                        discontinuity_ids=np.asarray(discontinuity_ids, dtype=np.int64),
                    )
                else:
                    assert x_projection is not None
                    y_projection = x_projection.subset(species_obs["source_rows"])
                    y_sim = project_parametric_y(
                        projection=y_projection,
                        model_time=np.asarray(sim_time, dtype=float),
                        model_y=np.asarray(model_series, dtype=float),
                        segment_ids=np.asarray(segment_ids, dtype=np.int64),
                        segment_keys=tuple(segment_keys),
                        discontinuity_ids=np.asarray(discontinuity_ids, dtype=np.int64),
                    )
                    selected_rows = (
                        y_projection.selected_model_x,
                        y_sim,
                        y_projection.segment_ids,
                        y_projection.segment_keys,
                        y_projection.discontinuity_ids,
                        y_projection.sample_kinds,
                        tuple(species_obs["raw_observation_id"] for _ in species_obs["source_rows"]),
                        tuple(species_obs["raw_column"] for _ in species_obs["source_rows"]),
                        tuple(int(row) for row in species_obs["source_rows"]),
                    )
            except FitSimulationError as exc:
                _raise_final_replay_candidate_error(
                    exc,
                    stage="final_replay_alignment",
                    failed_params=failed_param_snapshot,
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

            if not np.all(np.isfinite(y_sim_resid)):
                cause = ScientificUnavailable(
                    producer="global_fit_replay",
                    stage="parametric_x_alignment",
                    details={"dataset": ds_id, "species": species_name},
                )
                _raise_final_replay_candidate_error(
                    cause,
                    stage="final_replay_alignment",
                    failed_params=failed_param_snapshot,
                )
            with np.errstate(over="ignore", invalid="ignore"):
                raw_residual = np.asarray(y_sim_resid, dtype=float) - y_exp
            if not np.all(np.isfinite(raw_residual)):
                cause = ScientificUnavailable(
                    producer="global_fit_replay",
                    stage="objective_residual_arithmetic",
                    details={"dataset": ds_id, "species": species_name},
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
                cause = ScientificUnavailable(
                    producer="global_fit_replay",
                    stage="objective_residual_arithmetic",
                    details={"dataset": ds_id, "species": species_name},
                )
                _raise_final_replay_candidate_error(
                    cause,
                    stage="final_replay_metrics",
                    failed_params=failed_param_snapshot,
                )
            weighted_residual_blocks.append(weighted_residual)

        if x_projection is not None:
            x_weight = (
                _objective_dataset_weight_multiplier(weights.get(ds_id, 1.0))
                * float(
                    _normalized_target_weight_multipliers(
                        species_list=payload.target_roster,
                        target_weights=dict(payload.target_weights),
                    )[payload.x_name]
                )
            )
            with np.errstate(over="ignore", invalid="ignore"):
                raw_x_residual = np.asarray(x_projection.delta_x, dtype=float)
                weighted_x_residual = x_weight * raw_x_residual
            if (
                not np.all(np.isfinite(weighted_x_residual))
                or finite_sum_of_squares(weighted_x_residual) is None
            ):
                raise FitSimulationError(
                    "Final Fit replay produced non-finite X residuals.",
                    failed_params=failed_param_snapshot,
                    details={"stage": "final_replay_metrics", "target": payload.x_name},
                )
            x_residual_blocks.append(raw_x_residual)
            weighted_x_residual_blocks.append(weighted_x_residual)

        residuals = np.concatenate(residual_blocks) if residual_blocks else np.asarray([], dtype=float)
        if x_residual_blocks:
            residuals = np.concatenate([residuals, *x_residual_blocks])
        weighted_residuals = (
            np.concatenate(weighted_residual_blocks)
            if weighted_residual_blocks
            else np.asarray([], dtype=float)
        )
        if weighted_x_residual_blocks:
            weighted_residuals = np.concatenate(
                [weighted_residuals, *weighted_x_residual_blocks]
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
                x_point_count=0 if x_projection is None else int(x_projection.size),
                x_weighted_residual_sse=(
                    None
                    if not weighted_x_residual_blocks
                    else fit_error_metrics(
                        np.concatenate(weighted_x_residual_blocks)
                    ).weighted_sse
                ),
                x_weighted_residual_mse=(
                    None
                    if not weighted_x_residual_blocks
                    else fit_error_metrics(
                        np.concatenate(weighted_x_residual_blocks)
                    ).weighted_mse
                ),
                x_weighted_rmse=(
                    None
                    if not weighted_x_residual_blocks
                    else fit_error_metrics(
                        np.concatenate(weighted_x_residual_blocks)
                    ).weighted_rmse
                ),
                x_weighted_mae=(
                    None
                    if not weighted_x_residual_blocks
                    else fit_error_metrics(
                        np.concatenate(weighted_x_residual_blocks)
                    ).weighted_mae
                ),
                x_delta_x=(
                    () if x_projection is None else tuple(float(item) for item in x_projection.delta_x)
                ),
                x_match_kinds=(
                    () if x_projection is None else x_projection.match_kinds
                ),
                x_sample_kinds=(
                    () if x_projection is None else x_projection.sample_kinds
                ),
                x_source_rows=(
                    () if x_projection is None else x_projection.source_rows
                ),
                x_identity_hash=_x_identity_hash(
                    payload,
                    x_projection,
                    dataset_weight=weights.get(ds_id, 1.0),
                ),
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
                completion_optimizer_diagnostic.failure.details
                if isinstance(completion_optimizer_diagnostic.failure, FactualCause)
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

    centred_settlements = (
        layout.centred_settlements_for_vector(candidate_vector)
        if layout.centred_var_order
        else ()
    )
    return _FitReplayState(
        shared_params=fitted_params,
        dataset_params=combined_dataset_params,
        weighted_residual_sse=aggregate_metrics.weighted_sse,
        weighted_residual_mse=aggregate_metrics.weighted_mse,
        dataset_info=tuple(dataset_info),
        message=completion_message,
        optimizer_vector=candidate_vector,
        scipy_nfev=scipy_nfev,
        terminal_replay_count=terminal_replay_count,
        objective_call_count=objective_call_count,
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
        model_x_projection_by_dataset=model_x_projection_map,
        simulation_results_by_dataset=simulation_results_by_dataset,
        centred_fixed_effects_settlements=centred_settlements,
    )
