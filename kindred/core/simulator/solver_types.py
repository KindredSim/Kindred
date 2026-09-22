# SPDX-License-Identifier: MIT
from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, Optional, Protocol, Tuple, Literal

import numpy as np

from kindred.core.intervention_schedule_compiler import InterventionExecutionContract
from kindred.core.symbolic.jacobian_execution import SymbolicJacobianExecution
from kindred.core.temperature import TemperatureScheduleProtocol
from kindred.core.species import (
    TIME_MAX_S,
    validate_runtime_concentration,
    validate_time_seconds,
)
from kindred.core.time_grid import normalize_time_grid_policy
from kindred.core.volume import SimulationBoundaryState, VolumeTrajectory

DEFAULT_SOLVER_NAME = "BDF"


class ODERhs(Protocol):
    def __call__(self, t: float, y: np.ndarray, *, T: float) -> np.ndarray: ...


Rhs2 = Callable[[float, np.ndarray], np.ndarray]


@dataclass(frozen=True, slots=True)
class SimulationCancellationControl:
    """Typed, nonblocking cancellation poll owned by one simulation request."""

    poll: Callable[[], bool]

    def __post_init__(self) -> None:
        if not callable(self.poll):
            raise TypeError("Simulation cancellation control requires a callable poll.")

    def cancel_requested(self) -> bool:
        """Return cancellation state without waiting on pause or worker state."""

        return bool(self.poll())


@dataclass(frozen=True, slots=True)
class SimulationScientificUnavailability:
    """Immutable simulation-owned result for candidate-only unavailable science."""

    producer: str
    stage: str
    details: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        producer = str(self.producer or "").strip()
        stage = str(self.stage or "").strip()
        if not producer or not stage:
            raise ValueError("Simulation unavailability producer and stage are required.")
        if not isinstance(self.details, Mapping):
            raise TypeError("Simulation unavailability details must be a mapping.")
        object.__setattr__(self, "producer", producer)
        object.__setattr__(self, "stage", stage)
        object.__setattr__(self, "details", _freeze_solver_value(dict(self.details)))


class _SimulationScientificUnavailabilitySignal(RuntimeError):
    """Private segment unwind normalized by ``solve_ode`` in fitting mode."""

    def __init__(self, outcome: SimulationScientificUnavailability) -> None:
        if not isinstance(outcome, SimulationScientificUnavailability):
            raise TypeError("Simulation unavailability signal requires its typed result.")
        self.outcome = outcome
        super().__init__("scientific simulation unavailable")


def _freeze_solver_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_solver_value(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_solver_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze_solver_value(item) for item in value)
    if isinstance(value, np.ndarray):
        array = np.asarray(value).copy()
        array.setflags(write=False)
        return array
    return value


@dataclass(frozen=True, slots=True)
class SimulationPresentationBoundary:
    """Ordered executed public states at one exact canonical boundary row."""

    time: float
    states: tuple[SimulationBoundaryState, ...]

    def __post_init__(self) -> None:
        time = validate_time_seconds(self.time, field="presentation boundary time")
        states = tuple(self.states)
        if len(states) < 2:
            raise ValueError("Presentation boundary requires an executed transition.")
        names = None
        for state in states:
            pairs = _finite_state_pairs(state.concentrations, field="boundary state")
            current_names = tuple(name for name, _ in pairs)
            if names is not None and current_names != names:
                raise ValueError("Presentation boundary species must align exactly.")
            names = current_names
        object.__setattr__(self, "time", time)
        object.__setattr__(self, "states", states)

    @property
    def pre_state(self) -> tuple[tuple[str, float], ...]:
        return self.states[0].concentrations

    @property
    def post_state(self) -> tuple[tuple[str, float], ...]:
        return self.states[-1].concentrations


def _finite_state_pairs(
    values: Iterable[tuple[str, object]],
    *,
    field: str,
) -> tuple[tuple[str, float], ...]:
    normalized: list[tuple[str, float]] = []
    names: set[str] = set()
    for raw_name, raw_value in values:
        name = str(raw_name).strip()
        if not name:
            raise ValueError(f"{field} species names must be nonblank.")
        if name in names:
            raise ValueError(f"{field} contains duplicate species {name!r}.")
        value = float(raw_value)
        value = validate_runtime_concentration(value, field=f"{field} value")
        names.add(name)
        normalized.append((name, value))
    if not normalized:
        raise ValueError(f"{field} must not be empty.")
    return tuple(normalized)


@dataclass(frozen=True, slots=True)
class SimulationTermination:
    actual_endpoint: float
    cause: Literal["endpoint", "stop"]
    stop_declaration_ordinal: int | None = None


@dataclass(frozen=True)
class SimulationRequest:
    rhs: ODERhs
    reference_temperature_K: float
    t_span: Tuple[float, float]
    y0: np.ndarray
    intervention_execution: InterventionExecutionContract
    symbolic_jacobian: SymbolicJacobianExecution
    solver: str = DEFAULT_SOLVER_NAME
    rtol: float = 1e-6
    atol: float = 1e-12
    max_step: Optional[float] = None
    first_step: Optional[float] = None
    t_eval: Optional[np.ndarray] = None
    grid: Optional[Mapping[str, float | int]] = None

    temperature_schedule: TemperatureScheduleProtocol | None = None
    species_names: Optional[Tuple[str, ...]] = None
    cancellation_control: SimulationCancellationControl | None = None
    candidate_evaluation: bool = False
    stop_observable_value: Callable[[str, float, np.ndarray, float], float] | None = None
    parameter_values_at_temperature: Callable[[float, tuple[str, ...]], Mapping[str, float]] | None = None

    def __post_init__(self) -> None:
        if not isinstance(
            self.intervention_execution, InterventionExecutionContract
        ):
            raise TypeError(
                "intervention_execution must be an InterventionExecutionContract."
            )
        if not isinstance(self.symbolic_jacobian, SymbolicJacobianExecution):
            raise TypeError(
                "symbolic_jacobian must be a SymbolicJacobianExecution."
            )
        if self.cancellation_control is not None and not isinstance(
            self.cancellation_control,
            SimulationCancellationControl,
        ):
            raise TypeError(
                "cancellation_control must be a SimulationCancellationControl."
            )
        object.__setattr__(self, "candidate_evaluation", bool(self.candidate_evaluation))
        start = validate_time_seconds(self.t_span[0], field="t_span start")
        end = validate_time_seconds(self.t_span[1], field="t_span end")
        if end <= start:
            raise ValueError("t_span end must be greater than start.")
        object.__setattr__(self, "t_span", (start, end))
        y0 = np.asarray(self.y0, dtype=float).reshape(-1).copy()
        if y0.size == 0 or np.any(~np.isfinite(y0)) or np.any(y0 < 0.0):
            raise ValueError("y0 must contain finite nonnegative runtime states.")
        y0.setflags(write=False)
        object.__setattr__(self, "y0", y0)
        if self.t_eval is not None:
            t_eval = np.asarray(self.t_eval, dtype=float).copy()
            if (
                t_eval.ndim != 1
                or np.any(~np.isfinite(t_eval))
                or np.any(t_eval < start)
                or np.any(t_eval > end)
                or (t_eval.size > 1 and np.any(np.diff(t_eval) <= 0.0))
            ):
                raise ValueError("t_eval must be strictly increasing within t_span.")
            t_eval.setflags(write=False)
            object.__setattr__(self, "t_eval", t_eval)
        if self.grid is not None:
            if not isinstance(self.grid, Mapping):
                raise TypeError("grid must be a mapping or None.")
            normalized_grid = normalize_time_grid_policy(self.grid)
            object.__setattr__(
                self,
                "grid",
                _freeze_solver_value(normalized_grid),
            )
        if self.species_names is not None:
            object.__setattr__(
                self,
                "species_names",
                tuple(str(name) for name in self.species_names),
            )


@dataclass(frozen=True, kw_only=True)
class FinalSimulationOutput:
    """One fully composed execution with direct identity for every row."""

    segment_ids: np.ndarray
    segment_keys: tuple[str, ...]
    discontinuity_ids: np.ndarray
    t: np.ndarray
    Y: np.ndarray
    temperature_K: np.ndarray
    provenance: Mapping[str, object]
    intervention_execution: InterventionExecutionContract
    presentation_boundaries: tuple[SimulationPresentationBoundary, ...] = ()
    termination: SimulationTermination
    volume: VolumeTrajectory | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.intervention_execution, InterventionExecutionContract):
            raise TypeError(
                "intervention_execution must be an InterventionExecutionContract."
            )
        # The schedule executor has already admitted and newly owned these
        # arrays.  Finalization adopts them directly; this boundary only
        # validates their scientific shape and freezes the owned buffers.
        t = np.asarray(self.t, dtype=float)
        Y = np.asarray(self.Y, dtype=float)
        if (
            t.ndim != 1
            or np.any(~np.isfinite(t))
            or np.any(t < 0.0)
            or np.any(t > TIME_MAX_S)
            or (t.size > 1 and np.any(np.diff(t) <= 0.0))
        ):
            raise ValueError("Simulation output times must be bounded and increasing.")
        if Y.ndim != 2 or Y.shape[1] != t.size:
            raise ValueError("Simulation output state rows must align with time.")
        if self.volume is not None and self.volume.first_index + self.volume.values.size != t.size:
            raise ValueError("Vessel volume must cover a suffix of simulation rows.")
        if np.any(~np.isfinite(Y)) or np.any(Y < 0.0):
            raise ValueError("Simulation output states must be finite and nonnegative.")
        temperature = np.asarray(self.temperature_K, dtype=float)
        if (temperature.shape != t.shape or np.any(~np.isfinite(temperature))
                or np.any(temperature <= 0.0)):
            raise ValueError("Executed temperatures must be positive, finite, and aligned with output rows.")
        temperature.setflags(write=False)
        object.__setattr__(self, "temperature_K", temperature)
        t.setflags(write=False)
        Y.setflags(write=False)
        object.__setattr__(self, "t", t)
        object.__setattr__(self, "Y", Y)
        if not isinstance(self.provenance, Mapping):
            raise TypeError("provenance must be a mapping.")
        boundaries = tuple(self.presentation_boundaries)
        if any(
            not isinstance(boundary, SimulationPresentationBoundary)
            for boundary in boundaries
        ):
            raise TypeError(
                "presentation_boundaries must contain only "
                "SimulationPresentationBoundary values."
            )
        if any(
            current.time <= previous.time
            for previous, current in zip(boundaries, boundaries[1:])
        ):
            raise ValueError(
                "presentation boundaries must be strictly ordered by time."
            )
        if any(not bool(np.any(t == boundary.time)) for boundary in boundaries):
            raise ValueError(
                "presentation boundaries must align with canonical output times."
            )
        object.__setattr__(self, "presentation_boundaries", boundaries)
        if not self.t.size:
            raise ValueError("Final simulation output cannot be empty.")

        def identity_vector(value: object, *, field: str) -> np.ndarray:
            raw = np.asarray(value)
            if raw.ndim != 1 or raw.dtype.kind not in {"i", "u"}:
                raise ValueError(
                    f"{field} must be a one-dimensional integer array."
                )
            if raw.dtype.kind == "u" and raw.size:
                if int(np.max(raw)) > np.iinfo(np.int64).max:
                    raise ValueError(f"{field} exceeds int64 range.")
            values = raw.astype(np.int64, copy=False)
            if np.any(values < 0) or np.any(np.diff(values) < 0):
                raise ValueError(f"{field} must be ordered and nonnegative.")
            values.setflags(write=False)
            return values

        segment_ids = identity_vector(
            self.segment_ids,
            field="result segment identity",
        )
        discontinuity_ids = identity_vector(
            self.discontinuity_ids,
            field="result discontinuity identity",
        )
        segment_keys = tuple(str(value).strip() for value in self.segment_keys)
        if any(not value for value in segment_keys):
            raise ValueError("Result segment keys must be nonblank.")
        expected = int(self.t.size)
        if (
            segment_ids.size != expected
            or len(segment_keys) != expected
            or discontinuity_ids.size != expected
        ):
            raise ValueError("Result row identity must align with output time.")
        object.__setattr__(self, "segment_ids", segment_ids)
        object.__setattr__(self, "segment_keys", segment_keys)
        object.__setattr__(self, "discontinuity_ids", discontinuity_ids)
