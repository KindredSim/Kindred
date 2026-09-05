# SPDX-License-Identifier: MIT
from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, Optional, Protocol, Tuple

import numpy as np

from kindred.core.intervention_schedule_compiler import InterventionExecutionContract
from kindred.core.symbolic.jacobian_execution import SymbolicJacobianExecution
from kindred.core.temperature import TemperatureScheduleProtocol
from kindred.core.species import (
    TIME_MAX_S,
    validate_runtime_concentration,
    validate_time_seconds,
)
from kindred.core.time_grid import build_time_grid, normalize_time_grid_policy

DEFAULT_SOLVER_NAME = "BDF"


class ODERhsNoTemp(Protocol):
    def __call__(self, t: float, y: np.ndarray) -> np.ndarray: ...


class ODERhsWithTemp(Protocol):
    def __call__(self, t: float, y: np.ndarray, *, T: float) -> np.ndarray: ...


ODERhs = ODERhsNoTemp | ODERhsWithTemp

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
    """Exact public pre/post state for one intervention boundary.

    This is a presentation-only carrier.  The executor retains raw trigger
    roots locally for action ordering and continuation; only detached public
    endpoint values cross the solver/output boundary.
    """

    time: float
    pre_state: tuple[tuple[str, float], ...]
    post_state: tuple[tuple[str, float], ...]

    def __post_init__(self) -> None:
        time = validate_time_seconds(self.time, field="presentation boundary time")
        pre = _finite_state_pairs(self.pre_state, field="pre_state")
        post = _finite_state_pairs(self.post_state, field="post_state")
        if tuple(name for name, _value in pre) != tuple(
            name for name, _value in post
        ):
            raise ValueError(
                "Presentation boundary pre/post states must align exactly."
            )
        if not any(
            pre_value != post_value
            for (_pre_name, pre_value), (_post_name, post_value) in zip(
                pre,
                post,
                strict=True,
            )
        ):
            raise ValueError(
                "Presentation boundary must contain an exact changed state."
            )
        object.__setattr__(self, "time", time)
        object.__setattr__(self, "pre_state", pre)
        object.__setattr__(self, "post_state", post)


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


@dataclass(frozen=True)
class SimulationRequest:
    rhs: ODERhs
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
            # Validate generated-grid representability at request construction,
            # before any worker or solver invocation.  Explicit t_eval remains
            # an exact output request and does not require realizing N.
            if self.t_eval is None:
                build_time_grid(start, end, normalized_grid)
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
    provenance: Mapping[str, object]
    intervention_execution: InterventionExecutionContract
    presentation_boundaries: tuple[SimulationPresentationBoundary, ...] = ()

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
        if np.any(~np.isfinite(Y)) or np.any(Y < 0.0):
            raise ValueError("Simulation output states must be finite and nonnegative.")
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
