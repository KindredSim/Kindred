# SPDX-License-Identifier: MIT
from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Mapping, Optional, Protocol, Tuple, Literal

import numpy as np

from kindred.core.intervention_schedule_compiler import InterventionExecutionContract
from kindred.core.symbolic.jacobian_execution import SymbolicJacobianExecution
from kindred.core.temperature import TemperatureSchedule
from kindred.core.volume import SimulationBoundaryState, VolumeTrajectory

DEFAULT_SOLVER_NAME = "BDF"


class ODERhs(Protocol):
    def __call__(self, t: float, y: np.ndarray, *, T: float) -> np.ndarray: ...


Rhs2 = Callable[[float, np.ndarray], np.ndarray]


@dataclass(frozen=True, slots=True)
class SimulationCancellationControl:
    """Typed, nonblocking cancellation poll owned by one simulation request."""

    poll: Callable[[], bool]

    def cancel_requested(self) -> bool:
        """Return cancellation state without waiting on pause or worker state."""

        return bool(self.poll())


# The one sentence for a simulation that produced nonfinite values, in Run
# and as a Fit candidate's cause.
SIMULATION_NONFINITE_MESSAGE = "Simulation produced nonfinite values."


@dataclass(frozen=True, slots=True)
class SimulationScientificUnavailability:
    """Immutable simulation-owned result for candidate-only unavailable science."""

    producer: str
    stage: str
    message: str
    details: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Producers name themselves and their stage with literals.
        object.__setattr__(self, "details", freeze_retained_value(dict(self.details)))


class _SimulationScientificUnavailabilitySignal(RuntimeError):
    """Private segment unwind normalized by ``solve_ode`` in fitting mode."""

    def __init__(self, outcome: SimulationScientificUnavailability) -> None:
        self.outcome = outcome
        super().__init__("scientific simulation unavailable")


def freeze_retained_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): freeze_retained_value(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(freeze_retained_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(freeze_retained_value(item) for item in value)
    if isinstance(value, np.ndarray):
        array = np.asarray(value).copy()
        array.setflags(write=False)
        return array
    return value


@dataclass(frozen=True, slots=True)
class SimulationPresentationBoundary:
    """Ordered executed public states at one exact canonical boundary row.

    The schedule executor records it at an admitted boundary time when at least
    two distinct projected public states (one per mechanism species) occurred.
    """

    time: float
    states: tuple[SimulationBoundaryState, ...]


@dataclass(frozen=True, slots=True)
class SimulationTermination:
    actual_endpoint: float
    cause: Literal["endpoint", "stop"]


@dataclass(frozen=True)
class SimulationRequest:
    rhs: ODERhs
    reference_temperature_K: float
    t_span: Tuple[float, float]
    y0: np.ndarray
    intervention_execution: InterventionExecutionContract
    symbolic_jacobian: SymbolicJacobianExecution
    # (line number, authored text) of each intervention declaration, by ordinal.
    intervention_lines: tuple[tuple[int, str], ...]
    solver: str = DEFAULT_SOLVER_NAME
    rtol: float = 1e-6
    atol: float = 1e-12
    t_eval: Optional[np.ndarray] = None
    grid: Optional[Mapping[str, float | int]] = None

    temperature_schedule: TemperatureSchedule | None = None
    species_names: Optional[Tuple[str, ...]] = None
    cancellation_control: SimulationCancellationControl | None = None
    candidate_evaluation: bool = False
    # Also return the solver's accepted steps as rows, so the curve between
    # rows is a straight line across one solver step (measured-X Fit).
    sample_solver_steps: bool = False
    stop_observable_value: Callable[[str, float, np.ndarray, float], float] | None = None
    parameter_values_at_temperature: Callable[[float, tuple[str, ...]], Mapping[str, float]] | None = None

    def __post_init__(self) -> None:
        # The span, solver policy, grid and Fit sample times arrive admitted with
        # the execution request.  Fit evaluation supplies candidate initial
        # concentrations here, so their state is checked at this boundary.
        y0 = np.asarray(self.y0, dtype=float).reshape(-1).copy()
        if y0.size == 0 or np.any(~np.isfinite(y0)) or np.any(y0 < 0.0):
            raise ValueError("y0 must contain finite nonnegative runtime states.")
        y0.setflags(write=False)
        object.__setattr__(self, "y0", y0)
        if self.t_eval is not None:
            t_eval = np.asarray(self.t_eval, dtype=float).copy()
            t_eval.setflags(write=False)
            object.__setattr__(self, "t_eval", t_eval)


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
        # Its sole producer, the schedule executor, admits these rows: solver
        # segments refuse nonfinite state, every published state is projected
        # nonnegative, rows are appended in strictly increasing time with one
        # integer segment and discontinuity identity each, temperatures come
        # from the admitted schedule, and the volume trajectory is sampled on
        # these rows.  This carrier only freezes the owned buffers.
        for name in ("t", "Y", "temperature_K", "segment_ids", "discontinuity_ids"):
            getattr(self, name).setflags(write=False)
        object.__setattr__(self, "segment_keys", tuple(self.segment_keys))
        object.__setattr__(self, "presentation_boundaries", tuple(self.presentation_boundaries))
