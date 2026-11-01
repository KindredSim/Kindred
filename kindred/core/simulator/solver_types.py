# SPDX-License-Identifier: MIT
from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Mapping, Optional, Protocol, Tuple, Literal

import numpy as np

from kindred.core.intervention_schedule_compiler import InterventionExecutionContract
from kindred.core.symbolic.jacobian_execution import SymbolicJacobianExecution
from kindred.core.volume import SimulationBoundaryState, VolumeTrajectory



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
    parameter_values_at_temperature: Callable[[float, tuple[str, ...]], Mapping[str, float]]
    solver: str
    rtol: float
    atol: float
    # Fit's admitted sample times; None realizes ``grid`` instead.
    t_eval: Optional[np.ndarray]
    grid: Mapping[str, float | int]
    species_names: Tuple[str, ...]
    cancellation_control: SimulationCancellationControl
    candidate_evaluation: bool
    # Also return the solver's accepted steps as rows, so the curve between
    # rows is a straight line across one solver step (measured-X Fit).
    sample_solver_steps: bool
    # Also return each species' CTC over the executed span, integrated on the
    # solver's own step interpolant (Run and Preview without interventions).
    species_ctc: bool
    # Also return each species' largest concentration over the executed span
    # and its time, located on the solver's own step interpolant.
    species_maximum: bool
    stop_observable_value: Callable[[str, float, np.ndarray, float], float] | None

    def __post_init__(self) -> None:
        # Starting states arrive admitted nonnegative and finite, Fit
        # candidates within their nonnegative bounds; the run keeps its own copy.
        y0 = np.array(self.y0, dtype=float)
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
    discontinuity_ids: np.ndarray
    t: np.ndarray
    Y: np.ndarray
    temperature_K: np.ndarray
    provenance: Mapping[str, object]
    intervention_execution: InterventionExecutionContract
    presentation_boundaries: tuple[SimulationPresentationBoundary, ...] = ()
    termination: SimulationTermination
    volume: VolumeTrajectory | None = None
    # Per species row, the integral of concentration minus its final state over
    # the executed span; present only when requested.
    species_ctc: np.ndarray | None = None
    # Per species row, the largest concentration over the executed span and its
    # time; present only when requested.
    species_maximum: tuple[np.ndarray, np.ndarray] | None = None

    def __post_init__(self) -> None:
        # Its sole producer, the schedule executor, admits these rows: solver
        # segments refuse nonfinite state, every published state is projected
        # nonnegative, rows are appended in strictly increasing time with one
        # integer segment and discontinuity identity each, temperatures are
        # those each row executed at, and the volume trajectory is sampled on
        # these rows.  This carrier only freezes the owned buffers.
        for name in ("t", "Y", "temperature_K", "segment_ids", "discontinuity_ids"):
            getattr(self, name).setflags(write=False)
        object.__setattr__(self, "presentation_boundaries", tuple(self.presentation_boundaries))
