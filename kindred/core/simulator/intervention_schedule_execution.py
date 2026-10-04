# SPDX-License-Identifier: MIT
from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, replace
from operator import attrgetter
from typing import Callable, Mapping, Protocol, Sequence

import numpy as np

from kindred.core.exceptions import SimulationCancelled
from kindred.core.intervention_schedule import (InterventionScheduleError, InterventionStopAction, SpeciesTriggerCondition, ObservableTriggerCondition, intervention_species)
from kindred.core.intervention_schedule_compiler import (
    ExecutableFlux,
    ExecutableHold,
    ExecutableInstant,
    ExecutableTrigger,
    InterventionExecutionContract,
    TriggerFiring,
    validate_boundary_conflicts,
    resolve_executable_values,
)
from kindred.core.temperature import declared_temperature_step_boundaries, temperature_at_execution_time
from .volume_execution import VolumeExecution
from kindred.core.volume import SimulationBoundaryState
from kindred.core.species import TIME_MAX_S
from .nonnegative_invariant import (
    NonnegativeProjection,
    project_nonnegative_trajectory,
)
from .solver_types import (
    FinalSimulationOutput,
    ODERhs,
    Rhs2,
    SimulationPresentationBoundary,
    SimulationRequest,
    SimulationTermination,
)


@dataclass(frozen=True, slots=True)
class SegmentIdentity:
    kind: str
    ordinal: int
    t_start: float
    t_end: float

    def __post_init__(self) -> None:
        kind = str(self.kind).strip()
        ordinal = int(self.ordinal)
        t_start = float(self.t_start)
        t_end = float(self.t_end)
        if not kind:
            raise ValueError("Segment identity kind must be nonblank.")
        if ordinal < 0:
            raise ValueError("Segment identity ordinal must be nonnegative.")
        if not math.isfinite(t_start) or not math.isfinite(t_end):
            raise ValueError("Segment identity bounds must be finite.")
        if t_end < t_start:
            raise ValueError("Segment identity end must not precede its start.")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "ordinal", ordinal)
        object.__setattr__(self, "t_start", t_start)
        object.__setattr__(self, "t_end", t_end)

    @property
    def key(self) -> str:
        return f"{self.kind}:{self.ordinal}"

    def to_payload(self) -> dict[str, object]:
        return {
            "key": self.key,
            "kind": self.kind,
            "ordinal": self.ordinal,
            "t_start": self.t_start,
            "t_end": self.t_end,
        }


@dataclass(frozen=True)
class SegmentExecutionRequest:
    identity: SegmentIdentity
    request: SimulationRequest
    rhs: Rhs2
    t0: float
    t1: float
    y0: np.ndarray
    t_eval: np.ndarray
    method: str
    boundary_events: tuple[Callable[[float, np.ndarray], float], ...] = ()
    boundary_terminal_flags: tuple[bool, ...] = ()
    volume: VolumeExecution | None = None


@dataclass(frozen=True)
class SegmentRunResult:
    t: np.ndarray
    Y: np.ndarray
    event_times: tuple[tuple[float, ...], ...]
    raw_event_states: tuple[tuple[np.ndarray, ...], ...]
    raw_terminal_state: np.ndarray | None
    trajectory_projection: NonnegativeProjection
    event_state_projection: NonnegativeProjection
    symbolic_jacobian_used: bool = False
    dense_output: Callable[[np.ndarray], np.ndarray] | None = None

    def __post_init__(self) -> None:
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
        state = self.raw_terminal_state
        if state is None:
            return
        if state.ndim != 1 or not np.all(np.isfinite(state)):
            raise ValueError(
                "Raw solver terminal state must be finite and one-dimensional."
            )
        state.setflags(write=False)
        object.__setattr__(self, "raw_terminal_state", state)


@dataclass(frozen=True, slots=True)
class SegmentProjection:
    identity: SegmentIdentity
    trajectory: NonnegativeProjection
    event_states: NonnegativeProjection
    continuation_state: NonnegativeProjection = NonnegativeProjection()

    def with_continuation(
        self,
        projection: NonnegativeProjection,
    ) -> "SegmentProjection":
        return replace(
            self,
            continuation_state=NonnegativeProjection.aggregate(
                (self.continuation_state, projection)
            ),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "segment_identity": self.identity.to_payload(),
            "trajectory": self.trajectory.to_payload(),
            "event_states": self.event_states.to_payload(),
            "continuation_state": self.continuation_state.to_payload(),
        }


@dataclass(frozen=True, slots=True)
class _BoundaryApplicationResult:
    continuation_state: np.ndarray
    projection: NonnegativeProjection
    temperature_K: float
    presentation_boundary: SimulationPresentationBoundary | None = None
    termination: SimulationTermination | None = None
    skipped_instants: tuple[ExecutableInstant, ...] = ()


class BaseSegmentRunner(Protocol):
    def __call__(self, request: SegmentExecutionRequest) -> SegmentRunResult: ...


@dataclass(frozen=True)
class ScheduleExecutionRequest:
    request: SimulationRequest
    t_eval: np.ndarray
    provenance: dict[str, object]
    method: str


class InterventionScheduleExecutionOwner:
    def __init__(self, segment_runner: BaseSegmentRunner) -> None:
        self._segment_runner = segment_runner

    def execute(self, request: ScheduleExecutionRequest) -> FinalSimulationOutput:
        req = request.request

        def temperature_at(time, *, side="following"):
            return temperature_at_execution_time(
                req.temperature_schedule, req.reference_temperature_K, time, side=side,
            )

        t0, t1 = req.t_span
        t_eval = np.asarray(request.t_eval, dtype=float).reshape(-1)
        contract = req.intervention_execution
        has_stops = any(isinstance(item.then, InterventionStopAction) for item in contract.triggers)
        dense_segments: list[tuple[SegmentIdentity, Callable[[np.ndarray], np.ndarray], tuple[ExecutableHold, ...], int]] = []
        segment_result = None
        dense = None
        try:
            species_index = _species_index(req, contract)
            volume = VolumeExecution(species_names=species_index, parameter_values=req.parameter_values_at_temperature,
                                     end_time=t1)
            current_y = np.asarray(req.y0, dtype=float).reshape(-1).copy()
            cancellation_check = (
                req.cancellation_control.cancel_requested
                if req.cancellation_control is not None
                else None
            )
            trigger_counts = [0 for _ in contract.triggers]
            trigger_last_times: list[float | None] = [None for _ in contract.triggers]
            trigger_firings: list[TriggerFiring] = []
            executed_values: dict[ExecutableInstant, ExecutableInstant] = {}
            outputs_t: list[np.ndarray] = []
            outputs_y: list[np.ndarray] = []
            outputs_temperature: list[np.ndarray] = []
            output_segment_ids: list[np.ndarray] = []
            output_segment_keys: list[tuple[str, ...]] = []
            output_discontinuity_ids: list[np.ndarray] = []
            presentation_boundaries: list[SimulationPresentationBoundary] = []
            integration_segments: list[dict[str, object]] = []
            segment_projections: dict[str, SegmentProjection] = {}
            segment_count = 0
            discontinuity_id = 0
            temperature_boundaries = declared_temperature_step_boundaries(
                req.temperature_schedule,
                t_start=t0,
                t_end=t1,
            )

            boundaries = {
                t0,
                t1,
                *(
                    value
                    for value in contract.discontinuities
                    if t0 <= float(value) <= t1
                ),
                *temperature_boundaries,
            }
            if any(not math.isfinite(value) for value in boundaries):
                raise InterventionScheduleError(
                    "Execution boundary arithmetic must remain finite."
                )
            ordered_boundaries = sorted(boundaries)

            _raise_if_cancelled(cancellation_check)
            current_temperature = temperature_at(t0)
            if contract.initial_volume is not None:
                volume.initialize(contract.initial_volume, t0, current_temperature)
            initial_state_transaction = t0 in contract.discontinuities
            initial_boundary = _apply_boundary(
                current_y,
                boundary=t0,
                contract=contract,
                trigger_indices=(),
                trigger_state=None,
                input_is_raw_solver_state=False,
                trigger_counts=trigger_counts,
                trigger_last_times=trigger_last_times,
                trigger_firings=trigger_firings,
                species_index=species_index,
                cancellation_check=cancellation_check,
                observable_value=req.stop_observable_value,
                temperature_step=t0 in temperature_boundaries,
            parameter_values=req.parameter_values_at_temperature, executed_values=executed_values, volume=volume,
                                   temperature_K=current_temperature,
                                   temperature_at=temperature_at,
                               )
            current_y = initial_boundary.continuation_state
            current_temperature = initial_boundary.temperature_K
            termination = initial_boundary.termination
            skipped_instants = initial_boundary.skipped_instants
            _append_presentation_boundary(
                presentation_boundaries,
                initial_boundary.presentation_boundary,
            )
            _append_identity_row(
                outputs_t,
                outputs_y,
                output_segment_ids,
                output_segment_keys,
                output_discontinuity_ids,
                t0,
                current_y,
                segment_id=0,
                segment_key="integration:0",
                discontinuity_id=discontinuity_id,
                output_temperatures=outputs_temperature,
                temperature_K=current_temperature,
            )
            _raise_if_cancelled(cancellation_check)
            segment_starts_after_discontinuity = initial_state_transaction
            # Species released by a drain-release event, and the time of that
            # release; they are not held again in the sub-segment starting then.
            released_time: float | None = None
            released: frozenset[int] = frozenset()
            for boundary_index in range(len(ordered_boundaries) - 1):
                if termination is not None:
                    break
                outer_start = float(ordered_boundaries[boundary_index])
                outer_end = float(ordered_boundaries[boundary_index + 1])
                if outer_end <= outer_start:
                    continue
                sub_start = outer_start
                while sub_start < outer_end and termination is None:
                    _raise_if_cancelled(cancellation_check)
                    next_rearm = _next_rearm_boundary(
                        contract.triggers,
                        segment_start=sub_start,
                        segment_end=outer_end,
                        trigger_counts=trigger_counts,
                        trigger_last_times=trigger_last_times,
                    )
                    sub_end = outer_end if next_rearm is None else float(next_rearm)
                    active_fluxes = _active_fluxes(contract, sub_start)
                    active_holds = _active_holds(contract, sub_start)
                    current_y = _apply_holds(
                        current_y, active_holds, species_index=species_index,
                        cancellation_check=cancellation_check, time=sub_start,
                        temperature_K=current_temperature,
                        parameter_values=req.parameter_values_at_temperature,
                    )

                    def segment_temperature(time, end=sub_end):
                        return temperature_at(time, side=(
                            "preceding" if time == end and end in temperature_boundaries
                            else "following"
                        ))

                    # A drained species at zero whose free derivative is
                    # negative is held at zero like an authored hold; a
                    # terminal release event ends that regime.
                    free_rhs = _InterventionRhs(
                        req.rhs, fluxes=active_fluxes, holds=active_holds, volume=volume,
                        species_index=species_index,
                        parameter_values=req.parameter_values_at_temperature,
                        temperature_at=segment_temperature,
                        cancellation_check=cancellation_check,
                    )
                    release_events, release_mapping = _drain_release_events(
                        free_rhs, active_holds=active_holds, species_index=species_index,
                        state=current_y, segment_start=sub_start,
                        released=released if released_time == sub_start else frozenset(),
                    )
                    active_holds = active_holds + tuple(
                        ExecutableHold(
                            species=name, start_time=sub_start, end_time=sub_end,
                            value=0.0, declaration_ordinal=-1,
                        )
                        for name, index in species_index.items()
                        if index in release_mapping
                    )

                    segment_rhs = _InterventionRhs(
                        req.rhs, fluxes=active_fluxes, holds=active_holds, volume=volume,
                        species_index=species_index,
                        parameter_values=req.parameter_values_at_temperature,
                        temperature_at=segment_temperature,
                        cancellation_check=cancellation_check,
                    )
                    trigger_callables, trigger_mapping = _trigger_callables(
                        contract.triggers, species_index=species_index,
                        segment_start=sub_start, segment_state=current_y,
                        trigger_counts=trigger_counts, trigger_last_times=trigger_last_times,
                        observable_value=req.stop_observable_value,
                        temperature_at=segment_temperature,
                        parameter_values=req.parameter_values_at_temperature, holds=active_holds,
                    )
                    # SciPy truncates a simultaneous active-event set at the first
                    # terminal event.  Register every trigger first as a
                    # non-terminal observer, then repeat the same callables as
                    # terminal stoppers.  The observer block therefore retains
                    # every exact equal-time root before the first stopper ends
                    # the segment.
                    trigger_observers = trigger_callables
                    trigger_stoppers = trigger_callables
                    purge_observers, purge_mapping = _purge_zero_crossing_events(
                        active_fluxes,
                        active_holds=active_holds,
                        species_index=species_index,
                        state=current_y,
                        segment_start=sub_start,
                    )
                    purge_observers = purge_observers + release_events
                    purge_mapping = purge_mapping + release_mapping
                    purge_stoppers = purge_observers
                    segment_events = (
                        trigger_observers
                        + purge_observers
                        + trigger_stoppers
                        + purge_stoppers
                    )
                    segment_request = req
                    segment_terminal_flags = (
                            tuple(False for _ in trigger_observers)
                            + tuple(False for _ in purge_observers)
                            + tuple(True for _ in trigger_stoppers)
                            + tuple(True for _ in purge_stoppers)
                    )
                    segment_symbolic_jacobian = segment_request.symbolic_jacobian
                    if segment_symbolic_jacobian.jacobian_func is not None:
                        segment_request = replace(
                            segment_request,
                            symbolic_jacobian=replace(
                                segment_symbolic_jacobian,
                                jacobian_func=_InterventionJacobian(
                                    segment_symbolic_jacobian.jacobian_func,
                                    rhs=segment_rhs, holds=active_holds, fluxes=active_fluxes,
                                    species_index=species_index,
                                ),
                            ),
                        )
                    requested = _requested_eval(
                        t_eval,
                        start=sub_start,
                        end=sub_end,
                        is_final=(sub_end == t1 and sub_end not in temperature_boundaries),
                    )
                    if has_stops:
                        requested = np.empty(0, dtype=float)
                    internal_eval = requested
                    if not internal_eval.size or float(internal_eval[-1]) != sub_end:
                        internal_eval = np.concatenate(
                            [internal_eval, np.array([sub_end], dtype=float)]
                        )
                    segment_identity = SegmentIdentity(
                        kind="integration",
                        ordinal=segment_count,
                        t_start=sub_start,
                        t_end=sub_end,
                    )
                    _raise_if_cancelled(cancellation_check)
                    segment_result = self._segment_runner(
                        SegmentExecutionRequest(
                            identity=segment_identity,
                            request=segment_request,
                            rhs=segment_rhs,
                            t0=sub_start,
                            t1=sub_end,
                            y0=current_y,
                            t_eval=internal_eval,
                            method=request.method,
                            boundary_events=segment_events,
                            boundary_terminal_flags=segment_terminal_flags,
                            volume=volume if volume.value is not None else None,
                        )
                    )
                    _raise_if_cancelled(cancellation_check)
                    event_times = segment_result.event_times
                    event_states = segment_result.raw_event_states
                    trigger_hit = (
                        _first_trigger_hit(
                            event_times,
                            event_states,
                            trigger_mapping=trigger_mapping,
                        )
                    )
                    purge_hit = (
                        _first_purge_hit(
                            event_times,
                            event_states,
                            purge_mapping=purge_mapping,
                            event_offset=len(trigger_observers),
                        )
                    )
                    if (
                        trigger_hit is not None
                        and purge_hit is not None
                        and trigger_hit[0] == purge_hit[0]
                    ):
                        trigger_hit = (
                            trigger_hit[0],
                            trigger_hit[1],
                            purge_hit[2].copy(),
                        )
                    if (
                        purge_hit is not None
                        and (
                            trigger_hit is None
                            or purge_hit[0] < trigger_hit[0]
                        )
                    ):
                        trigger_hit = None
                    seg_t = segment_result.t
                    seg_temperature = _selected_temperatures(seg_t, segment_temperature, cancellation_check)
                    seg_y = _apply_holds_to_trajectory(
                        segment_result.Y,
                        active_holds,
                        species_index=species_index,
                        cancellation_check=cancellation_check,
                    times=seg_t, temperatures=seg_temperature, parameter_values=req.parameter_values_at_temperature,)
                    actual_segment_end = (
                        float(trigger_hit[0])
                        if trigger_hit is not None
                        else (
                            float(purge_hit[0])
                            if purge_hit is not None
                            else (
                                sub_end
                            )
                        )
                    )
                    segment_output_identity = replace(
                        segment_identity,
                        t_end=actual_segment_end,
                    )
                    if has_stops:
                        dense_segments.append((segment_output_identity, segment_result.dense_output, active_holds, discontinuity_id))
                    _record_segment_projection(
                        segment_projections,
                        segment_result,
                        identity=segment_output_identity,
                    )
                    if segment_count == 0:
                        _record_continuation_projection(
                            segment_projections,
                            identity=segment_output_identity,
                            projection=initial_boundary.projection,
                        )
                    segment_payload: dict[str, object] = {
                        "segment_id": segment_count,
                        "segment_key": segment_output_identity.key,
                        "segment_identity": segment_output_identity.to_payload(),
                        "t_start": segment_output_identity.t_start,
                        "t_end": segment_output_identity.t_end,
                        "starts_after_discontinuity": bool(
                            segment_starts_after_discontinuity
                        ),
                        "temperature_step_boundary": (
                            segment_count > 0
                            and sub_start in temperature_boundaries
                        ),
                        "symbolic_jacobian_used": bool(
                            segment_result.symbolic_jacobian_used
                        ),
                    }
                    integration_segments.append(segment_payload)
                    segment_count += 1
                    current_temperature = (
                        float(seg_temperature[-1])
                        if seg_t.size and seg_t[-1] == actual_segment_end
                        else segment_temperature(actual_segment_end)
                    )
                    if trigger_hit is not None:
                        terminal_state = trigger_hit[2]
                    elif purge_hit is not None:
                        terminal_state = purge_hit[2]
                    else:
                        terminal_state = segment_result.raw_terminal_state
                        if terminal_state is None:
                            raise InterventionScheduleError(
                                "Completed solver segment is missing its exact raw terminal state."
                            )
                    current_y = _apply_holds(
                        terminal_state, active_holds, species_index=species_index,
                        cancellation_check=cancellation_check, time=actual_segment_end,
                        temperature_K=current_temperature,
                        parameter_values=req.parameter_values_at_temperature,
                    )
                    if trigger_hit is not None:
                        trigger_hit = (trigger_hit[0], trigger_hit[1], current_y.copy())
                    elif purge_hit is not None:
                        current_y, purge_projection = project_nonnegative_trajectory(
                            current_y.reshape(-1, 1),
                            context=f"Purge continuation state at t={purge_hit[0]:g}",
                        )
                        current_y = current_y.reshape(-1)
                        _record_continuation_projection(
                            segment_projections,
                            identity=segment_output_identity,
                            projection=purge_projection,
                        )
                        purge_hit = (purge_hit[0], purge_hit[1], current_y.copy())
                    scheduled_discontinuity = (
                        trigger_hit is None
                        and (
                            purge_hit is None
                            or purge_hit[0] == sub_end
                        )
                        and sub_end in contract.discontinuities
                    )
                    emit_count = min(requested.size, seg_t.size)
                    emit_t = seg_t[:emit_count]
                    emit_y = seg_y[:, :emit_count]
                    emit_temperature = seg_temperature[:emit_count]
                    if trigger_hit is not None:
                        emit_mask = emit_t < float(trigger_hit[0])
                        emit_t = emit_t[emit_mask]
                        emit_y = emit_y[:, emit_mask]
                        emit_temperature = emit_temperature[emit_mask]
                    elif purge_hit is not None:
                        emit_mask = emit_t < float(purge_hit[0])
                        emit_t = emit_t[emit_mask]
                        emit_y = emit_y[:, emit_mask]
                        emit_temperature = emit_temperature[emit_mask]
                    elif scheduled_discontinuity:
                        emit_mask = emit_t < float(sub_end)
                        emit_t = emit_t[emit_mask]
                        emit_y = emit_y[:, emit_mask]
                        emit_temperature = emit_temperature[emit_mask]
                    _append_identity_samples(
                        outputs_t,
                        outputs_y,
                        output_segment_ids,
                        output_segment_keys,
                        output_discontinuity_ids,
                        emit_t,
                        emit_y,
                        segment_id=segment_identity.ordinal,
                        segment_key=segment_identity.key,
                        discontinuity_id=discontinuity_id,
                        output_temperatures=outputs_temperature,
                        temperatures=emit_temperature,
                    )
                    if trigger_hit is None:
                        if purge_hit is not None and purge_hit[0] < sub_end:
                            _append_identity_row(
                                outputs_t,
                                outputs_y,
                                output_segment_ids,
                                output_segment_keys,
                                output_discontinuity_ids,
                                float(purge_hit[0]),
                                np.asarray(purge_hit[2], dtype=float).reshape(-1),
                                segment_id=segment_count,
                                segment_key=f"integration:{segment_count}",
                                discontinuity_id=discontinuity_id,
                                output_temperatures=outputs_temperature,
                                temperature_K=current_temperature,
                            )
                            sub_start = float(purge_hit[0])
                            released_time = sub_start
                            released = frozenset(
                                index for index in purge_hit[1] if index in release_mapping
                            )
                            segment_starts_after_discontinuity = False
                            continue
                        if next_rearm is not None and sub_end < outer_end:
                            if purge_hit is not None and purge_hit[0] == sub_end:
                                _append_identity_row(
                                    outputs_t,
                                    outputs_y,
                                    output_segment_ids,
                                    output_segment_keys,
                                    output_discontinuity_ids,
                                    float(purge_hit[0]),
                                    np.asarray(purge_hit[2], dtype=float).reshape(-1),
                                    segment_id=segment_count,
                                    segment_key=f"integration:{segment_count}",
                                    discontinuity_id=discontinuity_id,
                                    output_temperatures=outputs_temperature,
                                    temperature_K=current_temperature,
                                )
                            boundary_result = _apply_boundary(
                                current_y,
                                boundary=sub_end,
                                contract=contract,
                                trigger_indices=(),
                                trigger_state=None,
                                input_is_raw_solver_state=True,
                                trigger_counts=trigger_counts,
                                trigger_last_times=trigger_last_times,
                                trigger_firings=trigger_firings,
                                species_index=species_index,
                                cancellation_check=cancellation_check,
                                observable_value=req.stop_observable_value,
                                temperature_step=sub_end in temperature_boundaries,
                            parameter_values=req.parameter_values_at_temperature, executed_values=executed_values, volume=volume,
                                                  temperature_K=current_temperature,
                                                  temperature_at=temperature_at,
                                              )
                            current_y = boundary_result.continuation_state
                            current_temperature = boundary_result.temperature_K
                            termination = boundary_result.termination
                            skipped_instants = boundary_result.skipped_instants
                            _append_presentation_boundary(
                                presentation_boundaries,
                                boundary_result.presentation_boundary,
                            )
                            _raise_if_cancelled(cancellation_check)
                            _record_continuation_projection(
                                segment_projections,
                                identity=segment_output_identity,
                                projection=boundary_result.projection,
                            )
                            segment_starts_after_discontinuity = False
                            sub_start = sub_end
                            if termination is not None:
                                break
                            continue
                        if sub_end == outer_end:
                            boundary_result = _apply_boundary(
                                current_y,
                                boundary=outer_end,
                                contract=contract,
                                trigger_indices=(),
                                trigger_state=None,
                                input_is_raw_solver_state=True,
                                trigger_counts=trigger_counts,
                                trigger_last_times=trigger_last_times,
                                trigger_firings=trigger_firings,
                                species_index=species_index,
                                cancellation_check=cancellation_check,
                                observable_value=req.stop_observable_value,
                                temperature_step=outer_end in temperature_boundaries,
                            parameter_values=req.parameter_values_at_temperature, executed_values=executed_values, volume=volume,
                                                  temperature_K=current_temperature,
                                                  temperature_at=temperature_at,
                                              )
                            current_y = boundary_result.continuation_state
                            current_temperature = boundary_result.temperature_K
                            termination = boundary_result.termination
                            skipped_instants = boundary_result.skipped_instants
                            _append_presentation_boundary(
                                presentation_boundaries,
                                boundary_result.presentation_boundary,
                            )
                            _raise_if_cancelled(cancellation_check)
                            _record_continuation_projection(
                                segment_projections,
                                identity=segment_output_identity,
                                projection=boundary_result.projection,
                            )
                            if scheduled_discontinuity:
                                discontinuity_id += 1
                                post_segment_id = (
                                    segment_output_identity.ordinal
                                    if outer_end == t1 or termination is not None
                                    else segment_count
                                )
                                _append_identity_row(
                                    outputs_t,
                                    outputs_y,
                                    output_segment_ids,
                                    output_segment_keys,
                                    output_discontinuity_ids,
                                    float(outer_end),
                                    np.asarray(current_y, dtype=float).reshape(-1),
                                    segment_id=post_segment_id,
                                    segment_key=f"integration:{post_segment_id}",
                                    discontinuity_id=discontinuity_id,
                                    output_temperatures=outputs_temperature,
                                    temperature_K=current_temperature,
                                )
                                segment_starts_after_discontinuity = outer_end < t1
                            elif (
                                outer_end in temperature_boundaries
                                and outer_end < t1
                            ):
                                post_segment_id = (
                                    segment_output_identity.ordinal
                                    if termination is not None
                                    else segment_count
                                )
                                _append_identity_row(
                                    outputs_t,
                                    outputs_y,
                                    output_segment_ids,
                                    output_segment_keys,
                                    output_discontinuity_ids,
                                    float(outer_end),
                                    np.asarray(current_y, dtype=float).reshape(-1),
                                    segment_id=post_segment_id,
                                    segment_key=f"integration:{post_segment_id}",
                                    discontinuity_id=discontinuity_id,
                                    output_temperatures=outputs_temperature,
                                    temperature_K=current_temperature,
                                )
                                segment_starts_after_discontinuity = False
                            elif outer_end == t1:
                                _append_identity_row(
                                    outputs_t,
                                    outputs_y,
                                    output_segment_ids,
                                    output_segment_keys,
                                    output_discontinuity_ids,
                                    float(outer_end),
                                    np.asarray(current_y, dtype=float).reshape(-1),
                                    segment_id=segment_output_identity.ordinal,
                                    segment_key=segment_output_identity.key,
                                    discontinuity_id=discontinuity_id,
                                    output_temperatures=outputs_temperature,
                                    temperature_K=current_temperature,
                                )
                                segment_starts_after_discontinuity = False
                            else:
                                segment_starts_after_discontinuity = False
                        break
                    root_time, trigger_indices, root_state = trigger_hit
                    boundary_result = _apply_boundary(
                        current_y,
                        boundary=root_time,
                        contract=contract,
                        trigger_indices=trigger_indices,
                        trigger_state=root_state,
                        input_is_raw_solver_state=True,
                        trigger_counts=trigger_counts,
                        trigger_last_times=trigger_last_times,
                        trigger_firings=trigger_firings,
                        species_index=species_index,
                        cancellation_check=cancellation_check,
                        observable_value=req.stop_observable_value,
                        temperature_step=root_time in temperature_boundaries,
                    parameter_values=req.parameter_values_at_temperature, executed_values=executed_values, volume=volume,
                                          temperature_K=current_temperature,
                                          temperature_at=temperature_at,
                                      )
                    current_y = boundary_result.continuation_state
                    current_temperature = boundary_result.temperature_K
                    termination = boundary_result.termination
                    skipped_instants = boundary_result.skipped_instants
                    _append_presentation_boundary(
                        presentation_boundaries,
                        boundary_result.presentation_boundary,
                    )
                    _raise_if_cancelled(cancellation_check)
                    _record_continuation_projection(
                        segment_projections,
                        identity=segment_output_identity,
                        projection=boundary_result.projection,
                    )
                    if termination is None or boundary_result.presentation_boundary is not None:
                        discontinuity_id += 1
                    post_segment_id = (
                        segment_output_identity.ordinal
                        if root_time == t1 or termination is not None
                        else segment_count
                    )
                    _append_identity_row(
                        outputs_t,
                        outputs_y,
                        output_segment_ids,
                        output_segment_keys,
                        output_discontinuity_ids,
                        float(root_time),
                        np.asarray(current_y, dtype=float).reshape(-1),
                        segment_id=post_segment_id,
                        segment_key=f"integration:{post_segment_id}",
                        discontinuity_id=discontinuity_id,
                        output_temperatures=outputs_temperature,
                        temperature_K=current_temperature,
                    )
                    segment_starts_after_discontinuity = root_time < t1
                    sub_start = root_time

            actual_end = t1 if termination is None else termination.actual_endpoint
            if termination is None:
                termination = SimulationTermination(t1, "endpoint")
            if outputs_t:
                output_t = np.concatenate(outputs_t)
                output_y = np.concatenate(outputs_y, axis=1)
                output_temperature = np.concatenate(outputs_temperature)
            else:
                output_t = np.empty(0, dtype=float)
                output_y = np.empty((current_y.size, 0), dtype=float)
                output_temperature = np.empty(0, dtype=float)
            if not output_t.size or float(output_t[-1]) != actual_end:
                _append_identity_row(
                    outputs_t,
                    outputs_y,
                    output_segment_ids,
                    output_segment_keys,
                    output_discontinuity_ids,
                    actual_end,
                    np.asarray(current_y, dtype=float).reshape(-1),
                    segment_id=max(0, segment_count - 1),
                    segment_key=f"integration:{max(0, segment_count - 1)}",
                    discontinuity_id=discontinuity_id,
                    output_temperatures=outputs_temperature,
                    temperature_K=current_temperature,
                )
                output_t = np.concatenate(outputs_t)
                output_y = np.concatenate(outputs_y, axis=1)
                output_temperature = np.concatenate(outputs_temperature)
            if has_stops:
                from kindred.core.time_grid import build_time_grid
                try:
                    base_times = (np.array([t0], dtype=float) if actual_end == t0
                                  else build_time_grid(t0, actual_end, req.grid))
                except ValueError as exc:
                    raise InterventionScheduleError(f"Actual-interval sampling failed: {exc}") from exc
                boundary_ids, boundary_keys, boundary_discontinuities = _compose_direct_result_sample_identity(
                    output_segment_ids, output_segment_keys, output_discontinuity_ids,
                )
                final_times = np.union1d(base_times, output_t)
                final_states = np.empty((current_y.size, final_times.size), dtype=float)
                final_temperatures = np.empty(final_times.size, dtype=float)
                final_ids = np.empty(final_times.size, dtype=np.int64)
                final_keys = np.empty(final_times.size, dtype=object)
                final_discontinuities = np.empty(final_times.size, dtype=np.int64)

                # Exact executor rows own intervention and termination boundaries.
                boundary_rows = np.searchsorted(final_times, output_t)
                final_states[:, boundary_rows] = output_y
                final_temperatures[boundary_rows] = output_temperature
                final_ids[boundary_rows] = boundary_ids
                final_keys[boundary_rows] = boundary_keys
                final_discontinuities[boundary_rows] = boundary_discontinuities
                needs_sampling = np.ones(final_times.size, dtype=bool)
                needs_sampling[boundary_rows] = False
                sample_rows = np.flatnonzero(needs_sampling)
                sample_times = final_times[sample_rows]
                next_sample = 0

                for dense_index, (identity_record, dense, holds, discontinuity) in enumerate(dense_segments):
                    _raise_if_cancelled(cancellation_check)
                    # An internal endpoint belongs to the following segment. Use
                    # executed bounds: SciPy's last step can extend past a root.
                    sample_end = int(np.searchsorted(
                        sample_times, identity_record.t_end,
                        side="right" if dense_index + 1 == len(dense_segments) else "left",
                    ))
                    if sample_end == next_sample:
                        continue
                    if sample_times[next_sample] < identity_record.t_start:
                        raise InterventionScheduleError("Actual-interval sample is outside its continuous segment.")
                    projected_count = 0
                    max_negative_magnitude = 0.0
                    # Bound interpolation temporaries and check cancellation on
                    # both sides of every group, including a short final group.
                    for sample_start in range(next_sample, sample_end, 256):
                        _raise_if_cancelled(cancellation_check)
                        chunk_end = min(sample_start + 256, sample_end)
                        rows = sample_rows[sample_start:chunk_end]
                        selected_times = sample_times[sample_start:chunk_end]
                        selected_temperatures = _selected_temperatures(selected_times, temperature_at, cancellation_check)
                        raw_states = dense(selected_times)
                        _raise_if_cancelled(cancellation_check)
                        raw_states = _apply_holds_to_trajectory(
                            raw_states, holds, species_index=species_index,
                            cancellation_check=cancellation_check,
                        times=selected_times, temperatures=selected_temperatures, parameter_values=req.parameter_values_at_temperature,)
                        projected, projection = project_nonnegative_trajectory(
                            raw_states, context="Actual-interval sampling",
                        )
                        _raise_if_cancelled(cancellation_check)
                        final_states[:, rows] = projected
                        final_temperatures[rows] = selected_temperatures
                        final_ids[rows] = identity_record.ordinal
                        final_keys[rows] = identity_record.key
                        final_discontinuities[rows] = discontinuity
                        projected_count += projection.count
                        max_negative_magnitude = max(
                            max_negative_magnitude, projection.max_negative_magnitude,
                        )
                    _record_continuation_projection(
                        segment_projections, identity=identity_record,
                        projection=NonnegativeProjection(
                            projected=bool(projected_count), count=projected_count,
                            max_negative_magnitude=max_negative_magnitude,
                        ),
                    )
                    next_sample = sample_end
                if next_sample != sample_times.size:
                    raise InterventionScheduleError("Actual-interval sample is outside its continuous segment.")
                _raise_if_cancelled(cancellation_check)
                output_t, output_y = final_times, final_states
                output_temperature = final_temperatures
                output_segment_ids, output_segment_keys, output_discontinuity_ids = [final_ids], [tuple(final_keys)], [final_discontinuities]
                dense_segments.clear()
            if output_t.size > 1 and np.any(np.diff(output_t) <= 0.0):
                raise InterventionScheduleError(
                    "Public result row times must be strictly increasing."
                )
            # Concatenation creates the final owned buffers.  Freeze them here so
            # the standalone result can adopt them without defensive recapture.
            output_t.setflags(write=False)
            output_y.setflags(write=False)
            output_temperature.setflags(write=False)
            final_contract = replace(
                contract,
                trigger_firings=tuple(trigger_firings),
                instants=tuple(executed_values.get(item, item) for item in contract.instants),
                skipped_instants=skipped_instants,
            )
            provenance = request.provenance
            _raise_if_cancelled(cancellation_check)
            provenance["intervention_execution"] = final_contract.provenance_projection()
            provenance["integration_segments"] = integration_segments
            provenance["intervention_segments"] = segment_count
            (
                result_segment_ids,
                result_segment_keys,
                result_discontinuity_ids,
            ) = _compose_direct_result_sample_identity(
                output_segment_ids,
                output_segment_keys,
                output_discontinuity_ids,
            )
            _publish_segment_projection_provenance(
                provenance,
                segment_projections,
            )
            _raise_if_cancelled(cancellation_check)
            return FinalSimulationOutput(
                t=output_t,
                Y=output_y,
                volume=volume.trajectory(output_t),
                temperature_K=output_temperature,
                provenance=provenance,
                intervention_execution=final_contract,
                presentation_boundaries=tuple(presentation_boundaries),
                segment_ids=result_segment_ids,
                segment_keys=result_segment_keys,
                discontinuity_ids=result_discontinuity_ids,
                termination=termination,
            )
        finally:
            dense_segments.clear()
            segment_result = None
            dense = None


def _append_presentation_boundary(
    boundaries: list[SimulationPresentationBoundary],
    boundary: SimulationPresentationBoundary | None,
) -> None:
    if boundary is None:
        return
    if boundaries and boundary.time <= boundaries[-1].time:
        raise InterventionScheduleError(
            "Presentation boundaries must remain strictly ordered."
        )
    boundaries.append(boundary)


def _raise_if_cancelled(check: Callable[[], bool] | None) -> None:
    if check is not None and check():
        raise SimulationCancelled()


def _compose_direct_result_sample_identity(
    segment_ids: Sequence[np.ndarray],
    segment_keys: Sequence[tuple[str, ...]],
    discontinuity_ids: Sequence[np.ndarray],
) -> tuple[np.ndarray, tuple[str, ...], np.ndarray]:
    """Compose row identity captured at emission time.

    Identity is captured while each unique segment/root row is emitted, so no
    downstream boundary reconstructs execution authority from timestamps.
    """

    ids = (
        np.concatenate(segment_ids).astype(np.int64, copy=False)
        if segment_ids
        else np.empty(0, dtype=np.int64)
    )
    discontinuities = (
        np.concatenate(discontinuity_ids).astype(np.int64, copy=False)
        if discontinuity_ids
        else np.empty(0, dtype=np.int64)
    )
    keys = tuple(key for group in segment_keys for key in group)
    expected = int(ids.size)
    if len(keys) != expected or discontinuities.size != expected:
        raise InterventionScheduleError(
            "Direct result row identity is not aligned with emitted rows."
        )
    ids.setflags(write=False)
    discontinuities.setflags(write=False)
    return ids, keys, discontinuities


def _record_segment_projection(
    records: dict[str, SegmentProjection],
    result: SegmentRunResult,
    *,
    identity: SegmentIdentity,
) -> None:
    key = identity.key
    if key in records:
        raise InterventionScheduleError(
            f"Duplicate segment projection identity: {key!r}."
        )
    records[key] = SegmentProjection(
        identity=identity,
        trajectory=result.trajectory_projection,
        event_states=result.event_state_projection,
    )


def _record_continuation_projection(
    records: dict[str, SegmentProjection],
    *,
    identity: SegmentIdentity,
    projection: NonnegativeProjection,
) -> None:
    key = identity.key
    record = records.get(key)
    if record is None or record.identity != identity:
        raise InterventionScheduleError(
            f"Continuation projection has no exact segment identity: {key!r}."
        )
    records[key] = record.with_continuation(projection)


def _publish_segment_projection_provenance(
    provenance: dict[str, object],
    records: Mapping[str, SegmentProjection],
) -> None:
    if not records:
        return
    provenance["segment_nonnegative_projections"] = {
        key: record.to_payload()
        for key, record in records.items()
    }
    summaries = (
        (
            "nonnegative_projection",
            NonnegativeProjection.aggregate(
                record.trajectory for record in records.values()
            ),
        ),
        (
            "event_state_nonnegative_projection",
            NonnegativeProjection.aggregate(
                record.event_states for record in records.values()
            ),
        ),
        (
            "continuation_state_nonnegative_projection",
            NonnegativeProjection.aggregate(
                record.continuation_state for record in records.values()
            ),
        ),
    )
    for provenance_key, aggregate in summaries:
        if aggregate.projected:
            provenance[provenance_key] = aggregate.to_payload()
        else:
            provenance.pop(provenance_key, None)


def _species_index(
    req: SimulationRequest, contract: InterventionExecutionContract
) -> dict[str, int]:
    referenced = {
        *(name for item in contract.instants for name in intervention_species(item)),
        *(name for item in contract.fluxes for name in intervention_species(item)),
        *(item.species for item in contract.holds),
        *(item.condition.name for item in contract.triggers if isinstance(item.condition, SpeciesTriggerCondition)),
        *(name for item in contract.triggers if not isinstance(item.then, InterventionStopAction) for name in intervention_species(item.then)),
    }
    names = tuple(str(name) for name in (req.species_names or ()))
    if not names and referenced:
        raise InterventionScheduleError(
            "species_names are required when intervention_schedule references species."
        )
    available = set(names)
    missing = sorted(referenced - available)
    if missing:
        raise InterventionScheduleError(
            f"Unknown intervention species: {missing[0]}"
        )
    return {name: index for index, name in enumerate(names)}


def _static_instants_at(
    contract: InterventionExecutionContract, time: float
) -> tuple[ExecutableInstant, ...]:
    # The compiler owns chronological order, including ties within each time.
    by_time = attrgetter("time")
    first = bisect_left(contract.instants, time, key=by_time)
    last = bisect_right(contract.instants, time, lo=first, key=by_time)
    return contract.instants[first:last]


def _active_fluxes(
    contract: InterventionExecutionContract, time: float
) -> tuple[ExecutableFlux, ...]:
    return tuple(
        item
        for item in contract.fluxes
        if item.start_time <= time < item.end_time
    )


def _active_holds(
    contract: InterventionExecutionContract, time: float
) -> tuple[ExecutableHold, ...]:
    return tuple(
        item
        for item in contract.holds
        if item.start_time <= time < item.end_time
    )


def _purge_zero_crossing_events(
    fluxes: Sequence[ExecutableFlux],
    *,
    active_holds: Sequence[ExecutableHold],
    species_index: Mapping[str, int],
    state: np.ndarray,
    segment_start: float = 0.0,
) -> tuple[
    tuple[Callable[[float, np.ndarray], float], ...],
    tuple[int, ...],
]:
    """Create private terminal events for active downward zero crossings."""

    purge_indices: set[int] = set()
    for flux in fluxes:
        if flux.physical is not None:
            continue
        index = species_index[flux.species]
        if flux.flux_rate_param is not None or flux.flux_rate < 0.0:
            purge_indices.add(index)
    held = {species_index[hold.species] for hold in active_holds}
    values = np.asarray(state, dtype=float).reshape(-1)
    indices = tuple(index for index in sorted(purge_indices) if index not in held)
    initial_values = values.copy()
    events: list[Callable[[float, np.ndarray], float]] = []
    for index in indices:

        def _zero_crossing(
            _time: float,
            values: np.ndarray,
            *,
            _index: int = index,
        ) -> float:
            value = float(values[_index])
            if initial_values[_index] == 0.0 and (
                _time == segment_start or value == 0.0
            ):
                return -1.0
            return value

        _zero_crossing.terminal = True  # type: ignore[attr-defined]
        _zero_crossing.direction = -1.0  # type: ignore[attr-defined]
        events.append(_zero_crossing)
    return tuple(events), indices


def _trigger_action_instant(
    trigger: ExecutableTrigger, *, time: float
) -> ExecutableInstant:
    action = trigger.then
    return ExecutableInstant(
        op=action.op,
        physical=action.physical,
        species=action.species,
        time=time,
        value=action.value, value_param=action.value_param,
        delta=action.delta, delta_param=action.delta_param,
        declaration_ordinal=trigger.declaration_ordinal,
        phase="ordinary",
    )


def _trigger_value(
    trigger: ExecutableTrigger, time: float, state: np.ndarray,
    *, species_index: Mapping[str, int],
    observable_value: Callable[[str, float, np.ndarray, float], float] | None,
    temperature_K: float,
) -> float:
    try:
        if isinstance(trigger.condition, SpeciesTriggerCondition):
            value = float(state[species_index[trigger.condition.name]])
        else:
            value = float(observable_value(trigger.condition.name, time, state, temperature_K))
        if not math.isfinite(value):
            raise ValueError("the watched quantity is not finite")
        return value
    except Exception as exc:
        raise InterventionScheduleError(
            f"Stopping declaration {trigger.declaration_ordinal + 1}, quantity {trigger.condition.name!r}: {exc}",
            operation="trigger", declaration_ordinal=trigger.declaration_ordinal,
            field="watch_observable" if isinstance(trigger.condition, ObservableTriggerCondition) else "watch_species",
            details={"quantity": trigger.condition.name, "time": time},
        ) from exc


def _crossed(before: float, after: float, threshold: float, direction: str) -> bool:
    return ((direction in {"rising", "either"} and before < threshold <= after)
            or (direction in {"falling", "either"} and before > threshold >= after))


def _apply_boundary(
    state: np.ndarray,
    *,
    boundary: float,
    contract: InterventionExecutionContract,
    trigger_indices: Sequence[int],
    trigger_state: np.ndarray | None,
    input_is_raw_solver_state: bool,
    trigger_counts: list[int],
    trigger_last_times: list[float | None],
    trigger_firings: list[TriggerFiring],
    species_index: Mapping[str, int],
    cancellation_check: Callable[[], bool] | None,
    observable_value: Callable[[str, float, np.ndarray, float], float] | None = None,
    temperature_step: bool = False,
    temperature_K: float,
    temperature_at,
    parameter_values=None, executed_values=None, volume,
) -> _BoundaryApplicationResult:
    if executed_values is None:
        executed_values = {}
    _raise_if_cancelled(cancellation_check)
    current = np.asarray(trigger_state if trigger_state is not None else state, dtype=float).reshape(-1).copy()
    exact_root = current.copy()
    projectable_solver_negatives = exact_root < 0.0 if input_is_raw_solver_state else np.zeros(exact_root.shape, dtype=bool)
    stops = tuple(index for index, item in enumerate(contract.triggers) if isinstance(item.then, InterventionStopAction))
    active_holds = _active_holds(contract, boundary)
    static = _static_instants_at(contract, boundary)
    entering_holds = tuple(item for item in active_holds if item.start_time == boundary)
    chemical_indices = tuple(index for index in trigger_indices if index not in stops)
    trigger_instants = tuple(_trigger_action_instant(contract.triggers[index], time=boundary) for index in chemical_indices)
    root_temperature = temperature_K
    executed_instants: list[ExecutableInstant] = []
    states: list[SimulationBoundaryState] = []

    def snapshot() -> None:
        public = _project_public_event_state({name: float(current[index]) for name, index in species_index.items()})
        state = SimulationBoundaryState(tuple(public.items()), volume.value, temperature_K)
        if not states or state != states[-1]:
            states.append(state)

    snapshot()

    def record(index: int, action=None) -> None:
        trigger = contract.triggers[index]
        trigger_counts[index] += 1
        trigger_last_times[index] = boundary
        action_payload = trigger.then.to_payload() if action is None else action.to_payload()
        trigger_firings.append(TriggerFiring(
            declaration_ordinal=trigger.declaration_ordinal, condition=trigger.condition,
            threshold=_resolve_items((trigger,), boundary, parameter_values, temperature_K=root_temperature if index in trigger_indices else temperature_K)[0].threshold, direction=trigger.direction, root_time=boundary,
            event_state=_project_public_event_state({name: float((current if isinstance(trigger.then, InterventionStopAction) else exact_root)[position]) for name, position in species_index.items()}),
            firing_ordinal=trigger_counts[index], nested_action_payload=action_payload,
        ))

    def finish(stop: int | None = None) -> _BoundaryApplicationResult:
        if stop is None and boundary < volume.end_time:
            for flux in _active_fluxes(contract, boundary):
                if flux.physical is not None:
                    volume.require_volume(flux, boundary)
        snapshot()
        invalid_negative = (current < 0.0) & ~projectable_solver_negatives
        if np.any(invalid_negative):
            raise InterventionScheduleError("Intervention boundary produced a scientifically invalid negative continuation state.")
        continuation, projection = project_nonnegative_trajectory(current.reshape(-1, 1), context=f"Intervention continuation state at t={boundary:g}")
        termination = None
        if stop is not None:
            record(stop)
            termination = SimulationTermination(boundary, "stop", contract.triggers[stop].declaration_ordinal)
        return _BoundaryApplicationResult(
            continuation.reshape(-1), projection, temperature_K,
            SimulationPresentationBoundary(boundary, tuple(states)) if len(states) > 1 else None,
            termination,
            skipped_instants=(
                tuple(item for item in static if item not in executed_values)
                if stop is not None else ()
            ),
        )

    def crossed_stop(before: np.ndarray, *, before_temperature_K: float | None = None) -> int | None:
        for index in stops:
            trigger = contract.triggers[index]
            previous = _trigger_value(trigger, boundary, before, species_index=species_index,
                                      observable_value=observable_value, temperature_K=temperature_K if before_temperature_K is None else before_temperature_K)
            following = _trigger_value(trigger, boundary, current, species_index=species_index, observable_value=observable_value, temperature_K=temperature_K)
            previous_threshold = _resolve_items((trigger,), boundary, parameter_values, temperature_K=temperature_K if before_temperature_K is None else before_temperature_K)[0].threshold
            following_threshold = _resolve_items((trigger,), boundary, parameter_values, temperature_K=temperature_K)[0].threshold
            if _crossed(previous - previous_threshold, following - following_threshold, 0.0, trigger.direction):
                return index
        return None

    eligible_stops = tuple(index for index in trigger_indices if index in stops)
    if eligible_stops:
        return finish(min(eligible_stops, key=lambda index: contract.triggers[index].declaration_ordinal))
    if temperature_step:
        temperature_K = temperature_at(boundary)
        snapshot()
    if not stops:
        validate_boundary_conflicts(
            instants=_resolve_items(static + trigger_instants, boundary, parameter_values, temperature_K=temperature_K),
            entering_holds=_resolve_items(entering_holds, boundary, parameter_values, temperature_K=temperature_K),
            cancellation_check=cancellation_check,
        )
    if temperature_step and stops:
        before_temperature = current.copy()
        current = _apply_holds(current, tuple(item for item in active_holds if item.start_time < boundary), species_index=species_index, cancellation_check=cancellation_check, time=boundary, temperature_K=temperature_K, parameter_values=parameter_values)
        snapshot()
        stop = crossed_stop(before_temperature, before_temperature_K=root_temperature)
        if stop is not None:
            return finish(stop)
    executed_entering_holds: list[ExecutableHold] = []
    held_indices: set[int] = set()
    stages = (
        tuple(item for item in active_holds if item.start_time < boundary),
        tuple(sorted((item for item in static if item.phase in {"ordinary", "before"}), key=_instant_canonical_order)),
        active_holds,
        tuple(sorted(trigger_instants, key=_instant_canonical_order)),
        tuple(sorted((item for item in static if item.phase == "after"), key=_instant_canonical_order)),
        active_holds,
    )
    trigger_by_declaration = {contract.triggers[index].declaration_ordinal: index for index in chemical_indices}
    for stage in stages:
        for action in stage:
            _raise_if_cancelled(cancellation_check)
            before = current.copy()
            original = action
            action = _resolve_items((action,), boundary, parameter_values, temperature_K=temperature_K)[0]
            if isinstance(action, ExecutableInstant):
                if stops:
                    validate_boundary_conflicts(instants=(*executed_instants, action), entering_holds=executed_entering_holds, cancellation_check=cancellation_check)
                if action.physical is not None:
                    if action.physical.op != "scale":
                        volume.require_volume(action, boundary)
                    current, action = volume.apply(action, current, time=boundary, temperature=temperature_K,
                        held_indices=held_indices)
                else:
                    current = _apply_instants(current, (action,), species_index=species_index, cancellation_check=cancellation_check, held_indices=held_indices)
                executed_instants.append(action)
                executed_values[original] = action
                if action.declaration_ordinal in trigger_by_declaration:
                    record(trigger_by_declaration[action.declaration_ordinal], action)
            else:
                if original in entering_holds and action not in executed_entering_holds:
                    if stops:
                        validate_boundary_conflicts(instants=executed_instants, entering_holds=(*executed_entering_holds, action), cancellation_check=cancellation_check)
                    executed_entering_holds.append(action)
                current = _apply_holds(current, (action,), species_index=species_index, cancellation_check=cancellation_check, time=boundary, temperature_K=temperature_K, parameter_values=parameter_values)
                held_indices.add(species_index[action.species])
            snapshot()
            if stops and not np.array_equal(before, current):
                # Validate the executed prefix before admitting its terminal state.
                invalid = (current < 0.0) & ~projectable_solver_negatives
                if np.any(invalid):
                    raise InterventionScheduleError("Intervention boundary produced a scientifically invalid negative continuation state.")
                stop = crossed_stop(before)
                if stop is not None:
                    return finish(stop)
    return finish()


def _instant_canonical_order(
    instant: ExecutableInstant,
) -> tuple[int, int, int, int]:
    return (
        int(instant.declaration_ordinal),
        {"ordinary": 0, "before": 1, "after": 2}[instant.phase],
        -1 if instant.window_ordinal is None else int(instant.window_ordinal),
        -1 if instant.authored_ordinal is None else int(instant.authored_ordinal),
    )


def _project_public_event_state(
    raw_event_state: Mapping[str, float],
) -> dict[str, float]:
    """Project one raw solver-root mapping before publishing a firing.

    The mapping is intentionally call-local: root crossing, action ordering,
    and continuation continue to use their raw state, while the returned
    mapping is a detached public pre-action snapshot.
    """

    names = tuple(str(name) for name in raw_event_state)
    raw_values = np.asarray(
        [float(raw_event_state[name]) for name in names],
        dtype=float,
    ).reshape(-1, 1)
    projected, _projection = project_nonnegative_trajectory(
        raw_values,
        context="Trigger public event state",
    )
    return {
        name: float(projected[index, 0])
        for index, name in enumerate(names)
    }


def _apply_instants(
    state: np.ndarray,
    instants: Sequence[ExecutableInstant],
    *,
    species_index: Mapping[str, int],
    cancellation_check: Callable[[], bool] | None,
    held_indices: set[int],
) -> np.ndarray:
    if not instants:
        return state
    output = np.asarray(state, dtype=float).reshape(-1).copy()
    for instant in instants:
        _raise_if_cancelled(cancellation_check)
        index = species_index[instant.species]
        if instant.op == "set":
            value = float(instant.value)
            if not math.isfinite(value) or value < 0.0:
                raise InterventionScheduleError(
                    "Intervention set produced an invalid state."
                )
            if index not in held_indices:
                output[index] = value
        elif instant.op == "jump":
            if index not in held_indices:
                output[index] = max(
                    0.0, float(output[index]) + float(instant.delta)
                )
        else:
            raise InterventionScheduleError(
                f"Unsupported executable instant op: {instant.op!r}."
            )
        _raise_if_cancelled(cancellation_check)
    if not np.all(np.isfinite(output)):
        raise InterventionScheduleError(
            "Intervention boundary produced an invalid state."
        )
    return output


def _resolve_items(items, time, parameter_values, *, temperature_K):
    roots = tuple(sorted({
        name
        for item in items
        for field in ("value", "delta", "flux_rate", "threshold")
        if (name := getattr(item, f"{field}_param", None)) is not None
    }))
    if not roots:
        return tuple(items)
    try:
        values = parameter_values(temperature_K, roots) if parameter_values is not None else {}
        return tuple(
            resolve_executable_values(item, values, time=time) for item in items
        )
    except Exception as exc:
        if isinstance(exc, SimulationCancelled):
            raise
        raise InterventionScheduleError(
            f"Intervention values at t={time:g}: {exc}", details={"time": time}
        ) from exc


def _apply_holds(
    state, holds, *, species_index, cancellation_check, time, temperature_K, parameter_values
):
    if not holds:
        return state
    resolved = _resolve_items(holds, time, parameter_values, temperature_K=temperature_K)
    output = np.asarray(state, dtype=float).reshape(-1).copy()
    for hold in resolved:
        _raise_if_cancelled(cancellation_check)
        value = float(hold.value)
        if not math.isfinite(value) or value < 0.0:
            raise InterventionScheduleError(
                "Intervention hold produced an invalid state."
            )
        output[species_index[hold.species]] = value
    return output


def _apply_holds_to_trajectory(
    states,
    holds,
    *,
    species_index,
    cancellation_check,
    times,
    temperatures,
    parameter_values,
):
    output = np.asarray(states, dtype=float).copy()
    if not holds:
        return output
    for column in range(output.shape[1]):
        output[:, column] = _apply_holds(
            output[:, column],
            holds,
            species_index=species_index,
            cancellation_check=cancellation_check,
            time=float(times[column]),
            temperature_K=float(temperatures[column]),
            parameter_values=parameter_values,
        )
    return output


def _selected_temperatures(times, temperature_at, cancellation_check):
    temperatures = np.empty(len(times), dtype=float)
    for index, time in enumerate(times):
        if index % 256 == 0:
            _raise_if_cancelled(cancellation_check)
        temperatures[index] = temperature_at(float(time))
    _raise_if_cancelled(cancellation_check)
    return temperatures


class _InterventionRhs:
    def __init__(
        self,
        rhs: ODERhs,
        volume: VolumeExecution,
        *,
        fluxes: Sequence[ExecutableFlux],
        holds: Sequence[ExecutableHold],
        species_index: Mapping[str, int],
        parameter_values,
        temperature_at,
        cancellation_check,
    ) -> None:
        self._rhs = rhs
        self.volume = volume
        self._fluxes = tuple(fluxes)
        self._holds = tuple(holds)
        self._species_index = dict(species_index)
        self._parameter_values = parameter_values
        self.temperature_at = temperature_at
        self.cancellation_check = cancellation_check

    def _flux_values(self, time, temperature_K):
        sums = {}
        negative = set()
        for flux in _resolve_items(
            self._fluxes, time, self._parameter_values, temperature_K=temperature_K
        ):
            if flux.physical is not None:
                continue
            index = self._species_index[flux.species]
            sums[index] = sums.get(index, 0.0) + float(flux.flux_rate)
            if flux.flux_rate < 0.0:
                negative.add(index)
        return sums, negative

    def _base_evaluation_state(
        self,
        state: np.ndarray,
        time: float,
        temperature_K: float,
    ) -> tuple[np.ndarray, set[int]]:
        raw = np.asarray(state, dtype=float).reshape(-1)
        evaluation_state = raw[:len(self._species_index)].copy()
        fixed: set[int] = set()
        evaluation_state = _apply_holds(
            evaluation_state,
            self._holds,
            species_index=self._species_index,
            cancellation_check=None,
            time=time,
            temperature_K=temperature_K,
            parameter_values=self._parameter_values,
        )
        fixed = {self._species_index[hold.species] for hold in self._holds}
        _sums, negative = self._flux_values(time, temperature_K)
        for index in negative:
            if evaluation_state[index] < 0.0:
                evaluation_state[index] = 0.0
        return evaluation_state, fixed

    def __call__(self, time: float, state: np.ndarray) -> np.ndarray:
        _raise_if_cancelled(self.cancellation_check)
        temperature_K = self.temperature_at(time)
        raw_state = np.asarray(state, dtype=float).reshape(-1)
        evaluation_state, fixed = self._base_evaluation_state(raw_state, time, temperature_K)
        derivative = np.asarray(
            self._rhs(time, evaluation_state, T=temperature_K), dtype=float
        ).copy()
        volume_rate = 0.0
        if self.volume.value is not None:
            vessel = raw_state[-1]
            volume_rate, mixing = self.volume.flow(self._fluxes, time, temperature_K, vessel, evaluation_state, fixed)
            derivative += mixing
        sums, _negative = self._flux_values(time, temperature_K)
        for index, rate in sums.items():
            derivative[index] += rate
        for index in fixed:
            derivative[index] = 0.0
        return np.append(derivative, volume_rate) if self.volume.value is not None else derivative

    def clamped_columns(self, time: float, state: np.ndarray, temperature_K: float) -> set[int]:
        """Drained species evaluated at zero past an unlocated root."""

        raw_state = np.asarray(state, dtype=float).reshape(-1)
        _sums, negative = self._flux_values(time, temperature_K)
        return {index for index in negative if raw_state[index] < 0.0}


def _drain_release_events(
    free_rhs: _InterventionRhs,
    *,
    active_holds: Sequence[ExecutableHold],
    species_index: Mapping[str, int],
    state: np.ndarray,
    segment_start: float,
    released: frozenset[int],
) -> tuple[tuple[Callable[[float, np.ndarray], float], ...], tuple[int, ...]]:
    """Hold drained species at zero until their free derivative turns positive.

    Only a strictly negative free derivative holds a species, and a species
    just released at ``segment_start`` is not held again there, so every
    release makes progress.
    """

    values = np.asarray(state, dtype=float).reshape(-1)
    sums, _negative = free_rhs._flux_values(segment_start, free_rhs.temperature_at(segment_start))
    # An authored hold governs its species, as for purge events.
    held = {species_index[hold.species] for hold in active_holds}
    zero = sorted(
        index for index in sums
        if index not in held and index not in released and values[index] <= 0.0
    )
    if not zero:
        return (), ()
    vessel = () if free_rhs.volume.value is None else (free_rhs.volume.value,)
    free = free_rhs(segment_start, np.append(values, vessel))
    indices = tuple(index for index in zero if free[index] < 0.0)
    if not indices:
        return (), ()
    last: list[object] = [None, None]

    def free_derivative(time: float, state: np.ndarray) -> np.ndarray:
        key = (time, state.tobytes())
        if last[0] != key:
            held = np.array(state, dtype=float)
            held[list(indices)] = 0.0
            last[0], last[1] = key, free_rhs(time, held)
        return last[1]

    # A non-positive free derivative reads as -1, so a release fires only when
    # the derivative turns positive and never at the segment start.
    events = []
    for index in indices:
        def release(time: float, state: np.ndarray, *, _index: int = index) -> float:
            rate = float(free_derivative(float(time), state)[_index])
            return rate if rate > 0.0 else -1.0

        release.terminal = True  # type: ignore[attr-defined]
        release.direction = 1.0  # type: ignore[attr-defined]
        events.append(release)
    return tuple(events), indices


class _InterventionJacobian:
    """Exact Jacobian transform for holds, fluxes, and purge branches."""

    def __init__(
        self,
        jacobian: Callable[..., np.ndarray],
        *,
        rhs: _InterventionRhs,
        holds: Sequence[ExecutableHold],
        fluxes: Sequence[ExecutableFlux],
        species_index: Mapping[str, int],
    ) -> None:
        self._jacobian = jacobian
        self._rhs = rhs
        self._holds = tuple(holds)
        self._fluxes = tuple(fluxes)
        self._species_index = dict(species_index)

    def _evaluate_state(self, state: np.ndarray, time: float, temperature_K: float) -> tuple[np.ndarray, set[int]]:
        values = np.asarray(state, dtype=float).reshape(-1)
        held = {
            self._species_index[hold.species]
            for hold in self._holds
        }
        used, _fixed = self._rhs._base_evaluation_state(values, time, temperature_K)
        return used, held

    def __call__(self, time: float, state: np.ndarray) -> np.ndarray:
        _raise_if_cancelled(self._rhs.cancellation_check)
        temperature_K = self._rhs.temperature_at(time)
        used, held = self._evaluate_state(state, time, temperature_K)
        matrix = np.asarray(self._jacobian(time, used, T=temperature_K), dtype=float).copy()
        size = used.size
        if matrix.shape != (size, size):
            raise ValueError(
                f"intervention Jacobian expected {(size, size)}, got {matrix.shape}"
            )
        if self._rhs.volume.value is not None:
            vessel = float(state[-1])
            rate, mixing = self._rhs.volume.flow(self._fluxes, time, temperature_K, vessel, used, held)
            matrix -= np.eye(size) * (rate / vessel)
            coupled = np.zeros((size + 1, size + 1))
            coupled[:size, :size] = matrix
            coupled[:size, -1] = -mixing / vessel
            matrix = coupled
        if held:
            indices = np.fromiter(sorted(held), dtype=np.int64)
            matrix[indices, :] = 0.0
            matrix[:, indices] = 0.0
        purge_columns = self._rhs.clamped_columns(float(time), state, temperature_K)
        if purge_columns:
            matrix[:, np.fromiter(sorted(purge_columns), dtype=np.int64)] = 0.0
        return matrix

def _trigger_direction(direction: str) -> float:
    return {"rising": 1.0, "falling": -1.0, "either": 0.0}[direction]


def _next_eligible_time(
    trigger: ExecutableTrigger,
    *,
    index: int,
    counts: Sequence[int],
    last_times: Sequence[float | None],
) -> float | None:
    if isinstance(trigger.then, InterventionStopAction):
        return -math.inf
    if counts[index] >= trigger.max_count:
        return None
    last = last_times[index]
    if last is None:
        return -math.inf
    added = float(last) + trigger.min_interval
    if not math.isfinite(added):
        raise InterventionScheduleError(
            "Trigger rearm arithmetic must remain finite."
        )
    return max(added, float(np.nextafter(float(last), np.inf)))


def _trigger_callables(
    triggers: Sequence[ExecutableTrigger],
    *,
    species_index: Mapping[str, int],
    segment_start: float,
    segment_state: np.ndarray,
    trigger_counts: Sequence[int],
    trigger_last_times: Sequence[float | None],
    observable_value: Callable[[str, float, np.ndarray, float], float] | None = None,
    temperature_at,
    parameter_values=None, holds=(),
) -> tuple[tuple[Callable[[float, np.ndarray], float], ...], tuple[int, ...]]:
    callables: list[Callable[[float, np.ndarray], float]] = []
    mapping: list[int] = []
    for index, trigger in enumerate(triggers):
        ready = _next_eligible_time(
            trigger,
            index=index,
            counts=trigger_counts,
            last_times=trigger_last_times,
        )
        if ready is None or segment_start < ready:
            continue
        initial_temperature = temperature_at(segment_start)
        threshold = _resolve_items((trigger,), segment_start, parameter_values, temperature_K=initial_temperature)[0].threshold
        starts_at_threshold = _trigger_value(trigger, segment_start, _apply_holds(segment_state, holds, species_index=species_index, cancellation_check=None, time=segment_start, temperature_K=initial_temperature, parameter_values=parameter_values), species_index=species_index, observable_value=observable_value, temperature_K=initial_temperature) == threshold
        directions = (
            ("rising", "falling")
            if trigger.direction == "either"
            else (trigger.direction,)
        )
        for direction in directions:
            # Initial equality is neutral: it is neither side of a crossing.
            # At the immutable segment start, report the target side's sign so
            # the first departure can only be neutral or look like the
            # opposite direction, which SciPy ignores.  Every later evaluation
            # is the pure threshold function. A target that remains exactly
            # at its initial threshold stays neutral until departure.
            # Thus analytic-Jacobian retry and a
            # fresh solve observe identical event semantics.  ``either`` uses
            # one rising and one falling event so genuine later crossings in
            # both directions remain observable.
            neutral_value = 1.0 if direction == "rising" else -1.0

            def event(
                time: float,
                state: np.ndarray,
                *,
                _trigger: ExecutableTrigger = trigger,
                _threshold: float = threshold,
                _segment_start: float = float(segment_start),
                _starts_at_threshold: bool = starts_at_threshold,
                _neutral_value: float = neutral_value,
            ) -> float:
                temperature_K = temperature_at(time)
                state = _apply_holds(state, holds, species_index=species_index, cancellation_check=None, time=time, temperature_K=temperature_K, parameter_values=parameter_values)
                threshold = _resolve_items((_trigger,), time, parameter_values, temperature_K=temperature_K)[0].threshold
                value = _trigger_value(_trigger, time, state, species_index=species_index, observable_value=observable_value, temperature_K=temperature_K) - threshold
                if _starts_at_threshold and (float(time) == _segment_start or value == 0.0):
                    return _neutral_value
                return value

            event.terminal = True  # type: ignore[attr-defined]
            event.direction = _trigger_direction(direction)  # type: ignore[attr-defined]
            callables.append(event)
            mapping.append(index)
    return tuple(callables), tuple(mapping)


def _next_rearm_boundary(
    triggers: Sequence[ExecutableTrigger],
    *,
    segment_start: float,
    segment_end: float,
    trigger_counts: Sequence[int],
    trigger_last_times: Sequence[float | None],
) -> float | None:
    candidates: list[float] = []
    for index, trigger in enumerate(triggers):
        ready = _next_eligible_time(
            trigger,
            index=index,
            counts=trigger_counts,
            last_times=trigger_last_times,
        )
        if (
            ready is not None
            and math.isfinite(ready)
            and segment_start < ready < segment_end
        ):
            candidates.append(ready)
    return min(candidates) if candidates else None


def _first_trigger_hit(
    event_times: Sequence[Sequence[float]],
    event_states: Sequence[Sequence[np.ndarray]],
    *,
    trigger_mapping: Sequence[int],
) -> tuple[float, tuple[int, ...], np.ndarray] | None:
    hits: list[tuple[float, int, np.ndarray]] = []
    for offset, trigger_index in enumerate(trigger_mapping):
        event_index = offset
        if event_index >= len(event_times) or event_index >= len(event_states):
            continue
        for position, raw_time in enumerate(event_times[event_index]):
            if position >= len(event_states[event_index]):
                raise InterventionScheduleError(
                    "Exact trigger root is missing its event-time state."
                )
            hits.append(
                (
                    float(raw_time),
                    trigger_index,
                    np.asarray(
                        event_states[event_index][position], dtype=float
                    ).reshape(-1),
                )
            )
    if not hits:
        return None
    root_time = min(item[0] for item in hits)
    simultaneous = [item for item in hits if item[0] == root_time]
    indices = tuple(
        dict.fromkeys(
            item[1]
            for item in sorted(simultaneous, key=lambda item: item[1])
        )
    )
    state = simultaneous[0][2]
    if any(not np.array_equal(state, item[2]) for item in simultaneous[1:]):
        raise InterventionScheduleError(
            "Simultaneous exact roots reported incompatible event-time states."
        )
    return root_time, indices, state


def _first_purge_hit(
    event_times: Sequence[Sequence[float]],
    event_states: Sequence[Sequence[np.ndarray]],
    *,
    purge_mapping: Sequence[int],
    event_offset: int,
) -> tuple[float, tuple[int, ...], np.ndarray] | None:
    hits: list[tuple[float, int, np.ndarray]] = []
    for offset, species_index in enumerate(purge_mapping):
        event_index = event_offset + offset
        if event_index >= len(event_times) or event_index >= len(event_states):
            continue
        for position, raw_time in enumerate(event_times[event_index]):
            if position >= len(event_states[event_index]):
                raise InterventionScheduleError(
                    "Exact purge root is missing its event-time state."
                )
            hits.append(
                (
                    float(raw_time),
                    int(species_index),
                    np.asarray(
                        event_states[event_index][position], dtype=float
                    ).reshape(-1),
                )
            )
    if not hits:
        return None
    root_time = min(item[0] for item in hits)
    simultaneous = [item for item in hits if item[0] == root_time]
    state = simultaneous[0][2].copy()
    if any(item[2].size != state.size for item in simultaneous):
        raise InterventionScheduleError(
            "Simultaneous exact purge roots reported incompatible states."
        )
    for _time, species_index, _state in simultaneous:
        state[species_index] = 0.0
    return (
        root_time,
        tuple(dict.fromkeys(item[1] for item in simultaneous)),
        state,
    )


def _requested_eval(
    values: np.ndarray, *, start: float, end: float, is_final: bool
) -> np.ndarray:
    if is_final:
        mask = (values >= start) & (values <= end)
    else:
        mask = (values >= start) & (values < end)
    selected = np.asarray(values[mask], dtype=float).reshape(-1).copy()
    if selected.size > 1:
        selected = selected[
            np.concatenate([np.array([True]), np.diff(selected) > 0.0])
        ]
    return selected


def _append_identity_samples(
    output_times: list[np.ndarray],
    output_states: list[np.ndarray],
    output_segment_ids: list[np.ndarray],
    output_segment_keys: list[tuple[str, ...]],
    output_discontinuity_ids: list[np.ndarray],
    times: np.ndarray,
    states: np.ndarray,
    *,
    output_temperatures: list[np.ndarray],
    temperatures: np.ndarray,
    segment_id: int,
    segment_key: str,
    discontinuity_id: int,
) -> None:
    emit_times = np.asarray(times, dtype=float).reshape(-1)
    emit_states = np.asarray(states, dtype=float)
    emit_temperatures = np.asarray(temperatures, dtype=float)
    if not emit_times.size:
        return
    if (emit_states.ndim != 2 or emit_states.shape[1] != emit_times.size
            or emit_temperatures.shape != emit_times.shape):
        raise InterventionScheduleError(
            "Direct result rows must have one state column per emitted time."
        )
    if np.any(~np.isfinite(emit_times)) or (
        emit_times.size > 1 and np.any(np.diff(emit_times) <= 0.0)
    ):
        raise InterventionScheduleError(
            "Direct result row times must be finite and strictly increasing."
        )
    if output_times and output_times[-1].size:
        last_time = float(output_times[-1][-1])
        if emit_times[0] == last_time:
            emit_times = emit_times[1:]
            emit_states = emit_states[:, 1:]
            emit_temperatures = emit_temperatures[1:]
        elif emit_times[0] < last_time:
            raise InterventionScheduleError(
                "Direct result row emission must remain strictly ordered."
            )
    if not emit_times.size:
        return
    output_times.append(emit_times)
    output_states.append(emit_states)
    output_temperatures.append(emit_temperatures)
    output_segment_ids.append(
        np.full(emit_times.size, int(segment_id), dtype=np.int64)
    )
    output_segment_keys.append(tuple(str(segment_key) for _ in emit_times))
    output_discontinuity_ids.append(
        np.full(emit_times.size, int(discontinuity_id), dtype=np.int64)
    )


def _append_identity_row(
    output_times: list[np.ndarray],
    output_states: list[np.ndarray],
    output_segment_ids: list[np.ndarray],
    output_segment_keys: list[tuple[str, ...]],
    output_discontinuity_ids: list[np.ndarray],
    time: float,
    state: np.ndarray,
    *,
    output_temperatures: list[np.ndarray],
    temperature_K: float,
    segment_id: int,
    segment_key: str,
    discontinuity_id: int,
) -> None:
    _append_identity_samples(
        output_times,
        output_states,
        output_segment_ids,
        output_segment_keys,
        output_discontinuity_ids,
        np.array([float(time)], dtype=float),
        np.asarray(state, dtype=float).reshape(-1, 1),
        output_temperatures=output_temperatures,
        temperatures=np.array([temperature_K], dtype=float),
        segment_id=segment_id,
        segment_key=segment_key,
        discontinuity_id=discontinuity_id,
    )
