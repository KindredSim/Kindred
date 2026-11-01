# SPDX-License-Identifier: MIT
from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, replace
from operator import attrgetter
from typing import Callable, Mapping, Protocol, Sequence

import numpy as np

from kindred.core.algebra.errors import AlgebraError
from kindred.core.exceptions import SimulationCancelled, ValueDomainError
from kindred.core.intervention_schedule import (
    InterventionScheduleError, InterventionStopAction, SpeciesTriggerCondition, locate_declaration,
)
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
from kindred.core.temperature import TemperatureSegment, starting_temperature_segment
from .volume_execution import VolumeExecution
from kindred.core.volume import SimulationBoundaryState
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
from kindred.core.time_grid import build_time_grid


@dataclass(frozen=True, slots=True)
class SegmentIdentity:
    kind: str
    ordinal: int
    t_start: float
    t_end: float

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
    # Each species' integral of concentration minus this segment's end state.
    species_ctc: np.ndarray | None = None
    # The solver's accepted step ends, on the absolute clock.
    step_times: np.ndarray | None = None

    def __post_init__(self) -> None:
        # The solver adapter builds aligned rows and projects them nonnegative.
        t = np.asarray(self.t, dtype=float)
        Y = np.asarray(self.Y, dtype=float)
        t.setflags(write=False)
        Y.setflags(write=False)
        object.__setattr__(self, "t", t)
        object.__setattr__(self, "Y", Y)
        state = self.raw_terminal_state
        if state is None:
            return
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
        t0, t1 = req.t_span
        t_eval = request.t_eval
        contract = req.intervention_execution
        has_stops = any(isinstance(item.then, InterventionStopAction) for item in contract.triggers)
        dense_segments: list[tuple[SegmentIdentity, Callable[[np.ndarray], np.ndarray], tuple[ExecutableHold, ...], int, TemperatureSegment]] = []
        segment_result = None
        dense = None
        try:
            species_index = _species_index(req)
            volume = VolumeExecution(species_names=species_index, parameter_values=req.parameter_values_at_temperature,
                                     end_time=t1, lines=req.intervention_lines)
            current_y = req.y0.copy()
            cancellation_check = req.cancellation_control.cancel_requested
            # Each trigger's latest firing: its count, time and watched value.
            trigger_last_firings: list[TriggerFiring | None] = [None for _ in contract.triggers]
            trigger_firings: list[TriggerFiring] = []
            executed_values: dict[ExecutableInstant, ExecutableInstant] = {}
            outputs_t: list[np.ndarray] = []
            outputs_y: list[np.ndarray] = []
            outputs_temperature: list[np.ndarray] = []
            output_segment_ids: list[np.ndarray] = []
            output_discontinuity_ids: list[np.ndarray] = []
            presentation_boundaries: list[SimulationPresentationBoundary] = []
            # (length, end state, CTC from that end state) of each segment.
            ctc_segments: list[tuple[float, np.ndarray, np.ndarray]] | None = (
                [] if req.species_ctc else None
            )
            # Each species' largest published concentration so far and its time.
            maximum: tuple[np.ndarray, np.ndarray] | None = (
                (np.full(len(req.y0), -np.inf), np.zeros(len(req.y0))) if req.species_maximum else None
            )
            integration_segments: list[dict[str, object]] = []
            segment_projections: dict[str, SegmentProjection] = {}
            segment_count = 0
            discontinuity_id = 0
            # The executor carries the temperature: each segment runs on the
            # curve in force when it starts, and a change executes at its
            # boundary like any intervention.
            temperature = starting_temperature_segment(
                contract.temperature_changes, req.reference_temperature_K, t0,
            )
            temperature_changes = {
                change.time: change
                for change in contract.temperature_changes
                if t0 < change.time <= t1
            }

            def changed_temperature(time: float) -> float | None:
                nonlocal temperature
                change = temperature_changes.get(time)
                if change is None:
                    return None
                temperature = temperature.after(change)
                return temperature(time)

            boundaries = {
                t0,
                t1,
                *(
                    value
                    for value in contract.discontinuities
                    if t0 <= float(value) <= t1
                ),
                *temperature_changes,
            }
            # Admitted times are finite and bounded; the filter above drops
            # anything outside the span.
            ordered_boundaries = sorted(boundaries)

            _raise_if_cancelled(cancellation_check)
            current_temperature = temperature(t0)
            if contract.initial_volume is not None:
                volume.initialize(contract.initial_volume, t0, current_temperature)
            initial_state_transaction = t0 in contract.discontinuities
            initial_boundary = _apply_boundary(
                current_y,
                boundary=t0,
                contract=contract,
                trigger_indices=(),
                trigger_state=None,
                trigger_last_firings=trigger_last_firings,
                trigger_firings=trigger_firings,
                species_index=species_index,
                cancellation_check=cancellation_check,
                observable_value=req.stop_observable_value,
            parameter_values=req.parameter_values_at_temperature, executed_values=executed_values, volume=volume,
                                   temperature_K=current_temperature,
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
                output_discontinuity_ids,
                t0,
                current_y,
                segment_id=0,
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
                sub_start = outer_start
                while sub_start < outer_end and termination is None:
                    _raise_if_cancelled(cancellation_check)
                    next_rearm = _next_rearm_boundary(
                        contract.triggers,
                        segment_start=sub_start,
                        segment_end=outer_end,
                        trigger_last_firings=trigger_last_firings,
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

                    segment_temperature = temperature

                    # A drained species at zero whose free derivative is not
                    # positive is held at zero like an authored hold; a
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
                        trigger_last_firings=trigger_last_firings,
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
                                    rhs=segment_rhs,
                                ),
                            ),
                        )
                    requested = _requested_eval(
                        t_eval,
                        start=sub_start,
                        end=sub_end,
                        is_final=(sub_end == t1 and sub_end not in temperature_changes),
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
                    if ctc_segments is not None:
                        ctc_segments.append(
                            (sub_end - sub_start, segment_result.Y[:, -1], segment_result.species_ctc)
                        )
                    event_times = segment_result.event_times
                    event_states = segment_result.raw_event_states
                    trigger_hit = _first_hit(
                        event_times, event_states, mapping=trigger_mapping,
                    )
                    purge_hit = _first_hit(
                        event_times, event_states, mapping=purge_mapping,
                        event_offset=len(trigger_observers),
                    )
                    # A located zero places its drained species exactly at zero.
                    if purge_hit is not None:
                        purge_hit[2][list(purge_hit[1])] = 0.0
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
                    seg_temperature = segment_temperature.values(seg_t)
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
                        dense_segments.append((segment_output_identity, segment_result.dense_output, active_holds, discontinuity_id, segment_temperature))
                    if maximum is not None:
                        # Executed bounds: SciPy's last step can extend past a root.
                        _fold_maximum(maximum, _segment_maximum(
                            segment_result, sub_start, actual_segment_end, active_holds, segment_temperature,
                            species_index=species_index, parameter_values=req.parameter_values_at_temperature,
                            cancellation_check=cancellation_check,
                        ))
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
                        "starts_at_temperature_change": (
                            segment_count > 0
                            and sub_start in temperature_changes
                        ),
                        "symbolic_jacobian_used": bool(
                            segment_result.symbolic_jacobian_used
                        ),
                    }
                    integration_segments.append(segment_payload)
                    segment_count += 1
                    current_temperature = segment_temperature(actual_segment_end)
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
                            context=f"Purge continuation state at t = {purge_hit[0]:g} s",
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
                    # Rows before the segment end (requested times, and any
                    # solver steps); the end row only when it was requested.
                    emit_mask = (seg_t < sub_end) | (
                        (seg_t == sub_end) & bool(requested.size and requested[-1] == sub_end)
                    )
                    emit_t = seg_t[emit_mask]
                    emit_y = seg_y[:, emit_mask]
                    emit_temperature = seg_temperature[emit_mask]
                    # A root or a scheduled discontinuity emits its own row.
                    if trigger_hit is not None or purge_hit is not None or scheduled_discontinuity:
                        emit_mask = emit_t < actual_segment_end
                        emit_t = emit_t[emit_mask]
                        emit_y = emit_y[:, emit_mask]
                        emit_temperature = emit_temperature[emit_mask]
                    _append_identity_samples(
                        outputs_t,
                        outputs_y,
                        output_segment_ids,
                        output_discontinuity_ids,
                        emit_t,
                        emit_y,
                        segment_id=segment_identity.ordinal,
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
                                output_discontinuity_ids,
                                float(purge_hit[0]),
                                np.asarray(purge_hit[2], dtype=float).reshape(-1),
                                segment_id=segment_count,
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
                                    output_discontinuity_ids,
                                    float(purge_hit[0]),
                                    np.asarray(purge_hit[2], dtype=float).reshape(-1),
                                    segment_id=segment_count,
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
                                trigger_last_firings=trigger_last_firings,
                                trigger_firings=trigger_firings,
                                species_index=species_index,
                                cancellation_check=cancellation_check,
                                observable_value=req.stop_observable_value,
                            parameter_values=req.parameter_values_at_temperature, executed_values=executed_values, volume=volume,
                                                  temperature_K=current_temperature,
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
                                trigger_last_firings=trigger_last_firings,
                                trigger_firings=trigger_firings,
                                species_index=species_index,
                                cancellation_check=cancellation_check,
                                observable_value=req.stop_observable_value,
                                changed_temperature_K=changed_temperature(outer_end),
                            parameter_values=req.parameter_values_at_temperature, executed_values=executed_values, volume=volume,
                                                  temperature_K=current_temperature,
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
                                    output_discontinuity_ids,
                                    float(outer_end),
                                    np.asarray(current_y, dtype=float).reshape(-1),
                                    segment_id=post_segment_id,
                                    discontinuity_id=discontinuity_id,
                                    output_temperatures=outputs_temperature,
                                    temperature_K=current_temperature,
                                )
                                segment_starts_after_discontinuity = outer_end < t1
                            elif (
                                outer_end in temperature_changes
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
                                    output_discontinuity_ids,
                                    float(outer_end),
                                    np.asarray(current_y, dtype=float).reshape(-1),
                                    segment_id=post_segment_id,
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
                                    output_discontinuity_ids,
                                    float(outer_end),
                                    np.asarray(current_y, dtype=float).reshape(-1),
                                    segment_id=segment_output_identity.ordinal,
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
                        trigger_last_firings=trigger_last_firings,
                        trigger_firings=trigger_firings,
                        species_index=species_index,
                        cancellation_check=cancellation_check,
                        observable_value=req.stop_observable_value,
                        changed_temperature_K=changed_temperature(root_time),
                    parameter_values=req.parameter_values_at_temperature, executed_values=executed_values, volume=volume,
                                          temperature_K=current_temperature,
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
                        output_discontinuity_ids,
                        float(root_time),
                        np.asarray(current_y, dtype=float).reshape(-1),
                        segment_id=post_segment_id,
                        discontinuity_id=discontinuity_id,
                        output_temperatures=outputs_temperature,
                        temperature_K=current_temperature,
                    )
                    segment_starts_after_discontinuity = root_time < t1
                    sub_start = root_time

            actual_end = t1 if termination is None else termination.actual_endpoint
            if termination is None:
                termination = SimulationTermination(t1, "endpoint")
            # The starting row is always emitted first.
            output_t = np.concatenate(outputs_t)
            output_y = np.concatenate(outputs_y, axis=1)
            output_temperature = np.concatenate(outputs_temperature)
            if float(output_t[-1]) != actual_end:
                _append_identity_row(
                    outputs_t,
                    outputs_y,
                    output_segment_ids,
                    output_discontinuity_ids,
                    actual_end,
                    np.asarray(current_y, dtype=float).reshape(-1),
                    segment_id=max(0, segment_count - 1),
                    discontinuity_id=discontinuity_id,
                    output_temperatures=outputs_temperature,
                    temperature_K=current_temperature,
                )
                output_t = np.concatenate(outputs_t)
                output_y = np.concatenate(outputs_y, axis=1)
                output_temperature = np.concatenate(outputs_temperature)
            if has_stops:
                try:
                    base_times = (np.array([t0], dtype=float) if actual_end == t0
                                  else build_time_grid(t0, actual_end, req.grid))
                except ValueError as exc:
                    raise InterventionScheduleError(str(exc)) from exc
                boundary_ids, boundary_discontinuities = _compose_direct_result_sample_identity(
                    output_segment_ids, output_discontinuity_ids,
                )
                final_times = np.union1d(base_times, output_t)
                final_states = np.empty((current_y.size, final_times.size), dtype=float)
                final_temperatures = np.empty(final_times.size, dtype=float)
                final_ids = np.empty(final_times.size, dtype=np.int64)
                final_discontinuities = np.empty(final_times.size, dtype=np.int64)

                # Exact executor rows own intervention and termination boundaries.
                boundary_rows = np.searchsorted(final_times, output_t)
                final_states[:, boundary_rows] = output_y
                final_temperatures[boundary_rows] = output_temperature
                final_ids[boundary_rows] = boundary_ids
                final_discontinuities[boundary_rows] = boundary_discontinuities
                needs_sampling = np.ones(final_times.size, dtype=bool)
                needs_sampling[boundary_rows] = False
                sample_rows = np.flatnonzero(needs_sampling)
                sample_times = final_times[sample_rows]
                next_sample = 0

                for dense_index, (identity_record, dense, holds, discontinuity, dense_temperature) in enumerate(dense_segments):
                    _raise_if_cancelled(cancellation_check)
                    # An internal endpoint belongs to the following segment. Use
                    # executed bounds: SciPy's last step can extend past a root.
                    sample_end = int(np.searchsorted(
                        sample_times, identity_record.t_end,
                        side="right" if dense_index + 1 == len(dense_segments) else "left",
                    ))
                    if sample_end == next_sample:
                        continue
                    projected_count = 0
                    max_negative_magnitude = 0.0
                    # Bound interpolation temporaries and check cancellation on
                    # both sides of every group, including a short final group.
                    for sample_start in range(next_sample, sample_end, 256):
                        _raise_if_cancelled(cancellation_check)
                        chunk_end = min(sample_start + 256, sample_end)
                        rows = sample_rows[sample_start:chunk_end]
                        selected_times = sample_times[sample_start:chunk_end]
                        selected_temperatures = dense_temperature.values(selected_times)
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
                # The stored segments are contiguous from t0 and cover every sample.
                _raise_if_cancelled(cancellation_check)
                output_t, output_y = final_times, final_states
                output_temperature = final_temperatures
                output_segment_ids, output_discontinuity_ids = [final_ids], [final_discontinuities]
                dense_segments.clear()
            # Every emitted block increases and starts after the previous row.
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
                result_discontinuity_ids,
            ) = _compose_direct_result_sample_identity(
                output_segment_ids,
                output_discontinuity_ids,
            )
            _publish_segment_projection_provenance(
                provenance,
                segment_projections,
            )
            species_ctc = None
            if ctc_segments is not None:
                # Without interventions concentrations are continuous across
                # segments; rebase each segment's CTC on the run's final state.
                final_state = output_y[:, -1]
                species_ctc = sum(
                    ctc + length * (end - final_state)
                    for length, end, ctc in ctc_segments
                )
            if maximum is not None:
                # An instantaneous action's states before and after it count too.
                for boundary in presentation_boundaries:
                    for state in boundary.states:
                        values = np.full(len(req.y0), -np.inf)
                        for name, value in state.concentrations:
                            values[species_index[name]] = value
                        _fold_maximum(maximum, (values, np.full(values.size, boundary.time)))
            _raise_if_cancelled(cancellation_check)
            return FinalSimulationOutput(
                t=output_t,
                Y=output_y,
                volume=volume.trajectory(output_t),
                species_ctc=species_ctc,
                species_maximum=maximum,
                temperature_K=output_temperature,
                provenance=provenance,
                intervention_execution=final_contract,
                presentation_boundaries=tuple(presentation_boundaries),
                segment_ids=result_segment_ids,
                discontinuity_ids=result_discontinuity_ids,
                termination=termination,
            )
        except (ValueDomainError, InterventionScheduleError) as exc:
            # Values params supply at run time, and conflicts among them, are
            # refused on their declaration.
            locate_declaration(exc, req.intervention_lines)
            raise
        finally:
            dense_segments.clear()
            segment_result = None
            dense = None


# A golden-section step keeps 1/phi of the bracket, so 12 steps shrink each
# bracket to about 0.3% of its width.  The maximum's value is then limited by
# solver tolerances alone, and its time lies well inside the span over which
# the curve stays within solver tolerance of that value, so finer time steps
# would not make it more meaningful.
_INVERSE_GOLDEN = (math.sqrt(5.0) - 1.0) / 2.0
_GOLDEN_STEPS = 12


def _segment_maximum(
    segment_result: SegmentRunResult,
    start: float,
    end: float,
    holds: tuple[ExecutableHold, ...],
    temperature: TemperatureSegment,
    *,
    species_index: Mapping[str, int],
    parameter_values: Callable[[float, tuple[str, ...]], Mapping[str, float]],
    cancellation_check: Callable[[], bool] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Each species' largest published concentration on one continuous segment
    and its time.

    The solver's step ends locate each maximum and its step interpolant refines
    it, so solver tolerances, not output points, control its accuracy.  Holds
    and the nonnegative projection apply as they do to published rows.
    """

    def states(times: np.ndarray) -> np.ndarray:
        held = _apply_holds_to_trajectory(
            segment_result.dense_output(times), holds, species_index=species_index,
            cancellation_check=cancellation_check, times=times,
            temperatures=temperature.values(times), parameter_values=parameter_values,
        )
        return project_nonnegative_trajectory(held, context="Maximum")[0]

    steps = segment_result.step_times
    times = np.unique(np.concatenate(([start, end], steps[(steps > start) & (steps < end)])))
    values = states(times)
    index = np.argmax(values, axis=1)
    best = values[np.arange(values.shape[0]), index]
    best_times = times[index]
    # The maximum lies within the steps on either side of the largest step end.
    low = times[np.maximum(index - 1, 0)]
    width = times[np.minimum(index + 1, times.size - 1)] - low
    searched = np.flatnonzero(width > 0.0)
    if not searched.size:
        return best, best_times
    low, width = low[searched], width[searched]
    columns = np.arange(searched.size)

    def own(offsets: np.ndarray) -> np.ndarray:
        # Each searched species at its own candidate time.  Offsets from
        # ``low`` keep a late clock from coarsening the search.
        return states(low + offsets)[searched, columns]

    # One golden-section search per species, advanced together, so each step
    # evaluates the interpolant once for every species.
    a, b = np.zeros(searched.size), width
    c, d = b - _INVERSE_GOLDEN * b, _INVERSE_GOLDEN * b
    fc, fd = own(c), own(d)
    for _ in range(_GOLDEN_STEPS):
        _raise_if_cancelled(cancellation_check)
        left = fc > fd
        a, b = np.where(left, a, c), np.where(left, d, b)
        c, d = np.where(left, b - _INVERSE_GOLDEN * (b - a), d), np.where(left, c, a + _INVERSE_GOLDEN * (b - a))
        found = own(np.where(left, c, d))
        fc, fd = np.where(left, found, fd), np.where(left, fc, found)
    found = np.maximum(fc, fd)
    better = found > best[searched]
    best[searched[better]] = found[better]
    best_times[searched[better]] = (low + np.where(fc >= fd, c, d))[better]
    return best, best_times


def _fold_maximum(
    maximum: tuple[np.ndarray, np.ndarray], candidate: tuple[np.ndarray, np.ndarray],
) -> None:
    """Keep each species' larger value; a tie keeps the earlier time."""

    larger = candidate[0] > maximum[0]
    maximum[0][larger] = candidate[0][larger]
    maximum[1][larger] = candidate[1][larger]


def _append_presentation_boundary(
    boundaries: list[SimulationPresentationBoundary],
    boundary: SimulationPresentationBoundary | None,
) -> None:
    if boundary is None:
        return
    # Boundaries apply at t0, then at strictly later ends or roots.
    boundaries.append(boundary)


def _raise_if_cancelled(check: Callable[[], bool] | None) -> None:
    if check is not None and check():
        raise SimulationCancelled()


def _compose_direct_result_sample_identity(
    segment_ids: Sequence[np.ndarray],
    discontinuity_ids: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Compose row identity captured at emission time.

    Identity is captured while each unique segment/root row is emitted, so no
    downstream boundary reconstructs execution authority from timestamps.
    """

    ids = np.concatenate(segment_ids).astype(np.int64, copy=False)
    discontinuities = np.concatenate(discontinuity_ids).astype(np.int64, copy=False)
    ids.setflags(write=False)
    discontinuities.setflags(write=False)
    return ids, discontinuities


def _record_segment_projection(
    records: dict[str, SegmentProjection],
    result: SegmentRunResult,
    *,
    identity: SegmentIdentity,
) -> None:
    records[identity.key] = SegmentProjection(
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
    records[identity.key] = records[identity.key].with_continuation(projection)


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


def _species_index(req: SimulationRequest) -> dict[str, int]:
    # Schedule admission checked every referenced species against the mechanism.
    return {name: index for index, name in enumerate(req.species_names)}


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
    segment_start: float,
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
    if isinstance(trigger.condition, SpeciesTriggerCondition):
        return float(state[species_index[trigger.condition.name]])
    try:
        return float(observable_value(trigger.condition.name, time, state, temperature_K))
    except (AlgebraError, ValueDomainError) as exc:
        # The value that failed already names its inputs.
        exc.annotate(
            prefix=f"Stopping declaration {trigger.declaration_ordinal + 1}, quantity {trigger.condition.name!r}: ",
            quantity=trigger.condition.name, time=time,
        )
        raise


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
    trigger_last_firings: list[TriggerFiring | None],
    trigger_firings: list[TriggerFiring],
    species_index: Mapping[str, int],
    cancellation_check: Callable[[], bool] | None,
    observable_value: Callable[[str, float, np.ndarray, float], float] | None,
    changed_temperature_K: float | None = None,
    temperature_K: float,
    parameter_values, executed_values: dict, volume,
) -> _BoundaryApplicationResult:
    _raise_if_cancelled(cancellation_check)
    current = np.asarray(trigger_state if trigger_state is not None else state, dtype=float).reshape(-1).copy()
    exact_root = current.copy()
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
        last = trigger_last_firings[index]
        action_payload = trigger.then.to_payload() if action is None else action.to_payload()
        firing = TriggerFiring(
            declaration_ordinal=trigger.declaration_ordinal, condition=trigger.condition,
            threshold=_resolve_items((trigger,), boundary, parameter_values, temperature_K=root_temperature if index in trigger_indices else temperature_K)[0].threshold, direction=trigger.direction, root_time=boundary,
            event_state=_project_public_event_state({name: float((current if isinstance(trigger.then, InterventionStopAction) else exact_root)[position]) for name, position in species_index.items()}),
            firing_ordinal=1 if last is None else last.firing_ordinal + 1, nested_action_payload=action_payload,
        )
        trigger_last_firings[index] = firing
        trigger_firings.append(firing)

    def finish(stop: int | None = None) -> _BoundaryApplicationResult:
        snapshot()
        continuation, projection = project_nonnegative_trajectory(current.reshape(-1, 1), context=f"Intervention continuation state at t = {boundary:g} s")
        termination = None
        if stop is not None:
            record(stop)
            termination = SimulationTermination(boundary, "stop")
        return _BoundaryApplicationResult(
            continuation.reshape(-1), projection, temperature_K,
            SimulationPresentationBoundary(boundary, tuple(states)) if len(states) > 1 else None,
            termination,
            skipped_instants=(
                tuple(item for item in static if item not in executed_values)
                if stop is not None else ()
            ),
        )

    def crossed_stop(before: np.ndarray, before_temperature_K: float) -> int | None:
        # A stop the last change of state or temperature crossed.
        if not stops or (before_temperature_K == temperature_K and np.array_equal(before, current)):
            return None
        for index in stops:
            trigger = contract.triggers[index]
            previous = _trigger_value(trigger, boundary, before, species_index=species_index,
                                      observable_value=observable_value, temperature_K=before_temperature_K)
            following = _trigger_value(trigger, boundary, current, species_index=species_index, observable_value=observable_value, temperature_K=temperature_K)
            previous_threshold = _resolve_items((trigger,), boundary, parameter_values, temperature_K=before_temperature_K)[0].threshold
            following_threshold = _resolve_items((trigger,), boundary, parameter_values, temperature_K=temperature_K)[0].threshold
            if _crossed(previous - previous_threshold, following - following_threshold, 0.0, trigger.direction):
                return index
        return None

    eligible_stops = tuple(index for index in trigger_indices if index in stops)
    if eligible_stops:
        return finish(min(eligible_stops, key=lambda index: contract.triggers[index].declaration_ordinal))
    if changed_temperature_K is not None:
        # The temperature change executes first; holds in force follow it.
        before = current.copy()
        temperature_K = changed_temperature_K
        snapshot()
        current = _apply_holds(current, tuple(item for item in active_holds if item.start_time < boundary), species_index=species_index, cancellation_check=cancellation_check, time=boundary, temperature_K=temperature_K, parameter_values=parameter_values)
        snapshot()
        stop = crossed_stop(before, root_temperature)
        if stop is not None:
            return finish(stop)
    if not stops:
        validate_boundary_conflicts(
            instants=_resolve_items(static + trigger_instants, boundary, parameter_values, temperature_K=temperature_K),
            entering_holds=_resolve_items(entering_holds, boundary, parameter_values, temperature_K=temperature_K),
            cancellation_check=cancellation_check,
        )
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
                    # Admission requires initial_volume for every volume-changing action.
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
            stop = crossed_stop(before, temperature_K)
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
    output = np.asarray(state, dtype=float).reshape(-1).copy()
    for instant in instants:
        _raise_if_cancelled(cancellation_check)
        index = species_index[instant.species]
        # Non-physical instants are only set or jump.
        if instant.op == "set":
            if index not in held_indices:
                output[index] = float(instant.value)
        elif index not in held_indices:
            output[index] = max(
                0.0, float(output[index]) + float(instant.delta)
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
        values = parameter_values(temperature_K, roots)
        return tuple(
            resolve_executable_values(item, values, time=time) for item in items
        )
    except (AlgebraError, ValueDomainError) as exc:
        exc.annotate(prefix=f"Intervention values at t = {time:g} s: ", time=time)
        raise


def _apply_holds(
    state, holds, *, species_index, cancellation_check, time, temperature_K, parameter_values
):
    if not holds:
        return state
    resolved = _resolve_items(holds, time, parameter_values, temperature_K=temperature_K)
    output = np.asarray(state, dtype=float).reshape(-1).copy()
    for hold in resolved:
        _raise_if_cancelled(cancellation_check)
        output[species_index[hold.species]] = float(hold.value)
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
    # A held value is a literal or a param, which changes only with
    # temperature, so each run of rows at one temperature takes the values
    # resolved at its first row.
    if any(hold.value_param is not None for hold in holds):
        starts = np.flatnonzero(np.diff(temperatures, prepend=np.nan) != 0.0).tolist()
    else:
        starts = [0] if len(temperatures) else []
    for start, end in zip(starts, [*starts[1:], len(temperatures)]):
        _raise_if_cancelled(cancellation_check)
        for hold in _resolve_items(
            holds, float(times[start]), parameter_values,
            temperature_K=float(temperatures[start]),
        ):
            output[species_index[hold.species], start:end] = float(hold.value)
    return output


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
        self.fluxes = tuple(fluxes)
        self._holds = tuple(holds)
        self._species_index = dict(species_index)
        # The held species, whose derivatives are zero.
        self.fixed = frozenset(self._species_index[hold.species] for hold in self._holds)
        self._parameter_values = parameter_values
        self.temperature_at = temperature_at
        self.cancellation_check = cancellation_check
        # Without holds, flows or a vessel the segment is the mechanism alone;
        # its right-hand side only reads the solver's state.
        self._mechanism_only = not self.fluxes and not self._holds and volume.value is None

    def _flux_values(self, time, temperature_K):
        sums = {}
        negative = set()
        for flux in _resolve_items(
            self.fluxes, time, self._parameter_values, temperature_K=temperature_K
        ):
            if flux.physical is not None:
                continue
            index = self._species_index[flux.species]
            sums[index] = sums.get(index, 0.0) + float(flux.flux_rate)
            if flux.flux_rate < 0.0:
                negative.add(index)
        return sums, negative

    def base_evaluation_state(
        self,
        state: np.ndarray,
        time: float,
        temperature_K: float,
    ) -> tuple[np.ndarray, dict[int, float], set[int]]:
        """The state the mechanism is evaluated at, with the flux sums and
        draining species at this time."""
        raw = np.asarray(state, dtype=float).reshape(-1)
        evaluation_state = _apply_holds(
            raw[:len(self._species_index)].copy(),
            self._holds,
            species_index=self._species_index,
            cancellation_check=None,
            time=time,
            temperature_K=temperature_K,
            parameter_values=self._parameter_values,
        )
        sums, negative = self._flux_values(time, temperature_K)
        for index in negative:
            if evaluation_state[index] < 0.0:
                evaluation_state[index] = 0.0
        return evaluation_state, sums, negative

    def __call__(self, time: float, state: np.ndarray) -> np.ndarray:
        _raise_if_cancelled(self.cancellation_check)
        temperature_K = self.temperature_at(time)
        if self._mechanism_only:
            return self._rhs(time, state, T=temperature_K)
        raw_state = np.asarray(state, dtype=float).reshape(-1)
        evaluation_state, sums, _negative = self.base_evaluation_state(raw_state, time, temperature_K)
        # The mechanism's right-hand side returns a fresh array.
        derivative = self._rhs(time, evaluation_state, T=temperature_K)
        volume_rate = 0.0
        if self.volume.value is not None:
            vessel = raw_state[-1]
            volume_rate, mixing = self.volume.flow(self.fluxes, time, temperature_K, vessel, evaluation_state, self.fixed)
            derivative += mixing
        for index, rate in sums.items():
            derivative[index] += rate
        for index in self.fixed:
            derivative[index] = 0.0
        return np.append(derivative, volume_rate) if self.volume.value is not None else derivative


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

    A free derivative that is not positive holds a species, and a species
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
    indices = tuple(index for index in zero if free[index] <= 0.0)
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
    ) -> None:
        self._jacobian = jacobian
        self._rhs = rhs

    def __call__(self, time: float, state: np.ndarray) -> np.ndarray:
        _raise_if_cancelled(self._rhs.cancellation_check)
        temperature_K = self._rhs.temperature_at(time)
        raw_state = np.asarray(state, dtype=float).reshape(-1)
        used, _sums, negative = self._rhs.base_evaluation_state(raw_state, time, temperature_K)
        held = self._rhs.fixed
        matrix = np.asarray(self._jacobian(time, used, T=temperature_K), dtype=float).copy()
        size = used.size
        if self._rhs.volume.value is not None:
            vessel = float(state[-1])
            rate, mixing = self._rhs.volume.flow(self._rhs.fluxes, time, temperature_K, vessel, used, held)
            matrix -= np.eye(size) * (rate / vessel)
            coupled = np.zeros((size + 1, size + 1))
            coupled[:size, :size] = matrix
            coupled[:size, -1] = -mixing / vessel
            matrix = coupled
        if held:
            indices = np.fromiter(sorted(held), dtype=np.int64)
            matrix[indices, :] = 0.0
            matrix[:, indices] = 0.0
        # Drained species evaluated at zero past an unlocated root.
        purge_columns = {index for index in negative if raw_state[index] < 0.0}
        if purge_columns:
            matrix[:, np.fromiter(sorted(purge_columns), dtype=np.int64)] = 0.0
        return matrix

def _trigger_direction(direction: str) -> float:
    return {"rising": 1.0, "falling": -1.0, "either": 0.0}[direction]


def _next_eligible_time(
    trigger: ExecutableTrigger, last: TriggerFiring | None,
) -> float | None:
    if isinstance(trigger.then, InterventionStopAction):
        return -math.inf
    if last is None:
        return -math.inf
    if last.firing_ordinal >= trigger.max_count:
        return None
    # Without a minimum interval a trigger re-arms at its own firing.
    return last.root_time + trigger.min_interval


def _trigger_callables(
    triggers: Sequence[ExecutableTrigger],
    *,
    species_index: Mapping[str, int],
    segment_start: float,
    segment_state: np.ndarray,
    trigger_last_firings: Sequence[TriggerFiring | None],
    observable_value: Callable[[str, float, np.ndarray, float], float] | None,
    temperature_at,
    parameter_values, holds,
) -> tuple[tuple[Callable[[float, np.ndarray], float], ...], tuple[int, ...]]:
    callables: list[Callable[[float, np.ndarray], float]] = []
    mapping: list[int] = []
    for index, trigger in enumerate(triggers):
        last = trigger_last_firings[index]
        ready = _next_eligible_time(trigger, last)
        if ready is None or segment_start < ready:
            continue
        initial_temperature = temperature_at(segment_start)
        threshold = _resolve_items((trigger,), segment_start, parameter_values, temperature_K=initial_temperature)[0].threshold
        start_value = _trigger_value(trigger, segment_start, _apply_holds(segment_state, holds, species_index=species_index, cancellation_check=None, time=segment_start, temperature_K=initial_temperature, parameter_values=parameter_values), species_index=species_index, observable_value=observable_value, temperature_K=initial_temperature)
        # The located root lies on the threshold only to the root finder's
        # precision.  A trigger re-armed at its own firing whose watched value
        # that boundary left unchanged measures from that root's offset from
        # the threshold it fired at, so the root's last ulp is not a crossing
        # while a threshold change at the boundary still counts.
        offset = (
            last.event_state[trigger.condition.name] - last.threshold
            if last is not None and last.root_time == segment_start
            and start_value == last.event_state[trigger.condition.name]
            else 0.0
        )
        starts_at_threshold = start_value - threshold - offset == 0.0
        # SciPy evaluates the observer and stopper copies, and both directions
        # of "either", at the same time and state.
        last_value: list[object] = [None, None]

        def threshold_distance(
            time: float,
            state: np.ndarray,
            *,
            _trigger: ExecutableTrigger = trigger,
            _last: list[object] = last_value,
        ) -> float:
            key = (time, state.tobytes())
            if _last[0] != key:
                temperature_K = temperature_at(time)
                held = _apply_holds(state, holds, species_index=species_index, cancellation_check=None, time=time, temperature_K=temperature_K, parameter_values=parameter_values)
                threshold = _resolve_items((_trigger,), time, parameter_values, temperature_K=temperature_K)[0].threshold
                _last[0], _last[1] = key, _trigger_value(_trigger, time, held, species_index=species_index, observable_value=observable_value, temperature_K=temperature_K) - threshold
            return _last[1]

        directions = (
            ("rising", "falling")
            if trigger.direction == "either"
            else (trigger.direction,)
        )
        for direction in directions:
            # Initial equality is neutral: it is neither side of a crossing.
            # While the target remains exactly at equality, including at the
            # segment start, report the target side's sign so the first
            # departure can only be neutral or look like the opposite
            # direction, which SciPy ignores.  Every other evaluation is the
            # pure threshold function.  ``either`` uses one rising and one
            # falling event so genuine later crossings in both directions
            # remain observable.
            neutral_value = 1.0 if direction == "rising" else -1.0

            def event(
                time: float,
                state: np.ndarray,
                *,
                _distance: Callable[[float, np.ndarray], float] = threshold_distance,
                _offset: float = offset,
                _starts_at_threshold: bool = starts_at_threshold,
                _neutral_value: float = neutral_value,
            ) -> float:
                value = _distance(time, state) - _offset
                if _starts_at_threshold and value == 0.0:
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
    trigger_last_firings: Sequence[TriggerFiring | None],
) -> float | None:
    candidates: list[float] = []
    for trigger, last in zip(triggers, trigger_last_firings):
        ready = _next_eligible_time(trigger, last)
        if (
            ready is not None
            and math.isfinite(ready)
            and segment_start < ready < segment_end
        ):
            candidates.append(ready)
    return min(candidates) if candidates else None


def _first_hit(
    event_times: Sequence[Sequence[float]],
    event_states: Sequence[Sequence[np.ndarray]],
    *,
    mapping: Sequence[int],
    event_offset: int = 0,
) -> tuple[float, tuple[int, ...], np.ndarray] | None:
    """The earliest root of the mapped events, the sorted mapped indices with
    a root at exactly that time, and a writable copy of the state there."""
    hits = [
        (float(raw_time), mapped, event_states[event_offset + offset][position])
        for offset, mapped in enumerate(mapping)
        for position, raw_time in enumerate(event_times[event_offset + offset])
    ]
    if not hits:
        return None
    root_time = min(item[0] for item in hits)
    simultaneous = [item for item in hits if item[0] == root_time]
    return (
        root_time,
        tuple(sorted({item[1] for item in simultaneous})),
        np.array(simultaneous[0][2], dtype=float).reshape(-1),
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
    output_discontinuity_ids: list[np.ndarray],
    times: np.ndarray,
    states: np.ndarray,
    *,
    output_temperatures: list[np.ndarray],
    temperatures: np.ndarray,
    segment_id: int,
    discontinuity_id: int,
) -> None:
    emit_times = np.asarray(times, dtype=float).reshape(-1)
    emit_states = np.asarray(states, dtype=float)
    emit_temperatures = np.asarray(temperatures, dtype=float)
    if not emit_times.size:
        return
    if output_times:
        last_time = float(output_times[-1][-1])
        if emit_times[0] == last_time:
            emit_times = emit_times[1:]
            emit_states = emit_states[:, 1:]
            emit_temperatures = emit_temperatures[1:]
    if not emit_times.size:
        return
    output_times.append(emit_times)
    output_states.append(emit_states)
    output_temperatures.append(emit_temperatures)
    output_segment_ids.append(
        np.full(emit_times.size, int(segment_id), dtype=np.int64)
    )
    output_discontinuity_ids.append(
        np.full(emit_times.size, int(discontinuity_id), dtype=np.int64)
    )


def _append_identity_row(
    output_times: list[np.ndarray],
    output_states: list[np.ndarray],
    output_segment_ids: list[np.ndarray],
    output_discontinuity_ids: list[np.ndarray],
    time: float,
    state: np.ndarray,
    *,
    output_temperatures: list[np.ndarray],
    temperature_K: float,
    segment_id: int,
    discontinuity_id: int,
) -> None:
    _append_identity_samples(
        output_times,
        output_states,
        output_segment_ids,
        output_discontinuity_ids,
        np.array([float(time)], dtype=float),
        np.asarray(state, dtype=float).reshape(-1, 1),
        output_temperatures=output_temperatures,
        temperatures=np.array([temperature_K], dtype=float),
        segment_id=segment_id,
        discontinuity_id=discontinuity_id,
    )
