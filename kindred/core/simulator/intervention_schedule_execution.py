# SPDX-License-Identifier: MIT
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Callable, Mapping, Protocol, Sequence

import numpy as np

from kindred.core.exceptions import SimulationCancelled, create_solver_error
from kindred.core.intervention_schedule import InterventionScheduleError
from kindred.core.intervention_schedule_compiler import (
    ExecutableFlux,
    ExecutableHold,
    ExecutableInstant,
    ExecutableTrigger,
    InterventionExecutionContract,
    TriggerFiring,
    validate_boundary_conflicts,
)
from kindred.core.temperature import declared_temperature_step_boundaries
from kindred.core.symbolic.jacobian_execution import SymbolicJacobianExecution
from kindred.core.symbolic.artifacts import SYMBOLIC_JACOBIAN_IDENTITY_ATTR

from .nonnegative_invariant import (
    NonnegativeProjection,
    project_nonnegative_trajectory,
)
from .solver_types import Rhs2, SimulationOutput, SimulationRequest


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
    rhs_for_jac: Rhs2
    t0: float
    t1: float
    y0: np.ndarray
    t_eval: np.ndarray
    provenance: dict[str, object]
    method: str


@dataclass(frozen=True)
class SegmentRunResult:
    identity: SegmentIdentity
    output: SimulationOutput
    event_times: Sequence[Sequence[float]]
    raw_event_states: Sequence[Sequence[np.ndarray]]
    raw_terminal_state: np.ndarray | None
    trajectory_projection: NonnegativeProjection
    event_state_projection: NonnegativeProjection
    solver_used: str
    symbolic_jacobian_used: bool = False
    jacobian_sparsity_hint: bool = False

    def __post_init__(self) -> None:
        if self.raw_terminal_state is None:
            return
        state = np.asarray(
            self.raw_terminal_state,
            dtype=float,
        ).reshape(-1).copy()
        if not np.all(np.isfinite(state)):
            raise ValueError("Raw solver terminal state must be finite.")
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


class BaseSegmentRunner(Protocol):
    def __call__(self, request: SegmentExecutionRequest) -> SegmentRunResult: ...


@dataclass(frozen=True)
class ScheduleExecutionRequest:
    request: SimulationRequest
    rhs: Rhs2
    rhs_for_jac: Rhs2
    t0: float
    t1: float
    y0: np.ndarray
    t_eval: np.ndarray
    provenance: dict[str, object]
    method: str
    schedule: InterventionExecutionContract


class InterventionScheduleExecutionOwner:
    def __init__(self, segment_runner: BaseSegmentRunner) -> None:
        self._segment_runner = segment_runner

    def execute(self, request: ScheduleExecutionRequest) -> SimulationOutput:
        req = request.request
        t0 = float(request.t0)
        t1 = float(request.t1)
        t_eval = np.asarray(request.t_eval, dtype=float).reshape(-1)
        contract = request.schedule
        species_index = _species_index(req, contract)
        current_y = np.asarray(request.y0, dtype=float).reshape(-1).copy()
        user_events = tuple(req.events or ())
        cancellation_check = _cancellation_check(user_events)
        base_request = replace(req, events=user_events) if user_events else req
        user_terminal = _event_terminal_flags(base_request, user_events)
        user_event_provenance = [[] for _ in user_events]
        trigger_counts = [0 for _ in contract.triggers]
        trigger_last_times: list[float | None] = [None for _ in contract.triggers]
        trigger_firings: list[TriggerFiring] = []
        outputs_t: list[np.ndarray] = []
        outputs_y: list[np.ndarray] = []
        output_segment_ids: list[np.ndarray] = []
        output_segment_keys: list[tuple[str, ...]] = []
        output_discontinuity_ids: list[np.ndarray] = []
        segment_solvers: list[str] = []
        segment_symbolic_jacobian_usage: list[bool] = []
        segment_symbolic_jacobian_runtime_fallbacks: list[
            dict[str, object]
        ] = []
        integration_segments: list[dict[str, object]] = []
        segment_projections: dict[str, SegmentProjection] = {}
        segment_count = 0
        discontinuity_id = 0
        terminal_stop = False
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
        )
        current_y = initial_boundary.continuation_state
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
        )
        _raise_if_cancelled(cancellation_check)
        segment_starts_after_discontinuity = initial_state_transaction
        for boundary_index in range(len(ordered_boundaries) - 1):
            outer_start = float(ordered_boundaries[boundary_index])
            outer_end = float(ordered_boundaries[boundary_index + 1])
            if outer_end <= outer_start:
                continue
            sub_start = outer_start
            while sub_start < outer_end:
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
                segment_rhs: Rhs2 = request.rhs
                segment_rhs_for_jac: Rhs2 = request.rhs_for_jac
                if active_fluxes or active_holds:
                    segment_rhs = _InterventionRhs(
                        request.rhs,
                        fluxes=active_fluxes,
                        holds=active_holds,
                        species_index=species_index,
                    )
                    segment_rhs_for_jac = _InterventionRhs(
                        request.rhs_for_jac,
                        fluxes=active_fluxes,
                        holds=active_holds,
                        species_index=species_index,
                    )
                trigger_callables, trigger_mapping = _trigger_callables(
                    contract.triggers,
                    species_index=species_index,
                    segment_start=sub_start,
                    segment_state=current_y,
                    trigger_counts=trigger_counts,
                    trigger_last_times=trigger_last_times,
                )
                # SciPy truncates a simultaneous active-event set at the first
                # terminal event.  Register every trigger first as a
                # non-terminal observer, then repeat the same callables as
                # terminal stoppers.  The observer block therefore retains
                # every exact equal-time root before the first stopper ends
                # the segment.
                trigger_observers = trigger_callables
                trigger_stoppers = trigger_callables
                segment_events = (
                    user_events + trigger_observers + trigger_stoppers
                )
                segment_request = base_request
                if segment_events:
                    segment_request = replace(
                        segment_request,
                        events=segment_events,
                        event_terminal=tuple(user_terminal)
                        + tuple(False for _ in trigger_observers)
                        + tuple(True for _ in trigger_stoppers),
                    )
                if active_holds or active_fluxes:
                    transformed_jacobian = (
                        None
                        if segment_request.jacobian_func is None
                        else _InterventionJacobian(
                            segment_request.jacobian_func,
                            rhs=segment_rhs,
                            holds=active_holds,
                            fluxes=active_fluxes,
                            species_index=species_index,
                        )
                    )
                    segment_request = replace(
                        segment_request,
                        jacobian_func=transformed_jacobian,
                        jac_sparsity=_transform_jacobian_sparsity(
                            segment_request.jac_sparsity,
                            holds=active_holds,
                            species_index=species_index,
                            size=current_y.size,
                        ),
                    )
                segment_request = _request_for_internal_segment(
                    segment_request, t0=sub_start, t1=sub_end
                )
                requested = _requested_eval(
                    t_eval,
                    start=sub_start,
                    end=sub_end,
                    is_final=(sub_end == t1),
                )
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
                segment_provenance = dict(request.provenance)
                segment_provenance["intervention_segment_index"] = segment_count
                _raise_if_cancelled(cancellation_check)
                segment_result = self._segment_runner(
                    SegmentExecutionRequest(
                        identity=segment_identity,
                        request=segment_request,
                        rhs=segment_rhs,
                        rhs_for_jac=segment_rhs_for_jac,
                        t0=sub_start,
                        t1=sub_end,
                        y0=current_y,
                        t_eval=internal_eval,
                        provenance=segment_provenance,
                        method=request.method,
                    )
                )
                _raise_if_cancelled(cancellation_check)
                if segment_result.identity != segment_identity:
                    raise InterventionScheduleError(
                        "Segment runner returned a mismatched segment identity."
                    )
                event_times = _normal_event_times(
                    segment_result.event_times, count=len(segment_events)
                )
                event_states = _normal_event_states(
                    segment_result.raw_event_states,
                    count=len(segment_events),
                    species_count=current_y.size,
                )
                for index, times in enumerate(event_times[: len(user_events)]):
                    user_event_provenance[index].extend(times)
                terminal_event_time = _terminal_user_event_time(
                    user_events, user_terminal, event_times[: len(user_events)]
                )
                terminal_stop = terminal_event_time is not None
                trigger_hit = (
                    None
                    if terminal_stop
                    else _first_trigger_hit(
                        event_times,
                        event_states,
                        trigger_mapping=trigger_mapping,
                        user_event_count=len(user_events),
                    )
                )
                seg_t = np.asarray(segment_result.output.t, dtype=float).reshape(-1)
                seg_y = np.asarray(segment_result.output.Y, dtype=float)
                _validate_segment_output(
                    seg_t,
                    seg_y,
                    species_count=current_y.size,
                    solver=req.solver,
                    time=sub_start,
                )
                actual_segment_end = (
                    float(trigger_hit[0])
                    if trigger_hit is not None
                    else (
                        float(terminal_event_time)
                        if terminal_event_time is not None
                        else sub_end
                    )
                )
                segment_output_identity = replace(
                    segment_result.identity,
                    t_end=actual_segment_end,
                )
                _record_segment_projection(
                    segment_projections,
                    replace(segment_result, identity=segment_output_identity),
                )
                if segment_count == 0:
                    _record_continuation_projection(
                        segment_projections,
                        identity=segment_output_identity,
                        projection=initial_boundary.projection,
                    )
                segment_solvers.append(
                    str(segment_result.solver_used or request.method)
                )
                segment_symbolic_jacobian_usage.append(
                    bool(segment_result.symbolic_jacobian_used)
                )
                runtime_fallback = segment_result.output.provenance.get(
                    "symbolic_jacobian_runtime_fallback"
                )
                runtime_fallback_payload = (
                    dict(runtime_fallback)
                    if isinstance(runtime_fallback, Mapping)
                    else None
                )
                if runtime_fallback_payload is not None:
                    runtime_fallback_payload["segment"] = (
                        segment_output_identity.to_payload()
                    )
                    segment_symbolic_jacobian_runtime_fallbacks.append(
                        runtime_fallback_payload
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
                    "jacobian_sparsity_hint": bool(
                        segment_result.jacobian_sparsity_hint
                    ),
                }
                if runtime_fallback_payload is not None:
                    segment_payload["symbolic_jacobian_runtime_fallback"] = (
                        runtime_fallback_payload
                    )
                integration_segments.append(segment_payload)
                segment_count += 1
                if trigger_hit is not None:
                    root_time, trigger_indices, root_state = trigger_hit
                    current_y = root_state.copy()
                elif not terminal_stop:
                    current_y = _required_raw_terminal_state(
                        segment_result,
                        species_count=current_y.size,
                    )
                scheduled_discontinuity = (
                    trigger_hit is None
                    and not terminal_stop
                    and sub_end in contract.discontinuities
                )
                emit_count = min(requested.size, seg_t.size)
                emit_t = seg_t[:emit_count]
                emit_y = seg_y[:, :emit_count]
                if trigger_hit is not None:
                    emit_mask = emit_t < float(trigger_hit[0])
                    emit_t = emit_t[emit_mask]
                    emit_y = emit_y[:, emit_mask]
                elif scheduled_discontinuity:
                    emit_mask = emit_t < float(sub_end)
                    emit_t = emit_t[emit_mask]
                    emit_y = emit_y[:, emit_mask]
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
                )
                if terminal_stop:
                    break
                if trigger_hit is None:
                    if next_rearm is not None and sub_end < outer_end:
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
                        )
                        current_y = boundary_result.continuation_state
                        _raise_if_cancelled(cancellation_check)
                        _record_continuation_projection(
                            segment_projections,
                            identity=segment_output_identity,
                            projection=boundary_result.projection,
                        )
                        segment_starts_after_discontinuity = False
                        sub_start = sub_end
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
                        )
                        current_y = boundary_result.continuation_state
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
                                if outer_end == t1
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
                            )
                            segment_starts_after_discontinuity = outer_end < t1
                        elif (
                            outer_end in temperature_boundaries
                            and outer_end < t1
                        ):
                            _append_identity_row(
                                outputs_t,
                                outputs_y,
                                output_segment_ids,
                                output_segment_keys,
                                output_discontinuity_ids,
                                float(outer_end),
                                np.asarray(current_y, dtype=float).reshape(-1),
                                segment_id=segment_count,
                                segment_key=f"integration:{segment_count}",
                                discontinuity_id=discontinuity_id,
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
                )
                current_y = boundary_result.continuation_state
                _raise_if_cancelled(cancellation_check)
                _record_continuation_projection(
                    segment_projections,
                    identity=segment_output_identity,
                    projection=boundary_result.projection,
                )
                discontinuity_id += 1
                post_segment_id = (
                    segment_output_identity.ordinal
                    if root_time == t1
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
                )
                segment_starts_after_discontinuity = root_time < t1
                sub_start = root_time
            if terminal_stop:
                break

        if outputs_t:
            output_t = np.concatenate(outputs_t)
            output_y = np.concatenate(outputs_y, axis=1)
        else:
            output_t = np.empty(0, dtype=float)
            output_y = np.empty((current_y.size, 0), dtype=float)
        _ensure_finite_nonnegative(
            output_y,
            solver=req.solver,
            time=t0,
            context="Scheduled simulation output",
        )
        final_contract = contract.with_trigger_firings(trigger_firings)
        provenance = request.provenance
        _raise_if_cancelled(cancellation_check)
        provenance["intervention_execution"] = final_contract.provenance_projection()
        provenance["integration_segments"] = integration_segments
        provenance["intervention_segments"] = segment_count
        if segment_symbolic_jacobian_runtime_fallbacks:
            provenance["symbolic_jacobian_runtime_fallbacks"] = list(
                segment_symbolic_jacobian_runtime_fallbacks
            )
        base_symbolic_jacobian = SymbolicJacobianExecution.from_request_fields(
            jacobian_func=base_request.jacobian_func,
            jac_sparsity=base_request.jac_sparsity,
            status=base_request.symbolic_jacobian_status,
        )
        if (
            base_symbolic_jacobian.has_executable_jacobian
            and segment_symbolic_jacobian_usage
            and not all(segment_symbolic_jacobian_usage)
        ):
            partially_disabled = any(segment_symbolic_jacobian_usage)
            if segment_symbolic_jacobian_runtime_fallbacks:
                status_code = "nonfinite-runtime"
                status_reason = (
                    "Symbolic Jacobian disabled only for segments where its "
                    "runtime evaluation was non-finite or invalid."
                    if partially_disabled
                    else
                    "Symbolic Jacobian disabled for every executed segment "
                    "because its runtime evaluation was non-finite or "
                    "invalid."
                )
            else:
                status_code = "singular-state"
                status_reason = (
                    "Symbolic Jacobian disabled only for segments with a "
                    "non-finite initial symbolic Jacobian."
                    if partially_disabled
                    else
                    "Symbolic Jacobian disabled for every executed segment "
                    "because it was non-finite at the segment initial state."
                )
            runtime_symbolic_jacobian = (
                base_symbolic_jacobian.with_runtime_disabled(
                    partially=partially_disabled,
                    code=status_code,
                    reason=status_reason,
                )
            )
            provenance.pop("symbolic_jacobian_identity", None)
            provenance.update(runtime_symbolic_jacobian.provenance_fields())
        _publish_direct_result_sample_identity(
            provenance,
            output_segment_ids,
            output_segment_keys,
            output_discontinuity_ids,
        )
        _publish_segment_projection_provenance(
            provenance,
            segment_projections,
        )
        if segment_solvers:
            distinct = list(dict.fromkeys(segment_solvers))
            provenance["solver_used"] = (
                distinct[0] if len(distinct) == 1 else "segmented"
            )
            if len(distinct) > 1:
                provenance["intervention_segment_solvers"] = segment_solvers
        if user_event_provenance:
            provenance["events"] = [
                list(values) for values in user_event_provenance
            ]
        _raise_if_cancelled(cancellation_check)
        return SimulationOutput(
            t=output_t,
            Y=output_y,
            provenance=provenance,
            intervention_execution=final_contract,
        )


def _cancellation_check(
    events: Sequence[Callable[[float, np.ndarray], float]],
) -> Callable[[], bool] | None:
    callbacks = tuple(
        callback
        for event in events
        if bool(getattr(event, "_kindred_cancel_event", False))
        for callback in (getattr(event, "_kindred_cancelled", None),)
        if callable(callback)
    )
    if not callbacks:
        return None

    def cancelled() -> bool:
        return any(bool(callback()) for callback in callbacks)

    return cancelled


def _raise_if_cancelled(check: Callable[[], bool] | None) -> None:
    if check is not None and check():
        raise SimulationCancelled()


def _publish_direct_result_sample_identity(
    provenance: dict[str, object],
    segment_ids: Sequence[np.ndarray],
    segment_keys: Sequence[tuple[str, ...]],
    discontinuity_ids: Sequence[np.ndarray],
) -> None:
    """Publish row identity captured at emission time.

    Identity is attached while each unique segment/root row is emitted, so no
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
    provenance["result_segment_ids"] = tuple(int(value) for value in ids)
    provenance["result_segment_keys"] = keys
    provenance["result_discontinuity_ids"] = tuple(
        int(value) for value in discontinuities
    )


def _required_raw_terminal_state(
    result: SegmentRunResult,
    *,
    species_count: int,
) -> np.ndarray:
    state = result.raw_terminal_state
    if state is None:
        raise InterventionScheduleError(
            "Completed solver segment is missing its exact raw terminal state."
        )
    values = np.asarray(state, dtype=float).reshape(-1).copy()
    if values.size != int(species_count) or not np.all(np.isfinite(values)):
        raise InterventionScheduleError(
            "Completed solver segment returned an invalid raw terminal state."
        )
    return values


def _record_segment_projection(
    records: dict[str, SegmentProjection],
    result: SegmentRunResult,
) -> None:
    key = result.identity.key
    if key in records:
        raise InterventionScheduleError(
            f"Duplicate segment projection identity: {key!r}."
        )
    records[key] = SegmentProjection(
        identity=result.identity,
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
        *(item.species for item in contract.instants),
        *(item.species for item in contract.fluxes),
        *(item.species for item in contract.holds),
        *(item.watch_species for item in contract.triggers),
        *(item.then.species for item in contract.triggers),
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
    return tuple(item for item in contract.instants if item.time == time)


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


def _trigger_action_instant(
    trigger: ExecutableTrigger, *, time: float
) -> ExecutableInstant:
    action = trigger.then
    return ExecutableInstant(
        op=action.op,
        species=action.species,
        time=time,
        value=action.value,
        delta=action.delta,
        declaration_ordinal=trigger.declaration_ordinal,
        phase="ordinary",
    )


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
) -> _BoundaryApplicationResult:
    _raise_if_cancelled(cancellation_check)
    current = (
        np.asarray(trigger_state, dtype=float).reshape(-1).copy()
        if trigger_state is not None
        else np.asarray(state, dtype=float).reshape(-1).copy()
    )
    exact_root = current.copy()
    projectable_solver_negatives = (
        exact_root < 0.0
        if input_is_raw_solver_state
        else np.zeros(exact_root.shape, dtype=bool)
    )
    exact_event_state = {
        species: float(current[index])
        for species, index in species_index.items()
    }
    static = _static_instants_at(contract, boundary)
    active_holds = _active_holds(contract, boundary)
    entering_holds = tuple(
        hold for hold in active_holds if hold.start_time == boundary
    )
    trigger_instants = tuple(
        _trigger_action_instant(contract.triggers[index], time=boundary)
        for index in trigger_indices
    )
    validate_boundary_conflicts(
        instants=static + trigger_instants,
        entering_holds=entering_holds,
    )
    stage_three = tuple(
        sorted(
            (
                item
                for item in static
                if item.phase in {"ordinary", "before"}
            ),
            key=_instant_canonical_order,
        )
    )
    stage_six = tuple(
        sorted(
            trigger_instants,
            key=_instant_canonical_order,
        )
    )
    stage_seven = tuple(
        sorted(
            (item for item in static if item.phase == "after"),
            key=_instant_canonical_order,
        )
    )
    current = _apply_instants(
        current,
        stage_three,
        species_index=species_index,
        cancellation_check=cancellation_check,
    )
    current = _apply_holds(
        current,
        active_holds,
        species_index=species_index,
        cancellation_check=cancellation_check,
    )
    new_firings: list[tuple[int, TriggerFiring]] = []
    if stage_six:
        current = _apply_instants(
            current,
            stage_six,
            species_index=species_index,
            cancellation_check=cancellation_check,
        )
        for index in trigger_indices:
            trigger = contract.triggers[index]
            next_count = trigger_counts[index] + 1
            new_firings.append(
                (
                    index,
                    TriggerFiring(
                        declaration_ordinal=trigger.declaration_ordinal,
                        watch_species=trigger.watch_species,
                        threshold=trigger.threshold,
                        direction=trigger.direction,
                        root_time=boundary,
                        event_state=exact_event_state,
                        firing_ordinal=next_count,
                        nested_action_payload=trigger.then.to_payload(),
                    ),
                )
            )
    current = _apply_instants(
        current,
        stage_seven,
        species_index=species_index,
        cancellation_check=cancellation_check,
    )
    invalid_negative = (current < 0.0) & (
        ~projectable_solver_negatives | (current < exact_root)
    )
    if np.any(invalid_negative):
        raise InterventionScheduleError(
            "Intervention boundary produced a scientifically invalid "
            "negative continuation state."
        )
    continuation, projection = project_nonnegative_trajectory(
        current.reshape(-1, 1),
        context=f"Intervention continuation state at t={boundary:g}",
    )
    _raise_if_cancelled(cancellation_check)
    for index, firing in new_firings:
        trigger_counts[index] += 1
        trigger_last_times[index] = boundary
        trigger_firings.append(firing)
    return _BoundaryApplicationResult(
        continuation_state=continuation.reshape(-1),
        projection=projection,
    )


def _instant_canonical_order(
    instant: ExecutableInstant,
) -> tuple[int, int, int, int]:
    return (
        int(instant.declaration_ordinal),
        {"ordinary": 0, "before": 1, "after": 2}[instant.phase],
        -1 if instant.window_ordinal is None else int(instant.window_ordinal),
        -1 if instant.authored_ordinal is None else int(instant.authored_ordinal),
    )


def _apply_instants(
    state: np.ndarray,
    instants: Sequence[ExecutableInstant],
    *,
    species_index: Mapping[str, int],
    cancellation_check: Callable[[], bool] | None,
) -> np.ndarray:
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
            output[index] = value
        elif instant.op == "jump":
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


def _apply_holds(
    state: np.ndarray,
    holds: Sequence[ExecutableHold],
    *,
    species_index: Mapping[str, int],
    cancellation_check: Callable[[], bool] | None,
) -> np.ndarray:
    output = np.asarray(state, dtype=float).reshape(-1).copy()
    for hold in holds:
        _raise_if_cancelled(cancellation_check)
        value = float(hold.value)
        if not math.isfinite(value) or value < 0.0:
            raise InterventionScheduleError(
                "Intervention hold produced an invalid state."
            )
        output[species_index[hold.species]] = value
        _raise_if_cancelled(cancellation_check)
    return output


class _InterventionRhs:
    def __init__(
        self,
        rhs: Rhs2,
        *,
        fluxes: Sequence[ExecutableFlux],
        holds: Sequence[ExecutableHold],
        species_index: Mapping[str, int],
    ) -> None:
        self._rhs = rhs
        self._fluxes = tuple(fluxes)
        self._holds = tuple(holds)
        self._species_index = dict(species_index)

    def __call__(self, time: float, state: np.ndarray) -> np.ndarray:
        evaluation_state = np.asarray(state, dtype=float).copy()
        fixed: set[int] = set()
        for hold in self._holds:
            index = self._species_index[hold.species]
            evaluation_state[index] = hold.value
            fixed.add(index)
        derivative = np.asarray(
            self._rhs(time, evaluation_state), dtype=float
        ).copy()
        signed_flux_by_index: dict[int, float] = {}
        has_negative_flux: set[int] = set()
        for flux in self._fluxes:
            index = self._species_index[flux.species]
            signed_flux_by_index[index] = (
                signed_flux_by_index.get(index, 0.0) + flux.flux_rate
            )
            if flux.flux_rate < 0.0:
                has_negative_flux.add(index)
        for index, rate in signed_flux_by_index.items():
            candidate = float(derivative[index]) + rate
            if index in has_negative_flux and evaluation_state[index] <= 0.0:
                derivative[index] = max(0.0, candidate)
            else:
                derivative[index] = candidate
        for index in fixed:
            derivative[index] = 0.0
        return derivative

    def purge_jacobian_branches(
        self,
        time: float,
        state: np.ndarray,
    ) -> tuple[set[int], set[int]]:
        """Return constant rows and branch boundaries for active purges.

        A negative flux is a production-aware purge: at zero concentration it
        removes up to the available production and therefore has a constant
        zero derivative when the base production cannot satisfy the purge.
        Equality is a piecewise branch boundary; the symbolic Jacobian is
        unavailable there unless the transformed base row is itself constant.
        """

        evaluation_state = np.asarray(state, dtype=float).copy()
        for hold in self._holds:
            evaluation_state[self._species_index[hold.species]] = hold.value
        derivative = np.asarray(self._rhs(time, evaluation_state), dtype=float)
        signed_flux_by_index: dict[int, float] = {}
        for flux in self._fluxes:
            index = self._species_index[flux.species]
            signed_flux_by_index[index] = (
                signed_flux_by_index.get(index, 0.0) + flux.flux_rate
            )
        zero_rows: set[int] = set()
        branch_boundaries: set[int] = set()
        for index, rate in signed_flux_by_index.items():
            if rate >= 0.0 or evaluation_state[index] > 0.0:
                continue
            candidate = float(derivative[index]) + rate
            if candidate < 0.0:
                zero_rows.add(index)
            elif candidate == 0.0:
                branch_boundaries.add(index)
        return zero_rows, branch_boundaries


class _InterventionJacobian:
    """Exact Jacobian transform for holds, fluxes, and purge branches."""

    def __init__(
        self,
        jacobian: Callable[[float, np.ndarray], np.ndarray],
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
        identity = getattr(jacobian, SYMBOLIC_JACOBIAN_IDENTITY_ATTR, None)
        if identity is not None:
            setattr(self, SYMBOLIC_JACOBIAN_IDENTITY_ATTR, identity)
        setattr(
            self,
            "_kindred_symbolic_jacobian_finite_at_state",
            self._finite_at_state,
        )

    def _evaluate_state(self, state: np.ndarray) -> tuple[np.ndarray, set[int]]:
        values = np.asarray(state, dtype=float).reshape(-1)
        held = {
            self._species_index[hold.species]
            for hold in self._holds
        }
        used = values.copy()
        for hold in self._holds:
            used[self._species_index[hold.species]] = float(hold.value)
        return used, held

    def __call__(self, time: float, state: np.ndarray) -> np.ndarray:
        used, held = self._evaluate_state(state)
        matrix = np.asarray(self._jacobian(float(time), used), dtype=float).copy()
        size = used.size
        if matrix.shape != (size, size):
            raise ValueError(
                f"intervention Jacobian expected {(size, size)}, got {matrix.shape}"
            )
        if held:
            indices = np.fromiter(sorted(held), dtype=np.int64)
            matrix[indices, :] = 0.0
            matrix[:, indices] = 0.0
        purge_rows, purge_boundaries = self._rhs.purge_jacobian_branches(
            float(time),
            np.asarray(state, dtype=float),
        )
        if purge_boundaries and any(
            np.any(matrix[index, :] != 0.0)
            for index in purge_boundaries
        ):
            raise ValueError(
                "production-aware purge Jacobian is unavailable at its "
                "piecewise branch boundary"
            )
        if purge_rows:
            matrix[np.fromiter(sorted(purge_rows), dtype=np.int64), :] = 0.0
        return matrix

    def _finite_at_state(self, time: float, state: np.ndarray) -> bool:
        try:
            matrix = self(float(time), np.asarray(state, dtype=float))
        except (ArithmeticError, TypeError, ValueError, OverflowError):
            return False
        return bool(np.all(np.isfinite(matrix)))


def _transform_jacobian_sparsity(
    sparsity: object,
    *,
    holds: Sequence[ExecutableHold],
    species_index: Mapping[str, int],
    size: int,
) -> object:
    """Apply the same row/column mask as the exact hold Jacobian."""

    if sparsity is None:
        return None
    try:
        dense = sparsity.toarray()  # scipy sparse matrices
    except AttributeError:
        dense = sparsity
    matrix = np.asarray(dense, dtype=bool).copy()
    if matrix.shape != (int(size), int(size)):
        raise ValueError(
            f"Jacobian sparsity expected {(int(size), int(size))}, got {matrix.shape}"
        )
    held = {
        species_index[hold.species]
        for hold in holds
    }
    if held:
        indices = np.fromiter(sorted(held), dtype=np.int64)
        matrix[indices, :] = False
        matrix[:, indices] = False
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
        species = species_index[trigger.watch_species]
        threshold = trigger.threshold
        starts_at_threshold = float(segment_state[species]) == threshold
        directions = (
            ("rising", "falling")
            if trigger.direction == "either"
            else (trigger.direction,)
        )
        for direction in directions:
            # Initial equality is neutral: it is neither side of a crossing.
            # Keep this directional event inert until one strict side has
            # actually been occupied.  The neutral sign is chosen so that
            # the first departure can only look like the opposite direction,
            # which SciPy ignores.  ``either`` is represented by one rising
            # and one falling event so both genuine later crossings remain
            # observable without perturbing the accepted initial state.
            strict_side_occupied = [not starts_at_threshold]
            neutral_value = 1.0 if direction == "rising" else -1.0

            def event(
                _time: float,
                state: np.ndarray,
                *,
                _species: int = species,
                _threshold: float = threshold,
                _strict_side_occupied: list[bool] = strict_side_occupied,
                _neutral_value: float = neutral_value,
            ) -> float:
                value = float(state[_species]) - _threshold
                if not _strict_side_occupied[0]:
                    if value == 0.0:
                        return _neutral_value
                    _strict_side_occupied[0] = True
                return value

            event.terminal = True  # type: ignore[attr-defined]
            event.direction = _trigger_direction(direction)  # type: ignore[attr-defined]
            event._kindred_intervention_trigger_index = index  # type: ignore[attr-defined]
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
    user_event_count: int,
) -> tuple[float, tuple[int, ...], np.ndarray] | None:
    hits: list[tuple[float, int, np.ndarray]] = []
    for offset, trigger_index in enumerate(trigger_mapping):
        event_index = user_event_count + offset
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


def _normal_event_times(
    values: object, *, count: int
) -> list[list[float]]:
    output = [[] for _ in range(count)]
    if not isinstance(values, Sequence):
        return output
    for index, raw in enumerate(values[:count]):
        try:
            output[index] = [float(value) for value in raw]
        except TypeError:
            output[index] = []
    return output


def _normal_event_states(
    values: object, *, count: int, species_count: int
) -> list[list[np.ndarray]]:
    output: list[list[np.ndarray]] = [[] for _ in range(count)]
    if not isinstance(values, Sequence):
        return output
    for index, raw in enumerate(values[:count]):
        if not isinstance(raw, Sequence):
            continue
        for state in raw:
            array = np.asarray(state, dtype=float).reshape(-1)
            if array.size != species_count:
                raise InterventionScheduleError(
                    "Trigger event-time state has the wrong species count."
                )
            output[index].append(array)
    return output


def _event_terminal_flags(
    request: SimulationRequest,
    events: Sequence[Callable[[float, np.ndarray], float]],
) -> tuple[bool, ...]:
    explicit = tuple(request.event_terminal or ())
    return tuple(
        bool(explicit[index])
        if index < len(explicit)
        else bool(getattr(event, "terminal", False))
        for index, event in enumerate(events)
    )


def _terminal_user_event_time(
    events: Sequence[Callable[[float, np.ndarray], float]],
    terminal: Sequence[bool],
    times: Sequence[Sequence[float]],
) -> float | None:
    hits: list[float] = []
    for index, values in enumerate(times):
        if not values or index >= len(terminal) or not terminal[index]:
            continue
        if bool(getattr(events[index], "_kindred_cancel_event", False)):
            continue
        hits.extend(float(value) for value in values)
    return min(hits) if hits else None


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
    segment_id: int,
    segment_key: str,
    discontinuity_id: int,
) -> None:
    emit_times = np.asarray(times, dtype=float).reshape(-1)
    emit_states = np.asarray(states, dtype=float)
    if not emit_times.size:
        return
    if emit_states.ndim != 2 or emit_states.shape[1] != emit_times.size:
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
        elif emit_times[0] < last_time:
            raise InterventionScheduleError(
                "Direct result row emission must remain strictly ordered."
            )
    if not emit_times.size:
        return
    output_times.append(emit_times)
    output_states.append(emit_states)
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
        segment_id=segment_id,
        segment_key=segment_key,
        discontinuity_id=discontinuity_id,
    )


def _validate_segment_output(
    times: np.ndarray,
    states: np.ndarray,
    *,
    species_count: int,
    solver: str,
    time: float,
) -> None:
    if states.ndim != 2 or states.shape != (species_count, times.size):
        raise create_solver_error(
            solver, time, "Scheduled simulation segment returned invalid shape."
        )
    _ensure_finite_nonnegative(
        states,
        solver=solver,
        time=time,
        context="Scheduled simulation segment",
    )


def _ensure_finite_nonnegative(
    values: object, *, solver: str, time: float, context: str
) -> None:
    array = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(array)):
        raise create_solver_error(
            solver, time, f"{context} contains non-finite concentration values."
        )
    if np.any(array < 0.0):
        raise create_solver_error(
            solver, time, f"{context} contains negative concentration values."
        )


def _request_for_internal_segment(
    request: SimulationRequest, *, t0: float, t1: float
) -> SimulationRequest:
    if request.first_step is None:
        return request
    if float(request.first_step) <= abs(t1 - t0):
        return request
    return replace(request, first_step=None)
