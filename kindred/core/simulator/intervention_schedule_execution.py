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
        segment_solvers: list[str] = []
        segment_symbolic_jacobian_usage: list[bool] = []
        integration_segments: list[dict[str, object]] = []
        segment_projections: dict[str, SegmentProjection] = {}
        pre_boundary_samples: list[dict[str, object]] = []
        segment_count = 0
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
        _raise_if_cancelled(cancellation_check)
        if _contains_time(t_eval, t0):
            outputs_t.append(np.array([t0], dtype=float))
            outputs_y.append(current_y.reshape(-1, 1))

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
                if active_fluxes or active_holds:
                    segment_request = replace(
                        segment_request, jacobian_func=None, jac_sparsity=None
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
                _record_segment_projection(segment_projections, segment_result)
                segment_solvers.append(
                    str(segment_result.solver_used or request.method)
                )
                segment_symbolic_jacobian_usage.append(
                    bool(segment_result.symbolic_jacobian_used)
                )
                integration_segments.append(
                    {
                        "segment_id": segment_count,
                        "segment_key": segment_identity.key,
                        "segment_identity": segment_identity.to_payload(),
                        "t_start": sub_start,
                        "t_end": sub_end,
                        "starts_after_discontinuity": segment_count > 0,
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
                )
                segment_count += 1
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
                terminal_stop = _terminal_user_event_hit(
                    user_events, user_terminal, event_times[: len(user_events)]
                )
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
                if trigger_hit is not None:
                    root_time, trigger_indices, root_state = trigger_hit
                    current_y = root_state.copy()
                elif not terminal_stop:
                    current_y = _required_raw_terminal_state(
                        segment_result,
                        species_count=current_y.size,
                    )
                if (
                    not terminal_stop
                    and (
                        trigger_hit is not None
                        or sub_end in contract.discontinuities
                        or sub_end in temperature_boundaries
                    )
                ):
                    _record_pre_boundary_sample(
                        pre_boundary_samples,
                        boundary=(
                            float(trigger_hit[0])
                            if trigger_hit is not None
                            else sub_end
                        ),
                        state=current_y,
                        segment_identity=segment_result.identity,
                    )
                emit_count = min(requested.size, seg_t.size)
                _append_samples(
                    outputs_t,
                    outputs_y,
                    seg_t[:emit_count],
                    seg_y[:, :emit_count],
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
                            identity=segment_result.identity,
                            projection=boundary_result.projection,
                        )
                        _replace_boundary_sample(
                            outputs_t,
                            outputs_y,
                            sub_end,
                            current_y,
                        )
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
                            identity=segment_result.identity,
                            projection=boundary_result.projection,
                        )
                        _replace_boundary_sample(
                            outputs_t, outputs_y, outer_end, current_y
                        )
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
                    identity=segment_result.identity,
                    projection=boundary_result.projection,
                )
                _replace_boundary_sample(outputs_t, outputs_y, root_time, current_y)
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
            runtime_symbolic_jacobian = (
                base_symbolic_jacobian.with_runtime_disabled(
                    partially=partially_disabled,
                    code="intervention-segment-disable",
                    reason=(
                        "Symbolic Jacobian disabled only for intervention "
                        "segments with active holds or fluxes."
                        if partially_disabled
                        else
                        "Symbolic Jacobian disabled for every executed segment "
                        "because active holds or fluxes changed the segment RHS."
                    ),
                )
            )
            provenance.pop("symbolic_jacobian_identity", None)
            provenance.update(runtime_symbolic_jacobian.provenance_fields())
        _publish_result_sample_identity(
            provenance,
            output_t,
            integration_segments=integration_segments,
            discontinuities=final_contract.runtime_discontinuities_for_span(
                t0,
                t1,
            ),
        )
        _publish_pre_boundary_sample_identity(
            provenance,
            pre_boundary_samples,
            discontinuities=final_contract.runtime_discontinuities_for_span(
                t0,
                t1,
            ),
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


def _publish_result_sample_identity(
    provenance: dict[str, object],
    times: np.ndarray,
    *,
    integration_segments: Sequence[Mapping[str, object]],
    discontinuities: Sequence[float],
) -> None:
    sample_times = np.asarray(times, dtype=float).reshape(-1)
    if not sample_times.size:
        provenance["result_segment_ids"] = ()
        provenance["result_segment_keys"] = ()
        provenance["result_discontinuity_ids"] = ()
        provenance["state_at_boundary"] = "post-stage-7"
        return
    if not integration_segments:
        segment_ids = np.zeros(sample_times.size, dtype=np.int64)
        segment_keys = tuple("unsegmented:0" for _ in sample_times)
    else:
        starts = np.asarray(
            [float(item["t_start"]) for item in integration_segments],
            dtype=float,
        )
        segment_ids = np.searchsorted(starts, sample_times, side="right") - 1
        segment_ids = np.clip(
            segment_ids,
            0,
            len(integration_segments) - 1,
        ).astype(np.int64, copy=False)
        keys = tuple(str(item["segment_key"]) for item in integration_segments)
        segment_keys = tuple(keys[int(index)] for index in segment_ids)
    discontinuity_ids = np.searchsorted(
        np.asarray(tuple(discontinuities), dtype=float),
        sample_times,
        side="right",
    ).astype(np.int64, copy=False)
    provenance["result_segment_ids"] = tuple(
        int(value) for value in segment_ids
    )
    provenance["result_segment_keys"] = segment_keys
    provenance["result_discontinuity_ids"] = tuple(
        int(value) for value in discontinuity_ids
    )
    provenance["state_at_boundary"] = "post-stage-7"


def _record_pre_boundary_sample(
    records: list[dict[str, object]],
    *,
    boundary: float,
    state: np.ndarray,
    segment_identity: SegmentIdentity,
) -> None:
    projected, _projection = project_nonnegative_trajectory(
        np.asarray(state, dtype=float).reshape(-1, 1),
        context=f"Pre-boundary state at t={boundary:g}",
    )
    record = {
        "time": float(boundary),
        "segment_id": int(segment_identity.ordinal),
        "segment_key": segment_identity.key,
        "state": tuple(
            float(value)
            for value in np.asarray(projected, dtype=float).reshape(-1)
        ),
    }
    identity = (record["time"], record["segment_key"])
    if any(
        (existing["time"], existing["segment_key"]) == identity
        for existing in records
    ):
        raise InterventionScheduleError(
            "Duplicate exact pre-boundary segment identity."
        )
    records.append(record)


def _publish_pre_boundary_sample_identity(
    provenance: dict[str, object],
    records: Sequence[Mapping[str, object]],
    *,
    discontinuities: Sequence[float],
) -> None:
    if not records:
        provenance["pre_boundary_samples"] = ()
        return
    discontinuity_times = np.asarray(
        tuple(discontinuities),
        dtype=float,
    )
    payload: list[dict[str, object]] = []
    for record in records:
        time = float(record["time"])
        payload.append(
            {
                **dict(record),
                "discontinuity_id": int(
                    np.searchsorted(
                        discontinuity_times,
                        time,
                        side="left",
                    )
                ),
                "state_at_boundary": "pre-stage-4",
            }
        )
    provenance["pre_boundary_samples"] = tuple(payload)


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
    names = tuple(str(name) for name in (req.species_names or ()))
    if not names:
        raise InterventionScheduleError(
            "species_names are required when intervention_schedule is present."
        )
    available = set(names)
    referenced = {
        *(item.species for item in contract.instants),
        *(item.species for item in contract.fluxes),
        *(item.species for item in contract.holds),
        *(item.watch_species for item in contract.triggers),
        *(item.then.species for item in contract.triggers),
    }
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
    admitted_holds = tuple(
        replace(hold, start_time=boundary)
        for hold in active_holds
    )
    trigger_instants = tuple(
        _trigger_action_instant(contract.triggers[index], time=boundary)
        for index in trigger_indices
    )
    validate_boundary_conflicts(
        instants=static + trigger_instants,
        entering_holds=admitted_holds,
    )
    stage_four = tuple(
        item for item in static if item.phase == "before"
    )
    stage_five = tuple(
        sorted(
            (
                *(item for item in static if item.phase == "ordinary"),
                *trigger_instants,
            ),
            key=_instant_canonical_order,
        )
    )
    stage_seven = tuple(item for item in static if item.phase == "after")
    current = _apply_instants(
        current,
        stage_four,
        species_index=species_index,
        cancellation_check=cancellation_check,
    )
    current = _apply_instants(
        current,
        stage_five,
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
    if trigger_instants:
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
) -> tuple[int, int, int]:
    return (
        int(instant.declaration_ordinal),
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
            if starts_at_threshold and trigger.direction == "either"
            else (trigger.direction,)
        )
        for direction in directions:
            equality_side = (
                1.0
                if starts_at_threshold and direction == "rising"
                else (
                    -1.0
                    if starts_at_threshold and direction == "falling"
                    else 0.0
                )
            )

            def event(
                _time: float,
                state: np.ndarray,
                *,
                _species: int = species,
                _threshold: float = threshold,
                _equality_side: float = equality_side,
            ) -> float:
                value = float(state[_species]) - _threshold
                if value == 0.0 and _equality_side:
                    return float(np.nextafter(0.0, _equality_side))
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


def _terminal_user_event_hit(
    events: Sequence[Callable[[float, np.ndarray], float]],
    terminal: Sequence[bool],
    times: Sequence[Sequence[float]],
) -> bool:
    for index, values in enumerate(times):
        if not values or index >= len(terminal) or not terminal[index]:
            continue
        if bool(getattr(events[index], "_kindred_cancel_event", False)):
            continue
        return True
    return False


def _requested_eval(
    values: np.ndarray, *, start: float, end: float, is_final: bool
) -> np.ndarray:
    start_floor = float(np.nextafter(start, -np.inf))
    end_floor = float(np.nextafter(end, -np.inf))
    if is_final:
        mask = (values >= start_floor) & (
            values <= float(np.nextafter(end, np.inf))
        )
    else:
        mask = (values >= start_floor) & (values < end_floor)
    selected = np.asarray(values[mask], dtype=float).reshape(-1).copy()
    selected[
        (selected >= start_floor)
        & (selected <= float(np.nextafter(start, np.inf)))
    ] = start
    selected[
        (selected >= end_floor)
        & (selected <= float(np.nextafter(end, np.inf)))
    ] = end
    if selected.size > 1:
        selected = selected[
            np.concatenate([np.array([True]), np.diff(selected) > 0.0])
        ]
    return selected


def _append_samples(
    output_times: list[np.ndarray],
    output_states: list[np.ndarray],
    times: np.ndarray,
    states: np.ndarray,
) -> None:
    if not times.size:
        return
    emit_times = np.asarray(times, dtype=float).reshape(-1)
    emit_states = np.asarray(states, dtype=float)
    if output_times and output_times[-1].size:
        previous = float(output_times[-1][-1])
        if emit_times.size and emit_times[0] == previous:
            emit_times = emit_times[1:]
            emit_states = emit_states[:, 1:]
    if emit_times.size:
        output_times.append(emit_times)
        output_states.append(emit_states)


def _replace_boundary_sample(
    output_times: list[np.ndarray],
    output_states: list[np.ndarray],
    boundary: float,
    state: np.ndarray,
) -> None:
    for times, states in zip(
        reversed(output_times), reversed(output_states), strict=True
    ):
        if times.size and times[-1] == boundary:
            states[:, -1] = state
            return


def _contains_time(values: np.ndarray, time: float) -> bool:
    return bool(np.any(values == time))


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
