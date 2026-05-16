import numpy as np
import pytest

from kindred.core.exceptions import SimulationCancelled
from kindred.core.intervention_schedule import (
    InterventionInterval,
    InterventionSchedule,
    InterventionTriggerEvent,
)
from kindred.core.simulator import solvers
from kindred.core.simulator.intervention_schedule_execution import (
    InterventionScheduleExecutionOwner,
    ScheduleExecutionRequest,
    SegmentRunResult,
)


pytestmark = pytest.mark.unit


def _base_request(
    *,
    schedule: InterventionSchedule,
    y0: np.ndarray | None = None,
    t_eval: np.ndarray | None = None,
    species_names: tuple[str, ...] = ("A",),
    **overrides,
) -> ScheduleExecutionRequest:
    req = solvers.SimulationRequest(
        rhs=lambda _t, y: -np.asarray(y, dtype=float),
        t_span=(0.0, 1.0),
        y0=np.asarray([1.0] if y0 is None else y0, dtype=float),
        t_eval=np.asarray([0.0, 0.5, 1.0] if t_eval is None else t_eval, dtype=float),
        intervention_schedule=schedule,
        species_names=species_names,
        **overrides,
    )
    return ScheduleExecutionRequest(
        request=req,
        rhs=lambda _t, y: -np.asarray(y, dtype=float),
        rhs_for_jac=lambda _t, y: -np.asarray(y, dtype=float),
        t0=0.0,
        t1=1.0,
        y0=np.asarray(req.y0, dtype=float),
        t_eval=np.asarray(req.t_eval, dtype=float),
        provenance={
            "solver_requested": "BDF",
            "has_intervention_schedule": True,
            "jacobian_sparsity_hint": req.jac_sparsity is not None,
        },
        method="BDF",
        note=None,
        schedule=schedule,
    )


class LedgerSegmentRunner:
    def __init__(self):
        self.requests = []
        self.results = []

    def queue(self, result):
        self.results.append(result)
        return self

    def __call__(self, segment_request):
        self.requests.append(segment_request)
        if isinstance(self.results[0], BaseException):
            raise self.results.pop(0)
        result = self.results.pop(0)
        if callable(result):
            return result(segment_request)
        return result


def _segment_result(
    segment_request,
    *,
    t: np.ndarray | None = None,
    y: np.ndarray | None = None,
    event_times=(),
    event_states=(),
    fallback_occurred: bool = False,
    fallback_message: str | None = None,
    solver_alternative_used: str | None = None,
) -> SegmentRunResult:
    t_out = np.asarray(segment_request.t_eval if t is None else t, dtype=float).reshape(-1)
    y0 = np.asarray(segment_request.y0, dtype=float).reshape(-1)
    if y is None:
        Y = np.repeat(y0.reshape(-1, 1), repeats=t_out.size, axis=1)
    else:
        Y = np.asarray(y, dtype=float)
    provenance = dict(segment_request.provenance)
    provenance["solver_used"] = "BDF"
    provenance["symbolic_jacobian"] = segment_request.request.jacobian_func is not None
    provenance["jacobian_sparsity_hint"] = segment_request.request.jac_sparsity is not None
    if solver_alternative_used:
        provenance["solver_alternative_used"] = solver_alternative_used
    return SegmentRunResult(
        output=solvers.SimulationOutput(
            t=t_out,
            Y=Y,
            provenance=provenance,
            fallback_occurred=fallback_occurred,
            fallback_message=fallback_message,
        ),
        event_times=event_times,
        event_states=event_states,
        solver_used="BDF",
        solver_alternative_used=solver_alternative_used,
        fallback_occurred=fallback_occurred,
        fallback_message=fallback_message,
        symbolic_jacobian_used=segment_request.request.jacobian_func is not None,
        jacobian_sparsity_hint=segment_request.request.jac_sparsity is not None,
    )


def test_schedule_owner_rewrites_segment_requests_and_disables_active_interval_jacobian_policy():
    sparsity = np.asarray([[True]], dtype=bool)
    schedule = InterventionSchedule(
        intervals=(
            InterventionInterval(start=0.0, end=0.5, species="A", kind="source", rate=2.0),
        )
    )
    runner = LedgerSegmentRunner().queue(_segment_result).queue(_segment_result)
    request = _base_request(
        schedule=schedule,
        jacobian_func=lambda _t, y: np.asarray([[-1.0]], dtype=float),
        jac_sparsity=sparsity,
        first_step=0.75,
    )

    out = InterventionScheduleExecutionOwner(runner).execute(request)

    assert len(runner.requests) == 2
    active_segment, inactive_segment = runner.requests
    assert active_segment.t0 == 0.0
    assert active_segment.t1 == 0.5
    assert active_segment.request.jacobian_func is None
    assert active_segment.request.jac_sparsity is None
    assert active_segment.request.first_step is None
    assert inactive_segment.t0 == 0.5
    assert inactive_segment.t1 == 1.0
    assert inactive_segment.request.jacobian_func is request.request.jacobian_func
    assert inactive_segment.request.jac_sparsity is sparsity
    assert inactive_segment.request.first_step is None
    assert out.provenance["intervention_segments"] == 2
    assert out.provenance["intervention_segment_jacobian_sparsity_hints"] == [False, True]
    assert out.provenance["intervention_jacobian_sparsity_hint_partially_disabled"] is True


def test_schedule_owner_uses_typed_event_states_to_resume_from_trigger_time():
    schedule = InterventionSchedule(
        trigger_events=(
            InterventionTriggerEvent(
                trigger_species="A",
                threshold=0.5,
                direction="rising",
                species="B",
                op="add",
                amount=3.0,
                max_count=1,
                min_interval=0.0,
            ),
        )
    )

    def first_segment(segment_request):
        event_times = (tuple([0.4]),)
        event_states = ((np.asarray([0.5, 0.0], dtype=float),),)
        return _segment_result(
            segment_request,
            t=np.asarray([0.0, 0.4], dtype=float),
            y=np.asarray([[0.0, 0.5], [0.0, 0.0]], dtype=float),
            event_times=event_times,
            event_states=event_states,
        )

    runner = LedgerSegmentRunner().queue(first_segment).queue(_segment_result)
    request = _base_request(
        schedule=schedule,
        y0=np.asarray([0.0, 0.0]),
        t_eval=np.asarray([0.0, 0.4, 1.0]),
        species_names=("A", "B"),
    )

    out = InterventionScheduleExecutionOwner(runner).execute(request)

    assert len(runner.requests) == 2
    resumed = runner.requests[1]
    assert resumed.t0 == pytest.approx(0.4)
    np.testing.assert_allclose(resumed.y0, np.asarray([0.5, 3.0]))
    assert out.provenance["intervention_trigger_events"] == [
        {"time": 0.4, "trigger_species": "A", "species": "B", "action": "add"}
    ]


def test_schedule_owner_aggregates_typed_segment_fallback_metadata():
    schedule = InterventionSchedule(
        intervals=(
            InterventionInterval(start=0.0, end=0.5, species="A", kind="source", rate=1.0),
        )
    )

    def fallback_segment(segment_request):
        return _segment_result(
            segment_request,
            fallback_occurred=True,
            fallback_message="BDF failed; succeeded with Radau",
            solver_alternative_used="Radau",
        )

    runner = LedgerSegmentRunner().queue(fallback_segment).queue(_segment_result)
    request = _base_request(schedule=schedule)

    out = InterventionScheduleExecutionOwner(runner).execute(request)

    assert out.fallback_occurred is True
    assert out.fallback_message == "BDF failed; succeeded with Radau"
    assert out.provenance["solver_alternative_used"] == "Radau"
    assert out.provenance["intervention_segment_solvers"] == ["BDF", "BDF"]


def test_schedule_owner_propagates_cancellation_without_treating_it_as_schedule_terminal_stop():
    schedule = InterventionSchedule(
        intervals=(
            InterventionInterval(start=0.0, end=0.5, species="A", kind="source", rate=1.0),
        )
    )
    runner = LedgerSegmentRunner().queue(SimulationCancelled())
    request = _base_request(schedule=schedule)

    with pytest.raises(SimulationCancelled):
        InterventionScheduleExecutionOwner(runner).execute(request)


def test_solve_ode_delegates_scheduled_requests_to_schedule_execution_owner(monkeypatch):
    schedule = InterventionSchedule(
        intervals=(
            InterventionInterval(start=0.0, end=0.5, species="A", kind="source", rate=1.0),
        )
    )
    captured = {}

    class FakeOwner:
        def __init__(self, segment_runner):
            captured["segment_runner"] = segment_runner

        def execute(self, execution_request):
            captured["execution_request"] = execution_request
            return solvers.SimulationOutput(
                t=np.asarray(execution_request.t_eval, dtype=float),
                Y=np.repeat(np.asarray(execution_request.y0, dtype=float).reshape(-1, 1), 3, axis=1),
                provenance=dict(execution_request.provenance),
            )

    monkeypatch.setattr(solvers, "InterventionScheduleExecutionOwner", FakeOwner, raising=False)

    out = solvers.solve_ode(
        solvers.SimulationRequest(
            rhs=lambda _t, y: -np.asarray(y, dtype=float),
            t_span=(0.0, 1.0),
            y0=np.asarray([1.0]),
            t_eval=np.asarray([0.0, 0.5, 1.0]),
            intervention_schedule=schedule,
            species_names=("A",),
        )
    )

    assert out.t.tolist() == [0.0, 0.5, 1.0]
    assert captured["execution_request"].schedule is schedule
    assert captured["execution_request"].method == "BDF"
    assert callable(captured["segment_runner"])
