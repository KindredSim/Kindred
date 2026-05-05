from __future__ import annotations

import pickle

import pytest

pytestmark = pytest.mark.unit


def test_intervention_schedule_normalizes_payload_and_fingerprint() -> None:
    from kindred.core.intervention_schedule import (
        InterventionSchedule,
        coerce_intervention_schedule,
    )

    schedule = InterventionSchedule.from_payload(
        {
            "instant_events": [
                {"time": 2.0, "species": "A", "op": "add", "amount": 0.25},
                {"time": 1.0, "species": "A", "op": "set", "value": 3.0},
            ],
            "repeated_events": [
                {
                    "start": 4.0,
                    "every": 2.0,
                    "count": 2,
                    "species": "B",
                    "op": "add",
                    "amount": 0.5,
                }
            ],
            "intervals": [
                {"start": 1.0, "end": 3.0, "species": "B", "kind": "source", "rate": 0.1}
            ],
        }
    )

    payload = schedule.to_payload()

    assert payload["version"] == 1
    assert payload["instant_events"] == [
        {"time": 1.0, "species": "A", "op": "set", "value": 3.0},
        {"time": 2.0, "species": "A", "op": "add", "amount": 0.25},
        {"time": 4.0, "species": "B", "op": "add", "amount": 0.5},
        {"time": 6.0, "species": "B", "op": "add", "amount": 0.5},
    ]
    assert payload["intervals"] == [
        {"start": 1.0, "end": 3.0, "species": "B", "kind": "source", "rate": 0.1}
    ]
    assert schedule.fingerprint == coerce_intervention_schedule(payload).fingerprint
    pickle.dumps(payload)


def test_empty_intervention_schedule_object_normalizes_to_no_schedule() -> None:
    from kindred.core.intervention_schedule import (
        InterventionSchedule,
        coerce_intervention_schedule,
    )

    assert coerce_intervention_schedule(None) is None
    assert coerce_intervention_schedule({}) is None
    assert coerce_intervention_schedule(InterventionSchedule()) is None


def test_intervention_schedule_rejects_conflicting_absolute_events() -> None:
    from kindred.core.intervention_schedule import InterventionSchedule, InterventionScheduleError

    with pytest.raises(InterventionScheduleError, match="Conflicting absolute"):
        InterventionSchedule.from_payload(
            {
                "instant_events": [
                    {"time": 1.0, "species": "A", "op": "set", "value": 2.0},
                    {"time": 1.0, "species": "A", "op": "clear"},
                ]
            }
        )


def test_dsl_intervention_directives_build_core_schedule_metadata() -> None:
    from kindred.core.mechanism_metadata import MechanismMetadataKeys
    from kindred.core.simulator.dsl import parse_dsl_to_mechanism

    mechanism = parse_dsl_to_mechanism(
        "\n".join(
            [
                "reaction: A -> B; k=0.1",
                "initial: A=1.0",
                "initial: B=0.0",
                "intervention: op=set; species=A; time=1.0; value=3.0",
                "intervention: op=pulse; species=B; start=2.0; every=1.0; count=2; amount=0.25",
                "intervention: op=source; species=B; start=1.0; end=4.0; rate=0.5",
            ]
        )
    )

    schedule = mechanism.metadata[MechanismMetadataKeys.INTERVENTION_SCHEDULE]
    payload = schedule.to_payload()

    assert [event["time"] for event in payload["instant_events"]] == [1.0, 2.0, 3.0]
    assert payload["intervals"] == [
        {"start": 1.0, "end": 4.0, "species": "B", "kind": "source", "rate": 0.5}
    ]


def test_simulation_execution_request_round_trips_intervention_schedule_payload() -> None:
    from kindred.core.intervention_schedule import InterventionSchedule
    from kindred.core.simulation_plan import SimulationAlgebraPolicy, SimulationPlan
    from kindred.core.simulation_preparation import SimulationExecutionRequest

    schedule = InterventionSchedule.from_payload(
        {"instant_events": [{"time": 1.0, "species": "A", "op": "add", "amount": 2.0}]}
    )
    request = SimulationExecutionRequest(
        prepared_payload=None,
        initials={"A": 1.0},
        t_span=(0.0, 2.0),
        solver_config={"solver": "BDF", "grid": {"N": 3}},
        mechanism_text="reaction: A -> B; k=0.1",
        intervention_schedule=schedule,
    )
    plan = SimulationPlan.from_execution_request(
        request,
        execution_mode="explicit",
        algebra_policy=SimulationAlgebraPolicy.GUI_BEST_EFFORT,
    )

    restored = SimulationPlan.from_payload(plan.to_payload()).to_execution_request()

    assert restored.to_payload()["intervention_schedule"] == schedule.to_payload()


def test_simulation_plan_explicit_empty_schedule_stays_noop_over_dsl_text() -> None:
    from kindred.core.simulation_plan import SimulationAlgebraPolicy, SimulationPlan
    from kindred.core.simulation_preparation import (
        SimulationExecutionRequest,
        prepare_simulation_worker_run,
    )

    request = SimulationExecutionRequest(
        prepared_payload=None,
        initials={"A": 1.0, "B": 0.0},
        t_span=(0.0, 1.0),
        solver_config={"solver": "BDF", "grid": {"N": 3}},
        mechanism_text="\n".join(
            [
                "reaction: A -> B; k=0",
                "initial: A=1.0",
                "initial: B=0.0",
                "intervention: op=set; species=A; time=0.0; value=3.0",
            ]
        ),
        intervention_schedule={},
    )

    plan = SimulationPlan.from_execution_request(
        request,
        execution_mode="explicit",
        algebra_policy=SimulationAlgebraPolicy.GUI_BEST_EFFORT,
    )
    restored_payload = SimulationPlan.from_payload(plan.to_payload()).to_execution_request().to_payload()
    prepared = prepare_simulation_worker_run(execution_request=restored_payload)

    assert restored_payload["intervention_schedule"] == {}
    assert prepared.request.intervention_schedule is None


def test_prepared_runtime_reuse_clears_removed_intervention_schedule() -> None:
    from kindred.core.simulation_preparation import (
        SimulationExecutionRequest,
        prepare_simulation_worker_run,
        prepared_simulation_run_for_execution_request,
    )

    scheduled_text = "\n".join(
        [
            "reaction: A -> B; k=0",
            "initial: A=1.0",
            "initial: B=0.0",
            "intervention: op=set; species=A; time=0.0; value=2.0",
        ]
    )
    unscheduled_text = "\n".join(
        [
            "reaction: A -> B; k=0",
            "initial: A=1.0",
            "initial: B=0.0",
        ]
    )
    solver_config = {"solver": "BDF", "grid": {"N": 3}}
    prepared = prepare_simulation_worker_run(
        mechanism_text=scheduled_text,
        initials={"A": 1.0, "B": 0.0},
        t_span=(0.0, 2.0),
        solver_config=solver_config,
    )
    request = SimulationExecutionRequest(
        prepared_payload=None,
        initials={"A": 1.0, "B": 0.0},
        t_span=(0.0, 2.0),
        solver_config=solver_config,
        mechanism_text=unscheduled_text,
        intervention_schedule=None,
    )

    reused = prepared_simulation_run_for_execution_request(prepared, request)

    assert prepared.request.intervention_schedule is not None
    assert request.intervention_schedule is None
    assert reused.request.intervention_schedule is None


def test_execution_request_payload_makes_schedule_presence_explicit() -> None:
    from kindred.core.simulation_preparation import SimulationExecutionRequest

    scheduled_text = "\n".join(
        [
            "reaction: A -> B; k=0",
            "initial: A=1.0",
            "initial: B=0.0",
            "intervention: op=set; species=A; time=0.0; value=2.0",
        ]
    )
    unscheduled_text = "\n".join(
        [
            "reaction: A -> B; k=0",
            "initial: A=1.0",
            "initial: B=0.0",
        ]
    )

    scheduled_payload = SimulationExecutionRequest(
        prepared_payload=None,
        initials={"A": 1.0, "B": 0.0},
        t_span=(0.0, 2.0),
        solver_config={"solver": "BDF", "grid": {"N": 3}},
        mechanism_text=scheduled_text,
    ).to_payload()
    unscheduled_payload = SimulationExecutionRequest(
        prepared_payload=None,
        initials={"A": 1.0, "B": 0.0},
        t_span=(0.0, 2.0),
        solver_config={"solver": "BDF", "grid": {"N": 3}},
        mechanism_text=unscheduled_text,
    ).to_payload()

    assert scheduled_payload["intervention_schedule"]["instant_events"][0]["value"] == 2.0
    assert "intervention_schedule" not in unscheduled_payload


def test_execution_request_explicit_none_schedule_overrides_scheduled_mechanism_text() -> None:
    from kindred.core.simulation_preparation import SimulationExecutionRequest

    scheduled_text = "\n".join(
        [
            "reaction: A -> B; k=0",
            "initial: A=1.0",
            "initial: B=0.0",
            "intervention: op=set; species=A; time=0.0; value=2.0",
        ]
    )

    payload = SimulationExecutionRequest(
        prepared_payload=None,
        initials={"A": 1.0, "B": 0.0},
        t_span=(0.0, 2.0),
        solver_config={"solver": "BDF", "grid": {"N": 3}},
        mechanism_text=scheduled_text,
        intervention_schedule=None,
    ).to_payload()

    assert payload["intervention_schedule"] is None


def test_prepared_payload_schedule_is_preserved_when_request_has_no_schedule_authority() -> None:
    from kindred.core.simulation_preparation import (
        prepare_bound_mechanism,
        prepare_simulation_worker_run,
    )

    scheduled_text = "\n".join(
        [
            "reaction: A -> B; k=0",
            "initial: A=1.0",
            "initial: B=0.0",
            "intervention: op=set; species=A; time=0.0; value=2.0",
        ]
    )
    unscheduled_text = "\n".join(
        [
            "reaction: A -> B; k=0",
            "initial: A=1.0",
            "initial: B=0.0",
        ]
    )
    bound = prepare_bound_mechanism(
        scheduled_text,
        [],
        temperature_K=298.15,
        initials={"A": 1.0, "B": 0.0},
        use_advanced_dsl=True,
        wegscheider_cyclicity_enabled=False,
    )
    request_payload = {
        "version": 1,
        "prepared_payload": bound.as_serializable_execution_payload(),
        "initials": {"A": 1.0, "B": 0.0},
        "t_span": (0.0, 2.0),
        "solver_config": {"solver": "BDF", "grid": {"N": 3}},
        "mechanism_text": unscheduled_text,
    }

    prepared = prepare_simulation_worker_run(execution_request=request_payload)

    assert request_payload["prepared_payload"]["intervention_schedule"] is not None
    assert "intervention_schedule" not in request_payload
    assert prepared.request.intervention_schedule is not None
    assert prepared.request.intervention_schedule.to_payload()["instant_events"][0]["value"] == 2.0


def test_simulation_plan_round_trip_preserves_absent_schedule_authority() -> None:
    from kindred.core.simulation_plan import SimulationPlan

    payload = {
        "version": 1,
        "execution_mode": "explicit",
        "algebra_policy": "gui_best_effort",
        "execution_request": {
            "version": 1,
            "prepared_payload": None,
            "initials": {"A": 1.0, "B": 0.0},
            "t_span": (0.0, 2.0),
            "solver_config": {"solver": "BDF", "grid": {"N": 3}},
            "mechanism_text": "\n".join(
                [
                    "reaction: A -> B; k=0",
                    "initial: A=1.0",
                    "initial: B=0.0",
                ]
            ),
        },
    }

    restored_payload = SimulationPlan.from_payload(payload).to_payload()

    assert "intervention_schedule" not in restored_payload["execution_request"]


def test_prepared_payload_schedule_does_not_override_request_local_removal() -> None:
    from kindred.core.simulation_preparation import (
        SimulationExecutionRequest,
        prepare_bound_mechanism,
        prepare_simulation_worker_run,
    )

    scheduled_text = "\n".join(
        [
            "reaction: A -> B; k=0",
            "initial: A=1.0",
            "initial: B=0.0",
            "intervention: op=set; species=A; time=0.0; value=2.0",
        ]
    )
    unscheduled_text = "\n".join(
        [
            "reaction: A -> B; k=0",
            "initial: A=1.0",
            "initial: B=0.0",
        ]
    )
    bound = prepare_bound_mechanism(
        scheduled_text,
        [],
        temperature_K=298.15,
        initials={"A": 1.0, "B": 0.0},
        use_advanced_dsl=True,
        wegscheider_cyclicity_enabled=False,
    )
    request = SimulationExecutionRequest(
        prepared_payload=None,
        initials={"A": 1.0, "B": 0.0},
        t_span=(0.0, 2.0),
        solver_config={"solver": "BDF", "grid": {"N": 3}},
        mechanism_text=unscheduled_text,
        intervention_schedule=None,
    )

    prepared = prepare_simulation_worker_run(
        execution_request=request,
        prepared_payload=bound.as_serializable_execution_payload(),
    )

    assert bound.as_serializable_execution_payload()["intervention_schedule"] is not None
    assert prepared.request.intervention_schedule is None
