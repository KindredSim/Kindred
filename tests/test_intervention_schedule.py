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
    ]
    assert payload["repeated_events"] == [
        {
            "species": "B",
            "op": "add",
            "count": 2,
            "start": 4.0,
            "every": 2.0,
            "amount": 0.5,
        },
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


def test_intervention_schedule_rejects_absolute_and_delta_events_without_ordering_policy() -> None:
    from kindred.core.intervention_schedule import InterventionSchedule, InterventionScheduleError

    with pytest.raises(InterventionScheduleError, match="Cannot combine absolute and add/remove"):
        InterventionSchedule.from_payload(
            {
                "instant_events": [
                    {"time": 1.0, "species": "A", "op": "set", "value": 2.0},
                    {"time": 1.0, "species": "A", "op": "add", "amount": 0.5},
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
                "intervention: op=trigger; trigger_species=A; threshold=0.5; direction=falling; action=add; species=B; amount=0.1; max_count=1; min_interval=0.0",
                "intervention: op=source; species=B; start=1.0; end=4.0; rate=0.5",
            ]
        )
    )

    schedule = mechanism.metadata[MechanismMetadataKeys.INTERVENTION_SCHEDULE]
    payload = schedule.to_payload()

    assert [event["time"] for event in payload["instant_events"]] == [1.0]
    assert payload["repeated_events"] == [
        {
            "species": "B",
            "op": "pulse",
            "count": 2,
            "start": 2.0,
            "every": 1.0,
            "amount": 0.25,
        }
    ]
    assert payload["trigger_events"] == [
        {
            "trigger_species": "A",
            "direction": "falling",
            "species": "B",
            "action": "add",
            "max_count": 1,
            "min_interval": 0.0,
            "threshold": 0.5,
            "amount": 0.1,
        }
    ]
    assert payload["intervals"] == [
        {"start": 1.0, "end": 4.0, "species": "B", "kind": "source", "rate": 0.5}
    ]


def test_dsl_intervention_directives_preserve_parameterized_schedule_fields() -> None:
    from kindred.core.mechanism_metadata import MechanismMetadataKeys
    from kindred.core.simulator.dsl import parse_dsl_to_mechanism

    mechanism = parse_dsl_to_mechanism(
        "\n".join(
            [
                "reaction: A -> B; k=0.1",
                "initial: A=1.0",
                "initial: B=0.0",
                "intervention: op=add; species=A; time_param=pulse_time; amount_param=dose",
                "intervention: op=pulse; species=A; start_param=pulse_start; every_param=pulse_gap; count=2; amount_param=pulse_amount",
                "intervention: op=source; species=B; start=1.0; end_param=stop_time; rate_param=feed",
            ]
        )
    )

    schedule = mechanism.metadata[MechanismMetadataKeys.INTERVENTION_SCHEDULE]
    payload = schedule.to_payload()

    assert payload["instant_events"] == [
        {"time_param": "pulse_time", "species": "A", "op": "add", "amount_param": "dose"}
    ]
    assert payload["repeated_events"] == [
        {
            "species": "A",
            "op": "pulse",
            "count": 2,
            "start_param": "pulse_start",
            "every_param": "pulse_gap",
            "amount_param": "pulse_amount",
        }
    ]
    assert payload["intervals"] == [
        {
            "start": 1.0,
            "end_param": "stop_time",
            "species": "B",
            "kind": "source",
            "rate_param": "feed",
        }
    ]


@pytest.mark.parametrize(
    ("directive", "param_name"),
    [
        ("intervention: op=add; species=A; time_param=K1; amount=1.0", "K1"),
        ("intervention: op=set; species=A; time=1.0; value_param=K1", "K1"),
        ("intervention: op=pulse; species=A; start_param=K1; every=1.0; count=1; amount=1.0", "K1"),
        ("intervention: op=pulse; species=A; start=1.0; every_param=K1; count=1; amount=1.0", "K1"),
        ("intervention: op=source; species=A; start_param=K1; end=2.0; rate=1.0", "K1"),
        ("intervention: op=source; species=A; start=0.0; end_param=K1; rate=1.0", "K1"),
        ("intervention: op=source; species=A; start=0.0; end=2.0; rate_param=K1", "K1"),
        ("intervention: op=reservoir; species=A; start=0.0; end=2.0; value_param=K1", "K1"),
        (
            "intervention: op=trigger; trigger_species=A; direction=falling; threshold_param=K1; species=B; action=add; amount=1.0; max_count=1; min_interval=0.0",
            "K1",
        ),
    ],
)
def test_schedule_param_fields_reject_indexed_k_on_reversible_only_step(directive: str, param_name: str) -> None:
    from kindred.core.mechanism_metadata import MechanismMetadataKeys
    from kindred.core.simulation_preparation import partition_simulation_parameter_values
    from kindred.core.simulator.dsl import parse_dsl_to_mechanism

    mechanism = parse_dsl_to_mechanism(
        "\n".join(
            [
                "equilibrium: A <-> B; kf=1.0; kr=0.5",
                "initial: A=1.0",
                "initial: B=0.0",
                directive,
            ]
        ),
        initials={},
    )
    schedule = mechanism.metadata[MechanismMetadataKeys.INTERVENTION_SCHEDULE]

    partition = partition_simulation_parameter_values(
        mechanism=mechanism,
        parameter_overrides={param_name: 2.0},
        unresolved_intervention_schedule=schedule,
    )

    message = partition.invalid_parameter_identifier_messages[param_name]
    assert "not a valid indexed parameter identifier" in message
    assert "kf1" in message
    assert "kr1" in message
    assert "Keq1" in message


def test_schedule_longer_non_exact_indexed_like_name_remains_ordinary_parameter() -> None:
    from kindred.core.mechanism_metadata import MechanismMetadataKeys
    from kindred.core.simulation_preparation import partition_simulation_parameter_values
    from kindred.core.simulator.dsl import parse_dsl_to_mechanism

    mechanism = parse_dsl_to_mechanism(
        "\n".join(
            [
                "equilibrium: A <-> B; kf=1.0; kr=0.5",
                "initial: A=1.0",
                "initial: B=0.0",
                "intervention: op=add; species=A; time=1.0; amount_param=dose_K1",
            ]
        ),
        initials={},
    )
    schedule = mechanism.metadata[MechanismMetadataKeys.INTERVENTION_SCHEDULE]

    partition = partition_simulation_parameter_values(
        mechanism=mechanism,
        parameter_overrides={"dose_K1": 2.0},
        unresolved_intervention_schedule=schedule,
    )

    assert partition.invalid_parameter_identifier_messages == {}
    assert partition.schedule_only_parameter_names == frozenset({"dose_K1"})


def test_schedule_scalar_shared_parameter_is_additive_not_schedule_only() -> None:
    from kindred.core.mechanism_metadata import MechanismMetadataKeys
    from kindred.core.simulation_preparation import prepare_bound_mechanism, partition_simulation_parameter_values

    bound = prepare_bound_mechanism(
        mechanism_text="\n".join(
            [
                "reaction: A -> B; k=1.0",
                "initial: A=1.0",
                "initial: B=0.0",
                "param scale = 1.0",
                "param k1 = scale",
                "intervention: op=add; species=A; time=1.0; amount_param=scale",
            ]
        ),
        param_names=["scale"],
        temperature_K=298.15,
        initials={},
        use_advanced_dsl=True,
        wegscheider_cyclicity_enabled=False,
    )
    schedule = bound.mechanism.metadata[MechanismMetadataKeys.INTERVENTION_SCHEDULE]

    partition = partition_simulation_parameter_values(
        mechanism=bound.mechanism,
        parameter_overrides={"scale": 2.0},
        unresolved_intervention_schedule=schedule,
    )

    assert partition.invalid_parameter_identifier_messages == {}
    assert partition.schedule_parameter_names == frozenset({"scale"})
    assert partition.scalar_parameter_names == frozenset({"scale"})
    assert partition.schedule_only_parameter_names == frozenset()
    assert partition.mechanism_binding_values["scale"] == pytest.approx(2.0)
    assert partition.schedule_resolution_values["scale"] == pytest.approx(2.0)


def test_schedule_declared_scalar_shared_parameter_is_additive_not_schedule_only() -> None:
    from kindred.core.mechanism_metadata import MechanismMetadataKeys
    from kindred.core.simulation_preparation import partition_simulation_parameter_values
    from kindred.core.simulator.dsl import parse_dsl_to_mechanism

    mechanism = parse_dsl_to_mechanism(
        "\n".join(
            [
                "reaction: A -> B; k=1.0",
                "initial: A=1.0",
                "initial: B=0.0",
                "intervention: op=add; species=A; time=1.0; amount_param=scale",
            ]
        ),
        initials={},
    )
    schedule = mechanism.metadata[MechanismMetadataKeys.INTERVENTION_SCHEDULE]

    partition = partition_simulation_parameter_values(
        mechanism=mechanism,
        parameter_overrides={"scale": 2.0},
        unresolved_intervention_schedule=schedule,
        scalar_parameter_names={"scale"},
    )

    assert partition.invalid_parameter_identifier_messages == {}
    assert partition.schedule_parameter_names == frozenset({"scale"})
    assert partition.scalar_parameter_names == frozenset({"scale"})
    assert partition.schedule_only_parameter_names == frozenset()
    assert partition.mechanism_binding_values["scale"] == pytest.approx(2.0)
    assert partition.schedule_resolution_values["scale"] == pytest.approx(2.0)


def test_schedule_runtime_shared_parameter_is_additive_not_schedule_only() -> None:
    from kindred.core.mechanism_metadata import MechanismMetadataKeys
    from kindred.core.simulation_preparation import partition_simulation_parameter_values
    from kindred.core.simulator.dsl import parse_dsl_to_mechanism

    mechanism = parse_dsl_to_mechanism(
        "\n".join(
            [
                "reaction: A -> B; k=1.0",
                "initial: A=1.0",
                "initial: B=0.0",
                "intervention: op=add; species=A; time=1.0; amount_param=runtime_scale",
            ]
        ),
        initials={},
    )
    schedule = mechanism.metadata[MechanismMetadataKeys.INTERVENTION_SCHEDULE]

    partition = partition_simulation_parameter_values(
        mechanism=mechanism,
        parameter_overrides={"runtime_scale": 2.0},
        unresolved_intervention_schedule=schedule,
        runtime_parameter_names={"runtime_scale"},
    )

    assert partition.invalid_parameter_identifier_messages == {}
    assert partition.schedule_parameter_names == frozenset({"runtime_scale"})
    assert partition.runtime_parameter_names == frozenset({"runtime_scale"})
    assert partition.schedule_only_parameter_names == frozenset()
    assert partition.mechanism_binding_values["runtime_scale"] == pytest.approx(2.0)
    assert partition.schedule_resolution_values["runtime_scale"] == pytest.approx(2.0)


def test_parameterized_intervention_schedule_requires_request_parameter_values_before_solve() -> None:
    from kindred.core.simulation_preparation import (
        SimulationExecutionRequest,
        SimulationPreparationError,
        prepare_simulation_worker_run,
    )

    request = SimulationExecutionRequest(
        prepared_payload=None,
        initials={"A": 1.0, "B": 0.0},
        t_span=(0.0, 2.0),
        solver_config={"solver": "BDF", "grid": {"N": 3}},
        mechanism_text="\n".join(
            [
                "reaction: A -> B; k=0",
                "initial: A=1.0",
                "initial: B=0.0",
                "intervention: op=add; species=A; time=1.0; amount_param=dose",
            ]
        ),
        intervention_schedule={
            "instant_events": [
                {"time": 1.0, "species": "A", "op": "add", "amount_param": "dose"}
            ]
        },
    )

    with pytest.raises(SimulationPreparationError, match="Missing intervention schedule parameter: dose"):
        prepare_simulation_worker_run(execution_request=request)


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


def test_prepared_runtime_reuse_preserves_schedule_when_request_has_absent_schedule_authority() -> None:
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
    solver_config = {"solver": "BDF", "grid": {"N": 3}}
    prepared = prepare_simulation_worker_run(
        mechanism_text=scheduled_text,
        initials={"A": 1.0, "B": 0.0},
        t_span=(0.0, 2.0),
        solver_config=solver_config,
    )
    request = SimulationExecutionRequest(
        prepared_payload=None,
        initials={"A": 0.5, "B": 0.0},
        t_span=(0.0, 2.0),
        solver_config=solver_config,
        mechanism_text=scheduled_text,
    )

    reused = prepared_simulation_run_for_execution_request(prepared, request)

    assert request.has_intervention_schedule_authority is False
    assert "intervention_schedule" not in request.to_payload()
    assert prepared.request.intervention_schedule is not None
    assert reused.unresolved_intervention_schedule is not None
    assert reused.unresolved_intervention_schedule.to_payload() == prepared.request.intervention_schedule.to_payload()
    assert reused.request.intervention_schedule is not None
    assert reused.request.intervention_schedule.to_payload() == prepared.request.intervention_schedule.to_payload()
    assert reused.request.y0[0] == pytest.approx(0.5)


def test_prepared_runtime_reuse_replaces_schedule_before_parameter_resolution() -> None:
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
        mechanism_text=scheduled_text,
        parameter_overrides={"dose": 4.0},
        intervention_schedule={
            "instant_events": [
                {"time": 1.0, "species": "A", "op": "add", "amount_param": "dose"}
            ]
        },
    )

    reused = prepared_simulation_run_for_execution_request(prepared, request)

    assert request.has_intervention_schedule_authority is True
    assert reused.unresolved_intervention_schedule is not prepared.unresolved_intervention_schedule
    assert reused.unresolved_intervention_schedule.to_payload()["instant_events"] == [
        {"time": 1.0, "species": "A", "op": "add", "amount_param": "dose"}
    ]
    assert reused.request.intervention_schedule.to_payload()["instant_events"] == [
        {"time": 1.0, "species": "A", "op": "add", "amount": 4.0}
    ]


@pytest.mark.parametrize(
    ("mechanism_lines", "include_schedule_key", "intervention_schedule", "expected_payload_value"),
    [
        (
            [
                "reaction: A -> B; k=0",
                "initial: A=1.0",
                "initial: B=0.0",
                "intervention: op=set; species=A; time=0.0; value=2.0",
            ],
            False,
            None,
            None,
        ),
        (
            [
                "reaction: A -> B; k=0",
                "initial: A=1.0",
                "initial: B=0.0",
            ],
            False,
            None,
            None,
        ),
        (
            [
                "reaction: A -> B; k=0",
                "initial: A=1.0",
                "initial: B=0.0",
                "intervention: op=set; species=A; time=0.0; value=2.0",
            ],
            True,
            None,
            None,
        ),
    ],
)
def test_execution_request_payload_serializes_only_explicit_schedule_authority(
    mechanism_lines,
    include_schedule_key,
    intervention_schedule,
    expected_payload_value,
) -> None:
    from kindred.core.simulation_preparation import SimulationExecutionRequest

    kwargs = {}
    if include_schedule_key:
        kwargs["intervention_schedule"] = intervention_schedule

    payload = SimulationExecutionRequest(
        prepared_payload=None,
        initials={"A": 1.0, "B": 0.0},
        t_span=(0.0, 2.0),
        solver_config={"solver": "BDF", "grid": {"N": 3}},
        mechanism_text="\n".join(mechanism_lines),
        **kwargs,
    ).to_payload()

    if include_schedule_key:
        assert payload["intervention_schedule"] == expected_payload_value
    else:
        assert "intervention_schedule" not in payload


def test_unscheduled_execution_request_does_not_inherit_prepared_payload_schedule() -> None:
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
    assert prepared.request.intervention_schedule is None


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
