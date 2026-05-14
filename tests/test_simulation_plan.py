from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from kindred.core.simulation_preparation import SimulationExecutionRequest
from kindred.core.simulation_result_payload import build_simulation_success_payload


def test_text_request_round_trip_preserves_request_identity_and_policy() -> None:
    from kindred.core.simulation_plan import SimulationAlgebraPolicy, SimulationPlan

    request = SimulationExecutionRequest(
        prepared_payload=None,
        initials={"A": 1.25},
        t_span=(0.0, 12.5),
        solver_config={"solver": "BDF", "grid": {"N": 25}},
        mechanism_text="reaction: A -> B; k=1",
        simulation_identity={"schema_id": "schema", "param_fingerprint": "params"},
    )

    plan = SimulationPlan.from_execution_request(
        request,
        execution_mode="explicit",
        algebra_policy=SimulationAlgebraPolicy.GUI_BEST_EFFORT,
        cache_identity_payload={"cache_key": "explicit-cache"},
        metadata={"set_id": "set-1"},
    )

    payload = plan.to_payload()
    restored = SimulationPlan.from_payload(payload)

    assert restored.execution_mode == "explicit"
    assert restored.algebra_policy is SimulationAlgebraPolicy.GUI_BEST_EFFORT
    assert restored.cache_identity_payload == {"cache_key": "explicit-cache"}
    assert restored.metadata == {"set_id": "set-1"}
    assert restored.to_execution_request().to_payload() == request.to_payload()


def test_cache_identity_and_scope_helpers_return_defensive_payload_views() -> None:
    from kindred.core.simulation_plan import SimulationAlgebraPolicy, SimulationPlan

    request = SimulationExecutionRequest(
        prepared_payload=None,
        initials={"A": 1.25},
        t_span=(0.0, 12.5),
        solver_config={"solver": "BDF", "grid": {"N": 25}},
        mechanism_text="reaction: A -> B; k=1",
        simulation_identity={"schema_id": "schema", "param_fingerprint": "params"},
    )
    plan = SimulationPlan.from_execution_request(
        request,
        execution_mode="preview",
        algebra_policy=SimulationAlgebraPolicy.GUI_BEST_EFFORT,
        cache_identity_payload={
            "cache_key": "preview-cache",
            "simulation_identity": {"schema_id": "schema", "param_fingerprint": "params"},
            "preview_batch_cache_token": "preview-token",
        },
        cache_scope_payload={
            "scope_identity": {"schema_id": "scope", "queue_fingerprint": "queue"},
            "queue_ids": ["id1", "id2"],
        },
    )

    identity_payload = plan.simulation_identity_payload()
    scope_payload = plan.scope_identity_payload()
    queue_ids = plan.cache_queue_ids()

    assert plan.cache_key() == "preview-cache"
    assert plan.preview_batch_cache_token() == "preview-token"
    assert identity_payload == {"schema_id": "schema", "param_fingerprint": "params"}
    assert scope_payload == {"schema_id": "scope", "queue_fingerprint": "queue"}
    assert queue_ids == ("id1", "id2")

    identity_payload["schema_id"] = "mutated"
    scope_payload["schema_id"] = "mutated"
    assert plan.simulation_identity_payload() == {"schema_id": "schema", "param_fingerprint": "params"}
    assert plan.scope_identity_payload() == {"schema_id": "scope", "queue_fingerprint": "queue"}


def test_cache_identity_schedule_fingerprint_requires_execution_schedule_authority() -> None:
    from kindred.core.simulation_plan import SimulationAlgebraPolicy, SimulationPlan

    request = SimulationExecutionRequest(
        prepared_payload=None,
        initials={"A": 1.0},
        t_span=(0.0, 2.0),
        solver_config={"solver": "BDF"},
        mechanism_text="reaction: A -> B; k=1",
        simulation_identity={"schema_id": "schema"},
    )

    with pytest.raises(ValueError, match="intervention_schedule_fingerprint"):
        SimulationPlan.from_execution_request(
            request,
            execution_mode="explicit",
            algebra_policy=SimulationAlgebraPolicy.GUI_BEST_EFFORT,
            cache_identity_payload={
                "simulation_identity": {
                    "schema_id": "schema",
                    "intervention_schedule_fingerprint": "missing-schedule-authority",
                }
            },
        )


def test_prepared_request_round_trip_preserves_payload_and_copies_arrays_defensively() -> None:
    from kindred.core.simulation_plan import SimulationAlgebraPolicy, SimulationPlan

    y0 = np.array([1.0, 2.0], dtype=float)
    prepared_payload = {
        "version": 2,
        "mechanism": {"kind": "serializable"},
        "species_names": ["A", "B"],
        "y0": y0,
        "mechanism_text": "prepared text",
        "temperature_schedule": None,
        "bindings": {"k": 3.0},
    }
    request = SimulationExecutionRequest(
        prepared_payload=prepared_payload,
        initials={"A": 1.0, "B": 2.0},
        t_span=(0.0, 4.0),
        solver_config={"solver": "Radau"},
        mechanism_text="stale text ignored by prepared execution",
    )

    plan = SimulationPlan.from_execution_request(
        request,
        execution_mode="preview",
        algebra_policy=SimulationAlgebraPolicy.BATCH_BEST_EFFORT,
        cache_identity_payload={"preview_batch_cache_token": "token-1"},
        cache_scope_payload={"entries": [{"set_id": "set-1", "identity": {"cache": "scope"}}]},
    )

    y0[:] = [99.0, 100.0]
    prepared_payload["bindings"]["k"] = 9.0
    round_tripped = SimulationPlan.from_payload(plan.to_payload()).to_execution_request().to_payload()

    assert round_tripped["prepared_payload"]["species_names"] == ["A", "B"]
    assert round_tripped["prepared_payload"]["bindings"] == {"k": 3.0}
    np.testing.assert_allclose(round_tripped["prepared_payload"]["y0"], [1.0, 2.0])
    assert round_tripped["prepared_payload"]["y0"] is not y0


def test_malformed_plan_payload_fails_clearly() -> None:
    from kindred.core.simulation_plan import SimulationPlan

    with pytest.raises(ValueError, match="execution_request"):
        SimulationPlan.from_payload({"version": 1, "execution_mode": "explicit"})

    with pytest.raises(ValueError, match="algebra_policy"):
        SimulationPlan.from_payload(
            {
                "version": 1,
                "execution_mode": "explicit",
                "algebra_policy": "optimistic",
                "execution_request": {
                    "prepared_payload": None,
                    "initials": {},
                    "t_span": (0.0, 1.0),
                    "solver_config": {},
                },
            }
        )


def test_explicit_gui_style_plan_can_represent_absent_prepared_payload() -> None:
    from kindred.core.simulation_plan import SimulationAlgebraPolicy, SimulationPlan

    plan = SimulationPlan.from_payload(
        {
            "version": 1,
            "execution_mode": "explicit",
            "algebra_policy": "gui_best_effort",
            "execution_request": {
                "prepared_payload": None,
                "initials": {"A": 1.0},
                "t_span": (0.0, 1.0),
                "solver_config": {"solver": "BDF"},
                "mechanism_text": "reaction: A -> B; k=1",
            },
        }
    )

    assert plan.algebra_policy is SimulationAlgebraPolicy.GUI_BEST_EFFORT
    assert plan.to_execution_request().prepared_payload is None


def test_preview_style_plan_can_represent_prepared_payload_plus_preview_identity() -> None:
    from kindred.core.simulation_plan import SimulationAlgebraPolicy, SimulationPlan

    plan = SimulationPlan.from_payload(
        {
            "version": 1,
            "execution_mode": "preview",
            "algebra_policy": "gui_best_effort",
            "cache_identity_payload": {"preview_batch_cache_token": "preview-token"},
            "execution_request": {
                "prepared_payload": {
                    "version": 2,
                    "mechanism": {"kind": "serializable"},
                    "species_names": ["A"],
                    "y0": [2.0],
                    "mechanism_text": "prepared text",
                    "temperature_schedule": None,
                },
                "initials": {"A": 2.0},
                "t_span": (0.0, 2.0),
                "solver_config": {"solver": "BDF"},
                "mechanism_text": "",
            },
        }
    )

    assert plan.execution_mode == "preview"
    assert plan.algebra_policy is SimulationAlgebraPolicy.GUI_BEST_EFFORT
    assert plan.cache_identity_payload == {"preview_batch_cache_token": "preview-token"}
    np.testing.assert_allclose(plan.to_execution_request().to_payload()["prepared_payload"]["y0"], [2.0])


def test_fitting_style_plan_can_represent_serializable_prepared_payload_with_bindings() -> None:
    from kindred.core.simulation_plan import SimulationAlgebraPolicy, SimulationPlan

    plan = SimulationPlan.from_payload(
        {
            "version": 1,
            "execution_mode": "fitting",
            "algebra_policy": "fitting_strict",
            "metadata": {"requested_param_names": ["k"]},
            "execution_request": {
                "prepared_payload": {
                    "version": 2,
                    "mechanism": {"kind": "serializable"},
                    "species_names": ["A"],
                    "y0": [1.0],
                    "mechanism_text": "prepared text",
                    "temperature_schedule": None,
                    "bindings": {"k": 1.5},
                },
                "initials": {},
                "t_span": (0.0, 10.0),
                "solver_config": {"solver": "BDF", "grid": {"N": 20}},
                "mechanism_text": "reaction: A -> B; k=1",
            },
        }
    )

    payload = plan.to_execution_request().to_payload()
    assert plan.algebra_policy is SimulationAlgebraPolicy.FITTING_STRICT
    assert payload["prepared_payload"]["bindings"] == {"k": 1.5}
    assert payload["solver_config"]["grid"] == {"N": 20}


def test_result_wrapper_round_trips_success_payload_without_changing_shape() -> None:
    from kindred.core.simulation_plan import SimulationExecutionResult

    ode_result = SimpleNamespace(
        t=np.array([0.0, 1.0]),
        Y=np.array([[1.0, 0.5]]),
        nfev=7,
        provenance={"source": "unit"},
        fallback_occurred=False,
        fallback_message=None,
    )
    mechanism = object()
    payload = build_simulation_success_payload(
        result=ode_result,
        y=ode_result.Y,
        species_names=["A"],
        base_species_count=1,
        algebra_scalars={"obs": 1.0},
        algebra_errors=[],
        warnings=[{"kind": "preparation_warning"}],
        solver="BDF",
        mechanism_text="reaction: A -> B; k=1",
        solver_config={"solver": "BDF"},
        mechanism=mechanism,
        extra_fields={"set_id": "set-1"},
    )

    restored = SimulationExecutionResult.from_success_payload(payload).to_success_payload()

    assert restored.keys() == payload.keys()
    assert restored["mechanism"] is mechanism
    assert restored["set_id"] == "set-1"
    np.testing.assert_allclose(restored["t"], payload["t"])
    np.testing.assert_allclose(restored["Y"], payload["Y"])
    assert restored["species_names"] == ["A"]
    assert restored["warnings"] == [{"kind": "preparation_warning"}]
