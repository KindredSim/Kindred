from __future__ import annotations

import pickle
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from kindred.core.simulation_plan import SimulationAlgebraPolicy, SimulationPlan
from kindred.core.simulation_preparation import (
    SimulationExecutionRequest,
    prepare_bound_mechanism,
)
from kindred.core.simulator.solvers import SimulationRequest

pytestmark = pytest.mark.unit


def _identity_rhs(_t: float, y: np.ndarray) -> np.ndarray:
    return np.asarray(y, dtype=float)


def _normal_plan() -> SimulationPlan:
    mechanism_text = "\n".join(
        [
            "A -> B ; k=1.0",
            "initial: A=1.0",
            "initial: B=0.0",
        ]
    )
    bound = prepare_bound_mechanism(
        mechanism_text,
        [],
        temperature_K=298.15,
        initials={"A": 1.0, "B": 0.0},
        use_advanced_dsl=True,
        wegscheider_cyclicity_enabled=False,
    )
    request = SimulationExecutionRequest(
        prepared_payload=bound.as_serializable_execution_payload(),
        initials={"A": 1.0, "B": 0.0},
        t_span=(0.0, 1.0),
        solver_config={"solver": "BDF", "grid": {"N": 5}, "use_sparse_jacobian": False},
        mechanism_text=mechanism_text,
        simulation_identity={"schema_id": "normal", "param_fingerprint": "normal"},
    )
    return SimulationPlan.from_execution_request(
        request,
        execution_mode="explicit",
        algebra_policy=SimulationAlgebraPolicy.GUI_BEST_EFFORT,
        metadata={"case": "normal"},
    )


def _energy_scheduled_plan() -> SimulationPlan:
    mechanism_text = "\n".join(
        [
            "energy=kJ/mol",
            "temp_step: t=[0,0.5,1.0], T=[298.15,310.15]",
            "equilibrium: A <-> B; kf=10.0; kr=2.0; dG_eq=4.0",
            "initial: A=1.0",
            "initial: B=0.0",
        ]
    )
    bound = prepare_bound_mechanism(
        mechanism_text,
        ["dG_eq_fast__feq__A__B"],
        temperature_K=298.15,
        initials={"A": 1.0, "B": 0.0},
        use_advanced_dsl=True,
        wegscheider_cyclicity_enabled=False,
    )
    request = SimulationExecutionRequest(
        prepared_payload=bound.as_serializable_execution_payload(),
        initials={"A": 1.0, "B": 0.0},
        t_span=(0.0, 1.0),
        solver_config={"solver": "BDF", "grid": {"N": 5}, "use_sparse_jacobian": False},
        mechanism_text=mechanism_text,
        simulation_identity={"schema_id": "energy-scheduled", "param_fingerprint": "energy-scheduled"},
    )
    return SimulationPlan.from_execution_request(
        request,
        execution_mode="explicit",
        algebra_policy=SimulationAlgebraPolicy.GUI_BEST_EFFORT,
        metadata={"case": "energy-scheduled"},
    )


def _payload_copy_with_distinct_y0(
    payload: dict[str, Any],
    y0: np.ndarray | list[float] | None = None,
) -> dict[str, Any]:
    copied = dict(payload)
    copied["execution_request"] = dict(payload["execution_request"])
    prepared_payload = dict(copied["execution_request"]["prepared_payload"])
    source_y0 = prepared_payload["y0"] if y0 is None else y0
    prepared_payload["y0"] = np.array(source_y0, copy=True, dtype=float).reshape(-1)
    copied["execution_request"]["prepared_payload"] = prepared_payload
    return copied


def test_payload_boundary_rejects_simulation_request_instance():
    from kindred.core.simulation_containment import (
        SimulationContainmentPayloadError,
        validate_contained_simulation_payload,
    )

    request = SimulationRequest(
        rhs=_identity_rhs,
        t_span=(0.0, 1.0),
        y0=np.asarray([1.0], dtype=float),
        solver="BDF",
        grid={"N": 3},
    )

    with pytest.raises(SimulationContainmentPayloadError, match="SimulationRequest"):
        validate_contained_simulation_payload(request)


@pytest.mark.parametrize("field_name", ["rhs", "jacobian_func", "events", "progress_callback"])
def test_payload_boundary_rejects_callable_solver_boundary_fields(field_name: str):
    from kindred.core.simulation_containment import (
        SimulationContainmentPayloadError,
        validate_contained_simulation_payload,
    )

    payload: dict[str, Any] = _normal_plan().to_payload()
    execution_request = payload["execution_request"]
    prepared_payload = dict(execution_request["prepared_payload"])
    if field_name == "events":
        execution_request[field_name] = [lambda _t, _y: 1.0]
    elif field_name == "progress_callback":
        execution_request[field_name] = lambda _t, _start, _end: None
    else:
        prepared_payload[field_name] = lambda _t, y: y
        execution_request["prepared_payload"] = prepared_payload

    with pytest.raises(SimulationContainmentPayloadError, match=field_name):
        validate_contained_simulation_payload(payload)


def test_payload_boundary_rejects_version_1_prepared_payload_with_rhs():
    from kindred.core.simulation_containment import (
        SimulationContainmentPayloadError,
        validate_contained_simulation_payload,
    )

    payload = _normal_plan().to_payload()
    prepared_payload = dict(payload["execution_request"]["prepared_payload"])
    prepared_payload["version"] = 1
    prepared_payload["rhs"] = _identity_rhs
    payload["execution_request"]["prepared_payload"] = prepared_payload

    with pytest.raises(SimulationContainmentPayloadError, match="version-1 prepared payload"):
        validate_contained_simulation_payload(payload)


def test_payload_boundary_rejects_simulation_execution_request_instance_inside_mapping():
    from kindred.core.simulation_containment import (
        SimulationContainmentPayloadError,
        validate_contained_simulation_payload,
    )

    payload = _normal_plan().to_payload()
    payload["execution_request"] = SimulationExecutionRequest.from_mapping(payload["execution_request"])

    with pytest.raises(SimulationContainmentPayloadError, match="SimulationExecutionRequest"):
        validate_contained_simulation_payload(payload)


def test_payload_boundary_rejects_qt_like_object_without_importing_qt():
    from kindred.core.simulation_containment import (
        SimulationContainmentPayloadError,
        validate_contained_simulation_payload,
    )

    QtLike = type("QObject", (), {"__module__": "PySide6.QtCore"})
    payload = _normal_plan().to_payload()
    payload["metadata"]["qt_object"] = QtLike()

    with pytest.raises(SimulationContainmentPayloadError, match="Qt"):
        validate_contained_simulation_payload(payload)


def test_payload_boundary_rejects_unsafe_copied_plan_payload_state():
    from kindred.core.simulation_containment import (
        SimulationContainmentPayloadError,
        validate_contained_simulation_payload,
    )

    class Unpickleable:
        def __getstate__(self):
            raise TypeError("not safe for spawn")

    payload = _normal_plan().to_payload()
    payload["metadata"]["unsafe"] = Unpickleable()

    with pytest.raises(SimulationContainmentPayloadError, match="pickle"):
        validate_contained_simulation_payload(payload)


def test_build_contained_payload_accepts_and_pickles_normal_version_2_plan():
    from kindred.core.simulation_containment import build_contained_simulation_plan_payload

    payload = build_contained_simulation_plan_payload(_normal_plan())

    assert payload["execution_request"]["prepared_payload"]["version"] == 2
    pickle.dumps(payload)


def test_build_contained_payload_accepts_and_pickles_energy_scheduled_version_2_plan():
    from kindred.core.simulation_containment import build_contained_simulation_plan_payload

    payload = build_contained_simulation_plan_payload(_energy_scheduled_plan())

    assert payload["execution_request"]["prepared_payload"]["version"] == 2
    assert payload["execution_request"]["prepared_payload"]["temperature_schedule"] is not None
    pickle.dumps(payload)


def test_contained_payload_identity_treats_copied_numpy_y0_as_equal():
    from kindred.core.simulation_containment import contained_payloads_equal

    payload = _normal_plan().to_payload()
    equivalent = _payload_copy_with_distinct_y0(payload)

    assert payload["execution_request"]["prepared_payload"]["y0"] is not equivalent["execution_request"]["prepared_payload"]["y0"]
    assert contained_payloads_equal(payload, equivalent) is True


def test_contained_payload_identity_distinguishes_different_numpy_y0_without_raising():
    from kindred.core.simulation_containment import contained_payloads_equal

    payload = _normal_plan().to_payload()
    different = _payload_copy_with_distinct_y0(payload, [1.0, 0.5])

    assert contained_payloads_equal(payload, different) is False


def test_preview_owner_identity_matches_when_only_parameter_value_changes(monkeypatch):
    import kindred.core.simulation_containment as containment
    from kindred.core.simulation_containment import (
        _SimulationChildHandler,
        build_contained_simulation_plan_payload,
        contained_owner_payloads_match,
        contained_payloads_equal,
    )

    mechanism_text = "reaction: A -> B; k=1.0\ninitial: A=1.0\ninitial: B=0.0"
    bound = prepare_bound_mechanism(
        mechanism_text,
        ["k1"],
        temperature_K=298.15,
        initials={"A": 1.0, "B": 0.0},
        use_advanced_dsl=True,
        wegscheider_cyclicity_enabled=False,
    )
    owner_identity = {
        "version": 1,
        "execution_mode": "preview",
        "schema_id": "schema",
        "mechanism_text": mechanism_text,
        "solver_config": {"solver": "BDF", "grid": {"N": 5}, "use_sparse_jacobian": False},
        "t_end": 1.0,
        "set_id": "id1",
        "parameter_names": ["k1"],
    }

    def _plan(value: float, fingerprint: str) -> dict[str, Any]:
        request = SimulationExecutionRequest(
            prepared_payload=bound.as_serializable_execution_payload(),
            initials={"A": 1.0, "B": 0.0},
            t_span=(0.0, 1.0),
            solver_config={"solver": "BDF", "grid": {"N": 5}, "use_sparse_jacobian": False},
            mechanism_text=f"reaction: A -> B; k={float(value):g}\ninitial: A=1.0\ninitial: B=0.0",
            simulation_identity={"schema_id": "schema", "param_fingerprint": str(fingerprint)},
            parameter_overrides={"k1": float(value)},
        )
        plan = SimulationPlan.from_execution_request(
            request,
            execution_mode="preview",
            algebra_policy=SimulationAlgebraPolicy.GUI_BEST_EFFORT,
            metadata={"contained_owner_identity": dict(owner_identity)},
        )
        return build_contained_simulation_plan_payload(plan)

    startup_payload = _plan(1.0, "value-1")
    changed_payload = _plan(2.0, "value-2")
    prepare_calls = 0
    real_prepare = containment.prepare_simulation_worker_run

    def _counting_prepare(*args, **kwargs):
        nonlocal prepare_calls
        prepare_calls += 1
        return real_prepare(*args, **kwargs)

    monkeypatch.setattr(containment, "prepare_simulation_worker_run", _counting_prepare)
    handler = _SimulationChildHandler(startup_payload)

    prepared_request = handler._prepare_request({"simulation_plan_payload": changed_payload})

    assert contained_payloads_equal(startup_payload, changed_payload) is False
    assert contained_owner_payloads_match(startup_payload, changed_payload) is True
    assert prepare_calls == 1
    assert prepared_request.plan.to_execution_request().parameter_overrides == {"k1": 2.0}


def test_contained_owner_identity_distinguishes_mechanism_but_not_preview_parameter_value():
    from kindred.core.simulation_containment import contained_owner_payloads_match
    from kindred.core.simulation_identity import contained_simulation_owner_identity

    solver_config = {"solver": "BDF", "grid": {"N": 5}, "use_sparse_jacobian": False}
    first_identity = contained_simulation_owner_identity(
        execution_mode="preview",
        owner_mechanism_text="A -> B ; k=1.0",
        solver_config=solver_config,
        t_end=1.0,
        set_id="set-1",
        parameter_names=["k1"],
        simulation_identity={
            "version": 1,
            "schema_id": "schema-a",
            "param_fingerprint": "value-1",
            "solver": {
                "solver": "BDF",
                "rtol": 1e-6,
                "atol": 1e-12,
                "grid_n": 5,
                "temperature_K": 298.15,
                "use_sparse_jacobian": False,
                "wegscheider_cyclicity_enabled": False,
            },
            "t_end": 1.0,
            "preview_batch_cache_token": "",
            "execution_flags": ["fast_mode"],
        },
    )
    changed_value_identity = contained_simulation_owner_identity(
        execution_mode="preview",
        owner_mechanism_text="A -> B ; k=2.0",
        solver_config=solver_config,
        t_end=1.0,
        set_id="set-1",
        parameter_names=["k1"],
        simulation_identity={
            "version": 1,
            "schema_id": "schema-a-after-value-materialization",
            "param_fingerprint": "value-2",
            "solver": {
                "solver": "BDF",
                "rtol": 1e-6,
                "atol": 1e-12,
                "grid_n": 5,
                "temperature_K": 298.15,
                "use_sparse_jacobian": False,
                "wegscheider_cyclicity_enabled": False,
            },
            "t_end": 1.0,
            "preview_batch_cache_token": "",
            "execution_flags": ["fast_mode"],
        },
    )
    changed_mechanism_identity = contained_simulation_owner_identity(
        execution_mode="preview",
        owner_mechanism_text="A -> C ; k=1.0",
        solver_config=solver_config,
        t_end=1.0,
        set_id="set-1",
        parameter_names=["k1"],
        simulation_identity={
            "version": 1,
            "schema_id": "schema-b",
            "param_fingerprint": "value-1",
            "solver": {
                "solver": "BDF",
                "rtol": 1e-6,
                "atol": 1e-12,
                "grid_n": 5,
                "temperature_K": 298.15,
                "use_sparse_jacobian": False,
                "wegscheider_cyclicity_enabled": False,
            },
            "t_end": 1.0,
            "preview_batch_cache_token": "",
            "execution_flags": ["fast_mode"],
        },
    )

    first_payload = {"metadata": {"contained_owner_identity": first_identity}}
    changed_value_payload = {"metadata": {"contained_owner_identity": changed_value_identity}}
    changed_mechanism_payload = {"metadata": {"contained_owner_identity": changed_mechanism_identity}}

    assert contained_owner_payloads_match(first_payload, changed_value_payload) is True
    assert contained_owner_payloads_match(first_payload, changed_mechanism_payload) is False


def test_preview_owner_identity_includes_structural_semicolon_directives():
    from kindred.core.simulation_containment import contained_owner_payloads_match
    from kindred.core.simulation_identity import contained_simulation_owner_identity

    solver_config = {"solver": "BDF", "grid": {"N": 5}, "use_sparse_jacobian": False}
    base_identity = contained_simulation_owner_identity(
        execution_mode="preview",
        owner_mechanism_text="reaction: A -> B; A=1e12; Ea=50; fast",
        solver_config=solver_config,
        t_end=1.0,
        set_id="set-1",
        parameter_names=["k1"],
    )
    changed_directive_identity = contained_simulation_owner_identity(
        execution_mode="preview",
        owner_mechanism_text="reaction: A -> B; A=1e11; Ea=50; fast",
        solver_config=solver_config,
        t_end=1.0,
        set_id="set-1",
        parameter_names=["k1"],
    )
    changed_mutable_value_identity = contained_simulation_owner_identity(
        execution_mode="preview",
        owner_mechanism_text="reaction: A -> B; k=2.0",
        solver_config=solver_config,
        t_end=1.0,
        set_id="set-1",
        parameter_names=["k1"],
    )
    original_mutable_value_identity = contained_simulation_owner_identity(
        execution_mode="preview",
        owner_mechanism_text="reaction: A -> B; k=1.0",
        solver_config=solver_config,
        t_end=1.0,
        set_id="set-1",
        parameter_names=["k1"],
    )
    alias_canonicalized_value_identity = contained_simulation_owner_identity(
        execution_mode="preview",
        owner_mechanism_text="A -> B ; kf = 1.0",
        solver_config=solver_config,
        t_end=1.0,
        set_id="set-1",
        parameter_names=["k1"],
    )
    action_canonicalized_value_identity = contained_simulation_owner_identity(
        execution_mode="preview",
        owner_mechanism_text="A -> B ; k=2.0",
        solver_config=solver_config,
        t_end=1.0,
        set_id="set-1",
        parameter_names=["k1"],
    )
    keq_value_identity = contained_simulation_owner_identity(
        execution_mode="preview",
        owner_mechanism_text="A <-> B ; kf=1.0; Keq=3.0",
        solver_config=solver_config,
        t_end=1.0,
        set_id="set-1",
        parameter_names=["kf1", "kr1", "Keq1"],
    )
    changed_keq_value_identity = contained_simulation_owner_identity(
        execution_mode="preview",
        owner_mechanism_text="A <-> B ; kf=1.0; Keq=8.0",
        solver_config=solver_config,
        t_end=1.0,
        set_id="set-1",
        parameter_names=["kf1", "kr1", "Keq1"],
    )
    uppercase_rate_alias_identity = contained_simulation_owner_identity(
        execution_mode="preview",
        owner_mechanism_text="A <-> B ; KF=1.0; Kr=0.01",
        solver_config=solver_config,
        t_end=1.0,
        set_id="set-1",
        parameter_names=["kf1", "kr1"],
    )
    changed_uppercase_rate_alias_identity = contained_simulation_owner_identity(
        execution_mode="preview",
        owner_mechanism_text="A <-> B ; KF=2.0; Kr=0.01",
        solver_config=solver_config,
        t_end=1.0,
        set_id="set-1",
        parameter_names=["kf1", "kr1"],
    )
    structural_alias_identity = contained_simulation_owner_identity(
        execution_mode="preview",
        owner_mechanism_text="reaction: A -> B; a=1e12; ea=50",
        solver_config=solver_config,
        t_end=1.0,
        set_id="set-1",
        parameter_names=["a", "ea", "k1"],
    )
    changed_structural_alias_identity = contained_simulation_owner_identity(
        execution_mode="preview",
        owner_mechanism_text="reaction: A -> B; a=1e11; ea=50",
        solver_config=solver_config,
        t_end=1.0,
        set_id="set-1",
        parameter_names=["a", "ea", "k1"],
    )
    scalar_param_identity = contained_simulation_owner_identity(
        execution_mode="preview",
        owner_mechanism_text="A <-> B ; kf=1.0; kr=0.01\nparam a = 5\nparam kr1 = a*kf1",
        solver_config=solver_config,
        t_end=1.0,
        set_id="set-1",
        parameter_names=["a", "kf1", "kr1"],
    )
    changed_scalar_param_value_identity = contained_simulation_owner_identity(
        execution_mode="preview",
        owner_mechanism_text="A <-> B ; kf=1.0; kr=0.01\nparam a = 6\nparam kr1 = a*kf1",
        solver_config=solver_config,
        t_end=1.0,
        set_id="set-1",
        parameter_names=["a", "kf1", "kr1"],
    )
    changed_derived_param_expression_identity = contained_simulation_owner_identity(
        execution_mode="preview",
        owner_mechanism_text="A <-> B ; kf=1.0; kr=0.01\nparam a = 5\nparam kr1 = 2*a*kf1",
        solver_config=solver_config,
        t_end=1.0,
        set_id="set-1",
        parameter_names=["a", "kf1", "kr1"],
    )

    assert contained_owner_payloads_match(
        {"metadata": {"contained_owner_identity": base_identity}},
        {"metadata": {"contained_owner_identity": changed_directive_identity}},
    ) is False
    assert contained_owner_payloads_match(
        {"metadata": {"contained_owner_identity": original_mutable_value_identity}},
        {"metadata": {"contained_owner_identity": changed_mutable_value_identity}},
    ) is True
    assert contained_owner_payloads_match(
        {"metadata": {"contained_owner_identity": alias_canonicalized_value_identity}},
        {"metadata": {"contained_owner_identity": action_canonicalized_value_identity}},
    ) is True
    assert contained_owner_payloads_match(
        {"metadata": {"contained_owner_identity": keq_value_identity}},
        {"metadata": {"contained_owner_identity": changed_keq_value_identity}},
    ) is True
    assert contained_owner_payloads_match(
        {"metadata": {"contained_owner_identity": uppercase_rate_alias_identity}},
        {"metadata": {"contained_owner_identity": changed_uppercase_rate_alias_identity}},
    ) is True
    assert contained_owner_payloads_match(
        {"metadata": {"contained_owner_identity": structural_alias_identity}},
        {"metadata": {"contained_owner_identity": changed_structural_alias_identity}},
    ) is False
    assert contained_owner_payloads_match(
        {"metadata": {"contained_owner_identity": scalar_param_identity}},
        {"metadata": {"contained_owner_identity": changed_scalar_param_value_identity}},
    ) is True
    assert contained_owner_payloads_match(
        {"metadata": {"contained_owner_identity": scalar_param_identity}},
        {"metadata": {"contained_owner_identity": changed_derived_param_expression_identity}},
    ) is False


def test_prepare_only_request_updates_child_startup_payload_without_solving(monkeypatch):
    import kindred.core.simulation_containment as containment
    from kindred.core.simulation_containment import _SimulationChildHandler

    first_payload = _normal_plan().to_payload()
    second_payload = _energy_scheduled_plan().to_payload()
    prepare_calls = 0
    solve_calls = 0
    real_prepare = containment.prepare_simulation_worker_run

    def _counting_prepare(*args, **kwargs):
        nonlocal prepare_calls
        prepare_calls += 1
        return real_prepare(*args, **kwargs)

    def _counting_solve(*args, **kwargs):
        nonlocal solve_calls
        solve_calls += 1
        raise AssertionError("prepare-only request must not solve")

    monkeypatch.setattr(containment, "prepare_simulation_worker_run", _counting_prepare)
    monkeypatch.setattr(containment, "solve_ode", _counting_solve)
    handler = _SimulationChildHandler(first_payload)

    result = handler.handle_request(
        {"simulation_plan_payload": second_payload, "prepare_only": True},
        SimpleNamespace(request_id=1),
    )
    prepared_again = handler._prepare_request({"simulation_plan_payload": second_payload})

    assert result == {"success": True, "prepared": True}
    assert prepare_calls == 2
    assert solve_calls == 0
    assert prepared_again.execution_request["mechanism_text"] == second_payload["execution_request"]["mechanism_text"]
    assert prepare_calls == 2


def test_warm_simulation_owner_reprepares_running_runtime_without_replacing_process_owner():
    from kindred.core.simulation_containment import WarmSimulationOwner

    first_payload = _normal_plan().to_payload()
    second_payload = _energy_scheduled_plan().to_payload()

    class _RuntimeOwner:
        owner_epoch = 1
        is_running = True
        is_ready = True

        def __init__(self) -> None:
            self.solve_calls: list[dict[str, object]] = []

        def solve(self, payload, *, active_timeout_s, cancellation_check=None):
            self.solve_calls.append(
                {
                    "payload": dict(payload),
                    "active_timeout_s": float(active_timeout_s),
                    "cancelled": cancellation_check,
                }
            )
            return {"success": True, "prepared": True}

        def close(self, *, kill: bool = False) -> None:
            raise AssertionError("prepare must not replace or close the runtime owner")

    owner = WarmSimulationOwner(first_payload, active_timeout_s=12.0)
    runtime_owner = _RuntimeOwner()
    owner._runtime_owner = runtime_owner

    owner.prepare_runtime_payload(second_payload, wait=True)

    assert owner.simulation_plan_payload == second_payload
    assert runtime_owner.solve_calls == [
        {
            "payload": {
                "simulation_plan_payload": second_payload,
                "prepare_only": True,
            },
            "active_timeout_s": 12.0,
            "cancelled": None,
        }
    ]


def test_contained_worker_sends_exact_plan_even_when_owner_has_matching_startup_payload(qt_app):
    from kindred.gui.simulation_worker import ContainedSimulationWorker
    from kindred.core.simulation_containment import contained_payloads_equal

    _ = qt_app
    startup_payload = _normal_plan().to_payload()
    captured: dict[str, Any] = {}

    class _Owner:
        @property
        def simulation_plan_payload(self) -> dict[str, Any]:
            return dict(startup_payload)

        def solve(self, payload, *, cancellation_check):
            captured["payload"] = dict(payload)
            captured["cancelled_before_solve"] = bool(cancellation_check())
            return {
                "success": True,
                "t": np.asarray([0.0], dtype=float),
                "Y": np.asarray([[1.0]], dtype=float),
                "species_names": ["A"],
                "algebra_scalars": {},
                "algebra_errors": [],
            }

    worker = ContainedSimulationWorker(
        owner=_Owner(),
        simulation_plan_payload=startup_payload,
        include_mechanism_in_result_payload=False,
    )
    captured_results: list[dict[str, Any]] = []
    worker.result_ready.connect(lambda payload: captured_results.append(dict(payload)))

    worker.run()

    assert captured_results
    assert captured["cancelled_before_solve"] is False
    assert contained_payloads_equal(captured["payload"]["simulation_plan_payload"], startup_payload)
    assert captured["payload"]["include_mechanism_in_result_payload"] is False


def test_contained_worker_warm_owner_does_not_emit_cold_initializing_status(qt_app):
    from kindred.gui.simulation_worker import ContainedSimulationWorker

    _ = qt_app
    startup_payload = _normal_plan().to_payload()

    class _Owner:
        @property
        def is_running(self) -> bool:
            return True

        @property
        def simulation_plan_payload(self) -> dict[str, Any]:
            return dict(startup_payload)

        def solve(self, payload, *, cancellation_check):
            _ = payload
            assert cancellation_check() is False
            return {"success": True}

    worker = ContainedSimulationWorker(
        owner=_Owner(),
        simulation_plan_payload=startup_payload,
        include_mechanism_in_result_payload=False,
    )
    progress_messages: list[str] = []
    worker.progress.connect(lambda _percent, message: progress_messages.append(str(message)))

    worker.run()

    assert "Initializing simulation..." not in progress_messages
    assert progress_messages[0] == "Running simulation..."


def test_contained_worker_sends_exact_plan_for_equivalent_copied_numpy_y0(qt_app):
    from kindred.gui.simulation_worker import ContainedSimulationWorker
    from kindred.core.simulation_containment import contained_payloads_equal

    _ = qt_app
    startup_payload = _normal_plan().to_payload()
    worker_payload = _payload_copy_with_distinct_y0(startup_payload)
    captured: dict[str, Any] = {}

    class _Owner:
        @property
        def simulation_plan_payload(self) -> dict[str, Any]:
            return dict(startup_payload)

        def solve(self, payload, *, cancellation_check):
            captured["payload"] = dict(payload)
            return {"success": True}

    worker = ContainedSimulationWorker(
        owner=_Owner(),
        simulation_plan_payload=worker_payload,
        include_mechanism_in_result_payload=False,
    )
    captured_results: list[dict[str, Any]] = []
    captured_errors: list[object] = []
    worker.result_ready.connect(lambda payload: captured_results.append(dict(payload)))
    worker.error.connect(lambda payload: captured_errors.append(payload))

    worker.run()

    assert captured_errors == []
    assert captured_results == [{"success": True}]
    assert "simulation_plan_payload" in captured["payload"]
    assert contained_payloads_equal(captured["payload"]["simulation_plan_payload"], worker_payload)


def test_contained_worker_sends_changed_preview_plan_when_owner_identity_matches(qt_app):
    from kindred.core.simulation_containment import build_contained_simulation_plan_payload, contained_payloads_equal
    from kindred.gui.simulation_worker import ContainedSimulationWorker

    _ = qt_app
    owner_identity = {
        "version": 1,
        "execution_mode": "preview",
        "mechanism_text": "A -> B ; k=1.0",
        "solver_config": {"solver": "BDF", "grid": {"N": 5}, "use_sparse_jacobian": False},
        "t_end": 1.0,
        "set_id": "id1",
        "parameter_names": ["k1"],
    }

    def _plan(value: float) -> dict[str, Any]:
        request = SimulationExecutionRequest(
            prepared_payload=None,
            initials={"A": 1.0, "B": 0.0},
            t_span=(0.0, 1.0),
            solver_config={"solver": "BDF", "grid": {"N": 5}, "use_sparse_jacobian": False},
            mechanism_text=f"A -> B ; k={float(value):g}\ninitial: A=1.0\ninitial: B=0.0",
            simulation_identity={"schema_id": "schema", "param_fingerprint": str(value)},
            parameter_overrides={"k1": float(value)},
        )
        return build_contained_simulation_plan_payload(
            SimulationPlan.from_execution_request(
                request,
                execution_mode="preview",
                algebra_policy=SimulationAlgebraPolicy.GUI_BEST_EFFORT,
                metadata={"contained_owner_identity": dict(owner_identity)},
            )
        )

    startup_payload = _plan(1.0)
    changed_payload = _plan(2.0)
    captured: dict[str, Any] = {}

    class _Owner:
        @property
        def simulation_plan_payload(self) -> dict[str, Any]:
            return dict(changed_payload)

        def solve(self, payload, *, cancellation_check):
            captured["payload"] = dict(payload)
            return {"success": True}

    worker = ContainedSimulationWorker(
        owner=_Owner(),
        simulation_plan_payload=changed_payload,
        include_mechanism_in_result_payload=False,
    )

    worker.run()

    sent_payload = captured["payload"]["simulation_plan_payload"]
    sent_request = SimulationPlan.from_payload(sent_payload).to_execution_request().to_payload()
    assert sent_request["mechanism_text"].startswith("A -> B ; k=2")
    assert sent_request["parameter_overrides"] == {"k1": 2.0}
    assert not contained_payloads_equal(sent_payload, startup_payload)


def test_contained_worker_includes_plan_for_different_copied_numpy_y0(qt_app):
    from kindred.gui.simulation_worker import ContainedSimulationWorker

    _ = qt_app
    startup_payload = _normal_plan().to_payload()
    worker_payload = _payload_copy_with_distinct_y0(startup_payload, [1.0, 0.5])
    captured: dict[str, Any] = {}

    class _Owner:
        @property
        def simulation_plan_payload(self) -> dict[str, Any]:
            return dict(startup_payload)

        def solve(self, payload, *, cancellation_check):
            captured["payload"] = dict(payload)
            return {"success": True}

    worker = ContainedSimulationWorker(
        owner=_Owner(),
        simulation_plan_payload=worker_payload,
        include_mechanism_in_result_payload=False,
    )
    captured_errors: list[object] = []
    worker.error.connect(lambda payload: captured_errors.append(payload))

    worker.run()

    assert captured_errors == []
    assert "simulation_plan_payload" in captured["payload"]


def test_contained_worker_closes_owner_once_when_cancelled_before_solve(qt_app):
    from kindred.gui.simulation_worker import ContainedSimulationWorker

    _ = qt_app
    startup_payload = _normal_plan().to_payload()

    class _Owner:
        def __init__(self) -> None:
            self.close_calls: list[bool] = []
            self.solve_calls = 0

        @property
        def simulation_plan_payload(self) -> dict[str, Any]:
            return dict(startup_payload)

        def solve(self, payload, *, cancellation_check):
            self.solve_calls += 1
            return {"success": True}

        def close(self, *, kill: bool = False) -> None:
            self.close_calls.append(bool(kill))

    owner = _Owner()
    worker = ContainedSimulationWorker(owner=owner, simulation_plan_payload=startup_payload)
    worker.cancel()
    captured_errors: list[dict[str, Any]] = []
    worker.error.connect(lambda payload: captured_errors.append(dict(payload)))

    worker.run()

    assert owner.solve_calls == 0
    assert [error["kind"] for error in captured_errors] == ["cancelled"]
    assert owner.close_calls == [True]
    worker.cleanup()
    assert owner.close_calls == [True]


def test_contained_worker_closes_owner_once_when_solve_reports_cancelled(qt_app):
    from kindred.core.exceptions import SimulationCancelled
    from kindred.gui.simulation_worker import ContainedSimulationWorker

    _ = qt_app
    startup_payload = _normal_plan().to_payload()

    class _Owner:
        def __init__(self) -> None:
            self.close_calls: list[bool] = []

        @property
        def simulation_plan_payload(self) -> dict[str, Any]:
            return dict(startup_payload)

        def solve(self, payload, *, cancellation_check):
            raise SimulationCancelled()

        def close(self, *, kill: bool = False) -> None:
            self.close_calls.append(bool(kill))

    owner = _Owner()
    worker = ContainedSimulationWorker(owner=owner, simulation_plan_payload=startup_payload)
    captured_errors: list[dict[str, Any]] = []
    worker.error.connect(lambda payload: captured_errors.append(dict(payload)))

    worker.run()

    assert [error["kind"] for error in captured_errors] == ["cancelled"]
    assert owner.close_calls == [True]
    worker.cleanup()
    assert owner.close_calls == [True]


@pytest.mark.parametrize(
    ("exception_name", "expected_phase", "expected_timeout_s"),
    [
        ("startup", "startup", 0.125),
        ("accept", "accept", 0.25),
    ],
)
def test_contained_worker_serializes_pre_active_timeouts_as_timeout_failures(
    qt_app,
    exception_name: str,
    expected_phase: str,
    expected_timeout_s: float,
):
    from kindred.core.simulation_containment import (
        SimulationContainmentAcceptTimeout,
        SimulationContainmentStartupTimeout,
    )
    from kindred.gui.simulation_worker import ContainedSimulationWorker

    _ = qt_app
    startup_payload = _normal_plan().to_payload()

    class _Owner:
        def __init__(self) -> None:
            self.close_calls: list[bool] = []

        @property
        def simulation_plan_payload(self) -> dict[str, Any]:
            return dict(startup_payload)

        def solve(self, payload, *, cancellation_check):
            _ = (payload, cancellation_check)
            if exception_name == "startup":
                raise SimulationContainmentStartupTimeout(expected_timeout_s)
            raise SimulationContainmentAcceptTimeout(expected_timeout_s)

        def close(self, *, kill: bool = False) -> None:
            self.close_calls.append(bool(kill))

    owner = _Owner()
    worker = ContainedSimulationWorker(owner=owner, simulation_plan_payload=startup_payload)
    captured_errors: list[dict[str, Any]] = []
    worker.error.connect(lambda payload: captured_errors.append(dict(payload)))

    worker.run()

    assert owner.close_calls == [True]
    assert [error["kind"] for error in captured_errors] == ["timeout"]
    assert captured_errors[0]["details"]["timeout_phase"] == expected_phase
    assert captured_errors[0]["details"][f"{expected_phase}_timeout_s"] == pytest.approx(expected_timeout_s)
