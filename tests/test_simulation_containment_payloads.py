from __future__ import annotations

import pickle
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


def test_contained_worker_omits_plan_from_request_when_owner_has_matching_startup_payload(qt_app):
    from kindred.gui.simulation_worker import ContainedSimulationWorker

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
    assert "simulation_plan_payload" not in captured["payload"]
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


def test_contained_worker_omits_plan_for_equivalent_copied_numpy_y0(qt_app):
    from kindred.gui.simulation_worker import ContainedSimulationWorker

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
    assert "simulation_plan_payload" not in captured["payload"]


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
