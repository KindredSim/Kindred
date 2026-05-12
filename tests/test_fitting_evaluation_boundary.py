from __future__ import annotations

import pickle

import numpy as np
import pytest

pytestmark = pytest.mark.unit


def _assert_execution_request_payloads_equal(left: dict, right: dict) -> None:
    left_prepared = dict(left.get("prepared_payload") or {})
    right_prepared = dict(right.get("prepared_payload") or {})
    left_y0 = left_prepared.pop("y0", None)
    right_y0 = right_prepared.pop("y0", None)
    left_prepared.pop("mechanism", None)
    right_prepared.pop("mechanism", None)
    for prepared in (left_prepared, right_prepared):
        bindings = prepared.get("bindings")
        if isinstance(bindings, dict):
            normalized_bindings = {}
            for name, binding in bindings.items():
                if isinstance(binding, dict):
                    binding_name = binding.get("name", name)
                    binding_value = binding.get("value")
                elif isinstance(binding, (int, float)):
                    binding_name = name
                    binding_value = binding
                else:
                    binding_name = getattr(binding, "name", name)
                    binding_value = getattr(binding, "value", None)
                if binding_value is None and callable(binding):
                    binding_value = binding()
                if binding_value is None:
                    binding_value = 0.0
                normalized_bindings[str(name)] = {
                    "name": str(binding_name),
                    "value": float(binding_value),
                }
            prepared["bindings"] = normalized_bindings
    assert {**left, "prepared_payload": left_prepared} == {**right, "prepared_payload": right_prepared}
    if left_y0 is not None or right_y0 is not None:
        np.testing.assert_allclose(left_y0, right_y0)



def test_prepare_fitting_execution_context_uses_serializable_execution_request_payload() -> None:
    from kindred.core.fitting_evaluation import prepare_fitting_execution_context

    mechanism_text = "\n".join(
        [
            "reaction: A -> B; k=0.2",
            "initial: A=1.0",
            "initial: B=0.0",
        ]
    )
    context = prepare_fitting_execution_context(
        mechanism_text=mechanism_text,
        param_names=["k1"],
        t_end=1.0,
        num_points=5,
        solver="BDF",
        rtol=1e-6,
        atol=1e-12,
        initial_prefix="init:",
    )

    payload = context.execution_request.to_payload()

    assert payload["prepared_payload"] is not None
    assert "rhs" not in payload["prepared_payload"]
    assert "bindings" in payload["prepared_payload"]
    pickle.dumps(payload)


def test_prepare_fitting_execution_context_carries_authoritative_fitting_plan() -> None:
    from kindred.core.fitting_evaluation import prepare_fitting_execution_context
    from kindred.core.simulation_plan import SimulationAlgebraPolicy, SimulationPlan

    mechanism_text = "\n".join(
        [
            "reaction: A -> B; k=0.2",
            "initial: A=1.0",
            "initial: B=0.0",
        ]
    )
    context = prepare_fitting_execution_context(
        mechanism_text=mechanism_text,
        param_names=["k1"],
        t_end=1.0,
        num_points=5,
        solver="BDF",
        rtol=1e-6,
        atol=1e-12,
        initial_prefix="init:",
    )

    assert isinstance(context.simulation_plan, SimulationPlan)
    assert context.simulation_plan.execution_mode == "fitting"
    assert context.simulation_plan.algebra_policy is SimulationAlgebraPolicy.FITTING_STRICT
    assert context.execution_request is context.simulation_plan.execution_request
    _assert_execution_request_payloads_equal(
        context.execution_request.to_payload(),
        context.simulation_plan.execution_request.to_payload(),
    )


def test_prepare_fitting_execution_context_preserves_solver_flags_in_execution_request_and_metadata() -> None:
    from kindred.core.fitting_evaluation import prepare_fitting_execution_context
    from kindred.core.mechanism_metadata import MechanismMetadataKeys

    mechanism_text = "\n".join(
        [
            "reaction: A -> B; k=0.2",
            "initial: A=1.0",
            "initial: B=0.0",
        ]
    )
    context = prepare_fitting_execution_context(
        mechanism_text=mechanism_text,
        param_names=["k1"],
        t_end=1.0,
        num_points=5,
        solver="RADAU",
        rtol=1e-7,
        atol=1e-13,
        use_sparse_jacobian=True,
        wegscheider_cyclicity_enabled=True,
        initial_prefix="init:",
    )

    solver_config = context.execution_request.to_payload()["solver_config"]

    assert solver_config["solver"] == context.prepared_metadata.solver_normalized
    assert solver_config["use_sparse_jacobian"] is True
    assert solver_config[MechanismMetadataKeys.WEGSCHEIDER_CYCLICITY_ENABLED] is True
    assert context.prepared_metadata.solver_normalized.lower() == "radau"
    assert context.prepared_metadata.use_sparse_jacobian is True
    assert context.prepared_metadata.wegscheider_cyclicity_enabled is True


def test_serial_fitting_evaluator_updates_metadata_with_actual_symbolic_jacobian_identity() -> None:
    from kindred.core.fitting_evaluation import SerialFittingEvaluator, prepare_fitting_execution_context

    mechanism_text = "\n".join(
        [
            "reaction: A -> B; k=0.2",
            "initial: A=1.0",
            "initial: B=0.0",
        ]
    )
    context = prepare_fitting_execution_context(
        mechanism_text=mechanism_text,
        param_names=[],
        t_end=1.0,
        num_points=5,
        solver="BDF",
        rtol=1e-6,
        atol=1e-12,
        use_sparse_jacobian=True,
        initial_prefix="init:",
    )
    evaluator = SerialFittingEvaluator(context)

    _ = evaluator({"init:A": 1.0})

    identity = evaluator.prepared_metadata.symbolic_jacobian_identity
    assert identity is not None
    assert identity["kind"] == "jacobian"
    assert identity["fingerprint"]


def test_serial_fitting_process_payload_includes_actual_symbolic_jacobian_identity() -> None:
    from kindred.core.fitting_evaluation import SerialFittingEvaluator, prepare_fitting_execution_context

    mechanism_text = "\n".join(
        [
            "reaction: A -> B; k=0.2",
            "initial: A=1.0",
            "initial: B=0.0",
        ]
    )
    context = prepare_fitting_execution_context(
        mechanism_text=mechanism_text,
        param_names=[],
        t_end=1.0,
        num_points=5,
        solver="BDF",
        rtol=1e-6,
        atol=1e-12,
        use_sparse_jacobian=True,
        initial_prefix="init:",
    )
    evaluator = SerialFittingEvaluator(context)

    payload = evaluator.to_process_payload()

    identity = payload["prepared_metadata"]["symbolic_jacobian_identity"]
    assert identity["kind"] == "jacobian"
    assert identity["artifact_fingerprint"]


def test_serial_fitting_evaluator_rebinds_symbolic_jacobian_per_candidate(monkeypatch) -> None:
    import kindred.core.fitting_evaluation as fitting_evaluation
    from kindred.core.fitting_evaluation import SerialFittingEvaluator, prepare_fitting_execution_context
    from kindred.core.simulator.solvers import SimulationOutput

    mechanism_text = "\n".join(
        [
            "reaction: A -> B; k=0.2",
            "initial: A=1.0",
            "initial: B=0.0",
        ]
    )
    context = prepare_fitting_execution_context(
        mechanism_text=mechanism_text,
        param_names=["k1"],
        t_end=1.0,
        num_points=5,
        solver="BDF",
        rtol=1e-6,
        atol=1e-12,
        use_sparse_jacobian=True,
        initial_prefix="init:",
    )
    evaluator = SerialFittingEvaluator(context)
    identities: list[dict] = []
    jacobian_values: list[np.ndarray] = []

    def _capture_request(request):
        identity = getattr(request.jacobian_func, "_kindred_symbolic_jacobian_identity", None)
        identities.append(dict(identity or {}))
        assert callable(request.jacobian_func)
        jacobian_values.append(
            np.asarray(request.jacobian_func(0.0, np.asarray([2.0, 0.0], dtype=float)), dtype=float)
        )
        t = np.asarray(request.t_eval, dtype=float)
        y0 = np.asarray(request.y0, dtype=float).reshape(-1)
        return SimulationOutput(
            t=t,
            Y=np.vstack([np.full_like(t, value) for value in y0]),
            provenance={},
        )

    monkeypatch.setattr(fitting_evaluation, "_solve_request", _capture_request)

    evaluator({"k1": 0.2})
    evaluator({"k1": 0.7})

    assert identities[0]["parameter_symbols"] == ["k1"]
    assert identities[0]["structure_fingerprint"] == identities[1]["structure_fingerprint"]
    assert identities[0]["evaluation_snapshot_fingerprint"] != identities[1]["evaluation_snapshot_fingerprint"]
    np.testing.assert_allclose(jacobian_values[0], [[-0.2, 0.0], [0.2, 0.0]])
    np.testing.assert_allclose(jacobian_values[1], [[-0.7, 0.0], [0.7, 0.0]])


def test_serial_fitting_solver_request_carries_symbolic_wegscheider_identity(monkeypatch) -> None:
    import kindred.core.fitting_evaluation as fitting_evaluation
    from kindred.core.fitting_evaluation import SerialFittingEvaluator, prepare_fitting_execution_context
    from kindred.core.simulator.solvers import SimulationOutput

    mechanism_text = "\n".join(
        [
            "equilibrium: A <-> B; kf=2.0; K=2.0",
            "equilibrium: B <-> C; kf=3.0; K=3.0",
            "equilibrium: C <-> A; kf=1.0; K=1.0",
            "param Keq3 = 1 / (Keq1 * Keq2)",
            "init: A=1.0, B=0.0, C=0.0",
        ]
    )
    context = prepare_fitting_execution_context(
        mechanism_text=mechanism_text,
        param_names=[],
        t_end=1.0,
        num_points=3,
        solver="BDF",
        rtol=1e-6,
        atol=1e-12,
        use_sparse_jacobian=True,
        wegscheider_cyclicity_enabled=True,
        initial_prefix="init:",
    )
    evaluator = SerialFittingEvaluator(context)
    captured = {}

    def _capture_request(request):
        captured["symbolic_wegscheider_identity"] = request.symbolic_wegscheider_identity
        t = np.linspace(float(request.t_span[0]), float(request.t_span[1]), 3)
        y0 = np.asarray(request.y0, dtype=float).reshape(-1)
        return SimulationOutput(
            t=t,
            Y=np.vstack([np.full_like(t, value) for value in y0]),
            provenance={},
        )

    monkeypatch.setattr(fitting_evaluation, "_solve_request", _capture_request)

    evaluator({})

    identity = captured["symbolic_wegscheider_identity"]
    assert identity is not None
    assert identity["kind"] == "wegscheider_cyclicity"
    assert identity["fingerprint"]


def test_prepare_fitting_execution_context_rejects_dependent_wegscheider_keq_parameter() -> None:
    from kindred.core.exceptions import FitSimulationError
    from kindred.core.fitting_evaluation import prepare_fitting_execution_context

    mechanism_text = "\n".join(
        [
            "equilibrium: A <-> B; kf=2.0; K=2.0",
            "equilibrium: B <-> C; kf=3.0; K=3.0",
            "equilibrium: C <-> A; kf=1.0; K=1.0",
            "param Keq3 = 1 / (Keq1 * Keq2)",
            "init: A=1.0, B=0.0, C=0.0",
        ]
    )

    with pytest.raises(FitSimulationError) as excinfo:
        prepare_fitting_execution_context(
            mechanism_text=mechanism_text,
            param_names=["Keq1", "Keq2", "Keq3"],
            t_end=1.0,
            num_points=5,
            solver="BDF",
            rtol=1e-6,
            atol=1e-12,
            use_sparse_jacobian=True,
            wegscheider_cyclicity_enabled=True,
            initial_prefix="init:",
        )

    assert "Dependent equilibrium parameter" in str(excinfo.value)


def test_serial_fitting_evaluator_runs_from_structured_context() -> None:
    from kindred.core.fitting_evaluation import SerialFittingEvaluator, prepare_fitting_execution_context

    mechanism_text = "\n".join(
        [
            "reaction: A -> B; k=0.2",
            "initial: A=1.0",
            "initial: B=0.0",
        ]
    )
    context = prepare_fitting_execution_context(
        mechanism_text=mechanism_text,
        param_names=["k1"],
        t_end=1.0,
        num_points=6,
        solver="BDF",
        rtol=1e-6,
        atol=1e-12,
        initial_prefix="init:",
    )
    evaluator = SerialFittingEvaluator(context).with_fixed_params({"k1": 0.2})

    result = evaluator({"init:A": 1.0})

    assert np.asarray(result.t, dtype=float).size == 6
    assert set(result.species) == {"A", "B"}
    assert np.asarray(result.species["A"], dtype=float).shape == (6,)


def test_serial_fitting_evaluator_applies_parameterized_schedule_amount_per_candidate() -> None:
    from kindred.core.fitting_evaluation import SerialFittingEvaluator, prepare_fitting_execution_context

    mechanism_text = "\n".join(
        [
            "reaction: A -> B; k=0",
            "initial: A=1.0",
            "initial: B=0.0",
            "intervention: op=add; species=A; time=1.0; amount_param=dose",
        ]
    )
    context = prepare_fitting_execution_context(
        mechanism_text=mechanism_text,
        param_names=["dose"],
        t_end=2.0,
        num_points=3,
        solver="BDF",
        rtol=1e-6,
        atol=1e-12,
        initial_prefix="init:",
    )
    evaluator = SerialFittingEvaluator(context)

    low = evaluator({"dose": 1.0})
    high = evaluator({"dose": 3.0})

    assert float(np.asarray(low.species["A"], dtype=float)[-1]) == pytest.approx(2.0, abs=1e-6)
    assert float(np.asarray(high.species["A"], dtype=float)[-1]) == pytest.approx(4.0, abs=1e-6)


def test_serial_fitting_evaluator_applies_parameterized_state_trigger_per_candidate() -> None:
    from kindred.core.fitting_evaluation import SerialFittingEvaluator, prepare_fitting_execution_context

    mechanism_text = "\n".join(
        [
            "reaction: A -> B; k=1.0",
            "reaction: C -> D; k=0",
            "initial: A=1.0",
            "initial: B=0.0",
            "initial: C=0.0",
            "initial: D=0.0",
            "intervention: op=trigger; trigger_species=A; threshold=0.5; direction=falling; action=add; species=C; amount_param=trigger_dose; max_count=1; min_interval=0.0",
        ]
    )
    context = prepare_fitting_execution_context(
        mechanism_text=mechanism_text,
        param_names=["trigger_dose"],
        t_end=2.0,
        num_points=9,
        solver="BDF",
        rtol=1e-6,
        atol=1e-12,
        initial_prefix="init:",
    )
    evaluator = SerialFittingEvaluator(context)

    low = evaluator({"trigger_dose": 1.0})
    high = evaluator({"trigger_dose": 3.0})

    assert float(np.asarray(low.species["C"], dtype=float)[-1]) == pytest.approx(1.0, abs=1e-6)
    assert float(np.asarray(high.species["C"], dtype=float)[-1]) == pytest.approx(3.0, abs=1e-6)


def test_serial_fitting_evaluator_preserves_runtime_param_precedence_over_fixed_params() -> None:
    from kindred.core.fitting_evaluation import SerialFittingEvaluator, prepare_fitting_execution_context

    mechanism_text = "\n".join(
        [
            "reaction: A -> B; k=0.2",
            "initial: A=1.0",
            "initial: B=0.0",
        ]
    )
    context = prepare_fitting_execution_context(
        mechanism_text=mechanism_text,
        param_names=["k1"],
        t_end=1.0,
        num_points=6,
        solver="BDF",
        rtol=1e-6,
        atol=1e-12,
        initial_prefix="init:",
    )
    base = SerialFittingEvaluator(context)
    expected = base({"k1": 0.8, "init:A": 1.0})
    actual = base.with_fixed_params({"k1": 0.1})({"k1": 0.8, "init:A": 1.0})

    assert np.allclose(np.asarray(actual.species["B"], dtype=float), np.asarray(expected.species["B"], dtype=float))


def test_serial_fitting_evaluator_with_fixed_params_keeps_evaluators_isolated() -> None:
    from kindred.core.fitting_evaluation import SerialFittingEvaluator, prepare_fitting_execution_context

    mechanism_text = "\n".join(
        [
            "reaction: A -> B; k=0.2",
            "initial: A=1.0",
            "initial: B=0.0",
        ]
    )
    context = prepare_fitting_execution_context(
        mechanism_text=mechanism_text,
        param_names=["k1"],
        t_end=1.0,
        num_points=6,
        solver="BDF",
        rtol=1e-6,
        atol=1e-12,
        initial_prefix="init:",
    )
    base = SerialFittingEvaluator(context)
    derived = base.with_fixed_params({"k1": 0.2})

    assert base.context is not derived.context
    assert base.context.simulation_plan is not derived.context.simulation_plan
    assert base.context.execution_request is not derived.context.execution_request
    assert base.context.execution_request is base.context.simulation_plan.execution_request
    assert derived.context.execution_request is derived.context.simulation_plan.execution_request
    assert base.context.execution_request.prepared_payload is not derived.context.execution_request.prepared_payload
    assert (
        base.context.execution_request.prepared_payload["bindings"]
        is not derived.context.execution_request.prepared_payload["bindings"]
    )

    baseline = base({"k1": 0.8, "init:A": 1.0})
    _ = derived({"k1": 0.2, "init:A": 1.0})
    repeat = base({"k1": 0.8, "init:A": 1.0})

    assert np.allclose(
        np.asarray(baseline.species["B"], dtype=float),
        np.asarray(repeat.species["B"], dtype=float),
    )


def test_serial_fitting_evaluator_lane_clone_keeps_context_and_bindings_isolated() -> None:
    from kindred.core.fitting_evaluation import (
        SerialFittingEvaluator,
        _clone_fitting_series_evaluator_lane,
        prepare_fitting_execution_context,
    )

    mechanism_text = "\n".join(
        [
            "reaction: A -> B; k=0.2",
            "initial: A=1.0",
            "initial: B=0.0",
        ]
    )
    context = prepare_fitting_execution_context(
        mechanism_text=mechanism_text,
        param_names=["k1"],
        t_end=1.0,
        num_points=6,
        solver="BDF",
        rtol=1e-6,
        atol=1e-12,
        initial_prefix="init:",
    )
    base = SerialFittingEvaluator(context).with_fixed_params({"k1": 0.1})
    lane = _clone_fitting_series_evaluator_lane(base)

    assert lane is not None
    assert lane is not base
    assert lane.context is not base.context
    assert lane.context.simulation_plan is not base.context.simulation_plan
    assert lane.context.execution_request is not base.context.execution_request
    assert lane.context.execution_request is lane.context.simulation_plan.execution_request
    assert base.context.execution_request is base.context.simulation_plan.execution_request
    assert lane.context.execution_request.prepared_payload is not base.context.execution_request.prepared_payload
    assert (
        lane.context.execution_request.prepared_payload["bindings"]
        is not base.context.execution_request.prepared_payload["bindings"]
    )

    baseline = base({"init:A": 1.0})
    lane_result = lane({"k1": 1.0, "init:A": 1.0})
    repeat = base({"init:A": 1.0})

    np.testing.assert_allclose(
        np.asarray(repeat.species["B"], dtype=float),
        np.asarray(baseline.species["B"], dtype=float),
    )
    assert not np.allclose(
        np.asarray(lane_result.species["B"], dtype=float),
        np.asarray(baseline.species["B"], dtype=float),
    )


def test_fitting_evaluator_lane_clone_rejects_self_clone() -> None:
    from kindred.core.fitting_evaluation import _clone_fitting_series_evaluator_lane

    class _SelfCloningEvaluator:
        def _kindred_clone_fitting_evaluator_lane(self):
            return self

        def evaluate_series(self, params):
            return {
                "t": np.asarray([0.0, 1.0], dtype=float),
                "species": {"A": np.asarray([0.0, 0.0], dtype=float)},
            }

    evaluator = _SelfCloningEvaluator()

    assert _clone_fitting_series_evaluator_lane(evaluator) is None


def test_fitting_evaluator_lane_clone_rejects_invalid_clone() -> None:
    from kindred.core.fitting_evaluation import _clone_fitting_series_evaluator_lane

    class _InvalidCloningEvaluator:
        def _kindred_clone_fitting_evaluator_lane(self):
            return object()

        def evaluate_series(self, params):
            return {
                "t": np.asarray([0.0, 1.0], dtype=float),
                "species": {"A": np.asarray([0.0, 0.0], dtype=float)},
            }

    assert _clone_fitting_series_evaluator_lane(_InvalidCloningEvaluator()) is None


def test_serial_fitting_evaluator_lane_cancellation_check_reaches_solver_events(monkeypatch) -> None:
    from kindred.core.fitting_evaluation import (
        SerialFittingEvaluator,
        _clone_fitting_series_evaluator_lane,
        _with_fitting_evaluator_cancellation_check,
        prepare_fitting_execution_context,
    )
    import kindred.core.fitting_evaluation as fitting_evaluation
    from kindred.core.simulator.solvers import SimulationOutput

    mechanism_text = "\n".join(
        [
            "reaction: A -> B; k=0.2",
            "initial: A=1.0",
            "initial: B=0.0",
        ]
    )
    context = prepare_fitting_execution_context(
        mechanism_text=mechanism_text,
        param_names=["k1"],
        t_end=1.0,
        num_points=3,
        solver="BDF",
        rtol=1e-6,
        atol=1e-12,
        initial_prefix="init:",
    )
    base = SerialFittingEvaluator(context).with_fixed_params({"k1": 0.2})
    lane = _clone_fitting_series_evaluator_lane(base)
    blocking_check_calls = 0

    def _blocking_check() -> bool:
        nonlocal blocking_check_calls
        blocking_check_calls += 1
        return False

    _blocking_check._kindred_nonblocking_cancelled = lambda: False
    lane = _with_fitting_evaluator_cancellation_check(lane, _blocking_check)
    captured = {}

    def _capture_request(request):
        events = list(request.events or [])
        captured["cancel_events"] = [
            event for event in events if bool(getattr(event, "_kindred_cancel_event", False))
        ]
        t = np.linspace(float(request.t_span[0]), float(request.t_span[1]), 3)
        y0 = np.asarray(request.y0, dtype=float).reshape(-1)
        return SimulationOutput(
            t=t,
            Y=np.vstack([np.full_like(t, y0[0]), np.full_like(t, y0[1])]),
            provenance={},
        )

    monkeypatch.setattr(fitting_evaluation, "_solve_request", _capture_request)

    result = lane({"init:A": 1.0})

    assert np.asarray(result.t, dtype=float).size == 3
    assert len(captured["cancel_events"]) == 1
    assert captured["cancel_events"][0](0.0, np.asarray([1.0, 0.0], dtype=float)) == 1.0
    assert blocking_check_calls == 0


def test_serial_fitting_evaluator_lane_rejects_cancellation_before_solver(monkeypatch) -> None:
    from kindred.core.exceptions import FittingCancelled
    from kindred.core.fitting_evaluation import (
        SerialFittingEvaluator,
        _clone_fitting_series_evaluator_lane,
        _with_fitting_evaluator_cancellation_check,
        prepare_fitting_execution_context,
    )
    import kindred.core.fitting_evaluation as fitting_evaluation

    mechanism_text = "\n".join(
        [
            "reaction: A -> B; k=0.2",
            "initial: A=1.0",
            "initial: B=0.0",
        ]
    )
    context = prepare_fitting_execution_context(
        mechanism_text=mechanism_text,
        param_names=["k1"],
        t_end=1.0,
        num_points=3,
        solver="BDF",
        rtol=1e-6,
        atol=1e-12,
        initial_prefix="init:",
    )
    base = SerialFittingEvaluator(context).with_fixed_params({"k1": 0.2})
    lane = _clone_fitting_series_evaluator_lane(base)
    lane = _with_fitting_evaluator_cancellation_check(lane, lambda: True)

    def _unexpected_solve(_request):
        raise AssertionError("cancelled fitting lane must not reach the solver")

    monkeypatch.setattr(fitting_evaluation, "_solve_request", _unexpected_solve)

    with pytest.raises(FittingCancelled):
        lane({"init:A": 1.0})


def test_serial_fitting_evaluator_lane_solver_event_reports_mid_solve_cancellation(monkeypatch) -> None:
    from kindred.core.exceptions import SimulationCancelled
    from kindred.core.fitting_evaluation import (
        SerialFittingEvaluator,
        _clone_fitting_series_evaluator_lane,
        _with_fitting_evaluator_cancellation_check,
        prepare_fitting_execution_context,
    )
    import kindred.core.fitting_evaluation as fitting_evaluation

    mechanism_text = "\n".join(
        [
            "reaction: A -> B; k=0.2",
            "initial: A=1.0",
            "initial: B=0.0",
        ]
    )
    context = prepare_fitting_execution_context(
        mechanism_text=mechanism_text,
        param_names=["k1"],
        t_end=1.0,
        num_points=3,
        solver="BDF",
        rtol=1e-6,
        atol=1e-12,
        initial_prefix="init:",
    )
    cancel_state = {"cancelled": False}

    def _blocking_check() -> bool:
        return False

    _blocking_check._kindred_nonblocking_cancelled = lambda: bool(cancel_state["cancelled"])
    base = SerialFittingEvaluator(context).with_fixed_params({"k1": 0.2})
    lane = _clone_fitting_series_evaluator_lane(base)
    lane = _with_fitting_evaluator_cancellation_check(lane, _blocking_check)

    def _simulate_event_cancellation(request):
        events = [
            event for event in list(request.events or []) if bool(getattr(event, "_kindred_cancel_event", False))
        ]
        assert len(events) == 1
        assert events[0](0.0, np.asarray([1.0, 0.0], dtype=float)) == 1.0
        cancel_state["cancelled"] = True
        assert events[0](0.5, np.asarray([1.0, 0.0], dtype=float)) == -1.0
        raise SimulationCancelled()

    monkeypatch.setattr(fitting_evaluation, "_solve_request", _simulate_event_cancellation)

    with pytest.raises(SimulationCancelled):
        lane({"init:A": 1.0})


def test_serial_fitting_evaluator_process_payload_round_trip_matches_original() -> None:
    from kindred.core.fitting_evaluation import SerialFittingEvaluator, prepare_fitting_execution_context
    from kindred.core.simulation_plan import SimulationAlgebraPolicy, SimulationPlan

    mechanism_text = "\n".join(
        [
            "reaction: A -> B; k=0.2",
            "initial: A=1.0",
            "initial: B=0.0",
        ]
    )
    context = prepare_fitting_execution_context(
        mechanism_text=mechanism_text,
        param_names=["k1"],
        t_end=1.0,
        num_points=6,
        solver="BDF",
        rtol=1e-6,
        atol=1e-12,
        initial_prefix="init:",
    )
    evaluator = SerialFittingEvaluator(context).with_fixed_params({"k1": 0.2})

    expected = evaluator({"init:A": 1.0})
    payload = evaluator.to_process_payload()
    assert "simulation_plan" in payload
    assert "execution_request" not in payload
    process_plan = SimulationPlan.from_payload(payload["simulation_plan"])
    assert process_plan.execution_mode == "fitting"
    assert process_plan.algebra_policy is SimulationAlgebraPolicy.FITTING_STRICT

    restored = SerialFittingEvaluator.from_process_payload(payload)
    assert restored.context.simulation_plan.execution_mode == "fitting"
    assert restored.context.simulation_plan.algebra_policy is SimulationAlgebraPolicy.FITTING_STRICT
    assert restored.context.execution_request is restored.context.simulation_plan.execution_request
    _assert_execution_request_payloads_equal(
        restored.context.execution_request.to_payload(),
        context.execution_request.to_payload(),
    )
    actual = restored({"init:A": 1.0})

    np.testing.assert_allclose(np.asarray(actual.t, dtype=float), np.asarray(expected.t, dtype=float))
    np.testing.assert_allclose(
        np.asarray(actual.species["A"], dtype=float),
        np.asarray(expected.species["A"], dtype=float),
    )
    np.testing.assert_allclose(
        np.asarray(actual.species["B"], dtype=float),
        np.asarray(expected.species["B"], dtype=float),
    )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("temperature_K", 999.0, "temperature_K"),
        ("initial_prefix", "stale:", "initial_prefix"),
        ("requested_param_names", ["other"], "requested_param_names"),
    ],
)
def test_serial_fitting_process_payload_rejects_plan_conflicting_duplicate_fields(field, value, match) -> None:
    from kindred.core.fitting_evaluation import SerialFittingEvaluator, prepare_fitting_execution_context

    mechanism_text = "\n".join(
        [
            "reaction: A -> B; k=0.2",
            "initial: A=1.0",
            "initial: B=0.0",
        ]
    )
    context = prepare_fitting_execution_context(
        mechanism_text=mechanism_text,
        param_names=["k1"],
        t_end=1.0,
        num_points=6,
        solver="BDF",
        rtol=1e-6,
        atol=1e-12,
        temperature_K=310.0,
        initial_prefix="init:",
    )
    payload = SerialFittingEvaluator(context).to_process_payload()
    payload[field] = value

    with pytest.raises(ValueError, match=match):
        SerialFittingEvaluator.from_process_payload(payload)


def test_serial_fitting_process_payload_rejects_schedule_fingerprint_conflict() -> None:
    from kindred.core.fitting_evaluation import SerialFittingEvaluator, prepare_fitting_execution_context

    mechanism_text = "\n".join(
        [
            "reaction: A -> B; k=0.2",
            "initial: A=1.0",
            "initial: B=0.0",
            "intervention: op=set; species=A; time=0.0; value=2.0",
        ]
    )
    context = prepare_fitting_execution_context(
        mechanism_text=mechanism_text,
        param_names=["k1"],
        t_end=1.0,
        num_points=6,
        solver="BDF",
        rtol=1e-6,
        atol=1e-12,
        temperature_K=310.0,
        initial_prefix="init:",
    )
    payload = SerialFittingEvaluator(context).to_process_payload()
    payload["prepared_metadata"] = dict(payload["prepared_metadata"])
    payload["prepared_metadata"]["intervention_schedule_fingerprint"] = "stale-schedule"

    with pytest.raises(ValueError, match="intervention_schedule_fingerprint"):
        SerialFittingEvaluator.from_process_payload(payload)


def test_serial_fitting_process_payload_preserves_and_executes_intervention_schedule() -> None:
    from kindred.core.fitting_evaluation import SerialFittingEvaluator, prepare_fitting_execution_context
    from kindred.core.intervention_schedule import coerce_intervention_schedule

    mechanism_text = "\n".join(
        [
            "reaction: A -> B; k=0",
            "initial: A=1.0",
            "initial: B=0.0",
            "intervention: op=set; species=A; time=0.0; value=3.0",
        ]
    )
    context = prepare_fitting_execution_context(
        mechanism_text=mechanism_text,
        param_names=["k1"],
        t_end=1.0,
        num_points=4,
        solver="BDF",
        rtol=1e-6,
        atol=1e-12,
        initial_prefix="init:",
    )

    schedule = coerce_intervention_schedule(context.execution_request.intervention_schedule)
    assert schedule is not None
    assert context.prepared_metadata.intervention_schedule_fingerprint == schedule.fingerprint

    payload = SerialFittingEvaluator(context).to_process_payload()
    plan_request = payload["simulation_plan"]["execution_request"]
    assert plan_request["intervention_schedule"] == schedule.to_payload()
    assert plan_request["prepared_payload"]["intervention_schedule"] == schedule.to_payload()
    assert payload["prepared_metadata"]["intervention_schedule_fingerprint"] == schedule.fingerprint

    restored = SerialFittingEvaluator.from_process_payload(payload)
    result = restored({"init:A": 1.0, "k1": 0.0})

    assert float(np.asarray(result.species["A"], dtype=float)[0]) == pytest.approx(3.0)


def test_serial_fitting_evaluator_process_payload_is_picklable_without_prepared_rhs() -> None:
    from kindred.core.fitting_evaluation import SerialFittingEvaluator, prepare_fitting_execution_context

    mechanism_text = "\n".join(
        [
            "reaction: A -> B; k=0.2",
            "initial: A=1.0",
            "initial: B=0.0",
        ]
    )
    context = prepare_fitting_execution_context(
        mechanism_text=mechanism_text,
        param_names=["k1"],
        t_end=1.0,
        num_points=5,
        solver="BDF",
        rtol=1e-6,
        atol=1e-12,
        initial_prefix="init:",
    )
    evaluator = SerialFittingEvaluator(context)
    _ = evaluator({"k1": 0.2, "init:A": 1.0})

    payload = evaluator.to_process_payload()

    assert "execution_request" not in payload
    plan_request = payload["simulation_plan"]["execution_request"]
    assert plan_request["prepared_payload"] is not None
    assert "rhs" not in plan_request["prepared_payload"]
    pickle.dumps(payload)


def test_fitting_context_accepts_equivalent_plan_and_request_with_temperature_schedule() -> None:
    from kindred.core.fitting_evaluation import PreparedFittingExecutionContext, prepare_fitting_execution_context
    from kindred.core.simulation_plan import SimulationAlgebraPolicy, SimulationPlan
    from kindred.core.temperature import TemperatureSchedule

    mechanism_text = "\n".join(
        [
            "reaction: A -> B; k=0.2",
            "initial: A=1.0",
            "initial: B=0.0",
        ]
    )
    context = prepare_fitting_execution_context(
        mechanism_text=mechanism_text,
        param_names=["k1"],
        t_end=1.0,
        num_points=5,
        solver="BDF",
        rtol=1e-6,
        atol=1e-12,
        initial_prefix="init:",
    )
    request_payload = context.execution_request.to_payload()
    request_payload["prepared_payload"]["temperature_schedule"] = TemperatureSchedule.constant(298.15)
    plan_payload = SimulationPlan.from_execution_request(
        request_payload,
        execution_mode="fitting",
        algebra_policy=SimulationAlgebraPolicy.FITTING_STRICT,
    ).to_payload()

    restored = PreparedFittingExecutionContext(
        simulation_plan=plan_payload,
        execution_request=request_payload,
        requested_param_names=list(context.requested_param_names),
        prepared_metadata=context.prepared_metadata,
        temperature_K=context.temperature_K,
        initial_prefix=context.initial_prefix,
    )

    assert restored.execution_request is restored.simulation_plan.execution_request


def test_serial_fitting_evaluator_from_process_payload_rejects_malformed_simulation_plan() -> None:
    from kindred.core.fitting_evaluation import SerialFittingEvaluator, prepare_fitting_execution_context

    mechanism_text = "\n".join(
        [
            "reaction: A -> B; k=0.2",
            "initial: A=1.0",
            "initial: B=0.0",
        ]
    )
    context = prepare_fitting_execution_context(
        mechanism_text=mechanism_text,
        param_names=["k1"],
        t_end=1.0,
        num_points=5,
        solver="BDF",
        rtol=1e-6,
        atol=1e-12,
        initial_prefix="init:",
    )
    payload = SerialFittingEvaluator(context).to_process_payload()
    payload["simulation_plan"] = "broken"

    with pytest.raises(TypeError, match="simulation_plan"):
        SerialFittingEvaluator.from_process_payload(payload)


def test_serial_fitting_evaluator_from_process_payload_rejects_legacy_execution_request_key() -> None:
    from kindred.core.fitting_evaluation import SerialFittingEvaluator, prepare_fitting_execution_context

    mechanism_text = "\n".join(
        [
            "reaction: A -> B; k=0.2",
            "initial: A=1.0",
            "initial: B=0.0",
        ]
    )
    context = prepare_fitting_execution_context(
        mechanism_text=mechanism_text,
        param_names=["k1"],
        t_end=1.0,
        num_points=5,
        solver="BDF",
        rtol=1e-6,
        atol=1e-12,
        initial_prefix="init:",
    )
    payload = SerialFittingEvaluator(context).to_process_payload()
    payload["execution_request"] = context.execution_request.to_payload()

    with pytest.raises(KeyError, match="execution_request"):
        SerialFittingEvaluator.from_process_payload(payload)


def test_serial_fitting_evaluator_process_payload_handles_empty_param_names() -> None:
    from kindred.core.fitting_evaluation import SerialFittingEvaluator, prepare_fitting_execution_context

    mechanism_text = "\n".join(
        [
            "reaction: A -> B; k=0.2",
            "initial: A=1.0",
            "initial: B=0.0",
        ]
    )
    context = prepare_fitting_execution_context(
        mechanism_text=mechanism_text,
        param_names=[],
        t_end=1.0,
        num_points=4,
        solver="BDF",
        rtol=1e-6,
        atol=1e-12,
        initial_prefix="init:",
    )
    evaluator = SerialFittingEvaluator(context)

    restored = SerialFittingEvaluator.from_process_payload(evaluator.to_process_payload())
    result = restored({"init:A": 1.0})

    assert np.asarray(result.t, dtype=float).size == 4
    assert set(result.species) == {"A", "B"}


def test_serial_fitting_evaluator_process_payload_handles_large_mechanism() -> None:
    from kindred.core.fitting_evaluation import SerialFittingEvaluator, prepare_fitting_execution_context

    lines = []
    for idx in range(100):
        lines.append(f"reaction: S{idx} -> S{idx + 1}; k=0.1")
    for idx in range(101):
        value = "1.0" if idx == 0 else "0.0"
        lines.append(f"initial: S{idx}={value}")
    mechanism_text = "\n".join(lines)
    context = prepare_fitting_execution_context(
        mechanism_text=mechanism_text,
        param_names=[],
        t_end=1.0,
        num_points=3,
        solver="BDF",
        rtol=1e-6,
        atol=1e-12,
        initial_prefix="init:",
    )
    evaluator = SerialFittingEvaluator(context)

    payload = evaluator.to_process_payload()
    restored = SerialFittingEvaluator.from_process_payload(payload)
    result = restored({"init:S0": 1.0})

    assert payload["requested_param_names"] == []
    assert np.asarray(result.t, dtype=float).size == 3
    assert "S100" in result.species


def test_serial_fitting_evaluator_from_process_payload_requires_all_fields() -> None:
    from kindred.core.fitting_evaluation import SerialFittingEvaluator, prepare_fitting_execution_context

    context = prepare_fitting_execution_context(
        mechanism_text="\n".join(
            [
                "reaction: A -> B; k=0.2",
                "initial: A=1.0",
                "initial: B=0.0",
            ]
        ),
        param_names=["k"],
        t_end=1.0,
        num_points=3,
        solver="BDF",
        rtol=1e-6,
        atol=1e-12,
        initial_prefix="init:",
    )
    payload = SerialFittingEvaluator(context).to_process_payload()

    incomplete_payload = dict(payload)
    incomplete_payload.pop("simulation_plan")
    with pytest.raises(KeyError, match="simulation_plan"):
        SerialFittingEvaluator.from_process_payload(incomplete_payload)

    with pytest.raises(KeyError, match="simulation_plan"):
        SerialFittingEvaluator.from_process_payload(
            {
                name: value
                for name, value in payload.items()
                if name != "simulation_plan"
            }
        )


def test_serial_fitting_evaluator_rejects_nonfinite_consumed_parameter_values() -> None:
    from kindred.core.exceptions import FitSimulationError
    from kindred.core.fitting_evaluation import SerialFittingEvaluator, prepare_fitting_execution_context

    mechanism_text = "\n".join(
        [
            "reaction: A -> B; k=0.2",
            "initial: A=1.0",
            "initial: B=0.0",
        ]
    )
    context = prepare_fitting_execution_context(
        mechanism_text=mechanism_text,
        param_names=["k1"],
        t_end=1.0,
        num_points=6,
        solver="BDF",
        rtol=1e-6,
        atol=1e-12,
        initial_prefix="init:",
    )
    evaluator = SerialFittingEvaluator(context)

    with pytest.raises(FitSimulationError, match="Non-finite parameter value") as exc_info:
        evaluator({"k1": float("nan"), "init:A": 1.0})

    assert exc_info.value.details["fatal"] is True


def test_serial_fitting_evaluator_ignores_nonfinite_unconsumed_parameter_values() -> None:
    from kindred.core.fitting_evaluation import SerialFittingEvaluator, prepare_fitting_execution_context

    mechanism_text = "\n".join(
        [
            "reaction: A -> B; k=0.2",
            "initial: A=1.0",
            "initial: B=0.0",
        ]
    )
    context = prepare_fitting_execution_context(
        mechanism_text=mechanism_text,
        param_names=["k1"],
        t_end=1.0,
        num_points=6,
        solver="BDF",
        rtol=1e-6,
        atol=1e-12,
        initial_prefix="init:",
    )
    evaluator = SerialFittingEvaluator(context)

    expected = evaluator({"k1": 0.2, "init:A": 1.0})
    actual = evaluator(
        {
            "k1": 0.2,
            "init:A": 1.0,
            "init:Removed": float("nan"),
            "unused_rate": float("inf"),
            "arbitrary_extra": float("-inf"),
        }
    )

    assert np.allclose(
        np.asarray(actual.species["B"], dtype=float),
        np.asarray(expected.species["B"], dtype=float),
    )


def test_serial_fitting_evaluator_rejects_non_numeric_parameter_values_fatally() -> None:
    from kindred.core.exceptions import FitSimulationError
    from kindred.core.fitting_evaluation import SerialFittingEvaluator, prepare_fitting_execution_context

    mechanism_text = "\n".join(
        [
            "reaction: A -> B; k=0.2",
            "initial: A=1.0",
            "initial: B=0.0",
        ]
    )
    context = prepare_fitting_execution_context(
        mechanism_text=mechanism_text,
        param_names=["k1"],
        t_end=1.0,
        num_points=6,
        solver="BDF",
        rtol=1e-6,
        atol=1e-12,
        initial_prefix="init:",
    )
    evaluator = SerialFittingEvaluator(context)

    with pytest.raises(FitSimulationError, match="Invalid parameter value") as exc_info:
        evaluator({"k1": "not-a-number", "init:A": 1.0})

    assert exc_info.value.details["fatal"] is True


def test_serial_fitting_evaluator_rejects_nonfinite_fixed_params_fatally() -> None:
    from kindred.core.exceptions import FitSimulationError
    from kindred.core.fitting_evaluation import SerialFittingEvaluator, prepare_fitting_execution_context

    mechanism_text = "\n".join(
        [
            "reaction: A -> B; k=0.2",
            "initial: A=1.0",
            "initial: B=0.0",
        ]
    )
    context = prepare_fitting_execution_context(
        mechanism_text=mechanism_text,
        param_names=["k1"],
        t_end=1.0,
        num_points=6,
        solver="BDF",
        rtol=1e-6,
        atol=1e-12,
        initial_prefix="init:",
    )
    evaluator = SerialFittingEvaluator(context).with_fixed_params({"k1": float("nan")})

    with pytest.raises(FitSimulationError, match="Non-finite parameter value") as exc_info:
        evaluator({"init:A": 1.0})

    assert exc_info.value.details["fatal"] is True


def test_serial_fitting_evaluator_normalizes_solver_failures_to_fit_simulation_error(monkeypatch) -> None:
    from kindred.core.exceptions import FitSimulationError
    from kindred.core.fitting_evaluation import SerialFittingEvaluator, prepare_fitting_execution_context
    import kindred.core.fitting_evaluation as fitting_evaluation

    mechanism_text = "\n".join(
        [
            "reaction: A -> B; k=0.2",
            "initial: A=1.0",
            "initial: B=0.0",
        ]
    )
    context = prepare_fitting_execution_context(
        mechanism_text=mechanism_text,
        param_names=["k1"],
        t_end=1.0,
        num_points=6,
        solver="BDF",
        rtol=1e-6,
        atol=1e-12,
        initial_prefix="init:",
    )
    evaluator = SerialFittingEvaluator(context)

    def _raise_solver_error(_request):
        raise ValueError("array must not contain infs or NaNs")

    monkeypatch.setattr(fitting_evaluation, "_solve_request", _raise_solver_error)

    with pytest.raises(FitSimulationError, match="Fitting simulation failed"):
        evaluator({"k1": 1e308, "init:A": 1.0})
