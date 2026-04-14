from __future__ import annotations

import pickle

import numpy as np
import pytest


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
    assert base.context.execution_request is not derived.context.execution_request
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
    assert lane.context.execution_request is not base.context.execution_request
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
    restored = SerialFittingEvaluator.from_process_payload(payload)
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

    assert payload["execution_request"]["prepared_payload"] is not None
    assert "rhs" not in payload["execution_request"]["prepared_payload"]
    pickle.dumps(payload)


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
    from kindred.core.fitting_evaluation import SerialFittingEvaluator

    with pytest.raises(KeyError, match="execution_request"):
        SerialFittingEvaluator.from_process_payload(
            {
                "requested_param_names": [],
                "prepared_metadata": {},
                "temperature_K": 298.15,
                "initial_prefix": "init:",
                "fixed_params": {},
                "fixed_param_origins": {},
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
