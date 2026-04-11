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
        solver="LSODA",
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
        solver="LSODA",
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
        solver="LSODA",
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
        solver="LSODA",
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
        solver="LSODA",
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
        solver="LSODA",
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
        solver="LSODA",
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
        solver="LSODA",
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
        solver="LSODA",
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
