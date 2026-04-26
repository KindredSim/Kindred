from __future__ import annotations

import numpy as np
import pytest

from kindred.core.simulator.dsl import parse_dsl_to_mechanism


def _dsl_with_partial_observable_failure() -> str:
    return "\n".join(
        [
            "reaction: A -> B; k=1.0",
            "initial: A=1.0",
            "initial: B=0.0",
            "# Algebra",
            "let ok = [A] + [B]",
            "let bad = missing_symbol + 1",
        ]
    )


def test_best_effort_algebra_policy_preserves_current_gui_and_batch_partial_error_behavior() -> None:
    from kindred.core.simulation_algebra_policy import evaluate_simulation_algebra
    from kindred.core.simulation_plan import SimulationAlgebraPolicy

    mechanism = parse_dsl_to_mechanism(_dsl_with_partial_observable_failure(), initials={})
    t = np.asarray([0.0, 1.0], dtype=float)
    species_series = {
        "A": np.asarray([1.0, 0.5], dtype=float),
        "B": np.asarray([0.0, 0.5], dtype=float),
    }
    initials = {"A": 1.0, "B": 0.0}

    gui_result = evaluate_simulation_algebra(
        SimulationAlgebraPolicy.GUI_BEST_EFFORT,
        mechanism,
        t=t,
        species_series=species_series,
        initials=initials,
    )
    batch_result = evaluate_simulation_algebra(
        SimulationAlgebraPolicy.BATCH_BEST_EFFORT,
        mechanism,
        t=t,
        species_series=species_series,
        initials=initials,
    )

    assert "ok" in gui_result.series
    assert any(error.get("name") == "bad" for error in gui_result.errors)
    assert gui_result.warning is None

    assert "ok" in batch_result.series
    assert batch_result.errors == []
    assert batch_result.warning is None


def test_fitting_strict_policy_rejects_best_effort_simulation_algebra_helper() -> None:
    from kindred.core.simulation_algebra_policy import evaluate_simulation_algebra
    from kindred.core.simulation_plan import SimulationAlgebraPolicy

    mechanism = parse_dsl_to_mechanism(_dsl_with_partial_observable_failure(), initials={})

    with pytest.raises(ValueError, match="fitting_strict"):
        evaluate_simulation_algebra(
            SimulationAlgebraPolicy.FITTING_STRICT,
            mechanism,
            t=np.asarray([0.0], dtype=float),
            species_series={"A": np.asarray([1.0]), "B": np.asarray([0.0])},
            initials={"A": 1.0, "B": 0.0},
        )


def test_fitting_strict_policy_preserves_current_fatal_and_nonfatal_error_split() -> None:
    from types import SimpleNamespace

    from kindred.core.algebra.errors import (
        AlgebraDomainError,
        AlgebraNameError,
        AlgebraShadowError,
        AlgebraSyntaxError,
    )
    from kindred.core.simulation_algebra_policy import (
        fitting_strict_evaluation_error,
        fitting_strict_parse_error,
        fitting_strict_time_ref_error,
    )

    fatal_errors = [
        AlgebraNameError("unknown symbol", line=1, col=5, line_text="let bad = missing"),
        AlgebraSyntaxError("unexpected token", line=2, col=1, line_text="let ="),
        AlgebraShadowError("shadowed symbol", line=3, col=5, line_text="let A = 1"),
    ]
    for error in fatal_errors:
        fit_error = fitting_strict_evaluation_error(error, message_prefix="Algebra evaluation failed")
        assert fit_error.details["fatal"] is True
        assert fit_error.context.line == error.line
        assert fit_error.context.col == error.col
        assert fit_error.context.line_text == error.line_text

    nonfatal_domain = fitting_strict_evaluation_error(
        AlgebraDomainError("domain error", line=4, col=5, line_text="let bad = log(-1)"),
        message_prefix="Algebra evaluation failed",
    )
    nonfatal_runtime = fitting_strict_evaluation_error(
        RuntimeError("temporary evaluator failure"),
        message_prefix="Algebra evaluation failed",
    )

    assert nonfatal_domain.details["fatal"] is False
    assert nonfatal_runtime.details["fatal"] is False

    parse_error = fitting_strict_parse_error(
        AlgebraSyntaxError("bad algebra", line=5, col=1, line_text="# Algebra"),
        message_prefix="Failed to parse algebra block",
    )
    time_ref_error = fitting_strict_time_ref_error(
        SimpleNamespace(line=6, col=9, line_text="let baseline = [A](T0)")
    )

    assert parse_error.details["fatal"] is True
    assert time_ref_error.details["fatal"] is True
    assert time_ref_error.context.line == 6
    assert time_ref_error.context.col == 9
    assert time_ref_error.context.line_text == "let baseline = [A](T0)"


def test_simulation_algebra_evaluation_copies_mutable_inputs_defensively() -> None:
    from kindred.core.simulation_algebra_policy import SimulationAlgebraEvaluation

    obs = np.asarray([1.0], dtype=float)
    errors = [{"name": "bad", "message": "original"}]
    warning = {"kind": "algebra_warning", "details": {"stage": "algebra_evaluation"}}

    evaluation = SimulationAlgebraEvaluation(
        series={"obs": obs},
        scalars={"s": 2.0},
        errors=errors,
        warning=warning,
    )

    obs[0] = 99.0
    errors[0]["message"] = "mutated"
    warning["details"]["stage"] = "mutated"

    assert evaluation.series["obs"][0] == 1.0
    assert evaluation.scalars == {"s": 2.0}
    assert evaluation.errors == [{"name": "bad", "message": "original"}]
    assert evaluation.warning == {
        "kind": "algebra_warning",
        "details": {"stage": "algebra_evaluation"},
    }
