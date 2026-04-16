from __future__ import annotations

import numpy as np
import pytest

from kindred.core.batch_parallel import run_batch_simulation_task
from kindred.core.fitting_evaluation import (
    SerialFittingEvaluator,
    evaluate_fitting_series,
    prepare_fitting_execution_context,
)
from kindred.core.simulation_preparation import (
    prepare_bound_mechanism,
    prepare_simulation_worker_run,
)
from kindred.core.simulator.solvers import SimulationRequest, solve_ode


pytestmark = pytest.mark.unit


_OVERRIDE_MECHANISM_TEXT = "\n".join(
    [
        "A <-> B ; kf=1.0, kr=0.01",
        "B -> C ; k=0.2",
        "param a = 5",
        "param kr1 = a*kf1",
    ]
)

_CONTROL_MECHANISM_TEXT = "\n".join(
    [
        "A <-> B ; kf=1.0, kr=5.0",
        "B -> C ; k=0.2",
    ]
)

_INITIALS = {"A": 1.0, "B": 0.0, "C": 0.0}
_SOLVER_CONFIG = {
    "solver": "BDF",
    "rtol": 1e-8,
    "atol": 1e-10,
    "grid": {"N": 80},
    "use_sparse_jacobian": False,
}


def _solve_canonical_series(mechanism_text: str) -> tuple[list[str], np.ndarray]:
    prepared = prepare_simulation_worker_run(
        mechanism_text=mechanism_text,
        initials=dict(_INITIALS),
        t_span=(0.0, 20.0),
        solver_config=dict(_SOLVER_CONFIG),
    )
    result = solve_ode(prepared.request)
    return list(prepared.species_names), np.asarray(result.Y, dtype=float)


def _solve_preview_series(mechanism_text: str) -> tuple[list[str], np.ndarray]:
    bound = prepare_bound_mechanism(
        mechanism_text,
        ["a"],
        temperature_K=298.15,
        initials=dict(_INITIALS),
        use_advanced_dsl=True,
        wegscheider_cyclicity_enabled=False,
    )
    request = SimulationRequest(
        rhs=bound.rhs,
        t_span=(0.0, 20.0),
        y0=np.asarray(bound.y0, dtype=float).reshape(-1),
        solver=str(_SOLVER_CONFIG["solver"]),
        rtol=float(_SOLVER_CONFIG["rtol"]),
        atol=float(_SOLVER_CONFIG["atol"]),
        grid=dict(_SOLVER_CONFIG["grid"]),
    )
    result = solve_ode(request)
    return list(bound.species_names), np.asarray(result.Y, dtype=float)


def _solve_batch_explicit_series(mechanism_text: str) -> tuple[list[str], np.ndarray]:
    payload = run_batch_simulation_task(
        {
            "run_id": 1,
            "set_id": "id1",
            "set_name": "set1",
            "execution_request": {
                "prepared_payload": None,
                "initials": dict(_INITIALS),
                "t_span": (0.0, 20.0),
                "solver_config": dict(_SOLVER_CONFIG),
                "mechanism_text": mechanism_text,
                "simulation_identity": {
                    "schema_id": "override-runtime-regression",
                    "param_fingerprint": mechanism_text,
                },
            },
        }
    )
    assert payload["success"] is True
    return list(payload["species_names"]), np.asarray(payload["Y"], dtype=float)


def _solve_fitting_series(mechanism_text: str, params: dict[str, float]) -> dict[str, np.ndarray]:
    context = prepare_fitting_execution_context(
        mechanism_text=mechanism_text,
        param_names=["a"] if "a" in params else [],
        t_end=20.0,
        num_points=80,
        temperature_K=298.15,
        solver="BDF",
        rtol=1e-8,
        atol=1e-10,
        use_sparse_jacobian=False,
        wegscheider_cyclicity_enabled=False,
        initial_prefix="init:",
    )
    evaluator = SerialFittingEvaluator(context)
    result = evaluate_fitting_series(evaluator, dict(params))
    return {name: np.asarray(series, dtype=float) for name, series in result["species"].items()}


def _assert_final_state_matches_control(
    actual_names: list[str],
    actual_y: np.ndarray,
    control_names: list[str],
    control_y: np.ndarray,
) -> None:
    assert actual_names == control_names
    np.testing.assert_allclose(actual_y[:, -1], control_y[:, -1], rtol=1e-6, atol=1e-8)


def test_canonical_explicit_run_uses_algebra_derived_equilibrium_rate_override() -> None:
    actual_names, actual_y = _solve_canonical_series(_OVERRIDE_MECHANISM_TEXT)
    control_names, control_y = _solve_canonical_series(_CONTROL_MECHANISM_TEXT)

    _assert_final_state_matches_control(actual_names, actual_y, control_names, control_y)


def test_batch_parallel_explicit_run_uses_algebra_derived_equilibrium_rate_override() -> None:
    actual_names, actual_y = _solve_batch_explicit_series(_OVERRIDE_MECHANISM_TEXT)
    control_names, control_y = _solve_batch_explicit_series(_CONTROL_MECHANISM_TEXT)

    _assert_final_state_matches_control(actual_names, actual_y, control_names, control_y)


def test_canonical_and_preview_paths_match_for_equilibrium_rate_override() -> None:
    canonical_names, canonical_y = _solve_canonical_series(_OVERRIDE_MECHANISM_TEXT)
    preview_names, preview_y = _solve_preview_series(_OVERRIDE_MECHANISM_TEXT)

    _assert_final_state_matches_control(canonical_names, canonical_y, preview_names, preview_y)


def test_fitting_evaluation_uses_algebra_derived_equilibrium_rate_override() -> None:
    actual = _solve_fitting_series(
        _OVERRIDE_MECHANISM_TEXT,
        {"a": 5.0, "init:A": 1.0},
    )
    control = _solve_fitting_series(
        _CONTROL_MECHANISM_TEXT,
        {"init:A": 1.0},
    )

    for species_name in ("A", "B", "C"):
        np.testing.assert_allclose(
            actual[species_name][-1],
            control[species_name][-1],
            rtol=1e-6,
            atol=1e-8,
        )


def test_structured_prepared_payload_rebuilds_rhs_after_algebra_override() -> None:
    bound = prepare_bound_mechanism(
        _OVERRIDE_MECHANISM_TEXT,
        ["a"],
        temperature_K=298.15,
        initials=dict(_INITIALS),
        use_advanced_dsl=True,
        wegscheider_cyclicity_enabled=False,
    )
    bound.bindings["kr1"].set(0.01)
    prepared = prepare_simulation_worker_run(
        execution_request={
            "prepared_payload": bound.as_serializable_execution_payload(),
            "initials": dict(_INITIALS),
            "t_span": (0.0, 20.0),
            "solver_config": dict(_SOLVER_CONFIG),
            "mechanism_text": _OVERRIDE_MECHANISM_TEXT,
        }
    )
    result = solve_ode(prepared.request)
    control_names, control_y = _solve_canonical_series(_CONTROL_MECHANISM_TEXT)

    _assert_final_state_matches_control(
        list(prepared.species_names),
        np.asarray(result.Y, dtype=float),
        control_names,
        control_y,
    )
