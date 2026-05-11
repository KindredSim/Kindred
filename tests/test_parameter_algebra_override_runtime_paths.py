from __future__ import annotations

import numpy as np
import pytest

from kindred.core.batch_parallel import run_batch_simulation_task
from kindred.core.exceptions import FitSimulationError
from kindred.core.fitting_evaluation import (
    SerialFittingEvaluator,
    evaluate_fitting_series,
    prepare_fitting_execution_context,
)
from kindred.core.simulation_plan import SimulationAlgebraPolicy, SimulationPlan
from kindred.core.simulation_preparation import (
    SimulationExecutionRequest,
    build_prepared_simulation_func,
    prepared_simulation_run_for_execution_request,
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
    execution_request = {
        "prepared_payload": None,
        "initials": dict(_INITIALS),
        "t_span": (0.0, 20.0),
        "solver_config": dict(_SOLVER_CONFIG),
        "mechanism_text": mechanism_text,
        "simulation_identity": {
            "schema_id": "override-runtime-regression",
            "param_fingerprint": mechanism_text,
        },
    }
    payload = run_batch_simulation_task(
        {
            "run_id": 1,
            "set_id": "id1",
            "set_name": "set1",
            "simulation_plan": SimulationPlan.from_execution_request(
                execution_request,
                execution_mode="explicit",
                algebra_policy=SimulationAlgebraPolicy.BATCH_BEST_EFFORT,
                metadata={"set_id": "id1", "set_name": "set1"},
            ).to_payload(),
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


def test_prepared_simulation_run_reuses_bound_parameter_override_without_reparse() -> None:
    mechanism_text = "reaction: A -> B; k=1.0"
    solver_config = {
        "solver": "BDF",
        "rtol": 1e-8,
        "atol": 1e-10,
        "grid": {"N": 40},
        "use_sparse_jacobian": False,
    }
    startup = SimulationExecutionRequest(
        prepared_payload=None,
        initials={"A": 1.0, "B": 0.0},
        t_span=(0.0, 1.0),
        solver_config=dict(solver_config),
        mechanism_text=mechanism_text,
        parameter_overrides={"k1": 1.0},
    )
    prepared = prepare_simulation_worker_run(execution_request=startup)

    changed = prepared_simulation_run_for_execution_request(
        prepared,
        SimulationExecutionRequest(
            prepared_payload=None,
            initials={"A": 1.0, "B": 0.0},
            t_span=(0.0, 1.0),
            solver_config=dict(solver_config),
            mechanism_text=mechanism_text,
            parameter_overrides={"k1": 2.0},
        ),
    )
    result = solve_ode(changed.request)
    control = prepare_simulation_worker_run(
        mechanism_text="reaction: A -> B; k=2.0",
        initials={"A": 1.0, "B": 0.0},
        t_span=(0.0, 1.0),
        solver_config=dict(solver_config),
    )
    control_result = solve_ode(control.request)

    np.testing.assert_allclose(
        np.asarray(result.Y, dtype=float)[:, -1],
        np.asarray(control_result.Y, dtype=float)[:, -1],
        rtol=1e-6,
        atol=1e-8,
    )


def test_prepared_simulation_run_reprepares_when_warmed_payload_lacks_mutable_binding() -> None:
    solver_config = {
        "solver": "BDF",
        "rtol": 1e-8,
        "atol": 1e-10,
        "grid": {"N": 40},
        "use_sparse_jacobian": False,
    }
    prepared = prepare_simulation_worker_run(
        execution_request=SimulationExecutionRequest(
            prepared_payload=None,
            initials={"A": 1.0, "B": 0.0},
            t_span=(0.0, 1.0),
            solver_config=dict(solver_config),
            mechanism_text="A -> B ; kf = 1.0",
        )
    )

    changed = prepared_simulation_run_for_execution_request(
        prepared,
        SimulationExecutionRequest(
            prepared_payload=None,
            initials={"A": 1.0, "B": 0.0},
            t_span=(0.0, 1.0),
            solver_config=dict(solver_config),
            mechanism_text="A -> B ; k=2.0",
            parameter_overrides={"k1": 2.0},
        ),
    )
    result = solve_ode(changed.request)
    control = prepare_simulation_worker_run(
        execution_request=SimulationExecutionRequest(
            prepared_payload=None,
            initials={"A": 1.0, "B": 0.0},
            t_span=(0.0, 1.0),
            solver_config=dict(solver_config),
            mechanism_text="A -> B ; k=2.0",
            parameter_overrides={"k1": 2.0},
        )
    )
    control_result = solve_ode(control.request)

    np.testing.assert_allclose(
        np.asarray(result.Y, dtype=float)[:, -1],
        np.asarray(control_result.Y, dtype=float)[:, -1],
        rtol=1e-6,
        atol=1e-8,
    )


def test_prepared_simulation_run_reprepares_when_any_override_lacks_mutable_binding() -> None:
    solver_config = {
        "solver": "BDF",
        "rtol": 1e-8,
        "atol": 1e-10,
        "grid": {"N": 80},
        "use_sparse_jacobian": False,
    }
    prepared = prepare_simulation_worker_run(
        execution_request=SimulationExecutionRequest(
            prepared_payload=None,
            initials={"A": 1.0, "B": 0.0, "C": 0.0},
            t_span=(0.0, 2.0),
            solver_config=dict(solver_config),
            mechanism_text="\n".join(
                [
                    "A -> B ; k=1.0",
                    "B -> C ; k=1.0",
                ]
            ),
            parameter_overrides={"k1": 1.0},
        )
    )

    changed = prepared_simulation_run_for_execution_request(
        prepared,
        SimulationExecutionRequest(
            prepared_payload=None,
            initials={"A": 1.0, "B": 0.0, "C": 0.0},
            t_span=(0.0, 2.0),
            solver_config=dict(solver_config),
            mechanism_text="\n".join(
                [
                    "A -> B ; k=2.0",
                    "B -> C ; k=3.0",
                ]
            ),
            parameter_overrides={"k1": 2.0, "k2": 3.0},
        ),
    )
    result = solve_ode(changed.request)
    fresh_control = solve_ode(
        prepare_simulation_worker_run(
            execution_request=SimulationExecutionRequest(
                prepared_payload=None,
                initials={"A": 1.0, "B": 0.0, "C": 0.0},
                t_span=(0.0, 2.0),
                solver_config=dict(solver_config),
                mechanism_text="\n".join(
                    [
                        "A -> B ; k=2.0",
                        "B -> C ; k=3.0",
                    ]
                ),
                parameter_overrides={"k1": 2.0, "k2": 3.0},
            )
        ).request
    )
    stale_partial_control = solve_ode(
        prepare_simulation_worker_run(
            execution_request=SimulationExecutionRequest(
                prepared_payload=None,
                initials={"A": 1.0, "B": 0.0, "C": 0.0},
                t_span=(0.0, 2.0),
                solver_config=dict(solver_config),
                mechanism_text="\n".join(
                    [
                        "A -> B ; k=2.0",
                        "B -> C ; k=1.0",
                    ]
                ),
                parameter_overrides={"k1": 2.0},
            )
        ).request
    )

    np.testing.assert_allclose(
        np.asarray(result.Y, dtype=float)[:, -1],
        np.asarray(fresh_control.Y, dtype=float)[:, -1],
        rtol=1e-6,
        atol=1e-8,
    )
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(
            np.asarray(result.Y, dtype=float)[:, -1],
            np.asarray(stale_partial_control.Y, dtype=float)[:, -1],
            rtol=1e-6,
            atol=1e-8,
        )


def test_reused_prepared_preview_applies_scalar_algebra_parameter_override() -> None:
    startup = SimulationExecutionRequest(
        prepared_payload=None,
        initials=dict(_INITIALS),
        t_span=(0.0, 20.0),
        solver_config=dict(_SOLVER_CONFIG),
        mechanism_text="\n".join(
            [
                "A <-> B ; kf=1.0, kr=0.01",
                "B -> C ; k=0.2",
                "param a = 1",
                "param kr1 = a*kf1",
            ]
        ),
        parameter_overrides={"a": 1.0},
    )
    prepared = prepare_simulation_worker_run(execution_request=startup)

    changed = prepared_simulation_run_for_execution_request(
        prepared,
        SimulationExecutionRequest(
            prepared_payload=None,
            initials=dict(_INITIALS),
            t_span=(0.0, 20.0),
            solver_config=dict(_SOLVER_CONFIG),
            mechanism_text=startup.mechanism_text,
            parameter_overrides={"a": 5.0},
        ),
    )
    result = solve_ode(changed.request)
    stale_control = prepare_simulation_worker_run(
        mechanism_text="\n".join(
            [
                "A <-> B ; kf=1.0, kr=1.0",
                "B -> C ; k=0.2",
            ]
        ),
        initials=dict(_INITIALS),
        t_span=(0.0, 20.0),
        solver_config=dict(_SOLVER_CONFIG),
    )
    fresh_control = prepare_simulation_worker_run(
        mechanism_text=_CONTROL_MECHANISM_TEXT,
        initials=dict(_INITIALS),
        t_span=(0.0, 20.0),
        solver_config=dict(_SOLVER_CONFIG),
    )
    stale_result = solve_ode(stale_control.request)
    fresh_result = solve_ode(fresh_control.request)

    np.testing.assert_allclose(
        np.asarray(result.Y, dtype=float)[:, -1],
        np.asarray(fresh_result.Y, dtype=float)[:, -1],
        rtol=1e-6,
        atol=1e-8,
    )
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(
            np.asarray(result.Y, dtype=float)[:, -1],
            np.asarray(stale_result.Y, dtype=float)[:, -1],
            rtol=1e-6,
            atol=1e-8,
        )


def test_reused_prepared_preview_applies_keq_override_to_rebuilt_rhs() -> None:
    solver_config = dict(_SOLVER_CONFIG)
    solver_config["grid"] = {"N": 80}
    startup_text = "A <-> B ; kf=1.0; Keq=2.0"
    startup = SimulationExecutionRequest(
        prepared_payload=None,
        initials={"A": 1.0, "B": 0.0},
        t_span=(0.0, 20.0),
        solver_config=solver_config,
        mechanism_text=startup_text,
        parameter_overrides={"Keq1": 2.0},
    )
    prepared = prepare_simulation_worker_run(execution_request=startup)

    changed = prepared_simulation_run_for_execution_request(
        prepared,
        SimulationExecutionRequest(
            prepared_payload=None,
            initials={"A": 1.0, "B": 0.0},
            t_span=(0.0, 20.0),
            solver_config=solver_config,
            mechanism_text="A <-> B ; kf=1.0; Keq=8.0",
            parameter_overrides={"Keq1": 8.0},
        ),
    )
    result = solve_ode(changed.request)
    stale_control = prepare_simulation_worker_run(
        mechanism_text=startup_text,
        initials={"A": 1.0, "B": 0.0},
        t_span=(0.0, 20.0),
        solver_config=solver_config,
    )
    fresh_control = prepare_simulation_worker_run(
        mechanism_text="A <-> B ; kf=1.0; Keq=8.0",
        initials={"A": 1.0, "B": 0.0},
        t_span=(0.0, 20.0),
        solver_config=solver_config,
    )
    stale_result = solve_ode(stale_control.request)
    fresh_result = solve_ode(fresh_control.request)

    np.testing.assert_allclose(
        np.asarray(result.Y, dtype=float)[:, -1],
        np.asarray(fresh_result.Y, dtype=float)[:, -1],
        rtol=1e-6,
        atol=1e-8,
    )
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(
            np.asarray(result.Y, dtype=float)[:, -1],
            np.asarray(stale_result.Y, dtype=float)[:, -1],
            rtol=1e-6,
            atol=1e-8,
        )


def test_reused_prepared_preview_accepts_case_insensitive_step_override_names() -> None:
    solver_config = dict(_SOLVER_CONFIG)
    solver_config["grid"] = {"N": 80}
    keq_startup = SimulationExecutionRequest(
        prepared_payload=None,
        initials={"A": 1.0, "B": 0.0},
        t_span=(0.0, 20.0),
        solver_config=solver_config,
        mechanism_text="A <-> B ; kf=1.0; Keq=2.0",
        parameter_overrides={"Keq1": 2.0},
    )
    keq_prepared = prepare_simulation_worker_run(execution_request=keq_startup)
    keq_changed = prepared_simulation_run_for_execution_request(
        keq_prepared,
        SimulationExecutionRequest(
            prepared_payload=None,
            initials={"A": 1.0, "B": 0.0},
            t_span=(0.0, 20.0),
            solver_config=solver_config,
            mechanism_text=keq_startup.mechanism_text,
            parameter_overrides={"keq1": 8.0},
        ),
    )
    keq_result = solve_ode(keq_changed.request)
    keq_control = solve_ode(
        prepare_simulation_worker_run(
            mechanism_text="A <-> B ; kf=1.0; Keq=8.0",
            initials={"A": 1.0, "B": 0.0},
            t_span=(0.0, 20.0),
            solver_config=solver_config,
        ).request
    )

    np.testing.assert_allclose(
        np.asarray(keq_result.Y, dtype=float)[:, -1],
        np.asarray(keq_control.Y, dtype=float)[:, -1],
        rtol=1e-6,
        atol=1e-8,
    )

    rate_startup = SimulationExecutionRequest(
        prepared_payload=None,
        initials={"A": 1.0, "B": 0.0},
        t_span=(0.0, 20.0),
        solver_config=solver_config,
        mechanism_text="A <-> B ; kf=1.0; kr=0.5",
        parameter_overrides={"kf1": 1.0},
    )
    rate_prepared = prepare_simulation_worker_run(execution_request=rate_startup)
    rate_changed = prepared_simulation_run_for_execution_request(
        rate_prepared,
        SimulationExecutionRequest(
            prepared_payload=None,
            initials={"A": 1.0, "B": 0.0},
            t_span=(0.0, 20.0),
            solver_config=solver_config,
            mechanism_text=rate_startup.mechanism_text,
            parameter_overrides={"KF1": 2.0},
        ),
    )
    rate_result = solve_ode(rate_changed.request)
    rate_control = solve_ode(
        prepare_simulation_worker_run(
            mechanism_text="A <-> B ; kf=2.0; kr=0.5",
            initials={"A": 1.0, "B": 0.0},
            t_span=(0.0, 20.0),
            solver_config=solver_config,
        ).request
    )

    np.testing.assert_allclose(
        np.asarray(rate_result.Y, dtype=float)[:, -1],
        np.asarray(rate_control.Y, dtype=float)[:, -1],
        rtol=1e-6,
        atol=1e-8,
    )


def test_reused_prepared_preview_recomputes_algebra_after_step_override() -> None:
    startup = SimulationExecutionRequest(
        prepared_payload=None,
        initials=dict(_INITIALS),
        t_span=(0.0, 20.0),
        solver_config=dict(_SOLVER_CONFIG),
        mechanism_text="\n".join(
            [
                "A <-> B ; kf=1.0, kr=0.01",
                "B -> C ; k=0.2",
                "param a = 5",
                "param kr1 = a*kf1",
            ]
        ),
        parameter_overrides={"kf1": 1.0},
    )
    prepared = prepare_simulation_worker_run(execution_request=startup)

    changed = prepared_simulation_run_for_execution_request(
        prepared,
        SimulationExecutionRequest(
            prepared_payload=None,
            initials=dict(_INITIALS),
            t_span=(0.0, 20.0),
            solver_config=dict(_SOLVER_CONFIG),
            mechanism_text=startup.mechanism_text,
            parameter_overrides={"kf1": 2.0},
        ),
    )
    result = solve_ode(changed.request)
    stale_control = prepare_simulation_worker_run(
        mechanism_text="\n".join(
            [
                "A <-> B ; kf=2.0, kr=5.0",
                "B -> C ; k=0.2",
            ]
        ),
        initials=dict(_INITIALS),
        t_span=(0.0, 20.0),
        solver_config=dict(_SOLVER_CONFIG),
    )
    fresh_control = prepare_simulation_worker_run(
        mechanism_text="\n".join(
            [
                "A <-> B ; kf=2.0, kr=10.0",
                "B -> C ; k=0.2",
            ]
        ),
        initials=dict(_INITIALS),
        t_span=(0.0, 20.0),
        solver_config=dict(_SOLVER_CONFIG),
    )
    stale_result = solve_ode(stale_control.request)
    fresh_result = solve_ode(fresh_control.request)

    np.testing.assert_allclose(
        np.asarray(result.Y, dtype=float)[:, -1],
        np.asarray(fresh_result.Y, dtype=float)[:, -1],
        rtol=1e-6,
        atol=1e-8,
    )
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(
            np.asarray(result.Y, dtype=float)[:, -1],
            np.asarray(stale_result.Y, dtype=float)[:, -1],
            rtol=1e-6,
            atol=1e-8,
        )


def test_execution_request_scalar_override_updates_mutable_algebra_source() -> None:
    mechanism_text = "\n".join(
        [
            "reaction: A -> B; k=1.0",
            "init: A=1.0, B=0.0",
            "param scale = 1.0",
            "param k1 = scale",
        ]
    )
    prepared = prepare_simulation_worker_run(
        execution_request=SimulationExecutionRequest(
            prepared_payload=None,
            initials={},
            t_span=(0.0, 1.0),
            solver_config=dict(_SOLVER_CONFIG),
            mechanism_text=mechanism_text,
            parameter_overrides={"scale": 2.0},
        )
    )

    rate = prepared.mechanism.reactions[0].rate
    result = solve_ode(prepared.request)
    control = solve_ode(
        prepare_simulation_worker_run(
            mechanism_text="reaction: A -> B; k=2.0\ninit: A=1.0, B=0.0",
            initials={},
            t_span=(0.0, 1.0),
            solver_config=dict(_SOLVER_CONFIG),
        ).request
    )

    assert float(rate() if callable(rate) else rate) == pytest.approx(2.0)
    np.testing.assert_allclose(
        np.asarray(result.Y, dtype=float)[:, -1],
        np.asarray(control.Y, dtype=float)[:, -1],
        rtol=1e-6,
        atol=1e-8,
    )


def test_prepared_reuse_scalar_override_updates_mutable_algebra_source() -> None:
    mechanism_text = "\n".join(
        [
            "reaction: A -> B; k=1.0",
            "init: A=1.0, B=0.0",
            "param scale = 1.0",
            "param k1 = scale",
        ]
    )
    prepared = prepare_simulation_worker_run(
        execution_request=SimulationExecutionRequest(
            prepared_payload=None,
            initials={},
            t_span=(0.0, 1.0),
            solver_config=dict(_SOLVER_CONFIG),
            mechanism_text=mechanism_text,
        )
    )

    changed = prepared_simulation_run_for_execution_request(
        prepared,
        SimulationExecutionRequest(
            prepared_payload=None,
            initials={},
            t_span=(0.0, 1.0),
            solver_config=dict(_SOLVER_CONFIG),
            mechanism_text=mechanism_text,
            parameter_overrides={"scale": 2.0},
        ),
    )
    rate = changed.mechanism.reactions[0].rate
    result = solve_ode(changed.request)
    control = solve_ode(
        prepare_simulation_worker_run(
            mechanism_text="reaction: A -> B; k=2.0\ninit: A=1.0, B=0.0",
            initials={},
            t_span=(0.0, 1.0),
            solver_config=dict(_SOLVER_CONFIG),
        ).request
    )

    assert float(rate() if callable(rate) else rate) == pytest.approx(2.0)
    np.testing.assert_allclose(
        np.asarray(result.Y, dtype=float)[:, -1],
        np.asarray(control.Y, dtype=float)[:, -1],
        rtol=1e-6,
        atol=1e-8,
    )


def test_prepare_fitting_execution_context_rejects_algebra_owned_rate_as_requested_fit_parameter() -> None:
    with pytest.raises(FitSimulationError, match="algebra-owned mechanism parameter"):
        prepare_fitting_execution_context(
            mechanism_text="\n".join(
                [
                    "reaction: A -> B; k=1.0",
                    "init: A=1.0, B=0.0",
                    "param scale = 1.0",
                    "param k1 = scale",
                ]
            ),
            param_names=["k1"],
            t_end=1.0,
            num_points=5,
            temperature_K=298.15,
            solver="BDF",
            rtol=1e-8,
            atol=1e-10,
            use_sparse_jacobian=False,
            wegscheider_cyclicity_enabled=False,
            initial_prefix="init:",
        )


def test_build_prepared_simulation_func_rejects_algebra_owned_rate_as_fit_parameter() -> None:
    simulation_func = build_prepared_simulation_func(
        mechanism_text="\n".join(
            [
                "reaction: A -> B; k=1.0",
                "init: A=1.0, B=0.0",
                "param scale = 1.0",
                "param k1 = scale",
            ]
        ),
        param_names=["k1"],
        t_end=1.0,
        num_points=5,
        temperature_K=298.15,
        solver="BDF",
        rtol=1e-8,
        atol=1e-10,
        use_sparse_jacobian=False,
        wegscheider_cyclicity_enabled=False,
        initial_prefix="init:",
    )

    with pytest.raises(FitSimulationError, match="algebra-owned mechanism parameter"):
        simulation_func({"k1": 2.0})


def test_prepare_bound_mechanism_keeps_internal_algebra_owned_rate_binding_for_scalar_fit_dimension() -> None:
    bound = prepare_bound_mechanism(
        mechanism_text="\n".join(
            [
                "reaction: A -> B; k=1.0",
                "init: A=1.0, B=0.0",
                "param scale = 1.0",
                "param k1 = scale",
            ]
        ),
        param_names=["scale"],
        temperature_K=298.15,
        initials={},
        use_advanced_dsl=True,
        wegscheider_cyclicity_enabled=False,
    )

    bound.bindings["scale"].set(2.0)
    from kindred.core.simulator.parameter_algebra import apply_parameter_algebra_to_mechanism

    apply_parameter_algebra_to_mechanism(
        "\n".join(
            [
                "reaction: A -> B; k=1.0",
                "init: A=1.0, B=0.0",
                "param scale = 1.0",
                "param k1 = scale",
            ]
        ),
        mechanism=bound.mechanism,
        require_mutable=True,
    )

    assert "scale" in bound.bindings
    assert "k1" in bound.bindings
    assert bound.bindings["k1"]() == pytest.approx(2.0)
