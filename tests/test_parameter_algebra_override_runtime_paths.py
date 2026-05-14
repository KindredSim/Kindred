from __future__ import annotations

import numpy as np
import pytest

from kindred.core.algebra.symbol_table import build_algebra_symbol_table
from kindred.core.batch_parallel import run_batch_simulation_task
from kindred.core.exceptions import FitSimulationError
from kindred.core.fitting_evaluation import (
    SerialFittingEvaluator,
    evaluate_fitting_series,
    prepare_fitting_execution_context,
)
from kindred.core.intervention_schedule import parse_intervention_schedule_from_dsl
from kindred.core.simulation_plan import SimulationAlgebraPolicy, SimulationPlan
from kindred.core.simulation_preparation import (
    SimulationExecutionRequest,
    SimulationPreparationError,
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


def _run_batch_explicit_payload(mechanism_text: str, *, parameter_overrides: dict[str, float]) -> dict:
    execution_request = {
        "prepared_payload": None,
        "initials": dict(_INITIALS),
        "t_span": (0.0, 20.0),
        "solver_config": dict(_SOLVER_CONFIG),
        "mechanism_text": mechanism_text,
        "parameter_overrides": dict(parameter_overrides),
        "simulation_identity": {
            "schema_id": "override-runtime-regression",
            "param_fingerprint": mechanism_text,
        },
    }
    return run_batch_simulation_task(
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


def test_batch_parallel_rejects_indexed_k_override_for_reversible_step() -> None:
    payload = _run_batch_explicit_payload(
        "A <-> B ; kf=1.0, kr=0.5\nB -> C ; k=0.2",
        parameter_overrides={"K1": 8.0},
    )

    assert payload["success"] is False
    assert payload["error"]["details"]["stage"] == "parameter_overrides"
    assert "K1" in payload["error"]["message"]
    assert "Keq1" in payload["error"]["message"]


@pytest.mark.parametrize("name", ["k1", "kf2", "kr2", "Keq2", "K999", "kf999", "kr999", "Keq999"])
def test_runtime_rejects_invalid_exact_indexed_protected_override_names(name: str) -> None:
    with pytest.raises(SimulationPreparationError, match=name):
        prepare_simulation_worker_run(
            execution_request=SimulationExecutionRequest(
                prepared_payload=None,
                initials=dict(_INITIALS),
                t_span=(0.0, 20.0),
                solver_config=dict(_SOLVER_CONFIG),
                mechanism_text="A <-> B ; kf=1.0, kr=0.5\nB -> C ; k=0.2",
                parameter_overrides={name: 8.0},
            ),
        )


@pytest.mark.parametrize("name", ["k", "kf", "kr", "K", "Keq"])
def test_runtime_rejects_bare_step_key_override_names(name: str) -> None:
    with pytest.raises(SimulationPreparationError, match="Bare step-local DSL keys"):
        prepare_simulation_worker_run(
            execution_request=SimulationExecutionRequest(
                prepared_payload=None,
                initials=dict(_INITIALS),
                t_span=(0.0, 20.0),
                solver_config=dict(_SOLVER_CONFIG),
                mechanism_text="A -> B ; k=1.0\nB <-> C ; kf=1.0, kr=0.5",
                parameter_overrides={name: 8.0},
            ),
        )


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


def test_reused_prepared_preview_keq_symbol_table_uses_mutable_override() -> None:
    solver_config = dict(_SOLVER_CONFIG)
    solver_config["grid"] = {"N": 80}
    startup = SimulationExecutionRequest(
        prepared_payload=None,
        initials={"A": 1.0, "B": 0.0},
        t_span=(0.0, 20.0),
        solver_config=solver_config,
        mechanism_text="A <-> B ; kf=1.0; Keq=2.0",
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
            mechanism_text=startup.mechanism_text,
            parameter_overrides={"keq1": 8.0},
        ),
    )
    symtab = build_algebra_symbol_table(changed.mechanism)

    assert symtab.get("Keq1") == pytest.approx(8.0)
    with pytest.raises(KeyError):
        symtab.get("K1")


def test_reused_prepared_preview_implicit_keq_symbol_table_uses_current_rates() -> None:
    solver_config = dict(_SOLVER_CONFIG)
    solver_config["grid"] = {"N": 80}
    startup = SimulationExecutionRequest(
        prepared_payload=None,
        initials={"A": 1.0, "B": 0.0},
        t_span=(0.0, 20.0),
        solver_config=solver_config,
        mechanism_text="A <-> B ; kf=1.0; kr=0.5",
        parameter_overrides={"kf1": 1.0},
    )
    prepared = prepare_simulation_worker_run(execution_request=startup)

    changed = prepared_simulation_run_for_execution_request(
        prepared,
        SimulationExecutionRequest(
            prepared_payload=None,
            initials={"A": 1.0, "B": 0.0},
            t_span=(0.0, 20.0),
            solver_config=solver_config,
            mechanism_text=startup.mechanism_text,
            parameter_overrides={"kf1": 2.0},
        ),
    )
    symtab = build_algebra_symbol_table(changed.mechanism)

    assert symtab.get("kf1") == pytest.approx(2.0)
    assert symtab.get("kr1") == pytest.approx(0.5)
    assert symtab.get("Keq1") == pytest.approx(4.0)
    with pytest.raises(KeyError):
        symtab.get("K1")


def test_algebra_symbol_table_rejects_protected_indexed_scalar_metadata_names() -> None:
    from kindred.core.simulator.dsl import parse_dsl_to_mechanism

    mechanism = parse_dsl_to_mechanism(
        "\n".join(
            [
                "reaction: A -> B; k=1.0",
                "initial: A=1.0",
                "initial: B=0.0",
            ]
        ),
        initials={},
    )
    mechanism.metadata.setdefault("scalar_params", {})["K1"] = 3.0

    with pytest.raises(ValueError, match="K1.*protected indexed"):
        build_algebra_symbol_table(mechanism)


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


def test_reused_prepared_preview_rejects_K1_on_reversible_step_without_irreversible_k1_and_suggests_existing_canonical_names() -> None:
    solver_config = dict(_SOLVER_CONFIG)
    solver_config["grid"] = {"N": 80}
    prepared = prepare_simulation_worker_run(
        execution_request=SimulationExecutionRequest(
            prepared_payload=None,
            initials={"A": 1.0, "B": 0.0},
            t_span=(0.0, 20.0),
            solver_config=solver_config,
            mechanism_text="A <-> B ; kf=1.0; kr=0.5",
        )
    )

    with pytest.raises(SimulationPreparationError) as exc_info:
        prepared_simulation_run_for_execution_request(
            prepared,
            SimulationExecutionRequest(
                prepared_payload=None,
                initials={"A": 1.0, "B": 0.0},
                t_span=(0.0, 20.0),
                solver_config=solver_config,
                mechanism_text="A <-> B ; kf=1.0; kr=0.5",
                parameter_overrides={"K1": 8.0},
            ),
        )
    message = str(exc_info.value)
    assert "K1" in message
    assert "not a valid indexed parameter identifier" in message
    assert "kf1" in message
    assert "kr1" in message
    assert "Keq1" in message


def test_reused_prepared_preview_rejects_K1_schedule_name_on_reversible_step_without_irreversible_k1_and_suggests_existing_canonical_names() -> None:
    solver_config = dict(_SOLVER_CONFIG)
    solver_config["grid"] = {"N": 5}
    mechanism_text = "\n".join(
        [
            "equilibrium: A <-> B; kf=0.1; kr=0.1",
            "initial: A=1.0",
            "initial: B=0.0",
            "intervention: op=add; species=A; time=1.0; amount_param=K1",
        ]
    )
    with pytest.raises(SimulationPreparationError) as exc_info:
        prepare_simulation_worker_run(
            execution_request=SimulationExecutionRequest(
                prepared_payload=None,
                initials={"A": 1.0, "B": 0.0},
                t_span=(0.0, 2.0),
                solver_config=solver_config,
                mechanism_text=mechanism_text,
                intervention_schedule=parse_intervention_schedule_from_dsl(mechanism_text),
                parameter_overrides={"K1": 1.0},
            )
        )
    message = str(exc_info.value)
    assert "K1" in message
    assert "not a valid indexed parameter identifier" in message
    assert "kf1" in message
    assert "kr1" in message
    assert "Keq1" in message


def test_reused_prepared_preview_routes_protected_schedule_name_through_canonical_mechanism_parameter() -> None:
    solver_config = dict(_SOLVER_CONFIG)
    solver_config["grid"] = {"N": 5}
    mechanism_text = "\n".join(
        [
            "reaction: A -> B; k=0.1",
            "initial: A=1.0",
            "initial: B=0.0",
            "intervention: op=add; species=A; time=1.0; amount_param=K1",
        ]
    )
    prepared = prepare_simulation_worker_run(
        execution_request=SimulationExecutionRequest(
            prepared_payload=None,
            initials={"A": 1.0, "B": 0.0},
                t_span=(0.0, 2.0),
                solver_config=solver_config,
                mechanism_text=mechanism_text,
                intervention_schedule=parse_intervention_schedule_from_dsl(mechanism_text),
                parameter_overrides={"K1": 3.0},
            )
        )

    assert prepared.request.intervention_schedule.to_payload()["instant_events"][0]["amount"] == pytest.approx(3.0)
    assert (
        prepared.unresolved_intervention_schedule.to_payload()["instant_events"][0]["amount_param"]
        == "K1"
    )


def test_chained_prepared_preview_preserves_request_local_unresolved_schedule_source() -> None:
    solver_config = dict(_SOLVER_CONFIG)
    solver_config["grid"] = {"N": 5}
    startup_text = "\n".join(
        [
            "equilibrium: A <-> B; kf=0.1; kr=0.1",
            "initial: A=1.0",
            "initial: B=0.0",
            "intervention: op=add; species=A; time=1.0; amount_param=dose1",
        ]
    )
    request_text = "\n".join(
        [
            "equilibrium: A <-> B; kf=0.1; kr=0.1",
            "initial: A=1.0",
            "initial: B=0.0",
            "intervention: op=add; species=A; time=1.0; amount_param=dose2",
        ]
    )
    prepared = prepare_simulation_worker_run(
        execution_request=SimulationExecutionRequest(
            prepared_payload=None,
            initials={"A": 1.0, "B": 0.0},
                t_span=(0.0, 2.0),
                solver_config=solver_config,
                mechanism_text=startup_text,
                intervention_schedule=parse_intervention_schedule_from_dsl(startup_text),
                parameter_overrides={"dose1": 1.0},
            )
        )

    changed = prepared_simulation_run_for_execution_request(
        prepared,
        SimulationExecutionRequest(
            prepared_payload=None,
            initials={"A": 1.0, "B": 0.0},
            t_span=(0.0, 2.0),
            solver_config=solver_config,
            mechanism_text=request_text,
            intervention_schedule=parse_intervention_schedule_from_dsl(request_text),
            parameter_overrides={"dose2": 3.0},
        ),
    )
    chained = prepared_simulation_run_for_execution_request(
        changed,
        SimulationExecutionRequest(
            prepared_payload=None,
            initials={"A": 1.0, "B": 0.0},
            t_span=(0.0, 2.0),
            solver_config=solver_config,
            mechanism_text="",
            intervention_schedule=changed.unresolved_intervention_schedule,
            parameter_overrides={"dose2": 5.0},
        ),
    )

    assert changed.unresolved_intervention_schedule.to_payload()["instant_events"][0]["amount_param"] == "dose2"
    assert chained.request.intervention_schedule.to_payload()["instant_events"][0]["amount"] == pytest.approx(5.0)


def test_structured_prepared_payload_uses_explicit_request_schedule_for_ordinary_schedule_name() -> None:
    solver_config = dict(_SOLVER_CONFIG)
    solver_config["grid"] = {"N": 5}
    mechanism_text = "\n".join(
        [
            "equilibrium: A <-> B; kf=0.1; kr=0.1",
            "initial: A=1.0",
            "initial: B=0.0",
            "intervention: op=add; species=A; time=1.0; amount_param=dose",
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

    prepared = prepare_simulation_worker_run(
        prepared_payload=bound.as_serializable_execution_payload(),
        execution_request=SimulationExecutionRequest(
            prepared_payload=None,
            initials={"A": 1.0, "B": 0.0},
                t_span=(0.0, 2.0),
                solver_config=solver_config,
                mechanism_text="",
                intervention_schedule=bound.unresolved_intervention_schedule,
                parameter_overrides={"dose": 4.0},
            ),
        )

    assert prepared.request.intervention_schedule.to_payload()["instant_events"][0]["amount"] == pytest.approx(4.0)


def test_symbolic_snapshot_binds_only_partitioned_mechanism_override_names(monkeypatch) -> None:
    import kindred.core.simulation_preparation as simulation_preparation

    solver_config = simulation_preparation._build_solver_config(
        solver_input="BDF",
        rtol=1e-6,
        atol=1e-12,
        grid={"N": 5},
        use_sparse_jacobian=True,
        wegscheider_cyclicity_enabled=False,
    )
    mechanism_text = "\n".join(
        [
            "equilibrium: A <-> B; kf=0.1; kr=0.1",
            "intervention: op=add; species=A; time=1.0; amount_param=dose",
        ]
    )
    original_bind = simulation_preparation._bind_parameters_to_mechanism
    seen_names: list[list[str]] = []

    def _recording_bind(mechanism, names):
        seen_names.append(list(names))
        return original_bind(mechanism, names)

    monkeypatch.setattr(simulation_preparation, "_bind_parameters_to_mechanism", _recording_bind)

    simulation_preparation._symbolic_jacobian_snapshot_values_for_execution_text(
        mechanism_text=mechanism_text,
        prepared_solver_config=solver_config,
        temperature_K=298.15,
        parameter_overrides={"dose": 4.0, "kf1": 0.2},
        parameter_symbols=("kf1", "kr1"),
    )

    assert seen_names
    assert all("dose" not in names for names in seen_names)
    assert any("kf1" in names for names in seen_names)


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


def test_reused_prepared_preview_applies_shared_schedule_scalar_parameter_override() -> None:
    mechanism_text = "\n".join(
        [
            "reaction: A -> B; k=1.0",
            "initial: A=1.0",
            "initial: B=0.0",
            "param scale = 1.0",
            "param k1 = scale",
            "intervention: op=add; species=A; time=1.0; amount_param=scale",
        ]
    )
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
        t_span=(0.0, 2.0),
        solver_config=dict(solver_config),
        mechanism_text=mechanism_text,
        parameter_overrides={"scale": 1.0},
        intervention_schedule=parse_intervention_schedule_from_dsl(mechanism_text),
    )
    prepared = prepare_simulation_worker_run(execution_request=startup)

    changed = prepared_simulation_run_for_execution_request(
        prepared,
        SimulationExecutionRequest(
            prepared_payload=None,
            initials={"A": 1.0, "B": 0.0},
            t_span=(0.0, 2.0),
            solver_config=dict(solver_config),
            mechanism_text=mechanism_text,
            parameter_overrides={"scale": 2.0},
            intervention_schedule=parse_intervention_schedule_from_dsl(mechanism_text),
        ),
    )
    result = solve_ode(changed.request)
    fresh_control_text = "\n".join(
        [
            "reaction: A -> B; k=2.0",
            "initial: A=1.0",
            "initial: B=0.0",
            "intervention: op=add; species=A; time=1.0; amount=2.0",
        ]
    )
    stale_control_text = "\n".join(
        [
            "reaction: A -> B; k=1.0",
            "initial: A=1.0",
            "initial: B=0.0",
            "intervention: op=add; species=A; time=1.0; amount=2.0",
        ]
    )
    fresh_control = prepare_simulation_worker_run(
        execution_request=SimulationExecutionRequest(
            prepared_payload=None,
            initials={"A": 1.0, "B": 0.0},
            t_span=(0.0, 2.0),
            solver_config=dict(solver_config),
            mechanism_text=fresh_control_text,
            intervention_schedule=parse_intervention_schedule_from_dsl(fresh_control_text),
        )
    )
    stale_control = prepare_simulation_worker_run(
        execution_request=SimulationExecutionRequest(
            prepared_payload=None,
            initials={"A": 1.0, "B": 0.0},
            t_span=(0.0, 2.0),
            solver_config=dict(solver_config),
            mechanism_text=stale_control_text,
            intervention_schedule=parse_intervention_schedule_from_dsl(stale_control_text),
        )
    )
    fresh_result = solve_ode(fresh_control.request)
    stale_result = solve_ode(stale_control.request)

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


def test_build_prepared_simulation_func_accepts_indexed_k_direct_spelling_for_irreversible_name() -> None:
    simulation_func = build_prepared_simulation_func(
        mechanism_text="\n".join(
            [
                "reaction: A -> B; k=0.1",
                "init: A=1.0, B=0.0",
            ]
        ),
        param_names=["k1"],
        t_end=1.0,
        num_points=6,
        temperature_K=298.15,
        solver="BDF",
        rtol=1e-8,
        atol=1e-10,
        use_sparse_jacobian=False,
        wegscheider_cyclicity_enabled=False,
        initial_prefix="init:",
    )

    payload = simulation_func({"K1": 2.0})

    assert np.asarray(payload.species["A"], dtype=float)[-1] < 0.5


def test_build_prepared_simulation_func_applies_shared_schedule_scalar_parameter_to_rate() -> None:
    simulation_func = build_prepared_simulation_func(
        mechanism_text="\n".join(
            [
                "reaction: A -> B; k=1.0",
                "init: A=1.0, B=0.0",
                "param scale = 1.0",
                "param k1 = scale",
                "intervention: op=add; species=A; time=1.0; amount_param=scale",
            ]
        ),
        param_names=["scale"],
        t_end=2.0,
        num_points=20,
        temperature_K=298.15,
        solver="BDF",
        rtol=1e-8,
        atol=1e-10,
        use_sparse_jacobian=False,
        wegscheider_cyclicity_enabled=False,
        initial_prefix="init:",
    )
    stale_control = build_prepared_simulation_func(
        mechanism_text="\n".join(
            [
                "reaction: A -> B; k=1.0",
                "init: A=1.0, B=0.0",
                "intervention: op=add; species=A; time=1.0; amount=2.0",
            ]
        ),
        param_names=[],
        t_end=2.0,
        num_points=20,
        temperature_K=298.15,
        solver="BDF",
        rtol=1e-8,
        atol=1e-10,
        use_sparse_jacobian=False,
        wegscheider_cyclicity_enabled=False,
        initial_prefix="init:",
    )({})
    fresh_control = build_prepared_simulation_func(
        mechanism_text="\n".join(
            [
                "reaction: A -> B; k=2.0",
                "init: A=1.0, B=0.0",
                "intervention: op=add; species=A; time=1.0; amount=2.0",
            ]
        ),
        param_names=[],
        t_end=2.0,
        num_points=20,
        temperature_K=298.15,
        solver="BDF",
        rtol=1e-8,
        atol=1e-10,
        use_sparse_jacobian=False,
        wegscheider_cyclicity_enabled=False,
        initial_prefix="init:",
    )({})

    result = simulation_func({"scale": 2.0})

    np.testing.assert_allclose(
        np.asarray(result.species["A"], dtype=float)[-1],
        np.asarray(fresh_control.species["A"], dtype=float)[-1],
        rtol=1e-6,
        atol=1e-8,
    )
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(
            np.asarray(result.species["A"], dtype=float)[-1],
            np.asarray(stale_control.species["A"], dtype=float)[-1],
            rtol=1e-6,
            atol=1e-8,
        )


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


def test_prepare_bound_mechanism_accepts_indexed_k_direct_spelling_for_irreversible_name() -> None:
    bound = prepare_bound_mechanism(
        mechanism_text="\n".join(
            [
                "reaction: A -> B; k=1.0",
                "init: A=1.0, B=0.0",
            ]
        ),
        param_names=["K1"],
        temperature_K=298.15,
        initials={},
        use_advanced_dsl=True,
        wegscheider_cyclicity_enabled=False,
    )

    assert "k1" in bound.bindings
    assert "K1" not in bound.bindings


def test_prepare_bound_mechanism_rejects_implicit_keq_as_fit_parameter() -> None:
    with pytest.raises(FitSimulationError, match="Keq1.*not writable"):
        prepare_bound_mechanism(
            mechanism_text="\n".join(
                [
                    "equilibrium: A <-> B; kf=6.0; kr=2.0",
                    "init: A=1.0, B=0.0",
                ]
            ),
            param_names=["Keq1"],
            temperature_K=298.15,
            initials={},
            use_advanced_dsl=True,
            wegscheider_cyclicity_enabled=False,
        )


def test_prepare_simulation_worker_run_rejects_implicit_keq_runtime_override() -> None:
    with pytest.raises(SimulationPreparationError, match="Keq1.*not writable"):
        prepare_simulation_worker_run(
            execution_request=SimulationExecutionRequest(
                prepared_payload=None,
                mechanism_text="\n".join(
                    [
                        "equilibrium: A <-> B; kf=6.0; kr=2.0",
                        "init: A=1.0, B=0.0",
                    ]
                ),
                initials={"A": 1.0, "B": 0.0},
                t_span=(0.0, 1.0),
                solver_config={"solver": "BDF", "grid": {"N": 4}},
                parameter_overrides={"Keq1": 4.0},
            )
        )
