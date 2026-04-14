from __future__ import annotations

import numpy as np
import pytest


def _fake_solve_constant(request):
    from kindred.core.simulator.solvers import SimulationOutput

    y0 = np.asarray(request.y0, dtype=float).reshape(-1)
    if getattr(request, "t_eval", None) is not None:
        tt = np.asarray(request.t_eval, dtype=float).reshape(-1)
    else:
        grid_n = int((request.grid or {}).get("N") or 5)
        t0, t1 = request.t_span
        tt = np.linspace(float(t0), float(t1), grid_n, dtype=float)
    yy = np.tile(y0.reshape(-1, 1), (1, tt.size))
    return SimulationOutput(t=tt, Y=yy, provenance={})


def test_prepared_simulation_emits_algebra_observables(monkeypatch):
    monkeypatch.setattr("kindred.core.simulator.solvers.solve_ode", _fake_solve_constant)

    from kindred.core.simulation_preparation import build_prepared_simulation_func

    dsl = "\n".join(
        [
            "reaction: A -> B; k=0.2",
            "initial: A=1.0",
            "initial: B=0.0",
            "# Algebra",
            "total = [A] + [B]",
        ]
    )
    prepared = build_prepared_simulation_func(
        mechanism_text=dsl,
        param_names=["k1"],
        t_end=1.0,
        num_points=6,
        temperature_K=298.15,
        solver="BDF",
        rtol=1e-6,
        atol=1e-12,
        use_sparse_jacobian=False,
        initial_prefix="init:",
    )

    sim = prepared({"k1": 0.2, "init:A": 1.0})
    assert "total" in sim["species"]
    assert sim["species"]["total"].shape == np.asarray(sim["t"]).shape


def test_prepared_simulation_algebra_initial_reflects_init_override(monkeypatch):
    monkeypatch.setattr("kindred.core.simulator.solvers.solve_ode", _fake_solve_constant)

    from kindred.core.simulation_preparation import build_prepared_simulation_func

    dsl = "\n".join(
        [
            "reaction: A -> B; k=0.2",
            "initial: A=2.0",
            "initial: B=0.0",
            "# Algebra",
            "A0 = [A]_0 + 0*[A]",
        ]
    )
    prepared = build_prepared_simulation_func(
        mechanism_text=dsl,
        param_names=["k1"],
        t_end=1.0,
        num_points=6,
        temperature_K=298.15,
        solver="BDF",
        rtol=1e-6,
        atol=1e-12,
        use_sparse_jacobian=False,
        initial_prefix="init:",
    )

    sim = prepared({"k1": 0.2, "init:A": 3.0})
    assert np.allclose(sim["species"]["A0"], 3.0)


def test_prepared_simulation_algebra_sees_scalar_param(monkeypatch):
    monkeypatch.setattr("kindred.core.simulator.solvers.solve_ode", _fake_solve_constant)

    from kindred.core.simulation_preparation import build_prepared_simulation_func

    dsl = "\n".join(
        [
            "reaction: A -> B; k=0.2",
            "initial: A=1.0",
            "initial: B=0.0",
            "# Algebra",
            "param a = 2",
            "obs = a * [A]",
        ]
    )
    prepared = build_prepared_simulation_func(
        mechanism_text=dsl,
        param_names=["k1"],
        t_end=1.0,
        num_points=6,
        temperature_K=298.15,
        solver="BDF",
        rtol=1e-6,
        atol=1e-12,
        use_sparse_jacobian=False,
        initial_prefix="init:",
    )

    sim2 = prepared({"k1": 0.2, "a": 2.0})
    sim3 = prepared({"k1": 0.2, "a": 3.0})
    assert np.allclose(sim2["species"]["obs"], 2.0)
    assert np.allclose(sim3["species"]["obs"], 3.0)


def test_prepared_simulation_uses_patched_solver_after_module_import(monkeypatch):
    import kindred.core.simulation_preparation as simulation_preparation

    monkeypatch.setattr("kindred.core.simulator.solvers.solve_ode", _fake_solve_constant)

    prepared = simulation_preparation.build_prepared_simulation_func(
        mechanism_text="\n".join(
            [
                "reaction: A -> B; k=0.2",
                "initial: A=1.0",
                "initial: B=0.0",
                "# Algebra",
                "param a = 2",
                "obs = a * [A]",
            ]
        ),
        param_names=["k1"],
        t_end=1.0,
        num_points=6,
        temperature_K=298.15,
        solver="BDF",
        rtol=1e-6,
        atol=1e-12,
        use_sparse_jacobian=False,
        initial_prefix="init:",
    )

    sim = prepared({"k1": 0.2, "a": 2.0})
    assert np.allclose(sim["species"]["obs"], 2.0)


def test_prepared_simulation_rejects_baseline_time_refs_for_fitting(monkeypatch):
    monkeypatch.setattr("kindred.core.simulator.solvers.solve_ode", _fake_solve_constant)

    from kindred.core.simulation_preparation import build_prepared_simulation_func
    from kindred.core.exceptions import FitSimulationError

    dsl = "\n".join(
        [
            "reaction: A -> B; k=0.2",
            "initial: A=1.0",
            "initial: B=0.0",
            "# Algebra",
            "bad = [A](T0)",
        ]
    )
    prepared = build_prepared_simulation_func(
        mechanism_text=dsl,
        param_names=["k1"],
        t_end=1.0,
        num_points=6,
        temperature_K=298.15,
        solver="BDF",
        rtol=1e-6,
        atol=1e-12,
        use_sparse_jacobian=False,
        initial_prefix="init:",
    )

    with pytest.raises(FitSimulationError, match=r"\(T0\)"):
        _ = prepared({"k1": 0.2, "init:A": 1.0})


def test_prepared_simulation_parses_dsl_once_even_with_algebra(monkeypatch):
    monkeypatch.setattr("kindred.core.simulator.solvers.solve_ode", _fake_solve_constant)

    from kindred.core.simulation_preparation import build_prepared_simulation_func
    from kindred.core.simulator.dsl import parse_dsl_to_mechanism as _orig_parse

    counts = {"parse": 0}

    def _counting_parse(*args, **kwargs):
        counts["parse"] += 1
        return _orig_parse(*args, **kwargs)

    monkeypatch.setattr("kindred.core.simulator.dsl.parse_dsl_to_mechanism", _counting_parse)

    dsl = "\n".join(
        [
            "reaction: A -> B; k=0.2",
            "initial: A=1.0",
            "initial: B=0.0",
            "# Algebra",
            "total = [A] + [B]",
        ]
    )
    prepared = build_prepared_simulation_func(
        mechanism_text=dsl,
        param_names=["k1"],
        t_end=1.0,
        num_points=6,
        temperature_K=298.15,
        solver="BDF",
        rtol=1e-6,
        atol=1e-12,
        use_sparse_jacobian=False,
        initial_prefix="init:",
    )

    for k in (0.1, 0.2, 0.3):
        sim = prepared({"k1": float(k), "init:A": 1.0})
        assert "total" in sim["species"]

    assert counts["parse"] == 1


def test_local_fitting_objective_accepts_algebra_target(monkeypatch):
    monkeypatch.setattr("kindred.core.simulator.solvers.solve_ode", _fake_solve_constant)

    from kindred.core.fitting_objective import build_fitting_objective

    dsl = "\n".join(
        [
            "reaction: A -> B; k=0.2",
            "initial: A=1.0",
            "initial: B=0.0",
            "# Algebra",
            "total = [A] + [B]",
        ]
    )
    t_exp = np.linspace(0.0, 1.0, 6)
    y_exp = np.ones_like(t_exp)

    obj = build_fitting_objective(
        mechanism_text=dsl,
        param_names=["k1"],
        t_exp=t_exp,
        y_exp=y_exp,
        target_species="total",
        temperature_K=298.15,
        initials={"A": 1.0, "B": 0.0},
        solver="BDF",
        rtol=1e-6,
        atol=1e-12,
    )

    residuals = obj(np.array([0.2], dtype=float))
    assert np.allclose(residuals, 0.0)
    assert np.allclose(obj.last_model, y_exp)
