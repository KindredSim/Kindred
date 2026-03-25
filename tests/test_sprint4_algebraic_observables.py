from __future__ import annotations

import numpy as np


def test_detect_unknown_scalar_identifiers_uses_algebra_ast():
    from kindred.core.algebra.observable_introspection import detect_unknown_scalar_identifiers

    unknown = detect_unknown_scalar_identifiers(
        "scale * ([A] + [B]) + R + sin(1) + k1",
        observable_name="total",
        known_identifiers={"k1", "already_known"},
        mechanism_species={"A", "B"},
    )
    assert unknown == {"scale"}


def test_detect_unknown_scalar_identifiers_flags_bare_species_names():
    from kindred.core.algebra.observable_introspection import analyze_observable_expression

    analysis = analyze_observable_expression("A + [A] + foo")
    assert analysis.identifiers == {"A", "foo"}
    assert analysis.species_refs == {"A"}


def test_extract_observables_from_algebra_text_includes_let_and_gui_style():
    from kindred.core.algebra.observable_introspection import extract_observables_from_algebra_text

    algebra_text = "\n".join(
        [
            "param a = 2.0",
            "let total_PBMP = [PBMPBPIN] + [PBMP]",
            "selectivity = [PBMP] / max([PBMP] + [pinBOH], 1e-18)",
            "# comment",
        ]
    )
    obs = extract_observables_from_algebra_text(algebra_text)
    assert obs["total_PBMP"] == "[PBMPBPIN] + [PBMP]"
    assert "max(" in obs["selectivity"]
    assert "a" not in obs


def test_global_fit_targets_algebra_observable_series_shared_scalar(monkeypatch):
    from kindred.core.analysis.global_fitting import fit_global
    from kindred.core.simulation_preparation import build_prepared_simulation_func

    def _fake_solve_constant(request):
        from kindred.core.simulator.solvers import SimulationOutput

        y0 = np.asarray(request.y0, dtype=float).reshape(-1)
        grid_n = int((request.grid or {}).get("N") or 6)
        t0, t1 = request.t_span
        tt = np.linspace(float(t0), float(t1), grid_n, dtype=float)
        yy = np.tile(y0.reshape(-1, 1), (1, tt.size))
        return SimulationOutput(t=tt, Y=yy, provenance={})

    monkeypatch.setattr("kindred.core.simulator.solvers.solve_ode", _fake_solve_constant)

    dsl = "\n".join(
        [
            "reaction: A -> B; k=0.2",
            "initial: A=1.0",
            "initial: B=0.0",
            "# Algebra",
            "param a = 2.0",
            "signal = a * [A]",
        ]
    )
    prepared = build_prepared_simulation_func(
        mechanism_text=dsl,
        param_names=["k1"],
        t_end=1.0,
        num_points=6,
        temperature_K=298.15,
        solver="LSODA",
        rtol=1e-6,
        atol=1e-12,
        use_sparse_jacobian=False,
        initial_prefix="init:",
    )

    t_exp = np.linspace(0.0, 1.0, 6)
    y_exp = np.full_like(t_exp, 2.0)
    datasets = [{"id": "ds1", "t": t_exp, "y": y_exp, "species": "signal"}]

    result = fit_global(prepared, datasets, {"a": 2.0}, method="lm", max_nfev=2)
    assert result.success
    assert result.objective_residuals is not None
    assert np.all(np.isfinite(result.objective_residuals))


def test_global_fit_targets_algebra_observable_series_per_dataset_scalar(monkeypatch):
    from kindred.core.analysis.global_fitting import fit_global
    from kindred.core.simulation_preparation import build_prepared_simulation_func

    def _fake_solve_constant(request):
        from kindred.core.simulator.solvers import SimulationOutput

        y0 = np.asarray(request.y0, dtype=float).reshape(-1)
        grid_n = int((request.grid or {}).get("N") or 6)
        t0, t1 = request.t_span
        tt = np.linspace(float(t0), float(t1), grid_n, dtype=float)
        yy = np.tile(y0.reshape(-1, 1), (1, tt.size))
        return SimulationOutput(t=tt, Y=yy, provenance={})

    monkeypatch.setattr("kindred.core.simulator.solvers.solve_ode", _fake_solve_constant)

    dsl = "\n".join(
        [
            "reaction: A -> B; k=0.2",
            "initial: A=1.0",
            "initial: B=0.0",
            "# Algebra",
            "param a = 1.0",
            "signal = a * [A]",
        ]
    )
    prepared = build_prepared_simulation_func(
        mechanism_text=dsl,
        param_names=["k1"],
        t_end=1.0,
        num_points=6,
        temperature_K=298.15,
        solver="LSODA",
        rtol=1e-6,
        atol=1e-12,
        use_sparse_jacobian=False,
        initial_prefix="init:",
    )

    t_exp = np.linspace(0.0, 1.0, 6)
    datasets = [
        {"id": "ds1", "t": t_exp, "y": np.full_like(t_exp, 2.0), "species": "signal"},
        {"id": "ds2", "t": t_exp, "y": np.full_like(t_exp, 3.0), "species": "signal"},
    ]
    dataset_variable_params = {
        "ds1": {"a": {"initial": 2.0, "min": -np.inf, "max": np.inf}},
        "ds2": {"a": {"initial": 3.0, "min": -np.inf, "max": np.inf}},
    }

    result = fit_global(
        prepared,
        datasets,
        {},
        dataset_variable_params=dataset_variable_params,
        method="lm",
        max_nfev=2,
    )
    assert result.success
    assert result.objective_residuals is not None
    assert np.all(np.isfinite(result.objective_residuals))
