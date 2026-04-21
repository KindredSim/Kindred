from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.gui
def test_global_fit_simulation_func_parses_once_per_session(main_window, monkeypatch):
    """
    Regression: global-fit simulation callbacks must not re-parse the mechanism DSL
    for every evaluation. Parsing should happen O(1) times per fit session.
    """
    # Seed a single dataset so _run_global_fit avoids dataset selection UI.
    t = np.linspace(0.0, 1.0, 6)
    y = np.exp(-0.2 * t)
    dataset = {"t": t, "species": {"A": y}}
    data_panel = main_window._right_panel._data_manager
    data_panel._datasets.clear()
    data_panel._datasets["ds1"] = dataset

    main_window._mechanism_editor._reactions_text.setPlainText(
        "\n".join(
            [
                "reaction: A -> B; k=0.2",
                "initial: A=1.0",
                "initial: B=0.0",
            ]
        )
    )

    # Avoid relying on slider variable metadata during this unit-style regression.
    monkeypatch.setattr(main_window, "_apply_parameter_overrides_to_dsl", lambda mech, _params: mech)

    # Avoid any setup-time parsing that is unrelated to evaluation.
    monkeypatch.setattr(
        main_window._dataset_manager,
        "scan_mechanism_parameters",
        lambda _dsl: [{"name": "k1", "value": 0.2, "min": 0.01, "max": 1.0}],
    )
    monkeypatch.setattr(
        type(main_window),
        "_extract_mechanism_initials",
        lambda _self, _dsl: {"A": 1.0, "B": 0.0},
        raising=False,
    )

    from PySide6 import QtWidgets

    captured = {}

    class _FakeWindow(QtWidgets.QDialog):
        def __init__(self, *args, **kwargs):
            super().__init__()
            captured["kwargs"] = kwargs

        def setWindowTitle(self, *_):
            pass

        def show(self):
            pass

        def raise_(self):
            pass

        def activateWindow(self):
            pass

    monkeypatch.setattr("kindred.gui.fitting.window.FittingWindow", _FakeWindow)

    # Count session-level fitting execution-context construction at the launch boundary.
    import kindred.gui.fitting.launch as fitting_launch
    from kindred.core.fitting_evaluation import SerialFittingEvaluator

    counts = {"build_context": 0}
    _orig_prepare_fitting_execution_context = fitting_launch.prepare_fitting_execution_context

    def _counting_prepare_fitting_execution_context(*args, **kwargs):
        counts["build_context"] += 1
        return _orig_prepare_fitting_execution_context(*args, **kwargs)

    monkeypatch.setattr(
        fitting_launch,
        "prepare_fitting_execution_context",
        _counting_prepare_fitting_execution_context,
    )

    # Stub solve_ode to keep this test fast and deterministic; parsing still happens.
    from kindred.core.simulator.solvers import SimulationOutput

    def _fake_solve(request):
        n = int(np.asarray(request.y0).size)
        grid_n = int((request.grid or {}).get("N") or 5)
        t0, t1 = request.t_span
        tt = np.linspace(float(t0), float(t1), grid_n, dtype=float)
        yy = np.tile(np.asarray(request.y0, dtype=float).reshape(n, 1), (1, tt.size))
        return SimulationOutput(t=tt, Y=yy, provenance={})

    monkeypatch.setattr("kindred.core.simulator.solvers.solve_ode", _fake_solve)

    main_window._run_global_fit()
    sim_func = captured["kwargs"]["simulation_func"]
    assert isinstance(sim_func, SerialFittingEvaluator)
    assert type(sim_func) is SerialFittingEvaluator
    assert counts["build_context"] == 1

    for k in (0.05, 0.1, 0.2, 0.4, 0.8):
        _ = sim_func({"k1": float(k), "init:A": 1.0})

    assert counts["build_context"] == 1


@pytest.mark.unit
def test_prepared_global_fit_simulation_matches_parse_each_call(monkeypatch):
    """
    Numerical equivalence check: prepared (parse-once) simulation should match the
    legacy "parse each call" approach for simple mechanisms.
    """
    from kindred.core.simulation_preparation import build_prepared_simulation_func
    from kindred.core.ode_builder import build_ode_rhs_from_mechanism
    from kindred.core.simulator.dsl import parse_dsl_to_mechanism
    from kindred.core.simulator.solvers import SimulationRequest, solve_ode
    from kindred.core.units import UnitsModel

    def _simulate_with_reparse(k_value: float):
        dsl = "\n".join(
            [
                f"reaction: A -> B; k={float(k_value)}",
                "initial: A=1.0",
                "initial: B=0.0",
            ]
        )
        mech = parse_dsl_to_mechanism(dsl, initials={}, units=UnitsModel(temperature_K=298.15, energy_unit="kJ/mol"))
        rhs = build_ode_rhs_from_mechanism(mech)
        names = mech.species_names()
        y0 = np.array([mech.species[n].initial_conc for n in names], dtype=float)
        req = SimulationRequest(
            rhs=rhs,
            t_span=(0.0, 2.0),
            y0=y0,
            solver="Radau",
            rtol=1e-8,
            atol=1e-12,
            grid={"N": 60},
            temperature_schedule=getattr(mech, "metadata", {}).get("temperature_schedule"),
        )
        out = solve_ode(req)
        return out.t, out.Y[names.index("B"), :]

    base_mechanism = "\n".join(
        [
            "reaction: A -> B; k=0.2",
            "initial: A=1.0",
            "initial: B=0.0",
        ]
    )
    prepared = build_prepared_simulation_func(
        mechanism_text=base_mechanism,
        param_names=["k1"],
        t_end=2.0,
        num_points=60,
        temperature_K=298.15,
        solver="Radau",
        rtol=1e-8,
        atol=1e-12,
        use_sparse_jacobian=False,
        initial_prefix="init:",
    )

    for k in (0.05, 0.2, 0.75):
        t_old, b_old = _simulate_with_reparse(k)
        sim = prepared({"k1": float(k), "init:A": 1.0})
        t_new = np.asarray(sim["t"], dtype=float)
        b_new = np.asarray(sim["species"]["B"], dtype=float)
        assert np.allclose(t_new, t_old, rtol=0.0, atol=0.0)
        assert np.allclose(b_new, b_old, rtol=1e-6, atol=1e-9)
