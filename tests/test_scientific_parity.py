import re

import numpy as np

from kindred.gui.controllers.dataset_manager import DatasetManager


def test_analysis_simulation_honors_temperature_schedule_in_dsl(main_window):
    """
    Regression: the synchronous simulation helper used by fitting/global-fit must
    honor temperature schedules encoded in the DSL.
    """
    dsl_with_schedule = "\n".join(
        [
            "energy=kJ/mol",
            "temp_step: t=[0,5,10], T=[300,600]",
            "reaction: A -> B; A=1e3; Ea=50",
            "initial: A=1.0",
            "initial: B=0.0",
        ]
    )
    dsl_isothermal = "\n".join(
        [
            "energy=kJ/mol",
            "reaction: A -> B; A=1e3; Ea=50",
            "initial: A=1.0",
            "initial: B=0.0",
        ]
    )

    main_window._initial_solver = "Radau"
    main_window._temperature_spinbox.setValue(300.0)

    scheduled = main_window._simulate_mechanism(dsl_with_schedule, t_end=10.0, num_points=200)
    isothermal = main_window._simulate_mechanism(dsl_isothermal, t_end=10.0, num_points=200)

    t_series = np.asarray(scheduled["t"], dtype=float).reshape(-1)
    b_series = np.asarray(scheduled["species"]["B"], dtype=float).reshape(-1)
    b_sched = float(b_series[-1])
    b_iso = float(np.asarray(isothermal["species"]["B"], dtype=float).reshape(-1)[-1])

    mid_idx = int(np.searchsorted(t_series, 5.0))
    pre_delta = float(b_series[mid_idx] - b_series[0])
    post_delta = float(b_series[-1] - b_series[mid_idx])

    assert post_delta > pre_delta
    assert b_sched > b_iso * 5.0


def test_analysis_simulation_honors_temp_response_schedule_in_dsl(main_window):
    dsl_with_response = "\n".join(
        [
            "energy=kJ/mol",
            "temp_response: t=[0,5,10], T=[300,600], tau=2.0",
            "reaction: A -> B; A=1e3; Ea=50",
            "initial: A=1.0",
            "initial: B=0.0",
        ]
    )
    dsl_with_step = "\n".join(
        [
            "energy=kJ/mol",
            "temp_step: t=[0,5,10], T=[300,600]",
            "reaction: A -> B; A=1e3; Ea=50",
            "initial: A=1.0",
            "initial: B=0.0",
        ]
    )

    main_window._initial_solver = "Radau"
    main_window._temperature_spinbox.setValue(300.0)

    response = main_window._simulate_mechanism(dsl_with_response, t_end=10.0, num_points=200)
    step = main_window._simulate_mechanism(dsl_with_step, t_end=10.0, num_points=200)

    b_response = float(np.asarray(response["species"]["B"], dtype=float).reshape(-1)[-1])
    b_step = float(np.asarray(step["species"]["B"], dtype=float).reshape(-1)[-1])

    assert b_response > 0.0
    assert b_response < b_step


def test_prepare_fit_job_uses_solver_settings_getter(monkeypatch):
    """
    Regression: solver/rtol/atol must flow into the objective builder for
    dataset_manager-driven fit entry points (e.g., local objective jobs).
    """
    captured = {}

    def _fake_build_fitting_objective(**kwargs):
        captured["solver"] = kwargs["solver"]
        captured["rtol"] = float(kwargs["rtol"])
        captured["atol"] = float(kwargs["atol"])
        return lambda _params: np.zeros(5, dtype=float)

    monkeypatch.setattr("kindred.core.fitting_objective.build_fitting_objective", _fake_build_fitting_objective)

    dataset = {
        "t": np.linspace(0.0, 1.0, 5),
        "species": {"A": np.ones(5, dtype=float)},
    }

    dm = DatasetManager(
        plot_tabs=None,
        dataset_resolver=lambda name: dataset if name == "ds1" else None,
        solver_settings_getter=lambda: {"solver": "Radau", "rtol": 1e-3, "atol": 1e-9},
    )

    dm.prepare_fit_job(
        config={"dataset": "ds1", "parameters": {"k1": 0.1}},
        mechanism_text="reaction: A -> B; k=0.1\ninitial: A=1.0\ninitial: B=0.0",
        state_network_text="",
        temperature_K=298.15,
    )

    assert captured == {"solver": "Radau", "rtol": 1e-3, "atol": 1e-9}


def test_prepare_fit_job_does_not_double_append_state_network(monkeypatch):
    """
    Regression: some workflows may pass a full DSL that already includes the
    '# State Network' section while also supplying state_network_text. Ensure we
    do not append the state network twice.
    """
    captured = {}

    def _fake_build_fitting_objective(**kwargs):
        captured["mechanism_text"] = kwargs["mechanism_text"]
        return lambda _params: np.zeros(5, dtype=float)

    monkeypatch.setattr("kindred.core.fitting_objective.build_fitting_objective", _fake_build_fitting_objective)

    dataset = {
        "t": np.linspace(0.0, 1.0, 5),
        "species": {"A": np.ones(5, dtype=float)},
    }

    state_network_text = "States: A,B\nedge: A->B"
    mechanism_text_with_network = "\n".join(
        [
            "reaction: A -> B; k=0.1",
            "initial: A=1.0",
            "initial: B=0.0",
            "",
            "# State Network",
            state_network_text,
        ]
    )

    dm = DatasetManager(
        plot_tabs=None,
        dataset_resolver=lambda name: dataset if name == "ds1" else None,
        solver_settings_getter=lambda: {"solver": "Radau", "rtol": 1e-6, "atol": 1e-12},
    )

    dm.prepare_fit_job(
        config={"dataset": "ds1", "parameters": {"k1": 0.1}},
        mechanism_text=mechanism_text_with_network,
        state_network_text=state_network_text,
        temperature_K=298.15,
    )

    text = str(captured["mechanism_text"])
    headers = re.findall(r"(?im)^\s*#\s*state\s+network\b", text)
    assert len(headers) == 1
