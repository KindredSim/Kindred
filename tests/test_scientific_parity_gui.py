import numpy as np
import pytest

pytestmark = pytest.mark.gui


@pytest.mark.integration
def test_analysis_simulation_honors_temperature_schedule_in_dsl(main_window):
    """Regression: fitting/global-fit simulation helper must honor temperature schedules."""
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


def test_analysis_simulation_disables_symbolic_jacobian_for_temperature_schedule(
    main_window,
    monkeypatch,
):
    from kindred.core.simulator.solvers import SimulationOutput

    captured: dict[str, object] = {}

    def _fake_solve_ode(request):
        captured["temperature_schedule"] = request.temperature_schedule
        captured["jacobian_func"] = request.jacobian_func
        return SimulationOutput(
            t=np.asarray([0.0, 1.0], dtype=float),
            Y=np.asarray([[1.0, 1.0], [0.0, 0.0]], dtype=float),
            provenance={},
        )

    monkeypatch.setattr("kindred.core.simulator.solvers.solve_ode", _fake_solve_ode)
    main_window._initial_solver = "BDF"
    main_window._use_sparse_jacobian = True

    main_window._simulate_mechanism(
        "\n".join(
                [
                "temp_step: t=[0,0.5,1], T=[300,310]",
                "reaction: A -> B; k=1",
                "initial: A=1.0",
                "initial: B=0.0",
            ]
        ),
        t_end=1.0,
        num_points=3,
    )

    assert captured["temperature_schedule"] is not None
    assert captured["jacobian_func"] is None


@pytest.mark.integration
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


@pytest.mark.integration
def test_analysis_simulation_honors_intervention_schedule_in_dsl(main_window):
    dsl_with_intervention = "\n".join(
        [
            "reaction: A -> B; k=0",
            "initial: A=0.0",
            "initial: B=0.0",
            "intervention: op=set; species=A; time=0.0; value=3.0",
        ]
    )

    result = main_window._simulate_mechanism(dsl_with_intervention, t_end=2.0, num_points=5)

    a_series = np.asarray(result["species"]["A"], dtype=float).reshape(-1)
    b_series = np.asarray(result["species"]["B"], dtype=float).reshape(-1)
    assert float(a_series[0]) == pytest.approx(3.0)
    assert float(a_series[-1]) == pytest.approx(3.0)
    assert float(b_series[-1]) == pytest.approx(0.0)
