import numpy as np

from kindred.core.ode_builder import build_ode_rhs_from_mechanism
from kindred.core.simulator.dsl import parse_dsl_to_mechanism
from kindred.core.simulator.solvers import SimulationRequest, solve_ode
from kindred.core.temperature import TemperatureSchedule


ARRHENIUS_DSL = "\n".join(
    [
        "energy=kJ/mol",
        "reaction: A -> B; A=1e3; Ea=50",
        "[A]=1.0",
        "[B]=0.0",
    ]
)


def _run_arrhenius(schedule: TemperatureSchedule):
    mech = parse_dsl_to_mechanism(ARRHENIUS_DSL, initials={})
    rhs = build_ode_rhs_from_mechanism(mech)
    species_names = mech.species_names()
    y0 = np.array([mech.species[sp].initial_conc for sp in species_names])
    req = SimulationRequest(
        rhs=rhs,
        t_span=(0.0, 10.0),
        y0=y0,
        solver="Radau",
        grid={"N": 200},
        temperature_schedule=schedule,
    )
    return solve_ode(req), species_names


def test_arrhenius_rates_follow_temperature():
    low_T = TemperatureSchedule.constant(300.0)
    high_T = TemperatureSchedule.constant(600.0)

    low_result, names = _run_arrhenius(low_T)
    high_result, _ = _run_arrhenius(high_T)

    b_idx = names.index("B")
    b_low = low_result.Y[b_idx, -1]
    b_high = high_result.Y[b_idx, -1]

    assert b_high > b_low * 50.0
    assert b_high > 0.05


def test_temperature_schedule_step_changes_rate():
    low_T = TemperatureSchedule.constant(300.0)
    high_T = TemperatureSchedule.constant(600.0)
    step_schedule = TemperatureSchedule.piecewise(
        [(0.0, 5.0, 300.0), (5.0, 10.0, 600.0)]
    )

    low_result, names = _run_arrhenius(low_T)
    step_result, _ = _run_arrhenius(step_schedule)
    high_result, _ = _run_arrhenius(high_T)

    b_idx = names.index("B")
    b_series = step_result.Y[b_idx, :]
    t_series = step_result.t
    mid_idx = int(np.searchsorted(t_series, 5.0))

    pre_delta = b_series[mid_idx] - b_series[0]
    post_delta = b_series[-1] - b_series[mid_idx]

    b_low_final = low_result.Y[b_idx, -1]
    b_high_final = high_result.Y[b_idx, -1]

    assert post_delta > pre_delta
    assert b_series[-1] > b_low_final * 5.0
    assert b_series[-1] < b_high_final


def test_temperature_response_schedule_changes_rate_inside_interval():
    low_T = TemperatureSchedule.constant(300.0)
    step_schedule = TemperatureSchedule.piecewise(
        [(0.0, 5.0, 300.0), (5.0, 10.0, 600.0)]
    )
    response_schedule = TemperatureSchedule.response(
        [0.0, 5.0, 10.0], [300.0, 600.0], tau=2.0
    )

    low_result, names = _run_arrhenius(low_T)
    response_result, _ = _run_arrhenius(response_schedule)
    step_result, _ = _run_arrhenius(step_schedule)

    b_idx = names.index("B")
    response_series = response_result.Y[b_idx, :]
    t_series = response_result.t
    post_idx = int(np.searchsorted(t_series, 6.0))

    b_low_final = low_result.Y[b_idx, -1]
    b_step_final = step_result.Y[b_idx, -1]
    b_response_final = response_series[-1]

    assert response_schedule(6.0) > 300.0
    assert response_schedule(6.0) < 600.0
    assert response_series[post_idx] > low_result.Y[b_idx, post_idx]
    assert b_low_final < b_response_final < b_step_final
