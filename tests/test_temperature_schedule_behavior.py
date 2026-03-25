import numpy as np

from kindred.core.ode_builder import build_ode_rhs_from_mechanism
from kindred.core.simulator.dsl import parse_dsl_to_mechanism
from kindred.core.simulator.solvers import SimulationRequest, solve_ode
from kindred.core.temperature import TemperatureSchedule


ARRHENIUS_DSL = "\n".join(
    [
        "energy=kJ/mol",
        "reaction: A -> B; A=5e5; Ea=60",
        "initial: A=1.0",
        "initial: B=0.0",
    ]
)


def _run(schedule: TemperatureSchedule):
    mechanism = parse_dsl_to_mechanism(ARRHENIUS_DSL, initials={})
    rhs = build_ode_rhs_from_mechanism(mechanism)
    species_names = mechanism.species_names()
    y0 = np.array([mechanism.species[sp].initial_conc for sp in species_names])
    req = SimulationRequest(
        rhs=rhs,
        t_span=(0.0, 12.0),
        y0=y0,
        solver="Radau",
        grid={"N": 200},
        temperature_schedule=schedule,
    )
    result = solve_ode(req)
    a_idx = species_names.index("A")
    return float(result.Y[a_idx, -1])


def test_temperature_schedule_changes_rate_monotonically():
    low_T = TemperatureSchedule.constant(290.0)
    high_T = TemperatureSchedule.constant(520.0)
    step_schedule = TemperatureSchedule.piecewise(
        [(0.0, 4.0, 290.0), (4.0, 8.0, 520.0)]
    )

    a_low = _run(low_T)
    a_high = _run(high_T)
    a_step = _run(step_schedule)

    assert a_high < a_low * 0.1  # Much faster decay at high temperature
    assert a_low > a_high
    assert a_step < a_low * 0.1  # Step schedule accelerates after switch
    assert a_step > a_high  # Intermediate between low and high extremes
