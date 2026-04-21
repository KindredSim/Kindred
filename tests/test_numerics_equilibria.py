import numpy as np

from kindred.core.ode_builder import build_ode_rhs_from_mechanism
from kindred.core.simulator.dsl import parse_dsl_to_mechanism
from kindred.core.simulator.solvers import SimulationRequest, solve_ode
import pytest

pytestmark = pytest.mark.integration



def test_reversible_equilibrium_reaches_expected_state():
    kf = 1.2
    kr = 0.3
    a0 = 1.0
    b0 = 0.0
    total = a0 + b0
    eq_ratio = kf / kr
    a_eq = total / (1.0 + eq_ratio)
    b_eq = total - a_eq

    dsl = "\n".join(
        [
            f"equilibrium: A <-> B; kf={kf}; kr={kr}",
            f"initial: A={a0}",
            f"initial: B={b0}",
        ]
    )

    mechanism = parse_dsl_to_mechanism(dsl, initials={})
    rhs = build_ode_rhs_from_mechanism(mechanism)
    species_names = mechanism.species_names()
    y0 = np.array([mechanism.species[name].initial_conc for name in species_names])

    req = SimulationRequest(
        rhs=rhs,
        t_span=(0.0, 25.0),
        y0=y0,
        solver="Radau",
        rtol=1e-9,
        atol=1e-12,
        grid={"N": 400},
    )
    result = solve_ode(req)

    a_idx = species_names.index("A")
    b_idx = species_names.index("B")
    sim_a = result.Y[a_idx, :]
    sim_b = result.Y[b_idx, :]

    assert np.all(sim_b <= total + 1e-9)  # No overshoot above conserved mass
    assert np.isclose(sim_a[-1] + sim_b[-1], total, atol=1e-8)
    assert abs(sim_a[-1] - a_eq) < 5e-6
    assert abs(sim_b[-1] - b_eq) < 5e-6
