import numpy as np

from kindred.core.ode_builder import build_ode_rhs_from_mechanism
from kindred.core.simulator.dsl import parse_dsl_to_mechanism
from kindred.core.simulator.solvers import SimulationRequest, solve_ode


def test_irreversible_first_order_matches_analytic():
    k = 0.4
    a0 = 1.25
    b0 = 0.0
    dsl = "\n".join(
        [
            f"reaction: A -> B; k={k}",
            f"initial: A={a0}",
            f"initial: B={b0}",
        ]
    )

    mechanism = parse_dsl_to_mechanism(dsl, initials={})
    rhs = build_ode_rhs_from_mechanism(mechanism)
    species_names = mechanism.species_names()
    y0 = np.array([mechanism.species[name].initial_conc for name in species_names])

    t_final = 8.0
    req = SimulationRequest(
        rhs=rhs,
        t_span=(0.0, t_final),
        y0=y0,
        solver="Radau",
        rtol=1e-9,
        atol=1e-12,
        grid={"N": 201},
    )
    result = solve_ode(req)

    t = result.t
    Y = result.Y
    a_idx = species_names.index("A")
    b_idx = species_names.index("B")
    sim_a = Y[a_idx, :]
    sim_b = Y[b_idx, :]

    analytic_a = a0 * np.exp(-k * t)
    analytic_b = b0 + a0 * (1.0 - np.exp(-k * t))

    max_err_a = np.max(np.abs(sim_a - analytic_a))
    max_err_b = np.max(np.abs(sim_b - analytic_b))

    assert max_err_a < 1e-6
    assert max_err_b < 1e-6
