from __future__ import annotations

import numpy as np
import pytest

from kindred.core.ode_builder import build_ode_rhs_from_mechanism
from kindred.core.simulator.computational_mode import (
    COMP_BLOCK_END,
    COMP_BLOCK_START,
    GENERATED_BLOCK_END,
    GENERATED_BLOCK_START,
    compile_comp_spec,
    parse_comp_block,
)
from kindred.core.simulator.dsl import parse_dsl_to_mechanism
from kindred.core.simulator.solvers import SimulationRequest, solve_ode


pytestmark = [pytest.mark.unit]


def test_computational_mode_fast_equilibrium_is_reversible_in_simulation():
    """
    Regression: Computational Mode emits fast equilibria as explicit `equilibrium:` lines.
    These must contribute both forward and reverse fluxes in the ODE RHS, even when
    state-network equilibria are also present.
    """
    comp_body = "\n".join(
        [
            "comp: T = 298.15 K",
            "comp: pressure = 1 atm",
            "comp: energy_unit = hartree",
            "comp: std_default = 1 M",
            "comp: kfast_default = 1e9",
            "comp: species A type=GS G=-500 degeneracy=1",
            "comp: species B type=TS G=-499.96 degeneracy=1",
            "comp: species C type=GS G=-500.01 degeneracy=1",
            "comp: species D type=GS G=-499.99 degeneracy=1",
            "comp: channel A <-> C via B",
            "comp: rxn A <-> D",
        ]
    )

    spec = parse_comp_block(comp_body)
    compiled = compile_comp_spec(spec)
    full_dsl = (
        f"{COMP_BLOCK_START}\n"
        f"{comp_body}\n"
        f"{COMP_BLOCK_END}\n\n"
        f"{GENERATED_BLOCK_START}\n"
        f"{compiled.generated_reaction_dsl}\n"
        f"{GENERATED_BLOCK_END}\n"
    )

    mechanism = parse_dsl_to_mechanism(
        full_dsl,
        initials={"A": 1.0, "C": 0.0, "D": 0.0},
    )
    species_names = mechanism.species_names()

    fast_eqs = [
        eq for eq in mechanism.equilibria if bool((getattr(eq, "metadata", {}) or {}).get("fast_equilibrium"))
    ]
    assert len(fast_eqs) == 1
    fast_eq = fast_eqs[0]
    expected_ratio = float(fast_eq.kf) / float(fast_eq.kr)

    rhs = build_ode_rhs_from_mechanism(mechanism)

    # Deterministic check: if D is above equilibrium, reverse flux must decrease D.
    y_test = np.zeros(len(species_names), dtype=float)
    y_test[species_names.index("A")] = 0.5
    y_test[species_names.index("D")] = 0.5
    dy = rhs(0.0, y_test)
    assert float(dy[species_names.index("D")]) < 0.0

    y0 = np.array([mechanism.species[nm].initial_conc for nm in species_names], dtype=float)
    # The production fast-equilibrium default is intentionally large. Integrate
    # across enough fast time constants to prove the simulated reversible ratio
    # without forcing implicit solvers through a long post-equilibrium stiff tail.
    req = SimulationRequest(
        rhs=rhs,
        t_span=(0.0, 1e-6),
        y0=y0,
        solver="BDF",
        rtol=1e-6,
        atol=1e-12,
        grid={"N": 50},
    )
    result = solve_ode(req)

    A_final = float(result.Y[species_names.index("A"), -1])
    D_final = float(result.Y[species_names.index("D"), -1])
    ratio_final = D_final / A_final if A_final else float("inf")

    assert ratio_final == pytest.approx(expected_ratio, rel=0.05, abs=0.0)
