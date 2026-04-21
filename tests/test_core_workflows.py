"""
Targeted regression tests for low-level simulation and fitting workflows.

These tests focus on adversarial scenarios that were not covered by the
integration suite—minimal solver grids and mismatched fitting targets.
"""

import numpy as np
import pytest

from kindred.core.exceptions import FitSimulationError
from kindred.core.fitting_objective import build_fitting_objective
from kindred.core.fitting_optimization import fit_parameters
from kindred.core.simulator.dsl import parse_dsl_to_mechanism
from kindred.core.ode_builder import build_ode_rhs_from_mechanism
from kindred.core.simulator.solvers import SimulationRequest, solve_ode

pytestmark = pytest.mark.integration



def test_simulation_pipeline_handles_minimal_grid():
    """A simple mechanism should integrate correctly even with the smallest grid."""
    dsl = "\n".join(
        [
            "reaction: A -> B; k=0.5",
            "initial: A=1.0",
            "initial: B=0.0",
        ]
    )
    mechanism = parse_dsl_to_mechanism(dsl, initials={})
    y0 = np.array([mechanism.species["A"].initial_conc, mechanism.species["B"].initial_conc])
    rhs = build_ode_rhs_from_mechanism(mechanism)

    request = SimulationRequest(
        rhs=rhs,
        t_span=(0.0, 2.0),
        y0=y0,
        solver="BDF",
        rtol=1e-7,
        atol=1e-12,
        grid={"N": 4},
    )
    result = solve_ode(request)
    assert result.t.size == 4
    assert np.all(np.diff(result.t) > 0)
    # Concentration of A should monotonically decrease for a first-order decay.
    assert np.all(np.diff(result.Y[0]) <= 1e-12)


def test_reversible_equilibrium_converges_to_expected_ratio():
    """A reversible equilibrium should settle to the kf/kr ratio."""
    dsl = "\n".join(
        [
            "equilibrium: A <-> B; kf=2.0; kr=1.0",
            "initial: A=1.0",
            "initial: B=0.0",
        ]
    )
    mechanism = parse_dsl_to_mechanism(dsl, initials={})
    species_names = list(mechanism.species.keys())
    y0 = np.array([mechanism.species[name].initial_conc for name in species_names])
    rhs = build_ode_rhs_from_mechanism(mechanism)

    request = SimulationRequest(
        rhs=rhs,
        t_span=(0.0, 10.0),
        y0=y0,
        solver="BDF",
        rtol=1e-8,
        atol=1e-12,
        grid={"N": 50},
    )
    result = solve_ode(request)
    final_A = result.Y[species_names.index("A"), -1]
    final_B = result.Y[species_names.index("B"), -1]
    total = final_A + final_B

    # Mass balance must hold and the equilibrium ratio should match kf/kr.
    assert pytest.approx(total, rel=1e-6) == 1.0
    assert pytest.approx(final_B / final_A, rel=1e-2) == 2.0


def test_build_fitting_objective_raises_on_missing_target():
    """Objective should raise a clear error if the requested species is absent."""
    mechanism = "\n".join(
        [
            "reaction: A -> B; k=0.2",
            "initial: A=1.0",
            "initial: B=0.0",
        ]
    )
    t_exp = np.linspace(0.0, 1.0, 5)
    y_exp = np.zeros_like(t_exp)
    objective = build_fitting_objective(
        mechanism_text=mechanism,
        param_names=["k1"],
        t_exp=t_exp,
        y_exp=y_exp,
        target_species="C",  # Species absent from mechanism
    )

    with pytest.raises(FitSimulationError) as excinfo:
        objective(np.array([0.2]))
    assert "Target species 'C' not found" in str(excinfo.value)


def test_precompiled_objective_converges_on_simple_fit():
    """Fitting should converge to the known rate without re-parsing the DSL."""
    k_true = 0.4
    mechanism = "\n".join(
        [
            "reaction: A -> B; k=0.4",
            "initial: A=1.0",
            "initial: B=0.0",
        ]
    )
    t_exp = np.linspace(0.0, 5.0, 25)
    y_exp = 1.0 - np.exp(-k_true * t_exp)  # [B](t) for A -> B

    objective = build_fitting_objective(
        mechanism_text=mechanism,
        param_names=["k1"],
        t_exp=t_exp,
        y_exp=y_exp,
        target_species="B",
        solver="BDF",
        rtol=1e-7,
        atol=1e-12,
    )

    result = fit_parameters(
        objective,
        initial_params={"k1": 0.1},
        method="trf",
        max_nfev=200,
    )

    assert result.success
    assert pytest.approx(result.parameters["k1"], rel=1e-2) == k_true
    assert np.linalg.norm(result.residuals) < 1e-2


def test_build_fitting_objective_parses_once(monkeypatch):
    """Objective construction should parse the mechanism once, not per call."""
    parse_calls = {"count": 0}
    real_parse = parse_dsl_to_mechanism

    def _wrapped_parse(text, *args, **kwargs):
        parse_calls["count"] += 1
        return real_parse(text, *args, **kwargs)

    monkeypatch.setattr(
        "kindred.core.simulator.dsl.parse_dsl_to_mechanism",
        _wrapped_parse,
    )

    mechanism = "\n".join(
        [
            "reaction: A -> B; k=0.3",
            "initial: A=1.0",
            "initial: B=0.0",
        ]
    )
    t_exp = np.linspace(0.0, 1.0, 4)
    y_exp = 1.0 - np.exp(-0.3 * t_exp)

    objective = build_fitting_objective(
        mechanism_text=mechanism,
        param_names=["k1"],
        t_exp=t_exp,
        y_exp=y_exp,
        target_species="B",
    )

    objective(np.array([0.25]))
    objective(np.array([0.35]))

    assert parse_calls["count"] == 1
