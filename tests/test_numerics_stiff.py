import numpy as np
import pytest

from kindred.core.exceptions import SolverError
from kindred.core.ode_builder import build_ode_rhs_from_mechanism
from kindred.core.simulator import solvers
from kindred.core.simulator.dsl import parse_dsl_to_mechanism
from kindred.core.simulator.solvers import SimulationRequest, solve_ode

pytestmark = pytest.mark.integration



def _robertson_rhs():
    dsl = "\n".join(
        [
            "reaction: A -> B; k=0.04",
            "reaction: B + C -> A + C; k=1.0e4",
            "reaction: B + B -> B + C; k=3.0e7",
            "initial: A=1.0",
            "initial: B=0.0",
            "initial: C=0.0",
        ]
    )
    mechanism = parse_dsl_to_mechanism(dsl, initials={})
    rhs = build_ode_rhs_from_mechanism(mechanism)
    species_names = mechanism.species_names()
    y0 = np.array([mechanism.species[name].initial_conc for name in species_names], dtype=float)
    return rhs, species_names, y0


def test_robertson_stiff_system_uses_implicit_solver():
    rhs, species_names, y0 = _robertson_rhs()

    req = SimulationRequest(
        rhs=rhs,
        t_span=(0.0, 1e3),
        y0=y0,
        solver="BDF",
        rtol=1e-6,
        atol=1e-12,
        grid={"N": 400},
    )
    result = solve_ode(req)

    a_idx = species_names.index("A")
    b_idx = species_names.index("B")
    c_idx = species_names.index("C")
    total_mass = result.Y[a_idx, :] + result.Y[b_idx, :] + result.Y[c_idx, :]

    assert not result.fallback_occurred
    assert result.provenance.get("solver_requested") == "BDF"
    assert result.provenance.get("solver_used") in ("BDF", "Radau")
    assert result.provenance.get("solver_used") != "RK4_fixed"
    assert "solver" not in result.provenance
    assert np.all(np.isfinite(result.Y))
    assert np.all(total_mass <= 1.0 + 1e-6)
    assert np.all(total_mass >= -1e-9)
    assert result.Y[c_idx, -1] == pytest.approx(0.663123, rel=1e-5)
    assert result.Y[b_idx, -1] < 1e-4
    assert result.Y[a_idx, -1] == pytest.approx(0.336875, rel=1e-5)


def test_stiff_solver_failure_raises_without_rk4(monkeypatch):
    rhs, _species_names, y0 = _robertson_rhs()
    called_methods: list[str] = []

    def _always_fail(*args, **kwargs):
        method = kwargs.get("method")
        t_span = kwargs.get("t_span", args[1] if len(args) > 1 else (0.0, 0.0))
        called_methods.append(method)

        class DummySolution:
            success = False
            message = f"{method} failed to converge"
            t = np.array([float(t_span[0]) + 1e-6])

        return DummySolution()

    monkeypatch.setattr(solvers, "_solve_ivp", _always_fail)

    req = SimulationRequest(
        rhs=rhs,
        t_span=(0.0, 1.0),
        y0=y0,
        solver="BDF",
        rtol=1e-6,
        atol=1e-12,
        grid={"N": 5},
    )

    with pytest.raises(SolverError) as excinfo:
        solve_ode(req)

    assert "attempted methods" in str(excinfo.value)
    assert called_methods[:2] == ["BDF", "Radau"]
    assert all(method != "RK4_fixed" for method in called_methods)
