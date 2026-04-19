import numpy as np
import pytest

from kindred.core.exceptions import SolverError
from kindred.core.simulator import solvers

pytestmark = [pytest.mark.unit]


def test_solver_failure_raises_solvererror(monkeypatch):
    """Solver should raise SolverError after exhausting implicit alternatives."""

    class DummySolution:
        def __init__(self):
            self.success = False
            self.message = "forced failure"
            self.t = np.array([0.0, 0.5])

    call_count = {"n": 0}

    def _fail(*args, **kwargs):
        call_count["n"] += 1
        return DummySolution()

    monkeypatch.setattr(solvers, "_solve_ivp", _fail)

    req = solvers.SimulationRequest(
        rhs=lambda t, y: -y,
        t_span=(0.0, 1.0),
        y0=np.array([1.0]),
        solver="BDF",
        grid={"N": 4},
    )

    with pytest.raises(SolverError) as exc_info:
        solvers.solve_ode(req)

    # Should attempt primary + alternatives and include context in the error
    assert call_count["n"] == 2  # BDF + Radau
    message = str(exc_info.value)
    assert "Solver 'BDF'" in message
    assert "attempted methods: BDF, Radau" in message
    assert "forced failure" in message


def test_solver_records_successful_fallback(monkeypatch):
    """When the primary SciPy method fails but a fallback succeeds, record provenance."""

    class DummySolution:
        def __init__(self, success: bool, message: str, t_eval):
            self.success = success
            self.message = message
            self.t = np.asarray(t_eval, float)
            self.y = np.vstack([np.ones_like(self.t)])

    calls = []

    def _fake_solve_ivp(*args, **kwargs):
        method = kwargs.get("method")
        calls.append(method)
        t_eval = kwargs.get("t_eval")
        if t_eval is None:
            t_eval = np.array([0.0, 1.0])
        if method == "BDF":
            return DummySolution(False, "BDF failure", t_eval[:1])
        return DummySolution(True, "ok", t_eval)

    monkeypatch.setattr(solvers, "_solve_ivp", _fake_solve_ivp)

    req = solvers.SimulationRequest(
        rhs=lambda t, y: -y,
        t_span=(0.0, 1.0),
        y0=np.array([1.0]),
        solver="BDF",
        grid={"N": 2},
    )

    out = solvers.solve_ode(req)

    assert calls == ["BDF", "Radau"]
    assert out.fallback_occurred is True
    assert out.fallback_message == "BDF failed; succeeded with Radau"
    assert out.provenance.get("solver_requested") == "BDF"
    assert out.provenance.get("solver_used") == "Radau"
    assert "solver" not in out.provenance
