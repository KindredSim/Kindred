import numpy as np
import pytest

from kindred.core.exceptions import FitSimulationError
from kindred.core.fitting_objective import build_fitting_objective
from kindred.core.fitting_optimization import fit_parameters
from kindred.core.simulator.solvers import SimulationOutput


MECHANISM = "\n".join(
    [
        "reaction: A -> B; k=0.5",
        "initial: A=1.0",
        "initial: B=0.0",
    ]
)


def _build_simple_objective():
    t_exp = np.linspace(0.0, 1.0, 5)
    y_exp = np.zeros_like(t_exp)
    return build_fitting_objective(
        mechanism_text=MECHANISM,
        param_names=["k1"],
        t_exp=t_exp,
        y_exp=y_exp,
        target_species="B",
        solver="LSODA",
    )


def test_objective_raises_fit_simulation_error(monkeypatch):
    def _boom(*_args, **_kwargs):
        raise RuntimeError("integration blew up")

    monkeypatch.setattr("kindred.core.simulator.solvers.solve_ode", _boom)
    objective = _build_simple_objective()

    with pytest.raises(FitSimulationError) as excinfo:
        objective(np.array([0.25]))

    msg = str(excinfo.value).lower()
    assert "integration blew up" in msg
    assert "simulation failed" in msg


def test_fit_parameters_returns_failure_without_penalty(monkeypatch):
    call_counter = {"n": 0}

    def _boom(*_args, **_kwargs):
        call_counter["n"] += 1
        raise RuntimeError("solver diverged")

    monkeypatch.setattr("kindred.core.simulator.solvers.solve_ode", _boom)
    objective = _build_simple_objective()

    result = fit_parameters(
        objective,
        initial_params={"k1": 0.1},
        method="trf",
        max_nfev=5,
    )

    assert call_counter["n"] >= 1
    assert not result.success
    assert result.residuals.size == 0  # No flat 1e6 residual penalty vector
    assert np.isinf(result.chi_squared)
    assert "solver diverged" in result.message.lower()
    assert abs(result.parameters["k1"] - 0.1) < 1e-9


def test_objective_penalizes_nonfinite_simulation(monkeypatch):
    t_exp = np.linspace(0.0, 1.0, 5)
    y_exp = np.zeros_like(t_exp)

    def _nan_solver(request):
        Y = np.vstack(
            [
                np.zeros_like(request.t_eval),
                np.full_like(request.t_eval, np.nan),
            ]
        )
        return SimulationOutput(
            t=np.asarray(request.t_eval, dtype=float),
            Y=Y,
            provenance={"solver": "nan_stub"},
        )

    monkeypatch.setattr("kindred.core.simulator.solvers.solve_ode", _nan_solver)
    objective = build_fitting_objective(
        mechanism_text=MECHANISM,
        param_names=["k1"],
        t_exp=t_exp,
        y_exp=y_exp,
        target_species="B",
        solver="LSODA",
    )

    residuals = objective(np.array([0.25]))
    assert residuals.shape == y_exp.shape
    assert np.all(np.isfinite(residuals))
    assert np.allclose(residuals, 1e6)
    assert isinstance(getattr(objective, "last_error", None), FitSimulationError)
    assert getattr(objective, "last_error_provenance", None) == {"solver": "nan_stub"}

    result = fit_parameters(
        objective,
        initial_params={"k1": 0.1},
        method="trf",
        max_nfev=5,
    )

    assert result.residuals.shape == y_exp.shape
    assert np.all(np.isfinite(result.residuals))


def test_de_objective_penalizes_nonfinite(monkeypatch):
    t_exp = np.linspace(0.0, 1.0, 4)
    y_exp = np.zeros_like(t_exp)

    def _nan_solver(request):
        Y = np.vstack(
            [
                np.zeros_like(request.t_eval),
                np.full_like(request.t_eval, np.nan),
            ]
        )
        return SimulationOutput(
            t=np.asarray(request.t_eval, dtype=float),
            Y=Y,
            provenance={"solver": "nan_stub"},
        )

    monkeypatch.setattr("kindred.core.simulator.solvers.solve_ode", _nan_solver)
    objective = build_fitting_objective(
        mechanism_text=MECHANISM,
        param_names=["k1"],
        t_exp=t_exp,
        y_exp=y_exp,
        target_species="B",
        solver="LSODA",
    )

    result = fit_parameters(
        objective,
        initial_params={"k1": 0.1},
        bounds={"k1": (0.0, 1.0)},
        method="de",
        max_nfev=20,
        seed=0,
    )

    assert result.success
    assert np.all(np.isfinite(result.residuals))
    assert np.isfinite(result.chi_squared)
    assert isinstance(getattr(objective, "last_error", None), FitSimulationError)
