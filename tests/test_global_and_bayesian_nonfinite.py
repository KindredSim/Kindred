import numpy as np
import pytest

from kindred.core.analysis.fit_dataset_payload import FitDatasetSpec
from kindred.core.analysis.global_fitting import (
    _FitParameterLayout,
    _GlobalFitObjective,
    fit_global,
)
from kindred.core.objective import ObjectiveContext
from kindred.core.simulation_preparation import build_prepared_simulation_func
from kindred.core.simulator.solvers import SimulationOutput


def test_global_fit_penalty_on_nonfinite_dataset():
    t = np.linspace(0.0, 1.0, 4)
    datasets = [
        {"id": "ok", "t": t, "y": np.zeros_like(t), "species": "A"},
        {"id": "bad", "t": t, "y": np.zeros_like(t), "species": "A"},
    ]

    def _sim(_params):
        return {"t": t, "A": np.array([0.0, np.nan, 0.0, 0.0])}

    result = fit_global(
        _sim,
        datasets,
        shared_params={"k": 0.1},
        method="trf",
        max_nfev=10,
    )

    expected_len = sum(len(ds["t"]) for ds in datasets)
    assert result.objective_residuals is not None
    assert result.objective_residuals.shape == (expected_len,)
    assert np.all(np.isfinite(result.objective_residuals))
    assert "last_error_dataset" in result.message.lower()


@pytest.mark.unit
def test_global_fit_objective_penalizes_nonfinite_param_without_stale_binding_reuse(monkeypatch):
    mechanism_text = "\n".join(
        [
            "reaction: A -> B; k=0.5",
            "reaction: B -> C; k=0.1",
            "initial: A=1.0",
            "initial: B=0.0",
            "initial: C=0.0",
        ]
    )

    def _binding_sensitive_solver(request):
        dy0 = np.asarray(request.rhs(0.0, np.asarray(request.y0, dtype=float)), dtype=float).reshape(-1)
        b_rate = float(dy0[1])
        t = np.asarray([0.0, float(request.t_span[1])], dtype=float)
        y = np.vstack(
            [
                np.ones_like(t),
                np.full_like(t, b_rate, dtype=float),
                np.zeros_like(t),
            ]
        )
        return SimulationOutput(t=t, Y=y, provenance={"solver": "binding_stub"})

    monkeypatch.setattr("kindred.core.simulator.solvers.solve_ode", _binding_sensitive_solver)

    simulation_func = build_prepared_simulation_func(
        mechanism_text=mechanism_text,
        param_names=["k1", "k2"],
        t_end=1.0,
        num_points=2,
        solver="LSODA",
        rtol=1e-6,
        atol=1e-12,
    )
    payload = FitDatasetSpec(
        dataset_id="ds1",
        t_exp=np.asarray([0.0, 1.0], dtype=float),
        species_list=["B"],
        y_matrix=np.zeros((1, 2), dtype=float),
        point_count=2,
        x_name="t",
        x_obs=None,
        x_mode="auto",
    )
    layout = _FitParameterLayout(
        param_names=["k1", "k2"],
        shared_log10={},
        dataset_var_order=[],
        dataset_var_index={},
        dataset_var_log10={},
        x0=np.asarray([0.25, 0.5], dtype=float),
        lower=np.asarray([0.0, 0.0], dtype=float),
        upper=np.asarray([1.0, 1.0], dtype=float),
    )
    objective = _GlobalFitObjective(
        simulation_func=simulation_func,
        payloads=[payload],
        shared_params={"k1": 0.25, "k2": 0.5},
        dataset_params={},
        weights={"ds1": 1.0},
        layout=layout,
        penalty_value=1e6,
        ctx=ObjectiveContext(),
        progress_callback=None,
        cancellation_check=None,
    )

    finite_residuals = objective(np.asarray([0.25, 0.5], dtype=float))
    assert np.allclose(finite_residuals, np.asarray([0.25, 0.25], dtype=float))

    nonfinite_residuals = objective(np.asarray([np.nan, 0.5], dtype=float))
    assert np.all(np.isfinite(nonfinite_residuals))
    assert np.allclose(nonfinite_residuals, np.full(2, 1e6, dtype=float))
