from __future__ import annotations

import numpy as np
import pytest


pytestmark = [pytest.mark.unit]


def test_parametric_x_alignment_failure_yields_finite_penalty_and_failed_result(monkeypatch) -> None:
    from kindred.core.analysis import global_fitting
    import kindred.core.fitting_optimization as fitting_optimization

    t_obs = np.linspace(0.0, 1.0, 6, dtype=float)
    # Observed X values far outside the model X range [0, 1] in-window.
    x_obs = np.full_like(t_obs, 10.0, dtype=float)
    y_obs = np.zeros_like(t_obs, dtype=float).reshape(1, -1)

    datasets = [
        {
            "id": "ds1",
            "t": t_obs.copy(),
            "y": y_obs.copy(),
            "species": ["Y"],
            "x_name": "X",
            "x_obs": x_obs.copy(),
            "x_mapping_mode": "auto",
        }
    ]

    t_sim = np.linspace(0.0, 1.0, 101, dtype=float)

    def simulation_func(_params):
        return {"t": t_sim.copy(), "species": {"X": t_sim.copy(), "Y": np.zeros_like(t_sim)}}

    def _fake_least_squares(fun, x0, **_kwargs):
        # Evaluate objective twice to mimic iterative optimizer calls.
        r1 = np.asarray(fun(np.asarray(x0, dtype=float)), dtype=float).reshape(-1)
        r2 = np.asarray(fun(np.asarray(x0, dtype=float)), dtype=float).reshape(-1)
        assert r1.size == r2.size

        class _Result:
            pass

        result = _Result()
        result.x = np.asarray(x0, dtype=float)
        result.success = True
        result.message = "fake"
        result.nfev = 2
        result.fun = r2
        result.jac = np.zeros((r2.size, result.x.size), dtype=float)
        return result

    monkeypatch.setattr(
        fitting_optimization,
        "load_scipy_optimize",
        lambda: (_fake_least_squares, lambda *_a, **_k: None),
    )

    result = global_fitting.fit_global(
        simulation_func,
        datasets=datasets,
        shared_params={"k1": 0.2},
        weights=None,
        method="trf",
        max_nfev=2,
    )

    residuals = np.asarray(result.objective_residuals, dtype=float).reshape(-1)
    # dx-penalty residuals add another block of length N for parametric-X datasets.
    assert residuals.size == 2 * int(t_obs.size)
    assert np.all(np.isfinite(residuals))
    assert float(np.max(np.abs(residuals))) > 1.0

    assert np.isfinite(float(result.global_chi_squared))
    assert result.success is True
    assert getattr(result, "dataset_errors", {}) == {}
    warnings = getattr(result, "dataset_warnings", {}) or {}
    assert "ds1" in warnings
