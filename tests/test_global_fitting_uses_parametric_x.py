from __future__ import annotations

import numpy as np
import pytest


pytestmark = [pytest.mark.unit]


def test_global_fitting_objective_uses_parametric_x_alignment(monkeypatch) -> None:
    from kindred.core.analysis import global_fitting
    import kindred.core.fitting_optimization as fitting_optimization

    t_obs = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
    x_obs = np.asarray([0.0, 0.5, 1.0, 1.5, 2.0], dtype=float)
    y_obs = (x_obs**2).reshape(1, -1)

    datasets = [
        {
            "id": "ds1",
            "t": t_obs.copy(),
            "y": y_obs.copy(),
            "species": ["Y"],
            "x_name": "X",
            "x_obs": x_obs.copy(),
        }
    ]

    t_sim = np.linspace(0.0, 4.0, 401)

    def simulation_func(_params):
        return {"t": t_sim.copy(), "species": {"X": t_sim.copy(), "Y": (t_sim**2).copy()}}

    def _fake_least_squares(fun, x0, **_kwargs):
        residuals = np.asarray(fun(np.asarray(x0, dtype=float)), dtype=float).reshape(-1)

        class _Result:
            pass

        result = _Result()
        result.x = np.asarray(x0, dtype=float)
        result.success = True
        result.message = "fake"
        result.nfev = 1
        result.fun = residuals
        result.jac = np.zeros((residuals.size, result.x.size), dtype=float)
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
        max_nfev=1,
    )

    residuals = np.asarray(result.objective_residuals, dtype=float).reshape(-1)
    assert residuals.size == 2 * int(t_obs.size)
    y_resid = residuals[: int(t_obs.size)]
    dx_resid = residuals[int(t_obs.size) :]
    assert float(np.max(np.abs(y_resid))) == pytest.approx(0.0)
    assert float(np.max(np.abs(dx_resid))) == pytest.approx(0.0)
