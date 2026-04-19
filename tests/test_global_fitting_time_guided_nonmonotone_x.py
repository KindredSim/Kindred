from __future__ import annotations

import numpy as np
import pytest
from kindred.core.simulation_failure import simulation_failure_user_message


pytestmark = [pytest.mark.unit]


def test_global_fitting_time_guided_mode_allows_nonmonotone_x(monkeypatch) -> None:
    from kindred.core.analysis import global_fitting
    import kindred.core.fitting_optimization as fitting_optimization

    t_obs = np.linspace(0.0, 1.0, 11, dtype=float)
    x_obs = t_obs * (1.0 - t_obs)
    y_obs = t_obs.reshape(1, -1)

    datasets = [
        {
            "id": "ds1",
            "t": t_obs.copy(),
            "y": y_obs.copy(),
            "species": ["Y"],
            "x_name": "X",
            "x_obs": x_obs.copy(),
            "x_mapping_mode": "time_guided",
        }
    ]

    t_sim = np.linspace(0.0, 1.0, 2001, dtype=float)

    def simulation_func(_params):
        return {
            "t": t_sim.copy(),
            "species": {
                "X": (t_sim * (1.0 - t_sim)).copy(),
                "Y": t_sim.copy(),
            },
        }

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
    assert float(np.max(np.abs(y_resid))) == pytest.approx(0.0, abs=1e-2)
    assert float(np.max(np.abs(dx_resid))) == pytest.approx(0.0, abs=1e-8)

    assert result.dataset_info
    assert float(result.dataset_info[0].rmse) == pytest.approx(0.0, abs=1e-2)


def test_global_fitting_monotone_only_rejects_nonmonotone_x(monkeypatch) -> None:
    from kindred.core.analysis import global_fitting
    import kindred.core.fitting_optimization as fitting_optimization

    t_obs = np.linspace(0.0, 1.0, 11, dtype=float)
    x_obs = t_obs * (1.0 - t_obs)
    y_obs = t_obs.reshape(1, -1)

    datasets = [
        {
            "id": "ds1",
            "t": t_obs.copy(),
            "y": y_obs.copy(),
            "species": ["Y"],
            "x_name": "X",
            "x_obs": x_obs.copy(),
            "x_mapping_mode": "monotone",
        }
    ]

    t_sim = np.linspace(0.0, 1.0, 2001, dtype=float)

    def simulation_func(_params):
        return {
            "t": t_sim.copy(),
            "species": {
                "X": (t_sim * (1.0 - t_sim)).copy(),
                "Y": t_sim.copy(),
            },
        }

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
    assert float(np.max(np.abs(residuals))) > 1e5
    assert result.completion.status == "fail"
    messages = [
        simulation_failure_user_message(diagnostic.failure).lower()
        for diagnostic in result.completion.dataset_failures.values()
    ]
    assert any("monotone" in message for message in messages)
    assert {
        diagnostic.remediation
        for diagnostic in result.completion.dataset_failures.values()
    } == {"x_axis_mapping"}
