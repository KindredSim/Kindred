from __future__ import annotations

import numpy as np
import pytest


pytestmark = [pytest.mark.integration]


def test_global_fit_parametric_x_penalized_mapping_produces_warnings_not_errors(monkeypatch) -> None:
    """
    Regression: parametric-X (X!=t) runs must not hard-fail when X alignment has no crossing
    within the sampled window; instead they should finish with finite metrics and warnings.
    """
    from kindred.core.analysis import global_fitting
    import kindred.core.fitting_optimization as fitting_optimization

    # Observed grid (this is the sampled window/domain for alignment).
    t_obs = np.linspace(0.0, 1.0, 41, dtype=float)
    x_target = t_obs * (1.0 - t_obs)  # max=0.25
    x_obs = x_target.copy()
    y_obs = t_obs.copy().reshape(1, -1)

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

    t_sim = np.linspace(0.0, 1.0, 401, dtype=float)

    def simulation_func(params):
        # Shared parameter 'a' controls X amplitude; with a<=0.5, X max is <=0.125,
        # so many x_obs points have no crossing.
        a = float(params.get("a", 0.5))
        x_sim = a * t_sim * (1.0 - t_sim)
        y_sim = t_sim.copy()
        return {"t": t_sim.copy(), "species": {"X": x_sim, "Y": y_sim}}

    def _fake_least_squares(fun, x0, **_kwargs):
        # Evaluate objective twice to mimic the optimizer; keep the same x0.
        r1 = np.asarray(fun(np.asarray(x0, dtype=float)), dtype=float).reshape(-1)
        r2 = np.asarray(fun(np.asarray(x0, dtype=float)), dtype=float).reshape(-1)
        assert r1.size == r2.size
        assert np.all(np.isfinite(r1))
        assert np.all(np.isfinite(r2))

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

    # Bound 'a' so alignment cannot become exact; expect warnings, not errors.
    result = global_fitting.fit_global(
        simulation_func,
        datasets=datasets,
        shared_params={"a": 0.5},
        bounds={"a": (0.1, 0.5)},
        weights=None,
        method="trf",
        max_nfev=2,
    )

    assert bool(np.isfinite(float(result.global_chi_squared)))
    assert getattr(result, "dataset_errors", {}) == {}

    warnings = getattr(result, "dataset_warnings", None)
    assert isinstance(warnings, dict) and warnings, "Expected alignment warnings when crossings are impossible."
    assert "ds1" in warnings

    residuals = np.asarray(result.objective_residuals, dtype=float).reshape(-1)
    assert np.all(np.isfinite(residuals))
    # Penalized dx residuals add another block of length N for this dataset/species.
    assert residuals.size == 2 * int(t_obs.size)
