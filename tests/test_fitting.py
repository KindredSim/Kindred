import types

import numpy as np
import pytest

from kindred.core.exceptions import FitSimulationError, create_solver_error
import kindred.core.fitting_optimization as fitting_optimization


pytestmark = pytest.mark.unit


def test_de_penalty_scales_with_failure_time(monkeypatch):
    seen: dict[str, float] = {}

    def fake_differential_evolution(func, *, bounds, **_kwargs):
        x = np.array([(float(lo) + float(hi)) / 2.0 for lo, hi in bounds], dtype=float)
        seen["penalty"] = float(func(x))
        return types.SimpleNamespace(x=x, success=False, message="forced failure", nfev=1)

    def fake_load_scipy_optimize():
        return (None, fake_differential_evolution)

    monkeypatch.setattr(fitting_optimization, "load_scipy_optimize", fake_load_scipy_optimize)

    def objective(_params: np.ndarray) -> np.ndarray:
        try:
            raise create_solver_error("BDF", 0.25, "boom")
        except Exception as cause:
            raise FitSimulationError("Sim failed") from cause

    objective._kindred_t_span = (0.0, 1.0)  # type: ignore[attr-defined]

    result = fitting_optimization.fit_parameters(
        objective,
        initial_params={"k": 1.0},
        bounds={"k": (0.0, 2.0)},
        method="de",
        max_nfev=1,
    )

    assert "penalty" in seen
    assert np.isclose(seen["penalty"], 1e12 * 1.75)
    assert result.success is False
