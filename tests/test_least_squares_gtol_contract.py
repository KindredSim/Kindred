import types

import numpy as np
import pytest

import kindred.core.analysis.global_fitting as global_fitting
import kindred.core.fitting_optimization as fitting_optimization
from kindred.core.exceptions import FittingCancelled
from kindred.core.fitting_evaluation import CallableFittingEvaluator
from kindred.core.optimization_least_squares import build_least_squares_kwargs


def test_build_least_squares_kwargs_enforces_gtol_equals_ftol():
    kwargs = build_least_squares_kwargs(ftol=1e-9, xtol=1e-8, max_nfev=10)
    assert kwargs["gtol"] == kwargs["ftol"]

    with pytest.raises(ValueError):
        build_least_squares_kwargs(ftol=1e-9, xtol=1e-8, max_nfev=10, gtol=1e-12)


def test_fit_parameters_passes_gtol_equal_ftol(monkeypatch):
    captured = {}

    def fake_least_squares(fun, x0, **kwargs):
        captured.update(kwargs)
        x = np.asarray(x0, dtype=float)
        fun_val = np.asarray(fun(x), dtype=float).reshape(-1)
        return types.SimpleNamespace(
            x=x,
            success=True,
            message="ok",
            nfev=1,
            jac=np.eye(x.size, dtype=float),
            fun=fun_val,
        )

    monkeypatch.setattr(
        fitting_optimization,
        "load_scipy_optimize",
        lambda: (fake_least_squares, lambda *_a, **_k: None),
    )

    def objective(x):
        return np.asarray([float(x[0]) - 1.0], dtype=float)

    _ = fitting_optimization.fit_parameters(
        objective,
        {"k": 0.0},
        method="trf",
        max_nfev=5,
        ftol=1e-9,
        xtol=1e-8,
    )

    assert captured["gtol"] == pytest.approx(captured["ftol"])


def test_fit_global_passes_gtol_equal_ftol(monkeypatch):
    def fake_least_squares(*_a, **kwargs):
        assert kwargs["gtol"] == pytest.approx(kwargs["ftol"])
        raise RuntimeError("cancelled test")

    monkeypatch.setattr(
        fitting_optimization,
        "load_scipy_optimize",
        lambda: (fake_least_squares, lambda *_a, **_k: None),
    )

    datasets = [
        {
            "id": "d1",
            "t": np.array([0.0, 1.0], dtype=float),
            "y": np.array([0.0, 0.0], dtype=float),
            "species": "B",
        }
    ]

    def simulate(_params):
        return {"t": np.array([0.0, 1.0], dtype=float), "B": np.array([0.0, 0.0], dtype=float)}

    with pytest.raises(FittingCancelled, match="cancelled"):
        global_fitting.fit_global(
            CallableFittingEvaluator(simulate),
            datasets,
            {"k": 0.1},
            method="trf",
            max_nfev=1,
            ftol=1e-9,
            xtol=1e-8,
        )


def test_fit_parameters_passes_scalar_diff_step_for_lm(monkeypatch):
    captured = {}

    def fake_least_squares(fun, x0, **kwargs):
        captured.update(kwargs)
        x = np.asarray(x0, dtype=float)
        fun_val = np.asarray(fun(x), dtype=float).reshape(-1)
        return types.SimpleNamespace(
            x=x,
            success=True,
            message="ok",
            nfev=1,
            jac=np.eye(x.size, dtype=float),
            fun=fun_val,
        )

    monkeypatch.setattr(
        fitting_optimization,
        "load_scipy_optimize",
        lambda: (fake_least_squares, lambda *_a, **_k: None),
    )

    result = fitting_optimization.fit_parameters(
        lambda x: np.asarray([float(x[0]) - 1.0, float(x[1]) - 2.0], dtype=float),
        {"k1": 0.0, "k2": 1.5},
        method="lm",
        max_nfev=5,
        ftol=1e-9,
        xtol=1e-8,
    )

    assert result.success is True
    assert np.isscalar(captured["diff_step"])
    assert not isinstance(captured["diff_step"], np.ndarray)
    assert float(captured["diff_step"]) > 0.0


def test_fit_global_passes_scalar_diff_step_for_lm(monkeypatch):
    captured = {}

    def fake_least_squares(fun, x0, **kwargs):
        captured.update(kwargs)
        x = np.asarray(x0, dtype=float)
        fun_val = np.asarray(fun(x), dtype=float).reshape(-1)
        return types.SimpleNamespace(
            x=x,
            success=True,
            message="ok",
            nfev=1,
            jac=np.eye(x.size, dtype=float),
            fun=fun_val,
        )

    monkeypatch.setattr(
        fitting_optimization,
        "load_scipy_optimize",
        lambda: (fake_least_squares, lambda *_a, **_k: None),
    )

    datasets = [
        {
            "id": "d1",
            "t": np.array([0.0, 1.0], dtype=float),
            "y": np.array([0.0, 0.0], dtype=float),
            "species": "B",
        }
    ]

    def simulate(_params):
        return {"t": np.array([0.0, 1.0], dtype=float), "B": np.array([0.0, 0.0], dtype=float)}

    result = global_fitting.fit_global(
        CallableFittingEvaluator(simulate),
        datasets,
        {"k": 0.1},
        method="lm",
        max_nfev=5,
        ftol=1e-9,
        xtol=1e-8,
    )

    assert result.completion.status == "ok"
    assert np.isscalar(captured["diff_step"])
    assert not isinstance(captured["diff_step"], np.ndarray)
    assert float(captured["diff_step"]) > 0.0
