from __future__ import annotations

import numpy as np
import pytest

import kindred.core.fitting_optimization as fitting_optimization
from kindred.core.exceptions import FittingCancelled
from kindred.gui.fitting.worker import GlobalFitWorker


pytestmark = [pytest.mark.gui]


def test_fit_parameters_raises_typed_cancellation(monkeypatch):
    def fake_least_squares(fun, x0, **_kwargs):
        fun(np.asarray(x0, dtype=float))
        raise AssertionError("least_squares should not continue after cancellation")

    monkeypatch.setattr(
        fitting_optimization,
        "load_scipy_optimize",
        lambda: (fake_least_squares, lambda *_a, **_k: None),
    )

    with pytest.raises(FittingCancelled, match="cancelled"):
        fitting_optimization.fit_parameters(
            lambda x: np.asarray([float(x[0])], dtype=float),
            {"k": 0.0},
            method="trf",
            cancellation_check=lambda: True,
        )


def test_global_fit_worker_translates_typed_cancellation_to_standard_message(qtbot):
    t = np.array([0.0, 1.0], dtype=float)
    datasets = [{"id": "ds", "t": t, "y": np.array([0.0, 0.0], dtype=float), "species": "A"}]

    def simulation(_params):
        return {"t": t, "species": {"A": np.array([0.0, 0.0], dtype=float)}}

    def fake_fit_global(*_args, **_kwargs):
        raise FittingCancelled("typed cancellation from core")

    worker = GlobalFitWorker(
        datasets,
        {"k": 1.0},
        fit_evaluator=simulation,
        fit_func=fake_fit_global,
    )

    with qtbot.waitSignal(worker.error, timeout=3000) as blocker:
        worker.start()

    worker.wait(1000)
    payload = blocker.args[0]
    assert payload["kind"] == "cancelled"
    assert payload["message"] == "Fit cancelled by user"
    assert payload["details"]["origin_message"] == "typed cancellation from core"
