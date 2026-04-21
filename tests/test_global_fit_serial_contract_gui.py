from __future__ import annotations

import inspect

import numpy as np
import pytest

pytestmark = pytest.mark.gui


def test_serial_only_gui_worker_contract(qt_app, monkeypatch) -> None:
    from PySide6 import QtCore

    from kindred.core.analysis.fit_dataset_payload import coerce_fit_dataset_specs
    from kindred.gui.fitting.window import FittingWindow
    from kindred.gui.fitting.worker import GlobalFitWorker

    worker_signature = inspect.signature(GlobalFitWorker)
    window_signature = inspect.signature(FittingWindow)
    removed_names = {"parallel_enabled", "max_parallel_workers", "limit_blas_threads"}
    assert removed_names.isdisjoint(worker_signature.parameters)
    assert "shared_solver_settings_getter" not in window_signature.parameters

    t = np.linspace(0.0, 1.0, 5)

    def _sim(params):
        value = float(dict(params).get("k", 1.0))
        return {"t": t.copy(), "A": np.full_like(t, value, dtype=float)}

    captured: dict[str, object] = {}

    class _FakeWorker(QtCore.QObject):
        progress = QtCore.Signal(int, str)
        bestUpdated = QtCore.Signal(dict)
        finished = QtCore.Signal(dict)
        error = QtCore.Signal(object)

        def __init__(self, datasets, shared_params, **kwargs):
            super().__init__()
            captured["datasets"] = list(datasets)
            captured["shared_params"] = dict(shared_params)
            captured["kwargs"] = dict(kwargs)

        def start(self):
            return None

        def isRunning(self):
            return False

        def cancel(self):
            return None

    monkeypatch.setattr("kindred.gui.fitting.window.GlobalFitWorker", _FakeWorker)

    dataset_payloads = [
        {"id": "ds1", "t": t.copy(), "y": np.vstack([np.ones_like(t)]), "species": ["A"]},
    ]
    window = FittingWindow(
        mode="global",
        parameter_defs=[{"name": "k", "value": 1.0, "min": 0.1, "max": 2.0}],
        dataset_entries=[
            {
                "id": "ds1",
                "label": "ds1",
                "t": t.copy(),
                "species_data": {"A": np.ones_like(t)},
                "selected_species": ["A"],
                "weight": 1.0,
                "include": True,
            }
        ],
        simulation_func=_sim,
        dataset_payloads=dataset_payloads,
        dataset_weights={"ds1": 1.0},
    )
    try:
        assert not hasattr(window, "_parallel_fit_runtime_settings_for_run")
        window._start_global_fit_worker(
            datasets=coerce_fit_dataset_specs(dataset_payloads),
            config={
                "parameters": {"k": 1.0},
                "max_nfev": 5,
            },
            dataset_overrides=[],
            weights={"ds1": 1.0},
            requested_solver="BDF",
            requested_rtol=1e-6,
            requested_atol=1e-12,
            fit_evaluator=_sim,
            stamp={},
            stamp_hash="stamp-hash",
            stamp_short="stamp",
        )
    finally:
        window.close()

    worker_kwargs = captured["kwargs"]
    assert removed_names.isdisjoint(worker_kwargs)
    assert "max_parallel_batch_workers" not in worker_kwargs
    assert "limit_blas_threads_per_worker" not in worker_kwargs
