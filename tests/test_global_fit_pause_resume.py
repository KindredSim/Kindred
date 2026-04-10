from __future__ import annotations

import threading
import time

import numpy as np
import pytest
from PySide6.QtTest import QSignalSpy

from kindred.gui.fitting.worker import GlobalFitWorker

pytestmark = pytest.mark.gui


def _build_fake_fit_global():
    gate_waiting = threading.Event()
    entered_check = threading.Event()
    proceed = threading.Event()
    proceed.set()

    def fake_fit_global(
        simulation_func,
        datasets,
        shared_params,
        *,
        progress_callback=None,
        cancellation_check=None,
        **_kwargs,
    ):
        iteration = 0
        cost = 10.0
        while True:
            gate_waiting.set()
            proceed.wait(timeout=5)
            proceed.clear()
            gate_waiting.clear()

            entered_check.set()
            if cancellation_check is not None and cancellation_check():
                raise RuntimeError("Fit cancelled by user")
            entered_check.clear()

            simulation_func(dict(shared_params))
            cost -= 1.0
            if progress_callback is not None:
                progress_callback(iteration + 1, cost, {"k": cost})
            iteration += 1
            time.sleep(0.01)

    return fake_fit_global, gate_waiting, entered_check, proceed


def _build_worker(*, fit_func):
    t = np.array([0.0, 1.0], dtype=float)
    datasets = [{"id": "ds", "t": t, "y": np.array([0.0, 0.0], dtype=float), "species": "A"}]

    def simulation(_params):
        return {"t": t, "species": {"A": np.array([0.0, 0.0], dtype=float)}}

    worker = GlobalFitWorker(
        datasets,
        {"k": 1.0},
        fit_evaluator=simulation,
        fit_func=fit_func,
        best_update_interval_s=0.0,
        max_nfev=1000,
    )
    return worker


def test_global_fit_worker_pause_blocks_until_resume(qtbot):
    fake_fit_global, gate_waiting, entered_check, proceed = _build_fake_fit_global()
    worker = _build_worker(fit_func=fake_fit_global)

    assert hasattr(worker, "pause")
    assert hasattr(worker, "resume")

    best_spy = QSignalSpy(worker.bestUpdated)
    progress_spy = QSignalSpy(worker.progress)
    error_spy = QSignalSpy(worker.error)

    try:
        worker.start()
        qtbot.waitUntil(lambda: best_spy.count() >= 1, timeout=2000)
        qtbot.waitUntil(lambda: progress_spy.count() >= 1, timeout=2000)

        qtbot.waitUntil(gate_waiting.is_set, timeout=2000)
        worker.pause()
        proceed.set()
        qtbot.waitUntil(entered_check.is_set, timeout=2000)

        best_before = best_spy.count()
        progress_before = progress_spy.count()
        qtbot.wait(250)
        assert best_spy.count() == best_before
        assert progress_spy.count() == progress_before

        worker.resume()
        qtbot.waitUntil(lambda: best_spy.count() > best_before, timeout=2000)
        qtbot.waitUntil(lambda: progress_spy.count() > progress_before, timeout=2000)

        qtbot.waitUntil(gate_waiting.is_set, timeout=2000)
        worker.cancel()
        proceed.set()
        qtbot.waitUntil(lambda: error_spy.count() == 1, timeout=2000)
    finally:
        if worker.isRunning():
            if hasattr(worker, "resume"):
                worker.resume()
            worker.cancel()
            proceed.set()
            worker.wait(2000)


def test_pause_then_cancel_unblocks_and_exits(qtbot):
    fake_fit_global, gate_waiting, entered_check, proceed = _build_fake_fit_global()
    worker = _build_worker(fit_func=fake_fit_global)

    assert hasattr(worker, "pause")
    assert hasattr(worker, "resume")

    best_spy = QSignalSpy(worker.bestUpdated)
    error_spy = QSignalSpy(worker.error)

    try:
        worker.start()
        qtbot.waitUntil(lambda: best_spy.count() >= 1, timeout=2000)

        qtbot.waitUntil(gate_waiting.is_set, timeout=2000)
        worker.pause()
        proceed.set()
        qtbot.waitUntil(entered_check.is_set, timeout=2000)

        worker.cancel()
        qtbot.waitUntil(lambda: error_spy.count() == 1, timeout=2000)
        worker.wait(2000)
        assert not worker.isRunning()
    finally:
        if worker.isRunning():
            if hasattr(worker, "resume"):
                worker.resume()
            worker.cancel()
            proceed.set()
            worker.wait(2000)
