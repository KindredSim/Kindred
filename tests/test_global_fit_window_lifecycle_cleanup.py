from __future__ import annotations

from contextlib import suppress
from unittest import mock

import pytest
import numpy as np
import shiboken6
from PySide6 import QtCore, QtGui

from kindred.core.analysis.global_fitting import DatasetFitInfo, GlobalFitResult
from kindred.gui.fitting.window import FittingWindow

pytestmark = pytest.mark.gui


def _process_deferred_deletes(iterations: int = 5) -> None:
    for _ in range(max(1, int(iterations))):
        QtCore.QCoreApplication.processEvents()
    with suppress(RuntimeError, TypeError):
        QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
    for _ in range(max(1, int(iterations))):
        QtCore.QCoreApplication.processEvents()


def _build_success_result(*, dataset_id: str = "ds1", param_name: str = "k", value: float = 1.0) -> GlobalFitResult:
    model = np.asarray([1.0, 0.8, 0.6], dtype=float)
    return GlobalFitResult(
        success=True,
        shared_params={str(param_name): float(value)},
        dataset_params={str(dataset_id): {}},
        uncertainties=None,
        global_chi_squared=1.0,
        global_r_squared=0.0,
        dataset_info=[
            DatasetFitInfo(
                dataset_id=str(dataset_id),
                r_squared=0.0,
                chi_squared=1.0,
                rmse=1.0,
                mae=1.0,
                residuals=np.asarray([0.0], dtype=float),
                n_points=1,
                weight=1.0,
            )
        ],
        nfev=1,
        message="ok",
        covariance=None,
        objective_residuals=np.asarray([0.0], dtype=float),
        model_series={str(dataset_id): {"A": model}},
        residual_series={str(dataset_id): {"A": np.asarray([0.0, 0.0, 0.0], dtype=float)}},
    )


def _build_window() -> FittingWindow:
    dataset_entries = [
        {
            "id": "ds1",
            "label": "Dataset 1",
            "t": np.asarray([0.0, 1.0, 2.0], dtype=float),
            "species_data": {"A": np.asarray([1.0, 0.8, 0.6], dtype=float)},
            "selected_species": ["A"],
            "weight": 1.0,
            "include": True,
        }
    ]
    dataset_payloads = [{"id": "ds1", "t": np.asarray([0.0, 1.0, 2.0]), "y": np.asarray([1.0, 0.8, 0.6]), "species": "A"}]
    return FittingWindow(
        mode="global",
        parameter_defs=[{"name": "k", "value": 1.0, "min": 0.0, "max": 2.0}],
        dataset_entries=dataset_entries,
        dataset_payloads=dataset_payloads,
        mechanism_species=["A"],
        simulation_func=lambda _params: {"t": np.asarray([0.0, 1.0, 2.0]), "species": {"A": np.asarray([1.0, 0.8, 0.6])}},
    )


class _SignalWorker(QtCore.QObject):
    progress = QtCore.Signal(int, str)
    bestUpdated = QtCore.Signal(dict)
    finished = QtCore.Signal(dict)
    error = QtCore.Signal(object)

    def __init__(self, *_args, **_kwargs) -> None:
        super().__init__()
        self._running = True
        self.cancel_called = False

    def start(self) -> None:
        return None

    def isRunning(self) -> bool:
        return bool(self._running)

    def cancel(self) -> None:
        self.cancel_called = True


def test_global_fit_window_close_deletes_dialog(qt_app, qtbot):
    dataset_entries = [
        {
            "id": "ds1",
            "label": "Dataset 1",
            "t": np.asarray([0.0, 1.0, 2.0]),
            "species_data": {"A": np.asarray([1.0, 0.8, 0.6])},
            "selected_species": ["A"],
            "weight": 1.0,
            "include": True,
        }
    ]

    window = FittingWindow(
        mode="global",
        parameter_defs=[],
        dataset_entries=dataset_entries,
        dataset_payloads=[],
        mechanism_species=["A"],
        simulation_func=lambda _params: {"t": np.asarray([0.0]), "species": {"A": np.asarray([0.0])}},
    )
    qtbot.addWidget(window)
    destroyed = {"fired": False}
    window.destroyed.connect(lambda *_args: destroyed.__setitem__("fired", True))

    window.show()
    qtbot.wait(10)
    window.close()
    _process_deferred_deletes()

    qtbot.waitUntil(lambda: not shiboken6.isValid(window), timeout=2000)
    assert destroyed["fired"] is True


def test_global_fit_window_deletes_worker_after_run(qt_app, qtbot):
    t_axis = np.asarray([0.0, 1.0, 2.0], dtype=float)
    y_axis = np.asarray([1.0, 0.8, 0.6], dtype=float)

    def fake_fit_global(_simulate, datasets, shared_params, **kwargs):
        progress = kwargs.get("progress_callback")
        assert callable(progress)
        progress(1, 1.0, dict(shared_params))
        return GlobalFitResult(
            success=True,
            shared_params=dict(shared_params),
            dataset_params={str(datasets[0]["id"]): {}},
            uncertainties=None,
            global_chi_squared=1.0,
            global_r_squared=0.0,
            dataset_info=[
                DatasetFitInfo(
                    dataset_id=str(datasets[0]["id"]),
                    r_squared=0.0,
                    chi_squared=1.0,
                    rmse=1.0,
                    mae=1.0,
                    residuals=np.asarray([0.0], dtype=float),
                    n_points=1,
                    weight=1.0,
                )
            ],
            nfev=1,
            message="ok",
            covariance=None,
            objective_residuals=np.asarray([0.0], dtype=float),
            model_series={str(datasets[0]["id"]): {"A": y_axis}},
            residual_series={str(datasets[0]["id"]): {"A": np.asarray([0.0, 0.0, 0.0], dtype=float)}},
        )

    dataset_entries = [
        {
            "id": "ds1",
            "label": "Dataset 1",
            "t": t_axis,
            "species_data": {"A": y_axis},
            "selected_species": ["A"],
            "weight": 1.0,
            "include": True,
        }
    ]
    dataset_payloads = [{"id": "ds1", "t": t_axis, "y": y_axis, "species": "A"}]

    window = FittingWindow(
        mode="global",
        parameter_defs=[{"name": "k", "value": 1.0, "min": 0.0, "max": 2.0}],
        dataset_entries=dataset_entries,
        dataset_payloads=dataset_payloads,
        mechanism_species=["A"],
        simulation_func=lambda _params: {"t": t_axis, "species": {"A": y_axis}},
        fit_func=fake_fit_global,
    )
    qtbot.addWidget(window)

    config = {
        "parameters": {"k": 1.0},
        "bounds": {"k": (0.0, 2.0)},
        "fixed_params": {},
        "method": "trf",
        "max_nfev": 1,
        "seed": None,
        "log10_params": {},
    }
    dataset_selection = {"rows": [{"id": "ds1", "label": "Dataset 1", "species": "A", "include": True, "weight": 1.0}], "ids": ["ds1"]}

    window._start_global_fit(config, dataset_selection)
    worker = window._worker
    assert worker is not None
    qtbot.waitUntil(lambda: (not shiboken6.isValid(worker)) or worker.isFinished(), timeout=5000)
    _process_deferred_deletes(iterations=10)
    qtbot.waitUntil(lambda: not shiboken6.isValid(worker), timeout=2000)


def test_global_fit_window_close_hard_terminates_stuck_worker(qt_app, qtbot):
    dataset_entries = [
        {
            "id": "ds1",
            "label": "Dataset 1",
            "t": np.asarray([0.0, 1.0, 2.0]),
            "species_data": {"A": np.asarray([1.0, 0.8, 0.6])},
            "selected_species": ["A"],
            "weight": 1.0,
            "include": True,
        }
    ]

    window = FittingWindow(
        mode="global",
        parameter_defs=[],
        dataset_entries=dataset_entries,
        dataset_payloads=[],
        mechanism_species=["A"],
        simulation_func=lambda _params: {"t": np.asarray([0.0]), "species": {"A": np.asarray([0.0])}},
    )
    qtbot.addWidget(window)

    class _StuckWorker:
        def __init__(self):
            self.cancel_called = False
            self.terminate_called = False
            self.wait_calls = []
            self.deleted = False
            self._running = True
            self.finished = mock.MagicMock()

        def isRunning(self):
            return bool(self._running)

        def cancel(self):
            self.cancel_called = True

        def wait(self, msecs: int | None = None):
            self.wait_calls.append(msecs)
            return not self._running

        def terminate(self):
            self.terminate_called = True

        def deleteLater(self):
            self.deleted = True

    stuck = _StuckWorker()
    window._worker = stuck  # type: ignore[assignment]

    event = QtGui.QCloseEvent()
    window.closeEvent(event)

    assert stuck.cancel_called is True
    assert stuck.wait_calls and int(stuck.wait_calls[0]) == 2000
    assert stuck.terminate_called is True
    assert stuck.deleted is False
    assert window._worker_registry.contains_thread(window._worker)
    assert event.isAccepted() is True
    window._worker_registry.release_thread(window._worker)
    window._worker = None


def test_stale_finished_from_older_fit_worker_does_not_clear_newer_worker(qt_app, monkeypatch):
    workers: list[_SignalWorker] = []

    class _FactoryWorker(_SignalWorker):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            workers.append(self)

    monkeypatch.setattr("kindred.gui.fitting.window.GlobalFitWorker", _FactoryWorker)
    monkeypatch.setattr(QtCore.QTimer, "singleShot", lambda _ms, fn: fn())

    window = _build_window()
    try:
        config = {
            "parameters": {"k": 1.0},
            "bounds": {"k": (0.0, 2.0)},
            "fixed_params": {},
            "method": "trf",
            "max_nfev": 2,
            "seed": None,
            "log10_params": {},
        }

        window._set_running_state(True)
        window._start_global_fit_worker(
            datasets=[],
            config=config,
            dataset_overrides=[],
            weights=None,
            requested_solver="LSODA",
            requested_rtol=1e-6,
            requested_atol=1e-12,
            fit_evaluator=lambda _params: {},
            stamp={},
            stamp_hash="old",
            stamp_short="old",
        )
        old_worker = workers[-1]
        old_worker._running = False

        window._set_running_state(True)
        window._start_global_fit_worker(
            datasets=[],
            config=config,
            dataset_overrides=[],
            weights=None,
            requested_solver="LSODA",
            requested_rtol=1e-6,
            requested_atol=1e-12,
            fit_evaluator=lambda _params: {},
            stamp={},
            stamp_hash="new",
            stamp_short="new",
        )
        new_worker = workers[-1]
        assert window._worker is new_worker
        assert window._stop_button.isEnabled() is True

        old_worker.finished.emit({"result": _build_success_result()})

        assert window._worker is new_worker
        assert window._stop_button.isEnabled() is True
    finally:
        window.close()


def test_stale_error_and_best_update_from_older_fit_worker_do_not_clobber_newer_run(qt_app, monkeypatch):
    workers: list[_SignalWorker] = []

    class _FactoryWorker(_SignalWorker):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            workers.append(self)

    monkeypatch.setattr("kindred.gui.fitting.window.GlobalFitWorker", _FactoryWorker)
    warning_calls = []
    monkeypatch.setattr(
        "PySide6.QtWidgets.QMessageBox.warning",
        lambda *_args, **_kwargs: warning_calls.append(True),
    )
    monkeypatch.setattr(QtCore.QTimer, "singleShot", lambda _ms, fn: fn())

    window = _build_window()
    try:
        config = {
            "parameters": {"k": 1.0},
            "bounds": {"k": (0.0, 2.0)},
            "fixed_params": {},
            "method": "trf",
            "max_nfev": 2,
            "seed": None,
            "log10_params": {},
        }

        window._set_running_state(True)
        window._start_global_fit_worker(
            datasets=[],
            config=config,
            dataset_overrides=[],
            weights=None,
            requested_solver="LSODA",
            requested_rtol=1e-6,
            requested_atol=1e-12,
            fit_evaluator=lambda _params: {},
            stamp={},
            stamp_hash="old",
            stamp_short="old",
        )
        old_worker = workers[-1]
        old_worker._running = False

        window._set_running_state(True)
        window._start_global_fit_worker(
            datasets=[],
            config=config,
            dataset_overrides=[],
            weights=None,
            requested_solver="LSODA",
            requested_rtol=1e-6,
            requested_atol=1e-12,
            fit_evaluator=lambda _params: {},
            stamp={},
            stamp_hash="new",
            stamp_short="new",
        )
        new_worker = workers[-1]
        window._params_ics_tab.set_last_fit_params({"k": 77.0})
        window._best_cost = 123.0
        window._status_label.setText("Running newer fit")

        old_worker.bestUpdated.emit(
            {
                "cost": 9.0,
                "shared_params": {"k": 0.1},
                "dataset_params": {"ds1": {}},
                "model_series": {},
                "dataset_stats": {},
            }
        )
        window._apply_pending_best_update()

        assert window._worker is new_worker
        assert window._params_ics_tab.get_last_fit_params() == {"k": 77.0}
        assert window._best_cost == 123.0
        assert window._status_label.text() == "Running newer fit"

        old_worker.error.emit({"kind": "fitting_error", "message": "stale boom"})

        assert window._worker is new_worker
        assert window._stop_button.isEnabled() is True
        assert window._status_label.text() == "Running newer fit"
        assert warning_calls == []
    finally:
        window.close()


def test_detached_fit_worker_registry_releases_without_custom_finished_payload(qt_app, qtbot):
    window = _build_window()

    class _StuckWorker:
        def __init__(self):
            self.cancel_called = False
            self.terminate_called = False
            self.wait_calls = []
            self.deleted = False
            self._running = True
            self.finished = mock.MagicMock()

        def isRunning(self):
            return bool(self._running)

        def cancel(self):
            self.cancel_called = True

        def wait(self, msecs: int | None = None):
            self.wait_calls.append(msecs)
            return not self._running

        def terminate(self):
            self.terminate_called = True

        def deleteLater(self):
            self.deleted = True

        def setParent(self, _parent):
            return None

    stuck = _StuckWorker()
    window._worker = stuck  # type: ignore[assignment]

    event = QtGui.QCloseEvent()
    window.closeEvent(event)
    assert window._worker_registry.contains_thread(stuck)

    stuck._running = False
    qtbot.waitUntil(lambda: not window._worker_registry.contains_thread(stuck), timeout=2000)
    qtbot.waitUntil(lambda: stuck.deleted is True, timeout=2000)
    assert stuck.deleted is True


def test_detached_fit_worker_late_emissions_do_not_reenter_deleted_dialog(qt_app, qtbot, monkeypatch):
    workers: list[_SignalWorker] = []

    class _LateSignalWorker(_SignalWorker):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.wait_calls: list[int] = []
            self.terminate_called = False
            workers.append(self)

        def wait(self, msecs: int | None = None) -> bool:
            self.wait_calls.append(int(msecs or 0))
            return False

        def terminate(self) -> None:
            self.terminate_called = True

    monkeypatch.setattr("kindred.gui.fitting.window.GlobalFitWorker", _LateSignalWorker)

    window = _build_window()
    callbacks: list[tuple[str, object]] = []
    window._on_worker_progress = lambda percent, message, *, worker=None: callbacks.append(  # type: ignore[method-assign]
        ("progress", worker)
    )
    window._handle_global_best_update = lambda payload, *, worker=None: callbacks.append(  # type: ignore[method-assign]
        ("best", worker)
    )
    window._handle_global_fit_complete = lambda payload, *, worker=None: callbacks.append(  # type: ignore[method-assign]
        ("finished", worker)
    )
    window._on_worker_error = lambda error, *, worker=None: callbacks.append(("error", worker))  # type: ignore[method-assign]
    window._schedule_worker_cleanup = lambda worker: callbacks.append(("cleanup", worker))  # type: ignore[method-assign]

    config = {
        "parameters": {"k": 1.0},
        "bounds": {"k": (0.0, 2.0)},
        "fixed_params": {},
        "method": "trf",
        "max_nfev": 2,
        "seed": None,
        "log10_params": {},
    }

    window._set_running_state(True)
    window._start_global_fit_worker(
        datasets=[],
        config=config,
        dataset_overrides=[],
        weights=None,
        requested_solver="LSODA",
        requested_rtol=1e-6,
        requested_atol=1e-12,
        fit_evaluator=lambda _params: {},
        stamp={},
        stamp_hash="late",
        stamp_short="late",
    )
    worker = workers[-1]

    window.close()
    _process_deferred_deletes(iterations=10)
    qtbot.waitUntil(lambda: not shiboken6.isValid(window), timeout=2000)
    assert window._worker_registry.contains_thread(worker)

    worker.progress.emit(17, "late progress")
    worker.bestUpdated.emit({"cost": 3.14})
    worker.finished.emit({"result": _build_success_result()})
    worker.error.emit({"kind": "fitting_error", "message": "late error"})
    QtCore.QCoreApplication.processEvents()
    QtCore.QCoreApplication.processEvents()

    assert callbacks == []


def test_consecutive_fit_dispatch_cycles_leave_clean_state(qt_app, monkeypatch):
    workers: list[_SignalWorker] = []

    class _FactoryWorker(_SignalWorker):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.quit_called = False
            self.wait_calls: list[int] = []
            workers.append(self)

        def quit(self) -> None:
            self.quit_called = True

        def wait(self, msecs: int | None = None) -> bool:
            self.wait_calls.append(int(msecs or 0))
            self._running = False
            return True

        def deleteLater(self) -> None:
            return None

    monkeypatch.setattr("kindred.gui.fitting.window.GlobalFitWorker", _FactoryWorker)
    monkeypatch.setattr("PySide6.QtWidgets.QMessageBox.information", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("PySide6.QtWidgets.QMessageBox.warning", lambda *_args, **_kwargs: None)

    window = _build_window()
    try:
        config = {
            "parameters": {"k": 1.0},
            "bounds": {"k": (0.0, 2.0)},
            "fixed_params": {},
            "method": "trf",
            "max_nfev": 2,
            "seed": None,
            "log10_params": {},
        }

        for run_index in range(3):
            window._set_running_state(True)
            window._start_global_fit_worker(
                datasets=[],
                config=config,
                dataset_overrides=[],
                weights=None,
                requested_solver="LSODA",
                requested_rtol=1e-6,
                requested_atol=1e-12,
                fit_evaluator=lambda _params: {},
                stamp={},
                stamp_hash=f"run-{run_index}",
                stamp_short=f"run-{run_index}",
            )
            worker = workers[-1]
            window._pending_best_payload = {"iteration": run_index + 1}
            window._pending_best_worker = worker
            window._pending_best_timer.start()

            worker._running = False
            worker.finished.emit({"result": _build_success_result(value=float(run_index + 1))})
            QtCore.QCoreApplication.processEvents()

            assert window._worker is None
            assert not window._pending_best_timer.isActive()
            assert window._pending_best_payload is None
            assert window._pending_best_worker is None
            assert worker.quit_called is True
            assert worker.wait_calls == [2000]
    finally:
        window.close()


def test_old_worker_best_update_is_disconnected_after_completion(qt_app, monkeypatch):
    workers: list[_SignalWorker] = []

    class _FactoryWorker(_SignalWorker):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.wait_calls: list[int] = []
            workers.append(self)

        def quit(self) -> None:
            return None

        def wait(self, msecs: int | None = None) -> bool:
            self.wait_calls.append(int(msecs or 0))
            self._running = False
            return True

        def deleteLater(self) -> None:
            return None

    monkeypatch.setattr("kindred.gui.fitting.window.GlobalFitWorker", _FactoryWorker)
    monkeypatch.setattr("PySide6.QtWidgets.QMessageBox.information", lambda *_args, **_kwargs: None)

    window = _build_window()
    try:
        config = {
            "parameters": {"k": 1.0},
            "bounds": {"k": (0.0, 2.0)},
            "fixed_params": {},
            "method": "trf",
            "max_nfev": 2,
            "seed": None,
            "log10_params": {},
        }
        callbacks: list[dict] = []
        monkeypatch.setattr(
            window,
            "_dispatch_fit_worker_best_update",
            lambda payload: callbacks.append(dict(payload)),
        )

        window._set_running_state(True)
        window._start_global_fit_worker(
            datasets=[],
            config=config,
            dataset_overrides=[],
            weights=None,
            requested_solver="LSODA",
            requested_rtol=1e-6,
            requested_atol=1e-12,
            fit_evaluator=lambda _params: {},
            stamp={},
            stamp_hash="done",
            stamp_short="done",
        )
        worker = workers[-1]
        worker._running = False
        worker.finished.emit({"result": _build_success_result()})
        QtCore.QCoreApplication.processEvents()

        worker.bestUpdated.emit({"cost": 99.0})
        QtCore.QCoreApplication.processEvents()

        assert callbacks == []
    finally:
        window.close()


def test_completion_stops_pending_best_timer_before_dialog(qt_app, monkeypatch):
    window = _build_window()
    try:
        window.show()
        qt_app.processEvents()
        worker = _SignalWorker()
        worker._running = False
        window._worker = worker
        window._pending_best_payload = {"cost": 1.0}
        window._pending_best_worker = worker
        window._pending_best_timer.start()

        states: list[tuple[bool, object, object]] = []

        def _capture_dialog(*_args, **_kwargs):
            states.append(
                (
                    window._pending_best_timer.isActive(),
                    window._pending_best_payload,
                    window._pending_best_worker,
                )
            )
            return None

        monkeypatch.setattr("PySide6.QtWidgets.QMessageBox.information", _capture_dialog)

        window._handle_global_fit_complete({"result": _build_success_result()}, worker=worker)

        assert states == [(False, None, None)]
    finally:
        window.close()


def test_error_stops_pending_best_timer_before_dialog(qt_app, monkeypatch):
    window = _build_window()
    try:
        window.show()
        qt_app.processEvents()
        worker = _SignalWorker()
        worker._running = False
        window._worker = worker
        window._pending_best_payload = {"cost": 1.0}
        window._pending_best_worker = worker
        window._pending_best_timer.start()

        states: list[tuple[bool, object, object]] = []

        def _capture_dialog(*_args, **_kwargs):
            states.append(
                (
                    window._pending_best_timer.isActive(),
                    window._pending_best_payload,
                    window._pending_best_worker,
                )
            )
            return None

        monkeypatch.setattr("PySide6.QtWidgets.QMessageBox.warning", _capture_dialog)

        window._on_worker_error({"kind": "fitting_error", "message": "boom"}, worker=worker)

        assert states == [(False, None, None)]
    finally:
        window.close()


def test_close_teardown_disconnects_worker_signals(qt_app, monkeypatch):
    """_hard_teardown_worker must disconnect signals so queued emissions
    after closeEvent cannot reach handlers on a partially-destroyed window."""
    workers: list[_SignalWorker] = []

    class _TeardownWorker(_SignalWorker):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.wait_calls: list[int] = []
            workers.append(self)

        def wait(self, msecs: int | None = None) -> bool:
            self.wait_calls.append(int(msecs or 0))
            self._running = False
            return True

        def quit(self) -> None:
            return None

        def deleteLater(self) -> None:
            return None

    monkeypatch.setattr("kindred.gui.fitting.window.GlobalFitWorker", _TeardownWorker)

    window = _build_window()
    try:
        config = {
            "parameters": {"k": 1.0},
            "bounds": {"k": (0.0, 2.0)},
            "fixed_params": {},
            "method": "trf",
            "max_nfev": 2,
            "seed": None,
            "log10_params": {},
        }
        window._set_running_state(True)
        window._start_global_fit_worker(
            datasets=[],
            config=config,
            dataset_overrides=[],
            weights=None,
            requested_solver="LSODA",
            requested_rtol=1e-6,
            requested_atol=1e-12,
            fit_evaluator=lambda _params: {},
            stamp={},
            stamp_hash="teardown",
            stamp_short="teardown",
        )
        worker = workers[-1]
        assert window._worker is worker

        # Track calls through downstream handlers (not dispatch slots) so that
        # _disconnect_fit_worker_signals sees the original slot methods.
        best_handler_calls: list[dict] = []
        progress_handler_calls: list[tuple] = []
        window._handle_global_best_update = lambda payload, *, worker=None: best_handler_calls.append(dict(payload))
        window._on_worker_progress = lambda p, m, *, worker=None: progress_handler_calls.append((p, m))

        window._hard_teardown_worker(reason="test teardown", disable_ui=False)

        worker.bestUpdated.emit({"cost": 42.0})
        worker.progress.emit(50, "should not arrive")
        QtCore.QCoreApplication.processEvents()
        QtCore.QCoreApplication.processEvents()

        assert best_handler_calls == [], "bestUpdated signal should be disconnected after hard teardown"
        assert progress_handler_calls == [], "progress signal should be disconnected after hard teardown"
    finally:
        window.close()


def test_on_worker_progress_returns_early_when_closing(qt_app):
    """_on_worker_progress must not touch widgets when _closing is True."""
    window = _build_window()
    try:
        set_value_calls: list[int] = []
        set_text_calls: list[str] = []
        window._progress_bar.setValue = lambda v: set_value_calls.append(v)
        window._status_label.setText = lambda t: set_text_calls.append(t)

        window._closing = True
        window._on_worker_progress(50, "should be ignored")

        assert set_value_calls == [], "_progress_bar.setValue called despite _closing=True"
        assert set_text_calls == [], "_status_label.setText called despite _closing=True"
    finally:
        window._closing = False
        window.close()


def test_start_fit_clears_cached_state_before_launch(monkeypatch):
    window = _build_window()
    try:
        window._latest_model_series = {"stale": {"A": np.asarray([1.0], dtype=float)}}
        window._latest_dataset_stats = {"stale": {"chi_squared": 1.0}}
        window._latest_plot_model_series = {"stale": {"A": np.asarray([1.0], dtype=float)}}
        window._latest_plot_model_x = {"stale": np.asarray([0.0], dtype=float)}
        window._best_cost = 9.0
        window._best_effort_failures.add("stale.best_effort")
        window._teardown_disable_failures.add("stale.disable")

        monkeypatch.setattr(
            window._params_ics_tab,
            "_collect_parameter_config",
            lambda: {
                "parameters": {"k": 1.0},
                "bounds": {"k": (0.0, 2.0)},
                "fixed_params": {},
                "method": "trf",
                "max_nfev": 2,
                "seed": None,
                "log10_params": {},
            },
        )
        monkeypatch.setattr(
            window,
            "_collect_dataset_selection",
            lambda: {
                "rows": [{"id": "ds1", "label": "Dataset 1", "species": "A", "include": True, "weight": 1.0}],
                "ids": ["ds1"],
            },
        )
        launch_state: list[tuple[dict, dict]] = []
        monkeypatch.setattr(
            window,
            "_start_global_fit",
            lambda config, dataset_selection, *, solver="Radau", rtol=1e-6, atol=1e-12: launch_state.append(
                (dict(config), dict(dataset_selection))
            ),
        )

        window._start_fit()

        assert launch_state
        assert window._latest_model_series == {}
        assert window._latest_dataset_stats == {}
        assert window._latest_plot_model_series == {}
        assert window._latest_plot_model_x == {}
        assert window._best_cost is None
        assert window._best_effort_failures == set()
        assert window._teardown_disable_failures == set()
    finally:
        window.close()
