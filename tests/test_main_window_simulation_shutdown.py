from __future__ import annotations

import shiboken6
from PySide6 import QtCore, QtGui, QtWidgets
import pytest

pytestmark = pytest.mark.gui


def test_hidden_main_window_close_prepares_simulation_shutdown(qt_app, tmp_path, monkeypatch):
    from kindred.gui.main_window import MainWindow

    templates_dir = tmp_path / "templates"
    monkeypatch.setattr(
        "kindred.config.templates.TemplateManager._get_templates_directory",
        lambda _self: templates_dir,
    )
    monkeypatch.setattr(MainWindow, "_add_to_recent_files", lambda self, path: None)
    main_window = MainWindow()
    calls: list[bool] = []
    monkeypatch.setattr(
        main_window.simulation_controller,
        "prepare_simulation_shutdown_for_close",
        lambda: calls.append(True) or True,
    )

    try:
        assert main_window.isVisible() is False
        main_window.close()
    finally:
        main_window.simulation_controller.shutdown_batch_lane_pool(force_terminate=True)
        main_window.simulation_controller.release_current_simulation_worker()

    assert calls == [True]



class _StubbornSimulationWorker(QtCore.QObject):
    finished = QtCore.Signal()
    progress = QtCore.Signal(int, str)
    result_ready = QtCore.Signal(dict)
    error = QtCore.Signal(object)

    def __init__(self, *, parent: QtCore.QObject | None) -> None:
        super().__init__(parent)
        self._running = True
        self.cancel_called = False
        self.wait_calls: list[int] = []
        self.progress.connect(lambda *_args: None)
        self.result_ready.connect(lambda *_args: None)
        self.error.connect(lambda *_args: None)

    def isRunning(self) -> bool:
        return bool(self._running)

    def cancel(self) -> None:
        self.cancel_called = True

    def wait(self, msecs: int | None = None) -> bool:
        self.wait_calls.append(int(msecs or 0))
        return False


def test_main_window_close_event_defers_for_stubborn_simulation_worker(main_window):
    worker = _StubbornSimulationWorker(parent=main_window.simulation_controller)
    main_window.simulation_controller.run_state.simulation_worker = worker
    try:
        event = QtGui.QCloseEvent()
        main_window.closeEvent(event)

        assert worker.cancel_called is True
        assert worker.wait_calls == []
        assert event.isAccepted() is False
        assert worker in main_window.simulation_controller._retained_simulation_workers
    finally:
        worker._running = False
        worker.finished.emit()


def test_main_window_close_event_retries_cleanup_for_retained_running_worker_after_current_pointer_cleared(main_window):
    worker = _StubbornSimulationWorker(parent=main_window.simulation_controller)
    controller = main_window.simulation_controller
    controller.run_state.simulation_worker = worker
    try:
        controller._release_current_simulation_worker()
        assert controller.run_state.simulation_worker is None
        assert worker in controller._retained_simulation_workers

        worker.cancel_called = False
        worker.wait_calls.clear()

        event = QtGui.QCloseEvent()
        main_window.closeEvent(event)

        assert worker.cancel_called is True
        assert worker.wait_calls == []
        assert event.isAccepted() is False
        assert worker in controller._retained_simulation_workers
    finally:
        worker._running = False
        worker.finished.emit()


def test_main_window_close_event_deferral_keeps_tracked_fit_windows_open(main_window):
    class _TrackedFitWindow(QtWidgets.QDialog):
        def __init__(self, parent: QtWidgets.QWidget | None) -> None:
            super().__init__(parent)
            self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, True)
            self.close_events = 0

        def closeEvent(self, event: QtGui.QCloseEvent) -> None:
            self.close_events += 1
            super().closeEvent(event)

    worker = _StubbornSimulationWorker(parent=main_window.simulation_controller)
    main_window.simulation_controller.run_state.simulation_worker = worker

    fit_window = _TrackedFitWindow(parent=main_window)
    main_window._register_fit_window(fit_window)
    assert fit_window.isVisible() is True
    assert fit_window in list(getattr(main_window, "_active_fit_windows", []) or [])

    try:
        event = QtGui.QCloseEvent()
        main_window.closeEvent(event)

        assert worker.cancel_called is True
        assert worker.wait_calls == []
        assert event.isAccepted() is False
        assert fit_window.close_events == 0
        assert fit_window.isVisible() is True
        assert fit_window in list(getattr(main_window, "_active_fit_windows", []) or [])
    finally:
        fit_window.close()
        worker._running = False
        worker.finished.emit()


def test_main_window_close_event_closes_tracked_fit_windows_before_accepting(main_window, qtbot):
    class _TrackedFitWindow(QtWidgets.QDialog):
        def __init__(self, parent: QtWidgets.QWidget | None) -> None:
            super().__init__(parent)
            self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, True)
            self.close_events = 0

        def closeEvent(self, event: QtGui.QCloseEvent) -> None:
            self.close_events += 1
            super().closeEvent(event)

    fit_window = _TrackedFitWindow(parent=main_window)
    main_window._register_fit_window(fit_window)
    assert fit_window.isVisible() is True
    assert fit_window in list(getattr(main_window, "_active_fit_windows", []) or [])

    event = QtGui.QCloseEvent()
    main_window.closeEvent(event)

    assert event.isAccepted() is True
    assert fit_window.close_events == 1
    qtbot.waitUntil(lambda: not shiboken6.isValid(fit_window), timeout=2000)
    assert list(getattr(main_window, "_active_fit_windows", []) or []) == []
