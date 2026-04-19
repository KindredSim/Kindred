from __future__ import annotations

from contextlib import suppress
from unittest import mock

import pytest
import numpy as np
import shiboken6
from PySide6 import QtCore, QtGui, QtWidgets

from kindred.core.analysis.global_fitting import DatasetFitInfo, GlobalFitResult
from kindred.core.fitting_completion import FitDetailSection, FitDiagnostic, GlobalFitCompletion
from kindred.core.simulation_failure import build_simulation_failure
from kindred.gui.controllers.dataset_manager import DatasetManager
from kindred.gui.fitting.window import FittingWindow, _PROJECT_APPLY_SCOPE_INITIAL_CONDITIONS

pytestmark = pytest.mark.gui


def _process_deferred_deletes(iterations: int = 5) -> None:
    for _ in range(max(1, int(iterations))):
        QtCore.QCoreApplication.processEvents()
    with suppress(RuntimeError, TypeError):
        QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
    for _ in range(max(1, int(iterations))):
        QtCore.QCoreApplication.processEvents()


def _build_completion(
    *,
    status: str,
    optimizer_converged: bool | None = None,
    nonfinite_metrics: bool = False,
    optimizer_diagnostic: FitDiagnostic | None = None,
    dataset_failures: dict[str, FitDiagnostic] | None = None,
    dataset_warnings: dict[str, str] | None = None,
    detail_sections: list[FitDetailSection] | None = None,
) -> GlobalFitCompletion:
    if optimizer_converged is None:
        optimizer_converged = status == "ok"
    return GlobalFitCompletion(
        status=status,
        optimizer_converged=optimizer_converged,
        nonfinite_metrics=nonfinite_metrics,
        optimizer_diagnostic=optimizer_diagnostic,
        dataset_failures=dataset_failures or {},
        dataset_warnings=dataset_warnings or {},
        detail_sections=detail_sections or [],
    )


def _make_diagnostic(
    *,
    phase: str,
    dataset_id: str | None = None,
    message: str,
    stack_trace: str | None = None,
    remediation: str | None = None,
    failure_kind: str = "simulation_error",
    details: dict[str, object] | None = None,
) -> FitDiagnostic:
    return FitDiagnostic(
        phase=phase,
        dataset_id=dataset_id,
        failure=build_simulation_failure(
            kind=failure_kind,
            message=message,
            context={"stack_trace": stack_trace} if stack_trace is not None else None,
            details=details,
        ),
        remediation=remediation,
    )


def _make_detail_section(*, dataset_id: str | None = None, message: str, stack_trace: str) -> FitDetailSection:
    return FitDetailSection(
        dataset_id=dataset_id,
        failure=build_simulation_failure(
            kind="simulation_error",
            message=message,
            context={"stack_trace": stack_trace},
        ),
    )


def _build_success_result(*, dataset_id: str = "ds1", param_name: str = "k", value: float = 1.0) -> GlobalFitResult:
    model = np.asarray([1.0, 0.8, 0.6], dtype=float)
    return GlobalFitResult(
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
        completion=_build_completion(status="ok", optimizer_converged=True),
        covariance=None,
        objective_residuals=np.asarray([0.0], dtype=float),
        model_series={str(dataset_id): {"A": model}},
        residual_series={str(dataset_id): {"A": np.asarray([0.0, 0.0, 0.0], dtype=float)}},
    )


def _build_completion_result(
    *,
    status: str,
    optimizer_converged: bool | None = None,
    dataset_id: str = "ds1",
    param_name: str = "k",
    value: float = 1.0,
    message: str | None = None,
) -> GlobalFitResult:
    result = _build_success_result(dataset_id=dataset_id, param_name=param_name, value=value)
    result.message = str(message or ("ok" if status == "ok" else "failed"))
    completion_kwargs: dict[str, object] = {
        "status": status,
        "optimizer_converged": optimizer_converged,
    }
    if status == "fail":
        completion_kwargs["optimizer_diagnostic"] = _make_diagnostic(
            phase="fatal",
            message=str(message or "failed"),
            remediation="generic_retry",
        )
    result.completion = _build_completion(**completion_kwargs)
    return result


def _build_window(
    *,
    dataset_entries: list[dict[str, object]] | None = None,
    dataset_payloads: list[dict[str, object]] | None = None,
) -> FittingWindow:
    if dataset_entries is None:
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
    if dataset_payloads is None:
        dataset_payloads = [
            {"id": "ds1", "t": np.asarray([0.0, 1.0, 2.0]), "y": np.asarray([1.0, 0.8, 0.6]), "species": "A"}
        ]
    return FittingWindow(
        mode="global",
        parameter_defs=[{"name": "k", "value": 1.0, "min": 0.0, "max": 2.0}],
        dataset_entries=list(dataset_entries),
        dataset_payloads=list(dataset_payloads),
        mechanism_species=["A"],
        simulation_func=lambda _params: {"t": np.asarray([0.0, 1.0, 2.0]), "species": {"A": np.asarray([1.0, 0.8, 0.6])}},
    )


class _DatasetPanelSignal:
    def connect(self, _callback) -> None:
        return None


class _DatasetPanel:
    def __init__(self) -> None:
        self.simulateRequested = _DatasetPanelSignal()


class _DatasetPlotTabs:
    def __init__(self) -> None:
        self.grid_payload = []

    def sync_dataset_tab(self, _name: str, **_dataset_payload):
        return _DatasetPanel()

    def sync_dataset_grid(self, dataset_entries) -> None:
        self.grid_payload = list(dataset_entries)

    def remove_dataset_tab(self, _name: str) -> None:
        return None


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
            completion=_build_completion(status="ok", optimizer_converged=True),
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
            requested_solver="BDF",
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
            requested_solver="BDF",
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
            requested_solver="BDF",
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
            requested_solver="BDF",
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
        requested_solver="BDF",
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
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "exec",
        lambda self: int(QtWidgets.QMessageBox.StandardButton.Ok),
    )

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
                requested_solver="BDF",
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
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "exec",
        lambda self: int(QtWidgets.QMessageBox.StandardButton.Ok),
    )

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
            requested_solver="BDF",
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

        def _capture_exec(self):
            states.append(
                (
                    window._pending_best_timer.isActive(),
                    window._pending_best_payload,
                    window._pending_best_worker,
                )
            )
            return int(QtWidgets.QMessageBox.StandardButton.Ok)

        monkeypatch.setattr(QtWidgets.QMessageBox, "exec", _capture_exec)

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
        captured = {"details": None}

        def _capture_warning(*_args, **_kwargs):
            states.append(
                (
                    window._pending_best_timer.isActive(),
                    window._pending_best_payload,
                    window._pending_best_worker,
                )
            )
            captured["details"] = None
            return int(QtWidgets.QMessageBox.StandardButton.Ok)

        def _capture_exec(self):
            states.append(
                (
                    window._pending_best_timer.isActive(),
                    window._pending_best_payload,
                    window._pending_best_worker,
                )
            )
            captured["details"] = self.detailedText()
            return int(QtWidgets.QMessageBox.StandardButton.Ok)

        monkeypatch.setattr(QtWidgets.QMessageBox, "warning", staticmethod(_capture_warning))
        monkeypatch.setattr(QtWidgets.QMessageBox, "exec", _capture_exec)

        window._on_worker_error({"kind": "fitting_error", "message": "boom"}, worker=worker)

        assert states == [(False, None, None)]
        assert captured["details"] == ""
    finally:
        window.close()


def test_worker_error_logs_traceback_and_populates_dialog_details(qt_app, monkeypatch, caplog):
    window = _build_window()
    try:
        window.show()
        qt_app.processEvents()
        worker = _SignalWorker()
        worker._running = False
        window._worker = worker
        dialogs: list[dict[str, object]] = []

        def _unexpected_warning(*_args, **_kwargs):
            raise AssertionError("worker errors should use an instance-based warning dialog")

        def _capture_exec(self):
            dialogs.append(
                {
                    "text": self.text(),
                    "details": self.detailedText(),
                    "title": self.windowTitle(),
                    "icon": self.icon(),
                }
            )
            return int(QtWidgets.QMessageBox.StandardButton.Ok)

        monkeypatch.setattr(QtWidgets.QMessageBox, "warning", staticmethod(_unexpected_warning))
        monkeypatch.setattr(QtWidgets.QMessageBox, "exec", _capture_exec)

        stack_trace = "Traceback line 1\nTraceback line 2"
        with caplog.at_level("WARNING", logger="kindred.gui.fitting.window"):
            window._on_worker_error(
                {
                    "kind": "fitting_error",
                    "message": "boom",
                    "context": {"stack_trace": stack_trace},
                },
                worker=worker,
            )

        assert dialogs == [
            {
                "text": "boom",
                "details": stack_trace,
                "title": "Fitting",
                "icon": QtWidgets.QMessageBox.Icon.Warning,
            }
        ]
        messages = [record.getMessage() for record in caplog.records if record.name == "kindred.gui.fitting.window"]
        assert any(message.startswith("Fitting worker reported error:") for message in messages)
        assert stack_trace in messages
    finally:
        window.close()


def test_completion_dialog_uses_instance_message_box_with_error_diagnostics_details(qt_app, monkeypatch):
    window = _build_window()
    try:
        window.show()
        qt_app.processEvents()
        result = _build_success_result()
        dialogs: list[dict[str, object]] = []

        def _unexpected_information(*_args, **_kwargs):
            raise AssertionError("completion dialog should use an instance-based message box")

        def _unexpected_warning(*_args, **_kwargs):
            raise AssertionError("completion dialog should use an instance-based message box")

        def _capture_exec(self):
            dialogs.append(
                {
                    "title": self.windowTitle(),
                    "text": self.text(),
                    "details": self.detailedText(),
                    "icon": self.icon(),
                }
            )
            return int(QtWidgets.QMessageBox.StandardButton.Ok)

        monkeypatch.setattr(QtWidgets.QMessageBox, "information", staticmethod(_unexpected_information))
        monkeypatch.setattr(QtWidgets.QMessageBox, "warning", staticmethod(_unexpected_warning))
        monkeypatch.setattr(QtWidgets.QMessageBox, "exec", _capture_exec)

        window._handle_global_fit_complete({"result": result})

        assert dialogs == [
            {
                "title": "Optimization Complete",
                "text": mock.ANY,
                "details": "",
                "icon": QtWidgets.QMessageBox.Icon.Information,
            }
        ]
    finally:
        window.close()


def test_completion_dialog_hides_detail_pane_when_no_detail_text_exists(qt_app, monkeypatch):
    window = _build_window()
    try:
        window.show()
        qt_app.processEvents()
        dialogs: list[str] = []

        def _capture_exec(self):
            dialogs.append(self.detailedText())
            return int(QtWidgets.QMessageBox.StandardButton.Ok)

        monkeypatch.setattr(
            QtWidgets.QMessageBox,
            "information",
            staticmethod(lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("unexpected static dialog"))),
        )
        monkeypatch.setattr(
            QtWidgets.QMessageBox,
            "warning",
            staticmethod(lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("unexpected static dialog"))),
        )
        monkeypatch.setattr(QtWidgets.QMessageBox, "exec", _capture_exec)

        window._handle_global_fit_complete({"result": _build_success_result()})

        assert dialogs == [""]
    finally:
        window.close()


def test_completion_dialog_aggregates_per_dataset_details(qt_app, monkeypatch):
    window = _build_window()
    try:
        window._dataset_entries.append(
            {
                "id": "ds2",
                "label": "Dataset 2",
                "t": np.asarray([0.0, 1.0, 2.0], dtype=float),
                "species_data": {"A": np.asarray([1.0, 0.8, 0.6], dtype=float)},
                "selected_species": ["A"],
                "weight": 1.0,
                "include": True,
            }
        )
        window.show()
        qt_app.processEvents()
        result = _build_completion_result(status="fail", message="failed")
        result.completion = _build_completion(
            status="fail",
            optimizer_converged=True,
            dataset_failures={
                "ds1": _make_diagnostic(phase="final_replay", dataset_id="ds1", message="first", stack_trace="trace one"),
                "ds2": _make_diagnostic(phase="final_replay", dataset_id="ds2", message="second", stack_trace="trace two"),
            },
            detail_sections=[
                _make_detail_section(dataset_id="ds1", message="first", stack_trace="trace one"),
                _make_detail_section(dataset_id="ds2", message="second", stack_trace="trace two"),
            ],
        )
        dialogs: list[str] = []

        def _capture_exec(self):
            dialogs.append(self.detailedText())
            return int(QtWidgets.QMessageBox.StandardButton.Ok)

        monkeypatch.setattr(
            QtWidgets.QMessageBox,
            "information",
            staticmethod(lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("unexpected static dialog"))),
        )
        monkeypatch.setattr(
            QtWidgets.QMessageBox,
            "warning",
            staticmethod(lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("unexpected static dialog"))),
        )
        monkeypatch.setattr(QtWidgets.QMessageBox, "exec", _capture_exec)

        window._handle_global_fit_complete({"result": result})

        assert dialogs
        detail_text = dialogs[0]
        assert "Dataset 1" in detail_text
        assert "trace one" in detail_text
        assert "Dataset 2" in detail_text
        assert "trace two" in detail_text
        assert "suppressed top trace" not in detail_text
        assert "\n\n---\n\n" in detail_text
    finally:
        window.close()


def test_completion_dialog_does_not_log_suppressed_top_level_detail_text_on_success(qt_app, monkeypatch, caplog):
    window = _build_window()
    try:
        window.show()
        qt_app.processEvents()
        result = _build_success_result()

        monkeypatch.setattr(
            QtWidgets.QMessageBox,
            "information",
            staticmethod(lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("unexpected static dialog"))),
        )
        monkeypatch.setattr(
            QtWidgets.QMessageBox,
            "warning",
            staticmethod(lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("unexpected static dialog"))),
        )
        monkeypatch.setattr(
            QtWidgets.QMessageBox,
            "exec",
            lambda self: int(QtWidgets.QMessageBox.StandardButton.Ok),
        )

        with caplog.at_level("WARNING", logger="kindred.gui.fitting.window"):
            window._handle_global_fit_complete({"result": result})

        messages = [record.getMessage() for record in caplog.records if record.name == "kindred.gui.fitting.window"]
        assert not messages
    finally:
        window.close()


def test_successful_completion_suppresses_top_level_error_diagnostics_logging_and_details(qt_app, monkeypatch):
    window = _build_window()
    try:
        window.show()
        qt_app.processEvents()
        result = _build_success_result()

        warning_mock = mock.Mock()
        dialogs: list[str] = []

        def _capture_exec(self):
            dialogs.append(self.detailedText())
            return int(QtWidgets.QMessageBox.StandardButton.Ok)

        monkeypatch.setattr("kindred.gui.fitting.window.logger.warning", warning_mock)
        monkeypatch.setattr(QtWidgets.QMessageBox, "exec", _capture_exec)

        window._handle_global_fit_complete({"result": result})

        assert dialogs == [""]
        warning_mock.assert_not_called()
    finally:
        window.close()


def test_warning_completion_labels_top_level_error_diagnostics_when_dataset_error_entry_missing(qt_app, monkeypatch):
    window = _build_window()
    try:
        window._dataset_entries.append(
            {
                "id": "ds_x",
                "label": "Dataset X",
                "t": np.asarray([0.0, 1.0, 2.0], dtype=float),
                "species_data": {"A": np.asarray([1.0, 0.8, 0.6], dtype=float)},
                "selected_species": ["A"],
                "weight": 1.0,
                "include": True,
            }
        )
        window.show()
        qt_app.processEvents()
        result = _build_completion_result(status="warn", optimizer_converged=False)
        result.completion = _build_completion(
            status="warn",
            optimizer_converged=False,
            optimizer_diagnostic=_make_diagnostic(
                phase="optimizer",
                dataset_id="ds_x",
                message="orphan failure",
                stack_trace="orphan trace",
            ),
            detail_sections=[
                _make_detail_section(dataset_id="ds_x", message="orphan failure", stack_trace="orphan trace"),
            ],
        )
        dialogs: list[dict[str, str]] = []

        def _capture_exec(self):
            dialogs.append({"title": self.windowTitle(), "details": self.detailedText()})
            return int(QtWidgets.QMessageBox.StandardButton.Ok)

        monkeypatch.setattr(QtWidgets.QMessageBox, "exec", _capture_exec)

        window._handle_global_fit_complete({"result": result})

        assert dialogs == [
            {
                "title": "Optimization Complete (Warnings)",
                "details": "Dataset X\norphan trace",
            }
        ]
    finally:
        window.close()


def test_warning_completion_surfaces_optimizer_diagnostic_message_in_dialog_body(qt_app, monkeypatch):
    window = _build_window()
    try:
        window.show()
        qt_app.processEvents()
        result = _build_completion_result(status="warn", optimizer_converged=False)
        result.completion = _build_completion(
            status="warn",
            optimizer_converged=False,
            optimizer_diagnostic=_make_diagnostic(
                phase="optimizer",
                dataset_id="ds1",
                message="optimizer step rejected",
            ),
        )
        dialogs: list[dict[str, str]] = []

        def _capture_exec(self):
            dialogs.append(
                {
                    "title": self.windowTitle(),
                    "text": self.text(),
                    "details": self.detailedText(),
                }
            )
            return int(QtWidgets.QMessageBox.StandardButton.Ok)

        monkeypatch.setattr(QtWidgets.QMessageBox, "exec", _capture_exec)

        window._handle_global_fit_complete({"result": result})

        assert dialogs == [
            {
                "title": "Optimization Complete (Warnings)",
                "text": (
                    "Final Chi-Squared (\u03c7\u00b2): 1\n\n"
                    "Fitted Parameters:\n"
                    "  k = 1\n\n"
                    "Warnings:\n"
                    "- Optimizer did not report convergence; results may be suboptimal.\n"
                    "- optimizer step rejected"
                ),
                "details": "",
            }
        ]
    finally:
        window.close()


def test_failed_completion_preparation_fatal_uses_preparation_remediation_not_x_axis(qt_app):
    window = _build_window()
    try:
        qt_app.processEvents()
        result = _build_completion_result(status="fail", message="failed")
        result.completion = _build_completion(
            status="fail",
            optimizer_converged=False,
            optimizer_diagnostic=_make_diagnostic(
                phase="fatal",
                message="undefined symbol k_total",
                remediation="preparation",
                failure_kind="preparation_error",
                details={"stage": "parameter_algebra"},
            ),
        )

        severity, title, text = window._global_fit_completion_dialog_spec(result)

        assert severity == "fail"
        assert title == "Global Fit Failed"
        assert "parameter algebra failed" in text.lower()
        assert "fix the preparation or parameter algebra error" in text.lower()
        assert "adjust t_min/t_max" not in text.lower()
    finally:
        window.close()


def test_failed_completion_alignment_failure_keeps_x_axis_remediation(qt_app):
    window = _build_window()
    try:
        qt_app.processEvents()
        result = _build_completion_result(status="fail", message="failed")
        result.completion = _build_completion(
            status="fail",
            optimizer_converged=True,
            dataset_failures={
                "ds1": _make_diagnostic(
                    phase="final_replay",
                    dataset_id="ds1",
                    message="Dataset 'ds1': observed X values fall outside model range. Adjust t_min/t_max.",
                    remediation="x_axis_mapping",
                ),
            },
        )

        severity, _title, text = window._global_fit_completion_dialog_spec(result)

        assert severity == "fail"
        assert "adjust t_min/t_max" in text.lower()
        assert "fix x axis / mapping" in text.lower()
    finally:
        window.close()


def test_failed_completion_nonfinite_metrics_shows_nonfinite_message_not_x_axis(qt_app):
    window = _build_window()
    try:
        qt_app.processEvents()
        result = _build_completion_result(status="fail", message="failed")
        result.completion = _build_completion(
            status="fail",
            optimizer_converged=True,
            nonfinite_metrics=True,
            optimizer_diagnostic=_make_diagnostic(
                phase="fatal",
                message="Final χ² is non-finite; results are invalid.",
                remediation="nonfinite_metrics",
            ),
        )

        severity, _title, text = window._global_fit_completion_dialog_spec(result)

        assert severity == "fail"
        assert "final χ² is non-finite; results are invalid." in text.lower()
        assert "inspect the fit objective and inputs for non-finite values" in text.lower()
        assert "fix x axis / mapping" not in text.lower()
    finally:
        window.close()


def test_failed_completion_dataset_failures_own_fail_body_summary_when_top_level_exists(qt_app):
    window = _build_window()
    try:
        qt_app.processEvents()
        result = _build_completion_result(status="fail", message="failed")
        result.completion = _build_completion(
            status="fail",
            optimizer_converged=True,
            optimizer_diagnostic=_make_diagnostic(
                phase="fatal",
                message="top-level process-pool failure",
                remediation="generic_retry",
                stack_trace="top trace",
            ),
            dataset_failures={
                "ds1": _make_diagnostic(
                    phase="final_replay",
                    dataset_id="ds1",
                    message="dataset replay failed",
                    remediation="generic_retry",
                ),
            },
            detail_sections=[
                _make_detail_section(message="top-level process-pool failure", stack_trace="top trace"),
            ],
        )

        severity, _title, text = window._global_fit_completion_dialog_spec(result)

        assert severity == "fail"
        assert "- Dataset 1: dataset replay failed" in text
        assert "top-level process-pool failure" not in text
    finally:
        window.close()


def test_failed_completion_does_not_leave_project_apply_scopes_enabled(qt_app, monkeypatch):
    window = _build_window()
    try:
        window.show()
        qt_app.processEvents()
        result = _build_completion_result(
            status="fail",
            message="Optimization terminated successfully.",
            value=0.5,
        )

        monkeypatch.setattr(
            QtWidgets.QMessageBox,
            "exec",
            lambda self: int(QtWidgets.QMessageBox.StandardButton.Ok),
        )

        window._handle_global_fit_complete({"result": result})

        assert window._params_ics_tab.get_last_fit_params() == {}
        assert window._available_project_apply_scopes() == set()
        assert not window._apply_to_project_button.isEnabled()
    finally:
        window.close()


def test_failed_completion_keeps_user_edited_value_cells_after_clearing_fit_authority(qt_app, monkeypatch):
    window = _build_window()
    try:
        window.show()
        qt_app.processEvents()
        window._params_ics_tab.set_last_fit_params({"k": 0.5})
        parameter_state = window._params_ics_tab.get_parameter_state()
        parameter_state[0]["value"] = 1.7
        parameter_state[0]["last_fit"] = 0.5
        window._params_ics_tab.set_parameter_state(parameter_state)
        window._params_ics_tab._populate_parameter_table()

        monkeypatch.setattr(
            QtWidgets.QMessageBox,
            "exec",
            lambda self: int(QtWidgets.QMessageBox.StandardButton.Ok),
        )

        result = _build_completion_result(
            status="fail",
            message="Optimization terminated successfully.",
            value=0.5,
        )
        result.model_series = {}
        result.dataset_info = []

        window._handle_global_fit_complete({"result": result})

        assert window._params_ics_tab.get_last_fit_params() == {}
        assert window._params_ics_tab._param_table.item(0, 3).text() == "1.7"
        assert window._params_ics_tab.get_parameter_state()[0]["last_fit"] is None
    finally:
        window.close()
        qt_app.processEvents()


def test_failed_completion_restores_pre_run_value_cells_after_live_best_update(qt_app, monkeypatch):
    window = _build_window()
    try:
        window.show()
        qt_app.processEvents()
        window._pre_run_parameter_state = window._params_ics_tab.get_parameter_state()
        window._params_ics_tab.push_best_update({"k": 0.9}, {})
        assert window._params_ics_tab._param_table.item(0, 3).text() == "0.9"

        monkeypatch.setattr(
            QtWidgets.QMessageBox,
            "exec",
            lambda self: int(QtWidgets.QMessageBox.StandardButton.Ok),
        )

        result = _build_completion_result(
            status="fail",
            message="Optimization terminated successfully.",
            value=0.5,
        )
        result.model_series = {}
        result.dataset_info = []

        window._handle_global_fit_complete({"result": result})

        assert window._params_ics_tab.get_last_fit_params() == {}
        assert window._params_ics_tab._param_table.item(0, 3).text() == "1"
        assert window._params_ics_tab.get_parameter_state()[0]["last_fit"] is None
    finally:
        window.close()
        qt_app.processEvents()


def test_failed_completion_restores_pre_run_staged_dataset_params(qt_app, monkeypatch):
    window = _build_window()
    try:
        window.show()
        qt_app.processEvents()
        window._params_ics_tab.set_staged_dataset_params({"ds1": {"init:A": 2.5}})
        window._pre_run_parameter_state = window._params_ics_tab.get_parameter_state()
        window._pre_run_staged_dataset_params = window._params_ics_tab.get_staged_dataset_params()
        window._params_ics_tab.push_best_update({"k": 0.9}, {"ds1": {"init:A": 0.4}})

        monkeypatch.setattr(
            QtWidgets.QMessageBox,
            "exec",
            lambda self: int(QtWidgets.QMessageBox.StandardButton.Ok),
        )

        result = _build_completion_result(
            status="fail",
            message="Optimization terminated successfully.",
            value=0.5,
        )
        result.model_series = {}
        result.dataset_info = []

        window._handle_global_fit_complete({"result": result})

        assert window._params_ics_tab.get_staged_dataset_params() == {"ds1": {"init:A": 2.5}}
        assert window._staged_initial_condition_parameters() == {"ds1": {"init:A": 2.5}}
        assert _PROJECT_APPLY_SCOPE_INITIAL_CONDITIONS in window._available_project_apply_scopes()
    finally:
        window.close()
        qt_app.processEvents()


def test_start_fit_launch_failure_after_mechanism_refresh_preserves_refreshed_parameter_rows(qt_app, monkeypatch):
    old_mechanism = "\n".join(
        [
            "reaction: A -> B; k=0.2",
            "initial: A=1.0",
            "initial: B=0.0",
        ]
    )
    refreshed_mechanism = "\n".join(
        [
            "reaction: A -> C; k=0.4",
            "initial: A=1.0",
            "initial: C=0.0",
        ]
    )

    class _PreparedEvaluator:
        def __init__(self, mechanism_text: str, param_names: list[str], *, solver: str, rtol: float, atol: float) -> None:
            self.prepared_metadata = {
                "version": 1,
                "mechanism_text_sha256": FittingWindow._mechanism_text_sha256(mechanism_text),
                "mechanism_text_len": len(mechanism_text),
                "param_names": list(param_names),
                "t_end": 2.0,
                "num_points": 3,
                "temperature_K": 298.15,
                "solver_requested": solver,
                "solver_normalized": solver,
                "solver_warning": None,
                "rtol": float(rtol),
                "atol": float(atol),
                "use_sparse_jacobian": False,
                "wegscheider_cyclicity_enabled": False,
                "initial_prefix": "init:",
            }

        def __call__(self, _params):
            return {
                "t": np.asarray([0.0, 1.0, 2.0], dtype=float),
                "species": {"A": np.asarray([1.0, 0.8, 0.6], dtype=float)},
            }

        def with_fixed_params(self, _fixed_params):
            return self

    class _DatasetManagerStub:
        @staticmethod
        def scan_mechanism_parameters(mechanism_text: str) -> list[dict[str, object]]:
            if "A -> C" in str(mechanism_text):
                return [{"name": "k2", "value": 0.4, "min": 0.0, "max": 1.0}]
            return [{"name": "k1", "value": 0.2, "min": 0.0, "max": 1.0}]

        @staticmethod
        def sync_fit_result_views(_model_series, *, dataset_stats=None, dataset_ids=None) -> None:
            return None

    window = FittingWindow(
        mode="global",
        parameter_defs=[{"name": "k1", "value": 0.2, "min": 0.0, "max": 1.0}],
        dataset_entries=[
            {
                "id": "ds1",
                "label": "Dataset 1",
                "t": np.asarray([0.0, 1.0, 2.0], dtype=float),
                "species_data": {"A": np.asarray([1.0, 0.8, 0.6], dtype=float)},
                "selected_species": ["A"],
                "weight": 1.0,
                "include": True,
            }
        ],
        dataset_payloads=[
            {"id": "ds1", "t": np.asarray([0.0, 1.0, 2.0]), "y": np.asarray([1.0, 0.8, 0.6]), "species": "A"}
        ],
        mechanism_species=["A", "B"],
        dataset_manager=_DatasetManagerStub(),
        simulation_func=_PreparedEvaluator(old_mechanism, ["k1"], solver="BDF", rtol=1e-6, atol=1e-12),
        simulation_builder=lambda mechanism_text, param_names, *, solver, rtol, atol: _PreparedEvaluator(
            mechanism_text,
            list(param_names),
            solver=solver,
            rtol=rtol,
            atol=atol,
        ),
        mechanism_text_getter=lambda: refreshed_mechanism,
        reactions_text_getter=lambda: refreshed_mechanism,
    )
    try:
        window.show()
        qt_app.processEvents()
        assert [str(entry.get("param_name") or "") for entry in window._params_ics_tab.get_parameter_state()] == ["k1"]

        monkeypatch.setattr(
            QtWidgets.QMessageBox,
            "warning",
            lambda *_args, **_kwargs: int(QtWidgets.QMessageBox.StandardButton.Ok),
        )
        monkeypatch.setattr(
            window,
            "_start_global_fit_worker",
            lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("launch boom")),
        )

        config = window._params_ics_tab._collect_parameter_config()
        assert config is not None
        selection = window._collect_dataset_selection()

        window._start_global_fit(config, selection, solver="BDF", rtol=1e-6, atol=1e-12)

        parameter_names = [str(entry.get("param_name") or "") for entry in window._params_ics_tab.get_parameter_state()]
        assert parameter_names == ["k2"]
        assert window._params_ics_tab._param_table.item(0, 2).text() == "k2"
        assert window._params_ics_tab.get_parameter_state()[0]["last_fit"] is None
        assert window._params_ics_tab.get_mechanism_species() == ["A", "C"]
    finally:
        window.close()
        qt_app.processEvents()


def test_failed_completion_clears_dataset_manager_fit_state_after_prior_success(qt_app, monkeypatch):
    window = _build_window()
    try:
        window.show()
        qt_app.processEvents()
        dataset_entry = window._dataset_entries[0]
        dataset = {
            "t": np.asarray(dataset_entry["t"], dtype=float),
            "species": {
                str(name): np.asarray(values, dtype=float)
                for name, values in dict(dataset_entry["species_data"]).items()
            },
        }
        plot_tabs = _DatasetPlotTabs()
        window._dataset_manager = DatasetManager(
            plot_tabs=plot_tabs,
            dataset_resolver=lambda name: dataset if name == "ds1" else None,
        )

        monkeypatch.setattr(
            QtWidgets.QMessageBox,
            "exec",
            lambda self: int(QtWidgets.QMessageBox.StandardButton.Ok),
        )

        window._handle_global_fit_complete({"result": _build_success_result()})
        ds_view = window._dataset_manager._dataset_views["ds1"]
        assert ds_view["model_series"] is not None
        assert ds_view["chi_squared"] == pytest.approx(1.0)

        window._active_fit_dataset_ids = ["ds1"]
        failed = _build_completion_result(
            status="fail",
            message="Optimization terminated successfully.",
            value=0.5,
        )
        failed.model_series = {}
        failed.dataset_info = []

        window._handle_global_fit_complete({"result": failed})

        ds_view = window._dataset_manager._dataset_views["ds1"]
        assert ds_view["model_series"] is None
        assert ds_view["model_x"] is None
        assert ds_view["model_y"] is None
        assert ds_view["chi_squared"] is None
        assert ds_view["r_squared"] is None
    finally:
        window.close()
        qt_app.processEvents()


def test_failed_completion_clears_results_summary_state_after_prior_success(qt_app, monkeypatch):
    window = _build_window()
    try:
        window.show()
        qt_app.processEvents()
        monkeypatch.setattr(
            QtWidgets.QMessageBox,
            "exec",
            lambda self: int(QtWidgets.QMessageBox.StandardButton.Ok),
        )

        window._run_results_tab.set_run_stamp({"solver": "BDF"}, "hash123", "hash123")
        window._results_summary_button.setEnabled(True)
        assert window._results_summary_button.isEnabled()
        assert window._run_results_tab._last_run_stamp == {"solver": "BDF"}

        failed = _build_completion_result(
            status="fail",
            message="Optimization terminated successfully.",
            value=0.5,
        )
        failed.model_series = {}
        failed.dataset_info = []

        window._handle_global_fit_complete({"result": failed})

        assert not window._results_summary_button.isEnabled()
        assert window._run_results_tab._last_run_stamp == {}
        assert window._run_results_tab._last_run_stamp_hash == ""
        assert window._run_results_tab._last_run_stamp_short == ""
    finally:
        window.close()
        qt_app.processEvents()


def test_failed_completion_with_pending_rebuild_uses_current_applied_targets(qt_app, monkeypatch):
    dataset_entries = [
        {
            "id": "ds1",
            "label": "Dataset 1",
            "t": np.asarray([0.0, 1.0, 2.0], dtype=float),
            "species_data": {
                "A": np.asarray([1.0, 0.8, 0.6], dtype=float),
                "B": np.asarray([0.2, 0.3, 0.4], dtype=float),
            },
            "selected_species": ["A"],
            "weight": 1.0,
            "include": True,
        }
    ]
    dataset_payloads = [
        {
            "id": "ds1",
            "t": np.asarray([0.0, 1.0, 2.0], dtype=float),
            "y": np.asarray(
                [
                    [1.0, 0.8, 0.6],
                    [0.2, 0.3, 0.4],
                ],
                dtype=float,
            ),
            "species": ["A", "B"],
        }
    ]
    window = _build_window(dataset_entries=dataset_entries, dataset_payloads=dataset_payloads)
    try:
        window.show()
        qt_app.processEvents()
        assert window._run_results_tab._fit_targets_by_dataset["ds1"] == ["A"]

        worker = _SignalWorker()
        window._worker = worker
        window._species_table._fit_targets_selection_applied["ds1"] = ["B"]
        window._on_targets_applied()
        qt_app.processEvents()
        assert window._results_rebuild_pending is True

        monkeypatch.setattr(
            QtWidgets.QMessageBox,
            "exec",
            lambda self: int(QtWidgets.QMessageBox.StandardButton.Ok),
        )

        window.hide()
        window._worker = None
        failed = _build_completion_result(
            status="fail",
            dataset_id="ds1",
            message="Optimization terminated successfully.",
            value=0.5,
        )
        failed.model_series = {}
        failed.dataset_info = []

        window._handle_global_fit_complete({"result": failed})

        assert window._results_rebuild_pending is False
        assert window._run_results_tab._fit_targets_by_dataset["ds1"] == ["B"]
        payload = window._run_results_tab._dataset_plot_views["ds1"]._datasets[0]
        assert payload["current_species"] == "B"
        assert sorted(payload["all_species"].keys()) == ["B"]
    finally:
        window.close()
        qt_app.processEvents()


def test_start_fit_failure_clears_prior_fit_state_before_worker_launch(qt_app, monkeypatch):
    window = _build_window()
    try:
        window.show()
        qt_app.processEvents()
        dataset_entry = window._dataset_entries[0]
        dataset = {
            "t": np.asarray(dataset_entry["t"], dtype=float),
            "species": {
                str(name): np.asarray(values, dtype=float)
                for name, values in dict(dataset_entry["species_data"]).items()
            },
        }
        plot_tabs = _DatasetPlotTabs()
        window._dataset_manager = DatasetManager(
            plot_tabs=plot_tabs,
            dataset_resolver=lambda name: dataset if name == "ds1" else None,
        )
        monkeypatch.setattr(
            QtWidgets.QMessageBox,
            "exec",
            lambda self: int(QtWidgets.QMessageBox.StandardButton.Ok),
        )
        window._handle_global_fit_complete({"result": _build_success_result()})

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
        monkeypatch.setattr(window._params_ics_tab, "collect_integration_settings", lambda: ("BDF", 1e-6, 1e-12))
        monkeypatch.setattr(window, "_datasets_payloads_for_run", lambda _ids: None)

        window._start_fit()

        ds_view = window._dataset_manager._dataset_views["ds1"]
        assert ds_view["model_series"] is None
        assert ds_view["chi_squared"] is None
        assert window._available_project_apply_scopes() == set()
    finally:
        window.close()
        qt_app.processEvents()


def test_start_global_fit_unavailable_callback_clears_prior_dataset_manager_fit_state(qt_app, monkeypatch):
    window = _build_window()
    try:
        window.show()
        qt_app.processEvents()
        dataset_entry = window._dataset_entries[0]
        dataset = {
            "t": np.asarray(dataset_entry["t"], dtype=float),
            "species": {
                str(name): np.asarray(values, dtype=float)
                for name, values in dict(dataset_entry["species_data"]).items()
            },
        }
        plot_tabs = _DatasetPlotTabs()
        window._dataset_manager = DatasetManager(
            plot_tabs=plot_tabs,
            dataset_resolver=lambda name: dataset if name == "ds1" else None,
        )

        monkeypatch.setattr(
            QtWidgets.QMessageBox,
            "exec",
            lambda self: int(QtWidgets.QMessageBox.StandardButton.Ok),
        )
        monkeypatch.setattr(
            QtWidgets.QMessageBox,
            "warning",
            lambda *_args, **_kwargs: int(QtWidgets.QMessageBox.StandardButton.Ok),
        )

        window._handle_global_fit_complete({"result": _build_success_result()})
        ds_view = window._dataset_manager._dataset_views["ds1"]
        assert ds_view["model_series"] is not None
        assert ds_view["chi_squared"] == pytest.approx(1.0)

        window._simulation_func = None
        window._simulation_builder = None

        config = window._params_ics_tab._collect_parameter_config()
        assert config is not None
        selection = window._collect_dataset_selection()

        window._start_global_fit(config, selection)

        ds_view = window._dataset_manager._dataset_views["ds1"]
        assert ds_view["model_series"] is None
        assert ds_view["model_x"] is None
        assert ds_view["model_y"] is None
        assert ds_view["chi_squared"] is None
        assert ds_view["r_squared"] is None
    finally:
        window.close()
        qt_app.processEvents()


def test_start_global_fit_unavailable_callback_clears_open_results_summary_state(qt_app, monkeypatch):
    window = _build_window()
    try:
        window.show()
        qt_app.processEvents()
        window._run_results_tab.set_run_stamp({"solver": "BDF"}, "hash123", "hash123")
        window._run_results_tab.update_statistics({"Datasets": 1})
        window._results_summary_button.setEnabled(True)
        window._run_results_tab.open_results_summary_dialog()
        qt_app.processEvents()
        assert window._run_results_tab._stamp_dialog is not None
        assert window._run_results_tab._stamp_dialog.isVisible()

        refresh_calls: list[tuple] = []
        original_refresh = window._run_results_tab._stamp_dialog.refresh

        def tracking_refresh(*args, **kwargs):
            refresh_calls.append((args, kwargs))
            return original_refresh(*args, **kwargs)

        monkeypatch.setattr(window._run_results_tab._stamp_dialog, "refresh", tracking_refresh)
        monkeypatch.setattr(
            QtWidgets.QMessageBox,
            "warning",
            lambda *_args, **_kwargs: int(QtWidgets.QMessageBox.StandardButton.Ok),
        )

        window._simulation_func = None
        window._simulation_builder = None

        config = window._params_ics_tab._collect_parameter_config()
        assert config is not None
        selection = window._collect_dataset_selection()

        window._start_global_fit(config, selection)

        assert not window._results_summary_button.isEnabled()
        assert window._run_results_tab._last_run_stamp == {}
        assert refresh_calls
        args, kwargs = refresh_calls[-1]
        assert args[0] == {}
        assert args[1] == ""
        assert args[2] == ""
        assert args[3] is None
        assert kwargs["fitted_params"] is None
        assert kwargs["dataset_fitted_params"] is None
    finally:
        if getattr(window._run_results_tab, "_stamp_dialog", None) is not None:
            window._run_results_tab._stamp_dialog.close()
        window.close()
        qt_app.processEvents()


def test_failed_completion_clears_only_active_run_dataset_fit_state_even_with_partial_failed_result(
    qt_app, monkeypatch
):
    dataset_entries = [
        {
            "id": "ds1",
            "label": "Dataset 1",
            "t": np.asarray([0.0, 1.0, 2.0], dtype=float),
            "species_data": {"A": np.asarray([1.0, 0.8, 0.6], dtype=float)},
            "selected_species": ["A"],
            "weight": 1.0,
            "include": True,
        },
        {
            "id": "ds2",
            "label": "Dataset 2",
            "t": np.asarray([0.0, 1.0, 2.0], dtype=float),
            "species_data": {"A": np.asarray([0.9, 0.7, 0.5], dtype=float)},
            "selected_species": ["A"],
            "weight": 1.0,
            "include": False,
        },
    ]
    dataset_payloads = [
        {"id": "ds1", "t": np.asarray([0.0, 1.0, 2.0]), "y": np.asarray([1.0, 0.8, 0.6]), "species": "A"},
        {"id": "ds2", "t": np.asarray([0.0, 1.0, 2.0]), "y": np.asarray([0.9, 0.7, 0.5]), "species": "A"},
    ]
    window = _build_window(dataset_entries=dataset_entries, dataset_payloads=dataset_payloads)
    try:
        window.show()
        qt_app.processEvents()
        datasets = {
            "ds1": {"t": np.asarray([0.0, 1.0, 2.0], dtype=float), "species": {"A": np.asarray([1.0, 0.8, 0.6], dtype=float)}},
            "ds2": {"t": np.asarray([0.0, 1.0, 2.0], dtype=float), "species": {"A": np.asarray([0.9, 0.7, 0.5], dtype=float)}},
        }
        plot_tabs = _DatasetPlotTabs()
        window._dataset_manager = DatasetManager(
            plot_tabs=plot_tabs,
            dataset_resolver=lambda name: datasets.get(str(name)),
        )
        window._dataset_manager.sync_fit_result_views(
            {
                "ds1": {"A": np.asarray([0.95, 0.75, 0.55], dtype=float)},
                "ds2": {"A": np.asarray([0.85, 0.65, 0.45], dtype=float)},
            },
            dataset_stats={
                "ds1": {"chi_squared": 1.0, "r_squared": 0.9},
                "ds2": {"chi_squared": 2.0, "r_squared": 0.8},
            },
            dataset_ids=["ds1", "ds2"],
        )

        monkeypatch.setattr(
            QtWidgets.QMessageBox,
            "exec",
            lambda self: int(QtWidgets.QMessageBox.StandardButton.Ok),
        )

        window._active_fit_dataset_ids = ["ds1"]
        failed = _build_completion_result(
            status="fail",
            dataset_id="ds1",
            message="Optimization terminated successfully.",
            value=0.5,
        )
        failed.model_series = {"ds1": {"A": np.asarray([0.5, 0.4, 0.3], dtype=float)}}
        failed.dataset_info = [
            DatasetFitInfo(
                dataset_id="ds1",
                r_squared=0.12,
                chi_squared=9.9,
                rmse=1.0,
                mae=1.0,
                residuals=np.asarray([1.0], dtype=float),
                n_points=1,
                weight=1.0,
            )
        ]

        window._handle_global_fit_complete({"result": failed})

        ds1_view = window._dataset_manager._dataset_views["ds1"]
        assert ds1_view["model_series"] is None
        assert ds1_view["chi_squared"] is None
        assert ds1_view["r_squared"] is None
        assert window._run_results_tab._latest_model_series_by_dataset == {}
        assert window._run_results_tab._last_stats == {}

        ds2_view = window._dataset_manager._dataset_views["ds2"]
        assert ds2_view["model_series"] is not None
        assert ds2_view["chi_squared"] == pytest.approx(2.0)
        assert ds2_view["r_squared"] == pytest.approx(0.8)
    finally:
        window.close()
        qt_app.processEvents()


def test_worker_error_clears_active_run_fit_state_after_prior_success(qt_app, monkeypatch):
    window = _build_window()
    try:
        window.show()
        qt_app.processEvents()
        dataset_entry = window._dataset_entries[0]
        dataset = {
            "t": np.asarray(dataset_entry["t"], dtype=float),
            "species": {
                str(name): np.asarray(values, dtype=float)
                for name, values in dict(dataset_entry["species_data"]).items()
            },
        }
        plot_tabs = _DatasetPlotTabs()
        window._dataset_manager = DatasetManager(
            plot_tabs=plot_tabs,
            dataset_resolver=lambda name: dataset if name == "ds1" else None,
        )
        monkeypatch.setattr(
            QtWidgets.QMessageBox,
            "exec",
            lambda self: int(QtWidgets.QMessageBox.StandardButton.Ok),
        )
        window._handle_global_fit_complete({"result": _build_success_result()})
        window._params_ics_tab.set_last_fit_params({"k": 0.5})
        window._active_fit_dataset_ids = ["ds1"]

        worker = _SignalWorker()
        worker._running = False
        window._worker = worker

        window._on_worker_error({"kind": "fitting_error", "message": "boom"}, worker=worker)

        ds_view = window._dataset_manager._dataset_views["ds1"]
        assert ds_view["model_series"] is None
        assert ds_view["chi_squared"] is None
        assert ds_view["r_squared"] is None
        assert window._params_ics_tab.get_last_fit_params() == {}
        assert window._available_project_apply_scopes() == set()
        assert window._run_results_tab._latest_model_series_by_dataset == {}
        assert window._run_results_tab._last_stats == {}
    finally:
        window.close()
        qt_app.processEvents()


def test_failed_completion_routes_through_failed_run_closeout_helper(qt_app, monkeypatch):
    window = _build_window()
    try:
        window.show()
        qt_app.processEvents()
        result = _build_completion_result(
            status="fail",
            message="Optimization terminated successfully.",
            value=0.5,
        )
        calls: list[object] = []

        monkeypatch.setattr(
            window,
            "_clear_failed_run_visual_state",
            lambda current_result=None: calls.append(current_result),
        )
        monkeypatch.setattr(
            QtWidgets.QMessageBox,
            "exec",
            lambda self: int(QtWidgets.QMessageBox.StandardButton.Ok),
        )

        window._handle_global_fit_complete({"result": result})

        assert calls == [result]
    finally:
        window.close()
        qt_app.processEvents()


def test_worker_error_routes_through_failed_run_closeout_helper(qt_app, monkeypatch):
    window = _build_window()
    try:
        window.show()
        qt_app.processEvents()
        calls: list[object] = []

        monkeypatch.setattr(
            window,
            "_clear_failed_run_visual_state",
            lambda current_result=None: calls.append(current_result),
        )
        monkeypatch.setattr(
            QtWidgets.QMessageBox,
            "exec",
            lambda self: int(QtWidgets.QMessageBox.StandardButton.Ok),
        )

        worker = _SignalWorker()
        worker._running = False
        window._worker = worker
        window._active_fit_dataset_ids = ["ds1"]

        window._on_worker_error({"kind": "fitting_error", "message": "boom"}, worker=worker)

        assert calls == [None]
    finally:
        window.close()
        qt_app.processEvents()


def test_failed_completion_keeps_top_level_message_in_details_without_stack_trace(qt_app, monkeypatch):
    window = _build_window()
    try:
        window.show()
        qt_app.processEvents()
        result = _build_completion_result(status="fail", message="failed")
        result.completion = _build_completion(
            status="fail",
            optimizer_converged=True,
            optimizer_diagnostic=_make_diagnostic(
                phase="fatal",
                message="top-level process-pool failure",
                remediation="generic_retry",
            ),
            dataset_failures={
                "ds1": _make_diagnostic(
                    phase="final_replay",
                    dataset_id="ds1",
                    message="dataset replay failed",
                    remediation="generic_retry",
                ),
            },
            detail_sections=[
                FitDetailSection(
                    dataset_id=None,
                    failure=build_simulation_failure(
                        kind="simulation_error",
                        message="top-level process-pool failure",
                    ),
                )
            ],
        )
        dialogs: list[str] = []

        def _capture_exec(self):
            dialogs.append(self.detailedText())
            return int(QtWidgets.QMessageBox.StandardButton.Ok)

        monkeypatch.setattr(QtWidgets.QMessageBox, "exec", _capture_exec)

        window._handle_global_fit_complete({"result": result})

        assert dialogs == ["top-level process-pool failure"]
    finally:
        window.close()


def test_failed_completion_keeps_top_level_trace_when_matching_dataset_error_has_no_detail(qt_app, monkeypatch):
    window = _build_window()
    try:
        window.show()
        qt_app.processEvents()
        result = _build_completion_result(status="fail", message="failed")
        result.completion = _build_completion(
            status="fail",
            optimizer_converged=True,
            optimizer_diagnostic=_make_diagnostic(
                phase="optimizer",
                dataset_id="ds1",
                message="top-level failure",
                stack_trace="top trace",
            ),
            dataset_failures={
                "ds1": _make_diagnostic(phase="final_replay", dataset_id="ds1", message="dataset message only"),
            },
            detail_sections=[
                _make_detail_section(dataset_id="ds1", message="top-level failure", stack_trace="top trace"),
            ],
        )
        dialogs: list[str] = []

        def _capture_exec(self):
            dialogs.append(self.detailedText())
            return int(QtWidgets.QMessageBox.StandardButton.Ok)

        monkeypatch.setattr(QtWidgets.QMessageBox, "exec", _capture_exec)

        window._handle_global_fit_complete({"result": result})

        assert dialogs == ["Dataset 1\ntop trace"]
    finally:
        window.close()


def test_failed_completion_keeps_distinct_top_level_trace_for_same_dataset(qt_app, monkeypatch):
    window = _build_window()
    try:
        window.show()
        qt_app.processEvents()
        result = _build_completion_result(status="fail", message="failed")
        result.completion = _build_completion(
            status="fail",
            optimizer_converged=True,
            optimizer_diagnostic=_make_diagnostic(
                phase="optimizer",
                dataset_id="ghost_ds",
                message="optimizer-time failure",
                stack_trace="TOP TRACE",
            ),
            dataset_failures={
                "ghost_ds": _make_diagnostic(
                    phase="final_replay",
                    dataset_id="ghost_ds",
                    message="per-dataset failure",
                    stack_trace="DATASET TRACE",
                ),
            },
            detail_sections=[
                _make_detail_section(dataset_id="ghost_ds", message="optimizer-time failure", stack_trace="TOP TRACE"),
                _make_detail_section(dataset_id="ghost_ds", message="per-dataset failure", stack_trace="DATASET TRACE"),
            ],
        )
        dialogs: list[dict[str, object]] = []

        def _capture_exec(self):
            dialogs.append(
                {
                    "title": self.windowTitle(),
                    "text": self.text(),
                    "details": self.detailedText(),
                }
            )
            return int(QtWidgets.QMessageBox.StandardButton.Ok)

        monkeypatch.setattr(QtWidgets.QMessageBox, "exec", _capture_exec)

        window._handle_global_fit_complete({"result": result})

        assert dialogs == [
            {
                "title": "Global Fit Failed",
                "text": mock.ANY,
                "details": "ghost_ds\nTOP TRACE\n\n---\n\nghost_ds\nDATASET TRACE",
            }
        ]
    finally:
        window.close()


def test_failed_completion_suppresses_duplicate_top_level_dataset_trace(qt_app, monkeypatch):
    window = _build_window()
    try:
        window._dataset_entries.append(
            {
                "id": "ds_x",
                "label": "Dataset X",
                "t": np.asarray([0.0, 1.0, 2.0], dtype=float),
                "species_data": {"A": np.asarray([1.0, 0.8, 0.6], dtype=float)},
                "selected_species": ["A"],
                "weight": 1.0,
                "include": True,
            }
        )
        window.show()
        qt_app.processEvents()
        result = _build_completion_result(status="fail", message="failed")
        result.completion = _build_completion(
            status="fail",
            optimizer_converged=True,
            optimizer_diagnostic=_make_diagnostic(
                phase="optimizer",
                dataset_id="ds_x",
                message="duplicate failure",
                stack_trace="duplicate trace",
            ),
            dataset_failures={
                "ds_x": _make_diagnostic(
                    phase="final_replay",
                    dataset_id="ds_x",
                    message="duplicate failure",
                    stack_trace="duplicate trace",
                ),
            },
            detail_sections=[
                _make_detail_section(dataset_id="ds_x", message="duplicate failure", stack_trace="duplicate trace"),
            ],
        )
        dialogs: list[str] = []

        def _capture_exec(self):
            dialogs.append(self.detailedText())
            return int(QtWidgets.QMessageBox.StandardButton.Ok)

        monkeypatch.setattr(QtWidgets.QMessageBox, "exec", _capture_exec)

        window._handle_global_fit_complete({"result": result})

        assert dialogs == ["Dataset X\nduplicate trace"]
    finally:
        window.close()


def test_warning_completion_keeps_unlabeled_top_level_trace_without_dataset_tag(qt_app, monkeypatch):
    window = _build_window()
    try:
        window.show()
        qt_app.processEvents()
        result = _build_completion_result(status="warn", optimizer_converged=False)
        result.completion = _build_completion(
            status="warn",
            optimizer_converged=False,
            optimizer_diagnostic=_make_diagnostic(
                phase="fatal",
                message="fatal failure",
                stack_trace="fatal trace",
            ),
            detail_sections=[
                _make_detail_section(message="fatal failure", stack_trace="fatal trace"),
            ],
        )
        warning_mock = mock.Mock()
        dialogs: list[dict[str, str]] = []

        def _capture_exec(self):
            dialogs.append({"title": self.windowTitle(), "details": self.detailedText()})
            return int(QtWidgets.QMessageBox.StandardButton.Ok)

        monkeypatch.setattr("kindred.gui.fitting.window.logger.warning", warning_mock)
        monkeypatch.setattr(QtWidgets.QMessageBox, "exec", _capture_exec)

        window._handle_global_fit_complete({"result": result})

        assert dialogs == [
            {
                "title": "Optimization Complete (Warnings)",
                "details": "fatal trace",
            }
        ]
        assert warning_mock.call_args_list == [
            mock.call("Global fit completed with warnings: %s", "failed"),
            mock.call("%s", "fatal trace"),
        ]
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
            requested_solver="BDF",
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
