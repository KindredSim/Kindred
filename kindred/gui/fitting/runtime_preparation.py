from __future__ import annotations

from typing import Protocol

from PySide6 import QtCore

from kindred.gui.fitting.launch import FittingLaunchPurpose
from kindred.gui.fitting.runtime_readiness import FittingRuntimeReadinessState

class FittingRuntimePreparationWindow(Protocol):
    @property
    def is_closing(self) -> bool: ...
    @property
    def fit_runtime_readiness(self): ...
    def is_fit_running(self) -> bool: ...
    def build_current_launch_result(self, *, purpose, **kwargs): ...
    def render_launch_rejection(self, result, *, purpose) -> None: ...
    def refresh_run_button_enabled_state(self) -> None: ...
    def set_fit_status(self, text: str) -> None: ...
    def set_fit_stop_enabled(self, enabled: bool) -> None: ...
    def set_fit_controls_locked(self, locked: bool) -> None: ...
    def close(self) -> None: ...


class FittingRuntimePreparationOwner:
    """Owns passive fitting runtime preparation scheduling and close-after lifecycle."""

    def __init__(self, window: FittingRuntimePreparationWindow) -> None:
        self._window = window
        self._refresh_pending = False
        self._close_after_prepare = False

    @property
    def refresh_pending(self) -> bool:
        return bool(self._refresh_pending)

    @refresh_pending.setter
    def refresh_pending(self, value: bool) -> None:
        self._refresh_pending = bool(value)

    @property
    def close_after_prepare(self) -> bool:
        return bool(self._close_after_prepare)

    @close_after_prepare.setter
    def close_after_prepare(self, value: bool) -> None:
        self._close_after_prepare = bool(value)

    def schedule_refresh(self) -> None:
        window = self._window
        if window.is_closing:
            return
        if window.is_fit_running():
            return
        if window.fit_runtime_readiness.snapshot().state is FittingRuntimeReadinessState.PREPARING:
            self._refresh_pending = True
            return
        if self._refresh_pending:
            return
        self._refresh_pending = True
        QtCore.QTimer.singleShot(0, self.run_refresh)

    def run_refresh(self) -> None:
        if not self._refresh_pending:
            return
        self._refresh_pending = False
        self.prepare_current_state()

    def prepare_current_state(self) -> None:
        window = self._window
        if window.is_closing:
            return
        if window.fit_runtime_readiness.snapshot().state is FittingRuntimeReadinessState.PREPARING:
            return
        if window.is_fit_running():
            return
        try:
            launch_result = window.build_current_launch_result(
                purpose=FittingLaunchPurpose.PASSIVE_READINESS,
            )
        except RuntimeError as exc:
            window.fit_runtime_readiness.set_blocked(exc)
            window.set_fit_status(f"Fitting runtime not ready: {exc}")
            window.refresh_run_button_enabled_state()
            return
        identity = launch_result.identity
        if identity is None:
            window.fit_runtime_readiness.set_blocked()
            window.render_launch_rejection(
                launch_result,
                purpose=FittingLaunchPurpose.PASSIVE_READINESS,
            )
            window.refresh_run_button_enabled_state()
            return
        window.fit_runtime_readiness.set_desired_identity(identity)
        snapshot = window.fit_runtime_readiness.snapshot()
        if snapshot.state is FittingRuntimeReadinessState.PREPARING:
            window.set_fit_status("Preparing fitting runtime...")
            window.set_fit_stop_enabled(True)
        elif snapshot.state is FittingRuntimeReadinessState.READY:
            window.set_fit_status("Fitting runtime ready")
            window.set_fit_stop_enabled(False)
        window.refresh_run_button_enabled_state()

    def cancel_preparation(self, *, kill: bool) -> bool:
        return self._window.fit_runtime_readiness.cancel(kill=bool(kill))

    def queue_pending_refresh(self) -> None:
        window = self._window
        if window.is_closing:
            return
        if not self._refresh_pending:
            return
        if window.fit_runtime_readiness.snapshot().state is FittingRuntimeReadinessState.PREPARING:
            return
        self.run_refresh()

    def request_close_after_prepare(self) -> None:
        self._close_after_prepare = True

    def close_after_preparation_if_requested(self) -> bool:
        if not self._close_after_prepare:
            return False
        self._close_after_prepare = False
        QtCore.QTimer.singleShot(0, self._window.close)
        return True

    @QtCore.Slot()
    def poll_preparation(self) -> None:
        window = self._window
        if not window.fit_runtime_readiness.handle_worker_finished():
            QtCore.QTimer.singleShot(0, self.poll_preparation)
            return
        snapshot = window.fit_runtime_readiness.snapshot()
        if snapshot.state is FittingRuntimeReadinessState.READY:
            window.set_fit_status("Fitting runtime ready")
        elif snapshot.state is FittingRuntimeReadinessState.FAILED:
            message = str(snapshot.error or "Unknown fitting runtime preparation failure.")
            window.set_fit_status(f"Fitting runtime preparation failed: {message}")
        elif snapshot.state is FittingRuntimeReadinessState.BLOCKED:
            window.set_fit_status("Fitting runtime not ready")
        window.set_fit_stop_enabled(False)
        window.set_fit_controls_locked(False)
        window.refresh_run_button_enabled_state()
        if snapshot.state is FittingRuntimeReadinessState.FAILED:
            self._refresh_pending = False
            return
        if not self.close_after_preparation_if_requested():
            self.queue_pending_refresh()
