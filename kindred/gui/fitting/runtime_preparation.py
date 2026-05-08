from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6 import QtCore

from kindred.gui.fitting.launch import FittingLaunchPurpose
from kindred.gui.fitting.runtime_readiness import FittingRuntimeReadinessState

if TYPE_CHECKING:
    from kindred.gui.fitting.window import FittingWindow


class FittingRuntimePreparationOwner:
    """Owns passive fitting runtime preparation scheduling and close-after lifecycle."""

    def __init__(self, window: "FittingWindow") -> None:
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
        if getattr(window, "_closing", False):
            return
        if bool(getattr(window, "_worker", None) and getattr(window._worker, "isRunning", lambda: False)()):
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
        if getattr(window, "_closing", False):
            return
        if window.fit_runtime_readiness.snapshot().state is FittingRuntimeReadinessState.PREPARING:
            return
        if bool(getattr(window, "_worker", None) and getattr(window._worker, "isRunning", lambda: False)()):
            return
        try:
            launch_result = window.fit_launch_identity_owner.build_current_launch_result(
                purpose=FittingLaunchPurpose.PASSIVE_READINESS,
            )
        except RuntimeError as exc:
            window.fit_runtime_readiness.set_blocked(exc)
            if hasattr(window, "_status_label"):
                window._status_label.setText(f"Fitting runtime not ready: {exc}")
            window._refresh_run_button_enabled_state()
            return
        identity = launch_result.identity
        if identity is None:
            window.fit_runtime_readiness.set_blocked()
            window.fit_launch_identity_owner.render_launch_rejection(
                launch_result,
                purpose=FittingLaunchPurpose.PASSIVE_READINESS,
            )
            window._refresh_run_button_enabled_state()
            return
        window.fit_runtime_readiness.set_desired_identity(identity)
        if hasattr(window, "_status_label"):
            snapshot = window.fit_runtime_readiness.snapshot()
            if snapshot.state is FittingRuntimeReadinessState.PREPARING:
                window._status_label.setText("Preparing fitting runtime...")
                if hasattr(window, "_stop_button"):
                    window._stop_button.setEnabled(True)
            elif snapshot.state is FittingRuntimeReadinessState.READY:
                window._status_label.setText("Fitting runtime ready")
                if hasattr(window, "_stop_button"):
                    window._stop_button.setEnabled(False)
        window._refresh_run_button_enabled_state()

    def cancel_preparation(self, *, kill: bool) -> bool:
        return self._window.fit_runtime_readiness.cancel(kill=bool(kill))

    def queue_pending_refresh(self) -> None:
        window = self._window
        if getattr(window, "_closing", False):
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
        if hasattr(window, "_status_label"):
            if snapshot.state is FittingRuntimeReadinessState.READY:
                window._status_label.setText("Fitting runtime ready")
            elif snapshot.state is FittingRuntimeReadinessState.FAILED:
                message = str(snapshot.error or "Unknown fitting runtime preparation failure.")
                window._status_label.setText(f"Fitting runtime preparation failed: {message}")
            elif snapshot.state is FittingRuntimeReadinessState.BLOCKED:
                window._status_label.setText("Fitting runtime not ready")
        if hasattr(window, "_stop_button"):
            window._stop_button.setEnabled(False)
        window._set_fit_controls_locked(False)
        window._refresh_run_button_enabled_state()
        if snapshot.state is FittingRuntimeReadinessState.FAILED:
            self._refresh_pending = False
            return
        if not self.close_after_preparation_if_requested():
            self.queue_pending_refresh()
