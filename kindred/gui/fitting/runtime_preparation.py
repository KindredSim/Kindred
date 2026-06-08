from __future__ import annotations

from typing import Protocol

from PySide6 import QtCore

from kindred.gui.fitting.launch import FittingLaunchPurpose
from kindred.gui.fitting.runtime_readiness import (
    FittingRuntimePostPreparationAction,
    FittingRuntimeReadinessState,
)

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
    def run_fit(self) -> None: ...
    def close(self) -> None: ...


class FittingRuntimePreparationOwner:
    """Owns passive fitting runtime preparation scheduling."""

    def __init__(self, window: FittingRuntimePreparationWindow) -> None:
        self._window = window
        self._refresh_pending = False

    @property
    def refresh_pending(self) -> bool:
        return bool(self._refresh_pending)

    @refresh_pending.setter
    def refresh_pending(self, value: bool) -> None:
        self._refresh_pending = bool(value)

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
        snapshot = window.fit_runtime_readiness.snapshot()
        is_ready_for = getattr(window.fit_runtime_readiness, "is_ready_for", None)
        ready_for_current_identity = (
            snapshot.state is FittingRuntimeReadinessState.READY
            and callable(is_ready_for)
            and bool(is_ready_for(identity))
        )
        if ready_for_current_identity:
            window.set_fit_status("Fitting runtime ready")
        else:
            window.set_fit_status("Fitting inputs ready")
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
        action = window.fit_runtime_readiness.consume_post_preparation_action(snapshot)
        if action is FittingRuntimePostPreparationAction.CLOSE:
            QtCore.QTimer.singleShot(0, window.close)
            return
        if action is FittingRuntimePostPreparationAction.RUN_PENDING:
            if window.is_closing or window.is_fit_running():
                return
            window.run_fit()
            return
        if snapshot.state is FittingRuntimeReadinessState.FAILED:
            self._refresh_pending = False
            return
        self.queue_pending_refresh()
