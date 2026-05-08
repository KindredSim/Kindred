from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6 import QtWidgets

from kindred.gui.fitting.runtime_readiness import (
    FittingRuntimeLaunchDecisionState,
    FittingRuntimeReadinessState,
)

if TYPE_CHECKING:
    from kindred.gui.fitting.window import FittingWindow


class FittingRunCommandOwner:
    """Owns visible Run Fit command policy from identity collection through accepted launch dispatch."""

    def __init__(self, window: "FittingWindow") -> None:
        self._window = window

    def run_fit(self) -> None:
        window = self._window
        if window._worker and window._worker.isRunning():
            QtWidgets.QMessageBox.information(window, "Fit Running", "A fit is already in progress.")
            return
        window._reset_fit_run_cached_state()
        window._species_table.flush_visible_weight_edits()
        window._species_table.flush_dataset_weight_editor()
        try:
            identity = window.fit_launch_identity_owner.build_current_fit_runtime_identity(
                show_dataset_messages=True,
            )
        except RuntimeError as exc:
            window.fit_runtime_readiness.set_blocked(exc)
            if hasattr(window, "_status_label"):
                window._status_label.setText(f"Fitting runtime not ready: {exc}")
            window._refresh_run_button_enabled_state()
            return
        if identity is None:
            window.fit_runtime_readiness.set_blocked()
            window._clear_failed_run_visual_state()
            window._refresh_run_button_enabled_state()
            return
        window._capture_failed_fit_restore_baseline()
        launch_decision = window.fit_runtime_readiness.prepare_or_accept_launch(identity)
        if launch_decision.state is not FittingRuntimeLaunchDecisionState.ACCEPTED:
            if hasattr(window, "_status_label"):
                snapshot = launch_decision.snapshot or window.fit_runtime_readiness.snapshot()
                if snapshot.state is FittingRuntimeReadinessState.PREPARING:
                    window._status_label.setText("Preparing fitting runtime...")
                    if hasattr(window, "_stop_button"):
                        window._stop_button.setEnabled(True)
                else:
                    window._status_label.setText("Fitting runtime is not ready")
            window._refresh_run_button_enabled_state()
            return
        accepted_launch = launch_decision.accepted_launch
        if accepted_launch is None:
            window._refresh_run_button_enabled_state()
            return
        window.fit_worker_launch_owner.start_runtime_launch(accepted_launch)
