from __future__ import annotations

from contextlib import suppress
from typing import Callable, Optional

from kindred.gui.ui_helpers import set_bounded_label_text


class SimulationRunUiOwner:
    """Renders simulation run controls from controller-owned launch state."""

    def __init__(
        self,
        *,
        results_table_getter: Callable[[], object | None],
    ) -> None:
        self._results_table_getter = results_table_getter
        self._run_button_requested_enabled = True
        self._launch_available = True
        self._run_button = None
        self._run_action = None
        self._stop_button = None
        self._progress = None
        self._status_label = None
        self._algebra_status_label = None
        self._mechanism_editor = None

    @property
    def requested_run_enabled(self) -> bool:
        return bool(self._run_button_requested_enabled)

    @property
    def launch_available(self) -> bool:
        return bool(self._launch_available)

    def bind_widgets(
        self,
        *,
        run_button: object,
        stop_button: object,
        progress: object,
        status_label: object,
        algebra_status_label: object,
        mechanism_editor: Optional[object] = None,
    ) -> None:
        self._run_button = run_button
        self._stop_button = stop_button
        self._progress = progress
        self._status_label = status_label
        self._algebra_status_label = algebra_status_label
        self._mechanism_editor = mechanism_editor
        self._apply_run_button_state()

    def bind_run_action(self, action: object | None) -> None:
        self._run_action = action
        self._apply_run_button_state()

    def run_button_is_enabled(self) -> bool:
        button = self._run_button
        return bool(button is not None and button.isEnabled())

    def set_run_button_enabled(self, enabled: bool) -> None:
        self._run_button_requested_enabled = bool(enabled)
        self._apply_run_button_state()

    def render_launch_available(self, available: bool) -> None:
        self._launch_available = bool(available)
        self._apply_run_button_state()

    def render_runtime_readiness(self, state: object) -> None:
        self._launch_available = bool(getattr(state, "launch_available", False))
        message = str(getattr(state, "status_text", "") or "").strip()
        if message:
            self.set_status_text(message)
        elif bool(getattr(state, "clear_status", False)):
            self.set_status_text("")
        if bool(getattr(state, "failed", False)):
            self.set_stop_button_enabled(False)
            self.set_sim_progress_value(0)
        self._apply_run_button_state()

    def refresh_run_button_state(self) -> None:
        self._apply_run_button_state()

    def set_stop_button_enabled(self, enabled: bool) -> None:
        if self._stop_button is not None:
            self._stop_button.setEnabled(bool(enabled))

    def set_status_text(self, text: str) -> None:
        if self._status_label is not None:
            set_bounded_label_text(self._status_label, str(text), max_width=420)

    def set_sim_progress_value(self, value: int) -> None:
        if self._progress is not None:
            self._progress.setValue(int(value))

    def repaint_simulation_widgets(self) -> None:
        with suppress(RuntimeError, AttributeError):
            self._progress.update()
        with suppress(RuntimeError, AttributeError):
            self._status_label.update()
        table = self._results_table_getter()
        if table is not None:
            with suppress(RuntimeError, AttributeError):
                table.viewport().update()

    def set_algebra_status_text(self, text: str, *, details: str | None = None) -> None:
        if self._algebra_status_label is not None:
            set_bounded_label_text(self._algebra_status_label, str(text), max_width=420)
            self._algebra_status_label.setToolTip(str(details or ""))

    def _apply_run_button_state(self) -> None:
        button = self._run_button
        if button is None:
            return
        effective_enabled = bool(self._run_button_requested_enabled and self._launch_available)
        button.setEnabled(effective_enabled)
        if self._run_action is not None:
            self._run_action.setEnabled(effective_enabled)
        editor = self._mechanism_editor
        if editor is None:
            return
        if effective_enabled:
            editor.run_btn.setEnabled(editor.is_mechanism_valid())
        else:
            editor.run_btn.setEnabled(False)
