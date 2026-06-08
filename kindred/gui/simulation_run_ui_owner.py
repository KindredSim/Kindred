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
        self._runtime_launch_available = False
        self._launch_readiness_stale = False
        self._run_target_available = False
        self._run_button_text = "Run"
        self._run_button_tooltip = ""
        self._run_button = None
        self._run_action = None
        self._stop_button = None
        self._progress = None
        self._status_label = None
        self._algebra_status_label = None
        self._mechanism_editor = None
        self._last_runtime_readiness_status = ""

    @property
    def requested_run_enabled(self) -> bool:
        return bool(self._run_button_requested_enabled)

    @property
    def runtime_launch_available(self) -> bool:
        return bool(self._runtime_launch_available)

    @property
    def launch_available(self) -> bool:
        return bool(self._runtime_launch_available and not self._launch_readiness_stale)

    @property
    def launch_readiness_stale(self) -> bool:
        return bool(self._launch_readiness_stale)

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

    def render_run_target_state(
        self,
        *,
        button_text: str,
        target_available: bool,
        tooltip: str,
    ) -> None:
        self._run_target_available = bool(target_available)
        self._run_button_text = str(button_text or "Run")
        self._run_button_tooltip = str(tooltip or "")
        self._apply_run_button_state()

    def render_launch_available(self, available: bool) -> None:
        self._runtime_launch_available = bool(available)
        self._launch_readiness_stale = False
        self._apply_run_button_state()

    def mark_launch_readiness_stale(self) -> None:
        # Runtime readiness truth is owned by render_runtime_readiness() /
        # render_launch_available().  This is only a UI gate saying that the
        # last published readiness is stale relative to newer inputs.
        self._launch_readiness_stale = True
        self._apply_run_button_state()

    def render_runtime_readiness(self, state: object) -> None:
        self._runtime_launch_available = bool(getattr(state, "launch_available", False))
        self._launch_readiness_stale = False
        message = str(getattr(state, "status_text", "") or "").strip()
        if message:
            current_status = self._status_text_value()
            if not current_status or current_status == "Ready" or current_status == self._last_runtime_readiness_status:
                self._set_status_text(message, source="runtime_readiness")
        elif bool(getattr(state, "clear_status", False)):
            if self._status_text_value() == self._last_runtime_readiness_status:
                self._set_status_text("", source="runtime_readiness")
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
        self._set_status_text(text, source="explicit")

    def _status_text_value(self) -> str:
        label = self._status_label
        if label is None:
            return ""
        try:
            return str(label.text())
        except (RuntimeError, AttributeError):
            return ""

    def _set_status_text(self, text: str, *, source: str) -> None:
        if self._status_label is not None:
            set_bounded_label_text(self._status_label, str(text), max_width=420)
        if str(source) == "runtime_readiness":
            self._last_runtime_readiness_status = str(text)
        else:
            self._last_runtime_readiness_status = ""

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

    def _mechanism_run_target_is_valid(self) -> bool:
        editor = self._mechanism_editor
        if editor is None or not hasattr(editor, "is_mechanism_valid"):
            return True
        try:
            return bool(editor.is_mechanism_valid())
        except (RuntimeError, AttributeError):
            return False

    def _apply_run_button_state(self) -> None:
        button = self._run_button
        if button is None:
            return
        button.setText(self._run_button_text)
        button.setToolTip(self._run_button_tooltip)
        mechanism_valid = self._mechanism_run_target_is_valid()
        effective_enabled = bool(
            self._run_button_requested_enabled
            and self._runtime_launch_available
            and not self._launch_readiness_stale
            and self._run_target_available
            and mechanism_valid
        )
        button.setEnabled(effective_enabled)
        if self._run_action is not None:
            self._run_action.setEnabled(effective_enabled)
            self._run_action.setText(self._run_button_text)
            self._run_action.setToolTip(self._run_button_tooltip)
        editor = self._mechanism_editor
        if editor is None:
            return
        editor.run_btn.setText(self._run_button_text)
        editor.run_btn.setToolTip(self._run_button_tooltip)
        editor.run_btn.setEnabled(effective_enabled)
