from __future__ import annotations

from contextlib import suppress
from typing import Callable, Optional

from kindred.gui.ui_helpers import set_bounded_label_text


class SimulationRunUiOwner:
    """Owns the simulation run controls' UI state and runtime-ready gate."""

    def __init__(
        self,
        *,
        schedule_runtime_availability_refresh: Callable[..., None],
        results_table_getter: Callable[[], object | None],
    ) -> None:
        self._schedule_runtime_availability_refresh = schedule_runtime_availability_refresh
        self._results_table_getter = results_table_getter
        self._run_button_requested_enabled = True
        self._runtime_ready = True
        self._run_button = None
        self._stop_button = None
        self._progress = None
        self._status_label = None
        self._algebra_status_label = None
        self._mechanism_editor = None

    @property
    def requested_run_enabled(self) -> bool:
        return bool(self._run_button_requested_enabled)

    @property
    def runtime_ready(self) -> bool:
        return bool(self._runtime_ready)

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

    def run_button_is_enabled(self) -> bool:
        button = self._run_button
        return bool(button is not None and button.isEnabled())

    def set_run_button_enabled(self, enabled: bool) -> None:
        self._run_button_requested_enabled = bool(enabled)
        self._apply_run_button_state()

    def set_runtime_backed_run_controls_ready(self, ready: bool) -> None:
        self._runtime_ready = bool(ready)
        self._apply_run_button_state()

    def schedule_runtime_availability_refresh(self) -> None:
        self._schedule_runtime_availability_refresh(wait=False)

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
        effective_enabled = bool(self._run_button_requested_enabled and self._runtime_ready)
        button.setEnabled(effective_enabled)
        editor = self._mechanism_editor
        if editor is None:
            return
        if hasattr(editor, "set_run_gated"):
            editor.set_run_gated(not effective_enabled)
        elif effective_enabled:
            editor.run_btn.setEnabled(editor.is_mechanism_valid())
        else:
            editor.run_btn.setEnabled(False)
