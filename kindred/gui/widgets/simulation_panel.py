"""Initial Conditions panel with one canonical Run-settings candidate."""

from __future__ import annotations

from kindred.gui.display_name_policy import scientific_symbol_html, scientific_symbol_spoken

from PySide6 import QtCore, QtWidgets

from kindred.core.project_document import (
    SimulationExecutionSettings,
    SUPPORTED_SOLVERS,
    parse_points_text,
    validate_simulation_execution_settings,
)
from kindred.core.validation import parse_finite_numeric_text
from kindred.gui.ui_helpers import (
    ControlRowsLayout, SOLVER_TOOLTIP, configure_compact_numeric_edit, ignore_wheel,
)
from kindred.gui.widgets.batch_initial_conditions_table import (
    BatchInitialConditionsTableModel,
    BatchInitialConditionsTableView,
)

__all__ = ["SimulationPanel"]


class _ControlsScrollArea(QtWidgets.QScrollArea):
    """Give wrapping controls their full content height and native overflow."""

    def __init__(self, hosts: tuple[QtWidgets.QWidget, ...], parent) -> None:
        super().__init__(parent)
        self._hosts = hosts
        self._content_height = 0
        self._pending = False
        self.setObjectName("initial_conditions_controls_scroll")
        self.setStyleSheet(
            "QScrollArea#initial_conditions_controls_scroll { border: none; padding: 0; margin: 0; }"
        )
        self.setFrameShape(QtWidgets.QFrame.NoFrame)
        # Clicks on its buttons must not move focus here (see the Run button).
        self.setFocusPolicy(QtCore.Qt.NoFocus)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        content = QtWidgets.QWidget(self)
        layout = QtWidgets.QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.setSizeConstraint(QtWidgets.QLayout.SetNoConstraint)
        for host in hosts:
            layout.addWidget(host)
            host.installEventFilter(self)
            for control in host.findChildren(QtWidgets.QWidget):
                control.installEventFilter(self)
        self.setWidget(content)
        content.installEventFilter(self)
        self.viewport().installEventFilter(self)
        self._schedule_measure()

    def sizeHint(self) -> QtCore.QSize:
        size = super().sizeHint()
        size.setHeight(self._content_height)
        return size

    def minimumSizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(0, self.fontMetrics().height() + 8)

    def eventFilter(self, watched, event):
        if event.type() == QtCore.QEvent.FocusIn:
            # Finish native focus scrolling first, then reveal the whole control
            # (the style's smaller focus rectangle can leave its edges clipped).
            QtCore.QTimer.singleShot(0, self, self._reveal_focused_control)
        if event.type() in {
            QtCore.QEvent.Resize, QtCore.QEvent.LayoutRequest,
            QtCore.QEvent.Show, QtCore.QEvent.Hide,
            QtCore.QEvent.FontChange, QtCore.QEvent.StyleChange,
        }:
            self._schedule_measure()
        return super().eventFilter(watched, event)

    def _reveal_focused_control(self) -> None:
        control = self.focusWidget()
        if control is not None and self.widget().isAncestorOf(control):
            position = control.mapTo(self.widget(), QtCore.QPoint())
            self.ensureVisible(
                position.x(), position.y() + control.height() // 2,
                0, (control.height() + 1) // 2,
            )

    def _schedule_measure(self) -> None:
        if not self._pending:
            self._pending = True
            QtCore.QTimer.singleShot(0, self._measure)

    def _measure(self) -> None:
        self._pending = False
        spacing = self.widget().layout().spacing() * (len(self._hosts) - 1)
        # A scrollbar can itself force an extra wrap. Test the unobstructed
        # viewport first so that extra row cannot keep an unnecessary bar alive.
        width = self.maximumViewportSize().width()
        heights = [max(0, host.layout().totalHeightForWidth(width)) for host in self._hosts]
        if sum(heights) + spacing > self.viewport().height():
            width = self.viewport().width()
            heights = [max(0, host.layout().totalHeightForWidth(width)) for host in self._hosts]
        for host, height in zip(self._hosts, heights):
            if host.minimumHeight() != height or host.maximumHeight() != height:
                host.setFixedHeight(height)
        required = sum(heights) + spacing
        if self.widget().minimumHeight() != required:
            self.widget().setMinimumHeight(required)
        if required != self._content_height:
            self._content_height = required
            self.setMaximumHeight(required)
            self.updateGeometry()


class SimulationPanel(QtWidgets.QWidget):
    addSetRequested = QtCore.Signal()
    moveSelectedRequested = QtCore.Signal(int)
    deleteSelectedRequested = QtCore.Signal()
    runSelectedRequested = QtCore.Signal()
    stopRequested = QtCore.Signal()
    workspaceRunDefaultsRequested = QtCore.Signal(object)
    settingsValidationFailed = QtCore.Signal(str)
    runDefaultsDraftStateChanged = QtCore.Signal(bool)

    def __init__(
        self,
        *,
        batch_model: BatchInitialConditionsTableModel,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._canonical_settings = SimulationExecutionSettings()
        self._rendering = False
        self._candidate_dirty = False
        self._simulation_job_kind: str | None = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self._controls_host = QtWidgets.QWidget(self)
        controls = ControlRowsLayout(self._controls_host)
        self.solver_method_combo = QtWidgets.QComboBox(self)
        ignore_wheel(self.solver_method_combo)
        self.solver_method_combo.addItems(list(SUPPORTED_SOLVERS))
        self.solver_method_combo.setToolTip(SOLVER_TOOLTIP)
        self.t_start_edit = self._number_edit("", "Absolute Run start time")
        self.t_end_edit = self._number_edit("", "Absolute Run end time")
        self.points_edit = self._number_edit(
            "",
            "Number of evenly spaced base samples over the completed interval, "
            "including both endpoints. Intervention boundary rows are additional. "
            "A stop at the start returns one sample.",
        )
        for label, widget in (
            ("Solver", self.solver_method_combo),
            ("t_start", self.t_start_edit),
            ("t_end", self.t_end_edit),
            ("Points", self.points_edit),
        ):
            name = label
            label_widget = QtWidgets.QLabel(label, self)
            label_widget.setBuddy(widget)
            if name == "Points":
                label_widget.setToolTip(widget.toolTip())
            if name in {"t_start", "t_end"}:
                label_widget.setTextFormat(QtCore.Qt.RichText)
                label_widget.setText(scientific_symbol_html(name, family="time", role=name))
            widget.setAccessibleName(scientific_symbol_spoken(name, family="time", role=name))
            group = QtWidgets.QWidget(self)
            group_layout = QtWidgets.QHBoxLayout(group)
            group_layout.setContentsMargins(0, 0, 0, 0)
            group_layout.addWidget(label_widget)
            group_layout.addWidget(widget)
            controls.addWidget(group)

        self.solver_method_combo.currentTextChanged.connect(self._mark_candidate_dirty)
        for edit in (self.t_start_edit, self.t_end_edit, self.points_edit):
            edit.textEdited.connect(self._mark_candidate_dirty)
            edit.editingFinished.connect(self._settle_after_edit)
        self.solver_method_combo.activated.connect(self._settle_after_edit)

        self._actions_host = QtWidgets.QWidget(self)
        actions_layout = ControlRowsLayout(self._actions_host)
        self.add_batch_set_btn = self._button("Add Set", "addBatchSetButton")
        self.move_batch_up_btn = self._button("Move Up", "moveBatchSetUpButton")
        self.move_batch_down_btn = self._button("Move Down", "moveBatchSetDownButton")
        self.delete_batch_set_btn = self._button(
            "Delete Set(s)\N{HORIZONTAL ELLIPSIS}", "deleteBatchSetButton"
        )
        self.run_btn = self._button("Run", "runSelectedButton")
        # A pointer Run must not take focus: focus loss would commit the open
        # cell or Run-setting editor before Run admits Reactions.
        self.run_btn.setFocusPolicy(QtCore.Qt.TabFocus)
        self.stop_btn = self._button("Stop", "stopButton")
        action_buttons = (
            self.add_batch_set_btn, self.move_batch_up_btn,
            self.move_batch_down_btn, self.delete_batch_set_btn,
            self.run_btn, self.stop_btn,
        )
        for button in action_buttons:
            actions_layout.addWidget(button)
        self._controls_scroll = _ControlsScrollArea(
            (self._controls_host, self._actions_host), self
        )
        layout.addWidget(self._controls_scroll, 0)

        self.add_batch_set_btn.clicked.connect(self.addSetRequested)
        self.move_batch_up_btn.clicked.connect(
            lambda: self.moveSelectedRequested.emit(-1)
        )
        self.move_batch_down_btn.clicked.connect(
            lambda: self.moveSelectedRequested.emit(1)
        )
        self.delete_batch_set_btn.clicked.connect(self.deleteSelectedRequested)
        self.run_btn.clicked.connect(self._request_run)
        self.stop_btn.clicked.connect(self.stopRequested)

        self.batch_table = BatchInitialConditionsTableView(self)
        self.batch_table.setObjectName("batchTable")
        self.batch_table.setModel(batch_model)
        layout.addWidget(self.batch_table, 1)

        self._reproject_canonical()
        self._sync_controls()

    @staticmethod
    def _button(text: str, object_name: str) -> QtWidgets.QPushButton:
        button = QtWidgets.QPushButton(text)
        button.setObjectName(object_name)
        return button

    def _number_edit(self, value: str, tooltip: str) -> QtWidgets.QLineEdit:
        edit = QtWidgets.QLineEdit(value, self)
        configure_compact_numeric_edit(edit, characters=6)
        edit.setToolTip(tooltip)
        return edit

    def render_run_defaults(
        self,
        settings: SimulationExecutionSettings,
        *,
        simulation_job_kind: str | None,
    ) -> None:
        kind = simulation_job_kind
        previous_rendering = self._rendering
        self._rendering = True
        try:
            preserve_candidate = (
                self._candidate_dirty and settings == self._canonical_settings
            )
            self._canonical_settings = settings
            self._simulation_job_kind = kind
            if not preserve_candidate:
                self._candidate_dirty = False
                self._reproject_canonical()
            self._sync_controls()
        finally:
            self._rendering = previous_rendering

    def render_job_state(self, simulation_job_kind: str | None) -> None:
        """Project the live simulation owner's job state into Stop."""
        self._simulation_job_kind = simulation_job_kind
        self._sync_controls()

    @property
    def has_run_defaults_draft(self) -> bool:
        """Whether the visible controls contain an unsettled authored draft."""

        return bool(self._candidate_dirty)

    def capture_run_defaults_draft(self) -> bool:
        """Capture a focused control before a modal boundary, without publishing."""

        was_dirty = self._candidate_dirty
        focused = QtWidgets.QApplication.focusWidget()
        controls = (
            self.solver_method_combo,
            self.t_start_edit,
            self.t_end_edit,
            self.points_edit,
        )
        if focused is None or not any(
            focused is control or control.isAncestorOf(focused)
            for control in controls
        ):
            return False
        # A focused control is not necessarily edited.  Preserve the exact
        # candidate state and only promote it to a draft when its current
        # value differs from canonical (or cannot be validated).
        if not self._candidate_dirty:
            try:
                self._candidate_dirty = (
                    self.current_settings_candidate()
                    != self._canonical_settings
                )
            except ValueError:
                self._candidate_dirty = True
        blockers = tuple(QtCore.QSignalBlocker(control) for control in controls)
        try:
            focused.clearFocus()
        finally:
            del blockers
        self._sync_controls()
        if self._candidate_dirty != was_dirty:
            self.runDefaultsDraftStateChanged.emit(bool(self._candidate_dirty))
        return True

    def discard_run_defaults_draft(self) -> bool:
        if not self._candidate_dirty:
            return False
        self._candidate_dirty = False
        self._reproject_canonical()
        self._sync_controls()
        self.runDefaultsDraftStateChanged.emit(False)
        return True

    def settle_saved_run_defaults(
        self,
        settings: SimulationExecutionSettings,
    ) -> None:
        """Adopt settings already published by the enclosing Save candidate."""

        had_draft = self._candidate_dirty
        self._canonical_settings = settings
        self._candidate_dirty = False
        self._reproject_canonical()
        self._sync_controls()
        if had_draft:
            self.runDefaultsDraftStateChanged.emit(False)

    def settle_run_defaults(self) -> bool:
        """Commit one complete candidate or keep the authored draft and abort once."""
        if not self._candidate_dirty:
            return True
        try:
            candidate = self.current_settings_candidate()
        except ValueError as exc:
            # Keep the authored Draft visible and blocking.  Reprojecting the
            # canonical values here would make a focus-loss event silently
            # erase the invalid text before Run can observe it.
            self._sync_controls()
            self.settingsValidationFailed.emit(str(exc))
            return False
        self._candidate_dirty = False
        if candidate != self._canonical_settings:
            self.workspaceRunDefaultsRequested.emit(candidate)
        self.runDefaultsDraftStateChanged.emit(False)
        return True

    def run_defaults_draft_is_valid(self) -> bool:
        """Whether the visible Run-settings draft admits as settings."""

        try:
            self.current_settings_candidate()
        except ValueError:
            return False
        return True

    def current_settings_candidate(self) -> SimulationExecutionSettings:
        settings = self._canonical_settings
        # Only text that differs from the exact projection is parsed;
        # untouched fields come from the accepted backing object.
        changed = set()
        if self.t_start_edit.text() != repr(float(settings.t_start)):
            changed.add("t_start")
        if self.t_end_edit.text() != repr(float(settings.t_end)):
            changed.add("t_end")
        if self.points_edit.text() != str(settings.integration_points):
            changed.add("integration_points")
        t_start, t_end = settings.t_start, settings.t_end
        if "t_start" in changed:
            text = self.t_start_edit.text()
            t_start = parse_finite_numeric_text(text, field=f"Start time {text!r}")
        if "t_end" in changed:
            text = self.t_end_edit.text()
            t_end = parse_finite_numeric_text(text, field=f"End time {text!r}")
        points = (
            parse_points_text(self.points_edit.text())
            if "integration_points" in changed
            else settings.integration_points
        )
        candidate = SimulationExecutionSettings(
            solver=str(self.solver_method_combo.currentText()),
            t_start=float(t_start),
            t_end=float(t_end),
            integration_points=points,
        )
        return validate_simulation_execution_settings(candidate)

    def _reproject_canonical(self) -> None:
        settings = self._canonical_settings
        previous_rendering = self._rendering
        self._rendering = True
        try:
            self.solver_method_combo.setCurrentText(settings.solver)
            self.t_start_edit.setText(repr(float(settings.t_start)))
            self.t_end_edit.setText(repr(float(settings.t_end)))
            self.points_edit.setText(str(settings.integration_points))
        finally:
            self._rendering = previous_rendering

    @QtCore.Slot()
    def _mark_candidate_dirty(self, *_args) -> None:
        if self._rendering:
            return
        was_dirty = self._candidate_dirty
        self._candidate_dirty = True
        if not was_dirty:
            self.runDefaultsDraftStateChanged.emit(True)

    @QtCore.Slot()
    def _settle_after_edit(self, *_args) -> None:
        if not self._rendering:
            self.settle_run_defaults()

    @QtCore.Slot()
    def _request_run(self) -> None:
        # The Run owner captures focused editors and Run-setting drafts after
        # it admits Reactions; nothing is published before that admission.
        self.runSelectedRequested.emit()

    def _sync_controls(self) -> None:
        show_stop = self._simulation_job_kind in {"run", "preview"}
        self.stop_btn.setVisible(show_stop)
        self.stop_btn.setEnabled(show_stop)
