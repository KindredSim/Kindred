"""Initial Conditions panel with one canonical Run-settings candidate."""

from __future__ import annotations

from kindred.gui.display_name_policy import scientific_symbol_html, scientific_symbol_spoken

from PySide6 import QtCore, QtGui, QtWidgets

from kindred.core.project_document import (
    SIMULATION_POINT_RANGE,
    SimulationExecutionSettings,
    SUPPORTED_SOLVERS,
    validate_simulation_execution_settings,
)
from kindred.core.validation import parse_finite_numeric_text
from kindred.gui.ui_helpers import (
    ControlRowsLayout, configure_compact_numeric_edit, setup_scientific_validator,
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
    runEnabledChanged = QtCore.Signal(bool)
    runDefaultsDraftStateChanged = QtCore.Signal(bool)

    def __init__(
        self,
        *,
        batch_model: BatchInitialConditionsTableModel,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._canonical_settings = SimulationExecutionSettings()
        self._candidate_values: dict[str, object] = {
            "solver": self._canonical_settings.solver,
            "t_start": self._canonical_settings.t_start,
            "t_end": self._canonical_settings.t_end,
            "integration_points": self._canonical_settings.integration_points,
        }
        self._candidate_changed_fields: set[str] = set()
        self._rendering = False
        self._candidate_dirty = False
        self._run_admitted = True
        self._simulation_job_kind: str | None = None
        self._last_run_enabled: bool | None = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self._controls_host = QtWidgets.QWidget(self)
        controls = ControlRowsLayout(self._controls_host)
        self.solver_method_combo = QtWidgets.QComboBox(self)
        self.solver_method_combo.addItems(list(SUPPORTED_SOLVERS))
        self.solver_method_combo.setToolTip(
            "Choose the numerical method used to calculate the simulation."
        )
        self.t_start_edit = self._number_edit("0.0", "Absolute Run start time")
        self.t_end_edit = self._number_edit("10.0", "Absolute Run end time")
        self.num_points_spinbox = QtWidgets.QSpinBox(self)
        self.num_points_spinbox.setRange(*SIMULATION_POINT_RANGE)
        self.num_points_spinbox.setSingleStep(10)
        self.num_points_spinbox.setToolTip(
            "Number of evenly spaced base samples over the completed interval, "
            "including both endpoints. Intervention boundary rows are additional. "
            "A stop at the start returns one sample."
        )
        for label, widget in (
            ("Solver", self.solver_method_combo),
            ("t_start", self.t_start_edit),
            ("t_end", self.t_end_edit),
            ("Points", self.num_points_spinbox),
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
        self.t_start_edit.textEdited.connect(self._mark_candidate_dirty)
        self.t_end_edit.textEdited.connect(self._mark_candidate_dirty)
        # QSpinBox normalizes its value before ``valueChanged`` reaches us.
        # Capture the embedded editor's authored text first so an invalid or
        # intermediate draft remains visible until the owner admits or
        # explicitly discards it.
        self.num_points_spinbox.lineEdit().textEdited.connect(
            self._capture_integration_points_text
        )
        self.num_points_spinbox.valueChanged.connect(self._mark_candidate_dirty)
        self.solver_method_combo.activated.connect(self._settle_after_edit)
        self.t_start_edit.editingFinished.connect(self._settle_after_edit)
        self.t_end_edit.editingFinished.connect(self._settle_after_edit)
        self.num_points_spinbox.editingFinished.connect(self._settle_after_edit)

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
        self.batch_table.selectionModel().selectionChanged.connect(
            self._sync_run_enabled
        )
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
        setup_scientific_validator(edit)
        return edit

    def render_run_defaults(
        self,
        settings: SimulationExecutionSettings,
        *,
        run_admitted: bool,
        simulation_job_kind: str | None,
    ) -> None:
        validate_simulation_execution_settings(settings)
        kind = None if simulation_job_kind is None else str(simulation_job_kind)
        if kind not in {None, "run", "preview"}:
            raise ValueError("simulation_job_kind must be None, 'run', or 'preview'.")
        previous_rendering = self._rendering
        self._rendering = True
        blockers = tuple(
            QtCore.QSignalBlocker(widget)
            for widget in (
                self.solver_method_combo,
                self.t_start_edit,
                self.t_end_edit,
                self.num_points_spinbox,
            )
        )
        try:
            preserve_candidate = (
                self._candidate_dirty and settings == self._canonical_settings
            )
            self._canonical_settings = settings
            if not preserve_candidate:
                self._candidate_dirty = False
                self._candidate_values = {
                    "solver": settings.solver,
                    "t_start": settings.t_start,
                    "t_end": settings.t_end,
                    "integration_points": settings.integration_points,
                }
                self._candidate_changed_fields.clear()
            self._run_admitted = bool(run_admitted)
            self._simulation_job_kind = kind
            if not preserve_candidate:
                self._reproject_canonical()
            self._sync_controls()
        finally:
            del blockers
            self._rendering = previous_rendering

    def render_job_state(self, simulation_job_kind: str | None) -> None:
        """Project the live simulation owner's job state into Stop."""
        kind = (
            None
            if simulation_job_kind is None
            else str(simulation_job_kind)
        )
        if kind not in {None, "run", "preview"}:
            raise ValueError(
                "simulation_job_kind must be None, 'run', or 'preview'."
            )
        self._simulation_job_kind = kind
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
            self.num_points_spinbox,
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
            except (TypeError, ValueError):
                self._candidate_dirty = True
                points_edit = self.num_points_spinbox.lineEdit()
                if points_edit.text() != str(
                    int(self._candidate_values["integration_points"])
                ):
                    self._candidate_values["integration_points"] = (
                        points_edit.text()
                    )
                    self._candidate_changed_fields.add("integration_points")
        if focused is self.solver_method_combo:
            self._candidate_changed_fields.add("solver")
            self._candidate_values["solver"] = str(
                self.solver_method_combo.currentText()
            )
        elif focused is self.t_start_edit:
            self._candidate_changed_fields.add("t_start")
        elif focused is self.t_end_edit:
            self._candidate_changed_fields.add("t_end")
        elif focused is self.num_points_spinbox:
            self._candidate_changed_fields.add("integration_points")
        elif (
            focused is self.num_points_spinbox.lineEdit()
            or self.num_points_spinbox.lineEdit().isAncestorOf(focused)
        ):
            self._candidate_changed_fields.add("integration_points")
        blockers = tuple(QtCore.QSignalBlocker(control) for control in controls)
        try:
            focused.clearFocus()
            self._restore_integration_points_draft()
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

        validate_simulation_execution_settings(settings)
        had_draft = self._candidate_dirty
        self._canonical_settings = settings
        self._candidate_dirty = False
        self._candidate_values = {
            "solver": settings.solver,
            "t_start": settings.t_start,
            "t_end": settings.t_end,
            "integration_points": settings.integration_points,
        }
        self._candidate_changed_fields.clear()
        self._reproject_canonical()
        self._sync_controls()
        if had_draft:
            self.runDefaultsDraftStateChanged.emit(False)

    def settle_run_defaults(self) -> bool:
        """Commit one complete candidate or reproject canonical and abort once."""
        if not self._candidate_dirty:
            return True
        try:
            candidate = self.current_settings_candidate()
        except (TypeError, ValueError) as exc:
            # Keep the authored Draft visible and blocking.  Reprojecting the
            # canonical values here would make a focus-loss event silently
            # erase the invalid text before Run can observe it.
            self._candidate_dirty = True
            self._restore_integration_points_draft()
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
        except (TypeError, ValueError):
            return False
        return True

    def current_settings_candidate(self) -> SimulationExecutionSettings:
        values = dict(self._candidate_values)
        changed = set(self._candidate_changed_fields)
        # Direct callers may set a field without the user-edit signal. Detect
        # only text that differs from the exact projection; untouched fields
        # continue to come from the accepted backing object.
        for field, edit in (
            ("t_start", self.t_start_edit),
            ("t_end", self.t_end_edit),
        ):
            projected = repr(float(values[field]))
            if edit.text() != projected:
                changed.add(field)
        for field, edit in (
            ("t_start", self.t_start_edit),
            ("t_end", self.t_end_edit),
        ):
            if field not in changed:
                continue
            values[field] = parse_finite_numeric_text(edit.text(), field=field)
        points_edit = self.num_points_spinbox.lineEdit()
        projected_points = str(int(values["integration_points"]))
        if points_edit.text() != projected_points:
            changed.add("integration_points")
            if "integration_points" not in self._candidate_changed_fields:
                values["integration_points"] = points_edit.text()
                self._candidate_values["integration_points"] = points_edit.text()
                self._candidate_changed_fields.add("integration_points")
        if "integration_points" in changed:
            raw_points = values["integration_points"]
            raw_text = str(raw_points)
            state, _corrected, _position = self.num_points_spinbox.validate(
                raw_text,
                0,
            )
            if state is not QtGui.QValidator.State.Acceptable:
                minimum, maximum = SIMULATION_POINT_RANGE
                raise ValueError(
                    f"Integration points {raw_text!r} must be an integer between "
                    f"{minimum} and {maximum}."
                )
            try:
                values["integration_points"] = int(raw_text)
            except (TypeError, ValueError, OverflowError) as exc:
                minimum, maximum = SIMULATION_POINT_RANGE
                raise ValueError(
                    f"Integration points {raw_text!r} must be an integer between "
                    f"{minimum} and {maximum}."
                ) from exc
        candidate = SimulationExecutionSettings(
            solver=str(self.solver_method_combo.currentText()),
            t_start=float(values["t_start"]),
            t_end=float(values["t_end"]),
            integration_points=int(values["integration_points"]),
        )
        try:
            validate_simulation_execution_settings(candidate)
        except ValueError as exc:
            raise ValueError(f"t_start={values['t_start']!r}, t_end={values['t_end']!r}, "
                             f"integration_points={values['integration_points']!r}: {exc}") from exc
        return candidate

    def _reproject_canonical(self) -> None:
        settings = self._canonical_settings
        self._candidate_values = {
            "solver": settings.solver,
            "t_start": settings.t_start,
            "t_end": settings.t_end,
            "integration_points": settings.integration_points,
        }
        self._candidate_changed_fields.clear()
        previous_rendering = self._rendering
        self._rendering = True
        try:
            self.solver_method_combo.setCurrentText(settings.solver)
            self.t_start_edit.setText(repr(float(settings.t_start)))
            self.t_end_edit.setText(repr(float(settings.t_end)))
            self.num_points_spinbox.setValue(settings.integration_points)
        finally:
            self._rendering = previous_rendering

    @QtCore.Slot()
    def _mark_candidate_dirty(self, *_args) -> None:
        if self._rendering:
            return
        was_dirty = self._candidate_dirty
        self._candidate_dirty = True
        sender = self.sender()
        if sender is self.solver_method_combo:
            self._candidate_changed_fields.add("solver")
            self._candidate_values["solver"] = str(
                self.solver_method_combo.currentText()
            )
        elif sender is self.t_start_edit:
            self._candidate_changed_fields.add("t_start")
        elif sender is self.t_end_edit:
            self._candidate_changed_fields.add("t_end")
        elif sender is self.num_points_spinbox:
            self._candidate_changed_fields.add("integration_points")
            stored = self._candidate_values.get("integration_points")
            try:
                stored_value = int(stored) if isinstance(stored, str) else None
            except (TypeError, ValueError, OverflowError):
                stored_value = None
            if stored_value != int(self.num_points_spinbox.value()):
                self._candidate_values["integration_points"] = (
                    self.num_points_spinbox.lineEdit().text()
                )
        if not was_dirty:
            self.runDefaultsDraftStateChanged.emit(True)

    @QtCore.Slot(str)
    def _capture_integration_points_text(self, text: str) -> None:
        if self._rendering:
            return
        was_dirty = self._candidate_dirty
        self._candidate_dirty = True
        self._candidate_changed_fields.add("integration_points")
        self._candidate_values["integration_points"] = str(text)
        if not was_dirty:
            self.runDefaultsDraftStateChanged.emit(True)

    def _restore_integration_points_draft(self) -> None:
        """Restore exact authored Points text after Qt editor normalization."""

        if "integration_points" not in self._candidate_changed_fields:
            return
        value = self._candidate_values.get("integration_points")
        if not isinstance(value, str):
            return
        edit = self.num_points_spinbox.lineEdit()
        blocker = QtCore.QSignalBlocker(edit)
        try:
            edit.setText(value)
        finally:
            del blocker

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
        self._sync_run_enabled()

    @QtCore.Slot()
    def _sync_run_enabled(self, *_args) -> None:
        enabled = bool(
            bool(self.batch_table.selected_set_ids())
            and self._run_admitted
        )
        self.run_btn.setEnabled(enabled)
        if enabled != self._last_run_enabled:
            self._last_run_enabled = enabled
            if not self._rendering:
                self.runEnabledChanged.emit(enabled)
