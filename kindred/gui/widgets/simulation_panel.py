"""Initial Conditions panel with one canonical Run-settings candidate."""

from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

from kindred.core.project_document import (
    SimulationExecutionSettings,
    validate_simulation_execution_settings,
)
from kindred.gui.widgets.batch_initial_conditions_table import (
    BatchInitialConditionsTableModel,
    BatchInitialConditionsTableView,
)

__all__ = ["SimulationPanel"]


class SimulationPanel(QtWidgets.QWidget):
    addSetRequested = QtCore.Signal()
    moveSelectedRequested = QtCore.Signal(int)
    deleteSelectedRequested = QtCore.Signal()
    runSelectedRequested = QtCore.Signal()
    stopRequested = QtCore.Signal()
    workspaceRunDefaultsRequested = QtCore.Signal(object)
    defaultsEditStarted = QtCore.Signal()
    settingsValidationFailed = QtCore.Signal(str)
    runEnabledChanged = QtCore.Signal(bool)

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
        self._run_admitted = True
        self._simulation_job_kind: str | None = None
        self._last_run_enabled: bool | None = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        controls = QtWidgets.QHBoxLayout()
        self.solver_method_combo = QtWidgets.QComboBox(self)
        self.solver_method_combo.addItems(["Radau", "BDF"])
        self.t_start_edit = self._number_edit("0", "Absolute Run start time")
        self.t_end_edit = self._number_edit("10", "Absolute Run end time")
        self.num_points_spinbox = QtWidgets.QSpinBox(self)
        self.num_points_spinbox.setRange(10, 100_000)
        self.num_points_spinbox.setSingleStep(10)
        for label, widget in (
            ("Solver:", self.solver_method_combo),
            ("t_start:", self.t_start_edit),
            ("t_end:", self.t_end_edit),
            ("Points:", self.num_points_spinbox),
        ):
            controls.addWidget(QtWidgets.QLabel(label, self))
            controls.addWidget(widget)
        controls.addStretch(1)
        layout.addLayout(controls)

        self.solver_method_combo.currentTextChanged.connect(self._mark_candidate_dirty)
        self.t_start_edit.textEdited.connect(self._mark_candidate_dirty)
        self.t_end_edit.textEdited.connect(self._mark_candidate_dirty)
        self.num_points_spinbox.valueChanged.connect(self._mark_candidate_dirty)
        self.solver_method_combo.activated.connect(self._settle_after_edit)
        self.t_start_edit.editingFinished.connect(self._settle_after_edit)
        self.t_end_edit.editingFinished.connect(self._settle_after_edit)
        self.num_points_spinbox.editingFinished.connect(self._settle_after_edit)

        row_actions = QtWidgets.QHBoxLayout()
        self.add_batch_set_btn = self._button("Add Set", "addBatchSetButton")
        self.move_batch_up_btn = self._button("Move Up", "moveBatchSetUpButton")
        self.move_batch_down_btn = self._button("Move Down", "moveBatchSetDownButton")
        self.delete_batch_set_btn = self._button(
            "Delete Set(s)\N{HORIZONTAL ELLIPSIS}", "deleteBatchSetButton"
        )
        self.run_btn = self._button("Run", "runSelectedButton")
        self.stop_btn = self._button("Stop", "stopButton")
        for button in (
            self.add_batch_set_btn,
            self.move_batch_up_btn,
            self.move_batch_down_btn,
            self.delete_batch_set_btn,
            self.run_btn,
            self.stop_btn,
        ):
            row_actions.addWidget(button)
        row_actions.addStretch(1)
        layout.addLayout(row_actions)

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

        self.sim_progress = QtWidgets.QProgressBar(self)
        self.sim_progress.setVisible(False)
        layout.addWidget(self.sim_progress)
        self._sync_controls()

    @staticmethod
    def _button(text: str, object_name: str) -> QtWidgets.QPushButton:
        button = QtWidgets.QPushButton(text)
        button.setObjectName(object_name)
        return button

    def _number_edit(self, value: str, tooltip: str) -> QtWidgets.QLineEdit:
        edit = QtWidgets.QLineEdit(value, self)
        edit.setMaximumWidth(120)
        edit.setToolTip(tooltip)
        validator = QtGui.QDoubleValidator(0.0, 1e300, 12, edit)
        validator.setNotation(QtGui.QDoubleValidator.Notation.ScientificNotation)
        edit.setValidator(validator)
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

    def has_unsettled_run_defaults(self) -> bool:
        return bool(self._candidate_dirty)

    def run_is_enabled(self) -> bool:
        """Shared local truth for every Run entry point."""
        return bool(self.run_btn.isEnabled())

    def settle_run_defaults(self) -> bool:
        """Commit one complete candidate or reproject canonical and abort once."""
        self.batch_table.settle_active_editor()
        if not self._candidate_dirty:
            return True
        try:
            candidate = self.current_settings_candidate()
        except (TypeError, ValueError) as exc:
            self._candidate_dirty = False
            self._reproject_canonical()
            self.settingsValidationFailed.emit(str(exc))
            return False
        self._candidate_dirty = False
        if candidate != self._canonical_settings:
            self.workspaceRunDefaultsRequested.emit(candidate)
        return True

    def current_settings_candidate(self) -> SimulationExecutionSettings:
        try:
            t_start = float(self.t_start_edit.text())
            t_end = float(self.t_end_edit.text())
        except ValueError as exc:
            raise ValueError("Run start and end times must be numeric.") from exc
        candidate = SimulationExecutionSettings(
            solver=str(self.solver_method_combo.currentText()),
            t_start=t_start,
            t_end=t_end,
            integration_points=int(self.num_points_spinbox.value()),
        )
        validate_simulation_execution_settings(candidate)
        return candidate

    def _reproject_canonical(self) -> None:
        settings = self._canonical_settings
        previous_rendering = self._rendering
        self._rendering = True
        try:
            self.solver_method_combo.setCurrentText(settings.solver)
            self.t_start_edit.setText(f"{settings.t_start:.12g}")
            self.t_end_edit.setText(f"{settings.t_end:.12g}")
            self.num_points_spinbox.setValue(settings.integration_points)
        finally:
            self._rendering = previous_rendering

    @QtCore.Slot()
    def _mark_candidate_dirty(self, *_args) -> None:
        if self._rendering:
            return
        first_edit = not self._candidate_dirty
        self._candidate_dirty = True
        if first_edit:
            self.defaultsEditStarted.emit()

    @QtCore.Slot()
    def _settle_after_edit(self, *_args) -> None:
        if not self._rendering:
            self.settle_run_defaults()

    @QtCore.Slot()
    def _request_run(self) -> None:
        if self.settle_run_defaults():
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
