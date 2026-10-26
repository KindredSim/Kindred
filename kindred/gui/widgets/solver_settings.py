"""Complete execution-settings editor for one exact IC set."""

from __future__ import annotations

from kindred.gui.display_name_policy import scientific_symbol_html, scientific_symbol_spoken

from PySide6 import QtCore, QtWidgets

from kindred.core.project_document import (
    SimulationExecutionSettings,
    SUPPORTED_SOLVERS,
    TabScientificSettings,
    parse_points_text,
    validate_simulation_execution_settings,
    validate_tab_scientific_settings,
)
from kindred.core.validation import parse_finite_numeric_text, parse_whole_number_text
from kindred.gui.ui_helpers import SOLVER_TOOLTIP, clamp_initial_size, ignore_wheel, make_scroll_area

__all__ = [
    "SimulationSettingsDialog",
    "SolverSettingsDialog",
]


def _number_edit(
    parent: QtWidgets.QWidget, value: str, tooltip: str,
) -> QtWidgets.QLineEdit:
    edit = QtWidgets.QLineEdit(str(value), parent)
    edit.setToolTip(str(tooltip))
    edit.setMaximumWidth(190)
    return edit


class _RunSettingsFields:
    """The four Run-setting editors shared by both settings dialogs."""

    def __init__(self, form: QtWidgets.QFormLayout, parent: QtWidgets.QWidget) -> None:
        self.solver_combo = QtWidgets.QComboBox(parent)
        self.solver_combo.addItems(list(SUPPORTED_SOLVERS))
        self.solver_combo.setToolTip(SOLVER_TOOLTIP)
        self.t_start_edit = _number_edit(parent, "", "Absolute Run start time")
        self.t_end_edit = _number_edit(parent, "", "Absolute Run end time")
        self.points_edit = _number_edit(
            parent,
            "",
            "Number of evenly spaced base samples over the completed interval, "
            "including both endpoints. Intervention boundary rows are additional. "
            "A stop at the start returns one sample.",
        )
        for label, widget in (
            ("Solver", self.solver_combo),
            ("t_start", self.t_start_edit),
            ("t_end", self.t_end_edit),
            ("Points", self.points_edit),
        ):
            form.addRow(label, widget)
            name_label = form.labelForField(widget)
            if label in {"t_start", "t_end"}:
                name_label.setTextFormat(QtCore.Qt.RichText)
                name_label.setText(scientific_symbol_html(label, family="time", role=label))
            widget.setAccessibleName(scientific_symbol_spoken(label, family="time", role=label))

    def set_settings(self, settings: SimulationExecutionSettings) -> None:
        self.solver_combo.setCurrentText(settings.solver)
        self.t_start_edit.setText(repr(float(settings.t_start)))
        self.t_end_edit.setText(repr(float(settings.t_end)))
        self.points_edit.setText(str(settings.integration_points))

    def settings(self) -> SimulationExecutionSettings:
        t_start, t_end = self.t_start_edit.text(), self.t_end_edit.text()
        candidate = SimulationExecutionSettings(
            solver=str(self.solver_combo.currentText()),
            t_start=parse_finite_numeric_text(t_start, field=f"Start time {t_start!r}"),
            t_end=parse_finite_numeric_text(t_end, field=f"End time {t_end!r}"),
            integration_points=parse_points_text(self.points_edit.text()),
        )
        return validate_simulation_execution_settings(candidate)


class SolverSettingsDialog(QtWidgets.QDialog):
    """Edit one complete override; partial values never leave the dialog."""

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        *,
        title: str = "Custom Run Settings",
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(str(title))
        self.setModal(True)

        layout = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()
        self._fields = _RunSettingsFields(form, self)
        layout.addLayout(form)

        self._buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        self._buttons.accepted.connect(self._accept_candidate)
        self._buttons.rejected.connect(self.reject)
        layout.addWidget(self._buttons)

    def set_execution_settings(self, settings: SimulationExecutionSettings) -> None:
        self._fields.set_settings(settings)

    def execution_settings(self) -> SimulationExecutionSettings:
        return self._fields.settings()

    @QtCore.Slot()
    def _accept_candidate(self) -> None:
        try:
            self.execution_settings()
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid Run Settings",
                str(exc),
            )
            return
        self.accept()


class SimulationSettingsDialog(QtWidgets.QDialog):
    """Passive editor for the supported application and tab settings.

    The dialog stores only widget draft state.  MainWindow publishes the
    accepted values through the existing tab-action and ConfigController
    owners after the modal returns.
    """

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        *,
        preview_points: int,
        preview_delay_ms: int,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("simulationSettingsDialog")
        self.setWindowTitle("Simulation Settings")
        self.setModal(True)

        layout = QtWidgets.QVBoxLayout(self)
        scroll = make_scroll_area(self)
        scroll.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        body = QtWidgets.QWidget(scroll)
        body_layout = QtWidgets.QVBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        science_box = QtWidgets.QGroupBox("Mechanism Settings", body)
        science_form = QtWidgets.QFormLayout(science_box)
        factory = TabScientificSettings()
        self.temperature_edit = _number_edit(
            science_box,
            repr(float(factory.temperature)),
            "Default nominal temperature for this mechanism. An explicit T= directive "
            "overrides it. A temperature schedule uses it before its first listed time, "
            "and a response schedule starts from it.",
        )
        self.standard_conc_edit = QtWidgets.QLineEdit(str(factory.standard_conc_M), science_box)
        self.energy_unit_combo = QtWidgets.QComboBox(science_box)
        self.energy_unit_combo.addItems(["J/mol", "kJ/mol", "kcal/mol"])
        self.energy_unit_combo.setCurrentText(factory.energy_unit)
        self.rtol_edit = QtWidgets.QLineEdit("1e-6", science_box)
        self.atol_edit = QtWidgets.QLineEdit("1e-12", science_box)
        self.standard_conc_edit.setToolTip(
            "Default standard-state concentration in M for Eyring rates and thermodynamic "
            "equilibrium constants. An explicit C0= directive overrides it."
        )
        self.energy_unit_combo.setToolTip(
            "Default units for authored energies. An explicit energy= directive overrides "
            "this setting. Changing the units changes how existing energy numbers are interpreted."
        )
        self.rtol_edit.setToolTip(
            "Controls solver error relative to the magnitude of each concentration."
        )
        self.atol_edit.setToolTip(
            "Controls solver error when concentrations are near zero."
        )
        self.analytic_jacobian_check = QtWidgets.QCheckBox(
            "Prefer Analytic Jacobian When Supported",
            science_box,
        )
        self.analytic_jacobian_check.setToolTip(
            "Uses the symbolic analytic Jacobian when the mechanism supports it, "
            "otherwise the solver's numerical Jacobian."
        )
        self.scientific_defaults_check = QtWidgets.QCheckBox(
            "Use These Settings as Defaults for New Projects and Tabs",
            science_box,
        )
        self.scientific_defaults_check.setChecked(False)
        self.scientific_defaults_check.setToolTip(
            "Save all settings in this group, including the Jacobian and stopping "
            "preferences, as defaults for new projects and tabs. "
            "Existing tabs keep their settings."
        )
        science_form.addRow("Default temperature (K)", self.temperature_edit)
        science_form.addRow("Default standard concentration (M)", self.standard_conc_edit)
        science_form.addRow("Default energy units", self.energy_unit_combo)
        science_form.addRow("Relative tolerance", self.rtol_edit)
        science_form.addRow("Absolute tolerance", self.atol_edit)
        science_form.addRow("", self.analytic_jacobian_check)

        self.use_authored_stops_check = QtWidgets.QCheckBox(
            "Use Authored Stopping Interventions", science_box
        )
        self.use_authored_stops_check.setToolTip(
            "Apply stopping interventions authored in this mechanism to Run and Preview "
            "for every initial-condition set, and to Fit datasets with a measured X. "
            "When enabled, a rule may end integration before t_end. When disabled, "
            "runs continue to t_end. Fitting against time always ignores stopping interventions."
        )
        self._stopping_declarations: tuple[str, ...] = ()
        self.stopping_info_button = QtWidgets.QPushButton("Authored Rules…", science_box)
        self.stopping_info_button.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed,
        )
        self.stopping_info_button.clicked.connect(self._show_stopping_declarations)
        stopping_row = QtWidgets.QHBoxLayout()
        stopping_row.addWidget(self.use_authored_stops_check)
        stopping_row.addWidget(self.stopping_info_button)
        stopping_row.addStretch()
        science_form.addRow("", stopping_row)
        science_form.addRow("", self.scientific_defaults_check)
        self.set_stopping_declarations(())
        body_layout.addWidget(science_box)

        run_box = QtWidgets.QGroupBox("Run Defaults for New Projects and Tabs", body)
        run_box.setToolTip(
            "Run settings that new projects and tabs start with. Existing tabs "
            "keep their settings. Edit those in the Run panel of each tab."
        )
        run_form = QtWidgets.QFormLayout(run_box)
        self._run_default_fields = _RunSettingsFields(run_form, run_box)
        body_layout.addWidget(run_box)

        preview_box = QtWidgets.QGroupBox("Preview", body)
        preview_form = QtWidgets.QFormLayout(preview_box)
        self.preview_points_edit = _number_edit(
            preview_box,
            str(int(preview_points)),
            "Output points for all slider previews, including release and typed changes.",
        )
        self.preview_delay_edit = _number_edit(
            preview_box,
            str(int(preview_delay_ms)),
            "Delay before requesting a preview while dragging any slider. "
            "Release, clicks, and typed changes request a preview immediately.",
        )
        preview_form.addRow("Preview points", self.preview_points_edit)
        preview_form.addRow("Preview delay (ms)", self.preview_delay_edit)
        body_layout.addWidget(preview_box)

        parallel_box = QtWidgets.QGroupBox("Parallel Execution", body)
        parallel_form = QtWidgets.QFormLayout(parallel_box)
        self.parallel_automatic_check = QtWidgets.QCheckBox(
            "Automatic",
            parallel_box,
        )
        self.parallel_automatic_check.setToolTip(
            "Run one simulation fewer at once than the machine has physical cores."
        )
        # The box and its "/ cores" label share one tooltip on their row.
        parallel_row = QtWidgets.QWidget(parallel_box)
        parallel_row.setToolTip(
            "How many simulations Run, Fit and Chemical Space Scan may run at once, "
            "from 1 to the machine's physical cores."
        )
        self.parallel_maximum_edit = _number_edit(parallel_row, "", "")
        self._parallel_cores_label = QtWidgets.QLabel(parallel_row)
        parallel_row_layout = QtWidgets.QHBoxLayout(parallel_row)
        parallel_row_layout.setContentsMargins(0, 0, 0, 0)
        parallel_row_layout.addWidget(self.parallel_maximum_edit)
        parallel_row_layout.addWidget(self._parallel_cores_label)
        parallel_row_layout.addStretch(1)
        self._physical_cores = 1
        self._automatic_parallel_workers = 1
        self.parallel_automatic_check.toggled.connect(self._show_parallel_automatic)
        parallel_form.addRow("Simultaneous simulations", self.parallel_automatic_check)
        parallel_form.addRow("Explicit maximum", parallel_row)
        body_layout.addWidget(parallel_box)
        scroll.setWidget(body)
        layout.addWidget(scroll, 1)
        for control in (self.energy_unit_combo, self._run_default_fields.solver_combo):
            ignore_wheel(control)

        for form in (science_form, run_form, preview_form, parallel_form):
            for row in range(form.rowCount()):
                field = form.itemAt(row, QtWidgets.QFormLayout.ItemRole.FieldRole).widget()
                label = form.labelForField(field)
                if label is not None:
                    label.setToolTip(field.toolTip())

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setToolTip(
            "Apply these settings and close."
        )
        buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Cancel).setToolTip(
            "Close without applying changes."
        )
        buttons.accepted.connect(self._accept_candidate)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        # Scroll only vertically, and open at the groups' natural height when
        # the screen allows; QScrollArea caps its own size hint.
        self.ensurePolished()  # the theme style sheet sets the scroll frame
        chrome = 2 * scroll.frameWidth()
        scroll.setMinimumWidth(
            body.minimumSizeHint().width()
            + scroll.verticalScrollBar().sizeHint().width()
            + chrome
        )
        clamp_initial_size(
            self,
            self.sizeHint().width(),
            self.sizeHint().height() - scroll.sizeHint().height()
            + body.sizeHint().height() + chrome,
        )

    def set_stopping_declarations(self, declarations: tuple[str, ...]) -> None:
        self._stopping_declarations = declarations
        self.stopping_info_button.setEnabled(bool(declarations))

    def _show_stopping_declarations(self) -> None:
        message = QtWidgets.QMessageBox(self)
        message.setWindowTitle("Authored Stopping Interventions")
        message.setTextFormat(QtCore.Qt.PlainText)
        message.setText("\n".join(self._stopping_declarations))
        message.setStandardButtons(QtWidgets.QMessageBox.Ok)
        message.exec()

    def set_scientific_settings(self, settings: TabScientificSettings) -> None:
        self.temperature_edit.setText(repr(float(settings.temperature)))
        self.standard_conc_edit.setText(repr(float(settings.standard_conc_M)))
        self.energy_unit_combo.setCurrentText(settings.energy_unit)
        self.rtol_edit.setText(repr(float(settings.rtol)))
        self.atol_edit.setText(repr(float(settings.atol)))
        self.analytic_jacobian_check.setChecked(settings.analytic_jacobian_policy)
        self.use_authored_stops_check.setChecked(settings.use_authored_stops)

    def scientific_settings(self) -> TabScientificSettings:
        candidate = TabScientificSettings(
            temperature=parse_finite_numeric_text(
                self.temperature_edit.text(), field="Default temperature"
            ),
            standard_conc_M=parse_finite_numeric_text(
                self.standard_conc_edit.text(), field="Default standard concentration"
            ),
            energy_unit=self.energy_unit_combo.currentText(),
            rtol=parse_finite_numeric_text(
                self.rtol_edit.text(), field="Relative tolerance"
            ),
            atol=parse_finite_numeric_text(
                self.atol_edit.text(), field="Absolute tolerance"
            ),
            analytic_jacobian_policy=bool(self.analytic_jacobian_check.isChecked()),
            use_authored_stops=self.use_authored_stops_check.isChecked(),
        )
        return validate_tab_scientific_settings(candidate)

    def persist_scientific_defaults_requested(self) -> bool:
        """Return the passive application-default persistence intent."""

        return bool(self.scientific_defaults_check.isChecked())

    def set_new_tab_run_defaults(self, settings: SimulationExecutionSettings) -> None:
        self._run_default_fields.set_settings(settings)

    def new_tab_run_defaults(self) -> SimulationExecutionSettings:
        return self._run_default_fields.settings()

    def preview_settings(self) -> tuple[int, int]:
        return (
            parse_points_text(self.preview_points_edit.text(), field="Preview points"),
            parse_whole_number_text(
                self.preview_delay_edit.text(), field="Preview delay (ms)", minimum=0, maximum=1000,
            ),
        )

    def set_parallel_maximum(
        self,
        value: int | None,
        *,
        physical_cores: int,
        automatic_workers: int,
    ) -> None:
        self._physical_cores = physical_cores
        self._parallel_cores_label.setText(f"/ {physical_cores}")
        self._automatic_parallel_workers = automatic_workers
        automatic = value is None
        blocker = QtCore.QSignalBlocker(self.parallel_automatic_check)
        try:
            self.parallel_automatic_check.setChecked(automatic)
        finally:
            del blocker
        self._show_parallel_automatic(automatic)
        if not automatic:
            self.parallel_maximum_edit.setText(str(value))

    def _show_parallel_automatic(self, automatic: bool) -> None:
        self.parallel_maximum_edit.setDisabled(automatic)
        # While Automatic is ticked the box shows the count it chooses.
        if automatic:
            self.parallel_maximum_edit.setText(str(self._automatic_parallel_workers))

    def parallel_maximum(self) -> int | None:
        if self.parallel_automatic_check.isChecked():
            return None
        return parse_whole_number_text(
            self.parallel_maximum_edit.text(),
            field="Explicit maximum",
            minimum=1,
            maximum=self._physical_cores,
        )

    @QtCore.Slot()
    def _accept_candidate(self) -> None:
        try:
            self.scientific_settings()
            self.new_tab_run_defaults()
            self.preview_settings()
            self.parallel_maximum()
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid Simulation Settings",
                str(exc),
            )
            return
        self.accept()
