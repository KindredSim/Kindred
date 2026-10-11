"""Complete execution-settings editor for one exact IC set."""

from __future__ import annotations

from kindred.gui.display_name_policy import scientific_symbol_html, scientific_symbol_spoken

from PySide6 import QtCore, QtWidgets

from kindred.core.project_document import (
    TEMPERATURE_RANGE_K,
    SimulationExecutionSettings,
    SUPPORTED_SOLVERS,
    TabScientificSettings,
    parse_points_text,
    validate_simulation_execution_settings,
    validate_tab_scientific_settings,
)
from kindred.core.runtime_defaults import MAX_PARALLEL_WORKERS_CEILING
from kindred.core.validation import parse_finite_numeric_text
from kindred.gui.ui_helpers import SOLVER_TOOLTIP, clamp_initial_size, ignore_wheel, make_scroll_area

__all__ = [
    "SimulationSettingsDialog",
    "SolverSettingsDialog",
]


class _RunSettingsFields:
    """The four Run-setting editors shared by both settings dialogs."""

    def __init__(self, form: QtWidgets.QFormLayout, parent: QtWidgets.QWidget) -> None:
        self.solver_combo = QtWidgets.QComboBox(parent)
        self.solver_combo.addItems(list(SUPPORTED_SOLVERS))
        self.solver_combo.setToolTip(SOLVER_TOOLTIP)
        self.t_start_edit = self._time_edit(parent, "0", "Absolute Run start time")
        self.t_end_edit = self._time_edit(parent, "10", "Absolute Run end time")
        self.points_edit = self._time_edit(
            parent,
            "100",
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

    @staticmethod
    def _time_edit(
        parent: QtWidgets.QWidget, value: str, tooltip: str,
    ) -> QtWidgets.QLineEdit:
        edit = QtWidgets.QLineEdit(str(value), parent)
        edit.setToolTip(str(tooltip))
        edit.setMaximumWidth(190)
        return edit

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
        except (TypeError, ValueError) as exc:
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
        self.temperature_spin = self._positive_spin(
            minimum=TEMPERATURE_RANGE_K[0],
            maximum=TEMPERATURE_RANGE_K[1],
            value=factory.temperature,
            decimals=2,
        )
        self.standard_conc_edit = QtWidgets.QLineEdit(str(factory.standard_conc_M), science_box)
        self.energy_unit_combo = QtWidgets.QComboBox(science_box)
        self.energy_unit_combo.addItems(["J/mol", "kJ/mol", "kcal/mol"])
        self.energy_unit_combo.setCurrentText(factory.energy_unit)
        self.rtol_edit = QtWidgets.QLineEdit("1e-6", science_box)
        self.atol_edit = QtWidgets.QLineEdit("1e-12", science_box)
        self.temperature_spin.setToolTip(
            "Default nominal temperature for this mechanism. An explicit T= directive "
            "overrides it. Temperature schedules start from the resolved nominal temperature."
        )
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
            "Uses the symbolic analytic Jacobian when supported. If any "
            "individual reaction-order or equilibrium concentration exponent "
            "is non-integral, Kindred uses the numerical Jacobian of the solver "
            "from the start. Supported intervention schedules retain analytic "
            "Jacobian handling. If analytic Jacobian evaluation fails after "
            "execution begins, the execution fails and is not retried."
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
        science_form.addRow("Default temperature (K)", self.temperature_spin)
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
            "for every initial-condition set. When enabled, a rule may end integration "
            "before t_end. When disabled, runs continue to t_end. "
            "Fitting always ignores stopping interventions."
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
        self.preview_points_edit = QtWidgets.QLineEdit(str(int(preview_points)), preview_box)
        self.preview_points_edit.setMaximumWidth(190)
        self.preview_delay_spin = self._integer_spin(
            0, 1000, preview_delay_ms, preview_box,
        )
        self.preview_points_edit.setToolTip(
            "Output points for all slider previews, including release and typed changes."
        )
        self.preview_delay_spin.setToolTip(
            "Delay before requesting a preview while dragging any slider. "
            "Release, clicks, and typed changes request a preview immediately."
        )
        preview_form.addRow("Preview points", self.preview_points_edit)
        preview_form.addRow("Preview delay (ms)", self.preview_delay_spin)
        body_layout.addWidget(preview_box)

        parallel_box = QtWidgets.QGroupBox("Parallel Execution", body)
        parallel_form = QtWidgets.QFormLayout(parallel_box)
        self.run_maximum_automatic_check = QtWidgets.QCheckBox(
            "Automatic (CPU Fallback)",
            parallel_box,
        )
        self.run_maximum_spin = self._integer_spin(
            1,
            MAX_PARALLEL_WORKERS_CEILING,
            1,
            parallel_box,
        )
        self.run_maximum_automatic_check.setToolTip(
            "Choose parallel execution capacity from the available logical CPUs, "
            "reserving one when possible and using at most 16 simultaneous simulations."
        )
        self.run_maximum_spin.setToolTip(
            "Maximum number of initial-condition sets simulated simultaneously. "
            "Execution also respects available CPU capacity and the number of sets. "
            "Set to 1 for sequential execution."
        )
        self.run_maximum_automatic_check.toggled.connect(
            self.run_maximum_spin.setDisabled
        )
        parallel_form.addRow("Run user maximum", self.run_maximum_automatic_check)
        parallel_form.addRow("Explicit maximum", self.run_maximum_spin)
        body_layout.addWidget(parallel_box)

        cache_box = QtWidgets.QGroupBox("Result Cache", body)
        cache_form = QtWidgets.QFormLayout(cache_box)
        self.cache_capacity_spin = self._integer_spin(
            1,
            (1 << 31) - 1,
            32,
            cache_box,
        )
        self.cache_capacity_spin.setToolTip(
            "Number of past simulation results kept in memory per mechanism tab. "
            "Matching results can be restored when enabled for plotting. Larger "
            "values use more memory. When the cache is full, the least recently "
            "used results are removed."
        )
        cache_form.addRow(
            "Explicit-result cache capacity",
            self.cache_capacity_spin,
        )
        body_layout.addWidget(cache_box)
        scroll.setWidget(body)
        layout.addWidget(scroll, 1)
        for control in (
            self.temperature_spin, self.energy_unit_combo, self._run_default_fields.solver_combo,
            self.preview_delay_spin, self.run_maximum_spin, self.cache_capacity_spin,
        ):
            ignore_wheel(control)

        for form in (science_form, run_form, preview_form, parallel_form, cache_form):
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

    @staticmethod
    def _integer_spin(
        minimum: int,
        maximum: int,
        value: int,
        parent: QtWidgets.QWidget,
    ) -> QtWidgets.QSpinBox:
        spin = QtWidgets.QSpinBox(parent)
        spin.setRange(int(minimum), int(maximum))
        spin.setValue(int(value))
        return spin

    @staticmethod
    def _positive_spin(
        *,
        minimum: float,
        maximum: float,
        value: float,
        decimals: int,
    ) -> QtWidgets.QDoubleSpinBox:
        spin = QtWidgets.QDoubleSpinBox()
        spin.setDecimals(int(decimals))
        spin.setRange(float(minimum), float(maximum))
        spin.setValue(float(value))
        spin.setMaximumWidth(190)
        return spin

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
        self.temperature_spin.setValue(settings.temperature)
        self.standard_conc_edit.setText(repr(float(settings.standard_conc_M)))
        self.energy_unit_combo.setCurrentText(settings.energy_unit)
        self.rtol_edit.setText(repr(float(settings.rtol)))
        self.atol_edit.setText(repr(float(settings.atol)))
        self.analytic_jacobian_check.setChecked(settings.analytic_jacobian_policy)
        self.use_authored_stops_check.setChecked(settings.use_authored_stops)

    def scientific_settings(self) -> TabScientificSettings:
        candidate = TabScientificSettings(
            temperature=float(self.temperature_spin.value()),
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
            self.preview_delay_spin.value(),
        )

    def set_run_user_maximum(self, value: int | None) -> None:
        self.run_maximum_spin.setRange(1, MAX_PARALLEL_WORKERS_CEILING)
        automatic = value is None
        blocker = QtCore.QSignalBlocker(self.run_maximum_automatic_check)
        try:
            self.run_maximum_automatic_check.setChecked(automatic)
        finally:
            del blocker
        self.run_maximum_spin.setDisabled(automatic)
        if not automatic:
            self.run_maximum_spin.setValue(int(value))

    def run_user_maximum(self) -> int | None:
        if self.run_maximum_automatic_check.isChecked():
            return None
        return int(self.run_maximum_spin.value())

    def set_result_cache_capacity(self, value: int) -> None:
        self.cache_capacity_spin.setValue(int(value))

    def result_cache_capacity(self) -> int:
        return int(self.cache_capacity_spin.value())

    @QtCore.Slot()
    def _accept_candidate(self) -> None:
        try:
            self.scientific_settings()
            self.new_tab_run_defaults()
            self.preview_settings()
            self.run_user_maximum()
            self.result_cache_capacity()
        except (TypeError, ValueError, OverflowError) as exc:
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid Simulation Settings",
                str(exc),
            )
            return
        self.accept()
