"""Complete four-field execution-settings editor for one exact IC set."""

from __future__ import annotations

import math

from PySide6 import QtCore, QtWidgets

from kindred.core.project_document import (
    TEMPERATURE_RANGE_K,
    SIMULATION_POINT_RANGE,
    SimulationExecutionSettings,
    SUPPORTED_SOLVERS,
    TabScientificSettings,
    validate_simulation_execution_settings,
    validate_tab_scientific_settings,
)
from kindred.core.runtime_defaults import MAX_PARALLEL_WORKERS_CEILING
from kindred.gui.ui_helpers import setup_scientific_validator

__all__ = [
    "SimulationSettingsDialog",
    "SolverSettingsDialog",
]


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
        self.solver_combo = QtWidgets.QComboBox(self)
        self.solver_combo.addItems(list(SUPPORTED_SOLVERS))
        self.t_start_edit = self._time_edit(
            "0",
            "Absolute Run start time",
        )
        self.t_end_edit = self._time_edit(
            "10",
            "Absolute Run end time",
        )
        self.integration_points_spin = QtWidgets.QSpinBox(self)
        self.integration_points_spin.setRange(*SIMULATION_POINT_RANGE)
        self.integration_points_spin.setSingleStep(10)
        for label, widget in (
            ("Solver", self.solver_combo),
            ("t_start", self.t_start_edit),
            ("t_end", self.t_end_edit),
            ("Integration points", self.integration_points_spin),
        ):
            form.addRow(label, widget)
        layout.addLayout(form)

        self._buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        self._buttons.accepted.connect(self._accept_candidate)
        self._buttons.rejected.connect(self.reject)
        layout.addWidget(self._buttons)

    def _time_edit(self, value: str, tooltip: str) -> QtWidgets.QLineEdit:
        edit = QtWidgets.QLineEdit(str(value), self)
        edit.setToolTip(str(tooltip))
        edit.setMaximumWidth(190)
        setup_scientific_validator(edit)
        return edit

    def set_execution_settings(self, settings: SimulationExecutionSettings) -> None:
        validate_simulation_execution_settings(settings)
        self.solver_combo.setCurrentText(settings.solver)
        blockers = (
            QtCore.QSignalBlocker(self.t_start_edit),
            QtCore.QSignalBlocker(self.t_end_edit),
        )
        try:
            self.t_start_edit.setText(repr(float(settings.t_start)))
            self.t_end_edit.setText(repr(float(settings.t_end)))
        finally:
            del blockers
        self.integration_points_spin.setValue(settings.integration_points)

    @staticmethod
    def _parse_time_edit(edit: QtWidgets.QLineEdit, field: str) -> float:
        try:
            value = float(edit.text().strip())
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"{field} must be a finite, nonnegative number."
            ) from exc
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(
                f"{field} must be a finite, nonnegative number."
            )
        return value

    def execution_settings(self) -> SimulationExecutionSettings:
        candidate = SimulationExecutionSettings(
            solver=str(self.solver_combo.currentText()),
            t_start=self._parse_time_edit(self.t_start_edit, "t_start"),
            t_end=self._parse_time_edit(self.t_end_edit, "t_end"),
            integration_points=int(self.integration_points_spin.value()),
        )
        validate_simulation_execution_settings(candidate)
        return candidate

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
    ) -> None:
        super().__init__(parent)
        self.setObjectName("simulationSettingsDialog")
        self.setWindowTitle("Simulation Settings")
        self.setModal(True)

        layout = QtWidgets.QVBoxLayout(self)
        science_box = QtWidgets.QGroupBox("Tab scientific", self)
        science_form = QtWidgets.QFormLayout(science_box)
        self.temperature_spin = self._positive_spin(
            minimum=TEMPERATURE_RANGE_K[0],
            maximum=TEMPERATURE_RANGE_K[1],
            value=298.15,
            decimals=8,
        )
        self.rtol_edit = QtWidgets.QLineEdit("1e-6", science_box)
        self.atol_edit = QtWidgets.QLineEdit("1e-12", science_box)
        setup_scientific_validator(self.rtol_edit)
        setup_scientific_validator(self.atol_edit)
        self.analytic_jacobian_check = QtWidgets.QCheckBox(
            "Use analytic Jacobian",
            science_box,
        )
        self.scientific_defaults_check = QtWidgets.QCheckBox(
            "Use these settings as defaults for new projects and tabs.",
            science_box,
        )
        self.scientific_defaults_check.setChecked(False)
        science_form.addRow("Temperature (K)", self.temperature_spin)
        science_form.addRow("Relative tolerance", self.rtol_edit)
        science_form.addRow("Absolute tolerance", self.atol_edit)
        science_form.addRow("", self.analytic_jacobian_check)
        science_form.addRow("", self.scientific_defaults_check)
        layout.addWidget(science_box)

        preview_box = QtWidgets.QGroupBox("Preview", self)
        preview_form = QtWidgets.QFormLayout(preview_box)
        self.parameter_debounce_spin = self._integer_spin(
            0,
            1000,
            80,
            preview_box,
        )
        self.equilibrium_debounce_spin = self._integer_spin(
            0,
            1000,
            150,
            preview_box,
        )
        preview_form.addRow(
            "Parameter debounce (ms)",
            self.parameter_debounce_spin,
        )
        preview_form.addRow(
            "Equilibrium debounce (ms)",
            self.equilibrium_debounce_spin,
        )
        layout.addWidget(preview_box)

        parallel_box = QtWidgets.QGroupBox("Parallel execution", self)
        parallel_form = QtWidgets.QFormLayout(parallel_box)
        self.run_maximum_automatic_check = QtWidgets.QCheckBox(
            "Automatic (CPU fallback)",
            parallel_box,
        )
        self.run_maximum_spin = self._integer_spin(
            1,
            MAX_PARALLEL_WORKERS_CEILING,
            1,
            parallel_box,
        )
        self.run_maximum_automatic_check.toggled.connect(
            self.run_maximum_spin.setDisabled
        )
        parallel_form.addRow("Run user maximum", self.run_maximum_automatic_check)
        parallel_form.addRow("Explicit maximum", self.run_maximum_spin)
        layout.addWidget(parallel_box)

        cache_box = QtWidgets.QGroupBox("Result cache", self)
        cache_form = QtWidgets.QFormLayout(cache_box)
        self.cache_capacity_spin = self._integer_spin(
            1,
            (1 << 31) - 1,
            32,
            cache_box,
        )
        cache_form.addRow(
            "Explicit-result cache capacity",
            self.cache_capacity_spin,
        )
        layout.addWidget(cache_box)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        buttons.accepted.connect(self._accept_candidate)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

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
        spin.setDecimals(max(17, int(decimals)))
        spin.setRange(float(minimum), float(maximum))
        spin.setValue(float(value))
        spin.setMaximumWidth(190)
        return spin

    def set_scientific_settings(self, settings: TabScientificSettings) -> None:
        validate_tab_scientific_settings(settings)
        self.temperature_spin.setValue(settings.temperature)
        self.rtol_edit.setText(repr(float(settings.rtol)))
        self.atol_edit.setText(repr(float(settings.atol)))
        self.analytic_jacobian_check.setChecked(settings.analytic_jacobian_policy)

    def scientific_settings(self) -> TabScientificSettings:
        try:
            rtol = float(self.rtol_edit.text())
            atol = float(self.atol_edit.text())
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("rtol and atol must be numbers.") from exc
        candidate = TabScientificSettings(
            temperature=float(self.temperature_spin.value()),
            rtol=rtol,
            atol=atol,
            analytic_jacobian_policy=bool(self.analytic_jacobian_check.isChecked()),
        )
        return validate_tab_scientific_settings(candidate)

    def persist_scientific_defaults_requested(self) -> bool:
        """Return the passive application-default persistence intent."""

        return bool(self.scientific_defaults_check.isChecked())

    def set_preview_debounce(self, parameter: int, equilibrium: int) -> None:
        self.parameter_debounce_spin.setValue(int(parameter))
        self.equilibrium_debounce_spin.setValue(int(equilibrium))

    def preview_debounce(self) -> tuple[int, int]:
        return (
            int(self.parameter_debounce_spin.value()),
            int(self.equilibrium_debounce_spin.value()),
        )

    def set_run_user_maximum(self, value: int | None) -> None:
        self.run_maximum_spin.setRange(1, MAX_PARALLEL_WORKERS_CEILING)
        automatic = value is None
        if not automatic and type(value) is not int:
            raise TypeError("Run user maximum must be an integer or None.")
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
            self.preview_debounce()
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
