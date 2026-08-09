"""Complete four-field execution-settings editor for one exact IC set."""

from __future__ import annotations

import math

from PySide6 import QtCore, QtWidgets

from kindred.core.project_document import (
    SimulationExecutionSettings,
    SUPPORTED_SOLVERS,
    TabScientificSettings,
    validate_simulation_execution_settings,
    validate_tab_scientific_settings,
)
from kindred.gui.ui_helpers import setup_scientific_validator

__all__ = ["SolverSettingsDialog", "TabScientificSettingsDialog"]


class SolverSettingsDialog(QtWidgets.QDialog):
    """Edit one complete override; partial values never leave the dialog."""

    settingsAccepted = QtCore.Signal(object)

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
        self.integration_points_spin.setRange(10, 100_000)
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
            candidate = self.execution_settings()
        except (TypeError, ValueError) as exc:
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid Run Settings",
                str(exc),
            )
            return
        self.settingsAccepted.emit(candidate)
        self.accept()


class TabScientificSettingsDialog(QtWidgets.QDialog):
    """Edit tab-wide science separately from per-set execution overrides."""

    settingsAccepted = QtCore.Signal(object)

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Tab Scientific Settings")
        self.setModal(True)

        layout = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()
        self.temperature_spin = self._positive_spin(
            minimum=0.1,
            maximum=10_000.0,
            value=298.15,
            decimals=8,
        )
        self.rtol_edit = QtWidgets.QLineEdit("1e-6", self)
        self.atol_edit = QtWidgets.QLineEdit("1e-12", self)
        setup_scientific_validator(self.rtol_edit)
        setup_scientific_validator(self.atol_edit)
        self.analytic_jacobian_check = QtWidgets.QCheckBox(
            "Use analytic Jacobian",
            self,
        )
        self.wegscheider_check = QtWidgets.QCheckBox(
            "Enforce Wegscheider cyclicity",
            self,
        )
        form.addRow("Temperature (K)", self.temperature_spin)
        form.addRow("Relative tolerance", self.rtol_edit)
        form.addRow("Absolute tolerance", self.atol_edit)
        form.addRow("", self.analytic_jacobian_check)
        form.addRow("", self.wegscheider_check)
        layout.addLayout(form)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        buttons.accepted.connect(self._accept_candidate)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _positive_spin(
        self,
        *,
        minimum: float,
        maximum: float,
        value: float,
        decimals: int,
    ) -> QtWidgets.QDoubleSpinBox:
        spin = QtWidgets.QDoubleSpinBox(self)
        # Never let a display precision choice rewrite the accepted value.
        spin.setDecimals(max(17, int(decimals)))
        spin.setRange(float(minimum), float(maximum))
        spin.setValue(float(value))
        spin.setMaximumWidth(190)
        return spin

    def set_scientific_settings(self, settings: TabScientificSettings) -> None:
        validate_tab_scientific_settings(settings)
        blockers = tuple(
            QtCore.QSignalBlocker(widget)
            for widget in (self.temperature_spin, self.rtol_edit, self.atol_edit)
        )
        try:
            self.temperature_spin.setValue(settings.temperature)
            self.rtol_edit.setText(repr(float(settings.rtol)))
            self.atol_edit.setText(repr(float(settings.atol)))
        finally:
            del blockers
        self.analytic_jacobian_check.setChecked(settings.analytic_jacobian_policy)
        self.wegscheider_check.setChecked(settings.wegscheider_policy)

    def scientific_settings(self) -> TabScientificSettings:
        candidate = TabScientificSettings(
            temperature=float(self.temperature_spin.value()),
            rtol=self._positive_scientific_text(self.rtol_edit, "rtol"),
            atol=self._positive_scientific_text(self.atol_edit, "atol"),
            analytic_jacobian_policy=bool(
                self.analytic_jacobian_check.isChecked()
            ),
            wegscheider_policy=bool(self.wegscheider_check.isChecked()),
        )
        validate_tab_scientific_settings(candidate)
        return candidate

    @staticmethod
    def _positive_scientific_text(
        edit: QtWidgets.QLineEdit,
        field: str,
    ) -> float:
        try:
            value = float(edit.text())
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{field} must be a number.") from exc
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{field} must be finite and positive.")
        return value

    @QtCore.Slot()
    def _accept_candidate(self) -> None:
        try:
            candidate = self.scientific_settings()
        except (TypeError, ValueError) as exc:
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid Scientific Settings",
                str(exc),
            )
            return
        self.settingsAccepted.emit(candidate)
        self.accept()
