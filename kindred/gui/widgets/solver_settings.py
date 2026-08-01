"""Complete four-field execution-settings editor for one exact IC set."""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from kindred.core.project_document import (
    SimulationExecutionSettings,
    TabScientificSettings,
    validate_simulation_execution_settings,
    validate_tab_scientific_settings,
)

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
        self.solver_combo.addItems(["Radau", "BDF"])
        self.t_start_spin = self._time_spin(0.0)
        self.t_end_spin = self._time_spin(10.0)
        self.integration_points_spin = QtWidgets.QSpinBox(self)
        self.integration_points_spin.setRange(10, 100_000)
        self.integration_points_spin.setSingleStep(10)
        for label, widget in (
            ("Solver", self.solver_combo),
            ("t_start", self.t_start_spin),
            ("t_end", self.t_end_spin),
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

    def _time_spin(self, value: float) -> QtWidgets.QDoubleSpinBox:
        spin = QtWidgets.QDoubleSpinBox(self)
        spin.setDecimals(12)
        spin.setRange(0.0, 1e300)
        spin.setValue(float(value))
        spin.setMaximumWidth(190)
        return spin

    def set_execution_settings(self, settings: SimulationExecutionSettings) -> None:
        validate_simulation_execution_settings(settings)
        self.solver_combo.setCurrentText(settings.solver)
        self.t_start_spin.setValue(settings.t_start)
        self.t_end_spin.setValue(settings.t_end)
        self.integration_points_spin.setValue(settings.integration_points)

    def execution_settings(self) -> SimulationExecutionSettings:
        candidate = SimulationExecutionSettings(
            solver=str(self.solver_combo.currentText()),
            t_start=float(self.t_start_spin.value()),
            t_end=float(self.t_end_spin.value()),
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
        self.rtol_spin = self._positive_spin(
            minimum=1e-300,
            maximum=1e300,
            value=1e-6,
            decimals=15,
        )
        self.atol_spin = self._positive_spin(
            minimum=1e-300,
            maximum=1e300,
            value=1e-12,
            decimals=18,
        )
        self.sparse_jacobian_check = QtWidgets.QCheckBox(
            "Use sparse Jacobian",
            self,
        )
        self.wegscheider_check = QtWidgets.QCheckBox(
            "Enforce Wegscheider cyclicity",
            self,
        )
        form.addRow("Temperature (K)", self.temperature_spin)
        form.addRow("Relative tolerance", self.rtol_spin)
        form.addRow("Absolute tolerance", self.atol_spin)
        form.addRow("", self.sparse_jacobian_check)
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
        spin.setDecimals(int(decimals))
        spin.setRange(float(minimum), float(maximum))
        spin.setValue(float(value))
        spin.setMaximumWidth(190)
        return spin

    def set_scientific_settings(self, settings: TabScientificSettings) -> None:
        validate_tab_scientific_settings(settings)
        self.temperature_spin.setValue(settings.temperature)
        self.rtol_spin.setValue(settings.rtol)
        self.atol_spin.setValue(settings.atol)
        self.sparse_jacobian_check.setChecked(settings.sparse_jacobian_policy)
        self.wegscheider_check.setChecked(settings.wegscheider_policy)

    def scientific_settings(self) -> TabScientificSettings:
        candidate = TabScientificSettings(
            temperature=float(self.temperature_spin.value()),
            rtol=float(self.rtol_spin.value()),
            atol=float(self.atol_spin.value()),
            sparse_jacobian_policy=bool(
                self.sparse_jacobian_check.isChecked()
            ),
            wegscheider_policy=bool(self.wegscheider_check.isChecked()),
        )
        validate_tab_scientific_settings(candidate)
        return candidate

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
