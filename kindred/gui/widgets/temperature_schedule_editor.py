"""
Temperature schedule editor dialog with visual preview.

Provides interactive GUI for creating and editing temperature schedules:
- Table-based interval editor
- Live preview plot
- Template presets (linear ramp, exponential, cyclic)
- Export to DSL syntax
"""

from __future__ import annotations

import logging
from typing import List, Tuple, Optional

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from kindred.core.temperature import TemperatureSchedule, TemperatureScheduleError

logger = logging.getLogger(__name__)

__all__ = ["TemperatureScheduleDialog"]


class TemperatureScheduleDialog(QtWidgets.QDialog):
    """
    Dialog for creating and editing temperature schedules.

    Features:
    - Table view for interval editing
    - Add/remove intervals
    - Template presets
    - Live preview plot
    - Export to DSL syntax

    Signals
    -------
    scheduleCreated : Signal(str)
        Emitted when user creates a schedule, contains DSL text
    """

    scheduleCreated = QtCore.Signal(str)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        """Initialize temperature schedule editor."""
        super().__init__(parent)

        self.setWindowTitle("Temperature Schedule Editor")
        self.setModal(True)
        self.resize(700, 600)

        layout = QtWidgets.QVBoxLayout(self)

        # Title and description
        title = QtWidgets.QLabel("Temperature Schedule Editor")
        title.setStyleSheet("font-size: 14pt; font-weight: bold;")
        layout.addWidget(title)

        self._desc_label = QtWidgets.QLabel()
        self._desc_label.setWordWrap(True)
        layout.addWidget(self._desc_label)

        # Interval table
        table_label = QtWidgets.QLabel("Temperature Intervals:")
        table_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        layout.addWidget(table_label)

        self._table = QtWidgets.QTableWidget()
        self._table.setColumnCount(3)
        self._table.setHorizontalHeaderLabels(["Start Time (s)", "End Time (s)", "Temperature (K)"])
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.setSelectionBehavior(QtWidgets.QTableWidget.SelectionBehavior.SelectRows)
        self._table.setMinimumHeight(200)

        # Set default row
        self._add_default_interval()

        layout.addWidget(self._table)

        # Table controls
        table_controls = QtWidgets.QHBoxLayout()

        self._add_btn = QtWidgets.QPushButton("+ Add Interval")
        self._add_btn.clicked.connect(self._add_interval)
        table_controls.addWidget(self._add_btn)

        self._remove_btn = QtWidgets.QPushButton("- Remove Selected")
        self._remove_btn.clicked.connect(self._remove_interval)
        table_controls.addWidget(self._remove_btn)

        table_controls.addWidget(QtWidgets.QLabel("Mode:"))
        self._mode_combo = QtWidgets.QComboBox()
        self._mode_combo.addItems(["Step Schedule", "First-Order Response"])
        self._mode_combo.currentTextChanged.connect(self._on_mode_changed)
        table_controls.addWidget(self._mode_combo)

        # Template dropdown
        table_controls.addWidget(QtWidgets.QLabel("Templates:"))
        self._template_combo = QtWidgets.QComboBox()
        self._template_combo.addItems([
            "Custom",
            "Linear Ramp (25°C → 100°C)",
            "Step Change (25°C → 50°C at t=50)",
            "Temperature Cycling (25°C ⇄ 50°C)",
            "Cooling Curve (100°C → 25°C)"
        ])
        self._template_combo.currentTextChanged.connect(self._apply_template)
        table_controls.addWidget(self._template_combo)

        self._tau_label = QtWidgets.QLabel("Tau (s):")
        table_controls.addWidget(self._tau_label)
        self._tau_spin = QtWidgets.QDoubleSpinBox()
        self._tau_spin.setRange(0.001, 1_000_000_000.0)
        self._tau_spin.setDecimals(3)
        self._tau_spin.setValue(10.0)
        self._tau_spin.setMaximumWidth(120)
        self._tau_spin.valueChanged.connect(lambda _value: self._refresh_outputs())
        table_controls.addWidget(self._tau_spin)

        table_controls.addStretch()
        layout.addLayout(table_controls)

        # Preview section
        preview_label = QtWidgets.QLabel("Preview:")
        preview_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        layout.addWidget(preview_label)

        # Use PyQtGraph for preview if available, otherwise text preview
        try:
            import pyqtgraph as pg

            self._plot_widget = pg.PlotWidget()
            self._plot_widget.setLabel('left', 'Temperature', units='K')
            self._plot_widget.setLabel('bottom', 'Time', units='s')
            self._plot_widget.setMinimumHeight(200)
            self._plot_widget.showGrid(x=True, y=True, alpha=0.3)
            layout.addWidget(self._plot_widget)
            self._has_plot = True

        except ImportError:
            # Fallback to text preview
            self._preview_text = QtWidgets.QPlainTextEdit()
            self._preview_text.setReadOnly(True)
            self._preview_text.setMaximumHeight(100)
            self._preview_text.setPlaceholderText("PyQtGraph not available - install for visual preview")
            layout.addWidget(self._preview_text)
            self._has_plot = False

        # DSL output
        dsl_label = QtWidgets.QLabel("DSL Output:")
        dsl_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        layout.addWidget(dsl_label)

        self._dsl_output = QtWidgets.QPlainTextEdit()
        self._dsl_output.setReadOnly(True)
        self._dsl_output.setMaximumHeight(60)
        self._dsl_output.setFont(QtGui.QFont("Courier New", 9))
        layout.addWidget(self._dsl_output)

        self._table.cellChanged.connect(lambda _row, _col: self._refresh_outputs())

        # Dialog buttons
        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok |
            QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self._on_accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        # Initial preview and DSL
        self._update_mode_controls()
        self._refresh_outputs()

    def _add_default_interval(self):
        """Add default interval (0-100s at 298.15K)."""
        row = self._table.rowCount()
        self._table.insertRow(row)

        self._table.setItem(row, 0, QtWidgets.QTableWidgetItem("0"))
        self._table.setItem(row, 1, QtWidgets.QTableWidgetItem("100"))
        self._table.setItem(row, 2, QtWidgets.QTableWidgetItem("298.15"))

    def _is_response_mode(self) -> bool:
        return self._mode_combo.currentText() == "First-Order Response"

    def _on_mode_changed(self, _mode_name: str) -> None:
        self._update_mode_controls()
        self._refresh_outputs()

    def _update_mode_controls(self) -> None:
        response_mode = self._is_response_mode()
        self._tau_label.setVisible(response_mode)
        self._tau_spin.setVisible(response_mode)
        if response_mode:
            self._desc_label.setText(
                "Create a first-order response temperature schedule for time-varying kinetics.\n"
                "Intervals define the step setpoint schedule and tau controls the response speed."
            )
        else:
            self._desc_label.setText(
                "Create piecewise constant temperature schedules for time-varying kinetics.\n"
                "Each interval specifies a constant temperature over a time range."
            )

    def _add_interval(self):
        """Add new interval after last one."""
        last_row = self._table.rowCount() - 1

        if last_row >= 0:
            # Get end time of last interval
            try:
                last_end = float(self._table.item(last_row, 1).text())
                new_start = last_end
                new_end = last_end + 50  # Default 50s interval
            except (ValueError, AttributeError):
                new_start = 0
                new_end = 50
        else:
            new_start = 0
            new_end = 50

        row = self._table.rowCount()
        self._table.insertRow(row)

        self._table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(new_start)))
        self._table.setItem(row, 1, QtWidgets.QTableWidgetItem(str(new_end)))
        self._table.setItem(row, 2, QtWidgets.QTableWidgetItem("298.15"))

    def _remove_interval(self):
        """Remove selected interval."""
        selected = self._table.selectedItems()
        if not selected:
            QtWidgets.QMessageBox.warning(
                self,
                "No Selection",
                "Please select an interval to remove."
            )
            return

        rows = set(item.row() for item in selected)
        for row in sorted(rows, reverse=True):
            self._table.removeRow(row)

    def _apply_template(self, template_name: str):
        """Apply temperature template."""
        if template_name == "Custom":
            return

        if template_name == "Linear Ramp (25°C → 100°C)":
            # 5 intervals: 25, 43.75, 62.5, 81.25, 100°C
            intervals = [
                (0, 25, 298.15),
                (25, 50, 316.9),
                (50, 75, 335.65),
                (75, 100, 354.4),
                (100, 125, 373.15)
            ]

        elif template_name == "Step Change (25°C → 50°C at t=50)":
            intervals = [
                (0, 50, 298.15),
                (50, 100, 323.15)
            ]

        elif template_name == "Temperature Cycling (25°C ⇄ 50°C)":
            intervals = [
                (0, 25, 298.15),
                (25, 50, 323.15),
                (50, 75, 298.15),
                (75, 100, 323.15),
                (100, 125, 298.15)
            ]

        elif template_name == "Cooling Curve (100°C → 25°C)":
            # Exponential-like cooling
            intervals = [
                (0, 25, 373.15),
                (25, 50, 348.15),
                (50, 75, 323.15),
                (75, 100, 310.65),
                (100, 125, 298.15)
            ]

        else:
            return

        self._set_table_intervals(intervals)

    def _set_table_intervals(self, intervals: List[Tuple[float, float, float]]) -> None:
        self._table.blockSignals(True)
        self._table.setRowCount(0)
        for t_start, t_end, T in intervals:
            row = self._table.rowCount()
            self._table.insertRow(row)
            self._table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(t_start)))
            self._table.setItem(row, 1, QtWidgets.QTableWidgetItem(str(t_end)))
            self._table.setItem(row, 2, QtWidgets.QTableWidgetItem(str(T)))
        self._table.blockSignals(False)
        self._refresh_outputs()

    def _get_intervals(self) -> List[Tuple[float, float, float]]:
        """Get intervals from table."""
        intervals = []
        for row in range(self._table.rowCount()):
            try:
                t_start = float(self._table.item(row, 0).text())
                t_end = float(self._table.item(row, 1).text())
                T = float(self._table.item(row, 2).text())
                intervals.append((t_start, t_end, T))
            except (ValueError, AttributeError):
                continue

        return intervals

    def _build_time_points_and_temperatures(
        self, intervals: List[Tuple[float, float, float]]
    ) -> Tuple[List[float], List[float]]:
        sorted_intervals = sorted(intervals)
        time_points = [sorted_intervals[0][0]]
        for _, t_end, _ in sorted_intervals:
            time_points.append(t_end)
        temperatures = [T for _, _, T in sorted_intervals]
        return time_points, temperatures

    def _build_current_schedule(self) -> TemperatureSchedule:
        intervals = self._get_intervals()
        if not intervals:
            raise TemperatureScheduleError("No intervals defined")
        time_points, temperatures = self._build_time_points_and_temperatures(intervals)
        if self._is_response_mode():
            return TemperatureSchedule.response(
                time_points,
                temperatures,
                tau=float(self._tau_spin.value()),
            )
        return TemperatureSchedule.piecewise(sorted(intervals))

    def _build_preview_series(self) -> Tuple[List[float], List[float], List[float]]:
        schedule = self._build_current_schedule()
        intervals = schedule.get_intervals()
        if not intervals:
            raise TemperatureScheduleError("No intervals defined")

        if not self._is_response_mode():
            times = []
            temps = []
            for t_start, t_end, T in sorted(self._get_intervals()):
                times.extend([t_start, t_end])
                temps.extend([T, T])
            return times, temps, temps.copy()

        _, setpoints = self._build_time_points_and_temperatures(self._get_intervals())
        last_end = float(intervals[-1].t_end)
        sample_count = max(50, 10 * len(intervals))
        times_arr = np.linspace(0.0, last_end, num=sample_count, dtype=float)
        actual = [float(schedule(float(t))) for t in times_arr.tolist()]
        setpoint = []
        for t in times_arr.tolist():
            for i, interval in enumerate(intervals):
                if i == len(intervals) - 1:
                    in_interval = interval.contains_inclusive(float(t))
                else:
                    in_interval = interval.contains(float(t))
                if in_interval:
                    setpoint.append(float(interval.temperature))
                    break
            else:
                setpoint.append(float(setpoints[-1]))
        return times_arr.tolist(), actual, setpoint

    def _refresh_outputs(self) -> None:
        self._update_preview()
        self._update_dsl_output()

    def _update_preview(self):
        """Update preview plot."""
        try:
            times, actual, setpoint = self._build_preview_series()
            if self._has_plot:
                import pyqtgraph as pg

                self._plot_widget.clear()
                if self._is_response_mode():
                    self._plot_widget.plot(
                        times,
                        setpoint,
                        pen=pg.mkPen(color=(120, 120, 120), width=1, style=QtCore.Qt.DashLine),
                    )
                self._plot_widget.plot(times, actual, pen=pg.mkPen(color='r', width=2))
            else:
                preview_lines = [
                    f"t={times[idx]:.3g}s  T={actual[idx]:.3f}K"
                    for idx in range(min(len(times), 6))
                ]
                self._preview_text.setPlainText("\n".join(preview_lines))

        except Exception as e:
            logger.warning(f"Failed to update preview: {e}")
            if not self._has_plot:
                self._preview_text.setPlainText("")

    def _update_dsl_output(self):
        """Update DSL syntax output."""
        try:
            intervals = self._get_intervals()
        except Exception:
            intervals = []
        if not intervals:
            self._dsl_output.setPlainText("# No intervals defined")
            return

        time_points, temperatures = self._build_time_points_and_temperatures(intervals)

        # Format DSL
        times_str = ",".join(str(t) for t in time_points)
        temps_str = ",".join(str(T) for T in temperatures)
        if self._is_response_mode():
            dsl = f"temp_response: t=[{times_str}], T=[{temps_str}], tau={self._tau_spin.value()}"
        else:
            dsl = f"temp_step: t=[{times_str}], T=[{temps_str}]"
        self._dsl_output.setPlainText(dsl)

    def _validate_intervals(self) -> Tuple[bool, str]:
        """
        Validate intervals.

        Returns
        -------
        (valid, error_message)
        """
        intervals = self._get_intervals()

        if not intervals:
            return False, "No intervals defined"

        try:
            _ = self._build_current_schedule()
        except TemperatureScheduleError as exc:
            return False, str(exc)
        return True, ""

    def _on_accept(self):
        """Handle OK button - validate and emit DSL."""
        valid, error = self._validate_intervals()

        if not valid:
            QtWidgets.QMessageBox.critical(
                self,
                "Invalid Temperature Schedule",
                f"Cannot create temperature schedule:\n\n{error}\n\n"
                "Please fix the intervals and try again."
            )
            return

        # Get DSL text
        dsl = self._dsl_output.toPlainText()

        # Emit signal
        self.scheduleCreated.emit(dsl)

        # Accept dialog
        self.accept()

    def get_dsl(self) -> str:
        """Get DSL text for current schedule."""
        return self._dsl_output.toPlainText()
