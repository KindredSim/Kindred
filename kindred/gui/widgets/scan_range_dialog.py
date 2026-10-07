"""GUI interval selection and authored-quantity constraint controls."""
from __future__ import annotations

import html

from kindred.core.quantity_units import format_unit_for_display, presentation_unit
from kindred.core.validation import parse_finite_numeric_text


from PySide6 import QtCore, QtWidgets

from kindred.scan.config import InputRange, ObservableConstraint
from kindred.scan.intervals import Interval, normalize
from kindred.gui.ui_helpers import (
    SCIENTIFIC_NAME_HTML_ROLE,
    ScientificNameComboBox,
)
from kindred.gui.display_name_policy import scientific_symbol_html


def number(edit):
    return parse_finite_numeric_text(edit.text().strip())


def entry(value, parent=None):
    edit = QtWidgets.QLineEdit(str(value), parent)
    return edit


class ScanRangeDialog(QtWidgets.QDialog):
    def __init__(self, label, selection, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Allowed values — {label}")
        self.resize(720, 370)
        self.selection = selection
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel("Add allowed intervals. Endpoint coverage counts toward the simulation budget."))
        self.table = QtWidgets.QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["Lower", "Include", "Sample Exactly", "Upper", "Include", "Sample Exactly"])
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table)
        for interval in selection.intervals:
            self.add_interval(interval)
        row = QtWidgets.QHBoxLayout()
        add = QtWidgets.QPushButton("Add Interval")
        add.clicked.connect(lambda: self.add_interval(Interval(0, 1)))
        remove = QtWidgets.QPushButton("Remove Selected Interval")
        remove.clicked.connect(lambda: self.table.removeRow(self.table.currentRow()))
        row.addWidget(add)
        row.addWidget(remove)
        layout.addLayout(row)
        self.allocation = QtWidgets.QComboBox()
        self.allocation.addItem("Proportional to interval width", "width")
        self.allocation.addItem("Equal per interval", "equal")
        self.allocation.setCurrentIndex(self.allocation.findData(selection.allocation))
        layout.addWidget(self.allocation)
        self.error = QtWidgets.QLabel()
        self.error.setWordWrap(True)
        layout.addWidget(self.error)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def add_interval(self, interval):
        row = self.table.rowCount()
        self.table.insertRow(row)
        for offset, value, included in ((0, interval.lower, interval.include_lower), (3, interval.upper, interval.include_upper)):
            self.table.setCellWidget(row, offset, entry(value))
            include = QtWidgets.QCheckBox()
            include.setChecked(included)
            exact = QtWidgets.QCheckBox()
            exact.setChecked(included and value in self.selection.endpoints)
            exact.setEnabled(included)
            include.toggled.connect(exact.setEnabled)
            include.toggled.connect(lambda checked, box=exact: box.setChecked(box.isChecked() and checked))
            self.table.setCellWidget(row, offset + 1, include)
            self.table.setCellWidget(row, offset + 2, exact)

    def _accept(self):
        try:
            intervals, endpoints = [], []
            for row in range(self.table.rowCount()):
                low, high = number(self.table.cellWidget(row, 0)), number(self.table.cellWidget(row, 3))
                lc, uc = self.table.cellWidget(row, 1).isChecked(), self.table.cellWidget(row, 4).isChecked()
                intervals.append(Interval(low, high, lc, uc))
                if self.table.cellWidget(row, 2).isChecked():
                    endpoints.append(low)
                if self.table.cellWidget(row, 5).isChecked():
                    endpoints.append(high)
            self.selection = InputRange(normalize(tuple(intervals)), self.allocation.currentData(), tuple(dict.fromkeys(endpoints)))
        except ValueError as exc:
            self.error.setText(str(exc))
            return
        self.accept()


class ScanConstraintDialog(QtWidgets.QDialog):
    def __init__(self, statements, inputs, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Constrain an Authored Quantity")
        self.resize(580, 350)
        layout = QtWidgets.QFormLayout(self)
        self.quantity = ScientificNameComboBox()
        for statement in statements:
            unit = presentation_unit(statement.expr.quantity_unit, statement.declared_unit)
            label = statement.name + (f" [{unit.symbol}]" if unit is not None else " [units unspecified]")
            self.quantity.addItem(label, statement)
            if unit is not None:
                self.quantity.setItemData(
                    self.quantity.count() - 1,
                    f"{html.escape(statement.name)} [{format_unit_for_display(unit)}]",
                    SCIENTIFIC_NAME_HTML_ROLE,
                )
        self.definition = QtWidgets.QLabel()
        self.definition.setWordWrap(True)
        self.definition.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self.quantity.currentIndexChanged.connect(self._definition)
        self._definition()
        self.condition = QtWidgets.QComboBox()
        self.condition.addItem("Equals", "equals")
        self.condition.addItem("Within range", "range")
        self.lower, self.upper = entry(1), entry(1)
        self.include_lower, self.include_upper = QtWidgets.QCheckBox("Include Lower"), QtWidgets.QCheckBox("Include Upper")
        self.include_lower.setChecked(True)
        self.include_upper.setChecked(True)
        self.dependent = ScientificNameComboBox()
        for item in inputs:
            unit_html = format_unit_for_display(item.unit)
            label_html = scientific_symbol_html(item.label, family=item.family, role=item.role)
            self.dependent.addItem(f"{item.label} [{item.unit}]", item.key)
            self.dependent.setItemData(
                self.dependent.count() - 1,
                f"{label_html} [{unit_html}]",
                SCIENTIFIC_NAME_HTML_ROLE,
            )
        layout.addRow("Authored quantity", self.quantity)
        layout.addRow(self.definition)
        layout.addRow("Condition", self.condition)
        layout.addRow("Target / lower value", self.lower)
        layout.addRow(self.include_lower)
        layout.addRow("Upper value", self.upper)
        layout.addRow(self.include_upper)
        layout.addRow("Input to calculate / sample conditionally", self.dependent)
        self.condition.currentIndexChanged.connect(self._mode)
        self._mode()
        self.error = QtWidgets.QLabel()
        self.error.setWordWrap(True)
        layout.addRow(self.error)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def _definition(self):
        statement = self.quantity.currentData()
        self.definition.setText(statement.line_text if statement else "No authored quantities are available.")

    def _mode(self):
        interval = self.condition.currentData() == "range"
        for widget in (self.upper, self.include_lower, self.include_upper):
            widget.setEnabled(interval)

    def _accept(self):
        try:
            if self.quantity.currentData() is None or self.dependent.currentData() is None:
                raise ValueError("Select an authored quantity and a scanned independent input.")
            low = number(self.lower)
            equality = self.condition.currentData() == "equals"
            high = low if equality else number(self.upper)
            lc, uc = (True, True) if equality else (self.include_lower.isChecked(), self.include_upper.isChecked())
            normalize((Interval(low, high, lc, uc),))
            self.constraint = ObservableConstraint(self.quantity.currentData().name, self.condition.currentData(), low, high,
                                                  self.dependent.currentData(), lc, uc)
        except ValueError as exc:
            self.error.setText(str(exc))
            return
        self.accept()
