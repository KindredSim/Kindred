# kindred/gui/widgets/parameter_statistics_table.py
"""Solver parameter summary table widget."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

from PySide6 import QtCore, QtWidgets

__all__ = ["ParameterStatisticsTable"]


class ParameterStatisticsTable(QtWidgets.QTableWidget):
    """
    Bottom panel showing current solver parameter values and units.

    Columns:
    - Name
    - Value
    - Unit
    """

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setColumnCount(3)
        self.setHorizontalHeaderLabels(["Name", "Value", "Unit"])
        header = self.horizontalHeader()
        header.setStretchLastSection(False)
        for col in range(self.columnCount()):
            header.setSectionResizeMode(col, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.Interactive)  # Unit
        self.setColumnWidth(2, 140)
        self.setAlternatingRowColors(True)
        self.setMaximumHeight(130)

    def update_parameters(self, parameters: Dict[str, Tuple[float, str]]) -> None:
        """
        Update table with parameter values.

        Parameters
        ----------
        parameters : dict
            Mapping name -> (value, unit)
        """
        self.setRowCount(0)
        for row, name in enumerate(sorted(parameters.keys(), key=_param_sort_key)):
            value, unit = parameters[name]
            self.insertRow(row)
            _name_item = QtWidgets.QTableWidgetItem(str(name))
            _name_item.setTextAlignment(QtCore.Qt.AlignCenter)
            self.setItem(row, 0, _name_item)
            _val_item = QtWidgets.QTableWidgetItem(f"{float(value):.6g}")
            _val_item.setTextAlignment(QtCore.Qt.AlignCenter)
            self.setItem(row, 1, _val_item)
            _unit_item = QtWidgets.QTableWidgetItem(str(unit))
            _unit_item.setTextAlignment(QtCore.Qt.AlignCenter)
            self.setItem(row, 2, _unit_item)
        self.viewport().repaint()


def _param_sort_key(name: str):
    # Natural-ish ordering for k/kf/kr/Keq with numeric suffix; otherwise alpha.
    from kindred.core.simulator.parameter_namespace import protected_indexed_identifier_step_index

    idx = protected_indexed_identifier_step_index(str(name))
    if idx is not None:
        name_s = str(name)
        family = "Keq" if name_s.lower().startswith("keq") else name_s.rstrip("0123456789")
        fam_order = {"k": 0, "kf": 1, "kr": 2, "Keq": 3}.get(family, 9)
        return (0, fam_order, idx)
    return (1, str(name))
