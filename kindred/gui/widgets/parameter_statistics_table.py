# kindred/gui/widgets/parameter_statistics_table.py
"""Solver parameter summary table widget."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

from PySide6 import QtWidgets

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
        header.setStretchLastSection(True)
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
            self.setItem(row, 0, QtWidgets.QTableWidgetItem(str(name)))
            self.setItem(row, 1, QtWidgets.QTableWidgetItem(f"{float(value):.6g}"))
            self.setItem(row, 2, QtWidgets.QTableWidgetItem(str(unit)))

        self.resizeColumnsToContents()
        self.viewport().repaint()


def _param_sort_key(name: str):
    # Natural-ish ordering for k/kf/kr/K with numeric suffix; otherwise alpha.
    import re

    m = re.match(r"^(kf|kr|k|K)(\d+)$", str(name))
    if m:
        family = m.group(1)
        idx = int(m.group(2))
        fam_order = {"k": 0, "kf": 1, "kr": 2, "K": 3}.get(family, 9)
        return (0, fam_order, idx)
    return (1, str(name))
