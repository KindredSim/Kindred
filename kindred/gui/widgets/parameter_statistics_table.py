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
        self.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setColumnCount(3)
        self.setHorizontalHeaderLabels(["Name", "Value", "Unit"])
        header = self.horizontalHeader()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.Stretch)
        self.setAlternatingRowColors(True)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )

    def update_parameters(self, parameters: Dict[str, Tuple[float, str]]) -> None:
        """
        Update table with parameter values.

        Parameters
        ----------
        parameters : dict
            Mapping name -> (value, unit)
        """
        names = tuple(sorted(parameters.keys(), key=_param_sort_key))
        current_names = tuple(
            str(self.item(row, 0).data(QtCore.Qt.ItemDataRole.UserRole) or "")
            if self.item(row, 0) is not None
            else ""
            for row in range(self.rowCount())
        )
        if current_names != names:
            self.setRowCount(len(names))
            for row, name in enumerate(names):
                name_item = QtWidgets.QTableWidgetItem()
                name_item.setData(QtCore.Qt.ItemDataRole.UserRole, str(name))
                name_item.setTextAlignment(QtCore.Qt.AlignCenter)
                self.setItem(row, 0, name_item)
                for column in (1, 2):
                    item = QtWidgets.QTableWidgetItem()
                    item.setTextAlignment(QtCore.Qt.AlignCenter)
                    self.setItem(row, column, item)
        for row, name in enumerate(names):
            value, unit = parameters[name]
            self.item(row, 0).setText(str(name))
            self.item(row, 1).setText(f"{float(value):.6g}")
            self.item(row, 2).setText(str(unit))


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
