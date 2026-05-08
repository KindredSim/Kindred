from __future__ import annotations

from typing import Optional

from PySide6 import QtWidgets


class SimulationDialogs:
    """Thin Qt dialog boundary for simulation-controller user messages."""

    def __init__(self, parent: QtWidgets.QWidget) -> None:
        self._parent = parent

    def message_box_warning(self, title: str, message: str) -> None:
        QtWidgets.QMessageBox.warning(self._parent, str(title), str(message))

    def message_box_critical(self, title: str, message: str, *, details: Optional[str] = None) -> None:
        if not details:
            QtWidgets.QMessageBox.critical(self._parent, str(title), str(message))
            return
        dialog = QtWidgets.QMessageBox(self._parent)
        dialog.setIcon(QtWidgets.QMessageBox.Icon.Critical)
        dialog.setWindowTitle(str(title))
        dialog.setText(str(message))
        dialog.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Ok)
        dialog.setDetailedText(str(details))
        dialog.exec()
