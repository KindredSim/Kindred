from __future__ import annotations

from typing import Mapping, Optional, Sequence

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

    def message_box_question(self, title: str, message: str, *, accept_label: str = "Apply") -> bool:
        dialog = QtWidgets.QMessageBox(self._parent)
        dialog.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        dialog.setWindowTitle(str(title))
        dialog.setText(str(message))
        apply_button = dialog.addButton(str(accept_label or "Apply"), QtWidgets.QMessageBox.ButtonRole.AcceptRole)
        dialog.addButton(QtWidgets.QMessageBox.StandardButton.Cancel)
        dialog.exec()
        return dialog.clickedButton() is apply_button

    def choose_wegscheider_resolution(
        self,
        title: str,
        message: str,
        choices: Mapping[str, Sequence[Mapping[str, str]]],
    ) -> dict[str, str] | None:
        dialog = QtWidgets.QDialog(self._parent)
        dialog.setWindowTitle(str(title))
        layout = QtWidgets.QVBoxLayout(dialog)
        label = QtWidgets.QLabel(str(message))
        label.setWordWrap(True)
        layout.addWidget(label)

        selectors: dict[str, QtWidgets.QComboBox] = {}
        for cycle_id, options in dict(choices or {}).items():
            cycle_key = str(cycle_id)
            group = QtWidgets.QGroupBox(cycle_key)
            group_layout = QtWidgets.QVBoxLayout(group)
            combo = QtWidgets.QComboBox(group)
            for option in list(options or []):
                parameter = str(dict(option).get("parameter_name") or "")
                line = str(dict(option).get("line") or parameter)
                if parameter:
                    combo.addItem(line, parameter)
            if combo.count() <= 0:
                continue
            selectors[cycle_key] = combo
            group_layout.addWidget(combo)
            layout.addWidget(group)

        if not selectors:
            return None

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        apply_button = buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
        if apply_button is not None:
            apply_button.setText("Apply")
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return None
        return {
            cycle_id: str(combo.currentData() or combo.currentText())
            for cycle_id, combo in selectors.items()
        }
