# kindred/gui/widgets/editor_panel.py
"""Mechanism editor container widget."""

from __future__ import annotations

from typing import Optional

from PySide6 import QtWidgets

# Direct imports required to avoid circular dependency with widgets/__init__.py
from kindred.gui.widgets.mechanism_editor import MechanismEditorTabbed

__all__ = ["EditorPanel"]


class EditorPanel(QtWidgets.QWidget):
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.editor = MechanismEditorTabbed()
        self.editor.setObjectName("mechanismEditor")

        layout.addWidget(self.editor)

