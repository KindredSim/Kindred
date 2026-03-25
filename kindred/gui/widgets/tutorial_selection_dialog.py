"""
Tutorial selection dialog.

Allows users to browse and launch interactive tutorials.
"""

from __future__ import annotations

import logging
from typing import Optional

from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import Qt

logger = logging.getLogger(__name__)

__all__ = ["TutorialSelectionDialog"]


class TutorialSelectionDialog(QtWidgets.QDialog):
    """Dialog for selecting and launching tutorials."""

    tutorialSelected = QtCore.Signal(str)  # Emits tutorial ID

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        """Initialize tutorial selection dialog."""
        super().__init__(parent)

        self.setWindowTitle("Interactive Tutorials")
        self.setModal(True)
        self.resize(600, 450)

        layout = QtWidgets.QVBoxLayout(self)

        # Title
        title = QtWidgets.QLabel("Interactive Tutorials")
        title.setStyleSheet("font-size: 14pt; font-weight: bold;")
        layout.addWidget(title)

        desc = QtWidgets.QLabel(
            "Learn Kindred through step-by-step interactive tutorials. "
            "Each tutorial highlights relevant UI elements and guides you through a workflow."
        )
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # Tutorial list
        self._list = QtWidgets.QListWidget()
        self._list.setAlternatingRowColors(True)
        self._list.itemDoubleClicked.connect(self._on_launch)

        # Populate tutorials
        from kindred.gui.tutorial_manager import TutorialManager

        tutorials = TutorialManager.get_tutorial_list()
        for tutorial in tutorials:
            item = QtWidgets.QListWidgetItem()
            item.setText(f"{tutorial['title']}")
            item.setData(Qt.UserRole, tutorial['id'])  # Store ID
            item.setToolTip(f"{tutorial['description']}\n\nDuration: ~{tutorial['duration']}")
            self._list.addItem(item)

        if self._list.count() > 0:
            self._list.setCurrentRow(0)

        layout.addWidget(self._list)

        # Description panel
        desc_group = QtWidgets.QGroupBox("Tutorial Description")
        desc_layout = QtWidgets.QVBoxLayout(desc_group)

        self._desc_label = QtWidgets.QLabel()
        self._desc_label.setWordWrap(True)
        self._desc_label.setTextFormat(Qt.TextFormat.RichText)
        desc_layout.addWidget(self._desc_label)

        self._duration_label = QtWidgets.QLabel()
        self._duration_label.setStyleSheet("font-style: italic;")
        desc_layout.addWidget(self._duration_label)

        layout.addWidget(desc_group)

        # Update description when selection changes
        self._list.currentItemChanged.connect(self._on_selection_changed)
        if self._list.count() > 0:
            self._on_selection_changed(self._list.item(0), None)

        # Buttons
        button_box = QtWidgets.QDialogButtonBox()

        launch_btn = button_box.addButton("Start Tutorial", QtWidgets.QDialogButtonBox.AcceptRole)
        launch_btn.clicked.connect(self._on_launch)

        close_btn = button_box.addButton(QtWidgets.QDialogButtonBox.Close)
        close_btn.clicked.connect(self.reject)

        layout.addWidget(button_box)

    def _on_selection_changed(self, current: QtWidgets.QListWidgetItem, previous: QtWidgets.QListWidgetItem):
        """Update description when selection changes."""
        if not current:
            return

        tutorial_id = current.data(Qt.UserRole)

        from kindred.gui.tutorial_manager import TutorialManager
        tutorials = TutorialManager.get_tutorial_list()

        for tutorial in tutorials:
            if tutorial['id'] == tutorial_id:
                self._desc_label.setText(tutorial['description'])
                self._duration_label.setText(f"Estimated duration: {tutorial['duration']}")
                break

    def _on_launch(self):
        """Launch selected tutorial."""
        current = self._list.currentItem()
        if not current:
            return

        tutorial_id = current.data(Qt.UserRole)
        self.tutorialSelected.emit(tutorial_id)
        self.accept()
