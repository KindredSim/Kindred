"""
Keyboard Shortcuts Dialog

Allows users to view and customize keyboard shortcuts.

This dialog intentionally avoids per-row cell-widget tables (e.g., `setCellWidget`) to prevent
Qt widget leaks when rows are rebuilt or cleared.
"""

import logging

from PySide6 import QtWidgets, QtCore, QtGui
from PySide6.QtCore import Qt

logger = logging.getLogger(__name__)


class _KeySequenceEditDelegate(QtWidgets.QStyledItemDelegate):
    def createEditor(self, parent: QtWidgets.QWidget, option, index):  # type: ignore[override]
        editor = QtWidgets.QKeySequenceEdit(parent)
        try:
            editor.setClearButtonEnabled(True)
        except (AttributeError, RuntimeError) as exc:
            logger.debug("Failed to enable clear button on QKeySequenceEdit: %s", exc, exc_info=True)

        def _commit() -> None:
            try:
                self.commitData.emit(editor)
                self.closeEditor.emit(editor)
            except RuntimeError as exc:
                logger.debug("Failed to commit/close key sequence editor: %s", exc, exc_info=True)

        editor.editingFinished.connect(_commit)
        return editor

    def setEditorData(self, editor: QtWidgets.QWidget, index: QtCore.QModelIndex) -> None:  # type: ignore[override]
        if not isinstance(editor, QtWidgets.QKeySequenceEdit):
            return
        value = index.data(QtCore.Qt.ItemDataRole.EditRole)
        if value is None:
            value = index.data(QtCore.Qt.ItemDataRole.DisplayRole)
        text = str(value or "").strip()
        if text.lower() == "none":
            text = ""
        editor.setKeySequence(QtGui.QKeySequence(text))

    def setModelData(self, editor: QtWidgets.QWidget, model: QtCore.QAbstractItemModel, index: QtCore.QModelIndex) -> None:  # type: ignore[override]
        if not isinstance(editor, QtWidgets.QKeySequenceEdit):
            return
        seq = editor.keySequence()
        text = seq.toString().strip()
        model.setData(index, text if text else "None", QtCore.Qt.ItemDataRole.EditRole)


class ShortcutsDialog(QtWidgets.QDialog):
    """
    Dialog for viewing and customizing keyboard shortcuts.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Keyboard Shortcuts")
        self.resize(700, 500)

        # Store shortcuts: {action_name: (description, shortcut_string, QAction)}
        self._shortcuts = {}
        self._modified = False

        self._init_ui()

    def _init_ui(self):
        """Initialize UI components."""
        layout = QtWidgets.QVBoxLayout(self)

        # Info label
        info_label = QtWidgets.QLabel(
            "Click on a shortcut to change it. Press Esc while capturing to cancel."
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # Shortcuts table
        self._table = QtWidgets.QTableWidget()
        self._table.setColumnCount(3)
        self._table.setHorizontalHeaderLabels(["Action", "Description", "Shortcut"])
        self._table.horizontalHeader().setStretchLastSection(False)
        self._table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.Fixed)
        self._table.setColumnWidth(2, 150)
        self._table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditKeyPressed
            | QtWidgets.QAbstractItemView.DoubleClicked
            | QtWidgets.QAbstractItemView.SelectedClicked
        )
        self._table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        layout.addWidget(self._table)

        self._shortcut_delegate = _KeySequenceEditDelegate(self._table)
        self._table.setItemDelegateForColumn(2, self._shortcut_delegate)
        self._table.itemChanged.connect(self._on_shortcut_item_changed)

        # Buttons
        button_layout = QtWidgets.QHBoxLayout()

        self._reset_btn = QtWidgets.QPushButton("Reset to Defaults")
        self._reset_btn.clicked.connect(self._reset_to_defaults)
        button_layout.addWidget(self._reset_btn)

        button_layout.addStretch()

        self._cancel_btn = QtWidgets.QPushButton("Cancel")
        self._cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self._cancel_btn)

        self._apply_btn = QtWidgets.QPushButton("Apply")
        self._apply_btn.clicked.connect(self._apply_changes)
        button_layout.addWidget(self._apply_btn)

        self._ok_btn = QtWidgets.QPushButton("OK")
        self._ok_btn.setDefault(True)
        self._ok_btn.clicked.connect(self.accept)
        button_layout.addWidget(self._ok_btn)

        layout.addLayout(button_layout)

    def add_shortcut(self, action_name: str, description: str, action: QtGui.QAction, default_shortcut: str = ""):
        """
        Add a shortcut to the dialog.

        Args:
            action_name: Internal name for the action
            description: User-friendly description
            action: QAction object to update
            default_shortcut: Default shortcut string
        """
        current_shortcut = action.shortcut().toString() if action.shortcut() else default_shortcut
        self._shortcuts[action_name] = {
            "description": description,
            "shortcut": current_shortcut,
            "default": default_shortcut,
            "action": action
        }

    def populate_table(self):
        """Populate the table with shortcuts."""
        self._row_for_action: dict[str, int] = {}
        with QtCore.QSignalBlocker(self._table):
            self._table.setRowCount(0)
            self._table.setRowCount(len(self._shortcuts))

            row = 0
            for action_name, data in sorted(self._shortcuts.items()):
                self._row_for_action[str(action_name)] = int(row)

                # Action name (read-only)
                name_item = QtWidgets.QTableWidgetItem(str(action_name))
                name_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                self._table.setItem(row, 0, name_item)

                # Description (read-only)
                desc_item = QtWidgets.QTableWidgetItem(str(data["description"]))
                desc_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                self._table.setItem(row, 1, desc_item)

                # Shortcut (editable via delegate)
                shortcut_str = str(data.get("shortcut") or "").strip()
                shortcut_item = QtWidgets.QTableWidgetItem(shortcut_str if shortcut_str else "None")
                shortcut_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsEditable)
                self._table.setItem(row, 2, shortcut_item)

                row += 1

    def _row_for(self, action_name: str) -> int | None:
        try:
            return int(getattr(self, "_row_for_action", {}).get(str(action_name)))  # type: ignore[union-attr]
        except (TypeError, ValueError, AttributeError):
            return None

    def _set_table_shortcut_text(self, action_name: str, value: str) -> None:
        row = self._row_for(action_name)
        if row is None:
            return
        item = self._table.item(row, 2)
        if item is None:
            item = QtWidgets.QTableWidgetItem()
            self._table.setItem(row, 2, item)
        text = str(value or "").strip()
        item.setText(text if text else "None")

    def _on_shortcut_changed(self, action_name: str, new_shortcut: str):
        """Handle shortcut change."""
        new_shortcut = str(new_shortcut or "").strip()
        if new_shortcut.lower() == "none":
            new_shortcut = ""

        # Check for conflicts
        conflicts = []
        for name, data in self._shortcuts.items():
            if name != action_name and data["shortcut"] == new_shortcut and new_shortcut:
                conflicts.append(name)

        if conflicts:
            reply = QtWidgets.QMessageBox.warning(
                self,
                "Shortcut Conflict",
                f"The shortcut '{new_shortcut}' is already used by:\n" +
                "\n".join(f"  • {c}" for c in conflicts) +
                "\n\nDo you want to use it anyway? (The other shortcut will be cleared)",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No
            )

            if reply == QtWidgets.QMessageBox.No:
                # Revert to old shortcut
                old_shortcut = self._shortcuts[action_name]["shortcut"]
                with QtCore.QSignalBlocker(self._table):
                    self._set_table_shortcut_text(action_name, old_shortcut)
                return
            else:
                # Clear conflicting shortcuts
                with QtCore.QSignalBlocker(self._table):
                    for conflict_name in conflicts:
                        self._shortcuts[conflict_name]["shortcut"] = ""
                        self._set_table_shortcut_text(conflict_name, "")

        # Update shortcut
        self._shortcuts[action_name]["shortcut"] = new_shortcut
        self._modified = True
        logger.debug("Shortcut changed: %s -> %s", action_name, new_shortcut)

    def _on_shortcut_item_changed(self, item: QtWidgets.QTableWidgetItem) -> None:
        if item is None:
            return
        if item.column() != 2:
            return
        name_item = self._table.item(item.row(), 0)
        if name_item is None:
            return
        action_name = str(name_item.text()).strip()
        if not action_name:
            return
        new_shortcut = str(item.text()).strip()
        old_shortcut = str(self._shortcuts.get(action_name, {}).get("shortcut") or "")
        try:
            self._on_shortcut_changed(action_name, new_shortcut)
        except Exception as exc:
            logger.debug(
                "Failed to handle shortcut edit for %s -> %s; reverting: %s",
                action_name,
                new_shortcut,
                exc,
                exc_info=True,
            )
            # If conflict handling failed, revert for safety.
            with QtCore.QSignalBlocker(self._table):
                self._set_table_shortcut_text(action_name, old_shortcut)

    def _apply_changes(self):
        """Apply shortcut changes to actions."""
        for action_name, data in self._shortcuts.items():
            action = data["action"]
            shortcut_str = data["shortcut"]

            if shortcut_str:
                action.setShortcut(QtGui.QKeySequence(shortcut_str))
            else:
                action.setShortcut(QtGui.QKeySequence())

        self._modified = False
        logger.info("Shortcuts applied")

        # Show confirmation
        QtWidgets.QMessageBox.information(
            self,
            "Shortcuts Applied",
            "Keyboard shortcuts have been updated."
        )

    def _reset_to_defaults(self):
        """Reset all shortcuts to default values."""
        reply = QtWidgets.QMessageBox.question(
            self,
            "Reset Shortcuts",
            "Reset all shortcuts to their default values?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No
        )

        if reply == QtWidgets.QMessageBox.Yes:
            with QtCore.QSignalBlocker(self._table):
                for action_name, data in self._shortcuts.items():
                    default_shortcut = data["default"]
                    self._shortcuts[action_name]["shortcut"] = default_shortcut
                    self._set_table_shortcut_text(action_name, default_shortcut)

            self._modified = True
            logger.info("Shortcuts reset to defaults")

    def accept(self):
        """Apply changes and close dialog."""
        if self._modified:
            self._apply_changes()
        super().accept()

    def reject(self):
        """Close dialog without applying changes."""
        if self._modified:
            reply = QtWidgets.QMessageBox.question(
                self,
                "Unsaved Changes",
                "You have unsaved changes. Discard them?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No
            )

            if reply == QtWidgets.QMessageBox.No:
                return

        super().reject()

    def get_shortcuts_dict(self) -> dict:
        """Get current shortcuts as a dictionary."""
        return {name: data["shortcut"] for name, data in self._shortcuts.items()}
