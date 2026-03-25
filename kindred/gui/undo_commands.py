"""
Undo/Redo commands for Kindred GUI.

Provides QUndoCommand subclasses for tracking mechanism editing operations.
Each command encapsulates a reversible operation on the mechanism editor.
"""

from __future__ import annotations

from PySide6 import QtWidgets, QtGui

__all__ = [
    "SetMechanismTextCommand",
    "SetMechanismEditorTextsCommand",
    "InsertTextCommand",
    "DeleteTextCommand",
    "ReplaceTextCommand",
]


class SetMechanismTextCommand(QtGui.QUndoCommand):
    """
    Command for setting the entire mechanism text.

    Used for operations like loading a preset, importing a file, or
    clearing the mechanism.
    """

    def __init__(
        self,
        text_widget: QtWidgets.QPlainTextEdit,
        new_text: str,
        old_text: str,
        description: str = "Set mechanism text",
    ):
        """
        Initialize command.

        Parameters
        ----------
        text_widget : QPlainTextEdit
            The text editor widget
        new_text : str
            New text to set
        old_text : str
            Previous text (for undo)
        description : str
            Human-readable description for the undo stack
        """
        super().__init__(description)
        self._text_widget = text_widget
        self._new_text = new_text
        self._old_text = old_text

    def redo(self) -> None:
        """Apply the command (set new text)."""
        self._text_widget.blockSignals(True)
        try:
            self._text_widget.setPlainText(self._new_text)
        finally:
            self._text_widget.blockSignals(False)
        # Manually trigger text changed after setting
        self._text_widget.document().contentsChanged.emit()

    def undo(self) -> None:
        """Revert the command (restore old text)."""
        self._text_widget.blockSignals(True)
        try:
            self._text_widget.setPlainText(self._old_text)
        finally:
            self._text_widget.blockSignals(False)
        # Manually trigger text changed after setting
        self._text_widget.document().contentsChanged.emit()


class SetMechanismEditorTextsCommand(QtGui.QUndoCommand):
    """Command for updating authoritative reactions and state-network editors together."""

    def __init__(
        self,
        reactions_widget: QtWidgets.QPlainTextEdit,
        state_network_editor,
        *,
        new_reactions_text: str,
        old_reactions_text: str,
        new_state_network_dsl: str,
        old_state_network_dsl: str,
        description: str = "Set mechanism editor texts",
    ):
        super().__init__(description)
        self._reactions_widget = reactions_widget
        self._state_network_editor = state_network_editor
        self._new_reactions_text = str(new_reactions_text)
        self._old_reactions_text = str(old_reactions_text)
        self._new_state_network_dsl = str(new_state_network_dsl)
        self._old_state_network_dsl = str(old_state_network_dsl)

    def _set_reactions_text(self, text: str) -> None:
        widget = self._reactions_widget
        if widget is None:
            return
        current_text = widget.toPlainText()
        if current_text == text:
            return
        widget.blockSignals(True)
        try:
            widget.setPlainText(text)
        finally:
            widget.blockSignals(False)
        widget.document().contentsChanged.emit()

    def _set_state_network_dsl(self, dsl_text: str) -> None:
        editor = self._state_network_editor
        if (
            editor is None
            or not hasattr(editor, "get_state_network_dsl")
            or not hasattr(editor, "set_state_network_dsl")
        ):
            return
        current_text = str(editor.get_state_network_dsl() or "")
        if current_text == dsl_text:
            return
        editor.set_state_network_dsl(dsl_text)

    def _apply(self, reactions_text: str, state_network_dsl: str) -> None:
        self._set_reactions_text(str(reactions_text))
        self._set_state_network_dsl(str(state_network_dsl))

    def redo(self) -> None:
        self._apply(self._new_reactions_text, self._new_state_network_dsl)

    def undo(self) -> None:
        self._apply(self._old_reactions_text, self._old_state_network_dsl)


class InsertTextCommand(QtGui.QUndoCommand):
    """
    Command for inserting text at a specific position.

    Used for operations like appending reactions or inserting templates.
    """

    def __init__(
        self,
        text_widget: QtWidgets.QPlainTextEdit,
        position: int,
        text: str,
        description: str = "Insert text",
    ):
        """
        Initialize command.

        Parameters
        ----------
        text_widget : QPlainTextEdit
            The text editor widget
        position : int
            Character position for insertion
        text : str
            Text to insert
        description : str
            Human-readable description
        """
        super().__init__(description)
        self._text_widget = text_widget
        self._position = position
        self._text = text

    def redo(self) -> None:
        """Apply the command (insert text)."""
        cursor = self._text_widget.textCursor()
        cursor.setPosition(self._position)
        self._text_widget.setTextCursor(cursor)
        cursor.insertText(self._text)

    def undo(self) -> None:
        """Revert the command (delete inserted text)."""
        cursor = self._text_widget.textCursor()
        cursor.setPosition(self._position)
        cursor.setPosition(self._position + len(self._text),
                          QtGui.QTextCursor.MoveMode.KeepAnchor)
        cursor.removeSelectedText()


class DeleteTextCommand(QtGui.QUndoCommand):
    """
    Command for deleting text in a range.

    Used for operations like removing reactions or clearing sections.
    """

    def __init__(
        self,
        text_widget: QtWidgets.QPlainTextEdit,
        start: int,
        end: int,
        deleted_text: str,
        description: str = "Delete text",
    ):
        """
        Initialize command.

        Parameters
        ----------
        text_widget : QPlainTextEdit
            The text editor widget
        start : int
            Start position of deletion
        end : int
            End position of deletion
        deleted_text : str
            Text that was deleted (for undo)
        description : str
            Human-readable description
        """
        super().__init__(description)
        self._text_widget = text_widget
        self._start = start
        self._end = end
        self._deleted_text = deleted_text

    def redo(self) -> None:
        """Apply the command (delete text)."""
        cursor = self._text_widget.textCursor()
        cursor.setPosition(self._start)
        cursor.setPosition(self._end, QtGui.QTextCursor.MoveMode.KeepAnchor)
        cursor.removeSelectedText()

    def undo(self) -> None:
        """Revert the command (restore deleted text)."""
        cursor = self._text_widget.textCursor()
        cursor.setPosition(self._start)
        cursor.insertText(self._deleted_text)


class ReplaceTextCommand(QtGui.QUndoCommand):
    """
    Command for replacing text in a range.

    Used for operations like find-and-replace or updating parameter values.
    """

    def __init__(
        self,
        text_widget: QtWidgets.QPlainTextEdit,
        start: int,
        end: int,
        old_text: str,
        new_text: str,
        description: str = "Replace text",
    ):
        """
        Initialize command.

        Parameters
        ----------
        text_widget : QPlainTextEdit
            The text editor widget
        start : int
            Start position of replacement
        end : int
            End position of replacement
        old_text : str
            Text to be replaced (for undo)
        new_text : str
            New text to insert
        description : str
            Human-readable description
        """
        super().__init__(description)
        self._text_widget = text_widget
        self._start = start
        self._end = end
        self._old_text = old_text
        self._new_text = new_text

    def redo(self) -> None:
        """Apply the command (replace text)."""
        cursor = self._text_widget.textCursor()
        cursor.setPosition(self._start)
        cursor.setPosition(self._end, QtGui.QTextCursor.MoveMode.KeepAnchor)
        cursor.removeSelectedText()
        cursor.insertText(self._new_text)

    def undo(self) -> None:
        """Revert the command (restore old text)."""
        cursor = self._text_widget.textCursor()
        cursor.setPosition(self._start)
        cursor.setPosition(
            self._start + len(self._new_text),
            QtGui.QTextCursor.MoveMode.KeepAnchor,
        )
        cursor.removeSelectedText()
        cursor.insertText(self._old_text)


class MacroCommand(QtGui.QUndoCommand):
    """
    Command that groups multiple commands together.

    Used for complex operations that consist of multiple atomic changes.
    """

    def __init__(
        self,
        commands: list[QtGui.QUndoCommand],
        description: str = "Multiple changes",
    ):
        """
        Initialize macro command.

        Parameters
        ----------
        commands : list of QUndoCommand
            Child commands to execute
        description : str
            Human-readable description
        """
        super().__init__(description)
        for cmd in commands:
            cmd.setParent(self)
