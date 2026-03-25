import logging

import pytest
from PySide6 import QtWidgets


@pytest.mark.gui
def test_shortcut_registry_is_populated_during_menu_creation(main_window):
    from PySide6 import QtGui

    def _std_default(key: QtGui.QKeySequence.StandardKey) -> str:
        bindings = list(QtGui.QKeySequence.keyBindings(key))
        if not bindings:
            return ""
        return bindings[0].toString()

    expected = {
        "Load Project": {
            "object_name": "loadProjectAction",
            "default": _std_default(QtGui.QKeySequence.Open),
        },
        "Save Project": {
            "object_name": "saveProjectAction",
            "default": _std_default(QtGui.QKeySequence.Save),
        },
        "Load Data": {
            "object_name": "loadDataAction",
            "default": QtGui.QKeySequence("Ctrl+Shift+L").toString(),
        },
        "Export CSV": {
            "object_name": "exportCsvAction",
            "default": QtGui.QKeySequence("Ctrl+E").toString(),
        },
        "Exit": {
            "object_name": "exitAction",
            "default": _std_default(QtGui.QKeySequence.Quit),
        },
        "Undo": {
            "object_name": "undoAction",
            "default": _std_default(QtGui.QKeySequence.Undo),
        },
        "Redo": {
            "object_name": "redoAction",
            "default": _std_default(QtGui.QKeySequence.Redo),
        },
        "Preferences": {
            "object_name": "preferencesAction",
            "default": _std_default(QtGui.QKeySequence.Preferences),
        },
        "Customize Keyboard Shortcuts": {
            "object_name": "customizeShortcutsAction",
            "default": QtGui.QKeySequence("Ctrl+K").toString(),
        },
        "Reset Layout": {
            "object_name": "resetLayoutAction",
            "default": QtGui.QKeySequence("Ctrl+Shift+R").toString(),
        },
        "Dark Mode": {
            "object_name": "darkModeAction",
            "default": QtGui.QKeySequence("Ctrl+Shift+D").toString(),
        },
        "Template Manager": {
            "object_name": "templateManagerAction",
            "default": QtGui.QKeySequence("Ctrl+T").toString(),
        },
        "Run": {
            "object_name": "runSimulationAction",
            "default": QtGui.QKeySequence("Ctrl+R").toString(),
        },
        "Stop": {
            "object_name": "stopSimulationAction",
            "default": QtGui.QKeySequence("Esc").toString(),
        },
        "Global Fit": {
            "object_name": "globalFitAction",
            "default": QtGui.QKeySequence("Ctrl+Shift+F").toString(),
        },
        "Documentation": {
            "object_name": "documentationAction",
            "default": _std_default(QtGui.QKeySequence.HelpContents),
        },
        "Keyboard Shortcuts": {
            "object_name": "keyboardShortcutsAction",
            "default": QtGui.QKeySequence("Ctrl+?").toString(),
        },
    }

    registry = getattr(main_window, "_shortcut_actions", None)
    assert isinstance(registry, dict)

    for action_name, meta in expected.items():
        action = main_window.findChild(QtGui.QAction, meta["object_name"])
        assert action is not None
        assert action.objectName() == meta["object_name"]
        assert action.shortcut().toString() == meta["default"]

        if meta["default"]:
            entry = registry.get(action_name)
            assert entry is not None
            assert entry["action"] is action
            assert entry["default"] == meta["default"]
        else:
            assert action_name not in registry

    # Multi-shortcut action: preserve the legacy "Ctrl+R or F5" behavior.
    run_entry = registry["Run"]
    run_action = run_entry["action"]
    shortcuts = {seq.toString() for seq in run_action.shortcuts()}
    assert QtGui.QKeySequence("Ctrl+R").toString() in shortcuts
    assert QtGui.QKeySequence("F5").toString() in shortcuts

    # Checkable action remains checkable.
    dark_entry = registry["Dark Mode"]
    assert bool(dark_entry["action"].isCheckable()) is True


@pytest.mark.gui
def test_collect_shortcut_actions_scan_is_removed():
    from kindred.gui.main_window import MainWindow

    assert not hasattr(MainWindow, "_collect_shortcut_actions")


@pytest.mark.gui
def test_redo_noop_does_not_report_insert_reset(main_window, caplog, monkeypatch):
    caplog.set_level(logging.DEBUG, logger="kindred.gui.main_window")
    monkeypatch.setattr(main_window._undo_stack, "canRedo", lambda: False)

    button = main_window.findChild(QtWidgets.QPushButton, "addBatchSetButton")
    assert button is not None
    button.setFocus()
    QtWidgets.QApplication.processEvents()

    main_window._redo()

    assert main_window._status_label.text() == "Nothing to redo"
    assert "Insert behavior preference reset" not in caplog.text
    assert "Insert preference reset" not in caplog.text


@pytest.mark.gui
def test_redo_text_edit_preserves_redo_feedback_without_insert_reset(main_window, caplog, monkeypatch):
    caplog.set_level(logging.DEBUG, logger="kindred.gui.main_window")
    monkeypatch.setattr(main_window._undo_stack, "canRedo", lambda: False)

    editor = QtWidgets.QPlainTextEdit()
    redo_calls: list[str] = []

    class _RedoDoc:
        @staticmethod
        def isRedoAvailable() -> bool:
            return True

    monkeypatch.setattr(QtWidgets.QApplication, "focusWidget", lambda: editor)
    monkeypatch.setattr(editor, "document", lambda: _RedoDoc())
    monkeypatch.setattr(editor, "redo", lambda: redo_calls.append("redo"))

    main_window._redo()

    assert redo_calls == ["redo"]
    assert main_window._status_label.text() == "Redo (text edit)"
    assert "Redo action performed (text editor)" in caplog.text
    assert "Insert behavior preference reset" not in caplog.text
    assert "Insert preference reset" not in caplog.text
