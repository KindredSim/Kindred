import json

import pytest
from PySide6 import QtCore, QtWidgets

from kindred.gui.main_window import MainWindow

pytestmark = [pytest.mark.gui]


def _process_events_bounded(iterations: int = 20) -> None:
    app = QtWidgets.QApplication.instance()
    assert app is not None
    for _ in range(int(iterations)):
        app.processEvents()


def _find_state_network_dialog() -> QtWidgets.QDialog:
    for widget in QtWidgets.QApplication.topLevelWidgets():
        if isinstance(widget, QtWidgets.QDialog) and widget.windowTitle() == "State Network Editor":
            return widget
    raise AssertionError("State Network Editor dialog not found")


def _unlock_reactions_editing(main_window: MainWindow, monkeypatch: pytest.MonkeyPatch) -> None:
    reactions_widget = main_window._mechanism_editor._reactions_text
    if not reactions_widget.toPlainText().strip():
        reactions_widget.setPlainText("reaction: A -> B; k=1.0")
    monkeypatch.setattr(main_window, "_prompt_mechanism_edit_unlock_warning", lambda: True)
    main_window._mechanism_edit_lock_action.trigger()
    assert main_window.mechanism_editing_locked() is False


def _load_project_via_dialog(
    main_window: MainWindow,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    payload: dict[str, object],
) -> None:
    project_path = tmp_path / "state_network_guardrails_load.kin"
    project_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getOpenFileName",
        lambda *args, **kwargs: (str(project_path), "Kindred Project (*.kin)"),
    )
    main_window.project_controller.load_project()


def test_state_network_editor_is_not_a_toplevel_window(main_window: MainWindow):
    """
    Regression test for State Network Editor open-time instability.

    The editor widget is owned by the Mechanism editor and should not be constructed as a
    top-level window (Qt.Window). Some Qt platforms behave poorly when a top-level widget
    is repeatedly re-parented into dialogs.
    """
    editor = main_window._mechanism_editor._state_network_editor
    assert editor.parent() is main_window._mechanism_editor
    assert not bool(editor.windowFlags() & QtCore.Qt.Window)


def test_open_state_network_dialog_is_non_blocking_and_restores_parent(
    main_window: MainWindow,
    monkeypatch: pytest.MonkeyPatch,
):
    """
    Opening the dialog should not block the caller, and closing it should restore the
    editor widget to its original owner.
    """
    _unlock_reactions_editing(main_window, monkeypatch)
    editor = main_window._mechanism_editor._state_network_editor
    start_parent = editor.parent()

    returned = {"value": False}
    returned_visible_in_dialog_loop = {"value": None}

    def _close_from_event_loop() -> None:
        dialog = _find_state_network_dialog()
        returned_visible_in_dialog_loop["value"] = bool(returned["value"])

        # Exercise a minimal interaction path while the dialog is open.
        editor._add_state_btn.click()
        _process_events_bounded()
        assert editor._states_table.rowCount() >= 1

        dialog.reject()

    # Schedule closure via the Qt event loop so this test cannot hang even if the dialog
    # is implemented with a blocking exec().
    QtCore.QTimer.singleShot(0, _close_from_event_loop)
    main_window._open_state_network()
    returned["value"] = True
    QtWidgets.QApplication.processEvents()

    assert returned_visible_in_dialog_loop["value"] is True
    assert editor.parent() is start_parent


def test_locked_state_network_dialog_opens_read_only_and_blocks_mutation(main_window: MainWindow):
    editor = main_window._mechanism_editor._state_network_editor
    baseline = "\n".join(
        [
            "state: A, kind=GS, energy=0, energy_unit=kJ/mol, degeneracy=1",
            "state: TS1, kind=TS, energy=10, energy_unit=kJ/mol, degeneracy=1",
            "edge: A,TS1",
        ]
    )
    editor.set_state_network_dsl(baseline)

    observed = {"dsl": None}

    def _inspect_and_close() -> None:
        dialog = _find_state_network_dialog()
        info_label = dialog.findChild(QtWidgets.QLabel, "stateNetworkDialogInfoLabel")
        assert info_label is not None
        info_text = info_label.text().lower()
        assert "read-only" in info_text
        assert "allow editing" in info_text

        assert editor._add_state_btn.isEnabled() is False
        assert editor._remove_state_btn.isEnabled() is False
        assert editor._add_edge_btn.isEnabled() is False
        assert editor._remove_edge_btn.isEnabled() is False
        assert editor._states_table.editTriggers() == QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        assert editor._edges_table.editTriggers() == QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers

        rows_before = editor._states_table.rowCount()
        editor._add_state_btn.click()
        _process_events_bounded()

        assert editor._states_table.rowCount() == rows_before
        observed["dsl"] = main_window.mechanism_state_network_dsl_raw()
        dialog.reject()

    QtCore.QTimer.singleShot(0, _inspect_and_close)
    main_window._open_state_network()
    QtWidgets.QApplication.processEvents()

    assert observed["dsl"] == baseline


def test_unlocked_state_network_dialog_still_applies_immediate_edits(
    main_window: MainWindow,
    monkeypatch: pytest.MonkeyPatch,
):
    _unlock_reactions_editing(main_window, monkeypatch)
    editor = main_window._mechanism_editor._state_network_editor
    editor.clear()

    observed = {"dsl": None}

    def _edit_and_close() -> None:
        dialog = _find_state_network_dialog()
        info_label = dialog.findChild(QtWidgets.QLabel, "stateNetworkDialogInfoLabel")
        assert info_label is not None
        assert "read-only" not in info_label.text().lower()

        assert editor._add_state_btn.isEnabled() is True
        assert editor._add_edge_btn.isEnabled() is True
        assert editor._states_table.editTriggers() != QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        assert editor._edges_table.editTriggers() != QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers

        editor._add_state_btn.click()
        assert editor._states_table.rowCount() >= 1
        editor._states_table.item(editor._states_table.rowCount() - 1, 0).setText("A")
        _process_events_bounded()
        observed["dsl"] = main_window.mechanism_state_network_dsl_raw()
        dialog.reject()

    QtCore.QTimer.singleShot(0, _edit_and_close)
    main_window._open_state_network()
    _process_events_bounded()

    assert observed["dsl"] is not None
    assert "state:" in observed["dsl"]


def test_locking_state_network_dialog_closes_active_cell_editor_without_committing(
    main_window: MainWindow,
    monkeypatch: pytest.MonkeyPatch,
):
    _unlock_reactions_editing(main_window, monkeypatch)
    editor = main_window._mechanism_editor._state_network_editor
    editor.clear()

    observed = {"dsl": None}

    def _edit_then_lock() -> None:
        dialog = _find_state_network_dialog()
        editor._add_state_btn.click()
        _process_events_bounded()
        item = editor._states_table.item(0, 0)
        assert item is not None
        editor._states_table.editItem(item)
        _process_events_bounded()

        active_editor = editor.findChild(QtWidgets.QLineEdit)
        assert active_editor is not None
        active_editor.setText("LOCKED_WRITE")
        _process_events_bounded()

        main_window._set_mechanism_edit_locked(True)
        _process_events_bounded()

        editor._edges_table.setFocus()
        _process_events_bounded()

        assert editor._states_table.editTriggers() == QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        assert editor.findChild(QtWidgets.QLineEdit) is None
        observed["dsl"] = main_window.mechanism_state_network_dsl_raw()
        dialog.reject()

    QtCore.QTimer.singleShot(0, _edit_then_lock)
    main_window._open_state_network()
    _process_events_bounded()

    assert observed["dsl"] == ""


def test_state_network_dialog_updates_lock_banner_while_open(
    main_window: MainWindow,
    monkeypatch: pytest.MonkeyPatch,
):
    _unlock_reactions_editing(main_window, monkeypatch)
    editor = main_window._mechanism_editor._state_network_editor
    editor.clear()

    observed = {"locked_text": None, "unlocked_text": None}

    def _toggle_lock_and_close() -> None:
        dialog = _find_state_network_dialog()
        info_label = dialog.findChild(QtWidgets.QLabel, "stateNetworkDialogInfoLabel")
        assert info_label is not None
        assert "read-only" not in info_label.text().lower()

        main_window._set_mechanism_edit_locked(True)
        _process_events_bounded()
        observed["locked_text"] = info_label.text()
        assert "read-only" in observed["locked_text"].lower()
        assert "allow editing" in observed["locked_text"].lower()

        main_window._set_mechanism_edit_locked(False)
        _process_events_bounded()
        observed["unlocked_text"] = info_label.text()
        assert "read-only" not in observed["unlocked_text"].lower()
        assert "edit the state network" in observed["unlocked_text"].lower()

        dialog.reject()

    QtCore.QTimer.singleShot(0, _toggle_lock_and_close)
    main_window._open_state_network()
    _process_events_bounded()


def test_load_project_then_open_state_network_remains_locked_and_read_only(
    main_window: MainWindow,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    payload = {
        "mechanism": "reaction: A -> B; k=1.0",
        "notes": "loaded project notes",
        "state_network": "\n".join(
            [
                "state: A, kind=GS, energy=0, energy_unit=kJ/mol, degeneracy=1",
                "state: TS1, kind=TS, energy=10, energy_unit=kJ/mol, degeneracy=1",
                "edge: A,TS1",
            ]
        ),
    }

    _load_project_via_dialog(main_window, tmp_path, monkeypatch, payload)

    assert main_window.mechanism_editing_locked() is True
    assert main_window._mechanism_editor._reactions_text.isReadOnly() is True
    assert main_window.mechanism_state_network_dsl_raw() == str(payload["state_network"])

    observed = {"dsl": None}
    editor = main_window._mechanism_editor._state_network_editor

    def _inspect_and_close() -> None:
        dialog = _find_state_network_dialog()
        info_label = dialog.findChild(QtWidgets.QLabel, "stateNetworkDialogInfoLabel")
        assert info_label is not None
        assert "read-only" in info_label.text().lower()
        assert "allow editing" in info_label.text().lower()

        assert editor._add_state_btn.isEnabled() is False
        assert editor._remove_state_btn.isEnabled() is False
        assert editor._add_edge_btn.isEnabled() is False
        assert editor._remove_edge_btn.isEnabled() is False
        assert editor._states_table.editTriggers() == QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        assert editor._edges_table.editTriggers() == QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers

        rows_before = editor._states_table.rowCount()
        editor._add_state_btn.click()
        _process_events_bounded()
        assert editor._states_table.rowCount() == rows_before

        observed["dsl"] = main_window.mechanism_state_network_dsl_raw()
        dialog.reject()

    QtCore.QTimer.singleShot(0, _inspect_and_close)
    main_window._open_state_network()
    _process_events_bounded()

    assert observed["dsl"] == str(payload["state_network"])
