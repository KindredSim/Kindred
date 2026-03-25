import pytest
from PySide6 import QtCore, QtWidgets

from kindred.gui.main_window import MainWindow

pytestmark = [pytest.mark.gui]


def _bounded_process_events(iterations: int = 50) -> None:
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
    monkeypatch.setattr(main_window, "_prompt_mechanism_edit_unlock_warning", lambda: True)
    main_window._mechanism_edit_lock_action.trigger()
    assert main_window.mechanism_editing_locked() is False


def test_state_network_add_state_then_select_row_does_not_churn_widgets(
    main_window: MainWindow,
    monkeypatch: pytest.MonkeyPatch,
):
    """
    Regression test for catastrophic UI freeze/memory ramp during normal interaction:
    open dialog -> Add State -> select row.

    The fix removes persistent per-cell QWidget editors (especially QComboBox) and ensures
    the widget tree stays stable after the interaction.
    """
    _unlock_reactions_editing(main_window, monkeypatch)

    editor = main_window._mechanism_editor._state_network_editor

    validate_calls = {"n": 0}
    emitted = {"n": 0}
    original_validate = editor._validate

    def _wrapped_validate():
        validate_calls["n"] += 1
        return original_validate()

    editor._validate = _wrapped_validate  # type: ignore[method-assign]
    editor.stateNetworkChanged.connect(lambda: emitted.__setitem__("n", emitted["n"] + 1))

    exercised = {"value": False}

    def _exercise() -> None:
        dialog = _find_state_network_dialog()

        # Step 3: Add state.
        editor._add_state_btn.click()
        _bounded_process_events(20)
        assert editor._states_table.rowCount() >= 1

        # Step 4: Select the newly added row.
        editor._states_table.selectRow(editor._states_table.rowCount() - 1)
        _bounded_process_events(20)

        widgets_before = len(editor.findChildren(QtWidgets.QWidget))
        combos_before = len(editor.findChildren(QtWidgets.QComboBox))

        # Process a bounded number of event cycles; object counts should plateau.
        _bounded_process_events(50)

        assert len(editor.findChildren(QtWidgets.QWidget)) <= widgets_before + 2

        # No persistent per-cell QComboBox widgets should exist (delegate-based editing).
        assert len(editor.findChildren(QtWidgets.QComboBox)) == combos_before == 0

        # Add-state should coalesce into a single validate + change emission.
        assert validate_calls["n"] == 1
        assert emitted["n"] == 1

        exercised["value"] = True
        dialog.reject()

    # Run inside the Qt event loop so this cannot hang even if dialog handling regresses.
    QtCore.QTimer.singleShot(0, _exercise)
    main_window._open_state_network()
    _bounded_process_events(50)

    assert exercised["value"] is True
