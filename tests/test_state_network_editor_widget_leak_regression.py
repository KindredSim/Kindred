import pytest
from PySide6 import QtWidgets

from kindred.gui.widgets.state_network_editor import StateNetworkEditor

pytestmark = [pytest.mark.gui]


_DSL = "\n".join(
    [
        "state: A, kind=GS, energy=0, energy_unit=kJ/mol, degeneracy=1",
        "state: B, kind=GS, energy=5, energy_unit=kJ/mol, degeneracy=1",
        "state: TS1, kind=TS, energy=20, energy_unit=kJ/mol, degeneracy=1",
        "edge: A,TS1",
        "edge: B,TS1",
    ]
)


def _combo_count(editor: StateNetworkEditor) -> int:
    return len(editor.findChildren(QtWidgets.QComboBox))


def test_state_network_editor_does_not_leak_cell_widgets(qt_app):
    """
    Regression test for UI freeze + runaway memory usage.

    StateNetworkEditor previously used persistent QComboBox widgets via setCellWidget().
    Those widgets are heavy and can leak/churn on some Qt platforms. The editor now uses
    delegate-based editing (ephemeral editors), so there should be no persistent QComboBox
    children after repeated loads/clears/removals.
    """
    editor = StateNetworkEditor()

    for _ in range(5):
        editor.set_state_network_dsl(_DSL)
        QtWidgets.QApplication.processEvents()
        assert editor._states_table.rowCount() == 3
        assert _combo_count(editor) == 0

    editor._states_table.selectRow(0)
    editor._remove_state_btn.click()
    QtWidgets.QApplication.processEvents()
    assert editor._states_table.rowCount() == 2
    assert _combo_count(editor) == 0

    editor.clear()
    QtWidgets.QApplication.processEvents()
    assert editor._states_table.rowCount() == 0
    assert _combo_count(editor) == 0

    editor.set_state_network_dsl(_DSL)
    QtWidgets.QApplication.processEvents()
    assert editor._states_table.rowCount() == 3
    assert _combo_count(editor) == 0
