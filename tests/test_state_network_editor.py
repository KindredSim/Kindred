from __future__ import annotations

import pytest

from PySide6.QtCore import Qt

from kindred.gui.widgets.state_network_editor import StateNetworkEditor


pytestmark = pytest.mark.unit


def test_state_network_editor_clears_hidden_preserved_fields_after_visible_row_edit(qt_app, qtbot) -> None:
    editor = StateNetworkEditor()
    qtbot.addWidget(editor)
    editor.set_state_network_dsl(
        "\n".join(
            [
                "state: AB, kind=GS, energy=0, energy_unit=kJ/mol, degeneracy=1, members=A+B, std=2, standard_state=C0",
                "state: C, kind=GS, energy=1, energy_unit=kJ/mol, degeneracy=1",
                "edge: AB,C",
            ]
        )
    )

    loaded = editor.get_state_network_dsl()
    assert "members=A+B" in loaded
    assert "std=2" in loaded
    assert "standard_state=C0" in loaded

    name_item = editor._states_table.item(0, 0)
    assert name_item is not None
    assert name_item.data(Qt.ItemDataRole.UserRole)

    name_item.setText("A_visible")

    exported = editor.get_state_network_dsl()
    assert "state: A_visible" in exported
    assert "members=" not in exported
    assert "std=" not in exported
    assert "standard_state=" not in exported
    assert not name_item.data(Qt.ItemDataRole.UserRole)
