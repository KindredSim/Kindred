from __future__ import annotations

import pytest
from PySide6 import QtCore, QtWidgets

from kindred.gui.widgets.mechanism_editor import MechanismEditorTabbed


pytestmark = [pytest.mark.gui]


def _click_at_global(qtbot, owner: QtWidgets.QWidget, global_pos: QtCore.QPoint) -> None:
    qtbot.waitUntil(
        lambda: QtWidgets.QApplication.widgetAt(global_pos) is not None
        or owner.childAt(owner.mapFromGlobal(global_pos)) is not None,
        timeout=1000,
    )
    target = QtWidgets.QApplication.widgetAt(global_pos)
    if target is None:
        target = owner.childAt(owner.mapFromGlobal(global_pos))
    assert target is not None
    qtbot.mouseClick(target, QtCore.Qt.LeftButton, pos=target.mapFromGlobal(global_pos))
    QtWidgets.QApplication.processEvents()


def test_mechanism_editor_tabs_switch_back_to_reactions_via_click(qt_app, qtbot):
    """
    Regression: A widget must not overlay the QTabBar such that the "Reactions" tab becomes unclickable
    after switching to "Notes".
    """
    editor = MechanismEditorTabbed()
    qtbot.addWidget(editor)
    editor.resize(900, 650)
    editor.show()
    QtWidgets.QApplication.processEvents()

    tabs = editor._tabs
    tab_bar = tabs.tabBar()

    assert tabs.currentIndex() == 0

    notes_pos = tab_bar.mapToGlobal(tab_bar.tabRect(1).center())
    reactions_pos = tab_bar.mapToGlobal(tab_bar.tabRect(0).center())

    _click_at_global(qtbot, editor, notes_pos)
    assert tabs.currentIndex() == 1

    _click_at_global(qtbot, editor, reactions_pos)
    assert tabs.currentIndex() == 0
