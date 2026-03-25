import pytest
from PySide6 import QtCore, QtWidgets

from kindred.gui.widgets.pyqtgraph_plot_panel_impl import (
    PYQTGRAPH_AVAILABLE,
    PyQtGraphPlotPanel,
)


pytestmark = pytest.mark.gui


@pytest.mark.skipif(not PYQTGRAPH_AVAILABLE, reason="PyQtGraph not available")
def test_pyqtgraph_native_menus_disabled_and_custom_actions_present(qtbot, monkeypatch):
    panel = PyQtGraphPlotPanel()
    qtbot.addWidget(panel)
    panel.show()
    QtWidgets.QApplication.processEvents()

    assert panel._plot_item.menuEnabled() is False
    assert panel._plot_item.getViewBox().menuEnabled() is False

    scene = panel._plot_item.scene()
    export_calls = {"n": 0}

    def _fake_show_export_dialog(_self):
        export_calls["n"] += 1

    monkeypatch.setattr(type(scene), "showExportDialog", _fake_show_export_dialog)

    captured_menus = []

    def _fake_exec(self, *_args, **_kwargs):
        captured_menus.append(self)
        return None

    monkeypatch.setattr(QtWidgets.QMenu, "exec_", _fake_exec)

    panel._show_context_menu(QtCore.QPoint(0, 0))

    assert captured_menus, "Expected _show_context_menu to call QMenu.exec_"
    menu = captured_menus[0]

    def _action_index(text: str) -> int:
        for idx, action in enumerate(menu.actions()):
            if action.text() == text:
                return idx
        raise AssertionError(f"Missing action {text!r}")

    export_idx = _action_index("Export Plot...")
    mouse_idx = _action_index("Mouse Mode")
    reset_idx = _action_index("Reset View")

    assert export_idx < reset_idx
    assert mouse_idx < reset_idx
    assert menu.actions()[reset_idx - 1].isSeparator()

    mouse_action = menu.actions()[mouse_idx]
    mouse_menu = mouse_action.menu()
    assert mouse_menu is not None

    pan_action = next(a for a in mouse_menu.actions() if a.text() == "Pan (3-Button)")
    rect_action = next(a for a in mouse_menu.actions() if a.text() == "Rect Zoom (1-Button)")

    import pyqtgraph as pg

    viewbox = panel._plot_item.getViewBox()
    pan_action.trigger()
    assert viewbox.state.get("mouseMode") == pg.ViewBox.PanMode
    rect_action.trigger()
    assert viewbox.state.get("mouseMode") == pg.ViewBox.RectMode

    menu.actions()[export_idx].trigger()
    assert export_calls["n"] == 1
