from __future__ import annotations

import pytest
from PySide6 import QtCore, QtWidgets

from kindred.gui.widgets.mechanism_editor import MechanismEditorTabbed


pytestmark = pytest.mark.gui


def _prime_left_slider_workspace(main_window) -> None:
    main_window._mechanism_editor._reactions_text.setPlainText(
        "reaction: A -> B; k=1.0\ninitial: A=1.0\ninitial: B=0.0"
    )
    main_window._batch_model.set_species(["A", "B"])
    model = main_window._batch_model
    assert model.setData(model.index(0, 1), "1.0")
    assert model.setData(model.index(0, 2), "0.0")
    table = main_window._batch_table
    assert table is not None
    table.setCurrentIndex(model.index(0, 0))
    selection = table.selectionModel()
    assert selection is not None
    selection.clearSelection()
    selection.select(
        model.index(0, 0),
        QtCore.QItemSelectionModel.Select | QtCore.QItemSelectionModel.Rows,
    )
    main_window._extract_and_populate_variables()


def _find_slider_visibility_action(main_window, entry_kind: str, name: str):
    picker = main_window.findChild(QtWidgets.QToolButton, "sliderVisibilityPickerButton")
    assert picker is not None
    menu = picker.menu()
    assert menu is not None
    main_window._mechanism_editor._rebuild_slider_visibility_menu()
    for action in menu.actions():
        if action.data() == (str(entry_kind), str(name)):
            return action
    raise AssertionError(f"Missing visibility action for {(entry_kind, name)!r}")


def test_empty_mechanism_editor_keeps_reactions_primary(qt_app, qtbot) -> None:
    editor = MechanismEditorTabbed()
    qtbot.addWidget(editor)
    editor.resize(700, 650)
    editor.show()
    QtWidgets.QApplication.processEvents()

    splitter = editor._reactions_splitter

    assert splitter.widget(1).height() < editor._reactions_text.height()


def test_side_first_default_shell_stacks_mechanism_sliders_and_batch_docks(main_window) -> None:
    main_window.show()
    QtWidgets.QApplication.processEvents()

    mechanism_geometry = main_window._mechanism_dock.geometry()
    sliders_geometry = main_window._sliders_dock.geometry()
    batch_geometry = main_window._batch_dock.geometry()

    assert main_window.dockWidgetArea(main_window._mechanism_dock) == QtCore.Qt.LeftDockWidgetArea
    assert main_window.dockWidgetArea(main_window._sliders_dock) == QtCore.Qt.LeftDockWidgetArea
    assert main_window.dockWidgetArea(main_window._batch_dock) == QtCore.Qt.LeftDockWidgetArea
    assert mechanism_geometry.left() == sliders_geometry.left() == batch_geometry.left()
    assert mechanism_geometry.width() == sliders_geometry.width() == batch_geometry.width()
    assert sliders_geometry.top() >= mechanism_geometry.bottom()
    assert batch_geometry.top() >= sliders_geometry.bottom()


def test_simulation_controls_are_positioned_above_batch_table(main_window) -> None:
    main_window.show()
    QtWidgets.QApplication.processEvents()

    panel = main_window._batch_panel
    controls = panel.findChild(QtWidgets.QWidget, "batchSolverControlsRow")

    assert controls is not None
    assert controls.geometry().y() < panel.batch_table.geometry().y()


def test_slider_surface_starts_with_no_visible_sliders_until_enabled(main_window, qtbot) -> None:
    _prime_left_slider_workspace(main_window)

    qtbot.addWidget(main_window)
    main_window.show()
    QtWidgets.QApplication.processEvents()

    mechanism_sliders = main_window._mechanism_editor._variable_sliders
    assert mechanism_sliders.has_variable("k1")
    mechanism_slider = mechanism_sliders._sliders["k1"]
    species_slider = main_window.findChild(QtWidgets.QSlider, "speciesSlider_A")

    assert species_slider is not None
    assert mechanism_slider.isVisible() is False
    assert species_slider.isVisible() is False


def test_slider_visibility_picker_stays_open_while_toggling_multiple_items(main_window, qtbot) -> None:
    _prime_left_slider_workspace(main_window)

    qtbot.addWidget(main_window)
    main_window.show()
    QtWidgets.QApplication.processEvents()

    picker = main_window.findChild(QtWidgets.QToolButton, "sliderVisibilityPickerButton")
    assert picker is not None
    menu = picker.menu()
    assert menu is not None

    menu.popup(picker.mapToGlobal(picker.rect().bottomLeft()))
    QtWidgets.QApplication.processEvents()
    assert menu.isVisible() is True

    first_action = next(action for action in menu.actions() if action.isCheckable())
    qtbot.mouseClick(menu, QtCore.Qt.LeftButton, pos=menu.actionGeometry(first_action).center())
    QtWidgets.QApplication.processEvents()

    assert menu.isVisible() is True


def test_enabled_sliders_do_not_show_separate_mechanism_and_concentration_headers(main_window, qtbot) -> None:
    _prime_left_slider_workspace(main_window)

    qtbot.addWidget(main_window)
    main_window.show()
    QtWidgets.QApplication.processEvents()

    _find_slider_visibility_action(main_window, "mechanism", "k1").trigger()
    _find_slider_visibility_action(main_window, "species", "A").trigger()
    QtWidgets.QApplication.processEvents()

    visible_headers = {
        label.text()
        for label in main_window.findChildren(QtWidgets.QLabel)
        if label.isVisible() and label.text() in {"Variable Sliders", "Initial Concentration Sliders"}
    }

    assert visible_headers == set()


def test_run_selected_sits_next_to_delete_in_batch_action_row(main_window, qtbot) -> None:
    qtbot.addWidget(main_window)
    main_window.show()
    QtWidgets.QApplication.processEvents()

    delete_btn = main_window.findChild(QtWidgets.QPushButton, "deleteBatchSetButton")
    run_btn = main_window.findChild(QtWidgets.QPushButton, "runSelectedButton")

    assert delete_btn is not None
    assert run_btn is not None
    assert delete_btn.geometry().y() == run_btn.geometry().y()
    assert run_btn.geometry().x() > delete_btn.geometry().x()
