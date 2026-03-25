import os

import pytest
from PySide6 import QtWidgets


pytestmark = [pytest.mark.gui]


def _find_dialog(title: str) -> QtWidgets.QDialog | None:
    for widget in QtWidgets.QApplication.topLevelWidgets():
        if isinstance(widget, QtWidgets.QDialog) and widget.windowTitle() == title:
            return widget
    return None


def _counts(dialog: QtWidgets.QDialog) -> tuple[int, int]:
    widgets = dialog.findChildren(QtWidgets.QWidget)
    combos = dialog.findChildren(QtWidgets.QComboBox)
    return len(widgets), len(combos)


def test_gui_computational_mode_does_not_create_persistent_cell_widgets_or_widget_leaks(main_window):
    # Ensure any optional diagnostics are off for deterministic test output.
    os.environ.pop("KINDRED_DEBUG_QT_LEAKS", None)

    main_window._open_computational_mode()
    QtWidgets.QApplication.processEvents()

    dialog = _find_dialog("Computational Mode")
    assert dialog is not None
    ui = getattr(dialog, "_cm_ui", None)
    assert ui is not None

    widgets0, combos0 = _counts(dialog)
    assert widgets0 > 0
    assert combos0 > 0  # pressure unit + energy unit

    # Add rows and ensure we don't create per-row QComboBox cell widgets (delegate-based editors only).
    for _ in range(25):
        ui.add_species_btn.click()
    QtWidgets.QApplication.processEvents()

    assert ui.species_table.rowCount() >= 25

    widgets1, combos1 = _counts(dialog)
    assert combos1 == combos0, "QComboBox count grew after adding rows (persistent cell widgets leak class)"

    # Select a row and process events; counts should plateau.
    ui.species_table.setCurrentCell(0, 0)
    ui.species_table.selectRow(0)
    QtWidgets.QApplication.processEvents()

    for _ in range(50):
        QtWidgets.QApplication.processEvents()

    widgets2, combos2 = _counts(dialog)
    assert combos2 == combos0
    assert widgets2 <= widgets1 + 5, "Widget count kept growing during idle event processing"

    # Remove all rows; any editor widgets should have been transient and not leak.
    ui.species_table.setRowCount(0)
    QtWidgets.QApplication.processEvents()
    widgets3, combos3 = _counts(dialog)
    assert combos3 == combos0
    assert widgets3 <= widgets0 + 10, "Widgets did not return near baseline after clearing the table"
