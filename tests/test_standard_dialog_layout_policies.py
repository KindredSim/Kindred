from __future__ import annotations

import importlib.util

import pytest
from PySide6 import QtWidgets

from kindred.gui.widgets.export_dialog import ExportDialog
from kindred.gui.widgets.solver_settings import SolverSettingsDialog


pytestmark = pytest.mark.gui


def _find_groupbox(dialog: QtWidgets.QDialog, title: str) -> QtWidgets.QGroupBox:
    groups = [g for g in dialog.findChildren(QtWidgets.QGroupBox) if g.title() == title]
    assert len(groups) == 1
    return groups[0]


def test_insert_dialog_removed_from_product():
    assert importlib.util.find_spec("kindred.gui.widgets.insert_dialog") is None


def test_export_dialog_uses_dynamic_layout_policies(qtbot):
    assert "sizeHint" not in ExportDialog.__dict__

    dialog = ExportDialog()
    qtbot.addWidget(dialog)

    mode = _find_groupbox(dialog, "Export mode")
    mode_policy = mode.sizePolicy()
    assert mode_policy.horizontalPolicy() == QtWidgets.QSizePolicy.Policy.Expanding
    assert mode_policy.verticalPolicy() == QtWidgets.QSizePolicy.Policy.Minimum

    scope = _find_groupbox(dialog, "Scope (Default mode only)")
    scope_policy = scope.sizePolicy()
    assert scope_policy.horizontalPolicy() == QtWidgets.QSizePolicy.Policy.Expanding
    assert scope_policy.verticalPolicy() == QtWidgets.QSizePolicy.Policy.Minimum

    layout = dialog.layout()
    assert isinstance(layout, QtWidgets.QBoxLayout)
    assert layout.stretch(3) == 1  # spacer stretch before buttons

    assert dialog.minimumWidth() == 0
    assert dialog.minimumHeight() == 0


def test_solver_settings_dialog_uses_dynamic_layout_policies(qtbot):
    assert "sizeHint" not in SolverSettingsDialog.__dict__

    dialog = SolverSettingsDialog()
    qtbot.addWidget(dialog)

    cache_section = dialog.findChild(QtWidgets.QWidget, "simulationCachingSection")
    assert cache_section is not None
    cache_policy = cache_section.sizePolicy()
    assert cache_policy.horizontalPolicy() == QtWidgets.QSizePolicy.Policy.Expanding
    assert cache_policy.verticalPolicy() == QtWidgets.QSizePolicy.Policy.Minimum

    assert dialog.findChildren(QtWidgets.QGroupBox) == []
    headers = [h for h in dialog.findChildren(QtWidgets.QLabel) if h.text() == "Simulation caching"]
    assert len(headers) == 1
    assert headers[0].font().bold()

    layout = dialog.layout()
    assert isinstance(layout, QtWidgets.QBoxLayout)

    boxes = dialog.findChildren(QtWidgets.QDialogButtonBox)
    assert len(boxes) == 1
    button_box = boxes[0]

    button_index = None
    for i in range(layout.count()):
        item = layout.itemAt(i)
        if item is not None and item.widget() is button_box:
            button_index = i
            break
    assert button_index is not None
    assert button_index > 0

    spacer_item = layout.itemAt(button_index - 1)
    assert spacer_item is not None
    assert spacer_item.spacerItem() is not None
    assert layout.stretch(button_index - 1) == 1

    assert dialog.minimumWidth() == 0
    assert dialog.minimumHeight() == 0
    assert dialog.maximumWidth() > 10_000
    assert dialog.maximumHeight() > 10_000
