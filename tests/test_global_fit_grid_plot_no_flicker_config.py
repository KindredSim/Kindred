from __future__ import annotations

import numpy as np
import pytest
from PySide6 import QtCore

from kindred.gui.widgets.dataset_subset_widget import DatasetSubsetWidget
from kindred.gui.widgets.grid_plot_view import GridPlotView

pytestmark = pytest.mark.gui


def _make_grid_datasets(*, scale: float = 1.0) -> list[dict]:
    x = np.linspace(0.0, 2.0, 25)
    y = np.linspace(0.0, scale * 1e6, x.size)
    all_species = {"A": y}
    return [
        {
            "name": "DS1",
            "data_x": x,
            "data_y": y,
            "model_x": x,
            "model_y": y * 0.9,
            "model_series": {"A": y * 0.9},
            "chi_squared": 1e-3 * scale,
            "r_squared": None,
            "all_species": all_species,
            "current_species": "A",
        },
        {
            "name": "DS2",
            "data_x": x,
            "data_y": y * 0.5,
            "model_x": x,
            "model_y": y * 0.45,
            "model_series": {"A": y * 0.45},
            "chi_squared": 2e-3 * scale,
            "r_squared": None,
            "all_species": {"A": y * 0.5},
            "current_species": "A",
        },
    ]


def test_grid_plot_view_locked_disables_scrollbars_and_freezes_axis_geometry(qtbot):
    view = GridPlotView()
    qtbot.addWidget(view)
    view.resize(900, 600)
    view.show()

    view.set_datasets(_make_grid_datasets(scale=1.0))
    QtCore.QCoreApplication.processEvents()

    view.set_autorange_locked(True)
    QtCore.QCoreApplication.processEvents()

    gl = getattr(view, "_graphics_layout", None)
    assert gl is not None
    assert gl.horizontalScrollBarPolicy() == QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
    assert gl.verticalScrollBarPolicy() == QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff

    # Rebuild with new data while locked; new axes should still be frozen.
    view.set_datasets(_make_grid_datasets(scale=10.0))
    QtCore.QCoreApplication.processEvents()

    plot_items = list(getattr(view, "_plot_items", []) or [])
    assert plot_items
    plot0 = plot_items[0]
    left_axis = plot0.getAxis("left")
    bottom_axis = plot0.getAxis("bottom")

    assert left_axis.style.get("autoExpandTextSpace") is False
    assert bottom_axis.style.get("autoExpandTextSpace") is False

    locked_left = getattr(view, "_locked_left_axis_width", None)
    locked_bottom = getattr(view, "_locked_bottom_axis_height", None)
    assert locked_left is not None
    assert locked_bottom is not None
    assert left_axis.width() == float(locked_left)
    assert bottom_axis.height() == float(locked_bottom)


def test_dataset_subset_widget_does_not_reapply_species_selection_when_union_unchanged(monkeypatch, qtbot):
    t = np.linspace(0.0, 1.0, 8)
    entries = [
        {
            "id": "ds1",
            "label": "DS1",
            "t": t,
            "species_data": {"A": np.linspace(0.0, 1.0, t.size), "B": np.linspace(1.0, 0.0, t.size)},
            "selected_species": ["A", "B"],
            "include": True,
        }
    ]
    widget = DatasetSubsetWidget(dataset_entries=entries)
    qtbot.addWidget(widget)

    monkeypatch.setattr(widget._selector, "selected_dataset_species", lambda: {"ds1": {"A", "B"}})

    calls = {"n": 0}
    original = widget._grid.set_species_selection

    def _wrapped(species_names):
        calls["n"] += 1
        return original(species_names)

    monkeypatch.setattr(widget._grid, "set_species_selection", _wrapped)

    # Ensure the first refresh applies the selection once, and the second refresh
    # does not reapply when the union of species is unchanged.
    setattr(widget, "_last_union_species_for_grid_selection", None)
    widget.refresh()
    widget.refresh()

    assert calls["n"] == 1

