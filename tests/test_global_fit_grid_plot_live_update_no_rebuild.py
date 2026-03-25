from __future__ import annotations

import numpy as np
import pytest
from PySide6 import QtCore

from kindred.gui.widgets.grid_plot_view import GridPlotView

pytestmark = pytest.mark.gui


def _make_grid_datasets(*, scale: float) -> list[dict]:
    x = np.linspace(0.0, 2.0, 25)
    y = np.linspace(0.0, scale * 1e6, x.size)
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
            "all_species": {"A": y},
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


def test_live_updates_do_not_clear_or_recreate_plotitems(monkeypatch, qtbot):
    view = GridPlotView()
    qtbot.addWidget(view)
    view.resize(900, 600)
    view.show()

    clear_calls = {"n": 0}
    gl = getattr(view, "_graphics_layout", None)
    assert gl is not None
    original_clear = gl.clear

    def _wrapped_clear(*args, **kwargs):
        clear_calls["n"] += 1
        return original_clear(*args, **kwargs)

    monkeypatch.setattr(gl, "clear", _wrapped_clear)

    view.set_datasets(_make_grid_datasets(scale=1.0))
    QtCore.QCoreApplication.processEvents()
    assert clear_calls["n"] == 1

    plot_ids_before = [id(p) for p in (getattr(view, "_plot_items", []) or [])]
    assert plot_ids_before

    # Simulate two live best-updates: same dataset names/order, changed y values.
    view.set_datasets(_make_grid_datasets(scale=2.0))
    QtCore.QCoreApplication.processEvents()
    view.set_datasets(_make_grid_datasets(scale=3.0))
    QtCore.QCoreApplication.processEvents()

    assert clear_calls["n"] == 1
    plot_ids_after = [id(p) for p in (getattr(view, "_plot_items", []) or [])]
    assert plot_ids_after == plot_ids_before

