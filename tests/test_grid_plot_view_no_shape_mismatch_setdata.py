from __future__ import annotations

import numpy as np
import pytest


pytestmark = [pytest.mark.gui]


def test_grid_plot_view_never_calls_curve_setdata_with_mismatched_shapes(qt_app, monkeypatch) -> None:
    from pyqtgraph.graphicsItems.PlotDataItem import PlotDataItem

    from kindred.gui.widgets.grid_plot_view import GridPlotView

    mismatches: list[tuple[int, int]] = []

    original = PlotDataItem.setData

    def _checked_set_data(self, *args, **kwargs):
        if len(args) >= 2:
            x = np.asarray(args[0]).reshape(-1)
            y = np.asarray(args[1]).reshape(-1)
            if x.size != y.size:
                mismatches.append((int(x.size), int(y.size)))
                # Skip calling into pyqtgraph to avoid partial state mutation / exception storms.
                return None
        return original(self, *args, **kwargs)

    monkeypatch.setattr(PlotDataItem, "setData", _checked_set_data, raising=True)

    view = GridPlotView()

    data_x = np.linspace(0.0, 1.0, 241, dtype=float)
    y_data = np.sin(2.0 * np.pi * data_x)

    model_x_ok = np.linspace(0.0, 1.0, 1000, dtype=float)
    y_model_ok = np.cos(2.0 * np.pi * model_x_ok)

    # First update: valid (x,y) lengths match on model grid.
    view.set_datasets(
        [
            {
                "name": "ds",
                "data_x": data_x,
                "data_y": y_data,
                "model_x": model_x_ok,
                "model_y": None,
                "model_series": {"A": y_model_ok},
                "chi_squared": None,
                "r_squared": None,
                "all_species": {"A": y_data},
                "current_species": "A",
                "x_label": "X",
                "x_units": None,
            }
        ]
    )
    qt_app.processEvents()

    # Second update: simulate the reported failure mode where model_x remains solver-grid (1000)
    # but the model series is only available on the sampled observation grid (241).
    view.set_datasets(
        [
            {
                "name": "ds",
                "data_x": data_x,
                "data_y": y_data,
                "model_x": model_x_ok,
                "model_y": None,
                "model_series": {"A": y_data.copy()},
                "chi_squared": None,
                "r_squared": None,
                "all_species": {"A": y_data},
                "current_species": "A",
                "x_label": "X",
                "x_units": None,
            }
        ]
    )
    qt_app.processEvents()

    # Trigger view-range change (pyqtgraph can re-enter setData during viewRangeChanged).
    assert view._plot_items
    view._plot_items[0].setXRange(float(data_x.min()), float(data_x.max()), padding=0)
    qt_app.processEvents()

    assert mismatches == []
