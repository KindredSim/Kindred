from __future__ import annotations

import numpy as np
import pytest


pytestmark = [pytest.mark.gui]


def test_grid_plot_view_set_datasets_copies_payload_arrays_on_gui_thread(qt_app) -> None:
    """
    Stability guardrail: the grid plot must not retain references to caller-owned numpy buffers.

    Global Fit updates can deliver many best-so-far payloads. If the plotting layer holds views
    into caller-owned arrays, those arrays can later be reused/mutated, risking native crashes
    inside PyQtGraph/Qt during paint.
    """
    from kindred.gui.widgets.grid_plot_view import GridPlotView

    view = GridPlotView()
    try:
        x = np.linspace(0.0, 1.0, 8, dtype=float)
        y = np.linspace(1.0, 2.0, 8, dtype=float)
        model_x = np.linspace(0.0, 1.0, 10, dtype=float)
        model_y = np.linspace(1.0, 2.0, 10, dtype=float)

        datasets = [
            {
                "name": "ds",
                "data_x": x,
                "data_y": y,
                "model_x": model_x,
                "model_y": model_y,
                "model_series": {"A": model_y},
                "all_species": {"A": y},
                "current_species": "A",
                "x_label": "X",
                "x_units": "s",
            }
        ]

        view.set_datasets(datasets)
        qt_app.processEvents()

        stored = list(getattr(view, "_datasets", []) or [])
        assert stored, "Expected GridPlotView to store datasets."
        stored0 = stored[0]

        assert isinstance(stored0.get("data_x"), np.ndarray)
        assert isinstance(stored0.get("data_y"), np.ndarray)
        assert isinstance(stored0.get("model_x"), np.ndarray)
        assert isinstance(stored0.get("model_y"), np.ndarray)

        assert stored0["data_x"] is not x
        assert stored0["data_y"] is not y
        assert stored0["model_x"] is not model_x
        assert stored0["model_y"] is not model_y
        assert stored0.get("all_species", {}).get("A") is not y
        assert stored0.get("model_series", {}).get("A") is not model_y

        assert not np.shares_memory(stored0["data_x"], x)
        assert not np.shares_memory(stored0["data_y"], y)
        assert not np.shares_memory(stored0["model_x"], model_x)
        assert not np.shares_memory(stored0["model_y"], model_y)
        assert not np.shares_memory(stored0.get("all_species", {}).get("A"), y)
        assert not np.shares_memory(stored0.get("model_series", {}).get("A"), model_y)
    finally:
        view.close()
        qt_app.processEvents()
