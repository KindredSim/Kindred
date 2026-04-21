import numpy as np

from kindred.gui.widgets.grid_plot_view import GridPlotView
import pytest

pytestmark = pytest.mark.gui



def test_grid_plot_view_renders_model_overlay_when_present(qt_app):
    view = GridPlotView()
    t = np.linspace(0.0, 1.0, 5)
    y = np.linspace(1.0, 0.5, 5)
    model = np.linspace(1.0, 0.4, 5)

    view.add_dataset(
        "ds",
        t,
        y,
        model_x=t,
        model_y=model,
        all_species={"A": y},
        current_species="A",
    )

    for _ in range(20):
        qt_app.processEvents()

    assert len(view._plot_items) == 1
    plot = view._plot_items[0]
    # Expect at least 2 plotted series: experimental + model overlay.
    assert len(plot.listDataItems()) >= 2
