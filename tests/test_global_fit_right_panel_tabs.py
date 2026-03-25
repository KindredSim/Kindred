import numpy as np
import pytest

from kindred.gui.fitting.window import FittingWindow


pytestmark = [pytest.mark.gui]


def _make_window() -> FittingWindow:
    t = np.linspace(0.0, 1.0, 5)
    y = np.linspace(1.0, 0.5, t.size)
    model = np.linspace(1.0, 0.4, t.size)
    return FittingWindow(
        mode="global",
        parameter_defs=[{"name": "k1", "value": 1.23, "min": 0.01, "max": 10.0}],
        dataset_entries=[
            {
                "id": "ds1",
                "label": "ds1",
                "t": t.copy(),
                "species_data": {"A": y.copy()},
                "selected_species": ["A"],
                "weight": 1.0,
                "include": True,
            }
        ],
        simulation_func=lambda _params: {"t": t.copy(), "species": {"A": model.copy()}},
        dataset_payloads=[{"id": "ds1", "t": t.copy(), "y": np.vstack([y.copy()]), "species": ["A"]}],
        dataset_weights={"ds1": 1.0},
    )


def test_global_fit_right_panel_tabs_include_setup_parameters_statistics(qt_app):
    window = _make_window()
    try:
        titles = [window._tabs.tabText(i) for i in range(window._tabs.count())]
        assert "Setup" in titles
        assert "Parameters" in titles
        assert "Statistics" in titles

        setup_widget = window._tabs.widget(titles.index("Setup"))
        params_widget = window._tabs.widget(titles.index("Parameters"))

        assert not setup_widget.isAncestorOf(window._param_table)
        assert params_widget.isAncestorOf(window._param_table)
        assert params_widget.isAncestorOf(window._method_combo)
    finally:
        window.close()
