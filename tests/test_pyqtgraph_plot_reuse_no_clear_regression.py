from __future__ import annotations

import numpy as np
import pytest

from kindred.gui.plot_config import get_plot_panel_class, is_pyqtgraph_available


pytestmark = [pytest.mark.gui]


@pytest.mark.skipif(not is_pyqtgraph_available(), reason="pyqtgraph not installed")
def test_simulation_curve_items_reused_without_remove_all(qt_app, monkeypatch):
    panel_cls = get_plot_panel_class()
    panel = panel_cls()

    t = np.array([0.0, 1.0, 2.0], dtype=float)
    series_first = {
        "A": np.array([1.0, 0.5, 0.2], dtype=float),
        "B": np.array([0.0, 0.3, 0.8], dtype=float),
    }
    series_second = {
        "A": np.array([1.2, 0.7, 0.1], dtype=float),
        "B": np.array([0.1, 0.4, 0.9], dtype=float),
    }
    overlays_first = [
        {
            "label": "set2",
            "t": t,
            "series": {
                "A": np.array([0.8, 0.4, 0.1], dtype=float),
                "B": np.array([0.2, 0.5, 0.7], dtype=float),
            },
        }
    ]
    overlays_second = [
        {
            "label": "set2",
            "t": t,
            "series": {
                "A": np.array([0.9, 0.45, 0.15], dtype=float),
                "B": np.array([0.25, 0.55, 0.75], dtype=float),
            },
        }
    ]

    panel.set_data(t, series_first, label="set1", overlays=overlays_first)
    first_ids = {key: id(item) for key, item in panel._plot_items.items()}
    assert first_ids

    removed: list[object] = []
    original_remove_item = panel._plot_item.removeItem

    def _spy_remove_item(item):
        removed.append(item)
        return original_remove_item(item)

    monkeypatch.setattr(panel._plot_item, "removeItem", _spy_remove_item, raising=True)

    panel.set_data(t, series_second, label="set1", overlays=overlays_second)

    second_ids = {key: id(item) for key, item in panel._plot_items.items()}
    assert set(second_ids.keys()) == set(first_ids.keys())
    assert second_ids == first_ids
    assert removed == []

    primary_label = panel._format_species_set_label("A", "set1")
    x_data, y_data = panel._plot_items[primary_label].getData()
    np.testing.assert_allclose(np.asarray(x_data, dtype=float), t)
    np.testing.assert_allclose(np.asarray(y_data, dtype=float), series_second["A"])

    panel.deleteLater()


@pytest.mark.skipif(not is_pyqtgraph_available(), reason="pyqtgraph not installed")
def test_dataset_overlay_scatter_items_reused_across_plot_refresh(qt_app, monkeypatch):
    panel_cls = get_plot_panel_class()
    panel = panel_cls()

    try:
        t = np.array([0.0, 1.0, 2.0], dtype=float)
        panel.set_data(
            t,
            {"A": np.array([1.0, 0.5, 0.2], dtype=float)},
            label="set1",
            owned_species=["A", "A_conc"],
        )
        panel.set_overlay_catalog(
            {
                "ds1": {
                    "t": t,
                    "species": {
                        "A": np.array([0.8, 0.4, 0.1], dtype=float),
                        "A_conc": np.array([0.9, 0.45, 0.15], dtype=float),
                    },
                }
            }
        )
        panel._overlay_panel._selected["ds1"] = True
        panel._overlay_panel._enabled_species["ds1"] = {"A_conc"}
        panel.set_selected_series(["A"])
        panel._on_x_axis_changed("t")
        for _ in range(3):
            qt_app.processEvents()

        overlay_key = ("ds1", "A")
        first_item = panel._overlay_items[overlay_key]
        first_id = id(first_item)

        removed: list[object] = []
        original_remove_item = panel._plot_item.removeItem

        def _spy_remove_item(item):
            removed.append(item)
            return original_remove_item(item)

        monkeypatch.setattr(panel._plot_item, "removeItem", _spy_remove_item, raising=True)

        panel._update_plot()
        for _ in range(3):
            qt_app.processEvents()

        assert overlay_key in panel._overlay_items
        assert id(panel._overlay_items[overlay_key]) == first_id
        assert first_item not in removed
    finally:
        panel.deleteLater()


@pytest.mark.skipif(not is_pyqtgraph_available(), reason="pyqtgraph not installed")
def test_simulation_plot_auto_range_refits_view_on_successive_set_data(qt_app):
    panel_cls = get_plot_panel_class()
    panel = panel_cls()

    try:
        t_first = np.array([0.0, 1.0, 2.0], dtype=float)
        series_first = {"A": np.array([0.0, 0.5, 1.0], dtype=float)}
        panel.set_data(t_first, series_first, label="set1", overlays=[])
        for _ in range(5):
            qt_app.processEvents()

        t_second = np.array([0.0, 10.0, 20.0], dtype=float)
        series_second = {"A": np.array([0.0, 50.0, 100.0], dtype=float)}
        panel.set_data(t_second, series_second, label="set1", overlays=[])
        for _ in range(5):
            qt_app.processEvents()

        view_range = panel._plot_item.viewRange()
        assert view_range[0][1] >= 20.0
        assert view_range[1][1] >= 100.0
    finally:
        panel.deleteLater()


@pytest.mark.skipif(not is_pyqtgraph_available(), reason="pyqtgraph not installed")
def test_simulation_plot_manual_range_survives_successive_set_data(qt_app):
    panel_cls = get_plot_panel_class()
    panel = panel_cls()

    try:
        t_first = np.array([0.0, 1.0, 2.0], dtype=float)
        series_first = {"A": np.array([0.0, 0.5, 1.0], dtype=float)}
        panel.set_data(t_first, series_first, label="set1", overlays=[])
        for _ in range(5):
            qt_app.processEvents()

        panel._toolbar.set_auto_range(False)
        panel._plot_item.setRange(xRange=(0.0, 1.0), yRange=(0.0, 1.0), padding=0.0)
        for _ in range(5):
            qt_app.processEvents()

        t_second = np.array([0.0, 10.0, 20.0], dtype=float)
        series_second = {"A": np.array([0.0, 50.0, 100.0], dtype=float)}
        panel.set_data(t_second, series_second, label="set1", overlays=[])
        for _ in range(5):
            qt_app.processEvents()

        view_range = panel._plot_item.viewRange()
        assert view_range[0][0] == pytest.approx(0.0)
        assert view_range[0][1] == pytest.approx(1.0)
        assert view_range[1][0] == pytest.approx(0.0)
        assert view_range[1][1] == pytest.approx(1.0)
    finally:
        panel.deleteLater()
