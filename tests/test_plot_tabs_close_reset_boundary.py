from __future__ import annotations

import numpy as np
import pytest

from kindred.gui.plot_config import is_pyqtgraph_available
from kindred.gui.widgets.plot_tabs import PlotTabsWidget

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not is_pyqtgraph_available(), reason="pyqtgraph not installed"),
]


def _sync_dataset_tab(plot_tabs: PlotTabsWidget, *, scale: float = 1.0):
    t = np.asarray([0.0, 1.0, 2.0], dtype=float)
    data_y = np.asarray([1.0, 0.5, 0.2], dtype=float) * float(scale)
    model_y = data_y * 0.9
    return plot_tabs.sync_dataset_tab(
        "ds1",
        t=t,
        data_y=data_y,
        model_x=t,
        model_y=model_y,
        ylabel="A",
        all_species={"A": data_y},
        chi_squared=0.5 * float(scale),
        r_squared=None,
        model_series={"A": model_y},
    )


def test_remove_dataset_tab_routes_close_cleanup_through_panel_public_api(qt_app, monkeypatch) -> None:
    plot_tabs = PlotTabsWidget()
    try:
        panel = _sync_dataset_tab(plot_tabs)
        backend = panel._plot_panel

        assert hasattr(panel, "reset_for_tab_close")

        backend._t = np.asarray([9.0], dtype=float)
        backend._series = {"sentinel": np.asarray([1.0], dtype=float)}
        backend._visible = {"sentinel": True}
        backend._colors = {"sentinel": (1, 2, 3)}

        calls: list[str] = []

        def _stub_reset_for_tab_close() -> None:
            calls.append("called")

        monkeypatch.setattr(panel, "reset_for_tab_close", _stub_reset_for_tab_close, raising=True)

        plot_tabs.remove_dataset_tab("ds1")

        assert calls == ["called"]
        assert backend._t is not None
        assert set(backend._series.keys()) == {"sentinel"}
        assert backend._visible == {"sentinel": True}
        assert backend._colors == {"sentinel": (1, 2, 3)}
        assert plot_tabs._tabs.count() == 2
    finally:
        plot_tabs.close()
        qt_app.processEvents()


def test_remove_dataset_tab_clears_panel_and_backend_close_state(qt_app) -> None:
    plot_tabs = PlotTabsWidget()
    try:
        panel = _sync_dataset_tab(plot_tabs)
        backend = panel._plot_panel

        backend._dataset_scatter_items = {"A": object()}
        backend._dataset_model_items = {"A": object()}
        backend._plot_items = {"stale_curve": object()}
        backend._overlay_items = {("ds1", "A"): object()}
        backend._active_overlay_series = [object()]
        backend._annotations = [object()]
        backend._guide_items = [object()]
        backend._scalar_values = {"k1": 1.23}

        assert panel.export_payload() is not None
        assert panel._species_checkboxes

        plot_tabs.remove_dataset_tab("ds1")

        assert plot_tabs._tabs.count() == 2
        assert panel.export_payload() is None
        assert panel.get_dataset_data() == {}
        assert panel._species_checkboxes == {}
        assert backend._t is None
        assert backend._series == {}
        assert backend._visible == {}
        assert backend._colors == {}
        assert backend._dataset_scatter_items == {}
        assert backend._dataset_model_items == {}
        assert backend._plot_items == {}
        assert backend._overlay_items == {}
        assert backend._active_overlay_series == []
        assert backend._annotations == []
        assert backend._guide_items == []
        assert backend._scalar_values == {}
    finally:
        plot_tabs.close()
        qt_app.processEvents()


def test_sync_dataset_tab_updates_do_not_call_backend_clear(qt_app, monkeypatch) -> None:
    plot_tabs = PlotTabsWidget()
    try:
        panel = _sync_dataset_tab(plot_tabs, scale=1.0)

        clear_calls = {"count": 0}
        plot_item_clear_calls = {"count": 0}
        original_clear = panel._plot_panel.clear
        original_plot_item_clear = panel._plot_panel._plot_item.clear

        def _wrapped_clear() -> None:
            clear_calls["count"] += 1
            original_clear()

        def _wrapped_plot_item_clear() -> None:
            plot_item_clear_calls["count"] += 1
            original_plot_item_clear()

        monkeypatch.setattr(panel._plot_panel, "clear", _wrapped_clear, raising=True)
        monkeypatch.setattr(panel._plot_panel._plot_item, "clear", _wrapped_plot_item_clear, raising=True)

        panel_second = _sync_dataset_tab(plot_tabs, scale=2.0)

        assert panel_second is panel
        assert clear_calls["count"] == 0
        assert plot_item_clear_calls["count"] == 0
    finally:
        plot_tabs.close()
        qt_app.processEvents()
