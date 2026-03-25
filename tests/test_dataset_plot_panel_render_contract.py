from __future__ import annotations

import importlib.resources

import numpy as np
import pyqtgraph as pg
import pytest
from PySide6 import QtWidgets

from kindred.gui.plot_config import is_pyqtgraph_available
from kindred.gui.widgets.dataset_plot_panel import DatasetPlotPanel
from kindred.gui.widgets.plot_tabs import PlotTabsWidget

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not is_pyqtgraph_available(), reason="pyqtgraph not installed"),
]


def _sync_dataset_tab(plot_tabs: PlotTabsWidget, *, scale: float = 1.0) -> DatasetPlotPanel:
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
        all_species={"A": data_y, "B": data_y * 0.5},
        chi_squared=0.5 * float(scale),
        r_squared=None,
        model_series={"A": model_y},
    )


def test_dataset_plot_panel_source_uses_backend_render_contract_only() -> None:
    source = (
        importlib.resources.files("kindred.gui.widgets")
        .joinpath("dataset_plot_panel.py")
        .read_text(encoding="utf-8")
    )

    assert ".render_dataset_layers(" in source
    assert "._plot_item" not in source


def test_dataset_plot_panel_delegates_rendering_to_backend_public_contract(qt_app, monkeypatch) -> None:
    panel = DatasetPlotPanel(dataset_name="ds1")
    try:
        calls: list[dict[str, object]] = []

        def _record_render_dataset_layers(*, data_t, dataset_series, model_t=None, model_series=None, visible_species, xlabel, ylabel) -> None:
            calls.append(
                {
                    "data_t": np.asarray(data_t, dtype=float),
                    "dataset_series": {str(k): np.asarray(v, dtype=float) for k, v in dataset_series.items()},
                    "model_t": None if model_t is None else np.asarray(model_t, dtype=float),
                    "model_series": None if model_series is None else {str(k): np.asarray(v, dtype=float) for k, v in model_series.items()},
                    "visible_species": tuple(str(name) for name in visible_species),
                    "xlabel": str(xlabel),
                    "ylabel": str(ylabel),
                }
            )

        monkeypatch.setattr(panel._plot_panel, "render_dataset_layers", _record_render_dataset_layers, raising=False)

        t = np.asarray([0.0, 1.0, 2.0], dtype=float)
        a = np.asarray([1.0, 0.5, 0.2], dtype=float)
        b = np.asarray([0.4, 0.3, 0.1], dtype=float)
        panel.set_data(t, a, xlabel="Time", ylabel="A", all_species={"A": a, "B": b})

        assert len(calls) == 1
        assert set(calls[0]["dataset_series"].keys()) == {"A", "B"}
        assert calls[0]["model_t"] is None
        assert calls[0]["model_series"] is None
        assert set(calls[0]["visible_species"]) == {"A", "B"}
        assert calls[0]["xlabel"] == "Time"
        assert calls[0]["ylabel"] == "Concentration"

        panel.plot_simulation_results(t, {"A": a * 0.9})

        assert len(calls) == 2
        assert set(calls[1]["dataset_series"].keys()) == {"A", "B"}
        assert calls[1]["model_t"] is not None
        assert set((calls[1]["model_series"] or {}).keys()) == {"A"}
        assert set(calls[1]["visible_species"]) == {"A", "B"}
    finally:
        panel.close()
        qt_app.processEvents()


def test_dataset_sync_updates_preserve_backend_guides_annotations_and_overlay_items(qt_app, monkeypatch) -> None:
    plot_tabs = PlotTabsWidget()
    try:
        panel = _sync_dataset_tab(plot_tabs, scale=1.0)
        backend = panel._plot_panel

        backend._add_guide_line(0.25)
        guide = backend._guide_items[-1]

        monkeypatch.setattr(QtWidgets.QInputDialog, "getText", lambda *args, **kwargs: ("note", True))
        backend._add_annotation()
        annotation = backend._annotations[-1]

        overlay = pg.ScatterPlotItem(
            x=np.asarray([0.0, 1.0, 2.0], dtype=float),
            y=np.asarray([0.1, 0.2, 0.3], dtype=float),
            pen=None,
            brush=pg.mkBrush(80, 80, 80, 150),
            size=6,
            symbol="o",
            name="overlay",
        )
        backend._plot_item.addItem(overlay)
        backend._overlay_items[("manual", "A")] = overlay

        assert guide.scene() is not None
        assert annotation.scene() is not None
        assert overlay.scene() is not None

        panel_second = _sync_dataset_tab(plot_tabs, scale=2.0)
        assert panel_second is panel

        assert guide.scene() is not None
        assert annotation.scene() is not None
        assert overlay.scene() is not None
    finally:
        plot_tabs.close()
        qt_app.processEvents()


def test_dataset_plot_panel_export_helpers_remain_panel_owned(qt_app) -> None:
    panel = DatasetPlotPanel(dataset_name="ds1")
    try:
        t = np.asarray([0.0, 1.0, 2.0], dtype=float)
        a = np.asarray([1.0, 0.5, 0.2], dtype=float)
        b = np.asarray([0.4, 0.3, 0.1], dtype=float)
        panel.set_data(t, a, xlabel="Time", ylabel="A", all_species={"A": a, "B": b})

        panel._species_checkboxes["B"].setChecked(False)
        qt_app.processEvents()

        payload = panel.export_payload()
        assert payload is not None
        assert set((payload.get("series") or {}).keys()) == {"A", "B"}

        header, rows = panel.build_visible_export("axis")
        assert header == ["Time", "A"]
        assert list(rows) == [[0.0, 1.0], [1.0, 0.5], [2.0, 0.2]]
    finally:
        panel.close()
        qt_app.processEvents()
