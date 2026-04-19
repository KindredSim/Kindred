from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from kindred.gui.fitting.window import FittingWindow
from kindred.gui.widgets.plot_tabs import PlotTabsWidget

pytestmark = [pytest.mark.gui]


class _RecordingDatasetManager:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def sync_fit_result_views(
        self,
        model_series: dict[str, dict[str, np.ndarray]],
        *,
        dataset_stats: dict[str, dict[str, float]] | None = None,
        dataset_ids: list[str] | None = None,
    ) -> None:
        self.calls.append(
            {
                "model_series": model_series,
                "dataset_stats": dataset_stats,
                "dataset_ids": dataset_ids,
            }
        )


def _grid_entry(*, name: str, scale: float) -> dict[str, object]:
    t = np.linspace(0.0, 1.0, 4)
    y = np.linspace(0.0, scale, t.size)
    return {
        "name": name,
        "t": t,
        "data_y": y,
        "model_x": t,
        "model_y": y * 0.9,
        "model_series": {"A": y * 0.9},
        "chi_squared": 0.1 * scale,
        "r_squared": None,
        "all_species": {"A": y},
        "current_species": "A",
    }


def _make_fitting_window(*, dataset_manager) -> FittingWindow:
    t = np.linspace(0.0, 1.0, 5)
    y_a = np.linspace(1.0, 0.5, t.size)
    return FittingWindow(
        mode="global",
        parameter_defs=[{"name": "k1", "value": 0.2, "min": 0.01, "max": 1.0}],
        dataset_entries=[
            {
                "id": "ds1",
                "label": "ds1",
                "t": t.copy(),
                "species_data": {"A": y_a.copy()},
                "selected_species": ["A"],
                "weight": 1.0,
                "include": True,
            }
        ],
        dataset_manager=dataset_manager,
        simulation_func=lambda _params: {"t": t.copy(), "species": {"A": y_a.copy()}},
        dataset_payloads=[{"id": "ds1", "t": t.copy(), "y": np.vstack([y_a.copy()]), "species": ["A"]}],
        dataset_weights={"ds1": 1.0},
    )


def test_plot_tabs_public_sync_seams_reuse_tab_and_preserve_grid_plot_items(qt_app) -> None:
    plot_tabs = PlotTabsWidget()
    try:
        panel_first = plot_tabs.sync_dataset_tab(
            "ds1",
            t=np.asarray([0.0, 1.0, 2.0], dtype=float),
            data_y=np.asarray([1.0, 0.5, 0.2], dtype=float),
            model_x=np.asarray([0.0, 1.0, 2.0], dtype=float),
            model_y=np.asarray([0.9, 0.45, 0.18], dtype=float),
            ylabel="A",
            all_species={"A": np.asarray([1.0, 0.5, 0.2], dtype=float)},
            chi_squared=0.5,
            r_squared=None,
            model_series={"A": np.asarray([0.9, 0.45, 0.18], dtype=float)},
        )
        panel_second = plot_tabs.sync_dataset_tab(
            "ds1",
            t=np.asarray([0.0, 1.0, 2.0], dtype=float),
            data_y=np.asarray([1.1, 0.6, 0.3], dtype=float),
            model_x=np.asarray([0.0, 1.0, 2.0], dtype=float),
            model_y=np.asarray([0.95, 0.5, 0.25], dtype=float),
            ylabel="A",
            all_species={"A": np.asarray([1.1, 0.6, 0.3], dtype=float)},
            chi_squared=0.25,
            r_squared=None,
            model_series={"A": np.asarray([0.95, 0.5, 0.25], dtype=float)},
        )

        assert panel_second is panel_first

        plot_tabs.sync_dataset_grid([_grid_entry(name="DS1", scale=1.0), _grid_entry(name="DS2", scale=2.0)])
        qt_app.processEvents()

        plot_ids_before = [id(item) for item in getattr(plot_tabs._grid_view, "_plot_items", [])]
        assert plot_ids_before

        plot_tabs.sync_dataset_grid([_grid_entry(name="DS1", scale=3.0), _grid_entry(name="DS2", scale=4.0)])
        qt_app.processEvents()

        plot_ids_after = [id(item) for item in getattr(plot_tabs._grid_view, "_plot_items", [])]
        assert plot_ids_after == plot_ids_before
    finally:
        plot_tabs.close()
        qt_app.processEvents()


def test_fitting_window_delegates_fit_view_map_updates_to_dataset_manager(qt_app) -> None:
    manager = _RecordingDatasetManager()
    window = _make_fitting_window(dataset_manager=manager)
    try:
        model_series = {"ds1": {"A": np.asarray([1.0, 0.9, 0.8, 0.7, 0.6])}}
        dataset_stats = {"ds1": {"chi_squared": 1.25, "r_squared": 0.9}}

        window._update_dataset_views_from_maps(model_series, dataset_stats)

        assert len(manager.calls) == 1
        assert manager.calls[0]["model_series"] is model_series
        assert manager.calls[0]["dataset_stats"] is dataset_stats
        assert manager.calls[0]["dataset_ids"] is None
    finally:
        window.close()
        qt_app.processEvents()


def test_fitting_window_delegates_global_fit_result_updates_to_dataset_manager(qt_app) -> None:
    manager = _RecordingDatasetManager()
    window = _make_fitting_window(dataset_manager=manager)
    try:
        result = SimpleNamespace(
            model_series={"ds1": {"A": np.asarray([1.0, 0.8, 0.6, 0.4, 0.2])}},
            dataset_info=[SimpleNamespace(dataset_id="ds1", chi_squared=0.5, r_squared=0.97)],
        )

        window._update_dataset_views_from_global(result)

        assert len(manager.calls) == 1
        assert manager.calls[0]["model_series"] is result.model_series
        assert manager.calls[0]["dataset_stats"] == {"ds1": {"chi_squared": 0.5, "r_squared": 0.97}}
        assert manager.calls[0]["dataset_ids"] == ["ds1"]
    finally:
        window.close()
        qt_app.processEvents()


def test_fitting_window_targets_dataset_manager_updates_only_for_result_datasets(qt_app) -> None:
    manager = _RecordingDatasetManager()
    t = np.linspace(0.0, 1.0, 5)
    y_a = np.linspace(1.0, 0.5, t.size)
    window = FittingWindow(
        mode="global",
        parameter_defs=[{"name": "k1", "value": 0.2, "min": 0.01, "max": 1.0}],
        dataset_entries=[
            {
                "id": "ds1",
                "label": "ds1",
                "t": t.copy(),
                "species_data": {"A": y_a.copy()},
                "selected_species": ["A"],
                "weight": 1.0,
                "include": True,
            },
            {
                "id": "ds2",
                "label": "ds2",
                "t": t.copy(),
                "species_data": {"A": y_a.copy()},
                "selected_species": ["A"],
                "weight": 1.0,
                "include": False,
            },
        ],
        dataset_manager=manager,
        simulation_func=lambda _params: {"t": t.copy(), "species": {"A": y_a.copy()}},
        dataset_payloads=[
            {"id": "ds1", "t": t.copy(), "y": np.vstack([y_a.copy()]), "species": ["A"]},
            {"id": "ds2", "t": t.copy(), "y": np.vstack([y_a.copy()]), "species": ["A"]},
        ],
        dataset_weights={"ds1": 1.0, "ds2": 1.0},
    )
    try:
        result = SimpleNamespace(
            model_series={"ds1": {"A": np.asarray([1.0, 0.8, 0.6, 0.4, 0.2])}},
            dataset_info=[SimpleNamespace(dataset_id="ds1", chi_squared=0.5, r_squared=0.97)],
            completion=None,
        )

        window._update_dataset_views_from_global(result)

        assert len(manager.calls) == 1
        assert manager.calls[0]["model_series"] is result.model_series
        assert manager.calls[0]["dataset_stats"] == {"ds1": {"chi_squared": 0.5, "r_squared": 0.97}}
        assert manager.calls[0]["dataset_ids"] == ["ds1"]
    finally:
        window.close()
        qt_app.processEvents()
