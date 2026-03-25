from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from kindred.gui.controllers.dataset_manager import DatasetManager
from kindred.gui.fitting.window import FittingWindow
from kindred.gui.widgets.plot_tabs import PlotTabsWidget


class _Signal:
    def __init__(self) -> None:
        self._callbacks = []

    def connect(self, callback) -> None:
        self._callbacks.append(callback)


class _FakeDatasetPanel:
    def __init__(self) -> None:
        self.simulateRequested = _Signal()
        self.set_data_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
        self.plot_calls: list[tuple[np.ndarray, dict[str, np.ndarray]]] = []

    def set_data(self, *args, **kwargs) -> None:
        self.set_data_calls.append((args, kwargs))

    def plot_simulation_results(self, t: np.ndarray, species_data: dict[str, np.ndarray]) -> None:
        self.plot_calls.append((np.asarray(t, dtype=float), dict(species_data)))


class _FakeGridView:
    def __init__(self) -> None:
        self.datasets: list[dict[str, object]] = []

    def replace(self, dataset_entries) -> None:
        self.datasets = []
        for entry in dataset_entries:
            self.datasets.append(
                {
                    "name": entry["name"],
                    "t": np.asarray(entry["t"], dtype=float),
                    "data_y": np.asarray(entry["data_y"], dtype=float),
                    "model_x": None if entry.get("model_x") is None else np.asarray(entry["model_x"], dtype=float),
                    "model_y": None if entry.get("model_y") is None else np.asarray(entry["model_y"], dtype=float),
                    "model_series": None if entry.get("model_series") is None else dict(entry["model_series"]),
                    "chi_squared": entry.get("chi_squared"),
                    "r_squared": entry.get("r_squared"),
                    "all_species": None if entry.get("all_species") is None else dict(entry["all_species"]),
                    "current_species": entry.get("current_species"),
                }
            )


class _FakePlotTabs:
    def __init__(self) -> None:
        self.grid = _FakeGridView()
        self.tab_panels: dict[str, _FakeDatasetPanel] = {}
        self.tab_sync_calls: list[dict[str, object]] = []

    def sync_dataset_tab(
        self,
        name: str,
        *,
        t,
        data_y,
        model_x=None,
        model_y=None,
        ylabel: str,
        all_species=None,
        chi_squared=None,
        r_squared=None,
        model_series=None,
    ) -> _FakeDatasetPanel:
        panel = self.tab_panels.get(str(name))
        if panel is None:
            panel = _FakeDatasetPanel()
            self.tab_panels[str(name)] = panel
        panel.set_data(
            np.asarray(t, dtype=float),
            np.asarray(data_y, dtype=float),
            model_x=None if model_x is None else np.asarray(model_x, dtype=float),
            model_y=None if model_y is None else np.asarray(model_y, dtype=float),
            xlabel="Time",
            ylabel=ylabel,
            all_species=all_species,
        )
        if isinstance(model_series, dict) and model_series and model_x is not None:
            panel.plot_simulation_results(np.asarray(model_x, dtype=float), dict(model_series))
        self.tab_sync_calls.append(
            {
                "name": str(name),
                "chi_squared": chi_squared,
                "r_squared": r_squared,
                "ylabel": ylabel,
                "all_species": None if all_species is None else dict(all_species),
            }
        )
        return panel

    def sync_dataset_grid(self, dataset_entries) -> None:
        self.grid.replace(dataset_entries)

    def remove_dataset_tab(self, _name: str) -> None:
        return None


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


@pytest.mark.unit
def test_dataset_manager_sync_fit_result_views_uses_plot_tabs_public_sync_contract() -> None:
    t = np.linspace(0.0, 1.0, 4)
    dataset = {
        "t": t,
        "species": {
            "A": np.asarray([1.0, 0.7, 0.4, 0.2]),
            "B": np.asarray([0.2, 0.3, 0.5, 0.8]),
        },
    }
    plot_tabs = _FakePlotTabs()
    manager = DatasetManager(plot_tabs=plot_tabs, dataset_resolver=lambda name: dataset if name == "ds1" else None)

    manager.sync_fit_result_views(
        {"ds1": {"B": np.asarray([0.25, 0.35, 0.55, 0.75])}},
        dataset_stats={"ds1": {"chi_squared": 2.5, "r_squared": 0.75}},
    )

    assert list(plot_tabs.tab_panels.keys()) == ["ds1"]
    panel = plot_tabs.tab_panels["ds1"]
    assert len(panel.set_data_calls) == 1
    _args, kwargs = panel.set_data_calls[0]
    assert kwargs["ylabel"] == "B"
    assert set(kwargs["all_species"].keys()) == {"B"}

    assert len(plot_tabs.tab_sync_calls) == 1
    tab_call = plot_tabs.tab_sync_calls[0]
    assert tab_call["name"] == "ds1"
    assert tab_call["chi_squared"] == pytest.approx(2.5)
    assert tab_call["r_squared"] == pytest.approx(0.75)
    assert tab_call["ylabel"] == "B"
    tab_all_species = tab_call["all_species"]
    assert isinstance(tab_all_species, dict)
    assert set(tab_all_species.keys()) == {"B"}
    assert np.allclose(np.asarray(tab_all_species["B"], dtype=float), dataset["species"]["B"])

    assert len(panel.plot_calls) == 1
    plotted_t, plotted_series = panel.plot_calls[0]
    assert np.allclose(plotted_t, t)
    assert set(plotted_series.keys()) == {"B"}

    assert len(plot_tabs.grid.datasets) == 1
    grid_entry = plot_tabs.grid.datasets[0]
    assert grid_entry["name"] == "ds1"
    assert set((grid_entry["all_species"] or {}).keys()) == {"B"}
    assert grid_entry["current_species"] == "B"
    assert grid_entry["chi_squared"] == pytest.approx(2.5)
    assert grid_entry["r_squared"] == pytest.approx(0.75)


@pytest.mark.gui
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


@pytest.mark.gui
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


@pytest.mark.gui
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
