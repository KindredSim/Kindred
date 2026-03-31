from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PySide6 import QtWidgets

from kindred.core.analysis.global_fitting import DatasetFitInfo, GlobalFitResult
from kindred.gui.fitting.window import FittingWindow


pytestmark = [pytest.mark.gui]


def _make_dataset_entries(count: int = 3) -> list[dict]:
    t = np.linspace(0.0, 1.0, 5)
    entries: list[dict] = []
    for idx in range(count):
        ds_id = f"ds{idx + 1}"
        species_name = chr(ord("A") + idx)
        entries.append(
            {
                "id": ds_id,
                "label": f"Dataset {idx + 1}",
                "t": t.copy(),
                "species_data": {species_name: np.linspace(1.0, 0.5 + 0.1 * idx, t.size)},
                "selected_species": [species_name],
                "weight": 1.0,
                "include": True,
            }
        )
    return entries


def _make_targets(entries: list[dict]) -> dict[str, list[str]]:
    return {
        str(entry["id"]): list(entry.get("selected_species") or sorted((entry.get("species_data") or {}).keys()))
        for entry in entries
    }


def _make_live_payload(entries: list[dict]) -> dict:
    plot_model_series: dict[str, dict[str, np.ndarray]] = {}
    plot_model_x: dict[str, np.ndarray] = {}
    dataset_stats: dict[str, dict[str, float]] = {}
    for entry in entries:
        ds_id = str(entry["id"])
        species_map = entry.get("species_data") or {}
        plot_model_series[ds_id] = {
            str(name): np.asarray(values, dtype=float).reshape(-1) * 0.9
            for name, values in species_map.items()
        }
        x_source = entry.get("x_obs") if entry.get("x_obs") is not None else entry.get("t")
        plot_model_x[ds_id] = np.asarray(x_source, dtype=float).reshape(-1)
        dataset_stats[ds_id] = {"chi_squared": 1.0, "r_squared": 0.95}
    return {
        "iteration": 7,
        "cost": 1.25,
        "shared_params": {"k1": 1.2, "k2": 0.5},
        "dataset_params": {str(entry["id"]): {} for entry in entries},
        "plot_model_series": plot_model_series,
        "plot_model_x": plot_model_x,
        "dataset_stats": dataset_stats,
    }


def _make_final_result(entries: list[dict]) -> GlobalFitResult:
    dataset_info: list[DatasetFitInfo] = []
    plot_model_series: dict[str, dict[str, np.ndarray]] = {}
    plot_model_x: dict[str, np.ndarray] = {}
    model_series: dict[str, dict[str, np.ndarray]] = {}
    objective_blocks: list[np.ndarray] = []
    for entry in entries:
        ds_id = str(entry["id"])
        x_source = entry.get("x_obs") if entry.get("x_obs") is not None else entry.get("t")
        x_values = np.asarray(x_source, dtype=float).reshape(-1)
        species_map = {
            str(name): np.asarray(values, dtype=float).reshape(-1) * 0.85
            for name, values in (entry.get("species_data") or {}).items()
        }
        plot_model_series[ds_id] = {name: values.copy() for name, values in species_map.items()}
        plot_model_x[ds_id] = x_values.copy()
        model_series[ds_id] = {name: values.copy() for name, values in species_map.items()}
        residuals = np.full(x_values.size, 0.1, dtype=float)
        objective_blocks.append(residuals)
        dataset_info.append(
            DatasetFitInfo(
                dataset_id=ds_id,
                r_squared=0.97,
                chi_squared=0.25,
                rmse=0.1,
                mae=0.1,
                residuals=residuals,
                n_points=int(x_values.size),
                weight=1.0,
            )
        )
    return GlobalFitResult(
        success=True,
        shared_params={"k1": 1.5},
        dataset_params={str(entry["id"]): {} for entry in entries},
        uncertainties=None,
        global_chi_squared=0.25,
        global_r_squared=0.97,
        dataset_info=dataset_info,
        nfev=11,
        message="ok",
        objective_residuals=np.concatenate(objective_blocks) if objective_blocks else np.asarray([], dtype=float),
        model_series=model_series,
        plot_model_x=plot_model_x,
        plot_model_series=plot_model_series,
    )


def _make_window(entries: list[dict]) -> FittingWindow:
    first_species = next(iter(entries[0]["species_data"].keys()))
    t = np.asarray(entries[0]["t"], dtype=float).reshape(-1)
    y = np.asarray(entries[0]["species_data"][first_species], dtype=float).reshape(-1)
    return FittingWindow(
        mode="global",
        parameter_defs=[{"name": "k1", "value": 1.23, "min": 0.01, "max": 10.0}],
        dataset_entries=entries,
        simulation_func=lambda _params: {"t": t.copy(), "species": {first_species: y.copy()}},
        dataset_payloads=[
            {"id": str(entry["id"]), "t": np.asarray(entry["t"], dtype=float), "y": np.vstack(list(entry["species_data"].values())), "species": list(entry["species_data"].keys())}
            for entry in entries
        ],
        dataset_weights={str(entry["id"]): float(entry.get("weight", 1.0)) for entry in entries},
    )


def test_run_results_tab_construction_builds_tracker_and_subtab_host(qt_app):
    from kindred.gui.fitting.run_results_tab import RunResultsTab

    tab = RunResultsTab(parent=None)
    try:
        assert tab.findChild(QtWidgets.QWidget, "global_fit_tracker_panel") is not None
        assert tab.findChild(QtWidgets.QTabBar, "global_fit_results_subtabs") is not None
        assert tab.findChild(QtWidgets.QStackedWidget, "global_fit_results_subtab_stack") is not None
    finally:
        tab.close()
        qt_app.processEvents()


def test_fit_tracker_panel_states_empty_live_and_final(qt_app):
    from kindred.gui.fitting.run_results_tab import FitTrackerPanel

    panel = FitTrackerPanel(parent=None)
    try:
        summary = panel.findChild(QtWidgets.QLabel, "global_fit_tracker_summary_label")
        params = panel.findChild(QtWidgets.QLabel, "global_fit_tracker_params_label")
        assert summary is not None
        assert params is not None
        assert "Run a fit" in summary.text()

        panel.update_from_best({"iteration": 4, "cost": 2.5, "shared_params": {"k1": 1.0}})
        assert "Iter 4" in summary.text()
        assert "Cost" in summary.text()
        assert "k1=" in params.text()

        panel.update_final(iteration=9, cost=1.25, shared_params={"k1": 1.5})
        assert "Iter 9" in summary.text()
        assert "delta" not in summary.text().lower()
        assert "k1=" in params.text()
    finally:
        panel.close()
        qt_app.processEvents()


def test_rebuild_subtabs_creates_dataset_tabs_plus_overlay(qt_app):
    from kindred.gui.fitting.run_results_tab import RunResultsTab

    entries = _make_dataset_entries(3)
    tab = RunResultsTab(parent=None)
    try:
        tab.rebuild_subtabs(entries, _make_targets(entries))
        tab_bar = tab.findChild(QtWidgets.QTabBar, "global_fit_results_subtabs")
        assert tab_bar is not None
        assert tab_bar.count() == 4
        assert [tab_bar.tabText(i) for i in range(tab_bar.count())] == [
            "Dataset 1",
            "Dataset 2",
            "Dataset 3",
            "All Datasets",
        ]
    finally:
        tab.close()
        qt_app.processEvents()


def test_rebuild_subtabs_replaces_previous_dataset_count(qt_app):
    from kindred.gui.fitting.run_results_tab import RunResultsTab

    tab = RunResultsTab(parent=None)
    try:
        entries = _make_dataset_entries(3)
        tab.rebuild_subtabs(entries, _make_targets(entries))
        fewer_entries = _make_dataset_entries(1)
        tab.rebuild_subtabs(fewer_entries, _make_targets(fewer_entries))
        tab_bar = tab.findChild(QtWidgets.QTabBar, "global_fit_results_subtabs")
        assert tab_bar is not None
        assert tab_bar.count() == 2
        assert [tab_bar.tabText(i) for i in range(tab_bar.count())] == ["Dataset 1", "All Datasets"]
    finally:
        tab.close()
        qt_app.processEvents()


def test_push_live_update_updates_tracker_and_subtab_payloads(qt_app):
    from kindred.gui.fitting.run_results_tab import RunResultsTab

    entries = _make_dataset_entries(2)
    tab = RunResultsTab(parent=None)
    try:
        tab.rebuild_subtabs(entries, _make_targets(entries))
        payload = _make_live_payload(entries)
        tab.push_live_update(payload)

        tracker = tab.findChild(QtWidgets.QLabel, "global_fit_tracker_summary_label")
        assert tracker is not None
        assert "Iter 7" in tracker.text()

        dataset_views = getattr(tab, "_dataset_plot_views", {})
        assert "ds1" in dataset_views
        ds1_grid = dataset_views["ds1"]
        assert getattr(ds1_grid, "_datasets", [])
        assert getattr(ds1_grid, "_datasets", [])[0]["model_series"]
    finally:
        tab.close()
        qt_app.processEvents()


def test_push_final_result_updates_tracker_and_subtab_payloads(qt_app):
    from kindred.gui.fitting.run_results_tab import RunResultsTab

    entries = _make_dataset_entries(2)
    tab = RunResultsTab(parent=None)
    try:
        tab.rebuild_subtabs(entries, _make_targets(entries))
        result = _make_final_result(entries)
        tab.push_final_result(result, entries)

        tracker = tab.findChild(QtWidgets.QLabel, "global_fit_tracker_summary_label")
        assert tracker is not None
        assert "Iter 11" in tracker.text()

        all_view = getattr(tab, "_all_datasets_plot_view", None)
        assert all_view is not None
        assert len(getattr(all_view, "_datasets", [])) == 2
    finally:
        tab.close()
        qt_app.processEvents()


def test_existing_summary_api_is_preserved(qt_app):
    from kindred.gui.fitting.run_results_tab import RunResultsTab

    tab = RunResultsTab(parent=None)
    try:
        tab.set_run_stamp({"solver": "LSODA"}, "abc123hash", "abc123")
        tab.update_statistics({"Datasets": 3, "SSQ": 1.5})
        tab.open_results_summary_dialog()
        qt_app.processEvents()

        assert tab._last_run_stamp_short == "abc123"
        assert tab._last_stats["Datasets"] == 3
        assert tab._stamp_dialog is not None
    finally:
        if getattr(tab, "_stamp_dialog", None) is not None:
            tab._stamp_dialog.close()
        tab.close()
        qt_app.processEvents()


def test_window_source_has_no_subset_widget_or_splitter_references() -> None:
    window_py = Path("/home/ph283804/kindred-vDEV/kindred/gui/fitting/window.py").read_text(encoding="utf-8")
    assert "_subset_widget" not in window_py
    assert "DatasetSubsetWidget" not in window_py
    assert "_main_splitter" not in window_py
    assert "global_fit_shell_splitter" not in window_py


def test_results_tab_switching_uses_single_host_without_splitter(qt_app):
    entries = _make_dataset_entries(2)
    window = _make_window(entries)
    try:
        window.show()
        qt_app.processEvents()

        assert window.findChild(QtWidgets.QSplitter, "global_fit_shell_splitter") is None
        assert not hasattr(window, "_subset_widget")

        top_tabs = window.findChild(QtWidgets.QTabBar, "global_fit_top_tabs")
        stack = window.findChild(QtWidgets.QStackedWidget, "global_fit_current_tab_stack")
        assert top_tabs is not None
        assert stack is not None

        results_index = [top_tabs.tabText(i) for i in range(top_tabs.count())].index("Results")
        top_tabs.setCurrentIndex(results_index)
        qt_app.processEvents()

        assert stack.currentIndex() == results_index
        assert stack.widget(results_index) is window._run_results_tab
    finally:
        window.close()
        qt_app.processEvents()
