from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest
from PySide6 import QtCore, QtWidgets
import kindred.gui.fitting.worker as fit_worker_module

from kindred.core.analysis.global_fitting import DatasetFitInfo, GlobalFitResult
from kindred.core.fitting_completion import GlobalFitCompletion
from kindred.gui.fitting.window import FittingWindow
from kindred.gui.fitting.worker import GlobalFitWorker


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
        shared_params={"k1": 1.5},
        dataset_params={str(entry["id"]): {} for entry in entries},
        uncertainties=None,
        global_chi_squared=0.25,
        global_r_squared=0.97,
        dataset_info=dataset_info,
        nfev=11,
        message="ok",
        completion=GlobalFitCompletion(
            status="ok",
            optimizer_converged=True,
            nonfinite_metrics=False,
        ),
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


def _make_best_update_worker(
    *,
    best_update_interval_s: float = 0.0,
    plot_update_interval_s: float = 2.0,
) -> GlobalFitWorker:
    t = np.array([0.0, 1.0], dtype=float)
    return GlobalFitWorker(
        [{"id": "ds1", "t": t.copy(), "y": np.array([1.0, 0.5], dtype=float), "species": "A"}],
        {"k1": 1.0},
        fit_evaluator=lambda _params: {"t": t.copy(), "species": {"A": np.array([1.0, 0.5], dtype=float)}},
        best_update_interval_s=best_update_interval_s,
        plot_update_interval_s=plot_update_interval_s,
    )


def _make_params_tab(parameter_state: list[dict]) -> QtWidgets.QWidget:
    from kindred.gui.fitting.parameters_ics_tab import ParametersIcsTab
    from kindred.gui.fitting.unified_species_table import UnifiedSpeciesTable

    entries = [{"id": "ds1", "label": "DS 1"}]
    species_table = UnifiedSpeciesTable(
        dataset_entries=list(entries),
        mechanism_species=["A"],
        dataset_entries_getter=lambda: list(entries),
        included_dataset_ids_getter=lambda: [str(entry["id"]) for entry in entries],
        dataset_label_getter=lambda ds_id: str(ds_id),
        dataset_weight_getter=lambda ds_id: 1.0,
        persist_dataset_weight_callback=lambda ds_id, w: None,
        dataset_manager_getter=lambda: None,
        worker_running_getter=lambda: False,
    )
    return ParametersIcsTab(
        parameter_state=[dict(entry) for entry in parameter_state],
        initial_parameter_snapshot=[dict(entry) for entry in parameter_state],
        global_dataset_params={},
        global_dataset_variable_params={},
        fixed_shared_params={},
        shared_param_definitions={},
        mechanism_species=["A"],
        dataset_entries=list(entries),
        prepared_param_names=["k1", "init:A"],
        selected_dataset_ids_getter=lambda: [str(entry["id"]) for entry in entries],
        dataset_entries_getter=lambda: list(entries),
        worker_running_getter=lambda: False,
        dataset_manager_getter=lambda: None,
        reactions_text_getter=lambda: "",
        integration_defaults=("BDF", 1e-6, 1e-12),
        config_defaults={},
        ic_panel=species_table,
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


def test_push_live_update_only_refreshes_visible_subtab_and_refreshes_stale_tab_on_switch(
    monkeypatch,
    qt_app,
):
    from kindred.gui.fitting.run_results_tab import RunResultsTab

    entries = _make_dataset_entries(3)
    tab = RunResultsTab(parent=None)
    try:
        tab.rebuild_subtabs(entries, _make_targets(entries))
        tab._subtabs.setCurrentIndex(0)

        dataset_calls: dict[str, int] = {ds_id: 0 for ds_id in tab._dataset_plot_views}
        for ds_id, plot_view in tab._dataset_plot_views.items():
            original = plot_view.set_datasets

            def _spy_set_datasets(
                datasets,
                *,
                _orig=original,
                _ds_id=ds_id,
            ):
                dataset_calls[_ds_id] += 1
                return _orig(datasets)

            monkeypatch.setattr(plot_view, "set_datasets", _spy_set_datasets)

        all_calls = {"count": 0}
        all_view = tab._all_datasets_plot_view
        assert all_view is not None
        original_all = all_view.set_datasets

        def _spy_all_set_datasets(datasets, *, _orig=original_all):
            all_calls["count"] += 1
            return _orig(datasets)

        monkeypatch.setattr(all_view, "set_datasets", _spy_all_set_datasets)

        payload = _make_live_payload(entries)
        tab.push_live_update(payload)

        assert dataset_calls["ds1"] == 1
        assert dataset_calls["ds2"] == 0
        assert dataset_calls["ds3"] == 0
        assert all_calls["count"] == 0

        ds2_payload_before = tab._dataset_plot_views["ds2"]._datasets[0]
        assert ds2_payload_before["model_series"] is None

        tab._subtabs.setCurrentIndex(1)
        qt_app.processEvents()

        assert dataset_calls["ds2"] == 1
        ds2_payload_after = tab._dataset_plot_views["ds2"]._datasets[0]
        assert ds2_payload_after["model_series"]
    finally:
        tab.close()
        qt_app.processEvents()


def test_push_final_result_refreshes_all_subtabs_even_when_one_is_visible(monkeypatch, qt_app):
    from kindred.gui.fitting.run_results_tab import RunResultsTab

    entries = _make_dataset_entries(3)
    tab = RunResultsTab(parent=None)
    try:
        tab.rebuild_subtabs(entries, _make_targets(entries))
        tab._subtabs.setCurrentIndex(0)

        dataset_calls: dict[str, int] = {ds_id: 0 for ds_id in tab._dataset_plot_views}
        for ds_id, plot_view in tab._dataset_plot_views.items():
            original = plot_view.set_datasets

            def _spy_set_datasets(
                datasets,
                *,
                _orig=original,
                _ds_id=ds_id,
            ):
                dataset_calls[_ds_id] += 1
                return _orig(datasets)

            monkeypatch.setattr(plot_view, "set_datasets", _spy_set_datasets)

        all_calls = {"count": 0}
        all_view = tab._all_datasets_plot_view
        assert all_view is not None
        original_all = all_view.set_datasets

        def _spy_all_set_datasets(datasets, *, _orig=original_all):
            all_calls["count"] += 1
            return _orig(datasets)

        monkeypatch.setattr(all_view, "set_datasets", _spy_all_set_datasets)

        tab.push_final_result(_make_final_result(entries), entries)

        assert dataset_calls == {"ds1": 1, "ds2": 1, "ds3": 1}
        assert all_calls["count"] == 1
    finally:
        tab.close()
        qt_app.processEvents()


def test_live_update_calls_species_selection_once_per_updated_plot(monkeypatch, qt_app):
    from kindred.gui.fitting.run_results_tab import RunResultsTab

    entries = _make_dataset_entries(3)
    tab = RunResultsTab(parent=None)
    try:
        tab.rebuild_subtabs(entries, _make_targets(entries))
        tab._subtabs.setCurrentIndex(0)
        for _ in range(5):
            qt_app.processEvents()

        species_calls: dict[str, int] = {ds_id: 0 for ds_id in tab._dataset_plot_views}
        for ds_id, plot_view in tab._dataset_plot_views.items():
            original = plot_view.set_species_selection

            def _spy_set_species_selection(
                species_names,
                *,
                _orig=original,
                _ds_id=ds_id,
            ):
                species_calls[_ds_id] += 1
                return _orig(species_names)

            monkeypatch.setattr(plot_view, "set_species_selection", _spy_set_species_selection)

        tab.push_live_update(_make_live_payload(entries))
        for _ in range(5):
            qt_app.processEvents()

        assert species_calls["ds1"] == 1
        assert species_calls["ds2"] == 0
        assert species_calls["ds3"] == 0
        assert inspect.getsource(type(tab)._apply_plot_species_selection).count("QTimer.singleShot") <= 1
    finally:
        tab.close()
        qt_app.processEvents()


def test_push_live_update_does_not_reapply_dark_mode_but_theme_change_does(monkeypatch, qt_app):
    from kindred.gui.fitting.run_results_tab import RunResultsTab

    entries = _make_dataset_entries(3)
    tab = RunResultsTab(parent=None)
    try:
        tab.rebuild_subtabs(entries, _make_targets(entries))

        calls: list[tuple[str, bool]] = []
        for plot_view in list(tab._dataset_plot_views.values()) + [tab._all_datasets_plot_view]:
            assert plot_view is not None

            def _spy_set_dark_mode(enabled: bool, *, _name=plot_view.objectName()):
                calls.append((_name, bool(enabled)))

            monkeypatch.setattr(plot_view, "set_dark_mode", _spy_set_dark_mode)

        tab.push_live_update(_make_live_payload(entries))
        assert calls == []

        tab.set_dark_mode(True)
        assert sorted(name for name, _enabled in calls) == sorted(
            plot_view.objectName()
            for plot_view in list(tab._dataset_plot_views.values()) + [tab._all_datasets_plot_view]
            if plot_view is not None
        )
        assert all(enabled is True for _name, enabled in calls)
    finally:
        tab.close()
        qt_app.processEvents()


def test_best_updated_lightweight_payload_skips_plot_fields_within_plot_interval(monkeypatch, qt_app):
    worker = _make_best_update_worker()
    emitted: list[dict] = []
    worker.bestUpdated.connect(lambda payload: emitted.append(payload))

    monkeypatch.setattr(fit_worker_module.time, "monotonic", lambda: 11.0)
    worker._last_heavy_emit_ts = 10.0

    build_calls = {"count": 0}

    def _fail_build(*, shared_params, dataset_params):
        build_calls["count"] += 1
        return {}, {}, {}, {}, {}

    monkeypatch.setattr(worker, "_build_best_payload_series", _fail_build)

    worker._maybe_emit_best(1, 0.9, {"k1": 0.9})

    assert len(emitted) == 1
    payload = emitted[0]
    assert payload["model_series"] is None
    assert payload["plot_model_series"] is None
    assert payload["plot_model_x"] is None
    assert payload["dataset_stats"] is None
    assert build_calls["count"] == 0


def test_best_updated_heavyweight_payload_includes_plot_fields_after_plot_interval(monkeypatch, qt_app):
    worker = _make_best_update_worker()
    emitted: list[dict] = []
    worker.bestUpdated.connect(lambda payload: emitted.append(payload))

    sentinel_payload = (
        {"ds1": {"A": np.array([0.9, 0.4], dtype=float)}},
        {"ds1": {"A": np.array([-0.1, -0.1], dtype=float)}},
        {"ds1": {"A": np.array([0.9, 0.4], dtype=float)}},
        {"ds1": np.array([0.0, 1.0], dtype=float)},
        {"ds1": {"chi_squared": 1.0, "r_squared": 0.95}},
    )

    monkeypatch.setattr(fit_worker_module.time, "monotonic", lambda: 12.5)
    worker._last_heavy_emit_ts = 10.0
    monkeypatch.setattr(
        worker,
        "_build_best_payload_series",
        lambda *, shared_params, dataset_params: sentinel_payload,
    )

    worker._maybe_emit_best(1, 0.9, {"k1": 0.9})

    assert len(emitted) == 1
    payload = emitted[0]
    assert payload["model_series"] == sentinel_payload[0]
    assert payload["plot_model_series"] == sentinel_payload[2]
    assert payload["plot_model_x"] == sentinel_payload[3]
    assert payload["dataset_stats"] == sentinel_payload[4]
    assert worker._last_heavy_emit_ts == pytest.approx(12.5)


def test_flush_pending_best_always_emits_heavyweight_payload(monkeypatch, qt_app):
    worker = _make_best_update_worker()
    emitted: list[dict] = []
    worker.bestUpdated.connect(lambda payload: emitted.append(payload))

    sentinel_payload = (
        {"ds1": {"A": np.array([0.8, 0.3], dtype=float)}},
        {"ds1": {"A": np.array([-0.2, -0.2], dtype=float)}},
        {"ds1": {"A": np.array([0.8, 0.3], dtype=float)}},
        {"ds1": np.array([0.0, 1.0], dtype=float)},
        {"ds1": {"chi_squared": 0.5, "r_squared": 0.99}},
    )

    monkeypatch.setattr(fit_worker_module.time, "monotonic", lambda: 20.5)
    monkeypatch.setattr(
        worker,
        "_build_best_payload_series",
        lambda *, shared_params, dataset_params: sentinel_payload,
    )
    worker._last_heavy_emit_ts = 20.0
    worker._best_iteration = 5
    worker._best_cost = 0.75
    worker._best_params = {"k1": 0.75}
    worker._pending_best = True

    worker._flush_pending_best()

    assert len(emitted) == 1
    payload = emitted[0]
    assert payload["model_series"] == sentinel_payload[0]
    assert payload["plot_model_series"] == sentinel_payload[2]
    assert payload["dataset_stats"] == sentinel_payload[4]


def test_flush_pending_best_emits_heavyweight_payload_after_lightweight_emit(monkeypatch, qt_app):
    worker = _make_best_update_worker()
    emitted: list[dict] = []
    worker.bestUpdated.connect(lambda payload: emitted.append(payload))

    sentinel_payload = (
        {"ds1": {"A": np.array([0.7, 0.2], dtype=float)}},
        {"ds1": {"A": np.array([-0.3, -0.3], dtype=float)}},
        {"ds1": {"A": np.array([0.7, 0.2], dtype=float)}},
        {"ds1": np.array([0.0, 1.0], dtype=float)},
        {"ds1": {"chi_squared": 0.25, "r_squared": 0.98}},
    )

    times = iter([11.0, 11.0, 11.5])
    monkeypatch.setattr(fit_worker_module.time, "monotonic", lambda: next(times))
    worker._last_heavy_emit_ts = 10.0
    monkeypatch.setattr(
        worker,
        "_build_best_payload_series",
        lambda *, shared_params, dataset_params: sentinel_payload,
    )

    worker._maybe_emit_best(1, 0.9, {"k1": 0.9})
    worker._flush_pending_best()

    assert len(emitted) == 2
    assert emitted[0]["model_series"] is None
    assert emitted[1]["model_series"] == sentinel_payload[0]
    assert emitted[1]["plot_model_series"] == sentinel_payload[2]
    assert emitted[1]["dataset_stats"] == sentinel_payload[4]


def test_window_lightweight_best_update_updates_tracker_without_starting_plot_timer(qt_app):
    entries = _make_dataset_entries(2)
    window = _make_window(entries)
    try:
        window._worker = _FakeWorker()
        stale_model_series = {"stale": {"A": np.array([1.0], dtype=float)}}
        stale_plot_series = {"stale": {"A": np.array([1.0], dtype=float)}}
        stale_plot_x = {"stale": np.array([0.0], dtype=float)}
        stale_stats = {"stale": {"chi_squared": 3.0}}
        window._latest_model_series = stale_model_series
        window._latest_plot_model_series = stale_plot_series
        window._latest_plot_model_x = stale_plot_x
        window._latest_dataset_stats = stale_stats

        payload = {
            "iteration": 7,
            "cost": 1.25,
            "shared_params": {"k1": 2.5},
            "dataset_params": {str(entry["id"]): {} for entry in entries},
            "model_series": None,
            "plot_model_series": None,
            "plot_model_x": None,
            "dataset_stats": None,
        }

        window._handle_global_best_update(payload, worker=window._worker)

        tracker = window._run_results_tab.findChild(QtWidgets.QLabel, "global_fit_tracker_summary_label")
        assert tracker is not None
        assert "Iter 7" in tracker.text()
        assert window._params_ics_tab._param_table.item(0, 3).text() == "2.5"
        assert window._pending_best_payload is None
        assert not window._pending_best_timer.isActive()
        assert window._latest_model_series is stale_model_series
        assert window._latest_plot_model_series is stale_plot_series
        assert window._latest_plot_model_x is stale_plot_x
        assert window._latest_dataset_stats is stale_stats
    finally:
        window._worker = None
        window.close()
        qt_app.processEvents()


def test_window_heavyweight_best_update_starts_plot_timer_and_refreshes_results(monkeypatch, qt_app):
    entries = _make_dataset_entries(2)
    window = _make_window(entries)
    try:
        window._worker = _FakeWorker()
        push_calls: list[dict] = []
        original_push = window._run_results_tab.push_live_update

        def _spy_push_live_update(payload: dict, **kwargs) -> None:
            push_calls.append(dict(payload))
            original_push(payload, **kwargs)

        monkeypatch.setattr(window._run_results_tab, "push_live_update", _spy_push_live_update)

        payload = _make_live_payload(entries)
        payload["model_series"] = {
            ds_id: {name: values.copy() for name, values in species_map.items()}
            for ds_id, species_map in payload["plot_model_series"].items()
        }
        window._handle_global_best_update(payload, worker=window._worker)

        assert window._pending_best_timer.isActive()
        assert isinstance(window._pending_best_payload, dict)
        assert set(window._latest_model_series.keys()) == {"ds1", "ds2"}

        window._apply_pending_best_update()
        qt_app.processEvents()

        assert len(push_calls) == 1
        ds1_payload = window._run_results_tab._dataset_plot_views["ds1"]._datasets[0]
        assert ds1_payload["model_series"]
    finally:
        window._worker = None
        window.close()
        qt_app.processEvents()


def test_window_heavyweight_best_update_does_not_rewrite_tracker_delta_on_deferred_refresh(qt_app):
    entries = _make_dataset_entries(2)
    window = _make_window(entries)
    try:
        window._worker = _FakeWorker()
        payload = _make_live_payload(entries)
        payload["iteration"] = 9
        payload["cost"] = 1.25
        payload["shared_params"] = {"k1": 2.5}
        payload["model_series"] = {
            ds_id: {name: values.copy() for name, values in species_map.items()}
            for ds_id, species_map in payload["plot_model_series"].items()
        }

        tracker = window._run_results_tab.findChild(QtWidgets.QLabel, "global_fit_tracker_summary_label")
        assert tracker is not None

        window._handle_global_best_update(payload, worker=window._worker)
        summary_before_refresh = tracker.text()
        assert "delta" not in summary_before_refresh.lower()

        window._apply_pending_best_update()
        qt_app.processEvents()

        assert tracker.text() == summary_before_refresh
        assert "delta 0.00e+00" not in tracker.text()
    finally:
        window._worker = None
        window.close()
        qt_app.processEvents()


def test_results_subtabs_hide_grid_controls_and_apply_dark_mode(monkeypatch, qt_app):
    from kindred.gui.fitting.run_results_tab import RunResultsTab

    entries = _make_dataset_entries(2)
    tab = RunResultsTab(parent=None)
    try:
        tab.set_dark_mode(True)
        tab.rebuild_subtabs(entries, _make_targets(entries))

        for plot_view in list(tab._dataset_plot_views.values()) + [tab._all_datasets_plot_view]:
            assert plot_view is not None
            controls = plot_view.findChild(QtWidgets.QWidget, "grid_plot_view_controls")
            assert controls is not None
            assert not controls.isVisible()

        calls: list[tuple[str, bool]] = []
        for plot_view in list(tab._dataset_plot_views.values()) + [tab._all_datasets_plot_view]:
            assert plot_view is not None
            monkeypatch.setattr(
                plot_view,
                "set_dark_mode",
                lambda enabled, name=plot_view.objectName(): calls.append((name, bool(enabled))),
            )
        tab.set_dark_mode(False)
        assert calls
        assert all(enabled is False for _, enabled in calls)
    finally:
        tab.close()
        qt_app.processEvents()


def test_push_best_update_updates_existing_value_and_last_fit_items_without_full_rebuild(monkeypatch, qt_app):
    parameter_state = [
        {
            "name": "k1",
            "param_name": "k1",
            "scope": "shared",
            "value": 1.0,
            "min": 0.1,
            "max": 10.0,
            "fit": True,
            "log10": False,
            "last_fit": 0.5,
        },
        {
            "name": "init:A (DS 1)",
            "param_name": "init:A",
            "scope": "dataset",
            "dataset_id": "ds1",
            "value": 2.0,
            "min": 0.0,
            "max": 5.0,
            "fit": True,
            "log10": False,
            "last_fit": 1.0,
        },
    ]
    tab = _make_params_tab(parameter_state)
    try:
        value_item_before = tab._param_table.item(0, 3)
        last_fit_item_before = tab._param_table.item(0, 6)
        dataset_value_item_before = tab._param_table.item(1, 3)
        dataset_last_fit_item_before = tab._param_table.item(1, 6)
        assert value_item_before is not None
        assert last_fit_item_before is not None
        assert dataset_value_item_before is not None
        assert dataset_last_fit_item_before is not None

        stable_columns_before = [
            [tab._param_table.item(row, col).text() for col in (2, 4, 5)]
            for row in range(tab._param_table.rowCount())
        ]

        def _fail_full_rebuild():
            raise AssertionError("push_best_update should not rebuild the full parameter table")

        monkeypatch.setattr(tab, "_populate_parameter_table", _fail_full_rebuild)

        tab.push_best_update(
            shared_params={"k1": 2.5},
            dataset_params={"ds1": {"init:A": 4.0}},
        )

        assert tab._param_table.rowCount() == 2
        assert tab._param_table.item(0, 3) is value_item_before
        assert tab._param_table.item(0, 6) is last_fit_item_before
        assert tab._param_table.item(1, 3) is dataset_value_item_before
        assert tab._param_table.item(1, 6) is dataset_last_fit_item_before
        assert tab._param_table.item(0, 3).text() == "2.5"
        assert tab._param_table.item(0, 6).text() == "2.5"
        assert tab._param_table.item(1, 3).text() == "4"
        assert tab._param_table.item(1, 6).text() == "4"
        assert stable_columns_before == [
            [tab._param_table.item(row, col).text() for col in (2, 4, 5)]
            for row in range(tab._param_table.rowCount())
        ]
    finally:
        tab.close()
        qt_app.processEvents()


def test_window_palette_change_reapplies_results_tab_dark_mode(qt_app, monkeypatch):
    entries = _make_dataset_entries(1)
    window = _make_window(entries)
    try:
        calls: list[bool] = []
        original = window._run_results_tab.set_dark_mode

        def _spy_set_dark_mode(enabled: bool):
            calls.append(bool(enabled))
            return original(enabled)

        monkeypatch.setattr(window._run_results_tab, "set_dark_mode", _spy_set_dark_mode)
        window.changeEvent(QtCore.QEvent(QtCore.QEvent.Type.PaletteChange))
        assert calls
        assert calls[-1] == window._current_results_dark_mode()
    finally:
        window.close()
        qt_app.processEvents()


def test_results_subtab_rebuild_honors_passed_selected_species(qt_app):
    t = np.linspace(0.0, 1.0, 5)
    entries = [
        {
            "id": "ds1",
            "label": "Dataset 1",
            "t": t.copy(),
            "species_data": {
                "A": np.linspace(1.0, 0.5, t.size),
                "B": np.linspace(0.2, 0.8, t.size),
            },
            "selected_species": ["A"],
            "weight": 1.0,
            "include": True,
        }
    ]
    window = _make_window(entries)
    try:
        initial_payload = window._run_results_tab._dataset_plot_views["ds1"]._datasets[0]
        assert initial_payload["current_species"] == "A"

        updated_entries = [dict(entries[0], selected_species=["B"])]
        window._run_results_tab.rebuild_subtabs(updated_entries, {"ds1": ["B"]})

        payload = window._run_results_tab._dataset_plot_views["ds1"]._datasets[0]
        assert payload["current_species"] == "B"
        assert sorted(payload["all_species"].keys()) == ["B"]
    finally:
        window.close()
        qt_app.processEvents()


def test_results_subtab_rebuild_falls_back_to_applied_targets_when_selection_missing(qt_app):
    t = np.linspace(0.0, 1.0, 5)
    entries = [
        {
            "id": "ds1",
            "label": "Dataset 1",
            "t": t.copy(),
            "species_data": {
                "A": np.linspace(1.0, 0.5, t.size),
                "B": np.linspace(0.2, 0.8, t.size),
            },
            "selected_species": ["A"],
            "weight": 1.0,
            "include": True,
        }
    ]
    window = _make_window(entries)
    try:
        updated_entries = [
            {
                "id": "ds1",
                "label": "Dataset 1",
                "t": t.copy(),
                "species_data": {
                    "A": np.linspace(1.0, 0.5, t.size),
                    "B": np.linspace(0.2, 0.8, t.size),
                },
                "weight": 1.0,
                "include": True,
            }
        ]
        window._dataset_entries = updated_entries
        window._run_results_tab.rebuild_subtabs(window._dataset_entries, window._results_fit_targets_by_dataset())

        payload = window._run_results_tab._dataset_plot_views["ds1"]._datasets[0]
        assert payload["current_species"] == "A"
        assert sorted(payload["all_species"].keys()) == ["A"]
    finally:
        window.close()
        qt_app.processEvents()


def test_rebuild_subtabs_selects_all_fitted_species_immediately(qt_app):
    from kindred.gui.fitting.run_results_tab import RunResultsTab

    t = np.linspace(0.0, 1.0, 5)
    entries = [
        {
            "id": "ds1",
            "label": "Dataset 1",
            "t": t.copy(),
            "species_data": {
                "A": np.linspace(1.0, 0.5, t.size),
                "B": np.linspace(0.2, 0.8, t.size),
            },
            "selected_species": ["A", "B"],
            "weight": 1.0,
            "include": True,
        }
    ]
    tab = RunResultsTab(parent=None)
    try:
        tab.rebuild_subtabs(entries, {"ds1": ["A", "B"]})
        for _ in range(5):
            qt_app.processEvents()

        plot_view = tab._dataset_plot_views["ds1"]
        assert sorted(plot_view._selected_species_list) == ["A", "B"]
        assert [item.text() for item in plot_view._species_list.selectedItems()] == ["A", "B"]
    finally:
        tab.close()
        qt_app.processEvents()


def test_existing_summary_api_is_preserved(qt_app):
    from kindred.gui.fitting.run_results_tab import RunResultsTab

    tab = RunResultsTab(parent=None)
    try:
        tab.set_run_stamp({"solver": "BDF"}, "abc123hash", "abc123")
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
    repo_root = Path(__file__).resolve().parents[1]
    window_py = (repo_root / "kindred" / "gui" / "fitting" / "window.py").read_text(encoding="utf-8")
    assert "_subset_widget" not in window_py
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


# ------------------------------------------------------------------
# Rebuild-subtabs guard during active fit
# ------------------------------------------------------------------

class _FakeWorker:
    """Minimal stand-in for a running QThread worker."""

    def isRunning(self) -> bool:
        return True


def test_rebuild_subtabs_not_called_during_fit(qt_app):
    """rebuild_subtabs must be deferred while a fit is running."""
    entries = _make_dataset_entries(2)
    window = _make_window(entries)
    try:
        window.show()
        qt_app.processEvents()

        # Simulate a running fit by injecting a fake worker
        window._worker = _FakeWorker()

        # Record calls to rebuild_subtabs
        calls: list[tuple] = []
        orig = window._run_results_tab.rebuild_subtabs

        def tracking_rebuild(*a, **kw):
            calls.append((a, kw))
            return orig(*a, **kw)

        window._run_results_tab.rebuild_subtabs = tracking_rebuild

        # Trigger include-changed (row 0, toggle off)
        window._on_data_tab_include_changed(0, str(entries[0]["id"]), False)
        qt_app.processEvents()

        assert len(calls) == 0, "rebuild_subtabs must not fire while a fit is running"
        assert window._results_rebuild_pending is True
    finally:
        window._worker = None
        window.close()
        qt_app.processEvents()


def test_pending_rebuild_fires_on_fit_complete(qt_app):
    """A pending rebuild must fire when the fit completes, before pushing results."""
    entries = _make_dataset_entries(2)
    window = _make_window(entries)
    try:
        window.show()
        qt_app.processEvents()

        # Simulate a running fit
        window._worker = _FakeWorker()
        window._on_data_tab_include_changed(0, str(entries[0]["id"]), False)
        qt_app.processEvents()
        assert window._results_rebuild_pending is True

        # Record rebuild_subtabs calls
        calls: list[tuple] = []
        orig = window._run_results_tab.rebuild_subtabs

        def tracking_rebuild(*a, **kw):
            calls.append((a, kw))
            return orig(*a, **kw)

        window._run_results_tab.rebuild_subtabs = tracking_rebuild

        # Simulate fit completion (hide window to suppress QMessageBox)
        window._worker = None
        window.hide()
        result = GlobalFitResult(
            shared_params={"k1": 1.0},
            dataset_params={},
            uncertainties=None,
            global_chi_squared=0.01,
            global_r_squared=0.99,
            nfev=100,
            dataset_info=[
                DatasetFitInfo(
                    dataset_id=str(e["id"]),
                    chi_squared=0.01,
                    r_squared=0.99,
                    rmse=0.1,
                    mae=0.05,
                    n_points=5,
                    residuals=np.zeros(5),
                    weight=1.0,
                )
                for e in entries
            ],
            model_series={
                str(e["id"]): {list(e["species_data"].keys())[0]: np.zeros(5)} for e in entries
            },
            objective_residuals=np.zeros(10),
            message="ok",
            completion=GlobalFitCompletion(
                status="ok",
                optimizer_converged=True,
                nonfinite_metrics=False,
            ),
        )
        window._handle_global_fit_complete({"result": result})
        qt_app.processEvents()

        assert len(calls) >= 1, "rebuild_subtabs must fire on fit completion when pending"
        assert window._results_rebuild_pending is False
    finally:
        window._worker = None
        window.close()
        qt_app.processEvents()


def test_pending_rebuild_uses_current_dataset_state(qt_app):
    """The deferred rebuild must use the post-toggle dataset entries, not stale state."""
    entries = _make_dataset_entries(3)
    window = _make_window(entries)
    try:
        window.show()
        qt_app.processEvents()

        # Simulate a running fit and toggle dataset 0 off
        window._worker = _FakeWorker()
        window._on_data_tab_include_changed(0, str(entries[0]["id"]), False)
        qt_app.processEvents()

        # Capture the entries passed to rebuild_subtabs at completion
        captured_entries: list = []
        orig = window._run_results_tab.rebuild_subtabs

        def capture_rebuild(ds_entries, *a, **kw):
            captured_entries.append(list(ds_entries))
            return orig(ds_entries, *a, **kw)

        window._run_results_tab.rebuild_subtabs = capture_rebuild

        # Complete the fit (hide window to suppress QMessageBox)
        window._worker = None
        window.hide()
        result = GlobalFitResult(
            shared_params={"k1": 1.0},
            dataset_params={},
            uncertainties=None,
            global_chi_squared=0.01,
            global_r_squared=0.99,
            nfev=100,
            dataset_info=[
                DatasetFitInfo(
                    dataset_id=str(e["id"]),
                    chi_squared=0.01,
                    r_squared=0.99,
                    rmse=0.1,
                    mae=0.05,
                    n_points=5,
                    residuals=np.zeros(5),
                    weight=1.0,
                )
                for e in entries
            ],
            model_series={
                str(e["id"]): {list(e["species_data"].keys())[0]: np.zeros(5)} for e in entries
            },
            objective_residuals=np.zeros(15),
            message="ok",
            completion=GlobalFitCompletion(
                status="ok",
                optimizer_converged=True,
                nonfinite_metrics=False,
            ),
        )
        window._handle_global_fit_complete({"result": result})
        qt_app.processEvents()

        assert len(captured_entries) >= 1
        # The first dataset should reflect the toggled-off state
        rebuilt_first = captured_entries[0][0]
        assert rebuilt_first["include"] is False, (
            "Deferred rebuild must use the current (post-toggle) dataset state"
        )
    finally:
        window._worker = None
        window.close()
        qt_app.processEvents()
