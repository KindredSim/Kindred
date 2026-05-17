from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
from PySide6 import QtCore, QtWidgets

from tests.batch_context_test_helpers import seed_batch_context
from tests.workflow_helpers import (
    current_preview_time_axis,
    parameter_table_numeric_value,
    select_batch_rows,
    set_batch_current_and_selected_rows,
    set_shown_rows,
    slider_handle_center,
)
from tests.worker_stubs import make_contained_simulation_worker_stub


pytestmark = [pytest.mark.gui, pytest.mark.slow]


def _selected_set_ids(main_window) -> list[str]:
    return [str(set_id) for set_id in main_window._batch_set_ids_for_scope("selected") if str(set_id)]


def _stub_payload(worker) -> dict:
    point_count = 350 if bool(getattr(worker, "_fast_mode", False)) else 100
    t_end = 50.0 if bool(getattr(worker, "_fast_mode", False)) else 1.0
    seed = float((worker._initials or {}).get("A", 1.0) or 1.0)
    t = np.asarray(np.linspace(0.0, t_end, point_count, dtype=float))
    y = np.vstack(
        [
            np.asarray(np.linspace(seed, seed * 0.3, t.size), dtype=float),
            np.asarray(np.linspace(0.0, seed * 0.7, t.size), dtype=float),
        ]
    )
    return {
        "t": t,
        "Y": y,
        "species_names": ["A", "B"],
        "algebra_scalars": {},
        "algebra_errors": [],
        "mechanism": None,
        "mechanism_text": worker._mechanism_text,
        "solver_config": dict(worker._solver_config),
        "provenance": {},
        "fallback_occurred": False,
        "fallback_message": None,
    }


def test_dirty_slider_preview_reselect_run_selected_clears_only_targeted_dirty_state(
    main_window,
    qtbot,
    monkeypatch,
):
    main_window._mechanism_editor._reactions_text.setPlainText("A -> B ; k=1.0")
    main_window._extract_and_populate_variables()
    main_window._mechanism_editor.set_slider_points_value(350)
    main_window._batch_model.set_species(["A"])

    add_btn = main_window.findChild(QtWidgets.QPushButton, "addBatchSetButton")
    assert add_btn is not None
    add_btn.click()
    qtbot.wait(10)

    cache = main_window.simulation_controller.batch_cache
    main_window._preview_session.sync_committed_slider_values({"k1": 1.0})

    select_batch_rows(main_window, [0, 1])
    qtbot.waitUntil(lambda: len(_selected_set_ids(main_window)) == 2, timeout=1000)
    first_set_id, second_set_id = _selected_set_ids(main_window)
    main_window._preview_session.stage_slider_value("k1", 2.0, target_set_ids=[first_set_id])
    main_window._preview_session.stage_slider_value("k1", 3.0, target_set_ids=[second_set_id])

    preview_t = current_preview_time_axis(main_window)
    first_preview_series = np.asarray(np.linspace(9.0, 18.0, preview_t.size, dtype=float))
    second_preview_series = np.asarray(np.linspace(11.0, 22.0, preview_t.size, dtype=float))
    first_preview_key = "workflow-first-preview-key"
    second_preview_key = "workflow-second-preview-key"
    for set_id, preview_key, preview_series in (
        (first_set_id, first_preview_key, first_preview_series),
        (second_set_id, second_preview_key, second_preview_series),
    ):
        mechanism_text = main_window._simulation_batch_owner.mechanism_text_for_workspace_selection(set_id=set_id)
        solver_config, _, preview_token = main_window._simulation_batch_owner.current_workspace_preview_context(
            set_id=set_id,
            mechanism_text=mechanism_text,
        )
        cache.preview_cache[f"{preview_key}::{set_id}"] = {
            "t": preview_t,
            "series": {"A": preview_series},
            "algebra_scalars": {},
            "mechanism_text": mechanism_text,
            "solver_config": dict(solver_config),
            "preview_batch_cache_token": str(preview_token or ""),
        }

    cache.active_preview_cache_key = first_preview_key
    cache.active_preview_scope_set_ids = (first_set_id,)

    select_batch_rows(main_window, [0])
    qtbot.waitUntil(lambda: main_window.active_batch_selection()[0] == first_set_id, timeout=1000)

    plot = main_window._plot_tabs._main_plot
    assert np.allclose(
        np.asarray((getattr(plot, "_series", {}) or {})["A"], dtype=float),
        first_preview_series,
    )
    assert main_window.variable_slider_values()["k1"] == pytest.approx(2.0)
    assert parameter_table_numeric_value(main_window, "k1") == pytest.approx(2.0)

    monkeypatch.setattr(
        "kindred.gui.simulation_worker.ContainedSimulationWorker",
        make_contained_simulation_worker_stub(payload_factory=_stub_payload, emit_progress=(100, "done")),
    )
    main_window.simulation_controller._run_simulation()
    qtbot.waitUntil(lambda: main_window._preview_session.local_mechanism_workspace(first_set_id) == {}, timeout=2000)

    assert main_window._preview_session.local_mechanism_workspace(first_set_id) == {}
    assert main_window._preview_session.local_mechanism_workspace(second_set_id) == {"k1": pytest.approx(3.0)}

    explicit_series = np.asarray(np.linspace(1.0, 0.3, 100, dtype=float))
    assert np.allclose(
        np.asarray((getattr(plot, "_series", {}) or {})["A"], dtype=float),
        explicit_series,
    )

    sel = main_window._batch_table.selectionModel()
    assert sel is not None
    sel.clearSelection()
    QtWidgets.QApplication.processEvents()

    select_batch_rows(main_window, [0])
    qtbot.waitUntil(lambda: main_window.active_batch_selection()[0] == first_set_id, timeout=1000)
    assert np.allclose(
        np.asarray((getattr(plot, "_series", {}) or {})["A"], dtype=float),
        explicit_series,
    )
    assert main_window.variable_slider_values()["k1"] == pytest.approx(1.0)
    assert parameter_table_numeric_value(main_window, "k1") == pytest.approx(1.0)

    cache.active_preview_cache_key = second_preview_key
    cache.active_preview_scope_set_ids = (second_set_id,)
    select_batch_rows(main_window, [1])
    qtbot.waitUntil(lambda: main_window.active_batch_selection()[0] == second_set_id, timeout=1000)

    assert np.allclose(
        np.asarray((getattr(plot, "_series", {}) or {})["A"], dtype=float),
        second_preview_series,
    )
    assert main_window.variable_slider_values()["k1"] == pytest.approx(3.0)
    assert parameter_table_numeric_value(main_window, "k1") == pytest.approx(3.0)


def test_cached_explicit_selection_change_reuses_cached_overlay_without_recompute(
    main_window,
    qtbot,
    monkeypatch,
):
    main_window._mechanism_editor._reactions_text.setPlainText("A -> B ; k=1.0")
    main_window._extract_and_populate_variables()
    main_window._batch_model.set_species(["A"])

    add_btn = main_window.findChild(QtWidgets.QPushButton, "addBatchSetButton")
    assert add_btn is not None
    add_btn.click()
    qtbot.wait(10)

    assert main_window._batch_model.setData(main_window._batch_model.index(0, 1), "1.0")
    assert main_window._batch_model.setData(main_window._batch_model.index(1, 1), "0.5")

    select_batch_rows(main_window, [0, 1])
    set_shown_rows(main_window, [0, 1])
    qtbot.waitUntil(lambda: len(_selected_set_ids(main_window)) == 2, timeout=1000)
    primary_id, secondary_id = _selected_set_ids(main_window)
    cache = main_window.simulation_controller.batch_cache
    cache_key = "workflow-cached-explicit-selection-change"
    primary_expected = np.asarray([1.0, 0.3], dtype=float)
    secondary_expected = np.asarray([0.5, 0.15], dtype=float)
    for set_id, series in (
        (primary_id, primary_expected),
        (secondary_id, secondary_expected),
    ):
        cache.result_cache[f"{cache_key}::{set_id}"] = {
            "t": np.asarray([0.0, 1.0], dtype=float),
            "series": {"A": series},
            "algebra_scalars": {},
        }
    cache.active_cache_key = cache_key
    cache.active_cache_valid_set_ids = (primary_id, secondary_id)
    main_window.set_active_batch_selection(primary_id, str(main_window.batch_set_name_for_id(primary_id) or ""), [primary_id, secondary_id])

    worker_initials: list[dict[str, float]] = []
    monkeypatch.setattr(
        "kindred.gui.simulation_worker.ContainedSimulationWorker",
        make_contained_simulation_worker_stub(
            on_start=lambda worker: worker_initials.append(dict(worker._initials or {})),
            payload_factory=_stub_payload,
            emit_progress=(100, "done"),
        ),
    )

    main_window._refresh_batch_display_from_focus_and_shown()
    qtbot.waitUntil(lambda: main_window.active_batch_selection()[0] == primary_id, timeout=1000)
    plot = main_window._plot_tabs._main_plot
    assert np.allclose(
        np.asarray((getattr(plot, "_series", {}) or {})["A"], dtype=float),
        primary_expected,
    )
    overlays = list(getattr(plot, "_simulation_overlays", []) or [])
    secondary_overlay = next(
        entry
        for entry in overlays
        if str(entry.get("set_id") or "") == secondary_id and str(entry.get("curve_role") or "") != "canonical_ghost"
    )
    assert np.allclose(
        np.asarray((secondary_overlay.get("series") or {})["A"], dtype=float),
        secondary_expected,
    )

    completed_runs = len(worker_initials)
    set_batch_current_and_selected_rows(main_window, current_row=1, selected_rows=[0, 1])
    main_window._refresh_batch_display_from_focus_and_shown()
    qtbot.waitUntil(lambda: main_window.active_batch_selection()[0] == secondary_id, timeout=1000)

    assert len(worker_initials) == completed_runs
    assert np.allclose(
        np.asarray((getattr(plot, "_series", {}) or {})["A"], dtype=float),
        secondary_expected,
    )
    overlays = list(getattr(plot, "_simulation_overlays", []) or [])
    primary_overlay = next(
        entry
        for entry in overlays
        if str(entry.get("set_id") or "") == primary_id and str(entry.get("curve_role") or "") != "canonical_ghost"
    )
    assert np.allclose(
        np.asarray((primary_overlay.get("series") or {})["A"], dtype=float),
        primary_expected,
    )


def test_species_mode_slider_overlay_commit_and_reset_follow_transaction_boundaries(
    main_window,
    qtbot,
    monkeypatch,
):
    main_window._batch_model.set_species(["A", "B"])
    model = main_window._batch_model

    add_btn = main_window.findChild(QtWidgets.QPushButton, "addBatchSetButton")
    assert add_btn is not None
    add_btn.click()
    QtWidgets.QApplication.processEvents()

    assert model.setData(model.index(0, 1), "1.0")
    assert model.setData(model.index(0, 2), "2.0")
    assert model.setData(model.index(1, 1), "0.25")
    assert model.setData(model.index(1, 2), "0.75")

    set_batch_current_and_selected_rows(main_window, current_row=0, selected_rows=[0, 1])
    main_window.set_slider_edit_target_set_ids(
        [
            str(main_window.batch_set_id_for_row(0) or ""),
            str(main_window.batch_set_id_for_row(1) or ""),
        ]
    )

    monkeypatch.setattr(main_window.simulation_controller, "launch_pending_slider_preview_replay", lambda: None)

    qtbot.addWidget(main_window)
    main_window.show()

    slider_a = main_window.findChild(QtWidgets.QSlider, "speciesSlider_A")
    commit_btn = main_window.findChild(QtWidgets.QPushButton, "commitSliderOverridesButton")
    reset_btn = main_window.findChild(QtWidgets.QPushButton, "resetSliderOverridesButton")
    assert slider_a is not None
    assert commit_btn is not None
    assert reset_btn is not None

    press_pos = slider_handle_center(slider_a)
    qtbot.mousePress(slider_a, QtCore.Qt.LeftButton, pos=press_pos)
    slider_a.setValue(5000)
    qtbot.mouseRelease(slider_a, QtCore.Qt.LeftButton, pos=press_pos)
    QtWidgets.QApplication.processEvents()

    staged_row_0 = main_window._preview_session.preview_initials_for_row(0, main_window.batch_initials_for_row(0))
    staged_row_1 = main_window._preview_session.preview_initials_for_row(1, main_window.batch_initials_for_row(1))
    assert float(main_window._batch_store.get_value(0, "A")) == pytest.approx(1.0, rel=1e-6, abs=1e-9)
    assert float(main_window._batch_store.get_value(1, "A")) == pytest.approx(0.25, rel=1e-6, abs=1e-9)
    assert float(staged_row_0["A"]) == pytest.approx(2.5, rel=1e-6, abs=1e-9)
    assert float(staged_row_1["A"]) == pytest.approx(2.5, rel=1e-6, abs=1e-9)

    qtbot.mouseClick(commit_btn, QtCore.Qt.LeftButton)
    QtWidgets.QApplication.processEvents()

    assert float(main_window._batch_store.get_value(0, "A")) == pytest.approx(2.5, rel=1e-6, abs=1e-9)
    assert float(main_window._batch_store.get_value(1, "A")) == pytest.approx(2.5, rel=1e-6, abs=1e-9)
    assert commit_btn.isEnabled() is False
    assert reset_btn.isEnabled() is False

    qtbot.mousePress(slider_a, QtCore.Qt.LeftButton, pos=press_pos)
    slider_a.setValue(0)
    qtbot.mouseRelease(slider_a, QtCore.Qt.LeftButton, pos=press_pos)
    QtWidgets.QApplication.processEvents()

    dirty_row_0 = main_window._preview_session.preview_initials_for_row(0, main_window.batch_initials_for_row(0))
    dirty_row_1 = main_window._preview_session.preview_initials_for_row(1, main_window.batch_initials_for_row(1))
    assert float(main_window._batch_store.get_value(0, "A")) == pytest.approx(2.5, rel=1e-6, abs=1e-9)
    assert float(main_window._batch_store.get_value(1, "A")) == pytest.approx(2.5, rel=1e-6, abs=1e-9)
    assert float(dirty_row_0["A"]) != pytest.approx(2.5, rel=1e-6, abs=1e-9)
    assert float(dirty_row_1["A"]) != pytest.approx(2.5, rel=1e-6, abs=1e-9)

    qtbot.mouseClick(reset_btn, QtCore.Qt.LeftButton)
    QtWidgets.QApplication.processEvents()

    reset_row_0 = main_window._preview_session.preview_initials_for_row(0, main_window.batch_initials_for_row(0))
    reset_row_1 = main_window._preview_session.preview_initials_for_row(1, main_window.batch_initials_for_row(1))
    assert float(main_window._batch_store.get_value(0, "A")) == pytest.approx(2.5, rel=1e-6, abs=1e-9)
    assert float(main_window._batch_store.get_value(1, "A")) == pytest.approx(2.5, rel=1e-6, abs=1e-9)
    assert float(reset_row_0["A"]) == pytest.approx(2.5, rel=1e-6, abs=1e-9)
    assert float(reset_row_1["A"]) == pytest.approx(2.5, rel=1e-6, abs=1e-9)


def test_stale_simulation_completion_does_not_publish_cache_or_display(main_window):
    controller = main_window.simulation_controller
    controller._active_run_id = 3
    controller._latest_sim_request_id = 5
    controller._simulation_running = True
    controller._authoritative_runtime_input_global_epoch = 0
    controller._authoritative_runtime_input_epoch = 0
    controller._authoritative_runtime_input_set_epoch_by_set_id = {"id1": 2}
    seed_batch_context(
        controller.batch_context_owner,
        active=True,
        parallel=False,
        fast_mode=False,
        run_id=3,
        request_id=5,
        runtime_input_global_epoch=0,
        runtime_input_set_epoch_by_set_id={"id1": 1},
        cache_key="workflow-stale-completion",
        queue_ids=["id1"],
        queue_names=["set1"],
        rows=[0],
        pos=0,
        total=1,
        primary_set_id="id1",
    )
    callback_identity = controller._capture_simulation_callback_identity(
        run_id=3,
        fast_mode=False,
        request_id=5,
        owner_epoch=None,
        batch_set="set1",
        batch_set_id="id1",
        cache_key="workflow-stale-completion",
        callback_context=controller.batch_context_owner.callback_context_snapshot(),
        simulation_identity={},
    )

    controller._cache_admin.publish_completion_cache_truth = MagicMock()
    controller._cache_admin.publish_completion_cache = MagicMock()
    controller.ui.results.publish_simulation_completion_result = MagicMock()
    controller.ui.results.publish_completion_intervention_annotations = MagicMock()
    controller.ui.provenance.publish_simulation_completion_provenance = MagicMock()

    controller.on_simulation_complete(
        {
            "t": np.asarray([0.0, 1.0], dtype=float),
            "Y": np.asarray([[1.0, 0.5]], dtype=float),
            "species_names": ["A"],
            "algebra_scalars": {},
            "mechanism": None,
            "mechanism_text": "reaction: A -> B; k=1",
            "solver_config": {"solver": "BDF"},
            "fallback_occurred": False,
            "fallback_message": None,
        },
        callback_identity=callback_identity,
    )

    controller._cache_admin.publish_completion_cache_truth.assert_not_called()
    controller._cache_admin.publish_completion_cache.assert_not_called()
    controller.ui.results.publish_simulation_completion_result.assert_not_called()
    controller.ui.results.publish_completion_intervention_annotations.assert_not_called()
    controller.ui.provenance.publish_simulation_completion_provenance.assert_not_called()
    policy_context = controller.batch_context_owner.completion_policy_context()
    assert policy_context is not None
    assert policy_context.active is True
