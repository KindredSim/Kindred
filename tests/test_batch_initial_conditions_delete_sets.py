from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6 import QtCore, QtWidgets

from kindred.core.batch_containment import BatchLaneOutcome
from tests.worker_stubs import make_contained_simulation_worker_stub


def _select_rows(main_window, rows: list[int]) -> None:
    table = main_window._batch_table
    assert table is not None
    sel = table.selectionModel()
    assert sel is not None
    sel.clearSelection()
    table.setCurrentIndex(main_window._batch_model.index(int(rows[0]), 0))
    for row in rows:
        idx = main_window._batch_model.index(int(row), 0)
        sel.select(idx, QtCore.QItemSelectionModel.Select | QtCore.QItemSelectionModel.Rows)


def _run_all_and_wait(main_window, qt_app, timeout_s: float = 5.0) -> None:
    # "Run All" was intentionally removed; emulate it by selecting all rows and
    # using the existing "Run Selected" entry point.
    _select_rows(main_window, list(range(int(main_window._batch_store.row_count()))))
    main_window.simulation_controller.run_simulation()
    deadline = time.time() + float(timeout_s)
    while time.time() < deadline:
        qt_app.processEvents()
        if (
            not bool(getattr(main_window, "_simulation_running", False))
            and not main_window.simulation_controller.parallel_batch.has_active_requests()
        ):
            qt_app.processEvents()
            return
        QtCore.QThread.msleep(5)
    raise AssertionError("Timed out waiting for Run All completion")


@pytest.mark.gui
def test_active_batch_cache_key_reads_batch_cache_state(main_window):
    main_window.simulation_controller.batch_cache.active_cache_key = "wrapper-regression-cache"

    assert main_window.active_batch_cache_key() == "wrapper-regression-cache"


@pytest.mark.gui
def test_delete_selected_batch_sets_updates_store_mapping_cache_and_selection(main_window, qt_app, monkeypatch):
    main_window._batch_model.set_species(["A", "B"])

    # Build three sets.
    add_btn = main_window.findChild(QtWidgets.QPushButton, "addBatchSetButton")
    assert add_btn is not None
    add_btn.click()
    add_btn.click()
    qt_app.processEvents()

    names_before = list(main_window._batch_store.set_names())
    assert names_before[:3] == ["set1", "set2", "set3"]

    # Seed deterministic cache entries for all sets.
    cache_key = "delete-regression-cache"
    cache_payload: dict[str, dict[str, object]] = {}
    for row, name in enumerate(names_before[:3]):
        t = np.array([0.0, 1.0], dtype=float)
        series = {"A": np.array([1.0 + row, 0.5 + row], dtype=float), "B": np.array([0.0, 0.5], dtype=float)}
        set_key = name
        if hasattr(main_window._batch_store, "set_id_for_row"):
            set_key = str(main_window._batch_store.set_id_for_row(int(row)))
        cache_payload[str(set_key)] = {"t": t, "series": series, "algebra_scalars": {}}

    for set_key, payload in cache_payload.items():
        main_window.simulation_controller.batch_cache.result_cache[f"{cache_key}::{set_key}"] = payload
    main_window.simulation_controller.batch_cache.active_cache_key = cache_key

    # Attach dataset mappings to the first two sets.
    s1 = main_window._dataset_manager.get_fit_settings("dataset_alpha")
    s1.batch_set = names_before[0]
    s2 = main_window._dataset_manager.get_fit_settings("dataset_beta")
    s2.batch_set = names_before[1]

    if hasattr(s1, "batch_set_id") and hasattr(main_window._batch_store, "set_id_for_row"):
        s1.batch_set_id = str(main_window._batch_store.set_id_for_row(0))
        s2.batch_set_id = str(main_window._batch_store.set_id_for_row(1))

    # Prime plot overlays from cached selection.
    ok = main_window._simulation_batch_owner.display_cached_batch_selection(
        cache_key=cache_key,
        selected_sets=names_before[:3],
        prefer_set=names_before[0],
    )
    assert ok is True

    # Multi-delete should always confirm.
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        lambda *args, **kwargs: QtWidgets.QMessageBox.StandardButton.Yes,
    )

    _select_rows(main_window, [0, 1])

    deleted_keys = set()
    for row in [0, 1]:
        if hasattr(main_window._batch_store, "set_id_for_row"):
            deleted_keys.add(str(main_window._batch_store.set_id_for_row(int(row))))
        else:
            deleted_keys.add(str(names_before[row]))

    # Direct handler invocation (headless regression path).
    main_window._delete_selected_batch_sets()
    qt_app.processEvents()

    assert int(main_window._batch_store.row_count()) == 1

    remaining_keys = set()
    for row in range(int(main_window._batch_store.row_count())):
        if hasattr(main_window._batch_store, "set_id_for_row"):
            remaining_keys.add(str(main_window._batch_store.set_id_for_row(int(row))))
        else:
            remaining_keys.add(str(main_window._batch_store.set_names()[row]))
    assert remaining_keys.isdisjoint(deleted_keys)

    # Cache entries for deleted sets should be removed.
    cached_after = set()
    prefix = f"{cache_key}::"
    for k in (main_window.simulation_controller.batch_cache.result_cache or {}).keys():
        token = str(k or "")
        if token.startswith(prefix):
            cached_after.add(token[len(prefix) :])
    assert cached_after.isdisjoint(deleted_keys)

    # Dataset mappings that pointed to deleted sets must be unmapped.
    assert getattr(s1, "batch_set", None) in (None, "")
    assert getattr(s2, "batch_set", None) in (None, "")
    if hasattr(s1, "batch_set_id"):
        assert getattr(s1, "batch_set_id", None) in (None, "")
        assert getattr(s2, "batch_set_id", None) in (None, "")

    table = main_window._batch_table
    assert table is not None


@pytest.mark.gui
def test_delete_selected_batch_sets_dirty_overlay_cancel_aborts_before_delete(main_window, qt_app, monkeypatch):
    main_window._batch_model.set_species(["A", "B"])

    add_btn = main_window.findChild(QtWidgets.QPushButton, "addBatchSetButton")
    assert add_btn is not None
    add_btn.click()
    qt_app.processEvents()

    main_window._batch_store.set_value(0, "A", "1.0")
    main_window._batch_store.set_value(1, "A", "2.0")
    main_window._preview_session.stage_concentration_value_for_rows([0], species="A", value=2.5)

    prompt_calls: list[str] = []
    confirm_calls: list[str] = []

    monkeypatch.setattr(
        main_window,
        "_prompt_slider_transaction_invalidation",
        lambda *_args, **_kwargs: prompt_calls.append("prompt") or "cancel",
        raising=False,
    )
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        lambda *args, **kwargs: confirm_calls.append("confirm") or QtWidgets.QMessageBox.StandardButton.Yes,
    )

    _select_rows(main_window, [0, 1])
    main_window._delete_selected_batch_sets()
    qt_app.processEvents()

    assert confirm_calls == ["confirm"]
    assert prompt_calls == ["prompt"]
    assert int(main_window._batch_store.row_count()) == 2
    assert main_window._preview_session.has_dirty_transaction() is True


@pytest.mark.gui
def test_delete_selected_batch_sets_dirty_mechanism_cancel_aborts_before_delete(main_window, qt_app, monkeypatch):
    main_window._mechanism_editor._reactions_text.setPlainText("reaction: A -> B; k=1.0")
    main_window._extract_and_populate_variables()
    main_window._preview_session.stage_slider_value("k1", 2.0)

    add_btn = main_window.findChild(QtWidgets.QPushButton, "addBatchSetButton")
    assert add_btn is not None
    add_btn.click()
    qt_app.processEvents()

    prompt_calls: list[str] = []
    confirm_calls: list[str] = []

    monkeypatch.setattr(
        main_window,
        "_prompt_slider_transaction_invalidation",
        lambda *_args, **_kwargs: prompt_calls.append("prompt") or "cancel",
        raising=False,
    )
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        lambda *args, **kwargs: confirm_calls.append("confirm") or QtWidgets.QMessageBox.StandardButton.Yes,
    )

    _select_rows(main_window, [0, 1])
    main_window._delete_selected_batch_sets()
    qt_app.processEvents()

    assert confirm_calls == ["confirm"]
    assert prompt_calls == ["prompt"]
    assert int(main_window._batch_store.row_count()) == 2
    assert main_window._preview_session.has_dirty_transaction() is True


@pytest.mark.gui
def test_delete_selected_batch_sets_dirty_overlay_on_other_rows_cancel_aborts_before_delete(
    main_window, qt_app, monkeypatch
):
    main_window._batch_model.set_species(["A", "B"])

    add_btn = main_window.findChild(QtWidgets.QPushButton, "addBatchSetButton")
    assert add_btn is not None
    add_btn.click()
    add_btn.click()
    qt_app.processEvents()

    main_window._batch_store.set_value(0, "A", "1.0")
    main_window._batch_store.set_value(1, "A", "2.0")
    main_window._batch_store.set_value(2, "A", "3.0")
    main_window._preview_session.stage_concentration_value_for_rows([2], species="A", value=7.5)

    prompt_calls: list[str] = []
    confirm_calls: list[str] = []

    monkeypatch.setattr(
        main_window,
        "_prompt_slider_transaction_invalidation",
        lambda *_args, **_kwargs: prompt_calls.append("prompt") or "cancel",
        raising=False,
    )
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        lambda *args, **kwargs: confirm_calls.append("confirm") or QtWidgets.QMessageBox.StandardButton.Yes,
    )

    _select_rows(main_window, [0, 1])
    main_window._delete_selected_batch_sets()
    qt_app.processEvents()

    assert confirm_calls == ["confirm"]
    assert prompt_calls == ["prompt"]
    assert int(main_window._batch_store.row_count()) == 3
    assert main_window._preview_session.has_dirty_transaction() is True


@pytest.mark.gui
def test_delete_selected_batch_sets_delete_confirm_cancel_keeps_dirty_transaction(main_window, qt_app, monkeypatch):
    main_window._batch_model.set_species(["A", "B"])

    add_btn = main_window.findChild(QtWidgets.QPushButton, "addBatchSetButton")
    assert add_btn is not None
    add_btn.click()
    qt_app.processEvents()

    main_window._batch_store.set_value(0, "A", "1.0")
    main_window._batch_store.set_value(1, "A", "2.0")
    main_window._preview_session.stage_concentration_value_for_rows([0], species="A", value=2.5)

    monkeypatch.setattr(
        main_window,
        "_prompt_slider_transaction_invalidation",
        lambda *_args, **_kwargs: "discard",
        raising=False,
    )
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        lambda *args, **kwargs: QtWidgets.QMessageBox.StandardButton.Cancel,
    )

    _select_rows(main_window, [0, 1])
    main_window._delete_selected_batch_sets()
    qt_app.processEvents()

    assert main_window._preview_session.has_dirty_transaction() is True
    assert int(main_window._batch_store.row_count()) == 2


@pytest.mark.gui
def test_run_all_twice_does_not_accumulate_worker_objects_or_plot_curves(main_window, qt_app, monkeypatch):
    if hasattr(main_window, "set_simulation_cache_caps"):
        main_window.set_simulation_cache_caps(result_cap=20, preview_cap=20)

    def _payload(worker) -> dict:
        n = int((worker._solver_config.get("grid") or {}).get("N", 40) or 40)
        t = np.linspace(0.0, 1.0, n)
        a0 = float(worker._initials.get("A", 1.0))
        y = np.vstack(
            [
                np.linspace(a0, a0 * 0.2, n),
                np.linspace(0.0, a0 * 0.8, n),
            ]
        )
        return {
            "t": t,
            "Y": y,
            "species_names": ["A", "B"],
            "algebra_scalars": {},
            "mechanism": None,
            "mechanism_text": worker._mechanism_text,
            "solver_config": dict(worker._solver_config),
            "fallback_occurred": False,
            "fallback_message": None,
        }

    _StubWorker = make_contained_simulation_worker_stub(payload_factory=_payload)
    monkeypatch.setattr("kindred.gui.simulation_worker.ContainedSimulationWorker", _StubWorker)

    class _ReadyLanePool:
        def run(self, task, *, run_id: int, request_id: int, set_id: str, active_timeout_s: float):
            _ = active_timeout_s
            task_map = dict(task or {})
            worker = SimpleNamespace(
                _solver_config=dict(task_map.get("solver_config") or {}),
                _initials=dict(task_map.get("initials") or {}),
                _mechanism_text=str(task_map.get("mechanism_text") or ""),
            )
            payload = dict(_payload(worker))
            payload.update(
                {
                    "run_id": int(run_id),
                    "request_id": int(request_id),
                    "set_id": str(set_id),
                    "set_name": str(task_map.get("set_name") or set_id),
                }
            )
            return BatchLaneOutcome(
                lane_id=f"test-lane-{set_id}",
                run_id=int(run_id),
                request_id=int(request_id),
                set_id=str(set_id),
                owner_epoch=int(task_map.get("owner_epoch") or run_id),
                success=True,
                payload=payload,
            )

        def close(self, *, kill: bool = False) -> None:
            _ = kill

    main_window._mechanism_editor._reactions_text.setPlainText("reaction: A -> B ; k=0.5")
    main_window._batch_model.set_species(["A", "B"])

    add_btn = main_window.findChild(QtWidgets.QPushButton, "addBatchSetButton")
    assert add_btn is not None
    add_btn.click()
    add_btn.click()

    for idx, name in enumerate(main_window._batch_store.set_names()[:3]):
        row = main_window._batch_store.row_for_set(name)
        assert row is not None
        main_window._batch_store.set_value(int(row), "A", f"{1.0 + idx:.6g}")
        main_window._batch_store.set_value(int(row), "B", "0.0")

    main_window.simulation_controller.parallel_batch.max_parallel_workers = int(
        main_window._batch_store.row_count()
    )
    _select_rows(main_window, [0, 1, 2])
    main_window.simulation_controller.parallel_batch.lane_pool_factory = (
        lambda _max_lanes, _limit_blas_threads: _ReadyLanePool()
    )

    main_window.simulation_controller.parallel_batch.ensure_lane_pool(
        max_lanes=int(main_window._batch_store.row_count())
    )
    main_window._simulation_run_ui_owner.set_runtime_backed_run_controls_ready(True)
    _run_all_and_wait(main_window, qt_app)
    key1 = str(main_window.simulation_controller.batch_cache.active_cache_key or "")
    prefix1 = f"{key1}::"
    cache_entries_1 = sum(
        1
        for k in (main_window.simulation_controller.batch_cache.result_cache or {}).keys()
        if str(k or "").startswith(prefix1)
    )
    curve_count_1 = len(getattr(main_window._plot_tabs._main_plot, "_plot_items", {}))
    workers_1 = len(main_window.findChildren(_StubWorker))
    marker_count_1 = main_window._mechanism_editor._reactions_text.toPlainText().count("Computational Mode")

    main_window.simulation_controller.parallel_batch.ensure_lane_pool(
        max_lanes=int(main_window._batch_store.row_count())
    )
    main_window._simulation_run_ui_owner.set_runtime_backed_run_controls_ready(True)
    _run_all_and_wait(main_window, qt_app)
    key2 = str(main_window.simulation_controller.batch_cache.active_cache_key or "")
    prefix2 = f"{key2}::"
    cache_entries_2 = sum(
        1
        for k in (main_window.simulation_controller.batch_cache.result_cache or {}).keys()
        if str(k or "").startswith(prefix2)
    )
    curve_count_2 = len(getattr(main_window._plot_tabs._main_plot, "_plot_items", {}))
    workers_2 = len(main_window.findChildren(_StubWorker))
    marker_count_2 = main_window._mechanism_editor._reactions_text.toPlainText().count("Computational Mode")

    assert key1 == key2
    assert cache_entries_1 == 3
    assert cache_entries_2 == 3
    assert curve_count_1 == curve_count_2
    assert workers_2 == workers_1
    assert marker_count_2 == marker_count_1
