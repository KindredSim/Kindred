from __future__ import annotations

import time
from dataclasses import dataclass, field
from queue import Queue

import numpy as np
import pytest
from PySide6 import QtCore

from kindred.core.batch_simulation_cache import BatchSimulationCache
from kindred.core.batch_containment import BatchLaneOutcome
from kindred.core.batch_parallel import run_batch_simulation_task
from kindred.gui.controllers.simulation_cache_admin import SimulationCacheAdmin
from kindred.gui.controllers.simulation_callback_identity import SimulationCallbackIdentity
from kindred.gui.controllers.simulation_run_state import PreviewOwnershipState
from tests.batch_context_test_helpers import seed_batch_context

def _fake_sim_result(*, marker: float = 1.0) -> dict:
    t = np.asarray([0.0, 1.0], dtype=float)
    Y = np.asarray([[marker, marker * 2.0]], dtype=float)
    return {
        "t": t,
        "Y": Y,
        "species_names": ["A"],
        "algebra_scalars": {},
        "mechanism": None,
        "mechanism_text": "reaction: A -> B; k1=1.0",
        "solver_config": {"solver": "BDF", "rtol": 1e-6, "atol": 1e-12, "grid": {"N": 10}, "temperature_K": 298.15},
        "fallback_occurred": False,
        "fallback_message": None,
    }


def _callback_identity(
    controller,
    *,
    run_id=0,
    fast_mode=False,
    request_id=0,
    owner_epoch=None,
    batch_set=None,
    batch_set_id=None,
    cache_key="",
) -> SimulationCallbackIdentity:
    return SimulationCallbackIdentity.capture(
        run_id=run_id,
        fast_mode=fast_mode,
        request_id=request_id,
        owner_epoch=owner_epoch,
        batch_set=batch_set,
        batch_set_id=batch_set_id,
        cache_key=cache_key,
        callback_context=controller.batch_context_owner.callback_context_snapshot(),
        simulation_identity={},
        preview_batch_cache_token="",
    )


def test_simulation_cache_admin_publishes_completion_cache_entry_and_preview_identity() -> None:
    cache = BatchSimulationCache()
    admin = SimulationCacheAdmin(
        cache=cache,
        settings_set_value=lambda *_args, **_kwargs: None,
        settings_sync=lambda: None,
        record_nonfatal_exception=lambda *_args, **_kwargs: None,
    )
    t = np.asarray([0.0, 1.0], dtype=float)
    series = {"A": np.asarray([1.0, 0.5], dtype=float)}

    result = admin.publish_completion_cache(
        cache_key="preview-key",
        cache_token="preview-token",
        set_id="id1",
        is_preview=True,
        t=t,
        series=series,
        algebra_scalars={"S": 1.0},
        mechanism=None,
        mechanism_text="reaction: A -> B; k=1",
        simulation_identity={"schema_id": "schema"},
        solver_config={"solver": "BDF"},
        preview_batch_cache_token="preview-token",
        fallback_occurred=False,
        fallback_message=None,
        solver_provenance={"interventions": []},
        warnings=[{"kind": "preparation_warning", "message": "symbolic disabled"}],
        preview_scope_set_ids=("id1",),
    )

    assert result.cache_token == "preview-token"
    assert cache.active_preview_cache_key == "preview-key"
    assert cache.active_preview_scope_set_ids == ("id1",)
    payload = cache.entry_for_set(cache_key="preview-token", set_id="id1", is_preview=True).entry
    assert isinstance(payload, dict)
    np.testing.assert_allclose(np.asarray(payload["t"]), t)
    assert payload["warnings"] == [{"kind": "preparation_warning", "message": "symbolic disabled"}]

def _select_rows(main_window, rows: list[int]) -> None:
    from PySide6 import QtCore

    table = main_window._batch_table
    assert table is not None
    sel = table.selectionModel()
    assert sel is not None
    sel.clearSelection()
    table.setCurrentIndex(main_window._batch_model.index(int(rows[0]), 0))
    for row in rows:
        idx = main_window._batch_model.index(int(row), 0)
        sel.select(idx, QtCore.QItemSelectionModel.Select | QtCore.QItemSelectionModel.Rows)
    main_window._refresh_batch_display_from_focus_and_shown()

def _set_shown_rows(main_window, rows: list[int]) -> None:
    from PySide6 import QtCore

    model = main_window._batch_model
    shown_rows = {int(row) for row in rows}
    for row in range(model.rowCount()):
        model.set_row_shown(row, row in shown_rows)
        expected = QtCore.Qt.Checked if row in shown_rows else QtCore.Qt.Unchecked
        assert model.data(model.index(row, model.show_column()), QtCore.Qt.CheckStateRole) == expected
    main_window._refresh_batch_display_from_focus_and_shown()

def _set_edit_target_rows(main_window, rows: list[int]) -> None:
    set_ids = [str(main_window._batch_set_id_for_row(int(row)) or "") for row in rows]
    main_window.set_slider_edit_target_set_ids([set_id for set_id in set_ids if set_id])

def _assert_no_cache_warning(main_window) -> None:
    assert main_window._status_label.text() not in (
        "Result not cached (evicted). Press Run to compute.",
        "Cached result invalid. Press Run to compute.",
    )

def _parameter_table_numeric_value(main_window, name: str) -> float:
    table = main_window.main_plot().parameter_table()
    for row in range(table.rowCount()):
        item = table.item(row, 0)
        if item is None or item.text() != str(name):
            continue
        value_item = table.item(row, 1)
        assert value_item is not None
        return float(value_item.text())
    raise AssertionError(f"Missing parameter-table row for {name!r}")

def _assert_selection_plot_cleared(main_window) -> None:
    plot = main_window._plot_tabs._main_plot
    assert getattr(plot, "_t", None) is None
    assert dict(getattr(plot, "_series", {}) or {}) == {}

def _assert_main_plot_matches(main_window, t, series: dict[str, np.ndarray]) -> None:
    plot = main_window._plot_tabs._main_plot
    np.testing.assert_allclose(getattr(plot, "_t", None), np.asarray(t, dtype=float))
    actual = dict(getattr(plot, "_series", {}) or {})
    assert set(actual) == set(series)
    for name, expected in series.items():
        np.testing.assert_allclose(actual[name], np.asarray(expected, dtype=float))

def _current_preview_solver_config(main_window) -> dict:
    from kindred.core.simulator.solvers import DEFAULT_SOLVER_NAME, normalize_solver_name

    solver_label = str(main_window._simulation_solver_owner.initial_solver_name() or DEFAULT_SOLVER_NAME).strip()
    solver_label = solver_label or DEFAULT_SOLVER_NAME
    solver_method, solver_warning = normalize_solver_name(solver_label)
    n_points = int(main_window._simulation_solver_owner.num_points_spinbox_value())
    points_override = main_window._simulation_mechanism_owner.mechanism_slider_points_value()
    if points_override is not None:
        n_points = max(50, int(points_override))
    else:
        n_points = max(50, n_points)
    solver_override = main_window._simulation_mechanism_owner.mechanism_slider_solver_value()
    if solver_override is not None:
        solver_label = str(solver_override).strip() or solver_label
        solver_method, solver_warning = normalize_solver_name(solver_label)
    last_change_name = main_window._preview_session.last_slider_change_name()
    preview_mode = bool(
        main_window._preview_session.slider_drag_active()
        and isinstance(last_change_name, str)
        and last_change_name.startswith("Keq")
        and last_change_name[4:].isdigit()
    )
    if preview_mode:
        n_points = min(int(n_points), 120)
    return {
        "solver": str(solver_method),
        "solver_label": str(solver_label),
        "solver_warning": str(solver_warning) if solver_warning else None,
        "rtol": main_window._simulation_solver_owner.initial_rtol() or 1e-6,
        "atol": main_window._simulation_solver_owner.initial_atol() or 1e-12,
        "grid": {"N": int(n_points)},
        "temperature_K": float(main_window._simulation_solver_owner.temperature_spinbox_value()),
        "use_sparse_jacobian": bool(main_window._simulation_solver_owner.use_sparse_jacobian()),
        "wegscheider_cyclicity_enabled": bool(main_window._simulation_solver_owner.wegscheider_cyclicity_enabled()),
    }

def _current_preview_time_axis(main_window) -> np.ndarray:
    points = int((_current_preview_solver_config(main_window).get("grid") or {}).get("N") or 0)
    return np.linspace(0.0, float(main_window.parse_sim_time_seconds()), max(2, points), dtype=float)

@dataclass
class _FakeLanePoolSubmission:
    fn: object
    args: tuple[object, ...]
    kwargs: dict[str, object]
    _result_queue: Queue[object] = field(default_factory=Queue)
    completed: bool = False

    def complete(self, payload: object) -> None:
        if self.completed:
            return
        self.completed = True
        self._result_queue.put(payload)

    def cancel(self) -> None:
        self.complete({"success": False, "error": {"kind": "cancelled"}})

    def wait_result(self, *, timeout_s: float) -> object:
        return self._result_queue.get(timeout=float(timeout_s))

class _FakeLanePool:
    def __init__(self) -> None:
        self.submissions: list[_FakeLanePoolSubmission] = []
        self.ready_lane_count = 999

    def warm_lanes(self, max_lanes: int, *, wait: bool = True) -> None:
        _ = wait
        self.ready_lane_count = max(1, int(max_lanes))

    def run(self, task, *, run_id: int, request_id: int, set_id: str, active_timeout_s: float):
        _ = run_id, request_id, set_id, active_timeout_s
        sub = _FakeLanePoolSubmission(fn=run_batch_simulation_task, args=(dict(task),), kwargs={})
        self.submissions.append(sub)
        payload = sub.wait_result(timeout_s=5.0)
        return BatchLaneOutcome(
            lane_id="fake-lane",
            run_id=int(run_id),
            request_id=int(request_id),
            set_id=str(set_id),
            owner_epoch=int(payload.get("owner_epoch", 1)) if isinstance(payload, dict) else 1,
            success=not (isinstance(payload, dict) and payload.get("success") is False),
            payload=payload if isinstance(payload, dict) else {"payload": payload},
        )

    def submit(self, fn, *args, **kwargs):
        sub = _FakeLanePoolSubmission(fn=fn, args=args, kwargs=dict(kwargs))
        self.submissions.append(sub)
        return sub

    def _close_requests(self, *, kill: bool = False):
        for sub in self.submissions:
            if not sub.completed:
                if bool(kill):
                    sub.cancel()
                else:
                    task = dict(sub.args[0] if sub.args else {})
                    sid = str(task.get("set_id") or "")
                    sub.complete({"success": True, "run_id": int(task.get("run_id") or 0), "set_id": sid})
        return None

    def close(self, *, kill: bool = False):
        self._close_requests(kill=bool(kill))

def _simulation_submissions(lane_pool: _FakeLanePool) -> list[_FakeLanePoolSubmission]:
    return [sub for sub in lane_pool.submissions if sub.fn is run_batch_simulation_task]


def _wait_for_submission_count(lane_pool: _FakeLanePool, expected: int, timeout_s: float = 1.0) -> None:
    deadline = time.monotonic() + float(timeout_s)
    while time.monotonic() < deadline:
        QtCore.QCoreApplication.processEvents()
        if len(_simulation_submissions(lane_pool)) >= int(expected):
            return
        time.sleep(0.005)


def _join_batch_request(main_window, task: dict[str, object]) -> None:
    _ = task
    main_window.simulation_controller.parallel_batch.join_active_requests(timeout_s=1.0)


def _clear_parallel_runtime_pool(main_window) -> None:
    controller = main_window.simulation_controller
    if controller.parallel_batch.has_lane_pool():
        controller.shutdown_batch_lane_pool(force_terminate=True)
    assert not controller.parallel_batch.has_lane_pool()

def _current_preview_identity_payload(
    main_window,
    *,
    set_id: str,
    preview_batch_cache_token: str | None = None,
) -> dict:
    identity = main_window._simulation_batch_owner.current_workspace_preview_identity(set_id=str(set_id))
    payload = identity.to_payload()
    if preview_batch_cache_token is not None:
        assert str(payload.get("preview_batch_cache_token") or "") == str(preview_batch_cache_token or "")
    return payload

def _workspace_preview_payload(
    main_window,
    *,
    set_id: str,
    series: dict[str, np.ndarray],
    mechanism_text: str | None = None,
    solver_config: dict | None = None,
    preview_batch_cache_token: str | None = None,
    simulation_identity: dict | None = None,
) -> dict:
    preview_t = _current_preview_time_axis(main_window)
    payload = {
        "t": preview_t,
        "series": {str(name): np.asarray(values, dtype=float) for name, values in dict(series).items()},
        "algebra_scalars": {},
        "mechanism_text": (
            str(mechanism_text)
            if mechanism_text is not None
            else main_window._simulation_batch_owner.mechanism_text_for_workspace_selection(set_id=str(set_id))
        ),
        "solver_config": dict(solver_config or _current_preview_solver_config(main_window)),
        "preview_batch_cache_token": str(preview_batch_cache_token or ""),
    }
    if simulation_identity is not None:
        payload["simulation_identity"] = dict(simulation_identity)
    return payload

@pytest.mark.gui
def test_preview_results_go_to_preview_cache_and_are_bounded(main_window, qt_app):
    # Configure tiny caps so eviction is easy to observe.
    assert hasattr(main_window, "set_simulation_cache_caps")
    main_window.set_simulation_cache_caps(result_cap=10, preview_cap=1)

    main_window.simulation_controller.run_state.latest_sim_request_id = 1
    main_window.simulation_controller.run_state.active_run_id = 1
    main_window.simulation_controller.run_state.preview_ownership = PreviewOwnershipState(
        request_id=1,
        epoch=1,
        target_set_ids=("set1",),
    )

    main_window.simulation_controller.on_simulation_complete(
        _fake_sim_result(marker=1.0),
        callback_identity=_callback_identity(
            main_window.simulation_controller,
            run_id=1,
            fast_mode=True,
            request_id=1,
            batch_set="set1",
            batch_set_id="set1",
            cache_key="preview-k1",
        ),
    )
    qt_app.processEvents()

    main_window.simulation_controller.run_state.latest_sim_request_id = 2
    main_window.simulation_controller.run_state.active_run_id = 2
    main_window.simulation_controller.run_state.preview_ownership = PreviewOwnershipState(
        request_id=2,
        epoch=2,
        target_set_ids=("set1",),
    )
    main_window.simulation_controller.on_simulation_complete(
        _fake_sim_result(marker=2.0),
        callback_identity=_callback_identity(
            main_window.simulation_controller,
            run_id=2,
            fast_mode=True,
            request_id=2,
            batch_set="set1",
            batch_set_id="set1",
            cache_key="preview-k2",
        ),
    )
    qt_app.processEvents()

    preview = main_window.simulation_controller.batch_cache.preview_cache
    results = main_window.simulation_controller.batch_cache.result_cache

    assert "preview-k2::set1" in preview
    assert "preview-k1::set1" not in preview
    assert "preview-k1::set1" not in results
    assert "preview-k2::set1" not in results

@pytest.mark.gui
def test_result_cache_is_bounded_separately_from_preview(main_window, qt_app):
    assert hasattr(main_window, "set_simulation_cache_caps")
    main_window.set_simulation_cache_caps(result_cap=1, preview_cap=10)

    main_window.simulation_controller.run_state.latest_sim_request_id = 10
    main_window.simulation_controller.run_state.active_run_id = 10
    main_window.simulation_controller.on_simulation_complete(
        _fake_sim_result(marker=1.0),
        callback_identity=_callback_identity(
            main_window.simulation_controller,
            run_id=10,
            fast_mode=False,
            request_id=10,
            batch_set="set1",
            batch_set_id="set1",
            cache_key="result-k1",
        ),
    )
    qt_app.processEvents()

    main_window.simulation_controller.run_state.latest_sim_request_id = 11
    main_window.simulation_controller.run_state.active_run_id = 11
    main_window.simulation_controller.on_simulation_complete(
        _fake_sim_result(marker=2.0),
        callback_identity=_callback_identity(
            main_window.simulation_controller,
            run_id=11,
            fast_mode=False,
            request_id=11,
            batch_set="set1",
            batch_set_id="set1",
            cache_key="result-k2",
        ),
    )
    qt_app.processEvents()

    preview = main_window.simulation_controller.batch_cache.preview_cache
    results = main_window.simulation_controller.batch_cache.result_cache

    assert "result-k2::set1" in results
    assert "result-k1::set1" not in results
    assert "result-k1::set1" not in preview
    assert "result-k2::set1" not in preview

@pytest.mark.gui
def test_cache_miss_on_clean_selection_preserves_plot_and_sets_evicted_message(main_window, monkeypatch, qt_app):
    from PySide6 import QtWidgets

    # Ensure there are at least two batch sets and a selection exists.
    main_window._batch_model.set_species(["A"])
    add_btn = main_window.findChild(QtWidgets.QPushButton, "addBatchSetButton")
    assert add_btn is not None
    add_btn.click()
    add_btn.click()
    qt_app.processEvents()
    _select_rows(main_window, [0, 1])

    # Establish a known plot state before the cache-miss selection reconciliation.
    t0 = np.asarray([0.0, 1.0], dtype=float)
    series0 = {"A": np.asarray([1.0, 0.5], dtype=float)}
    main_window.set_data(t0, series0, label="baseline", overlays=[])

    # A selection change must never trigger a run; hard-fail if it does.
    monkeypatch.setattr(
        main_window.simulation_controller,
        "run_simulation_internal",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("selection change triggered run")),
        raising=True,
    )

    # Simulate a cache miss for the currently-active cache key.
    main_window.simulation_controller.batch_cache.active_cache_key = "missing-cache-key"
    main_window._refresh_batch_display_from_focus_and_shown()
    qt_app.processEvents()

    assert main_window._status_label.text() == "Result not cached (evicted). Press Run to compute."
    _assert_main_plot_matches(main_window, t0, series0)

@pytest.mark.gui
def test_lru_evicted_clean_selection_preserves_current_plot_and_reports_cache_miss(main_window, monkeypatch, qt_app):
    from PySide6 import QtWidgets

    main_window._batch_model.set_species(["A"])
    add_btn = main_window.findChild(QtWidgets.QPushButton, "addBatchSetButton")
    assert add_btn is not None
    add_btn.click()
    add_btn.click()
    qt_app.processEvents()

    t0 = np.asarray([0.0, 1.0], dtype=float)
    series0 = {"A": np.asarray([1.0, 0.5], dtype=float)}
    main_window.set_data(t0, series0, label="baseline", overlays=[])

    first_set_id = str(main_window._batch_set_id_for_row(0))
    second_set_id = str(main_window._batch_set_id_for_row(1))
    cache_key = "bounded-cache-key"
    evicting_cache_key = "evicting-cache-key"
    cache = main_window.simulation_controller.batch_cache
    main_window.set_simulation_cache_caps(result_cap=1, preview_cap=10)
    cache.result_cache.put(f"{cache_key}::{first_set_id}", {"t": [0.0, 1.0], "series": {"A": [1.0, 0.9]}})
    cache.result_cache.put(f"{evicting_cache_key}::{second_set_id}", {"t": [0.0, 1.0], "series": {"A": [2.0, 1.8]}})
    assert f"{cache_key}::{first_set_id}" not in cache.result_cache

    monkeypatch.setattr(
        main_window.simulation_controller,
        "run_simulation_internal",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("selection change triggered run")),
        raising=True,
    )

    cache.active_cache_key = cache_key
    _select_rows(main_window, [0])
    qt_app.processEvents()

    assert main_window._status_label.text() == "Result not cached (evicted). Press Run to compute."
    _assert_main_plot_matches(main_window, t0, series0)

@pytest.mark.gui
def test_selection_change_without_active_cache_context_does_not_show_evicted_warning(main_window, monkeypatch, qt_app):
    from PySide6 import QtWidgets

    t0 = np.asarray([0.0, 1.0], dtype=float)
    series0 = {"A": np.asarray([1.0, 0.5], dtype=float)}
    main_window.set_data(t0, series0, label="baseline", overlays=[])
    main_window._status_label.setText("Ready")

    main_window._batch_model.set_species(["A"])
    add_btn = main_window.findChild(QtWidgets.QPushButton, "addBatchSetButton")
    assert add_btn is not None
    add_btn.click()
    add_btn.click()
    qt_app.processEvents()

    _select_rows(main_window, [0, 1])
    qt_app.processEvents()

    monkeypatch.setattr(
        main_window.simulation_controller,
        "run_simulation_internal",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("selection change triggered run")),
        raising=True,
    )

    cache = main_window.simulation_controller.batch_cache
    cache.active_cache_key = None
    cache.active_preview_cache_key = None

    main_window._refresh_batch_display_from_focus_and_shown()
    qt_app.processEvents()

    _assert_no_cache_warning(main_window)
    _assert_selection_plot_cleared(main_window)

@pytest.mark.gui
def test_selection_change_without_active_cache_context_clears_stale_cache_warning(main_window, monkeypatch, qt_app):
    from PySide6 import QtWidgets

    t0 = np.asarray([0.0, 1.0], dtype=float)
    series0 = {"A": np.asarray([1.0, 0.5], dtype=float)}
    main_window.set_data(t0, series0, label="baseline", overlays=[])
    main_window._status_label.setText("Result not cached (evicted). Press Run to compute.")

    main_window._batch_model.set_species(["A"])
    add_btn = main_window.findChild(QtWidgets.QPushButton, "addBatchSetButton")
    assert add_btn is not None
    add_btn.click()
    add_btn.click()
    qt_app.processEvents()

    _select_rows(main_window, [0, 1])
    qt_app.processEvents()

    monkeypatch.setattr(
        main_window.simulation_controller,
        "run_simulation_internal",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("selection change triggered run")),
        raising=True,
    )

    cache = main_window.simulation_controller.batch_cache
    cache.clear_active_selection_state()

    main_window._refresh_batch_display_from_focus_and_shown()
    qt_app.processEvents()

    _assert_no_cache_warning(main_window)
    _assert_selection_plot_cleared(main_window)

@pytest.mark.gui

@pytest.mark.gui

@pytest.mark.gui
def test_selection_change_stale_workspace_preview_surfaces_preview_pending_without_explicit_fallback(
    main_window, monkeypatch, qt_app
):
    from PySide6 import QtWidgets

    main_window._mechanism_editor._reactions_text.setPlainText(
        "reaction: A -> B; k=1.0\ninitial: A=1.0\ninitial: B=0.0"
    )
    main_window._extract_and_populate_variables()
    main_window._batch_model.set_species(["A"])
    add_btn = main_window.findChild(QtWidgets.QPushButton, "addBatchSetButton")
    assert add_btn is not None
    add_btn.click()
    qt_app.processEvents()

    _select_rows(main_window, [0, 1])
    qt_app.processEvents()

    monkeypatch.setattr(
        main_window.simulation_controller,
        "run_simulation_internal",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("selection change triggered run")),
        raising=True,
    )

    selected_ids = main_window._batch_set_ids_for_scope("selected")
    assert len(selected_ids) >= 2
    primary_id = str(selected_ids[0])
    secondary_id = str(selected_ids[1])

    cache = main_window.simulation_controller.batch_cache
    explicit_key = "explicit-stale-preview-selection-cache-key"
    preview_key = "preview-stale-selection-cache-key"
    cache.result_cache[f"{explicit_key}::{primary_id}"] = {
        "t": np.asarray([0.0, 1.0], dtype=float),
        "series": {"A": np.asarray([3.0, 6.0], dtype=float)},
        "algebra_scalars": {},
    }
    cache.result_cache[f"{explicit_key}::{secondary_id}"] = {
        "t": np.asarray([0.0, 1.0], dtype=float),
        "series": {"A": np.asarray([2.5, 5.0], dtype=float)},
        "algebra_scalars": {},
    }
    cache.preview_cache[f"{preview_key}::{primary_id}"] = {
        "t": np.asarray([0.0, 1.0], dtype=float),
        "series": {"A": np.asarray([7.0, 14.0], dtype=float)},
        "algebra_scalars": {},
    }
    cache.preview_cache[f"{preview_key}::{secondary_id}"] = {
        "t": np.asarray([0.0, 1.0], dtype=float),
        "series": {"A": np.asarray([6.0, 12.0], dtype=float)},
        "algebra_scalars": {},
    }
    cache.active_cache_key = explicit_key
    main_window._preview_session.sync_committed_slider_values({"k1": 1.0})
    main_window._preview_session.stage_slider_value("k1", 2.0)
    stale_preview_mechanism_text = main_window._simulation_batch_owner.mechanism_text_for_workspace_selection(set_id=primary_id)
    cache.active_preview_cache_key = preview_key
    main_window._preview_session.stage_slider_value("k1", 3.0)
    cache.preview_cache[f"{preview_key}::{primary_id}"]["mechanism_text"] = stale_preview_mechanism_text
    cache.preview_cache[f"{preview_key}::{secondary_id}"]["mechanism_text"] = stale_preview_mechanism_text

    main_window._refresh_batch_display_from_focus_and_shown()
    qt_app.processEvents()

    assert main_window._status_label.text() == "Preview pending for current selection."
    assert main_window.active_batch_selection() == ("", "")
    _assert_selection_plot_cleared(main_window)
    assert main_window.simulation_controller.batch_cache.last_display_selection == []

@pytest.mark.gui
def test_selection_change_keeps_plot_and_mechanism_controls_on_same_set(main_window, monkeypatch, qt_app):
    from PySide6 import QtWidgets

    main_window._mechanism_editor._reactions_text.setPlainText(
        "reaction: A -> B; k=1.0\ninitial: A=1.0\ninitial: B=0.0"
    )
    main_window._extract_and_populate_variables()
    main_window._batch_model.set_species(["A"])

    add_btn = main_window.findChild(QtWidgets.QPushButton, "addBatchSetButton")
    assert add_btn is not None
    add_btn.click()
    qt_app.processEvents()

    monkeypatch.setattr(
        main_window.simulation_controller,
        "run_simulation_internal",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("selection change triggered run")),
        raising=True,
    )
    monkeypatch.setattr(main_window.simulation_controller, "launch_pending_slider_preview_replay", lambda: None)

    _select_rows(main_window, [0])
    qt_app.processEvents()
    selected_ids = main_window._batch_set_ids_for_scope("selected")
    assert len(selected_ids) == 1
    set0_id = str(selected_ids[0])
    set1_id = str(main_window._batch_set_id_for_row(1) or "")
    assert set1_id and set1_id != set0_id
    main_window.set_slider_edit_target_set_ids([set0_id])

    cache = main_window.simulation_controller.batch_cache
    explicit_key = "explicit-selection-control-sync-cache-key"
    cache.result_cache[f"{explicit_key}::{set0_id}"] = {
        "t": np.asarray([0.0, 1.0], dtype=float),
        "series": {"A": np.asarray([7.0, 14.0], dtype=float)},
        "algebra_scalars": {},
    }
    cache.result_cache[f"{explicit_key}::{set1_id}"] = {
        "t": np.asarray([0.0, 1.0], dtype=float),
        "series": {"A": np.asarray([3.0, 6.0], dtype=float)},
        "algebra_scalars": {},
    }
    cache.active_cache_key = explicit_key

    sliders = main_window._mechanism_editor._variable_sliders
    sliders.update_variable("k1", 2.0)
    main_window._on_variable_changed("k1", 2.0)
    main_window._preview_session.stop_variable_update_timer()
    qt_app.processEvents()
    assert sliders.get_variables()["k1"] == pytest.approx(2.0)
    assert _parameter_table_numeric_value(main_window, "k1") == pytest.approx(2.0)

    _select_rows(main_window, [1])
    _set_shown_rows(main_window, [1])
    qt_app.processEvents()

    plot = main_window._plot_tabs._main_plot
    assert main_window.active_batch_selection()[0] == set1_id
    assert np.allclose(
        np.asarray((getattr(plot, "_series", {}) or {})["A"], dtype=float),
        np.asarray([3.0, 6.0], dtype=float),
    )
    assert sliders.get_variables()["k1"] == pytest.approx(1.0)
    assert _parameter_table_numeric_value(main_window, "k1") == pytest.approx(1.0)

@pytest.mark.gui
@pytest.mark.parametrize(
    ("preserved_reason", "expected_status", "expected_focus_value", "next_value"),
    [
        ("no_cached_results", "Result not cached (evicted). Press Run to compute.", 1.0, 1.5),
        ("invalid_cache_entry", "Cached result invalid. Press Run to compute.", 1.0, 1.5),
        ("preview_pending", "Preview pending for current selection.", 1.5, 1.75),
    ],
)
def test_selection_change_no_display_branch_clears_plot_and_retargets_controls(
    main_window,
    monkeypatch,
    qt_app,
    preserved_reason,
    expected_status,
    expected_focus_value,
    next_value,
):
    from PySide6 import QtWidgets

    main_window._mechanism_editor._reactions_text.setPlainText(
        "reaction: A -> B; k=1.0\ninitial: A=1.0\ninitial: B=0.0"
    )
    main_window._extract_and_populate_variables()
    main_window._batch_model.set_species(["A"])

    add_btn = main_window.findChild(QtWidgets.QPushButton, "addBatchSetButton")
    assert add_btn is not None
    add_btn.click()
    qt_app.processEvents()

    monkeypatch.setattr(
        main_window.simulation_controller,
        "run_simulation_internal",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("selection change triggered run")),
        raising=True,
    )
    monkeypatch.setattr(main_window.simulation_controller, "launch_pending_slider_preview_replay", lambda: None)

    _select_rows(main_window, [0])
    qt_app.processEvents()
    selected_ids = main_window._batch_set_ids_for_scope("selected")
    assert len(selected_ids) == 1
    displayed_set_id = str(selected_ids[0])
    focused_set_id = str(main_window._batch_set_id_for_row(1) or "")
    assert focused_set_id and focused_set_id != displayed_set_id
    main_window.set_slider_edit_target_set_ids([displayed_set_id])

    cache = main_window.simulation_controller.batch_cache
    explicit_key = f"selection-control-display-coherence-{preserved_reason}"
    cache.result_cache[f"{explicit_key}::{displayed_set_id}"] = {
        "t": np.asarray([0.0, 1.0], dtype=float),
        "series": {"A": np.asarray([7.0, 14.0], dtype=float)},
        "algebra_scalars": {},
    }
    if preserved_reason == "invalid_cache_entry":
        cache.result_cache[f"{explicit_key}::{focused_set_id}"] = {
            "t": np.asarray([0.0, 1.0], dtype=float),
            "series": object(),
            "algebra_scalars": {},
        }
    cache.active_cache_key = explicit_key

    outcome = main_window.results_controller.display_cached_batch_selection_outcome(
        cache_key=explicit_key,
        selected_sets=[displayed_set_id],
        prefer_set=displayed_set_id,
        allow_fallback=False,
    )
    assert outcome.displayed is True
    qt_app.processEvents()

    sliders = main_window._mechanism_editor._variable_sliders
    sliders.update_variable("k1", 2.0)
    main_window._on_variable_changed("k1", 2.0)
    main_window._preview_session.stop_variable_update_timer()
    qt_app.processEvents()

    if preserved_reason == "preview_pending":
        main_window._preview_session.stage_slider_value("k1", 1.5, target_set_ids=[focused_set_id])
        cache.active_preview_cache_key = f"missing-preview-selection-cache-key-{preserved_reason}"
        cache.active_preview_scope_set_ids = (focused_set_id,)

    assert main_window.active_batch_selection()[0] == displayed_set_id
    assert sliders.get_variables()["k1"] == pytest.approx(2.0)
    assert _parameter_table_numeric_value(main_window, "k1") == pytest.approx(2.0)

    _select_rows(main_window, [1])
    _set_shown_rows(main_window, [1])
    qt_app.processEvents()

    preview = main_window._preview_session
    assert main_window._status_label.text() == expected_status
    assert main_window.active_batch_selection() == ("", "")
    assert cache.last_display_selection == []
    _assert_selection_plot_cleared(main_window)
    assert sliders.get_variables()["k1"] == pytest.approx(expected_focus_value)
    assert _parameter_table_numeric_value(main_window, "k1") == pytest.approx(expected_focus_value)
    assert preview.local_mechanism_workspace(displayed_set_id) == {"k1": pytest.approx(2.0)}
    expected_workspace = {} if preserved_reason != "preview_pending" else {"k1": pytest.approx(1.5)}
    assert preview.local_mechanism_workspace(focused_set_id) == expected_workspace

    sliders.update_variable("k1", next_value)
    main_window._on_variable_changed("k1", next_value)
    main_window._preview_session.stop_variable_update_timer()
    qt_app.processEvents()

    assert preview.local_mechanism_workspace(displayed_set_id) == {"k1": pytest.approx(next_value)}
    assert preview.local_mechanism_workspace(focused_set_id) == {"k1": pytest.approx(next_value)}
    assert sliders.get_variables()["k1"] == pytest.approx(next_value)
    assert _parameter_table_numeric_value(main_window, "k1") == pytest.approx(next_value)

@pytest.mark.gui
def test_selection_change_reuses_current_workspace_preview_from_inactive_preview_key(
    main_window, monkeypatch, qt_app
):
    t0 = np.asarray([0.0, 1.0], dtype=float)
    series0 = {"A": np.asarray([1.0, 0.5], dtype=float)}
    main_window.set_data(t0, series0, label="baseline", overlays=[])
    plot = main_window._plot_tabs._main_plot

    monkeypatch.setattr(
        main_window.simulation_controller,
        "run_simulation_internal",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("selection change triggered run")),
        raising=True,
    )

    _select_rows(main_window, [0])
    qt_app.processEvents()
    selected_ids = main_window._batch_set_ids_for_scope("selected")
    assert len(selected_ids) == 1
    primary_id = str(selected_ids[0])

    cache = main_window.simulation_controller.batch_cache
    main_window._preview_session.sync_committed_slider_values({"k1": 1.0})
    main_window._preview_session.stage_slider_value("k1", 2.0)

    reusable_preview_key = "preview-current-context-selection-cache-key"
    active_preview_key = "preview-active-context-selection-cache-key"
    preview_mechanism_text = main_window._simulation_batch_owner.mechanism_text_for_workspace_selection(set_id=primary_id)
    preview_t = _current_preview_time_axis(main_window)
    preview_series = {"A": np.linspace(9.0, 18.0, preview_t.size, dtype=float)}
    cache.preview_cache[f"{reusable_preview_key}::{primary_id}"] = {
        "t": preview_t,
        "series": preview_series,
        "algebra_scalars": {},
        "mechanism_text": preview_mechanism_text,
        "solver_config": _current_preview_solver_config(main_window),
    }
    cache.active_preview_cache_key = active_preview_key
    cache.active_preview_scope_set_ids = (primary_id,)
    cache.active_cache_key = None

    main_window._refresh_batch_display_from_focus_and_shown()
    qt_app.processEvents()

    assert np.allclose(np.asarray(getattr(plot, "_t", np.array([])), dtype=float), preview_t)
    assert np.allclose(
        np.asarray((getattr(plot, "_series", {}) or {})["A"], dtype=float),
        np.asarray(preview_series["A"], dtype=float),
    )
    assert main_window.active_batch_selection()[0] == primary_id
    _assert_no_cache_warning(main_window)

@pytest.mark.gui
def test_selection_change_reuses_current_workspace_preview_from_same_key_outside_active_scope(
    main_window, monkeypatch, qt_app
):
    from PySide6 import QtWidgets

    main_window._mechanism_editor._reactions_text.setPlainText(
        "reaction: A -> B; k=1.0\ninitial: A=1.0\ninitial: B=0.0"
    )
    main_window._extract_and_populate_variables()
    main_window._batch_model.set_species(["A"])
    add_btn = main_window.findChild(QtWidgets.QPushButton, "addBatchSetButton")
    assert add_btn is not None
    add_btn.click()
    qt_app.processEvents()

    t0 = np.asarray([0.0, 1.0], dtype=float)
    series0 = {"A": np.asarray([1.0, 0.5], dtype=float)}
    main_window.set_data(t0, series0, label="baseline", overlays=[])
    plot = main_window._plot_tabs._main_plot

    monkeypatch.setattr(
        main_window.simulation_controller,
        "run_simulation_internal",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("selection change triggered run")),
        raising=True,
    )

    _select_rows(main_window, [0, 1])
    qt_app.processEvents()
    selected_ids = main_window._batch_set_ids_for_scope("selected")
    assert len(selected_ids) >= 2
    primary_id = str(selected_ids[0])
    secondary_id = str(selected_ids[1])

    cache = main_window.simulation_controller.batch_cache
    main_window._preview_session.sync_committed_slider_values({"k1": 1.0})
    _select_rows(main_window, [0])
    qt_app.processEvents()
    main_window._preview_session.stage_slider_value("k1", 2.0)

    preview_key = "preview-current-context-same-key-selection-cache-key"
    preview_t = _current_preview_time_axis(main_window)
    preview_series = {"A": np.linspace(4.0, 8.0, preview_t.size, dtype=float)}
    cache.preview_cache[f"{preview_key}::{primary_id}"] = {
        "t": preview_t,
        "series": preview_series,
        "algebra_scalars": {},
        "mechanism_text": main_window._simulation_batch_owner.mechanism_text_for_workspace_selection(set_id=primary_id),
        "solver_config": _current_preview_solver_config(main_window),
    }
    cache.active_preview_cache_key = preview_key
    cache.active_preview_scope_set_ids = (secondary_id,)
    cache.active_cache_key = None

    main_window._refresh_batch_display_from_focus_and_shown()
    qt_app.processEvents()

    assert np.allclose(np.asarray(getattr(plot, "_t", np.array([])), dtype=float), preview_t)
    assert np.allclose(
        np.asarray((getattr(plot, "_series", {}) or {})["A"], dtype=float),
        np.asarray(preview_series["A"], dtype=float),
    )
    assert main_window.active_batch_selection()[0] == primary_id
    _assert_no_cache_warning(main_window)

@pytest.mark.gui
def test_selection_change_rejects_workspace_preview_with_wrong_overlay_token(
    main_window, monkeypatch, qt_app
):
    t0 = np.asarray([0.0, 1.0], dtype=float)
    series0 = {"A": np.asarray([1.0, 0.5], dtype=float)}
    main_window.set_data(t0, series0, label="baseline", overlays=[])

    monkeypatch.setattr(
        main_window.simulation_controller,
        "run_simulation_internal",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("selection change triggered run")),
        raising=True,
    )

    _select_rows(main_window, [0])
    qt_app.processEvents()
    selected_ids = main_window._batch_set_ids_for_scope("selected")
    assert len(selected_ids) == 1
    primary_id = str(selected_ids[0])

    cache = main_window.simulation_controller.batch_cache
    main_window._preview_session.sync_committed_slider_values({"k1": 1.0})
    main_window._preview_session.stage_slider_value("k1", 2.0)
    assert main_window._preview_session.stage_concentration_value_for_rows([0], species="A", value=2.5) is True
    expected_overlay_token = str(main_window._preview_session.preview_batch_cache_token([0]) or "")
    assert expected_overlay_token

    preview_key = "preview-wrong-overlay-token-selection-cache-key"
    cache.preview_cache[f"{preview_key}::{primary_id}"] = _workspace_preview_payload(
        main_window,
        set_id=primary_id,
        series={"A": np.linspace(9.0, 18.0, _current_preview_time_axis(main_window).size, dtype=float)},
        preview_batch_cache_token="set:other|A=7.5",
    )
    cache.active_preview_cache_key = "preview-other-active-overlay-selection-cache-key"
    cache.active_preview_scope_set_ids = (primary_id,)
    cache.active_cache_key = None

    main_window._refresh_batch_display_from_focus_and_shown()
    qt_app.processEvents()

    assert main_window._status_label.text() == "Preview pending for current selection."
    _assert_selection_plot_cleared(main_window)

@pytest.mark.gui
def test_selection_change_reuses_workspace_preview_with_matching_overlay_token(
    main_window, monkeypatch, qt_app
):
    t0 = np.asarray([0.0, 1.0], dtype=float)
    series0 = {"A": np.asarray([1.0, 0.5], dtype=float)}
    main_window.set_data(t0, series0, label="baseline", overlays=[])
    plot = main_window._plot_tabs._main_plot

    monkeypatch.setattr(
        main_window.simulation_controller,
        "run_simulation_internal",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("selection change triggered run")),
        raising=True,
    )

    _select_rows(main_window, [0])
    qt_app.processEvents()
    selected_ids = main_window._batch_set_ids_for_scope("selected")
    assert len(selected_ids) == 1
    primary_id = str(selected_ids[0])

    cache = main_window.simulation_controller.batch_cache
    main_window._preview_session.sync_committed_slider_values({"k1": 1.0})
    main_window._preview_session.stage_slider_value("k1", 2.0)
    assert main_window._preview_session.stage_concentration_value_for_rows([0], species="A", value=2.5) is True
    expected_overlay_token = str(main_window._preview_session.preview_batch_cache_token([0]) or "")
    assert expected_overlay_token

    preview_key = "preview-matching-overlay-token-selection-cache-key"
    preview_t = _current_preview_time_axis(main_window)
    preview_series = {"A": np.linspace(9.0, 18.0, preview_t.size, dtype=float)}
    cache.preview_cache[f"{preview_key}::{primary_id}"] = _workspace_preview_payload(
        main_window,
        set_id=primary_id,
        series=preview_series,
        preview_batch_cache_token=expected_overlay_token,
    )
    cache.active_preview_cache_key = "preview-other-active-overlay-selection-cache-key"
    cache.active_preview_scope_set_ids = (primary_id,)
    cache.active_cache_key = None

    main_window._refresh_batch_display_from_focus_and_shown()
    qt_app.processEvents()

    assert np.allclose(np.asarray(getattr(plot, "_t", np.array([])), dtype=float), preview_t)
    assert np.allclose(
        np.asarray((getattr(plot, "_series", {}) or {})["A"], dtype=float),
        np.asarray(preview_series["A"], dtype=float),
    )
    _assert_no_cache_warning(main_window)

@pytest.mark.gui
def test_selection_change_reuses_workspace_preview_with_matching_structured_identity_despite_text_witness_difference(
    main_window, monkeypatch, qt_app
):
    t0 = np.asarray([0.0, 1.0], dtype=float)
    series0 = {"A": np.asarray([1.0, 0.5], dtype=float)}
    main_window.set_data(t0, series0, label="baseline", overlays=[])
    plot = main_window._plot_tabs._main_plot

    monkeypatch.setattr(
        main_window.simulation_controller,
        "run_simulation_internal",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("selection change triggered run")),
        raising=True,
    )

    _select_rows(main_window, [0])
    qt_app.processEvents()
    selected_ids = main_window._batch_set_ids_for_scope("selected")
    assert len(selected_ids) == 1
    primary_id = str(selected_ids[0])

    cache = main_window.simulation_controller.batch_cache
    main_window._preview_session.sync_committed_slider_values({"k1": 1.0})
    main_window._preview_session.stage_slider_value("k1", 2.0)
    assert main_window._preview_session.stage_concentration_value_for_rows([0], species="A", value=2.5) is True
    expected_overlay_token = str(main_window._preview_session.preview_batch_cache_token([0]) or "")
    assert expected_overlay_token
    expected_identity = _current_preview_identity_payload(
        main_window,
        set_id=primary_id,
        preview_batch_cache_token=expected_overlay_token,
    )

    preview_key = "preview-matching-structured-identity-selection-cache-key"
    preview_t = _current_preview_time_axis(main_window)
    preview_series = {"A": np.linspace(6.0, 12.0, preview_t.size, dtype=float)}
    cache.preview_cache[f"{preview_key}::{primary_id}"] = _workspace_preview_payload(
        main_window,
        set_id=primary_id,
        series=preview_series,
        mechanism_text="reaction: A -> B; k1=999.0\n# witness should not drive identity",
        preview_batch_cache_token=expected_overlay_token,
        simulation_identity=expected_identity,
    )
    cache.active_preview_cache_key = "preview-other-structured-identity-selection-cache-key"
    cache.active_preview_scope_set_ids = (primary_id,)
    cache.active_cache_key = None

    main_window._refresh_batch_display_from_focus_and_shown()
    qt_app.processEvents()

    assert np.allclose(np.asarray(getattr(plot, "_t", np.array([])), dtype=float), preview_t)
    assert np.allclose(
        np.asarray((getattr(plot, "_series", {}) or {})["A"], dtype=float),
        np.asarray(preview_series["A"], dtype=float),
    )
    _assert_no_cache_warning(main_window)

@pytest.mark.gui
def test_selection_change_reuses_workspace_preview_with_slider_override_solver_and_points(
    main_window, monkeypatch, qt_app
):
    t0 = np.asarray([0.0, 1.0], dtype=float)
    series0 = {"A": np.asarray([1.0, 0.5], dtype=float)}
    main_window.set_data(t0, series0, label="baseline", overlays=[])
    plot = main_window._plot_tabs._main_plot

    monkeypatch.setattr(
        main_window.simulation_controller,
        "run_simulation_internal",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("selection change triggered run")),
        raising=True,
    )
    monkeypatch.setattr(main_window._simulation_mechanism_owner, "mechanism_slider_points_value", lambda: 35, raising=True)
    monkeypatch.setattr(main_window._simulation_mechanism_owner, "mechanism_slider_solver_value", lambda: "Radau", raising=True)

    _select_rows(main_window, [0])
    qt_app.processEvents()
    selected_ids = main_window._batch_set_ids_for_scope("selected")
    assert len(selected_ids) == 1
    primary_id = str(selected_ids[0])

    cache = main_window.simulation_controller.batch_cache
    main_window._preview_session.sync_committed_slider_values({"k1": 1.0})
    main_window._preview_session.stage_slider_value("k1", 2.0)

    preview_key = "preview-slider-override-context-selection-cache-key"
    preview_t = _current_preview_time_axis(main_window)
    preview_series = {"A": np.linspace(4.0, 8.0, preview_t.size, dtype=float)}
    solver_config = _current_preview_solver_config(main_window)
    assert solver_config["solver"] == "Radau"
    assert int((solver_config.get("grid") or {}).get("N") or 0) == 50
    cache.preview_cache[f"{preview_key}::{primary_id}"] = _workspace_preview_payload(
        main_window,
        set_id=primary_id,
        series=preview_series,
        solver_config=solver_config,
    )
    cache.active_preview_cache_key = "preview-other-slider-context-selection-cache-key"
    cache.active_preview_scope_set_ids = (primary_id,)
    cache.active_cache_key = None

    main_window._refresh_batch_display_from_focus_and_shown()
    qt_app.processEvents()

    assert np.allclose(np.asarray(getattr(plot, "_t", np.array([])), dtype=float), preview_t)
    assert np.allclose(
        np.asarray((getattr(plot, "_series", {}) or {})["A"], dtype=float),
        np.asarray(preview_series["A"], dtype=float),
    )
    _assert_no_cache_warning(main_window)

@pytest.mark.gui
def test_selection_change_rejects_workspace_preview_with_wrong_solver_context(
    main_window, monkeypatch, qt_app
):
    t0 = np.asarray([0.0, 1.0], dtype=float)
    series0 = {"A": np.asarray([1.0, 0.5], dtype=float)}
    main_window.set_data(t0, series0, label="baseline", overlays=[])

    monkeypatch.setattr(
        main_window.simulation_controller,
        "run_simulation_internal",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("selection change triggered run")),
        raising=True,
    )

    _select_rows(main_window, [0])
    qt_app.processEvents()
    selected_ids = main_window._batch_set_ids_for_scope("selected")
    assert len(selected_ids) == 1
    primary_id = str(selected_ids[0])

    cache = main_window.simulation_controller.batch_cache
    main_window._preview_session.sync_committed_slider_values({"k1": 1.0})
    main_window._preview_session.stage_slider_value("k1", 2.0)

    preview_key = "preview-wrong-solver-context-selection-cache-key"
    wrong_solver_config = _current_preview_solver_config(main_window)
    wrong_solver_config["grid"] = {"N": int((wrong_solver_config.get("grid") or {}).get("N") or 2) + 1}
    cache.preview_cache[f"{preview_key}::{primary_id}"] = {
        "t": _current_preview_time_axis(main_window),
        "series": {"A": np.linspace(9.0, 18.0, _current_preview_time_axis(main_window).size, dtype=float)},
        "algebra_scalars": {},
        "mechanism_text": main_window._simulation_batch_owner.mechanism_text_for_workspace_selection(set_id=primary_id),
        "solver_config": wrong_solver_config,
    }
    cache.active_preview_cache_key = "preview-other-active-context-selection-cache-key"
    cache.active_preview_scope_set_ids = (primary_id,)
    cache.active_cache_key = None

    main_window._refresh_batch_display_from_focus_and_shown()
    qt_app.processEvents()

    assert main_window._status_label.text() == "Preview pending for current selection."
    _assert_selection_plot_cleared(main_window)

@pytest.mark.gui
def test_selection_change_partial_workspace_preview_keeps_resolved_dirty_preview_visible_with_pending_status(
    main_window, monkeypatch, qt_app
):
    from PySide6 import QtWidgets

    baseline_t = np.asarray([0.0, 1.0], dtype=float)
    baseline_series = {"A": np.asarray([1.0, 2.0], dtype=float)}
    main_window.set_data(baseline_t, baseline_series, label="baseline", overlays=[])
    preview_t = _current_preview_time_axis(main_window)
    preview_series = {"A": np.linspace(9.0, 18.0, preview_t.size, dtype=float)}
    plot = main_window._plot_tabs._main_plot

    main_window._mechanism_editor._reactions_text.setPlainText(
        "reaction: A -> B; k=1.0\ninitial: A=1.0\ninitial: B=0.0"
    )
    main_window._extract_and_populate_variables()
    main_window._batch_model.set_species(["A"])
    add_btn = main_window.findChild(QtWidgets.QPushButton, "addBatchSetButton")
    assert add_btn is not None
    add_btn.click()
    qt_app.processEvents()

    monkeypatch.setattr(
        main_window.simulation_controller,
        "run_simulation_internal",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("selection change triggered run")),
        raising=True,
    )

    _select_rows(main_window, [0, 1])
    qt_app.processEvents()
    selected_ids = main_window._batch_set_ids_for_scope("selected")
    assert len(selected_ids) >= 2
    primary_id = str(selected_ids[0])
    secondary_id = str(selected_ids[1])

    cache = main_window.simulation_controller.batch_cache
    explicit_key = "explicit-pending-workspace-selection-cache-key"
    preview_key = "preview-pending-workspace-selection-cache-key"
    cache.result_cache[f"{explicit_key}::{primary_id}"] = {
        "t": np.asarray([0.0, 1.0], dtype=float),
        "series": {"A": np.asarray([4.0, 8.0], dtype=float)},
        "algebra_scalars": {},
    }
    cache.result_cache[f"{explicit_key}::{secondary_id}"] = {
        "t": np.asarray([0.0, 1.0], dtype=float),
        "series": {"A": np.asarray([3.5, 7.0], dtype=float)},
        "algebra_scalars": {},
    }

    main_window._preview_session.sync_committed_slider_values({"k1": 1.0})
    main_window._preview_session.stage_slider_value("k1", 2.0, target_set_ids=selected_ids)

    cache.preview_cache[f"{preview_key}::{primary_id}"] = _workspace_preview_payload(
        main_window,
        set_id=primary_id,
        series={"A": np.asarray(preview_series["A"], dtype=float)},
    )
    cache.active_preview_cache_key = preview_key
    cache.active_preview_scope_set_ids = tuple(str(set_id) for set_id in selected_ids)
    cache.active_cache_key = explicit_key

    main_window._refresh_batch_display_from_focus_and_shown()
    qt_app.processEvents()

    assert main_window._status_label.text() == "Preview pending for current selection."
    assert np.allclose(np.asarray(getattr(plot, "_t", np.array([])), dtype=float), preview_t)
    assert np.allclose(
        np.asarray((getattr(plot, "_series", {}) or {})["A"], dtype=float),
        np.asarray(preview_series["A"], dtype=float),
    )
    assert main_window.active_batch_selection()[0] == primary_id
    overlays = list(getattr(plot, "_simulation_overlays", []) or [])
    assert len(overlays) == 1
    primary_ghost = next(
        entry
        for entry in overlays
        if str(entry.get("set_id") or "") == primary_id and str(entry.get("curve_role") or "") == "canonical_ghost"
    )
    assert np.allclose(
        np.asarray((primary_ghost.get("series") or {})["A"], dtype=float),
        np.asarray([4.0, 8.0], dtype=float),
    )

@pytest.mark.gui

@pytest.mark.gui

@pytest.mark.gui

@pytest.mark.gui

@pytest.mark.gui
def test_selection_change_clean_focused_partial_workspace_preview_keeps_resolved_results_visible_with_pending_status(
    main_window, monkeypatch, qt_app
):
    from PySide6 import QtWidgets

    main_window._mechanism_editor._reactions_text.setPlainText(
        "reaction: A -> B; k=1.0\ninitial: A=1.0\ninitial: B=0.0"
    )
    main_window._extract_and_populate_variables()
    main_window._batch_model.set_species(["A"])
    add_btn = main_window.findChild(QtWidgets.QPushButton, "addBatchSetButton")
    assert add_btn is not None
    add_btn.click()
    add_btn.click()
    qt_app.processEvents()

    monkeypatch.setattr(
        main_window.simulation_controller,
        "run_simulation_internal",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("selection change triggered run")),
        raising=True,
    )

    _select_rows(main_window, [0, 1, 2])
    qt_app.processEvents()
    selected_ids = main_window._batch_set_ids_for_scope("selected")
    assert len(selected_ids) >= 3
    focused_id = str(selected_ids[0])
    resolved_dirty_id = str(selected_ids[1])
    unresolved_dirty_id = str(selected_ids[2])

    cache = main_window.simulation_controller.batch_cache
    explicit_key = "explicit-clean-focused-partial-workspace-selection-cache-key"
    preview_key = "preview-clean-focused-partial-workspace-selection-cache-key"
    cache.result_cache[f"{explicit_key}::{focused_id}"] = {
        "t": np.asarray([0.0, 1.0], dtype=float),
        "series": {"A": np.asarray([5.0, 10.0], dtype=float)},
        "algebra_scalars": {},
    }
    cache.result_cache[f"{explicit_key}::{resolved_dirty_id}"] = {
        "t": np.asarray([0.0, 1.0], dtype=float),
        "series": {"A": np.asarray([3.0, 6.0], dtype=float)},
        "algebra_scalars": {},
    }

    preview_t = _current_preview_time_axis(main_window)
    resolved_dirty_preview = np.linspace(9.0, 18.0, preview_t.size, dtype=float)

    main_window._preview_session.sync_committed_slider_values({"k1": 1.0})
    main_window._preview_session.stage_slider_value("k1", 2.0, target_set_ids=[resolved_dirty_id, unresolved_dirty_id])
    cache.preview_cache[f"{preview_key}::{resolved_dirty_id}"] = _workspace_preview_payload(
        main_window,
        set_id=resolved_dirty_id,
        series={"A": resolved_dirty_preview},
    )
    cache.active_preview_cache_key = preview_key
    cache.active_preview_scope_set_ids = (resolved_dirty_id, unresolved_dirty_id)
    cache.active_cache_key = explicit_key

    main_window._refresh_batch_display_from_focus_and_shown()
    qt_app.processEvents()

    plot = main_window._plot_tabs._main_plot
    assert main_window._status_label.text() == "Preview pending for current selection."
    assert main_window.active_batch_selection()[0] == focused_id
    assert np.allclose(
        np.asarray((getattr(plot, "_series", {}) or {})["A"], dtype=float),
        np.asarray([5.0, 10.0], dtype=float),
    )
    overlays = list(getattr(plot, "_simulation_overlays", []) or [])
    assert len(overlays) == 2
    resolved_dirty_overlay = next(
        entry
        for entry in overlays
        if str(entry.get("set_id") or "") == resolved_dirty_id and str(entry.get("curve_role") or "") != "canonical_ghost"
    )
    assert np.allclose(
        np.asarray((resolved_dirty_overlay.get("series") or {})["A"], dtype=float),
        np.asarray(resolved_dirty_preview, dtype=float),
    )
    resolved_dirty_ghost = next(
        entry
        for entry in overlays
        if str(entry.get("set_id") or "") == resolved_dirty_id and str(entry.get("curve_role") or "") == "canonical_ghost"
    )
    assert np.allclose(
        np.asarray((resolved_dirty_ghost.get("series") or {})["A"], dtype=float),
        np.asarray([3.0, 6.0], dtype=float),
    )
    assert main_window._mechanism_editor._variable_sliders.get_variables()["k1"] == pytest.approx(1.0)
    assert _parameter_table_numeric_value(main_window, "k1") == pytest.approx(1.0)

@pytest.mark.gui
def test_selection_change_clean_focused_partial_no_cached_results_keeps_resolved_results_visible_with_status(
    main_window, monkeypatch, qt_app
):
    from PySide6 import QtWidgets

    main_window._mechanism_editor._reactions_text.setPlainText(
        "reaction: A -> B; k=1.0\ninitial: A=1.0\ninitial: B=0.0"
    )
    main_window._extract_and_populate_variables()
    main_window._batch_model.set_species(["A"])
    add_btn = main_window.findChild(QtWidgets.QPushButton, "addBatchSetButton")
    assert add_btn is not None
    add_btn.click()
    add_btn.click()
    qt_app.processEvents()

    monkeypatch.setattr(
        main_window.simulation_controller,
        "run_simulation_internal",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("selection change triggered run")),
        raising=True,
    )

    _select_rows(main_window, [0, 1, 2])
    qt_app.processEvents()
    selected_ids = main_window._batch_set_ids_for_scope("selected")
    assert len(selected_ids) >= 3
    focused_id = str(selected_ids[0])
    resolved_dirty_id = str(selected_ids[1])

    cache = main_window.simulation_controller.batch_cache
    explicit_key = "explicit-clean-focused-partial-no-cache-selection-key"
    preview_key = "preview-clean-focused-partial-no-cache-selection-key"
    cache.result_cache[f"{explicit_key}::{focused_id}"] = {
        "t": np.asarray([0.0, 1.0], dtype=float),
        "series": {"A": np.asarray([5.0, 10.0], dtype=float)},
        "algebra_scalars": {},
    }
    cache.result_cache[f"{explicit_key}::{resolved_dirty_id}"] = {
        "t": np.asarray([0.0, 1.0], dtype=float),
        "series": {"A": np.asarray([3.0, 6.0], dtype=float)},
        "algebra_scalars": {},
    }

    preview_t = _current_preview_time_axis(main_window)
    resolved_dirty_preview = np.linspace(9.0, 18.0, preview_t.size, dtype=float)

    main_window._preview_session.sync_committed_slider_values({"k1": 1.0})
    main_window._preview_session.stage_slider_value("k1", 2.0, target_set_ids=[resolved_dirty_id])
    cache.preview_cache[f"{preview_key}::{resolved_dirty_id}"] = _workspace_preview_payload(
        main_window,
        set_id=resolved_dirty_id,
        series={"A": resolved_dirty_preview},
    )
    cache.active_preview_cache_key = preview_key
    cache.active_preview_scope_set_ids = (resolved_dirty_id,)
    cache.active_cache_key = explicit_key

    main_window._refresh_batch_display_from_focus_and_shown()
    qt_app.processEvents()

    plot = main_window._plot_tabs._main_plot
    assert main_window._status_label.text() == "Result not cached (evicted). Press Run to compute."
    assert main_window.active_batch_selection()[0] == focused_id
    assert np.allclose(
        np.asarray((getattr(plot, "_series", {}) or {})["A"], dtype=float),
        np.asarray([5.0, 10.0], dtype=float),
    )
    overlays = list(getattr(plot, "_simulation_overlays", []) or [])
    assert len(overlays) == 2
    resolved_dirty_overlay = next(
        entry
        for entry in overlays
        if str(entry.get("set_id") or "") == resolved_dirty_id and str(entry.get("curve_role") or "") != "canonical_ghost"
    )
    assert np.allclose(
        np.asarray((resolved_dirty_overlay.get("series") or {})["A"], dtype=float),
        np.asarray(resolved_dirty_preview, dtype=float),
    )
    resolved_dirty_ghost = next(
        entry
        for entry in overlays
        if str(entry.get("set_id") or "") == resolved_dirty_id and str(entry.get("curve_role") or "") == "canonical_ghost"
    )
    assert np.allclose(
        np.asarray((resolved_dirty_ghost.get("series") or {})["A"], dtype=float),
        np.asarray([3.0, 6.0], dtype=float),
    )
    assert main_window._mechanism_editor._variable_sliders.get_variables()["k1"] == pytest.approx(1.0)
    assert _parameter_table_numeric_value(main_window, "k1") == pytest.approx(1.0)

@pytest.mark.gui
def test_dirty_focused_set_with_missing_preview_does_not_fall_back_to_explicit_cache(
    main_window, monkeypatch, qt_app
):
    main_window._mechanism_editor._reactions_text.setPlainText("A -> B ; k=1.0")
    main_window._extract_and_populate_variables()
    main_window._batch_model.set_species(["A"])

    monkeypatch.setattr(
        main_window.simulation_controller,
        "run_simulation_internal",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("selection change triggered run")),
        raising=True,
    )

    _select_rows(main_window, [0])
    qt_app.processEvents()
    selected_ids = main_window._batch_set_ids_for_scope("selected")
    assert len(selected_ids) == 1
    set_id = str(selected_ids[0])

    cache = main_window.simulation_controller.batch_cache
    explicit_key = "dirty-focused-set-explicit-fallback-key"
    cache.result_cache[f"{explicit_key}::{set_id}"] = {
        "t": np.asarray([0.0, 1.0], dtype=float),
        "series": {"A": np.asarray([2.0, 4.0], dtype=float)},
        "algebra_scalars": {},
    }
    cache.active_cache_key = explicit_key
    cache.active_cache_valid_set_ids = (set_id,)

    main_window._preview_session.sync_committed_slider_values({"k1": 1.0})
    main_window._preview_session.stage_slider_value("k1", 2.0, target_set_ids=[set_id])
    cache.active_preview_cache_key = "missing-dirty-preview-key"
    cache.active_preview_scope_set_ids = (set_id,)

    main_window._refresh_batch_display_from_focus_and_shown()
    qt_app.processEvents()

    assert main_window._status_label.text() == "Preview pending for current selection."
    assert main_window.active_batch_selection() == ("", "")
    _assert_selection_plot_cleared(main_window)

@pytest.mark.gui

@pytest.mark.gui

@pytest.mark.gui
def test_selection_change_partial_preview_coverage_within_active_scope_keeps_dirty_preview_visible(
    main_window, monkeypatch, qt_app
):
    from PySide6 import QtWidgets

    main_window._batch_model.set_species(["A"])
    add_btn = main_window.findChild(QtWidgets.QPushButton, "addBatchSetButton")
    assert add_btn is not None
    add_btn.click()
    qt_app.processEvents()

    _select_rows(main_window, [0, 1])
    qt_app.processEvents()

    monkeypatch.setattr(
        main_window.simulation_controller,
        "run_simulation_internal",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("selection change triggered run")),
        raising=True,
    )

    selected_ids = main_window._batch_set_ids_for_scope("selected")
    assert len(selected_ids) >= 2
    primary_id = str(selected_ids[0])
    secondary_id = str(selected_ids[1])

    preview_t = _current_preview_time_axis(main_window)
    preview_series = {"A": np.linspace(9.0, 18.0, preview_t.size, dtype=float)}
    main_window.set_data(preview_t, preview_series, label="preview", overlays=[])

    cache = main_window.simulation_controller.batch_cache
    explicit_key = "explicit-partial-preview-selection-cache-key"
    preview_key = "preview-partial-selection-cache-key"
    cache.result_cache[f"{explicit_key}::{primary_id}"] = {
        "t": np.asarray([0.0, 1.0], dtype=float),
        "series": {"A": np.asarray([4.0, 8.0], dtype=float)},
        "algebra_scalars": {},
    }
    cache.result_cache[f"{explicit_key}::{secondary_id}"] = {
        "t": np.asarray([0.0, 1.0], dtype=float),
        "series": {"A": np.asarray([3.5, 7.0], dtype=float)},
        "algebra_scalars": {},
    }
    cache.preview_cache[f"{preview_key}::{primary_id}"] = _workspace_preview_payload(
        main_window,
        set_id=primary_id,
        series={"A": np.asarray(preview_series["A"], dtype=float)},
    )
    cache.active_cache_key = explicit_key

    main_window._preview_session.sync_committed_slider_values({"k1": 1.0})
    main_window._preview_session.stage_slider_value("k1", 2.0)
    cache.active_preview_cache_key = preview_key
    cache.active_preview_scope_set_ids = (primary_id, secondary_id)

    main_window._refresh_batch_display_from_focus_and_shown()
    qt_app.processEvents()

    plot = main_window._plot_tabs._main_plot
    _assert_no_cache_warning(main_window)
    assert np.allclose(np.asarray(getattr(plot, "_t", np.array([])), dtype=float), preview_t)
    assert np.allclose(
        np.asarray((getattr(plot, "_series", {}) or {})["A"], dtype=float),
        np.asarray(preview_series["A"], dtype=float),
    )

@pytest.mark.gui
def test_selection_change_partial_preview_without_explicit_cache_keeps_dirty_preview_visible_without_warning(
    main_window, monkeypatch, qt_app
):
    from PySide6 import QtWidgets

    t0 = np.asarray([0.0, 1.0], dtype=float)
    series0 = {"A": np.asarray([1.0, 0.5], dtype=float)}
    main_window.set_data(t0, series0, label="baseline", overlays=[])

    main_window._batch_model.set_species(["A"])
    add_btn = main_window.findChild(QtWidgets.QPushButton, "addBatchSetButton")
    assert add_btn is not None
    add_btn.click()
    qt_app.processEvents()

    _select_rows(main_window, [0, 1])
    qt_app.processEvents()

    monkeypatch.setattr(
        main_window.simulation_controller,
        "run_simulation_internal",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("selection change triggered run")),
        raising=True,
    )

    selected_ids = main_window._batch_set_ids_for_scope("selected")
    assert len(selected_ids) >= 2
    primary_id = str(selected_ids[0])

    cache = main_window.simulation_controller.batch_cache
    preview_key = "preview-partial-no-explicit-selection-cache-key"
    preview_t = _current_preview_time_axis(main_window)
    preview_series = np.linspace(9.0, 18.0, preview_t.size, dtype=float)
    cache.preview_cache[f"{preview_key}::{primary_id}"] = _workspace_preview_payload(
        main_window,
        set_id=primary_id,
        series={"A": preview_series},
    )

    main_window._preview_session.sync_committed_slider_values({"k1": 1.0})
    main_window._preview_session.stage_slider_value("k1", 2.0)
    cache.active_preview_cache_key = preview_key
    cache.active_preview_scope_set_ids = tuple(str(set_id) for set_id in selected_ids)
    cache.active_cache_key = None

    main_window._refresh_batch_display_from_focus_and_shown()
    qt_app.processEvents()

    _assert_no_cache_warning(main_window)
    plot = main_window._plot_tabs._main_plot
    assert np.allclose(np.asarray(getattr(plot, "_t", np.array([])), dtype=float), preview_t)
    assert np.allclose(
        np.asarray((getattr(plot, "_series", {}) or {})["A"], dtype=float),
        np.asarray(preview_series, dtype=float),
    )

@pytest.mark.gui
def test_selection_change_new_current_row_without_workspace_or_cache_context_clears_plot_without_warning(
    main_window, monkeypatch, qt_app
):
    from PySide6 import QtWidgets

    preview_t = np.asarray([0.0, 1.0], dtype=float)
    preview_series = {"A": np.asarray([9.0, 18.0], dtype=float)}
    main_window.set_data(preview_t, preview_series, label="set1", overlays=[])

    main_window._batch_model.set_species(["A"])
    add_btn = main_window.findChild(QtWidgets.QPushButton, "addBatchSetButton")
    assert add_btn is not None
    add_btn.click()
    qt_app.processEvents()

    _select_rows(main_window, [0])
    qt_app.processEvents()

    monkeypatch.setattr(
        main_window.simulation_controller,
        "run_simulation_internal",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("selection change triggered run")),
        raising=True,
    )

    selected_ids = main_window._batch_set_ids_for_scope("selected")
    assert len(selected_ids) == 1
    primary_id = str(selected_ids[0])
    secondary_id = str(main_window._batch_set_id_for_row(1) or "")
    assert secondary_id and secondary_id != primary_id

    cache = main_window.simulation_controller.batch_cache
    preview_key = "preview-partial-current-row-pending-selection-cache-key"
    cache.preview_cache[f"{preview_key}::{primary_id}"] = {
        "t": np.asarray([0.0, 1.0], dtype=float),
        "series": {"A": np.asarray([9.0, 18.0], dtype=float)},
        "algebra_scalars": {},
    }

    main_window._preview_session.sync_committed_slider_values({"k1": 1.0})
    main_window._preview_session.stage_slider_value("k1", 2.0)
    cache.active_preview_cache_key = preview_key
    cache.active_preview_scope_set_ids = (primary_id, secondary_id)
    cache.active_batch_set_id = primary_id
    cache.active_batch_set = str(main_window.batch_set_name_for_id(primary_id) or primary_id)
    cache.last_display_selection = [primary_id]

    _select_rows(main_window, [1])
    qt_app.processEvents()
    main_window._refresh_batch_display_from_focus_and_shown()
    qt_app.processEvents()

    _assert_no_cache_warning(main_window)
    _assert_selection_plot_cleared(main_window)
    assert main_window.active_batch_selection() == ("", "")

@pytest.mark.gui
def test_selection_change_partial_preview_with_invalid_explicit_cache_keeps_dirty_preview_visible(
    main_window, monkeypatch, qt_app
):
    from PySide6 import QtWidgets

    preview_t = _current_preview_time_axis(main_window)
    preview_series = {"A": np.linspace(9.0, 18.0, preview_t.size, dtype=float)}
    main_window.set_data(preview_t, preview_series, label="preview", overlays=[])

    main_window._batch_model.set_species(["A"])
    add_btn = main_window.findChild(QtWidgets.QPushButton, "addBatchSetButton")
    assert add_btn is not None
    add_btn.click()
    qt_app.processEvents()

    _select_rows(main_window, [0, 1])
    qt_app.processEvents()

    monkeypatch.setattr(
        main_window.simulation_controller,
        "run_simulation_internal",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("selection change triggered run")),
        raising=True,
    )

    selected_ids = main_window._batch_set_ids_for_scope("selected")
    assert len(selected_ids) >= 2
    primary_id = str(selected_ids[0])

    cache = main_window.simulation_controller.batch_cache
    preview_key = "preview-partial-invalid-explicit-selection-cache-key"
    explicit_key = "invalid-explicit-selection-cache-key"
    cache.preview_cache[f"{preview_key}::{primary_id}"] = _workspace_preview_payload(
        main_window,
        set_id=primary_id,
        series={"A": np.asarray(preview_series["A"], dtype=float)},
    )
    cache.result_cache[f"{explicit_key}::{primary_id}"] = {
        "t": np.asarray([0.0, 1.0], dtype=float),
        "series": object(),
        "algebra_scalars": {},
    }

    main_window._preview_session.sync_committed_slider_values({"k1": 1.0})
    main_window._preview_session.stage_slider_value("k1", 2.0)
    cache.active_preview_cache_key = preview_key
    cache.active_preview_scope_set_ids = tuple(str(set_id) for set_id in selected_ids)
    cache.active_cache_key = explicit_key

    main_window._refresh_batch_display_from_focus_and_shown()
    qt_app.processEvents()

    plot = main_window._plot_tabs._main_plot
    _assert_no_cache_warning(main_window)
    assert np.allclose(np.asarray(getattr(plot, "_t", np.array([])), dtype=float), preview_t)
    assert np.allclose(
        np.asarray((getattr(plot, "_series", {}) or {})["A"], dtype=float),
        np.asarray(preview_series["A"], dtype=float),
    )

@pytest.mark.gui
def test_selection_change_invalid_preview_without_usable_explicit_fallback_preserves_invalid_diagnostic(
    main_window, monkeypatch, qt_app
):
    from PySide6 import QtWidgets

    t0 = np.asarray([0.0, 1.0], dtype=float)
    series0 = {"A": np.asarray([1.0, 0.5], dtype=float)}
    main_window.set_data(t0, series0, label="baseline", overlays=[])

    main_window._batch_model.set_species(["A"])
    add_btn = main_window.findChild(QtWidgets.QPushButton, "addBatchSetButton")
    assert add_btn is not None
    add_btn.click()
    qt_app.processEvents()

    _select_rows(main_window, [0, 1])
    qt_app.processEvents()

    monkeypatch.setattr(
        main_window.simulation_controller,
        "run_simulation_internal",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("selection change triggered run")),
        raising=True,
    )

    selected_ids = main_window._batch_set_ids_for_scope("selected")
    assert len(selected_ids) >= 2
    primary_id = str(selected_ids[0])
    secondary_id = str(selected_ids[1])

    cache = main_window.simulation_controller.batch_cache
    preview_key = "preview-invalid-diagnostic-selection-cache-key"
    cache.preview_cache[f"{preview_key}::{primary_id}"] = {
        "t": np.asarray([0.0, 1.0], dtype=float),
        "series": object(),
        "algebra_scalars": {},
    }
    cache.preview_cache[f"{preview_key}::{secondary_id}"] = {
        "t": np.asarray([0.0, 1.0], dtype=float),
        "series": {"A": np.asarray([9.0, 18.0], dtype=float)},
        "algebra_scalars": {},
    }

    main_window._preview_session.sync_committed_slider_values({"k1": 1.0})
    main_window._preview_session.stage_slider_value("k1", 2.0)
    cache.active_preview_cache_key = preview_key
    cache.active_cache_key = None

    main_window._refresh_batch_display_from_focus_and_shown()
    qt_app.processEvents()

    assert main_window._status_label.text() == "Cached result invalid. Press Run to compute."
    assert main_window.active_batch_selection() == ("", "")
    assert cache.last_display_selection == []
    _assert_selection_plot_cleared(main_window)

@pytest.mark.gui
def test_selection_change_missing_dirty_preview_does_not_fall_back_to_explicit_cache(
    main_window, monkeypatch, qt_app
):
    from PySide6 import QtWidgets

    main_window._batch_model.set_species(["A"])
    add_btn = main_window.findChild(QtWidgets.QPushButton, "addBatchSetButton")
    assert add_btn is not None
    add_btn.click()
    qt_app.processEvents()

    _select_rows(main_window, [0, 1])
    qt_app.processEvents()

    monkeypatch.setattr(
        main_window.simulation_controller,
        "run_simulation_internal",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("selection change triggered run")),
        raising=True,
    )

    selected_ids = main_window._batch_set_ids_for_scope("selected")
    assert len(selected_ids) >= 2
    primary_id = str(selected_ids[0])
    secondary_id = str(selected_ids[1])

    cache = main_window.simulation_controller.batch_cache
    explicit_key = "explicit-fallback-selection-cache-key"
    cache.result_cache[f"{explicit_key}::{primary_id}"] = {
        "t": np.asarray([0.0, 1.0], dtype=float),
        "series": {"A": np.asarray([3.0, 6.0], dtype=float)},
        "algebra_scalars": {},
    }
    cache.result_cache[f"{explicit_key}::{secondary_id}"] = {
        "t": np.asarray([0.0, 1.0], dtype=float),
        "series": {"A": np.asarray([2.5, 5.0], dtype=float)},
        "algebra_scalars": {},
    }
    cache.active_cache_key = explicit_key

    main_window._preview_session.sync_committed_slider_values({"k1": 1.0})
    main_window._preview_session.stage_slider_value("k1", 2.0)
    cache.active_preview_cache_key = "missing-preview-selection-cache-key"

    main_window._refresh_batch_display_from_focus_and_shown()
    qt_app.processEvents()

    assert main_window._status_label.text() == "Preview pending for current selection."
    assert main_window.active_batch_selection() == ("", "")
    _assert_selection_plot_cleared(main_window)

@pytest.mark.gui
def test_invalid_cache_entry_on_selection_change_sets_invalid_message_and_clears_display_state(
    main_window, monkeypatch, qt_app
):
    from PySide6 import QtWidgets

    t0 = np.asarray([0.0, 1.0], dtype=float)
    series0 = {"A": np.asarray([1.0, 0.5], dtype=float)}
    main_window.set_data(t0, series0, label="baseline", overlays=[])

    main_window._batch_model.set_species(["A"])
    add_btn = main_window.findChild(QtWidgets.QPushButton, "addBatchSetButton")
    assert add_btn is not None
    add_btn.click()
    qt_app.processEvents()
    _select_rows(main_window, [0, 1])

    monkeypatch.setattr(
        main_window.simulation_controller,
        "run_simulation_internal",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("selection change triggered run")),
        raising=True,
    )

    cache_key = "invalid-cache-key"
    selected_ids = main_window._batch_set_ids_for_scope("selected")
    assert selected_ids
    main_window.simulation_controller.batch_cache.result_cache[f"{cache_key}::{selected_ids[0]}"] = {
        "t": np.asarray([0.0, 1.0], dtype=float),
        "series": object(),
        "algebra_scalars": {},
    }
    main_window.simulation_controller.batch_cache.active_cache_key = cache_key

    main_window._refresh_batch_display_from_focus_and_shown()
    qt_app.processEvents()

    assert main_window._status_label.text() == "Cached result invalid. Press Run to compute."
    assert main_window.active_batch_selection() == ("", "")
    assert main_window.simulation_controller.batch_cache.last_display_selection == []
    _assert_selection_plot_cleared(main_window)


@pytest.mark.gui
def test_live_multiset_preview_completion_keeps_schema_stable_and_workspace_preview_matchable(
    main_window, monkeypatch, qt_app
):
    from PySide6 import QtWidgets

    main_window._mechanism_editor._reactions_text.setPlainText(
        "reaction: A -> B; k=1.0\ninitial: A=1.0\ninitial: B=0.0"
    )
    main_window._extract_and_populate_variables()
    _clear_parallel_runtime_pool(main_window)
    main_window._batch_model.set_species(["A"])

    add_btn = main_window.findChild(QtWidgets.QPushButton, "addBatchSetButton")
    assert add_btn is not None
    add_btn.click()
    qt_app.processEvents()

    _select_rows(main_window, [0, 1])
    qt_app.processEvents()
    selected_ids = [str(set_id) for set_id in main_window._batch_set_ids_for_scope("selected")]
    assert len(selected_ids) >= 2
    primary_id = selected_ids[0]
    _set_edit_target_rows(main_window, [0, 1])
    qt_app.processEvents()

    fake = _FakeLanePool()
    monkeypatch.setattr(main_window.simulation_controller.parallel_batch, "max_parallel_workers", 12, raising=True)
    monkeypatch.setattr(
        main_window.simulation_controller.parallel_batch,
        "lane_pool_factory",
        lambda max_workers, limit_blas_threads: fake,
        raising=True,
    )
    monkeypatch.setattr("kindred.core.batch_parallel.os.cpu_count", lambda: 8)

    baseline_text = str(main_window.mechanism_reactions_text_raw() or "")
    baseline_schema_id = str(main_window.simulation_schema_id() or "")
    preview_t = _current_preview_time_axis(main_window)
    preview_series = np.linspace(9.0, 18.0, preview_t.size, dtype=float)

    main_window._on_slider_drag_started("k1")
    main_window._on_variable_changed("k1", 2.0)
    main_window._preview_session.stop_variable_update_timer()
    main_window.simulation_controller.launch_pending_slider_preview_replay()
    qt_app.processEvents()

    assert main_window.simulation_controller.parallel_batch.lane_pool_token() == id(fake)
    _wait_for_submission_count(fake, 2)
    simulation_submissions = _simulation_submissions(fake)
    assert len(simulation_submissions) >= 2

    first_task = dict(simulation_submissions[0].args[0])
    assert str(first_task.get("set_id") or "") == primary_id
    simulation_submissions[0].complete(
        {
            "run_id": int(first_task.get("run_id") or 0),
            "set_id": str(first_task.get("set_id") or ""),
            "set_name": str(first_task.get("set_name") or ""),
            "t": np.asarray(preview_t, dtype=float),
            "Y": np.asarray([preview_series], dtype=float),
            "species_names": ["A"],
            "algebra_scalars": {},
            "mechanism": None,
            "mechanism_text": str(first_task.get("mechanism_text") or ""),
            "solver_config": dict(first_task.get("solver_config") or {}),
            "fallback_occurred": False,
            "fallback_message": None,
            "base_species_count": 1,
        }
    )
    _join_batch_request(main_window, first_task)

    main_window.simulation_controller.poll_parallel_batch_completions()
    qt_app.processEvents()

    assert str(main_window.mechanism_reactions_text_raw() or "") == baseline_text
    assert str(main_window.simulation_schema_id() or "") == baseline_schema_id

    preview_entry = main_window._simulation_batch_owner.matching_preview_entry_for_workspace_set(set_id=primary_id)
    assert preview_entry.entry is not None

    main_window._refresh_batch_display_from_focus_and_shown()
    qt_app.processEvents()

    plot = main_window._plot_tabs._main_plot
    assert main_window._status_label.text() == "Preview pending for current selection."
    assert np.allclose(np.asarray(getattr(plot, "_t", np.array([])), dtype=float), preview_t)
    assert np.allclose(
        np.asarray((getattr(plot, "_series", {}) or {})["A"], dtype=float),
        np.asarray(preview_series, dtype=float),
    )


@pytest.mark.gui
def test_simulation_schema_id_tracks_canonical_edit_while_dirty_workspace_exists(
    main_window, qt_app
):
    set_id = str(main_window._batch_set_id_for_row(0) or "set1")
    main_window._mechanism_editor._reactions_text.setPlainText(
        "reaction: A -> B; k=1\n"
        "initial: A=1.0\n"
        "initial: B=0.0\n"
    )
    qt_app.processEvents()
    baseline_schema_id = str(main_window.simulation_schema_id() or "")

    param_store = main_window._preview_session.param_store
    param_store.sync_shared_params({"k1": 1.0})
    assert param_store.stage_override(set_id, "k1", 2.0) is True
    assert main_window._preview_session.has_local_mechanism_workspace(set_id) is True

    main_window._mechanism_editor._reactions_text.setPlainText(
        "reaction: A -> C; k=1\n"
        "initial: A=1.0\n"
        "initial: C=0.0\n"
    )
    qt_app.processEvents()
    changed_schema_id = str(main_window.simulation_schema_id() or "")

    assert changed_schema_id
    assert changed_schema_id != baseline_schema_id
    assert main_window._preview_session.local_mechanism_workspace(set_id) == {"k1": pytest.approx(2.0)}


@pytest.mark.gui
def test_live_multiset_parameter_preview_replays_after_partial_stale_completion(
    main_window, monkeypatch, qt_app
):
    from PySide6 import QtWidgets

    main_window._mechanism_editor._reactions_text.setPlainText(
        "reaction: A -> B; k=1.0\ninitial: A=1.0\ninitial: B=0.0"
    )
    main_window._extract_and_populate_variables()
    _clear_parallel_runtime_pool(main_window)
    main_window._batch_model.set_species(["A"])

    add_btn = main_window.findChild(QtWidgets.QPushButton, "addBatchSetButton")
    assert add_btn is not None
    add_btn.click()
    qt_app.processEvents()

    _select_rows(main_window, [0, 1])
    qt_app.processEvents()
    selected_ids = [str(set_id) for set_id in main_window._batch_set_ids_for_scope("selected")]
    assert len(selected_ids) >= 2
    primary_id = selected_ids[0]
    _set_edit_target_rows(main_window, [0, 1])
    qt_app.processEvents()

    fake = _FakeLanePool()
    monkeypatch.setattr(main_window.simulation_controller.parallel_batch, "max_parallel_workers", 12, raising=True)
    monkeypatch.setattr(
        main_window.simulation_controller.parallel_batch,
        "lane_pool_factory",
        lambda max_workers, limit_blas_threads: fake,
        raising=True,
    )
    monkeypatch.setattr("kindred.core.batch_parallel.os.cpu_count", lambda: 8)

    preview_t = _current_preview_time_axis(main_window)
    preview_series = np.linspace(9.0, 18.0, preview_t.size, dtype=float)

    main_window._on_slider_drag_started("k1")
    main_window._on_variable_changed("k1", 2.0)
    main_window._preview_session.stop_variable_update_timer()
    main_window.simulation_controller.launch_pending_slider_preview_replay()
    qt_app.processEvents()

    assert main_window.simulation_controller.parallel_batch.lane_pool_token() == id(fake)
    _wait_for_submission_count(fake, 2)
    simulation_submissions = _simulation_submissions(fake)
    assert len(simulation_submissions) == 2

    main_window._on_variable_changed("k1", 3.0)
    main_window._preview_session.stop_variable_update_timer()
    main_window.simulation_controller.launch_pending_slider_preview_replay()
    qt_app.processEvents()

    simulation_submissions = _simulation_submissions(fake)
    assert len(simulation_submissions) == 2
    for stale_submission in simulation_submissions:
        stale_task = dict(stale_submission.args[0])
        stale_submission.complete(
            {
                "success": True,
                "run_id": int(stale_task.get("run_id") or 0),
                "set_id": str(stale_task.get("set_id") or ""),
                "set_name": str(stale_task.get("set_name") or ""),
            }
        )
    _wait_for_submission_count(fake, 4)
    simulation_submissions = _simulation_submissions(fake)
    assert len(simulation_submissions) == 4

    first_generation_task = dict(simulation_submissions[0].args[0])
    replay_submission = next(
        sub for sub in simulation_submissions[2:] if str(sub.args[0].get("set_id") or "") == str(primary_id)
    )
    replay_task = dict(replay_submission.args[0])
    replay_metadata = main_window.simulation_controller.parallel_batch.active_request_metadata(primary_id)
    replay_identity = replay_metadata.get("callback_identity")
    assert replay_metadata["preview_owner_epoch"] == replay_identity.owner_epoch
    assert replay_metadata["owner_epoch"] == replay_identity.owner_epoch
    replay_submission.complete(
        {
            "run_id": int(replay_task.get("run_id") or 0),
                "set_id": str(replay_task.get("set_id") or ""),
                "set_name": str(replay_task.get("set_name") or ""),
                "owner_epoch": int(replay_identity.owner_epoch or 0),
                "t": np.asarray(preview_t, dtype=float),
            "Y": np.asarray([preview_series], dtype=float),
            "species_names": ["A"],
            "algebra_scalars": {},
            "mechanism": None,
            "mechanism_text": str(replay_task.get("mechanism_text") or ""),
            "solver_config": dict(replay_task.get("solver_config") or {}),
            "fallback_occurred": False,
            "fallback_message": None,
            "base_species_count": 1,
        }
    )
    _join_batch_request(main_window, replay_task)

    main_window.simulation_controller.poll_parallel_batch_completions()
    qt_app.processEvents()
    qt_app.processEvents()
    coalescer_timer = main_window.simulation_controller.plot_coalescer.timer
    if coalescer_timer.isActive():
        coalescer_timer.timeout.emit()
        qt_app.processEvents()

    assert len(_simulation_submissions(fake)) == 4
    assert int(replay_task.get("request_id") or 0) > int(first_generation_task.get("request_id") or 0)
    plot = main_window._plot_tabs._main_plot
    assert np.allclose(np.asarray(getattr(plot, "_t", np.array([])), dtype=float), preview_t)
    assert np.allclose(
        np.asarray((getattr(plot, "_series", {}) or {})["A"], dtype=float),
        np.asarray(preview_series, dtype=float),
    )


@pytest.mark.gui
def test_selection_change_displays_selected_valid_cached_row_outside_latest_valid_subset(
    main_window, monkeypatch, qt_app
):
    from PySide6 import QtWidgets

    t0 = np.asarray([0.0, 1.0], dtype=float)
    series0 = {"A": np.asarray([1.0, 0.5], dtype=float)}
    main_window.set_data(t0, series0, label="baseline", overlays=[])
    plot = main_window._plot_tabs._main_plot

    main_window._batch_model.set_species(["A"])
    add_btn = main_window.findChild(QtWidgets.QPushButton, "addBatchSetButton")
    assert add_btn is not None
    add_btn.click()
    qt_app.processEvents()

    selected_ids = main_window._batch_set_ids_for_scope("selected")
    if len(selected_ids) < 2:
        _select_rows(main_window, [0, 1])
        qt_app.processEvents()
        selected_ids = main_window._batch_set_ids_for_scope("selected")
    assert len(selected_ids) >= 2

    older_id = str(selected_ids[0])
    newer_id = str(selected_ids[1])

    monkeypatch.setattr(
        main_window.simulation_controller,
        "run_simulation_internal",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("selection change triggered run")),
        raising=True,
    )

    cache_key = "partial-invalid-cache-key"
    main_window.simulation_controller.batch_cache.result_cache[f"{cache_key}::{older_id}"] = {
        "t": np.asarray([0.0, 1.0], dtype=float),
        "series": {"A": np.asarray([9.0, 9.0], dtype=float)},
        "algebra_scalars": {},
    }
    main_window.simulation_controller.batch_cache.result_cache[f"{cache_key}::{newer_id}"] = {
        "t": np.asarray([0.0, 1.0], dtype=float),
        "series": {"A": np.asarray([2.0, 1.5], dtype=float)},
        "algebra_scalars": {},
    }
    cache = main_window.simulation_controller.batch_cache
    cache.active_cache_key = cache_key
    cache.active_cache_valid_set_ids = (newer_id,)
    cache.active_batch_set_id = newer_id
    cache.active_batch_set = str(main_window.batch_set_name_for_id(newer_id) or newer_id)
    cache.last_display_selection = [newer_id]

    _select_rows(main_window, [0])
    qt_app.processEvents()
    main_window._refresh_batch_display_from_focus_and_shown()
    qt_app.processEvents()

    _assert_no_cache_warning(main_window)
    assert np.allclose(np.asarray(getattr(plot, "_t", np.array([])), dtype=float), np.asarray([0.0, 1.0], dtype=float))
    assert np.allclose(
        np.asarray((getattr(plot, "_series", {}) or {})["A"], dtype=float),
        np.asarray([9.0, 9.0], dtype=float),
    )
    assert main_window.active_batch_selection()[0] == older_id


@pytest.mark.gui
def test_completion_redraw_keeps_newer_valid_result_authoritative_after_active_subset_narrows(main_window, qt_app):
    from PySide6 import QtWidgets

    t0 = np.asarray([0.0, 1.0], dtype=float)
    series0 = {"A": np.asarray([1.0, 0.5], dtype=float)}
    main_window.set_data(t0, series0, label="baseline", overlays=[])
    plot = main_window._plot_tabs._main_plot

    main_window._batch_model.set_species(["A"])
    add_btn = main_window.findChild(QtWidgets.QPushButton, "addBatchSetButton")
    assert add_btn is not None
    add_btn.click()
    qt_app.processEvents()

    _select_rows(main_window, [0])
    qt_app.processEvents()

    selected_ids = main_window._batch_set_ids_for_scope("selected")
    assert selected_ids
    older_id = str(selected_ids[0])
    newer_id = str(main_window._batch_set_id_for_row(1) or "")
    assert newer_id and newer_id != older_id

    cache_key = "completion-invalid-redraw-cache-key"
    cache = main_window.simulation_controller.batch_cache
    cache.result_cache[f"{cache_key}::{older_id}"] = {
        "t": np.asarray([0.0, 1.0], dtype=float),
        "series": {"A": np.asarray([9.0, 9.0], dtype=float)},
        "algebra_scalars": {},
    }
    cache.active_cache_key = cache_key
    cache.active_cache_valid_set_ids = (newer_id,)
    cache.active_batch_set_id = newer_id
    cache.active_batch_set = str(main_window.batch_set_name_for_id(newer_id) or newer_id)
    cache.last_display_selection = [newer_id]

    main_window.simulation_controller.run_state.latest_sim_request_id = 21
    main_window.simulation_controller.run_state.active_run_id = 8
    seed_batch_context(main_window.simulation_controller.batch_context_owner, active=True, parallel=False, run_id=8, request_id=21, fast_mode=False, cache_key=cache_key, queue_ids=[newer_id], queue_names=[
                str(main_window.batch_set_name_for_id(newer_id) or newer_id),
            ], primary_set_id=newer_id, total=1, explicit_cache_valid_set_ids=(newer_id,))
    callback_identity = main_window.simulation_controller._capture_simulation_callback_identity(
        run_id=8,
        fast_mode=False,
        request_id=21,
        owner_epoch=None,
        batch_set=str(main_window.batch_set_name_for_id(newer_id) or newer_id),
        batch_set_id=newer_id,
        cache_key=cache_key,
        callback_context=main_window.simulation_controller.batch_context_owner.callback_context_snapshot(),
        simulation_identity={},
        preview_batch_cache_token="",
    )

    main_window.simulation_controller.on_simulation_complete(
        _fake_sim_result(marker=2.0),
        callback_identity=callback_identity,
    )
    qt_app.processEvents()

    _assert_no_cache_warning(main_window)
    assert np.allclose(np.asarray(getattr(plot, "_t", np.array([])), dtype=float), np.asarray([0.0, 1.0], dtype=float))
    assert np.allclose(
        np.asarray((getattr(plot, "_series", {}) or {})["A"], dtype=float),
        np.asarray([2.0, 4.0], dtype=float),
    )
    assert main_window.active_batch_selection()[0] == newer_id
