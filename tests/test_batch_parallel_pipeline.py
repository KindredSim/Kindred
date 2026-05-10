from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import pytest
from PySide6 import QtCore

from kindred.core.batch_containment import BatchLaneOutcome
from kindred.core.batch_parallel import run_batch_simulation_task

pytestmark = [pytest.mark.gui]


def _batch_task_with_plan(task: dict[str, Any]) -> dict[str, Any]:
    from kindred.core.simulation_plan import SimulationAlgebraPolicy, SimulationPlan

    copied = dict(task)
    t_span_raw = copied.get("t_span") or (0.0, float(copied.get("t_end") or 0.0))
    execution_request = {
        "prepared_payload": copied.get("prepared_payload"),
        "initials": dict(copied.get("initials") or {}),
        "t_span": (float(t_span_raw[0]), float(t_span_raw[1])),
        "solver_config": dict(copied.get("solver_config") or {}),
        "mechanism_text": str(copied.get("mechanism_text") or ""),
    }
    copied["simulation_plan"] = SimulationPlan.from_execution_request(
        execution_request,
        execution_mode="explicit",
        algebra_policy=SimulationAlgebraPolicy.BATCH_BEST_EFFORT,
    ).to_payload()
    return copied


@dataclass
class _Submission:
    fn: Any
    args: tuple[Any, ...]
    kwargs: Dict[str, Any]
    result_placeholder: Any = None


class _FakeLanePool:
    def __init__(self, *, done_immediately: bool = False, value_marker: float = 1.0) -> None:
        self.done_immediately = bool(done_immediately)
        self.value_marker = float(value_marker)
        self.submissions: List[_Submission] = []
        self.close_calls: List[Dict[str, Any]] = []

    def run(self, task, *, run_id: int, request_id: int, set_id: str, active_timeout_s: float):
        _ = active_timeout_s
        args = (dict(task or {}),)
        sub = _Submission(fn=run_batch_simulation_task, args=args, kwargs={}, result_placeholder=None)
        self.submissions.append(sub)
        sid = str(task.get("set_id") or task.get("batch_set_id") or set_id or "")
        payload = {
            "run_id": int(task.get("run_id") or run_id or 0),
            "request_id": int(task.get("request_id") or request_id or 0),
            "set_id": sid,
            "set_name": str(task.get("set_name") or sid or "set"),
            "t": np.array([0.0, 1.0]),
            "Y": np.array([[self.value_marker, self.value_marker]]),
            "species_names": ["A"],
            "algebra_scalars": {},
            "mechanism": None,
            "mechanism_text": str(task.get("mechanism_text") or "reaction: A -> B ; k=0.1"),
            "solver_config": dict(task.get("solver_config") or {}),
            "fallback_occurred": False,
            "fallback_message": None,
        }
        return BatchLaneOutcome(
            lane_id="fake-lane",
            run_id=int(run_id),
            request_id=int(request_id),
            set_id=sid,
            owner_epoch=1,
            success=True,
            payload=payload,
        )

    def _close_requests(self, *, kill: bool = False):
        self.close_calls.append(
            {
                "kill": bool(kill),
            }
        )

    def close(self, *, kill: bool = False):
        self.close_calls.append(
            {
                "wait": False,
                "kill": bool(kill),
            }
        )



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



def _prime_three_batch_sets(main_window) -> list[str]:
    main_window._mechanism_editor._reactions_text.setPlainText("reaction: A -> B ; k=0.25")
    main_window._batch_model.set_species(["A"])

    add_btn = main_window.findChild(type(main_window._run_btn), "addBatchSetButton")
    assert add_btn is not None
    add_btn.click()
    add_btn.click()

    names = list(main_window._batch_store.set_names())
    for idx, name in enumerate(names[:3]):
        row = main_window._batch_store.row_for_set(name)
        assert row is not None
        main_window._batch_store.set_value(int(row), "A", f"{1.0 + idx:.6g}")
    return names[:3]


def _simulation_submissions(lane_pool: _FakeLanePool) -> list[_Submission]:
    return [sub for sub in lane_pool.submissions if sub.fn is run_batch_simulation_task]


def _wait_for_submission_count(lane_pool: _FakeLanePool, expected: int, timeout_s: float = 1.0) -> None:
    deadline = time.monotonic() + float(timeout_s)
    while time.monotonic() < deadline:
        if len(_simulation_submissions(lane_pool)) >= int(expected):
            return
        time.sleep(0.005)



def test_worker_policy_uses_num_sets_cpu_minus_one_and_cap(monkeypatch):
    from kindred.core.batch_parallel import compute_effective_batch_workers

    monkeypatch.setattr("kindred.core.batch_parallel.os.cpu_count", lambda: 20)
    assert compute_effective_batch_workers(num_sets=30, max_parallel_workers=12) == 12
    assert compute_effective_batch_workers(num_sets=5, max_parallel_workers=12) == 5

    monkeypatch.setattr("kindred.core.batch_parallel.os.cpu_count", lambda: 2)
    assert compute_effective_batch_workers(num_sets=10, max_parallel_workers=12) == 1



def test_parallel_pipeline_submits_all_sets_without_serial_wait(main_window, monkeypatch, qtbot):
    names = _prime_three_batch_sets(main_window)
    _select_rows(main_window, [0, 1, 2])
    monkeypatch.setattr("kindred.core.batch_parallel.os.cpu_count", lambda: 8)

    fake = _FakeLanePool(done_immediately=True, value_marker=2.0)

    monkeypatch.setattr(main_window.simulation_controller.parallel_batch, "max_parallel_workers", 12, raising=True)
    monkeypatch.setattr(
        main_window.simulation_controller.parallel_batch,
        "lane_pool_factory",
        lambda max_workers, limit_blas_threads: fake,
        raising=True,
    )
    main_window.simulation_controller.parallel_batch.ensure_lane_pool(max_lanes=len(names))
    main_window._simulation_run_ui_owner.set_runtime_backed_run_controls_ready(True)

    # "Run All" was intentionally removed; emulate it via Select All + Run Selected.
    _select_rows(main_window, [0, 1, 2])
    main_window.simulation_controller.run_simulation()
    _wait_for_submission_count(fake, len(names))
    qtbot.wait(40)

    assert len(_simulation_submissions(fake)) == len(names)



def test_new_run_cancels_old_lane_pool_and_rejects_stale_results(main_window, monkeypatch):
    if hasattr(main_window, "set_simulation_cache_caps"):
        main_window.set_simulation_cache_caps(result_cap=20, preview_cap=20)
    names = _prime_three_batch_sets(main_window)
    _select_rows(main_window, [0, 1, 2])
    monkeypatch.setattr("kindred.core.batch_parallel.os.cpu_count", lambda: 8)

    lane_pools: List[_FakeLanePool] = []

    def _factory(max_workers, limit_blas_threads):
        fake = _FakeLanePool(done_immediately=False, value_marker=float(len(lane_pools) + 1))
        lane_pools.append(fake)
        return fake

    monkeypatch.setattr(main_window.simulation_controller.parallel_batch, "max_parallel_workers", 12, raising=True)
    monkeypatch.setattr(main_window.simulation_controller.parallel_batch, "lane_pool_factory", _factory, raising=True)
    main_window.simulation_controller.parallel_batch.ensure_lane_pool(max_lanes=len(names))
    req1 = main_window.simulation_controller.next_sim_request_id()
    main_window.simulation_controller.run_simulation_internal(
        fast_mode=False,
        request_id=int(req1),
        batch_rows=[0, 1, 2],
        reuse_parallel_lane_pool=True,
    )
    assert len(lane_pools) == 1
    _wait_for_submission_count(lane_pools[0], len(names))
    old_callback_identity_by_set_id = {}
    for sub in _simulation_submissions(lane_pools[0]):
        task = dict(sub.args[0] if sub.args else {})
        sid = str(task.get("set_id") or "")
        metadata = main_window.simulation_controller.parallel_batch.active_request_metadata(sid)
        old_callback_identity_by_set_id[sid] = metadata.get("callback_identity")

    main_window.simulation_controller._shutdown_batch_lane_pool(force_terminate=True)
    main_window.simulation_controller.parallel_batch.ensure_lane_pool(max_lanes=len(names))
    req2 = main_window.simulation_controller.next_sim_request_id()
    main_window.simulation_controller.run_simulation_internal(
        fast_mode=False,
        request_id=int(req2),
        batch_rows=[0, 1, 2],
        reuse_parallel_lane_pool=True,
    )
    assert len(lane_pools) == 2
    _wait_for_submission_count(lane_pools[1], len(names))
    new_callback_identity_by_set_id = {}
    for sub in _simulation_submissions(lane_pools[1]):
        task = dict(sub.args[0] if sub.args else {})
        sid = str(task.get("set_id") or "")
        metadata = main_window.simulation_controller.parallel_batch.active_request_metadata(sid)
        new_callback_identity_by_set_id[sid] = metadata.get("callback_identity")

    old_pool = lane_pools[0]
    new_pool = lane_pools[1]

    assert old_pool.close_calls

    cache_key = str(main_window.simulation_controller.batch_cache.active_cache_key or "")
    assert cache_key

    stale_task = dict(_simulation_submissions(old_pool)[0].args[0] if _simulation_submissions(old_pool)[0].args else {})
    stale_sid = str(stale_task.get("set_id") or "")
    stale_callback_identity = old_callback_identity_by_set_id[stale_sid]
    assert stale_callback_identity is not None
    main_window.simulation_controller.on_simulation_complete(
        {
            "run_id": int(stale_task.get("run_id") or 0),
            "set_id": stale_sid,
            "set_name": str(stale_task.get("set_name") or stale_sid),
            "t": np.array([0.0, 1.0]),
            "Y": np.array([[111.0, 111.0]]),
            "species_names": ["A"],
            "algebra_scalars": {},
            "mechanism": None,
            "mechanism_text": str(stale_task.get("mechanism_text") or "reaction: A -> B ; k=0.1"),
            "solver_config": dict(stale_task.get("solver_config") or {}),
            "fallback_occurred": False,
            "fallback_message": None,
        },
        callback_identity=stale_callback_identity,
    )

    assert (
        main_window.simulation_controller.batch_cache.entry_for_set(
            cache_key=cache_key,
            set_id=stale_sid,
            is_preview=False,
        ).entry
        is None
    )

    for sub in _simulation_submissions(new_pool):
        task = dict(sub.args[0] if sub.args else {})
        sid = str(task.get("set_id") or "")
        callback_identity = new_callback_identity_by_set_id[sid]
        assert callback_identity is not None
        main_window.simulation_controller.on_simulation_complete(
            {
                "run_id": int(task.get("run_id") or 0),
                "set_id": sid,
                "set_name": str(task.get("set_name") or sid),
                "t": np.array([0.0, 1.0]),
                "Y": np.array([[222.0, 222.0]]),
                "species_names": ["A"],
                "algebra_scalars": {},
                "mechanism": None,
                "mechanism_text": str(task.get("mechanism_text") or "reaction: A -> B ; k=0.1"),
                "solver_config": dict(task.get("solver_config") or {}),
                "fallback_occurred": False,
                "fallback_message": None,
            },
            callback_identity=callback_identity,
        )

    cached_payloads = []
    for sub in _simulation_submissions(new_pool):
        task = dict(sub.args[0] if sub.args else {})
        sid = str(task.get("set_id") or "")
        payload = main_window.simulation_controller.batch_cache.entry_for_set(
            cache_key=cache_key,
            set_id=sid,
            is_preview=False,
        ).entry
        assert isinstance(payload, dict)
        cached_payloads.append(payload)
    assert cached_payloads
    for payload in cached_payloads:
        series = payload.get("series") or {}
        arr = np.asarray(series.get("A"))
        assert arr.size
        assert float(arr[-1]) == pytest.approx(222.0)


def test_batch_task_surfaces_solver_validation_from_preparation_owner(monkeypatch):
    from kindred.core import batch_parallel
    from kindred.core.simulation_preparation import SimulationPreparationError

    class _FakeMechanism:
        def clone(self):
            return self

    fake_bound = type(
        "_FakeBound",
        (),
        {
            "mechanism": _FakeMechanism(),
            "y0": np.asarray([1.0], dtype=float),
        },
    )()

    monkeypatch.setattr(
        batch_parallel,
        "_prepared_entry",
        lambda **_kwargs: {
            "bound": fake_bound,
            "prepared_payload": {
                "version": 1,
                "mechanism": fake_bound.mechanism,
                "rhs": lambda _t, y: y,
                "y0": np.asarray([1.0], dtype=float),
                "species_names": ["A"],
                "mechanism_text": "reaction: A -> A; k=1",
                "temperature_schedule": None,
                "jacobian_func": None,
            },
        },
    )

    seen: Dict[str, Any] = {}

    def _raise_from_prepare(*, solver_config, **_kwargs):
        seen["solver_config"] = dict(solver_config or {})
        raise SimulationPreparationError("solver_config", "owner sentinel")

    monkeypatch.setattr(
        "kindred.core.simulation_preparation.prepare_simulation_worker_run",
        _raise_from_prepare,
    )

    payload = batch_parallel.run_batch_simulation_task(
        _batch_task_with_plan({
            "mechanism_text": "reaction: A -> A; k=1",
            "solver_config": {"solver": "BDF", "rtol": "bad"},
            "t_end": 1.0,
            "set_id": "id1",
            "set_name": "set1",
        })
    )

    assert seen["solver_config"]["rtol"] == "bad"
    assert payload["success"] is False
    assert payload["error"]["kind"] == "preparation_error"
    assert payload["error"]["message"] == "owner sentinel"
    assert payload["error"]["details"]["stage"] == "solver_config"


def test_batch_task_surfaces_solver_validation_from_preparation_owner_after_prepared_entry_failure(
    monkeypatch,
):
    from kindred.core import batch_parallel
    from kindred.core.simulation_preparation import SimulationPreparationError

    seen: Dict[str, Any] = {}

    def _prepared_entry_fail(**_kwargs):
        raise RuntimeError("prepared entry exploded")

    def _raise_from_prepare(*, solver_config, prepared_payload, t_span, initials, **_kwargs):
        seen["solver_config"] = dict(solver_config or {})
        seen["prepared_payload"] = dict(prepared_payload or {})
        seen["t_span"] = tuple(t_span)
        seen["initials"] = dict(initials or {})
        raise SimulationPreparationError("solver_config", "owner fallback sentinel")

    monkeypatch.setattr(batch_parallel, "_prepared_entry", _prepared_entry_fail)
    monkeypatch.setattr(
        "kindred.core.simulation_preparation.prepare_simulation_worker_run",
        _raise_from_prepare,
    )

    payload = batch_parallel.run_batch_simulation_task(
        _batch_task_with_plan({
            "mechanism_text": "reaction: A -> A; k=1",
            "solver_config": {"solver": "BDF", "rtol": "bad"},
            "t_span": (2.0, 5.0),
            "initials": {"A": 3.0},
            "set_id": "id1",
            "set_name": "set1",
        })
    )

    assert seen["solver_config"]["rtol"] == "bad"
    assert seen["prepared_payload"] == {"version": 0}
    assert seen["t_span"] == (2.0, 5.0)
    assert seen["initials"] == {"A": 3.0}
    assert payload["success"] is False
    assert payload["error"]["kind"] == "preparation_error"
    assert payload["error"]["message"] == "owner fallback sentinel"
    assert payload["error"]["details"]["stage"] == "solver_config"



def test_blas_thread_limit_sets_worker_env(monkeypatch):
    from kindred.core.batch_parallel import BLAS_THREAD_ENV_VARS, apply_worker_blas_limits
    from kindred.core.runtime_defaults import CONTAINED_CHILD_BLAS_THREAD_ENV_VARS

    assert BLAS_THREAD_ENV_VARS == CONTAINED_CHILD_BLAS_THREAD_ENV_VARS

    env: Dict[str, str] = {}
    apply_worker_blas_limits(enabled=True, environ=env)
    for var in BLAS_THREAD_ENV_VARS:
        assert env.get(var) == "1"

    env2: Dict[str, str] = {"OMP_NUM_THREADS": "8"}
    apply_worker_blas_limits(enabled=False, environ=env2)
    assert env2["OMP_NUM_THREADS"] == "8"


def test_open_solver_settings_refreshes_runtime_after_final_settings_apply(main_window, monkeypatch):
    class _FakeDialog:
        def __init__(self, _parent, *, cache_port=None):
            self._settings = {}
            self._cache_port = cache_port

        def set_settings(self, settings):
            self._settings = dict(settings or {})

        def exec(self):
            return True

        def get_settings(self):
            return {
                "solver": "Radau",
                "rtol": 2e-6,
                "atol": 1e-12,
                "use_sparse_jacobian": False,
                "max_parallel_batch_workers": 7,
                "limit_blas_threads_per_worker": False,
                "slider_preview_solver": "Radau",
                "slider_preview_points": 375,
            }

    observations: list[dict[str, object]] = []

    def _snapshot(event: str) -> None:
        observations.append(
            {
                "event": str(event),
                "solver": str(main_window._initial_solver),
                "rtol": float(main_window._initial_rtol),
                "slider_preview_solver": str(main_window._mechanism_editor.slider_solver_value()),
                "slider_preview_points": int(main_window._mechanism_editor.slider_points_value()),
                "max_workers": int(main_window.simulation_controller.parallel_batch.max_parallel_workers),
                "limit_blas": bool(
                    main_window.simulation_controller.parallel_batch.limit_blas_threads_per_worker
                ),
            }
        )

    monkeypatch.setattr(
        "kindred.gui.widgets.solver_settings.SolverSettingsDialog",
        _FakeDialog,
    )
    monkeypatch.setattr(
        main_window.simulation_controller,
        "simulation_runtime_inputs_changed",
        lambda: _snapshot("controller_refresh"),
        raising=False,
    )
    monkeypatch.setattr(
        main_window,
        "_schedule_simulation_runtime_availability_refresh",
        lambda *, wait=False, force_when_hidden=False: _snapshot("readiness_schedule"),
    )

    main_window._open_solver_settings()

    assert int(main_window.simulation_controller.parallel_batch.max_parallel_workers) == 7
    assert bool(main_window.simulation_controller.parallel_batch.limit_blas_threads_per_worker) is False
    assert observations == [
        {
            "event": "controller_refresh",
            "solver": "Radau",
            "rtol": pytest.approx(2e-6),
            "slider_preview_solver": "Radau",
            "slider_preview_points": 375,
            "max_workers": 7,
            "limit_blas": False,
        },
        {
            "event": "readiness_schedule",
            "solver": "Radau",
            "rtol": pytest.approx(2e-6),
            "slider_preview_solver": "Radau",
            "slider_preview_points": 375,
            "max_workers": 7,
            "limit_blas": False,
        },
    ]


def test_open_solver_settings_notifies_active_fit_windows_for_runtime_changes(main_window, monkeypatch):
    class _FakeDialog:
        def __init__(self, _parent, *, cache_port=None):
            self._settings = {}
            self._cache_port = cache_port

        def set_settings(self, settings):
            self._settings = dict(settings or {})

        def exec(self):
            return True

        def get_settings(self):
            updated = dict(self._settings)
            updated["use_sparse_jacobian"] = not bool(updated.get("use_sparse_jacobian", False))
            return updated

    notifications: list[str] = []

    class _FitWindow:
        def handle_external_runtime_inputs_changed(self) -> None:
            notifications.append("notified")

        def close(self) -> bool:
            return True

    monkeypatch.setattr(
        "kindred.gui.widgets.solver_settings.SolverSettingsDialog",
        _FakeDialog,
    )
    main_window._active_fit_windows = [_FitWindow()]

    main_window._open_solver_settings()

    assert notifications == ["notified"]


def test_open_solver_settings_does_not_notify_fit_windows_for_local_fit_integration_defaults(main_window, monkeypatch):
    class _FakeDialog:
        def __init__(self, _parent, *, cache_port=None):
            self._settings = {}
            self._cache_port = cache_port

        def set_settings(self, settings):
            self._settings = dict(settings or {})

        def exec(self):
            return True

        def get_settings(self):
            updated = dict(self._settings)
            updated["rtol"] = 2e-6
            return updated

    notifications: list[str] = []

    class _FitWindow:
        def handle_external_runtime_inputs_changed(self) -> None:
            notifications.append("notified")

        def close(self) -> bool:
            return True

    monkeypatch.setattr(
        "kindred.gui.widgets.solver_settings.SolverSettingsDialog",
        _FakeDialog,
    )
    main_window._active_fit_windows = [_FitWindow()]

    main_window._open_solver_settings()

    assert notifications == []


def test_active_fit_window_runtime_notification_continues_after_stale_window_error(main_window):
    notifications: list[str] = []

    class _BadFitWindow:
        def handle_external_runtime_inputs_changed(self) -> None:
            raise RuntimeError("deleted wrapper")

        def close(self) -> bool:
            return True

    class _GoodFitWindow:
        def handle_external_runtime_inputs_changed(self) -> None:
            notifications.append("notified")

        def close(self) -> bool:
            return True

    main_window._active_fit_windows = [_BadFitWindow(), _GoodFitWindow()]

    main_window._notify_active_fit_windows_runtime_inputs_changed()

    assert notifications == ["notified"]


def test_open_solver_settings_persists_preview_debounce_controls(main_window, monkeypatch):
    class _FakeDialog:
        def __init__(self, _parent, *, cache_port=None):
            self._settings = {}
            self._cache_port = cache_port

        def set_settings(self, settings):
            self._settings = dict(settings or {})

        def exec(self):
            return True

        def get_settings(self):
            updated = dict(self._settings)
            updated["slider_preview_solver"] = "BDF"
            updated["slider_preview_points"] = 350
            updated["parameter_preview_debounce_ms"] = 35
            updated["equilibrium_preview_debounce_ms"] = 90
            return updated

    monkeypatch.setattr(
        "kindred.gui.widgets.solver_settings.SolverSettingsDialog",
        _FakeDialog,
    )

    main_window._open_solver_settings()

    settings = main_window._settings_owner.qsettings
    assert main_window._mechanism_editor.slider_solver_value() == "BDF"
    assert main_window._mechanism_editor.slider_points_value() == 350
    assert settings.value("simulation/slider_preview_solver", type=str) == "BDF"
    assert settings.value("simulation/slider_preview_points", type=int) == 350
    assert settings.value("simulation/parameter_preview_debounce_ms", type=int) == 35
    assert settings.value("simulation/equilibrium_preview_debounce_ms", type=int) == 90
    assert main_window._preview_session.variable_preview_debounce_ms("k1") == 35
    assert main_window._preview_session.variable_preview_debounce_ms("Keq1") == 90


def test_slider_release_timer_uses_persisted_preview_debounce_controls(main_window, monkeypatch, qtbot):
    class _FakeDialog:
        def __init__(self, _parent, *, cache_port=None):
            self._settings = {}
            self._cache_port = cache_port

        def set_settings(self, settings):
            self._settings = dict(settings or {})

        def exec(self):
            return True

        def get_settings(self):
            updated = dict(self._settings)
            updated["parameter_preview_debounce_ms"] = 35
            updated["equilibrium_preview_debounce_ms"] = 90
            return updated

    monkeypatch.setattr(
        "kindred.gui.widgets.solver_settings.SolverSettingsDialog",
        _FakeDialog,
    )

    main_window._open_solver_settings()
    main_window._mechanism_editor._reactions_text.setPlainText(
        "equilibrium: A <-> B ; kf=0.0928966, K=0.00963829\n"
        "reaction: B -> C ; k=0.1\n"
        "init: A=1, B=0, C=0\n"
    )
    main_window._extract_and_populate_variables()
    monkeypatch.setattr(main_window.simulation_controller, "launch_pending_slider_preview_replay", lambda: None)

    preview = main_window._preview_session
    sliders = main_window._mechanism_editor._variable_sliders

    main_window._on_slider_drag_started("Keq1")
    slider_widget = sliders._sliders["Keq1"]
    slider_widget.setValue(min(slider_widget.maximum(), slider_widget.value() + 50))

    qtbot.waitUntil(lambda: "Keq1" in preview._pending_slider_values, timeout=1000)
    main_window._on_slider_drag_finished("Keq1")

    release_timer = getattr(preview, "_slider_release_commit_timer", None)
    assert release_timer is not None
    assert release_timer.interval() == 90
    release_timer.stop()
