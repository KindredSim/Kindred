from __future__ import annotations

from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import pytest
from PySide6 import QtCore


pytestmark = [pytest.mark.gui]


@dataclass
class _Submission:
    fn: Any
    args: tuple[Any, ...]
    kwargs: Dict[str, Any]
    future: Future


class _FakeExecutor:
    def __init__(self, *, done_immediately: bool = False, value_marker: float = 1.0) -> None:
        self.done_immediately = bool(done_immediately)
        self.value_marker = float(value_marker)
        self.submissions: List[_Submission] = []
        self.shutdown_calls: List[Dict[str, Any]] = []

    def submit(self, fn, *args, **kwargs):
        fut: Future = Future()
        sub = _Submission(fn=fn, args=args, kwargs=dict(kwargs), future=fut)
        self.submissions.append(sub)
        if self.done_immediately:
            task = dict(args[0] if args else {})
            sid = str(task.get("set_id") or task.get("batch_set_id") or "")
            fut.set_result(
                {
                    "run_id": int(task.get("run_id") or 0),
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
            )
        return fut

    def shutdown(self, wait=True, cancel_futures=False):
        self.shutdown_calls.append(
            {
                "wait": bool(wait),
                "cancel_futures": bool(cancel_futures),
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

    fake = _FakeExecutor(done_immediately=True, value_marker=2.0)

    monkeypatch.setattr(main_window.simulation_controller.parallel_batch, "max_parallel_workers", 12, raising=True)
    monkeypatch.setattr(
        main_window.simulation_controller.parallel_batch,
        "executor_factory",
        lambda max_workers, limit_blas_threads: fake,
        raising=True,
    )

    # "Run All" was intentionally removed; emulate it via Select All + Run Selected.
    _select_rows(main_window, [0, 1, 2])
    main_window.simulation_controller.run_simulation()
    qtbot.wait(40)

    assert len(fake.submissions) == len(names)



def test_new_run_cancels_old_executor_and_rejects_stale_results(main_window, monkeypatch, qtbot):
    if hasattr(main_window, "set_simulation_cache_caps"):
        main_window.set_simulation_cache_caps(result_cap=20, preview_cap=20)
    _prime_three_batch_sets(main_window)
    _select_rows(main_window, [0, 1, 2])
    monkeypatch.setattr("kindred.core.batch_parallel.os.cpu_count", lambda: 8)

    executors: List[_FakeExecutor] = []

    def _factory(max_workers, limit_blas_threads):
        fake = _FakeExecutor(done_immediately=False, value_marker=float(len(executors) + 1))
        executors.append(fake)
        return fake

    monkeypatch.setattr(main_window.simulation_controller.parallel_batch, "max_parallel_workers", 12, raising=True)
    monkeypatch.setattr(main_window.simulation_controller.parallel_batch, "executor_factory", _factory, raising=True)

    req1 = main_window.simulation_controller.next_sim_request_id()
    main_window.simulation_controller.run_simulation_internal(fast_mode=False, request_id=int(req1), batch_rows=[0, 1, 2])
    assert len(executors) == 1

    req2 = main_window.simulation_controller.next_sim_request_id()
    main_window.simulation_controller.run_simulation_internal(fast_mode=False, request_id=int(req2), batch_rows=[0, 1, 2])
    assert len(executors) == 2

    old_exec = executors[0]
    new_exec = executors[1]

    assert old_exec.shutdown_calls

    cache_key = str(main_window.simulation_controller.batch_cache.active_cache_key or "")
    assert cache_key

    for sub in old_exec.submissions:
        task = dict(sub.args[0] if sub.args else {})
        sid = str(task.get("set_id") or "")
        sub.future.set_result(
            {
                "run_id": int(task.get("run_id") or 0),
                "set_id": sid,
                "set_name": str(task.get("set_name") or sid),
                "t": np.array([0.0, 1.0]),
                "Y": np.array([[111.0, 111.0]]),
                "species_names": ["A"],
                "algebra_scalars": {},
                "mechanism": None,
                "mechanism_text": str(task.get("mechanism_text") or "reaction: A -> B ; k=0.1"),
                "solver_config": dict(task.get("solver_config") or {}),
                "fallback_occurred": False,
                "fallback_message": None,
            }
        )

    for sub in new_exec.submissions:
        task = dict(sub.args[0] if sub.args else {})
        sid = str(task.get("set_id") or "")
        sub.future.set_result(
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
            }
        )

    qtbot.wait(80)

    cached_payloads = []
    for sub in new_exec.submissions:
        task = dict(sub.args[0] if sub.args else {})
        sid = str(task.get("set_id") or "")
        payload = main_window.simulation_controller.batch_cache.result_cache.get(f"{cache_key}::{sid}")
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
        {
            "mechanism_text": "reaction: A -> A; k=1",
            "solver_config": {"solver": "LSODA", "rtol": "bad"},
            "t_end": 1.0,
            "set_id": "id1",
            "set_name": "set1",
        }
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
        {
            "mechanism_text": "reaction: A -> A; k=1",
            "solver_config": {"solver": "LSODA", "rtol": "bad"},
            "t_span": (2.0, 5.0),
            "initials": {"A": 3.0},
            "set_id": "id1",
            "set_name": "set1",
        }
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

    env: Dict[str, str] = {}
    apply_worker_blas_limits(enabled=True, environ=env)
    for var in BLAS_THREAD_ENV_VARS:
        assert env.get(var) == "1"

    env2: Dict[str, str] = {"OMP_NUM_THREADS": "8"}
    apply_worker_blas_limits(enabled=False, environ=env2)
    assert env2["OMP_NUM_THREADS"] == "8"


def test_open_solver_settings_wires_parallel_batch_controls(main_window, monkeypatch):
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
                "solver": "LSODA",
                "rtol": 1e-6,
                "atol": 1e-12,
                "use_sparse_jacobian": False,
                "max_parallel_batch_workers": 7,
                "limit_blas_threads_per_worker": False,
            }

    monkeypatch.setattr(
        "kindred.gui.widgets.solver_settings.SolverSettingsDialog",
        _FakeDialog,
    )
    main_window._open_solver_settings()
    assert int(main_window.simulation_controller.parallel_batch.max_parallel_workers) == 7
    assert bool(main_window.simulation_controller.parallel_batch.limit_blas_threads_per_worker) is False


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

    assert main_window._mechanism_editor.slider_solver_value() == "BDF"
    assert main_window._mechanism_editor.slider_points_value() == 350
    assert main_window._settings.value("simulation/slider_preview_solver", type=str) == "BDF"
    assert main_window._settings.value("simulation/slider_preview_points", type=int) == 350
    assert main_window._settings.value("simulation/parameter_preview_debounce_ms", type=int) == 35
    assert main_window._settings.value("simulation/equilibrium_preview_debounce_ms", type=int) == 90
    assert main_window._preview_session.variable_preview_debounce_ms("k1") == 35
    assert main_window._preview_session.variable_preview_debounce_ms("K1") == 90


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
    monkeypatch.setattr(main_window.simulation_controller, "run_simulation_from_slider", lambda: None)

    preview = main_window._preview_session
    sliders = main_window._mechanism_editor._variable_sliders

    main_window._on_slider_drag_started("K1")
    slider_widget = sliders._sliders["K1"]
    slider_widget.setValue(min(slider_widget.maximum(), slider_widget.value() + 50))

    qtbot.waitUntil(lambda: "K1" in preview._pending_slider_values, timeout=1000)
    main_window._on_slider_drag_finished("K1")

    release_timer = getattr(preview, "_slider_release_commit_timer", None)
    assert release_timer is not None
    assert release_timer.interval() == 90
    release_timer.stop()
