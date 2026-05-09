from __future__ import annotations

import time
from dataclasses import dataclass, field
from queue import Queue
from typing import Any, Dict, List

import numpy as np
import pytest
from PySide6 import QtCore

from kindred.core.batch_containment import BatchLaneOutcome
from kindred.core.batch_parallel import run_batch_simulation_task
from kindred.gui.controllers.simulation_run_state import PreviewOwnershipState
from tests.batch_context_test_helpers import seed_batch_context

pytestmark = [pytest.mark.gui]


@dataclass
class _Submission:
    fn: Any
    args: tuple[Any, ...]
    kwargs: Dict[str, Any]
    _result_queue: Queue[Any] = field(default_factory=Queue)
    completed: bool = False

    def complete(self, payload: Any) -> None:
        if self.completed:
            return
        self.completed = True
        self._result_queue.put(payload)

    def cancel(self) -> None:
        self.complete({"success": False, "error": {"kind": "cancelled"}})

    def wait_result(self, *, timeout_s: float) -> Any:
        return self._result_queue.get(timeout=float(timeout_s))


class _FakeLanePool:
    def __init__(self) -> None:
        self.submissions: List[_Submission] = []
        self.close_calls: List[Dict[str, Any]] = []

    def run(self, task, *, run_id: int, request_id: int, set_id: str, active_timeout_s: float):
        _ = run_id, request_id, set_id, active_timeout_s
        sub = _Submission(fn=run_batch_simulation_task, args=(dict(task),), kwargs={})
        self.submissions.append(sub)
        payload = sub.wait_result(timeout_s=5.0)
        return BatchLaneOutcome(
            lane_id="fake-lane",
            run_id=int(run_id),
            request_id=int(request_id),
            set_id=str(set_id),
            owner_epoch=1,
            success=not (isinstance(payload, dict) and payload.get("success") is False),
            payload=payload if isinstance(payload, dict) else {"payload": payload},
        )

    def submit(self, fn, *args, **kwargs):
        sub = _Submission(fn=fn, args=args, kwargs=dict(kwargs))
        self.submissions.append(sub)
        return sub

    def _close_requests(self, *, kill: bool = False):
        self.close_calls.append(
            {
                "kill": bool(kill),
            }
        )
        for sub in self.submissions:
            if not sub.completed:
                if bool(kill):
                    sub.cancel()
                else:
                    task = dict(sub.args[0] if sub.args else {})
                    sub.complete(_result_payload(task, marker=0.0))

    def close(self, *, kill: bool = False):
        self._close_requests(kill=bool(kill))


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
    main_window._batch_model.set_species(["A", "B"])

    add_btn = main_window.findChild(type(main_window._run_btn), "addBatchSetButton")
    assert add_btn is not None
    add_btn.click()
    add_btn.click()

    names = list(main_window._batch_store.set_names())
    for idx, name in enumerate(names[:3]):
        row = main_window._batch_store.row_for_set(name)
        assert row is not None
        main_window._batch_store.set_value(int(row), "A", f"{1.0 + idx:.6g}")
        main_window._batch_store.set_value(int(row), "B", "0.0")
    return names[:3]


def _queue_slider_run(main_window) -> None:
    main_window.simulation_controller.queue_pending_slider_preview_replay(
        target_set_ids=main_window._batch_set_ids_for_scope("selected"),
        request_id=main_window.simulation_controller.next_sim_request_id(),
    )
    main_window.simulation_controller.launch_pending_slider_preview_replay()


def _simulation_submissions(lane_pool: _FakeLanePool) -> list[_Submission]:
    return [sub for sub in lane_pool.submissions if sub.fn is run_batch_simulation_task]


def _wait_for_submission_count(lane_pool: _FakeLanePool, expected: int, timeout_s: float = 1.0) -> None:
    deadline = time.monotonic() + float(timeout_s)
    while time.monotonic() < deadline:
        QtCore.QCoreApplication.processEvents()
        if len(_simulation_submissions(lane_pool)) >= int(expected):
            return
        time.sleep(0.005)


def _join_batch_request(main_window, task: Dict[str, Any]) -> None:
    _ = task
    main_window.simulation_controller.parallel_batch.join_active_requests(timeout_s=1.0)


def _result_payload(task: Dict[str, Any], marker: float) -> Dict[str, Any]:
    sid = str(task.get("set_id") or "")
    return {
        "run_id": int(task.get("run_id") or 0),
        "set_id": sid,
        "set_name": str(task.get("set_name") or sid or "set"),
        "t": np.array([0.0, 1.0], dtype=float),
        "Y": np.array([[marker, marker]], dtype=float),
        "species_names": ["A"],
        "algebra_scalars": {},
        "mechanism": None,
        "mechanism_text": str(task.get("mechanism_text") or "reaction: A -> B ; k=0.1"),
        "solver_config": dict(task.get("solver_config") or {}),
        "fallback_occurred": False,
        "fallback_message": None,
    }


def test_parallel_completion_consumes_completed_lane_records_in_completion_order(main_window, monkeypatch):
    _prime_three_batch_sets(main_window)
    _select_rows(main_window, [0, 1, 2])
    monkeypatch.setattr("kindred.core.batch_parallel.os.cpu_count", lambda: 8)

    fake = _FakeLanePool()
    monkeypatch.setattr(main_window.simulation_controller.parallel_batch, "max_parallel_workers", 12, raising=True)
    monkeypatch.setattr(
        main_window.simulation_controller.parallel_batch,
        "lane_pool_factory",
        lambda max_workers, limit_blas_threads: fake,
        raising=True,
    )

    _queue_slider_run(main_window)
    _wait_for_submission_count(fake, 3)
    simulation_submissions = _simulation_submissions(fake)
    assert len(simulation_submissions) >= 3

    processed: list[str] = []
    monkeypatch.setattr(
        main_window.simulation_controller._completion_callback_owner,
        "handle_completion",
        lambda _result, **kwargs: processed.append(str(kwargs.get("batch_set_id") or "")),
        raising=True,
    )

    first = dict(simulation_submissions[0].args[0])
    last = dict(simulation_submissions[-1].args[0])
    sid_first = str(first.get("set_id") or "")
    sid_last = str(last.get("set_id") or "")
    assert sid_first and sid_last and sid_first != sid_last

    simulation_submissions[-1].complete(_result_payload(last, marker=3.0))
    _join_batch_request(main_window, last)
    simulation_submissions[0].complete(_result_payload(first, marker=1.0))
    _join_batch_request(main_window, first)

    main_window.simulation_controller.poll_parallel_batch_completions()

    assert processed[:2] == [sid_last, sid_first]
    main_window.simulation_controller.shutdown_batch_lane_pool(force_terminate=True)


def test_slider_parallel_plot_updates_are_coalesced_per_ui_tick(main_window, monkeypatch):
    main_window.simulation_controller.run_state.latest_sim_request_id = 101
    main_window.simulation_controller.run_state.active_run_id = 77
    main_window.simulation_controller.run_state.preview_ownership = PreviewOwnershipState(
        request_id=101,
        epoch=1,
        target_set_ids=("set1", "set2", "set3", "set4", "set5"),
    )
    main_window.simulation_controller.batch_cache.active_cache_key = "coalesce-key"
    seed_batch_context(main_window.simulation_controller.batch_context_owner, active=True, parallel=True, run_id=77, request_id=101, fast_mode=True, cache_key="coalesce-key", primary_set_id="set1", total=5, completed_set_ids=[], queue_ids=["set1", "set2", "set3", "set4", "set5"], queue_names=["set1", "set2", "set3", "set4", "set5"])

    batch_port = main_window.simulation_controller.ui.batch
    monkeypatch.setattr(batch_port, "batch_set_ids_for_scope", lambda _scope: ["set1", "set2", "set3"], raising=True)
    monkeypatch.setattr(batch_port, "batch_current_row", lambda: 0, raising=True)
    monkeypatch.setattr(batch_port, "batch_set_id_for_row", lambda _row: "set1", raising=True)

    display_calls: list[Dict[str, Any]] = []

    def _display(**kwargs):
        display_calls.append(dict(kwargs))
        return True

    monkeypatch.setattr(batch_port, "display_cached_batch_selection", _display, raising=True)

    result = {
        "t": np.array([0.0, 1.0], dtype=float),
        "Y": np.array([[1.0, 0.5]], dtype=float),
        "species_names": ["A"],
        "algebra_scalars": {},
        "mechanism": None,
        "mechanism_text": "reaction: A -> B ; k=0.1",
        "solver_config": {},
        "fallback_occurred": False,
        "fallback_message": None,
    }

    main_window.simulation_controller.on_simulation_complete(
        result,
        run_id=77,
        fast_mode=True,
        request_id=101,
        batch_set="set2",
        batch_set_id="set2",
        cache_key="coalesce-key",
    )
    main_window.simulation_controller.on_simulation_complete(
        result,
        run_id=77,
        fast_mode=True,
        request_id=101,
        batch_set="set3",
        batch_set_id="set3",
        cache_key="coalesce-key",
    )

    assert display_calls == []
    assert set(main_window.simulation_controller.plot_coalescer.pending.set_ids) == {"set2", "set3"}

    timer = main_window.simulation_controller.plot_coalescer.timer
    assert timer.isActive() is True

    timer.timeout.emit()

    assert len(display_calls) == 1
    assert display_calls[0].get("cache_key") == "coalesce-key"
    assert set(main_window.simulation_controller.plot_coalescer.pending.set_ids) == set()


def test_slider_supersession_still_reuses_parallel_lane_pool(main_window, monkeypatch):
    _prime_three_batch_sets(main_window)
    _select_rows(main_window, [0, 1, 2])
    monkeypatch.setattr("kindred.core.batch_parallel.os.cpu_count", lambda: 8)

    lane_pools: List[_FakeLanePool] = []

    def _factory(max_workers, limit_blas_threads):
        fake = _FakeLanePool()
        lane_pools.append(fake)
        return fake

    monkeypatch.setattr(main_window.simulation_controller.parallel_batch, "max_parallel_workers", 12, raising=True)
    monkeypatch.setattr(main_window.simulation_controller.parallel_batch, "lane_pool_factory", _factory, raising=True)

    _queue_slider_run(main_window)
    _queue_slider_run(main_window)

    assert len(lane_pools) == 1
    assert lane_pools[0].close_calls == []
    main_window.simulation_controller.shutdown_batch_lane_pool(force_terminate=True)
