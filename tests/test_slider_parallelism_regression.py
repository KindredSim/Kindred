from __future__ import annotations

import time
from dataclasses import dataclass, field
from queue import Queue
from typing import Any, Dict, List

import pytest
from PySide6 import QtCore

from kindred.core.batch_containment import BatchLaneOutcome
from kindred.core.batch_parallel import run_batch_simulation_task
from kindred.core.simulation_plan import SimulationPlan

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
                    sid = str(task.get("set_id") or "")
                    sub.complete({"success": True, "run_id": int(task.get("run_id") or 0), "set_id": sid})

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
        if len(_simulation_submissions(lane_pool)) >= int(expected):
            return
        time.sleep(0.005)


def _clear_eager_parallel_pool(main_window) -> None:
    controller = main_window.simulation_controller
    if controller.parallel_batch.has_lane_pool():
        controller.shutdown_batch_lane_pool(force_terminate=True)
    assert not controller.parallel_batch.has_lane_pool()


def test_slider_updates_coalesce_to_one_timer_fire(main_window, monkeypatch):
    main_window._mechanism_editor._reactions_text.setPlainText(
        "reaction: A -> B ; k=0.25\ninitial: A=1.0\ninitial: B=0.0"
    )
    main_window._extract_and_populate_variables()
    calls: list[str] = []
    monkeypatch.setattr(main_window.simulation_controller, "launch_pending_slider_preview_replay", lambda: calls.append("run"))

    main_window._on_slider_drag_started("k1")
    main_window._on_variable_changed("k1", 0.2)
    main_window._on_variable_changed("k1", 0.3)

    assert hasattr(main_window._preview_session, "_variable_update_timer")
    timer = main_window._preview_session._variable_update_timer
    assert timer.isSingleShot() is True

    timer.timeout.emit()
    assert calls == ["run"]


def test_slider_parallel_path_submits_all_selected_sets(main_window, monkeypatch):
    names = _prime_three_batch_sets(main_window)
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

    _wait_for_submission_count(fake, len(names))
    assert len(_simulation_submissions(fake)) == len(names)
    assert bool(main_window.simulation_controller.batch_run_context.get("parallel")) is True
    assert int(main_window.simulation_controller.batch_run_context.get("effective_workers") or 0) > 1
    main_window.simulation_controller.shutdown_batch_lane_pool(force_terminate=True)


def test_slider_parallel_path_uses_per_set_local_mechanism_workspaces(main_window, monkeypatch):
    _prime_three_batch_sets(main_window)
    main_window._extract_and_populate_variables()
    _clear_eager_parallel_pool(main_window)
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

    set0_id = str(main_window._batch_set_id_for_row(0) or "")
    set1_id = str(main_window._batch_set_id_for_row(1) or "")
    set2_id = str(main_window._batch_set_id_for_row(2) or "")
    main_window._preview_session.sync_committed_slider_values({"k1": 0.25})
    main_window._preview_session.stage_slider_value("k1", 0.5, target_set_ids=[set0_id])
    main_window._preview_session.stage_slider_value("k1", 0.75, target_set_ids=[set1_id])

    _queue_slider_run(main_window)

    assert main_window.simulation_controller.parallel_batch.lane_pool_token() == id(fake)
    _wait_for_submission_count(fake, 3)
    submitted_by_set_id = {str(sub.args[0]["set_id"]): dict(sub.args[0]) for sub in _simulation_submissions(fake)}
    assert len(submitted_by_set_id) == 3
    submitted_text_by_set_id = {
        set_id: SimulationPlan.from_payload(task["simulation_plan"]).to_execution_request().mechanism_text
        for set_id, task in submitted_by_set_id.items()
    }
    assert "k=0.5" in submitted_text_by_set_id[set0_id]
    assert "k=0.75" in submitted_text_by_set_id[set1_id]
    assert "k=0.25" in submitted_text_by_set_id[set2_id]
    main_window.simulation_controller.shutdown_batch_lane_pool(force_terminate=True)


def test_slider_parallel_supersession_reuses_lane_pool(main_window, monkeypatch):
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
