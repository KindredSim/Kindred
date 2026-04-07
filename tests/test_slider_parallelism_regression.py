from __future__ import annotations

from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any, Dict, List

import pytest
from PySide6 import QtCore

from kindred.core.batch_parallel import run_batch_simulation_task

pytestmark = [pytest.mark.gui]


@dataclass
class _Submission:
    fn: Any
    args: tuple[Any, ...]
    kwargs: Dict[str, Any]
    future: Future


class _FakeExecutor:
    def __init__(self) -> None:
        self.submissions: List[_Submission] = []
        self.shutdown_calls: List[Dict[str, Any]] = []

    def submit(self, fn, *args, **kwargs):
        fut: Future = Future()
        self.submissions.append(_Submission(fn=fn, args=args, kwargs=dict(kwargs), future=fut))
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
    main_window.simulation_controller.run_state.pending_slider_sim_request_id = (
        main_window.simulation_controller.next_sim_request_id()
    )
    main_window.simulation_controller.run_state.pending_slider_simulation = True
    main_window.simulation_controller.run_simulation_from_slider()


def _simulation_submissions(executor: _FakeExecutor) -> list[_Submission]:
    return [sub for sub in executor.submissions if sub.fn is run_batch_simulation_task]


def test_slider_updates_coalesce_to_one_timer_fire(main_window, monkeypatch):
    main_window._mechanism_editor._reactions_text.setPlainText(
        "reaction: A -> B ; k=0.25\ninitial: A=1.0\ninitial: B=0.0"
    )
    main_window._extract_and_populate_variables()
    calls: list[str] = []
    monkeypatch.setattr(main_window.simulation_controller, "run_simulation_from_slider", lambda: calls.append("run"))

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

    fake = _FakeExecutor()
    monkeypatch.setattr(main_window.simulation_controller.parallel_batch, "max_parallel_workers", 12, raising=True)
    monkeypatch.setattr(
        main_window.simulation_controller.parallel_batch,
        "executor_factory",
        lambda max_workers, limit_blas_threads: fake,
        raising=True,
    )

    _queue_slider_run(main_window)

    assert len(_simulation_submissions(fake)) == len(names)
    assert bool(main_window.simulation_controller.batch_run_context.get("parallel")) is True
    assert int(main_window.simulation_controller.batch_run_context.get("effective_workers") or 0) > 1
    main_window.simulation_controller.shutdown_batch_executor(force_terminate=True)


def test_slider_parallel_path_uses_per_set_local_mechanism_workspaces(main_window, monkeypatch):
    _prime_three_batch_sets(main_window)
    main_window._extract_and_populate_variables()
    _select_rows(main_window, [0, 1, 2])
    monkeypatch.setattr("kindred.core.batch_parallel.os.cpu_count", lambda: 8)

    fake = _FakeExecutor()
    monkeypatch.setattr(main_window.simulation_controller.parallel_batch, "max_parallel_workers", 12, raising=True)
    monkeypatch.setattr(
        main_window.simulation_controller.parallel_batch,
        "executor_factory",
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

    submitted_by_set_id = {str(sub.args[0]["set_id"]): dict(sub.args[0]) for sub in _simulation_submissions(fake)}
    assert len(submitted_by_set_id) == 3
    assert "k=0.5" in str(submitted_by_set_id[set0_id]["mechanism_text"])
    assert "k=0.75" in str(submitted_by_set_id[set1_id]["mechanism_text"])
    assert "k=0.25" in str(submitted_by_set_id[set2_id]["mechanism_text"])
    main_window.simulation_controller.shutdown_batch_executor(force_terminate=True)


def test_slider_parallel_supersession_reuses_executor(main_window, monkeypatch):
    _prime_three_batch_sets(main_window)
    _select_rows(main_window, [0, 1, 2])
    monkeypatch.setattr("kindred.core.batch_parallel.os.cpu_count", lambda: 8)

    executors: List[_FakeExecutor] = []

    def _factory(max_workers, limit_blas_threads):
        fake = _FakeExecutor()
        executors.append(fake)
        return fake

    monkeypatch.setattr(main_window.simulation_controller.parallel_batch, "max_parallel_workers", 12, raising=True)
    monkeypatch.setattr(main_window.simulation_controller.parallel_batch, "executor_factory", _factory, raising=True)

    _queue_slider_run(main_window)
    _queue_slider_run(main_window)

    assert len(executors) == 1
    assert not executors[0].shutdown_calls
    main_window.simulation_controller.shutdown_batch_executor(force_terminate=True)
