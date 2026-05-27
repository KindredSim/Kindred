from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import pytest
from PySide6 import QtCore, QtWidgets

from kindred.core.batch_containment import BatchLaneOutcome
from kindred.core.batch_parallel import run_batch_simulation_task
from kindred.gui.controllers.parallel_batch_outcome import (
    resolve_parallel_batch_outcome,
)
from kindred.gui.controllers.simulation_completion_publication import (
    CompletionCallbackState,
    CompletionResultState,
    SimulationCompletionPublicationDependencies,
)
from kindred.gui.ports import (
    CompletionDisplayEntry,
    DisplayEventKind,
    DisplayStatus,
    DisplayTransitionCause,
    DisplayTransitionOutcome,
    DisplayTransitionOutcomeKind,
    SimulationCompletionDisplayOutcome,
)
from tests.workflow_helpers import completion_provenance_payload

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
        self.ready_lane_count = 999

    def run(self, task, *, run_id: int, request_id: int, set_id: str, active_timeout_s: float):
        from kindred.core.simulation_plan import SimulationPlan

        _ = active_timeout_s
        args = (dict(task or {}),)
        sub = _Submission(fn=run_batch_simulation_task, args=args, kwargs={}, result_placeholder=None)
        self.submissions.append(sub)
        sid = str(task.get("set_id") or task.get("batch_set_id") or set_id or "")
        initials = {}
        plan_payload = task.get("simulation_plan")
        if isinstance(plan_payload, dict):
            request = SimulationPlan.from_payload(plan_payload).to_execution_request().to_payload()
            initials = dict(request.get("initials") or {})
        species_names = [str(name) for name in initials.keys() if str(name)] or ["A"]
        y = np.vstack(
            [
                np.full(2, self.value_marker + float(index), dtype=float)
                for index, _species_name in enumerate(species_names)
            ]
        )
        payload = {
            "run_id": int(task.get("run_id") or run_id or 0),
            "request_id": int(task.get("request_id") or request_id or 0),
            "set_id": sid,
            "set_name": str(task.get("set_name") or sid or "set"),
            "t": np.array([0.0, 1.0]),
            "Y": y,
            "species_names": species_names,
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
            lane_owner_epoch=1,
            success=True,
            payload=payload,
        )

    def warm_lanes(self, required: int, *, wait: bool = True) -> None:
        _ = wait
        self.ready_lane_count = max(int(self.ready_lane_count), int(required))

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


class _PublicationResultsProbe:
    def __init__(self) -> None:
        self.deferred_calls = 0
        self.deferred_calls_kwargs: list[dict[str, Any]] = []
        self.unavailable_calls: list[dict[str, Any]] = []

    def publish_deferred_display_request(self, **kwargs: Any) -> SimulationCompletionDisplayOutcome:
        self.deferred_calls += 1
        self.deferred_calls_kwargs.append(dict(kwargs))
        return SimulationCompletionDisplayOutcome(
            transition_outcome=DisplayTransitionOutcome(
                kind=DisplayTransitionOutcomeKind.DEFERRED,
                active_transaction=None,
                previous_transaction=None,
                display_status=DisplayStatus.DISPLAY_DEFERRED,
                requested_show_set_ids=tuple(kwargs.get("requested_show_set_ids") or ()),
                requested_labels_by_set_id=dict(kwargs.get("requested_labels_by_set_id") or {}),
                affected_set_ids=tuple(kwargs.get("affected_set_ids") or ()),
                unresolved_intent_set_ids=tuple(kwargs.get("unresolved_intent_set_ids") or ()),
                missing_intent_set_ids=tuple(kwargs.get("missing_intent_set_ids") or ()),
                failed_intent_set_ids=tuple(kwargs.get("failed_intent_set_ids") or ()),
                semantic_unavailable_set_ids=tuple(kwargs.get("semantic_unavailable_set_ids") or ()),
                event_kind=DisplayEventKind.SHOW_SCOPE_CHANGED,
                cause=DisplayTransitionCause.QUEUED_DISPLAY,
            )
        )

    def publish_completed_run_display_unavailable(self, **kwargs: Any) -> SimulationCompletionDisplayOutcome:
        self.unavailable_calls.append(dict(kwargs))
        return SimulationCompletionDisplayOutcome(
            transition_outcome=DisplayTransitionOutcome(
                kind=DisplayTransitionOutcomeKind.FAILED,
                active_transaction=None,
                previous_transaction=None,
                display_status=DisplayStatus.NO_COMPLETE_DISPLAYABLE_REQUEST_SCOPE,
                requested_show_set_ids=tuple(kwargs.get("requested_show_set_ids") or ()),
                requested_labels_by_set_id=dict(kwargs.get("requested_labels_by_set_id") or {}),
                affected_set_ids=tuple(kwargs.get("affected_set_ids") or ()),
                unresolved_intent_set_ids=tuple(kwargs.get("unresolved_intent_set_ids") or ()),
                missing_intent_set_ids=tuple(kwargs.get("missing_intent_set_ids") or ()),
                failed_intent_set_ids=tuple(kwargs.get("failed_intent_set_ids") or ()),
                semantic_unavailable_set_ids=tuple(kwargs.get("semantic_unavailable_set_ids") or ()),
                event_kind=DisplayEventKind.COMPLETED_RUN_COVERAGE_UNAVAILABLE,
                cause=kwargs.get("cause"),
            )
        )

    def publish_completed_run_display_transaction(self, transaction: Any) -> SimulationCompletionDisplayOutcome:
        raise AssertionError(f"unexpected completed-run transaction publication: {transaction!r}")

    def publish_direct_completion_result(self, **kwargs: Any) -> SimulationCompletionDisplayOutcome:
        raise AssertionError(f"unexpected direct completion publication: {kwargs!r}")


class _PublicationUiProbe:
    def __init__(self, results: _PublicationResultsProbe) -> None:
        self.results = results


class _RunUiProbe:
    def __init__(self) -> None:
        self.status_texts: list[str] = []
        self.progress_values: list[int] = []
        self.run_enabled: list[bool] = []
        self.stop_enabled: list[bool] = []

    def set_status_text(self, text: str) -> None:
        self.status_texts.append(str(text))

    def set_sim_progress_value(self, value: int) -> None:
        self.progress_values.append(int(value))

    def set_run_button_enabled(self, enabled: bool) -> None:
        self.run_enabled.append(bool(enabled))

    def set_stop_button_enabled(self, enabled: bool) -> None:
        self.stop_enabled.append(bool(enabled))


class _SliderProbe:
    def __init__(self) -> None:
        self.slider_triggered: list[bool] = []

    def set_slider_triggered_simulation(self, value: bool) -> None:
        self.slider_triggered.append(bool(value))


class _ParallelOutcomeUiProbe:
    def __init__(self) -> None:
        self.run_ui = _RunUiProbe()
        self.slider = _SliderProbe()


class _BatchCacheProbe:
    def __init__(self) -> None:
        self.failure_cache_states: list[dict[str, Any]] = []

    def record_explicit_scoped_failure_cache_state(self, **kwargs: Any) -> None:
        self.failure_cache_states.append(dict(kwargs))


def _publication_dependencies_probe() -> SimulationCompletionPublicationDependencies:
    return SimulationCompletionPublicationDependencies(
        apply_lifecycle_effects=lambda **kwargs: None,
        record_nonfatal_exception=lambda source, exc: None,
        queue_slider_plot_update=lambda **kwargs: None,
        finalize_explicit_batch_dirty_reset=lambda **kwargs: {},
        flush_slider_plot_updates=lambda **kwargs: None,
        show_scoped_batch_failure_summary=lambda **kwargs: None,
        has_deferred_preview_replay_intent=lambda: False,
        start_next_batch_simulation=lambda: None,
        clear_pending_progress_status=lambda: None,
    )


def _completion_state_for_publication(*, set_id: str, ctx: dict[str, Any]) -> CompletionCallbackState:
    return CompletionCallbackState(
        run_id=41,
        request_id=42,
        batch_set=f"Set {set_id[-1].upper()}",
        batch_set_id=set_id,
        cache_key="publication-in-flight",
        policy_context=None,
        ctx=ctx,
        shutdown_requested=False,
        is_preview=False,
        slider_triggered=False,
        explicit_batch_coalescing=False,
    )


def _completion_result_for_publication() -> CompletionResultState:
    return CompletionResultState(
        t=np.array([0.0, 1.0]),
        Y=np.array([[1.0, 0.5]]),
        species_names=("A",),
        algebra_scalars={},
        algebra_errors=(),
        solver_provenance={},
        mechanism=None,
        base_species_count=1,
        mechanism_text="reaction: A -> B ; k=0.1",
        solver_config={},
        warnings=(),
        fallback_occurred=False,
        fallback_message=None,
        series={"A": np.array([1.0, 0.5])},
        is_primary=True,
        energy_mode=False,
        redraw_valid_set_ids=None,
        has_redraw_subset=False,
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


def _prime_three_m1_batch_sets(main_window) -> list[str]:
    main_window._load_preset_mechanism("M1")
    QtWidgets.QApplication.processEvents()
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
        main_window._batch_store.set_value(int(row), "B", "0")
    return names[:3]


def _simulation_submissions(lane_pool: _FakeLanePool) -> list[_Submission]:
    return [sub for sub in lane_pool.submissions if sub.fn is run_batch_simulation_task]


def _wait_for_submission_count(lane_pool: _FakeLanePool, expected: int, timeout_s: float = 1.0) -> None:
    deadline = time.monotonic() + float(timeout_s)
    while time.monotonic() < deadline:
        if len(_simulation_submissions(lane_pool)) >= int(expected):
            return
        time.sleep(0.005)


def _slider_handle_center(slider: QtWidgets.QSlider) -> QtCore.QPoint:
    option = QtWidgets.QStyleOptionSlider()
    slider.initStyleOption(option)
    handle = slider.style().subControlRect(
        QtWidgets.QStyle.CC_Slider,
        option,
        QtWidgets.QStyle.SC_SliderHandle,
        slider,
    )
    return handle.center()



def _completion_entry(*, set_id: str, label: str, values: tuple[float, float]) -> CompletionDisplayEntry:
    t = np.asarray([0.0, 1.0], dtype=float)
    series = {"A": np.asarray(values, dtype=float)}
    return CompletionDisplayEntry(
        set_id=set_id,
        label=label,
        t=t,
        series=series,
        algebra_scalars={},
        solver_provenance={},
        mechanism_text="reaction: A -> B ; k=0.1",
        solver_config={},
        warnings=(),
        completion_provenance=completion_provenance_payload(
            t=t,
            series=series,
            mechanism_text="reaction: A -> B ; k=0.1",
        ),
        owned_species=("A",),
    )

def test_fast_parallel_preview_accepts_lane_epoch_distinct_from_preview_epoch():
    for set_id in ("id1", "id2"):
        outcome = BatchLaneOutcome(
            lane_id=f"lane-{set_id}",
            run_id=7,
            request_id=11,
            set_id=set_id,
            lane_owner_epoch=1,
            success=True,
            payload={"success": True, "set_id": set_id},
        )

        resolution = resolve_parallel_batch_outcome(
            set_id=set_id,
            outcome=outcome,
            metadata={
                "run_id": 7,
                "request_id": 11,
                "set_name": set_id,
                "preview_owner_epoch": 2,
            },
        )

        assert resolution.stale is False
        assert resolution.preview_owner_epoch == 2
        assert resolution.payload == {"success": True, "set_id": set_id}



def test_multi_set_species_slider_parallel_preview_completes_with_distinct_lane_epoch(
    main_window,
    monkeypatch,
    qtbot,
):
    _prime_three_m1_batch_sets(main_window)
    set_ids = [str(main_window.batch_set_id_for_row(row) or "") for row in (0, 1)]
    assert all(set_ids)
    _select_rows(main_window, [0, 1])
    main_window.set_slider_edit_target_set_ids(set_ids)
    main_window._on_slider_edit_targets_changed()

    fake = _FakeLanePool(done_immediately=True, value_marker=3.0)
    monkeypatch.setattr(main_window.simulation_controller.parallel_batch, "max_parallel_workers", 12, raising=True)
    monkeypatch.setattr(
        main_window.simulation_controller.parallel_batch,
        "lane_pool_factory",
        lambda max_workers, limit_blas_threads: fake,
        raising=True,
    )
    main_window.simulation_controller.parallel_batch.ensure_lane_pool(max_lanes=2)
    main_window._set_runtime_backed_controls_ready(True)
    main_window.simulation_controller._claim_preview_ownership(
        request_id=1,
        target_set_ids=set_ids,
    )

    qtbot.addWidget(main_window)
    main_window.show()
    QtWidgets.QApplication.processEvents()

    slider_a = main_window.findChild(QtWidgets.QSlider, "speciesSlider_A")
    assert slider_a is not None
    press_pos = _slider_handle_center(slider_a)
    qtbot.mousePress(slider_a, QtCore.Qt.LeftButton, pos=press_pos)
    slider_a.setValue(5000)
    qtbot.mouseRelease(slider_a, QtCore.Qt.LeftButton, pos=press_pos)
    QtWidgets.QApplication.processEvents()

    staged_row_0 = main_window._preview_session.preview_initials_for_row(0, main_window.batch_initials_for_row(0))
    staged_row_1 = main_window._preview_session.preview_initials_for_row(1, main_window.batch_initials_for_row(1))
    assert float(staged_row_0["A"]) == pytest.approx(2.5, rel=1e-6, abs=1e-9)
    assert float(staged_row_1["A"]) == pytest.approx(2.5, rel=1e-6, abs=1e-9)

    qtbot.waitUntil(lambda: len(_simulation_submissions(fake)) >= 2, timeout=3000)
    qtbot.waitUntil(
        lambda: (
            not main_window.simulation_controller.simulation_running
            and not bool(main_window.simulation_controller._slider_simulation_active)
            and main_window.simulation_controller._batch_context_owner.active_batch_state() is None
        ),
        timeout=3000,
    )

    assert main_window.simulation_controller._preview_ownership.epoch == 2
    assert str(main_window.simulation_controller._last_nonfatal_exception or "").find("stale batch lane outcome") == -1
    assert len(_simulation_submissions(fake)) == 2



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


def _fit_runtime_publisher(main_window):
    publisher = getattr(main_window, "fitting_runtime_input_publisher", None)
    if publisher is None:
        publisher = getattr(main_window, "_fitting_runtime_input_publisher", None)
    assert publisher is not None
    return publisher


def _runtime_inputs_payload(runtime_inputs) -> dict[str, object]:
    evaluator = runtime_inputs.evaluator
    return {
        "evaluator": {
            "temperature_K": float(evaluator.temperature_K),
            "use_sparse_jacobian": bool(evaluator.use_sparse_jacobian),
            "wegscheider_cyclicity_enabled": bool(evaluator.wegscheider_cyclicity_enabled),
        },
        "batch_runtime_lane_budget": int(runtime_inputs.batch_runtime_lane_budget),
        "lane_count_for_two_datasets": int(runtime_inputs.lane_count_for_dataset_count(2)),
    }


def test_open_solver_settings_publishes_typed_fit_runtime_inputs_for_runtime_changes(main_window, monkeypatch):
    class _FakeDialog:
        update: dict[str, object] = {}

        def __init__(self, _parent, *, cache_port=None):
            self._settings = {}
            self._cache_port = cache_port

        def set_settings(self, settings):
            self._settings = dict(settings or {})

        def exec(self):
            return True

        def get_settings(self):
            updated = dict(self._settings)
            updated.update(type(self).update)
            return updated

    notifications: list[dict[str, object]] = []

    class _FitWindow:
        def apply_runtime_inputs(self, runtime_inputs, **_kwargs) -> None:
            notifications.append(_runtime_inputs_payload(runtime_inputs))

        def close(self) -> bool:
            return True

    monkeypatch.setattr(
        "kindred.gui.widgets.solver_settings.SolverSettingsDialog",
        _FakeDialog,
    )
    _fit_runtime_publisher(main_window).register_window(_FitWindow())

    original_sparse = bool(main_window._use_sparse_jacobian)
    original_wegscheider = bool(main_window._wegscheider_cyclicity_enabled)
    original_lane_budget = int(main_window.simulation_controller.batch_runtime_lane_budget)

    _FakeDialog.update = {"use_sparse_jacobian": not original_sparse}
    main_window._open_solver_settings()
    _FakeDialog.update = {"wegscheider_cyclicity_enabled": not original_wegscheider}
    main_window._open_solver_settings()
    _FakeDialog.update = {"batch_runtime_lane_budget": original_lane_budget + 1}
    main_window._open_solver_settings()

    assert len(notifications) == 3
    assert notifications[0]["evaluator"] == {
        "temperature_K": float(main_window._temperature_spinbox.value()),
        "use_sparse_jacobian": (not original_sparse),
        "wegscheider_cyclicity_enabled": original_wegscheider,
    }
    assert notifications[1]["evaluator"] == {
        "temperature_K": float(main_window._temperature_spinbox.value()),
        "use_sparse_jacobian": (not original_sparse),
        "wegscheider_cyclicity_enabled": (not original_wegscheider),
    }
    assert notifications[2]["evaluator"] == notifications[1]["evaluator"]
    assert notifications[2]["batch_runtime_lane_budget"] == original_lane_budget + 1
    assert set(notifications[2]["evaluator"]) == {
        "temperature_K",
        "use_sparse_jacobian",
        "wegscheider_cyclicity_enabled",
    }
    assert notifications[2]["lane_count_for_two_datasets"] >= 1


def test_open_solver_settings_does_not_notify_fit_windows_for_local_fit_integration_defaults(main_window, monkeypatch):
    class _FakeDialog:
        update: dict[str, object] = {}

        def __init__(self, _parent, *, cache_port=None):
            self._settings = {}
            self._cache_port = cache_port

        def set_settings(self, settings):
            self._settings = dict(settings or {})

        def exec(self):
            return True

        def get_settings(self):
            updated = dict(self._settings)
            updated.update(type(self).update)
            return updated

    notifications: list[str] = []

    class _FitWindow:
        def apply_runtime_inputs(self, runtime_inputs, **_kwargs) -> None:
            _ = runtime_inputs
            notifications.append("notified")

        def close(self) -> bool:
            return True

    monkeypatch.setattr(
        "kindred.gui.widgets.solver_settings.SolverSettingsDialog",
        _FakeDialog,
    )
    _fit_runtime_publisher(main_window).register_window(_FitWindow())

    unrelated_updates = [
        {"solver": "Radau"},
        {"rtol": 2e-6},
        {"atol": 2e-12},
        {"slider_preview_solver": "Radau"},
        {"slider_preview_points": 375},
        {"parameter_preview_debounce_ms": 250},
        {"equilibrium_preview_debounce_ms": 250},
        {"limit_blas_threads_per_worker": False},
        {"result_cache_cap": 123},
        {"preview_cache_cap": 45},
        {
            "max_parallel_batch_workers": int(
                main_window.simulation_controller.parallel_batch.max_parallel_workers
            )
            + 1,
        },
    ]
    for update in unrelated_updates:
        _FakeDialog.update = update
        main_window._open_solver_settings()

    assert notifications == []


def test_active_fit_window_runtime_notification_continues_after_stale_window_error(main_window):
    notifications: list[dict[str, object]] = []

    class _BadFitWindow:
        def apply_runtime_inputs(self, runtime_inputs, **_kwargs) -> None:
            _ = runtime_inputs
            raise RuntimeError("deleted wrapper")

        def close(self) -> bool:
            return True

    class _GoodFitWindow:
        def apply_runtime_inputs(self, runtime_inputs, **_kwargs) -> None:
            notifications.append(_runtime_inputs_payload(runtime_inputs))

        def close(self) -> bool:
            return True

    publisher = _fit_runtime_publisher(main_window)
    publisher.register_window(_BadFitWindow())
    publisher.register_window(_GoodFitWindow())

    publisher.publish_current(reason="test forced active-window notification", force=True)

    assert len(notifications) == 1
    assert set(notifications[0]["evaluator"]) == {
        "temperature_K",
        "use_sparse_jacobian",
        "wegscheider_cyclicity_enabled",
    }
    recorded_failures = getattr(main_window, "_fitting_best_effort_failures", set())
    assert any("runtime" in str(key) for key in recorded_failures)


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
        "equilibrium: A <-> B ; kf=0.0928966, Keq=0.00963829\n"
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
