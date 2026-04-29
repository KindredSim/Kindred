from __future__ import annotations

import threading
from typing import Any

import pytest

from kindred.core.batch_containment import BatchLaneOutcome, WarmBatchSimulationLane
from kindred.core.runtime_defaults import MAX_PARALLEL_WORKERS_CEILING
from kindred.gui.controllers.parallel_batch_executor import ParallelBatchExecutor


@pytest.mark.unit
def test_lane_pool_factory_change_marks_existing_pool_stale() -> None:
    class _FakeLanePool:
        def __init__(self, label: str) -> None:
            self.label = str(label)
            self.close_calls: list[bool] = []

        def close(self, *, kill: bool = False) -> None:
            self.close_calls.append(bool(kill))

    created: list[_FakeLanePool] = []

    def _factory_a(_max_lanes: int, _limit_blas_threads: bool) -> _FakeLanePool:
        pool = _FakeLanePool("a")
        created.append(pool)
        return pool

    def _factory_b(_max_lanes: int, _limit_blas_threads: bool) -> _FakeLanePool:
        pool = _FakeLanePool("b")
        created.append(pool)
        return pool

    batch = ParallelBatchExecutor(lane_pool_factory=_factory_a)
    first = batch.ensure_lane_pool(max_lanes=2)

    batch.lane_pool_factory = _factory_b

    assert batch.is_pool_stale is True
    second = batch.ensure_lane_pool(max_lanes=2)
    assert first.close_calls == [False]
    assert second is created[-1]
    assert second.label == "b"
    assert batch.is_pool_stale is False


@pytest.mark.unit
def test_submit_task_uses_lane_request_handle_not_future_facade() -> None:
    class _FakeLanePool:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def run(self, task, *, run_id: int, request_id: int, set_id: str, active_timeout_s: float):
            self.calls.append(
                {
                    "task": dict(task),
                    "run_id": int(run_id),
                    "request_id": int(request_id),
                    "set_id": str(set_id),
                    "active_timeout_s": float(active_timeout_s),
                }
            )
            return BatchLaneOutcome(
                lane_id="lane-a",
                run_id=run_id,
                request_id=request_id,
                set_id=set_id,
                owner_epoch=3,
                success=True,
                payload={"ok": True},
            )

        def close(self, *, kill: bool = False) -> None:
            return None

    created: list[_FakeLanePool] = []

    def _factory(max_lanes: int, limit_blas_threads: bool) -> _FakeLanePool:
        assert max_lanes == 2
        assert limit_blas_threads is True
        pool = _FakeLanePool()
        created.append(pool)
        return pool

    batch = ParallelBatchExecutor(lane_pool_factory=_factory)
    batch.ensure_lane_pool(max_lanes=2)
    batch.begin_run(
        run_id=11,
        request_id=22,
        fast_mode=False,
        queue_ids=["set-a"],
        queue_names=["Set A"],
        keep_lane_pool_alive=False,
        preview_owner_epoch=5,
        active_timeout_s=1.5,
    )

    handle = batch.submit_task(
        {"value": 1},
        set_id="set-a",
        set_name="Set A",
    )
    handle.join(timeout=2.0)

    assert not hasattr(handle, "cancel")
    assert not hasattr(handle, "result")
    assert created[0].calls == [
        {
            "task": {"value": 1},
            "run_id": 11,
            "request_id": 22,
            "set_id": "set-a",
            "active_timeout_s": 1.5,
        }
    ]

    polled = batch.poll_completed_records()
    assert [item.set_id for item in polled] == ["set-a"]
    outcome = polled[0].record.outcome
    assert outcome.success is True
    assert outcome.owner_epoch == 3
    assert outcome.payload == {"ok": True}
    assert not batch.has_active_requests()


@pytest.mark.unit
def test_soft_supersede_invalidates_lane_requests_without_retained_superseded_bookkeeping() -> None:
    release = threading.Event()

    class _SlowLanePool:
        def __init__(self) -> None:
            self.close_calls: list[bool] = []

        def run(self, task, *, run_id: int, request_id: int, set_id: str, active_timeout_s: float):
            release.wait(timeout=2.0)
            return BatchLaneOutcome(
                lane_id="lane-a",
                run_id=run_id,
                request_id=request_id,
                set_id=set_id,
                owner_epoch=1,
                success=False,
                failure={"kind": "cancelled", "phase": "internal", "message": "superseded"},
            )

        def close(self, *, kill: bool = False) -> None:
            self.close_calls.append(bool(kill))
            release.set()

    pool = _SlowLanePool()
    batch = ParallelBatchExecutor(lane_pool_factory=lambda _max_lanes, _limit_blas: pool)
    batch.ensure_lane_pool(max_lanes=1)
    batch.begin_run(
        run_id=1,
        request_id=2,
        fast_mode=True,
        queue_ids=["set-a"],
        queue_names=["Set A"],
        keep_lane_pool_alive=True,
        preview_owner_epoch=None,
        active_timeout_s=5.0,
    )
    handle = batch.submit_task(
        {"value": 1},
        set_id="set-a",
        set_name="Set A",
    )

    cancelled, running = batch.soft_supersede()
    handle.join(timeout=2.0)

    assert cancelled == 0
    assert running == 1
    assert not batch.has_active_requests()
    assert batch.has_lane_pool()
    assert pool.close_calls == []
    assert batch.poll_completed_records() == []


@pytest.mark.unit
def test_request_workers_remain_tracked_across_soft_supersede() -> None:
    started = 0
    started_lock = threading.Lock()
    two_started = threading.Event()
    release = threading.Event()

    class _BlockingLanePool:
        def __init__(self) -> None:
            self.close_calls: list[bool] = []

        def run(self, task, *, run_id: int, request_id: int, set_id: str, active_timeout_s: float):
            nonlocal started
            with started_lock:
                started += 1
                if started == 2:
                    two_started.set()
            release.wait(timeout=2.0)
            return BatchLaneOutcome(
                lane_id="lane",
                run_id=run_id,
                request_id=request_id,
                set_id=set_id,
                owner_epoch=1,
                success=True,
                payload={"set_id": set_id},
            )

        def close(self, *, kill: bool = False) -> None:
            self.close_calls.append(bool(kill))
            release.set()

    pool = _BlockingLanePool()
    batch = ParallelBatchExecutor(lane_pool_factory=lambda _max_lanes, _limit_blas: pool)
    batch.ensure_lane_pool(max_lanes=2)
    batch.begin_run(
        run_id=1,
        request_id=2,
        fast_mode=True,
        queue_ids=[f"set-{index}" for index in range(5)],
        queue_names=[f"Set {index}" for index in range(5)],
        keep_lane_pool_alive=True,
        preview_owner_epoch=None,
        active_timeout_s=5.0,
    )

    handles = [
        batch.submit_task(
            {"value": index},
            set_id=f"set-{index}",
            set_name=f"Set {index}",
        )
        for index in range(5)
    ]
    assert two_started.wait(timeout=1.0)
    with started_lock:
        assert started == 2

    worker_count = batch.request_worker_count()
    cancelled, running = batch.soft_supersede()
    assert batch.has_active_requests()
    assert batch.active_request_count() == len(handles)

    release.set()
    for handle in handles:
        handle.join(timeout=1.0)

    assert cancelled == 0
    assert running == 5
    assert pool.close_calls == []
    assert batch.has_lane_pool()
    assert not batch.has_active_requests()
    assert batch.active_request_count() == 0
    assert batch.request_worker_count() == worker_count
    with started_lock:
        assert started == 2

    batch.shutdown(force_terminate=True, record_nonfatal_exception=lambda _msg, _exc: None)
    assert batch.request_worker_count() == 0


@pytest.mark.unit
def test_nonblocking_warm_growth_is_waited_by_existing_background_warm_thread() -> None:
    wait_started = threading.Event()
    release_wait = threading.Event()
    warm_calls: list[tuple[int, bool]] = []

    class _WarmLanePool:
        def warm_lanes(self, max_lanes: int, *, wait: bool = True) -> None:
            warm_calls.append((int(max_lanes), bool(wait)))
            if bool(wait):
                wait_started.set()
                release_wait.wait(timeout=2.0)

        def close(self, *, kill: bool = False) -> None:
            _ = kill
            release_wait.set()

    pool = _WarmLanePool()
    batch = ParallelBatchExecutor(lane_pool_factory=lambda _max_lanes, _limit_blas: pool)

    batch.ensure_warm_lane_pool(max_lanes=1, wait=False)
    assert wait_started.wait(timeout=1.0)
    batch.ensure_warm_lane_pool(max_lanes=2, wait=False)
    release_wait.set()

    for _ in range(100):
        if (2, True) in warm_calls:
            break
        threading.Event().wait(timeout=0.01)

    assert warm_calls[:3] == [(1, False), (1, True), (2, False)]
    assert (2, True) in warm_calls


@pytest.mark.unit
def test_repeated_soft_supersede_tracks_same_set_inflight_requests_independently() -> None:
    started = 0
    started_lock = threading.Lock()
    first_started = threading.Event()
    second_started = threading.Event()
    release = threading.Event()

    class _BlockingLanePool:
        def run(self, task, *, run_id: int, request_id: int, set_id: str, active_timeout_s: float):
            nonlocal started
            with started_lock:
                started += 1
                if started == 1:
                    first_started.set()
                if started == 2:
                    second_started.set()
            release.wait(timeout=2.0)
            return BatchLaneOutcome(
                lane_id=f"lane-{run_id}",
                run_id=run_id,
                request_id=request_id,
                set_id=set_id,
                owner_epoch=1,
                success=True,
                payload={"run_id": run_id, "set_id": set_id},
            )

        def close(self, *, kill: bool = False) -> None:
            release.set()

    pool = _BlockingLanePool()
    batch = ParallelBatchExecutor(lane_pool_factory=lambda _max_lanes, _limit_blas: pool)
    batch.ensure_lane_pool(max_lanes=2)

    batch.begin_run(
        run_id=1,
        request_id=10,
        fast_mode=True,
        queue_ids=["set-a"],
        queue_names=["Set A"],
        keep_lane_pool_alive=True,
        preview_owner_epoch=None,
        active_timeout_s=5.0,
    )
    first = batch.submit_task({"value": 1}, set_id="set-a", set_name="Set A")
    assert first_started.wait(timeout=1.0)
    batch.soft_supersede()

    batch.begin_run(
        run_id=2,
        request_id=20,
        fast_mode=True,
        queue_ids=["set-a"],
        queue_names=["Set A"],
        keep_lane_pool_alive=True,
        preview_owner_epoch=None,
        active_timeout_s=5.0,
    )
    second = batch.submit_task({"value": 2}, set_id="set-a", set_name="Set A")
    assert second_started.wait(timeout=1.0)

    batch.soft_supersede()

    assert batch.has_active_requests()
    assert batch.active_request_count() == 2

    release.set()
    first.join(timeout=1.0)
    second.join(timeout=1.0)

    assert not batch.has_active_requests()
    assert batch.active_request_count() == 0


@pytest.mark.unit
def test_ensure_lane_pool_recreates_pool_when_worker_count_increases() -> None:
    created: list[tuple[int, bool, Any]] = []

    class _FakeLanePool:
        def __init__(self, max_lanes: int) -> None:
            self.max_lanes = int(max_lanes)
            self.close_calls: list[bool] = []

        def close(self, *, kill: bool = False) -> None:
            self.close_calls.append(bool(kill))

    def _factory(max_lanes: int, limit_blas_threads: bool) -> _FakeLanePool:
        pool = _FakeLanePool(max_lanes=max_lanes)
        created.append((int(max_lanes), bool(limit_blas_threads), pool))
        return pool

    batch = ParallelBatchExecutor(lane_pool_factory=_factory)

    first = batch.ensure_lane_pool(max_lanes=2)
    second = batch.ensure_lane_pool(max_lanes=6)

    assert [item[:2] for item in created] == [(2, True), (6, True)]
    assert second is not first
    assert first.close_calls == [True]
    assert batch.lane_pool_token() == id(second)


@pytest.mark.unit
def test_ensure_lane_pool_does_not_resize_downward() -> None:
    created: list[tuple[int, bool]] = []

    def _factory(max_lanes: int, limit_blas_threads: bool) -> object:
        created.append((int(max_lanes), bool(limit_blas_threads)))
        return object()

    batch = ParallelBatchExecutor(lane_pool_factory=_factory)

    first = batch.ensure_lane_pool(max_lanes=6)
    second = batch.ensure_lane_pool(max_lanes=4)

    assert created == [(6, True)]
    assert second is first


@pytest.mark.unit
def test_ensure_lane_pool_does_not_resize_when_worker_count_matches() -> None:
    created: list[tuple[int, bool]] = []

    def _factory(max_lanes: int, limit_blas_threads: bool) -> object:
        created.append((int(max_lanes), bool(limit_blas_threads)))
        return object()

    batch = ParallelBatchExecutor(lane_pool_factory=_factory)

    first = batch.ensure_lane_pool(max_lanes=6)
    second = batch.ensure_lane_pool(max_lanes=6)

    assert created == [(6, True)]
    assert second is first


@pytest.mark.unit
def test_ensure_lane_pool_recreates_stale_pool_even_when_worker_count_matches() -> None:
    created: list[Any] = []

    class _FakeLanePool:
        def __init__(self, label: str) -> None:
            self.label = str(label)
            self.close_calls: list[bool] = []

        def close(self, *, kill: bool = False) -> None:
            self.close_calls.append(bool(kill))

    def _factory(max_lanes: int, limit_blas_threads: bool) -> _FakeLanePool:
        pool = _FakeLanePool(label=f"pool-{len(created) + 1}-w{int(max_lanes)}")
        created.append((int(max_lanes), bool(limit_blas_threads), pool))
        return pool

    batch = ParallelBatchExecutor(lane_pool_factory=_factory)

    first = batch.ensure_lane_pool(max_lanes=4)
    batch.mark_pool_stale()
    second = batch.ensure_lane_pool(max_lanes=4)

    assert second is not first
    assert first.close_calls == [False]
    assert [item[:2] for item in created] == [(4, True), (4, True)]
    assert batch.is_pool_stale is False


@pytest.mark.unit
def test_ensure_lane_pool_force_closes_stale_pool_with_active_requests_even_without_resize() -> None:
    release = threading.Event()

    class _BlockingLanePool:
        def __init__(self, label: str) -> None:
            self.label = str(label)
            self.close_calls: list[bool] = []

        def run(self, task, *, run_id: int, request_id: int, set_id: str, active_timeout_s: float):
            release.wait(timeout=2.0)
            return BatchLaneOutcome(
                lane_id=self.label,
                run_id=run_id,
                request_id=request_id,
                set_id=set_id,
                owner_epoch=1,
                success=True,
                payload={"set_id": set_id},
            )

        def close(self, *, kill: bool = False) -> None:
            self.close_calls.append(bool(kill))
            if bool(kill):
                release.set()

    pools: list[_BlockingLanePool] = []

    def _factory(max_lanes: int, limit_blas_threads: bool) -> _BlockingLanePool:
        pool = _BlockingLanePool(f"pool-{len(pools) + 1}")
        pools.append(pool)
        return pool

    batch = ParallelBatchExecutor(lane_pool_factory=_factory)
    first_pool = batch.ensure_lane_pool(max_lanes=2)
    batch.begin_run(
        run_id=1,
        request_id=2,
        fast_mode=True,
        queue_ids=["set-a"],
        queue_names=["Set A"],
        keep_lane_pool_alive=True,
        preview_owner_epoch=None,
        active_timeout_s=5.0,
    )
    handle = batch.submit_task({"value": 1}, set_id="set-a", set_name="Set A")
    batch.mark_pool_stale()

    second_pool = batch.ensure_lane_pool(max_lanes=2)

    assert second_pool is not first_pool
    assert first_pool.close_calls == [True]
    handle.join(timeout=1.0)
    batch.shutdown(force_terminate=True, record_nonfatal_exception=lambda _msg, _exc: None)


@pytest.mark.unit
def test_shutdown_recreate_keeps_unjoined_old_request_workers_counted() -> None:
    release = threading.Event()
    first_started = threading.Event()

    class _BlockingLanePool:
        def __init__(self, label: str) -> None:
            self.label = str(label)
            self.close_calls: list[bool] = []

        def run(self, task, *, run_id: int, request_id: int, set_id: str, active_timeout_s: float):
            first_started.set()
            release.wait(timeout=2.0)
            return BatchLaneOutcome(
                lane_id=self.label,
                run_id=run_id,
                request_id=request_id,
                set_id=set_id,
                owner_epoch=1,
                success=True,
                payload={"set_id": set_id},
            )

        def close(self, *, kill: bool = False) -> None:
            self.close_calls.append(bool(kill))

    pools: list[_BlockingLanePool] = []

    def _factory(max_lanes: int, limit_blas_threads: bool) -> _BlockingLanePool:
        pool = _BlockingLanePool(f"pool-{len(pools) + 1}")
        pools.append(pool)
        return pool

    batch = ParallelBatchExecutor(lane_pool_factory=_factory)
    batch.ensure_lane_pool(max_lanes=1)
    batch.begin_run(
        run_id=1,
        request_id=2,
        fast_mode=True,
        queue_ids=["set-a"],
        queue_names=["Set A"],
        keep_lane_pool_alive=True,
        preview_owner_epoch=None,
        active_timeout_s=5.0,
    )
    handle = batch.submit_task({"value": 1}, set_id="set-a", set_name="Set A")
    assert first_started.wait(timeout=1.0)

    batch.shutdown(force_terminate=False, record_nonfatal_exception=lambda _msg, _exc: None)
    batch.ensure_lane_pool(max_lanes=1)

    assert batch.request_worker_count() == 2

    release.set()
    handle.join(timeout=1.0)
    batch.shutdown(force_terminate=True, record_nonfatal_exception=lambda _msg, _exc: None)
    assert batch.request_worker_count() == 0


@pytest.mark.unit
def test_parallel_batch_adapter_does_not_expose_executor_or_future_compatibility_aliases() -> None:
    batch = ParallelBatchExecutor(lane_pool_factory=lambda _max_lanes, _limit_blas_threads: object())

    assert not hasattr(batch, "executor")
    assert not hasattr(batch, "executor_factory")
    assert not hasattr(batch, "ensure_executor")
    assert not hasattr(batch, "_terminate_processes_best_effort")


@pytest.mark.unit
def test_batch_runtime_owner_does_not_expose_legacy_future_outcome_bridge() -> None:
    batch = ParallelBatchExecutor(lane_pool_factory=lambda _max_lanes, _limit_blas_threads: object())

    assert not hasattr(batch._runtime_owner, "pop_completed_outcome")
    assert not hasattr(batch._runtime_owner, "_record_from_legacy_outcome")


@pytest.mark.unit
def test_ensure_lane_pool_tracks_current_max_workers_without_pool_introspection() -> None:
    created: list[tuple[int, bool]] = []

    def _factory(max_lanes: int, limit_blas_threads: bool) -> object:
        created.append((int(max_lanes), bool(limit_blas_threads)))
        return object()

    batch = ParallelBatchExecutor(lane_pool_factory=_factory)

    first = batch.ensure_lane_pool(max_lanes=2)
    second = batch.ensure_lane_pool(max_lanes=6)

    assert created == [(2, True), (6, True)]
    assert batch.current_max_workers == 6
    assert second is not first


@pytest.mark.unit
def test_ensure_lane_pool_caps_worker_count_at_shared_ceiling() -> None:
    created: list[tuple[int, bool]] = []

    def _factory(max_lanes: int, limit_blas_threads: bool) -> object:
        created.append((int(max_lanes), bool(limit_blas_threads)))
        return object()

    batch = ParallelBatchExecutor(lane_pool_factory=_factory)

    pool = batch.ensure_lane_pool(max_lanes=200)

    assert created == [(int(MAX_PARALLEL_WORKERS_CEILING), True)]
    assert batch.current_max_workers == int(MAX_PARALLEL_WORKERS_CEILING)
    assert batch.lane_pool_token() == id(pool)


@pytest.mark.unit
def test_ensure_lane_pool_resize_factory_failure_leaves_wrapper_consistent() -> None:
    created: list[Any] = []

    class _FakeLanePool:
        def __init__(self) -> None:
            self.close_calls: list[bool] = []

        def close(self, *, kill: bool = False) -> None:
            self.close_calls.append(bool(kill))

    def _factory(max_lanes: int, _limit_blas_threads: bool) -> _FakeLanePool:
        if int(max_lanes) == 6:
            raise RuntimeError("factory boom")
        pool = _FakeLanePool()
        created.append(pool)
        return pool

    batch = ParallelBatchExecutor(lane_pool_factory=_factory)
    first = batch.ensure_lane_pool(max_lanes=2)

    with pytest.raises(RuntimeError, match="factory boom"):
        batch.ensure_lane_pool(max_lanes=6)

    assert not batch.has_lane_pool()
    assert created == [first]
    assert first.close_calls == [True]


@pytest.mark.unit
def test_create_lane_pool_factory_failure_clears_pool_and_records_failure() -> None:
    recorded: list[tuple[str, str]] = []

    def _factory(_max_lanes: int, _limit_blas_threads: bool) -> object:
        raise RuntimeError("factory boom")

    def _record(message: str, exc: BaseException) -> None:
        recorded.append((str(message), str(exc)))

    batch = ParallelBatchExecutor(
        lane_pool_factory=_factory,
        record_nonfatal_exception=_record,
    )

    with pytest.raises(RuntimeError, match="factory boom"):
        batch.ensure_lane_pool(max_lanes=3)

    assert not batch.has_lane_pool()
    assert batch.current_max_workers is None
    assert batch.runtime_snapshot().warm_failure == "RuntimeError: factory boom"
    assert recorded == [("Failed to create batch lane pool", "factory boom")]


@pytest.mark.unit
def test_ensure_warm_lane_pool_failure_is_retained_after_shutdown() -> None:
    recorded: list[tuple[str, str]] = []

    class _FailingWarmPool:
        def __init__(self) -> None:
            self.close_calls: list[bool] = []

        def warm_lanes(self, _max_lanes: int, *, wait: bool = True) -> None:
            _ = wait
            raise RuntimeError("warm boom")

        def close(self, *, kill: bool = False) -> None:
            self.close_calls.append(bool(kill))

    pool = _FailingWarmPool()
    batch = ParallelBatchExecutor(
        lane_pool_factory=lambda _max_lanes, _limit_blas_threads: pool,
        record_nonfatal_exception=lambda message, exc: recorded.append((str(message), str(exc))),
    )

    with pytest.raises(RuntimeError, match="warm boom"):
        batch.ensure_warm_lane_pool(max_lanes=2, wait=True)

    assert not batch.has_lane_pool()
    assert batch.runtime_snapshot().warm_failure == "RuntimeError: warm boom"
    assert pool.close_calls == [True]
    assert recorded == [("Failed to warm batch lane pool", "warm boom")]


@pytest.mark.unit
def test_busy_warm_batch_lane_close_kills_owner_instead_of_skipping(monkeypatch) -> None:
    created: list[object] = []

    class _FakeRuntimeOwner:
        owner_epoch = 1
        is_running = True
        is_ready = True

        def __init__(self, **_kwargs) -> None:
            self.close_calls: list[bool] = []
            created.append(self)

        def close(self, *, kill: bool = False) -> None:
            self.close_calls.append(bool(kill))

        def drain_events(self) -> list[object]:
            return []

    monkeypatch.setattr("kindred.core.batch_containment.SimulationRuntimeOwner", _FakeRuntimeOwner)
    lane = WarmBatchSimulationLane(lane_id="test-lane")
    assert lane.reserve() is True

    try:
        lane.close(kill=False)

        owner = created[0]
        assert owner.close_calls == [True]
    finally:
        lane.release_reservation()


@pytest.mark.unit
def test_shutdown_force_terminate_closes_lane_pool_with_kill() -> None:
    class _FakeLanePool:
        def __init__(self) -> None:
            self.close_calls: list[bool] = []

        def close(self, *, kill: bool = False) -> None:
            self.close_calls.append(bool(kill))

    pool = _FakeLanePool()
    batch = ParallelBatchExecutor(
        lane_pool_factory=lambda _max_lanes, _limit_blas_threads: _FakeLanePool(),
        lane_pool=pool,
    )

    recorded: list[tuple[str, str]] = []

    batch.shutdown(
        force_terminate=True,
        record_nonfatal_exception=lambda message, exc: recorded.append((str(message), str(exc))),
    )

    assert pool.close_calls == [True]
    assert recorded == []
