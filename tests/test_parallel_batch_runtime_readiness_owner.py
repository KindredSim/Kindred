from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

import kindred.gui.controllers.parallel_batch_runtime_readiness_owner as readiness_module
from kindred.gui.controllers.parallel_batch_runtime_readiness_owner import (
    ParallelBatchRuntimeReadinessOwner,
)


pytestmark = pytest.mark.unit


class _FakeBatchParallel:
    def __init__(self, *, ready: bool = True, has_pool: bool = True, stale: bool = False) -> None:
        self.ready = bool(ready)
        self._has_pool = bool(has_pool)
        self.is_pool_stale = bool(stale)
        self.current_max_workers = 4 if has_pool else None
        self.ensure_calls: list[int] = []
        self.warm_calls: list[dict[str, object]] = []
        self.pool = object()
        self.snapshot = SimpleNamespace(
            current_generation=7,
            has_lane_pool=bool(has_pool),
            pool_stale=bool(stale),
            warm_failure=None,
        )

    def has_lane_pool(self) -> bool:
        return self._has_pool

    def has_ready_lane_pool(self, *, max_lanes: int) -> bool:
        return bool(self.ready and int(max_lanes) <= int(self.current_max_workers or 0))

    def ensure_lane_pool(self, *, max_lanes: int):
        self.ensure_calls.append(int(max_lanes))
        return self.pool

    def lane_pool_token(self) -> int | None:
        return id(self.pool) if self._has_pool else None

    def ensure_warm_lane_pool(self, *, max_lanes: int, wait: bool):
        self.warm_calls.append({"max_lanes": int(max_lanes), "wait": bool(wait)})
        self._has_pool = True
        self.current_max_workers = max(int(self.current_max_workers or 0), int(max_lanes))
        if bool(wait):
            self.ready = True
        return self.pool

    def runtime_snapshot(self):
        return self.snapshot


def test_run_start_availability_returns_existing_ready_lane_pool() -> None:
    batch_parallel = _FakeBatchParallel(ready=True, has_pool=True, stale=False)
    owner = ParallelBatchRuntimeReadinessOwner(
        batch_parallel=batch_parallel,
        capacity_getter=lambda: 4,
    )

    availability = owner.run_start_availability(required_lanes=3)

    assert availability.ready is True
    assert availability.lane_pool_token == id(batch_parallel.pool)
    assert availability.snapshot.ready is True
    assert availability.snapshot.status == "ready"
    assert availability.snapshot.generation == 7
    assert batch_parallel.ensure_calls == []


def test_run_start_availability_reports_waiting_without_creating_lane_pool() -> None:
    batch_parallel = _FakeBatchParallel(ready=False, has_pool=True, stale=False)
    owner = ParallelBatchRuntimeReadinessOwner(
        batch_parallel=batch_parallel,
        capacity_getter=lambda: 4,
    )

    availability = owner.run_start_availability(required_lanes=3)

    assert availability.ready is False
    assert availability.lane_pool_token is None
    assert availability.error is None
    assert availability.snapshot.ready is False
    assert availability.snapshot.status == "warming"
    assert availability.snapshot.message == "Preparing batch runtime..."
    assert batch_parallel.ensure_calls == []


def test_batch_waiting_runtime_status_mapping_has_one_owner() -> None:
    source = inspect.getsource(ParallelBatchRuntimeReadinessOwner)

    assert 'getattr(snapshot, "pool_stale", False)' not in source
    assert 'getattr(snapshot, "has_lane_pool", False)' not in source
    assert 'getattr(snapshot, "warm_failure", None)' not in source
    assert "snapshot.pool_stale" in source
    assert "snapshot.has_lane_pool" in source
    assert "snapshot.warm_failure" in source


def test_ensure_warms_only_required_lanes_for_current_workflow() -> None:
    batch_parallel = _FakeBatchParallel(ready=False, has_pool=False)
    owner = ParallelBatchRuntimeReadinessOwner(
        batch_parallel=batch_parallel,
        capacity_getter=lambda: 8,
    )

    owner.ensure(wait=True, required_lanes=3)

    assert batch_parallel.warm_calls == [{"max_lanes": 3, "wait": True}]
    assert owner.ready(required_lanes=3) is True


def test_nonblocking_ensure_starts_larger_required_lane_warm(monkeypatch) -> None:
    batch_parallel = _FakeBatchParallel(ready=False, has_pool=False)
    owner = ParallelBatchRuntimeReadinessOwner(
        batch_parallel=batch_parallel,
        capacity_getter=lambda: 8,
    )
    created_threads: list[object] = []

    class _FakeThread:
        def __init__(self, *, target, kwargs, name: str, daemon: bool) -> None:
            self.target = target
            self.kwargs = dict(kwargs)
            self.name = str(name)
            self.daemon = bool(daemon)
            self.started = False
            self._alive = True
            created_threads.append(self)

        def start(self) -> None:
            self.started = True

        def is_alive(self) -> bool:
            return bool(self._alive)

    monkeypatch.setattr(readiness_module.threading, "Thread", _FakeThread)

    owner.ensure(wait=False, required_lanes=1)
    owner.ensure(wait=False, required_lanes=3)
    owner.ensure(wait=False, required_lanes=2)

    assert [thread.kwargs["required_lanes"] for thread in created_threads] == [1, 3]
    assert all(thread.started for thread in created_threads)
