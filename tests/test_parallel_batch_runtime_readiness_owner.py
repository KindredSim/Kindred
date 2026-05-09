from __future__ import annotations

from types import SimpleNamespace

import pytest

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
        self.pool = object()
        self.snapshot = SimpleNamespace(
            current_generation=7,
            has_lane_pool=bool(has_pool),
            pool_stale=bool(stale),
        )

    def has_lane_pool(self) -> bool:
        return self._has_pool

    def has_ready_lane_pool(self, *, max_lanes: int) -> bool:
        return bool(self.ready and int(max_lanes) <= int(self.current_max_workers or 0))

    def ensure_lane_pool(self, *, max_lanes: int):
        self.ensure_calls.append(int(max_lanes))
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
    assert availability.lane_pool is batch_parallel.pool
    assert availability.snapshot.ready is True
    assert availability.snapshot.status == "ready"
    assert availability.snapshot.generation == 7
    assert batch_parallel.ensure_calls == [3]
    assert owner.eagerly_created is True


def test_run_start_availability_reports_waiting_without_creating_lane_pool() -> None:
    batch_parallel = _FakeBatchParallel(ready=False, has_pool=True, stale=False)
    owner = ParallelBatchRuntimeReadinessOwner(
        batch_parallel=batch_parallel,
        capacity_getter=lambda: 4,
    )

    availability = owner.run_start_availability(required_lanes=3)

    assert availability.ready is False
    assert availability.lane_pool is None
    assert availability.error is None
    assert availability.snapshot.ready is False
    assert availability.snapshot.status == "warming"
    assert availability.snapshot.message == "Preparing batch runtime..."
    assert batch_parallel.ensure_calls == []
    assert owner.eagerly_created is False
