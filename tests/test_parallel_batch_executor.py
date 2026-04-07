from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from kindred.core.batch_parallel import prewarm_worker_imports
from kindred.gui.controllers.parallel_batch_executor import ParallelBatchExecutor


@dataclass
class _Submission:
    fn: Any
    args: tuple[Any, ...]
    kwargs: dict[str, Any]


class _FakeExecutor:
    def __init__(self, *, max_workers: int, expose_max_workers: bool = True) -> None:
        if bool(expose_max_workers):
            self._max_workers = int(max_workers)
        self.submissions: list[_Submission] = []
        self.shutdown_calls: list[dict[str, Any]] = []

    def submit(self, fn, *args, **kwargs):
        self.submissions.append(_Submission(fn=fn, args=args, kwargs=dict(kwargs)))
        return object()

    def shutdown(self, wait=True, cancel_futures=False):
        self.shutdown_calls.append(
            {
                "wait": bool(wait),
                "cancel_futures": bool(cancel_futures),
            }
        )


@pytest.mark.unit
def test_ensure_executor_recreates_pool_when_worker_count_increases() -> None:
    created: list[tuple[int, bool, _FakeExecutor]] = []

    def _factory(max_workers: int, limit_blas_threads: bool) -> _FakeExecutor:
        executor = _FakeExecutor(max_workers=max_workers)
        created.append((int(max_workers), bool(limit_blas_threads), executor))
        return executor

    batch = ParallelBatchExecutor(executor_factory=_factory)

    first = batch.ensure_executor(max_workers=2)
    second = batch.ensure_executor(max_workers=6)

    assert [item[:2] for item in created] == [(2, True), (6, True)]
    assert second is not first
    assert first.shutdown_calls == [{"wait": False, "cancel_futures": True}]
    assert batch.executor is second


@pytest.mark.unit
def test_ensure_executor_does_not_resize_downward() -> None:
    created: list[tuple[int, bool]] = []

    def _factory(max_workers: int, limit_blas_threads: bool) -> _FakeExecutor:
        created.append((int(max_workers), bool(limit_blas_threads)))
        return _FakeExecutor(max_workers=max_workers)

    batch = ParallelBatchExecutor(executor_factory=_factory)

    first = batch.ensure_executor(max_workers=6)
    second = batch.ensure_executor(max_workers=4)

    assert created == [(6, True)]
    assert second is first


@pytest.mark.unit
def test_ensure_executor_does_not_resize_when_worker_count_matches() -> None:
    created: list[tuple[int, bool]] = []

    def _factory(max_workers: int, limit_blas_threads: bool) -> _FakeExecutor:
        created.append((int(max_workers), bool(limit_blas_threads)))
        return _FakeExecutor(max_workers=max_workers)

    batch = ParallelBatchExecutor(executor_factory=_factory)

    first = batch.ensure_executor(max_workers=6)
    second = batch.ensure_executor(max_workers=6)

    assert created == [(6, True)]
    assert second is first


@pytest.mark.unit
def test_ensure_executor_reuses_existing_pool_when_size_metadata_is_missing() -> None:
    created: list[tuple[int, bool]] = []

    def _factory(max_workers: int, limit_blas_threads: bool) -> _FakeExecutor:
        created.append((int(max_workers), bool(limit_blas_threads)))
        return _FakeExecutor(max_workers=max_workers, expose_max_workers=False)

    batch = ParallelBatchExecutor(executor_factory=_factory)

    first = batch.ensure_executor(max_workers=2)
    second = batch.ensure_executor(max_workers=6)

    assert created == [(2, True)]
    assert second is first


@pytest.mark.unit
def test_ensure_executor_prewarms_all_workers_after_fresh_creation() -> None:
    batch = ParallelBatchExecutor(executor_factory=lambda max_workers, _limit: _FakeExecutor(max_workers=max_workers))

    executor = batch.ensure_executor(max_workers=4)

    assert len(executor.submissions) == 4
    assert all(sub.fn is prewarm_worker_imports for sub in executor.submissions)
    assert all(sub.args == () for sub in executor.submissions)
    assert all(sub.kwargs == {} for sub in executor.submissions)


@pytest.mark.unit
def test_ensure_executor_resize_factory_failure_leaves_wrapper_consistent() -> None:
    created: list[_FakeExecutor] = []

    def _factory(max_workers: int, _limit_blas_threads: bool) -> _FakeExecutor:
        if int(max_workers) == 6:
            raise RuntimeError("factory boom")
        executor = _FakeExecutor(max_workers=max_workers)
        created.append(executor)
        return executor

    batch = ParallelBatchExecutor(executor_factory=_factory)
    first = batch.ensure_executor(max_workers=2)

    with pytest.raises(RuntimeError, match="factory boom"):
        batch.ensure_executor(max_workers=6)

    assert batch.executor is None
    assert created == [first]
    assert first.shutdown_calls == [{"wait": False, "cancel_futures": True}]
