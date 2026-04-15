from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from kindred.core.runtime_defaults import MAX_PARALLEL_WORKERS_CEILING
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


class _FakeProcess:
    def __init__(self) -> None:
        self.terminate_calls = 0
        self._alive = True

    def is_alive(self) -> bool:
        return bool(self._alive)

    def terminate(self) -> None:
        self.terminate_calls += 1
        self._alive = False


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
def test_ensure_executor_tracks_current_max_workers_without_executor_introspection() -> None:
    created: list[tuple[int, bool]] = []

    def _factory(max_workers: int, limit_blas_threads: bool) -> _FakeExecutor:
        created.append((int(max_workers), bool(limit_blas_threads)))
        return _FakeExecutor(max_workers=max_workers, expose_max_workers=False)

    batch = ParallelBatchExecutor(executor_factory=_factory)

    first = batch.ensure_executor(max_workers=2)
    second = batch.ensure_executor(max_workers=6)

    assert created == [(2, True), (6, True)]
    assert batch._current_max_workers == 6
    assert second is not first


@pytest.mark.unit
def test_ensure_executor_prewarms_all_workers_after_fresh_creation() -> None:
    batch = ParallelBatchExecutor(executor_factory=lambda max_workers, _limit: _FakeExecutor(max_workers=max_workers))

    executor = batch.ensure_executor(max_workers=4)

    assert len(executor.submissions) == 4
    assert all(sub.fn is prewarm_worker_imports for sub in executor.submissions)
    assert all(sub.args == () for sub in executor.submissions)
    assert all(sub.kwargs == {} for sub in executor.submissions)


@pytest.mark.unit
def test_ensure_executor_caps_worker_count_at_shared_ceiling() -> None:
    created: list[tuple[int, bool]] = []

    def _factory(max_workers: int, limit_blas_threads: bool) -> _FakeExecutor:
        created.append((int(max_workers), bool(limit_blas_threads)))
        return _FakeExecutor(max_workers=max_workers)

    batch = ParallelBatchExecutor(executor_factory=_factory)

    executor = batch.ensure_executor(max_workers=200)

    assert created == [(int(MAX_PARALLEL_WORKERS_CEILING), True)]
    assert batch._current_max_workers == int(MAX_PARALLEL_WORKERS_CEILING)
    assert len(executor.submissions) == int(MAX_PARALLEL_WORKERS_CEILING)


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


@pytest.mark.unit
def test_create_and_prewarm_executor_submit_failure_clears_executor_and_records_failure() -> None:
    created: list[_FakeExecutor] = []
    recorded: list[tuple[str, str]] = []

    class _SubmitFailExecutor(_FakeExecutor):
        def submit(self, fn, *args, **kwargs):
            raise RuntimeError("submit boom")

    def _factory(max_workers: int, _limit_blas_threads: bool) -> _SubmitFailExecutor:
        executor = _SubmitFailExecutor(max_workers=max_workers)
        created.append(executor)
        return executor

    def _record(message: str, exc: BaseException) -> None:
        recorded.append((str(message), str(exc)))

    batch = ParallelBatchExecutor(
        executor_factory=_factory,
        record_nonfatal_exception=_record,
    )

    with pytest.raises(RuntimeError, match="submit boom"):
        batch.ensure_executor(max_workers=3)

    assert len(created) == 1
    assert created[0].shutdown_calls == [{"wait": False, "cancel_futures": True}]
    assert batch.executor is None
    assert batch._current_max_workers is None
    assert recorded == [("Failed to create and prewarm batch executor", "submit boom")]


@pytest.mark.unit
def test_shutdown_force_terminate_uses_process_snapshot_after_executor_shutdown() -> None:
    process = _FakeProcess()

    class _ClearingExecutor(_FakeExecutor):
        def __init__(self) -> None:
            super().__init__(max_workers=1)
            self._processes = {1: process}

        def shutdown(self, wait=True, cancel_futures=False):
            super().shutdown(wait=wait, cancel_futures=cancel_futures)
            self._processes = None

    batch = ParallelBatchExecutor(executor_factory=lambda max_workers, limit_blas_threads: _ClearingExecutor())
    executor = _ClearingExecutor()
    batch.executor = executor
    batch._current_max_workers = 1

    recorded: list[tuple[str, str]] = []

    batch.shutdown(
        force_terminate=True,
        record_nonfatal_exception=lambda message, exc: recorded.append((str(message), str(exc))),
    )

    assert executor.shutdown_calls == [{"wait": False, "cancel_futures": True}]
    assert process.terminate_calls == 1
    assert recorded == []
