from __future__ import annotations

from collections.abc import Callable
from typing import Any, Dict, Optional

from kindred.core.batch_containment import (
    BatchLanePool,
    BatchPolledCompletion,
    BatchRequestHandle,
    BatchRuntimeLaneOwner,
)
from kindred.core.batch_runtime_session import (
    BatchRuntimeSession,
    BatchRuntimeSessionRequest,
    BatchRuntimeSessionSnapshot,
)
from kindred.core.runtime_defaults import MAX_PARALLEL_WORKERS_CEILING
from kindred.gui.project_schema import PROJECT_DEFAULTS


def _noop_record_nonfatal_exception(_message: str, _exc: BaseException) -> None:
    return None


def default_batch_lane_pool_factory(max_lanes: int, limit_blas_threads: bool) -> BatchLanePool:
    return BatchLanePool(
        max_lanes=max(1, int(max_lanes)),
        limit_blas_threads_per_worker=bool(limit_blas_threads),
    )


class ParallelBatchExecutor:
    """Controller adapter over the non-GUI batch runtime session."""

    __slots__ = ("_runtime_session",)

    def __init__(
        self,
        lane_pool_factory: Callable[[int, bool], Any] | None = None,
        *,
        max_parallel_workers: int = int(PROJECT_DEFAULTS["max_parallel_batch_workers"]),
        limit_blas_threads_per_worker: bool = bool(PROJECT_DEFAULTS["limit_blas_threads_per_worker"]),
        record_nonfatal_exception: Callable[[str, BaseException], None] = _noop_record_nonfatal_exception,
        lane_pool: Any = None,
    ) -> None:
        runtime_owner = BatchRuntimeLaneOwner(
            lane_pool_factory=lane_pool_factory or default_batch_lane_pool_factory,
            max_parallel_workers=int(max_parallel_workers),
            limit_blas_threads_per_worker=bool(limit_blas_threads_per_worker),
            record_nonfatal_exception=record_nonfatal_exception,
            lane_pool=lane_pool,
        )
        self._runtime_session = BatchRuntimeSession(runtime_owner)

    @property
    def lane_pool_factory(self) -> Callable[[int, bool], Any]:
        return self._runtime_session.lane_pool_factory

    @lane_pool_factory.setter
    def lane_pool_factory(self, value: Callable[[int, bool], Any]) -> None:
        self._runtime_session.reconfigure_runtime_pool(
            max_parallel_workers=self.max_parallel_workers,
            limit_blas_threads_per_worker=self.limit_blas_threads_per_worker,
            lane_pool_factory=value,
            record_nonfatal_exception=self.record_nonfatal_exception,
        )

    @property
    def max_parallel_workers(self) -> int:
        return int(self._runtime_session.max_parallel_workers)

    @max_parallel_workers.setter
    def max_parallel_workers(self, value: int) -> None:
        self._runtime_session.reconfigure_runtime_pool(
            max_parallel_workers=int(value),
            limit_blas_threads_per_worker=self.limit_blas_threads_per_worker,
            lane_pool_factory=self.lane_pool_factory,
            record_nonfatal_exception=self.record_nonfatal_exception,
        )

    @property
    def limit_blas_threads_per_worker(self) -> bool:
        return bool(self._runtime_session.limit_blas_threads_per_worker)

    @limit_blas_threads_per_worker.setter
    def limit_blas_threads_per_worker(self, value: bool) -> None:
        self._runtime_session.reconfigure_runtime_pool(
            max_parallel_workers=self.max_parallel_workers,
            limit_blas_threads_per_worker=bool(value),
            lane_pool_factory=self.lane_pool_factory,
            record_nonfatal_exception=self.record_nonfatal_exception,
        )

    @property
    def record_nonfatal_exception(self) -> Callable[[str, BaseException], None]:
        return self._runtime_session.record_nonfatal_exception

    @record_nonfatal_exception.setter
    def record_nonfatal_exception(self, value: Callable[[str, BaseException], None]) -> None:
        self._runtime_session.reconfigure_runtime_pool(
            max_parallel_workers=self.max_parallel_workers,
            limit_blas_threads_per_worker=self.limit_blas_threads_per_worker,
            lane_pool_factory=self.lane_pool_factory,
            record_nonfatal_exception=value,
        )

    @property
    def current_max_workers(self) -> Optional[int]:
        return self._runtime_session.current_max_workers

    def begin_run(
        self,
        *,
        run_id: int,
        request_id: int,
        fast_mode: bool,
        queue_ids: tuple[str, ...] | list[str],
        queue_names: tuple[str, ...] | list[str],
        keep_lane_pool_alive: bool,
        preview_owner_epoch: int | None = None,
        active_timeout_s: float = 60.0,
        cache_key: str = "",
    ) -> None:
        self._runtime_session.begin(
            BatchRuntimeSessionRequest(
                run_id=int(run_id),
                request_id=int(request_id),
                fast_mode=bool(fast_mode),
                queue_ids=tuple(str(item) for item in queue_ids),
                queue_names=tuple(str(item) for item in queue_names),
                keep_lane_pool_alive=bool(keep_lane_pool_alive),
                preview_owner_epoch=preview_owner_epoch,
                active_timeout_s=float(active_timeout_s),
                cache_key=str(cache_key or ""),
            )
        )

    def runtime_snapshot(self) -> BatchRuntimeSessionSnapshot:
        return self._runtime_session.snapshot()

    def active_request_count(self) -> int:
        return self._runtime_session.active_request_count()

    def has_active_requests(self) -> bool:
        return self._runtime_session.has_active_requests()

    def request_worker_count(self) -> int:
        return self._runtime_session.request_worker_count()

    def join_active_requests(self, *, timeout_s: float = 2.0) -> None:
        self._runtime_session.join_active_requests(timeout_s=float(timeout_s))

    def has_lane_pool(self) -> bool:
        return self._runtime_session.has_lane_pool()

    def has_ready_lane_pool(self, *, max_lanes: int) -> bool:
        requested_lanes = min(
            int(MAX_PARALLEL_WORKERS_CEILING),
            max(1, int(max_lanes)),
        )
        return self._runtime_session.has_ready_lane_pool(max_lanes=requested_lanes)

    def lane_pool_token(self) -> int | None:
        return self._runtime_session.lane_pool_token()

    def active_request_metadata(self, set_id: str) -> Dict[str, Any]:
        sid = str(set_id or "")
        return self._runtime_session.active_request_metadata(sid)

    def discard_request(self, set_id: str) -> None:
        sid = str(set_id or "")
        self._runtime_session.discard_request(sid)

    def reset_active_run_state(self) -> None:
        self._runtime_session.reset_active_run_state()

    def reset_run_state(self) -> None:
        self._runtime_session.reset_run_state()

    def drain_completion_queue(self) -> None:
        self._runtime_session.drain_completion_queue()

    def enqueue_completion(self, set_id: str) -> None:
        self._runtime_session.enqueue_completion(set_id)

    def clear_stale_requests(self) -> None:
        self._runtime_session.clear_stale_requests()

    def poll_completed_records(self) -> list[BatchPolledCompletion]:
        return self._runtime_session.poll_completed_records()

    def ensure_lane_pool(self, *, max_lanes: int) -> Any:
        requested_lanes = min(
            int(MAX_PARALLEL_WORKERS_CEILING),
            max(1, int(max_lanes)),
        )
        return self._runtime_session.ensure_lane_pool(max_lanes=requested_lanes)

    def ensure_warm_lane_pool(self, *, max_lanes: int, wait: bool = True) -> Any:
        requested_lanes = min(
            int(MAX_PARALLEL_WORKERS_CEILING),
            max(1, int(max_lanes)),
        )
        return self._runtime_session.ensure_warm_lane_pool(
            max_lanes=requested_lanes,
            wait=bool(wait),
        )

    def submit_task(
        self,
        task,
        *,
        set_id: str,
        set_name: str,
        expected_owner_epoch: object = None,
        callback_identity: object,
    ) -> BatchRequestHandle:
        expected_lane_owner_epoch = None if expected_owner_epoch is None else int(expected_owner_epoch)
        sid = str(set_id or "")
        if callback_identity is None:
            raise ValueError("Parallel batch task submission requires callback_identity.")
        return self._runtime_session.submit_task(
            task,
            set_id=sid,
            set_name=str(set_name or set_id or ""),
            expected_owner_epoch=expected_lane_owner_epoch,
            callback_identity=callback_identity,
        )

    def shutdown(
        self,
        *,
        force_terminate: bool,
        record_nonfatal_exception: Callable[[str, BaseException], None],
    ) -> None:
        self._runtime_session.shutdown(
            force_terminate=bool(force_terminate),
            record_nonfatal_exception=record_nonfatal_exception,
        )

    def soft_supersede(self) -> tuple[int, int]:
        return self._runtime_session.soft_supersede_active_run()

    def finish_after_run(
        self,
        *,
        keep_lane_pool_alive: bool,
        record_nonfatal_exception: Callable[[str, BaseException], None],
    ) -> None:
        self._runtime_session.finish_after_run(
            keep_lane_pool_alive=bool(keep_lane_pool_alive),
            record_nonfatal_exception=record_nonfatal_exception,
        )

    @property
    def is_pool_stale(self) -> bool:
        return bool(self._runtime_session.is_pool_stale)

    def mark_pool_stale(self) -> None:
        self._runtime_session.mark_pool_stale()
