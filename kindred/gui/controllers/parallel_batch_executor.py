from __future__ import annotations

from collections.abc import Callable
from typing import Any

from kindred.core.batch_containment import (
    BatchLanePool,
    BatchRequestHandle,
    BatchRuntimeLaneOwner,
)
from kindred.core.batch_runtime_session import (
    BatchRuntimeSession,
    BatchRuntimeSessionRequest,
)
from kindred.core.runtime_defaults import MAX_PARALLEL_WORKERS_CEILING
from kindred.gui.controllers.runtime_lane_allocation import (
    RuntimeBackendLease,
    RuntimeCompatibilityKey,
    RuntimeReleaseReason,
)
from kindred.gui.controllers.simulation_runtime_backend import (
    RuntimeBackendCancelResult,
    RuntimeBackendCloseResult,
    RuntimeBackendPollResult,
)
from kindred.gui.project_schema import PROJECT_DEFAULTS


def _noop_record_nonfatal_exception(_message: str, _exc: BaseException) -> None:
    return None


def default_batch_lane_pool_factory(max_lanes: int, limit_blas_threads: bool) -> BatchLanePool:
    return BatchLanePool(
        max_lanes=max(1, int(max_lanes)),
        limit_blas_threads_per_worker=bool(limit_blas_threads),
    )


class ParallelBatchExecutor:
    """Backend dispatch and lease-provider adapter for runtime lifecycle."""

    __slots__ = ("_lane_owner", "_runtime_session")

    def __init__(
        self,
        lane_pool_factory: Callable[[int, bool], Any] | None = None,
        *,
        max_parallel_workers: int = int(PROJECT_DEFAULTS["max_parallel_batch_workers"]),
        limit_blas_threads_per_worker: bool = bool(PROJECT_DEFAULTS["limit_blas_threads_per_worker"]),
        record_nonfatal_exception: Callable[[str, BaseException], None] = _noop_record_nonfatal_exception,
        lane_pool: Any = None,
    ) -> None:
        self._lane_owner = BatchRuntimeLaneOwner(
            lane_pool_factory=lane_pool_factory or default_batch_lane_pool_factory,
            max_parallel_workers=int(max_parallel_workers),
            limit_blas_threads_per_worker=bool(limit_blas_threads_per_worker),
            record_nonfatal_exception=record_nonfatal_exception,
            lane_pool=lane_pool,
        )
        self._runtime_session = BatchRuntimeSession(self._lane_owner)

    @property
    def lane_pool_factory(self) -> Callable[[int, bool], Any]:
        return self._lane_owner.lane_pool_factory

    @lane_pool_factory.setter
    def lane_pool_factory(self, value: Callable[[int, bool], Any]) -> None:
        self._reconfigure_lane_owner(
            max_parallel_workers=self.max_parallel_workers,
            limit_blas_threads_per_worker=self.limit_blas_threads_per_worker,
            lane_pool_factory=value,
            record_nonfatal_exception=self.record_nonfatal_exception,
        )

    @property
    def max_parallel_workers(self) -> int:
        return int(self._lane_owner.max_parallel_workers)

    @max_parallel_workers.setter
    def max_parallel_workers(self, value: int) -> None:
        self._reconfigure_lane_owner(
            max_parallel_workers=int(value),
            limit_blas_threads_per_worker=self.limit_blas_threads_per_worker,
            lane_pool_factory=self.lane_pool_factory,
            record_nonfatal_exception=self.record_nonfatal_exception,
        )

    @property
    def limit_blas_threads_per_worker(self) -> bool:
        return bool(self._lane_owner.limit_blas_threads_per_worker)

    @limit_blas_threads_per_worker.setter
    def limit_blas_threads_per_worker(self, value: bool) -> None:
        self._reconfigure_lane_owner(
            max_parallel_workers=self.max_parallel_workers,
            limit_blas_threads_per_worker=bool(value),
            lane_pool_factory=self.lane_pool_factory,
            record_nonfatal_exception=self.record_nonfatal_exception,
        )

    @property
    def record_nonfatal_exception(self) -> Callable[[str, BaseException], None]:
        return self._lane_owner.record_nonfatal_exception

    @record_nonfatal_exception.setter
    def record_nonfatal_exception(self, value: Callable[[str, BaseException], None]) -> None:
        self._reconfigure_lane_owner(
            max_parallel_workers=self.max_parallel_workers,
            limit_blas_threads_per_worker=self.limit_blas_threads_per_worker,
            lane_pool_factory=self.lane_pool_factory,
            record_nonfatal_exception=value,
        )

    def _reconfigure_lane_owner(
        self,
        *,
        max_parallel_workers: int,
        limit_blas_threads_per_worker: bool,
        lane_pool_factory: Callable[[int, bool], Any],
        record_nonfatal_exception: Callable[[str, BaseException], None],
    ) -> None:
        if int(self._lane_owner.max_parallel_workers) != int(max_parallel_workers):
            self._lane_owner.max_parallel_workers = int(max_parallel_workers)
        if bool(self._lane_owner.limit_blas_threads_per_worker) != bool(limit_blas_threads_per_worker):
            self._lane_owner.limit_blas_threads_per_worker = bool(limit_blas_threads_per_worker)
        if self._lane_owner.lane_pool_factory is not lane_pool_factory:
            self._lane_owner.lane_pool_factory = lane_pool_factory
        if self._lane_owner.record_nonfatal_exception is not record_nonfatal_exception:
            self._lane_owner.record_nonfatal_exception = record_nonfatal_exception

    def begin_run(
        self,
        *,
        run_id: int,
        request_id: int,
        fast_mode: bool,
        queue_ids: tuple[str, ...] | list[str],
        queue_names: tuple[str, ...] | list[str],
        preview_owner_epoch: int | None = None,
        active_timeout_s: float = 60.0,
    ) -> None:
        self._runtime_session.begin(
            BatchRuntimeSessionRequest(
                run_id=int(run_id),
                request_id=int(request_id),
                fast_mode=bool(fast_mode),
                queue_ids=tuple(str(item) for item in queue_ids),
                queue_names=tuple(str(item) for item in queue_names),
                preview_owner_epoch=preview_owner_epoch,
                active_timeout_s=float(active_timeout_s),
            )
        )

    def ensure_backend_lease(
        self,
        compatibility_key: RuntimeCompatibilityKey,
        capacity: int,
        *,
        wait: bool,
    ) -> RuntimeBackendLease | None:
        requested_lanes = min(
            int(MAX_PARALLEL_WORKERS_CEILING),
            max(1, int(capacity)),
        )
        warm_failure = self._lane_owner.consume_warm_failure()
        if warm_failure:
            raise RuntimeError(str(warm_failure))
        self._lane_owner.ensure_warm_lane_pool(
            max_lanes=requested_lanes,
            wait=bool(wait),
        )
        if not self._lane_owner.has_ready_lane_pool(max_lanes=requested_lanes):
            warm_failure = self._lane_owner.consume_warm_failure()
            if warm_failure:
                raise RuntimeError(str(warm_failure))
            return None
        pool_token = self._lane_owner.lane_pool_token()
        if pool_token is None or bool(self._lane_owner.is_pool_stale):
            return None
        return RuntimeBackendLease(
            lease_id=f"batch-runtime:{pool_token}:{int(self._lane_owner.current_generation)}:{requested_lanes}",
            pool_token=str(pool_token),
            generation=int(self._lane_owner.current_generation),
            compatibility_key=compatibility_key,
            capacity=requested_lanes,
        )

    def invalidate_backend_lease(
        self,
        lease: RuntimeBackendLease | None,
        *,
        reason: RuntimeReleaseReason,
    ) -> None:
        _ = reason
        if self._backend_lease_matches_current_pool(lease):
            self._lane_owner.mark_pool_stale()

    def submit_task(
        self,
        task,
        *,
        set_id: str,
        set_name: str,
        callback_identity: object,
    ) -> BatchRequestHandle:
        sid = str(set_id or "")
        if callback_identity is None:
            raise ValueError("Parallel batch task submission requires callback_identity.")
        return self._runtime_session.submit_task(
            task,
            set_id=sid,
            set_name=str(set_name or set_id or ""),
            callback_identity=callback_identity,
        )

    def poll_completed_records(self) -> RuntimeBackendPollResult:
        return RuntimeBackendPollResult(
            records=tuple(self._runtime_session.poll_completed_records()),
            active_after_poll=1 if self._lane_owner.has_active_requests() else 0,
        )

    def supersede_current_run(self) -> RuntimeBackendCancelResult:
        cancelled, running = self._runtime_session.soft_supersede_active_run()
        return RuntimeBackendCancelResult(cancelled=cancelled, running=running)

    def close_current_run(self, *, force_terminate: bool) -> RuntimeBackendCloseResult:
        pool_token = self._lane_owner.lane_pool_token()
        generation = int(self._lane_owner.current_generation)
        self._runtime_session.shutdown(
            force_terminate=bool(force_terminate),
            record_nonfatal_exception=self.record_nonfatal_exception,
        )
        return RuntimeBackendCloseResult(
            active_after_close=1 if self._lane_owner.has_active_requests() else 0,
            pool_closed=not bool(self._lane_owner.has_lane_pool()),
            pool_token=str(pool_token or ""),
            generation=generation,
        )

    def _backend_lease_matches_current_pool(self, lease: RuntimeBackendLease | None) -> bool:
        if lease is None:
            return False
        pool_token = self._lane_owner.lane_pool_token()
        if pool_token is None or str(pool_token) != str(lease.pool_token):
            return False
        if int(self._lane_owner.current_generation) != int(lease.generation):
            return False
        if self._lane_owner.current_max_workers is not None and int(lease.capacity) > int(self._lane_owner.current_max_workers):
            return False
        return True
