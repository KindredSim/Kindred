from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Callable, Optional

from kindred.core.simulation_runtime_readiness import RuntimeReadinessSnapshot


def _runtime_snapshot(
    *,
    status: str,
    ready: bool,
    generation: int = 0,
    failure: str | None = None,
    message: str | None = None,
    polling: bool | None = None,
) -> RuntimeReadinessSnapshot:
    ready_value = bool(ready)
    polling_value = bool((not ready_value) and status in {"missing", "warming", "stale"}) if polling is None else bool(polling)
    return RuntimeReadinessSnapshot(
        mode="batch",
        status=str(status),
        ready=ready_value,
        generation=int(generation),
        failure=failure,
        message=message,
        required=True,
        controls_ready=ready_value,
        polling=polling_value,
    )


@dataclass(frozen=True)
class ParallelBatchRunStartAvailability:
    ready: bool
    lane_pool: object | None
    snapshot: RuntimeReadinessSnapshot
    error: BaseException | None = None


class ParallelBatchRuntimeReadinessOwner:
    """Owns parallel batch runtime readiness state and nonblocking warm lifecycle."""

    def __init__(
        self,
        *,
        batch_parallel: object,
        capacity_getter: Callable[[], int],
    ) -> None:
        self._batch_parallel = batch_parallel
        self._capacity_getter = capacity_getter
        self._eagerly_created = False
        self._eager_creation_thread: Optional[threading.Thread] = None
        self._eager_creation_lock = threading.RLock()

    @property
    def batch_parallel(self) -> object:
        return self._batch_parallel

    @batch_parallel.setter
    def batch_parallel(self, value: object) -> None:
        self._batch_parallel = value

    @property
    def eagerly_created(self) -> bool:
        return bool(self._eagerly_created)

    @eagerly_created.setter
    def eagerly_created(self, value: bool) -> None:
        self._eagerly_created = bool(value)

    @property
    def eager_creation_thread(self) -> Optional[threading.Thread]:
        return self._eager_creation_thread

    def mark_ready(self) -> None:
        self._eagerly_created = True

    def mark_not_ready(self) -> None:
        self._eagerly_created = False

    def ready(self) -> bool:
        effective_workers = self._effective_workers()
        try:
            ready = bool(
                self._batch_parallel.has_ready_lane_pool(max_lanes=max(1, int(effective_workers)))
            )
        except Exception:
            ready = False
        self._eagerly_created = bool(ready)
        return bool(ready)

    def runtime_snapshot_for_selection(
        self,
        *,
        row_count: int,
        required_lanes: int,
    ) -> RuntimeReadinessSnapshot:
        if int(row_count) <= 1 or int(required_lanes) <= 1:
            return RuntimeReadinessSnapshot(
                mode="batch",
                status="not_applicable",
                ready=False,
                generation=0,
                failure=None,
                message="Parallel batch runtime is not required for the current selection.",
                required=False,
                controls_ready=True,
                polling=False,
            )
        required = max(1, int(required_lanes))
        try:
            ready = bool(self._batch_parallel.has_ready_lane_pool(max_lanes=required))
            snapshot = self._batch_parallel.runtime_snapshot()
        except Exception as exc:
            self._eagerly_created = False
            return _runtime_snapshot(
                status="failed",
                ready=False,
                failure=f"{type(exc).__name__}: {exc}",
                message=f"Batch runtime readiness check failed: {exc}",
                polling=False,
            )
        generation = int(getattr(snapshot, "current_generation", 0) or 0)
        if ready:
            self._eagerly_created = True
            return _runtime_snapshot(
                status="ready",
                ready=True,
                generation=generation,
                polling=False,
            )
        self._eagerly_created = False
        failure = getattr(snapshot, "warm_failure", None)
        if failure:
            return _runtime_snapshot(
                status="failed",
                ready=False,
                generation=generation,
                failure=str(failure),
                message=f"Batch runtime failed to prepare. {failure}",
                polling=False,
            )
        return self._waiting_runtime_snapshot(snapshot)

    def ensure(self, *, wait: bool = False) -> None:
        effective_workers = self._effective_workers()
        try:
            if self._batch_parallel.has_ready_lane_pool(max_lanes=max(1, int(effective_workers))):
                self._eagerly_created = True
                return
        except Exception:
            self._eagerly_created = False
        self._eagerly_created = False
        if not bool(wait):
            with self._eager_creation_lock:
                existing = self._eager_creation_thread
                if existing is not None and existing.is_alive():
                    return
                thread = threading.Thread(
                    target=self.ensure,
                    kwargs={"wait": True},
                    name="kindred-batch-runtime-readiness",
                    daemon=True,
                )
                self._eager_creation_thread = thread
                thread.start()
            return
        try:
            self._batch_parallel.ensure_warm_lane_pool(
                max_lanes=max(1, int(effective_workers)),
                wait=bool(wait),
            )
        except Exception:
            self._eagerly_created = False
            return
        self._eagerly_created = bool(
            self._batch_parallel.has_ready_lane_pool(max_lanes=max(1, int(effective_workers)))
        )

    def run_start_availability(self, *, required_lanes: int) -> ParallelBatchRunStartAvailability:
        required = max(1, int(required_lanes))
        try:
            existing_capacity = self._batch_parallel.current_max_workers
            pool_ready = bool(
                self._batch_parallel.has_lane_pool()
                and (not self._batch_parallel.is_pool_stale)
                and existing_capacity is not None
                and int(existing_capacity) >= int(required)
                and self._batch_parallel.has_ready_lane_pool(max_lanes=int(required))
            )
            if pool_ready:
                lane_pool = self._batch_parallel.ensure_lane_pool(max_lanes=int(required))
                snapshot = self._batch_parallel.runtime_snapshot()
                self._eagerly_created = True
                return ParallelBatchRunStartAvailability(
                    ready=True,
                    lane_pool=lane_pool,
                    snapshot=_runtime_snapshot(
                        status="ready",
                        ready=True,
                        generation=int(getattr(snapshot, "current_generation", 0) or 0),
                        polling=False,
                    ),
                )
            snapshot = self._batch_parallel.runtime_snapshot()
        except Exception as exc:
            self._eagerly_created = False
            return ParallelBatchRunStartAvailability(
                ready=False,
                lane_pool=None,
                snapshot=_runtime_snapshot(
                    status="failed",
                    ready=False,
                    failure=str(exc),
                    message=f"Batch runtime readiness check failed: {exc}",
                    polling=False,
                ),
                error=exc,
            )

        self._eagerly_created = False
        return ParallelBatchRunStartAvailability(
            ready=False,
            lane_pool=None,
            snapshot=self._waiting_runtime_snapshot(snapshot),
        )

    def _waiting_runtime_snapshot(self, snapshot: object) -> RuntimeReadinessSnapshot:
        if bool(getattr(snapshot, "pool_stale", False)):
            status = "stale"
            message = "Rebuilding batch runtime..."
        elif bool(getattr(snapshot, "has_lane_pool", False)):
            status = "warming"
            message = "Preparing batch runtime..."
        else:
            status = "missing"
            message = "Preparing batch runtime..."
        return _runtime_snapshot(
            status=status,
            ready=False,
            generation=int(getattr(snapshot, "current_generation", 0) or 0),
            message=message,
            polling=True,
        )

    def _effective_workers(self) -> int:
        try:
            return max(1, int(self._capacity_getter()))
        except Exception:
            return 1
