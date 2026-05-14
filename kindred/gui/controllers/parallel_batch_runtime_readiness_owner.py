from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Callable, Optional

from kindred.core.batch_runtime_session import BatchRuntimeSessionSnapshot
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
    lane_pool_token: int | None
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
        self._warm_thread: Optional[threading.Thread] = None
        self._warm_required_lanes: int | None = None
        self._warm_lock = threading.RLock()

    def wait_for_background_warm(self, *, timeout_s: float = 1.0) -> bool:
        with self._warm_lock:
            thread = self._warm_thread
        if thread is None:
            return False
        thread.join(timeout=max(0.0, float(timeout_s)))
        return not thread.is_alive()

    def ready(self, *, required_lanes: int | None = None) -> bool:
        effective_workers = self._effective_workers(required_lanes=required_lanes)
        try:
            ready = bool(
                self._batch_parallel.has_ready_lane_pool(max_lanes=max(1, int(effective_workers)))
            )
        except Exception:
            ready = False
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
            return _runtime_snapshot(
                status="failed",
                ready=False,
                failure=f"{type(exc).__name__}: {exc}",
                message=f"Batch runtime readiness check failed: {exc}",
                polling=False,
            )
        generation = int(snapshot.current_generation)
        if ready:
            return _runtime_snapshot(
                status="ready",
                ready=True,
                generation=generation,
                polling=False,
            )
        failure = snapshot.warm_failure
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

    def ensure(self, *, wait: bool = False, required_lanes: int | None = None) -> None:
        effective_workers = self._effective_workers(required_lanes=required_lanes)
        required = max(1, int(effective_workers))
        try:
            if self._batch_parallel.has_ready_lane_pool(max_lanes=required):
                return
        except Exception:
            pass
        if not bool(wait):
            with self._warm_lock:
                existing = self._warm_thread
                existing_required = self._warm_required_lanes
                if (
                    existing is not None
                    and existing.is_alive()
                    and existing_required is not None
                    and int(existing_required) >= int(required)
                ):
                    return
                thread = threading.Thread(
                    target=self.ensure,
                    kwargs={"wait": True, "required_lanes": required},
                    name="kindred-batch-runtime-readiness",
                    daemon=True,
                )
                self._warm_thread = thread
                self._warm_required_lanes = required
                thread.start()
            return
        try:
            self._batch_parallel.ensure_warm_lane_pool(
                max_lanes=required,
                wait=bool(wait),
            )
        except Exception:
            return

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
                lane_pool_token = self._batch_parallel.lane_pool_token()
                snapshot = self._batch_parallel.runtime_snapshot()
                return ParallelBatchRunStartAvailability(
                    ready=True,
                    lane_pool_token=None if lane_pool_token is None else int(lane_pool_token),
                    snapshot=_runtime_snapshot(
                        status="ready",
                        ready=True,
                        generation=int(snapshot.current_generation),
                        polling=False,
                    ),
                )
            snapshot = self._batch_parallel.runtime_snapshot()
        except Exception as exc:
            return ParallelBatchRunStartAvailability(
                ready=False,
                lane_pool_token=None,
                snapshot=_runtime_snapshot(
                    status="failed",
                    ready=False,
                    failure=str(exc),
                    message=f"Batch runtime readiness check failed: {exc}",
                    polling=False,
                ),
                error=exc,
            )

        return ParallelBatchRunStartAvailability(
            ready=False,
            lane_pool_token=None,
            snapshot=self._waiting_runtime_snapshot(snapshot),
        )

    def _waiting_runtime_snapshot(self, snapshot: BatchRuntimeSessionSnapshot) -> RuntimeReadinessSnapshot:
        if bool(snapshot.pool_stale):
            status = "stale"
            message = "Rebuilding batch runtime..."
        elif bool(snapshot.has_lane_pool):
            status = "warming"
            message = "Preparing batch runtime..."
        else:
            status = "missing"
            message = "Preparing batch runtime..."
        return _runtime_snapshot(
            status=status,
            ready=False,
            generation=int(snapshot.current_generation),
            message=message,
            polling=True,
        )

    def _effective_workers(self, *, required_lanes: int | None = None) -> int:
        if required_lanes is not None:
            return max(1, int(required_lanes))
        try:
            return max(1, int(self._capacity_getter()))
        except Exception:
            return 1
