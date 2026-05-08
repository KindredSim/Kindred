from __future__ import annotations

import threading
from typing import Callable, Optional


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

    def _effective_workers(self) -> int:
        try:
            return max(1, int(self._capacity_getter()))
        except Exception:
            return 1
