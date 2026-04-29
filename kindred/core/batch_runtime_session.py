from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

from kindred.core.batch_containment import BatchPolledCompletion, BatchRequestHandle


class BatchRuntimeSessionState(Enum):
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    SUPERSEDED = "superseded"
    SHUTDOWN = "shutdown"


@dataclass(frozen=True)
class BatchRuntimeSessionRequest:
    run_id: int
    request_id: int
    fast_mode: bool
    queue_ids: tuple[str, ...]
    queue_names: tuple[str, ...]
    keep_lane_pool_alive: bool
    preview_owner_epoch: int | None = None
    active_timeout_s: float = 60.0
    cache_key: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", int(self.run_id))
        object.__setattr__(self, "request_id", int(self.request_id))
        object.__setattr__(self, "fast_mode", bool(self.fast_mode))
        object.__setattr__(self, "queue_ids", tuple(str(item) for item in self.queue_ids if str(item)))
        object.__setattr__(self, "queue_names", tuple(str(item) for item in self.queue_names))
        object.__setattr__(self, "keep_lane_pool_alive", bool(self.keep_lane_pool_alive))
        object.__setattr__(
            self,
            "preview_owner_epoch",
            None if self.preview_owner_epoch is None else int(self.preview_owner_epoch),
        )
        object.__setattr__(self, "active_timeout_s", float(self.active_timeout_s))
        object.__setattr__(self, "cache_key", str(self.cache_key or ""))


@dataclass(frozen=True)
class BatchRuntimeSessionSnapshot:
    state: BatchRuntimeSessionState
    active: bool
    run_id: int
    request_id: int
    fast_mode: bool
    queue_ids: tuple[str, ...]
    queue_names: tuple[str, ...]
    completed_set_ids: tuple[str, ...]
    keep_lane_pool_alive: bool
    active_request_count: int
    request_worker_count: int
    current_generation: int
    current_max_workers: int | None
    pool_stale: bool
    has_lane_pool: bool
    cache_key: str
    warm_failure: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "active", bool(self.active))
        object.__setattr__(self, "run_id", int(self.run_id))
        object.__setattr__(self, "request_id", int(self.request_id))
        object.__setattr__(self, "fast_mode", bool(self.fast_mode))
        object.__setattr__(self, "queue_ids", tuple(str(item) for item in self.queue_ids if str(item)))
        object.__setattr__(self, "queue_names", tuple(str(item) for item in self.queue_names))
        object.__setattr__(self, "completed_set_ids", tuple(str(item) for item in self.completed_set_ids if str(item)))
        object.__setattr__(self, "keep_lane_pool_alive", bool(self.keep_lane_pool_alive))
        object.__setattr__(self, "active_request_count", int(self.active_request_count))
        object.__setattr__(self, "request_worker_count", int(self.request_worker_count))
        object.__setattr__(self, "current_generation", int(self.current_generation))
        object.__setattr__(
            self,
            "current_max_workers",
            None if self.current_max_workers is None else int(self.current_max_workers),
        )
        object.__setattr__(self, "pool_stale", bool(self.pool_stale))
        object.__setattr__(self, "has_lane_pool", bool(self.has_lane_pool))
        object.__setattr__(self, "cache_key", str(self.cache_key or ""))
        object.__setattr__(
            self,
            "warm_failure",
            None if self.warm_failure is None else str(self.warm_failure),
        )


class BatchRuntimeSession:
    def __init__(self, lane_owner: Any) -> None:
        self._lane_owner = lane_owner
        self._state = BatchRuntimeSessionState.IDLE
        self._request: BatchRuntimeSessionRequest | None = None
        self._completed_set_ids: list[str] = []

    @property
    def lane_owner(self) -> Any:
        return self._lane_owner

    @property
    def state(self) -> BatchRuntimeSessionState:
        return self._state

    @property
    def active(self) -> bool:
        return self._state is BatchRuntimeSessionState.RUNNING

    @property
    def is_pool_stale(self) -> bool:
        return bool(getattr(self._lane_owner, "is_pool_stale", False))

    @property
    def current_max_workers(self) -> int | None:
        value = getattr(self._lane_owner, "current_max_workers", None)
        return None if value is None else int(value)

    @property
    def warm_failure(self) -> str | None:
        value = getattr(self._lane_owner, "warm_failure", None)
        if callable(value):
            value = value()
        return None if value is None else str(value)

    def begin(self, request: BatchRuntimeSessionRequest) -> None:
        self._request = request
        self._completed_set_ids = []
        self._state = BatchRuntimeSessionState.RUNNING
        reset = getattr(self._lane_owner, "reset_active_run_state", None)
        if callable(reset):
            reset()

    def snapshot(self) -> BatchRuntimeSessionSnapshot:
        request = self._request
        return BatchRuntimeSessionSnapshot(
            state=self._state,
            active=self.active,
            run_id=0 if request is None else request.run_id,
            request_id=0 if request is None else request.request_id,
            fast_mode=False if request is None else request.fast_mode,
            queue_ids=() if request is None else request.queue_ids,
            queue_names=() if request is None else request.queue_names,
            completed_set_ids=tuple(self._completed_set_ids),
            keep_lane_pool_alive=False if request is None else request.keep_lane_pool_alive,
            active_request_count=self.active_request_count(),
            request_worker_count=self.request_worker_count(),
            current_generation=int(getattr(self._lane_owner, "current_generation", 0) or 0),
            current_max_workers=self.current_max_workers,
            pool_stale=self.is_pool_stale,
            has_lane_pool=self.has_lane_pool(),
            cache_key="" if request is None else request.cache_key,
            warm_failure=self.warm_failure,
        )

    def ensure_lane_pool(self, *, max_lanes: int) -> Any:
        return self._lane_owner.ensure_lane_pool(max_lanes=int(max_lanes))

    def ensure_warm_lane_pool(self, *, max_lanes: int, wait: bool = True) -> Any:
        return self._lane_owner.ensure_warm_lane_pool(max_lanes=int(max_lanes), wait=bool(wait))

    def submit_task(
        self,
        task: Mapping[str, Any],
        *,
        set_id: str,
        set_name: str,
        expected_owner_epoch: int | None = None,
        active_timeout_s: float | None = None,
    ) -> BatchRequestHandle:
        request = self._require_running_request()
        return self._lane_owner.submit_task(
            dict(task or {}),
            run_id=int(request.run_id),
            request_id=int(request.request_id),
            set_id=str(set_id or ""),
            set_name=str(set_name or set_id or ""),
            preview_owner_epoch=request.preview_owner_epoch,
            active_timeout_s=float(request.active_timeout_s if active_timeout_s is None else active_timeout_s),
            expected_owner_epoch=None if expected_owner_epoch is None else int(expected_owner_epoch),
        )

    def poll_completed_records(self) -> list[BatchPolledCompletion]:
        request = self._request
        polled = list(self._lane_owner.poll_completed_records())
        if request is None or self._state is not BatchRuntimeSessionState.RUNNING:
            return []
        accepted: list[BatchPolledCompletion] = []
        for item in polled:
            record = item.record
            if int(record.run_id) != int(request.run_id) or int(record.request_id) != int(request.request_id):
                continue
            sid = str(item.set_id or record.set_id or "")
            if not sid:
                continue
            if sid not in self._completed_set_ids:
                self._completed_set_ids.append(sid)
            accepted.append(item)
        if self._is_current_request_complete(request):
            self._state = BatchRuntimeSessionState.COMPLETED
        return accepted

    def soft_supersede_active_run(self) -> tuple[int, int]:
        cancelled, running = self._lane_owner.soft_supersede()
        self._state = BatchRuntimeSessionState.SUPERSEDED
        return int(cancelled), int(running)

    def finish_after_run(
        self,
        *,
        keep_lane_pool_alive: bool,
        record_nonfatal_exception: Callable[[str, BaseException], None],
    ) -> None:
        if bool(keep_lane_pool_alive) and not self.is_pool_stale:
            self._lane_owner.reset_active_run_state()
            self._state = BatchRuntimeSessionState.IDLE
            return
        self.shutdown(force_terminate=False, record_nonfatal_exception=record_nonfatal_exception)

    def mark_pool_stale(self) -> None:
        self._lane_owner.mark_pool_stale()

    def shutdown(
        self,
        *,
        force_terminate: bool,
        record_nonfatal_exception: Callable[[str, BaseException], None],
    ) -> None:
        self._lane_owner.shutdown(
            force_terminate=bool(force_terminate),
            record_nonfatal_exception=record_nonfatal_exception,
        )
        self._state = BatchRuntimeSessionState.SHUTDOWN
        self._completed_set_ids = []

    def has_lane_pool(self) -> bool:
        has_lane_pool = getattr(self._lane_owner, "has_lane_pool", None)
        return bool(has_lane_pool()) if callable(has_lane_pool) else False

    def has_ready_lane_pool(self, *, max_lanes: int) -> bool:
        has_ready_lane_pool = getattr(self._lane_owner, "has_ready_lane_pool", None)
        return bool(has_ready_lane_pool(max_lanes=int(max_lanes))) if callable(has_ready_lane_pool) else False

    def active_request_count(self) -> int:
        active_request_count = getattr(self._lane_owner, "active_request_count", None)
        return int(active_request_count()) if callable(active_request_count) else 0

    def has_active_requests(self) -> bool:
        has_active_requests = getattr(self._lane_owner, "has_active_requests", None)
        return bool(has_active_requests()) if callable(has_active_requests) else bool(self.active_request_count())

    def request_worker_count(self) -> int:
        request_worker_count = getattr(self._lane_owner, "request_worker_count", None)
        return int(request_worker_count()) if callable(request_worker_count) else 0

    def join_active_requests(self, *, timeout_s: float = 2.0) -> None:
        join = getattr(self._lane_owner, "join_active_requests", None)
        if callable(join):
            join(timeout_s=float(timeout_s))

    def lane_pool_token(self) -> int | None:
        lane_pool_token = getattr(self._lane_owner, "lane_pool_token", None)
        if callable(lane_pool_token):
            value = lane_pool_token()
            return None if value is None else int(value)
        return None

    def drain_completion_queue(self) -> None:
        drain = getattr(self._lane_owner, "drain_completion_queue", None)
        if callable(drain):
            drain()

    def enqueue_completion(self, set_id: str) -> None:
        enqueue = getattr(self._lane_owner, "enqueue_completion", None)
        if callable(enqueue):
            enqueue(str(set_id or ""))

    def clear_stale_requests(self) -> None:
        clear = getattr(self._lane_owner, "clear_stale_requests", None)
        if callable(clear):
            clear()

    def active_request_metadata(self, set_id: str) -> dict[str, Any]:
        metadata = getattr(self._lane_owner, "active_request_metadata", None)
        return dict(metadata(str(set_id or ""))) if callable(metadata) else {}

    def discard_request(self, set_id: str) -> None:
        discard = getattr(self._lane_owner, "discard_request", None)
        if callable(discard):
            discard(str(set_id or ""))

    def _require_running_request(self) -> BatchRuntimeSessionRequest:
        request = self._request
        if request is None or self._state is not BatchRuntimeSessionState.RUNNING:
            raise RuntimeError("Batch runtime session is not running.")
        return request

    def _is_current_request_complete(self, request: BatchRuntimeSessionRequest) -> bool:
        queue_ids = _normalize_ids(request.queue_ids)
        if queue_ids and not queue_ids.issubset(set(self._completed_set_ids)):
            return False
        return not self.has_active_requests()


def _normalize_ids(values: Iterable[str]) -> set[str]:
    return {str(item) for item in values if str(item)}
