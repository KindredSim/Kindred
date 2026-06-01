from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

from kindred.core.batch_containment import BatchCompletionRecord, BatchPolledCompletion, BatchRequestHandle


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


class BatchRuntimeLaneOwnerProtocol(Protocol):
    current_generation: int
    current_max_workers: int | None
    is_pool_stale: bool
    lane_pool_factory: Callable[[int, bool], Any]
    limit_blas_threads_per_worker: bool
    max_parallel_workers: int
    record_nonfatal_exception: Callable[[str, BaseException], None]
    warm_failure: str | None

    def active_request_count(self) -> int: ...
    def active_request_metadata(self, set_id: str) -> dict[str, Any]: ...
    def clear_stale_requests(self) -> None: ...
    def consume_warm_failure(self) -> str | None: ...
    def discard_request(self, set_id: str) -> None: ...
    def drain_completion_queue(self) -> None: ...
    def enqueue_completion(self, set_id: str) -> None: ...
    def ensure_lane_pool(self, *, max_lanes: int) -> Any: ...
    def ensure_warm_lane_pool(self, *, max_lanes: int, wait: bool = True) -> Any: ...
    def has_lane_pool(self) -> bool: ...
    def has_ready_lane_pool(self, *, max_lanes: int) -> bool: ...
    def has_active_requests(self) -> bool: ...
    def join_active_requests(self, *, timeout_s: float = 2.0) -> None: ...
    def lane_pool_token(self) -> int | None: ...
    def mark_pool_stale(self) -> None: ...
    def poll_completed_records(self) -> list[BatchPolledCompletion]: ...
    def request_worker_count(self) -> int: ...
    def reset_active_run_state(self) -> None: ...
    def reset_run_state(self) -> None: ...
    def shutdown(
        self,
        *,
        force_terminate: bool,
        record_nonfatal_exception: Callable[[str, BaseException], None],
    ) -> None: ...
    def soft_supersede(self) -> tuple[int, int]: ...
    def submit_task(
        self,
        task: Any,
        *,
        run_id: int,
        request_id: int,
        set_id: str,
        set_name: str,
        preview_owner_epoch: int | None,
        active_timeout_s: float,
        request_metadata: Mapping[str, Any] | None = None,
    ) -> BatchRequestHandle: ...


class BatchRuntimeSession:
    def __init__(self, lane_owner: BatchRuntimeLaneOwnerProtocol) -> None:
        self._lane_owner = lane_owner
        self._state = BatchRuntimeSessionState.IDLE
        self._request: BatchRuntimeSessionRequest | None = None
        self._completed_set_ids: list[str] = []

    @property
    def state(self) -> BatchRuntimeSessionState:
        return self._state

    @property
    def active(self) -> bool:
        return self._state is BatchRuntimeSessionState.RUNNING

    @property
    def is_pool_stale(self) -> bool:
        return bool(self._lane_owner.is_pool_stale)

    @property
    def current_max_workers(self) -> int | None:
        value = self._lane_owner.current_max_workers
        return None if value is None else int(value)

    @property
    def warm_failure(self) -> str | None:
        value = self._lane_owner.warm_failure
        return None if value is None else str(value)

    def consume_warm_failure(self) -> str | None:
        value = self._lane_owner.consume_warm_failure()
        return None if value is None else str(value)

    @property
    def lane_pool_factory(self) -> Callable[[int, bool], Any]:
        return self._lane_owner.lane_pool_factory

    @lane_pool_factory.setter
    def lane_pool_factory(self, value: Callable[[int, bool], Any]) -> None:
        previous = self._lane_owner.lane_pool_factory
        self._lane_owner.lane_pool_factory = value
        if value is not previous and self.has_lane_pool():
            self.mark_pool_stale()

    @property
    def max_parallel_workers(self) -> int:
        return int(self._lane_owner.max_parallel_workers)

    @max_parallel_workers.setter
    def max_parallel_workers(self, value: int) -> None:
        self._lane_owner.max_parallel_workers = int(value)

    @property
    def limit_blas_threads_per_worker(self) -> bool:
        return bool(self._lane_owner.limit_blas_threads_per_worker)

    @limit_blas_threads_per_worker.setter
    def limit_blas_threads_per_worker(self, value: bool) -> None:
        self._lane_owner.limit_blas_threads_per_worker = bool(value)

    @property
    def record_nonfatal_exception(self) -> Callable[[str, BaseException], None]:
        return self._lane_owner.record_nonfatal_exception

    @record_nonfatal_exception.setter
    def record_nonfatal_exception(self, value: Callable[[str, BaseException], None]) -> None:
        self._lane_owner.record_nonfatal_exception = value

    def reconfigure_runtime_pool(
        self,
        *,
        max_parallel_workers: int,
        limit_blas_threads_per_worker: bool,
        lane_pool_factory: Callable[[int, bool], Any],
        record_nonfatal_exception: Callable[[str, BaseException], None],
    ) -> None:
        changed = False
        if int(self._lane_owner.max_parallel_workers) != int(max_parallel_workers):
            self._lane_owner.max_parallel_workers = int(max_parallel_workers)
            changed = True
        if bool(self._lane_owner.limit_blas_threads_per_worker) != bool(limit_blas_threads_per_worker):
            self._lane_owner.limit_blas_threads_per_worker = bool(limit_blas_threads_per_worker)
            changed = True
        if self._lane_owner.lane_pool_factory is not lane_pool_factory:
            self._lane_owner.lane_pool_factory = lane_pool_factory
            changed = True
        if self._lane_owner.record_nonfatal_exception is not record_nonfatal_exception:
            self._lane_owner.record_nonfatal_exception = record_nonfatal_exception
        if changed and self.has_lane_pool():
            self.mark_pool_stale()

    def begin(self, request: BatchRuntimeSessionRequest) -> None:
        self._request = request
        self._completed_set_ids = []
        self._state = BatchRuntimeSessionState.RUNNING
        self._lane_owner.reset_active_run_state()

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
            current_generation=int(self._lane_owner.current_generation or 0),
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
        task: Any,
        *,
        set_id: str,
        set_name: str,
        active_timeout_s: float | None = None,
        callback_identity: Any,
    ) -> BatchRequestHandle:
        request = self._require_running_request()
        if callback_identity is None:
            raise ValueError("Batch runtime session task submission requires callback_identity.")
        _require_runtime_task_executable_payload(task)
        request_metadata = _runtime_task_request_metadata(task)
        request_metadata["callback_identity"] = callback_identity
        return self._lane_owner.submit_task(
            task,
            run_id=int(request.run_id),
            request_id=int(request.request_id),
            set_id=str(set_id or ""),
            set_name=str(set_name or set_id or ""),
            preview_owner_epoch=request.preview_owner_epoch,
            active_timeout_s=float(request.active_timeout_s if active_timeout_s is None else active_timeout_s),
            request_metadata=request_metadata,
        )

    def poll_completed_records(self) -> list[BatchPolledCompletion]:
        request = self._request
        polled = list(self._lane_owner.poll_completed_records())
        if request is None or self._state is not BatchRuntimeSessionState.RUNNING:
            return []
        accepted: list[BatchPolledCompletion] = []
        for item in polled:
            record = item.record
            sid = str(item.set_id or record.set_id or "")
            if not sid:
                continue
            is_current_session_record = (
                int(record.run_id) == int(request.run_id)
                and int(record.request_id) == int(request.request_id)
                and (
                    request.preview_owner_epoch is None
                    or (
                        record.preview_owner_epoch is not None
                        and int(record.preview_owner_epoch) == int(request.preview_owner_epoch)
                    )
                )
            )
            if not is_current_session_record:
                stale_metadata = dict(record.request_metadata or {})
                stale_metadata["runtime_session_stale"] = {
                    "expected_run_id": int(request.run_id),
                    "expected_request_id": int(request.request_id),
                    "expected_preview_owner_epoch": request.preview_owner_epoch,
                    "actual_run_id": int(record.run_id),
                    "actual_request_id": int(record.request_id),
                    "actual_preview_owner_epoch": record.preview_owner_epoch,
                }
                record = BatchCompletionRecord(
                    metadata=record.metadata,
                    outcome=record.outcome,
                    completed_ts=record.completed_ts,
                    request_metadata=stale_metadata,
                )
                accepted.append(
                    BatchPolledCompletion(
                        set_id=sid,
                        record=record,
                        source=item.source,
                        completed_ts=item.completed_ts,
                    )
                )
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

    def reset_active_run_state(self) -> None:
        self._lane_owner.reset_active_run_state()

    def reset_run_state(self) -> None:
        self._lane_owner.reset_run_state()
        self._completed_set_ids = []
        self._state = BatchRuntimeSessionState.IDLE

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
        return bool(self._lane_owner.has_lane_pool())

    def has_ready_lane_pool(self, *, max_lanes: int) -> bool:
        return bool(self._lane_owner.has_ready_lane_pool(max_lanes=int(max_lanes)))

    def active_request_count(self) -> int:
        return int(self._lane_owner.active_request_count())

    def has_active_requests(self) -> bool:
        return bool(self._lane_owner.has_active_requests())

    def request_worker_count(self) -> int:
        return int(self._lane_owner.request_worker_count())

    def join_active_requests(self, *, timeout_s: float = 2.0) -> None:
        self._lane_owner.join_active_requests(timeout_s=float(timeout_s))

    def lane_pool_token(self) -> int | None:
        value = self._lane_owner.lane_pool_token()
        return None if value is None else int(value)

    def drain_completion_queue(self) -> None:
        self._lane_owner.drain_completion_queue()

    def enqueue_completion(self, set_id: str) -> None:
        self._lane_owner.enqueue_completion(str(set_id or ""))

    def clear_stale_requests(self) -> None:
        self._lane_owner.clear_stale_requests()

    def active_request_metadata(self, set_id: str) -> dict[str, Any]:
        return dict(self._lane_owner.active_request_metadata(str(set_id or "")))

    def discard_request(self, set_id: str) -> None:
        self._lane_owner.discard_request(str(set_id or ""))

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


def _runtime_task_request_metadata(task: Any) -> dict[str, Any]:
    request_metadata = getattr(task, "request_metadata", None)
    if not callable(request_metadata):
        raise TypeError("Batch runtime session requires a runtime task with request_metadata().")
    try:
        raw = request_metadata()
    except Exception as exc:
        raise TypeError("Batch runtime session could not read runtime task metadata.") from exc
    if not isinstance(raw, Mapping):
        raise TypeError("Batch runtime session requires runtime task metadata to be a mapping.")
    return dict(raw)


def _require_runtime_task_executable_payload(task: Any) -> None:
    executable_payload = getattr(task, "executable_payload", None)
    if not callable(executable_payload):
        raise TypeError("Batch runtime session requires a runtime task with executable_payload().")
