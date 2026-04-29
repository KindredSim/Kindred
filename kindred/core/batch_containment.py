from __future__ import annotations

import multiprocessing
import threading
import traceback
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from queue import Empty, Queue, SimpleQueue
from time import perf_counter
from typing import Any, Optional

from kindred.core.containment_kernel import (
    ContainmentKernelAcceptTimeout,
    ContainmentKernelActiveTimeout,
    ContainmentKernelCancelled,
    ContainmentKernelChildFailure,
    ContainmentKernelEvent,
    ContainmentKernelProtocolError,
    ContainmentKernelStartupTimeout,
)
from kindred.core.simulation_runtime_service import SimulationRuntimeOwner

_DEFAULT_BATCH_HANDLER_IMPORT_PATH = "kindred.core.batch_containment:make_batch_simulation_handler"
_DEFAULT_READY_TIMEOUT_S = 30.0
_DEFAULT_ACCEPT_TIMEOUT_S = 10.0
_DEFAULT_ACTIVE_TIMEOUT_S = 60.0
_DEFAULT_EVENT_HISTORY_LIMIT = 256

_BLAS_THREAD_ENV_VARS: tuple[str, ...] = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


@dataclass(frozen=True)
class BatchLaneOutcome:
    lane_id: str
    run_id: int
    request_id: int
    set_id: str
    owner_epoch: int
    success: bool
    payload: Mapping[str, Any] | None = None
    failure: Mapping[str, Any] | None = None
    events: tuple[ContainmentKernelEvent, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "lane_id", str(self.lane_id))
        object.__setattr__(self, "run_id", int(self.run_id))
        object.__setattr__(self, "request_id", int(self.request_id))
        object.__setattr__(self, "set_id", str(self.set_id))
        object.__setattr__(self, "owner_epoch", int(self.owner_epoch))
        object.__setattr__(self, "success", bool(self.success))
        object.__setattr__(self, "payload", None if self.payload is None else dict(self.payload))
        object.__setattr__(self, "failure", None if self.failure is None else dict(self.failure))
        object.__setattr__(self, "events", tuple(self.events or ()))


@dataclass(frozen=True)
class BatchRequestMetadata:
    set_id: str
    set_name: str
    run_id: int
    request_id: int
    generation: int
    preview_owner_epoch: int | None = None
    expected_owner_epoch: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "set_id", str(self.set_id or ""))
        object.__setattr__(self, "set_name", str(self.set_name or self.set_id or ""))
        object.__setattr__(self, "run_id", int(self.run_id))
        object.__setattr__(self, "request_id", int(self.request_id))
        object.__setattr__(self, "generation", int(self.generation))
        object.__setattr__(
            self,
            "preview_owner_epoch",
            None if self.preview_owner_epoch is None else int(self.preview_owner_epoch),
        )
        object.__setattr__(
            self,
            "expected_owner_epoch",
            None if self.expected_owner_epoch is None else int(self.expected_owner_epoch),
        )


@dataclass(frozen=True)
class BatchCompletionRecord:
    metadata: BatchRequestMetadata
    outcome: BatchLaneOutcome
    completed_ts: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "completed_ts", float(self.completed_ts))

    @property
    def set_id(self) -> str:
        return self.metadata.set_id

    @property
    def set_name(self) -> str:
        return self.metadata.set_name

    @property
    def run_id(self) -> int:
        return self.metadata.run_id

    @property
    def request_id(self) -> int:
        return self.metadata.request_id

    @property
    def generation(self) -> int:
        return self.metadata.generation

    @property
    def preview_owner_epoch(self) -> int | None:
        return self.metadata.preview_owner_epoch

    @property
    def expected_owner_epoch(self) -> int | None:
        return self.metadata.expected_owner_epoch


@dataclass(frozen=True)
class BatchPolledCompletion:
    set_id: str
    record: BatchCompletionRecord
    source: str
    completed_ts: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "set_id", str(self.set_id or ""))
        object.__setattr__(self, "source", str(self.source or ""))
        object.__setattr__(self, "completed_ts", float(self.completed_ts))


class _BatchSimulationHandler:
    def __init__(self, _startup_payload: Mapping[str, Any]) -> None:
        from kindred.core.batch_parallel import prewarm_worker_imports, run_batch_simulation_task

        prewarm_worker_imports()
        self._run_batch_simulation_task = run_batch_simulation_task

    def handle_request(self, payload: Mapping[str, Any], _context: Any) -> dict[str, Any]:
        return self._run_batch_simulation_task(dict(payload or {}))


def make_batch_simulation_handler(startup_payload: Mapping[str, Any]) -> _BatchSimulationHandler:
    return _BatchSimulationHandler(dict(startup_payload or {}))


def _batch_handler_env(*, limit_blas_threads_per_worker: bool) -> dict[str, str]:
    if not bool(limit_blas_threads_per_worker):
        return {}
    return {name: "1" for name in _BLAS_THREAD_ENV_VARS}


def _format_exception(exc: BaseException) -> str:
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)).strip()


def _failure(
    *,
    kind: str,
    phase: str,
    message: str,
    exc: BaseException | None = None,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "kind": str(kind),
        "phase": str(phase),
        "message": str(message),
    }
    if exc is not None:
        payload["exc_type"] = type(exc).__name__
        stack_trace = _format_exception(exc)
        if stack_trace:
            payload["context"] = {"stack_trace": stack_trace}
    if details:
        payload["details"] = dict(details)
    return payload


class WarmBatchSimulationLane:
    def __init__(
        self,
        *,
        lane_id: str,
        handler_import_path: str = _DEFAULT_BATCH_HANDLER_IMPORT_PATH,
        startup_payload: Mapping[str, Any] | None = None,
        limit_blas_threads_per_worker: bool = True,
        ready_timeout_s: float = _DEFAULT_READY_TIMEOUT_S,
        accept_timeout_s: float = _DEFAULT_ACCEPT_TIMEOUT_S,
        event_history_limit: int = _DEFAULT_EVENT_HISTORY_LIMIT,
        mp_context: Optional[multiprocessing.context.BaseContext] = None,
    ) -> None:
        self._lane_id = str(lane_id or "")
        if not self._lane_id:
            raise ValueError("Batch lane id must not be empty.")
        self._event_history_limit = max(1, int(event_history_limit))
        self._events: deque[ContainmentKernelEvent] = deque(maxlen=self._event_history_limit)
        self._lock = threading.Lock()
        self._owner = SimulationRuntimeOwner(
            handler_import_path=str(handler_import_path),
            startup_payload=dict(startup_payload or {}),
            handler_env=_batch_handler_env(limit_blas_threads_per_worker=bool(limit_blas_threads_per_worker)),
            ready_timeout_s=float(ready_timeout_s),
            accept_timeout_s=float(accept_timeout_s),
            mp_context=mp_context,
        )

    @property
    def lane_id(self) -> str:
        return str(self._lane_id)

    @property
    def owner_epoch(self) -> int:
        return int(self._owner.owner_epoch)

    @property
    def is_running(self) -> bool:
        return bool(self._owner.is_running)

    @property
    def is_ready(self) -> bool:
        return bool(getattr(self._owner, "is_ready", False))

    def warm(self, *, wait: bool = True) -> None:
        with self._lock:
            self._owner.warm(wait=bool(wait))
            self._extend_events(self._owner.drain_events())

    @property
    def is_busy(self) -> bool:
        locked = getattr(self._lock, "locked", None)
        return bool(locked()) if callable(locked) else False

    def reserve(self) -> bool:
        return bool(self._lock.acquire(blocking=False))

    def release_reservation(self) -> None:
        self._lock.release()

    def drain_events(self) -> list[ContainmentKernelEvent]:
        events = list(self._events)
        self._events.clear()
        return events

    def diagnostic_events(self) -> tuple[ContainmentKernelEvent, ...]:
        return tuple(self._events)

    def run(
        self,
        task: Mapping[str, Any],
        *,
        run_id: int,
        request_id: int,
        set_id: str,
        active_timeout_s: float = _DEFAULT_ACTIVE_TIMEOUT_S,
    ) -> BatchLaneOutcome:
        with self._lock:
            return self._run_locked(
                dict(task or {}),
                run_id=int(run_id),
                request_id=int(request_id),
                set_id=str(set_id or ""),
                active_timeout_s=float(active_timeout_s),
            )

    def run_reserved(
        self,
        task: Mapping[str, Any],
        *,
        run_id: int,
        request_id: int,
        set_id: str,
        active_timeout_s: float = _DEFAULT_ACTIVE_TIMEOUT_S,
    ) -> BatchLaneOutcome:
        try:
            return self._run_locked(
                dict(task or {}),
                run_id=int(run_id),
                request_id=int(request_id),
                set_id=str(set_id or ""),
                active_timeout_s=float(active_timeout_s),
            )
        finally:
            self.release_reservation()

    def close(self, *, kill: bool = False) -> None:
        if bool(kill):
            self._owner.close(kill=True)
            if self._lock.acquire(blocking=False):
                try:
                    self._extend_events(self._owner.drain_events())
                finally:
                    self._lock.release()
            return
        if not self._lock.acquire(blocking=False):
            self._owner.close(kill=True)
            return
        try:
            self._owner.close(kill=False)
            self._extend_events(self._owner.drain_events())
        finally:
            self._lock.release()

    def _run_locked(
        self,
        task: Mapping[str, Any],
        *,
        run_id: int,
        request_id: int,
        set_id: str,
        active_timeout_s: float,
    ) -> BatchLaneOutcome:
        rid = int(run_id)
        req_id = int(request_id)
        sid = str(set_id or "")
        payload = dict(task or {})
        payload.setdefault("run_id", rid)
        payload.setdefault("request_id", req_id)
        payload.setdefault("set_id", sid)
        owner_epoch = int(self._owner.owner_epoch)
        events: list[ContainmentKernelEvent] = []
        try:
            result = self._owner.solve(
                payload,
                active_timeout_s=float(active_timeout_s),
                reply_fields={"run_id": rid, "batch_request_id": req_id, "set_id": sid},
            )
            owner_epoch = int(self._owner.owner_epoch)
            events = self._drain_owner_events()
            return BatchLaneOutcome(
                lane_id=self._lane_id,
                run_id=rid,
                request_id=req_id,
                set_id=sid,
                owner_epoch=owner_epoch,
                success=True,
                payload=result,
                events=tuple(events),
            )
        except ContainmentKernelStartupTimeout as exc:
            owner_epoch = max(owner_epoch, int(self._owner.owner_epoch))
            events = self._drain_owner_events()
            return self._failed_outcome(
                run_id=rid,
                request_id=req_id,
                set_id=sid,
                owner_epoch=owner_epoch,
                events=events,
                failure=_failure(kind="startup_timeout", phase="startup", message=str(exc), exc=exc),
            )
        except ContainmentKernelAcceptTimeout as exc:
            owner_epoch = max(owner_epoch, int(self._owner.owner_epoch))
            events = self._drain_owner_events()
            return self._failed_outcome(
                run_id=rid,
                request_id=req_id,
                set_id=sid,
                owner_epoch=owner_epoch,
                events=events,
                failure=_failure(kind="accept_timeout", phase="accept", message=str(exc), exc=exc),
            )
        except ContainmentKernelActiveTimeout as exc:
            owner_epoch = max(owner_epoch, int(self._owner.owner_epoch))
            events = self._drain_owner_events()
            return self._failed_outcome(
                run_id=rid,
                request_id=req_id,
                set_id=sid,
                owner_epoch=owner_epoch,
                events=events,
                failure=_failure(kind="active_timeout", phase="active", message=str(exc), exc=exc),
            )
        except ContainmentKernelCancelled as exc:
            owner_epoch = max(owner_epoch, int(self._owner.owner_epoch))
            events = self._drain_owner_events()
            return self._failed_outcome(
                run_id=rid,
                request_id=req_id,
                set_id=sid,
                owner_epoch=owner_epoch,
                events=events,
                failure=_failure(kind="cancelled", phase="internal", message=str(exc) or "Contained request cancelled.", exc=exc),
            )
        except ContainmentKernelChildFailure as exc:
            owner_epoch = max(owner_epoch, int(self._owner.owner_epoch))
            events = self._drain_owner_events()
            raw_failure = dict(exc.failure or {})
            raw_failure.setdefault("kind", "child_failure")
            raw_failure.setdefault("phase", "child")
            raw_failure.setdefault("message", str(exc) or "Contained child failed.")
            return self._failed_outcome(
                run_id=rid,
                request_id=req_id,
                set_id=sid,
                owner_epoch=owner_epoch,
                events=events,
                failure=raw_failure,
            )
        except ContainmentKernelProtocolError as exc:
            owner_epoch = max(owner_epoch, int(self._owner.owner_epoch))
            events = self._drain_owner_events()
            self._owner.close(kill=True)
            return self._failed_outcome(
                run_id=rid,
                request_id=req_id,
                set_id=sid,
                owner_epoch=owner_epoch,
                events=events,
                failure=_failure(kind="protocol_error", phase="protocol", message=str(exc), exc=exc),
            )
        except BaseException as exc:  # noqa: BLE001 - lane boundary must return structured failure
            owner_epoch = max(owner_epoch, int(self._owner.owner_epoch))
            events = self._drain_owner_events()
            self._owner.close(kill=True)
            return self._failed_outcome(
                run_id=rid,
                request_id=req_id,
                set_id=sid,
                owner_epoch=owner_epoch,
                events=events,
                failure=_failure(kind="internal_error", phase="internal", message=str(exc), exc=exc),
            )

    def _failed_outcome(
        self,
        *,
        run_id: int,
        request_id: int,
        set_id: str,
        owner_epoch: int,
        events: list[ContainmentKernelEvent],
        failure: Mapping[str, Any],
    ) -> BatchLaneOutcome:
        return BatchLaneOutcome(
            lane_id=self._lane_id,
            run_id=int(run_id),
            request_id=int(request_id),
            set_id=str(set_id),
            owner_epoch=int(owner_epoch),
            success=False,
            failure=dict(failure),
            events=tuple(events),
        )

    def _drain_owner_events(self) -> list[ContainmentKernelEvent]:
        events = self._owner.drain_events()
        self._extend_events(events)
        return list(events)

    def _extend_events(self, events: list[ContainmentKernelEvent]) -> None:
        for event in events:
            self._events.append(event)


class BatchLanePool:
    def __init__(
        self,
        *,
        max_lanes: int,
        handler_import_path: str = _DEFAULT_BATCH_HANDLER_IMPORT_PATH,
        startup_payload: Mapping[str, Any] | None = None,
        limit_blas_threads_per_worker: bool = True,
        ready_timeout_s: float = _DEFAULT_READY_TIMEOUT_S,
        accept_timeout_s: float = _DEFAULT_ACCEPT_TIMEOUT_S,
        event_history_limit: int = _DEFAULT_EVENT_HISTORY_LIMIT,
        mp_context: Optional[multiprocessing.context.BaseContext] = None,
    ) -> None:
        self._max_lanes = max(1, int(max_lanes))
        self._handler_import_path = str(handler_import_path)
        self._startup_payload = dict(startup_payload or {})
        self._limit_blas_threads_per_worker = bool(limit_blas_threads_per_worker)
        self._ready_timeout_s = float(ready_timeout_s)
        self._accept_timeout_s = float(accept_timeout_s)
        self._event_history_limit = max(1, int(event_history_limit))
        self._mp_context = mp_context
        self._lanes: list[WarmBatchSimulationLane] = []
        self._next_lane_index = 0
        self._events: deque[ContainmentKernelEvent] = deque(maxlen=self._event_history_limit)
        self._lock = threading.Lock()
        self._closed = False

    @property
    def retained_lane_count(self) -> int:
        return len(self._lanes)

    @property
    def ready_lane_count(self) -> int:
        return int(
            len(
                [
                    lane
                    for lane in list(self._lanes)
                    if bool(getattr(lane, "is_ready", False))
                ]
            )
        )

    def diagnostic_events(self) -> tuple[ContainmentKernelEvent, ...]:
        return tuple(self._events)

    def run(
        self,
        task: Mapping[str, Any],
        *,
        run_id: int,
        request_id: int,
        set_id: str,
        active_timeout_s: float = _DEFAULT_ACTIVE_TIMEOUT_S,
    ) -> BatchLaneOutcome:
        lane, reserved = self._reserve_next_lane()
        run_method = lane.run_reserved if reserved else lane.run
        outcome = run_method(
            dict(task or {}),
            run_id=int(run_id),
            request_id=int(request_id),
            set_id=str(set_id or ""),
            active_timeout_s=float(active_timeout_s),
        )
        for event in outcome.events:
            self._events.append(event)
        return outcome

    def warm_lanes(self, max_lanes: int, *, wait: bool = True) -> None:
        requested_lanes = min(self._max_lanes, max(1, int(max_lanes)))
        with self._lock:
            if self._closed:
                raise RuntimeError("Batch lane pool is closed.")
            while len(self._lanes) < requested_lanes:
                self._lanes.append(self._create_lane_unlocked())
            lanes = tuple(self._lanes[:requested_lanes])
        for lane in lanes:
            lane.warm(wait=bool(wait))
            for event in lane.drain_events():
                self._events.append(event)

    def close(self, *, kill: bool = False) -> None:
        with self._lock:
            self._closed = True
        for lane in list(self._lanes):
            lane.close(kill=bool(kill))
            for event in lane.drain_events():
                self._events.append(event)

    def _next_lane(self) -> WarmBatchSimulationLane:
        lane, reserved = self._reserve_next_lane()
        if reserved:
            release = getattr(lane, "release_reservation", None)
            if callable(release):
                release()
        return lane

    def _reserve_next_lane(self) -> tuple[WarmBatchSimulationLane, bool]:
        with self._lock:
            if self._closed:
                raise RuntimeError("Batch lane pool is closed.")
            lane_count = len(self._lanes)
            for offset in range(lane_count):
                index = (self._next_lane_index + offset) % lane_count
                candidate = self._lanes[index]
                reserved = self._try_reserve_lane(candidate)
                if reserved or self._lane_is_available(candidate):
                    self._next_lane_index = (index + 1) % lane_count
                    return candidate, reserved
            if len(self._lanes) < self._max_lanes:
                lane = self._create_lane_unlocked()
                self._lanes.append(lane)
                return lane, self._try_reserve_lane(lane)
            lane = self._lanes[self._next_lane_index % lane_count]
            self._next_lane_index = (self._next_lane_index + 1) % lane_count
            return lane, False

    def _create_lane_unlocked(self) -> WarmBatchSimulationLane:
        return WarmBatchSimulationLane(
            lane_id=f"batch-lane-{len(self._lanes) + 1}",
            handler_import_path=self._handler_import_path,
            startup_payload=dict(self._startup_payload),
            limit_blas_threads_per_worker=bool(self._limit_blas_threads_per_worker),
            ready_timeout_s=float(self._ready_timeout_s),
            accept_timeout_s=float(self._accept_timeout_s),
            event_history_limit=int(self._event_history_limit),
            mp_context=self._mp_context,
        )

    @staticmethod
    def _try_reserve_lane(lane: WarmBatchSimulationLane) -> bool:
        reserve = getattr(lane, "reserve", None)
        if not callable(reserve):
            return False
        try:
            return bool(reserve())
        except Exception:
            return False

    @staticmethod
    def _lane_is_available(lane: WarmBatchSimulationLane) -> bool:
        is_busy = getattr(lane, "is_busy", None)
        if is_busy is not None:
            return not bool(is_busy() if callable(is_busy) else is_busy)
        lock = getattr(lane, "_lock", None)
        locked = getattr(lock, "locked", None)
        if callable(locked):
            return not bool(locked())
        is_running = getattr(lane, "is_running", None)
        if is_running is not None:
            return not bool(is_running() if callable(is_running) else is_running)
        return False


class BatchRequestHandle:
    def __init__(self, metadata: BatchRequestMetadata) -> None:
        self.metadata = metadata
        self.set_id = metadata.set_id
        self.set_name = metadata.set_name
        self.run_id = metadata.run_id
        self.request_id = metadata.request_id
        self.preview_owner_epoch = metadata.preview_owner_epoch
        self.completed_ts: Optional[float] = None
        self.outcome: Optional[BatchLaneOutcome] = None
        self.superseded = False
        self._done = threading.Event()

    def is_done(self) -> bool:
        return bool(self._done.is_set())

    def join(self, timeout: float | None = None) -> None:
        self._done.wait(timeout=timeout)

    def mark_superseded(self) -> None:
        self.superseded = True

    def finish(self, outcome: BatchLaneOutcome | None) -> None:
        self.outcome = outcome
        self.completed_ts = float(perf_counter())
        self._done.set()


class BatchRuntimeLaneOwner:
    def __init__(
        self,
        *,
        lane_pool_factory: Callable[[int, bool], Any],
        max_parallel_workers: int,
        limit_blas_threads_per_worker: bool,
        record_nonfatal_exception: Callable[[str, BaseException], None],
        lane_pool: Any = None,
    ) -> None:
        self.lane_pool_factory = lane_pool_factory
        self.max_parallel_workers = int(max_parallel_workers)
        self.limit_blas_threads_per_worker = bool(limit_blas_threads_per_worker)
        self.record_nonfatal_exception = record_nonfatal_exception
        self.lane_pool = lane_pool
        self.active_request_map: dict[str, BatchRequestHandle] = {}
        self.active_request_meta: dict[str, dict[str, Any]] = {}
        self.completed_outcome_map: dict[str, BatchCompletionRecord] = {}
        self.superseded_request_map: dict[str, BatchRequestHandle] = {}
        self.completed_queue: SimpleQueue[tuple[str, float]] = SimpleQueue()
        self._current_max_workers: Optional[int] = None
        self._pool_stale = False
        self._lock = threading.Lock()
        self._work_queue: Queue[Any] | None = None
        self._worker_threads: list[threading.Thread] = []
        self._retired_worker_threads: list[threading.Thread] = []
        self._warm_thread: threading.Thread | None = None
        self._warm_requested_max_lanes = 0
        self._generation = 0
        self._warm_failure: str | None = None
        self._shutdown_requested = False

    @property
    def current_generation(self) -> int:
        return int(self._generation)

    @property
    def current_max_workers(self) -> Optional[int]:
        return self._current_max_workers

    @current_max_workers.setter
    def current_max_workers(self, value: Optional[int]) -> None:
        self._current_max_workers = None if value is None else int(value)

    @property
    def is_pool_stale(self) -> bool:
        return bool(self._pool_stale)

    @property
    def warm_failure(self) -> str | None:
        with self._lock:
            return self._warm_failure

    def mark_pool_stale(self) -> None:
        self._pool_stale = True

    def active_request_count(self) -> int:
        with self._lock:
            self._prune_completed_superseded_locked()
            return int(len(self.active_request_map or {}) + len(self.superseded_request_map or {}))

    def has_active_requests(self) -> bool:
        return bool(self.active_request_count())

    def request_worker_count(self) -> int:
        self._prune_retired_worker_threads()
        return int(
            len([thread for thread in self._worker_threads if thread.is_alive()])
            + len([thread for thread in self._retired_worker_threads if thread.is_alive()])
        )

    def _prune_retired_worker_threads(self) -> None:
        self._retired_worker_threads = [
            thread for thread in self._retired_worker_threads if thread.is_alive()
        ]

    def join_active_requests(self, *, timeout_s: float = 2.0) -> None:
        with self._lock:
            handles = list((self.active_request_map or {}).values()) + list(
                (self.superseded_request_map or {}).values()
            )
        for handle in handles:
            join = getattr(handle, "join", None)
            if callable(join):
                join(timeout=max(0.0, float(timeout_s)))

    def has_lane_pool(self) -> bool:
        return self.lane_pool is not None

    def has_ready_lane_pool(self, *, max_lanes: int) -> bool:
        pool = self.lane_pool
        if pool is None or self._pool_stale:
            return False
        if self._current_max_workers is None or int(self._current_max_workers) < int(max_lanes):
            return False
        ready_lane_count = getattr(pool, "ready_lane_count", None)
        if ready_lane_count is None:
            warm_lanes = getattr(pool, "warm_lanes", None)
            return not callable(warm_lanes)
        try:
            value = ready_lane_count() if callable(ready_lane_count) else ready_lane_count
            return int(value) >= max(1, int(max_lanes))
        except Exception:
            return False

    def lane_pool_token(self) -> int | None:
        pool = self.lane_pool
        return None if pool is None else int(id(pool))

    def active_request_metadata(self, set_id: str) -> dict[str, Any]:
        sid = str(set_id or "")
        with self._lock:
            return dict((self.active_request_meta or {}).get(sid) or {})

    def discard_request(self, set_id: str) -> None:
        sid = str(set_id or "")
        with self._lock:
            self.active_request_map.pop(sid, None)
            self.active_request_meta.pop(sid, None)
            self.completed_outcome_map.pop(sid, None)

    def reset_active_run_state(self) -> None:
        with self._lock:
            self.active_request_map = {}
            self.active_request_meta = {}
            self.completed_outcome_map = {}

    def reset_run_state(self) -> None:
        self.reset_active_run_state()
        with self._lock:
            self.superseded_request_map = {}
        self.drain_completion_queue()

    def _prune_completed_superseded_locked(self) -> None:
        self.superseded_request_map = {
            key: handle
            for key, handle in (self.superseded_request_map or {}).items()
            if isinstance(handle, BatchRequestHandle) and not handle.is_done()
        }

    def _discard_superseded_handle(self, handle: BatchRequestHandle) -> None:
        with self._lock:
            stale_ids = [
                sid
                for sid, tracked in (self.superseded_request_map or {}).items()
                if tracked is handle
            ]
            for sid in stale_ids:
                self.superseded_request_map.pop(sid, None)

    @staticmethod
    def _superseded_request_key(handle: BatchRequestHandle) -> str:
        metadata = handle.metadata
        return (
            f"{int(metadata.generation)}:"
            f"{int(metadata.run_id)}:"
            f"{int(metadata.request_id)}:"
            f"{str(metadata.set_id)}:"
            f"{id(handle)}"
        )

    def clear_stale_requests(self) -> None:
        self.reset_active_run_state()

    def drain_completion_queue(self) -> None:
        while True:
            try:
                self.completed_queue.get_nowait()
            except Empty:
                break
            except Exception:
                break

    def enqueue_completion(self, set_id: str) -> None:
        sid = str(set_id or "")
        if not sid:
            return
        self.completed_queue.put((sid, float(perf_counter())))

    def ensure_lane_pool(self, *, max_lanes: int) -> Any:
        requested_lanes = max(1, int(max_lanes))
        pool = self.lane_pool
        if pool is not None and self._pool_stale:
            force_terminate = (
                self.has_active_requests()
                or (
                    self._current_max_workers is not None
                    and requested_lanes > int(self._current_max_workers)
                )
            )
            self.shutdown(
                force_terminate=bool(force_terminate),
                record_nonfatal_exception=self.record_nonfatal_exception,
            )
            return self._create_lane_pool(max_lanes=requested_lanes)
        if pool is None:
            return self._create_lane_pool(max_lanes=requested_lanes)
        if self._current_max_workers is None:
            return pool
        if requested_lanes <= int(self._current_max_workers):
            self._ensure_worker_threads(max_workers=max(1, int(self._current_max_workers)))
            return pool
        self.shutdown(force_terminate=True, record_nonfatal_exception=self.record_nonfatal_exception)
        return self._create_lane_pool(max_lanes=requested_lanes)

    def _create_lane_pool(self, *, max_lanes: int) -> Any:
        try:
            with self._lock:
                self._shutdown_requested = False
                self._warm_failure = None
            pool = self.lane_pool_factory(int(max_lanes), bool(self.limit_blas_threads_per_worker))
            self.lane_pool = pool
            self._current_max_workers = int(max_lanes)
            self._pool_stale = False
            self._warm_requested_max_lanes = 0
            self._ensure_worker_threads(max_workers=int(max_lanes))
        except Exception as exc:
            with self._lock:
                self._warm_failure = f"{type(exc).__name__}: {exc}"
            self.shutdown(
                force_terminate=True,
                record_nonfatal_exception=self.record_nonfatal_exception,
                clear_warm_failure=False,
            )
            self.lane_pool = None
            self.record_nonfatal_exception("Failed to create batch lane pool", exc)
            raise
        return self.lane_pool

    def ensure_warm_lane_pool(self, *, max_lanes: int, wait: bool = True) -> Any:
        pool = self.ensure_lane_pool(max_lanes=int(max_lanes))
        try:
            warm_lanes = getattr(pool, "warm_lanes", None)
            if callable(warm_lanes):
                with self._lock:
                    self._warm_failure = None
                warm_lanes(max(1, int(max_lanes)), wait=bool(wait))
                self._warm_requested_max_lanes = max(
                    int(self._warm_requested_max_lanes),
                    max(1, int(max_lanes)),
                )
                if not bool(wait):
                    self._start_background_ready_wait(pool=pool, max_lanes=max(1, int(max_lanes)))
        except Exception as exc:
            if self._suppress_warm_failure_for_pool(pool):
                return pool
            with self._lock:
                self._warm_failure = f"{type(exc).__name__}: {exc}"
            self.shutdown(
                force_terminate=True,
                record_nonfatal_exception=self.record_nonfatal_exception,
                clear_warm_failure=False,
            )
            self.record_nonfatal_exception("Failed to warm batch lane pool", exc)
            raise
        return pool

    def _suppress_warm_failure_for_pool(self, pool: Any) -> bool:
        with self._lock:
            return bool(self._shutdown_requested or self.lane_pool is not pool)

    def _start_background_ready_wait(self, *, pool: Any, max_lanes: int) -> None:
        existing = self._warm_thread
        if existing is not None and existing.is_alive():
            return

        def _wait_until_ready() -> None:
            try:
                warm_lanes = getattr(pool, "warm_lanes", None)
                if callable(warm_lanes):
                    warm_lanes(max(1, int(max_lanes)), wait=True)
            except Exception as exc:
                if not self._suppress_warm_failure_for_pool(pool):
                    with self._lock:
                        self._warm_failure = f"{type(exc).__name__}: {exc}"
                    self.record_nonfatal_exception("Failed to complete background batch lane warmup", exc)
                    self.shutdown(
                        force_terminate=True,
                        record_nonfatal_exception=self.record_nonfatal_exception,
                        clear_warm_failure=False,
                    )

        thread = threading.Thread(
            target=_wait_until_ready,
            name="kindred-batch-lane-warmup",
            daemon=True,
        )
        self._warm_thread = thread
        thread.start()

    def _ensure_worker_threads(self, *, max_workers: int) -> None:
        self._prune_retired_worker_threads()
        live_workers = [thread for thread in self._worker_threads if thread.is_alive()]
        if len(live_workers) == int(max_workers) and self._work_queue is not None:
            self._worker_threads = live_workers
            return
        self._stop_worker_threads(join_timeout_s=0.2)
        worker_count = max(1, int(max_workers))
        queue: Queue[Any] = Queue()
        self._work_queue = queue
        self._worker_threads = []
        for index in range(worker_count):
            thread = threading.Thread(
                target=self._worker_loop,
                args=(queue,),
                name=f"kindred-batch-request-worker-{index + 1}",
                daemon=True,
            )
            self._worker_threads.append(thread)
            thread.start()

    def _stop_worker_threads(self, *, join_timeout_s: float) -> None:
        queue = self._work_queue
        workers = list(self._worker_threads)
        self._work_queue = None
        if queue is None:
            live = [thread for thread in workers if thread.is_alive()]
            self._retired_worker_threads.extend(live)
            self._worker_threads = []
            self._prune_retired_worker_threads()
            return
        self._mark_queued_requests_superseded(queue)
        for _thread in workers:
            try:
                queue.put_nowait(None)
            except Exception:
                pass
        for thread in workers:
            try:
                thread.join(timeout=max(0.0, float(join_timeout_s)))
            except Exception:
                pass
        live = [thread for thread in workers if thread.is_alive()]
        self._retired_worker_threads.extend(live)
        self._worker_threads = []
        self._prune_retired_worker_threads()

    @staticmethod
    def _mark_queued_requests_superseded(queue: Queue[Any]) -> None:
        retained: list[Any] = []
        while True:
            try:
                item = queue.get_nowait()
            except Empty:
                break
            if item is None:
                retained.append(item)
                continue
            try:
                handle, _target = item
            except Exception:
                continue
            if isinstance(handle, BatchRequestHandle):
                handle.mark_superseded()
                handle.finish(None)
        for item in retained:
            try:
                queue.put_nowait(item)
            except Exception:
                pass

    def _worker_loop(self, queue: Queue[Any]) -> None:
        while True:
            item = queue.get()
            if item is None:
                return
            try:
                handle, target = item
            except Exception:
                continue
            if not isinstance(handle, BatchRequestHandle):
                continue
            if handle.superseded or int(handle.metadata.generation) != int(self._generation):
                handle.finish(None)
                self._discard_superseded_handle(handle)
                continue
            try:
                outcome = target()
            except BaseException as exc:  # noqa: BLE001 - worker boundary returns structured outcome
                outcome = BatchLaneOutcome(
                    lane_id="runtime-owner",
                    run_id=handle.run_id,
                    request_id=handle.request_id,
                    set_id=handle.set_id,
                    owner_epoch=0,
                    success=False,
                    failure={
                        "kind": "internal_error",
                        "phase": "runtime_owner",
                        "message": str(exc),
                        "exc_type": type(exc).__name__,
                    },
                )
            handle.finish(outcome)
            if handle.superseded or int(handle.metadata.generation) != int(self._generation):
                self._discard_superseded_handle(handle)
                continue
            self._handle_completed_request(handle)

    def submit_task(
        self,
        task: Mapping[str, Any],
        *,
        run_id: int,
        request_id: int,
        set_id: str,
        set_name: str,
        preview_owner_epoch: int | None,
        active_timeout_s: float,
        expected_owner_epoch: int | None = None,
    ) -> BatchRequestHandle:
        pool = self.lane_pool
        if pool is None:
            raise RuntimeError("Batch lane pool is not initialized.")
        sid = str(set_id or "")
        metadata = BatchRequestMetadata(
            set_id=sid,
            set_name=str(set_name or sid),
            run_id=int(run_id),
            request_id=int(request_id),
            generation=int(self._generation),
            preview_owner_epoch=preview_owner_epoch,
            expected_owner_epoch=None if expected_owner_epoch is None else int(expected_owner_epoch),
        )
        handle = BatchRequestHandle(metadata)
        with self._lock:
            self.active_request_map[sid] = handle
            self.active_request_meta[sid] = {
                "set_name": metadata.set_name,
                "preview_owner_epoch": metadata.preview_owner_epoch,
                "owner_epoch": metadata.expected_owner_epoch,
                "generation": metadata.generation,
            }

        def _target() -> BatchLaneOutcome:
            return pool.run(
                dict(task or {}),
                run_id=int(run_id),
                request_id=int(request_id),
                set_id=sid,
                active_timeout_s=float(active_timeout_s),
            )

        queue = self._work_queue
        if queue is None:
            self._ensure_worker_threads(max_workers=max(1, int(self._current_max_workers or self.max_parallel_workers)))
            queue = self._work_queue
        if queue is None:
            with self._lock:
                self.active_request_map.pop(sid, None)
                self.active_request_meta.pop(sid, None)
            raise RuntimeError("Batch request worker queue is not initialized.")
        queue.put((handle, _target))
        return handle

    def poll_completed_records(self) -> list[BatchPolledCompletion]:
        polled: list[BatchPolledCompletion] = []
        processed: set[str] = set()
        while True:
            try:
                sid_raw, completed_ts = self.completed_queue.get_nowait()
            except Empty:
                break
            except Exception:
                break
            sid = str(sid_raw or "")
            if not sid or sid in processed:
                continue
            with self._lock:
                handle = (self.active_request_map or {}).get(sid)
            if handle is None or not self._handle_is_done(handle):
                continue
            record = self.pop_completed_record(sid)
            if record is None:
                continue
            processed.add(sid)
            polled.append(
                BatchPolledCompletion(
                    set_id=sid,
                    record=record,
                    source="callback",
                    completed_ts=float(completed_ts),
                )
            )

        with self._lock:
            active_items = list((self.active_request_map or {}).items())
        for set_id, handle in active_items:
            sid = str(set_id or "")
            if not sid or sid in processed:
                continue
            if not self._handle_is_done(handle):
                continue
            record = self.pop_completed_record(sid)
            if record is None:
                continue
            processed.add(sid)
            polled.append(
                BatchPolledCompletion(
                    set_id=sid,
                    record=record,
                    source="scan",
                    completed_ts=record.completed_ts,
                )
            )
        return polled

    @staticmethod
    def _handle_is_done(handle: Any) -> bool:
        is_done = getattr(handle, "is_done", None)
        if callable(is_done):
            return bool(is_done())
        return False

    def pop_completed_record(self, set_id: str) -> BatchCompletionRecord | None:
        sid = str(set_id or "")
        with self._lock:
            self.active_request_map.pop(sid, None)
            self.active_request_meta.pop(sid, None)
            record_or_outcome = self.completed_outcome_map.pop(sid, None)
        if isinstance(record_or_outcome, BatchCompletionRecord):
            return record_or_outcome
        return None

    def shutdown(
        self,
        *,
        force_terminate: bool,
        record_nonfatal_exception: Callable[[str, BaseException], None],
        clear_warm_failure: bool = True,
    ) -> None:
        with self._lock:
            self._shutdown_requested = True
            if bool(clear_warm_failure):
                self._warm_failure = None
            active_handles = list((self.active_request_map or {}).values())
        for handle in active_handles:
            if isinstance(handle, BatchRequestHandle):
                handle.mark_superseded()
        self._stop_worker_threads(join_timeout_s=0.5)
        pool = self.lane_pool
        self.lane_pool = None
        self._current_max_workers = None
        self._pool_stale = False
        if pool is not None:
            try:
                close = getattr(pool, "close", None)
                if callable(close):
                    close(kill=bool(force_terminate))
            except Exception as exc:
                record_nonfatal_exception("Failed batch lane pool shutdown", exc)
        self.reset_run_state()

    def soft_supersede(self) -> tuple[int, int]:
        with self._lock:
            active = dict(self.active_request_map or {})
            self._generation += 1
            self.active_request_map = {}
            self.active_request_meta = {}
            self.completed_outcome_map = {}
            for handle in active.values():
                if isinstance(handle, BatchRequestHandle):
                    self.superseded_request_map[self._superseded_request_key(handle)] = handle
        running = 0
        for handle in active.values():
            if isinstance(handle, BatchRequestHandle):
                handle.mark_superseded()
            running += 1
        self.drain_completion_queue()
        return 0, running

    def _handle_completed_request(self, handle: BatchRequestHandle) -> None:
        if handle.superseded or int(handle.metadata.generation) != int(self._generation):
            return
        sid = str(handle.set_id or "")
        if not sid:
            return
        if not isinstance(handle.outcome, BatchLaneOutcome):
            return
        metadata = handle.metadata
        record = BatchCompletionRecord(
            metadata=metadata,
            outcome=handle.outcome,
            completed_ts=float(handle.completed_ts if handle.completed_ts is not None else perf_counter()),
        )
        with self._lock:
            if self.active_request_map.get(sid) is not handle:
                return
            self.completed_outcome_map[sid] = record
        self.completed_queue.put((sid, record.completed_ts))
