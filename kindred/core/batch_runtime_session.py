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
    preview_owner_epoch: int | None = None
    active_timeout_s: float = 60.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", int(self.run_id))
        object.__setattr__(self, "request_id", int(self.request_id))
        object.__setattr__(self, "fast_mode", bool(self.fast_mode))
        object.__setattr__(self, "queue_ids", tuple(str(item) for item in self.queue_ids if str(item)))
        object.__setattr__(self, "queue_names", tuple(str(item) for item in self.queue_names))
        object.__setattr__(
            self,
            "preview_owner_epoch",
            None if self.preview_owner_epoch is None else int(self.preview_owner_epoch),
        )
        object.__setattr__(self, "active_timeout_s", float(self.active_timeout_s))


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

    def __post_init__(self) -> None:
        object.__setattr__(self, "active", bool(self.active))
        object.__setattr__(self, "run_id", int(self.run_id))
        object.__setattr__(self, "request_id", int(self.request_id))
        object.__setattr__(self, "fast_mode", bool(self.fast_mode))
        object.__setattr__(self, "queue_ids", tuple(str(item) for item in self.queue_ids if str(item)))
        object.__setattr__(self, "queue_names", tuple(str(item) for item in self.queue_names))
        object.__setattr__(self, "completed_set_ids", tuple(str(item) for item in self.completed_set_ids if str(item)))


class BatchRuntimeLaneOwnerProtocol(Protocol):
    def poll_completed_records(self) -> list[BatchPolledCompletion]: ...
    def shutdown(
        self,
        *,
        force_terminate: bool,
        record_nonfatal_exception: Callable[[str, BaseException], None],
    ) -> None: ...
    def soft_supersede(self) -> tuple[int, int, str]: ...
    def superseded_drain_token_drained(self, token: str) -> bool: ...
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

    def begin(self, request: BatchRuntimeSessionRequest) -> None:
        self._request = request
        self._completed_set_ids = []
        self._state = BatchRuntimeSessionState.RUNNING

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
        )

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
            record_run_id = self._record_identity_int(
                record,
                "run_id",
            )
            record_request_id = self._record_identity_int(
                record,
                "request_id",
            )
            record_preview_owner_epoch = self._record_identity_optional_int(
                record,
                "preview_owner_epoch",
            )
            is_current_session_record = (
                record_run_id is not None
                and record_request_id is not None
                and int(record_run_id) == int(request.run_id)
                and int(record_request_id) == int(request.request_id)
                and (
                    request.preview_owner_epoch is None
                    or (
                        record_preview_owner_epoch is not None
                        and int(record_preview_owner_epoch) == int(request.preview_owner_epoch)
                    )
                )
            )
            if not is_current_session_record:
                stale_metadata = dict(record.request_metadata or {})
                stale_metadata["runtime_session_stale"] = {
                    "expected_run_id": int(request.run_id),
                    "expected_request_id": int(request.request_id),
                    "expected_preview_owner_epoch": request.preview_owner_epoch,
                    "actual_run_id": record_run_id,
                    "actual_request_id": record_request_id,
                    "actual_preview_owner_epoch": record_preview_owner_epoch,
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
        self._update_completion_state()
        return accepted

    def soft_supersede_active_run(self) -> tuple[int, int, str]:
        cancelled, running, drain_token = self._lane_owner.soft_supersede()
        self._state = BatchRuntimeSessionState.SUPERSEDED
        return int(cancelled), int(running), str(drain_token or "")

    def superseded_drain_token_drained(self, token: str) -> bool:
        return bool(self._lane_owner.superseded_drain_token_drained(token))


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


    def _require_running_request(self) -> BatchRuntimeSessionRequest:
        request = self._request
        if request is None or self._state is not BatchRuntimeSessionState.RUNNING:
            raise RuntimeError("Batch runtime session is not running.")
        return request

    def _update_completion_state(self) -> None:
        request = self._request
        if request is None or self._state is not BatchRuntimeSessionState.RUNNING:
            return
        expected = _normalize_ids(request.queue_ids)
        if expected and expected.issubset(_normalize_ids(self._completed_set_ids)):
            self._state = BatchRuntimeSessionState.COMPLETED

    @staticmethod
    def _record_identity_int(
        record: BatchCompletionRecord,
        name: str,
    ) -> int | None:
        try:
            return int(getattr(record, name))
        except Exception:
            raw = getattr(record, "request_metadata", None)
            if isinstance(raw, Mapping) and raw.get(name) is not None:
                try:
                    return int(raw.get(name))
                except Exception:
                    return None
            return None

    @staticmethod
    def _record_identity_optional_int(
        record: BatchCompletionRecord,
        name: str,
    ) -> int | None:
        try:
            value = getattr(record, name)
        except Exception:
            raw = getattr(record, "request_metadata", None)
            value = raw.get(name) if isinstance(raw, Mapping) else None
        if value is None:
            return None
        try:
            return int(value)
        except Exception:
            return None



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
