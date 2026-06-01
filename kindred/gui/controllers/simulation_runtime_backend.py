from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from kindred.core.batch_containment import BatchPolledCompletion, BatchRequestHandle
from kindred.gui.controllers.runtime_lane_allocation import (
    RuntimeBackendLease,
    RuntimeBackendTask,
    RuntimeCompatibilityKey,
    RuntimeReleaseReason,
)


@dataclass(frozen=True)
class RuntimeBackendCancelResult:
    cancelled: int
    running: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "cancelled", max(0, int(self.cancelled or 0)))
        object.__setattr__(self, "running", max(0, int(self.running or 0)))


@dataclass(frozen=True)
class RuntimeBackendCloseResult:
    active_after_close: int
    pool_closed: bool
    pool_token: str = ""
    generation: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "active_after_close", max(0, int(self.active_after_close or 0)))
        object.__setattr__(self, "pool_closed", bool(self.pool_closed))
        object.__setattr__(self, "pool_token", str(self.pool_token or ""))
        object.__setattr__(self, "generation", max(0, int(self.generation or 0)))


@dataclass(frozen=True)
class RuntimeBackendPollResult:
    records: tuple[BatchPolledCompletion, ...] = ()
    active_after_poll: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "records", tuple(self.records or ()))
        object.__setattr__(self, "active_after_poll", max(0, int(self.active_after_poll or 0)))

    @property
    def backend_idle(self) -> bool:
        return int(self.active_after_poll) <= 0


@dataclass(frozen=True)
class RuntimeCompletionEvent:
    set_id: str
    record: Any
    outcome: Any
    source: str
    completed_ts: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "set_id", str(self.set_id or ""))
        object.__setattr__(self, "source", str(self.source or ""))
        object.__setattr__(self, "completed_ts", float(self.completed_ts or 0.0))


@dataclass(frozen=True)
class RuntimeCompletionDecision:
    accepted: bool = True
    terminal: bool = False
    failed: bool = False
    message: str = ""

    @classmethod
    def accepted_current(cls) -> RuntimeCompletionDecision:
        return cls(accepted=True)

    @classmethod
    def ignored_stale(cls) -> RuntimeCompletionDecision:
        return cls(accepted=False)

    @classmethod
    def terminal_failure(cls, message: str = "") -> RuntimeCompletionDecision:
        return cls(
            accepted=False,
            terminal=True,
            failed=True,
            message=str(message or ""),
        )


class RuntimeCompletionConsumer(Protocol):
    def consume_runtime_completion(self, event: RuntimeCompletionEvent) -> RuntimeCompletionDecision: ...


@runtime_checkable
class RuntimeBackendPort(Protocol):
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
    ) -> None: ...

    def submit_task(
        self,
        task: RuntimeBackendTask,
        *,
        set_id: str,
        set_name: str,
        callback_identity: object,
    ) -> BatchRequestHandle: ...

    def poll_completed_records(self) -> RuntimeBackendPollResult: ...

    def supersede_current_run(self) -> RuntimeBackendCancelResult: ...

    def close_current_run(self, *, force_terminate: bool) -> RuntimeBackendCloseResult: ...

    def ensure_backend_lease(
        self,
        compatibility_key: RuntimeCompatibilityKey,
        capacity: int,
        *,
        wait: bool,
    ) -> RuntimeBackendLease | None: ...

    def invalidate_backend_lease(
        self,
        lease: RuntimeBackendLease | None,
        *,
        reason: RuntimeReleaseReason,
    ) -> None: ...
