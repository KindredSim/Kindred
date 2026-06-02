from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, runtime_checkable

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
    superseded_drain_token: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "cancelled", max(0, int(self.cancelled or 0)))
        object.__setattr__(self, "running", max(0, int(self.running or 0)))
        object.__setattr__(self, "superseded_drain_token", str(self.superseded_drain_token or ""))


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
    drained_superseded_release_tokens: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "records", tuple(self.records or ()))
        object.__setattr__(self, "active_after_poll", max(0, int(self.active_after_poll or 0)))
        object.__setattr__(
            self,
            "drained_superseded_release_tokens",
            tuple(str(token) for token in self.drained_superseded_release_tokens or () if str(token)),
        )

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
class RuntimeScopedFailureProgress:
    set_label: str
    completed: int
    total: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "set_label", str(self.set_label or "set"))
        object.__setattr__(self, "completed", max(0, int(self.completed or 0)))
        object.__setattr__(self, "total", max(1, int(self.total or 1)))


@dataclass(frozen=True)
class RuntimeScopedFailureSummary:
    failed_set_ids: tuple[str, ...] = ()
    failed_errors: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "failed_set_ids",
            tuple(str(set_id) for set_id in self.failed_set_ids or () if str(set_id)),
        )
        object.__setattr__(self, "failed_errors", dict(self.failed_errors or {}))


@dataclass(frozen=True)
class RuntimeCompletionDecision:
    accepted: bool = True
    consumed: bool = True
    terminal: bool = False
    failed: bool = False
    message: str = ""
    stop_current_poll_batch: bool = False
    scoped_failure_progress: RuntimeScopedFailureProgress | None = None
    final_scoped_failure: bool = False
    scoped_failure_summary: RuntimeScopedFailureSummary | None = None
    terminal_failure_preview_replay_needed: bool = False
    terminal_failure_preview_replay_fast_mode: bool = False
    current_preview_failure_status_text: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "final_scoped_failure", bool(self.final_scoped_failure))
        object.__setattr__(
            self,
            "terminal_failure_preview_replay_needed",
            bool(self.terminal_failure_preview_replay_needed),
        )
        object.__setattr__(
            self,
            "terminal_failure_preview_replay_fast_mode",
            bool(self.terminal_failure_preview_replay_fast_mode),
        )
        object.__setattr__(
            self,
            "current_preview_failure_status_text",
            str(self.current_preview_failure_status_text or ""),
        )
        if not isinstance(self.scoped_failure_progress, RuntimeScopedFailureProgress):
            object.__setattr__(self, "scoped_failure_progress", None)
        if not isinstance(self.scoped_failure_summary, RuntimeScopedFailureSummary):
            object.__setattr__(self, "scoped_failure_summary", None)

    @classmethod
    def accepted_current(
        cls,
        *,
        scoped_failure_progress: RuntimeScopedFailureProgress | None = None,
        final_scoped_failure: bool = False,
        scoped_failure_summary: RuntimeScopedFailureSummary | None = None,
        terminal_failure_preview_replay_needed: bool = False,
        terminal_failure_preview_replay_fast_mode: bool = False,
        current_preview_failure_status_text: str = "",
    ) -> RuntimeCompletionDecision:
        return cls(
            accepted=True,
            consumed=True,
            scoped_failure_progress=scoped_failure_progress,
            final_scoped_failure=bool(final_scoped_failure),
            scoped_failure_summary=scoped_failure_summary,
            terminal_failure_preview_replay_needed=bool(terminal_failure_preview_replay_needed),
            terminal_failure_preview_replay_fast_mode=bool(
                terminal_failure_preview_replay_fast_mode
            ),
            current_preview_failure_status_text=str(current_preview_failure_status_text or ""),
        )

    @classmethod
    def ignored_stale(cls, *, consumed: bool = True) -> RuntimeCompletionDecision:
        return cls(accepted=False, consumed=bool(consumed))

    @classmethod
    def reset_requested(cls, *, consumed: bool = False) -> RuntimeCompletionDecision:
        return cls(
            accepted=False,
            consumed=bool(consumed),
            stop_current_poll_batch=True,
        )

    @classmethod
    def terminal_failure(
        cls,
        message: str = "",
        *,
        terminal_failure_preview_replay_needed: bool = False,
        terminal_failure_preview_replay_fast_mode: bool = False,
    ) -> RuntimeCompletionDecision:
        return cls(
            accepted=False,
            consumed=False,
            terminal=True,
            failed=True,
            message=str(message or ""),
            terminal_failure_preview_replay_needed=bool(
                terminal_failure_preview_replay_needed
            ),
            terminal_failure_preview_replay_fast_mode=bool(
                terminal_failure_preview_replay_fast_mode
            ),
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
