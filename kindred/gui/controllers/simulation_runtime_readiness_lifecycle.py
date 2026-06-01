from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

from PySide6 import QtCore

from kindred.gui.controllers.runtime_lane_allocation import (
    PreparedRuntimeRequestSet,
    RuntimeDispatchPlan,
    RuntimeLaneAllocationRequest,
    RuntimeLaneAllocator,
    RuntimeLaunchAllocation,
    RuntimeLaunchIntent,
)


@dataclass(frozen=True)
class SimulationRuntimeReadinessRenderState:
    status: str
    status_text: str = ""
    launch_available: bool = False
    preview_available: bool = False
    failed: bool = False
    retryable: bool = False
    clear_status: bool = False
    preview_unavailable_status: str = ""


class SimulationRuntimeReadinessLifecycle(QtCore.QObject):
    def __init__(
        self,
        *,
        allocator: RuntimeLaneAllocator,
        render: Callable[[SimulationRuntimeReadinessRenderState], object] | None = None,
        current_runtime_input_epochs: Callable[[Sequence[str]], Mapping[str, int]] | None = None,
        prepared_request_is_current: Callable[[PreparedRuntimeRequestSet], bool] | None = None,
        dispatch_ready: Callable[[RuntimeDispatchPlan], object] | None = None,
        warmup_retry_interval_ms: int = 25,
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._allocator = allocator
        self._render = render
        self._current_runtime_input_epochs = current_runtime_input_epochs
        self._prepared_request_current_provider = prepared_request_is_current
        self._dispatch_ready = dispatch_ready
        self._warmup_retry_interval_ms = max(0, int(warmup_retry_interval_ms))
        self._warmup_retry_queued = False
        self._last_prepared: PreparedRuntimeRequestSet | None = None
        self._last_allocation: RuntimeLaunchAllocation | None = None
        self._last_dispatch_plan: RuntimeDispatchPlan | None = None
        self._retry_prepared: PreparedRuntimeRequestSet | None = None
        self._preview_replay_intent: RuntimeLaunchIntent | None = None

    @property
    def retry_available(self) -> bool:
        return self._retry_prepared is not None

    @property
    def last_dispatch_plan(self) -> RuntimeDispatchPlan | None:
        return self._last_dispatch_plan

    def accept_prepared_request(
        self,
        prepared: PreparedRuntimeRequestSet,
    ) -> RuntimeDispatchPlan | None:
        self._last_prepared = prepared
        if not prepared.prepared:
            reason = prepared.blocked_reason
            self._retry_prepared = prepared if bool(getattr(reason, "retryable", False)) else None
            self._render_state(
                SimulationRuntimeReadinessRenderState(
                    status="blocked",
                    status_text=str(reason.message if reason is not None else "Runtime request is blocked."),
                    failed=True,
                    retryable=bool(reason.retryable) if reason is not None else False,
                    preview_unavailable_status=(
                        str(reason.message)
                        if reason is not None and prepared.intent.intent_kind == "preview"
                        else ""
                    ),
                )
            )
            return None
        try:
            self._allocator.ensure_ready_lanes(
                compatibility_key=prepared.compatibility_key,
                capacity=int(prepared.preferred_lane_capacity or prepared.required_lane_capacity),
                task_count=len(prepared.task_descriptors),
            )
            allocation = self._allocator.allocate(
                RuntimeLaneAllocationRequest(
                    compatibility_key=prepared.compatibility_key,
                    required_lane_capacity=int(prepared.required_lane_capacity),
                    preferred_lane_capacity=int(prepared.preferred_lane_capacity or prepared.required_lane_capacity),
                    task_count=len(prepared.task_descriptors),
                    request_token=prepared.intent.request_token,
                    scope=prepared.intent.intent_kind,
                )
            )
        except Exception as exc:
            self._retry_prepared = prepared
            self._render_state(
                SimulationRuntimeReadinessRenderState(
                    status="failed",
                    status_text=str(exc) or "Runtime lane allocation failed.",
                    failed=True,
                    retryable=True,
                    preview_unavailable_status=(
                        str(exc)
                        if prepared.intent.intent_kind == "preview"
                        else ""
                    ),
                )
            )
            return None
        self._last_allocation = allocation
        consume = self._allocator.consume(allocation, prepared, expected=prepared)
        if consume.dispatch_plan is None:
            self._retry_prepared = prepared if consume.retryable else None
            self._render_state(
                SimulationRuntimeReadinessRenderState(
                    status=consume.status,
                    status_text=consume.message or "Runtime lanes are not ready.",
                    failed=consume.status not in {"waiting", "ready"},
                    retryable=consume.retryable,
                    preview_unavailable_status=(
                        consume.message
                        if prepared.intent.intent_kind == "preview"
                        else ""
                    ),
                )
            )
            if consume.retryable and consume.status in {"waiting", "warming", "not_ready"}:
                self._schedule_pending_warmup_poll()
            return None
        self._last_dispatch_plan = consume.dispatch_plan
        self._retry_prepared = None
        self._warmup_retry_queued = False
        self._render_state(
            SimulationRuntimeReadinessRenderState(
                status="ready",
                launch_available=True,
                preview_available=prepared.intent.intent_kind == "preview",
                clear_status=True,
            )
        )
        return consume.dispatch_plan

    def accept_preview_replay_intent(
        self,
        intent: RuntimeLaunchIntent,
        *,
        prepare: Callable[[RuntimeLaunchIntent], PreparedRuntimeRequestSet],
    ) -> RuntimeDispatchPlan | None:
        self._preview_replay_intent = intent
        prepared = prepare(intent)
        if prepared.intent != self._preview_replay_intent:
            self._retry_prepared = prepared if prepared.prepared else None
            self._render_state(
                SimulationRuntimeReadinessRenderState(
                    status="stale",
                    status_text="Preview replay intent changed before runtime allocation.",
                    failed=True,
                    retryable=bool(prepared.prepared),
                    preview_unavailable_status="Preview replay intent changed before runtime allocation.",
                )
            )
            return None
        dispatch_plan = self.accept_prepared_request(prepared)
        if dispatch_plan is not None:
            self._preview_replay_intent = None
        return dispatch_plan

    def retry_runtime_readiness(self) -> RuntimeDispatchPlan | None:
        prepared = self._retry_prepared
        if prepared is None:
            return None
        if not self._prepared_request_is_current(prepared):
            self._retry_prepared = None
            message = "Prepared runtime launch intent changed after the retry request was prepared."
            self._render_state(
                SimulationRuntimeReadinessRenderState(
                    status="stale",
                    status_text=message,
                    failed=True,
                    retryable=False,
                    preview_unavailable_status=(
                        message
                        if prepared.intent.intent_kind == "preview"
                        else ""
                    ),
                )
            )
            return None
        if not self._prepared_request_epochs_current(prepared):
            self._retry_prepared = None
            self._render_state(
                SimulationRuntimeReadinessRenderState(
                    status="stale",
                    status_text="Runtime inputs changed after the retry request was prepared.",
                    failed=True,
                    retryable=False,
                    preview_unavailable_status=(
                        "Runtime inputs changed after the retry request was prepared."
                        if prepared.intent.intent_kind == "preview"
                        else ""
                    ),
                )
            )
            return None
        return self.accept_prepared_request(prepared)

    def poll_pending_warmup(self) -> RuntimeDispatchPlan | None:
        self._warmup_retry_queued = False
        dispatch_plan = self.retry_runtime_readiness()
        if dispatch_plan is not None and callable(self._dispatch_ready):
            self._dispatch_ready(dispatch_plan)
        return dispatch_plan

    def render_failure(self, message: str, *, retryable: bool = True) -> None:
        if self._last_prepared is not None and bool(retryable):
            self._retry_prepared = self._last_prepared
        self._render_state(
            SimulationRuntimeReadinessRenderState(
                status="failed",
                status_text=str(message or "Runtime launch failed."),
                failed=True,
                retryable=bool(retryable),
            )
        )

    def release_dispatch_plan(self, dispatch_plan: RuntimeDispatchPlan | None, *, failed: bool = False) -> bool:
        if dispatch_plan is None:
            return False
        return self._allocator.release(
            dispatch_plan.launch_allocation.allocation_id,
            failed=bool(failed),
        )

    def release_all(self, *, failed: bool = False) -> None:
        self._allocator.release_all(failed=bool(failed))

    def _prepared_request_epochs_current(self, prepared: PreparedRuntimeRequestSet) -> bool:
        provider = self._current_runtime_input_epochs
        if provider is None:
            return True
        expected = {
            str(key): int(value)
            for key, value in dict(prepared.intent.runtime_input_epochs or {}).items()
            if str(key)
        }
        if not expected:
            return True
        try:
            current = {
                str(key): int(value)
                for key, value in dict(provider(prepared.intent.set_ids)).items()
                if str(key)
            }
        except Exception:
            return False
        for key, expected_value in expected.items():
            if int(current.get(key, -1)) != int(expected_value):
                return False
        return True

    def _prepared_request_is_current(self, prepared: PreparedRuntimeRequestSet) -> bool:
        provider = self._prepared_request_current_provider
        if provider is None:
            return True
        try:
            return bool(provider(prepared))
        except Exception:
            return False

    def _render_state(self, state: SimulationRuntimeReadinessRenderState) -> None:
        if callable(self._render):
            self._render(state)

    def _schedule_pending_warmup_poll(self) -> None:
        if self._retry_prepared is None or not callable(self._dispatch_ready):
            return
        if self._warmup_retry_queued:
            return
        self._warmup_retry_queued = True
        QtCore.QTimer.singleShot(
            int(self._warmup_retry_interval_ms),
            self.poll_pending_warmup,
        )
