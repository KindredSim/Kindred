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


@dataclass(frozen=True)
class SimulationRuntimeReadinessEndpointState:
    manual_retry_available: bool = False
    backend_warmup_pending: bool = False
    pending_intent_kind: str = ""


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
        self._manual_retry_prepared: PreparedRuntimeRequestSet | None = None
        self._backend_warmup_prepared: PreparedRuntimeRequestSet | None = None
        self._preview_replay_intent: RuntimeLaunchIntent | None = None
        self._pending_backend_idle_release_plan: RuntimeDispatchPlan | None = None

    @property
    def retry_available(self) -> bool:
        return self.manual_retry_available

    @property
    def manual_retry_available(self) -> bool:
        return self._manual_retry_prepared is not None

    @property
    def backend_warmup_pending(self) -> bool:
        return self._backend_warmup_prepared is not None

    @property
    def pending_intent_kind(self) -> str:
        prepared = self._backend_warmup_prepared or self._manual_retry_prepared
        if prepared is None:
            return ""
        return self._intent_kind(prepared)

    @property
    def endpoint_state(self) -> SimulationRuntimeReadinessEndpointState:
        return SimulationRuntimeReadinessEndpointState(
            manual_retry_available=self.manual_retry_available,
            backend_warmup_pending=self.backend_warmup_pending,
            pending_intent_kind=self.pending_intent_kind,
        )

    @property
    def last_dispatch_plan(self) -> RuntimeDispatchPlan | None:
        return self._last_dispatch_plan

    def accept_prepared_request(
        self,
        prepared: PreparedRuntimeRequestSet,
    ) -> RuntimeDispatchPlan | None:
        return self._accept_prepared_request(prepared)

    def _accept_prepared_request(
        self,
        prepared: PreparedRuntimeRequestSet,
        *,
        backend_warmup_continuation: bool = False,
    ) -> RuntimeDispatchPlan | None:
        self._last_prepared = prepared
        self._manual_retry_prepared = None
        if not bool(backend_warmup_continuation):
            self._clear_backend_warmup_state()
        if not prepared.prepared:
            reason = prepared.blocked_reason
            self._set_manual_retry_prepared(
                prepared if bool(getattr(reason, "retryable", False)) else None
            )
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
        if not self._prepared_request_is_current(prepared):
            self._clear_endpoint_state()
            message = "Prepared runtime launch intent changed before runtime allocation."
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
            self._clear_endpoint_state()
            message = "Runtime inputs changed before runtime allocation."
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
                    require_backend_lease=True,
                )
            )
        except Exception as exc:
            self._clear_backend_warmup_state()
            self._set_manual_retry_prepared(prepared)
            self._render_state(
                SimulationRuntimeReadinessRenderState(
                    status="failed",
                    status_text=str(exc) or "Runtime lane allocation failed.",
                    failed=True,
                    retryable=True,
                    launch_available=prepared.intent.intent_kind != "preview",
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
            if consume.retryable and consume.status in {"waiting", "warming", "not_ready"}:
                self._backend_warmup_prepared = prepared
            else:
                self._clear_backend_warmup_state()
                self._set_manual_retry_prepared(prepared if consume.retryable else None)
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
        self._clear_endpoint_state()
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
            self._clear_backend_warmup_state()
            self._set_manual_retry_prepared(prepared if prepared.prepared else None)
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
        prepared = self._manual_retry_prepared
        if prepared is None:
            if self._backend_warmup_prepared is not None and self.pending_intent_kind == "ordinary":
                message = "Ordinary runtime retry requires a fresh Run request."
                self._render_state(
                    SimulationRuntimeReadinessRenderState(
                        status="retry_unavailable",
                        status_text=message,
                        failed=True,
                        retryable=False,
                    )
                )
            return None
        if str(prepared.intent.intent_kind or "ordinary") != "preview":
            self._manual_retry_prepared = None
            message = "Ordinary runtime retry requires a fresh Run request."
            self._render_state(
                SimulationRuntimeReadinessRenderState(
                    status="retry_unavailable",
                    status_text=message,
                    failed=True,
                    retryable=False,
                )
            )
            return None
        if not self._prepared_request_is_current(prepared):
            self._manual_retry_prepared = None
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
            self._manual_retry_prepared = None
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
        return self._accept_prepared_request(prepared)

    def poll_pending_warmup(self) -> RuntimeDispatchPlan | None:
        self._warmup_retry_queued = False
        prepared = self._backend_warmup_prepared
        if prepared is None:
            return None
        dispatch_plan = self._accept_prepared_request(
            prepared,
            backend_warmup_continuation=True,
        )
        if dispatch_plan is not None and callable(self._dispatch_ready):
            self._dispatch_ready(dispatch_plan)
        return dispatch_plan

    def render_failure(self, message: str, *, retryable: bool = True) -> None:
        self._clear_backend_warmup_state()
        prepared = self._last_prepared
        intent_kind = self._intent_kind(prepared) if prepared is not None else "ordinary"
        if bool(retryable) and prepared is not None and intent_kind == "preview":
            self._set_manual_retry_prepared(prepared)
        else:
            self._manual_retry_prepared = None
        status_text = str(message or "Runtime dispatch failed.")
        self._render_state(
            SimulationRuntimeReadinessRenderState(
                status="failed",
                status_text=status_text,
                failed=True,
                retryable=bool(retryable),
                launch_available=bool(retryable) and intent_kind != "preview",
                preview_unavailable_status=(
                    status_text if intent_kind == "preview" else ""
                ),
            )
        )

    def release_dispatch_plan(self, dispatch_plan: RuntimeDispatchPlan | None, *, failed: bool = False) -> bool:
        if dispatch_plan is None:
            return False
        released = self._allocator.release(
            dispatch_plan.launch_allocation.allocation_id,
            failed=bool(failed),
        )
        if bool(released) and self._same_dispatch_plan(dispatch_plan, self._pending_backend_idle_release_plan):
            self._pending_backend_idle_release_plan = None
        return bool(released)

    def defer_dispatch_plan_release_until_backend_idle(
        self,
        dispatch_plan: RuntimeDispatchPlan | None,
    ) -> bool:
        if dispatch_plan is None:
            return False
        self._pending_backend_idle_release_plan = dispatch_plan
        return True

    def release_pending_dispatch_plan_if_backend_idle(
        self,
        *,
        active_request_count: int,
        failed: bool = False,
    ) -> RuntimeDispatchPlan | None:
        if int(active_request_count or 0) > 0:
            return None
        dispatch_plan = self._pending_backend_idle_release_plan
        if dispatch_plan is None:
            return None
        if self.release_dispatch_plan(dispatch_plan, failed=bool(failed)):
            return dispatch_plan
        return None

    def release_all(self, *, failed: bool = False) -> None:
        self._allocator.release_all(failed=bool(failed))
        self._pending_backend_idle_release_plan = None

    def release_all_launch_contexts(self, *, failed: bool = False) -> None:
        self._clear_endpoint_state()
        self._allocator.release_all(failed=bool(failed))
        self._pending_backend_idle_release_plan = None

    def release_preview_launch_contexts(self, *, failed: bool = False) -> int:
        self._clear_preview_endpoint_state()
        clear_pending_preview = self._dispatch_plan_intent_kind(self._pending_backend_idle_release_plan) == "preview"
        released = self._allocator.release_by_intent_kind("preview", failed=bool(failed))
        if clear_pending_preview and int(released or 0) > 0:
            self._pending_backend_idle_release_plan = None
        return int(released or 0)

    def release_launch_contexts_for_run_kind(
        self,
        *,
        fast_mode: bool | None,
        failed: bool = False,
    ) -> None:
        if bool(fast_mode):
            self.release_preview_launch_contexts(failed=bool(failed))
            return
        self.release_all_launch_contexts(failed=bool(failed))

    def _set_manual_retry_prepared(
        self,
        prepared: PreparedRuntimeRequestSet | None,
    ) -> None:
        if prepared is not None and self._intent_kind(prepared) == "preview":
            self._manual_retry_prepared = prepared
            return
        self._manual_retry_prepared = None

    def _clear_backend_warmup_state(self) -> None:
        self._backend_warmup_prepared = None
        self._warmup_retry_queued = False

    def _clear_endpoint_state(self) -> None:
        self._manual_retry_prepared = None
        self._clear_backend_warmup_state()

    def _clear_preview_endpoint_state(self) -> None:
        if (
            self._manual_retry_prepared is not None
            and self._intent_kind(self._manual_retry_prepared) == "preview"
        ):
            self._manual_retry_prepared = None
        if (
            self._backend_warmup_prepared is not None
            and self._intent_kind(self._backend_warmup_prepared) == "preview"
        ):
            self._clear_backend_warmup_state()

    @staticmethod
    def _intent_kind(prepared: PreparedRuntimeRequestSet) -> str:
        return str(prepared.intent.intent_kind or "ordinary")

    @staticmethod
    def _same_dispatch_plan(
        left: RuntimeDispatchPlan | None,
        right: RuntimeDispatchPlan | None,
    ) -> bool:
        if left is None or right is None:
            return False
        return (
            str(left.launch_allocation.allocation_id)
            == str(right.launch_allocation.allocation_id)
        )

    @staticmethod
    def _dispatch_plan_intent_kind(dispatch_plan: RuntimeDispatchPlan | None) -> str:
        if dispatch_plan is None:
            return ""
        intent = dispatch_plan.launch_allocation.launch_intent
        if intent is None and dispatch_plan.launch_allocation.prepared_request_set is not None:
            intent = dispatch_plan.launch_allocation.prepared_request_set.intent
        return str(getattr(intent, "intent_kind", "") or "")

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
        if self._backend_warmup_prepared is None or not callable(self._dispatch_ready):
            return
        if self._warmup_retry_queued:
            return
        self._warmup_retry_queued = True
        QtCore.QTimer.singleShot(
            int(self._warmup_retry_interval_ms),
            self.poll_pending_warmup,
        )
