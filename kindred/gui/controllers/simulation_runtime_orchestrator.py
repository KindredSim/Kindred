from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

from kindred.gui.controllers.runtime_lane_allocation import (
    PreparedRuntimeRequestSet,
    RuntimeDispatchPlan,
    RuntimeLaneAllocationRequest,
    RuntimeLaneAllocator,
    RuntimeReleaseReason,
    RuntimeReleaseResult,
)
from kindred.gui.controllers.simulation_runtime_backend import (
    RuntimeBackendPort,
    RuntimeCompletionDecision,
    RuntimeCompletionConsumer,
    RuntimeCompletionEvent,
)
from kindred.gui.controllers.simulation_runtime_readiness_lifecycle import (
    SimulationRuntimeReadinessEndpointState,
    SimulationRuntimeReadinessRenderState,
)


@dataclass(frozen=True)
class QueuePreviewReplay:
    request_id: int | None
    target_set_ids: tuple[str, ...]
    stop_timers: bool = True


@dataclass(frozen=True)
class RuntimePreviewReplayUpdate:
    state: "RuntimePreviewReplayState"
    clear_plot_updates: bool = False


@dataclass(frozen=True)
class RuntimePreviewReplayState:
    active: bool = False
    request_id: int | None = None
    target_set_ids: tuple[str, ...] = ()
    handoff_queued: bool = False
    replay_generation: int = 0

    @classmethod
    def from_pending(cls, pending: object) -> "RuntimePreviewReplayState":
        return cls(
            active=bool(getattr(pending, "active", False)),
            request_id=_optional_int(getattr(pending, "request_id", None)),
            target_set_ids=_normalized_targets(getattr(pending, "target_set_ids", ())),
            handoff_queued=bool(getattr(pending, "handoff_queued", False)),
            replay_generation=int(getattr(pending, "replay_generation", 0) or 0),
        )


@dataclass(frozen=True)
class RuntimePreviewReplaySnapshot:
    active: bool = False
    request_id: int | None = None
    target_set_ids: tuple[str, ...] = ()
    replay_generation: int = 0
    dirty_generation_by_set_id: tuple[tuple[str, int], ...] = ()


@dataclass(frozen=True)
class RuntimeUiEffect:
    stop_completion_polling: bool = False
    start_completion_polling: bool = False
    queue_preview_replay: QueuePreviewReplay | None = None
    set_preview_replay: RuntimePreviewReplayUpdate | None = None
    clear_preview_replay: bool = False
    clear_preview_plot_updates: bool = False
    dispatch_ready: RuntimeDispatchPlan | None = None
    surface_failure: str = ""
    render_state: SimulationRuntimeReadinessRenderState | None = None
    simulation_running: bool | None = None
    slider_simulation_active: bool | None = None
    run_enabled: bool | None = None
    stop_enabled: bool | None = None
    progress_value: int | None = None
    status_text: str | None = None
    stop_debounce_timers: bool = False


@dataclass(frozen=True)
class RuntimeDispatchConsequence:
    effects: tuple[RuntimeUiEffect, ...] = ()
    release_result: RuntimeReleaseResult | None = None


@dataclass(frozen=True)
class RuntimePreviewReplayDecision:
    effects: tuple[RuntimeUiEffect, ...] = ()
    defer_context_deactivation_until_after_display: bool = False
    pending_replay_queued: bool = False


@dataclass(frozen=True)
class RuntimeDispatchState:
    plan: RuntimeDispatchPlan | None = None
    expected_completion_set_ids: frozenset[str] = frozenset()
    accepted_completion_set_ids: frozenset[str] = frozenset()
    pending_release_reason: RuntimeReleaseReason | None = None
    release_requires_backend_idle: bool = False
    release_result: RuntimeReleaseResult | None = None

    @classmethod
    def started(cls, dispatch_plan: RuntimeDispatchPlan) -> "RuntimeDispatchState":
        return cls(
            plan=dispatch_plan,
            expected_completion_set_ids=frozenset(
                str(descriptor.set_id)
                for descriptor in dispatch_plan.ordered_task_descriptors or ()
                if str(descriptor.set_id)
            ),
        )

    @property
    def active(self) -> bool:
        return (
            self.plan is not None
            and self.release_result is None
            and self.pending_release_reason is None
        )

    @property
    def all_expected_completions_consumed(self) -> bool:
        if not self.expected_completion_set_ids:
            return False
        return self.expected_completion_set_ids.issubset(self.accepted_completion_set_ids)

    def accepted_completion(self, set_id: str) -> "RuntimeDispatchState":
        normalized = str(set_id or "").strip()
        if not normalized:
            return self
        accepted = set(self.accepted_completion_set_ids)
        accepted.add(normalized)
        return RuntimeDispatchState(
            plan=self.plan,
            expected_completion_set_ids=self.expected_completion_set_ids,
            accepted_completion_set_ids=frozenset(accepted),
            pending_release_reason=self.pending_release_reason,
            release_requires_backend_idle=self.release_requires_backend_idle,
            release_result=self.release_result,
        )

    def with_pending_release(
        self,
        reason: RuntimeReleaseReason,
        *,
        after_backend_idle: bool,
    ) -> "RuntimeDispatchState":
        if self.plan is None:
            return self
        return RuntimeDispatchState(
            plan=self.plan,
            expected_completion_set_ids=self.expected_completion_set_ids,
            accepted_completion_set_ids=self.accepted_completion_set_ids,
            pending_release_reason=reason,
            release_requires_backend_idle=bool(after_backend_idle),
            release_result=self.release_result,
        )

    def with_release_result(
        self,
        release_result: RuntimeReleaseResult,
    ) -> "RuntimeDispatchState":
        return RuntimeDispatchState(
            plan=self.plan,
            expected_completion_set_ids=self.expected_completion_set_ids,
            accepted_completion_set_ids=self.accepted_completion_set_ids,
            pending_release_reason=None,
            release_requires_backend_idle=False,
            release_result=release_result,
        )


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _normalized_targets(values: object) -> tuple[str, ...]:
    if not values:
        return ()
    if isinstance(values, str):
        values = (values,)
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:  # type: ignore[union-attr]
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        normalized.append(text)
    return tuple(normalized)


def _normalized_dirty_generations(
    *,
    target_set_ids: Sequence[str],
    dirty_generation_by_set_id: Mapping[str, object],
) -> tuple[tuple[str, int], ...] | None:
    normalized: list[tuple[str, int]] = []
    for set_id in _normalized_targets(target_set_ids):
        generation = _optional_int(dirty_generation_by_set_id.get(str(set_id)))
        if generation is None:
            return None
        normalized.append((str(set_id), int(generation)))
    return tuple(normalized)


class SimulationRuntimeOrchestrator:
    def __init__(
        self,
        *,
        allocator: RuntimeLaneAllocator,
        backend: RuntimeBackendPort | None = None,
        render: Callable[[SimulationRuntimeReadinessRenderState], object] | None = None,
        current_runtime_input_epochs: Callable[[Sequence[str]], object] | None = None,
        prepared_request_is_current: Callable[[PreparedRuntimeRequestSet], bool] | None = None,
        next_preview_replay_request_id: Callable[[], int] | None = None,
        completion_consumer: RuntimeCompletionConsumer | None = None,
    ) -> None:
        self._allocator = allocator
        self._backend = backend
        self._render = render
        self._current_runtime_input_epochs = current_runtime_input_epochs
        self._prepared_request_is_current = prepared_request_is_current
        self._next_preview_replay_request_id = next_preview_replay_request_id
        self._completion_consumer = completion_consumer
        self._last_prepared: PreparedRuntimeRequestSet | None = None
        self._last_dispatch_plan: RuntimeDispatchPlan | None = None
        self._manual_retry_prepared: PreparedRuntimeRequestSet | None = None
        self._backend_warmup_prepared: PreparedRuntimeRequestSet | None = None
        self._preview_replay_prepared: PreparedRuntimeRequestSet | None = None
        self._dispatch_state = RuntimeDispatchState()

    @property
    def endpoint_state(self) -> SimulationRuntimeReadinessEndpointState:
        pending = self._backend_warmup_prepared or self._manual_retry_prepared
        return SimulationRuntimeReadinessEndpointState(
            manual_retry_available=self._manual_retry_prepared is not None,
            backend_warmup_pending=self._backend_warmup_prepared is not None,
            pending_intent_kind="" if pending is None else str(pending.intent.intent_kind),
        )

    @property
    def last_dispatch_plan(self) -> RuntimeDispatchPlan | None:
        return self._last_dispatch_plan

    @property
    def retry_available(self) -> bool:
        return self._manual_retry_prepared is not None

    @property
    def manual_retry_available(self) -> bool:
        return self._manual_retry_prepared is not None

    @property
    def backend_warmup_pending(self) -> bool:
        return self._backend_warmup_prepared is not None

    @property
    def all_expected_completions_consumed(self) -> bool:
        return self._dispatch_state.all_expected_completions_consumed

    @property
    def pending_intent_kind(self) -> str:
        pending = self._backend_warmup_prepared or self._manual_retry_prepared
        return "" if pending is None else str(pending.intent.intent_kind)

    def set_completion_consumer(self, completion_consumer: RuntimeCompletionConsumer | None) -> None:
        self._completion_consumer = completion_consumer

    def refresh_readiness(self, prepared: PreparedRuntimeRequestSet) -> bool:
        self._last_prepared = prepared
        self._manual_retry_prepared = None
        self._last_dispatch_plan = None
        if not prepared.prepared:
            reason = prepared.blocked_reason
            state = SimulationRuntimeReadinessRenderState(
                status="blocked",
                status_text=(
                    str(reason.message)
                    if reason is not None
                    else "Select at least one set before running."
                ),
                failed=True,
                retryable=bool(reason.retryable) if reason is not None else False,
                launch_available=False,
                preview_available=False,
            )
            self._render_state(state)
            return False
        if not self._prepared_is_current(prepared):
            state = SimulationRuntimeReadinessRenderState(
                status="stale",
                status_text="Runtime launch readiness changed before backend warmup completed.",
                failed=True,
                launch_available=False,
            )
            self._render_state(state)
            return False
        self._allocator.ensure_ready_lanes(
            compatibility_key=prepared.compatibility_key,
            capacity=prepared.preferred_lane_capacity,
            task_count=len(prepared.task_descriptors),
            nonblocking=True,
        )
        allocation = self._allocator.allocate(
            RuntimeLaneAllocationRequest(
                compatibility_key=prepared.compatibility_key,
                required_lane_capacity=prepared.required_lane_capacity,
                preferred_lane_capacity=prepared.preferred_lane_capacity,
                task_count=len(prepared.task_descriptors),
                request_token=prepared.intent.request_token,
                scope=prepared.intent.intent_kind,
                require_backend_lease=True,
            )
        )
        if allocation.status != "ready":
            self._backend_warmup_prepared = prepared
            state = SimulationRuntimeReadinessRenderState(
                status="warming",
                status_text=allocation.message or "Preparing runtime lanes...",
                launch_available=False,
                preview_available=prepared.intent.intent_kind == "preview",
            )
            self._render_state(state)
            return False
        consumed = self._allocator.consume(allocation, prepared, expected=prepared)
        if consumed.dispatch_plan is None:
            state = SimulationRuntimeReadinessRenderState(
                status="blocked",
                status_text=consumed.message or "Runtime request is blocked.",
                failed=True,
                retryable=consumed.retryable,
                launch_available=False,
            )
            self._render_state(state)
            return False
        self._last_dispatch_plan = consumed.dispatch_plan
        self._render_state(
            SimulationRuntimeReadinessRenderState(
                status="ready",
                status_text="Ready",
                launch_available=prepared.intent.intent_kind != "preview",
                preview_available=prepared.intent.intent_kind == "preview",
            )
        )
        return True

    def accept_prepared_request(self, prepared: PreparedRuntimeRequestSet) -> RuntimeDispatchPlan | None:
        if not self.refresh_readiness(prepared):
            return None
        return self._last_dispatch_plan

    def retry_runtime_readiness(self) -> RuntimeDispatchPlan | None:
        prepared = self._manual_retry_prepared or self._backend_warmup_prepared or self._last_prepared
        if prepared is None:
            return None
        return self.accept_prepared_request(prepared)

    def prewarm_compatible_runtime_lanes(self, prepared: PreparedRuntimeRequestSet) -> bool:
        if not prepared.prepared:
            return False
        self._allocator.ensure_ready_lanes(
            compatibility_key=prepared.compatibility_key,
            capacity=prepared.preferred_lane_capacity,
            task_count=len(prepared.task_descriptors),
            nonblocking=True,
        )
        return True

    def dispatch_started(self, dispatch_plan: RuntimeDispatchPlan) -> RuntimeDispatchConsequence:
        self._start_dispatch_state(dispatch_plan)
        descriptors = tuple(dispatch_plan.ordered_task_descriptors or ())
        accepted_capacity = max(1, int(dispatch_plan.launch_allocation.accepted_capacity or 1))
        fast_mode = dispatch_plan.launch_allocation.launch_intent.intent_kind == "preview"
        return RuntimeDispatchConsequence(
            effects=(
                RuntimeUiEffect(
                    start_completion_polling=True,
                    simulation_running=True,
                    slider_simulation_active=bool(fast_mode),
                    run_enabled=False,
                    stop_enabled=True,
                    progress_value=0,
                    status_text=(
                        f"Running {len(descriptors)} sets on {accepted_capacity} runtime lanes..."
                    ),
                ),
            )
        )

    def dispatch_aborted(
        self,
        dispatch_plan: RuntimeDispatchPlan | None = None,
        *,
        message: str = "",
        retryable: bool = True,
        backend_failure: bool = False,
    ) -> RuntimeDispatchConsequence:
        if dispatch_plan is not None:
            self._start_dispatch_state(dispatch_plan)
        if self._backend is not None:
            self._backend.close_current_run(force_terminate=True)
        effects: list[RuntimeUiEffect] = [
            RuntimeUiEffect(
                stop_completion_polling=True,
                simulation_running=False,
                slider_simulation_active=False,
                run_enabled=True,
                stop_enabled=False,
                surface_failure=str(message or ""),
                render_state=self._failure_render_state(message, retryable=retryable)
                if message
                else None,
            )
        ]
        return RuntimeDispatchConsequence(
            effects=tuple(effects),
            release_result=self._release_current_dispatch(
                RuntimeReleaseReason.FAILURE,
                backend_failure=bool(backend_failure),
            ),
        )

    def dispatch_rejected(
        self,
        dispatch_plan: RuntimeDispatchPlan,
        *,
        message: str = "",
        retryable: bool = True,
    ) -> RuntimeDispatchConsequence:
        effect = RuntimeUiEffect(
            simulation_running=False,
            slider_simulation_active=False,
            run_enabled=True,
            stop_enabled=False,
            surface_failure=str(message or ""),
            render_state=self._failure_render_state(message, retryable=retryable)
            if message
            else None,
        )
        return RuntimeDispatchConsequence(
            effects=(effect,),
            release_result=self._release_dispatch_plan(
                dispatch_plan,
                RuntimeReleaseReason.NEUTRAL_RETAIN,
            )
        )

    def cancel_requested(self, *, kind: str = "shutdown") -> list[RuntimeUiEffect]:
        if kind in {
            "terminal_failure",
            "preview_failure",
            "parallel_outcome_reset",
        }:
            return self.terminal_failure()
        if kind == "polling_failure":
            return self.terminal_failure(backend_failed=True)
        if kind == "pool_settings_changed":
            return self.pool_settings_changed()
        if kind == "soft_supersede":
            if self._backend is None:
                self._release_current_dispatch(RuntimeReleaseReason.SUPERSEDED)
                return [RuntimeUiEffect(stop_completion_polling=True)]
            result = self._backend.supersede_current_run()
            if result.running > 0:
                self._defer_current_dispatch_release(RuntimeReleaseReason.SUPERSEDED)
                return []
            else:
                self._release_current_dispatch(RuntimeReleaseReason.SUPERSEDED)
            return [RuntimeUiEffect(stop_completion_polling=True)]
        if self._backend is not None:
            self._backend.close_current_run(force_terminate=kind != "soft_shutdown")
        self._release_current_dispatch(RuntimeReleaseReason.SHUTDOWN)
        return [RuntimeUiEffect(stop_completion_polling=True)]

    def display_completed(self, *, kind: str = "success") -> RuntimeReleaseResult:
        if kind in {"scoped_failure", "failure", "preview_failure"}:
            return self._release_current_dispatch(RuntimeReleaseReason.FAILURE)
        if kind == "stale_fast":
            self._defer_current_dispatch_release(RuntimeReleaseReason.SUCCESS_RETAIN)
            return RuntimeReleaseResult(status="deferred", reason=RuntimeReleaseReason.SUCCESS_RETAIN)
        return self._release_current_dispatch(RuntimeReleaseReason.SUCCESS_RETAIN)

    def consume_progress_tick(self) -> list[RuntimeUiEffect]:
        backend = self._backend
        if backend is None:
            self._release_pending_dispatch_plan_after_backend_idle(backend_idle=True)
            return self._polling_effects_after_runtime_consequence_update()
        try:
            poll_result = backend.poll_completed_records()
            polled_records = tuple(poll_result.records)
            backend_idle = bool(poll_result.backend_idle)
            if not polled_records:
                released = self._release_pending_dispatch_plan_after_backend_idle(
                    backend_idle=backend_idle,
                )
                if released is not None:
                    return self._polling_effects_after_runtime_consequence_update()
                if self._dispatch_state.active or self._dispatch_state.pending_release_reason is not None:
                    return []
                return self._polling_effects_after_runtime_consequence_update()
            if not polled_records and self._dispatch_state.active:
                return []
            for polled in polled_records:
                record = polled.record
                event = RuntimeCompletionEvent(
                    set_id=str(polled.set_id or ""),
                    record=record,
                    outcome=getattr(record, "outcome", None),
                    source=str(polled.source or ""),
                    completed_ts=float(polled.completed_ts or 0.0),
                )
                consumer = self._completion_consumer
                decision = (
                    consumer.consume_runtime_completion(event)
                    if consumer is not None
                    else RuntimeCompletionDecision.accepted_current()
                )
                if decision.terminal:
                    if decision.failed:
                        self._release_current_dispatch(RuntimeReleaseReason.FAILURE)
                    return [
                        RuntimeUiEffect(
                            stop_completion_polling=True,
                            surface_failure=str(decision.message or ""),
                        )
                    ]
                if not decision.accepted:
                    return []
                self._record_accepted_completion(event)
            self._release_pending_dispatch_plan_after_backend_idle(backend_idle=backend_idle)
            return self._polling_effects_after_runtime_consequence_update()
        except Exception as exc:
            if backend is not None:
                backend.close_current_run(force_terminate=True)
            self._release_current_dispatch(
                RuntimeReleaseReason.FAILURE,
                backend_failure=True,
            )
            return [
                RuntimeUiEffect(
                    stop_completion_polling=True,
                    surface_failure=str(exc),
                )
            ]

    def pool_settings_changed(self) -> list[RuntimeUiEffect]:
        return self.cancel_requested(kind="soft_shutdown")

    def project_applied(self) -> list[RuntimeUiEffect]:
        return self.cancel_requested(kind="shutdown")

    def polling_failed(self, message: str) -> list[RuntimeUiEffect]:
        return [
            RuntimeUiEffect(
                stop_completion_polling=True,
                simulation_running=False,
                slider_simulation_active=False,
                run_enabled=True,
                stop_enabled=False,
                surface_failure=str(message or ""),
                render_state=self._failure_render_state(message, retryable=True),
            )
        ] + self.terminal_failure(backend_failed=True)

    def terminal_failure(self, *, backend_failed: bool = False) -> list[RuntimeUiEffect]:
        if self._backend is not None:
            self._backend.close_current_run(force_terminate=True)
        self._release_current_dispatch(
            RuntimeReleaseReason.FAILURE,
            backend_failure=bool(backend_failed),
        )
        return [RuntimeUiEffect(stop_completion_polling=True)]

    def close_requested(self, *, force_terminate: bool = True) -> list[RuntimeUiEffect]:
        if self._backend is not None:
            self._backend.close_current_run(force_terminate=bool(force_terminate))
        self._release_current_dispatch(RuntimeReleaseReason.SHUTDOWN)
        return [RuntimeUiEffect(stop_completion_polling=True)]

    def preview_replay_exists(
        self,
        *,
        active: bool,
        target_set_ids: Sequence[str],
    ) -> bool:
        return bool(active or _normalized_targets(target_set_ids))

    def preview_replay_exists_for_state(self, state: object) -> bool:
        pending = (
            state
            if isinstance(state, RuntimePreviewReplayState)
            else RuntimePreviewReplayState.from_pending(state)
        )
        return self.preview_replay_exists(
            active=bool(pending.active),
            target_set_ids=pending.target_set_ids,
        )

    def preview_replay_requested(
        self,
        *,
        active: bool,
        request_id: int | None,
        target_set_ids: Sequence[str],
        handoff_queued: bool,
        stop_timers: bool = True,
    ) -> list[RuntimeUiEffect]:
        targets = _normalized_targets(target_set_ids)
        if not self.preview_replay_exists(active=bool(active), target_set_ids=targets):
            return []
        if bool(handoff_queued):
            return []
        replay_request_id = request_id
        if replay_request_id is None and callable(self._next_preview_replay_request_id):
            replay_request_id = self._next_preview_replay_request_id()
        if replay_request_id is None:
            return []
        return [
            RuntimeUiEffect(
                queue_preview_replay=QueuePreviewReplay(
                    request_id=int(replay_request_id),
                    target_set_ids=targets,
                    stop_timers=bool(stop_timers),
                )
            )
        ]

    def preview_replay_state_requested(
        self,
        *,
        current_state: object,
        target_set_ids: Sequence[str],
        request_id: int | None = None,
        preserve_existing_request: bool = False,
        clear_plot_updates: bool = False,
    ) -> list[RuntimeUiEffect]:
        current = RuntimePreviewReplayState.from_pending(current_state)
        normalized_targets = _normalized_targets(target_set_ids)
        next_request_id: int | None
        if request_id is not None:
            next_request_id = int(request_id)
        elif bool(preserve_existing_request):
            next_request_id = current.request_id
        elif callable(self._next_preview_replay_request_id):
            next_request_id = self._next_preview_replay_request_id()
        else:
            next_request_id = None
        preserve_handoff_queued = bool(
            current.handoff_queued
            and current.target_set_ids == normalized_targets
            and current.request_id == next_request_id
        )
        state = RuntimePreviewReplayState(
            active=True,
            request_id=next_request_id,
            target_set_ids=normalized_targets,
            handoff_queued=preserve_handoff_queued,
            replay_generation=int(current.replay_generation) + 1,
        )
        return [
            RuntimeUiEffect(
                set_preview_replay=RuntimePreviewReplayUpdate(
                    state=state,
                    clear_plot_updates=bool(clear_plot_updates),
                )
            )
        ]

    def preview_replay_cleared(self, *, clear_plot_updates: bool = True) -> list[RuntimeUiEffect]:
        return [
            RuntimeUiEffect(
                clear_preview_replay=True,
                clear_preview_plot_updates=bool(clear_plot_updates),
                stop_debounce_timers=bool(clear_plot_updates),
            )
        ]

    def preview_replay_after_preflight_abort(
        self,
        *,
        current_state: object,
        explicit_run: bool,
    ) -> list[RuntimeUiEffect]:
        pending = RuntimePreviewReplayState.from_pending(current_state)
        if not bool(explicit_run) or not self.preview_replay_exists_for_state(pending):
            return []
        return self.preview_replay_state_requested(
            current_state=pending,
            target_set_ids=pending.target_set_ids,
            preserve_existing_request=True,
        )

    def preview_replay_after_canonical_reset(
        self,
        *,
        current_state: object,
        reset_set_ids: Sequence[str],
    ) -> list[RuntimeUiEffect]:
        pending = RuntimePreviewReplayState.from_pending(current_state)
        reset_ids = set(_normalized_targets(reset_set_ids))
        surviving = tuple(set_id for set_id in pending.target_set_ids if set_id not in reset_ids)
        if surviving:
            return self.preview_replay_state_requested(
                current_state=pending,
                target_set_ids=surviving,
                preserve_existing_request=True,
                clear_plot_updates=True,
            )
        return self.preview_replay_cleared(clear_plot_updates=True)

    def stale_fast_completion_replay_decision(
        self,
        *,
        current_state: object,
        display_current_preview: bool,
        shutdown_requested: bool,
        context_parallel: bool,
    ) -> RuntimePreviewReplayDecision:
        pending = RuntimePreviewReplayState.from_pending(current_state)
        replay_exists = self.preview_replay_exists_for_state(pending)
        should_queue = bool(
            replay_exists
            and not bool(pending.handoff_queued)
            and not bool(shutdown_requested)
        )
        if bool(display_current_preview):
            if not replay_exists or bool(shutdown_requested):
                return RuntimePreviewReplayDecision()
            effects = tuple(
                self.preview_replay_state_requested(
                    current_state=pending,
                    target_set_ids=pending.target_set_ids,
                    preserve_existing_request=True,
                )
            )
            return RuntimePreviewReplayDecision(
                effects=effects,
                defer_context_deactivation_until_after_display=bool(
                    should_queue and not bool(context_parallel)
                ),
                pending_replay_queued=False,
            )
        if not replay_exists or bool(shutdown_requested):
            return RuntimePreviewReplayDecision(
                effects=tuple(self.preview_replay_cleared(clear_plot_updates=False)),
            )
        effects = tuple(
            self.preview_replay_requested(
                active=bool(pending.active),
                request_id=pending.request_id,
                target_set_ids=pending.target_set_ids,
                handoff_queued=bool(pending.handoff_queued),
                stop_timers=False,
            )
        )
        return RuntimePreviewReplayDecision(
            effects=effects,
            pending_replay_queued=bool(effects),
        )

    def stale_fast_error_replay_decision(
        self,
        *,
        current_state: object,
    ) -> RuntimePreviewReplayDecision:
        pending = RuntimePreviewReplayState.from_pending(current_state)
        if not self.preview_replay_exists_for_state(pending):
            return RuntimePreviewReplayDecision(
                effects=tuple(self.preview_replay_cleared(clear_plot_updates=False)),
            )
        effects = tuple(
            self.preview_replay_requested(
                active=bool(pending.active),
                request_id=pending.request_id,
                target_set_ids=pending.target_set_ids,
                handoff_queued=bool(pending.handoff_queued),
                stop_timers=False,
            )
        )
        return RuntimePreviewReplayDecision(
            effects=effects,
            pending_replay_queued=bool(effects),
        )

    def pending_preview_replay_requested(
        self,
        *,
        current_state: object,
        shutdown_requested: bool = False,
        stop_timers: bool = False,
    ) -> list[RuntimeUiEffect]:
        if bool(shutdown_requested):
            return []
        pending = RuntimePreviewReplayState.from_pending(current_state)
        return self.preview_replay_requested(
            active=bool(pending.active),
            request_id=pending.request_id,
            target_set_ids=pending.target_set_ids,
            handoff_queued=bool(pending.handoff_queued),
            stop_timers=bool(stop_timers),
        )

    def terminal_failure_replay_snapshot(
        self,
        *,
        pending_state: object,
        dirty_generation_by_set_id: Mapping[str, object],
    ) -> RuntimePreviewReplaySnapshot:
        pending = (
            pending_state
            if isinstance(pending_state, RuntimePreviewReplayState)
            else RuntimePreviewReplayState.from_pending(pending_state)
        )
        target_set_ids = _normalized_targets(pending.target_set_ids)
        if not self.preview_replay_exists(active=bool(pending.active), target_set_ids=target_set_ids):
            return RuntimePreviewReplaySnapshot(
                active=False,
                request_id=pending.request_id,
                target_set_ids=target_set_ids,
                replay_generation=int(pending.replay_generation),
            )
        dirty_generations = _normalized_dirty_generations(
            target_set_ids=target_set_ids,
            dirty_generation_by_set_id=dirty_generation_by_set_id,
        )
        if dirty_generations is None:
            return RuntimePreviewReplaySnapshot(
                active=False,
                request_id=pending.request_id,
                target_set_ids=target_set_ids,
                replay_generation=int(pending.replay_generation),
            )
        return RuntimePreviewReplaySnapshot(
            active=True,
            request_id=pending.request_id,
            target_set_ids=target_set_ids,
            replay_generation=int(pending.replay_generation),
            dirty_generation_by_set_id=dirty_generations,
        )

    def terminal_failure_replay_requested(
        self,
        *,
        fast_mode: bool,
        pending_state: object,
        replay_snapshot: RuntimePreviewReplaySnapshot | None,
        dirty_generation_by_set_id: Mapping[str, object],
    ) -> list[RuntimeUiEffect]:
        pending = (
            pending_state
            if isinstance(pending_state, RuntimePreviewReplayState)
            else RuntimePreviewReplayState.from_pending(pending_state)
        )
        if (
            replay_snapshot is not None
            and self._terminal_failure_replay_snapshot_matches_pending(
                snapshot=replay_snapshot,
                pending=pending,
            )
            and not self._terminal_failure_replay_snapshot_still_current(
                snapshot=replay_snapshot,
                dirty_generation_by_set_id=dirty_generation_by_set_id,
            )
        ):
            return [RuntimeUiEffect(clear_preview_replay=True, clear_preview_plot_updates=False)]
        if bool(fast_mode) or not self.preview_replay_exists_for_state(pending):
            return [RuntimeUiEffect(clear_preview_replay=True, clear_preview_plot_updates=False)]
        return self.preview_replay_requested(
            active=bool(pending.active),
            request_id=pending.request_id,
            target_set_ids=pending.target_set_ids,
            handoff_queued=bool(pending.handoff_queued),
            stop_timers=True,
        )

    def completion_without_result_finalized(
        self,
        *,
        pending_state: object,
        shutdown_requested: bool,
        display_kind: str = "success",
    ) -> list[RuntimeUiEffect]:
        self.display_completed(kind=display_kind or "success")
        effects: list[RuntimeUiEffect] = [
            RuntimeUiEffect(
                simulation_running=False,
                slider_simulation_active=False,
                run_enabled=True,
                stop_enabled=False,
            )
        ]
        if bool(shutdown_requested):
            return effects
        pending = (
            pending_state
            if isinstance(pending_state, RuntimePreviewReplayState)
            else RuntimePreviewReplayState.from_pending(pending_state)
        )
        effects.extend(
            self.preview_replay_requested(
                active=bool(pending.active),
                request_id=pending.request_id,
                target_set_ids=pending.target_set_ids,
                handoff_queued=bool(pending.handoff_queued),
                stop_timers=True,
            )
        )
        return effects

    def render_failure(self, message: str, *, retryable: bool = True) -> None:
        self._manual_retry_prepared = self._last_prepared if bool(retryable) else None
        self._render_state(
            SimulationRuntimeReadinessRenderState(
                status="failed",
                status_text=str(message),
                launch_available=bool(retryable),
                failed=True,
                retryable=bool(retryable),
            )
        )

    def _prepared_is_current(self, prepared: PreparedRuntimeRequestSet) -> bool:
        provider = self._prepared_request_is_current
        if provider is None:
            return True
        try:
            return bool(provider(prepared))
        except Exception:
            return False

    def _start_dispatch_state(self, dispatch_plan: RuntimeDispatchPlan) -> None:
        self._last_dispatch_plan = dispatch_plan
        self._dispatch_state = RuntimeDispatchState.started(dispatch_plan)

    def _record_accepted_completion(self, event: RuntimeCompletionEvent) -> None:
        self._dispatch_state = self._dispatch_state.accepted_completion(event.set_id)

    def _release_current_dispatch(
        self,
        reason: RuntimeReleaseReason,
        *,
        backend_failure: bool = False,
    ) -> RuntimeReleaseResult:
        plan = self._dispatch_state.plan
        if plan is None:
            return RuntimeReleaseResult(status="missing", reason=reason)
        return self._release_dispatch_plan(
            plan,
            reason,
            backend_failure=bool(backend_failure),
        )

    def _release_dispatch_plan(
        self,
        dispatch_plan: RuntimeDispatchPlan,
        reason: RuntimeReleaseReason,
        *,
        backend_failure: bool = False,
    ) -> RuntimeReleaseResult:
        allocation_id = str(dispatch_plan.launch_allocation.allocation_id)
        result = self._allocator.release(
            allocation_id,
            reason=reason,
            backend_failure=bool(backend_failure),
        )
        if self._dispatch_state.plan is dispatch_plan:
            if reason is RuntimeReleaseReason.SUCCESS_RETAIN:
                self._dispatch_state = self._dispatch_state.with_release_result(result)
            else:
                self._dispatch_state = RuntimeDispatchState()
        return result

    def _defer_current_dispatch_release(self, reason: RuntimeReleaseReason) -> None:
        if self._dispatch_state.plan is None:
            return
        self._dispatch_state = self._dispatch_state.with_pending_release(
            reason,
            after_backend_idle=True,
        )

    def _release_pending_dispatch_plan_after_backend_idle(
        self,
        *,
        backend_idle: bool,
    ) -> RuntimeReleaseResult | None:
        plan = self._dispatch_state.plan
        reason = self._dispatch_state.pending_release_reason
        if plan is None or reason is None:
            return None
        if self._dispatch_state.release_requires_backend_idle and not bool(backend_idle):
            return None
        return self._release_dispatch_plan(plan, reason)

    def _runtime_consequences_complete(self) -> bool:
        state = self._dispatch_state
        if state.plan is None:
            return True
        if state.active or state.pending_release_reason is not None:
            return False
        if state.release_result is None:
            return False
        return state.all_expected_completions_consumed

    def _polling_effects_after_runtime_consequence_update(self) -> list[RuntimeUiEffect]:
        if not self._runtime_consequences_complete():
            return []
        if self._dispatch_state.release_result is not None:
            self._dispatch_state = RuntimeDispatchState()
        return [RuntimeUiEffect(stop_completion_polling=True)]

    def _render_state(self, state: SimulationRuntimeReadinessRenderState) -> None:
        if callable(self._render):
            self._render(state)

    def _failure_render_state(
        self,
        message: str,
        *,
        retryable: bool,
    ) -> SimulationRuntimeReadinessRenderState:
        return SimulationRuntimeReadinessRenderState(
            status="failed",
            status_text=str(message or ""),
            launch_available=bool(retryable),
            failed=True,
            retryable=bool(retryable),
        )

    @staticmethod
    def _terminal_failure_replay_snapshot_matches_pending(
        *,
        snapshot: RuntimePreviewReplaySnapshot,
        pending: RuntimePreviewReplayState,
    ) -> bool:
        if not bool(snapshot.active) or not snapshot.target_set_ids:
            return False
        return (
            _normalized_targets(pending.target_set_ids) == tuple(snapshot.target_set_ids)
            and pending.request_id == snapshot.request_id
            and int(pending.replay_generation) == int(snapshot.replay_generation)
        )

    @staticmethod
    def _terminal_failure_replay_snapshot_still_current(
        *,
        snapshot: RuntimePreviewReplaySnapshot,
        dirty_generation_by_set_id: Mapping[str, object],
    ) -> bool:
        if not bool(snapshot.active) or not snapshot.target_set_ids:
            return False
        expected = {
            str(set_id): int(generation)
            for set_id, generation in (snapshot.dirty_generation_by_set_id or ())
            if str(set_id)
        }
        if set(expected) != set(snapshot.target_set_ids):
            return False
        current = _normalized_dirty_generations(
            target_set_ids=snapshot.target_set_ids,
            dirty_generation_by_set_id=dirty_generation_by_set_id,
        )
        if current is None:
            return False
        return {
            str(set_id): int(generation)
            for set_id, generation in current
        } == expected
