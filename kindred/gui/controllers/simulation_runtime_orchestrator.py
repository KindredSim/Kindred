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
from kindred.gui.controllers.simulation_completion_policy import (
    normalize_preview_target_set_ids,
)
from kindred.gui.controllers.simulation_runtime_backend import (
    RuntimeBackendPort,
    RuntimeCompletionDecision,
    RuntimeCompletionConsumer,
    RuntimeCompletionEvent,
    RuntimeScopedFailureSummary,
)
from kindred.gui.controllers.simulation_runtime_readiness_lifecycle import (
    SimulationRuntimeReadinessEndpointState,
    SimulationRuntimeReadinessRenderState,
)

_RUNTIME_WARMUP_RETRY_REFRESH_READINESS = "refresh_readiness"
_RUNTIME_WARMUP_RETRY_ACCEPT_DISPATCH = "accept_dispatch"
_RUNTIME_WARMUP_RETRY_KINDS = (
    _RUNTIME_WARMUP_RETRY_REFRESH_READINESS,
    _RUNTIME_WARMUP_RETRY_ACCEPT_DISPATCH,
)
_RUNTIME_WARMUP_RETRY_DELAY_MS = 50


@dataclass(frozen=True)
class QueuePreviewReplay:
    request_id: int | None
    target_set_ids: tuple[str, ...]
    stop_timers: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "target_set_ids",
            normalize_preview_target_set_ids(self.target_set_ids),
        )


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

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_id", _optional_int(self.request_id))
        object.__setattr__(
            self,
            "target_set_ids",
            normalize_preview_target_set_ids(self.target_set_ids),
        )
        object.__setattr__(self, "handoff_queued", bool(self.handoff_queued))
        object.__setattr__(self, "replay_generation", max(0, int(self.replay_generation or 0)))

    @classmethod
    def from_pending(cls, pending: object) -> "RuntimePreviewReplayState":
        return cls(
            active=bool(getattr(pending, "active", False)),
            request_id=_optional_int(getattr(pending, "request_id", None)),
            target_set_ids=normalize_preview_target_set_ids(
                getattr(pending, "target_set_ids", ())
            ),
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

    def __post_init__(self) -> None:
        target_set_ids = normalize_preview_target_set_ids(self.target_set_ids)
        dirty_generations: list[tuple[str, int]] = []
        target_set = set(target_set_ids)
        for set_id, generation in tuple(self.dirty_generation_by_set_id or ()):
            sid = str(set_id or "").strip()
            normalized_generation = _optional_int(generation)
            if sid and sid in target_set and normalized_generation is not None:
                dirty_generations.append((sid, int(normalized_generation)))
        object.__setattr__(self, "request_id", _optional_int(self.request_id))
        object.__setattr__(self, "target_set_ids", target_set_ids)
        object.__setattr__(self, "replay_generation", max(0, int(self.replay_generation or 0)))
        object.__setattr__(self, "dirty_generation_by_set_id", tuple(dirty_generations))


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
    surface_failure_detail_text: str = ""
    preview_failure_status_text: str = ""
    preview_failure_context: Mapping[str, object] | None = None
    superseded_fast_failure: bool = False
    superseded_fast_failure_reset_status_progress: bool = False
    superseded_fast_failure_deactivate_context_immediately: bool = False
    render_state: SimulationRuntimeReadinessRenderState | None = None
    simulation_running: bool | None = None
    slider_simulation_active: bool | None = None
    reset_slider_triggered: bool = False
    run_enabled: bool | None = None
    stop_enabled: bool | None = None
    progress_value: int | None = None
    status_text: str | None = None
    stop_debounce_timers: bool = False
    cancel_warmup_retry_kinds: tuple[str, ...] = ()
    warmup_retry_kind: str = ""
    warmup_retry_delay_ms: int = 0
    scoped_failure_summary: RuntimeScopedFailureSummary | None = None


@dataclass(frozen=True)
class RuntimeDispatchConsequence:
    effects: tuple[RuntimeUiEffect, ...] = ()
    release_result: RuntimeReleaseResult | None = None


@dataclass(frozen=True)
class RuntimeReadinessConsequence:
    launch_available: bool = False
    render_state: SimulationRuntimeReadinessRenderState | None = None
    effects: tuple[RuntimeUiEffect, ...] = ()


@dataclass(frozen=True)
class RuntimeDispatchAcceptance:
    dispatch_plan: RuntimeDispatchPlan | None = None
    effects: tuple[RuntimeUiEffect, ...] = ()


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
    consumed_completion_set_ids: frozenset[str] = frozenset()
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
        return self.expected_completion_set_ids.issubset(self.consumed_completion_set_ids)

    def accepted_completion(self, set_id: str) -> "RuntimeDispatchState":
        normalized = str(set_id or "").strip()
        if not normalized:
            return self
        accepted = set(self.accepted_completion_set_ids)
        consumed = set(self.consumed_completion_set_ids)
        accepted.add(normalized)
        consumed.add(normalized)
        return RuntimeDispatchState(
            plan=self.plan,
            expected_completion_set_ids=self.expected_completion_set_ids,
            accepted_completion_set_ids=frozenset(accepted),
            consumed_completion_set_ids=frozenset(consumed),
            pending_release_reason=self.pending_release_reason,
            release_requires_backend_idle=self.release_requires_backend_idle,
            release_result=self.release_result,
        )

    def consumed_completion(self, set_id: str) -> "RuntimeDispatchState":
        normalized = str(set_id or "").strip()
        if not normalized:
            return self
        consumed = set(self.consumed_completion_set_ids)
        consumed.add(normalized)
        return RuntimeDispatchState(
            plan=self.plan,
            expected_completion_set_ids=self.expected_completion_set_ids,
            accepted_completion_set_ids=self.accepted_completion_set_ids,
            consumed_completion_set_ids=frozenset(consumed),
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
            consumed_completion_set_ids=self.consumed_completion_set_ids,
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
            consumed_completion_set_ids=self.consumed_completion_set_ids,
            pending_release_reason=None,
            release_requires_backend_idle=False,
            release_result=release_result,
        )


@dataclass(frozen=True)
class RuntimeDeferredReleaseState:
    dispatch_plan: RuntimeDispatchPlan | None = None
    reason: RuntimeReleaseReason | None = None
    wait_for_backend_idle: bool = False
    superseded_drain_token: str = ""


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _normalized_dirty_generations(
    *,
    target_set_ids: Sequence[str],
    dirty_generation_by_set_id: Mapping[str, object],
) -> tuple[tuple[str, int], ...] | None:
    normalized: list[tuple[str, int]] = []
    for set_id in normalize_preview_target_set_ids(target_set_ids):
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
        next_request_id: Callable[[], int] | None = None,
        reserve_request_id: Callable[[], int] | None = None,
        completion_consumer: RuntimeCompletionConsumer | None = None,
        preview_replay_state: Callable[[], object] | None = None,
        dirty_generation_by_set_id: Callable[[Sequence[str]], Mapping[str, object]] | None = None,
    ) -> None:
        self._allocator = allocator
        self._backend = backend
        self._render = render
        self._current_runtime_input_epochs = current_runtime_input_epochs
        self._prepared_request_is_current = prepared_request_is_current
        self._next_request_id = next_request_id
        self._reserve_request_id = reserve_request_id
        self._completion_consumer = completion_consumer
        self._preview_replay_state = preview_replay_state
        self._dirty_generation_by_set_id = dirty_generation_by_set_id
        self._last_dispatch_plan: RuntimeDispatchPlan | None = None
        self._manual_retry_prepared: PreparedRuntimeRequestSet | None = None
        self._manual_retry_kind: str = ""
        self._readiness_retry_prepared: PreparedRuntimeRequestSet | None = None
        self._acceptance_retry_prepared: PreparedRuntimeRequestSet | None = None
        self._dispatch_state = RuntimeDispatchState()
        self._deferred_release_states: tuple[RuntimeDeferredReleaseState, ...] = ()

    @property
    def endpoint_state(self) -> SimulationRuntimeReadinessEndpointState:
        pending = self._pending_retry_prepared()
        return SimulationRuntimeReadinessEndpointState(
            manual_retry_available=self._manual_retry_prepared is not None,
            backend_warmup_pending=bool(
                self._readiness_retry_prepared is not None
                or self._acceptance_retry_prepared is not None
            ),
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
        return bool(
            self._readiness_retry_prepared is not None
            or self._acceptance_retry_prepared is not None
        )

    @property
    def all_expected_completions_consumed(self) -> bool:
        return self._dispatch_state.all_expected_completions_consumed

    @property
    def pending_intent_kind(self) -> str:
        pending = self._pending_retry_prepared()
        return "" if pending is None else str(pending.intent.intent_kind)

    def set_completion_consumer(self, completion_consumer: RuntimeCompletionConsumer | None) -> None:
        self._completion_consumer = completion_consumer

    def refresh_readiness_consequence(
        self,
        prepared: PreparedRuntimeRequestSet,
    ) -> RuntimeReadinessConsequence:
        self._manual_retry_prepared = None
        self._manual_retry_kind = ""
        self._last_dispatch_plan = None
        self._clear_warmup_retry_prepared(_RUNTIME_WARMUP_RETRY_REFRESH_READINESS)
        return self._probe_readiness_consequence(
            prepared,
            warmup_retry_kind=_RUNTIME_WARMUP_RETRY_REFRESH_READINESS,
        )

    def refresh_readiness(self, prepared: PreparedRuntimeRequestSet) -> bool:
        return bool(self.refresh_readiness_consequence(prepared).launch_available)

    def accept_prepared_request(self, prepared: PreparedRuntimeRequestSet) -> RuntimeDispatchAcceptance:
        self._manual_retry_prepared = None
        self._manual_retry_kind = ""
        self._last_dispatch_plan = None
        self._clear_warmup_retry_prepared(_RUNTIME_WARMUP_RETRY_REFRESH_READINESS)
        self._clear_warmup_retry_prepared(_RUNTIME_WARMUP_RETRY_ACCEPT_DISPATCH)
        cancel_readiness_retry = (
            self._cancel_warmup_retry_effect(_RUNTIME_WARMUP_RETRY_REFRESH_READINESS),
        )
        readiness = self._probe_readiness_consequence(
            prepared,
            warmup_retry_kind=_RUNTIME_WARMUP_RETRY_ACCEPT_DISPATCH,
        )
        if not self._readiness_state_is_ready(readiness):
            return RuntimeDispatchAcceptance(
                effects=cancel_readiness_retry + tuple(readiness.effects or ())
            )
        try:
            self._allocator.ensure_ready_lanes(
                compatibility_key=prepared.compatibility_key,
                capacity=prepared.preferred_lane_capacity,
                task_count=len(prepared.task_descriptors),
                nonblocking=True,
            )
            allocation = self._allocator.allocate(self._allocation_request(prepared))
        except Exception as exc:
            self._set_manual_retry_prepared(
                prepared,
                retry_kind=_RUNTIME_WARMUP_RETRY_ACCEPT_DISPATCH,
            )
            self._clear_warmup_retry_prepared(_RUNTIME_WARMUP_RETRY_ACCEPT_DISPATCH)
            return self._dispatch_acceptance_consequence(
                self._failure_render_state(str(exc), retryable=True),
                extra_effects=cancel_readiness_retry,
            )
        if allocation.status != "ready":
            self._store_warmup_retry_prepared(
                _RUNTIME_WARMUP_RETRY_ACCEPT_DISPATCH,
                prepared,
            )
            return self._dispatch_acceptance_consequence(
                SimulationRuntimeReadinessRenderState(
                    status="warming",
                    status_text=allocation.message or "Preparing runtime lanes...",
                    launch_available=False,
                    preview_available=prepared.intent.intent_kind == "preview",
                ),
                extra_effects=(
                    self._warmup_retry_effect(
                        _RUNTIME_WARMUP_RETRY_ACCEPT_DISPATCH,
                        cancel_retry_kinds=(_RUNTIME_WARMUP_RETRY_REFRESH_READINESS,),
                    ),
                ),
            )
        consumed = self._allocator.consume(allocation, prepared, expected=prepared)
        if consumed.dispatch_plan is None:
            return self._dispatch_acceptance_consequence(
                SimulationRuntimeReadinessRenderState(
                    status="blocked",
                    status_text=consumed.message or "Runtime request is blocked.",
                    failed=True,
                    retryable=consumed.retryable,
                    launch_available=False,
                ),
                extra_effects=cancel_readiness_retry,
            )
        self._clear_retry_prepared_states(clear_manual_retry=False)
        self._last_dispatch_plan = consumed.dispatch_plan
        return self._dispatch_acceptance_consequence(
            SimulationRuntimeReadinessRenderState(
                status="ready",
                status_text="Ready",
                launch_available=prepared.intent.intent_kind != "preview",
                preview_available=prepared.intent.intent_kind == "preview",
            ),
            dispatch_plan=self._last_dispatch_plan,
            extra_effects=cancel_readiness_retry,
        )

    def retry_readiness_refresh(self) -> RuntimeReadinessConsequence:
        prepared = self._readiness_retry_prepared
        if prepared is None and self._manual_retry_kind == _RUNTIME_WARMUP_RETRY_REFRESH_READINESS:
            prepared = self._manual_retry_prepared
        if prepared is None:
            return RuntimeReadinessConsequence()
        return self.refresh_readiness_consequence(prepared)

    def retry_dispatch_acceptance(self) -> RuntimeDispatchAcceptance:
        prepared = self._acceptance_retry_prepared
        if prepared is None and self._manual_retry_kind == _RUNTIME_WARMUP_RETRY_ACCEPT_DISPATCH:
            prepared = self._manual_retry_prepared
        if prepared is None:
            return RuntimeDispatchAcceptance()
        return self.accept_prepared_request(prepared)

    def prewarm_compatible_runtime_lanes(
        self,
        prepared: PreparedRuntimeRequestSet,
        *,
        prepared_is_current: Callable[[PreparedRuntimeRequestSet], bool] | None = None,
    ) -> bool:
        if not prepared.prepared:
            return False
        if prepared_is_current is not None and not bool(prepared_is_current(prepared)):
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
                    cancel_warmup_retry_kinds=_RUNTIME_WARMUP_RETRY_KINDS,
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
        release_result, close_failure = self._close_failed_runtime(
            backend_failure=bool(backend_failure),
        )
        effects: list[RuntimeUiEffect] = [
            RuntimeUiEffect(
                stop_completion_polling=True,
                cancel_warmup_retry_kinds=_RUNTIME_WARMUP_RETRY_KINDS,
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
        if close_failure:
            effects.append(RuntimeUiEffect(surface_failure=close_failure))
        return RuntimeDispatchConsequence(
            effects=tuple(effects),
            release_result=release_result,
        )

    def dispatch_rejected(
        self,
        dispatch_plan: RuntimeDispatchPlan,
        *,
        message: str = "",
        retryable: bool = True,
    ) -> RuntimeDispatchConsequence:
        effect = RuntimeUiEffect(
            cancel_warmup_retry_kinds=_RUNTIME_WARMUP_RETRY_KINDS,
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
        if kind == "parallel_outcome_reset":
            return self.parallel_outcome_reset()
        if kind in {
            "terminal_failure",
            "preview_failure",
        }:
            return self.terminal_failure()
        if kind == "polling_failure":
            return self.terminal_failure(backend_failed=True)
        if kind == "pool_settings_changed":
            return self.pool_settings_changed()
        if kind == "soft_supersede":
            if self._backend is None:
                self._clear_retry_prepared_states()
                self._release_current_dispatch(RuntimeReleaseReason.SUPERSEDED)
                self._release_invalidated_dispatches(reason=RuntimeReleaseReason.SUPERSEDED)
                return [self._ready_reset_effect(stop_completion_polling=True)]
            try:
                result = self._backend.supersede_current_run()
            except Exception as exc:
                _, close_failure = self._close_failed_runtime(backend_failure=True)
                message = f"Runtime backend supersede failed: {exc}"
                if close_failure:
                    message = f"{message}; {close_failure}"
                return [
                    self._ready_reset_effect(stop_completion_polling=True),
                    RuntimeUiEffect(surface_failure=message),
                ]
            if result.running > 0:
                if not str(result.superseded_drain_token or ""):
                    _, close_failure = self._close_failed_runtime(backend_failure=True)
                    message = "Runtime backend soft-supersede reported running work with a missing superseded drain token."
                    if close_failure:
                        message = f"{message}; {close_failure}"
                    return [
                        self._ready_reset_effect(stop_completion_polling=True),
                        RuntimeUiEffect(surface_failure=message),
                    ]
                self._clear_retry_prepared_states()
                self._last_dispatch_plan = None
                self._stage_invalidated_dispatch_release(
                    RuntimeReleaseReason.SUPERSEDED,
                    superseded_drain_token=result.superseded_drain_token,
                )
                return [self._ready_reset_effect(stop_completion_polling=False)]
            self._clear_retry_prepared_states()
            self._release_current_dispatch(RuntimeReleaseReason.SUPERSEDED)
            return [
                self._ready_reset_effect(
                    stop_completion_polling=not bool(self._deferred_release_states)
                )
            ]
        self._clear_retry_prepared_states()
        self._release_current_dispatch(RuntimeReleaseReason.SHUTDOWN)
        self._release_invalidated_dispatches(reason=RuntimeReleaseReason.SHUTDOWN)
        close_failure = self._close_backend_run(
            force_terminate=kind != "soft_shutdown",
            reason=RuntimeReleaseReason.SHUTDOWN,
        )
        effects = [self._ready_reset_effect(stop_completion_polling=True)]
        if close_failure:
            effects.append(RuntimeUiEffect(surface_failure=close_failure))
        return effects

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
            drained_superseded_release_tokens = tuple(
                poll_result.drained_superseded_release_tokens
            )
            if not polled_records:
                released = self._release_pending_dispatch_plan_after_backend_idle(
                    backend_idle=backend_idle,
                    drained_superseded_release_tokens=drained_superseded_release_tokens,
                )
                if released is not None:
                    return self._polling_effects_after_runtime_consequence_update()
                if (
                    self._dispatch_state.active
                    or self._dispatch_state.pending_release_reason is not None
                    or bool(self._deferred_release_states)
                ):
                    return []
                return self._polling_effects_after_runtime_consequence_update()
            if not polled_records and self._dispatch_state.active:
                return []
            accumulated_decision_effects: list[RuntimeUiEffect] = []
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
                runtime_consequences_pending = not self._runtime_consequences_complete()
                decision = (
                    consumer.consume_runtime_completion(event)
                    if consumer is not None
                    else RuntimeCompletionDecision.accepted_current()
                )
                if decision.terminal:
                    if decision.failed:
                        _, close_failure = self._close_failed_runtime()
                        effects = [
                            self._ready_reset_effect(stop_completion_polling=True)
                        ]
                        effects.extend(self._completion_terminal_replay_effects(decision))
                        if decision.message:
                            effects.append(
                                RuntimeUiEffect(
                                    surface_failure=str(decision.message),
                                    surface_failure_detail_text=str(
                                        getattr(decision, "failure_detail_text", "") or ""
                                    ),
                                )
                            )
                        if close_failure:
                            effects.append(RuntimeUiEffect(surface_failure=close_failure))
                        return effects
                    return [
                        RuntimeUiEffect(
                            stop_completion_polling=True,
                            cancel_warmup_retry_kinds=_RUNTIME_WARMUP_RETRY_KINDS,
                            surface_failure=str(decision.message or ""),
                        )
                    ]
                if decision.accepted:
                    self._record_accepted_completion(event)
                    decision_effects = self._completion_decision_effects(decision)
                    if (
                        runtime_consequences_pending
                        and self._runtime_consequences_complete()
                    ):
                        accumulated_decision_effects = list(decision_effects)
                        break
                    if decision_effects:
                        accumulated_decision_effects.extend(decision_effects)
                    if decision.stop_current_poll_batch:
                        break
                    continue
                if decision.consumed:
                    self._record_consumed_completion(event)
                    decision_effects = self._completion_decision_effects(decision)
                    if (
                        runtime_consequences_pending
                        and self._runtime_consequences_complete()
                    ):
                        accumulated_decision_effects = list(decision_effects)
                        break
                    if decision_effects:
                        accumulated_decision_effects.extend(decision_effects)
                    if decision.stop_current_poll_batch:
                        break
                    continue
                if decision.stop_current_poll_batch and not decision.consumed:
                    return self.parallel_outcome_reset()
                if decision.stop_current_poll_batch:
                    break
            released = self._release_pending_dispatch_plan_after_backend_idle(
                backend_idle=backend_idle,
                drained_superseded_release_tokens=drained_superseded_release_tokens,
            )
            if accumulated_decision_effects:
                return accumulated_decision_effects
            return self._polling_effects_after_runtime_consequence_update()
        except Exception as exc:
            _, close_failure = self._close_failed_runtime(backend_failure=True)
            message = str(exc)
            if close_failure:
                message = f"{message}; {close_failure}"
            return [
                RuntimeUiEffect(
                    stop_completion_polling=True,
                    cancel_warmup_retry_kinds=_RUNTIME_WARMUP_RETRY_KINDS,
                    surface_failure=message,
                )
            ]

    def _completion_decision_effects(
        self,
        decision: RuntimeCompletionDecision,
    ) -> list[RuntimeUiEffect]:
        effects: list[RuntimeUiEffect] = []
        failure_context = getattr(decision, "failure_context", None)
        preview_failure_context = failure_context if isinstance(failure_context, Mapping) else None
        if bool(getattr(decision, "superseded_fast_failure", False)):
            if bool(
                getattr(
                    decision,
                    "superseded_fast_failure_deactivate_context_immediately",
                    False,
                )
            ):
                effects.extend(self.cancel_requested(kind="soft_shutdown"))
            effects.append(
                RuntimeUiEffect(
                    superseded_fast_failure=True,
                    superseded_fast_failure_reset_status_progress=bool(
                        getattr(
                            decision,
                            "superseded_fast_failure_reset_status_progress",
                            False,
                        )
                    ),
                    superseded_fast_failure_deactivate_context_immediately=bool(
                        getattr(
                            decision,
                            "superseded_fast_failure_deactivate_context_immediately",
                            False,
                        )
                    ),
                    preview_failure_context=preview_failure_context,
                )
            )
        preview_failure_status_text = str(
            getattr(decision, "preview_failure_status_text", "") or ""
        )
        if preview_failure_status_text:
            effects.extend(self.terminal_failure())
            effects.append(
                RuntimeUiEffect(
                    preview_failure_status_text=preview_failure_status_text,
                    preview_failure_context=preview_failure_context,
                )
            )
        progress = getattr(decision, "scoped_failure_progress", None)
        if progress is not None:
            completed = max(0, int(getattr(progress, "completed", 0) or 0))
            total = max(1, int(getattr(progress, "total", 1) or 1))
            label = str(getattr(progress, "set_label", "") or "set")
            effects.append(
                RuntimeUiEffect(
                    progress_value=max(0, min(100, int((completed / float(total)) * 100.0))),
                    status_text=f"Failed {label} ({completed}/{total})",
                )
            )
        if bool(getattr(decision, "final_scoped_failure", False)):
            self.display_completed(kind="scoped_failure")
            effects.append(self._scoped_failure_completion_effect())
            summary = getattr(decision, "scoped_failure_summary", None)
            if isinstance(summary, RuntimeScopedFailureSummary):
                effects.append(RuntimeUiEffect(scoped_failure_summary=summary))
            effects.extend(self._completion_terminal_replay_effects(decision))
        return effects

    def _completion_terminal_replay_effects(
        self,
        decision: RuntimeCompletionDecision,
    ) -> list[RuntimeUiEffect]:
        if not bool(getattr(decision, "terminal_failure_preview_replay_needed", False)):
            return []
        state_provider = self._preview_replay_state
        dirty_provider = self._dirty_generation_by_set_id
        if not callable(state_provider) or not callable(dirty_provider):
            return []
        pending_state = state_provider()
        pending = RuntimePreviewReplayState.from_pending(pending_state)
        dirty_generations = dirty_provider(pending.target_set_ids)
        return self.terminal_failure_replay_requested(
            fast_mode=bool(getattr(decision, "terminal_failure_preview_replay_fast_mode", False)),
            pending_state=pending,
            replay_snapshot=None,
            dirty_generation_by_set_id=dirty_generations,
        )

    def _scoped_failure_completion_effect(self) -> RuntimeUiEffect:
        return RuntimeUiEffect(
            stop_completion_polling=True,
            cancel_warmup_retry_kinds=_RUNTIME_WARMUP_RETRY_KINDS,
            simulation_running=False,
            slider_simulation_active=False,
            reset_slider_triggered=True,
            run_enabled=True,
            stop_enabled=False,
            progress_value=100,
            status_text="Ready",
        )

    def pool_settings_changed(self) -> list[RuntimeUiEffect]:
        return self.cancel_requested(kind="soft_shutdown")

    def project_applied(self) -> list[RuntimeUiEffect]:
        return self.cancel_requested(kind="shutdown")

    def polling_failed(self, message: str) -> list[RuntimeUiEffect]:
        return [
            RuntimeUiEffect(
                stop_completion_polling=True,
                cancel_warmup_retry_kinds=_RUNTIME_WARMUP_RETRY_KINDS,
                simulation_running=False,
                slider_simulation_active=False,
                run_enabled=True,
                stop_enabled=False,
                surface_failure=str(message or ""),
                render_state=self._failure_render_state(message, retryable=True),
            )
        ] + self.terminal_failure(backend_failed=True)

    def terminal_failure(self, *, backend_failed: bool = False) -> list[RuntimeUiEffect]:
        _, close_failure = self._close_failed_runtime(
            backend_failure=bool(backend_failed),
        )
        effects = [
            RuntimeUiEffect(
                stop_completion_polling=True,
                cancel_warmup_retry_kinds=_RUNTIME_WARMUP_RETRY_KINDS,
            )
        ]
        if close_failure:
            effects.append(RuntimeUiEffect(surface_failure=close_failure))
        return effects

    def parallel_outcome_reset(self) -> list[RuntimeUiEffect]:
        self._clear_retry_prepared_states()
        self._release_current_dispatch(RuntimeReleaseReason.FAILURE)
        self._release_invalidated_dispatches(reason=RuntimeReleaseReason.FAILURE)
        close_failure = self._close_backend_run(
            force_terminate=True,
            reason=RuntimeReleaseReason.FAILURE,
        )
        effects = [self._ready_reset_effect(stop_completion_polling=True)]
        if close_failure:
            effects.append(RuntimeUiEffect(surface_failure=close_failure))
        return effects

    def close_requested(self, *, force_terminate: bool = True) -> list[RuntimeUiEffect]:
        self._clear_retry_prepared_states()
        self._release_current_dispatch(RuntimeReleaseReason.SHUTDOWN)
        self._release_invalidated_dispatches(reason=RuntimeReleaseReason.SHUTDOWN)
        close_failure = self._close_backend_run(
            force_terminate=bool(force_terminate),
            reason=RuntimeReleaseReason.SHUTDOWN,
        )
        effects = [self._ready_reset_effect(stop_completion_polling=True)]
        if close_failure:
            effects.append(RuntimeUiEffect(surface_failure=close_failure))
        return effects

    def preview_replay_exists(
        self,
        *,
        active: bool,
        target_set_ids: Sequence[str],
    ) -> bool:
        return bool(active or normalize_preview_target_set_ids(target_set_ids))

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
        targets = normalize_preview_target_set_ids(target_set_ids)
        if not self.preview_replay_exists(active=bool(active), target_set_ids=targets):
            return []
        if bool(handoff_queued):
            return []
        replay_request_id = request_id
        if replay_request_id is None:
            replay_request_id = self._allocate_preview_replay_request_id()
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
        normalized_targets = normalize_preview_target_set_ids(target_set_ids)
        next_request_id: int | None
        if request_id is not None:
            next_request_id = int(request_id)
        elif bool(preserve_existing_request) and current.request_id is not None:
            next_request_id = current.request_id
        else:
            next_request_id = self._allocate_preview_replay_request_id()
        if next_request_id is None:
            return []
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
        reset_ids = set(normalize_preview_target_set_ids(reset_set_ids))
        surviving = tuple(set_id for set_id in pending.target_set_ids if set_id not in reset_ids)
        if surviving:
            return self.preview_replay_state_requested(
                current_state=pending,
                target_set_ids=surviving,
                preserve_existing_request=True,
                clear_plot_updates=True,
            )
        return self.preview_replay_cleared(clear_plot_updates=True)

    def preview_replay_launch_unavailable(self, *, current_state: object) -> list[RuntimeUiEffect]:
        pending = RuntimePreviewReplayState.from_pending(current_state)
        if not self.preview_replay_exists_for_state(pending):
            return []
        return self.preview_replay_cleared(clear_plot_updates=False)

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

    def preview_invalidation_settled(
        self,
        *,
        has_active_explicit_simulation: bool,
    ) -> list[RuntimeUiEffect]:
        if bool(has_active_explicit_simulation):
            return []
        return [
            RuntimeUiEffect(
                simulation_running=False,
                slider_simulation_active=False,
                run_enabled=True,
                stop_enabled=False,
                progress_value=0,
                status_text="Ready",
            )
        ]

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
        target_set_ids = normalize_preview_target_set_ids(pending.target_set_ids)
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

    def render_failure(self, message: str, *, retryable: bool = True) -> None:
        self._clear_retry_prepared_states(clear_manual_retry=False)
        if not bool(retryable):
            self._manual_retry_prepared = None
            self._manual_retry_kind = ""
        self._render_state(
            SimulationRuntimeReadinessRenderState(
                status="failed",
                status_text=str(message),
                launch_available=False,
                failed=True,
                retryable=bool(retryable),
            )
        )

    def _close_backend_run(
        self,
        *,
        force_terminate: bool,
        reason: RuntimeReleaseReason,
    ) -> str:
        backend = self._backend
        if backend is None:
            return ""
        try:
            result = backend.close_current_run(force_terminate=bool(force_terminate))
        except Exception as exc:
            return f"Runtime backend close failed: {exc}"
        self._clear_backend_pool_after_close(result, reason=reason)
        return ""

    def _clear_backend_pool_after_close(
        self,
        result: object,
        *,
        reason: RuntimeReleaseReason,
    ) -> None:
        pool_token = str(getattr(result, "pool_token", "") or "")
        if not pool_token:
            return
        self._allocator.clear_backend_pool(
            pool_token=pool_token,
            generation=int(getattr(result, "generation", 0) or 0),
            reason=reason,
        )

    def _prepared_is_current(self, prepared: PreparedRuntimeRequestSet) -> bool:
        provider = self._prepared_request_is_current
        if provider is None:
            return True
        try:
            return bool(provider(prepared))
        except Exception:
            return False

    def _prepared_runtime_input_epochs_current(self, prepared: PreparedRuntimeRequestSet) -> bool:
        provider = self._current_runtime_input_epochs
        if provider is None:
            return True
        expected = {
            str(key): int(value)
            for key, value in dict(getattr(prepared.intent, "runtime_input_epochs", {}) or {}).items()
            if str(key)
        }
        if not expected:
            return True
        try:
            current_raw = provider(tuple(prepared.intent.set_ids or ()))
        except Exception:
            return False
        current = {
            str(key): int(value)
            for key, value in dict(current_raw or {}).items()
            if str(key)
        }
        return current == expected

    def _allocation_request(self, prepared: PreparedRuntimeRequestSet) -> RuntimeLaneAllocationRequest:
        return RuntimeLaneAllocationRequest(
            compatibility_key=prepared.compatibility_key,
            required_lane_capacity=prepared.required_lane_capacity,
            preferred_lane_capacity=prepared.preferred_lane_capacity,
            task_count=len(prepared.task_descriptors),
            request_token=prepared.intent.request_token,
            scope=prepared.intent.intent_kind,
            require_backend_lease=True,
        )

    def _probe_readiness_consequence(
        self,
        prepared: PreparedRuntimeRequestSet,
        *,
        warmup_retry_kind: str,
    ) -> RuntimeReadinessConsequence:
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
            return self._readiness_consequence(state)
        if not self._prepared_is_current(prepared):
            state = SimulationRuntimeReadinessRenderState(
                status="stale",
                status_text="Runtime launch readiness changed before backend warmup completed.",
                failed=True,
                launch_available=False,
            )
            return self._readiness_consequence(state)
        if not self._prepared_runtime_input_epochs_current(prepared):
            state = SimulationRuntimeReadinessRenderState(
                status="stale",
                status_text="Runtime input changed before launch readiness completed.",
                failed=True,
                launch_available=False,
            )
            return self._readiness_consequence(state)
        try:
            self._allocator.ensure_ready_lanes(
                compatibility_key=prepared.compatibility_key,
                capacity=prepared.preferred_lane_capacity,
                task_count=len(prepared.task_descriptors),
                nonblocking=True,
            )
            probe = self._allocator.probe_readiness(
                self._allocation_request(prepared),
            )
        except Exception as exc:
            self._set_manual_retry_prepared(
                prepared,
                retry_kind=warmup_retry_kind,
            )
            self._clear_warmup_retry_prepared(warmup_retry_kind)
            state = self._failure_render_state(str(exc), retryable=True)
            return self._readiness_consequence(state)
        if not probe.ready:
            self._store_warmup_retry_prepared(warmup_retry_kind, prepared)
            state = SimulationRuntimeReadinessRenderState(
                status="warming",
                status_text=probe.message or "Preparing runtime lanes...",
                launch_available=False,
                preview_available=prepared.intent.intent_kind == "preview",
            )
            return self._readiness_consequence(
                state,
                extra_effects=(self._warmup_retry_effect(warmup_retry_kind),),
            )
        self._clear_warmup_retry_prepared(warmup_retry_kind)
        state = SimulationRuntimeReadinessRenderState(
            status="ready",
            status_text="Ready",
            launch_available=prepared.intent.intent_kind != "preview",
            preview_available=prepared.intent.intent_kind == "preview",
        )
        return self._readiness_consequence(state)

    def _readiness_state_is_ready(self, consequence: RuntimeReadinessConsequence) -> bool:
        return bool(
            consequence.render_state is not None
            and str(consequence.render_state.status or "") == "ready"
        )

    def _dispatch_acceptance_consequence(
        self,
        state: SimulationRuntimeReadinessRenderState,
        *,
        dispatch_plan: RuntimeDispatchPlan | None = None,
        extra_effects: Sequence[RuntimeUiEffect] = (),
    ) -> RuntimeDispatchAcceptance:
        readiness = self._readiness_consequence(state, extra_effects=extra_effects)
        return RuntimeDispatchAcceptance(
            dispatch_plan=dispatch_plan,
            effects=readiness.effects,
        )

    def _allocate_preview_replay_request_id(self) -> int | None:
        allocator = (
            self._reserve_request_id
            if self._active_preview_dispatch_in_flight()
            else self._next_request_id
        )
        if not callable(allocator):
            return None
        try:
            return int(allocator())
        except (TypeError, ValueError, OverflowError):
            return None

    def _active_preview_dispatch_in_flight(self) -> bool:
        plan = self._dispatch_state.plan
        return bool(
            self._dispatch_state.active
            and plan is not None
            and str(plan.launch_allocation.launch_intent.intent_kind or "") == "preview"
        )

    def _readiness_consequence(
        self,
        state: SimulationRuntimeReadinessRenderState,
        *,
        extra_effects: tuple[RuntimeUiEffect, ...] = (),
    ) -> RuntimeReadinessConsequence:
        self._render_state(state)
        return RuntimeReadinessConsequence(
            launch_available=bool(state.launch_available),
            render_state=state,
            effects=tuple(extra_effects or ()),
        )

    def _start_dispatch_state(self, dispatch_plan: RuntimeDispatchPlan) -> None:
        self._last_dispatch_plan = dispatch_plan
        self._dispatch_state = RuntimeDispatchState.started(dispatch_plan)

    def _record_accepted_completion(self, event: RuntimeCompletionEvent) -> None:
        self._dispatch_state = self._dispatch_state.accepted_completion(event.set_id)

    def _record_consumed_completion(self, event: RuntimeCompletionEvent) -> None:
        self._dispatch_state = self._dispatch_state.consumed_completion(event.set_id)

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

    def _close_failed_runtime(
        self,
        *,
        backend_failure: bool = False,
    ) -> tuple[RuntimeReleaseResult, str]:
        self._clear_retry_prepared_states()
        release_result = self._release_current_dispatch(
            RuntimeReleaseReason.FAILURE,
            backend_failure=bool(backend_failure),
        )
        self._release_invalidated_dispatches(
            reason=RuntimeReleaseReason.FAILURE,
            backend_failure=bool(backend_failure),
        )
        close_failure = self._close_backend_run(
            force_terminate=True,
            reason=RuntimeReleaseReason.FAILURE,
        )
        return release_result, close_failure

    def _ready_reset_effect(self, *, stop_completion_polling: bool) -> RuntimeUiEffect:
        return RuntimeUiEffect(
            stop_completion_polling=bool(stop_completion_polling),
            cancel_warmup_retry_kinds=_RUNTIME_WARMUP_RETRY_KINDS,
            simulation_running=False,
            slider_simulation_active=False,
            reset_slider_triggered=True,
            run_enabled=True,
            stop_enabled=False,
            progress_value=0,
            status_text="Ready",
        )

    def _stage_invalidated_dispatch_release(
        self,
        reason: RuntimeReleaseReason,
        *,
        superseded_drain_token: str = "",
    ) -> None:
        dispatch_plan = self._dispatch_state.plan
        if dispatch_plan is None:
            return
        normalized_token = str(superseded_drain_token or "")
        self._deferred_release_states = self._deferred_release_states + (
            RuntimeDeferredReleaseState(
                dispatch_plan=dispatch_plan,
                reason=reason,
                wait_for_backend_idle=not bool(normalized_token),
                superseded_drain_token=normalized_token,
            ),
        )
        self._dispatch_state = RuntimeDispatchState()

    def _release_invalidated_dispatches(
        self,
        *,
        reason: RuntimeReleaseReason | None = None,
        backend_idle: bool | None = None,
        drained_superseded_release_tokens: Sequence[str] = (),
        backend_failure: bool = False,
    ) -> tuple[RuntimeReleaseResult, ...]:
        if not self._deferred_release_states:
            return ()
        drained_tokens = {
            str(token)
            for token in drained_superseded_release_tokens or ()
            if str(token)
        }
        pending = self._deferred_release_states
        retained: list[RuntimeDeferredReleaseState] = []
        released: list[RuntimeReleaseResult] = []
        force_release = reason is not None or bool(backend_failure)
        for state in pending:
            state_reason = reason if reason is not None else state.reason
            if state.dispatch_plan is None or state_reason is None:
                continue
            drain_token = str(state.superseded_drain_token or "")
            if not force_release and drain_token and drain_token not in drained_tokens:
                retained.append(state)
                continue
            if (
                not force_release
                and not drain_token
                and backend_idle is not None
                and state.wait_for_backend_idle
                and not bool(backend_idle)
            ):
                retained.append(state)
                continue
            released.append(
                self._release_dispatch_plan(
                    state.dispatch_plan,
                    state_reason,
                    backend_failure=bool(backend_failure),
                )
            )
        self._deferred_release_states = tuple(retained)
        return tuple(released)

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
        drained_superseded_release_tokens: Sequence[str] = (),
    ) -> RuntimeReleaseResult | None:
        deferred_results = self._release_invalidated_dispatches(
            backend_idle=backend_idle,
            drained_superseded_release_tokens=drained_superseded_release_tokens,
        )
        if deferred_results:
            return deferred_results[-1]
        plan = self._dispatch_state.plan
        reason = self._dispatch_state.pending_release_reason
        if plan is None or reason is None:
            return None
        if self._dispatch_state.release_requires_backend_idle and not bool(backend_idle):
            return None
        return self._release_dispatch_plan(plan, reason)

    def _pending_retry_prepared(self) -> PreparedRuntimeRequestSet | None:
        return (
            self._acceptance_retry_prepared
            or self._readiness_retry_prepared
            or self._manual_retry_prepared
        )

    def _set_manual_retry_prepared(
        self,
        prepared: PreparedRuntimeRequestSet | None,
        *,
        retry_kind: str,
    ) -> None:
        self._manual_retry_prepared = prepared
        self._manual_retry_kind = (
            str(retry_kind or "")
            if prepared is not None and str(retry_kind or "") in _RUNTIME_WARMUP_RETRY_KINDS
            else ""
        )

    def _store_warmup_retry_prepared(
        self,
        warmup_retry_kind: str,
        prepared: PreparedRuntimeRequestSet,
    ) -> None:
        if warmup_retry_kind == _RUNTIME_WARMUP_RETRY_REFRESH_READINESS:
            self._readiness_retry_prepared = prepared
            return
        if warmup_retry_kind == _RUNTIME_WARMUP_RETRY_ACCEPT_DISPATCH:
            self._acceptance_retry_prepared = prepared

    def _clear_warmup_retry_prepared(self, warmup_retry_kind: str) -> None:
        if warmup_retry_kind == _RUNTIME_WARMUP_RETRY_REFRESH_READINESS:
            self._readiness_retry_prepared = None
            return
        if warmup_retry_kind == _RUNTIME_WARMUP_RETRY_ACCEPT_DISPATCH:
            self._acceptance_retry_prepared = None

    def _clear_retry_prepared_states(self, *, clear_manual_retry: bool = True) -> None:
        self._readiness_retry_prepared = None
        self._acceptance_retry_prepared = None
        if clear_manual_retry:
            self._manual_retry_prepared = None
            self._manual_retry_kind = ""

    def _warmup_retry_effect(
        self,
        warmup_retry_kind: str,
        *,
        cancel_retry_kinds: Sequence[str] = (),
    ) -> RuntimeUiEffect:
        return RuntimeUiEffect(
            cancel_warmup_retry_kinds=tuple(
                str(kind or "")
                for kind in tuple(cancel_retry_kinds or ())
                if str(kind or "") in _RUNTIME_WARMUP_RETRY_KINDS
            ),
            warmup_retry_kind=str(warmup_retry_kind or ""),
            warmup_retry_delay_ms=int(_RUNTIME_WARMUP_RETRY_DELAY_MS),
        )

    def _cancel_warmup_retry_effect(self, *retry_kinds: str) -> RuntimeUiEffect:
        normalized = tuple(
            str(kind or "")
            for kind in tuple(retry_kinds or _RUNTIME_WARMUP_RETRY_KINDS)
            if str(kind or "") in _RUNTIME_WARMUP_RETRY_KINDS
        )
        return RuntimeUiEffect(cancel_warmup_retry_kinds=normalized)

    def _runtime_consequences_complete(self) -> bool:
        if self._deferred_release_states:
            return False
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
            launch_available=False,
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
            normalize_preview_target_set_ids(pending.target_set_ids)
            == tuple(snapshot.target_set_ids)
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
