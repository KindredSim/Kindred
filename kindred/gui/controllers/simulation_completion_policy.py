from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Mapping, Optional


def _normalize_optional_int(value: object) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _normalize_optional_str(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _normalize_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"false", "0", "no", "off", ""}:
            return False
        if text in {"true", "1", "yes", "on"}:
            return True
    return bool(value)


def _normalize_str_sequence(values: object, *, dedupe: bool) -> tuple[str, ...]:
    if not values:
        return ()
    if isinstance(values, str):
        values = (values,)
    elif isinstance(values, Mapping):
        values = values.keys()
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        if dedupe and text in seen:
            continue
        seen.add(text)
        normalized.append(text)
    return tuple(normalized)


def _normalize_set_ids(values: object) -> tuple[str, ...]:
    return _normalize_str_sequence(values, dedupe=True)


def _normalize_name_sequence(values: object) -> tuple[str, ...]:
    return _normalize_str_sequence(values, dedupe=False)


def _normalize_optional_set_ids(values: object) -> Optional[tuple[str, ...]]:
    if values is None:
        return None
    return _normalize_set_ids(values)


def _try_float(value: object) -> Optional[float]:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return None
    if out != out or out in (float("inf"), float("-inf")):
        return None
    return float(out)


def _normalize_nonnegative_int(value: object, *, default: int = 0) -> int:
    normalized = _normalize_optional_int(value)
    if normalized is None:
        return int(default)
    return max(0, int(normalized))


def pending_initial_seed_for_set(
    pending_init_seed: object,
    *,
    set_name: str,
) -> dict[str, object]:
    if not isinstance(pending_init_seed, Mapping) or not pending_init_seed:
        return {}

    nested_seed = pending_init_seed.get(str(set_name))
    if isinstance(nested_seed, Mapping):
        return {str(species): value for species, value in nested_seed.items()}
    return {}


@dataclass(frozen=True, slots=True)
class CompletionPolicyContext:
    active: bool
    request_id: Optional[int]
    run_id: Optional[int]
    fast_mode: bool
    parallel: bool
    keep_executor_alive: bool
    queue_ids: tuple[str, ...] = ()
    queue_names: tuple[str, ...] = ()
    total: int = 0
    pos: int = 0
    primary_set_id: Optional[str] = None
    completed_set_ids: tuple[str, ...] = ()
    pending_workspace_reset_set_ids: tuple[str, ...] = ()
    pending_dirty_reset_generation_by_set_id: dict[str, int] = field(default_factory=dict)
    pending_init_seed: dict[str, dict[str, float]] = field(default_factory=dict)
    pending_init_rewrite: Optional[str] = None
    pending_init_applied: bool = False
    explicit_cache_preview_token: Optional[str] = None
    explicit_cache_preview_scope_set_ids: Optional[tuple[str, ...]] = None
    explicit_cache_valid_set_ids: Optional[tuple[str, ...]] = None
    explicit_cache_invalidated_set_ids: Optional[tuple[str, ...]] = None
    preview_scope_set_ids: Optional[tuple[str, ...]] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_id", _normalize_optional_int(self.request_id))
        object.__setattr__(self, "run_id", _normalize_optional_int(self.run_id))
        object.__setattr__(self, "active", _normalize_bool(self.active))
        object.__setattr__(self, "fast_mode", _normalize_bool(self.fast_mode))
        object.__setattr__(self, "parallel", _normalize_bool(self.parallel))
        object.__setattr__(self, "keep_executor_alive", _normalize_bool(self.keep_executor_alive))
        object.__setattr__(self, "queue_ids", _normalize_set_ids(self.queue_ids))
        object.__setattr__(self, "queue_names", _normalize_name_sequence(self.queue_names))
        object.__setattr__(self, "total", _normalize_nonnegative_int(self.total, default=0))
        object.__setattr__(self, "pos", _normalize_nonnegative_int(self.pos, default=0))
        object.__setattr__(self, "primary_set_id", _normalize_optional_str(self.primary_set_id))
        object.__setattr__(self, "completed_set_ids", _normalize_set_ids(self.completed_set_ids))
        object.__setattr__(
            self,
            "pending_workspace_reset_set_ids",
            _normalize_set_ids(self.pending_workspace_reset_set_ids),
        )
        object.__setattr__(
            self,
            "pending_dirty_reset_generation_by_set_id",
            {
                str(set_id): int(normalized_generation)
                for set_id, generation in dict(self.pending_dirty_reset_generation_by_set_id or {}).items()
                if str(set_id) and (normalized_generation := _normalize_optional_int(generation)) is not None
            },
        )
        object.__setattr__(
            self,
            "pending_init_seed",
            {
                str(set_name): {
                    str(species): float(value)
                    for species, value in dict(seed).items()
                    if str(species) and _try_float(value) is not None
                }
                for set_name, seed in dict(self.pending_init_seed or {}).items()
                if str(set_name) and isinstance(seed, Mapping)
            },
        )
        object.__setattr__(self, "pending_init_rewrite", _normalize_optional_str(self.pending_init_rewrite))
        object.__setattr__(self, "pending_init_applied", _normalize_bool(self.pending_init_applied))
        object.__setattr__(self, "explicit_cache_preview_token", _normalize_optional_str(self.explicit_cache_preview_token))
        object.__setattr__(
            self,
            "explicit_cache_preview_scope_set_ids",
            _normalize_optional_set_ids(self.explicit_cache_preview_scope_set_ids),
        )
        object.__setattr__(
            self,
            "explicit_cache_valid_set_ids",
            _normalize_optional_set_ids(self.explicit_cache_valid_set_ids),
        )
        object.__setattr__(
            self,
            "explicit_cache_invalidated_set_ids",
            _normalize_optional_set_ids(self.explicit_cache_invalidated_set_ids),
        )
        object.__setattr__(self, "preview_scope_set_ids", _normalize_optional_set_ids(self.preview_scope_set_ids))

    def evolve(self, **changes: object) -> CompletionPolicyContext:
        return replace(self, **changes)


@dataclass(frozen=True, slots=True)
class RunActivitySnapshot:
    latest_request_id: int
    simulation_running: bool
    slider_simulation_active: bool
    worker_running: bool
    worker_fast_mode: Optional[bool]
    worker_request_id: Optional[int]
    discarded_slider_preview_generation_id: Optional[int] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "latest_request_id", _normalize_nonnegative_int(self.latest_request_id, default=0))
        object.__setattr__(self, "simulation_running", _normalize_bool(self.simulation_running))
        object.__setattr__(self, "slider_simulation_active", _normalize_bool(self.slider_simulation_active))
        object.__setattr__(self, "worker_running", _normalize_bool(self.worker_running))
        worker_fast_mode = None if self.worker_fast_mode is None else _normalize_bool(self.worker_fast_mode)
        object.__setattr__(self, "worker_fast_mode", worker_fast_mode)
        object.__setattr__(self, "worker_request_id", _normalize_optional_int(self.worker_request_id))
        object.__setattr__(
            self,
            "discarded_slider_preview_generation_id",
            _normalize_optional_int(self.discarded_slider_preview_generation_id),
        )


@dataclass(frozen=True, slots=True)
class PendingReplayState:
    active: bool
    request_id: Optional[int]
    target_set_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "active", _normalize_bool(self.active))
        object.__setattr__(self, "request_id", _normalize_optional_int(self.request_id))
        object.__setattr__(self, "target_set_ids", _normalize_set_ids(self.target_set_ids))


@dataclass(frozen=True, slots=True)
class DirtySetState:
    is_dirty: bool
    generation: Optional[int]

    def __post_init__(self) -> None:
        object.__setattr__(self, "is_dirty", _normalize_bool(self.is_dirty))
        object.__setattr__(self, "generation", _normalize_optional_int(self.generation))


@dataclass(frozen=True, slots=True)
class CacheAuthorityState:
    active_cache_key: Optional[str]
    active_cache_preview_token: Optional[str]
    active_cache_preview_scope_set_ids: Optional[tuple[str, ...]]
    active_cache_valid_set_ids: Optional[tuple[str, ...]]
    active_cache_invalidated_set_ids: Optional[tuple[str, ...]]

    def __post_init__(self) -> None:
        object.__setattr__(self, "active_cache_key", _normalize_optional_str(self.active_cache_key))
        object.__setattr__(self, "active_cache_preview_token", _normalize_optional_str(self.active_cache_preview_token))
        object.__setattr__(
            self,
            "active_cache_preview_scope_set_ids",
            _normalize_optional_set_ids(self.active_cache_preview_scope_set_ids),
        )
        object.__setattr__(self, "active_cache_valid_set_ids", _normalize_optional_set_ids(self.active_cache_valid_set_ids))
        object.__setattr__(
            self,
            "active_cache_invalidated_set_ids",
            _normalize_optional_set_ids(self.active_cache_invalidated_set_ids),
        )


@dataclass(frozen=True, slots=True)
class PendingReplayDirective:
    action: str
    target_set_ids: tuple[str, ...] = ()
    clear_plot_updates: bool = False
    needs_fresh_request_id: bool = False
    preserve_existing_request: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "action", str(self.action))
        object.__setattr__(self, "target_set_ids", _normalize_set_ids(self.target_set_ids))
        object.__setattr__(self, "clear_plot_updates", bool(self.clear_plot_updates))
        object.__setattr__(self, "needs_fresh_request_id", bool(self.needs_fresh_request_id))
        object.__setattr__(self, "preserve_existing_request", bool(self.preserve_existing_request))

    @classmethod
    def clear(cls, *, clear_plot_updates: bool) -> PendingReplayDirective:
        return cls(action="clear", clear_plot_updates=clear_plot_updates)

    @classmethod
    def preserve(
        cls,
        *,
        target_set_ids: tuple[str, ...],
        clear_plot_updates: bool,
    ) -> PendingReplayDirective:
        return cls(action="preserve", target_set_ids=target_set_ids, clear_plot_updates=clear_plot_updates)

    @classmethod
    def queue_fresh(
        cls,
        *,
        target_set_ids: tuple[str, ...],
    ) -> PendingReplayDirective:
        return cls(action="queue_fresh", target_set_ids=target_set_ids, needs_fresh_request_id=True)

    @classmethod
    def arm_existing(
        cls,
        *,
        target_set_ids: tuple[str, ...],
    ) -> PendingReplayDirective:
        return cls(
            action="arm_existing",
            target_set_ids=target_set_ids,
            preserve_existing_request=True,
        )


@dataclass(frozen=True, slots=True)
class PolicyStatePatch:
    context: Optional[CompletionPolicyContext] = None
    pending_replay: Optional[PendingReplayDirective] = None
    clear_discarded_slider_preview_generation: bool = False


@dataclass(frozen=True, slots=True)
class SupersededFastDecision:
    display_current_preview: bool
    schedule_pending_preview_run: bool
    reset_status_progress: bool
    defer_context_deactivation_until_after_display: bool
    deactivate_context_immediately: bool
    state_patch: PolicyStatePatch


@dataclass(frozen=True, slots=True)
class CacheReconciliationDecision:
    clear_active_selection_state: bool
    active_cache_key: Optional[str]
    active_cache_preview_token: Optional[str]
    active_cache_preview_scope_set_ids: Optional[tuple[str, ...]]
    active_cache_valid_set_ids: Optional[tuple[str, ...]]
    active_cache_invalidated_set_ids: Optional[tuple[str, ...]]
    redraw_valid_set_ids: Optional[tuple[str, ...]]
    has_redraw_subset: bool


@dataclass(frozen=True, slots=True)
class PendingInitCompletionDecision:
    should_attempt_apply: bool
    seed_for_ui: dict[str, float] = field(default_factory=dict)
    rewrite: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "should_attempt_apply", bool(self.should_attempt_apply))
        object.__setattr__(
            self,
            "seed_for_ui",
            {
                str(species): float(value)
                for species, value in dict(self.seed_for_ui or {}).items()
                if str(species) and _try_float(value) is not None
            },
        )
        object.__setattr__(self, "rewrite", _normalize_optional_str(self.rewrite))


@dataclass(frozen=True, slots=True)
class PendingInitFailureDecision:
    should_invalidate_preserved_results: bool
    state_patch: PolicyStatePatch


@dataclass(frozen=True, slots=True)
class ExplicitDirtyResetDecision:
    eligible_reset_set_ids: tuple[str, ...]
    state_patch: PolicyStatePatch


@dataclass(frozen=True, slots=True)
class RunStartCacheDecision:
    explicit_cache_valid_set_ids: Optional[tuple[str, ...]]
    explicit_cache_invalidated_set_ids: Optional[tuple[str, ...]]
    explicit_preview_scope_set_ids: Optional[tuple[str, ...]]
    preview_scope_set_ids: Optional[tuple[str, ...]]


@dataclass(frozen=True, slots=True)
class DirtyResetTrackingDecision:
    pending_workspace_reset_set_ids: tuple[str, ...]
    pending_dirty_reset_generation_by_set_id: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "pending_workspace_reset_set_ids",
            _normalize_set_ids(self.pending_workspace_reset_set_ids),
        )
        object.__setattr__(
            self,
            "pending_dirty_reset_generation_by_set_id",
            {
                str(set_id): int(generation)
                for set_id, generation in dict(self.pending_dirty_reset_generation_by_set_id or {}).items()
                if str(set_id)
            },
        )


class SimulationCompletionPolicy:
    def has_active_explicit_simulation(
        self,
        *,
        activity: RunActivitySnapshot,
        context: Optional[CompletionPolicyContext],
    ) -> bool:
        if context is not None and context.active:
            return not context.fast_mode
        if activity.worker_running:
            return not bool(activity.worker_fast_mode)
        return bool(activity.simulation_running) and (not bool(activity.slider_simulation_active))

    def has_active_fast_preview_in_flight(
        self,
        *,
        activity: RunActivitySnapshot,
        context: Optional[CompletionPolicyContext],
    ) -> bool:
        if context is not None and context.active and context.fast_mode:
            return True
        if activity.worker_running:
            return bool(activity.worker_fast_mode)
        return bool(activity.simulation_running) and bool(activity.slider_simulation_active)

    def stale_fast_request_still_owns_current_state(
        self,
        *,
        activity: RunActivitySnapshot,
        context: Optional[CompletionPolicyContext],
        request_id: int,
    ) -> bool:
        if context is not None and context.active:
            if not context.fast_mode:
                return False
            if context.request_id is None:
                return False
            return int(context.request_id) == int(request_id)
        # A concrete fast-worker request id remains the ownership signal while
        # preview mode is still active, but only if that worker still matches
        # the latest known preview intent.
        if activity.worker_request_id is not None:
            if not bool(activity.worker_fast_mode):
                return False
            if int(activity.worker_request_id) != int(activity.latest_request_id):
                return False
            if bool(activity.worker_running) or bool(activity.slider_simulation_active):
                return int(activity.worker_request_id) == int(request_id)
            return False
        if activity.worker_running:
            if not bool(activity.worker_fast_mode):
                return False
            return False
        return bool(activity.slider_simulation_active)

    def preview_request_can_display(
        self,
        *,
        activity: RunActivitySnapshot,
        context: Optional[CompletionPolicyContext],
        request_id: Optional[int],
    ) -> bool:
        if request_id is None:
            return True
        if int(request_id) == int(activity.latest_request_id):
            return True
        if activity.discarded_slider_preview_generation_id == int(activity.latest_request_id):
            return False
        return self.stale_fast_request_still_owns_current_state(
            activity=activity,
            context=context,
            request_id=int(request_id),
        )

    def build_run_start_cache_decision(
        self,
        *,
        fast_mode: bool,
        queue_ids: tuple[str, ...],
    ) -> RunStartCacheDecision:
        normalized_queue_ids = _normalize_set_ids(queue_ids)
        explicit_valid_set_ids = normalized_queue_ids if normalized_queue_ids else None
        if fast_mode:
            return RunStartCacheDecision(
                explicit_cache_valid_set_ids=None,
                explicit_cache_invalidated_set_ids=None,
                explicit_preview_scope_set_ids=None,
                preview_scope_set_ids=explicit_valid_set_ids,
            )
        return RunStartCacheDecision(
            explicit_cache_valid_set_ids=explicit_valid_set_ids,
            explicit_cache_invalidated_set_ids=None,
            explicit_preview_scope_set_ids=None,
            preview_scope_set_ids=None,
        )

    def capture_dirty_reset_tracking(
        self,
        *,
        fast_mode: bool,
        queue_ids: tuple[str, ...],
        dirty_state_by_set_id: Mapping[str, DirtySetState],
    ) -> DirtyResetTrackingDecision:
        if fast_mode:
            return DirtyResetTrackingDecision((), {})
        tracking: dict[str, int] = {}
        for set_id in _normalize_set_ids(queue_ids):
            state = dirty_state_by_set_id.get(str(set_id))
            if state is None or (not state.is_dirty) or state.generation is None:
                continue
            tracking[str(set_id)] = int(state.generation)
        return DirtyResetTrackingDecision(tuple(tracking.keys()), tracking)

    def resolve_superseded_fast_completion(
        self,
        *,
        activity: RunActivitySnapshot,
        context: Optional[CompletionPolicyContext],
        request_id: int,
        pending_replay: PendingReplayState,
        shutdown_requested: bool,
    ) -> SupersededFastDecision:
        request_id_i = int(request_id)
        should_schedule_pending = bool(pending_replay.active) or (
            pending_replay.request_id is not None and pending_replay.request_id == int(activity.latest_request_id)
        )
        stale_fast_owns_current_state = self.stale_fast_request_still_owns_current_state(
            activity=activity,
            context=context,
            request_id=request_id_i,
        )
        suppress_discard_ui = activity.discarded_slider_preview_generation_id == int(activity.latest_request_id)
        display_current_preview = bool(stale_fast_owns_current_state) and (not bool(suppress_discard_ui))
        if display_current_preview:
            defer_handoff = bool(should_schedule_pending) and not bool(context is not None and context.active and context.parallel)
            return SupersededFastDecision(
                display_current_preview=True,
                schedule_pending_preview_run=bool(should_schedule_pending and (not shutdown_requested)),
                reset_status_progress=False,
                defer_context_deactivation_until_after_display=defer_handoff,
                deactivate_context_immediately=False,
                state_patch=PolicyStatePatch(
                    pending_replay=(
                        PendingReplayDirective.arm_existing(target_set_ids=pending_replay.target_set_ids)
                        if should_schedule_pending and (not shutdown_requested)
                        else None
                    ),
                ),
            )
        updated_context = context
        if stale_fast_owns_current_state and context is not None and context.active:
            updated_context = context.evolve(active=False)
        pending_directive = None
        reset_status_progress = bool(stale_fast_owns_current_state) and (not bool(suppress_discard_ui))
        if should_schedule_pending and (not shutdown_requested):
            pending_directive = PendingReplayDirective.queue_fresh(
                target_set_ids=pending_replay.target_set_ids,
            )
        else:
            pending_directive = PendingReplayDirective.clear(clear_plot_updates=False)
        return SupersededFastDecision(
            display_current_preview=False,
            schedule_pending_preview_run=bool(should_schedule_pending and (not shutdown_requested)),
            reset_status_progress=reset_status_progress,
            defer_context_deactivation_until_after_display=False,
            deactivate_context_immediately=bool(stale_fast_owns_current_state),
            state_patch=PolicyStatePatch(
                context=updated_context,
                pending_replay=pending_directive,
                clear_discarded_slider_preview_generation=True,
            ),
        )

    def resolve_superseded_fast_error(
        self,
        *,
        activity: RunActivitySnapshot,
        context: Optional[CompletionPolicyContext],
        request_id: int,
        pending_replay: PendingReplayState,
    ) -> SupersededFastDecision:
        request_id_i = int(request_id)
        should_schedule_pending = bool(pending_replay.active) or (
            pending_replay.request_id is not None and pending_replay.request_id == int(activity.latest_request_id)
        )
        stale_fast_owns_current_state = self.stale_fast_request_still_owns_current_state(
            activity=activity,
            context=context,
            request_id=request_id_i,
        )
        suppress_discard_ui = activity.discarded_slider_preview_generation_id == int(activity.latest_request_id)
        updated_context = context
        if stale_fast_owns_current_state and context is not None and context.active:
            updated_context = context.evolve(active=False)
        if should_schedule_pending:
            pending_directive = PendingReplayDirective.queue_fresh(target_set_ids=pending_replay.target_set_ids)
        else:
            pending_directive = PendingReplayDirective.clear(clear_plot_updates=False)
        return SupersededFastDecision(
            display_current_preview=False,
            schedule_pending_preview_run=bool(should_schedule_pending),
            reset_status_progress=bool(stale_fast_owns_current_state) and (not bool(suppress_discard_ui)),
            defer_context_deactivation_until_after_display=False,
            deactivate_context_immediately=bool(stale_fast_owns_current_state),
            state_patch=PolicyStatePatch(
                context=updated_context,
                pending_replay=pending_directive,
                clear_discarded_slider_preview_generation=True,
            ),
        )

    def build_explicit_cache_reconciliation(
        self,
        *,
        context: CompletionPolicyContext,
        cache_state: CacheAuthorityState,
        cache_key: Optional[str],
    ) -> CacheReconciliationDecision:
        cache_key_norm = _normalize_optional_str(cache_key)
        preview_token = context.explicit_cache_preview_token
        preview_scope_set_ids = context.explicit_cache_preview_scope_set_ids
        valid_set_ids = None
        invalidated_set_ids = None
        has_valid_subset = context.explicit_cache_valid_set_ids is not None
        if has_valid_subset:
            valid_set_ids = context.explicit_cache_valid_set_ids or ()
            invalidated_set_ids = context.explicit_cache_invalidated_set_ids or ()
        else:
            valid_set_ids = context.queue_ids if context.queue_ids else None

        clear_active_selection_state = bool(has_valid_subset and valid_set_ids == ())
        redraw_valid_set_ids = None
        has_redraw_subset = False
        if has_valid_subset:
            has_redraw_subset = True
            redraw_valid_set_ids = valid_set_ids or ()
        elif cache_state.active_cache_valid_set_ids is not None:
            has_redraw_subset = True
            redraw_valid_set_ids = cache_state.active_cache_valid_set_ids

        return CacheReconciliationDecision(
            clear_active_selection_state=clear_active_selection_state,
            active_cache_key=None if clear_active_selection_state else cache_key_norm,
            active_cache_preview_token=None if clear_active_selection_state else preview_token,
            active_cache_preview_scope_set_ids=None if clear_active_selection_state else preview_scope_set_ids,
            active_cache_valid_set_ids=None if clear_active_selection_state else valid_set_ids,
            active_cache_invalidated_set_ids=(
                None if clear_active_selection_state else (invalidated_set_ids if has_valid_subset else None)
            ),
            redraw_valid_set_ids=redraw_valid_set_ids,
            has_redraw_subset=has_redraw_subset,
        )

    def build_context_update_from_cache_truth(
        self,
        *,
        context: CompletionPolicyContext,
        cache_state: CacheAuthorityState,
        cache_key: Optional[str],
    ) -> CompletionPolicyContext:
        if cache_state.active_cache_key == _normalize_optional_str(cache_key):
            return context.evolve(
                explicit_cache_preview_token=cache_state.active_cache_preview_token,
                explicit_cache_preview_scope_set_ids=cache_state.active_cache_preview_scope_set_ids,
                explicit_cache_valid_set_ids=cache_state.active_cache_valid_set_ids,
                explicit_cache_invalidated_set_ids=cache_state.active_cache_invalidated_set_ids,
            )
        return context.evolve(
            explicit_cache_preview_token=None,
            explicit_cache_preview_scope_set_ids=(),
            explicit_cache_valid_set_ids=(),
            explicit_cache_invalidated_set_ids=(),
        )

    def resolve_pending_init_completion(
        self,
        *,
        context: CompletionPolicyContext,
        batch_set: Optional[str],
        is_preview: bool,
        is_primary: bool,
    ) -> PendingInitCompletionDecision:
        if bool(is_preview) or (not bool(is_primary)):
            return PendingInitCompletionDecision(False)
        if context.pending_init_applied:
            return PendingInitCompletionDecision(False)
        rewrite = context.pending_init_rewrite
        if not rewrite:
            return PendingInitCompletionDecision(False)
        seed_for_ui: dict[str, float] = {}
        for species, value in pending_initial_seed_for_set(context.pending_init_seed, set_name=str(batch_set or "")).items():
            float_value = _try_float(value)
            if float_value is None:
                continue
            seed_for_ui[str(species)] = float_value
        if not seed_for_ui:
            return PendingInitCompletionDecision(False)
        return PendingInitCompletionDecision(True, seed_for_ui=seed_for_ui, rewrite=rewrite)

    def note_pending_init_apply_result(
        self,
        *,
        context: CompletionPolicyContext,
        applied: bool,
    ) -> CompletionPolicyContext:
        if not applied:
            return context
        return context.evolve(pending_init_applied=True)

    def should_arm_pending_init_guard(
        self,
        *,
        context: CompletionPolicyContext,
        is_preview: bool,
        is_primary: bool,
    ) -> Optional[str]:
        if bool(is_preview) or (not bool(is_primary)) or (not bool(context.pending_init_applied)):
            return None
        return context.pending_init_rewrite

    def resolve_pending_init_failure(
        self,
        context: Optional[CompletionPolicyContext],
    ) -> PendingInitFailureDecision:
        if context is None or (not bool(context.pending_init_applied)):
            return PendingInitFailureDecision(False, PolicyStatePatch())
        return PendingInitFailureDecision(
            True,
            PolicyStatePatch(context=context.evolve(pending_init_applied=False)),
        )

    def resolve_preflight_abort_pending_replay(
        self,
        *,
        pending_replay: PendingReplayState,
        explicit_run: bool,
    ) -> Optional[PendingReplayDirective]:
        if not bool(explicit_run):
            return None
        if (not bool(pending_replay.active)) and (not bool(pending_replay.target_set_ids)):
            return None
        return PendingReplayDirective.queue_fresh(target_set_ids=pending_replay.target_set_ids)

    def resolve_pending_replay_after_canonical_reset(
        self,
        *,
        pending_replay: PendingReplayState,
        reset_set_ids: tuple[str, ...],
    ) -> PendingReplayDirective:
        surviving_target_set_ids = tuple(
            set_id for set_id in pending_replay.target_set_ids if set_id not in set(reset_set_ids)
        )
        if surviving_target_set_ids:
            return PendingReplayDirective.preserve(
                target_set_ids=surviving_target_set_ids,
                clear_plot_updates=True,
            )
        return PendingReplayDirective.clear(clear_plot_updates=True)

    def resolve_explicit_error_pending_replay(
        self,
        *,
        fast_mode: bool,
        pending_replay: PendingReplayState,
    ) -> PendingReplayDirective:
        if (not bool(fast_mode)) and (bool(pending_replay.active) or bool(pending_replay.target_set_ids)):
            return PendingReplayDirective.queue_fresh(target_set_ids=pending_replay.target_set_ids)
        return PendingReplayDirective.clear(clear_plot_updates=False)

    def resolve_explicit_dirty_reset(
        self,
        *,
        context: CompletionPolicyContext,
        dirty_state_by_set_id: Mapping[str, DirtySetState],
    ) -> ExplicitDirtyResetDecision:
        eligible_reset_set_ids: list[str] = []
        for set_id in context.pending_workspace_reset_set_ids:
            expected_generation = context.pending_dirty_reset_generation_by_set_id.get(str(set_id))
            state = dirty_state_by_set_id.get(str(set_id))
            if expected_generation is None or state is None:
                continue
            if (not state.is_dirty) or state.generation is None:
                continue
            if int(state.generation) != int(expected_generation):
                continue
            eligible_reset_set_ids.append(str(set_id))
        updated_context = context.evolve(
            pending_workspace_reset_set_ids=(),
            pending_dirty_reset_generation_by_set_id={},
        )
        return ExplicitDirtyResetDecision(
            eligible_reset_set_ids=tuple(eligible_reset_set_ids),
            state_patch=PolicyStatePatch(context=updated_context),
        )
