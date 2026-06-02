from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Optional

from kindred.core.simulation_failure import (
    coerce_simulation_failure,
    is_cancelled_failure,
    simulation_failure_detail_text,
    simulation_failure_user_message,
)

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


def normalize_preview_target_set_ids(values: object) -> tuple[str, ...]:
    return _normalize_str_sequence(values, dedupe=True)


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


def cache_truth_generation_value(value: object) -> int:
    return _normalize_nonnegative_int(value, default=0)


def next_cache_truth_generation(value: object) -> int:
    return cache_truth_generation_value(value) + 1


@dataclass(frozen=True, slots=True)
class CompletionPolicyContext:
    active: bool
    request_id: Optional[int]
    run_id: Optional[int]
    fast_mode: bool
    parallel: bool
    keep_lane_pool_alive: bool
    queue_ids: tuple[str, ...] = ()
    queue_names: tuple[str, ...] = ()
    total: int = 0
    pos: int = 0
    primary_set_id: Optional[str] = None
    completed_set_ids: tuple[str, ...] = ()
    pending_workspace_reset_set_ids: tuple[str, ...] = ()
    pending_dirty_reset_generation_by_set_id: dict[str, int] = field(default_factory=dict)
    explicit_cache_preview_token: Optional[str] = None
    explicit_cache_preview_scope_set_ids: Optional[tuple[str, ...]] = None
    explicit_cache_valid_set_ids: Optional[tuple[str, ...]] = None
    explicit_cache_invalidated_set_ids: Optional[tuple[str, ...]] = None
    explicit_cache_truth_generation: Optional[int] = None
    preview_scope_set_ids: Optional[tuple[str, ...]] = None
    preview_owner_epoch: Optional[int] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_id", _normalize_optional_int(self.request_id))
        object.__setattr__(self, "run_id", _normalize_optional_int(self.run_id))
        object.__setattr__(self, "active", _normalize_bool(self.active))
        object.__setattr__(self, "fast_mode", _normalize_bool(self.fast_mode))
        object.__setattr__(self, "parallel", _normalize_bool(self.parallel))
        object.__setattr__(self, "keep_lane_pool_alive", _normalize_bool(self.keep_lane_pool_alive))
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
        object.__setattr__(
            self,
            "explicit_cache_truth_generation",
            _normalize_optional_int(self.explicit_cache_truth_generation),
        )
        object.__setattr__(self, "preview_scope_set_ids", _normalize_optional_set_ids(self.preview_scope_set_ids))
        object.__setattr__(self, "preview_owner_epoch", _normalize_optional_int(self.preview_owner_epoch))

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
class PolicyStatePatch:
    context: Optional[CompletionPolicyContext] = None
    clear_discarded_slider_preview_generation: bool = False


@dataclass(frozen=True, slots=True)
class SupersededFastDecision:
    display_current_preview: bool
    reset_status_progress: bool
    deactivate_context_immediately: bool
    state_patch: PolicyStatePatch


@dataclass(frozen=True, slots=True)
class SimulationFailurePolicyDecision:
    kind: str
    error_payload: Mapping[str, Any]
    error_text: str
    error_detail_text: str = ""
    preview_status_text: str = ""
    cancelled: bool = False

    @property
    def status_only_preview(self) -> bool:
        return self.kind == "status_only_preview"

    @property
    def terminal(self) -> bool:
        return self.kind == "terminal"


@dataclass(frozen=True, slots=True)
class CacheReconciliationDecision:
    clear_active_cache_identity_state: bool
    active_cache_key: Optional[str]
    active_cache_preview_token: Optional[str]
    active_cache_preview_scope_set_ids: Optional[tuple[str, ...]]
    active_cache_valid_set_ids: Optional[tuple[str, ...]]
    active_cache_invalidated_set_ids: Optional[tuple[str, ...]]
    redraw_valid_set_ids: Optional[tuple[str, ...]]
    has_redraw_subset: bool


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
    @staticmethod
    def resolve_simulation_failure(
        error_payload: object,
        *,
        fast_mode: bool,
    ) -> SimulationFailurePolicyDecision:
        payload = coerce_simulation_failure(error_payload)
        error_text = simulation_failure_user_message(payload)
        error_detail_text = simulation_failure_detail_text(payload)
        cancelled = is_cancelled_failure(payload)
        if bool(fast_mode) and not cancelled:
            status_text = SimulationCompletionPolicy._status_only_preview_failure_text(payload)
            if status_text:
                return SimulationFailurePolicyDecision(
                    kind="status_only_preview",
                    error_payload=payload,
                    error_text=error_text,
                    error_detail_text=error_detail_text,
                    preview_status_text=status_text,
                    cancelled=False,
                )
        return SimulationFailurePolicyDecision(
            kind="terminal",
            error_payload=payload,
            error_text=error_text,
            error_detail_text=error_detail_text,
            cancelled=bool(cancelled),
        )

    @staticmethod
    def _status_only_preview_failure_text(error_payload: Mapping[str, Any]) -> str:
        payload = coerce_simulation_failure(error_payload)
        kind = str(payload.get("kind") or "").strip().lower()
        details = payload.get("details") if isinstance(payload.get("details"), Mapping) else {}
        source = str(details.get("source") or "").strip().lower()
        stage = str(details.get("stage") or "").strip().lower()
        status_only = (
            kind == "timeout"
            or kind.endswith("_timeout")
            or kind.startswith("simulation_containment")
            or source == "simulation_containment"
            or stage == "wegscheider_cyclicity"
        )
        if not status_only:
            return ""
        if kind == "timeout":
            return "Preview timed out. Adjust sliders or run again."
        if stage == "wegscheider_cyclicity":
            return str(payload.get("message") or "Unresolved Wegscheider cyclicity.")
        return "Preview unavailable. Adjust sliders or run again."

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
        preview_ownership: Any,
        request_id: int,
        preview_owner_epoch: Optional[int] = None,
    ) -> bool:
        if preview_ownership.request_id is None:
            return False
        if int(preview_ownership.request_id) != int(request_id):
            return False
        if preview_owner_epoch is None:
            return True
        return int(preview_ownership.epoch) == int(preview_owner_epoch)

    def preview_request_can_display(
        self,
        *,
        preview_ownership: Any,
        request_id: Optional[int],
    ) -> bool:
        if request_id is None:
            return True
        return self.stale_fast_request_still_owns_current_state(
            preview_ownership=preview_ownership,
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
        preview_ownership: Any,
        context: Optional[CompletionPolicyContext],
        request_id: int,
        preview_owner_epoch: Optional[int] = None,
    ) -> SupersededFastDecision:
        request_id_i = int(request_id)
        context_matches_request = bool(
            context is not None
            and context.active
            and context.request_id is not None
            and int(context.request_id) == request_id_i
        )
        stale_fast_owns_current_state = self.stale_fast_request_still_owns_current_state(
            preview_ownership=preview_ownership,
            request_id=request_id_i,
            preview_owner_epoch=preview_owner_epoch,
        )
        display_current_preview = bool(stale_fast_owns_current_state)
        if display_current_preview:
            return SupersededFastDecision(
                display_current_preview=True,
                reset_status_progress=False,
                deactivate_context_immediately=False,
                state_patch=PolicyStatePatch(),
            )
        updated_context = context
        if context_matches_request and context is not None:
            updated_context = context.evolve(active=False)
        reset_status_progress = bool(context_matches_request)
        return SupersededFastDecision(
            display_current_preview=False,
            reset_status_progress=reset_status_progress,
            deactivate_context_immediately=bool(context_matches_request),
            state_patch=PolicyStatePatch(
                context=updated_context,
                clear_discarded_slider_preview_generation=True,
            ),
        )

    def resolve_superseded_fast_error(
        self,
        *,
        preview_ownership: Any,
        context: Optional[CompletionPolicyContext],
        request_id: int,
        preview_owner_epoch: Optional[int] = None,
    ) -> SupersededFastDecision:
        request_id_i = int(request_id)
        context_matches_request = bool(
            context is not None
            and context.active
            and context.request_id is not None
            and int(context.request_id) == request_id_i
        )
        self.stale_fast_request_still_owns_current_state(
            preview_ownership=preview_ownership,
            request_id=request_id_i,
            preview_owner_epoch=preview_owner_epoch,
        )
        updated_context = context
        if context_matches_request and context is not None:
            updated_context = context.evolve(active=False)
        return SupersededFastDecision(
            display_current_preview=False,
            reset_status_progress=bool(context_matches_request),
            deactivate_context_immediately=bool(context_matches_request),
            state_patch=PolicyStatePatch(
                context=updated_context,
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

        clear_active_cache_identity_state = bool(has_valid_subset and valid_set_ids == ())
        redraw_valid_set_ids = None
        has_redraw_subset = False
        if has_valid_subset:
            has_redraw_subset = True
            redraw_valid_set_ids = valid_set_ids or ()
        elif cache_state.active_cache_valid_set_ids is not None:
            has_redraw_subset = True
            redraw_valid_set_ids = cache_state.active_cache_valid_set_ids

        return CacheReconciliationDecision(
            clear_active_cache_identity_state=clear_active_cache_identity_state,
            active_cache_key=None if clear_active_cache_identity_state else cache_key_norm,
            active_cache_preview_token=None if clear_active_cache_identity_state else preview_token,
            active_cache_preview_scope_set_ids=None if clear_active_cache_identity_state else preview_scope_set_ids,
            active_cache_valid_set_ids=None if clear_active_cache_identity_state else valid_set_ids,
            active_cache_invalidated_set_ids=(
                None if clear_active_cache_identity_state else (invalidated_set_ids if has_valid_subset else None)
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
        next_generation = next_cache_truth_generation(context.explicit_cache_truth_generation)
        if cache_state.active_cache_key == _normalize_optional_str(cache_key):
            return context.evolve(
                explicit_cache_preview_token=cache_state.active_cache_preview_token,
                explicit_cache_preview_scope_set_ids=cache_state.active_cache_preview_scope_set_ids,
                explicit_cache_valid_set_ids=cache_state.active_cache_valid_set_ids,
                explicit_cache_invalidated_set_ids=cache_state.active_cache_invalidated_set_ids,
                explicit_cache_truth_generation=next_generation,
            )
        return context.evolve(
            explicit_cache_preview_token=None,
            explicit_cache_preview_scope_set_ids=(),
            explicit_cache_valid_set_ids=(),
            explicit_cache_invalidated_set_ids=(),
            explicit_cache_truth_generation=next_generation,
        )

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
