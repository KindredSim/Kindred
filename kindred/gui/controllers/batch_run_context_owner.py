from __future__ import annotations

from copy import deepcopy
from collections.abc import Mapping as MappingABC
from dataclasses import dataclass, fields
from typing import Any, Callable, Dict, Iterator, Mapping, Sequence

from kindred.gui.controllers.simulation_completion_policy import (
    CompletionPolicyContext,
    cache_truth_generation_value,
    next_cache_truth_generation,
)
from kindred.gui.ports import (
    CompletedRunDisplayCoverage,
    CompletedRunDisplayIntent,
    CompletedRunDisplayTransaction,
    CompletionDisplayEntry,
    DisplayTransitionCause,
)


@dataclass(frozen=True)
class BatchContextTransition:
    context: Dict[str, Any]
    completed_count: int
    batch_done: bool = False


@dataclass(frozen=True)
class BatchContextSeed:
    active: bool | None = None
    request_id: int | None = None
    run_id: int | None = None
    runtime_input_epoch: int | None = None
    runtime_input_global_epoch: int | None = None
    runtime_input_set_epoch_by_set_id: Mapping[str, Any] | None = None
    fast_mode: bool | None = None
    reuse_parallel_lane_pool: bool | None = None
    keep_lane_pool_alive: bool | None = None
    parallel: bool | None = None
    effective_workers: int | None = None
    prepared: Mapping[str, Any] | None = None
    prepared_by_set_id: Mapping[str, Mapping[str, Any]] | None = None
    simulation_plan: Mapping[str, Any] | None = None
    simulation_plan_by_set_id: Mapping[str, Mapping[str, Any]] | None = None
    cache_key: Any = None
    scope_identity: Mapping[str, Any] | None = None
    full_dsl: str | None = None
    mechanism_text_by_set_id: Mapping[str, str] | None = None
    mechanism_signature: str | None = None
    mechanism_signature_by_set_id: Mapping[str, str] | None = None
    simulation_identity_by_set_id: Mapping[str, Mapping[str, Any]] | None = None
    solver_config: Mapping[str, Any] | None = None
    t_end: float | None = None
    rows: Sequence[int] | None = None
    queue_ids: Sequence[str] | None = None
    queue_names: Sequence[str] | None = None
    pending_workspace_reset_set_ids: Sequence[str] | None = None
    pending_dirty_reset_generation_by_set_id: Mapping[str, Any] | None = None
    pos: int | None = None
    primary_set_id: str | None = None
    total: int | None = None
    completed_set_ids: Sequence[str] | None = None
    stale_runtime_input_set_ids: Sequence[str] | None = None
    failed_set_ids: Sequence[str] | None = None
    failed_set_errors: Mapping[str, Any] | None = None
    pending_init_seed: Mapping[str, Mapping[str, Any]] | None = None
    pending_init_rewrite: str | None = None
    pending_init_applied: bool | None = None
    explicit_cache_preview_token: str | None = None
    explicit_cache_preview_scope_set_ids: Sequence[str] | None = None
    explicit_cache_valid_set_ids: Sequence[str] | None = None
    explicit_cache_invalidated_set_ids: Sequence[str] | None = None
    preview_scope_set_ids: Sequence[str] | None = None
    preview_owner_epoch: int | None = None
    preview_batch_cache_token_by_set_id: Mapping[str, str] | None = None
    runtime_waiting: bool | None = None
    active_timeout_s: float | None = None
    completed_run_display_intent: CompletedRunDisplayIntent | None = None
    computed_owned_species_by_set_id: Mapping[str, Sequence[str]] | None = None

    def to_context(self) -> Dict[str, Any]:
        context: Dict[str, Any] = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if value is None:
                continue
            if isinstance(value, Mapping):
                context[field.name] = deepcopy(dict(value))
            elif isinstance(value, (list, tuple, set)):
                context[field.name] = list(value)
            else:
                context[field.name] = value
        return context


@dataclass(frozen=True)
class BatchCallbackContext(MappingABC[str, Any]):
    active: bool
    request_id: int | None
    run_id: int | None
    cache_key: str
    fast_mode: bool
    parallel: bool
    keep_lane_pool_alive: bool
    queue_ids: tuple[str, ...] = ()
    queue_names: tuple[str, ...] = ()
    total: int = 0
    pos: int = 0
    primary_set_id: str | None = None
    completed_set_ids: tuple[str, ...] = ()
    failed_set_ids: tuple[str, ...] = ()
    stale_runtime_input_set_ids: tuple[str, ...] = ()
    runtime_input_epoch: int | None = None
    runtime_input_global_epoch: int | None = None
    runtime_input_set_epoch_by_set_id: Mapping[str, int] | None = None
    pending_workspace_reset_set_ids: tuple[str, ...] = ()
    pending_dirty_reset_generation_by_set_id: Mapping[str, int] | None = None
    pending_init_seed: Mapping[str, Mapping[str, float]] | None = None
    pending_init_rewrite: str | None = None
    pending_init_applied: bool = False
    explicit_cache_preview_token: str | None = None
    explicit_cache_preview_scope_set_ids: tuple[str, ...] | None = None
    explicit_cache_valid_set_ids: tuple[str, ...] | None = None
    explicit_cache_invalidated_set_ids: tuple[str, ...] | None = None
    explicit_cache_truth_generation: int | None = None
    preview_scope_set_ids: tuple[str, ...] | None = None
    preview_owner_epoch: int | None = None
    completed_run_display_intent: CompletedRunDisplayIntent | None = None
    computed_owned_species_by_set_id: Mapping[str, Sequence[str]] | None = None

    def to_context(self) -> Dict[str, Any]:
        context: Dict[str, Any] = {
            "active": bool(self.active),
            "request_id": self.request_id,
            "run_id": self.run_id,
            "cache_key": str(self.cache_key or ""),
            "fast_mode": bool(self.fast_mode),
            "parallel": bool(self.parallel),
            "keep_lane_pool_alive": bool(self.keep_lane_pool_alive),
            "queue_ids": list(self.queue_ids),
            "queue_names": list(self.queue_names),
            "total": int(self.total),
            "pos": int(self.pos),
            "primary_set_id": self.primary_set_id,
            "completed_set_ids": list(self.completed_set_ids),
            "failed_set_ids": list(self.failed_set_ids),
            "stale_runtime_input_set_ids": list(self.stale_runtime_input_set_ids),
            "pending_workspace_reset_set_ids": list(self.pending_workspace_reset_set_ids),
            "pending_dirty_reset_generation_by_set_id": dict(self.pending_dirty_reset_generation_by_set_id or {}),
            "pending_init_seed": {
                str(set_name): {str(species): float(value) for species, value in dict(seed).items()}
                for set_name, seed in dict(self.pending_init_seed or {}).items()
                if str(set_name) and isinstance(seed, Mapping)
            },
            "pending_init_rewrite": self.pending_init_rewrite,
            "pending_init_applied": bool(self.pending_init_applied),
            "explicit_cache_preview_token": self.explicit_cache_preview_token,
            "explicit_cache_preview_scope_set_ids": self.explicit_cache_preview_scope_set_ids,
            "explicit_cache_valid_set_ids": self.explicit_cache_valid_set_ids,
            "explicit_cache_invalidated_set_ids": self.explicit_cache_invalidated_set_ids,
            "explicit_cache_truth_generation": self.explicit_cache_truth_generation,
            "preview_scope_set_ids": self.preview_scope_set_ids,
            "preview_owner_epoch": self.preview_owner_epoch,
        }
        if self.completed_run_display_intent is not None:
            context["completed_run_display_intent"] = self.completed_run_display_intent
        if self.computed_owned_species_by_set_id:
            context["computed_owned_species_by_set_id"] = {
                str(set_id): tuple(str(name) for name in (names or ()) if str(name))
                for set_id, names in dict(self.computed_owned_species_by_set_id or {}).items()
                if str(set_id)
            }
        if self.runtime_input_epoch is not None:
            context["runtime_input_epoch"] = self.runtime_input_epoch
        if self.runtime_input_global_epoch is not None:
            context["runtime_input_global_epoch"] = self.runtime_input_global_epoch
        if self.runtime_input_set_epoch_by_set_id:
            context["runtime_input_set_epoch_by_set_id"] = dict(self.runtime_input_set_epoch_by_set_id)
        return context

    def __getitem__(self, key: str) -> Any:
        return self.to_context()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_context())

    def __len__(self) -> int:
        return len(self.to_context())


@dataclass(frozen=True)
class BatchRunStartRequest:
    request_id: int
    run_id: int | None
    runtime_input_epoch: int
    runtime_input_global_epoch: int
    runtime_input_set_epoch_by_set_id: Mapping[str, Any]
    fast_mode: bool
    reuse_parallel_lane_pool: bool
    parallel: bool
    effective_workers: int
    retain_prepared_payloads_in_context: bool
    prepared_payload: Mapping[str, Any] | None
    prepared_payload_by_set_id: Mapping[str, Mapping[str, Any]]
    primary_simulation_plan: Mapping[str, Any] | None
    simulation_plan_by_set_id: Mapping[str, Mapping[str, Any]]
    cache_key: Any
    scope_identity: Mapping[str, Any]
    full_dsl: str
    mechanism_text_by_set_id: Mapping[str, str]
    mechanism_signature: str
    mechanism_signature_by_set_id: Mapping[str, str]
    simulation_identity_by_set_id: Mapping[str, Mapping[str, Any]]
    solver_config: Mapping[str, Any]
    t_end: float
    rows: Sequence[int]
    queue_ids: Sequence[str]
    queue_names: Sequence[str]
    pending_workspace_reset_set_ids: Sequence[str]
    pending_dirty_reset_generation_by_set_id: Mapping[str, Any]
    primary_set_id: str | None
    pending_init_seed: Mapping[str, Mapping[str, Any]]
    pending_init_rewrite: str | None
    pending_init_applied: bool
    explicit_cache_preview_token: str | None
    explicit_cache_preview_scope_set_ids: Sequence[str] | None
    explicit_cache_valid_set_ids: Sequence[str] | None
    explicit_cache_invalidated_set_ids: Sequence[str] | None
    preview_scope_set_ids: Sequence[str] | None
    preview_owner_epoch: int | None
    preview_batch_cache_token_by_set_id: Mapping[str, str]
    computed_owned_species_by_set_id: Mapping[str, Sequence[str]]
    completed_run_display_intent: CompletedRunDisplayIntent

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "runtime_input_set_epoch_by_set_id",
            deepcopy(dict(self.runtime_input_set_epoch_by_set_id or {})),
        )
        object.__setattr__(
            self,
            "prepared_payload",
            deepcopy(dict(self.prepared_payload)) if isinstance(self.prepared_payload, Mapping) else None,
        )
        object.__setattr__(
            self,
            "prepared_payload_by_set_id",
            {
                str(set_id): deepcopy(dict(payload))
                for set_id, payload in dict(self.prepared_payload_by_set_id or {}).items()
                if str(set_id) and isinstance(payload, Mapping)
            },
        )
        object.__setattr__(
            self,
            "primary_simulation_plan",
            deepcopy(dict(self.primary_simulation_plan))
            if isinstance(self.primary_simulation_plan, Mapping)
            else None,
        )
        object.__setattr__(
            self,
            "simulation_plan_by_set_id",
            {
                str(set_id): deepcopy(dict(payload))
                for set_id, payload in dict(self.simulation_plan_by_set_id or {}).items()
                if str(set_id) and isinstance(payload, Mapping)
            },
        )
        object.__setattr__(self, "scope_identity", deepcopy(dict(self.scope_identity or {})))
        object.__setattr__(
            self,
            "mechanism_text_by_set_id",
            {
                str(set_id): str(text)
                for set_id, text in dict(self.mechanism_text_by_set_id or {}).items()
                if str(set_id)
            },
        )
        object.__setattr__(
            self,
            "mechanism_signature_by_set_id",
            {
                str(set_id): str(signature)
                for set_id, signature in dict(self.mechanism_signature_by_set_id or {}).items()
                if str(set_id)
            },
        )
        object.__setattr__(
            self,
            "simulation_identity_by_set_id",
            {
                str(set_id): deepcopy(dict(payload))
                for set_id, payload in dict(self.simulation_identity_by_set_id or {}).items()
                if str(set_id) and isinstance(payload, Mapping)
            },
        )
        object.__setattr__(self, "solver_config", deepcopy(dict(self.solver_config or {})))
        object.__setattr__(self, "rows", tuple(int(row) for row in self.rows))
        object.__setattr__(self, "queue_ids", tuple(str(set_id) for set_id in self.queue_ids))
        object.__setattr__(self, "queue_names", tuple(str(name) for name in self.queue_names))
        object.__setattr__(
            self,
            "pending_workspace_reset_set_ids",
            tuple(str(set_id) for set_id in self.pending_workspace_reset_set_ids if str(set_id)),
        )
        object.__setattr__(
            self,
            "pending_dirty_reset_generation_by_set_id",
            {
                str(set_id): value
                for set_id, value in dict(self.pending_dirty_reset_generation_by_set_id or {}).items()
                if str(set_id)
            },
        )
        object.__setattr__(
            self,
            "pending_init_seed",
            {
                str(set_name): {str(species): float(value) for species, value in dict(seed).items()}
                for set_name, seed in dict(self.pending_init_seed or {}).items()
                if str(set_name) and isinstance(seed, Mapping)
            },
        )
        for name in (
            "explicit_cache_preview_scope_set_ids",
            "explicit_cache_valid_set_ids",
            "explicit_cache_invalidated_set_ids",
            "preview_scope_set_ids",
        ):
            value = getattr(self, name)
            object.__setattr__(
                self,
                name,
                tuple(str(set_id) for set_id in value if str(set_id)) if value is not None else None,
            )
        object.__setattr__(
            self,
            "preview_batch_cache_token_by_set_id",
            {
                str(set_id): str(token)
                for set_id, token in dict(self.preview_batch_cache_token_by_set_id or {}).items()
                if str(set_id)
            },
        )
        object.__setattr__(
            self,
            "computed_owned_species_by_set_id",
            {
                str(set_id): tuple(str(name) for name in (names or ()) if str(name))
                for set_id, names in dict(self.computed_owned_species_by_set_id or {}).items()
                if str(set_id)
            },
        )
        if not isinstance(self.completed_run_display_intent, CompletedRunDisplayIntent):
            raise TypeError("BatchRunStartRequest requires a CompletedRunDisplayIntent.")


@dataclass(frozen=True)
class BatchCompletionSummary:
    fast_mode: bool
    has_truthful_success: bool
    failed_set_ids: tuple[str, ...]
    failed_errors: Mapping[str, Any]


@dataclass(frozen=True)
class BatchCompletionCleanupState:
    fast_mode: bool
    parallel: bool
    keep_lane_pool_alive: bool


@dataclass(frozen=True)
class BatchCompletionFlushContext:
    cache_key: str
    request_id: int | None
    run_id: int | None


@dataclass(frozen=True)
class BatchScopedFailureCacheState:
    cache_key: str
    explicit_cache_valid_set_ids: tuple[str, ...]
    explicit_cache_invalidated_set_ids: tuple[str, ...]
    failed_count: int


@dataclass(frozen=True)
class BatchPendingDirtyResetState:
    set_ids: tuple[str, ...]
    generation_by_set_id: Dict[str, int]

    @property
    def empty(self) -> bool:
        return not self.set_ids and not self.generation_by_set_id


@dataclass(frozen=True)
class BatchExecutionPayloadState:
    prepared: Dict[str, Any] | None
    prepared_by_set_id: Dict[str, Dict[str, Any]]
    simulation_plan_by_set_id: Dict[str, Dict[str, Any]]
    mechanism_text_by_set_id: Dict[str, str]
    mechanism_signature_by_set_id: Dict[str, str]
    solver_config: Dict[str, Any]


@dataclass(frozen=True)
class BatchActiveState:
    active: bool
    parallel: bool
    fast_mode: bool
    runtime_waiting: bool
    run_id: int | None
    request_id: int | None
    rows: tuple[int, ...]
    queue_ids: tuple[str, ...]
    queue_names: tuple[str, ...]
    pos: int
    effective_workers: int


@dataclass(frozen=True)
class BatchErrorDispatchContext:
    run_id: int
    request_id: int
    fast_mode: bool
    preview_owner_epoch: int | None
    cache_key: str
    callback_context: BatchCallbackContext
    simulation_identity: Dict[str, Any]


@dataclass(frozen=True)
class BatchParallelStartPayload:
    run_id: int
    request_id: int
    rows: tuple[int, ...]
    queue_ids: tuple[str, ...]
    queue_names: tuple[str, ...]
    fast_mode: bool
    effective_workers: int
    keep_lane_pool_alive: bool
    preview_owner_epoch: int | None
    active_timeout_s: float
    cache_key: str
    full_dsl: str
    solver_config: Dict[str, Any]
    t_end: float
    simulation_plan_by_set_id: Dict[str, Dict[str, Any]]
    mechanism_text_by_set_id: Dict[str, str]
    simulation_identity_by_set_id: Dict[str, Dict[str, Any]]
    scope_identity: Dict[str, Any]
    preview_batch_cache_token_by_set_id: Dict[str, str]
    pending_init_seed: Any
    pending_init_applied: bool


@dataclass(frozen=True)
class BatchSerialNextPayload:
    pos: int
    total: int
    row: int
    set_id: str
    set_name: str | None
    queue_ids: tuple[str, ...]
    fast_mode: bool
    request_id: int
    cache_key: str
    preview_owner_epoch: int | None
    full_dsl: str
    solver_config: Dict[str, Any]
    t_end: float
    simulation_plan: Dict[str, Any] | None
    simulation_plan_by_set_id: Dict[str, Dict[str, Any]]
    mechanism_text_by_set_id: Dict[str, str]
    mechanism_signature_by_set_id: Dict[str, str]
    simulation_identity_by_set_id: Dict[str, Dict[str, Any]]
    prepared: Dict[str, Any] | None
    prepared_by_set_id: Dict[str, Dict[str, Any]]
    scope_identity: Dict[str, Any]
    preview_batch_cache_token_by_set_id: Dict[str, str]
    pending_init_seed: Any
    pending_init_applied: bool


@dataclass(frozen=True)
class BatchCompletionState:
    active: bool
    parallel: bool
    fast_mode: bool
    keep_lane_pool_alive: bool
    pos: int
    total: int
    queue_ids: tuple[str, ...]
    queue_names: tuple[str, ...]
    completed_set_ids: tuple[str, ...]
    stale_runtime_input_set_ids: tuple[str, ...]

    @property
    def completed_count(self) -> int:
        return len(self.completed_set_ids)


class BatchRunContextOwner:
    """Owns the mutable batch-run context for SimulationController."""

    _COMPLETED_RUN_DISPLAY_INTENT_KEY = "completed_run_display_intent"
    _COMPLETION_DISPLAY_ENTRIES_KEY = "completion_display_entries_by_set_id"
    _SEMANTIC_DISPLAY_UNAVAILABLE_SET_IDS_KEY = "semantic_display_unavailable_set_ids"
    _COMPUTED_OWNED_SPECIES_BY_SET_ID_KEY = "computed_owned_species_by_set_id"

    def __init__(self) -> None:
        self._context: Dict[str, Any] = {}

    def _current_context(self) -> Dict[str, Any]:
        return deepcopy(self._context)

    def callback_context_snapshot(
        self,
        context: Mapping[str, Any] | None = None,
    ) -> BatchCallbackContext:
        ctx = context if isinstance(context, Mapping) else self._context
        display_intent = ctx.get(self._COMPLETED_RUN_DISPLAY_INTENT_KEY)
        return BatchCallbackContext(
            active=self._coerce_bool(ctx.get("active")),
            request_id=self._optional_int(ctx.get("request_id")),
            run_id=self._optional_int(ctx.get("run_id")),
            cache_key=str(ctx.get("cache_key") or ""),
            fast_mode=self._coerce_bool(ctx.get("fast_mode")),
            parallel=self._coerce_bool(ctx.get("parallel")),
            keep_lane_pool_alive=self._coerce_bool(ctx.get("keep_lane_pool_alive")),
            queue_ids=self._str_tuple(ctx.get("queue_ids"), dedupe=False),
            queue_names=self._str_tuple(ctx.get("queue_names"), dedupe=False),
            total=max(0, self._int_value(ctx.get("total"), default=0)),
            pos=max(0, self._int_value(ctx.get("pos"), default=0)),
            primary_set_id=(str(ctx.get("primary_set_id")).strip() if ctx.get("primary_set_id") is not None else None),
            completed_set_ids=self._str_tuple(ctx.get("completed_set_ids"), dedupe=True),
            failed_set_ids=self._str_tuple(ctx.get("failed_set_ids"), dedupe=True),
            stale_runtime_input_set_ids=self._str_tuple(ctx.get("stale_runtime_input_set_ids"), dedupe=True),
            runtime_input_epoch=self._optional_int(ctx.get("runtime_input_epoch")),
            runtime_input_global_epoch=self._optional_int(ctx.get("runtime_input_global_epoch")),
            runtime_input_set_epoch_by_set_id={
                str(set_id): self._int_value(value, default=0)
                for set_id, value in dict(ctx.get("runtime_input_set_epoch_by_set_id") or {}).items()
                if str(set_id)
            },
            pending_workspace_reset_set_ids=self._str_tuple(ctx.get("pending_workspace_reset_set_ids"), dedupe=True),
            pending_dirty_reset_generation_by_set_id={
                str(set_id): self._int_value(value, default=0)
                for set_id, value in dict(ctx.get("pending_dirty_reset_generation_by_set_id") or {}).items()
                if str(set_id)
            },
            pending_init_seed={
                str(set_name): {str(species): float(value) for species, value in dict(seed).items()}
                for set_name, seed in dict(ctx.get("pending_init_seed") or {}).items()
                if str(set_name) and isinstance(seed, Mapping)
            },
            pending_init_rewrite=(
                str(ctx.get("pending_init_rewrite")) if ctx.get("pending_init_rewrite") is not None else None
            ),
            pending_init_applied=self._coerce_bool(ctx.get("pending_init_applied")),
            explicit_cache_preview_token=(
                str(ctx.get("explicit_cache_preview_token")) if ctx.get("explicit_cache_preview_token") is not None else None
            ),
            explicit_cache_preview_scope_set_ids=(
                self._str_tuple(ctx.get("explicit_cache_preview_scope_set_ids"), dedupe=True)
                if ctx.get("explicit_cache_preview_scope_set_ids") is not None
                else None
            ),
            explicit_cache_valid_set_ids=(
                self._str_tuple(ctx.get("explicit_cache_valid_set_ids"), dedupe=True)
                if ctx.get("explicit_cache_valid_set_ids") is not None
                else None
            ),
            explicit_cache_invalidated_set_ids=(
                self._str_tuple(ctx.get("explicit_cache_invalidated_set_ids"), dedupe=True)
                if ctx.get("explicit_cache_invalidated_set_ids") is not None
                else None
            ),
            explicit_cache_truth_generation=self._optional_int(ctx.get("explicit_cache_truth_generation")),
            preview_scope_set_ids=(
                self._str_tuple(ctx.get("preview_scope_set_ids"), dedupe=True)
                if ctx.get("preview_scope_set_ids") is not None
                else None
            ),
            preview_owner_epoch=self._optional_int(ctx.get("preview_owner_epoch")),
            completed_run_display_intent=display_intent,
            computed_owned_species_by_set_id={
                str(set_id): tuple(str(name) for name in (names or ()) if str(name))
                for set_id, names in dict(ctx.get(self._COMPUTED_OWNED_SPECIES_BY_SET_ID_KEY) or {}).items()
                if str(set_id)
            },
        )

    def _matches_current_identity(self, context: Mapping[str, Any]) -> bool:
        current = self._context
        if not isinstance(current, Mapping):
            return True
        has_identity = any(key in context or key in current for key in ("run_id", "request_id", "cache_key"))
        if not has_identity:
            return True
        for key in ("run_id", "request_id", "cache_key"):
            if key in context or key in current:
                if str(context.get(key) or "") != str(current.get(key) or ""):
                    return False
        for key in (
            "queue_ids",
            "completed_set_ids",
            "failed_set_ids",
            "stale_runtime_input_set_ids",
        ):
            if key in context or key in current:
                if self._str_tuple(context.get(key), dedupe=False) != self._str_tuple(
                    current.get(key), dedupe=False
                ):
                    return False
        for key in ("pos", "runtime_input_epoch", "runtime_input_global_epoch"):
            if key in context or key in current:
                if str(context.get(key) or "") != str(current.get(key) or ""):
                    return False
        if "runtime_input_set_epoch_by_set_id" in context or "runtime_input_set_epoch_by_set_id" in current:
            if {
                str(set_id): self._int_value(value, default=0)
                for set_id, value in dict(context.get("runtime_input_set_epoch_by_set_id") or {}).items()
            } != {
                str(set_id): self._int_value(value, default=0)
                for set_id, value in dict(current.get("runtime_input_set_epoch_by_set_id") or {}).items()
            }:
                return False
        return True

    def context_matches_current_identity(self, context: Mapping[str, Any] | None) -> bool:
        if not isinstance(context, Mapping):
            return True
        return self._matches_current_identity(context)

    def context_matches_current_run_identity(self, context: Mapping[str, Any] | None) -> bool:
        if not isinstance(context, Mapping):
            return True
        current = self._context
        if not isinstance(current, Mapping):
            return True
        for key in ("run_id", "request_id", "cache_key"):
            if key in context or key in current:
                if str(context.get(key) or "") != str(current.get(key) or ""):
                    return False
        return True

    def load_context(self, seed: BatchContextSeed) -> None:
        context = seed.to_context()
        self._context = context

    def active_batch_state(
        self,
        context: Mapping[str, Any] | None = None,
    ) -> BatchActiveState | None:
        ctx = context if isinstance(context, Mapping) else self._context
        if not isinstance(ctx, Mapping):
            return None
        active = self._coerce_bool(ctx.get("active"))
        runtime_waiting = self._coerce_bool(ctx.get("runtime_waiting"))
        if not active and not runtime_waiting:
            return None
        return BatchActiveState(
            active=active,
            parallel=self._coerce_bool(ctx.get("parallel")),
            fast_mode=self._coerce_bool(ctx.get("fast_mode")),
            runtime_waiting=runtime_waiting,
            run_id=self._optional_int(ctx.get("run_id")),
            request_id=self._optional_int(ctx.get("request_id")),
            rows=self._int_tuple(ctx.get("rows")),
            queue_ids=self._str_tuple(ctx.get("queue_ids"), dedupe=False),
            queue_names=self._str_tuple(ctx.get("queue_names"), dedupe=False),
            pos=max(0, self._int_value(ctx.get("pos"), default=0)),
            effective_workers=max(1, self._int_value(ctx.get("effective_workers"), default=1)),
        )

    def active_fast_preview_scope_set_ids(
        self,
        context: Mapping[str, Any] | None = None,
    ) -> tuple[str, ...] | None:
        ctx = context if isinstance(context, Mapping) else self._context
        state = self.active_batch_state(ctx)
        if state is None or not state.active or not state.fast_mode:
            return None
        queue_ids = self._str_tuple(ctx.get("queue_ids"), dedupe=True)
        if queue_ids:
            return queue_ids
        return self._str_tuple(ctx.get("preview_scope_set_ids"), dedupe=True)

    def active_parallel_error_dispatch_context(
        self,
        context: Mapping[str, Any] | None = None,
    ) -> BatchErrorDispatchContext | None:
        ctx = context if isinstance(context, Mapping) else self._context
        state = self.active_batch_state(ctx)
        if state is None or not state.active or not state.parallel:
            return None
        return BatchErrorDispatchContext(
            run_id=int(state.run_id or 0),
            request_id=int(state.request_id or 0),
            fast_mode=bool(state.fast_mode),
            preview_owner_epoch=self._optional_int(ctx.get("preview_owner_epoch")),
            cache_key=str(ctx.get("cache_key") or ""),
            callback_context=self.callback_context_snapshot(ctx),
            simulation_identity=deepcopy(dict(ctx.get("scope_identity") or {})),
        )

    def parallel_start_payload(
        self,
        context: Mapping[str, Any] | None = None,
    ) -> BatchParallelStartPayload | None:
        ctx = context if isinstance(context, Mapping) else self._context
        state = self.active_batch_state(ctx)
        if state is None or not state.active or not state.parallel:
            return None
        return BatchParallelStartPayload(
            run_id=int(state.run_id or 0),
            request_id=int(state.request_id or 0),
            rows=state.rows,
            queue_ids=state.queue_ids,
            queue_names=state.queue_names,
            fast_mode=bool(state.fast_mode),
            effective_workers=max(1, int(state.effective_workers)),
            keep_lane_pool_alive=self._coerce_bool(ctx.get("keep_lane_pool_alive")),
            preview_owner_epoch=self._optional_int(ctx.get("preview_owner_epoch")),
            active_timeout_s=float(self._float_value(ctx.get("active_timeout_s"), default=60.0)),
            cache_key=str(ctx.get("cache_key") or ""),
            full_dsl=str(ctx.get("full_dsl") or ""),
            solver_config=deepcopy(dict(ctx.get("solver_config") or {})),
            t_end=float(self._float_value(ctx.get("t_end"), default=0.0)),
            simulation_plan_by_set_id={
                str(set_id): deepcopy(dict(payload))
                for set_id, payload in dict(ctx.get("simulation_plan_by_set_id") or {}).items()
                if str(set_id) and isinstance(payload, Mapping)
            },
            mechanism_text_by_set_id={
                str(set_id): str(text)
                for set_id, text in dict(ctx.get("mechanism_text_by_set_id") or {}).items()
                if str(set_id)
            },
            simulation_identity_by_set_id={
                str(set_id): deepcopy(dict(payload))
                for set_id, payload in dict(ctx.get("simulation_identity_by_set_id") or {}).items()
                if str(set_id) and isinstance(payload, Mapping)
            },
            scope_identity=deepcopy(dict(ctx.get("scope_identity") or {})),
            preview_batch_cache_token_by_set_id={
                str(set_id): str(token)
                for set_id, token in dict(ctx.get("preview_batch_cache_token_by_set_id") or {}).items()
                if str(set_id)
            },
            pending_init_seed=deepcopy(ctx.get("pending_init_seed")),
            pending_init_applied=self._coerce_bool(ctx.get("pending_init_applied")),
        )

    def serial_next_payload(
        self,
        context: Mapping[str, Any] | None = None,
    ) -> BatchSerialNextPayload | None:
        ctx = context if isinstance(context, Mapping) else self._context
        state = self.active_batch_state(ctx)
        if state is None or not state.active or state.parallel:
            return None
        queue_ids = state.queue_ids
        pos = max(0, int(state.pos))
        if not (0 <= pos < len(queue_ids)):
            return None
        rows = state.rows
        row = int(rows[pos]) if 0 <= pos < len(rows) else 0
        queue_names = state.queue_names
        set_id = str(queue_ids[pos])
        set_name = str(queue_names[pos]) if 0 <= pos < len(queue_names) else None
        simulation_plan = ctx.get("simulation_plan")
        prepared = ctx.get("prepared")
        return BatchSerialNextPayload(
            pos=pos,
            total=len(queue_ids),
            row=row,
            set_id=set_id,
            set_name=set_name,
            queue_ids=queue_ids,
            fast_mode=bool(state.fast_mode),
            request_id=int(state.request_id or 0),
            cache_key=str(ctx.get("cache_key") or ""),
            preview_owner_epoch=self._optional_int(ctx.get("preview_owner_epoch")),
            full_dsl=str(ctx.get("full_dsl") or ""),
            solver_config=deepcopy(dict(ctx.get("solver_config") or {})),
            t_end=float(self._float_value(ctx.get("t_end"), default=0.0)),
            simulation_plan=(
                deepcopy(dict(simulation_plan))
                if isinstance(simulation_plan, Mapping)
                else None
            ),
            simulation_plan_by_set_id={
                str(set_id): deepcopy(dict(payload))
                for set_id, payload in dict(ctx.get("simulation_plan_by_set_id") or {}).items()
                if str(set_id) and isinstance(payload, Mapping)
            },
            mechanism_text_by_set_id={
                str(set_id): str(text)
                for set_id, text in dict(ctx.get("mechanism_text_by_set_id") or {}).items()
                if str(set_id)
            },
            mechanism_signature_by_set_id={
                str(set_id): str(signature)
                for set_id, signature in dict(ctx.get("mechanism_signature_by_set_id") or {}).items()
                if str(set_id)
            },
            simulation_identity_by_set_id={
                str(set_id): deepcopy(dict(payload))
                for set_id, payload in dict(ctx.get("simulation_identity_by_set_id") or {}).items()
                if str(set_id) and isinstance(payload, Mapping)
            },
            prepared=deepcopy(dict(prepared)) if isinstance(prepared, Mapping) else None,
            prepared_by_set_id={
                str(set_id): deepcopy(dict(payload))
                for set_id, payload in dict(ctx.get("prepared_by_set_id") or {}).items()
                if str(set_id) and isinstance(payload, Mapping)
            },
            scope_identity=deepcopy(dict(ctx.get("scope_identity") or {})),
            preview_batch_cache_token_by_set_id={
                str(set_id): str(token)
                for set_id, token in dict(ctx.get("preview_batch_cache_token_by_set_id") or {}).items()
                if str(set_id)
            },
            pending_init_seed=deepcopy(ctx.get("pending_init_seed")),
            pending_init_applied=self._coerce_bool(ctx.get("pending_init_applied")),
        )

    def completion_state(
        self,
        context: Mapping[str, Any] | None = None,
    ) -> BatchCompletionState | None:
        ctx = context if isinstance(context, Mapping) else self._context
        if not isinstance(ctx, Mapping):
            return None
        active = self._coerce_bool(ctx.get("active"))
        if not active and not self._coerce_bool(ctx.get("runtime_waiting")):
            return None
        queue_ids = self._str_tuple(ctx.get("queue_ids"), dedupe=False)
        completed_set_ids = self._str_tuple(ctx.get("completed_set_ids"), dedupe=True)
        total = self._int_value(ctx.get("total"), default=len(queue_ids))
        if total <= 0:
            total = len(queue_ids)
        return BatchCompletionState(
            active=active,
            parallel=self._coerce_bool(ctx.get("parallel")),
            fast_mode=self._coerce_bool(ctx.get("fast_mode")),
            keep_lane_pool_alive=self._coerce_bool(ctx.get("keep_lane_pool_alive")),
            pos=max(0, self._int_value(ctx.get("pos"), default=0)),
            total=max(0, int(total)),
            queue_ids=queue_ids,
            queue_names=self._str_tuple(ctx.get("queue_names"), dedupe=False),
            completed_set_ids=completed_set_ids,
            stale_runtime_input_set_ids=self._str_tuple(ctx.get("stale_runtime_input_set_ids"), dedupe=True),
        )

    def completion_cleanup_state(
        self,
        context: Mapping[str, Any] | None = None,
    ) -> BatchCompletionCleanupState:
        ctx = context if isinstance(context, Mapping) else self._context
        return BatchCompletionCleanupState(
            fast_mode=self._coerce_bool(ctx.get("fast_mode")),
            parallel=self._coerce_bool(ctx.get("parallel")),
            keep_lane_pool_alive=self._coerce_bool(ctx.get("keep_lane_pool_alive")),
        )

    def completion_flush_context(
        self,
        context: Mapping[str, Any] | None = None,
    ) -> BatchCompletionFlushContext:
        ctx = context if isinstance(context, Mapping) else self._context
        return BatchCompletionFlushContext(
            cache_key=str(ctx.get("cache_key") or ""),
            request_id=self._optional_int(ctx.get("request_id")),
            run_id=self._optional_int(ctx.get("run_id")),
        )

    def scoped_failure_cache_state(
        self,
        context: Mapping[str, Any] | None = None,
    ) -> BatchScopedFailureCacheState:
        ctx = context if isinstance(context, Mapping) else self._context
        failed_set_ids = self._str_tuple(ctx.get("failed_set_ids"), dedupe=True)
        return BatchScopedFailureCacheState(
            cache_key=str(ctx.get("cache_key") or ""),
            explicit_cache_valid_set_ids=self._str_tuple(
                ctx.get("explicit_cache_valid_set_ids"),
                dedupe=True,
            ),
            explicit_cache_invalidated_set_ids=self._str_tuple(
                ctx.get("explicit_cache_invalidated_set_ids"),
                dedupe=True,
            ),
            failed_count=len(failed_set_ids),
        )

    def pending_dirty_reset_state(
        self,
        context: Mapping[str, Any] | None = None,
    ) -> BatchPendingDirtyResetState:
        ctx = context if isinstance(context, Mapping) else self._context
        set_ids = self._str_tuple(ctx.get("pending_workspace_reset_set_ids"), dedupe=True)
        generation_by_set_id: Dict[str, int] = {}
        for set_id, value in dict(ctx.get("pending_dirty_reset_generation_by_set_id") or {}).items():
            sid = str(set_id or "").strip()
            if not sid:
                continue
            generation_by_set_id[sid] = self._int_value(value, default=0)
        if set_ids:
            generation_by_set_id = {
                set_id: int(generation_by_set_id[set_id])
                for set_id in set_ids
                if set_id in generation_by_set_id
            }
        return BatchPendingDirtyResetState(
            set_ids=set_ids,
            generation_by_set_id=generation_by_set_id,
        )

    def simulation_plan_payload_for_set(
        self,
        set_id: str | None,
        context: Mapping[str, Any] | None = None,
    ) -> Dict[str, Any]:
        sid = str(set_id or "").strip()
        ctx = context if isinstance(context, Mapping) else self._context
        if not sid or not isinstance(ctx, Mapping):
            return {}
        plan_by_set_id = ctx.get("simulation_plan_by_set_id")
        if not isinstance(plan_by_set_id, Mapping):
            return {}
        payload = plan_by_set_id.get(sid)
        if not isinstance(payload, Mapping):
            return {}
        return deepcopy(dict(payload))

    def execution_payload_state(
        self,
        context: Mapping[str, Any] | None = None,
    ) -> BatchExecutionPayloadState:
        ctx = context if isinstance(context, Mapping) else self._context
        prepared = ctx.get("prepared") if isinstance(ctx, Mapping) else None
        return BatchExecutionPayloadState(
            prepared=deepcopy(dict(prepared)) if isinstance(prepared, Mapping) else None,
            prepared_by_set_id={
                str(set_id): deepcopy(dict(payload))
                for set_id, payload in dict(ctx.get("prepared_by_set_id") or {}).items()
                if str(set_id) and isinstance(payload, Mapping)
            } if isinstance(ctx, Mapping) else {},
            simulation_plan_by_set_id={
                str(set_id): deepcopy(dict(payload))
                for set_id, payload in dict(ctx.get("simulation_plan_by_set_id") or {}).items()
                if str(set_id) and isinstance(payload, Mapping)
            } if isinstance(ctx, Mapping) else {},
            mechanism_text_by_set_id={
                str(set_id): str(text)
                for set_id, text in dict(ctx.get("mechanism_text_by_set_id") or {}).items()
                if str(set_id)
            } if isinstance(ctx, Mapping) else {},
            mechanism_signature_by_set_id={
                str(set_id): str(signature)
                for set_id, signature in dict(ctx.get("mechanism_signature_by_set_id") or {}).items()
                if str(set_id)
            } if isinstance(ctx, Mapping) else {},
            solver_config=deepcopy(dict(ctx.get("solver_config") or {})) if isinstance(ctx, Mapping) else {},
        )

    def preview_batch_cache_token_for_set(
        self,
        set_id: str | None,
        context: Mapping[str, Any] | None = None,
    ) -> str:
        sid = str(set_id or "").strip()
        ctx = context if isinstance(context, Mapping) else self._context
        if not sid or not isinstance(ctx, Mapping):
            return ""
        token_by_set_id = ctx.get("preview_batch_cache_token_by_set_id")
        if not isinstance(token_by_set_id, Mapping):
            return ""
        return str(token_by_set_id.get(sid) or "")

    def explicit_batch_coalescing_for_completion(
        self,
        *,
        slider_triggered: bool,
        context: Mapping[str, Any] | None = None,
    ) -> bool:
        ctx = context if isinstance(context, Mapping) else self._context
        return bool(
            self._coerce_bool(ctx.get("parallel"))
            and (not bool(slider_triggered))
            and self._int_value(ctx.get("total"), default=0) > 1
        )

    def completion_cache_key(
        self,
        context: Mapping[str, Any] | None = None,
    ) -> str:
        ctx = context if isinstance(context, Mapping) else self._context
        return str(ctx.get("cache_key") or "")

    def primary_set_id(
        self,
        context: Mapping[str, Any] | None = None,
    ) -> str:
        ctx = context if isinstance(context, Mapping) else self._context
        return str(ctx.get("primary_set_id") or "").strip()

    def include_mechanism_in_result_payload(
        self,
        *,
        fast_mode: bool,
        batch_set_id: str | None,
        context: Mapping[str, Any] | None = None,
    ) -> bool:
        if bool(fast_mode):
            return False
        set_id = str(batch_set_id or "").strip()
        if not set_id:
            return True
        primary_set = self.primary_set_id(context if isinstance(context, Mapping) else None)
        if primary_set:
            return set_id == primary_set
        return True

    def simulation_identity_for_set(
        self,
        set_id: str,
        context: Mapping[str, Any] | None = None,
    ) -> Dict[str, Any]:
        sid = str(set_id or "").strip()
        if not sid:
            return {}
        ctx = context if isinstance(context, Mapping) else self._context
        raw_by_set_id = ctx.get("simulation_identity_by_set_id")
        if not isinstance(raw_by_set_id, Mapping):
            return {}
        raw_identity = raw_by_set_id.get(sid)
        return deepcopy(dict(raw_identity)) if isinstance(raw_identity, Mapping) else {}

    def active_fast_mode(
        self,
        context: Mapping[str, Any] | None = None,
    ) -> bool:
        state = self.active_batch_state(context)
        if state is not None:
            return bool(state.fast_mode)
        ctx = context if isinstance(context, Mapping) else self._context
        return self._coerce_bool(ctx.get("fast_mode"))

    def start_run(self, request: BatchRunStartRequest) -> Dict[str, Any]:
        retain_prepared = bool(request.retain_prepared_payloads_in_context)
        serial_explicit = bool((not request.fast_mode) and retain_prepared)
        context: Dict[str, Any] = {
            "active": True,
            "request_id": int(request.request_id),
            "run_id": request.run_id,
            "runtime_input_epoch": int(request.runtime_input_epoch),
            "runtime_input_global_epoch": int(request.runtime_input_global_epoch),
            "runtime_input_set_epoch_by_set_id": {
                str(set_id): value
                for set_id, value in dict(request.runtime_input_set_epoch_by_set_id or {}).items()
                if str(set_id)
            },
            "fast_mode": bool(request.fast_mode),
            "reuse_parallel_lane_pool": bool(request.reuse_parallel_lane_pool),
            "keep_lane_pool_alive": bool(request.reuse_parallel_lane_pool and request.parallel),
            "parallel": bool(request.parallel),
            "effective_workers": int(request.effective_workers),
            "prepared": (
                deepcopy(dict(request.prepared_payload))
                if serial_explicit and isinstance(request.prepared_payload, Mapping)
                else None
            ),
            "prepared_by_set_id": (
                {
                    str(set_id): deepcopy(dict(payload))
                    for set_id, payload in dict(request.prepared_payload_by_set_id or {}).items()
                    if str(set_id) and isinstance(payload, Mapping)
                }
                if retain_prepared
                else {}
            ),
            "simulation_plan": (
                deepcopy(dict(request.primary_simulation_plan))
                if ((not request.fast_mode) and isinstance(request.primary_simulation_plan, Mapping))
                else None
            ),
            "simulation_plan_by_set_id": {
                str(set_id): deepcopy(dict(payload))
                for set_id, payload in dict(request.simulation_plan_by_set_id or {}).items()
                if str(set_id) and isinstance(payload, Mapping)
            },
            "cache_key": request.cache_key,
            "scope_identity": deepcopy(dict(request.scope_identity or {})),
            "full_dsl": str(request.full_dsl),
            "mechanism_text_by_set_id": {
                str(set_id): str(text)
                for set_id, text in dict(request.mechanism_text_by_set_id or {}).items()
                if str(set_id)
            },
            "mechanism_signature": str(request.mechanism_signature),
            "mechanism_signature_by_set_id": {
                str(set_id): str(signature)
                for set_id, signature in dict(request.mechanism_signature_by_set_id or {}).items()
                if str(set_id)
            },
            "simulation_identity_by_set_id": {
                str(set_id): deepcopy(dict(payload))
                for set_id, payload in dict(request.simulation_identity_by_set_id or {}).items()
                if str(set_id) and isinstance(payload, Mapping)
            },
            "solver_config": deepcopy(dict(request.solver_config or {})),
            "t_end": float(request.t_end),
            "rows": [int(row) for row in request.rows],
            "queue_ids": [str(set_id) for set_id in request.queue_ids],
            "queue_names": [str(name) for name in request.queue_names],
            "pending_workspace_reset_set_ids": [
                str(set_id) for set_id in request.pending_workspace_reset_set_ids if str(set_id)
            ],
            "pending_dirty_reset_generation_by_set_id": {
                str(set_id): value
                for set_id, value in dict(request.pending_dirty_reset_generation_by_set_id or {}).items()
                if str(set_id)
            },
            "pos": 0,
            "primary_set_id": request.primary_set_id,
            "total": len(request.queue_ids),
            "completed_set_ids": [],
            "pending_init_seed": {
                str(set_name): {str(species): float(value) for species, value in dict(seed).items()}
                for set_name, seed in dict(request.pending_init_seed or {}).items()
                if str(set_name) and isinstance(seed, Mapping)
            },
            "pending_init_rewrite": request.pending_init_rewrite,
            "pending_init_applied": bool(request.pending_init_applied),
            "explicit_cache_preview_token": request.explicit_cache_preview_token,
            "explicit_cache_preview_scope_set_ids": request.explicit_cache_preview_scope_set_ids,
            "explicit_cache_valid_set_ids": request.explicit_cache_valid_set_ids,
            "explicit_cache_invalidated_set_ids": request.explicit_cache_invalidated_set_ids,
            "explicit_cache_truth_generation": 0,
            "preview_scope_set_ids": request.preview_scope_set_ids,
            "preview_owner_epoch": request.preview_owner_epoch,
            "preview_batch_cache_token_by_set_id": {
                str(set_id): str(token)
                for set_id, token in dict(request.preview_batch_cache_token_by_set_id or {}).items()
                if str(set_id)
            },
            self._COMPUTED_OWNED_SPECIES_BY_SET_ID_KEY: {
                str(set_id): tuple(str(name) for name in (names or ()) if str(name))
                for set_id, names in dict(request.computed_owned_species_by_set_id or {}).items()
                if str(set_id)
            },
        }
        context[self._COMPLETED_RUN_DISPLAY_INTENT_KEY] = request.completed_run_display_intent
        self._context = context
        return self._current_context()

    def clear(self) -> Dict[str, Any]:
        self._context = {}
        return self._current_context()

    def _update(self, **updates: Any) -> Dict[str, Any]:
        context = dict(self._context)
        context.update(updates)
        self._context = context
        return self._current_context()

    def deactivate(self) -> Dict[str, Any]:
        if not self._context:
            return self._current_context()
        return self._update(active=False)

    def record_completion_display_entry(
        self,
        context: Mapping[str, Any] | None,
        *,
        set_id: str | None,
        label: str | None,
        entry: CompletionDisplayEntry,
    ) -> Dict[str, Any]:
        sid = str(set_id or "").strip()
        if not sid or not isinstance(entry, CompletionDisplayEntry):
            return self._current_context() if context is None else deepcopy(dict(context))
        owned_species = tuple(str(name) for name in entry.owned_species if str(name))
        if not owned_species:
            return self.record_completion_display_unavailable(
                context,
                set_id=sid,
                cause=DisplayTransitionCause.SEMANTIC_METADATA_UNAVAILABLE,
            )
        base = context if isinstance(context, Mapping) else self._context
        raw = (
            dict(self._context)
            if isinstance(base, Mapping) and self.context_matches_current_run_identity(base)
            else dict(base or {})
        )
        entries_raw = raw.get(self._COMPLETION_DISPLAY_ENTRIES_KEY)
        entries = deepcopy(dict(entries_raw)) if isinstance(entries_raw, Mapping) else {}
        entries[sid] = CompletionDisplayEntry(
            set_id=sid,
            label=str(label or sid),
            t=entry.t,
            series=deepcopy(dict(entry.series or {})),
            algebra_scalars=deepcopy(dict(entry.algebra_scalars or {})),
            solver_provenance=deepcopy(dict(entry.solver_provenance or {})),
            mechanism_text=str(entry.mechanism_text or ""),
            solver_config=deepcopy(dict(entry.solver_config or {})),
            warnings=tuple(deepcopy(dict(warning)) for warning in entry.warnings if isinstance(warning, Mapping)),
            completion_provenance=deepcopy(dict(entry.completion_provenance or {})),
            owned_species=owned_species,
        )
        raw[self._COMPLETION_DISPLAY_ENTRIES_KEY] = entries
        unavailable = [
            set_id_s
            for set_id_s in self._str_tuple(
                raw.get(self._SEMANTIC_DISPLAY_UNAVAILABLE_SET_IDS_KEY),
                dedupe=True,
            )
            if set_id_s != sid
        ]
        if unavailable:
            raw[self._SEMANTIC_DISPLAY_UNAVAILABLE_SET_IDS_KEY] = tuple(unavailable)
        else:
            raw.pop(self._SEMANTIC_DISPLAY_UNAVAILABLE_SET_IDS_KEY, None)
        if self._matches_current_identity(raw):
            self._context = raw
            return self._current_context()
        return deepcopy(raw)

    @staticmethod
    def launch_owned_species_for_computed_result(
        context: Mapping[str, Any] | None,
        *,
        set_id: str | None,
    ) -> tuple[str, ...]:
        if not isinstance(context, Mapping):
            return ()
        sid = str(set_id or "").strip()
        if not sid:
            return ()
        owned_by_set = context.get(BatchRunContextOwner._COMPUTED_OWNED_SPECIES_BY_SET_ID_KEY)
        if not isinstance(owned_by_set, Mapping):
            return ()
        return tuple(str(name) for name in (owned_by_set.get(sid) or ()) if str(name))

    def record_completion_display_unavailable(
        self,
        context: Mapping[str, Any] | None,
        *,
        set_id: str | None,
        cause: DisplayTransitionCause = DisplayTransitionCause.SEMANTIC_METADATA_UNAVAILABLE,
    ) -> Dict[str, Any]:
        if not isinstance(cause, DisplayTransitionCause):
            raise TypeError("Completion display unavailable requires DisplayTransitionCause")
        sid = str(set_id or "").strip()
        if not sid:
            return self._current_context() if context is None else deepcopy(dict(context))
        base = context if isinstance(context, Mapping) else self._context
        raw = (
            dict(self._context)
            if isinstance(base, Mapping) and self.context_matches_current_run_identity(base)
            else dict(base or {})
        )
        entries_raw = raw.get(self._COMPLETION_DISPLAY_ENTRIES_KEY)
        entries = deepcopy(dict(entries_raw)) if isinstance(entries_raw, Mapping) else {}
        entries.pop(sid, None)
        if entries:
            raw[self._COMPLETION_DISPLAY_ENTRIES_KEY] = entries
        else:
            raw.pop(self._COMPLETION_DISPLAY_ENTRIES_KEY, None)

        unavailable = list(
            self._str_tuple(
                raw.get(self._SEMANTIC_DISPLAY_UNAVAILABLE_SET_IDS_KEY),
                dedupe=True,
            )
        )
        if sid not in unavailable:
            unavailable.append(sid)
        raw[self._SEMANTIC_DISPLAY_UNAVAILABLE_SET_IDS_KEY] = tuple(unavailable)
        if isinstance(cause, DisplayTransitionCause):
            raw["completion_display_unavailable_cause"] = cause.value
        if self._matches_current_identity(raw):
            self._context = raw
            return self._current_context()
        return deepcopy(raw)

    def completed_run_display_coverage(
        self,
        context: Mapping[str, Any] | None = None,
    ) -> CompletedRunDisplayCoverage:
        ctx = context if isinstance(context, Mapping) else self._context
        if not isinstance(ctx, Mapping):
            return CompletedRunDisplayCoverage(cause=DisplayTransitionCause.NO_DISPLAYABLE_COMPLETION_RESULTS)
        intent = ctx.get(self._COMPLETED_RUN_DISPLAY_INTENT_KEY)
        if not isinstance(intent, CompletedRunDisplayIntent) or not intent.requested_show_set_ids:
            return CompletedRunDisplayCoverage(cause=DisplayTransitionCause.NO_DISPLAYABLE_COMPLETION_RESULTS)
        intent_set_ids = tuple(str(set_id) for set_id in intent.requested_show_set_ids if str(set_id))
        run_target_ids_raw = tuple(str(set_id) for set_id in intent.run_target_set_ids if str(set_id))
        run_target_members = set(run_target_ids_raw)
        if run_target_members:
            run_target_set_ids = tuple(set_id for set_id in intent_set_ids if set_id in run_target_members)
        else:
            run_target_set_ids = intent_set_ids
        run_target_members = set(run_target_set_ids)
        non_run_requested_ids = tuple(set_id for set_id in intent_set_ids if set_id not in run_target_members)
        failed_set_ids = {str(set_id) for set_id in (ctx.get("failed_set_ids") or ()) if str(set_id)}
        failed_intent_set_ids = tuple(set_id for set_id in intent_set_ids if set_id in failed_set_ids)
        failed_run_target_set_ids = tuple(set_id for set_id in run_target_set_ids if set_id in failed_set_ids)
        display_set_ids = tuple(set_id for set_id in run_target_set_ids if set_id not in failed_set_ids)

        def _intent_ordered_union(*groups: Sequence[str]) -> tuple[str, ...]:
            members = {str(set_id) for group in groups for set_id in (group or ()) if str(set_id)}
            return tuple(set_id for set_id in intent_set_ids if set_id in members)

        if not display_set_ids:
            unresolved = _intent_ordered_union(failed_run_target_set_ids, non_run_requested_ids)
            return CompletedRunDisplayCoverage(
                intent=intent,
                missing_set_ids=non_run_requested_ids,
                unavailable_set_ids=unresolved,
                unresolved_intent_set_ids=unresolved,
                failed_intent_set_ids=failed_intent_set_ids,
                cause=DisplayTransitionCause.NO_DISPLAYABLE_COMPLETION_RESULTS,
            )
        display_primary_set_id = (
            str(intent.primary_set_id)
            if str(intent.primary_set_id or "") in set(display_set_ids)
            else str(display_set_ids[0])
        )
        unavailable_set_ids = set(
            self._str_tuple(
                ctx.get(self._SEMANTIC_DISPLAY_UNAVAILABLE_SET_IDS_KEY),
                dedupe=True,
            )
        )
        if unavailable_set_ids and all(set_id in unavailable_set_ids for set_id in display_set_ids):
            semantic_ids = tuple(set_id for set_id in display_set_ids if set_id in unavailable_set_ids)
            unresolved = _intent_ordered_union(failed_run_target_set_ids, semantic_ids, non_run_requested_ids)
            return CompletedRunDisplayCoverage(
                intent=intent,
                missing_set_ids=non_run_requested_ids,
                unavailable_set_ids=unresolved,
                unresolved_intent_set_ids=unresolved,
                failed_intent_set_ids=failed_intent_set_ids,
                semantic_unavailable_set_ids=semantic_ids,
                cause=DisplayTransitionCause.SEMANTIC_METADATA_UNAVAILABLE,
            )

        entries_raw = ctx.get(self._COMPLETION_DISPLAY_ENTRIES_KEY)
        if not isinstance(entries_raw, Mapping):
            if unavailable_set_ids:
                semantic_ids = tuple(
                    set_id
                    for set_id in intent_set_ids
                    if set_id in unavailable_set_ids and set_id not in failed_set_ids
                )
                missing_ids = tuple(set_id for set_id in display_set_ids if set_id not in unavailable_set_ids)
                unresolved = _intent_ordered_union(failed_run_target_set_ids, missing_ids, semantic_ids, non_run_requested_ids)
                return CompletedRunDisplayCoverage(
                    intent=intent,
                    missing_set_ids=(*missing_ids, *non_run_requested_ids),
                    unavailable_set_ids=unresolved,
                    unresolved_intent_set_ids=unresolved,
                    failed_intent_set_ids=failed_intent_set_ids,
                    semantic_unavailable_set_ids=semantic_ids,
                    cause=(
                        DisplayTransitionCause.IN_FLIGHT_COVERAGE_UNAVAILABLE
                        if missing_ids
                        else DisplayTransitionCause.SEMANTIC_METADATA_UNAVAILABLE
                    ),
                )
            if failed_run_target_set_ids:
                unresolved = _intent_ordered_union(failed_run_target_set_ids, display_set_ids, non_run_requested_ids)
                return CompletedRunDisplayCoverage(
                    intent=intent,
                    missing_set_ids=display_set_ids,
                    unavailable_set_ids=unresolved,
                    unresolved_intent_set_ids=unresolved,
                    failed_intent_set_ids=failed_intent_set_ids,
                    cause=DisplayTransitionCause.IN_FLIGHT_COVERAGE_UNAVAILABLE,
                )
            return CompletedRunDisplayCoverage(
                intent=intent,
                missing_set_ids=(*display_set_ids, *non_run_requested_ids),
                unresolved_intent_set_ids=_intent_ordered_union(display_set_ids, non_run_requested_ids),
                failed_intent_set_ids=failed_intent_set_ids,
                cause=DisplayTransitionCause.IN_FLIGHT_COVERAGE_UNAVAILABLE,
            )
        completion_entries: list[CompletionDisplayEntry] = []
        missing_set_ids: list[str] = []
        semantic_unavailable_set_ids: list[str] = []
        for set_id in display_set_ids:
            if str(set_id) in unavailable_set_ids:
                semantic_unavailable_set_ids.append(str(set_id))
                continue
            raw_entry = entries_raw.get(str(set_id))
            entry = self._coerce_completion_display_entry(
                raw_entry,
                set_id=str(set_id),
                label=str(intent.labels_by_set_id.get(str(set_id), str(set_id))),
            )
            if entry is None:
                missing_set_ids.append(str(set_id))
                continue
            if not tuple(str(name) for name in entry.owned_species if str(name)):
                semantic_unavailable_set_ids.append(str(set_id))
                continue
            completion_entries.append(entry)
        if completion_entries:
            completed_display_set_ids = tuple(str(entry.set_id) for entry in completion_entries if str(entry.set_id))
            missing_ids = tuple(missing_set_ids)
            semantic_ids = tuple(semantic_unavailable_set_ids)
            unresolved = _intent_ordered_union(
                failed_run_target_set_ids,
                missing_ids,
                semantic_ids,
                non_run_requested_ids,
            )
            if missing_ids and bool(ctx.get("active", False)):
                return CompletedRunDisplayCoverage(
                    intent=intent,
                    missing_set_ids=(*missing_ids, *non_run_requested_ids),
                    unavailable_set_ids=unresolved,
                    unresolved_intent_set_ids=unresolved,
                    failed_intent_set_ids=failed_intent_set_ids,
                    semantic_unavailable_set_ids=semantic_ids,
                    cause=DisplayTransitionCause.IN_FLIGHT_COVERAGE_UNAVAILABLE,
                )
            completed_members = set(completed_display_set_ids)
            display_primary_set_id = (
                str(intent.primary_set_id)
                if str(intent.primary_set_id or "") in completed_members
                else str(completed_display_set_ids[0])
            )
            return CompletedRunDisplayCoverage(
                intent=intent,
                transaction=CompletedRunDisplayTransaction(
                    intent=intent,
                    completion_entries=tuple(completion_entries),
                    display_set_ids=completed_display_set_ids,
                    display_primary_set_id=display_primary_set_id,
                    failed_set_ids=failed_run_target_set_ids,
                    unresolved_intent_set_ids=unresolved,
                    missing_intent_set_ids=(*missing_ids, *non_run_requested_ids),
                    failed_intent_set_ids=failed_intent_set_ids,
                    semantic_unavailable_set_ids=semantic_ids,
                ),
                missing_set_ids=(*missing_ids, *non_run_requested_ids),
                unavailable_set_ids=unresolved,
                unresolved_intent_set_ids=unresolved,
                failed_intent_set_ids=failed_intent_set_ids,
                semantic_unavailable_set_ids=semantic_ids,
            )
        if semantic_unavailable_set_ids:
            semantic_ids = tuple(semantic_unavailable_set_ids)
            missing_ids = tuple(missing_set_ids)
            unresolved = _intent_ordered_union(failed_run_target_set_ids, missing_ids, semantic_ids, non_run_requested_ids)
            return CompletedRunDisplayCoverage(
                intent=intent,
                missing_set_ids=(*missing_ids, *non_run_requested_ids),
                unavailable_set_ids=unresolved,
                unresolved_intent_set_ids=unresolved,
                failed_intent_set_ids=failed_intent_set_ids,
                semantic_unavailable_set_ids=semantic_ids,
                cause=DisplayTransitionCause.SEMANTIC_METADATA_UNAVAILABLE,
            )
        if missing_set_ids:
            unresolved = _intent_ordered_union(
                failed_run_target_set_ids,
                tuple(missing_set_ids),
                non_run_requested_ids,
            )
            return CompletedRunDisplayCoverage(
                intent=intent,
                missing_set_ids=(*tuple(missing_set_ids), *non_run_requested_ids),
                unresolved_intent_set_ids=unresolved,
                failed_intent_set_ids=failed_intent_set_ids,
                cause=DisplayTransitionCause.IN_FLIGHT_COVERAGE_UNAVAILABLE,
            )
        return CompletedRunDisplayCoverage(
            intent=intent,
            transaction=CompletedRunDisplayTransaction(
                intent=intent,
                completion_entries=tuple(completion_entries),
                display_set_ids=display_set_ids,
                display_primary_set_id=display_primary_set_id,
                failed_set_ids=failed_run_target_set_ids,
                unresolved_intent_set_ids=non_run_requested_ids,
                missing_intent_set_ids=non_run_requested_ids,
                failed_intent_set_ids=failed_intent_set_ids,
                semantic_unavailable_set_ids=(),
            ),
            missing_set_ids=non_run_requested_ids,
            unresolved_intent_set_ids=non_run_requested_ids,
            failed_intent_set_ids=failed_intent_set_ids,
        )

    @staticmethod
    def _coerce_completion_display_entry(
        value: Any,
        *,
        set_id: str,
        label: str,
    ) -> CompletionDisplayEntry | None:
        if isinstance(value, CompletionDisplayEntry):
            return CompletionDisplayEntry(
                set_id=str(value.set_id or set_id),
                label=str(value.label or label),
                t=value.t,
                series=deepcopy(dict(value.series or {})),
                algebra_scalars=deepcopy(dict(value.algebra_scalars or {})),
                solver_provenance=deepcopy(dict(value.solver_provenance or {})),
                mechanism_text=str(value.mechanism_text or ""),
                solver_config=deepcopy(dict(value.solver_config or {})),
                warnings=tuple(deepcopy(dict(warning)) for warning in value.warnings if isinstance(warning, Mapping)),
                completion_provenance=deepcopy(dict(value.completion_provenance or {})),
                owned_species=tuple(str(name) for name in value.owned_species if str(name)),
            )
        return None

    def deactivate_if_active(
        self,
        context: Mapping[str, Any] | None = None,
    ) -> Dict[str, Any]:
        ctx = context if isinstance(context, Mapping) else self._context
        if not isinstance(ctx, Mapping):
            return self._current_context()
        if isinstance(context, Mapping) and not self._matches_current_identity(context):
            return self._current_context()
        if not self._coerce_bool(ctx.get("active")):
            return deepcopy(dict(ctx))
        self._context = dict(ctx)
        return self.deactivate()

    def record_cache_key(self, cache_key: str) -> Dict[str, Any]:
        return self._update(cache_key=str(cache_key))

    def mark_runtime_waiting(self, *, required_lanes: int | None = None) -> Dict[str, Any]:
        context = dict(self._context)
        context["runtime_waiting"] = True
        context["active"] = False
        if required_lanes is None:
            context.pop("runtime_waiting_required_lanes", None)
        else:
            context["runtime_waiting_required_lanes"] = max(1, int(required_lanes))
        self._context = context
        return self._current_context()

    def clear_runtime_waiting(self) -> Dict[str, Any]:
        context = dict(self._context)
        context.pop("runtime_waiting", None)
        context.pop("runtime_waiting_required_lanes", None)
        self._context = context
        return self._current_context()

    def completion_policy_context(
        self,
        context: Mapping[str, Any] | None = None,
    ) -> CompletionPolicyContext | None:
        ctx = context if isinstance(context, Mapping) else self._context
        if not isinstance(ctx, Mapping):
            return None
        return CompletionPolicyContext(
            active=ctx.get("active"),
            request_id=ctx.get("request_id"),
            run_id=ctx.get("run_id"),
            fast_mode=ctx.get("fast_mode"),
            parallel=ctx.get("parallel"),
            keep_lane_pool_alive=ctx.get("keep_lane_pool_alive"),
            queue_ids=ctx.get("queue_ids"),
            queue_names=ctx.get("queue_names"),
            total=ctx.get("total"),
            pos=ctx.get("pos"),
            primary_set_id=ctx.get("primary_set_id"),
            completed_set_ids=ctx.get("completed_set_ids"),
            pending_workspace_reset_set_ids=ctx.get("pending_workspace_reset_set_ids"),
            pending_dirty_reset_generation_by_set_id=ctx.get("pending_dirty_reset_generation_by_set_id"),
            pending_init_seed=ctx.get("pending_init_seed"),
            pending_init_rewrite=ctx.get("pending_init_rewrite"),
            pending_init_applied=ctx.get("pending_init_applied", False),
            explicit_cache_preview_token=ctx.get("explicit_cache_preview_token"),
            explicit_cache_preview_scope_set_ids=ctx.get("explicit_cache_preview_scope_set_ids"),
            explicit_cache_valid_set_ids=ctx.get("explicit_cache_valid_set_ids"),
            explicit_cache_invalidated_set_ids=ctx.get("explicit_cache_invalidated_set_ids"),
            explicit_cache_truth_generation=ctx.get("explicit_cache_truth_generation"),
            preview_scope_set_ids=ctx.get("preview_scope_set_ids"),
            preview_owner_epoch=ctx.get("preview_owner_epoch"),
        )

    def completion_publication_policy_context(
        self,
        *,
        callback_context: Mapping[str, Any] | None,
        policy_context: CompletionPolicyContext | None,
    ) -> CompletionPolicyContext | None:
        context = policy_context
        if context is None and isinstance(callback_context, Mapping):
            context = self.completion_policy_context(callback_context)
        if (
            context is not None
            and isinstance(callback_context, Mapping)
            and isinstance(self._context, Mapping)
            and self.context_matches_current_run_identity(callback_context)
        ):
            current_context = self.completion_policy_context()
            if current_context is not None and cache_truth_generation_value(
                current_context.explicit_cache_truth_generation
            ) > cache_truth_generation_value(context.explicit_cache_truth_generation):
                context = context.evolve(
                    explicit_cache_preview_token=current_context.explicit_cache_preview_token,
                    explicit_cache_preview_scope_set_ids=current_context.explicit_cache_preview_scope_set_ids,
                    explicit_cache_valid_set_ids=current_context.explicit_cache_valid_set_ids,
                    explicit_cache_invalidated_set_ids=current_context.explicit_cache_invalidated_set_ids,
                    explicit_cache_truth_generation=current_context.explicit_cache_truth_generation,
                )
        return context

    def callback_context_with_cache_truth(
        self,
        callback_context: Mapping[str, Any],
        cache_truth_context: CompletionPolicyContext,
    ) -> BatchCallbackContext:
        base = self.callback_context_snapshot(callback_context).to_context()
        base.update(
            {
                "explicit_cache_preview_token": cache_truth_context.explicit_cache_preview_token,
                "explicit_cache_preview_scope_set_ids": cache_truth_context.explicit_cache_preview_scope_set_ids,
                "explicit_cache_valid_set_ids": cache_truth_context.explicit_cache_valid_set_ids,
                "explicit_cache_invalidated_set_ids": cache_truth_context.explicit_cache_invalidated_set_ids,
                    "explicit_cache_truth_generation": cache_truth_context.explicit_cache_truth_generation
                    if cache_truth_context.explicit_cache_truth_generation is not None
                    else next_cache_truth_generation(base.get("explicit_cache_truth_generation")),
            }
        )
        return self.callback_context_snapshot(base)

    def serialize_completion_policy_context(
        self,
        context: CompletionPolicyContext,
        *,
        base_context: Mapping[str, Any] | None = None,
    ) -> Dict[str, Any]:
        if (
            base_context is not None
            and isinstance(self._context, Mapping)
            and self._matches_current_identity(base_context)
        ):
            raw = dict(self._context)
        else:
            raw = dict(base_context or self._context or {})
        raw["active"] = bool(context.active)
        raw["request_id"] = context.request_id
        raw["run_id"] = context.run_id
        raw["fast_mode"] = bool(context.fast_mode)
        raw["parallel"] = bool(context.parallel)
        raw["keep_lane_pool_alive"] = bool(context.keep_lane_pool_alive)
        raw["queue_ids"] = list(context.queue_ids)
        raw["queue_names"] = list(context.queue_names)
        raw["total"] = int(context.total)
        raw["pos"] = int(context.pos)
        raw["primary_set_id"] = context.primary_set_id
        raw["completed_set_ids"] = list(context.completed_set_ids)
        raw["pending_workspace_reset_set_ids"] = list(context.pending_workspace_reset_set_ids)
        raw["pending_dirty_reset_generation_by_set_id"] = dict(context.pending_dirty_reset_generation_by_set_id)
        raw["pending_init_seed"] = {
            str(set_name): {str(species): float(value) for species, value in dict(seed).items()}
            for set_name, seed in context.pending_init_seed.items()
        }
        raw["pending_init_rewrite"] = context.pending_init_rewrite
        raw["pending_init_applied"] = bool(context.pending_init_applied)
        raw["explicit_cache_preview_token"] = context.explicit_cache_preview_token
        raw["explicit_cache_preview_scope_set_ids"] = context.explicit_cache_preview_scope_set_ids
        raw["explicit_cache_valid_set_ids"] = context.explicit_cache_valid_set_ids
        raw["explicit_cache_invalidated_set_ids"] = context.explicit_cache_invalidated_set_ids
        raw["explicit_cache_truth_generation"] = context.explicit_cache_truth_generation
        raw["preview_scope_set_ids"] = context.preview_scope_set_ids
        raw["preview_owner_epoch"] = context.preview_owner_epoch
        if base_context is None or self._matches_current_identity(raw):
            self._context = raw
            return self._current_context()
        return deepcopy(raw)

    def completion_summary(
        self,
        context: Mapping[str, Any] | None = None,
    ) -> BatchCompletionSummary:
        ctx = context if isinstance(context, Mapping) else self._context
        completed_ids = {
            str(set_id) for set_id in (ctx.get("completed_set_ids") or ()) if str(set_id)
        }
        stale_ids = {
            str(set_id) for set_id in (ctx.get("stale_runtime_input_set_ids") or ()) if str(set_id)
        }
        failed_set_ids = tuple(
            str(set_id) for set_id in (ctx.get("failed_set_ids") or ()) if str(set_id)
        )
        raw_failed_errors = ctx.get("failed_set_errors")
        failed_errors = raw_failed_errors if isinstance(raw_failed_errors, Mapping) else {}
        return BatchCompletionSummary(
            fast_mode=bool(ctx.get("fast_mode")),
            has_truthful_success=bool(completed_ids - stale_ids),
            failed_set_ids=failed_set_ids,
            failed_errors=failed_errors,
        )

    def runtime_input_stale_for_set(
        self,
        context: Mapping[str, Any],
        *,
        batch_set_id: str | None,
        current_global_epoch: int,
        current_set_epoch: int,
        current_epoch: int,
    ) -> bool:
        set_id = str(batch_set_id or "").strip()
        context_has_global_epoch = "runtime_input_global_epoch" in context
        try:
            context_global_epoch = int(context.get("runtime_input_global_epoch", 0) or 0)
        except Exception:
            context_global_epoch = 0
        if context_has_global_epoch and context_global_epoch != int(current_global_epoch):
            return True
        if set_id:
            raw_set_epochs = context.get("runtime_input_set_epoch_by_set_id")
            if isinstance(raw_set_epochs, Mapping):
                try:
                    context_set_epoch = int(raw_set_epochs.get(set_id, 0) or 0)
                except Exception:
                    context_set_epoch = 0
                return context_set_epoch != int(current_set_epoch)
            if context_has_global_epoch:
                return False
        if context.get("runtime_input_epoch") is not None:
            try:
                return int(context.get("runtime_input_epoch") or 0) != int(current_epoch)
            except Exception:
                return True
        return False

    def active_runtime_input_stale_for_set(
        self,
        *,
        batch_set_id: str | None,
        current_global_epoch: int,
        current_set_epoch: int,
        current_epoch: int,
    ) -> bool:
        context = self._context
        if not self._coerce_bool(context.get("active")):
            return False
        return self.runtime_input_stale_for_set(
            context,
            batch_set_id=batch_set_id,
            current_global_epoch=int(current_global_epoch),
            current_set_epoch=int(current_set_epoch),
            current_epoch=int(current_epoch),
        )

    def record_parallel_success(self, *, set_id: str | None, total: int) -> BatchContextTransition:
        context = dict(self._context)
        completed_ids = {
            str(item) for item in (context.get("completed_set_ids") or ()) if str(item)
        }
        if set_id:
            completed_ids.add(str(set_id))
        context["completed_set_ids"] = sorted(completed_ids)
        completed_count = len(completed_ids)
        batch_done = bool(completed_count >= max(1, int(total or 1)))
        if batch_done:
            context["active"] = False
        self._context = context
        return BatchContextTransition(
            context=self._current_context(),
            completed_count=completed_count,
            batch_done=batch_done,
        )

    def record_serial_success(
        self,
        *,
        set_id: str | None,
        shutdown_requested: bool = False,
    ) -> BatchContextTransition:
        context = dict(self._context)
        queue_ids = [str(item) for item in (context.get("queue_ids") or ()) if str(item)]
        try:
            pos = int(context.get("pos", 0) or 0)
        except Exception:
            pos = 0
        if 0 <= pos < len(queue_ids):
            expected = str(queue_ids[pos])
            if set_id is None or str(set_id) == expected:
                pos += 1
        context["pos"] = pos
        batch_done = bool(shutdown_requested or pos >= len(queue_ids))
        if batch_done:
            context["active"] = False
        self._context = context
        return BatchContextTransition(
            context=self._current_context(),
            completed_count=pos,
            batch_done=batch_done,
        )

    def record_scoped_failure(
        self,
        *,
        set_id: str,
        failure: Mapping[str, Any],
    ) -> BatchContextTransition:
        sid = str(set_id or "")
        context = dict(self._context)
        if not sid:
            self._context = context
            return BatchContextTransition(
                context=self._current_context(),
                completed_count=len(context.get("completed_set_ids") or ()),
                batch_done=False,
            )
        completed_ids = {
            str(item) for item in (context.get("completed_set_ids") or ()) if str(item)
        }
        failed_set_ids = {
            str(item) for item in (context.get("failed_set_ids") or ()) if str(item)
        }
        failed_errors = dict(context.get("failed_set_errors") or {})
        completed_ids.add(sid)
        failed_set_ids.add(sid)
        failed_errors[sid] = dict(failure)

        pending_reset_ids = [
            str(item)
            for item in (context.get("pending_workspace_reset_set_ids") or ())
            if str(item) and str(item) != sid
        ]
        pending_reset_generations = dict(context.get("pending_dirty_reset_generation_by_set_id") or {})
        pending_reset_generations.pop(sid, None)
        self._record_failure_cache_state(context, sid)

        context["completed_set_ids"] = sorted(completed_ids)
        context["failed_set_ids"] = sorted(failed_set_ids)
        context["failed_set_errors"] = failed_errors
        context["pending_workspace_reset_set_ids"] = pending_reset_ids
        context["pending_dirty_reset_generation_by_set_id"] = pending_reset_generations
        self._context = context
        return BatchContextTransition(
            context=self._current_context(),
            completed_count=len(completed_ids),
            batch_done=False,
        )

    def mark_stale_runtime_input_set_consumed(
        self,
        *,
        set_id: str,
        next_pos: int | None = None,
    ) -> Dict[str, Any]:
        sid = str(set_id or "").strip()
        context = dict(self._context)
        if not sid:
            self._context = context
            return self._current_context()
        completed_ids = {
            str(item) for item in (context.get("completed_set_ids") or ()) if str(item)
        }
        completed_ids.add(sid)
        context["completed_set_ids"] = sorted(completed_ids)
        stale_ids = {
            str(item)
            for item in (context.get("stale_runtime_input_set_ids") or ())
            if str(item)
        }
        stale_ids.add(sid)
        context["stale_runtime_input_set_ids"] = sorted(stale_ids)
        if next_pos is not None:
            context["pos"] = int(next_pos)
        self._remove_pending_dirty_reset_set_ids(context, (sid,))
        self._context = context
        return self._current_context()

    def record_active_serial_runtime_input_superseded(self, *, active_set_id: str) -> Dict[str, Any]:
        context = dict(self._context)
        sid = str(active_set_id or "").strip()
        if not sid:
            self._context = context
            return self._current_context()
        queue_ids = [str(item) for item in (context.get("queue_ids") or ()) if str(item)]
        try:
            pos = int(context.get("pos", 0) or 0)
        except Exception:
            pos = 0
        if 0 <= pos < len(queue_ids) and str(queue_ids[pos]) == sid:
            next_pos = pos + 1
        else:
            try:
                next_pos = queue_ids.index(sid) + 1
            except ValueError:
                next_pos = pos
        self._context = context
        return self.mark_stale_runtime_input_set_consumed(
            set_id=sid,
            next_pos=next_pos,
        )

    def consume_stale_serial_queue_prefix(
        self,
        *,
        is_stale_set: Callable[[str], bool],
    ) -> BatchContextTransition:
        context = dict(self._context)
        rows = list(context.get("rows") or [])
        queue_ids = [str(item) for item in (context.get("queue_ids") or [])]
        try:
            pos = int(context.get("pos", 0) or 0)
        except Exception:
            pos = 0
        while pos < len(queue_ids) and pos < len(rows) and bool(is_stale_set(str(queue_ids[pos]))):
            consumed_set_id = str(queue_ids[pos])
            pos += 1
            self._context = context
            context = self.mark_stale_runtime_input_set_consumed(
                set_id=consumed_set_id,
                next_pos=pos,
            )
        batch_done = bool(pos >= len(queue_ids) or pos >= len(rows))
        return BatchContextTransition(
            context=context,
            completed_count=pos,
            batch_done=batch_done,
        )

    def consume_stale_serial_queue_prefix_for_current_epochs(
        self,
        *,
        current_global_epoch: int,
        current_set_epoch_by_set_id: Mapping[str, Any],
        current_epoch: int,
    ) -> BatchContextTransition:
        context = self._context

        def is_stale_set(set_id: str) -> bool:
            sid = str(set_id or "").strip()
            try:
                current_set_epoch = int(dict(current_set_epoch_by_set_id or {}).get(sid, 0) or 0)
            except Exception:
                current_set_epoch = 0
            return self.runtime_input_stale_for_set(
                context,
                batch_set_id=sid,
                current_global_epoch=int(current_global_epoch),
                current_set_epoch=current_set_epoch,
                current_epoch=int(current_epoch),
            )

        return self.consume_stale_serial_queue_prefix(is_stale_set=is_stale_set)

    def record_parallel_stale_callback_consumed(self, *, set_id: str) -> BatchContextTransition:
        context = self.mark_stale_runtime_input_set_consumed(set_id=set_id)
        completed_count = len({
            str(item) for item in (context.get("completed_set_ids") or ()) if str(item)
        })
        try:
            total = int(context.get("total") or 0)
        except Exception:
            total = 0
        if total <= 0:
            total = len([str(item) for item in (context.get("queue_ids") or ()) if str(item)])
        batch_done = bool(completed_count >= max(1, int(total or 1)))
        if batch_done:
            context = self.deactivate()
        return BatchContextTransition(
            context=context,
            completed_count=completed_count,
            batch_done=batch_done,
        )

    def record_parallel_stale_callback_consumed_if_active(
        self,
        *,
        set_id: str,
    ) -> BatchContextTransition | None:
        context = dict(self._context)
        sid = str(set_id or "").strip()
        if not sid or not context.get("active") or not bool(context.get("parallel")):
            self._context = context
            return None
        return self.record_parallel_stale_callback_consumed(set_id=sid)

    @staticmethod
    def _record_failure_cache_state(context: Dict[str, Any], failed_set_id: str) -> None:
        sid = str(failed_set_id or "")
        if not sid:
            return
        valid_raw = context.get("explicit_cache_valid_set_ids")
        if valid_raw is None:
            return
        valid_set_ids = tuple(str(item) for item in (valid_raw or ()) if str(item) and str(item) != sid)
        invalidated_seen = {
            str(item)
            for item in (context.get("explicit_cache_invalidated_set_ids") or ())
            if str(item)
        }
        invalidated_seen.add(sid)
        queue_order = [str(item) for item in (context.get("queue_ids") or ()) if str(item)]
        queued_seen = set(queue_order)
        invalidated_set_ids = tuple(
            item for item in queue_order if item in invalidated_seen
        ) + tuple(sorted(item for item in invalidated_seen if item not in queued_seen))
        context["explicit_cache_valid_set_ids"] = valid_set_ids
        context["explicit_cache_invalidated_set_ids"] = invalidated_set_ids
        context["explicit_cache_truth_generation"] = next_cache_truth_generation(
            context.get("explicit_cache_truth_generation")
        )

    @staticmethod
    def _remove_pending_dirty_reset_set_ids(context: Dict[str, Any], set_ids: Sequence[str]) -> None:
        remove_ids = {str(set_id) for set_id in set_ids if str(set_id)}
        if not remove_ids:
            return
        context["pending_workspace_reset_set_ids"] = [
            str(item)
            for item in (context.get("pending_workspace_reset_set_ids") or ())
            if str(item) and str(item) not in remove_ids
        ]
        pending_generations = dict(context.get("pending_dirty_reset_generation_by_set_id") or {})
        for set_id in remove_ids:
            pending_generations.pop(str(set_id), None)
        context["pending_dirty_reset_generation_by_set_id"] = pending_generations

    @staticmethod
    def _coerce_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            text = value.strip().lower()
            if text in {"false", "0", "no", "off", ""}:
                return False
            if text in {"true", "1", "yes", "on"}:
                return True
        return bool(value)

    @staticmethod
    def _optional_int(value: Any) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError, OverflowError):
            return None

    @classmethod
    def _int_value(cls, value: Any, *, default: int) -> int:
        normalized = cls._optional_int(value)
        return int(default) if normalized is None else int(normalized)

    @staticmethod
    def _float_value(value: Any, *, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError, OverflowError):
            return float(default)

    @classmethod
    def _int_tuple(cls, values: Any) -> tuple[int, ...]:
        if values is None:
            return ()
        if isinstance(values, Mapping):
            values = values.keys()
        elif isinstance(values, (str, bytes)):
            values = (values,)
        normalized: list[int] = []
        for value in values:
            parsed = cls._optional_int(value)
            if parsed is not None:
                normalized.append(int(parsed))
        return tuple(normalized)

    @staticmethod
    def _str_tuple(values: Any, *, dedupe: bool) -> tuple[str, ...]:
        if values is None:
            return ()
        if isinstance(values, Mapping):
            values = values.keys()
        elif isinstance(values, (str, bytes)):
            values = (values,)
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
