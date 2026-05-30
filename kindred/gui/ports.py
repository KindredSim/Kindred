from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol, Sequence, Set, Tuple

from kindred.core.mechanism_source import MechanismAuthoringSource


@dataclass(frozen=True, slots=True)
class RunAutoLockResult:
    success: bool
    affected_rows: tuple[int, ...] = ()

    def __bool__(self) -> bool:
        return bool(self.success)


class DisplayRefreshSource(Enum):
    INCIDENTAL_REFRESH = "incidental_refresh"
    EXPLICIT_SHOW_REQUEST = "explicit_show_request"
    PROGRAMMATIC_SHOW_REQUEST = "programmatic_show_request"
    SLIDER_REPLAY = "slider_replay"


class SliderReplayScopeKind(Enum):
    FUTURE_TARGET_MEMBERSHIP = "future_target_membership"
    CAPTURED_TRANSACTION = "captured_transaction"


class ActiveDisplayKind(Enum):
    COMPLETED_RUN = "completed_run"
    CACHED_RESULT = "cached_result"
    RESOLVED_RESULT = "resolved_result"
    WORKSPACE_PREVIEW = "workspace_preview"
    FRESH_PREVIEW = "fresh_preview"
    DIRECT_SINGLE_RESULT = "direct_single_result"


class DisplaySetRole(Enum):
    PRIMARY_RESULT = "primary_result"
    RESULT_OVERLAY = "result_overlay"
    REFERENCE_OVERLAY = "reference_overlay"


class PlotLayerKind(Enum):
    PRIMARY_SERIES = "primary_series"
    RESULT_SERIES = "result_series"
    REFERENCE_SERIES = "reference_series"


@dataclass(frozen=True, slots=True)
class PlotDisplayLayer:
    layer_id: str
    source_id: str
    label: str
    kind: PlotLayerKind
    x: Any
    y: Mapping[str, Any]
    y_series: tuple[str, ...]
    visible: bool = True
    style_metadata: Mapping[str, Any] = field(default_factory=dict)
    source_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "layer_id", str(self.layer_id or "").strip())
        object.__setattr__(self, "source_id", str(self.source_id or "").strip())
        object.__setattr__(self, "label", str(self.label or "").strip())
        object.__setattr__(self, "kind", _normalized_plot_kind(self.kind))
        object.__setattr__(self, "x", _readonly_display_value(self.x))
        object.__setattr__(self, "y", _immutable_display_mapping(self.y))
        object.__setattr__(self, "y_series", _normalized_str_tuple(self.y_series))
        object.__setattr__(self, "visible", bool(self.visible))
        object.__setattr__(self, "style_metadata", _immutable_display_mapping(self.style_metadata))
        object.__setattr__(self, "source_metadata", _immutable_display_mapping(self.source_metadata))


@dataclass(frozen=True, slots=True)
class PlotDisplayLayersPayload:
    transaction_id: str
    primary_layer_id: str
    layers: tuple[PlotDisplayLayer, ...]
    intervention_annotations: tuple[Mapping[str, Any], ...] = ()
    show_intervention_annotations: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "transaction_id", str(self.transaction_id or "").strip())
        object.__setattr__(self, "primary_layer_id", str(self.primary_layer_id or "").strip())
        object.__setattr__(
            self,
            "layers",
            tuple(layer for layer in (self.layers or ()) if isinstance(layer, PlotDisplayLayer)),
        )
        object.__setattr__(
            self,
            "intervention_annotations",
            _immutable_display_mapping_tuple(self.intervention_annotations),
        )
        object.__setattr__(self, "show_intervention_annotations", bool(self.show_intervention_annotations))


@dataclass(frozen=True, slots=True)
class PlotCsvExportColumn:
    header: str
    values: Any


@dataclass(frozen=True, slots=True)
class CopyAllDisplayBlock:
    set_id: str
    label: str
    t: Any
    series: Dict[str, Any]
    layer_id: str = ""
    owned_species: tuple[str, ...] = ()
    display_species: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CopyAllMissingItem:
    set_id: str
    label: str
    popup_label: str
    reason: str


@dataclass(frozen=True, slots=True)
class CopyAllExportPlan:
    display_blocks: List[CopyAllDisplayBlock]
    missing_items: List[CopyAllMissingItem]


class DisplayEventKind(Enum):
    CACHE_DISPLAY_SCOPE_READY = "cache_display_scope_ready"
    RESOLVED_DISPLAY_REQUEST_READY = "resolved_display_request_ready"
    WORKSPACE_PREVIEW_READY = "workspace_preview_ready"
    COMPLETED_RUN_COVERAGE_READY = "completed_run_coverage_ready"
    COMPLETED_RUN_COVERAGE_UNAVAILABLE = "completed_run_coverage_unavailable"
    FRESH_PREVIEW_READY = "fresh_preview_ready"
    DIRECT_RESULT_READY = "direct_result_ready"
    SHOW_SCOPE_CHANGED = "show_scope_changed"
    DISPLAY_FAILURE = "display_failure"
    DISPLAY_CLEARED = "display_cleared"


class DisplayTransitionOutcomeKind(Enum):
    PUBLISHED = "published"
    CLEARED = "cleared"
    DENIED = "denied"
    FAILED = "failed"
    DEFERRED = "deferred"


class DisplayTransitionCause(Enum):
    CACHE_DISPLAY_SCOPE_READY = "cache_display_scope_ready"
    RESOLVED_DISPLAY_REQUEST_READY = "resolved_display_request_ready"
    WORKSPACE_PREVIEW_READY = "workspace_preview_ready"
    COMPLETED_RUN_COVERAGE_READY = "completed_run_coverage_ready"
    SEMANTIC_METADATA_UNAVAILABLE = "semantic_metadata_unavailable"
    NO_DISPLAYABLE_COMPLETION_RESULTS = "no_displayable_completion_results"
    NO_DISPLAYABLE_PREVIEW_RESULTS = "no_displayable_preview_results"
    IN_FLIGHT_COVERAGE_UNAVAILABLE = "in_flight_coverage_unavailable"
    FRESH_PREVIEW_READY = "fresh_preview_ready"
    DIRECT_RESULT_READY = "direct_result_ready"
    SHOW_REMOVED_ACTIVE_SET = "show_removed_active_set"
    AFFECTED_SCOPE_INTERSECTS_ACTIVE_DISPLAY = "affected_scope_intersects_active_display"
    DELETED_ACTIVE_SET = "deleted_active_set"
    DISPLAY_MUTATION_DENIED = "display_mutation_denied"
    DISPLAY_MUTATION_FAILED = "display_mutation_failed"
    QUEUED_DISPLAY = "queued_display"
    CACHE_RESULT_UNAVAILABLE = "cache_result_unavailable"
    INVALID_CACHE_ENTRY = "invalid_cache_entry"
    INVALID_DISPLAY_OUTCOME = "invalid_display_outcome"
    MANUAL_CLEAR = "manual_clear"


class DisplayStatus(Enum):
    DISPLAYED_COMPLETED_RUN = "displayed_completed_run"
    DISPLAYED_CACHED_RESULT = "displayed_cached_result"
    DISPLAYED_RESOLVED_RESULT = "displayed_resolved_result"
    DISPLAYED_WORKSPACE_PREVIEW = "displayed_workspace_preview"
    DISPLAYED_FRESH_PREVIEW = "displayed_fresh_preview"
    DISPLAYED_DIRECT_RESULT = "displayed_direct_result"
    DISPLAY_CLEARED = "display_cleared"
    DISPLAY_FAILED = "display_failed"
    DISPLAY_DENIED = "display_denied"
    DISPLAY_DEFERRED = "display_deferred"
    NO_COMPLETE_DISPLAYABLE_REQUEST_SCOPE = "no_complete_displayable_request_scope"


@dataclass(frozen=True, slots=True)
class ConcentrationSetInteractionTransaction:
    gesture: str
    row: Optional[int]
    set_id: str
    focus_change: bool
    selection_change: bool
    requested_show_set_ids: tuple[str, ...]
    explicit_slider_target_set_ids: tuple[str, ...]
    effective_slider_edit_target_set_ids: tuple[str, ...]
    effective_slider_edit_target_rows: tuple[int, ...]
    run_selected_rows: tuple[int, ...]
    empty_run_target_reason: str
    display_refresh_needed: bool
    display_refresh_reason: str
    slider_rebuild_needed: bool
    slider_rebuild_reason: str
    runtime_readiness_refresh_needed: bool
    runtime_readiness_refresh_reason: str


def _normalized_str_tuple(values: Sequence[str] | object) -> tuple[str, ...]:
    if isinstance(values, str):
        values = (values,)
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values or ():
        item = str(value or "").strip()
        if not item or item in seen:
            continue
        seen.add(item)
        normalized.append(item)
    return tuple(normalized)


def _normalized_plot_kind(value: PlotLayerKind | str | object) -> PlotLayerKind:
    if isinstance(value, PlotLayerKind):
        return value
    raw = str(value or "").strip()
    for candidate in PlotLayerKind:
        if raw == candidate.value or raw == candidate.name:
            return candidate
    raise ValueError(f"Unknown plot layer kind: {value!r}")


def _readonly_display_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _immutable_display_mapping(value)
    if isinstance(value, list):
        return tuple(_readonly_display_value(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_readonly_display_value(item) for item in value)
    copier = getattr(value, "copy", None)
    if callable(copier):
        try:
            copied = copier()
        except Exception:
            return value
        setflags = getattr(copied, "setflags", None)
        if callable(setflags):
            try:
                setflags(write=False)
            except Exception:
                pass
        return copied
    return value


def _immutable_display_mapping(values: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return MappingProxyType(
        {
            str(key): _readonly_display_value(value)
            for key, value in dict(values or {}).items()
            if str(key)
        }
    )


def _immutable_display_mapping_tuple(values: Sequence[Mapping[str, Any]] | object) -> tuple[Mapping[str, Any], ...]:
    if isinstance(values, Mapping):
        values = (values,)
    return tuple(
        _immutable_display_mapping(item)
        for item in (values or ())
        if isinstance(item, Mapping)
    )


@dataclass(frozen=True, slots=True)
class DisplayRequestScopeSnapshot:
    requested_show_set_ids: tuple[str, ...] = ()
    requested_labels_by_set_id: Mapping[str, str] = field(default_factory=dict)
    focused_set_id: str = ""
    current_row_set_id: str = ""
    row_selection_set_ids: tuple[str, ...] = ()
    explicit_slider_target_set_ids: tuple[str, ...] = ()
    effective_slider_target_set_ids: tuple[str, ...] = ()
    run_target_set_ids: tuple[str, ...] = ()
    cache_key: str = ""
    run_id: int | None = None
    request_id: int | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "requested_show_set_ids",
            "row_selection_set_ids",
            "explicit_slider_target_set_ids",
            "effective_slider_target_set_ids",
            "run_target_set_ids",
        ):
            object.__setattr__(
                self,
                field_name,
                _normalized_str_tuple(getattr(self, field_name)),
            )
        object.__setattr__(self, "focused_set_id", str(self.focused_set_id or "").strip())
        object.__setattr__(self, "current_row_set_id", str(self.current_row_set_id or "").strip())
        object.__setattr__(self, "cache_key", str(self.cache_key or "").strip())
        object.__setattr__(
            self,
            "requested_labels_by_set_id",
            MappingProxyType(
                {
                    str(set_id): str(label)
                    for set_id, label in dict(self.requested_labels_by_set_id or {}).items()
                    if str(set_id)
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class DisplaySetMetadata:
    set_id: str
    label: str
    role: DisplaySetRole
    t: Any
    series: Mapping[str, Any]
    owned_species: tuple[str, ...]
    display_species: tuple[str, ...]
    layer_id: str
    provenance: Mapping[str, Any] | None = None
    completion_provenance: Mapping[str, Any] | None = None
    workspace_preview_provenance: Mapping[str, Any] | None = None
    visible: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "set_id", str(self.set_id or "").strip())
        object.__setattr__(self, "label", str(self.label or "").strip())
        object.__setattr__(self, "layer_id", str(self.layer_id or "").strip())
        object.__setattr__(self, "t", _readonly_display_value(self.t))
        object.__setattr__(self, "series", _immutable_display_mapping(self.series))
        object.__setattr__(
            self,
            "provenance",
            _immutable_display_mapping(self.provenance)
            if isinstance(self.provenance, Mapping)
            else None,
        )
        object.__setattr__(
            self,
            "completion_provenance",
            (
                _immutable_display_mapping(self.completion_provenance)
                if isinstance(self.completion_provenance, Mapping)
                else None
            ),
        )
        object.__setattr__(
            self,
            "workspace_preview_provenance",
            (
                _immutable_display_mapping(self.workspace_preview_provenance)
                if isinstance(self.workspace_preview_provenance, Mapping)
                else None
            ),
        )
        object.__setattr__(self, "owned_species", _normalized_str_tuple(self.owned_species))
        object.__setattr__(self, "display_species", _normalized_str_tuple(self.display_species))
        object.__setattr__(self, "visible", bool(self.visible))


@dataclass(frozen=True, slots=True)
class ActiveDisplayTransaction:
    transaction_id: str
    kind: ActiveDisplayKind
    display_set_ids: tuple[str, ...]
    primary_display_set_id: str
    sets: Mapping[str, DisplaySetMetadata]
    status: DisplayStatus
    intervention_annotations: tuple[Mapping[str, Any], ...] = ()
    show_intervention_annotations: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "transaction_id", str(self.transaction_id or "").strip())
        object.__setattr__(self, "display_set_ids", _normalized_str_tuple(self.display_set_ids))
        object.__setattr__(
            self,
            "primary_display_set_id",
            str(self.primary_display_set_id or "").strip(),
        )
        object.__setattr__(
            self,
            "sets",
            MappingProxyType({
                str(set_id): metadata
                for set_id, metadata in dict(self.sets or {}).items()
                if str(set_id) and isinstance(metadata, DisplaySetMetadata)
            }),
        )
        object.__setattr__(
            self,
            "intervention_annotations",
            _immutable_display_mapping_tuple(self.intervention_annotations),
        )
        object.__setattr__(
            self,
            "show_intervention_annotations",
            bool(self.show_intervention_annotations),
        )


@dataclass(frozen=True, slots=True)
class DisplayTransitionOutcome:
    kind: DisplayTransitionOutcomeKind
    active_transaction: ActiveDisplayTransaction | None
    previous_transaction: ActiveDisplayTransaction | None
    display_status: DisplayStatus
    request_scope: DisplayRequestScopeSnapshot = field(default_factory=DisplayRequestScopeSnapshot)
    requested_show_set_ids: tuple[str, ...] = ()
    requested_labels_by_set_id: Mapping[str, str] = field(default_factory=dict)
    display_set_ids: tuple[str, ...] = ()
    attempted_display_set_ids: tuple[str, ...] = ()
    affected_set_ids: tuple[str, ...] = ()
    unresolved_intent_set_ids: tuple[str, ...] = ()
    missing_intent_set_ids: tuple[str, ...] = ()
    failed_intent_set_ids: tuple[str, ...] = ()
    semantic_unavailable_set_ids: tuple[str, ...] = ()
    event_kind: DisplayEventKind | None = None
    cause: DisplayTransitionCause | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "request_scope",
            self.request_scope
            if isinstance(self.request_scope, DisplayRequestScopeSnapshot)
            else DisplayRequestScopeSnapshot(),
        )
        object.__setattr__(
            self,
            "requested_show_set_ids",
            _normalized_str_tuple(self.requested_show_set_ids),
        )
        object.__setattr__(
            self,
            "requested_labels_by_set_id",
            MappingProxyType(
                {
                    str(key).strip(): str(value)
                    for key, value in dict(self.requested_labels_by_set_id or {}).items()
                    if str(key).strip()
                }
            ),
        )
        object.__setattr__(self, "display_set_ids", _normalized_str_tuple(self.display_set_ids))
        object.__setattr__(
            self,
            "attempted_display_set_ids",
            _normalized_str_tuple(self.attempted_display_set_ids),
        )
        object.__setattr__(self, "affected_set_ids", _normalized_str_tuple(self.affected_set_ids))
        object.__setattr__(
            self,
            "unresolved_intent_set_ids",
            _normalized_str_tuple(self.unresolved_intent_set_ids),
        )
        object.__setattr__(
            self,
            "missing_intent_set_ids",
            _normalized_str_tuple(self.missing_intent_set_ids),
        )
        object.__setattr__(
            self,
            "failed_intent_set_ids",
            _normalized_str_tuple(self.failed_intent_set_ids),
        )
        object.__setattr__(
            self,
            "semantic_unavailable_set_ids",
            _normalized_str_tuple(self.semantic_unavailable_set_ids),
        )

    def __bool__(self) -> bool:
        raise TypeError("Use DisplayTransitionOutcome.kind or display_status explicitly")


@dataclass(frozen=True, slots=True)
class SimulationCompletionDisplayOutcome:
    transition_outcome: DisplayTransitionOutcome | None = None

    def __bool__(self) -> bool:
        raise TypeError("Use SimulationCompletionDisplayOutcome.transition_outcome explicitly")


@dataclass(frozen=True, slots=True)
class CachedBatchDisplayScopeOutcome:
    transition_outcome: DisplayTransitionOutcome | None = None

    def __bool__(self) -> bool:
        raise TypeError("Use CachedBatchDisplayScopeOutcome.transition_outcome explicitly")


@dataclass(frozen=True, slots=True)
class BatchDisplayRefreshOutcome:
    focused_controls_use_workspace: Optional[bool] = None
    transition_outcome: DisplayTransitionOutcome | None = None


@dataclass(frozen=True, slots=True)
class ResolvedBatchDisplayRequestEntry:
    set_id: str
    label: str
    entry: Mapping[str, Any]
    canonical_entry: Mapping[str, Any] | None = None
    workspace_preview_provenance: Dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class CompletedRunDisplayIntent:
    requested_show_set_ids: tuple[str, ...]
    labels_by_set_id: Mapping[str, str]
    primary_set_id: str
    cache_key: str
    run_id: int | None = None
    request_id: int | None = None
    owned_species_by_set_id: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    run_target_set_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        requested_show_set_ids = tuple(str(set_id) for set_id in self.requested_show_set_ids if str(set_id))
        object.__setattr__(self, "requested_show_set_ids", requested_show_set_ids)
        object.__setattr__(
            self,
            "labels_by_set_id",
            {
                str(set_id): str(label)
                for set_id, label in dict(self.labels_by_set_id or {}).items()
                if str(set_id)
            },
        )
        object.__setattr__(self, "primary_set_id", str(self.primary_set_id or ""))
        object.__setattr__(self, "cache_key", str(self.cache_key or ""))
        object.__setattr__(
            self,
            "owned_species_by_set_id",
            {
                str(set_id): tuple(str(name) for name in (names or ()) if str(name))
                for set_id, names in dict(self.owned_species_by_set_id or {}).items()
                if str(set_id)
            },
        )
        object.__setattr__(
            self,
            "run_target_set_ids",
            _normalized_str_tuple(self.run_target_set_ids),
        )


@dataclass(frozen=True, slots=True)
class CompletionDisplayEntry:
    set_id: str
    label: str
    t: Any
    series: Mapping[str, Any]
    algebra_scalars: Mapping[str, object]
    solver_provenance: Mapping[str, Any] | None
    mechanism_text: str
    solver_config: Mapping[str, object]
    warnings: tuple[Mapping[str, Any], ...]
    completion_provenance: Mapping[str, Any] | None
    owned_species: tuple[str, ...]
    display_species: tuple[str, ...]

    def to_display_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "t": self.t,
            "series": dict(self.series or {}),
            "algebra_scalars": dict(self.algebra_scalars or {}),
            "solver_provenance": dict(self.solver_provenance or {}),
            "mechanism_text": str(self.mechanism_text or ""),
            "solver_config": dict(self.solver_config or {}),
            "warnings": [dict(warning) for warning in self.warnings if isinstance(warning, Mapping)],
        }
        if isinstance(self.completion_provenance, Mapping):
            payload["completion_provenance"] = dict(self.completion_provenance)
        if self.owned_species:
            payload["owned_species"] = tuple(str(name) for name in self.owned_species if str(name))
        if self.display_species:
            payload["display_species"] = tuple(str(name) for name in self.display_species if str(name))
        return payload


@dataclass(frozen=True, slots=True)
class CompletedRunDisplayTransaction:
    intent: CompletedRunDisplayIntent
    completion_entries: tuple[CompletionDisplayEntry, ...]
    display_set_ids: tuple[str, ...]
    display_primary_set_id: str
    failed_set_ids: tuple[str, ...]
    unresolved_intent_set_ids: tuple[str, ...] = ()
    missing_intent_set_ids: tuple[str, ...] = ()
    failed_intent_set_ids: tuple[str, ...] = ()
    semantic_unavailable_set_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CompletedRunDisplayCoverage:
    intent: CompletedRunDisplayIntent | None = None
    transaction: CompletedRunDisplayTransaction | None = None
    missing_set_ids: tuple[str, ...] = ()
    unavailable_set_ids: tuple[str, ...] = ()
    unresolved_intent_set_ids: tuple[str, ...] = ()
    failed_intent_set_ids: tuple[str, ...] = ()
    semantic_unavailable_set_ids: tuple[str, ...] = ()
    cause: DisplayTransitionCause | None = None


@dataclass(frozen=True, slots=True)
class FreshPreviewDisplayEntry:
    set_id: str
    label: str
    t: Any
    series: Mapping[str, Any]
    algebra_scalars: Mapping[str, object]
    solver_provenance: Mapping[str, Any] | None
    completion_provenance: Mapping[str, Any] | None
    owned_species: tuple[str, ...]
    display_species: tuple[str, ...]
    workspace_preview_provenance: Mapping[str, Any] | None = None

    def to_display_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "t": self.t,
            "series": dict(self.series or {}),
            "algebra_scalars": dict(self.algebra_scalars or {}),
            "solver_provenance": dict(self.solver_provenance or {}),
        }
        if isinstance(self.completion_provenance, Mapping):
            payload["completion_provenance"] = dict(self.completion_provenance)
        if self.owned_species:
            payload["owned_species"] = tuple(str(name) for name in self.owned_species if str(name))
        if self.display_species:
            payload["display_species"] = tuple(str(name) for name in self.display_species if str(name))
        return payload


@dataclass(frozen=True, slots=True)
class FreshPreviewDisplayTransaction:
    entries: tuple[FreshPreviewDisplayEntry, ...]
    display_set_ids: tuple[str, ...]
    target_set_ids: tuple[str, ...]
    display_primary_set_id: str
    cache_key: str
    display_source: DisplayRefreshSource
    requested_show_set_ids: tuple[str, ...] = ()
    requested_labels_by_set_id: Mapping[str, str] = field(default_factory=dict)
    request_id: int | None = None
    run_id: int | None = None


class BatchDisplayRequestCoverage(Enum):
    INCOMPLETE = "incomplete"
    FULL = "full"


@dataclass(frozen=True, slots=True)
class BatchDisplayRequestResolution:
    resolved_entries: tuple[ResolvedBatchDisplayRequestEntry, ...] = ()
    unavailable_cause: DisplayTransitionCause | None = None
    coverage: BatchDisplayRequestCoverage = BatchDisplayRequestCoverage.INCOMPLETE
    has_workspace_display_request: bool = False
    has_resolved_workspace_preview: bool = False
    focused_uses_workspace_controls: bool = False
    focused_has_resolved_entry: bool = False


def _normalize_slider_replay_target_set_ids(values: Sequence[str] | object) -> tuple[str, ...]:
    normalized: list[str] = []
    seen: set[str] = set()
    if isinstance(values, str):
        values = (values,)
    for value in values or ():
        set_id = str(value or "").strip()
        if not set_id or set_id in seen:
            continue
        seen.add(set_id)
        normalized.append(set_id)
    return tuple(normalized)


def _normalize_slider_replay_scope_kind(value: SliderReplayScopeKind | str | object) -> SliderReplayScopeKind:
    if isinstance(value, SliderReplayScopeKind):
        return value
    raw = str(value or "").strip()
    for candidate in SliderReplayScopeKind:
        if raw == candidate.value or raw == candidate.name:
            return candidate
    return SliderReplayScopeKind.FUTURE_TARGET_MEMBERSHIP


@dataclass(frozen=True, slots=True)
class SliderReplayIntent:
    target_set_ids: tuple[str, ...] = ()
    source: str = ""
    scope_kind: SliderReplayScopeKind = SliderReplayScopeKind.FUTURE_TARGET_MEMBERSHIP

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "target_set_ids",
            _normalize_slider_replay_target_set_ids(self.target_set_ids),
        )
        object.__setattr__(self, "source", str(self.source or "").strip())
        object.__setattr__(
            self,
            "scope_kind",
            _normalize_slider_replay_scope_kind(self.scope_kind),
        )


class SliderPreviewLifecyclePort(Protocol):
    def submit_slider_preview_replay_intent(
        self,
        intent: SliderReplayIntent,
        *,
        preserve_existing_request: bool = False,
    ) -> None: ...

    def clear_pending_slider_preview_replay(self, *, clear_plot_updates: bool = True) -> None: ...

    def invalidate_slider_preview_work(self) -> None: ...

    def deauthorize_completed_run_display_for_slider_preview_scope(
        self,
        target_set_ids: Sequence[str],
    ) -> bool: ...

    def launch_pending_slider_preview_replay(self) -> None: ...


@dataclass(frozen=True)
class SimulationCacheOpResult:
    ok: bool
    operation: str
    message: str = ""
    stats: Optional[Dict[str, Dict[str, int]]] = None
    cache_state_changed: bool = False


class SimulationDialogsPort(Protocol):
    def message_box_warning(self, title: str, message: str) -> None: ...

    def message_box_critical(self, title: str, message: str, *, details: Optional[str] = None) -> None: ...

    def message_box_question(self, title: str, message: str, *, accept_label: str = "Apply") -> bool: ...

    def choose_wegscheider_resolution(
        self,
        title: str,
        message: str,
        choices: Mapping[str, Sequence[Mapping[str, str]]],
    ) -> Optional[Dict[str, str]]: ...


class SimulationSettingsPort(Protocol):
    def settings_set_value(self, key: str, value: object) -> None: ...

    def settings_sync(self) -> None: ...


class SimulationCacheControlsPort(Protocol):
    def set_simulation_cache_caps(
        self,
        *,
        result_cap: int,
        preview_cap: int,
        persist: bool = True,
    ) -> SimulationCacheOpResult: ...

    def simulation_cache_stats(self) -> SimulationCacheOpResult: ...

    def purge_simulation_result_cache(self) -> SimulationCacheOpResult: ...

    def purge_simulation_preview_cache(self) -> SimulationCacheOpResult: ...

    def purge_simulation_all_caches(self) -> SimulationCacheOpResult: ...


class SimulationRunUiPort(Protocol):
    def run_button_is_enabled(self) -> bool: ...

    def set_run_button_enabled(self, enabled: bool) -> None: ...

    def set_runtime_backed_run_controls_ready(self, ready: bool) -> None: ...

    def schedule_runtime_availability_refresh(self) -> None: ...

    def set_stop_button_enabled(self, enabled: bool) -> None: ...

    def set_status_text(self, text: str) -> None: ...

    def set_sim_progress_value(self, value: int) -> None: ...

    def repaint_simulation_widgets(self) -> None: ...

    def set_algebra_status_text(self, text: str, *, details: str | None = None) -> None: ...


class SimulationSliderPort(Protocol):
    def stop_slider_release_commit_timer(self) -> None: ...

    def has_pending_slider_values(self) -> bool: ...

    def has_dirty_transaction(self) -> bool: ...

    def is_mechanism_valid_for_preview(self) -> bool: ...

    def show_preview_unavailable_for_dirty_state(self, message: str) -> None: ...

    def finalize_slider_release_commit(self) -> None: ...

    def stop_variable_update_timer(self) -> None: ...

    def stop_species_slider_update_timer(self) -> None: ...

    def set_slider_triggered_simulation(self, value: bool) -> None: ...

    def slider_triggered_simulation(self) -> bool: ...

    def last_slider_change_name(self) -> str: ...

    def slider_drag_active(self) -> bool: ...

    def suppress_slider_refresh(self) -> bool: ...

    def slider_gesture_target_set_ids_snapshot(self) -> List[str]: ...

    def preview_initials_for_row(self, row: int, baseline: Dict[str, float]) -> Dict[str, float]: ...

    def preview_batch_cache_token(self, rows: Sequence[int]) -> str: ...

    def has_dirty_state_for_set(self, set_id: str) -> bool: ...

    def dirty_state_generation(self, set_id: str) -> int: ...

    def reset_mechanism_workspaces(self, set_ids: Sequence[str]) -> bool: ...

    def discard_concentration_overlays_for_rows(self, rows: Sequence[int]) -> bool: ...

    def discard_concentration_overlays_for_set_ids(self, set_ids: Sequence[str]) -> bool: ...


class SimulationBatchPort(Protocol):
    def batch_rows_for_scope(self, scope: str) -> List[int]: ...

    def batch_set_ids_for_scope(self, scope: str) -> List[str]: ...

    def requested_show_batch_set_ids(self) -> List[str]: ...

    def slider_edit_target_set_ids(self) -> List[str]: ...

    def effective_slider_edit_target_set_ids(self, *, focused_row: Optional[int] = None) -> List[str]: ...

    def concentration_set_interaction_transaction(
        self,
        *,
        gesture: str,
        row: Optional[int] = None,
    ) -> ConcentrationSetInteractionTransaction: ...

    def run_selected_empty_target_reason(self) -> str: ...

    def focused_batch_set_id(self) -> Optional[str]: ...

    def batch_current_row(self) -> Optional[int]: ...

    def batch_set_id_for_row(self, row: int) -> Optional[str]: ...

    def batch_set_name_for_id(self, set_id: str) -> Optional[str]: ...

    def batch_set_id_for_name(self, name: str) -> Optional[str]: ...

    def batch_preferred_primary_set_id(self, rows: Sequence[int]) -> Optional[str]: ...

    def current_workspace_preview_identity_payload(self, *, set_id: str) -> Optional[Dict[str, Any]]: ...

    def clear_active_preview_cache_identity_state(self) -> None: ...

    def batch_cache_key(
        self,
        *,
        scope_identity: object | None = None,
        mechanism_text: str = "",
        solver_config: Optional[Dict[str, Any]] = None,
        t_end: float = 0.0,
    ) -> str: ...

    def batch_store_row_count(self) -> int: ...

    def batch_store_set_names(self) -> List[str]: ...

    def batch_store_visible_species(self) -> List[str]: ...

    def batch_model_validate_rows(self, rows: Sequence[int]) -> Set[Tuple[int, str]]: ...

    def batch_initials_for_row(self, row: int) -> Dict[str, float]: ...

    def update_batch_row_controls_state(self) -> None: ...

    def sync_batch_species_columns(
        self,
        species_names: Sequence[str],
        *,
        retain_active_cache_identity: bool = False,
    ) -> None: ...


class SimulationMechanismPort(Protocol):
    def auto_lock_for_run(self) -> RunAutoLockResult: ...

    def is_mechanism_ready_for_run(self) -> bool: ...

    def mechanism_reactions_text_raw(self) -> str: ...

    def mechanism_source_for_run(self, *, fast_mode: bool) -> MechanismAuthoringSource: ...

    def mechanism_source_for_run_set(
        self,
        source: MechanismAuthoringSource,
        *,
        set_id: Optional[str] = None,
        apply_parameter_overrides: bool = True,
        strip_initial_concentrations: bool = False,
    ) -> MechanismAuthoringSource: ...

    def pending_initials_for_run_source_set(
        self,
        source: MechanismAuthoringSource,
        *,
        set_name: str,
    ) -> Dict[str, float]: ...

    def mechanism_slider_points_value(self) -> Optional[int]: ...

    def mechanism_slider_solver_value(self) -> Optional[str]: ...

    def set_variable_sliders(
        self,
        variables: Dict[str, float],
        *,
        metadata: Optional[Dict[str, Dict[str, object]]] = None,
    ) -> None: ...

    def variable_slider_values(self) -> Dict[str, float]: ...

    def variable_metadata(self) -> Dict[str, Dict[str, object]]: ...

    def clear_variable_sliders(self) -> None: ...

    def has_slider_overrides(self) -> bool: ...

    def simulation_schema_id(self, *, fast_mode: bool = False) -> str: ...

    def simulation_param_fingerprint(self, set_id: Optional[str] = None, *, fast_mode: bool = False) -> str: ...

    def slider_overrides(self, set_id: Optional[str] = None) -> Dict[str, float]: ...

    def get_mechanism_text(self) -> str: ...

    def apply_wegscheider_resolution_reactions_rewrite(self, reactions_text: str) -> None: ...


class SimulationSolverPort(Protocol):
    def initial_solver_name(self) -> Optional[str]: ...

    def initial_rtol(self) -> Optional[float]: ...

    def initial_atol(self) -> Optional[float]: ...

    def temperature_spinbox_value(self) -> float: ...

    def num_points_spinbox_value(self) -> int: ...

    def sim_time_spinbox_text(self) -> str: ...

    def parse_sim_time_seconds(self) -> float: ...

    def dsl_global_temperature_K(self, dsl_text: str) -> Optional[float]: ...

    def use_sparse_jacobian(self) -> bool: ...

    def wegscheider_cyclicity_enabled(self) -> bool: ...


class SimulationRuntimePort(Protocol):
    def prepare_slider_runtime(
        self,
        param_names: Optional[list[str]] = None,
        *,
        set_id: Optional[str] = None,
    ) -> Optional[object]: ...

    def apply_slider_overrides_to_bindings(self, runtime: object, *, set_id: Optional[str] = None) -> bool: ...

    def set_slider_runtime_dirty(self, value: bool) -> None: ...

    def is_energy_mode_mechanism(self, mechanism: object) -> bool: ...

    def dsl_has_computational_mode_generated_block(self, mechanism_text: str) -> bool: ...

    def sync_energy_mode_temperature_from_mechanism(self, mechanism: object) -> None: ...

    def populate_energy_mode_variables_from_mechanism(
        self,
        mechanism: object,
        *,
        refresh_sliders: bool,
        preserve_visibility: bool = False,
    ) -> None: ...

    def extract_and_populate_variables(self, *, preserve_visibility: bool = False) -> None: ...


class SimulationResultsPort(Protocol):
    def active_display_transaction(self) -> ActiveDisplayTransaction | None: ...

    def publish_deferred_display_request(
        self,
        *,
        affected_set_ids: Sequence[str] = (),
        requested_show_set_ids: Sequence[str] | None = None,
        requested_labels_by_set_id: Mapping[str, str] | None = None,
        unresolved_intent_set_ids: Sequence[str] = (),
        missing_intent_set_ids: Sequence[str] = (),
        failed_intent_set_ids: Sequence[str] = (),
        semantic_unavailable_set_ids: Sequence[str] = (),
    ) -> SimulationCompletionDisplayOutcome: ...

    def publish_direct_completion_result(
        self,
        *,
        t: Any,
        series: Dict[str, Any],
        batch_set: Optional[str],
        batch_set_id: Optional[str],
        algebra_scalars: Optional[Mapping[str, object]],
        direct_completion_provenance: Mapping[str, Any],
        owned_species: Sequence[str],
        display_species: Sequence[str],
        solver_provenance: Optional[Mapping[str, Any]] = None,
    ) -> SimulationCompletionDisplayOutcome: ...

    def publish_completed_run_display_transaction(
        self,
        transaction: CompletedRunDisplayTransaction,
    ) -> SimulationCompletionDisplayOutcome: ...

    def publish_completed_run_display_unavailable(
        self,
        *,
        cause: DisplayTransitionCause,
        affected_set_ids: Sequence[str],
        requested_show_set_ids: Sequence[str] | None = None,
        requested_labels_by_set_id: Mapping[str, str] | None = None,
        attempted_display_set_ids: Sequence[str] = (),
        unresolved_intent_set_ids: Sequence[str] = (),
        missing_intent_set_ids: Sequence[str] = (),
        failed_intent_set_ids: Sequence[str] = (),
        semantic_unavailable_set_ids: Sequence[str] = (),
    ) -> SimulationCompletionDisplayOutcome: ...

    def publish_fresh_preview_display(
        self,
        transaction: FreshPreviewDisplayTransaction,
    ) -> SimulationCompletionDisplayOutcome: ...

    def deauthorize_completed_run_display_for_runtime_input_preview(
        self,
        *,
        affected_set_ids: Sequence[str],
        affected_scope_is_global: bool,
    ) -> DisplayTransitionOutcome | None: ...

    def publish_cached_batch_display_scope(
        self,
        *,
        cache_key: str,
        requested_show_set_ids: Sequence[str],
        prefer_set: Optional[str] = None,
        display_source: Optional[DisplayRefreshSource] = None,
    ) -> "CachedBatchDisplayScopeOutcome": ...

    def refresh_display_from_request_scope(
        self,
        *,
        display_source: Optional[DisplayRefreshSource] = None,
    ) -> "BatchDisplayRefreshOutcome": ...


class SimulationProvenancePort(Protocol):
    def snapshot_datasets(self) -> Dict[str, Any]: ...

    def last_fit_metadata(self) -> Optional[Dict[str, Any]]: ...

    def set_last_simulation_provenance(self, provenance: Dict[str, Any]) -> None: ...

    def set_last_simulation_ctc(self, ctc: Dict[str, float]) -> None: ...

    def integrate_ctc(
        self,
        t: Any,
        y: Any,
        *,
        uniformity_eps: float,
        tail_strategy: str,
    ) -> Tuple[float, str, bool, float, str]: ...

    def publish_simulation_completion_provenance(
        self,
        *,
        mechanism_text: str,
        solver_method: str,
        solver_label: str,
        solver_warning: Optional[str],
        solver_config: Mapping[str, Any],
        temperature_K: float,
        temperature_source: str,
        energy_unit: Optional[str],
        energy_mode: bool,
        simulation_time: float | str,
        num_points_requested: int,
        species_names: Sequence[str],
        t: Any,
        series: Mapping[str, Any],
        algebra_scalars: Optional[Mapping[str, Any]] = None,
        dataset_overlays: Any = None,
        display_transaction: Optional[Mapping[str, Any]] = None,
        display_sets: Optional[Sequence[Mapping[str, Any]]] = None,
        solver_provenance: Optional[Mapping[str, Any]] = None,
        warnings: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> Dict[str, Any]: ...


class SimulationMechanismHelpersPort(Protocol):
    def last_mechanism(self) -> Optional[object]: ...

    def last_mechanism_context(self) -> Dict[str, Any]: ...

    def remember_last_mechanism(self, mechanism: object, mechanism_text: str, solver_config: Dict[str, Any]) -> None: ...

    def set_temperature_override_state(self, *, enabled: bool, tooltip: str) -> None: ...

    def set_temperature_mode_indicator_text(self, text: str) -> None: ...

    def update_temperature_mode_indicator(self) -> None: ...

    def authoritative_structure_snapshot(
        self,
        *,
        source: MechanismAuthoringSource,
        units_identity: Sequence[object] = (),
        builder: Callable[[str], object],
    ) -> object: ...

    def sync_mechanism_controls_to_focused_batch_set(self, *, use_workspace: bool = True) -> None: ...

@dataclass(frozen=True, slots=True)
class SimulationUiPorts:
    dialogs: SimulationDialogsPort
    settings: SimulationSettingsPort
    run_ui: SimulationRunUiPort
    slider: SimulationSliderPort
    batch: SimulationBatchPort
    mechanism: SimulationMechanismPort
    solver: SimulationSolverPort
    runtime: SimulationRuntimePort
    results: SimulationResultsPort
    provenance: SimulationProvenancePort
    mechanism_helpers: SimulationMechanismHelpersPort
