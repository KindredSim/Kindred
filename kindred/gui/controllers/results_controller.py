from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence

import numpy as np
from PySide6 import QtCore, QtWidgets

from kindred.core.batch_cache_contracts import (
    BatchCacheResultReadSnapshot,
    BatchCacheEntryReadResult,
    BatchCacheEntryV1,
)
from kindred.gui.controllers.results_display_builders import (
    active_transaction_for_display_commit,
    deduped_set_ids,
    display_overlay_entry,
    owned_species_for_display_entry,
    transaction_overlay_is_reference,
)
from kindred.gui.controllers.results_display_projections import (
    build_copy_all_export_plan,
    build_main_plot_csv_export,
    cache_resolution_cause_for_transition,
    display_status_for_unpublished_request,
    display_status_is_displayed,
    display_transaction_provenance_payload,
    display_transition_status_text,
    ordered_display_transaction_metadata,
    published_display_status_text,
)
from kindred.gui.ports import (
    ActiveDisplayKind,
    ActiveDisplayTransaction,
    BatchDisplayRefreshOutcome,
    BatchDisplayRequestCoverage,
    BatchDisplayRequestResolution,
    CachedBatchDisplayScopeOutcome,
    CompletionDisplayEntry,
    CompletedRunDisplayTransaction,
    DisplayEventKind,
    DisplayRefreshSource,
    DisplayRequestScopeSnapshot,
    DisplaySetMetadata,
    DisplaySetRole,
    DisplayStatus,
    DisplayTransitionCause,
    DisplayTransitionOutcome,
    DisplayTransitionOutcomeKind,
    FreshPreviewDisplayEntry,
    FreshPreviewDisplayTransaction,
    ResolvedBatchDisplayRequestEntry,
    SimulationCompletionDisplayOutcome,
)

logger = logging.getLogger(__name__)

__all__ = ["ResultsController"]


class ResultsDisplayPlotPort(Protocol):
    def set_scalar_values(self, scalars: Dict[str, object]) -> None: ...
    def set_statistics_results(self, results: Dict[str, object], *, prefer: str) -> None: ...
    def stats_table(self) -> object: ...
    def overlay_snapshot(self) -> Dict[str, object]: ...
    def set_simulation_popup_labels(self, *, primary_set_id: str, popup_labels_by_set_id: Mapping[str, str]) -> None: ...
    def clear_display_transaction_state(self) -> None: ...
    def transaction_export_axis_state(self, scope: str) -> Dict[str, object]: ...
    def append_dataset_overlay_export_columns(
        self,
        header: Sequence[str],
        rows: Sequence[Sequence[object]],
        scope: str,
    ) -> tuple[List[str], List[List[object]]]: ...
    def intervention_annotation_state(self) -> Dict[str, object]: ...
    def set_intervention_annotations_from_provenance(self, provenance: Mapping[str, object] | None) -> None: ...
    def reference_layers_visible(self) -> bool: ...


def _coerce_display_refresh_source(source: object | None) -> DisplayRefreshSource:
    if isinstance(source, DisplayRefreshSource):
        return source
    raw = str(source or "").strip()
    if raw:
        for candidate in DisplayRefreshSource:
            if raw == candidate.value or raw == candidate.name:
                return candidate
    return DisplayRefreshSource.INCIDENTAL_REFRESH


@dataclass(frozen=True)
class ResultsControllerPort:
    parent: QtCore.QObject
    main_plot: Callable[[], ResultsDisplayPlotPort]
    batch_name_for_id: Callable[[str], str | None]
    batch_id_for_name: Callable[[str], str | None]
    batch_set_ids_for_scope: Callable[[str], list[str]]
    requested_show_batch_set_ids: Callable[[], list[str]]
    explicit_slider_target_set_ids: Callable[[], Sequence[str]]
    effective_slider_target_set_ids: Callable[[], Sequence[str]]
    focused_batch_set_id: Callable[[], str | None]
    current_batch_row: Callable[[], int | None]
    batch_set_id_for_row: Callable[[int], str | None]
    batch_row_for_set_id: Callable[[str], int | None]
    active_batch_cache_key: Callable[[], str]
    active_result_cache_read_snapshot: Callable[..., BatchCacheResultReadSnapshot]
    clear_active_preview_cache_identity_state: Callable[[], None]
    set_last_simulation_provenance: Callable[[Dict[str, Any]], None]
    set_last_simulation_ctc: Callable[[Dict[str, float]], None]
    publish_simulation_completion_provenance: Callable[..., Dict[str, Any]]
    update_display_transaction_provenance: Callable[..., Dict[str, Any]]
    set_main_plot_scalar_values: Callable[[dict[str, object]], None]
    update_main_plot_statistics: Callable[..., None]
    main_plot_stats_table: Callable[[], object]
    publish_main_plot_results_table: Callable[[object], None]
    set_main_plot_data: Callable[..., None]
    show_simulation_tab: Callable[[], None]
    refresh_simulation_plot_views: Callable[[], None]
    schedule_main_plot_refresh: Callable[[Sequence[int]], None]
    set_status_text: Callable[[str], None]
    update_batch_row_controls_state: Callable[[], None]
    focused_show_request_is_dirty: Callable[[Sequence[str], Optional[str]], bool]
    focused_batch_set_is_dirty: Callable[[], bool]
    show_request_uses_fresh_explicit_cache_after_post_run_sync: Callable[[Sequence[str]], bool]
    workspace_display_request_resolution: Callable[[Sequence[str]], "BatchDisplayRequestResolution"]
    current_workspace_preview_identity_payload: Callable[[str], Optional[Dict[str, Any]]]


@dataclass(frozen=True, slots=True)
class CachedBatchAvailability:
    available_ids: List[str]
    has_invalid_entry: bool = False


@dataclass(frozen=True, slots=True)
class CachedBatchDisplayScopeCoverage:
    requested_show_set_ids: List[str]
    available_ids: List[str]
    full_coverage: bool
    unavailable_cause: DisplayTransitionCause | None = None


@dataclass(frozen=True, slots=True)
class BatchDisplayRefreshRequest:
    requested_show_set_ids: tuple[str, ...] = ()
    prefer_set_id: Optional[str] = None
    active_cache_key: str = ""
    display_source: DisplayRefreshSource = DisplayRefreshSource.INCIDENTAL_REFRESH
    focused_dirty: bool = False
    focused_set_dirty: bool = False
    fresh_explicit_cache_after_post_run_sync: bool = False
    resolution: BatchDisplayRequestResolution = BatchDisplayRequestResolution()


@dataclass(frozen=True, slots=True)
class AuthoritativeResultDisplayTransitionOutcome:
    refresh_requested: bool = False


class CompletedRunDisplayConflictPolicy(Enum):
    DENY_PUBLICATION = "deny_publication"
    REPLACE_ACTIVE_DISPLAY = "replace_active_display"
    CLEAR_INTERSECTING_COMPLETED_RUN = "clear_intersecting_completed_run"


@dataclass(frozen=True, slots=True)
class DisplayPublicationTransition:
    active_kind: ActiveDisplayKind
    event_kind: DisplayEventKind
    cause: DisplayTransitionCause
    completed_run_conflict_policy: CompletedRunDisplayConflictPolicy = (
        CompletedRunDisplayConflictPolicy.DENY_PUBLICATION
    )


_DISPLAY_TRANSITION_CACHED_REFRESH = DisplayPublicationTransition(
    active_kind=ActiveDisplayKind.CACHED_RESULT,
    event_kind=DisplayEventKind.CACHE_DISPLAY_SCOPE_READY,
    cause=DisplayTransitionCause.CACHE_DISPLAY_SCOPE_READY,
)
_DISPLAY_TRANSITION_CACHED_REFRESH_REPLACE_ACTIVE = DisplayPublicationTransition(
    active_kind=ActiveDisplayKind.CACHED_RESULT,
    event_kind=DisplayEventKind.CACHE_DISPLAY_SCOPE_READY,
    cause=DisplayTransitionCause.CACHE_DISPLAY_SCOPE_READY,
    completed_run_conflict_policy=CompletedRunDisplayConflictPolicy.REPLACE_ACTIVE_DISPLAY,
)
_DISPLAY_TRANSITION_RESOLVED_REFRESH = DisplayPublicationTransition(
    active_kind=ActiveDisplayKind.RESOLVED_RESULT,
    event_kind=DisplayEventKind.RESOLVED_DISPLAY_REQUEST_READY,
    cause=DisplayTransitionCause.RESOLVED_DISPLAY_REQUEST_READY,
)
_DISPLAY_TRANSITION_RESOLVED_REFRESH_REPLACE_ACTIVE = DisplayPublicationTransition(
    active_kind=ActiveDisplayKind.RESOLVED_RESULT,
    event_kind=DisplayEventKind.RESOLVED_DISPLAY_REQUEST_READY,
    cause=DisplayTransitionCause.RESOLVED_DISPLAY_REQUEST_READY,
    completed_run_conflict_policy=CompletedRunDisplayConflictPolicy.REPLACE_ACTIVE_DISPLAY,
)
_DISPLAY_TRANSITION_WORKSPACE_PREVIEW = DisplayPublicationTransition(
    active_kind=ActiveDisplayKind.WORKSPACE_PREVIEW,
    event_kind=DisplayEventKind.WORKSPACE_PREVIEW_READY,
    cause=DisplayTransitionCause.WORKSPACE_PREVIEW_READY,
)
_DISPLAY_TRANSITION_WORKSPACE_PREVIEW_REPLACE_ACTIVE = DisplayPublicationTransition(
    active_kind=ActiveDisplayKind.WORKSPACE_PREVIEW,
    event_kind=DisplayEventKind.WORKSPACE_PREVIEW_READY,
    cause=DisplayTransitionCause.WORKSPACE_PREVIEW_READY,
    completed_run_conflict_policy=CompletedRunDisplayConflictPolicy.REPLACE_ACTIVE_DISPLAY,
)
_DISPLAY_TRANSITION_COMPLETED_RUN_FINAL = DisplayPublicationTransition(
    active_kind=ActiveDisplayKind.COMPLETED_RUN,
    event_kind=DisplayEventKind.COMPLETED_RUN_COVERAGE_READY,
    cause=DisplayTransitionCause.COMPLETED_RUN_COVERAGE_READY,
    completed_run_conflict_policy=CompletedRunDisplayConflictPolicy.REPLACE_ACTIVE_DISPLAY,
)
_DISPLAY_TRANSITION_DIRECT_RAW = DisplayPublicationTransition(
    active_kind=ActiveDisplayKind.DIRECT_SINGLE_RESULT,
    event_kind=DisplayEventKind.DIRECT_RESULT_READY,
    cause=DisplayTransitionCause.DIRECT_RESULT_READY,
    completed_run_conflict_policy=CompletedRunDisplayConflictPolicy.REPLACE_ACTIVE_DISPLAY,
)
_DISPLAY_TRANSITION_AUTHORITATIVE_INVALIDATION = DisplayPublicationTransition(
    active_kind=ActiveDisplayKind.CACHED_RESULT,
    event_kind=DisplayEventKind.DISPLAY_CLEARED,
    cause=DisplayTransitionCause.AFFECTED_SCOPE_INTERSECTS_ACTIVE_DISPLAY,
    completed_run_conflict_policy=CompletedRunDisplayConflictPolicy.CLEAR_INTERSECTING_COMPLETED_RUN,
)
_DISPLAY_TRANSITION_RUNTIME_INPUT_PREVIEW_DEAUTHORIZATION = DisplayPublicationTransition(
    active_kind=ActiveDisplayKind.CACHED_RESULT,
    event_kind=DisplayEventKind.DISPLAY_CLEARED,
    cause=DisplayTransitionCause.AFFECTED_SCOPE_INTERSECTS_ACTIVE_DISPLAY,
    completed_run_conflict_policy=CompletedRunDisplayConflictPolicy.CLEAR_INTERSECTING_COMPLETED_RUN,
)
_DISPLAY_TRANSITION_DELETED_SET_DEAUTHORIZATION = DisplayPublicationTransition(
    active_kind=ActiveDisplayKind.CACHED_RESULT,
    event_kind=DisplayEventKind.DISPLAY_CLEARED,
    cause=DisplayTransitionCause.DELETED_ACTIVE_SET,
    completed_run_conflict_policy=CompletedRunDisplayConflictPolicy.CLEAR_INTERSECTING_COMPLETED_RUN,
)
_DISPLAY_TRANSITION_DISPLAY_SCOPE_REMOVAL_DEAUTHORIZATION = DisplayPublicationTransition(
    active_kind=ActiveDisplayKind.CACHED_RESULT,
    event_kind=DisplayEventKind.SHOW_SCOPE_CHANGED,
    cause=DisplayTransitionCause.SHOW_REMOVED_ACTIVE_SET,
    completed_run_conflict_policy=CompletedRunDisplayConflictPolicy.CLEAR_INTERSECTING_COMPLETED_RUN,
)
_DISPLAY_TRANSITION_FRESH_PREVIEW = DisplayPublicationTransition(
    active_kind=ActiveDisplayKind.FRESH_PREVIEW,
    event_kind=DisplayEventKind.FRESH_PREVIEW_READY,
    cause=DisplayTransitionCause.FRESH_PREVIEW_READY,
)
_DISPLAY_TRANSITION_FRESH_PREVIEW_REPLACE_ACTIVE = DisplayPublicationTransition(
    active_kind=ActiveDisplayKind.FRESH_PREVIEW,
    event_kind=DisplayEventKind.FRESH_PREVIEW_READY,
    cause=DisplayTransitionCause.FRESH_PREVIEW_READY,
    completed_run_conflict_policy=CompletedRunDisplayConflictPolicy.REPLACE_ACTIVE_DISPLAY,
)


class ResultsController(QtCore.QObject):
    """
    Results + plot presentation controller.

    This keeps `MainWindow` focused on UI composition while preserving behavior
    behind a narrow results-specific UI port.
    """

    def __init__(self, ui: ResultsControllerPort):
        super().__init__(ui.parent)
        self._ui = ui
        self._active_display_transaction: ActiveDisplayTransaction | None = None
        self._last_display_transition_outcome: DisplayTransitionOutcome | None = None

    def active_display_transaction(self) -> ActiveDisplayTransaction | None:
        return self._active_display_transaction

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
    ) -> SimulationCompletionDisplayOutcome:
        return SimulationCompletionDisplayOutcome(
            transition_outcome=self._record_unpublished_display_request_outcome(
                affected_set_ids=affected_set_ids,
                requested_show_set_ids=requested_show_set_ids,
                requested_labels_by_set_id=requested_labels_by_set_id,
                unresolved_intent_set_ids=unresolved_intent_set_ids,
                missing_intent_set_ids=missing_intent_set_ids,
                failed_intent_set_ids=failed_intent_set_ids,
                semantic_unavailable_set_ids=semantic_unavailable_set_ids,
                cause=DisplayTransitionCause.QUEUED_DISPLAY,
                outcome_kind=DisplayTransitionOutcomeKind.DEFERRED,
                display_status=DisplayStatus.DISPLAY_DEFERRED,
            )
        )

    def update_reference_overlay_visibility(self, visible: bool) -> None:
        transaction = self._active_display_transaction
        if transaction is None:
            return
        updated_sets: Dict[str, DisplaySetMetadata] = {}
        changed = False
        for layer_id, metadata in dict(transaction.sets or {}).items():
            if metadata.role is DisplaySetRole.REFERENCE_OVERLAY:
                updated = replace(metadata, visible=bool(visible))
                changed = changed or metadata.visible != bool(visible)
                updated_sets[str(layer_id)] = updated
                continue
            updated_sets[str(layer_id)] = metadata
        if changed:
            updated_transaction = replace(transaction, sets=updated_sets)
            self._active_display_transaction = updated_transaction
            self._record_display_transition_outcome(
                outcome_kind=DisplayTransitionOutcomeKind.PUBLISHED,
                active_transaction=updated_transaction,
                previous_transaction=transaction,
                display_status=updated_transaction.status,
                display_set_ids=updated_transaction.display_set_ids,
                affected_set_ids=updated_transaction.display_set_ids,
                event_kind=DisplayEventKind.SHOW_SCOPE_CHANGED,
                cause=self._display_cause_for_active_kind(updated_transaction.kind),
            )
            provenance_payload = self._display_transaction_provenance_payload(updated_transaction)
            self._ui.update_display_transaction_provenance(**provenance_payload)

    @staticmethod
    def _deduped_set_ids(values: Sequence[str]) -> tuple[str, ...]:
        return deduped_set_ids(values)

    def _current_display_request_scope(
        self,
        *,
        requested_show_set_ids: Sequence[str] | None = None,
        requested_labels_by_set_id: Mapping[str, str] | None = None,
        run_target_set_ids: Sequence[str] = (),
        cache_key: str = "",
        run_id: int | None = None,
        request_id: int | None = None,
    ) -> DisplayRequestScopeSnapshot:
        focused_set_id = ""
        current_row_set_id = ""
        try:
            focused_set_id = str(self._ui.focused_batch_set_id() or "").strip()
        except Exception as exc:
            logger.debug("Failed to snapshot focused batch set id: %s", exc, exc_info=True)
        try:
            current_row = self._ui.current_batch_row()
            if current_row is not None:
                current_row_set_id = str(self._ui.batch_set_id_for_row(int(current_row)) or "").strip()
        except Exception as exc:
            logger.debug("Failed to snapshot current batch row set id: %s", exc, exc_info=True)
        if requested_show_set_ids is None:
            try:
                requested_show_ids = tuple(
                    str(set_id) for set_id in (self._ui.requested_show_batch_set_ids() or ()) if str(set_id)
                )
            except Exception as exc:
                logger.debug("Failed to snapshot requested Show set ids: %s", exc, exc_info=True)
                requested_show_ids = ()
        else:
            requested_show_ids = tuple(self._deduped_set_ids(requested_show_set_ids))
        if requested_labels_by_set_id is None:
            requested_labels = self._popup_labels_by_set_id(requested_show_ids)
        else:
            requested_labels = {
                str(set_id): str(label)
                for set_id, label in dict(requested_labels_by_set_id or {}).items()
                if str(set_id)
            }
        try:
            row_selection_set_ids = tuple(
                str(set_id)
                for set_id in (self._ui.batch_set_ids_for_scope("selected") or ())
                if str(set_id)
            )
        except Exception as exc:
            logger.debug("Failed to snapshot selected row set ids: %s", exc, exc_info=True)
            row_selection_set_ids = ()
        try:
            explicit_slider_targets = tuple(
                str(set_id)
                for set_id in (self._ui.explicit_slider_target_set_ids() or ())
                if str(set_id)
            )
        except Exception as exc:
            logger.debug("Failed to snapshot explicit slider target ids: %s", exc, exc_info=True)
            explicit_slider_targets = ()
        try:
            effective_slider_targets = tuple(
                str(set_id)
                for set_id in (self._ui.effective_slider_target_set_ids() or ())
                if str(set_id)
            )
        except Exception as exc:
            logger.debug("Failed to snapshot effective slider target ids: %s", exc, exc_info=True)
            effective_slider_targets = ()
        return DisplayRequestScopeSnapshot(
            requested_show_set_ids=requested_show_ids,
            requested_labels_by_set_id=requested_labels,
            focused_set_id=focused_set_id,
            current_row_set_id=current_row_set_id,
            row_selection_set_ids=row_selection_set_ids,
            explicit_slider_target_set_ids=explicit_slider_targets,
            effective_slider_target_set_ids=effective_slider_targets,
            run_target_set_ids=self._deduped_set_ids(run_target_set_ids),
            cache_key=str(cache_key or "").strip(),
            run_id=run_id,
            request_id=request_id,
        )

    @staticmethod
    def _active_display_kind_for_transition(transition: DisplayPublicationTransition) -> ActiveDisplayKind:
        return transition.active_kind

    @staticmethod
    def _display_event_kind_for_transition(transition: DisplayPublicationTransition) -> DisplayEventKind:
        return transition.event_kind

    @staticmethod
    def _display_status_for_kind(kind: ActiveDisplayKind) -> DisplayStatus:
        if kind is ActiveDisplayKind.COMPLETED_RUN:
            return DisplayStatus.DISPLAYED_COMPLETED_RUN
        if kind is ActiveDisplayKind.FRESH_PREVIEW:
            return DisplayStatus.DISPLAYED_FRESH_PREVIEW
        if kind is ActiveDisplayKind.DIRECT_SINGLE_RESULT:
            return DisplayStatus.DISPLAYED_DIRECT_RESULT
        if kind is ActiveDisplayKind.WORKSPACE_PREVIEW:
            return DisplayStatus.DISPLAYED_WORKSPACE_PREVIEW
        if kind is ActiveDisplayKind.RESOLVED_RESULT:
            return DisplayStatus.DISPLAYED_RESOLVED_RESULT
        return DisplayStatus.DISPLAYED_CACHED_RESULT

    @staticmethod
    def _display_cause_for_transition(transition: DisplayPublicationTransition) -> DisplayTransitionCause:
        return transition.cause

    @staticmethod
    def _display_cause_for_active_kind(kind: ActiveDisplayKind) -> DisplayTransitionCause:
        if kind is ActiveDisplayKind.COMPLETED_RUN:
            return DisplayTransitionCause.COMPLETED_RUN_COVERAGE_READY
        if kind is ActiveDisplayKind.CACHED_RESULT:
            return DisplayTransitionCause.CACHE_DISPLAY_SCOPE_READY
        if kind is ActiveDisplayKind.RESOLVED_RESULT:
            return DisplayTransitionCause.RESOLVED_DISPLAY_REQUEST_READY
        if kind is ActiveDisplayKind.WORKSPACE_PREVIEW:
            return DisplayTransitionCause.WORKSPACE_PREVIEW_READY
        if kind is ActiveDisplayKind.FRESH_PREVIEW:
            return DisplayTransitionCause.FRESH_PREVIEW_READY
        return DisplayTransitionCause.DIRECT_RESULT_READY

    @staticmethod
    def _transition_outcome(outcome: object) -> DisplayTransitionOutcome | None:
        if not isinstance(
            outcome,
            (
                BatchDisplayRefreshOutcome,
                CachedBatchDisplayScopeOutcome,
                SimulationCompletionDisplayOutcome,
            ),
        ):
            return None
        transition_outcome = outcome.transition_outcome
        return (
            transition_outcome
            if isinstance(transition_outcome, DisplayTransitionOutcome)
            else None
        )

    @classmethod
    def _outcome_published(cls, outcome: object) -> bool:
        transition_outcome = cls._transition_outcome(outcome)
        return (
            transition_outcome is not None
            and transition_outcome.kind is DisplayTransitionOutcomeKind.PUBLISHED
        )

    @classmethod
    def _outcome_denied_by_active_completed_run(cls, outcome: object) -> bool:
        transition_outcome = cls._transition_outcome(outcome)
        return (
            transition_outcome is not None
            and transition_outcome.kind is DisplayTransitionOutcomeKind.DENIED
            and transition_outcome.cause is DisplayTransitionCause.DISPLAY_MUTATION_DENIED
            and isinstance(transition_outcome.active_transaction, ActiveDisplayTransaction)
            and transition_outcome.active_transaction.kind is ActiveDisplayKind.COMPLETED_RUN
        )

    @staticmethod
    def _display_transition_status_text(
        transition_outcome: DisplayTransitionOutcome | None,
    ) -> str:
        return display_transition_status_text(transition_outcome)

    def _set_status_from_display_transition(
        self,
        transition_outcome: DisplayTransitionOutcome | None,
    ) -> None:
        self._ui.set_status_text(self._display_transition_status_text(transition_outcome))

    @staticmethod
    def _published_display_status_text(display_status: DisplayStatus) -> str:
        return published_display_status_text(display_status)

    @staticmethod
    def _display_status_is_displayed(display_status: DisplayStatus) -> bool:
        return display_status_is_displayed(display_status)

    @classmethod
    def _display_status_for_unpublished_request(
        cls,
        *,
        requested_status: DisplayStatus,
        active_transaction: ActiveDisplayTransaction | None,
    ) -> DisplayStatus:
        _ = cls
        return display_status_for_unpublished_request(
            requested_status=requested_status,
            active_transaction=active_transaction,
        )

    @staticmethod
    def _cache_resolution_cause_for_transition(
        transition_outcome: DisplayTransitionOutcome | None,
        *,
        default: DisplayTransitionCause = DisplayTransitionCause.CACHE_RESULT_UNAVAILABLE,
    ) -> DisplayTransitionCause:
        return cache_resolution_cause_for_transition(
            transition_outcome,
            default=default,
        )

    def _main_plot(self) -> ResultsDisplayPlotPort:
        return self._ui.main_plot()

    def _normalize_batch_set_id(self, token: str) -> Optional[str]:
        raw = str(token or "").strip()
        if not raw:
            return None
        if self._ui.batch_name_for_id(raw) is not None:
            return raw
        sid = self._ui.batch_id_for_name(raw)
        if sid:
            return sid
        return None

    def _cache_entry_for_set_id(
        self,
        *,
        set_id: str,
        snapshot: BatchCacheResultReadSnapshot,
        require_completion_provenance: bool = False,
    ) -> BatchCacheEntryReadResult:
        sid = str(set_id or "").strip()
        if not sid:
            return BatchCacheEntryReadResult("missing")
        direct = snapshot.entry_result_for_set(
            sid,
            require_completion_provenance=bool(require_completion_provenance),
        )
        if direct.entry is not None or direct.state == "invalid":
            return direct
        return BatchCacheEntryReadResult("missing")

    def _normalized_requested_show_batch_ids(self, requested_show_set_ids: Sequence[str]) -> List[str]:
        requested_show_ids: List[str] = []
        for token in [str(n) for n in (requested_show_set_ids or []) if str(n)]:
            sid = self._normalize_batch_set_id(token)
            if sid and sid not in requested_show_ids:
                requested_show_ids.append(str(sid))
        return requested_show_ids

    def _available_cached_batch_ids(
        self,
        *,
        requested_show_set_ids: Sequence[str],
        snapshot: BatchCacheResultReadSnapshot,
        require_completion_provenance: bool = True,
    ) -> CachedBatchAvailability:
        requested_show_ids = self._normalized_requested_show_batch_ids(requested_show_set_ids)
        allowed_ids = None
        invalidated_ids = {str(sid) for sid in (snapshot.invalidated_set_ids or ()) if str(sid)}
        has_invalid_entry = False
        if snapshot.valid_set_ids:
            allowed_ids = {str(sid) for sid in (snapshot.valid_set_ids or ()) if str(sid)}
        available: List[str] = []
        for sid in requested_show_ids:
            if allowed_ids is not None and sid not in allowed_ids:
                continue
            result = self._cache_entry_for_set_id(
                set_id=sid,
                snapshot=snapshot,
                require_completion_provenance=bool(require_completion_provenance),
            )
            if sid in invalidated_ids:
                has_invalid_entry = True
                continue
            if result.entry is not None:
                available.append(sid)
                continue
            if result.state == "invalid":
                has_invalid_entry = True
        return CachedBatchAvailability(available, has_invalid_entry=has_invalid_entry)

    @staticmethod
    def _missing_owned_species_set_ids(entries_by_set_id: Mapping[str, Mapping[str, Any]]) -> tuple[str, ...]:
        missing: list[str] = []
        for set_id, entry in entries_by_set_id.items():
            owned = owned_species_for_display_entry(entry)
            series = entry.get("series") if isinstance(entry, Mapping) else None
            available = {str(name) for name in dict(series or {}) if str(name)} if isinstance(series, Mapping) else set()
            if not owned or any(str(name) not in available for name in owned):
                missing.append(str(set_id))
        return tuple(missing)

    def _primary_cached_batch_id(
        self,
        *,
        available: Sequence[str],
        prefer_set: Optional[str],
        snapshot: BatchCacheResultReadSnapshot,
    ) -> str:
        if isinstance(prefer_set, str):
            prefer_id = self._normalize_batch_set_id(prefer_set)
            if (
                prefer_id
                and prefer_id in available
                and self._cache_entry_for_set_id(
                    set_id=prefer_id,
                    snapshot=snapshot,
                    require_completion_provenance=True,
                ).entry
                is not None
            ):
                return str(prefer_id)
        focused_id = str(self._ui.focused_batch_set_id() or "")
        if focused_id and focused_id in available:
            focused_entry = self._cache_entry_for_set_id(
                set_id=focused_id,
                snapshot=snapshot,
                require_completion_provenance=True,
            ).entry
            if focused_entry is not None:
                return focused_id
        return str(available[0])

    def _cached_batch_display_entries_by_set_id(
        self,
        *,
        available: Sequence[str],
        primary: str,
        primary_entry: Mapping[str, Any],
        snapshot: BatchCacheResultReadSnapshot,
        active_entries_by_set_id: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> dict[str, Mapping[str, Any]]:
        displayed_entries: dict[str, Mapping[str, Any]] = {str(primary): primary_entry}
        active_entries = {
            str(set_id): entry
            for set_id, entry in dict(active_entries_by_set_id or {}).items()
            if str(set_id) and isinstance(entry, Mapping)
        }
        for sid in available:
            if sid == primary:
                continue
            if str(sid) in active_entries:
                displayed_entries[str(sid)] = active_entries[str(sid)]
                continue
            other = self._cache_entry_for_set_id(
                set_id=sid,
                snapshot=snapshot,
                require_completion_provenance=True,
            ).entry
            if other is None:
                continue
            displayed_entries[str(sid)] = other
        return displayed_entries

    def _active_display_entries_by_requested_id(
        self,
        requested_show_set_ids: Sequence[str],
        *,
        excluded_set_ids: Sequence[str] = (),
    ) -> dict[str, Mapping[str, Any]]:
        transaction = self._active_display_transaction
        if transaction is None:
            return {}
        requested_ids = set(self._deduped_set_ids(requested_show_set_ids))
        excluded_ids = set(self._deduped_set_ids(excluded_set_ids))
        active_display_ids = set(self._deduped_set_ids(transaction.display_set_ids))
        entries: dict[str, Mapping[str, Any]] = {}
        for metadata in dict(transaction.sets or {}).values():
            if not isinstance(metadata, DisplaySetMetadata):
                continue
            set_id = str(metadata.set_id or "").strip()
            if (
                not set_id
                or set_id in excluded_ids
                or set_id not in requested_ids
                or set_id not in active_display_ids
            ):
                continue
            if metadata.role is DisplaySetRole.REFERENCE_OVERLAY:
                continue
            payload: Dict[str, object] = {
                "t": metadata.t,
                "series": dict(metadata.series or {}),
            }
            if metadata.owned_species:
                payload["owned_species"] = tuple(metadata.owned_species)
            if isinstance(metadata.completion_provenance, Mapping):
                payload["completion_provenance"] = dict(metadata.completion_provenance)
            entries[set_id] = payload
        return entries

    @staticmethod
    def _owned_species_for_display_entry(entry: Mapping[str, Any]) -> tuple[str, ...]:
        return owned_species_for_display_entry(entry)

    @staticmethod
    def _display_overlay_entry(
        *,
        label: str,
        entry: Mapping[str, Any],
        set_id: str,
        layer_kind: str,
        layer_id: str,
        owned_species: Sequence[str] | None = None,
        visible: bool | None = None,
    ) -> Dict[str, object]:
        return display_overlay_entry(
            label=label,
            entry=entry,
            set_id=set_id,
            layer_kind=layer_kind,
            layer_id=layer_id,
            owned_species=owned_species,
            visible=visible,
        )

    def _cached_batch_overlays(
        self,
        *,
        transaction_entries_by_set_id: Mapping[str, Mapping[str, Any]],
        primary: str,
    ) -> List[Dict[str, object]]:
        overlays: List[Dict[str, object]] = []
        for sid, entry in transaction_entries_by_set_id.items():
            if str(sid) == str(primary):
                continue
            overlay_label = self._ui.batch_name_for_id(str(sid)) or str(sid)
            overlays.append(
                self._display_overlay_entry(
                    label=overlay_label,
                    entry=entry,
                    set_id=str(sid),
                    layer_id=f"result:{sid}",
                    layer_kind="result",
                )
            )
        return overlays

    def _publish_main_plot_results_table(self, *, plot: ResultsDisplayPlotPort | None = None) -> None:
        table = plot.stats_table() if plot is not None else self._ui.main_plot_stats_table()
        self._ui.publish_main_plot_results_table(table)

    def _apply_cached_batch_plot_metadata(
        self,
        *,
        plot: ResultsDisplayPlotPort,
        cache_key: str,
        available: Sequence[str],
        transaction_entries_by_set_id: Mapping[str, Mapping[str, Any]],
        primary: str,
        primary_label: str,
        entry: BatchCacheEntryV1,
        t: np.ndarray,
        series: Dict[str, np.ndarray],
    ) -> Optional[str]:
        scalars = entry.get("algebra_scalars") or {}
        try:
            self._ui.set_main_plot_scalar_values(scalars)
        except Exception as exc:
            logger.exception(
                "Failed to set plot scalar values for cached batch display request (primary=%s): %s",
                primary_label,
                exc,
            )
            return "metadata_scalar_failed"

        stats_results_map: Dict[str, Dict[str, object]] = {}
        for sid in available:
            _ = cache_key
            payload = transaction_entries_by_set_id.get(str(sid))
            if payload is None:
                continue
            series_payload = payload.get("series") or {}
            if not series_payload:
                continue
            label = self._ui.batch_name_for_id(sid) or str(sid)
            layer_id = f"result:{sid}"
            stats_results_map[layer_id] = {
                "t": payload["t"],
                "series": dict(series_payload),
                "label": str(label),
                "layer_id": layer_id,
                "layer_kind": "result",
                "set_id": str(sid),
            }
        try:
            self._ui.update_main_plot_statistics(
                stats_results_map=stats_results_map,
                prefer=f"result:{primary}",
                t=np.asarray(t, dtype=float),
                series={str(k): np.asarray(v, dtype=float) for k, v in series.items()},
            )
        except Exception as exc:
            logger.exception(
                "Failed to update plot statistics for cached batch display request (primary=%s): %s",
                primary_label,
                exc,
            )
            return "metadata_statistics_failed"
        try:
            self._publish_main_plot_results_table()
        except Exception as exc:
            logger.exception("Failed to fetch stats table from plot after cached display request: %s", exc)
            return "metadata_table_failed"
        return None

    def _apply_resolved_batch_plot_metadata(
        self,
        *,
        plot: object,
        resolved_entries: Sequence[ResolvedBatchDisplayRequestEntry],
        primary: ResolvedBatchDisplayRequestEntry,
    ) -> Optional[str]:
        scalars = primary.entry.get("algebra_scalars") or {}
        try:
            self._ui.set_main_plot_scalar_values(scalars)
        except Exception as exc:
            logger.exception(
                "Failed to set plot scalar values for resolved batch display request (primary=%s): %s",
                primary.label,
                exc,
            )
            return "metadata_scalar_failed"

        stats_results_map: Dict[str, Dict[str, object]] = {}
        for resolved in resolved_entries:
            series_payload = resolved.entry.get("series") or {}
            if not series_payload:
                continue
            result_layer_id = f"result:{resolved.set_id}"
            stats_results_map[result_layer_id] = {
                "t": resolved.entry["t"],
                "series": dict(series_payload),
                "label": str(resolved.label),
                "layer_id": result_layer_id,
                "layer_kind": "result",
                "set_id": str(resolved.set_id),
            }
            canonical_entry = resolved.canonical_entry
            if canonical_entry is not None:
                canonical_series = canonical_entry.get("series") or {}
                if canonical_series:
                    reference_layer_id = f"reference:{resolved.set_id}"
                    stats_results_map[reference_layer_id] = {
                        "t": canonical_entry["t"],
                        "series": dict(canonical_series),
                        "label": f"{resolved.label} [ref]",
                        "layer_id": reference_layer_id,
                        "layer_kind": "reference",
                        "set_id": str(resolved.set_id),
                    }
        try:
            self._ui.update_main_plot_statistics(
                stats_results_map=stats_results_map,
                prefer=f"result:{primary.set_id}",
                t=np.asarray(primary.entry["t"], dtype=float),
                series={
                    str(k): np.asarray(v, dtype=float)
                    for k, v in (primary.entry.get("series") or {}).items()
                },
            )
        except Exception as exc:
            logger.exception(
                "Failed to update plot statistics for resolved batch display request (primary=%s): %s",
                primary.label,
                exc,
            )
            return "metadata_statistics_failed"
        try:
            self._publish_main_plot_results_table()
        except Exception as exc:
            logger.exception("Failed to fetch stats table from plot after resolved display request: %s", exc)
            return "metadata_table_failed"
        return None

    def _apply_completed_run_plot_metadata(
        self,
        *,
        plot: ResultsDisplayPlotPort,
        active_display_transaction: ActiveDisplayTransaction,
        completion_entries: Sequence[CompletionDisplayEntry],
        primary: CompletionDisplayEntry,
        semantic_series_by_set_id: Mapping[str, Mapping[str, object]],
    ) -> Optional[str]:
        scalars = primary.algebra_scalars or {}
        try:
            self._ui.set_main_plot_scalar_values(scalars)
        except Exception as exc:
            logger.exception(
                "Failed to set plot scalar values for completed run display (primary=%s): %s",
                primary.label,
                exc,
            )
            return "metadata_scalar_failed"

        _ = completion_entries
        _ = semantic_series_by_set_id
        stats_results_map: Dict[str, Dict[str, object]] = {}
        metadata_values = self._ordered_display_transaction_metadata(active_display_transaction)
        for metadata in metadata_values:
            if metadata.role is DisplaySetRole.REFERENCE_OVERLAY:
                continue
            layer_id = str(metadata.layer_id or f"result:{metadata.set_id}")
            stats_results_map[layer_id] = {
                "t": metadata.t,
                "series": dict(metadata.series or {}),
                "label": str(metadata.label or metadata.set_id),
                "layer_id": layer_id,
                "layer_kind": "result",
                "set_id": str(metadata.set_id),
            }
        try:
            self._ui.update_main_plot_statistics(
                stats_results_map=stats_results_map,
                prefer=f"result:{primary.set_id}",
                t=np.asarray(primary.t, dtype=float),
                series={
                    str(k): np.asarray(v, dtype=float)
                    for k, v in semantic_series_by_set_id.get(str(primary.set_id), {}).items()
                },
            )
        except Exception as exc:
            logger.exception(
                "Failed to update plot statistics for completed run display (primary=%s): %s",
                primary.label,
                exc,
            )
            return "metadata_statistics_failed"
        try:
            self._publish_main_plot_results_table()
        except Exception as exc:
            logger.exception("Failed to fetch stats table after completed run display: %s", exc)
            return "metadata_table_failed"
        return None

    def _apply_direct_completion_plot_metadata(
        self,
        *,
        plot: object,
        t: np.ndarray,
        series: Mapping[str, Any],
        display_label: str,
        algebra_scalars: Mapping[str, object] | None,
        layer_id: str,
        set_id: str,
    ) -> Optional[str]:
        try:
            self._ui.set_main_plot_scalar_values(dict(algebra_scalars or {}))
        except Exception as exc:
            logger.exception("Failed to set plot scalar values after simulation completion: %s", exc)
            return "metadata_scalar_failed"

        normalized_series = {str(k): np.asarray(v, dtype=float) for k, v in series.items()}
        layer_id_s = str(layer_id or "").strip() or "result:live"
        set_id_s = str(set_id or "").strip()
        try:
            self._ui.update_main_plot_statistics(
                stats_results_map={
                    layer_id_s: {
                        "t": t,
                        "series": normalized_series,
                        "label": str(display_label),
                        "layer_id": layer_id_s,
                        "layer_kind": "result",
                        "set_id": set_id_s,
                    }
                },
                prefer=layer_id_s,
                t=np.asarray(t, dtype=float),
                series=normalized_series,
            )
        except Exception as exc:
            logger.exception(
                "Failed to update plot statistics after simulation completion (label=%s): %s",
                display_label,
                exc,
            )
            return "metadata_statistics_failed"
        try:
            self._publish_main_plot_results_table(plot=plot)
        except Exception as exc:
            logger.exception("Failed to update results table after simulation completion: %s", exc)
            return "metadata_table_failed"
        return None

    def _publish_direct_completion_provenance(
        self,
        *,
        plot: object,
        direct_completion_provenance: Mapping[str, Any] | None,
        active_display_transaction: ActiveDisplayTransaction | None = None,
    ) -> Optional[str]:
        transaction_payload = self._display_transaction_provenance_payload(active_display_transaction)
        if not isinstance(direct_completion_provenance, Mapping):
            if not transaction_payload:
                self._clear_direct_completion_provenance()
                return None
            payload = transaction_payload
        else:
            payload = dict(direct_completion_provenance)
            payload.update(transaction_payload)
        if isinstance(active_display_transaction, ActiveDisplayTransaction):
            primary_metadata = next(
                (
                    metadata
                    for metadata in dict(active_display_transaction.sets or {}).values()
                    if metadata.role is DisplaySetRole.PRIMARY_RESULT
                    and str(metadata.set_id) == str(active_display_transaction.primary_display_set_id)
                ),
                None,
            )
            if isinstance(primary_metadata, DisplaySetMetadata):
                raw_display_series = {
                    str(name): np.asarray(values, dtype=float)
                    for name, values in dict(primary_metadata.series or {}).items()
                    if str(name)
                }
                display_species = [
                    str(name)
                    for name in (primary_metadata.display_species or tuple(raw_display_series))
                    if str(name) and str(name) in raw_display_series
                ] or [str(name) for name in raw_display_series]
                display_series = {
                    name: raw_display_series[name]
                    for name in display_species
                    if name in raw_display_series
                }
                payload["t"] = np.asarray(primary_metadata.t, dtype=float)
                payload["series"] = display_series
                payload["species_names"] = display_species
        if not payload:
            self._clear_direct_completion_provenance()
            return None
        try:
            payload["dataset_overlays"] = plot.overlay_snapshot()
            self._ui.publish_simulation_completion_provenance(**payload)
        except Exception as exc:
            logger.exception("Failed to publish direct completion display provenance: %s", exc)
            return "direct_provenance_failed"
        return None

    @staticmethod
    def _display_transaction_provenance_payload(
        transaction: ActiveDisplayTransaction | None,
    ) -> Dict[str, object]:
        return display_transaction_provenance_payload(transaction)

    def _clear_direct_completion_provenance(self) -> None:
        self._ui.set_last_simulation_provenance({})
        self._ui.set_last_simulation_ctc({})

    def _popup_labels_by_set_id(self, set_ids: Sequence[str]) -> Dict[str, str]:
        labels_by_id: Dict[str, str] = {}
        label_counts: Dict[str, int] = {}
        for raw_set_id in set_ids or ():
            set_id = str(raw_set_id or "").strip()
            if not set_id:
                continue
            label = str(self._ui.batch_name_for_id(set_id) or set_id)
            labels_by_id[set_id] = label
            label_counts[label] = int(label_counts.get(label, 0)) + 1

        popup_labels: Dict[str, str] = {}
        for set_id, label in labels_by_id.items():
            popup_label = str(label)
            if int(label_counts.get(label, 0)) > 1:
                try:
                    row = self._ui.batch_row_for_set_id(set_id)
                except Exception as exc:
                    logger.debug("Failed to resolve duplicate batch label row for %s: %s", set_id, exc, exc_info=True)
                    row = None
                if row is not None:
                    popup_label = f"{label} (row {int(row) + 1})"
            popup_labels[set_id] = popup_label
        return popup_labels

    def _sync_main_plot_copy_labels(self, *, primary_set_id: str, display_set_ids: Sequence[str]) -> None:
        primary_set_id_s = str(primary_set_id or "").strip()
        display_ids: list[str] = []
        for raw_set_id in display_set_ids or ():
            set_id = str(raw_set_id or "").strip()
            if not set_id or set_id in display_ids:
                continue
            display_ids.append(set_id)
        if primary_set_id_s and primary_set_id_s not in display_ids:
            display_ids.insert(0, primary_set_id_s)
        plot = self._main_plot()
        plot.set_simulation_popup_labels(
            primary_set_id=primary_set_id_s,
            popup_labels_by_set_id=self._popup_labels_by_set_id(display_ids),
        )

    def _deauthorize_active_display_transaction_outputs(self) -> ActiveDisplayTransaction | None:
        previous_transaction = self._active_display_transaction
        self._active_display_transaction = None
        try:
            plot = self._main_plot()
            plot.clear_display_transaction_state()
        except Exception as exc:
            logger.debug("Failed to clear display transaction plot state: %s", exc, exc_info=True)
        try:
            self._clear_direct_completion_provenance()
        except Exception as exc:
            logger.debug("Failed to clear display provenance during display deauthorization: %s", exc, exc_info=True)
        try:
            self._ui.show_simulation_tab()
        except Exception as exc:
            logger.debug("Failed to show simulation tab during display deauthorization: %s", exc, exc_info=True)
        try:
            self._ui.refresh_simulation_plot_views()
        except Exception as exc:
            logger.debug("Failed to refresh simulation plot views during display deauthorization: %s", exc, exc_info=True)
        return previous_transaction

    def clear_active_display_transaction(
        self,
        *,
        outcome_kind: DisplayTransitionOutcomeKind = DisplayTransitionOutcomeKind.CLEARED,
        display_status: DisplayStatus = DisplayStatus.DISPLAY_CLEARED,
        event_kind: DisplayEventKind = DisplayEventKind.DISPLAY_CLEARED,
        cause: DisplayTransitionCause = DisplayTransitionCause.MANUAL_CLEAR,
        requested_show_set_ids: Sequence[str] | None = None,
        requested_labels_by_set_id: Mapping[str, str] | None = None,
        attempted_display_set_ids: Sequence[str] = (),
        affected_set_ids: Sequence[str] = (),
        unresolved_intent_set_ids: Sequence[str] = (),
        missing_intent_set_ids: Sequence[str] = (),
        failed_intent_set_ids: Sequence[str] = (),
        semantic_unavailable_set_ids: Sequence[str] = (),
    ) -> DisplayTransitionOutcome:
        previous_transaction = self._deauthorize_active_display_transaction_outputs()
        transition = self._record_display_transition_outcome(
            outcome_kind=outcome_kind,
            active_transaction=None,
            previous_transaction=previous_transaction,
            display_status=display_status,
            requested_show_set_ids=(
                requested_show_set_ids
                if requested_show_set_ids is not None
                else affected_set_ids
            ),
            requested_labels_by_set_id=requested_labels_by_set_id,
            display_set_ids=(),
            attempted_display_set_ids=attempted_display_set_ids,
            affected_set_ids=affected_set_ids,
            unresolved_intent_set_ids=unresolved_intent_set_ids,
            missing_intent_set_ids=missing_intent_set_ids,
            failed_intent_set_ids=failed_intent_set_ids,
            semantic_unavailable_set_ids=semantic_unavailable_set_ids,
            event_kind=event_kind,
            cause=cause,
        )
        self._set_status_from_display_transition(transition)
        return transition

    def clear_display_if_workspace_previews_were_displayed(
        self,
        set_ids: Sequence[str],
    ) -> bool:
        displayed_set_ids = self._displayed_workspace_preview_set_ids(set_ids)
        if not displayed_set_ids:
            return False
        self._ui.clear_active_preview_cache_identity_state()
        self.clear_active_display_transaction()
        return True

    def _displayed_workspace_preview_set_ids(
        self,
        set_ids: Sequence[str],
    ) -> list[str]:
        candidates = [str(set_id) for set_id in (set_ids or ()) if str(set_id)]
        if not candidates:
            return []
        active_transaction = self._active_display_transaction
        if isinstance(active_transaction, ActiveDisplayTransaction):
            preview_display_ids = {
                str(metadata.set_id or "").strip()
                for metadata in dict(active_transaction.sets or {}).values()
                if isinstance(metadata.workspace_preview_provenance, Mapping)
                and str(metadata.set_id or "").strip()
                and str(metadata.set_id or "").strip() in set(active_transaction.display_set_ids)
            }
            matched_display_ids = [set_id for set_id in candidates if set_id in preview_display_ids]
            if matched_display_ids:
                return matched_display_ids
        return [
            set_id
            for set_id in candidates
            if self.displayed_workspace_preview_provenance_matches_current_workspace(set_id=set_id)
        ]

    def authoritative_result_transition_required(
        self,
        *,
        cache_stale_scope_is_global: bool,
        cache_stale_set_ids: Sequence[str],
    ) -> bool:
        return bool(
            self._active_display_transaction is not None
            or bool(cache_stale_scope_is_global)
            or any(str(set_id) for set_id in (cache_stale_set_ids or ()))
        )

    def apply_authoritative_result_display_transition(
        self,
        *,
        active_cache_key: str,
        display_scope_ids: Sequence[str],
        active_cache_invalidated_set_ids: Sequence[str],
        display_clear_set_ids: Sequence[str],
        display_clear_scope_is_global: bool,
    ) -> AuthoritativeResultDisplayTransitionOutcome:
        display_scope_ids_t = tuple(str(set_id) for set_id in (display_scope_ids or ()) if str(set_id))
        if bool(display_clear_scope_is_global) and self._active_display_transaction is not None:
            affected_ids = tuple(
                str(set_id) for set_id in (display_clear_set_ids or ()) if str(set_id)
            ) or tuple(str(set_id) for set_id in self._active_display_transaction.display_set_ids if str(set_id))
            self.clear_active_display_transaction(
                display_status=DisplayStatus.NO_COMPLETE_DISPLAYABLE_REQUEST_SCOPE,
                cause=DisplayTransitionCause.AFFECTED_SCOPE_INTERSECTS_ACTIVE_DISPLAY,
                affected_set_ids=affected_ids,
                unresolved_intent_set_ids=affected_ids,
            )
            return AuthoritativeResultDisplayTransitionOutcome(refresh_requested=False)
        if self._deauthorize_completed_run_display(
            transition=_DISPLAY_TRANSITION_AUTHORITATIVE_INVALIDATION,
            affected_set_ids=display_clear_set_ids,
            affected_scope_is_global=bool(display_clear_scope_is_global),
        ) is not None:
            return AuthoritativeResultDisplayTransitionOutcome(refresh_requested=False)
        if self._completed_run_display_transaction_active():
            return AuthoritativeResultDisplayTransitionOutcome(refresh_requested=False)
        display_cleared = False
        if active_cache_key and self._affected_scope_intersects_active_plotted_display(
            affected_set_ids=display_clear_set_ids,
            affected_scope_is_global=bool(display_clear_scope_is_global),
        ):
            display_cleared = True

        return self._finish_authoritative_result_display_update(
            active_cache_key=active_cache_key,
            display_scope_ids=display_scope_ids_t,
            active_cache_invalidated_set_ids=active_cache_invalidated_set_ids,
            display_cleared=display_cleared,
        )

    @staticmethod
    def _normalized_set_id_set(set_ids: Sequence[str]) -> set[str]:
        return {str(set_id) for set_id in (set_ids or ()) if str(set_id)}

    def _active_plotted_display_set_ids(self) -> set[str]:
        transaction = self._active_display_transaction
        if transaction is None:
            return set()
        return {str(set_id) for set_id in transaction.display_set_ids if str(set_id)}

    def _affected_scope_intersects_active_plotted_display(
        self,
        *,
        affected_set_ids: Sequence[str],
        affected_scope_is_global: bool,
    ) -> bool:
        if bool(affected_scope_is_global):
            return self._active_display_transaction is not None
        affected_scope = self._normalized_set_id_set(affected_set_ids)
        if not affected_scope:
            return False
        return bool(affected_scope & self._active_plotted_display_set_ids())

    def _finish_authoritative_result_display_update(
        self,
        *,
        active_cache_key: str,
        display_scope_ids: Sequence[str],
        active_cache_invalidated_set_ids: Sequence[str],
        display_cleared: bool,
    ) -> AuthoritativeResultDisplayTransitionOutcome:
        if active_cache_key and active_cache_invalidated_set_ids and display_cleared:
            affected_ids = tuple(
                str(set_id) for set_id in (active_cache_invalidated_set_ids or ()) if str(set_id)
            )
            self.clear_active_display_transaction(
                display_status=DisplayStatus.NO_COMPLETE_DISPLAYABLE_REQUEST_SCOPE,
                cause=DisplayTransitionCause.CACHE_RESULT_UNAVAILABLE,
                affected_set_ids=affected_ids,
                unresolved_intent_set_ids=affected_ids,
            )
            return AuthoritativeResultDisplayTransitionOutcome(refresh_requested=False)
        if display_scope_ids and (active_cache_key or self._active_display_transaction is not None):
            return AuthoritativeResultDisplayTransitionOutcome(refresh_requested=True)
        if display_scope_ids:
            return AuthoritativeResultDisplayTransitionOutcome(refresh_requested=False)
        self.clear_active_display_transaction()
        return AuthoritativeResultDisplayTransitionOutcome(refresh_requested=False)

    @staticmethod
    def _transaction_overlay_is_reference(entry: Mapping[str, Any]) -> bool:
        return transaction_overlay_is_reference(entry)

    def _reference_overlay_visible_for_publication(self) -> bool:
        try:
            return bool(self._main_plot().reference_layers_visible())
        except Exception as exc:
            logger.debug(
                "Reference overlay visibility unavailable during display publication: %s",
                exc,
                exc_info=True,
            )
            return True

    def current_workspace_preview_identity_payload(self, *, set_id: str) -> Optional[Dict[str, Any]]:
        sid = str(set_id or "").strip()
        if not sid:
            return None
        payload = self._ui.current_workspace_preview_identity_payload(sid)
        return dict(payload) if isinstance(payload, Mapping) else None

    def displayed_workspace_preview_provenance_matches(
        self,
        *,
        set_id: str,
        current_payload: Mapping[str, Any],
    ) -> bool:
        sid = str(set_id or "").strip()
        if not sid or not isinstance(current_payload, Mapping):
            return False
        active_transaction = self._active_display_transaction
        if not isinstance(active_transaction, ActiveDisplayTransaction):
            return False
        for metadata in dict(active_transaction.sets or {}).values():
            if str(metadata.set_id or "").strip() != sid:
                continue
            stored_payload = metadata.workspace_preview_provenance
            if isinstance(stored_payload, Mapping) and dict(stored_payload) == dict(current_payload):
                return True
        return False

    def displayed_workspace_preview_provenance_matches_current_workspace(self, *, set_id: str) -> bool:
        sid = str(set_id or "").strip()
        if not sid:
            return False
        current_payload = self.current_workspace_preview_identity_payload(set_id=sid)
        if not isinstance(current_payload, Mapping):
            return False
        return self.displayed_workspace_preview_provenance_matches(
            set_id=sid,
            current_payload=current_payload,
        )

    def _completed_run_display_transaction_active(self) -> bool:
        transaction = self._active_display_transaction
        return (
            transaction is not None
            and transaction.kind is ActiveDisplayKind.COMPLETED_RUN
            and bool(transaction.display_set_ids)
        )

    def _completed_run_display_set_ids(self) -> set[str]:
        if not self._completed_run_display_transaction_active():
            return set()
        return self._active_plotted_display_set_ids()

    def _display_mutation_denied(
        self,
        *,
        transition: DisplayPublicationTransition,
    ) -> bool:
        if (
            transition.completed_run_conflict_policy
            is not CompletedRunDisplayConflictPolicy.DENY_PUBLICATION
        ):
            return False
        return self._completed_run_display_transaction_active()

    @staticmethod
    def _completed_run_display_noop_refresh_outcome(
        *,
        focused_controls_use_workspace: Optional[bool] = None,
    ) -> BatchDisplayRefreshOutcome:
        return BatchDisplayRefreshOutcome(
            focused_controls_use_workspace=focused_controls_use_workspace,
        )

    @staticmethod
    def _completed_run_display_noop_refresh_outcome_for_request(
        *,
        request: BatchDisplayRefreshRequest,
        resolution: BatchDisplayRequestResolution | None = None,
    ) -> BatchDisplayRefreshOutcome:
        focus_uses_workspace = bool(
            request.focused_dirty
            or request.focused_set_dirty
            or (
                resolution is not None
                and bool(resolution.focused_uses_workspace_controls)
            )
        )
        return ResultsController._completed_run_display_noop_refresh_outcome(
            focused_controls_use_workspace=focus_uses_workspace,
        )

    @staticmethod
    def _completed_run_display_deauthorized_refresh_outcome() -> BatchDisplayRefreshOutcome:
        return BatchDisplayRefreshOutcome(
            focused_controls_use_workspace=None,
        )

    def _deauthorize_active_display(
        self,
        *,
        transition: DisplayPublicationTransition,
        affected_set_ids: Sequence[str],
        affected_scope_is_global: bool,
    ) -> DisplayTransitionOutcome | None:
        if self._active_display_transaction is None:
            return None
        if not self._affected_scope_intersects_active_plotted_display(
            affected_set_ids=affected_set_ids,
            affected_scope_is_global=bool(affected_scope_is_global),
        ):
            return None
        return self.clear_active_display_transaction(
            display_status=DisplayStatus.NO_COMPLETE_DISPLAYABLE_REQUEST_SCOPE,
            affected_set_ids=affected_set_ids,
            unresolved_intent_set_ids=affected_set_ids,
            event_kind=transition.event_kind,
            cause=transition.cause,
        )

    def _deauthorize_completed_run_display(
        self,
        *,
        transition: DisplayPublicationTransition,
        affected_set_ids: Sequence[str],
        affected_scope_is_global: bool,
    ) -> DisplayTransitionOutcome | None:
        if (
            transition.completed_run_conflict_policy
            is not CompletedRunDisplayConflictPolicy.CLEAR_INTERSECTING_COMPLETED_RUN
        ):
            return None
        if not self._completed_run_display_transaction_active():
            return None
        return self._deauthorize_active_display(
            transition=transition,
            affected_set_ids=affected_set_ids,
            affected_scope_is_global=affected_scope_is_global,
        )

    def deauthorize_display_for_deleted_sets(
        self,
        *,
        set_ids: Sequence[str],
    ) -> DisplayTransitionOutcome | None:
        return self._deauthorize_active_display(
            transition=_DISPLAY_TRANSITION_DELETED_SET_DEAUTHORIZATION,
            affected_set_ids=tuple(str(set_id) for set_id in (set_ids or ()) if str(set_id)),
            affected_scope_is_global=False,
        )

    def _handle_semantic_display_unavailable(
        self,
        *,
        affected_set_ids: Sequence[str],
        requested_show_set_ids: Sequence[str] | None = None,
        requested_labels_by_set_id: Mapping[str, str] | None = None,
        attempted_display_set_ids: Sequence[str] = (),
        unresolved_intent_set_ids: Sequence[str] = (),
        missing_intent_set_ids: Sequence[str] = (),
        failed_intent_set_ids: Sequence[str] = (),
        semantic_unavailable_set_ids: Sequence[str] = (),
    ) -> DisplayTransitionOutcome:
        affected_scope = self._normalized_set_id_set(affected_set_ids)
        unresolved_scope = tuple(self._deduped_set_ids(unresolved_intent_set_ids or tuple(affected_scope)))
        failed_scope = tuple(self._deduped_set_ids(failed_intent_set_ids))
        semantic_scope = tuple(
            self._deduped_set_ids(
                semantic_unavailable_set_ids
                or tuple(set_id for set_id in affected_scope if set_id not in set(failed_scope))
            )
        )
        active_transaction = self._active_display_transaction
        active_display_ids = {
            str(set_id)
            for set_id in (
                active_transaction.display_set_ids
                if isinstance(active_transaction, ActiveDisplayTransaction)
                else ()
            )
            if str(set_id)
        }
        if active_display_ids and affected_scope.isdisjoint(active_display_ids):
            return self._record_unpublished_display_request_outcome(
                outcome_kind=DisplayTransitionOutcomeKind.DENIED,
                display_status=DisplayStatus.DISPLAY_DENIED,
                affected_set_ids=tuple(affected_scope),
                requested_show_set_ids=requested_show_set_ids,
                requested_labels_by_set_id=requested_labels_by_set_id,
                attempted_display_set_ids=attempted_display_set_ids,
                unresolved_intent_set_ids=unresolved_scope,
                missing_intent_set_ids=missing_intent_set_ids,
                failed_intent_set_ids=failed_scope,
                semantic_unavailable_set_ids=semantic_scope,
                event_kind=DisplayEventKind.COMPLETED_RUN_COVERAGE_UNAVAILABLE,
                cause=DisplayTransitionCause.SEMANTIC_METADATA_UNAVAILABLE,
            )
        return self.clear_active_display_transaction(
            outcome_kind=DisplayTransitionOutcomeKind.FAILED,
            display_status=DisplayStatus.NO_COMPLETE_DISPLAYABLE_REQUEST_SCOPE,
            requested_show_set_ids=requested_show_set_ids,
            requested_labels_by_set_id=requested_labels_by_set_id,
            attempted_display_set_ids=attempted_display_set_ids,
            affected_set_ids=tuple(affected_scope),
            unresolved_intent_set_ids=unresolved_scope,
            missing_intent_set_ids=missing_intent_set_ids,
            failed_intent_set_ids=failed_scope,
            semantic_unavailable_set_ids=semantic_scope,
            event_kind=DisplayEventKind.COMPLETED_RUN_COVERAGE_UNAVAILABLE,
            cause=DisplayTransitionCause.SEMANTIC_METADATA_UNAVAILABLE,
        )

    def _handle_completed_run_display_unavailable(
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
    ) -> DisplayTransitionOutcome:
        if not isinstance(cause, DisplayTransitionCause):
            raise TypeError("Completed-run display unavailable requires DisplayTransitionCause")
        affected_scope = self._normalized_set_id_set(affected_set_ids)
        unresolved_scope = tuple(self._deduped_set_ids(unresolved_intent_set_ids or tuple(affected_scope)))
        failed_scope = tuple(
            self._deduped_set_ids(
                failed_intent_set_ids
                or (
                    tuple(affected_scope)
                    if cause is DisplayTransitionCause.NO_DISPLAYABLE_COMPLETION_RESULTS
                    else ()
                )
            )
        )
        active_transaction = self._active_display_transaction
        active_display_ids = {
            str(set_id)
            for set_id in (
                active_transaction.display_set_ids
                if isinstance(active_transaction, ActiveDisplayTransaction)
                else ()
            )
            if str(set_id)
        }
        if active_display_ids and affected_scope.isdisjoint(active_display_ids):
            return self._record_unpublished_display_request_outcome(
                outcome_kind=DisplayTransitionOutcomeKind.FAILED,
                display_status=DisplayStatus.NO_COMPLETE_DISPLAYABLE_REQUEST_SCOPE,
                affected_set_ids=tuple(affected_scope),
                requested_show_set_ids=requested_show_set_ids,
                requested_labels_by_set_id=requested_labels_by_set_id,
                attempted_display_set_ids=attempted_display_set_ids,
                unresolved_intent_set_ids=unresolved_scope,
                missing_intent_set_ids=missing_intent_set_ids,
                failed_intent_set_ids=failed_scope,
                event_kind=DisplayEventKind.COMPLETED_RUN_COVERAGE_UNAVAILABLE,
                cause=cause,
            )
        return self.clear_active_display_transaction(
            outcome_kind=DisplayTransitionOutcomeKind.FAILED,
            display_status=DisplayStatus.NO_COMPLETE_DISPLAYABLE_REQUEST_SCOPE,
            requested_show_set_ids=requested_show_set_ids,
            requested_labels_by_set_id=requested_labels_by_set_id,
            attempted_display_set_ids=attempted_display_set_ids,
            affected_set_ids=tuple(affected_scope),
            unresolved_intent_set_ids=unresolved_scope,
            missing_intent_set_ids=missing_intent_set_ids,
            failed_intent_set_ids=failed_scope,
            event_kind=DisplayEventKind.COMPLETED_RUN_COVERAGE_UNAVAILABLE,
            cause=cause,
        )

    def _deauthorize_active_display_after_display_scope_refresh(
        self,
        *,
        request: BatchDisplayRefreshRequest,
        requested_show_set_ids: Sequence[str],
    ) -> DisplayTransitionOutcome | None:
        if request.display_source is not DisplayRefreshSource.EXPLICIT_SHOW_REQUEST:
            return None
        display_set_ids = self._active_plotted_display_set_ids()
        if not display_set_ids:
            return None
        requested_show_set = {str(set_id) for set_id in (requested_show_set_ids or ()) if str(set_id)}
        removed_display_ids = tuple(sorted(display_set_ids - requested_show_set))
        if not removed_display_ids:
            return None
        return self._deauthorize_active_display(
            transition=_DISPLAY_TRANSITION_DISPLAY_SCOPE_REMOVAL_DEAUTHORIZATION,
            affected_set_ids=removed_display_ids,
            affected_scope_is_global=False,
        )

    def _focus_unavailable_cause_for_completed_run_denial(
        self,
        *,
        request: BatchDisplayRefreshRequest,
        resolution: BatchDisplayRequestResolution,
        requested_show_set_ids: Sequence[str],
    ) -> DisplayTransitionCause | None:
        if request.display_source is not DisplayRefreshSource.EXPLICIT_SHOW_REQUEST:
            return None
        focused_set_id = str(request.prefer_set_id or "").strip()
        if not focused_set_id:
            return None
        requested_show_set = {str(set_id) for set_id in (requested_show_set_ids or ()) if str(set_id)}
        active_completed_ids = set(self._completed_run_display_set_ids())
        resolved_ids = {
            str(entry.set_id)
            for entry in (resolution.resolved_entries or ())
            if str(entry.set_id)
        }
        available_ids = active_completed_ids | resolved_ids
        if requested_show_set and requested_show_set.issubset(available_ids):
            return None
        focus_resolution = resolution
        if focused_set_id not in requested_show_set:
            try:
                focus_resolution = self._ui.workspace_display_request_resolution((focused_set_id,))
            except Exception as exc:
                logger.debug("Failed to resolve focused set display availability: %s", exc, exc_info=True)
                focus_resolution = resolution
        if focus_resolution.resolved_entries and focus_resolution.focused_has_resolved_entry:
            return None
        return (
            focus_resolution.unavailable_cause
            or resolution.unavailable_cause
            or DisplayTransitionCause.CACHE_RESULT_UNAVAILABLE
        )

    def _completed_run_denied_unavailable_outcome(
        self,
        *,
        request: BatchDisplayRefreshRequest,
        resolution: BatchDisplayRequestResolution,
        requested_show_set_ids: Sequence[str],
    ) -> DisplayTransitionOutcome | None:
        cause = self._focus_unavailable_cause_for_completed_run_denial(
            request=request,
            resolution=resolution,
            requested_show_set_ids=requested_show_set_ids,
        )
        if cause is None:
            return None
        requested_ids = tuple(self._deduped_set_ids(requested_show_set_ids))
        available_ids = {
            *self._completed_run_display_set_ids(),
            *(
                str(entry.set_id)
                for entry in (resolution.resolved_entries or ())
                if str(entry.set_id)
            ),
        }
        missing_ids = tuple(
            set_id
            for set_id in requested_ids
            if set_id not in available_ids
        )
        unresolved_ids = missing_ids or requested_ids
        outcome = self._record_unpublished_display_request_outcome(
            cause=cause,
            affected_set_ids=requested_ids,
            requested_show_set_ids=requested_ids,
            requested_labels_by_set_id=self._popup_labels_by_set_id(requested_ids),
            unresolved_intent_set_ids=unresolved_ids,
            missing_intent_set_ids=missing_ids,
            outcome_kind=DisplayTransitionOutcomeKind.DENIED,
            display_status=DisplayStatus.DISPLAY_DENIED,
        )
        self._set_status_from_display_transition(outcome)
        return outcome

    def deauthorize_completed_run_display_for_runtime_input_preview(
        self,
        *,
        affected_set_ids: Sequence[str],
        affected_scope_is_global: bool,
    ) -> DisplayTransitionOutcome | None:
        return self._deauthorize_completed_run_display(
            transition=_DISPLAY_TRANSITION_RUNTIME_INPUT_PREVIEW_DEAUTHORIZATION,
            affected_set_ids=affected_set_ids,
            affected_scope_is_global=bool(affected_scope_is_global),
        )

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
    ) -> SimulationCompletionDisplayOutcome:
        if not isinstance(cause, DisplayTransitionCause):
            raise TypeError("Completed-run display unavailable requires DisplayTransitionCause")
        typed_cause = cause
        transition_outcome: DisplayTransitionOutcome
        if typed_cause is DisplayTransitionCause.SEMANTIC_METADATA_UNAVAILABLE:
            transition_outcome = self._handle_semantic_display_unavailable(
                affected_set_ids=affected_set_ids,
                requested_show_set_ids=requested_show_set_ids,
                requested_labels_by_set_id=requested_labels_by_set_id,
                attempted_display_set_ids=attempted_display_set_ids,
                unresolved_intent_set_ids=unresolved_intent_set_ids,
                missing_intent_set_ids=missing_intent_set_ids,
                failed_intent_set_ids=failed_intent_set_ids,
                semantic_unavailable_set_ids=semantic_unavailable_set_ids,
            )
        elif typed_cause is DisplayTransitionCause.NO_DISPLAYABLE_COMPLETION_RESULTS:
            transition_outcome = self._handle_completed_run_display_unavailable(
                cause=typed_cause,
                affected_set_ids=affected_set_ids,
                requested_show_set_ids=requested_show_set_ids,
                requested_labels_by_set_id=requested_labels_by_set_id,
                attempted_display_set_ids=attempted_display_set_ids,
                unresolved_intent_set_ids=unresolved_intent_set_ids,
                missing_intent_set_ids=missing_intent_set_ids,
                failed_intent_set_ids=failed_intent_set_ids,
            )
        else:
            transition_outcome = self._handle_completed_run_display_unavailable(
                cause=typed_cause,
                affected_set_ids=affected_set_ids,
                requested_show_set_ids=requested_show_set_ids,
                requested_labels_by_set_id=requested_labels_by_set_id,
                attempted_display_set_ids=attempted_display_set_ids,
                unresolved_intent_set_ids=unresolved_intent_set_ids,
                missing_intent_set_ids=missing_intent_set_ids,
                failed_intent_set_ids=failed_intent_set_ids,
            )
        return SimulationCompletionDisplayOutcome(
            transition_outcome=transition_outcome,
        )

    @staticmethod
    def _direct_completion_owned_species(
        *,
        series: Mapping[str, object],
        direct_completion_provenance: Mapping[str, Any] | None,
    ) -> tuple[str, ...]:
        if not isinstance(direct_completion_provenance, Mapping):
            return ()
        available_series = {str(name) for name in dict(series or {}) if str(name)}
        owned_species: list[str] = []
        for raw_name in direct_completion_provenance.get("species_names") or ():
            name = str(raw_name or "").strip()
            if not name or name in owned_species:
                continue
            if available_series and name not in available_series:
                continue
            owned_species.append(name)
        return tuple(owned_species)

    @staticmethod
    def _completion_entry_matches_intent_owned_species(
        transaction: CompletedRunDisplayTransaction,
        entry: CompletionDisplayEntry,
    ) -> bool:
        sid = str(entry.set_id or "").strip()
        if not sid:
            return False
        intent_owned_species = tuple(
            str(name)
            for name in dict(transaction.intent.owned_species_by_set_id or {}).get(sid, ())
            if str(name)
        )
        entry_owned_species = tuple(str(name) for name in entry.owned_species if str(name))
        return bool(intent_owned_species) and entry_owned_species == intent_owned_species

    @staticmethod
    def _completion_entry_semantic_series(
        entry: CompletionDisplayEntry,
    ) -> Optional[Dict[str, object]]:
        owned_species = tuple(str(name) for name in entry.owned_species if str(name))
        raw_series = entry.series if isinstance(entry.series, Mapping) else {}
        semantic_series: Dict[str, object] = {}
        for species_name in owned_species:
            if species_name not in raw_series:
                return None
            semantic_series[species_name] = raw_series[species_name]
        return semantic_series or None

    def build_main_plot_copy_all_export_plan(self) -> object | None:
        return build_copy_all_export_plan(self._active_display_transaction)

    @staticmethod
    def _ordered_display_transaction_metadata(
        transaction: ActiveDisplayTransaction,
    ) -> list[DisplaySetMetadata]:
        return ordered_display_transaction_metadata(transaction)

    def build_main_plot_csv_export(self, scope: str) -> tuple[list[str], list[list[object]]]:
        active_transaction = self._active_display_transaction
        if active_transaction is None:
            raise ValueError("No active simulation display transaction is available to export.")
        normalized_scope = str(scope or "axis")
        axis_state = {
            "x_name": "t",
            "x_header": "t",
            "y_names": (),
        }
        plot = self._main_plot()
        candidate = plot.transaction_export_axis_state(normalized_scope)
        if isinstance(candidate, Mapping):
            axis_state.update(dict(candidate))
        header, rows = build_main_plot_csv_export(
            active_transaction=active_transaction,
            scope=normalized_scope,
            axis_state=axis_state,
        )
        return plot.append_dataset_overlay_export_columns(header, rows, normalized_scope)

    def reset_stale_cache_warning_status(self) -> None:
        transition = self._last_display_transition_outcome
        stale_causes = {
            DisplayTransitionCause.CACHE_RESULT_UNAVAILABLE,
            DisplayTransitionCause.INVALID_CACHE_ENTRY,
            DisplayTransitionCause.IN_FLIGHT_COVERAGE_UNAVAILABLE,
            DisplayTransitionCause.SHOW_REMOVED_ACTIVE_SET,
            DisplayTransitionCause.AFFECTED_SCOPE_INTERSECTS_ACTIVE_DISPLAY,
            DisplayTransitionCause.DELETED_ACTIVE_SET,
            DisplayTransitionCause.SEMANTIC_METADATA_UNAVAILABLE,
            DisplayTransitionCause.NO_DISPLAYABLE_COMPLETION_RESULTS,
        }
        if (
            isinstance(transition, DisplayTransitionOutcome)
            and transition.kind is not DisplayTransitionOutcomeKind.PUBLISHED
            and transition.cause in stale_causes
        ):
            self._ui.set_status_text("Ready")
            self._last_display_transition_outcome = None

    def _clear_unpublished_batch_display_request(
        self,
        *,
        clear_plot: bool = True,
    ) -> None:
        if self._completed_run_display_transaction_active():
            return
        if bool(clear_plot):
            self.clear_active_display_transaction()

    def _active_transaction_for_display_commit(
        self,
        *,
        t: np.ndarray,
        series: Mapping[str, Any],
        overlays: Sequence[Mapping[str, Any]],
        primary_set_id: str,
        primary_label: str,
        display_set_ids: Sequence[str],
        owned_species: Sequence[str] | None,
        completion_provenance: Mapping[str, Any] | None,
        workspace_preview_provenance_by_set_id: Mapping[str, Mapping[str, Any]] | None,
        display_transition: DisplayPublicationTransition,
        run_id: int | None = None,
        request_id: int | None = None,
        intervention_annotations: Sequence[Mapping[str, Any]] = (),
        show_intervention_annotations: bool = False,
    ) -> ActiveDisplayTransaction:
        kind = self._active_display_kind_for_transition(display_transition)
        status = self._display_status_for_kind(kind)
        return active_transaction_for_display_commit(
            t=t,
            series=series,
            overlays=overlays,
            primary_set_id=primary_set_id,
            primary_label=primary_label,
            display_set_ids=display_set_ids,
            owned_species=owned_species,
            completion_provenance=completion_provenance,
            workspace_preview_provenance_by_set_id=workspace_preview_provenance_by_set_id,
            active_kind=kind,
            status=status,
            request_id=request_id,
            run_id=run_id,
            intervention_annotations=intervention_annotations,
            show_intervention_annotations=show_intervention_annotations,
        )

    def _record_display_transition_outcome(
        self,
        *,
        outcome_kind: DisplayTransitionOutcomeKind,
        active_transaction: ActiveDisplayTransaction | None,
        previous_transaction: ActiveDisplayTransaction | None,
        display_status: DisplayStatus,
        request_scope: DisplayRequestScopeSnapshot | None = None,
        requested_show_set_ids: Sequence[str] | None = None,
        requested_labels_by_set_id: Mapping[str, str] | None = None,
        display_set_ids: Sequence[str] | None = None,
        attempted_display_set_ids: Sequence[str] | None = None,
        affected_set_ids: Sequence[str] = (),
        unresolved_intent_set_ids: Sequence[str] = (),
        missing_intent_set_ids: Sequence[str] = (),
        failed_intent_set_ids: Sequence[str] = (),
        semantic_unavailable_set_ids: Sequence[str] = (),
        event_kind: DisplayEventKind | None = None,
        cause: DisplayTransitionCause | None = None,
    ) -> DisplayTransitionOutcome:
        resolved_request_scope = (
            request_scope
            if isinstance(request_scope, DisplayRequestScopeSnapshot)
            else None
        )
        outcome_requested_ids = tuple(
            self._deduped_set_ids(
                requested_show_set_ids
                if requested_show_set_ids is not None
                else (
                    resolved_request_scope.requested_show_set_ids
                    if resolved_request_scope is not None
                    else ()
                )
            )
        )
        outcome_labels = {
            str(set_id): str(label)
            for set_id, label in dict(
                requested_labels_by_set_id
                if requested_labels_by_set_id is not None
                else (
                    resolved_request_scope.requested_labels_by_set_id
                    if resolved_request_scope is not None
                    else {}
                )
            ).items()
            if str(set_id)
        }
        if resolved_request_scope is None:
            resolved_request_scope = self._current_display_request_scope(
                requested_show_set_ids=outcome_requested_ids,
                requested_labels_by_set_id=outcome_labels,
            )
        outcome_display_ids = tuple(
            self._deduped_set_ids(
                display_set_ids
                if display_set_ids is not None
                else (
                    active_transaction.display_set_ids
                    if isinstance(active_transaction, ActiveDisplayTransaction)
                    else ()
                )
            )
        )
        outcome_attempted_ids = tuple(
            self._deduped_set_ids(
                attempted_display_set_ids
                if attempted_display_set_ids is not None
                else outcome_display_ids
            )
        )
        outcome = DisplayTransitionOutcome(
            kind=outcome_kind,
            active_transaction=active_transaction,
            previous_transaction=previous_transaction,
            display_status=display_status,
            request_scope=resolved_request_scope,
            requested_show_set_ids=outcome_requested_ids,
            requested_labels_by_set_id=outcome_labels,
            display_set_ids=outcome_display_ids,
            attempted_display_set_ids=outcome_attempted_ids,
            affected_set_ids=tuple(affected_set_ids),
            unresolved_intent_set_ids=tuple(unresolved_intent_set_ids),
            missing_intent_set_ids=tuple(missing_intent_set_ids),
            failed_intent_set_ids=tuple(failed_intent_set_ids),
            semantic_unavailable_set_ids=tuple(semantic_unavailable_set_ids),
            event_kind=event_kind,
            cause=cause,
        )
        self._last_display_transition_outcome = outcome
        return outcome

    def _record_unpublished_display_request_outcome(
        self,
        *,
        affected_set_ids: Sequence[str] = (),
        requested_show_set_ids: Sequence[str] | None = None,
        requested_labels_by_set_id: Mapping[str, str] | None = None,
        attempted_display_set_ids: Sequence[str] = (),
        unresolved_intent_set_ids: Sequence[str] = (),
        missing_intent_set_ids: Sequence[str] = (),
        failed_intent_set_ids: Sequence[str] = (),
        semantic_unavailable_set_ids: Sequence[str] = (),
        outcome_kind: DisplayTransitionOutcomeKind = DisplayTransitionOutcomeKind.DENIED,
        display_status: DisplayStatus = DisplayStatus.DISPLAY_DENIED,
        event_kind: DisplayEventKind = DisplayEventKind.DISPLAY_FAILURE,
        cause: DisplayTransitionCause = DisplayTransitionCause.DISPLAY_MUTATION_DENIED,
    ) -> DisplayTransitionOutcome:
        active_transaction = self._active_display_transaction
        resolved_status = self._display_status_for_unpublished_request(
            requested_status=display_status,
            active_transaction=active_transaction,
        )
        outcome = self._record_display_transition_outcome(
            outcome_kind=outcome_kind,
            active_transaction=active_transaction,
            previous_transaction=active_transaction,
            display_status=resolved_status,
            requested_show_set_ids=(
                requested_show_set_ids
                if requested_show_set_ids is not None
                else affected_set_ids
            ),
            requested_labels_by_set_id=requested_labels_by_set_id,
            display_set_ids=(),
            attempted_display_set_ids=attempted_display_set_ids,
            affected_set_ids=affected_set_ids,
            unresolved_intent_set_ids=unresolved_intent_set_ids,
            missing_intent_set_ids=missing_intent_set_ids,
            failed_intent_set_ids=failed_intent_set_ids,
            semantic_unavailable_set_ids=semantic_unavailable_set_ids,
            event_kind=event_kind,
            cause=cause,
        )
        self._set_status_from_display_transition(outcome)
        return outcome

    def _simulation_no_display_outcome(
        self,
        cause: DisplayTransitionCause,
        *,
        affected_set_ids: Sequence[str] = (),
        requested_show_set_ids: Sequence[str] | None = None,
        requested_labels_by_set_id: Mapping[str, str] | None = None,
        attempted_display_set_ids: Sequence[str] = (),
        unresolved_intent_set_ids: Sequence[str] = (),
        missing_intent_set_ids: Sequence[str] = (),
        failed_intent_set_ids: Sequence[str] = (),
        semantic_unavailable_set_ids: Sequence[str] = (),
        outcome_kind: DisplayTransitionOutcomeKind = DisplayTransitionOutcomeKind.DENIED,
        display_status: DisplayStatus = DisplayStatus.DISPLAY_DENIED,
    ) -> SimulationCompletionDisplayOutcome:
        affected_ids = tuple(self._deduped_set_ids(affected_set_ids))
        unresolved_ids = tuple(self._deduped_set_ids(unresolved_intent_set_ids or affected_ids))
        if (
            self._active_display_transaction is not None
            and affected_ids
            and not self._affected_scope_intersects_active_plotted_display(
                affected_set_ids=affected_ids,
                affected_scope_is_global=False,
            )
        ):
            transition_outcome = self._record_unpublished_display_request_outcome(
                affected_set_ids=affected_ids,
                requested_show_set_ids=requested_show_set_ids,
                requested_labels_by_set_id=requested_labels_by_set_id,
                attempted_display_set_ids=attempted_display_set_ids,
                unresolved_intent_set_ids=unresolved_ids,
                missing_intent_set_ids=missing_intent_set_ids,
                failed_intent_set_ids=failed_intent_set_ids,
                semantic_unavailable_set_ids=semantic_unavailable_set_ids,
                outcome_kind=outcome_kind,
                display_status=display_status,
                event_kind=DisplayEventKind.DISPLAY_FAILURE,
                cause=cause,
            )
            return SimulationCompletionDisplayOutcome(
                transition_outcome=transition_outcome,
            )
        transition_outcome = self.clear_active_display_transaction(
            cause=cause,
            requested_show_set_ids=requested_show_set_ids,
            requested_labels_by_set_id=requested_labels_by_set_id,
            attempted_display_set_ids=attempted_display_set_ids,
            affected_set_ids=affected_ids,
            unresolved_intent_set_ids=unresolved_ids,
            missing_intent_set_ids=missing_intent_set_ids,
            failed_intent_set_ids=failed_intent_set_ids,
            semantic_unavailable_set_ids=semantic_unavailable_set_ids,
            outcome_kind=outcome_kind,
            display_status=display_status,
            event_kind=DisplayEventKind.DISPLAY_FAILURE,
        )
        return SimulationCompletionDisplayOutcome(
            transition_outcome=transition_outcome,
        )

    def _record_failed_display_transaction_attempt(
        self,
        *,
        previous_transaction: ActiveDisplayTransaction | None,
        affected_set_ids: Sequence[str],
        attempted_transaction: ActiveDisplayTransaction | None = None,
        request_scope: DisplayRequestScopeSnapshot | None = None,
        requested_show_set_ids: Sequence[str] | None = None,
        requested_labels_by_set_id: Mapping[str, str] | None = None,
        unresolved_intent_set_ids: Sequence[str] = (),
        missing_intent_set_ids: Sequence[str] = (),
        failed_intent_set_ids: Sequence[str] = (),
        semantic_unavailable_set_ids: Sequence[str] = (),
    ) -> DisplayTransitionOutcome:
        affected_ids = tuple(self._deduped_set_ids(affected_set_ids))
        unresolved_ids = tuple(self._deduped_set_ids(unresolved_intent_set_ids or affected_ids))
        self._deauthorize_active_display_transaction_outputs()
        transition = self._record_display_transition_outcome(
            outcome_kind=DisplayTransitionOutcomeKind.FAILED,
            active_transaction=None,
            previous_transaction=previous_transaction,
            display_status=DisplayStatus.DISPLAY_FAILED,
            request_scope=request_scope,
            requested_show_set_ids=(
                requested_show_set_ids
                if requested_show_set_ids is not None
                else (
                    request_scope.requested_show_set_ids
                    if isinstance(request_scope, DisplayRequestScopeSnapshot)
                    else affected_ids
                )
            ),
            requested_labels_by_set_id=(
                requested_labels_by_set_id
                if requested_labels_by_set_id is not None
                else (
                    request_scope.requested_labels_by_set_id
                    if isinstance(request_scope, DisplayRequestScopeSnapshot)
                    else {}
                )
            ),
            display_set_ids=(),
            attempted_display_set_ids=(
                attempted_transaction.display_set_ids
                if isinstance(attempted_transaction, ActiveDisplayTransaction)
                else affected_ids
            ),
            affected_set_ids=affected_ids,
            unresolved_intent_set_ids=unresolved_ids,
            missing_intent_set_ids=missing_intent_set_ids,
            failed_intent_set_ids=failed_intent_set_ids,
            semantic_unavailable_set_ids=semantic_unavailable_set_ids,
            event_kind=DisplayEventKind.DISPLAY_FAILURE,
            cause=DisplayTransitionCause.DISPLAY_MUTATION_FAILED,
        )
        self._set_status_from_display_transition(transition)
        return transition

    @staticmethod
    def _display_annotation_state(
        plot: ResultsDisplayPlotPort,
    ) -> tuple[tuple[Dict[str, object], ...], bool]:
        try:
            state = plot.intervention_annotation_state()
        except Exception as exc:
            logger.debug("Failed to read display annotation state: %s", exc, exc_info=True)
            return (), False
        if not isinstance(state, Mapping):
            return (), False
        annotations = tuple(
            dict(item)
            for item in (state.get("intervention_annotations") or ())
            if isinstance(item, Mapping)
        )
        return annotations, bool(state.get("show_intervention_annotations"))

    def _apply_batch_display_transaction(
        self,
        *,
        t: np.ndarray,
        series: Mapping[str, Any],
        label: str,
        overlays: Sequence[Dict[str, object]],
        metadata_applier: Callable[[ResultsDisplayPlotPort, ActiveDisplayTransaction], Optional[str]],
        annotation_entry: Mapping[str, Any],
        primary_set_id: str,
        primary_label: str,
        display_set_ids: Sequence[str],
        workspace_preview_provenance_by_set_id: Mapping[str, Mapping[str, Any]] | None = None,
        completion_provenance: Mapping[str, Any] | None = None,
        owned_species: Sequence[str] | None = None,
        display_transition: DisplayPublicationTransition,
        requested_show_set_ids: Sequence[str] | None = None,
        requested_labels_by_set_id: Mapping[str, str] | None = None,
        run_target_set_ids: Sequence[str] = (),
        unresolved_intent_set_ids: Sequence[str] = (),
        missing_intent_set_ids: Sequence[str] = (),
        failed_intent_set_ids: Sequence[str] = (),
        semantic_unavailable_set_ids: Sequence[str] = (),
        cache_key: str = "",
        run_id: int | None = None,
        request_id: int | None = None,
    ) -> CachedBatchDisplayScopeOutcome:
        attempted_display_ids = tuple(self._deduped_set_ids(display_set_ids))
        request_scope = self._current_display_request_scope(
            requested_show_set_ids=requested_show_set_ids,
            requested_labels_by_set_id=requested_labels_by_set_id,
            run_target_set_ids=run_target_set_ids,
            cache_key=cache_key,
            run_id=run_id,
            request_id=request_id,
        )
        display_denied = self._display_mutation_denied(
            transition=display_transition,
        )
        if display_denied:
            denied_ids = tuple(self._deduped_set_ids(requested_show_set_ids or attempted_display_ids))
            transition_outcome = self._record_unpublished_display_request_outcome(
                affected_set_ids=denied_ids,
                requested_show_set_ids=requested_show_set_ids,
                requested_labels_by_set_id=requested_labels_by_set_id,
                attempted_display_set_ids=attempted_display_ids,
                unresolved_intent_set_ids=denied_ids,
                cause=DisplayTransitionCause.DISPLAY_MUTATION_DENIED,
            )
            return CachedBatchDisplayScopeOutcome(
                transition_outcome=transition_outcome,
            )
        primary_set_id_s = str(primary_set_id or "").strip()
        previous_transaction = self._active_display_transaction
        active_transaction = self._active_transaction_for_display_commit(
            t=np.asarray(t, dtype=float),
            series=series,
            overlays=overlays,
            primary_set_id=str(primary_set_id),
            primary_label=str(primary_label),
            display_set_ids=attempted_display_ids,
            owned_species=owned_species,
            completion_provenance=completion_provenance,
            workspace_preview_provenance_by_set_id=workspace_preview_provenance_by_set_id,
            display_transition=display_transition,
            run_id=run_id,
            request_id=request_id,
        )
        plot = self._main_plot()
        metadata_failure = metadata_applier(plot, active_transaction)
        if metadata_failure:
            transition_outcome = self._record_failed_display_transaction_attempt(
                previous_transaction=previous_transaction,
                affected_set_ids=attempted_display_ids,
                attempted_transaction=active_transaction,
                request_scope=request_scope,
                requested_show_set_ids=requested_show_set_ids,
                requested_labels_by_set_id=requested_labels_by_set_id,
                unresolved_intent_set_ids=unresolved_intent_set_ids,
                missing_intent_set_ids=missing_intent_set_ids,
                failed_intent_set_ids=failed_intent_set_ids,
                semantic_unavailable_set_ids=semantic_unavailable_set_ids,
            )
            return CachedBatchDisplayScopeOutcome(
                transition_outcome=transition_outcome,
            )
        if not self._set_plot_data(
            np.asarray(t, dtype=float),
            {str(k): np.asarray(v, dtype=float) for k, v in series.items()},
            label=str(label),
            primary_set_id=primary_set_id_s,
            layer_id=f"result:{primary_set_id_s}" if primary_set_id_s else None,
            overlays=overlays,
            owned_species=owned_species,
        ):
            transition_outcome = self._record_failed_display_transaction_attempt(
                previous_transaction=previous_transaction,
                affected_set_ids=attempted_display_ids,
                attempted_transaction=active_transaction,
                request_scope=request_scope,
                requested_show_set_ids=requested_show_set_ids,
                requested_labels_by_set_id=requested_labels_by_set_id,
                unresolved_intent_set_ids=unresolved_intent_set_ids,
                missing_intent_set_ids=missing_intent_set_ids,
                failed_intent_set_ids=failed_intent_set_ids,
                semantic_unavailable_set_ids=semantic_unavailable_set_ids,
            )
            return CachedBatchDisplayScopeOutcome(
                transition_outcome=transition_outcome,
            )

        try:
            self._apply_intervention_annotations(plot=plot, entry=annotation_entry)
        except Exception as exc:
            logger.exception("Failed to apply batch intervention annotations: %s", exc)
            transition_outcome = self._record_failed_display_transaction_attempt(
                previous_transaction=previous_transaction,
                affected_set_ids=attempted_display_ids,
                attempted_transaction=active_transaction,
                request_scope=request_scope,
                requested_show_set_ids=requested_show_set_ids,
                requested_labels_by_set_id=requested_labels_by_set_id,
                unresolved_intent_set_ids=unresolved_intent_set_ids,
                missing_intent_set_ids=missing_intent_set_ids,
                failed_intent_set_ids=failed_intent_set_ids,
                semantic_unavailable_set_ids=semantic_unavailable_set_ids,
            )
            return CachedBatchDisplayScopeOutcome(
                transition_outcome=transition_outcome,
            )

        intervention_annotations, show_intervention_annotations = self._display_annotation_state(plot)
        if intervention_annotations or show_intervention_annotations:
            active_transaction = replace(
                active_transaction,
                intervention_annotations=intervention_annotations,
                show_intervention_annotations=show_intervention_annotations,
            )
        copy_metadata_failure = self._sync_plot_copy_metadata_from_transaction(
            primary_set_id=str(primary_set_id),
            display_set_ids=attempted_display_ids,
        )
        if copy_metadata_failure:
            transition_outcome = self._record_failed_display_transaction_attempt(
                previous_transaction=previous_transaction,
                affected_set_ids=attempted_display_ids,
                attempted_transaction=active_transaction,
                request_scope=request_scope,
                requested_show_set_ids=requested_show_set_ids,
                requested_labels_by_set_id=requested_labels_by_set_id,
                unresolved_intent_set_ids=unresolved_intent_set_ids,
                missing_intent_set_ids=missing_intent_set_ids,
                failed_intent_set_ids=failed_intent_set_ids,
                semantic_unavailable_set_ids=semantic_unavailable_set_ids,
            )
            return CachedBatchDisplayScopeOutcome(
                transition_outcome=transition_outcome,
            )
        provenance_failure = self._publish_direct_completion_provenance(
            plot=plot,
            direct_completion_provenance=completion_provenance,
            active_display_transaction=active_transaction,
        )
        if provenance_failure:
            transition_outcome = self._record_failed_display_transaction_attempt(
                previous_transaction=previous_transaction,
                affected_set_ids=attempted_display_ids,
                attempted_transaction=active_transaction,
                request_scope=request_scope,
                requested_show_set_ids=requested_show_set_ids,
                requested_labels_by_set_id=requested_labels_by_set_id,
                unresolved_intent_set_ids=unresolved_intent_set_ids,
                missing_intent_set_ids=missing_intent_set_ids,
                failed_intent_set_ids=failed_intent_set_ids,
                semantic_unavailable_set_ids=semantic_unavailable_set_ids,
            )
            return CachedBatchDisplayScopeOutcome(
                transition_outcome=transition_outcome,
            )
        self._active_display_transaction = active_transaction
        transition_outcome = self._record_display_transition_outcome(
            outcome_kind=DisplayTransitionOutcomeKind.PUBLISHED,
            active_transaction=active_transaction,
            previous_transaction=previous_transaction,
            display_status=active_transaction.status,
            request_scope=request_scope,
            requested_show_set_ids=requested_show_set_ids,
            requested_labels_by_set_id=requested_labels_by_set_id,
            display_set_ids=active_transaction.display_set_ids,
            attempted_display_set_ids=attempted_display_ids,
            affected_set_ids=active_transaction.display_set_ids,
            unresolved_intent_set_ids=unresolved_intent_set_ids,
            missing_intent_set_ids=missing_intent_set_ids,
            failed_intent_set_ids=failed_intent_set_ids,
            semantic_unavailable_set_ids=semantic_unavailable_set_ids,
            event_kind=self._display_event_kind_for_transition(display_transition),
            cause=self._display_cause_for_transition(display_transition),
        )
        self._commit_successful_plot_display(
            t=np.asarray(t, dtype=float),
            series=series,
            transition_outcome=transition_outcome,
        )
        return CachedBatchDisplayScopeOutcome(
            transition_outcome=transition_outcome,
        )

    def _display_cached_active_request_scope(
        self,
        *,
        request: BatchDisplayRefreshRequest,
    ) -> CachedBatchDisplayScopeOutcome:
        return self.publish_cached_batch_display_scope(
            cache_key=str(request.active_cache_key or ""),
            requested_show_set_ids=request.requested_show_set_ids,
            prefer_set=request.prefer_set_id,
            display_source=request.display_source,
        )

    @staticmethod
    def _refresh_success(
        *,
        display_outcome: CachedBatchDisplayScopeOutcome,
        focused_controls_use_workspace: bool,
    ) -> BatchDisplayRefreshOutcome:
        return BatchDisplayRefreshOutcome(
            focused_controls_use_workspace=bool(focused_controls_use_workspace),
            transition_outcome=display_outcome.transition_outcome,
        )

    @staticmethod
    def _refresh_failed(
        *,
        focused_controls_use_workspace: Optional[bool],
        transition_outcome: DisplayTransitionOutcome | None = None,
    ) -> BatchDisplayRefreshOutcome:
        return BatchDisplayRefreshOutcome(
            focused_controls_use_workspace=(
                None
                if focused_controls_use_workspace is None
                else bool(focused_controls_use_workspace)
            ),
            transition_outcome=transition_outcome,
        )

    def _publish_fresh_explicit_dirty_refresh(
        self,
        *,
        request: BatchDisplayRefreshRequest,
    ) -> Optional[BatchDisplayRefreshOutcome]:
        if not (request.focused_dirty and request.fresh_explicit_cache_after_post_run_sync):
            return None
        outcome = self._display_cached_active_request_scope(request=request)
        if self._outcome_published(outcome):
            self.reset_stale_cache_warning_status()
            return self._refresh_success(display_outcome=outcome, focused_controls_use_workspace=True)
        if self._outcome_denied_by_active_completed_run(outcome):
            return self._completed_run_display_noop_refresh_outcome_for_request(
                request=request,
            )
        transition_outcome = self._transition_outcome(outcome)
        self._set_status_from_display_transition(transition_outcome)
        return self._refresh_failed(
            focused_controls_use_workspace=False,
            transition_outcome=transition_outcome,
        )

    def _publish_fully_resolved_refresh(
        self,
        *,
        request: BatchDisplayRefreshRequest,
        resolution: BatchDisplayRequestResolution,
    ) -> tuple[Optional[BatchDisplayRefreshOutcome], BatchDisplayRequestResolution]:
        if not (
            resolution.coverage is BatchDisplayRequestCoverage.FULL
            and resolution.resolved_entries
        ):
            return None, resolution
        outcome = self.publish_resolved_batch_display_request(
            resolved_entries=resolution.resolved_entries,
            prefer_set=request.prefer_set_id,
            display_source=request.display_source,
            requested_show_set_ids=request.requested_show_set_ids,
            requested_labels_by_set_id=self._popup_labels_by_set_id(request.requested_show_set_ids),
        )
        if self._outcome_published(outcome):
            return (
                self._refresh_success(
                    display_outcome=outcome,
                    focused_controls_use_workspace=bool(resolution.focused_uses_workspace_controls),
                ),
                resolution,
            )
        if self._outcome_denied_by_active_completed_run(outcome):
            return (
                self._completed_run_display_noop_refresh_outcome_for_request(
                    request=request,
                    resolution=resolution,
                ),
                resolution,
            )
        transition_outcome = self._transition_outcome(outcome)
        if (
            isinstance(transition_outcome, DisplayTransitionOutcome)
            and transition_outcome.cause is DisplayTransitionCause.DISPLAY_MUTATION_FAILED
        ):
            self._set_status_from_display_transition(transition_outcome)
            return (
                self._refresh_failed(
                    focused_controls_use_workspace=bool(resolution.focused_uses_workspace_controls),
                    transition_outcome=transition_outcome,
                ),
                resolution,
            )
        return (
            None,
            BatchDisplayRequestResolution(
                resolved_entries=resolution.resolved_entries,
                unavailable_cause=self._cache_resolution_cause_for_transition(
                    transition_outcome,
                    default=resolution.unavailable_cause or DisplayTransitionCause.CACHE_RESULT_UNAVAILABLE,
                ),
                coverage=resolution.coverage,
                has_workspace_display_request=resolution.has_workspace_display_request,
                has_resolved_workspace_preview=resolution.has_resolved_workspace_preview,
                focused_uses_workspace_controls=resolution.focused_uses_workspace_controls,
                focused_has_resolved_entry=resolution.focused_has_resolved_entry,
            ),
        )

    @staticmethod
    def _can_publish_resolved_preview(
        *,
        request: BatchDisplayRefreshRequest,
        resolution: BatchDisplayRequestResolution,
    ) -> bool:
        return bool(
            resolution.resolved_entries
            and resolution.has_workspace_display_request
            and resolution.unavailable_cause in {
                DisplayTransitionCause.IN_FLIGHT_COVERAGE_UNAVAILABLE,
                DisplayTransitionCause.CACHE_RESULT_UNAVAILABLE,
            }
            and resolution.has_resolved_workspace_preview
            and (
                bool(resolution.focused_uses_workspace_controls)
                or ((not bool(request.focused_dirty)) and bool(resolution.focused_has_resolved_entry))
            )
        )

    def _publish_resolved_preview_refresh(
        self,
        *,
        request: BatchDisplayRefreshRequest,
        resolution: BatchDisplayRequestResolution,
    ) -> tuple[Optional[BatchDisplayRefreshOutcome], BatchDisplayRequestResolution]:
        if not self._can_publish_resolved_preview(request=request, resolution=resolution):
            return None, resolution
        outcome = self.publish_resolved_batch_display_request(
            resolved_entries=resolution.resolved_entries,
            prefer_set=request.prefer_set_id,
            display_source=request.display_source,
            requested_show_set_ids=request.requested_show_set_ids,
            requested_labels_by_set_id=self._popup_labels_by_set_id(request.requested_show_set_ids),
            unresolved_intent_set_ids=tuple(
                set_id
                for set_id in request.requested_show_set_ids
                if set_id not in {str(entry.set_id) for entry in resolution.resolved_entries if str(entry.set_id)}
            ),
            missing_intent_set_ids=tuple(
                set_id
                for set_id in request.requested_show_set_ids
                if set_id not in {str(entry.set_id) for entry in resolution.resolved_entries if str(entry.set_id)}
            ),
        )
        if self._outcome_published(outcome):
            return (
                self._refresh_success(
                    display_outcome=outcome,
                    focused_controls_use_workspace=bool(resolution.focused_uses_workspace_controls),
                ),
                resolution,
            )
        if self._outcome_denied_by_active_completed_run(outcome):
            return (
                self._completed_run_display_noop_refresh_outcome_for_request(
                    request=request,
                    resolution=resolution,
                ),
                resolution,
            )
        transition_outcome = self._transition_outcome(outcome)
        if (
            isinstance(transition_outcome, DisplayTransitionOutcome)
            and transition_outcome.cause is DisplayTransitionCause.DISPLAY_MUTATION_FAILED
        ):
            self._set_status_from_display_transition(transition_outcome)
            return (
                self._refresh_failed(
                    focused_controls_use_workspace=bool(resolution.focused_uses_workspace_controls),
                    transition_outcome=transition_outcome,
                ),
                resolution,
            )
        if resolution.unavailable_cause is DisplayTransitionCause.IN_FLIGHT_COVERAGE_UNAVAILABLE:
            transition_outcome = self._record_unpublished_display_request_outcome(
                cause=DisplayTransitionCause.IN_FLIGHT_COVERAGE_UNAVAILABLE,
                display_status=DisplayStatus.DISPLAY_DENIED,
            )
            self._set_status_from_display_transition(transition_outcome)
            return (
                self._refresh_failed(
                    focused_controls_use_workspace=request.focused_dirty,
                    transition_outcome=transition_outcome,
                ),
                resolution,
            )
        return (
            None,
            BatchDisplayRequestResolution(
                resolved_entries=resolution.resolved_entries,
                unavailable_cause=self._cache_resolution_cause_for_transition(
                    transition_outcome,
                    default=resolution.unavailable_cause or DisplayTransitionCause.CACHE_RESULT_UNAVAILABLE,
                ),
                coverage=resolution.coverage,
                has_workspace_display_request=resolution.has_workspace_display_request,
                has_resolved_workspace_preview=resolution.has_resolved_workspace_preview,
                focused_uses_workspace_controls=resolution.focused_uses_workspace_controls,
                focused_has_resolved_entry=resolution.focused_has_resolved_entry,
            ),
        )

    def _publish_active_explicit_refresh(
        self,
        *,
        request: BatchDisplayRefreshRequest,
        resolution: BatchDisplayRequestResolution,
    ) -> tuple[Optional[BatchDisplayRefreshOutcome], BatchDisplayRequestResolution]:
        can_try_active_cache = (
            (not resolution.has_workspace_display_request)
            and resolution.unavailable_cause in {
                DisplayTransitionCause.IN_FLIGHT_COVERAGE_UNAVAILABLE,
                DisplayTransitionCause.CACHE_RESULT_UNAVAILABLE,
            }
            and bool(request.active_cache_key)
        )
        if not can_try_active_cache:
            return None, resolution
        outcome = self._display_cached_active_request_scope(request=request)
        if self._outcome_published(outcome):
            self.reset_stale_cache_warning_status()
            return self._refresh_success(display_outcome=outcome, focused_controls_use_workspace=False), resolution
        if self._outcome_denied_by_active_completed_run(outcome):
            return (
                self._completed_run_display_noop_refresh_outcome_for_request(
                    request=request,
                    resolution=resolution,
                ),
                resolution,
            )
        if self._cache_resolution_cause_for_transition(
            self._transition_outcome(outcome),
            default=DisplayTransitionCause.CACHE_RESULT_UNAVAILABLE,
        ) is not DisplayTransitionCause.INVALID_CACHE_ENTRY:
            return None, resolution
        return (
            None,
            BatchDisplayRequestResolution(
                resolved_entries=resolution.resolved_entries,
                unavailable_cause=DisplayTransitionCause.INVALID_CACHE_ENTRY,
                coverage=resolution.coverage,
                has_workspace_display_request=resolution.has_workspace_display_request,
                has_resolved_workspace_preview=resolution.has_resolved_workspace_preview,
                focused_uses_workspace_controls=resolution.focused_uses_workspace_controls,
                focused_has_resolved_entry=resolution.focused_has_resolved_entry,
            ),
        )

    def _clear_unpublished_refresh(
        self,
        *,
        request: BatchDisplayRefreshRequest,
        resolution: BatchDisplayRequestResolution,
        requested_show_set_ids: Sequence[str],
    ) -> BatchDisplayRefreshOutcome:
        requested_ids = tuple(self._deduped_set_ids(requested_show_set_ids))
        scope_deauthorization = self._deauthorize_active_display_after_display_scope_refresh(
            request=request,
            requested_show_set_ids=requested_show_set_ids,
        )
        if scope_deauthorization is not None:
            return BatchDisplayRefreshOutcome(
                focused_controls_use_workspace=None,
                transition_outcome=scope_deauthorization,
            )
        if self._completed_run_display_transaction_active():
            transition_outcome = self._completed_run_denied_unavailable_outcome(
                request=request,
                resolution=resolution,
                requested_show_set_ids=requested_show_set_ids,
            )
            outcome = self._completed_run_display_noop_refresh_outcome_for_request(
                request=request,
                resolution=resolution,
            )
            return BatchDisplayRefreshOutcome(
                focused_controls_use_workspace=outcome.focused_controls_use_workspace,
                transition_outcome=transition_outcome,
            )
        if (not resolution.has_workspace_display_request) and (not request.active_cache_key):
            self.reset_stale_cache_warning_status()
            active_display_ids = self._active_plotted_display_set_ids()
            unavailable_ids = tuple(
                set_id for set_id in requested_ids if set_id not in active_display_ids
            )
            transition_outcome = self._record_unpublished_display_request_outcome(
                affected_set_ids=requested_ids,
                requested_show_set_ids=requested_ids,
                unresolved_intent_set_ids=unavailable_ids,
                missing_intent_set_ids=unavailable_ids,
                cause=DisplayTransitionCause.CACHE_RESULT_UNAVAILABLE,
                display_status=DisplayStatus.DISPLAY_DENIED,
            )
            return self._refresh_failed(
                focused_controls_use_workspace=False,
                transition_outcome=transition_outcome,
            )

        if resolution.unavailable_cause is DisplayTransitionCause.INVALID_CACHE_ENTRY:
            transition_outcome = self._record_unpublished_display_request_outcome(
                affected_set_ids=requested_ids,
                unresolved_intent_set_ids=requested_ids,
                cause=DisplayTransitionCause.INVALID_CACHE_ENTRY,
                display_status=DisplayStatus.DISPLAY_DENIED,
            )
            self._set_status_from_display_transition(transition_outcome)
            return self._refresh_failed(
                focused_controls_use_workspace=request.focused_dirty,
                transition_outcome=transition_outcome,
            )
        if resolution.unavailable_cause is DisplayTransitionCause.IN_FLIGHT_COVERAGE_UNAVAILABLE:
            transition_outcome = self._record_unpublished_display_request_outcome(
                affected_set_ids=requested_ids,
                unresolved_intent_set_ids=requested_ids,
                cause=DisplayTransitionCause.IN_FLIGHT_COVERAGE_UNAVAILABLE,
                display_status=DisplayStatus.DISPLAY_DENIED,
            )
            self._set_status_from_display_transition(transition_outcome)
            return self._refresh_failed(
                focused_controls_use_workspace=request.focused_dirty,
                transition_outcome=transition_outcome,
            )

        transition_outcome = self._record_unpublished_display_request_outcome(
            affected_set_ids=requested_ids,
            unresolved_intent_set_ids=requested_ids,
            missing_intent_set_ids=requested_ids,
            cause=DisplayTransitionCause.CACHE_RESULT_UNAVAILABLE,
            display_status=DisplayStatus.DISPLAY_DENIED,
        )
        self._set_status_from_display_transition(transition_outcome)
        return self._refresh_failed(
            focused_controls_use_workspace=bool(request.focused_dirty),
            transition_outcome=transition_outcome,
        )

    def _refresh_batch_display_request(
        self,
        request: BatchDisplayRefreshRequest,
    ) -> BatchDisplayRefreshOutcome:
        requested_show_set_ids = tuple(str(set_id) for set_id in (request.requested_show_set_ids or ()) if str(set_id))
        if not requested_show_set_ids:
            scope_deauthorization = self._deauthorize_active_display_after_display_scope_refresh(
                request=request,
                requested_show_set_ids=requested_show_set_ids,
            )
            if scope_deauthorization is not None:
                return BatchDisplayRefreshOutcome(
                    focused_controls_use_workspace=None,
                    transition_outcome=scope_deauthorization,
                )
            if self._completed_run_display_transaction_active():
                transition_outcome = self._completed_run_denied_unavailable_outcome(
                    request=request,
                    resolution=request.resolution,
                    requested_show_set_ids=requested_show_set_ids,
                )
                outcome = self._completed_run_display_noop_refresh_outcome_for_request(
                    request=request,
                    resolution=request.resolution,
                )
                return BatchDisplayRefreshOutcome(
                    focused_controls_use_workspace=outcome.focused_controls_use_workspace,
                    transition_outcome=transition_outcome,
                )
            self._clear_unpublished_batch_display_request()
            return BatchDisplayRefreshOutcome(
                focused_controls_use_workspace=bool(request.focused_set_dirty),
            )

        outcome = self._publish_fresh_explicit_dirty_refresh(request=request)
        if outcome is not None:
            return outcome

        resolution = request.resolution
        outcome, resolution = self._publish_fully_resolved_refresh(request=request, resolution=resolution)
        if outcome is not None:
            return outcome

        outcome, resolution = self._publish_resolved_preview_refresh(request=request, resolution=resolution)
        if outcome is not None:
            return outcome

        outcome, resolution = self._publish_active_explicit_refresh(request=request, resolution=resolution)
        if outcome is not None:
            return outcome

        return self._clear_unpublished_refresh(request=request, resolution=resolution, requested_show_set_ids=requested_show_set_ids)

    def refresh_display_from_request_scope(
        self,
        *,
        display_source: object | None = None,
    ) -> BatchDisplayRefreshOutcome:
        requested_show_set_ids = tuple(
            str(set_id) for set_id in (self._ui.requested_show_batch_set_ids() or ()) if str(set_id)
        )
        prefer = self._ui.focused_batch_set_id()
        focused_dirty = bool(self._ui.focused_show_request_is_dirty(requested_show_set_ids, prefer))
        request = BatchDisplayRefreshRequest(
            requested_show_set_ids=requested_show_set_ids,
            prefer_set_id=prefer,
            active_cache_key=str(self._ui.active_batch_cache_key() or ""),
            display_source=_coerce_display_refresh_source(display_source),
            focused_dirty=focused_dirty,
            focused_set_dirty=bool(self._ui.focused_batch_set_is_dirty()),
            fresh_explicit_cache_after_post_run_sync=(
                self._ui.show_request_uses_fresh_explicit_cache_after_post_run_sync(requested_show_set_ids)
                if requested_show_set_ids
                else False
            ),
            resolution=(
                self._ui.workspace_display_request_resolution(requested_show_set_ids)
                if requested_show_set_ids
                else BatchDisplayRequestResolution()
            ),
        )
        outcome = self._refresh_batch_display_request(request)
        self._ui.update_batch_row_controls_state()
        return outcome

    def _sync_plot_copy_metadata_from_transaction(
        self,
        *,
        primary_set_id: str,
        display_set_ids: Sequence[str],
    ) -> Optional[str]:
        try:
            if str(primary_set_id or "").strip():
                self._sync_main_plot_copy_labels(
                    primary_set_id=str(primary_set_id),
                    display_set_ids=list(display_set_ids),
                )
            else:
                self._sync_main_plot_copy_labels(primary_set_id="", display_set_ids=[])
        except Exception as exc:
            logger.exception("Failed to sync display transaction copy metadata: %s", exc)
            return "display_transaction_copy_metadata_failed"
        return None

    def _cached_batch_display_scope_availability(
        self,
        *,
        requested_show_set_ids: Sequence[str],
        snapshot: BatchCacheResultReadSnapshot,
    ) -> CachedBatchAvailability:
        if not str(snapshot.cache_key or ""):
            return CachedBatchAvailability([])
        return self._available_cached_batch_ids(
            requested_show_set_ids=requested_show_set_ids,
            snapshot=snapshot,
            require_completion_provenance=True,
        )

    def _cached_batch_display_scope_coverage(
        self,
        *,
        requested_show_set_ids: Sequence[str],
        snapshot: BatchCacheResultReadSnapshot,
    ) -> CachedBatchDisplayScopeCoverage:
        requested_show_ids = self._normalized_requested_show_batch_ids(requested_show_set_ids)
        availability = self._cached_batch_display_scope_availability(
            requested_show_set_ids=requested_show_set_ids,
            snapshot=snapshot,
        )
        if not requested_show_ids:
            unavailable_cause = (
                DisplayTransitionCause.INVALID_CACHE_ENTRY
                if availability.has_invalid_entry
                else DisplayTransitionCause.CACHE_RESULT_UNAVAILABLE
            )
            return CachedBatchDisplayScopeCoverage(
                requested_show_set_ids=[],
                available_ids=list(availability.available_ids),
                full_coverage=False,
                unavailable_cause=unavailable_cause,
            )
        full_coverage = (
            len(availability.available_ids) == len(requested_show_ids)
            and set(availability.available_ids) == set(requested_show_ids)
        )
        unavailable_cause = None
        if not full_coverage:
            unavailable_cause = (
                DisplayTransitionCause.INVALID_CACHE_ENTRY
                if availability.has_invalid_entry
                else DisplayTransitionCause.CACHE_RESULT_UNAVAILABLE
            )
        return CachedBatchDisplayScopeCoverage(
            requested_show_set_ids=list(requested_show_ids),
            available_ids=list(availability.available_ids),
            full_coverage=bool(full_coverage),
            unavailable_cause=unavailable_cause,
        )

    def publish_cached_batch_display_scope(
        self,
        *,
        cache_key: str,
        requested_show_set_ids: Sequence[str],
        prefer_set: Optional[str] = None,
        display_source: object | None = None,
    ) -> CachedBatchDisplayScopeOutcome:
        cache_key = str(cache_key or "")
        requested_ids = tuple(self._deduped_set_ids(requested_show_set_ids))
        if not cache_key:
            transition_outcome = self._record_unpublished_display_request_outcome(
                affected_set_ids=requested_ids,
                unresolved_intent_set_ids=requested_ids,
                cause=DisplayTransitionCause.DISPLAY_MUTATION_DENIED,
            )
            return CachedBatchDisplayScopeOutcome(
                transition_outcome=transition_outcome,
            )
        normalized_display_source = _coerce_display_refresh_source(display_source)
        display_transition = (
            _DISPLAY_TRANSITION_CACHED_REFRESH_REPLACE_ACTIVE
            if normalized_display_source
            in {DisplayRefreshSource.EXPLICIT_SHOW_REQUEST, DisplayRefreshSource.SLIDER_REPLAY}
            else _DISPLAY_TRANSITION_CACHED_REFRESH
        )
        display_denied = self._display_mutation_denied(
            transition=display_transition,
        )
        snapshot = self._ui.active_result_cache_read_snapshot(cache_key=cache_key)
        invalidated_ids = tuple(
            str(set_id)
            for set_id in (snapshot.invalidated_set_ids or ())
            if str(set_id)
        )
        coverage = self._cached_batch_display_scope_coverage(
            requested_show_set_ids=requested_show_set_ids,
            snapshot=snapshot,
        )
        cache_available_ids = tuple(str(set_id) for set_id in coverage.available_ids if str(set_id))
        active_entries_by_set_id = self._active_display_entries_by_requested_id(
            requested_ids,
            excluded_set_ids=invalidated_ids,
        )
        invalidated_requested_ids = tuple(
            set_id for set_id in requested_ids if set_id in set(invalidated_ids)
        )
        active_display_ids = set(self._active_plotted_display_set_ids())
        if (
            invalidated_requested_ids
            and not cache_available_ids
            and active_display_ids.intersection(invalidated_requested_ids)
        ):
            transition_outcome = self.clear_active_display_transaction(
                outcome_kind=DisplayTransitionOutcomeKind.DENIED,
                display_status=DisplayStatus.NO_COMPLETE_DISPLAYABLE_REQUEST_SCOPE,
                requested_show_set_ids=requested_ids,
                requested_labels_by_set_id={
                    str(set_id): str(self._ui.batch_name_for_id(str(set_id)) or set_id)
                    for set_id in requested_ids
                },
                attempted_display_set_ids=(),
                affected_set_ids=requested_ids,
                unresolved_intent_set_ids=invalidated_requested_ids,
                missing_intent_set_ids=invalidated_requested_ids,
                event_kind=display_transition.event_kind,
                cause=coverage.unavailable_cause or DisplayTransitionCause.INVALID_CACHE_ENTRY,
            )
            return CachedBatchDisplayScopeOutcome(
                transition_outcome=transition_outcome,
            )
        if active_entries_by_set_id and not cache_available_ids:
            missing_ids = tuple(
                set_id for set_id in requested_ids if set_id not in active_entries_by_set_id
            )
            transition_outcome = self._record_unpublished_display_request_outcome(
                cause=coverage.unavailable_cause or DisplayTransitionCause.CACHE_RESULT_UNAVAILABLE,
                affected_set_ids=requested_ids,
                requested_show_set_ids=requested_ids,
                requested_labels_by_set_id={
                    str(set_id): str(self._ui.batch_name_for_id(str(set_id)) or set_id)
                    for set_id in requested_ids
                },
                unresolved_intent_set_ids=missing_ids,
                missing_intent_set_ids=missing_ids,
            )
            return CachedBatchDisplayScopeOutcome(
                transition_outcome=transition_outcome,
            )
        if active_entries_by_set_id:
            merged_available_ids = tuple(
                self._deduped_set_ids(
                    (
                        *coverage.available_ids,
                        *(
                            set_id
                            for set_id in requested_ids
                            if set_id in active_entries_by_set_id
                        ),
                    )
                )
            )
            requested_id_set = set(coverage.requested_show_set_ids)
            merged_id_set = set(merged_available_ids)
            coverage = CachedBatchDisplayScopeCoverage(
                requested_show_set_ids=list(coverage.requested_show_set_ids),
                available_ids=list(merged_available_ids),
                full_coverage=bool(requested_id_set and requested_id_set <= merged_id_set),
                unavailable_cause=(
                    None
                    if requested_id_set and requested_id_set <= merged_id_set
                    else coverage.unavailable_cause
                ),
            )
        if not coverage.full_coverage:
            missing_ids = tuple(
                set_id
                for set_id in tuple(str(set_id) for set_id in coverage.requested_show_set_ids if str(set_id))
                if set_id not in {str(available_id) for available_id in coverage.available_ids if str(available_id)}
            )
            if not tuple(str(set_id) for set_id in coverage.available_ids if str(set_id)):
                transition_outcome = self._record_unpublished_display_request_outcome(
                    cause=coverage.unavailable_cause or DisplayTransitionCause.CACHE_RESULT_UNAVAILABLE,
                    affected_set_ids=tuple(
                        str(set_id) for set_id in coverage.requested_show_set_ids if str(set_id)
                    ),
                    unresolved_intent_set_ids=tuple(
                        str(set_id) for set_id in coverage.requested_show_set_ids if str(set_id)
                    ),
                    missing_intent_set_ids=missing_ids,
                )
                return CachedBatchDisplayScopeOutcome(
                    transition_outcome=transition_outcome,
                )

        primary = self._primary_cached_batch_id(
            available=coverage.available_ids,
            prefer_set=prefer_set,
            snapshot=snapshot,
        )

        entry_result = self._cache_entry_for_set_id(
            set_id=primary,
            snapshot=snapshot,
            require_completion_provenance=True,
        )
        entry = entry_result.entry
        if entry is None:
            cause = (
                DisplayTransitionCause.INVALID_CACHE_ENTRY
                if entry_result.state == "invalid"
                else DisplayTransitionCause.CACHE_RESULT_UNAVAILABLE
            )
            transition_outcome = self._record_unpublished_display_request_outcome(
                cause=cause,
                affected_set_ids=requested_ids,
                requested_show_set_ids=requested_ids,
                attempted_display_set_ids=(str(primary),),
                unresolved_intent_set_ids=requested_ids,
                missing_intent_set_ids=tuple(
                    set_id
                    for set_id in requested_ids
                    if set_id not in {str(available_id) for available_id in coverage.available_ids if str(available_id)}
                ),
            )
            return CachedBatchDisplayScopeOutcome(
                transition_outcome=transition_outcome,
            )
        t = entry["t"]
        series = entry["series"]

        primary_label = self._ui.batch_name_for_id(primary) or str(primary)
        transaction_entries_by_set_id = self._cached_batch_display_entries_by_set_id(
            available=coverage.available_ids,
            primary=primary,
            primary_entry=entry,
            snapshot=snapshot,
            active_entries_by_set_id=active_entries_by_set_id,
        )
        missing_owned_species = self._missing_owned_species_set_ids(transaction_entries_by_set_id)
        if missing_owned_species:
            displayable_entries_by_set_id = {
                str(set_id): entry_payload
                for set_id, entry_payload in transaction_entries_by_set_id.items()
                if str(set_id) not in set(missing_owned_species)
            }
            if not displayable_entries_by_set_id:
                if invalidated_requested_ids and active_display_ids.intersection(invalidated_requested_ids):
                    transition_outcome = self.clear_active_display_transaction(
                        outcome_kind=DisplayTransitionOutcomeKind.DENIED,
                        display_status=DisplayStatus.NO_COMPLETE_DISPLAYABLE_REQUEST_SCOPE,
                        requested_show_set_ids=requested_ids,
                        requested_labels_by_set_id={
                            str(set_id): str(self._ui.batch_name_for_id(str(set_id)) or set_id)
                            for set_id in requested_ids
                        },
                        attempted_display_set_ids=tuple(coverage.available_ids),
                        affected_set_ids=requested_ids,
                        unresolved_intent_set_ids=tuple(
                            self._deduped_set_ids((*missing_owned_species, *invalidated_requested_ids))
                        ),
                        missing_intent_set_ids=invalidated_requested_ids,
                        semantic_unavailable_set_ids=missing_owned_species,
                        event_kind=display_transition.event_kind,
                        cause=DisplayTransitionCause.SEMANTIC_METADATA_UNAVAILABLE,
                    )
                else:
                    transition_outcome = self._record_unpublished_display_request_outcome(
                        cause=DisplayTransitionCause.SEMANTIC_METADATA_UNAVAILABLE,
                        affected_set_ids=requested_ids,
                        requested_show_set_ids=requested_ids,
                        attempted_display_set_ids=tuple(coverage.available_ids),
                        unresolved_intent_set_ids=tuple(
                            self._deduped_set_ids((*missing_owned_species, *requested_ids))
                        ),
                        semantic_unavailable_set_ids=missing_owned_species,
                    )
                return CachedBatchDisplayScopeOutcome(
                    transition_outcome=transition_outcome,
                )
            transaction_entries_by_set_id = displayable_entries_by_set_id
            if str(primary) not in transaction_entries_by_set_id:
                primary = next(
                    str(set_id)
                    for set_id in coverage.available_ids
                    if str(set_id) in transaction_entries_by_set_id
                )
                entry = transaction_entries_by_set_id[str(primary)]
                t = entry["t"]
                series = entry["series"]
                primary_label = self._ui.batch_name_for_id(primary) or str(primary)
        overlays = self._cached_batch_overlays(
            transaction_entries_by_set_id=transaction_entries_by_set_id,
            primary=primary,
        )
        transaction_display_set_ids = tuple(str(set_id) for set_id in coverage.available_ids if str(set_id) in transaction_entries_by_set_id)
        unavailable_intent_set_ids = tuple(
            self._deduped_set_ids(
                (
                    *(
                        set_id
                        for set_id in requested_ids
                        if set_id not in {str(available_id) for available_id in coverage.available_ids if str(available_id)}
                    ),
                    *missing_owned_species,
                )
            )
        )
        missing_intent_set_ids = (
            tuple(
                set_id
                for set_id in requested_ids
                if set_id not in {str(available_id) for available_id in coverage.available_ids if str(available_id)}
            )
            if not coverage.full_coverage
            else ()
        )
        if display_denied:
            transition_outcome = self._record_unpublished_display_request_outcome(
                affected_set_ids=requested_ids,
                requested_show_set_ids=requested_ids,
                requested_labels_by_set_id={
                    str(set_id): str(self._ui.batch_name_for_id(str(set_id)) or set_id)
                    for set_id in requested_ids
                },
                attempted_display_set_ids=transaction_display_set_ids or coverage.available_ids,
                unresolved_intent_set_ids=unavailable_intent_set_ids or requested_ids,
                missing_intent_set_ids=missing_intent_set_ids,
                semantic_unavailable_set_ids=missing_owned_species,
                cause=DisplayTransitionCause.DISPLAY_MUTATION_DENIED,
            )
            return CachedBatchDisplayScopeOutcome(
                transition_outcome=transition_outcome,
            )

        return self._apply_batch_display_transaction(
            t=np.asarray(t, dtype=float),
            series=series,
            label=str(primary_label),
            overlays=overlays,
            metadata_applier=lambda plot, _active_transaction: self._apply_cached_batch_plot_metadata(
                plot=plot,
                cache_key=cache_key,
                available=coverage.available_ids,
                transaction_entries_by_set_id=transaction_entries_by_set_id,
                primary=primary,
                primary_label=str(primary_label),
                entry=entry,
                t=np.asarray(t, dtype=float),
                series={str(k): np.asarray(v, dtype=float) for k, v in series.items()},
            ),
            annotation_entry=entry,
            primary_set_id=str(primary),
            primary_label=str(primary_label),
            display_set_ids=list(transaction_display_set_ids),
            completion_provenance=entry.get("completion_provenance") if isinstance(entry, Mapping) else None,
            owned_species=self._owned_species_for_display_entry(entry),
            display_transition=display_transition,
            requested_show_set_ids=requested_ids,
            requested_labels_by_set_id={
                str(set_id): str(self._ui.batch_name_for_id(str(set_id)) or set_id)
                for set_id in requested_ids
            },
            unresolved_intent_set_ids=unavailable_intent_set_ids,
            missing_intent_set_ids=missing_intent_set_ids,
            semantic_unavailable_set_ids=missing_owned_species,
            cache_key=cache_key,
        )

    def publish_resolved_batch_display_request(
        self,
        *,
        resolved_entries: Sequence[ResolvedBatchDisplayRequestEntry],
        prefer_set: Optional[str] = None,
        display_source: object | None = None,
        requested_show_set_ids: Sequence[str] | None = None,
        requested_labels_by_set_id: Mapping[str, str] | None = None,
        unresolved_intent_set_ids: Sequence[str] = (),
        missing_intent_set_ids: Sequence[str] = (),
    ) -> CachedBatchDisplayScopeOutcome:
        requested_ids_for_outcome = tuple(
            self._deduped_set_ids(
                requested_show_set_ids
                if requested_show_set_ids is not None
                else tuple(str(resolved.set_id) for resolved in resolved_entries if str(resolved.set_id))
            )
        )
        requested_labels_for_outcome = {
            **{
                str(resolved.set_id): str(resolved.label or resolved.set_id)
                for resolved in resolved_entries
                if str(resolved.set_id)
            },
            **{
                str(set_id): str(label)
                for set_id, label in dict(requested_labels_by_set_id or {}).items()
                if str(set_id)
            },
        }

        def _resolved_unpublished_outcome(
            *,
            cause: DisplayTransitionCause,
            affected_set_ids: Sequence[str] = (),
            semantic_unavailable_set_ids: Sequence[str] = (),
        ) -> CachedBatchDisplayScopeOutcome:
            semantic_ids = tuple(self._deduped_set_ids(semantic_unavailable_set_ids))
            affected_ids = tuple(
                self._deduped_set_ids(requested_ids_for_outcome or affected_set_ids)
            )
            unresolved_ids = tuple(
                self._deduped_set_ids(
                    (
                        *unresolved_intent_set_ids,
                        *missing_intent_set_ids,
                        *semantic_ids,
                        *tuple(affected_set_ids),
                    )
                )
            )
            transition_outcome = self._record_unpublished_display_request_outcome(
                cause=cause,
                affected_set_ids=affected_ids,
                requested_show_set_ids=requested_ids_for_outcome or affected_ids,
                requested_labels_by_set_id=requested_labels_for_outcome,
                unresolved_intent_set_ids=unresolved_ids or affected_ids,
                missing_intent_set_ids=missing_intent_set_ids,
                semantic_unavailable_set_ids=semantic_ids,
            )
            return CachedBatchDisplayScopeOutcome(
                transition_outcome=transition_outcome,
            )

        if not resolved_entries:
            return _resolved_unpublished_outcome(
                cause=DisplayTransitionCause.CACHE_RESULT_UNAVAILABLE
            )

        entries_by_id = {
            str(resolved.set_id): resolved
            for resolved in resolved_entries
            if str(resolved.set_id)
        }
        if not entries_by_id:
            return _resolved_unpublished_outcome(
                cause=DisplayTransitionCause.CACHE_RESULT_UNAVAILABLE
            )
        semantic_unavailable_ids: tuple[str, ...] = ()
        displayable_entries_by_id = dict(entries_by_id)
        for resolved in entries_by_id.values():
            resolved_id = str(resolved.set_id)
            has_workspace_provenance = isinstance(resolved.workspace_preview_provenance, Mapping)
            has_completion_provenance = isinstance(resolved.entry.get("completion_provenance"), Mapping)
            if not (has_workspace_provenance or has_completion_provenance):
                return _resolved_unpublished_outcome(
                    cause=DisplayTransitionCause.INVALID_CACHE_ENTRY,
                    affected_set_ids=(resolved_id,),
                )
            if not self._owned_species_for_display_entry(resolved.entry):
                semantic_unavailable_ids = tuple(self._deduped_set_ids((*semantic_unavailable_ids, resolved_id)))
                displayable_entries_by_id.pop(resolved_id, None)
                continue
            if self._missing_owned_species_set_ids({resolved_id: resolved.entry}):
                semantic_unavailable_ids = tuple(self._deduped_set_ids((*semantic_unavailable_ids, resolved_id)))
                displayable_entries_by_id.pop(resolved_id, None)
                continue
            if resolved.canonical_entry is not None and not isinstance(
                resolved.canonical_entry.get("completion_provenance"),
                Mapping,
            ):
                return _resolved_unpublished_outcome(
                    cause=DisplayTransitionCause.INVALID_CACHE_ENTRY,
                    affected_set_ids=(resolved_id,),
                )
            if resolved.canonical_entry is not None and not self._owned_species_for_display_entry(
                resolved.canonical_entry,
            ):
                semantic_unavailable_ids = tuple(self._deduped_set_ids((*semantic_unavailable_ids, resolved_id)))
                displayable_entries_by_id.pop(resolved_id, None)
                continue
            if resolved.canonical_entry is not None and self._missing_owned_species_set_ids(
                {resolved_id: resolved.canonical_entry}
            ):
                semantic_unavailable_ids = tuple(self._deduped_set_ids((*semantic_unavailable_ids, resolved_id)))
                displayable_entries_by_id.pop(resolved_id, None)
                continue

        entries_by_id = displayable_entries_by_id
        if not entries_by_id:
            return _resolved_unpublished_outcome(
                cause=DisplayTransitionCause.SEMANTIC_METADATA_UNAVAILABLE,
                affected_set_ids=semantic_unavailable_ids,
                semantic_unavailable_set_ids=semantic_unavailable_ids,
            )
        resolved_entries = tuple(entries_by_id.values())
        unresolved_outcome_ids = tuple(
            self._deduped_set_ids((*unresolved_intent_set_ids, *semantic_unavailable_ids))
        )

        primary = None
        preferred_id = self._normalize_batch_set_id(str(prefer_set or "")) if prefer_set is not None else None
        if preferred_id:
            primary = entries_by_id.get(str(preferred_id))
        if primary is None:
            focused_id = str(self._ui.focused_batch_set_id() or "")
            primary = entries_by_id.get(focused_id)
        if primary is None:
            primary = next(iter(entries_by_id.values()))

        overlays = [
            self._display_overlay_entry(
                label=resolved.label,
                entry=resolved.entry,
                set_id=resolved.set_id,
                layer_kind="result",
                layer_id=f"result:{resolved.set_id}",
            )
            for resolved in resolved_entries
            if str(resolved.set_id) != str(primary.set_id)
        ]
        reference_overlay_visible = self._reference_overlay_visible_for_publication()
        for resolved in resolved_entries:
            if resolved.canonical_entry is None:
                continue
            overlays.append(
                self._display_overlay_entry(
                    label=f"{resolved.label} [ref]",
                    entry=resolved.canonical_entry,
                    set_id=resolved.set_id,
                    layer_kind="reference",
                    layer_id=f"reference:{resolved.set_id}",
                    visible=reference_overlay_visible,
                )
            )
        transaction_display_set_ids = tuple(str(resolved.set_id) for resolved in resolved_entries if str(resolved.set_id))
        workspace_provenance_by_set_id = {
            str(resolved.set_id): dict(resolved.workspace_preview_provenance)
            for resolved in resolved_entries
            if str(resolved.set_id) and isinstance(resolved.workspace_preview_provenance, Mapping)
        }
        display_transition = (
            _DISPLAY_TRANSITION_WORKSPACE_PREVIEW
            if workspace_provenance_by_set_id
            else _DISPLAY_TRANSITION_RESOLVED_REFRESH
        )
        normalized_display_source = _coerce_display_refresh_source(display_source)
        if normalized_display_source in {
            DisplayRefreshSource.EXPLICIT_SHOW_REQUEST,
            DisplayRefreshSource.SLIDER_REPLAY,
        }:
            display_transition = (
                _DISPLAY_TRANSITION_WORKSPACE_PREVIEW_REPLACE_ACTIVE
                if display_transition is _DISPLAY_TRANSITION_WORKSPACE_PREVIEW
                else _DISPLAY_TRANSITION_RESOLVED_REFRESH_REPLACE_ACTIVE
            )
        display_denied = self._display_mutation_denied(
            transition=display_transition,
        )
        if display_denied:
            transition_outcome = self._record_unpublished_display_request_outcome(
                affected_set_ids=requested_ids_for_outcome or transaction_display_set_ids,
                requested_show_set_ids=requested_ids_for_outcome or transaction_display_set_ids,
                requested_labels_by_set_id=requested_labels_for_outcome,
                attempted_display_set_ids=transaction_display_set_ids,
                unresolved_intent_set_ids=requested_ids_for_outcome or transaction_display_set_ids,
                missing_intent_set_ids=missing_intent_set_ids,
                semantic_unavailable_set_ids=semantic_unavailable_ids,
                cause=DisplayTransitionCause.DISPLAY_MUTATION_DENIED,
            )
            return CachedBatchDisplayScopeOutcome(
                transition_outcome=transition_outcome,
            )
        return self._apply_batch_display_transaction(
            t=np.asarray(primary.entry["t"], dtype=float),
            series=primary.entry.get("series") or {},
            label=str(primary.label),
            overlays=overlays,
            metadata_applier=lambda plot, _active_transaction: self._apply_resolved_batch_plot_metadata(
                plot=plot,
                resolved_entries=list(resolved_entries),
                primary=primary,
            ),
            annotation_entry=primary.entry,
            primary_set_id=str(primary.set_id),
            primary_label=str(primary.label),
            display_set_ids=transaction_display_set_ids,
            workspace_preview_provenance_by_set_id=workspace_provenance_by_set_id,
            completion_provenance=(
                primary.entry.get("completion_provenance")
                if isinstance(primary.entry.get("completion_provenance"), Mapping)
                else None
            ),
            owned_species=self._owned_species_for_display_entry(primary.entry),
            display_transition=display_transition,
            requested_show_set_ids=(
                tuple(self._deduped_set_ids(requested_show_set_ids))
                if requested_show_set_ids is not None
                else transaction_display_set_ids
            ),
            requested_labels_by_set_id={
                **{
                    str(resolved.set_id): str(resolved.label or resolved.set_id)
                    for resolved in resolved_entries
                    if str(resolved.set_id)
                },
                **{
                    str(set_id): str(label)
                    for set_id, label in dict(requested_labels_by_set_id or {}).items()
                    if str(set_id)
                },
            },
            unresolved_intent_set_ids=unresolved_outcome_ids,
            missing_intent_set_ids=missing_intent_set_ids,
            semantic_unavailable_set_ids=semantic_unavailable_ids,
        )

    def publish_completed_run_display_transaction(
        self,
        transaction: CompletedRunDisplayTransaction,
    ) -> SimulationCompletionDisplayOutcome:
        intent_ids = tuple(str(set_id) for set_id in transaction.intent.requested_show_set_ids if str(set_id))
        initial_affected_ids = tuple(self._deduped_set_ids(transaction.display_set_ids or intent_ids))
        completion_entries = tuple(transaction.completion_entries or ())
        if not completion_entries:
            return self._simulation_no_display_outcome(
                DisplayTransitionCause.IN_FLIGHT_COVERAGE_UNAVAILABLE,
                affected_set_ids=initial_affected_ids,
                requested_show_set_ids=intent_ids,
                requested_labels_by_set_id=transaction.intent.labels_by_set_id,
                attempted_display_set_ids=initial_affected_ids,
                unresolved_intent_set_ids=intent_ids,
                missing_intent_set_ids=intent_ids,
                outcome_kind=DisplayTransitionOutcomeKind.FAILED,
                display_status=DisplayStatus.NO_COMPLETE_DISPLAYABLE_REQUEST_SCOPE,
            )
        expected_ids = tuple(str(set_id) for set_id in transaction.display_set_ids if str(set_id))
        completed_ids = tuple(str(entry.set_id) for entry in completion_entries if str(entry.set_id))
        failed_ids = tuple(str(set_id) for set_id in transaction.failed_set_ids if str(set_id))
        unresolved_intent_ids = tuple(
            self._deduped_set_ids(
                (
                    *tuple(str(set_id) for set_id in transaction.unresolved_intent_set_ids if str(set_id)),
                    *failed_ids,
                )
            )
        )
        missing_intent_ids = tuple(
            self._deduped_set_ids(
                tuple(str(set_id) for set_id in transaction.missing_intent_set_ids if str(set_id))
            )
        )
        failed_intent_ids = tuple(
            self._deduped_set_ids(
                (
                    *tuple(str(set_id) for set_id in transaction.failed_intent_set_ids if str(set_id)),
                    *failed_ids,
                )
            )
        )
        semantic_unavailable_ids = tuple(
            self._deduped_set_ids(
                tuple(str(set_id) for set_id in transaction.semantic_unavailable_set_ids if str(set_id))
            )
        )
        if expected_ids and intent_ids and not set(expected_ids).issubset(set(intent_ids)):
            missing_intent_ids = tuple(set_id for set_id in expected_ids if set_id not in set(intent_ids))
            return self.publish_completed_run_display_unavailable(
                cause=DisplayTransitionCause.IN_FLIGHT_COVERAGE_UNAVAILABLE,
                affected_set_ids=missing_intent_ids,
                requested_show_set_ids=intent_ids,
                requested_labels_by_set_id=transaction.intent.labels_by_set_id,
                attempted_display_set_ids=completed_ids,
                unresolved_intent_set_ids=missing_intent_ids,
                missing_intent_set_ids=missing_intent_ids,
            )
        known_expected_ids = tuple(
            normalized for set_id in expected_ids if (normalized := self._normalize_batch_set_id(set_id))
        )
        raw_run_target_ids = tuple(str(set_id) for set_id in transaction.intent.run_target_set_ids if str(set_id))
        if raw_run_target_ids and expected_ids and len(known_expected_ids) != len(expected_ids):
            missing_live_ids = tuple(
                set_id for set_id in expected_ids if self._normalize_batch_set_id(set_id) is None
            )
            return self.publish_completed_run_display_unavailable(
                cause=DisplayTransitionCause.IN_FLIGHT_COVERAGE_UNAVAILABLE,
                affected_set_ids=missing_live_ids or expected_ids,
                requested_show_set_ids=intent_ids,
                requested_labels_by_set_id=transaction.intent.labels_by_set_id,
                attempted_display_set_ids=completed_ids,
                unresolved_intent_set_ids=missing_live_ids or expected_ids,
                missing_intent_set_ids=missing_live_ids or expected_ids,
            )
        if not expected_ids:
            return self._simulation_no_display_outcome(
                DisplayTransitionCause.DISPLAY_MUTATION_DENIED,
                affected_set_ids=intent_ids,
                requested_show_set_ids=intent_ids,
                requested_labels_by_set_id=transaction.intent.labels_by_set_id,
                attempted_display_set_ids=completed_ids,
                unresolved_intent_set_ids=intent_ids,
                outcome_kind=DisplayTransitionOutcomeKind.FAILED,
                display_status=DisplayStatus.NO_COMPLETE_DISPLAYABLE_REQUEST_SCOPE,
            )
        if expected_ids != completed_ids:
            return self._simulation_no_display_outcome(
                DisplayTransitionCause.IN_FLIGHT_COVERAGE_UNAVAILABLE,
                affected_set_ids=expected_ids,
                requested_show_set_ids=intent_ids,
                requested_labels_by_set_id=transaction.intent.labels_by_set_id,
                attempted_display_set_ids=completed_ids,
                unresolved_intent_set_ids=expected_ids,
                missing_intent_set_ids=expected_ids,
                outcome_kind=DisplayTransitionOutcomeKind.FAILED,
                display_status=DisplayStatus.NO_COMPLETE_DISPLAYABLE_REQUEST_SCOPE,
            )
        entries_by_id = {str(entry.set_id): entry for entry in completion_entries}
        primary_id = str(transaction.display_primary_set_id or "").strip()
        primary = entries_by_id.get(primary_id)
        if primary is None:
            return self._simulation_no_display_outcome(
                DisplayTransitionCause.DISPLAY_MUTATION_DENIED,
                affected_set_ids=expected_ids,
                requested_show_set_ids=intent_ids,
                requested_labels_by_set_id=transaction.intent.labels_by_set_id,
                attempted_display_set_ids=completed_ids,
                unresolved_intent_set_ids=expected_ids,
                missing_intent_set_ids=expected_ids,
                outcome_kind=DisplayTransitionOutcomeKind.FAILED,
                display_status=DisplayStatus.NO_COMPLETE_DISPLAYABLE_REQUEST_SCOPE,
            )

        def semantic_unavailable(
            semantic_ids: Sequence[str],
        ) -> SimulationCompletionDisplayOutcome:
            semantic_ids = tuple(self._deduped_set_ids(tuple(str(set_id) for set_id in semantic_ids if str(set_id))))
            unresolved = tuple(self._deduped_set_ids((*unresolved_intent_ids, *failed_ids, *semantic_ids)))
            missing_ids = tuple(self._deduped_set_ids(missing_intent_ids))
            return self.publish_completed_run_display_unavailable(
                cause=DisplayTransitionCause.SEMANTIC_METADATA_UNAVAILABLE,
                affected_set_ids=tuple(self._deduped_set_ids((*unresolved, *missing_ids))) or semantic_ids,
                requested_show_set_ids=intent_ids,
                requested_labels_by_set_id=transaction.intent.labels_by_set_id,
                attempted_display_set_ids=completed_ids,
                unresolved_intent_set_ids=unresolved,
                missing_intent_set_ids=missing_ids,
                failed_intent_set_ids=failed_intent_ids,
                semantic_unavailable_set_ids=tuple(self._deduped_set_ids((*semantic_unavailable_ids, *semantic_ids))),
            )

        displayable_entries: list[CompletionDisplayEntry] = []
        for completion_entry in completion_entries:
            if not isinstance(completion_entry.completion_provenance, Mapping):
                return self._simulation_no_display_outcome(
                    DisplayTransitionCause.NO_DISPLAYABLE_COMPLETION_RESULTS,
                    affected_set_ids=tuple(
                        self._deduped_set_ids(
                            (
                                str(completion_entry.set_id),
                                *missing_intent_ids,
                                *failed_intent_ids,
                                *semantic_unavailable_ids,
                            )
                        )
                    ),
                    requested_show_set_ids=intent_ids,
                    requested_labels_by_set_id=transaction.intent.labels_by_set_id,
                    attempted_display_set_ids=completed_ids,
                    unresolved_intent_set_ids=tuple(
                        self._deduped_set_ids(
                            (str(completion_entry.set_id), *missing_intent_ids, *failed_intent_ids)
                        )
                    ),
                    missing_intent_set_ids=missing_intent_ids,
                    failed_intent_set_ids=failed_intent_ids,
                    semantic_unavailable_set_ids=semantic_unavailable_ids,
                    outcome_kind=DisplayTransitionOutcomeKind.FAILED,
                    display_status=DisplayStatus.NO_COMPLETE_DISPLAYABLE_REQUEST_SCOPE,
                )
            if completion_entry.t is None or not isinstance(completion_entry.series, Mapping):
                return self._simulation_no_display_outcome(
                    DisplayTransitionCause.NO_DISPLAYABLE_COMPLETION_RESULTS,
                    affected_set_ids=tuple(
                        self._deduped_set_ids(
                            (
                                str(completion_entry.set_id),
                                *missing_intent_ids,
                                *failed_intent_ids,
                                *semantic_unavailable_ids,
                            )
                        )
                    ),
                    requested_show_set_ids=intent_ids,
                    requested_labels_by_set_id=transaction.intent.labels_by_set_id,
                    attempted_display_set_ids=completed_ids,
                    unresolved_intent_set_ids=tuple(
                        self._deduped_set_ids(
                            (str(completion_entry.set_id), *missing_intent_ids, *failed_intent_ids)
                        )
                    ),
                    missing_intent_set_ids=missing_intent_ids,
                    failed_intent_set_ids=failed_intent_ids,
                    semantic_unavailable_set_ids=semantic_unavailable_ids,
                    outcome_kind=DisplayTransitionOutcomeKind.FAILED,
                    display_status=DisplayStatus.NO_COMPLETE_DISPLAYABLE_REQUEST_SCOPE,
                )
            if not tuple(str(name) for name in completion_entry.owned_species if str(name)):
                semantic_unavailable_ids = tuple(
                    self._deduped_set_ids((*semantic_unavailable_ids, str(completion_entry.set_id)))
                )
                continue
            if not self._completion_entry_matches_intent_owned_species(transaction, completion_entry):
                semantic_unavailable_ids = tuple(
                    self._deduped_set_ids((*semantic_unavailable_ids, str(completion_entry.set_id)))
                )
                continue
            displayable_entries.append(completion_entry)
        semantic_series_by_set_id: Dict[str, Dict[str, object]] = {}
        for completion_entry in displayable_entries:
            semantic_series = self._completion_entry_semantic_series(completion_entry)
            if semantic_series is None:
                semantic_unavailable_ids = tuple(
                    self._deduped_set_ids((*semantic_unavailable_ids, str(completion_entry.set_id)))
                )
                continue
            semantic_series_by_set_id[str(completion_entry.set_id)] = semantic_series
        displayable_entries = [
            entry
            for entry in displayable_entries
            if str(entry.set_id) in semantic_series_by_set_id
        ]
        if not displayable_entries:
            return semantic_unavailable(semantic_unavailable_ids or expected_ids)
        if str(primary.set_id) not in {str(entry.set_id) for entry in displayable_entries}:
            primary = displayable_entries[0]
        overlays = [
            self._display_overlay_entry(
                label=completion_entry.label,
                entry={
                    **completion_entry.to_display_payload(),
                    "series": semantic_series_by_set_id[str(completion_entry.set_id)],
                },
                set_id=completion_entry.set_id,
                layer_kind="result",
                layer_id=f"result:{completion_entry.set_id}",
                owned_species=tuple(str(name) for name in completion_entry.owned_species if str(name)),
            )
            for completion_entry in completion_entries
            if str(completion_entry.set_id) != str(primary.set_id)
            and str(completion_entry.set_id) in semantic_series_by_set_id
        ]
        primary_owned_species = tuple(str(name) for name in primary.owned_species if str(name))
        if not primary_owned_species:
            return semantic_unavailable((str(primary.set_id),))
        primary_payload = primary.to_display_payload()
        primary_series = semantic_series_by_set_id[str(primary.set_id)]
        primary_payload["series"] = primary_series
        displayable_ids = tuple(str(entry.set_id) for entry in displayable_entries if str(entry.set_id))
        outcome = self._apply_batch_display_transaction(
            t=np.asarray(primary.t, dtype=float),
            series=primary_series,
            label=str(primary.label),
            overlays=overlays,
            metadata_applier=lambda plot, active_transaction: self._apply_completed_run_plot_metadata(
                plot=plot,
                active_display_transaction=active_transaction,
                completion_entries=list(displayable_entries),
                primary=primary,
                semantic_series_by_set_id=semantic_series_by_set_id,
            ),
            annotation_entry=primary_payload,
            primary_set_id=str(primary.set_id),
            primary_label=str(primary.label),
            display_set_ids=displayable_ids,
            completion_provenance=primary.completion_provenance,
            owned_species=primary_owned_species,
            display_transition=_DISPLAY_TRANSITION_COMPLETED_RUN_FINAL,
            requested_show_set_ids=transaction.intent.requested_show_set_ids,
            requested_labels_by_set_id=transaction.intent.labels_by_set_id,
            run_target_set_ids=transaction.intent.run_target_set_ids,
            unresolved_intent_set_ids=tuple(
                self._deduped_set_ids((*unresolved_intent_ids, *semantic_unavailable_ids))
            ),
            missing_intent_set_ids=missing_intent_ids,
            failed_intent_set_ids=failed_intent_ids,
            semantic_unavailable_set_ids=semantic_unavailable_ids,
            cache_key=transaction.intent.cache_key,
            run_id=transaction.intent.run_id,
            request_id=transaction.intent.request_id,
        )
        return SimulationCompletionDisplayOutcome(
            transition_outcome=outcome.transition_outcome,
        )

    @staticmethod
    def _fresh_preview_entry_display_series(
        entry: FreshPreviewDisplayEntry,
    ) -> Optional[Dict[str, object]]:
        raw_series = entry.series if isinstance(entry.series, Mapping) else {}
        cleaned: Dict[str, object] = {}
        for raw_name, raw_values in raw_series.items():
            name = str(raw_name)
            if name:
                cleaned[name] = raw_values
        return cleaned or None

    def publish_fresh_preview_display(
        self,
        transaction: FreshPreviewDisplayTransaction,
    ) -> SimulationCompletionDisplayOutcome:
        cache_key = str(transaction.cache_key or "").strip()
        if not cache_key or transaction.request_id is None:
            return self._simulation_no_display_outcome(DisplayTransitionCause.DISPLAY_MUTATION_DENIED)
        display_source = _coerce_display_refresh_source(transaction.display_source)
        entries = tuple(transaction.entries or ())
        if not entries:
            return self._simulation_no_display_outcome(DisplayTransitionCause.IN_FLIGHT_COVERAGE_UNAVAILABLE)
        expected_ids = tuple(str(set_id) for set_id in transaction.display_set_ids if str(set_id))
        target_ids = tuple(str(set_id) for set_id in transaction.target_set_ids if str(set_id))
        if not expected_ids:
            return self._simulation_no_display_outcome(DisplayTransitionCause.DISPLAY_MUTATION_DENIED)
        if not target_ids or set(target_ids) != set(expected_ids):
            return self._simulation_no_display_outcome(
                DisplayTransitionCause.DISPLAY_MUTATION_DENIED,
                affected_set_ids=expected_ids,
            )
        entries_by_id = {str(entry.set_id): entry for entry in entries if str(entry.set_id)}
        if set(expected_ids) != set(entries_by_id) or set(target_ids) != set(entries_by_id):
            return self._simulation_no_display_outcome(
                DisplayTransitionCause.IN_FLIGHT_COVERAGE_UNAVAILABLE,
                affected_set_ids=expected_ids,
            )
        primary_id = str(transaction.display_primary_set_id or "").strip()
        primary = entries_by_id.get(primary_id)
        if primary is None:
            return self._simulation_no_display_outcome(
                DisplayTransitionCause.DISPLAY_MUTATION_DENIED,
                affected_set_ids=expected_ids,
            )

        series_by_set_id: Dict[str, Dict[str, object]] = {}
        for entry in entries:
            if entry.t is None:
                return self._simulation_no_display_outcome(
                    DisplayTransitionCause.NO_DISPLAYABLE_PREVIEW_RESULTS,
                    affected_set_ids=(str(entry.set_id),),
                )
            display_series = self._fresh_preview_entry_display_series(entry)
            if display_series is None:
                return self._simulation_no_display_outcome(
                    DisplayTransitionCause.NO_DISPLAYABLE_PREVIEW_RESULTS,
                    affected_set_ids=(str(entry.set_id),),
                )
            series_by_set_id[str(entry.set_id)] = display_series

        overlays = [
            self._display_overlay_entry(
                label=entry.label,
                entry={
                    **entry.to_display_payload(),
                    "series": series_by_set_id[str(entry.set_id)],
                },
                set_id=entry.set_id,
                layer_kind="result",
                layer_id=f"result:{entry.set_id}",
                owned_species=tuple(str(name) for name in entry.owned_species if str(name)),
            )
            for entry in entries
            if str(entry.set_id) != str(primary.set_id)
        ]
        workspace_provenance_by_set_id = {
            str(entry.set_id): dict(entry.workspace_preview_provenance)
            for entry in entries
            if str(entry.set_id) and isinstance(entry.workspace_preview_provenance, Mapping)
        }
        if set(workspace_provenance_by_set_id) != set(expected_ids):
            return self._simulation_no_display_outcome(
                DisplayTransitionCause.IN_FLIGHT_COVERAGE_UNAVAILABLE,
                affected_set_ids=expected_ids,
            )
        primary_series = series_by_set_id[str(primary.set_id)]
        outcome = self._apply_batch_display_transaction(
            t=np.asarray(primary.t, dtype=float),
            series=primary_series,
            label=str(primary.label),
            overlays=overlays,
            metadata_applier=lambda plot, _active_transaction: self._apply_direct_completion_plot_metadata(
                plot=plot,
                t=np.asarray(primary.t, dtype=float),
                series=primary_series,
                display_label=str(primary.label or primary.set_id or "Preview"),
                algebra_scalars=primary.algebra_scalars,
                layer_id=f"result:{primary.set_id}" if primary.set_id else "result:preview",
                set_id=str(primary.set_id or ""),
            ),
            annotation_entry={"solver_provenance": primary.solver_provenance},
            primary_set_id=str(primary.set_id),
            primary_label=str(primary.label),
            display_set_ids=expected_ids,
            workspace_preview_provenance_by_set_id=workspace_provenance_by_set_id,
            completion_provenance=primary.completion_provenance,
            owned_species=tuple(str(name) for name in primary.owned_species if str(name)),
            display_transition=(
                _DISPLAY_TRANSITION_FRESH_PREVIEW_REPLACE_ACTIVE
                if display_source
                in {DisplayRefreshSource.EXPLICIT_SHOW_REQUEST, DisplayRefreshSource.SLIDER_REPLAY}
                else _DISPLAY_TRANSITION_FRESH_PREVIEW
            ),
            requested_show_set_ids=transaction.requested_show_set_ids or transaction.display_set_ids,
            requested_labels_by_set_id=transaction.requested_labels_by_set_id,
            run_target_set_ids=transaction.target_set_ids,
            cache_key=transaction.cache_key,
            run_id=transaction.run_id,
            request_id=transaction.request_id,
        )
        return SimulationCompletionDisplayOutcome(
            transition_outcome=outcome.transition_outcome,
        )

    def publish_direct_completion_result(
        self,
        *,
        t: np.ndarray,
        series: Dict[str, np.ndarray],
        batch_set: str | None,
        batch_set_id: str | None,
        algebra_scalars: Mapping[str, object] | None,
        direct_completion_provenance: Mapping[str, Any],
        solver_provenance: Mapping[str, Any] | None = None,
    ) -> SimulationCompletionDisplayOutcome:
        set_id = str(batch_set_id or "").strip()
        set_name = str(batch_set or "").strip()
        primary_layer_id = f"result:{set_id}" if set_id else "result:live"
        display_label = set_name or set_id or "Results"
        owned_species = self._direct_completion_owned_species(
            series=series,
            direct_completion_provenance=direct_completion_provenance,
        )
        if not owned_species:
            return self._simulation_no_display_outcome(
                DisplayTransitionCause.SEMANTIC_METADATA_UNAVAILABLE,
                affected_set_ids=([set_id] if set_id else []),
                unresolved_intent_set_ids=([set_id] if set_id else []),
                semantic_unavailable_set_ids=([set_id] if set_id else []),
                outcome_kind=DisplayTransitionOutcomeKind.FAILED,
                display_status=DisplayStatus.NO_COMPLETE_DISPLAYABLE_REQUEST_SCOPE,
            )
        outcome = self._apply_batch_display_transaction(
            t=np.asarray(t, dtype=float),
            series=series,
            label=(set_name or display_label),
            overlays=[],
            metadata_applier=lambda plot, _active_transaction: self._apply_direct_completion_plot_metadata(
                plot=plot,
                t=np.asarray(t, dtype=float),
                series=series,
                display_label=display_label,
                algebra_scalars=algebra_scalars,
                layer_id=primary_layer_id,
                set_id=set_id,
            ),
            annotation_entry={"solver_provenance": solver_provenance},
            primary_set_id=set_id,
            primary_label=set_name,
            display_set_ids=([set_id] if set_id else []),
            completion_provenance=direct_completion_provenance,
            owned_species=owned_species,
            display_transition=_DISPLAY_TRANSITION_DIRECT_RAW,
            run_target_set_ids=([set_id] if set_id else []),
        )
        return SimulationCompletionDisplayOutcome(
            transition_outcome=outcome.transition_outcome,
        )

    def _set_plot_data(
        self,
        t: np.ndarray,
        series: Dict[str, np.ndarray],
        *,
        label: Optional[str] = None,
        primary_set_id: Optional[str] = None,
        layer_id: Optional[str] = None,
        overlays: Optional[Sequence[Dict[str, object]]] = None,
        owned_species: Optional[Sequence[str]] = None,
    ) -> bool:
        """Set simulation data to plot."""
        try:
            self._ui.set_main_plot_data(
                t,
                series,
                label=label,
                primary_set_id=primary_set_id,
                layer_id=layer_id,
                overlays=overlays,
                owned_species=owned_species,
            )
            return True
        except Exception as exc:
            logger.warning("Failed to set data: %s", exc, exc_info=True)
            QtWidgets.QMessageBox.warning(self._ui.parent, "Error", f"Failed to set data: {exc}")
            return False

    def _commit_successful_plot_display(
        self,
        *,
        t: np.ndarray,
        series: Mapping[str, Any],
        transition_outcome: DisplayTransitionOutcome,
    ) -> None:
        try:
            self._ui.show_simulation_tab()
            self._ui.refresh_simulation_plot_views()
            self._ui.schedule_main_plot_refresh((50, 100))
            self._set_status_from_display_transition(transition_outcome)
            logger.info("Data set: %s species, %s points", int(len(series)), int(len(t)))
        except Exception as exc:
            logger.exception("Failed to apply post-commit plot display UI refresh: %s", exc)

    def _apply_intervention_annotations(self, *, plot: ResultsDisplayPlotPort, entry: Mapping[str, Any]) -> None:
        solver_provenance = entry.get("solver_provenance") if isinstance(entry, Mapping) else None
        plot.set_intervention_annotations_from_provenance(
            solver_provenance if isinstance(solver_provenance, Mapping) else None
        )
