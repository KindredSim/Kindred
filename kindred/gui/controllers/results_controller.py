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
    display_metadata_for_entry,
    owned_species_for_display_entry,
    series_for_display_species,
)
from kindred.gui.controllers.results_display_projections import (
    build_copy_all_export_plan,
    build_main_plot_csv_export,
    cache_resolution_cause_for_transition,
    display_transaction_provenance_payload,
    display_transition_status_text,
    ordered_display_transaction_metadata,
    plot_display_layers_payload,
    stats_results_map_from_display_transaction,
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
    DisplayProjectionState,
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
    PlotDisplayLayersPayload,
    PlotLayerKind,
    ResolvedBatchDisplayRequestEntry,
    SimulationCompletionDisplayOutcome,
)

logger = logging.getLogger(__name__)

__all__ = ["ResultsController"]


class ResultsDisplayPlotPort(Protocol):
    def set_scalar_values(self, scalars: Dict[str, object]) -> None: ...
    def set_display_layers(self, payload: PlotDisplayLayersPayload) -> None: ...
    def set_statistics_results(self, results: Dict[str, object], *, prefer: str) -> None: ...
    def stats_table(self) -> object: ...
    def overlay_snapshot(self) -> Dict[str, object]: ...
    def clear_display_transaction_state(self, *, preserve_y_selection_state: bool = False) -> None: ...
    def transaction_export_axis_state(self, scope: str) -> Dict[str, object]: ...
    def intervention_annotation_state(self) -> Dict[str, object]: ...
    def set_intervention_annotations_from_provenance(self, provenance: Mapping[str, object] | None) -> None: ...


def _coerce_display_refresh_source(source: object | None) -> DisplayRefreshSource:
    if isinstance(source, DisplayRefreshSource):
        return source
    raw = str(source or "").strip()
    if raw:
        for candidate in DisplayRefreshSource:
            if raw == candidate.value or raw == candidate.name:
                return candidate
    return DisplayRefreshSource.INCIDENTAL_REFRESH


def _display_transition_published(outcome: SimulationCompletionDisplayOutcome | None) -> bool:
    transition = outcome.transition_outcome if outcome is not None else None
    return (
        isinstance(transition, DisplayTransitionOutcome)
        and transition.kind is DisplayTransitionOutcomeKind.PUBLISHED
    )


def _display_transition_log_reason(outcome: SimulationCompletionDisplayOutcome | None) -> str:
    transition = outcome.transition_outcome if outcome is not None else None
    if isinstance(transition, DisplayTransitionOutcome):
        cause = transition.cause
        if cause is not None:
            return str(cause.value)
        return str(transition.kind.value)
    return ""


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
    clear_active_cache_identity_state: Callable[[], None]
    active_preview_cache_identity_matches_current_workspace: Callable[[], bool]
    set_last_simulation_provenance: Callable[[Dict[str, Any]], None]
    set_last_simulation_ctc: Callable[[Dict[str, float]], None]
    publish_simulation_completion_provenance: Callable[..., Dict[str, Any]]
    update_display_transaction_provenance: Callable[..., Dict[str, Any]]
    set_main_plot_scalar_values: Callable[[dict[str, object]], None]
    update_main_plot_statistics: Callable[..., None]
    main_plot_stats_table: Callable[[], object]
    publish_main_plot_results_table: Callable[[object], None]
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


@dataclass(frozen=True, slots=True)
class RuntimeSliderReplayDisplayRefresh:
    display_outcome: SimulationCompletionDisplayOutcome | None = None
    displayed: bool = False
    focused_controls_use_workspace: Optional[bool] = None
    log_reason: str = ""


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
        self._reference_overlays_visible: bool = True
        self._display_projection_state = DisplayProjectionState(
            reference_overlays_visible=self._reference_overlays_visible,
        )
        self._last_display_transition_outcome: DisplayTransitionOutcome | None = None

    def active_display_transaction(self) -> ActiveDisplayTransaction | None:
        return self._active_display_transaction

    def set_reference_overlays_visible(self, visible: bool) -> None:
        show = bool(visible)
        previous_visible = self._reference_overlays_visible
        previous_state = self._display_projection_state
        previous_transaction = self._active_display_transaction
        self._reference_overlays_visible = show
        self._display_projection_state = replace(
            previous_state,
            reference_overlays_visible=show,
        )

        active_transaction = self._active_display_transaction
        if not isinstance(active_transaction, ActiveDisplayTransaction):
            return

        if show:
            active_transaction = self._hydrate_reference_overlays_for_transaction(active_transaction)
            self._active_display_transaction = active_transaction

        outcome = self._apply_active_display_projection(
            requested_show_set_ids=self._display_projection_state.visible_result_set_ids,
            prefer_set_id=self._display_projection_state.primary_visible_set_id,
            display_source=DisplayRefreshSource.EXPLICIT_SHOW_REQUEST,
            require_full_coverage=False,
        )
        if outcome is None:
            self._active_display_transaction = previous_transaction
            self._display_projection_state = previous_state
            self._reference_overlays_visible = bool(previous_visible)
        else:
            self.refresh_active_display_transaction_provenance_projection()

    def _set_display_projection_state(
        self,
        *,
        visible_result_set_ids: Sequence[str] | None,
        primary_visible_set_id: str = "",
        reference_overlays_visible: bool | None = None,
    ) -> None:
        self._display_projection_state = DisplayProjectionState(
            visible_result_set_ids=(
                None
                if visible_result_set_ids is None
                else tuple(str(set_id) for set_id in (visible_result_set_ids or ()) if str(set_id))
            ),
            primary_visible_set_id=str(primary_visible_set_id or ""),
            reference_overlays_visible=(
                self._reference_overlays_visible
                if reference_overlays_visible is None
                else bool(reference_overlays_visible)
            ),
        )

    def _result_metadata_by_set_id(
        self,
        transaction: ActiveDisplayTransaction,
    ) -> Dict[str, DisplaySetMetadata]:
        metadata_by_set_id: Dict[str, DisplaySetMetadata] = {}
        for metadata in ordered_display_transaction_metadata(transaction):
            if metadata.role is DisplaySetRole.REFERENCE_OVERLAY:
                continue
            set_id = str(metadata.set_id or "").strip()
            if not set_id or set_id in metadata_by_set_id:
                continue
            metadata_by_set_id[set_id] = metadata
        return metadata_by_set_id

    def _projection_visible_result_set_ids(
        self,
        transaction: ActiveDisplayTransaction,
        *,
        requested_show_set_ids: Sequence[str] | None = None,
    ) -> tuple[str, ...]:
        active_ids = tuple(deduped_set_ids(transaction.display_set_ids or ()))
        active_set = set(active_ids)
        if requested_show_set_ids is not None:
            requested_ids = tuple(
                deduped_set_ids(
                    tuple(str(set_id) for set_id in requested_show_set_ids if str(set_id))
                )
            )
        else:
            state_ids = self._display_projection_state.visible_result_set_ids
            requested_ids = active_ids if state_ids is None else tuple(deduped_set_ids(state_ids))
        return tuple(set_id for set_id in requested_ids if set_id in active_set)

    def _projected_transaction_for_display_provenance(
        self,
        transaction: ActiveDisplayTransaction,
    ) -> ActiveDisplayTransaction | None:
        projected = self._projected_transaction_for_plot(transaction)
        if projected is not None:
            return projected
        if not isinstance(transaction, ActiveDisplayTransaction):
            return None
        # Preserve the display transaction identity/status while making the
        # current projection explicit: no result/reference layers are visible.
        return replace(
            transaction,
            display_set_ids=(),
            primary_display_set_id="",
            sets={},
        )

    def _projected_transaction_for_plot(
        self,
        transaction: ActiveDisplayTransaction,
        *,
        requested_show_set_ids: Sequence[str] | None = None,
        prefer_set_id: str | None = None,
        reference_overlays_visible: bool | None = None,
    ) -> ActiveDisplayTransaction | None:
        visible_ids = self._projection_visible_result_set_ids(
            transaction,
            requested_show_set_ids=requested_show_set_ids,
        )
        reference_visible = (
            self._display_projection_state.reference_overlays_visible
            if reference_overlays_visible is None
            else bool(reference_overlays_visible)
        )

        # Direct, non-batch displays have no display_set_ids.  They still need a
        # renderable primary layer; projection is only a batch set-id filter.
        if not tuple(deduped_set_ids(transaction.display_set_ids or ())) and dict(transaction.sets or {}):
            updated_sets: Dict[str, DisplaySetMetadata] = {}
            for raw_layer_id, metadata in dict(transaction.sets or {}).items():
                layer_id = str(raw_layer_id or metadata.layer_id or "").strip()
                if not layer_id:
                    continue
                if metadata.role is DisplaySetRole.REFERENCE_OVERLAY:
                    metadata = replace(metadata, visible=reference_visible)
                updated_sets[layer_id] = metadata
            return replace(transaction, sets=updated_sets)

        if not visible_ids:
            return None
        preferred_id = str(prefer_set_id or self._display_projection_state.primary_visible_set_id or "").strip()
        if preferred_id not in visible_ids:
            transaction_primary = str(transaction.primary_display_set_id or "").strip()
            preferred_id = transaction_primary if transaction_primary in visible_ids else str(visible_ids[0])

        result_metadata = self._result_metadata_by_set_id(transaction)
        projected_sets: Dict[str, DisplaySetMetadata] = {}
        for set_id in visible_ids:
            metadata = result_metadata.get(str(set_id))
            if metadata is None:
                continue
            layer_id = str(metadata.layer_id or f"result:{set_id}").strip()
            role = DisplaySetRole.PRIMARY_RESULT if str(set_id) == preferred_id else DisplaySetRole.RESULT_OVERLAY
            projected_sets[layer_id] = replace(metadata, role=role, visible=True)

        if not projected_sets:
            return None
        if not any(metadata.role is DisplaySetRole.PRIMARY_RESULT for metadata in projected_sets.values()):
            first_layer_id, first_metadata = next(iter(projected_sets.items()))
            preferred_id = str(first_metadata.set_id or "").strip()
            projected_sets[first_layer_id] = replace(
                first_metadata,
                role=DisplaySetRole.PRIMARY_RESULT,
                visible=True,
            )

        if reference_visible:
            visible_set = set(visible_ids)
            for metadata in dict(transaction.sets or {}).values():
                if metadata.role is not DisplaySetRole.REFERENCE_OVERLAY:
                    continue
                set_id = str(metadata.set_id or "").strip()
                if set_id not in visible_set:
                    continue
                layer_id = str(metadata.layer_id or f"reference:{set_id}").strip()
                if not layer_id:
                    continue
                projected_sets[layer_id] = replace(metadata, visible=True)

        return replace(
            transaction,
            display_set_ids=visible_ids,
            primary_display_set_id=preferred_id,
            sets=projected_sets,
        )

    @staticmethod
    def _primary_result_metadata_from_transaction(
        transaction: ActiveDisplayTransaction,
    ) -> DisplaySetMetadata | None:
        for metadata in ordered_display_transaction_metadata(transaction):
            if metadata.role is DisplaySetRole.PRIMARY_RESULT:
                return metadata
        return None

    @staticmethod
    def _prefer_layer_id_for_projected_transaction(
        transaction: ActiveDisplayTransaction,
    ) -> str:
        prefer = f"result:{transaction.primary_display_set_id}"
        for metadata in dict(transaction.sets or {}).values():
            if (
                metadata.role is DisplaySetRole.PRIMARY_RESULT
                and str(metadata.set_id) == str(transaction.primary_display_set_id)
            ):
                return str(metadata.layer_id or prefer)
        return prefer

    def _refresh_projected_plot_statistics(
        self,
        *,
        plot: ResultsDisplayPlotPort,
        projected_transaction: ActiveDisplayTransaction | None,
    ) -> None:
        if projected_transaction is None:
            try:
                plot.set_statistics_results({}, prefer="")
                self._publish_main_plot_results_table(plot=plot)
            except Exception as exc:
                logger.debug("Failed to clear projected plot statistics: %s", exc, exc_info=True)
            return
        stats_results_map = stats_results_map_from_display_transaction(
            projected_transaction,
            include_reference_overlays=True,
            presentation_labels_by_set_id=self._popup_labels_by_set_id(projected_transaction.display_set_ids),
        )
        try:
            plot.set_statistics_results(
                stats_results_map,
                prefer=self._prefer_layer_id_for_projected_transaction(projected_transaction),
            )
            self._publish_main_plot_results_table(plot=plot)
        except Exception as exc:
            logger.exception("Failed to refresh projected plot statistics: %s", exc)

    def _apply_active_display_projection(
        self,
        *,
        requested_show_set_ids: Sequence[str] | None,
        prefer_set_id: str | None = None,
        display_source: DisplayRefreshSource = DisplayRefreshSource.INCIDENTAL_REFRESH,
        require_full_coverage: bool = True,
    ) -> BatchDisplayRefreshOutcome | None:
        _ = display_source
        active_transaction = self._active_display_transaction
        if not isinstance(active_transaction, ActiveDisplayTransaction):
            return None
        requested_ids = (
            None
            if requested_show_set_ids is None
            else tuple(deduped_set_ids(tuple(str(set_id) for set_id in requested_show_set_ids if str(set_id))))
        )
        if requested_ids is not None and bool(require_full_coverage):
            active_ids = set(active_transaction.display_set_ids or ())
            if any(set_id not in active_ids for set_id in requested_ids):
                return None
        projected_ids = self._projection_visible_result_set_ids(
            active_transaction,
            requested_show_set_ids=requested_ids,
        )
        if requested_ids is not None:
            preferred = str(prefer_set_id or "").strip()
            if preferred not in projected_ids:
                preferred = str(projected_ids[0]) if projected_ids else ""
            self._set_display_projection_state(
                visible_result_set_ids=projected_ids,
                primary_visible_set_id=preferred,
                reference_overlays_visible=self._reference_overlays_visible,
            )
        projected_transaction = self._projected_transaction_for_plot(active_transaction)
        plot = self._main_plot()
        if projected_transaction is None:
            try:
                plot.clear_display_transaction_state(preserve_y_selection_state=True)
            except Exception as exc:
                logger.debug("Failed to clear projected plot view: %s", exc, exc_info=True)
            self._refresh_projected_plot_statistics(plot=plot, projected_transaction=None)
            try:
                self._ui.refresh_simulation_plot_views()
            except Exception as exc:
                logger.debug("Failed to refresh plot views after projection clear: %s", exc, exc_info=True)
        else:
            if not self._set_plot_display_layers(plot=plot, active_display_transaction=active_transaction):
                return None
            self._refresh_projected_plot_statistics(
                plot=plot,
                projected_transaction=projected_transaction,
            )
            try:
                self._ui.show_simulation_tab()
                self._ui.refresh_simulation_plot_views()
            except Exception as exc:
                logger.debug("Failed to refresh plot views after projection update: %s", exc, exc_info=True)

        transition_outcome = self._record_display_transition_outcome(
            outcome_kind=DisplayTransitionOutcomeKind.PUBLISHED,
            active_transaction=active_transaction,
            previous_transaction=active_transaction,
            display_status=active_transaction.status,
            requested_show_set_ids=(requested_ids if requested_ids is not None else projected_ids),
            requested_labels_by_set_id=self._popup_labels_by_set_id(
                requested_ids if requested_ids is not None else projected_ids
            ),
            display_set_ids=projected_ids,
            attempted_display_set_ids=(requested_ids if requested_ids is not None else projected_ids),
            affected_set_ids=(requested_ids if requested_ids is not None else projected_ids),
            event_kind=DisplayEventKind.SHOW_SCOPE_CHANGED,
            cause=self._display_cause_for_active_kind(active_transaction.kind),
        )
        if not projected_ids:
            self._ui.set_status_text("Plot hidden; active results retained.")
        else:
            self._ui.set_status_text(display_transition_status_text(transition_outcome))
        self.refresh_active_display_transaction_provenance_projection()
        return BatchDisplayRefreshOutcome(
            focused_controls_use_workspace=None,
            transition_outcome=transition_outcome,
        )

    def _active_display_projection_covers_request(
        self,
        requested_show_set_ids: Sequence[str],
    ) -> bool:
        active_transaction = self._active_display_transaction
        if not isinstance(active_transaction, ActiveDisplayTransaction):
            return False
        requested_ids = tuple(deduped_set_ids(tuple(str(set_id) for set_id in requested_show_set_ids if str(set_id))))
        if not requested_ids:
            return True
        active_ids = set(active_transaction.display_set_ids or ())
        return all(set_id in active_ids for set_id in requested_ids)

    def _cached_reference_entry_for_set_id(self, set_id: str) -> Mapping[str, Any] | None:
        cache_key = str(self._ui.active_batch_cache_key() or "").strip()
        snapshot = None
        try:
            snapshot = self._ui.active_result_cache_read_snapshot(cache_key=cache_key or None)
        except TypeError:
            try:
                snapshot = self._ui.active_result_cache_read_snapshot(cache_key=cache_key)
            except Exception as exc:
                logger.debug("Failed to read active result cache snapshot for reference overlays: %s", exc, exc_info=True)
        except Exception as exc:
            logger.debug("Failed to read active result cache snapshot for reference overlays: %s", exc, exc_info=True)
        if not isinstance(snapshot, BatchCacheResultReadSnapshot):
            return None
        invalidated_ids = {str(sid) for sid in (snapshot.invalidated_set_ids or ()) if str(sid)}
        if str(set_id) in invalidated_ids:
            return None
        result = self._cache_entry_for_set_id(
            set_id=str(set_id),
            snapshot=snapshot,
            require_completion_provenance=True,
        )
        if isinstance(result.entry, Mapping):
            return result.entry
        return None

    def _hydrate_reference_overlays_for_transaction(
        self,
        transaction: ActiveDisplayTransaction,
    ) -> ActiveDisplayTransaction:
        updated_sets: Dict[str, DisplaySetMetadata] = dict(transaction.sets or {})
        existing_reference_ids = {
            str(metadata.set_id or "").strip()
            for metadata in updated_sets.values()
            if metadata.role is DisplaySetRole.REFERENCE_OVERLAY
        }
        result_metadata_by_id = self._result_metadata_by_set_id(transaction)
        labels_by_id = self._popup_labels_by_set_id(transaction.display_set_ids)
        changed = False
        for set_id in tuple(transaction.display_set_ids or ()):
            sid = str(set_id or "").strip()
            if not sid or sid in existing_reference_ids:
                continue
            entry = self._cached_reference_entry_for_set_id(sid)
            if not isinstance(entry, Mapping):
                # Canonical reference overlays are cache/display-authority truth.
                # Never fabricate them from the currently displayed result.
                continue
            entry_payload = {
                **dict(entry),
                "display_species": self._explicit_display_species_for_entry(entry),
            }
            if not entry_payload.get("display_species"):
                source_metadata = result_metadata_by_id.get(sid)
                if source_metadata is not None:
                    entry_payload["display_species"] = tuple(source_metadata.display_species or ())
            if not owned_species_for_display_entry(entry_payload):
                source_metadata = result_metadata_by_id.get(sid)
                if source_metadata is not None:
                    entry_payload["owned_species"] = tuple(source_metadata.owned_species or ())
            if self._semantic_unavailable_display_set_ids({sid: entry_payload}):
                continue
            display_metadata = display_metadata_for_entry(
                label=labels_by_id.get(sid, sid),
                entry=entry_payload,
                set_id=sid,
                role=DisplaySetRole.REFERENCE_OVERLAY,
                layer_id=f"reference:{sid}",
                visible=self._reference_overlays_visible,
            )
            if display_metadata is None:
                continue
            updated_sets[str(display_metadata.layer_id or f"reference:{sid}")] = display_metadata
            changed = True
        if not changed:
            return transaction
        return replace(transaction, sets=updated_sets)

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
            requested_show_ids = tuple(deduped_set_ids(requested_show_set_ids))
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
            run_target_set_ids=deduped_set_ids(run_target_set_ids),
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
    def _semantic_unavailable_display_set_ids(entries_by_set_id: Mapping[str, Mapping[str, Any]]) -> tuple[str, ...]:
        unavailable: list[str] = []
        for set_id, entry in entries_by_set_id.items():
            owned = owned_species_for_display_entry(entry)
            display_series = ResultsController._display_series_for_entry(entry)
            available = set(display_series or {})
            if (
                not owned
                or display_series is None
                or any(str(name) not in available for name in owned)
            ):
                unavailable.append(str(set_id))
        return tuple(unavailable)

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
    ) -> dict[str, Mapping[str, Any]]:
        displayed_entries: dict[str, Mapping[str, Any]] = {str(primary): primary_entry}
        for sid in available:
            if sid == primary:
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

    @staticmethod
    def _explicit_display_species_for_entry(entry: Mapping[str, Any]) -> tuple[str, ...]:
        raw_display_species = entry.get("display_species")
        if not isinstance(raw_display_species, Sequence) or isinstance(raw_display_species, (str, bytes)):
            return ()
        return deduped_set_ids(tuple(str(name) for name in raw_display_species if str(name)))

    @staticmethod
    def _display_series_for_entry(entry: Mapping[str, Any]) -> Optional[Dict[str, np.ndarray]]:
        if not isinstance(entry, Mapping):
            return None
        return series_for_display_species(
            series=entry.get("series"),
            display_species=ResultsController._explicit_display_species_for_entry(entry),
        )

    def _publish_main_plot_results_table(self, *, plot: ResultsDisplayPlotPort | None = None) -> None:
        table = plot.stats_table() if plot is not None else self._ui.main_plot_stats_table()
        self._ui.publish_main_plot_results_table(table)

    def _apply_cached_batch_plot_metadata(
        self,
        *,
        plot: ResultsDisplayPlotPort,
        active_display_transaction: ActiveDisplayTransaction,
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

        stats_results_map = stats_results_map_from_display_transaction(
            active_display_transaction,
            include_reference_overlays=False,
            presentation_labels_by_set_id=self._popup_labels_by_set_id(active_display_transaction.display_set_ids),
        )
        primary_metadata = self._primary_result_metadata_from_transaction(active_display_transaction)
        preferred_t = primary_metadata.t if primary_metadata is not None else t
        preferred_series = primary_metadata.series if primary_metadata is not None else series
        try:
            self._ui.update_main_plot_statistics(
                stats_results_map=stats_results_map,
                prefer=self._prefer_layer_id_for_projected_transaction(active_display_transaction),
                t=np.asarray(preferred_t, dtype=float),
                series={str(k): np.asarray(v, dtype=float) for k, v in dict(preferred_series or {}).items()},
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
        active_display_transaction: ActiveDisplayTransaction,
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

        stats_results_map = stats_results_map_from_display_transaction(
            active_display_transaction,
            include_reference_overlays=True,
            presentation_labels_by_set_id=self._popup_labels_by_set_id(active_display_transaction.display_set_ids),
        )
        primary_metadata = self._primary_result_metadata_from_transaction(active_display_transaction)
        preferred_t = (
            primary_metadata.t
            if primary_metadata is not None
            else primary.entry.get("t")
        )
        preferred_series = (
            primary_metadata.series
            if primary_metadata is not None
            else (primary.entry.get("series") or {})
        )
        try:
            self._ui.update_main_plot_statistics(
                stats_results_map=stats_results_map,
                prefer=self._prefer_layer_id_for_projected_transaction(active_display_transaction),
                t=np.asarray(preferred_t, dtype=float),
                series={
                    str(k): np.asarray(v, dtype=float)
                    for k, v in dict(preferred_series or {}).items()
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
        display_series_by_set_id: Mapping[str, Mapping[str, object]],
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
        _ = display_series_by_set_id
        stats_results_map = stats_results_map_from_display_transaction(
            active_display_transaction,
            include_reference_overlays=True,
            presentation_labels_by_set_id=self._popup_labels_by_set_id(active_display_transaction.display_set_ids),
        )
        primary_metadata = self._primary_result_metadata_from_transaction(active_display_transaction)
        preferred_t = primary_metadata.t if primary_metadata is not None else primary.t
        preferred_series = (
            primary_metadata.series
            if primary_metadata is not None
            else display_series_by_set_id.get(str(primary.set_id), {})
        )
        try:
            self._ui.update_main_plot_statistics(
                stats_results_map=stats_results_map,
                prefer=self._prefer_layer_id_for_projected_transaction(active_display_transaction),
                t=np.asarray(preferred_t, dtype=float),
                series={
                    str(k): np.asarray(v, dtype=float)
                    for k, v in dict(preferred_series or {}).items()
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

    def _apply_active_transaction_plot_metadata(
        self,
        *,
        plot: object,
        active_display_transaction: ActiveDisplayTransaction,
        primary_t: np.ndarray,
        primary_series: Mapping[str, Any],
        algebra_scalars: Mapping[str, object] | None,
        prefer_layer_id: str,
        context_label: str,
    ) -> Optional[str]:
        try:
            self._ui.set_main_plot_scalar_values(dict(algebra_scalars or {}))
        except Exception as exc:
            logger.exception("Failed to set plot scalar values for %s: %s", context_label, exc)
            return "metadata_scalar_failed"

        normalized_series = {str(k): np.asarray(v, dtype=float) for k, v in dict(primary_series or {}).items()}
        stats_results_map = stats_results_map_from_display_transaction(
            active_display_transaction,
            include_reference_overlays=True,
            presentation_labels_by_set_id=self._popup_labels_by_set_id(active_display_transaction.display_set_ids),
        )
        try:
            self._ui.update_main_plot_statistics(
                stats_results_map=stats_results_map,
                prefer=str(prefer_layer_id or ""),
                t=np.asarray(primary_t, dtype=float),
                series=normalized_series,
            )
        except Exception as exc:
            logger.exception("Failed to update plot statistics for %s: %s", context_label, exc)
            return "metadata_statistics_failed"
        try:
            self._publish_main_plot_results_table(plot=plot)
        except Exception as exc:
            logger.exception("Failed to update results table for %s: %s", context_label, exc)
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
                        "kind": PlotLayerKind.RESULT_SERIES,
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
        provenance_transaction = (
            self._projected_transaction_for_display_provenance(active_display_transaction)
            if isinstance(active_display_transaction, ActiveDisplayTransaction)
            else None
        )
        transaction_payload = display_transaction_provenance_payload(
            provenance_transaction,
        )
        if not isinstance(direct_completion_provenance, Mapping):
            if not transaction_payload:
                self._clear_direct_completion_provenance()
                return None
            payload = transaction_payload
        else:
            payload = dict(direct_completion_provenance)
            payload.update(transaction_payload)
        if isinstance(provenance_transaction, ActiveDisplayTransaction):
            primary_metadata = next(
                (
                    metadata
                    for metadata in dict(provenance_transaction.sets or {}).values()
                    if metadata.role is DisplaySetRole.PRIMARY_RESULT
                    and str(metadata.set_id) == str(provenance_transaction.primary_display_set_id)
                ),
                None,
            )
            if isinstance(primary_metadata, DisplaySetMetadata):
                display_series_by_name = {
                    str(name): np.asarray(values, dtype=float)
                    for name, values in dict(primary_metadata.series or {}).items()
                    if str(name)
                }
                display_species = [
                    str(name)
                    for name in (primary_metadata.display_species or ())
                    if str(name) and str(name) in display_series_by_name
                ]
                display_series = {
                    name: display_series_by_name[name]
                    for name in display_species
                    if name in display_series_by_name
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

    def refresh_active_display_transaction_provenance_projection(self, *_args: object) -> None:
        active_transaction = self._active_display_transaction
        if not isinstance(active_transaction, ActiveDisplayTransaction):
            return
        payload = display_transaction_provenance_payload(
            self._projected_transaction_for_display_provenance(active_transaction),
        )
        try:
            self._ui.update_display_transaction_provenance(
                display_transaction=payload.get("display_transaction"),
                display_sets=payload.get("display_sets"),
            )
        except Exception as exc:
            logger.exception("Failed to refresh display transaction provenance projection: %s", exc)

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

    def _deauthorize_active_display_transaction_outputs(
        self,
        *,
        clear_plot: bool = True,
    ) -> ActiveDisplayTransaction | None:
        previous_transaction = self._active_display_transaction
        self._active_display_transaction = None
        self._set_display_projection_state(
            visible_result_set_ids=None,
            primary_visible_set_id="",
            reference_overlays_visible=self._reference_overlays_visible,
        )
        if bool(clear_plot):
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
        request_scope: DisplayRequestScopeSnapshot | None = None,
        attempted_display_set_ids: Sequence[str] = (),
        affected_set_ids: Sequence[str] = (),
        unresolved_intent_set_ids: Sequence[str] = (),
        missing_intent_set_ids: Sequence[str] = (),
        failed_intent_set_ids: Sequence[str] = (),
        semantic_unavailable_set_ids: Sequence[str] = (),
        clear_plot: bool = True,
    ) -> DisplayTransitionOutcome:
        previous_transaction = self._deauthorize_active_display_transaction_outputs(clear_plot=clear_plot)
        transition = self._record_display_transition_outcome(
            outcome_kind=outcome_kind,
            active_transaction=None,
            previous_transaction=previous_transaction,
            display_status=display_status,
            request_scope=request_scope,
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
        self._ui.set_status_text(display_transition_status_text(transition))
        return transition

    def clear_display_if_workspace_previews_were_displayed(
        self,
        set_ids: Sequence[str],
        *,
        clear_plot: bool = True,
    ) -> bool:
        displayed_set_ids = self._displayed_workspace_preview_set_ids(set_ids)
        if not displayed_set_ids:
            return False
        self._ui.clear_active_preview_cache_identity_state()
        self.clear_active_display_transaction(
            display_status=DisplayStatus.NO_COMPLETE_DISPLAYABLE_REQUEST_SCOPE,
            cause=DisplayTransitionCause.AFFECTED_SCOPE_INTERSECTS_ACTIVE_DISPLAY,
            affected_set_ids=displayed_set_ids,
            unresolved_intent_set_ids=displayed_set_ids,
            clear_plot=bool(clear_plot),
        )
        return True

    def deauthorize_current_preview_failure(
        self,
        *,
        target_set_ids: Sequence[str],
        request_id: Optional[int] = None,
        run_id: Optional[int] = None,
        status_text: str = "",
    ) -> DisplayTransitionOutcome | None:
        target_ids = deduped_set_ids(target_set_ids)
        if not target_ids:
            return None
        active_transaction = self._active_display_transaction
        if not isinstance(active_transaction, ActiveDisplayTransaction):
            self._ui.clear_active_preview_cache_identity_state()
            return None
        display_ids = deduped_set_ids(active_transaction.display_set_ids)
        if not set(target_ids).intersection(display_ids):
            self._ui.clear_active_preview_cache_identity_state()
            return None

        preview_backed = active_transaction.kind in {
            ActiveDisplayKind.FRESH_PREVIEW,
            ActiveDisplayKind.WORKSPACE_PREVIEW,
        }
        if not preview_backed:
            preview_backed = any(
                isinstance(metadata.workspace_preview_provenance, Mapping)
                for metadata in dict(active_transaction.sets or {}).values()
                if str(metadata.set_id or "") in set(target_ids)
            )
        if not preview_backed:
            self._ui.clear_active_preview_cache_identity_state()
            return None

        self._ui.clear_active_preview_cache_identity_state()
        request_scope = self._current_display_request_scope(
            requested_show_set_ids=target_ids,
            run_target_set_ids=target_ids,
            run_id=run_id,
            request_id=request_id,
        )
        outcome = self.clear_active_display_transaction(
            outcome_kind=DisplayTransitionOutcomeKind.CLEARED,
            display_status=DisplayStatus.NO_COMPLETE_DISPLAYABLE_REQUEST_SCOPE,
            event_kind=DisplayEventKind.DISPLAY_CLEARED,
            cause=DisplayTransitionCause.CURRENT_PREVIEW_FAILED,
            requested_show_set_ids=target_ids,
            requested_labels_by_set_id=request_scope.requested_labels_by_set_id,
            request_scope=request_scope,
            attempted_display_set_ids=target_ids,
            affected_set_ids=target_ids,
            unresolved_intent_set_ids=target_ids,
            failed_intent_set_ids=target_ids,
            clear_plot=True,
        )
        if status_text:
            self._ui.set_status_text(str(status_text))
        return outcome

    def invalidate_workspace_preview_display_and_cache(
        self,
        target_set_ids: Sequence[str],
        *,
        clear_plot: bool = True,
        clear_active_cache_identity: bool | None = None,
    ) -> bool:
        invalidated = bool(
            self.clear_display_if_workspace_previews_were_displayed(
                target_set_ids,
                clear_plot=bool(clear_plot),
            )
        )
        should_clear_active_cache_identity = clear_active_cache_identity
        if should_clear_active_cache_identity is None:
            try:
                should_clear_active_cache_identity = bool(
                    self._ui.active_preview_cache_identity_matches_current_workspace()
                )
            except Exception:
                should_clear_active_cache_identity = False
        if bool(should_clear_active_cache_identity):
            self._ui.clear_active_cache_identity_state()
            invalidated = True
        return invalidated

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

    def deauthorize_completed_run_display_for_runtime_input_preview(
        self,
        *,
        affected_set_ids: Sequence[str],
        affected_scope_is_global: bool,
    ) -> DisplayTransitionOutcome | None:
        if not self._completed_run_display_transaction_active():
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
            event_kind=_DISPLAY_TRANSITION_RUNTIME_INPUT_PREVIEW_DEAUTHORIZATION.event_kind,
            cause=_DISPLAY_TRANSITION_RUNTIME_INPUT_PREVIEW_DEAUTHORIZATION.cause,
            clear_plot=False,
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
        unresolved_scope = tuple(deduped_set_ids(unresolved_intent_set_ids or tuple(affected_scope)))
        failed_scope = tuple(deduped_set_ids(failed_intent_set_ids))
        semantic_scope = tuple(
            deduped_set_ids(
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
        unresolved_scope = tuple(deduped_set_ids(unresolved_intent_set_ids or tuple(affected_scope)))
        failed_scope = tuple(
            deduped_set_ids(
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
        """Apply an explicit Plot-checkbox view change without destroying results.

        Plot membership is a reversible view projection over the active
        transaction. True deauthorization still happens through mechanism,
        cache, run, and deleted-set invalidation paths.
        """
        if request.display_source is not DisplayRefreshSource.EXPLICIT_SHOW_REQUEST:
            return None
        if not self._active_display_projection_covers_request(requested_show_set_ids):
            return None
        outcome = self._apply_active_display_projection(
            requested_show_set_ids=requested_show_set_ids,
            prefer_set_id=request.prefer_set_id,
            display_source=request.display_source,
            require_full_coverage=True,
        )
        return outcome.transition_outcome if outcome is not None else None

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
        requested_ids = tuple(deduped_set_ids(requested_show_set_ids))
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
        self._ui.set_status_text(display_transition_status_text(outcome))
        return outcome

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
    def _completion_entry_display_series(
        entry: CompletionDisplayEntry,
    ) -> Optional[Dict[str, object]]:
        display_series = series_for_display_species(
            series=entry.series,
            display_species=entry.display_species,
        )
        if display_series is None:
            return None
        owned_species = tuple(str(name) for name in entry.owned_species if str(name))
        for species_name in owned_species:
            if species_name not in display_series:
                return None
        return display_series

    def build_main_plot_copy_all_export_plan(self) -> object | None:
        active_transaction = self._active_display_transaction
        projected_transaction = (
            self._projected_transaction_for_plot(active_transaction)
            if isinstance(active_transaction, ActiveDisplayTransaction)
            else None
        )
        return build_copy_all_export_plan(projected_transaction)

    def build_main_plot_csv_export(self, scope: str) -> tuple[list[str], list[list[object]]]:
        active_transaction = self._active_display_transaction
        if active_transaction is None:
            raise ValueError("No active simulation display transaction is available to export.")
        projected_transaction = self._projected_transaction_for_plot(active_transaction)
        if projected_transaction is None:
            raise ValueError("No visible simulation display series are available to export.")
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
        return build_main_plot_csv_export(
            active_transaction=projected_transaction,
            scope=normalized_scope,
            axis_state=axis_state,
        )

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
        primary_set_id: str,
        primary_label: str,
        display_set_ids: Sequence[str],
        owned_species: Sequence[str] | None,
        display_species: Sequence[str],
        completion_provenance: Mapping[str, Any] | None,
        workspace_preview_provenance_by_set_id: Mapping[str, Mapping[str, Any]] | None,
        display_transition: DisplayPublicationTransition,
        additional_metadata: Sequence[DisplaySetMetadata] = (),
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
            primary_set_id=primary_set_id,
            primary_label=primary_label,
            display_set_ids=display_set_ids,
            owned_species=owned_species,
            display_species=display_species,
            completion_provenance=completion_provenance,
            workspace_preview_provenance_by_set_id=workspace_preview_provenance_by_set_id,
            additional_metadata=additional_metadata,
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
            deduped_set_ids(
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
            deduped_set_ids(
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
            deduped_set_ids(
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
        outcome = self._record_display_transition_outcome(
            outcome_kind=outcome_kind,
            active_transaction=active_transaction,
            previous_transaction=active_transaction,
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
        self._ui.set_status_text(display_transition_status_text(outcome))
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
        affected_ids = tuple(deduped_set_ids(affected_set_ids))
        unresolved_ids = tuple(deduped_set_ids(unresolved_intent_set_ids or affected_ids))
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
        affected_ids = tuple(deduped_set_ids(affected_set_ids))
        unresolved_ids = tuple(deduped_set_ids(unresolved_intent_set_ids or affected_ids))
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
        self._ui.set_status_text(display_transition_status_text(transition))
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
        metadata_applier: Callable[[ResultsDisplayPlotPort, ActiveDisplayTransaction], Optional[str]],
        annotation_entry: Mapping[str, Any],
        primary_set_id: str,
        primary_label: str,
        display_set_ids: Sequence[str],
        workspace_preview_provenance_by_set_id: Mapping[str, Mapping[str, Any]] | None = None,
        completion_provenance: Mapping[str, Any] | None = None,
        owned_species: Sequence[str] | None = None,
        display_species: Sequence[str],
        display_transition: DisplayPublicationTransition,
        additional_metadata: Sequence[DisplaySetMetadata] = (),
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
        attempted_display_ids = tuple(deduped_set_ids(display_set_ids))
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
            denied_ids = tuple(deduped_set_ids(requested_show_set_ids or attempted_display_ids))
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
        previous_transaction = self._active_display_transaction
        active_transaction = self._active_transaction_for_display_commit(
            t=np.asarray(t, dtype=float),
            series=series,
            primary_set_id=str(primary_set_id),
            primary_label=str(primary_label),
            display_set_ids=attempted_display_ids,
            owned_species=owned_species,
            display_species=display_species,
            completion_provenance=completion_provenance,
            workspace_preview_provenance_by_set_id=workspace_preview_provenance_by_set_id,
            additional_metadata=additional_metadata,
            display_transition=display_transition,
            run_id=run_id,
            request_id=request_id,
        )
        projected_initial_ids = tuple(deduped_set_ids(requested_show_set_ids or attempted_display_ids))
        if not projected_initial_ids:
            projected_initial_ids = attempted_display_ids
        preferred_projection_id = str(primary_set_id or "").strip()
        if preferred_projection_id not in set(projected_initial_ids):
            preferred_projection_id = str(projected_initial_ids[0]) if projected_initial_ids else ""
        self._set_display_projection_state(
            visible_result_set_ids=projected_initial_ids,
            primary_visible_set_id=preferred_projection_id,
            reference_overlays_visible=self._reference_overlays_visible,
        )
        if self._reference_overlays_visible:
            active_transaction = self._hydrate_reference_overlays_for_transaction(active_transaction)
        plot = self._main_plot()
        metadata_transaction = self._projected_transaction_for_plot(active_transaction) or active_transaction
        metadata_failure = metadata_applier(plot, metadata_transaction)
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
        self._active_display_transaction = active_transaction
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
            self._active_display_transaction = active_transaction
        # Keep the active transaction as result truth; the Plot checkbox state is
        # a separate projection that was initialized from the request scope above.
        if not self._set_plot_display_layers(plot=plot, active_display_transaction=active_transaction):
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
        self._ui.set_status_text(display_transition_status_text(transition_outcome))
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
            self._ui.set_status_text(display_transition_status_text(transition_outcome))
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
                unavailable_cause=cache_resolution_cause_for_transition(
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
            self._ui.set_status_text(display_transition_status_text(transition_outcome))
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
            self._ui.set_status_text(display_transition_status_text(transition_outcome))
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
                unavailable_cause=cache_resolution_cause_for_transition(
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
        if cache_resolution_cause_for_transition(
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
        requested_ids = tuple(deduped_set_ids(requested_show_set_ids))
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
            self._ui.set_status_text(display_transition_status_text(transition_outcome))
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
            self._ui.set_status_text(display_transition_status_text(transition_outcome))
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
        self._ui.set_status_text(display_transition_status_text(transition_outcome))
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
            projection_outcome = self._apply_active_display_projection(
                requested_show_set_ids=(),
                prefer_set_id=request.prefer_set_id,
                display_source=request.display_source,
                require_full_coverage=False,
            )
            if projection_outcome is not None:
                return BatchDisplayRefreshOutcome(
                    focused_controls_use_workspace=bool(request.focused_set_dirty),
                    transition_outcome=projection_outcome.transition_outcome,
                )
            self._clear_unpublished_batch_display_request(clear_plot=True)
            return BatchDisplayRefreshOutcome(
                focused_controls_use_workspace=bool(request.focused_set_dirty),
            )

        outcome = self._publish_fresh_explicit_dirty_refresh(request=request)
        if outcome is not None:
            return outcome

        resolution = request.resolution
        if (
            request.display_source in {
                DisplayRefreshSource.EXPLICIT_SHOW_REQUEST,
                DisplayRefreshSource.PROGRAMMATIC_SHOW_REQUEST,
            }
            and not bool(request.focused_dirty)
            and not bool(request.focused_set_dirty)
            and not bool(getattr(resolution, "has_workspace_display_request", False))
            and self._active_display_projection_covers_request(requested_show_set_ids)
        ):
            projection_outcome = self._apply_active_display_projection(
                requested_show_set_ids=requested_show_set_ids,
                prefer_set_id=request.prefer_set_id,
                display_source=request.display_source,
                require_full_coverage=True,
            )
            if projection_outcome is not None:
                return BatchDisplayRefreshOutcome(
                    focused_controls_use_workspace=bool(request.focused_set_dirty),
                    transition_outcome=projection_outcome.transition_outcome,
                )

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
        display_request_ids_getter = getattr(self._ui, "effective_display_request_set_ids", None)
        if callable(display_request_ids_getter):
            requested_show_set_ids = tuple(
                str(set_id) for set_id in (display_request_ids_getter() or ()) if str(set_id)
            )
        else:
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
        requested_ids = tuple(deduped_set_ids(requested_show_set_ids))
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

        primary_label = self._ui.batch_name_for_id(primary) or str(primary)
        transaction_entries_by_set_id = self._cached_batch_display_entries_by_set_id(
            available=coverage.available_ids,
            primary=primary,
            primary_entry=entry,
            snapshot=snapshot,
        )
        semantic_unavailable_ids = self._semantic_unavailable_display_set_ids(transaction_entries_by_set_id)
        if semantic_unavailable_ids:
            displayable_entries_by_set_id = {
                str(set_id): entry_payload
                for set_id, entry_payload in transaction_entries_by_set_id.items()
                if str(set_id) not in set(semantic_unavailable_ids)
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
                            deduped_set_ids((*semantic_unavailable_ids, *invalidated_requested_ids))
                        ),
                        missing_intent_set_ids=invalidated_requested_ids,
                        semantic_unavailable_set_ids=semantic_unavailable_ids,
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
                            deduped_set_ids((*semantic_unavailable_ids, *requested_ids))
                        ),
                        semantic_unavailable_set_ids=semantic_unavailable_ids,
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
                primary_label = self._ui.batch_name_for_id(primary) or str(primary)
        transaction_display_set_ids = tuple(str(set_id) for set_id in coverage.available_ids if str(set_id) in transaction_entries_by_set_id)
        additional_metadata: List[DisplaySetMetadata] = []
        for sid, metadata_entry in transaction_entries_by_set_id.items():
            if str(sid) == str(primary):
                continue
            display_metadata = display_metadata_for_entry(
                label=str(self._ui.batch_name_for_id(str(sid)) or sid),
                entry={
                    **dict(metadata_entry),
                    "display_species": self._explicit_display_species_for_entry(metadata_entry),
                },
                set_id=str(sid),
                role=DisplaySetRole.RESULT_OVERLAY,
                layer_id=f"result:{sid}",
            )
            if display_metadata is not None:
                additional_metadata.append(display_metadata)
        unavailable_intent_set_ids = tuple(
            deduped_set_ids(
                (
                    *(
                        set_id
                        for set_id in requested_ids
                        if set_id not in {str(available_id) for available_id in coverage.available_ids if str(available_id)}
                    ),
                    *semantic_unavailable_ids,
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
                semantic_unavailable_set_ids=semantic_unavailable_ids,
                cause=DisplayTransitionCause.DISPLAY_MUTATION_DENIED,
            )
            return CachedBatchDisplayScopeOutcome(
                transition_outcome=transition_outcome,
            )

        primary_display_series = self._display_series_for_entry(entry)
        if primary_display_series is None:
            transition_outcome = self._record_unpublished_display_request_outcome(
                cause=DisplayTransitionCause.SEMANTIC_METADATA_UNAVAILABLE,
                affected_set_ids=requested_ids,
                requested_show_set_ids=requested_ids,
                attempted_display_set_ids=transaction_display_set_ids or coverage.available_ids,
                unresolved_intent_set_ids=unavailable_intent_set_ids or requested_ids,
                semantic_unavailable_set_ids=(str(primary),),
            )
            return CachedBatchDisplayScopeOutcome(
                transition_outcome=transition_outcome,
            )

        return self._apply_batch_display_transaction(
            t=np.asarray(t, dtype=float),
            series=primary_display_series,
            label=str(primary_label),
            metadata_applier=lambda plot, active_transaction: self._apply_cached_batch_plot_metadata(
                plot=plot,
                active_display_transaction=active_transaction,
                primary=primary,
                primary_label=str(primary_label),
                entry=entry,
                t=np.asarray(t, dtype=float),
                series=primary_display_series,
            ),
            annotation_entry=entry,
            primary_set_id=str(primary),
            primary_label=str(primary_label),
            display_set_ids=list(transaction_display_set_ids),
            display_species=self._explicit_display_species_for_entry(entry),
            completion_provenance=entry.get("completion_provenance") if isinstance(entry, Mapping) else None,
            owned_species=owned_species_for_display_entry(entry),
            display_transition=display_transition,
            additional_metadata=tuple(additional_metadata),
            requested_show_set_ids=requested_ids,
            requested_labels_by_set_id={
                str(set_id): str(self._ui.batch_name_for_id(str(set_id)) or set_id)
                for set_id in requested_ids
            },
            unresolved_intent_set_ids=unavailable_intent_set_ids,
            missing_intent_set_ids=missing_intent_set_ids,
            semantic_unavailable_set_ids=semantic_unavailable_ids,
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
            deduped_set_ids(
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
            semantic_ids = tuple(deduped_set_ids(semantic_unavailable_set_ids))
            affected_ids = tuple(
                deduped_set_ids(requested_ids_for_outcome or affected_set_ids)
            )
            unresolved_ids = tuple(
                deduped_set_ids(
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
            if not owned_species_for_display_entry(resolved.entry):
                semantic_unavailable_ids = tuple(deduped_set_ids((*semantic_unavailable_ids, resolved_id)))
                displayable_entries_by_id.pop(resolved_id, None)
                continue
            if self._semantic_unavailable_display_set_ids({resolved_id: resolved.entry}):
                semantic_unavailable_ids = tuple(deduped_set_ids((*semantic_unavailable_ids, resolved_id)))
                displayable_entries_by_id.pop(resolved_id, None)
                continue
            canonical_reference_entry = resolved.authority.canonical_reference_entry
            if canonical_reference_entry is not None and not isinstance(
                canonical_reference_entry.get("completion_provenance"),
                Mapping,
            ):
                return _resolved_unpublished_outcome(
                    cause=DisplayTransitionCause.INVALID_CACHE_ENTRY,
                    affected_set_ids=(resolved_id,),
                )
            if canonical_reference_entry is not None and not owned_species_for_display_entry(
                canonical_reference_entry,
            ):
                semantic_unavailable_ids = tuple(deduped_set_ids((*semantic_unavailable_ids, resolved_id)))
                displayable_entries_by_id.pop(resolved_id, None)
                continue
            if canonical_reference_entry is not None and self._semantic_unavailable_display_set_ids(
                {resolved_id: canonical_reference_entry}
            ):
                semantic_unavailable_ids = tuple(deduped_set_ids((*semantic_unavailable_ids, resolved_id)))
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
            deduped_set_ids((*unresolved_intent_set_ids, *semantic_unavailable_ids))
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

        additional_metadata: List[DisplaySetMetadata] = []
        for resolved in resolved_entries:
            if str(resolved.set_id) == str(primary.set_id):
                continue
            display_metadata = display_metadata_for_entry(
                label=str(resolved.label),
                entry={
                    **dict(resolved.entry),
                    "display_species": self._explicit_display_species_for_entry(resolved.entry),
                },
                set_id=str(resolved.set_id),
                role=DisplaySetRole.RESULT_OVERLAY,
                layer_id=f"result:{resolved.set_id}",
                workspace_preview_provenance=resolved.workspace_preview_provenance,
            )
            if display_metadata is not None:
                additional_metadata.append(display_metadata)
        for resolved in resolved_entries:
            canonical_reference_entry = resolved.authority.canonical_reference_entry
            if canonical_reference_entry is None:
                continue
            display_metadata = display_metadata_for_entry(
                label=str(resolved.label),
                entry={
                    **dict(canonical_reference_entry),
                    "display_species": self._explicit_display_species_for_entry(canonical_reference_entry),
                },
                set_id=str(resolved.set_id),
                role=DisplaySetRole.REFERENCE_OVERLAY,
                layer_id=f"reference:{resolved.set_id}",
                visible=self._reference_overlays_visible,
            )
            if display_metadata is not None:
                additional_metadata.append(display_metadata)
        transaction_display_set_ids = tuple(str(resolved.set_id) for resolved in resolved_entries if str(resolved.set_id))
        workspace_provenance_by_set_id = {
            str(resolved.set_id): dict(resolved.workspace_preview_provenance)
            for resolved in resolved_entries
            if str(resolved.set_id) and isinstance(resolved.workspace_preview_provenance, Mapping)
        }
        primary_has_workspace_provenance = (
            isinstance(primary.workspace_preview_provenance, Mapping)
            and bool(primary.workspace_preview_provenance)
        )
        display_transition = (
            _DISPLAY_TRANSITION_WORKSPACE_PREVIEW
            if primary_has_workspace_provenance
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
        primary_display_series = self._display_series_for_entry(primary.entry)
        if primary_display_series is None:
            return _resolved_unpublished_outcome(
                cause=DisplayTransitionCause.SEMANTIC_METADATA_UNAVAILABLE,
                affected_set_ids=(str(primary.set_id),),
                semantic_unavailable_set_ids=(str(primary.set_id),),
            )
        return self._apply_batch_display_transaction(
            t=np.asarray(primary.entry["t"], dtype=float),
            series=primary_display_series,
            label=str(primary.label),
            metadata_applier=lambda plot, active_transaction: self._apply_resolved_batch_plot_metadata(
                plot=plot,
                active_display_transaction=active_transaction,
                primary=primary,
            ),
            annotation_entry=primary.entry,
            primary_set_id=str(primary.set_id),
            primary_label=str(primary.label),
            display_set_ids=transaction_display_set_ids,
            display_species=self._explicit_display_species_for_entry(primary.entry),
            workspace_preview_provenance_by_set_id=workspace_provenance_by_set_id,
            completion_provenance=(
                primary.entry.get("completion_provenance")
                if isinstance(primary.entry.get("completion_provenance"), Mapping)
                else None
            ),
            owned_species=owned_species_for_display_entry(primary.entry),
            display_transition=display_transition,
            additional_metadata=tuple(additional_metadata),
            requested_show_set_ids=(
                tuple(deduped_set_ids(requested_show_set_ids))
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
        initial_affected_ids = tuple(deduped_set_ids(transaction.display_set_ids or intent_ids))
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
            deduped_set_ids(
                (
                    *tuple(str(set_id) for set_id in transaction.unresolved_intent_set_ids if str(set_id)),
                    *failed_ids,
                )
            )
        )
        missing_intent_ids = tuple(
            deduped_set_ids(
                tuple(str(set_id) for set_id in transaction.missing_intent_set_ids if str(set_id))
            )
        )
        failed_intent_ids = tuple(
            deduped_set_ids(
                (
                    *tuple(str(set_id) for set_id in transaction.failed_intent_set_ids if str(set_id)),
                    *failed_ids,
                )
            )
        )
        semantic_unavailable_ids = tuple(
            deduped_set_ids(
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
            semantic_ids = tuple(deduped_set_ids(tuple(str(set_id) for set_id in semantic_ids if str(set_id))))
            unresolved = tuple(deduped_set_ids((*unresolved_intent_ids, *failed_ids, *semantic_ids)))
            missing_ids = tuple(deduped_set_ids(missing_intent_ids))
            return self.publish_completed_run_display_unavailable(
                cause=DisplayTransitionCause.SEMANTIC_METADATA_UNAVAILABLE,
                affected_set_ids=tuple(deduped_set_ids((*unresolved, *missing_ids))) or semantic_ids,
                requested_show_set_ids=intent_ids,
                requested_labels_by_set_id=transaction.intent.labels_by_set_id,
                attempted_display_set_ids=completed_ids,
                unresolved_intent_set_ids=unresolved,
                missing_intent_set_ids=missing_ids,
                failed_intent_set_ids=failed_intent_ids,
                semantic_unavailable_set_ids=tuple(deduped_set_ids((*semantic_unavailable_ids, *semantic_ids))),
            )

        displayable_entries: list[CompletionDisplayEntry] = []
        for completion_entry in completion_entries:
            if not isinstance(completion_entry.completion_provenance, Mapping):
                return self._simulation_no_display_outcome(
                    DisplayTransitionCause.NO_DISPLAYABLE_COMPLETION_RESULTS,
                    affected_set_ids=tuple(
                        deduped_set_ids(
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
                        deduped_set_ids(
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
                        deduped_set_ids(
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
                        deduped_set_ids(
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
                    deduped_set_ids((*semantic_unavailable_ids, str(completion_entry.set_id)))
                )
                continue
            if not self._completion_entry_matches_intent_owned_species(transaction, completion_entry):
                semantic_unavailable_ids = tuple(
                    deduped_set_ids((*semantic_unavailable_ids, str(completion_entry.set_id)))
                )
                continue
            displayable_entries.append(completion_entry)
        display_series_by_set_id: Dict[str, Dict[str, object]] = {}
        for completion_entry in displayable_entries:
            display_series = self._completion_entry_display_series(completion_entry)
            if display_series is None:
                semantic_unavailable_ids = tuple(
                    deduped_set_ids((*semantic_unavailable_ids, str(completion_entry.set_id)))
                )
                continue
            display_series_by_set_id[str(completion_entry.set_id)] = display_series
        displayable_entries = [
            entry
            for entry in displayable_entries
            if str(entry.set_id) in display_series_by_set_id
        ]
        if not displayable_entries:
            return semantic_unavailable(semantic_unavailable_ids or expected_ids)
        if str(primary.set_id) not in {str(entry.set_id) for entry in displayable_entries}:
            primary = displayable_entries[0]
        additional_metadata: List[DisplaySetMetadata] = []
        for completion_entry in displayable_entries:
            if str(completion_entry.set_id) == str(primary.set_id):
                continue
            display_metadata = display_metadata_for_entry(
                label=str(completion_entry.label),
                entry={
                    **completion_entry.to_display_payload(),
                    "series": display_series_by_set_id[str(completion_entry.set_id)],
                    "display_species": tuple(completion_entry.display_species),
                },
                set_id=str(completion_entry.set_id),
                role=DisplaySetRole.RESULT_OVERLAY,
                layer_id=f"result:{completion_entry.set_id}",
                owned_species=tuple(str(name) for name in completion_entry.owned_species if str(name)),
            )
            if display_metadata is not None:
                additional_metadata.append(display_metadata)
        primary_owned_species = tuple(str(name) for name in primary.owned_species if str(name))
        if not primary_owned_species:
            return semantic_unavailable((str(primary.set_id),))
        primary_payload = primary.to_display_payload()
        primary_series = display_series_by_set_id[str(primary.set_id)]
        primary_payload["series"] = primary_series
        displayable_ids = tuple(str(entry.set_id) for entry in displayable_entries if str(entry.set_id))
        outcome = self._apply_batch_display_transaction(
            t=np.asarray(primary.t, dtype=float),
            series=primary_series,
            label=str(primary.label),
            metadata_applier=lambda plot, active_transaction: self._apply_completed_run_plot_metadata(
                plot=plot,
                active_display_transaction=active_transaction,
                completion_entries=list(displayable_entries),
                primary=primary,
                display_series_by_set_id=display_series_by_set_id,
            ),
            annotation_entry=primary_payload,
            primary_set_id=str(primary.set_id),
            primary_label=str(primary.label),
            display_set_ids=displayable_ids,
            display_species=tuple(primary.display_species),
            completion_provenance=primary.completion_provenance,
            owned_species=primary_owned_species,
            display_transition=_DISPLAY_TRANSITION_COMPLETED_RUN_FINAL,
            additional_metadata=tuple(additional_metadata),
            requested_show_set_ids=transaction.intent.requested_show_set_ids,
            requested_labels_by_set_id=transaction.intent.labels_by_set_id,
            run_target_set_ids=transaction.intent.run_target_set_ids,
            unresolved_intent_set_ids=tuple(
                deduped_set_ids((*unresolved_intent_ids, *semantic_unavailable_ids))
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

    def publish_fresh_preview_from_entries(
        self,
        *,
        fresh_preview_entries: Mapping[str, FreshPreviewDisplayEntry],
        requested_show_set_ids: Sequence[str],
        target_set_ids: Sequence[str],
        prefer_set: Optional[str],
        cache_key: str,
        request_id: Optional[int],
        run_id: Optional[int],
    ) -> Optional[SimulationCompletionDisplayOutcome]:
        entries_by_id = {
            str(set_id): entry
            for set_id, entry in dict(fresh_preview_entries or {}).items()
            if str(set_id) and isinstance(entry, FreshPreviewDisplayEntry)
        }
        if not entries_by_id:
            return None
        requested_show_ids = tuple(deduped_set_ids(requested_show_set_ids or ()))
        target_ids = tuple(deduped_set_ids(target_set_ids or ()))
        if not target_ids:
            return None
        display_ids = requested_show_ids or target_ids
        if (
            not display_ids
            or set(display_ids) != set(target_ids)
            or set(target_ids) != set(entries_by_id)
        ):
            return None
        primary_id = str(prefer_set or "").strip()
        if primary_id not in display_ids:
            primary_id = str(display_ids[0])
        transaction = FreshPreviewDisplayTransaction(
            entries=tuple(entries_by_id[set_id] for set_id in display_ids),
            display_set_ids=display_ids,
            target_set_ids=target_ids,
            display_primary_set_id=primary_id,
            cache_key=str(cache_key or ""),
            display_source=DisplayRefreshSource.SLIDER_REPLAY,
            requested_show_set_ids=requested_show_ids or display_ids,
            requested_labels_by_set_id={
                str(set_id): str(self._ui.batch_name_for_id(str(set_id)) or set_id)
                for set_id in (requested_show_ids or display_ids)
                if str(set_id)
            },
            request_id=(int(request_id) if request_id is not None else None),
            run_id=(int(run_id) if run_id is not None else None),
        )
        return self.publish_fresh_preview_display(transaction)

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
            owned_species = tuple(str(name) for name in entry.owned_species if str(name))
            if not owned_species:
                return self._simulation_no_display_outcome(
                    DisplayTransitionCause.SEMANTIC_METADATA_UNAVAILABLE,
                    affected_set_ids=(str(entry.set_id),),
                    unresolved_intent_set_ids=(str(entry.set_id),),
                    semantic_unavailable_set_ids=(str(entry.set_id),),
                    outcome_kind=DisplayTransitionOutcomeKind.FAILED,
                    display_status=DisplayStatus.NO_COMPLETE_DISPLAYABLE_REQUEST_SCOPE,
                )
            display_series = series_for_display_species(
                series=entry.series,
                display_species=entry.display_species,
            )
            if display_series is None:
                return self._simulation_no_display_outcome(
                    DisplayTransitionCause.NO_DISPLAYABLE_PREVIEW_RESULTS,
                    affected_set_ids=(str(entry.set_id),),
                )
            if any(species_name not in display_series for species_name in owned_species):
                return self._simulation_no_display_outcome(
                    DisplayTransitionCause.SEMANTIC_METADATA_UNAVAILABLE,
                    affected_set_ids=(str(entry.set_id),),
                    unresolved_intent_set_ids=(str(entry.set_id),),
                    semantic_unavailable_set_ids=(str(entry.set_id),),
                    outcome_kind=DisplayTransitionOutcomeKind.FAILED,
                    display_status=DisplayStatus.NO_COMPLETE_DISPLAYABLE_REQUEST_SCOPE,
                )
            series_by_set_id[str(entry.set_id)] = display_series

        additional_metadata: List[DisplaySetMetadata] = []
        for entry in entries:
            if str(entry.set_id) == str(primary.set_id):
                continue
            display_metadata = display_metadata_for_entry(
                label=str(entry.label),
                entry={
                    **entry.to_display_payload(),
                    "series": series_by_set_id[str(entry.set_id)],
                    "display_species": tuple(entry.display_species),
                },
                set_id=str(entry.set_id),
                role=DisplaySetRole.RESULT_OVERLAY,
                layer_id=f"result:{entry.set_id}",
                owned_species=tuple(str(name) for name in entry.owned_species if str(name)),
                workspace_preview_provenance=entry.workspace_preview_provenance,
            )
            if display_metadata is not None:
                additional_metadata.append(display_metadata)
        for entry in entries:
            display_metadata = None
            canonical_reference_entry = entry.authority.canonical_reference_entry
            if isinstance(canonical_reference_entry, Mapping):
                reference_entry = dict(canonical_reference_entry)
                display_metadata = display_metadata_for_entry(
                    label=str(entry.label),
                    entry={
                        **reference_entry,
                        "display_species": tuple(
                            str(name)
                            for name in (reference_entry.get("display_species") or ())
                            if str(name)
                        ),
                    },
                    set_id=str(entry.set_id),
                    role=DisplaySetRole.REFERENCE_OVERLAY,
                    layer_id=f"reference:{entry.set_id}",
                    visible=self._reference_overlays_visible,
                )
            if display_metadata is not None:
                additional_metadata.append(display_metadata)
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
            metadata_applier=lambda plot, active_transaction: self._apply_active_transaction_plot_metadata(
                plot=plot,
                active_display_transaction=active_transaction,
                primary_t=np.asarray(primary.t, dtype=float),
                primary_series=primary_series,
                algebra_scalars=primary.algebra_scalars,
                prefer_layer_id=f"result:{primary.set_id}" if primary.set_id else "result:preview",
                context_label=f"fresh preview display (primary={primary.label or primary.set_id or 'Preview'})",
            ),
            annotation_entry={"solver_provenance": primary.solver_provenance},
            primary_set_id=str(primary.set_id),
            primary_label=str(primary.label),
            display_set_ids=expected_ids,
            display_species=tuple(primary.display_species),
            workspace_preview_provenance_by_set_id=workspace_provenance_by_set_id,
            completion_provenance=primary.completion_provenance,
            owned_species=tuple(str(name) for name in primary.owned_species if str(name)),
            display_transition=(
                _DISPLAY_TRANSITION_FRESH_PREVIEW_REPLACE_ACTIVE
                if display_source
                in {DisplayRefreshSource.EXPLICIT_SHOW_REQUEST, DisplayRefreshSource.SLIDER_REPLAY}
                else _DISPLAY_TRANSITION_FRESH_PREVIEW
            ),
            additional_metadata=tuple(additional_metadata),
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

    def publish_slider_replay_display_scope(
        self,
        *,
        cache_admin: object | None,
        cache_key: str,
        cache_kind: str,
        requested_show_set_ids: Sequence[str],
        target_set_ids: Sequence[str],
        prefer_set: Optional[str],
        valid_set_ids: Sequence[str],
        fresh_preview_entries: Mapping[str, FreshPreviewDisplayEntry],
        request_id: Optional[int],
        run_id: Optional[int],
    ) -> Optional[SimulationCompletionDisplayOutcome]:
        if str(cache_kind or "") == "preview":
            fresh_outcome = self.publish_fresh_preview_from_entries(
                fresh_preview_entries=fresh_preview_entries,
                requested_show_set_ids=requested_show_set_ids,
                target_set_ids=target_set_ids,
                prefer_set=prefer_set,
                cache_key=cache_key,
                request_id=request_id,
                run_id=run_id,
            )
            if _display_transition_published(fresh_outcome):
                return fresh_outcome
            if fresh_preview_entries:
                return fresh_outcome
            outcome = self.refresh_display_from_request_scope(
                display_source=DisplayRefreshSource.SLIDER_REPLAY,
            )
            if _display_transition_published(outcome):
                return outcome
            return fresh_outcome
        if valid_set_ids:
            publisher = getattr(cache_admin, "publish_completion_cache_truth", None)
            if callable(publisher):
                publisher(
                    is_preview=False,
                    cache_key=str(cache_key),
                    clear_active_cache_identity_state=False,
                    active_cache_key=str(cache_key),
                    active_cache_preview_token=None,
                    active_cache_preview_scope_set_ids=None,
                    active_cache_valid_set_ids=valid_set_ids,
                    active_cache_invalidated_set_ids=None,
                )
        return self.publish_cached_batch_display_scope(
            cache_key=str(cache_key),
            requested_show_set_ids=requested_show_set_ids,
            prefer_set=prefer_set,
            display_source=DisplayRefreshSource.SLIDER_REPLAY,
        )

    def publish_runtime_slider_replay_display(
        self,
        *,
        cache_admin: object | None,
        cache_key: str,
        cache_kind: str,
        live_requested_show_set_ids: Sequence[str],
        target_set_ids: Sequence[str],
        focused_set_id: Optional[str],
        valid_set_ids: Sequence[str],
        fresh_preview_entries: Mapping[str, FreshPreviewDisplayEntry],
        request_id: Optional[int],
        run_id: Optional[int],
        accepted_preview_request_id: Optional[int],
        accepted_preview_owner_epoch: Optional[int],
        current_preview_request_id: Optional[int],
        current_preview_owner_epoch: Optional[int],
        latest_request_id: int,
        active_run_id: int,
    ) -> Optional[SimulationCompletionDisplayOutcome]:
        cache_kind_s = str(cache_kind or "")
        request_current = self._runtime_slider_replay_request_is_current(
            cache_kind=cache_kind_s,
            request_id=request_id,
            accepted_preview_request_id=accepted_preview_request_id,
            accepted_preview_owner_epoch=accepted_preview_owner_epoch,
            current_preview_request_id=current_preview_request_id,
            current_preview_owner_epoch=current_preview_owner_epoch,
            latest_request_id=latest_request_id,
        )
        if request_id is not None and not bool(request_current):
            return None
        if run_id is not None and int(run_id) != int(active_run_id):
            return None
        target_ids = tuple(str(set_id) for set_id in target_set_ids or () if str(set_id))
        live_requested_ids = tuple(
            str(set_id) for set_id in live_requested_show_set_ids or () if str(set_id)
        )
        requested_show_ids = live_requested_ids
        if not requested_show_ids:
            return None
        prefer_set = str(focused_set_id or "") or None
        return self.publish_slider_replay_display_scope(
            cache_admin=cache_admin,
            cache_key=str(cache_key),
            cache_kind=str(cache_kind),
            requested_show_set_ids=requested_show_ids,
            target_set_ids=target_ids,
            prefer_set=prefer_set,
            valid_set_ids=valid_set_ids,
            fresh_preview_entries=fresh_preview_entries,
            request_id=request_id,
            run_id=run_id,
        )

    def publish_runtime_slider_replay_display_from_pending(
        self,
        *,
        cache_admin: object | None,
        pending: object,
        cache_key: Optional[str] = None,
        request_id: Optional[int] = None,
        run_id: Optional[int] = None,
        current_preview_request_id: Optional[int],
        current_preview_owner_epoch: Optional[int],
        latest_request_id: int,
        active_run_id: int,
    ) -> RuntimeSliderReplayDisplayRefresh:
        pending_cache_key = str(getattr(pending, "cache_key", "") or "")
        pending_cache_kind = str(getattr(pending, "cache_kind", "") or "")
        resolved_cache_key = str(cache_key or pending_cache_key or "")
        if not resolved_cache_key:
            return RuntimeSliderReplayDisplayRefresh()
        resolved_request_id = (
            getattr(pending, "request_id", None)
            if request_id is None
            else request_id
        )
        resolved_run_id = getattr(pending, "run_id", None) if run_id is None else run_id
        target_set_ids = tuple(
            str(set_id)
            for set_id in getattr(pending, "target_set_ids", ())
            if str(set_id)
        )
        try:
            live_requested_show_set_ids = tuple(
                str(set_id)
                for set_id in (self._ui.requested_show_batch_set_ids() or ())
                if str(set_id)
            )
        except Exception as exc:
            logger.debug("Failed to snapshot requested Show ids for slider replay: %s", exc, exc_info=True)
            live_requested_show_set_ids = ()
        try:
            focused_set_id = self._ui.focused_batch_set_id()
        except Exception as exc:
            logger.debug("Failed to snapshot focused set id for slider replay: %s", exc, exc_info=True)
            focused_set_id = None
        display_outcome = self.publish_runtime_slider_replay_display(
            cache_admin=cache_admin,
            cache_key=resolved_cache_key,
            cache_kind=pending_cache_kind,
            live_requested_show_set_ids=live_requested_show_set_ids,
            target_set_ids=target_set_ids,
            focused_set_id=focused_set_id,
            valid_set_ids=getattr(pending, "valid_set_ids", ()),
            fresh_preview_entries=dict(getattr(pending, "fresh_preview_entries", {}) or {}),
            request_id=resolved_request_id,
            run_id=resolved_run_id,
            accepted_preview_request_id=getattr(pending, "accepted_preview_request_id", None),
            accepted_preview_owner_epoch=getattr(pending, "accepted_preview_owner_epoch", None),
            current_preview_request_id=current_preview_request_id,
            current_preview_owner_epoch=current_preview_owner_epoch,
            latest_request_id=int(latest_request_id),
            active_run_id=int(active_run_id),
        )
        displayed = _display_transition_published(display_outcome)
        requested_show_ids_for_focus = (
            target_set_ids
            if pending_cache_kind == "preview" and target_set_ids
            else live_requested_show_set_ids
        )
        focus_sync = (
            getattr(display_outcome, "focused_controls_use_workspace", None)
            if bool(displayed) and pending_cache_kind == "preview"
            else None
        )
        if bool(displayed) and pending_cache_kind == "preview" and focus_sync is None:
            try:
                focus_sync = bool(
                    self._ui.focused_show_request_is_dirty(
                        requested_show_ids_for_focus,
                        focused_set_id,
                    )
                )
            except Exception as exc:
                logger.debug(
                    "Failed to derive focused workspace sync for slider replay refresh: %s",
                    exc,
                    exc_info=True,
                )
                focus_sync = None
        return RuntimeSliderReplayDisplayRefresh(
            display_outcome=display_outcome,
            displayed=bool(displayed),
            focused_controls_use_workspace=focus_sync,
            log_reason=_display_transition_log_reason(display_outcome),
        )

    @staticmethod
    def _runtime_slider_replay_request_is_current(
        *,
        cache_kind: str,
        request_id: Optional[int],
        accepted_preview_request_id: Optional[int],
        accepted_preview_owner_epoch: Optional[int],
        current_preview_request_id: Optional[int],
        current_preview_owner_epoch: Optional[int],
        latest_request_id: int,
    ) -> bool:
        if request_id is None:
            return True
        if str(cache_kind or "") != "preview":
            return int(request_id) == int(latest_request_id)
        if (
            accepted_preview_request_id is None
            or accepted_preview_owner_epoch is None
            or current_preview_request_id is None
            or current_preview_owner_epoch is None
        ):
            return False
        return (
            int(current_preview_request_id) == int(request_id)
            and int(accepted_preview_request_id) == int(request_id)
            and int(current_preview_owner_epoch) == int(accepted_preview_owner_epoch)
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
        owned_species: Sequence[str],
        display_species: Sequence[str],
        solver_provenance: Mapping[str, Any] | None = None,
    ) -> SimulationCompletionDisplayOutcome:
        set_id = str(batch_set_id or "").strip()
        set_name = str(batch_set or "").strip()
        primary_layer_id = f"result:{set_id}" if set_id else "result:live"
        display_label = set_name or set_id or "Results"
        resolved_owned_species = tuple(str(name) for name in (owned_species or ()) if str(name))
        if not resolved_owned_species:
            return self._simulation_no_display_outcome(
                DisplayTransitionCause.SEMANTIC_METADATA_UNAVAILABLE,
                affected_set_ids=([set_id] if set_id else []),
                unresolved_intent_set_ids=([set_id] if set_id else []),
                semantic_unavailable_set_ids=([set_id] if set_id else []),
                outcome_kind=DisplayTransitionOutcomeKind.FAILED,
                display_status=DisplayStatus.NO_COMPLETE_DISPLAYABLE_REQUEST_SCOPE,
            )
        display_series = series_for_display_species(
            series=series,
            display_species=display_species,
        )
        if display_series is None or any(species_name not in display_series for species_name in resolved_owned_species):
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
            series=display_series,
            label=(set_name or display_label),
            metadata_applier=lambda plot, _active_transaction: self._apply_direct_completion_plot_metadata(
                plot=plot,
                t=np.asarray(t, dtype=float),
                series=display_series,
                display_label=display_label,
                algebra_scalars=algebra_scalars,
                layer_id=primary_layer_id,
                set_id=set_id,
            ),
            annotation_entry={"solver_provenance": solver_provenance},
            primary_set_id=set_id,
            primary_label=set_name,
            display_set_ids=([set_id] if set_id else []),
            display_species=tuple(str(name) for name in (display_species or ()) if str(name)),
            completion_provenance=direct_completion_provenance,
            owned_species=resolved_owned_species,
            display_transition=_DISPLAY_TRANSITION_DIRECT_RAW,
            run_target_set_ids=([set_id] if set_id else []),
        )
        return SimulationCompletionDisplayOutcome(
            transition_outcome=outcome.transition_outcome,
        )

    def _set_plot_display_layers(
        self,
        *,
        plot: ResultsDisplayPlotPort,
        active_display_transaction: ActiveDisplayTransaction,
    ) -> bool:
        try:
            projected_transaction = self._projected_transaction_for_plot(active_display_transaction)
            if projected_transaction is None:
                plot.clear_display_transaction_state(preserve_y_selection_state=True)
                return True
            plot.set_display_layers(
                plot_display_layers_payload(
                    projected_transaction,
                    presentation_labels_by_set_id=self._popup_labels_by_set_id(
                        projected_transaction.display_set_ids,
                    ),
                )
            )
            return True
        except Exception as exc:
            logger.warning("Failed to render display layers: %s", exc, exc_info=True)
            QtWidgets.QMessageBox.warning(self._ui.parent, "Error", f"Failed to render display layers: {exc}")
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
            self._ui.set_status_text(display_transition_status_text(transition_outcome))
            logger.info("Data set: %s species, %s points", int(len(series)), int(len(t)))
        except Exception as exc:
            logger.exception("Failed to apply post-commit plot display UI refresh: %s", exc)

    def _apply_intervention_annotations(self, *, plot: ResultsDisplayPlotPort, entry: Mapping[str, Any]) -> None:
        solver_provenance = entry.get("solver_provenance") if isinstance(entry, Mapping) else None
        plot.set_intervention_annotations_from_provenance(
            solver_provenance if isinstance(solver_provenance, Mapping) else None
        )
