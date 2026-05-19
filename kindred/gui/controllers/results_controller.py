from __future__ import annotations

import logging
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
from PySide6 import QtCore, QtWidgets

from kindred.core.batch_simulation_cache import BatchSimulationCache
from kindred.core.batch_cache_contracts import (
    BatchCacheEntryReadResult,
    BatchCacheEntryV1,
    build_overlay_entry,
    read_batch_cache_entry,
)
from kindred.gui.ports import (
    BatchDisplayRefreshOutcome,
    BatchDisplaySelectionResolution,
    CachedBatchSelectionDisplayOutcome,
    CompletionDisplayEntry,
    CompletedRunDisplayTransaction,
    ResolvedBatchSelectionEntry,
    SimulationCompletionDisplayOutcome,
)

logger = logging.getLogger(__name__)

__all__ = ["ResultsController"]


@dataclass(frozen=True)
class ResultsControllerPort:
    parent: QtCore.QObject
    main_plot: Callable[[], object]
    main_plot_has_data: Callable[[], bool]
    main_plot_selected_series: Callable[[], list[str]]
    set_main_plot_selected_series: Callable[[Sequence[str]], None]
    batch_name_for_id: Callable[[str], str | None]
    batch_id_for_name: Callable[[str], str | None]
    shown_batch_set_ids: Callable[[], list[str]]
    focused_batch_set_id: Callable[[], str | None]
    selected_batch_set_ids: Callable[[], list[str]]
    current_batch_row: Callable[[], int | None]
    batch_set_id_for_row: Callable[[int], str | None]
    batch_row_for_set_id: Callable[[str], int | None]
    active_batch_cache_key: Callable[[], str]
    active_batch_valid_set_ids: Callable[[], Sequence[str] | None]
    active_batch_invalidated_set_ids: Callable[[], Sequence[str] | None]
    active_batch_selection: Callable[[], tuple[str, str]]
    set_active_batch_selection: Callable[[str, str, Sequence[str]], None]
    clear_display_selection_state: Callable[[], None]
    clear_active_preview_selection_state: Callable[[], None]
    last_display_selection: Callable[[], Sequence[str]]
    last_simulation_provenance: Callable[[], Mapping[str, Any]]
    last_simulation_ctc: Callable[[], Mapping[str, float]]
    set_last_simulation_provenance: Callable[[Dict[str, Any]], None]
    set_last_simulation_ctc: Callable[[Dict[str, float]], None]
    publish_simulation_completion_provenance: Callable[..., Dict[str, Any]]
    result_cache_store: Callable[[], MutableMapping[str, dict[str, object]]]
    set_main_plot_scalar_values: Callable[[dict[str, object]], None]
    update_main_plot_statistics: Callable[..., None]
    main_plot_stats_table: Callable[[], object]
    set_results_table: Callable[[object], None]
    set_main_plot_data: Callable[..., None]
    show_simulation_tab: Callable[[], None]
    refresh_simulation_plot_views: Callable[[], None]
    schedule_main_plot_refresh: Callable[[Sequence[int]], None]
    set_status_text: Callable[[str], None]
    status_text_getter: Callable[[], str]
    update_batch_row_controls_state: Callable[[], None]
    focused_batch_selection_is_dirty: Callable[[Sequence[str], Optional[str]], bool]
    focused_batch_set_is_dirty: Callable[[], bool]
    selection_uses_fresh_explicit_cache_after_post_run_sync: Callable[[Sequence[str]], bool]
    workspace_selection_resolution: Callable[[Sequence[str]], "BatchDisplaySelectionResolution"]
    preview_launch_pending: Callable[[], bool]
    current_workspace_preview_identity_payload: Callable[[str], Optional[Dict[str, Any]]]
    active_explicit_entry_matches_displayed_entry: Callable[[str, Mapping[str, Any]], bool]


@dataclass(frozen=True, slots=True)
class MainPlotSnapshot:
    had_existing_data: bool
    previous_selection: List[str]


@dataclass(frozen=True, slots=True)
class CachedBatchAvailability:
    available_ids: List[str]
    has_invalid_entry: bool = False


@dataclass(frozen=True, slots=True)
class CachedBatchSelectionCoverage:
    selected_ids: List[str]
    available_ids: List[str]
    full_coverage: bool
    reason: Optional[str] = None


@dataclass(frozen=True, slots=True)
class BatchDisplayRefreshRequest:
    shown_set_ids: tuple[str, ...] = ()
    prefer_set_id: Optional[str] = None
    active_cache_key: str = ""
    focused_dirty: bool = False
    focused_set_dirty: bool = False
    fresh_explicit_cache_after_post_run_sync: bool = False
    active_cache_valid_set_ids: Optional[tuple[str, ...]] = None
    active_cache_invalidated_set_ids: Optional[tuple[str, ...]] = None
    resolution: BatchDisplaySelectionResolution = BatchDisplaySelectionResolution()


@dataclass(frozen=True, slots=True)
class AuthoritativeResultDisplayTransitionOutcome:
    refresh_requested: bool = False


class ResultsController(QtCore.QObject):
    """
    Results + plot presentation controller.

    This keeps `MainWindow` focused on UI composition while preserving behavior
    behind a narrow results-specific UI port.
    """

    def __init__(self, ui: ResultsControllerPort):
        super().__init__(ui.parent)
        self._ui = ui

    def _main_plot(self) -> object | None:
        try:
            return self._ui.main_plot()
        except Exception as exc:
            logger.debug("Main plot unavailable through MainWindow seam: %s", exc, exc_info=True)
            return None

    def main_plot(self) -> object:
        return self._ui.main_plot()

    def update_main_plot_parameter_summary(self, parameters: Dict[str, tuple[float, str]]) -> None:
        plot = self._ui.main_plot()
        if hasattr(plot, "update_parameters"):
            plot.update_parameters(dict(parameters))

    def set_results_table(self, table: object) -> None:
        self._ui.set_results_table(table)

    def _main_plot_snapshot(self, plot: object) -> MainPlotSnapshot:
        had_existing_data = False
        previous_selection: List[str] = []
        try:
            had_existing_data = bool(self._ui.main_plot_has_data())
        except Exception as exc:
            logger.debug("Failed to inspect existing main-plot data: %s", exc, exc_info=True)
        try:
            previous_selection = list(self._ui.main_plot_selected_series())
        except Exception as exc:
            logger.debug("Failed to snapshot main-plot selection: %s", exc, exc_info=True)
        return MainPlotSnapshot(
            had_existing_data=bool(had_existing_data),
            previous_selection=previous_selection,
        )

    def _restore_main_plot_selection(self, snapshot: MainPlotSnapshot) -> None:
        if not snapshot.had_existing_data and not snapshot.previous_selection:
            return
        try:
            self._ui.set_main_plot_selected_series(snapshot.previous_selection)
        except Exception as exc:
            logger.exception("Failed to restore plot series selection after set_data(): %s", exc)

    def _normalize_batch_set_id(self, token: str) -> Optional[str]:
        raw = str(token or "").strip()
        if not raw:
            return None
        if self._ui.batch_name_for_id(raw) is not None:
            return raw
        sid = self._ui.batch_id_for_name(raw)
        if sid:
            return sid
        return raw

    def _cache_entry_for_set_id(
        self,
        *,
        store: MutableMapping[str, Dict[str, Any]],
        cache_key: str,
        set_id: str,
        require_completion_provenance: bool = False,
    ) -> BatchCacheEntryReadResult:
        sid = str(set_id or "").strip()
        if not sid:
            return BatchCacheEntryReadResult("missing")
        direct = read_batch_cache_entry(
            (store or {}).get(BatchSimulationCache.entry_key(cache_key, sid)),
            require_completion_provenance=bool(require_completion_provenance),
        )
        if direct.entry is not None:
            return direct
        name = self._ui.batch_name_for_id(sid)
        by_name = BatchCacheEntryReadResult("missing")
        if name:
            by_name = read_batch_cache_entry(
                (store or {}).get(BatchSimulationCache.entry_key(cache_key, str(name))),
                require_completion_provenance=bool(require_completion_provenance),
            )
            if by_name.entry is not None:
                return by_name
        if direct.state == "invalid" or by_name.state == "invalid":
            return BatchCacheEntryReadResult("invalid")
        return BatchCacheEntryReadResult("missing")

    def _normalized_selected_batch_ids(self, selected_sets: Sequence[str]) -> List[str]:
        selected_ids: List[str] = []
        for token in [str(n) for n in (selected_sets or []) if str(n)]:
            sid = self._normalize_batch_set_id(token)
            if sid and sid not in selected_ids:
                selected_ids.append(str(sid))
        return selected_ids

    def _available_cached_batch_ids(
        self,
        *,
        store: MutableMapping[str, Dict[str, Any]],
        cache_key: str,
        selected_sets: Sequence[str],
        valid_set_ids: Optional[Sequence[str]] = None,
        invalidated_set_ids: Optional[Sequence[str]] = None,
        require_completion_provenance: bool = True,
    ) -> CachedBatchAvailability:
        selected_ids = self._normalized_selected_batch_ids(selected_sets)
        allowed_ids = None
        invalidated_ids = {str(sid) for sid in (invalidated_set_ids or ()) if str(sid)}
        has_invalid_entry = False
        if valid_set_ids is not None:
            allowed_ids = {str(sid) for sid in (valid_set_ids or ()) if str(sid)}
        available: List[str] = []
        for sid in selected_ids:
            if allowed_ids is not None and sid not in allowed_ids:
                continue
            result = self._cache_entry_for_set_id(
                store=store,
                cache_key=cache_key,
                set_id=sid,
                require_completion_provenance=bool(require_completion_provenance),
            )
            if sid in invalidated_ids:
                if result.state == "invalid":
                    has_invalid_entry = True
                continue
            if result.entry is not None:
                available.append(sid)
                continue
            if result.state == "invalid":
                has_invalid_entry = True
        if available:
            return CachedBatchAvailability(available, has_invalid_entry=has_invalid_entry)
        return CachedBatchAvailability([], has_invalid_entry=has_invalid_entry)

    def _primary_cached_batch_id(
        self,
        *,
        store: MutableMapping[str, Dict[str, Any]],
        cache_key: str,
        available: Sequence[str],
        prefer_set: Optional[str],
    ) -> str:
        if isinstance(prefer_set, str):
            prefer_id = self._normalize_batch_set_id(prefer_set)
            if (
                prefer_id
                and prefer_id in available
                and self._cache_entry_for_set_id(
                    store=store,
                    cache_key=cache_key,
                    set_id=prefer_id,
                    require_completion_provenance=True,
                ).entry
                is not None
            ):
                return str(prefer_id)
        focused_id = str(self._ui.focused_batch_set_id() or "")
        if focused_id and focused_id in available:
            focused_entry = self._cache_entry_for_set_id(
                store=store,
                cache_key=cache_key,
                set_id=focused_id,
                require_completion_provenance=True,
            ).entry
            if focused_entry is not None:
                return focused_id
        return str(available[0])

    def _cached_batch_display_entries_by_set_id(
        self,
        *,
        store: MutableMapping[str, Dict[str, Any]],
        cache_key: str,
        available: Sequence[str],
        primary: str,
        primary_entry: Mapping[str, Any],
    ) -> dict[str, Mapping[str, Any]]:
        displayed_entries: dict[str, Mapping[str, Any]] = {str(primary): primary_entry}
        for sid in available:
            if sid == primary:
                continue
            other = self._cache_entry_for_set_id(
                store=store,
                cache_key=cache_key,
                set_id=sid,
                require_completion_provenance=True,
            ).entry
            if other is None:
                continue
            displayed_entries[str(sid)] = other
        return displayed_entries

    def _cached_batch_overlays(
        self,
        *,
        displayed_entries_by_set_id: Mapping[str, Mapping[str, Any]],
        primary: str,
    ) -> List[Dict[str, object]]:
        overlays: List[Dict[str, object]] = []
        for sid, entry in displayed_entries_by_set_id.items():
            if str(sid) == str(primary):
                continue
            overlay_label = self._ui.batch_name_for_id(str(sid)) or str(sid)
            overlays.append(
                dict(
                    build_overlay_entry(
                        label=overlay_label,
                        entry=entry,
                        set_id=str(sid),
                        layer_id=f"result:{sid}",
                        layer_kind="result",
                    )
                )
            )
        return overlays

    def _apply_cached_batch_plot_metadata(
        self,
        *,
        plot: object,
        store: MutableMapping[str, Dict[str, Any]],
        cache_key: str,
        available: Sequence[str],
        displayed_entries_by_set_id: Mapping[str, Mapping[str, Any]],
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
                "Failed to set plot scalar values for cached batch selection (primary=%s): %s",
                primary_label,
                exc,
            )
            return "metadata_scalar_failed"

        stats_results_map: Dict[str, Dict[str, object]] = {}
        for sid in available:
            _ = store, cache_key
            payload = displayed_entries_by_set_id.get(str(sid))
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
                "Failed to update plot statistics for cached batch selection (primary=%s): %s",
                primary_label,
                exc,
            )
            return "metadata_statistics_failed"
        try:
            self._ui.set_results_table(self._ui.main_plot_stats_table())
        except Exception as exc:
            logger.exception("Failed to fetch stats table from plot after cached selection: %s", exc)
            return "metadata_table_failed"
        return None

    def _apply_resolved_batch_plot_metadata(
        self,
        *,
        plot: object,
        resolved_entries: Sequence[ResolvedBatchSelectionEntry],
        primary: ResolvedBatchSelectionEntry,
    ) -> Optional[str]:
        scalars = primary.entry.get("algebra_scalars") or {}
        try:
            self._ui.set_main_plot_scalar_values(scalars)
        except Exception as exc:
            logger.exception(
                "Failed to set plot scalar values for resolved batch selection (primary=%s): %s",
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
                "Failed to update plot statistics for resolved batch selection (primary=%s): %s",
                primary.label,
                exc,
            )
            return "metadata_statistics_failed"
        try:
            self._ui.set_results_table(self._ui.main_plot_stats_table())
        except Exception as exc:
            logger.exception("Failed to fetch stats table from plot after resolved selection: %s", exc)
            return "metadata_table_failed"
        return None

    def _apply_completed_run_plot_metadata(
        self,
        *,
        plot: object,
        completion_entries: Sequence[CompletionDisplayEntry],
        primary: CompletionDisplayEntry,
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

        stats_results_map: Dict[str, Dict[str, object]] = {}
        for completion_entry in completion_entries:
            series_payload = completion_entry.series or {}
            if not series_payload:
                continue
            result_layer_id = f"result:{completion_entry.set_id}"
            stats_results_map[result_layer_id] = {
                "t": completion_entry.t,
                "series": dict(series_payload),
                "label": str(completion_entry.label),
                "layer_id": result_layer_id,
                "layer_kind": "result",
                "set_id": str(completion_entry.set_id),
            }
        try:
            self._ui.update_main_plot_statistics(
                stats_results_map=stats_results_map,
                prefer=f"result:{primary.set_id}",
                t=np.asarray(primary.t, dtype=float),
                series={
                    str(k): np.asarray(v, dtype=float)
                    for k, v in (primary.series or {}).items()
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
            self._ui.set_results_table(self._ui.main_plot_stats_table())
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
            table_getter = getattr(plot, "stats_table", None)
            table = table_getter() if callable(table_getter) else self._ui.main_plot_stats_table()
            self._ui.set_results_table(table)
        except Exception as exc:
            logger.exception("Failed to update results table after simulation completion: %s", exc)
            return "metadata_table_failed"
        return None

    def _publish_direct_completion_provenance(
        self,
        *,
        plot: object,
        direct_completion_provenance: Mapping[str, Any] | None,
    ) -> Optional[str]:
        if not isinstance(direct_completion_provenance, Mapping):
            self._clear_direct_completion_provenance()
            return None
        payload = dict(direct_completion_provenance)
        try:
            overlay_snapshot = getattr(plot, "overlay_snapshot", None)
            payload["dataset_overlays"] = overlay_snapshot() if callable(overlay_snapshot) else None
            self._ui.publish_simulation_completion_provenance(**payload)
        except Exception as exc:
            logger.exception("Failed to publish direct completion display provenance: %s", exc)
            return "direct_provenance_failed"
        return None

    def _clear_direct_completion_provenance(self) -> None:
        self._ui.set_last_simulation_provenance({})
        self._ui.set_last_simulation_ctc({})

    def _direct_completion_provenance_snapshot(self) -> Dict[str, Dict[str, Any]]:
        return {
            "provenance": dict(self._ui.last_simulation_provenance() or {}),
            "ctc": dict(self._ui.last_simulation_ctc() or {}),
        }

    def _restore_direct_completion_provenance_snapshot(self, snapshot: Mapping[str, Mapping[str, Any]]) -> None:
        try:
            self._ui.set_last_simulation_provenance(dict(snapshot.get("provenance") or {}))
            self._ui.set_last_simulation_ctc(dict(snapshot.get("ctc") or {}))
        except Exception as exc:
            logger.exception("Failed to restore direct completion provenance after display transaction failure: %s", exc)

    @staticmethod
    def _direct_display_failure_snapshot(plot: object | None) -> Optional[Dict[str, Any]]:
        if plot is None:
            return None
        snapshotter = getattr(plot, "display_transaction_snapshot", None)
        if not callable(snapshotter):
            return None
        try:
            snapshot = snapshotter()
        except Exception as exc:
            logger.exception("Failed to snapshot plot display transaction state: %s", exc)
            return None
        if not isinstance(snapshot, Mapping):
            return None
        snapshot_dict = dict(snapshot)
        return snapshot_dict if bool(snapshot_dict.get("_kindred_display_transaction_snapshot")) else None

    def _restore_direct_display_failure_snapshot(self, snapshot: Optional[Mapping[str, Any]]) -> None:
        plot = self._main_plot()
        if plot is None or not isinstance(snapshot, Mapping):
            return
        if not bool(snapshot.get("_kindred_display_transaction_snapshot")):
            return
        restorer = getattr(plot, "restore_display_transaction_snapshot", None)
        if not callable(restorer):
            return
        try:
            restorer(snapshot)
        except Exception as exc:
            logger.exception("Failed to restore plot after failed direct display transaction: %s", exc)
            return
        try:
            self._ui.set_results_table(self._ui.main_plot_stats_table())
        except Exception as exc:
            logger.exception("Failed to restore results table after display transaction failure: %s", exc)

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

    def _sync_main_plot_copy_labels(self, *, primary_set_id: str, selected_set_ids: Sequence[str]) -> None:
        primary_set_id_s = str(primary_set_id or "").strip()
        selected_ids: list[str] = []
        for raw_set_id in selected_set_ids or ():
            set_id = str(raw_set_id or "").strip()
            if not set_id or set_id in selected_ids:
                continue
            selected_ids.append(set_id)
        if primary_set_id_s and primary_set_id_s not in selected_ids:
            selected_ids.insert(0, primary_set_id_s)
        plot = self._main_plot()
        setter = getattr(plot, "set_simulation_popup_labels", None) if plot is not None else None
        if callable(setter):
            setter(
                primary_set_id=primary_set_id_s,
                popup_labels_by_set_id=self._popup_labels_by_set_id(selected_ids),
            )

    def _plot_display_claim_snapshot(self) -> Dict[str, Any]:
        plot = self._main_plot()
        plot_snapshot: Mapping[str, Any] = {}
        snapshotter = getattr(plot, "display_claim_snapshot", None) if plot is not None else None
        if callable(snapshotter):
            try:
                candidate = snapshotter()
                if isinstance(candidate, Mapping):
                    plot_snapshot = candidate
            except Exception as exc:
                logger.exception("Failed to snapshot plot display claim state: %s", exc)
        return {
            "active_set_id": self._ui.active_batch_selection()[0],
            "active_set_name": self._ui.active_batch_selection()[1],
            "last_display_selection": list(self._ui.last_display_selection() or ()),
            "plot": dict(plot_snapshot),
        }

    def _restore_plot_display_claim_snapshot(self, snapshot: Mapping[str, Any]) -> None:
        active_set_id = str(snapshot.get("active_set_id") or "")
        active_set_name = str(snapshot.get("active_set_name") or "")
        selected_ids = [str(set_id) for set_id in (snapshot.get("last_display_selection") or ()) if str(set_id)]
        if active_set_id or active_set_name or selected_ids:
            self._ui.set_active_batch_selection(active_set_id, active_set_name, selected_ids)
        else:
            self._ui.clear_display_selection_state()
        plot = self._main_plot()
        restorer = getattr(plot, "restore_display_claim_snapshot", None) if plot is not None else None
        if callable(restorer):
            try:
                restorer(snapshot.get("plot") or {})
            except Exception as exc:
                logger.exception("Failed to restore plot display claim state: %s", exc)

    def clear_batch_display_publication(self) -> None:
        plot = self._main_plot()
        if plot is None:
            return
        clear_display = getattr(plot, "clear_display_transaction_state", None)
        if callable(clear_display):
            clear_display()
        self._clear_direct_completion_provenance()
        self._ui.show_simulation_tab()
        self._ui.refresh_simulation_plot_views()

    def clear_display_if_workspace_previews_were_shown(
        self,
        set_ids: Sequence[str],
        *,
        pre_reset_display_snapshot: Optional[Mapping[str, Any]] = None,
    ) -> bool:
        displayed_set_ids = self._displayed_workspace_preview_set_ids(
            set_ids,
            pre_reset_display_snapshot=pre_reset_display_snapshot,
        )
        if not displayed_set_ids or not self.main_plot_has_data():
            return False
        self._ui.clear_active_preview_selection_state()
        self._ui.clear_display_selection_state()
        self.clear_batch_display_publication()
        return True

    def _displayed_workspace_preview_set_ids(
        self,
        set_ids: Sequence[str],
        *,
        pre_reset_display_snapshot: Optional[Mapping[str, Any]] = None,
    ) -> list[str]:
        candidates = [str(set_id) for set_id in (set_ids or ()) if str(set_id)]
        if not candidates:
            return []
        if isinstance(pre_reset_display_snapshot, Mapping):
            transaction_snapshot = pre_reset_display_snapshot.get("transaction_snapshot")
            if isinstance(transaction_snapshot, Mapping):
                provenance_by_set_id = transaction_snapshot.get("workspace_preview_display_provenance_by_set_id")
                if isinstance(provenance_by_set_id, Mapping):
                    displayed = {str(set_id) for set_id in provenance_by_set_id if str(set_id)}
                    return [set_id for set_id in candidates if set_id in displayed]
            snapshot_set_id = str(pre_reset_display_snapshot.get("set_id") or "").strip()
            if snapshot_set_id:
                return [set_id for set_id in candidates if set_id == snapshot_set_id]
        return [
            set_id
            for set_id in candidates
            if self.displayed_workspace_preview_provenance_matches_current_workspace(set_id=set_id)
        ]

    def authoritative_result_transition_required(
        self,
        *,
        batch_has_active_display: bool,
        cache_stale_scope_is_global: bool,
        cache_stale_set_ids: Sequence[str],
    ) -> bool:
        return bool(
            batch_has_active_display
            or self.main_plot_has_data()
            or bool(cache_stale_scope_is_global)
            or any(str(set_id) for set_id in (cache_stale_set_ids or ()))
        )

    def apply_authoritative_result_display_transition(
        self,
        *,
        preserve_current_display: Optional[Mapping[str, Any]],
        active_cache_key: str,
        selected_ids: Sequence[str],
        last_display_selection: Sequence[str],
        active_batch_set_id: str,
        active_batch_set_name: str,
        active_cache_invalidated_set_ids: Sequence[str],
        display_clear_set_ids: Sequence[str],
        display_clear_scope_is_global: bool,
    ) -> AuthoritativeResultDisplayTransitionOutcome:
        selected_ids_t = tuple(str(set_id) for set_id in (selected_ids or ()) if str(set_id))
        display_cleared = False
        if active_cache_key and self._display_clear_scope_affects_visible_results(
            display_clear_set_ids=display_clear_set_ids,
            display_clear_scope_is_global=bool(display_clear_scope_is_global),
            selected_ids=selected_ids_t,
            last_display_selection=last_display_selection,
            active_batch_set_id=active_batch_set_id,
        ):
            self._ui.clear_display_selection_state()
            display_cleared = True

        if self._clear_orphaned_visible_results_for_authoritative_result_update(
            preserve_current_display=preserve_current_display,
            active_cache_key=active_cache_key,
            last_display_selection=last_display_selection,
            active_batch_set_id=active_batch_set_id,
            active_batch_set_name=active_batch_set_name,
            display_clear_set_ids=display_clear_set_ids,
            display_clear_scope_is_global=bool(display_clear_scope_is_global),
            selected_ids=selected_ids_t,
        ):
            display_cleared = True

        if self._restore_preserved_authoritative_display(preserve_current_display=preserve_current_display):
            return AuthoritativeResultDisplayTransitionOutcome(refresh_requested=False)

        return self._finish_authoritative_result_display_update(
            active_cache_key=active_cache_key,
            selected_ids=selected_ids_t,
            active_cache_invalidated_set_ids=active_cache_invalidated_set_ids,
            display_cleared=display_cleared,
            last_display_selection=last_display_selection,
            active_batch_set_id=active_batch_set_id,
        )

    @staticmethod
    def _visible_result_display_set_ids(
        *,
        selected_ids: Sequence[str],
        last_display_selection: Sequence[str],
        active_batch_set_id: str,
    ) -> set[str]:
        visible_scope = {str(set_id) for set_id in (selected_ids or ()) if str(set_id)}
        visible_scope.update(str(set_id) for set_id in (last_display_selection or ()) if str(set_id))
        active_batch_set_id_s = str(active_batch_set_id or "").strip()
        if active_batch_set_id_s:
            visible_scope.add(active_batch_set_id_s)
        return visible_scope

    def _display_clear_scope_affects_visible_results(
        self,
        *,
        display_clear_set_ids: Sequence[str],
        display_clear_scope_is_global: bool,
        selected_ids: Sequence[str],
        last_display_selection: Sequence[str],
        active_batch_set_id: str,
    ) -> bool:
        if bool(display_clear_scope_is_global):
            return True
        clear_scope = {str(set_id) for set_id in (display_clear_set_ids or ()) if str(set_id)}
        return bool(
            clear_scope
            & self._visible_result_display_set_ids(
                selected_ids=selected_ids,
                last_display_selection=last_display_selection,
                active_batch_set_id=active_batch_set_id,
            )
        )

    def _clear_orphaned_visible_results_for_authoritative_result_update(
        self,
        *,
        preserve_current_display: Optional[Mapping[str, Any]],
        active_cache_key: str,
        last_display_selection: Sequence[str],
        active_batch_set_id: str,
        active_batch_set_name: str,
        display_clear_set_ids: Sequence[str],
        display_clear_scope_is_global: bool,
        selected_ids: Sequence[str],
    ) -> bool:
        if active_cache_key or preserve_current_display or not self.main_plot_has_data():
            return False
        display_selection = tuple(str(set_id) for set_id in (last_display_selection or ()) if str(set_id))
        has_display_provenance = bool(
            str(active_batch_set_id or "").strip()
            or str(active_batch_set_name or "").strip()
            or display_selection
        )
        if has_display_provenance:
            return False
        if not self._display_clear_scope_affects_visible_results(
            display_clear_set_ids=display_clear_set_ids,
            display_clear_scope_is_global=bool(display_clear_scope_is_global),
            selected_ids=selected_ids,
            last_display_selection=last_display_selection,
            active_batch_set_id=active_batch_set_id,
        ):
            return False
        self.clear_batch_display_publication()
        return True

    def _restore_preserved_authoritative_display(
        self,
        *,
        preserve_current_display: Optional[Mapping[str, Any]],
    ) -> bool:
        if not preserve_current_display or not self.main_plot_has_data():
            return False
        preserved_set_id = str(preserve_current_display.get("set_id") or "").strip()
        preserved_set_name = str(preserve_current_display.get("set_name") or "").strip()
        preserved_selected_ids = [
            str(set_id) for set_id in (preserve_current_display.get("selected_ids") or ()) if str(set_id)
        ]
        transaction_snapshot = preserve_current_display.get("transaction_snapshot")
        if not isinstance(transaction_snapshot, Mapping):
            return False
        preserved_claim_set_ids = preserved_selected_ids or ([preserved_set_id] if preserved_set_id else [])
        outcome = self.publish_preserved_batch_display(
            primary_set_id=preserved_set_id,
            primary_set_name=(preserved_set_name or preserved_set_id),
            selected_set_ids=preserved_claim_set_ids,
            transaction_snapshot=transaction_snapshot,
        )
        if not outcome.displayed:
            return False
        self.reset_stale_cache_warning_status()
        return True

    def _finish_authoritative_result_display_update(
        self,
        *,
        active_cache_key: str,
        selected_ids: Sequence[str],
        active_cache_invalidated_set_ids: Sequence[str],
        display_cleared: bool,
        last_display_selection: Sequence[str],
        active_batch_set_id: str,
    ) -> AuthoritativeResultDisplayTransitionOutcome:
        if active_cache_key and active_cache_invalidated_set_ids and display_cleared:
            self.clear_batch_display_publication()
            self._ui.set_status_text("Result not cached (evicted). Press Run to compute.")
            return AuthoritativeResultDisplayTransitionOutcome(refresh_requested=False)
        if selected_ids and (active_cache_key or active_batch_set_id or last_display_selection):
            return AuthoritativeResultDisplayTransitionOutcome(refresh_requested=True)
        if selected_ids:
            return AuthoritativeResultDisplayTransitionOutcome(refresh_requested=False)
        self.clear_batch_display_publication()
        return AuthoritativeResultDisplayTransitionOutcome(refresh_requested=False)

    @staticmethod
    def _transaction_overlay_is_reference(entry: Mapping[str, Any]) -> bool:
        return str(entry.get("layer_kind") or "").strip() == "reference"

    def preserved_workspace_preview_display_snapshot(
        self,
        *,
        active_set_id: str,
        active_set_name: str,
        selected_set_ids: Sequence[str],
        preview_launch_pending: bool,
        displayed_workspace_preview_matches: Callable[[str], bool],
        active_explicit_entry_matches: Callable[[str, Mapping[str, Any]], bool],
    ) -> Optional[Dict[str, Any]]:
        if bool(preview_launch_pending) or not self._ui.main_plot_has_data():
            return None
        active_set_id_s = str(active_set_id or "").strip()
        if not active_set_id_s:
            return None
        plot = self._main_plot()
        snapshotter = getattr(plot, "display_transaction_snapshot", None) if plot is not None else None
        snapshot = snapshotter() if callable(snapshotter) else None
        if not isinstance(snapshot, Mapping):
            return None
        if not bool(snapshot.get("_kindred_display_transaction_snapshot")):
            return None
        if np.asarray(snapshot.get("t") if snapshot.get("t") is not None else [], dtype=float).reshape(-1).size <= 0:
            return None
        if not dict(snapshot.get("series") or {}):
            return None
        overlay_snapshots = [dict(entry) for entry in (snapshot.get("overlays") or ()) if isinstance(entry, Mapping)]
        active_plot_is_truthful_dirty_preview = bool(displayed_workspace_preview_matches(active_set_id_s))

        has_truthful_preserved_overlay = False
        truthful_preserved_preview_set_ids: set[str] = set()
        for entry in overlay_snapshots:
            overlay_label = str(entry.get("label") or "").strip()
            overlay_set_id = str(entry.get("set_id") or "").strip()
            overlay_is_reference = self._transaction_overlay_is_reference(entry)
            if not overlay_label or not overlay_set_id or overlay_set_id == active_set_id_s:
                continue
            if overlay_is_reference:
                if (
                    not bool(active_explicit_entry_matches(overlay_set_id, entry))
                    or overlay_set_id not in truthful_preserved_preview_set_ids
                ):
                    continue
                has_truthful_preserved_overlay = True
            else:
                overlay_is_truthful_dirty_preview = bool(displayed_workspace_preview_matches(overlay_set_id))
                overlay_matches_explicit = bool(active_explicit_entry_matches(overlay_set_id, entry))
                if not overlay_is_truthful_dirty_preview or overlay_matches_explicit:
                    continue
                truthful_preserved_preview_set_ids.add(overlay_set_id)
                has_truthful_preserved_overlay = True

        if active_plot_is_truthful_dirty_preview:
            for entry in overlay_snapshots:
                if not self._transaction_overlay_is_reference(entry):
                    continue
                if str(entry.get("set_id") or "").strip() != active_set_id_s:
                    continue
                if active_explicit_entry_matches(active_set_id_s, entry):
                    has_truthful_preserved_overlay = True
                    break

        if (not active_plot_is_truthful_dirty_preview) and (not has_truthful_preserved_overlay):
            return None
        return {
            "set_id": active_set_id_s,
            "set_name": str(active_set_name or active_set_id_s),
            "selected_ids": [str(set_id) for set_id in (selected_set_ids or ()) if str(set_id)],
            "transaction_snapshot": dict(snapshot),
        }

    def active_workspace_preview_display_snapshot(self) -> Optional[Dict[str, Any]]:
        active_set_id, active_set_name = self._ui.active_batch_selection()
        active_set_id = str(active_set_id or "").strip()
        if not active_set_id:
            return None
        selected_ids = [str(set_id) for set_id in (self._ui.last_display_selection() or ()) if str(set_id)]
        if not selected_ids:
            selected_ids = [str(set_id) for set_id in (self._ui.shown_batch_set_ids() or ()) if str(set_id)]
        if active_set_id not in selected_ids:
            selected_ids = [active_set_id, *[set_id for set_id in selected_ids if set_id != active_set_id]]
        return self.preserved_workspace_preview_display_snapshot(
            active_set_id=active_set_id,
            active_set_name=str(active_set_name or self._ui.batch_name_for_id(active_set_id) or active_set_id),
            selected_set_ids=selected_ids,
            preview_launch_pending=bool(self._ui.preview_launch_pending()),
            displayed_workspace_preview_matches=(
                lambda set_id: self.displayed_workspace_preview_provenance_matches(
                    set_id=str(set_id),
                    current_payload=self._ui.current_workspace_preview_identity_payload(str(set_id)) or {},
                )
            ),
            active_explicit_entry_matches=(
                lambda set_id, entry: self._ui.active_explicit_entry_matches_displayed_entry(str(set_id), entry)
            ),
        )

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
        plot = self._main_plot()
        snapshotter = getattr(plot, "workspace_preview_display_provenance_snapshot", None) if plot is not None else None
        raw = snapshotter() if callable(snapshotter) else None
        if not isinstance(raw, Mapping):
            return False
        stored_payload = raw.get(sid)
        return isinstance(stored_payload, Mapping) and dict(stored_payload) == dict(current_payload)

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

    @staticmethod
    def _copy_all_shown_block_from_snapshot(
        *,
        set_id: str,
        label: str,
        entry: Mapping[str, Any],
        layer_id: str = "",
    ) -> object | None:
        from kindred.gui.widgets.pyqtgraph_plot_panel_impl import CopyAllShownBlock

        t = np.asarray(entry.get("t") if entry.get("t") is not None else [], dtype=float).reshape(-1)
        series_raw = entry.get("series") or {}
        if t.size <= 0 or not isinstance(series_raw, Mapping):
            return None
        series = {
            str(name): np.asarray(values, dtype=float).reshape(-1)
            for name, values in dict(series_raw).items()
        }
        if not series:
            return None
        return CopyAllShownBlock(
            set_id=str(set_id),
            label=str(label),
            t=t,
            series=series,
            layer_id=str(layer_id or entry.get("layer_id") or ""),
        )

    def _display_transaction_snapshot(self) -> Optional[Dict[str, object]]:
        plot = self._main_plot()
        snapshotter = getattr(plot, "display_transaction_snapshot", None) if plot is not None else None
        if not callable(snapshotter):
            return None
        snapshot = snapshotter()
        return dict(snapshot) if isinstance(snapshot, Mapping) else None

    def _copy_all_popup_labels_by_set_id(self, set_ids: Sequence[str]) -> Dict[str, str]:
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
                row = self._ui.batch_row_for_set_id(set_id)
                if row is not None:
                    popup_label = f"{label} (row {int(row) + 1})"
            popup_labels[set_id] = popup_label
        return popup_labels

    def build_main_plot_copy_all_export_plan(self) -> object:
        from kindred.gui.widgets.pyqtgraph_plot_panel_impl import CopyAllExportPlan, CopyAllMissingItem

        snapshot = self._display_transaction_snapshot()
        transaction_set_ids: list[str] = []
        if snapshot is not None:
            primary_snapshot_set_id = str(snapshot.get("primary_set_id") or "").strip()
            if primary_snapshot_set_id:
                transaction_set_ids.append(primary_snapshot_set_id)
            for raw_entry in list(snapshot.get("overlays") or []):
                if not isinstance(raw_entry, Mapping) or self._transaction_overlay_is_reference(raw_entry):
                    continue
                sid = str(raw_entry.get("set_id") or "").strip()
                if sid and sid not in transaction_set_ids:
                    transaction_set_ids.append(sid)
        shown_set_ids = (
            transaction_set_ids
            if transaction_set_ids
            else [str(set_id) for set_id in (self._ui.shown_batch_set_ids() or []) if str(set_id)]
        )
        non_batch_live_primary_label = ""
        if not shown_set_ids and self.main_plot_has_data():
            active_set_id = str(self._ui.active_batch_selection()[0] or "").strip()
            if active_set_id:
                shown_set_ids = [active_set_id]
            else:
                non_batch_live_primary_label = "Results"
        popup_labels = self._copy_all_popup_labels_by_set_id(shown_set_ids)
        shown_set_id_set = {str(set_id) for set_id in shown_set_ids if str(set_id)}
        show_reference_layers = True
        snapshot = self._display_transaction_snapshot()
        if snapshot is not None:
            show_reference_layers = bool(snapshot.get("show_reference_layers", True))

        shown_blocks: list[object] = []
        missing_items = []
        displayed_result_ids: set[str] = set()

        if snapshot is not None:
            primary_set_id = str(snapshot.get("primary_set_id") or "").strip()
            primary_effective_set_id = primary_set_id
            if not primary_effective_set_id and len(shown_set_ids) == 1:
                primary_effective_set_id = str(shown_set_ids[0])
            include_primary = (
                bool(non_batch_live_primary_label)
                or not shown_set_id_set
                or (primary_effective_set_id in shown_set_id_set)
            )
            if include_primary:
                primary_label = str(
                    popup_labels.get(primary_effective_set_id)
                    or snapshot.get("popup_label")
                    or snapshot.get("label")
                    or non_batch_live_primary_label
                    or primary_effective_set_id
                    or "Results"
                )
                block = self._copy_all_shown_block_from_snapshot(
                    set_id=primary_effective_set_id,
                    label=primary_label,
                    entry=snapshot,
                    layer_id=str(snapshot.get("primary_layer_id") or f"result:{primary_effective_set_id or 'live'}"),
                )
                if block is not None:
                    shown_blocks.append(block)
                    if primary_effective_set_id:
                        displayed_result_ids.add(primary_effective_set_id)

            for raw_entry in list(snapshot.get("overlays") or []):
                if not isinstance(raw_entry, Mapping):
                    continue
                entry = dict(raw_entry)
                sid = str(entry.get("set_id") or "").strip()
                if not sid:
                    continue
                if self._transaction_overlay_is_reference(entry):
                    if not show_reference_layers:
                        continue
                    if shown_set_id_set and sid not in shown_set_id_set and sid not in displayed_result_ids:
                        continue
                    fallback_label = str(popup_labels.get(sid) or self._ui.batch_name_for_id(sid) or sid)
                    ref_label = str(entry.get("popup_label") or entry.get("label") or f"{fallback_label} [ref]")
                    if "[ref]" not in ref_label:
                        ref_label = f"{fallback_label} [ref]"
                    block = self._copy_all_shown_block_from_snapshot(
                        set_id=f"{sid}:canonical_reference",
                        label=ref_label,
                        entry=entry,
                        layer_id=str(entry.get("layer_id") or f"reference:{sid}"),
                    )
                    if block is not None:
                        shown_blocks.append(block)
                    continue
                if shown_set_id_set and sid not in shown_set_id_set:
                    continue
                label = str(entry.get("popup_label") or entry.get("label") or popup_labels.get(sid) or sid)
                block = self._copy_all_shown_block_from_snapshot(
                    set_id=sid,
                    label=label,
                    entry=entry,
                    layer_id=str(entry.get("layer_id") or f"result:{sid}"),
                )
                if block is not None:
                    shown_blocks.append(block)
                    displayed_result_ids.add(sid)

        for set_id in shown_set_ids:
            if str(set_id) in displayed_result_ids:
                continue
            label = str(self._ui.batch_name_for_id(set_id) or set_id)
            missing_items.append(
                CopyAllMissingItem(
                    set_id=str(set_id),
                    label=label,
                    popup_label=str(popup_labels.get(str(set_id), label)),
                    reason="no_simulation_data" if snapshot is None else "no_cached_results",
                )
            )
        if non_batch_live_primary_label:
            has_live_block = any(str(getattr(block, "set_id", "")) == "" for block in shown_blocks)
            if not has_live_block:
                missing_items.append(
                    CopyAllMissingItem(
                        set_id="",
                        label=non_batch_live_primary_label,
                        popup_label=non_batch_live_primary_label,
                        reason="no_simulation_data",
                    )
                )
        return CopyAllExportPlan(shown_blocks=shown_blocks, missing_items=missing_items)

    def reset_stale_cache_warning_status(self) -> None:
        stale_messages = {
            "Result not cached (evicted). Press Run to compute.",
            "Cached result invalid. Press Run to compute.",
            "Preview pending for current selection.",
        }
        try:
            current_status = str(self._ui.status_text_getter() or "")
        except Exception as exc:
            logger.debug("Failed to inspect current status text: %s", exc, exc_info=True)
            current_status = ""
        if current_status in stale_messages:
            self._ui.set_status_text("Ready")

    def _clear_non_displayed_batch_selection(
        self,
        *,
        clear_plot: bool = True,
    ) -> None:
        self._ui.clear_display_selection_state()
        if bool(clear_plot):
            self.clear_batch_display_publication()

    def _apply_batch_display_transaction(
        self,
        *,
        t: np.ndarray,
        series: Mapping[str, Any],
        label: str,
        overlays: Sequence[Dict[str, object]],
        metadata_applier: Callable[[object], Optional[str]],
        annotation_entry: Mapping[str, Any],
        primary_set_id: str,
        primary_label: str,
        selected_set_ids: Sequence[str],
        workspace_preview_provenance_by_set_id: Mapping[str, Mapping[str, Any]] | None = None,
        completion_provenance: Mapping[str, Any] | None = None,
    ) -> CachedBatchSelectionDisplayOutcome:
        rollback_snapshot = self._direct_display_failure_snapshot(self._main_plot())
        primary_set_id_s = str(primary_set_id or "").strip()
        if not self._set_plot_data(
            np.asarray(t, dtype=float),
            {str(k): np.asarray(v, dtype=float) for k, v in series.items()},
            label=str(label),
            primary_set_id=primary_set_id_s,
            layer_id=f"result:{primary_set_id_s}" if primary_set_id_s else None,
            overlays=overlays,
        ):
            self._restore_direct_display_failure_snapshot(rollback_snapshot)
            return CachedBatchSelectionDisplayOutcome(False, reason="display_failed")

        plot = self._main_plot()
        if plot is None:
            self._restore_direct_display_failure_snapshot(rollback_snapshot)
            return CachedBatchSelectionDisplayOutcome(False, reason="display_failed")
        metadata_failure = metadata_applier(plot)
        if metadata_failure:
            self._restore_direct_display_failure_snapshot(rollback_snapshot)
            return CachedBatchSelectionDisplayOutcome(False, reason=metadata_failure)
        try:
            self._apply_intervention_annotations(plot=plot, entry=annotation_entry)
        except Exception as exc:
            logger.exception("Failed to apply batch intervention annotations: %s", exc)
            self._restore_direct_display_failure_snapshot(rollback_snapshot)
            return CachedBatchSelectionDisplayOutcome(False, reason="annotation_failed")

        provenance_snapshot = self._direct_completion_provenance_snapshot()
        provenance_failure = self._publish_direct_completion_provenance(
            plot=plot,
            direct_completion_provenance=completion_provenance,
        )
        if provenance_failure:
            self._restore_direct_completion_provenance_snapshot(provenance_snapshot)
            self._restore_direct_display_failure_snapshot(rollback_snapshot)
            return CachedBatchSelectionDisplayOutcome(False, reason=provenance_failure)

        displayed_ids = tuple(str(set_id) for set_id in selected_set_ids if str(set_id))
        commit_failure = self._commit_display_claim_state(
            primary_set_id=str(primary_set_id),
            primary_label=str(primary_label),
            selected_set_ids=displayed_ids,
            workspace_preview_provenance_by_set_id=workspace_preview_provenance_by_set_id,
        )
        if commit_failure:
            self._restore_direct_completion_provenance_snapshot(provenance_snapshot)
            self._restore_direct_display_failure_snapshot(rollback_snapshot)
            return CachedBatchSelectionDisplayOutcome(False, reason=commit_failure)
        self._commit_successful_plot_display(t=np.asarray(t, dtype=float), series=series)
        return CachedBatchSelectionDisplayOutcome(
            True,
            primary_set_id=str(primary_set_id),
            displayed_set_ids=displayed_ids,
        )

    @staticmethod
    def _display_reason_status_text(reason: Optional[str]) -> str:
        if reason == "invalid_cache_entry":
            return "Cached result invalid. Press Run to compute."
        if reason == "preview_pending":
            return "Preview pending for current selection."
        return "Result not cached (evicted). Press Run to compute."

    def _display_cached_active_selection(
        self,
        *,
        request: BatchDisplayRefreshRequest,
    ) -> CachedBatchSelectionDisplayOutcome:
        return self.publish_cached_batch_selection(
            cache_key=str(request.active_cache_key or ""),
            selected_sets=request.shown_set_ids,
            prefer_set=request.prefer_set_id,
            valid_set_ids=request.active_cache_valid_set_ids,
            invalidated_set_ids=request.active_cache_invalidated_set_ids,
        )

    @staticmethod
    def _refresh_success(
        *,
        display_outcome: CachedBatchSelectionDisplayOutcome,
        focused_controls_use_workspace: bool,
    ) -> BatchDisplayRefreshOutcome:
        return BatchDisplayRefreshOutcome(
            focused_controls_use_workspace=bool(focused_controls_use_workspace),
            displayed=True,
            primary_set_id=display_outcome.primary_set_id,
            displayed_set_ids=display_outcome.displayed_set_ids,
        )

    @staticmethod
    def _refresh_failed(
        *,
        reason: Optional[str],
        focused_controls_use_workspace: bool,
    ) -> BatchDisplayRefreshOutcome:
        return BatchDisplayRefreshOutcome(
            focused_controls_use_workspace=bool(focused_controls_use_workspace),
            reason=reason,
        )

    def _publish_fresh_explicit_dirty_refresh(
        self,
        *,
        request: BatchDisplayRefreshRequest,
    ) -> Optional[BatchDisplayRefreshOutcome]:
        if not (request.focused_dirty and request.fresh_explicit_cache_after_post_run_sync):
            return None
        outcome = self._display_cached_active_selection(request=request)
        if outcome.displayed:
            self.reset_stale_cache_warning_status()
            return self._refresh_success(display_outcome=outcome, focused_controls_use_workspace=True)
        self._ui.set_status_text(self._display_reason_status_text(outcome.reason))
        return self._refresh_failed(reason=outcome.reason, focused_controls_use_workspace=False)

    def _publish_fully_resolved_refresh(
        self,
        *,
        request: BatchDisplayRefreshRequest,
        resolution: BatchDisplaySelectionResolution,
    ) -> tuple[Optional[BatchDisplayRefreshOutcome], BatchDisplaySelectionResolution]:
        if not (resolution.all_selected_sets_resolved and resolution.resolved_entries):
            return None, resolution
        outcome = self.publish_resolved_batch_selection(
            resolved_entries=resolution.resolved_entries,
            prefer_set=request.prefer_set_id,
        )
        if outcome.displayed:
            self._apply_successful_resolved_refresh_status(resolution=resolution)
            return (
                self._refresh_success(
                    display_outcome=outcome,
                    focused_controls_use_workspace=bool(resolution.focused_uses_workspace_controls),
                ),
                resolution,
            )
        return (
            None,
            BatchDisplaySelectionResolution(
                resolved_entries=resolution.resolved_entries,
                reason=outcome.reason,
                all_selected_sets_resolved=resolution.all_selected_sets_resolved,
                has_workspace_selection=resolution.has_workspace_selection,
                has_resolved_workspace_preview=resolution.has_resolved_workspace_preview,
                focused_uses_workspace_controls=resolution.focused_uses_workspace_controls,
                focused_has_resolved_entry=resolution.focused_has_resolved_entry,
            ),
        )

    def _apply_successful_resolved_refresh_status(self, *, resolution: BatchDisplaySelectionResolution) -> None:
        if resolution.reason == "preview_pending":
            self._ui.set_status_text("Preview pending for current selection.")
            return
        if resolution.reason == "no_cached_results" and not bool(resolution.focused_uses_workspace_controls):
            self._ui.set_status_text("Result not cached (evicted). Press Run to compute.")
            return
        self.reset_stale_cache_warning_status()

    @staticmethod
    def _can_publish_resolved_preview(
        *,
        request: BatchDisplayRefreshRequest,
        resolution: BatchDisplaySelectionResolution,
    ) -> bool:
        return bool(
            resolution.resolved_entries
            and resolution.has_workspace_selection
            and resolution.reason in {"preview_pending", "no_cached_results"}
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
        resolution: BatchDisplaySelectionResolution,
    ) -> tuple[Optional[BatchDisplayRefreshOutcome], BatchDisplaySelectionResolution]:
        if not self._can_publish_resolved_preview(request=request, resolution=resolution):
            return None, resolution
        outcome = self.publish_resolved_batch_selection(
            resolved_entries=resolution.resolved_entries,
            prefer_set=request.prefer_set_id,
        )
        if outcome.displayed:
            self._apply_successful_resolved_refresh_status(resolution=resolution)
            return (
                self._refresh_success(
                    display_outcome=outcome,
                    focused_controls_use_workspace=bool(resolution.focused_uses_workspace_controls),
                ),
                resolution,
            )
        if resolution.reason == "preview_pending":
            self._ui.set_status_text("Preview pending for current selection.")
            self._clear_non_displayed_batch_selection()
            return (
                self._refresh_failed(reason="preview_pending", focused_controls_use_workspace=request.focused_dirty),
                resolution,
            )
        return (
            None,
            BatchDisplaySelectionResolution(
                resolved_entries=resolution.resolved_entries,
                reason=outcome.reason or resolution.reason,
                all_selected_sets_resolved=resolution.all_selected_sets_resolved,
                has_workspace_selection=resolution.has_workspace_selection,
                has_resolved_workspace_preview=resolution.has_resolved_workspace_preview,
                focused_uses_workspace_controls=resolution.focused_uses_workspace_controls,
                focused_has_resolved_entry=resolution.focused_has_resolved_entry,
            ),
        )

    def _publish_active_explicit_refresh(
        self,
        *,
        request: BatchDisplayRefreshRequest,
        resolution: BatchDisplaySelectionResolution,
    ) -> tuple[Optional[BatchDisplayRefreshOutcome], BatchDisplaySelectionResolution]:
        can_try_active_cache = (
            (not resolution.has_workspace_selection)
            and resolution.reason in {"preview_pending", "no_cached_results"}
            and bool(request.active_cache_key)
        )
        if not can_try_active_cache:
            return None, resolution
        outcome = self._display_cached_active_selection(request=request)
        if outcome.displayed:
            self.reset_stale_cache_warning_status()
            return self._refresh_success(display_outcome=outcome, focused_controls_use_workspace=False), resolution
        if outcome.reason != "invalid_cache_entry":
            return None, resolution
        return (
            None,
            BatchDisplaySelectionResolution(
                resolved_entries=resolution.resolved_entries,
                reason="invalid_cache_entry",
                all_selected_sets_resolved=resolution.all_selected_sets_resolved,
                has_workspace_selection=resolution.has_workspace_selection,
                has_resolved_workspace_preview=resolution.has_resolved_workspace_preview,
                focused_uses_workspace_controls=resolution.focused_uses_workspace_controls,
                focused_has_resolved_entry=resolution.focused_has_resolved_entry,
            ),
        )

    def _clear_unpublished_refresh(
        self,
        *,
        request: BatchDisplayRefreshRequest,
        resolution: BatchDisplaySelectionResolution,
        shown_sets: Sequence[str],
    ) -> BatchDisplayRefreshOutcome:
        if (not resolution.has_workspace_selection) and (not request.active_cache_key):
            self._clear_non_displayed_batch_selection()
            self.reset_stale_cache_warning_status()
            return self._refresh_failed(reason=resolution.reason, focused_controls_use_workspace=False)

        if resolution.reason == "invalid_cache_entry":
            self._ui.set_status_text("Cached result invalid. Press Run to compute.")
            self._clear_non_displayed_batch_selection()
            return self._refresh_failed(reason="invalid_cache_entry", focused_controls_use_workspace=request.focused_dirty)
        if resolution.reason == "preview_pending":
            self._ui.set_status_text("Preview pending for current selection.")
            self._clear_non_displayed_batch_selection()
            return self._refresh_failed(reason="preview_pending", focused_controls_use_workspace=request.focused_dirty)

        self._ui.set_status_text("Result not cached (evicted). Press Run to compute.")
        if not request.active_cache_key:
            self._clear_non_displayed_batch_selection()
        return self._refresh_failed(
            reason=str(resolution.reason or "no_cached_results"),
            focused_controls_use_workspace=bool(request.focused_dirty),
        )

    def refresh_batch_selection_display(
        self,
        request: BatchDisplayRefreshRequest,
    ) -> BatchDisplayRefreshOutcome:
        shown_sets = tuple(str(set_id) for set_id in (request.shown_set_ids or ()) if str(set_id))
        if not shown_sets:
            self._clear_non_displayed_batch_selection()
            return BatchDisplayRefreshOutcome(
                focused_controls_use_workspace=bool(request.focused_set_dirty),
                reason="no_selection",
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

        return self._clear_unpublished_refresh(request=request, resolution=resolution, shown_sets=shown_sets)

    def refresh_display_from_focus_and_shown(self) -> BatchDisplayRefreshOutcome:
        self._ui.update_batch_row_controls_state()
        shown_sets = tuple(str(set_id) for set_id in (self._ui.shown_batch_set_ids() or ()) if str(set_id))
        prefer = self._ui.focused_batch_set_id()
        focused_dirty = bool(self._ui.focused_batch_selection_is_dirty(shown_sets, prefer))
        valid_set_ids_raw = self._ui.active_batch_valid_set_ids()
        invalidated_set_ids_raw = self._ui.active_batch_invalidated_set_ids()
        request = BatchDisplayRefreshRequest(
            shown_set_ids=shown_sets,
            prefer_set_id=prefer,
            active_cache_key=str(self._ui.active_batch_cache_key() or ""),
            focused_dirty=focused_dirty,
            focused_set_dirty=bool(self._ui.focused_batch_set_is_dirty()),
            fresh_explicit_cache_after_post_run_sync=(
                self._ui.selection_uses_fresh_explicit_cache_after_post_run_sync(shown_sets)
                if shown_sets
                else False
            ),
            active_cache_valid_set_ids=(
                tuple(str(set_id) for set_id in valid_set_ids_raw if str(set_id))
                if valid_set_ids_raw is not None
                else None
            ),
            active_cache_invalidated_set_ids=(
                tuple(str(set_id) for set_id in invalidated_set_ids_raw if str(set_id))
                if invalidated_set_ids_raw is not None
                else None
            ),
            resolution=(
                self._ui.workspace_selection_resolution(shown_sets)
                if shown_sets
                else BatchDisplaySelectionResolution()
            ),
        )
        outcome = self.refresh_batch_selection_display(request)
        return outcome

    def main_plot_has_data(self) -> bool:
        return bool(self._ui.main_plot_has_data())

    def _commit_display_claim_state(
        self,
        *,
        primary_set_id: str,
        primary_label: str,
        selected_set_ids: Sequence[str],
        workspace_preview_provenance_by_set_id: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> Optional[str]:
        snapshot = self._plot_display_claim_snapshot()
        workspace_provenance = {
            str(set_id): dict(payload)
            for set_id, payload in dict(workspace_preview_provenance_by_set_id or {}).items()
            if str(set_id) and isinstance(payload, Mapping)
        }
        try:
            if str(primary_set_id or "").strip():
                self._ui.set_active_batch_selection(str(primary_set_id), str(primary_label), list(selected_set_ids))
                self._sync_main_plot_copy_labels(
                    primary_set_id=str(primary_set_id),
                    selected_set_ids=list(selected_set_ids),
                )
            else:
                self._ui.clear_display_selection_state()
                self._sync_main_plot_copy_labels(primary_set_id="", selected_set_ids=[])
            if workspace_provenance:
                plot = self._main_plot()
                setter = getattr(plot, "set_workspace_preview_display_provenance", None) if plot is not None else None
                if callable(setter):
                    setter(workspace_provenance)
        except Exception as exc:
            logger.exception("Failed to commit display claim state: %s", exc)
            self._restore_plot_display_claim_snapshot(snapshot)
            return "display_claim_commit_failed"
        return None

    def cached_batch_selection_availability(
        self,
        *,
        cache_key: str,
        selected_sets: Sequence[str],
        cache_store: Optional[MutableMapping[str, Dict[str, Any]]] = None,
        valid_set_ids: Optional[Sequence[str]] = None,
        invalidated_set_ids: Optional[Sequence[str]] = None,
    ) -> CachedBatchAvailability:
        cache_key = str(cache_key or "")
        if not cache_key:
            return CachedBatchAvailability([])
        store = cache_store if cache_store is not None else self._ui.result_cache_store()
        return self._available_cached_batch_ids(
            store=store,
            cache_key=cache_key,
            selected_sets=selected_sets,
            valid_set_ids=valid_set_ids,
            invalidated_set_ids=invalidated_set_ids,
            require_completion_provenance=True,
        )

    def cached_batch_selection_coverage(
        self,
        *,
        cache_key: str,
        selected_sets: Sequence[str],
        cache_store: Optional[MutableMapping[str, Dict[str, Any]]] = None,
        valid_set_ids: Optional[Sequence[str]] = None,
        invalidated_set_ids: Optional[Sequence[str]] = None,
    ) -> CachedBatchSelectionCoverage:
        selected_ids = self._normalized_selected_batch_ids(selected_sets)
        availability = self.cached_batch_selection_availability(
            cache_key=cache_key,
            selected_sets=selected_sets,
            cache_store=cache_store,
            valid_set_ids=valid_set_ids,
            invalidated_set_ids=invalidated_set_ids,
        )
        if not selected_ids:
            reason = "invalid_cache_entry" if availability.has_invalid_entry else "no_cached_results"
            return CachedBatchSelectionCoverage(
                selected_ids=[],
                available_ids=list(availability.available_ids),
                full_coverage=False,
                reason=reason,
            )
        full_coverage = (
            len(availability.available_ids) == len(selected_ids)
            and set(availability.available_ids) == set(selected_ids)
        )
        reason = None
        if not full_coverage:
            reason = "invalid_cache_entry" if availability.has_invalid_entry else "no_cached_results"
        return CachedBatchSelectionCoverage(
            selected_ids=list(selected_ids),
            available_ids=list(availability.available_ids),
            full_coverage=bool(full_coverage),
            reason=reason,
        )

    def publish_cached_batch_selection(
        self,
        *,
        cache_key: str,
        selected_sets: Sequence[str],
        prefer_set: Optional[str] = None,
        cache_store: Optional[MutableMapping[str, Dict[str, Any]]] = None,
        valid_set_ids: Optional[Sequence[str]] = None,
        invalidated_set_ids: Optional[Sequence[str]] = None,
    ) -> CachedBatchSelectionDisplayOutcome:
        cache_key = str(cache_key or "")
        if not cache_key:
            return CachedBatchSelectionDisplayOutcome(False, reason="cache_key_empty")
        store = cache_store if cache_store is not None else self._ui.result_cache_store()
        coverage = self.cached_batch_selection_coverage(
            cache_key=cache_key,
            selected_sets=selected_sets,
            cache_store=store,
            valid_set_ids=valid_set_ids,
            invalidated_set_ids=invalidated_set_ids,
        )
        if not coverage.full_coverage:
            return CachedBatchSelectionDisplayOutcome(False, reason=str(coverage.reason or "no_cached_results"))

        primary = self._primary_cached_batch_id(
            store=store,
            cache_key=cache_key,
            available=coverage.available_ids,
            prefer_set=prefer_set,
        )

        entry_result = self._cache_entry_for_set_id(
            store=store,
            cache_key=cache_key,
            set_id=primary,
            require_completion_provenance=True,
        )
        entry = entry_result.entry
        if entry is None:
            reason = "invalid_cache_entry" if entry_result.state == "invalid" else "no_cached_results"
            return CachedBatchSelectionDisplayOutcome(False, reason=reason)
        t = entry["t"]
        series = entry["series"]

        primary_label = self._ui.batch_name_for_id(primary) or str(primary)
        displayed_entries_by_set_id = self._cached_batch_display_entries_by_set_id(
                store=store,
                cache_key=cache_key,
                available=coverage.available_ids,
                primary=primary,
                primary_entry=entry,
            )
        overlays = self._cached_batch_overlays(
            displayed_entries_by_set_id=displayed_entries_by_set_id,
            primary=primary,
        )
        displayed_ids = tuple(str(set_id) for set_id in coverage.available_ids if str(set_id) in displayed_entries_by_set_id)

        return self._apply_batch_display_transaction(
            t=np.asarray(t, dtype=float),
            series=series,
            label=str(primary_label),
            overlays=overlays,
            metadata_applier=lambda plot: self._apply_cached_batch_plot_metadata(
                plot=plot,
                store=store,
                cache_key=cache_key,
                available=coverage.available_ids,
                displayed_entries_by_set_id=displayed_entries_by_set_id,
                primary=primary,
                primary_label=str(primary_label),
                entry=entry,
                t=np.asarray(t, dtype=float),
                series={str(k): np.asarray(v, dtype=float) for k, v in series.items()},
            ),
            annotation_entry=entry,
            primary_set_id=str(primary),
            primary_label=str(primary_label),
            selected_set_ids=list(displayed_ids),
            completion_provenance=entry.get("completion_provenance") if isinstance(entry, Mapping) else None,
        )

    def publish_resolved_batch_selection(
        self,
        *,
        resolved_entries: Sequence[ResolvedBatchSelectionEntry],
        prefer_set: Optional[str] = None,
    ) -> CachedBatchSelectionDisplayOutcome:
        if not resolved_entries:
            return CachedBatchSelectionDisplayOutcome(False, reason="no_cached_results")

        entries_by_id = {
            str(resolved.set_id): resolved
            for resolved in resolved_entries
            if str(resolved.set_id)
        }
        if not entries_by_id:
            return CachedBatchSelectionDisplayOutcome(False, reason="no_cached_results")
        for resolved in entries_by_id.values():
            has_workspace_provenance = isinstance(resolved.workspace_preview_provenance, Mapping)
            has_completion_provenance = isinstance(resolved.entry.get("completion_provenance"), Mapping)
            if not (has_workspace_provenance or has_completion_provenance):
                return CachedBatchSelectionDisplayOutcome(False, reason="invalid_cache_entry")
            if resolved.canonical_entry is not None and not isinstance(
                resolved.canonical_entry.get("completion_provenance"),
                Mapping,
            ):
                return CachedBatchSelectionDisplayOutcome(False, reason="invalid_cache_entry")

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
            dict(
                build_overlay_entry(
                    label=resolved.label,
                    entry=resolved.entry,
                    set_id=resolved.set_id,
                    layer_kind="result",
                    layer_id=f"result:{resolved.set_id}",
                )
            )
            for resolved in resolved_entries
            if str(resolved.set_id) != str(primary.set_id)
        ]
        for resolved in resolved_entries:
            if resolved.canonical_entry is None:
                continue
            overlays.append(
                dict(
                    build_overlay_entry(
                        label=resolved.label,
                        entry=resolved.canonical_entry,
                        set_id=resolved.set_id,
                        layer_kind="reference",
                        layer_id=f"reference:{resolved.set_id}",
                    )
                )
            )
        displayed_ids = tuple(str(resolved.set_id) for resolved in resolved_entries if str(resolved.set_id))
        workspace_provenance_by_set_id = {
            str(resolved.set_id): dict(resolved.workspace_preview_provenance)
            for resolved in resolved_entries
            if str(resolved.set_id) and isinstance(resolved.workspace_preview_provenance, Mapping)
        }
        return self._apply_batch_display_transaction(
            t=np.asarray(primary.entry["t"], dtype=float),
            series=primary.entry.get("series") or {},
            label=str(primary.label),
            overlays=overlays,
            metadata_applier=lambda plot: self._apply_resolved_batch_plot_metadata(
                plot=plot,
                resolved_entries=list(resolved_entries),
                primary=primary,
            ),
            annotation_entry=primary.entry,
            primary_set_id=str(primary.set_id),
            primary_label=str(primary.label),
            selected_set_ids=displayed_ids,
            workspace_preview_provenance_by_set_id=workspace_provenance_by_set_id,
            completion_provenance=(
                primary.entry.get("completion_provenance")
                if isinstance(primary.entry.get("completion_provenance"), Mapping)
                else None
            ),
        )

    def publish_completed_run_display_transaction(
        self,
        transaction: CompletedRunDisplayTransaction,
    ) -> SimulationCompletionDisplayOutcome:
        intent = transaction.intent
        completion_entries = tuple(transaction.completion_entries or ())
        if not completion_entries:
            return SimulationCompletionDisplayOutcome(
                False,
                direct_completion_displayed=False,
                reason="in_flight_completion_coverage_unavailable",
            )
        expected_ids = tuple(str(set_id) for set_id in intent.set_ids if str(set_id))
        completed_ids = tuple(str(entry.set_id) for entry in completion_entries if str(entry.set_id))
        if expected_ids != completed_ids:
            return SimulationCompletionDisplayOutcome(
                False,
                direct_completion_displayed=False,
                reason="in_flight_completion_coverage_unavailable",
            )
        for completion_entry in completion_entries:
            if not isinstance(completion_entry.completion_provenance, Mapping):
                return SimulationCompletionDisplayOutcome(
                    False,
                    direct_completion_displayed=False,
                    reason="invalid_completion_display_entry",
                )
            if completion_entry.t is None or not isinstance(completion_entry.series, Mapping):
                return SimulationCompletionDisplayOutcome(
                    False,
                    direct_completion_displayed=False,
                    reason="invalid_completion_display_entry",
                )

        entries_by_id = {str(entry.set_id): entry for entry in completion_entries}
        primary_id = str(intent.primary_set_id or "").strip()
        primary = entries_by_id.get(primary_id)
        if primary is None:
            return SimulationCompletionDisplayOutcome(
                False,
                direct_completion_displayed=False,
                reason="invalid_completed_run_display_intent",
            )

        overlays = [
            dict(
                build_overlay_entry(
                    label=completion_entry.label,
                    entry=completion_entry.to_display_payload(),
                    set_id=completion_entry.set_id,
                    layer_kind="result",
                    layer_id=f"result:{completion_entry.set_id}",
                )
            )
            for completion_entry in completion_entries
            if str(completion_entry.set_id) != str(primary.set_id)
        ]
        primary_payload = primary.to_display_payload()
        outcome = self._apply_batch_display_transaction(
            t=np.asarray(primary.t, dtype=float),
            series=primary.series or {},
            label=str(primary.label),
            overlays=overlays,
            metadata_applier=lambda plot: self._apply_completed_run_plot_metadata(
                plot=plot,
                completion_entries=list(completion_entries),
                primary=primary,
            ),
            annotation_entry=primary_payload,
            primary_set_id=str(primary.set_id),
            primary_label=str(primary.label),
            selected_set_ids=expected_ids,
            completion_provenance=primary.completion_provenance,
        )
        return SimulationCompletionDisplayOutcome(
            bool(outcome.displayed),
            direct_completion_displayed=False,
            reason=outcome.reason,
            primary_set_id=outcome.primary_set_id,
            displayed_set_ids=outcome.displayed_set_ids,
        )

    @staticmethod
    def _fresh_completion_cache_miss_can_publish_directly(
        *,
        reason: Optional[str],
        batch_set_id: str | None,
        selected_sets: Sequence[str],
        direct_completion_provenance: Mapping[str, Any] | None,
    ) -> bool:
        if str(reason or "") not in {"no_cached_results", "invalid_cache_entry"}:
            return False
        set_id = str(batch_set_id or "").strip()
        if not set_id or not isinstance(direct_completion_provenance, Mapping):
            return False
        selected_ids = [str(raw_set_id or "").strip() for raw_set_id in selected_sets or ()]
        selected_ids = [raw_set_id for raw_set_id in selected_ids if raw_set_id]
        return selected_ids == [set_id]

    def publish_simulation_completion_result(
        self,
        *,
        t: np.ndarray,
        series: Dict[str, np.ndarray],
        cache_key: str | None,
        batch_set: str | None,
        batch_set_id: str | None,
        redraw_valid_set_ids: Sequence[str] | None,
        has_redraw_subset: bool,
        slider_triggered: bool,
        explicit_batch_coalescing: bool,
        algebra_scalars: Mapping[str, object] | None,
        solver_provenance: Mapping[str, Any] | None = None,
        direct_completion_provenance: Mapping[str, Any] | None = None,
        owned_species: Sequence[str] | None = None,
    ) -> SimulationCompletionDisplayOutcome:
        displayed = False
        cache_display_outcome = CachedBatchSelectionDisplayOutcome(False)
        normalized_cache_key = str(cache_key or "").strip()
        selected_sets: Sequence[str] = ()
        prefer_set = None
        if normalized_cache_key:
            selected_sets = self._ui.selected_batch_set_ids()
            current_row = self._ui.current_batch_row()
            if current_row is not None:
                prefer_set = self._ui.batch_set_id_for_row(int(current_row))
            cache_display_outcome = self.publish_cached_batch_selection(
                cache_key=normalized_cache_key,
                selected_sets=selected_sets,
                prefer_set=prefer_set,
                cache_store=None,
                valid_set_ids=(redraw_valid_set_ids if bool(has_redraw_subset) else None),
            )
            displayed = bool(cache_display_outcome.displayed)
        if displayed:
            return SimulationCompletionDisplayOutcome(
                True,
                direct_completion_displayed=False,
                primary_set_id=cache_display_outcome.primary_set_id,
                displayed_set_ids=cache_display_outcome.displayed_set_ids,
            )
        if normalized_cache_key:
            if self._fresh_completion_cache_miss_can_publish_directly(
                reason=cache_display_outcome.reason,
                batch_set_id=batch_set_id,
                selected_sets=selected_sets,
                direct_completion_provenance=direct_completion_provenance,
            ):
                return self._publish_direct_completion_result(
                    t=t,
                    series=series,
                    batch_set=batch_set,
                    batch_set_id=batch_set_id,
                    algebra_scalars=algebra_scalars,
                    solver_provenance=solver_provenance,
                    direct_completion_provenance=direct_completion_provenance,
                    owned_species=owned_species,
                )
            return SimulationCompletionDisplayOutcome(
                False,
                direct_completion_displayed=False,
                reason=str(cache_display_outcome.reason or "cached_display_failed"),
            )

        return self._publish_direct_completion_result(
            t=t,
            series=series,
            batch_set=batch_set,
            batch_set_id=batch_set_id,
            algebra_scalars=algebra_scalars,
            solver_provenance=solver_provenance,
            direct_completion_provenance=direct_completion_provenance,
            owned_species=owned_species,
        )

    def _publish_direct_completion_result(
        self,
        *,
        t: np.ndarray,
        series: Dict[str, np.ndarray],
        batch_set: str | None,
        batch_set_id: str | None,
        algebra_scalars: Mapping[str, object] | None,
        solver_provenance: Mapping[str, Any] | None,
        direct_completion_provenance: Mapping[str, Any] | None,
        owned_species: Sequence[str] | None,
    ) -> SimulationCompletionDisplayOutcome:
        set_id = str(batch_set_id or "").strip()
        set_name = str(batch_set or "").strip()
        rollback_snapshot = self._direct_display_failure_snapshot(self._main_plot())

        def _fail(reason: str) -> SimulationCompletionDisplayOutcome:
            self._restore_direct_display_failure_snapshot(rollback_snapshot)
            return SimulationCompletionDisplayOutcome(False, direct_completion_displayed=False, reason=reason)

        primary_layer_id = f"result:{set_id}" if set_id else "result:live"
        if not self._set_plot_data(
            t,
            series,
            label=(set_name or None),
            primary_set_id=(set_id or None),
            layer_id=primary_layer_id,
            overlays=[],
            owned_species=owned_species,
        ):
            return _fail("direct_display_failed")

        plot = self._main_plot()
        if plot is None:
            return _fail("main_plot_unavailable_after_direct_display")
        display_label = set_name or set_id or "Results"
        metadata_failure = self._apply_direct_completion_plot_metadata(
            plot=plot,
            t=t,
            series=series,
            display_label=display_label,
            algebra_scalars=algebra_scalars,
            layer_id=primary_layer_id,
            set_id=set_id,
        )
        if metadata_failure:
            return _fail(metadata_failure)
        try:
            self.publish_completion_intervention_annotations(solver_provenance)
        except Exception as exc:
            logger.exception("Failed to apply direct completion intervention annotations: %s", exc)
            return _fail("annotation_failed")
        provenance_snapshot = self._direct_completion_provenance_snapshot()
        provenance_failure = self._publish_direct_completion_provenance(
            plot=plot,
            direct_completion_provenance=direct_completion_provenance,
        )
        if provenance_failure:
            self._restore_direct_completion_provenance_snapshot(provenance_snapshot)
            return _fail(provenance_failure)
        commit_failure = self._commit_display_claim_state(
            primary_set_id=set_id,
            primary_label=set_name,
            selected_set_ids=([set_id] if set_id else []),
        )
        if commit_failure:
            self._restore_direct_completion_provenance_snapshot(provenance_snapshot)
            return _fail(commit_failure)
        self._commit_successful_plot_display(t=np.asarray(t, dtype=float), series=series)
        return SimulationCompletionDisplayOutcome(
            True,
            direct_completion_displayed=True,
            primary_set_id=(set_id or None),
            displayed_set_ids=((set_id,) if set_id else ()),
        )

    def publish_preserved_batch_display(
        self,
        *,
        primary_set_id: str,
        primary_set_name: str,
        selected_set_ids: Sequence[str],
        transaction_snapshot: Mapping[str, Any],
    ) -> CachedBatchSelectionDisplayOutcome:
        primary_set_id_s = str(primary_set_id or "").strip()
        selected_ids = [str(set_id) for set_id in (selected_set_ids or ()) if str(set_id)]
        if primary_set_id_s and primary_set_id_s not in selected_ids:
            selected_ids.append(primary_set_id_s)
        label = str(primary_set_name or primary_set_id_s or "Results")
        if not bool(transaction_snapshot.get("_kindred_display_transaction_snapshot")):
            return CachedBatchSelectionDisplayOutcome(False, reason="display_failed")
        rollback_snapshot = self._direct_display_failure_snapshot(self._main_plot())
        plot = self._main_plot()
        restorer = getattr(plot, "restore_display_transaction_snapshot", None) if plot is not None else None
        if not callable(restorer):
            return CachedBatchSelectionDisplayOutcome(False, reason="display_failed")
        try:
            restorer(transaction_snapshot)
            self._ui.set_results_table(self._ui.main_plot_stats_table())
        except Exception as exc:
            logger.exception("Failed to restore preserved batch display transaction: %s", exc)
            self._restore_direct_display_failure_snapshot(rollback_snapshot)
            return CachedBatchSelectionDisplayOutcome(False, reason="display_failed")
        commit_failure = self._commit_display_claim_state(
            primary_set_id=primary_set_id_s,
            primary_label=label,
            selected_set_ids=selected_ids,
            workspace_preview_provenance_by_set_id=dict(
                transaction_snapshot.get("workspace_preview_display_provenance_by_set_id") or {}
            ),
        )
        if commit_failure:
            self._restore_direct_display_failure_snapshot(rollback_snapshot)
            return CachedBatchSelectionDisplayOutcome(False, reason=commit_failure)
        self._commit_successful_plot_display(
            t=np.asarray(transaction_snapshot.get("t") if transaction_snapshot.get("t") is not None else [], dtype=float),
            series={
                str(name): np.asarray(values, dtype=float)
                for name, values in dict(transaction_snapshot.get("series") or {}).items()
            },
        )
        return CachedBatchSelectionDisplayOutcome(
            True,
            primary_set_id=primary_set_id_s or None,
            displayed_set_ids=tuple(selected_ids),
        )

    def publish_completion_intervention_annotations(
        self,
        solver_provenance: Mapping[str, Any] | None,
    ) -> None:
        plot = self._main_plot()
        if plot is None:
            return
        setter = getattr(plot, "set_intervention_annotations_from_provenance", None)
        if callable(setter):
            setter(solver_provenance if isinstance(solver_provenance, Mapping) else None)

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
            plot = self._main_plot()
            if plot is None:
                raise RuntimeError("Main plot widget not available")
            snapshot = self._main_plot_snapshot(plot)

            self._ui.set_main_plot_data(
                t,
                series,
                label=label,
                primary_set_id=primary_set_id,
                layer_id=layer_id,
                overlays=overlays,
                owned_species=owned_species,
            )
            self._restore_main_plot_selection(snapshot)
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
    ) -> None:
        try:
            self._ui.show_simulation_tab()
            self._ui.refresh_simulation_plot_views()
            self._ui.schedule_main_plot_refresh((50, 100))
            self._ui.set_status_text(f"Loaded {len(series)} species, {len(t)} timepoints")
            logger.info("Data set: %s species, %s points", int(len(series)), int(len(t)))
        except Exception as exc:
            logger.exception("Failed to apply post-commit plot display UI refresh: %s", exc)

    def _apply_intervention_annotations(self, *, plot: object, entry: Mapping[str, Any]) -> None:
        setter = getattr(plot, "set_intervention_annotations_from_provenance", None)
        if not callable(setter):
            return
        solver_provenance = entry.get("solver_provenance") if isinstance(entry, Mapping) else None
        setter(solver_provenance if isinstance(solver_provenance, Mapping) else None)
