from __future__ import annotations

import logging
from collections.abc import MutableMapping
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
from PySide6 import QtCore, QtWidgets

from kindred.core.batch_simulation_cache import BatchSimulationCache
from kindred.gui.controllers.cache_contracts import (
    BatchCacheEntryReadResult,
    BatchCacheEntryV1,
    build_overlay_entry,
    read_batch_cache_entry,
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
    active_batch_cache_key: Callable[[], str]
    active_batch_valid_set_ids: Callable[[], Sequence[str] | None]
    active_batch_invalidated_set_ids: Callable[[], Sequence[str] | None]
    active_batch_selection: Callable[[], tuple[str, str]]
    set_active_batch_selection: Callable[[str, str, Sequence[str]], None]
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


@dataclass(frozen=True, slots=True)
class CachedBatchSelectionDisplayOutcome:
    displayed: bool
    reason: Optional[str] = None


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
class ResolvedBatchSelectionEntry:
    set_id: str
    label: str
    entry: BatchCacheEntryV1
    canonical_entry: BatchCacheEntryV1 | None = None


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
    ) -> BatchCacheEntryReadResult:
        sid = str(set_id or "").strip()
        if not sid:
            return BatchCacheEntryReadResult("missing")
        direct = read_batch_cache_entry((store or {}).get(BatchSimulationCache.entry_key(cache_key, sid)))
        if direct.entry is not None:
            return direct
        name = self._ui.batch_name_for_id(sid)
        by_name = BatchCacheEntryReadResult("missing")
        if name:
            by_name = read_batch_cache_entry((store or {}).get(BatchSimulationCache.entry_key(cache_key, str(name))))
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
        allow_fallback: bool = True,
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
            result = self._cache_entry_for_set_id(store=store, cache_key=cache_key, set_id=sid)
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
        if not allow_fallback or not selected_ids:
            return CachedBatchAvailability([], has_invalid_entry=has_invalid_entry)
        focused_id = str(self._ui.focused_batch_set_id() or "")
        if focused_id and focused_id in selected_ids and focused_id not in invalidated_ids and (
            allowed_ids is None or focused_id in allowed_ids
        ):
            fallback_result = self._cache_entry_for_set_id(store=store, cache_key=cache_key, set_id=focused_id)
            if fallback_result.entry is not None:
                return CachedBatchAvailability([focused_id], has_invalid_entry=has_invalid_entry)
            if fallback_result.state == "invalid":
                has_invalid_entry = True
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
                and self._cache_entry_for_set_id(store=store, cache_key=cache_key, set_id=prefer_id).entry is not None
            ):
                return str(prefer_id)
        focused_id = str(self._ui.focused_batch_set_id() or "")
        if focused_id and focused_id in available:
            focused_entry = self._cache_entry_for_set_id(store=store, cache_key=cache_key, set_id=focused_id).entry
            if focused_entry is not None:
                return focused_id
        return str(available[0])

    def _cached_batch_overlays(
        self,
        *,
        store: MutableMapping[str, Dict[str, Any]],
        cache_key: str,
        available: Sequence[str],
        primary: str,
    ) -> List[Dict[str, object]]:
        overlays: List[Dict[str, object]] = []
        for sid in available:
            if sid == primary:
                continue
            other = self._cache_entry_for_set_id(store=store, cache_key=cache_key, set_id=sid).entry
            if other is None:
                continue
            overlay_label = self._ui.batch_name_for_id(sid) or str(sid)
            overlays.append(dict(build_overlay_entry(label=overlay_label, entry=other)))
        return overlays

    def _apply_cached_batch_plot_metadata(
        self,
        *,
        plot: object,
        store: MutableMapping[str, Dict[str, Any]],
        cache_key: str,
        available: Sequence[str],
        primary: str,
        primary_label: str,
        entry: BatchCacheEntryV1,
        t: np.ndarray,
        series: Dict[str, np.ndarray],
    ) -> None:
        scalars = entry.get("algebra_scalars") or {}
        try:
            self._ui.set_main_plot_scalar_values(scalars)
        except Exception as exc:
            logger.exception(
                "Failed to set plot scalar values for cached batch selection (primary=%s): %s",
                primary_label,
                exc,
            )

        stats_results_map: Dict[str, Dict[str, object]] = {}
        for sid in available:
            payload = self._cache_entry_for_set_id(store=store, cache_key=cache_key, set_id=sid).entry
            if payload is None:
                continue
            series_payload = payload.get("series") or {}
            if not series_payload:
                continue
            label = self._ui.batch_name_for_id(sid) or str(sid)
            stats_results_map[str(label)] = {"t": payload["t"], "series": dict(series_payload)}
        try:
            self._ui.update_main_plot_statistics(
                stats_results_map=stats_results_map,
                prefer=str(primary_label),
                t=np.asarray(t, dtype=float),
                series={str(k): np.asarray(v, dtype=float) for k, v in series.items()},
            )
        except Exception as exc:
            logger.exception(
                "Failed to update plot statistics for cached batch selection (primary=%s): %s",
                primary_label,
                exc,
            )
        try:
            self._ui.set_results_table(self._ui.main_plot_stats_table())
        except Exception as exc:
            logger.exception("Failed to fetch stats table from plot after cached selection: %s", exc)

    def _apply_resolved_batch_plot_metadata(
        self,
        *,
        plot: object,
        resolved_entries: Sequence[ResolvedBatchSelectionEntry],
        primary: ResolvedBatchSelectionEntry,
    ) -> None:
        scalars = primary.entry.get("algebra_scalars") or {}
        try:
            self._ui.set_main_plot_scalar_values(scalars)
        except Exception as exc:
            logger.exception(
                "Failed to set plot scalar values for resolved batch selection (primary=%s): %s",
                primary.label,
                exc,
            )

        stats_results_map: Dict[str, Dict[str, object]] = {}
        for resolved in resolved_entries:
            series_payload = resolved.entry.get("series") or {}
            if not series_payload:
                continue
            stats_results_map[str(resolved.label)] = {
                "t": resolved.entry["t"],
                "series": dict(series_payload),
            }
        try:
            self._ui.update_main_plot_statistics(
                stats_results_map=stats_results_map,
                prefer=str(primary.label),
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
        try:
            self._ui.set_results_table(self._ui.main_plot_stats_table())
        except Exception as exc:
            logger.exception("Failed to fetch stats table from plot after resolved selection: %s", exc)

    def _update_cached_batch_controller_state(self, *, primary: str, primary_label: str, available: Sequence[str]) -> None:
        self._ui.set_active_batch_selection(str(primary), str(primary_label), list(available))

    def refresh_batch_plot_after_set_mutation(self) -> None:
        cache_key = str(self._ui.active_batch_cache_key() or "")
        plot = self._main_plot()
        shown_ids = self._ui.shown_batch_set_ids()
        valid_set_ids_raw = self._ui.active_batch_valid_set_ids()
        invalidated_set_ids_raw = self._ui.active_batch_invalidated_set_ids()
        valid_set_ids = (
            tuple(str(set_id) for set_id in valid_set_ids_raw)
            if valid_set_ids_raw is not None
            else None
        )
        invalidated_set_ids = (
            tuple(str(set_id) for set_id in invalidated_set_ids_raw)
            if invalidated_set_ids_raw is not None
            else None
        )
        prefer_id = self._ui.focused_batch_set_id()
        displayed = False
        if cache_key:
            displayed = self.display_cached_batch_selection(
                cache_key=cache_key,
                selected_sets=shown_ids,
                prefer_set=prefer_id,
                valid_set_ids=valid_set_ids,
                invalidated_set_ids=invalidated_set_ids,
                allow_fallback=(valid_set_ids is None),
            )
        if displayed or plot is None:
            return
        if valid_set_ids is not None or invalidated_set_ids is not None:
            return
        try:
            t_existing = getattr(plot, "_t", None)
            series_existing = getattr(plot, "_series", {}) or {}
            if t_existing is None or not isinstance(series_existing, dict) or not series_existing:
                return
            plot.set_data(
                np.asarray(t_existing, dtype=float),
                {str(k): np.asarray(v, dtype=float) for k, v in series_existing.items()},
                label=self._ui.active_batch_selection()[1],
                overlays=[],
            )
        except Exception as exc:
            logger.exception(
                "Failed to refresh batch plot after set mutation from existing plot state (cache_key=%s): %s",
                cache_key,
                exc,
            )

    def cached_batch_selection_availability(
        self,
        *,
        cache_key: str,
        selected_sets: Sequence[str],
        cache_store: Optional[MutableMapping[str, Dict[str, Any]]] = None,
        valid_set_ids: Optional[Sequence[str]] = None,
        invalidated_set_ids: Optional[Sequence[str]] = None,
        allow_fallback: bool = True,
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
            allow_fallback=bool(allow_fallback),
        )

    def cached_batch_selection_coverage(
        self,
        *,
        cache_key: str,
        selected_sets: Sequence[str],
        cache_store: Optional[MutableMapping[str, Dict[str, Any]]] = None,
        valid_set_ids: Optional[Sequence[str]] = None,
        invalidated_set_ids: Optional[Sequence[str]] = None,
        allow_fallback: bool = True,
    ) -> CachedBatchSelectionCoverage:
        selected_ids = self._normalized_selected_batch_ids(selected_sets)
        availability = self.cached_batch_selection_availability(
            cache_key=cache_key,
            selected_sets=selected_sets,
            cache_store=cache_store,
            valid_set_ids=valid_set_ids,
            invalidated_set_ids=invalidated_set_ids,
            allow_fallback=bool(allow_fallback),
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

    def display_cached_batch_selection(
        self,
        *,
        cache_key: str,
        selected_sets: Sequence[str],
        prefer_set: Optional[str] = None,
        cache_store: Optional[MutableMapping[str, Dict[str, Any]]] = None,
        valid_set_ids: Optional[Sequence[str]] = None,
        invalidated_set_ids: Optional[Sequence[str]] = None,
        allow_fallback: bool = True,
    ) -> bool:
        return bool(
            self.display_cached_batch_selection_outcome(
                cache_key=cache_key,
                selected_sets=selected_sets,
                prefer_set=prefer_set,
                cache_store=cache_store,
                valid_set_ids=valid_set_ids,
                invalidated_set_ids=invalidated_set_ids,
                allow_fallback=bool(allow_fallback),
            ).displayed
        )

    def display_cached_batch_selection_outcome(
        self,
        *,
        cache_key: str,
        selected_sets: Sequence[str],
        prefer_set: Optional[str] = None,
        cache_store: Optional[MutableMapping[str, Dict[str, Any]]] = None,
        valid_set_ids: Optional[Sequence[str]] = None,
        invalidated_set_ids: Optional[Sequence[str]] = None,
        allow_fallback: bool = True,
    ) -> CachedBatchSelectionDisplayOutcome:
        cache_key = str(cache_key or "")
        if not cache_key:
            return CachedBatchSelectionDisplayOutcome(False, reason="cache_key_empty")
        store = cache_store if cache_store is not None else self._ui.result_cache_store()
        availability = self.cached_batch_selection_availability(
            cache_key=cache_key,
            selected_sets=selected_sets,
            cache_store=store,
            valid_set_ids=valid_set_ids,
            invalidated_set_ids=invalidated_set_ids,
            allow_fallback=bool(allow_fallback),
        )
        if not availability.available_ids:
            reason = "invalid_cache_entry" if availability.has_invalid_entry else "no_cached_results"
            return CachedBatchSelectionDisplayOutcome(False, reason=reason)

        primary = self._primary_cached_batch_id(
            store=store,
            cache_key=cache_key,
            available=availability.available_ids,
            prefer_set=prefer_set,
        )

        entry_result = self._cache_entry_for_set_id(store=store, cache_key=cache_key, set_id=primary)
        entry = entry_result.entry
        if entry is None:
            reason = "invalid_cache_entry" if entry_result.state == "invalid" else "no_cached_results"
            return CachedBatchSelectionDisplayOutcome(False, reason=reason)
        t = entry["t"]
        series = entry["series"]

        primary_label = self._ui.batch_name_for_id(primary) or str(primary)
        overlays = self._cached_batch_overlays(
            store=store,
            cache_key=cache_key,
            available=availability.available_ids,
            primary=primary,
        )

        self.set_data(
            np.asarray(t, dtype=float),
            {str(k): np.asarray(v, dtype=float) for k, v in series.items()},
            label=primary_label,
            overlays=overlays,
        )

        plot = self._main_plot()
        if plot is not None:
            self._apply_cached_batch_plot_metadata(
                plot=plot,
                store=store,
                cache_key=cache_key,
                available=availability.available_ids,
                primary=primary,
                primary_label=str(primary_label),
                entry=entry,
                t=np.asarray(t, dtype=float),
                series={str(k): np.asarray(v, dtype=float) for k, v in series.items()},
            )

        self._update_cached_batch_controller_state(
            primary=primary,
            primary_label=str(primary_label),
            available=availability.available_ids,
        )
        return CachedBatchSelectionDisplayOutcome(True)

    def display_resolved_batch_selection_outcome(
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
            dict(build_overlay_entry(label=resolved.label, entry=resolved.entry, set_id=resolved.set_id))
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
                        curve_role="canonical_ghost",
                    )
                )
            )
        self.set_data(
            np.asarray(primary.entry["t"], dtype=float),
            {
                str(k): np.asarray(v, dtype=float)
                for k, v in (primary.entry.get("series") or {}).items()
            },
            label=str(primary.label),
            overlays=overlays,
        )

        plot = self._main_plot()
        if plot is not None:
            self._apply_resolved_batch_plot_metadata(
                plot=plot,
                resolved_entries=list(resolved_entries),
                primary=primary,
            )

        self._update_cached_batch_controller_state(
            primary=str(primary.set_id),
            primary_label=str(primary.label),
            available=[str(resolved.set_id) for resolved in resolved_entries],
        )
        return CachedBatchSelectionDisplayOutcome(True)

    def set_data(
        self,
        t: np.ndarray,
        series: Dict[str, np.ndarray],
        *,
        label: Optional[str] = None,
        overlays: Optional[Sequence[Dict[str, object]]] = None,
        owned_species: Optional[Sequence[str]] = None,
    ) -> None:
        """Set simulation data to plot."""
        try:
            plot = self._main_plot()
            if plot is None:
                raise RuntimeError("Main plot widget not available")
            snapshot = self._main_plot_snapshot(plot)

            self._ui.set_main_plot_data(t, series, label=label, overlays=overlays, owned_species=owned_species)
            self._restore_main_plot_selection(snapshot)
            self._ui.show_simulation_tab()
            self._ui.refresh_simulation_plot_views()
            self._ui.schedule_main_plot_refresh((50, 100))
            self._ui.set_status_text(f"Loaded {len(series)} species, {len(t)} timepoints")
            logger.info("Data set: %s species, %s points", int(len(series)), int(len(t)))
        except Exception as exc:
            logger.warning("Failed to set data: %s", exc, exc_info=True)
            QtWidgets.QMessageBox.warning(self._ui.parent, "Error", f"Failed to set data: {exc}")
