from __future__ import annotations

import hashlib
import math
from typing import Any, Callable, Dict, List, MutableMapping, Optional, Sequence, Set, Tuple

import numpy as np

from kindred.core.batch_initial_conditions import (
    migrate_reaction_dsl_initial_concentration_sets,
    strip_reaction_dsl_initial_concentrations,
)
from kindred.core.batch_simulation_cache import BatchSimulationCache
from kindred.core.simulation_identity import (
    SimulationIdentity,
    canonical_initials_fingerprint,
    coerce_simulation_identity,
)
from kindred.core.validation import try_parse_finite_float
from kindred.core.batch_cache_contracts import BatchCacheEntryReadResult, read_batch_cache_entry
from kindred.gui.controllers.simulation_completion_policy import pending_initial_seed_for_set


class SimulationBatchOwner:
    """Thin Qt adapter for batch table/store selection and batch display state."""

    def __init__(
        self,
        *,
        batch_rows_for_scope: Callable[[str], Sequence[int]],
        batch_set_ids_for_scope: Callable[[str], Sequence[str]],
        shown_batch_set_ids: Callable[[], Sequence[str]],
        slider_edit_target_set_ids: Callable[[], Sequence[str]],
        focused_batch_set_id: Callable[[], Optional[str]],
        batch_current_row: Callable[[], Optional[int]],
        batch_set_id_for_row: Callable[[int], Optional[str]],
        batch_set_name_for_id: Callable[[str], Optional[str]],
        batch_set_id_for_name: Callable[[str], Optional[str]],
        batch_preferred_primary_set_id: Callable[[Sequence[int]], Optional[str]],
        batch_cache_key: Callable[..., str],
        batch_cache_getter: Callable[[], object],
        batch_store: object,
        batch_model: object,
        batch_initials_for_row: Callable[[int], Dict[str, float]],
        preview_session: object,
        mechanism_owner: object,
        solver_owner: object,
        results_controller_getter: Callable[[], object],
        set_status_text: Callable[[str], None],
        update_batch_row_controls_state: Callable[[], None],
        sync_batch_species_columns: Callable[..., None],
    ) -> None:
        self._batch_rows_for_scope = batch_rows_for_scope
        self._batch_set_ids_for_scope = batch_set_ids_for_scope
        self._shown_batch_set_ids = shown_batch_set_ids
        self._slider_edit_target_set_ids = slider_edit_target_set_ids
        self._focused_batch_set_id = focused_batch_set_id
        self._batch_current_row = batch_current_row
        self._batch_set_id_for_row = batch_set_id_for_row
        self._batch_set_name_for_id = batch_set_name_for_id
        self._batch_set_id_for_name = batch_set_id_for_name
        self._batch_preferred_primary_set_id = batch_preferred_primary_set_id
        self._batch_cache_key = batch_cache_key
        self._batch_cache_getter = batch_cache_getter
        self._batch_store = batch_store
        self._batch_model = batch_model
        self._batch_initials_for_row = batch_initials_for_row
        self._preview_session = preview_session
        self._mechanism_owner = mechanism_owner
        self._solver_owner = solver_owner
        self._results_controller_getter = results_controller_getter
        self._set_status_text = set_status_text
        self._update_batch_row_controls_state = update_batch_row_controls_state
        self._sync_batch_species_columns = sync_batch_species_columns

    def batch_rows_for_scope(self, scope: str) -> List[int]:
        return [int(row) for row in (self._batch_rows_for_scope(str(scope)) or [])]

    def batch_set_ids_for_scope(self, scope: str) -> List[str]:
        return [str(set_id) for set_id in (self._batch_set_ids_for_scope(str(scope)) or [])]

    def shown_batch_set_ids(self) -> List[str]:
        return [str(set_id) for set_id in (self._shown_batch_set_ids() or [])]

    def slider_edit_target_set_ids(self) -> List[str]:
        return [str(set_id) for set_id in (self._slider_edit_target_set_ids() or [])]

    def focused_batch_set_id(self) -> Optional[str]:
        value = self._focused_batch_set_id()
        return str(value) if value else None

    def batch_current_row(self) -> Optional[int]:
        row = self._batch_current_row()
        return int(row) if row is not None else None

    def batch_set_id_for_row(self, row: int) -> Optional[str]:
        value = self._batch_set_id_for_row(int(row))
        return str(value) if value is not None else None

    def batch_set_name_for_id(self, set_id: str) -> Optional[str]:
        value = self._batch_set_name_for_id(str(set_id))
        return str(value) if value is not None else None

    def batch_set_id_for_name(self, name: str) -> Optional[str]:
        value = self._batch_set_id_for_name(str(name))
        return str(value) if value is not None else None

    def batch_preferred_primary_set_id(self, rows: Sequence[int]) -> Optional[str]:
        value = self._batch_preferred_primary_set_id([int(row) for row in rows])
        return str(value) if value is not None else None

    def set_active_batch_selection(self, set_id: str, set_name: str, selected_ids: Sequence[str]) -> None:
        batch_cache = self._batch_cache()
        batch_cache.active_batch_set_id = str(set_id)
        batch_cache.active_batch_set = str(set_name)
        batch_cache.last_display_selection = [str(item) for item in (selected_ids or []) if str(item)]

    def active_batch_selection(self) -> tuple[str, str]:
        batch_cache = self._batch_cache()
        return (
            str(getattr(batch_cache, "active_batch_set_id", "") or ""),
            str(getattr(batch_cache, "active_batch_set", "") or ""),
        )

    def clear_display_selection_state(self) -> None:
        clear_display = getattr(self._batch_cache(), "clear_display_selection_state", None)
        if callable(clear_display):
            clear_display()

    def batch_cache_key(
        self,
        *,
        scope_identity: object | None = None,
        mechanism_text: str = "",
        solver_config: Optional[Dict[str, Any]] = None,
        t_end: float = 0.0,
    ) -> str:
        return str(
            self._batch_cache_key(
                scope_identity=scope_identity,
                mechanism_text=str(mechanism_text),
                solver_config=dict(solver_config or {}),
                t_end=float(t_end),
            )
        )

    def batch_result_cache_store(self) -> MutableMapping[str, Dict[str, Any]]:
        return self._batch_cache().result_cache

    def active_explicit_cache_entry_for_set(
        self,
        *,
        set_id: str,
        cache_key: str | None = None,
    ) -> BatchCacheEntryReadResult:
        batch_cache = self._batch_cache()
        active_cache_key = str(
            cache_key
            if cache_key is not None
            else getattr(batch_cache, "active_cache_key", "")
            or ""
        ).strip()
        if not active_cache_key:
            return BatchCacheEntryReadResult("missing")
        return self._cache_entry_for_set_id_from_store(
            store=getattr(batch_cache, "result_cache", {}),
            cache_key=active_cache_key,
            set_id=str(set_id),
        )

    def batch_cache_contains_set(self, *, set_id: str, set_name: str) -> bool:
        contains = getattr(self._batch_cache(), "contains_set_identifier", None)
        if callable(contains):
            return bool(contains(set_id=str(set_id), set_name=str(set_name)))
        return False

    def purge_batch_cache_for_deleted_sets(
        self,
        *,
        set_ids: Sequence[str],
        set_names: Sequence[str],
    ) -> int:
        purge = getattr(self._batch_cache(), "purge_entries_for_set_identifiers", None)
        if callable(purge):
            return int(
                purge(
                    set_ids=tuple(str(set_id) for set_id in set_ids),
                    set_names=tuple(str(name) for name in set_names),
                )
            )
        return 0

    def batch_store_row_count(self) -> int:
        return int(self._batch_store.row_count())

    def batch_store_set_names(self) -> List[str]:
        return [str(name) for name in (self._batch_store.set_names() or [])]

    def batch_store_visible_species(self) -> List[str]:
        return [str(name) for name in (self._batch_store.visible_species() or [])]

    def batch_model_validate_rows(self, rows: Sequence[int]) -> Set[Tuple[int, str]]:
        invalid = self._batch_model.validate_rows([int(row) for row in rows])
        if not invalid:
            return set()
        return {(int(row), str(species)) for row, species in invalid}

    def batch_initials_for_row(self, row: int) -> Dict[str, float]:
        initials = self._batch_initials_for_row(int(row))
        if not initials:
            return {}
        if isinstance(initials, dict):
            return {str(key): float(value) for key, value in initials.items()}
        return dict(initials)

    def display_cached_batch_selection(
        self,
        *,
        cache_key: str,
        selected_sets: Sequence[str],
        prefer_set: Optional[str] = None,
        cache_store: Optional[object] = None,
        valid_set_ids: Optional[Sequence[str]] = None,
        invalidated_set_ids: Optional[Sequence[str]] = None,
        allow_fallback: bool = True,
    ) -> bool:
        batch_cache = self._batch_cache()
        normalized_selected_sets = [str(set_id) for set_id in (selected_sets or ()) if str(set_id)]
        if cache_store is getattr(batch_cache, "preview_cache", None) and normalized_selected_sets:
            workspace_displayed = self._display_workspace_aware_preview_batch_selection(
                selected_sets=normalized_selected_sets,
                prefer_set=prefer_set,
                preview_cache_key=str(cache_key or ""),
            )
            if workspace_displayed:
                return True
            if len(normalized_selected_sets) > 1:
                return False
            if not bool(allow_fallback):
                single_set_id = str(normalized_selected_sets[0] or "")
                if single_set_id and self._preview_has_dirty_state_for_set(single_set_id):
                    preview_entry = self.matching_preview_entry_for_workspace_set(
                        set_id=single_set_id,
                        preview_cache_key=str(cache_key or ""),
                    )
                    if preview_entry.entry is None:
                        return False
        resolved_invalidated_set_ids = invalidated_set_ids
        if (
            resolved_invalidated_set_ids is None
            and str(getattr(batch_cache, "active_cache_key", "") or "") == str(cache_key)
        ):
            resolved_invalidated_set_ids = getattr(batch_cache, "active_cache_invalidated_set_ids", None)
        displayed = bool(
            self._results_controller().display_cached_batch_selection(
                cache_key=str(cache_key),
                selected_sets=normalized_selected_sets,
                prefer_set=str(prefer_set) if prefer_set is not None else None,
                cache_store=cache_store,
                valid_set_ids=(
                    tuple(str(set_id) for set_id in valid_set_ids)
                    if valid_set_ids is not None
                    else None
                ),
                invalidated_set_ids=(
                    tuple(str(set_id) for set_id in resolved_invalidated_set_ids)
                    if resolved_invalidated_set_ids is not None
                    else None
                ),
                allow_fallback=bool(allow_fallback),
            )
        )
        if displayed:
            self.record_current_main_plot_workspace_preview_provenance(selected_set_ids=normalized_selected_sets)
        return displayed

    def display_workspace_aware_batch_selection(
        self,
        *,
        selected_sets: Sequence[str],
        prefer_set: Optional[str] = None,
        preview_cache_key: Optional[str] = None,
    ) -> bool:
        return bool(
            self._display_workspace_aware_preview_batch_selection(
                selected_sets=selected_sets,
                prefer_set=prefer_set,
                preview_cache_key=preview_cache_key,
            )
        )

    def focused_batch_selection_is_dirty(
        self,
        *,
        selected_sets: Sequence[str],
        prefer_set: Optional[str] = None,
    ) -> bool:
        focused_set_id = str(prefer_set or (selected_sets[0] if selected_sets else "") or "").strip()
        if not focused_set_id:
            return False
        return self._preview_has_dirty_state_for_set(focused_set_id)

    def selection_uses_fresh_explicit_cache_after_post_run_sync(
        self,
        *,
        selected_sets: Sequence[str],
    ) -> bool:
        normalized_selected_sets = [str(set_id) for set_id in (selected_sets or ()) if str(set_id)]
        if not normalized_selected_sets:
            return False
        batch_cache = self._batch_cache()
        active_cache_key = str(getattr(batch_cache, "active_cache_key", "") or "").strip()
        active_preview_token = str(getattr(batch_cache, "active_cache_preview_token", "") or "").strip()
        if not active_cache_key or not active_preview_token:
            return False
        active_valid_set_ids = {
            str(set_id) for set_id in (getattr(batch_cache, "active_cache_valid_set_ids", None) or ()) if str(set_id)
        }
        if active_valid_set_ids and any(set_id not in active_valid_set_ids for set_id in normalized_selected_sets):
            return False
        active_preview_scope_ids = {
            str(set_id)
            for set_id in (getattr(batch_cache, "active_cache_preview_scope_set_ids", None) or ())
            if str(set_id)
        }
        if active_preview_scope_ids and any(set_id not in active_preview_scope_ids for set_id in normalized_selected_sets):
            return False
        scope_rows: list[int] = []
        row_for_set_id = getattr(self._batch_store, "row_for_set_id", None)
        if not callable(row_for_set_id):
            return False
        for set_id in normalized_selected_sets:
            try:
                row = row_for_set_id(str(set_id))
            except Exception:
                row = None
            if row is None:
                return False
            scope_rows.append(int(row))
        try:
            current_preview_token = str(self._preview_session.preview_batch_cache_token(scope_rows) or "").strip()
        except Exception:
            return False
        return bool(current_preview_token) and current_preview_token == active_preview_token

    def resolve_workspace_aware_batch_selection(
        self,
        *,
        selected_sets: Sequence[str],
        preview_cache_key: Optional[str] = None,
    ) -> Tuple[List[object], Optional[str], bool, bool, bool, bool, bool]:
        from kindred.gui.controllers.results_controller import ResolvedBatchSelectionEntry

        batch_cache = self._batch_cache()
        active_cache_key = str(getattr(batch_cache, "active_cache_key", "") or "").strip()
        invalidated_set_ids = {
            str(set_id) for set_id in (getattr(batch_cache, "active_cache_invalidated_set_ids", None) or ()) if str(set_id)
        }
        focused_set_id = str(self._focused_batch_set_id() or "").strip()
        if (not focused_set_id) and selected_sets:
            focused_set_id = str(selected_sets[0] or "").strip()

        resolved_entries: List[object] = []
        has_workspace_selection = False
        has_resolved_workspace_preview = False
        focused_selection_uses_workspace_controls = False
        focused_selection_has_resolved_entry = False
        missing_workspace_entry = False
        missing_explicit_entry = False
        invalid_entry = False

        for raw_set_id in selected_sets or ():
            set_id = str(raw_set_id or "").strip()
            if not set_id:
                continue
            label = self.batch_set_name_for_id(set_id) or set_id
            if self._preview_has_dirty_state_for_set(set_id):
                has_workspace_selection = True
                preview_entry = self.matching_preview_entry_for_workspace_set(
                    set_id=set_id,
                    preview_cache_key=preview_cache_key,
                )
                if preview_entry.entry is not None:
                    has_resolved_workspace_preview = True
                    canonical_entry = None
                    if active_cache_key and set_id not in invalidated_set_ids:
                        explicit_entry = self._cache_entry_for_set_id_from_store(
                            store=getattr(batch_cache, "result_cache", {}),
                            cache_key=active_cache_key,
                            set_id=set_id,
                        )
                        canonical_entry = explicit_entry.entry
                    resolved_entries.append(
                        ResolvedBatchSelectionEntry(
                            set_id=str(set_id),
                            label=str(label),
                            entry=preview_entry.entry,
                            canonical_entry=canonical_entry,
                        )
                    )
                    if set_id == focused_set_id:
                        focused_selection_uses_workspace_controls = True
                        focused_selection_has_resolved_entry = True
                elif preview_entry.state == "invalid":
                    invalid_entry = True
                else:
                    missing_workspace_entry = True
                continue

            if not active_cache_key:
                missing_explicit_entry = True
                continue
            explicit_entry = self._cache_entry_for_set_id_from_store(
                store=getattr(batch_cache, "result_cache", {}),
                cache_key=active_cache_key,
                set_id=set_id,
            )
            if set_id in invalidated_set_ids:
                if explicit_entry.state == "invalid":
                    invalid_entry = True
                else:
                    missing_explicit_entry = True
                continue
            if explicit_entry.entry is not None:
                resolved_entries.append(
                    ResolvedBatchSelectionEntry(set_id=str(set_id), label=str(label), entry=explicit_entry.entry)
                )
                if set_id == focused_set_id:
                    focused_selection_uses_workspace_controls = False
                    focused_selection_has_resolved_entry = True
            elif explicit_entry.state == "invalid":
                invalid_entry = True
            else:
                missing_explicit_entry = True

        all_selected_sets_resolved = len(resolved_entries) == len(
            [str(set_id) for set_id in (selected_sets or ()) if str(set_id)]
        )
        reason: Optional[str] = None
        if invalid_entry:
            reason = "invalid_cache_entry"
        elif missing_workspace_entry:
            reason = "preview_pending"
        elif missing_explicit_entry:
            reason = "no_cached_results"
        return (
            resolved_entries,
            reason,
            all_selected_sets_resolved,
            has_workspace_selection,
            has_resolved_workspace_preview,
            focused_selection_uses_workspace_controls,
            focused_selection_has_resolved_entry,
        )

    def matching_preview_entry_for_workspace_set(
        self,
        *,
        set_id: str,
        preview_cache_key: Optional[str] = None,
    ) -> BatchCacheEntryReadResult:
        preview_store = getattr(self._batch_cache(), "preview_cache", {})
        expected_mechanism_text = self._mechanism_text_for_workspace_selection(set_id=str(set_id))
        resolved_preview_cache_key = str(
            preview_cache_key
            if preview_cache_key is not None
            else (getattr(self._batch_cache(), "active_preview_cache_key", "") or "")
        ).strip()

        try:
            expected_identity = self.current_workspace_preview_identity(set_id=str(set_id))
            expected_solver_config, expected_t_end, expected_overlay_token = self._current_workspace_preview_context(
                set_id=str(set_id),
                mechanism_text=str(expected_mechanism_text),
            )
        except Exception:
            return BatchCacheEntryReadResult("missing")

        def _entry_matches_expected(result: BatchCacheEntryReadResult) -> bool:
            if result.entry is None:
                return False
            entry_identity = coerce_simulation_identity(result.entry.get("simulation_identity"))
            if entry_identity is not None:
                if entry_identity != expected_identity:
                    return False
            else:
                if str(result.entry.get("mechanism_text") or "") != str(expected_mechanism_text):
                    return False
                if dict(result.entry.get("solver_config") or {}) != dict(expected_solver_config):
                    return False
                if str(result.entry.get("preview_batch_cache_token") or "") != str(expected_overlay_token):
                    return False
            entry_t_payload = result.entry.get("t")
            entry_t = np.asarray(entry_t_payload if entry_t_payload is not None else [], dtype=float).reshape(-1)
            expected_grid_n = int((expected_solver_config.get("grid") or {}).get("N") or 0)
            if expected_grid_n > 0 and int(entry_t.size) != expected_grid_n:
                return False
            if entry_t.size <= 0:
                return False
            return math.isclose(float(entry_t[-1]), float(expected_t_end), rel_tol=1e-9, abs_tol=1e-12)

        invalid_found = False
        direct = self._cache_entry_for_set_id_from_store(
            store=preview_store,
            cache_key=resolved_preview_cache_key,
            set_id=str(set_id),
        )
        if _entry_matches_expected(direct):
            return direct
        if direct.state == "invalid":
            invalid_found = True

        candidate_suffixes = {f"::{str(set_id)}"}
        set_name = self.batch_set_name_for_id(str(set_id))
        if set_name:
            candidate_suffixes.add(f"::{str(set_name)}")

        preview_data = getattr(preview_store, "_data", None)
        if hasattr(preview_data, "items"):
            preview_items = list(preview_data.items())
        else:
            preview_items = list((preview_store or {}).items())

        for key, payload in reversed(preview_items):
            key_s = str(key)
            if not any(key_s.endswith(suffix) for suffix in candidate_suffixes):
                continue
            if resolved_preview_cache_key and key_s.startswith(f"{resolved_preview_cache_key}::"):
                continue
            result = read_batch_cache_entry(payload)
            if _entry_matches_expected(result):
                return result
            invalid_found = invalid_found or result.state == "invalid"

        return BatchCacheEntryReadResult("invalid" if invalid_found else "missing")

    def current_workspace_preview_identity_payload(self, *, set_id: str) -> Optional[Dict[str, Any]]:
        sid = str(set_id or "").strip()
        if not sid:
            return None
        try:
            identity = self.current_workspace_preview_identity(set_id=sid)
        except Exception:
            return None
        try:
            return dict(identity.to_payload())
        except Exception:
            return None

    def current_workspace_preview_identity(self, *, set_id: str) -> SimulationIdentity:
        mechanism_text = self._mechanism_text_for_workspace_selection(set_id=str(set_id))
        expected_solver_config, expected_t_end, expected_overlay_token = self._current_workspace_preview_context(
            set_id=str(set_id),
            mechanism_text=mechanism_text,
        )
        try:
            from kindred.core.intervention_schedule import intervention_schedule_fingerprint_from_dsl_text

            intervention_schedule_fingerprint = str(
                intervention_schedule_fingerprint_from_dsl_text(str(mechanism_text or "")) or ""
            )
        except Exception:
            intervention_schedule_fingerprint = hashlib.sha256(
                str(mechanism_text or "").encode("utf-8", "surrogatepass")
            ).hexdigest()
        initials_fingerprint = ""
        row = self._batch_row_for_set_id(str(set_id))
        if row is not None:
            try:
                baseline_initials = self.batch_initials_for_row(int(row))
                reactions_text_raw = self._mechanism_owner.mechanism_reactions_text_raw()
                if self._mechanism_owner.has_slider_overrides():
                    reactions_text_raw = self._mechanism_owner.apply_overrides_to_text(
                        reactions_text_raw,
                        set_id=str(set_id),
                    )
                try:
                    pending_init_seed, _migrated = migrate_reaction_dsl_initial_concentration_sets(
                        reactions_text_raw,
                        default_set_name="set1",
                    )
                except Exception:
                    pending_init_seed = {}
                set_name = str(self.batch_set_name_for_id(str(set_id)) or "")
                for species, value in pending_initial_seed_for_set(
                    pending_init_seed,
                    set_name=set_name,
                ).items():
                    parsed, ok = try_parse_finite_float(value)
                    if ok:
                        baseline_initials[str(species)] = float(parsed)
                preview_initials = self._preview_session.preview_initials_for_row(int(row), baseline_initials)
                initials_fingerprint = canonical_initials_fingerprint(preview_initials)
            except Exception:
                initials_fingerprint = ""
        return SimulationIdentity.build(
            schema_id=self._mechanism_owner.simulation_schema_id(),
            param_fingerprint=self._mechanism_owner.simulation_param_fingerprint(set_id=str(set_id)),
            canonical_initials_fingerprint=initials_fingerprint,
            solver_config=expected_solver_config,
            t_end=expected_t_end,
            intervention_schedule_fingerprint=intervention_schedule_fingerprint,
            preview_batch_cache_token=expected_overlay_token,
            execution_flags=("fast_mode",),
        )

    def current_workspace_preview_context(
        self,
        *,
        set_id: str,
        mechanism_text: str,
    ) -> tuple[Dict[str, Any], float, str]:
        return self._current_workspace_preview_context(
            set_id=str(set_id),
            mechanism_text=str(mechanism_text),
        )

    def mechanism_text_for_workspace_selection(self, *, set_id: str) -> str:
        return self._mechanism_text_for_workspace_selection(set_id=str(set_id))

    def record_current_main_plot_workspace_preview_provenance(
        self,
        *,
        selected_set_ids: Sequence[str],
    ) -> None:
        selected_ids = [str(set_id) for set_id in (selected_set_ids or ()) if str(set_id)]
        plot = self._main_plot()
        if plot is None or not selected_ids:
            self._set_main_plot_workspace_preview_provenance({})
            return

        active_set_id = str(self.active_batch_selection()[0] or "").strip()
        if (not active_set_id) and selected_ids:
            active_set_id = selected_ids[0]
        if not active_set_id:
            self._set_main_plot_workspace_preview_provenance({})
            return

        current_t_raw = getattr(plot, "_t", None)
        current_t = np.asarray(current_t_raw if current_t_raw is not None else [], dtype=float).reshape(-1)
        current_series = dict(getattr(plot, "_series", {}) or {})
        if current_t.size <= 0 or not current_series:
            self._set_main_plot_workspace_preview_provenance({})
            return

        selected_local_workspace_ids = {
            set_id for set_id in selected_ids if self._preview_has_local_mechanism_workspace(set_id)
        }
        selected_overlay_dirty_ids: set[str] = set()
        for set_id in selected_ids:
            row = self._batch_row_for_set_id(str(set_id))
            if row is not None:
                try:
                    if bool(self._preview_session.preview_batch_cache_token([int(row)])):
                        selected_overlay_dirty_ids.add(str(set_id))
                except Exception:
                    continue

        selected_dirty_overlay_ids = {
            str(set_id)
            for set_id in selected_ids
            if str(set_id)
            and (
                str(set_id) in selected_local_workspace_ids
                or str(set_id) in selected_overlay_dirty_ids
            )
        }
        provenance_by_set_id: Dict[str, Dict[str, Any]] = {}
        active_requires_truthful_dirty_preview = bool(
            active_set_id in selected_local_workspace_ids or active_set_id in selected_overlay_dirty_ids
        )
        if active_requires_truthful_dirty_preview:
            active_preview_entry = self.matching_preview_entry_for_workspace_set(set_id=active_set_id)
            if self._batch_cache_entry_matches_plot_payload(
                entry=active_preview_entry.entry,
                t=current_t,
                series=current_series,
            ):
                active_payload = self.current_workspace_preview_identity_payload(set_id=active_set_id)
                if isinstance(active_payload, dict):
                    provenance_by_set_id[active_set_id] = active_payload

        overlay_label_to_set_id: Dict[str, str] = {}
        for set_id in selected_ids:
            set_id_s = str(set_id or "").strip()
            if not set_id_s:
                continue
            overlay_label_to_set_id[set_id_s] = set_id_s
            set_name = str(self.batch_set_name_for_id(set_id_s) or "").strip()
            if set_name:
                overlay_label_to_set_id[set_name] = set_id_s
        for entry in list(getattr(plot, "_simulation_overlays", []) or []):
            if not isinstance(entry, dict):
                continue
            overlay_label = str(entry.get("label") or "").strip()
            overlay_set_id = str(entry.get("set_id") or "").strip() or overlay_label_to_set_id.get(overlay_label, "")
            if not overlay_set_id or overlay_set_id not in selected_dirty_overlay_ids:
                continue
            overlay_t = np.asarray(entry.get("t") if entry.get("t") is not None else [], dtype=float).reshape(-1)
            overlay_series_raw = entry.get("series") or {}
            if overlay_t.size <= 0 or not isinstance(overlay_series_raw, dict):
                continue
            overlay_series: Dict[str, np.ndarray] = {}
            for species_name, values in overlay_series_raw.items():
                overlay_arr = np.asarray(values, dtype=float).reshape(-1)
                if overlay_arr.size <= 0:
                    continue
                overlay_series[str(species_name)] = overlay_arr
            if not overlay_series:
                continue
            overlay_preview_entry = self.matching_preview_entry_for_workspace_set(set_id=overlay_set_id)
            if not self._batch_cache_entry_matches_plot_payload(
                entry=overlay_preview_entry.entry,
                t=overlay_t,
                series=overlay_series,
            ):
                continue
            overlay_payload = self.current_workspace_preview_identity_payload(set_id=overlay_set_id)
            if isinstance(overlay_payload, dict):
                provenance_by_set_id[overlay_set_id] = overlay_payload

        self._set_main_plot_workspace_preview_provenance(provenance_by_set_id)

    def displayed_workspace_preview_provenance_matches_current_workspace(self, *, set_id: str) -> bool:
        sid = str(set_id or "").strip()
        if not sid:
            return False
        current_payload = self.current_workspace_preview_identity_payload(set_id=sid)
        if not isinstance(current_payload, dict):
            return False
        stored_payload = self._main_plot_workspace_preview_provenance().get(sid)
        return isinstance(stored_payload, dict) and stored_payload == current_payload

    def batch_cache_entry_matches_plot_payload(
        self,
        *,
        entry: Optional[MutableMapping[str, Any]],
        t: np.ndarray,
        series: MutableMapping[str, Any],
    ) -> bool:
        return self._batch_cache_entry_matches_plot_payload(entry=entry, t=t, series=series)

    def update_batch_row_controls_state(self) -> None:
        self._update_batch_row_controls_state()

    def sync_batch_species_columns(
        self,
        species_names: Sequence[str],
        *,
        preserve_active_cache: bool = False,
    ) -> None:
        self._sync_batch_species_columns(
            [str(species) for species in species_names],
            preserve_active_cache=bool(preserve_active_cache),
        )

    def _batch_cache(self) -> object:
        return self._batch_cache_getter()

    def _results_controller(self) -> object:
        return self._results_controller_getter()

    def _main_plot(self) -> object | None:
        main_plot = getattr(self._results_controller(), "main_plot", None)
        if not callable(main_plot):
            return None
        try:
            return main_plot()
        except Exception:
            return None

    def _set_main_plot_workspace_preview_provenance(
        self,
        provenance_by_set_id: MutableMapping[str, MutableMapping[str, Any]] | Dict[str, Dict[str, Any]],
    ) -> None:
        plot = self._main_plot()
        if plot is None:
            return
        cleaned: Dict[str, Dict[str, Any]] = {}
        for raw_set_id, raw_payload in dict(provenance_by_set_id or {}).items():
            set_id = str(raw_set_id or "").strip()
            if not set_id or not isinstance(raw_payload, dict):
                continue
            cleaned[set_id] = dict(raw_payload)
        setattr(plot, "_workspace_preview_display_provenance_by_set_id", cleaned)

    def _main_plot_workspace_preview_provenance(self) -> Dict[str, Dict[str, Any]]:
        plot = self._main_plot()
        raw = getattr(plot, "_workspace_preview_display_provenance_by_set_id", None) if plot is not None else None
        if not isinstance(raw, dict):
            return {}
        cleaned: Dict[str, Dict[str, Any]] = {}
        for raw_set_id, raw_payload in dict(raw).items():
            set_id = str(raw_set_id or "").strip()
            if not set_id or not isinstance(raw_payload, dict):
                continue
            cleaned[set_id] = dict(raw_payload)
        return cleaned

    def _batch_cache_entry_matches_plot_payload(
        self,
        *,
        entry: Optional[MutableMapping[str, Any]],
        t: np.ndarray,
        series: MutableMapping[str, Any],
    ) -> bool:
        if not isinstance(entry, dict):
            return False
        entry_t_payload = entry.get("t")
        entry_t = np.asarray(entry_t_payload if entry_t_payload is not None else [], dtype=float).reshape(-1)
        plot_t = np.asarray(t, dtype=float).reshape(-1)
        if entry_t.size <= 0 or entry_t.shape != plot_t.shape:
            return False
        if not np.allclose(entry_t, plot_t, rtol=1e-9, atol=1e-12):
            return False
        entry_series_raw = entry.get("series") or {}
        if not isinstance(entry_series_raw, dict):
            return False
        plot_series = {
            str(species_name): np.asarray(values, dtype=float).reshape(-1)
            for species_name, values in dict(series or {}).items()
        }
        entry_series = {
            str(species_name): np.asarray(values, dtype=float).reshape(-1)
            for species_name, values in dict(entry_series_raw).items()
        }
        if set(entry_series.keys()) != set(plot_series.keys()):
            return False
        for species_name, plot_values in plot_series.items():
            entry_values = entry_series.get(str(species_name))
            if entry_values is None or entry_values.shape != plot_values.shape:
                return False
            if not np.allclose(entry_values, plot_values, rtol=1e-9, atol=1e-12):
                return False
        return True

    def _display_workspace_aware_preview_batch_selection(
        self,
        *,
        selected_sets: Sequence[str],
        prefer_set: Optional[str] = None,
        preview_cache_key: Optional[str] = None,
    ) -> bool:
        normalized_selected_sets = [str(set_id) for set_id in (selected_sets or ()) if str(set_id)]
        if not normalized_selected_sets:
            return False
        focused_selection_is_dirty = self.focused_batch_selection_is_dirty(
            selected_sets=normalized_selected_sets,
            prefer_set=prefer_set,
        )
        (
            resolved_entries,
            outcome_reason,
            all_selected_sets_resolved,
            has_workspace_selection,
            has_resolved_workspace_preview,
            focused_selection_uses_workspace_controls,
            focused_selection_has_resolved_entry,
        ) = self.resolve_workspace_aware_batch_selection(
            selected_sets=normalized_selected_sets,
            preview_cache_key=preview_cache_key,
        )
        if not has_workspace_selection:
            return False
        if all_selected_sets_resolved and resolved_entries:
            outcome = self._results_controller().display_resolved_batch_selection_outcome(
                resolved_entries=resolved_entries,
                prefer_set=prefer_set,
            )
            if outcome.displayed:
                self.record_current_main_plot_workspace_preview_provenance(
                    selected_set_ids=normalized_selected_sets
                )
            return bool(outcome.displayed)
        if (
            resolved_entries
            and outcome_reason in {"preview_pending", "no_cached_results"}
            and has_resolved_workspace_preview
            and (
                bool(focused_selection_uses_workspace_controls)
                or ((not bool(focused_selection_is_dirty)) and bool(focused_selection_has_resolved_entry))
            )
        ):
            outcome = self._results_controller().display_resolved_batch_selection_outcome(
                resolved_entries=resolved_entries,
                prefer_set=prefer_set,
            )
            if outcome.displayed:
                self.record_current_main_plot_workspace_preview_provenance(
                    selected_set_ids=normalized_selected_sets
                )
                if outcome_reason == "preview_pending":
                    self._set_status_text("Preview pending for current selection.")
                else:
                    self._set_status_text("Result not cached (evicted). Press Run to compute.")
            return bool(outcome.displayed)
        return False

    def _cache_entry_for_set_id_from_store(
        self,
        *,
        store: MutableMapping[str, Dict[str, Any]],
        cache_key: str,
        set_id: str,
    ) -> BatchCacheEntryReadResult:
        sid = str(set_id or "").strip()
        if not sid or not cache_key:
            return BatchCacheEntryReadResult("missing")
        direct = read_batch_cache_entry((store or {}).get(BatchSimulationCache.entry_key(cache_key, sid)))
        if direct.entry is not None:
            return direct
        name = self.batch_set_name_for_id(sid)
        by_name = BatchCacheEntryReadResult("missing")
        if name:
            by_name = read_batch_cache_entry((store or {}).get(BatchSimulationCache.entry_key(cache_key, str(name))))
            if by_name.entry is not None:
                return by_name
        if direct.state == "invalid" or by_name.state == "invalid":
            return BatchCacheEntryReadResult("invalid")
        return BatchCacheEntryReadResult("missing")

    def _mechanism_text_for_workspace_selection(self, *, set_id: str) -> str:
        reactions_text = self._mechanism_owner.mechanism_reactions_text_raw()
        if self._mechanism_owner.has_slider_overrides():
            reactions_text = self._mechanism_owner.apply_overrides_to_text(reactions_text, set_id=str(set_id))
        reactions_text = strip_reaction_dsl_initial_concentrations(reactions_text)

        state_network_dsl = self._mechanism_owner.mechanism_state_network_dsl_raw()
        if self._mechanism_owner.has_slider_overrides():
            state_network_dsl = self._mechanism_owner.apply_overrides_to_state_network_dsl(
                state_network_dsl,
                set_id=str(set_id),
            )

        full_dsl = reactions_text
        if state_network_dsl.strip():
            full_dsl += "\n\n# State Network\n" + state_network_dsl
        if self._mechanism_owner.has_slider_overrides():
            full_dsl = self._mechanism_owner.apply_parameter_overrides_to_dsl(
                full_dsl,
                self._normalized_slider_overrides(set_id=str(set_id)),
            )
        return str(full_dsl)

    def _current_workspace_preview_context(
        self,
        *,
        set_id: str,
        mechanism_text: str,
    ) -> tuple[Dict[str, Any], float, str]:
        from kindred.gui.controllers.simulation_run_preparation import build_fast_preview_solver_grid_context

        solver_grid_context = build_fast_preview_solver_grid_context(
            initial_solver_name=self._solver_owner.initial_solver_name(),
            num_points=int(self._solver_owner.num_points_spinbox_value()),
            fast_mode=True,
            slider_points_override=self._mechanism_owner.mechanism_slider_points_value(),
            slider_solver_override=self._mechanism_owner.mechanism_slider_solver_value(),
            slider_drag_active=bool(self._preview_session.slider_drag_active()),
            last_slider_change_name=str(self._preview_session.last_slider_change_name() or ""),
        )
        temperature_k = float(self._solver_owner.temperature_spinbox_value())
        t_override = self._solver_owner.dsl_global_temperature_K(str(mechanism_text))
        if t_override is not None:
            temperature_k = float(t_override)
        solver_config = {
            "solver": str(solver_grid_context.get("solver") or ""),
            "solver_label": str(solver_grid_context.get("solver_label") or ""),
            "solver_warning": (
                str(solver_grid_context.get("solver_warning"))
                if solver_grid_context.get("solver_warning")
                else None
            ),
            "rtol": self._solver_owner.initial_rtol() or 1e-6,
            "atol": self._solver_owner.initial_atol() or 1e-12,
            "grid": dict(solver_grid_context.get("grid") or {"N": int(self._solver_owner.num_points_spinbox_value())}),
            "temperature_K": float(temperature_k),
            "use_sparse_jacobian": bool(self._solver_owner.use_sparse_jacobian()),
            "wegscheider_cyclicity_enabled": bool(self._solver_owner.wegscheider_cyclicity_enabled()),
        }
        overlay_token = ""
        row = self._batch_row_for_set_id(str(set_id))
        if row is not None:
            try:
                overlay_token = str(self._preview_session.preview_batch_cache_token([int(row)]) or "")
            except Exception:
                overlay_token = ""
        return solver_config, float(self._solver_owner.parse_sim_time_seconds()), overlay_token

    def _normalized_slider_overrides(
        self,
        *,
        set_id: Optional[str] = None,
        overrides: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        raw = self._mechanism_owner.slider_overrides(set_id=set_id) if overrides is None else dict(overrides or {})
        normalized: Dict[str, float] = {}
        for key, value in raw.items():
            parsed, ok = try_parse_finite_float(value)
            if not ok:
                continue
            normalized[str(key)] = float(parsed)
        return normalized

    def _batch_row_for_set_id(self, set_id: str) -> Optional[int]:
        row_for_set_id = getattr(self._batch_store, "row_for_set_id", None)
        if not callable(row_for_set_id):
            return None
        try:
            row = row_for_set_id(str(set_id or ""))
        except Exception:
            return None
        return int(row) if row is not None else None

    def _preview_has_dirty_state_for_set(self, set_id: str) -> bool:
        try:
            return bool(self._preview_session.has_dirty_state_for_set(str(set_id)))
        except Exception:
            return False

    def _preview_has_local_mechanism_workspace(self, set_id: str) -> bool:
        try:
            return bool(self._preview_session.has_local_mechanism_workspace(str(set_id)))
        except Exception:
            return False
