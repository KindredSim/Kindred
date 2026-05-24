from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

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
from kindred.core.batch_cache_contracts import (
    BatchCacheResultReadSnapshot,
    BatchCacheEntryReadResult,
)
from kindred.gui.ports import (
    BatchDisplayRequestCoverage,
    BatchDisplayRequestResolution,
    DisplayTransitionCause,
    ResolvedBatchDisplayRequestEntry,
)
from kindred.gui.controllers.simulation_completion_policy import pending_initial_seed_for_set


@dataclass(frozen=True, slots=True)
class BatchSpeciesColumnSyncSnapshot:
    active_cache_key: str
    active_preview_token: str
    active_preview_scope_ids: tuple[str, ...]
    active_valid_set_ids: tuple[str, ...]
    scope_tokens_before: Mapping[str, str]


class SimulationBatchOwner:
    """Thin Qt adapter for batch table/store, cache identity, and display-request resolution."""

    def __init__(
        self,
        *,
        batch_rows_for_scope: Callable[[str], Sequence[int]],
        batch_set_ids_for_scope: Callable[[str], Sequence[str]],
        requested_show_batch_set_ids: Callable[[], Sequence[str]],
        slider_edit_target_set_ids: Callable[[], Sequence[str]],
        focused_batch_set_id: Callable[[], Optional[str]],
        batch_current_row: Callable[[], Optional[int]],
        batch_set_id_for_row: Callable[[int], Optional[str]],
        batch_set_name_for_id: Callable[[str], Optional[str]],
        batch_set_id_for_name: Callable[[str], Optional[str]],
        batch_preferred_primary_set_id: Callable[[Sequence[int]], Optional[str]],
        batch_cache_key: Callable[..., str],
        batch_cache_getter: Callable[[], BatchSimulationCache],
        batch_store: object,
        batch_model: object,
        batch_initials_for_row: Callable[[int], Dict[str, float]],
        preview_session: object,
        preview_launch_pending: Callable[[], bool],
        mechanism_owner: object,
        solver_owner: object,
        update_batch_row_controls_state: Callable[[], None],
        sync_batch_species_columns: Callable[..., None],
        sync_mechanism_controls_to_focused_batch_set: Callable[..., None],
    ) -> None:
        self._batch_rows_for_scope = batch_rows_for_scope
        self._batch_set_ids_for_scope = batch_set_ids_for_scope
        self._requested_show_batch_set_ids = requested_show_batch_set_ids
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
        self._preview_launch_pending = preview_launch_pending
        self._mechanism_owner = mechanism_owner
        self._solver_owner = solver_owner
        self._update_batch_row_controls_state = update_batch_row_controls_state
        self._sync_batch_species_columns = sync_batch_species_columns
        self._sync_mechanism_controls_to_focused_batch_set = sync_mechanism_controls_to_focused_batch_set
        self._symbolic_wegscheider_identity_cache: Dict[tuple[str, str], Dict[str, Any]] = {}

    def batch_rows_for_scope(self, scope: str) -> List[int]:
        return [int(row) for row in (self._batch_rows_for_scope(str(scope)) or [])]

    def batch_set_ids_for_scope(self, scope: str) -> List[str]:
        return [str(set_id) for set_id in (self._batch_set_ids_for_scope(str(scope)) or [])]

    def requested_show_batch_set_ids(self) -> List[str]:
        return [str(set_id) for set_id in (self._requested_show_batch_set_ids() or [])]

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

    def active_cache_key(self) -> str:
        return str(self._batch_cache().active_cache_key or "")

    def active_cache_valid_set_ids(self) -> Optional[tuple[str, ...]]:
        value = self._batch_cache().active_cache_valid_set_ids
        if value is None:
            return None
        return tuple(str(set_id) for set_id in value if str(set_id))

    def active_cache_invalidated_set_ids(self) -> Optional[tuple[str, ...]]:
        value = self._batch_cache().active_cache_invalidated_set_ids
        if value is None:
            return None
        return tuple(str(set_id) for set_id in value if str(set_id))

    def clear_active_cache_identity_state(self) -> None:
        self._batch_cache().clear_active_cache_identity_state()

    def clear_active_preview_cache_identity_state(self) -> None:
        self._batch_cache().clear_active_preview_cache_identity_state()

    def apply_active_cache_preview_reconciliation(
        self,
        *,
        valid_set_ids: Sequence[str],
        invalidated_set_ids: Sequence[str],
        preview_scope_set_ids: Sequence[str],
        preview_token: str | None,
    ) -> None:
        self._batch_cache().apply_active_cache_preview_reconciliation(
            valid_set_ids=valid_set_ids,
            invalidated_set_ids=invalidated_set_ids,
            preview_scope_set_ids=preview_scope_set_ids,
            preview_token=preview_token,
        )

    def record_active_result_cache_staleness(
        self,
        *,
        set_ids: Sequence[str] = (),
        is_global: bool = False,
    ) -> tuple[str, ...]:
        return tuple(
            str(set_id)
            for set_id in self._batch_cache().record_active_result_cache_staleness(
                set_ids=set_ids,
                is_global=bool(is_global),
            )
            if str(set_id)
        )

    def reset_runtime_state(self) -> None:
        self._batch_cache().reset_runtime_state()

    def batch_species_column_sync_snapshot(self) -> BatchSpeciesColumnSyncSnapshot:
        batch_cache = self._batch_cache()
        active_preview_scope_ids = tuple(
            str(set_id)
            for set_id in (batch_cache.active_cache_preview_scope_set_ids or ())
            if str(set_id)
        )
        active_preview_token = str(batch_cache.active_cache_preview_token or "").strip()
        return BatchSpeciesColumnSyncSnapshot(
            active_cache_key=str(batch_cache.active_cache_key or "").strip(),
            active_preview_token=active_preview_token,
            active_preview_scope_ids=active_preview_scope_ids,
            active_valid_set_ids=tuple(
                str(set_id)
                for set_id in (batch_cache.active_cache_valid_set_ids or ())
                if str(set_id)
            ),
            scope_tokens_before=self._overlay_tokens_for_set_ids(active_preview_scope_ids),
        )

    def reconcile_active_cache_after_species_column_sync(
        self,
        snapshot: BatchSpeciesColumnSyncSnapshot,
        *,
        retain_active_cache_identity: bool,
    ) -> None:
        if not snapshot.active_cache_key:
            return
        if not bool(retain_active_cache_identity):
            self.clear_active_cache_identity_state()
            return
        if not snapshot.active_preview_token:
            return
        if self._active_scope_overlay_token(snapshot) == snapshot.active_preview_token:
            return
        scope_tokens_after = self._overlay_tokens_for_set_ids(snapshot.active_preview_scope_ids)
        invalidated_set_ids = {
            str(set_id)
            for set_id, before_token in snapshot.scope_tokens_before.items()
            if str(before_token) != str(scope_tokens_after.get(str(set_id), ""))
        }
        if not invalidated_set_ids:
            self.clear_active_cache_identity_state()
            return
        valid_ids = snapshot.active_valid_set_ids or snapshot.active_preview_scope_ids
        narrowed_valid_ids = tuple(str(set_id) for set_id in valid_ids if str(set_id) not in invalidated_set_ids)
        if not narrowed_valid_ids:
            self.clear_active_cache_identity_state()
            return
        narrowed_valid_set = set(narrowed_valid_ids)
        narrowed_scope_ids = tuple(
            str(set_id)
            for set_id in snapshot.active_preview_scope_ids
            if str(set_id) in narrowed_valid_set
        )
        self.apply_active_cache_preview_reconciliation(
            valid_set_ids=narrowed_valid_ids,
            invalidated_set_ids=tuple(str(set_id) for set_id in invalidated_set_ids if str(set_id)),
            preview_scope_set_ids=narrowed_scope_ids,
            preview_token=self._overlay_token_for_set_ids(narrowed_scope_ids),
        )

    def active_preview_cache_identity_matches_current_workspace(self) -> bool:
        batch_cache = self._batch_cache()
        active_preview_token = str(batch_cache.active_cache_preview_token or "").strip()
        if not active_preview_token:
            return False
        snapshot = BatchSpeciesColumnSyncSnapshot(
            active_cache_key=str(batch_cache.active_cache_key or "").strip(),
            active_preview_token=active_preview_token,
            active_preview_scope_ids=tuple(
                str(set_id)
                for set_id in (batch_cache.active_cache_preview_scope_set_ids or ())
                if str(set_id)
            ),
            active_valid_set_ids=(),
            scope_tokens_before={},
        )
        current_token = self._active_scope_overlay_token(snapshot)
        return bool(current_token) and str(current_token) == active_preview_token

    def _active_scope_overlay_token(self, snapshot: BatchSpeciesColumnSyncSnapshot) -> str:
        if not snapshot.active_preview_token:
            return ""
        if snapshot.active_preview_scope_ids:
            token = self._overlay_token_for_set_ids(snapshot.active_preview_scope_ids)
            return str(token or "")
        if not bool(self._preview_session.has_staged_concentration_overlays()):
            return ""
        try:
            row_count = int(getattr(self._batch_store, "row_count")())
        except Exception:
            row_count = 0
        if row_count <= 0:
            return ""
        try:
            return str(self._preview_session.preview_batch_cache_token(list(range(int(row_count)))) or "")
        except Exception:
            return ""

    def _overlay_tokens_for_set_ids(self, set_ids: Sequence[str]) -> dict[str, str]:
        tokens: dict[str, str] = {}
        for set_id in set_ids or ():
            row = self._batch_row_for_set_id(str(set_id))
            if row is None:
                tokens[str(set_id)] = ""
                continue
            try:
                tokens[str(set_id)] = str(self._preview_session.preview_batch_cache_token([int(row)]) or "")
            except Exception:
                tokens[str(set_id)] = ""
        return tokens

    def _overlay_token_for_set_ids(self, set_ids: Sequence[str]) -> Optional[str]:
        scope_rows: list[int] = []
        for set_id in set_ids or ():
            row = self._batch_row_for_set_id(str(set_id))
            if row is not None:
                scope_rows.append(int(row))
        if not scope_rows:
            return None
        try:
            token = str(self._preview_session.preview_batch_cache_token(scope_rows) or "")
        except Exception:
            return None
        return token or None

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

    def active_result_cache_read_snapshot(
        self,
        *,
        cache_key: str | None = None,
    ) -> BatchCacheResultReadSnapshot:
        return self._batch_cache().result_cache_read_snapshot(cache_key=cache_key)

    def batch_cache_contains_set(self, *, set_id: str, set_name: str) -> bool:
        return bool(self._batch_cache().contains_set_identifier(set_id=str(set_id), set_name=str(set_name)))

    def purge_batch_cache_for_deleted_sets(
        self,
        *,
        set_ids: Sequence[str],
        set_names: Sequence[str],
    ) -> int:
        return int(
            self._batch_cache().purge_entries_for_set_identifiers(
                set_ids=tuple(str(set_id) for set_id in set_ids),
                set_names=tuple(str(name) for name in set_names),
            )
        )

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

    def focused_batch_set_is_dirty(self) -> bool:
        focused_set_id = str(self.focused_batch_set_id() or "").strip()
        return bool(focused_set_id and self._preview_has_dirty_state_for_set(focused_set_id))

    def focused_show_request_is_dirty(
        self,
        *,
        requested_show_set_ids: Sequence[str],
        prefer_set: Optional[str] = None,
    ) -> bool:
        focused_set_id = str(
            prefer_set or (requested_show_set_ids[0] if requested_show_set_ids else "") or ""
        ).strip()
        if not focused_set_id:
            return False
        return self._preview_has_dirty_state_for_set(focused_set_id)

    def show_request_uses_fresh_explicit_cache_after_post_run_sync(
        self,
        *,
        requested_show_set_ids: Sequence[str],
    ) -> bool:
        normalized_requested_show_set_ids = [
            str(set_id) for set_id in (requested_show_set_ids or ()) if str(set_id)
        ]
        if not normalized_requested_show_set_ids:
            return False
        batch_cache = self._batch_cache()
        active_cache_key = str(batch_cache.active_cache_key or "").strip()
        active_preview_token = str(batch_cache.active_cache_preview_token or "").strip()
        if not active_cache_key or not active_preview_token:
            return False
        active_valid_set_ids = {
            str(set_id) for set_id in (batch_cache.active_cache_valid_set_ids or ()) if str(set_id)
        }
        if active_valid_set_ids and any(
            set_id not in active_valid_set_ids for set_id in normalized_requested_show_set_ids
        ):
            return False
        active_preview_scope_ids = {
            str(set_id)
            for set_id in (batch_cache.active_cache_preview_scope_set_ids or ())
            if str(set_id)
        }
        if active_preview_scope_ids and any(
            set_id not in active_preview_scope_ids for set_id in normalized_requested_show_set_ids
        ):
            return False
        scope_rows: list[int] = []
        row_for_set_id = getattr(self._batch_store, "row_for_set_id", None)
        if not callable(row_for_set_id):
            return False
        for set_id in normalized_requested_show_set_ids:
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

    def _resolve_workspace_aware_display_request(
        self,
        *,
        requested_show_set_ids: Sequence[str],
        preview_cache_key: Optional[str] = None,
    ) -> BatchDisplayRequestResolution:
        batch_cache = self._batch_cache()
        active_cache_key = str(batch_cache.active_cache_key or "").strip()
        active_valid_set_ids = {
            str(set_id) for set_id in (batch_cache.active_cache_valid_set_ids or ()) if str(set_id)
        }
        invalidated_set_ids = {
            str(set_id) for set_id in (batch_cache.active_cache_invalidated_set_ids or ()) if str(set_id)
        }
        focused_set_id = str(self._focused_batch_set_id() or "").strip()
        if (not focused_set_id) and requested_show_set_ids:
            focused_set_id = str(requested_show_set_ids[0] or "").strip()

        resolved_entries: list[ResolvedBatchDisplayRequestEntry] = []
        has_workspace_display_request = False
        has_resolved_workspace_preview = False
        focused_request_uses_workspace_controls = False
        focused_request_has_resolved_entry = False
        missing_workspace_entry = False
        missing_explicit_entry = False
        invalid_entry = False

        for raw_set_id in requested_show_set_ids or ():
            set_id = str(raw_set_id or "").strip()
            if not set_id:
                continue
            label = self.batch_set_name_for_id(set_id) or set_id
            if self._preview_has_dirty_state_for_set(set_id):
                has_workspace_display_request = True
                preview_entry = self.matching_preview_entry_for_workspace_set(
                    set_id=set_id,
                    preview_cache_key=preview_cache_key,
                )
                if preview_entry.entry is not None:
                    has_resolved_workspace_preview = True
                    canonical_entry = None
                    if (
                        active_cache_key
                        and set_id not in invalidated_set_ids
                        and (not active_valid_set_ids or set_id in active_valid_set_ids)
                    ):
                        explicit_entry = batch_cache.entry_for_set(
                            cache_key=active_cache_key,
                            set_id=set_id,
                            is_preview=False,
                            require_completion_provenance=True,
                        )
                        canonical_entry = explicit_entry.entry
                    resolved_entries.append(
                        ResolvedBatchDisplayRequestEntry(
                            set_id=str(set_id),
                            label=str(label),
                            entry=preview_entry.entry,
                            canonical_entry=canonical_entry,
                            workspace_preview_provenance=self.workspace_preview_display_provenance_for_entry(
                                set_id,
                                preview_entry.entry,
                            ),
                        )
                    )
                    if set_id == focused_set_id:
                        focused_request_uses_workspace_controls = True
                        focused_request_has_resolved_entry = True
                elif preview_entry.state == "invalid":
                    invalid_entry = True
                else:
                    missing_workspace_entry = True
                continue

            if not active_cache_key:
                missing_explicit_entry = True
                continue
            if active_valid_set_ids and set_id not in active_valid_set_ids:
                missing_explicit_entry = True
                continue
            explicit_entry = batch_cache.entry_for_set(
                cache_key=active_cache_key,
                set_id=set_id,
                is_preview=False,
                require_completion_provenance=True,
            )
            if set_id in invalidated_set_ids:
                if explicit_entry.state == "invalid":
                    invalid_entry = True
                else:
                    missing_explicit_entry = True
                continue
            if explicit_entry.entry is not None:
                resolved_entries.append(
                    ResolvedBatchDisplayRequestEntry(set_id=str(set_id), label=str(label), entry=explicit_entry.entry)
                )
                if set_id == focused_set_id:
                    focused_request_uses_workspace_controls = False
                    focused_request_has_resolved_entry = True
            elif explicit_entry.state == "invalid":
                invalid_entry = True
            else:
                missing_explicit_entry = True

        coverage = (
            BatchDisplayRequestCoverage.FULL
            if len(resolved_entries)
            == len(
                [str(set_id) for set_id in (requested_show_set_ids or ()) if str(set_id)]
            )
            else BatchDisplayRequestCoverage.INCOMPLETE
        )
        unavailable_cause: DisplayTransitionCause | None = None
        if invalid_entry:
            unavailable_cause = DisplayTransitionCause.INVALID_CACHE_ENTRY
        elif missing_workspace_entry:
            unavailable_cause = DisplayTransitionCause.IN_FLIGHT_COVERAGE_UNAVAILABLE
        elif missing_explicit_entry:
            unavailable_cause = DisplayTransitionCause.CACHE_RESULT_UNAVAILABLE
        return BatchDisplayRequestResolution(
            resolved_entries=tuple(resolved_entries),
            unavailable_cause=unavailable_cause,
            coverage=coverage,
            has_workspace_display_request=bool(has_workspace_display_request),
            has_resolved_workspace_preview=bool(has_resolved_workspace_preview),
            focused_uses_workspace_controls=bool(focused_request_uses_workspace_controls),
            focused_has_resolved_entry=bool(focused_request_has_resolved_entry),
        )

    def workspace_display_request_resolution(
        self,
        requested_show_set_ids: Sequence[str],
    ) -> BatchDisplayRequestResolution:
        return self._resolve_workspace_aware_display_request(requested_show_set_ids=requested_show_set_ids)

    def matching_preview_entry_for_workspace_set(
        self,
        *,
        set_id: str,
        preview_cache_key: Optional[str] = None,
    ) -> BatchCacheEntryReadResult:
        expected_mechanism_text = self._mechanism_text_for_workspace_selection(set_id=str(set_id))
        batch_cache = self._batch_cache()
        resolved_preview_cache_key = str(
            preview_cache_key
            if preview_cache_key is not None
            else (batch_cache.active_preview_cache_key or "")
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
            has_semantic_identity = entry_identity is not None
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
            if entry_t.size <= 0:
                return False
            if has_semantic_identity:
                return True
            expected_grid_n = int((expected_solver_config.get("grid") or {}).get("N") or 0)
            if expected_grid_n > 0 and int(entry_t.size) != expected_grid_n:
                return False
            return math.isclose(float(entry_t[-1]), float(expected_t_end), rel_tol=1e-9, abs_tol=1e-12)

        invalid_found = False
        direct = batch_cache.entry_for_set(
            cache_key=resolved_preview_cache_key,
            set_id=str(set_id),
            is_preview=True,
        )
        if _entry_matches_expected(direct):
            return direct
        if direct.state == "invalid":
            invalid_found = True

        for result in batch_cache.preview_entry_results_for_set_id(
            set_id=str(set_id),
            exclude_cache_key=resolved_preview_cache_key,
        ):
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

    def workspace_preview_display_provenance_for_entry(
        self,
        set_id: str,
        entry: Mapping[str, Any],
    ) -> Optional[Dict[str, Any]]:
        sid = str(set_id or "").strip()
        if not sid or not isinstance(entry, Mapping):
            return None
        if self._preview_has_local_mechanism_workspace(sid):
            preview_entry = self.matching_preview_entry_for_workspace_set(set_id=sid)
            if self._entry_matches_reference_payload(preview_entry.entry, entry):
                return self.current_workspace_preview_identity_payload(set_id=sid)
            return None
        row = self._batch_row_for_set_id(sid)
        if row is None:
            return None
        try:
            has_preview_token = bool(self._preview_session.preview_batch_cache_token([int(row)]))
        except Exception:
            has_preview_token = False
        if not has_preview_token:
            return None
        preview_entry = self.matching_preview_entry_for_workspace_set(set_id=sid)
        if self._entry_matches_reference_payload(preview_entry.entry, entry):
            return self.current_workspace_preview_identity_payload(set_id=sid)
        return None

    def _symbolic_jacobian_identity_for_preview(
        self,
        *,
        set_id: str,
        mechanism_text: str,
        solver_config: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        solver_name = str(dict(solver_config or {}).get("solver") or "").strip().lower()
        if solver_name not in {"bdf", "radau"}:
            return {}
        if not bool(dict(solver_config or {}).get("use_sparse_jacobian", False)):
            return {}
        if not bool(self._mechanism_owner.has_slider_overrides()):
            return {}
        try:
            mechanism_identity_text = strip_reaction_dsl_initial_concentrations(
                str(mechanism_text or "")
            )
            parameter_overrides = self._normalized_slider_overrides(set_id=str(set_id))
            from kindred.core.simulation_preparation import (
                symbolic_jacobian_identity_for_execution_text,
            )

            payload = symbolic_jacobian_identity_for_execution_text(
                mechanism_text=mechanism_identity_text,
                solver_config=dict(solver_config or {}),
                parameter_overrides=parameter_overrides,
            )
            if not payload:
                return {}
            return dict(payload)
        except Exception:
            return {}

    def _symbolic_wegscheider_identity_for_preview(
        self,
        *,
        mechanism_text: str,
        solver_config: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        if not bool(dict(solver_config or {}).get("wegscheider_cyclicity_enabled", True)):
            return {}
        if not bool(self._mechanism_owner.has_slider_overrides()):
            return {}
        try:
            mechanism_identity_text = strip_reaction_dsl_initial_concentrations(
                str(mechanism_text or "")
            )
            solver_identity = repr(
                {
                    "temperature_K": dict(solver_config or {}).get("temperature_K"),
                    "wegscheider_cyclicity_enabled": bool(
                        dict(solver_config or {}).get("wegscheider_cyclicity_enabled", True)
                    ),
                }
            )
            cache_key = (mechanism_identity_text, solver_identity)
            cached = self._symbolic_wegscheider_identity_cache.get(cache_key)
            if cached is not None:
                return dict(cached)
            from kindred.core.simulation_preparation import (
                symbolic_wegscheider_identity_for_execution_text,
            )

            payload = symbolic_wegscheider_identity_for_execution_text(
                mechanism_text=mechanism_identity_text,
                solver_config=dict(solver_config or {}),
            )
            if not payload:
                return {}
            self._symbolic_wegscheider_identity_cache[cache_key] = dict(payload)
            return dict(payload)
        except Exception:
            return {}

    def current_workspace_preview_identity(self, *, set_id: str) -> SimulationIdentity:
        mechanism_text = self._mechanism_text_for_workspace_selection(set_id=str(set_id))
        symbolic_mechanism_text = self._mechanism_text_for_workspace_selection(
            set_id=str(set_id),
            apply_parameter_overrides=False,
        )
        expected_solver_config, expected_t_end, expected_overlay_token = self._current_workspace_preview_context(
            set_id=str(set_id),
            mechanism_text=mechanism_text,
        )
        from kindred.core.intervention_schedule import normalized_intervention_schedule_identity_fingerprints_from_dsl_text

        (
            intervention_schedule_declarative_fingerprint,
            intervention_schedule_executable_fingerprint,
        ) = normalized_intervention_schedule_identity_fingerprints_from_dsl_text(
            str(mechanism_text or "")
        )
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
            intervention_schedule_declarative_fingerprint=intervention_schedule_declarative_fingerprint,
            intervention_schedule_executable_fingerprint=intervention_schedule_executable_fingerprint,
            preview_batch_cache_token=expected_overlay_token,
            execution_flags=("fast_mode",),
            symbolic_jacobian_identity=self._symbolic_jacobian_identity_for_preview(
                set_id=str(set_id),
                mechanism_text=symbolic_mechanism_text,
                solver_config=expected_solver_config,
            ),
            symbolic_wegscheider_identity=self._symbolic_wegscheider_identity_for_preview(
                mechanism_text=mechanism_text,
                solver_config=expected_solver_config,
            ),
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

    def _entry_matches_reference_payload(
        self,
        expected_entry: Optional[MutableMapping[str, Any]],
        reference_entry: Mapping[str, Any],
    ) -> bool:
        if not isinstance(reference_entry, Mapping):
            return False
        reference_t = np.asarray(
            reference_entry.get("t") if reference_entry.get("t") is not None else [],
            dtype=float,
        ).reshape(-1)
        reference_series = reference_entry.get("series") or {}
        if not isinstance(reference_series, MutableMapping):
            reference_series = dict(reference_series) if isinstance(reference_series, Mapping) else {}
        return self._batch_cache_entry_matches_plot_payload(
            entry=expected_entry,
            t=reference_t,
            series=reference_series,
        )

    def update_batch_row_controls_state(self) -> None:
        self._update_batch_row_controls_state()

    def sync_batch_species_columns(
        self,
        species_names: Sequence[str],
        *,
        retain_active_cache_identity: bool = False,
    ) -> None:
        self._sync_batch_species_columns(
            [str(species) for species in species_names],
            retain_active_cache_identity=bool(retain_active_cache_identity),
        )

    def _batch_cache(self) -> BatchSimulationCache:
        return self._batch_cache_getter()

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

    def _mechanism_text_for_workspace_selection(
        self,
        *,
        set_id: str,
        apply_parameter_overrides: bool = True,
    ) -> str:
        reactions_text = self._mechanism_owner.mechanism_reactions_text_raw()
        if self._mechanism_owner.has_slider_overrides() and bool(apply_parameter_overrides):
            reactions_text = self._mechanism_owner.apply_overrides_to_text(reactions_text, set_id=str(set_id))
        reactions_text = strip_reaction_dsl_initial_concentrations(reactions_text)

        state_network_dsl = self._mechanism_owner.mechanism_state_network_dsl_raw()
        if self._mechanism_owner.has_slider_overrides() and bool(apply_parameter_overrides):
            state_network_dsl = self._mechanism_owner.apply_overrides_to_state_network_dsl(
                state_network_dsl,
                set_id=str(set_id),
            )

        full_dsl = reactions_text
        if state_network_dsl.strip():
            full_dsl += "\n\n# State Network\n" + state_network_dsl
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
