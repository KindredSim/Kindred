from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

import numpy as np

from kindred.core.batch_initial_conditions import (
    resolve_run_scope,
)
from kindred.core.batch_simulation_cache import BatchSimulationCache
from kindred.core.mechanism_source import MechanismAuthoringSource
from kindred.core.simulation_identity import (
    SimulationIdentity,
    canonical_initials_fingerprint,
    coerce_simulation_identity,
)
from kindred.core.batch_cache_contracts import (
    BatchCacheResultReadSnapshot,
    BatchCacheEntryReadResult,
)
from kindred.gui.display_authority import (
    compose_resolved_display_request_entry,
    resolve_canonical_reference_authority,
)
from kindred.gui.ports import (
    BatchDisplayRequestCoverage,
    BatchDisplayRequestResolution,
    ConcentrationSetInteractionTransaction,
    DisplayTransitionCause,
    RequestScopeRestoreTruth,
    ResolvedBatchDisplayRequestEntry,
)


@dataclass(frozen=True, slots=True)
class BatchSpeciesColumnSyncSnapshot:
    active_cache_key: str
    active_preview_token: str
    active_preview_scope_ids: tuple[str, ...]
    active_valid_set_ids: tuple[str, ...]
    scope_tokens_before: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class BatchSelectionStateSnapshot:
    selected_set_ids: tuple[str, ...]
    focused_set_id: str = ""


@dataclass(frozen=True, slots=True)
class BatchSelectionStateResolution:
    selected_rows: tuple[int, ...]
    focused_row: Optional[int] = None
    focused_set_id: str = ""


@dataclass(frozen=True, slots=True)
class RunTargetUiState:
    target_rows: tuple[int, ...]
    target_set_ids: tuple[str, ...]
    button_text: str
    enabled: bool
    empty_reason: str = ""


class SimulationBatchOwner:
    """Thin Qt adapter for batch table/store, cache identity, and display-request resolution."""

    def __init__(
        self,
        *,
        batch_selected_rows: Callable[[], Sequence[int]],
        requested_show_batch_set_ids: Callable[[], Sequence[str]],
        slider_edit_target_set_ids: Callable[[], Sequence[str]],
        batch_current_row: Callable[[], Optional[int]],
        batch_set_id_for_row: Callable[[int], Optional[str]],
        batch_set_name_for_id: Callable[[str], Optional[str]],
        batch_set_id_for_name: Callable[[str], Optional[str]],
        batch_preferred_primary_set_id: Callable[[Sequence[int]], Optional[str]],
        batch_cache_key: Callable[..., str],
        batch_cache_getter: Callable[[], BatchSimulationCache],
        batch_store_getter: Callable[[], object],
        batch_model_getter: Callable[[], object],
        batch_initials_for_row: Callable[[int], Dict[str, float]],
        preview_session: object,
        preview_launch_pending: Callable[[], bool],
        mechanism_owner: object,
        solver_owner: object,
        update_batch_row_controls_state: Callable[[], None],
        sync_batch_species_columns: Callable[..., None],
        sync_mechanism_controls_to_focused_batch_set: Callable[..., None],
        active_batch_set_id: Callable[[], Optional[str]] | None = None,
    ) -> None:
        if active_batch_set_id is None:
            def active_batch_set_id() -> Optional[str]:
                return None
        self._batch_selected_rows = batch_selected_rows
        self._requested_show_batch_set_ids = requested_show_batch_set_ids
        self._slider_edit_target_set_ids = slider_edit_target_set_ids
        self._active_batch_set_id = active_batch_set_id
        self._batch_current_row = batch_current_row
        self._batch_set_id_for_row = batch_set_id_for_row
        self._batch_set_name_for_id = batch_set_name_for_id
        self._batch_set_id_for_name = batch_set_id_for_name
        self._batch_preferred_primary_set_id = batch_preferred_primary_set_id
        self._batch_cache_key = batch_cache_key
        self._batch_cache_getter = batch_cache_getter
        self._batch_store_getter = batch_store_getter
        self._batch_model_getter = batch_model_getter
        self._batch_initials_for_row = batch_initials_for_row
        self._preview_session = preview_session
        self._preview_launch_pending = preview_launch_pending
        self._mechanism_owner = mechanism_owner
        self._solver_owner = solver_owner
        self._update_batch_row_controls_state = update_batch_row_controls_state
        self._sync_batch_species_columns = sync_batch_species_columns
        self._sync_mechanism_controls_to_focused_batch_set = sync_mechanism_controls_to_focused_batch_set

    @property
    def _batch_store(self) -> object:
        store = self._batch_store_getter()
        if store is None:
            raise RuntimeError("Batch initial conditions store is unavailable.")
        return store

    @property
    def _batch_model(self) -> object:
        model = self._batch_model_getter()
        if model is None:
            raise RuntimeError("Batch initial conditions model is unavailable.")
        return model

    def batch_rows_for_scope(self, scope: str) -> List[int]:
        total = int(self._batch_store.row_count())
        if total <= 0:
            return []
        scope_key = str(scope or "").strip().lower()
        if scope_key == "all":
            return list(range(total))
        if scope_key in {"run", "selected-else-active", "selected_else_active"}:
            return [int(row) for row in self.run_target_ui_state().target_rows]
        return resolve_run_scope(
            selected_rows=self._batch_selected_rows(),
            total_rows=total,
            mode="selected",
        )

    def batch_set_ids_for_scope(self, scope: str) -> List[str]:
        set_ids: List[str] = []
        for row in self.batch_rows_for_scope(scope):
            set_id = self.batch_set_id_for_row(int(row))
            if set_id:
                set_ids.append(str(set_id))
        return set_ids

    def batch_selection_state_snapshot(self) -> BatchSelectionStateSnapshot:
        return BatchSelectionStateSnapshot(
            selected_set_ids=tuple(self.batch_set_ids_for_scope("selected")),
            focused_set_id=str(self.focused_batch_set_id() or "").strip(),
        )

    def resolve_batch_selection_state_snapshot(
        self,
        snapshot: BatchSelectionStateSnapshot,
    ) -> BatchSelectionStateResolution:
        selected_rows: list[int] = []
        seen_rows: set[int] = set()
        for set_id in tuple(getattr(snapshot, "selected_set_ids", ()) or ()):
            row = self._batch_row_for_set_id(str(set_id))
            if row is None:
                continue
            row_i = int(row)
            if row_i in seen_rows:
                continue
            seen_rows.add(row_i)
            selected_rows.append(row_i)
        focused_set_id = str(getattr(snapshot, "focused_set_id", "") or "").strip()
        focused_row = self._batch_row_for_set_id(focused_set_id) if focused_set_id else None
        if focused_row is None and selected_rows:
            focused_row = int(selected_rows[0])
            focused_set_id = str(self.batch_set_id_for_row(int(focused_row)) or "").strip()
        return BatchSelectionStateResolution(
            selected_rows=tuple(selected_rows),
            focused_row=int(focused_row) if focused_row is not None else None,
            focused_set_id=focused_set_id,
        )

    def requested_show_batch_set_ids(self) -> List[str]:
        return [str(set_id) for set_id in (self._requested_show_batch_set_ids() or [])]

    def effective_display_request_set_ids(self, *, focused_row: Optional[int] = None) -> List[str]:
        _ = focused_row
        return self._normalized_valid_set_ids(self.requested_show_batch_set_ids())

    def slider_edit_target_set_ids(self) -> List[str]:
        return [str(set_id) for set_id in (self._slider_edit_target_set_ids() or [])]

    def _normalized_valid_set_ids(self, set_ids: Sequence[str]) -> List[str]:
        valid_ids = {str(set_id) for set_id in self._batch_store.set_ids()}
        normalized: List[str] = []
        for raw_set_id in set_ids or ():
            set_id = str(raw_set_id or "").strip()
            if not set_id or set_id not in valid_ids or set_id in normalized:
                continue
            normalized.append(set_id)
        return normalized

    def _set_id_for_current_row(self) -> Optional[str]:
        row = self.batch_current_row()
        if row is None:
            return None
        return self.batch_set_id_for_row(int(row))

    def effective_slider_edit_target_set_ids(self, *, focused_row: Optional[int] = None) -> List[str]:
        explicit_ids = self._normalized_valid_set_ids(self.slider_edit_target_set_ids())
        if explicit_ids:
            return explicit_ids
        if focused_row is not None:
            set_id = self.batch_set_id_for_row(int(focused_row))
        else:
            set_id = self.focused_batch_set_id()
        return [str(set_id)] if set_id else []

    def focused_effective_slider_target_set_id(self) -> str:
        target_ids = self.effective_slider_edit_target_set_ids()
        return str(target_ids[0]) if target_ids else ""

    def run_target_ui_state(self) -> RunTargetUiState:
        total = int(self._batch_store.row_count())
        if total <= 0:
            return RunTargetUiState(
                target_rows=(),
                target_set_ids=(),
                button_text="Run",
                enabled=False,
                empty_reason="Add at least one set before running.",
            )

        selected_rows = tuple(int(row) for row in self.batch_rows_for_scope("selected"))
        if selected_rows:
            target_rows = selected_rows
            button_text = (
                "Run Selected Set"
                if len(target_rows) == 1
                else f"Run Selected ({len(target_rows)})"
            )
        else:
            active_set_id = str(self.focused_batch_set_id() or "").strip()
            active_row = self._batch_row_for_set_id(active_set_id) if active_set_id else None
            target_rows = (int(active_row),) if active_row is not None else ()
            button_text = "Run Active Set" if target_rows else "Run"

        target_set_ids = tuple(
            str(set_id)
            for set_id in (
                self.batch_set_id_for_row(int(row))
                for row in target_rows
            )
            if str(set_id)
        )
        enabled = bool(target_rows and target_set_ids)
        empty_reason = "" if enabled else "Choose an active set or select one or more sets to run."
        return RunTargetUiState(
            target_rows=tuple(int(row) for row in target_rows if 0 <= int(row) < total),
            target_set_ids=target_set_ids,
            button_text=button_text,
            enabled=enabled,
            empty_reason=empty_reason,
        )

    def run_selected_empty_target_reason(self) -> str:
        return self.run_target_ui_state().empty_reason

    def selected_run_target_rows(self) -> List[int]:
        return [int(row) for row in self.run_target_ui_state().target_rows]

    def concentration_set_interaction_transaction(
        self,
        *,
        gesture: str,
        row: Optional[int] = None,
    ) -> ConcentrationSetInteractionTransaction:
        gesture_key = str(gesture or "").strip().lower()
        row_i: Optional[int]
        try:
            row_i = int(row) if row is not None else self.batch_current_row()
        except (TypeError, ValueError):
            row_i = None
        total = int(self._batch_store.row_count())
        if row_i is not None and not (0 <= int(row_i) < total):
            row_i = None
        set_id = self.batch_set_id_for_row(int(row_i)) if row_i is not None else None
        set_id_s = str(set_id or "")

        focused_set_id = str(self.focused_batch_set_id() or "").strip()
        selected_rows = tuple(int(row) for row in self.batch_rows_for_scope("selected"))
        target_selection_rows = selected_rows
        if gesture_key in {"row_body_click", "initial_default_selection"} and row_i is not None:
            target_selection_rows = (int(row_i),)

        requested_show_ids = tuple(self._normalized_valid_set_ids(self.requested_show_batch_set_ids()))
        explicit_slider_ids = tuple(self._normalized_valid_set_ids(self.slider_edit_target_set_ids()))
        effective_slider_ids = tuple(self.effective_slider_edit_target_set_ids(focused_row=row_i))
        effective_slider_rows = tuple(
            int(row)
            for row in (
                self._batch_row_for_set_id(str(set_id))
                for set_id in effective_slider_ids
            )
            if row is not None
        )

        focus_change = bool(
            gesture_key in {"row_body_click", "initial_default_selection"}
            and set_id_s
            and focused_set_id != set_id_s
        )
        selection_change = bool(
            gesture_key == "selection_change"
            or tuple(selected_rows) != tuple(target_selection_rows)
        )
        display_refresh_needed = bool(gesture_key in {"show_checkbox", "show_membership_change"})

        slider_rebuild_needed = gesture_key in {"slider_checkbox", "slider_target_change"}
        run_rows = self.run_target_ui_state().target_rows
        empty_reason = self.run_target_ui_state().empty_reason if not run_rows else ""

        return ConcentrationSetInteractionTransaction(
            gesture=gesture_key,
            row=row_i,
            set_id=set_id_s,
            focus_change=focus_change,
            selection_change=selection_change,
            requested_show_set_ids=requested_show_ids,
            explicit_slider_target_set_ids=explicit_slider_ids,
            effective_slider_edit_target_set_ids=effective_slider_ids,
            effective_slider_edit_target_rows=effective_slider_rows,
            run_selected_rows=run_rows,
            empty_run_target_reason=empty_reason,
            display_refresh_needed=display_refresh_needed,
            display_refresh_reason=gesture_key if display_refresh_needed else "",
            slider_rebuild_needed=slider_rebuild_needed,
            slider_rebuild_reason=gesture_key if slider_rebuild_needed else "",
        )

    def focused_batch_set_id(self) -> Optional[str]:
        value = self._active_batch_set_id()
        return str(value) if value else None

    def preview_initials_for_row(self, row: int, baseline: Dict[str, float]) -> Dict[str, float]:
        preview = getattr(self._preview_session, "preview_initials_for_row", None)
        if callable(preview):
            try:
                return dict(preview(int(row), dict(baseline or {})))
            except Exception:
                return dict(baseline or {})
        return dict(baseline or {})

    def stage_concentration_value_for_rows(self, rows: Sequence[int], *, species: str, value: float) -> bool:
        stage = getattr(self._preview_session, "stage_concentration_value_for_rows", None)
        if callable(stage):
            try:
                return bool(stage(rows, species=str(species), value=float(value)))
            except Exception:
                return False
        return False

    def discard_concentration_overlays_for_rows(self, rows: Sequence[int]) -> bool:
        discard = getattr(self._preview_session, "discard_concentration_overlays_for_rows", None)
        if callable(discard):
            try:
                return bool(discard(rows))
            except Exception:
                return False
        return False

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
        _ = requested_show_set_ids
        focused_set_id = str(prefer_set or "").strip()
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
        focused_set_id = str(self.focused_batch_set_id() or "").strip()

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
                    workspace_preview_provenance = self.workspace_preview_display_provenance_for_entry(
                        set_id,
                        preview_entry.entry,
                    )
                    reference_authority = resolve_canonical_reference_authority(
                        set_id=set_id,
                        active_cache_key=active_cache_key,
                        active_cache_valid_set_ids=active_valid_set_ids,
                        active_cache_invalidated_set_ids=invalidated_set_ids,
                        workspace_preview_provenance=workspace_preview_provenance,
                        load_canonical_reference_candidate=lambda: batch_cache.entry_for_set(
                            cache_key=active_cache_key,
                            set_id=set_id,
                            is_preview=False,
                            require_completion_provenance=True,
                        ).entry,
                    )
                    resolved_entries.append(
                        compose_resolved_display_request_entry(
                            set_id=str(set_id),
                            label=str(label),
                            active_display_payload=preview_entry.entry,
                            canonical_reference_candidate=(
                                reference_authority.canonical_reference_candidate
                            ),
                            canonical_reference_eligible_for_current_inputs=(
                                reference_authority.canonical_reference_eligible_for_current_inputs
                            ),
                            invalidation_context=reference_authority.invalidation_context,
                            request_scope_restore_truth=RequestScopeRestoreTruth.WORKSPACE_PREVIEW,
                            workspace_preview_provenance=workspace_preview_provenance,
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
            if explicit_entry.entry is not None:
                explicit_authority = resolve_canonical_reference_authority(
                    set_id=set_id,
                    active_cache_key=active_cache_key,
                    active_cache_valid_set_ids=active_valid_set_ids,
                    active_cache_invalidated_set_ids=invalidated_set_ids,
                    workspace_preview_provenance={"restore_truth": "explicit_cache"},
                    load_canonical_reference_candidate=lambda: explicit_entry.entry,
                )
                resolved_entries.append(
                    compose_resolved_display_request_entry(
                        set_id=str(set_id),
                        label=str(label),
                        active_display_payload=explicit_entry.entry,
                        canonical_reference_candidate=(
                            explicit_authority.canonical_reference_candidate
                        ),
                        canonical_reference_eligible_for_current_inputs=(
                            explicit_authority.canonical_reference_eligible_for_current_inputs
                        ),
                        invalidation_context=explicit_authority.invalidation_context,
                        request_scope_restore_truth=RequestScopeRestoreTruth.EXPLICIT_CACHE,
                    )
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
        batch_cache = self._batch_cache()
        resolved_preview_cache_key = str(
            preview_cache_key
            if preview_cache_key is not None
            else (batch_cache.active_preview_cache_key or "")
        ).strip()

        try:
            expected_source = self._mechanism_source_for_workspace_selection(set_id=str(set_id))
            expected_mechanism_text = expected_source.full_dsl
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

    def current_workspace_preview_identity(self, *, set_id: str) -> SimulationIdentity:
        mechanism_source = self._mechanism_source_for_workspace_selection(set_id=str(set_id))
        mechanism_text = mechanism_source.full_dsl
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
                set_name = self._batch_set_name_for_id(str(set_id))
                pending_initials = self._mechanism_owner.pending_initials_for_run_source_set(
                    self._mechanism_owner.mechanism_source_for_run(fast_mode=True),
                    set_name=str(set_name or set_id),
                )
                baseline_initials = {**baseline_initials, **dict(pending_initials or {})}
                preview_initials = self._preview_session.preview_initials_for_row(int(row), baseline_initials)
                initials_fingerprint = canonical_initials_fingerprint(preview_initials)
            except Exception:
                initials_fingerprint = ""
        return SimulationIdentity.build(
            schema_id=self._mechanism_owner.simulation_schema_id(fast_mode=True),
            param_fingerprint=self._mechanism_owner.runtime_parameter_fingerprint_for_set(
                set_id=str(set_id),
                fast_mode=True,
            ),
            canonical_initials_fingerprint=initials_fingerprint,
            solver_config=expected_solver_config,
            t_end=expected_t_end,
            intervention_schedule_declarative_fingerprint=intervention_schedule_declarative_fingerprint,
            intervention_schedule_executable_fingerprint=intervention_schedule_executable_fingerprint,
            preview_batch_cache_token=expected_overlay_token,
            execution_flags=("fast_mode",),
            symbolic_jacobian_identity=self._workspace_symbolic_jacobian_identity(
                set_id=str(set_id),
                solver_config=expected_solver_config,
            ),
            symbolic_wegscheider_identity=self._workspace_symbolic_wegscheider_identity(
                set_id=str(set_id),
                mechanism_text=mechanism_text,
                solver_config=expected_solver_config,
            ),
        )

    def _workspace_symbolic_jacobian_identity(
        self,
        *,
        set_id: str,
        solver_config: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        solver_name = str(dict(solver_config or {}).get("solver") or "").strip().lower()
        if solver_name not in {"bdf", "radau"}:
            return {}
        if not bool(dict(solver_config or {}).get("use_sparse_jacobian", False)):
            return {}
        try:
            mechanism_text = self._mechanism_source_for_workspace_selection(
                set_id=str(set_id),
                apply_parameter_overrides=False,
            ).full_dsl
            parameter_values = getattr(self._mechanism_owner, "runtime_parameter_values_for_set", None)
            parameter_overrides = (
                dict(parameter_values(set_id=str(set_id)))
                if callable(parameter_values)
                else {}
            )
            from kindred.core.simulation_preparation import (
                symbolic_jacobian_identity_for_execution_text,
            )

            payload = symbolic_jacobian_identity_for_execution_text(
                mechanism_text=str(mechanism_text or ""),
                solver_config=dict(solver_config or {}),
                parameter_overrides=parameter_overrides,
            )
            return dict(payload or {})
        except Exception:
            return {}

    def _workspace_symbolic_wegscheider_identity(
        self,
        *,
        set_id: str,
        mechanism_text: str,
        solver_config: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        if not bool(dict(solver_config or {}).get("wegscheider_cyclicity_enabled", True)):
            return {}
        try:
            from kindred.core.simulation_preparation import (
                symbolic_wegscheider_identity_for_execution_text,
            )

            payload = symbolic_wegscheider_identity_for_execution_text(
                mechanism_text=str(mechanism_text or ""),
                solver_config=dict(solver_config or {}),
            )
            return dict(payload or {})
        except Exception:
            return {}

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

    def _mechanism_source_for_workspace_selection(
        self,
        *,
        set_id: str,
        apply_parameter_overrides: bool = True,
    ) -> MechanismAuthoringSource:
        return self._mechanism_owner.mechanism_source_for_run_set(
            self._mechanism_owner.mechanism_source_for_run(fast_mode=True),
            set_id=str(set_id),
            apply_parameter_overrides=bool(apply_parameter_overrides),
            strip_initial_concentrations=True,
        )

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
