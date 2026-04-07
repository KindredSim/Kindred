from __future__ import annotations

import json
import logging
import math
from typing import TYPE_CHECKING, Dict, Optional, Sequence

from PySide6 import QtCore

from kindred.core.document_parameter_store import DocumentParameterStore

if TYPE_CHECKING:
    from kindred.gui.main_window import MainWindow

logger = logging.getLogger(__name__)


class MainWindowPreviewSession:
    """Owns MainWindow's slider/species preview gesture state and debounce timers.

    Parameter storage (shared baseline and per-set overrides) is delegated to
    a :class:`DocumentParameterStore` instance.  This class retains ownership
    of gesture state, debounce timers, and concentration overlays.
    """

    def __init__(
        self,
        main_window: "MainWindow",
        *,
        param_store: Optional[DocumentParameterStore] = None,
    ) -> None:
        self._mw = main_window
        self._param_store: DocumentParameterStore = param_store or DocumentParameterStore()
        self._slider_drag_active = False
        self._slider_triggered_simulation = False
        self._last_slider_change_name = ""
        self._pending_slider_values: dict[str, float] = {}
        self._staged_concentration_overlays_by_set_id: dict[str, dict[str, float]] = {}
        self._dirty_state_generation_by_set_id: dict[str, int] = {}
        self._slider_gesture_target_set_ids_snapshot: list[str] = []
        self._drag_baseline_text: Optional[str] = None
        self._drag_baseline_state_network_dsl: Optional[str] = None
        self._suppress_slider_refresh = False
        self._slider_release_in_progress = False
        self._slider_release_primary_name = ""
        self._variable_update_timer = None
        self._species_slider_update_timer = None
        self._slider_release_commit_timer = None

    @property
    def param_store(self) -> DocumentParameterStore:
        """The canonical parameter store backing this session."""
        return self._param_store

    def drag_baseline_text(self) -> Optional[str]:
        return self._drag_baseline_text

    def drag_baseline_state_network_dsl(self) -> Optional[str]:
        return self._drag_baseline_state_network_dsl

    def clear_pending_slider_values(self) -> None:
        self._pending_slider_values.clear()

    def _refresh_transaction_button_state(self) -> None:
        mw = self._mw
        try:
            refresh = getattr(mw, "_refresh_slider_transaction_button_state", None)
            if callable(refresh):
                refresh()
        except Exception:
            logger.debug("Failed to refresh slider transaction button state", exc_info=True)

    def _clear_active_preview_cache_state(self) -> None:
        mw = self._mw
        try:
            controller = getattr(mw, "_sim_controller", None)
            batch_cache = getattr(controller, "batch_cache", None)
            if batch_cache is not None:
                clear_preview = getattr(batch_cache, "clear_active_preview_selection_state", None)
                if callable(clear_preview):
                    clear_preview()
                else:
                    batch_cache.active_preview_cache_key = None
                    if hasattr(batch_cache, "active_preview_scope_set_ids"):
                        batch_cache.active_preview_scope_set_ids = None
        except Exception:
            logger.debug("Failed to clear active preview cache state", exc_info=True)

    def _focused_mechanism_workspace_set_id(self) -> str:
        mw = self._mw
        try:
            focused_set_id = mw.focused_batch_set_id()
        except Exception:
            focused_set_id = None
        if focused_set_id:
            return str(focused_set_id)
        try:
            row_count = int(mw.batch_store_row_count())
        except Exception:
            row_count = 0
        if row_count > 0:
            try:
                set_id = mw.batch_set_id_for_row(0)
            except Exception:
                set_id = None
            if set_id:
                return str(set_id)
        return ""

    def focused_mechanism_workspace_set_id(self) -> str:
        return str(self._focused_mechanism_workspace_set_id() or "")

    def _selected_mechanism_target_set_ids(self) -> list[str]:
        return self.effective_slider_edit_target_set_ids()

    def effective_slider_edit_target_set_ids(self) -> list[str]:
        mw = self._mw
        try:
            helper = getattr(mw, "_effective_slider_edit_target_set_ids")
        except Exception:
            helper = None
        try:
            selected = helper() if callable(helper) else []
        except Exception:
            selected = []
        normalized: list[str] = []
        seen: set[str] = set()
        for set_id in selected or []:
            set_id_s = str(set_id or "").strip()
            if not set_id_s or set_id_s in seen:
                continue
            seen.add(set_id_s)
            normalized.append(set_id_s)
        if normalized:
            return normalized
        focused_set_id = self._focused_mechanism_workspace_set_id()
        return [focused_set_id] if focused_set_id else []

    def _workspace_for_set_id(self, set_id: str) -> dict[str, float]:
        return self._param_store.local_overrides_for_set(str(set_id or ""))

    def local_mechanism_workspace(self, set_id: str) -> dict[str, float]:
        return self._workspace_for_set_id(str(set_id or ""))

    def local_mechanism_workspace_set_ids(self) -> list[str]:
        return self._param_store.set_ids_with_local_overrides()

    def has_dirty_state_for_set(self, set_id: str) -> bool:
        set_id_s = str(set_id or "").strip()
        if not set_id_s:
            return False
        return bool(self.has_local_mechanism_workspace(set_id_s)) or bool(
            self._staged_concentration_overlays_by_set_id.get(set_id_s)
        )

    def dirty_state_generation(self, set_id: str) -> int:
        return int(self._dirty_state_generation_by_set_id.get(str(set_id or "").strip(), 0) or 0)

    def _bump_dirty_state_generation(self, set_ids: Sequence[str]) -> None:
        for set_id in set_ids or ():
            set_id_s = str(set_id or "").strip()
            if not set_id_s:
                continue
            current = int(self._dirty_state_generation_by_set_id.get(set_id_s, 0) or 0)
            self._dirty_state_generation_by_set_id[set_id_s] = current + 1

    def has_local_mechanism_workspaces(self) -> bool:
        return self._param_store.has_any_local_overrides()

    def has_local_mechanism_workspace(self, set_id: str) -> bool:
        return self._param_store.has_local_overrides_for_set(str(set_id or ""))

    def clear_local_mechanism_workspace(self, set_id: str) -> bool:
        set_id_s = str(set_id or "").strip()
        removed = self._param_store.clear_local_overrides_for_set(set_id_s)
        if not removed:
            return False
        self._bump_dirty_state_generation([set_id_s])
        self._clear_active_preview_cache_state()
        self._refresh_transaction_button_state()
        return True

    def clear_all_local_mechanism_workspaces(self) -> None:
        if not self._param_store.has_any_local_overrides():
            self._refresh_transaction_button_state()
            return
        cleared_set_ids = self.local_mechanism_workspace_set_ids()
        self._param_store.clear_all_local_overrides()
        self._bump_dirty_state_generation(cleared_set_ids)
        self._clear_active_preview_cache_state()
        self._refresh_transaction_button_state()

    def reset_current_mechanism_workspace(self) -> bool:
        set_id = self._focused_mechanism_workspace_set_id()
        if not set_id:
            return False
        removed = self._param_store.clear_local_overrides_for_set(set_id)
        if not removed:
            return False
        self._bump_dirty_state_generation([set_id])
        self.reset_preview_state()
        self._refresh_transaction_button_state()
        return True

    def reset_mechanism_workspaces(self, set_ids: Sequence[str]) -> bool:
        changed = False
        seen: set[str] = set()
        cleared_set_ids: list[str] = []
        for set_id in set_ids or ():
            set_id_s = str(set_id or "").strip()
            if not set_id_s or set_id_s in seen:
                continue
            seen.add(set_id_s)
            if self._param_store.clear_local_overrides_for_set(set_id_s):
                changed = True
                cleared_set_ids.append(set_id_s)
        if not changed:
            return False
        self._bump_dirty_state_generation(cleared_set_ids)
        self.reset_preview_state()
        self._refresh_transaction_button_state()
        return True

    def commit_current_mechanism_workspace(self) -> dict[str, float]:
        focused_set_id = self._focused_mechanism_workspace_set_id()
        result = self._param_store.commit_effective_as_shared(
            focused_set_id
        )
        if focused_set_id:
            self._bump_dirty_state_generation([focused_set_id])
        self.reset_preview_state()
        self._refresh_transaction_button_state()
        return result

    def slider_gesture_target_set_ids_snapshot(self) -> list[str]:
        return [str(set_id) for set_id in self._slider_gesture_target_set_ids_snapshot]

    def _capture_slider_gesture_target_snapshot(self) -> list[str]:
        snapshot = self._selected_mechanism_target_set_ids()
        self._slider_gesture_target_set_ids_snapshot = [str(set_id) for set_id in snapshot]
        return self.slider_gesture_target_set_ids_snapshot()

    def _clear_slider_gesture_target_snapshot(self) -> None:
        self._slider_gesture_target_set_ids_snapshot.clear()

    def _ensure_slider_gesture_target_snapshot(self) -> list[str]:
        if self._slider_drag_active or self._slider_release_in_progress:
            if not self._slider_gesture_target_set_ids_snapshot:
                return self._capture_slider_gesture_target_snapshot()
            return self.slider_gesture_target_set_ids_snapshot()
        return self._capture_slider_gesture_target_snapshot()

    def sync_committed_slider_values(self, values: Dict[str, float]) -> None:
        overrides_changed = self._param_store.sync_shared_params(values or {})
        if overrides_changed:
            self._clear_active_preview_cache_state()
        self._refresh_transaction_button_state()

    def clear_staged_slider_values(self) -> None:
        self.clear_all_local_mechanism_workspaces()

    def slider_overrides(self, set_id: Optional[str] = None) -> dict[str, float]:
        target_set_id = str(set_id or "").strip() or self._focused_mechanism_workspace_set_id()
        return self._workspace_for_set_id(target_set_id)

    def stage_slider_value(self, name: str, value: float, *, target_set_ids: Optional[Sequence[str]] = None) -> None:
        name_s = str(name)
        value_f = float(value)
        candidate_targets = target_set_ids
        if candidate_targets is None:
            if (
                (self._slider_drag_active or self._slider_release_in_progress)
                and self._slider_gesture_target_set_ids_snapshot
            ):
                candidate_targets = self.slider_gesture_target_set_ids_snapshot()
            else:
                focused_set_id = self._focused_mechanism_workspace_set_id()
                candidate_targets = [focused_set_id] if focused_set_id else []
        changed = False
        seen: set[str] = set()
        changed_set_ids: list[str] = []
        for set_id in candidate_targets or []:
            set_id_s = str(set_id or "").strip()
            if not set_id_s or set_id_s in seen:
                continue
            seen.add(set_id_s)
            if self._param_store.stage_override(set_id_s, name_s, value_f):
                changed = True
                changed_set_ids.append(set_id_s)
        if changed:
            self._bump_dirty_state_generation(changed_set_ids)
            self._clear_active_preview_cache_state()
        self._refresh_transaction_button_state()

    def effective_slider_values(self, set_id: Optional[str] = None) -> dict[str, float]:
        target_set_id = str(set_id or "").strip() or self._focused_mechanism_workspace_set_id()
        return self._param_store.effective_params(target_set_id)

    def effective_slider_values_for_set(self, set_id: str) -> dict[str, float]:
        return self.effective_slider_values(set_id=str(set_id or ""))

    def preview_initials_for_row(self, row: int, baseline: Dict[str, float]) -> dict[str, float]:
        merged = {str(key): float(value) for key, value in dict(baseline or {}).items()}
        overlay = self._staged_concentration_overlay_for_row(int(row))
        if overlay:
            merged.update(overlay)
        return merged

    def preview_batch_cache_token(self, rows: Sequence[int]) -> str:
        payload: list[tuple[str, tuple[tuple[str, float], ...]]] = []
        for row in rows or []:
            set_id = self._set_id_for_row(int(row))
            if not set_id:
                continue
            overlay = self._staged_concentration_overlays_by_set_id.get(set_id) or {}
            if not overlay:
                continue
            payload.append(
                (
                    str(set_id),
                    tuple(sorted((str(species), float(value)) for species, value in overlay.items())),
                )
            )
        if not payload:
            return ""
        return json.dumps(payload, separators=(",", ":"))

    def stage_concentration_value_for_rows(self, rows: Sequence[int], *, species: str, value: float) -> bool:
        species_s = str(species)
        value_f = max(0.0, float(value))
        changed = False
        changed_set_ids: list[str] = []
        for row in rows or []:
            set_id = self._set_id_for_row(int(row))
            if not set_id:
                continue
            baseline = self._committed_concentration_value(int(row), species_s)
            overlay = dict(self._staged_concentration_overlays_by_set_id.get(set_id) or {})
            row_changed = False
            if baseline is not None and math.isclose(value_f, float(baseline), rel_tol=1e-12, abs_tol=1e-12):
                if species_s in overlay:
                    overlay.pop(species_s, None)
                    changed = True
                    row_changed = True
            else:
                if overlay.get(species_s) != value_f:
                    overlay[species_s] = value_f
                    changed = True
                    row_changed = True
            if overlay:
                self._staged_concentration_overlays_by_set_id[set_id] = overlay
            else:
                self._staged_concentration_overlays_by_set_id.pop(set_id, None)
            if row_changed and set_id not in changed_set_ids:
                changed_set_ids.append(str(set_id))
        if changed:
            self._bump_dirty_state_generation(changed_set_ids)
            self._refresh_transaction_button_state()
        return bool(changed)

    def clear_staged_concentration_overlays(self) -> None:
        cleared_set_ids = list(self._staged_concentration_overlays_by_set_id.keys())
        self._staged_concentration_overlays_by_set_id.clear()
        self._bump_dirty_state_generation(cleared_set_ids)
        self._refresh_transaction_button_state()

    def discard_concentration_overlays_for_rows(self, rows: Sequence[int]) -> bool:
        target_set_ids: list[str] = []
        for row in rows or []:
            set_id = self._set_id_for_row(int(row))
            if set_id:
                target_set_ids.append(str(set_id))
        return self.discard_concentration_overlays_for_set_ids(target_set_ids)

    def discard_concentration_overlays_for_set_ids(self, set_ids: Sequence[str]) -> bool:
        changed = False
        seen: set[str] = set()
        cleared_set_ids: list[str] = []
        for set_id in set_ids or []:
            set_id_s = str(set_id or "").strip()
            if not set_id_s or set_id_s in seen:
                continue
            seen.add(set_id_s)
            if set_id_s in self._staged_concentration_overlays_by_set_id:
                self._staged_concentration_overlays_by_set_id.pop(set_id_s, None)
                changed = True
                cleared_set_ids.append(set_id_s)
        if changed:
            self._bump_dirty_state_generation(cleared_set_ids)
            self._refresh_transaction_button_state()
        return bool(changed)

    def prune_staged_concentration_overlays_to_species(self, species_names: Sequence[str]) -> bool:
        allowed_species = {str(species) for species in (species_names or []) if str(species)}
        changed = False
        pruned_overlays: dict[str, dict[str, float]] = {}
        for set_id, overlay in list(self._staged_concentration_overlays_by_set_id.items()):
            filtered = {
                str(species): float(value)
                for species, value in dict(overlay or {}).items()
                if str(species) in allowed_species
            }
            if filtered != dict(overlay or {}):
                changed = True
                self._bump_dirty_state_generation([str(set_id)])
            if filtered:
                pruned_overlays[str(set_id)] = filtered
        if not changed:
            return False
        self._staged_concentration_overlays_by_set_id = pruned_overlays
        self._refresh_transaction_button_state()
        return True

    def has_staged_concentration_overlays(self) -> bool:
        return any(bool(values) for values in self._staged_concentration_overlays_by_set_id.values())

    def has_dirty_transaction(self) -> bool:
        return bool(self.has_local_mechanism_workspaces()) or bool(self.has_staged_concentration_overlays())

    def apply_staged_concentration_overlays(self, model: object) -> list[int]:
        if model is None or not self._staged_concentration_overlays_by_set_id:
            return []
        touched_rows: list[int] = []
        touched_set_ids: list[str] = []
        try:
            store = model.store()
            species_list = list(store.visible_species())
        except Exception:
            return []
        for row in range(int(model.rowCount())):
            set_id = self._set_id_for_row(int(row))
            overlay = self._staged_concentration_overlays_by_set_id.get(set_id or "")
            if not overlay:
                continue
            if set_id:
                touched_set_ids.append(str(set_id))
            touched_rows.append(int(row))
            for species, value in overlay.items():
                try:
                    col = 1 + species_list.index(str(species))
                except ValueError:
                    continue
                model_index = model.index(int(row), int(col))
                if not model_index.isValid():
                    continue
                model.setData(model_index, f"{float(value):.6g}")
        self._staged_concentration_overlays_by_set_id.clear()
        self._bump_dirty_state_generation(touched_set_ids)
        self._refresh_transaction_button_state()
        return touched_rows

    def clear_working_transaction(self, *, clear_committed_slider_values: bool = False) -> None:
        changed_set_ids = self.local_mechanism_workspace_set_ids() + list(
            self._staged_concentration_overlays_by_set_id.keys()
        )
        self._param_store.clear_all_local_overrides()
        self._staged_concentration_overlays_by_set_id.clear()
        self._bump_dirty_state_generation(changed_set_ids)
        if clear_committed_slider_values:
            self._param_store.clear_shared_params()
        self.reset_preview_state()
        self._refresh_transaction_button_state()

    def stop_slider_release_commit_timer(self) -> None:
        timer = self._slider_release_commit_timer
        if timer is not None:
            timer.stop()

    def has_pending_slider_values(self) -> bool:
        return bool(self._pending_slider_values)

    def finalize_slider_release_commit(self) -> None:
        self._finalize_slider_release_commit()

    def stop_variable_update_timer(self) -> None:
        timer = self._variable_update_timer
        if timer is not None:
            timer.stop()

    def stop_species_slider_update_timer(self) -> None:
        timer = self._species_slider_update_timer
        if timer is not None:
            try:
                timer.stop()
            except Exception as exc:
                logger.debug("Failed to stop species slider update timer: %s", exc, exc_info=True)

    def set_slider_triggered_simulation(self, value: bool) -> None:
        self._slider_triggered_simulation = bool(value)

    def _queue_pending_slider_preview_replay(
        self,
        *,
        set_ids: Sequence[str],
        request_id: Optional[int] = None,
        preserve_existing_request: bool = False,
    ) -> None:
        controller = getattr(self._mw, "_sim_controller", None)
        if controller is None:
            return
        queue_pending = getattr(controller, "queue_pending_slider_preview_replay", None)
        if callable(queue_pending):
            queue_pending(
                target_set_ids=set_ids,
                request_id=request_id,
                preserve_existing_request=bool(preserve_existing_request),
            )
            return
        run_state = getattr(controller, "run_state", None)
        if run_state is None:
            return
        normalized: list[str] = []
        seen: set[str] = set()
        for set_id in set_ids or ():
            set_id_s = str(set_id or "").strip()
            if not set_id_s or set_id_s in seen:
                continue
            seen.add(set_id_s)
            normalized.append(set_id_s)
        if request_id is not None:
            run_state.pending_slider_sim_request_id = int(request_id)
        elif not bool(preserve_existing_request):
            run_state.pending_slider_sim_request_id = None
        run_state.pending_slider_target_set_ids = tuple(normalized)
        run_state.pending_slider_simulation = True

    def slider_triggered_simulation(self) -> bool:
        return bool(self._slider_triggered_simulation)

    def last_slider_change_name(self) -> str:
        return str(self._last_slider_change_name or "")

    def slider_drag_active(self) -> bool:
        return bool(self._slider_drag_active)

    def suppress_slider_refresh(self) -> bool:
        return bool(self._suppress_slider_refresh)

    def reset_preview_state(self) -> None:
        self._pending_slider_values.clear()
        self._slider_triggered_simulation = False
        self._slider_drag_active = False
        self._clear_slider_gesture_target_snapshot()
        self._drag_baseline_text = None
        self._drag_baseline_state_network_dsl = None
        self._suppress_slider_refresh = False
        self._slider_release_in_progress = False
        self._slider_release_primary_name = ""
        self._clear_active_preview_cache_state()

        for timer_attr in (
            "_variable_update_timer",
            "_species_slider_update_timer",
            "_slider_release_commit_timer",
        ):
            timer = getattr(self, timer_attr, None)
            try:
                if timer is not None and timer.isActive():
                    timer.stop()
            except RuntimeError as exc:
                logger.debug("Timer %s was invalid while resetting overrides: %s", timer_attr, exc, exc_info=True)
                setattr(self, timer_attr, None)

    def deactivate_species_preview_timer(self) -> None:
        timer = self._species_slider_update_timer
        try:
            if timer is not None and timer.isActive():
                timer.stop()
        except RuntimeError as exc:
            logger.debug("Species slider timer was invalid while disabling: %s", exc, exc_info=True)
            self._species_slider_update_timer = None

    def _read_preview_delay_setting(self, key: str, *, default: int) -> int:
        mw = self._mw
        settings = getattr(mw, "_settings", None)
        raw_value = default
        if settings is not None and hasattr(settings, "value"):
            try:
                raw_value = settings.value(str(key), int(default))
            except Exception:
                raw_value = default
        try:
            parsed = int(raw_value)
        except Exception:
            parsed = int(default)
        return max(0, min(1000, int(parsed)))

    def variable_preview_debounce_ms(self, name: str) -> int:
        name_s = str(name or "")
        if name_s.startswith("Keq") and name_s[3:].isdigit():
            return self._read_preview_delay_setting(
                "simulation/equilibrium_preview_debounce_ms",
                default=150,
            )
        return self._read_preview_delay_setting(
            "simulation/parameter_preview_debounce_ms",
            default=80,
        )

    def on_variable_changed(self, name: str, value: float) -> None:
        """Handle variable slider change and queue a fast preview simulation."""
        mw = self._mw
        logger.debug("Variable %s changed to %s", name, value)
        self._last_slider_change_name = name
        target_set_ids = self._ensure_slider_gesture_target_snapshot()
        controller = getattr(mw, "_sim_controller", None)
        run_state = getattr(controller, "run_state", None)
        if controller is not None:
            next_slider_preview_request_id = getattr(controller, "next_slider_preview_request_id", None)
            if callable(next_slider_preview_request_id):
                request_id = int(next_slider_preview_request_id())
            else:
                request_id = int(mw._sim_controller.next_sim_request_id())
        else:
            request_id = int(mw._sim_controller.next_sim_request_id())
        if run_state is not None:
            self._queue_pending_slider_preview_replay(
                set_ids=target_set_ids,
                request_id=int(request_id),
            )
        self.stage_slider_value(name, float(value), target_set_ids=target_set_ids)
        if self._slider_drag_active or self._slider_release_in_progress:
            self._pending_slider_values[name] = value
        mw._refresh_derived_parameters_display()
        meta = (mw.variable_metadata() or {}).get(name, {})
        if isinstance(meta, dict) and meta.get("type") == "energy":
            mw._refresh_energy_mode_derived_parameter_table()
        else:
            mw._update_parameter_table_from_sliders()

        if self._slider_release_in_progress:
            timer = self._slider_release_commit_timer
            if timer is not None:
                timer.stop()
                timer.start()
            self._queue_pending_slider_preview_replay(
                set_ids=target_set_ids,
                preserve_existing_request=True,
            )
            return

        self._slider_triggered_simulation = True
        interval_ms = self.variable_preview_debounce_ms(name)
        if self._slider_drag_active:
            mw._status_label.setText(f"Adjusting {name} = {value:.3g}")
        else:
            mw._status_label.setText(f"Previewing {name} = {value:.3g}")
        timer = self._ensure_variable_update_timer(interval_ms=interval_ms)
        self._queue_pending_slider_preview_replay(
            set_ids=target_set_ids,
            preserve_existing_request=True,
        )
        timer.stop()
        timer.setInterval(interval_ms)
        timer.start()

    def commit_slider_value(self, name: str, value: float) -> None:
        """Stage a programmatic slider change for preview runs without mutating editor text."""
        mw = self._mw
        self._last_slider_change_name = name
        target_set_ids = self._ensure_slider_gesture_target_snapshot()
        self._queue_pending_slider_preview_replay(
            set_ids=target_set_ids,
            preserve_existing_request=True,
        )
        self.stage_slider_value(name, float(value), target_set_ids=target_set_ids)
        meta = (mw.variable_metadata() or {}).get(name, {})
        self._slider_triggered_simulation = True
        mw._status_label.setText(f"Updating {name} = {value:.3g}...")
        if isinstance(meta, dict) and meta.get("type") == "energy":
            mw._refresh_energy_mode_derived_parameter_table()
        else:
            mw._update_parameter_table_from_sliders()

        interval_ms = self.variable_preview_debounce_ms(name)
        timer = self._ensure_variable_update_timer(interval_ms=interval_ms)
        self._queue_pending_slider_preview_replay(set_ids=target_set_ids)
        timer.stop()
        timer.setInterval(interval_ms)
        timer.start()

    def on_slider_drag_started(self, name: str) -> None:
        """Mark that a slider drag gesture has begun."""
        mw = self._mw
        self._slider_drag_active = True
        self._capture_slider_gesture_target_snapshot()
        self._drag_baseline_text = mw._mechanism_editor._reactions_text.toPlainText()
        self._drag_baseline_state_network_dsl = mw._mechanism_editor._state_network_editor.get_state_network_dsl()
        self._suppress_slider_refresh = True
        mw._mechanism_editor._variable_sliders.begin_live_drag()

    def on_slider_drag_finished(self, name: str) -> None:
        """Commit all pending slider values when drag completes."""
        mw = self._mw
        self._slider_drag_active = False
        self._drag_baseline_text = None
        self._drag_baseline_state_network_dsl = None
        mw._mechanism_editor._variable_sliders.end_live_drag()
        timer = self._variable_update_timer
        if timer is not None and timer.isActive():
            timer.stop()
        if not self._pending_slider_values:
            self._clear_slider_gesture_target_snapshot()
            self._suppress_slider_refresh = False
            return

        self._slider_release_in_progress = True
        self._slider_release_primary_name = str(name)
        self._queue_pending_slider_preview_replay(
            set_ids=self.slider_gesture_target_set_ids_snapshot(),
            preserve_existing_request=True,
        )

        timer = self._ensure_slider_release_commit_timer()
        interval_ms = self.variable_preview_debounce_ms(name)
        timer.setInterval(interval_ms)
        timer.stop()
        timer.start()

    def queue_species_slider_simulation(self, *, label: str, delay_ms: int) -> None:
        """Queue a fast preview run for species-mode slider edits."""
        mw = self._mw
        try:
            delay_ms_i = int(delay_ms)
        except Exception:
            delay_ms_i = 80
        delay_ms_i = max(0, min(500, delay_ms_i))

        self._last_slider_change_name = str(label or "init")
        request_id = mw._sim_controller.next_sim_request_id()
        self._queue_pending_slider_preview_replay(
            set_ids=self._selected_mechanism_target_set_ids(),
            request_id=int(request_id),
        )

        timer = self._ensure_species_slider_update_timer()
        try:
            timer.stop()
        except RuntimeError as exc:
            logger.debug("Species slider timer was invalid; recreating: %s", exc, exc_info=True)
            self._species_slider_update_timer = self._create_species_slider_update_timer()
            timer = self._species_slider_update_timer
        timer.setInterval(int(delay_ms_i))
        timer.start()

    def _finalize_slider_release_commit(self) -> None:
        """Finalize a drag gesture by running a single fast preview simulation."""
        pending = dict(self._pending_slider_values or {})
        target_set_ids = self.slider_gesture_target_set_ids_snapshot()
        self._pending_slider_values.clear()
        self._suppress_slider_refresh = False
        self._slider_release_in_progress = False
        self._slider_release_primary_name = ""
        self._clear_slider_gesture_target_snapshot()

        if not pending:
            return

        self._queue_pending_slider_preview_replay(
            set_ids=target_set_ids,
            preserve_existing_request=True,
        )
        timer = self._ensure_variable_update_timer()
        timer.stop()
        timer.setInterval(0)
        timer.start()

    def _ensure_variable_update_timer(self, *, interval_ms: Optional[int] = None):
        if self._variable_update_timer is None:
            self._variable_update_timer = self._create_variable_update_timer()
        if interval_ms is not None:
            self._variable_update_timer.setInterval(int(interval_ms))
        return self._variable_update_timer

    def _ensure_species_slider_update_timer(self):
        if self._species_slider_update_timer is None:
            self._species_slider_update_timer = self._create_species_slider_update_timer()
        return self._species_slider_update_timer

    def _ensure_slider_release_commit_timer(self):
        if self._slider_release_commit_timer is None:
            self._slider_release_commit_timer = self._create_slider_release_commit_timer()
        return self._slider_release_commit_timer

    def _create_variable_update_timer(self):
        timer = QtCore.QTimer(self._mw)
        timer.setSingleShot(True)
        timer.timeout.connect(self._mw._sim_controller.run_simulation_from_slider)
        return timer

    def _create_species_slider_update_timer(self):
        timer = QtCore.QTimer(self._mw)
        timer.setSingleShot(True)
        timer.timeout.connect(self._mw._sim_controller.run_simulation_from_slider)
        return timer

    def _create_slider_release_commit_timer(self):
        timer = QtCore.QTimer(self._mw)
        timer.setSingleShot(True)
        timer.timeout.connect(self._finalize_slider_release_commit)
        return timer

    def _staged_concentration_overlay_for_row(self, row: int) -> dict[str, float]:
        set_id = self._set_id_for_row(int(row))
        if not set_id:
            return {}
        return {
            str(species): float(value)
            for species, value in (self._staged_concentration_overlays_by_set_id.get(set_id) or {}).items()
        }

    def _committed_concentration_value(self, row: int, species: str) -> Optional[float]:
        mw = self._mw
        try:
            raw = mw._batch_store.get_value(int(row), str(species))
            value = float(raw)
        except Exception:
            return None
        if not math.isfinite(value):
            return None
        return float(value)

    def _set_id_for_row(self, row: int) -> str:
        mw = self._mw
        try:
            set_id = mw._batch_set_id_for_row(int(row))
        except Exception:
            set_id = None
        return str(set_id or "")
