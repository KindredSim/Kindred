from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import math
from typing import TYPE_CHECKING, Dict, Optional, Sequence

from PySide6 import QtCore

from kindred.core.document_parameter_store import DocumentParameterStore
from kindred.gui.ports import SliderPreviewLifecyclePort, SliderReplayIntent

if TYPE_CHECKING:
    from kindred.gui.main_window import MainWindow

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConcentrationOverlayCommitResult:
    touched_rows: tuple[int, ...] = ()
    touched_set_ids: tuple[str, ...] = ()


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
        self._current_slider_replay_intent: Optional[SliderReplayIntent] = None
        self._last_submitted_slider_replay_intent: Optional[SliderReplayIntent] = None
        self._slider_preview_lifecycle_port: Optional[SliderPreviewLifecyclePort] = None

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

    def set_slider_preview_lifecycle_port(self, port: SliderPreviewLifecyclePort) -> None:
        self._slider_preview_lifecycle_port = port

    def _require_slider_preview_lifecycle_port(self) -> SliderPreviewLifecyclePort:
        port = self._slider_preview_lifecycle_port
        if port is None:
            raise RuntimeError("Slider preview lifecycle port is not bound.")
        return port

    def _deauthorize_completed_run_display_for_slider_preview_scope(
        self,
        target_set_ids: Sequence[str],
    ) -> bool:
        port = self._slider_preview_lifecycle_port
        if port is None:
            return False
        deauthorize = getattr(port, "deauthorize_completed_run_display_for_slider_preview_scope", None)
        if not callable(deauthorize):
            return False
        try:
            return bool(deauthorize(target_set_ids))
        except Exception:
            logger.debug("Failed to deauthorize completed-run display for slider preview scope", exc_info=True)
            return False

    def _refresh_transaction_button_state(self) -> None:
        mw = self._mw
        try:
            refresh = getattr(mw, "_refresh_slider_transaction_button_state", None)
            if callable(refresh):
                refresh()
        except Exception:
            logger.debug("Failed to refresh slider transaction button state", exc_info=True)

    def is_mechanism_valid_for_preview(self) -> bool:
        return bool(self._mw.is_mechanism_valid_for_preview())

    def _clear_active_preview_cache_state(self) -> None:
        mw = self._mw
        try:
            controller = getattr(mw, "_sim_controller", None)
            batch_cache = getattr(controller, "batch_cache", None)
            if batch_cache is not None:
                clear_preview = getattr(batch_cache, "clear_active_preview_cache_identity_state", None)
                if callable(clear_preview):
                    clear_preview()
                else:
                    batch_cache.active_preview_cache_key = None
                    if hasattr(batch_cache, "active_preview_scope_set_ids"):
                        batch_cache.active_preview_scope_set_ids = None
        except Exception:
            logger.debug("Failed to clear active preview cache state", exc_info=True)

    def _show_invalid_preview_state(self) -> None:
        mw = self._mw
        self._clear_active_preview_cache_state()
        mw._refresh_batch_display_from_request_scope()
        reason = ""
        variable_runtime = getattr(mw, "_variable_runtime", None)
        reason_getter = getattr(variable_runtime, "slider_runtime_unavailable_reason", None)
        if callable(reason_getter):
            try:
                reason = str(reason_getter() or "")
            except Exception:
                reason = ""
        if reason == "unresolved Wegscheider cyclicity":
            mw._status_label.setText("Unresolved Wegscheider cyclicity.")
        else:
            mw._status_label.setText("Mechanism invalid — no preview available.")

    def show_preview_unavailable_for_dirty_state(self, message: str) -> None:
        mw = self._mw
        self._clear_active_preview_cache_state()
        mw._refresh_batch_display_from_request_scope()
        mw._status_label.setText(str(message or "Preview unavailable. Adjust sliders or run again."))

    def _focused_mechanism_workspace_set_id(self) -> str:
        target_ids = self._main_window_effective_slider_target_set_ids()
        return str(target_ids[0]) if target_ids else ""

    def _main_window_effective_slider_target_set_ids(self) -> list[str]:
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
        return normalized

    def focused_mechanism_workspace_set_id(self) -> str:
        return str(self._focused_mechanism_workspace_set_id() or "")

    def _selected_mechanism_target_set_ids(self) -> list[str]:
        return self.effective_slider_edit_target_set_ids()

    def effective_slider_edit_target_set_ids(self) -> list[str]:
        return self._main_window_effective_slider_target_set_ids()

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
        self._require_slider_preview_lifecycle_port().invalidate_slider_preview_work()
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
        self._prune_targeted_preview_state_for_reset(cleared_set_ids)
        self._refresh_transaction_button_state()
        return True

    def commit_current_mechanism_workspace(self, *, invalidate_preview_work: bool = True) -> dict[str, float]:
        focused_set_id = self._focused_mechanism_workspace_set_id()
        result = self._param_store.commit_effective_as_shared(
            focused_set_id
        )
        if focused_set_id:
            self._bump_dirty_state_generation([focused_set_id])
        self.reset_preview_state()
        if invalidate_preview_work:
            self._require_slider_preview_lifecycle_port().invalidate_slider_preview_work()
        self._refresh_transaction_button_state()
        return result

    def slider_gesture_target_set_ids_snapshot(self) -> list[str]:
        return [str(set_id) for set_id in self._slider_gesture_target_set_ids_snapshot]

    def current_slider_replay_intent(self) -> Optional[SliderReplayIntent]:
        intent = self._current_slider_replay_intent
        return intent if isinstance(intent, SliderReplayIntent) else None

    def _stop_preview_timer(self, timer_attr: str) -> None:
        timer = getattr(self, timer_attr, None)
        try:
            if timer is not None and timer.isActive():
                timer.stop()
        except RuntimeError as exc:
            logger.debug("Timer %s was invalid while pruning preview state: %s", timer_attr, exc, exc_info=True)
            setattr(self, timer_attr, None)

    def _pruned_slider_replay_intent_for_reset(
        self,
        intent: Optional[SliderReplayIntent],
        *,
        cleared_set_ids: set[str],
    ) -> Optional[SliderReplayIntent]:
        if not isinstance(intent, SliderReplayIntent):
            return None
        current_target_ids = tuple(str(set_id) for set_id in intent.target_set_ids if str(set_id))
        if not current_target_ids:
            return None
        if intent.source == "species_slider":
            truthful_target_ids = set(self._staged_concentration_overlay_target_set_ids())
            surviving_target_ids = tuple(
                set_id for set_id in current_target_ids if set_id in truthful_target_ids
            )
        elif intent.source == "reset":
            truthful_target_ids = set(self._dirty_slider_replay_target_set_ids_for_reset())
            surviving_target_ids = tuple(
                set_id for set_id in current_target_ids if set_id in truthful_target_ids
            )
        else:
            surviving_target_ids = tuple(
                set_id for set_id in current_target_ids if set_id not in cleared_set_ids
            )
        return self.build_slider_replay_intent(
            set_ids=surviving_target_ids,
            source=intent.source,
        )

    def _surviving_slider_replay_intents(self) -> tuple[SliderReplayIntent, ...]:
        return tuple(
            intent
            for intent in (
                self._current_slider_replay_intent,
                self._last_submitted_slider_replay_intent,
            )
            if isinstance(intent, SliderReplayIntent)
        )

    def _apply_surviving_slider_preview_scope(
        self,
        *,
        surviving_gesture_target_ids: Sequence[str],
    ) -> None:
        self._slider_gesture_target_set_ids_snapshot = [
            str(set_id)
            for set_id in surviving_gesture_target_ids
            if str(set_id)
        ]
        has_surviving_gesture_scope = bool(self._slider_gesture_target_set_ids_snapshot)
        surviving_replay_intents = self._surviving_slider_replay_intents()
        has_surviving_replay_scope = bool(surviving_replay_intents)

        if not has_surviving_gesture_scope:
            self._pending_slider_values.clear()
            self._slider_drag_active = False
            self._slider_release_in_progress = False
            self._slider_release_primary_name = ""
            self._drag_baseline_text = None
            self._drag_baseline_state_network_dsl = None
            self._suppress_slider_refresh = False
            self._stop_preview_timer("_slider_release_commit_timer")

        if not (has_surviving_gesture_scope or has_surviving_replay_scope):
            self._slider_triggered_simulation = False
            self._last_slider_change_name = ""
            self._stop_preview_timer("_variable_update_timer")
            self._stop_preview_timer("_species_slider_update_timer")
            return

        replay_sources = {str(intent.source) for intent in surviving_replay_intents if str(intent.source)}
        if (not has_surviving_gesture_scope) and replay_sources.isdisjoint({"variable_slider", "drag_release", "reset"}):
            self._stop_preview_timer("_variable_update_timer")
        if "species_slider" not in replay_sources:
            self._stop_preview_timer("_species_slider_update_timer")

    def _prune_targeted_preview_state_for_reset(self, cleared_set_ids: Sequence[str]) -> None:
        cleared_set_id_set = {
            str(set_id or "").strip()
            for set_id in (cleared_set_ids or ())
            if str(set_id or "").strip()
        }
        if not cleared_set_id_set:
            return

        self._current_slider_replay_intent = self._pruned_slider_replay_intent_for_reset(
            self._current_slider_replay_intent,
            cleared_set_ids=cleared_set_id_set,
        )
        self._last_submitted_slider_replay_intent = self._pruned_slider_replay_intent_for_reset(
            self._last_submitted_slider_replay_intent,
            cleared_set_ids=cleared_set_id_set,
        )

        surviving_gesture_target_ids = [
            str(set_id)
            for set_id in self._slider_gesture_target_set_ids_snapshot
            if str(set_id) and str(set_id) not in cleared_set_id_set
        ]
        self._apply_surviving_slider_preview_scope(
            surviving_gesture_target_ids=surviving_gesture_target_ids,
        )

    @staticmethod
    def _normalized_target_set_id_tuple(set_ids: Sequence[str]) -> tuple[str, ...]:
        normalized: list[str] = []
        seen: set[str] = set()
        for raw_set_id in set_ids or ():
            set_id = str(raw_set_id or "").strip()
            if not set_id or set_id in seen:
                continue
            seen.add(set_id)
            normalized.append(set_id)
        return tuple(normalized)

    def _pruned_slider_replay_intent_for_allowed_targets(
        self,
        intent: Optional[SliderReplayIntent],
        *,
        allowed_set_ids: set[str],
    ) -> tuple[Optional[SliderReplayIntent], tuple[str, ...]]:
        if not isinstance(intent, SliderReplayIntent):
            return None, ()
        current_target_ids = self._normalized_target_set_id_tuple(intent.target_set_ids)
        if not current_target_ids:
            return None, ()
        surviving_target_ids = tuple(set_id for set_id in current_target_ids if set_id in allowed_set_ids)
        removed_target_ids = tuple(set_id for set_id in current_target_ids if set_id not in allowed_set_ids)
        return (
            self.build_slider_replay_intent(
                set_ids=surviving_target_ids,
                source=intent.source,
            ),
            removed_target_ids,
        )

    def reconcile_slider_target_membership(self, target_set_ids: Sequence[str]) -> tuple[str, ...]:
        allowed_target_ids = set(self._normalized_target_set_id_tuple(target_set_ids))
        removed_target_ids: list[str] = []

        def record_removed(ids: Sequence[str]) -> None:
            for raw_set_id in ids or ():
                set_id = str(raw_set_id or "").strip()
                if set_id and set_id not in removed_target_ids:
                    removed_target_ids.append(set_id)

        current_gesture_ids = self._normalized_target_set_id_tuple(
            self._slider_gesture_target_set_ids_snapshot
        )
        surviving_gesture_ids = tuple(
            set_id for set_id in current_gesture_ids if set_id in allowed_target_ids
        )
        if surviving_gesture_ids != current_gesture_ids:
            record_removed(set_id for set_id in current_gesture_ids if set_id not in allowed_target_ids)
            self._slider_gesture_target_set_ids_snapshot = list(surviving_gesture_ids)

        next_current_intent, removed_current_ids = self._pruned_slider_replay_intent_for_allowed_targets(
            self._current_slider_replay_intent,
            allowed_set_ids=allowed_target_ids,
        )
        record_removed(removed_current_ids)
        self._current_slider_replay_intent = next_current_intent

        next_submitted_intent, removed_submitted_ids = self._pruned_slider_replay_intent_for_allowed_targets(
            self._last_submitted_slider_replay_intent,
            allowed_set_ids=allowed_target_ids,
        )
        record_removed(removed_submitted_ids)
        self._last_submitted_slider_replay_intent = next_submitted_intent

        if not removed_target_ids:
            return ()

        self._apply_surviving_slider_preview_scope(
            surviving_gesture_target_ids=self._slider_gesture_target_set_ids_snapshot,
        )
        return tuple(removed_target_ids)

    def build_slider_replay_intent(
        self,
        *,
        set_ids: Sequence[str] | str,
        source: str,
    ) -> Optional[SliderReplayIntent]:
        intent = SliderReplayIntent(target_set_ids=set_ids, source=str(source or ""))
        if not intent.target_set_ids or not intent.source:
            return None
        return intent

    def stage_slider_replay_intent(
        self,
        *,
        set_ids: Sequence[str],
        source: str,
    ) -> Optional[SliderReplayIntent]:
        intent = self.build_slider_replay_intent(set_ids=set_ids, source=source)
        self._current_slider_replay_intent = intent
        return intent

    def _reconcile_species_slider_replay_intent(self) -> None:
        current = self.current_slider_replay_intent()
        submitted_intent = self._last_submitted_slider_replay_intent
        submitted_species_intent = (
            submitted_intent
            if isinstance(submitted_intent, SliderReplayIntent) and submitted_intent.source == "species_slider"
            else None
        )
        if current is not None and current.source != "species_slider":
            return
        if current is None and submitted_species_intent is None:
            return
        next_intent = self.build_slider_replay_intent(
            set_ids=self._staged_concentration_overlay_target_set_ids(),
            source="species_slider",
        )
        if submitted_species_intent is not None:
            if next_intent == submitted_species_intent:
                self._current_slider_replay_intent = next_intent
                return
            self._last_submitted_slider_replay_intent = None
            self._require_slider_preview_lifecycle_port().invalidate_slider_preview_work()
            if next_intent is None:
                self._current_slider_replay_intent = None
                return
            self.submit_slider_replay_intent(
                next_intent,
                preserve_existing_request=True,
            )
            return
        self._current_slider_replay_intent = next_intent

    def submit_slider_replay_intent(
        self,
        intent: Optional[SliderReplayIntent],
        *,
        preserve_existing_request: bool = False,
    ) -> None:
        normalized_intent = (
            intent
            if isinstance(intent, SliderReplayIntent) and intent.target_set_ids and intent.source
            else None
        )
        lifecycle_port = self._require_slider_preview_lifecycle_port()
        if normalized_intent is None:
            self._current_slider_replay_intent = None
            self._last_submitted_slider_replay_intent = None
            lifecycle_port.clear_pending_slider_preview_replay(clear_plot_updates=False)
            return
        lifecycle_port.submit_slider_preview_replay_intent(
            normalized_intent,
            preserve_existing_request=bool(preserve_existing_request),
        )
        self._current_slider_replay_intent = normalized_intent
        self._last_submitted_slider_replay_intent = normalized_intent

    def _staged_concentration_overlay_target_set_ids(self) -> tuple[str, ...]:
        return tuple(
            str(set_id)
            for set_id, overlay in self._staged_concentration_overlays_by_set_id.items()
            if str(set_id) and bool(overlay)
        )

    def _prune_current_species_slider_replay_intent(self) -> None:
        self._reconcile_species_slider_replay_intent()

    def _dirty_slider_replay_target_set_ids_for_reset(self) -> tuple[str, ...]:
        dirty_workspace_ids = {
            str(set_id)
            for set_id in self.local_mechanism_workspace_set_ids()
            if str(set_id)
        }
        dirty_overlay_ids = set(self._staged_concentration_overlay_target_set_ids())
        dirty_target_ids = dirty_workspace_ids | dirty_overlay_ids
        if not dirty_target_ids:
            return ()
        ordered_target_ids: list[str] = []
        seen: set[str] = set()
        for set_id in self.effective_slider_edit_target_set_ids():
            set_id_s = str(set_id or "").strip()
            if not set_id_s or set_id_s in seen or set_id_s not in dirty_target_ids:
                continue
            seen.add(set_id_s)
            ordered_target_ids.append(set_id_s)
        for set_id in (*self.local_mechanism_workspace_set_ids(), *self._staged_concentration_overlay_target_set_ids()):
            set_id_s = str(set_id or "").strip()
            if not set_id_s or set_id_s in seen or set_id_s not in dirty_target_ids:
                continue
            seen.add(set_id_s)
            ordered_target_ids.append(set_id_s)
        return tuple(ordered_target_ids)

    def capture_reset_slider_replay_intent(self) -> Optional[SliderReplayIntent]:
        intent = self.build_slider_replay_intent(
            set_ids=self._dirty_slider_replay_target_set_ids_for_reset(),
            source="reset",
        )
        if intent is not None:
            return intent
        current = self.current_slider_replay_intent()
        if current is None:
            return None
        intent = self.build_slider_replay_intent(
            set_ids=current.target_set_ids,
            source="reset",
        )
        if intent is not None:
            return intent
        return None

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
            self._deauthorize_completed_run_display_for_slider_preview_scope(changed_set_ids)
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
            self._deauthorize_completed_run_display_for_slider_preview_scope(changed_set_ids)
            self.stage_slider_replay_intent(
                set_ids=changed_set_ids,
                source="species_slider",
            )
            self._reconcile_species_slider_replay_intent()
            self._refresh_transaction_button_state()
        return bool(changed)

    def clear_staged_concentration_overlays(self) -> None:
        cleared_set_ids = list(self._staged_concentration_overlays_by_set_id.keys())
        self._staged_concentration_overlays_by_set_id.clear()
        self._bump_dirty_state_generation(cleared_set_ids)
        self._prune_current_species_slider_replay_intent()
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
            self._prune_current_species_slider_replay_intent()
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
        self._prune_current_species_slider_replay_intent()
        self._refresh_transaction_button_state()
        return True

    def has_staged_concentration_overlays(self) -> bool:
        return any(bool(values) for values in self._staged_concentration_overlays_by_set_id.values())

    def has_dirty_transaction(self) -> bool:
        return bool(self.has_local_mechanism_workspaces()) or bool(self.has_staged_concentration_overlays())

    def apply_staged_concentration_overlays(self, model: object) -> ConcentrationOverlayCommitResult:
        if model is None or not self._staged_concentration_overlays_by_set_id:
            return ConcentrationOverlayCommitResult()
        touched_rows: list[int] = []
        touched_set_ids: list[str] = []
        try:
            store = model.store()
            species_list = list(store.visible_species())
        except Exception:
            return ConcentrationOverlayCommitResult()
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
        self._prune_current_species_slider_replay_intent()
        self._refresh_transaction_button_state()
        return ConcentrationOverlayCommitResult(
            touched_rows=tuple(touched_rows),
            touched_set_ids=tuple(touched_set_ids),
        )

    def clear_working_transaction(
        self,
        *,
        clear_committed_slider_values: bool = False,
        invalidate_preview_work: bool = True,
    ) -> None:
        changed_set_ids = self.local_mechanism_workspace_set_ids() + list(
            self._staged_concentration_overlays_by_set_id.keys()
        )
        self._param_store.clear_all_local_overrides()
        self._staged_concentration_overlays_by_set_id.clear()
        self._bump_dirty_state_generation(changed_set_ids)
        if clear_committed_slider_values:
            self._param_store.clear_shared_params()
        self.reset_preview_state()
        if invalidate_preview_work:
            self._require_slider_preview_lifecycle_port().invalidate_slider_preview_work()
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
        self._current_slider_replay_intent = None
        self._last_submitted_slider_replay_intent = None
        self._drag_baseline_text = None
        self._drag_baseline_state_network_dsl = None
        self._suppress_slider_refresh = False
        self._slider_release_in_progress = False
        self._slider_release_primary_name = ""

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
        settings_owner = getattr(mw, "_settings_owner", None)
        settings = getattr(settings_owner, "qsettings", None)
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
        if not self.is_mechanism_valid_for_preview():
            self._show_invalid_preview_state()
            return
        logger.debug("Variable %s changed to %s", name, value)
        self._last_slider_change_name = name
        target_set_ids = self._ensure_slider_gesture_target_snapshot()
        intent = self.stage_slider_replay_intent(
            set_ids=target_set_ids,
            source="variable_slider",
        )
        self.submit_slider_replay_intent(
            intent,
            preserve_existing_request=True,
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
            return

        self._slider_triggered_simulation = True
        interval_ms = self.variable_preview_debounce_ms(name)
        if self._slider_drag_active:
            mw._status_label.setText(f"Adjusting {name} = {value:.3g}")
        else:
            mw._status_label.setText(f"Previewing {name} = {value:.3g}")
        timer = self._ensure_variable_update_timer(interval_ms=interval_ms)
        timer.stop()
        timer.setInterval(interval_ms)
        timer.start()

    def commit_slider_value(self, name: str, value: float) -> None:
        """Stage a programmatic slider change for preview runs without mutating editor text."""
        mw = self._mw
        if not self.is_mechanism_valid_for_preview():
            self._show_invalid_preview_state()
            return
        self._last_slider_change_name = name
        target_set_ids = self._ensure_slider_gesture_target_snapshot()
        intent = self.stage_slider_replay_intent(
            set_ids=target_set_ids,
            source="variable_slider",
        )
        self.submit_slider_replay_intent(
            intent,
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

        timer = self._ensure_slider_release_commit_timer()
        interval_ms = self.variable_preview_debounce_ms(name)
        timer.setInterval(interval_ms)
        timer.stop()
        timer.start()

    def queue_species_slider_simulation(self, *, label: str, delay_ms: int) -> None:
        """Queue a fast preview run for species-mode slider edits."""
        if not self.is_mechanism_valid_for_preview():
            self._show_invalid_preview_state()
            return
        try:
            delay_ms_i = int(delay_ms)
        except Exception:
            delay_ms_i = 80
        delay_ms_i = max(0, min(500, delay_ms_i))

        self._last_slider_change_name = str(label or "init")
        intent = self.stage_slider_replay_intent(
            set_ids=self._selected_mechanism_target_set_ids(),
            source="species_slider",
        )
        self.submit_slider_replay_intent(
            intent,
            preserve_existing_request=True,
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
        if not self.is_mechanism_valid_for_preview():
            self._show_invalid_preview_state()
            return

        pending = dict(self._pending_slider_values or {})
        target_set_ids = self.slider_gesture_target_set_ids_snapshot()
        self._pending_slider_values.clear()
        self._suppress_slider_refresh = False
        self._slider_release_in_progress = False
        self._slider_release_primary_name = ""
        self._clear_slider_gesture_target_snapshot()

        if not pending:
            return

        intent = self.stage_slider_replay_intent(
            set_ids=target_set_ids,
            source="drag_release",
        )
        self.submit_slider_replay_intent(
            intent,
            preserve_existing_request=True,
        )
        timer = self._ensure_variable_update_timer()
        timer.stop()
        timer.setInterval(0)
        timer.start()

    def _dispatch_variable_slider_preview_if_valid(self) -> None:
        if not self.is_mechanism_valid_for_preview():
            self._show_invalid_preview_state()
            return
        self._require_slider_preview_lifecycle_port().launch_pending_slider_preview_replay()

    def _dispatch_species_slider_preview_if_valid(self) -> None:
        if not self.is_mechanism_valid_for_preview():
            self._show_invalid_preview_state()
            return
        self._require_slider_preview_lifecycle_port().launch_pending_slider_preview_replay()

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
        timer.timeout.connect(self._dispatch_variable_slider_preview_if_valid)
        return timer

    def _create_species_slider_update_timer(self):
        timer = QtCore.QTimer(self._mw)
        timer.setSingleShot(True)
        timer.timeout.connect(self._dispatch_species_slider_preview_if_valid)
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
