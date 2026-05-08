from __future__ import annotations

import time
from types import SimpleNamespace

import pytest
from PySide6 import QtCore

from kindred.gui.main_window_preview_session import MainWindowPreviewSession
from kindred.gui.ports import SliderReplayIntent
from kindred.gui.controllers.simulation_run_state import PreviewOwnershipState

pytestmark = pytest.mark.gui


PREVIEW_SESSION_STATE_FIELDS = (
    "_slider_drag_active",
    "_slider_triggered_simulation",
    "_last_slider_change_name",
    "_pending_slider_values",
    "_param_store",
    "_dirty_state_generation_by_set_id",
    "_staged_concentration_overlays_by_set_id",
    "_slider_gesture_target_set_ids_snapshot",
    "_drag_baseline_text",
    "_drag_baseline_state_network_dsl",
    "_suppress_slider_refresh",
    "_slider_release_in_progress",
    "_slider_release_primary_name",
    "_variable_update_timer",
    "_species_slider_update_timer",
    "_slider_release_commit_timer",
    "_current_slider_replay_intent",
    "_last_submitted_slider_replay_intent",
)


def _set_batch_current_and_selected_rows(
    main_window,
    *,
    current_row: int,
    selected_rows: list[int],
) -> None:
    table = main_window._batch_table
    model = main_window._batch_model
    assert table is not None
    sel = table.selectionModel()
    assert sel is not None
    current_idx = model.index(int(current_row), 0)
    assert current_idx.isValid()
    table.setCurrentIndex(current_idx)
    sel.clearSelection()
    for row in selected_rows:
        idx = model.index(int(row), 0)
        assert idx.isValid()
        sel.select(idx, QtCore.QItemSelectionModel.Select | QtCore.QItemSelectionModel.Rows)


def _ensure_batch_rows(main_window, count: int) -> None:
    while int(main_window._batch_store.row_count()) < int(count):
        main_window._add_batch_set()


def _set_valid_preview_mechanism(main_window) -> None:
    main_window._mechanism_editor._reactions_text.setPlainText(
        "reaction: A -> B; k=1.0\ninitial: A=1.0\ninitial: B=0.0"
    )
    main_window._extract_and_populate_variables()


def _wait_for_timer_to_settle(timer: QtCore.QTimer, *, timeout_ms: int = 500) -> None:
    deadline = time.monotonic() + (float(timeout_ms) / 1000.0)
    while time.monotonic() < deadline:
        QtCore.QCoreApplication.processEvents()
        if not timer.isActive():
            return
        time.sleep(0.01)
    raise AssertionError(f"Timer did not settle within {int(timeout_ms)} ms")


def _pending_slider_preview_launch(main_window):
    return main_window.simulation_controller.run_state.pending_slider_preview_launch


class _RecordingSliderPreviewLifecyclePort:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def submit_slider_preview_replay_intent(
        self,
        intent: SliderReplayIntent,
        *,
        preserve_existing_request: bool = False,
    ) -> None:
        self.calls.append(("submit", intent, bool(preserve_existing_request)))

    def clear_pending_slider_preview_replay(self, *, clear_plot_updates: bool = True) -> None:
        self.calls.append(("clear", bool(clear_plot_updates)))

    def invalidate_slider_preview_work(self) -> None:
        self.calls.append(("invalidate",))

    def launch_pending_slider_preview_replay(self) -> None:
        self.calls.append(("launch",))


class _RecordingActiveTimer:
    def __init__(self) -> None:
        self.active = True
        self.stop_calls = 0

    def isActive(self) -> bool:
        return bool(self.active)

    def stop(self) -> None:
        self.stop_calls += 1
        self.active = False


def _make_minimal_preview_host() -> object:
    return SimpleNamespace(
        is_mechanism_valid_for_preview=lambda: True,
        _status_label=SimpleNamespace(setText=lambda value: None),
    )


def _arm_pending_preview_state(main_window) -> tuple[str, int]:
    owner = main_window._preview_session
    controller = main_window.simulation_controller
    _set_valid_preview_mechanism(main_window)
    _set_batch_current_and_selected_rows(main_window, current_row=0, selected_rows=[0])
    set0_id = str(main_window.batch_set_id_for_row(0) or "")
    assert set0_id

    owner.sync_committed_slider_values({"k1": 1.0})
    owner.stage_slider_value("k1", 2.0, target_set_ids=[set0_id])
    intent = SliderReplayIntent(target_set_ids=(set0_id,), source="variable_slider")
    owner.submit_slider_replay_intent(intent, preserve_existing_request=True)

    request_id = controller.run_state.pending_slider_preview_launch.request_id
    assert request_id is not None
    controller.run_state.preview_ownership = PreviewOwnershipState(
        request_id=int(request_id),
        epoch=1,
        target_set_ids=(set0_id,),
    )
    controller._active_run_id = 7
    controller._latest_sim_request_id = int(request_id)
    controller._queue_slider_plot_update(
        set_id=set0_id,
        cache_key="pending-preview-cache",
        request_id=int(request_id),
        run_id=7,
        slider_triggered=True,
    )
    return set0_id, int(request_id)


def test_main_window_preview_session_owns_preview_state(main_window) -> None:
    owner = getattr(main_window, "_preview_session", None)

    assert owner is not None, "Expected MainWindow to expose an explicit preview-session owner."
    assert "_slider_overrides" not in main_window.__dict__, (
        "Mechanism working slider state must not be stored directly on MainWindow."
    )

    for field_name in PREVIEW_SESSION_STATE_FIELDS:
        assert field_name not in main_window.__dict__, (
            f"Preview-session field {field_name} must not be stored directly on MainWindow."
        )
        assert field_name in owner.__dict__, (
            f"Preview-session owner must physically store {field_name}."
        )


def test_main_window_no_longer_exposes_preview_session_compatibility_aliases(main_window) -> None:
    for field_name in PREVIEW_SESSION_STATE_FIELDS:
        assert field_name not in type(main_window).__dict__, (
            f"Preview-session compatibility alias {field_name} should not be defined on MainWindow."
        )


def test_preview_session_submit_and_launch_use_explicit_lifecycle_boundary_without_main_window_controller_attr() -> None:
    owner = MainWindowPreviewSession(_make_minimal_preview_host())
    lifecycle_port = _RecordingSliderPreviewLifecyclePort()
    owner.set_slider_preview_lifecycle_port(lifecycle_port)
    intent = SliderReplayIntent(target_set_ids=("set-1",), source="variable_slider")

    owner.submit_slider_replay_intent(intent, preserve_existing_request=True)
    owner._dispatch_variable_slider_preview_if_valid()

    assert lifecycle_port.calls == [
        ("submit", intent, True),
        ("launch",),
    ]


def test_preview_session_reset_preview_state_clears_local_replay_state_without_invalidating_lifecycle() -> None:
    owner = MainWindowPreviewSession(_make_minimal_preview_host())
    lifecycle_port = _RecordingSliderPreviewLifecyclePort()
    owner.set_slider_preview_lifecycle_port(lifecycle_port)
    owner._pending_slider_values["k1"] = 2.0
    owner._slider_triggered_simulation = True
    owner._slider_drag_active = True
    owner._slider_gesture_target_set_ids_snapshot = ["set-1"]
    owner._current_slider_replay_intent = SliderReplayIntent(target_set_ids=("set-1",), source="variable_slider")
    owner._last_submitted_slider_replay_intent = SliderReplayIntent(
        target_set_ids=("set-1",),
        source="variable_slider",
    )
    owner._drag_baseline_text = "reaction: A -> B"
    owner._drag_baseline_state_network_dsl = "state A"
    owner._suppress_slider_refresh = True
    owner._slider_release_in_progress = True
    owner._slider_release_primary_name = "k1"

    owner.reset_preview_state()

    assert owner._pending_slider_values == {}
    assert owner._slider_triggered_simulation is False
    assert owner._slider_drag_active is False
    assert owner._slider_gesture_target_set_ids_snapshot == []
    assert owner._current_slider_replay_intent is None
    assert owner._last_submitted_slider_replay_intent is None
    assert owner._drag_baseline_text is None
    assert owner._drag_baseline_state_network_dsl is None
    assert owner._suppress_slider_refresh is False
    assert owner._slider_release_in_progress is False
    assert owner._slider_release_primary_name == ""
    assert lifecycle_port.calls == []


def test_main_window_preview_session_reports_whole_transaction_dirty_state(main_window) -> None:
    owner = main_window._preview_session

    main_window._batch_store.set_species(["A"])
    main_window._batch_store.set_value(0, "A", "1.0")

    assert owner.has_dirty_transaction() is False

    owner._pending_slider_values["k1"] = 2.0
    assert owner.has_dirty_transaction() is False
    owner.clear_pending_slider_values()

    owner.sync_committed_slider_values({"k1": 1.0})
    owner.stage_slider_value("k1", 2.0)
    assert owner.has_dirty_transaction() is True

    owner.clear_working_transaction()
    assert owner.has_dirty_transaction() is False

    changed = owner.stage_concentration_value_for_rows([0], species="A", value=2.5)
    assert changed is True
    assert owner.has_dirty_transaction() is True


def test_main_window_preview_session_tracks_per_set_mechanism_workspaces(main_window) -> None:
    owner = main_window._preview_session
    owner.sync_committed_slider_values({"k1": 1.0})
    _ensure_batch_rows(main_window, 2)

    _set_batch_current_and_selected_rows(main_window, current_row=0, selected_rows=[0])
    set0_id = str(main_window.batch_set_id_for_row(0) or "")
    set1_id = str(main_window.batch_set_id_for_row(1) or "")
    assert set0_id
    assert set1_id

    owner.stage_slider_value("k1", 2.0)
    assert owner.slider_overrides() == {"k1": pytest.approx(2.0)}
    assert owner.effective_slider_values() == {"k1": pytest.approx(2.0)}
    assert owner.local_mechanism_workspace(set0_id) == {"k1": pytest.approx(2.0)}
    assert owner.local_mechanism_workspace(set1_id) == {}
    assert owner.local_mechanism_workspace_set_ids() == [set0_id]

    _set_batch_current_and_selected_rows(main_window, current_row=1, selected_rows=[1])
    assert owner.slider_overrides() == {}
    assert owner.effective_slider_values() == {"k1": pytest.approx(1.0)}

    owner.stage_slider_value("k1", 3.0)
    assert owner.local_mechanism_workspace(set0_id) == {"k1": pytest.approx(2.0)}
    assert owner.local_mechanism_workspace(set1_id) == {"k1": pytest.approx(3.0)}
    assert owner.local_mechanism_workspace_set_ids() == [set0_id, set1_id]

    _set_batch_current_and_selected_rows(main_window, current_row=0, selected_rows=[0])
    assert owner.slider_overrides() == {"k1": pytest.approx(2.0)}
    assert owner.effective_slider_values() == {"k1": pytest.approx(2.0)}


def test_main_window_preview_session_stage_slider_value_explicit_target_set_ids_updates_only_targets(main_window) -> None:
    owner = main_window._preview_session
    owner.sync_committed_slider_values({"k1": 1.0})
    _ensure_batch_rows(main_window, 3)

    set0_id = str(main_window.batch_set_id_for_row(0) or "")
    set1_id = str(main_window.batch_set_id_for_row(1) or "")
    set2_id = str(main_window.batch_set_id_for_row(2) or "")
    assert set0_id
    assert set1_id
    assert set2_id

    owner.stage_slider_value("k1", 2.5, target_set_ids=[set0_id, set1_id])

    assert owner.local_mechanism_workspace(set0_id) == {"k1": pytest.approx(2.5)}
    assert owner.local_mechanism_workspace(set1_id) == {"k1": pytest.approx(2.5)}
    assert owner.local_mechanism_workspace(set2_id) == {}


def test_main_window_preview_session_dirty_state_includes_any_local_mechanism_workspace(main_window) -> None:
    owner = main_window._preview_session
    owner.sync_committed_slider_values({"k1": 1.0})
    _ensure_batch_rows(main_window, 2)

    _set_batch_current_and_selected_rows(main_window, current_row=0, selected_rows=[0])
    set0_id = str(main_window.batch_set_id_for_row(0) or "")
    set1_id = str(main_window.batch_set_id_for_row(1) or "")
    assert set0_id
    assert set1_id

    owner.stage_slider_value("k1", 2.0)
    _set_batch_current_and_selected_rows(main_window, current_row=1, selected_rows=[1])

    assert owner.slider_overrides() == {}
    assert owner.has_dirty_transaction() is True
    assert owner.has_local_mechanism_workspace(set0_id) is True
    assert owner.has_local_mechanism_workspace(set1_id) is False

    owner.clear_local_mechanism_workspace(set0_id)
    assert owner.has_dirty_transaction() is False


def test_main_window_preview_session_dirty_state_generation_tracks_per_set_mutations(main_window) -> None:
    owner = main_window._preview_session
    owner.sync_committed_slider_values({"k1": 1.0})
    main_window._batch_store.set_species(["A"])
    main_window._batch_store.set_value(0, "A", "1.0")
    _ensure_batch_rows(main_window, 2)

    _set_batch_current_and_selected_rows(main_window, current_row=0, selected_rows=[0])
    set0_id = str(main_window.batch_set_id_for_row(0) or "")
    set1_id = str(main_window.batch_set_id_for_row(1) or "")
    assert set0_id
    assert set1_id

    assert owner.dirty_state_generation(set0_id) == 0
    assert owner.dirty_state_generation(set1_id) == 0

    owner.stage_slider_value("k1", 2.0)
    assert owner.has_dirty_state_for_set(set0_id) is True
    assert owner.dirty_state_generation(set0_id) == 1
    assert owner.dirty_state_generation(set1_id) == 0

    changed = owner.stage_concentration_value_for_rows([1], species="A", value=2.5)
    assert changed is True
    assert owner.has_dirty_state_for_set(set1_id) is True
    assert owner.dirty_state_generation(set0_id) == 1
    assert owner.dirty_state_generation(set1_id) == 1

    cleared = owner.discard_concentration_overlays_for_set_ids([set1_id])
    assert cleared is True
    assert owner.has_dirty_state_for_set(set1_id) is False
    assert owner.dirty_state_generation(set1_id) == 2


def test_main_window_preview_session_snapshots_slider_gesture_targets(main_window) -> None:
    owner = main_window._preview_session
    _ensure_batch_rows(main_window, 2)
    _set_batch_current_and_selected_rows(main_window, current_row=0, selected_rows=[0, 1])
    set0_id = str(main_window.batch_set_id_for_row(0) or "")
    main_window.set_slider_edit_target_set_ids([set0_id])
    expected_snapshot = [set0_id]

    owner.on_slider_drag_started("k1")
    assert owner.slider_gesture_target_set_ids_snapshot() == expected_snapshot

    _set_batch_current_and_selected_rows(main_window, current_row=1, selected_rows=[1])
    assert owner.slider_gesture_target_set_ids_snapshot() == expected_snapshot

    owner.on_slider_drag_finished("k1")
    assert owner.slider_gesture_target_set_ids_snapshot() == []


def test_main_window_preview_session_non_drag_changes_seed_target_snapshot(main_window) -> None:
    owner = main_window._preview_session
    _set_valid_preview_mechanism(main_window)
    owner.sync_committed_slider_values({"k1": 1.0})
    _ensure_batch_rows(main_window, 2)
    _set_batch_current_and_selected_rows(main_window, current_row=0, selected_rows=[0, 1])
    set0_id = str(main_window.batch_set_id_for_row(0) or "")
    main_window.set_slider_edit_target_set_ids([set0_id])
    expected_snapshot = [set0_id]

    owner.on_variable_changed("k1", 2.0)

    assert owner.slider_gesture_target_set_ids_snapshot() == expected_snapshot
    owner.stop_variable_update_timer()


def test_main_window_preview_session_invalid_mechanism_skips_variable_preview_dispatch(
    main_window,
    monkeypatch,
) -> None:
    owner = main_window._preview_session
    _set_valid_preview_mechanism(main_window)
    owner.sync_committed_slider_values({"k1": 1.0})
    _ensure_batch_rows(main_window, 1)
    _set_batch_current_and_selected_rows(main_window, current_row=0, selected_rows=[0])
    set0_id = str(main_window.batch_set_id_for_row(0) or "")
    assert set0_id

    calls = {"validity": 0, "refresh": 0, "dispatch": 0}
    cache = main_window.simulation_controller.batch_cache
    cache.active_preview_cache_key = "stale-preview"
    cache.active_preview_scope_set_ids = (set0_id,)

    def _invalid_preview() -> bool:
        calls["validity"] += 1
        return False

    monkeypatch.setattr(main_window, "is_mechanism_valid_for_preview", _invalid_preview)
    monkeypatch.setattr(
        main_window,
        "_refresh_batch_display_from_focus_and_shown",
        lambda: calls.__setitem__("refresh", calls["refresh"] + 1),
    )
    monkeypatch.setattr(
        main_window.simulation_controller,
        "launch_pending_slider_preview_replay",
        lambda: calls.__setitem__("dispatch", calls["dispatch"] + 1),
    )

    owner.on_variable_changed("k1", 2.0)

    assert calls == {"validity": 1, "refresh": 1, "dispatch": 0}
    assert owner.local_mechanism_workspace(set0_id) == {}
    assert owner.slider_gesture_target_set_ids_snapshot() == []
    assert owner._variable_update_timer is None
    assert owner._pending_slider_values == {}
    assert main_window._status_label.text() == "Mechanism invalid — no preview available."
    assert cache.active_preview_cache_key is None
    assert cache.active_preview_scope_set_ids is None
    pending_launch = _pending_slider_preview_launch(main_window)
    assert pending_launch.active is False
    assert pending_launch.target_set_ids == ()


def test_main_window_preview_session_valid_mechanism_allows_variable_preview_dispatch(
    main_window,
    monkeypatch,
) -> None:
    owner = main_window._preview_session
    _set_valid_preview_mechanism(main_window)
    owner.sync_committed_slider_values({"k1": 1.0})
    _ensure_batch_rows(main_window, 1)
    _set_batch_current_and_selected_rows(main_window, current_row=0, selected_rows=[0])

    calls = {"validity": 0}

    def _valid_preview() -> bool:
        calls["validity"] += 1
        return True

    monkeypatch.setattr(main_window, "is_mechanism_valid_for_preview", _valid_preview)

    owner.on_variable_changed("k1", 2.0)

    assert calls["validity"] == 1
    assert owner._variable_update_timer is not None
    assert owner._variable_update_timer.isActive() is True
    owner.stop_variable_update_timer()


def test_main_window_preview_session_timer_rechecks_variable_preview_validity_before_dispatch(
    main_window,
    monkeypatch,
) -> None:
    owner = main_window._preview_session
    _set_valid_preview_mechanism(main_window)
    owner.sync_committed_slider_values({"k1": 1.0})
    _ensure_batch_rows(main_window, 1)
    _set_batch_current_and_selected_rows(main_window, current_row=0, selected_rows=[0])
    set0_id = str(main_window.batch_set_id_for_row(0) or "")
    assert set0_id

    calls = {"validity": 0, "refresh": 0, "dispatch": 0}
    cache = main_window.simulation_controller.batch_cache
    cache.active_preview_cache_key = "stale-preview"
    cache.active_preview_scope_set_ids = (set0_id,)
    preview_is_valid = {"value": True}

    def _preview_validity() -> bool:
        calls["validity"] += 1
        return bool(preview_is_valid["value"])

    monkeypatch.setattr(main_window, "is_mechanism_valid_for_preview", _preview_validity)
    monkeypatch.setattr(
        main_window,
        "_refresh_batch_display_from_focus_and_shown",
        lambda: calls.__setitem__("refresh", calls["refresh"] + 1),
    )
    monkeypatch.setattr(
        main_window.simulation_controller,
        "launch_pending_slider_preview_replay",
        lambda: calls.__setitem__("dispatch", calls["dispatch"] + 1),
    )

    owner.on_variable_changed("k1", 2.0)
    timer = owner._variable_update_timer
    assert timer is not None
    assert timer.isActive() is True

    preview_is_valid["value"] = False
    _wait_for_timer_to_settle(timer)

    assert calls == {"validity": 2, "refresh": 1, "dispatch": 0}
    assert main_window._status_label.text() == "Mechanism invalid — no preview available."
    assert cache.active_preview_cache_key is None
    assert cache.active_preview_scope_set_ids is None


def test_main_window_preview_session_non_drag_changes_stage_focused_target_set_by_default(main_window) -> None:
    owner = main_window._preview_session
    _set_valid_preview_mechanism(main_window)
    owner.sync_committed_slider_values({"k1": 1.0})
    _ensure_batch_rows(main_window, 3)
    _set_batch_current_and_selected_rows(main_window, current_row=0, selected_rows=[0, 1])

    set0_id = str(main_window.batch_set_id_for_row(0) or "")
    set1_id = str(main_window.batch_set_id_for_row(1) or "")
    set2_id = str(main_window.batch_set_id_for_row(2) or "")
    main_window.set_slider_edit_target_set_ids([])

    assert main_window.slider_edit_target_set_ids() == []

    owner.on_variable_changed("k1", 2.0)

    assert owner.local_mechanism_workspace(set0_id) == {"k1": pytest.approx(2.0)}
    assert owner.local_mechanism_workspace(set1_id) == {}
    assert owner.local_mechanism_workspace(set2_id) == {}
    owner.stop_variable_update_timer()


def test_main_window_preview_session_invalid_mechanism_skips_species_preview_dispatch(
    main_window,
    monkeypatch,
) -> None:
    owner = main_window._preview_session
    _set_valid_preview_mechanism(main_window)
    _ensure_batch_rows(main_window, 1)
    _set_batch_current_and_selected_rows(main_window, current_row=0, selected_rows=[0])
    set0_id = str(main_window.batch_set_id_for_row(0) or "")
    assert set0_id

    calls = {"validity": 0, "refresh": 0, "dispatch": 0}
    cache = main_window.simulation_controller.batch_cache
    cache.active_preview_cache_key = "stale-species-preview"
    cache.active_preview_scope_set_ids = (set0_id,)

    def _invalid_preview() -> bool:
        calls["validity"] += 1
        return False

    monkeypatch.setattr(main_window, "is_mechanism_valid_for_preview", _invalid_preview)
    monkeypatch.setattr(
        main_window,
        "_refresh_batch_display_from_focus_and_shown",
        lambda: calls.__setitem__("refresh", calls["refresh"] + 1),
    )
    monkeypatch.setattr(
        main_window.simulation_controller,
        "launch_pending_slider_preview_replay",
        lambda: calls.__setitem__("dispatch", calls["dispatch"] + 1),
    )

    owner.queue_species_slider_simulation(label="init:A", delay_ms=80)

    assert calls == {"validity": 1, "refresh": 1, "dispatch": 0}
    assert owner._species_slider_update_timer is None
    assert main_window._status_label.text() == "Mechanism invalid — no preview available."
    assert cache.active_preview_cache_key is None
    assert cache.active_preview_scope_set_ids is None
    pending_launch = _pending_slider_preview_launch(main_window)
    assert pending_launch.active is False
    assert pending_launch.target_set_ids == ()


def test_main_window_preview_session_focus_navigation_does_not_accumulate_hidden_targets(main_window) -> None:
    owner = main_window._preview_session
    _set_valid_preview_mechanism(main_window)
    owner.sync_committed_slider_values({"k1": 1.0})
    _ensure_batch_rows(main_window, 3)

    set0_id = str(main_window.batch_set_id_for_row(0) or "")
    set1_id = str(main_window.batch_set_id_for_row(1) or "")
    set2_id = str(main_window.batch_set_id_for_row(2) or "")
    assert set0_id and set1_id and set2_id

    _set_batch_current_and_selected_rows(main_window, current_row=0, selected_rows=[0])
    main_window.set_slider_edit_target_set_ids([])
    assert main_window.slider_edit_target_set_ids() == []

    _set_batch_current_and_selected_rows(main_window, current_row=1, selected_rows=[1])
    _set_batch_current_and_selected_rows(main_window, current_row=2, selected_rows=[2])

    assert main_window.slider_edit_target_set_ids() == []

    owner.on_variable_changed("k1", 2.0)

    assert owner.local_mechanism_workspace(set0_id) == {}
    assert owner.local_mechanism_workspace(set1_id) == {}
    assert owner.local_mechanism_workspace(set2_id) == {"k1": pytest.approx(2.0)}
    owner.stop_variable_update_timer()


def test_main_window_preview_session_non_drag_changes_stage_explicit_edit_targets_not_selected_rows(main_window) -> None:
    owner = main_window._preview_session
    _set_valid_preview_mechanism(main_window)
    owner.sync_committed_slider_values({"k1": 1.0})
    _ensure_batch_rows(main_window, 3)
    _set_batch_current_and_selected_rows(main_window, current_row=0, selected_rows=[0, 1])

    set0_id = str(main_window.batch_set_id_for_row(0) or "")
    set1_id = str(main_window.batch_set_id_for_row(1) or "")
    set2_id = str(main_window.batch_set_id_for_row(2) or "")

    main_window.set_slider_edit_target_set_ids([set2_id, set0_id])
    owner.on_variable_changed("k1", 2.0)

    assert owner.local_mechanism_workspace(set0_id) == {"k1": pytest.approx(2.0)}
    assert owner.local_mechanism_workspace(set1_id) == {}
    assert owner.local_mechanism_workspace(set2_id) == {"k1": pytest.approx(2.0)}
    owner.stop_variable_update_timer()


def test_main_window_preview_session_non_drag_changes_stage_focused_plus_explicit_targets_and_drops_old_focus(
    main_window,
) -> None:
    owner = main_window._preview_session
    _set_valid_preview_mechanism(main_window)
    owner.sync_committed_slider_values({"k1": 1.0})
    _ensure_batch_rows(main_window, 4)

    set0_id = str(main_window.batch_set_id_for_row(0) or "")
    set1_id = str(main_window.batch_set_id_for_row(1) or "")
    set2_id = str(main_window.batch_set_id_for_row(2) or "")
    set3_id = str(main_window.batch_set_id_for_row(3) or "")
    assert set0_id and set1_id and set2_id and set3_id

    main_window.set_slider_edit_target_set_ids([set0_id, set1_id])
    _set_batch_current_and_selected_rows(main_window, current_row=2, selected_rows=[2])

    assert main_window.slider_edit_target_set_ids() == [set0_id, set1_id]

    owner.on_variable_changed("k1", 2.0)

    assert owner.local_mechanism_workspace(set0_id) == {"k1": pytest.approx(2.0)}
    assert owner.local_mechanism_workspace(set1_id) == {"k1": pytest.approx(2.0)}
    assert owner.local_mechanism_workspace(set2_id) == {"k1": pytest.approx(2.0)}
    assert owner.local_mechanism_workspace(set3_id) == {}
    owner.stop_variable_update_timer()

    _set_batch_current_and_selected_rows(main_window, current_row=3, selected_rows=[3])
    owner.on_variable_changed("k1", 3.0)

    assert owner.local_mechanism_workspace(set0_id) == {"k1": pytest.approx(3.0)}
    assert owner.local_mechanism_workspace(set1_id) == {"k1": pytest.approx(3.0)}
    assert owner.local_mechanism_workspace(set2_id) == {"k1": pytest.approx(2.0)}
    assert owner.local_mechanism_workspace(set3_id) == {"k1": pytest.approx(3.0)}
    owner.stop_variable_update_timer()


def test_main_window_preview_session_drag_staging_uses_snapshotted_edit_targets_after_selection_change(main_window) -> None:
    owner = main_window._preview_session
    _set_valid_preview_mechanism(main_window)
    owner.sync_committed_slider_values({"k1": 1.0})
    _ensure_batch_rows(main_window, 3)
    _set_batch_current_and_selected_rows(main_window, current_row=0, selected_rows=[0, 1])

    set0_id = str(main_window.batch_set_id_for_row(0) or "")
    set1_id = str(main_window.batch_set_id_for_row(1) or "")
    set2_id = str(main_window.batch_set_id_for_row(2) or "")
    main_window.set_slider_edit_target_set_ids([set0_id])

    owner.on_slider_drag_started("k1")
    _set_batch_current_and_selected_rows(main_window, current_row=2, selected_rows=[2])
    owner.on_variable_changed("k1", 2.0)

    assert owner.local_mechanism_workspace(set0_id) == {"k1": pytest.approx(2.0)}
    assert owner.local_mechanism_workspace(set1_id) == {}
    assert owner.local_mechanism_workspace(set2_id) == {}


def test_main_window_preview_session_drag_snapshot_uses_explicit_edit_targets_after_selection_change(main_window) -> None:
    owner = main_window._preview_session
    _set_valid_preview_mechanism(main_window)
    owner.sync_committed_slider_values({"k1": 1.0})
    _ensure_batch_rows(main_window, 3)
    _set_batch_current_and_selected_rows(main_window, current_row=0, selected_rows=[0, 1])

    set0_id = str(main_window.batch_set_id_for_row(0) or "")
    set1_id = str(main_window.batch_set_id_for_row(1) or "")
    set2_id = str(main_window.batch_set_id_for_row(2) or "")

    main_window.set_slider_edit_target_set_ids([set0_id, set2_id])
    owner.on_slider_drag_started("k1")
    _set_batch_current_and_selected_rows(main_window, current_row=1, selected_rows=[1])
    owner.on_variable_changed("k1", 2.0)

    assert owner.local_mechanism_workspace(set0_id) == {"k1": pytest.approx(2.0)}
    assert owner.local_mechanism_workspace(set1_id) == {}
    assert owner.local_mechanism_workspace(set2_id) == {"k1": pytest.approx(2.0)}


def test_main_window_preview_session_finalize_drag_release_preserves_original_target_sets(main_window) -> None:
    owner = main_window._preview_session
    _set_valid_preview_mechanism(main_window)
    owner.sync_committed_slider_values({"k1": 1.0})
    _ensure_batch_rows(main_window, 3)
    _set_batch_current_and_selected_rows(main_window, current_row=0, selected_rows=[0, 1])

    set0_id = str(main_window.batch_set_id_for_row(0) or "")
    set2_id = str(main_window.batch_set_id_for_row(2) or "")
    main_window.set_slider_edit_target_set_ids([set0_id])

    owner.on_slider_drag_started("k1")
    owner.on_variable_changed("k1", 2.0)
    owner.on_slider_drag_finished("k1")

    pending_launch = main_window._sim_controller.run_state.pending_slider_preview_launch
    assert pending_launch.target_set_ids == (set0_id,)

    _set_batch_current_and_selected_rows(main_window, current_row=2, selected_rows=[2])
    owner.finalize_slider_release_commit()

    assert owner.slider_gesture_target_set_ids_snapshot() == []
    pending_launch = main_window._sim_controller.run_state.pending_slider_preview_launch
    assert pending_launch.target_set_ids == (set0_id,)
    assert pending_launch.target_set_ids != (set2_id,)
    owner.stop_variable_update_timer()


def test_main_window_preview_session_tracks_explicit_slider_replay_intent(main_window) -> None:
    owner = main_window._preview_session
    _set_valid_preview_mechanism(main_window)
    owner.sync_committed_slider_values({"k1": 1.0})
    _ensure_batch_rows(main_window, 3)
    _set_batch_current_and_selected_rows(main_window, current_row=0, selected_rows=[0, 1])

    set0_id = str(main_window.batch_set_id_for_row(0) or "")
    main_window.set_slider_edit_target_set_ids([set0_id])

    owner.on_slider_drag_started("k1")
    _set_batch_current_and_selected_rows(main_window, current_row=2, selected_rows=[2])
    owner.on_variable_changed("k1", 2.0)

    assert owner.current_slider_replay_intent() == SliderReplayIntent(
        target_set_ids=(set0_id,),
        source="variable_slider",
    )


def test_main_window_preview_session_build_slider_replay_intent_preserves_single_string_target_identity(
    main_window,
) -> None:
    owner = main_window._preview_session
    set0_id = str(main_window.batch_set_id_for_row(0) or "set-0")

    assert owner.build_slider_replay_intent(
        set_ids=set0_id,
        source="reset",
    ) == SliderReplayIntent(
        target_set_ids=(set0_id,),
        source="reset",
    )


def test_main_window_preview_session_submit_slider_replay_intent_none_clears_controller_pending_state(
    main_window,
) -> None:
    owner = main_window._preview_session
    controller = main_window.simulation_controller
    set0_id = str(main_window.batch_set_id_for_row(0) or "")

    intent = owner.build_slider_replay_intent(
        set_ids=[set0_id],
        source="variable_slider",
    )
    owner.submit_slider_replay_intent(intent, preserve_existing_request=True)
    assert controller.run_state.pending_slider_preview_launch.active is True

    owner.submit_slider_replay_intent(None)

    assert owner.current_slider_replay_intent() is None
    pending_launch = controller.run_state.pending_slider_preview_launch
    assert pending_launch.active is False
    assert pending_launch.request_id is None
    assert pending_launch.target_set_ids == ()


def test_main_window_preview_session_submit_slider_replay_intent_rejects_invalid_direct_intent() -> None:
    owner = MainWindowPreviewSession(_make_minimal_preview_host())
    lifecycle_port = _RecordingSliderPreviewLifecyclePort()
    owner.set_slider_preview_lifecycle_port(lifecycle_port)
    invalid_intent = SliderReplayIntent(target_set_ids=(), source="")

    owner.submit_slider_replay_intent(invalid_intent, preserve_existing_request=True)

    assert owner.current_slider_replay_intent() is None
    assert owner._last_submitted_slider_replay_intent is None
    assert lifecycle_port.calls == [("clear", False)]


def _submit_current_species_replay_intent(main_window) -> SliderReplayIntent:
    owner = main_window._preview_session
    intent = owner.current_slider_replay_intent()
    assert intent is not None
    assert intent.source == "species_slider"
    owner.submit_slider_replay_intent(intent, preserve_existing_request=True)
    return intent


def _submit_species_replay_intent_for_set_ids(main_window, set_ids: list[str] | tuple[str, ...]) -> SliderReplayIntent:
    owner = main_window._preview_session
    intent = owner.build_slider_replay_intent(
        set_ids=set_ids,
        source="species_slider",
    )
    assert intent is not None
    owner.submit_slider_replay_intent(intent, preserve_existing_request=True)
    return intent


def test_main_window_preview_session_species_baseline_reversion_clears_replay_intent_and_pending_launch(
    main_window,
) -> None:
    owner = main_window._preview_session
    controller = main_window.simulation_controller
    main_window._batch_model.set_species(["A"])
    main_window._batch_store.set_value(0, "A", "1.0")

    assert owner.stage_concentration_value_for_rows([0], species="A", value=2.5) is True
    set0_id = str(main_window.batch_set_id_for_row(0) or "")
    assert _submit_current_species_replay_intent(main_window) == SliderReplayIntent(
        target_set_ids=(set0_id,),
        source="species_slider",
    )
    assert controller.run_state.pending_slider_preview_launch.active is True

    assert owner.stage_concentration_value_for_rows([0], species="A", value=1.0) is True

    assert owner.current_slider_replay_intent() is None
    pending_launch = controller.run_state.pending_slider_preview_launch
    assert pending_launch.active is False
    assert pending_launch.request_id is None
    assert pending_launch.target_set_ids == ()


def test_main_window_preview_session_species_partial_baseline_reversion_preserves_remaining_pending_scope(
    main_window,
) -> None:
    owner = main_window._preview_session
    controller = main_window.simulation_controller
    main_window._batch_model.set_species(["A"])
    main_window._batch_store.set_value(0, "A", "1.0")
    _ensure_batch_rows(main_window, 2)
    main_window._batch_store.set_value(1, "A", "2.0")

    set0_id = str(main_window.batch_set_id_for_row(0) or "")
    set1_id = str(main_window.batch_set_id_for_row(1) or "")
    assert set0_id
    assert set1_id

    assert owner.stage_concentration_value_for_rows([0], species="A", value=2.5) is True
    assert owner.stage_concentration_value_for_rows([1], species="A", value=3.5) is True
    _submit_species_replay_intent_for_set_ids(main_window, [set0_id, set1_id])
    assert controller.run_state.pending_slider_preview_launch.target_set_ids == (set0_id, set1_id)

    assert owner.stage_concentration_value_for_rows([1], species="A", value=2.0) is True

    assert owner.current_slider_replay_intent() == SliderReplayIntent(
        target_set_ids=(set0_id,),
        source="species_slider",
    )
    pending_launch = controller.run_state.pending_slider_preview_launch
    assert pending_launch.active is True
    assert pending_launch.target_set_ids == (set0_id,)


def test_main_window_preview_session_clear_staged_concentration_overlays_clears_pending_launch_after_submit(
    main_window,
) -> None:
    owner = main_window._preview_session
    controller = main_window.simulation_controller
    main_window._batch_model.set_species(["A"])
    main_window._batch_store.set_value(0, "A", "1.0")

    assert owner.stage_concentration_value_for_rows([0], species="A", value=2.5) is True
    _submit_current_species_replay_intent(main_window)
    assert controller.run_state.pending_slider_preview_launch.active is True

    owner.clear_staged_concentration_overlays()

    assert owner.current_slider_replay_intent() is None
    pending_launch = controller.run_state.pending_slider_preview_launch
    assert pending_launch.active is False
    assert pending_launch.request_id is None
    assert pending_launch.target_set_ids == ()


def test_main_window_preview_session_discard_concentration_overlays_prunes_pending_launch_after_submit(
    main_window,
) -> None:
    owner = main_window._preview_session
    controller = main_window.simulation_controller
    main_window._batch_model.set_species(["A"])
    main_window._batch_store.set_value(0, "A", "1.0")
    _ensure_batch_rows(main_window, 2)
    main_window._batch_store.set_value(1, "A", "2.0")

    set0_id = str(main_window.batch_set_id_for_row(0) or "")
    set1_id = str(main_window.batch_set_id_for_row(1) or "")
    assert set0_id
    assert set1_id

    assert owner.stage_concentration_value_for_rows([0], species="A", value=2.5) is True
    assert owner.stage_concentration_value_for_rows([1], species="A", value=3.5) is True
    _submit_species_replay_intent_for_set_ids(main_window, [set0_id, set1_id])
    assert controller.run_state.pending_slider_preview_launch.target_set_ids == (set0_id, set1_id)

    assert owner.discard_concentration_overlays_for_set_ids([set1_id]) is True

    assert owner.current_slider_replay_intent() == SliderReplayIntent(
        target_set_ids=(set0_id,),
        source="species_slider",
    )
    pending_launch = controller.run_state.pending_slider_preview_launch
    assert pending_launch.active is True
    assert pending_launch.target_set_ids == (set0_id,)


def test_main_window_preview_session_prune_staged_concentration_overlays_updates_pending_launch_after_submit(
    main_window,
) -> None:
    owner = main_window._preview_session
    controller = main_window.simulation_controller
    main_window._batch_model.set_species(["A", "B"])
    main_window._batch_store.set_value(0, "A", "1.0")
    main_window._batch_store.set_value(0, "B", "2.0")
    _ensure_batch_rows(main_window, 2)
    main_window._batch_store.set_value(1, "A", "3.0")
    main_window._batch_store.set_value(1, "B", "4.0")

    set0_id = str(main_window.batch_set_id_for_row(0) or "")
    set1_id = str(main_window.batch_set_id_for_row(1) or "")
    assert set0_id
    assert set1_id

    assert owner.stage_concentration_value_for_rows([0], species="A", value=2.5) is True
    assert owner.stage_concentration_value_for_rows([1], species="B", value=5.5) is True
    _submit_species_replay_intent_for_set_ids(main_window, [set0_id, set1_id])
    assert controller.run_state.pending_slider_preview_launch.target_set_ids == (set0_id, set1_id)

    assert owner.prune_staged_concentration_overlays_to_species(["A"]) is True

    assert owner.current_slider_replay_intent() == SliderReplayIntent(
        target_set_ids=(set0_id,),
        source="species_slider",
    )
    pending_launch = controller.run_state.pending_slider_preview_launch
    assert pending_launch.active is True
    assert pending_launch.target_set_ids == (set0_id,)


def test_main_window_preview_session_apply_staged_concentration_overlays_clears_pending_launch_after_submit(
    main_window,
) -> None:
    owner = main_window._preview_session
    controller = main_window.simulation_controller
    main_window._batch_model.set_species(["A"])
    main_window._batch_store.set_value(0, "A", "1.0")

    assert owner.stage_concentration_value_for_rows([0], species="A", value=2.5) is True
    _submit_current_species_replay_intent(main_window)
    assert controller.run_state.pending_slider_preview_launch.active is True

    commit_result = owner.apply_staged_concentration_overlays(main_window._batch_model)

    assert commit_result.touched_rows == (0,)
    assert owner.current_slider_replay_intent() is None
    pending_launch = controller.run_state.pending_slider_preview_launch
    assert pending_launch.active is False
    assert pending_launch.request_id is None
    assert pending_launch.target_set_ids == ()


def test_main_window_preview_session_species_scope_clear_rejects_queued_preview_plot_update(
    main_window,
    monkeypatch,
) -> None:
    owner = main_window._preview_session
    controller = main_window.simulation_controller
    main_window._batch_model.set_species(["A"])
    main_window._batch_store.set_value(0, "A", "1.0")
    set0_id = str(main_window.batch_set_id_for_row(0) or "")
    assert set0_id

    assert owner.stage_concentration_value_for_rows([0], species="A", value=2.5) is True
    _submit_current_species_replay_intent(main_window)
    request_id = controller.run_state.pending_slider_preview_launch.request_id
    assert request_id is not None

    controller._active_run_id = 2
    controller._latest_sim_request_id = int(request_id)
    controller.run_state.preview_ownership = PreviewOwnershipState(
        request_id=int(request_id),
        epoch=1,
        target_set_ids=(set0_id,),
    )
    controller._queue_slider_plot_update(
        set_id=set0_id,
        cache_key="cache-key",
        request_id=int(request_id),
        run_id=2,
        slider_triggered=True,
    )

    display_calls = {"count": 0}
    monkeypatch.setattr(main_window, "_shown_batch_set_ids", lambda: [set0_id])
    monkeypatch.setattr(main_window, "_batch_current_row", lambda: 0)
    monkeypatch.setattr(main_window, "_batch_set_id_for_row", lambda row: set0_id)
    monkeypatch.setattr(
        main_window._simulation_batch_owner,
        "display_cached_batch_selection",
        lambda *args, **kwargs: display_calls.__setitem__("count", display_calls["count"] + 1) or True,
    )

    assert owner.stage_concentration_value_for_rows([0], species="A", value=1.0) is True

    assert controller._flush_slider_plot_updates() is False
    assert display_calls["count"] == 0


def test_on_programmatic_mechanism_load_clears_pending_preview_state_without_display_state(main_window) -> None:
    controller = main_window.simulation_controller
    set0_id, request_id = _arm_pending_preview_state(main_window)

    assert main_window.main_plot_has_data() is False
    assert controller.run_state.pending_slider_preview_launch.active is True
    assert controller.run_state.preview_ownership.request_id == request_id
    assert controller._plot_coalescer.pending.set_ids == {set0_id}

    main_window._on_programmatic_mechanism_load()

    pending_launch = controller.run_state.pending_slider_preview_launch
    assert pending_launch.active is False
    assert pending_launch.request_id is None
    assert pending_launch.target_set_ids == ()
    ownership = controller.run_state.preview_ownership
    assert ownership.request_id is None
    assert ownership.target_set_ids == ()
    queued_plot = controller._plot_coalescer.pending
    assert queued_plot.set_ids == set()
    assert queued_plot.request_id is None
    assert queued_plot.cache_key is None


def test_main_window_preview_session_finalize_drag_release_invalid_preview_does_not_start_timer(
    main_window,
    monkeypatch,
) -> None:
    owner = main_window._preview_session
    _set_valid_preview_mechanism(main_window)
    owner._pending_slider_values["k1"] = 2.0
    owner._slider_release_in_progress = True
    owner._slider_release_primary_name = "k1"
    owner._capture_slider_gesture_target_snapshot()

    calls = {"validity": 0, "refresh": 0, "dispatch": 0}

    def _invalid_preview() -> bool:
        calls["validity"] += 1
        return False

    monkeypatch.setattr(main_window, "is_mechanism_valid_for_preview", _invalid_preview)
    monkeypatch.setattr(
        main_window,
        "_refresh_batch_display_from_focus_and_shown",
        lambda: calls.__setitem__("refresh", calls["refresh"] + 1),
    )
    monkeypatch.setattr(
        main_window.simulation_controller,
        "launch_pending_slider_preview_replay",
        lambda: calls.__setitem__("dispatch", calls["dispatch"] + 1),
    )

    timer = owner._ensure_variable_update_timer()
    owner.finalize_slider_release_commit()

    assert calls == {"validity": 1, "refresh": 1, "dispatch": 0}
    assert timer.isActive() is False
    assert owner._pending_slider_values == {"k1": 2.0}
    assert owner._slider_release_in_progress is True
    assert owner._slider_release_primary_name == "k1"
    assert main_window._status_label.text() == "Mechanism invalid — no preview available."


def test_main_window_preview_session_commit_slider_value_uses_focused_target_set_without_drag(
    main_window,
    monkeypatch,
) -> None:
    owner = main_window._preview_session
    main_window._mechanism_editor._reactions_text.setPlainText(
        "reaction: A -> B; k=1.0\ninitial: A=1.0\ninitial: B=0.0"
    )
    main_window._extract_and_populate_variables()
    owner.sync_committed_slider_values({"k1": 1.0})
    _ensure_batch_rows(main_window, 3)
    _set_batch_current_and_selected_rows(main_window, current_row=0, selected_rows=[0, 1])
    baseline_text = main_window._mechanism_editor._reactions_text.toPlainText()

    set0_id = str(main_window.batch_set_id_for_row(0) or "")
    set1_id = str(main_window.batch_set_id_for_row(1) or "")
    set2_id = str(main_window.batch_set_id_for_row(2) or "")
    main_window.set_slider_edit_target_set_ids([set0_id])
    preview_validity_calls = {"count": 0}

    def _valid_preview() -> bool:
        preview_validity_calls["count"] += 1
        return True

    monkeypatch.setattr(main_window, "is_mechanism_valid_for_preview", _valid_preview)
    monkeypatch.setattr(main_window.simulation_controller, "launch_pending_slider_preview_replay", lambda: None)
    owner.commit_slider_value("k1", 2.0)
    owner.stop_variable_update_timer()

    assert preview_validity_calls["count"] == 1
    assert owner.local_mechanism_workspace(set0_id) == {"k1": pytest.approx(2.0)}
    assert owner.local_mechanism_workspace(set1_id) == {}
    assert owner.local_mechanism_workspace(set2_id) == {}
    assert main_window._mechanism_editor._reactions_text.toPlainText() == baseline_text


def test_main_window_preview_session_commit_current_workspace_globalizes_current_and_preserves_other_dirty_sets(
    main_window,
) -> None:
    owner = main_window._preview_session
    owner.sync_committed_slider_values({"k1": 1.0})
    _ensure_batch_rows(main_window, 2)

    set0_id = str(main_window.batch_set_id_for_row(0) or "")
    set1_id = str(main_window.batch_set_id_for_row(1) or "")
    assert set0_id
    assert set1_id

    _set_batch_current_and_selected_rows(main_window, current_row=0, selected_rows=[0])
    owner.stage_slider_value("k1", 2.0)
    _set_batch_current_and_selected_rows(main_window, current_row=1, selected_rows=[1])
    owner.stage_slider_value("k1", 3.0)

    _set_batch_current_and_selected_rows(main_window, current_row=0, selected_rows=[0])
    owner.commit_current_mechanism_workspace()

    assert owner.effective_slider_values_for_set(set0_id) == {"k1": pytest.approx(2.0)}
    assert owner.effective_slider_values_for_set(set1_id) == {"k1": pytest.approx(3.0)}
    assert owner.local_mechanism_workspace(set0_id) == {}
    assert owner.local_mechanism_workspace(set1_id) == {"k1": pytest.approx(3.0)}
    assert owner.has_local_mechanism_workspaces() is True


def test_main_window_preview_session_reset_current_workspace_only_discards_focused_set(main_window) -> None:
    owner = main_window._preview_session
    owner.sync_committed_slider_values({"k1": 1.0})
    _ensure_batch_rows(main_window, 2)

    set0_id = str(main_window.batch_set_id_for_row(0) or "")
    set1_id = str(main_window.batch_set_id_for_row(1) or "")
    assert set0_id
    assert set1_id

    _set_batch_current_and_selected_rows(main_window, current_row=0, selected_rows=[0])
    owner.stage_slider_value("k1", 2.0)
    _set_batch_current_and_selected_rows(main_window, current_row=1, selected_rows=[1])
    owner.stage_slider_value("k1", 3.0)

    _set_batch_current_and_selected_rows(main_window, current_row=0, selected_rows=[0])
    changed = owner.reset_current_mechanism_workspace()

    assert changed is True
    assert owner.local_mechanism_workspace(set0_id) == {}
    assert owner.local_mechanism_workspace(set1_id) == {"k1": pytest.approx(3.0)}
    assert owner.effective_slider_values_for_set(set0_id) == {"k1": pytest.approx(1.0)}
    assert owner.effective_slider_values_for_set(set1_id) == {"k1": pytest.approx(3.0)}


def test_main_window_preview_session_reset_current_workspace_invalidates_bound_lifecycle_once(main_window) -> None:
    owner = main_window._preview_session
    lifecycle_port = _RecordingSliderPreviewLifecyclePort()
    owner.set_slider_preview_lifecycle_port(lifecycle_port)
    owner.sync_committed_slider_values({"k1": 1.0})

    _set_batch_current_and_selected_rows(main_window, current_row=0, selected_rows=[0])
    owner.stage_slider_value("k1", 2.0)

    assert owner.reset_current_mechanism_workspace() is True
    assert lifecycle_port.calls == [("invalidate",)]


def test_main_window_preview_session_reset_all_workspaces_discards_every_local_workspace(main_window) -> None:
    owner = main_window._preview_session
    owner.sync_committed_slider_values({"k1": 1.0})
    _ensure_batch_rows(main_window, 2)

    _set_batch_current_and_selected_rows(main_window, current_row=0, selected_rows=[0])
    owner.stage_slider_value("k1", 2.0)
    _set_batch_current_and_selected_rows(main_window, current_row=1, selected_rows=[1])
    owner.stage_slider_value("k1", 3.0)

    owner.clear_all_local_mechanism_workspaces()

    assert owner.has_local_mechanism_workspaces() is False
    assert owner.slider_overrides() == {}


def test_main_window_preview_session_reset_mechanism_workspaces_does_not_invalidate_bound_lifecycle(
    main_window,
) -> None:
    owner = main_window._preview_session
    lifecycle_port = _RecordingSliderPreviewLifecyclePort()
    owner.set_slider_preview_lifecycle_port(lifecycle_port)
    owner.sync_committed_slider_values({"k1": 1.0})
    _ensure_batch_rows(main_window, 2)

    _set_batch_current_and_selected_rows(main_window, current_row=0, selected_rows=[0])
    owner.stage_slider_value("k1", 2.0)
    _set_batch_current_and_selected_rows(main_window, current_row=1, selected_rows=[1])
    owner.stage_slider_value("k1", 3.0)

    set0_id = str(main_window.batch_set_id_for_row(0) or "")
    set1_id = str(main_window.batch_set_id_for_row(1) or "")
    assert owner.reset_mechanism_workspaces([set0_id, set1_id]) is True
    assert lifecycle_port.calls == []


def test_main_window_preview_session_reset_mechanism_workspaces_preserves_surviving_preview_scope(
    main_window,
) -> None:
    owner = main_window._preview_session
    lifecycle_port = _RecordingSliderPreviewLifecyclePort()
    owner.set_slider_preview_lifecycle_port(lifecycle_port)
    owner.sync_committed_slider_values({"k1": 1.0})
    _ensure_batch_rows(main_window, 2)

    _set_batch_current_and_selected_rows(main_window, current_row=0, selected_rows=[0])
    owner.stage_slider_value("k1", 2.0)
    _set_batch_current_and_selected_rows(main_window, current_row=1, selected_rows=[1])
    owner.stage_slider_value("k1", 3.0)

    set0_id = str(main_window.batch_set_id_for_row(0) or "")
    set1_id = str(main_window.batch_set_id_for_row(1) or "")
    assert set0_id
    assert set1_id

    owner._current_slider_replay_intent = SliderReplayIntent(
        target_set_ids=(set0_id, set1_id),
        source="variable_slider",
    )
    owner._last_submitted_slider_replay_intent = SliderReplayIntent(
        target_set_ids=(set0_id, set1_id),
        source="variable_slider",
    )
    owner._slider_gesture_target_set_ids_snapshot = [set0_id, set1_id]
    owner._pending_slider_values["k1"] = 3.0
    owner._slider_triggered_simulation = True
    owner._last_slider_change_name = "k1"
    owner._slider_release_in_progress = True
    owner._slider_release_primary_name = "k1"
    owner._drag_baseline_text = "reaction: A -> B"
    owner._drag_baseline_state_network_dsl = "state A"
    owner._suppress_slider_refresh = True
    owner._variable_update_timer = _RecordingActiveTimer()
    owner._slider_release_commit_timer = _RecordingActiveTimer()

    assert owner.reset_mechanism_workspaces([set0_id]) is True

    assert owner.local_mechanism_workspace(set0_id) == {}
    assert owner.local_mechanism_workspace(set1_id) == {"k1": pytest.approx(3.0)}
    assert owner.current_slider_replay_intent() == SliderReplayIntent(
        target_set_ids=(set1_id,),
        source="variable_slider",
    )
    assert owner._last_submitted_slider_replay_intent == SliderReplayIntent(
        target_set_ids=(set1_id,),
        source="variable_slider",
    )
    assert owner.slider_gesture_target_set_ids_snapshot() == [set1_id]
    assert owner._pending_slider_values == {"k1": pytest.approx(3.0)}
    assert owner._slider_triggered_simulation is True
    assert owner._last_slider_change_name == "k1"
    assert owner._slider_release_in_progress is True
    assert owner._slider_release_primary_name == "k1"
    assert owner._drag_baseline_text == "reaction: A -> B"
    assert owner._drag_baseline_state_network_dsl == "state A"
    assert owner._suppress_slider_refresh is True
    assert owner._variable_update_timer.stop_calls == 0
    assert owner._slider_release_commit_timer.stop_calls == 0
    assert lifecycle_port.calls == []


def test_main_window_preview_session_reset_mechanism_workspaces_preserves_existing_species_replay_scope(
    main_window,
) -> None:
    owner = main_window._preview_session
    lifecycle_port = _RecordingSliderPreviewLifecyclePort()
    owner.set_slider_preview_lifecycle_port(lifecycle_port)
    owner.sync_committed_slider_values({"k1": 1.0})
    _ensure_batch_rows(main_window, 2)
    main_window._batch_model.set_species(["A"])
    main_window._batch_store.set_value(0, "A", "1.0")
    main_window._batch_store.set_value(1, "A", "2.0")

    _set_batch_current_and_selected_rows(main_window, current_row=0, selected_rows=[0])
    owner.stage_slider_value("k1", 2.0)

    set0_id = str(main_window.batch_set_id_for_row(0) or "")
    set1_id = str(main_window.batch_set_id_for_row(1) or "")
    assert set0_id
    assert set1_id

    assert owner.stage_concentration_value_for_rows([0], species="A", value=2.5) is True
    assert owner.stage_concentration_value_for_rows([1], species="A", value=3.5) is True
    owner._current_slider_replay_intent = SliderReplayIntent(
        target_set_ids=(set0_id, set1_id),
        source="species_slider",
    )
    owner._last_submitted_slider_replay_intent = SliderReplayIntent(
        target_set_ids=(set0_id, set1_id),
        source="species_slider",
    )
    owner._last_slider_change_name = "init:A"
    owner._species_slider_update_timer = _RecordingActiveTimer()

    assert owner.reset_mechanism_workspaces([set0_id]) is True

    assert owner.current_slider_replay_intent() == SliderReplayIntent(
        target_set_ids=(set0_id, set1_id),
        source="species_slider",
    )
    assert owner._last_submitted_slider_replay_intent == SliderReplayIntent(
        target_set_ids=(set0_id, set1_id),
        source="species_slider",
    )
    assert owner._last_slider_change_name == "init:A"
    assert owner._species_slider_update_timer.stop_calls == 0
    assert lifecycle_port.calls == []


def test_main_window_preview_session_commit_current_workspace_invalidates_bound_lifecycle_once(main_window) -> None:
    owner = main_window._preview_session
    lifecycle_port = _RecordingSliderPreviewLifecyclePort()
    owner.set_slider_preview_lifecycle_port(lifecycle_port)
    owner.sync_committed_slider_values({"k1": 1.0})

    _set_batch_current_and_selected_rows(main_window, current_row=0, selected_rows=[0])
    owner.stage_slider_value("k1", 2.0)

    owner.commit_current_mechanism_workspace()

    assert lifecycle_port.calls == [("invalidate",)]


def test_main_window_preview_session_commit_current_workspace_can_skip_bound_lifecycle_invalidation(
    main_window,
) -> None:
    owner = main_window._preview_session
    lifecycle_port = _RecordingSliderPreviewLifecyclePort()
    owner.set_slider_preview_lifecycle_port(lifecycle_port)
    owner.sync_committed_slider_values({"k1": 1.0})

    _set_batch_current_and_selected_rows(main_window, current_row=0, selected_rows=[0])
    owner.stage_slider_value("k1", 2.0)

    owner.commit_current_mechanism_workspace(invalidate_preview_work=False)

    assert lifecycle_port.calls == []


def test_main_window_preview_session_clear_working_transaction_invalidates_bound_lifecycle_once(
    main_window,
) -> None:
    owner = main_window._preview_session
    lifecycle_port = _RecordingSliderPreviewLifecyclePort()
    owner.set_slider_preview_lifecycle_port(lifecycle_port)
    owner.sync_committed_slider_values({"k1": 1.0})

    _set_batch_current_and_selected_rows(main_window, current_row=0, selected_rows=[0])
    owner.stage_slider_value("k1", 2.0)

    owner.clear_working_transaction()

    assert lifecycle_port.calls == [("invalidate",)]


def test_main_window_preview_session_clear_working_transaction_can_skip_bound_lifecycle_invalidation(
    main_window,
) -> None:
    owner = main_window._preview_session
    lifecycle_port = _RecordingSliderPreviewLifecyclePort()
    owner.set_slider_preview_lifecycle_port(lifecycle_port)
    owner.sync_committed_slider_values({"k1": 1.0})

    _set_batch_current_and_selected_rows(main_window, current_row=0, selected_rows=[0])
    owner.stage_slider_value("k1", 2.0)

    owner.clear_working_transaction(invalidate_preview_work=False)

    assert lifecycle_port.calls == []
