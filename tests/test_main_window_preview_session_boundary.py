from __future__ import annotations

import pytest
from PySide6 import QtCore


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
        "run_simulation_from_slider",
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
    assert main_window.simulation_controller.run_state.pending_slider_simulation is False
    assert tuple(main_window.simulation_controller.run_state.pending_slider_target_set_ids) == ()


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
        "run_simulation_from_slider",
        lambda: calls.__setitem__("dispatch", calls["dispatch"] + 1),
    )

    owner.queue_species_slider_simulation(label="init:A", delay_ms=80)

    assert calls == {"validity": 1, "refresh": 1, "dispatch": 0}
    assert owner._species_slider_update_timer is None
    assert main_window._status_label.text() == "Mechanism invalid — no preview available."
    assert cache.active_preview_cache_key is None
    assert cache.active_preview_scope_set_ids is None
    assert main_window.simulation_controller.run_state.pending_slider_simulation is False
    assert tuple(main_window.simulation_controller.run_state.pending_slider_target_set_ids) == ()


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

    assert tuple(main_window._sim_controller.run_state.pending_slider_target_set_ids) == (set0_id,)

    _set_batch_current_and_selected_rows(main_window, current_row=2, selected_rows=[2])
    owner.finalize_slider_release_commit()

    assert owner.slider_gesture_target_set_ids_snapshot() == []
    assert tuple(main_window._sim_controller.run_state.pending_slider_target_set_ids) == (set0_id,)
    assert tuple(main_window._sim_controller.run_state.pending_slider_target_set_ids) != (set2_id,)
    owner.stop_variable_update_timer()


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
    monkeypatch.setattr(main_window.simulation_controller, "run_simulation_from_slider", lambda: None)
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
