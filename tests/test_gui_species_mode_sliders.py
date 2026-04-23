from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from PySide6 import QtCore, QtGui, QtWidgets

from kindred.core.batch_initial_conditions import BatchInitialConditionsStore
from kindred.gui.widgets.batch_initial_conditions_table import BatchInitialConditionsTableModel

pytestmark = [pytest.mark.gui]

def _slider_handle_center(slider: QtWidgets.QSlider) -> QtCore.QPoint:
    option = QtWidgets.QStyleOptionSlider()
    slider.initStyleOption(option)
    handle = slider.style().subControlRect(
        QtWidgets.QStyle.CC_Slider,
        option,
        QtWidgets.QStyle.SC_SliderHandle,
        slider,
    )
    return handle.center()

def _set_batch_current_and_selected_rows(
    main_window,
    *,
    current_row: int,
    selected_rows: list[int],
) -> QtCore.QItemSelectionModel:
    table = main_window._batch_table
    model = main_window._batch_model
    assert table is not None
    sel = table.selectionModel()
    assert sel is not None
    assert selected_rows
    current_idx = model.index(int(current_row), 0)
    assert current_idx.isValid()
    table.setCurrentIndex(current_idx)
    sel.clearSelection()
    for row in selected_rows:
        idx = model.index(int(row), 0)
        assert idx.isValid()
        sel.select(idx, QtCore.QItemSelectionModel.Select | QtCore.QItemSelectionModel.Rows)
    return sel

def _select_batch_rows(main_window, rows: list[int]) -> QtCore.QItemSelectionModel:
    return _set_batch_current_and_selected_rows(
        main_window,
        current_row=int(rows[0]),
        selected_rows=list(rows),
    )

def _species_slider_row(main_window, species: str):
    panel = main_window._mechanism_editor.species_sliders_widget()
    assert panel is not None
    row = panel._rows.get(str(species))
    assert row is not None
    return row

def _set_valid_preview_mechanism(main_window) -> None:
    main_window._mechanism_editor._reactions_text.setPlainText(
        "reaction: A -> B; k=1.0\ninitial: A=1.0\ninitial: B=0.0"
    )
    main_window._extract_and_populate_variables()

def _column_for_header(model: QtCore.QAbstractItemModel, header: str) -> int:
    for column in range(model.columnCount()):
        if model.headerData(column, QtCore.Qt.Horizontal, QtCore.Qt.DisplayRole) == str(header):
            return int(column)
    raise AssertionError(f"Missing column header {header!r}")

def _table_cell_center(table: QtWidgets.QTableView, index: QtCore.QModelIndex) -> QtCore.QPoint:
    rect = table.visualRect(index).intersected(table.viewport().rect())
    assert rect.isValid()
    return rect.center()

def _visual_headers(table: QtWidgets.QTableView, model: QtCore.QAbstractItemModel) -> list[str]:
    header = table.horizontalHeader()
    return [
        str(model.headerData(header.logicalIndex(visual), QtCore.Qt.Horizontal, QtCore.Qt.DisplayRole) or "")
        for visual in range(model.columnCount())
    ]

def _track_checkstate_set_data(model: QtCore.QAbstractItemModel, monkeypatch) -> list[tuple[int, int, int, object]]:
    original = model.setData
    calls: list[tuple[int, int, int, object]] = []

    def _tracked(index: QtCore.QModelIndex, value, role: int = QtCore.Qt.EditRole):
        calls.append((int(index.row()), int(index.column()), int(role), value))
        return original(index, value, role)

    monkeypatch.setattr(model, "setData", _tracked)
    return calls


def _pending_slider_preview_launch(main_window):
    return main_window.simulation_controller.run_state.pending_slider_preview_launch

def _find_slider_visibility_action(main_window, entry_kind: str, name: str) -> QtGui.QAction:
    picker = main_window.findChild(QtWidgets.QToolButton, "sliderVisibilityPickerButton")
    assert picker is not None
    menu = picker.menu()
    assert menu is not None
    main_window._mechanism_editor._rebuild_slider_visibility_menu()
    for action in menu.actions():
        if action.data() == (str(entry_kind), str(name)):
            return action
    raise AssertionError(f"Missing visibility action for {(entry_kind, name)!r}")

def _set_slider_visibility(main_window, entry_kind: str, name: str, *, visible: bool) -> None:
    action = _find_slider_visibility_action(main_window, entry_kind, name)
    if bool(action.isChecked()) != bool(visible):
        action.trigger()
        QtWidgets.QApplication.processEvents()
    assert bool(action.isChecked()) is bool(visible)

def test_unified_surface_builds_species_sliders_and_syncs_with_batch_table(main_window, qtbot, monkeypatch):
    _set_valid_preview_mechanism(main_window)
    # Ensure species columns exist and seed numeric values via model edits.
    main_window._batch_model.set_species(["A", "B"])
    model = main_window._batch_model

    idx_a = model.index(0, 1)  # A column
    idx_b = model.index(0, 2)  # B column
    assert idx_a.isValid()
    assert idx_b.isValid()
    assert model.setData(idx_a, "1.0")
    assert model.setData(idx_b, "2.0")

    table = main_window._batch_table
    assert table is not None
    table.setCurrentIndex(model.index(0, 0))
    sel = table.selectionModel()
    assert sel is not None
    sel.clearSelection()
    sel.select(model.index(0, 0), QtCore.QItemSelectionModel.Select | QtCore.QItemSelectionModel.Rows)

    # Spy on the fast slider-run path.
    calls = {"n": 0}
    reset_runs = {"n": 0}

    def _count_runs():
        calls["n"] += 1

    monkeypatch.setattr(main_window.simulation_controller, "launch_pending_slider_preview_replay", _count_runs)
    monkeypatch.setattr(main_window.simulation_controller, "run_simulation", lambda: reset_runs.__setitem__("n", reset_runs["n"] + 1))

    qtbot.addWidget(main_window)
    main_window.show()
    assert main_window.findChild(QtWidgets.QToolButton, "sliderVisibilityPickerButton") is not None

    slider_a = main_window.findChild(QtWidgets.QSlider, "speciesSlider_A")
    slider_b = main_window.findChild(QtWidgets.QSlider, "speciesSlider_B")
    assert slider_a is not None
    assert slider_b is not None

    # Change A via a press + value update (drag-like) and confirm staging updates.
    press_pos = _slider_handle_center(slider_a)
    qtbot.mousePress(slider_a, QtCore.Qt.LeftButton, pos=press_pos)
    slider_a.setValue(5000)
    qtbot.mouseRelease(slider_a, QtCore.Qt.LeftButton, pos=press_pos)
    QtWidgets.QApplication.processEvents()

    # A slider_max for initial row is 5.0 (row_max=2.0, v=1.0 => max(1, 4, 5)=5).
    expected_a = 2.5
    staged = main_window._preview_session.preview_initials_for_row(0, main_window.batch_initials_for_row(0))
    assert float(main_window._batch_store.get_value(0, "A")) == pytest.approx(1.0, rel=1e-6, abs=1e-9)
    assert float(staged["A"]) == pytest.approx(expected_a, rel=1e-6, abs=1e-9)

    qtbot.waitUntil(lambda: calls["n"] >= 1, timeout=1500)

    # Programmatic table edit should update the slider when not dragging.
    assert model.setData(idx_a, "4.0")
    QtWidgets.QApplication.processEvents()

    # With row_max=4.0, slider_max for A becomes 20.0, so 4/20*10000 = 2000.
    qtbot.waitUntil(lambda: slider_a.value() == 2000, timeout=1500)

    # The unified Reset button is context-aware; the legacy "Reset row" button is removed.
    legacy_reset_btn = main_window.findChild(QtWidgets.QPushButton, "resetSpeciesRowButton")
    assert legacy_reset_btn is None

    reset_btn = main_window.findChild(QtWidgets.QPushButton, "resetSliderOverridesButton")
    assert reset_btn is not None
    prev_calls = int(calls["n"])
    qtbot.mouseClick(reset_btn, QtCore.Qt.LeftButton)
    QtWidgets.QApplication.processEvents()
    assert calls["n"] == prev_calls + 1
    assert reset_runs["n"] == 0
    assert float(main_window._batch_store.get_value(0, "A")) == pytest.approx(4.0, rel=1e-6, abs=1e-9)
    assert float(main_window._batch_store.get_value(0, "B")) == pytest.approx(2.0, rel=1e-6, abs=1e-9)

def test_unified_surface_refreshes_species_sliders_after_species_list_reset_without_selection_change(main_window, qtbot):
    main_window._batch_model.set_species(["A", "B"])
    model = main_window._batch_model
    assert model.setData(model.index(0, 1), "1.0")
    assert model.setData(model.index(0, 2), "2.0")

    table = main_window._batch_table
    assert table is not None
    table.setCurrentIndex(model.index(0, 0))
    sel = table.selectionModel()
    assert sel is not None
    sel.clearSelection()
    sel.select(model.index(0, 0), QtCore.QItemSelectionModel.Select | QtCore.QItemSelectionModel.Rows)

    qtbot.addWidget(main_window)
    main_window.show()

    assert main_window.findChild(QtWidgets.QSlider, "speciesSlider_A") is not None
    assert main_window.findChild(QtWidgets.QSlider, "speciesSlider_B") is not None

    main_window._sync_batch_species_columns(["A", "C"])
    QtWidgets.QApplication.processEvents()

    assert main_window.findChild(QtWidgets.QSlider, "speciesSlider_A") is not None
    assert main_window.findChild(QtWidgets.QSlider, "speciesSlider_B") is None
    assert main_window.findChild(QtWidgets.QSlider, "speciesSlider_C") is not None
    assert table.currentIndex().isValid()

def test_species_set_change_prunes_removed_species_overlays_and_clears_dirty_state(main_window, qtbot, monkeypatch):
    main_window._batch_model.set_species(["A", "B"])
    model = main_window._batch_model
    assert model.setData(model.index(0, 1), "1.0")
    assert model.setData(model.index(0, 2), "2.0")

    _set_batch_current_and_selected_rows(main_window, current_row=0, selected_rows=[0])

    monkeypatch.setattr(main_window.simulation_controller, "launch_pending_slider_preview_replay", lambda: None)

    qtbot.addWidget(main_window)
    main_window.show()

    slider_b = main_window.findChild(QtWidgets.QSlider, "speciesSlider_B")
    assert slider_b is not None

    press_pos = _slider_handle_center(slider_b)
    qtbot.mousePress(slider_b, QtCore.Qt.LeftButton, pos=press_pos)
    slider_b.setValue(5000)
    qtbot.mouseRelease(slider_b, QtCore.Qt.LeftButton, pos=press_pos)
    QtWidgets.QApplication.processEvents()

    commit_btn = main_window.findChild(QtWidgets.QPushButton, "commitSliderOverridesButton")
    reset_btn = main_window.findChild(QtWidgets.QPushButton, "resetSliderOverridesButton")
    assert commit_btn is not None
    assert reset_btn is not None

    before = main_window._preview_session.preview_initials_for_row(0, main_window.batch_initials_for_row(0))
    assert "B" in before
    assert float(before["B"]) == pytest.approx(5.0, rel=1e-6, abs=1e-9)
    assert main_window._preview_session.has_dirty_transaction() is True
    assert commit_btn.isEnabled() is True
    assert reset_btn.isEnabled() is True

    main_window._sync_batch_species_columns(["A", "C"])
    QtWidgets.QApplication.processEvents()

    after = main_window._preview_session.preview_initials_for_row(0, main_window.batch_initials_for_row(0))
    assert "B" not in after
    assert main_window.findChild(QtWidgets.QSlider, "speciesSlider_B") is None
    assert main_window.findChild(QtWidgets.QSlider, "speciesSlider_C") is not None
    assert main_window._preview_session.has_dirty_transaction() is False
    assert commit_btn.isEnabled() is False
    assert reset_btn.isEnabled() is False

def test_species_set_change_clears_active_overlay_display_state_after_pruning(main_window, qtbot, monkeypatch):
    main_window._batch_model.set_species(["A", "B"])
    model = main_window._batch_model
    assert model.setData(model.index(0, 1), "1.0")
    assert model.setData(model.index(0, 2), "2.0")

    _set_batch_current_and_selected_rows(main_window, current_row=0, selected_rows=[0])

    monkeypatch.setattr(main_window.simulation_controller, "launch_pending_slider_preview_replay", lambda: None)

    display_calls: list[dict[str, object]] = []

    def _display(**kwargs):
        display_calls.append(dict(kwargs))
        return SimpleNamespace(displayed=True, reason=None)

    monkeypatch.setattr(main_window.results_controller, "display_cached_batch_selection_outcome", _display)

    qtbot.addWidget(main_window)
    main_window.show()

    slider_b = main_window.findChild(QtWidgets.QSlider, "speciesSlider_B")
    assert slider_b is not None

    press_pos = _slider_handle_center(slider_b)
    qtbot.mousePress(slider_b, QtCore.Qt.LeftButton, pos=press_pos)
    slider_b.setValue(5000)
    qtbot.mouseRelease(slider_b, QtCore.Qt.LeftButton, pos=press_pos)
    QtWidgets.QApplication.processEvents()

    cache = main_window.simulation_controller.batch_cache
    current_set_id = str(main_window._batch_store.set_id_for_row(0) or "set1")
    cache.active_cache_key = "species-overlay-cache"
    cache.active_cache_preview_token = main_window._preview_session.preview_batch_cache_token([0])
    cache.active_cache_preview_scope_set_ids = (current_set_id,)
    cache.active_batch_set = "set1"
    cache.active_batch_set_id = current_set_id
    cache.last_display_selection = [current_set_id]

    main_window._sync_batch_species_columns(["A", "C"])
    QtWidgets.QApplication.processEvents()

    assert cache.active_cache_key is None
    assert cache.active_cache_preview_token is None
    assert cache.active_cache_preview_scope_set_ids is None
    assert cache.active_batch_set is None
    assert cache.active_batch_set_id is None
    assert cache.last_display_selection == []

    main_window._refresh_batch_display_from_focus_and_shown()
    QtWidgets.QApplication.processEvents()
    assert display_calls == []

def test_species_set_change_clears_old_explicit_cache_even_when_unrelated_row_prune_preserves_active_overlay_scope(
    main_window, qtbot, monkeypatch
):
    main_window._batch_model.set_species(["A", "B"])
    model = main_window._batch_model

    add_btn = main_window.findChild(QtWidgets.QPushButton, "addBatchSetButton")
    assert add_btn is not None
    add_btn.click()
    QtWidgets.QApplication.processEvents()
    assert model.rowCount() >= 2

    assert model.setData(model.index(0, 1), "1.0")
    assert model.setData(model.index(0, 2), "2.0")
    assert model.setData(model.index(1, 1), "0.25")
    assert model.setData(model.index(1, 2), "0.75")

    _set_batch_current_and_selected_rows(main_window, current_row=0, selected_rows=[0])

    monkeypatch.setattr(main_window.simulation_controller, "launch_pending_slider_preview_replay", lambda: None)

    display_calls: list[dict[str, object]] = []

    def _display(**kwargs):
        display_calls.append(dict(kwargs))
        return SimpleNamespace(displayed=True, reason=None)

    monkeypatch.setattr(main_window.results_controller, "display_cached_batch_selection_outcome", _display)

    qtbot.addWidget(main_window)
    main_window.show()

    assert main_window._preview_session.stage_concentration_value_for_rows([0], species="A", value=2.5) is True
    assert main_window._preview_session.stage_concentration_value_for_rows([1], species="B", value=7.5) is True

    cache = main_window.simulation_controller.batch_cache
    active_set_id = str(main_window._batch_store.set_id_for_row(0) or "set1")
    active_preview_token = main_window._preview_session.preview_batch_cache_token([0])
    cache.active_cache_key = "species-overlay-cache"
    cache.active_cache_preview_token = active_preview_token
    cache.active_cache_preview_scope_set_ids = (active_set_id,)
    cache.active_cache_valid_set_ids = (active_set_id,)
    cache.active_batch_set = "set1"
    cache.active_batch_set_id = active_set_id
    cache.last_display_selection = [active_set_id]

    commit_btn = main_window.findChild(QtWidgets.QPushButton, "commitSliderOverridesButton")
    reset_btn = main_window.findChild(QtWidgets.QPushButton, "resetSliderOverridesButton")
    assert commit_btn is not None
    assert reset_btn is not None

    main_window._sync_batch_species_columns(["A", "C"])
    QtWidgets.QApplication.processEvents()

    assert main_window._preview_session.preview_batch_cache_token([0]) == active_preview_token
    assert cache.active_cache_key is None
    assert cache.active_cache_preview_token is None
    assert cache.active_cache_preview_scope_set_ids is None
    assert cache.active_cache_valid_set_ids is None
    assert cache.active_batch_set is None
    assert cache.active_batch_set_id is None
    assert cache.last_display_selection == []
    assert main_window._preview_session.has_dirty_transaction() is True
    assert commit_btn.isEnabled() is True
    assert reset_btn.isEnabled() is True

    main_window._refresh_batch_display_from_focus_and_shown()
    QtWidgets.QApplication.processEvents()
    assert display_calls == []

def test_species_set_change_preserves_fresh_explicit_cache_during_post_run_sync(
    main_window, qtbot, monkeypatch
):
    main_window._batch_model.set_species(["A", "B"])
    model = main_window._batch_model

    add_btn = main_window.findChild(QtWidgets.QPushButton, "addBatchSetButton")
    assert add_btn is not None
    add_btn.click()
    QtWidgets.QApplication.processEvents()
    assert model.rowCount() >= 2

    assert model.setData(model.index(0, 1), "1.0")
    assert model.setData(model.index(0, 2), "2.0")
    assert model.setData(model.index(1, 1), "0.25")
    assert model.setData(model.index(1, 2), "0.75")

    _set_batch_current_and_selected_rows(main_window, current_row=0, selected_rows=[0])

    monkeypatch.setattr(main_window.simulation_controller, "launch_pending_slider_preview_replay", lambda: None)

    display_calls: list[dict[str, object]] = []

    def _display(**kwargs):
        display_calls.append(dict(kwargs))
        return SimpleNamespace(displayed=True, reason=None)

    monkeypatch.setattr(main_window.results_controller, "display_cached_batch_selection_outcome", _display)

    qtbot.addWidget(main_window)
    main_window.show()

    assert main_window._preview_session.stage_concentration_value_for_rows([0], species="A", value=2.5) is True
    assert main_window._preview_session.stage_concentration_value_for_rows([1], species="B", value=7.5) is True

    cache = main_window.simulation_controller.batch_cache
    active_set_id = str(main_window._batch_store.set_id_for_row(0) or "set1")
    active_preview_token = main_window._preview_session.preview_batch_cache_token([0])
    cache.active_cache_key = "fresh-current-cache"
    cache.active_cache_preview_token = active_preview_token
    cache.active_cache_preview_scope_set_ids = (active_set_id,)
    cache.active_cache_valid_set_ids = (active_set_id,)
    cache.active_batch_set = "set1"
    cache.active_batch_set_id = active_set_id
    cache.last_display_selection = [active_set_id]

    commit_btn = main_window.findChild(QtWidgets.QPushButton, "commitSliderOverridesButton")
    reset_btn = main_window.findChild(QtWidgets.QPushButton, "resetSliderOverridesButton")
    assert commit_btn is not None
    assert reset_btn is not None

    main_window._sync_batch_species_columns(["A", "C"], preserve_active_cache=True)
    QtWidgets.QApplication.processEvents()

    assert main_window._preview_session.preview_batch_cache_token([0]) == active_preview_token
    assert cache.active_cache_key == "fresh-current-cache"
    assert cache.active_cache_preview_token == active_preview_token
    assert cache.active_cache_preview_scope_set_ids == (active_set_id,)
    assert cache.active_cache_valid_set_ids == (active_set_id,)
    assert cache.active_batch_set == "set1"
    assert cache.active_batch_set_id == active_set_id
    assert cache.last_display_selection == [active_set_id]
    assert main_window._preview_session.has_dirty_transaction() is True
    assert commit_btn.isEnabled() is True
    assert reset_btn.isEnabled() is True

    assert main_window._batch_model.set_row_shown(1, False) is True
    QtWidgets.QApplication.processEvents()
    assert len(display_calls) == 1
    assert display_calls[0]["cache_key"] == "fresh-current-cache"

def test_species_set_change_clears_fresh_explicit_cache_during_post_run_sync_when_active_token_changes(
    main_window, qtbot, monkeypatch
):
    main_window._batch_model.set_species(["A", "B"])
    model = main_window._batch_model

    add_btn = main_window.findChild(QtWidgets.QPushButton, "addBatchSetButton")
    assert add_btn is not None
    add_btn.click()
    QtWidgets.QApplication.processEvents()
    assert model.rowCount() >= 2

    assert model.setData(model.index(0, 1), "1.0")
    assert model.setData(model.index(0, 2), "2.0")
    assert model.setData(model.index(1, 1), "0.25")
    assert model.setData(model.index(1, 2), "0.75")

    _set_batch_current_and_selected_rows(main_window, current_row=0, selected_rows=[0])

    monkeypatch.setattr(main_window.simulation_controller, "launch_pending_slider_preview_replay", lambda: None)

    display_calls: list[dict[str, object]] = []

    def _display(**kwargs):
        display_calls.append(dict(kwargs))
        return SimpleNamespace(displayed=True, reason=None)

    monkeypatch.setattr(main_window.results_controller, "display_cached_batch_selection_outcome", _display)

    qtbot.addWidget(main_window)
    main_window.show()

    assert main_window._preview_session.stage_concentration_value_for_rows([0], species="A", value=2.5) is True
    assert main_window._preview_session.stage_concentration_value_for_rows([0], species="B", value=5.0) is True
    assert main_window._preview_session.stage_concentration_value_for_rows([1], species="A", value=7.5) is True

    cache = main_window.simulation_controller.batch_cache
    active_set_id = str(main_window._batch_store.set_id_for_row(0) or "set1")
    active_preview_token = main_window._preview_session.preview_batch_cache_token([0])
    cache.active_cache_key = "fresh-current-cache"
    cache.active_cache_preview_token = active_preview_token
    cache.active_cache_preview_scope_set_ids = (active_set_id,)
    cache.active_cache_valid_set_ids = (active_set_id,)
    cache.active_batch_set = "set1"
    cache.active_batch_set_id = active_set_id
    cache.last_display_selection = [active_set_id]

    assert "B" in main_window._preview_session.preview_initials_for_row(0, main_window.batch_initials_for_row(0))

    main_window._sync_batch_species_columns(["A", "C"], preserve_active_cache=True)
    QtWidgets.QApplication.processEvents()

    assert main_window._preview_session.preview_batch_cache_token([0]) != active_preview_token
    assert "B" not in main_window._preview_session.preview_initials_for_row(0, main_window.batch_initials_for_row(0))
    assert cache.active_cache_key is None
    assert cache.active_cache_preview_token is None
    assert cache.active_cache_preview_scope_set_ids is None
    assert cache.active_cache_valid_set_ids is None
    assert cache.active_batch_set is None
    assert cache.active_batch_set_id is None
    assert cache.last_display_selection == []

    main_window._refresh_batch_display_from_focus_and_shown()
    QtWidgets.QApplication.processEvents()
    assert display_calls == []

def test_species_set_change_narrows_active_valid_subset_after_partial_post_run_prune(
    main_window, qtbot, monkeypatch
):
    main_window._batch_model.set_species(["A", "B"])
    model = main_window._batch_model

    add_btn = main_window.findChild(QtWidgets.QPushButton, "addBatchSetButton")
    assert add_btn is not None
    add_btn.click()
    QtWidgets.QApplication.processEvents()
    assert model.rowCount() >= 2

    assert model.setData(model.index(0, 1), "1.0")
    assert model.setData(model.index(0, 2), "2.0")
    assert model.setData(model.index(1, 1), "0.25")
    assert model.setData(model.index(1, 2), "0.75")

    monkeypatch.setattr(main_window.simulation_controller, "launch_pending_slider_preview_replay", lambda: None)

    qtbot.addWidget(main_window)
    main_window.show()

    set0_id = str(main_window._batch_store.set_id_for_row(0) or "set1")
    set1_id = str(main_window._batch_store.set_id_for_row(1) or "set2")
    _set_batch_current_and_selected_rows(main_window, current_row=0, selected_rows=[0, 1])

    assert main_window._preview_session.stage_concentration_value_for_rows([0], species="A", value=2.5) is True
    assert main_window._preview_session.stage_concentration_value_for_rows([1], species="B", value=7.5) is True

    cache_key = "fresh-partial-cache"
    cache = main_window.simulation_controller.batch_cache
    cache.active_cache_key = cache_key
    cache.active_cache_preview_token = main_window._preview_session.preview_batch_cache_token([0, 1])
    cache.active_cache_preview_scope_set_ids = (set0_id, set1_id)
    cache.active_cache_valid_set_ids = (set0_id, set1_id)
    cache.active_batch_set = "set1"
    cache.active_batch_set_id = set0_id
    cache.last_display_selection = [set0_id, set1_id]
    cache.result_cache[f"{cache_key}::{set0_id}"] = {
        "t": np.asarray([0.0, 1.0], dtype=float),
        "series": {"A": np.asarray([4.0, 3.5], dtype=float)},
        "algebra_scalars": {},
    }
    cache.result_cache[f"{cache_key}::{set1_id}"] = {
        "t": np.asarray([0.0, 1.0], dtype=float),
        "series": {"A": np.asarray([8.0, 7.5], dtype=float)},
        "algebra_scalars": {},
    }

    main_window._sync_batch_species_columns(["A", "C"], preserve_active_cache=True)
    QtWidgets.QApplication.processEvents()

    assert cache.active_cache_key == cache_key
    assert cache.active_cache_preview_scope_set_ids == (set0_id,)
    assert cache.active_cache_preview_token == main_window._preview_session.preview_batch_cache_token([0])
    assert cache.active_cache_valid_set_ids == (set0_id,)

    assert main_window._batch_model.set_row_shown(1, False) is True
    _set_batch_current_and_selected_rows(main_window, current_row=0, selected_rows=[0])
    main_window._refresh_batch_display_from_focus_and_shown()
    QtWidgets.QApplication.processEvents()

    plot = main_window._plot_tabs._main_plot
    assert np.allclose(np.asarray(getattr(plot, "_t", np.array([])), dtype=float), np.asarray([0.0, 1.0], dtype=float))
    assert np.allclose(np.asarray((getattr(plot, "_series", {}) or {}).get("A"), dtype=float), np.asarray([4.0, 3.5]))

def test_hidden_species_state_does_not_leak_into_new_species_set(main_window, qtbot):
    main_window._batch_model.set_species(["A", "B"])
    model = main_window._batch_model
    assert model.setData(model.index(0, 1), "1.0")
    assert model.setData(model.index(0, 2), "2.0")

    _set_batch_current_and_selected_rows(main_window, current_row=0, selected_rows=[0])

    qtbot.addWidget(main_window)
    main_window.show()

    slider_a = main_window.findChild(QtWidgets.QSlider, "speciesSlider_A")
    assert slider_a is not None
    assert slider_a.isVisible() is False

    _set_slider_visibility(main_window, "species", "A", visible=False)
    assert slider_a.isVisible() is False

    main_window._sync_batch_species_columns(["A", "C"])
    QtWidgets.QApplication.processEvents()

    reloaded_slider_a = main_window.findChild(QtWidgets.QSlider, "speciesSlider_A")
    assert reloaded_slider_a is not None
    assert reloaded_slider_a.isVisible() is False
    action = _find_slider_visibility_action(main_window, "species", "A")
    assert action.isChecked() is False

def test_species_set_change_cancels_queued_species_preview_after_pruning(main_window, qtbot, monkeypatch):
    _set_valid_preview_mechanism(main_window)
    main_window._batch_model.set_species(["A", "B"])
    model = main_window._batch_model
    assert model.setData(model.index(0, 1), "1.0")
    assert model.setData(model.index(0, 2), "2.0")

    _set_batch_current_and_selected_rows(main_window, current_row=0, selected_rows=[0])

    preview_runs: list[str] = []
    monkeypatch.setattr(
        main_window.simulation_controller,
        "launch_pending_slider_preview_replay",
        lambda: preview_runs.append("run"),
    )

    qtbot.addWidget(main_window)
    main_window.show()

    assert main_window._preview_session.stage_concentration_value_for_rows([0], species="B", value=5.0) is True
    main_window._queue_species_slider_simulation(label="init:B", delay_ms=80)
    timer = getattr(main_window._preview_session, "_species_slider_update_timer", None)
    assert timer is not None
    assert timer.isActive() is True
    assert _pending_slider_preview_launch(main_window).active is True

    main_window._sync_batch_species_columns(["A", "C"])
    QtWidgets.QApplication.processEvents()

    assert timer.isActive() is False
    assert _pending_slider_preview_launch(main_window).active is False
    qtbot.wait(120)
    QtWidgets.QApplication.processEvents()
    assert preview_runs == []

def test_species_mode_slider_fans_out_to_all_selected_rows(main_window, qtbot, monkeypatch):
    _set_valid_preview_mechanism(main_window)
    main_window._batch_model.set_species(["A", "B"])
    model = main_window._batch_model

    add_btn = main_window.findChild(QtWidgets.QPushButton, "addBatchSetButton")
    assert add_btn is not None
    add_btn.click()
    add_btn.click()
    QtWidgets.QApplication.processEvents()
    assert model.rowCount() >= 3

    assert model.setData(model.index(0, 1), "1.0")
    assert model.setData(model.index(0, 2), "2.0")
    assert model.setData(model.index(1, 1), "0.25")
    assert model.setData(model.index(1, 2), "0.75")
    assert model.setData(model.index(2, 1), "9.0")
    assert model.setData(model.index(2, 2), "4.0")

    _select_batch_rows(main_window, [0, 1])
    main_window.set_slider_edit_target_set_ids(
        [
            str(main_window.batch_set_id_for_row(0) or ""),
            str(main_window.batch_set_id_for_row(1) or ""),
        ]
    )

    calls = {"n": 0}

    def _count_runs():
        calls["n"] += 1

    monkeypatch.setattr(main_window.simulation_controller, "launch_pending_slider_preview_replay", _count_runs)

    qtbot.addWidget(main_window)
    main_window.show()

    slider_a = main_window.findChild(QtWidgets.QSlider, "speciesSlider_A")
    assert slider_a is not None

    press_pos = _slider_handle_center(slider_a)
    qtbot.mousePress(slider_a, QtCore.Qt.LeftButton, pos=press_pos)
    slider_a.setValue(5000)
    qtbot.mouseRelease(slider_a, QtCore.Qt.LeftButton, pos=press_pos)
    QtWidgets.QApplication.processEvents()

    expected_a = 2.5
    staged_row_0 = main_window._preview_session.preview_initials_for_row(0, main_window.batch_initials_for_row(0))
    staged_row_1 = main_window._preview_session.preview_initials_for_row(1, main_window.batch_initials_for_row(1))
    assert float(main_window._batch_store.get_value(0, "A")) == pytest.approx(1.0, rel=1e-6, abs=1e-9)
    assert float(main_window._batch_store.get_value(1, "A")) == pytest.approx(0.25, rel=1e-6, abs=1e-9)
    assert float(main_window._batch_store.get_value(2, "A")) == pytest.approx(9.0, rel=1e-6, abs=1e-9)
    assert float(staged_row_0["A"]) == pytest.approx(expected_a, rel=1e-6, abs=1e-9)
    assert float(staged_row_1["A"]) == pytest.approx(expected_a, rel=1e-6, abs=1e-9)

    qtbot.waitUntil(lambda: calls["n"] >= 1, timeout=1500)

def test_concentration_surface_shows_mixed_value_before_edit_for_multi_selection(main_window, qtbot, monkeypatch):
    _set_valid_preview_mechanism(main_window)
    main_window._batch_model.set_species(["A", "B"])
    model = main_window._batch_model

    add_btn = main_window.findChild(QtWidgets.QPushButton, "addBatchSetButton")
    assert add_btn is not None
    add_btn.click()
    QtWidgets.QApplication.processEvents()
    assert model.rowCount() >= 2

    assert model.setData(model.index(0, 1), "1.0")
    assert model.setData(model.index(0, 2), "2.0")
    assert model.setData(model.index(1, 1), "0.25")
    assert model.setData(model.index(1, 2), "0.75")

    _set_batch_current_and_selected_rows(main_window, current_row=0, selected_rows=[0, 1])
    main_window.set_slider_edit_target_set_ids(
        [
            str(main_window.batch_set_id_for_row(0) or ""),
            str(main_window.batch_set_id_for_row(1) or ""),
        ]
    )

    calls = {"n": 0}

    def _count_runs():
        calls["n"] += 1

    monkeypatch.setattr(main_window.simulation_controller, "launch_pending_slider_preview_replay", _count_runs)

    qtbot.addWidget(main_window)
    main_window.show()

    slider_a = main_window.findChild(QtWidgets.QSlider, "speciesSlider_A")
    assert slider_a is not None

    qtbot.waitUntil(lambda: _species_slider_row(main_window, "A").value_label.text() == "Multiple values", timeout=1500)
    assert _species_slider_row(main_window, "B").value_label.text() == "Multiple values"

    press_pos = _slider_handle_center(slider_a)
    qtbot.mousePress(slider_a, QtCore.Qt.LeftButton, pos=press_pos)
    slider_a.setValue(5000)
    qtbot.mouseRelease(slider_a, QtCore.Qt.LeftButton, pos=press_pos)
    QtWidgets.QApplication.processEvents()

    expected_a = 2.5
    staged_row_0 = main_window._preview_session.preview_initials_for_row(0, main_window.batch_initials_for_row(0))
    staged_row_1 = main_window._preview_session.preview_initials_for_row(1, main_window.batch_initials_for_row(1))
    assert _species_slider_row(main_window, "A").value_label.text() == "2.500"
    assert _species_slider_row(main_window, "B").value_label.text() == "Multiple values"
    assert float(staged_row_0["A"]) == pytest.approx(expected_a, rel=1e-6, abs=1e-9)
    assert float(staged_row_1["A"]) == pytest.approx(expected_a, rel=1e-6, abs=1e-9)

    qtbot.waitUntil(lambda: calls["n"] >= 1, timeout=1500)

def test_target_checkbox_toggle_uses_model_path_and_does_not_move_focus_or_row_selection(
    main_window,
    qtbot,
    monkeypatch,
):
    main_window._batch_model.set_species(["A"])
    model = main_window._batch_model
    table = main_window._batch_table
    assert table is not None

    add_btn = main_window.findChild(QtWidgets.QPushButton, "addBatchSetButton")
    assert add_btn is not None
    add_btn.click()
    add_btn.click()
    QtWidgets.QApplication.processEvents()
    qtbot.addWidget(main_window)
    main_window.show()

    set0_id = str(main_window.batch_set_id_for_row(0) or "")
    set1_id = str(main_window.batch_set_id_for_row(1) or "")
    set2_id = str(main_window.batch_set_id_for_row(2) or "")
    assert set0_id and set1_id and set2_id
    main_window.set_slider_edit_target_set_ids([set0_id, set2_id])

    calls = _track_checkstate_set_data(model, monkeypatch)

    _set_batch_current_and_selected_rows(main_window, current_row=0, selected_rows=[0, 2])
    edit_column = _column_for_header(model, "Slider")
    focus_before = table.currentIndex()
    selected_before = [idx.row() for idx in table.selectionModel().selectedRows(0)]

    target = model.index(1, edit_column)
    table.scrollTo(target)
    qtbot.mouseClick(table.viewport(), QtCore.Qt.LeftButton, pos=_table_cell_center(table, target))
    QtWidgets.QApplication.processEvents()

    assert any(
        row == 1 and column == edit_column and role == int(QtCore.Qt.CheckStateRole)
        for row, column, role, _value in calls
    )
    assert table.currentIndex().row() == focus_before.row()
    assert [idx.row() for idx in table.selectionModel().selectedRows(0)] == selected_before
    assert str(main_window.focused_batch_set_id()) == str(main_window.batch_set_id_for_row(0) or "")
    assert main_window.slider_edit_target_set_ids() == [set0_id, set2_id, set1_id]
    assert main_window.slider_edit_target_set_ids()[:2] == [set0_id, set2_id]
    assert set1_id in main_window.shown_batch_set_ids()

    calls.clear()
    qtbot.mouseClick(table.viewport(), QtCore.Qt.LeftButton, pos=_table_cell_center(table, target))
    QtWidgets.QApplication.processEvents()

    assert any(
        row == 1 and column == edit_column and role == int(QtCore.Qt.CheckStateRole)
        for row, column, role, _value in calls
    )
    assert table.currentIndex().row() == focus_before.row()
    assert [idx.row() for idx in table.selectionModel().selectedRows(0)] == selected_before
    assert str(main_window.focused_batch_set_id()) == str(main_window.batch_set_id_for_row(0) or "")
    assert main_window.slider_edit_target_set_ids() == [set0_id, set2_id]

def test_show_checkbox_toggle_uses_model_path_and_does_not_move_focus_or_row_selection(
    main_window,
    qtbot,
    monkeypatch,
):
    main_window._batch_model.set_species(["A"])
    model = main_window._batch_model
    table = main_window._batch_table
    assert table is not None

    add_btn = main_window.findChild(QtWidgets.QPushButton, "addBatchSetButton")
    assert add_btn is not None
    add_btn.click()
    add_btn.click()
    QtWidgets.QApplication.processEvents()
    qtbot.addWidget(main_window)
    main_window.show()

    set0_id = str(main_window.batch_set_id_for_row(0) or "")
    set2_id = str(main_window.batch_set_id_for_row(2) or "")
    assert set0_id and set2_id
    main_window.set_slider_edit_target_set_ids([set0_id, set2_id])

    calls = _track_checkstate_set_data(model, monkeypatch)

    _set_batch_current_and_selected_rows(main_window, current_row=0, selected_rows=[0, 2])
    show_column = _column_for_header(model, "Show")
    focus_before = table.currentIndex()
    selected_before = [idx.row() for idx in table.selectionModel().selectedRows(0)]
    edit_targets_before = list(main_window.slider_edit_target_set_ids())

    target = model.index(1, show_column)
    table.scrollTo(target)
    click_pos = _table_cell_center(table, target)
    assert table.indexAt(click_pos) == target
    qtbot.mouseClick(table.viewport(), QtCore.Qt.LeftButton, pos=click_pos)
    QtWidgets.QApplication.processEvents()

    assert any(
        row == 1 and column == show_column and role == int(QtCore.Qt.CheckStateRole)
        for row, column, role, _value in calls
    )
    assert table.currentIndex().row() == focus_before.row()
    assert [idx.row() for idx in table.selectionModel().selectedRows(0)] == selected_before
    assert str(main_window.focused_batch_set_id()) == str(main_window.batch_set_id_for_row(0) or "")
    assert str(main_window.batch_set_id_for_row(1) or "") not in main_window.shown_batch_set_ids()
    assert main_window.slider_edit_target_set_ids() == edit_targets_before

    calls.clear()
    qtbot.mouseClick(table.viewport(), QtCore.Qt.LeftButton, pos=click_pos)
    QtWidgets.QApplication.processEvents()

    assert any(
        row == 1 and column == show_column and role == int(QtCore.Qt.CheckStateRole)
        for row, column, role, _value in calls
    )
    assert table.currentIndex().row() == focus_before.row()
    assert [idx.row() for idx in table.selectionModel().selectedRows(0)] == selected_before
    assert str(main_window.focused_batch_set_id()) == str(main_window.batch_set_id_for_row(0) or "")
    assert str(main_window.batch_set_id_for_row(1) or "") in main_window.shown_batch_set_ids()
    assert main_window.slider_edit_target_set_ids() == edit_targets_before

def test_edit_checkbox_click_on_focused_row_toggles_explicit_membership_without_focus_side_effects(
    main_window,
    qtbot,
):
    main_window._batch_model.set_species(["A"])
    model = main_window._batch_model
    table = main_window._batch_table
    assert table is not None

    qtbot.addWidget(main_window)
    main_window.show()

    _set_batch_current_and_selected_rows(main_window, current_row=0, selected_rows=[0])
    edit_column = _column_for_header(model, "Slider")
    focus_before = table.currentIndex()
    selected_before = [idx.row() for idx in table.selectionModel().selectedRows(0)]
    set0_id = str(main_window.batch_set_id_for_row(0) or "")
    assert set0_id

    main_window.set_slider_edit_target_set_ids([])
    assert main_window.slider_edit_target_set_ids() == []

    target = model.index(0, edit_column)
    table.scrollTo(target)
    qtbot.mouseClick(table.viewport(), QtCore.Qt.LeftButton, pos=_table_cell_center(table, target))
    QtWidgets.QApplication.processEvents()

    assert table.currentIndex().row() == focus_before.row()
    assert [idx.row() for idx in table.selectionModel().selectedRows(0)] == selected_before
    assert str(main_window.focused_batch_set_id()) == str(main_window.batch_set_id_for_row(0) or "")
    assert main_window.slider_edit_target_set_ids() == [set0_id]

    qtbot.mouseClick(table.viewport(), QtCore.Qt.LeftButton, pos=_table_cell_center(table, target))
    QtWidgets.QApplication.processEvents()

    assert table.currentIndex().row() == focus_before.row()
    assert [idx.row() for idx in table.selectionModel().selectedRows(0)] == selected_before
    assert str(main_window.focused_batch_set_id()) == str(main_window.batch_set_id_for_row(0) or "")
    assert main_window.slider_edit_target_set_ids() == []
    assert main_window._mechanism_editor._slider_edit_targets_label.text() == "Slider edit targets: set1"

def test_show_checkbox_click_on_focused_row_is_rejected_without_focus_side_effects(main_window, qtbot):
    main_window._batch_model.set_species(["A"])
    model = main_window._batch_model
    table = main_window._batch_table
    assert table is not None

    qtbot.addWidget(main_window)
    main_window.show()

    _set_batch_current_and_selected_rows(main_window, current_row=0, selected_rows=[0])
    show_column = _column_for_header(model, "Show")
    focus_before = table.currentIndex()
    selected_before = [idx.row() for idx in table.selectionModel().selectedRows(0)]
    edit_targets_before = list(main_window.slider_edit_target_set_ids())

    target = model.index(0, show_column)
    table.scrollTo(target)
    qtbot.mouseClick(table.viewport(), QtCore.Qt.LeftButton, pos=_table_cell_center(table, target))
    QtWidgets.QApplication.processEvents()

    assert table.currentIndex().row() == focus_before.row()
    assert [idx.row() for idx in table.selectionModel().selectedRows(0)] == selected_before
    assert str(main_window.focused_batch_set_id()) == str(main_window.batch_set_id_for_row(0) or "")
    assert str(main_window.batch_set_id_for_row(0) or "") in main_window.shown_batch_set_ids()
    assert main_window.slider_edit_target_set_ids() == edit_targets_before

def test_batch_table_visual_order_places_state_controls_before_set_name(main_window):
    main_window._batch_model.set_species(["A", "B"])
    table = main_window._batch_table
    model = main_window._batch_model
    assert table is not None

    assert _visual_headers(table, model)[:5] == ["Slider", "Show", "Set Name", "A (M)", "B (M)"]
    assert main_window._mechanism_editor._slider_edit_targets_label.text() == "Slider edit targets: set1"

def test_focus_change_preserves_explicit_target_membership_without_accumulating_focus(main_window):
    main_window._batch_model.set_species(["A"])
    add_btn = main_window.findChild(QtWidgets.QPushButton, "addBatchSetButton")
    assert add_btn is not None
    add_btn.click()
    add_btn.click()
    QtWidgets.QApplication.processEvents()

    set0_id = str(main_window.batch_set_id_for_row(0) or "")
    set1_id = str(main_window.batch_set_id_for_row(1) or "")
    set2_id = str(main_window.batch_set_id_for_row(2) or "")
    assert set0_id and set1_id and set2_id

    _set_batch_current_and_selected_rows(main_window, current_row=0, selected_rows=[0])
    main_window.set_slider_edit_target_set_ids([set0_id, set2_id])
    _set_batch_current_and_selected_rows(main_window, current_row=1, selected_rows=[1])
    QtWidgets.QApplication.processEvents()

    assert main_window.slider_edit_target_set_ids() == [set0_id, set2_id]
    assert main_window._batch_model.data(
        main_window._batch_model.index(0, main_window._batch_model.edit_target_column()),
        QtCore.Qt.CheckStateRole,
    ) == QtCore.Qt.Checked
    assert main_window._batch_model.data(
        main_window._batch_model.index(1, main_window._batch_model.edit_target_column()),
        QtCore.Qt.CheckStateRole,
    ) == QtCore.Qt.Unchecked
    assert main_window._batch_model.data(
        main_window._batch_model.index(2, main_window._batch_model.edit_target_column()),
        QtCore.Qt.CheckStateRole,
    ) == QtCore.Qt.Checked
    assert main_window._mechanism_editor._slider_edit_targets_label.text() == "Slider edit targets: set2 + 2 explicit"

def test_focused_unchecked_target_row_shows_row_level_focus_marker_and_stages_only_effective_targets(main_window):
    _set_valid_preview_mechanism(main_window)
    main_window._batch_model.set_species(["A"])
    add_btn = main_window.findChild(QtWidgets.QPushButton, "addBatchSetButton")
    assert add_btn is not None
    add_btn.click()
    add_btn.click()
    QtWidgets.QApplication.processEvents()

    owner = main_window._preview_session
    owner.sync_committed_slider_values({"k1": 1.0})

    set0_id = str(main_window.batch_set_id_for_row(0) or "")
    set1_id = str(main_window.batch_set_id_for_row(1) or "")
    set2_id = str(main_window.batch_set_id_for_row(2) or "")
    assert set0_id and set1_id and set2_id

    main_window.set_slider_edit_target_set_ids([set2_id])
    _set_batch_current_and_selected_rows(main_window, current_row=1, selected_rows=[0])
    QtWidgets.QApplication.processEvents()

    target_column = main_window._batch_model.edit_target_column()
    focused_target_index = main_window._batch_model.index(1, target_column)
    explicit_target_index = main_window._batch_model.index(2, target_column)

    assert main_window.slider_edit_target_set_ids() == [set2_id]
    assert main_window._effective_slider_edit_target_set_ids() == [set1_id, set2_id]
    assert main_window._batch_model.data(focused_target_index, QtCore.Qt.CheckStateRole) == QtCore.Qt.Unchecked
    assert main_window._batch_model.data(focused_target_index, QtCore.Qt.DisplayRole) == "focus"
    assert main_window._batch_model.data(explicit_target_index, QtCore.Qt.CheckStateRole) == QtCore.Qt.Checked
    assert main_window._batch_model.data(explicit_target_index, QtCore.Qt.DisplayRole) == ""
    assert main_window._mechanism_editor._slider_edit_targets_label.text() == "Slider edit targets: set2 + 1 explicit"

    owner.on_variable_changed("k1", 2.0)

    assert owner.local_mechanism_workspace(set0_id) == {}
    assert owner.local_mechanism_workspace(set1_id) == {"k1": pytest.approx(2.0)}
    assert owner.local_mechanism_workspace(set2_id) == {"k1": pytest.approx(2.0)}
    owner.stop_variable_update_timer()

    _set_batch_current_and_selected_rows(main_window, current_row=0, selected_rows=[0])
    QtWidgets.QApplication.processEvents()

    assert main_window._effective_slider_edit_target_set_ids() == [set0_id, set2_id]
    assert main_window._batch_model.data(focused_target_index, QtCore.Qt.DisplayRole) == ""
    assert main_window._batch_model.data(main_window._batch_model.index(0, target_column), QtCore.Qt.DisplayRole) == "focus"
    assert main_window._batch_model.data(explicit_target_index, QtCore.Qt.CheckStateRole) == QtCore.Qt.Checked

def test_batch_selection_and_current_handlers_are_single_wired_on_startup(main_window, monkeypatch):
    main_window._batch_model.set_species(["A"])
    add_btn = main_window.findChild(QtWidgets.QPushButton, "addBatchSetButton")
    assert add_btn is not None
    add_btn.click()
    QtWidgets.QApplication.processEvents()

    table = main_window._batch_table
    model = main_window._batch_model
    assert table is not None

    _set_batch_current_and_selected_rows(main_window, current_row=0, selected_rows=[0])
    QtWidgets.QApplication.processEvents()

    refresh_calls: list[str] = []
    summary_calls: list[str] = []
    monkeypatch.setattr(
        main_window,
        "_refresh_batch_display_from_focus_and_shown",
        lambda: refresh_calls.append("refresh"),
    )
    monkeypatch.setattr(
        main_window,
        "_refresh_slider_edit_targets_summary",
        lambda: summary_calls.append("summary"),
    )

    table.setCurrentIndex(model.index(1, 0))
    QtWidgets.QApplication.processEvents()

    assert refresh_calls == ["refresh"]
    assert summary_calls == ["summary"]

    _set_batch_current_and_selected_rows(main_window, current_row=1, selected_rows=[1])
    QtWidgets.QApplication.processEvents()

    row_control_calls: list[str] = []
    monkeypatch.setattr(
        main_window,
        "_update_batch_row_controls_state",
        lambda: row_control_calls.append("controls"),
    )

    sel = table.selectionModel()
    assert sel is not None
    sel.select(
        model.index(0, 0),
        QtCore.QItemSelectionModel.ClearAndSelect | QtCore.QItemSelectionModel.Rows,
    )
    QtWidgets.QApplication.processEvents()

    assert table.currentIndex().row() == 1
    assert row_control_calls == ["controls"]

def test_species_mode_slider_targets_explicit_edit_rows_not_selected_rows(main_window, qtbot, monkeypatch):
    _set_valid_preview_mechanism(main_window)
    main_window._batch_model.set_species(["A", "B"])
    model = main_window._batch_model

    add_btn = main_window.findChild(QtWidgets.QPushButton, "addBatchSetButton")
    assert add_btn is not None
    add_btn.click()
    add_btn.click()
    QtWidgets.QApplication.processEvents()
    assert model.rowCount() >= 3

    assert model.setData(model.index(0, 1), "1.0")
    assert model.setData(model.index(0, 2), "2.0")
    assert model.setData(model.index(1, 1), "0.25")
    assert model.setData(model.index(1, 2), "0.75")
    assert model.setData(model.index(2, 1), "9.0")
    assert model.setData(model.index(2, 2), "4.0")

    _set_batch_current_and_selected_rows(main_window, current_row=0, selected_rows=[0, 1])
    main_window.set_slider_edit_target_set_ids(
        [
            str(main_window.batch_set_id_for_row(0) or ""),
            str(main_window.batch_set_id_for_row(1) or ""),
        ]
    )
    main_window.set_slider_edit_target_set_ids(
        [
            str(main_window.batch_set_id_for_row(0) or ""),
            str(main_window.batch_set_id_for_row(2) or ""),
        ]
    )

    calls = {"n": 0}
    monkeypatch.setattr(main_window.simulation_controller, "launch_pending_slider_preview_replay", lambda: calls.__setitem__("n", calls["n"] + 1))

    qtbot.addWidget(main_window)
    main_window.show()

    slider_a = main_window.findChild(QtWidgets.QSlider, "speciesSlider_A")
    assert slider_a is not None
    qtbot.waitUntil(lambda: _species_slider_row(main_window, "A").value_label.text() == "Multiple values", timeout=1500)

    press_pos = _slider_handle_center(slider_a)
    qtbot.mousePress(slider_a, QtCore.Qt.LeftButton, pos=press_pos)
    slider_a.setValue(5000)
    qtbot.mouseRelease(slider_a, QtCore.Qt.LeftButton, pos=press_pos)
    QtWidgets.QApplication.processEvents()

    staged_row_0 = main_window._preview_session.preview_initials_for_row(0, main_window.batch_initials_for_row(0))
    staged_row_1 = main_window._preview_session.preview_initials_for_row(1, main_window.batch_initials_for_row(1))
    staged_row_2 = main_window._preview_session.preview_initials_for_row(2, main_window.batch_initials_for_row(2))
    assert float(staged_row_0["A"]) == pytest.approx(2.5, rel=1e-6, abs=1e-9)
    assert float(staged_row_1["A"]) == pytest.approx(0.25, rel=1e-6, abs=1e-9)
    assert float(staged_row_2["A"]) == pytest.approx(2.5, rel=1e-6, abs=1e-9)

    qtbot.waitUntil(lambda: calls["n"] >= 1, timeout=1500)

def test_concentration_surface_uses_focused_row_as_display_source_when_selection_diverges(
    main_window,
    qtbot,
    monkeypatch,
):
    _set_valid_preview_mechanism(main_window)
    main_window._batch_model.set_species(["A", "B"])
    model = main_window._batch_model

    add_btn = main_window.findChild(QtWidgets.QPushButton, "addBatchSetButton")
    assert add_btn is not None
    add_btn.click()
    QtWidgets.QApplication.processEvents()
    assert model.rowCount() >= 2

    assert model.setData(model.index(0, 1), "1.0")
    assert model.setData(model.index(0, 2), "10.0")
    assert model.setData(model.index(1, 1), "8.0")
    assert model.setData(model.index(1, 2), "1.0")

    calls = {"n": 0}

    def _count_runs():
        calls["n"] += 1

    monkeypatch.setattr(main_window.simulation_controller, "launch_pending_slider_preview_replay", _count_runs)

    qtbot.addWidget(main_window)
    main_window.show()
    _set_batch_current_and_selected_rows(main_window, current_row=0, selected_rows=[1])
    main_window.set_slider_edit_target_set_ids([str(main_window.batch_set_id_for_row(0) or "")])
    QtWidgets.QApplication.processEvents()

    slider_a = main_window.findChild(QtWidgets.QSlider, "speciesSlider_A")
    assert slider_a is not None
    qtbot.waitUntil(lambda: _species_slider_row(main_window, "A").value_label.text() == "1.000", timeout=1500)

    press_pos = _slider_handle_center(slider_a)
    qtbot.mousePress(slider_a, QtCore.Qt.LeftButton, pos=press_pos)
    slider_a.setValue(5000)
    qtbot.mouseRelease(slider_a, QtCore.Qt.LeftButton, pos=press_pos)
    QtWidgets.QApplication.processEvents()

    staged = main_window._preview_session.preview_initials_for_row(0, main_window.batch_initials_for_row(0))
    assert float(main_window._batch_store.get_value(0, "A")) == pytest.approx(1.0, rel=1e-6, abs=1e-9)
    assert float(main_window._batch_store.get_value(1, "A")) == pytest.approx(8.0, rel=1e-6, abs=1e-9)
    assert float(staged["A"]) == pytest.approx(10.0, rel=1e-6, abs=1e-9)
    qtbot.waitUntil(lambda: calls["n"] >= 1, timeout=1500)

def test_species_mode_reset_restores_all_explicit_edit_targets_even_if_only_one_is_focused(
    main_window,
    qtbot,
    monkeypatch,
):
    _set_valid_preview_mechanism(main_window)
    main_window._batch_model.set_species(["A", "B"])
    model = main_window._batch_model

    add_btn = main_window.findChild(QtWidgets.QPushButton, "addBatchSetButton")
    assert add_btn is not None
    add_btn.click()
    QtWidgets.QApplication.processEvents()
    assert model.rowCount() >= 2

    assert model.setData(model.index(0, 1), "1.0")
    assert model.setData(model.index(0, 2), "2.0")
    assert model.setData(model.index(1, 1), "0.25")
    assert model.setData(model.index(1, 2), "0.75")

    _set_batch_current_and_selected_rows(main_window, current_row=0, selected_rows=[0, 1])
    main_window.set_slider_edit_target_set_ids(
        [
            str(main_window.batch_set_id_for_row(0) or ""),
            str(main_window.batch_set_id_for_row(1) or ""),
        ]
    )

    calls = {"n": 0}

    def _count_runs():
        calls["n"] += 1

    monkeypatch.setattr(main_window.simulation_controller, "launch_pending_slider_preview_replay", _count_runs)

    qtbot.addWidget(main_window)
    main_window.show()

    slider_a = main_window.findChild(QtWidgets.QSlider, "speciesSlider_A")
    assert slider_a is not None

    press_pos = _slider_handle_center(slider_a)
    qtbot.mousePress(slider_a, QtCore.Qt.LeftButton, pos=press_pos)
    slider_a.setValue(5000)
    qtbot.mouseRelease(slider_a, QtCore.Qt.LeftButton, pos=press_pos)
    QtWidgets.QApplication.processEvents()

    staged_row_0 = main_window._preview_session.preview_initials_for_row(0, main_window.batch_initials_for_row(0))
    staged_row_1 = main_window._preview_session.preview_initials_for_row(1, main_window.batch_initials_for_row(1))
    assert float(main_window._batch_store.get_value(0, "A")) == pytest.approx(1.0, rel=1e-6, abs=1e-9)
    assert float(main_window._batch_store.get_value(1, "A")) == pytest.approx(0.25, rel=1e-6, abs=1e-9)
    assert float(staged_row_0["A"]) == pytest.approx(2.5, rel=1e-6, abs=1e-9)
    assert float(staged_row_1["A"]) == pytest.approx(2.5, rel=1e-6, abs=1e-9)

    reset_btn = main_window.findChild(QtWidgets.QPushButton, "resetSliderOverridesButton")
    assert reset_btn is not None
    qtbot.mouseClick(reset_btn, QtCore.Qt.LeftButton)
    QtWidgets.QApplication.processEvents()

    assert float(main_window._batch_store.get_value(0, "A")) == pytest.approx(1.0, rel=1e-6, abs=1e-9)
    assert float(main_window._batch_store.get_value(0, "B")) == pytest.approx(2.0, rel=1e-6, abs=1e-9)
    assert float(main_window._batch_store.get_value(1, "A")) == pytest.approx(0.25, rel=1e-6, abs=1e-9)
    assert float(main_window._batch_store.get_value(1, "B")) == pytest.approx(0.75, rel=1e-6, abs=1e-9)
    qtbot.waitUntil(lambda: calls["n"] >= 2, timeout=1500)

def test_species_slider_queue_marks_pending_slider_simulation(main_window, qtbot):
    _set_valid_preview_mechanism(main_window)
    qtbot.addWidget(main_window)
    pending_launch = _pending_slider_preview_launch(main_window)
    assert pending_launch.active is False
    assert pending_launch.request_id is None

    main_window._queue_species_slider_simulation(label="init:A", delay_ms=500)

    pending_launch = _pending_slider_preview_launch(main_window)
    assert pending_launch.active is True
    assert isinstance(pending_launch.request_id, int)
    timer = getattr(main_window._preview_session, "_species_slider_update_timer", None)
    assert timer is not None
    assert timer.isActive()
    timer.stop()

def test_species_mode_rebinds_after_project_apply(main_window, qtbot):
    main_window._batch_model.set_species(["A", "B"])
    model = main_window._batch_model

    idx_a = model.index(0, 1)
    idx_b = model.index(0, 2)
    assert model.setData(idx_a, "1.0")
    assert model.setData(idx_b, "2.0")

    table = main_window._batch_table
    assert table is not None
    table.setCurrentIndex(model.index(0, 0))
    sel = table.selectionModel()
    assert sel is not None
    sel.clearSelection()
    sel.select(model.index(0, 0), QtCore.QItemSelectionModel.Select | QtCore.QItemSelectionModel.Rows)

    qtbot.addWidget(main_window)
    main_window.show()

    assert main_window.findChild(QtWidgets.QSlider, "speciesSlider_A") is not None
    assert main_window.findChild(QtWidgets.QSlider, "speciesSlider_B") is not None

    restored_store = BatchInitialConditionsStore()
    restored_store.set_species(["C", "D"])
    restored_store.set_value(0, "C", "3.0")
    restored_store.set_value(0, "D", "1.0")

    payload = dict(main_window._serialize_project_state())
    payload["batch_initial_conditions"] = restored_store.as_serializable()

    main_window._apply_project_payload(payload, record_undo=False)
    QtWidgets.QApplication.processEvents()

    slider_a = main_window.findChild(QtWidgets.QSlider, "speciesSlider_A")
    slider_b = main_window.findChild(QtWidgets.QSlider, "speciesSlider_B")
    slider_c = main_window.findChild(QtWidgets.QSlider, "speciesSlider_C")
    slider_d = main_window.findChild(QtWidgets.QSlider, "speciesSlider_D")

    assert slider_a is None
    assert slider_b is None
    assert slider_c is not None
    assert slider_d is not None
    assert table.currentIndex().isValid()
    assert float(main_window._batch_store.get_value(0, "C")) == pytest.approx(3.0, rel=1e-6, abs=1e-9)
    assert float(main_window._batch_store.get_value(0, "D")) == pytest.approx(1.0, rel=1e-6, abs=1e-9)

    press_pos = _slider_handle_center(slider_c)
    qtbot.mousePress(slider_c, QtCore.Qt.LeftButton, pos=press_pos)
    slider_c.setValue(5000)
    qtbot.mouseRelease(slider_c, QtCore.Qt.LeftButton, pos=press_pos)
    QtWidgets.QApplication.processEvents()

    staged = main_window._preview_session.preview_initials_for_row(0, main_window.batch_initials_for_row(0))
    assert float(main_window._batch_store.get_value(0, "C")) == pytest.approx(3.0, rel=1e-6, abs=1e-9)
    assert float(staged["C"]) == pytest.approx(7.5, rel=1e-6, abs=1e-9)

def test_project_apply_disconnects_stale_batch_semantics_signals(main_window, qtbot, monkeypatch):
    main_window._batch_model.set_species(["A"])
    add_btn = main_window.findChild(QtWidgets.QPushButton, "addBatchSetButton")
    assert add_btn is not None
    add_btn.click()
    QtWidgets.QApplication.processEvents()

    old_model = main_window._batch_model
    old_set0_id = str(main_window.batch_set_id_for_row(0) or "")
    old_set1_id = str(main_window.batch_set_id_for_row(1) or "")
    assert old_set0_id and old_set1_id

    qtbot.addWidget(main_window)
    main_window.show()

    restored_store = BatchInitialConditionsStore.from_serializable(
        {
            "sets": [
                {
                    "set_id": "replacement-0",
                    "name": "restored-1",
                    "shown": True,
                    "values": {"A": "1.0"},
                },
                {
                    "set_id": "replacement-1",
                    "name": "restored-2",
                    "shown": True,
                    "values": {"A": "2.0"},
                },
            ],
            "visible_species": ["A"],
        }
    )
    payload = dict(main_window._serialize_project_state())
    payload["batch_initial_conditions"] = restored_store.as_serializable()
    main_window._apply_project_payload(payload, record_undo=False)
    QtWidgets.QApplication.processEvents()

    panel = main_window._mechanism_editor.species_sliders_widget()
    assert panel is not None
    refresh_calls: list[str] = []
    rebuild_calls: list[str] = []
    monkeypatch.setattr(
        main_window,
        "_refresh_batch_display_from_focus_and_shown",
        lambda: refresh_calls.append("refresh"),
    )
    monkeypatch.setattr(panel, "rebuild_from_current_row", lambda: rebuild_calls.append("rebuild"))

    assert old_model.set_row_shown(1, False) is True
    assert old_model.set_slider_edit_target_set_ids([old_set0_id, old_set1_id]) is True
    QtWidgets.QApplication.processEvents()
    assert refresh_calls == []
    assert rebuild_calls == []

    current_model = main_window._batch_model
    current_set0_id = str(main_window.batch_set_id_for_row(0) or "")
    current_set1_id = str(main_window.batch_set_id_for_row(1) or "")
    assert current_set0_id and current_set1_id

    assert current_model.set_slider_edit_target_set_ids([current_set0_id, current_set1_id]) is True
    QtWidgets.QApplication.processEvents()
    assert rebuild_calls == ["rebuild"]

    assert current_model.set_row_shown(1, False) is True
    QtWidgets.QApplication.processEvents()
    assert refresh_calls == ["refresh"]

def test_species_mode_attach_uses_unified_transaction_clear_for_reused_set_id(main_window, qtbot, monkeypatch):
    main_window._batch_model.set_species(["A", "B"])
    model = main_window._batch_model

    assert model.setData(model.index(0, 1), "1.0")
    assert model.setData(model.index(0, 2), "2.0")

    _set_batch_current_and_selected_rows(main_window, current_row=0, selected_rows=[0])

    monkeypatch.setattr(main_window.simulation_controller, "launch_pending_slider_preview_replay", lambda: None)

    qtbot.addWidget(main_window)
    main_window.show()

    slider_a = main_window.findChild(QtWidgets.QSlider, "speciesSlider_A")
    assert slider_a is not None

    press_pos = _slider_handle_center(slider_a)
    qtbot.mousePress(slider_a, QtCore.Qt.LeftButton, pos=press_pos)
    slider_a.setValue(5000)
    qtbot.mouseRelease(slider_a, QtCore.Qt.LeftButton, pos=press_pos)
    QtWidgets.QApplication.processEvents()

    original_set_id = str(main_window._batch_store.set_id_for_row(0))
    replacement_store = BatchInitialConditionsStore.from_serializable(
        {
            "sets": [
                {
                    "set_id": original_set_id,
                    "name": "set1",
                    "values": {"A": "7.0", "B": "1.5"},
                }
            ],
            "visible_species": ["A", "B"],
        }
    )
    replacement_model = BatchInitialConditionsTableModel(replacement_store, parent=main_window)
    main_window._batch_store = replacement_store
    main_window._batch_model = replacement_model
    main_window._batch_table.setModel(replacement_model)
    _set_batch_current_and_selected_rows(main_window, current_row=0, selected_rows=[0])

    panel = main_window._mechanism_editor.species_sliders_widget()
    panel.attach(table=main_window._batch_table, model=replacement_model)
    panel.activate()
    QtWidgets.QApplication.processEvents()

    slider_a = main_window.findChild(QtWidgets.QSlider, "speciesSlider_A")
    assert slider_a is not None
    main_window._preview_session.clear_working_transaction()
    assert float(main_window._batch_store.get_value(0, "A")) == pytest.approx(7.0, rel=1e-6, abs=1e-9)

    press_pos = _slider_handle_center(slider_a)
    qtbot.mousePress(slider_a, QtCore.Qt.LeftButton, pos=press_pos)
    slider_a.setValue(0)
    qtbot.mouseRelease(slider_a, QtCore.Qt.LeftButton, pos=press_pos)
    QtWidgets.QApplication.processEvents()

    staged = main_window._preview_session.preview_initials_for_row(0, main_window.batch_initials_for_row(0))
    assert float(main_window._batch_store.get_value(0, "A")) == pytest.approx(7.0, rel=1e-6, abs=1e-9)
    assert float(staged["A"]) == pytest.approx(0.0, rel=1e-6, abs=1e-9)

    reset_btn = main_window.findChild(QtWidgets.QPushButton, "resetSliderOverridesButton")
    assert reset_btn is not None
    qtbot.mouseClick(reset_btn, QtCore.Qt.LeftButton)
    QtWidgets.QApplication.processEvents()

    assert float(main_window._batch_store.get_value(0, "A")) == pytest.approx(7.0, rel=1e-6, abs=1e-9)
    assert float(main_window._batch_store.get_value(0, "B")) == pytest.approx(1.5, rel=1e-6, abs=1e-9)

def test_hiding_species_slider_preserves_hidden_staged_overlays_until_reset(main_window, qtbot, monkeypatch):
    main_window._batch_model.set_species(["A", "B"])
    model = main_window._batch_model

    assert model.setData(model.index(0, 1), "1.0")
    assert model.setData(model.index(0, 2), "2.0")

    _set_batch_current_and_selected_rows(main_window, current_row=0, selected_rows=[0])

    monkeypatch.setattr(main_window.simulation_controller, "launch_pending_slider_preview_replay", lambda: None)
    monkeypatch.setattr(
        main_window,
        "_prompt_slider_transaction_invalidation",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("species mode toggle should not prompt")),
        raising=False,
    )

    qtbot.addWidget(main_window)
    main_window.show()

    slider_a = main_window.findChild(QtWidgets.QSlider, "speciesSlider_A")
    assert slider_a is not None

    press_pos = _slider_handle_center(slider_a)
    qtbot.mousePress(slider_a, QtCore.Qt.LeftButton, pos=press_pos)
    slider_a.setValue(5000)
    qtbot.mouseRelease(slider_a, QtCore.Qt.LeftButton, pos=press_pos)
    QtWidgets.QApplication.processEvents()

    staged = main_window._preview_session.preview_initials_for_row(0, main_window.batch_initials_for_row(0))
    assert float(main_window._batch_store.get_value(0, "A")) == pytest.approx(1.0, rel=1e-6, abs=1e-9)
    assert float(staged["A"]) == pytest.approx(2.5, rel=1e-6, abs=1e-9)
    commit_btn = main_window.findChild(QtWidgets.QPushButton, "commitSliderOverridesButton")
    reset_btn = main_window.findChild(QtWidgets.QPushButton, "resetSliderOverridesButton")
    assert commit_btn is not None
    assert reset_btn is not None
    assert commit_btn.isEnabled() is True
    assert reset_btn.isEnabled() is True

    _set_slider_visibility(main_window, "species", "A", visible=False)
    assert slider_a.isVisible() is False

    hidden = main_window._preview_session.preview_initials_for_row(0, main_window.batch_initials_for_row(0))
    assert float(main_window._batch_store.get_value(0, "A")) == pytest.approx(1.0, rel=1e-6, abs=1e-9)
    assert float(hidden["A"]) == pytest.approx(2.5, rel=1e-6, abs=1e-9)
    assert commit_btn.isEnabled() is True
    assert reset_btn.isEnabled() is True

    reset_btn = main_window.findChild(QtWidgets.QPushButton, "resetSliderOverridesButton")
    assert reset_btn is not None
    qtbot.mouseClick(reset_btn, QtCore.Qt.LeftButton)
    QtWidgets.QApplication.processEvents()

    reset = main_window._preview_session.preview_initials_for_row(0, main_window.batch_initials_for_row(0))
    assert float(main_window._batch_store.get_value(0, "A")) == pytest.approx(1.0, rel=1e-6, abs=1e-9)
    assert float(reset["A"]) == pytest.approx(1.0, rel=1e-6, abs=1e-9)
    assert commit_btn.isEnabled() is False
    assert reset_btn.isEnabled() is False

def test_batch_selection_change_with_hidden_overlay_does_not_prompt(main_window, qtbot, monkeypatch):
    main_window._batch_model.set_species(["A", "B"])
    model = main_window._batch_model

    add_btn = main_window.findChild(QtWidgets.QPushButton, "addBatchSetButton")
    assert add_btn is not None
    add_btn.click()
    QtWidgets.QApplication.processEvents()

    assert model.setData(model.index(0, 1), "1.0")
    assert model.setData(model.index(0, 2), "2.0")
    assert model.setData(model.index(1, 1), "3.0")
    assert model.setData(model.index(1, 2), "4.0")

    _set_batch_current_and_selected_rows(main_window, current_row=0, selected_rows=[0])

    monkeypatch.setattr(main_window.simulation_controller, "launch_pending_slider_preview_replay", lambda: None)
    monkeypatch.setattr(
        main_window,
        "_prompt_slider_transaction_invalidation",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("batch selection change should not prompt")),
        raising=False,
    )

    qtbot.addWidget(main_window)
    main_window.show()

    slider_a = main_window.findChild(QtWidgets.QSlider, "speciesSlider_A")
    assert slider_a is not None

    press_pos = _slider_handle_center(slider_a)
    qtbot.mousePress(slider_a, QtCore.Qt.LeftButton, pos=press_pos)
    slider_a.setValue(5000)
    qtbot.mouseRelease(slider_a, QtCore.Qt.LeftButton, pos=press_pos)
    QtWidgets.QApplication.processEvents()

    _set_slider_visibility(main_window, "species", "A", visible=False)
    _set_batch_current_and_selected_rows(main_window, current_row=1, selected_rows=[1])
    main_window._refresh_batch_display_from_focus_and_shown()
    QtWidgets.QApplication.processEvents()

    hidden = main_window._preview_session.preview_initials_for_row(0, main_window.batch_initials_for_row(0))
    assert float(main_window._batch_store.get_value(0, "A")) == pytest.approx(1.0, rel=1e-6, abs=1e-9)
    assert float(hidden["A"]) == pytest.approx(2.5, rel=1e-6, abs=1e-9)

def test_species_mode_reset_clears_overlay_derived_explicit_cache_selection_state(main_window, qtbot, monkeypatch):
    main_window._batch_model.set_species(["A", "B"])
    model = main_window._batch_model

    assert model.setData(model.index(0, 1), "1.0")
    assert model.setData(model.index(0, 2), "2.0")

    _set_batch_current_and_selected_rows(main_window, current_row=0, selected_rows=[0])

    monkeypatch.setattr(main_window.simulation_controller, "launch_pending_slider_preview_replay", lambda: None)

    display_calls: list[dict[str, object]] = []

    def _display(**kwargs):
        display_calls.append(dict(kwargs))
        return SimpleNamespace(displayed=True, reason=None)

    monkeypatch.setattr(main_window.results_controller, "display_cached_batch_selection_outcome", _display)

    qtbot.addWidget(main_window)
    main_window.show()

    slider_a = main_window.findChild(QtWidgets.QSlider, "speciesSlider_A")
    assert slider_a is not None

    press_pos = _slider_handle_center(slider_a)
    qtbot.mousePress(slider_a, QtCore.Qt.LeftButton, pos=press_pos)
    slider_a.setValue(5000)
    qtbot.mouseRelease(slider_a, QtCore.Qt.LeftButton, pos=press_pos)
    QtWidgets.QApplication.processEvents()

    cache = main_window.simulation_controller.batch_cache
    cache.active_cache_key = "overlay-explicit-cache"
    cache.active_cache_preview_token = main_window._preview_session.preview_batch_cache_token([0])
    cache.active_cache_preview_scope_set_ids = (str(main_window._batch_store.set_id_for_row(0) or "set1"),)
    cache.active_batch_set = "set1"
    cache.active_batch_set_id = str(main_window._batch_store.set_id_for_row(0) or "set1")
    cache.last_display_selection = [str(cache.active_batch_set_id)]

    reset_btn = main_window.findChild(QtWidgets.QPushButton, "resetSliderOverridesButton")
    assert reset_btn is not None
    qtbot.mouseClick(reset_btn, QtCore.Qt.LeftButton)
    QtWidgets.QApplication.processEvents()

    assert cache.active_cache_key is None
    assert cache.active_cache_preview_token is None
    assert cache.active_cache_preview_scope_set_ids is None
    assert cache.active_batch_set is None
    assert cache.active_batch_set_id is None
    assert cache.last_display_selection == []

    main_window._refresh_batch_display_from_focus_and_shown()
    QtWidgets.QApplication.processEvents()

    assert display_calls == []

def test_species_mode_reset_preserves_baseline_explicit_cache_selection_state_when_overlay_was_never_run(
    main_window, qtbot, monkeypatch
):
    main_window._batch_model.set_species(["A", "B"])
    model = main_window._batch_model

    assert model.setData(model.index(0, 1), "1.0")
    assert model.setData(model.index(0, 2), "2.0")

    _set_batch_current_and_selected_rows(main_window, current_row=0, selected_rows=[0])

    monkeypatch.setattr(main_window.simulation_controller, "launch_pending_slider_preview_replay", lambda: None)

    display_calls: list[dict[str, object]] = []

    def _display(**kwargs):
        display_calls.append(dict(kwargs))
        return SimpleNamespace(displayed=True, reason=None)

    monkeypatch.setattr(main_window.results_controller, "display_cached_batch_selection_outcome", _display)

    qtbot.addWidget(main_window)
    main_window.show()

    slider_a = main_window.findChild(QtWidgets.QSlider, "speciesSlider_A")
    assert slider_a is not None

    press_pos = _slider_handle_center(slider_a)
    qtbot.mousePress(slider_a, QtCore.Qt.LeftButton, pos=press_pos)
    slider_a.setValue(5000)
    qtbot.mouseRelease(slider_a, QtCore.Qt.LeftButton, pos=press_pos)
    QtWidgets.QApplication.processEvents()

    cache = main_window.simulation_controller.batch_cache
    cache.active_cache_key = "baseline-explicit-cache"
    cache.active_cache_preview_token = None
    cache.active_cache_preview_scope_set_ids = None
    cache.active_batch_set = "set1"
    cache.active_batch_set_id = str(main_window._batch_store.set_id_for_row(0) or "set1")
    cache.last_display_selection = [str(cache.active_batch_set_id)]

    reset_btn = main_window.findChild(QtWidgets.QPushButton, "resetSliderOverridesButton")
    assert reset_btn is not None
    qtbot.mouseClick(reset_btn, QtCore.Qt.LeftButton)
    QtWidgets.QApplication.processEvents()

    assert cache.active_cache_key == "baseline-explicit-cache"
    assert cache.active_cache_preview_token is None
    assert cache.active_cache_preview_scope_set_ids is None
    assert cache.active_batch_set == "set1"
    assert cache.active_batch_set_id == str(main_window._batch_store.set_id_for_row(0) or "set1")
    assert cache.last_display_selection == [str(cache.active_batch_set_id)]

    main_window._refresh_batch_display_from_focus_and_shown()
    QtWidgets.QApplication.processEvents()

    assert len(display_calls) == 1
    assert display_calls[0]["cache_key"] == "baseline-explicit-cache"

def test_species_mode_reset_clears_subset_scope_overlay_cache_even_with_unrelated_dirty_rows(
    main_window, qtbot, monkeypatch
):
    main_window._batch_model.set_species(["A", "B"])
    model = main_window._batch_model

    add_btn = main_window.findChild(QtWidgets.QPushButton, "addBatchSetButton")
    assert add_btn is not None
    add_btn.click()
    QtWidgets.QApplication.processEvents()

    assert model.setData(model.index(0, 1), "1.0")
    assert model.setData(model.index(0, 2), "2.0")
    assert model.setData(model.index(1, 1), "3.0")
    assert model.setData(model.index(1, 2), "4.0")

    _set_batch_current_and_selected_rows(main_window, current_row=0, selected_rows=[0])

    monkeypatch.setattr(main_window.simulation_controller, "launch_pending_slider_preview_replay", lambda: None)

    display_calls: list[dict[str, object]] = []

    def _display(**kwargs):
        display_calls.append(dict(kwargs))
        return SimpleNamespace(displayed=True, reason=None)

    monkeypatch.setattr(main_window.results_controller, "display_cached_batch_selection_outcome", _display)

    qtbot.addWidget(main_window)
    main_window.show()

    main_window._preview_session.stage_concentration_value_for_rows([0], species="A", value=2.5)
    main_window._preview_session.stage_concentration_value_for_rows([1], species="A", value=7.5)

    set0_id = str(main_window._batch_store.set_id_for_row(0) or "set1")
    cache = main_window.simulation_controller.batch_cache
    cache.active_cache_key = "subset-overlay-explicit-cache"
    cache.active_cache_preview_token = main_window._preview_session.preview_batch_cache_token([0])
    cache.active_cache_preview_scope_set_ids = (set0_id,)
    cache.active_batch_set = "set1"
    cache.active_batch_set_id = set0_id
    cache.last_display_selection = [set0_id]

    reset_btn = main_window.findChild(QtWidgets.QPushButton, "resetSliderOverridesButton")
    assert reset_btn is not None
    qtbot.mouseClick(reset_btn, QtCore.Qt.LeftButton)
    QtWidgets.QApplication.processEvents()

    assert cache.active_cache_key is None
    assert cache.active_cache_preview_token is None
    assert cache.active_cache_preview_scope_set_ids is None

    main_window._refresh_batch_display_from_focus_and_shown()
    QtWidgets.QApplication.processEvents()

    assert display_calls == []

def test_batch_table_auto_fit_on_model_reset_and_minimum_widths(main_window):
    """Auto-fit fires on modelReset (species change) and enforces floor widths."""
    model = main_window._batch_model
    table = main_window._batch_table
    assert table is not None

    model.set_species(["A", "B"])
    QtWidgets.QApplication.processEvents()

    # Column 0 minimum is font-metrics-based; species columns have no floor
    fm = table.horizontalHeader().fontMetrics()
    set_name_min = fm.horizontalAdvance(" Set Name ") + 20
    assert table.columnWidth(0) >= set_name_min, "Set Name column below minimum width"
    assert table.columnWidth(1) > 0, "Species column 1 should have positive width"
    assert table.columnWidth(2) > 0, "Species column 2 should have positive width"

    # A second modelReset (species change) should re-apply auto-fit
    model.set_species(["A", "B", "LongerSpeciesName"])
    QtWidgets.QApplication.processEvents()

    assert table.columnWidth(0) >= set_name_min, "Set Name column below minimum after reset"
    assert table.columnWidth(1) > 0, "Species column 1 should have positive width after reset"
    assert table.columnWidth(2) > 0, "Species column 2 should have positive width after reset"
    assert table.columnWidth(3) > 0, "Species column 3 should have positive width after reset"
