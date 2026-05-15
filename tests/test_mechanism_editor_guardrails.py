import json
import time
from unittest.mock import MagicMock

import numpy as np
import pytest
from PySide6 import QtCore, QtGui, QtWidgets

from kindred.gui.ports import SliderReplayIntent

pytestmark = pytest.mark.gui


def _unlock_reactions_editing(main_window, monkeypatch) -> None:
    reactions_widget = main_window._mechanism_editor._reactions_text
    if not reactions_widget.toPlainText().strip():
        reactions_widget.setPlainText("reaction: A -> B; k=1.0")
    monkeypatch.setattr(
        main_window,
        "_prompt_mechanism_edit_unlock_warning",
        lambda: True,
    )
    main_window._mechanism_edit_lock_action.trigger()
    assert main_window.mechanism_editing_locked() is False


def _consumer_call_recorder(main_window, monkeypatch):
    calls: list[str] = []
    original_temperature = main_window._update_temperature_mode_indicator
    original_authoritative = main_window._on_authoritative_mechanism_input_changed
    original_overlay = main_window._refresh_overlay_swatches_for_current_mechanism

    def _record_temperature() -> None:
        calls.append("temperature")
        original_temperature()

    def _record_authoritative() -> None:
        calls.append("authoritative")
        original_authoritative()

    def _record_overlay() -> None:
        calls.append("overlay")
        original_overlay()

    monkeypatch.setattr(main_window, "_update_temperature_mode_indicator", _record_temperature)
    monkeypatch.setattr(main_window, "_on_authoritative_mechanism_input_changed", _record_authoritative)
    monkeypatch.setattr(main_window, "_refresh_overlay_swatches_for_current_mechanism", _record_overlay)
    return calls


def _transition_effect_recorder(main_window, monkeypatch):
    calls: list[str] = []
    original_temperature = main_window._update_temperature_mode_indicator
    original_overlay = main_window._refresh_overlay_swatches_for_current_mechanism

    def _record_temperature() -> None:
        calls.append("temperature")
        original_temperature()

    def _record_overlay() -> None:
        calls.append("overlay")
        original_overlay()

    def _record_supersede(*, epoch: int) -> None:
        calls.append(f"supersede:{int(epoch)}")

    monkeypatch.setattr(main_window, "_update_temperature_mode_indicator", _record_temperature)
    monkeypatch.setattr(main_window, "_refresh_overlay_swatches_for_current_mechanism", _record_overlay)
    monkeypatch.setattr(
        main_window.simulation_controller,
        "supersede_active_work_for_authoritative_mechanism_transition",
        _record_supersede,
    )
    return calls


def _set_invalid_reactions_text(main_window, qt_app) -> None:
    main_window._mechanism_editor._reactions_text.setPlainText("this line does not parse")
    _wait_for_mechanism_validity(main_window, qt_app, expected_valid=False)


def _set_valid_reactions_text(main_window, qt_app, text: str = "T=400\nreaction: A -> B; k=1.0") -> None:
    main_window._mechanism_editor._reactions_text.setPlainText(str(text))
    if not main_window.mechanism_editing_locked():
        _process_events_bounded(qt_app, iterations=1)
        return
    _wait_for_mechanism_validity(main_window, qt_app, expected_valid=True)




def _append_text(editor: QtWidgets.QPlainTextEdit, text: str) -> None:
    cursor = editor.textCursor()
    cursor.movePosition(QtGui.QTextCursor.MoveOperation.End)
    editor.setTextCursor(cursor)
    editor.insertPlainText(str(text))


def _select_batch_rows(main_window, rows: list[int]) -> None:
    table = main_window._batch_table
    assert table is not None
    sel = table.selectionModel()
    assert sel is not None
    sel.clearSelection()
    table.setCurrentIndex(main_window._batch_model.index(int(rows[0]), 0))
    for row in rows:
        idx = main_window._batch_model.index(int(row), 0)
        sel.select(idx, QtCore.QItemSelectionModel.Select | QtCore.QItemSelectionModel.Rows)
    main_window._refresh_batch_display_from_focus_and_shown()


def _process_events_bounded(qt_app, iterations: int = 20) -> None:
    for _ in range(int(iterations)):
        qt_app.processEvents()
        QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)


def _invalid_state_network_dsl() -> str:
    return "\n".join(
        [
            "state: A, kind=GS, energy=0, energy_unit=kJ/mol, degeneracy=1",
            "state: TS1, kind=TS, energy=10, energy_unit=kJ/mol, degeneracy=1",
            "edge: A,TS1",
        ]
    )


def _valid_state_network_dsl() -> str:
    return "\n".join(
        [
            "state: A, kind=GS, energy=0, energy_unit=kJ/mol, degeneracy=1",
            "state: TS1, kind=TS, energy=10, energy_unit=kJ/mol, degeneracy=1",
            "state: B, kind=GS, energy=0, energy_unit=kJ/mol, degeneracy=1",
            "edge: A,TS1",
            "edge: TS1,B",
        ]
    )


def _wait_for_mechanism_validity(main_window, qt_app, expected_valid: bool, timeout_ms: int = 1500) -> None:
    deadline = time.monotonic() + (float(timeout_ms) / 1000.0)
    while time.monotonic() < deadline:
        _process_events_bounded(qt_app, iterations=1)
        if bool(main_window._mechanism_editor.is_mechanism_valid()) is bool(expected_valid):
            return
        time.sleep(0.01)
    raise AssertionError(
        f"Mechanism validity did not become {bool(expected_valid)} within {int(timeout_ms)} ms"
    )


def _project_payload(*, mechanism: str, notes: str) -> dict[str, object]:
    return {
        "mechanism": str(mechanism),
        "notes": str(notes),
        "state_network": "",
    }


def _load_project_via_dialog(main_window, tmp_path, monkeypatch, payload) -> None:
    project_path = tmp_path / "guardrails_project.kin"
    project_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getOpenFileName",
        lambda *args, **kwargs: (str(project_path), "Kindred Project (*.kin)"),
    )
    main_window.project_controller.load_project()

def test_reactions_editor_is_locked_by_default(main_window):
    reactions_widget = main_window._mechanism_editor._reactions_text
    action = main_window._mechanism_edit_lock_action

    assert main_window.mechanism_editing_locked() is True
    assert reactions_widget.isReadOnly() is True
    assert action is not None
    assert action.isCheckable() is True
    assert action.isChecked() is False


def test_locked_state_network_programmatic_setters_still_work(main_window):
    state_editor = main_window._mechanism_editor._state_network_editor
    state_text = "\n".join(
        [
            "state: A, kind=GS, energy=0, energy_unit=kJ/mol, degeneracy=1",
            "state: TS1, kind=TS, energy=10, energy_unit=kJ/mol, degeneracy=1",
            "edge: A,TS1",
        ]
    )

    assert main_window.mechanism_editing_locked() is True

    state_editor.set_state_network_dsl(state_text)

    assert main_window.mechanism_state_network_dsl_raw() == state_text

    state_editor.clear()

    assert main_window.mechanism_state_network_dsl_raw() == ""


def test_owner_canonical_text_remains_stable_while_unlocked_edits_are_staged(main_window, monkeypatch, qt_app):
    baseline_reactions = "reaction: A -> B; k=1.0"
    baseline_state = _valid_state_network_dsl()
    staged_reactions = "T=410\nreaction: A -> B; k=2.0"
    staged_state = "\n".join(
        [
            "state: A, kind=GS, energy=0, energy_unit=kJ/mol, degeneracy=1",
            "state: TS1, kind=TS, energy=12, energy_unit=kJ/mol, degeneracy=1",
            "state: B, kind=GS, energy=-1, energy_unit=kJ/mol, degeneracy=1",
            "edge: A,TS1",
            "edge: TS1,B",
        ]
    )

    main_window._mechanism_editor._reactions_text.setPlainText(baseline_reactions)
    main_window._mechanism_editor._state_network_editor.set_state_network_dsl(baseline_state)
    _wait_for_mechanism_validity(main_window, qt_app, expected_valid=True)

    _unlock_reactions_editing(main_window, monkeypatch)
    main_window._mechanism_editor._reactions_text.setPlainText(staged_reactions)
    main_window._mechanism_editor._state_network_editor.set_state_network_dsl(staged_state)
    qt_app.processEvents()

    owner = main_window._mechanism_session_owner
    assert owner.canonical_reactions_text == baseline_reactions
    assert owner.canonical_state_network_dsl == baseline_state
    assert main_window.mechanism_reactions_text_raw() == baseline_reactions
    assert main_window.mechanism_state_network_dsl_raw() == baseline_state


def test_project_serialize_uses_canonical_mechanism_while_edit_session_active(main_window, qt_app):
    baseline_reactions = "reaction: A -> B; k=1.0"
    baseline_state = _valid_state_network_dsl()
    staged_reactions = "reaction: X -> Y; k=99.0"
    staged_state = "\n".join(
        [
            "state: X, kind=GS, energy=0, energy_unit=kJ/mol, degeneracy=1",
            "state: TS9, kind=TS, energy=20, energy_unit=kJ/mol, degeneracy=1",
            "state: Y, kind=GS, energy=-2, energy_unit=kJ/mol, degeneracy=1",
            "edge: X,TS9",
            "edge: TS9,Y",
        ]
    )

    main_window._mechanism_editor._reactions_text.setPlainText(baseline_reactions)
    main_window._mechanism_editor._state_network_editor.set_state_network_dsl(baseline_state)
    _wait_for_mechanism_validity(main_window, qt_app, expected_valid=True)

    owner = main_window._mechanism_session_owner
    owner.begin_edit_session()

    # Project save must persist canonical mechanism, not in-progress draft.
    main_window._mechanism_editor._reactions_text.setPlainText(staged_reactions)
    main_window._mechanism_editor._state_network_editor.set_state_network_dsl(staged_state)
    qt_app.processEvents()

    assert owner.edit_session_active is True
    assert main_window.mechanism_reactions_text_raw() == baseline_reactions
    assert main_window.mechanism_state_network_dsl_raw() == baseline_state

    payload = main_window._serialize_project_state()

    assert payload["mechanism"] == baseline_reactions
    assert payload["mechanism"] != staged_reactions
    assert payload["state_network"] == baseline_state
    assert payload["state_network"] != staged_state


def test_owner_draft_tracks_unlocked_widget_edits_and_commits_on_lock(main_window, monkeypatch, qt_app):
    baseline_reactions = "reaction: A -> B; k=1.0"
    baseline_state = _valid_state_network_dsl()
    staged_reactions = "T=410\nreaction: A -> B; k=2.0"
    staged_state = "\n".join(
        [
            "state: A, kind=GS, energy=0, energy_unit=kJ/mol, degeneracy=1",
            "state: TS1, kind=TS, energy=12, energy_unit=kJ/mol, degeneracy=1",
            "state: B, kind=GS, energy=-1, energy_unit=kJ/mol, degeneracy=1",
            "edge: A,TS1",
            "edge: TS1,B",
        ]
    )

    main_window._mechanism_editor._reactions_text.setPlainText(baseline_reactions)
    main_window._mechanism_editor._state_network_editor.set_state_network_dsl(baseline_state)
    _wait_for_mechanism_validity(main_window, qt_app, expected_valid=True)

    _unlock_reactions_editing(main_window, monkeypatch)
    main_window._mechanism_editor._reactions_text.setPlainText(staged_reactions)
    main_window._mechanism_editor._state_network_editor.set_state_network_dsl(staged_state)
    qt_app.processEvents()

    owner = main_window._mechanism_session_owner
    assert owner.edit_session_active is True
    assert owner.draft_reactions_text == staged_reactions
    assert owner.draft_state_network_dsl == staged_state

    main_window._mechanism_edit_lock_action.trigger()
    qt_app.processEvents()

    assert owner.edit_session_active is False
    assert owner.canonical_reactions_text == staged_reactions
    assert owner.canonical_state_network_dsl == staged_state
    assert main_window.mechanism_reactions_text_raw() == staged_reactions
    assert main_window.mechanism_state_network_dsl_raw() == staged_state


def test_unlock_reactions_editing_cancel_keeps_editor_locked(main_window, monkeypatch):
    reactions_widget = main_window._mechanism_editor._reactions_text
    action = main_window._mechanism_edit_lock_action

    prompts: list[str] = []
    monkeypatch.setattr(
        main_window,
        "_prompt_mechanism_edit_unlock_warning",
        lambda: prompts.append("warn") or False,
    )

    action.trigger()

    assert prompts == ["warn"]
    assert main_window.mechanism_editing_locked() is True
    assert reactions_widget.isReadOnly() is True
    assert action.isChecked() is False


def test_unlock_reactions_editing_warns_once_per_window(main_window, monkeypatch):
    reactions_widget = main_window._mechanism_editor._reactions_text
    action = main_window._mechanism_edit_lock_action
    reactions_widget.setPlainText("reaction: A -> B; k=1.0")

    prompts: list[str] = []
    monkeypatch.setattr(
        main_window,
        "_prompt_mechanism_edit_unlock_warning",
        lambda: prompts.append("warn") or True,
    )

    action.trigger()

    assert prompts == ["warn"]
    assert main_window.mechanism_editing_locked() is False
    assert reactions_widget.isReadOnly() is False
    assert action.isChecked() is True

    action.trigger()

    assert main_window.mechanism_editing_locked() is True
    assert reactions_widget.isReadOnly() is True
    assert action.isChecked() is False

    action.trigger()

    assert prompts == ["warn"]
    assert main_window.mechanism_editing_locked() is False
    assert reactions_widget.isReadOnly() is False
    assert action.isChecked() is True


def test_consumer_guards_suppress_main_window_work_while_unlocked(main_window, monkeypatch, qt_app):
    _unlock_reactions_editing(main_window, monkeypatch)
    calls = _consumer_call_recorder(main_window, monkeypatch)
    main_window._temperature_spinbox.setValue(298.15)
    _set_valid_reactions_text(main_window, qt_app)
    qt_app.processEvents()

    assert calls == []
    assert main_window._temperature_spinbox.value() == pytest.approx(298.15)


def test_spinbox_change_while_unlocked_does_not_read_staged_text(main_window, monkeypatch, qt_app):
    _set_valid_reactions_text(main_window, qt_app, text="reaction: A -> B; k=1.0")
    main_window._mechanism_editor._state_network_editor.clear()
    main_window._temperature_spinbox.setValue(305.0)
    qt_app.processEvents()
    baseline_indicator = main_window._temperature_mode_indicator.text()
    assert "from DSL" not in baseline_indicator

    compute_calls: list[tuple[str, str]] = []
    original_compute = main_window._compute_temperature_indicator_state

    def _record_compute(*, reactions_text: str, state_network_text: str):
        compute_calls.append((str(reactions_text), str(state_network_text)))
        return original_compute(
            reactions_text=str(reactions_text),
            state_network_text=str(state_network_text),
        )

    monkeypatch.setattr(main_window, "_compute_temperature_indicator_state", _record_compute)

    _unlock_reactions_editing(main_window, monkeypatch)
    main_window._mechanism_editor._reactions_text.setPlainText("T=400\nreaction: A -> B; k=1.0")
    qt_app.processEvents()

    main_window._temperature_spinbox.setValue(315.0)
    qt_app.processEvents()

    assert main_window._temperature_mode_indicator.text() == baseline_indicator
    assert "400" not in main_window._temperature_mode_indicator.text()
    assert "from DSL" not in main_window._temperature_mode_indicator.text()
    assert compute_calls[-1] == ("reaction: A -> B; k=1.0", "")


def test_spinbox_change_while_locked_fires_temperature_indicator(main_window, monkeypatch, qt_app):
    _unlock_reactions_editing(main_window, monkeypatch)
    calls: list[str] = []

    def _record_temperature() -> None:
        calls.append("temperature")

    monkeypatch.setattr(main_window, "_update_temperature_mode_indicator", _record_temperature)

    main_window._force_lock_editor()
    assert main_window.mechanism_editing_locked() is True

    main_window._temperature_spinbox.setValue(float(main_window._temperature_spinbox.value()) + 10.0)
    qt_app.processEvents()

    assert calls == ["temperature"]


def test_state_network_change_fires_temperature_indicator_when_locked(main_window, monkeypatch, qt_app):
    _unlock_reactions_editing(main_window, monkeypatch)
    calls: list[str] = []

    def _record_temperature() -> None:
        calls.append("temperature")

    monkeypatch.setattr(main_window, "_update_temperature_mode_indicator", _record_temperature)

    main_window._force_lock_editor()
    assert main_window.mechanism_editing_locked() is True

    main_window._mechanism_editor._state_network_editor.set_state_network_dsl(_valid_state_network_dsl())
    qt_app.processEvents()

    assert calls == ["temperature"]


def test_state_network_change_suppressed_while_unlocked(main_window, monkeypatch, qt_app):
    _unlock_reactions_editing(main_window, monkeypatch)
    calls: list[str] = []

    def _record_temperature() -> None:
        calls.append("temperature")

    monkeypatch.setattr(main_window, "_update_temperature_mode_indicator", _record_temperature)

    main_window._force_lock_editor()
    main_window._set_mechanism_edit_locked(False)
    assert main_window.mechanism_editing_locked() is False

    main_window._mechanism_editor._state_network_editor.set_state_network_dsl(_valid_state_network_dsl())
    qt_app.processEvents()

    assert calls == []


def test_consumer_guards_resume_main_window_work_after_successful_lock(main_window, monkeypatch, qt_app):
    _unlock_reactions_editing(main_window, monkeypatch)
    calls = _transition_effect_recorder(main_window, monkeypatch)
    original_validate = main_window._mechanism_editor._validate_dsl

    def _validate_and_record() -> None:
        calls.append("validate")
        original_validate()

    monkeypatch.setattr(main_window._mechanism_editor, "_validate_dsl", _validate_and_record)
    main_window._temperature_spinbox.setValue(298.15)
    _set_valid_reactions_text(main_window, qt_app)
    qt_app.processEvents()

    assert calls == []
    assert main_window._temperature_spinbox.value() == pytest.approx(298.15)

    main_window._mechanism_edit_lock_action.trigger()
    qt_app.processEvents()

    assert main_window.mechanism_editing_locked() is True
    assert "validate" in calls
    assert calls.count("temperature") == 1
    assert calls.count("overlay") == 1
    assert any(call.startswith("supersede:") for call in calls)
    assert main_window._temperature_spinbox.value() == pytest.approx(400.0)

    calls.clear()
    main_window._mechanism_editor._reactions_text.setPlainText("T=410\nreaction: A -> B; k=1.0")
    qt_app.processEvents()
    assert "temperature" in calls
    assert "overlay" in calls
    assert any(call.startswith("supersede:") for call in calls)


def test_failed_lock_preserves_all_cached_state(main_window, monkeypatch, qt_app):
    main_window._load_preset_mechanism("M1")
    qt_app.processEvents()
    _select_batch_rows(main_window, [0])
    qt_app.processEvents()

    plot = main_window._plot_tabs._main_plot
    cache = main_window.simulation_controller.batch_cache
    result_t = np.asarray([0.0, 0.5, 1.0], dtype=float)
    result_series = {
        "A": np.asarray([1.0, 0.7, 0.4], dtype=float),
        "B": np.asarray([0.0, 0.3, 0.6], dtype=float),
    }
    cache.active_cache_key = "guardrails-explicit-cache-key"
    cache.active_batch_set_id = str(main_window._batch_set_id_for_row(0) or "")
    cache.active_batch_set = str(main_window.batch_set_name_for_id(cache.active_batch_set_id) or "")
    cache.last_display_selection = [cache.active_batch_set_id]
    cache.active_cache_invalidated_set_ids = None
    cache.result_cache[f"{cache.active_cache_key}::{cache.active_batch_set_id}"] = {
        "t": result_t,
        "series": result_series,
    }
    plot.set_data(result_t, result_series, label=cache.active_batch_set or "set1")
    main_window._status_label.setText("Ready")

    initial_plot_t = tuple(float(value) for value in plot._t.tolist())
    initial_plot_series = {
        str(name): tuple(float(value) for value in values.tolist())
        for name, values in dict(getattr(plot, "_series", {}) or {}).items()
    }
    initial_cache_key = str(cache.active_cache_key or "")
    initial_invalidated = cache.active_cache_invalidated_set_ids

    _unlock_reactions_editing(main_window, monkeypatch)
    slider_runtime_token = object()
    main_window._variable_runtime._slider_runtime = slider_runtime_token
    main_window._variable_runtime.set_slider_runtime_dirty(False)
    main_window._temperature_spinbox.setValue(298.15)
    _set_invalid_reactions_text(main_window, qt_app)

    main_window._mechanism_edit_lock_action.trigger()
    qt_app.processEvents()

    assert main_window.mechanism_editing_locked() is False
    assert main_window._mechanism_edit_lock_action.isChecked() is True
    assert main_window._temperature_spinbox.value() == pytest.approx(298.15)
    assert str(cache.active_cache_key or "") == initial_cache_key
    assert cache.active_cache_invalidated_set_ids == initial_invalidated
    assert tuple(float(value) for value in plot._t.tolist()) == initial_plot_t
    assert {
        str(name): tuple(float(value) for value in values.tolist())
        for name, values in dict(getattr(plot, "_series", {}) or {}).items()
    } == initial_plot_series
    assert main_window._variable_runtime._slider_runtime is slider_runtime_token
    assert main_window._variable_runtime.slider_runtime_dirty() is False
    assert main_window._status_label.text() != "Result not cached (evicted). Press Run to compute."


def test_failed_lock_after_invalid_state_network_edit_preserves_all_cached_state(main_window, monkeypatch, qt_app):
    main_window._load_preset_mechanism("M1")
    qt_app.processEvents()
    _select_batch_rows(main_window, [0])
    qt_app.processEvents()

    plot = main_window._plot_tabs._main_plot
    cache = main_window.simulation_controller.batch_cache
    result_t = np.asarray([0.0, 0.5, 1.0], dtype=float)
    result_series = {
        "A": np.asarray([1.0, 0.7, 0.4], dtype=float),
        "B": np.asarray([0.0, 0.3, 0.6], dtype=float),
    }
    cache.active_cache_key = "guardrails-state-network-cache-key"
    cache.active_batch_set_id = str(main_window._batch_set_id_for_row(0) or "")
    cache.active_batch_set = str(main_window.batch_set_name_for_id(cache.active_batch_set_id) or "")
    cache.last_display_selection = [cache.active_batch_set_id]
    cache.active_cache_invalidated_set_ids = None
    cache.result_cache[f"{cache.active_cache_key}::{cache.active_batch_set_id}"] = {
        "t": result_t,
        "series": result_series,
    }
    plot.set_data(result_t, result_series, label=cache.active_batch_set or "set1")
    main_window._status_label.setText("Ready")

    initial_plot_t = tuple(float(value) for value in plot._t.tolist())
    initial_plot_series = {
        str(name): tuple(float(value) for value in values.tolist())
        for name, values in dict(getattr(plot, "_series", {}) or {}).items()
    }
    initial_cache_key = str(cache.active_cache_key or "")
    initial_invalidated = cache.active_cache_invalidated_set_ids

    _unlock_reactions_editing(main_window, monkeypatch)
    slider_runtime_token = object()
    main_window._variable_runtime._slider_runtime = slider_runtime_token
    main_window._variable_runtime.set_slider_runtime_dirty(False)
    main_window._mechanism_editor._state_network_editor.set_state_network_dsl(_invalid_state_network_dsl())
    qt_app.processEvents()

    assert main_window.mechanism_editing_locked() is False
    assert str(cache.active_cache_key or "") == initial_cache_key
    assert cache.active_cache_invalidated_set_ids == initial_invalidated
    assert tuple(float(value) for value in plot._t.tolist()) == initial_plot_t
    assert {
        str(name): tuple(float(value) for value in values.tolist())
        for name, values in dict(getattr(plot, "_series", {}) or {}).items()
    } == initial_plot_series
    assert main_window._variable_runtime._slider_runtime is slider_runtime_token
    assert main_window._variable_runtime.slider_runtime_dirty() is False

    main_window._mechanism_edit_lock_action.trigger()
    qt_app.processEvents()

    assert main_window.mechanism_editing_locked() is False
    assert main_window._mechanism_edit_lock_action.isChecked() is True
    assert str(cache.active_cache_key or "") == initial_cache_key
    assert cache.active_cache_invalidated_set_ids == initial_invalidated
    assert tuple(float(value) for value in plot._t.tolist()) == initial_plot_t
    assert {
        str(name): tuple(float(value) for value in values.tolist())
        for name, values in dict(getattr(plot, "_series", {}) or {}).items()
    } == initial_plot_series
    assert main_window._variable_runtime._slider_runtime is slider_runtime_token
    assert main_window._variable_runtime.slider_runtime_dirty() is False
    assert main_window._status_label.text() != "Result not cached (evicted). Press Run to compute."


def test_lock_action_toggle_reverts_on_failure(main_window, monkeypatch, qt_app):
    _unlock_reactions_editing(main_window, monkeypatch)

    reactions_widget = main_window._mechanism_editor._reactions_text
    _set_invalid_reactions_text(main_window, qt_app)

    main_window._mechanism_edit_lock_action.trigger()
    qt_app.processEvents()

    assert main_window.mechanism_editing_locked() is False
    assert reactions_widget.isReadOnly() is False
    assert main_window._mechanism_edit_lock_action.isChecked() is True
    assert main_window._mechanism_editor.is_mechanism_valid() is False


def test_auto_lock_for_run_refuses_unchanged_invalid_mechanism(main_window, monkeypatch, qt_app):
    monkeypatch.setattr(
        main_window,
        "_prompt_mechanism_edit_unlock_warning",
        lambda: True,
    )
    main_window._mechanism_edit_lock_action.trigger()
    qt_app.processEvents()

    assert main_window.mechanism_editing_locked() is False
    assert main_window._simulation_mechanism_owner.auto_lock_for_run() is False
    assert main_window.mechanism_editing_locked() is False
    assert main_window._mechanism_editor.is_mechanism_valid() is False


def test_is_mechanism_ready_for_run_uses_canonical_state_while_unlocked(main_window, monkeypatch, qt_app):
    _set_valid_reactions_text(main_window, qt_app, text="reaction: A -> B; k=1.0")
    main_window._mechanism_editor._state_network_editor.clear()
    qt_app.processEvents()
    assert main_window.mechanism_editing_locked() is True
    assert main_window._simulation_mechanism_owner.is_mechanism_ready_for_run() is True

    main_window._force_lock_editor()
    _set_invalid_reactions_text(main_window, qt_app)
    assert main_window.mechanism_editing_locked() is True
    assert main_window._simulation_mechanism_owner.is_mechanism_ready_for_run() is False

    _unlock_reactions_editing(main_window, monkeypatch)
    _set_valid_reactions_text(main_window, qt_app, text="reaction: A -> B; k=1.0")
    assert main_window.mechanism_editing_locked() is False
    assert main_window._simulation_mechanism_owner.is_mechanism_ready_for_run() is False

    _set_invalid_reactions_text(main_window, qt_app)
    assert main_window.mechanism_editing_locked() is False
    assert main_window._simulation_mechanism_owner.is_mechanism_ready_for_run() is False


def test_is_mechanism_ready_for_run_does_not_apply_authoritative_updates(main_window, monkeypatch, qt_app):
    _set_valid_reactions_text(main_window, qt_app, text="reaction: A -> B; k=1.0")
    owner = main_window._mechanism_session_owner
    calls: list[tuple[str, str]] = []
    original_apply = owner.apply_authoritative_update

    def _record_apply(reactions_text: str, state_network_dsl: str) -> None:
        calls.append((str(reactions_text), str(state_network_dsl)))
        original_apply(reactions_text, state_network_dsl)

    monkeypatch.setattr(owner, "apply_authoritative_update", _record_apply)

    assert main_window._simulation_mechanism_owner.is_mechanism_ready_for_run() is True
    assert calls == []


def test_empty_reactions_valid_state_network_is_ready_for_run(main_window, qt_app):
    main_window._mechanism_editor._reactions_text.setPlainText("")
    main_window._mechanism_editor._state_network_editor.set_state_network_dsl(_valid_state_network_dsl())
    qt_app.processEvents()

    assert main_window._mechanism_editor._state_network_editor.is_valid() is True
    assert main_window._simulation_mechanism_owner.is_mechanism_ready_for_run() is True


def test_empty_reactions_no_state_network_not_ready_for_run(main_window, qt_app):
    main_window._mechanism_editor._reactions_text.setPlainText("")
    main_window._mechanism_editor._state_network_editor.clear()
    qt_app.processEvents()

    assert main_window._simulation_mechanism_owner.is_mechanism_ready_for_run() is False


def test_invalid_reactions_valid_state_network_not_ready_for_run(main_window, qt_app):
    main_window._mechanism_editor._reactions_text.setPlainText("this line does not parse")
    main_window._mechanism_editor._state_network_editor.set_state_network_dsl(_valid_state_network_dsl())
    qt_app.processEvents()

    assert main_window._mechanism_editor._state_network_editor.is_valid() is True
    assert main_window._simulation_mechanism_owner.is_mechanism_ready_for_run() is False


def test_valid_reactions_invalid_state_network_not_ready_for_run(main_window, qt_app):
    main_window._mechanism_editor._reactions_text.setPlainText("reaction: A -> B; k=1.0")
    main_window._mechanism_editor._state_network_editor.set_state_network_dsl(_invalid_state_network_dsl())
    qt_app.processEvents()

    assert main_window._mechanism_editor._state_network_editor.is_valid() is False
    assert main_window._simulation_mechanism_owner.is_mechanism_ready_for_run() is False


def test_named_inline_initial_set_blocks_are_ready_for_run(main_window, qt_app):
    main_window._mechanism_editor._reactions_text.setPlainText(
        "reaction: A -> B; k=1.0\n"
        "\n"
        "Set B = {\n"
        "[A] = 1.0\n"
        "}\n"
    )
    main_window._mechanism_editor._state_network_editor.clear()
    qt_app.processEvents()
    _wait_for_mechanism_validity(main_window, qt_app, expected_valid=True)

    assert main_window._simulation_mechanism_owner.is_mechanism_ready_for_run() is True


def test_state_network_validation_blocks_lock(main_window, monkeypatch, qt_app):
    _unlock_reactions_editing(main_window, monkeypatch)

    main_window._mechanism_editor._reactions_text.setPlainText("reaction: A -> B; k=1.0")
    main_window._mechanism_editor._state_network_editor.set_state_network_dsl(_invalid_state_network_dsl())
    qt_app.processEvents()

    main_window._mechanism_edit_lock_action.trigger()
    qt_app.processEvents()

    assert main_window.mechanism_editing_locked() is False
    assert main_window._mechanism_edit_lock_action.isChecked() is True
    assert main_window._mechanism_editor._state_network_editor.is_valid() is False


def test_state_network_active_editor_invalid_numeric_input_blocks_lock(main_window, monkeypatch, qt_app):
    _unlock_reactions_editing(main_window, monkeypatch)
    editor = main_window._mechanism_editor._state_network_editor
    editor.clear()
    editor._add_state_btn.click()
    qt_app.processEvents()

    name_item = editor._states_table.item(0, 0)
    energy_item = editor._states_table.item(0, 2)
    assert name_item is not None
    assert energy_item is not None

    name_item.setText("A")
    editor._states_table.editItem(energy_item)
    qt_app.processEvents()

    active_editor = editor.findChild(QtWidgets.QLineEdit)
    assert active_editor is not None
    active_editor.clear()
    active_editor.setText("not-a-number")
    qt_app.processEvents()

    main_window._mechanism_edit_lock_action.trigger()
    qt_app.processEvents()

    assert main_window.mechanism_editing_locked() is False
    assert main_window._mechanism_edit_lock_action.isChecked() is True
    assert editor._states_table.item(0, 2).text() == "not-a-number"
    assert editor.is_valid() is False


def test_auto_lock_for_run_checks_state_network(main_window, monkeypatch, qt_app):
    _unlock_reactions_editing(main_window, monkeypatch)
    main_window._mechanism_editor._reactions_text.setPlainText("reaction: A -> B; k=1.0")
    main_window._mechanism_editor._state_network_editor.set_state_network_dsl(_invalid_state_network_dsl())
    qt_app.processEvents()
    main_window.simulation_controller.run_simulation_internal = MagicMock()

    main_window.simulation_controller.run_simulation()
    qt_app.processEvents()

    assert main_window.mechanism_editing_locked() is False
    assert main_window._mechanism_edit_lock_action.isChecked() is True
    main_window.simulation_controller.run_simulation_internal.assert_not_called()
    assert main_window._status_label.text() == "Cannot run: mechanism has errors. Fix and try again."


def test_locked_invalid_dsl_blocks_explicit_run(main_window, qt_app):
    main_window._force_lock_editor()
    main_window._mechanism_editor.set_reactions_text("this line does not parse", block_signals=True)
    main_window.simulation_controller.run_simulation_internal = MagicMock()

    main_window.simulation_controller.run_simulation()
    qt_app.processEvents()

    main_window.simulation_controller.run_simulation_internal.assert_not_called()
    assert main_window.mechanism_editing_locked() is True
    assert main_window._simulation_mechanism_owner.is_mechanism_ready_for_run() is False
    assert main_window._status_label.text() == "Cannot run: mechanism has errors. Fix and try again."


def test_run_auto_locks_editor_from_main_window(main_window, monkeypatch, qt_app):
    _unlock_reactions_editing(main_window, monkeypatch)
    main_window._mechanism_editor._reactions_text.setPlainText("reaction: A -> B; k=1.0")
    qt_app.processEvents()
    main_window.simulation_controller.run_simulation_internal = MagicMock()

    main_window.simulation_controller.run_simulation()
    qt_app.processEvents()

    assert main_window.mechanism_editing_locked() is True
    main_window.simulation_controller.run_simulation_internal.assert_called_once()


def test_run_auto_locks_and_proceeds_for_state_network_only_mechanism(main_window, monkeypatch, qt_app):
    _unlock_reactions_editing(main_window, monkeypatch)
    main_window._mechanism_editor._reactions_text.setPlainText("")
    main_window._mechanism_editor._state_network_editor.set_state_network_dsl(_valid_state_network_dsl())
    qt_app.processEvents()
    main_window.simulation_controller.run_simulation_internal = MagicMock()

    main_window.simulation_controller.run_simulation()
    qt_app.processEvents()

    assert main_window.mechanism_editing_locked() is True
    main_window.simulation_controller.run_simulation_internal.assert_called_once()


def test_run_wegscheider_resolution_accept_uses_authoritative_transition(
    main_window,
    monkeypatch,
    qt_app,
):
    text = "\n".join(
        [
            "equilibrium: A <-> B ; kf=1 ; K=2",
            "equilibrium: B <-> C ; kf=1 ; K=3",
            "equilibrium: C <-> A ; kf=1 ; K=7",
            "initial: A=1",
            "initial: B=0",
            "initial: C=0",
        ]
    )
    main_window._force_lock_editor()
    main_window._mechanism_editor._reactions_text.setPlainText(text)
    main_window._wegscheider_cyclicity_enabled = True
    before_epoch = int(main_window.simulation_controller.authoritative_mechanism_transition_epoch)
    monkeypatch.setattr(
        main_window._simulation_dialogs,
        "choose_wegscheider_resolution",
        lambda _title, _message, _choices: {"cycle_1": "Keq3"},
    )
    main_window.simulation_controller.run_simulation_internal = MagicMock()

    main_window.simulation_controller.run_simulation()
    qt_app.processEvents()

    assert "param Keq3 = 1 / (Keq1 * Keq2)" in main_window.mechanism_reactions_text_raw()
    assert int(main_window.simulation_controller.authoritative_mechanism_transition_epoch) > before_epoch
    main_window.simulation_controller.run_simulation_internal.assert_called_once()


def test_run_wegscheider_resolution_cancel_preserves_authoritative_source_and_transition(
    main_window,
    monkeypatch,
    qt_app,
):
    text = "\n".join(
        [
            "equilibrium: A <-> B ; kf=1 ; K=2",
            "equilibrium: B <-> C ; kf=1 ; K=3",
            "equilibrium: C <-> A ; kf=1 ; K=7",
            "initial: A=1",
            "initial: B=0",
            "initial: C=0",
        ]
    )
    main_window._force_lock_editor()
    main_window._mechanism_editor._reactions_text.setPlainText(text)
    main_window._wegscheider_cyclicity_enabled = True
    before_epoch = int(main_window.simulation_controller.authoritative_mechanism_transition_epoch)
    monkeypatch.setattr(
        main_window._simulation_dialogs,
        "choose_wegscheider_resolution",
        lambda _title, _message, _choices: None,
    )
    main_window.simulation_controller.run_simulation_internal = MagicMock()

    main_window.simulation_controller.run_simulation()
    qt_app.processEvents()

    assert main_window.mechanism_reactions_text_raw() == text
    assert int(main_window.simulation_controller.authoritative_mechanism_transition_epoch) == before_epoch
    main_window.simulation_controller.run_simulation_internal.assert_not_called()


def test_mechanism_editor_run_button_stays_reactions_gated_for_state_network_only_mechanism(
    main_window,
    monkeypatch,
    qt_app,
):
    _unlock_reactions_editing(main_window, monkeypatch)
    main_window._mechanism_editor._reactions_text.setPlainText("")
    main_window._mechanism_editor._state_network_editor.set_state_network_dsl(_valid_state_network_dsl())
    _process_events_bounded(qt_app)

    assert main_window._mechanism_editor.run_btn.isEnabled() is False


def test_run_aborts_if_unchanged_invalid_editor_is_unlocked(main_window, monkeypatch, qt_app):
    _unlock_reactions_editing(main_window, monkeypatch)
    main_window._mechanism_editor._reactions_text.setPlainText("")
    qt_app.processEvents()
    main_window.simulation_controller.run_simulation_internal = MagicMock()

    main_window.simulation_controller.run_simulation()
    qt_app.processEvents()

    assert main_window.mechanism_editing_locked() is False
    assert main_window._mechanism_edit_lock_action.isChecked() is True
    main_window.simulation_controller.run_simulation_internal.assert_not_called()
    assert main_window._status_label.text() == "Cannot run: mechanism has errors. Fix and try again."


def test_state_network_invalid_blocks_run_even_when_locked(main_window, qt_app):
    main_window._mechanism_editor._reactions_text.setPlainText("reaction: A -> B; k=1.0")
    main_window._mechanism_editor._state_network_editor.set_state_network_dsl(_invalid_state_network_dsl())
    qt_app.processEvents()
    main_window._force_lock_editor()
    main_window.simulation_controller.run_simulation_internal = MagicMock()

    main_window.simulation_controller.run_simulation()
    qt_app.processEvents()

    main_window.simulation_controller.run_simulation_internal.assert_not_called()
    assert main_window._simulation_mechanism_owner.is_mechanism_ready_for_run() is False
    assert main_window._status_label.text() == "Cannot run: mechanism has errors. Fix and try again."


def test_locked_user_facing_undo_does_not_mutate_reactions_text(main_window, monkeypatch, qt_app):
    reactions_widget = main_window._mechanism_editor._reactions_text
    _unlock_reactions_editing(main_window, monkeypatch)

    baseline = "reaction: A -> B; k=1.0"
    reactions_widget.setPlainText(baseline)
    reactions_widget.setFocus()
    qt_app.processEvents()
    _append_text(reactions_widget, "\n# deliberate unlocked edit")
    changed_text = reactions_widget.toPlainText()
    assert changed_text != baseline

    main_window._mechanism_edit_lock_action.trigger()
    reactions_widget.setFocus()
    qt_app.processEvents()
    monkeypatch.setattr(main_window._undo_stack, "canUndo", lambda: False)
    monkeypatch.setattr(QtWidgets.QApplication, "focusWidget", lambda: reactions_widget)

    main_window._undo()

    assert main_window.mechanism_editing_locked() is True
    assert reactions_widget.toPlainText() == changed_text
    assert main_window._status_label.text() == "Allow Editing to undo mechanism changes"


def test_locked_user_facing_redo_does_not_mutate_reactions_text(main_window, monkeypatch, qt_app):
    reactions_widget = main_window._mechanism_editor._reactions_text
    _unlock_reactions_editing(main_window, monkeypatch)

    baseline = "reaction: A -> B; k=1.0"
    reactions_widget.setPlainText(baseline)
    reactions_widget.setFocus()
    qt_app.processEvents()
    _append_text(reactions_widget, "\n# deliberate unlocked edit")
    changed_text = reactions_widget.toPlainText()

    monkeypatch.setattr(main_window._undo_stack, "canUndo", lambda: False)
    monkeypatch.setattr(main_window._undo_stack, "canRedo", lambda: False)
    monkeypatch.setattr(QtWidgets.QApplication, "focusWidget", lambda: reactions_widget)

    main_window._undo()
    assert reactions_widget.toPlainText() == baseline

    main_window._mechanism_edit_lock_action.trigger()
    reactions_widget.setFocus()
    qt_app.processEvents()

    main_window._redo()

    assert main_window.mechanism_editing_locked() is True
    assert reactions_widget.toPlainText() == baseline
    assert main_window._status_label.text() == "Allow Editing to redo mechanism changes"
    assert reactions_widget.document().isRedoAvailable() is True
    assert changed_text != baseline


def test_unlocking_restores_user_facing_reactions_undo_redo(main_window, monkeypatch, qt_app):
    reactions_widget = main_window._mechanism_editor._reactions_text
    _unlock_reactions_editing(main_window, monkeypatch)

    baseline = "reaction: A -> B; k=1.0"
    reactions_widget.setPlainText(baseline)
    reactions_widget.setFocus()
    qt_app.processEvents()
    _append_text(reactions_widget, "\n# deliberate unlocked edit")
    changed_text = reactions_widget.toPlainText()
    monkeypatch.setattr(main_window._undo_stack, "canUndo", lambda: False)
    monkeypatch.setattr(main_window._undo_stack, "canRedo", lambda: False)
    monkeypatch.setattr(QtWidgets.QApplication, "focusWidget", lambda: reactions_widget)

    main_window._undo()
    assert reactions_widget.toPlainText() == baseline

    main_window._redo()
    assert reactions_widget.toPlainText() == changed_text


def test_locked_user_facing_undo_redo_do_not_apply_mechanism_undo_stack(main_window, qt_app):
    reactions_widget = main_window._mechanism_editor._reactions_text
    baseline = "reaction: A -> B; k=1.0"
    changed = "reaction: A -> B; k=2.0"

    reactions_widget.setPlainText(baseline)
    main_window.set_mechanism_reactions_text_with_optional_undo(
        changed,
        "Programmatic rewrite while locked",
        record_undo=True,
    )
    assert reactions_widget.toPlainText() == changed

    main_window._undo()
    assert reactions_widget.toPlainText() == changed
    assert main_window._status_label.text() == "Allow Editing to undo mechanism changes"

    main_window._undo_stack.undo()
    qt_app.processEvents()
    assert reactions_widget.toPlainText() == baseline

    main_window._redo()
    assert reactions_widget.toPlainText() == baseline
    assert main_window._status_label.text() == "Allow Editing to redo mechanism changes"


def test_locked_user_facing_undo_redo_do_not_apply_state_network_only_authoritative_command(
    main_window,
    qt_app,
):
    reactions_text = "reaction: A -> B; k=1.0"
    baseline_state = "\n".join(
        [
            "state: A, kind=GS, energy=0, energy_unit=kJ/mol, degeneracy=1",
            "state: TS1, kind=TS, energy=10, energy_unit=kJ/mol, degeneracy=1",
            "edge: A,TS1",
        ]
    )
    changed_state = "\n".join(
        [
            "state: A, kind=GS, energy=0, energy_unit=kJ/mol, degeneracy=1",
            "state: TS1, kind=TS, energy=10, energy_unit=kJ/mol, degeneracy=1",
            "state: B, kind=GS, energy=0, energy_unit=kJ/mol, degeneracy=1",
            "edge: A,TS1",
            "edge: TS1,B",
        ]
    )

    main_window._mechanism_editor._reactions_text.setPlainText(reactions_text)
    main_window._mechanism_editor._state_network_editor.set_state_network_dsl(baseline_state)

    main_window._set_authoritative_mechanism_editor_texts(
        reactions_text=reactions_text,
        state_network_dsl=changed_state,
        description="Programmatic state-network rewrite while locked",
    )
    assert main_window.mechanism_state_network_dsl_raw() == changed_state

    main_window._undo()
    assert main_window.mechanism_state_network_dsl_raw() == changed_state
    assert main_window._status_label.text() == "Allow Editing to undo mechanism changes"

    main_window._undo_stack.undo()
    qt_app.processEvents()
    assert main_window.mechanism_state_network_dsl_raw() == baseline_state

    main_window._redo()
    assert main_window.mechanism_state_network_dsl_raw() == baseline_state
    assert main_window._status_label.text() == "Allow Editing to redo mechanism changes"


def test_locked_mode_preserves_notes_undo_redo(main_window, qt_app):
    notes_widget = main_window._mechanism_editor._notes_text
    baseline = "These are notes."

    assert main_window.mechanism_editing_locked() is True

    notes_widget.setPlainText(baseline)
    notes_widget.setFocus()
    qt_app.processEvents()
    _append_text(notes_widget, " More")
    changed_text = notes_widget.toPlainText()
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(main_window._undo_stack, "canUndo", lambda: False)
    monkeypatch.setattr(main_window._undo_stack, "canRedo", lambda: False)
    monkeypatch.setattr(QtWidgets.QApplication, "focusWidget", lambda: notes_widget)

    try:
        main_window._undo()
        assert notes_widget.toPlainText() == baseline

        main_window._redo()
        assert notes_widget.toPlainText() == changed_text
    finally:
        monkeypatch.undo()


def test_locked_user_facing_undo_redo_after_project_load_uses_notes_widget_local_history(
    main_window,
    tmp_path,
    monkeypatch,
    qt_app,
):
    reactions_widget = main_window._mechanism_editor._reactions_text
    notes_widget = main_window._mechanism_editor._notes_text
    loaded_payload = _project_payload(
        mechanism="reaction: A -> B; k=2.0",
        notes="loaded project notes",
    )

    _load_project_via_dialog(main_window, tmp_path, monkeypatch, loaded_payload)

    assert main_window.mechanism_editing_locked() is True
    assert reactions_widget.toPlainText() == loaded_payload["mechanism"]
    assert notes_widget.toPlainText() == loaded_payload["notes"]
    assert main_window._undo_stack.canUndo() is False

    notes_widget.setFocus()
    qt_app.processEvents()
    _append_text(notes_widget, " More")
    changed_notes = notes_widget.toPlainText()
    monkeypatch.setattr(QtWidgets.QApplication, "focusWidget", lambda: notes_widget)

    main_window._undo()
    assert reactions_widget.toPlainText() == loaded_payload["mechanism"]
    assert notes_widget.toPlainText() == loaded_payload["notes"]

    main_window._redo()
    assert reactions_widget.toPlainText() == loaded_payload["mechanism"]
    assert notes_widget.toPlainText() == changed_notes


def test_load_preset_updates_reactions_while_locked(main_window):
    reactions_widget = main_window._mechanism_editor._reactions_text

    assert reactions_widget.isReadOnly() is True

    main_window._load_preset_mechanism("M1")

    assert main_window.mechanism_editing_locked() is True
    assert reactions_widget.isReadOnly() is True
    assert reactions_widget.toPlainText().strip()


def test_load_preset_while_unlocked_relocks_and_invalidates(main_window, monkeypatch, qt_app):
    _unlock_reactions_editing(main_window, monkeypatch)
    calls = _transition_effect_recorder(main_window, monkeypatch)

    main_window._load_preset_mechanism("M1")
    qt_app.processEvents()

    assert main_window.mechanism_editing_locked() is True
    assert any(call.startswith("supersede:") for call in calls)
    assert "overlay" in calls
    assert main_window._mechanism_editor._reactions_text.toPlainText().strip()


def test_programmatic_load_always_locks_even_with_invalid_dsl(main_window, monkeypatch, qt_app):
    _unlock_reactions_editing(main_window, monkeypatch)

    main_window.set_mechanism_reactions_text_with_optional_undo(
        "this line does not parse",
        "Programmatic invalid load",
        record_undo=False,
    )
    main_window._on_programmatic_mechanism_load()
    _wait_for_mechanism_validity(main_window, qt_app, expected_valid=False)

    assert main_window.mechanism_editing_locked() is True
    assert main_window._mechanism_edit_lock_action.isChecked() is False
    assert main_window._mechanism_editor._reactions_text.isReadOnly() is True
    assert main_window._mechanism_editor.is_mechanism_valid() is False
    assert main_window.mechanism_reactions_text_raw() == "this line does not parse"


def test_programmatic_load_refreshes_temperature(main_window, monkeypatch, qt_app):
    _unlock_reactions_editing(main_window, monkeypatch)
    main_window._temperature_spinbox.setValue(298.15)
    main_window.set_mechanism_reactions_text_with_optional_undo(
        "T=400\nreaction: A -> B; k=1.0",
        "Programmatic valid load",
        record_undo=False,
    )

    main_window._on_programmatic_mechanism_load()
    qt_app.processEvents()

    assert main_window.mechanism_editing_locked() is True
    assert main_window._temperature_spinbox.value() == pytest.approx(400.0)


def test_programmatic_project_load_syncs_owner_canonical_texts(main_window, monkeypatch, qt_app):
    _unlock_reactions_editing(main_window, monkeypatch)
    state_text = _valid_state_network_dsl()
    payload = {
        "mechanism": "reaction: A -> B; k=3.0",
        "notes": "",
        "state_network": state_text,
    }

    assert main_window.apply_project_payload(payload, record_undo=False) is True
    _wait_for_mechanism_validity(main_window, qt_app, expected_valid=True)

    owner = main_window._mechanism_session_owner
    assert main_window.mechanism_editing_locked() is True
    assert owner.canonical_reactions_text == payload["mechanism"]
    assert owner.canonical_state_network_dsl == state_text
    assert main_window.mechanism_reactions_text_raw() == payload["mechanism"]
    assert main_window.mechanism_state_network_dsl_raw() == state_text


def test_programmatic_load_while_unlocked_does_not_hit_validation_gate(main_window, monkeypatch, qt_app):
    _unlock_reactions_editing(main_window, monkeypatch)
    try_lock_calls: list[str] = []
    invalid_text = "this line does not parse"

    monkeypatch.setattr(
        "kindred.io.resources.get_preset_mechanism",
        lambda _preset_id: invalid_text,
    )
    monkeypatch.setattr(
        main_window,
        "_try_lock_mechanism_editor",
        lambda: try_lock_calls.append("called") or False,
    )

    main_window._load_preset_mechanism("M1")
    _wait_for_mechanism_validity(main_window, qt_app, expected_valid=False)

    assert try_lock_calls == []
    assert main_window.mechanism_editing_locked() is True
    assert main_window._mechanism_edit_lock_action.isChecked() is False
    assert main_window._mechanism_editor._reactions_text.isReadOnly() is True
    assert main_window.mechanism_reactions_text_raw() == invalid_text
    assert main_window._mechanism_editor.is_mechanism_valid() is False


def test_programmatic_load_while_unlocked_restores_guarded_main_window_work(main_window, monkeypatch, qt_app):
    _unlock_reactions_editing(main_window, monkeypatch)
    calls = _transition_effect_recorder(main_window, monkeypatch)

    main_window.set_mechanism_reactions_text_with_optional_undo(
        "reaction: A -> B; k=1.0",
        "Programmatic valid load",
        record_undo=False,
    )
    main_window._on_programmatic_mechanism_load()
    qt_app.processEvents()

    assert main_window.mechanism_editing_locked() is True
    assert calls.count("temperature") == 1
    assert calls.count("overlay") == 1
    assert any(call.startswith("supersede:") for call in calls)

    calls.clear()
    main_window._mechanism_editor._reactions_text.setPlainText("T=410\nreaction: A -> B; k=1.0")
    qt_app.processEvents()
    assert "temperature" in calls
    assert "overlay" in calls
    assert any(call.startswith("supersede:") for call in calls)


def test_unlocked_draft_edit_does_not_schedule_validation(main_window, monkeypatch, qt_app):
    _unlock_reactions_editing(main_window, monkeypatch)

    main_window._mechanism_editor._reactions_text.setPlainText("this line does not parse")
    _process_events_bounded(qt_app, iterations=1)

    assert "Validating" not in main_window._mechanism_editor._validation_label.text()
    assert "Editing draft" in main_window._mechanism_editor._validation_label.text()

    deadline = time.monotonic() + 0.8
    while time.monotonic() < deadline:
        _process_events_bounded(qt_app, iterations=1)
        time.sleep(0.01)

    assert main_window.mechanism_editing_locked() is False
    assert main_window._mechanism_editor.is_mechanism_valid() is False
    label_text = main_window._mechanism_editor._validation_label.text()
    assert "Editing draft" in label_text
    assert "pause" not in label_text.lower()
    assert "Error:" not in label_text


def test_failed_lock_attempt_surfaces_invalid_draft_error(main_window, monkeypatch, qt_app):
    _unlock_reactions_editing(main_window, monkeypatch)

    main_window._mechanism_editor._reactions_text.setPlainText("this line does not parse")
    _process_events_bounded(qt_app, iterations=1)

    assert "Editing draft" in main_window._mechanism_editor._validation_label.text()
    assert "Error:" not in main_window._mechanism_editor._validation_label.text()

    main_window._mechanism_edit_lock_action.trigger()
    qt_app.processEvents()

    assert main_window.mechanism_editing_locked() is False
    assert main_window._mechanism_edit_lock_action.isChecked() is True
    assert main_window._mechanism_editor.is_mechanism_valid() is False
    assert "Error:" in main_window._mechanism_editor._validation_label.text()


def test_preview_run_not_gated_by_validity(main_window, qt_app):
    main_window._force_lock_editor()
    main_window._mechanism_editor.set_reactions_text("this line does not parse", block_signals=True)
    main_window.simulation_controller.run_simulation_internal = MagicMock()
    main_window.simulation_controller.submit_slider_preview_replay_intent(
        SliderReplayIntent(
            target_set_ids=tuple(main_window._preview_session.effective_slider_edit_target_set_ids()),
            source="test",
        ),
    )
    main_window.simulation_controller._simulation_worker = None
    main_window.simulation_controller._simulation_running = False
    main_window._run_btn.setEnabled(True)

    main_window.simulation_controller.launch_pending_slider_preview_replay()
    qt_app.processEvents()

    main_window.simulation_controller.run_simulation_internal.assert_called_once()
    kwargs = main_window.simulation_controller.run_simulation_internal.call_args.kwargs
    assert kwargs["fast_mode"] is True


def test_pending_init_migration_rewrites_reactions_while_locked(main_window, qt_app):
    reactions_widget = main_window._mechanism_editor._reactions_text
    rewrite = "\n".join(
        [
            "reaction: A -> B; k=1.0",
            "# Initial concentrations moved to Batch Initial Conditions table (set1). Edit there.",
        ]
    )

    applied = main_window.apply_pending_init_migration(
        seed_sets={"set1": {"A": 1.0}},
        rewrite=rewrite,
    )
    qt_app.processEvents()

    assert applied is True
    assert main_window.mechanism_editing_locked() is True
    assert reactions_widget.isReadOnly() is True
    assert reactions_widget.toPlainText() == rewrite


def test_pending_init_migration_while_unlocked_force_locks_without_validation_gate(
    main_window,
    monkeypatch,
    qt_app,
):
    _unlock_reactions_editing(main_window, monkeypatch)
    try_lock_calls: list[str] = []
    rewrite = "this line does not parse"

    monkeypatch.setattr(
        main_window,
        "_try_lock_mechanism_editor",
        lambda: try_lock_calls.append("called") or False,
    )

    applied = main_window.apply_pending_init_migration(
        seed_sets={"set1": {"A": 1.0}},
        rewrite=rewrite,
    )
    _wait_for_mechanism_validity(main_window, qt_app, expected_valid=False)

    assert try_lock_calls == []
    assert applied is True
    assert main_window.mechanism_editing_locked() is True
    assert main_window._mechanism_edit_lock_action.isChecked() is False
    assert main_window.mechanism_reactions_text_raw() == rewrite
    assert main_window._mechanism_editor.is_mechanism_valid() is False


def test_pending_init_migration_while_unlocked_dispatches_consumers_and_refreshes_temperature_control(
    main_window,
    monkeypatch,
    qt_app,
):
    _unlock_reactions_editing(main_window, monkeypatch)
    calls = _transition_effect_recorder(main_window, monkeypatch)
    main_window._temperature_spinbox.setValue(298.15)

    applied = main_window.apply_pending_init_migration(
        seed_sets={"set1": {"A": 1.0}},
        rewrite="T=400\nthis line does not parse",
    )
    _wait_for_mechanism_validity(main_window, qt_app, expected_valid=False)

    assert applied is True
    assert calls == ["temperature", "overlay"]
    assert main_window._temperature_spinbox.value() == pytest.approx(400.0)


def test_pending_init_migration_dispatches_consumers_after_force_lock(main_window, monkeypatch, qt_app):
    _unlock_reactions_editing(main_window, monkeypatch)
    events: list[str] = []
    original_set_locked = main_window._set_mechanism_edit_locked
    original_refresh = main_window._refresh_authoritative_mechanism_derived_ui

    def _record_set_locked(locked: bool) -> bool:
        events.append(f"lock:{bool(locked)}")
        return original_set_locked(bool(locked))

    def _record_transition_refresh() -> None:
        events.append("transition_refresh")
        original_refresh()

    monkeypatch.setattr(main_window, "_set_mechanism_edit_locked", _record_set_locked)
    monkeypatch.setattr(
        main_window,
        "_refresh_authoritative_mechanism_derived_ui",
        _record_transition_refresh,
    )

    applied = main_window.apply_pending_init_migration(
        seed_sets={"set1": {"A": 1.0}},
        rewrite="reaction: A -> B; k=2.0",
    )
    qt_app.processEvents()

    assert applied is True
    assert "lock:True" in events
    assert "transition_refresh" in events
    assert events.index("lock:True") < events.index("transition_refresh")


def test_unlock_warning_and_state_network_banner_describe_draft_semantics(main_window, monkeypatch):
    captured: dict[str, str] = {}
    original_set_text = QtWidgets.QMessageBox.setText
    original_set_informative_text = QtWidgets.QMessageBox.setInformativeText

    def _record_text(box, text: str) -> None:
        captured["text"] = str(text)
        original_set_text(box, text)

    def _record_informative_text(box, text: str) -> None:
        captured["informative"] = str(text)
        original_set_informative_text(box, text)

    monkeypatch.setattr(QtWidgets.QMessageBox, "setText", _record_text)
    monkeypatch.setattr(QtWidgets.QMessageBox, "setInformativeText", _record_informative_text)
    monkeypatch.setattr(QtWidgets.QMessageBox, "exec", lambda self: None)

    warning_result = main_window._prompt_mechanism_edit_unlock_warning()
    banner = main_window._state_network_dialog_info_text(locked=False)

    assert warning_result is False
    assert "draft" in captured["text"].lower()
    assert "lock" in captured["informative"].lower()
    assert "&" not in captured["text"]
    assert "&" not in captured["informative"]
    assert "draft" in banner.lower()
    assert "lock" in banner.lower()
    assert "&" not in banner


def test_authoritative_editor_rewrite_updates_reactions_while_locked(main_window):
    reactions_widget = main_window._mechanism_editor._reactions_text

    main_window._set_authoritative_mechanism_editor_texts(
        reactions_text="reaction: A -> B; k=2.0",
        state_network_dsl="",
        description="Programmatic rewrite while locked",
    )

    assert main_window.mechanism_editing_locked() is True
    assert reactions_widget.isReadOnly() is True
    assert main_window.mechanism_reactions_text_raw() == "reaction: A -> B; k=2.0"
