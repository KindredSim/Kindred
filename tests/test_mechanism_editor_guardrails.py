import json
from unittest.mock import MagicMock

import pytest
from PySide6 import QtGui, QtWidgets


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




def _append_text(editor: QtWidgets.QPlainTextEdit, text: str) -> None:
    cursor = editor.textCursor()
    cursor.movePosition(QtGui.QTextCursor.MoveOperation.End)
    editor.setTextCursor(cursor)
    editor.insertPlainText(str(text))


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


def test_textchanged_suppressed_while_editing_unlocked(main_window, monkeypatch, qt_app):
    _unlock_reactions_editing(main_window, monkeypatch)
    refresh_events: list[str] = []
    main_window._invalidate_slider_runtime = lambda: refresh_events.append("invalidate")
    main_window._plot_tabs._main_plot.refresh_overlay_presentation_for_current_roster = lambda: refresh_events.append(
        "overlay"
    )

    main_window._temperature_spinbox.setValue(298.15)
    main_window._mechanism_editor._reactions_text.setPlainText("T=400\nreaction: A -> B; k=1.0")
    qt_app.processEvents()

    assert getattr(main_window, "_editing_suppression_active", False) is True
    assert refresh_events == []
    assert main_window._temperature_spinbox.value() == pytest.approx(298.15)

    main_window._mechanism_edit_lock_action.trigger()
    qt_app.processEvents()

    assert main_window.mechanism_editing_locked() is True
    assert refresh_events == ["invalidate", "overlay"]
    assert main_window._temperature_spinbox.value() == pytest.approx(400.0)


def test_relock_triggers_full_refresh(main_window, monkeypatch, qt_app):
    _unlock_reactions_editing(main_window, monkeypatch)
    refresh_events: list[str] = []
    main_window._invalidate_slider_runtime = lambda: refresh_events.append("invalidate")
    main_window._plot_tabs._main_plot.refresh_overlay_presentation_for_current_roster = lambda: refresh_events.append(
        "overlay"
    )

    main_window._temperature_spinbox.setValue(298.15)
    main_window._mechanism_editor._reactions_text.setPlainText("T=400\nreaction: A -> B; k=1.0")
    qt_app.processEvents()

    assert refresh_events == []
    assert main_window._temperature_spinbox.value() == pytest.approx(298.15)

    main_window._mechanism_edit_lock_action.trigger()
    qt_app.processEvents()

    assert main_window.mechanism_editing_locked() is True
    assert refresh_events == ["invalidate", "overlay"]
    assert main_window._temperature_spinbox.value() == pytest.approx(400.0)


def test_relock_refused_on_invalid_mechanism(main_window, monkeypatch, qt_app):
    _unlock_reactions_editing(main_window, monkeypatch)

    reactions_widget = main_window._mechanism_editor._reactions_text
    reactions_widget.setPlainText("this line does not parse")
    qt_app.processEvents()

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
    assert main_window.auto_lock_for_run() is False
    assert main_window.mechanism_editing_locked() is False
    assert main_window._mechanism_editor.is_mechanism_valid() is False


def test_run_auto_locks_editor_from_main_window(main_window, monkeypatch, qt_app):
    _unlock_reactions_editing(main_window, monkeypatch)
    main_window._mechanism_editor._reactions_text.setPlainText("reaction: A -> B; k=1.0")
    qt_app.processEvents()
    main_window.simulation_controller.run_simulation_internal = MagicMock()

    main_window.simulation_controller.run_simulation()
    qt_app.processEvents()

    assert main_window.mechanism_editing_locked() is True
    main_window.simulation_controller.run_simulation_internal.assert_called_once()


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
    invalidations: list[str] = []
    main_window._invalidate_slider_runtime = lambda: invalidations.append("invalidate")

    main_window._load_preset_mechanism("M1")
    qt_app.processEvents()

    assert main_window.mechanism_editing_locked() is True
    assert getattr(main_window, "_editing_suppression_active", False) is False
    assert invalidations
    assert main_window._mechanism_editor._reactions_text.toPlainText().strip()


def test_pending_init_migration_rewrites_reactions_while_locked(main_window, qt_app):
    reactions_widget = main_window._mechanism_editor._reactions_text
    rewrite = "\n".join(
        [
            "reaction: A -> B; k=1.0",
            "# Initial concentrations moved to Batch Initial Conditions table (set1). Edit there.",
        ]
    )

    applied = main_window.apply_pending_init_migration(
        seed={"A": 1.0},
        rewrite=rewrite,
    )
    qt_app.processEvents()

    assert applied is True
    assert main_window.mechanism_editing_locked() is True
    assert reactions_widget.isReadOnly() is True
    assert reactions_widget.toPlainText() == rewrite


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
