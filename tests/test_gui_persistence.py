import os
import math

import pytest
from PySide6 import QtCore, QtWidgets

from kindred.gui.main_window import MainWindow
from kindred.gui.tutorial_manager import launch_tutorial
from kindred.gui.widgets.tutorial_overlay import TutorialOverlay

pytestmark = [pytest.mark.gui, pytest.mark.slow]


@pytest.fixture(scope="session")
def qt_app():
    """Ensure a QApplication exists for GUI-driven tests."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QtCore.QStandardPaths.setTestModeEnabled(True)
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    QtCore.QCoreApplication.setApplicationName("KindredTests")
    QtCore.QCoreApplication.setOrganizationName("KindredTests")
    return app


def test_project_round_trip_includes_all_dsl(tmp_path, monkeypatch, qt_app):
    """Saving then loading restores reactions/algebra and migrates inline initials to batch sets."""
    reactions_text = "\n".join(
        [
            "reaction: A -> B; k=0.2",
            "initial: A=1.0",
            "initial: B=0.0",
            "# Algebra",
            "let rate_ratio = [B] / max([A], 1e-6)",
        ]
    )
    notes_text = "These are free-form notes (never parsed).\n"
    state_network_text = "\n".join([
        "state: S1, kind=GS, energy=0, energy_unit=kJ/mol, degeneracy=1",
        "state: TS1, kind=TS, energy=10, energy_unit=kJ/mol, degeneracy=1",
        "edge: S1,TS1",
    ])

    templates_dir = tmp_path / "templates"

    def _fake_templates_dir(_self):
        return templates_dir

    monkeypatch.setattr(
        "kindred.config.templates.TemplateManager._get_templates_directory",
        _fake_templates_dir,
    )
    monkeypatch.setattr(MainWindow, "_add_to_recent_files", lambda self, path: None)

    window = MainWindow()
    try:
        window._mechanism_editor._reactions_text.setPlainText(reactions_text)
        window._mechanism_editor._notes_text.setPlainText(notes_text)
        window._mechanism_editor._state_network_editor.set_state_network_dsl(state_network_text)

        project_path = tmp_path / "round_trip.kin"
        monkeypatch.setattr(
            QtWidgets.QFileDialog,
            "getSaveFileName",
            lambda *args, **kwargs: (str(project_path), "Kindred Project (*.kin)")
        )
        window.project_controller.save_project()
    finally:
        window.close()

    loader = MainWindow()
    try:
        monkeypatch.setattr(
            QtWidgets.QFileDialog,
            "getOpenFileName",
            lambda *args, **kwargs: (str(project_path), "Kindred Project (*.kin)")
        )
        loader.project_controller.load_project()

        expected_reactions_text = "\n".join(
            [
                "reaction: A -> B; k=0.2",
                "# Initial concentrations moved to Batch Initial Conditions table (set1). Edit there.",
                "# Algebra",
                "let rate_ratio = [B] / max([A], 1e-6)",
            ]
        )
        assert loader._mechanism_editor._reactions_text.toPlainText() == expected_reactions_text
        assert loader._mechanism_editor._reactions_text.isReadOnly() is True
        assert loader._mechanism_editor._notes_text.toPlainText() == notes_text
        assert loader._mechanism_editor._state_network_editor.get_state_network_dsl() == state_network_text
        assert loader._batch_store.set_names()[:1] == ["set1"]
        assert loader._batch_store.values_for_set("set1") == {"A": "1", "B": "0"}
    finally:
        loader.close()


def test_project_payload_without_state_network_clears_existing_state_network(main_window, qt_app):
    stale_state_network_text = "\n".join(
        [
            "state: S_old, kind=GS, energy=0, energy_unit=kJ/mol, degeneracy=1",
            "state: TS_old, kind=TS, energy=10, energy_unit=kJ/mol, degeneracy=1",
            "state: P_old, kind=GS, energy=-1, energy_unit=kJ/mol, degeneracy=1",
            "edge: S_old,TS_old",
            "edge: TS_old,P_old",
        ]
    )
    replacement_mechanism = "reaction: ProjectA -> ProjectB; k=0.4"

    main_window._mechanism_editor.set_reactions_text("reaction: OldA -> OldB; k=1", block_signals=True)
    main_window._mechanism_editor._state_network_editor.set_state_network_dsl(stale_state_network_text)
    main_window._sync_mechanism_session_owner_after_authoritative_widget_write(dispatch_consumers=False)
    assert main_window.mechanism_state_network_dsl_raw() == stale_state_network_text

    payload = dict(main_window._serialize_project_state())
    payload["mechanism_source"] = {
        "reactions_text": replacement_mechanism,
        "state_network_dsl": "",
    }
    payload["notes"] = ""
    payload["batch_initial_conditions"] = {}

    main_window._apply_project_payload(payload, record_undo=False)
    qt_app.processEvents()

    assert main_window.mechanism_reactions_text_raw() == replacement_mechanism
    assert main_window.mechanism_state_network_dsl_raw() == ""
    assert "# State Network" not in main_window.get_mechanism_text()
    assert stale_state_network_text not in main_window.get_mechanism_text()


def test_invalid_project_mechanism_source_is_rejected_before_session_reset(main_window, monkeypatch):
    payload = dict(main_window._serialize_project_state())
    payload["mechanism_source"] = {
        "reactions_text": "reaction: ReplacementA -> ReplacementB; k=1",
    }

    def fail_if_reset_runs() -> None:
        raise AssertionError("project session reset ran before mechanism source validation")

    monkeypatch.setattr(main_window, "_reset_project_apply_dirty_session_state", fail_if_reset_runs)

    with pytest.raises(ValueError, match="state_network_dsl"):
        main_window._apply_project_payload(payload, record_undo=False)


def test_project_mechanism_source_rejects_unknown_source_fields_before_session_reset(
    main_window,
    monkeypatch,
):
    payload = dict(main_window._serialize_project_state())
    payload["mechanism_source"] = {
        "reactions_text": "reaction: ReplacementA -> ReplacementB; k=1",
        "state_network_dsl": "",
        "mechanism_text": "reaction: LegacyA -> LegacyB; k=2",
    }

    def fail_if_reset_runs() -> None:
        raise AssertionError("project session reset ran before mechanism source validation")

    monkeypatch.setattr(main_window, "_reset_project_apply_dirty_session_state", fail_if_reset_runs)

    with pytest.raises(ValueError, match="mechanism_text"):
        main_window._apply_project_payload(payload, record_undo=False)


@pytest.mark.parametrize(
    ("field", "bad_value", "message"),
    [
        ("notes", {}, "notes"),
        ("batch_initial_conditions", [], "batch_initial_conditions"),
        ("num_points", "100", "num_points"),
        ("temperature_K", "298.15", "temperature_K"),
        ("rtol", math.nan, "rtol"),
    ],
)
def test_invalid_current_project_payload_fields_are_rejected_before_session_reset(
    main_window,
    monkeypatch,
    field,
    bad_value,
    message,
):
    payload = dict(main_window._serialize_project_state())
    payload["mechanism_source"] = {
        "reactions_text": "reaction: ReplacementA -> ReplacementB; k=1",
        "state_network_dsl": "",
    }
    payload[field] = bad_value

    def fail_if_reset_runs() -> None:
        raise AssertionError(f"project session reset ran before {field} validation")

    monkeypatch.setattr(main_window, "_reset_project_apply_dirty_session_state", fail_if_reset_runs)

    with pytest.raises((TypeError, ValueError), match=message):
        main_window._apply_project_payload(payload, record_undo=False)


def test_project_state_rejects_legacy_advanced_flag_before_session_reset(main_window, monkeypatch):
    """Serialized projects neither persist nor accept the removed Advanced DSL toggle."""
    payload = main_window._serialize_project_state()
    assert 'use_advanced_dsl' not in payload

    legacy_payload = dict(payload)
    legacy_payload['use_advanced_dsl'] = False

    def fail_if_reset_runs() -> None:
        raise AssertionError("project session reset ran before top-level payload validation")

    monkeypatch.setattr(main_window, "_reset_project_apply_dirty_session_state", fail_if_reset_runs)

    with pytest.raises(ValueError, match="use_advanced_dsl"):
        main_window._apply_project_payload(legacy_payload, record_undo=False)


def test_public_project_apply_validates_before_slider_transaction_guard(main_window, monkeypatch):
    payload = main_window._serialize_project_state()
    payload["project_schema_version"] = 5

    def fail_if_slider_guard_runs(*args, **kwargs) -> bool:
        raise AssertionError("slider transaction guard ran before project payload validation")

    monkeypatch.setattr(main_window, "_guard_slider_transaction_invalidation", fail_if_slider_guard_runs)

    with pytest.raises(ValueError, match="project_schema_version"):
        main_window.apply_project_payload(payload, record_undo=False)


def test_new_project_uses_complete_current_project_payload(main_window, monkeypatch, qt_app):
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        lambda *args, **kwargs: QtWidgets.QMessageBox.StandardButton.Discard,
    )

    main_window._mechanism_editor.set_reactions_text("reaction: ExistingA -> ExistingB; k=1", block_signals=True)
    main_window._sync_mechanism_session_owner_after_authoritative_widget_write(dispatch_consumers=False)
    main_window._apply_solver_runtime_state(solver="Radau", rtol=1e-6, atol=1e-12)
    main_window.config_controller.update_user_preference("solver", "BDF")

    main_window.project_controller.new_project()
    qt_app.processEvents()

    payload = main_window._serialize_project_state()
    assert set(payload) >= {"version", "solver_method", "solver_warning"}
    assert payload["solver"] == "BDF"
    assert payload["solver_method"] == "BDF"
    assert payload["solver_warning"] is None
    assert payload["mechanism_source"] == {"reactions_text": "", "state_network_dsl": ""}
    assert main_window.mechanism_reactions_text_raw() == ""
    assert main_window.mechanism_state_network_dsl_raw() == ""
    assert main_window._status_label.text() == "New project"


def test_profile_activation_updates_widgets(tmp_path, monkeypatch, qt_app):
    """Activating a profile updates solver/grid widgets and persists selection."""
    templates_dir = tmp_path / "templates_profile"

    def _fake_templates_dir(_self):
        return templates_dir

    monkeypatch.setattr(
        "kindred.config.templates.TemplateManager._get_templates_directory",
        _fake_templates_dir,
    )
    monkeypatch.setattr(MainWindow, "_add_to_recent_files", lambda self, path: None)

    window = MainWindow()
    try:
        profile = window._profile_manager.get_profile("Fast")
        assert profile is not None, "Expected bundled 'Fast' profile"

        window._activate_profile("Fast")

        assert window._num_points_spinbox.value() == profile.grid_n
        assert window._initial_solver == profile.solver_method
        assert f"Solver: {profile.solver_method}" in window._solver_summary_label.text()

        stored = window._settings_owner.qsettings.value("profiles/active", "", type=str)
        assert stored == "Fast"
    finally:
        window._settings_owner.qsettings.clear()
        window.close()


def test_tutorial_overlay_highlights_widget(tmp_path, monkeypatch, qt_app):
    """Launching tutorials finds actual widgets described in the steps."""
    templates_dir = tmp_path / "templates_tutorial"

    def _fake_templates_dir(_self):
        return templates_dir

    monkeypatch.setattr(
        "kindred.config.templates.TemplateManager._get_templates_directory",
        _fake_templates_dir,
    )
    monkeypatch.setattr(MainWindow, "_add_to_recent_files", lambda self, path: None)

    window = MainWindow()
    overlay: TutorialOverlay | None = None
    try:
        overlay = launch_tutorial(window, "getting_started")
        assert isinstance(overlay, TutorialOverlay)

        QtWidgets.QApplication.processEvents()
        overlay._next_btn.click()  # Move to Mechanism Editor step
        QtWidgets.QApplication.processEvents()

        highlight_rect = overlay._get_highlight_rect()
        assert highlight_rect is not None and not highlight_rect.isNull()
    finally:
        if overlay is not None:
            overlay._on_skip()
        window.close()
