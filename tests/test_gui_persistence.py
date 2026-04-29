import os

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


def test_project_state_omits_legacy_advanced_flag(main_window):
    """Serialized projects no longer persist the legacy Advanced DSL toggle."""
    payload = main_window._serialize_project_state()
    assert 'use_advanced_dsl' not in payload

    legacy_payload = dict(payload)
    legacy_payload['use_advanced_dsl'] = False
    main_window._apply_project_payload(legacy_payload)


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

        stored = window._settings.value("profiles/active", "", type=str)
        assert stored == "Fast"
    finally:
        window._settings.clear()
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
