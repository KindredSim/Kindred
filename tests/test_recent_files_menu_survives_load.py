import pytest
from PySide6 import QtCore, QtWidgets
from shiboken6 import isValid

from kindred.gui.main_window import MainWindow

pytestmark = [pytest.mark.gui]


def _recent_projects_menus(window: MainWindow) -> list[QtWidgets.QMenu]:
    return [
        menu
        for menu in window.menuBar().findChildren(QtWidgets.QMenu)
        if isValid(menu) and menu.title() == "Recent Projects"
    ]


def test_load_project_recovers_if_recent_menu_becomes_stale_during_update(tmp_path, monkeypatch, qt_app):
    """
    Regression: project load should not error if the Recent Projects submenu is
    deleted/stale by the time `_update_recent_files_menu()` attempts to clear it.

    This simulates the "became invalid after acquisition" failure mode by
    returning a menu, then immediately scheduling its deletion before the caller
    mutates it.
    """

    templates_dir = tmp_path / "templates"

    def _fake_templates_dir(_self):
        return templates_dir

    monkeypatch.setattr(
        "kindred.config.templates.TemplateManager._get_templates_directory",
        _fake_templates_dir,
    )

    critical_calls = []

    def _capture_critical(*args, **kwargs):
        critical_calls.append((args, kwargs))
        return QtWidgets.QMessageBox.StandardButton.Ok

    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", _capture_critical)

    project_path = tmp_path / "recent_menu_stale_on_load.kin"
    project_path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getOpenFileName",
        lambda *args, **kwargs: (str(project_path), "Kindred Project (*.kin)"),
    )
    monkeypatch.setattr(MainWindow, "_apply_project_payload", lambda self, data, record_undo=True: None)

    window = MainWindow()
    try:
        window._settings.clear()
        window._settings.sync()

        recent_menus = _recent_projects_menus(window)
        assert recent_menus, "Expected Recent Projects submenu to exist"
        menu_to_invalidate = recent_menus[0]
        assert isValid(menu_to_invalidate)

        original_get_recent_menu = window.config_controller.get_recent_menu

        def _stale_recent_menu_getter(*args, **kwargs):
            if kwargs.get("force_recreate"):
                try:
                    return original_get_recent_menu(*args, **kwargs)
                except TypeError:
                    return original_get_recent_menu()

            menu = menu_to_invalidate
            menu.deleteLater()
            QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
            QtWidgets.QApplication.processEvents()
            return menu

        monkeypatch.setattr(window.config_controller, "get_recent_menu", _stale_recent_menu_getter)

        window.project_controller.load_project()

        assert not critical_calls, "Load should not show an error when recent menu becomes stale"

        recent_menus = _recent_projects_menus(window)
        assert len(recent_menus) == 1, "File menu should contain a single Recent Projects submenu"
        live_recent_menu = recent_menus[0]
        assert isValid(live_recent_menu), "Recent Projects submenu should be valid after load"

        recent_action_titles = [
            action.text()
            for action in live_recent_menu.actions()
            if not action.isSeparator()
        ]
        assert project_path.name in recent_action_titles
    finally:
        window._settings.clear()
        window._settings.sync()
        window.close()
        QtWidgets.QApplication.processEvents()
