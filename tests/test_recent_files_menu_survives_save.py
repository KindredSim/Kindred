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


def test_save_project_recovers_if_recent_menu_was_deleted(tmp_path, monkeypatch, qt_app):
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

    project_path = tmp_path / "recent_menu_deleted.kin"
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getSaveFileName",
        lambda *args, **kwargs: (str(project_path), "Kindred Project (*.kin)"),
    )

    window = MainWindow()
    try:
        window._settings.clear()
        window._settings.sync()

        recent_menus = _recent_projects_menus(window)
        assert recent_menus, "Expected Recent Projects submenu to exist"
        stale_recent_menu = recent_menus[0]
        window._recent_menu = stale_recent_menu
        stale_recent_menu.deleteLater()
        QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
        QtWidgets.QApplication.processEvents()
        assert not isValid(window._recent_menu)

        window.project_controller.save_project()

        assert project_path.exists(), "Project file should be written during save"
        assert not critical_calls, "Save should not show an error when recent menu was deleted"

        recent_menus = _recent_projects_menus(window)
        assert len(recent_menus) == 1, "File menu should contain a single Recent Projects submenu"
        live_recent_menu = recent_menus[0]
        assert isValid(live_recent_menu), "Recent Projects submenu should be valid after save"

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
