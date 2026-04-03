from __future__ import annotations

import pytest


def _patch_templates_dir(monkeypatch, tmp_path) -> None:
    def _fake_templates_dir(_self):
        target = tmp_path / "templates"
        target.mkdir(parents=True, exist_ok=True)
        return target

    monkeypatch.setattr(
        "kindred.config.templates.TemplateManager._get_templates_directory",
        _fake_templates_dir,
    )


def _clear_gui_settings() -> None:
    from PySide6 import QtCore

    settings = QtCore.QSettings("Kindred", "KindredGUI")
    settings.clear()
    settings.sync()


@pytest.mark.unit
def test_build_profile_and_template_managers_loads_profiles_once(monkeypatch) -> None:
    from kindred.config.profiles import ProfileManager
    from kindred.gui.app_wiring import build_profile_and_template_managers

    calls: list[int] = []

    def _spy_load_profiles(self) -> None:
        calls.append(id(self))

    monkeypatch.setattr(ProfileManager, "load_profiles", _spy_load_profiles)

    managers = build_profile_and_template_managers()

    assert isinstance(managers.profile_manager, ProfileManager)
    assert len(calls) == 1


@pytest.mark.gui
def test_main_window_initializes_mixin_ports_once_after_menu_prerequisites(qt_app, monkeypatch, tmp_path) -> None:
    import kindred.gui.main_window as mw_mod

    _patch_templates_dir(monkeypatch, tmp_path)
    _clear_gui_settings()

    calls: list[dict[str, bool]] = []
    original = mw_mod.MainWindow._init_mixin_ports

    def _spy_init_mixin_ports(self) -> None:
        calls.append(
            {
                "has_profiles_menu": hasattr(self, "_profiles_menu"),
                "has_dark_mode_action": hasattr(self, "_dark_mode_action"),
                "has_status_label": hasattr(self, "_status_label"),
                "has_profile_indicator": hasattr(self, "_profile_indicator"),
            }
        )
        original(self)

    monkeypatch.setattr(mw_mod.MainWindow, "_init_mixin_ports", _spy_init_mixin_ports)

    window = mw_mod.MainWindow()
    try:
        assert calls == [
            {
                "has_profiles_menu": False,  # Hidden: Profiles menu removed from menu bar (entry point commented out in main_window.py)
                "has_dark_mode_action": True,
                "has_status_label": True,
                "has_profile_indicator": True,
            }
        ]
        assert window._profile_ports.dark_mode_action is window._dark_mode_action
        assert window._profile_ports.profiles_menu_getter() is None  # Hidden: Profiles menu removed
    finally:
        window.close()
        _clear_gui_settings()


@pytest.mark.gui
def test_main_window_load_settings_runs_after_bootstrap_prerequisites(qt_app, monkeypatch, tmp_path) -> None:
    import kindred.gui.main_window as mw_mod
    from kindred.gui.controllers.config_controller import ConfigController

    _patch_templates_dir(monkeypatch, tmp_path)
    _clear_gui_settings()

    init_count = {"value": 0}
    observed: dict[str, bool | int] = {}
    original_init_mixin_ports = mw_mod.MainWindow._init_mixin_ports

    def _spy_init_mixin_ports(self) -> None:
        init_count["value"] += 1
        original_init_mixin_ports(self)

    def _spy_load_settings(self) -> None:
        window = self._ui.parent
        observed.update(
            {
                "init_count": init_count["value"],
                "has_raw_main_window_backref": hasattr(self._ui, "main_window"),
                "has_theme_manager": hasattr(window, "_theme_manager"),
                "has_mechanism_dock": hasattr(window, "_mechanism_dock"),
                "has_sliders_dock": hasattr(window, "_sliders_dock"),
                "has_batch_dock": hasattr(window, "_batch_dock"),
                "has_right_dock": hasattr(window, "_right_dock"),
                "has_analysis_dock": hasattr(window, "_analysis_dock"),
                "has_status_label": hasattr(window, "_status_label"),
                "has_profile_indicator": hasattr(window, "_profile_indicator"),
                "has_profiles_menu": hasattr(window, "_profiles_menu"),
                "has_dark_mode_action": hasattr(window, "_dark_mode_action"),
                "has_ribbon_toolbar": hasattr(window, "_ribbon_toolbar"),
                "has_ribbon_host": hasattr(window, "_ribbon_host"),
                "profile_ports_ready": getattr(window, "_profile_ports", None) is not None
                and window._profile_ports.dark_mode_action is window._dark_mode_action,
            }
        )

    monkeypatch.setattr(mw_mod.MainWindow, "_init_mixin_ports", _spy_init_mixin_ports)
    monkeypatch.setattr(ConfigController, "load_settings", _spy_load_settings)

    window = mw_mod.MainWindow()
    try:
        assert observed == {
            "init_count": 1,
            "has_raw_main_window_backref": False,
            "has_theme_manager": True,
            "has_mechanism_dock": True,
            "has_sliders_dock": True,
            "has_batch_dock": True,
            "has_right_dock": True,
            "has_analysis_dock": True,
            "has_status_label": True,
            "has_profile_indicator": True,
            "has_profiles_menu": False,  # Hidden: Profiles menu removed (entry point commented out in main_window.py)
            "has_dark_mode_action": True,
            "has_ribbon_toolbar": False,  # Hidden: Ribbon removed (entry point commented out in main_window.py)
            "has_ribbon_host": False,  # Hidden: Ribbon removed (entry point commented out in main_window.py)
            "profile_ports_ready": True,
        }
    finally:
        window.close()
        _clear_gui_settings()
