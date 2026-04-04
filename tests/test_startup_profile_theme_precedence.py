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


def _seed_gui_settings(*, dark_mode: bool | None = None, active_profile: str = "") -> None:
    from PySide6 import QtCore

    settings = QtCore.QSettings("Kindred", "KindredGUI")
    settings.clear()
    if dark_mode is not None:
        settings.setValue("ui/dark_mode", bool(dark_mode))
    if active_profile:
        settings.setValue("profiles/active", str(active_profile))
    settings.sync()


def _patch_profiles(monkeypatch, *, profiles) -> None:
    from kindred.config.profiles import ProfileManager

    def _fake_load_profiles(self) -> None:
        self._profiles = {profile.name: profile for profile in profiles}

    monkeypatch.setattr(ProfileManager, "load_profiles", _fake_load_profiles)


def _spy_theme_apply(monkeypatch) -> list[bool]:
    import kindred.gui.theme_manager as theme_manager

    calls: list[bool] = []
    original_apply = theme_manager.ThemeManager.apply

    def _spy_apply(self, dark_mode: bool) -> None:
        calls.append(bool(dark_mode))
        original_apply(self, dark_mode)

    monkeypatch.setattr(theme_manager.ThemeManager, "apply", _spy_apply)
    return calls


def _enable_profiles_menu(monkeypatch) -> None:
    """Make profiles_menu_getter return a real menu for tests that need profiles enabled."""
    import dataclasses

    from PySide6 import QtWidgets

    from kindred.gui.main_window import MainWindow

    _original = MainWindow._init_mixin_ports

    def _patched(self):
        _original(self)
        menu = QtWidgets.QMenu(self)
        self._profile_ports = dataclasses.replace(
            self._profile_ports, profiles_menu_getter=lambda: menu,
        )

    monkeypatch.setattr(MainWindow, "_init_mixin_ports", _patched)


@pytest.mark.gui
def test_stored_profile_theme_wins_over_persisted_dark_mode_on_startup(qt_app, monkeypatch, tmp_path) -> None:
    from kindred.config.profiles import Profile
    from kindred.gui.main_window import MainWindow

    _patch_templates_dir(monkeypatch, tmp_path)
    _patch_profiles(
        monkeypatch,
        profiles=[
            Profile(name="Stored Dark", solver_method="BDF", grid_n=200, theme="dark"),
        ],
    )
    _seed_gui_settings(dark_mode=False, active_profile="Stored Dark")
    apply_calls = _spy_theme_apply(monkeypatch)
    monkeypatch.setattr(MainWindow, "_add_to_recent_files", lambda self, path: None)
    _enable_profiles_menu(monkeypatch)

    window = MainWindow()
    try:
        active = window._profile_manager.get_active_profile()
        assert active is not None
        assert active.name == "Stored Dark"
        assert window._dark_mode is True
        assert window._dark_mode_action.isChecked() is True
        assert window._theme_manager.is_dark() is True
        assert apply_calls == [True]
    finally:
        window.close()
        _clear_gui_settings()


@pytest.mark.gui
def test_explicit_profile_theme_wins_over_persisted_settings_on_startup(qt_app, monkeypatch, tmp_path) -> None:
    from kindred.config.profiles import Profile
    from kindred.gui.main_window import MainWindow

    _patch_templates_dir(monkeypatch, tmp_path)
    _patch_profiles(
        monkeypatch,
        profiles=[
            Profile(name="Stored Dark", solver_method="BDF", grid_n=200, theme="dark"),
            Profile(name="Explicit Light", solver_method="Radau", grid_n=500, theme="default"),
        ],
    )
    _seed_gui_settings(dark_mode=True, active_profile="Stored Dark")
    apply_calls = _spy_theme_apply(monkeypatch)
    monkeypatch.setattr(MainWindow, "_add_to_recent_files", lambda self, path: None)
    _enable_profiles_menu(monkeypatch)

    window = MainWindow(profile="Explicit Light")
    try:
        active = window._profile_manager.get_active_profile()
        assert active is not None
        assert active.name == "Explicit Light"
        assert window._dark_mode is False
        assert window._dark_mode_action.isChecked() is False
        assert window._theme_manager.is_dark() is False
        assert apply_calls == []
    finally:
        window.close()
        _clear_gui_settings()


@pytest.mark.gui
def test_explicit_profile_bootstrap_preserves_explicit_startup_solver_overrides(qt_app, monkeypatch, tmp_path) -> None:
    from kindred.config.profiles import Profile
    from kindred.gui.main_window import MainWindow

    _patch_templates_dir(monkeypatch, tmp_path)
    _patch_profiles(
        monkeypatch,
        profiles=[
            Profile(name="Stored Dark", solver_method="BDF", grid_n=200, theme="dark"),
            Profile(name="Explicit Light", solver_method="Radau", grid_n=500, theme="default"),
        ],
    )
    _seed_gui_settings(dark_mode=True, active_profile="Stored Dark")
    monkeypatch.setattr(MainWindow, "_add_to_recent_files", lambda self, path: None)
    _enable_profiles_menu(monkeypatch)

    window = MainWindow(profile="Explicit Light", solver="LSODA", rtol=1e-8, atol=1e-13)
    try:
        active = window._profile_manager.get_active_profile()
        assert active is not None
        assert active.name == "Explicit Light"
        assert window._initial_solver == "LSODA"
        assert window._initial_rtol == pytest.approx(1e-8)
        assert window._initial_atol == pytest.approx(1e-13)
        assert window._solver_method_combo.currentText() == "LSODA"
        assert window._dark_mode is False
    finally:
        window.close()
        _clear_gui_settings()


@pytest.mark.gui
def test_persisted_profile_bootstrap_preserves_explicit_startup_solver_overrides(qt_app, monkeypatch, tmp_path) -> None:
    from kindred.config.profiles import Profile
    from kindred.gui.main_window import MainWindow

    _patch_templates_dir(monkeypatch, tmp_path)
    _patch_profiles(
        monkeypatch,
        profiles=[
            Profile(name="Stored Dark", solver_method="BDF", grid_n=200, theme="dark"),
        ],
    )
    _seed_gui_settings(dark_mode=False, active_profile="Stored Dark")
    monkeypatch.setattr(MainWindow, "_add_to_recent_files", lambda self, path: None)
    _enable_profiles_menu(monkeypatch)

    window = MainWindow(solver="LSODA", rtol=1e-8, atol=1e-13)
    try:
        active = window._profile_manager.get_active_profile()
        assert active is not None
        assert active.name == "Stored Dark"
        assert window._initial_solver == "LSODA"
        assert window._initial_rtol == pytest.approx(1e-8)
        assert window._initial_atol == pytest.approx(1e-13)
        assert window._solver_method_combo.currentText() == "LSODA"
        assert window._dark_mode is True
    finally:
        window.close()
        _clear_gui_settings()


@pytest.mark.gui
def test_persisted_dark_mode_remains_fallback_when_no_profile_applies(qt_app, monkeypatch, tmp_path) -> None:
    from kindred.config.profiles import Profile
    from kindred.gui.main_window import MainWindow

    _patch_templates_dir(monkeypatch, tmp_path)
    _patch_profiles(
        monkeypatch,
        profiles=[
            Profile(name="Unused Dark", solver_method="BDF", grid_n=200, theme="dark"),
        ],
    )
    _seed_gui_settings(dark_mode=True)
    apply_calls = _spy_theme_apply(monkeypatch)
    monkeypatch.setattr(MainWindow, "_add_to_recent_files", lambda self, path: None)
    _enable_profiles_menu(monkeypatch)

    window = MainWindow()
    try:
        assert window._profile_manager.get_active_profile() is None
        assert window._dark_mode is True
        assert window._dark_mode_action.isChecked() is True
        assert window._theme_manager.is_dark() is True
        assert apply_calls == [True]
    finally:
        window.close()
        _clear_gui_settings()


@pytest.mark.gui
def test_stranded_profile_not_applied_when_profiles_menu_hidden(qt_app, monkeypatch, tmp_path) -> None:
    """A profile persisted in QSettings must not be applied when the Profiles menu is hidden."""
    from PySide6 import QtCore

    from kindred.config.profiles import Profile
    from kindred.gui.main_window import MainWindow

    _patch_templates_dir(monkeypatch, tmp_path)
    _patch_profiles(
        monkeypatch,
        profiles=[
            Profile(name="Stranded Dark", solver_method="BDF", grid_n=200, theme="dark"),
        ],
    )
    _seed_gui_settings(dark_mode=False, active_profile="Stranded Dark")
    apply_calls = _spy_theme_apply(monkeypatch)
    monkeypatch.setattr(MainWindow, "_add_to_recent_files", lambda self, path: None)
    # Do NOT call _enable_profiles_menu — profiles_menu_getter returns None (hidden)

    window = MainWindow()
    try:
        # Profile must not be activated
        assert window._profile_manager.get_active_profile() is None
        # Persisted dark_mode=False must apply, not the profile's dark theme
        assert window._dark_mode is False
        assert window._dark_mode_action.isChecked() is False
        assert apply_calls == [False]
        # Stranded QSettings key must be cleared
        settings = QtCore.QSettings("Kindred", "KindredGUI")
        assert settings.value("profiles/active", "", type=str) == ""
    finally:
        window.close()
        _clear_gui_settings()
