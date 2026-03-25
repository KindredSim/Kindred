import pytest
from PySide6 import QtCore, QtGui, QtWidgets
from shiboken6 import isValid

from kindred.gui.main_window import MainWindow

pytestmark = [pytest.mark.gui]

_NONDEFAULT_DOCK_AREAS = {
    "_mechanism_dock": QtCore.Qt.RightDockWidgetArea,
    "_sliders_dock": QtCore.Qt.RightDockWidgetArea,
    "_batch_dock": QtCore.Qt.RightDockWidgetArea,
    "_right_dock": QtCore.Qt.LeftDockWidgetArea,
    "_analysis_dock": QtCore.Qt.LeftDockWidgetArea,
}
_LEFT_STACK_ATTRS = ("_mechanism_dock", "_sliders_dock", "_batch_dock")
_RIGHT_STACK_ATTRS = ("_right_dock", "_analysis_dock")


def _get_recent_menu(window) -> QtWidgets.QMenu:
    menu = window.config_controller.get_recent_menu(force_recreate=False)
    assert menu is not None
    assert isValid(menu)
    return menu


def _assert_vertical_stack_contract(main_window, dock_attrs: tuple[str, ...], area) -> None:
    docks = [getattr(main_window, attr) for attr in dock_attrs]
    geometries = [dock.geometry() for dock in docks]

    for dock, geometry in zip(docks, geometries):
        assert main_window.dockWidgetArea(dock) == area
        assert dock.isFloating() is False
        assert dock.isHidden() is False
        assert geometry.isValid() is True
        assert geometry.width() > 0
        assert geometry.height() > 0

    first = geometries[0]
    for geometry in geometries[1:]:
        assert geometry.left() == first.left()
        assert geometry.width() == first.width()

    for earlier, later in zip(geometries, geometries[1:]):
        assert later.top() >= earlier.bottom()


def _assert_default_shell_contract(main_window) -> None:
    assert main_window.centralWidget() is main_window._plot_tabs
    _assert_vertical_stack_contract(main_window, _LEFT_STACK_ATTRS, QtCore.Qt.LeftDockWidgetArea)
    _assert_vertical_stack_contract(main_window, _RIGHT_STACK_ATTRS, QtCore.Qt.RightDockWidgetArea)


def _arrange_analysis_dock_into_mechanism_region(main_window, qt_app) -> None:
    main_window.show()
    qt_app.processEvents()
    main_window._reset_layout()
    qt_app.processEvents()
    main_window.splitDockWidget(main_window._mechanism_dock, main_window._analysis_dock, QtCore.Qt.Vertical)
    qt_app.processEvents()


def test_recent_files_menu_populates_placeholder_when_empty(main_window):
    main_window._settings.setValue("recent_files", [])
    main_window._settings.sync()

    main_window.config_controller.update_recent_files_menu()
    menu = _get_recent_menu(main_window)
    actions = [action for action in menu.actions() if not action.isSeparator()]

    assert actions, "Expected Recent Projects submenu to contain at least one action"
    assert actions[0].text() == "No recent projects"
    assert not actions[0].isEnabled()


def test_add_to_recent_files_dedupes_and_caps(main_window, tmp_path):
    main_window._settings.setValue("recent_files", [])
    main_window._settings.sync()

    paths = [tmp_path / f"proj_{idx:02d}.kin" for idx in range(12)]
    for path in paths:
        path.write_text("{}", encoding="utf-8")
        main_window.config_controller.add_to_recent_files(str(path))

    recent_files = main_window._settings.value("recent_files", [])
    assert isinstance(recent_files, list)
    assert len(recent_files) == 10
    assert recent_files[0].endswith("proj_11.kin")
    assert recent_files[-1].endswith("proj_02.kin")

    main_window.config_controller.add_to_recent_files(str(paths[5]))
    recent_files = main_window._settings.value("recent_files", [])
    assert isinstance(recent_files, list)
    assert len(recent_files) == 10
    assert recent_files[0].endswith("proj_05.kin")


def test_toggle_theme_persists_setting(main_window):
    original_checked = bool(main_window._dark_mode_action.isChecked())
    main_window._dark_mode_action.setChecked(not original_checked)

    main_window.config_controller.toggle_theme()

    assert main_window._dark_mode == (not original_checked)
    assert main_window._settings.value("ui/dark_mode", True, type=bool) == (not original_checked)


def test_config_controller_port_exposes_bounded_surface_without_raw_main_window(main_window):
    port = main_window.config_controller._ui

    assert not hasattr(port, "main_window")
    assert port.parent is main_window
    assert port.settings() is main_window._settings
    assert port.dark_mode_action() is main_window._dark_mode_action
    assert port.debug_sliders_action() is main_window._debug_sliders_action


def test_save_then_load_settings_round_trip(main_window):
    settings = main_window._settings
    settings.clear()
    settings.sync()

    main_window._temperature_spinbox.setValue(310.0)
    main_window._sim_time_spinbox.setText("12.5")
    main_window._num_points_spinbox.setValue(123)
    main_window._mechanism_editor.set_slider_points_value(250)
    main_window._mechanism_editor.set_slider_solver_value("BDF")
    main_window._use_sparse_jacobian = True
    main_window._wegscheider_cyclicity_enabled = True
    main_window.simulation_controller.parallel_batch.max_parallel_workers = 4
    main_window.simulation_controller.parallel_batch.limit_blas_threads_per_worker = False

    main_window.config_controller.save_settings()

    main_window._temperature_spinbox.setValue(298.15)
    main_window._sim_time_spinbox.setText("1.0")
    main_window._num_points_spinbox.setValue(1)
    main_window._mechanism_editor.set_slider_points_value(100)
    main_window._mechanism_editor.set_slider_solver_value("LSODA")
    main_window._use_sparse_jacobian = False
    main_window._wegscheider_cyclicity_enabled = False
    main_window.simulation_controller.parallel_batch.max_parallel_workers = 1
    main_window.simulation_controller.parallel_batch.limit_blas_threads_per_worker = True

    main_window.config_controller.load_settings()

    assert main_window._temperature_spinbox.value() == pytest.approx(310.0)
    assert main_window._sim_time_spinbox.text().strip() == "12.5"
    assert main_window._num_points_spinbox.value() == 123
    assert main_window._mechanism_editor.slider_points_value() == 250
    assert main_window._mechanism_editor.slider_solver_value() == "BDF"
    assert main_window._use_sparse_jacobian is True
    assert main_window._wegscheider_cyclicity_enabled is True
    assert main_window.simulation_controller.parallel_batch.max_parallel_workers == 4
    assert main_window.simulation_controller.parallel_batch.limit_blas_threads_per_worker is False


def test_insert_preference_surface_removed(main_window):
    assert not hasattr(main_window, "_insert_preference")
    assert not hasattr(main_window, "_reset_insert_preference")
    assert main_window.findChild(QtGui.QAction, "resetInsertAction") is None


def test_legacy_insert_preference_setting_is_ignored(main_window):
    settings = main_window._settings
    settings.setValue("ui/insert_preference", "append")
    settings.sync()

    if hasattr(main_window, "_insert_preference"):
        delattr(main_window, "_insert_preference")

    main_window.config_controller.load_settings()

    assert not hasattr(main_window, "_insert_preference")


def test_solver_settings_persist_across_restart_and_restore_visible_combo(main_window, qt_app):
    settings = main_window._settings
    settings.clear()
    settings.sync()

    main_window._initial_solver = "BDF"
    main_window._initial_rtol = 1e-5
    main_window._initial_atol = 1e-9

    main_window.config_controller.save_settings()
    main_window.close()
    qt_app.processEvents()

    restored = MainWindow()
    try:
        assert restored._initial_solver == "BDF"
        assert restored._initial_rtol == pytest.approx(1e-5)
        assert restored._initial_atol == pytest.approx(1e-9)
        assert restored._solver_method_combo.currentText() == "BDF"
    finally:
        restored.close()
        qt_app.processEvents()


def test_slider_preview_preferences_persist_across_restart(main_window, qt_app):
    settings = main_window._settings
    settings.clear()
    settings.sync()

    main_window._mechanism_editor.set_slider_points_value(350)
    main_window._mechanism_editor.set_slider_solver_value("BDF")

    main_window.config_controller.save_settings()
    main_window.close()
    qt_app.processEvents()

    restored = MainWindow()
    try:
        assert restored._mechanism_editor.slider_points_value() == 350
        assert restored._mechanism_editor.slider_solver_value() == "BDF"
    finally:
        restored.close()
        qt_app.processEvents()


def test_ribbon_collapsed_state_persists_across_restart(main_window, qt_app):
    settings = main_window._settings
    settings.clear()
    settings.sync()

    ribbon_host = getattr(main_window, "_ribbon_host", None)
    assert ribbon_host is not None
    assert hasattr(ribbon_host, "set_collapsed")
    assert callable(ribbon_host.set_collapsed)
    assert hasattr(main_window, "ribbon_collapsed")
    assert callable(main_window.ribbon_collapsed)

    ribbon_host.set_collapsed(True)
    qt_app.processEvents()
    assert main_window.ribbon_collapsed() is True

    main_window.config_controller.save_settings()
    main_window.close()
    qt_app.processEvents()

    restored = MainWindow()
    try:
        assert restored.ribbon_collapsed() is True
        restored_host = getattr(restored, "_ribbon_host", None)
        assert restored_host is not None
        assert restored_host.is_collapsed() is True
    finally:
        restored.close()
        qt_app.processEvents()


def test_saved_maximized_state_stays_hidden_until_explicit_show(qt_app):
    settings = QtCore.QSettings("Kindred", "KindredGUI")
    settings.clear()
    settings.setValue("window/is_maximized", True)
    settings.sync()

    restored = MainWindow()
    try:
        assert restored.isVisible() is False

        restored.show()
        qt_app.processEvents()

        assert restored.isMaximized() is True
    finally:
        restored.close()
        settings.clear()
        settings.sync()
        qt_app.processEvents()


def test_maximized_state_round_trips_via_window_settings(main_window, qt_app):
    settings = main_window._settings
    settings.clear()
    settings.sync()

    main_window.show()
    main_window.showMaximized()
    qt_app.processEvents()

    assert main_window.isMaximized() is True

    main_window.config_controller.save_settings()

    assert settings.value("window/is_maximized", False, type=bool) is True

    main_window.showNormal()
    qt_app.processEvents()
    assert main_window.isMaximized() is False

    main_window.config_controller.load_settings()
    qt_app.processEvents()

    assert main_window.isMaximized() is True


@pytest.mark.parametrize("dock_attr", ["_mechanism_dock", "_sliders_dock", "_batch_dock", "_right_dock", "_analysis_dock"])
def test_dock_visibility_round_trips_via_window_state(main_window, qt_app, dock_attr):
    settings = main_window._settings
    settings.clear()
    settings.sync()

    dock = getattr(main_window, dock_attr)
    dock.hide()
    qt_app.processEvents()

    main_window.config_controller.save_settings()

    dock.show()
    qt_app.processEvents()
    assert dock.isHidden() is False

    main_window.config_controller.load_settings()
    qt_app.processEvents()

    assert dock.isHidden() is True


@pytest.mark.parametrize("dock_attr", ["_mechanism_dock", "_sliders_dock", "_batch_dock", "_right_dock", "_analysis_dock"])
def test_dock_floating_state_round_trips_via_window_state(main_window, qt_app, dock_attr):
    settings = main_window._settings
    settings.clear()
    settings.sync()

    dock = getattr(main_window, dock_attr)
    dock.show()
    dock.setFloating(True)
    dock.resize(420, 280)
    qt_app.processEvents()

    main_window.config_controller.save_settings()

    dock.setFloating(False)
    qt_app.processEvents()
    assert dock.isFloating() is False

    main_window.config_controller.load_settings()
    qt_app.processEvents()

    assert dock.isFloating() is True
    assert dock.isVisible() is True


def test_unsafe_restored_floating_dock_state_redocks_to_last_area(main_window, qt_app, monkeypatch):
    settings = main_window._settings
    settings.clear()
    settings.sync()

    dock = main_window._right_dock
    target_area = _NONDEFAULT_DOCK_AREAS["_right_dock"]

    main_window.show()
    qt_app.processEvents()
    main_window.addDockWidget(target_area, dock)
    dock.show()
    dock.setFloating(True)
    qt_app.processEvents()

    main_window.config_controller.save_settings()

    dock.setFloating(False)
    main_window.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)
    qt_app.processEvents()
    assert dock.isFloating() is False
    assert main_window.dockWidgetArea(dock) == QtCore.Qt.RightDockWidgetArea

    monkeypatch.setattr(
        MainWindow,
        "_is_restored_floating_dock_unsafe",
        lambda self, current_dock: current_dock is dock,
        raising=False,
    )

    main_window.config_controller.load_settings()
    qt_app.processEvents()

    assert dock.isFloating() is False
    assert dock.isVisible() is True
    assert main_window.dockWidgetArea(dock) == target_area


def test_unsafe_restored_floating_dock_recovery_preserves_maximized_main_window(main_window, qt_app, monkeypatch):
    settings = main_window._settings
    settings.clear()
    settings.sync()

    dock = main_window._right_dock
    target_area = _NONDEFAULT_DOCK_AREAS["_right_dock"]

    main_window.show()
    main_window.showMaximized()
    qt_app.processEvents()

    main_window.addDockWidget(target_area, dock)
    dock.show()
    dock.setFloating(True)
    qt_app.processEvents()

    main_window.config_controller.save_settings()

    original_set_floating = type(dock).setFloating

    def _drop_maximized_on_redock(current_dock, floating):
        original_set_floating(current_dock, floating)
        if current_dock is dock and not floating:
            main_window.showNormal()

    monkeypatch.setattr(type(dock), "setFloating", _drop_maximized_on_redock, raising=False)
    monkeypatch.setattr(
        MainWindow,
        "_is_restored_floating_dock_unsafe",
        lambda self, current_dock: current_dock is dock,
        raising=False,
    )

    main_window.config_controller.load_settings()
    qt_app.processEvents()

    assert dock.isFloating() is False
    assert dock.isVisible() is True
    assert main_window.dockWidgetArea(dock) == target_area
    assert main_window.isMaximized() is True


@pytest.mark.parametrize(
    ("dock_attr", "target_area"),
    [
        ("_mechanism_dock", _NONDEFAULT_DOCK_AREAS["_mechanism_dock"]),
        ("_sliders_dock", _NONDEFAULT_DOCK_AREAS["_sliders_dock"]),
        ("_batch_dock", _NONDEFAULT_DOCK_AREAS["_batch_dock"]),
        ("_right_dock", _NONDEFAULT_DOCK_AREAS["_right_dock"]),
        ("_analysis_dock", _NONDEFAULT_DOCK_AREAS["_analysis_dock"]),
    ],
)
def test_dock_area_round_trips_via_window_state(main_window, qt_app, dock_attr, target_area):
    settings = main_window._settings
    settings.clear()
    settings.sync()

    main_window.show()
    qt_app.processEvents()

    dock = getattr(main_window, dock_attr)
    main_window.addDockWidget(target_area, dock)
    dock.show()
    qt_app.processEvents()

    assert main_window.dockWidgetArea(dock) == target_area

    main_window.config_controller.save_settings()

    main_window.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)
    qt_app.processEvents()

    main_window.config_controller.load_settings()
    qt_app.processEvents()

    assert main_window.dockWidgetArea(dock) == target_area
    assert dock.isHidden() is False


def test_default_side_first_arrangement_round_trips_via_window_state(main_window, qt_app):
    settings = main_window._settings
    settings.clear()
    settings.sync()

    main_window.show()
    qt_app.processEvents()
    main_window._reset_layout()
    qt_app.processEvents()
    _assert_default_shell_contract(main_window)

    main_window.config_controller.save_settings()
    main_window.addDockWidget(QtCore.Qt.LeftDockWidgetArea, main_window._analysis_dock)
    main_window._sliders_dock.hide()
    qt_app.processEvents()

    assert main_window.dockWidgetArea(main_window._analysis_dock) == QtCore.Qt.LeftDockWidgetArea
    assert main_window._sliders_dock.isHidden() is True

    main_window.config_controller.load_settings()
    qt_app.processEvents()

    _assert_default_shell_contract(main_window)


def test_splitter_state_round_trips_via_main_plot_workspace_splitter_even_with_dataset_tabs(qt_app, monkeypatch):
    original_find_child = MainWindow.findChild

    window = MainWindow()
    try:
        settings = window._settings
        settings.clear()
        settings.sync()

        dataset_panel = window._plot_tabs.add_dataset_tab("dataset-a")
        qt_app.processEvents()

        workspace_splitter = window._plot_tabs._main_plot._main_splitter
        assert workspace_splitter is not None
        dataset_splitter = dataset_panel._plot_panel._main_splitter
        assert dataset_splitter is not None
        assert dataset_splitter is not workspace_splitter

        def _misdirect_named_splitter_lookup(self, *args, **kwargs):
            if args and args[0] is QtWidgets.QSplitter:
                name = args[1] if len(args) > 1 else kwargs.get("name", "")
                if name == "mainPlotWorkspaceSplitter":
                    return dataset_splitter
            return original_find_child(self, *args, **kwargs)

        monkeypatch.setattr(MainWindow, "findChild", _misdirect_named_splitter_lookup)

        workspace_splitter.setSizes([900, 0])
        dataset_splitter.setSizes([2, 8])
        qt_app.processEvents()

        expected_workspace_state = workspace_splitter.saveState()
        dataset_splitter_state = dataset_splitter.saveState()

        window.config_controller.save_settings()

        stored_state = settings.value("window/splitter_state")
        assert stored_state == expected_workspace_state
        assert stored_state != dataset_splitter_state

        workspace_splitter.setSizes([500, 400])
        dataset_splitter.setSizes([8, 2])
        qt_app.processEvents()

        window.config_controller.load_settings()
        qt_app.processEvents()

        assert workspace_splitter.saveState() == expected_workspace_state
        assert dataset_splitter.saveState() != expected_workspace_state
    finally:
        window.close()
        qt_app.processEvents()


def test_compact_on_screen_sliders_dock_floating_state_round_trips_via_window_state(main_window, qt_app, monkeypatch):
    settings = main_window._settings
    settings.clear()
    settings.sync()

    class _FakeScreen:
        def availableGeometry(self) -> QtCore.QRect:
            return QtCore.QRect(0, 0, 1600, 900)

    monkeypatch.setattr(QtGui.QGuiApplication, "screens", lambda: [_FakeScreen()])

    dock = main_window._sliders_dock
    main_window.show()
    qt_app.processEvents()
    dock.show()
    dock.setFloating(True)
    dock.move(40, 40)
    dock.resize(320, 100)
    qt_app.processEvents()

    assert dock.frameGeometry().height() < 120

    main_window.config_controller.save_settings()

    dock.setFloating(False)
    qt_app.processEvents()
    assert dock.isFloating() is False

    main_window.config_controller.load_settings()
    qt_app.processEvents()

    assert dock.isFloating() is True
    assert dock.isVisible() is True


def test_explicit_startup_solver_overrides_survive_settings_load(main_window, qt_app):
    settings = QtCore.QSettings("Kindred", "KindredGUI")
    settings.clear()
    settings.setValue("simulation/solver", "LSODA")
    settings.setValue("simulation/rtol", "1e-4")
    settings.setValue("simulation/atol", "1e-7")
    settings.sync()

    restored = MainWindow(solver="BDF", rtol=1e-8, atol=1e-13)
    try:
        assert restored._initial_solver == "BDF"
        assert restored._initial_rtol == pytest.approx(1e-8)
        assert restored._initial_atol == pytest.approx(1e-13)
        assert restored._solver_method_combo.currentText() == "BDF"
    finally:
        restored.close()
        qt_app.processEvents()
        settings.clear()
        settings.sync()


def test_load_settings_invalid_integer_values_fall_back_and_log(main_window, caplog):
    settings = main_window._settings
    settings.clear()
    settings.setValue("simulation/max_parallel_batch_workers", "not-an-int")
    settings.setValue("simulation/result_cache_cap", "bad-cap")
    settings.setValue("simulation/preview_cache_cap", "bad-preview")
    settings.sync()

    default_result_cap = int(main_window.simulation_controller.batch_cache.result_cache.max_entries())
    default_preview_cap = int(main_window.simulation_controller.batch_cache.preview_cache.max_entries())

    with caplog.at_level("WARNING"):
        main_window.config_controller.load_settings()

    assert main_window.simulation_controller.parallel_batch.max_parallel_workers == 12
    assert int(main_window.simulation_controller.batch_cache.result_cache.max_entries()) == default_result_cap
    assert int(main_window.simulation_controller.batch_cache.preview_cache.max_entries()) == default_preview_cap
    assert "simulation/max_parallel_batch_workers='not-an-int'" in caplog.text
    assert "simulation/result_cache_cap='bad-cap'" in caplog.text
    assert "simulation/preview_cache_cap='bad-preview'" in caplog.text


def test_load_settings_invalid_tolerances_fall_back_without_crashing(main_window, caplog):
    settings = main_window._settings
    settings.clear()
    settings.setValue("simulation/rtol", "bad-rtol")
    settings.setValue("simulation/atol", "bad-atol")
    settings.sync()

    default_rtol = main_window._initial_rtol
    default_atol = main_window._initial_atol

    with caplog.at_level("WARNING"):
        main_window.config_controller.load_settings()

    assert main_window._initial_rtol == pytest.approx(default_rtol)
    assert main_window._initial_atol == pytest.approx(default_atol)
    assert "Invalid solver tolerance rtol='bad-rtol'" in caplog.text
    assert "Invalid solver tolerance atol='bad-atol'" in caplog.text
