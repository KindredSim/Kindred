from __future__ import annotations

import inspect

import pytest
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from kindred.gui import main_window as main_window_module
from kindred.gui.simulation_batch_owner import SimulationBatchOwner


pytestmark = pytest.mark.gui

_ALL_DOCK_AREAS = QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea
_NONDEFAULT_DOCK_AREAS = {
    "_mechanism_dock": QtCore.Qt.RightDockWidgetArea,
    "_sliders_dock": QtCore.Qt.RightDockWidgetArea,
    "_batch_dock": QtCore.Qt.LeftDockWidgetArea,
    "_right_dock": QtCore.Qt.LeftDockWidgetArea,
    "_analysis_dock": QtCore.Qt.LeftDockWidgetArea,
}
_LEFT_STACK_ATTRS = ("_mechanism_dock", "_sliders_dock")
_RIGHT_STACK_ATTRS = ("_batch_dock", "_right_dock", "_analysis_dock")
_EXPECTED_DOCK_METADATA = {
    "_mechanism_dock": ("Mechanism", "mechanismDock"),
    "_sliders_dock": ("Interactive Sliders", "slidersDock"),
    "_batch_dock": ("Initial Conditions", "batchDock"),
    "_right_dock": ("Data", "rightDock"),
    "_analysis_dock": ("Analysis", "analysisDock"),
}


def _view_menu(main_window) -> QtWidgets.QMenu:
    view_menu = getattr(main_window, "_view_menu", None)
    assert view_menu is not None
    return view_menu


def _help_menu(main_window) -> QtWidgets.QMenu:
    help_menu = getattr(main_window, "_help_menu", None)
    assert help_menu is not None
    return help_menu


def _analysis_surface_actions(main_window) -> dict[str, QtGui.QAction]:
    view_menu = _view_menu(main_window)
    analysis_action = next(
        action for action in view_menu.actions() if action.menu() is not None and action.text() == "Analysis Surfaces"
    )
    submenu = analysis_action.menu()
    assert submenu is not None
    return {action.text(): action for action in submenu.actions() if not action.isSeparator()}


def _current_tab_text(tabs: QtWidgets.QTabWidget) -> str:
    return str(tabs.tabText(tabs.currentIndex()))


def _view_action(main_window, text: str) -> QtGui.QAction:
    view_menu = _view_menu(main_window)
    return next(action for action in view_menu.actions() if action.menu() is None and action.text() == text)


def _normalized_action_text(action: QtGui.QAction) -> str:
    return str(action.text()).replace("&", "")


def _ribbon_toolbar(main_window) -> QtWidgets.QToolBar:
    toolbar = getattr(main_window, "_ribbon_toolbar", None)
    assert isinstance(toolbar, QtWidgets.QToolBar)
    return toolbar


def _ribbon_host(main_window) -> QtWidgets.QWidget:
    host = getattr(main_window, "_ribbon_host", None)
    assert isinstance(host, QtWidgets.QWidget)
    return host


def _ribbon_page(host: QtWidgets.QWidget, title: str):
    assert hasattr(host, "page")
    page_getter = getattr(host, "page")
    assert callable(page_getter)
    page = page_getter(title)
    assert page is not None
    return page


def _all_shell_docks(main_window) -> tuple[QtWidgets.QDockWidget, ...]:
    return tuple(getattr(main_window, attr) for attr in (*_LEFT_STACK_ATTRS, *_RIGHT_STACK_ATTRS))


def _arrange_analysis_dock_into_mechanism_region(main_window, qt_app) -> None:
    main_window.show()
    qt_app.processEvents()
    main_window._reset_layout()
    qt_app.processEvents()
    main_window.splitDockWidget(main_window._mechanism_dock, main_window._analysis_dock, QtCore.Qt.Vertical)
    qt_app.processEvents()


def _assert_vertical_stack_contract(main_window, dock_attrs: tuple[str, ...], area) -> None:
    docks = [getattr(main_window, attr) for attr in dock_attrs]
    geometries = [dock.geometry() for dock in docks]

    for dock, geometry in zip(docks, geometries):
        assert main_window.dockWidgetArea(dock) == area
        assert dock.isFloating() is False
        assert dock.isVisible() is True
        assert geometry.isValid() is True
        assert geometry.width() > 0
        assert geometry.height() > 0

    first = geometries[0]
    for geometry in geometries[1:]:
        assert geometry.left() == first.left()
        assert abs(geometry.width() - first.width()) <= 64

    for earlier, later in zip(geometries, geometries[1:]):
        assert later.top() >= earlier.bottom()


def _assert_right_column_contract(main_window) -> None:
    """Batch on top, Data and Analysis tabified below."""
    batch = main_window._batch_dock
    data = main_window._right_dock
    analysis = main_window._analysis_dock
    area = QtCore.Qt.RightDockWidgetArea

    for dock in (batch, data, analysis):
        assert main_window.dockWidgetArea(dock) == area
        assert dock.isFloating() is False

    batch_geo = batch.geometry()
    data_geo = data.geometry()

    assert batch_geo.isValid() is True
    assert batch_geo.width() > 0
    assert batch_geo.height() > 0
    assert data_geo.isValid() is True
    assert data_geo.width() > 0
    assert data_geo.height() > 0

    # Batch is above the Data/Analysis tab group.
    assert data_geo.top() >= batch_geo.bottom()

    # Data and Analysis are tabified.
    tabified = main_window.tabifiedDockWidgets(data)
    assert analysis in tabified


def _assert_default_shell_contract(main_window) -> None:
    assert main_window.centralWidget() is main_window._plot_tabs
    _assert_vertical_stack_contract(main_window, _LEFT_STACK_ATTRS, QtCore.Qt.LeftDockWidgetArea)
    _assert_right_column_contract(main_window)


_SIMULATION_PORT_METHODS = {
    "dialogs": ("message_box_warning", "message_box_critical"),
    "settings": ("settings_set_value", "settings_sync"),
    "run_ui": (
        "run_button_is_enabled",
        "set_run_button_enabled",
        "set_runtime_backed_run_controls_ready",
        "set_status_text",
    ),
    "slider": (
        "is_mechanism_valid_for_preview",
        "preview_initials_for_row",
        "preview_batch_cache_token",
        "reset_mechanism_workspaces",
    ),
    "batch": (
        "batch_rows_for_scope",
        "batch_set_id_for_row",
        "batch_cache_key",
        "display_cached_batch_selection",
    ),
    "mechanism": (
        "auto_lock_for_run",
        "is_mechanism_ready_for_run",
        "mechanism_reactions_text_raw",
        "get_mechanism_text",
        "apply_parameter_overrides_to_dsl",
    ),
    "solver": (
        "parse_sim_time_seconds",
        "initial_solver_name",
        "use_sparse_jacobian",
    ),
    "runtime": (
        "prepare_slider_runtime",
        "apply_slider_overrides_to_bindings",
        "is_energy_mode_mechanism",
        "dsl_has_computational_mode_generated_block",
        "sync_energy_mode_temperature_from_mechanism",
        "populate_energy_mode_variables_from_mechanism",
        "extract_and_populate_variables",
    ),
    "results": (
        "set_data",
        "main_plot",
        "set_results_table",
    ),
    "provenance": (
        "set_last_simulation_provenance",
        "set_last_simulation_ctc",
    ),
    "mechanism_helpers": (
        "authoritative_structure_snapshot",
        "last_mechanism",
        "remember_last_mechanism",
    ),
}


def _assert_port_capabilities(port: object, *, port_name: str, methods: tuple[str, ...]) -> None:
    missing = [method for method in methods if not callable(getattr(port, method, None))]
    assert missing == [], f"{port_name} port missing callable methods: {missing}"


def test_main_window_simulation_controller_wires_explicit_simulation_port_capabilities(main_window) -> None:
    ui = main_window.simulation_controller.ui

    assert getattr(main_window, "_preview_session", None) is not None
    assert getattr(main_window, "_variable_runtime", None) is not None
    assert getattr(main_window, "_mechanism_helpers", None) is not None
    for port_name, methods in _SIMULATION_PORT_METHODS.items():
        _assert_port_capabilities(
            getattr(ui, port_name),
            port_name=port_name,
            methods=methods,
        )

    assert ui.run_ui is not main_window
    assert ui.dialogs is main_window._simulation_dialogs
    assert ui.settings is main_window._settings_owner
    assert ui.settings is not main_window
    assert ui.run_ui is main_window._simulation_run_ui_owner
    assert ui.results is main_window.results_controller
    assert ui.results is not main_window
    assert ui.provenance is main_window._simulation_provenance_owner
    assert ui.solver is main_window._simulation_solver_owner
    assert ui.solver is not main_window
    assert ui.mechanism is main_window._simulation_mechanism_owner
    assert ui.mechanism is not main_window
    assert ui.batch is main_window._simulation_batch_owner
    assert ui.batch is not main_window
    assert not hasattr(main_window, "_settings")
    assert not hasattr(main_window, "settings_set_value")
    assert not hasattr(main_window, "settings_remove")
    assert not hasattr(main_window, "settings_sync")
    assert not hasattr(main_window, "_run_button_requested_enabled")
    assert not hasattr(main_window, "_simulation_runtime_run_ready")
    assert not hasattr(main_window, "_last_simulation_provenance")
    assert not hasattr(main_window, "_last_simulation_ctc")
    assert not hasattr(main_window, "display_cached_batch_selection")
    assert not hasattr(main_window, "display_workspace_aware_batch_selection")
    assert not hasattr(main_window, "_display_workspace_aware_preview_batch_selection")
    assert not hasattr(main_window, "_resolve_workspace_aware_batch_selection")
    assert not hasattr(main_window, "_matching_preview_entry_for_workspace_set")
    assert not hasattr(main_window, "_record_current_main_plot_workspace_preview_provenance")
    assert not hasattr(main_window, "_displayed_workspace_preview_provenance_matches_current_workspace")
    assert not hasattr(main_window, "_batch_cache_entry_matches_plot_payload")
    assert not hasattr(main_window, "_current_workspace_preview_identity_payload")
    assert not hasattr(main_window, "_active_explicit_cache_entry_for_set")
    assert not hasattr(main_window, "_batch_cache_contains_set")
    assert not hasattr(main_window, "_purge_batch_cache_for_deleted_sets")
    assert not hasattr(main_window, "batch_result_cache_store")
    assert not hasattr(main_window, "auto_lock_for_run")
    assert not hasattr(main_window, "is_mechanism_ready_for_run")
    assert not hasattr(main_window, "mechanism_slider_points_value")
    assert not hasattr(main_window, "mechanism_slider_solver_value")
    assert not hasattr(main_window, "num_points_spinbox_value")
    assert not hasattr(main_window, "sim_time_spinbox_text")
    assert not hasattr(main_window, "use_sparse_jacobian")
    assert not hasattr(main_window, "initial_solver_name")
    assert not hasattr(main_window, "initial_rtol")
    assert not hasattr(main_window, "initial_atol")
    assert not hasattr(main_window, "dsl_global_temperature_K")
    assert ui.slider is main_window._preview_session
    assert ui.runtime is main_window._variable_runtime
    assert ui.mechanism_helpers is main_window._mechanism_helpers

    batch_owner_init = inspect.signature(SimulationBatchOwner.__init__).parameters
    assert "display_cached_batch_selection" not in batch_owner_init
    assert "display_workspace_aware_batch_selection" not in batch_owner_init

    settings_key = "tests/simulation_settings_owner_contract"
    try:
        ui.settings.settings_set_value(settings_key, "owner-value")
        ui.settings.settings_sync()
        assert main_window._settings_owner.qsettings.value(settings_key) == "owner-value"
    finally:
        main_window._settings_owner.settings_remove(settings_key)
        main_window._settings_owner.settings_sync()

    provenance = {"solver": "BDF"}
    ui.provenance.set_last_simulation_provenance(provenance)
    provenance["solver"] = "LSODA"
    assert ui.provenance.last_simulation_provenance == {"solver": "BDF"}
    returned_provenance = ui.provenance.last_simulation_provenance
    returned_provenance["solver"] = "mutated"
    assert ui.provenance.last_simulation_provenance == {"solver": "BDF"}

    ui.results.set_data(
        np.asarray([0.0, 1.0]),
        {"A": np.asarray([1.0, 0.5])},
        label="owner-contract",
        overlays=[],
    )
    assert main_window.main_plot_has_data() is True

    ui.run_ui.set_runtime_backed_run_controls_ready(False)
    ui.run_ui.set_run_button_enabled(True)
    assert ui.run_ui.requested_run_enabled is True
    assert ui.run_ui.runtime_ready is False
    assert main_window._run_btn.isEnabled() is False

    ui.run_ui.set_runtime_backed_run_controls_ready(True)
    assert main_window._run_btn.isEnabled() is True


def test_main_window_source_does_not_read_batch_cache_stores_directly() -> None:
    source = inspect.getsource(main_window_module.MainWindow)

    assert ".batch_cache.result_cache." not in source
    assert ".batch_cache.preview_cache." not in source


def test_main_window_preview_session_uses_bound_lifecycle_port_instead_of_main_window_controller_attr(
    main_window,
    monkeypatch,
) -> None:
    owner = main_window._preview_session
    controller = main_window.simulation_controller
    calls: list[str] = []

    monkeypatch.setattr(
        controller,
        "launch_pending_slider_preview_replay",
        lambda: calls.append("launch"),
    )

    class _PoisonController:
        def launch_pending_slider_preview_replay(self) -> None:
            raise AssertionError("Preview session should not launch previews through main_window._sim_controller")

    monkeypatch.setattr(main_window, "_sim_controller", _PoisonController())
    monkeypatch.setattr(main_window, "is_mechanism_valid_for_preview", lambda: True)

    owner._dispatch_variable_slider_preview_if_valid()

    assert calls == ["launch"]


def test_right_panel_exposes_dataset_lookup_without_private_widget_reachthrough(main_window) -> None:
    panel = main_window._right_panel

    assert hasattr(panel, "get_dataset")


def test_reset_layout_redocks_panels_without_forcing_widths(main_window, monkeypatch, qt_app) -> None:
    resize_calls: list[tuple[tuple[str, ...], tuple[int, ...], object]] = []

    def _record_resize_docks(self, docks, sizes, orientation):
        resize_calls.append(
            (
                tuple(str(dock.objectName()) for dock in docks),
                tuple(int(size) for size in sizes),
                orientation,
            )
        )

    monkeypatch.setattr(type(main_window), "resizeDocks", _record_resize_docks, raising=False)
    main_window.show()
    qt_app.processEvents()

    for attr, area in _NONDEFAULT_DOCK_AREAS.items():
        main_window.addDockWidget(area, getattr(main_window, attr))
    qt_app.processEvents()

    for attr, area in _NONDEFAULT_DOCK_AREAS.items():
        assert main_window.dockWidgetArea(getattr(main_window, attr)) == area

    main_window._mechanism_dock.setFloating(True)
    main_window._sliders_dock.setVisible(False)
    main_window._right_dock.setVisible(False)
    main_window._analysis_dock.setFloating(True)
    main_window._analysis_dock.setVisible(False)
    qt_app.processEvents()

    main_window._reset_layout()
    qt_app.processEvents()

    assert resize_calls == []
    for dock in _all_shell_docks(main_window):
        assert dock.isHidden() is False
        assert dock.isFloating() is False
    _assert_default_shell_contract(main_window)


def test_dock_back_preserves_default_shared_right_region_contract_after_floating_analysis_dock(main_window, qt_app) -> None:
    main_window.show()
    qt_app.processEvents()
    main_window._reset_layout()
    qt_app.processEvents()
    _assert_default_shell_contract(main_window)

    main_window._analysis_dock.setFloating(True)
    qt_app.processEvents()

    dock_back_button = main_window._analysis_dock.findChild(QtWidgets.QPushButton, "dockBackButton")
    assert dock_back_button is not None

    dock_back_button.click()
    qt_app.processEvents()

    _assert_default_shell_contract(main_window)


def test_dock_back_preserves_maximized_main_window_when_redock_drops_it(main_window, qt_app, monkeypatch) -> None:
    dock = main_window._analysis_dock
    main_window.show()
    main_window.showMaximized()
    qt_app.processEvents()

    dock.setFloating(True)
    qt_app.processEvents()

    original_set_floating = type(dock).setFloating

    def _drop_maximized_on_redock(current_dock, floating):
        original_set_floating(current_dock, floating)
        if current_dock is dock and not floating:
            main_window.showNormal()

    monkeypatch.setattr(type(dock), "setFloating", _drop_maximized_on_redock, raising=False)

    dock_back_button = dock.findChild(QtWidgets.QPushButton, "dockBackButton")
    assert dock_back_button is not None

    dock_back_button.click()
    qt_app.processEvents()

    assert dock.isFloating() is False
    assert main_window.isMaximized() is True


def test_restore_default_layout_button_recovers_from_disturbed_five_dock_arrangement(main_window, qt_app) -> None:
    main_window.show()
    qt_app.processEvents()
    main_window._reset_layout()
    qt_app.processEvents()

    main_window.addDockWidget(QtCore.Qt.RightDockWidgetArea, main_window._sliders_dock)
    main_window.addDockWidget(QtCore.Qt.RightDockWidgetArea, main_window._batch_dock)
    main_window._analysis_dock.hide()
    main_window._batch_dock.setFloating(True)
    qt_app.processEvents()

    reset_layout_button = main_window._batch_dock.findChild(QtWidgets.QPushButton, "resetLayoutButton")
    assert reset_layout_button is not None

    reset_layout_button.click()
    qt_app.processEvents()

    _assert_default_shell_contract(main_window)


def test_reset_layout_preserves_maximized_main_window_when_relayout_drops_it(main_window, qt_app, monkeypatch) -> None:
    main_window.show()
    main_window.showMaximized()
    qt_app.processEvents()

    original_info = main_window_module.logger.info

    def _drop_maximized_after_reset(message, *args, **kwargs):
        original_info(message, *args, **kwargs)
        if str(message) == "Layout reset to default":
            main_window.showNormal()

    monkeypatch.setattr(
        main_window_module.logger,
        "info",
        _drop_maximized_after_reset,
        raising=False,
    )

    main_window._reset_layout()
    qt_app.processEvents()

    assert main_window.isMaximized() is True
    assert main_window.centralWidget() is main_window._plot_tabs
    for dock in _all_shell_docks(main_window):
        assert dock.isFloating() is False
        assert dock.isHidden() is False


def test_shell_contract_exposes_all_five_docks_and_detached_slider_surface(main_window) -> None:
    main_window.show()
    QtWidgets.QApplication.processEvents()
    view_menu = _view_menu(main_window)
    panels_action = next(action for action in view_menu.actions() if action.menu() is not None and action.text() == "Panels")
    panels_menu = panels_action.menu()
    assert panels_menu is not None
    panel_actions = {action.text() for action in panels_menu.actions()}
    surface_actions = _analysis_surface_actions(main_window)
    help_menu = _help_menu(main_window)
    panel_layout_tips_action = next(
        action for action in help_menu.actions() if action.menu() is None and action.text() == "Panel Layout Tips..."
    )

    assert main_window.centralWidget() is main_window._plot_tabs
    assert panel_actions == {"Mechanism", "Interactive Sliders", "Initial Conditions", "Data", "Analysis"}
    assert set(surface_actions) == {"Statistics", "Parameters"}
    assert panel_layout_tips_action.objectName() == "panelLayoutTipsAction"
    for dock in _all_shell_docks(main_window):
        assert dock.allowedAreas() == _ALL_DOCK_AREAS
        assert dock.features() == main_window._right_dock.features()
    _assert_default_shell_contract(main_window)
    assert {
        attr: (getattr(main_window, attr).windowTitle(), getattr(main_window, attr).objectName())
        for attr in _EXPECTED_DOCK_METADATA
    } == _EXPECTED_DOCK_METADATA
    assert main_window._mechanism_dock.findChild(QtWidgets.QToolButton, "sliderVisibilityPickerButton") is None
    assert main_window._sliders_dock.findChild(QtWidgets.QToolButton, "sliderVisibilityPickerButton") is not None

    details_tabs = main_window._analysis_dock.findChild(QtWidgets.QTabWidget, "mainPlotDetailTabs")
    assert details_tabs is not None
    assert [details_tabs.tabText(i) for i in range(details_tabs.count())] == [
        "Statistics",
        "Parameters",
    ]


@pytest.mark.skip(reason="Ribbon hidden from users — feature intact, entry point commented out in main_window.py")
def test_minimal_view_ribbon_host_sits_beneath_menu_bar_and_reuses_shared_view_actions(main_window, qt_app) -> None:
    main_window.show()
    qt_app.processEvents()

    toolbar = _ribbon_toolbar(main_window)
    host = _ribbon_host(main_window)
    toolbar_actions = toolbar.actions()
    assert len(toolbar_actions) == 1
    assert toolbar.widgetForAction(toolbar_actions[0]) is host
    assert main_window.toolBarArea(toolbar) == QtCore.Qt.TopToolBarArea
    assert main_window.centralWidget() is main_window._plot_tabs

    menubar = main_window.menuBar()
    assert menubar.isVisible() is True
    assert toolbar.isVisible() is True
    assert toolbar.geometry().top() >= menubar.geometry().bottom()

    action_texts = {
        _normalized_action_text(button.defaultAction())
        for button in host.findChildren(QtWidgets.QToolButton)
        if button.defaultAction() is not None
    }
    assert action_texts >= {
        "Mechanism",
        "Interactive Sliders",
        "Initial Conditions",
        "Data",
        "Analysis",
        "Statistics",
        "Parameters",
        "Panel Layout Tips...",
        "Reset Layout",
        "Dark Mode",
    }
    assert "Run" not in action_texts
    assert "Load Data..." not in action_texts


@pytest.mark.skip(reason="Ribbon hidden from users — feature intact, entry point commented out in main_window.py")
def test_view_ribbon_exposes_first_class_page_group_contract(main_window, qt_app) -> None:
    main_window.show()
    qt_app.processEvents()

    host = _ribbon_host(main_window)
    assert hasattr(host, "tab_titles")
    assert callable(host.tab_titles)
    assert host.tab_titles() == ["View"]
    assert hasattr(host, "page_titles")
    assert callable(host.page_titles)
    assert host.page_titles() == ["View"]

    view_page = _ribbon_page(host, "View")
    assert hasattr(view_page, "group_titles")
    assert callable(view_page.group_titles)
    assert view_page.group_titles() == ["Panels", "Analysis", "Window"]

    assert hasattr(view_page, "group")
    group_getter = getattr(view_page, "group")
    assert callable(group_getter)

    panels_group = group_getter("Panels")
    analysis_group = group_getter("Analysis")
    window_group = group_getter("Window")
    assert panels_group is not None
    assert analysis_group is not None
    assert window_group is not None

    assert hasattr(window_group, "primary_action_texts")
    assert callable(window_group.primary_action_texts)
    assert window_group.primary_action_texts() == ["Reset Layout"]
    assert hasattr(window_group, "compact_action_texts")
    assert callable(window_group.compact_action_texts)
    assert set(window_group.compact_action_texts()) == {"Panel Layout Tips...", "Dark Mode"}

    assert hasattr(panels_group, "compact_action_texts")
    assert callable(panels_group.compact_action_texts)
    assert set(panels_group.compact_action_texts()) == {
        "Mechanism",
        "Interactive Sliders",
        "Initial Conditions",
        "Data",
        "Analysis",
    }
    assert hasattr(analysis_group, "compact_action_texts")
    assert callable(analysis_group.compact_action_texts)
    assert set(analysis_group.compact_action_texts()) == {
        "Statistics",
        "Parameters",
    }


@pytest.mark.skip(reason="Ribbon hidden from users — feature intact, entry point commented out in main_window.py")
def test_ribbon_host_collapses_and_expands_without_changing_central_workspace_owner(main_window, qt_app) -> None:
    main_window.show()
    qt_app.processEvents()

    host = _ribbon_host(main_window)
    assert hasattr(host, "is_collapsed")
    assert callable(host.is_collapsed)
    assert host.is_collapsed() is False

    body = host.findChild(QtWidgets.QWidget, "ribbonContentStack")
    toggle = host.findChild(QtWidgets.QToolButton, "ribbonCollapseToggleButton")
    assert body is not None
    assert toggle is not None
    assert body.isVisible() is True

    toggle.click()
    qt_app.processEvents()

    assert host.is_collapsed() is True
    assert body.isHidden() is True
    assert main_window.centralWidget() is main_window._plot_tabs
    assert getattr(main_window, "_shortcut_actions", {}).get("Run", {}).get("action") is not None

    toggle.click()
    qt_app.processEvents()

    assert host.is_collapsed() is False
    assert body.isVisible() is True


@pytest.mark.skip(reason="Ribbon hidden from users — feature intact, entry point commented out in main_window.py")
def test_ribbon_expand_preserves_maximized_main_window_when_expand_drops_it(main_window, qt_app, monkeypatch) -> None:
    main_window.show()
    main_window.showMaximized()
    qt_app.processEvents()

    host = _ribbon_host(main_window)
    toggle = host.findChild(QtWidgets.QToolButton, "ribbonCollapseToggleButton")
    assert toggle is not None

    toggle.click()
    qt_app.processEvents()
    assert host.is_collapsed() is True

    original_set_collapsed = type(host).set_collapsed

    def _drop_maximized_on_expand(current_host, collapsed):
        original_set_collapsed(current_host, collapsed)
        if current_host is host and not collapsed:
            main_window.showNormal()

    monkeypatch.setattr(type(host), "set_collapsed", _drop_maximized_on_expand, raising=False)

    toggle.click()
    qt_app.processEvents()

    assert host.is_collapsed() is False
    assert main_window.isMaximized() is True


def test_panel_layout_tips_action_explains_same_side_docking(main_window, monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _record_information(parent, title, text):
        captured["parent"] = parent
        captured["title"] = title
        captured["text"] = text
        return QtWidgets.QMessageBox.StandardButton.Ok

    monkeypatch.setattr(QtWidgets.QMessageBox, "information", _record_information)

    help_menu = _help_menu(main_window)
    action = next(action for action in help_menu.actions() if action.menu() is None and action.text() == "Panel Layout Tips...")
    action.trigger()

    assert captured["parent"] is main_window
    assert captured["title"] == "Panel Layout Tips"
    message = str(captured["text"])
    assert "View > Panels" in message
    assert "Interactive Sliders" in message
    assert "same side" in message
    assert "Dock Back" in message
    assert "Reset Layout" in message


@pytest.mark.parametrize("surface_name", ["Statistics", "Parameters"])
def test_analysis_surface_action_shows_dock_and_selects_requested_tab(main_window, qt_app, surface_name) -> None:
    details_tabs = main_window._analysis_dock.findChild(QtWidgets.QTabWidget, "mainPlotDetailTabs")
    assert details_tabs is not None
    action = _analysis_surface_actions(main_window)[surface_name]

    details_tabs.setCurrentIndex(0)
    main_window._analysis_dock.hide()
    qt_app.processEvents()

    action.trigger()
    qt_app.processEvents()

    assert main_window.centralWidget() is main_window._plot_tabs
    assert main_window._analysis_dock.isHidden() is False
    assert _current_tab_text(details_tabs) == surface_name


def test_analysis_surface_action_preserves_floating_analysis_dock(main_window, qt_app) -> None:
    details_tabs = main_window._analysis_dock.findChild(QtWidgets.QTabWidget, "mainPlotDetailTabs")
    assert details_tabs is not None
    action = _analysis_surface_actions(main_window)["Parameters"]

    main_window._analysis_dock.setFloating(True)
    main_window._analysis_dock.hide()
    qt_app.processEvents()

    action.trigger()
    qt_app.processEvents()

    assert main_window._analysis_dock.isFloating() is True
    assert main_window._analysis_dock.isHidden() is False
    assert _current_tab_text(details_tabs) == "Parameters"


def test_analysis_surface_actions_disable_and_guard_when_surfaces_are_unavailable(
    main_window,
    qt_app,
    monkeypatch,
) -> None:
    actions = _analysis_surface_actions(main_window)

    monkeypatch.setattr(main_window._plot_tabs, "available_analysis_surfaces", lambda: [])
    monkeypatch.setattr(main_window._plot_tabs, "focus_analysis_surface", lambda _name: False)

    main_window._refresh_analysis_surface_actions()

    assert all(action.isEnabled() is False for action in actions.values())

    main_window._analysis_dock.hide()
    qt_app.processEvents()

    assert main_window._show_analysis_surface("Statistics") is False
    assert main_window._analysis_dock.isHidden() is True


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
def test_floating_dock_exposes_explicit_dock_back_path(main_window, qt_app, dock_attr, target_area) -> None:
    dock = getattr(main_window, dock_attr)

    banner = dock.findChild(QtWidgets.QFrame, "floatingDockBanner")
    assert banner is not None
    assert banner.isVisible() is False

    main_window.addDockWidget(target_area, dock)
    qt_app.processEvents()

    assert main_window.dockWidgetArea(dock) == target_area

    dock.setFloating(True)
    qt_app.processEvents()

    banner = dock.findChild(QtWidgets.QFrame, "floatingDockBanner")
    dock_back_button = dock.findChild(QtWidgets.QPushButton, "dockBackButton")
    reset_layout_button = dock.findChild(QtWidgets.QPushButton, "resetLayoutButton")

    assert banner is not None
    assert banner.isVisible() is True
    assert dock_back_button is not None
    assert dock_back_button.isVisible() is True
    assert dock_back_button.text() == "Dock Back"
    assert reset_layout_button is not None
    assert reset_layout_button.isVisible() is True
    assert reset_layout_button.text() == "Restore Default Layout"
    assert "default workspace layout" in reset_layout_button.toolTip()

    dock_back_button.click()
    qt_app.processEvents()

    assert dock.isFloating() is False
    assert main_window.dockWidgetArea(dock) == target_area
    assert banner.isVisible() is False


def test_startup_shell_size_uses_windows_first_soft_desktop_budget() -> None:
    class _FakeScreen:
        def availableGeometry(self) -> QtCore.QRect:
            return QtCore.QRect(0, 0, 1600, 900)

    assert main_window_module._startup_size_for_screen(_FakeScreen()) == QtCore.QSize(1376, 820)
    assert main_window_module._startup_size_for_screen(None) == QtCore.QSize(1440, 900)


def test_restored_floating_dock_safety_uses_dock_minimums_and_screen_intersection(main_window, monkeypatch) -> None:
    class _FakeScreen:
        def availableGeometry(self) -> QtCore.QRect:
            return QtCore.QRect(0, 0, 1600, 900)

    class _FakeDock:
        def __init__(self, rect: QtCore.QRect, *, minimum: QtCore.QSize, floating: bool = True):
            self._rect = QtCore.QRect(rect)
            self._minimum = QtCore.QSize(minimum)
            self._floating = bool(floating)

        def isFloating(self) -> bool:
            return self._floating

        def frameGeometry(self) -> QtCore.QRect:
            return QtCore.QRect(self._rect)

        def geometry(self) -> QtCore.QRect:
            return QtCore.QRect(self._rect)

        def minimumSizeHint(self) -> QtCore.QSize:
            return QtCore.QSize(self._minimum)

        def minimumSize(self) -> QtCore.QSize:
            return QtCore.QSize(self._minimum)

    monkeypatch.setattr(QtGui.QGuiApplication, "screens", lambda: [_FakeScreen()])

    sliders_dock = main_window._sliders_dock
    minimum = sliders_dock.minimumSizeHint().expandedTo(sliders_dock.minimumSize())
    compact_height = max(minimum.height(), 60)
    compact_width = max(minimum.width(), 240)

    compact_on_screen = _FakeDock(QtCore.QRect(40, 40, compact_width, compact_height), minimum=minimum)
    off_screen = _FakeDock(QtCore.QRect(3000, 3000, compact_width, compact_height), minimum=minimum)
    too_small = _FakeDock(
        QtCore.QRect(40, 40, max(1, minimum.width() - 1), max(1, minimum.height() - 1)),
        minimum=minimum,
    )

    assert main_window._is_restored_floating_dock_unsafe(compact_on_screen) is False
    assert main_window._is_restored_floating_dock_unsafe(off_screen) is True
    assert main_window._is_restored_floating_dock_unsafe(too_small) is True


def test_batch_dock_title_is_initial_conditions_and_tips_in_help_menu(main_window) -> None:
    """Dock title must be 'Initial Conditions' and Panel Layout Tips must be in Help menu."""
    assert main_window._batch_dock.windowTitle() == "Initial Conditions"

    help_menu = _help_menu(main_window)
    help_action_texts = [a.text() for a in help_menu.actions() if not a.isSeparator()]
    assert "Panel Layout Tips..." in help_action_texts

    view_menu = _view_menu(main_window)
    view_action_texts = [a.text() for a in view_menu.actions() if a.menu() is None and not a.isSeparator()]
    assert "Panel Layout Tips..." not in view_action_texts


def test_mechanism_dock_has_no_collapsible_section(main_window) -> None:
    """Mechanism dock mounts editor directly, no redundant CollapsibleSection wrapper."""
    from kindred.gui.widgets.collapsible_section import CollapsibleSection

    panel = main_window._mechanism_panel
    assert not isinstance(getattr(panel, "section", None), CollapsibleSection)
    assert main_window._mechanism_dock.findChild(CollapsibleSection) is None


def test_all_shell_docks_have_custom_title_bar(main_window) -> None:
    from kindred.gui.widgets.dock_title_bar import DockTitleBar

    for dock in _all_shell_docks(main_window):
        tb = dock.titleBarWidget()
        assert isinstance(tb, DockTitleBar), f"{dock.objectName()} missing custom title bar"
        assert tb.findChild(QtWidgets.QLabel, "dockTitleLabel") is not None
        assert tb.findChild(QtWidgets.QToolButton, "dockMinimizeButton") is not None
        assert tb.findChild(QtWidgets.QToolButton, "dockCloseButton") is not None


def test_minimize_hides_dock_content_and_restore_shows_it(main_window, qt_app) -> None:
    dock = main_window._mechanism_dock
    tb = dock.titleBarWidget()
    minimize_btn = tb.findChild(QtWidgets.QToolButton, "dockMinimizeButton")

    main_window.show()
    qt_app.processEvents()

    assert tb.is_minimized() is False
    assert dock.widget().isVisible() is True

    minimize_btn.click()
    qt_app.processEvents()

    assert tb.is_minimized() is True
    assert dock.widget().isVisible() is False

    minimize_btn.click()
    qt_app.processEvents()

    assert tb.is_minimized() is False
    assert dock.widget().isVisible() is True


def test_dock_title_bar_close_button_closes_dock(main_window, qt_app) -> None:
    dock = main_window._sliders_dock
    tb = dock.titleBarWidget()
    close_btn = tb.findChild(QtWidgets.QToolButton, "dockCloseButton")

    main_window.show()
    qt_app.processEvents()

    assert dock.isVisible() is True

    close_btn.click()
    qt_app.processEvents()

    assert dock.isVisible() is False


def test_dock_title_bar_drag_initiates_float(main_window, qt_app, qtbot) -> None:
    dock = main_window._mechanism_dock
    tb = dock.titleBarWidget()

    main_window.show()
    qt_app.processEvents()

    start_pos = QtCore.QPoint(12, max(4, tb.height() // 2))
    qtbot.mousePress(tb, QtCore.Qt.MouseButton.LeftButton, pos=start_pos)
    qtbot.mouseMove(tb, start_pos + QtCore.QPoint(40, 0))
    qtbot.mouseRelease(tb, QtCore.Qt.MouseButton.LeftButton, pos=start_pos + QtCore.QPoint(40, 0))
    qt_app.processEvents()

    assert dock.isFloating() is True


def test_dock_title_bar_double_click_toggles_float(main_window, qt_app, qtbot) -> None:
    dock = main_window._sliders_dock
    tb = dock.titleBarWidget()

    main_window.show()
    qt_app.processEvents()

    click_pos = QtCore.QPoint(12, max(4, tb.height() // 2))
    qtbot.mouseDClick(tb, QtCore.Qt.MouseButton.LeftButton, pos=click_pos)
    qt_app.processEvents()
    assert dock.isFloating() is True

    qtbot.mouseDClick(tb, QtCore.Qt.MouseButton.LeftButton, pos=click_pos)
    qt_app.processEvents()
    assert dock.isFloating() is False


def test_minimized_dock_title_bar_drag_still_floats(main_window, qt_app, qtbot) -> None:
    dock = main_window._batch_dock
    tb = dock.titleBarWidget()
    minimize_btn = tb.findChild(QtWidgets.QToolButton, "dockMinimizeButton")
    assert minimize_btn is not None

    main_window.show()
    qt_app.processEvents()
    minimize_btn.click()
    qt_app.processEvents()

    start_pos = QtCore.QPoint(12, max(4, tb.height() // 2))
    qtbot.mousePress(tb, QtCore.Qt.MouseButton.LeftButton, pos=start_pos)
    qtbot.mouseMove(tb, start_pos + QtCore.QPoint(40, 0))
    qtbot.mouseRelease(tb, QtCore.Qt.MouseButton.LeftButton, pos=start_pos + QtCore.QPoint(40, 0))
    qt_app.processEvents()

    assert dock.isFloating() is True


def test_reshow_after_close_while_minimized_auto_restores(main_window, qt_app) -> None:
    dock = main_window._batch_dock
    tb = dock.titleBarWidget()
    minimize_btn = tb.findChild(QtWidgets.QToolButton, "dockMinimizeButton")

    main_window.show()
    qt_app.processEvents()

    minimize_btn.click()
    qt_app.processEvents()
    assert tb.is_minimized() is True

    dock.close()
    qt_app.processEvents()

    dock.show()
    qt_app.processEvents()

    assert tb.is_minimized() is False
    assert dock.widget().isVisible() is True


def test_reset_layout_restores_minimized_docks(main_window, qt_app) -> None:
    dock = main_window._right_dock
    tb = dock.titleBarWidget()
    minimize_btn = tb.findChild(QtWidgets.QToolButton, "dockMinimizeButton")

    main_window.show()
    qt_app.processEvents()

    minimize_btn.click()
    qt_app.processEvents()
    assert tb.is_minimized() is True

    main_window._reset_layout()
    qt_app.processEvents()

    assert tb.is_minimized() is False
    assert dock.widget().isVisible() is True


def test_factory_layout_has_batch_on_right_and_data_analysis_tabified(main_window, qt_app) -> None:
    main_window.show()
    qt_app.processEvents()
    main_window._reset_layout()
    qt_app.processEvents()

    assert main_window.dockWidgetArea(main_window._batch_dock) == QtCore.Qt.RightDockWidgetArea
    assert main_window.dockWidgetArea(main_window._mechanism_dock) == QtCore.Qt.LeftDockWidgetArea
    assert main_window.dockWidgetArea(main_window._sliders_dock) == QtCore.Qt.LeftDockWidgetArea

    tabified = main_window.tabifiedDockWidgets(main_window._right_dock)
    assert main_window._analysis_dock in tabified
