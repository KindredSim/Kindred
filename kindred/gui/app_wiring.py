from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Callable, TYPE_CHECKING

from PySide6 import QtCore

from kindred.core.runtime_defaults import MAX_PARALLEL_WORKERS_CEILING

if TYPE_CHECKING:
    from PySide6.QtGui import QUndoStack
    from PySide6.QtWidgets import QDockWidget, QWidget

    from kindred.config.profiles import ProfileManager
    from kindred.config.templates import TemplateManager
    from kindred.core.batch_initial_conditions import BatchInitialConditionsStore
    from kindred.gui.controllers.config_controller import ConfigController
    from kindred.gui.controllers.dataset_manager import DatasetManager
    from kindred.gui.main_window import MainWindow
    from kindred.gui.controllers.project_controller import ProjectController
    from kindred.gui.controllers.results_controller import ResultsController
    from kindred.gui.controllers.simulation_controller import SimulationController
    from kindred.gui.ports import SimulationUiPorts
    from kindred.gui.theme_manager import ThemeManager
    from kindred.gui.widgets.batch_initial_conditions_table import BatchInitialConditionsTableModel
    from kindred.gui.widgets.plot_tabs import PlotTabsWidget
    from kindred.gui.widgets.right_panel import RightPanelTabbed


@dataclass(frozen=True)
class SimulationPlumbing:
    ui_port: SimulationUiPorts
    controller: SimulationController


@dataclass(frozen=True)
class BatchInitialConditionsComponents:
    store: BatchInitialConditionsStore
    model: BatchInitialConditionsTableModel


@dataclass(frozen=True)
class SettingsControllers:
    config_controller: ConfigController
    project_controller: ProjectController
    undo_stack: QUndoStack


@dataclass(frozen=True)
class ProfileTemplateManagers:
    profile_manager: ProfileManager
    template_manager: TemplateManager


@dataclass(frozen=True)
class WindowShellComponents:
    plot_tabs: PlotTabsWidget
    theme_manager: ThemeManager
    results_controller: ResultsController


@dataclass(frozen=True)
class PanelDockComponents:
    dock: QDockWidget
    container: QWidget
    panel: QWidget


@dataclass(frozen=True)
class RightDockComponents:
    dock: QDockWidget
    container: QWidget
    panel: RightPanelTabbed
    dataset_manager: DatasetManager


@dataclass(frozen=True)
class BottomDockComponents:
    dock: QDockWidget
    container: QWidget


@dataclass(frozen=True)
class SolverContract:
    default_solver_name: str
    normalize_solver_name: Callable[[str], tuple[str, str | None]]


@dataclass(frozen=True)
class DockShellSpec:
    identity_key: str
    attr_name: str
    title: str
    object_name: str
    default_area: QtCore.Qt.DockWidgetArea


_DEFAULT_DOCK_AREA_PROPERTY = "kindred/defaultDockArea"
MECHANISM_DOCK_SPEC = DockShellSpec(
    identity_key="mechanism",
    attr_name="_mechanism_dock",
    title="Mechanism",
    object_name="mechanismDock",
    default_area=QtCore.Qt.LeftDockWidgetArea,
)
SLIDERS_DOCK_SPEC = DockShellSpec(
    identity_key="interactive_sliders",
    attr_name="_sliders_dock",
    title="Interactive Sliders",
    object_name="slidersDock",
    default_area=QtCore.Qt.LeftDockWidgetArea,
)
BATCH_DOCK_SPEC = DockShellSpec(
    identity_key="batch_initial_conditions",
    attr_name="_batch_dock",
    title="Initial Conditions",
    object_name="batchDock",
    default_area=QtCore.Qt.RightDockWidgetArea,
)
RIGHT_DOCK_SPEC = DockShellSpec(
    identity_key="data",
    attr_name="_right_dock",
    title="Data",
    object_name="rightDock",
    default_area=QtCore.Qt.RightDockWidgetArea,
)
ANALYSIS_DOCK_SPEC = DockShellSpec(
    identity_key="analysis",
    attr_name="_analysis_dock",
    title="Analysis",
    object_name="analysisDock",
    default_area=QtCore.Qt.RightDockWidgetArea,
)
SYMBOLIC_CALCULATOR_DOCK_SPEC = DockShellSpec(
    identity_key="symbolic_calculator",
    attr_name="_symbolic_calculator_dock",
    title="Symbolic Calculator",
    object_name="symbolicCalculatorDock",
    default_area=QtCore.Qt.RightDockWidgetArea,
)
_DOCK_SHELL_SPECS: tuple[DockShellSpec, ...] = (
    MECHANISM_DOCK_SPEC,
    SLIDERS_DOCK_SPEC,
    BATCH_DOCK_SPEC,
    RIGHT_DOCK_SPEC,
    ANALYSIS_DOCK_SPEC,
    SYMBOLIC_CALCULATOR_DOCK_SPEC,
)


def _all_supported_dock_areas():
    return QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea


def _dock_area_value(area: Any) -> int:
    return int(getattr(area, "value", area))


def configure_module_dock_defaults(dock: QDockWidget, *, default_area: Any) -> None:
    dock.setAllowedAreas(_all_supported_dock_areas())
    dock.setProperty(_DEFAULT_DOCK_AREA_PROPERTY, _dock_area_value(default_area))


def dock_default_area(dock: QDockWidget):
    stored_area = dock.property(_DEFAULT_DOCK_AREA_PROPERTY)
    if stored_area is None:
        return QtCore.Qt.NoDockWidgetArea
    return QtCore.Qt.DockWidgetArea(_dock_area_value(stored_area))


def dock_shell_specs() -> tuple[DockShellSpec, ...]:
    return _DOCK_SHELL_SPECS


def _dock_features() -> Any:
    from PySide6 import QtWidgets

    return (
        QtWidgets.QDockWidget.DockWidgetMovable
        | QtWidgets.QDockWidget.DockWidgetFloatable
        | QtWidgets.QDockWidget.DockWidgetClosable
    )


def build_shell_dock(main_window: MainWindow, spec: DockShellSpec):
    from PySide6 import QtWidgets

    from kindred.gui.widgets.dock_title_bar import DockTitleBar

    dock = QtWidgets.QDockWidget(spec.title, main_window)
    dock.setObjectName(spec.object_name)
    configure_module_dock_defaults(dock, default_area=spec.default_area)
    dock.setFeatures(_dock_features())
    dock.setFloating(False)

    title_bar = DockTitleBar(title=spec.title, dock=dock, parent=dock)
    dock.setTitleBarWidget(title_bar)
    dock.windowTitleChanged.connect(title_bar.set_title)

    return dock


def build_simulation_plumbing(main_window: MainWindow) -> SimulationPlumbing:
    from kindred.gui.ports import SimulationUiPorts
    from kindred.gui.controllers.simulation_controller import SimulationController

    ui_port = SimulationUiPorts(
        dialogs=main_window._simulation_dialogs,
        settings=main_window._settings_owner,
        run_ui=main_window._simulation_run_ui_owner,
        slider=main_window._preview_session,
        batch=main_window._simulation_batch_owner,
        mechanism=main_window._simulation_mechanism_owner,
        solver=main_window._simulation_solver_owner,
        runtime=main_window._variable_runtime,
        results=main_window.results_controller,
        provenance=main_window._simulation_provenance_owner,
        mechanism_helpers=main_window._mechanism_helpers,
    )
    controller = SimulationController(ui_port, parent=main_window)
    main_window._preview_session.set_slider_preview_lifecycle_port(controller)
    return SimulationPlumbing(ui_port=ui_port, controller=controller)


def build_batch_initial_conditions(main_window: MainWindow) -> BatchInitialConditionsComponents:
    from kindred.core.batch_initial_conditions import BatchInitialConditionsStore
    from kindred.gui.widgets.batch_initial_conditions_table import BatchInitialConditionsTableModel

    store = BatchInitialConditionsStore()
    model = BatchInitialConditionsTableModel(store, parent=main_window)
    return BatchInitialConditionsComponents(store=store, model=model)


def build_settings_and_controllers(main_window: MainWindow) -> SettingsControllers:
    from PySide6 import QtGui

    from kindred.gui.controllers.config_controller import ConfigController, ConfigControllerPort
    from kindred.gui.controllers.project_controller import ProjectController

    undo_stack = QtGui.QUndoStack(main_window)
    undo_stack.setUndoLimit(50)
    config_port = ConfigControllerPort(
        parent=main_window,
        settings=lambda: main_window._settings_owner.qsettings,
        is_maximized=main_window.isMaximized,
        restore_maximized_state=main_window.restore_persisted_maximized_state,
        restore_geometry=main_window.restoreGeometry,
        save_geometry=main_window.saveGeometry,
        restore_window_state=main_window.restoreState,
        save_window_state=main_window.saveState,
        schedule_restored_floating_dock_recovery=main_window.schedule_restored_floating_dock_recovery,
        splitter=lambda: main_window._plot_tabs.main_plot_workspace_splitter(),
        set_temperature=lambda value: main_window._temperature_spinbox.setValue(value),
        temperature=main_window.temperature_spinbox_value,
        set_simulation_time_text=lambda value: main_window._sim_time_spinbox.setText(str(value)),
        simulation_time_text=lambda: str(main_window._sim_time_spinbox.text()),
        set_num_points=lambda value: main_window._num_points_spinbox.setValue(int(value)),
        num_points=lambda: int(main_window._num_points_spinbox.value()),
        set_slider_preview_points=lambda value: main_window._mechanism_editor.set_slider_points_value(int(value)),
        slider_preview_points=lambda: int(main_window._mechanism_editor.slider_points_value()),
        set_slider_preview_solver=lambda value: main_window._mechanism_editor.set_slider_solver_value(str(value)),
        slider_preview_solver=lambda: str(main_window._mechanism_editor.slider_solver_value()),
        apply_solver_runtime_state=main_window.apply_solver_runtime_state,
        has_explicit_startup_solver_override=main_window.has_explicit_startup_solver_override,
        has_explicit_startup_rtol_override=main_window.has_explicit_startup_rtol_override,
        has_explicit_startup_atol_override=main_window.has_explicit_startup_atol_override,
        initial_solver_name=main_window._simulation_solver_owner.initial_solver_name,
        explicit_startup_solver_name=main_window.explicit_startup_solver_name,
        initial_rtol=main_window._simulation_solver_owner.initial_rtol,
        explicit_startup_rtol=main_window.explicit_startup_rtol,
        initial_atol=main_window._simulation_solver_owner.initial_atol,
        explicit_startup_atol=main_window.explicit_startup_atol,
        set_use_sparse_jacobian=lambda enabled: setattr(main_window, "_use_sparse_jacobian", bool(enabled)),
        use_sparse_jacobian=main_window._simulation_solver_owner.use_sparse_jacobian,
        set_wegscheider_cyclicity_enabled=lambda enabled: setattr(
            main_window,
            "_wegscheider_cyclicity_enabled",
            bool(enabled),
        ),
        wegscheider_cyclicity_enabled=main_window._simulation_solver_owner.wegscheider_cyclicity_enabled,
        max_parallel_batch_workers=lambda: int(main_window.simulation_controller.parallel_batch.max_parallel_workers),
        set_max_parallel_batch_workers=lambda value: setattr(
            main_window.simulation_controller.parallel_batch,
            "max_parallel_workers",
            min(
                int(MAX_PARALLEL_WORKERS_CEILING),
                max(1, int(value)),
            ),
        ),
        batch_runtime_lane_budget=lambda: int(main_window.simulation_controller.batch_runtime_lane_budget),
        set_batch_runtime_lane_budget=lambda value: setattr(
            main_window.simulation_controller,
            "batch_runtime_lane_budget",
            min(
                int(MAX_PARALLEL_WORKERS_CEILING),
                max(1, int(value)),
            ),
        ),
        limit_blas_threads_per_worker=lambda: bool(
            main_window.simulation_controller.parallel_batch.limit_blas_threads_per_worker
        ),
        set_limit_blas_threads_per_worker=lambda enabled: setattr(
            main_window.simulation_controller.parallel_batch,
            "limit_blas_threads_per_worker",
            bool(enabled),
        ),
        result_cache_cap=lambda: int(main_window.simulation_controller.batch_cache.result_cache_max_entries()),
        preview_cache_cap=lambda: int(main_window.simulation_controller.batch_cache.preview_cache_max_entries()),
        set_cache_caps=lambda *, result_cap, preview_cap, persist: main_window.set_simulation_cache_caps(
            result_cap=int(result_cap),
            preview_cap=int(preview_cap),
            persist=bool(persist),
        ),
        set_ribbon_collapsed=main_window.set_ribbon_collapsed,
        ribbon_collapsed=main_window.ribbon_collapsed,
        set_dark_mode=lambda enabled: setattr(main_window, "_dark_mode", bool(enabled)),
        dark_mode=lambda: bool(main_window._dark_mode),
        dark_mode_action=lambda: getattr(main_window, "_dark_mode_action", None),
        apply_theme=lambda enabled: main_window._theme_manager.apply(bool(enabled)),
        profile_manager=lambda: main_window._profile_manager,
        apply_profile=main_window._apply_profile,
        set_profile_indicator_text=lambda text: main_window._profile_indicator.setText(str(text)),
        set_status_text=main_window.set_status_text,
        update_profiles_menu=main_window._update_profiles_menu,
        profiles_menu_available=lambda: main_window._profile_ports.profiles_menu_getter() is not None,
        load_custom_shortcuts=main_window._load_custom_shortcuts,
        recent_menu=lambda: getattr(main_window, "_recent_menu", None),
        set_recent_menu=lambda menu: setattr(main_window, "_recent_menu", menu),
        menu_bar=main_window.menuBar,
        load_recent_project=lambda filepath: main_window._load_recent_project(str(filepath)),
    )
    return SettingsControllers(
        config_controller=ConfigController(config_port),
        project_controller=ProjectController(main_window),
        undo_stack=undo_stack,
    )


def build_profile_and_template_managers() -> ProfileTemplateManagers:
    from kindred.config.profiles import ProfileManager
    from kindred.config.templates import TemplateManager

    profile_manager = ProfileManager()

    template_manager = TemplateManager()
    template_manager.load_templates()
    return ProfileTemplateManagers(profile_manager=profile_manager, template_manager=template_manager)


def build_window_shell(main_window: MainWindow) -> WindowShellComponents:
    from kindred.gui.controllers.results_controller import ResultsController, ResultsControllerPort, ResultsDisplayPlotPort
    from kindred.gui.theme_manager import ThemeManager
    from kindred.gui.widgets.plot_tabs import PlotTabsWidget

    plot_tabs = PlotTabsWidget(main_plot_embed_analysis_tabs=False)
    plot_tabs.setObjectName("plotTabs")
    theme_manager = ThemeManager(plot_tabs)

    def main_plot() -> ResultsDisplayPlotPort:
        return plot_tabs._main_plot

    def set_main_plot_scalar_values(scalars) -> None:
        main_plot().set_scalar_values(dict(scalars or {}))

    def update_main_plot_statistics(*, stats_results_map, prefer, t, series) -> None:
        main_plot().set_statistics_results(dict(stats_results_map or {}), prefer=str(prefer or ""))

    def main_plot_stats_table() -> object:
        return main_plot().stats_table()

    def publish_main_plot_results_table(table) -> None:
        main_window._results_table = table

    def set_main_plot_data(t, series, **kwargs) -> None:
        main_plot().set_data(t, series, **kwargs)

    def show_simulation_tab() -> None:
        plot_tabs._tabs.setCurrentIndex(0)

    def refresh_simulation_plot_views() -> None:
        main_plot().update()
        plot_tabs.update()
        main_window.update()

    def schedule_main_plot_refresh(delays_ms) -> None:
        plot = main_plot()

        def _safe_plot_update(plot_widget=plot) -> None:
            with suppress(RuntimeError):
                plot_widget.update()

        for delay_ms in delays_ms:
            QtCore.QTimer.singleShot(int(delay_ms), _safe_plot_update)

    results_port = ResultsControllerPort(
        parent=main_window,
        main_plot=main_plot,
        batch_name_for_id=main_window.batch_set_name_for_id,
        batch_id_for_name=main_window.batch_set_id_for_name,
        batch_set_ids_for_scope=main_window.batch_set_ids_for_scope,
        requested_show_batch_set_ids=main_window.requested_show_batch_set_ids,
        explicit_slider_target_set_ids=main_window._slider_edit_target_set_ids,
        effective_slider_target_set_ids=main_window._effective_slider_edit_target_set_ids,
        focused_batch_set_id=main_window.focused_batch_set_id,
        current_batch_row=main_window.batch_current_row,
        batch_set_id_for_row=main_window.batch_set_id_for_row,
        batch_row_for_set_id=main_window._batch_row_for_set_id,
        active_batch_cache_key=main_window.active_batch_cache_key,
        active_result_cache_read_snapshot=(
            main_window._simulation_batch_owner.active_result_cache_read_snapshot
        ),
        clear_active_preview_cache_identity_state=main_window._simulation_batch_owner.clear_active_preview_cache_identity_state,
        set_last_simulation_provenance=main_window._simulation_provenance_owner.set_last_simulation_provenance,
        set_last_simulation_ctc=main_window._simulation_provenance_owner.set_last_simulation_ctc,
        publish_simulation_completion_provenance=(
            main_window._simulation_provenance_owner.publish_simulation_completion_provenance
        ),
        update_display_transaction_provenance=(
            main_window._simulation_provenance_owner.update_display_transaction_provenance
        ),
        set_main_plot_scalar_values=set_main_plot_scalar_values,
        update_main_plot_statistics=update_main_plot_statistics,
        main_plot_stats_table=main_plot_stats_table,
        publish_main_plot_results_table=publish_main_plot_results_table,
        set_main_plot_data=set_main_plot_data,
        show_simulation_tab=show_simulation_tab,
        refresh_simulation_plot_views=refresh_simulation_plot_views,
        schedule_main_plot_refresh=schedule_main_plot_refresh,
        set_status_text=main_window.set_status_text,
        update_batch_row_controls_state=main_window._simulation_batch_owner.update_batch_row_controls_state,
        focused_show_request_is_dirty=lambda requested_show_set_ids, prefer_set: (
            main_window._simulation_batch_owner.focused_show_request_is_dirty(
                requested_show_set_ids=requested_show_set_ids,
                prefer_set=prefer_set,
            )
        ),
        focused_batch_set_is_dirty=main_window._simulation_batch_owner.focused_batch_set_is_dirty,
        show_request_uses_fresh_explicit_cache_after_post_run_sync=(
            lambda requested_show_set_ids: (
                main_window._simulation_batch_owner.show_request_uses_fresh_explicit_cache_after_post_run_sync(
                    requested_show_set_ids=requested_show_set_ids
                )
            )
        ),
        workspace_display_request_resolution=main_window._simulation_batch_owner.workspace_display_request_resolution,
        current_workspace_preview_identity_payload=lambda set_id: (
            main_window._simulation_batch_owner.current_workspace_preview_identity_payload(set_id=str(set_id))
        ),
    )
    results_controller = ResultsController(results_port)
    return WindowShellComponents(
        plot_tabs=plot_tabs,
        theme_manager=theme_manager,
        results_controller=results_controller,
    )


def build_bottom_analysis_dock(
    main_window: MainWindow,
    *,
    analysis_widget: QWidget,
) -> BottomDockComponents:
    from kindred.gui.widgets.floating_dock_container import FloatingDockContainer

    dock = build_shell_dock(main_window, ANALYSIS_DOCK_SPEC)

    container = FloatingDockContainer(
        content=analysis_widget,
        dock=dock,
        on_dock_back=lambda: main_window.redock_shell_dock(dock),
        on_reset_layout=main_window._reset_layout,
        parent=main_window,
    )
    return BottomDockComponents(dock=dock, container=container)


def build_symbolic_calculator_dock(
    main_window: MainWindow,
    *,
    panel: QWidget,
) -> PanelDockComponents:
    return build_module_dock(
        main_window,
        spec=SYMBOLIC_CALCULATOR_DOCK_SPEC,
        panel=panel,
    )


def build_module_dock(
    main_window: MainWindow,
    *,
    spec: DockShellSpec,
    panel: QWidget,
) -> PanelDockComponents:
    from kindred.gui.widgets.floating_dock_container import FloatingDockContainer

    dock = build_shell_dock(main_window, spec)
    container = FloatingDockContainer(
        content=panel,
        dock=dock,
        on_dock_back=lambda: main_window.redock_shell_dock(dock),
        on_reset_layout=main_window._reset_layout,
        parent=main_window,
    )
    return PanelDockComponents(dock=dock, container=container, panel=panel)


def build_mechanism_dock(main_window: MainWindow) -> PanelDockComponents:
    from kindred.gui.widgets.editor_panel import EditorPanel

    return build_module_dock(
        main_window,
        spec=MECHANISM_DOCK_SPEC,
        panel=EditorPanel(parent=main_window),
    )


def build_sliders_dock(main_window: MainWindow, *, panel: QWidget) -> PanelDockComponents:
    return build_module_dock(
        main_window,
        spec=SLIDERS_DOCK_SPEC,
        panel=panel,
    )


def build_batch_dock_panel(
    main_window: MainWindow,
    *,
    batch_model: BatchInitialConditionsTableModel,
    initial_solver: str,
    on_add_batch_set: Callable[[], None],
    on_move_selected_batch_sets_up: Callable[[], None],
    on_move_selected_batch_sets_down: Callable[[], None],
    on_delete_selected_batch_sets: Callable[[], None],
    on_run_selected: Callable[[], None],
    on_stop: Callable[[], None],
    on_solver_method_changed: Callable[[str], None],
    on_solver_summary_refresh: Callable[[], None],
) -> PanelDockComponents:
    from kindred.gui.widgets.simulation_panel import SimulationPanel

    panel = SimulationPanel(
        batch_model=batch_model,
        message_parent=main_window,
        initial_solver=str(initial_solver),
        on_add_batch_set=on_add_batch_set,
        on_move_selected_batch_sets_up=on_move_selected_batch_sets_up,
        on_move_selected_batch_sets_down=on_move_selected_batch_sets_down,
        on_delete_selected_batch_sets=on_delete_selected_batch_sets,
        on_run_selected=on_run_selected,
        on_stop=on_stop,
        on_solver_method_changed=on_solver_method_changed,
        on_solver_summary_refresh=on_solver_summary_refresh,
        parent=main_window,
    )
    return build_module_dock(
        main_window,
        spec=BATCH_DOCK_SPEC,
        panel=panel,
    )


def build_right_dock_and_dataset_manager(
    main_window: MainWindow,
    *,
    plot_tabs: PlotTabsWidget,
    mechanism_getter: Callable[[], str],
    simulation_runner: Callable[..., Any],
    solver_settings_getter: Callable[[], dict[str, Any]],
) -> RightDockComponents:
    from kindred.gui.controllers.dataset_manager import DatasetManager
    from kindred.gui.widgets.floating_dock_container import FloatingDockContainer
    from kindred.gui.widgets.right_panel import RightPanelTabbed

    dock = build_shell_dock(main_window, RIGHT_DOCK_SPEC)

    panel = RightPanelTabbed()
    container = FloatingDockContainer(
        content=panel,
        dock=dock,
        on_dock_back=lambda: main_window.redock_shell_dock(dock),
        on_reset_layout=main_window._reset_layout,
        parent=main_window,
    )
    dataset_manager = DatasetManager(
        plot_tabs=plot_tabs,
        dataset_resolver=panel.get_dataset,
        mechanism_getter=mechanism_getter,
        simulation_runner=simulation_runner,
        solver_settings_getter=solver_settings_getter,
    )
    return RightDockComponents(dock=dock, container=container, panel=panel, dataset_manager=dataset_manager)


def load_solver_contract() -> SolverContract:
    from kindred.core.simulator.solvers import DEFAULT_SOLVER_NAME, normalize_solver_name

    return SolverContract(
        default_solver_name=str(DEFAULT_SOLVER_NAME),
        normalize_solver_name=normalize_solver_name,
    )
