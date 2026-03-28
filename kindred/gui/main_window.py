# kindred/gui/main_window.py
# REDESIGNED GUI - Expert/Power-user focused with fitting capabilities

from __future__ import annotations

from contextlib import suppress
import json
import hashlib
import logging
import os
import platform
import re
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Callable, Sequence, TYPE_CHECKING, Mapping
from collections import OrderedDict
from collections.abc import MutableMapping
import math

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, QSettings

from kindred import __version__ as KINDRED_VERSION
from kindred.core.batch_simulation_cache import BatchSimulationCache
from kindred.core.simulator.dsl_text_update import (
    analyze_step_parameter_update,
    build_current_text_step_analysis_context,
    format_authoritative_parameter_value,
    step_rewrite_block_reason,
)
from kindred.core.validation import try_parse_finite_float
from kindred.gui.controllers.cache_contracts import BatchCacheEntryReadResult, read_batch_cache_entry

if TYPE_CHECKING:
    from kindred.core.mechanism import Mechanism
    from kindred.core.simulation_preparation import BoundMechanism
    from kindred.gui.controllers.results_controller import ResolvedBatchSelectionEntry
    from kindred.gui.controllers.simulation_controller import SimulationController
    from kindred.gui.ports import SimulationUiPorts
    from kindred.gui.widgets.batch_initial_conditions_table import BatchInitialConditionsTableView

from kindred.gui.app_wiring import (
    build_bottom_analysis_dock,
    build_batch_initial_conditions,
    build_batch_dock_panel,
    build_mechanism_dock,
    build_profile_and_template_managers,
    build_right_dock_and_dataset_manager,
    build_settings_and_controllers,
    build_sliders_dock,
    build_simulation_plumbing,
    build_window_shell,
    dock_default_area,
    dock_shell_specs,
    load_solver_contract,
)
from kindred.gui.diagnostics import record_best_effort_failure as record_gui_best_effort_failure
from kindred.gui.main_window_mechanism_helpers import MainWindowMechanismHelpers
from kindred.gui.main_window_preview_session import MainWindowPreviewSession
from kindred.gui.main_window_variable_runtime import MainWindowVariableRuntime
from kindred.gui.mixins.ports import FittingMixinPorts, ProfileMixinPorts
from kindred.gui.mixins.fitting_mixin import FittingMixin
from kindred.gui.mixins.profile_mixin import ProfileMixin
from kindred.gui.widgets.ribbon import CollapsibleRibbonHost, RibbonGroup, RibbonPage

logger = logging.getLogger(__name__)

__all__ = ["MainWindow"]

# Online documentation is not yet published.
DOCUMENTATION_URL: Optional[str] = None
PROJECT_SCHEMA_VERSION = 3
_SOLVER_STATE_UNSET = object()
_STARTUP_WIDTH_RATIO = 0.86
_STARTUP_HEIGHT_RATIO = 0.88
_MIN_STARTUP_WIDTH = 1280
_MIN_STARTUP_HEIGHT = 820
_FALLBACK_STARTUP_SIZE = QtCore.QSize(1440, 900)
_ANALYSIS_SURFACE_NAMES: tuple[str, ...] = ("Statistics", "Parameters", "Overlays")
_ABOUT_DIALOG_MIN_WIDTH = 420
_ABOUT_DIALOG_IMAGE_MAX_SIZE = 320


def _startup_size_for_screen(screen: Optional[QtGui.QScreen]) -> QtCore.QSize:
    if screen is None:
        return QtCore.QSize(_FALLBACK_STARTUP_SIZE)

    available = screen.availableGeometry()
    min_width = min(_MIN_STARTUP_WIDTH, available.width())
    min_height = min(_MIN_STARTUP_HEIGHT, available.height())
    return QtCore.QSize(
        max(int(available.width() * _STARTUP_WIDTH_RATIO), min_width),
        max(int(available.height() * _STARTUP_HEIGHT_RATIO), min_height),
    )


# ============================================================================
# MAIN WINDOW
# ============================================================================

class MainWindow(
    QtWidgets.QMainWindow,
    FittingMixin,
    ProfileMixin,
):
    """
    Redesigned Kindred main window.

    Layout:
    - LEFT: Mechanism, Interactive Sliders, Batch Initial Conditions
    - CENTER: Tabbed plot area (multiple datasets)
    - RIGHT: Data Manager + Analysis
    - STATUS BAR: Detailed progress info

    Mixins:
    - FittingMixin: Parameter fitting (standard, global)
    """

    _CACHED_PRESET_IDS: Optional[List[str]] = None

    @property
    def simulation_controller(self) -> SimulationController:
        return self._sim_controller

    def __init__(
        self,
        profile: Optional[str] = None,
        solver: Optional[str] = None,
        rtol: Optional[float] = None,
        atol: Optional[float] = None,
    ):
        super().__init__()
        self._init_cli_args(profile=profile, solver=solver, rtol=rtol, atol=atol)
        self._init_simulation_plumbing_and_state()
        self._init_batch_initial_conditions()
        self._init_settings_and_controllers()
        self._init_profile_and_template_managers()
        self._init_window_shell()
        self._init_mechanism_dock_and_panel()
        self._init_sliders_dock()
        self._init_batch_dock_and_panel()
        self._init_right_dock_and_datasets()
        self._init_bottom_analysis_dock()
        self._init_status_bar_widgets()
        self._finish_window_composition()
        self._bootstrap_window_state()

        logger.info("Kindred MainWindow initialized (new design)")

    def _init_cli_args(
        self,
        *,
        profile: Optional[str],
        solver: Optional[str],
        rtol: Optional[float],
        atol: Optional[float],
    ) -> None:
        # Store launch overrides for later use.
        self._initial_profile = profile
        self._initial_solver = solver
        self._initial_rtol = rtol
        self._initial_atol = atol
        self._explicit_startup_solver_value = solver
        self._explicit_startup_rtol_value = rtol
        self._explicit_startup_atol_value = atol
        self._explicit_startup_solver_override = solver is not None
        self._explicit_startup_rtol_override = rtol is not None
        self._explicit_startup_atol_override = atol is not None

    def _init_simulation_plumbing_and_state(self) -> None:
        self._preview_session = MainWindowPreviewSession(self)
        self._variable_runtime = MainWindowVariableRuntime(self)
        self._mechanism_helpers = MainWindowMechanismHelpers(self)

        # Simulation execution, batch orchestration, caching, and worker lifecycle.
        plumbing = build_simulation_plumbing(self)
        self._sim_ui_port: SimulationUiPorts = plumbing.ui_port
        self._sim_controller = plumbing.controller

        # Fitting state.
        self._last_fit_result = None

        # Provenance tracking for the last simulation.
        self._last_simulation_provenance = {}
        self._last_simulation_ctc = {}
        self._last_fit_metadata: Optional[Dict[str, Any]] = None

        # Best-effort failure recording (GUI hardening).
        self._best_effort_failures: set[str] = set()
        self._best_effort_failure_counts: Dict[str, int] = {}

        # Mechanism + simulation state.
        self._use_sparse_jacobian = False
        self._wegscheider_cyclicity_enabled = False
        self._last_batch_results: List[Dict[str, Any]] = []
        self._advanced_dsl_enabled = True  # Physics-aware DSL is always active.

        # Registry of actions that support customizable shortcuts.
        self._shortcut_actions: Dict[str, Dict[str, Any]] = {}

    def _init_batch_initial_conditions(self) -> None:
        # Batch initial conditions (source-of-truth for initials after first migration).
        batch_components = build_batch_initial_conditions(self)
        self._batch_store = batch_components.store
        self._batch_model = batch_components.model
        self._batch_table: Optional[BatchInitialConditionsTableView] = None
        self._focused_batch_set_id = ""
        self._connected_batch_semantics_model = None
        self._connected_batch_selection_model = None
        # (Batch run/caching/executor state lives in self._sim_controller.)

    def _init_settings_and_controllers(self) -> None:
        # Theme state.
        self._dark_mode = False  # Will be loaded from settings.

        # Settings manager (QSettings instance).
        self._settings = QSettings("Kindred", "KindredGUI")

        # Configuration persistence + settings-driven UI application.
        controllers = build_settings_and_controllers(self)
        self.config_controller = controllers.config_controller
        self.project_controller = controllers.project_controller

        # Menu/UI objects that controllers may reference.
        self._recent_menu = None
        self._debug_sliders_action = None
        self._mechanism_edit_locked = True
        self._mechanism_edit_unlock_warning_shown = False
        self._mechanism_edit_lock_action = None

        # Undo/Redo stack for high-level operations.
        self._undo_stack = controllers.undo_stack

    def settings_set_value(self, key: str, value: object) -> None:
        self._settings.setValue(str(key), value)

    def settings_remove(self, key: str) -> None:
        self._settings.remove(str(key))

    def settings_sync(self) -> None:
        self._settings.sync()

    def _init_profile_and_template_managers(self) -> None:
        managers = build_profile_and_template_managers()
        self._profile_manager = managers.profile_manager
        self._template_manager = managers.template_manager

    def _init_window_shell(self) -> None:
        self.setWindowTitle("Kindred")
        self.resize(_startup_size_for_screen(self.screen()))

        # Enable dockable widgets.
        self.setDockNestingEnabled(True)

        # Create center widget (plots) as central widget.
        shell = build_window_shell(self)
        self._plot_tabs = shell.plot_tabs
        self.setCentralWidget(self._plot_tabs)
        self._theme_manager = shell.theme_manager
        self.results_controller = shell.results_controller
        plot = getattr(self, "_plot_tabs", None)
        main_plot = getattr(plot, "_main_plot", None)
        set_copy_all_provider = getattr(main_plot, "set_copy_all_export_plan_provider", None)
        if callable(set_copy_all_provider):
            set_copy_all_provider(self._build_main_plot_copy_all_export_plan)

    def _init_mechanism_dock_and_panel(self) -> None:
        mechanism_dock_components = build_mechanism_dock(self)
        self._mechanism_dock = mechanism_dock_components.dock
        self._mechanism_panel = mechanism_dock_components.panel

        self._mechanism_section = self._mechanism_panel.section
        self._mechanism_editor = self._mechanism_panel.editor
        self._refresh_mechanism_edit_lock_ui()
        self._sliders_panel = self._mechanism_editor.detach_slider_pane_for_dock()
        self._species_panel_available = True
        self._slider_override_buttons_available = True

        self._mechanism_dock.setWidget(mechanism_dock_components.container)
        self.addDockWidget(self._default_dock_area(self._mechanism_dock), self._mechanism_dock)

    def _init_sliders_dock(self) -> None:
        sliders_panel = getattr(self, "_sliders_panel", None)
        if sliders_panel is None:
            sliders_panel = QtWidgets.QLabel("Interactive sliders unavailable", self)
            sliders_panel.setObjectName("interactiveSlidersUnavailableLabel")
            self._sliders_panel = sliders_panel
        sliders_dock_components = build_sliders_dock(self, panel=sliders_panel)
        self._sliders_dock = sliders_dock_components.dock
        self._sliders_dock.setWidget(sliders_dock_components.container)
        self.addDockWidget(self._default_dock_area(self._sliders_dock), self._sliders_dock)

    def _init_batch_dock_and_panel(self) -> None:
        solver_contract = load_solver_contract()
        batch_dock_components = build_batch_dock_panel(
            self,
            batch_model=self._batch_model,
            initial_solver=str(self._initial_solver or solver_contract.default_solver_name),
            on_add_batch_set=self._add_batch_set,
            on_move_selected_batch_sets_up=self._move_selected_batch_sets_up,
            on_move_selected_batch_sets_down=self._move_selected_batch_sets_down,
            on_delete_selected_batch_sets=self._delete_selected_batch_sets,
            on_run_selected=self._sim_controller.run_simulation,
            on_stop=self._sim_controller.stop_simulation,
            on_solver_method_changed=self._on_solver_method_changed,
            on_solver_summary_refresh=self._update_solver_summary_label,
        )
        self._batch_dock = batch_dock_components.dock
        self._batch_panel = batch_dock_components.panel

        self._wire_mechanism_editor_signals()
        self._bind_simulation_panel_widgets()

        self._batch_dock.setWidget(batch_dock_components.container)
        self.addDockWidget(self._default_dock_area(self._batch_dock), self._batch_dock)

    def _wire_mechanism_editor_signals(self) -> None:
        sliders = self._mechanism_editor._variable_sliders
        sliders.variableChanged.connect(self._on_variable_changed)
        sliders.sliderDragStarted.connect(self._on_slider_drag_started)
        sliders.sliderDragFinished.connect(self._on_slider_drag_finished)

        try:
            self._mechanism_editor._commit_slider_overrides_btn.clicked.connect(self._on_commit_slider_overrides_clicked)
            self._mechanism_editor._reset_slider_overrides_btn.clicked.connect(self._on_reset_slider_overrides_clicked)
        except Exception:
            logger.exception("Failed to connect slider override mode buttons")
            self._slider_override_buttons_available = False
            self._disable_slider_override_mode_buttons()

        self._mechanism_editor.speciesResetRequested.connect(self._on_species_reset_requested)
        self._refresh_slider_transaction_button_state()

    def _set_slider_override_mode_buttons_enabled(self, enabled: bool) -> None:
        for attr in ("_commit_slider_overrides_btn", "_reset_slider_overrides_btn"):
            btn = getattr(self._mechanism_editor, attr, None)
            if btn is None:
                continue
            try:
                btn.setEnabled(bool(enabled))
            except RuntimeError as exc:
                self._record_best_effort_failure(
                    f"main_window.slider_override_buttons.set_enabled.{attr}",
                    message="Failed to update slider override mode button state",
                    exc=exc,
                )

    def _disable_slider_override_mode_buttons(self) -> None:
        self._set_slider_override_mode_buttons_enabled(False)

    def _refresh_slider_transaction_button_state(self) -> None:
        if not bool(getattr(self, "_slider_override_buttons_available", False)):
            self._disable_slider_override_mode_buttons()
            return
        preview = getattr(self, "_preview_session", None)
        dirty = bool(preview is not None and hasattr(preview, "has_dirty_transaction") and preview.has_dirty_transaction())
        self._set_slider_override_mode_buttons_enabled(bool(dirty))

    def mechanism_editing_locked(self) -> bool:
        return bool(getattr(self, "_mechanism_edit_locked", True))

    @staticmethod
    def _state_network_dialog_info_text(*, locked: bool) -> str:
        if bool(locked):
            return (
                "State network is read-only while Reactions editing is locked. "
                "Use Unlock Reactions Editing to make deliberate changes."
            )
        return "Edit the state network with full validation. Changes apply directly to the current mechanism."

    def _refresh_mechanism_edit_lock_ui(self) -> None:
        locked = self.mechanism_editing_locked()
        button_text = "Unlock Reactions Editing" if locked else "Lock Reactions Editing"
        tooltip = (
            "Temporarily enable deliberate edits in the Reactions editor."
            if locked
            else "Return the Reactions editor to read-only mode."
        )
        status_text = (
            "Reactions editor is read-only. Use Unlock Reactions Editing before making deliberate changes."
            if locked
            else "Reactions editor is unlocked. Changes here modify the canonical mechanism text saved with the project."
        )
        editor = getattr(self, "_mechanism_editor", None)
        if editor is not None:
            set_read_only = getattr(editor, "set_reactions_read_only", None)
            if callable(set_read_only):
                set_read_only(locked)
            state_editor = getattr(editor, "_state_network_editor", None)
            set_state_read_only = getattr(state_editor, "set_read_only", None)
            if callable(set_state_read_only):
                set_state_read_only(locked)
            set_status_text = getattr(editor, "set_reactions_edit_status_text", None)
            if callable(set_status_text):
                set_status_text(status_text)
            action = getattr(self, "_mechanism_edit_lock_action", None)
            bind_action = getattr(editor, "set_reactions_edit_action", None)
            if action is not None and callable(bind_action):
                bind_action(action)
        action = getattr(self, "_mechanism_edit_lock_action", None)
        if action is not None:
            previous = action.blockSignals(True)
            try:
                action.setText(button_text)
                action.setToolTip(tooltip)
                action.setChecked(not locked)
            finally:
                action.blockSignals(previous)
        dialog = getattr(self, "_state_network_dialog", None)
        if dialog is not None:
            try:
                info_label = dialog.findChild(QtWidgets.QLabel, "stateNetworkDialogInfoLabel")
            except RuntimeError as exc:
                self._record_best_effort_failure(
                    "main_window.state_network_dialog.info_label.find",
                    message="Failed to refresh state network dialog lock banner",
                    exc=exc,
                )
            else:
                if info_label is not None:
                    info_label.setText(self._state_network_dialog_info_text(locked=locked))

    def _set_mechanism_edit_locked(self, locked: bool) -> None:
        self._mechanism_edit_locked = bool(locked)
        self._refresh_mechanism_edit_lock_ui()

    def _reactions_text_widget(self) -> Optional[QtWidgets.QPlainTextEdit]:
        editor = getattr(self, "_mechanism_editor", None)
        widget = getattr(editor, "_reactions_text", None)
        return widget if isinstance(widget, QtWidgets.QPlainTextEdit) else None

    def _focused_widget_targets_reactions_text(self, widget: Optional[QtWidgets.QWidget]) -> bool:
        reactions_widget = self._reactions_text_widget()
        if reactions_widget is None or widget is None:
            return False
        return bool(widget is reactions_widget or reactions_widget.isAncestorOf(widget))

    def _undo_command_targets_locked_mechanism_change(self, command: object) -> bool:
        if command is None:
            return False
        reactions_widget = self._reactions_text_widget()
        state_editor = getattr(getattr(self, "_mechanism_editor", None), "_state_network_editor", None)

        if reactions_widget is not None and getattr(command, "_text_widget", None) is reactions_widget:
            return True
        if getattr(command, "_reactions_widget", None) is reactions_widget:
            old_text = str(getattr(command, "_old_reactions_text", "") or "")
            new_text = str(getattr(command, "_new_reactions_text", "") or "")
            old_state = str(getattr(command, "_old_state_network_dsl", "") or "")
            new_state = str(getattr(command, "_new_state_network_dsl", "") or "")
            return old_text != new_text or old_state != new_state
        if state_editor is not None and getattr(command, "_state_network_editor", None) is state_editor:
            old_state = str(getattr(command, "_old_state_network_dsl", "") or "")
            new_state = str(getattr(command, "_new_state_network_dsl", "") or "")
            return old_state != new_state
        child_count = int(command.childCount()) if hasattr(command, "childCount") else 0
        for child_index in range(child_count):
            child_command = command.child(child_index)
            if self._undo_command_targets_locked_mechanism_change(child_command):
                return True
        return False

    def _next_undo_redo_targets_locked_mechanism_change(self, *, redo: bool) -> bool:
        if not self.mechanism_editing_locked():
            return False
        stack = getattr(self, "_undo_stack", None)
        if stack is None:
            return False
        available = bool(stack.canRedo()) if redo else bool(stack.canUndo())
        if not available:
            return False
        index = int(stack.index()) if redo else int(stack.index()) - 1
        if index < 0 or index >= int(stack.count()):
            return False
        command = stack.command(index)
        return self._undo_command_targets_locked_mechanism_change(command)

    def _report_locked_reactions_undo_redo_block(self, *, redo: bool) -> None:
        action_text = "redo" if redo else "undo"
        self._status_label.setText(f"Unlock Reactions Editing to {action_text} mechanism changes")
        logger.debug("Blocked %s while reactions editing is locked", action_text)

    def _prompt_mechanism_edit_unlock_warning(self) -> bool:
        box = QtWidgets.QMessageBox(self)
        box.setIcon(QtWidgets.QMessageBox.Warning)
        box.setWindowTitle("Unlock Reactions Editing")
        box.setText("Edits in the Reactions editor change the canonical mechanism text.")
        box.setInformativeText(
            "Use unlocking only for deliberate edits. Save/load and migration rewrites can still update the editor while it remains locked."
        )
        unlock_btn = box.addButton("Unlock", QtWidgets.QMessageBox.ButtonRole.AcceptRole)
        cancel_btn = box.addButton(QtWidgets.QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(cancel_btn)
        box.setEscapeButton(cancel_btn)
        box.exec()
        return box.clickedButton() is unlock_btn

    def _on_mechanism_edit_lock_action_triggered(self, checked: bool) -> None:
        if bool(checked):
            if not bool(getattr(self, "_mechanism_edit_unlock_warning_shown", False)):
                if not self._prompt_mechanism_edit_unlock_warning():
                    self._refresh_mechanism_edit_lock_ui()
                    self._status_label.setText("Reactions editing remains locked")
                    return
                self._mechanism_edit_unlock_warning_shown = True
            self._set_mechanism_edit_locked(False)
            self._status_label.setText("Reactions editing unlocked")
            return
        self._set_mechanism_edit_locked(True)
        self._status_label.setText("Reactions editing locked")

    def _prompt_slider_transaction_invalidation(self, action_text: str) -> str:
        box = QtWidgets.QMessageBox(self)
        box.setIcon(QtWidgets.QMessageBox.Warning)
        box.setWindowTitle("Pending Slider Changes")
        box.setText(f"{str(action_text)} will discard pending slider changes.")
        box.setInformativeText(
            "Commit applies the staged slider transaction before continuing.\n"
            "Discard clears it.\n"
            "Cancel keeps the current pending edits."
        )
        commit_btn = box.addButton("Commit", QtWidgets.QMessageBox.ButtonRole.AcceptRole)
        discard_btn = box.addButton("Discard", QtWidgets.QMessageBox.ButtonRole.DestructiveRole)
        cancel_btn = box.addButton(QtWidgets.QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(cancel_btn)
        box.setEscapeButton(cancel_btn)
        box.exec()
        clicked = box.clickedButton()
        if clicked is commit_btn:
            return "commit"
        if clicked is discard_btn:
            return "discard"
        return "cancel"

    def _guard_slider_transaction_invalidation(
        self,
        *,
        action_text: str,
        concentration_rows: Optional[Sequence[int]] = None,
    ) -> bool:
        preview = getattr(self, "_preview_session", None)
        if preview is None or not hasattr(preview, "has_dirty_transaction"):
            return True
        if not bool(preview.has_dirty_transaction()):
            return True

        if concentration_rows is not None:
            rows = [int(row) for row in concentration_rows if int(row) >= 0]
            if not rows:
                return True
            if not bool(preview.preview_batch_cache_token(rows)):
                return True

        decision = str(self._prompt_slider_transaction_invalidation(action_text) or "cancel").strip().lower()
        if decision == "cancel":
            return False

        if decision == "commit":
            self._on_commit_slider_overrides_clicked()
            if bool(preview.has_dirty_transaction()):
                self.message_box_warning(
                    "Pending Slider Changes",
                    f"{str(action_text)} was canceled because pending slider changes could not be committed.",
                )
                return False
            return True

        self._discard_slider_transaction_for_invalidation()
        if bool(preview.has_dirty_transaction()):
            self.message_box_warning(
                "Pending Slider Changes",
                f"{str(action_text)} was canceled because pending slider changes could not be discarded.",
            )
            return False
        return True

    def _bind_simulation_panel_widgets(self) -> None:
        species_panel = self._try_wire_species_panel()

        sim_panel = self._batch_panel
        self._batch_section = sim_panel.section
        self._batch_table = sim_panel.batch_table
        self._add_batch_set_btn = sim_panel.add_batch_set_btn
        self._move_batch_up_btn = sim_panel.move_batch_up_btn
        self._move_batch_down_btn = sim_panel.move_batch_down_btn
        self._delete_batch_set_btn = sim_panel.delete_batch_set_btn
        self._solver_method_combo = sim_panel.solver_method_combo
        self._sim_time_spinbox = sim_panel.sim_time_spinbox
        self._num_points_spinbox = sim_panel.num_points_spinbox
        self._run_btn = sim_panel.run_btn
        self._solver_summary_label = sim_panel.solver_summary_label
        self._stop_btn = sim_panel.stop_btn
        self._sim_progress = sim_panel.sim_progress
        self._temperature_spinbox = sim_panel.temperature_spinbox

        self._rebind_batch_semantics_signal_bindings()

        if species_panel is not None and hasattr(species_panel, "attach"):
            try:
                species_panel.attach(table=self._batch_table, model=self._batch_model)
                if hasattr(species_panel, "activate"):
                    self._ensure_batch_current_row_selected()
                    species_panel.activate()
            except RuntimeError as exc:
                logger.debug("Failed to attach species panel to batch table: %s", exc, exc_info=True)
                self._species_panel_available = False

        self._on_batch_current_changed()
        self._update_solver_summary_label()

    def _disconnect_batch_semantics_signal_bindings(
        self,
        *,
        model: object | None = None,
        selection_model: object | None = None,
    ) -> None:
        if model is not None:
            for signal_name, slot in (
                ("showMembershipChanged", self._on_batch_show_membership_changed),
                ("sliderEditTargetsChanged", self._on_slider_edit_targets_changed),
            ):
                signal = getattr(model, signal_name, None)
                if signal is None:
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    try:
                        signal.disconnect(slot)
                    except (TypeError, RuntimeError):
                        pass
        if selection_model is not None:
            for signal_name, slot in (
                ("selectionChanged", self._on_batch_selection_changed),
                ("currentChanged", self._on_batch_current_changed),
            ):
                signal = getattr(selection_model, signal_name, None)
                if signal is None:
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    try:
                        signal.disconnect(slot)
                    except (TypeError, RuntimeError):
                        pass

    def _rebind_batch_semantics_signal_bindings(self) -> None:
        table = getattr(self, "_batch_table", None)
        current_model = getattr(self, "_batch_model", None)
        current_selection_model = table.selectionModel() if table is not None else None

        disconnected_models: list[object] = []
        for model in (
            getattr(self, "_connected_batch_semantics_model", None),
            current_model,
        ):
            if model is None or any(model is existing for existing in disconnected_models):
                continue
            self._disconnect_batch_semantics_signal_bindings(model=model)
            disconnected_models.append(model)

        disconnected_selection_models: list[object] = []
        for selection_model in (
            getattr(self, "_connected_batch_selection_model", None),
            current_selection_model,
        ):
            if selection_model is None or any(selection_model is existing for existing in disconnected_selection_models):
                continue
            self._disconnect_batch_semantics_signal_bindings(selection_model=selection_model)
            disconnected_selection_models.append(selection_model)

        if current_model is not None:
            try:
                current_model.showMembershipChanged.connect(self._on_batch_show_membership_changed)
                current_model.sliderEditTargetsChanged.connect(self._on_slider_edit_targets_changed)
            except RuntimeError as exc:
                logger.debug("Failed to connect batch model semantics signals: %s", exc, exc_info=True)
        if current_selection_model is not None:
            try:
                current_selection_model.selectionChanged.connect(self._on_batch_selection_changed)
                current_selection_model.currentChanged.connect(self._on_batch_current_changed)
            except RuntimeError as exc:
                logger.debug("Failed to connect batch selection signals: %s", exc, exc_info=True)

        self._connected_batch_semantics_model = current_model
        self._connected_batch_selection_model = current_selection_model

    def _try_wire_species_panel(self):
        try:
            species_panel = self._mechanism_editor.species_sliders_widget()
        except (AttributeError, RuntimeError, TypeError) as exc:
            logger.debug("Species sliders widget unavailable: %s", exc, exc_info=True)
            self._species_panel_available = False
            return None

        if species_panel is None:
            self._species_panel_available = False
            return None

        try:
            if hasattr(species_panel, "set_transaction_owner"):
                species_panel.set_transaction_owner(self._preview_session)
            species_panel.speciesEdited.connect(self._on_species_slider_edited)
            species_panel.speciesDragFinished.connect(self._on_species_slider_drag_finished)
        except RuntimeError as exc:
            logger.debug("Failed to wire species sliders signals: %s", exc, exc_info=True)
            self._species_panel_available = False
            return None

        return species_panel

    def _init_right_dock_and_datasets(self) -> None:
        right_dock_components = build_right_dock_and_dataset_manager(
            self,
            plot_tabs=self._plot_tabs,
            mechanism_getter=self._get_mechanism_text,
            simulation_runner=self._run_dataset_simulation,
            solver_settings_getter=self._get_solver_settings,
        )
        self._right_dock = right_dock_components.dock
        self._right_panel = right_dock_components.panel
        self._right_dock.setWidget(right_dock_components.container)
        self.addDockWidget(self._default_dock_area(self._right_dock), self._right_dock)

        # Centralize dataset/fitting coordination outside the main window.
        self._dataset_manager = right_dock_components.dataset_manager
        self._bootstrap_existing_datasets()

        # Reference species statistics table within plot tabs.
        self._results_table = self._plot_tabs._main_plot.stats_table()

    def _init_bottom_analysis_dock(self) -> None:
        analysis_widget = self._plot_tabs.main_plot_analysis_widget()
        if analysis_widget is None:
            analysis_widget = QtWidgets.QLabel("Analysis surfaces unavailable", self)
            analysis_widget.setObjectName("analysisSurfaceUnavailableLabel")

        bottom_dock_components = build_bottom_analysis_dock(
            self,
            analysis_widget=analysis_widget,
        )
        self._analysis_dock = bottom_dock_components.dock
        self._analysis_dock.setWidget(bottom_dock_components.container)
        self.addDockWidget(self._default_dock_area(self._analysis_dock), self._analysis_dock)
        self._apply_default_dock_arrangement()

    def _init_status_bar_widgets(self) -> None:
        self._status_bar = self.statusBar()
        self._status_label = QtWidgets.QLabel("Ready")
        self._status_bar.addWidget(self._status_label, stretch=1)

        # Inline algebra evaluation summary (no tooltips; keep minimal).
        self._algebra_status_label = QtWidgets.QLabel("")
        self._algebra_status_label.setStyleSheet("QLabel { font-size: 10px; }")
        self._status_bar.addPermanentWidget(self._algebra_status_label)

        # Temperature mode indicator in status bar.
        self._temperature_mode_indicator = QtWidgets.QLabel("Temperature: 298.15 K (isothermal)")
        self._temperature_mode_indicator.setStyleSheet("QLabel { font-style: italic; }")
        self._status_bar.addPermanentWidget(self._temperature_mode_indicator)

        # Profile indicator in status bar.
        self._profile_indicator = QtWidgets.QLabel("Profile: None")
        self._profile_indicator.setStyleSheet("QLabel { font-style: italic; }")
        self._status_bar.addPermanentWidget(self._profile_indicator)

    def _finish_window_composition(self) -> None:
        """Complete window composition before any persisted/bootstrap state is applied."""
        self._create_menus()
        self._init_ribbon_host()
        self._init_mixin_ports()
        self._connect_signals()

    def _bootstrap_window_state(self) -> None:
        """Apply launch overrides and persisted startup state after composition prerequisites exist."""
        self._apply_initial_profile_from_cli()
        self._load_settings()
        self._update_temperature_mode_indicator()

    def _init_mixin_ports(self) -> None:
        data_manager = getattr(getattr(self, "_right_panel", None), "_data_manager", None)
        self._fitting_ports = FittingMixinPorts(
            mechanism_editor=self._mechanism_editor,
            dataset_manager=self._dataset_manager,
            data_manager_getter=lambda: getattr(getattr(self, "_right_panel", None), "_data_manager", data_manager),
            status_setter=lambda text: self._status_label.setText(str(text)),
            temperature_getter=lambda: float(self._temperature_spinbox.value()),
            num_points_getter=lambda: int(self._num_points_spinbox.value()),
        )
        self._profile_ports = ProfileMixinPorts(
            profile_manager=self._profile_manager,
            profiles_menu_getter=lambda: self._profiles_menu,
            profile_indicator_setter=lambda text: self._profile_indicator.setText(str(text)),
            status_setter=lambda text: self._status_label.setText(str(text)),
            settings_set_value=self.settings_set_value,
            settings_remove=self.settings_remove,
            num_points_spinbox=self._num_points_spinbox,
            dark_mode_action=getattr(self, "_dark_mode_action", None),
            toggle_theme=self._toggle_theme,
            update_solver_summary_label=self._update_solver_summary_label,
        )

    def _apply_initial_profile_from_cli(self) -> None:
        if not self._initial_profile:
            return
        if self._profile_manager.get_profile(self._initial_profile):
            self._activate_profile(self._initial_profile)
            logger.info(f"Applied startup profile: {self._initial_profile}")
            return
        logger.warning(f"Startup profile '{self._initial_profile}' not found")

    def _on_programmatic_mechanism_load(self) -> None:
        """
        Invalidate slider runtime and clear variable sliders/overrides.

        Programmatic loads often set editor text with signals blocked (undo commands and some load
        paths), so MainWindow's `textChanged`-wired invalidation is not guaranteed to run.
        """
        self._preview_session.clear_working_transaction(clear_committed_slider_values=True)
        self._set_mechanism_edit_locked(True)
        try:
            self._mechanism_editor._variable_sliders.clear()
        except Exception:
            logger.debug("Failed to clear variable sliders after programmatic mechanism load", exc_info=True)
            self._preview_session.clear_pending_slider_values()
            self._variable_runtime.clear_prepared_slider_runtime(dirty=True)

        self._on_authoritative_mechanism_input_changed()
        self._refresh_slider_transaction_button_state()

        try:
            self._update_parameter_table_from_sliders()
        except Exception:
            logger.debug("Failed to update parameter table after programmatic mechanism load", exc_info=True)
            QtCore.QTimer.singleShot(0, self._update_parameter_table_from_sliders)

        self._refresh_overlay_swatches_for_current_mechanism()

    def _bootstrap_existing_datasets(self):
        """Populate dataset visualizations for any datasets already loaded."""
        from kindred.gui.controllers.dataset_manager import DatasetManagerError

        self._sync_color_manager_authoritative_roster()
        existing = self._right_panel._data_manager.get_datasets()
        if not existing:
            return

        for name, payload in existing.items():
            try:
                self._dataset_manager.register_dataset(name, payload)
            except DatasetManagerError as exc:
                logger.warning("Skipping dataset '%s' during bootstrap: %s", name, exc)
                continue

        self._sync_overlay_catalog()

    def _current_mechanism_species_roster_for_colors(self) -> tuple[str, ...] | None:
        """Best-effort current mechanism species roster for color canonicalization."""
        from kindred.core.batch_initial_conditions import (
            strip_named_reaction_dsl_initial_concentration_sets,
        )

        mechanism_text = str(self._get_mechanism_text() or "")
        if not mechanism_text.strip():
            return ()
        try:
            mechanism_text = strip_named_reaction_dsl_initial_concentration_sets(mechanism_text)
        except Exception:
            logger.debug("Failed to parse current mechanism species roster for color sync", exc_info=True)
            return None

        def _clean(names: Any) -> tuple[str, ...]:
            return tuple(str(name).strip() for name in (names or ()) if str(name).strip())

        last_mechanism = self._mechanism_helpers.last_mechanism()
        last_context = self._mechanism_helpers.last_mechanism_context()
        if last_mechanism is not None and str(last_context.get("dsl_text") or "") == mechanism_text:
            try:
                return _clean(last_mechanism.species_names())
            except Exception:
                logger.debug("Failed to reuse cached mechanism species roster for color sync", exc_info=True)

        try:
            from kindred.core.simulator.dsl import parse_dsl_to_mechanism
            from kindred.core.units import UnitsModel

            units = UnitsModel(temperature_K=float(self._temperature_spinbox.value()))
            mechanism = parse_dsl_to_mechanism(mechanism_text, initials={}, units=units)
            return _clean(mechanism.species_names())
        except Exception:
            logger.debug("Failed to parse current mechanism species roster for color sync", exc_info=True)
            return None

    def _sync_color_manager_authoritative_roster(self) -> None:
        from kindred.gui.color_manager import ColorManager

        roster = self._current_mechanism_species_roster_for_colors()
        if roster is None:
            return
        ColorManager.instance().set_current_species_roster(roster)

    def _refresh_overlay_swatches_for_current_mechanism(self) -> None:
        self._sync_color_manager_authoritative_roster()
        plot = getattr(self._plot_tabs, "_main_plot", None)
        refresh_overlay_presentation = getattr(plot, "refresh_overlay_presentation_for_current_roster", None)
        if callable(refresh_overlay_presentation):
            refresh_overlay_presentation()
            return
        overlay_panel = getattr(plot, "_overlay_panel", None)
        refresh_swatches = getattr(overlay_panel, "refresh_color_swatches", None)
        if callable(refresh_swatches):
            refresh_swatches()

    def _sync_overlay_catalog(self):
        """Propagate loaded datasets into the simulation overlay controls."""
        plot = getattr(self._plot_tabs, "_main_plot", None)
        data_manager = getattr(self._right_panel, "_data_manager", None)
        if plot is None or data_manager is None:
            return
        if not hasattr(plot, "set_overlay_catalog"):
            return
        self._sync_color_manager_authoritative_roster()
        plot.set_overlay_catalog(data_manager.get_datasets())

    def _snapshot_datasets(self) -> Dict[str, Dict[str, Any]]:
        """Capture lightweight metadata about loaded datasets."""
        data_manager = getattr(self._right_panel, "_data_manager", None)
        if data_manager is None:
            return {}

        snapshot: Dict[str, Dict[str, Any]] = {}
        for name, dataset in data_manager.get_datasets().items():
            t = dataset.get("t")
            species = dataset.get("species") or {}
            num_points = int(len(np.asarray(t).reshape(-1))) if t is not None else 0
            snapshot[name] = {
                "num_points": num_points,
                "species": list(species.keys()),
            }
        return snapshot

    def _load_settings(self):
        """Load user preferences from QSettings."""
        self.config_controller.load_settings()

    def _save_settings(self):
        """Save user preferences to QSettings."""
        self.config_controller.save_settings()

    def set_simulation_cache_caps(self, *, result_cap: int, preview_cap: int, persist: bool = True) -> None:
        self._sim_controller.set_simulation_cache_caps(result_cap=result_cap, preview_cap=preview_cap, persist=persist)

    def simulation_cache_stats(self) -> Dict[str, Dict[str, int]]:
        return self._sim_controller.simulation_cache_stats()

    def purge_simulation_result_cache(self) -> None:
        self._sim_controller.purge_simulation_result_cache()

    def purge_simulation_preview_cache(self) -> None:
        self._sim_controller.purge_simulation_preview_cache()

    def purge_simulation_all_caches(self) -> None:
        self._sim_controller.purge_simulation_all_caches()

    def _set_slider_debug_logging(self, enabled: bool, *, persist: bool = True, announce: bool = True) -> None:
        """
        Toggle verbose logging for programmatic slider updates.

        When enabled, logs are written to a deterministic, user-writable file location and
        the path is surfaced in the UI (so this works in compiled desktop distributions
        without a terminal/console).
        """
        sliders = getattr(getattr(self, "_mechanism_editor", None), "_variable_sliders", None)
        log_path: Optional[str] = None
        if bool(enabled):
            override = os.environ.get("KINDRED_SLIDER_DEBUG_LOG")
            if override:
                log_path = str(override)
            else:
                base_dir = QtCore.QStandardPaths.writableLocation(QtCore.QStandardPaths.AppDataLocation)
                if not base_dir:
                    base_dir = QtCore.QStandardPaths.writableLocation(QtCore.QStandardPaths.TempLocation)
                if base_dir:
                    try:
                        os.makedirs(base_dir, exist_ok=True)
                        log_path = os.path.join(base_dir, "kindred-slider-updates.log")
                    except OSError as exc:
                        logger.debug("Failed to create slider debug log dir %s: %s", base_dir, exc, exc_info=True)
                        log_path = None

        if sliders is not None and hasattr(sliders, "set_debug_slider_updates"):
            applied = False
            try:
                if hasattr(sliders, "set_debug_slider_log_path"):
                    sliders.set_debug_slider_log_path(log_path)
                sliders.set_debug_slider_updates(bool(enabled))
                applied = True
            except RuntimeError as exc:
                logger.debug("Failed to apply slider debug logging: %s", exc, exc_info=True)
                enabled = False
            except (AttributeError, TypeError, ValueError) as exc:
                logger.warning("Slider debug logging failed: %s", exc, exc_info=True)
                enabled = False
            if bool(enabled) and not applied:
                enabled = False

        if persist:
            try:
                self.config_controller.persist_slider_debug_updates(bool(enabled))
            except Exception as exc:
                logger.warning("Failed to persist slider debug logging setting: %s", exc, exc_info=True)
                bar = getattr(self, "_status_bar", None)
                if bar is not None:
                    try:
                        bar.showMessage("Failed to persist slider debug logging setting (see logs)", 8000)
                    except RuntimeError as exc:
                        logger.debug("Failed to show slider debug persistence error in status bar: %s", exc, exc_info=True)
                        self._status_bar = None

        if announce:
            try:
                if bool(enabled) and log_path:
                    self._status_bar.showMessage(f"Slider debug logs: {log_path}", 15000)
                    QtWidgets.QMessageBox.information(
                        self,
                        "Slider Debug Logging Enabled",
                        f"Writing slider debug logs to:\n{log_path}",
                    )
                elif not bool(enabled):
                    self._status_bar.showMessage("Slider debug logging disabled", 5000)
                else:
                    self._status_bar.showMessage("Unable to enable slider debug logging (see logs)", 8000)
            except RuntimeError as exc:
                logger.debug("Failed to announce slider debug logging state: %s", exc, exc_info=True)
                self._status_bar = None

    def _create_menus(self):
        """Create the menu bar and shortcut customization registry."""
        self._shortcut_actions = {}
        menubar = self.menuBar()

        def add_items(menu: QtWidgets.QMenu, items: Sequence[Any]) -> None:
            for item in items:
                if item is None:
                    menu.addSeparator()
                    continue
                if isinstance(item, tuple) and item and item[0] == "submenu":
                    _, title, attr, after = item
                    submenu = menu.addMenu(str(title))
                    if isinstance(attr, str) and attr:
                        setattr(self, attr, submenu)
                    if callable(after):
                        after()
                    continue

                text, callback, shortcut, object_name, tooltip, *rest = item
                extra = rest[0] if rest else {}
                if extra is None:
                    extra = {}
                self._build_action(
                    menu=menu,
                    text=str(text),
                    callback=callback,
                    shortcut=shortcut,
                    object_name=object_name,
                    tooltip=tooltip,
                    **dict(extra),
                )

        file_menu = menubar.addMenu("&File")
        self._file_menu = file_menu
        add_items(
            file_menu,
            [
                ("Load Project...", self.project_controller.load_project, QtGui.QKeySequence.Open, "loadProjectAction", "Load a Kindred project file (.kin)"),
                ("Save Project...", self.project_controller.save_project, QtGui.QKeySequence.Save, "saveProjectAction", "Save current mechanism and settings to a project file"),
                ("Load Data...", self._load_data_via_action, "Ctrl+Shift+L", "loadDataAction", "Load experimental CSV data (same as the Data panel 'Load' button)"),
                ("submenu", "Recent Projects", "_recent_menu", self._update_recent_files_menu),
                None,
                ("Export CSV...", self.project_controller.export_data, "Ctrl+E", "exportCsvAction", "Export simulation results to CSV format"),
                None,
                ("E&xit", self.close, QtGui.QKeySequence.Quit, "exitAction", "Exit Kindred application"),
            ],
        )

        edit_menu = menubar.addMenu("&Edit")
        add_items(
            edit_menu,
            [
                ("&Undo", self._undo, QtGui.QKeySequence.Undo, "undoAction", "Undo last edit in mechanism editor"),
                ("&Redo", self._redo, QtGui.QKeySequence.Redo, "redoAction", "Redo previously undone edit"),
                ("Unlock Reactions Editing", self._on_mechanism_edit_lock_action_triggered, None, "mechanismEditLockAction", "Temporarily enable deliberate edits in the Reactions editor", {"checkable": True, "checked": False, "store_as": "_mechanism_edit_lock_action"}),
                None,
                ("&Species Registry...", self._open_species_registry, None, "speciesRegistryAction", "View and manage species definitions and properties"),
                ("State &Network Editor...", self._open_state_network, None, "stateNetworkAction", "Edit state transition networks for TST calculations"),
                ("&Computational Mode...", self._open_computational_mode, None, "computationalModeAction", "Convert absolute computed free energies into energy-mode DSL blocks"),
                None,
                ("Preferences...", self._open_preferences, QtGui.QKeySequence.Preferences, "preferencesAction", "Configure application preferences and settings"),
                ("Customize &Keyboard Shortcuts...", self._open_shortcuts_dialog, "Ctrl+K", "customizeShortcutsAction", "Customize keyboard shortcuts for actions"),
            ],
        )

        self._refresh_mechanism_edit_lock_ui()

        view_menu = menubar.addMenu("&View")
        self._view_menu = view_menu
        panels_menu = view_menu.addMenu("Panels")
        self._panels_menu = panels_menu
        self._dock_toggle_actions: Dict[str, QtGui.QAction] = {}
        for spec in dock_shell_specs():
            dock = getattr(self, spec.attr_name)
            dock_action = dock.toggleViewAction()
            dock_action.setText(spec.title)
            dock_action.setObjectName(f"{spec.identity_key}DockToggleAction")
            panels_menu.addAction(dock_action)
            self._dock_toggle_actions[spec.identity_key] = dock_action

        self._build_action(
            menu=view_menu,
            text="Panel Layout Tips...",
            callback=self._show_panel_layout_tips,
            shortcut=None,
            object_name="panelLayoutTipsAction",
            tooltip="Explain how to arrange multiple panels together on the same side of the workspace",
            store_as="_panel_layout_tips_action",
        )

        analysis_surfaces_menu = view_menu.addMenu("Analysis Surfaces")
        self._analysis_surfaces_menu = analysis_surfaces_menu
        self._analysis_surface_actions: Dict[str, QtGui.QAction] = {}
        for surface_name in _ANALYSIS_SURFACE_NAMES:
            action = self._build_action(
                menu=analysis_surfaces_menu,
                text=str(surface_name),
                callback=lambda _checked=False, surface=surface_name: self._show_analysis_surface(surface),
                shortcut=None,
                object_name=f"view{surface_name}SurfaceAction",
                tooltip=f"Show the {surface_name} surface in the Analysis dock",
            )
            self._analysis_surface_actions[str(surface_name)] = action
        self._refresh_analysis_surface_actions()

        view_menu.addSeparator()
        add_items(
            view_menu,
            [
                ("&Reset Layout", self._reset_layout, "Ctrl+Shift+R", "resetLayoutAction", "Restore dock panels to default layout", {"store_as": "_reset_layout_action"}),
                None,
                ("&Dark Mode", self._toggle_theme, "Ctrl+Shift+D", "darkModeAction", "Toggle between light and dark theme", {"checkable": True, "store_as": "_dark_mode_action"}),
            ],
        )

        self._profiles_menu = QtWidgets.QMenu("&Profiles", self)
        menubar.addMenu(self._profiles_menu)
        self._update_profiles_menu()

        examples_menu = menubar.addMenu("E&xamples")
        presets_submenu = examples_menu.addMenu("Preset Mechanisms")
        for preset_id in self._available_preset_ids():
            presets_submenu.addAction(preset_id, lambda pid=preset_id: self._load_preset_mechanism(pid))
        examples_menu.addSeparator()
        add_items(
            examples_menu,
            [("&Template Manager...", self._open_template_manager, "Ctrl+T", "templateManagerAction", "Browse and manage custom mechanism templates (Ctrl+T)")],
        )

        sim_menu = menubar.addMenu("&Simulation")
        add_items(
            sim_menu,
            [
                ("&Run", self._sim_controller.run_simulation, None, "runSimulationAction", "Run kinetic simulation with current mechanism (Ctrl+R or F5)", {"shortcuts": ["Ctrl+R", "F5"]}),
                ("&Stop", self._sim_controller.stop_simulation, "Esc", "stopSimulationAction", "Stop running simulation (Esc)"),
                None,
                ("Simulation &Settings...", self._open_solver_settings, None, "simulationSettingsAction", "Configure solver tolerances and advanced simulation settings"),
            ],
        )

        fit_menu = menubar.addMenu("&Fitting")
        add_items(
            fit_menu,
            [
                ("&Configure...", self._configure_fitting, None, "configureFitAction", "Configure fitting parameters and bounds"),
                ("Global Fit...", self._run_global_fit, "Ctrl+Shift+F", "globalFitAction", "Fit shared parameters across all loaded datasets (Ctrl+Shift+F)"),
            ],
        )

        tools_menu = menubar.addMenu("&Tools")
        add_items(
            tools_menu,
            [
                ("&Temperature Schedule...", self._open_temperature_schedule_editor, None, "temperatureScheduleAction", "Create piecewise temperature schedules with visual preview"),
                None,
            ],
        )
        debug_menu = tools_menu.addMenu("Debug")
        add_items(
            debug_menu,
            [
                ("Log Slider Updates", self._set_slider_debug_logging, None, "debugSlidersAction", "Log programmatic K-slider updates (for diagnosing slider snapping).", {"checkable": True, "signal": "toggled", "store_as": "_debug_sliders_action"}),
            ],
        )

        help_menu = menubar.addMenu("&Help")
        add_items(
            help_menu,
            [
                ("&Documentation", self._open_docs, QtGui.QKeySequence.HelpContents, "documentationAction", "Open Kindred documentation (shows offline guidance if online docs are unavailable)"),
                ("&Interactive Tutorials...", self._show_tutorials, None, "tutorialsAction", "Launch step-by-step interactive tutorials"),
                ("&Keyboard Shortcuts", self._show_keyboard_shortcuts, "Ctrl+?", "keyboardShortcutsAction", "View list of keyboard shortcuts (Ctrl+?)"),
                None,
                ("&About", self._show_about, None, "aboutAction", "About Kindred - version and license information"),
            ],
        )

    def _build_action(
        self,
        *,
        menu: QtWidgets.QMenu,
        text: str,
        callback: Optional[Callable[..., Any]] = None,
        shortcut: Optional[Any] = None,
        shortcuts: Optional[List[Any]] = None,
        object_name: Optional[str] = None,
        tooltip: Optional[str] = None,
        checkable: bool = False,
        checked: Optional[bool] = None,
        signal: str = "triggered",
        shortcut_registry_prefix: str = "",
        store_as: Optional[str] = None,
    ) -> QtGui.QAction:
        action = QtGui.QAction(text, self)

        if object_name:
            action.setObjectName(str(object_name))
        if tooltip:
            action.setToolTip(str(tooltip))

        if checkable:
            action.setCheckable(True)
        if checked is not None:
            action.setChecked(bool(checked))

        if shortcuts is not None:
            if shortcuts:
                action.setShortcut(shortcuts[0])
            action.setShortcuts(shortcuts)
        elif shortcut is not None:
            action.setShortcut(shortcut)

        if callback is not None:
            signal_obj = getattr(action, signal, None)
            if signal_obj is None:
                raise AttributeError(f"QAction has no signal: {signal!r}")
            signal_obj.connect(callback)

        menu.addAction(action)

        seq = action.shortcut()
        if seq is not None and not seq.isEmpty():
            title = action.text().replace("&", "").replace("...", "")
            full_name = f"{shortcut_registry_prefix}{title}"
            self._shortcut_actions[full_name] = {
                "action": action,
                "description": action.toolTip() or title,
                "default": seq.toString(),
            }

        if store_as:
            setattr(self, store_as, action)
        return action

    def _init_ribbon_host(self) -> None:
        self._ribbon_toolbar = QtWidgets.QToolBar("Ribbon", self)
        self._ribbon_toolbar.setObjectName("mainRibbonToolbar")
        self._ribbon_toolbar.setMovable(False)
        self._ribbon_toolbar.setFloatable(False)
        self._ribbon_toolbar.setAllowedAreas(QtCore.Qt.TopToolBarArea)
        self._ribbon_toolbar.setContextMenuPolicy(QtCore.Qt.PreventContextMenu)

        self._ribbon_host = CollapsibleRibbonHost(self)
        self._ribbon_host.add_page(self._build_view_ribbon_page())
        self._ribbon_host.collapseToggleRequested.connect(self.set_ribbon_collapsed)

        self._ribbon_toolbar.addWidget(self._ribbon_host)
        self.addToolBar(QtCore.Qt.TopToolBarArea, self._ribbon_toolbar)

    def _build_view_ribbon_page(self) -> RibbonPage:
        page = RibbonPage("View", self)
        page.setObjectName("viewRibbonPage")

        panels_group = RibbonGroup("Panels", page)
        for spec in dock_shell_specs():
            action = self._dock_toggle_actions[spec.identity_key]
            panels_group.add_compact_action(
                action,
                object_name=f"{spec.identity_key}RibbonButton",
            )

        analysis_group = RibbonGroup("Analysis", page)
        for surface_name in _ANALYSIS_SURFACE_NAMES:
            analysis_group.add_compact_action(
                self._analysis_surface_actions[str(surface_name)],
                object_name=f"{str(surface_name).lower()}SurfaceRibbonButton",
            )

        window_group = RibbonGroup("Window", page)
        window_group.add_primary_action(
            self._reset_layout_action,
            text_override="Restore Default Layout",
            object_name="resetLayoutRibbonButton",
        )
        window_group.add_compact_action(
            self._panel_layout_tips_action,
            object_name="panelLayoutTipsRibbonButton",
        )
        window_group.add_compact_action(
            self._dark_mode_action,
            object_name="darkModeRibbonButton",
        )

        page.add_group(panels_group)
        page.add_group(analysis_group)
        page.add_group(window_group)
        return page

    def set_ribbon_collapsed(self, collapsed: bool) -> None:
        ribbon_host = getattr(self, "_ribbon_host", None)
        if ribbon_host is None:
            return
        collapsed_value = bool(collapsed)
        if ribbon_host.is_collapsed() == collapsed_value:
            return
        self._run_preserving_maximized_state(lambda: ribbon_host.set_collapsed(collapsed_value))

    def ribbon_collapsed(self) -> bool:
        ribbon_host = getattr(self, "_ribbon_host", None)
        if ribbon_host is None:
            return False
        return bool(ribbon_host.is_collapsed())

    def _connect_signals(self):
        """Connect signals between components."""
        # Data manager signals
        self._right_panel._data_manager.datasetLoaded.connect(self._on_dataset_loaded)
        self._right_panel._data_manager.datasetRemoved.connect(self._on_dataset_removed)

        # Temperature mode indicator updates
        self._temperature_spinbox.valueChanged.connect(self._update_temperature_mode_indicator)
        self._mechanism_editor._reactions_text.textChanged.connect(self._update_temperature_mode_indicator)
        self._mechanism_editor._reactions_text.textChanged.connect(self._on_authoritative_mechanism_input_changed)
        self._mechanism_editor._reactions_text.textChanged.connect(self._refresh_overlay_swatches_for_current_mechanism)
        try:
            self._mechanism_editor._state_network_editor.stateNetworkChanged.connect(
                self._on_authoritative_mechanism_input_changed
            )
            self._mechanism_editor._state_network_editor.stateNetworkChanged.connect(
                self._refresh_overlay_swatches_for_current_mechanism
            )
        except Exception as exc:
            # State network editor may not expose the signal in some contexts
            self._state_network_editor_invalidation_signal_available = False
            self._record_best_effort_failure(
                "main_window.state_network_editor.stateNetworkChanged.connect",
                message="State network editor did not expose stateNetworkChanged signal",
                exc=exc,
            )

    def _record_best_effort_failure(
        self,
        key: str,
        *,
        message: str,
        exc: Optional[Exception] = None,
        max_logs: int = 3,
    ) -> None:
        record_gui_best_effort_failure(
            self,
            str(key),
            message=message,
            exc=exc,
            log=logger,
            max_logs=int(max_logs),
        )

    def _invalidate_slider_runtime(self):
        """Mark the cached slider runtime as stale."""
        self._variable_runtime.invalidate_slider_runtime()

    @staticmethod
    def _normalized_mechanism_text_for_invalidation_guard(text: str) -> str:
        return "\n".join(" ".join(str(line).split()) for line in str(text or "").splitlines()).strip()

    def _on_authoritative_mechanism_input_changed(self) -> None:
        """Invalidate stale displayed results when the authoritative mechanism changes."""
        if bool(getattr(self, "_suppress_authoritative_mechanism_input_change", False)):
            return
        # Pending-init rewrite normalizes the DSL after a successful explicit run;
        # it should not be treated like a new authoritative mechanism edit that
        # evicts the result that just produced the rewrite.
        if self._variable_runtime.suppress_slider_runtime_invalidation():
            return
        pending_init_rewrite = getattr(self, "_pending_init_migration_rewrite_for_invalidation", None)
        pending_init_state_network = getattr(
            self,
            "_pending_init_migration_state_network_for_invalidation",
            None,
        )
        if pending_init_rewrite is not None or pending_init_state_network is not None:
            self._pending_init_migration_rewrite_for_invalidation = None
            self._pending_init_migration_state_network_for_invalidation = None
            if (
                self._normalized_mechanism_text_for_invalidation_guard(self.mechanism_reactions_text_raw())
                == self._normalized_mechanism_text_for_invalidation_guard(str(pending_init_rewrite))
                and self._normalized_mechanism_text_for_invalidation_guard(self.mechanism_state_network_dsl_raw())
                == self._normalized_mechanism_text_for_invalidation_guard(str(pending_init_state_network))
            ):
                return
        self._invalidate_slider_runtime()
        batch_cache = getattr(self._sim_controller, "batch_cache", None)
        has_active_cache = bool(
            batch_cache is not None
            and (
                str(batch_cache.active_cache_key or "").strip()
                or str(batch_cache.active_preview_cache_key or "").strip()
            )
        )
        has_displayed_selection = bool(
            batch_cache is not None
            and (
                str(batch_cache.active_batch_set_id or "").strip()
                or str(batch_cache.active_batch_set or "").strip()
                or batch_cache.last_display_selection
            )
        )
        if not (has_active_cache or has_displayed_selection or self.main_plot_has_data()):
            return
        self._invalidate_active_results_after_authoritative_mechanism_change()

    def _parse_sim_time_seconds(self) -> float:
        raw = ""
        try:
            raw = str(self._sim_time_spinbox.text()).strip()
        except Exception:
            raw = ""
        if not raw:
            raise ValueError("t_end is empty")
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"t_end must be a number, got: {raw!r}") from exc
        if not math.isfinite(value):
            raise ValueError(f"t_end must be finite, got: {raw!r}")
        if value <= 0.0:
            raise ValueError(f"t_end must be > 0, got: {raw!r}")
        return float(value)

    def _coerce_solver_tolerance(
        self,
        value: object,
        *,
        field_name: str,
        current_value: Optional[float],
        default_value: float,
    ) -> Optional[float]:
        if value is None:
            return None
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            fallback = current_value if current_value is not None else default_value
            logger.warning("Invalid solver tolerance %s=%r; keeping %s", field_name, value, fallback)
            return current_value
        if not math.isfinite(parsed) or parsed <= 0.0:
            fallback = current_value if current_value is not None else default_value
            logger.warning("Invalid solver tolerance %s=%r; keeping %s", field_name, value, fallback)
            return current_value
        return float(parsed)

    def _apply_solver_runtime_state(
        self,
        *,
        solver: object = _SOLVER_STATE_UNSET,
        rtol: object = _SOLVER_STATE_UNSET,
        atol: object = _SOLVER_STATE_UNSET,
        sync_combo: bool = True,
    ) -> None:
        solver_contract = load_solver_contract()

        if solver is not _SOLVER_STATE_UNSET:
            solver_label = str(solver or "").strip() or solver_contract.default_solver_name
            self._initial_solver = str(solver_label)
        if rtol is not _SOLVER_STATE_UNSET:
            self._initial_rtol = self._coerce_solver_tolerance(
                rtol,
                field_name="rtol",
                current_value=self._initial_rtol,
                default_value=1e-6,
            )
        if atol is not _SOLVER_STATE_UNSET:
            self._initial_atol = self._coerce_solver_tolerance(
                atol,
                field_name="atol",
                current_value=self._initial_atol,
                default_value=1e-12,
            )

        if bool(sync_combo):
            combo = getattr(self, "_solver_method_combo", None)
            if combo is not None:
                solver_label = str(self._initial_solver or solver_contract.default_solver_name).strip() or solver_contract.default_solver_name
                combo.blockSignals(True)
                try:
                    idx = int(combo.findText(str(solver_label)))
                    if idx >= 0:
                        combo.setCurrentIndex(idx)
                    else:
                        default_idx = int(combo.findText(str(solver_contract.default_solver_name)))
                        if default_idx >= 0:
                            combo.setCurrentIndex(default_idx)
                finally:
                    combo.blockSignals(False)

        self._update_solver_summary_label()

    def _on_solver_method_changed(self, v: str) -> None:
        self._apply_solver_runtime_state(solver=str(v), sync_combo=False)

    def _update_solver_summary_label(self) -> None:
        """Refresh the solver summary label shown in the Solver section."""
        if not hasattr(self, "_solver_summary_label"):
            return

        solver_contract = load_solver_contract()
        solver_label = str(self._initial_solver or solver_contract.default_solver_name).strip() or solver_contract.default_solver_name
        solver_method, solver_warning = solver_contract.normalize_solver_name(solver_label)
        solver_display = str(solver_method)
        if str(solver_method) != str(solver_label):
            solver_display = f"{solver_label} → {solver_method}"
        if solver_warning:
            solver_display = f"{solver_display} ({solver_warning})"
        rtol = self._initial_rtol or 1e-6
        atol = self._initial_atol or 1e-12
        points = int(self._num_points_spinbox.value())
        sim_time_raw = ""
        try:
            sim_time_raw = str(self._sim_time_spinbox.text()).strip()
        except Exception:
            sim_time_raw = ""
        try:
            sim_time = self._parse_sim_time_seconds()
        except Exception:
            sim_time = None

        time_part = f"{sim_time:g} s" if isinstance(sim_time, float) else f"{sim_time_raw or '?'} s"
        summary = f"Solver: {solver_display} • rtol={rtol:.1e} • atol={atol:.1e} • Points: {points:,} • Time: {time_part}"
        if self._use_sparse_jacobian and str(solver_method).upper() in {"RADAU", "BDF"}:
            summary += " • Sparse J"
        self._solver_summary_label.setText(summary)

    def _update_temperature_mode_indicator(self) -> None:
        """
        Update temperature mode indicator in status bar.

        Shows one of:
        - "Temperature: XXX K (isothermal)" - when no schedule in DSL
        - "Temperature: Schedule (N intervals)" - when piecewise schedule detected
        - "Temperature: Schedule (constant)" - when temp_const detected
        - "Temperature: Schedule (response, N intervals)" - when temp_response detected

        Priority rule:
        - If temperature schedule is defined in DSL (temp_step, temp_response, or temp_const),
          it takes precedence during ODE integration.
        - Otherwise, the temperature spinbox value is used (isothermal).
        """
        if not hasattr(self, "_temperature_mode_indicator"):
            return

        # Get current mechanism text
        mechanism_text = self._mechanism_editor._reactions_text.toPlainText()
        state_network_text = ""
        try:
            state_network_text = self._mechanism_editor._state_network_editor.get_state_network_dsl()
        except Exception:
            state_network_text = ""
        energy_mode_active = bool(str(state_network_text or "").strip())
        if energy_mode_active:
            self._set_temperature_override_state(
                enabled=True,
                tooltip="Temperature for thermodynamic calculations (energy mode: add T=... to override).",
            )
        else:
            self._set_temperature_override_state(
                enabled=True,
                tooltip="Temperature for thermodynamic calculations",
            )

        if energy_mode_active:
            T_override = self._dsl_global_temperature_K(mechanism_text)
            if T_override is not None:
                indicator_text = f"Temperature: {T_override:.2f} K (from DSL)"
                self._temperature_mode_indicator.setText(indicator_text)
                self._set_temperature_override_state(
                    enabled=False,
                    tooltip="Overridden by energy-mode DSL (T=...).",
                )
                logger.debug(f"Temperature mode indicator updated: {indicator_text}")
                return

        # Try to detect temperature schedule in DSL
        schedule_defined = False
        try:
            from kindred.core.temperature_dsl import parse_temperature_schedule

            temp_schedule = parse_temperature_schedule(mechanism_text)
            schedule_defined = temp_schedule is not None

            if temp_schedule is not None:
                # Temperature schedule detected
                if temp_schedule.schedule_type == "constant":
                    # Constant temperature from DSL
                    T = temp_schedule(0.0)
                    indicator_text = f"Temperature: {T:.2f} K (constant from DSL)"
                elif temp_schedule.schedule_type == "piecewise":
                    # Piecewise schedule
                    intervals = temp_schedule.get_intervals()
                    n_intervals = len(intervals)
                    indicator_text = f"Temperature: Schedule ({n_intervals} interval{'s' if n_intervals != 1 else ''})"
                elif temp_schedule.schedule_type == "response":
                    intervals = temp_schedule.get_intervals()
                    n_intervals = len(intervals)
                    indicator_text = (
                        f"Temperature: Schedule (response, {n_intervals} interval"
                        f"{'s' if n_intervals != 1 else ''})"
                    )
                else:
                    indicator_text = "Temperature: Schedule (unknown type)"
            else:
                # No temperature schedule - use spinbox value
                T = self._temperature_spinbox.value()
                if energy_mode_active:
                    indicator_text = f"Temperature: {T:.2f} K (energy mode: set T=... in DSL)"
                else:
                    indicator_text = f"Temperature: {T:.2f} K (isothermal)"

        except Exception as e:
            # Fallback if parsing fails
            logger.debug(f"Temperature schedule parsing failed: {e}")
            T = self._temperature_spinbox.value()
            if energy_mode_active:
                indicator_text = f"Temperature: {T:.2f} K (energy mode: set T=... in DSL)"
            else:
                indicator_text = f"Temperature: {T:.2f} K (isothermal)"

        self._temperature_mode_indicator.setText(indicator_text)
        if energy_mode_active and schedule_defined:
            self._set_temperature_override_state(
                enabled=False,
                tooltip="Overridden by DSL temperature schedule.",
            )
        logger.debug(f"Temperature mode indicator updated: {indicator_text}")

    def _load_data_via_action(self) -> None:
        """Invoke the Data panel's Load dialog from the File menu."""
        data_manager = getattr(self._right_panel, "_data_manager", None)
        if data_manager is None:
            QtWidgets.QMessageBox.warning(
                self,
                "Data Panel Unavailable",
                "The Data panel is not available in the current layout."
            )
            return
        data_manager.trigger_load_dialog()

    def _on_dataset_loaded(self, name: str, data: dict):
        """Handle dataset loaded from data manager."""
        from kindred.gui.controllers.dataset_manager import DatasetManagerError

        logger.info(f"Dataset loaded: {name}")

        self._sync_color_manager_authoritative_roster()
        try:
            self._dataset_manager.register_dataset(name, data)
        except DatasetManagerError as exc:
            logger.warning(f"Dataset '{name}' missing usable species: {exc}")
            QtWidgets.QMessageBox.warning(
                self,
                "Dataset Skipped",
                f"Dataset '{name}' cannot be visualized:\n\n{exc}"
            )
            return

        self._status_label.setText(f"Dataset '{name}' loaded ({len(data['t'])} points)")
        self._sync_overlay_catalog()

    def _on_dataset_removed(self, name: str):
        """Handle dataset removal from the data manager."""
        logger.info(f"Dataset removed: {name}")

        removed_entry = self._dataset_manager.remove_dataset(name)

        if removed_entry:
            self._status_label.setText(f"Dataset '{name}' removed")
        self._sync_overlay_catalog()

    def _scan_mechanism_parameters(self):
        """
        Extract parameters from current mechanism DSL.

        P2 ENHANCEMENT: Now uses caching to avoid re-scanning unchanged mechanisms.
        This improves performance, especially for large mechanisms.
        """
        from kindred.gui.controllers.dataset_manager import DatasetManagerError

        mechanism_text = self._get_mechanism_text()

        if not mechanism_text.strip():
            QtWidgets.QMessageBox.warning(
                self,
                "No Mechanism",
                "Please define a mechanism before scanning for parameters."
            )
            return

        try:
            self._dataset_manager.scan_mechanism_parameters(mechanism_text)
        except DatasetManagerError as exc:
            QtWidgets.QMessageBox.information(
                self,
                "No Parameters Found",
                str(exc)
            )
            return
        except Exception as exc:
            logger.error(f"Error scanning parameters: {exc}", exc_info=True)
            QtWidgets.QMessageBox.critical(
                self,
                "Parameter Scan Error",
                f"Failed to scan mechanism for parameters:\n\n{exc}"
            )
            return

    def _get_preferred_target_species(self) -> Optional[str]:
        """
        Get the preferred target species from the currently visible series in the main plot.

        Returns
        -------
        str or None
            The first visible species name, or None if no plot or no visible series.
        """
        try:
            # Get the main plot panel from plot_tabs
            main_plot = getattr(self._plot_tabs, "_main_plot", None)
            if main_plot is None:
                logger.debug("Main plot not available for target species selection")
                return None

            # Check if the plot has a visible_series method (PyQtGraph plot)
            if not hasattr(main_plot, "visible_series"):
                logger.debug("Main plot does not have visible_series method")
                return None

            # Get visible series
            visible = main_plot.visible_series()
            if visible and len(visible) > 0:
                # Use the first visible species
                target = visible[0]
                logger.debug(f"Selected target species from visible series: {target}")
                return target
            else:
                logger.debug("No visible series in main plot")
                return None

        except Exception as exc:
            logger.debug(f"Failed to get preferred target species from plot: {exc}")
            return None

    def _load_preset_mechanism(self, preset_id: str):
        """Load a bundled preset mechanism into the mechanism editor."""
        try:
            from kindred.io.resources import get_preset_mechanism
            from kindred.gui.undo_commands import SetMechanismTextCommand

            mechanism_text = get_preset_mechanism(preset_id)
            if not self._guard_slider_transaction_invalidation(action_text=f"Loading preset {preset_id}"):
                self._status_label.setText("Canceled preset load")
                return
            old_text = self._mechanism_editor._reactions_text.toPlainText()
            command = SetMechanismTextCommand(
                self._mechanism_editor._reactions_text,
                mechanism_text,
                old_text,
                f"Load preset {preset_id}"
            )
            self._undo_stack.push(command)
            self._on_programmatic_mechanism_load()
            self._status_label.setText(f"Loaded preset mechanism: {preset_id}")
            logger.info(f"Loaded preset mechanism: {preset_id}")

        except Exception as e:
            logger.error(f"Failed to load preset {preset_id}: {e}", exc_info=True)
            QtWidgets.QMessageBox.critical(
                self,
                "Load Error",
                f"Failed to load preset mechanism {preset_id}:\n\n{e}"
            )

    def _open_template_manager(self):
        """Open template manager dialog."""
        from kindred.gui.widgets.template_manager_dialog import TemplateManagerDialog

        current_text = self._mechanism_editor._reactions_text.toPlainText()

        dialog = TemplateManagerDialog(
            parent=self,
            template_manager=self._template_manager,
            current_mechanism_text=current_text
        )

        # Connect signal to load template
        dialog.templateLoadRequested.connect(self._load_template_from_manager)

        dialog.exec()

    def _load_template_from_manager(self, mechanism_text: str):
        """Load template from template manager."""
        from kindred.gui.undo_commands import SetMechanismTextCommand

        if not self._guard_slider_transaction_invalidation(action_text="Loading this template"):
            self._status_label.setText("Canceled template load")
            return

        old_text = self._mechanism_editor._reactions_text.toPlainText()

        # Create undo command
        command = SetMechanismTextCommand(
            self._mechanism_editor._reactions_text,
            mechanism_text,
            old_text,
            "Load template"
        )
        self._undo_stack.push(command)
        self._on_programmatic_mechanism_load()

        self._status_label.setText("Loaded template from Template Manager")
        logger.info("Loaded template from Template Manager")

    def _open_preferences(self):
        """Open preferences dialog."""
        from kindred.core.simulator.solvers import DEFAULT_SOLVER_NAME

        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Preferences")
        dialog.setMinimumWidth(400)

        layout = QtWidgets.QVBoxLayout(dialog)

        # Default solver
        solver_group = QtWidgets.QGroupBox("Default Solver Settings")
        solver_layout = QtWidgets.QFormLayout()

        solver_combo = QtWidgets.QComboBox()
        solver_combo.addItems(['LSODA', 'Radau', 'BDF'])
        solver_combo.setCurrentText(self._initial_solver or str(DEFAULT_SOLVER_NAME))
        solver_layout.addRow("Solver:", solver_combo)

        rtol_spin = QtWidgets.QDoubleSpinBox()
        rtol_spin.setDecimals(12)
        rtol_spin.setRange(1e-15, 1e-3)
        rtol_spin.setValue(self._initial_rtol or 1e-6)
        rtol_spin.setSingleStep(1e-7)
        solver_layout.addRow("Relative Tolerance:", rtol_spin)

        atol_spin = QtWidgets.QDoubleSpinBox()
        atol_spin.setDecimals(15)
        atol_spin.setRange(1e-18, 1e-6)
        atol_spin.setValue(self._initial_atol or 1e-12)
        atol_spin.setSingleStep(1e-13)
        solver_layout.addRow("Absolute Tolerance:", atol_spin)

        solver_group.setLayout(solver_layout)
        layout.addWidget(solver_group)

        # Buttons
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        if dialog.exec() == QtWidgets.QDialog.Accepted:
            self._apply_solver_runtime_state(
                solver=solver_combo.currentText(),
                rtol=rtol_spin.value(),
                atol=atol_spin.value(),
            )
            logger.info(f"Preferences updated: solver={self._initial_solver}, rtol={self._initial_rtol}, atol={self._initial_atol}")
            self._status_label.setText("Preferences updated")

    def _load_custom_shortcuts(self, shortcuts_dict: dict):
        """
        Load custom keyboard shortcuts from saved settings.

        Args:
            shortcuts_dict: Dictionary mapping action names to shortcut strings
        """
        if not shortcuts_dict:
            return

        for action_name, shortcut_str in shortcuts_dict.items():
            entry = self._shortcut_actions.get(action_name)
            if not entry:
                continue

            action = entry["action"]
            if shortcut_str:
                action.setShortcut(QtGui.QKeySequence(shortcut_str))
            else:
                action.setShortcut(QtGui.QKeySequence())

        logger.debug("Loaded %d custom keyboard shortcuts", len(shortcuts_dict))

    def _open_shortcuts_dialog(self):
        """Open keyboard shortcuts customization dialog."""
        from kindred.gui.widgets.shortcuts_dialog import ShortcutsDialog

        dialog = ShortcutsDialog(self)

        # Add shortcuts to dialog
        for action_name, data in self._shortcut_actions.items():
            dialog.add_shortcut(
                action_name,
                data["description"],
                data["action"],
                data["default"]
            )

        # Populate the table
        dialog.populate_table()

        # Show dialog
        if dialog.exec() == QtWidgets.QDialog.Accepted:
            # Save shortcuts to QSettings
            shortcuts_dict = dialog.get_shortcuts_dict()
            self.config_controller.persist_keyboard_shortcuts(shortcuts_dict)
            logger.info("Keyboard shortcuts updated")
            self._status_label.setText("Keyboard shortcuts updated")

    def _open_species_registry(self):
        """Open the Species Registry dialog."""
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Species Registry")
        dialog.setMinimumSize(420, 360)

        layout = QtWidgets.QVBoxLayout(dialog)

        # Info label
        info_label = QtWidgets.QLabel(
            "Species are automatically detected from the current mechanism DSL.\n"
            "Use this view to confirm initials before running simulations."
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # Summary label updated after detection
        summary_label = QtWidgets.QLabel()
        summary_label.setObjectName("speciesRegistrySummary")
        layout.addWidget(summary_label)

        # Species list
        layout.addWidget(QtWidgets.QLabel("Detected Species:"))
        species_list = QtWidgets.QListWidget()
        species_list.setObjectName("speciesRegistryList")
        layout.addWidget(species_list, stretch=1)

        entries, error_message = self._gather_species_registry_entries()
        if entries:
            summary_label.setText(f"Detected {len(entries)} species")
            for index, (name, initial) in enumerate(entries, start=1):
                prec = f"{initial:.6g}" if isinstance(initial, (int, float)) else str(initial)
                species_list.addItem(f"{index}. {name} (initial={prec})")
        else:
            placeholder = error_message or "No species detected. Add reactions to the mechanism editor."
            summary_label.setText(placeholder)
            species_list.addItem(placeholder)

        # Close button
        button_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)

        dialog.exec()

    def _gather_species_registry_entries(self) -> Tuple[List[Tuple[str, float]], Optional[str]]:
        """
        Parse the mechanism DSL and return detected species entries.

        Returns
        -------
        tuple
            ([(name, initial_conc), ...], error_message_or_None)
        """
        from kindred.core.batch_initial_conditions import (
            strip_named_reaction_dsl_initial_concentration_sets,
        )

        text = self._mechanism_editor._reactions_text.toPlainText()
        if not text.strip():
            logger.info("Species registry requested with empty mechanism editor")
            return [], "Mechanism editor is empty. Enter reactions to detect species."
        try:
            parse_text = strip_named_reaction_dsl_initial_concentration_sets(text)
        except Exception as exc:
            logger.warning("Failed to preprocess DSL for species registry: %s", exc, exc_info=True)
            return [], f"DSL parse error: {exc}"

        algebra_guard = False
        invalid_line: Optional[str] = None
        for raw in parse_text.splitlines():
            stripped = raw.strip()
            if not stripped:
                continue
            lower = stripped.lower()
            if lower.startswith("# algebra"):
                algebra_guard = True
                continue
            if stripped.startswith("#"):
                if algebra_guard:
                    algebra_guard = False
                continue
            if algebra_guard:
                # Allow arbitrary algebra expressions
                continue
            if lower.startswith(
                (
                    "reaction:",
                    "equilibrium:",
                    "state:",
                    "edge:",
                    "comp:",
                    "energy=",
                    "t=",
                    "c0=",
                    "c°=",
                    "κ=",
                    "kappa=",
                    "init:",
                    "initial:",
                    "temp_const:",
                    "temp_step:",
                    "temp_response:",
                )
            ):
                continue
            if stripped.startswith("[") and "=" in stripped:
                continue
            if "->" in stripped or "<->" in stripped or "<=>" in stripped:
                continue
            invalid_line = stripped
            break

        if invalid_line:
            message = f"DSL parse error: unrecognized line: '{invalid_line}'"
            logger.warning(message)
            return [], message

        try:
            from kindred.core.simulator.dsl import parse_dsl_to_mechanism
            from kindred.core.units import UnitsModel
        except Exception as exc:  # pragma: no cover - import typically succeeds
            logger.error("Species registry unavailable: %s", exc, exc_info=True)
            return [], "DSL parser is unavailable. See logs for details."

        try:
            units = UnitsModel(
                temperature_K=self._temperature_spinbox.value(),
            )
            mechanism = parse_dsl_to_mechanism(parse_text, initials={}, units=units)
        except Exception as exc:
            logger.warning("Failed to parse DSL for species registry: %s", exc, exc_info=True)
            return [], f"DSL parse error: {exc}"

        entries: List[Tuple[str, float]] = []
        for name, species in mechanism.species.items():
            initial = getattr(species, "initial_conc", 0.0)
            try:
                entries.append((name, float(initial)))
            except Exception:
                entries.append((name, initial))

        return entries, None

    def _open_state_network(self):
        """Open the State Network Editor dialog."""
        logger.info("Opening State Network Editor dialog")

        existing = getattr(self, "_state_network_dialog", None)
        if existing is not None:
            try:
                if existing.isVisible():
                    existing.raise_()
                    existing.activateWindow()
                    return
            except RuntimeError as exc:
                logger.debug("State network dialog reference was invalid: %s", exc, exc_info=True)
                if getattr(self, "_state_network_dialog", None) is existing:
                    self._state_network_dialog = None

        dialog = QtWidgets.QDialog(self)
        self._state_network_dialog = dialog
        dialog.setWindowTitle("State Network Editor")
        dialog.setMinimumSize(700, 550)

        layout = QtWidgets.QVBoxLayout(dialog)

        locked = self.mechanism_editing_locked()
        info_label = QtWidgets.QLabel(self._state_network_dialog_info_text(locked=locked))
        info_label.setObjectName("stateNetworkDialogInfoLabel")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        base_editor = self._mechanism_editor._state_network_editor
        set_read_only = getattr(base_editor, "set_read_only", None)
        if callable(set_read_only):
            set_read_only(locked)
        previous_parent = base_editor.parent()
        previous_visibility = base_editor.isVisible()

        # Re-parent the editor into the dialog for live editing
        base_editor.setParent(dialog)
        layout.addWidget(base_editor, stretch=1)
        base_editor.show()
        logger.info(
            "State Network Editor widget shown (rows=%s edges=%s)",
            base_editor._states_table.rowCount(),
            base_editor._edges_table.rowCount(),
        )
        if os.environ.get("KINDRED_DEBUG_STATE_NET"):
            def _rss_kb() -> Optional[int]:
                if platform.system() != "Linux":
                    return None
                status_path = Path(os.sep) / "proc" / "self" / "status"
                try:
                    with status_path.open("r", encoding="utf-8") as handle:
                        for line in handle:
                            if line.startswith("VmRSS:"):
                                parts = line.split()
                                if len(parts) >= 2:
                                    return int(parts[1])
                except Exception:
                    return None
                return None

            def _log_state_net_diag(tag: str) -> None:
                try:
                    widgets_total = len(base_editor.findChildren(QtWidgets.QWidget))
                    combos_total = len(base_editor.findChildren(QtWidgets.QComboBox))
                except Exception:
                    widgets_total = -1
                    combos_total = -1
                try:
                    states_items_est = base_editor._states_table.rowCount() * base_editor._states_table.columnCount()
                except Exception:
                    states_items_est = -1
                try:
                    edges_items_est = base_editor._edges_table.rowCount() * base_editor._edges_table.columnCount()
                except Exception:
                    edges_items_est = -1

                logger.info(
                    "StateNetworkEditor diag (%s): rss_kb=%s widgets=%s combobox=%s items~=(states=%s edges=%s)",
                    tag,
                    _rss_kb(),
                    widgets_total,
                    combos_total,
                    states_items_est,
                    edges_items_est,
                )

            def _diag_1000ms() -> None:
                _log_state_net_diag("t=1000ms")

            def _diag_250ms() -> None:
                _log_state_net_diag("t=250ms")
                QtCore.QTimer.singleShot(750, _diag_1000ms)

            def _diag_0ms() -> None:
                _log_state_net_diag("t=0ms")
                QtCore.QTimer.singleShot(250, _diag_250ms)

            QtCore.QTimer.singleShot(0, _diag_0ms)

        # Button box
        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)

        def _restore_state_network_editor(result: int) -> None:
            restore_failed = False
            logger.info(
                "State Network Editor dialog closed (result=%s)",
                "accepted" if int(result) == int(QtWidgets.QDialog.Accepted) else "rejected",
            )

            # Restore editor to its original owner so the mechanism tab stays in sync
            try:
                layout.removeWidget(base_editor)
            except RuntimeError as exc:
                logger.debug("Failed to detach state network editor from dialog layout: %s", exc, exc_info=True)
                restore_failed = True
            try:
                base_editor.setParent(previous_parent)
            except RuntimeError as exc:
                logger.debug("Failed to restore state network editor parent: %s", exc, exc_info=True)
                restore_failed = True
            try:
                base_editor.setVisible(bool(previous_visibility))
            except RuntimeError as exc:
                logger.debug("Failed to restore state network editor visibility: %s", exc, exc_info=True)
                restore_failed = True

            if int(result) == int(QtWidgets.QDialog.Accepted):
                self._status_label.setText("State network updated")

            if getattr(self, "_state_network_dialog", None) is dialog:
                self._state_network_dialog = None
            try:
                dialog.deleteLater()
            except RuntimeError as exc:
                logger.debug("Failed to schedule state network dialog deletion: %s", exc, exc_info=True)
                restore_failed = True
            if restore_failed:
                self._state_network_editor_restore_failed = True

        dialog.finished.connect(_restore_state_network_editor)
        dialog.setWindowModality(Qt.ApplicationModal)
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _open_computational_mode(self) -> None:
        """Open the Computational Mode dialog."""
        logger.info("Opening Computational Mode dialog")

        existing = getattr(self, "_computational_mode_dialog", None)
        if existing is not None:
            try:
                if existing.isVisible():
                    existing.raise_()
                    existing.activateWindow()
                    return
            except RuntimeError as exc:
                logger.debug("Computational mode dialog reference was invalid: %s", exc, exc_info=True)
                if getattr(self, "_computational_mode_dialog", None) is existing:
                    self._computational_mode_dialog = None

        try:
            from kindred.gui.widgets.computational_mode_dialog import ComputationalModeDialog
        except Exception as exc:  # pragma: no cover - import typically succeeds
            logger.error("Failed to import Computational Mode dialog: %s", exc, exc_info=True)
            QtWidgets.QMessageBox.warning(
                self,
                "Computational Mode Unavailable",
                "Computational Mode dialog could not be loaded. See logs for details.",
            )
            return

        dialog = ComputationalModeDialog(self)
        self._computational_mode_dialog = dialog

        def _cleanup(_result: int) -> None:
            if getattr(self, "_computational_mode_dialog", None) is dialog:
                self._computational_mode_dialog = None
            try:
                dialog.deleteLater()
            except RuntimeError as exc:
                logger.debug("Failed to schedule computational mode dialog deletion: %s", exc, exc_info=True)
                self._computational_mode_dialog_delete_failed = True

        dialog.finished.connect(_cleanup)
        dialog.setWindowModality(Qt.ApplicationModal)
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _undo(self):
        """
        Undo the last action.

        Tries high-level undo stack first (for operations like Load Preset),
        then falls back to text editor undo for character-level edits.
        """
        if self._next_undo_redo_targets_locked_mechanism_change(redo=False):
            self._report_locked_reactions_undo_redo_block(redo=False)
            return

        # Try high-level undo stack first
        if self._undo_stack.canUndo():
            self._undo_stack.undo()
            self._status_label.setText(f"Undo: {self._undo_stack.undoText()}")
            logger.debug(f"Undo: {self._undo_stack.undoText()}")
            return

        # Fall back to text editor undo
        focused_widget = QtWidgets.QApplication.focusWidget()
        if isinstance(focused_widget, (QtWidgets.QPlainTextEdit, QtWidgets.QTextEdit)):
            if self.mechanism_editing_locked() and self._focused_widget_targets_reactions_text(focused_widget):
                self._report_locked_reactions_undo_redo_block(redo=False)
                return
            if focused_widget.document().isUndoAvailable():
                focused_widget.undo()
                self._status_label.setText("Undo (text edit)")
                logger.debug("Undo action performed (text editor)")
            else:
                self._status_label.setText("Nothing to undo")
        else:
            self._status_label.setText("Nothing to undo")

    def _redo(self):
        """
        Redo the last undone action.

        Tries high-level undo stack first (for operations like Load Preset),
        then falls back to text editor redo for character-level edits.
        """
        if self._next_undo_redo_targets_locked_mechanism_change(redo=True):
            self._report_locked_reactions_undo_redo_block(redo=True)
            return

        # Try high-level undo stack first
        if self._undo_stack.canRedo():
            self._undo_stack.redo()
            self._status_label.setText(f"Redo: {self._undo_stack.redoText()}")
            logger.debug(f"Redo: {self._undo_stack.redoText()}")
            return

        # Fall back to text editor redo
        focused_widget = QtWidgets.QApplication.focusWidget()
        if isinstance(focused_widget, (QtWidgets.QPlainTextEdit, QtWidgets.QTextEdit)):
            if self.mechanism_editing_locked() and self._focused_widget_targets_reactions_text(focused_widget):
                self._report_locked_reactions_undo_redo_block(redo=True)
                return
            if focused_widget.document().isRedoAvailable():
                focused_widget.redo()
                self._status_label.setText("Redo (text edit)")
                logger.debug("Redo action performed (text editor)")
            else:
                self._status_label.setText("Nothing to redo")
        else:
            self._status_label.setText("Nothing to redo")

    def _show_panel_layout_tips(self) -> None:
        QtWidgets.QMessageBox.information(
            self,
            "Panel Layout Tips",
            "Use View > Panels to show or hide the Mechanism, Interactive Sliders, Batch Initial Conditions, Data, and Analysis panels.\n\n"
            "To place panels together on the same side, drag a panel by its title bar and pause over an occupied "
            "dock area until Qt shows an inner drop guide. Dropping there lets that side share space with the "
            "existing panel.\n\n"
            "If a panel is floating, Dock Back returns it to its last docked side.\n\n"
            "Reset Layout restores all panels to the default workspace layout.",
        )

    def _restore_maximized_state_if_needed(self) -> None:
        if not self.isMaximized():
            self.showMaximized()

    def restore_persisted_maximized_state(self) -> None:
        if self.isVisible():
            self._restore_maximized_state_if_needed()
            return
        # Preserve the construct-then-show startup contract by recording the
        # maximized window state without forcing the hidden window visible.
        self.setWindowState(self.windowState() | Qt.WindowMaximized)

    def _run_preserving_maximized_state(self, operation: Callable[[], None]) -> None:
        should_restore_maximized = self.isMaximized()
        operation()
        if should_restore_maximized:
            self._restore_maximized_state_if_needed()
            QtCore.QTimer.singleShot(0, self._restore_maximized_state_if_needed)

    def _default_dock_area(self, dock: QtWidgets.QDockWidget) -> Qt.DockWidgetArea:
        area = dock_default_area(dock)
        if area != Qt.NoDockWidgetArea:
            return area
        return self.dockWidgetArea(dock)

    def _default_dock_layout(self) -> tuple[tuple[QtWidgets.QDockWidget, Qt.DockWidgetArea], ...]:
        return tuple(
            (getattr(self, spec.attr_name), self._default_dock_area(getattr(self, spec.attr_name)))
            for spec in dock_shell_specs()
        )

    def _shell_docks(self) -> tuple[QtWidgets.QDockWidget, ...]:
        return tuple(getattr(self, spec.attr_name) for spec in dock_shell_specs())

    def _apply_default_dock_arrangement(self) -> None:
        mechanism_dock = getattr(self, "_mechanism_dock", None)
        sliders_dock = getattr(self, "_sliders_dock", None)
        batch_dock = getattr(self, "_batch_dock", None)
        data_dock = getattr(self, "_right_dock", None)
        analysis_dock = getattr(self, "_analysis_dock", None)
        if not all(dock is not None for dock in (mechanism_dock, sliders_dock, batch_dock, data_dock, analysis_dock)):
            return

        # Left stack order: Mechanism, Interactive Sliders, Batch Initial Conditions.
        self.splitDockWidget(mechanism_dock, batch_dock, Qt.Vertical)
        self.splitDockWidget(mechanism_dock, sliders_dock, Qt.Vertical)

        # Right stack order: Data above Analysis.
        self.splitDockWidget(data_dock, analysis_dock, Qt.Vertical)

    def schedule_restored_floating_dock_recovery(self) -> None:
        if bool(getattr(self, "_restored_floating_dock_recovery_pending", False)):
            return
        self._restored_floating_dock_recovery_pending = True
        QtCore.QTimer.singleShot(0, self._recover_restored_floating_docks)

    def _recover_restored_floating_docks(self) -> None:
        self._restored_floating_dock_recovery_pending = False
        for dock in self._shell_docks():
            if not dock.isFloating():
                continue
            if not self._is_restored_floating_dock_unsafe(dock):
                continue
            rect = dock.frameGeometry()
            logger.warning(
                "Recovered unsafe floating dock restore for %s at %s",
                dock.objectName() or dock.windowTitle(),
                rect.getRect(),
            )
            self._redock_restored_floating_dock(dock)

    def _restored_floating_dock_minimum_size(self, dock: QtWidgets.QDockWidget) -> QtCore.QSize:
        minimum = dock.minimumSizeHint().expandedTo(dock.minimumSize())
        return QtCore.QSize(max(1, minimum.width()), max(1, minimum.height()))

    def _is_restored_floating_dock_unsafe(self, dock: QtWidgets.QDockWidget) -> bool:
        if not dock.isFloating():
            return False

        rect = dock.frameGeometry()
        if not rect.isValid():
            rect = dock.geometry()
        if not rect.isValid():
            return True
        minimum = self._restored_floating_dock_minimum_size(dock)
        if rect.width() < minimum.width() or rect.height() < minimum.height():
            return True

        available_geometries = tuple(
            screen.availableGeometry()
            for screen in QtGui.QGuiApplication.screens()
            if screen is not None and screen.availableGeometry().isValid()
        )
        if not available_geometries:
            return False
        return not any(rect.intersects(screen_rect) for screen_rect in available_geometries)

    def _redock_restored_floating_dock(self, dock: QtWidgets.QDockWidget) -> None:
        def _redock() -> None:
            was_hidden = dock.isHidden()
            if was_hidden:
                dock.show()
            dock.setFloating(False)
            if dock.isFloating():
                self.removeDockWidget(dock)
                self.addDockWidget(self._default_dock_area(dock), dock)
                dock.setFloating(False)
            if was_hidden:
                dock.hide()

        self._run_preserving_maximized_state(_redock)

    def _reset_layout(self):
        """Reset dock layout to the shell's default dock arrangement."""
        def _reset() -> None:
            dock_defaults = self._default_dock_layout()

            for dock, _area in dock_defaults:
                dock.setVisible(True)
                dock.setFloating(False)

            # Remove all docks and re-add them to default positions.
            for dock, _area in dock_defaults:
                self.removeDockWidget(dock)

            for dock, area in dock_defaults:
                self.addDockWidget(area, dock)
                dock.show()
                dock.raise_()

            self._apply_default_dock_arrangement()

            logger.info("Layout reset to default")

        self._run_preserving_maximized_state(_reset)

    def redock_shell_dock(self, dock: QtWidgets.QDockWidget) -> None:
        self._run_preserving_maximized_state(lambda: dock.setFloating(False))

    def _available_analysis_surfaces(self) -> set[str]:
        plot_tabs = getattr(self, "_plot_tabs", None)
        if plot_tabs is None or not hasattr(plot_tabs, "available_analysis_surfaces"):
            return set()
        try:
            names = plot_tabs.available_analysis_surfaces()
        except Exception as exc:
            logger.debug("Failed to query available analysis surfaces: %s", exc, exc_info=True)
            return set()
        return {str(name) for name in (names or []) if str(name)}

    def _refresh_analysis_surface_actions(self) -> None:
        actions = getattr(self, "_analysis_surface_actions", None)
        if not isinstance(actions, dict):
            return
        available = self._available_analysis_surfaces()
        for surface_name, action in actions.items():
            action.setEnabled(str(surface_name) in available)

    def _focus_analysis_dock(self) -> None:
        dock = getattr(self, "_analysis_dock", None)
        if dock is None:
            return
        dock.show()
        dock.raise_()
        if dock.isFloating():
            dock.activateWindow()

    def _show_analysis_surface(self, surface_name: str) -> bool:
        surface = str(surface_name).strip()
        if not surface:
            return False
        available = self._available_analysis_surfaces()
        if surface not in available:
            self._refresh_analysis_surface_actions()
            self.set_status_text(f"{surface} surface unavailable for the current plot backend")
            return False

        plot_tabs = getattr(self, "_plot_tabs", None)
        if plot_tabs is None or not hasattr(plot_tabs, "focus_analysis_surface"):
            self.set_status_text(f"{surface} surface unavailable for the current plot backend")
            return False

        if not bool(plot_tabs.focus_analysis_surface(surface)):
            self._refresh_analysis_surface_actions()
            self.set_status_text(f"Unable to show {surface} surface")
            return False

        self._analysis_dock.setVisible(True)
        self._focus_analysis_dock()
        self.set_status_text(f"{surface} surface ready")
        return True

    def _toggle_theme(self):
        """Toggle between dark and light themes."""
        self.config_controller.toggle_theme()

    def _get_mechanism_text(self) -> str:
        """Get the current mechanism DSL text from editor."""
        reactions_text = self._mechanism_editor._reactions_text.toPlainText()
        state_network_dsl = self._mechanism_editor._state_network_editor.get_state_network_dsl()
        if self.has_slider_overrides():
            state_network_dsl = self._apply_overrides_to_state_network_dsl(state_network_dsl)

        # Combine DSL texts
        full_dsl = reactions_text
        if state_network_dsl.strip():
            full_dsl += "\n\n# State Network\n" + state_network_dsl

        return full_dsl

    def _remember_last_mechanism(
        self,
        mechanism: Mechanism,
        dsl_text: str,
        solver_config: Dict[str, Any],
    ) -> None:
        """
        Cache the most recent successfully simulated mechanism for downstream exports.
        """
        self._mechanism_helpers.remember_last_mechanism(mechanism, dsl_text, solver_config)

    def _clear_last_mechanism(self) -> None:
        """Drop cached mechanism so exports cannot use stale results."""
        self._mechanism_helpers.clear_last_mechanism()

    def _clear_main_plot_display_state(self) -> None:
        """Clear the visible simulation plot while leaving mechanism controls untouched."""
        plot = self.main_plot()
        setattr(plot, "_workspace_preview_display_provenance_by_set_id", {})
        if hasattr(plot, "clear"):
            plot.clear()
        if hasattr(plot, "set_statistics_results"):
            plot.set_statistics_results({}, prefer="")

    def _clear_main_plot_project_apply_state(self) -> None:
        """Clear simulation-plot state that is not serialized into project files."""
        self._clear_main_plot_display_state()

    def _clear_batch_selection_display_state(self) -> None:
        """Drop the active displayed batch selection without discarding cache ownership state."""
        batch_cache = getattr(self._sim_controller, "batch_cache", None)
        if batch_cache is not None:
            clear_display = getattr(batch_cache, "clear_display_selection_state", None)
            if callable(clear_display):
                clear_display()
            else:
                batch_cache.last_display_selection = []
                batch_cache.active_batch_set = None
                batch_cache.active_batch_set_id = None
        self._clear_main_plot_display_state()
        self.show_simulation_tab()
        self.refresh_simulation_plot_views()

    def _batch_cache_entry_matches_plot_payload(
        self,
        *,
        entry: Optional[Mapping[str, Any]],
        t: np.ndarray,
        series: Mapping[str, Any],
    ) -> bool:
        if not isinstance(entry, Mapping):
            return False
        entry_t_payload = entry.get("t")
        entry_t = np.asarray(entry_t_payload if entry_t_payload is not None else [], dtype=float).reshape(-1)
        plot_t = np.asarray(t, dtype=float).reshape(-1)
        if entry_t.size <= 0 or entry_t.shape != plot_t.shape:
            return False
        if not np.allclose(entry_t, plot_t, rtol=1e-9, atol=1e-12):
            return False
        entry_series_raw = entry.get("series") or {}
        if not isinstance(entry_series_raw, Mapping):
            return False
        plot_series = {
            str(species_name): np.asarray(values, dtype=float).reshape(-1)
            for species_name, values in dict(series or {}).items()
        }
        entry_series = {
            str(species_name): np.asarray(values, dtype=float).reshape(-1)
            for species_name, values in dict(entry_series_raw).items()
        }
        if set(entry_series.keys()) != set(plot_series.keys()):
            return False
        for species_name, plot_values in plot_series.items():
            entry_values = entry_series.get(str(species_name))
            if entry_values is None or entry_values.shape != plot_values.shape:
                return False
            if not np.allclose(entry_values, plot_values, rtol=1e-9, atol=1e-12):
                return False
        return True

    def _active_explicit_cache_entry_for_set(self, *, set_id: str) -> BatchCacheEntryReadResult:
        batch_cache = getattr(getattr(self, "_sim_controller", None), "batch_cache", None)
        active_cache_key = str(getattr(batch_cache, "active_cache_key", "") or "").strip()
        if batch_cache is None or not active_cache_key:
            return BatchCacheEntryReadResult("missing")
        sid = str(set_id or "").strip()
        if not sid:
            return BatchCacheEntryReadResult("missing")
        store_data = getattr(batch_cache.result_cache, "_data", batch_cache.result_cache)
        direct = read_batch_cache_entry(
            (store_data or {}).get(BatchSimulationCache.entry_key(active_cache_key, sid))
        )
        if direct.entry is not None:
            return direct
        set_name = self.batch_set_name_for_id(sid)
        by_name = BatchCacheEntryReadResult("missing")
        if set_name:
            by_name = read_batch_cache_entry(
                (store_data or {}).get(BatchSimulationCache.entry_key(active_cache_key, str(set_name)))
            )
            if by_name.entry is not None:
                return by_name
        if direct.state == "invalid" or by_name.state == "invalid":
            return BatchCacheEntryReadResult("invalid")
        return BatchCacheEntryReadResult("missing")

    def _current_workspace_preview_identity_payload(self, *, set_id: str) -> Optional[Dict[str, Any]]:
        sid = str(set_id or "").strip()
        if not sid:
            return None
        try:
            identity = self._current_workspace_preview_identity(set_id=sid)
        except Exception:
            return None
        try:
            return dict(identity.to_payload())
        except Exception:
            return None

    def _main_plot_workspace_preview_provenance(self) -> Dict[str, Dict[str, Any]]:
        plot = getattr(getattr(self, "_plot_tabs", None), "_main_plot", None)
        raw = getattr(plot, "_workspace_preview_display_provenance_by_set_id", None) if plot is not None else None
        if not isinstance(raw, Mapping):
            return {}
        cleaned: Dict[str, Dict[str, Any]] = {}
        for raw_set_id, raw_payload in dict(raw).items():
            set_id = str(raw_set_id or "").strip()
            if not set_id or not isinstance(raw_payload, Mapping):
                continue
            cleaned[set_id] = dict(raw_payload)
        return cleaned

    def _set_main_plot_workspace_preview_provenance(
        self,
        provenance_by_set_id: Mapping[str, Mapping[str, Any]],
    ) -> None:
        plot = getattr(getattr(self, "_plot_tabs", None), "_main_plot", None)
        if plot is None:
            return
        cleaned: Dict[str, Dict[str, Any]] = {}
        for raw_set_id, raw_payload in dict(provenance_by_set_id or {}).items():
            set_id = str(raw_set_id or "").strip()
            if not set_id or not isinstance(raw_payload, Mapping):
                continue
            cleaned[set_id] = dict(raw_payload)
        setattr(plot, "_workspace_preview_display_provenance_by_set_id", cleaned)

    def _displayed_workspace_preview_provenance_matches_current_workspace(self, *, set_id: str) -> bool:
        sid = str(set_id or "").strip()
        if not sid:
            return False
        current_payload = self._current_workspace_preview_identity_payload(set_id=sid)
        if not isinstance(current_payload, dict):
            return False
        stored_payload = self._main_plot_workspace_preview_provenance().get(sid)
        return isinstance(stored_payload, dict) and stored_payload == current_payload

    def _record_current_main_plot_workspace_preview_provenance(
        self,
        *,
        selected_set_ids: Sequence[str],
    ) -> None:
        selected_ids = [str(set_id) for set_id in (selected_set_ids or ()) if str(set_id)]
        plot = getattr(getattr(self, "_plot_tabs", None), "_main_plot", None)
        if plot is None or not selected_ids:
            self._set_main_plot_workspace_preview_provenance({})
            return

        active_set_id = str(self.active_batch_selection()[0] or "").strip()
        if (not active_set_id) and selected_ids:
            active_set_id = selected_ids[0]
        if not active_set_id:
            self._set_main_plot_workspace_preview_provenance({})
            return

        current_t_raw = getattr(plot, "_t", None)
        current_t = np.asarray(current_t_raw if current_t_raw is not None else [], dtype=float).reshape(-1)
        current_series = dict(getattr(plot, "_series", {}) or {})
        if current_t.size <= 0 or not current_series:
            self._set_main_plot_workspace_preview_provenance({})
            return

        selected_local_workspace_ids = {
            set_id for set_id in selected_ids if self._preview_session.has_local_mechanism_workspace(set_id)
        }
        selected_overlay_dirty_ids: set[str] = set()
        for set_id in selected_ids:
            try:
                row = getattr(self, "_batch_store", None).row_for_set_id(str(set_id))
            except Exception:
                row = None
            if row is not None:
                try:
                    if bool(self._preview_session.preview_batch_cache_token([int(row)])):
                        selected_overlay_dirty_ids.add(str(set_id))
                except Exception:
                    continue

        selected_dirty_overlay_ids = {
            str(set_id)
            for set_id in selected_ids
            if str(set_id)
            and (
                str(set_id) in selected_local_workspace_ids
                or str(set_id) in selected_overlay_dirty_ids
            )
        }
        provenance_by_set_id: Dict[str, Dict[str, Any]] = {}
        active_requires_truthful_dirty_preview = bool(
            active_set_id in selected_local_workspace_ids or active_set_id in selected_overlay_dirty_ids
        )
        if active_requires_truthful_dirty_preview:
            active_preview_entry = self._matching_preview_entry_for_workspace_set(set_id=active_set_id)
            if self._batch_cache_entry_matches_plot_payload(
                entry=active_preview_entry.entry,
                t=current_t,
                series=current_series,
            ):
                active_payload = self._current_workspace_preview_identity_payload(set_id=active_set_id)
                if isinstance(active_payload, dict):
                    provenance_by_set_id[active_set_id] = active_payload

        overlay_label_to_set_id: Dict[str, str] = {}
        for set_id in selected_ids:
            set_id_s = str(set_id or "").strip()
            if not set_id_s:
                continue
            overlay_label_to_set_id[set_id_s] = set_id_s
            set_name = str(self.batch_set_name_for_id(set_id_s) or "").strip()
            if set_name:
                overlay_label_to_set_id[set_name] = set_id_s
        for entry in list(getattr(plot, "_simulation_overlays", []) or []):
            if not isinstance(entry, dict):
                continue
            overlay_label = str(entry.get("label") or "").strip()
            overlay_set_id = str(entry.get("set_id") or "").strip() or overlay_label_to_set_id.get(overlay_label, "")
            if not overlay_set_id or overlay_set_id not in selected_dirty_overlay_ids:
                continue
            overlay_t = np.asarray(entry.get("t") if entry.get("t") is not None else [], dtype=float).reshape(-1)
            overlay_series_raw = entry.get("series") or {}
            if overlay_t.size <= 0 or not isinstance(overlay_series_raw, dict):
                continue
            overlay_series: Dict[str, np.ndarray] = {}
            for species_name, values in overlay_series_raw.items():
                overlay_arr = np.asarray(values, dtype=float).reshape(-1)
                if overlay_arr.size <= 0:
                    continue
                overlay_series[str(species_name)] = overlay_arr
            if not overlay_series:
                continue
            overlay_preview_entry = self._matching_preview_entry_for_workspace_set(set_id=overlay_set_id)
            if not self._batch_cache_entry_matches_plot_payload(
                entry=overlay_preview_entry.entry,
                t=overlay_t,
                series=overlay_series,
            ):
                continue
            overlay_payload = self._current_workspace_preview_identity_payload(set_id=overlay_set_id)
            if isinstance(overlay_payload, dict):
                provenance_by_set_id[overlay_set_id] = overlay_payload

        self._set_main_plot_workspace_preview_provenance(provenance_by_set_id)

    def _active_workspace_preview_display_snapshot(self) -> Optional[Dict[str, Any]]:
        batch_cache = getattr(self._sim_controller, "batch_cache", None)
        if batch_cache is None or not self.main_plot_has_data():
            return None

        active_set_id = str(batch_cache.active_batch_set_id or "").strip()
        if not active_set_id:
            return None

        selected_ids = [str(set_id) for set_id in (batch_cache.last_display_selection or []) if str(set_id)]
        if not selected_ids:
            selected_ids = [str(set_id) for set_id in (self._shown_batch_set_ids() or []) if str(set_id)]
        if active_set_id not in selected_ids:
            selected_ids = [active_set_id, *[set_id for set_id in selected_ids if set_id != active_set_id]]

        selected_local_workspace_ids = {
            set_id for set_id in selected_ids if self._preview_session.has_local_mechanism_workspace(set_id)
        }
        selected_overlay_dirty_ids: set[str] = set()
        row_for_set_id = getattr(getattr(self, "_batch_store", None), "row_for_set_id", None)
        if callable(row_for_set_id):
            for set_id in selected_ids:
                try:
                    row = row_for_set_id(str(set_id))
                except Exception:
                    row = None
                if row is None:
                    continue
                try:
                    if bool(self._preview_session.preview_batch_cache_token([int(row)])):
                        selected_overlay_dirty_ids.add(str(set_id))
                except Exception:
                    continue

        active_has_local_mechanism_workspace = active_set_id in selected_local_workspace_ids
        active_has_dirty_overlay = active_set_id in selected_overlay_dirty_ids
        if not (active_has_local_mechanism_workspace or selected_overlay_dirty_ids):
            return None

        plot = getattr(getattr(self, "_plot_tabs", None), "_main_plot", None)
        if plot is None:
            return None

        run_state = getattr(getattr(self, "_sim_controller", None), "run_state", None)
        if bool(getattr(run_state, "pending_slider_simulation", False)):
            return None

        current_t_raw = getattr(plot, "_t", None)
        current_t = np.asarray(current_t_raw if current_t_raw is not None else [], dtype=float).reshape(-1)
        if current_t.size <= 0:
            return None

        current_series = dict(getattr(plot, "_series", {}) or {})
        if not current_series:
            return None

        active_plot_is_truthful_dirty_preview = False
        active_requires_truthful_dirty_preview = (
            active_has_local_mechanism_workspace or active_has_dirty_overlay
        )
        if active_requires_truthful_dirty_preview:
            active_preview_entry = self._matching_preview_entry_for_workspace_set(set_id=active_set_id)
            active_plot_is_truthful_dirty_preview = self._batch_cache_entry_matches_plot_payload(
                entry=active_preview_entry.entry,
                t=current_t,
                series=current_series,
            )
            if not active_plot_is_truthful_dirty_preview:
                active_plot_is_truthful_dirty_preview = self._displayed_workspace_preview_provenance_matches_current_workspace(
                    set_id=active_set_id,
                )

        selected_dirty_overlay_ids = {
            str(set_id)
            for set_id in selected_ids
            if str(set_id)
            and str(set_id) != active_set_id
            and (
                str(set_id) in selected_local_workspace_ids
                or str(set_id) in selected_overlay_dirty_ids
            )
        }
        preserved_overlays: list[Dict[str, object]] = []
        if selected_ids and selected_dirty_overlay_ids:
            overlay_label_to_set_id: Dict[str, str] = {}
            truthful_preserved_preview_set_ids: set[str] = set()
            for set_id in selected_ids:
                set_id_s = str(set_id or "").strip()
                if not set_id_s:
                    continue
                overlay_label_to_set_id[set_id_s] = set_id_s
                set_name = str(self.batch_set_name_for_id(set_id_s) or "").strip()
                if set_name:
                    overlay_label_to_set_id[set_name] = set_id_s
            for entry in list(getattr(plot, "_simulation_overlays", []) or []):
                if not isinstance(entry, dict):
                    continue
                overlay_label = str(entry.get("label") or "").strip()
                overlay_set_id = str(entry.get("set_id") or "").strip() or overlay_label_to_set_id.get(overlay_label, "")
                overlay_curve_role = str(entry.get("curve_role") or "").strip()
                if not overlay_label or not overlay_set_id or overlay_set_id not in selected_dirty_overlay_ids:
                    continue
                overlay_t = np.asarray(entry.get("t") if entry.get("t") is not None else [], dtype=float).reshape(-1)
                overlay_series_raw = entry.get("series") or {}
                if overlay_t.size <= 0 or not isinstance(overlay_series_raw, dict):
                    continue
                overlay_series: Dict[str, np.ndarray] = {}
                for species_name, values in overlay_series_raw.items():
                    overlay_arr = np.asarray(values, dtype=float).reshape(-1)
                    if overlay_arr.size <= 0:
                        continue
                    overlay_series[str(species_name)] = overlay_arr
                if not overlay_series:
                    continue
                explicit_overlay_entry = self._active_explicit_cache_entry_for_set(set_id=overlay_set_id)
                overlay_matches_explicit = self._batch_cache_entry_matches_plot_payload(
                    entry=explicit_overlay_entry.entry,
                    t=overlay_t,
                    series=overlay_series,
                )
                if overlay_curve_role == "canonical_ghost":
                    if (
                        overlay_set_id == active_set_id
                        or not overlay_matches_explicit
                        or overlay_set_id not in truthful_preserved_preview_set_ids
                    ):
                        continue
                else:
                    overlay_preview_entry = self._matching_preview_entry_for_workspace_set(set_id=overlay_set_id)
                    overlay_is_truthful_dirty_preview = self._batch_cache_entry_matches_plot_payload(
                        entry=overlay_preview_entry.entry,
                        t=overlay_t,
                        series=overlay_series,
                    )
                    if not overlay_is_truthful_dirty_preview:
                        overlay_is_truthful_dirty_preview = self._displayed_workspace_preview_provenance_matches_current_workspace(
                            set_id=overlay_set_id,
                        )
                    if not overlay_is_truthful_dirty_preview or overlay_matches_explicit:
                        continue
                    truthful_preserved_preview_set_ids.add(overlay_set_id)
                preserved_entry = {
                    "label": overlay_label,
                    "t": overlay_t,
                    "series": overlay_series,
                    "set_id": overlay_set_id,
                }
                if overlay_curve_role:
                    preserved_entry["curve_role"] = overlay_curve_role
                preserved_overlays.append(preserved_entry)

        active_canonical_ghost: Dict[str, object] | None = None
        if active_plot_is_truthful_dirty_preview:
            explicit_active_entry = self._active_explicit_cache_entry_for_set(set_id=active_set_id)
            for entry in list(getattr(plot, "_simulation_overlays", []) or []):
                if not isinstance(entry, dict):
                    continue
                if str(entry.get("curve_role") or "").strip() != "canonical_ghost":
                    continue
                if str(entry.get("set_id") or "").strip() != active_set_id:
                    continue
                overlay_t = np.asarray(entry.get("t") if entry.get("t") is not None else [], dtype=float).reshape(-1)
                overlay_series_raw = entry.get("series") or {}
                if overlay_t.size <= 0 or not isinstance(overlay_series_raw, dict):
                    continue
                overlay_series: Dict[str, np.ndarray] = {}
                for species_name, values in overlay_series_raw.items():
                    overlay_arr = np.asarray(values, dtype=float).reshape(-1)
                    if overlay_arr.size <= 0:
                        continue
                    overlay_series[str(species_name)] = overlay_arr
                if not overlay_series:
                    continue
                if not self._batch_cache_entry_matches_plot_payload(
                    entry=explicit_active_entry.entry,
                    t=overlay_t,
                    series=overlay_series,
                ):
                    continue
                active_canonical_ghost = {
                    "label": str(entry.get("label") or "").strip(),
                    "t": overlay_t,
                    "series": overlay_series,
                    "set_id": active_set_id,
                    "curve_role": "canonical_ghost",
                }
                break

        if active_canonical_ghost is not None:
            preserved_overlays.append(active_canonical_ghost)

        if active_requires_truthful_dirty_preview and (not active_plot_is_truthful_dirty_preview):
            return None
        if (not active_plot_is_truthful_dirty_preview) and (not preserved_overlays):
            return None

        return {
            "set_id": active_set_id,
            "set_name": str(batch_cache.active_batch_set or self.batch_set_name_for_id(active_set_id) or active_set_id),
            "selected_ids": selected_ids,
            "preserved_overlays": preserved_overlays,
        }

    def _invalidate_active_results_after_authoritative_mechanism_change(
        self,
        *,
        preserve_current_display: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Drop stale displayed results after the authoritative mechanism changes."""
        self._clear_last_mechanism()
        self._sim_controller.invalidate_active_explicit_simulation_for_authoritative_change()
        self._sim_controller.invalidate_slider_preview_work()

        batch_cache = getattr(self._sim_controller, "batch_cache", None)
        if batch_cache is None:
            return

        active_cache_key = str(batch_cache.active_cache_key or "").strip()
        selected_ids = [str(set_id) for set_id in (self._shown_batch_set_ids() or []) if str(set_id)]
        if active_cache_key:
            fallback_scope = ()
            cached_scope_ids: list[str] = []
            try:
                for raw_key in batch_cache.result_cache:
                    key_s = str(raw_key or "")
                    prefix = f"{active_cache_key}::"
                    if not key_s.startswith(prefix):
                        continue
                    set_id = str(key_s[len(prefix):] or "").strip()
                    if set_id and set_id not in cached_scope_ids:
                        cached_scope_ids.append(set_id)
            except Exception:
                cached_scope_ids = []
            active_valid_ids = tuple(str(set_id) for set_id in (batch_cache.active_cache_valid_set_ids or ()) if str(set_id))
            if cached_scope_ids:
                fallback_scope = tuple(cached_scope_ids)
            elif active_valid_ids:
                fallback_scope = active_valid_ids
            elif selected_ids:
                fallback_scope = tuple(selected_ids)
            elif batch_cache.last_display_selection:
                fallback_scope = tuple(str(set_id) for set_id in (batch_cache.last_display_selection or []) if str(set_id))
            else:
                active_batch_set_id = str(batch_cache.active_batch_set_id or "").strip()
                fallback_scope = (active_batch_set_id,) if active_batch_set_id else ()
            batch_cache.active_cache_invalidated_set_ids = fallback_scope or None
            clear_display = getattr(batch_cache, "clear_display_selection_state", None)
            if callable(clear_display):
                clear_display()
            else:
                batch_cache.last_display_selection = []
                batch_cache.active_batch_set = None
                batch_cache.active_batch_set_id = None
        if preserve_current_display and self.main_plot_has_data():
            preserved_set_id = str(preserve_current_display.get("set_id") or "").strip()
            preserved_set_name = str(preserve_current_display.get("set_name") or "").strip()
            preserved_selected_ids = [
                str(set_id) for set_id in (preserve_current_display.get("selected_ids") or ()) if str(set_id)
            ]
            preserved_overlays = [
                dict(entry)
                for entry in (preserve_current_display.get("preserved_overlays") or ())
                if isinstance(entry, dict)
            ]
            plot = getattr(getattr(self, "_plot_tabs", None), "_main_plot", None)
            plot_t = np.asarray(getattr(plot, "_t", None) if plot is not None else [], dtype=float).reshape(-1)
            plot_series = dict(getattr(plot, "_series", {}) or {}) if plot is not None else {}
            plot_owned_species = getattr(plot, "_owned_species", None) if plot is not None else None
            plot_has_overlays = bool(getattr(plot, "_simulation_overlays", []) or []) if plot is not None else False
            preserve_multiselect_overlays = bool(preserved_overlays)
            if preserved_set_id:
                batch_cache.active_batch_set_id = preserved_set_id
                batch_cache.active_batch_set = preserved_set_name or str(
                    self.batch_set_name_for_id(preserved_set_id) or preserved_set_id
                )
                if preserve_multiselect_overlays and preserved_selected_ids:
                    batch_cache.last_display_selection = preserved_selected_ids
                else:
                    batch_cache.last_display_selection = [preserved_set_id]
            elif preserved_selected_ids:
                batch_cache.last_display_selection = preserved_selected_ids
            if plot is not None and plot_t.size > 0 and plot_series and (plot_has_overlays or preserve_multiselect_overlays):
                plot_label = preserved_set_name or preserved_set_id or "Results"
                plot.set_data(
                    plot_t,
                    plot_series,
                    label=plot_label,
                    overlays=preserved_overlays if preserve_multiselect_overlays else [],
                    owned_species=plot_owned_species,
                )
                self.sync_main_plot_copy_labels(
                    preserved_set_id,
                    preserved_selected_ids or ([preserved_set_id] if preserved_set_id else []),
                )
                plot_results_map: Dict[str, Dict[str, object]] = {
                    plot_label: {
                        "t": plot_t,
                        "series": plot_series,
                    }
                }
                for overlay_entry in preserved_overlays if preserve_multiselect_overlays else ():
                    overlay_label = str(overlay_entry.get("label") or "").strip()
                    if str(overlay_entry.get("curve_role") or "") == "canonical_ghost":
                        continue
                    overlay_t = np.asarray(
                        overlay_entry.get("t") if overlay_entry.get("t") is not None else [],
                        dtype=float,
                    ).reshape(-1)
                    overlay_series = dict(overlay_entry.get("series") or {})
                    if not overlay_label or overlay_t.size <= 0 or not overlay_series:
                        continue
                    plot_results_map[overlay_label] = {
                        "t": overlay_t,
                        "series": overlay_series,
                    }
                plot.set_statistics_results(
                    plot_results_map,
                    prefer=plot_label,
                )
                replay_selected_ids = preserved_selected_ids or ([preserved_set_id] if preserved_set_id else [])
                self._record_current_main_plot_workspace_preview_provenance(
                    selected_set_ids=replay_selected_ids
                )
                self.show_simulation_tab()
                self.refresh_simulation_plot_views()
            label = getattr(self, "_status_label", None)
            if label is not None:
                try:
                    if str(label.text()) in (
                        "Result not cached (evicted). Press Run to compute.",
                        "Cached result invalid. Press Run to compute.",
                        "Preview pending for current selection.",
                    ):
                        label.setText("Ready")
                except RuntimeError:
                    self._status_label = None
            return
        if active_cache_key and batch_cache.active_cache_invalidated_set_ids:
            self._clear_batch_selection_display_state()
            label = getattr(self, "_status_label", None)
            if label is not None:
                try:
                    label.setText("Result not cached (evicted). Press Run to compute.")
                except RuntimeError:
                    self._status_label = None
            return
        if selected_ids:
            self._refresh_batch_display_from_focus_and_shown()
            return
        self._clear_batch_selection_display_state()

    def _reset_project_apply_dirty_session_state(self) -> None:
        """Clear non-serialized session state before applying a project payload."""
        if not self._prepare_fit_window_shutdown_for_close():
            logger.warning("Project apply requested while one or more fit windows remained open after close request")
        self._preview_session.clear_working_transaction(clear_committed_slider_values=True)
        self._variable_runtime.clear_prepared_slider_runtime(dirty=True)
        dataset_manager = getattr(self, "_dataset_manager", None)
        if dataset_manager is not None and hasattr(dataset_manager, "clear_all_datasets"):
            dataset_manager.clear_all_datasets()

        data_manager = getattr(getattr(self, "_right_panel", None), "_data_manager", None)
        if data_manager is not None and hasattr(data_manager, "clear_datasets"):
            data_manager.clear_datasets()

        self._clear_main_plot_project_apply_state()
        self._sync_overlay_catalog()

        batch_cache = getattr(self._sim_controller, "batch_cache", None)
        if batch_cache is not None and hasattr(batch_cache, "reset_runtime_state"):
            batch_cache.reset_runtime_state()

        self._refresh_slider_transaction_button_state()
        self.show_simulation_tab()
        self.refresh_simulation_plot_views()

    def _rebind_species_panel_after_batch_model_replacement(self) -> None:
        """Reattach Species mode to the replacement batch model/selection model."""
        table = getattr(self, "_batch_table", None)
        model = getattr(self, "_batch_model", None)
        if table is None or model is None:
            return

        panel = None
        try:
            panel = self._mechanism_editor.species_sliders_widget()
        except Exception:
            panel = None
        if panel is None or not hasattr(panel, "attach"):
            return

        try:
            if hasattr(panel, "set_transaction_owner"):
                panel.set_transaction_owner(self._preview_session)
            panel.attach(table=table, model=model)
        except RuntimeError as exc:
            logger.debug("Failed to reattach species panel after project apply: %s", exc, exc_info=True)
            self._species_panel_available = False
            return

        if hasattr(panel, "activate"):
            self._ensure_batch_current_row_selected()
            try:
                panel.activate()
            except RuntimeError as exc:
                logger.debug("Failed to reactivate species panel after project apply: %s", exc, exc_info=True)
                self._species_panel_available = False

    def _get_solver_settings(self) -> Dict[str, Any]:
        """Get the current solver settings."""
        solver_contract = load_solver_contract()
        solver_label = str(self._initial_solver or solver_contract.default_solver_name).strip() or solver_contract.default_solver_name
        solver_method, solver_warning = solver_contract.normalize_solver_name(solver_label)
        return {
            "solver": str(solver_method),
            "solver_label": str(solver_label),
            "solver_warning": str(solver_warning) if solver_warning else None,
            'rtol': self._initial_rtol or 1e-6,
            'atol': self._initial_atol or 1e-12,
            'use_sparse_jacobian': bool(self._use_sparse_jacobian),
            'wegscheider_cyclicity_enabled': bool(self._wegscheider_cyclicity_enabled),
            'max_parallel_batch_workers': int(self._sim_controller.parallel_batch.max_parallel_workers),
            'limit_blas_threads_per_worker': bool(self._sim_controller.parallel_batch.limit_blas_threads_per_worker),
        }

    def _serialize_project_state(self) -> Dict[str, Any]:
        """Create a versioned snapshot of the current project."""
        solver_contract = load_solver_contract()
        solver_label = str(self._initial_solver or solver_contract.default_solver_name).strip() or solver_contract.default_solver_name
        solver_method, solver_warning = solver_contract.normalize_solver_name(solver_label)
        return {
            'project_schema_version': PROJECT_SCHEMA_VERSION,
            'version': KINDRED_VERSION,
            'mechanism': self._mechanism_editor._reactions_text.toPlainText(),
            'notes': self._mechanism_editor._notes_text.toPlainText(),
            'state_network': self._mechanism_editor._state_network_editor.get_state_network_dsl(),
            "solver": str(solver_label),
            "solver_method": str(solver_method),
            "solver_warning": str(solver_warning) if solver_warning else None,
            'rtol': self._initial_rtol or 1e-6,
            'atol': self._initial_atol or 1e-12,
            'use_sparse_jacobian': bool(self._use_sparse_jacobian),
            'wegscheider_cyclicity_enabled': bool(self._wegscheider_cyclicity_enabled),
            'max_parallel_batch_workers': int(self._sim_controller.parallel_batch.max_parallel_workers),
            'limit_blas_threads_per_worker': bool(self._sim_controller.parallel_batch.limit_blas_threads_per_worker),
            'temperature_K': self._temperature_spinbox.value(),
            'simulation_time': str(self._sim_time_spinbox.text()).strip(),
            'num_points': int(self._num_points_spinbox.value()),
            'batch_initial_conditions': self._batch_store.as_serializable(),
        }

    def serialize_project_state(self) -> Dict[str, Any]:
        """Public project snapshot API for controllers (avoid reaching into `_` helpers)."""
        return self._serialize_project_state()

    def _set_text_with_optional_undo(
        self,
        widget: QtWidgets.QPlainTextEdit,
        new_text: str,
        description: str,
        record_undo: bool,
    ) -> None:
        """Set text on a QPlainTextEdit, optionally recording the change on the undo stack."""
        from kindred.gui.undo_commands import SetMechanismTextCommand

        current_text = widget.toPlainText()
        if new_text is None:
            new_text = ""

        if current_text == new_text:
            return

        if record_undo:
            command = SetMechanismTextCommand(widget, new_text, current_text, description)
            self._undo_stack.push(command)
        else:
            widget.blockSignals(True)
            try:
                widget.setPlainText(new_text)
            finally:
                widget.blockSignals(False)
            widget.document().contentsChanged.emit()

    def _apply_project_payload(self, data: Dict[str, Any], *, record_undo: bool = True) -> None:
        """Populate the UI from serialized project data."""
        from kindred.core.batch_initial_conditions import (
            BatchInitialConditionsStore,
            migrate_reaction_dsl_initial_concentration_sets,
        )
        from kindred.gui.widgets.batch_initial_conditions_table import BatchInitialConditionsTableModel

        project_version = int(data.get('project_schema_version', 1))
        if project_version > PROJECT_SCHEMA_VERSION:
            QtWidgets.QMessageBox.warning(
                self,
                "Unsupported Project Version",
                "This project was saved with a newer version of Kindred. "
                "Some settings may not load correctly."
            )
            logger.warning(
                "Loading newer project schema version (%s > %s)",
                project_version,
                PROJECT_SCHEMA_VERSION
            )

        self._reset_project_apply_dirty_session_state()
        self._clear_last_mechanism()

        mechanism_text = data.get('mechanism', "")
        batch_payload = data.get("batch_initial_conditions")
        seed_sets: Dict[str, Dict[str, float]]
        rewritten = mechanism_text
        if isinstance(batch_payload, dict):
            seed_sets = {}
        else:
            try:
                seed_sets, rewritten = migrate_reaction_dsl_initial_concentration_sets(
                    mechanism_text,
                    default_set_name="set1",
                )
            except Exception:
                seed_sets, rewritten = ({}, mechanism_text)
        notes_text = data.get('notes', "")
        self._set_text_with_optional_undo(
            self._mechanism_editor._reactions_text,
            mechanism_text,
            "Load project (reactions)",
            record_undo,
        )
        self._on_programmatic_mechanism_load()

        # Batch initial conditions (schema v3+). For older projects, migrate any
        # inline initial concentrations into set1 and rewrite the block stub.
        if isinstance(batch_payload, dict):
            try:
                self._batch_store = BatchInitialConditionsStore.from_serializable(batch_payload)
            except Exception:
                self._batch_store = BatchInitialConditionsStore()
        else:
            self._batch_store = BatchInitialConditionsStore()
            if seed_sets:
                self._materialize_migrated_initial_concentration_sets(seed_sets=seed_sets)
                self._set_text_with_optional_undo(
                    self._mechanism_editor._reactions_text,
                    rewritten,
                    "Migrate initial concentrations to batch table",
                    record_undo,
                )

        self._batch_model = BatchInitialConditionsTableModel(self._batch_store, parent=self)
        if self._batch_table is not None:
            self._batch_table.setModel(self._batch_model)
            self._rebind_batch_semantics_signal_bindings()
        self._rebind_species_panel_after_batch_model_replacement()
        self._update_batch_row_controls_state()
        self._on_batch_current_changed()

        self._set_text_with_optional_undo(
            self._mechanism_editor._notes_text,
            notes_text,
            "Load project (notes)",
            record_undo,
        )

        state_network_text = data.get('state_network') or ""
        state_editor = self._mechanism_editor._state_network_editor
        current_state_network = state_editor.get_state_network_dsl()
        if state_network_text.strip():
            if state_network_text.strip() != current_state_network.strip():
                state_editor.set_state_network_dsl(state_network_text)
        else:
            if current_state_network.strip():
                state_editor.clear()

        solver_contract = load_solver_contract()
        solver_value = data.get('solver', self._initial_solver or solver_contract.default_solver_name)
        rtol_value = data.get('rtol', self._initial_rtol or 1e-6)
        atol_value = data.get('atol', self._initial_atol or 1e-12)

        # Load solver/settings metadata (with safe defaults for older files)
        if 'use_sparse_jacobian' in data:
            self._use_sparse_jacobian = bool(data.get('use_sparse_jacobian'))
        if 'wegscheider_cyclicity_enabled' in data:
            self._wegscheider_cyclicity_enabled = bool(data.get('wegscheider_cyclicity_enabled'))
        if 'max_parallel_batch_workers' in data:
            try:
                self._sim_controller.parallel_batch.max_parallel_workers = max(1, int(data.get('max_parallel_batch_workers')))
            except Exception:
                self._sim_controller.parallel_batch.max_parallel_workers = 12
        if 'limit_blas_threads_per_worker' in data:
            self._sim_controller.parallel_batch.limit_blas_threads_per_worker = bool(
                data.get('limit_blas_threads_per_worker')
            )
        if 'use_advanced_dsl' in data:
            logger.info(
                "Loaded legacy project flag use_advanced_dsl=%s (ignored; advanced DSL always enabled)",
                data['use_advanced_dsl'],
            )
        if 'temperature_K' in data:
            self._temperature_spinbox.setValue(data['temperature_K'])
        if 'simulation_time' in data:
            sim_time = data.get('simulation_time')
            if isinstance(sim_time, (int, float)):
                sim_time_text = f"{float(sim_time):g}"
            else:
                sim_time_text = str(sim_time)
            self._sim_time_spinbox.setText(sim_time_text)
        if 'num_points' in data:
            self._num_points_spinbox.setValue(int(data['num_points']))

        self._apply_solver_runtime_state(
            solver=solver_value,
            rtol=rtol_value,
            atol=atol_value,
        )

    def apply_project_payload(self, data: Dict[str, Any], *, record_undo: bool = True) -> bool:
        """Public project apply API for controllers (avoid reaching into `_` helpers)."""
        if not self._guard_slider_transaction_invalidation(action_text="Loading this project"):
            return False
        self._apply_project_payload(data, record_undo=record_undo)
        return True

    def apply_solver_runtime_state(
        self,
        *,
        solver: object = _SOLVER_STATE_UNSET,
        rtol: object = _SOLVER_STATE_UNSET,
        atol: object = _SOLVER_STATE_UNSET,
        sync_combo: bool = True,
    ) -> None:
        self._apply_solver_runtime_state(
            solver=solver,
            rtol=rtol,
            atol=atol,
            sync_combo=bool(sync_combo),
        )

    def add_to_recent_files(self, filepath: str) -> None:
        """Public API used by controllers (avoid reaching into `_` helpers)."""
        self._add_to_recent_files(str(filepath))

    def set_status_text(self, text: str) -> None:
        """Public API used by controllers (avoid reaching into `_` widget fields)."""
        self._status_label.setText(str(text))

    def main_plot_has_data(self) -> bool:
        plot = self.main_plot()
        return bool(getattr(plot, "_series", {})) and getattr(plot, "_t", None) is not None

    def main_plot_selected_series(self) -> List[str]:
        return list(self.main_plot().selected_series())

    def set_main_plot_selected_series(self, series_names: Sequence[str]) -> None:
        self.main_plot().set_selected_series(list(series_names))

    def run_button_is_enabled(self) -> bool:
        return bool(self._run_btn.isEnabled())

    def set_run_button_enabled(self, enabled: bool) -> None:
        self._run_btn.setEnabled(bool(enabled))

    def set_stop_button_enabled(self, enabled: bool) -> None:
        self._stop_btn.setEnabled(bool(enabled))

    def set_sim_progress_value(self, value: int) -> None:
        self._sim_progress.setValue(int(value))

    def repaint_simulation_widgets(self) -> None:
        with suppress(RuntimeError):
            self._sim_progress.update()
        with suppress(RuntimeError):
            self._status_label.update()
        table = getattr(self, "_results_table", None)
        if table is not None:
            with suppress(RuntimeError):
                viewport = table.viewport()
                viewport.update()

    def set_algebra_status_text(self, text: str) -> None:
        self._algebra_status_label.setText(str(text))

    def message_box_warning(self, title: str, message: str) -> None:
        QtWidgets.QMessageBox.warning(self, str(title), str(message))

    def message_box_critical(self, title: str, message: str, *, details: Optional[str] = None) -> None:
        full_message = str(message)
        if details:
            full_message = f"{full_message}\n\nDetails:\n{details}"
        QtWidgets.QMessageBox.critical(self, str(title), full_message)

    def main_plot(self) -> object:
        return self._plot_tabs._main_plot

    def set_main_plot_data(
        self,
        t: np.ndarray,
        series: Dict[str, np.ndarray],
        *,
        label: Optional[str] = None,
        overlays: Optional[Sequence[Dict[str, object]]] = None,
        owned_species: Optional[Sequence[str]] = None,
    ) -> None:
        self.main_plot().set_data(t, series, label=label, overlays=overlays, owned_species=owned_species)

    def sync_main_plot_copy_labels(self, primary_set_id: str, selected_set_ids: Sequence[str]) -> None:
        plot = getattr(getattr(self, "_plot_tabs", None), "_main_plot", None)
        if plot is None:
            return
        primary_set_id_s = str(primary_set_id or "").strip()
        selected_ids: list[str] = []
        for raw_set_id in selected_set_ids or ():
            set_id = str(raw_set_id or "").strip()
            if not set_id or set_id in selected_ids:
                continue
            selected_ids.append(set_id)
        if primary_set_id_s and primary_set_id_s not in selected_ids:
            selected_ids.append(primary_set_id_s)
        popup_labels = self._copy_all_popup_labels_by_set_id(selected_ids)
        primary_label = str(getattr(plot, "_simulation_set_label", "") or "").strip()
        setattr(plot, "_simulation_set_popup_label", str(popup_labels.get(primary_set_id_s, primary_label)))
        for entry in list(getattr(plot, "_simulation_overlays", []) or []):
            if not isinstance(entry, dict):
                continue
            entry_set_id = str(entry.get("set_id") or "").strip()
            popup_label = str(popup_labels.get(entry_set_id, "")).strip()
            if popup_label:
                entry["popup_label"] = popup_label
            else:
                entry.pop("popup_label", None)

    def show_simulation_tab(self) -> None:
        self._plot_tabs._tabs.setCurrentIndex(0)

    def refresh_simulation_plot_views(self) -> None:
        self.main_plot().update()
        self._plot_tabs.update()
        self.update()

    def schedule_main_plot_refresh(self, delays_ms: Sequence[int]) -> None:
        plot = self.main_plot()

        def _safe_plot_update(plot_widget=plot) -> None:
            with suppress(RuntimeError):
                plot_widget.update()

        for delay_ms in delays_ms:
            QtCore.QTimer.singleShot(int(delay_ms), _safe_plot_update)

    def set_main_plot_scalar_values(self, scalars: Dict[str, object]) -> None:
        plot = self.main_plot()
        if hasattr(plot, "set_scalar_values"):
            plot.set_scalar_values(scalars)

    def update_main_plot_parameter_summary(self, parameters: Dict[str, tuple[float, str]]) -> None:
        plot = self.main_plot()
        if hasattr(plot, "update_parameters"):
            plot.update_parameters(dict(parameters))

    def integrate_ctc(
        self,
        t: Any,
        y: Any,
        *,
        uniformity_eps: float,
        tail_strategy: str,
    ) -> tuple[float, str, bool, float, str]:
        from kindred.core.results import integrate_ctc as _integrate_ctc

        return _integrate_ctc(
            t,
            y,
            uniformity_eps=float(uniformity_eps),
            tail_strategy=str(tail_strategy),
        )

    def update_main_plot_statistics(
        self,
        *,
        stats_results_map: Dict[str, Dict[str, object]],
        prefer: str,
        t: np.ndarray,
        series: Dict[str, np.ndarray],
    ) -> None:
        plot = self.main_plot()
        if hasattr(plot, "set_statistics_results"):
            plot.set_statistics_results(stats_results_map, prefer=prefer)
            return
        plot.update_statistics(t, series)

    def main_plot_stats_table(self) -> object:
        return self.main_plot().stats_table()

    def set_results_table(self, table: object) -> None:
        self._results_table = table

    def mechanism_reactions_text_raw(self) -> str:
        return str(self._mechanism_editor.reactions_text())

    def mechanism_state_network_dsl_raw(self) -> str:
        return str(self._mechanism_editor.state_network_dsl_raw() or "")

    def mechanism_slider_points_value(self) -> Optional[int]:
        try:
            return int(self._mechanism_editor.slider_points_value())
        except Exception:
            return None

    def mechanism_slider_solver_value(self) -> Optional[str]:
        try:
            value = self._mechanism_editor.slider_solver_value()
        except Exception:
            return None
        return str(value) if value is not None else None

    def set_variable_sliders(
        self,
        variables: Dict[str, float],
        *,
        metadata: Optional[Dict[str, Dict[str, object]]] = None,
        preserve_visibility: bool = False,
        visibility_scope_signature: object | None = None,
    ) -> None:
        self._mechanism_editor._variable_sliders.set_variables(
            dict(variables),
            metadata=dict(metadata or {}),
            preserve_visibility=bool(preserve_visibility),
            visibility_scope_signature=visibility_scope_signature,
        )

    def variable_slider_values(self) -> Dict[str, float]:
        sliders = getattr(getattr(self, "_mechanism_editor", None), "_variable_sliders", None)
        if sliders is None or not hasattr(sliders, "get_variables"):
            return {}
        values = sliders.get_variables() or {}
        return {str(name): float(value) for name, value in values.items()}

    def clear_variable_sliders(self) -> None:
        sliders = getattr(getattr(self, "_mechanism_editor", None), "_variable_sliders", None)
        if sliders is not None and hasattr(sliders, "clear"):
            sliders.clear()

    def temperature_spinbox_value(self) -> float:
        return float(self._temperature_spinbox.value())

    def num_points_spinbox_value(self) -> int:
        return int(self._num_points_spinbox.value())

    def sim_time_spinbox_text(self) -> str:
        return str(self._sim_time_spinbox.text())

    def parse_sim_time_seconds(self) -> float:
        return float(self._parse_sim_time_seconds())

    def use_sparse_jacobian(self) -> bool:
        return bool(self._use_sparse_jacobian)

    def wegscheider_cyclicity_enabled(self) -> bool:
        return bool(self._wegscheider_cyclicity_enabled)

    def set_mechanism_reactions_text_with_optional_undo(
        self,
        new_text: str,
        description: str,
        *,
        record_undo: bool,
    ) -> None:
        self._set_text_with_optional_undo(
            self._mechanism_editor._reactions_text,
            str(new_text),
            str(description),
            bool(record_undo),
        )

    def set_temperature_override_state(self, *, enabled: bool, tooltip: str) -> None:
        self._set_temperature_override_state(enabled=bool(enabled), tooltip=str(tooltip))

    def set_temperature_mode_indicator_text(self, text: str) -> None:
        self._temperature_mode_indicator.setText(str(text))

    def batch_rows_for_scope(self, scope: str) -> List[int]:
        return [int(r) for r in (self._batch_rows_for_scope(str(scope)) or [])]

    def batch_set_ids_for_scope(self, scope: str) -> List[str]:
        return [str(s) for s in (self._batch_set_ids_for_scope(str(scope)) or [])]

    def shown_batch_set_ids(self) -> List[str]:
        return [str(s) for s in (self._shown_batch_set_ids() or [])]

    def slider_edit_target_set_ids(self) -> List[str]:
        return [str(s) for s in (self._slider_edit_target_set_ids() or [])]

    def set_slider_edit_target_set_ids(self, set_ids: Sequence[str]) -> None:
        self._set_slider_edit_target_set_ids(set_ids)

    def batch_current_row(self) -> Optional[int]:
        row = self._batch_current_row()
        return int(row) if row is not None else None

    def focused_batch_set_id(self) -> Optional[str]:
        value = self._focused_batch_set_id_value()
        return str(value) if value else None

    def batch_set_id_for_row(self, row: int) -> Optional[str]:
        value = self._batch_set_id_for_row(int(row))
        return str(value) if value is not None else None

    def batch_set_name_for_id(self, set_id: str) -> Optional[str]:
        value = self._batch_set_name_for_id(str(set_id))
        return str(value) if value is not None else None

    def batch_set_id_for_name(self, name: str) -> Optional[str]:
        value = self._batch_set_id_for_name(str(name))
        return str(value) if value is not None else None

    def batch_preferred_primary_set_id(self, rows: Sequence[int]) -> Optional[str]:
        value = self._batch_preferred_primary_set_id(list(rows))
        return str(value) if value is not None else None

    def batch_cache_key(
        self,
        *,
        scope_identity: object | None = None,
        mechanism_text: str = "",
        solver_config: Optional[Dict[str, Any]] = None,
        t_end: float = 0.0,
    ) -> str:
        return str(
            self._batch_cache_key(
                scope_identity=scope_identity,
                mechanism_text=str(mechanism_text),
                solver_config=dict(solver_config or {}),
                t_end=float(t_end),
            )
        )

    def active_batch_cache_key(self) -> str:
        return str(self._sim_controller.batch_cache.active_cache_key or "")

    def active_batch_selection(self) -> tuple[str, str]:
        batch_cache = self._sim_controller.batch_cache
        return (
            str(batch_cache.active_batch_set_id or ""),
            str(batch_cache.active_batch_set or ""),
        )

    def set_active_batch_selection(self, set_id: str, set_name: str, selected_ids: Sequence[str]) -> None:
        batch_cache = self._sim_controller.batch_cache
        batch_cache.active_batch_set_id = str(set_id)
        batch_cache.active_batch_set = str(set_name)
        batch_cache.last_display_selection = [str(item) for item in (selected_ids or []) if str(item)]

    def clear_display_selection_state(self) -> None:
        clear_display = getattr(self._sim_controller.batch_cache, "clear_display_selection_state", None)
        if callable(clear_display):
            clear_display()

    def batch_result_cache_store(self) -> MutableMapping[str, Dict[str, Any]]:
        return self._sim_controller.batch_cache.result_cache

    def batch_store_row_count(self) -> int:
        return int(self._batch_store.row_count())

    def batch_store_set_names(self) -> List[str]:
        return [str(n) for n in (self._batch_store.set_names() or [])]

    def batch_store_visible_species(self) -> List[str]:
        return [str(n) for n in (self._batch_store.visible_species() or [])]

    def batch_model_validate_rows(self, rows: Sequence[int]) -> set[tuple[int, str]]:
        invalid = self._batch_model.validate_rows(list(rows))
        if not invalid:
            return set()
        return {(int(row), str(species)) for row, species in invalid}

    def batch_initials_for_row(self, row: int) -> Dict[str, float]:
        initials = self._batch_initials_for_row(int(row))
        if not initials:
            return {}
        if isinstance(initials, dict):
            return {str(key): float(value) for key, value in initials.items()}
        return dict(initials)

    def display_cached_batch_selection(
        self,
        *,
        cache_key: str,
        selected_sets: Sequence[str],
        prefer_set: Optional[str] = None,
        cache_store: Optional[object] = None,
        valid_set_ids: Optional[Sequence[str]] = None,
        invalidated_set_ids: Optional[Sequence[str]] = None,
        allow_fallback: bool = True,
    ) -> bool:
        batch_cache = self._sim_controller.batch_cache
        normalized_selected_sets = [str(set_id) for set_id in (selected_sets or ()) if str(set_id)]
        if cache_store is batch_cache.preview_cache and normalized_selected_sets:
            workspace_displayed = self._display_workspace_aware_preview_batch_selection(
                selected_sets=normalized_selected_sets,
                prefer_set=prefer_set,
                preview_cache_key=str(cache_key or ""),
            )
            if workspace_displayed:
                return True
            if len(normalized_selected_sets) > 1:
                return False
            if not bool(allow_fallback):
                single_set_id = str(normalized_selected_sets[0] or "")
                if single_set_id and self._preview_session.has_dirty_state_for_set(single_set_id):
                    preview_entry = self._matching_preview_entry_for_workspace_set(
                        set_id=single_set_id,
                        preview_cache_key=str(cache_key or ""),
                    )
                    if preview_entry.entry is None:
                        return False
        resolved_invalidated_set_ids = invalidated_set_ids
        if (
            resolved_invalidated_set_ids is None
            and str(batch_cache.active_cache_key or "") == str(cache_key)
        ):
            resolved_invalidated_set_ids = batch_cache.active_cache_invalidated_set_ids
        displayed = self.results_controller.display_cached_batch_selection(
            cache_key=str(cache_key),
            selected_sets=normalized_selected_sets,
            prefer_set=str(prefer_set) if prefer_set is not None else None,
            cache_store=cache_store,
            valid_set_ids=(
                tuple(str(set_id) for set_id in valid_set_ids)
                if valid_set_ids is not None
                else None
            ),
            invalidated_set_ids=(
                tuple(str(set_id) for set_id in resolved_invalidated_set_ids)
                if resolved_invalidated_set_ids is not None
                else None
            ),
            allow_fallback=bool(allow_fallback),
        )
        if displayed:
            self._record_current_main_plot_workspace_preview_provenance(selected_set_ids=normalized_selected_sets)
        return displayed

    def _focused_batch_selection_is_dirty(
        self,
        *,
        selected_sets: Sequence[str],
        prefer_set: Optional[str] = None,
    ) -> bool:
        focused_set_id = str(prefer_set or (selected_sets[0] if selected_sets else "") or "").strip()
        if not focused_set_id:
            return False
        try:
            return bool(self._preview_session.has_dirty_state_for_set(focused_set_id))
        except Exception:
            return False

    def _selection_uses_fresh_explicit_cache_after_post_run_sync(
        self,
        *,
        selected_sets: Sequence[str],
    ) -> bool:
        normalized_selected_sets = [str(set_id) for set_id in (selected_sets or ()) if str(set_id)]
        if not normalized_selected_sets:
            return False
        batch_cache = self._sim_controller.batch_cache
        active_cache_key = str(getattr(batch_cache, "active_cache_key", "") or "").strip()
        active_preview_token = str(getattr(batch_cache, "active_cache_preview_token", "") or "").strip()
        if not active_cache_key or not active_preview_token:
            return False
        active_valid_set_ids = {
            str(set_id) for set_id in (getattr(batch_cache, "active_cache_valid_set_ids", None) or ()) if str(set_id)
        }
        if active_valid_set_ids and any(set_id not in active_valid_set_ids for set_id in normalized_selected_sets):
            return False
        active_preview_scope_ids = {
            str(set_id)
            for set_id in (getattr(batch_cache, "active_cache_preview_scope_set_ids", None) or ())
            if str(set_id)
        }
        if active_preview_scope_ids and any(set_id not in active_preview_scope_ids for set_id in normalized_selected_sets):
            return False
        scope_rows: list[int] = []
        row_for_set_id = getattr(getattr(self, "_batch_store", None), "row_for_set_id", None)
        if not callable(row_for_set_id):
            return False
        for set_id in normalized_selected_sets:
            try:
                row = row_for_set_id(str(set_id))
            except Exception:
                row = None
            if row is None:
                return False
            scope_rows.append(int(row))
        try:
            current_preview_token = str(self._preview_session.preview_batch_cache_token(scope_rows) or "").strip()
        except Exception:
            return False
        return bool(current_preview_token) and current_preview_token == active_preview_token

    def _display_workspace_aware_preview_batch_selection(
        self,
        *,
        selected_sets: Sequence[str],
        prefer_set: Optional[str] = None,
        preview_cache_key: Optional[str] = None,
    ) -> bool:
        normalized_selected_sets = [str(set_id) for set_id in (selected_sets or ()) if str(set_id)]
        if not normalized_selected_sets:
            return False
        focused_selection_is_dirty = self._focused_batch_selection_is_dirty(
            selected_sets=normalized_selected_sets,
            prefer_set=prefer_set,
        )
        (
            resolved_entries,
            outcome_reason,
            all_selected_sets_resolved,
            has_workspace_selection,
            has_resolved_workspace_preview,
            focused_selection_uses_workspace_controls,
            focused_selection_has_resolved_entry,
        ) = self._resolve_workspace_aware_batch_selection(
            selected_sets=normalized_selected_sets,
            preview_cache_key=preview_cache_key,
        )
        if not has_workspace_selection:
            return False
        if all_selected_sets_resolved and resolved_entries:
            outcome = self.results_controller.display_resolved_batch_selection_outcome(
                resolved_entries=resolved_entries,
                prefer_set=prefer_set,
            )
            if outcome.displayed:
                self._record_current_main_plot_workspace_preview_provenance(
                    selected_set_ids=normalized_selected_sets
                )
            return bool(outcome.displayed)
        if (
            resolved_entries
            and outcome_reason in {"preview_pending", "no_cached_results"}
            and has_resolved_workspace_preview
            and (
                bool(focused_selection_uses_workspace_controls)
                or ((not bool(focused_selection_is_dirty)) and bool(focused_selection_has_resolved_entry))
            )
        ):
            outcome = self.results_controller.display_resolved_batch_selection_outcome(
                resolved_entries=resolved_entries,
                prefer_set=prefer_set,
            )
            if outcome.displayed:
                self._record_current_main_plot_workspace_preview_provenance(
                    selected_set_ids=normalized_selected_sets
                )
                if outcome_reason == "preview_pending":
                    self.set_status_text("Preview pending for current selection.")
                else:
                    self.set_status_text("Result not cached (evicted). Press Run to compute.")
            return bool(outcome.displayed)
        return False

    def display_workspace_aware_batch_selection(
        self,
        *,
        selected_sets: Sequence[str],
        prefer_set: Optional[str] = None,
        preview_cache_key: Optional[str] = None,
    ) -> bool:
        return bool(
            self._display_workspace_aware_preview_batch_selection(
                selected_sets=selected_sets,
                prefer_set=prefer_set,
                preview_cache_key=preview_cache_key,
            )
        )

    def update_batch_row_controls_state(self) -> None:
        self._update_batch_row_controls_state()

    def sync_batch_species_columns(
        self,
        species_names: Sequence[str],
        *,
        preserve_active_cache: bool = False,
    ) -> None:
        self._sync_batch_species_columns(list(species_names), preserve_active_cache=bool(preserve_active_cache))

    def has_slider_overrides(self) -> bool:
        return bool(self._preview_session.has_local_mechanism_workspaces())

    def _simulation_schema_text(self) -> str:
        reactions_text = self.mechanism_reactions_text_raw()
        state_network_dsl = self.mechanism_state_network_dsl_raw()
        full_dsl = reactions_text
        if state_network_dsl.strip():
            full_dsl += "\n\n# State Network\n" + state_network_dsl
        return str(full_dsl)

    def simulation_schema_id(self) -> str:
        param_store = self._preview_session.param_store
        schema_text = self._simulation_schema_text()
        if str(param_store.schema_text or "") != schema_text:
            param_store.set_schema(schema_text)
        return str(param_store.schema_id or "")

    def simulation_param_fingerprint(self, set_id: Optional[str] = None) -> str:
        self.simulation_schema_id()
        target_set_id = str(set_id or "").strip()
        if not self._preview_session.param_store.has_local_overrides_for_set(target_set_id):
            return ""
        return str(self._preview_session.param_store.param_fingerprint(target_set_id) or "")

    def _batch_store_is_pristine_default_placeholder(self) -> bool:
        if int(self._batch_store.row_count()) != 1:
            return False
        names = list(self._batch_store.set_names() or [])
        if names != ["set1"]:
            return False
        values = dict(self._batch_store.values_for_set("set1") or {})
        for raw in values.values():
            text = str(raw).strip()
            if not text:
                continue
            parsed, ok = try_parse_finite_float(text)
            if not ok or abs(float(parsed)) > 1e-12:
                return False
        return True

    def _materialize_migrated_initial_concentration_sets(
        self,
        *,
        seed_sets: Mapping[str, Mapping[str, object]],
    ) -> List[int]:
        rows: List[int] = []
        ordered_names = [str(name) for name in seed_sets.keys() if str(name).strip()]
        if not ordered_names:
            return rows

        reuse_default_row = bool(
            self._batch_store_is_pristine_default_placeholder()
            and str(ordered_names[0]) != "set1"
        )
        batch_model = getattr(self, "_batch_model", None)
        batch_model_attached = bool(
            batch_model is not None
            and hasattr(batch_model, "store")
            and callable(getattr(batch_model, "store"))
            and batch_model.store() is self._batch_store
        )
        row_by_name: Dict[str, int] = {}
        if reuse_default_row:
            self._batch_store.set_set_name(0, str(ordered_names[0]))
            row_by_name[str(ordered_names[0])] = 0

        seen_rows: set[int] = set()
        for set_name in ordered_names:
            seed = seed_sets.get(str(set_name)) or {}
            row_idx = row_by_name.get(str(set_name))
            if row_idx is None:
                existing_row = self._batch_store.row_for_set(str(set_name))
                if existing_row is None:
                    insert_at = int(self._batch_store.row_count())
                    if batch_model_attached:
                        batch_model.beginInsertRows(QtCore.QModelIndex(), insert_at, insert_at)
                    try:
                        row_idx = int(self._batch_store.ensure_set(str(set_name)))
                    finally:
                        if batch_model_attached:
                            batch_model.endInsertRows()
                else:
                    row_idx = int(existing_row)
                row_by_name[str(set_name)] = int(row_idx)
            for species, value in dict(seed).items():
                parsed, ok = try_parse_finite_float(value)
                if not ok:
                    continue
                self._batch_store.set_value(int(row_idx), str(species), f"{float(parsed):.6g}")
            if int(row_idx) not in seen_rows:
                rows.append(int(row_idx))
                seen_rows.add(int(row_idx))
        return rows

    def slider_overrides(self, set_id: Optional[str] = None) -> Dict[str, float]:
        raw = self._preview_session.slider_overrides(set_id=set_id)
        overrides: Dict[str, float] = {}
        for key, value in raw.items():
            parsed, ok = try_parse_finite_float(value)
            if not ok:
                continue
            overrides[str(key)] = float(parsed)
        return overrides

    def apply_overrides_to_text(self, base_text: str, *, set_id: Optional[str] = None) -> str:
        return str(self._apply_overrides_to_text(str(base_text), set_id=set_id))

    def apply_overrides_to_state_network_dsl(self, base_text: str, *, set_id: Optional[str] = None) -> str:
        return str(self._apply_overrides_to_state_network_dsl(str(base_text), set_id=set_id))

    def apply_parameter_overrides_to_dsl(self, mechanism_text: str, parameters: Dict[str, float]) -> str:
        return str(self._apply_parameter_overrides_to_dsl(str(mechanism_text), dict(parameters)))

    def get_mechanism_text(self) -> str:
        return str(self._get_mechanism_text())

    def initial_solver_name(self) -> Optional[str]:
        solver = self._initial_solver
        return str(solver) if solver is not None else None

    def explicit_startup_solver_name(self) -> Optional[str]:
        solver = getattr(self, "_explicit_startup_solver_value", None)
        return str(solver) if solver is not None else None

    def has_explicit_startup_solver_override(self) -> bool:
        return bool(getattr(self, "_explicit_startup_solver_override", False))

    def has_explicit_startup_rtol_override(self) -> bool:
        return bool(getattr(self, "_explicit_startup_rtol_override", False))

    def has_explicit_startup_atol_override(self) -> bool:
        return bool(getattr(self, "_explicit_startup_atol_override", False))

    def initial_rtol(self) -> Optional[float]:
        value = self._initial_rtol
        return float(value) if value is not None else None

    def explicit_startup_rtol(self) -> Optional[float]:
        value = getattr(self, "_explicit_startup_rtol_value", None)
        return float(value) if value is not None else None

    def initial_atol(self) -> Optional[float]:
        value = self._initial_atol
        return float(value) if value is not None else None

    def explicit_startup_atol(self) -> Optional[float]:
        value = getattr(self, "_explicit_startup_atol_value", None)
        return float(value) if value is not None else None

    def dsl_global_temperature_K(self, dsl_text: str) -> Optional[float]:
        value = self._dsl_global_temperature_K(str(dsl_text))
        return float(value) if value is not None else None

    def variable_metadata(self) -> Dict[str, Dict[str, object]]:
        return self._variable_runtime.variable_metadata()

    def set_variable_metadata(self, metadata: Dict[str, Dict[str, object]] | None) -> None:
        self._variable_runtime.set_variable_metadata(metadata)

    def _mutable_variable_metadata(self) -> Dict[str, Dict[str, object]]:
        return self._variable_runtime.mutable_variable_metadata()

    def snapshot_datasets(self) -> Dict[str, Any]:
        return dict(self._snapshot_datasets() or {})

    def last_fit_metadata(self) -> Optional[Dict[str, Any]]:
        value = self._last_fit_metadata
        return dict(value) if isinstance(value, dict) else None

    def set_last_simulation_provenance(self, provenance: Dict[str, Any]) -> None:
        self._last_simulation_provenance = dict(provenance)

    def set_last_simulation_ctc(self, ctc: Dict[str, float]) -> None:
        self._last_simulation_ctc = {str(key): float(value) for key, value in (ctc or {}).items()}

    def update_temperature_mode_indicator(self) -> None:
        self._update_temperature_mode_indicator()

    def populate_energy_mode_variables_from_mechanism(
        self,
        mechanism: object,
        *,
        refresh_sliders: bool,
        preserve_visibility: bool = False,
    ) -> None:
        self._populate_energy_mode_variables_from_mechanism(
            mechanism,
            refresh_sliders=bool(refresh_sliders),
            preserve_visibility=bool(preserve_visibility),
        )

    def apply_pending_init_migration(
        self,
        *,
        seed_sets: Mapping[str, Mapping[str, object]] | None = None,
        seed: Mapping[str, object] | None = None,
        rewrite: str,
    ) -> bool:
        if seed_sets is None and seed is not None:
            seed_sets = {"set1": dict(seed)}
        if not seed_sets or not rewrite:
            return False
        try:
            migrated_rows = self._materialize_migrated_initial_concentration_sets(seed_sets=seed_sets)
            for row_idx in migrated_rows:
                try:
                    top_left = self._batch_model.index(int(row_idx), 0)
                    bottom_right = self._batch_model.index(
                        int(row_idx),
                        max(0, int(self._batch_model.columnCount()) - 1),
                    )
                    self._batch_model.dataChanged.emit(
                        top_left,
                        bottom_right,
                        [QtCore.Qt.DisplayRole, QtCore.Qt.BackgroundRole],
                    )
                except Exception as exc:
                    logger.debug(
                        "Failed to emit batch model dataChanged after migrating pending init seed row %s: %s",
                        int(row_idx),
                        exc,
                        exc_info=True,
                    )
            previous_suppress = self._variable_runtime.suppress_slider_runtime_invalidation()
            previous_authoritative_suppress = bool(
                getattr(self, "_suppress_authoritative_mechanism_input_change", False)
            )
            self._variable_runtime.set_suppress_slider_runtime_invalidation(True)
            self._suppress_authoritative_mechanism_input_change = True
            restore_deferred = False
            try:
                self.set_mechanism_reactions_text_with_optional_undo(
                    str(rewrite),
                    "Migrate initial concentrations to batch table",
                    record_undo=True,
                )
                self._pending_init_migration_rewrite_for_invalidation = str(rewrite)
                self._pending_init_migration_state_network_for_invalidation = self.mechanism_state_network_dsl_raw()
                self._set_mechanism_edit_locked(True)
                restore_deferred = True
                def _restore_pending_init_rewrite_suppression() -> None:
                    self._variable_runtime.set_suppress_slider_runtime_invalidation(previous_suppress)
                    self._suppress_authoritative_mechanism_input_change = previous_authoritative_suppress

                QtCore.QTimer.singleShot(0, _restore_pending_init_rewrite_suppression)
            finally:
                if not restore_deferred:
                    self._variable_runtime.set_suppress_slider_runtime_invalidation(previous_suppress)
                    self._suppress_authoritative_mechanism_input_change = previous_authoritative_suppress
            return True
        except Exception:
            logger.debug("Failed to apply initial concentration migration to editor/store", exc_info=True)
            return False

    def arm_pending_init_result_invalidation_guard(self, *, rewrite: str | None = None) -> None:
        rewrite_text = str(self.mechanism_reactions_text_raw() or "") if rewrite is None else str(rewrite or "")
        self._pending_init_migration_rewrite_for_invalidation = rewrite_text
        self._pending_init_migration_state_network_for_invalidation = self.mechanism_state_network_dsl_raw()

    def invalidate_pending_init_preserved_results_after_failed_run(self) -> None:
        pending_init_rewrite = getattr(self, "_pending_init_migration_rewrite_for_invalidation", None)
        pending_init_state_network = getattr(
            self,
            "_pending_init_migration_state_network_for_invalidation",
            None,
        )
        self._pending_init_migration_rewrite_for_invalidation = None
        self._pending_init_migration_state_network_for_invalidation = None
        if pending_init_rewrite is None and pending_init_state_network is None:
            return
        self._invalidate_slider_runtime()
        batch_cache = getattr(self._sim_controller, "batch_cache", None)
        has_active_cache = bool(
            batch_cache is not None
            and (
                str(batch_cache.active_cache_key or "").strip()
                or str(batch_cache.active_preview_cache_key or "").strip()
            )
        )
        has_displayed_selection = bool(
            batch_cache is not None
            and (
                str(batch_cache.active_batch_set_id or "").strip()
                or str(batch_cache.active_batch_set or "").strip()
                or batch_cache.last_display_selection
            )
        )
        if not (has_active_cache or has_displayed_selection or self.main_plot_has_data()):
            return
        self._invalidate_active_results_after_authoritative_mechanism_change()

    def _apply_parameter_overrides_to_dsl(
        self,
        mechanism_text: str,
        parameters: Dict[str, float],
    ) -> str:
        """Return DSL text with parameter values replaced."""
        updated_text = mechanism_text
        step_analysis_context = None
        step_constraint_context = {
            "temperature_K": float(self._temperature_spinbox.value()),
            "wegscheider_cyclicity_enabled": bool(self.wegscheider_cyclicity_enabled()),
        }
        for name, value in parameters.items():
            name_str = str(name)
            if re.match(r"^(kf|kr|K|k)\d+$", name_str) and hasattr(self, "_update_variable_in_mechanism"):
                if re.match(r"^(kf|kr|K)\d+$", name_str) and (
                    step_analysis_context is None or step_analysis_context.source_text != updated_text
                ):
                    step_analysis_context = build_current_text_step_analysis_context(
                        updated_text,
                        step_constraint_context=step_constraint_context,
                    )
                previous_text = updated_text
                updated_text = self._update_variable_in_mechanism(
                    name_str,
                    float(value),
                    source_text=updated_text,
                    commit=False,
                    step_analysis_context=step_analysis_context,
                )
                if updated_text != previous_text:
                    step_analysis_context = None
                continue

            escaped_name = re.escape(name_str)
            pattern_assignment = rf"(?<![\w]){escaped_name}\s*=\s*(?P<value>[^\n#;,]+)"

            def _replace(match: re.Match) -> str:
                old_value = match.group("value").strip()
                return match.group(0).replace(old_value, f"{float(value):.6g}", 1)

            updated_text, replacements = re.subn(
                pattern_assignment,
                _replace,
                updated_text,
                count=1,
                flags=re.MULTILINE,
            )
            if replacements == 0:
                logger.debug("Parameter '%s' not found while applying overrides", name)
        return updated_text

    def _simulate_mechanism(
        self,
        mechanism_text: str,
        t_end: float,
        num_points: int,
        *,
        initials: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Simulate a mechanism synchronously and return the time-series payload."""
        from kindred.core.exceptions import SimulationError
        from kindred.core.ode_builder import build_ode_rhs_from_mechanism
        from kindred.core.simulator.dsl import parse_dsl_to_mechanism
        from kindred.core.simulator.parameter_algebra import apply_parameter_algebra_to_mechanism
        from kindred.core.simulator.solvers import SimulationRequest, solve_ode
        from kindred.core.units import UnitsModel

        solver_cfg = self._get_solver_settings()
        temperature = self._temperature_spinbox.value()

        units = UnitsModel(temperature_K=temperature)
        mechanism = parse_dsl_to_mechanism(mechanism_text, initials=initials or {}, units=units)
        _ = apply_parameter_algebra_to_mechanism(mechanism_text, mechanism=mechanism, require_mutable=False)
        species_names = mechanism.species_names()
        y0 = np.array([mechanism.species[sp].initial_conc for sp in species_names])
        rhs = build_ode_rhs_from_mechanism(mechanism)

        temperature_schedule = None
        try:
            meta = getattr(mechanism, "metadata", {}) or {}
            if isinstance(meta, dict):
                temperature_schedule = meta.get("temperature_schedule")
        except Exception:
            temperature_schedule = None

        jacobian_func = None
        from kindred.core.simulator.solvers import DEFAULT_SOLVER_NAME

        solver_name = str(solver_cfg.get("solver") or DEFAULT_SOLVER_NAME).upper()
        if solver_cfg.get("use_sparse_jacobian") and solver_name in {"RADAU", "BDF"}:
            try:
                from kindred.core.sparse_jacobian import build_sparse_jacobian

                jacobian_func = build_sparse_jacobian(mechanism)
                logger.info("Sparse Jacobian enabled for helper simulation (%s)", solver_name)
            except Exception as exc:
                logger.warning("Sparse Jacobian unavailable: %s", exc)
                jacobian_func = None
                solver_cfg["use_sparse_jacobian"] = False

        request = SimulationRequest(
            rhs=rhs,
            t_span=(0.0, float(t_end)),
            y0=y0,
            solver=solver_cfg["solver"],
            rtol=solver_cfg["rtol"],
            atol=solver_cfg["atol"],
            grid={"N": max(2, int(num_points))},
            jacobian_func=jacobian_func,
            temperature_schedule=temperature_schedule,
        )
        try:
            result = solve_ode(request)
        except SimulationError as exc:
            logger.error("Synchronous simulation failed: %s", exc)
            raise
        species_data = {name: result.Y[idx, :].copy() for idx, name in enumerate(species_names)}
        algebra_scalars: Dict[str, float] = {}
        try:
            from kindred.core.algebra.simulation_series import evaluate_algebra_series_for_simulation

            species_series = {name: result.Y[idx, :] for idx, name in enumerate(species_names)}
            initials_map = {sp: mechanism.species[sp].initial_conc for sp in species_names}
            algebra_series, algebra_scalars = evaluate_algebra_series_for_simulation(
                mechanism,
                t=result.t,
                species_series=species_series,
                initials=initials_map,
            )
            for name, values in (algebra_series or {}).items():
                species_data[str(name)] = np.asarray(values, dtype=float).copy()
        except Exception as exc:
            logger.warning("Algebra evaluation failed during synchronous simulation: %s", exc)
            algebra_scalars = {}
        return {
            "t": result.t.copy(),
            "species": species_data,
            "algebra_scalars": dict(algebra_scalars),
            "mechanism": mechanism,
            "solver_config": solver_cfg,
        }

    def _run_dataset_simulation(self, dsl_text: str) -> Dict[str, Any]:
        """
        Run simulation for a dataset tab (synchronous).

        Parameters
        ----------
        dsl_text : str
            Complete DSL text (mechanism + local initials)

        Returns
        -------
        dict
            Result dict with keys 't' and 'species'
        """
        try:
            sim_points = max(200, int(self._num_points_spinbox.value()))
            t_end = self._parse_sim_time_seconds()
            simulation = self._simulate_mechanism(
                dsl_text,
                t_end=float(t_end),
                num_points=sim_points,
            )
            mechanism = simulation.get('mechanism')
            solver_config = simulation.get('solver_config', self._get_solver_settings())
            if mechanism is not None:
                self._remember_last_mechanism(mechanism, dsl_text, solver_config)
            else:
                logger.warning("Dataset simulation returned no mechanism; clearing export cache")
                self._clear_last_mechanism()
            return {
                't': simulation['t'],
                'species': simulation['species'],
            }
        except Exception as e:
            logger.error(f"Dataset simulation failed: {e}")
            raise

    def _batch_selected_rows(self) -> List[int]:
        table = getattr(self, "_batch_table", None)
        if table is None:
            return []
        sel = table.selectionModel()
        if sel is None:
            return []
        rows = sorted({idx.row() for idx in sel.selectedRows() if idx.isValid()})
        return [int(r) for r in rows]

    def _batch_current_row(self) -> Optional[int]:
        table = getattr(self, "_batch_table", None)
        if table is None:
            return None
        idx = table.currentIndex()
        if not idx.isValid():
            return None
        return int(idx.row())

    def _batch_set_id_for_row(self, row: int) -> Optional[str]:
        try:
            if hasattr(self._batch_store, "set_id_for_row"):
                return str(self._batch_store.set_id_for_row(int(row)))
        except Exception:
            return None
        names = list(self._batch_store.set_names())
        if 0 <= int(row) < len(names):
            return str(names[int(row)])
        return None

    def _batch_set_id_for_name(self, set_name: str) -> Optional[str]:
        name = str(set_name or "").strip()
        if not name:
            return None
        row = self._batch_store.row_for_set(name)
        if row is None:
            return None
        return self._batch_set_id_for_row(int(row))

    def _batch_set_name_for_id(self, set_id: str) -> Optional[str]:
        sid = str(set_id or "").strip()
        if not sid:
            return None
        if hasattr(self._batch_store, "set_name_for_set_id"):
            try:
                name = self._batch_store.set_name_for_set_id(sid)
            except Exception as exc:
                logger.debug("Batch store set_name_for_set_id failed for %s: %s", sid, exc, exc_info=True)
                return None
            if isinstance(name, str) and name.strip():
                return str(name)
        return None

    def _batch_selected_set_ids(self) -> List[str]:
        rows = self._batch_selected_rows()
        ids: List[str] = []
        for row in rows:
            sid = self._batch_set_id_for_row(int(row))
            if sid:
                ids.append(str(sid))
        return ids

    def _shown_batch_set_ids(self) -> List[str]:
        model = getattr(self, "_batch_model", None)
        if model is None or not hasattr(model, "shown_set_ids"):
            return []
        return [str(set_id) for set_id in model.shown_set_ids() if str(set_id)]

    def _slider_edit_target_set_ids(self) -> List[str]:
        model = getattr(self, "_batch_model", None)
        if model is None or not hasattr(model, "slider_edit_target_set_ids"):
            return []
        return [str(set_id) for set_id in model.slider_edit_target_set_ids() if str(set_id)]

    def _set_slider_edit_target_set_ids(self, set_ids: Sequence[str]) -> bool:
        model = getattr(self, "_batch_model", None)
        if model is None or not hasattr(model, "set_slider_edit_target_set_ids"):
            return False
        return bool(model.set_slider_edit_target_set_ids(set_ids))

    def _effective_slider_edit_target_set_ids(self) -> List[str]:
        focused_set_id = self._focused_batch_set_id_value()
        effective_ids: list[str] = []
        seen: set[str] = set()
        if focused_set_id:
            effective_ids.append(focused_set_id)
            seen.add(focused_set_id)
        for set_id in self._slider_edit_target_set_ids():
            set_id_s = str(set_id or "").strip()
            if not set_id_s or set_id_s in seen:
                continue
            effective_ids.append(set_id_s)
            seen.add(set_id_s)
        return effective_ids

    def _focused_batch_set_id_value(self) -> str:
        current_row = self._batch_current_row()
        if current_row is not None:
            current_set_id = str(self._batch_set_id_for_row(int(current_row)) or "").strip()
            if current_set_id:
                return current_set_id
        focused_set_id = str(getattr(self, "_focused_batch_set_id", "") or "").strip()
        if focused_set_id and self._batch_store.row_for_set_id(focused_set_id) is not None:
            return focused_set_id
        if int(self._batch_store.row_count()) > 0:
            fallback = str(self._batch_set_id_for_row(0) or "").strip()
            if fallback:
                return fallback
        return ""

    def _batch_row_for_set_id(self, set_id: str) -> Optional[int]:
        try:
            return self._batch_store.row_for_set_id(str(set_id or ""))
        except Exception:
            return None

    def _copy_all_popup_labels_by_set_id(self, set_ids: Sequence[str]) -> Dict[str, str]:
        labels_by_id: Dict[str, str] = {}
        label_counts: Dict[str, int] = {}
        for raw_set_id in set_ids or ():
            set_id = str(raw_set_id or "").strip()
            if not set_id:
                continue
            label = str(self.batch_set_name_for_id(set_id) or set_id)
            labels_by_id[set_id] = label
            label_counts[label] = int(label_counts.get(label, 0)) + 1

        popup_labels: Dict[str, str] = {}
        for set_id, label in labels_by_id.items():
            popup_label = str(label)
            if int(label_counts.get(label, 0)) > 1:
                row = self._batch_row_for_set_id(set_id)
                if row is not None:
                    popup_label = f"{label} (row {int(row) + 1})"
            popup_labels[set_id] = popup_label
        return popup_labels

    def _copy_all_shown_block_from_entry(
        self,
        *,
        set_id: str,
        label: str,
        entry: Mapping[str, Any],
    ):
        from kindred.gui.widgets.pyqtgraph_plot_panel_impl import CopyAllShownBlock

        return CopyAllShownBlock(
            set_id=str(set_id),
            label=str(label),
            t=np.asarray(entry.get("t"), dtype=float).reshape(-1),
            series={
                str(name): np.asarray(values, dtype=float).reshape(-1)
                for name, values in dict(entry.get("series") or {}).items()
            },
        )

    def _copy_all_clean_shown_block(
        self,
        *,
        set_id: str,
        label: str,
        cache_key: str,
        valid_set_ids: Optional[Sequence[str]],
        invalidated_set_ids: Optional[Sequence[str]],
    ):
        if not cache_key:
            return None, "no_cached_results"
        coverage = self.results_controller.cached_batch_selection_coverage(
            cache_key=str(cache_key),
            selected_sets=[str(set_id)],
            cache_store=self._sim_controller.batch_cache.result_cache,
            valid_set_ids=valid_set_ids,
            invalidated_set_ids=invalidated_set_ids,
            allow_fallback=False,
        )
        if not coverage.available_ids:
            return None, str(coverage.reason or "no_cached_results")
        entry_result = self._cache_entry_for_set_id_from_store(
            store=self._sim_controller.batch_cache.result_cache,
            cache_key=str(cache_key),
            set_id=str(set_id),
        )
        if entry_result.entry is None:
            return None, "invalid_cache_entry" if entry_result.state == "invalid" else "no_cached_results"
        return self._copy_all_shown_block_from_entry(set_id=str(set_id), label=label, entry=entry_result.entry), None

    def _copy_all_live_primary_block(
        self,
        *,
        set_id: str,
        label: str,
    ):
        sid = str(set_id or "").strip()
        if not self.main_plot_has_data():
            return None
        if sid:
            active_set_id = str(self.active_batch_selection()[0] or "").strip()
            if active_set_id != sid:
                return None
        plot = getattr(getattr(self, "_plot_tabs", None), "_main_plot", None)
        if plot is None:
            return None
        plot_t = np.asarray(
            getattr(plot, "_t", None) if getattr(plot, "_t", None) is not None else [],
            dtype=float,
        ).reshape(-1)
        plot_series_raw = getattr(plot, "_series", {}) or {}
        if plot_t.size <= 0 or not isinstance(plot_series_raw, Mapping):
            return None
        plot_series = {
            str(name): np.asarray(values, dtype=float).reshape(-1)
            for name, values in dict(plot_series_raw).items()
            if np.asarray(values, dtype=float).reshape(-1).size > 0
        }
        if not plot_series:
            return None
        return self._copy_all_shown_block_from_entry(
            set_id=sid,
            label=label,
            entry={"t": plot_t, "series": plot_series},
        )

    def _copy_all_live_plot_shown_block(
        self,
        *,
        set_id: str,
        label: str,
        invalidated_set_ids: Optional[Sequence[str]],
    ):
        sid = str(set_id or "").strip()
        if not sid:
            return None
        invalidated = {str(raw_id) for raw_id in (invalidated_set_ids or ()) if str(raw_id)}
        if sid not in invalidated:
            return None
        plot = getattr(getattr(self, "_plot_tabs", None), "_main_plot", None)
        if plot is None:
            return None
        batch_cache = getattr(getattr(self, "_sim_controller", None), "batch_cache", None)
        active_set_id = str(batch_cache.active_batch_set_id or "").strip() if batch_cache is not None else ""

        if active_set_id == sid:
            block = self._copy_all_live_primary_block(set_id=sid, label=label)
            if block is not None:
                return block

        for entry in list(getattr(plot, "_simulation_overlays", []) or []):
            if not isinstance(entry, Mapping):
                continue
            if str(entry.get("curve_role") or "").strip() == "canonical_ghost":
                continue
            if str(entry.get("set_id") or "").strip() != sid:
                continue
            overlay_t = np.asarray(entry.get("t") if entry.get("t") is not None else [], dtype=float).reshape(-1)
            overlay_series_raw = entry.get("series") or {}
            if overlay_t.size <= 0 or not isinstance(overlay_series_raw, Mapping):
                continue
            overlay_series = {
                str(name): np.asarray(values, dtype=float).reshape(-1)
                for name, values in dict(overlay_series_raw).items()
                if np.asarray(values, dtype=float).reshape(-1).size > 0
            }
            if not overlay_series:
                continue
            return self._copy_all_shown_block_from_entry(
                set_id=sid,
                label=label,
                entry={"t": overlay_t, "series": overlay_series},
            )
        return None

    def _copy_all_dirty_shown_block(self, *, set_id: str, label: str):
        resolved_entries, reason, _, _, _, _, _ = self._resolve_workspace_aware_batch_selection(
            selected_sets=[str(set_id)]
        )
        resolved = next((entry for entry in resolved_entries if str(entry.set_id) == str(set_id)), None)
        if resolved is None or resolved.entry is None:
            return None, str(reason or "preview_pending")
        return self._copy_all_shown_block_from_entry(
            set_id=str(resolved.set_id),
            label=str(label),
            entry=resolved.entry,
        ), None

    def _build_main_plot_copy_all_export_plan(self):
        from kindred.gui.widgets.pyqtgraph_plot_panel_impl import CopyAllExportPlan, CopyAllMissingItem

        shown_set_ids = [str(set_id) for set_id in (self.shown_batch_set_ids() or []) if str(set_id)]
        live_primary_fallback_set_id = ""
        non_batch_live_primary_label = ""
        if not shown_set_ids and self.main_plot_has_data():
            active_set_id = str(self.active_batch_selection()[0] or "").strip()
            if active_set_id:
                shown_set_ids = [active_set_id]
                live_primary_fallback_set_id = active_set_id
            else:
                plot = getattr(getattr(self, "_plot_tabs", None), "_main_plot", None)
                non_batch_live_primary_label = str(
                    getattr(plot, "_simulation_set_label", "") if plot is not None else ""
                ).strip() or "Results"
        popup_labels = self._copy_all_popup_labels_by_set_id(shown_set_ids)
        batch_cache = self._sim_controller.batch_cache
        cache_key = str(batch_cache.active_cache_key or "")
        valid_set_ids = tuple(str(set_id) for set_id in (batch_cache.active_cache_valid_set_ids or ()) if str(set_id))
        invalidated_set_ids = tuple(
            str(set_id) for set_id in (batch_cache.active_cache_invalidated_set_ids or ()) if str(set_id)
        )

        shown_blocks = []
        missing_items = []
        for set_id in shown_set_ids:
            label = str(self.batch_set_name_for_id(set_id) or set_id)
            export_label = str(popup_labels.get(str(set_id), label))
            if self._preview_session.has_dirty_state_for_set(str(set_id)):
                block, reason = self._copy_all_dirty_shown_block(set_id=str(set_id), label=export_label)
            else:
                block, reason = self._copy_all_clean_shown_block(
                    set_id=str(set_id),
                    label=export_label,
                    cache_key=cache_key,
                    valid_set_ids=valid_set_ids or None,
                    invalidated_set_ids=invalidated_set_ids or None,
                )
                if block is None:
                    block = self._copy_all_live_plot_shown_block(
                        set_id=str(set_id),
                        label=export_label,
                        invalidated_set_ids=invalidated_set_ids or None,
                    )
                if block is None and str(set_id) == live_primary_fallback_set_id:
                    block = self._copy_all_live_primary_block(
                        set_id=str(set_id),
                        label=export_label,
                    )
            if block is not None:
                shown_blocks.append(block)
                continue
            missing_items.append(
                CopyAllMissingItem(
                    set_id=str(set_id),
                    label=label,
                    popup_label=str(popup_labels.get(str(set_id), label)),
                    reason=str(reason or "no_cached_results"),
                )
            )
        if non_batch_live_primary_label:
            block = self._copy_all_live_primary_block(
                set_id="",
                label=non_batch_live_primary_label,
            )
            if block is not None:
                shown_blocks.append(block)
            else:
                missing_items.append(
                    CopyAllMissingItem(
                        set_id="",
                        label=non_batch_live_primary_label,
                        popup_label=non_batch_live_primary_label,
                        reason="no_simulation_data",
                    )
                )
        return CopyAllExportPlan(shown_blocks=shown_blocks, missing_items=missing_items)

    def _set_cached_focused_batch_set_id(self, set_id: str) -> str:
        focused_set_id = str(set_id or "").strip()
        if focused_set_id and self._batch_row_for_set_id(focused_set_id) is None:
            focused_set_id = ""
        self._focused_batch_set_id = focused_set_id
        return focused_set_id

    def _update_focused_batch_set_id(self, *, row: Optional[int] = None) -> str:
        target_row = self._batch_current_row() if row is None else int(row)
        focused_set_id = ""
        if target_row is not None:
            focused_set_id = str(self._batch_set_id_for_row(int(target_row)) or "").strip()
        if not focused_set_id:
            focused_set_id = str(getattr(self, "_focused_batch_set_id", "") or "").strip()
            if focused_set_id and self._batch_row_for_set_id(focused_set_id) is None:
                focused_set_id = ""
        if (not focused_set_id) and int(self._batch_store.row_count()) > 0:
            focused_set_id = str(self._batch_set_id_for_row(0) or "").strip()
        return self._set_cached_focused_batch_set_id(focused_set_id)

    def _ensure_focused_batch_set_visible(self) -> None:
        focused_set_id = self._focused_batch_set_id_value()
        if not focused_set_id:
            return
        row = self._batch_row_for_set_id(focused_set_id)
        model = getattr(self, "_batch_model", None)
        if row is None or model is None or not hasattr(model, "set_row_shown"):
            return
        model.set_row_shown(int(row), True)

    def _next_default_batch_set_name(self) -> str:
        existing = {str(n) for n in (self._batch_store.set_names() or [])}
        counter = 1
        while True:
            candidate = f"set{counter}"
            if candidate not in existing:
                return candidate
            counter += 1

    def _add_batch_set(self) -> None:
        """Append a new batch set row with an auto-generated unique name."""
        name = self._next_default_batch_set_name()
        insert_at = int(self._batch_store.row_count())
        self._batch_model.beginInsertRows(QtCore.QModelIndex(), insert_at, insert_at)
        try:
            row_idx = int(self._batch_store.ensure_set(name))
        finally:
            self._batch_model.endInsertRows()

        table = getattr(self, "_batch_table", None)
        if table is None:
            return
        idx = self._batch_model.index(int(row_idx), 0)
        table.setCurrentIndex(idx)
        self._update_focused_batch_set_id(row=int(row_idx))
        sel = table.selectionModel()
        if sel is not None:
            signals_blocked = False
            try:
                sel.blockSignals(True)
                signals_blocked = True
            except RuntimeError as exc:
                logger.debug("Failed to block batch selection signals: %s", exc, exc_info=True)
                signals_blocked = False
            try:
                sel.clearSelection()
                sel.select(idx, QtCore.QItemSelectionModel.Select | QtCore.QItemSelectionModel.Rows)
            finally:
                if signals_blocked:
                    try:
                        sel.blockSignals(False)
                    except RuntimeError as exc:
                        logger.debug("Failed to unblock batch selection signals: %s", exc, exc_info=True)
                        signals_blocked = False
        try:
            table.scrollTo(idx)
        except RuntimeError as exc:
            logger.debug("Failed to scroll batch table to new row: %s", exc, exc_info=True)
            self._batch_table_scroll_failed = True
        self._update_batch_row_controls_state()

    def _move_selected_batch_sets_up(self) -> None:
        self._move_selected_batch_sets(delta=-1)

    def _move_selected_batch_sets_down(self) -> None:
        self._move_selected_batch_sets(delta=1)

    def _move_selected_batch_sets(self, *, delta: int) -> None:
        rows = self._batch_selected_rows()
        if len(rows) > 1:
            self._update_batch_row_controls_state()
            return
        if not rows:
            current = self._batch_current_row()
            if current is None:
                self._update_batch_row_controls_state()
                return
            rows = [int(current)]

        self._batch_model.beginResetModel()
        try:
            new_rows = self._batch_store.move_rows(rows, int(delta))
        finally:
            self._batch_model.endResetModel()
        try:
            self._batch_model.reset_invalid()
        except Exception as exc:
            logger.warning("Failed to reset invalid batch table state: %s", exc, exc_info=True)
            self._batch_invalid_state_stale = True

        table = getattr(self, "_batch_table", None)
        if table is None:
            return
        sel = table.selectionModel()
        if sel is None:
            return
        signals_blocked = False
        try:
            sel.blockSignals(True)
            signals_blocked = True
        except RuntimeError as exc:
            logger.debug("Failed to block batch selection signals: %s", exc, exc_info=True)
            signals_blocked = False
            signals_blocked = False
            signals_blocked = False
            signals_blocked = False
            signals_blocked = False
            signals_blocked = False
            signals_blocked = False
        try:
            sel.clearSelection()
            for r in new_rows:
                idx = self._batch_model.index(int(r), 0)
                sel.select(idx, QtCore.QItemSelectionModel.Select | QtCore.QItemSelectionModel.Rows)
            if new_rows:
                table.setCurrentIndex(self._batch_model.index(int(new_rows[0]), 0))
                self._update_focused_batch_set_id(row=int(new_rows[0]))
        finally:
            if signals_blocked:
                try:
                    sel.blockSignals(False)
                except RuntimeError as exc:
                    logger.debug("Failed to unblock batch selection signals: %s", exc, exc_info=True)
                    signals_blocked = False
                    signals_blocked = False
                    signals_blocked = False
                    signals_blocked = False
                    signals_blocked = False
        self._update_batch_row_controls_state()
        batch_cache = getattr(getattr(self, "_sim_controller", None), "batch_cache", None)
        active_set_id = str(self.active_batch_selection()[0] or "").strip()
        if batch_cache is not None and active_set_id:
            self.sync_main_plot_copy_labels(
                active_set_id,
                [str(set_id) for set_id in (batch_cache.last_display_selection or []) if str(set_id)],
            )

    def _batch_delete_target_rows(self) -> List[int]:
        rows = self._batch_selected_rows()
        if len(rows) >= 2:
            return [int(r) for r in rows]
        current = self._batch_current_row()
        if current is not None:
            return [int(current)]
        if rows:
            return [int(rows[0])]
        if int(self._batch_store.row_count()) > 0:
            return [0]
        return []

    def _batch_cache_contains_set(self, *, set_id: str, set_name: str) -> bool:
        sid = str(set_id or "")
        sname = str(set_name or "")
        for store in (
            self._sim_controller.batch_cache.result_cache,
            self._sim_controller.batch_cache.preview_cache,
        ):
            suffixes = []
            if sid:
                suffixes.append(f"::{sid}")
            if sname:
                suffixes.append(f"::{sname}")
            if not suffixes:
                continue
            for k in (store or {}).keys():
                token = str(k or "")
                if any(token.endswith(suf) for suf in suffixes):
                    return True
        return False

    def _datasets_mapped_to_batch_sets(self, *, set_ids: Sequence[str], set_names: Sequence[str]) -> List[str]:
        manager = getattr(self, "_dataset_manager", None)
        if manager is None:
            return []
        if hasattr(manager, "datasets_mapped_to_batch_sets"):
            return list(manager.datasets_mapped_to_batch_sets(set_ids=set_ids, set_names=set_names))
        return []

    def _confirm_delete_batch_sets(
        self,
        *,
        set_names: Sequence[str],
        mapped_datasets: Sequence[str],
        has_cached_results: bool,
        deleting_last_remaining: bool,
    ) -> bool:
        names = [str(v) for v in (set_names or []) if str(v)]
        mapped = [str(v) for v in (mapped_datasets or []) if str(v)]
        multi = len(names) > 1
        needs_confirm = bool(multi or has_cached_results or mapped or deleting_last_remaining)
        if not needs_confirm:
            return True

        lines: List[str] = []
        if multi:
            lines.append(f"Delete {len(names)} selected batch sets?")
        else:
            lines.append(f"Delete batch set '{names[0]}'?")

        if mapped:
            lines.append("")
            lines.append(f"This will unmap {len(mapped)} dataset(s).")
            if len(mapped) <= 5:
                lines.append(", ".join(mapped))
            else:
                lines.append(", ".join(mapped[:5]) + f", ... and {len(mapped) - 5} more")

        if has_cached_results:
            lines.append("")
            lines.append("Cached simulation results for deleted set(s) will be removed.")

        if deleting_last_remaining:
            lines.append("")
            lines.append("At least one set must remain. A new empty default set will be created.")

        message = "\n".join(lines)
        response = QtWidgets.QMessageBox.question(
            self,
            "Delete Batch Set(s)",
            message,
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.Cancel,
            QtWidgets.QMessageBox.StandardButton.Cancel,
        )
        return response == QtWidgets.QMessageBox.StandardButton.Yes

    def _unmap_datasets_for_deleted_batch_sets(self, *, set_ids: Sequence[str], set_names: Sequence[str]) -> List[str]:
        manager = getattr(self, "_dataset_manager", None)
        if manager is None:
            return []
        if hasattr(manager, "unmap_batch_sets"):
            return list(manager.unmap_batch_sets(set_ids=set_ids, set_names=set_names))
        return []

    def _purge_batch_cache_for_deleted_sets(self, *, set_ids: Sequence[str], set_names: Sequence[str]) -> None:
        id_targets = {str(v) for v in (set_ids or []) if str(v)}
        name_targets = {str(v) for v in (set_names or []) if str(v)}
        for store in (
            self._sim_controller.batch_cache.result_cache,
            self._sim_controller.batch_cache.preview_cache,
        ):
            for composite_key in list((store or {}).keys()):
                token = str(composite_key or "")
                if "::" not in token:
                    continue
                _prefix, sid = token.rsplit("::", 1)
                if sid in id_targets or sid in name_targets:
                    try:
                        store.pop(composite_key, None)  # type: ignore[union-attr]
                    except Exception as exc:
                        try:
                            del store[composite_key]  # type: ignore[union-attr]
                        except Exception as exc2:
                            logger.debug(
                                "Failed to purge cache key %s: %s / %s",
                                composite_key,
                                exc,
                                exc2,
                                exc_info=True,
                            )
                            continue

    def _select_single_batch_row(self, row: int) -> None:
        table = getattr(self, "_batch_table", None)
        if table is None:
            return
        if not (0 <= int(row) < int(self._batch_store.row_count())):
            return
        idx = self._batch_model.index(int(row), 0)
        table.setCurrentIndex(idx)
        self._update_focused_batch_set_id(row=int(row))
        sel = table.selectionModel()
        if sel is None:
            return
        signals_blocked = False
        try:
            sel.blockSignals(True)
            signals_blocked = True
        except RuntimeError as exc:
            logger.debug("Failed to block batch selection signals: %s", exc, exc_info=True)
            signals_blocked = False
        try:
            sel.clearSelection()
            sel.select(idx, QtCore.QItemSelectionModel.Select | QtCore.QItemSelectionModel.Rows)
        finally:
            if signals_blocked:
                try:
                    sel.blockSignals(False)
                except RuntimeError as exc:
                    logger.debug("Failed to unblock batch selection signals: %s", exc, exc_info=True)
                    signals_blocked = False
                    signals_blocked = False
        self._update_batch_row_controls_state()

    def _delete_selected_batch_sets(self) -> None:
        rows = sorted({int(r) for r in self._batch_delete_target_rows() if int(r) >= 0})
        if not rows:
            return

        names = list(self._batch_store.set_names())
        delete_names: List[str] = []
        delete_ids: List[str] = []
        for row in rows:
            if not (0 <= int(row) < len(names)):
                continue
            delete_names.append(str(names[int(row)]))
            sid = self._batch_set_id_for_row(int(row))
            if sid:
                delete_ids.append(str(sid))

        if not delete_ids:
            return

        mapped_datasets = self._datasets_mapped_to_batch_sets(set_ids=delete_ids, set_names=delete_names)
        has_cached_results = any(
            self._batch_cache_contains_set(set_id=sid, set_name=name)
            for sid, name in zip(delete_ids, delete_names)
        )
        deleting_last_remaining = len(delete_ids) >= int(self._batch_store.row_count())
        if not self._confirm_delete_batch_sets(
            set_names=delete_names,
            mapped_datasets=mapped_datasets,
            has_cached_results=bool(has_cached_results),
            deleting_last_remaining=bool(deleting_last_remaining),
        ):
            return

        if not self._guard_slider_transaction_invalidation(action_text="Deleting the selected batch set(s)"):
            return

        self._batch_model.beginResetModel()
        try:
            self._batch_store.delete_sets_by_ids(delete_ids)
            if int(self._batch_store.row_count()) <= 0:
                self._batch_store.ensure_set(self._next_default_batch_set_name())
        finally:
            self._batch_model.endResetModel()
        self._batch_model.reset_invalid()

        self._unmap_datasets_for_deleted_batch_sets(set_ids=delete_ids, set_names=delete_names)
        self._purge_batch_cache_for_deleted_sets(set_ids=delete_ids, set_names=delete_names)

        if str(self._sim_controller.batch_cache.active_batch_set_id or "") in set(delete_ids):
            self._sim_controller.batch_cache.active_batch_set_id = None
        if str(self._sim_controller.batch_cache.active_batch_set or "") in set(delete_names):
            self._sim_controller.batch_cache.active_batch_set = None

        target_row = min(rows[0], max(0, int(self._batch_store.row_count()) - 1))
        self._select_single_batch_row(target_row)
        self._update_batch_row_controls_state()
        self.results_controller.refresh_batch_plot_after_set_mutation()

    def _batch_set_names_for_scope(self, scope: str) -> List[str]:
        from kindred.core.batch_initial_conditions import resolve_run_scope

        names = list(self._batch_store.set_names())
        if not names:
            return []
        scope = str(scope or "").strip().lower()
        if scope == "all":
            return names
        rows = resolve_run_scope(
            selected_rows=self._batch_selected_rows(),
            total_rows=len(names),
            mode="selected",
            fallback_row=self._batch_current_row(),
        )
        return [names[r] for r in rows if 0 <= int(r) < len(names)]

    def _batch_set_ids_for_scope(self, scope: str) -> List[str]:
        names = self._batch_set_names_for_scope(scope)
        ids: List[str] = []
        for name in names:
            sid = self._batch_set_id_for_name(name)
            if sid:
                ids.append(str(sid))
        return ids

    def _shown_batch_rows(self) -> List[int]:
        rows: List[int] = []
        for set_id in self._shown_batch_set_ids():
            row = self._batch_row_for_set_id(set_id)
            if row is not None:
                rows.append(int(row))
        return rows

    def _batch_rows_for_scope(self, scope: str) -> List[int]:
        from kindred.core.batch_initial_conditions import resolve_run_scope

        total = int(self._batch_store.row_count())
        if total <= 0:
            return []
        scope = str(scope or "").strip().lower()
        if scope == "all":
            return list(range(total))
        return resolve_run_scope(
            selected_rows=self._batch_selected_rows(),
            total_rows=total,
            mode="selected",
            fallback_row=self._batch_current_row(),
        )

    def _update_batch_row_controls_state(self) -> None:
        selected_rows = self._batch_selected_rows()
        allow_reorder = len(selected_rows) <= 1
        if hasattr(self, "_move_batch_up_btn"):
            self._move_batch_up_btn.setEnabled(bool(allow_reorder))
        if hasattr(self, "_move_batch_down_btn"):
            self._move_batch_down_btn.setEnabled(bool(allow_reorder))
        model = getattr(self, "_batch_model", None)
        if model is not None and hasattr(model, "set_focused_effective_edit_target_set_id"):
            model.set_focused_effective_edit_target_set_id(self._focused_batch_set_id_value())

    def _batch_cache_key(
        self,
        *,
        scope_identity: object | None = None,
        mechanism_text: str,
        solver_config: Dict[str, Any],
        t_end: float,
    ) -> str:
        from kindred.core.simulation_identity import coerce_simulation_scope_identity

        structured = coerce_simulation_scope_identity(scope_identity)
        if structured is not None:
            return structured.cache_key()
        payload = {
            "dsl": str(mechanism_text or ""),
            "solver_config": dict(solver_config or {}),
            "t_end": float(t_end),
        }
        try:
            serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
        except Exception:
            serialized = repr(payload)
        return hashlib.sha256(serialized.encode("utf-8", "ignore")).hexdigest()

    def _batch_initials_for_row(self, row: int) -> Dict[str, float]:
        initials: Dict[str, float] = {}
        species = list(self._batch_store.visible_species())
        for sp in species:
            raw = str(self._batch_store.get_value(int(row), sp)).strip()
            try:
                val = float(raw)
            except Exception as exc:
                raise ValueError(f"Initial concentration for {sp} must be numeric") from exc
            if not math.isfinite(val):
                raise ValueError(f"Initial concentration for {sp} must be finite")
            initials[sp] = float(val)
        return initials

    def _batch_preferred_primary_set_id(self, rows: Sequence[int]) -> Optional[str]:
        total = int(self._batch_store.row_count())
        if total <= 0:
            return None
        focused_set_id = self._focused_batch_set_id_value()
        row_ids = {
            str(self._batch_set_id_for_row(int(row)) or "").strip()
            for row in rows or ()
            if 0 <= int(row) < total
        }
        if focused_set_id and focused_set_id in row_ids:
            return focused_set_id
        for r in rows:
            rr = int(r)
            if 0 <= rr < total:
                sid = self._batch_set_id_for_row(int(rr))
                if sid:
                    return sid
        return self._batch_set_id_for_row(0)

    def _sync_batch_species_columns(
        self,
        species_names: Sequence[str],
        *,
        preserve_active_cache: bool = False,
    ) -> None:
        """Update batch table columns to match the parsed mechanism species (no data loss)."""
        new_species = [str(s) for s in (species_names or []) if str(s)]
        if new_species == list(self._batch_store.visible_species()):
            return

        selected_sets = self._batch_set_names_for_scope("selected")
        current_row = self._batch_current_row()
        current_set = None
        if current_row is not None:
            names = list(self._batch_store.set_names())
            if 0 <= int(current_row) < len(names):
                current_set = str(names[int(current_row)])

        preview = getattr(self, "_preview_session", None)
        batch_cache = getattr(getattr(self, "_sim_controller", None), "batch_cache", None)
        active_cache_key = str(getattr(batch_cache, "active_cache_key", "") or "").strip()
        active_preview_token = str(getattr(batch_cache, "active_cache_preview_token", "") or "").strip()
        active_preview_scope_ids = tuple(
            str(set_id) for set_id in (getattr(batch_cache, "active_cache_preview_scope_set_ids", None) or ())
        )
        active_valid_set_ids = tuple(
            str(set_id) for set_id in (getattr(batch_cache, "active_cache_valid_set_ids", None) or ())
        )

        def _active_scope_overlay_token() -> str:
            if preview is None or (not active_preview_token):
                return ""
            if active_preview_scope_ids:
                scope_rows: list[int] = []
                for set_id in active_preview_scope_ids:
                    try:
                        row = getattr(self, "_batch_store", None).row_for_set_id(str(set_id))
                    except Exception:
                        row = None
                    if row is not None:
                        scope_rows.append(int(row))
                if scope_rows:
                    return str(preview.preview_batch_cache_token(scope_rows) or "")
                return ""
            if not bool(preview.has_staged_concentration_overlays()):
                return ""
            try:
                row_count = int(getattr(self, "_batch_store", None).row_count())
            except Exception:
                row_count = 0
            if row_count > 0:
                return str(preview.preview_batch_cache_token(list(range(int(row_count)))) or "")
            return ""

        def _overlay_tokens_for_set_ids(set_ids: Sequence[str]) -> dict[str, str]:
            tokens: dict[str, str] = {}
            if preview is None:
                return tokens
            for set_id in set_ids or ():
                try:
                    row = getattr(self, "_batch_store", None).row_for_set_id(str(set_id))
                except Exception:
                    row = None
                if row is None:
                    tokens[str(set_id)] = ""
                    continue
                tokens[str(set_id)] = str(preview.preview_batch_cache_token([int(row)]) or "")
            return tokens

        def _overlay_token_for_set_ids(set_ids: Sequence[str]) -> Optional[str]:
            if preview is None:
                return None
            scope_rows: list[int] = []
            for set_id in set_ids or ():
                try:
                    row = getattr(self, "_batch_store", None).row_for_set_id(str(set_id))
                except Exception:
                    row = None
                if row is not None:
                    scope_rows.append(int(row))
            if not scope_rows:
                return None
            token = str(preview.preview_batch_cache_token(scope_rows) or "")
            return token or None

        scope_tokens_before = _overlay_tokens_for_set_ids(active_preview_scope_ids)

        self._batch_model.set_species(new_species)
        prune_changed = False
        try:
            if preview is not None and hasattr(preview, "prune_staged_concentration_overlays_to_species"):
                prune_changed = bool(preview.prune_staged_concentration_overlays_to_species(new_species))
        except RuntimeError as exc:
            logger.debug("Failed to prune staged concentration overlays after batch species sync: %s", exc, exc_info=True)
        if prune_changed:
            try:
                preview.stop_species_slider_update_timer()
            except RuntimeError as exc:
                logger.debug("Failed to stop species slider timer after pruning overlays: %s", exc, exc_info=True)
            try:
                self._sim_controller.invalidate_slider_preview_work()
            except RuntimeError as exc:
                logger.debug("Failed to invalidate stale slider preview work after pruning overlays: %s", exc, exc_info=True)
        if batch_cache is not None and active_cache_key and (not bool(preserve_active_cache)):
            batch_cache.clear_active_selection_state()
        elif batch_cache is not None and active_cache_key and bool(preserve_active_cache) and active_preview_token:
            if _active_scope_overlay_token() != active_preview_token:
                scope_tokens_after = _overlay_tokens_for_set_ids(active_preview_scope_ids)
                invalidated_set_ids = {
                    str(set_id)
                    for set_id, before_token in scope_tokens_before.items()
                    if str(before_token) != str(scope_tokens_after.get(str(set_id), ""))
                }
                if invalidated_set_ids:
                    valid_ids = active_valid_set_ids or active_preview_scope_ids
                    narrowed_valid_ids = tuple(str(set_id) for set_id in valid_ids if str(set_id) not in invalidated_set_ids)
                    if not narrowed_valid_ids:
                        batch_cache.clear_active_selection_state()
                    else:
                        narrowed_valid_set = set(narrowed_valid_ids)
                        batch_cache.active_cache_valid_set_ids = narrowed_valid_ids
                        batch_cache.active_cache_invalidated_set_ids = tuple(
                            str(set_id) for set_id in invalidated_set_ids if str(set_id)
                        )
                        narrowed_scope_ids = tuple(
                            str(set_id) for set_id in active_preview_scope_ids if str(set_id) in narrowed_valid_set
                        )
                        batch_cache.active_cache_preview_scope_set_ids = narrowed_scope_ids or None
                        batch_cache.active_cache_preview_token = _overlay_token_for_set_ids(narrowed_scope_ids)
                        batch_cache.last_display_selection = [
                            str(set_id)
                            for set_id in (batch_cache.last_display_selection or [])
                            if str(set_id) in narrowed_valid_set
                        ]
                        if str(batch_cache.active_batch_set_id or "") not in narrowed_valid_set:
                            batch_cache.active_batch_set_id = None
                            batch_cache.active_batch_set = None
                else:
                    batch_cache.clear_active_selection_state()

        table = getattr(self, "_batch_table", None)
        if table is None:
            return
        sel = table.selectionModel()
        if sel is None:
            return
        signals_blocked = False
        try:
            sel.blockSignals(True)
            signals_blocked = True
        except RuntimeError as exc:
            logger.debug("Failed to block batch selection signals: %s", exc, exc_info=True)
            signals_blocked = False
        restored_focus_row: Optional[int] = None
        try:
            sel.clearSelection()
            for name in selected_sets:
                row = self._batch_store.row_for_set(name)
                if row is None:
                    continue
                idx = self._batch_model.index(int(row), 0)
                sel.select(idx, QtCore.QItemSelectionModel.Select | QtCore.QItemSelectionModel.Rows)
            if current_set:
                row = self._batch_store.row_for_set(current_set)
                if row is not None:
                    table.setCurrentIndex(self._batch_model.index(int(row), 0))
                    restored_focus_row = int(row)
        finally:
            if signals_blocked:
                try:
                    sel.blockSignals(False)
                except RuntimeError as exc:
                    logger.debug("Failed to unblock batch selection signals: %s", exc, exc_info=True)
                    signals_blocked = False
        self._update_focused_batch_set_id(row=restored_focus_row)

        try:
            panel = self._mechanism_editor.species_sliders_widget()
            if panel is not None and hasattr(panel, "rebuild_from_current_row"):
                panel.rebuild_from_current_row()
        except RuntimeError as exc:
            logger.debug("Failed to rebuild species panel after batch species sync: %s", exc, exc_info=True)
            self._species_panel_available = False

    def _on_batch_selection_changed(self, *_args) -> None:
        self._update_batch_row_controls_state()

    def _on_batch_current_changed(self, *_args) -> None:
        focused_set_id = self._update_focused_batch_set_id()
        if not focused_set_id:
            return
        self._ensure_focused_batch_set_visible()
        self._refresh_slider_edit_targets_summary()
        self._refresh_batch_display_from_focus_and_shown()

    def _on_batch_show_membership_changed(self) -> None:
        focused_set_id = self._focused_batch_set_id_value()
        if focused_set_id:
            row = self._batch_row_for_set_id(focused_set_id)
            if row is not None and not self._batch_store.is_shown(int(row)):
                self._batch_model.set_row_shown(int(row), True)
                return
        self._refresh_batch_display_from_focus_and_shown()

    def _on_slider_edit_targets_changed(self) -> None:
        self._refresh_slider_edit_targets_summary()
        try:
            panel = self._mechanism_editor.species_sliders_widget()
            if panel is not None and hasattr(panel, "rebuild_from_current_row"):
                panel.rebuild_from_current_row()
        except RuntimeError as exc:
            logger.debug("Failed to rebuild species panel after slider target change: %s", exc, exc_info=True)
            self._species_panel_available = False

    def _refresh_slider_edit_targets_summary(self) -> None:
        editor = getattr(self, "_mechanism_editor", None)
        if editor is None or not hasattr(editor, "set_slider_edit_targets_summary"):
            return
        effective_target_ids = self._effective_slider_edit_target_set_ids()
        primary_label = (
            self.batch_set_name_for_id(effective_target_ids[0])
            if effective_target_ids
            else None
        )
        explicit_extra_count = 0
        if effective_target_ids:
            primary_set_id = str(effective_target_ids[0] or "")
            explicit_extra_count = sum(
                1
                for set_id in self._slider_edit_target_set_ids()
                if str(set_id or "") and str(set_id or "") != primary_set_id
            )
        if not primary_label:
            summary = "Slider edit targets: none"
        elif explicit_extra_count <= 0:
            summary = f"Slider edit targets: {primary_label}"
        else:
            summary = f"Slider edit targets: {primary_label} + {explicit_extra_count} explicit"
        editor.set_slider_edit_targets_summary(summary)

    def _refresh_batch_display_from_focus_and_shown(self) -> None:
        self._update_batch_row_controls_state()
        batch_cache = self._sim_controller.batch_cache
        shown_sets = self._shown_batch_set_ids()
        if not shown_sets:
            self._clear_batch_selection_display_state()
            self._sync_mechanism_controls_to_focused_batch_set(
                use_workspace=bool(
                    self._preview_session.has_dirty_state_for_set(self._focused_batch_set_id_value())
                )
            )
            return
        prefer = None
        focused_set_id = self._focused_batch_set_id_value()
        if focused_set_id:
            prefer = focused_set_id
        focused_selection_is_dirty = self._focused_batch_selection_is_dirty(
            selected_sets=shown_sets,
            prefer_set=prefer,
        )
        active_cache_key = str(batch_cache.active_cache_key or "").strip()

        outcome = None
        outcome_reason = None
        miss_msg = "Result not cached (evicted). Press Run to compute."
        invalid_msg = "Cached result invalid. Press Run to compute."
        preview_pending_msg = "Preview pending for current selection."

        def _reset_stale_cache_warning_status() -> None:
            label = getattr(self, "_status_label", None)
            if label is None:
                return
            try:
                if str(label.text()) in (miss_msg, invalid_msg, preview_pending_msg):
                    label.setText("Ready")
            except RuntimeError as exc:
                logger.debug("Failed to update status label: %s", exc, exc_info=True)
                self._status_label = None

        def _set_selection_status(text: str) -> None:
            label = getattr(self, "_status_label", None)
            if label is None:
                return
            try:
                label.setText(str(text))
            except RuntimeError as exc:
                logger.debug("Failed to update status label: %s", exc, exc_info=True)
                self._status_label = None

        def _clear_non_displayed_selection_state() -> None:
            self._clear_batch_selection_display_state()

        def _finalize_displayed_selection_change() -> None:
            self._record_current_main_plot_workspace_preview_provenance(selected_set_ids=shown_sets)
            _reset_stale_cache_warning_status()

        if focused_selection_is_dirty and self._selection_uses_fresh_explicit_cache_after_post_run_sync(
            selected_sets=shown_sets
        ):
            valid_set_ids = None
            if batch_cache.active_cache_valid_set_ids is not None:
                valid_set_ids = tuple(str(set_id) for set_id in batch_cache.active_cache_valid_set_ids if str(set_id))
            invalidated_set_ids = None
            if batch_cache.active_cache_invalidated_set_ids is not None:
                invalidated_set_ids = tuple(
                    str(set_id) for set_id in batch_cache.active_cache_invalidated_set_ids if str(set_id)
                )
            outcome = self.results_controller.display_cached_batch_selection_outcome(
                cache_key=active_cache_key,
                selected_sets=shown_sets,
                prefer_set=prefer,
                valid_set_ids=valid_set_ids,
                invalidated_set_ids=invalidated_set_ids,
                allow_fallback=False,
            )
            if outcome.displayed:
                self._sync_mechanism_controls_to_focused_batch_set(use_workspace=True)
                _reset_stale_cache_warning_status()
                return
            if outcome.reason == "invalid_cache_entry":
                _set_selection_status(invalid_msg)
                return
            _set_selection_status(miss_msg)
            return

        (
            resolved_entries,
            outcome_reason,
            all_selected_sets_resolved,
            has_workspace_selection,
            has_resolved_workspace_preview,
            focused_selection_uses_workspace_controls,
            focused_selection_has_resolved_entry,
        ) = (
            self._resolve_workspace_aware_batch_selection(selected_sets=shown_sets)
        )
        if all_selected_sets_resolved and resolved_entries:
            outcome = self.results_controller.display_resolved_batch_selection_outcome(
                resolved_entries=resolved_entries,
                prefer_set=prefer,
            )
            outcome_reason = outcome.reason
            if outcome.displayed:
                self._sync_mechanism_controls_to_focused_batch_set(
                    use_workspace=bool(focused_selection_uses_workspace_controls)
                )
                _finalize_displayed_selection_change()
                return

        if resolved_entries and has_workspace_selection:
            if (
                outcome_reason == "preview_pending"
                and has_resolved_workspace_preview
                and (
                    bool(focused_selection_uses_workspace_controls)
                    or ((not bool(focused_selection_is_dirty)) and bool(focused_selection_has_resolved_entry))
                )
            ):
                outcome = self.results_controller.display_resolved_batch_selection_outcome(
                    resolved_entries=resolved_entries,
                    prefer_set=prefer,
                )
                if outcome.displayed:
                    self._record_current_main_plot_workspace_preview_provenance(selected_set_ids=shown_sets)
                    self._sync_mechanism_controls_to_focused_batch_set(
                        use_workspace=bool(focused_selection_uses_workspace_controls)
                    )
                    _set_selection_status(preview_pending_msg)
                    return
                _clear_non_displayed_selection_state()
                self._sync_mechanism_controls_to_focused_batch_set(
                    use_workspace=bool(focused_selection_is_dirty)
                )
                _set_selection_status(preview_pending_msg)
                return
            if (
                outcome_reason == "no_cached_results"
                and has_resolved_workspace_preview
                and (
                    bool(focused_selection_uses_workspace_controls)
                    or ((not bool(focused_selection_is_dirty)) and bool(focused_selection_has_resolved_entry))
                )
            ):
                outcome = self.results_controller.display_resolved_batch_selection_outcome(
                    resolved_entries=resolved_entries,
                    prefer_set=prefer,
                )
                if outcome.displayed:
                    self._sync_mechanism_controls_to_focused_batch_set(
                        use_workspace=bool(focused_selection_uses_workspace_controls)
                    )
                    _finalize_displayed_selection_change()
                    if not bool(focused_selection_uses_workspace_controls):
                        _set_selection_status(miss_msg)
                    return
            if outcome_reason == "no_cached_results":
                _clear_non_displayed_selection_state()
                self._sync_mechanism_controls_to_focused_batch_set(
                    use_workspace=bool(focused_selection_is_dirty)
                )
                _set_selection_status(miss_msg)
                return

        if (not has_workspace_selection) and outcome_reason in {"preview_pending", "no_cached_results"} and active_cache_key:
            valid_set_ids = None
            if batch_cache.active_cache_valid_set_ids is not None:
                valid_set_ids = tuple(str(set_id) for set_id in batch_cache.active_cache_valid_set_ids if str(set_id))
            invalidated_set_ids = None
            if batch_cache.active_cache_invalidated_set_ids is not None:
                invalidated_set_ids = tuple(
                    str(set_id) for set_id in batch_cache.active_cache_invalidated_set_ids if str(set_id)
                )
            outcome = self.results_controller.display_cached_batch_selection_outcome(
                cache_key=active_cache_key,
                selected_sets=shown_sets,
                prefer_set=prefer,
                valid_set_ids=valid_set_ids,
                invalidated_set_ids=invalidated_set_ids,
                allow_fallback=False,
            )
            if outcome.displayed:
                self._sync_mechanism_controls_to_focused_batch_set(use_workspace=False)
                _finalize_displayed_selection_change()
                return
            if outcome_reason != "invalid_cache_entry" and outcome.reason == "invalid_cache_entry":
                outcome_reason = "invalid_cache_entry"

        if (not has_workspace_selection) and (not active_cache_key):
            _clear_non_displayed_selection_state()
            self._sync_mechanism_controls_to_focused_batch_set(use_workspace=False)
            _reset_stale_cache_warning_status()
            return

        if outcome_reason == "invalid_cache_entry":
            _clear_non_displayed_selection_state()
            self._sync_mechanism_controls_to_focused_batch_set(
                use_workspace=bool(focused_selection_is_dirty)
            )
            _set_selection_status(invalid_msg)
            return
        if outcome_reason == "preview_pending":
            _clear_non_displayed_selection_state()
            self._sync_mechanism_controls_to_focused_batch_set(
                use_workspace=bool(focused_selection_is_dirty)
            )
            _set_selection_status(preview_pending_msg)
            return
        _clear_non_displayed_selection_state()
        self._sync_mechanism_controls_to_focused_batch_set(
            use_workspace=bool(focused_selection_is_dirty)
        )
        _set_selection_status(miss_msg)

    def _cache_entry_for_set_id_from_store(
        self,
        *,
        store: MutableMapping[str, Dict[str, Any]],
        cache_key: str,
        set_id: str,
    ) -> BatchCacheEntryReadResult:
        sid = str(set_id or "").strip()
        if not sid or not cache_key:
            return BatchCacheEntryReadResult("missing")
        direct = read_batch_cache_entry((store or {}).get(BatchSimulationCache.entry_key(cache_key, sid)))
        if direct.entry is not None:
            return direct
        name = self.batch_set_name_for_id(sid)
        by_name = BatchCacheEntryReadResult("missing")
        if name:
            by_name = read_batch_cache_entry((store or {}).get(BatchSimulationCache.entry_key(cache_key, str(name))))
            if by_name.entry is not None:
                return by_name
        if direct.state == "invalid" or by_name.state == "invalid":
            return BatchCacheEntryReadResult("invalid")
        return BatchCacheEntryReadResult("missing")

    def _mechanism_text_for_workspace_selection(self, *, set_id: str) -> str:
        from kindred.core.batch_initial_conditions import (
            strip_reaction_dsl_initial_concentrations,
        )

        reactions_text = self.mechanism_reactions_text_raw()
        if self.has_slider_overrides():
            reactions_text = self._apply_overrides_to_text(reactions_text, set_id=str(set_id))
        reactions_text = strip_reaction_dsl_initial_concentrations(reactions_text)

        state_network_dsl = self.mechanism_state_network_dsl_raw()
        if self.has_slider_overrides():
            state_network_dsl = self._apply_overrides_to_state_network_dsl(state_network_dsl, set_id=str(set_id))

        full_dsl = reactions_text
        if state_network_dsl.strip():
            full_dsl += "\n\n# State Network\n" + state_network_dsl
        if self.has_slider_overrides():
            full_dsl = self._apply_parameter_overrides_to_dsl(
                full_dsl,
                self._normalized_slider_overrides(set_id=str(set_id)),
            )
        return str(full_dsl)

    def _current_workspace_preview_context(
        self,
        *,
        set_id: str,
        mechanism_text: str,
    ) -> tuple[Dict[str, Any], float, str]:
        from kindred.gui.controllers.simulation_controller import build_fast_preview_solver_grid_context

        solver_grid_context = build_fast_preview_solver_grid_context(
            initial_solver_name=self._initial_solver,
            num_points=int(self.num_points_spinbox_value()),
            fast_mode=True,
            slider_points_override=self.mechanism_slider_points_value(),
            slider_solver_override=self.mechanism_slider_solver_value(),
            slider_drag_active=bool(self._preview_session.slider_drag_active()),
            last_slider_change_name=str(self._preview_session.last_slider_change_name() or ""),
        )
        temperature_k = float(self.temperature_spinbox_value())
        if "\n\n# State Network\n" in str(mechanism_text):
            t_override = self.dsl_global_temperature_K(str(mechanism_text))
            if t_override is not None:
                temperature_k = float(t_override)
        solver_config = {
            "solver": str(solver_grid_context.get("solver") or ""),
            "solver_label": str(solver_grid_context.get("solver_label") or ""),
            "solver_warning": (
                str(solver_grid_context.get("solver_warning"))
                if solver_grid_context.get("solver_warning")
                else None
            ),
            "rtol": self._initial_rtol or 1e-6,
            "atol": self._initial_atol or 1e-12,
            "grid": dict(solver_grid_context.get("grid") or {"N": int(self.num_points_spinbox_value())}),
            "temperature_K": float(temperature_k),
            "use_sparse_jacobian": bool(self.use_sparse_jacobian()),
            "wegscheider_cyclicity_enabled": bool(self.wegscheider_cyclicity_enabled()),
        }
        overlay_token = ""
        try:
            row = getattr(self, "_batch_store", None).row_for_set_id(str(set_id))
        except Exception:
            row = None
        if row is not None:
            overlay_token = str(self._preview_session.preview_batch_cache_token([int(row)]) or "")
        return solver_config, float(self.parse_sim_time_seconds()), overlay_token

    def _current_workspace_preview_identity(self, *, set_id: str):
        from kindred.core.simulation_identity import SimulationIdentity

        expected_solver_config, expected_t_end, expected_overlay_token = self._current_workspace_preview_context(
            set_id=str(set_id),
            mechanism_text=self._mechanism_text_for_workspace_selection(set_id=str(set_id)),
        )
        return SimulationIdentity.build(
            schema_id=self.simulation_schema_id(),
            param_fingerprint=self.simulation_param_fingerprint(set_id=str(set_id)),
            solver_config=expected_solver_config,
            t_end=expected_t_end,
            preview_batch_cache_token=expected_overlay_token,
            execution_flags=("fast_mode",),
        )

    def _matching_preview_entry_for_workspace_set(
        self,
        *,
        set_id: str,
        preview_cache_key: Optional[str] = None,
    ) -> BatchCacheEntryReadResult:
        from kindred.core.simulation_identity import coerce_simulation_identity

        preview_store = self._sim_controller.batch_cache.preview_cache
        expected_mechanism_text = self._mechanism_text_for_workspace_selection(set_id=str(set_id))
        resolved_preview_cache_key = str(
            preview_cache_key
            if preview_cache_key is not None
            else (self._sim_controller.batch_cache.active_preview_cache_key or "")
        ).strip()

        try:
            expected_identity = self._current_workspace_preview_identity(set_id=str(set_id))
            expected_solver_config, expected_t_end, expected_overlay_token = self._current_workspace_preview_context(
                set_id=str(set_id),
                mechanism_text=str(expected_mechanism_text),
            )
        except Exception:
            return BatchCacheEntryReadResult("missing")

        def _entry_matches_expected(result: BatchCacheEntryReadResult) -> bool:
            if result.entry is None:
                return False
            entry_identity = coerce_simulation_identity(result.entry.get("simulation_identity"))
            if entry_identity is not None:
                if entry_identity != expected_identity:
                    return False
            else:
                if str(result.entry.get("mechanism_text") or "") != str(expected_mechanism_text):
                    return False
                if dict(result.entry.get("solver_config") or {}) != dict(expected_solver_config):
                    return False
                if str(result.entry.get("preview_batch_cache_token") or "") != str(expected_overlay_token):
                    return False
            entry_t_payload = result.entry.get("t")
            entry_t = np.asarray(entry_t_payload if entry_t_payload is not None else [], dtype=float).reshape(-1)
            expected_grid_n = int((expected_solver_config.get("grid") or {}).get("N") or 0)
            if expected_grid_n > 0 and int(entry_t.size) != expected_grid_n:
                return False
            if entry_t.size <= 0:
                return False
            return math.isclose(float(entry_t[-1]), float(expected_t_end), rel_tol=1e-9, abs_tol=1e-12)

        invalid_found = False
        direct = self._cache_entry_for_set_id_from_store(
            store=preview_store,
            cache_key=resolved_preview_cache_key,
            set_id=str(set_id),
        )
        if _entry_matches_expected(direct):
            return direct
        if direct.state == "invalid":
            invalid_found = True

        candidate_suffixes = {f"::{str(set_id)}"}
        set_name = self.batch_set_name_for_id(str(set_id))
        if set_name:
            candidate_suffixes.add(f"::{str(set_name)}")

        preview_data = getattr(preview_store, "_data", None)
        if hasattr(preview_data, "items"):
            preview_items = list(preview_data.items())
        else:
            preview_items = list((preview_store or {}).items())

        for key, payload in reversed(preview_items):
            key_s = str(key)
            if not any(key_s.endswith(suffix) for suffix in candidate_suffixes):
                continue
            if resolved_preview_cache_key and key_s.startswith(f"{resolved_preview_cache_key}::"):
                continue
            result = read_batch_cache_entry(payload)
            if _entry_matches_expected(result):
                return result
            invalid_found = invalid_found or result.state == "invalid"

        return BatchCacheEntryReadResult("invalid" if invalid_found else "missing")

    def _resolve_workspace_aware_batch_selection(
        self,
        *,
        selected_sets: Sequence[str],
        preview_cache_key: Optional[str] = None,
    ) -> Tuple[List[ResolvedBatchSelectionEntry], Optional[str], bool, bool, bool, bool, bool]:
        from kindred.gui.controllers.results_controller import ResolvedBatchSelectionEntry

        batch_cache = self._sim_controller.batch_cache
        active_cache_key = str(batch_cache.active_cache_key or "").strip()
        invalidated_set_ids = {
            str(set_id) for set_id in (getattr(batch_cache, "active_cache_invalidated_set_ids", None) or ()) if str(set_id)
        }
        focused_set_id = str(self._focused_batch_set_id_value() or "").strip()
        if (not focused_set_id) and selected_sets:
            focused_set_id = str(selected_sets[0] or "").strip()

        resolved_entries: List[ResolvedBatchSelectionEntry] = []
        has_workspace_selection = False
        has_resolved_workspace_preview = False
        focused_selection_uses_workspace_controls = False
        focused_selection_has_resolved_entry = False
        missing_workspace_entry = False
        missing_explicit_entry = False
        invalid_entry = False

        for raw_set_id in selected_sets or ():
            set_id = str(raw_set_id or "").strip()
            if not set_id:
                continue
            label = self.batch_set_name_for_id(set_id) or set_id
            if self._preview_session.has_dirty_state_for_set(set_id):
                has_workspace_selection = True
                preview_entry = self._matching_preview_entry_for_workspace_set(
                    set_id=set_id,
                    preview_cache_key=preview_cache_key,
                )
                if preview_entry.entry is not None:
                    has_resolved_workspace_preview = True
                    canonical_entry = None
                    if active_cache_key and set_id not in invalidated_set_ids:
                        explicit_entry = self._cache_entry_for_set_id_from_store(
                            store=batch_cache.result_cache,
                            cache_key=active_cache_key,
                            set_id=set_id,
                        )
                        canonical_entry = explicit_entry.entry
                    resolved_entries.append(
                        ResolvedBatchSelectionEntry(
                            set_id=str(set_id),
                            label=str(label),
                            entry=preview_entry.entry,
                            canonical_entry=canonical_entry,
                        )
                    )
                    if set_id == focused_set_id:
                        focused_selection_uses_workspace_controls = True
                        focused_selection_has_resolved_entry = True
                elif preview_entry.state == "invalid":
                    invalid_entry = True
                else:
                    missing_workspace_entry = True
                continue

            if not active_cache_key:
                missing_explicit_entry = True
                continue
            explicit_entry = self._cache_entry_for_set_id_from_store(
                store=batch_cache.result_cache,
                cache_key=active_cache_key,
                set_id=set_id,
            )
            if set_id in invalidated_set_ids:
                if explicit_entry.state == "invalid":
                    invalid_entry = True
                else:
                    missing_explicit_entry = True
                continue
            if explicit_entry.entry is not None:
                resolved_entries.append(
                    ResolvedBatchSelectionEntry(set_id=str(set_id), label=str(label), entry=explicit_entry.entry)
                )
                if set_id == focused_set_id:
                    focused_selection_uses_workspace_controls = False
                    focused_selection_has_resolved_entry = True
            elif explicit_entry.state == "invalid":
                invalid_entry = True
            else:
                missing_explicit_entry = True

        all_selected_sets_resolved = len(resolved_entries) == len(
            [str(set_id) for set_id in (selected_sets or ()) if str(set_id)]
        )
        if invalid_entry:
            return (
                resolved_entries,
                "invalid_cache_entry",
                all_selected_sets_resolved,
                has_workspace_selection,
                has_resolved_workspace_preview,
                focused_selection_uses_workspace_controls,
                focused_selection_has_resolved_entry,
            )
        if missing_workspace_entry:
            return (
                resolved_entries,
                "preview_pending",
                all_selected_sets_resolved,
                has_workspace_selection,
                has_resolved_workspace_preview,
                focused_selection_uses_workspace_controls,
                focused_selection_has_resolved_entry,
            )
        if missing_explicit_entry:
            return (
                resolved_entries,
                "no_cached_results",
                all_selected_sets_resolved,
                has_workspace_selection,
                has_resolved_workspace_preview,
                focused_selection_uses_workspace_controls,
                focused_selection_has_resolved_entry,
            )
        return (
            resolved_entries,
            None,
            all_selected_sets_resolved,
            has_workspace_selection,
            has_resolved_workspace_preview,
            focused_selection_uses_workspace_controls,
            focused_selection_has_resolved_entry,
        )

    def _normalized_slider_overrides(
        self,
        *,
        set_id: Optional[str] = None,
        overrides: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        raw = self.slider_overrides(set_id=set_id) if overrides is None else dict(overrides or {})
        normalized: Dict[str, float] = {}
        for key, value in raw.items():
            parsed, ok = try_parse_finite_float(value)
            if not ok:
                continue
            normalized[str(key)] = float(parsed)
        return normalized

    def _apply_overrides_to_text(
        self,
        base_text: str,
        *,
        set_id: Optional[str] = None,
        overrides: Optional[Dict[str, float]] = None,
    ) -> str:
        """Return mechanism DSL with slider overrides applied."""
        drag_baseline_text = self._preview_session.drag_baseline_text()
        if self._preview_session.slider_drag_active() and drag_baseline_text is not None:
            text = drag_baseline_text
        else:
            text = base_text
        override_map = self._normalized_slider_overrides(set_id=set_id, overrides=overrides)
        has_energy_overrides = False
        metadata = self.variable_metadata() or {}
        step_analysis_context = None
        step_constraint_context = {
            "temperature_K": float(self._temperature_spinbox.value()),
            "wegscheider_cyclicity_enabled": bool(self.wegscheider_cyclicity_enabled()),
        }
        for var_name, var_value in override_map.items():
            meta = metadata.get(var_name, {})
            if isinstance(meta, dict) and meta.get("type") == "scalar":
                continue
            if isinstance(meta, dict) and meta.get("type") == "energy":
                has_energy_overrides = True
                continue
            if re.match(r"^(kf|kr|K)\d+$", str(var_name)):
                if step_analysis_context is None or step_analysis_context.source_text != text:
                    step_analysis_context = build_current_text_step_analysis_context(
                        text,
                        step_constraint_context=step_constraint_context,
                    )
            previous_text = text
            text = self._update_variable_in_mechanism(
                var_name,
                var_value,
                source_text=text,
                commit=False,
                metadata=metadata,
                step_analysis_context=step_analysis_context,
            )
            if text != previous_text:
                step_analysis_context = None
        if has_energy_overrides:
            text = self._apply_energy_overrides_to_inline_state_network(text, overrides=override_map)
            text = self._apply_energy_overrides_to_computational_mode_fast_equilibria(text, overrides=override_map)
        return text

    def _collect_energy_overrides(
        self,
        *,
        set_id: Optional[str] = None,
        overrides: Optional[Dict[str, float]] = None,
    ) -> list[tuple[str, float, dict]]:
        meta_map = self.variable_metadata() or {}
        override_map = self._normalized_slider_overrides(set_id=set_id, overrides=overrides)
        energy_overrides: list[tuple[str, float, dict]] = []
        for name, value in override_map.items():
            meta = meta_map.get(name)
            if not (isinstance(meta, dict) and meta.get("type") == "energy"):
                continue
            try:
                energy_overrides.append((str(name), float(value), dict(meta)))
            except (TypeError, ValueError) as exc:
                self._record_best_effort_failure(
                    "main_window.energy_overrides.parse",
                    message="Skipping invalid energy override value",
                    exc=exc,
                )
        return energy_overrides

    @staticmethod
    def _infer_global_energy_unit(lines: list[str], overrides: list[tuple[str, float, dict]]) -> str:
        global_unit = None
        for raw in lines:
            before_comment, _, _comment = raw.partition("#")
            stripped = before_comment.strip()
            if not stripped:
                continue
            if not stripped.lower().startswith("energy="):
                continue
            _key, _eq, rest = stripped.partition("=")
            unit_val = rest.strip().split()[0] if rest.strip() else ""
            if unit_val in ("kJ/mol", "kcal/mol", "J/mol"):
                global_unit = unit_val
        if global_unit is None:
            for _name, _value, meta in overrides:
                unit_val = meta.get("unit")
                if unit_val:
                    global_unit = str(unit_val)
                    break
        return str(global_unit or "kJ/mol")

    def _parse_inline_state_network_energy(
        self,
        *,
        lines: list[str],
        global_unit: str,
        normalize_energy_to_J_per_mol,
    ) -> tuple[dict[str, int], dict[str, str], dict[str, float]]:
        state_row_index: dict[str, int] = {}
        state_unit_out: dict[str, str] = {}
        energy_jmol: dict[str, float] = {}

        for idx, raw in enumerate(lines):
            before_comment, _, _comment = raw.partition("#")
            stripped = before_comment.strip()
            if not stripped.lower().startswith("state:"):
                continue
            _prefix, rest = stripped.split(":", 1)
            parts = [p.strip() for p in re.split(r"[;,]", rest) if p.strip()]
            name: str | None = None
            kv: dict[str, str] = {}
            for part in parts:
                if "=" in part:
                    k, v = part.split("=", 1)
                    kv[k.strip().lower()] = v.strip()
                elif name is None:
                    name = part.strip()
            if not name:
                name = kv.get("name") or kv.get("state")
            if not name:
                continue

            unit_line = (kv.get("energy_unit") or global_unit or "kJ/mol").strip()
            energy_val_str = kv.get("energy")

            ej = 0.0
            explicit_unit: str | None = None
            if energy_val_str is not None and str(energy_val_str).strip() != "":
                evs = str(energy_val_str).strip()
                if evs in ("kJ/mol", "kcal/mol", "J/mol"):
                    evs = ""
                if evs:
                    tok = evs.split()
                    if len(tok) == 2 and tok[1] in ("kJ/mol", "kcal/mol", "J/mol"):
                        explicit_unit = tok[1]
                        try:
                            v0 = float(tok[0])
                        except Exception:
                            v0 = None
                        if v0 is not None:
                            ej = normalize_energy_to_J_per_mol(v0, explicit_unit)
                    else:
                        try:
                            v0 = float(tok[0])
                        except Exception:
                            v0 = None
                        if v0 is not None:
                            ej = normalize_energy_to_J_per_mol(v0, unit_line)

            state_row_index[name] = int(idx)
            state_unit_out[name] = str(explicit_unit or unit_line or global_unit or "kJ/mol")
            energy_jmol[name] = float(ej)

        return state_row_index, state_unit_out, energy_jmol

    def _apply_energy_overrides_to_state_energies(
        self,
        *,
        energy_jmol: dict[str, float],
        overrides: list[tuple[str, float, dict]],
        global_unit: str,
        UnitsModel,
    ) -> set[str]:
        dG_eq_overrides = sorted([o for o in overrides if str(o[2].get("role") or "") == "dG_eq"], key=lambda o: o[0])
        dG_act_overrides = sorted([o for o in overrides if str(o[2].get("role") or "") == "dG_act_fwd"], key=lambda o: o[0])

        touched: set[str] = set()

        for _name, value, meta in dG_eq_overrides:
            reactant = str(meta.get("reactant") or "")
            product = str(meta.get("product") or "")
            unit_in = str(meta.get("unit") or global_unit or "kJ/mol")
            if reactant not in energy_jmol or product not in energy_jmol:
                continue
            try:
                dG_j = float(UnitsModel(energy_unit=unit_in).to_jmol(float(value)))
            except Exception as exc:
                self._record_best_effort_failure(
                    "main_window.inline_state_network.to_jmol.dG_eq",
                    message=f"Failed to convert dG_eq energy override to J/mol for '{reactant}->{product}'",
                    exc=exc,
                )
                continue
            new_e = float(energy_jmol[reactant] + dG_j)
            if not math.isclose(new_e, float(energy_jmol[product]), rel_tol=0.0, abs_tol=1e-6):
                energy_jmol[product] = new_e
                touched.add(product)

        for _name, value, meta in dG_act_overrides:
            reactant = str(meta.get("reactant") or "")
            ts = str(meta.get("ts") or "")
            unit_in = str(meta.get("unit") or global_unit or "kJ/mol")
            if reactant not in energy_jmol or ts not in energy_jmol:
                continue
            try:
                dG_j = float(UnitsModel(energy_unit=unit_in).to_jmol(float(value)))
            except Exception as exc:
                self._record_best_effort_failure(
                    "main_window.inline_state_network.to_jmol.dG_act_fwd",
                    message=f"Failed to convert dG_act_fwd energy override to J/mol for '{reactant}->{ts}'",
                    exc=exc,
                )
                continue
            new_e = float(energy_jmol[reactant] + dG_j)
            if not math.isclose(new_e, float(energy_jmol[ts]), rel_tol=0.0, abs_tol=1e-6):
                energy_jmol[ts] = new_e
                touched.add(ts)

        return touched

    def _rewrite_inline_state_network_energy_lines(
        self,
        *,
        lines: list[str],
        state_row_index: dict[str, int],
        state_unit_out: dict[str, str],
        energy_jmol: dict[str, float],
        touched: set[str],
        global_unit: str,
        UnitsModel,
    ) -> bool:
        energy_pat = re.compile(
            r"(\benergy\s*=\s*)([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)(\s*(?:kJ/mol|kcal/mol|J/mol))?",
            flags=re.IGNORECASE,
        )

        changed = False
        for name in sorted(touched):
            idx = state_row_index.get(name)
            if idx is None:
                continue
            raw = lines[idx]
            before_comment, hash_sep, comment_rest = raw.partition("#")
            unit_out = str(state_unit_out.get(name) or global_unit or "kJ/mol")
            try:
                new_val = float(UnitsModel(energy_unit=unit_out).from_jmol(float(energy_jmol[name])))
            except Exception as exc:
                self._record_best_effort_failure(
                    "main_window.inline_state_network.from_jmol",
                    message=f"Failed to convert energy back to '{unit_out}' for state '{name}'",
                    exc=exc,
                )
                continue
            new_val_str = f"{new_val:.6g}"

            def _repl(m: re.Match[str]) -> str:
                prefix = m.group(1)
                suffix = m.group(3) or ""
                if suffix:
                    return f"{prefix}{new_val_str}{suffix}"
                return f"{prefix}{new_val_str}"

            updated_before, n = energy_pat.subn(_repl, before_comment, count=1)
            if n == 0:
                continue
            updated_line = updated_before + (hash_sep + comment_rest if hash_sep else "")
            if updated_line != raw:
                lines[idx] = updated_line
                changed = True

        return changed

    @staticmethod
    def _parse_semicolon_kv_tokens(code: str) -> tuple[str, list[list[str]]]:
        chunks = [c.strip() for c in str(code or "").split(";") if c.strip()]
        if not chunks:
            return "", []
        prefix = chunks[0].strip()
        tokens: list[list[str]] = []
        for chunk in chunks[1:]:
            for part in [piece.strip() for piece in str(chunk).split(",") if piece.strip()]:
                if "=" not in part:
                    continue
                k, v = part.split("=", 1)
                tokens.append([k.strip(), v.strip()])
        return prefix, tokens

    @staticmethod
    def _get_semicolon_kv(tokens: list[list[str]], key: str) -> str | None:
        for k, v in tokens:
            if str(k).strip() == key:
                return str(v).strip()
        return None

    @staticmethod
    def _set_semicolon_kv(tokens: list[list[str]], key: str, value: str) -> None:
        for pair in tokens:
            if str(pair[0]).strip() == key:
                pair[1] = str(value)
                return
        tokens.append([str(key), str(value)])

    def _apply_energy_overrides_to_inline_state_network(
        self,
        base_text: str,
        *,
        set_id: Optional[str] = None,
        overrides: Optional[Dict[str, float]] = None,
    ) -> str:
        """
        Apply energy-mode slider overrides (ΔG‡_fwd and ΔG°) to any `state:` lines present in `base_text`.

        This supports both comma-style and semicolon-style state-network DSL,
        and preserves line formatting by only rewriting the numeric energy token.
        """
        energy_overrides = self._collect_energy_overrides(set_id=set_id, overrides=overrides)
        if not energy_overrides:
            return base_text

        try:
            from kindred.core.simulator.kinetics import normalize_energy_to_J_per_mol
            from kindred.core.units import UnitsModel
        except Exception:
            return base_text

        text = str(base_text or "")
        if not text.strip():
            return base_text
        lines = text.splitlines()

        global_unit = self._infer_global_energy_unit(lines, energy_overrides)
        state_row_index, state_unit_out, energy_jmol = self._parse_inline_state_network_energy(
            lines=lines,
            global_unit=global_unit,
            normalize_energy_to_J_per_mol=normalize_energy_to_J_per_mol,
        )
        if not state_row_index:
            return base_text

        touched = self._apply_energy_overrides_to_state_energies(
            energy_jmol=energy_jmol,
            overrides=energy_overrides,
            global_unit=global_unit,
            UnitsModel=UnitsModel,
        )
        if not touched:
            return base_text

        changed = self._rewrite_inline_state_network_energy_lines(
            lines=lines,
            state_row_index=state_row_index,
            state_unit_out=state_unit_out,
            energy_jmol=energy_jmol,
            touched=touched,
            global_unit=global_unit,
            UnitsModel=UnitsModel,
        )
        if not changed:
            return base_text
        return "\n".join(lines)

    def _apply_energy_overrides_to_computational_mode_fast_equilibria(
        self,
        base_text: str,
        *,
        set_id: Optional[str] = None,
        overrides: Optional[Dict[str, float]] = None,
    ) -> str:
        """
        Apply energy-mode ΔG° overrides to Computational Mode fast equilibria inside the generated block.

        These fast equilibria are emitted as explicit `equilibrium:` lines (kf/kr) and are updated by
        rewriting `dG_eq=` and `kr=` while keeping `kf=` fixed.
        """
        energy_overrides = [
            o
            for o in self._collect_energy_overrides(set_id=set_id, overrides=overrides)
            if str(o[2].get("role") or "") == "dG_eq_fast"
        ]
        if not energy_overrides:
            return base_text

        try:
            from kindred.core.simulator.computational_mode import (
                GENERATED_BLOCK_END,
                GENERATED_BLOCK_START,
                extract_marked_block,
                upsert_marked_block,
            )
        except Exception:
            return base_text

        body = extract_marked_block(base_text, start_marker=GENERATED_BLOCK_START, end_marker=GENERATED_BLOCK_END)
        if not body:
            return base_text
        full_lines = str(base_text).splitlines()
        generated_block_start_line = next(
            (
                idx
                for idx, raw_line in enumerate(full_lines)
                if str(raw_line).strip() == str(GENERATED_BLOCK_START)
            ),
            None,
        )
        if generated_block_start_line is None:
            return base_text

        T_override = self._dsl_global_temperature_K(base_text)
        try:
            T = float(T_override) if T_override is not None else float(getattr(self, "_temperature_spinbox").value())
        except Exception:
            return base_text
        if not (math.isfinite(T) and T > 0.0):
            return base_text

        from kindred.core.constants import R as R_J_per_mol_K
        from kindred.core.units import UnitsModel
        step_constraint_context = {
            "temperature_K": float(T),
            "wegscheider_cyclicity_enabled": bool(self.wegscheider_cyclicity_enabled()),
        }
        step_analysis_context = build_current_text_step_analysis_context(
            str(base_text or ""),
            step_constraint_context=step_constraint_context,
        )
        equilibrium_step_by_line_index = {
            int(line_index): int(step_index)
            for step_index, line_index in step_analysis_context.equilibrium_lines.items()
        }

        overrides_by_id: dict[str, tuple[float, dict]] = {}
        for _name, val, meta in energy_overrides:
            cm_id = str(meta.get("cm_id") or "")
            if not cm_id:
                continue
            overrides_by_id[cm_id] = (float(val), dict(meta))
        if not overrides_by_id:
            return base_text

        lines = str(body).splitlines()
        changed = False
        for i, raw in enumerate(lines):
            absolute_line_index = int(generated_block_start_line) + 1 + int(i)
            before_comment, hash_sep, comment_rest = str(raw).partition("#")
            code = before_comment.strip()
            if not code.lower().startswith("equilibrium:"):
                continue
            prefix, tokens = self._parse_semicolon_kv_tokens(code)
            cm_id = self._get_semicolon_kv(tokens, "cm_id") or ""
            if not cm_id or cm_id not in overrides_by_id:
                continue

            new_dG, meta = overrides_by_id[cm_id]
            unit_in = str(meta.get("unit") or "kJ/mol")

            kf_text = self._get_semicolon_kv(tokens, "kf")
            try:
                kf_fixed = float(meta.get("kf_fixed")) if meta.get("kf_fixed") is not None else float(kf_text or "nan")
            except Exception:
                kf_fixed = float("nan")
            try:
                std_ratio = float(meta.get("std_ratio") or 1.0)
            except Exception:
                std_ratio = 1.0
            if not (math.isfinite(std_ratio) and std_ratio > 0.0):
                std_ratio = 1.0
            if not (math.isfinite(kf_fixed) and kf_fixed > 0.0):
                continue

            try:
                dG_eq_J = float(UnitsModel(energy_unit=unit_in).to_jmol(float(new_dG)))
                K = float(math.exp(-float(dG_eq_J) / (float(R_J_per_mol_K) * float(T))))
            except Exception as exc:
                self._record_best_effort_failure(
                    "main_window.computational_mode_fast_eq.compute_K",
                    message=f"Failed to compute K from Computational Mode fast-equilibrium dG_eq override (cm_id='{cm_id}')",
                    exc=exc,
                )
                continue
            if not (math.isfinite(K) and K > 0.0):
                continue
            try:
                kr_new = float(kf_fixed / (K * std_ratio))
            except Exception as exc:
                self._record_best_effort_failure(
                    "main_window.computational_mode_fast_eq.compute_kr",
                    message=f"Failed to compute kr from Computational Mode fast-equilibrium override (cm_id='{cm_id}')",
                    exc=exc,
                )
                continue
            if not (math.isfinite(kr_new) and kr_new > 0.0):
                continue
            step_index = equilibrium_step_by_line_index.get(int(absolute_line_index))
            if step_index is None:
                continue
            warning_reason = step_rewrite_block_reason(
                step_index=int(step_index),
                affected_parameter_names=(f"kr{int(step_index)}",),
                step_analysis_context=step_analysis_context,
            )
            if warning_reason is not None:
                continue

            self._set_semicolon_kv(tokens, "dG_eq", f"{float(new_dG):.12g}")
            self._set_semicolon_kv(tokens, "kf", f"{float(kf_fixed):.17g}")
            self._set_semicolon_kv(tokens, "kr", f"{float(kr_new):.17g}")

            serialized = prefix
            if tokens:
                serialized += "; " + "; ".join(f"{k}={v}" for k, v in tokens if k)
            serialized = serialized + (hash_sep + comment_rest if hash_sep else "")
            if serialized != raw:
                lines[i] = serialized
                changed = True

        if not changed:
            return base_text
        new_body = "\n".join(lines).rstrip("\n")
        return upsert_marked_block(
            base_text,
            start_marker=GENERATED_BLOCK_START,
            end_marker=GENERATED_BLOCK_END,
            body=new_body,
            blank_line_before=True,
        )

    @staticmethod
    def _parse_state_network_dsl_state_line(line: str) -> tuple[str | None, list[list[str]], str | None]:
        before_comment, _, _comment = line.partition("#")
        if ":" not in before_comment:
            return None, [], None
        _prefix, rest = before_comment.split(":", 1)
        parts = [p.strip() for p in rest.split(",") if p.strip()]
        name: str | None = None
        tokens: list[list[str]] = []
        for part in parts:
            if "=" in part:
                key, val = part.split("=", 1)
                tokens.append([key.strip(), val.strip()])
            elif name is None:
                name = part
        if not name:
            for key, val in tokens:
                if key.strip().lower() in {"name", "state"}:
                    name = val.strip()
                    break
        unit: str | None = None
        energy_val_str: str | None = None
        for key, val in tokens:
            kl = key.strip().lower()
            if kl == "energy_unit":
                unit = val.strip()
            elif kl == "energy":
                energy_val_str = val.strip()
        if energy_val_str is not None:
            tok = energy_val_str.split()
            if len(tok) == 2 and tok[1] in ("kJ/mol", "kcal/mol", "J/mol"):
                unit = tok[1]
        return name, tokens, unit

    @staticmethod
    def _update_token_case_insensitive(tokens: list[list[str]], key: str, value: str) -> None:
        for pair in tokens:
            if pair and str(pair[0]).strip().lower() == str(key).strip().lower():
                pair[1] = str(value)
                return
        tokens.append([str(key), str(value)])

    def _apply_overrides_to_state_network_dsl(
        self,
        base_dsl: str,
        *,
        set_id: Optional[str] = None,
        overrides: Optional[Dict[str, float]] = None,
    ) -> str:
        """Return state-network DSL with energy-mode slider overrides applied."""
        energy_overrides = self._collect_energy_overrides(set_id=set_id, overrides=overrides)
        if not energy_overrides:
            return base_dsl

        drag_baseline_dsl = self._preview_session.drag_baseline_state_network_dsl()
        if self._preview_session.slider_drag_active() and drag_baseline_dsl is not None:
            dsl = str(drag_baseline_dsl or "")
        else:
            dsl = base_dsl

        try:
            from kindred.core.simulator.kinetics import normalize_energy_to_J_per_mol
            from kindred.core.units import UnitsModel
        except Exception:
            return dsl

        lines = (dsl or "").splitlines()
        state_rows: dict[str, dict[str, object]] = {}
        energy_jmol: dict[str, float] = {}

        for i, raw in enumerate(lines):
            stripped = raw.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if not stripped.lower().startswith("state:"):
                continue
            name, tokens, unit = self._parse_state_network_dsl_state_line(raw)
            if not name:
                continue
            unit = unit or "kJ/mol"
            val_str = None
            for key, val in tokens:
                if key.strip().lower() == "energy":
                    val_str = val.strip()
                    break
            try:
                energy_val = float(val_str) if val_str is not None else 0.0
            except Exception:
                energy_val = 0.0
            try:
                ej = normalize_energy_to_J_per_mol(energy_val, unit)
            except Exception as exc:
                self._record_best_effort_failure(
                    "main_window.state_network_dsl.normalize_energy",
                    message=f"Failed to normalize state-network energy '{name}' to J/mol",
                    exc=exc,
                )
                continue
            state_rows[str(name)] = {"line_index": i, "tokens": tokens, "unit": unit}
            energy_jmol[str(name)] = float(ej)

        if not state_rows:
            return dsl
        _ = self._apply_energy_overrides_to_state_energies(
            energy_jmol=energy_jmol,
            overrides=energy_overrides,
            global_unit="kJ/mol",
            UnitsModel=UnitsModel,
        )

        for name, row in state_rows.items():
            if name not in energy_jmol:
                continue
            try:
                idx = int(row["line_index"])
            except Exception as exc:
                self._record_best_effort_failure(
                    "main_window.state_network_dsl.state_line_index",
                    message=f"Invalid state-network line index for state '{name}'",
                    exc=exc,
                )
                continue
            unit_out = str(row.get("unit") or "kJ/mol")
            new_val = float(UnitsModel(energy_unit=str(unit_out)).from_jmol(float(energy_jmol[name])))
            tokens = row.get("tokens")
            if not isinstance(tokens, list):
                continue
            self._update_token_case_insensitive(tokens, "energy", f"{new_val:.6g}")
            params = ", ".join(f"{k}={v}" for k, v in tokens if k)
            lines[idx] = f"state: {name}, {params}".strip()

        return "\n".join(lines)

    def _updated_reactions_text_with_scalar_param(self, reactions_text: str, name: str, value: float) -> str:
        formatted_value = format_authoritative_parameter_value(value)
        reactions_lines = str(reactions_text or "").splitlines()

        updated = False
        in_algebra = False
        for i, raw in enumerate(reactions_lines):
            stripped = raw.strip()
            lower = stripped.lower()
            if lower.startswith("# algebra"):
                in_algebra = True
                continue
            if lower.startswith("# ") and in_algebra and not lower.startswith("# algebra"):
                in_algebra = False
            if not in_algebra:
                continue
            if not stripped or stripped.startswith("#"):
                continue
            before_comment, sep, comment = raw.partition("#")
            code = before_comment.strip()
            m = re.match(r"^param\s+([A-Za-z_][A-Za-z0-9_]*)\s*=", code, flags=re.IGNORECASE)
            if not m or m.group(1) != name:
                continue
            indent_match = re.match(r"^\\s*", before_comment)
            indent = indent_match.group(0) if indent_match else ""
            tail = f"{sep}{comment}" if sep else ""
            new_line = f"{indent}param {name} = {formatted_value}".rstrip()
            if tail:
                new_line = f"{new_line} {tail}".rstrip()
            reactions_lines[i] = new_line
            updated = True
            break

        if updated:
            return "\n".join(reactions_lines).rstrip("\n") + "\n"

        from kindred.core.simulator.algebra_section import upsert_lines_into_algebra_section

        return upsert_lines_into_algebra_section(
            "\n".join(reactions_lines).rstrip("\n") + "\n",
            [f"param {name} = {formatted_value}"],
            header="# Algebra",
        )

    def _apply_scalar_param_overrides_to_reactions_text(
        self,
        reactions_text: str,
        parameters: Dict[str, float],
    ) -> str:
        updated_text = str(reactions_text or "")
        for name, value in dict(parameters or {}).items():
            updated_text = self._updated_reactions_text_with_scalar_param(
                updated_text,
                str(name),
                float(value),
            )
        return updated_text

    def _update_scalar_param_in_algebra(self, name: str, value: float) -> None:
        """
        Update `param <name> = ...` inside the Reactions DSL `# Algebra` section.

        Notes are persisted separately and are never parsed or injected into the DSL.
        """
        reactions_widget = self._mechanism_editor._reactions_text
        new_text = self._updated_reactions_text_with_scalar_param(
            reactions_widget.toPlainText(),
            str(name),
            float(value),
        )

        if callable(getattr(self, "_set_text_with_optional_undo", None)):
            self._set_text_with_optional_undo(
                reactions_widget,
                new_text,
                f"Update param {name} in # Algebra",
                True,
            )
        else:
            reactions_widget.setPlainText(new_text)

    def _parameter_algebra_spec_for_ui(self, *, mechanism_param_names: set[str]):
        from kindred.core.simulator.algebra_section import extract_algebra_section_text

        reactions_text = self._mechanism_editor._reactions_text.toPlainText()
        if not extract_algebra_section_text(reactions_text).strip():
            return None
        from kindred.core.simulator.parameter_algebra import parse_parameter_algebra_spec_from_dsl_text

        return parse_parameter_algebra_spec_from_dsl_text(
            str(reactions_text or ""),
            mechanism_param_names=mechanism_param_names,
        )

    def _refresh_derived_parameters_display(self) -> None:
        sliders = self._mechanism_editor._variable_sliders
        current = sliders.get_variables()
        if not current:
            return
        mechanism_param_names = {k for k in current.keys() if re.match(r"^(k|kf|kr|K)\d+$", str(k))}
        spec = None
        try:
            spec = self._parameter_algebra_spec_for_ui(mechanism_param_names=mechanism_param_names)
        except Exception as exc:
            logger.debug("Parameter algebra not applied to sliders: %s", exc)
            spec = None

        from kindred.core.simulator.parameter_algebra import evaluate_parameter_algebra

        if spec is not None and getattr(spec, "param_statements", None):
            try:
                derived = evaluate_parameter_algebra(
                    spec,
                    base_values={k: float(v) for k, v in current.items()},
                )
            except Exception as exc:
                logger.debug("Parameter algebra evaluation failed for sliders: %s", exc)
                derived = {}
            for name, val in derived.items():
                sliders.update_variable_readout(name, float(val))
                meta_map = self._mutable_variable_metadata()
                meta = dict(meta_map.get(name) or {})
                meta["editable"] = False
                meta["derived"] = True
                meta_map[name] = meta
                sliders.update_metadata(name, meta)

        # Refresh snapshot after any derived updates.
        current = sliders.get_variables()

        # Apply K-implied equilibrium constraints (canonical step indexing) so derived
        # rates update immediately when K changes.
        step_map = getattr(self, "_step_index_map", None) or []
        if isinstance(step_map, list) and step_map:
            for entry in step_map:
                if not isinstance(entry, dict):
                    continue
                if str(entry.get("kind") or "") != "equilibrium":
                    continue
                if not bool(entry.get("has_K_param")):
                    continue
                try:
                    n = int(entry.get("step_index"))  # type: ignore[arg-type]
                except Exception as exc:
                    self._record_best_effort_failure(
                        "main_window.derived_K_constraints.step_index",
                        message="Invalid step_index while applying K-implied constraints",
                        exc=exc,
                    )
                    continue
                K_key = f"K{n}"
                kf_key = f"kf{n}"
                kr_key = f"kr{n}"
                if K_key not in current:
                    continue
                try:
                    K_val = float(current[K_key])
                except Exception as exc:
                    self._record_best_effort_failure(
                        "main_window.derived_K_constraints.K_val",
                        message=f"Invalid K value for {K_key} while applying K-implied constraints",
                        exc=exc,
                    )
                    continue
                if not math.isfinite(K_val) or abs(K_val) < 1e-30:
                    continue
                derive_rate = str(entry.get("derive_rate") or "kr")
                if derive_rate == "kf":
                    if kr_key not in current:
                        continue
                    try:
                        kr_val = float(current[kr_key])
                    except Exception as exc:
                        self._record_best_effort_failure(
                            "main_window.derived_K_constraints.kr_val",
                            message=f"Invalid kr value for {kr_key} while deriving kf",
                            exc=exc,
                        )
                        continue
                    kf_val = kr_val * K_val
                    sliders.update_variable_readout(kf_key, float(kf_val))
                    meta_map = self._mutable_variable_metadata()
                    meta = dict(meta_map.get(kf_key) or {})
                    meta["editable"] = False
                    meta["derived"] = True
                    meta_map[kf_key] = meta
                    sliders.update_metadata(kf_key, meta)
                else:
                    if kf_key not in current:
                        continue
                    try:
                        kf_val = float(current[kf_key])
                    except Exception as exc:
                        self._record_best_effort_failure(
                            "main_window.derived_K_constraints.kf_val",
                            message=f"Invalid kf value for {kf_key} while deriving kr",
                            exc=exc,
                        )
                        continue
                    kr_val = kf_val / K_val
                    sliders.update_variable_readout(kr_key, float(kr_val))
                    meta_map = self._mutable_variable_metadata()
                    meta = dict(meta_map.get(kr_key) or {})
                    meta["editable"] = False
                    meta["derived"] = True
                    meta_map[kr_key] = meta
                    sliders.update_metadata(kr_key, meta)

    def _energy_mode_energy_unit_from_metadata(self) -> str:
        meta_map = self.variable_metadata() or {}
        for _name, meta in meta_map.items():
            if isinstance(meta, dict) and meta.get("type") == "energy" and meta.get("unit"):
                return str(meta.get("unit"))
        return "kJ/mol"

    def _energy_mode_full_dsl_from_editor(self) -> str:
        reactions_text = self._mechanism_editor._reactions_text.toPlainText()
        state_network_dsl = self._mechanism_editor._state_network_editor.get_state_network_dsl()
        full_dsl = reactions_text
        if str(state_network_dsl or "").strip():
            full_dsl += "\n\n# State Network\n" + str(state_network_dsl)
        return str(full_dsl)

    def _energy_mode_temperature_K(self, full_dsl: str) -> float | None:
        T_override = self._dsl_global_temperature_K(full_dsl)
        try:
            T = float(T_override) if T_override is not None else float(self._temperature_spinbox.value())
        except Exception:
            return None
        if not (math.isfinite(T) and T > 0):
            return None
        return float(T)

    def _energy_mode_params_for_fast_equilibrium_channel(
        self,
        ch: dict[str, object],
        *,
        energy_unit: str,
        unit_conv,
        T: float,
        K_from_deltaG_eq,
        rate_units,
    ) -> Dict[str, Tuple[float, str]] | None:
        label = str(ch.get("label") or ch.get("cm_id") or "fast equilibrium")
        var_eq = str(ch.get("var_eq") or "")
        if not var_eq:
            cm_id = str(ch.get("cm_id") or "")
            slug = re.sub(r"[^A-Za-z0-9_]+", "_", cm_id).strip("_") or "fast_eq"
            var_eq = f"dG_eq_fast__{slug}"
        try:
            dG_eq = float(self.slider_overrides().get(var_eq, ch.get("dG_eq")))
        except Exception as exc:
            self._record_best_effort_failure(
                "main_window.energy_table.fast_eq.dG_eq",
                message=f"Failed to read fast-equilibrium dG_eq value ({label})",
                exc=exc,
            )
            return None
        if not math.isfinite(dG_eq):
            return None
        try:
            dG_eq_J = float(unit_conv.to_jmol(dG_eq))
        except Exception as exc:
            self._record_best_effort_failure(
                "main_window.energy_table.fast_eq.to_jmol",
                message=f"Failed to convert fast-equilibrium dG_eq to J/mol ({label})",
                exc=exc,
            )
            return None
        try:
            K = float(K_from_deltaG_eq(dG_eq_J, T))
        except Exception as exc:
            self._record_best_effort_failure(
                "main_window.energy_table.fast_eq.K_from_dG",
                message=f"Failed to compute K from fast-equilibrium dG_eq ({label})",
                exc=exc,
            )
            return None
        try:
            kf_fixed = float(ch.get("kf_fixed") or ch.get("kf") or float("nan"))
        except Exception:
            kf_fixed = float("nan")
        try:
            std_ratio = float(ch.get("std_ratio") or 1.0)
        except Exception:
            std_ratio = 1.0
        if not (math.isfinite(std_ratio) and std_ratio > 0.0):
            std_ratio = 1.0
        kr = float("nan")
        try:
            if math.isfinite(kf_fixed) and kf_fixed > 0.0 and math.isfinite(K) and K > 0.0:
                kr = float(kf_fixed / (K * std_ratio))
        except Exception:
            kr = float("nan")
        try:
            m_fwd = int(ch.get("molecularity_fwd") or 1)
        except Exception:
            m_fwd = 1
        try:
            m_rev = int(ch.get("molecularity_rev") or 1)
        except Exception:
            m_rev = 1
        unit_kf = "1/s"
        unit_kr = "1/s"
        try:
            unit_kf = str(rate_units(int(m_fwd)))
            unit_kr = str(rate_units(int(m_rev)))
        except Exception:
            unit_kf = "1/s"
            unit_kr = "1/s"
        return {
            f"ΔG° ({label})": (dG_eq, energy_unit),
            f"k_f ({label})": (kf_fixed, unit_kf),
            f"k_r ({label})": (kr, unit_kr),
            f"K ({label})": (K, "1"),
        }

    def _energy_mode_params_for_ts_channel(
        self,
        ch: dict[str, object],
        *,
        energy_unit: str,
        unit_conv,
        T: float,
        R: float,
        prefactor: float,
        K_from_deltaG_eq,
        rate_units,
    ) -> Dict[str, Tuple[float, str]] | None:
        reactant = str(ch.get("reactant") or "")
        product = str(ch.get("product") or "")
        ts = str(ch.get("ts") or "")
        if not (reactant and product and ts):
            return None
        label = str(ch.get("label") or f"{reactant}→{product} via {ts}")

        var_act = f"dGact_fwd__{ts}__{reactant}__{product}"
        var_eq = f"dG_eq__{ts}__{reactant}__{product}"
        try:
            overrides = self.slider_overrides()
            dG_act = float(overrides.get(var_act, ch.get("dG_act_fwd")))
            dG_eq = float(overrides.get(var_eq, ch.get("dG_eq")))
        except Exception as exc:
            self._record_best_effort_failure(
                "main_window.energy_table.ts_channel.read_dG",
                message=f"Failed to read energy-mode channel dG values ({label})",
                exc=exc,
            )
            return None
        if not (math.isfinite(dG_act) and math.isfinite(dG_eq)):
            return None
        try:
            dG_act_J = float(unit_conv.to_jmol(dG_act))
            dG_eq_J = float(unit_conv.to_jmol(dG_eq))
        except Exception as exc:
            self._record_best_effort_failure(
                "main_window.energy_table.ts_channel.to_jmol",
                message=f"Failed to convert energy-mode channel dG values to J/mol ({label})",
                exc=exc,
            )
            return None

        try:
            kappa = float(ch.get("kappa") or 1.0)
        except Exception:
            kappa = 1.0
        try:
            deg_ratio_fwd = float(ch.get("degeneracy_ratio_fwd") or 1.0)
        except Exception:
            deg_ratio_fwd = 1.0
        try:
            deg_ratio_rev = float(ch.get("degeneracy_ratio_rev") or 1.0)
        except Exception:
            deg_ratio_rev = 1.0

        try:
            K = float(K_from_deltaG_eq(dG_eq_J, T))
            dG_rev_J = float(dG_act_J - dG_eq_J)

            m_fwd = int(ch.get("molecularity_fwd") or 1)
            m_rev = int(ch.get("molecularity_rev") or 1)
            unit_kf = str(rate_units(int(m_fwd)))
            unit_kr = str(rate_units(int(m_rev)))

            std_ts = ch.get("std_conc_product_ts")
            std_react = ch.get("std_conc_product_reactant")
            std_prod = ch.get("std_conc_product_product")
            std_ts_f = float(std_ts) if std_ts is not None else 1.0
            std_react_f = float(std_react) if std_react is not None else 1.0
            std_prod_f = float(std_prod) if std_prod is not None else 1.0
            std_ratio_fwd = std_ts_f / std_react_f
            std_ratio_rev = std_ts_f / std_prod_f
            if not (math.isfinite(std_ratio_fwd) and std_ratio_fwd > 0.0):
                std_ratio_fwd = 1.0
            if not (math.isfinite(std_ratio_rev) and std_ratio_rev > 0.0):
                std_ratio_rev = 1.0

            expo_fwd = math.exp(-float(dG_act_J) / (R * float(T)))
            expo_rev = math.exp(-float(dG_rev_J) / (R * float(T)))
            kf = float(float(kappa) * float(prefactor) * float(deg_ratio_fwd) * float(expo_fwd) * float(std_ratio_fwd))
            kr = float(float(kappa) * float(prefactor) * float(deg_ratio_rev) * float(expo_rev) * float(std_ratio_rev))
        except Exception as exc:
            self._record_best_effort_failure(
                "main_window.energy_table.ts_channel.compute_rates",
                message=f"Failed to compute energy-mode channel rates ({label})",
                exc=exc,
            )
            return None

        return {
            f"ΔG‡_fwd ({label})": (dG_act, energy_unit),
            f"ΔG° ({label})": (dG_eq, energy_unit),
            f"k_f ({label})": (kf, unit_kf),
            f"k_r ({label})": (kr, unit_kr),
            f"K ({label})": (K, "1"),
        }

    def _refresh_energy_mode_derived_parameter_table(self) -> None:
        """
        Update the parameter table for energy-mode channels from current slider values.

        This is cheap (no ODE solve) and keeps the read-only k_f/k_r/K display in sync
        while the user drags energy sliders.
        """
        plot = getattr(getattr(self, "_plot_tabs", None), "_main_plot", None)
        if plot is None or not hasattr(plot, "update_parameters"):
            return

        channels = getattr(self, "_energy_mode_channels", None) or []
        if not isinstance(channels, list) or not channels:
            return

        energy_unit = self._energy_mode_energy_unit_from_metadata()
        full_dsl = self._energy_mode_full_dsl_from_editor()
        T = self._energy_mode_temperature_K(full_dsl)
        if T is None:
            return

        try:
            from kindred.core.constants import R
            from kindred.core.simulator.kinetics import K_from_deltaG_eq, eyring_prefactor, rate_units
            from kindred.core.units import UnitsModel
        except Exception:
            return

        unit_conv = UnitsModel(energy_unit=energy_unit)
        params: Dict[str, Tuple[float, str]] = {}
        prefactor = float(eyring_prefactor(T))

        for ch in channels:
            if not isinstance(ch, dict):
                continue
            kind = str(ch.get("kind") or "ts_channel")
            if kind == "fast_equilibrium":
                out = self._energy_mode_params_for_fast_equilibrium_channel(
                    ch,
                    energy_unit=energy_unit,
                    unit_conv=unit_conv,
                    T=T,
                    K_from_deltaG_eq=K_from_deltaG_eq,
                    rate_units=rate_units,
                )
            else:
                out = self._energy_mode_params_for_ts_channel(
                    ch,
                    energy_unit=energy_unit,
                    unit_conv=unit_conv,
                    T=T,
                    R=float(R),
                    prefactor=prefactor,
                    K_from_deltaG_eq=K_from_deltaG_eq,
                    rate_units=rate_units,
                )
            if out:
                params.update(out)

        if params:
            try:
                plot.update_parameters(params)
            except Exception as exc:
                logger.debug("Failed to update plot parameter summary: %s", exc, exc_info=True)
                self._plot_parameter_summary_stale = True

    def _prepare_slider_runtime(
        self,
        param_names: Optional[List[str]] = None,
        *,
        set_id: Optional[str] = None,
    ) -> Optional[BoundMechanism]:
        return self._variable_runtime.prepare_slider_runtime(param_names=param_names, set_id=set_id)

    def _apply_slider_overrides_to_bindings(
        self,
        runtime: Optional[BoundMechanism],
        *,
        set_id: Optional[str] = None,
    ) -> bool:
        return bool(self._variable_runtime.apply_slider_overrides_to_bindings(runtime, set_id=set_id))

    def _on_variable_changed(self, name: str, value: float):
        self._preview_session.on_variable_changed(name, value)
        self._refresh_slider_transaction_button_state()

    def _commit_slider_value(self, name: str, value: float) -> None:
        self._preview_session.commit_slider_value(name, value)
        self._sim_controller.invalidate_slider_preview_work()
        self._sim_controller.clear_pending_slider_preview_replay(clear_plot_updates=False)
        self._preview_session.stop_variable_update_timer()
        self._preview_session.stop_slider_release_commit_timer()
        self._materialize_direct_slider_commit_to_authoritative_editors(name, value)
        self._preview_session.commit_current_mechanism_workspace()
        self._sync_after_authoritative_slider_materialization()

    def _on_slider_drag_started(self, name: str) -> None:
        self._preview_session.on_slider_drag_started(name)

    def _on_slider_drag_finished(self, name: str) -> None:
        self._preview_session.on_slider_drag_finished(name)

    def _finalize_slider_release_commit(self) -> None:
        self._preview_session.finalize_slider_release_commit()

    def _slider_context_is_species_mode(self) -> bool:
        if not bool(getattr(self, "_species_panel_available", True)):
            return False
        editor = getattr(self, "_mechanism_editor", None)
        if editor is None or not hasattr(editor, "species_mode_enabled"):
            return False
        try:
            return bool(editor.species_mode_enabled())
        except Exception:
            return False

    def _commit_species_slider_values_to_selected_batch_rows(self) -> None:
        model = getattr(self, "_batch_model", None)
        if model is None:
            return
        try:
            self._preview_session.apply_staged_concentration_overlays(model)
        except Exception as exc:
            self._record_best_effort_failure(
                "main_window.species_mode.commit_species.apply",
                message="Failed to apply staged species-mode concentration overlays",
                exc=exc,
            )

    def _materialized_mechanism_editor_texts_for_effective_slider_values(
        self,
        values: Dict[str, float],
    ) -> tuple[str, str]:
        non_scalar_values, scalar_values, state_network_values = (
            self._partition_effective_slider_commit_values(values)
        )
        reactions_text = self.mechanism_reactions_text_raw()
        if non_scalar_values:
            reactions_text = self._apply_overrides_to_text(
                reactions_text,
                overrides=non_scalar_values,
            )
            reactions_text = self._apply_parameter_overrides_to_dsl(
                reactions_text,
                non_scalar_values,
            )
        if scalar_values:
            reactions_text = self._apply_scalar_param_overrides_to_reactions_text(
                reactions_text,
                scalar_values,
            )

        state_network_dsl = self.mechanism_state_network_dsl_raw()
        if state_network_dsl.strip() and state_network_values:
            state_network_dsl = self._apply_overrides_to_state_network_dsl(
                state_network_dsl,
                overrides=state_network_values,
            )
        return str(reactions_text), str(state_network_dsl)

    def _partition_effective_slider_commit_values(
        self,
        values: Dict[str, float],
    ) -> tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
        metadata = self.variable_metadata() or {}
        non_scalar_values: Dict[str, float] = {}
        scalar_values: Dict[str, float] = {}
        state_network_values: Dict[str, float] = {}
        for name, value in dict(values or {}).items():
            name_s = str(name)
            try:
                value_f = float(value)
            except Exception:
                continue
            meta = metadata.get(name_s, {})
            if isinstance(meta, dict) and meta.get("type") == "scalar":
                scalar_values[name_s] = value_f
                continue
            non_scalar_values[name_s] = value_f
            if isinstance(meta, dict) and meta.get("type") == "energy":
                state_network_values[name_s] = value_f
        return non_scalar_values, scalar_values, state_network_values

    def _set_authoritative_mechanism_editor_texts(
        self,
        *,
        reactions_text: str,
        state_network_dsl: str,
        description: str,
    ) -> None:
        from kindred.gui.undo_commands import SetMechanismEditorTextsCommand

        previous_suppress = self._variable_runtime.suppress_slider_runtime_invalidation()
        previous_authoritative_suppress = bool(
            getattr(self, "_suppress_authoritative_mechanism_input_change", False)
        )
        self._variable_runtime.set_suppress_slider_runtime_invalidation(True)
        self._suppress_authoritative_mechanism_input_change = True
        try:
            current_reactions = str(self.mechanism_reactions_text_raw() or "")
            state_editor = getattr(getattr(self, "_mechanism_editor", None), "_state_network_editor", None)
            state_editor_available = (
                state_editor is not None
                and hasattr(state_editor, "get_state_network_dsl")
                and hasattr(state_editor, "set_state_network_dsl")
            )
            current_state = (
                str(state_editor.get_state_network_dsl() or "")
                if state_editor_available
                else ""
            )
            reactions_text_s = str(reactions_text)
            state_network_dsl_s = str(state_network_dsl)
            reactions_changed = reactions_text_s != current_reactions
            state_changed = state_editor_available and state_network_dsl_s != current_state
            if not reactions_changed and not state_changed:
                return
            command = SetMechanismEditorTextsCommand(
                self._mechanism_editor._reactions_text,
                state_editor if state_editor_available else None,
                new_reactions_text=reactions_text_s,
                old_reactions_text=current_reactions,
                new_state_network_dsl=state_network_dsl_s,
                old_state_network_dsl=current_state,
                description=str(description),
            )
            self._undo_stack.push(command)
        finally:
            self._variable_runtime.set_suppress_slider_runtime_invalidation(previous_suppress)
            self._suppress_authoritative_mechanism_input_change = previous_authoritative_suppress

    def _materialize_direct_slider_commit_to_authoritative_editors(self, name: str, value: float) -> None:
        focused_set_id = str(self._preview_session.focused_mechanism_workspace_set_id() or "")
        effective_values = (
            self._preview_session.effective_slider_values_for_set(focused_set_id)
            if focused_set_id
            else {}
        )
        normalized_values = self._normalized_slider_overrides(
            overrides=effective_values or {str(name): float(value)}
        )
        if str(name) not in normalized_values:
            parsed, ok = try_parse_finite_float(value)
            if ok:
                normalized_values[str(name)] = float(parsed)
        self._apply_effective_slider_values_to_mechanism_editors(
            normalized_values,
            description=f"Commit slider {name}",
        )

    def _apply_effective_slider_values_to_mechanism_editors(
        self,
        values: Dict[str, float],
        *,
        description: str = "Commit slider overrides",
    ) -> None:
        reactions_text, state_network_dsl = self._materialized_mechanism_editor_texts_for_effective_slider_values(
            values
        )
        self._set_authoritative_mechanism_editor_texts(
            reactions_text=reactions_text,
            state_network_dsl=state_network_dsl,
            description=str(description),
        )

    def _sync_after_authoritative_slider_materialization(
        self,
        *,
        preserve_current_display: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._variable_runtime.clear_prepared_slider_runtime(dirty=True)
        try:
            self._extract_and_populate_variables(preserve_visibility=True)
        except Exception:
            logger.exception("Failed to refresh sliders after authoritative slider materialization")
            self._variable_runtime.clear_prepared_slider_runtime(dirty=True)
        try:
            panel = self._mechanism_editor.species_sliders_widget()
            if panel is not None and hasattr(panel, "rebuild_from_current_row"):
                panel.rebuild_from_current_row()
        except Exception:
            logger.exception("Failed to rebuild species panel after slider materialization")
        self._invalidate_active_results_after_authoritative_mechanism_change(
            preserve_current_display=preserve_current_display
        )
        self._refresh_slider_transaction_button_state()

    def _finalize_authoritative_slider_materialization(
        self,
        effective_values: Dict[str, float],
        *,
        description: str,
        apply_species_overlays: bool,
    ) -> None:
        preserve_current_display = self._active_workspace_preview_display_snapshot()
        self._sim_controller.invalidate_slider_preview_work()
        self._sim_controller.clear_pending_slider_preview_replay(clear_plot_updates=False)
        self._preview_session.stop_variable_update_timer()
        self._preview_session.stop_slider_release_commit_timer()
        self._apply_effective_slider_values_to_mechanism_editors(
            effective_values,
            description=str(description),
        )
        if bool(apply_species_overlays):
            self._commit_species_slider_values_to_selected_batch_rows()
        self._preview_session.commit_current_mechanism_workspace()
        self._sync_after_authoritative_slider_materialization(
            preserve_current_display=preserve_current_display
        )

    def _sync_mechanism_controls_to_focused_batch_set(self, *, use_workspace: bool = True) -> None:
        sliders = getattr(getattr(self, "_mechanism_editor", None), "_variable_sliders", None)
        if sliders is not None and hasattr(sliders, "get_variables"):
            current_values = sliders.get_variables() or {}
            if current_values:
                focused_set_id = self._preview_session.focused_mechanism_workspace_set_id()
                if bool(use_workspace):
                    effective_values = self._preview_session.effective_slider_values(set_id=focused_set_id)
                else:
                    effective_values = dict(self._preview_session.param_store.shared_params)
                metadata = self.variable_metadata() or {}

                for name, current_value in current_values.items():
                    meta = metadata.get(str(name)) or {}
                    if isinstance(meta, dict) and meta.get("editable") is False:
                        continue
                    target_value = effective_values.get(str(name))
                    if target_value is None:
                        continue
                    try:
                        current_float = float(current_value)
                        target_float = float(target_value)
                    except Exception:
                        continue
                    if math.isclose(current_float, target_float, rel_tol=1e-12, abs_tol=1e-12):
                        continue
                    sliders.update_variable(str(name), target_float)

                self._refresh_derived_parameters_display()
                if any(
                    isinstance(meta, dict) and meta.get("type") == "energy"
                    for meta in metadata.values()
                ):
                    self._refresh_energy_mode_derived_parameter_table()
                else:
                    self._update_parameter_table_from_sliders()

        try:
            panel = self._mechanism_editor.species_sliders_widget()
            if panel is not None and hasattr(panel, "rebuild_from_current_row"):
                panel.rebuild_from_current_row()
        except Exception:
            logger.exception("Failed to rebuild species panel while syncing focused mechanism controls")

    def _on_commit_slider_overrides_clicked(self) -> None:
        """
        Commit current slider values into the DSL editor.

        In override mode, sliders update preview simulations via bound overrides without rewriting
        the mechanism text. "Commit" applies the current editable slider values to the DSL.
        """
        focused_set_id = self._preview_session.focused_mechanism_workspace_set_id()
        effective_values = self._preview_session.effective_slider_values(set_id=focused_set_id)
        self._finalize_authoritative_slider_materialization(
            effective_values,
            description="Commit slider overrides",
            apply_species_overlays=True,
        )

    def _on_reset_slider_overrides_clicked(self) -> None:
        """Reset slider overrides back to the baseline DSL values and refresh slider widgets."""
        if bool(self._preview_session.has_staged_concentration_overlays()):
            self._discard_slider_transaction_for_invalidation()
        else:
            self._sim_controller.invalidate_slider_preview_work()
            self._sim_controller.clear_pending_slider_preview_replay(clear_plot_updates=False)
            target_set_ids = self._effective_slider_edit_target_set_ids()
            self._preview_session.reset_mechanism_workspaces(target_set_ids)
            self._variable_runtime.clear_prepared_slider_runtime(dirty=True)

            sliders = getattr(getattr(self, "_mechanism_editor", None), "_variable_sliders", None)
            if sliders is not None and hasattr(sliders, "end_live_drag"):
                sliders.end_live_drag()

            try:
                self._extract_and_populate_variables(preserve_visibility=True)
            except Exception:
                logger.exception("Failed to refresh sliders after override reset")
                self._variable_runtime.clear_prepared_slider_runtime(dirty=True)
            try:
                self._update_parameter_table_from_sliders()
            except Exception:
                logger.exception("Failed to update parameter table after override reset")
                QtCore.QTimer.singleShot(0, self._update_parameter_table_from_sliders)

            try:
                self._ensure_batch_current_row_selected()
                panel = self._mechanism_editor.species_sliders_widget()
                if panel is not None and hasattr(panel, "rebuild_from_current_row"):
                    panel.rebuild_from_current_row()
            except Exception:
                logger.exception("Failed to reset species row")
                self._species_panel_available = False
            self._refresh_slider_transaction_button_state()
            self._refresh_batch_display_from_focus_and_shown()
            return
        self._sim_controller.clear_pending_slider_preview_replay(clear_plot_updates=False)
        self._sim_controller.run_simulation_from_slider()

    def _discard_slider_transaction_for_invalidation(self) -> None:
        """Clear the staged transaction without scheduling a preview rerun."""
        batch_cache = self._sim_controller.batch_cache
        active_overlay_token = str(batch_cache.active_cache_preview_token or "").strip()
        active_overlay_scope_ids = tuple(str(set_id) for set_id in (batch_cache.active_cache_preview_scope_set_ids or ()))
        current_overlay_token = None
        if active_overlay_token and active_overlay_scope_ids:
            scope_rows = []
            for set_id in active_overlay_scope_ids:
                try:
                    row = getattr(self, "_batch_store", None).row_for_set_id(str(set_id))
                except Exception:
                    row = None
                if row is not None:
                    scope_rows.append(int(row))
            if scope_rows:
                current_overlay_token = self._preview_session.preview_batch_cache_token(scope_rows) or None
        elif active_overlay_token and bool(self._preview_session.has_staged_concentration_overlays()):
            try:
                row_count = int(getattr(self, "_batch_store", None).row_count())
            except Exception:
                row_count = 0
            if row_count > 0:
                current_overlay_token = self._preview_session.preview_batch_cache_token(list(range(int(row_count)))) or None
        self._sim_controller.invalidate_slider_preview_work()
        self._preview_session.clear_working_transaction()
        if current_overlay_token and active_overlay_token == str(current_overlay_token):
            batch_cache.clear_active_selection_state()
        self._sim_controller.clear_pending_slider_preview_replay(clear_plot_updates=False)
        self._variable_runtime.clear_prepared_slider_runtime(dirty=True)

        sliders = getattr(getattr(self, "_mechanism_editor", None), "_variable_sliders", None)
        if sliders is not None and hasattr(sliders, "end_live_drag"):
            sliders.end_live_drag()

        try:
            self._extract_and_populate_variables(preserve_visibility=True)
        except Exception:
            logger.exception("Failed to refresh sliders after override reset")
            self._variable_runtime.clear_prepared_slider_runtime(dirty=True)
        try:
            self._update_parameter_table_from_sliders()
        except Exception:
            logger.exception("Failed to update parameter table after override reset")
            QtCore.QTimer.singleShot(0, self._update_parameter_table_from_sliders)

        try:
            self._ensure_batch_current_row_selected()
            panel = self._mechanism_editor.species_sliders_widget()
            if panel is not None and hasattr(panel, "rebuild_from_current_row"):
                panel.rebuild_from_current_row()
        except Exception:
            logger.exception("Failed to reset species row")
            self._species_panel_available = False
        self._refresh_slider_transaction_button_state()

    # ------------------------------------------------------------------
    # Species mode (Batch Initial Conditions sliders)
    # ------------------------------------------------------------------

    def _ensure_batch_current_row_selected(self) -> None:
        table = getattr(self, "_batch_table", None)
        model = getattr(self, "_batch_model", None)
        if table is None or model is None:
            return
        if model.rowCount() <= 0:
            return
        current = table.currentIndex()
        if current.isValid():
            return
        idx = model.index(0, 0)
        if not idx.isValid():
            return
        table.setCurrentIndex(idx)
        self._update_focused_batch_set_id(row=0)
        sel = table.selectionModel()
        if sel is None:
            return
        sel.clearSelection()
        sel.select(idx, QtCore.QItemSelectionModel.Select | QtCore.QItemSelectionModel.Rows)

    def _on_species_mode_changed(self, enabled: bool) -> None:
        enabled = bool(enabled)
        panel = None
        try:
            panel = self._mechanism_editor.species_sliders_widget()
        except Exception:
            panel = None
        if not enabled:
            if panel is not None:
                if hasattr(panel, "deactivate"):
                    try:
                        panel.deactivate()
                    except RuntimeError as exc:
                        logger.debug("Failed to deactivate species panel: %s", exc, exc_info=True)
                        self._species_panel_available = False
            self._preview_session.deactivate_species_preview_timer()
            return

        self._ensure_batch_current_row_selected()
        if panel is not None:
            if hasattr(panel, "activate"):
                try:
                    panel.activate()
                except RuntimeError as exc:
                    logger.debug("Failed to activate species panel: %s", exc, exc_info=True)
                    self._species_panel_available = False

    def _on_species_reset_requested(self) -> None:
        panel = None
        try:
            panel = self._mechanism_editor.species_sliders_widget()
        except Exception:
            panel = None
        if panel is None:
            return
        try:
            rows = []
            for set_id in self._effective_slider_edit_target_set_ids():
                row = self._batch_row_for_set_id(set_id)
                if row is not None:
                    rows.append(int(row))
            if not rows:
                current_row = self._batch_current_row()
                rows = [int(current_row)] if current_row is not None else []
            changed = bool(self._preview_session.discard_concentration_overlays_for_rows(rows))
            if changed and hasattr(panel, "rebuild_from_current_row"):
                panel.rebuild_from_current_row()
        except Exception:
            changed = False
        if changed:
            self._refresh_slider_transaction_button_state()
            self._queue_species_slider_simulation(label="init:reset", delay_ms=0)

    def _on_species_slider_edited(self, species: str, value: float) -> None:
        _ = float(value)  # ensure numeric for callers; no side effects
        self._refresh_slider_transaction_button_state()
        self._preview_session.queue_species_slider_simulation(label=f"init:{species}", delay_ms=80)

    def _on_species_slider_drag_finished(self, species: str) -> None:
        self._preview_session.queue_species_slider_simulation(label=f"init:{species}", delay_ms=0)

    def _queue_species_slider_simulation(self, *, label: str, delay_ms: int) -> None:
        self._preview_session.queue_species_slider_simulation(label=label, delay_ms=delay_ms)

    @staticmethod
    def _parse_mechanism_semicolon_kv(line: str) -> tuple[str, list[list[str]], str]:
        before_comment, sep, comment = str(line or "").partition("#")
        comment_tail = f"{sep}{comment}" if sep else ""
        prefix, sep_params, rest = before_comment.partition(";")
        prefix = prefix.rstrip()
        tokens: list[list[str]] = []
        if sep_params:
            for token in re.split(r"[;,]", rest):
                token = token.strip()
                if not token:
                    continue
                key, _, val = token.partition("=")
                tokens.append([key.strip(), val.strip()])
        return prefix, tokens, comment_tail

    @staticmethod
    def _serialize_mechanism_semicolon_kv(prefix: str, tokens: list[list[str]], comment_tail: str) -> str:
        if tokens:
            params = ", ".join(f"{key}={val}" if val else f"{key}=" for key, val in tokens)
            base = f"{prefix} ; {params}"
        else:
            base = prefix
        if comment_tail:
            base = f"{base} {comment_tail.strip()}"
        return base.strip()

    @staticmethod
    def _dedupe_tokens_case_insensitive(tokens: list[list[str]]) -> list[list[str]]:
        seen = set()
        result: list[list[str]] = []
        for key, val in tokens:
            lower = str(key).lower()
            if lower in seen:
                continue
            seen.add(lower)
            result.append([key, val])
        return result

    @staticmethod
    def _get_token_float(tokens: list[list[str]], aliases: tuple[str, ...], default: Optional[float] = None) -> Optional[float]:
        # Important: treat "K" as case-sensitive so it is never conflated with the
        # "k"/"kf" forward-rate aliases on equilibrium lines.
        exact_aliases = set(aliases)
        alias_set = {alias.lower() for alias in aliases if alias != "K"}
        for key, val in tokens:
            if key == "K":
                if "K" not in exact_aliases:
                    continue
                try:
                    return float(val)
                except (TypeError, ValueError):
                    return default
            if str(key).lower() in alias_set:
                try:
                    return float(val)
                except (TypeError, ValueError):
                    return default
        return default

    @staticmethod
    def _set_token_float(
        tokens: list[list[str]],
        canonical_key: str,
        float_value: float,
        *,
        aliases: tuple[str, ...] = (),
        sig: int | None = None,
    ) -> None:
        if sig is None:
            sanitized = format_authoritative_parameter_value(float_value)
        else:
            sanitized = f"{float(float_value):.{int(sig)}g}"
        exact_aliases = set(aliases)

        if canonical_key == "K":
            target_index = None
            for idx, (key, _) in enumerate(tokens):
                if key == "K":
                    target_index = idx
                    break
            if target_index is not None:
                tokens[target_index][0] = canonical_key
                tokens[target_index][1] = sanitized
            else:
                tokens.append([canonical_key, sanitized])
            return

        alias_set = {canonical_key.lower()}
        alias_set.update(alias.lower() for alias in aliases if alias != "K")

        target_index = None
        for idx, (key, _) in enumerate(tokens):
            if key == "K" and "K" not in exact_aliases:
                continue
            if str(key).lower() in alias_set:
                target_index = idx
                break

        if target_index is not None:
            tokens[target_index][0] = canonical_key
            tokens[target_index][1] = sanitized
        else:
            tokens.append([canonical_key, sanitized])
            target_index = len(tokens) - 1

        for idx in range(len(tokens) - 1, -1, -1):
            if idx == target_index:
                continue
            token_key = tokens[idx][0]
            if token_key == "K" and canonical_key != "K" and "K" not in aliases:
                continue
            if str(token_key).lower() in alias_set:
                tokens.pop(idx)

    @staticmethod
    def _remove_token_aliases(tokens: list[list[str]], aliases: tuple[str, ...]) -> None:
        alias_set = {alias.lower() for alias in aliases}
        exact_aliases = set(aliases)
        filtered: list[list[str]] = []
        for key, val in tokens:
            lower = str(key).lower()
            if lower not in alias_set:
                filtered.append([key, val])
                continue
            if key == "K" and "K" not in exact_aliases:
                filtered.append([key, val])
                continue
        tokens[:] = filtered

    @staticmethod
    def _derived_equilibrium_role_from_metadata(metadata: dict[str, object], eq_index: int) -> str:
        kf_meta = metadata.get(f"kf{eq_index}")
        kr_meta = metadata.get(f"kr{eq_index}")
        if isinstance(kf_meta, dict) and (kf_meta.get("derived") is True or kf_meta.get("editable") is False):
            return "kf"
        if isinstance(kr_meta, dict) and (kr_meta.get("derived") is True or kr_meta.get("editable") is False):
            return "kr"
        # Default policy matches core: derive kr unless only kr was explicitly provided.
        return "kr"

    @staticmethod
    def _label_for_step_from_metadata(metadata: dict[str, object], step_index: int, fallback_prefix: str) -> str:
        # Preserve canonical labels generated from step_index_map ("Step N: ...").
        for key in (f"kf{step_index}", f"kr{step_index}", f"K{step_index}", f"k{step_index}"):
            meta0 = metadata.get(key)
            if isinstance(meta0, dict) and isinstance(meta0.get("label"), str) and meta0.get("label"):
                return str(meta0["label"])
        return f"Step {step_index}: {fallback_prefix}".strip()

    @staticmethod
    def _index_step_lines(lines: list[str]) -> tuple[Dict[int, int], Dict[int, int]]:
        reaction_lines: Dict[int, int] = {}
        equilibrium_lines: Dict[int, int] = {}
        line_counter = 0
        for idx, line in enumerate(lines):
            stripped = str(line).strip()
            if not stripped or stripped.startswith("#"):
                continue
            lower = stripped.lower()
            if "<->" in lower or "<=>" in lower:
                line_counter += 1
                equilibrium_lines[line_counter] = idx
                continue
            if "->" in lower:
                line_counter += 1
                reaction_lines[line_counter] = idx
        return reaction_lines, equilibrium_lines

    def _update_variable_in_mechanism(
        self,
        name: str,
        value: float,
        *,
        source_text: Optional[str] = None,
        commit: bool = True,
        metadata: Optional[Dict[str, Dict[str, object]]] = None,
        step_analysis_context: object | None = None,
    ) -> str:
        """
        Update a variable value in the mechanism text.

        Parameters
        ----------
        name : str
            Variable name (e.g., 'k1', 'K1', 'kf1')
        value : float
            New value
        source_text : str | None
            Optional mechanism DSL to apply the change to. If None, use editor text.
        commit : bool, default True
            When True, apply updates to the editor and slider widgets.

        Returns
        -------
        str
            Mechanism DSL with the updated value applied.
        """
        if source_text is None:
            mechanism_text = self._mechanism_editor.reactions_text()
        else:
            mechanism_text = source_text

        if metadata is None:
            metadata = dict(self.variable_metadata())
        slider_updates: list[tuple[str, float, Dict[str, object]]] = []
        sliders = getattr(self._mechanism_editor, "_variable_sliders", None) if commit else None
        step_constraint_context = {
            "temperature_K": float(self._temperature_spinbox.value()),
            "wegscheider_cyclicity_enabled": bool(self.wegscheider_cyclicity_enabled()),
        }
        try:
            outcome = analyze_step_parameter_update(
                mechanism_text,
                name,
                value,
                step_constraint_context=step_constraint_context,
                step_analysis_context=step_analysis_context,
            )
        except ValueError:
            logger.warning("Ignoring invalid step parameter update for %r", name)
            return mechanism_text

        if not outcome.found_target or not outcome.writable or not outcome.would_change_text:
            return mechanism_text

        family = str(outcome.parameter_family)
        step_index = int(outcome.step_index)
        line_index = int(outcome.line_index if outcome.line_index is not None else 0)
        line_label = str(outcome.line_prefix or "")
        resolved_values = {
            str(param_name): float(param_value)
            for param_name, param_value in outcome.resolved_values
        }
        new_text = str(outcome.updated_text)

        if family == "K":
            K_val = float(resolved_values[f"K{step_index}"])
            kf_val = float(resolved_values[f"kf{step_index}"])
            kr_val = float(resolved_values[f"kr{step_index}"])
            label_text = self._label_for_step_from_metadata(metadata, step_index, line_label)
            K_meta = dict(metadata.get(f"K{step_index}") or {})
            K_meta.update(
                {
                    "type": "equilibrium",
                    "index": step_index,
                    "label": K_meta.get("label") or label_text,
                    "line": line_index,
                    "role": "K",
                }
            )
            metadata[f"K{step_index}"] = K_meta
            kf_meta = dict(metadata.get(f"kf{step_index}") or {})
            kf_meta.update(
                {
                    "type": "equilibrium",
                    "index": step_index,
                    "label": kf_meta.get("label") or label_text,
                    "line": line_index,
                    "role": "kf",
                }
            )
            metadata[f"kf{step_index}"] = kf_meta
            kr_meta = dict(metadata.get(f"kr{step_index}") or {})
            kr_meta.update(
                {
                    "type": "equilibrium",
                    "index": step_index,
                    "label": kr_meta.get("label") or label_text,
                    "line": line_index,
                    "role": "kr",
                }
            )
            metadata[f"kr{step_index}"] = kr_meta
            if commit:
                slider_updates.append((f"K{step_index}", float(f"{K_val:.6g}"), metadata[f"K{step_index}"]))
                slider_updates.append((f"kf{step_index}", float(f"{kf_val:.6g}"), metadata[f"kf{step_index}"]))
                slider_updates.append((f"kr{step_index}", float(f"{kr_val:.6g}"), metadata[f"kr{step_index}"]))
        elif family == "kf":
            kf_val = float(resolved_values[f"kf{step_index}"])
            kr_val = float(resolved_values[f"kr{step_index}"])
            label_text = self._label_for_step_from_metadata(metadata, step_index, line_label)
            kf_meta = dict(metadata.get(f"kf{step_index}") or {})
            kf_meta.update(
                {
                    "type": "equilibrium",
                    "index": step_index,
                    "label": kf_meta.get("label") or label_text,
                    "line": line_index,
                    "role": "kf",
                }
            )
            metadata[f"kf{step_index}"] = kf_meta
            kr_meta = dict(metadata.get(f"kr{step_index}") or {})
            kr_meta.update(
                {
                    "type": "equilibrium",
                    "index": step_index,
                    "label": kr_meta.get("label") or label_text,
                    "line": line_index,
                    "role": "kr",
                }
            )
            metadata[f"kr{step_index}"] = kr_meta
            if f"K{step_index}" in resolved_values:
                K_val = float(resolved_values[f"K{step_index}"])
                K_meta = dict(metadata.get(f"K{step_index}") or {})
                K_meta.update(
                    {
                        "type": "equilibrium",
                        "index": step_index,
                        "label": K_meta.get("label") or label_text,
                        "line": line_index,
                        "role": "K",
                    }
                )
                metadata[f"K{step_index}"] = K_meta
            if commit:
                slider_updates.append((f"kf{step_index}", float(f"{kf_val:.6g}"), metadata[f"kf{step_index}"]))
                slider_updates.append((f"kr{step_index}", float(f"{kr_val:.6g}"), metadata[f"kr{step_index}"]))
                if f"K{step_index}" in resolved_values:
                    slider_updates.append((f"K{step_index}", float(f"{resolved_values[f'K{step_index}']:.6g}"), metadata[f"K{step_index}"]))
        elif family == "kr":
            kr_val = float(resolved_values[f"kr{step_index}"])
            kf_val = float(resolved_values[f"kf{step_index}"])
            label_text = self._label_for_step_from_metadata(metadata, step_index, line_label)
            kr_meta = dict(metadata.get(f"kr{step_index}") or {})
            kr_meta.update(
                {
                    "type": "equilibrium",
                    "index": step_index,
                    "label": kr_meta.get("label") or label_text,
                    "line": line_index,
                    "role": "kr",
                }
            )
            metadata[f"kr{step_index}"] = kr_meta
            kf_meta = dict(metadata.get(f"kf{step_index}") or {})
            kf_meta.update(
                {
                    "type": "equilibrium",
                    "index": step_index,
                    "label": kf_meta.get("label") or label_text,
                    "line": line_index,
                    "role": "kf",
                }
            )
            metadata[f"kf{step_index}"] = kf_meta
            if f"K{step_index}" in resolved_values:
                K_val = float(resolved_values[f"K{step_index}"])
                K_meta = dict(metadata.get(f"K{step_index}") or {})
                K_meta.update(
                    {
                        "type": "equilibrium",
                        "index": step_index,
                        "label": K_meta.get("label") or label_text,
                        "line": line_index,
                        "role": "K",
                    }
                )
                metadata[f"K{step_index}"] = K_meta
            if commit:
                slider_updates.append((f"kr{step_index}", float(f"{kr_val:.6g}"), metadata[f"kr{step_index}"]))
                slider_updates.append((f"kf{step_index}", float(f"{kf_val:.6g}"), metadata[f"kf{step_index}"]))
                if f"K{step_index}" in resolved_values:
                    slider_updates.append((f"K{step_index}", float(f"{resolved_values[f'K{step_index}']:.6g}"), metadata[f"K{step_index}"]))
        elif family == "k":
            label_text = self._label_for_step_from_metadata(metadata, step_index, line_label)
            metadata[name] = {
                "type": "reaction",
                "index": step_index,
                "label": label_text,
                "line": line_index,
                "role": "k",
            }
            if commit:
                slider_updates.append((name, float(f"{resolved_values[name]:.6g}"), metadata[name]))

        if not commit:
            return new_text

        self.set_variable_metadata(metadata)

        self._mechanism_editor.set_reactions_text(new_text, block_signals=True)

        for slider_name, slider_value, meta in slider_updates:
            if sliders is not None and sliders.has_variable(slider_name):
                sliders.update_variable(slider_name, slider_value)
                sliders.update_metadata(slider_name, meta)
                # Only persist user-editable slider values as overrides; derived/read-only
                # sliders (e.g., kf/kr implied by explicit K, or param-algebra constraints)
                # must never be rewritten back into the DSL.
                if not (isinstance(meta, dict) and meta.get("editable") is False):
                    self._preview_session.stage_slider_value(slider_name, slider_value)

        timer = getattr(self._mechanism_editor, "_network_update_timer", None)
        if timer is not None:
            timer.start()

        return new_text

    def _sanitize_mechanism_parameter_conflicts(self, text: str) -> tuple[str, "OrderedDict[str, float]", "OrderedDict[str, Dict[str, object]]"]:
        return self._variable_runtime.sanitize_mechanism_parameter_conflicts(text)

    def _extract_and_populate_variables(self, *, preserve_visibility: bool = False):
        return self._variable_runtime.extract_and_populate_variables(
            preserve_visibility=bool(preserve_visibility)
        )

    def _update_parameter_table_from_sliders(self) -> None:
        return self._variable_runtime.update_parameter_table_from_sliders()

    def _is_energy_mode_mechanism(self, mechanism: object) -> bool:
        return bool(self._variable_runtime.is_energy_mode_mechanism(mechanism))

    def _dsl_has_computational_mode_generated_block(self, dsl_text: str) -> bool:
        return bool(self._variable_runtime.dsl_has_computational_mode_generated_block(dsl_text))

    def _dsl_global_temperature_K(self, dsl_text: str) -> float | None:
        return self._variable_runtime.dsl_global_temperature_k(dsl_text)

    def _set_temperature_override_state(self, *, enabled: bool, tooltip: str | None = None) -> None:
        spin = getattr(self, "_temperature_spinbox", None)
        if spin is None:
            return
        spin.setEnabled(bool(enabled))
        if tooltip is not None:
            try:
                spin.setToolTip(str(tooltip))
            except RuntimeError as exc:
                logger.debug("Failed to set temperature tooltip: %s", exc, exc_info=True)
                self._temperature_tooltip_failed = True

    def _sync_energy_mode_temperature_from_mechanism(self, mechanism: object) -> None:
        return self._variable_runtime.sync_energy_mode_temperature_from_mechanism(mechanism)

    def _populate_energy_mode_variables_from_mechanism(
        self,
        mechanism: object,
        *,
        refresh_sliders: bool,
        preserve_visibility: bool = False,
    ) -> None:
        return self._variable_runtime.populate_energy_mode_variables_from_mechanism(
            mechanism,
            refresh_sliders=bool(refresh_sliders),
            preserve_visibility=bool(preserve_visibility),
        )

    # ===== Public API for setting data (compatibility) =====

    def _open_docs(self):
        """
        Open documentation in browser or show guidance if not yet published.
        """
        if DOCUMENTATION_URL:
            import webbrowser

            try:
                webbrowser.open(DOCUMENTATION_URL)
                logger.info(f"Opened documentation: {DOCUMENTATION_URL}")
                self._status_label.setText("Documentation opened in browser")
                return
            except Exception as e:
                logger.error(f"Failed to open documentation: {e}", exc_info=True)
                fallback = (
                    f"Online documentation ({DOCUMENTATION_URL}) could not be opened.\n\n"
                    "Available local resources:\n"
                    "- Help → Tutorials (guided workflows)\n"
                    "- README.md (project overview and installation)"
                )
                QtWidgets.QMessageBox.information(self, "Documentation", fallback)
                self._status_label.setText("Documentation URL unavailable")
        else:
            logger.info("Documentation requested but no online URL is configured")
            message = (
                f"Online documentation for Kindred v{KINDRED_VERSION} is not yet available.\n\n"
                "Available local resources:\n"
                "- Help → Tutorials for built-in walkthroughs\n"
                "- README.md for project overview and installation"
            )
            QtWidgets.QMessageBox.information(self, "Documentation", message)
            self._status_label.setText("Online documentation not yet available")

    def _update_recent_files_menu(self):
        """Update recent files menu from QSettings."""
        self.config_controller.update_recent_files_menu()

    def _add_to_recent_files(self, filepath: str):
        """Add filepath to recent files list."""
        self.config_controller.add_to_recent_files(filepath)

    def _load_recent_project(self, filepath: str):
        """Load a project from recent files."""
        self.project_controller.load_recent_project(filepath)

    def _clear_recent_files(self):
        """Clear recent files history."""
        self.config_controller.clear_recent_files()

    def _available_preset_ids(self) -> List[str]:
        """Return preset IDs discovered under kindred/data/presets."""
        if MainWindow._CACHED_PRESET_IDS is not None:
            return MainWindow._CACHED_PRESET_IDS
        default_ids = [f"M{i}" for i in range(1, 10)]
        try:
            from kindred.io.resources import get_resource_path
        except Exception as exc:  # pragma: no cover - defensive import
            logger.warning("Falling back to default preset list: %s", exc)
            return default_ids

        try:
            presets_dir = get_resource_path("presets")
        except FileNotFoundError as exc:
            logger.warning("Preset directory unavailable: %s", exc)
            return default_ids

        preset_files = sorted(presets_dir.glob("M*.txt"), key=self._preset_sort_key)
        if not preset_files:
            logger.warning("No preset files found in %s; defaulting to %s", presets_dir, default_ids)
            MainWindow._CACHED_PRESET_IDS = default_ids
            return default_ids

        MainWindow._CACHED_PRESET_IDS = [path.stem for path in preset_files]
        return MainWindow._CACHED_PRESET_IDS

    @staticmethod
    def _preset_sort_key(path: Path) -> Tuple[int, str]:
        """Sort presets numerically when possible."""
        match = re.match(r"^M(\\d+)$", path.stem)
        if match:
            return (0, int(match.group(1)))
        return (1, path.stem)

    def _show_tutorials(self):
        """Show tutorial selection dialog and launch selected tutorial."""
        from kindred.gui.widgets.tutorial_selection_dialog import TutorialSelectionDialog
        from kindred.gui.tutorial_manager import launch_tutorial

        dialog = TutorialSelectionDialog(self)

        # Connect signal to launch tutorial
        def on_tutorial_selected(tutorial_id: str):
            # Small delay to let dialog close
            QtCore.QTimer.singleShot(200, lambda: launch_tutorial(self, tutorial_id))

        dialog.tutorialSelected.connect(on_tutorial_selected)
        dialog.exec()

    def _about_brand_asset_path(self) -> str:
        """Return the bundled brand image best matched to the active theme."""
        theme_manager = getattr(self, "_theme_manager", None)
        is_dark = bool(theme_manager is not None and theme_manager.is_dark())
        if is_dark:
            return "assets/kindred-full-mark-dark.png"
        return "assets/kindred-full-mark.png"

    def _build_about_dialog(self) -> QtWidgets.QDialog:
        """Build the branded About dialog shown from Help -> About."""
        from kindred.io.resources import get_resource_path

        dialog = QtWidgets.QDialog(self)
        dialog.setObjectName("aboutKindredDialog")
        dialog.setWindowTitle("About Kindred")
        dialog.setMinimumWidth(_ABOUT_DIALOG_MIN_WIDTH)

        layout = QtWidgets.QVBoxLayout(dialog)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(12)

        brand_asset_path = self._about_brand_asset_path()
        dialog.setProperty("brand_asset_path", brand_asset_path)

        try:
            icon = QtGui.QIcon(str(get_resource_path("assets/kindred.ico")))
            if not icon.isNull():
                dialog.setWindowIcon(icon)
        except FileNotFoundError:
            pass

        brand_label = QtWidgets.QLabel(dialog)
        brand_label.setObjectName("aboutBrandImageLabel")
        brand_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        brand_label.setProperty("brand_asset_path", brand_asset_path)

        try:
            pixmap = QtGui.QPixmap(str(get_resource_path(brand_asset_path)))
        except FileNotFoundError:
            pixmap = QtGui.QPixmap()

        if not pixmap.isNull():
            brand_label.setPixmap(
                pixmap.scaled(
                    _ABOUT_DIALOG_IMAGE_MAX_SIZE,
                    _ABOUT_DIALOG_IMAGE_MAX_SIZE,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
        else:
            brand_label.setText("Kindred")

        title_label = QtWidgets.QLabel("Kindred", dialog)
        title_label.setObjectName("aboutTitleLabel")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_font = title_label.font()
        title_font.setBold(True)
        title_font.setPointSize(title_font.pointSize() + 6)
        title_label.setFont(title_font)

        version_label = QtWidgets.QLabel(f"Version {KINDRED_VERSION}", dialog)
        version_label.setObjectName("aboutVersionLabel")
        version_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        body_label = QtWidgets.QLabel(
            "Desktop GUI for kinetic modeling and fitting.\n"
            "MIT License.",
            dialog,
        )
        body_label.setObjectName("aboutBodyLabel")
        body_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        body_label.setWordWrap(True)

        button_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok, dialog)
        button_box.setObjectName("aboutDialogButtonBox")
        button_box.accepted.connect(dialog.accept)

        layout.addWidget(brand_label)
        layout.addWidget(title_label)
        layout.addWidget(version_label)
        layout.addWidget(body_label)
        layout.addWidget(button_box)

        return dialog

    def _show_about(self):
        self._build_about_dialog().exec()

    def _show_keyboard_shortcuts(self):
        """Show keyboard shortcuts reference dialog."""
        shortcuts_text = """
<h3>Keyboard Shortcuts</h3>

<h4>File Operations</h4>
<table>
<tr><td><b>Ctrl+O</b></td><td>Load Project</td></tr>
<tr><td><b>Ctrl+S</b></td><td>Save Project</td></tr>
<tr><td><b>Ctrl+E</b></td><td>Export CSV Data</td></tr>
<tr><td><b>Ctrl+Q</b></td><td>Exit</td></tr>
</table>

<h4>Simulation</h4>
<table>
<tr><td><b>Ctrl+R</b> or <b>F5</b></td><td>Run Simulation</td></tr>
<tr><td><b>Esc</b></td><td>Stop Simulation</td></tr>
</table>

<h4>Fitting</h4>
<table>
<tr><td><b>Ctrl+Shift+F</b></td><td>Run Parameter Fit</td></tr>
</table>

<h4>Tools</h4>
<table>
<tr><td><b>Ctrl+Shift+R</b></td><td>Reset Layout</td></tr>
<tr><td><b>Ctrl+,</b></td><td>Preferences (Mac)</td></tr>
</table>

<h4>Help</h4>
<table>
<tr><td><b>F1</b></td><td>Documentation</td></tr>
<tr><td><b>Ctrl+?</b></td><td>Keyboard Shortcuts</td></tr>
</table>
        """
        msg_box = QtWidgets.QMessageBox(self)
        msg_box.setWindowTitle("Keyboard Shortcuts")
        msg_box.setTextFormat(Qt.TextFormat.RichText)
        msg_box.setText(shortcuts_text)
        msg_box.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Ok)
        msg_box.exec()

    def _open_solver_settings(self):
        """Open solver settings dialog."""
        from kindred.gui.widgets.solver_settings import SolverSettingsDialog

        dialog = SolverSettingsDialog(self, cache_port=self._sim_controller)
        solver_contract = load_solver_contract()
        current_parameter_preview_debounce_ms = int(
            self._preview_session.variable_preview_debounce_ms("k1")
        )
        current_equilibrium_preview_debounce_ms = int(
            self._preview_session.variable_preview_debounce_ms("K1")
        )
        current_slider_preview_points = int(self._mechanism_editor.slider_points_value())
        current_slider_preview_solver = str(self._mechanism_editor.slider_solver_value() or "LSODA")

        # Populate dialog with current settings
        current_settings = {
            "solver": str(self._initial_solver or solver_contract.default_solver_name),
            'rtol': self._initial_rtol or 1e-6,
            'atol': self._initial_atol or 1e-12,
            'use_sparse_jacobian': self._use_sparse_jacobian,
            'wegscheider_cyclicity_enabled': bool(self._wegscheider_cyclicity_enabled),
            'max_parallel_batch_workers': int(self._sim_controller.parallel_batch.max_parallel_workers),
            'limit_blas_threads_per_worker': bool(self._sim_controller.parallel_batch.limit_blas_threads_per_worker),
            'slider_preview_solver': str(current_slider_preview_solver),
            'slider_preview_points': int(current_slider_preview_points),
            'parameter_preview_debounce_ms': int(current_parameter_preview_debounce_ms),
            'equilibrium_preview_debounce_ms': int(current_equilibrium_preview_debounce_ms),
            'result_cache_cap': int(self._sim_controller.batch_cache.result_cache.max_entries()),
            'preview_cache_cap': int(self._sim_controller.batch_cache.preview_cache.max_entries()),
        }
        dialog.set_settings(current_settings)

        if dialog.exec():
            settings = dialog.get_settings()
            current_solver = str(self._initial_solver or solver_contract.default_solver_name)
            current_rtol = float(self._initial_rtol or 1e-6)
            current_atol = float(self._initial_atol or 1e-12)
            current_sparse = bool(self._use_sparse_jacobian)
            current_wegscheider = bool(self._wegscheider_cyclicity_enabled)
            current_runtime_settings = {
                "solver": str(current_solver),
                "rtol": float(current_rtol),
                "atol": float(current_atol),
                "use_sparse_jacobian": bool(current_sparse),
                "wegscheider_cyclicity_enabled": bool(current_wegscheider),
                "slider_preview_solver": str(current_slider_preview_solver),
                "slider_preview_points": int(current_slider_preview_points),
                "parameter_preview_debounce_ms": int(current_parameter_preview_debounce_ms),
                "equilibrium_preview_debounce_ms": int(current_equilibrium_preview_debounce_ms),
            }
            next_runtime_settings = {
                "solver": str(settings.get('solver', current_solver) or current_solver),
                "rtol": float(settings.get('rtol', current_rtol) or current_rtol),
                "atol": float(settings.get('atol', current_atol) or current_atol),
                "use_sparse_jacobian": bool(settings.get('use_sparse_jacobian', current_sparse)),
                "wegscheider_cyclicity_enabled": bool(
                    settings.get('wegscheider_cyclicity_enabled', current_wegscheider)
                ),
                "slider_preview_solver": str(
                    settings.get('slider_preview_solver', current_slider_preview_solver) or current_slider_preview_solver
                ),
                "slider_preview_points": max(
                    50,
                    min(
                        20000,
                        int(settings.get('slider_preview_points', current_slider_preview_points)),
                    ),
                ),
                "parameter_preview_debounce_ms": max(
                    0,
                    min(
                        1000,
                        int(settings.get('parameter_preview_debounce_ms', current_parameter_preview_debounce_ms)),
                    ),
                ),
                "equilibrium_preview_debounce_ms": max(
                    0,
                    min(
                        1000,
                        int(
                            settings.get(
                                'equilibrium_preview_debounce_ms',
                                current_equilibrium_preview_debounce_ms,
                            )
                        ),
                    ),
                ),
            }
            if 'use_sparse_jacobian' in settings:
                self._use_sparse_jacobian = bool(settings['use_sparse_jacobian'])
            if 'wegscheider_cyclicity_enabled' in settings:
                self._wegscheider_cyclicity_enabled = bool(settings['wegscheider_cyclicity_enabled'])
            if 'max_parallel_batch_workers' in settings:
                try:
                    self._sim_controller.parallel_batch.max_parallel_workers = max(
                        1,
                        int(settings['max_parallel_batch_workers']),
                    )
                except Exception:
                    self._sim_controller.parallel_batch.max_parallel_workers = 12
            if 'limit_blas_threads_per_worker' in settings:
                self._sim_controller.parallel_batch.limit_blas_threads_per_worker = bool(
                    settings['limit_blas_threads_per_worker']
                )
            if 'result_cache_cap' in settings or 'preview_cache_cap' in settings:
                self.set_simulation_cache_caps(
                    result_cap=int(
                        settings.get(
                            'result_cache_cap',
                            self._sim_controller.batch_cache.result_cache.max_entries(),
                        )
                    ),
                    preview_cap=int(
                        settings.get(
                            'preview_cache_cap',
                            self._sim_controller.batch_cache.preview_cache.max_entries(),
                        )
                    ),
                    persist=True,
                )
            self.settings_set_value(
                "simulation/slider_preview_solver",
                str(next_runtime_settings["slider_preview_solver"]),
            )
            self.settings_set_value(
                "simulation/slider_preview_points",
                int(next_runtime_settings["slider_preview_points"]),
            )
            self.settings_set_value(
                "simulation/parameter_preview_debounce_ms",
                int(next_runtime_settings["parameter_preview_debounce_ms"]),
            )
            self.settings_set_value(
                "simulation/equilibrium_preview_debounce_ms",
                int(next_runtime_settings["equilibrium_preview_debounce_ms"]),
            )
            self._mechanism_editor.set_slider_solver_value(str(next_runtime_settings["slider_preview_solver"]))
            self._mechanism_editor.set_slider_points_value(int(next_runtime_settings["slider_preview_points"]))
            logger.info(f"Solver settings updated: {settings}")
            self._apply_solver_runtime_state(
                solver=settings.get('solver', self._initial_solver or solver_contract.default_solver_name),
                rtol=settings.get('rtol', self._initial_rtol or 1e-6),
                atol=settings.get('atol', self._initial_atol or 1e-12),
            )
            slider_schema_refresh_needed = bool(
                current_runtime_settings["wegscheider_cyclicity_enabled"]
                != next_runtime_settings["wegscheider_cyclicity_enabled"]
            )
            if slider_schema_refresh_needed:
                self._variable_runtime.clear_prepared_slider_runtime(dirty=True)
                try:
                    self._extract_and_populate_variables(preserve_visibility=True)
                    self._sync_mechanism_controls_to_focused_batch_set()
                except Exception:
                    logger.exception("Failed to refresh variables after solver settings update")
                    self._variable_runtime.clear_prepared_slider_runtime(dirty=True)
                    bar = getattr(self, "_status_bar", None)
                    if bar is not None:
                        try:
                            bar.showMessage("Failed to refresh variables after solver settings update (see logs)", 8000)
                        except RuntimeError as exc:
                            logger.debug("Failed to show solver settings refresh error in status bar: %s", exc, exc_info=True)
                            self._status_bar = None

    def _open_temperature_schedule_editor(self):
        """Open temperature schedule editor dialog."""
        from kindred.gui.widgets.temperature_schedule_editor import TemperatureScheduleDialog

        dialog = TemperatureScheduleDialog(self)

        # Connect signal to insert DSL into mechanism editor
        def on_schedule_created(dsl_text: str):
            # Get current mechanism text
            current_text = self._mechanism_editor._reactions_text.toPlainText()

            # Insert DSL at cursor position or append if empty
            cursor = self._mechanism_editor._reactions_text.textCursor()
            if cursor.hasSelection():
                # Replace selection
                cursor.insertText(dsl_text)
            elif current_text.strip():
                # Append with newline separator
                cursor.movePosition(QtGui.QTextCursor.End)
                if not current_text.endswith('\n'):
                    cursor.insertText('\n')
                cursor.insertText(dsl_text + '\n')
            else:
                # Insert at beginning
                cursor.insertText(dsl_text + '\n')

            logger.info(f"Temperature schedule inserted: {dsl_text}")

        dialog.scheduleCreated.connect(on_schedule_created)

        # Show dialog (non-modal to allow user to see mechanism editor)
        dialog.exec()

    def closeEvent(self, event):
        """
        Handle window close event - save settings and clean up worker threads.

        P3 ENHANCEMENT: Save user preferences before closing.
        """
        logger.info("Window close event - cleaning up")

        # P3 ENHANCEMENT: Save user preferences
        self._save_settings()

        close_ready = self._sim_controller.prepare_simulation_shutdown_for_close()
        if not close_ready:
            logger.warning("Deferring close event while simulation worker shutdown is still in progress")
            event.ignore()
            return

        fit_windows_ready = self._prepare_fit_window_shutdown_for_close()
        if not fit_windows_ready:
            logger.warning("Deferring close event while fit window shutdown is still in progress")
            event.ignore()
            return

        logger.info("Cleanup complete, accepting close event")
        event.accept()

    def _prepare_fit_window_shutdown_for_close(self) -> bool:
        tracked_windows = list(getattr(self, "_active_fit_windows", []) or [])
        all_closed = True

        for window in tracked_windows:
            if window is None:
                continue
            try:
                close_accepted = bool(window.close())
            except RuntimeError:
                continue
            except Exception as exc:
                logger.warning("Failed to close fit window during main-window shutdown: %s", exc, exc_info=True)
                all_closed = False
                continue

            if not close_accepted:
                all_closed = False
                continue

            try:
                if bool(window.isVisible()):
                    all_closed = False
            except RuntimeError:
                continue

        return all_closed

    # ========================================================================
    # PROFILE MANAGEMENT
    # ========================================================================

    def set_data(
        self,
        t: np.ndarray,
        series: Dict[str, np.ndarray],
        *,
        label: Optional[str] = None,
        overlays: Optional[Sequence[Dict[str, object]]] = None,
        owned_species: Optional[Sequence[str]] = None,
    ) -> None:
        self.results_controller.set_data(t, series, label=label, overlays=overlays, owned_species=owned_species)
