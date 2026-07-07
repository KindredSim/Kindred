# kindred/gui/main_window.py
# REDESIGNED GUI - Expert/Power-user focused with fitting capabilities

from __future__ import annotations

from dataclasses import dataclass
import json
import hashlib
import logging
import os
import platform
import re
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Callable, Sequence, TYPE_CHECKING, Mapping
import math

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt

from kindred import __version__ as KINDRED_VERSION
from kindred.core.mechanism_runtime_transition import (
    AuthoritativeMechanismSnapshot,
    MechanismTransitionOutcome,
    MechanismRuntimeTransitionService,
)
from kindred.core.runtime_defaults import MAX_PARALLEL_WORKERS_CEILING
from kindred.core.simulation_identity import canonical_initials_fingerprint
from kindred.core.simulator.dsl_text_update import (
    analyze_step_parameter_update,
    build_current_text_step_analysis_context,
    format_authoritative_parameter_value,
    materialize_generated_fast_dg_overrides_to_dsl_text,
)
from kindred.core.validation import try_parse_finite_float
from kindred.core.datasets.observation_payload import dense_view_from_observations, observations_from_payload
from kindred.gui.project_schema import (
    FITTING_DEFAULTS_KEYS,
    PROJECT_DEFAULTS,
    SIMULATION_TEMPERATURE_K_RANGE,
    validate_project_payload,
)

if TYPE_CHECKING:
    from kindred.core.mechanism_source import MechanismAuthoringSource
    from kindred.core.mechanism import Mechanism
    from kindred.gui.controllers.simulation_controller import SimulationController
    from kindred.gui.ports import SimulationUiPorts
    from kindred.gui.widgets.batch_initial_conditions_table import BatchInitialConditionsTableView

from kindred.gui.app_wiring import (
    build_bottom_analysis_dock,
    build_batch_initial_conditions,
    build_batch_dock_panel,
    build_mechanism_dock,
    build_right_dock_and_dataset_owners,
    build_settings_and_controllers,
    build_sliders_dock,
    build_simulation_plumbing,
    build_symbolic_calculator_dock,
    build_template_manager,
    build_window_shell,
    dock_default_area,
    dock_shell_specs,
    load_solver_contract,
)
from kindred.gui.mechanism_source_transaction_owner import MechanismSourceTransactionOwner
from kindred.gui.canonical_initials_transition_owner import CanonicalInitialsTransitionOwner
from kindred.gui.diagnostics import record_best_effort_failure as record_gui_best_effort_failure
from kindred.gui.reactions_draft_resolution_owner import (
    ReactionsDraftResolutionOwner,
)
from kindred.gui.fitting.runtime_inputs import (
    FittingEvaluatorRuntimeSettings,
    FittingRuntimeInputPublisher,
    FittingRuntimeInputs,
)
from kindred.gui.initial_conditions_import_owner import (
    InitialConditionsImportOwner,
)
from kindred.gui.initial_conditions_source_acceptance_owner import InitialConditionsSourceAcceptanceOwner
from kindred.gui.ports import (
    AuthoritativeRewriteOutcome,
    AuthoritativeRewriteStatus,
    DisplayRefreshSource,
    ReactionsDraftApplyResult,
)
from kindred.gui.main_window_mechanism_helpers import MainWindowMechanismHelpers
from kindred.gui.mechanism_session_owner import MechanismSessionOwner
from kindred.gui.main_window_preview_session import MainWindowPreviewSession
from kindred.gui.simulation_batch_owner import SimulationBatchOwner
from kindred.gui.simulation_dialogs import SimulationDialogs
from kindred.gui.simulation_mechanism_owner import SimulationMechanismOwner
from kindred.gui.simulation_provenance_owner import SimulationProvenanceOwner
from kindred.gui.simulation_run_ui_owner import SimulationRunUiOwner
from kindred.gui.simulation_settings_owner import SimulationSettingsOwner
from kindred.gui.simulation_runtime_settings_owner import RUNTIME_SETTING_UNSET, SimulationRuntimeSettingsOwner
from kindred.gui.main_window_variable_runtime import MainWindowVariableRuntime
from kindred.gui.mixins.ports import FittingMixinPorts
from kindred.gui.mixins.fitting_mixin import FittingMixin

logger = logging.getLogger(__name__)

__all__ = ["MainWindow"]

# Online documentation is not yet published.
DOCUMENTATION_URL: Optional[str] = None
_STARTUP_WIDTH_RATIO = 0.86
_STARTUP_HEIGHT_RATIO = 0.88
_MIN_STARTUP_WIDTH = 1280
_MIN_STARTUP_HEIGHT = 820
_FALLBACK_STARTUP_SIZE = QtCore.QSize(1440, 900)
_CANONICAL_BATCH_INITIALS_TRANSITION_DEBOUNCE_MS = 220
_INTERACTIVE_RUNTIME_READINESS_REFRESH_DEBOUNCE_MS = 80
_REACTIONS_DRAFT_DIRTY_REFRESH_MS = 100
_DRAFT_EDIT_WARNING_SETTINGS_KEY = "mechanism_editor/suppress_draft_edit_warning"


@dataclass(frozen=True)
class TemperatureIndicatorState:
    energy_mode_active: bool
    t_override_k: Optional[float]
    schedule_defined: bool
    schedule_kind: Optional[str]
    schedule_interval_count: Optional[int]
    constant_schedule_k: Optional[float]
    indicator_text: Optional[str]
_ANALYSIS_SURFACE_NAMES: tuple[str, ...] = ("Statistics", "Parameters")
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

    def __init__(self):
        super().__init__()
        self._settings_owner = SimulationSettingsOwner()
        self._init_simulation_plumbing_and_state()
        self._init_batch_initial_conditions()
        self._init_settings_and_controllers()
        self._init_template_manager()
        self._init_window_shell()
        self._init_simulation_controller()
        self._init_mechanism_dock_and_panel()
        self._init_sliders_dock()
        self._init_batch_dock_and_panel()
        self._init_fitting_runtime_input_publisher()
        self._init_right_dock_and_datasets()
        self._init_symbolic_calculator_dock()
        self._init_bottom_analysis_dock()
        self._init_status_bar_widgets()
        self._finish_window_composition()
        self._bootstrap_window_state()

        logger.info("Kindred MainWindow initialized (new design)")

    def _init_simulation_plumbing_and_state(self) -> None:
        self._preview_session = MainWindowPreviewSession(self)
        self._simulation_dialogs = SimulationDialogs(self)
        self._simulation_run_ui_owner = SimulationRunUiOwner(
            results_table_getter=lambda: getattr(self, "_results_table", None),
        )
        self._variable_runtime = MainWindowVariableRuntime(self)
        self._reactions_draft_resolution_owner = ReactionsDraftResolutionOwner(
            draft_dirty=self._reactions_draft_dirty,
            prompt_draft_decision=self._prompt_reactions_draft_before_authoritative_change,
            discard_draft=self._discard_reactions_draft,
            apply_dirty_draft_with_transition=self._apply_reactions_draft_with_transition,
            apply_dirty_draft_without_transition=self._apply_reactions_draft_for_authoritative_rewrite,
            draft_error_text=self._reactions_draft_apply_error_text,
        )
        self._canonical_initials_transition_owner = CanonicalInitialsTransitionOwner(
            batch_set_ids_for_rows=self._batch_set_ids_for_rows,
            canonical_initials_identity_by_set_id=self._canonical_batch_initials_identity_by_set_id,
            canonical_initials_identity_for_set_ids=self._canonical_batch_initials_identity_for_set_ids,
            apply_authoritative_transition=self._apply_authoritative_mechanism_transition,
            apply_canonical_initials_transition_effects=self._apply_canonical_batch_initials_transition_effects,
            qt_parent=self,
        )
        self._mechanism_source_transaction_owner = MechanismSourceTransactionOwner(
            mechanism_session_owner_getter=lambda: getattr(self, "_mechanism_session_owner", None),
            mechanism_editor_getter=lambda: getattr(self, "_mechanism_editor", None),
            undo_stack_getter=lambda: self._undo_stack,
            draft_resolution_owner=self._reactions_draft_resolution_owner,
            refresh_reactions_draft_state=self._refresh_reactions_draft_state,
            get_authoritative_suppress=lambda: bool(
                getattr(self, "_suppress_authoritative_mechanism_input_change", False)
            ),
            set_authoritative_suppress=lambda value: setattr(
                self,
                "_suppress_authoritative_mechanism_input_change",
                bool(value),
            ),
            get_programmatic_text_suppress=lambda: bool(
                getattr(self, "_suppress_programmatic_text_change_signals", False)
            ),
            set_programmatic_text_suppress=lambda value: setattr(
                self,
                "_suppress_programmatic_text_change_signals",
                bool(value),
            ),
            publish_transition=(
                self._canonical_initials_transition_owner.publish_authoritative_transition
            ),
            apply_transition=self._apply_authoritative_mechanism_transition,
            publish_aborted_source_load_attempt=(
                self._canonical_initials_transition_owner.publish_aborted_source_load_attempt
            ),
            execute_source_load_transaction=(
                self._canonical_initials_transition_owner.execute_source_load_transaction
            ),
            guard_slider_transaction_invalidation=self._guard_slider_transaction_invalidation,
            prepare_source_load_reconciliation=self._prepare_authoritative_source_load_reconciliation,
            apply_source_load_reconciliation_after_acceptance=(
                self._apply_authoritative_source_load_reconciliation_after_source_acceptance
            ),
            refresh_after_source_load=self._refresh_after_authoritative_source_load,
            source_load_plan_materializes_initial_conditions=(
                self._source_load_plan_materializes_initial_conditions
            ),
            set_status_text=lambda text: self._status_label.setText(str(text)),
        )
        self._mechanism_helpers = MainWindowMechanismHelpers()
        self._simulation_mechanism_owner = SimulationMechanismOwner(
            mechanism_session_owner_getter=lambda: getattr(self, "_mechanism_session_owner", None),
            mechanism_editor_getter=lambda: getattr(self, "_mechanism_editor", None),
            preview_session=self._preview_session,
            variable_runtime=self._variable_runtime,
            apply_dirty_reactions_for_run=self._apply_dirty_reactions_for_run,
            pending_initials_for_source_set=self._pending_initials_for_source_set,
            apply_reactions_overrides_to_text=self._apply_reactions_overrides_to_text,
            apply_wegscheider_resolution_reactions_rewrite=(
                self._apply_wegscheider_resolution_reactions_rewrite
            ),
        )
        self._simulation_runtime_settings_owner = SimulationRuntimeSettingsOwner(
            temperature_getter=lambda: float(self._temperature_spinbox.value()),
            num_points_getter=lambda: int(self._num_points_spinbox.value()),
            sim_time_text_getter=lambda: str(self._sim_time_spinbox.text()),
            parse_sim_time_seconds=self._parse_sim_time_seconds,
            dsl_global_temperature_getter=self._variable_runtime.dsl_global_temperature_k,
        )
        self._last_fit_metadata: Optional[Dict[str, Any]] = None
        self._reactions_draft_apply_failed = False
        self._reactions_draft_warning_prompted = False
        self._reactions_draft_refresh_timer = QtCore.QTimer(self)
        self._reactions_draft_refresh_timer.setSingleShot(True)
        self._reactions_draft_refresh_timer.setInterval(_REACTIONS_DRAFT_DIRTY_REFRESH_MS)
        self._reactions_draft_refresh_timer.timeout.connect(self._refresh_reactions_draft_state)
        self._simulation_provenance_owner = SimulationProvenanceOwner(
            dataset_snapshot_getter=self._snapshot_datasets,
            fit_metadata_getter=lambda: self._last_fit_metadata,
        )
        self._mechanism_runtime_transition = MechanismRuntimeTransitionService()
        self._authoritative_mechanism_runtime_refresh_defer_depth = 0
        self._simulation_runtime_input_refresh_defer_depth = 0
        self._pending_interactive_runtime_readiness_rows: tuple[int, ...] = ()
        self._interactive_runtime_readiness_refresh_queued = False
        self._last_interactive_runtime_readiness_signature: tuple[tuple[int, ...], tuple[str, ...]] | None = None
        self._interactive_runtime_readiness_refresh_timer = QtCore.QTimer(self)
        self._interactive_runtime_readiness_refresh_timer.setSingleShot(True)
        self._interactive_runtime_readiness_refresh_timer.timeout.connect(
            self._flush_interactive_runtime_readiness_refresh
        )

        # Best-effort failure recording (GUI hardening).
        self._best_effort_failures: set[str] = set()
        self._best_effort_failure_counts: Dict[str, int] = {}

        # Mechanism + simulation state.
        self._suppress_preference_updates = False  # Guard: True during document apply and settings load.
        self._pre_dsl_temperature: float | None = None  # Spinbox value before T= override.
        self._temperature_dsl_override_active = False
        self._fitting_defaults: Dict[str, object] = {}
        self._last_batch_results: List[Dict[str, Any]] = []
        self._simulation_runtime_preview_ready = True

        # Registry of actions that support customizable shortcuts.
        self._shortcut_actions: Dict[str, Dict[str, Any]] = {}

    def _init_simulation_controller(self) -> None:
        # Simulation execution, batch orchestration, caching, and worker lifecycle.
        plumbing = build_simulation_plumbing(self)
        self._sim_ui_port: SimulationUiPorts = plumbing.ui_port
        self._sim_controller = plumbing.controller
        self._sim_controller.runtime_readiness_render_requested.connect(
            self._apply_simulation_runtime_readiness_render_state
        )

    def _init_batch_initial_conditions(self) -> None:
        # Batch initial conditions (source-of-truth for initials after first migration).
        batch_components = build_batch_initial_conditions(self)
        self._batch_store = batch_components.store
        self._batch_model = batch_components.model
        self._initial_conditions_import_owner = InitialConditionsImportOwner(
            batch_store_getter=lambda: self._batch_store,
            batch_model_getter=lambda: self._batch_model,
            confirm_overwrite=self._confirm_initial_condition_import_overwrite,
            notify_rows_changed=self._notify_batch_initial_rows_changed,
            notify_canonical_rows_changed=self._notify_canonical_batch_initial_rows_changed,
            sync_species_columns=self._sync_batch_species_columns,
            temperature_getter=lambda: float(self._temperature_spinbox.value()),
        )
        self._initial_conditions_source_acceptance_owner = InitialConditionsSourceAcceptanceOwner(
            import_owner=self._initial_conditions_import_owner,
            show_reconciliation_error=self._show_initial_conditions_reconciliation_error,
        )
        self._simulation_batch_owner = SimulationBatchOwner(
            batch_selected_rows=self._batch_selected_rows,
            requested_show_batch_set_ids=self._requested_show_batch_set_ids,
            slider_edit_target_set_ids=self._slider_edit_target_set_ids,
            active_batch_set_id=self._active_batch_set_id,
            batch_current_row=self._batch_current_row,
            batch_set_id_for_row=self._batch_set_id_for_row,
            batch_set_name_for_id=self._batch_set_name_for_id,
            batch_set_id_for_name=self._batch_set_id_for_name,
            batch_preferred_primary_set_id=self._batch_preferred_primary_set_id,
            batch_cache_key=self._batch_cache_key,
            batch_cache_getter=lambda: self._sim_controller.batch_cache,
            batch_store_getter=lambda: self._batch_store,
            batch_model_getter=lambda: self._batch_model,
            batch_initials_for_row=self._batch_initials_for_row,
            preview_session=self._preview_session,
            preview_launch_pending=lambda: bool(
                getattr(
                    getattr(self.simulation_controller.run_state, "pending_slider_preview_launch", None),
                    "active",
                    False,
                )
            ),
            mechanism_owner=self._simulation_mechanism_owner,
            solver_owner=self._simulation_runtime_settings_owner,
            update_batch_row_controls_state=self._update_batch_row_controls_state,
            sync_batch_species_columns=self._sync_batch_species_columns,
            sync_mechanism_controls_to_focused_batch_set=self._sync_mechanism_controls_to_focused_batch_set,
        )
        self._batch_table: Optional[BatchInitialConditionsTableView] = None
        self._last_effective_slider_edit_target_set_ids: tuple[str, ...] = ()
        self._connected_batch_semantics_model = None
        self._connected_batch_selection_model = None
        # (Batch run/caching/executor state lives in self._sim_controller.)
        self._set_species_panel_transaction_owner()

    def _init_settings_and_controllers(self) -> None:
        # Theme state.
        self._dark_mode = False  # Will be loaded from settings.

        # Configuration persistence + settings-driven UI application.
        controllers = build_settings_and_controllers(self)
        self.config_controller = controllers.config_controller
        self.project_controller = controllers.project_controller

        # Menu/UI objects that controllers may reference.
        self._recent_menu = None

        # Undo/Redo stack for high-level operations.
        self._undo_stack = controllers.undo_stack

    def _init_template_manager(self) -> None:
        managers = build_template_manager()
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
            set_copy_all_provider(self.results_controller.build_main_plot_copy_all_export_plan)
        set_copy_status_text = getattr(main_plot, "set_copy_status_text_callback", None)
        if callable(set_copy_status_text):
            set_copy_status_text(self.set_status_text)
        main_plot.referenceLayerVisibilityRequested.connect(
            self.results_controller.set_reference_overlays_visible
        )

    def _init_mechanism_dock_and_panel(self) -> None:
        mechanism_dock_components = build_mechanism_dock(self)
        self._mechanism_dock = mechanism_dock_components.dock
        self._mechanism_panel = mechanism_dock_components.panel

        self._mechanism_editor = self._mechanism_panel.editor
        self._mechanism_session_owner = MechanismSessionOwner(
            topology_validator=self._mechanism_session_topology_is_valid,
        )
        self._sync_mechanism_session_owner_from_widgets(authoritative=True)
        self._mechanism_runtime_transition.reset_current_snapshot(
            self._authoritative_mechanism_transition_snapshot(),
            canonical_batch_initials_by_set_id=self._canonical_batch_initials_identity_by_set_id(),
        )
        self._refresh_reactions_draft_state()
        self._sliders_panel = self._mechanism_editor.detach_slider_pane_for_dock()
        self._species_panel_available = True
        self._runtime_parameter_buttons_available = True

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
        batch_dock_components = build_batch_dock_panel(
            self,
            batch_model=self._batch_model,
            current_solver=self._simulation_runtime_settings_owner.current_solver_name(),
            on_add_batch_set=self._add_batch_set,
            on_move_selected_batch_sets_up=self._move_selected_batch_sets_up,
            on_move_selected_batch_sets_down=self._move_selected_batch_sets_down,
            on_delete_selected_batch_sets=self._delete_selected_batch_sets,
            on_run_selected=self._run_simulation_after_committing_batch_initials,
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

    def _init_fitting_runtime_input_publisher(self) -> None:
        self._fitting_runtime_input_publisher = FittingRuntimeInputPublisher(
            capture_inputs=self._capture_fitting_runtime_inputs,
            record_failure=self._record_fitting_best_effort_failure,
        )

    @property
    def fitting_runtime_input_publisher(self) -> FittingRuntimeInputPublisher:
        return self._fitting_runtime_input_publisher

    def _capture_fitting_runtime_inputs(self) -> FittingRuntimeInputs:
        return FittingRuntimeInputs(
            evaluator=FittingEvaluatorRuntimeSettings(
                temperature_K=self._effective_fitting_runtime_temperature_K(),
                use_sparse_jacobian=self._require_global_fit_bool(
                    self._simulation_runtime_settings_owner.use_sparse_jacobian(),
                    "use_sparse_jacobian",
                ),
                wegscheider_cyclicity_enabled=self._require_global_fit_bool(
                    self._simulation_runtime_settings_owner.wegscheider_cyclicity_enabled(),
                    "wegscheider_cyclicity_enabled",
                ),
            ),
            batch_runtime_lane_budget=int(self._sim_controller.batch_runtime_lane_budget),
        )

    def _effective_fitting_runtime_temperature_K(self) -> float:
        try:
            source = self.canonical_mechanism_source()
            t_override_k = self._variable_runtime.dsl_global_temperature_k(source.full_dsl)
        except Exception:
            t_override_k = None
        if t_override_k is not None:
            return self._require_global_fit_temperature(t_override_k)
        return self._require_global_fit_temperature(self._temperature_spinbox.value())

    def _wire_mechanism_editor_signals(self) -> None:
        sliders = self._mechanism_editor._variable_sliders
        sliders.variableChanged.connect(self._on_variable_changed)
        sliders.sliderDragStarted.connect(self._on_slider_drag_started)
        sliders.sliderDragFinished.connect(self._on_slider_drag_finished)

        try:
            self._mechanism_editor._commit_runtime_parameters_btn.clicked.connect(self._on_commit_runtime_parameters_clicked)
            self._mechanism_editor._reset_runtime_parameters_btn.clicked.connect(self._on_reset_runtime_parameters_clicked)
        except Exception:
            logger.exception("Failed to connect runtime parameter buttons")
            self._runtime_parameter_buttons_available = False
            self._disable_runtime_parameter_buttons()

        self._mechanism_editor.speciesResetRequested.connect(self._on_species_reset_requested)
        self._mechanism_editor.validationStateChanged.connect(
            lambda _state: self._simulation_run_ui_owner.refresh_run_button_state()
        )
        set_reactions_guard = getattr(self._mechanism_editor, "set_reactions_user_mutation_guard", None)
        if callable(set_reactions_guard):
            set_reactions_guard(self._reactions_user_mutation_allowed)
        self._mechanism_editor.reactionsApplyRequested.connect(self._apply_reactions_draft)
        self._mechanism_editor.reactionsDiscardRequested.connect(self._discard_reactions_draft)
        self._mechanism_editor.run_btn.clicked.connect(self._run_simulation_after_committing_batch_initials)
        self._mechanism_editor.symbolic_calculator_btn.clicked.connect(self._open_symbolic_calculator_panel)
        self._mechanism_editor.mechanismInspectRequested.connect(self._open_mechanism_inspector)
        self._refresh_slider_transaction_button_state()

    def _set_runtime_parameter_buttons_enabled(self, enabled: bool) -> None:
        for attr in ("_commit_runtime_parameters_btn", "_reset_runtime_parameters_btn"):
            btn = getattr(self._mechanism_editor, attr, None)
            if btn is None:
                continue
            try:
                btn.setEnabled(bool(enabled))
            except RuntimeError as exc:
                self._record_best_effort_failure(
                    f"main_window.runtime_parameter_buttons.set_enabled.{attr}",
                    message="Failed to update runtime parameter button state",
                    exc=exc,
                )

    def _disable_runtime_parameter_buttons(self) -> None:
        self._set_runtime_parameter_buttons_enabled(False)

    def _refresh_slider_transaction_button_state(self) -> None:
        if not bool(getattr(self, "_runtime_parameter_buttons_available", False)):
            self._disable_runtime_parameter_buttons()
            return
        preview = getattr(self, "_preview_session", None)
        dirty = bool(preview is not None and hasattr(preview, "has_dirty_transaction") and preview.has_dirty_transaction())
        self._set_runtime_parameter_buttons_enabled(bool(dirty))

    def _mechanism_session_texts_from_widgets(self) -> tuple[str, str]:
        source = self._mechanism_source_transaction_owner.source_from_widgets()
        return str(source.reactions_text), str(source.state_network_dsl)

    def _mechanism_session_topology_is_valid(self) -> bool:
        editor = getattr(self, "_mechanism_editor", None)
        state_editor = getattr(editor, "_state_network_editor", None)
        if state_editor is None or not hasattr(state_editor, "is_valid"):
            raise RuntimeError("State network editor is unavailable.")
        return bool(state_editor.is_valid())

    def _sync_mechanism_session_owner_from_widgets(self, *, authoritative: bool) -> None:
        self._mechanism_source_transaction_owner.sync_from_widgets(
            authoritative=bool(authoritative)
        )

    def _sync_canonical_state_network_from_widget(self) -> None:
        self._mechanism_source_transaction_owner.sync_state_network_from_widget()

    def _refresh_authoritative_mechanism_derived_ui(self) -> None:
        self._update_temperature_mode_indicator()
        self._refresh_overlay_swatches_for_current_mechanism()

    def _authoritative_mechanism_transition_snapshot(self) -> AuthoritativeMechanismSnapshot:
        return AuthoritativeMechanismSnapshot.from_source(self.canonical_mechanism_source())

    def _authoritative_mechanism_runtime_refresh_deferred(self) -> bool:
        return int(getattr(self, "_authoritative_mechanism_runtime_refresh_defer_depth", 0) or 0) > 0

    def _consume_pending_authoritative_mechanism_runtime_refresh(self) -> bool:
        return self._mechanism_runtime_transition.consume_pending_readiness_epoch() is not None

    def _canonical_batch_initials_identity_by_set_id(self) -> Dict[str, str]:
        identities: Dict[str, str] = {}
        store = getattr(self, "_batch_store", None)
        if store is None:
            return identities
        try:
            row_count = int(store.row_count())
        except Exception:
            return identities
        for row in range(row_count):
            set_id = str(self._batch_set_id_for_row(int(row)) or "").strip()
            if not set_id:
                continue
            try:
                identities[set_id] = canonical_initials_fingerprint(self.batch_initials_for_row(int(row)))
            except Exception as exc:
                self._record_best_effort_failure(
                    "main_window.runtime_transition.canonical_initials_identity",
                    message="Failed to fingerprint canonical batch initials for runtime transition",
                    exc=exc,
                )
        return identities

    def _canonical_batch_initials_identity_for_set_ids(
        self,
        set_ids: Sequence[str],
    ) -> Dict[str, str]:
        identities: Dict[str, str] = {}
        requested_ids = tuple(
            dict.fromkeys(str(set_id).strip() for set_id in (set_ids or ()) if str(set_id).strip())
        )
        if not requested_ids:
            return identities
        for set_id in requested_ids:
            try:
                row = self._batch_row_for_set_id(str(set_id))
            except Exception as exc:
                row = None
                self._record_best_effort_failure(
                    "main_window.runtime_transition.canonical_initials_identity.row_lookup",
                    message="Failed to resolve batch row for scoped canonical initial fingerprint",
                    exc=exc,
                )
            if row is None:
                continue
            try:
                identities[str(set_id)] = canonical_initials_fingerprint(
                    self.batch_initials_for_row(int(row))
                )
            except Exception as exc:
                self._record_best_effort_failure(
                    "main_window.runtime_transition.canonical_initials_identity.scoped",
                    message="Failed to fingerprint scoped canonical batch initials for runtime transition",
                    exc=exc,
                )
        return identities

    def _batch_set_ids_for_rows(self, rows: Sequence[int]) -> tuple[str, ...]:
        set_ids: list[str] = []
        seen: set[str] = set()
        for row in rows or ():
            try:
                set_id = str(self._batch_set_id_for_row(int(row)) or "").strip()
            except Exception as exc:
                self._record_best_effort_failure(
                    "main_window.runtime_transition.canonical_initials_identity.row_lookup",
                    message="Failed to resolve batch set id for canonical batch initials transition",
                    exc=exc,
                )
                continue
            if not set_id or set_id in seen:
                continue
            set_ids.append(set_id)
            seen.add(set_id)
        return tuple(set_ids)

    def _apply_authoritative_mechanism_transition(
        self,
        *,
        schedule_runtime_refresh: Optional[bool] = None,
        force_runtime_invalidation: bool = False,
        transition_source: str = "authoritative_change",
        canonical_batch_initials_by_set_id: Optional[Mapping[str, object]] = None,
        affected_set_ids: Sequence[str] = (),
        canonical_batch_initials_scope_is_partial: bool = False,
    ) -> MechanismTransitionOutcome:
        if schedule_runtime_refresh is None:
            schedule_runtime_refresh = not self._authoritative_mechanism_runtime_refresh_deferred()
        outcome = self._mechanism_runtime_transition.apply_authoritative_transition(
            self._authoritative_mechanism_transition_snapshot(),
            source=str(transition_source or "authoritative_change"),
            force_runtime_invalidation=bool(force_runtime_invalidation),
            input_suppressed=bool(getattr(self, "_suppress_authoritative_mechanism_input_change", False)),
            schedule_runtime_refresh=bool(schedule_runtime_refresh),
            canonical_batch_initials_by_set_id=canonical_batch_initials_by_set_id,
            affected_set_ids=tuple(str(set_id) for set_id in (affected_set_ids or ()) if str(set_id)),
            canonical_batch_initials_scope_is_partial=bool(canonical_batch_initials_scope_is_partial),
        )
        self._apply_authoritative_runtime_transition_lifecycle(outcome)
        if outcome.display_cache_invalidation_allowed and self.results_controller.authoritative_result_transition_required(
            cache_stale_scope_is_global=bool(outcome.cache_stale_scope_is_global),
            cache_stale_set_ids=outcome.cache_stale_set_ids,
        ):
            self._apply_authoritative_result_truth_effects(
                cache_stale_set_ids=outcome.cache_stale_set_ids,
                cache_stale_scope_is_global=outcome.cache_stale_scope_is_global,
                display_clear_set_ids=outcome.display_clear_set_ids,
                display_clear_scope_is_global=outcome.display_clear_scope_is_global,
                clear_cached_mechanism=bool(outcome.runtime_invalidation_required),
            )
        transition_suppressed_noop = (
            str(transition_source or "authoritative_change") == "authoritative_change"
            and bool(
                getattr(self, "_suppress_authoritative_mechanism_input_change", False)
            )
            and not bool(outcome.runtime_invalidation_required)
            and not bool(outcome.runtime_input_invalidation_required)
        )
        if (
            not transition_suppressed_noop
            and not (outcome.runtime_input_invalidation_required and not outcome.runtime_invalidation_required)
        ):
            self._refresh_authoritative_mechanism_derived_ui()
        self._refresh_symbolic_calculator_state()
        if (
            bool(outcome.runtime_input_invalidation_required)
            or bool(outcome.runtime_invalidation_required)
        ):
            self._fitting_runtime_input_publisher.publish_current(
                reason="authoritative mechanism transition",
                force=True,
            )
        return outcome

    def _apply_authoritative_runtime_transition_lifecycle(
        self,
        outcome: MechanismTransitionOutcome,
    ) -> None:
        """Publish runtime identity before scheduling readiness against that identity."""
        if outcome.active_work_supersede_required:
            if outcome.runtime_invalidation_required:
                self._sim_controller.supersede_active_work_for_authoritative_mechanism_transition(
                    epoch=int(outcome.epoch),
                )
            else:
                self._sim_controller.supersede_active_work_for_authoritative_mechanism_transition(
                    epoch=int(outcome.epoch),
                    affected_set_ids=outcome.active_work_supersede_set_ids,
                    close_preview_runtime_owner=False,
                )
        if bool(outcome.dirty_preview_reset_scope_is_global):
            try:
                self._preview_session.clear_working_transaction(
                    invalidate_preview_work=False
                )
            except Exception as exc:
                self._record_best_effort_failure(
                    "main_window.authoritative_transition.clear_global_dirty_preview",
                    message="Failed to discard global dirty slider work after authoritative mechanism source change",
                    exc=exc,
                )
        if outcome.runtime_invalidation_required and outcome.readiness_schedule_required:
            self._sim_controller.simulation_runtime_inputs_changed(
                batch_runtime_pool_inputs_changed=False
            )
        elif outcome.runtime_invalidation_required and outcome.readiness_schedule_deferred:
            self._mark_launch_readiness_stale()

    def _apply_canonical_batch_initials_transition_effects(
        self,
        outcome: MechanismTransitionOutcome,
        *,
        affected_set_ids: Sequence[str],
        discard_dirty_preview: bool,
    ) -> None:
        affected_set_ids_t = tuple(
            dict.fromkeys(str(set_id) for set_id in (affected_set_ids or ()) if str(set_id))
        )
        invalidates_runtime_inputs = bool(
            outcome.runtime_input_invalidation_required
            or outcome.runtime_invalidation_required
        )
        if invalidates_runtime_inputs:
            self._last_interactive_runtime_readiness_signature = None
            self._maybe_queue_run_target_runtime_readiness_refresh(force=True)
        if bool(discard_dirty_preview):
            if bool(outcome.dirty_preview_reset_scope_is_global):
                try:
                    self.results_controller.clear_active_display_transaction()
                except Exception as exc:
                    self._record_best_effort_failure(
                        "main_window.canonical_initials.clear_global_display",
                        message="Failed to clear display after authoritative source change",
                        exc=exc,
                    )
                self._refresh_slider_transaction_button_state()
                if affected_set_ids_t:
                    self._refresh_species_sliders_after_canonical_initials_transition(
                        affected_set_ids_t
                    )
                return
            dirty_preview_reset_set_ids = outcome.dirty_preview_reset_set_ids
            if not dirty_preview_reset_set_ids:
                return
            preview_state_changed = False
            try:
                preview_state_changed = bool(
                    self._preview_session.clear_working_transaction_for_set_ids(
                        dirty_preview_reset_set_ids,
                        invalidate_preview_work=False,
                    )
                )
            except Exception as exc:
                self._record_best_effort_failure(
                    "main_window.canonical_initials.discard_dirty_preview",
                    message="Failed to discard dirty preview state after canonical batch initial edit",
                    exc=exc,
                )
            if preview_state_changed:
                self.results_controller.clear_display_if_workspace_previews_were_displayed(
                    dirty_preview_reset_set_ids,
                )
                self._refresh_species_sliders_after_canonical_initials_transition(dirty_preview_reset_set_ids)

    def _refresh_species_sliders_after_canonical_initials_transition(
        self,
        affected_set_ids: Sequence[str],
    ) -> None:
        affected = {str(set_id) for set_id in (affected_set_ids or ()) if str(set_id)}
        if not affected:
            return
        focused_set_id = str(self._preview_session.focused_mechanism_workspace_set_id() or "").strip()
        current_set_id = ""
        try:
            current_row = self._batch_current_row()
            if current_row is not None:
                current_set_id = str(self._batch_set_id_for_row(int(current_row)) or "").strip()
        except Exception:
            current_set_id = ""
        effective_slider_target_ids: set[str] = set()
        try:
            effective_slider_target_ids = {
                str(set_id)
                for set_id in (self._effective_slider_edit_target_set_ids() or ())
                if str(set_id)
            }
        except Exception:
            effective_slider_target_ids = set()
        if (
            focused_set_id not in affected
            and current_set_id not in affected
            and not bool(affected & effective_slider_target_ids)
        ):
            return
        try:
            panel = self._mechanism_editor.species_sliders_widget()
            if panel is not None and hasattr(panel, "refresh_current_row_from_model"):
                panel.refresh_current_row_from_model(recompute_ranges=True)
            elif panel is not None and hasattr(panel, "rebuild_from_current_row"):
                panel.rebuild_from_current_row()
        except Exception as exc:
            self._record_best_effort_failure(
                "main_window.canonical_initials.refresh_species_sliders",
                message="Failed to refresh species sliders after canonical batch initial edit",
                exc=exc,
            )

    def _sync_mechanism_session_owner_after_authoritative_widget_write(
        self,
        *,
        dispatch_consumers: bool,
    ) -> None:
        self._mechanism_source_transaction_owner.sync_after_authoritative_widget_write(
            dispatch_consumers=bool(dispatch_consumers)
        )

    def _restore_mechanism_widgets_from_owner_canonical(self) -> None:
        self._mechanism_source_transaction_owner.restore_widgets_from_canonical()

    def _on_reactions_text_changed_for_main_window(self) -> None:
        if bool(getattr(self, "_suppress_programmatic_text_change_signals", False)):
            self._queue_reactions_draft_refresh()
            return
        self._reactions_draft_apply_failed = False
        self._queue_reactions_draft_refresh()

    def _on_state_network_changed_for_main_window(self) -> None:
        self._sync_canonical_state_network_from_widget()
        self._apply_authoritative_mechanism_transition()
        self._refresh_reactions_draft_state()

    def _on_temperature_spinbox_value_changed_for_main_window(self) -> None:
        self._update_temperature_mode_indicator()
        self._fitting_runtime_input_publisher.publish_if_changed(reason="temperature changed")
        self._refresh_simulation_runtime_after_inputs_changed(
            batch_runtime_pool_inputs_changed=False
        )

    @staticmethod
    def _state_network_dialog_info_text() -> str:
        return (
            "Edit the state network with full validation. State Network edits apply immediately; "
            "Reactions drafts are applied separately."
        )

    def _pending_initials_for_source_set(self, source, *, set_name: str) -> dict[str, float]:
        return self._initial_conditions_import_owner.pending_initials_for_source_set(
            source,
            set_name=str(set_name or ""),
        )

    def _canonical_reactions_text(self) -> str:
        owner = getattr(self, "_mechanism_session_owner", None)
        if owner is None:
            raise RuntimeError("Mechanism session owner is unavailable.")
        return str(owner.canonical_reactions_text)

    def _editor_reactions_text(self) -> str:
        editor = getattr(self, "_mechanism_editor", None)
        if editor is None:
            return ""
        return str(editor.reactions_text())

    def _reactions_draft_dirty(self) -> bool:
        owner = getattr(self, "_mechanism_session_owner", None)
        editor = getattr(self, "_mechanism_editor", None)
        if owner is None or editor is None:
            return False
        return str(editor.reactions_text()) != str(owner.canonical_reactions_text)

    def _queue_reactions_draft_refresh(self) -> None:
        timer = getattr(self, "_reactions_draft_refresh_timer", None)
        if timer is not None:
            timer.start()
            return
        self._refresh_reactions_draft_state()

    def _format_reactions_applied_status(self, validation) -> str:
        return (
            "✓ Applied: "
            f"{int(getattr(validation, 'species_count', 0) or 0)} species, "
            f"{int(getattr(validation, 'reaction_count', 0) or 0)} reactions, "
            f"{int(getattr(validation, 'equilibria_count', 0) or 0)} equilibria\n"
            "Runs use this mechanism."
        )

    def _refresh_mechanism_run_launch_readiness(self, *, dirty: Optional[bool] = None) -> None:
        run_ui = getattr(self, "_simulation_run_ui_owner", None)
        owner = getattr(self, "_mechanism_session_owner", None)
        if run_ui is None or owner is None:
            return
        dirty_b = self._reactions_draft_dirty() if dirty is None else bool(dirty)
        if dirty_b:
            run_ui.render_draft_run_preflight_available(
                not bool(getattr(self, "_reactions_draft_apply_failed", False)),
                active=True,
            )
            run_ui.render_mechanism_launch_available(True, tooltip="")
            return
        run_ui.render_draft_run_preflight_available(False, active=False)
        ready = bool(owner.is_ready_for_explicit_run())
        tooltip = "" if ready else "Cannot run: mechanism has errors. Fix and try again."
        run_ui.render_mechanism_launch_available(ready, tooltip=tooltip)

    def _refresh_reactions_draft_state(self) -> None:
        owner = getattr(self, "_mechanism_session_owner", None)
        editor = getattr(self, "_mechanism_editor", None)
        if owner is None or editor is None:
            return
        dirty = self._reactions_draft_dirty()
        was_dirty = bool(getattr(self, "_reactions_draft_last_dirty", False))
        set_actions = getattr(editor, "set_reactions_draft_actions_enabled", None)
        if callable(set_actions):
            set_actions(bool(dirty))
        if bool(dirty):
            self._reactions_draft_last_dirty = True
            if not bool(getattr(self, "_reactions_draft_apply_failed", False)):
                editor.set_reactions_status_text(
                    "• Draft changes not applied\nAnalysis uses the applied mechanism; Run may apply a valid draft first.",
                    state="draft",
                )
            self._refresh_mechanism_run_launch_readiness(dirty=True)
            return
        self._reactions_draft_last_dirty = False
        self._reactions_draft_apply_failed = False
        validation = owner.validate_canonical()
        if validation.valid:
            editor.set_reactions_status_text(
                self._format_reactions_applied_status(validation),
                state="valid",
            )
        else:
            detail = str(validation.error_message or "Canonical mechanism is invalid.")
            editor.set_reactions_status_text(f"! Draft not applied: {detail}", state="invalid")
        self._refresh_mechanism_run_launch_readiness(dirty=False)
        self._refresh_symbolic_calculator_state()
        if was_dirty and not bool(
            getattr(self, "_suppress_dirty_to_clean_runtime_readiness_refresh", False)
        ):
            self._maybe_queue_run_target_runtime_readiness_refresh(force=True)

    def _set_reactions_widget_text_from_canonical(self) -> None:
        self._mechanism_source_transaction_owner.set_reactions_widget_text_from_canonical()

    def _set_reactions_widget_text(self, text: str) -> None:
        self._mechanism_source_transaction_owner.set_reactions_widget_text(str(text))

    def _reactions_user_mutation_allowed(self) -> bool:
        if bool(getattr(self, "_reactions_draft_warning_prompted", False)):
            return True
        settings = QtCore.QSettings("Kindred", "KindredGUI")
        if settings.value(_DRAFT_EDIT_WARNING_SETTINGS_KEY, False, type=bool):
            return True
        accepted, suppress = self._prompt_reactions_draft_edit_warning()
        if bool(accepted) and bool(suppress):
            settings.setValue(_DRAFT_EDIT_WARNING_SETTINGS_KEY, True)
        if not bool(accepted):
            self._reactions_draft_apply_failed = False
            self._refresh_reactions_draft_state()
            return False
        self._reactions_draft_warning_prompted = True
        return True

    def _prompt_reactions_draft_edit_warning(self) -> tuple[bool, bool]:
        box = QtWidgets.QMessageBox(self)
        box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        box.setWindowTitle("Reactions Draft")
        box.setText(
            "You are editing a draft. Analysis uses the applied mechanism until changes are applied; Run may apply a valid draft before launch."
        )
        checkbox = QtWidgets.QCheckBox("Don't show this again", box)
        box.setCheckBox(checkbox)
        start_btn = box.addButton("Start Editing", QtWidgets.QMessageBox.ButtonRole.AcceptRole)
        cancel_btn = box.addButton("Cancel", QtWidgets.QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(start_btn)
        box.setEscapeButton(cancel_btn)
        box.exec()
        return box.clickedButton() is start_btn, bool(checkbox.isChecked())

    def _set_reactions_apply_failure(self, message: str) -> None:
        editor = getattr(self, "_mechanism_editor", None)
        if editor is None:
            return
        detail = str(message or "Mechanism draft could not be applied.").strip()
        self._reactions_draft_apply_failed = True
        editor.set_reactions_draft_actions_enabled(True)
        editor.set_reactions_status_text(f"! Draft not applied: {detail}", state="invalid")
        self._refresh_mechanism_run_launch_readiness(dirty=True)
        self._refresh_symbolic_calculator_state()

    def _apply_reactions_draft(self) -> bool:
        result = self._apply_reactions_draft_with_transition()
        return bool(result)

    def _prepare_reactions_draft_for_canonical_acceptance(
        self,
    ) -> tuple[ReactionsDraftApplyResult, object | None, object | None]:
        owner = getattr(self, "_mechanism_session_owner", None)
        if owner is None:
            raise RuntimeError("Mechanism session owner is unavailable.")

        def _result(
            success: bool,
            affected_rows: tuple[int, ...] = (),
            *,
            source_applied: bool = False,
        ) -> ReactionsDraftApplyResult:
            return ReactionsDraftApplyResult(
                bool(success),
                tuple(int(row) for row in affected_rows),
                bool(source_applied),
            )

        if not self._reactions_draft_dirty():
            self._refresh_reactions_draft_state()
            return _result(True), None, None

        candidate_source = owner.source_for_reactions_text(self._editor_reactions_text())
        candidate_validation = owner.validate_source(candidate_source)
        if not candidate_validation.valid:
            self._set_reactions_apply_failure(candidate_validation.error_message)
            return _result(False), None, None
        if str(candidate_source.state_network_dsl or "").strip() and not owner.source_is_ready_for_explicit_run(
            candidate_source
        ):
            self._set_reactions_apply_failure("State network topology is invalid.")
            return _result(False), None, None

        reconciliation_plan = self._initial_conditions_source_acceptance_owner.prepare_for_authoritative_apply(
            candidate_source,
            prompt_overwrite=True,
        )
        if reconciliation_plan is None:
            self._set_reactions_apply_failure("Initial Conditions could not be reconciled.")
            return _result(False), None, None

        reconciled_source = reconciliation_plan.source
        validation = owner.validate_source(reconciled_source)
        if not validation.valid:
            self._set_reactions_apply_failure(validation.error_message)
            return _result(False), None, None
        if str(reconciled_source.state_network_dsl or "").strip() and not owner.source_is_ready_for_explicit_run(
            reconciled_source
        ):
            self._set_reactions_apply_failure("State network topology is invalid.")
            return _result(False), None, None

        return _result(True, source_applied=True), reconciled_source, reconciliation_plan

    def _accept_reactions_draft_source(self, source) -> None:
        self._write_complete_authoritative_mechanism_source_to_editor(source)
        previous_authoritative_suppress = bool(
            getattr(self, "_suppress_authoritative_mechanism_input_change", False)
        )
        previous_clean_readiness_suppress = bool(
            getattr(self, "_suppress_dirty_to_clean_runtime_readiness_refresh", False)
        )
        self._suppress_authoritative_mechanism_input_change = False
        self._suppress_dirty_to_clean_runtime_readiness_refresh = True
        try:
            self._reactions_draft_apply_failed = False
            self._refresh_reactions_draft_state()
        finally:
            self._suppress_authoritative_mechanism_input_change = previous_authoritative_suppress
            self._suppress_dirty_to_clean_runtime_readiness_refresh = previous_clean_readiness_suppress

    def _apply_prepared_reactions_draft_source_transaction(
        self,
        source,
        reconciliation_plan,
        result: ReactionsDraftApplyResult,
        *,
        transition_source: str,
        publish_on_success: bool,
    ) -> ReactionsDraftApplyResult:
        materialization_result_holder: dict[str, object | None] = {"result": None}
        previous_source = self.canonical_mechanism_source()

        def _commit_source() -> bool:
            self._accept_reactions_draft_source(source)
            return True

        def _materialize_initials() -> object | None:
            if reconciliation_plan is None:
                return None
            materialization_result = self._initial_conditions_source_acceptance_owner.apply_prepared_plan(
                reconciliation_plan,
                notify_canonical_rows_changed=False,
            )
            materialization_result_holder["result"] = materialization_result
            return materialization_result

        def _rollback_source() -> None:
            self._write_complete_authoritative_mechanism_source_to_editor(previous_source)

        self._canonical_initials_transition_owner.execute_source_load_transaction(
            transition_source=str(transition_source or "authoritative_change"),
            schedule_runtime_refresh=not self._authoritative_mechanism_runtime_refresh_deferred(),
            draft_apply_result=None,
            commit_source=_commit_source,
            materialize_initials=_materialize_initials,
            refresh_after_commit=lambda: None,
            publish_on_success=bool(publish_on_success),
            rollback_before_abort_publish=_rollback_source,
        )
        materialization_result = materialization_result_holder.get("result")
        affected_rows: tuple[int, ...] = tuple(
            int(row)
            for row in (getattr(materialization_result, "affected_rows", ()) or ())
        )
        return ReactionsDraftApplyResult(
            bool(result),
            affected_rows,
            bool(getattr(result, "source_applied", False)),
        )

    def _apply_reactions_draft_with_transition(self) -> ReactionsDraftApplyResult:
        result, source, reconciliation_plan = self._prepare_reactions_draft_for_canonical_acceptance()
        if bool(result) and source is not None:
            result = self._apply_prepared_reactions_draft_source_transaction(
                source,
                reconciliation_plan,
                result,
                transition_source="authoritative_change",
                publish_on_success=True,
            )
        return result

    def _apply_reactions_draft_for_authoritative_rewrite(self) -> ReactionsDraftApplyResult:
        result, source, reconciliation_plan = self._prepare_reactions_draft_for_canonical_acceptance()
        if bool(result) and source is not None:
            result = self._apply_prepared_reactions_draft_source_transaction(
                source,
                reconciliation_plan,
                result,
                transition_source="authoritative_rewrite_draft_apply",
                publish_on_success=False,
            )
        return result

    def _discard_reactions_draft(self) -> None:
        self._set_reactions_widget_text_from_canonical()
        self._reactions_draft_apply_failed = False
        self._refresh_reactions_draft_state()

    def _apply_dirty_reactions_for_run(self) -> ReactionsDraftApplyResult:
        result, source, reconciliation_plan = self._prepare_reactions_draft_for_canonical_acceptance()
        if bool(result) and source is not None:
            result = self._apply_prepared_reactions_draft_source_transaction(
                source,
                reconciliation_plan,
                result,
                transition_source="run_preflight_dirty_draft_apply",
                publish_on_success=False,
            )
        return result

    def _prompt_reactions_draft_before_authoritative_change(
        self,
        action_label: str,
        *,
        continue_label: str,
        apply_label: str | None = None,
        discard_label: str | None = None,
        informative_text: str | None = None,
    ) -> str:
        continue_label_s = str(continue_label or "Continue").strip() or "Continue"
        box = QtWidgets.QMessageBox(self)
        box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        box.setWindowTitle("Unapplied Reactions Draft")
        box.setText(f"{str(action_label)} while Reactions draft changes are not applied.")
        box.setInformativeText(
            str(informative_text or "Choose how to handle the draft before continuing.")
        )
        apply_label_s = str(apply_label or f"Apply Draft Then {continue_label_s}")
        discard_label_s = str(discard_label or f"Discard Draft and {continue_label_s}")
        apply_btn = box.addButton(
            apply_label_s,
            QtWidgets.QMessageBox.ButtonRole.AcceptRole,
        )
        discard_btn = box.addButton(
            discard_label_s,
            QtWidgets.QMessageBox.ButtonRole.DestructiveRole,
        )
        cancel_btn = box.addButton("Cancel", QtWidgets.QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(cancel_btn)
        box.setEscapeButton(cancel_btn)
        box.exec()
        clicked = box.clickedButton()
        if clicked is apply_btn:
            return "apply"
        if clicked is discard_btn:
            return "discard"
        return "cancel"

    def is_mechanism_valid_for_preview(self) -> bool:
        owner = getattr(self, "_mechanism_session_owner", None)
        if owner is None:
            return False
        return bool(owner.is_ready_for_preview())

    def _reactions_text_widget(self) -> Optional[QtWidgets.QPlainTextEdit]:
        editor = getattr(self, "_mechanism_editor", None)
        widget = getattr(editor, "_reactions_text", None)
        return widget if isinstance(widget, QtWidgets.QPlainTextEdit) else None

    def _prompt_slider_transaction_invalidation(self, action_text: str) -> str:
        box = QtWidgets.QMessageBox(self)
        box.setIcon(QtWidgets.QMessageBox.Warning)
        box.setWindowTitle("Pending Slider Changes")
        box.setText(f"{str(action_text)} will discard pending slider changes.")
        box.setInformativeText(
            "Apply saves the staged slider values before continuing.\n"
            "Discard clears it.\n"
            "Cancel keeps the current pending edits."
        )
        commit_btn = box.addButton("Apply", QtWidgets.QMessageBox.ButtonRole.AcceptRole)
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
            self._on_commit_runtime_parameters_clicked()
            if bool(preview.has_dirty_transaction()):
                self._simulation_dialogs.message_box_warning(
                    "Pending Slider Changes",
                    f"{str(action_text)} was canceled because pending slider changes could not be applied.",
                )
                return False
            return True

        self._discard_pending_slider_edits_for_invalidation(
            discard_entire_transaction=True,
        )
        if bool(preview.has_dirty_transaction()):
            self._simulation_dialogs.message_box_warning(
                "Pending Slider Changes",
                f"{str(action_text)} was canceled because pending slider changes could not be discarded.",
            )
            return False
        return True

    def _bind_simulation_panel_widgets(self) -> None:
        species_panel = self._try_wire_species_panel()

        sim_panel = self._batch_panel
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
        self._temperature_label = sim_panel.temperature_label

        self._rebind_batch_semantics_signal_bindings()

        if species_panel is not None and hasattr(species_panel, "attach"):
            try:
                species_panel.attach(table=self._batch_table, model=self._batch_model)
                if hasattr(species_panel, "activate"):
                    self._select_initial_batch_row_if_needed()
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
                ("canonicalInitialsChanged", self._on_batch_canonical_initials_changed),
                ("initialRowsEdited", self._on_batch_initial_rows_edited),
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
                current_model.canonicalInitialsChanged.connect(self._on_batch_canonical_initials_changed)
                initial_rows_edited = getattr(current_model, "initialRowsEdited", None)
                if initial_rows_edited is not None:
                    initial_rows_edited.connect(self._on_batch_initial_rows_edited)
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

    def _species_panel_transaction_owner(self) -> object:
        return getattr(self, "_simulation_batch_owner", None) or self._preview_session

    def _set_species_panel_transaction_owner(self) -> None:
        try:
            species_panel = self._mechanism_editor.species_sliders_widget()
        except (AttributeError, RuntimeError, TypeError):
            return
        if species_panel is not None and hasattr(species_panel, "set_transaction_owner"):
            species_panel.set_transaction_owner(self._species_panel_transaction_owner())

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
                species_panel.set_transaction_owner(self._species_panel_transaction_owner())
            species_panel.speciesEdited.connect(self._on_species_slider_edited)
            species_panel.speciesDragStarted.connect(self._on_species_slider_drag_started)
            species_panel.speciesDragFinished.connect(self._on_species_slider_drag_finished)
        except RuntimeError as exc:
            logger.debug("Failed to wire species sliders signals: %s", exc, exc_info=True)
            self._species_panel_available = False
            return None

        return species_panel

    def _init_right_dock_and_datasets(self) -> None:
        right_dock_components = build_right_dock_and_dataset_owners(
            self,
            plot_tabs=self._plot_tabs,
            mechanism_getter=self._get_mechanism_text,
            simulation_runner=self._run_dataset_simulation,
            wegscheider_cyclicity_enabled_getter=(
                self._simulation_runtime_settings_owner.wegscheider_cyclicity_enabled
            ),
        )
        self._right_dock = right_dock_components.dock
        self._right_panel = right_dock_components.panel
        _overlay_accessor = getattr(self._plot_tabs._main_plot, "overlay_panel", None)
        if callable(_overlay_accessor):
            self._right_panel.add_tab(_overlay_accessor(), "Overlays")
        self._right_dock.setWidget(right_dock_components.container)
        self.addDockWidget(self._default_dock_area(self._right_dock), self._right_dock)

        self._dataset_registry = right_dock_components.dataset_registry
        self._dataset_fit_settings_store = right_dock_components.dataset_fit_settings_store
        self._dataset_view_publisher = right_dock_components.dataset_view_publisher
        self._mechanism_parameter_scan_owner = right_dock_components.mechanism_parameter_scan_owner
        self._dataset_import_panel = right_dock_components.import_panel

        from kindred.gui.controllers.dataset_import_coordinator import DatasetImportCoordinator

        self._dataset_import_coordinator = DatasetImportCoordinator(
            registry=self._dataset_registry,
            fit_settings_store=self._dataset_fit_settings_store,
            view_publisher=self._dataset_view_publisher,
            import_panel=self._dataset_import_panel,
            status_setter=lambda text: self._status_label.setText(str(text)),
            overlay_sync=self._sync_overlay_catalog,
            color_sync=self._sync_color_manager_authoritative_roster,
            batch_store_getter=lambda: getattr(self, "_batch_store", None),
            batch_model_getter=lambda: getattr(self, "_batch_model", None),
            batch_table_getter=lambda: getattr(self, "_batch_table", None),
            mechanism_text_getter=self._get_mechanism_text,
            extract_mechanism_initials=self._extract_mechanism_initials,
            sync_batch_species_columns=self.sync_batch_species_columns,
            record_failure=self._record_best_effort_failure,
            runtime_inputs_changed=lambda: self._fitting_runtime_input_publisher.publish_if_changed(
                reason="dataset import runtime inputs changed"
            ),
            datasets_removed=self._fitting_runtime_input_publisher.publish_datasets_removed,
            message_parent=self,
        )

        # Reference species statistics table within plot tabs.
        self._results_table = self._plot_tabs._main_plot.stats_table()

    def _init_symbolic_calculator_dock(self) -> None:
        from kindred.gui.symbolic_calculator_owner import SymbolicCalculatorOwner
        from kindred.gui.widgets.symbolic_calculator_panel import SymbolicCalculatorPanel

        owner = SymbolicCalculatorOwner(
            mechanism_session_owner=self._mechanism_session_owner,
        )
        panel = SymbolicCalculatorPanel(
            on_evaluate=owner.evaluate,
            on_copy_result=owner.copy_compact_text,
            on_copy_context=owner.copy_context_text,
            parent=self,
        )
        self._symbolic_calculator_owner = owner
        components = build_symbolic_calculator_dock(self, panel=panel)
        self._symbolic_calculator_dock = components.dock
        self._symbolic_calculator_panel = components.panel
        self._symbolic_calculator_dock.setWidget(components.container)
        self.addDockWidget(self._default_dock_area(self._symbolic_calculator_dock), self._symbolic_calculator_dock)
        self._refresh_symbolic_calculator_state()

    def _refresh_symbolic_calculator_state(self) -> None:
        panel = getattr(self, "_symbolic_calculator_panel", None)
        action = getattr(self, "_symbolic_calculator_action", None)
        editor = getattr(self, "_mechanism_editor", None)
        button = getattr(editor, "symbolic_calculator_btn", None)
        owner = getattr(self, "_symbolic_calculator_owner", None)
        if owner is None:
            available = False
            reason = "Canonical mechanism unavailable."
            state = None
        else:
            state = owner.refresh()
            available = bool(state.available)
            reason = str(state.reason or "")
        if panel is not None:
            if state is not None:
                panel.render_state(state)
            else:
                panel.set_available(available, reason)
        if action is not None:
            action.setEnabled(bool(available))
            action.setToolTip(
                "Open Symbolic Calculator panel"
                if bool(available)
                else str(reason or "Symbolic calculator unavailable.")
            )
        if button is not None:
            button.setEnabled(bool(available))
            button.setToolTip(
                "Open Symbolic Calculator panel"
                if bool(available)
                else str(reason or "Symbolic calculator unavailable.")
            )

    def _open_symbolic_calculator_panel(self) -> None:
        self._refresh_symbolic_calculator_state()
        dock = getattr(self, "_symbolic_calculator_dock", None)
        if dock is None or not dock.isEnabled():
            return
        dock.show()
        dock.raise_()
        if dock.isFloating():
            dock.activateWindow()

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
        self._simulation_run_ui_owner.bind_widgets(
            run_button=self._run_btn,
            stop_button=self._stop_btn,
            progress=self._sim_progress,
            status_label=self._status_label,
            algebra_status_label=self._algebra_status_label,
            mechanism_editor=self._mechanism_editor,
        )

        # Temperature mode indicator in status bar.
        self._temperature_mode_indicator = QtWidgets.QLabel("Temperature: 298.15 K (isothermal)")
        self._temperature_mode_indicator.setStyleSheet("QLabel { font-style: italic; }")
        self._status_bar.addPermanentWidget(self._temperature_mode_indicator)

    def _finish_window_composition(self) -> None:
        """Complete window composition before any persisted/bootstrap state is applied."""
        self._create_menus()
        self._init_mixin_ports()
        self._connect_signals()

    def _bootstrap_window_state(self) -> None:
        """Apply launch overrides and persisted startup state after composition prerequisites exist."""
        self._suppress_preference_updates = True
        try:
            self._load_settings()
        finally:
            self._suppress_preference_updates = False
        self._update_temperature_mode_indicator()
        refresh_runtime_readiness = getattr(
            getattr(self, "_sim_controller", None),
            "refresh_interactive_runtime_readiness",
            None,
        )
        self._update_batch_row_controls_state()
        if callable(refresh_runtime_readiness):
            refresh_runtime_readiness()
            self._last_interactive_runtime_readiness_signature = self._current_run_target_readiness_signature()

    def _simulation_runtime_settings_snapshot(self) -> Dict[str, object]:
        solver_contract = load_solver_contract()
        solver_label = str(self._simulation_runtime_settings_owner.current_solver_name() or solver_contract.default_solver_name).strip()
        try:
            simulation_time_seconds = float(self._parse_sim_time_seconds())
        except Exception:
            simulation_time_seconds = str(self._sim_time_spinbox.text() or "").strip()
        return {
            "solver": str(solver_label or solver_contract.default_solver_name),
            "rtol": float(self._simulation_runtime_settings_owner.current_rtol() or 1e-6),
            "atol": float(self._simulation_runtime_settings_owner.current_atol() or 1e-12),
            "temperature_K": float(self._temperature_spinbox.value()),
            "simulation_time": simulation_time_seconds,
            "num_points": int(self._num_points_spinbox.value()),
            "use_sparse_jacobian": bool(self._simulation_runtime_settings_owner.use_sparse_jacobian()),
            "wegscheider_cyclicity_enabled": bool(
                self._simulation_runtime_settings_owner.wegscheider_cyclicity_enabled()
            ),
            "max_parallel_batch_workers": int(self._sim_controller.parallel_batch.max_parallel_workers),
            "batch_runtime_lane_budget": int(self._sim_controller.batch_runtime_lane_budget),
            "limit_blas_threads_per_worker": bool(
                self._sim_controller.parallel_batch.limit_blas_threads_per_worker
            ),
            "slider_preview_solver": str(self._mechanism_editor.slider_solver_value() or "BDF"),
            "slider_preview_points": int(self._mechanism_editor.slider_points_value()),
        }

    def _simulation_runtime_input_refresh_deferred(self) -> bool:
        return int(getattr(self, "_simulation_runtime_input_refresh_defer_depth", 0) or 0) > 0

    def _simulation_batch_runtime_pool_settings_snapshot(self) -> Dict[str, object]:
        return {
            "max_parallel_batch_workers": int(self._sim_controller.parallel_batch.max_parallel_workers),
            "batch_runtime_lane_budget": int(self._sim_controller.batch_runtime_lane_budget),
            "limit_blas_threads_per_worker": bool(
                self._sim_controller.parallel_batch.limit_blas_threads_per_worker
            ),
        }

    def _refresh_simulation_runtime_after_inputs_changed(
        self,
        *,
        batch_runtime_pool_inputs_changed: bool = True,
    ) -> None:
        if self._simulation_runtime_input_refresh_deferred():
            return
        self._invalidate_result_truth_for_simulation_runtime_inputs_changed()
        if bool(batch_runtime_pool_inputs_changed):
            self._sim_controller.simulation_runtime_inputs_changed()
        else:
            self._sim_controller.simulation_runtime_inputs_changed(
                batch_runtime_pool_inputs_changed=False
            )

    def _invalidate_result_truth_for_simulation_runtime_inputs_changed(self) -> None:
        required_attrs = (
            "_sim_controller",
            "results_controller",
            "_simulation_batch_owner",
            "_mechanism_helpers",
        )
        if not all(hasattr(self, attr) for attr in required_attrs):
            return
        try:
            self._sim_controller.invalidate_active_explicit_simulation_for_authoritative_change()
        except Exception as exc:
            self._record_best_effort_failure(
                "main_window.runtime_inputs.invalidate_active_run",
                message="Failed to invalidate active run after simulation runtime inputs changed",
                exc=exc,
            )
        try:
            self._apply_authoritative_result_truth_effects(
                cache_stale_scope_is_global=True,
                display_clear_scope_is_global=True,
                clear_cached_mechanism=True,
            )
        except Exception as exc:
            self._record_best_effort_failure(
                "main_window.runtime_inputs.invalidate_result_truth",
                message="Failed to invalidate result truth after simulation runtime inputs changed",
                exc=exc,
            )

    def _apply_simulation_runtime_readiness_render_state(
        self,
        state: object,
    ) -> None:
        self._simulation_run_ui_owner.render_runtime_readiness(state)
        self._render_preview_available(bool(getattr(state, "preview_available", False)))
        preview_unavailable_status = str(getattr(state, "preview_unavailable_status", "") or "").strip()
        if preview_unavailable_status:
            self._preview_session.show_preview_unavailable_for_dirty_state(preview_unavailable_status)

    def _render_launch_available(self, ready: bool) -> None:
        self._simulation_run_ui_owner.render_launch_available(bool(ready))

    def _render_preview_available(self, ready: bool) -> None:
        ready_value = bool(ready)
        self._simulation_runtime_preview_ready = ready_value

    def _init_mixin_ports(self) -> None:
        self._fitting_ports = FittingMixinPorts(
            mechanism_editor=self._mechanism_editor,
            dataset_registry=self._dataset_registry,
            dataset_fit_settings_store=self._dataset_fit_settings_store,
            dataset_view_publisher=self._dataset_view_publisher,
            mechanism_parameter_scan_owner=self._mechanism_parameter_scan_owner,
            status_setter=lambda text: self._status_label.setText(str(text)),
            temperature_getter=lambda: float(self._temperature_spinbox.value()),
            num_points_getter=lambda: int(self._num_points_spinbox.value()),
            wegscheider_cyclicity_enabled_getter=(
                self._simulation_runtime_settings_owner.wegscheider_cyclicity_enabled
            ),
        )

    def _refresh_after_authoritative_source_load(self) -> None:
        self._sync_mechanism_session_owner_after_authoritative_widget_write(dispatch_consumers=False)
        self._preview_session.clear_working_transaction(clear_committed_slider_values=True)
        try:
            self._mechanism_editor._variable_sliders.clear()
        except Exception:
            logger.debug("Failed to clear variable sliders after authoritative source load", exc_info=True)
            self._preview_session.clear_pending_slider_values()

        self._update_batch_row_controls_state()
        self._refresh_slider_transaction_button_state()

        try:
            self._variable_runtime.extract_and_populate_variables(
                preserve_visibility=True,
            )
        except Exception:
            logger.debug(
                "Failed to refresh variable sliders after authoritative source load",
                exc_info=True,
            )

        try:
            self._variable_runtime.update_parameter_table_from_sliders()
        except Exception:
            logger.debug("Failed to update parameter table after authoritative source load", exc_info=True)
            QtCore.QTimer.singleShot(
                0,
                self._variable_runtime.update_parameter_table_from_sliders,
            )

    def _source_load_plan_materializes_initial_conditions(self, reconciliation_plan) -> bool:
        if reconciliation_plan is None:
            return False
        seed_sets = getattr(reconciliation_plan, "seed_sets", {}) or {}
        if seed_sets:
            return True
        plan_species = [str(species) for species in (getattr(reconciliation_plan, "species_names", ()) or ()) if str(species)]
        try:
            current_species = [str(species) for species in (self._batch_store.visible_species() or ()) if str(species)]
        except Exception:
            current_species = []
        return bool(plan_species != current_species)

    def _prepare_authoritative_source_load_request(
        self,
        *,
        source,
        action_label: str,
        guard_action_text: str,
        prompt_overwrite: bool,
        transition_source: str,
        schedule_runtime_refresh: bool,
        canceled_status_text: str | None = None,
        continue_label: str = "Load",
    ) -> tuple[object, object, ReactionsDraftApplyResult | None] | None:
        return self._mechanism_source_transaction_owner.prepare_source_load_request(
            source=source,
            action_label=str(action_label),
            guard_action_text=str(guard_action_text),
            prompt_overwrite=bool(prompt_overwrite),
            transition_source=str(transition_source or "authoritative_source_load"),
            schedule_runtime_refresh=bool(schedule_runtime_refresh),
            canceled_status_text=canceled_status_text,
            continue_label=str(continue_label or "Load"),
        )

    def _execute_authoritative_source_load(
        self,
        *,
        source,
        reconciliation_plan,
        draft_apply_result: ReactionsDraftApplyResult | None,
        description: str,
        transition_source: str,
        schedule_runtime_refresh: bool,
        record_undo: bool = True,
        canonical_materialized: bool | None = None,
        full_snapshot_required: bool = False,
        before_materialize: Callable[[], None] | None = None,
        before_refresh: Callable[[], None] | None = None,
        rollback_before_abort_publish: Callable[[], None] | None = None,
    ) -> None:
        self._mechanism_source_transaction_owner.execute_source_load(
            source=source,
            reconciliation_plan=reconciliation_plan,
            draft_apply_result=draft_apply_result,
            description=str(description),
            transition_source=str(transition_source or "authoritative_source_load"),
            schedule_runtime_refresh=bool(schedule_runtime_refresh),
            record_undo=bool(record_undo),
            canonical_materialized=canonical_materialized,
            full_snapshot_required=bool(full_snapshot_required),
            before_materialize=before_materialize,
            before_refresh=before_refresh,
            rollback_before_abort_publish=rollback_before_abort_publish,
        )

    def _prepare_authoritative_source_load_reconciliation(
        self,
        source,
        *,
        prompt_overwrite: bool,
        destination_store=None,
    ):
        reconciliation_plan = self._initial_conditions_source_acceptance_owner.prepare_for_authoritative_apply(
            source,
            prompt_overwrite=bool(prompt_overwrite),
            destination_store=destination_store,
        )
        if reconciliation_plan is None:
            return None, None
        return reconciliation_plan.source, reconciliation_plan

    def _apply_authoritative_source_load_reconciliation_after_source_acceptance(
        self,
        reconciliation_plan,
    ):
        if reconciliation_plan is None:
            return None
        return self._initial_conditions_source_acceptance_owner.apply_prepared_plan(
            reconciliation_plan,
            notify_canonical_rows_changed=False,
        )

    def _current_mechanism_species_roster_for_colors(self) -> tuple[str, ...] | None:
        """Best-effort current mechanism species roster for color canonicalization."""
        from kindred.core.batch_initial_conditions import (
            strip_named_reaction_dsl_initial_concentration_sets,
        )
        from kindred.core.mechanism_source import MechanismAuthoringSource

        reactions_text = str(self.mechanism_reactions_text_raw() or "")
        state_network_text = str(self.mechanism_state_network_dsl_raw() or "")
        if not reactions_text.strip() and not state_network_text.strip():
            return ()
        try:
            reactions_text = strip_named_reaction_dsl_initial_concentration_sets(reactions_text)
        except Exception:
            logger.debug("Failed to parse current mechanism species roster for color sync", exc_info=True)
            return None
        source = MechanismAuthoringSource.from_parts(
            reactions_text=reactions_text,
            state_network_dsl=state_network_text,
        )
        mechanism_text = source.full_dsl

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
            from kindred.core.simulator.parameter_algebra import apply_parameter_algebra_to_mechanism
            from kindred.core.units import UnitsModel

            temperature_k = float(self._temperature_spinbox.value())
            units = UnitsModel(temperature_K=temperature_k)
            wegscheider_enabled = bool(self._simulation_runtime_settings_owner.wegscheider_cyclicity_enabled())

            def _build_structure_snapshot(full_dsl: str) -> object:
                mechanism_obj = parse_dsl_to_mechanism(full_dsl, initials={}, units=units)
                if isinstance(getattr(mechanism_obj, "metadata", None), dict):
                    mechanism_obj.metadata["wegscheider_cyclicity_enabled"] = wegscheider_enabled
                _ = apply_parameter_algebra_to_mechanism(
                    full_dsl,
                    mechanism=mechanism_obj,
                    require_mutable=False,
                )
                return mechanism_obj

            mechanism = self._mechanism_helpers.authoritative_structure_snapshot(
                source=source,
                units_identity=(
                    "temperature_K",
                    f"{temperature_k:.17g}",
                    "wegscheider",
                    str(wegscheider_enabled),
                ),
                builder=_build_structure_snapshot,
            ).mechanism
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
        _accessor = getattr(plot, "overlay_panel", None)
        overlay_panel = _accessor() if callable(_accessor) else None
        refresh_swatches = getattr(overlay_panel, "refresh_color_swatches", None)
        if callable(refresh_swatches):
            refresh_swatches()

    def _sync_overlay_catalog(self):
        """Propagate loaded datasets into the simulation overlay controls."""
        plot = getattr(self._plot_tabs, "_main_plot", None)
        dataset_registry = getattr(self, "_dataset_registry", None)
        if plot is None or dataset_registry is None:
            return
        if not hasattr(plot, "set_overlay_catalog"):
            return
        self._sync_color_manager_authoritative_roster()
        plot.set_overlay_catalog(dataset_registry.presentation_payloads_by_display_name())

    def _snapshot_datasets(self) -> Dict[str, Dict[str, Any]]:
        """Capture lightweight metadata about loaded datasets."""
        dataset_registry = getattr(self, "_dataset_registry", None)
        if dataset_registry is None:
            return {}

        snapshot: Dict[str, Dict[str, Any]] = {}
        for name, dataset in dataset_registry.presentation_payloads_by_display_name().items():
            observations = observations_from_payload(dataset)
            t_values, _species_data = dense_view_from_observations(observations)
            num_points = int(t_values.size)
            snapshot[name] = {
                "num_points": num_points,
                "species": list(observations.keys()),
            }
        return snapshot

    def _load_settings(self):
        """Load user preferences from QSettings."""
        self.config_controller.load_settings()
        self._init_fitting_defaults()

    def _save_settings(self):
        """Save user preferences to QSettings."""
        self.config_controller.save_settings()

    def _create_menus(self):
        """Create the menu bar and shortcut customization registry."""
        self._shortcut_actions = {}
        menubar = self.menuBar()
        self._menu_bar = menubar

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
                ("New Project", self.project_controller.new_project, QtGui.QKeySequence.New, "newProjectAction", "Start a new empty project"),
                ("Load Project...", self.project_controller.load_project, QtGui.QKeySequence.Open, "loadProjectAction", "Load a Kindred project file (.kin)"),
                ("Save Project", self.project_controller.save_project, QtGui.QKeySequence.Save, "saveProjectAction", "Save current project to its file"),
                ("Save Project As...", self.project_controller.save_project_as, QtGui.QKeySequence.SaveAs, "saveProjectAsAction", "Save current project to a new file"),
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
                ("State &Network Editor...", self._open_state_network, None, "stateNetworkAction", "Edit state transition networks for TST calculations"),
                None,
                ("Customize &Keyboard Shortcuts...", self._open_shortcuts_dialog, "Ctrl+K", "customizeShortcutsAction", "Customize keyboard shortcuts for actions"),
            ],
        )
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

        examples_menu = QtWidgets.QMenu("E&xamples", self)
        menubar.addMenu(examples_menu)
        self._examples_menu = examples_menu
        presets_submenu = QtWidgets.QMenu("Preset Mechanisms", examples_menu)
        examples_menu.addMenu(presets_submenu)
        self._preset_examples_menu = presets_submenu
        for preset_id in self._available_preset_ids():
            presets_submenu.addAction(preset_id, lambda pid=preset_id: self._load_preset_mechanism(pid))
        intervention_examples_submenu = QtWidgets.QMenu("Intervention Examples", examples_menu)
        examples_menu.addMenu(intervention_examples_submenu)
        self._intervention_examples_menu = intervention_examples_submenu
        for spec in self._available_intervention_example_specs():
            example_id = str(spec["id"])
            title = str(spec.get("title", "")).strip()
            action_text = f"{example_id}: {title}" if title and title != example_id else example_id
            action = intervention_examples_submenu.addAction(
                action_text,
                lambda _checked=False, eid=example_id: self._load_intervention_example(eid),
            )
            action.setData(example_id)
            action.setObjectName(f"loadInterventionExample{example_id}Action")
        examples_menu.addSeparator()
        add_items(
            examples_menu,
            [("&Template Manager...", self._open_template_manager, "Ctrl+T", "templateManagerAction", "Browse and manage custom mechanism templates (Ctrl+T)")],
        )

        sim_menu = menubar.addMenu("&Simulation")
        add_items(
            sim_menu,
            [
                ("&Run", self._run_simulation_after_committing_batch_initials, None, "runSimulationAction", "Run kinetic simulation with current mechanism (Ctrl+R or F5)", {"shortcuts": ["Ctrl+R", "F5"], "store_as": "_run_simulation_action"}),
                ("&Stop", self._sim_controller.stop_simulation, "Esc", "stopSimulationAction", "Stop running simulation (Esc)"),
                ("Symbolic Calculator...", self._open_symbolic_calculator_panel, None, "openSymbolicCalculatorAction", "Open Symbolic Calculator panel", {"store_as": "_symbolic_calculator_action"}),
                None,
                ("Simulation &Settings...", self._open_solver_settings, None, "simulationSettingsAction", "Configure solver tolerances and advanced simulation settings"),
            ],
        )
        self._refresh_symbolic_calculator_state()

        fit_menu = menubar.addMenu("&Fitting")
        add_items(
            fit_menu,
            [
                ("Fitting &Defaults...", self._configure_fitting, None, "fittingDefaultsAction", "Set default algorithm and integration settings for new fitting windows"),
                ("Global Fit...", self._run_global_fit, "Ctrl+Shift+F", "globalFitAction", "Fit shared parameters across all loaded datasets (Ctrl+Shift+F)"),
            ],
        )

        help_menu = menubar.addMenu("&Help")
        self._help_menu = help_menu
        add_items(
            help_menu,
            [
                ("&Documentation", self._open_docs, QtGui.QKeySequence.HelpContents, "documentationAction", "Open Kindred documentation (shows offline guidance if online docs are unavailable)"),
                ("&Interactive Tutorials...", self._show_tutorials, None, "tutorialsAction", "Launch step-by-step interactive tutorials"),
                None,
            ],
        )
        self._build_action(
            menu=help_menu,
            text="Panel Layout Tips...",
            callback=self._show_panel_layout_tips,
            shortcut=None,
            object_name="panelLayoutTipsAction",
            tooltip="Explain how to arrange multiple panels together on the same side of the workspace",
            store_as="_panel_layout_tips_action",
        )
        add_items(
            help_menu,
            [
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
        if object_name == "runSimulationAction":
            self._simulation_run_ui_owner.bind_run_action(action)
        return action

    def _connect_signals(self):
        """Connect signals between components."""
        # Dataset import panel signals
        self._dataset_import_panel.importCompleted.connect(
            self._dataset_import_coordinator.handle_completion
        )
        self._dataset_import_panel.datasetRemovalRequested.connect(
            self._dataset_import_coordinator.remove_dataset_by_id
        )

        # Temperature mode indicator and authoritative mechanism consumers.
        reactions_widget = getattr(self._mechanism_editor, "_reactions_text", None)
        state_editor = getattr(self._mechanism_editor, "_state_network_editor", None)
        if reactions_widget is not None:
            reactions_widget.textChanged.connect(self._on_reactions_text_changed_for_main_window)
        if state_editor is not None:
            state_editor.stateNetworkChanged.connect(self._on_state_network_changed_for_main_window)
        self._temperature_spinbox.valueChanged.connect(self._on_temperature_spinbox_value_changed_for_main_window)
        # User preference tracking for spinbox-only dual-persisted keys.
        self._temperature_spinbox.valueChanged.connect(self._on_temperature_user_edit)
        self._num_points_spinbox.valueChanged.connect(self._on_num_points_user_edit)
        self._sim_time_spinbox.textChanged.connect(self._on_sim_time_user_edit)

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

    def _on_authoritative_mechanism_input_changed(self) -> None:
        """Apply the ordered lifecycle for an authoritative mechanism transition."""
        self._apply_authoritative_mechanism_transition()

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

    def _apply_runtime_settings(
        self,
        *,
        solver: object = RUNTIME_SETTING_UNSET,
        rtol: object = RUNTIME_SETTING_UNSET,
        atol: object = RUNTIME_SETTING_UNSET,
        use_sparse_jacobian: object = RUNTIME_SETTING_UNSET,
        wegscheider_cyclicity_enabled: object = RUNTIME_SETTING_UNSET,
    ) -> None:
        solver_contract = load_solver_contract()
        self._simulation_runtime_settings_owner.apply_runtime_settings(
            solver=solver,
            rtol=rtol,
            atol=atol,
            use_sparse_jacobian=use_sparse_jacobian,
            wegscheider_cyclicity_enabled=wegscheider_cyclicity_enabled,
        )

        combo = getattr(self, "_solver_method_combo", None)
        if combo is not None:
            solver_label = str(self._simulation_runtime_settings_owner.current_solver_name() or solver_contract.default_solver_name).strip() or solver_contract.default_solver_name
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
        self._apply_runtime_settings(solver=str(v))
        if not self._suppress_preference_updates:
            self.config_controller.update_user_preference("solver", self._simulation_runtime_settings_owner.current_solver_name())
        self._refresh_simulation_runtime_after_inputs_changed(
            batch_runtime_pool_inputs_changed=False
        )

    def _update_solver_summary_label(self) -> None:
        """Refresh the solver summary label shown in the Solver section."""
        if not hasattr(self, "_solver_summary_label"):
            return

        solver_contract = load_solver_contract()
        solver_label = str(self._simulation_runtime_settings_owner.current_solver_name() or solver_contract.default_solver_name).strip() or solver_contract.default_solver_name
        solver_method = solver_contract.normalize_solver_name(solver_label)
        solver_display = str(solver_method)
        if str(solver_method) != str(solver_label):
            solver_display = f"{solver_label} → {solver_method}"
        rtol = self._simulation_runtime_settings_owner.current_rtol() or 1e-6
        atol = self._simulation_runtime_settings_owner.current_atol() or 1e-12
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
        if self._simulation_runtime_settings_owner.use_sparse_jacobian() and str(solver_method).upper() in {"RADAU", "BDF"}:
            summary += " • Symbolic J"
        self._solver_summary_label.setText(summary)

    def _on_temperature_user_edit(self, value: float) -> None:
        if not self._suppress_preference_updates:
            self.config_controller.update_user_preference("temperature_K", value)

    def _on_num_points_user_edit(self, value: int) -> None:
        if not self._suppress_preference_updates:
            self.config_controller.update_user_preference("num_points", value)
        self._refresh_simulation_runtime_after_inputs_changed(
            batch_runtime_pool_inputs_changed=False
        )

    def _on_sim_time_user_edit(self, text: str) -> None:
        try:
            sim_time = self._parse_sim_time_seconds()
        except Exception:
            return
        if not self._suppress_preference_updates:
            self.config_controller.update_user_preference("simulation_time", f"{float(sim_time):g}")
        self._refresh_simulation_runtime_after_inputs_changed(
            batch_runtime_pool_inputs_changed=False
        )

    def _compute_temperature_indicator_state(
        self,
        reactions_text: str,
        state_network_text: str,
    ) -> TemperatureIndicatorState:
        mechanism_text = str(reactions_text)
        energy_mode_active = self._temperature_indicator_energy_mode_active(
            reactions_text=mechanism_text,
            state_network_text=str(state_network_text or ""),
        )

        t_override_k = self._variable_runtime.dsl_global_temperature_k(mechanism_text)

        temp_schedule = None
        schedule_defined = False
        try:
            from kindred.core.temperature_dsl import parse_temperature_schedule

            temp_schedule = parse_temperature_schedule(mechanism_text)
            schedule_defined = temp_schedule is not None
        except Exception:
            temp_schedule = None
            schedule_defined = False

        schedule_kind: Optional[str] = None
        schedule_interval_count: Optional[int] = None
        constant_schedule_k: Optional[float] = None
        indicator_text: Optional[str] = None

        if t_override_k is not None and not schedule_defined:
            indicator_text = f"Temperature: {t_override_k:.2f} K (from DSL)"
        elif temp_schedule is not None:
            if temp_schedule.schedule_type == "constant":
                schedule_kind = "constant"
                constant_schedule_k = float(temp_schedule(0.0))
                indicator_text = f"Temperature: {constant_schedule_k:.2f} K (constant from DSL)"
            elif temp_schedule.schedule_type == "piecewise":
                schedule_kind = "piecewise"
                schedule_interval_count = len(temp_schedule.get_intervals())
                indicator_text = (
                    f"Temperature: Schedule ({schedule_interval_count} interval"
                    f"{'s' if schedule_interval_count != 1 else ''})"
                )
            elif temp_schedule.schedule_type == "response":
                schedule_kind = "response"
                schedule_interval_count = len(temp_schedule.get_intervals())
                indicator_text = (
                    f"Temperature: Schedule (response, {schedule_interval_count} interval"
                    f"{'s' if schedule_interval_count != 1 else ''})"
                )
            else:
                indicator_text = "Temperature: Schedule (unknown type)"

        return TemperatureIndicatorState(
            energy_mode_active=energy_mode_active,
            t_override_k=t_override_k,
            schedule_defined=schedule_defined,
            schedule_kind=schedule_kind,
            schedule_interval_count=schedule_interval_count,
            constant_schedule_k=constant_schedule_k,
            indicator_text=indicator_text,
        )

    def _temperature_indicator_energy_mode_active(
        self,
        *,
        reactions_text: str,
        state_network_text: str,
    ) -> bool:
        try:
            from kindred.core.mechanism_source import MechanismAuthoringSource
            from kindred.core.simulator.dsl import parse_dsl_to_mechanism

            source = MechanismAuthoringSource.from_parts(
                reactions_text=str(reactions_text or ""),
                state_network_dsl=str(state_network_text or ""),
            )
            mechanism = parse_dsl_to_mechanism(source.full_dsl, initials={})
        except Exception:
            return False
        return bool(self._variable_runtime.is_energy_mode_mechanism(mechanism))

    def _update_temperature_mode_indicator(self) -> None:
        """
        Update temperature mode indicator in status bar.

        Shows one of:
        - "Temperature: XXX K (from DSL)" - when T= directive found in reactions
        - "Temperature: XXX K (isothermal)" - when no schedule and no T= in DSL
        - "Temperature: Schedule (N intervals)" - when piecewise schedule detected
        - "Temperature: XXX K (constant from DSL)" - when temp_const detected
        - "Temperature: Schedule (response, N intervals)" - when temp_response detected

        Priority rule:
        1. Temperature schedule in DSL takes highest priority for indicator and solver.
        2. T= directive seeds the spinbox value but does not override a schedule.
        3. Otherwise, the temperature spinbox value is used (isothermal).
        """
        if not hasattr(self, "_temperature_mode_indicator"):
            return

        mechanism_text = self.mechanism_reactions_text_raw()
        state_network_text = self.mechanism_state_network_dsl_raw()
        state = self._compute_temperature_indicator_state(
            reactions_text=mechanism_text,
            state_network_text=state_network_text,
        )

        was_override_active = getattr(self, "_temperature_dsl_override_active", False)

        if state.t_override_k is not None:
            if not was_override_active:
                # Capture pre-override spinbox value before DSL write-back.
                self._pre_dsl_temperature = self._temperature_spinbox.value()
            self._temperature_dsl_override_active = True
            # Sync spinbox to DSL-derived temperature without firing preference persistence
            self._temperature_spinbox.blockSignals(True)
            try:
                self._temperature_spinbox.setValue(state.t_override_k)
            finally:
                self._temperature_spinbox.blockSignals(False)

            if not state.schedule_defined:
                # Bare T= without schedule: spinbox visible, disabled, "from DSL"
                indicator_text = str(state.indicator_text)
                self._temperature_mode_indicator.setText(indicator_text)
                self._set_temperature_override_state(
                    enabled=False,
                    tooltip="Overridden by DSL (T=...).",
                )
                self._set_temperature_spinbox_visible(True)
                logger.debug(f"Temperature mode indicator updated: {indicator_text}")
                return
            # Temperature schedule takes precedence over bare T= directive —
            # the T= value seeds the spinbox but the schedule dictates the indicator.

        if state.t_override_k is None:
            if was_override_active:
                restore = self._pre_dsl_temperature
                if restore is None:
                    restore = float(
                        self.config_controller.get_user_preference("temperature_K") or 298.15
                    )
                self._temperature_spinbox.blockSignals(True)
                try:
                    self._temperature_spinbox.setValue(float(restore))
                finally:
                    self._temperature_spinbox.blockSignals(False)
                self._pre_dsl_temperature = None
            self._temperature_dsl_override_active = False

        # Hide spinbox and restore editable state
        self._set_temperature_spinbox_visible(False)
        if state.energy_mode_active:
            self._set_temperature_override_state(
                enabled=True,
                tooltip="Temperature for thermodynamic calculations (energy mode: add T=... to override).",
            )
        else:
            self._set_temperature_override_state(
                enabled=True,
                tooltip="Temperature for thermodynamic calculations",
            )

        indicator_text = state.indicator_text
        if indicator_text is None:
            T = self._temperature_spinbox.value()
            if state.energy_mode_active:
                indicator_text = f"Temperature: {T:.2f} K (energy mode: set T=... in DSL)"
            else:
                indicator_text = f"Temperature: {T:.2f} K (isothermal)"

        self._temperature_mode_indicator.setText(indicator_text)
        if state.energy_mode_active and state.schedule_defined:
            self._set_temperature_override_state(
                enabled=False,
                tooltip="Overridden by DSL temperature schedule.",
            )
        logger.debug(f"Temperature mode indicator updated: {indicator_text}")

    def _load_data_via_action(self) -> None:
        """Invoke the Data panel's Load dialog from the File menu."""
        self._dataset_import_panel.trigger_load_dialog()

    def _scan_mechanism_parameters(self):
        """
        Extract parameters from current mechanism DSL.

        P2 ENHANCEMENT: Now uses caching to avoid re-scanning unchanged mechanisms.
        This improves performance, especially for large mechanisms.
        """
        from kindred.gui.controllers.dataset_errors import DatasetOwnerError

        mechanism_text = self._get_mechanism_text()

        if not mechanism_text.strip():
            QtWidgets.QMessageBox.warning(
                self,
                "No Mechanism",
                "Please define a mechanism before scanning for parameters."
            )
            return

        try:
            self._mechanism_parameter_scan_owner.scan_mechanism_parameters(mechanism_text)
        except DatasetOwnerError as exc:
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
            from kindred.io.resources import get_preset_mechanism_source

            source = get_preset_mechanism_source(preset_id)
            request = self._prepare_authoritative_source_load_request(
                source=source,
                action_label=f"Loading preset {preset_id}",
                guard_action_text=f"Loading preset {preset_id}",
                prompt_overwrite=True,
                transition_source="preset_load",
                schedule_runtime_refresh=True,
                canceled_status_text="Canceled preset load",
            )
            if request is None:
                return
            source, reconciliation_plan, draft_apply_result = request
            self._execute_authoritative_source_load(
                source=source,
                reconciliation_plan=reconciliation_plan,
                draft_apply_result=draft_apply_result,
                description=f"Load preset {preset_id}",
                transition_source="preset_load",
                schedule_runtime_refresh=True,
            )
            self._status_label.setText(f"Loaded preset mechanism: {preset_id}")
            logger.info(f"Loaded preset mechanism: {preset_id}")

        except Exception as e:
            logger.error(f"Failed to load preset {preset_id}: {e}", exc_info=True)
            QtWidgets.QMessageBox.critical(
                self,
                "Load Error",
                f"Failed to load preset mechanism {preset_id}:\n\n{e}"
            )

    def _load_intervention_example(self, example_id: str):
        """Load a bundled intervention example into the mechanism editor."""
        try:
            from kindred.io.resources import get_intervention_example_source

            source = get_intervention_example_source(example_id)
            request = self._prepare_authoritative_source_load_request(
                source=source,
                action_label=f"Loading intervention example {example_id}",
                guard_action_text=f"Loading intervention example {example_id}",
                prompt_overwrite=True,
                transition_source="intervention_example_load",
                schedule_runtime_refresh=True,
                canceled_status_text="Canceled intervention example load",
            )
            if request is None:
                return
            source, reconciliation_plan, draft_apply_result = request
            self._execute_authoritative_source_load(
                source=source,
                reconciliation_plan=reconciliation_plan,
                draft_apply_result=draft_apply_result,
                description=f"Load intervention example {example_id}",
                transition_source="intervention_example_load",
                schedule_runtime_refresh=True,
            )
            self._status_label.setText(f"Loaded intervention example: {example_id}")
            logger.info("Loaded intervention example: %s", example_id)

        except Exception as e:
            logger.error("Failed to load intervention example %s: %s", example_id, e, exc_info=True)
            QtWidgets.QMessageBox.critical(
                self,
                "Load Error",
                f"Failed to load intervention example {example_id}:\n\n{e}",
            )

    def _open_template_manager(self):
        """Open template manager dialog."""
        from kindred.gui.widgets.template_manager_dialog import TemplateManagerDialog

        dialog = TemplateManagerDialog(
            parent=self,
            template_manager=self._template_manager,
            current_mechanism_source=self._current_visible_mechanism_source(),
        )

        # Connect signal to load template
        dialog.templateLoadRequested.connect(self._load_template_from_manager)

        dialog.exec()

    def _open_mechanism_inspector(self) -> None:
        """Open a read-only inspector for the current mechanism text."""
        from kindred.gui.widgets.mechanism_inspector import MechanismInspectorDialog

        text = self._current_mechanism_inspector_source()
        dialog = MechanismInspectorDialog(text, parent=self)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self._mechanism_inspector_dialog = dialog
        dialog.show()

    def _current_mechanism_inspector_source(self) -> str:
        return str(self._current_visible_mechanism_source().full_dsl)

    def _current_visible_mechanism_source(self):
        owner = getattr(self, "_mechanism_session_owner", None)
        if owner is None:
            raise RuntimeError("Mechanism session owner is unavailable.")
        return owner.canonical_source

    def _load_template_from_manager(self, source):
        """Load template from template manager."""
        from kindred.core.mechanism_source import MechanismAuthoringSource

        if not isinstance(source, MechanismAuthoringSource):
            raise TypeError("Template load requires a complete mechanism source.")
        request = self._prepare_authoritative_source_load_request(
            source=source,
            action_label="Loading this template",
            guard_action_text="Loading this template",
            prompt_overwrite=True,
            transition_source="template_load",
            schedule_runtime_refresh=True,
            canceled_status_text="Canceled template load",
        )
        if request is None:
            return
        source, reconciliation_plan, draft_apply_result = request
        self._execute_authoritative_source_load(
            source=source,
            reconciliation_plan=reconciliation_plan,
            draft_apply_result=draft_apply_result,
            description="Load template",
            transition_source="template_load",
            schedule_runtime_refresh=True,
        )

        self._status_label.setText("Loaded template from Template Manager")
        logger.info("Loaded template from Template Manager")

    def _confirm_initial_condition_import_overwrite(self, set_names: Sequence[str]) -> bool:
        names = ", ".join(str(name) for name in set_names if str(name))
        box = QtWidgets.QMessageBox(self)
        box.setIcon(QtWidgets.QMessageBox.Warning)
        box.setWindowTitle("Overwrite Initial Conditions?")
        box.setText(f"Apply mechanism-text Initial Conditions to: {names}")
        box.setInformativeText(
            "Existing values in those set(s) will be overwritten. Other Initial Conditions sets are preserved."
        )
        cancel_button = box.addButton(QtWidgets.QMessageBox.Cancel)
        apply_button = box.addButton("Apply", QtWidgets.QMessageBox.AcceptRole)
        box.setDefaultButton(cancel_button)
        box.exec()
        return box.clickedButton() is apply_button

    def _show_initial_conditions_reconciliation_error(self, message: str) -> None:
        QtWidgets.QMessageBox.warning(
            self,
            "Initial Conditions Import Error",
            str(message or "Initial Conditions could not be reconciled."),
        )

    def _notify_batch_initial_rows_changed(self, rows: Sequence[int]) -> None:
        batch_model = getattr(self, "_batch_model", None)
        batch_store = getattr(self, "_batch_store", None)
        if batch_model is None or batch_store is None:
            return
        store = getattr(batch_model, "store", None)
        if callable(store):
            try:
                if store() is not batch_store:
                    return
            except Exception:
                return
        notify_rows_changed = getattr(batch_model, "notify_rows_changed", None)
        if not callable(notify_rows_changed):
            return
        try:
            notify_rows_changed([int(row) for row in rows])
        except Exception:
            logger.debug("Failed to notify batch model after migrated initial concentration rows changed", exc_info=True)

    def _notify_canonical_batch_initial_rows_changed(
        self,
        rows: Sequence[int],
        transition_source: str,
        discard_dirty_preview: bool,
    ) -> None:
        batch_model = getattr(self, "_batch_model", None)
        if batch_model is None or not hasattr(batch_model, "emit_canonical_initials_changed"):
            return
        affected_set_ids = tuple(
            dict.fromkeys(
                str(self._batch_set_id_for_row(int(row)) or "").strip()
                for row in rows or ()
                if str(self._batch_set_id_for_row(int(row)) or "").strip()
            )
        )
        if not affected_set_ids:
            return
        try:
            batch_model.emit_canonical_initials_changed(
                affected_set_ids=affected_set_ids,
                transition_source=str(transition_source or "initial_conditions_import_reconciliation"),
                discard_dirty_preview=bool(discard_dirty_preview),
            )
        except Exception:
            logger.debug(
                "Failed to emit canonical batch initials transition after migrated rows changed",
                exc_info=True,
            )

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

        info_label = QtWidgets.QLabel(self._state_network_dialog_info_text())
        info_label.setObjectName("stateNetworkDialogInfoLabel")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        base_editor = self._mechanism_editor._state_network_editor
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

        # State Network edits publish immediately through stateNetworkChanged.
        button_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
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

            self._status_label.setText("State network editor closed")

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

    def _undo(self):
        """
        Undo the last action.

        Tries high-level undo stack first (for operations like Load Preset),
        then falls back to text editor undo for character-level edits.
        """
        # Try high-level undo stack first
        if self._undo_stack.canUndo():
            resolution = self._reactions_draft_resolution_owner.resolve_for_stale_command(
                "Undoing an authoritative mechanism change",
                apply_label="Apply Draft; Do Not Undo",
                discard_label="Discard Draft and Undo",
                informative_text=(
                    "Applying the draft changes the mechanism source, so the current Undo command will not run."
                ),
            )
            if not resolution.may_continue:
                self._status_label.setText("Canceled undo")
                return
            if resolution.status == "applied":
                self._status_label.setText("Draft applied; undo command was not run.")
                return
            self._undo_stack.undo()
            self._status_label.setText(f"Undo: {self._undo_stack.undoText()}")
            logger.debug(f"Undo: {self._undo_stack.undoText()}")
            return

        # Fall back to text editor undo
        focused_widget = QtWidgets.QApplication.focusWidget()
        if isinstance(focused_widget, (QtWidgets.QPlainTextEdit, QtWidgets.QTextEdit)):
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
        # Try high-level undo stack first
        if self._undo_stack.canRedo():
            resolution = self._reactions_draft_resolution_owner.resolve_for_stale_command(
                "Redoing an authoritative mechanism change",
                apply_label="Apply Draft; Do Not Redo",
                discard_label="Discard Draft and Redo",
                informative_text=(
                    "Applying the draft changes the mechanism source, so the current Redo command will not run."
                ),
            )
            if not resolution.may_continue:
                self._status_label.setText("Canceled redo")
                return
            if resolution.status == "applied":
                self._status_label.setText("Draft applied; redo command was not run.")
                return
            self._undo_stack.redo()
            self._status_label.setText(f"Redo: {self._undo_stack.redoText()}")
            logger.debug(f"Redo: {self._undo_stack.redoText()}")
            return

        # Fall back to text editor redo
        focused_widget = QtWidgets.QApplication.focusWidget()
        if isinstance(focused_widget, (QtWidgets.QPlainTextEdit, QtWidgets.QTextEdit)):
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
            "Use View > Panels to show or hide the Mechanism, Symbolic Calculator, Interactive Sliders, Initial Conditions, Data, and Analysis panels.\n\n"
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
        symbolic_dock = getattr(self, "_symbolic_calculator_dock", None)
        if not all(dock is not None for dock in (mechanism_dock, sliders_dock, batch_dock, data_dock, analysis_dock, symbolic_dock)):
            return

        # Left column: Mechanism (top), Interactive Sliders (bottom).
        self.splitDockWidget(mechanism_dock, sliders_dock, Qt.Vertical)

        # Right column: Initial Conditions (top), Data and Analysis tabified (bottom).
        self.splitDockWidget(batch_dock, data_dock, Qt.Vertical)
        self.tabifyDockWidget(data_dock, analysis_dock)
        self.tabifyDockWidget(data_dock, symbolic_dock)
        data_dock.raise_()

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
                tb = dock.titleBarWidget()
                if tb is not None and hasattr(tb, "restore"):
                    tb.restore()
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
        """Get the canonical mechanism DSL text."""
        owner = getattr(self, "_mechanism_session_owner", None)
        if owner is None:
            raise RuntimeError("Mechanism session owner is unavailable.")
        return str(owner.canonical_full_dsl)

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

    def _apply_authoritative_result_truth_effects(
        self,
        *,
        cache_stale_set_ids: Sequence[str] = (),
        cache_stale_scope_is_global: bool = True,
        display_clear_set_ids: Sequence[str] = (),
        display_clear_scope_is_global: bool = True,
        clear_cached_mechanism: bool = True,
    ) -> None:
        if bool(clear_cached_mechanism):
            self._clear_last_mechanism()
        active_cache_key = self._simulation_batch_owner.active_cache_key()
        if active_cache_key:
            self._simulation_batch_owner.record_active_result_cache_staleness(
                set_ids=cache_stale_set_ids,
                is_global=bool(cache_stale_scope_is_global),
            )
        transition = self.results_controller.apply_authoritative_result_display_transition(
            active_cache_key=active_cache_key,
            display_scope_ids=self._simulation_batch_owner.effective_display_request_set_ids(),
            active_cache_invalidated_set_ids=self._simulation_batch_owner.active_cache_invalidated_set_ids() or (),
            display_clear_set_ids=display_clear_set_ids,
            display_clear_scope_is_global=display_clear_scope_is_global,
        )
        if transition.refresh_requested:
            self._refresh_batch_display_from_request_scope()

    def _invalidate_active_results_for_global_authoritative_change(self) -> None:
        self._sim_controller.supersede_active_work_for_authoritative_mechanism_transition(
            epoch=int(self._mechanism_runtime_transition.current_epoch)
        )
        self._apply_authoritative_result_truth_effects(
            cache_stale_scope_is_global=True,
            display_clear_scope_is_global=True,
        )

    def _reset_project_apply_dirty_session_state(self) -> None:
        """Clear non-serialized session state before applying a project payload."""
        symbolic_calculator_owner = getattr(self, "_symbolic_calculator_owner", None)
        reset_symbolic_calculator = getattr(symbolic_calculator_owner, "reset_project_session_state", None)
        if callable(reset_symbolic_calculator):
            reset_symbolic_calculator()
            self._refresh_symbolic_calculator_state()

        if not self._prepare_fit_window_shutdown_for_close():
            logger.warning("Project apply requested while one or more fit windows remained open after close request")
        runtime_epoch = max(
            int(getattr(self._mechanism_runtime_transition, "current_epoch", 0) or 0),
            int(getattr(self._sim_controller, "_authoritative_runtime_input_epoch", 0) or 0),
            int(getattr(self._sim_controller, "_authoritative_runtime_input_global_epoch", 0) or 0),
        ) + 1
        self._sim_controller.prepare_runtime_work_for_project_apply(epoch=int(runtime_epoch))
        self._preview_session.clear_working_transaction(clear_committed_slider_values=True)
        dataset_import_coordinator = getattr(self, "_dataset_import_coordinator", None)
        if dataset_import_coordinator is not None:
            dataset_import_coordinator.clear_all_datasets()

        self._pre_dsl_temperature = None
        self._temperature_dsl_override_active = False

        self.results_controller.clear_active_display_transaction()
        self._sync_overlay_catalog()

        self._simulation_batch_owner.reset_runtime_state()

        self._refresh_slider_transaction_button_state()

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
                panel.set_transaction_owner(self._species_panel_transaction_owner())
            panel.attach(table=table, model=model)
        except RuntimeError as exc:
            logger.debug("Failed to reattach species panel after project apply: %s", exc, exc_info=True)
            self._species_panel_available = False
            return

        if hasattr(panel, "activate"):
            self._select_initial_batch_row_if_needed()
            try:
                panel.activate()
            except RuntimeError as exc:
                logger.debug("Failed to reactivate species panel after project apply: %s", exc, exc_info=True)
                self._species_panel_available = False

    def _get_solver_settings(self) -> Dict[str, Any]:
        """Get the current solver settings."""
        solver_contract = load_solver_contract()
        solver_label = str(self._simulation_runtime_settings_owner.current_solver_name() or solver_contract.default_solver_name).strip() or solver_contract.default_solver_name
        solver_method = solver_contract.normalize_solver_name(solver_label)
        return {
            "solver": str(solver_method),
            'rtol': self._simulation_runtime_settings_owner.current_rtol() or 1e-6,
            'atol': self._simulation_runtime_settings_owner.current_atol() or 1e-12,
            'use_sparse_jacobian': bool(self._simulation_runtime_settings_owner.use_sparse_jacobian()),
            'wegscheider_cyclicity_enabled': bool(
                self._simulation_runtime_settings_owner.wegscheider_cyclicity_enabled()
            ),
            'max_parallel_batch_workers': int(self._sim_controller.parallel_batch.max_parallel_workers),
            'batch_runtime_lane_budget': int(self._sim_controller.batch_runtime_lane_budget),
            'limit_blas_threads_per_worker': bool(self._sim_controller.parallel_batch.limit_blas_threads_per_worker),
        }

    def _get_global_fit_launch_settings(self) -> object:
        from kindred.gui.fitting.launch import GlobalFitLaunchSettings

        return GlobalFitLaunchSettings(
            runtime_inputs=self._fitting_runtime_input_publisher.current_inputs(),
        )

    def _require_global_fit_bool(self, value: object, label: str) -> bool:
        if not isinstance(value, bool):
            raise RuntimeError(f"Global Fit launch settings require explicit {label}.")
        return value

    def _require_global_fit_temperature(self, value: object) -> float:
        if isinstance(value, bool):
            raise RuntimeError("Global Fit launch settings require numeric temperature_K.")
        try:
            numeric = float(value)
        except Exception as exc:
            raise RuntimeError("Global Fit launch settings require numeric temperature_K.") from exc
        min_temperature, max_temperature = SIMULATION_TEMPERATURE_K_RANGE
        if (
            not math.isfinite(numeric)
            or numeric < float(min_temperature)
            or numeric > float(max_temperature)
        ):
            raise RuntimeError(
                "Global Fit launch settings require temperature_K within the simulation temperature range."
            )
        return numeric

    def _serialize_project_state(self) -> Dict[str, Any]:
        """Create a snapshot of the current project."""
        from kindred.core.mechanism_source import MechanismAuthoringSource

        solver_contract = load_solver_contract()
        solver_label = str(self._simulation_runtime_settings_owner.current_solver_name() or solver_contract.default_solver_name).strip() or solver_contract.default_solver_name
        solver_name = solver_contract.normalize_solver_name(solver_label)
        mechanism_source = MechanismAuthoringSource.from_parts(
            reactions_text=self.mechanism_reactions_text_raw(),
            state_network_dsl=self.mechanism_state_network_dsl_raw(),
        )
        _pref = self.config_controller.get_user_preference
        fitting_defaults = {
            key: (
                self._fitting_defaults[key]
                if key in self._fitting_defaults
                else _pref(key)
            )
            for key in FITTING_DEFAULTS_KEYS
        }
        return {
            'version': KINDRED_VERSION,
            'mechanism_source': mechanism_source.to_payload(),
            'notes': self._mechanism_editor._notes_text.toPlainText(),
            "solver": str(solver_name),
            'rtol': self._simulation_runtime_settings_owner.current_rtol() or 1e-6,
            'atol': self._simulation_runtime_settings_owner.current_atol() or 1e-12,
            'use_sparse_jacobian': bool(self._simulation_runtime_settings_owner.use_sparse_jacobian()),
            'wegscheider_cyclicity_enabled': bool(
                self._simulation_runtime_settings_owner.wegscheider_cyclicity_enabled()
            ),
            'max_parallel_batch_workers': int(self._sim_controller.parallel_batch.max_parallel_workers),
            'batch_runtime_lane_budget': int(self._sim_controller.batch_runtime_lane_budget),
            'limit_blas_threads_per_worker': bool(self._sim_controller.parallel_batch.limit_blas_threads_per_worker),
            'temperature_K': (
                self._pre_dsl_temperature
                if self._pre_dsl_temperature is not None
                else self._temperature_spinbox.value()
            ),
            'simulation_time': str(self._sim_time_spinbox.text()).strip(),
            'num_points': int(self._num_points_spinbox.value()),
            'batch_initial_conditions': self._batch_store.as_serializable(),
            **fitting_defaults,
        }

    def serialize_project_state(self) -> Dict[str, Any]:
        """Public project snapshot API for controllers (avoid reaching into `_` helpers)."""
        return self._serialize_project_state()

    def _set_plain_text_widget_with_optional_undo(
        self,
        widget: QtWidgets.QPlainTextEdit,
        new_text: str,
        description: str,
        record_undo: bool,
    ) -> None:
        """Set text on a QPlainTextEdit, optionally recording the change on the undo stack."""
        from kindred.gui.undo_commands import SetPlainTextCommand

        current_text = widget.toPlainText()
        if new_text is None:
            new_text = ""

        if current_text == new_text:
            return

        if record_undo:
            command = SetPlainTextCommand(widget, new_text, current_text, description)
            if bool(
                getattr(self, "_suppress_authoritative_mechanism_input_change", False)
                and getattr(self, "_suppress_programmatic_text_change_signals", False)
            ):
                document = widget.document()
                previous_widget_block = widget.blockSignals(True)
                previous_document_block = document.blockSignals(True)
                try:
                    self._undo_stack.push(command)
                finally:
                    document.blockSignals(previous_document_block)
                    widget.blockSignals(previous_widget_block)
            else:
                self._undo_stack.push(command)
        else:
            widget.blockSignals(True)
            try:
                widget.setPlainText(new_text)
            finally:
                widget.blockSignals(False)
            widget.document().contentsChanged.emit()

    def _apply_project_payload(
        self,
        data: Dict[str, Any],
        *,
        record_undo: bool = True,
        draft_apply_result: ReactionsDraftApplyResult | None = None,
        validated_source: MechanismAuthoringSource | None = None,
    ) -> None:
        """Populate the UI from serialized project data."""
        self._suppress_preference_updates = True
        simulation_runtime_settings_changed = False
        batch_runtime_pool_settings_changed = False
        self._authoritative_mechanism_runtime_refresh_defer_depth = (
            int(getattr(self, "_authoritative_mechanism_runtime_refresh_defer_depth", 0) or 0) + 1
        )
        self._simulation_runtime_input_refresh_defer_depth = (
            int(getattr(self, "_simulation_runtime_input_refresh_defer_depth", 0) or 0) + 1
        )
        try:
            with self._fitting_runtime_input_publisher.transaction(reason="project apply"):
                (
                    simulation_runtime_settings_changed,
                    batch_runtime_pool_settings_changed,
                ) = self._apply_project_payload_inner(
                    data,
                    record_undo=record_undo,
                    draft_apply_result=draft_apply_result,
                    validated_source=validated_source,
                )
        finally:
            self._authoritative_mechanism_runtime_refresh_defer_depth = max(
                0,
                int(getattr(self, "_authoritative_mechanism_runtime_refresh_defer_depth", 0) or 0) - 1,
            )
            self._simulation_runtime_input_refresh_defer_depth = max(
                0,
                int(getattr(self, "_simulation_runtime_input_refresh_defer_depth", 0) or 0) - 1,
            )
            self._suppress_preference_updates = False
        pending_mechanism_runtime_refresh = (
            self._consume_pending_authoritative_mechanism_runtime_refresh()
        )
        if simulation_runtime_settings_changed or pending_mechanism_runtime_refresh:
            self._refresh_simulation_runtime_after_inputs_changed(
                batch_runtime_pool_inputs_changed=bool(batch_runtime_pool_settings_changed),
            )

    def _apply_project_payload_inner(
        self,
        data: Dict[str, Any],
        *,
        record_undo: bool = True,
        draft_apply_result: ReactionsDraftApplyResult | None = None,
        validated_source: MechanismAuthoringSource | None = None,
    ) -> tuple[bool, bool]:
        from kindred.core.batch_initial_conditions import (
            BatchInitialConditionsStore,
            batch_initial_conditions_store_is_true_placeholder,
        )
        from kindred.gui.widgets.batch_initial_conditions_table import BatchInitialConditionsTableModel

        previous_runtime_settings = self._simulation_runtime_settings_snapshot()
        previous_batch_runtime_pool_settings = (
            self._simulation_batch_runtime_pool_settings_snapshot()
        )
        source = (
            validated_source
            if validated_source is not None
            else validate_project_payload(data)
        )
        try:
            batch_payload = data.get("batch_initial_conditions")
            try:
                batch_store = (
                    BatchInitialConditionsStore.from_serializable(dict(batch_payload))
                    if batch_payload is not None
                    else BatchInitialConditionsStore()
                )
            except Exception as exc:
                raise ValueError("project payload field 'batch_initial_conditions' is invalid.") from exc

            notes_text = data.get('notes', "")

            batch_model = BatchInitialConditionsTableModel(batch_store, parent=self)
            batch_payload_materialized = (
                batch_payload is not None
                and not batch_initial_conditions_store_is_true_placeholder(batch_store)
            )
            source_acceptance_result = None
            reconciliation_plan = None
            if (
                batch_payload is not None
                and not batch_initial_conditions_store_is_true_placeholder(batch_store)
            ):
                source_acceptance_result = self._initial_conditions_source_acceptance_owner.accept_project_source_after_batch_load(
                    source,
                    batch_store=batch_store,
                    batch_payload_present=True,
                )
                if source_acceptance_result is not None:
                    source = source_acceptance_result.source
            else:
                source, reconciliation_plan = self._prepare_authoritative_source_load_reconciliation(
                    source,
                    prompt_overwrite=False,
                    destination_store=batch_store,
                )
                if reconciliation_plan is None:
                    raise ValueError("project mechanism source Initial Conditions could not be reconciled.")
        except Exception:
            self._canonical_initials_transition_owner.publish_aborted_source_load_attempt(
                transition_source="project_load",
                schedule_runtime_refresh=False,
                draft_apply_result=draft_apply_result,
            )
            raise

        def _prepare_project_source_load_ui() -> None:
            self._reset_project_apply_dirty_session_state()
            self._clear_last_mechanism()
            self._batch_store = batch_store
            self._batch_model = batch_model
            if self._batch_table is not None:
                self._batch_table.setModel(self._batch_model)
                self._rebind_batch_semantics_signal_bindings()
            self._rebind_species_panel_after_batch_model_replacement()
            self._update_batch_row_controls_state()
            self._on_batch_current_changed()
            self._set_plain_text_widget_with_optional_undo(
                self._mechanism_editor._notes_text,
                notes_text,
                "Load project (notes)",
                record_undo,
            )

        previous_source = self.canonical_mechanism_source()
        previous_batch_store = self._batch_store
        previous_batch_model = self._batch_model
        previous_notes_text = self._mechanism_editor._notes_text.toPlainText()
        previous_last_mechanism = self._mechanism_helpers.last_mechanism()
        previous_last_context = self._mechanism_helpers.last_mechanism_context()

        def _restore_project_source_load_ui_after_failure() -> None:
            self._write_complete_authoritative_mechanism_source_to_editor(previous_source)
            self._batch_store = previous_batch_store
            self._batch_model = previous_batch_model
            if self._batch_table is not None:
                self._batch_table.setModel(self._batch_model)
                self._rebind_batch_semantics_signal_bindings()
            self._rebind_species_panel_after_batch_model_replacement()
            self._update_batch_row_controls_state()
            self._on_batch_current_changed()
            self._set_plain_text_widget_with_optional_undo(
                self._mechanism_editor._notes_text,
                previous_notes_text,
                "Restore project notes after failed load",
                False,
            )
            if previous_last_mechanism is None:
                self._clear_last_mechanism()
            else:
                self._mechanism_helpers.remember_last_mechanism(
                    previous_last_mechanism,
                    str(previous_last_context.get("dsl_text") or ""),
                    dict(previous_last_context.get("solver_config") or {}),
                )

        self._execute_authoritative_source_load(
            source=source,
            reconciliation_plan=reconciliation_plan,
            draft_apply_result=draft_apply_result,
            description="Load project mechanism source",
            transition_source="project_load",
            schedule_runtime_refresh=False,
            record_undo=record_undo,
            canonical_materialized=True if batch_payload_materialized else None,
            full_snapshot_required=True,
            before_materialize=_prepare_project_source_load_ui,
            rollback_before_abort_publish=_restore_project_source_load_ui_after_failure,
        )

        _pref = self.config_controller.get_user_preference
        solver_value = data.get('solver', _pref('solver'))
        rtol_value = data.get('rtol', _pref('rtol'))
        atol_value = data.get('atol', _pref('atol'))

        try:
            self._sim_controller.parallel_batch.max_parallel_workers = min(
                int(MAX_PARALLEL_WORKERS_CEILING),
                max(1, int(data.get('max_parallel_batch_workers', _pref('max_parallel_batch_workers')))),
            )
        except Exception:
            self._sim_controller.parallel_batch.max_parallel_workers = int(PROJECT_DEFAULTS['max_parallel_batch_workers'])
        try:
            self._sim_controller.batch_runtime_lane_budget = min(
                int(MAX_PARALLEL_WORKERS_CEILING),
                max(1, int(data.get('batch_runtime_lane_budget', _pref('batch_runtime_lane_budget')))),
            )
        except Exception:
            self._sim_controller.batch_runtime_lane_budget = int(PROJECT_DEFAULTS['batch_runtime_lane_budget'])
        self._sim_controller.parallel_batch.limit_blas_threads_per_worker = bool(
            data.get('limit_blas_threads_per_worker', _pref('limit_blas_threads_per_worker'))
        )
        self._temperature_spinbox.setValue(
            data.get('temperature_K', _pref('temperature_K'))
        )
        # Update stash when project loads while T= override is active.
        if getattr(self, "_temperature_dsl_override_active", False):
            self._pre_dsl_temperature = data.get('temperature_K', _pref('temperature_K'))
        sim_time = data.get('simulation_time', _pref('simulation_time'))
        if isinstance(sim_time, (int, float)):
            sim_time_text = f"{float(sim_time):g}"
        else:
            sim_time_text = str(sim_time)
        self._sim_time_spinbox.setText(sim_time_text)
        self._num_points_spinbox.setValue(
            int(data.get('num_points', _pref('num_points')))
        )

        self._apply_runtime_settings(
            solver=solver_value,
            rtol=rtol_value,
            atol=atol_value,
            use_sparse_jacobian=bool(
                data.get('use_sparse_jacobian', _pref('use_sparse_jacobian'))
            ),
            wegscheider_cyclicity_enabled=bool(
                data.get('wegscheider_cyclicity_enabled', _pref('wegscheider_cyclicity_enabled'))
            ),
        )

        # Loaded project files carry a complete fitting-default snapshot.
        self._fitting_defaults = {
            key: data[key]
            for key in FITTING_DEFAULTS_KEYS
            if data[key] is not None
        }
        runtime_settings_changed = self._simulation_runtime_settings_snapshot() != previous_runtime_settings
        batch_runtime_pool_settings_changed = (
            self._simulation_batch_runtime_pool_settings_snapshot()
            != previous_batch_runtime_pool_settings
        )
        return runtime_settings_changed, batch_runtime_pool_settings_changed

    def apply_project_payload(self, data: Dict[str, Any], *, record_undo: bool = True) -> bool:
        """Public project apply API for controllers (avoid reaching into `_` helpers)."""
        source = validate_project_payload(data)
        preflight_ok, draft_apply_result = self._mechanism_source_transaction_owner.prepare_source_load_preflight(
            action_label="Loading this project",
            guard_action_text="Loading this project",
            transition_source="project_load",
            schedule_runtime_refresh=False,
            continue_label="Load",
        )
        if not preflight_ok:
            return False
        self._apply_project_payload(
            data,
            record_undo=record_undo,
            draft_apply_result=draft_apply_result,
            validated_source=source,
        )
        return True

    def apply_runtime_settings(
        self,
        *,
        solver: object = RUNTIME_SETTING_UNSET,
        rtol: object = RUNTIME_SETTING_UNSET,
        atol: object = RUNTIME_SETTING_UNSET,
        use_sparse_jacobian: object = RUNTIME_SETTING_UNSET,
        wegscheider_cyclicity_enabled: object = RUNTIME_SETTING_UNSET,
    ) -> None:
        self._apply_runtime_settings(
            solver=solver,
            rtol=rtol,
            atol=atol,
            use_sparse_jacobian=use_sparse_jacobian,
            wegscheider_cyclicity_enabled=wegscheider_cyclicity_enabled,
        )
        self._refresh_simulation_runtime_after_inputs_changed(
            batch_runtime_pool_inputs_changed=False
        )

    def add_to_recent_files(self, filepath: str) -> None:
        """Public API used by controllers (avoid reaching into `_` helpers)."""
        self._add_to_recent_files(str(filepath))

    def set_status_text(self, text: str) -> None:
        """Public API used by controllers (avoid reaching into `_` widget fields)."""
        self._simulation_run_ui_owner.set_status_text(str(text))

    def _status_text_value(self) -> str:
        label = getattr(self, "_status_label", None)
        if label is None:
            return ""
        try:
            return str(label.text())
        except RuntimeError:
            self._status_label = None
            return ""

    def main_plot(self) -> object:
        return self._plot_tabs._main_plot

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

    def mechanism_reactions_text_raw(self) -> str:
        return self._simulation_mechanism_owner.mechanism_reactions_text_raw()

    def mechanism_state_network_dsl_raw(self) -> str:
        return self._simulation_mechanism_owner.mechanism_state_network_dsl_raw()

    def canonical_mechanism_source(self):
        return self._mechanism_source_transaction_owner.canonical_source

    def _reactions_draft_apply_error_text(self) -> str:
        editor = getattr(self, "_mechanism_editor", None)
        label = getattr(editor, "_validation_label", None)
        if label is not None and hasattr(label, "text"):
            return str(label.text() or "")
        return ""

    def rewrite_authoritative_mechanism_source(
        self,
        *,
        action_label: str,
        description: str,
        build_source: Callable[[object], object | None],
        record_undo: bool = True,
        transition_source: str = "authoritative_editor_rewrite",
        before_publish: Callable[[], None] | None = None,
        discard_dirty_preview: bool = True,
    ) -> AuthoritativeRewriteOutcome:
        return self._mechanism_source_transaction_owner.rewrite_mechanism_source(
            action_label=str(action_label or description or "Changing the mechanism source"),
            description=str(description or action_label or "Rewrite mechanism source"),
            build_source=build_source,
            record_undo=bool(record_undo),
            transition_source=str(transition_source or "authoritative_editor_rewrite"),
            before_publish=before_publish,
            discard_dirty_preview=bool(discard_dirty_preview),
        )

    def rewrite_authoritative_reactions_text(
        self,
        *,
        action_label: str,
        description: str,
        rewrite_text: Callable[[str], str | None],
        record_undo: bool = True,
        transition_source: str = "authoritative_reactions_rewrite",
        before_publish: Callable[[], None] | None = None,
        discard_dirty_preview: bool = True,
    ) -> AuthoritativeRewriteOutcome:
        return self._mechanism_source_transaction_owner.rewrite_reactions_text(
            action_label=str(action_label or description or "Changing Reactions"),
            description=str(description or action_label or "Rewrite Reactions"),
            rewrite_text=rewrite_text,
            record_undo=bool(record_undo),
            transition_source=str(transition_source or "authoritative_reactions_rewrite"),
            before_publish=before_publish,
            discard_dirty_preview=bool(discard_dirty_preview),
        )

    def set_variable_sliders(
        self,
        variables: Dict[str, float],
        *,
        metadata: Optional[Dict[str, Dict[str, object]]] = None,
        preserve_visibility: bool = False,
        visibility_scope_signature: object | None = None,
    ) -> None:
        self._simulation_mechanism_owner.set_variable_sliders(
            variables,
            metadata=metadata,
            preserve_visibility=preserve_visibility,
            visibility_scope_signature=visibility_scope_signature,
        )

    def variable_slider_values(self) -> Dict[str, float]:
        return self._simulation_mechanism_owner.variable_slider_values()

    def clear_variable_sliders(self) -> None:
        self._simulation_mechanism_owner.clear_variable_sliders()

    def temperature_spinbox_value(self) -> float:
        return self._simulation_runtime_settings_owner.temperature_spinbox_value()

    def parse_sim_time_seconds(self) -> float:
        return self._simulation_runtime_settings_owner.parse_sim_time_seconds()

    def _apply_wegscheider_resolution_reactions_rewrite(self, reactions_text: str) -> AuthoritativeRewriteOutcome:
        return self.rewrite_authoritative_reactions_text(
            action_label="Resolving Wegscheider cyclicity",
            description="Resolve Wegscheider cyclicity",
            rewrite_text=lambda _current: str(reactions_text),
            transition_source="wegscheider_resolution",
        )

    def finalize_authoritative_mechanism_widget_write(self, *, dispatch_consumers: bool) -> None:
        self._sync_mechanism_session_owner_after_authoritative_widget_write(
            dispatch_consumers=bool(dispatch_consumers)
        )

    def set_temperature_override_state(self, *, enabled: bool, tooltip: str) -> None:
        self._set_temperature_override_state(enabled=bool(enabled), tooltip=str(tooltip))

    def set_temperature_mode_indicator_text(self, text: str) -> None:
        self._temperature_mode_indicator.setText(str(text))

    def batch_rows_for_scope(self, scope: str) -> List[int]:
        return self._simulation_batch_owner.batch_rows_for_scope(str(scope))

    def batch_set_ids_for_scope(self, scope: str) -> List[str]:
        return self._simulation_batch_owner.batch_set_ids_for_scope(str(scope))

    def run_target_ui_state(self) -> object:
        return self._simulation_batch_owner.run_target_ui_state()

    def requested_show_batch_set_ids(self) -> List[str]:
        return [str(s) for s in (self._requested_show_batch_set_ids() or [])]

    def effective_display_request_set_ids(self) -> List[str]:
        return self._simulation_batch_owner.effective_display_request_set_ids()

    def slider_edit_target_set_ids(self) -> List[str]:
        return [str(s) for s in (self._slider_edit_target_set_ids() or [])]

    def set_slider_edit_target_set_ids(self, set_ids: Sequence[str]) -> None:
        self._set_slider_edit_target_set_ids(set_ids)

    def batch_current_row(self) -> Optional[int]:
        row = self._batch_current_row()
        return int(row) if row is not None else None

    def focused_batch_set_id(self) -> Optional[str]:
        value = self._active_batch_set_id()
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
        return self._simulation_batch_owner.active_cache_key()

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

    def update_batch_row_controls_state(self) -> None:
        self._update_batch_row_controls_state()

    def sync_batch_species_columns(
        self,
        species_names: Sequence[str],
        *,
        retain_active_cache_identity: bool = False,
    ) -> None:
        self._sync_batch_species_columns(
            list(species_names),
            retain_active_cache_identity=bool(retain_active_cache_identity),
        )

    def has_local_runtime_parameter_values(self) -> bool:
        return self._simulation_mechanism_owner.has_local_runtime_parameter_values()

    def _simulation_schema_text(self) -> str:
        return self._simulation_mechanism_owner.get_mechanism_text()

    def simulation_schema_id(self, *, fast_mode: bool = False) -> str:
        return self._simulation_mechanism_owner.simulation_schema_id(fast_mode=bool(fast_mode))

    def runtime_parameter_fingerprint_for_set(self, set_id: Optional[str] = None, *, fast_mode: bool = True) -> str:
        return self._simulation_mechanism_owner.runtime_parameter_fingerprint_for_set(
            set_id=set_id,
            fast_mode=bool(fast_mode),
        )

    def runtime_parameter_names_for_set(self, set_id: Optional[str] = None, *, fast_mode: bool = True) -> Sequence[str]:
        return self._simulation_mechanism_owner.runtime_parameter_names_for_set(
            set_id=set_id,
            fast_mode=bool(fast_mode),
        )

    def local_runtime_parameter_overrides_for_set(
        self,
        *,
        set_id: Optional[str],
        fast_mode: bool = True,
    ) -> Dict[str, float]:
        return self._simulation_mechanism_owner.local_runtime_parameter_overrides_for_set(
            set_id=set_id,
            fast_mode=bool(fast_mode),
        )

    def focused_runtime_parameter_values(self, *, fast_mode: bool = True) -> Dict[str, float]:
        return self._simulation_mechanism_owner.focused_runtime_parameter_values(
            fast_mode=bool(fast_mode),
        )

    def get_mechanism_text(self) -> str:
        return self._simulation_mechanism_owner.get_mechanism_text()

    def variable_metadata(self) -> Dict[str, Dict[str, object]]:
        return self._simulation_mechanism_owner.variable_metadata()

    def set_variable_metadata(self, metadata: Dict[str, Dict[str, object]] | None) -> None:
        self._variable_runtime.set_variable_metadata(metadata)

    def update_temperature_mode_indicator(self) -> None:
        self._update_temperature_mode_indicator()

    def sync_mechanism_controls_to_focused_batch_set(
        self,
        *,
        use_workspace: bool = True,
    ) -> None:
        self._sync_mechanism_controls_to_focused_batch_set(
            use_workspace=bool(use_workspace),
        )

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
        from kindred.core.simulation_preparation import prepare_simulation_worker_run
        from kindred.core.simulator.solvers import solve_ode

        solver_cfg = self._get_solver_settings()
        temperature = self._temperature_spinbox.value()
        solver_cfg = {
            **dict(solver_cfg or {}),
            "temperature_K": float(temperature),
            "grid": {"N": max(2, int(num_points))},
        }
        prepared = prepare_simulation_worker_run(
            mechanism_text=str(mechanism_text or ""),
            initials=dict(initials or {}),
            t_span=(0.0, float(t_end)),
            solver_config=solver_cfg,
        )
        try:
            result = solve_ode(prepared.request)
        except SimulationError as exc:
            logger.error("Synchronous simulation failed: %s", exc)
            raise
        mechanism = prepared.mechanism
        species_names = list(prepared.species_names)
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
            "solver_provenance": dict(getattr(result, "provenance", {}) or {}),
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
                set_id = self._batch_store.set_id_for_row(int(row))
                return str(set_id) if set_id is not None else None
        except Exception:
            return None
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

    def _requested_show_batch_set_ids(self) -> List[str]:
        model = getattr(self, "_batch_model", None)
        if model is None or not hasattr(model, "requested_show_set_ids"):
            return []
        return [str(set_id) for set_id in model.requested_show_set_ids() if str(set_id)]

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
        owner = getattr(self, "_simulation_batch_owner", None)
        if owner is not None and hasattr(owner, "effective_slider_edit_target_set_ids"):
            try:
                return [
                    str(set_id)
                    for set_id in owner.effective_slider_edit_target_set_ids()
                    if str(set_id)
                ]
            except Exception as exc:
                logger.debug("Failed to resolve slider edit targets from batch owner: %s", exc, exc_info=True)
        return []

    def _effective_slider_edit_target_rows(self) -> tuple[int, ...]:
        rows: list[int] = []
        seen: set[int] = set()
        for set_id in self._effective_slider_edit_target_set_ids():
            row = self._batch_row_for_set_id(str(set_id))
            if row is None:
                continue
            row_i = int(row)
            if row_i in seen:
                continue
            rows.append(row_i)
            seen.add(row_i)
        return tuple(rows)

    @staticmethod
    def _normalized_slider_target_set_ids(set_ids: Sequence[str]) -> tuple[str, ...]:
        normalized: list[str] = []
        seen: set[str] = set()
        for raw_set_id in set_ids or ():
            set_id = str(raw_set_id or "").strip()
            if not set_id or set_id in seen:
                continue
            seen.add(set_id)
            normalized.append(set_id)
        return tuple(normalized)

    def _reconcile_slider_target_membership_change(self) -> None:
        current_target_ids = self._normalized_slider_target_set_ids(
            self._effective_slider_edit_target_set_ids()
        )
        self._last_effective_slider_edit_target_set_ids = current_target_ids
        try:
            preview_removed = self._preview_session.reconcile_slider_target_membership(current_target_ids)
        except RuntimeError as exc:
            logger.debug("Failed to prune slider preview state after target membership change: %s", exc, exc_info=True)
            preview_removed = ()
        if not preview_removed:
            return
        try:
            intersects = True
            inspector = getattr(self._sim_controller, "slider_preview_work_intersects_target_scope", None)
            if callable(inspector):
                intersects = bool(inspector(tuple(preview_removed)))
            if intersects:
                self._sim_controller.invalidate_slider_preview_work(
                    target_set_ids=tuple(preview_removed),
                )
        except RuntimeError as exc:
            logger.debug("Failed to invalidate stale slider preview targets: %s", exc, exc_info=True)

    def _active_batch_set_id(self) -> str:
        current_row = self._batch_current_row()
        if current_row is None:
            return ""
        active_set_id = str(self._batch_set_id_for_row(int(current_row)) or "").strip()
        if active_set_id and self._batch_row_for_set_id(active_set_id) is not None:
            return active_set_id
        return ""

    def _batch_row_for_set_id(self, set_id: str) -> Optional[int]:
        try:
            return self._batch_store.row_for_set_id(str(set_id or ""))
        except Exception:
            return None

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
        try:
            sel.clearSelection()
            for r in new_rows:
                idx = self._batch_model.index(int(r), 0)
                sel.select(idx, QtCore.QItemSelectionModel.Select | QtCore.QItemSelectionModel.Rows)
            if new_rows:
                table.setCurrentIndex(self._batch_model.index(int(new_rows[0]), 0))
        finally:
            if signals_blocked:
                try:
                    sel.blockSignals(False)
                except RuntimeError as exc:
                    logger.debug("Failed to unblock batch selection signals: %s", exc, exc_info=True)
                    signals_blocked = False
        self._update_batch_row_controls_state()

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

    def _datasets_mapped_to_batch_sets(self, *, set_ids: Sequence[str], set_names: Sequence[str]) -> List[str]:
        return list(
            self._dataset_fit_settings_store.datasets_mapped_to_batch_sets(
                set_ids=set_ids,
                set_names=set_names,
            )
        )

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
            lines.append(f"Delete {len(names)} selected sets?")
        else:
            lines.append(f"Delete set '{names[0]}'?")

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
            "Delete Set(s)",
            message,
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.Cancel,
            QtWidgets.QMessageBox.StandardButton.Cancel,
        )
        return response == QtWidgets.QMessageBox.StandardButton.Yes

    def _unmap_datasets_for_deleted_batch_sets(self, *, set_ids: Sequence[str], set_names: Sequence[str]) -> List[str]:
        affected = list(
            self._dataset_fit_settings_store.unmap_batch_sets(
                set_ids=set_ids,
                set_names=set_names,
            )
        )
        if affected:
            self._fitting_runtime_input_publisher.publish_current(
                reason="batch set dataset mapping changed",
                force=True,
            )
        return affected

    def _select_single_batch_row(self, row: int) -> None:
        table = getattr(self, "_batch_table", None)
        if table is None:
            return
        if not (0 <= int(row) < int(self._batch_store.row_count())):
            return
        idx = self._batch_model.index(int(row), 0)
        table.setCurrentIndex(idx)
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
            self._simulation_batch_owner.batch_cache_contains_set(set_id=sid, set_name=name)
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

        if not self._guard_slider_transaction_invalidation(action_text="Deleting the selected set(s)"):
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
        self._simulation_batch_owner.purge_batch_cache_for_deleted_sets(set_ids=delete_ids, set_names=delete_names)
        self.results_controller.deauthorize_display_for_deleted_sets(set_ids=delete_ids)

        target_row = min(rows[0], max(0, int(self._batch_store.row_count()) - 1))
        self._select_single_batch_row(target_row)
        self._update_batch_row_controls_state()
        self._refresh_batch_display_from_request_scope()

    def _requested_show_batch_rows(self) -> List[int]:
        rows: List[int] = []
        for set_id in self._requested_show_batch_set_ids():
            row = self._batch_row_for_set_id(set_id)
            if row is not None:
                rows.append(int(row))
        return rows

    def _update_batch_row_controls_state(self) -> None:
        selected_rows = self._batch_selected_rows()
        allow_reorder = len(selected_rows) <= 1
        if hasattr(self, "_move_batch_up_btn"):
            self._move_batch_up_btn.setEnabled(bool(allow_reorder))
        if hasattr(self, "_move_batch_down_btn"):
            self._move_batch_down_btn.setEnabled(bool(allow_reorder))
        model = getattr(self, "_batch_model", None)
        if model is not None and hasattr(model, "set_active_effective_edit_target_set_id"):
            active_target_set_id = ""
            owner = getattr(self, "_simulation_batch_owner", None)
            if owner is not None and hasattr(owner, "focused_effective_slider_target_set_id"):
                try:
                    active_target_set_id = owner.focused_effective_slider_target_set_id()
                except Exception as exc:
                    logger.debug("Failed to resolve focused slider target from batch owner: %s", exc, exc_info=True)
            model.set_active_effective_edit_target_set_id(active_target_set_id)
        self._refresh_run_target_ui_from_batch_scope()

    def _current_run_target_readiness_signature(self) -> tuple[tuple[int, ...], tuple[str, ...]]:
        owner = getattr(self, "_simulation_batch_owner", None)
        if owner is None or not hasattr(owner, "run_target_ui_state"):
            return ((), ())
        try:
            state = owner.run_target_ui_state()
        except Exception as exc:
            logger.debug("Failed to resolve run-target readiness signature: %s", exc, exc_info=True)
            return ((), ())
        rows = tuple(
            dict.fromkeys(
                int(row)
                for row in (getattr(state, "target_rows", ()) or ())
                if int(row) >= 0
            )
        )
        set_ids = tuple(
            dict.fromkeys(
                str(set_id).strip()
                for set_id in (getattr(state, "target_set_ids", ()) or ())
                if str(set_id).strip()
            )
        )
        return (rows, set_ids)

    def _mark_launch_readiness_stale(self) -> None:
        ui_owner = getattr(self, "_simulation_run_ui_owner", None)
        mark_stale = getattr(ui_owner, "mark_launch_readiness_stale", None)
        if callable(mark_stale):
            mark_stale()

    def _maybe_queue_run_target_runtime_readiness_refresh(self, *, force: bool = False) -> None:
        signature = self._current_run_target_readiness_signature()
        rows, set_ids = signature
        previous = getattr(self, "_last_interactive_runtime_readiness_signature", None)
        if not rows or not set_ids:
            self._last_interactive_runtime_readiness_signature = signature
            self._mark_launch_readiness_stale()
            return
        if not bool(force) and previous == signature:
            return
        self._last_interactive_runtime_readiness_signature = signature
        self._mark_launch_readiness_stale()
        self._queue_interactive_runtime_readiness_refresh(rows)

    def _commit_active_batch_table_editor(self) -> None:
        table = getattr(self, "_batch_table", None)
        commit = getattr(table, "commit_active_editor", None)
        if callable(commit):
            try:
                commit()
            except Exception as exc:
                logger.debug("Failed to commit active batch table editor before run: %s", exc, exc_info=True)

    def _refresh_interactive_runtime_readiness_for_rows(self, rows: Sequence[int] | None) -> None:
        refresh_runtime_readiness = getattr(
            getattr(self, "_sim_controller", None),
            "refresh_interactive_runtime_readiness",
            None,
        )
        if callable(refresh_runtime_readiness):
            if rows is None:
                refresh_runtime_readiness(rows=None, fast_mode=False)
            else:
                normalized_rows = tuple(int(row) for row in rows or ())
                refresh_runtime_readiness(rows=normalized_rows, fast_mode=False)

    def _queue_interactive_runtime_readiness_refresh(self, rows: Sequence[int]) -> None:
        normalized = tuple(dict.fromkeys(int(row) for row in (rows or ())))
        self._pending_interactive_runtime_readiness_rows = normalized
        self._interactive_runtime_readiness_refresh_queued = True
        if QtCore.QCoreApplication.instance() is None:
            self._flush_interactive_runtime_readiness_refresh()
            return
        timer = getattr(self, "_interactive_runtime_readiness_refresh_timer", None)
        if isinstance(timer, QtCore.QTimer):
            timer.start(_INTERACTIVE_RUNTIME_READINESS_REFRESH_DEBOUNCE_MS)
        else:
            QtCore.QTimer.singleShot(
                _INTERACTIVE_RUNTIME_READINESS_REFRESH_DEBOUNCE_MS,
                self._flush_interactive_runtime_readiness_refresh,
            )

    def _flush_interactive_runtime_readiness_refresh(self) -> None:
        timer = getattr(self, "_interactive_runtime_readiness_refresh_timer", None)
        if isinstance(timer, QtCore.QTimer) and timer.isActive():
            timer.stop()
        rows = tuple(getattr(self, "_pending_interactive_runtime_readiness_rows", ()) or ())
        self._pending_interactive_runtime_readiness_rows = ()
        self._interactive_runtime_readiness_refresh_queued = False
        self._refresh_interactive_runtime_readiness_for_rows(rows)

    def _run_simulation_after_committing_batch_initials(self) -> None:
        self._commit_active_batch_table_editor()
        if not self._guard_slider_transaction_invalidation(action_text="Running simulation"):
            return
        self._sim_controller.run_simulation()

    def _refresh_run_target_ui_from_batch_scope(self) -> None:
        owner = getattr(self, "_simulation_batch_owner", None)
        ui_owner = getattr(self, "_simulation_run_ui_owner", None)
        if owner is None or ui_owner is None or not hasattr(owner, "run_target_ui_state"):
            return
        try:
            state = owner.run_target_ui_state()
        except Exception as exc:
            logger.debug("Failed to resolve run-target UI state: %s", exc, exc_info=True)
            return
        render = getattr(ui_owner, "render_run_target_state", None)
        if callable(render):
            render(
                button_text=str(getattr(state, "button_text", "Run") or "Run"),
                target_available=bool(getattr(state, "enabled", False)),
                tooltip=str(getattr(state, "empty_reason", "") or ""),
            )
        self._maybe_queue_run_target_runtime_readiness_refresh()

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
        active_set_id = self._active_batch_set_id()
        row_ids = {
            str(self._batch_set_id_for_row(int(row)) or "").strip()
            for row in rows or ()
            if 0 <= int(row) < total
        }
        if active_set_id and active_set_id in row_ids:
            return active_set_id
        for r in rows:
            rr = int(r)
            if 0 <= rr < total:
                sid = self._batch_set_id_for_row(int(rr))
                if sid:
                    return sid
        return None

    def _sync_batch_species_columns(
        self,
        species_names: Sequence[str],
        *,
        retain_active_cache_identity: bool = False,
    ) -> None:
        """Update batch table columns to match the parsed mechanism species (no data loss)."""
        new_species = [str(s) for s in (species_names or []) if str(s)]
        if new_species == list(self._batch_store.visible_species()):
            return

        selection_snapshot = self._simulation_batch_owner.batch_selection_state_snapshot()
        species_sync_snapshot = self._simulation_batch_owner.batch_species_column_sync_snapshot()

        self._batch_model.set_species(new_species)
        pruned_overlay_set_ids: tuple[str, ...] = ()
        try:
            if hasattr(self._preview_session, "prune_staged_concentration_overlays_to_species"):
                prune_result = self._preview_session.prune_staged_concentration_overlays_to_species(new_species)
                if isinstance(prune_result, bool):
                    pruned_overlay_set_ids = () if not prune_result else self._effective_slider_edit_target_set_ids()
                else:
                    pruned_overlay_set_ids = tuple(
                        dict.fromkeys(str(set_id) for set_id in (prune_result or ()) if str(set_id))
                    )
        except RuntimeError as exc:
            logger.debug("Failed to prune staged concentration overlays after batch species sync: %s", exc, exc_info=True)
        if pruned_overlay_set_ids:
            try:
                self._preview_session.stop_species_slider_update_timer()
            except RuntimeError as exc:
                logger.debug("Failed to stop species slider timer after pruning overlays: %s", exc, exc_info=True)
            try:
                self._sim_controller.invalidate_slider_preview_work(
                    target_set_ids=pruned_overlay_set_ids,
                )
            except RuntimeError as exc:
                logger.debug("Failed to invalidate stale slider preview work after pruning overlays: %s", exc, exc_info=True)
        self._simulation_batch_owner.reconcile_active_cache_after_species_column_sync(
            species_sync_snapshot,
            retain_active_cache_identity=bool(retain_active_cache_identity),
        )
        selection_resolution = self._simulation_batch_owner.resolve_batch_selection_state_snapshot(
            selection_snapshot
        )
        self._apply_batch_selection_state_resolution(selection_resolution)

        try:
            panel = self._mechanism_editor.species_sliders_widget()
            if panel is not None and hasattr(panel, "rebuild_from_current_row"):
                panel.rebuild_from_current_row()
        except RuntimeError as exc:
            logger.debug("Failed to rebuild species panel after batch species sync: %s", exc, exc_info=True)
            self._species_panel_available = False

    def _apply_batch_selection_state_resolution(self, resolution: object) -> None:
        table = getattr(self, "_batch_table", None)
        if table is None:
            return
        sel = table.selectionModel()
        if sel is None:
            return
        rows = tuple(int(row) for row in (getattr(resolution, "selected_rows", ()) or ()))
        focused_row_raw = getattr(resolution, "focused_row", None)
        focused_row = int(focused_row_raw) if focused_row_raw is not None else None
        signals_blocked = False
        try:
            sel.blockSignals(True)
            signals_blocked = True
        except RuntimeError as exc:
            logger.debug("Failed to block batch selection signals: %s", exc, exc_info=True)
            signals_blocked = False
        try:
            sel.clearSelection()
            for row in rows:
                idx = self._batch_model.index(int(row), 0)
                sel.select(idx, QtCore.QItemSelectionModel.Select | QtCore.QItemSelectionModel.Rows)
            if focused_row is not None:
                table.setCurrentIndex(self._batch_model.index(int(focused_row), 0))
        finally:
            if signals_blocked:
                try:
                    sel.blockSignals(False)
                except RuntimeError as exc:
                    logger.debug("Failed to unblock batch selection signals: %s", exc, exc_info=True)

    def _on_batch_selection_changed(self, *_args) -> None:
        self._update_batch_row_controls_state()

    def _batch_current_change_display_source(self) -> DisplayRefreshSource:
        return DisplayRefreshSource.PROGRAMMATIC_SHOW_REQUEST

    def _on_batch_canonical_initials_changed(
        self,
        affected_set_ids: object,
        transition_source: str,
        discard_dirty_preview: bool,
    ) -> None:
        self._canonical_initials_transition_owner.publish_canonical_initials_transition(
            affected_set_ids=tuple(str(set_id) for set_id in (affected_set_ids or ()) if str(set_id)),
            transition_source=str(transition_source or "batch_initials_table_edit"),
            discard_dirty_preview=bool(discard_dirty_preview),
        )

    def _on_batch_initial_rows_edited(
        self,
        affected_set_ids: object,
        transition_source: str,
        discard_dirty_preview: bool,
    ) -> None:
        self._canonical_initials_transition_owner.queue(
            affected_set_ids=tuple(str(set_id) for set_id in (affected_set_ids or ()) if str(set_id)),
            transition_source=str(transition_source or "batch_initials_table_edit"),
            discard_dirty_preview=bool(discard_dirty_preview),
            debounce_ms=_CANONICAL_BATCH_INITIALS_TRANSITION_DEBOUNCE_MS,
        )

    @staticmethod
    def _row_from_model_index(index: object) -> Optional[int]:
        if index is None or not hasattr(index, "isValid") or not index.isValid():
            return None
        try:
            return int(index.row())
        except Exception:
            return None

    def _on_batch_current_changed(self, current: object = None, previous: object = None) -> None:
        self._update_batch_row_controls_state()
        current_row = self._row_from_model_index(current)
        if current_row is None:
            current_row = self._batch_current_row()
        previous_row = self._row_from_model_index(previous)
        if previous_row is not None and current_row == previous_row:
            return
        transaction = self._simulation_batch_owner.concentration_set_interaction_transaction(
            gesture="row_body_click",
            row=current_row,
        )
        try:
            self._refresh_slider_edit_targets_summary()
            self._reconcile_slider_target_membership_change()
        except Exception as exc:
            logger.debug("Failed to refresh slider target membership after row activation: %s", exc, exc_info=True)
        try:
            panel = self._mechanism_editor.species_sliders_widget()
            if panel is not None and hasattr(panel, "refresh_current_row_from_model"):
                panel.refresh_current_row_from_model(recompute_ranges=True)
            elif panel is not None and hasattr(panel, "rebuild_from_current_row"):
                panel.rebuild_from_current_row()
        except RuntimeError as exc:
            logger.debug("Failed to refresh species panel after row activation: %s", exc, exc_info=True)
            self._species_panel_available = False
        if transaction.display_refresh_needed:
            self._refresh_batch_display_from_request_scope(
                display_source=self._batch_current_change_display_source(),
                rebuild_species_sliders=False,
            )

    def _on_batch_show_membership_changed(self) -> None:
        transaction = self._simulation_batch_owner.concentration_set_interaction_transaction(
            gesture="show_membership_change",
            row=self._batch_current_row(),
        )
        if transaction.slider_rebuild_needed:
            self._reconcile_slider_target_membership_change()
        if transaction.display_refresh_needed:
            self._refresh_batch_display_from_request_scope(
                display_source=DisplayRefreshSource.EXPLICIT_SHOW_REQUEST,
            )

    def _on_slider_edit_targets_changed(self) -> None:
        try:
            panel = self._mechanism_editor.species_sliders_widget()
            if (
                panel is not None
                and hasattr(panel, "live_drag_active")
                and bool(panel.live_drag_active())
            ):
                self._slider_edit_targets_changed_deferred_during_species_drag = True
                self._refresh_slider_edit_targets_summary()
                return
        except RuntimeError as exc:
            logger.debug("Failed to inspect species panel during slider target change: %s", exc, exc_info=True)
            self._species_panel_available = False
        transaction = self._simulation_batch_owner.concentration_set_interaction_transaction(
            gesture="slider_target_change",
            row=self._batch_current_row(),
        )
        self._reconcile_slider_target_membership_change()
        self._refresh_slider_edit_targets_summary()
        if transaction.slider_rebuild_needed:
            try:
                panel = self._mechanism_editor.species_sliders_widget()
                if panel is not None and hasattr(panel, "refresh_current_row_from_model"):
                    panel.refresh_current_row_from_model(recompute_ranges=True)
                elif panel is not None and hasattr(panel, "rebuild_from_current_row"):
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

    def _refresh_batch_display_from_request_scope(
        self,
        *,
        display_source: object | None = None,
        rebuild_species_sliders: bool = True,
    ) -> None:
        outcome = self.results_controller.refresh_display_from_request_scope(
            display_source=display_source,
        )
        if outcome.focused_controls_use_workspace is not None:
            self._sync_mechanism_controls_to_focused_batch_set(
                use_workspace=bool(outcome.focused_controls_use_workspace),
                rebuild_species_sliders=bool(rebuild_species_sliders),
            )

    def _runtime_parameter_values_for_materialization(
        self,
        *,
        set_id: Optional[str] = None,
        overrides: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        if overrides is not None:
            raw = dict(overrides or {})
        elif set_id is not None:
            raw = self.local_runtime_parameter_overrides_for_set(set_id=set_id, fast_mode=True)
        else:
            raw = self.focused_runtime_parameter_values(fast_mode=True)
        normalized: Dict[str, float] = {}
        for key, value in raw.items():
            parsed, ok = try_parse_finite_float(value)
            if not ok:
                continue
            normalized[str(key)] = float(parsed)
        return normalized

    def _apply_reactions_overrides_to_text(
        self,
        base_text: str,
        *,
        set_id: Optional[str] = None,
        overrides: Optional[Dict[str, float]] = None,
    ) -> str:
        """Return Reactions-layer DSL with slider overrides applied."""
        drag_baseline_text = self._preview_session.drag_baseline_text()
        if self._preview_session.slider_drag_active() and drag_baseline_text is not None:
            text = drag_baseline_text
        else:
            text = base_text
        override_map = self._runtime_parameter_values_for_materialization(set_id=set_id, overrides=overrides)
        has_energy_overrides = False
        metadata = self.variable_metadata() or {}
        step_analysis_context = None
        step_constraint_context = {
            "temperature_K": float(self._temperature_spinbox.value()),
            "wegscheider_cyclicity_enabled": bool(self._simulation_runtime_settings_owner.wegscheider_cyclicity_enabled()),
        }
        for var_name, var_value in override_map.items():
            meta = metadata.get(var_name, {})
            if isinstance(meta, dict) and meta.get("type") == "scalar":
                continue
            if isinstance(meta, dict) and meta.get("type") == "energy":
                has_energy_overrides = True
                continue
            if isinstance(meta, dict) and meta.get("type") == "equilibrium" and meta.get("role") in {"kf", "kr", "Keq"}:
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
                metadata=metadata,
                step_analysis_context=step_analysis_context,
            )
            if text != previous_text:
                step_analysis_context = None
        if has_energy_overrides:
            text = self._apply_energy_overrides_to_computational_mode_fast_equilibria(text, overrides=override_map)
        return text

    def _collect_energy_overrides(
        self,
        *,
        set_id: Optional[str] = None,
        overrides: Optional[Dict[str, float]] = None,
        override_metadata: Optional[Dict[str, Dict[str, object]]] = None,
    ) -> list[tuple[str, float, dict]]:
        meta_map = dict(self.variable_metadata() or {})
        meta_map.update(
            {
                str(name): dict(meta or {})
                for name, meta in dict(override_metadata or {}).items()
                if str(name).strip() and isinstance(meta, dict)
            }
        )
        override_map = self._runtime_parameter_values_for_materialization(set_id=set_id, overrides=overrides)
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

    def _apply_energy_overrides_to_computational_mode_fast_equilibria(
        self,
        base_text: str,
        *,
        set_id: Optional[str] = None,
        overrides: Optional[Dict[str, float]] = None,
        override_metadata: Optional[Dict[str, Dict[str, object]]] = None,
    ) -> str:
        """
        Apply energy-mode ΔG° overrides to Computational Mode fast equilibria inside the generated block.

        Generated fast equilibria keep source-owned `kf=` and `cm_std_ratio=`
        values in the generated block and rewrite only source-owned `dG_eq=`.
        """
        energy_overrides = [
            o
            for o in self._collect_energy_overrides(
                set_id=set_id,
                overrides=overrides,
                override_metadata=override_metadata,
            )
            if str(o[2].get("role") or "") == "dG_eq_fast"
        ]
        if not energy_overrides:
            return base_text

        T_override = self._variable_runtime.dsl_global_temperature_k(base_text)
        try:
            T = float(T_override) if T_override is not None else float(getattr(self, "_temperature_spinbox").value())
        except Exception:
            return base_text
        if not (math.isfinite(T) and T > 0.0):
            return base_text

        return materialize_generated_fast_dg_overrides_to_dsl_text(
            base_text,
            energy_overrides,
            temperature_K=T,
        )

    def _updated_reactions_text_with_scalar_param(self, reactions_text: str, name: str, value: float) -> str:
        from kindred.core.simulator.algebra_section import (
            is_param_algebra_line,
            upsert_lines_into_algebra_section,
        )

        formatted_value = format_authoritative_parameter_value(value)
        reactions_lines = str(reactions_text or "").splitlines()

        updated = False
        for i, raw in enumerate(reactions_lines):
            stripped = raw.strip()
            if not stripped or stripped.startswith("#") or not is_param_algebra_line(raw):
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
        return upsert_lines_into_algebra_section(
            "\n".join(reactions_lines).rstrip("\n") + "\n",
            [f"param {name} = {formatted_value}"],
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
        Update `param <name> = ...` inside the Reactions DSL text.

        Notes are persisted separately and are never parsed or injected into the DSL.
        """
        self.rewrite_authoritative_reactions_text(
            action_label=f"Updating param {name} in Reactions DSL",
            description=f"Update param {name} in Reactions DSL",
            rewrite_text=lambda current_text: self._updated_reactions_text_with_scalar_param(
                str(current_text or ""),
                str(name),
                float(value),
            ),
            record_undo=True,
            transition_source="scalar_param_rewrite",
        )

    def _parameter_algebra_spec_for_ui(self):
        from kindred.core.simulator.algebra_section import extract_algebra_section_text

        reactions_text = self.mechanism_reactions_text_raw()
        if not extract_algebra_section_text(reactions_text).strip():
            return None
        from kindred.core.simulator.dsl import parse_dsl_to_mechanism
        from kindred.core.simulator.parameter_algebra import (
            mechanism_parameter_namespace,
            parse_parameter_algebra_spec_from_dsl_text,
        )

        mechanism = parse_dsl_to_mechanism(str(reactions_text or ""), initials={})
        mechanism_namespace = mechanism_parameter_namespace(mechanism)
        return parse_parameter_algebra_spec_from_dsl_text(
            str(reactions_text or ""),
            mechanism_namespace=mechanism_namespace,
        )

    def _refresh_derived_parameters_display(self) -> None:
        sliders = self._mechanism_editor._variable_sliders
        current = sliders.get_variables()
        if not current:
            return
        spec = None
        try:
            spec = self._parameter_algebra_spec_for_ui()
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
                meta = self._variable_runtime.update_variable_metadata_entry(
                    name,
                    {
                        "editable": False,
                        "derived": True,
                    },
                )
                sliders.update_metadata(name, meta)

        # Refresh snapshot after any derived updates.
        current = sliders.get_variables()

        # Apply authority-implied readout updates (canonical step indexing) so
        # derived rates update immediately when their authority inputs change.
        step_map = getattr(self, "_step_index_map", None) or []
        if isinstance(step_map, list) and step_map:
            from kindred.core.equilibrium_rate_authority import authority_readout_updates_from_step_entry

            for entry in step_map:
                if not isinstance(entry, dict):
                    continue
                if str(entry.get("kind") or "") != "equilibrium":
                    continue
                for update in authority_readout_updates_from_step_entry(entry, current):
                    current[update.name] = float(update.value)
                    sliders.update_variable_readout(update.name, float(update.value))
                    meta = self._variable_runtime.update_variable_metadata_entry(
                        update.name,
                        {
                            "editable": bool(update.editable),
                            "derived": bool(update.derived),
                        },
                    )
                    sliders.update_metadata(update.name, meta)

    def _energy_mode_energy_unit_from_metadata(self) -> str:
        meta_map = self.variable_metadata() or {}
        for _name, meta in meta_map.items():
            if isinstance(meta, dict) and meta.get("type") == "energy" and meta.get("unit"):
                return str(meta.get("unit"))
        return "kJ/mol"

    def _energy_mode_full_dsl_from_editor(self) -> str:
        return self.canonical_mechanism_source().full_dsl

    def _energy_mode_temperature_K(self, full_dsl: str) -> float | None:
        T_override = self._variable_runtime.dsl_global_temperature_k(full_dsl)
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
            return None
        try:
            dG_eq = float(
                self.focused_runtime_parameter_values(fast_mode=True).get(var_eq, ch.get("dG_eq"))
            )
        except Exception as exc:
            self._record_best_effort_failure(
                "main_window.energy_table.fast_eq.dG_eq",
                message=f"Failed to read fast-equilibrium dG_eq value ({label})",
                exc=exc,
            )
            return None
        if not math.isfinite(dG_eq):
            return None
        source_unit = str(ch.get("source_unit") or energy_unit)
        source_unit_conv = unit_conv
        try:
            from kindred.core.units import UnitsModel

            source_unit_conv = UnitsModel(energy_unit=source_unit)
        except Exception:
            source_unit = str(energy_unit)
            source_unit_conv = unit_conv
        try:
            dG_eq_J = float(source_unit_conv.to_jmol(dG_eq))
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
            f"ΔG° ({label})": (dG_eq, source_unit),
            f"k_f ({label})": (kf_fixed, unit_kf),
            f"k_r ({label})": (kr, unit_kr),
            f"Keq ({label})": (K, "1"),
        }

    def _refresh_energy_mode_derived_parameter_table(self) -> None:
        """
        Update the parameter table for energy-mode channels from current slider values.

        This is cheap (no ODE solve) and keeps the read-only k_f/k_r/Keq display in sync
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
            from kindred.core.kinetics import K_from_deltaG_eq
            from kindred.core.simulator.kinetics import rate_units
            from kindred.core.units import UnitsModel
        except Exception:
            return

        unit_conv = UnitsModel(energy_unit=energy_unit)
        params: Dict[str, Tuple[float, str]] = {}

        for ch in channels:
            if not isinstance(ch, dict):
                continue
            if str(ch.get("kind") or "") != "fast_equilibrium":
                continue
            out = self._energy_mode_params_for_fast_equilibrium_channel(
                ch,
                energy_unit=energy_unit,
                unit_conv=unit_conv,
                T=T,
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

    def _on_variable_changed(self, name: str, value: float):
        self._preview_session.on_variable_changed(name, value)
        self._refresh_slider_transaction_button_state()

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

    def _commit_species_slider_values_to_selected_batch_rows(self) -> tuple[str, ...]:
        model = getattr(self, "_batch_model", None)
        if model is None:
            return ()
        try:
            result = self._preview_session.apply_staged_concentration_overlays(model)
            return tuple(str(set_id) for set_id in getattr(result, "touched_set_ids", ()) if str(set_id))
        except Exception as exc:
            self._record_best_effort_failure(
                "main_window.species_mode.commit_species.apply",
                message="Failed to apply staged species-mode concentration overlays",
                exc=exc,
            )
            raise

    def _materialized_mechanism_editor_source_for_runtime_parameter_values(
        self,
        values: Dict[str, float],
        *,
        base_source=None,
    ):
        from kindred.core.mechanism_source import MechanismAuthoringSource

        source = base_source if base_source is not None else self.canonical_mechanism_source()
        if not isinstance(source, MechanismAuthoringSource):
            raise TypeError("base_source must be a MechanismAuthoringSource.")
        non_scalar_values, scalar_values = self._partition_effective_slider_commit_values(values)
        reactions_text = str(source.reactions_text)
        if non_scalar_values:
            reactions_text = self._apply_reactions_overrides_to_text(
                reactions_text,
                overrides=non_scalar_values,
            )
        if scalar_values:
            reactions_text = self._apply_scalar_param_overrides_to_reactions_text(
                reactions_text,
                scalar_values,
            )

        return MechanismAuthoringSource.from_parts(
            reactions_text=reactions_text,
            state_network_dsl=str(source.state_network_dsl),
        )

    def _partition_effective_slider_commit_values(
        self,
        values: Dict[str, float],
    ) -> tuple[Dict[str, float], Dict[str, float]]:
        metadata = self.variable_metadata() or {}
        non_scalar_values: Dict[str, float] = {}
        scalar_values: Dict[str, float] = {}
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
            if isinstance(meta, dict) and meta.get("type") == "energy":
                if str(meta.get("role") or "") == "dG_eq_fast":
                    non_scalar_values[name_s] = value_f
                continue
            non_scalar_values[name_s] = value_f
        return non_scalar_values, scalar_values

    def _set_authoritative_mechanism_editor_source(
        self,
        source,
        *,
        description: str,
        record_undo: bool = True,
    ) -> bool:
        return self._mechanism_source_transaction_owner.set_authoritative_editor_source(
            source,
            description=str(description),
            record_undo=bool(record_undo),
        )

    def _write_complete_authoritative_mechanism_source_to_editor(self, source) -> bool:
        return self._mechanism_source_transaction_owner.write_complete_source_to_editor(source)

    def _apply_mechanism_source_undo_stack_replacement(self, source) -> None:
        self._mechanism_source_transaction_owner.apply_undo_stack_replacement(source)

    def _rewrite_runtime_parameter_values_to_mechanism_editors(
        self,
        values: Optional[Dict[str, float]] = None,
        *,
        description: str = "Apply slider overrides",
        values_provider: Optional[Callable[[], Dict[str, float]]] = None,
        before_publish: Callable[[], None] | None = None,
    ) -> AuthoritativeRewriteOutcome:
        def _values_for_builder() -> Dict[str, float]:
            if callable(values_provider):
                return dict(values_provider() or {})
            return dict(values or {})

        outcome = self.rewrite_authoritative_mechanism_source(
            action_label=str(description),
            description=str(description),
            build_source=lambda source: self._materialized_mechanism_editor_source_for_runtime_parameter_values(
                _values_for_builder(),
                base_source=source,
            ),
            transition_source="slider_materialization",
            before_publish=before_publish,
        )
        return outcome

    def _refresh_after_authoritative_slider_materialization(
        self,
        *,
        affected_canonical_initial_set_ids: Sequence[str] = (),
    ) -> None:
        try:
            self._variable_runtime.extract_and_populate_variables(
                preserve_visibility=True,
            )
        except Exception:
            logger.exception("Failed to refresh sliders after authoritative slider materialization")
        try:
            panel = self._mechanism_editor.species_sliders_widget()
            if panel is not None and hasattr(panel, "rebuild_from_current_row"):
                panel.rebuild_from_current_row()
        except Exception:
            logger.exception("Failed to rebuild species panel after slider materialization")
        self._refresh_slider_transaction_button_state()

    def _commit_runtime_parameters_from_sliders(
        self,
        effective_values: Mapping[str, float] | None,
        *,
        description: str,
        apply_species_overlays: bool,
    ) -> None:
        species_overlays_pending = bool(apply_species_overlays) and bool(
            self._preview_session.has_staged_concentration_overlays()
        )
        affected_canonical_initial_set_ids: tuple[str, ...] = ()
        species_committed = False
        timers_stopped = False

        def stop_timers_once() -> None:
            nonlocal timers_stopped
            if timers_stopped:
                return
            self._preview_session.stop_variable_update_timer()
            self._preview_session.stop_slider_release_commit_timer()
            timers_stopped = True

        def commit_species_overlays_once() -> tuple[str, ...]:
            nonlocal affected_canonical_initial_set_ids, species_committed
            if species_committed or not species_overlays_pending:
                return affected_canonical_initial_set_ids
            stop_timers_once()
            affected_canonical_initial_set_ids = tuple(
                str(set_id)
                for set_id in (
                    self._commit_species_slider_values_to_selected_batch_rows() or ()
                )
                if str(set_id)
            )
            species_committed = True
            return affected_canonical_initial_set_ids

        outcome = self._rewrite_runtime_parameter_values_to_mechanism_editors(
            dict(effective_values or {}),
            description=str(description),
            before_publish=(
                commit_species_overlays_once
                if species_overlays_pending
                else None
            ),
        )
        if outcome.status is AuthoritativeRewriteStatus.CANCELLED:
            self._status_label.setText("Canceled slider commit")
            return
        if outcome.status is AuthoritativeRewriteStatus.FAILED:
            self._status_label.setText(str(outcome.error or "Slider commit failed"))
            return

        has_parameter_write = (
            outcome.status is AuthoritativeRewriteStatus.REWRITE_APPLIED
        )
        has_authoritative_source_write = bool(outcome.source_updated)
        if species_overlays_pending and not species_committed:
            commit_species_overlays_once()
        has_species_write = bool(affected_canonical_initial_set_ids)
        if (
            not has_parameter_write
            and not has_species_write
            and not has_authoritative_source_write
        ):
            return
        stop_timers_once()
        self._sim_controller.discard_slider_preview_work_preserving_runtime_owner(
            target_set_ids=affected_canonical_initial_set_ids,
        )
        self._preview_session.clear_current_mechanism_workspace_after_source_commit(
            invalidate_preview_work=False
        )
        self._refresh_after_authoritative_slider_materialization(
            affected_canonical_initial_set_ids=affected_canonical_initial_set_ids,
        )

    def _sync_mechanism_controls_to_focused_batch_set(
        self,
        *,
        use_workspace: bool = True,
        rebuild_species_sliders: bool = True,
    ) -> None:
        sliders = getattr(getattr(self, "_mechanism_editor", None), "_variable_sliders", None)
        if sliders is not None and hasattr(sliders, "get_variables"):
            current_values = sliders.get_variables() or {}
            if current_values:
                focused_set_id = self._preview_session.focused_mechanism_workspace_set_id()
                if bool(use_workspace):
                    effective_values = self._preview_session.runtime_parameter_values(set_id=focused_set_id)
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
                    self._variable_runtime.update_parameter_table_from_sliders()

        if bool(rebuild_species_sliders):
            try:
                panel = self._mechanism_editor.species_sliders_widget()
                if (
                    panel is not None
                    and hasattr(panel, "live_drag_active")
                    and bool(panel.live_drag_active())
                ):
                    return
                if panel is not None and hasattr(panel, "rebuild_from_current_row"):
                    panel.rebuild_from_current_row()
            except Exception:
                logger.exception("Failed to rebuild species panel while syncing focused mechanism controls")

    def _on_commit_runtime_parameters_clicked(self) -> None:
        """
        Commit current slider values into the DSL editor.

        Runtime parameter sliders update preview simulations without rewriting
        the mechanism text. "Commit" applies the current finite parameter values to the DSL.
        """
        focused_set_id = self._preview_session.focused_mechanism_workspace_set_id()
        effective_values = self._preview_session.runtime_parameter_values(set_id=focused_set_id)
        self._commit_runtime_parameters_from_sliders(
            effective_values,
            description="Apply runtime parameters",
            apply_species_overlays=True,
        )

    def _on_reset_runtime_parameters_clicked(self) -> None:
        """Reset staged runtime parameter values back to the baseline DSL values."""
        replay_intent = self._preview_session.capture_reset_slider_replay_intent()
        target_set_ids = tuple(
            str(set_id)
            for set_id in getattr(replay_intent, "target_set_ids", ())
            if str(set_id)
        )
        if not target_set_ids:
            target_set_ids = tuple(
                dict.fromkeys(
                    (
                        *tuple(self._preview_session.local_mechanism_workspace_set_ids()),
                        *tuple(self._preview_session.staged_concentration_overlay_target_set_ids()),
                    )
                )
            )
        if not target_set_ids:
            target_set_ids = self._effective_slider_edit_target_set_ids()
        self._discard_pending_slider_edits_for_invalidation(
            target_set_ids=target_set_ids,
            discard_entire_transaction=True,
        )
        self.results_controller.reset_preview_display_to_canonical(
            target_set_ids=target_set_ids,
        )
        self._sim_controller.refresh_interactive_runtime_readiness()
        return

    def _discard_pending_slider_edits_for_invalidation(
        self,
        *,
        target_set_ids: Sequence[str] | None = None,
        discard_species_preview: bool = True,
        discard_entire_transaction: bool = False,
    ) -> bool:
        """Discard staged slider edits and leave the visible display recoverable."""
        targets = target_set_ids or self._effective_slider_edit_target_set_ids()
        self._sim_controller.invalidate_slider_preview_work(
            target_set_ids=targets,
            clear_plot=False,
            clear_active_cache_identity=False,
            clear_pending_replay=bool(discard_entire_transaction),
        )
        if bool(discard_entire_transaction):
            self._preview_session.clear_working_transaction(invalidate_preview_work=False)
        elif target_set_ids is not None:
            if bool(discard_species_preview):
                self._preview_session.clear_working_transaction_for_set_ids(
                    targets,
                    invalidate_preview_work=False,
                )
            else:
                self._preview_session.reset_mechanism_workspaces(targets)
        else:
            self._preview_session.clear_working_transaction(invalidate_preview_work=False)

        sliders = getattr(getattr(self, "_mechanism_editor", None), "_variable_sliders", None)
        if sliders is not None and hasattr(sliders, "end_live_drag"):
            sliders.end_live_drag()

        try:
            self._variable_runtime.extract_and_populate_variables(
                preserve_visibility=True,
            )
        except Exception:
            logger.exception("Failed to refresh sliders after override reset")
        try:
            self._variable_runtime.update_parameter_table_from_sliders()
        except Exception:
            logger.exception("Failed to update parameter table after override reset")
            QtCore.QTimer.singleShot(
                0,
                self._variable_runtime.update_parameter_table_from_sliders,
            )

        try:
            panel = self._mechanism_editor.species_sliders_widget()
            if panel is not None and hasattr(panel, "rebuild_from_current_row"):
                panel.rebuild_from_current_row()
        except Exception:
            logger.exception("Failed to reset species row")
            self._species_panel_available = False
        self._refresh_slider_transaction_button_state()
        return True

    # ------------------------------------------------------------------
    # Species mode (Batch Initial Conditions sliders)
    # ------------------------------------------------------------------

    def _select_initial_batch_row_if_needed(self) -> None:
        table = getattr(self, "_batch_table", None)
        model = getattr(self, "_batch_model", None)
        if table is None or model is None:
            return
        if int(model.rowCount()) <= 0:
            return
        if table.currentIndex().isValid() or self._batch_selected_rows():
            return
        transaction = self._simulation_batch_owner.concentration_set_interaction_transaction(
            gesture="initial_default_selection",
            row=0,
        )
        if not (transaction.focus_change or transaction.selection_change):
            return
        self._select_single_batch_row(0)

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
        target_set_ids = self._effective_slider_edit_target_set_ids()
        try:
            rows = []
            for set_id in target_set_ids:
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
            self.results_controller.clear_display_if_workspace_previews_were_displayed(
                target_set_ids,
                clear_plot=False,
            )
            self._refresh_slider_transaction_button_state()
            self.results_controller.reset_preview_display_to_canonical(
                target_set_ids=target_set_ids,
            )
            self._sim_controller.refresh_interactive_runtime_readiness()

    def _on_species_slider_edited(self, species: str, value: float) -> None:
        changed = self._preview_session.on_species_slider_value_changed(species, float(value))
        self._refresh_slider_transaction_button_state()
        if changed:
            self._preview_session.queue_species_slider_simulation(label=f"init:{species}", delay_ms=80)

    def _on_species_slider_drag_started(self, species: str) -> None:
        self._preview_session.on_species_slider_drag_started(species)

    def _on_species_slider_drag_finished(self, species: str) -> None:
        self._preview_session.on_species_slider_drag_finished(species)
        if bool(getattr(self, "_slider_edit_targets_changed_deferred_during_species_drag", False)):
            self._slider_edit_targets_changed_deferred_during_species_drag = False
            self._on_slider_edit_targets_changed()

    def _queue_species_slider_simulation(self, *, label: str, delay_ms: int) -> None:
        self._preview_session.queue_species_slider_simulation(label=label, delay_ms=delay_ms)

    @staticmethod
    def _derived_equilibrium_role_from_metadata(metadata: dict[str, object], eq_index: int) -> str:
        kf_meta = metadata.get(f"kf{eq_index}")
        kr_meta = metadata.get(f"kr{eq_index}")
        if isinstance(kf_meta, dict) and (kf_meta.get("derived") is True or kf_meta.get("editable") is False):
            return "kf"
        if isinstance(kr_meta, dict) and (kr_meta.get("derived") is True or kr_meta.get("editable") is False):
            return "kr"
        # Default policy matches core: thermodynamic authority derives kr.
        return "kr"

    @staticmethod
    def _label_for_step_from_metadata(metadata: dict[str, object], step_index: int, fallback_prefix: str) -> str:
        # Preserve canonical labels generated from step_index_map ("Step N: ...").
        for key in (f"kf{step_index}", f"kr{step_index}", f"Keq{step_index}", f"k{step_index}"):
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
        strict: bool = False,
        metadata: Optional[Dict[str, Dict[str, object]]] = None,
        step_analysis_context: object | None = None,
    ) -> str:
        """
        Update a variable value in the mechanism text.

        Parameters
        ----------
        name : str
            Variable name (e.g., 'k1', 'Keq1', 'kf1')
        value : float
            New value
        source_text : str | None
            Optional mechanism DSL to apply the change to. If None, use canonical Reactions text.

        Returns
        -------
        str
            Mechanism DSL with the updated value applied.
        """
        if source_text is None:
            mechanism_text = str(self.canonical_mechanism_source().reactions_text)
        else:
            mechanism_text = source_text

        if metadata is None:
            metadata = dict(self.variable_metadata())
        step_constraint_context = {
            "temperature_K": float(self._temperature_spinbox.value()),
            "wegscheider_cyclicity_enabled": bool(self._simulation_runtime_settings_owner.wegscheider_cyclicity_enabled()),
        }
        try:
            outcome = analyze_step_parameter_update(
                mechanism_text,
                name,
                value,
                step_constraint_context=step_constraint_context,
                step_analysis_context=step_analysis_context,
            )
        except ValueError as exc:
            if strict:
                raise ValueError(f"Invalid request parameter override {name!r}: {exc}") from exc
            logger.warning("Ignoring invalid step parameter update for %r", name)
            return mechanism_text

        if not outcome.found_target or not outcome.writable:
            if strict:
                raise ValueError(f"Invalid request parameter override {name!r}.")
            return mechanism_text
        if not outcome.would_change_text:
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

        if family == "Keq":
            label_text = self._label_for_step_from_metadata(metadata, step_index, line_label)
            K_meta = dict(metadata.get(f"Keq{step_index}") or {})
            K_meta.update(
                {
                    "type": "equilibrium",
                    "index": step_index,
                    "label": K_meta.get("label") or label_text,
                    "line": line_index,
                    "role": "Keq",
                }
            )
            metadata[f"Keq{step_index}"] = K_meta
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
        elif family == "kf":
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
            if f"kr{step_index}" in resolved_values:
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
            if f"Keq{step_index}" in resolved_values:
                K_meta = dict(metadata.get(f"Keq{step_index}") or {})
                K_meta.update(
                    {
                        "type": "equilibrium",
                        "index": step_index,
                        "label": K_meta.get("label") or label_text,
                        "line": line_index,
                        "role": "Keq",
                    }
                )
                metadata[f"Keq{step_index}"] = K_meta
        elif family == "kr":
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
            if f"Keq{step_index}" in resolved_values:
                K_meta = dict(metadata.get(f"Keq{step_index}") or {})
                K_meta.update(
                    {
                        "type": "equilibrium",
                        "index": step_index,
                        "label": K_meta.get("label") or label_text,
                        "line": line_index,
                        "role": "Keq",
                    }
                )
                metadata[f"Keq{step_index}"] = K_meta
        elif family == "k":
            label_text = self._label_for_step_from_metadata(metadata, step_index, line_label)
            metadata[name] = {
                "type": "reaction",
                "index": step_index,
                "label": label_text,
                "line": line_index,
                "role": "k",
            }

        return new_text

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

    def _set_temperature_spinbox_visible(self, visible: bool) -> None:
        spin = getattr(self, "_temperature_spinbox", None)
        if spin is None:
            return
        spin.setVisible(bool(visible))
        label = getattr(self, "_temperature_label", None)
        if label is not None:
            label.setVisible(bool(visible))

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
                    "- README.md (project overview and installation)\n"
                    "- Check the project repository for documentation"
                )
                QtWidgets.QMessageBox.information(self, "Documentation", fallback)
                self._status_label.setText("Documentation URL unavailable")
        else:
            logger.info("Documentation requested but no online URL is configured")
            message = (
                f"Online documentation for Kindred v{KINDRED_VERSION} is not yet available.\n\n"
                "Available local resources:\n"
                "- README.md for project overview and installation\n"
                "- Check the project repository for documentation"
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

    def _available_intervention_example_specs(self) -> List[Dict[str, str]]:
        """Return bundled intervention examples discovered under kindred/data/interventions."""
        try:
            from kindred.io.resources import get_all_intervention_example_specs
        except Exception as exc:  # pragma: no cover - defensive import
            logger.warning("Intervention example resources unavailable: %s", exc)
            return []

        specs = []
        for spec in get_all_intervention_example_specs():
            example_id = str(spec.get("id", "")).strip()
            if not example_id:
                continue
            specs.append(
                {
                    "id": example_id,
                    "type": str(spec.get("type", "intervention")),
                    "path": str(spec.get("path", "")),
                    "title": str(spec.get("title", "")),
                }
            )
        return specs

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
            "Desktop GUI for reaction mechanism investigation, kinetic modeling, and data fitting.\n\n"
            "The initial development of Kindred was led primarily by Pedro Helou de Oliveira.\n"
            "Kindred is now developed and maintained jointly by Pedro Helou de Oliveira and Annabel Flook.\n\n"
            "Certain GUI design elements were informed by ideas seen in Velocity, a kinetics software package developed and maintained by Chen Li.\n\n"
            "Kindred source code is licensed under the MIT License.\n"
            "Includes Qt for Python (PySide6/shiboken6) under LGPLv3/GPL terms.\n"
            "See NOTICE.txt, COPYING, COPYING.LESSER, and THIRD_PARTY_LICENSES.txt.",
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

    def _open_solver_settings(self):
        """Open solver settings dialog."""
        from kindred.gui.widgets.solver_settings import SolverSettingsDialog

        dialog = SolverSettingsDialog(self, cache_port=self._sim_controller)
        solver_contract = load_solver_contract()
        current_parameter_preview_debounce_ms = int(
            self._preview_session.variable_preview_debounce_ms("k1")
        )
        current_equilibrium_preview_debounce_ms = int(
            self._preview_session.variable_preview_debounce_ms("Keq1")
        )
        current_slider_preview_points = int(self._mechanism_editor.slider_points_value())
        current_slider_preview_solver = str(self._mechanism_editor.slider_solver_value() or "BDF")

        # Populate dialog with current settings
        current_settings = {
            "solver": str(self._simulation_runtime_settings_owner.current_solver_name() or solver_contract.default_solver_name),
            'rtol': self._simulation_runtime_settings_owner.current_rtol() or 1e-6,
            'atol': self._simulation_runtime_settings_owner.current_atol() or 1e-12,
            'use_sparse_jacobian': bool(self._simulation_runtime_settings_owner.use_sparse_jacobian()),
            'wegscheider_cyclicity_enabled': bool(
                self._simulation_runtime_settings_owner.wegscheider_cyclicity_enabled()
            ),
            'max_parallel_batch_workers': int(self._sim_controller.parallel_batch.max_parallel_workers),
            'batch_runtime_lane_budget': int(self._sim_controller.batch_runtime_lane_budget),
            'limit_blas_threads_per_worker': bool(self._sim_controller.parallel_batch.limit_blas_threads_per_worker),
            'slider_preview_solver': str(current_slider_preview_solver),
            'slider_preview_points': int(current_slider_preview_points),
            'parameter_preview_debounce_ms': int(current_parameter_preview_debounce_ms),
            'equilibrium_preview_debounce_ms': int(current_equilibrium_preview_debounce_ms),
            'result_cache_cap': int(self._sim_controller.batch_cache.result_cache_max_entries()),
            'preview_cache_cap': int(self._sim_controller.batch_cache.preview_cache_max_entries()),
        }
        dialog.set_settings(current_settings)

        if dialog.exec():
            settings = dialog.get_settings()
            previous_simulation_runtime_settings = self._simulation_runtime_settings_snapshot()
            previous_batch_runtime_pool_settings = (
                self._simulation_batch_runtime_pool_settings_snapshot()
            )
            current_solver = str(self._simulation_runtime_settings_owner.current_solver_name() or solver_contract.default_solver_name)
            current_rtol = float(self._simulation_runtime_settings_owner.current_rtol() or 1e-6)
            current_atol = float(self._simulation_runtime_settings_owner.current_atol() or 1e-12)
            current_sparse = bool(self._simulation_runtime_settings_owner.use_sparse_jacobian())
            current_wegscheider = bool(
                self._simulation_runtime_settings_owner.wegscheider_cyclicity_enabled()
            )
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
            if 'max_parallel_batch_workers' in settings:
                try:
                    self._sim_controller.parallel_batch.max_parallel_workers = min(
                        int(MAX_PARALLEL_WORKERS_CEILING),
                        max(
                            1,
                            int(settings['max_parallel_batch_workers']),
                        ),
                    )
                except Exception:
                    self._sim_controller.parallel_batch.max_parallel_workers = int(
                        PROJECT_DEFAULTS["max_parallel_batch_workers"]
                    )
            if 'batch_runtime_lane_budget' in settings:
                try:
                    self._sim_controller.batch_runtime_lane_budget = min(
                        int(MAX_PARALLEL_WORKERS_CEILING),
                        max(
                            1,
                            int(settings['batch_runtime_lane_budget']),
                        ),
                    )
                except Exception:
                    self._sim_controller.batch_runtime_lane_budget = int(
                        PROJECT_DEFAULTS["batch_runtime_lane_budget"]
                    )
            if 'limit_blas_threads_per_worker' in settings:
                self._sim_controller.parallel_batch.limit_blas_threads_per_worker = bool(
                    settings['limit_blas_threads_per_worker']
                )
            if 'result_cache_cap' in settings or 'preview_cache_cap' in settings:
                cache_caps_result = self._sim_controller.set_simulation_cache_caps(
                    result_cap=int(
                        settings.get(
                            'result_cache_cap',
                            self._sim_controller.batch_cache.result_cache_max_entries(),
                        )
                    ),
                    preview_cap=int(
                        settings.get(
                            'preview_cache_cap',
                            self._sim_controller.batch_cache.preview_cache_max_entries(),
                        )
                    ),
                    persist=True,
                )
                if not cache_caps_result.ok:
                    self._simulation_dialogs.message_box_warning(
                        "Simulation Cache",
                        cache_caps_result.message or "Failed to update simulation cache settings.",
                    )
            self._settings_owner.settings_set_value(
                "simulation/slider_preview_solver",
                str(next_runtime_settings["slider_preview_solver"]),
            )
            self._settings_owner.settings_set_value(
                "simulation/slider_preview_points",
                int(next_runtime_settings["slider_preview_points"]),
            )
            self._settings_owner.settings_set_value(
                "simulation/parameter_preview_debounce_ms",
                int(next_runtime_settings["parameter_preview_debounce_ms"]),
            )
            self._settings_owner.settings_set_value(
                "simulation/equilibrium_preview_debounce_ms",
                int(next_runtime_settings["equilibrium_preview_debounce_ms"]),
            )
            self._mechanism_editor.set_slider_solver_value(str(next_runtime_settings["slider_preview_solver"]))
            self._mechanism_editor.set_slider_points_value(int(next_runtime_settings["slider_preview_points"]))
            logger.info(f"Solver settings updated: {settings}")
            self._apply_runtime_settings(
                solver=settings.get('solver', self._simulation_runtime_settings_owner.current_solver_name() or solver_contract.default_solver_name),
                rtol=settings.get('rtol', self._simulation_runtime_settings_owner.current_rtol() or 1e-6),
                atol=settings.get('atol', self._simulation_runtime_settings_owner.current_atol() or 1e-12),
                use_sparse_jacobian=settings.get('use_sparse_jacobian', current_sparse),
                wegscheider_cyclicity_enabled=settings.get(
                    'wegscheider_cyclicity_enabled',
                    current_wegscheider,
                ),
            )
            # Record dialog changes as user preferences.
            self.config_controller.update_user_preference("solver", self._simulation_runtime_settings_owner.current_solver_name())
            self.config_controller.update_user_preference("rtol", self._simulation_runtime_settings_owner.current_rtol())
            self.config_controller.update_user_preference("atol", self._simulation_runtime_settings_owner.current_atol())
            self.config_controller.update_user_preference(
                "use_sparse_jacobian",
                self._simulation_runtime_settings_owner.use_sparse_jacobian(),
            )
            self.config_controller.update_user_preference(
                "wegscheider_cyclicity_enabled",
                self._simulation_runtime_settings_owner.wegscheider_cyclicity_enabled(),
            )
            self.config_controller.update_user_preference(
                "max_parallel_batch_workers",
                self._sim_controller.parallel_batch.max_parallel_workers,
            )
            self.config_controller.update_user_preference(
                "batch_runtime_lane_budget",
                self._sim_controller.batch_runtime_lane_budget,
            )
            self.config_controller.update_user_preference(
                "limit_blas_threads_per_worker",
                self._sim_controller.parallel_batch.limit_blas_threads_per_worker,
            )
            self._fitting_runtime_input_publisher.publish_if_changed(reason="solver settings applied")
            slider_schema_refresh_needed = bool(
                current_runtime_settings["wegscheider_cyclicity_enabled"]
                != next_runtime_settings["wegscheider_cyclicity_enabled"]
            )
            if slider_schema_refresh_needed:
                try:
                    self._variable_runtime.extract_and_populate_variables(
                        preserve_visibility=True,
                    )
                    self._sync_mechanism_controls_to_focused_batch_set()
                except Exception:
                    logger.exception("Failed to refresh variables after solver settings update")
                    bar = getattr(self, "_status_bar", None)
                    if bar is not None:
                        try:
                            bar.showMessage("Failed to refresh variables after solver settings update (see logs)", 8000)
                        except RuntimeError as exc:
                            logger.debug("Failed to show solver settings refresh error in status bar: %s", exc, exc_info=True)
                            self._status_bar = None
            if self._simulation_runtime_settings_snapshot() != previous_simulation_runtime_settings:
                batch_runtime_pool_settings_changed = (
                    self._simulation_batch_runtime_pool_settings_snapshot()
                    != previous_batch_runtime_pool_settings
                )
                self._refresh_simulation_runtime_after_inputs_changed(
                    batch_runtime_pool_inputs_changed=bool(batch_runtime_pool_settings_changed)
                )

    def closeEvent(self, event):
        """
        Handle window close event - save settings and clean up worker threads.

        P3 ENHANCEMENT: Save user preferences before closing.
        """
        logger.info("Window close event - cleaning up")

        # P3 ENHANCEMENT: Save user preferences
        self._save_settings()

        self._sim_controller.prepare_simulation_shutdown_for_close()

        fit_windows_ready = self._prepare_fit_window_shutdown_for_close()
        if not fit_windows_ready:
            try:
                self._sim_controller.abort_simulation_shutdown_for_close()
            except Exception:
                logger.debug("Failed to abort simulation shutdown after deferred close", exc_info=True)
            logger.warning("Deferring close event while fit window shutdown is still in progress")
            event.ignore()
            return

        logger.info("Cleanup complete, accepting close event")
        event.accept()

    def showEvent(self, event):
        super().showEvent(event)

    def _prepare_fit_window_shutdown_for_close(self) -> bool:
        tracked_windows = list(self._fitting_runtime_input_publisher.active_windows())
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
                is_visible = getattr(window, "isVisible", None)
                if callable(is_visible) and bool(is_visible()):
                    all_closed = False
            except RuntimeError:
                continue

        return all_closed

    # ========================================================================
    # PROFILE MANAGEMENT
    # ========================================================================
