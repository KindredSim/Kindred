"""Kindred's single-window shell and direct application graph wiring."""

from __future__ import annotations

import logging
import io
import os
from dataclasses import dataclass, replace
from typing import Callable, Iterable

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt

from kindred._version import __version__ as KINDRED_VERSION
from kindred.gui.display_name_policy import compact_text, literal_tooltip
from kindred.core.authored_source_errors import AuthoredSourceError
from kindred.core.dataset_mapping import (
    DatasetMappingError,
    DatasetMappingChoice,
    OccupiedSetChoice,
    DatasetMappingBatchProposal,
    DatasetMappingProposal,
    apply_dataset_mapping_batch,
    export_original_source,
    plan_global_dataset_delete,
    validate_unique_dataset_name,
    with_confirmed_observation_mapping,
)
from kindred.gui.widgets.observation_mapping_widget import ObservationMappingDialog
from kindred.core.mechanism_source import MechanismAuthoringSource
from kindred.core.fitting_method_policy import SUPPORTED_FIT_METHODS
from kindred.core.project_document import (
    ExactCorrectionAssessment,
    FIT_MAX_NFEV_RANGE,
    FIT_SEED_RANGE,
    FitDefaults,
    MechanismTabDocument,
    MechanismSourceCollision,
    MechanismSourceReconciliation,
    ProjectDocument,
    SimulationExecutionSettings,
    canonical_species_names,
    canonical_observation_target_names,
    canonical_initial_values,
    prepare_authored_source_admission,
    assess_tab_exact_correction,
    reconcile_tab_mechanism_source,
    validate_project_document,
    validate_fit_defaults,
)
from kindred.core.project_session import (
    ProjectContentToken,
    ProjectSession,
)
from kindred.core.project_presentation_state import (
    AxisSeriesSelection,
    ExperimentalAxisSelection,
    ProjectPresentationState,
    SimulationSeriesAxisSelection,
)
from kindred.core.result_repository import (
    ResultRepository,
)
from kindred.core.species import concentration_from_text
from kindred.gui.computation_gate import ComputationGate
from kindred.gui.app_wiring import (
    ANALYSIS_DOCK_SPEC,
    BATCH_DOCK_SPEC,
    build_physical_shell,
    DOCK_SPECS,
    MECHANISM_DOCK_SPEC,
    RIGHT_DOCK_SPEC,
    SLIDERS_DOCK_SPEC,
    SYMBOLIC_CALCULATOR_DOCK_SPEC,
)
from kindred.gui.controllers.config_controller import (
    ConfigController,
)
from kindred.gui.controllers.project_controller import (
    ProjectController,
    ProjectTransientState,
)
from kindred.gui.controllers.simulation_job_owner import (
    ExplicitRunCapture,
    SimulationJobOwner,
    build_explicit_target,
)
from kindred.gui.dataset_import_staging import (
    DatasetImportCompletion,
    DatasetImportStaging,
    build_raw_record_batch_proposal,
)
from kindred.gui.xlsx_export_owner import XlsxExportOwner
from kindred.gui.import_workflow import ImportConfig
from kindred.gui.dataset_payload_snapshot import (
    build_data_manager_snapshot,
    build_experimental_layer_inputs,
    experimental_axis_label,
)
from kindred.gui.fit_session_owner import (
    FitSessionOwner,
)
from kindred.gui.fitting.constants import (
    DE_CONVERGENCE_TOLERANCE_LABEL,
    DIFFERENTIAL_EVOLUTION_METHOD,
    FUNCTION_TOLERANCE_LABEL,
    STEP_TOLERANCE_LABEL,
    admit_fit_tolerance,
    fit_method_label,
)
from kindred.gui.main_window_variable_runtime import TabSliderBinding
from kindred.gui.symbolic_calculator_owner import SymbolicCalculatorOwner
from kindred.gui.theme_manager import ThemeManager
from kindred.gui.widgets.batch_initial_conditions_table import (
    BatchInitialConditionsTableModel,
    BatchInitialConditionsTableView,
    InitialConditionsTableActions,
    is_valid_initial_text,
)
from kindred.gui.widgets.data_manager import (
    DataManagerPanel,
)
from kindred.gui.widgets.dataset_mapping_batch_dialog import DatasetMappingBatchDialog
from kindred.gui.widgets.dataset_overlay_panel import (
    DatasetOverlayPanel,
)
from kindred.gui.widgets.editor_panel import EditorPanel
from kindred.gui.widgets.exact_correction_basis_dialog import (
    ExactCorrectionBasisDialog,
)
from kindred.gui.widgets.mechanism_editor import (
    MechanismEditorBinding,
    MechanismEditorTabbed,
)
from kindred.gui.widgets.mechanism_workspace import ProjectWorkspaceWidget
from kindred.gui.widgets.pyqtgraph_plot_panel_impl import PyQtGraphPlotPanel
from kindred.gui.widgets.right_panel import RightPanelTabbed
from kindred.gui.widgets.simulation_panel import SimulationPanel
from kindred.gui.widgets.solver_settings import SimulationSettingsDialog
from kindred.gui.widgets.symbolic_calculator_panel import SymbolicCalculatorPanel
from kindred.gui.ui_helpers import clamp_initial_size, ignore_wheel
from kindred.gui.workspace_result_presenter import (
    ResultRenderInput,
    VisibleResultSet,
    compose_canonical_export_blocks,
    compose_result_input,
    iter_blocks_tsv_chunks,
)
from kindred.gui.workspace_tab_registry import (
    ScientificDocumentDelta,
    WorkspaceTabRegistry,
)
from kindred.io.kin_archive_store import KinArchiveStore
from kindred.io.logging import diagnostic_log_path
from kindred.io.paths import file_error_text, resolve_export_start_dir
from kindred.io.resources import (
    PRESET_MECHANISM_TITLES,
    InterventionExampleCatalogError,
    get_all_intervention_example_specs,
)

logger = logging.getLogger(__name__)

__all__ = ["MainWindow"]

_ABOUT_DIALOG_PREFERRED_WIDTH = 520
_ABOUT_DIALOG_IMAGE_MAX_SIZE = 320


@dataclass(slots=True)
class _TabSurfaceBundle:
    """MainWindow-owned retained non-plot physical state for one tab."""

    tab_id: str
    editor_panel: EditorPanel
    editor: MechanismEditorTabbed
    slider_panel: QtWidgets.QWidget
    batch_model: BatchInitialConditionsTableModel
    simulation_panel: SimulationPanel
    batch_table: BatchInitialConditionsTableView
    data_panel: DataManagerPanel
    overlay_panel: DatasetOverlayPanel
    right_panel: RightPanelTabbed
    calculator_owner: SymbolicCalculatorOwner
    calculator_panel: SymbolicCalculatorPanel
    editor_binding: MechanismEditorBinding
    initial_actions: InitialConditionsTableActions
    slider_binding: TabSliderBinding


class MainWindow(QtWidgets.QMainWindow):
    """One physical shell over one explicit, retained semantic owner graph."""

    def __init__(self) -> None:
        super().__init__()
        application = QtWidgets.QApplication.instance()
        if not isinstance(application, QtWidgets.QApplication):
            raise RuntimeError("MainWindow requires an existing QApplication.")

        self._rendering_workspace = False
        self._recent_menu: QtWidgets.QMenu | None = None
        self._tab_surfaces: dict[str, _TabSurfaceBundle] = {}
        self._dark_mode_action: QtGui.QAction | None = None
        self._run_action: QtGui.QAction | None = None
        self._scan_action: QtGui.QAction | None = None
        self._new_project_action: QtGui.QAction | None = None
        self._open_project_action: QtGui.QAction | None = None
        self._load_data_action: QtGui.QAction | None = None
        self._xlsx_export_action: QtGui.QAction | None = None
        self._diagnostic_log_dialog: QtWidgets.QDialog | None = None
        self._dataset_import_progress_dialog: QtWidgets.QProgressDialog | None = None
        self._last_export_directory: str | None = None
        self._slider_apply_refreshing: set[str] = set()
        self._plot_membership_refreshing: set[str] = set()
        self._current_row_refreshing: set[str] = set()
        self.setObjectName("kindredMainWindow")
        self.setWindowTitle("Kindred")
        startup_size = self._startup_size()
        clamp_initial_size(self, startup_size.width(), startup_size.height())
        self.setDockNestingEnabled(True)

        self._workspace = ProjectWorkspaceWidget(self)
        self.setCentralWidget(self._workspace)

        self._status_label = QtWidgets.QLabel("Ready", self)
        self.statusBar().addWidget(self._status_label, 1)

        self._theme_manager = ThemeManager(
            self,
            report_failure=self.set_status_text,
        )
        self.config_controller = ConfigController(QtCore.QSettings())

        initial_document = self.config_controller.default_project_document()
        self._project_session = ProjectSession(initial_document)
        self._computation_gate = ComputationGate(parent=self)
        self._calculator_owners: set[SymbolicCalculatorOwner] = set()
        self._spreadsheet_editor = None

        self._simulation_jobs = SimulationJobOwner(
            gate=self._computation_gate,
            repository_for_tab=self._repository_for_tab,
            user_parallel_maximum=(
                self.config_controller.run_user_maximum_preference
            ),
            parent=self,
        )
        self._workspace_tab_registry = WorkspaceTabRegistry(
            self._project_session,
            computation_fence=self._simulation_jobs.fence_document_delta,
            retire_interactive_plot_targets=(
                self._retire_interactive_plot_targets
            ),
            post_document_publication=self._post_document_publication,
            runtime_factory=self._new_tab_runtime,
        )
        self._dataset_import_staging = DatasetImportStaging(parent=self)
        self._fit_session_owner = FitSessionOwner(
            session=self._project_session,
            registry=self._workspace_tab_registry,
            gate=self._computation_gate,
            color_resolver=self.config_controller.species_color,
            observation_style_getter=lambda: (
                self.config_controller.experimental_point_shape(),
                self.config_controller.experimental_point_size(),
                self.config_controller.experimental_point_opacity(),
            ),
            parent_widget=self,
            parent=self,
        )

        self._surface_stacks = build_physical_shell(self._workspace)
        self._set_current_workspace_references()

        for tab in self._project_session.document.tabs:
            self._create_tab_surface(tab.tab_id)
        self._xlsx_export_owner = XlsxExportOwner(parent=self)
        from kindred.gui.chemical_space_scan_owner import ChemicalSpaceScanOwner

        self._scan_owner = ChemicalSpaceScanOwner(
            self._computation_gate, self._xlsx_export_owner, self._project_session, self
        )
        self._scan_owner.application_close_settlement_changed.connect(self._refresh_actions)
        self._scan_owner.status.connect(self.set_status_text)
        self._xlsx_export_owner.succeeded.connect(
            self._on_xlsx_export_succeeded
        )
        self._xlsx_export_owner.failed.connect(
            self._on_xlsx_export_failed
        )
        self._xlsx_export_owner.cancelled.connect(
            self._on_xlsx_export_cancelled
        )
        self._xlsx_export_owner.application_close_settlement_changed.connect(
            self._refresh_actions
        )

        self._archive_store = KinArchiveStore()
        self.project_controller = ProjectController(
            self,
            session=self._project_session,
            tabs=self._workspace_tab_registry,
            archive_store=self._archive_store,
            config=self.config_controller,
            admit_incoming_document=self._admit_incoming_document,
            simulation_jobs=self._simulation_jobs,
            dataset_imports=self._dataset_import_staging,
            fit_session=self._fit_session_owner,
            xlsx_exports=self._xlsx_export_owner,
            scan=self._scan_owner,
            calculator_owners=lambda: tuple(self._calculator_owners),
            spreadsheet_editor=lambda: self._spreadsheet_editor,
            set_status_text=self.set_status_text,
            capture_visible_tab_inputs=self._capture_visible_tab_inputs,
            prepare_before_save=self._prepare_before_save,
            commit_saved_document=self._commit_saved_document,
            inspect_transient_state=self._inspect_transient_state,
            discard_authoring=self._discard_authoring,
            discard_slider_drafts=self._discard_all_slider_drafts,
        )
        self.project_controller.projectStateChanged.connect(self._scan_owner.project_state_changed)

        from kindred.config.templates import TemplateManager

        self._template_manager = TemplateManager()
        try:
            self._template_manager.load_templates()
        except Exception as exc:
            logger.exception("Template collection could not be loaded")
            QtWidgets.QMessageBox.warning(
                self,
                "Template Collection",
                "The template collection could not be loaded. The stored "
                "collection was retained and template changes are disabled, "
                f"but Kindred can continue.\n\n{file_error_text(exc, self._template_manager.templates_directory)}",
            )

        self._create_menus()
        self._connect_application_graph()
        self._load_settings()
        self._refresh_workspace_document()
        from kindred.gui.tutorials import create_controller

        self._tutorials = create_controller(
            self, workspace=self._workspace, registry=self._workspace_tab_registry,
            surfaces=self._tab_surfaces, examples_menu=self._examples_menu, file_menu=self._file_menu,
            project_controller=self.project_controller, jobs=self._simulation_jobs,
        )
        logger.info("Kindred MainWindow initialized with the direct owner graph")

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def _new_tab_runtime(self, tab_id: str):
        from kindred.gui.workspace_tab_registry import TabRuntimeState

        return TabRuntimeState(
            result_repository=ResultRepository(
                tab_id,
                cache_capacity=self.config_controller.result_cache_capacity(),
            )
        )

    def set_theme(self, *, dark_mode: bool) -> None:
        """Forward the application theme to the current physical root."""

        self._workspace.set_theme(dark_mode=bool(dark_mode))

    def _set_current_workspace_references(self) -> None:
        """Resolve the current root's direct stacks and dock ownership."""

        stacks = self._surface_stacks
        self._mechanism_stack = stacks[MECHANISM_DOCK_SPEC.identity_key]
        self._slider_stack = stacks[SLIDERS_DOCK_SPEC.identity_key]
        self._initial_conditions_stack = stacks[
            BATCH_DOCK_SPEC.identity_key
        ]
        self._data_stack = stacks[RIGHT_DOCK_SPEC.identity_key]
        self._calculator_stack = stacks[
            SYMBOLIC_CALCULATOR_DOCK_SPEC.identity_key
        ]
        self._mechanism_dock = self._workspace.dock_for_identity(
            MECHANISM_DOCK_SPEC.identity_key
        )
        self._sliders_dock = self._workspace.dock_for_identity(
            SLIDERS_DOCK_SPEC.identity_key
        )
        self._batch_dock = self._workspace.dock_for_identity(
            BATCH_DOCK_SPEC.identity_key
        )
        self._right_dock = self._workspace.dock_for_identity(
            RIGHT_DOCK_SPEC.identity_key
        )
        self._analysis_dock = self._workspace.dock_for_identity(
            ANALYSIS_DOCK_SPEC.identity_key
        )
        self._symbolic_calculator_dock = self._workspace.dock_for_identity(
            SYMBOLIC_CALCULATOR_DOCK_SPEC.identity_key
        )

    def _create_tab_surface(
        self,
        tab_id: str,
    ) -> _TabSurfaceBundle:
        """Construct one complete tab surface beneath the permanent root."""

        surface_registry = self._workspace_tab_registry
        target = str(tab_id).strip()
        if not target:
            raise ValueError("tab_id must be nonblank.")
        if target in self._tab_surfaces:
            raise ValueError(f"Tab surface already exists: {target!r}.")
        editor_panel = EditorPanel(parent=self._mechanism_stack)
        editor = editor_panel.editor
        slider_panel = editor.slider_pane_for_dock()
        slider_panel.setParent(self._slider_stack)
        batch_model = BatchInitialConditionsTableModel(parent=self._workspace)
        simulation_panel = SimulationPanel(
            batch_model=batch_model,
            parent=self._initial_conditions_stack,
        )
        batch_model.setParent(simulation_panel)
        batch_table = simulation_panel.batch_table
        data_panel = DataManagerPanel(
            origin_tab_id=target,
            parent=self._data_stack,
        )
        data_panel.set_import_busy(self._dataset_import_staging.busy)
        data_panel.set_creation_busy(self._spreadsheet_editor is not None)
        overlay_panel = DatasetOverlayPanel(
            self._workspace,
            import_action=data_panel.import_action,
            color_for_species=self.config_controller.species_color,
            experimental_point_shape=self.config_controller.experimental_point_shape,
            experimental_point_size=self.config_controller.experimental_point_size,
            experimental_point_opacity=(
                self.config_controller.experimental_point_opacity
            ),
        )
        right_panel = RightPanelTabbed(
            self._data_stack,
            data_panel=data_panel,
            overlay_panel=overlay_panel,
        )
        calculator_owner = SymbolicCalculatorOwner(
            canonical_prepared_dsl=lambda captured=target: (
                self._workspace_tab_registry.tab(captured).prepared_dsl
            ),
            canonical_source=lambda captured=target: (
                self._workspace_tab_registry.tab(captured).mechanism_source
            ),
            gate=self._computation_gate,
            parent=self,
        )
        self._calculator_owners.add(calculator_owner)
        calculator_owner.disposed.connect(self._calculator_owner_disposed)
        calculator_owner.application_close_settlement_changed.connect(self._calculator_settlement_changed)
        calculator_panel = SymbolicCalculatorPanel(
            on_evaluate=lambda query, captured=target: self._evaluate_symbolic(
                captured,
                query,
            ),
            on_copy_result=calculator_owner.copy_compact_text,
            on_copy_context=calculator_owner.copy_context_text,
            on_use_result=calculator_owner.reuse_query,
            on_cancel=calculator_owner.cancel,
            parent=self._calculator_stack,
        )
        calculator_owner.stateChanged.connect(calculator_panel.render_state)
        editor_binding = MechanismEditorBinding(
            tab_id=target,
            editor=editor,
            registry=surface_registry,
            parent=editor_panel,
        )
        initial_actions = InitialConditionsTableActions(
            tab_id=target,
            registry=surface_registry,
            model=batch_model,
            table=batch_table,
            parent=simulation_panel,
        )
        slider_binding = TabSliderBinding(
            tab_id=target,
            registry=surface_registry,
            table=batch_table,
            variable_sliders=editor.variable_sliders_widget(),
            species_sliders=editor.species_sliders_widget(),
            slider_workspace_owner=editor,
            simulation_jobs=self._simulation_jobs,
            preview_points=self.config_controller.preview_points(),
            preview_delay_ms=self.config_controller.preview_delay_ms(),
            parent=editor_panel,
        )
        bundle = _TabSurfaceBundle(
            tab_id=target,
            editor_panel=editor_panel,
            editor=editor,
            slider_panel=slider_panel,
            batch_model=batch_model,
            simulation_panel=simulation_panel,
            batch_table=batch_table,
            data_panel=data_panel,
            overlay_panel=overlay_panel,
            right_panel=right_panel,
            calculator_owner=calculator_owner,
            calculator_panel=calculator_panel,
            editor_binding=editor_binding,
            initial_actions=initial_actions,
            slider_binding=slider_binding,
        )
        self._tab_surfaces[target] = bundle
        for stack, widget in (
            (self._mechanism_stack, editor_panel),
            (self._slider_stack, slider_panel),
            (self._initial_conditions_stack, simulation_panel),
            (self._data_stack, right_panel),
            (self._calculator_stack, calculator_panel),
        ):
            stack.addWidget(widget)
        self._connect_tab_surface(
            bundle,
        )
        return bundle

    def _connect_tab_surface(
        self,
        bundle: _TabSurfaceBundle,
    ) -> None:
        """Connect one page only to callbacks closed over its immutable tab ID."""

        tab_id = bundle.tab_id
        panel = bundle.simulation_panel
        model = bundle.batch_model
        editor = bundle.editor
        actions = bundle.initial_actions
        sliders = bundle.slider_binding

        panel.addSetRequested.connect(lambda captured=tab_id: self._add_initial_set(captured))
        panel.moveSelectedRequested.connect(
            lambda delta, captured=tab_id: self._move_initial_sets(captured, delta)
        )
        panel.deleteSelectedRequested.connect(
            lambda captured=tab_id: self._delete_initial_sets(captured)
        )
        panel.runSelectedRequested.connect(
            lambda captured=tab_id: self._start_selected_run(captured)
        )
        panel.stopRequested.connect(
            lambda captured=tab_id: self._stop_simulation(captured)
        )
        panel.workspaceRunDefaultsRequested.connect(
            lambda settings, captured=tab_id: self._set_workspace_run_defaults(
                captured,
                settings,
            )
        )
        panel.runDefaultsDraftStateChanged.connect(
            lambda _dirty, captured=tab_id: self._on_initial_draft_state_changed(
                captured
            )
        )
        panel.settingsValidationFailed.connect(self.set_status_text)
        model.plotMembershipChangeRequested.connect(
            lambda set_id, requested, captured=tab_id: (
                self._on_plot_membership_requested(
                    captured,
                    set_id,
                    requested,
                )
            )
        )
        model.sliderPinChangeRequested.connect(
            lambda set_id, pinned, captured=bundle: (
                self._on_slider_pin_change_requested(
                    captured,
                    set_id,
                    pinned,
                )
            )
        )

        editor.run_btn.clicked.connect(
            lambda _checked=False, captured=tab_id: self._start_selected_run(
                captured
            )
        )
        editor.sliderApplyRequested.connect(
            lambda captured=tab_id: self._apply_slider_draft(captured)
        )
        editor.sliderResetRequested.connect(
            lambda captured=tab_id: self._reset_slider_draft(captured)
        )
        sliders.sliderDraftStateChanged.connect(
            self._on_slider_draft_state_changed
        )
        editor.symbolic_calculator_btn.clicked.connect(
            lambda _checked=False, captured=tab_id: self._toggle_symbolic_calculator(
                captured
            )
        )
        editor.sliders_btn.clicked.connect(
            lambda _checked=False, captured=tab_id: self._toggle_interactive_sliders(
                captured
            )
        )
        bundle.editor_binding.draftStateChanged.connect(
            self._on_draft_state_changed
        )
        editor.nativeHistoryAvailabilityChanged.connect(
            lambda _undo, _redo: self._refresh_actions()
        )
        sliders.draftStateChanged.connect(self._on_draft_state_changed)
        sliders.errorRaised.connect(self.set_status_text)
        sliders.targetSetIdsChanged.connect(
            lambda ids, captured=tab_id: self._refresh_slider_target_summary(
                captured,
                ids,
            )
        )
        actions.errorRaised.connect(self.set_status_text)
        actions.setNameChanged.connect(self._on_initial_set_name_changed)
        actions.selectedRunSetIdsChanged.connect(
            lambda _ids, captured=tab_id: self._on_selected_run_sets_changed(
                captured
            )
        )
        actions.draftStateChanged.connect(
            lambda _captured, captured=tab_id: self._on_initial_draft_state_changed(
                captured
            )
        )
        actions.initialValuesChanged.connect(self._on_initial_values_changed)
        bundle.batch_table.selectionModel().currentChanged.connect(
            lambda _current, previous, captured=tab_id: self._on_current_row_changed(
                captured,
                previous,
            )
        )
        data = bundle.data_panel
        data.datasetImportRequested.connect(self._start_dataset_import)
        data.datasetCreationRequested.connect(self._open_spreadsheet)
        data.datasetRenameRequested.connect(self._rename_dataset)
        data.datasetDeleteRequested.connect(self._delete_dataset)
        data.datasetExportOriginalRequested.connect(self._export_original_dataset)
        data.datasetMappingRequested.connect(
            lambda dataset_id, action, set_id, occupied_choice, captured=tab_id: self._map_dataset(
                captured,
                dataset_id,
                action,
                set_id,
                occupied_choice,
            )
        )
        data.datasetObservationMappingRequested.connect(
            lambda dataset_id, captured=tab_id: self._edit_observation_mapping(
                captured,
                dataset_id,
            )
        )
        bundle.overlay_panel.overlaySelectionRequested.connect(
            lambda dataset_id, raw_observation_ids, captured=tab_id: self._set_overlay_selection(
                captured,
                dataset_id,
                raw_observation_ids,
            )
        )
        bundle.overlay_panel.speciesColorRequested.connect(
            self._set_species_color
        )
        bundle.overlay_panel.experimentalPointShapeRequested.connect(
            self._set_experimental_point_shape
        )
        bundle.overlay_panel.experimentalPointSizeRequested.connect(
            self._set_experimental_point_size
        )
        bundle.overlay_panel.experimentalPointOpacityRequested.connect(
            self._set_experimental_point_opacity
        )

    def _drop_tab_surface(
        self,
        tab_id: str,
    ) -> None:
        target = str(tab_id)
        bundle = self._tab_surfaces.pop(target, None)
        if bundle is None:
            return
        if self._load_data_action is bundle.data_panel.import_action:
            self._file_menu.removeAction(self._load_data_action)
            self._load_data_action = None
        bundle.slider_binding.dispose()
        bundle.calculator_owner.dispose()
        for stack, widget in (
            (self._mechanism_stack, bundle.editor_panel),
            (self._slider_stack, bundle.slider_panel),
            (self._initial_conditions_stack, bundle.simulation_panel),
            (self._data_stack, bundle.right_panel),
            (self._calculator_stack, bundle.calculator_panel),
        ):
            stack.removeWidget(widget)
            widget.deleteLater()

    def _set_species_color(
        self,
        species: object,
        color: object,
        known_species: object,
    ) -> None:
        if self._mutation_input_admitted("Change Species Color"):
            try:
                self.config_controller.set_species_color(
                    str(species),
                    color,
                    tuple(known_species or ()),
                )
            except OSError as exc:
                self.set_status_text(str(exc))

    def _set_experimental_point_shape(self, shape: str) -> None:
        if self._mutation_input_admitted("Change Experimental Point Shape"):
            try:
                self.config_controller.set_experimental_point_shape(shape)
            except OSError as exc:
                self.set_status_text(str(exc))
                self._refresh_application_plot_preferences()

    def _set_experimental_point_size(self, request: object) -> None:
        if self._mutation_input_admitted("Change Experimental Point Size"):
            try:
                self.config_controller.set_experimental_point_size(request)
            except OSError as exc:
                self.set_status_text(str(exc))
                self._refresh_application_plot_preferences()

    def _set_experimental_point_opacity(self, request: object) -> None:
        if self._mutation_input_admitted("Change Experimental Point Opacity"):
            try:
                self.config_controller.set_experimental_point_opacity(request)
            except OSError as exc:
                self.set_status_text(str(exc))
                self._refresh_application_plot_preferences()

    def _selected_bundle(self) -> _TabSurfaceBundle:
        return self._tab_surfaces[self._workspace_tab_registry.active_tab_id]

    def _connect_workspace_root(self) -> None:
        """Connect the permanent root's direct public signals once."""

        self._workspace.tabActivated.connect(self._select_tab)
        self._workspace.createTabRequested.connect(self._create_tab)
        self._workspace.tabMenuOpening.connect(self._finish_focused_tab_input)
        self._workspace.copyTabRequested.connect(self._copy_tab)
        self._workspace.reorderTabsRequested.connect(self._reorder_tabs)
        self._workspace.renameTabRequested.connect(self._rename_tab)
        self._workspace.closeTabRequested.connect(self._close_tab)
        self._workspace.plotCreated.connect(self._connect_workspace_plot)

    def _connect_application_graph(self) -> None:
        self.project_controller.projectStateChanged.connect(
            self._on_project_state_changed
        )
        self.project_controller.closeStarted.connect(self._on_close_started)
        self.project_controller.nativeCloseAuthorized.connect(
            self.authorize_and_schedule_close
        )

        self._connect_workspace_root()

        application = QtWidgets.QApplication.instance()
        if isinstance(application, QtWidgets.QApplication):
            application.focusChanged.connect(self._refresh_actions)
        self._simulation_jobs.state_changed.connect(
            self._on_simulation_state_changed
        )
        self._simulation_jobs.run_dispatched.connect(
            self._on_run_dispatched
        )
        self._simulation_jobs.results_changed.connect(self._on_results_changed)
        self._simulation_jobs.interactive_result_ready.connect(
            self._on_interactive_result_ready
        )
        self._simulation_jobs.interactive_target_failed.connect(
            self._on_interactive_target_failed
        )
        self._simulation_jobs.status_changed.connect(self.set_status_text)
        self._fit_session_owner.window_changed.connect(
            self._on_fit_window_changed
        )
        self._fit_session_owner.execution_changed.connect(
            self._on_fit_execution_changed
        )
        self._fit_session_owner.project_document_changed.connect(
            self._on_fit_project_document_changed
        )
        self._fit_session_owner.status_changed.connect(self.set_status_text)
        self._dataset_import_staging.completionAvailable.connect(
            self._commit_import_completion
        )
        self._dataset_import_staging.busyChanged.connect(
            self._on_import_busy_changed
        )
        self._dataset_import_staging.progressed.connect(
            self._on_import_progress
        )
        self._dataset_import_staging.failed.connect(
            self._on_import_failure
        )
        self._dataset_import_staging.cancelled.connect(
            self._on_import_cancellation
        )
        self.config_controller.applicationPlotPreferencesChanged.connect(
            self._refresh_application_plot_preferences
        )
        self.config_controller.previewSettingsChanged.connect(
            self._apply_slider_preview_settings
        )
        self.config_controller.resultCacheCapacityChanged.connect(
            self._resize_result_repositories
        )
        self.config_controller.recentProjectsChanged.connect(
            self._populate_recent_menu
        )

    def _mutation_input_admitted(self, action: str | None = None) -> bool:
        """Return whether user mutations may still enter the live shell."""

        controller = getattr(self, "project_controller", None)
        admitted = bool(controller is None or controller.mutation_admitted)
        if not admitted and action:
            self.set_status_text(
                f"{action} rejected: application Close is in progress."
            )
        return admitted

    @staticmethod
    def _menu_actions(menu: QtWidgets.QMenu) -> tuple[QtGui.QAction, ...]:
        actions: list[QtGui.QAction] = []
        for action in menu.actions():
            actions.append(action)
            child = action.menu()
            if child is not None:
                actions.extend(MainWindow._menu_actions(child))
        return tuple(actions)

    def _fence_shell_for_close(self) -> None:
        """Physically disable writable shell surfaces; Close is one-way."""

        for widget in (
            self._workspace,
            self._mechanism_dock,
            self._sliders_dock,
            self._batch_dock,
            self._right_dock,
            self._analysis_dock,
            self._symbolic_calculator_dock,
        ):
            widget.setEnabled(False)
        for menu in (
            self._file_menu,
            self._edit_menu,
            self._view_menu,
            self._examples_menu,
            self._simulation_menu,
            self._fitting_menu,
        ):
            menu.setEnabled(False)
            for action in self._menu_actions(menu):
                action.setEnabled(False)

        self._refresh_actions()

    # ------------------------------------------------------------------
    # Direct render
    # ------------------------------------------------------------------
    def _select_tab_surfaces(self, tab_id: str) -> None:
        """Raise retained pages; selecting a tab never reconstructs them."""

        target = str(tab_id)
        bundle = self._tab_surfaces[target]
        self._mechanism_stack.setCurrentWidget(bundle.editor_panel)
        self._slider_stack.setCurrentWidget(bundle.slider_panel)
        self._initial_conditions_stack.setCurrentWidget(bundle.simulation_panel)
        self._data_stack.setCurrentWidget(bundle.right_panel)
        self._calculator_stack.setCurrentWidget(bundle.calculator_panel)
        self._workspace.select_tab_surface(target)
        self._refresh_title()
        self._refresh_actions()

    def _refresh_workspace_tab_structure(self) -> None:
        """Rerender only the tab strip and retained active-page selection."""

        self._workspace.render(
            self._project_session.document,
            active_tab_id=self._workspace_tab_registry.active_tab_id,
        )
        self._render_tab_state()
        self._select_tab_surfaces(self._workspace_tab_registry.active_tab_id)

    @QtCore.Slot(str, object)
    def _connect_workspace_plot(
        self,
        tab_id: str,
        plot: PyQtGraphPlotPanel,
    ) -> None:
        """Bind one retained plot only to intents from its owning tab."""

        target = str(tab_id)
        plot.referenceLayerVisibilityRequested.connect(
            lambda visible: self._set_reference_lines_visible(target, visible)
        )
        plot.interventionAnnotationsVisibilityRequested.connect(
            lambda visible: self._set_intervention_annotations_visible(
                target,
                visible,
            )
        )
        plot.interventionLabelsVisibilityRequested.connect(
            lambda visible: self._set_intervention_labels_visible(target, visible)
        )
        plot.copyAllRequested.connect(
            lambda captured=target: self._copy_all_for_tab(captured)
        )
        plot.xSeriesRequested.connect(
            lambda selection: self._set_x_series(target, selection)
        )
        plot.ySeriesRequested.connect(
            lambda selections: self._set_y_series(target, selections)
        )
        plot.plotPresentationIntentRequested.connect(
            lambda option, enabled: self._set_plot_presentation(
                target,
                option,
                enabled,
            )
        )
        plot.resultDiagnostic.connect(self._on_workspace_result_diagnostic)

    def _refresh_workspace_document(self, project_replaced: bool = False) -> None:
        """Refresh canonical/global projections without changing selected page state."""

        if self._rendering_workspace:
            return
        self._rendering_workspace = True
        try:
            document = self._project_session.document
            document_ids = {tab.tab_id for tab in document.tabs}
            if project_replaced:
                self._simulation_jobs.reset_project_session()
                # Same-ID retained widgets otherwise keep old debounce/capture
                # state and can publish an old gesture into the new document.
                for tab_id, bundle in tuple(self._tab_surfaces.items()):
                    if tab_id not in document_ids:
                        continue
                    bundle.slider_binding.reset_project_session()
                    try:
                        self._workspace.plot_for_tab(tab_id).reset_plot_session_state()
                    except (KeyError, RuntimeError):
                        pass
            for tab_id in tuple(self._tab_surfaces):
                if tab_id not in document_ids:
                    self._drop_tab_surface(tab_id)
            for tab in document.tabs:
                if tab.tab_id not in self._tab_surfaces:
                    self._create_tab_surface(tab.tab_id)
            self._workspace.render(
                document,
                active_tab_id=self._workspace_tab_registry.active_tab_id,
            )
            self._select_tab_surfaces(
                self._workspace_tab_registry.active_tab_id,
            )
            self._apply_persisted_plot_states(document)
            for tab in document.tabs:
                self._refresh_tab_controls(tab.tab_id)
            if project_replaced:
                # Rendering the replacement text is an undoable editor edit;
                # clear afterwards so Undo cannot restore the prior project.
                for bundle in tuple(self._tab_surfaces.values()):
                    bundle.editor.clear_native_text_history()
            self._render_tab_state()
            self._refresh_title()
            self._refresh_actions()
        finally:
            self._rendering_workspace = False

    def _refresh_tab_controls(
        self,
        tab_id: str,
        *,
        preserve_reactions_editor: bool = False,
        preserve_slider_editor_drafts: bool = True,
    ) -> None:
        """Refresh one tab's controls, sliders, data views, and results."""

        target = str(tab_id)
        bundle = self._tab_surfaces[target]
        document = self._project_session.document
        tab = self._workspace_tab_registry.tab(target)
        presentation = self._workspace_tab_registry.presentation(target)
        try:
            if not preserve_reactions_editor:
                bundle.editor_binding.apply_render_projection(
                    bundle.editor_binding.prepare_render()
                )
            bundle.initial_actions.apply_render_projection(
                bundle.initial_actions.prepare_render()
            )
            bundle.simulation_panel.render_run_defaults(
                tab.run_defaults,
                simulation_job_kind=self._simulation_jobs.current_kind,
            )
            try:
                slider_projection = bundle.slider_binding.prepare_render()
                bundle.slider_binding.apply_render_projection(
                    slider_projection,
                    preserve_editor_drafts=preserve_slider_editor_drafts,
                )
                bundle.slider_binding.settle_render_projection(slider_projection)
            finally:
                bundle.slider_binding.publish_slider_draft_state()
            self._render_dataset_panels(bundle, document, tab, presentation)
            target_names = tuple(
                tab.initial_condition_set(set_id).name
                for set_id in slider_projection.targets
            )
            bundle.editor.set_slider_edit_targets_summary(
                "Initial-condition sets: "
                + (", ".join(target_names) if target_names else "none")
            )
            self._refresh_symbolic_calculator(target)
        finally:
            try:
                self._render_results_for_tab(target)
            finally:
                self._refresh_project_state()

    def _refresh_application_plot_preferences(self) -> None:
        self._fit_session_owner.refresh_plot_preferences()
        for tab in self._project_session.document.tabs:
            bundle = self._tab_surfaces.get(tab.tab_id)
            if bundle is None:
                continue
            bundle.overlay_panel.refresh_application_style_controls()
            bundle.overlay_panel.refresh_color_swatches(
                known_species=tuple(canonical_observation_target_names(tab)),
            )
            self._render_results_for_tab(tab.tab_id)
        self._refresh_actions()

    @staticmethod
    def _render_dataset_panels(
        bundle: _TabSurfaceBundle,
        document: ProjectDocument,
        tab: MechanismTabDocument,
        presentation: ProjectPresentationState,
    ) -> None:
        snapshot = build_data_manager_snapshot(
            document=document,
            active_tab=tab,
            presentation=presentation,
        )
        bundle.data_panel.render_snapshot(snapshot, document)
        bundle.overlay_panel.render_snapshot(
            records=snapshot.datasets,
            active_links=snapshot.active_links,
            known_species=tuple(canonical_observation_target_names(tab)),
        )

    def _refresh_dataset_views(
        self,
        tab_ids: Iterable[str] | None = None,
    ) -> None:
        """Rerender only Data, Overlay, and result projections."""

        document = self._project_session.document
        targets = (
            tuple(tab.tab_id for tab in document.tabs)
            if tab_ids is None
            else tuple(dict.fromkeys(str(tab_id) for tab_id in tab_ids))
        )
        try:
            for tab_id in targets:
                bundle = self._tab_surfaces.get(tab_id)
                if bundle is None:
                    continue
                tab = document.tab(tab_id)
                self._render_dataset_panels(
                    bundle, document, tab,
                    self._workspace_tab_registry.presentation(tab_id),
                )
                self._render_results_for_tab(tab_id)
        finally:
            self._refresh_project_state()

    def _apply_persisted_plot_states(self, document: ProjectDocument) -> None:
        """Apply each tab's persisted plot state to its retained panel."""

        for tab in document.tabs:
            self._apply_persisted_plot_state(tab.tab_id)

    def _apply_persisted_plot_state(self, tab_id: str) -> None:
        state = self._workspace_tab_registry.presentation(tab_id)
        self._workspace.plot_for_tab(tab_id).set_persisted_plot_state(
            state,
        )

    def _compose_result_input(self, tab_id: str) -> ResultRenderInput:
        current_document = self._project_session.document
        tab = current_document.tab(tab_id)
        presentation = self._workspace_tab_registry.presentation(tab_id)
        repository = self._repository_for_tab(tab_id)
        visible_sets = tuple(
            VisibleResultSet(
                set_id=set_id,
                label=tab.initial_condition_set(set_id).name,
            )
            for set_id in presentation.plot_set_ids
        )
        experimental_layers = build_experimental_layer_inputs(
            document=current_document,
            active_tab=tab,
            presentation=presentation,
        )
        analysis = tab.dsl_analysis
        modeled_species = tuple(canonical_observation_target_names(tab))
        simulation_series_names = list(
            ()
            if analysis is None
            else (*analysis.species_names, *analysis.observable_names)
        )
        for visible in visible_sets:
            results = (
                repository.current_result(visible.set_id),
                repository.reference_result(visible.set_id),
            )
            for result in results:
                if result is not None:
                    simulation_series_names.extend(result.series)
        simulation_series = tuple(dict.fromkeys(simulation_series_names))
        color_by_axis_selection: dict[AxisSeriesSelection, str] = {
            SimulationSeriesAxisSelection(species): (
                self.config_controller.species_color(
                    species,
                    modeled_species,
                ).name(QtGui.QColor.NameFormat.HexRgb)
            )
            for species in simulation_series
        }
        for layer in experimental_layers:
            color_name = layer.model_target or layer.raw_column
            color_by_axis_selection[layer.axis] = (
                self.config_controller.species_color(
                    color_name,
                    modeled_species,
                ).name(QtGui.QColor.NameFormat.HexRgb)
            )
        return compose_result_input(
            tab_id=tab_id,
            current_set_id=presentation.current_set_id,
            visible_sets=visible_sets,
            repository=repository,
            reference_lines_visible=presentation.reference_lines_visible,
            intervention_annotations_visible=(
                presentation.intervention_annotations_visible
            ),
            intervention_labels_visible=presentation.intervention_labels_visible,
            active_overlay_layers=experimental_layers,
            color_by_axis_selection=color_by_axis_selection,
            experimental_point_shape=self.config_controller.experimental_point_shape(),
            experimental_point_size=(
                self.config_controller.experimental_point_size()
            ),
            experimental_point_opacity=(
                self.config_controller.experimental_point_opacity()
            ),
            selected_x_series=presentation.selected_x_series,
            selected_x_label=(
                experimental_axis_label(
                    current_document, presentation.selected_x_series,
                )
                if isinstance(
                    presentation.selected_x_series, ExperimentalAxisSelection,
                )
                else ""
            ),
            selected_y_series=presentation.selected_y_series,
            selected_y_uses_default_all=(
                presentation.selected_y_uses_default_all
            ),
        )

    def _render_results_for_tab(self, tab_id: str) -> None:
        target = str(tab_id)
        try:
            self._project_session.document.tab(target)
        except KeyError:
            return
        if target not in self._tab_surfaces:
            return
        render_input = self._compose_result_input(target)
        plot = self._workspace.plot_for_tab(target)
        plot.set_copy_all_available(self._canonical_export_available(target))
        plot.apply_result_input(render_input)
        self._refresh_actions()

    def _canonical_export_available(self, tab_id: str) -> bool:
        """Return whether the active-tab repositories own any retained result."""

        target = str(tab_id)
        try:
            tab = self._project_session.document.tab(target)
            repository = self._repository_for_tab(target)
        except (KeyError, RuntimeError):
            return False
        return any(
            repository.current_result(initial_set.set_id) is not None
            or repository.reference_result(initial_set.set_id) is not None
            for initial_set in tab.initial_condition_sets
        )

    def _canonical_export_blocks(self, tab_id: str):
        target = str(tab_id)
        return compose_canonical_export_blocks(
            document=self._project_session.document,
            tab_id=target,
            repository=self._repository_for_tab(target),
        )

    def _copy_all_for_tab(self, tab_id: str) -> bool:
        """Serialize the active tab's complete canonical export roster once."""

        target = str(tab_id)
        blocks = self._canonical_export_blocks(target)
        if not blocks:
            self.set_status_text("No current result data is available to copy.")
            return False
        try:
            output = io.StringIO()
            output.writelines(iter_blocks_tsv_chunks(blocks))
            QtWidgets.QApplication.clipboard().setText(output.getvalue())
        except (TypeError, ValueError, OSError) as exc:
            QtWidgets.QMessageBox.critical(self, "Copy All", str(exc))
            return False
        return True

    @QtCore.Slot(str)
    def _on_workspace_result_diagnostic(self, message: str) -> None:
        """Surface a successful result input's exact diagnostic."""

        self.set_status_text(str(message))

    @QtCore.Slot(str)
    def _on_results_changed(self, tab_id: str) -> None:
        """Recompose result publications after their owner-side retirement."""

        target = str(tab_id)
        if self._rendering_workspace:
            return
        try:
            plot = self._workspace.plot_for_tab(target)
        except (KeyError, RuntimeError):
            return
        if target in self._plot_membership_refreshing:
            self._render_results_for_tab(target)
            return
        if (
            target in self._slider_apply_refreshing
            or target in self._current_row_refreshing
        ):
            return
        bundle = self._tab_surfaces.get(target)
        if bundle is not None and self._simulation_jobs.current_kind != "run":
            plot.retire_interactive_preview_targets(
                tuple(
                    dict.fromkeys(
                        (
                            *bundle.slider_binding.active_preview_set_ids,
                            *bundle.slider_binding.resolve_target_set_ids(),
                        )
                    )
                )
            )
        self._render_results_for_tab(target)

    @QtCore.Slot(str, str, object, object)
    def _on_interactive_result_ready(
        self,
        tab_id: str,
        set_id: str,
        request: object,
        result: object,
    ) -> None:
        """Project one canonical transient result onto its owning plot."""

        plot = self._workspace.plot_for_tab(tab_id)
        plot.apply_interactive_preview_result(
            tab_id,
            set_id,
            request,
            result,
        )

    @QtCore.Slot(str, str, object)
    def _on_interactive_target_failed(
        self,
        tab_id: str,
        set_id: str,
        request: object,
    ) -> None:
        """Retire one failed transient target from its owning plot."""

        self._workspace.plot_for_tab(tab_id).retire_interactive_preview_target(
            tab_id,
            set_id,
            request,
        )

    @QtCore.Slot(str, object)
    def _on_run_dispatched(self, tab_id: str, set_ids: object) -> None:
        target = str(tab_id)
        bundle = self._tab_surfaces.get(target)
        if bundle is None:
            return
        target_set_ids = tuple(str(set_id) for set_id in set_ids)
        self._workspace.plot_for_tab(target).retire_interactive_preview_targets(
            target_set_ids
        )
        if not bundle.initial_actions.union_plot_membership(target_set_ids):
            return
        self._render_results_for_tab(target)
        self._refresh_project_state()

    def _repository_for_tab(self, tab_id: str) -> ResultRepository:
        return self._workspace_tab_registry.tab_runtime(tab_id).result_repository

    def _retire_interactive_plot_targets(
        self,
        delta: ScientificDocumentDelta,
    ) -> None:
        """Retire only physical interactive Preview plot targets."""

        for tab_id, target_set_ids in delta.targets_by_tab().items():
            self._workspace.plot_for_tab(
                tab_id
            ).retire_interactive_preview_targets(target_set_ids)

    def _post_document_publication(
        self,
        delta: ScientificDocumentDelta,
    ) -> None:
        """Reproject each surviving result touched by one document delta."""

        for target in delta.targets_by_tab():
            if target in self._tab_surfaces:
                self._render_results_for_tab(target)

    def _build_explicit_target_for_tab(
        self,
        tab: MechanismTabDocument,
        set_id: str,
    ):
        """Build one bound Run target through the direct request path."""

        target = tab.initial_condition_set(str(set_id))
        return build_explicit_target(
            prepared_dsl=tab.prepared_dsl,
            tab_id=tab.tab_id,
            set_id=target.set_id,
            set_name=target.name,
            initial_values=canonical_initial_values(tab, target.set_id),
            execution_settings=tab.resolved_execution_settings(target.set_id),
            scientific_settings=tab.scientific_settings,
        )

    def _activate_materialized_set(
        self,
        tab_id: str,
        materialized_set_ids: tuple[str, ...],
    ) -> None:
        """Select the first newly materialized authored set after publication."""

        ids = tuple(str(set_id) for set_id in materialized_set_ids)
        if not ids:
            return
        target = str(tab_id)
        self._workspace_tab_registry.tab(target).initial_condition_set(ids[0])
        current = self._workspace_tab_registry.presentation(target)
        next_state = replace(current, current_set_id=ids[0])
        self._workspace_tab_registry.set_presentation(
            target,
            next_state,
        )
        self._workspace_tab_registry.tab_runtime(target).selected_run_set_ids = (
            ids[0],
        )

    def _publish_source_proposal(
        self,
        tab_id: str,
        visible: MechanismAuthoringSource,
        proposal: MechanismSourceReconciliation,
        *,
        expected_token: ProjectContentToken,
        refresh_after_admission: bool = True,
    ) -> ProjectContentToken | None:
        """Publish one admitted source proposal and clear its draft."""

        tab_id = str(tab_id)
        runtime = self._workspace_tab_registry.tab_runtime(tab_id)
        # An external source (template, preset, example) supersedes the visible
        # draft instead of rewriting it; decide before publication so nothing
        # after the publication can return early.
        rewrite_visible_draft = runtime.reactions_draft is visible
        result = self._workspace_tab_registry.replace_document(
            proposal.proposed_document,
            expected_token=expected_token,
        )
        if result is None:
            return None
        proposed_tab = proposal.proposed_document.tab(tab_id)
        bundle = self._tab_surfaces[tab_id]
        if rewrite_visible_draft:
            bundle.editor_binding.apply_accepted_source_rewrite(
                visible,
                proposed_tab.mechanism_source,
            )
        self._activate_materialized_set(tab_id, proposal.materialized_set_ids)
        runtime.reactions_draft = None
        valid_species = set(canonical_species_names(proposed_tab))
        surviving_sets = {item.set_id for item in proposed_tab.initial_condition_sets}
        runtime.initial_text_by_set_id = {
            set_id: {
                species: raw
                for species, raw in values.items()
                if species in valid_species
            }
            for set_id, values in runtime.initial_text_by_set_id.items()
            if set_id not in proposal.materialized_set_ids
            and set_id in surviving_sets
            and any(species in valid_species for species in values)
        }
        if refresh_after_admission:
            bundle.editor_binding.render()
            self._refresh_tab_controls(
                tab_id,
                preserve_reactions_editor=True,
            )
        return result

    def _tab_has_drafts(self, tab_id: str) -> bool:
        """Runtime drafts plus the Run-defaults draft owned by the Run panel."""

        return bool(
            self._workspace_tab_registry.tab_runtime(tab_id).has_dirty_drafts
            or self._tab_surfaces[tab_id].simulation_panel.has_run_defaults_draft
        )

    def _render_tab_state(self) -> None:
        for tab in self._project_session.document.tabs:
            self._workspace.set_tab_dirty(tab.tab_id, self._tab_has_drafts(tab.tab_id))

    # ------------------------------------------------------------------
    # Mechanism tab and retained-page intents
    # ------------------------------------------------------------------
    @QtCore.Slot(str)
    def _select_tab(self, tab_id: str) -> None:
        if not self._mutation_input_admitted("Select Tab"):
            return
        target = str(tab_id)
        previous = self._workspace_tab_registry.active_tab_id
        if target == previous:
            return
        self._settle_visible_tab_inputs()
        self._workspace_tab_registry.select_tab(target)
        self._select_tab_surfaces(target)

    @QtCore.Slot()
    def _create_tab(self) -> None:
        if not self._mutation_input_admitted("Create Tab"):
            return
        self._settle_visible_tab_inputs()
        tab = self._workspace_tab_registry.create_tab(
            run_defaults=self.config_controller.application_run_defaults(),
            scientific_settings=(
                self.config_controller.application_scientific_defaults()
            ),
            expected_token=self._project_session.capture_content_token(),
        )
        self._create_tab_surface(tab.tab_id)
        self._refresh_workspace_tab_structure()
        self._apply_persisted_plot_state(tab.tab_id)
        self._refresh_tab_controls(tab.tab_id)

    @QtCore.Slot()
    def _finish_focused_tab_input(self) -> None:
        # Finish only the focused field's ordinary editing interaction before
        # reading accepted setup. Do not apply the tab's other pending drafts.
        focused = QtWidgets.QApplication.focusWidget()
        if focused is not None:
            focused.clearFocus()

    @QtCore.Slot(str)
    def _copy_tab(self, tab_id: str) -> None:
        if not self._mutation_input_admitted("Create Copy"):
            return
        self._finish_focused_tab_input()
        tab = self._workspace_tab_registry.copy_tab(
            tab_id,
            expected_token=self._project_session.capture_content_token(),
        )
        self._create_tab_surface(tab.tab_id)
        self._refresh_workspace_tab_structure()
        self._apply_persisted_plot_state(tab.tab_id)
        self._refresh_tab_controls(tab.tab_id)

    @QtCore.Slot(object)
    def _reorder_tabs(self, tab_ids: tuple[str, ...]) -> None:
        if self._mutation_input_admitted("Reorder Tabs"):
            self._workspace_tab_registry.reorder_tabs(
                tab_ids,
                expected_token=self._project_session.capture_content_token(),
            )
        self._refresh_workspace_tab_structure()

    @QtCore.Slot(str, str)
    def _rename_tab(self, tab_id: str, name: str) -> None:
        if not self._mutation_input_admitted("Rename Tab"):
            return
        try:
            self._workspace_tab_registry.rename_tab(
                tab_id,
                name,
                expected_token=self._project_session.capture_content_token(),
            )
        except Exception as exc:
            self._workspace.show_rename_error(tab_id, str(exc))
            return
        self._refresh_workspace_tab_structure()

    @QtCore.Slot(str)
    def _close_tab(self, tab_id: str) -> None:
        if not self._mutation_input_admitted("Delete Tab"):
            return
        document = self._project_session.document
        if len(document.tabs) <= 1:
            self.set_status_text("The last mechanism tab cannot be deleted.")
            return
        self._capture_visible_tab_inputs(tab_id=str(tab_id))
        runtime = self._workspace_tab_registry.tab_runtime(tab_id)
        bundle = self._tab_surfaces[str(tab_id)]
        has_content = self._workspace_tab_registry.tab_has_content(
            tab_id,
            fresh_run_defaults=self.config_controller.application_run_defaults(),
            fresh_scientific_settings=(
                self.config_controller.application_scientific_defaults()
            ),
        )
        if has_content or self._tab_has_drafts(tab_id):
            dialog = QtWidgets.QMessageBox(self)
            dialog.setWindowTitle("Delete Mechanism")
            dialog.setIcon(QtWidgets.QMessageBox.Icon.Warning)
            details = [
                "Delete this mechanism tab and discard its canonical content?"
            ]
            if runtime.reactions_draft is not None:
                details.append("Visible Reactions edits will be discarded.")
            if runtime.notes_draft is not None:
                details.append("Visible Notes edits will be discarded.")
            if runtime.initial_text_by_set_id:
                details.append("Initial Conditions edits will be discarded.")
            if bundle.simulation_panel.has_run_defaults_draft:
                details.append("Run-default edits will be discarded.")
            if (
                runtime.slider_parameter_draft
                or runtime.slider_species_drafts_by_set_id
            ):
                details.append("Slider Preview drafts will be discarded.")
            dialog.setText("\n\n".join(details))
            discard = dialog.addButton(
                "Delete",
                QtWidgets.QMessageBox.ButtonRole.DestructiveRole,
            )
            dialog.addButton(QtWidgets.QMessageBox.StandardButton.Cancel)
            dialog.exec()
            if dialog.clickedButton() is not discard:
                return
        runtime.reactions_draft = None
        runtime.notes_draft = None
        runtime.initial_text_by_set_id.clear()
        bundle.simulation_panel.discard_run_defaults_draft()
        self._workspace_tab_registry.close_tab(
            tab_id,
            expected_token=self._project_session.capture_content_token(),
        )
        self._drop_tab_surface(
            tab_id,
        )
        self._workspace.drop_tab_session_state(tab_id)
        self._refresh_workspace_tab_structure()

    def _add_initial_set(self, tab_id: str) -> None:
        if not self._mutation_input_admitted("Add Initial Conditions Set"):
            return
        if not self._accept_visible_source(
            tab_id,
            description="Add Initial Conditions Set",
        ):
            return
        if self._tab_surfaces[tab_id].initial_actions.add_set() is not None:
            self._refresh_dataset_views((tab_id,))

    def _move_initial_sets(self, tab_id: str, delta: int) -> None:
        if not self._mutation_input_admitted("Move Initial Conditions Set"):
            return
        before = tuple(
            item.set_id
            for item in self._workspace_tab_registry.tab(
                tab_id
            ).initial_condition_sets
        )
        self._tab_surfaces[tab_id].initial_actions.move_selected(delta)
        after = tuple(
            item.set_id
            for item in self._workspace_tab_registry.tab(
                tab_id
            ).initial_condition_sets
        )
        if after != before:
            self._refresh_dataset_views((tab_id,))

    def _delete_initial_sets(self, tab_id: str) -> None:
        if not self._mutation_input_admitted("Delete Initial Conditions Set"):
            return
        bundle = self._tab_surfaces[tab_id]
        targets = bundle.batch_table.selected_set_ids()
        if not targets:
            return
        names = [
            self._workspace_tab_registry.tab(tab_id).initial_condition_set(set_id).name
            for set_id in targets
        ]
        answer = QtWidgets.QMessageBox.question(
            self,
            "Delete Initial Conditions",
            "Delete the selected initial-condition set(s) and their tab-local "
            "Run, mapping, Fit, result, and presentation references?\n\n"
            + ", ".join(names),
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.Cancel,
            QtWidgets.QMessageBox.StandardButton.Cancel,
        )
        if answer != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        if bundle.initial_actions.delete_selected():
            self._refresh_dataset_views((tab_id,))

    def _set_workspace_run_defaults(
        self,
        tab_id: str,
        settings: SimulationExecutionSettings,
    ) -> None:
        if not self._mutation_input_admitted("Configure Run Defaults"):
            return
        bundle = self._tab_surfaces[tab_id]
        token = self._project_session.capture_content_token()
        result = bundle.initial_actions.set_workspace_run_defaults(
            settings,
            expected_token=token,
        )
        if result != token:
            tab = self._workspace_tab_registry.tab(tab_id)
            bundle.simulation_panel.render_run_defaults(
                tab.run_defaults,
                simulation_job_kind=self._simulation_jobs.current_kind,
            )
            bundle.slider_binding.render()
            self._refresh_project_state()

    def _on_plot_membership_requested(
        self,
        tab_id: str,
        set_id: str,
        requested: bool,
    ) -> None:
        if not self._mutation_input_admitted("Change Plot Membership"):
            return
        changed = self._tab_surfaces[tab_id].initial_actions.set_plot_requested(
            set_id,
            requested,
        )
        if not changed:
            return
        self._refresh_project_state()
        if not requested:
            self._render_results_for_tab(tab_id)
            return
        slider_binding = self._tab_surfaces[tab_id].slider_binding
        slider_targets = slider_binding.resolve_target_set_ids()
        if set_id in slider_targets:
            runtime = self._workspace_tab_registry.tab_runtime(tab_id)
            protected_slider_state = bool(
                runtime.slider_parameter_draft
                or any(
                    runtime.slider_species_drafts_by_set_id.get(target_set_id)
                    for target_set_id in slider_targets
                )
                or slider_binding.active_preview_set_ids
                or self._simulation_jobs.pending_preview_tab_id == tab_id
                or (
                    self._simulation_jobs.current_kind == "preview"
                    and self._simulation_jobs.origin_tab_id == tab_id
                )
            )
            if protected_slider_state:
                self._render_results_for_tab(tab_id)
                return
        tab = self._workspace_tab_registry.tab(tab_id)
        # An empty mechanism has no parsed form and so no cached results.
        if tab.prepared_dsl is None:
            self._render_results_for_tab(tab_id)
            return
        target = self._build_explicit_target_for_tab(tab, set_id)
        self._plot_membership_refreshing.add(tab_id)
        try:
            cached = self._simulation_jobs.show_cached(tab_id, set_id, target.execution_request)
        finally:
            self._plot_membership_refreshing.discard(tab_id)
        if not cached:
            self._render_results_for_tab(tab_id)

    def _on_slider_pin_change_requested(
        self,
        bundle: _TabSurfaceBundle,
        set_id: str,
        pinned: bool,
    ) -> None:
        if bundle.initial_actions.set_slider_pinned(set_id, pinned):
            self._refresh_project_state()
            bundle.slider_binding.render()

    def _run_admitted(self, tab_id: str) -> bool:
        bundle = self._tab_surfaces[str(tab_id)]
        selected = bundle.initial_actions.selected_run_set_ids()
        if not self._computation_gate.is_idle:
            return False
        if self._simulation_work_active():
            return False
        runtime = self._workspace_tab_registry.tab_runtime(tab_id)
        if runtime.reactions_draft is None and (
            not selected or bundle.initial_actions.has_invalid_cells(selected)
        ):
            return False
        return self._has_visible_reactions(tab_id)

    def _has_visible_reactions(self, tab_id: str) -> bool:
        return bool(self._visible_source_for_tab(tab_id).reactions_text.strip())

    def _set_tab_run_enabled(self, tab_id: str, enabled: bool) -> None:
        admitted = bool(enabled) and self._mutation_input_admitted()
        bundle = self._tab_surfaces.get(str(tab_id))
        if bundle is None:
            return
        bundle.editor.set_run_enabled(admitted)
        bundle.simulation_panel.run_btn.setEnabled(admitted)
        if (
            self._run_action is not None
            and str(tab_id) == self._workspace_tab_registry.active_tab_id
        ):
            self._run_action.setEnabled(admitted)

    def _settle_visible_tab_inputs(self) -> None:
        """Publish the active tab's valid drafts once; invalid drafts stay on it."""

        target = self._workspace_tab_registry.active_tab_id
        self._capture_visible_tab_inputs(tab_id=target)
        candidate = self._captured_input_candidate(
            self._project_session.document,
            (target,),
            discard_invalid=True,
        )
        if candidate != self._project_session.document:
            self._workspace_tab_registry.replace_document(
                candidate,
                expected_token=self._project_session.capture_content_token(),
            )
        self._clear_settled_input_drafts((target,), candidate)
        self._refresh_tab_controls(target)

    def _capture_visible_tab_inputs(self, *, tab_id: str | None = None) -> None:
        """Capture the one focused IC or Run-default editor before inspection."""

        focused = QtWidgets.QApplication.focusWidget()
        targets = (
            (str(tab_id),)
            if tab_id is not None
            else tuple(self._tab_surfaces)
        )
        for target in targets:
            bundle = self._tab_surfaces.get(target)
            if bundle is None:
                continue
            # Check the table before the enclosing panel: delegate editors are
            # descendants of both widgets.
            if focused is not None and (
                focused is bundle.batch_table
                or bundle.batch_table.isAncestorOf(focused)
            ):
                bundle.initial_actions.capture_active_editor()
                return
            if focused is not None and (
                focused is bundle.simulation_panel
                or bundle.simulation_panel.isAncestorOf(focused)
            ) and bundle.simulation_panel.capture_run_defaults_draft():
                self._on_initial_draft_state_changed(target)
                return

    def _captured_input_candidate(
        self,
        document: ProjectDocument,
        tab_ids: Iterable[str],
        *,
        discard_invalid: bool,
        selected_by_tab: dict[str, tuple[str, ...]] | None = None,
        superseded_by_tab: dict[str, tuple[str, ...]] | None = None,
    ) -> ProjectDocument:
        """Project captured widget drafts into one detached document candidate.

        Invalid drafts are skipped when ``discard_invalid`` is set and raise
        ``ValueError`` otherwise.
        """

        tabs = list(document.tabs)
        for target in tuple(dict.fromkeys(str(item) for item in tab_ids)):
            runtime = self._workspace_tab_registry.tab_runtime(target)
            tab_index = next(
                index for index, item in enumerate(tabs) if item.tab_id == target
            )
            tab = tabs[tab_index]
            values_by_set = {
                item.set_id: dict(item.values)
                for item in tab.initial_condition_sets
            }
            species_roster = set(canonical_species_names(tab))
            selected = None if selected_by_tab is None else selected_by_tab[target]
            superseded = (superseded_by_tab or {}).get(target, ())
            for set_id, values in runtime.initial_text_by_set_id.items():
                if (
                    set_id not in values_by_set
                    or set_id in superseded
                    or (selected is not None and set_id not in selected)
                ):
                    continue
                baseline = tab.initial_condition_set(set_id).name
                for species, raw in values.items():
                    if species not in species_roster:
                        continue
                    try:
                        values_by_set[set_id][str(species)] = concentration_from_text(
                            str(raw),
                            field=f"Initial {species} {raw!r} in {baseline!r}",
                        )
                    except ValueError as exc:
                        if discard_invalid:
                            continue
                        raise ValueError(f"Mechanism tab {tab.name!r}: {exc}") from exc
            updated_sets = tuple(
                initial_set.with_values(values_by_set[initial_set.set_id])
                for initial_set in tab.initial_condition_sets
            )
            tab = tab.with_initial_condition_sets(updated_sets)
            panel = self._tab_surfaces[target].simulation_panel
            if panel.has_run_defaults_draft:
                try:
                    settings = panel.current_settings_candidate()
                except ValueError as exc:
                    if not discard_invalid:
                        raise ValueError(f"Mechanism tab {tab.name!r}: {exc}") from exc
                else:
                    tab = tab.with_run_defaults(settings)
            tabs[tab_index] = tab
        return replace(document, tabs=tuple(tabs))

    def _clear_settled_input_drafts(
        self,
        tab_ids: Iterable[str],
        document: ProjectDocument,
        *,
        discard_invalid: bool = False,
        selected_by_tab: dict[str, tuple[str, ...]] | None = None,
    ) -> None:
        for target in tuple(dict.fromkeys(str(item) for item in tab_ids)):
            runtime = self._workspace_tab_registry.tab_runtime(target)
            if discard_invalid:
                runtime.initial_text_by_set_id.clear()
            else:
                for set_id, values in tuple(runtime.initial_text_by_set_id.items()):
                    if (
                        selected_by_tab is not None
                        and set_id not in selected_by_tab[target]
                    ):
                        continue
                    remaining = {
                        species: raw
                        for species, raw in values.items()
                        if not is_valid_initial_text(raw)
                    }
                    if remaining:
                        runtime.initial_text_by_set_id[set_id] = remaining
                    else:
                        runtime.initial_text_by_set_id.pop(set_id, None)
            panel = self._tab_surfaces[target].simulation_panel
            if (
                not discard_invalid
                and panel.has_run_defaults_draft
                and not panel.run_defaults_draft_is_valid()
            ):
                continue  # An invalid Run-settings draft stays on its tab.
            panel.settle_saved_run_defaults(document.tab(target).run_defaults)

    def _confirm_discard_slider_preview(
        self,
        tab_id: str,
        action: str,
    ) -> bool:
        slider_binding = self._tab_surfaces[tab_id].slider_binding
        if not slider_binding.has_slider_draft():
            return True
        dialog = QtWidgets.QMessageBox(self)
        dialog.setWindowTitle(action)
        dialog.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        dialog.setText(
            f"{action} never consumes unapplied Slider Preview values."
        )
        discard = dialog.addButton(
            "Discard Slider Preview and Continue",
            QtWidgets.QMessageBox.ButtonRole.DestructiveRole,
        )
        dialog.addButton(QtWidgets.QMessageBox.StandardButton.Cancel)
        dialog.exec()
        if dialog.clickedButton() is not discard:
            return False
        self._discard_slider_draft(tab_id)
        return True

    def _discard_slider_draft(self, tab_id: str) -> bool:
        target = str(tab_id)
        binding = self._tab_surfaces[target].slider_binding
        ids = tuple(
            dict.fromkeys(
                (*binding.active_preview_set_ids, *binding.resolve_target_set_ids())
            )
        )
        if ids:
            self._workspace.plot_for_tab(target).retire_interactive_preview_targets(
                ids
            )
        changed = binding.discard_slider_draft()
        self._render_results_for_tab(target)
        return changed

    def _reset_slider_draft(self, tab_id: str) -> bool:
        target = str(tab_id)
        binding = self._tab_surfaces[target].slider_binding
        ids = tuple(
            dict.fromkeys(
                (*binding.active_preview_set_ids, *binding.resolve_target_set_ids())
            )
        )
        if ids:
            self._workspace.plot_for_tab(target).retire_interactive_preview_targets(
                ids
            )
        changed = binding.reset()
        self._render_results_for_tab(target)
        return changed

    def _start_selected_run(self, tab_id: str | None = None) -> bool:
        if not self._mutation_input_admitted("Run"):
            return False
        target = (
            self._workspace_tab_registry.active_tab_id
            if tab_id is None
            else str(tab_id)
        )
        bundle = self._tab_surfaces[target]
        if not self._computation_gate.is_idle:
            self.set_status_text("Run rejected: another computation is active.")
            return False
        self._capture_visible_tab_inputs(tab_id=target)
        if not self._confirm_discard_slider_preview(target, "Run"):
            return False
        capture_token = self._project_session.capture_content_token()
        document = self._project_session.document
        runtime = self._workspace_tab_registry.tab_runtime(target)
        visible_source = self._visible_source_for_tab(target)
        if runtime.reactions_draft is None:
            source_proposal = MechanismSourceReconciliation(proposed_document=document)
        else:
            source_proposal = self._prepare_source_reconciliation(
                document,
                target,
                visible_source,
                description="Run",
            )
            if source_proposal is None:
                return False
        candidate_tab = source_proposal.proposed_document.tab(target)
        surviving_ids = {item.set_id for item in candidate_tab.initial_condition_sets}
        selected = (
            source_proposal.materialized_set_ids[:1]
            if source_proposal.materialized_set_ids
            else tuple(
                set_id for set_id in bundle.initial_actions.selected_run_set_ids()
                if set_id in surviving_ids
            )
        )
        if not selected:
            self.set_status_text("Select one or more Initial Conditions rows.")
            return False
        try:
            candidate = self._captured_input_candidate(
                source_proposal.proposed_document,
                (target,),
                discard_invalid=False,
                selected_by_tab={target: selected},
                superseded_by_tab={target: source_proposal.materialized_set_ids},
            )
        except ValueError as exc:
            self.set_status_text(str(exc))
            return False
        source_proposal = replace(source_proposal, proposed_document=candidate)
        capture_tab = candidate.tab(target)
        capture = ExplicitRunCapture(
            tab_id=target,
            targets=tuple(
                self._build_explicit_target_for_tab(capture_tab, set_id)
                for set_id in selected
            ),
        )
        published = self._publish_source_proposal(
            target,
            visible_source,
            source_proposal,
            expected_token=capture_token,
        )
        if published is None:
            return False
        self._clear_settled_input_drafts(
            (target,), candidate, selected_by_tab={target: selected}
        )
        self._refresh_tab_controls(target, preserve_reactions_editor=True)
        bundle.slider_binding.cancel_pending_preview_intent()
        return bool(self._simulation_jobs.start_run(capture))

    def _inspect_transient_state(self) -> ProjectTransientState:
        reactions_draft = False
        notes_draft = False
        initial_condition_drafts = False
        invalid_authoring_tabs: list[str] = []
        run_defaults_drafts = False
        slider_drafts = False
        for tab in self._project_session.document.tabs:
            runtime = self._workspace_tab_registry.tab_runtime(tab.tab_id)
            if runtime.reactions_draft is not None:
                reactions_draft = True
            notes_draft = bool(
                notes_draft or runtime.notes_draft is not None
            )
            initial_condition_drafts = bool(
                initial_condition_drafts or runtime.initial_text_by_set_id
            )
            bundle = self._tab_surfaces[tab.tab_id]
            if bundle.initial_actions.has_invalid_cells() or (
                bundle.simulation_panel.has_run_defaults_draft
                and not bundle.simulation_panel.run_defaults_draft_is_valid()
            ):
                invalid_authoring_tabs.append(tab.name)
            run_defaults_drafts = bool(
                run_defaults_drafts
                or bundle.simulation_panel.has_run_defaults_draft
            )
            slider_drafts = bool(
                slider_drafts
                or runtime.slider_parameter_draft
                or runtime.slider_species_drafts_by_set_id
            )
        return ProjectTransientState(
            reactions_draft=reactions_draft,
            notes_draft=notes_draft,
            initial_condition_drafts=initial_condition_drafts,
            invalid_authoring_tabs=tuple(invalid_authoring_tabs),
            run_defaults_drafts=run_defaults_drafts,
            slider_drafts=slider_drafts,
            active_run_or_preview=self._simulation_work_active(),
            active_fit=self._fit_session_owner.active,
            active_import=bool(self._dataset_import_staging.busy),
            active_xlsx_export=self._xlsx_export_owner.is_exporting,
            active_scan=self._scan_owner.busy,
            active_calculator=any(owner.busy for owner in self._calculator_owners),
            active_spreadsheet=self._spreadsheet_editor is not None and self._spreadsheet_editor.has_changes,
        )

    def _prepare_before_save(
        self,
        discard_invalid_authoring: bool = False,
    ) -> tuple[
        ProjectDocument,
        tuple[tuple[str, tuple[str, ...]], ...],
    ] | None:
        """Build the exact archive candidate without disposing visible drafts."""

        candidate = self._project_session.document
        materialized_by_tab: list[tuple[str, tuple[str, ...]]] = []
        tab_ids = tuple(
            tab.tab_id for tab in candidate.tabs
        )
        for tab_id in tab_ids:
            runtime = self._workspace_tab_registry.tab_runtime(tab_id)
            if runtime.reactions_draft is not None:
                proposal = self._prepare_source_reconciliation(
                    candidate,
                    tab_id,
                    runtime.reactions_draft,
                    description="Save Project",
                )
                if proposal is None:
                    return None
                candidate = proposal.proposed_document
                materialized_by_tab.append(
                    (tab_id, tuple(proposal.materialized_set_ids))
                )
            if runtime.notes_draft is not None:
                candidate = replace(
                    candidate,
                    tabs=tuple(
                        replace(item, notes=runtime.notes_draft)
                        if item.tab_id == tab_id
                        else item
                        for item in candidate.tabs
                    ),
                )
        try:
            candidate = self._captured_input_candidate(
                candidate,
                tab_ids,
                discard_invalid=discard_invalid_authoring,
                superseded_by_tab=dict(materialized_by_tab),
            )
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Project Not Saved", str(exc))
            return None
        return candidate, tuple(materialized_by_tab)

    def _commit_saved_document(
        self,
        document: ProjectDocument,
        materialized_by_tab: tuple[tuple[str, tuple[str, ...]], ...],
        expected_token: ProjectContentToken,
    ) -> ProjectContentToken | None:
        """Semantically admit one exact save candidate before archive output."""

        captured_reaction_drafts: list[
            tuple[str, MechanismAuthoringSource]
        ] = []
        for tab in document.tabs:
            runtime = self._workspace_tab_registry.tab_runtime(tab.tab_id)
            draft = runtime.reactions_draft
            if draft is None:
                continue
            if self._tab_surfaces[tab.tab_id].editor.reactions_text() != (
                draft.reactions_text
            ):
                # The save candidate was prepared from a stale visible draft;
                # do not publish a canonical replacement that cannot be
                # represented by one exact native editor edit.
                return None
            captured_reaction_drafts.append((tab.tab_id, draft))
        refresh_tab_ids = tuple(
            tab_id for tab_id, _draft in captured_reaction_drafts
        )
        result = self._workspace_tab_registry.replace_document(
            document,
            expected_token=expected_token,
        )
        if result is None:
            return None
        self._clear_settled_input_drafts(
            (tab.tab_id for tab in document.tabs),
            document,
            discard_invalid=True,
        )
        for tab_id, visible_draft in captured_reaction_drafts:
            tab = document.tab(tab_id)
            self._tab_surfaces[
                tab_id
            ].editor_binding.apply_accepted_source_rewrite(
                visible_draft,
                tab.mechanism_source,
            )
        # The published candidate consumed every Reactions and Notes draft;
        # clear them here so a later archive-write failure cannot leave them.
        for tab in document.tabs:
            runtime = self._workspace_tab_registry.tab_runtime(tab.tab_id)
            runtime.reactions_draft = None
            runtime.notes_draft = None
            self._tab_surfaces[tab.tab_id].editor_binding.render()
        for tab_id, materialized_ids in materialized_by_tab:
            self._activate_materialized_set(tab_id, materialized_ids)
        for tab_id in refresh_tab_ids:
            self._refresh_tab_controls(
                tab_id,
                preserve_reactions_editor=True,
            )
        self._refresh_project_state()
        return result

    def _discard_authoring(self) -> None:
        focused_editor = QtWidgets.QApplication.focusWidget()
        if focused_editor is not None:
            for bundle in self._tab_surfaces.values():
                if (
                    focused_editor is not bundle.batch_table
                    and bundle.batch_table.isAncestorOf(focused_editor)
                ):
                    bundle.batch_table.closeEditor(
                        focused_editor,
                        QtWidgets.QAbstractItemDelegate.EndEditHint.RevertModelCache,
                    )
                    break
        for tab in self._project_session.document.tabs:
            runtime = self._workspace_tab_registry.tab_runtime(tab.tab_id)
            runtime.reactions_draft = None
            runtime.notes_draft = None
            runtime.initial_text_by_set_id.clear()
        for bundle in self._tab_surfaces.values():
            bundle.editor_binding.render()
            bundle.initial_actions.render()
            tab = self._workspace_tab_registry.tab(bundle.tab_id)
            bundle.simulation_panel.discard_run_defaults_draft()
            bundle.simulation_panel.render_run_defaults(
                tab.run_defaults,
                simulation_job_kind=self._simulation_jobs.current_kind,
            )

    def _discard_all_slider_drafts(self) -> None:
        for tab in self._project_session.document.tabs:
            self._discard_slider_draft(tab.tab_id)

    def _on_draft_state_changed(self, tab_id: str) -> None:
        if self._rendering_workspace:
            return
        current_tab_ids = {
            tab.tab_id for tab in self._project_session.document.tabs
        }
        if tab_id not in current_tab_ids:
            return
        runtime = self._workspace_tab_registry.tab_runtime(tab_id)
        if runtime.reactions_draft is not None:
            self._tab_surfaces[tab_id].slider_binding.fence_stop()
        # Reactions drafts invalidate only the calculator's current snapshot;
        # reverting the draft immediately rebuilds from the current
        # prepared-object identity. Slider and notes drafts leave executable identity
        # unchanged, but refreshing here keeps the projection synchronized for
        # every draft signal.
        self._refresh_symbolic_calculator(tab_id)
        self._workspace.set_tab_dirty(tab_id, self._tab_has_drafts(tab_id))
        if tab_id == self._workspace_tab_registry.active_tab_id:
            self._set_tab_run_enabled(tab_id, self._run_admitted(tab_id))
        self._refresh_title()
        self._refresh_actions()

    def _on_selected_run_sets_changed(self, tab_id: str) -> None:
        """Refresh one tab's Run eligibility after its selected rows change."""

        target = str(tab_id)
        self._set_tab_run_enabled(target, self._run_admitted(target))
        self._refresh_project_state()

    @QtCore.Slot(str, object)
    def _on_current_row_changed(
        self,
        tab_id: str,
        previous_index: QtCore.QModelIndex | None = None,
    ) -> None:
        """Refresh one current-row gesture through one guarded owner path."""

        target = str(tab_id)
        if self._rendering_workspace:
            return
        bundle = self._tab_surfaces.get(target)
        if bundle is None:
            return
        previous_set_id = (
            None
            if previous_index is None
            else bundle.batch_model.set_id_for_live_index(previous_index)
        )
        current_set_id = self._workspace_tab_registry.presentation(
            target
        ).current_set_id
        if previous_set_id is not None and previous_set_id == current_set_id:
            return
        if previous_set_id is not None and previous_set_id != current_set_id:
            self._workspace.plot_for_tab(target).retire_interactive_preview_targets(
                (previous_set_id,)
            )
        if target in self._current_row_refreshing:
            return
        # The guard coalesces results_changed emitted synchronously inside
        # this block; queued results published afterwards render normally.
        self._current_row_refreshing.add(target)
        try:
            # InitialConditionsTableActions has already admitted the distinct
            # current-row presentation intent.  Refresh Slider targets first,
            # then compose/draw results once, and finally publish shell state.
            bundle.slider_binding.render()
            self._render_results_for_tab(target)
            self._refresh_project_state()
        finally:
            self._current_row_refreshing.discard(target)

    def _on_initial_draft_state_changed(self, tab_id: str) -> None:
        """Apply current IC-draft admission to the owning tab."""

        target = str(tab_id)
        bundle = self._tab_surfaces.get(target)
        if bundle is None:
            return
        self._workspace.set_tab_dirty(target, self._tab_has_drafts(target))
        self._set_tab_run_enabled(target, self._run_admitted(target))
        self._refresh_title()
        self._refresh_actions()

    @QtCore.Slot(str)
    def _on_initial_values_changed(self, tab_id: str) -> None:
        """Refresh the owning Slider projection after canonical IC mutation."""

        target = str(tab_id)
        bundle = self._tab_surfaces.get(target)
        if bundle is None:
            return
        bundle.slider_binding.render()
        self._refresh_project_state()

    @QtCore.Slot(str)
    def _on_initial_set_name_changed(self, tab_id: str) -> None:
        """Reproject every display label after one accepted set rename."""

        target = str(tab_id)
        bundle = self._tab_surfaces.get(target)
        if bundle is None:
            return
        self._refresh_dataset_views((target,))
        self._refresh_slider_target_summary(
            target,
            bundle.slider_binding.resolve_target_set_ids(),
        )

    @QtCore.Slot(str, bool, bool)
    def _on_slider_draft_state_changed(
        self,
        tab_id: str,
        dirty: bool,
        reset_enabled: bool,
    ) -> None:
        """Project only one exact tab binding's Slider Draft actions."""

        target = str(tab_id)
        bundle = self._tab_surfaces.get(target)
        if bundle is None or bundle.slider_binding.tab_id != target:
            return
        bundle.editor.set_slider_draft_actions_enabled(dirty, reset_enabled)

    @QtCore.Slot(str, object)
    def _refresh_slider_target_summary(
        self,
        tab_id: str,
        target_set_ids: object,
    ) -> None:
        bundle = self._tab_surfaces[str(tab_id)]
        targets = tuple(str(set_id) for set_id in target_set_ids)
        tab = self._workspace_tab_registry.tab(tab_id)
        names = [tab.initial_condition_set(set_id).name for set_id in targets]
        bundle.editor.set_slider_edit_targets_summary(
            "Initial-condition sets: " + (", ".join(names) if names else "none")
        )

    @QtCore.Slot(str)
    def _apply_slider_draft(self, tab_id: str) -> bool:
        """Publish one Slider Apply, then reproject every existing consumer."""

        target = str(tab_id)
        bundle = self._tab_surfaces.get(target)
        if bundle is None or target in self._slider_apply_refreshing:
            return False
        self._slider_apply_refreshing.add(target)
        try:
            bundle.initial_actions.capture_active_editor()
            visible_source = self._visible_source_for_tab(target)
            if not bundle.slider_binding.apply():
                return False
            bundle.editor_binding.apply_accepted_source_rewrite(
                visible_source,
                self._workspace_tab_registry.tab(target).mechanism_source,
            )
            self._workspace_tab_registry.tab_runtime(target).reactions_draft = None
        finally:
            self._slider_apply_refreshing.discard(target)
        self._refresh_tab_controls(target, preserve_slider_editor_drafts=False)
        self.set_status_text("Slider values applied. The exact preview became the current results.")
        return True

    # ------------------------------------------------------------------
    # Source/examples/templates
    # ------------------------------------------------------------------
    def _visible_source_for_tab(self, tab_id: str) -> MechanismAuthoringSource:
        runtime = self._workspace_tab_registry.tab_runtime(tab_id)
        return (
            runtime.reactions_draft
            or self._project_session.document.tab(tab_id).mechanism_source
        )

    def _accept_visible_source_result(
        self,
        tab_id: str | None = None,
        *,
        description: str,
        source: MechanismAuthoringSource | None = None,
        refresh_after_admission: bool = True,
    ) -> MechanismSourceReconciliation | None:
        """Admit one action against retained truth or one exact visible draft."""
        if not self._mutation_input_admitted(description):
            return None
        tab_id = (
            self._workspace_tab_registry.active_tab_id
            if tab_id is None
            else str(tab_id)
        )
        document = self._project_session.document
        runtime = self._workspace_tab_registry.tab_runtime(tab_id)
        if source is None and runtime.reactions_draft is None:
            return MechanismSourceReconciliation(proposed_document=document)

        expected_token = self._project_session.capture_content_token()
        visible = source if source is not None else runtime.reactions_draft
        proposal = self._prepare_source_reconciliation(
            document,
            tab_id,
            visible,
            description=description,
        )
        if proposal is None:
            return None

        result = self._publish_source_proposal(
            tab_id,
            visible,
            proposal,
            expected_token=expected_token,
            refresh_after_admission=refresh_after_admission,
        )
        if result is None and proposal.proposed_document != document:
            return None
        if refresh_after_admission and result is None:
            self._refresh_project_state()
        return proposal

    def _accept_visible_source(
        self,
        tab_id: str | None = None,
        *,
        description: str,
        source: MechanismAuthoringSource | None = None,
        refresh_after_admission: bool = True,
    ) -> bool:
        return self._accept_visible_source_result(
            tab_id,
            description=description,
            source=source,
            refresh_after_admission=refresh_after_admission,
        ) is not None

    @staticmethod
    def _format_exact_correction_assessment(assessment: ExactCorrectionAssessment) -> str:
        """Show the core sentences about each cycle and the lines Accept adds."""

        if assessment.unavailable_reason:
            return assessment.unavailable_reason
        sections = [item.reason for item in assessment.composition.constraints if item.reason]
        if assessment.corrections:
            sections.append(
                ("Accept adds this line:" if len(assessment.corrections) == 1
                 else "Accept adds these lines:")
                + "".join(
                    f"\nparam {name} = {expression}"
                    for name, expression in assessment.corrections.items()
                )
            )
        return "\n\n".join(sections)

    def _show_exact_correction_review(
        self,
        *,
        assessment: ExactCorrectionAssessment,
        visible_source: MechanismAuthoringSource | None = None,
        action: str,
    ) -> MechanismAuthoringSource | None:
        """Show one local synchronous Accept/Cancel source review."""

        selected_assessment = assessment
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle(f"Review Exact Correction — {action}")
        layout = QtWidgets.QVBoxLayout(dialog)
        layout.addWidget(
            QtWidgets.QLabel(
                "Review the change before you continue.",
                dialog,
            )
        )
        details = QtWidgets.QPlainTextEdit(dialog)
        details.setReadOnly(True)
        details.setPlainText(self._format_exact_correction_assessment(selected_assessment))
        layout.addWidget(details)
        buttons = QtWidgets.QDialogButtonBox(dialog)
        buttons.addButton(QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        if assessment.proposal is not None:
            buttons.addButton("Accept", QtWidgets.QDialogButtonBox.ButtonRole.AcceptRole)
        if assessment.proposal is not None and visible_source is not None:
            choose_button = buttons.addButton(
                "Choose Dependent Parameters…",
                QtWidgets.QDialogButtonBox.ButtonRole.ActionRole,
            )

            def choose_basis() -> None:
                nonlocal selected_assessment
                chooser = ExactCorrectionBasisDialog(
                    assessment=selected_assessment,
                    visible_source=visible_source,
                    parent=dialog,
                )
                if chooser.exec() != QtWidgets.QDialog.DialogCode.Accepted:
                    return
                selected_assessment = chooser.selected_assessment
                details.setPlainText(self._format_exact_correction_assessment(selected_assessment))

            choose_button.clicked.connect(choose_basis)
        layout.addWidget(buttons)
        buttons.rejected.connect(dialog.reject)
        buttons.accepted.connect(dialog.accept)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return None
        return selected_assessment.proposal

    def _admit_incoming_document(
        self,
        document: ProjectDocument,
    ) -> ProjectDocument | None:
        """Review all incoming tabs once before controller publication."""

        corrections: list[tuple[MechanismTabDocument, ExactCorrectionAssessment]] = []
        blocked = False
        sections_by_tab: dict[str, str] = {}
        for tab in document.tabs:
            assessment = assess_tab_exact_correction(
                document,
                tab.tab_id,
                tab.mechanism_source,
            )
            summary = self._format_exact_correction_assessment(assessment)
            if assessment.unavailable_reason:
                blocked = True
                section = summary
            elif assessment.requires_correction:
                if assessment.proposal is None:
                    blocked = True
                else:
                    corrections.append((tab, assessment))
                section = summary
            else:
                section = "No exact correction is required for this tab."
            sections_by_tab[tab.tab_id] = f"{tab.name}\n{'=' * 40}\n{section}"
        if not corrections and not blocked:
            try:
                return validate_project_document(document)
            except Exception as exc:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Open Project Error",
                    f"The incoming project is invalid:\n\n{exc}",
                )
                return None

        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Review Incoming Project Exact Corrections")
        layout = QtWidgets.QVBoxLayout(dialog)
        layout.addWidget(
            QtWidgets.QLabel(
                "Review every incoming mechanism tab before opening the project.",
                dialog,
            )
        )
        details = QtWidgets.QPlainTextEdit(dialog)
        details.setReadOnly(True)
        details.setPlainText(
            "\n\n".join(sections_by_tab[tab.tab_id] for tab in document.tabs)
        )
        layout.addWidget(details)
        buttons = QtWidgets.QDialogButtonBox(dialog)
        buttons.addButton(QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        if not blocked:
            buttons.addButton("Accept", QtWidgets.QDialogButtonBox.ButtonRole.AcceptRole)
        if corrections and not blocked:
            tab_choice = QtWidgets.QComboBox(dialog)
            for index, (tab, _assessment) in enumerate(corrections):
                tab_choice.addItem(tab.name, index)
            choose_button = buttons.addButton(
                "Choose Dependent Parameters…",
                QtWidgets.QDialogButtonBox.ButtonRole.ActionRole,
            )

            def choose_incoming_basis() -> None:
                correction_index = int(tab_choice.currentData())
                tab, current_assessment = corrections[correction_index]
                chooser = ExactCorrectionBasisDialog(
                    assessment=current_assessment,
                    visible_source=tab.mechanism_source,
                    parent=dialog,
                )
                if chooser.exec() != QtWidgets.QDialog.DialogCode.Accepted:
                    return
                selected_assessment = chooser.selected_assessment
                corrections[correction_index] = (tab, selected_assessment)
                sections_by_tab[tab.tab_id] = (
                    f"{tab.name}\n{'=' * 40}\n"
                    + self._format_exact_correction_assessment(selected_assessment)
                )
                details.setPlainText(
                    "\n\n".join(
                        sections_by_tab[item.tab_id] for item in document.tabs
                    )
                )

            choose_button.clicked.connect(choose_incoming_basis)
            selector_row = QtWidgets.QHBoxLayout()
            selector_row.addWidget(QtWidgets.QLabel("Correction Tab:", dialog))
            selector_row.addWidget(tab_choice, stretch=1)
            layout.insertLayout(layout.count() - 1, selector_row)
        layout.addWidget(buttons)
        buttons.rejected.connect(dialog.reject)
        buttons.accepted.connect(dialog.accept)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted or blocked:
            return None

        candidate = document
        try:
            for tab, assessment in corrections:
                if assessment.proposal is None:
                    return None
                proposal = reconcile_tab_mechanism_source(
                    candidate,
                    tab.tab_id,
                    assessment.proposal,
                )
                if isinstance(proposal, MechanismSourceCollision):
                    proposal = proposal.replacement
                candidate = proposal.proposed_document
            return validate_project_document(candidate)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self,
                "Open Project Error",
                f"The incoming project correction could not be admitted:\n\n{exc}",
            )
            return None

    def _prepare_source_reconciliation(
        self,
        document,
        tab_id: str,
        visible: MechanismAuthoringSource,
        *,
        description: str,
    ) -> MechanismSourceReconciliation | None:
        """Assess exact source facts, then perform ordinary reconciliation."""

        assessment = assess_tab_exact_correction(document, tab_id, visible)
        if assessment.unavailable_reason:
            QtWidgets.QMessageBox.warning(self, "Invalid Reactions", assessment.unavailable_reason)
            return None
        source_for_admission = visible
        if assessment.requires_correction:
            corrected_proposal = self._show_exact_correction_review(
                assessment=assessment,
                visible_source=visible,
                action=description,
            )
            if corrected_proposal is None:
                return None
            source_for_admission = corrected_proposal

        try:
            proposal = reconcile_tab_mechanism_source(
                document,
                tab_id,
                source_for_admission,
            )
        except AuthoredSourceError as exc:
            QtWidgets.QMessageBox.warning(self, "Invalid Reactions", str(exc))
            return None
        if isinstance(proposal, MechanismSourceCollision):
            names = ", ".join(proposal.existing_names)
            dialog = QtWidgets.QMessageBox(self)
            dialog.setWindowTitle("Replace Initial Conditions")
            dialog.setIcon(QtWidgets.QMessageBox.Icon.Warning)
            dialog.setText(
                "The mechanism text has named initial-condition blocks for sets that "
                f"already exist: {names}.\n\nReplace those sets with the new values?"
            )
            replace_button = dialog.addButton(
                "Replace",
                QtWidgets.QMessageBox.ButtonRole.AcceptRole,
            )
            dialog.addButton(QtWidgets.QMessageBox.StandardButton.Cancel)
            dialog.exec()
            if dialog.clickedButton() is not replace_button:
                return None
            proposal = proposal.replacement
        return proposal

    def _accept_external_source(
        self,
        tab_id: str,
        source: MechanismAuthoringSource,
        *,
        description: str,
    ) -> bool:
        if not self._mutation_input_admitted(description):
            return False
        target = str(tab_id)
        runtime = self._workspace_tab_registry.tab_runtime(target)
        if (
            runtime.reactions_draft is not None
            or runtime.slider_parameter_draft
            or runtime.slider_species_drafts_by_set_id
        ):
            answer = QtWidgets.QMessageBox.question(
                self,
                "Replace Active Mechanism",
                "Loading this source affects only the active mechanism tab and "
                "will discard the unapplied Reactions and Slider drafts of that tab. "
                "Continue?",
                QtWidgets.QMessageBox.StandardButton.Yes
                | QtWidgets.QMessageBox.StandardButton.Cancel,
                QtWidgets.QMessageBox.StandardButton.Cancel,
            )
            if answer != QtWidgets.QMessageBox.StandardButton.Yes:
                return False
        if not self._accept_visible_source(
            target,
            description=description,
            source=source,
            refresh_after_admission=False,
        ):
            return False
        self._discard_slider_draft(target)
        self._refresh_tab_controls(target)
        return True

    def _load_preset_mechanism(self, preset_id: str) -> None:
        if not self._mutation_input_admitted("Load Preset Mechanism"):
            return
        try:
            from kindred.io.resources import get_preset_mechanism_source

            source = get_preset_mechanism_source(preset_id)
            if self._accept_external_source(
                self._workspace_tab_registry.active_tab_id,
                source,
                description=f"Load Preset {preset_id}",
            ):
                self.set_status_text(f"Loaded preset mechanism: {preset_id}")
        except Exception as exc:
            logger.exception("Preset load failed")
            QtWidgets.QMessageBox.critical(
                self,
                "Load Error",
                f"Failed to load preset mechanism {preset_id}:\n\n{exc}",
            )

    def _load_intervention_example(self, example_id: str) -> None:
        if not self._mutation_input_admitted("Load Intervention Example"):
            return
        try:
            from kindred.io.resources import get_intervention_example_source

            source = get_intervention_example_source(example_id)
            if self._accept_external_source(
                self._workspace_tab_registry.active_tab_id,
                source,
                description=f"Load Intervention Example {example_id}",
            ):
                self.set_status_text(f"Loaded intervention example: {example_id}")
        except Exception as exc:
            logger.exception("Intervention example load failed")
            QtWidgets.QMessageBox.critical(
                self,
                "Load Error",
                f"Failed to load intervention example {example_id}:\n\n{exc}",
            )

    def _populate_user_templates_menu(self) -> None:
        menu = self._user_templates_menu
        menu.clear()
        templates = sorted(
            self._template_manager.get_all_templates(),
            key=lambda template: template.name.casefold(),
        )
        if not templates:
            reason = self._template_manager.write_disabled_reason
            action = menu.addAction(
                "Templates unavailable" if reason else "No templates yet"
            )
            action.setEnabled(False)
            if reason:
                action.setToolTip(literal_tooltip(reason))
            else:
                action.setToolTip("Create a reusable mechanism in Template Manager.")
        for template in templates:
            label = compact_text(template.name, max_chars=60)
            action = menu.addAction(label.display.replace("&", "&&"))
            action.setToolTip(literal_tooltip(template.name))
            action.setData(template.id)
            action.triggered.connect(
                lambda _checked=False, identity=template.id: self._load_user_template(
                    identity
                )
            )

    def _load_user_template(self, template_id: str) -> None:
        template = self._template_manager.get_template(template_id)
        if template is None:
            return
        try:
            if self._accept_external_source(
                self._workspace_tab_registry.active_tab_id,
                template.source,
                description=f"Apply Template {template.name}",
            ):
                self.set_status_text(f"Applied template: {template.name}")
        except Exception as exc:
            logger.exception("Template could not be applied")
            QtWidgets.QMessageBox.critical(
                self, "Template Apply", f"The template could not be applied:\n\n{exc}"
            )

    def _open_template_manager(self) -> None:
        if not self._mutation_input_admitted("Open Template Manager"):
            return
        from kindred.gui.widgets.template_manager_dialog import TemplateManagerDialog

        source_tab = self._workspace_tab_registry.tab(
            self._workspace_tab_registry.active_tab_id
        )
        dialog = TemplateManagerDialog(
            parent=self,
            template_manager=self._template_manager,
            current_mechanism_source=self._visible_source_for_tab(source_tab.tab_id),
            source_preflight=lambda source: self._preflight_template_source(
                source_tab.tab_id,
                source,
            ),
            destination_apply=lambda source: self._accept_external_source(
                source_tab.tab_id,
                source,
                description="Apply Template",
            ),
        )
        dialog.exec()

    def _preflight_template_source(
        self,
        source_tab_id: str,
        source: MechanismAuthoringSource,
    ) -> MechanismAuthoringSource | None:
        """Scientifically admit a template candidate in its source tab context.

        Returns ``None`` when the user cancels the correction review.
        """

        document = self._project_session.document
        target = str(source_tab_id)
        assessment = assess_tab_exact_correction(document, target, source)
        if assessment.unavailable_reason:
            raise AuthoredSourceError(assessment.unavailable_reason)
        source_for_admission = source
        if assessment.requires_correction:
            corrected_proposal = self._show_exact_correction_review(
                assessment=assessment,
                visible_source=source,
                action="Save Template",
            )
            if corrected_proposal is None:
                return None
            source_for_admission = corrected_proposal
        prepare_authored_source_admission(
            source_for_admission,
            document.tab(target).scientific_settings,
        )
        return source_for_admission

    # ------------------------------------------------------------------
    # Dataset intents
    # ------------------------------------------------------------------
    @QtCore.Slot(str, object)
    def _start_dataset_import(
        self,
        origin_tab_id: str,
        configs: tuple[ImportConfig, ...],
    ) -> None:
        """Admit one configured import intent into the global staging owner."""
        if not self._mutation_input_admitted("Import Dataset"):
            return
        dataset_count = sum(len(config.plans) for config in configs)
        label = f"{dataset_count} dataset{'s' if dataset_count != 1 else ''}"
        self._dataset_import_progress_dialog = QtWidgets.QProgressDialog(
            f"Loading {label}…",
            "Cancel",
            0,
            100,
            self,
        )
        self._dataset_import_progress_dialog.setWindowTitle("Loading Datasets")
        self._dataset_import_progress_dialog.setWindowModality(Qt.WindowModal)
        self._dataset_import_progress_dialog.setMinimumDuration(0)
        self._dataset_import_progress_dialog.canceled.connect(
            self._cancel_dataset_import
        )
        self._dataset_import_progress_dialog.show()
        try:
            self._dataset_import_staging.start_import(
                origin_tab_id=origin_tab_id,
                configs=configs,
            )
        except Exception as exc:
            self._close_import_progress_dialog()
            logger.exception("Dataset import could not start")
            QtWidgets.QMessageBox.critical(
                self,
                "Dataset Import",
                str(exc),
            )

    @QtCore.Slot(int)
    def _on_import_progress(self, percent: int) -> None:
        dialog = self._dataset_import_progress_dialog
        if dialog is not None:
            dialog.setValue(int(percent))

    @QtCore.Slot(str)
    def _on_import_failure(self, message: str) -> None:
        self.set_status_text(message)
        QtWidgets.QMessageBox.critical(
            self,
            "Dataset Import",
            message,
        )

    @QtCore.Slot()
    def _on_import_cancellation(self) -> None:
        self.set_status_text("Dataset import canceled.")

    @QtCore.Slot()
    def _cancel_dataset_import(self) -> None:
        self._dataset_import_staging.cancel_active()

    def _close_import_progress_dialog(self) -> None:
        dialog = self._dataset_import_progress_dialog
        if dialog is None:
            return
        try:
            dialog.canceled.disconnect(self._cancel_dataset_import)
        except (TypeError, RuntimeError):
            pass
        dialog.close()
        self._dataset_import_progress_dialog = None

    @QtCore.Slot(bool)
    def _on_import_busy_changed(self, busy: bool) -> None:
        for bundle in self._tab_surfaces.values():
            bundle.data_panel.set_import_busy(bool(busy))
        if not busy:
            self._close_import_progress_dialog()
        self._refresh_actions()

    @QtCore.Slot(object)
    def _commit_import_completion(
        self,
        completion: DatasetImportCompletion,
    ) -> None:
        if not self._mutation_input_admitted("Commit Dataset Import"):
            self.set_status_text(
                "Dataset import completed after Close was requested, so publication was skipped."
            )
            return
        try:
            batch_proposal = build_raw_record_batch_proposal(
                completion,
                existing_records=self._project_session.document.datasets,
            )
            document = self._project_session.document.add_datasets(
                batch_proposal.records,
                sources=batch_proposal.sources,
            )
            self._workspace_tab_registry.replace_document(
                document,
                expected_token=self._project_session.capture_content_token(),
            )
        except Exception as exc:
            logger.exception("Raw dataset commit failed")
            QtWidgets.QMessageBox.critical(
                self,
                "Dataset Import",
                "The completed import could not be committed. "
                f"{exc}",
            )
            self.set_status_text(
                "Dataset import publication was rejected. Staging is idle and "
                "the next import is available."
            )
            self._refresh_project_state()
            return

        self._refresh_dataset_views()
        message = f"Imported {len(batch_proposal.records)} raw dataset(s)."
        origin_tab_id = batch_proposal.origin_tab_id
        if origin_tab_id not in self._tab_surfaces:
            self.set_status_text(
                message
                + " The initiating mechanism no longer exists. Raw data "
                "remains committed and optional mapping was skipped."
            )
            return
        proposal = self._accept_visible_source_result(
            origin_tab_id,
            description="Import Data",
        )
        if proposal is None:
            self.set_status_text(
                message
                + " Optional mapping was canceled or blocked by visible "
                "Reactions, so it was skipped."
                + " Raw data remains committed and unmapped. Fix Reactions, then"
                + " use Map."
            )
            return

        self.set_status_text(
            message + " Raw data was committed before optional mapping."
        )
        mapping_changed = False
        try:
            tab = self._workspace_tab_registry.tab(origin_tab_id)
            previous_set_ids = tuple(
                item.set_id for item in tab.initial_condition_sets
            )
            model_targets = canonical_observation_target_names(tab)
            if not model_targets:
                raise DatasetMappingError(
                    "Optional mapping requires a valid mechanism with observation targets."
                )
            review = DatasetMappingBatchDialog(
                document=self._project_session.document,
                records=batch_proposal.records,
                tab=tab,
                model_targets=model_targets,
                parent=self,
            )
            try:
                mapping_expected_token = self._project_session.capture_content_token()
                if review.exec() != QtWidgets.QDialog.DialogCode.Accepted:
                    self.set_status_text(
                        message + " Mapping canceled. Raw data remains unmapped."
                    )
                    return
                batch = review.proposal()
            finally:
                review.deleteLater()
            candidate_tab = apply_dataset_mapping_batch(
                document=self._project_session.document,
                tab=tab,
                batch=batch,
                model_targets=model_targets,
            )
            result = self._workspace_tab_registry.replace_tab(
                origin_tab_id,
                candidate_tab,
                expected_token=mapping_expected_token,
            )
            if result is None:
                self.set_status_text(
                    message + " Mapping was not published. Raw data remains unmapped."
                )
                return
            mapping_changed = result != mapping_expected_token
        except Exception as exc:
            logger.exception("Batch dataset mapping failed")
            QtWidgets.QMessageBox.warning(
                self,
                "Dataset Mapping",
                f"The reviewed mapping was not applied:\n\n{exc}",
            )
            self.set_status_text(
                message + " Mapping failed validation. Raw data remains unmapped."
            )
            return
        if mapping_changed:
            current_tab = self._workspace_tab_registry.tab(origin_tab_id)
            if (
                tuple(item.set_id for item in current_tab.initial_condition_sets)
                != previous_set_ids
            ):
                self._tab_surfaces[origin_tab_id].initial_actions.render()
        self._refresh_dataset_views((origin_tab_id,))
        self.set_status_text(
            message + f" Reviewed mapping applied for {len(batch.proposals)} dataset(s)."
        )

    @QtCore.Slot(str, str)
    def _rename_dataset(self, dataset_id: str, display_name: str) -> None:
        if not self._mutation_input_admitted("Rename Dataset"):
            return
        changed = False
        try:
            name = validate_unique_dataset_name(
                display_name,
                existing=(
                    (record.dataset_id, record.display_name)
                    for record in self._project_session.document.datasets
                ),
                excluding_dataset_id=dataset_id,
            )
            candidate = self._project_session.document.rename_dataset(
                dataset_id,
                name,
            )
            expected_token = self._project_session.capture_content_token()
            result = self._workspace_tab_registry.replace_document(
                candidate,
                expected_token=expected_token,
            )
            changed = result != expected_token
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Rename Dataset", str(exc))
        if changed:
            self._refresh_dataset_views()

    @QtCore.Slot(str)
    def _delete_dataset(self, dataset_id: str) -> None:
        if not self._mutation_input_admitted("Delete Dataset"):
            return
        presentations = dict(self._workspace_tab_registry.presentation_items())
        plan = plan_global_dataset_delete(
            dataset_id=dataset_id,
            links_by_tab={
                tab.tab_id: tab.dataset_links
                for tab in self._project_session.document.tabs
            },
            overlays_by_tab={
                tab_id: state.overlay_raw_observation_ids_by_dataset
                for tab_id, state in presentations.items()
            },
        )
        if plan.referenced:
            answer = QtWidgets.QMessageBox.question(
                self,
                "Delete Dataset",
                "This dataset is referenced by one or more mechanism tabs. "
                "Delete it globally and clear every initial-condition assignment, "
                "observation mapping, and overlay?",
                QtWidgets.QMessageBox.StandardButton.Yes
                | QtWidgets.QMessageBox.StandardButton.Cancel,
                QtWidgets.QMessageBox.StandardButton.Cancel,
            )
            if answer != QtWidgets.QMessageBox.StandardButton.Yes:
                return
        candidate = self._project_session.document.remove_dataset(
            dataset_id
        )
        self._workspace_tab_registry.replace_document(
            candidate,
            expected_token=self._project_session.capture_content_token(),
        )
        self._refresh_dataset_views()

    @QtCore.Slot(str, str)
    def _open_spreadsheet(self, origin_tab_id: str, selected_id: str) -> None:
        if not self._mutation_input_admitted("Create from Data") or self._spreadsheet_editor is not None:
            return
        from kindred.gui.widgets.spreadsheet_editor import SpreadsheetEditorDialog, SpreadsheetSelectionDialog

        if not self._accept_visible_source(origin_tab_id, description="Create from Data"):
            return
        document = self._project_session.document
        tab = document.tab(origin_tab_id)
        if not canonical_observation_target_names(tab):
            self.set_status_text("Create from Data requires an accepted mechanism with observation targets.")
            return
        selection = SpreadsheetSelectionDialog(document.datasets, selected_id, parent=self)
        try:
            if selection.exec() != QtWidgets.QDialog.Accepted:
                return
            selected = tuple(document.dataset(identity) for identity in selection.selected_ids())
        finally:
            selection.deleteLater()
        if not selected:
            return
        names, occupied = [], {record.display_name.casefold() for record in document.datasets}
        for record in selected:
            base = f"{record.display_name} — processed"
            name, number = base, 2
            while name.casefold() in occupied:
                name = f"{base} ({number})"
                number += 1
            occupied.add(name.casefold())
            names.append(name)
        editor = SpreadsheetEditorDialog(selected, tuple(names), parent=self)
        self._spreadsheet_editor = editor
        editor.outputsReady.connect(lambda records: self._create_spreadsheet_datasets(editor, origin_tab_id, records))
        editor.failed.connect(self.set_status_text)
        editor.settled.connect(self._spreadsheet_settled)
        editor.show()
        self._refresh_actions()

    def _create_spreadsheet_datasets(self, editor, origin_tab_id, records) -> None:
        if not self._mutation_input_admitted("Create from Data"):
            return
        # The worksheets are detached from the project, so outputs are reviewed
        # and published against the project as it is now, including visible
        # Reactions admitted like every other mapping action.
        if all(item.tab_id != origin_tab_id for item in self._project_session.document.tabs):
            QtWidgets.QMessageBox.warning(editor, "Create from Data", "The workspace tab that opened this editor was closed, so the new datasets have no tab to join.")
            return
        if not self._accept_visible_source(origin_tab_id, description="Create from Data"):
            return
        document = self._project_session.document
        token = self._project_session.capture_content_token()
        tab = document.tab(origin_tab_id)
        targets = canonical_observation_target_names(tab)
        if not targets:
            QtWidgets.QMessageBox.warning(editor, "Create from Data", "Create from Data requires an accepted mechanism with observation targets.")
            return
        try:
            existing = [(record.dataset_id, record.display_name) for record in document.datasets]
            for record in records:
                validate_unique_dataset_name(record.display_name, existing=existing)
                existing.append((record.dataset_id, record.display_name))
            candidate = document.add_datasets(records, sources=())
            review = DatasetMappingBatchDialog(document=candidate, records=records, tab=tab,
                                               model_targets=targets, create_only=True, parent=editor)
            try:
                if review.exec() != QtWidgets.QDialog.Accepted:
                    return
                batch = review.proposal()
            finally:
                review.deleteLater()
            mapped = apply_dataset_mapping_batch(document=candidate, tab=tab, batch=batch, model_targets=targets)
            candidate = validate_project_document(candidate.replace_tab(mapped))
            if not self._mutation_input_admitted("Create from Data"):
                return
            published = self._workspace_tab_registry.replace_document(candidate, expected_token=token)
            if published is None:
                QtWidgets.QMessageBox.warning(editor, "Create from Data", "The project changed during review. Review and create again.")
                return
        except (ValueError, TypeError, KeyError) as exc:
            QtWidgets.QMessageBox.warning(editor, "Create from Data", str(exc))
            return
        editor.finish_creation()
        bundle = self._tab_surfaces[origin_tab_id]
        bundle.initial_actions.render()
        bundle.slider_binding.render()
        self._refresh_dataset_views()
        self.set_status_text(f"Created {len(records)} experimental datasets, each with a new mapped initial-condition set.")

    @QtCore.Slot()
    def _spreadsheet_settled(self) -> None:
        editor = self._spreadsheet_editor
        self._spreadsheet_editor = None
        if editor is not None:
            editor.deleteLater()
        self._refresh_actions()
        self.project_controller.notify_close_progress()

    @QtCore.Slot(str)
    def _export_original_dataset(self, dataset_id: str) -> None:
        if not self._mutation_input_admitted("Export Original Dataset"):
            return
        document = self._project_session.document
        try:
            record = document.dataset(dataset_id)
            if record.source_id is None:
                self.set_status_text("This dataset was created in Kindred and has no original imported file.")
                return
            source = document.imported_source(record.source_id)
        except KeyError:
            return
        try:
            start_directory = resolve_export_start_dir(
                self._last_export_directory
            )
        except OSError as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "Export Original Dataset",
                str(exc),
            )
            return
        path, _selected = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export Original Dataset",
            os.path.join(
                start_directory,
                source.original_filename,
            ),
            "All Files (*)",
        )
        if not path:
            return
        if self._xlsx_export_owner.owns_destination(path):
            QtWidgets.QMessageBox.critical(
                self,
                "Export Original Dataset",
                "Cannot overwrite the destination of an active XLSX export.",
            )
            return
        if self._scan_owner.owns_destination(path):
            QtWidgets.QMessageBox.critical(
                self, "Export Original Dataset", "Cannot overwrite the destination of an active Chemical Space Scan.")
            return
        if self.project_controller.is_current_project_path(path):
            QtWidgets.QMessageBox.critical(
                self,
                "Export Original Dataset",
                "Cannot overwrite the currently open project archive.",
            )
            return
        self._remember_export_directory(path)
        try:
            exported = export_original_source(source, path)
            self.set_status_text(f"Exported original dataset: {exported}")
        except (DatasetMappingError, OSError) as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "Export Original Dataset",
                file_error_text(exc, path),
            )

    def _map_dataset(
        self,
        tab_id: str,
        dataset_id: str,
        action: str,
        set_id: str | None,
        occupied_choice: str | None,
    ) -> None:
        if not self._mutation_input_admitted("Map Dataset"):
            return
        changed = False
        previous_set_ids: tuple[str, ...] = ()
        try:
            choice = DatasetMappingChoice(action)
            if not self._accept_visible_source(tab_id, description="Map Dataset"):
                return
            expected_token = self._project_session.capture_content_token()
            tab = self._workspace_tab_registry.tab(tab_id)
            previous_set_ids = tuple(
                item.set_id for item in tab.initial_condition_sets
            )
            dataset = self._project_session.document.dataset(dataset_id)
            model_targets = canonical_observation_target_names(tab)
            if choice is DatasetMappingChoice.CREATE_SET:
                if not model_targets:
                    raise ValueError(
                        "Apply a valid mechanism with observation targets before creating a set from observations."
                    )
                dialog = ObservationMappingDialog(
                    dataset,
                    model_targets,
                    current_mapping=(
                        tab.dataset_link(dataset.dataset_id).observation_mapping
                        if tab.dataset_link(dataset.dataset_id) is not None else ()
                    ),
                    mechanism_species=canonical_species_names(tab),
                    create_set=True,
                    title="Confirm Observation Mapping",
                    parent=self,
                )
                try:
                    if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
                        return
                    proposal = DatasetMappingProposal(
                        dataset_id=dataset.dataset_id,
                        action=choice,
                        observation_mapping=dialog.editor.confirmed_mapping(),
                        initial_concentration_source=dialog.editor.creation_source(),
                    )
                finally:
                    dialog.deleteLater()
            elif choice is DatasetMappingChoice.MAP_EXISTING:
                if not set_id:
                    raise ValueError("Map Existing requires an initial-condition set.")
                if occupied_choice == OccupiedSetChoice.CANCEL.value:
                    return
                link = tab.dataset_link(dataset.dataset_id)
                proposal = DatasetMappingProposal(
                    dataset_id=dataset.dataset_id,
                    action=choice,
                    target_set_id=set_id,
                    occupied_choice=occupied_choice or None,
                    observation_mapping=(
                        link.observation_mapping if link is not None else ()
                    ),
                )
            else:
                return
            # The same fold as batch import mapping owns occupancy and naming.
            candidate_tab = apply_dataset_mapping_batch(
                document=self._project_session.document,
                tab=tab,
                batch=DatasetMappingBatchProposal(tab_id=tab.tab_id, proposals=(proposal,)),
                model_targets=model_targets,
            )
            result = self._workspace_tab_registry.replace_tab(
                tab_id,
                candidate_tab,
                expected_token=expected_token,
            )
            if result is not None:
                changed = result != expected_token
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Dataset Mapping", str(exc))
        if not changed:
            return
        tab = self._workspace_tab_registry.tab(tab_id)
        if (
            tuple(item.set_id for item in tab.initial_condition_sets)
            != previous_set_ids
        ):
            bundle = self._tab_surfaces[tab_id]
            bundle.initial_actions.render()
            bundle.slider_binding.render()
        self._refresh_dataset_views((tab_id,))

    def _edit_observation_mapping(self, tab_id: str, dataset_id: str) -> None:
        if not self._mutation_input_admitted("Edit Observation Mapping"):
            return
        changed = False
        try:
            if not self._accept_visible_source(
                tab_id,
                description="Edit Observation Mapping"
            ):
                return
            expected_token = self._project_session.capture_content_token()
            tab = self._workspace_tab_registry.tab(tab_id)
            dataset = self._project_session.document.dataset(dataset_id)
            model_targets = canonical_observation_target_names(tab)
            current = tab.dataset_link(dataset.dataset_id)
            dialog = ObservationMappingDialog(
                dataset,
                model_targets,
                current_mapping=current.observation_mapping if current is not None else (),
                title="Observation Mapping",
                parent=self,
            )
            try:
                if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
                    return
                mapping = dialog.editor.confirmed_mapping()
            finally:
                dialog.deleteLater()
            candidate = with_confirmed_observation_mapping(
                tab=tab,
                dataset=dataset,
                model_targets=model_targets,
                confirmed_mapping=mapping,
            )
            result = self._workspace_tab_registry.replace_tab(
                tab_id,
                candidate,
                expected_token=expected_token,
            )
            if result is not None:
                changed = result != expected_token
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Observation Mapping", str(exc))
        if changed:
            self._refresh_dataset_views((tab_id,))

    def _set_overlay_selection(
        self,
        tab_id: str,
        dataset_id: str,
        raw_observation_ids: tuple[str, ...],
    ) -> None:
        if not self._mutation_input_admitted("Change Dataset Overlay"):
            return
        state = self._workspace_tab_registry.presentation(tab_id)
        overlays = dict(state.overlay_raw_observation_ids_by_dataset)
        if raw_observation_ids:
            overlays[dataset_id] = raw_observation_ids
        else:
            overlays.pop(dataset_id, None)
        self._workspace_tab_registry.set_presentation(
            tab_id,
            replace(
                state,
                overlay_raw_observation_ids_by_dataset=tuple(overlays.items()),
            ),
        )
        self._tab_surfaces[tab_id].overlay_panel.render_overlay_selection(
            dataset_id=dataset_id,
            raw_observation_ids=raw_observation_ids,
        )
        self._render_results_for_tab(tab_id)
        self._refresh_project_state()

    def _set_reference_lines_visible(self, tab_id: str, visible: bool) -> None:
        if not self._mutation_input_admitted("Change Reference Lines"):
            return
        state = self._workspace_tab_registry.presentation(tab_id)
        self._workspace_tab_registry.set_presentation(
            tab_id,
            replace(state, reference_lines_visible=bool(visible)),
        )
        self._refresh_project_state()
        self._render_results_for_tab(tab_id)

    def _set_intervention_annotations_visible(
        self,
        tab_id: str,
        visible: bool,
    ) -> None:
        if not self._mutation_input_admitted(
            "Change Intervention Indicators"
        ):
            return
        self._workspace_tab_registry.set_intervention_annotations_visible(
            tab_id,
            visible,
        )
        self._refresh_project_state()
        self._render_results_for_tab(tab_id)

    def _set_intervention_labels_visible(
        self,
        tab_id: str,
        visible: bool,
    ) -> None:
        if not self._mutation_input_admitted("Change Intervention Labels"):
            return
        self._workspace_tab_registry.set_intervention_labels_visible(tab_id, visible)
        self._refresh_project_state()
        self._render_results_for_tab(tab_id)

    def _set_x_series(
        self,
        tab_id: str,
        selection: AxisSeriesSelection,
    ) -> None:
        if not self._mutation_input_admitted("Change X Series"):
            return
        state = self._workspace_tab_registry.presentation(tab_id)
        self._workspace_tab_registry.set_presentation(
            tab_id,
            replace(state, selected_x_series=selection),
        )
        self._refresh_project_state()
        self._render_results_for_tab(tab_id)

    def _set_y_series(self, tab_id: str, selections: object) -> None:
        if not self._mutation_input_admitted("Change Y Series"):
            return
        selected = tuple(selections or ())
        state = self._workspace_tab_registry.presentation(tab_id)
        self._workspace_tab_registry.set_presentation(
            tab_id,
            replace(
                state,
                selected_y_series=selected,
                selected_y_uses_default_all=False,
            ),
        )
        self._refresh_project_state()
        self._render_results_for_tab(tab_id)

    def _set_plot_presentation(
        self,
        tab_id: str,
        option: str,
        value: bool | float | str,
    ) -> None:
        if not self._mutation_input_admitted("Change Plot Presentation"):
            return
        state = self._workspace_tab_registry.presentation(tab_id)
        changes = {"grid_x": value, "grid_y": value} if option == "show_grid" else {option: value}
        next_state = replace(state, **changes)
        self._workspace_tab_registry.set_presentation(
            tab_id,
            next_state,
        )
        self._workspace.plot_for_tab(tab_id).set_persisted_plot_state(
            next_state,
        )
        self._refresh_project_state()
        self._render_results_for_tab(tab_id)

    def _populate_recent_menu(self) -> None:
        menu = self._recent_menu
        if menu is None:
            return
        menu.clear()
        recent = self.config_controller.recent_projects()
        for path in recent:
            action = menu.addAction(os.path.basename(path) or path)
            action.setToolTip(path)
            action.triggered.connect(
                lambda _checked=False, captured=path: self._open_recent_project(
                    captured
                )
            )
        if recent:
            menu.addSeparator()
        clear_action = menu.addAction("Clear Recent Projects")
        clear_action.setToolTip(
            "Clear the recent-project list without deleting any project files."
        )
        clear_action.setEnabled(bool(recent))
        clear_action.triggered.connect(self.config_controller.clear_recent_files)

    def _active_project_work(self) -> bool:
        """Work that blocks project actions; an open spreadsheet editor counts
        even when unchanged (Close disposition counts it only with changes)."""

        return bool(
            self._simulation_work_active()
            or self._fit_session_owner.active
            or self._dataset_import_staging.busy
            or self._xlsx_export_owner.is_exporting
            or self._scan_owner.busy
            or any(owner.busy for owner in self._calculator_owners)
            or self._spreadsheet_editor is not None
        )

    def _project_actions_idle(self, action: str) -> bool:
        if not self._mutation_input_admitted(action):
            return False
        if self._active_project_work():
            self.set_status_text(
                f"{action} rejected: active work must finish first."
            )
            return False
        if not self._computation_gate.is_idle:
            self.set_status_text(
                f"{action} rejected: another computation is active."
            )
            return False
        return True

    def _new_project(self) -> bool:
        if not self._project_actions_idle("New Project"):
            return False
        return bool(self.project_controller.new_project())

    def _open_project(self) -> bool:
        if not self._project_actions_idle("Open Project"):
            return False
        return bool(self.project_controller.load_project())

    def _open_recent_project(self, path: str) -> bool:
        if not self._project_actions_idle("Open Project"):
            return False
        return bool(self.project_controller.load_recent_project(path))

    def _reset_layout(self) -> None:
        if not self._mutation_input_admitted("Reset Layout"):
            return
        self._workspace.reset_visible_dock_layout()
        self.set_status_text("Panel layout reset.")

    def _restore_default_panel_layout(self) -> None:
        if not self._mutation_input_admitted("Restore Default Panel Layout"):
            return
        self._workspace.restore_default_panel_layout()
        self.set_status_text("Default panel layout restored.")

    # ------------------------------------------------------------------
    # Menus and preserved shell actions
    # ------------------------------------------------------------------
    def _create_menus(self) -> None:
        bar = self.menuBar()
        file_menu = bar.addMenu("&File")
        file_menu.setToolTipsVisible(True)
        self._file_menu = file_menu
        self._new_project_action = self._add_action(
            file_menu,
            "New Project",
            self._new_project,
            QtGui.QKeySequence.StandardKey.New,
            "newProjectAction",
            tooltip="Start a new project.",
        )
        self._open_project_action = self._add_action(
            file_menu,
            "Open Project…",
            self._open_project,
            QtGui.QKeySequence.StandardKey.Open,
            "openProjectAction",
            tooltip="Open a saved Kindred project.",
        )
        self._recent_menu = file_menu.addMenu("Open Recent")
        self._recent_menu.setToolTipsVisible(True)
        self._recent_menu.menuAction().setToolTip(
            "Reopen a recent project or clear the list."
        )
        self._recent_menu.setObjectName("recentProjectsMenu")
        self._populate_recent_menu()
        file_menu.addSeparator()
        self._add_action(
            file_menu,
            "Save Project",
            self.project_controller.save_project,
            QtGui.QKeySequence.StandardKey.Save,
            "saveProjectAction",
            tooltip="Save the current project.",
        )
        self._add_action(
            file_menu,
            "Save Project As…",
            self.project_controller.save_project_as,
            QtGui.QKeySequence.StandardKey.SaveAs,
            "saveProjectAsAction",
            tooltip="Save the current project under a different name or location.",
        )
        self._load_data_action = self._selected_bundle().data_panel.import_action
        file_menu.addAction(self._load_data_action)
        self._xlsx_export_action = self._add_action(
            file_menu,
            "Export Results XLSX…",
            self._export_xlsx,
            "Ctrl+E",
            "exportResultsXlsxAction",
            tooltip=(
                "Export the simulation results for this tab and the linked "
                "experimental data to an Excel workbook."
            ),
        )
        file_menu.addSeparator()
        self._add_action(
            file_menu,
            "Exit",
            self.close,
            QtGui.QKeySequence.StandardKey.Quit,
            "exitAction",
            tooltip="Close Kindred.",
        )

        edit_menu = bar.addMenu("&Edit")
        edit_menu.setToolTipsVisible(True)
        self._edit_menu = edit_menu
        self._undo_action = self._add_action(
            edit_menu,
            "Undo",
            lambda: self._selected_bundle().editor.undo_focused_text(),
            QtGui.QKeySequence.StandardKey.Undo,
            "undoAction",
            tooltip="Undo the last text change in the focused Reactions or Notes editor.",
        )
        self._redo_action = self._add_action(
            edit_menu,
            "Redo",
            lambda: self._selected_bundle().editor.redo_focused_text(),
            QtGui.QKeySequence.StandardKey.Redo,
            "redoAction",
            tooltip="Restore the last undone text change in the focused Reactions or Notes editor.",
        )
        view_menu = bar.addMenu("&View")
        view_menu.setToolTipsVisible(True)
        self._view_menu = view_menu
        panels = view_menu.addMenu("Panels")
        panels.setToolTipsVisible(True)
        panels.menuAction().setToolTip("Choose which panels are visible.")
        panel_tooltips = {
            "mechanism": "Show or hide the mechanism editor.",
            "interactive_sliders": "Show or hide the parameter and concentration sliders.",
            "batch_initial_conditions": "Show or hide the initial-condition sets.",
            "data": "Show or hide imported datasets and their mappings.",
            "analysis": "Show or hide model parameters and simulation statistics.",
            "symbolic_calculator": "Show or hide the symbolic calculation panel.",
        }
        for spec in DOCK_SPECS:
            dock = self._workspace.dock_for_identity(spec.identity_key)
            action = dock.toggleViewAction()
            action.setText(spec.title)
            action.setObjectName(f"{spec.identity_key}PanelAction")
            action.setToolTip(panel_tooltips[spec.identity_key])
            panels.addAction(action)
        view_menu.addSeparator()
        reset_layout_action = self._add_action(
            view_menu,
            "Reset Layout",
            self._reset_layout,
            "Ctrl+Shift+R",
            "resetLayoutAction",
        )
        reset_layout_action.setToolTip(
            "Restore the standard positions of currently visible panels "
            "without changing panel visibility."
        )
        restore_default_action = self._add_action(
            view_menu,
            "Restore Default Panel Layout",
            self._restore_default_panel_layout,
            None,
            "restoreDefaultPanelLayoutAction",
        )
        restore_default_action.setToolTip(
            "Restore the factory panel arrangement and default panel visibility."
        )
        self._dark_mode_action = self._add_action(
            view_menu,
            "Dark Mode",
            self._toggle_theme,
            "Ctrl+Shift+D",
            "darkModeAction",
            checkable=True,
            tooltip="Use a dark theme. Turn this off to return to the light theme.",
        )

        examples = bar.addMenu("&Examples")
        examples.setToolTipsVisible(True)
        self._examples_menu = examples
        presets = examples.addMenu("Preset Mechanisms")
        presets.setToolTipsVisible(True)
        presets.menuAction().setToolTip(
            "Replace the current mechanism with a built-in example."
        )
        for preset_id, title in PRESET_MECHANISM_TITLES.items():
            action = presets.addAction(f"{preset_id}: {title}")
            action.setData(preset_id)
            action.setObjectName(f"loadPreset{preset_id}Action")
            action.triggered.connect(
                lambda _checked=False, value=preset_id: self._load_preset_mechanism(
                    value
                )
            )
        interventions = examples.addMenu("Intervention Examples")
        interventions.setToolTipsVisible(True)
        interventions.menuAction().setToolTip(
            "Replace the current mechanism with an example of scheduled or "
            "condition-triggered changes."
        )
        try:
            intervention_specs = get_all_intervention_example_specs()
        except InterventionExampleCatalogError as exc:
            unavailable = interventions.addAction(str(exc))
            unavailable.setObjectName("interventionExampleCatalogUnavailableAction")
            unavailable.setEnabled(False)
            unavailable.setToolTip(literal_tooltip(
                f"The bundled intervention examples could not be loaded.\n{exc}"
            ))
        else:
            for spec in intervention_specs:
                example_id = spec.id
                title = spec.title
                text = f"{example_id}: {title}" if title else example_id
                action = interventions.addAction(text)
                action.setData(example_id)
                action.setObjectName(f"loadInterventionExample{example_id}Action")
                action.triggered.connect(
                    lambda _checked=False, value=example_id: self._load_intervention_example(
                        value
                    )
                )
        examples.addSeparator()
        self._add_action(
            examples,
            "Template Manager…",
            self._open_template_manager,
            "Ctrl+T",
            "templateManagerAction",
            tooltip="Create, edit, organize, and reuse saved mechanism templates.",
        )
        self._user_templates_menu = examples.addMenu("User-Created Templates")
        self._user_templates_menu.setToolTipsVisible(True)
        self._user_templates_menu.menuAction().setToolTip(
            "Replace the current mechanism with one of your saved templates."
        )
        self._user_templates_menu.aboutToShow.connect(self._populate_user_templates_menu)

        simulation = bar.addMenu("&Simulation")
        simulation.setToolTipsVisible(True)
        self._simulation_menu = simulation
        self._run_action = self._add_action(
            simulation,
            "Run",
            lambda: self._start_selected_run(),
            "Ctrl+R",
            "runSimulationAction",
            tooltip="Run the current mechanism for the selected initial-condition sets.",
        )
        self._run_action.setShortcuts(["Ctrl+R", "F5"])
        self._scan_action = self._add_action(
            simulation, "Chemical Space Scan…", self._open_chemical_space_scan,
            None, "chemicalSpaceScanAction",
            tooltip="Open the scan window to configure simulations across combinations of model inputs.",
        )
        self._stop_action = self._add_action(
            simulation,
            "Stop",
            lambda _checked=False: self._stop_simulation(),
            "Esc",
            "stopSimulationAction",
            tooltip=(
                "Stop simulation and preview work across all tabs and cancel "
                "an active Chemical Space Scan."
            ),
        )
        self._add_action(
            simulation,
            "Symbolic Calculator…",
            lambda: self._open_symbolic_calculator(
                self._workspace_tab_registry.active_tab_id
            ),
            None,
            "openSymbolicCalculatorAction",
            tooltip="Open the calculator to work with expressions and equations for this mechanism.",
        )
        self._add_action(
            simulation,
            "Inspect Mechanism…",
            lambda: self._open_mechanism_inspector(
                self._workspace_tab_registry.active_tab_id
            ),
            None,
            "inspectMechanismAction",
            tooltip="Examine the reactions, parameters, equations, and interventions of the current mechanism.",
        )
        simulation.addSeparator()
        self._add_action(
            simulation,
            "Simulation Settings…",
            self._open_simulation_settings,
            None,
            "simulationSettingsAction",
            tooltip=(
                "Configure mechanism settings, solver tolerances, stopping "
                "behavior, Run Defaults for New Projects and Tabs, previews, "
                "parallel execution, and result caching."
            ),
        )

        fitting = bar.addMenu("&Fitting")
        fitting.setToolTipsVisible(True)
        self._fitting_menu = fitting
        self._add_action(
            fitting,
            "Fitting Defaults…",
            self._configure_fitting,
            None,
            "fittingDefaultsAction",
            tooltip="Choose default fitting settings for the current project or new projects.",
        )
        self._global_fit_action = self._add_action(
            fitting,
            "Global Fit…",
            self._run_global_fit,
            "Ctrl+Shift+F",
            "globalFitAction",
            tooltip="Configure a fit of the current mechanism to experimental datasets.",
        )

        help_menu = bar.addMenu("&Help")
        help_menu.setToolTipsVisible(True)
        self._add_action(
            help_menu, "Interactive Tutorials…", lambda: self._tutorials.open(),
            None, "tutorialsAction",
            tooltip="Choose a guided lesson or return to your current tutorial.",
        )
        self._add_action(
            help_menu,
            "Documentation",
            self._open_docs,
            QtGui.QKeySequence.StandardKey.HelpContents,
            "documentationAction",
            tooltip="Online documentation is not available yet.",
        )
        self._add_action(
            help_menu,
            "Panel Layout Tips…",
            self._show_panel_layout_tips,
            None,
            "panelLayoutTipsAction",
            tooltip="Learn how to move, restore, and arrange panels.",
        )
        self._add_action(
            help_menu,
            "Diagnostic Log…",
            self._show_diagnostic_log,
            None,
            "diagnosticLogAction",
            tooltip="View the current diagnostic log for troubleshooting.",
        )
        help_menu.addSeparator()
        self._add_action(
            help_menu,
            "About",
            self._show_about,
            None,
            "aboutAction",
            tooltip="View version, credits, and license information for Kindred.",
        )

    def _add_action(
        self,
        menu: QtWidgets.QMenu,
        text: str,
        callback: Callable,
        shortcut,
        object_name: str,
        *,
        checkable: bool = False,
        tooltip: str = "",
    ) -> QtGui.QAction:
        action = menu.addAction(text)
        action.setObjectName(object_name)
        action.setCheckable(checkable)
        action.setToolTip(tooltip)
        if shortcut is not None:
            action.setShortcut(shortcut)
        action.triggered.connect(callback)
        return action

    def _open_simulation_settings(self) -> None:
        if not self._mutation_input_admitted("Configure Simulation Settings"):
            return
        tab_id = self._workspace_tab_registry.active_tab_id
        tab = self._workspace_tab_registry.tab(tab_id)
        dialog = SimulationSettingsDialog(
            self,
            preview_points=self.config_controller.preview_points(),
            preview_delay_ms=self.config_controller.preview_delay_ms(),
        )
        dialog.set_scientific_settings(tab.scientific_settings)
        previous_run_defaults = self.config_controller.application_run_defaults()
        dialog.set_new_tab_run_defaults(previous_run_defaults)
        dialog.set_stopping_declarations(
            () if tab.prepared_dsl is None
            else tab.prepared_dsl.authored_stop_declarations()
        )
        current_max = self.config_controller.run_user_maximum_preference()
        dialog.set_run_user_maximum(current_max)
        dialog.set_result_cache_capacity(
            self.config_controller.result_cache_capacity()
        )
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        changed = False
        try:
            settings = dialog.scientific_settings()
            run_defaults = dialog.new_tab_run_defaults()
            preview_points, preview_delay = dialog.preview_settings()
            run_maximum = dialog.run_user_maximum()
            cache_capacity = dialog.result_cache_capacity()
            persist_scientific_defaults = (
                dialog.persist_scientific_defaults_requested()
            )
            previous_settings = self._workspace_tab_registry.tab(
                tab_id
            ).scientific_settings
            if settings != previous_settings:
                self._workspace_tab_registry.set_scientific_settings(
                    tab_id,
                    settings,
                    expected_token=self._workspace_tab_registry.capture_content_token(),
                )
                changed = True
            if changed:
                self._refresh_tab_controls(tab_id)
            if run_defaults != previous_run_defaults:
                self.config_controller.persist_run_defaults(run_defaults)
            self.config_controller.set_preview_points(preview_points)
            self.config_controller.set_preview_delay_ms(preview_delay)
            self.config_controller.set_run_user_maximum(run_maximum)
            self.config_controller.set_result_cache_capacity(cache_capacity)
        except (TypeError, ValueError, KeyError, OverflowError) as exc:
            message = str(exc)
            self.set_status_text(message)
            QtWidgets.QMessageBox.warning(
                self,
                "Simulation Settings Rejected",
                message,
            )
            return
        except OSError as exc:
            message = str(exc)
            if changed:
                message += " The current project value remains applied for this session."
            self.set_status_text(message)
            QtWidgets.QMessageBox.warning(self, "Simulation Settings Not Saved", message)
            return
        if persist_scientific_defaults:
            try:
                self.config_controller.persist_scientific_defaults(settings)
            except OSError as exc:
                self.set_status_text(str(exc))

    def _stop_simulation(self, tab_id: str | None = None) -> bool:
        """Fence all latent and physical Preview work before cancellation.

        Stop cancels work on every tab; only the tab it was pressed on holds
        its Preview until the next gesture.
        """

        held_tab_id = tab_id or self._workspace_tab_registry.active_tab_id
        changed = False
        if self._scan_owner.busy:
            self._scan_owner.cancel()
            changed = True
        for bundle in tuple(self._tab_surfaces.values()):
            changed = bundle.slider_binding.fence_stop() or changed
        for target in tuple(self._tab_surfaces):
            try:
                changed = (
                    self._workspace.plot_for_tab(target).stop_interactive_display()
                    or changed
                )
            except KeyError:
                continue
        for target, bundle in tuple(self._tab_surfaces.items()):
            changed = (
                bundle.slider_binding.settle_stop(hold=target == held_tab_id)
                or changed
            )
        return bool(self._simulation_jobs.cancel_current() or changed)

    def _capture_scan_session(self):
        """Admit all visible authoring inputs once, without project publication."""
        tab_id = self._workspace_tab_registry.active_tab_id
        self._capture_visible_tab_inputs(tab_id=tab_id)
        document = self._project_session.document
        runtime = self._workspace_tab_registry.tab_runtime(tab_id)
        if runtime.reactions_draft is None:
            proposal = MechanismSourceReconciliation(proposed_document=document)
        else:
            proposal = self._prepare_source_reconciliation(
                document, tab_id, self._visible_source_for_tab(tab_id),
                description="Chemical Space Scan",
            )
        if proposal is None:
            # The review was canceled or its refusal was already shown.
            return None
        candidate = self._captured_input_candidate(
            proposal.proposed_document, (tab_id,), discard_invalid=False,
            superseded_by_tab={tab_id: proposal.materialized_set_ids},
        )
        tab = candidate.tab(tab_id)
        for baseline in tab.initial_condition_sets:
            # Consume the ordinary complete-roster and settings authorities.
            canonical_initial_values(tab, baseline.set_id)
            tab.resolved_execution_settings(baseline.set_id)
        return tab

    def _open_chemical_space_scan(self) -> None:
        if not self._mutation_input_admitted("Chemical Space Scan"):
            return
        try:
            self._scan_owner.open(self._capture_scan_session)
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Chemical Space Scan", str(exc))

    def _run_global_fit(self) -> bool:
        if not self._mutation_input_admitted("Global Fit"):
            return False
        tab_id = self._workspace_tab_registry.active_tab_id
        if not self._confirm_discard_slider_preview(tab_id, "Global Fit"):
            return False
        if self._accept_visible_source_result(
            tab_id,
            description="Global Fit",
        ) is None:
            return False
        return self._fit_session_owner.open_fit_configuration(tab_id) is not None

    def _configure_fitting(self) -> bool:
        if not self._mutation_input_admitted("Configure Fitting"):
            return False
        defaults = self._fit_session_owner.project_defaults
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Fitting Defaults")
        layout = QtWidgets.QVBoxLayout(dialog)
        scroll = QtWidgets.QScrollArea(dialog)
        scroll.setWidgetResizable(True)
        body = QtWidgets.QWidget(scroll)
        scroll.setWidget(body)
        layout.addWidget(scroll, 1)
        form = QtWidgets.QFormLayout(body)
        form.setRowWrapPolicy(QtWidgets.QFormLayout.RowWrapPolicy.WrapLongRows)
        scope = QtWidgets.QComboBox(dialog)
        scope.addItem("Current project", "project")
        scope.addItem("New projects (application default)", "application")
        method = QtWidgets.QComboBox(dialog)
        for method_id in SUPPORTED_FIT_METHODS:
            method.addItem(fit_method_label(method_id), method_id)

        def select_method(method_id: str) -> None:
            index = method.findData(method_id)
            if index < 0:
                raise ValueError(
                    f"Unknown canonical fitting method: {method_id!r}."
                )
            method.setCurrentIndex(index)

        select_method(defaults.method)
        max_nfev = QtWidgets.QSpinBox(dialog)
        max_nfev.setRange(*FIT_MAX_NFEV_RANGE)
        max_nfev.setValue(defaults.max_nfev)
        ftol = QtWidgets.QLineEdit(dialog)
        xtol = QtWidgets.QLineEdit(dialog)
        ftol.setText(repr(float(defaults.ftol)))
        xtol.setText(repr(float(defaults.xtol)))
        ftol_label = QtWidgets.QLabel(FUNCTION_TOLERANCE_LABEL, dialog)
        xtol_label = QtWidgets.QLabel(STEP_TOLERANCE_LABEL, dialog)
        use_seed = QtWidgets.QCheckBox("Use Deterministic Seed", dialog)
        use_seed.setChecked(defaults.use_seed)
        seed = QtWidgets.QSpinBox(dialog)
        seed.setRange(*FIT_SEED_RANGE)
        seed.setValue(defaults.seed)
        for control in (scope, method, max_nfev, seed):
            ignore_wheel(control)

        def project_tolerance(editor: QtWidgets.QLineEdit, value: object) -> None:
            # ``repr(float(...))`` is the shortest exact binary64 round-trip;
            # no widget-side clamp or alternate hidden value is permitted.
            editor.setText(repr(float(value)))

        def load_selected_defaults(_index: int) -> None:
            selected = (
                self._fit_session_owner.project_defaults
                if scope.currentData() == "project"
                else self.config_controller.application_fit_defaults()
            )
            select_method(selected.method)
            max_nfev.setValue(selected.max_nfev)
            project_tolerance(ftol, selected.ftol)
            project_tolerance(xtol, selected.xtol)
            use_seed.setChecked(selected.use_seed)
            seed.setValue(selected.seed)
            sync_method_controls()

        def sync_method_controls(_index: int = -1) -> None:
            is_de = method.currentData() == DIFFERENTIAL_EVOLUTION_METHOD
            ftol_label.setText(
                DE_CONVERGENCE_TOLERANCE_LABEL
                if is_de
                else FUNCTION_TOLERANCE_LABEL
            )
            xtol_label.setVisible(not is_de)
            xtol.setVisible(not is_de)
            xtol.setEnabled(not is_de)

        scope.currentIndexChanged.connect(load_selected_defaults)
        method.currentIndexChanged.connect(sync_method_controls)
        form.addRow("Save to", scope)
        form.addRow("Method", method)
        form.addRow("Maximum evaluations", max_nfev)
        form.addRow(ftol_label, ftol)
        form.addRow(xtol_label, xtol)
        form.addRow(use_seed)
        form.addRow("Seed", seed)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            parent=dialog,
        )

        def read_candidate() -> FitDefaults:
            return FitDefaults(
                method=method.currentData(),
                max_nfev=int(max_nfev.value()),
                ftol=admit_fit_tolerance(ftol.text(), ftol_label.text()),
                xtol=admit_fit_tolerance(xtol.text(), xtol_label.text()),
                use_seed=bool(use_seed.isChecked()),
                seed=int(seed.value()),
            )

        def accept_candidate() -> None:
            try:
                validate_fit_defaults(read_candidate())
            except (TypeError, ValueError, OverflowError) as exc:
                QtWidgets.QMessageBox.warning(
                    dialog,
                    "Invalid Fitting Defaults",
                    str(exc),
                )
                return
            dialog.accept()

        buttons.accepted.connect(accept_candidate)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        sync_method_controls()
        clamp_initial_size(dialog, 600, 480)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return False
        # accept_candidate admitted these values before the dialog closed.
        candidate = read_candidate()
        if scope.currentData() == "project":
            changed = self._fit_session_owner.set_project_defaults(candidate)
            if changed:
                self._refresh_project_state()
            return changed
        try:
            self.config_controller.persist_fit_defaults(candidate)
        except OSError as exc:
            self.set_status_text(str(exc))
            QtWidgets.QMessageBox.warning(self, "Fitting Defaults Not Saved", str(exc))
            return False
        return True

    def _export_xlsx(self) -> bool:
        if not self._mutation_input_admitted("Export Results XLSX"):
            return False
        try:
            blocks = self._canonical_export_blocks(
                self._workspace_tab_registry.active_tab_id
            )
        except (TypeError, ValueError) as exc:
            self.set_status_text(str(exc))
            QtWidgets.QMessageBox.critical(
                self,
                "Export Results XLSX",
                str(exc),
            )
            return False
        if not blocks:
            self.set_status_text("No result data is available to export.")
            return False
        try:
            start_directory = resolve_export_start_dir(
                self._last_export_directory
            )
        except OSError as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "Export Results XLSX",
                str(exc),
            )
            return False
        path, _selected = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export Results XLSX",
            start_directory,
            "Excel Workbook (*.xlsx)",
            options=QtWidgets.QFileDialog.Option.DontConfirmOverwrite,
        )
        if not path:
            return False
        path = os.path.abspath(os.fspath(path))
        if not path.lower().endswith(".xlsx"):
            path += ".xlsx"
        if self._xlsx_export_owner.owns_destination(path):
            self.set_status_text(
                "XLSX export rejected: another export owns that destination."
            )
            return False
        if self.project_controller.is_current_project_path(path):
            self.set_status_text(
                "XLSX export rejected: the destination is the current project archive."
            )
            return False
        if os.path.lexists(path):
            answer = QtWidgets.QMessageBox.question(
                self,
                "Overwrite XLSX",
                f"The XLSX destination already exists:\n{path}\n\n"
                "Overwrite it?",
                QtWidgets.QMessageBox.StandardButton.Yes
                | QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.No,
            )
            if answer != QtWidgets.QMessageBox.StandardButton.Yes:
                return False
        self._remember_export_directory(path)
        self._xlsx_export_owner.start(blocks, path)
        self.set_status_text(f"Exporting XLSX: {path}")
        self._refresh_actions()
        return True

    def _remember_export_directory(self, path: str) -> None:
        directory = os.path.dirname(os.path.abspath(os.fspath(path)))
        if os.path.isdir(directory):
            self._last_export_directory = directory

    @QtCore.Slot(str, object)
    def _on_xlsx_export_succeeded(
        self,
        path: str,
        completion: object,
    ) -> None:
        self.set_status_text(
            f"Exported XLSX: {path} "
            f"({completion.worksheet_count} worksheets, "
            f"{completion.data_row_count} data rows, "
            f"{completion.bytes_written} bytes)."
        )
        self._refresh_actions()

    @QtCore.Slot(str, str)
    def _on_xlsx_export_failed(self, path: str, message: str) -> None:
        self.set_status_text(f"XLSX export failed: {path}: {message}")
        QtWidgets.QMessageBox.critical(
            self,
            "Export Results XLSX",
            f"{path}\n\n{message}",
        )
        self._refresh_actions()

    @QtCore.Slot(str)
    def _on_xlsx_export_cancelled(self, path: str) -> None:
        self.set_status_text(f"XLSX export canceled: {path}")
        self._refresh_actions()

    # ------------------------------------------------------------------
    # Symbolic, inspector, and auxiliary UI
    # ------------------------------------------------------------------
    @QtCore.Slot(object)
    def _calculator_owner_disposed(self, owner) -> None:
        self._calculator_owners.discard(owner)
        owner.deleteLater()

    @QtCore.Slot()
    def _calculator_settlement_changed(self) -> None:
        self.project_controller.notify_close_progress()
        # A retiring owner may settle during ordinary tab-surface replacement.
        if self._workspace_tab_registry.active_tab_id in self._tab_surfaces:
            self._refresh_actions()

    def _refresh_symbolic_calculator(self, tab_id: str) -> None:
        bundle = self._tab_surfaces[str(tab_id)]
        if self._workspace_tab_registry.tab_runtime(
            tab_id
        ).reactions_draft is not None:
            state = bundle.calculator_owner.invalidate_current_snapshot()
        else:
            state = bundle.calculator_owner.refresh()
        bundle.calculator_panel.render_state(state)
        bundle.editor.symbolic_calculator_btn.setEnabled(state.available)

    def _toggle_symbolic_calculator(self, tab_id: str) -> None:
        if not self._mutation_input_admitted("Toggle Symbolic Calculator"):
            return
        identity = SYMBOLIC_CALCULATOR_DOCK_SPEC.identity_key
        dock = self._workspace.dock_for_identity(identity)
        if self._workspace.is_dock_frontmost(identity):
            dock.hide()
            return
        self._open_symbolic_calculator(tab_id)

    def _open_symbolic_calculator(self, tab_id: str) -> None:
        if not self._mutation_input_admitted("Open Symbolic Calculator"):
            return
        self._refresh_symbolic_calculator(tab_id)
        bundle = self._tab_surfaces[str(tab_id)]
        if str(tab_id) == self._workspace_tab_registry.active_tab_id:
            self._calculator_stack.setCurrentWidget(bundle.calculator_panel)
        self._workspace.reveal_dock(SYMBOLIC_CALCULATOR_DOCK_SPEC.identity_key)

    def _toggle_interactive_sliders(self, tab_id: str) -> None:
        if not self._mutation_input_admitted("Toggle Interactive Sliders"):
            return
        identity = SLIDERS_DOCK_SPEC.identity_key
        dock = self._workspace.dock_for_identity(identity)
        if self._workspace.is_dock_frontmost(identity):
            dock.hide()
            return
        bundle = self._tab_surfaces[str(tab_id)]
        if str(tab_id) == self._workspace_tab_registry.active_tab_id:
            self._slider_stack.setCurrentWidget(bundle.slider_panel)
        self._workspace.reveal_dock(identity)

    def _evaluate_symbolic(self, tab_id: str, query: str):
        bundle = self._tab_surfaces[str(tab_id)]
        if not self._mutation_input_admitted("Evaluate Symbolic Expression"):
            return bundle.calculator_owner.view_state()
        if self._workspace_tab_registry.tab_runtime(tab_id).reactions_draft is not None:
            return bundle.calculator_owner.invalidate_current_snapshot()
        return bundle.calculator_owner.evaluate(query)

    def _open_mechanism_inspector(self, tab_id: str) -> None:
        from kindred.gui.widgets.mechanism_inspector import MechanismInspectorDialog

        if not self._mutation_input_admitted("Open Mechanism Inspector"):
            return
        if self._workspace_tab_registry.tab_runtime(tab_id).reactions_draft is not None:
            self.set_status_text(
                "Mechanism Inspector is observational and requires accepted Reactions."
            )
            return
        try:
            dialog = MechanismInspectorDialog(
                self._workspace_tab_registry.tab(tab_id).prepared_dsl,
                self,
            )
            clamp_initial_size(dialog, 760, 560)
            dialog.exec()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Mechanism Inspector", str(exc))

    def _open_docs(self) -> None:
        QtWidgets.QMessageBox.information(
            self,
            "Documentation",
            "Online documentation is not yet published. The Reactions Help tab "
            "contains the current DSL reference.",
        )

    def _show_panel_layout_tips(self) -> None:
        QtWidgets.QMessageBox.information(
            self,
            "Panel Layout Tips",
            "Drag dock title bars to float, redock, or arrange panels on either "
            "side. Use View → Panels to restore hidden panels, Reset Layout to "
            "reposition visible panels, or Restore Default Panel Layout to restore "
            "the factory arrangement and visibility.",
        )

    def _build_diagnostic_log_dialog(self) -> QtWidgets.QDialog:
        """Build a readable view of the active diagnostic log."""

        dialog = QtWidgets.QDialog(self)
        dialog.setObjectName("diagnosticLogDialog")
        dialog.setWindowTitle("Diagnostic Log")
        clamp_initial_size(dialog, 800, 560)

        layout = QtWidgets.QVBoxLayout(dialog)
        log_view = QtWidgets.QPlainTextEdit(dialog)
        log_view.setObjectName("diagnosticLogText")
        log_view.setReadOnly(True)
        revision: tuple[str, int, int, int] | None = None

        def refresh_log() -> None:
            nonlocal revision
            path = None
            try:
                path = diagnostic_log_path()
                state = path.stat()
                current_revision = (
                    os.fspath(path),
                    int(state.st_ino),
                    int(state.st_mtime_ns),
                    int(state.st_size),
                )
                if current_revision == revision:
                    return
                text = path.read_text(encoding="utf-8", errors="replace")
            except (OSError, RuntimeError) as exc:
                current_revision = ("", 0, 0, 0)
                text = f"Diagnostic log could not be read:\n{file_error_text(exc, path)}"

            scroll_bar = log_view.verticalScrollBar()
            was_at_bottom = scroll_bar.value() >= scroll_bar.maximum()
            previous = log_view.toPlainText()
            if previous and text.startswith(previous):
                cursor = QtGui.QTextCursor(log_view.document())
                cursor.movePosition(QtGui.QTextCursor.MoveOperation.End)
                cursor.insertText(text[len(previous):])
            else:
                log_view.setPlainText(text)
            revision = current_revision
            if was_at_bottom:
                scroll_bar.setValue(scroll_bar.maximum())

        refresh_log()
        refresh_timer = QtCore.QTimer(dialog)
        refresh_timer.setInterval(500)
        refresh_timer.timeout.connect(refresh_log)
        refresh_timer.start()

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Close,
            dialog,
        )
        buttons.setObjectName("diagnosticLogButtonBox")
        copy_button = buttons.addButton(
            "Copy",
            QtWidgets.QDialogButtonBox.ButtonRole.ActionRole,
        )
        copy_button.setObjectName("copyDiagnosticLogButton")
        copy_button.clicked.connect(
            lambda _checked=False: QtWidgets.QApplication.clipboard().setText(
                log_view.toPlainText()
            )
        )
        buttons.rejected.connect(dialog.reject)

        layout.addWidget(log_view)
        layout.addWidget(buttons)
        return dialog

    def _show_diagnostic_log(self) -> None:
        dialog = self._diagnostic_log_dialog
        if dialog is None:
            dialog = self._build_diagnostic_log_dialog()
            dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
            dialog.destroyed.connect(self._diagnostic_log_destroyed)
            self._diagnostic_log_dialog = dialog
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _diagnostic_log_destroyed(self, _object: object = None) -> None:
        self._diagnostic_log_dialog = None

    def _about_brand_asset_path(self) -> str:
        """Return the bundled brand image for the currently applied theme."""
        action = self._dark_mode_action
        if action is not None and action.isChecked():
            return "assets/kindred-full-mark-dark.png"
        return "assets/kindred-full-mark.png"

    def _build_about_dialog(self) -> QtWidgets.QDialog:
        """Build the complete branded About dialog."""
        from kindred.io.resources import get_resource_path

        dialog = QtWidgets.QDialog(self)
        dialog.setObjectName("aboutKindredDialog")
        dialog.setWindowTitle("About Kindred")

        outer = QtWidgets.QVBoxLayout(dialog)
        scroll = QtWidgets.QScrollArea(dialog)
        scroll.setWidgetResizable(True)
        body = QtWidgets.QWidget(scroll)
        scroll.setWidget(body)
        outer.addWidget(scroll, 1)
        layout = QtWidgets.QVBoxLayout(body)
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
        if pixmap.isNull():
            brand_label.setText("Kindred")
        else:
            brand_label.setPixmap(
                pixmap.scaled(
                    _ABOUT_DIALOG_IMAGE_MAX_SIZE,
                    _ABOUT_DIALOG_IMAGE_MAX_SIZE,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )

        title_label = QtWidgets.QLabel("Kindred", dialog)
        title_label.setObjectName("aboutTitleLabel")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_font = title_label.font()
        title_font.setBold(True)
        title_font.setPointSize(title_font.pointSize() + 6)
        title_label.setFont(title_font)

        version_label = QtWidgets.QLabel(
            f"Version {KINDRED_VERSION}",
            dialog,
        )
        version_label.setObjectName("aboutVersionLabel")
        version_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        body_label = QtWidgets.QLabel(
            "Desktop GUI for reaction mechanism investigation, kinetic "
            "modeling, and data fitting.\n\n"
            "The initial development of Kindred was led primarily by Pedro "
            "Helou de Oliveira.\n"
            "Kindred is now developed and maintained jointly by Pedro Helou "
            "de Oliveira and Annabel Flook.\n\n"
            "Certain GUI design elements were informed by ideas seen in "
            "Velocity, a kinetics software package developed and maintained "
            "by Chen Li.\n\n"
            "Kindred source code is licensed under the MIT License.\n"
            "Includes Qt for Python (PySide6/shiboken6) under LGPLv3/GPL "
            "terms.\n"
            "See NOTICE.txt, COPYING, COPYING.LESSER, and "
            "THIRD_PARTY_LICENSES.txt.",
            dialog,
        )
        body_label.setObjectName("aboutBodyLabel")
        body_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        body_label.setWordWrap(True)

        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok,
            dialog,
        )
        button_box.setObjectName("aboutDialogButtonBox")
        button_box.accepted.connect(dialog.accept)

        layout.addWidget(brand_label)
        layout.addWidget(title_label)
        layout.addWidget(version_label)
        layout.addWidget(body_label)
        outer.addWidget(button_box)
        clamp_initial_size(dialog, _ABOUT_DIALOG_PREFERRED_WIDTH, 620)
        return dialog

    def _show_about(self) -> None:
        self._build_about_dialog().exec()

    # ------------------------------------------------------------------
    # Dock/layout/config helpers
    # ------------------------------------------------------------------
    def _load_settings(self) -> None:
        """Restore physical shell state owned by MainWindow and its owners."""

        restore_maximized, geometry, dock_state = (
            self.config_controller.window_layout_preference()
        )
        if geometry:
            try:
                if self.restoreGeometry(geometry) is False:
                    raise ValueError("restore rejected stored geometry")
            except (EOFError, RuntimeError, TypeError, ValueError) as exc:
                logger.warning("Ignoring invalid persisted window geometry: %s", exc)
        self._workspace.restore_dock_layout(dock_state)
        QtCore.QTimer.singleShot(0, self._workspace.recover_floating_docks)
        if not geometry and not restore_maximized:
            preferred = self._startup_size()
            QtCore.QTimer.singleShot(
                0, lambda: clamp_initial_size(self, preferred.width(), preferred.height())
            )
        if restore_maximized:
            self.restore_persisted_maximized_state()

        requested_dark = self.config_controller.theme_preference()
        if not self._theme_manager.apply(requested_dark, None):
            self.set_status_text("Could not apply the persisted theme preference.")
        if self._dark_mode_action is not None:
            self._dark_mode_action.setChecked(self._theme_manager.applied_dark_mode)

    def _save_settings(self) -> bool:
        """Persist physical shell state through Config's restore contract."""

        try:
            self.config_controller.persist_window_layout(
                self.isMaximized(),
                self.saveGeometry(),
                self._workspace.saveState(),
            )
        except OSError as exc:
            self.set_status_text(str(exc))
            return False
        return True

    def _toggle_theme(self) -> bool:
        action = self._dark_mode_action
        if action is None:
            return False
        requested = bool(action.isChecked())
        try:
            applied = self._theme_manager.apply(
                requested,
                self.config_controller.persist_theme,
            )
        finally:
            action.setChecked(self._theme_manager.applied_dark_mode)
        self._refresh_application_plot_preferences()
        return bool(applied)

    def _startup_size(self) -> QtCore.QSize:
        screen = self.screen()
        if screen is None:
            return QtCore.QSize(1440, 900)
        available = screen.availableGeometry()
        return QtCore.QSize(
            int(available.width() * 0.86),
            int(available.height() * 0.88),
        )

    def restore_persisted_maximized_state(self) -> None:
        QtCore.QTimer.singleShot(0, self.showMaximized)

    # ------------------------------------------------------------------
    # Project-controller workspace callbacks and native Close
    # ------------------------------------------------------------------
    def set_status_text(self, text: str) -> None:
        self._status_label.setText(str(text))

    @QtCore.Slot(bool)
    def _on_project_state_changed(self, project_replaced: bool) -> None:
        """Refresh the current root after semantic publication."""

        self._refresh_workspace_document(project_replaced)

    def _refresh_project_state(self) -> None:
        self._render_tab_state()
        self._refresh_title()
        self._refresh_actions()

    def _simulation_work_active(self) -> bool:
        return bool(self._simulation_jobs.busy)

    @QtCore.Slot(bool)
    def _on_fit_window_changed(self, _open: bool) -> None:
        self._refresh_actions()

    @QtCore.Slot(bool)
    def _on_fit_execution_changed(self, _executing: bool) -> None:
        self._refresh_actions()

    @QtCore.Slot(object)
    def _on_fit_project_document_changed(self, tab_id: object) -> None:
        if tab_id is None:
            self._refresh_project_state()
            return
        target = str(tab_id)
        if target not in self._tab_surfaces:
            self._create_tab_surface(target)
            self._refresh_workspace_tab_structure()
            self._apply_persisted_plot_state(target)
        self._refresh_tab_controls(target)

    @QtCore.Slot()
    def _on_simulation_state_changed(self) -> None:
        """Project the live simulation kind into every retained panel."""

        job_kind = self._simulation_jobs.current_kind
        for bundle in self._tab_surfaces.values():
            bundle.simulation_panel.render_job_state(job_kind)
        self._refresh_actions()

    @QtCore.Slot()
    def _on_close_started(self) -> None:
        self._fence_shell_for_close()
        for target in tuple(self._tab_surfaces):
            try:
                self._workspace.plot_for_tab(target).discard_interactive_display()
            except KeyError:
                continue
        self.set_status_text("Closing Kindred. Waiting for active work to stop.")

    def _refresh_title(self) -> None:
        path = self._project_session.current_path
        name = os.path.basename(path) if path else "Untitled"
        dirty = bool(
            self._project_session.is_dirty(
                self._workspace_tab_registry.workspace_presentation()
            )
            or any(self._tab_has_drafts(tab_id) for tab_id in self._tab_surfaces)
        )
        self.setWindowTitle(f"Kindred — {name}{' *' if dirty else ''}")

    def _refresh_actions(self) -> None:
        if not self._tab_surfaces:
            return
        tab_id = self._workspace_tab_registry.active_tab_id
        bundle = self._tab_surfaces.get(tab_id)
        if bundle is None:
            return
        mutation_admitted = self._mutation_input_admitted()
        for surface in self._tab_surfaces.values():
            surface.data_panel.set_creation_busy(self._spreadsheet_editor is not None or not mutation_admitted)
        if self._xlsx_export_action is not None:
            import_action = bundle.data_panel.import_action
            if self._load_data_action is not import_action:
                self._file_menu.insertAction(self._xlsx_export_action, import_action)
                if self._load_data_action is not None:
                    self._file_menu.removeAction(self._load_data_action)
                self._load_data_action = import_action
            self._load_data_action.setEnabled(
                mutation_admitted and not self._dataset_import_staging.busy
            )
        self._undo_action.setEnabled(
            mutation_admitted and bundle.editor.native_undo_available()
        )
        self._redo_action.setEnabled(
            mutation_admitted and bundle.editor.native_redo_available()
        )
        self._set_tab_run_enabled(
            tab_id,
            mutation_admitted and self._run_admitted(tab_id),
        )
        self._stop_action.setEnabled(
            mutation_admitted
            and (self._simulation_work_active() or self._scan_owner.busy)
        )
        has_reactions = self._has_visible_reactions(tab_id)
        self._global_fit_action.setEnabled(
            mutation_admitted
            and has_reactions
            and self._computation_gate.is_idle
            and not self._fit_session_owner.active
        )
        active_work = self._active_project_work()
        if self._scan_action is not None:
            self._scan_action.setEnabled(
                mutation_admitted and (self._scan_owner.dialog is not None
                                       or has_reactions and not active_work and self._computation_gate.is_idle)
            )
        project_actions_enabled = mutation_admitted and not active_work
        if self._new_project_action is not None:
            self._new_project_action.setEnabled(project_actions_enabled)
        if self._open_project_action is not None:
            self._open_project_action.setEnabled(project_actions_enabled)
        if self._recent_menu is not None:
            self._recent_menu.setEnabled(project_actions_enabled)
        if self._xlsx_export_action is not None:
            # Export snapshots retained results; only Run/Preview, Scan and
            # another export exclude it (Close is the mutation fence).
            self._xlsx_export_action.setEnabled(
                mutation_admitted
                and not self._simulation_work_active()
                and not self._scan_owner.busy
                and not self._xlsx_export_owner.is_exporting
                and self._canonical_export_available(
                    self._workspace_tab_registry.active_tab_id
                )
            )

    def _apply_slider_preview_settings(self) -> None:
        for bundle in self._tab_surfaces.values():
            bundle.slider_binding.set_preview_settings(
                points=self.config_controller.preview_points(),
                delay_ms=self.config_controller.preview_delay_ms(),
            )

    @QtCore.Slot(int)
    def _resize_result_repositories(self, capacity: int) -> None:
        for tab in self._project_session.document.tabs:
            self._repository_for_tab(tab.tab_id).resize_cache_capacity(
                int(capacity)
            )
            self._render_results_for_tab(tab.tab_id)

    def authorize_and_schedule_close(self) -> None:
        self._fence_shell_for_close()
        QtCore.QTimer.singleShot(0, self.close)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if not self.project_controller.consume_native_close_authorization():
            event.ignore()
            self.project_controller.request_close()
            return
        if not self._save_settings():
            QtWidgets.QMessageBox.warning(
                self,
                "Layout Not Saved",
                "Kindred could not save the current window layout. "
                "The previously saved layout will be restored the next time "
                "Kindred starts.",
            )
        event.accept()
