"""Kindred's single-window shell and direct application graph wiring."""

from __future__ import annotations

import logging
import os
import tempfile
import threading
import time
import webbrowser
from dataclasses import dataclass, replace
from typing import Callable, Iterable, Sequence

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt

from kindred import __version__ as KINDRED_VERSION
from kindred.core.dataset_mapping import (
    DatasetMappingError,
    DatasetMappingChoice,
    OccupiedSetChoice,
    SetCopyPlan,
    apply_dataset_set_mapping_plan,
    create_set_from_exact_zero,
    export_original_source,
    plan_dataset_to_set_mapping,
    plan_global_dataset_delete,
    plan_leave_unmapped,
    validate_unique_dataset_name,
    with_confirmed_observation_mapping,
    with_validated_fit_inclusion,
)
from kindred.core.datasets.observation_mapping import suggest_observation_mapping
from kindred.core.mechanism_source import MechanismAuthoringSource
from kindred.core.project_document import (
    FitDefaults,
    MechanismSourceCollision,
    MechanismSourceFailure,
    MechanismSourceReconciliation,
    ProjectDocument,
    SimulationExecutionSettings,
    new_set_id,
    next_set_name,
    reconcile_tab_mechanism_source,
)
from kindred.core.project_session import ProjectSession
from kindred.core.result_repository import (
    ExplicitResultKey,
    ResultRepository,
    scientific_identity,
    standard_concentration_identity,
)
from kindred.core.simulator.dsl_parse import (
    PreparedDslRequestSet,
    prepare_dsl_request_set,
)
from kindred.core.units import UnitsModel
from kindred.gui.app_wiring import (
    build_analysis_dock,
    build_data_dock,
    build_initial_conditions_dock,
    build_mechanism_dock,
    build_sliders_dock,
    build_symbolic_calculator_dock,
    dock_default_area,
    dock_shell_specs,
)
from kindred.gui.computation_gate import ComputationGate
from kindred.gui.controllers.config_controller import (
    ConfigController,
    ConfigControllerPort,
)
from kindred.gui.controllers.project_controller import (
    AuthoringDiscardMode,
    ProjectController,
    ProjectTransientState,
)
from kindred.gui.controllers.simulation_job_owner import (
    ExplicitRunCapture,
    SimulationJobOwner,
    build_explicit_target,
    resolve_run_parallel_capacity,
)
from kindred.gui.dataset_import_staging import (
    DatasetImportStaging,
    raw_record_batch_proposal,
)
from kindred.gui.dataset_payload_snapshot import (
    build_data_manager_snapshot,
    build_experimental_layer_inputs,
)
from kindred.gui.fit_session_owner import FitSessionOwner
from kindred.gui.fitting.constants import (
    FITTING_MAX_NFEV_RANGE,
    FITTING_METHODS,
    FITTING_SEED_RANGE,
)
from kindred.gui.main_window_variable_runtime import ActiveTabSliderBinding
from kindred.gui.plot_config import PlotPresentationIntent
from kindred.gui.symbolic_calculator_owner import SymbolicCalculatorOwner
from kindred.gui.theme_manager import ThemeManager
from kindred.gui.widgets.batch_initial_conditions_table import (
    BatchInitialConditionsTableModel,
    InitialConditionsTableActions,
    projected_analysis_identity,
    projected_initial_values,
    projected_species_names,
)
from kindred.gui.widgets.data_manager import (
    DataManagerPanel,
    DatasetDeleteIntent,
    DatasetFitInclusionIntent,
    DatasetMappingIntent,
    DatasetObservationMappingIntent,
    DatasetRenameIntent,
    DatasetExportOriginalIntent,
)
from kindred.gui.widgets.dataset_overlay_panel import (
    DatasetOverlayPanel,
    DatasetStyle,
    OverlaySelectionIntent,
)
from kindred.gui.widgets.mechanism_editor import (
    DetailedBalanceCorrectionOutcome,
    MechanismEditorBinding,
)
from kindred.gui.widgets.mechanism_workspace import MechanismWorkspaceView
from kindred.gui.widgets.right_panel import RightPanelTabbed
from kindred.gui.widgets.simulation_panel import SimulationPanel
from kindred.gui.widgets.solver_settings import TabScientificSettingsDialog
from kindred.gui.widgets.symbolic_calculator_panel import SymbolicCalculatorPanel
from kindred.gui.workspace_result_presenter import (
    VisibleResultSet,
    WorkspaceResultPresenter,
    WorkspaceResultProjection,
    XlsxExportCompletion,
)
from kindred.gui.workspace_tab_registry import (
    ScientificDocumentDelta,
    WorkspaceTabRegistry,
)
from kindred.io.kin_archive_store import KinArchiveStore
from kindred.io.paths import resolve_start_dir
from kindred.io.resources import (
    InterventionExampleCatalogError,
    get_all_intervention_example_specs,
)

logger = logging.getLogger(__name__)

__all__ = ["MainWindow"]

DOCUMENTATION_URL: str | None = None
_REPLACEMENT_TIMEOUT_SECONDS = 4.0
_ABOUT_DIALOG_MIN_WIDTH = 420
_ABOUT_DIALOG_IMAGE_MAX_SIZE = 320


@dataclass(frozen=True, slots=True)
class _WorkspaceRenderProjection:
    document: ProjectDocument
    active_tab_id: str
    presentation: object
    editor: object
    initial_conditions: object
    sliders: object
    data_snapshot: object
    result_projection: WorkspaceResultProjection
    species_names: tuple[str, ...]
    run_admitted: bool
    simulation_job_kind: str | None
    tab_states: tuple[tuple[str, bool, bool], ...]
    slider_target_summary: str


@dataclass(frozen=True, slots=True)
class _XlsxExportSuccess:
    generation: int
    path: str
    completion: XlsxExportCompletion


@dataclass(frozen=True, slots=True)
class _XlsxExportFailure:
    generation: int
    path: str
    message: str
    cancelled: bool


class _XlsxExportCancelled(Exception):
    """Internal cancellation fence for one application-owned XLSX export."""


class _XlsxExportWorker(QtCore.QThread):
    succeeded = QtCore.Signal(object)
    failed = QtCore.Signal(object)

    def __init__(
        self,
        *,
        generation: int,
        projection: WorkspaceResultProjection,
        path: str,
        parent: QtCore.QObject,
    ) -> None:
        super().__init__(parent)
        if not isinstance(projection, WorkspaceResultProjection):
            raise TypeError("projection must be WorkspaceResultProjection.")
        self._generation = int(generation)
        self._projection = projection
        self._path = os.path.abspath(os.fspath(path))
        self._finalize_lock = threading.Lock()

    def cancel(self) -> None:
        with self._finalize_lock:
            self.requestInterruption()

    def _check_cancelled(self) -> None:
        if self.isInterruptionRequested():
            raise _XlsxExportCancelled()

    def run(self) -> None:
        temporary_path = ""
        descriptor = -1
        try:
            self._check_cancelled()
            parent = os.path.dirname(self._path) or os.curdir
            if not os.path.isdir(parent):
                raise FileNotFoundError(
                    f"Export directory does not exist: {parent}"
                )
            descriptor, temporary_path = tempfile.mkstemp(
                prefix=f".{os.path.basename(self._path)}.",
                suffix=".partial",
                dir=parent,
            )
            self._check_cancelled()
            with os.fdopen(descriptor, "w+b") as destination:
                descriptor = -1
                self._check_cancelled()
                completion = WorkspaceResultPresenter.write_xlsx(
                    self._projection,
                    destination,
                    cancellation_check=self._check_cancelled,
                )
                if not isinstance(completion, XlsxExportCompletion):
                    raise TypeError(
                        "WorkspaceResultPresenter.write_xlsx() must return "
                        "XlsxExportCompletion."
                )
                self._check_cancelled()
                destination.flush()
                self._check_cancelled()
            with self._finalize_lock:
                self._check_cancelled()
                os.replace(temporary_path, self._path)
                temporary_path = ""
        except _XlsxExportCancelled:
            self.failed.emit(
                _XlsxExportFailure(
                    generation=self._generation,
                    path=self._path,
                    message="XLSX export was cancelled.",
                    cancelled=True,
                )
            )
        except Exception as exc:
            self.failed.emit(
                _XlsxExportFailure(
                    generation=self._generation,
                    path=self._path,
                    message=f"{type(exc).__name__}: {exc}",
                    cancelled=False,
                )
            )
        else:
            self.succeeded.emit(
                _XlsxExportSuccess(
                    generation=self._generation,
                    path=self._path,
                    completion=completion,
                )
            )
        finally:
            if descriptor >= 0:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
            if temporary_path:
                try:
                    os.unlink(temporary_path)
                except FileNotFoundError:
                    pass
                except OSError:
                    logger.warning(
                        "Could not remove partial XLSX export %s",
                        temporary_path,
                        exc_info=True,
                    )


class _XlsxExportOwner(QtCore.QObject):
    """Own one exact XLSX worker and its replacement/Close fences."""

    succeeded = QtCore.Signal(object)
    failed = QtCore.Signal(object)
    owned_qthreads_changed = QtCore.Signal()

    def __init__(self, *, parent: QtCore.QObject) -> None:
        super().__init__(parent)
        self._generation = 0
        self._active_worker: _XlsxExportWorker | None = None
        self._application_close_fenced = False

    @property
    def is_exporting(self) -> bool:
        return self._active_worker is not None

    def start(
        self,
        projection: WorkspaceResultProjection,
        path: str,
    ) -> bool:
        absolute_path = os.path.abspath(os.fspath(path))
        if self._application_close_fenced:
            self.failed.emit(
                _XlsxExportFailure(
                    generation=self._generation,
                    path=absolute_path,
                    message="XLSX export is unavailable while Kindred closes.",
                    cancelled=True,
                )
            )
            return False
        worker = self._active_worker
        if worker is not None:
            self.failed.emit(
                _XlsxExportFailure(
                    generation=self._generation,
                    path=absolute_path,
                    message="Another XLSX export is already active.",
                    cancelled=False,
                )
            )
            return False
        self._generation += 1
        generation = self._generation
        worker = _XlsxExportWorker(
            generation=generation,
            projection=projection,
            path=absolute_path,
            parent=self,
        )
        self._active_worker = worker
        worker.succeeded.connect(self._publish_success)
        worker.failed.connect(self._publish_failure)
        worker.finished.connect(self._worker_finished)
        worker.start()
        self.owned_qthreads_changed.emit()
        return True

    @QtCore.Slot(object)
    def _publish_success(self, outcome: object) -> None:
        if (
            isinstance(outcome, _XlsxExportSuccess)
            and outcome.generation == self._generation
        ):
            self.succeeded.emit(outcome)

    @QtCore.Slot(object)
    def _publish_failure(self, outcome: object) -> None:
        if (
            isinstance(outcome, _XlsxExportFailure)
            and outcome.generation == self._generation
        ):
            self.failed.emit(outcome)

    @QtCore.Slot()
    def _worker_finished(self) -> None:
        worker = self.sender()
        if worker is self._active_worker:
            self._active_worker = None
        if isinstance(worker, QtCore.QThread):
            worker.deleteLater()
        self.owned_qthreads_changed.emit()

    def _cancel_active(self) -> _XlsxExportWorker | None:
        self._generation += 1
        worker = self._active_worker
        if worker is not None:
            worker.cancel()
        return worker

    def cancel_and_wait_for_project_replacement(
        self,
        deadline: float,
    ) -> bool:
        worker = self._cancel_active()
        if worker is None or not worker.isRunning():
            return True
        remaining_ms = max(
            0,
            int((float(deadline) - time.monotonic()) * 1000.0),
        )
        return bool(worker.wait(remaining_ms))

    def cancel_for_application_close(self) -> None:
        self._application_close_fenced = True
        self._cancel_active()

    def owned_qthreads(self) -> tuple[QtCore.QThread, ...]:
        worker = self._active_worker
        return () if worker is None else (worker,)


class MainWindow(QtWidgets.QMainWindow):
    """One physical shell over one explicit, retained semantic owner graph."""

    def __init__(self) -> None:
        super().__init__()
        application = QtWidgets.QApplication.instance()
        if not isinstance(application, QtWidgets.QApplication):
            raise RuntimeError("MainWindow requires an existing QApplication.")

        self._native_close_authorized = False
        self._rendering_workspace = False
        self._recent_menu: QtWidgets.QMenu | None = None
        self._current_result_projection: WorkspaceResultProjection | None = None
        self._dark_mode_action: QtGui.QAction | None = None
        self._run_action: QtGui.QAction | None = None
        self._xlsx_export_action: QtGui.QAction | None = None

        self.setObjectName("kindredMainWindow")
        self.setWindowTitle("Kindred")
        self.resize(self._startup_size())
        self.setDockNestingEnabled(True)

        self._workspace = MechanismWorkspaceView(
            self,
            main_plot_embed_analysis_tabs=False,
        )
        self._workspace.setObjectName("mechanismWorkspace")
        self.setCentralWidget(self._workspace)

        self._status_label = QtWidgets.QLabel("Ready", self)
        self.statusBar().addWidget(self._status_label, 1)

        self._theme_manager = ThemeManager(
            application,
            self._workspace,
            report_failure=self.set_status_text,
        )
        self.config_controller = ConfigController(self._config_port())

        initial_document = self.config_controller.default_project_document()
        self._project_session = ProjectSession(initial_document)
        self._computation_gate = ComputationGate(parent=self)

        self._simulation_jobs = SimulationJobOwner(
            gate=self._computation_gate,
            repository_for_tab=self._repository_for_tab,
            explicit_result_key_is_current=self._explicit_result_key_is_current,
            commit_detailed_balance_source=self._commit_detailed_balance_source,
            user_parallel_maximum=(
                self.config_controller.run_user_maximum_preference
            ),
            parent=self,
        )
        self._workspace_tab_registry = WorkspaceTabRegistry(
            self._project_session,
            deletion_result_fence=self._simulation_jobs,
            runtime_factory=self._new_tab_runtime,
        )
        self._dataset_import_staging = DatasetImportStaging(parent=self)
        self._fit_session_owner = FitSessionOwner(
            session=self._project_session,
            registry=self._workspace_tab_registry,
            gate=self._computation_gate,
            deauthorize_targets=self._simulation_jobs.deauthorize_targets,
            commit_detailed_balance_source=self._commit_detailed_balance_source,
            color_resolver=lambda species: self.config_controller.species_color(
                species,
                projected_species_names(self._workspace_tab_registry.active_tab),
            ),
            parent_widget=self,
            parent=self,
        )

        self._build_physical_shell()
        self._build_bindings()
        self._build_sliders_dock()
        self._xlsx_export_owner = _XlsxExportOwner(parent=self)
        self._xlsx_export_owner.succeeded.connect(
            self._on_xlsx_export_succeeded
        )
        self._xlsx_export_owner.failed.connect(
            self._on_xlsx_export_failed
        )
        self._xlsx_export_owner.owned_qthreads_changed.connect(
            self._refresh_actions
        )

        self._archive_store = KinArchiveStore()
        self.project_controller = ProjectController(
            self,
            session=self._project_session,
            tabs=self._workspace_tab_registry,
            archive_store=self._archive_store,
            config=self.config_controller,
            simulation_jobs=self._simulation_jobs,
            dataset_imports=self._dataset_import_staging,
            fit_session=self._fit_session_owner,
            xlsx_exports=self._xlsx_export_owner,
            set_status_text=self.set_status_text,
            settle_execution_candidate=self._settle_execution_candidate,
            prepare_before_save=self._prepare_before_save,
            commit_saved_document=self._commit_saved_document,
            inspect_transient_state=self._inspect_transient_state,
            discard_authoring=self._discard_authoring,
            discard_slider_drafts=self._discard_all_slider_drafts,
            replacement_timeout_seconds=_REPLACEMENT_TIMEOUT_SECONDS,
        )

        from kindred.config.templates import TemplateManager

        self._template_manager = TemplateManager()
        self._template_manager.load_templates()
        self._symbolic_calculator_owner = SymbolicCalculatorOwner(
            canonical_prepared_dsl=lambda: (
                self._workspace_tab_registry.active_tab.prepared_dsl
            ),
        )
        self._build_symbolic_panel()

        self._create_menus()
        self._connect_application_graph()
        self._apply_default_dock_arrangement()
        self.config_controller.load_settings()
        self._render_workspace()
        logger.info("Kindred MainWindow initialized with the direct owner graph")

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def _config_port(self) -> ConfigControllerPort:
        return ConfigControllerPort(
            parent=self,
            settings=QtCore.QSettings,
            is_maximized=self.isMaximized,
            restore_maximized_state=self.restore_persisted_maximized_state,
            restore_geometry=self.restoreGeometry,
            save_geometry=self.saveGeometry,
            restore_window_state=self.restoreState,
            save_window_state=self.saveState,
            schedule_restored_floating_dock_recovery=(
                self.schedule_restored_floating_dock_recovery
            ),
            splitter=self._workspace.main_plot_workspace_splitter,
            dark_mode_action=lambda: self._dark_mode_action,
            apply_theme=self._theme_manager.apply,
            set_status_text=self.set_status_text,
            recent_menu=lambda: self._recent_menu,
            load_recent_project=lambda path: self.project_controller.load_recent_project(
                path
            ),
        )

    @staticmethod
    def _new_tab_runtime(tab_id: str):
        from kindred.gui.workspace_tab_registry import TabRuntimeState

        return TabRuntimeState(result_repository=ResultRepository(tab_id))

    def _build_physical_shell(self) -> None:
        mechanism = build_mechanism_dock(self)
        self._mechanism_dock = mechanism.dock
        self._mechanism_panel = mechanism.panel
        self._mechanism_editor = self._mechanism_panel.editor

        self._batch_model = BatchInitialConditionsTableModel(parent=self)
        self._simulation_panel = SimulationPanel(
            batch_model=self._batch_model,
            parent=self,
        )
        initials = build_initial_conditions_dock(
            self,
            panel=self._simulation_panel,
        )
        self._batch_dock = initials.dock
        self._batch_table = self._simulation_panel.batch_table

        self._dataset_import_panel = DataManagerPanel(
            self._dataset_import_staging,
            parent=self,
        )
        self._overlay_panel = DatasetOverlayPanel(
            self,
            color_for_species=self.config_controller.species_color,
            set_species_color=self.config_controller.set_species_color,
            experimental_style=lambda: DatasetStyle(
                size=self.config_controller.experimental_point_size(),
                opacity=round(
                    self.config_controller.experimental_point_opacity() * 255
                ),
            ),
            set_experimental_style=lambda style: (
                self.config_controller.set_experimental_point_size(style.size),
                self.config_controller.set_experimental_point_opacity(
                    style.opacity / 255.0
                ),
            ),
        )
        self._right_panel = RightPanelTabbed(
            self,
            data_panel=self._dataset_import_panel,
            overlay_panel=self._overlay_panel,
        )
        data = build_data_dock(self, panel=self._right_panel)
        self._right_dock = data.dock

        analysis = build_analysis_dock(
            self,
            panel=self._workspace.main_plot_analysis_widget(),
        )
        self._analysis_dock = analysis.dock

    def _build_sliders_dock(self) -> None:
        slider_panel = self._mechanism_editor.detach_slider_pane_for_dock()
        sliders = build_sliders_dock(self, panel=slider_panel)
        self._sliders_dock = sliders.dock

    def _build_bindings(self) -> None:
        self._result_presenter = WorkspaceResultPresenter()
        self._editor_binding = MechanismEditorBinding(
            editor=self._mechanism_editor,
            registry=self._workspace_tab_registry,
            simulation_jobs=self._simulation_jobs,
            parent=self,
        )
        self._initial_conditions_actions = InitialConditionsTableActions(
            registry=self._workspace_tab_registry,
            model=self._batch_model,
            table=self._batch_table,
            simulation_jobs=self._simulation_jobs,
            parent=self,
        )
        self._slider_binding = ActiveTabSliderBinding(
            registry=self._workspace_tab_registry,
            table=self._batch_table,
            variable_sliders=self._mechanism_editor.variable_sliders_widget(),
            species_sliders=self._mechanism_editor.species_sliders_widget(),
            simulation_jobs=self._simulation_jobs,
            preview_preflight=self._preview_preflight,
            parameter_preview_debounce_ms=(
                self.config_controller.mechanism_parameter_debounce_ms()
            ),
            equilibrium_preview_debounce_ms=(
                self.config_controller.equilibrium_k_debounce_ms()
            ),
            parent=self,
        )

    def _build_symbolic_panel(self) -> None:
        panel = SymbolicCalculatorPanel(
            on_evaluate=self._evaluate_symbolic,
            on_copy_result=self._symbolic_calculator_owner.copy_compact_text,
            on_copy_context=self._symbolic_calculator_owner.copy_context_text,
            parent=self,
        )
        symbolic = build_symbolic_calculator_dock(self, panel=panel)
        self._symbolic_calculator_dock = symbolic.dock
        self._symbolic_calculator_panel = panel

    def _connect_application_graph(self) -> None:
        self.project_controller.documentReplaced.connect(self._on_document_replaced)
        self.project_controller.projectSaved.connect(self._render_workspace)
        self.project_controller.projectStateChanged.connect(
            self._refresh_project_state
        )
        self.project_controller.closePendingChanged.connect(
            self._on_close_pending_changed
        )
        self.project_controller.nativeCloseAuthorized.connect(
            self.authorize_and_schedule_close
        )
        self.project_controller.hardExitAuthorized.connect(self._hard_exit)

        self._workspace.tabActivated.connect(self._select_tab)
        self._workspace.createTabRequested.connect(self._create_tab)
        self._workspace.renameTabRequested.connect(self._rename_tab)
        self._workspace.closeTabRequested.connect(self._close_tab)

        self._simulation_panel.addSetRequested.connect(self._add_initial_set)
        self._simulation_panel.moveSelectedRequested.connect(self._move_initial_sets)
        self._simulation_panel.deleteSelectedRequested.connect(self._delete_initial_sets)
        self._simulation_panel.runSelectedRequested.connect(
            self._start_selected_run
        )
        self._simulation_panel.stopRequested.connect(
            self._simulation_jobs.cancel_current
        )
        self._simulation_panel.workspaceRunDefaultsRequested.connect(
            self._set_workspace_run_defaults
        )
        self._simulation_panel.defaultsEditStarted.connect(
            self._initial_conditions_actions.fence_inheriting_run_defaults_edit
        )
        self._simulation_panel.runEnabledChanged.connect(
            self._set_shared_run_enabled
        )
        self._simulation_panel.settingsValidationFailed.connect(self.set_status_text)
        self._batch_model.runSettingsEditRequested.connect(
            self._initial_conditions_actions.edit_custom_run_settings
        )
        self._batch_model.plotMembershipChangeRequested.connect(
            self._on_plot_membership_requested
        )

        self._mechanism_editor.run_btn.clicked.connect(
            self._start_selected_run
        )
        self._mechanism_editor.sliderApplyRequested.connect(self._slider_binding.apply)
        self._mechanism_editor.sliderResetRequested.connect(self._slider_binding.reset)
        self._mechanism_editor.symbolic_calculator_btn.clicked.connect(
            self._open_symbolic_calculator
        )
        self._mechanism_editor.mechanismInspectRequested.connect(
            self._open_mechanism_inspector
        )
        self._editor_binding.draftStateChanged.connect(self._on_draft_state_changed)
        self._editor_binding.errorRaised.connect(self.set_status_text)
        self._mechanism_editor.nativeHistoryAvailabilityChanged.connect(
            lambda _undo, _redo: self._refresh_actions()
        )
        application = QtWidgets.QApplication.instance()
        if isinstance(application, QtWidgets.QApplication):
            application.focusChanged.connect(
                lambda _old, _new: self._refresh_actions()
            )
        self._slider_binding.draftStateChanged.connect(self._on_draft_state_changed)
        self._slider_binding.errorRaised.connect(self.set_status_text)
        self._slider_binding.targetSetIdsChanged.connect(
            lambda _ids: self._refresh_slider_target_summary()
        )
        self._slider_binding.previewDetailedBalanceCorrectionRequired.connect(
            self._mechanism_editor.show_preview_detailed_balance_block
        )
        self._slider_binding.previewDetailedBalanceCommitRequested.connect(
            self._commit_preview_detailed_balance_source
        )
        self._slider_binding.previewDetailedBalanceBlockCleared.connect(
            self._clear_preview_detailed_balance_block
        )
        self._mechanism_editor.detailedBalanceCorrectionCompleted.connect(
            self._simulation_jobs.complete_detailed_balance_correction
        )
        self._mechanism_editor.detailedBalanceCorrectionCompleted.connect(
            self._slider_binding.complete_preview_detailed_balance_correction
        )
        self._mechanism_editor.detailedBalanceCorrectionCompleted.connect(
            self._fit_session_owner.complete_detailed_balance_correction
        )
        self._initial_conditions_actions.errorRaised.connect(self.set_status_text)
        self._initial_conditions_actions.activeSnapshotChanged.connect(
            self._on_initial_snapshot_rendered
        )
        self._initial_conditions_actions.selectedRunSetIdsChanged.connect(
            self._on_run_selection_changed
        )
        selection_model = self._batch_table.selectionModel()
        selection_model.selectionChanged.connect(
            lambda _selected, _deselected: self._on_table_focus_changed()
        )
        selection_model.currentChanged.connect(
            lambda _current, _previous: self._on_table_focus_changed()
        )

        for signal in (
            self._batch_model.initialValueEditRequested,
            self._batch_model.setNameEditRequested,
            self._batch_model.plotMembershipChangeRequested,
            self._batch_model.sliderPinChangeRequested,
            self._mechanism_editor.sliderApplyRequested,
            self._mechanism_editor.sliderResetRequested,
        ):
            signal.connect(lambda *_args: QtCore.QTimer.singleShot(0, self._render_workspace))
        self._mechanism_editor.notesEdited.connect(self._on_notes_edited)

        self._simulation_jobs.state_changed.connect(self._refresh_actions)
        self._simulation_jobs.results_changed.connect(
            lambda tab_id: self._render_results_for_tab(str(tab_id))
        )
        self._simulation_jobs.status_changed.connect(self.set_status_text)
        self._simulation_jobs.detailed_balance_correction_requested.connect(
            self._mechanism_editor.present_detailed_balance_correction
        )
        self._fit_session_owner.window_changed.connect(
            lambda _open: self._render_workspace()
        )
        self._fit_session_owner.execution_changed.connect(
            lambda _executing: self._render_workspace()
        )
        self._fit_session_owner.canonical_applied.connect(
            self._on_fit_canonical_applied
        )
        self._fit_session_owner.status_changed.connect(self.set_status_text)
        self._fit_session_owner.detailed_balance_correction_requested.connect(
            self._mechanism_editor.present_detailed_balance_correction
        )

        self._dataset_import_staging.completionAvailable.connect(
            self._commit_import_completion
        )
        self._dataset_import_staging.busyChanged.connect(
            lambda _busy: self._refresh_actions()
        )
        self._dataset_import_staging.discarded.connect(self.set_status_text)
        self._dataset_import_panel.datasetRenameRequested.connect(
            self._rename_dataset
        )
        self._dataset_import_panel.datasetDeleteRequested.connect(
            self._delete_dataset
        )
        self._dataset_import_panel.datasetExportOriginalRequested.connect(
            self._export_original_dataset
        )
        self._dataset_import_panel.datasetMappingRequested.connect(
            self._map_dataset
        )
        self._dataset_import_panel.datasetObservationMappingRequested.connect(
            self._edit_observation_mapping
        )
        self._dataset_import_panel.datasetFitInclusionRequested.connect(
            self._set_dataset_fit_inclusion
        )
        self._overlay_panel.overlaySelectionRequested.connect(
            self._set_overlay_selection
        )
        self._overlay_panel.styleChanged.connect(self._render_all_results)
        self.config_controller.applicationPlotPreferencesChanged.connect(
            self._render_all_results
        )
        self._workspace.plotCreated.connect(self._connect_workspace_plot)
        self.config_controller.mechanismParameterDebounceChanged.connect(
            lambda _value: self._apply_slider_debounce_settings()
        )
        self.config_controller.equilibriumKDebounceChanged.connect(
            lambda _value: self._apply_slider_debounce_settings()
        )

    # ------------------------------------------------------------------
    # Direct render
    # ------------------------------------------------------------------
    def _render_active_tab_controls_only(self) -> None:
        """Refresh shared controls after a tab switch without touching results."""

        active_tab = self._workspace_tab_registry.active_tab
        presentation = self._workspace_tab_registry.active_presentation
        editor_projection = self._editor_binding.prepare_active_render()
        initial_projection = self._initial_conditions_actions.prepare_active_render()
        slider_projection = self._slider_binding.prepare_active_render()
        data_snapshot = build_data_manager_snapshot(
            document=self._project_session.document,
            active_tab=active_tab,
            presentation=presentation,
        )
        species_names = tuple(projected_species_names(active_tab))

        self._editor_binding.apply_render_projection(
            editor_projection,
            refresh_validation=False,
        )
        if self._workspace_tab_registry.active_runtime.reactions_draft is not None:
            self._mechanism_editor.set_reactions_status_text(
                "Visible mechanism source has changed; scientific actions will "
                "validate it.",
                state="draft",
            )
        elif active_tab.prepared_dsl is not None:
            self._mechanism_editor.set_reactions_status_text(
                "Accepted Reactions.",
                state="valid",
            )
        else:
            self._mechanism_editor.set_reactions_status_text("", state="idle")
        self._initial_conditions_actions.apply_render_projection(initial_projection)
        self._simulation_panel.render_run_defaults(
            active_tab.run_defaults,
            run_admitted=self._run_admitted(),
            simulation_job_kind=self._simulation_jobs.current_kind,
        )
        self._slider_binding.apply_render_projection(slider_projection)
        self._dataset_import_panel.render_snapshot(data_snapshot)
        self._overlay_panel.render_snapshot(
            records=data_snapshot.datasets,
            active_links=data_snapshot.active_links,
            known_species=species_names,
        )
        for tab in self._project_session.document.tabs:
            runtime = self._workspace_tab_registry.tab_runtime(tab.tab_id)
            self._workspace.set_tab_dirty(tab.tab_id, runtime.has_dirty_drafts)
            self._workspace.set_tab_close_enabled(
                tab.tab_id,
                len(self._project_session.document.tabs) > 1,
            )
        targets = tuple(slider_projection.targets)
        target_names = tuple(
            active_tab.initial_condition_set(set_id).name
            for set_id in targets
        )
        self._mechanism_editor.set_slider_edit_targets_summary(
            "Slider edit targets: "
            + (", ".join(target_names) if target_names else "none")
        )
        panel_projection = self._workspace.active_plot().applied_projection
        self._current_result_projection = (
            panel_projection
            if isinstance(panel_projection, WorkspaceResultProjection)
            and panel_projection.tab_id == active_tab.tab_id
            else None
        )
        self._sync_active_copy_all_capability()
        self._refresh_symbolic_calculator()
        self._refresh_title()
        self._refresh_actions()

    @QtCore.Slot(str, object)
    def _connect_workspace_plot(self, tab_id: str, plot: object) -> None:
        """Bind one retained plot only to intents from its owning tab."""

        target = str(tab_id)
        required_signals = (
            "referenceLayerVisibilityRequested",
            "interventionAnnotationsVisibilityRequested",
            "xSeriesRequested",
            "ySeriesRequested",
            "parametricModeRequested",
            "plotPresentationIntentRequested",
        )
        if any(
            not callable(getattr(getattr(plot, name, None), "connect", None))
            for name in required_signals
        ):
            raise TypeError("Workspace plot must expose the plot intent signals.")

        def active() -> bool:
            return target == self._workspace_tab_registry.active_tab_id

        plot.referenceLayerVisibilityRequested.connect(
            lambda visible: (
                self._set_reference_lines_visible(visible) if active() else None
            )
        )
        plot.interventionAnnotationsVisibilityRequested.connect(
            lambda visible: (
                self._set_intervention_annotations_visible(visible)
                if active()
                else None
            )
        )
        plot.xSeriesRequested.connect(
            lambda name: self._set_x_series(name) if active() else None
        )
        plot.ySeriesRequested.connect(
            lambda names: self._set_y_series(names) if active() else None
        )
        plot.parametricModeRequested.connect(
            lambda enabled: self._set_parametric_mode(enabled) if active() else None
        )
        plot.plotPresentationIntentRequested.connect(
            lambda intent: self._set_plot_presentation(intent) if active() else None
        )

    def _render_workspace(self) -> None:
        if self._rendering_workspace:
            return
        self._rendering_workspace = True
        try:
            try:
                projection = self._prepare_workspace_render_projection()
            except Exception as exc:
                self._invalidate_result_projection(
                    self._workspace_tab_registry.active_tab_id
                )
                self.set_status_text(f"Workspace refresh failed: {exc}")
                self._refresh_actions()
                raise
            failures = self._apply_workspace_render_projection(projection)
            if failures:
                self._invalidate_result_projection(projection.active_tab_id)
                self._current_result_projection = None
                self._workspace.set_active_copy_all_enabled(False)
                message = "Workspace refresh failed: " + "; ".join(failures)
                self.set_status_text(message)
                raise RuntimeError(message)
            self._slider_binding.settle_render_projection(
                projection.sliders
            )
            self._current_result_projection = projection.result_projection
            self._sync_active_copy_all_capability()
            self._refresh_symbolic_calculator()
            self._refresh_title()
            self._refresh_actions()
        finally:
            self._rendering_workspace = False

    def _prepare_workspace_render_projection(
        self,
    ) -> _WorkspaceRenderProjection:
        """Prepare every fallible workspace projection before widget mutation."""

        document = self._project_session.document
        active_tab = self._workspace_tab_registry.active_tab
        presentation = self._workspace_tab_registry.active_presentation
        prepared: dict[str, object] = {}
        failures: list[str] = []

        def prepare(label: str, operation) -> None:
            try:
                prepared[label] = operation()
            except (LookupError, RuntimeError, TypeError, ValueError) as exc:
                logger.exception("Workspace %s preparation failed", label)
                failures.append(f"{label}: {exc}")

        prepare(
            "dataset projection",
            lambda: build_data_manager_snapshot(
                document=document,
                active_tab=active_tab,
                presentation=presentation,
            ),
        )
        prepare(
            "result projection",
            lambda: self._compose_result_projection(active_tab.tab_id),
        )
        prepare(
            "species projection",
            lambda: tuple(projected_species_names(active_tab)),
        )
        prepare(
            "mechanism editor projection",
            self._editor_binding.prepare_active_render,
        )
        prepare(
            "initial conditions projection",
            self._initial_conditions_actions.prepare_active_render,
        )
        prepare(
            "slider projection",
            self._slider_binding.prepare_active_render,
        )
        prepare("run admission", self._run_admitted)
        if failures:
            message = "Workspace refresh failed: " + "; ".join(failures)
            raise RuntimeError(message)

        slider_projection = prepared["slider projection"]
        targets = tuple(getattr(slider_projection, "targets"))
        target_names = tuple(
            active_tab.initial_condition_set(set_id).name
            for set_id in targets
        )
        return _WorkspaceRenderProjection(
            document=document,
            active_tab_id=active_tab.tab_id,
            presentation=presentation,
            editor=prepared["mechanism editor projection"],
            initial_conditions=prepared["initial conditions projection"],
            sliders=slider_projection,
            data_snapshot=prepared["dataset projection"],
            result_projection=prepared["result projection"],
            species_names=prepared["species projection"],
            run_admitted=bool(prepared["run admission"]),
            simulation_job_kind=self._simulation_jobs.current_kind,
            tab_states=tuple(
                (
                    tab.tab_id,
                    self._workspace_tab_registry.tab_runtime(
                        tab.tab_id
                    ).has_dirty_drafts,
                    len(document.tabs) > 1,
                )
                for tab in document.tabs
            ),
            slider_target_summary=(
                "Slider edit targets: "
                + (", ".join(target_names) if target_names else "none")
            ),
        )

    def _apply_workspace_render_projection(
        self,
        projection: _WorkspaceRenderProjection,
    ) -> tuple[str, ...]:
        """Apply all consumers, returning typed failures for one report."""

        if not isinstance(projection, _WorkspaceRenderProjection):
            raise TypeError(
                "projection must be _WorkspaceRenderProjection."
            )
        active_tab = projection.document.tab(projection.active_tab_id)
        data_snapshot = projection.data_snapshot
        failures: list[str] = []

        def apply(label: str, operation) -> None:
            try:
                operation()
            except (LookupError, RuntimeError, TypeError, ValueError) as exc:
                logger.exception("Workspace %s application failed", label)
                failures.append(f"{label}: {exc}")

        apply(
            "tab strip",
            lambda: self._workspace.render(
                projection.document,
                active_tab_id=projection.active_tab_id,
            ),
        )
        apply(
            "plot presentation",
            lambda: self._apply_persisted_plot_states(projection.document),
        )
        apply(
            "mechanism editor",
            lambda: self._editor_binding.apply_render_projection(
                projection.editor
            ),
        )
        apply(
            "initial conditions",
            lambda: self._initial_conditions_actions.apply_render_projection(
                projection.initial_conditions
            ),
        )
        apply(
            "simulation settings",
            lambda: self._simulation_panel.render_run_defaults(
                active_tab.run_defaults,
                run_admitted=projection.run_admitted,
                simulation_job_kind=projection.simulation_job_kind,
            ),
        )
        apply(
            "sliders",
            lambda: self._slider_binding.apply_render_projection(
                projection.sliders
            ),
        )
        apply(
            "datasets",
            lambda: self._dataset_import_panel.render_snapshot(data_snapshot),
        )
        apply(
            "dataset overlays",
            lambda: self._overlay_panel.render_snapshot(
                records=data_snapshot.datasets,
                active_links=data_snapshot.active_links,
                known_species=projection.species_names,
            ),
        )
        apply(
            "results",
            lambda: self._result_presenter.apply(
                self._workspace.plot_for_tab(projection.active_tab_id),
                projection.result_projection,
            ),
        )

        def apply_tab_state() -> None:
            for tab_id, dirty, close_enabled in projection.tab_states:
                self._workspace.set_tab_dirty(tab_id, dirty)
                self._workspace.set_tab_close_enabled(
                    tab_id,
                    close_enabled,
                )

        apply("tab state", apply_tab_state)
        apply(
            "slider target summary",
            lambda: self._mechanism_editor.set_slider_edit_targets_summary(
                projection.slider_target_summary
            ),
        )
        return tuple(failures)

    def _apply_persisted_plot_states(self, document: ProjectDocument) -> None:
        """Apply each tab's persisted plot state to its retained panel."""

        for tab in document.tabs:
            state = self._workspace_tab_registry.presentation(tab.tab_id)
            self._workspace.plot_for_tab(tab.tab_id).set_persisted_plot_state(
                tab.tab_id,
                log_x=state.log_x,
                log_y=state.log_y,
                legend_visible=state.legend_visible,
                x_inverted=state.x_inverted,
                y_inverted=state.y_inverted,
            )

    def _compose_result_projection(
        self,
        tab_id: str,
    ) -> WorkspaceResultProjection:
        tab = self._project_session.document.tab(tab_id)
        presentation = self._workspace_tab_registry.presentation(tab_id)
        visible_sets = tuple(
            VisibleResultSet(
                set_id=set_id,
                label=tab.initial_condition_set(set_id).name,
            )
            for set_id in presentation.plot_set_ids
        )
        return self._result_presenter.compose(
            tab_id=tab_id,
            current_set_id=presentation.current_set_id,
            visible_sets=visible_sets,
            repository=self._repository_for_tab(tab_id),
            reference_lines_visible=presentation.reference_lines_visible,
            intervention_annotations_visible=(
                presentation.intervention_annotations_visible
            ),
            experimental_layers=build_experimental_layer_inputs(
                document=self._project_session.document,
                active_tab=tab,
                presentation=presentation,
            ),
            species_colors=self.config_controller.species_color_strings(
                projected_species_names(tab)
            ),
            experimental_point_size=(
                self.config_controller.experimental_point_size()
            ),
            experimental_point_opacity=(
                self.config_controller.experimental_point_opacity()
            ),
            selected_x_series=presentation.selected_x_series or "t",
            selected_y_series=presentation.selected_y_series,
            selected_y_uses_default_all=(
                presentation.selected_y_uses_default_all
            ),
            parametric_mode=presentation.parametric_mode,
        )

    def _render_results_for_tab(self, tab_id: str) -> bool:
        target = str(tab_id)
        if target != self._workspace_tab_registry.active_tab_id:
            try:
                projection = self._compose_result_projection(target)
                rendered = self._apply_result_projection(projection)
                self._sync_active_copy_all_capability()
                if not rendered:
                    self.set_status_text(
                        f"Result display failed for mechanism {target!r}; "
                        "its panel is unavailable until a later publication."
                    )
                return rendered
            except Exception as exc:
                logger.exception("Result render failed for inactive mechanism")
                self._invalidate_result_projection(target)
                self.set_status_text(
                    f"Result display failed for mechanism {target!r}: {exc}"
                )
                return False
        return self._render_results()

    def _render_all_results(self) -> None:
        """Apply one accepted result projection to every retained tab panel."""

        for tab in tuple(self._project_session.document.tabs):
            self._render_results_for_tab(tab.tab_id)

    def _render_results(self) -> bool:
        tab_id = self._workspace_tab_registry.active_tab_id
        try:
            projection = self._compose_result_projection(tab_id)
            if not self._apply_result_projection(projection):
                return False
            self._current_result_projection = projection
            self._sync_active_copy_all_capability()
            return True
        except Exception as exc:
            logger.exception("Result render failed")
            self._invalidate_result_projection(tab_id)
            self.set_status_text(
                "Result display failed for the active mechanism; Copy and XLSX "
                f"are disabled until a current projection renders successfully: {exc}"
            )
            self._refresh_actions()
            return False

    def _apply_result_projection(
        self,
        projection: WorkspaceResultProjection,
    ) -> bool:
        try:
            self._result_presenter.apply(
                self._workspace.plot_for_tab(projection.tab_id),
                projection,
            )
        except Exception as exc:
            self._invalidate_after_result_render_failure(
                projection.tab_id,
                str(exc),
            )
            raise
        return True

    def _invalidate_result_projection(self, tab_id: str) -> None:
        """Remove a failed tab's physical result surface without touching others."""

        target = str(tab_id)
        if target == self._workspace_tab_registry.active_tab_id:
            self._current_result_projection = None
            self._workspace.set_active_copy_all_enabled(False)
        try:
            self._workspace.plot_for_tab(target).clear_workspace_result_projection()
        except (KeyError, RuntimeError, TypeError, ValueError):
            logger.exception("Could not clear failed workspace result surface")

    def _active_current_export_projection(
        self,
    ) -> WorkspaceResultProjection | None:
        """Return export truth only when the live active panel applied it."""

        projection = self._current_result_projection
        if (
            projection is None
            or projection.tab_id != self._workspace_tab_registry.active_tab_id
            or not projection.copy_all_blocks
        ):
            return None
        try:
            panel = self._workspace.plot_for_tab(projection.tab_id)
        except (KeyError, RuntimeError):
            return None
        return projection if panel.applied_projection is projection else None

    def _sync_active_copy_all_capability(self) -> None:
        self._workspace.set_active_copy_all_enabled(
            self._active_current_export_projection() is not None
        )

    def _repository_for_tab(self, tab_id: str) -> ResultRepository:
        repository = self._workspace_tab_registry.tab_runtime(tab_id).result_repository
        if not isinstance(repository, ResultRepository):
            raise LookupError(f"Tab {tab_id!r} has no ResultRepository.")
        return repository

    def _consume_scientific_document_delta(
        self,
        delta: ScientificDocumentDelta | None,
    ) -> None:
        """Fence only the existing Run targets affected by one registry mutation."""
        if delta is None:
            return
        if not isinstance(delta, ScientificDocumentDelta):
            raise TypeError(
                "Registry mutations must return ScientificDocumentDelta or None."
            )
        for tab_id, set_ids in delta.affected_targets_by_tab:
            if set_ids:
                self._simulation_jobs.deauthorize_targets(tab_id, set_ids)

    def _explicit_result_key_is_current(self, key: ExplicitResultKey) -> bool:
        if not isinstance(key, ExplicitResultKey):
            return False
        try:
            tab = self._project_session.document.tab(key.tab_id)
            initial_values = projected_initial_values(tab, key.set_id)
            resolved_settings = tab.resolved_execution_settings(key.set_id)
        except (KeyError, ValueError):
            return False
        return bool(
            key.mechanism_source == tab.mechanism_source
            and key.analysis_identity == projected_analysis_identity(tab)
            and key.initial_values == initial_values
            and key.execution_settings == resolved_settings
            and key.scientific_settings == tab.scientific_settings
        )

    def _render_tab_state(self) -> None:
        tabs = self._project_session.document.tabs
        for tab in tabs:
            runtime = self._workspace_tab_registry.tab_runtime(tab.tab_id)
            self._workspace.set_tab_dirty(tab.tab_id, runtime.has_dirty_drafts)
            self._workspace.set_tab_close_enabled(
                tab.tab_id,
                len(tabs) > 1,
            )

    def _on_initial_snapshot_rendered(self) -> None:
        if self._rendering_workspace:
            return
        self._slider_binding.render_active()
        self._render_results()
        self._refresh_title()

    @QtCore.Slot(object)
    def _on_run_selection_changed(self, _set_ids: object) -> None:
        self._render_workspace()

    def _on_table_focus_changed(self) -> None:
        if self._rendering_workspace:
            return
        self._slider_binding.render_active()
        self._render_results()
        self._refresh_slider_target_summary()
        self._refresh_project_state()

    # ------------------------------------------------------------------
    # Mechanism tab and shared-widget intents
    # ------------------------------------------------------------------
    @QtCore.Slot(str)
    def _select_tab(self, tab_id: str) -> None:
        target = str(tab_id)
        previous = self._workspace_tab_registry.active_tab_id
        if target == previous:
            return
        if not self._settle_execution_candidate():
            self._workspace.select_tab_surface(previous)
            return
        self._workspace_tab_registry.select_tab(target)
        self._workspace.select_tab_surface(target)
        try:
            self._render_active_tab_controls_only()
        except Exception as exc:
            self.set_status_text(f"Could not switch mechanism tab: {exc}")
            logger.exception("Mechanism tab control refresh failed")

    @QtCore.Slot()
    def _create_tab(self) -> None:
        if not self._settle_execution_candidate():
            QtCore.QTimer.singleShot(0, self._render_workspace)
            return
        self._workspace_tab_registry.create_tab(
            run_defaults=self.config_controller.fresh_tab_run_defaults(),
            scientific_settings=(
                self.config_controller.fresh_tab_scientific_defaults()
            ),
        )
        self._render_workspace()

    @QtCore.Slot(str, str)
    def _rename_tab(self, tab_id: str, name: str) -> None:
        try:
            self._consume_scientific_document_delta(
                self._workspace_tab_registry.rename_tab(tab_id, name)
            )
        except Exception as exc:
            self._workspace.show_rename_error(tab_id, str(exc))
            return
        self._render_workspace()

    @QtCore.Slot(str)
    def _close_tab(self, tab_id: str) -> None:
        if not self._settle_execution_candidate():
            return
        document = self._project_session.document
        if len(document.tabs) <= 1:
            self.set_status_text("The last mechanism tab cannot be closed.")
            return
        runtime = self._workspace_tab_registry.tab_runtime(tab_id)
        invalid_reactions = bool(
            runtime.reactions_draft is not None
            and isinstance(
                reconcile_tab_mechanism_source(
                    document,
                    tab_id,
                    runtime.reactions_draft,
                ),
                MechanismSourceFailure,
            )
        )
        invalid_cells = bool(runtime.invalid_initial_text_by_set_id)
        has_content = self._workspace_tab_registry.tab_has_content_or_drafts(
            tab_id,
            fresh_run_defaults=self.config_controller.fresh_tab_run_defaults(),
            fresh_scientific_settings=(
                self.config_controller.fresh_tab_scientific_defaults()
            ),
        )
        if has_content:
            dialog = QtWidgets.QMessageBox(self)
            dialog.setWindowTitle("Close Mechanism")
            dialog.setIcon(QtWidgets.QMessageBox.Icon.Warning)
            details = [
                "Close this mechanism tab and discard its canonical content?"
            ]
            if runtime.reactions_draft is not None:
                details.append("Visible Reactions edits will be discarded.")
            if runtime.notes_draft is not None:
                details.append("Visible Notes edits will be discarded.")
            if runtime.invalid_initial_text_by_set_id:
                details.append("Invalid Initial Conditions edits will be discarded.")
            if (
                runtime.slider_parameter_draft
                or runtime.slider_species_drafts_by_set_id
            ):
                details.append("Slider Preview drafts will be discarded.")
            dialog.setText("\n\n".join(details))
            discard_label = "Discard and Close"
            if invalid_reactions and invalid_cells:
                discard_label = (
                    "Discard Invalid Reactions and Cell Edit(s) and Continue"
                )
            elif invalid_reactions:
                discard_label = "Discard Invalid Reactions and Continue"
            elif invalid_cells:
                discard_label = "Discard Invalid Cell Edit(s) and Continue"
            discard = dialog.addButton(
                discard_label,
                QtWidgets.QMessageBox.ButtonRole.DestructiveRole,
            )
            if invalid_reactions or invalid_cells:
                dialog.addButton(
                    "Return and Fix",
                    QtWidgets.QMessageBox.ButtonRole.RejectRole,
                )
            dialog.addButton(QtWidgets.QMessageBox.StandardButton.Cancel)
            dialog.exec()
            if dialog.clickedButton() is not discard:
                return
            if invalid_reactions and not invalid_cells:
                self._editor_binding.discard_reactions_draft(tab_id)
            elif invalid_cells and not invalid_reactions:
                runtime.invalid_initial_text_by_set_id.clear()
        self._slider_binding.drop_tab_runtime(tab_id)
        self._workspace.drop_tab_session_state(tab_id)
        self._editor_binding.discard_reactions_draft(tab_id)
        runtime.invalid_initial_text_by_set_id.clear()
        self._workspace_tab_registry.close_tab(tab_id)
        self._render_workspace()

    def _add_initial_set(self) -> None:
        active_tab_id = self._workspace_tab_registry.active_tab_id
        if not self._accept_visible_source(
            active_tab_id,
            description="Add Initial Conditions Set",
        ):
            return
        if self._initial_conditions_actions.add_set() is not None:
            self._render_workspace()

    @QtCore.Slot(int)
    def _move_initial_sets(self, delta: int) -> None:
        self._initial_conditions_actions.move_selected(delta)
        self._render_workspace()

    def _delete_initial_sets(self) -> None:
        targets = self._batch_table.selected_set_ids()
        if not targets:
            return
        names = [
            self._workspace_tab_registry.active_tab.initial_condition_set(set_id).name
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
        self._initial_conditions_actions.delete_selected()
        self._render_workspace()

    @QtCore.Slot(object)
    def _set_workspace_run_defaults(
        self,
        settings: SimulationExecutionSettings,
    ) -> None:
        self._initial_conditions_actions.set_workspace_run_defaults(settings)
        self._render_workspace()

    @QtCore.Slot(str, bool)
    def _on_plot_membership_requested(
        self,
        set_id: str,
        requested: bool,
    ) -> None:
        active_tab_id = self._workspace_tab_registry.active_tab_id
        if not self._accept_visible_source(
            active_tab_id,
            description="Plot membership",
        ):
            self._initial_conditions_actions.render_active()
            return
        self._initial_conditions_actions.set_plot_requested(
            set_id,
            requested,
        )
        if not requested:
            self._render_results()
            return
        tab = self._workspace_tab_registry.active_tab
        try:
            key = ExplicitResultKey(
                tab_id=tab.tab_id,
                set_id=set_id,
                mechanism_source=tab.mechanism_source,
                analysis_identity=projected_analysis_identity(tab),
                standard_concentration_identity=standard_concentration_identity(
                    tab.prepared_dsl
                ),
                scientific_identity=scientific_identity(tab.prepared_dsl),
                initial_values=projected_initial_values(tab, set_id),
                execution_settings=tab.resolved_execution_settings(set_id),
                scientific_settings=tab.scientific_settings,
            )
            if not self._simulation_jobs.show_cached(key):
                self._render_results()
        except Exception as exc:
            self.set_status_text(f"Could not show cached result: {exc}")

    @QtCore.Slot(str, object)
    def _on_fit_canonical_applied(
        self,
        tab_id: str,
        _scopes: object,
    ) -> None:
        try:
            self._project_session.document.tab(tab_id)
        except KeyError:
            return
        self._render_workspace()

    def _run_admitted(self) -> bool:
        selected = self._initial_conditions_actions.selected_run_set_ids()
        if not selected or not self._computation_gate.is_idle:
            return False
        if self._initial_conditions_actions.has_invalid_cells(selected):
            return False
        source = self._visible_source_for_tab(
            self._workspace_tab_registry.active_tab_id
        )
        return bool(source.reactions_text.strip())

    @QtCore.Slot(bool)
    def _set_shared_run_enabled(self, enabled: bool) -> None:
        admitted = bool(enabled)
        if self._run_action is not None:
            self._run_action.setEnabled(admitted)
        self._mechanism_editor.set_run_enabled(admitted)
        self._simulation_panel.run_btn.setEnabled(admitted)

    def _settle_execution_candidate(self) -> bool:
        return bool(self._simulation_panel.settle_run_defaults())

    def _preview_preflight(self, target_ids: tuple[str, ...]) -> bool:
        if not self._settle_execution_candidate():
            return False
        if not self._accept_visible_source(description="Preview"):
            return False
        if self._initial_conditions_actions.has_invalid_cells(target_ids):
            self.set_status_text(
                "Preview blocked by invalid visible Initial Conditions."
            )
            return False
        return True

    def _confirm_discard_slider_preview(self, action: str) -> bool:
        if not self._slider_binding.has_slider_draft():
            return True
        transaction = self._slider_binding.suspend_preview_disposition()
        try:
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
        except Exception:
            self._slider_binding.resume_preview_disposition(transaction)
            raise
        if dialog.clickedButton() is not discard:
            self._slider_binding.resume_preview_disposition(transaction)
            return False
        self._slider_binding.discard_preview_disposition(transaction)
        self._slider_binding.discard_slider_draft()
        return True

    @QtCore.Slot()
    def _start_selected_run(self) -> bool:
        selected = self._initial_conditions_actions.selected_run_set_ids()
        if not selected:
            self.set_status_text("Select one or more Initial Conditions rows.")
            return False
        if self._initial_conditions_actions.has_invalid_cells(selected):
            self.set_status_text(
                "Run blocked by invalid visible Initial Conditions in a selected row."
            )
            return False
        if not self._computation_gate.is_idle:
            self.set_status_text(
                "Run rejected: another computation is active."
            )
            return False
        if not self._confirm_discard_slider_preview("Run"):
            return False
        if not self._settle_execution_candidate():
            return False
        if not self._accept_visible_source(description="Run"):
            return False
        selected = self._initial_conditions_actions.selected_run_set_ids()
        if not selected:
            self.set_status_text("Select one or more Initial Conditions rows.")
            return False
        if self._initial_conditions_actions.has_invalid_cells(selected):
            self.set_status_text(
                "Run blocked by invalid visible Initial Conditions in a selected row."
            )
            return False
        if not self._computation_gate.is_idle:
            self.set_status_text(
                "Run rejected: another computation is active."
            )
            return False
        tab = self._workspace_tab_registry.active_tab
        try:
            targets = tuple(
                build_explicit_target(
                    prepared_dsl=tab.prepared_dsl,
                    tab_id=tab.tab_id,
                    set_id=set_id,
                    set_name=tab.initial_condition_set(set_id).name,
                    mechanism_source=tab.mechanism_source,
                    initial_values=projected_initial_values(tab, set_id),
                    execution_settings=tab.resolved_execution_settings(set_id),
                    scientific_settings=tab.scientific_settings,
                    publication_generation=(
                        self._simulation_jobs.explicit_publication_generation(
                            tab.tab_id,
                            set_id,
                        )
                    ),
                )
                for set_id in selected
            )
            capture = ExplicitRunCapture(tab_id=tab.tab_id, targets=targets)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Run", str(exc))
            return False
        self._slider_binding.cancel_pending_preview_intent(tab.tab_id)
        return bool(self._simulation_jobs.start_run(capture))

    def _inspect_transient_state(self) -> ProjectTransientState:
        reactions_draft = False
        notes_draft = False
        invalid_reactions = False
        invalid_initials = False
        slider_drafts = False
        for tab in self._project_session.document.tabs:
            runtime = self._workspace_tab_registry.tab_runtime(tab.tab_id)
            if runtime.reactions_draft is not None:
                reactions_draft = True
                if isinstance(
                    reconcile_tab_mechanism_source(
                        self._project_session.document,
                        tab.tab_id,
                        runtime.reactions_draft,
                    ),
                    MechanismSourceFailure,
                ):
                    invalid_reactions = True
            notes_draft = bool(
                notes_draft or runtime.notes_draft is not None
            )
            invalid_initials = bool(
                invalid_initials or runtime.invalid_initial_text_by_set_id
            )
            slider_drafts = bool(
                slider_drafts
                or runtime.slider_parameter_draft
                or runtime.slider_species_drafts_by_set_id
            )
        return ProjectTransientState(
            reactions_draft=reactions_draft,
            notes_draft=notes_draft,
            invalid_reactions=invalid_reactions,
            invalid_initial_conditions=invalid_initials,
            slider_drafts=slider_drafts,
        )

    def _prepare_before_save(self) -> ProjectDocument | None:
        """Build the exact archive candidate without disposing visible drafts."""

        if not self._settle_execution_candidate():
            return None
        candidate = self._project_session.document
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
                    report_error=True,
                )
                if proposal is None:
                    return None
                candidate = proposal.proposed_document
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
        if any(
            self._workspace_tab_registry.tab_runtime(tab_id)
            .invalid_initial_text_by_set_id
            for tab_id in tab_ids
        ):
            QtWidgets.QMessageBox.warning(
                self,
                "Save Project",
                "Save is blocked while any visible Initial Conditions cell is invalid.",
            )
            return None
        return candidate

    def _commit_saved_document(self, document: ProjectDocument) -> None:
        """Publish a successfully written archive candidate exactly once."""

        self._consume_scientific_document_delta(
            self._workspace_tab_registry.replace_document(document)
        )
        for tab in document.tabs:
            runtime = self._workspace_tab_registry.tab_runtime(tab.tab_id)
            if (
                tab.tab_id == self._workspace_tab_registry.active_tab_id
                and runtime.reactions_draft is not None
            ):
                try:
                    self._editor_binding.apply_accepted_source_rewrite(
                        runtime.reactions_draft,
                        tab.mechanism_source,
                    )
                except RuntimeError as exc:
                    logger.warning(
                        "Saved Reactions provenance could not join native "
                        "Undo history: %s",
                        exc,
                    )
            runtime.reactions_draft = None
            runtime.notes_draft = None
            valid_species = set(projected_species_names(tab))
            runtime.invalid_initial_text_by_set_id = {
                set_id: {
                    species: raw
                    for species, raw in values.items()
                    if species in valid_species
                }
                for set_id, values
                in runtime.invalid_initial_text_by_set_id.items()
                if any(species in valid_species for species in values)
            }

    def _discard_authoring(self, mode: AuthoringDiscardMode) -> None:
        for tab in self._project_session.document.tabs:
            runtime = self._workspace_tab_registry.tab_runtime(tab.tab_id)
            if mode is AuthoringDiscardMode.ALL_UNACCEPTED or (
                runtime.reactions_draft is not None
                and isinstance(
                    reconcile_tab_mechanism_source(
                        self._project_session.document,
                        tab.tab_id,
                        runtime.reactions_draft,
                    ),
                    MechanismSourceFailure,
                )
            ):
                runtime.reactions_draft = None
            if mode is AuthoringDiscardMode.ALL_UNACCEPTED:
                runtime.notes_draft = None
            runtime.invalid_initial_text_by_set_id.clear()
        self._editor_binding.render_active()
        self._initial_conditions_actions.render_active()

    def _discard_all_slider_drafts(self) -> None:
        for tab in self._project_session.document.tabs:
            self._slider_binding.discard_slider_draft(tab.tab_id)

    def _on_draft_state_changed(self, tab_id: str, _dirty: bool) -> None:
        if self._rendering_workspace:
            return
        current_tab_ids = {
            tab.tab_id for tab in self._project_session.document.tabs
        }
        if tab_id not in current_tab_ids:
            return
        runtime = self._workspace_tab_registry.tab_runtime(tab_id)
        self._workspace.set_tab_dirty(tab_id, runtime.has_dirty_drafts)
        if tab_id == self._workspace_tab_registry.active_tab_id:
            self._set_shared_run_enabled(self._run_admitted())
        self._refresh_title()

    @QtCore.Slot(str)
    def _on_notes_edited(self, _notes: str) -> None:
        self._refresh_title()
        self._refresh_actions()

    def _refresh_slider_target_summary(self) -> None:
        targets = self._slider_binding.resolve_target_set_ids()
        tab = self._workspace_tab_registry.active_tab
        names = [tab.initial_condition_set(set_id).name for set_id in targets]
        self._mechanism_editor.set_slider_edit_targets_summary(
            "Slider edit targets: " + (", ".join(names) if names else "none")
        )

    # ------------------------------------------------------------------
    # Source/examples/templates
    # ------------------------------------------------------------------
    def _visible_source_for_tab(self, tab_id: str) -> MechanismAuthoringSource:
        runtime = self._workspace_tab_registry.tab_runtime(tab_id)
        return (
            runtime.reactions_draft
            or self._project_session.document.tab(tab_id).mechanism_source
        )

    def _accept_visible_source(
        self,
        tab_id: str | None = None,
        *,
        description: str,
        source: MechanismAuthoringSource | None = None,
        report_error: bool = True,
    ) -> bool:
        """Run the one pure reconciliation and commit its complete proposal once."""
        tab_id = (
            self._workspace_tab_registry.active_tab_id
            if tab_id is None
            else str(tab_id)
        )
        visible = source or self._visible_source_for_tab(tab_id)
        document = self._project_session.document
        proposal = self._prepare_source_reconciliation(
            document,
            tab_id,
            visible,
            description=description,
            report_error=report_error,
        )
        if proposal is None:
            return False

        proposed_document = proposal.proposed_document
        proposed_tab = proposed_document.tab(tab_id)
        source_changed = document != proposed_document
        if source_changed:
            self._consume_scientific_document_delta(
                self._workspace_tab_registry.replace_document(
                    proposed_document
                )
            )
        runtime = self._workspace_tab_registry.tab_runtime(tab_id)
        if tab_id == self._workspace_tab_registry.active_tab_id:
            self._editor_binding.apply_accepted_source_rewrite(
                visible,
                proposed_tab.mechanism_source,
            )
        runtime.reactions_draft = None
        valid_species = set(projected_species_names(proposed_tab))
        runtime.invalid_initial_text_by_set_id = {
            set_id: {
                species: raw
                for species, raw in values.items()
                if species in valid_species
            }
            for set_id, values in runtime.invalid_initial_text_by_set_id.items()
            if any(species in valid_species for species in values)
        }
        if tab_id == self._workspace_tab_registry.active_tab_id:
            self._render_workspace()
        return True

    def _prepare_source_reconciliation(
        self,
        document,
        tab_id: str,
        visible: MechanismAuthoringSource,
        *,
        description: str,
        report_error: bool,
    ) -> MechanismSourceReconciliation | None:
        """Resolve Replace/Cancel while keeping the proposal mutation-free."""

        proposal = reconcile_tab_mechanism_source(document, tab_id, visible)
        if isinstance(proposal, MechanismSourceCollision):
            if not report_error:
                return None
            names = ", ".join(repr(name) for name in proposal.existing_names)
            dialog = QtWidgets.QMessageBox(self)
            dialog.setWindowTitle("Replace Initial Conditions")
            dialog.setIcon(QtWidgets.QMessageBox.Icon.Warning)
            dialog.setText(
                f"{description} contains named initial-condition block(s) "
                f"that target existing set(s): {names}.\n\n"
                "Replace those sets atomically with the imported values?"
            )
            replace_button = dialog.addButton(
                "Replace",
                QtWidgets.QMessageBox.ButtonRole.AcceptRole,
            )
            dialog.addButton(QtWidgets.QMessageBox.StandardButton.Cancel)
            dialog.exec()
            if dialog.clickedButton() is not replace_button:
                return None
            proposal = reconcile_tab_mechanism_source(
                document,
                tab_id,
                visible,
                replace_initial_condition_names=proposal.existing_names,
            )
        if isinstance(proposal, MechanismSourceFailure):
            if report_error:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Invalid Reactions",
                    f"{description} cannot continue:\n\n{proposal.message}",
                )
            return None
        if not isinstance(proposal, MechanismSourceReconciliation):
            raise TypeError("Mechanism reconciliation returned an invalid result.")
        return proposal

    def _commit_detailed_balance_source(
        self,
        tab_id: str,
        source: MechanismAuthoringSource,
    ) -> PreparedDslRequestSet | None:
        if not isinstance(source, MechanismAuthoringSource):
            raise TypeError("source must be MechanismAuthoringSource.")
        target = str(tab_id)
        try:
            self._project_session.document.tab(target)
        except KeyError:
            return None
        if not self._accept_visible_source(
            target,
            description="Apply detailed-balance correction",
            source=source,
            report_error=False,
        ):
            return None
        accepted = self._project_session.document.tab(target).prepared_dsl
        if not isinstance(accepted, PreparedDslRequestSet):
            raise RuntimeError(
                "Accepted detailed-balance source has no retained prepared DSL."
            )
        return accepted

    @QtCore.Slot(object)
    def _commit_preview_detailed_balance_source(
        self,
        outcome: DetailedBalanceCorrectionOutcome,
    ) -> None:
        if not isinstance(outcome, DetailedBalanceCorrectionOutcome):
            return
        if outcome.origin.strip().casefold() != "preview":
            return
        corrected = outcome.corrected_source
        if corrected is None:
            return
        try:
            current = self._visible_source_for_tab(outcome.tab_id)
        except KeyError:
            return
        if current.reactions_text not in {outcome.source, corrected}:
            return
        candidate = replace(current, reactions_text=corrected)
        if self._commit_detailed_balance_source(outcome.tab_id, candidate):
            self.set_status_text(
                "Detailed-balance correction applied. Preview was not retried; "
                "adjust a Slider to request a new Preview."
            )
            self._render_workspace()

    @QtCore.Slot(str)
    def _clear_preview_detailed_balance_block(self, tab_id: str) -> None:
        if str(tab_id) == self._workspace_tab_registry.active_tab_id:
            self._mechanism_editor.clear_preview_detailed_balance_block()

    def _accept_external_source(
        self,
        source: MechanismAuthoringSource,
        *,
        description: str,
    ) -> bool:
        tab_id = self._workspace_tab_registry.active_tab_id
        runtime = self._workspace_tab_registry.active_runtime
        if (
            runtime.reactions_draft is not None
            or runtime.slider_parameter_draft
            or runtime.slider_species_drafts_by_set_id
        ):
            answer = QtWidgets.QMessageBox.question(
                self,
                "Replace Active Mechanism",
                "Loading this source affects only the active mechanism tab and "
                "will discard that tab's unapplied Reactions and Slider drafts. "
                "Continue?",
                QtWidgets.QMessageBox.StandardButton.Yes
                | QtWidgets.QMessageBox.StandardButton.Cancel,
                QtWidgets.QMessageBox.StandardButton.Cancel,
            )
            if answer != QtWidgets.QMessageBox.StandardButton.Yes:
                return False
        if not self._accept_visible_source(
            tab_id,
            description=description,
            source=source,
        ):
            return False
        runtime.reactions_draft = None
        self._slider_binding.discard_slider_draft(tab_id)
        self._render_workspace()
        return True

    def _load_preset_mechanism(self, preset_id: str) -> None:
        try:
            from kindred.io.resources import get_preset_mechanism_source

            source = get_preset_mechanism_source(preset_id)
            if self._accept_external_source(
                source,
                description=f"Load preset {preset_id}",
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
        try:
            from kindred.io.resources import get_intervention_example_source

            source = get_intervention_example_source(example_id)
            if self._accept_external_source(
                source,
                description=f"Load intervention example {example_id}",
            ):
                self.set_status_text(f"Loaded intervention example: {example_id}")
        except Exception as exc:
            logger.exception("Intervention example load failed")
            QtWidgets.QMessageBox.critical(
                self,
                "Load Error",
                f"Failed to load intervention example {example_id}:\n\n{exc}",
            )

    def _open_template_manager(self) -> None:
        from kindred.gui.widgets.template_manager_dialog import TemplateManagerDialog

        if not self._accept_visible_source(
            description="Open Template Manager"
        ):
            return
        source_tab = self._workspace_tab_registry.active_tab
        dialog = TemplateManagerDialog(
            parent=self,
            template_manager=self._template_manager,
            current_mechanism_source=source_tab.mechanism_source,
            source_preflight=lambda source: self._preflight_template_source(
                source_tab.tab_id,
                source,
            ),
        )
        dialog.templateApplyRequested.connect(
            lambda source: self._accept_external_source(
                source,
                description="Apply mechanism template",
            )
        )
        dialog.exec()

    def _preflight_template_source(
        self,
        source_tab_id: str,
        source: MechanismAuthoringSource,
    ) -> MechanismAuthoringSource:
        """Scientifically admit a template candidate in its source tab context."""

        if not isinstance(source, MechanismAuthoringSource):
            raise TypeError("template source must be MechanismAuthoringSource.")
        source_tab = self._project_session.document.tab(str(source_tab_id))
        prepare_dsl_request_set(
            source.reactions_text,
            initials={},
            units=UnitsModel(
                temperature_K=source_tab.scientific_settings.temperature,
            ),
        )
        return source

    # ------------------------------------------------------------------
    # Dataset intents
    # ------------------------------------------------------------------
    @QtCore.Slot(int)
    def _commit_import_completion(self, import_id: int) -> None:
        completion = self._dataset_import_staging.take_completion_for_commit(import_id)
        if completion is None:
            return
        try:
            batch_proposal = raw_record_batch_proposal(
                completion,
                existing_records=self._project_session.document.datasets,
            )
            document = self._project_session.document.add_datasets(
                batch_proposal.records
            )
            self._consume_scientific_document_delta(
                self._workspace_tab_registry.replace_document(document)
            )
        except Exception as exc:
            logger.exception("Raw dataset commit failed")
            QtWidgets.QMessageBox.critical(
                self,
                "Dataset Import",
                f"The normalized import could not be committed:\n\n{exc}",
            )
            self._render_workspace()
            return

        self._render_workspace()
        message = f"Imported {len(batch_proposal.records)} raw dataset(s)."
        if batch_proposal.errors:
            message += " " + "; ".join(batch_proposal.errors)
        if not batch_proposal.records:
            self.set_status_text(message)
            return

        active_tab_id = self._workspace_tab_registry.active_tab_id
        visible = self._visible_source_for_tab(active_tab_id)
        proposal = reconcile_tab_mechanism_source(
            self._project_session.document,
            active_tab_id,
            visible,
        )
        if not isinstance(proposal, MechanismSourceReconciliation):
            self.set_status_text(
                message
                + " Visible Reactions requires correction or initial-set "
                "replacement consent, so optional mapping was skipped."
                + " Raw data remains committed and unmapped; fix Reactions, then"
                + " use Map."
            )
            return

        self.set_status_text(
            message + " Raw data was committed before optional mapping."
        )
        for record in batch_proposal.records:
            self._offer_import_mapping(record.dataset_id, record.display_name)

    def _offer_import_mapping(self, dataset_id: str, display_name: str) -> None:
        dialog = QtWidgets.QMessageBox(self)
        dialog.setWindowTitle("Map Imported Dataset")
        dialog.setIcon(QtWidgets.QMessageBox.Icon.Question)
        dialog.setText(
            f"Raw dataset '{display_name}' is committed.\n\n"
            "Choose its optional mapping in the active mechanism tab."
        )
        create_set = dialog.addButton(
            "Create Set From t=0",
            QtWidgets.QMessageBox.ButtonRole.AcceptRole,
        )
        map_existing = dialog.addButton(
            "Map Existing",
            QtWidgets.QMessageBox.ButtonRole.ActionRole,
        )
        leave_unmapped = dialog.addButton(
            "Leave Unmapped",
            QtWidgets.QMessageBox.ButtonRole.RejectRole,
        )
        dialog.setDefaultButton(leave_unmapped)
        dialog.exec()
        clicked = dialog.clickedButton()
        if clicked is create_set:
            intent = DatasetMappingIntent(
                dataset_id=dataset_id,
                action=DatasetMappingChoice.CREATE_SET.value,
            )
        elif clicked is map_existing:
            intent = self._prompt_import_existing_mapping_intent(dataset_id)
            if intent is None:
                return
        else:
            intent = DatasetMappingIntent(
                dataset_id=dataset_id,
                action=DatasetMappingChoice.LEAVE_UNMAPPED.value,
            )
        self._map_dataset(intent)

    def _prompt_import_existing_mapping_intent(
        self,
        dataset_id: str,
    ) -> DatasetMappingIntent | None:
        tab = self._workspace_tab_registry.active_tab
        if not tab.initial_condition_sets:
            QtWidgets.QMessageBox.information(
                self,
                "Map Existing",
                "The active mechanism tab has no available initial-condition set.",
            )
            return None
        labels = [item.name for item in tab.initial_condition_sets]
        label, accepted = QtWidgets.QInputDialog.getItem(
            self,
            "Map Existing",
            "Initial-condition set:",
            labels,
            editable=False,
        )
        if not accepted:
            return None
        target = next(
            item
            for item in tab.initial_condition_sets
            if item.name == str(label)
        )
        occupant = next(
            (
                linked_dataset_id
                for linked_dataset_id, link in tab.dataset_links
                if link.set_id == target.set_id
                and linked_dataset_id != str(dataset_id)
            ),
            None,
        )
        occupied_choice = None
        if occupant is not None:
            dialog = QtWidgets.QMessageBox(self)
            dialog.setWindowTitle("Set Already Mapped")
            dialog.setIcon(QtWidgets.QMessageBox.Icon.Question)
            dialog.setText(f"Set '{target.name}' is already mapped in the active tab.")
            override = dialog.addButton(
                "Override",
                QtWidgets.QMessageBox.ButtonRole.AcceptRole,
            )
            create_copy = dialog.addButton(
                "Create Copy",
                QtWidgets.QMessageBox.ButtonRole.ActionRole,
            )
            cancel = dialog.addButton(
                "Cancel",
                QtWidgets.QMessageBox.ButtonRole.RejectRole,
            )
            dialog.setDefaultButton(cancel)
            dialog.exec()
            clicked = dialog.clickedButton()
            if clicked is override:
                occupied_choice = OccupiedSetChoice.OVERRIDE.value
            elif clicked is create_copy:
                occupied_choice = OccupiedSetChoice.CREATE_COPY.value
            else:
                return None
        return DatasetMappingIntent(
            dataset_id=dataset_id,
            action=DatasetMappingChoice.MAP_EXISTING.value,
            set_id=target.set_id,
            occupied_choice=occupied_choice,
        )

    @QtCore.Slot(object)
    def _rename_dataset(self, intent: DatasetRenameIntent) -> None:
        if not isinstance(intent, DatasetRenameIntent):
            return
        try:
            name = validate_unique_dataset_name(
                intent.display_name,
                existing=(
                    (record.dataset_id, record.display_name)
                    for record in self._project_session.document.datasets
                ),
                excluding_dataset_id=intent.dataset_id,
            )
            candidate = self._project_session.document.rename_dataset(
                intent.dataset_id,
                name,
            )
            self._consume_scientific_document_delta(
                self._workspace_tab_registry.replace_document(candidate)
            )
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Rename Dataset", str(exc))
        self._render_workspace()

    @QtCore.Slot(object)
    def _delete_dataset(self, intent: DatasetDeleteIntent) -> None:
        if not isinstance(intent, DatasetDeleteIntent):
            return
        presentations = dict(self._workspace_tab_registry.presentation_items())
        plan = plan_global_dataset_delete(
            dataset_id=intent.dataset_id,
            links_by_tab={
                tab.tab_id: tab.dataset_links
                for tab in self._project_session.document.tabs
            },
            overlays_by_tab={
                tab_id: state.overlay_columns_by_dataset
                for tab_id, state in presentations.items()
            },
        )
        if plan.referenced:
            answer = QtWidgets.QMessageBox.question(
                self,
                "Delete Dataset",
                "This dataset is referenced by one or more mechanism tabs. "
                "Delete it globally and clear every mapping, observation mapping, "
                "overlay, and Fit inclusion?",
                QtWidgets.QMessageBox.StandardButton.Yes
                | QtWidgets.QMessageBox.StandardButton.Cancel,
                QtWidgets.QMessageBox.StandardButton.Cancel,
            )
            if answer != QtWidgets.QMessageBox.StandardButton.Yes:
                return
        candidate = self._project_session.document.remove_dataset(
            intent.dataset_id
        )
        self._consume_scientific_document_delta(
            self._workspace_tab_registry.replace_document(candidate)
        )
        self._render_workspace()

    @QtCore.Slot(object)
    def _export_original_dataset(
        self,
        intent: DatasetExportOriginalIntent,
    ) -> None:
        if not isinstance(intent, DatasetExportOriginalIntent):
            return
        try:
            record = self._project_session.document.dataset(intent.dataset_id)
        except KeyError:
            return
        path, _selected = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export Original Dataset",
            os.path.join(resolve_start_dir(None), record.original_filename),
            "All Files (*)",
        )
        if not path:
            return
        try:
            exported = export_original_source(record, path)
            self.set_status_text(f"Exported original dataset: {exported}")
        except (DatasetMappingError, OSError) as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "Export Original Dataset",
                str(exc),
            )

    @QtCore.Slot(object)
    def _map_dataset(self, intent: DatasetMappingIntent) -> None:
        if not isinstance(intent, DatasetMappingIntent):
            return
        try:
            choice = DatasetMappingChoice(intent.action)
            description = (
                "Leave Dataset Unmapped"
                if choice is DatasetMappingChoice.LEAVE_UNMAPPED
                else "Map Dataset"
            )
            if not self._accept_visible_source(description=description):
                return
            tab = self._workspace_tab_registry.active_tab
            dataset = self._project_session.document.dataset(intent.dataset_id)
            if choice is DatasetMappingChoice.CREATE_SET:
                species = projected_species_names(tab)
                if not species:
                    raise ValueError(
                        "Apply a valid mechanism before creating a set from observations."
                    )
                mapping = self._prompt_observation_mapping(
                    dataset,
                    tab.dataset_link(dataset.dataset_id).observation_mapping
                    if tab.dataset_link(dataset.dataset_id) is not None
                    else (),
                    species,
                    title="Confirm Observation Mapping",
                )
                if mapping is None:
                    return
                suggested_name = next_set_name(
                    item.name for item in tab.initial_condition_sets
                )
                name, accepted = QtWidgets.QInputDialog.getText(
                    self,
                    "Create Initial-Condition Set",
                    "Set name:",
                    QtWidgets.QLineEdit.EchoMode.Normal,
                    suggested_name,
                )
                if not accepted:
                    return
                candidate_tab = create_set_from_exact_zero(
                    tab=tab,
                    dataset=dataset,
                    set_id=new_set_id(),
                    set_name=str(name),
                    model_species=species,
                    confirmed_mapping=mapping,
                    mapping_confirmed=True,
                )
            elif choice is DatasetMappingChoice.MAP_EXISTING:
                if not intent.set_id:
                    raise ValueError("Map Existing requires an initial-condition set.")
                occupied = (
                    OccupiedSetChoice(intent.occupied_choice)
                    if intent.occupied_choice
                    else None
                )
                copy_plan = None
                if occupied is OccupiedSetChoice.CREATE_COPY:
                    source_set = tab.initial_condition_set(intent.set_id)
                    copy_plan = SetCopyPlan(
                        source_set_id=source_set.set_id,
                        new_set_id=new_set_id(),
                        new_name=next_set_name(
                            item.name for item in tab.initial_condition_sets
                        ),
                        values=source_set.values,
                    )
                plan = plan_dataset_to_set_mapping(
                    links=tab.dataset_links,
                    dataset_id=intent.dataset_id,
                    target_set_id=intent.set_id,
                    occupied_choice=occupied,
                    copy_plan=copy_plan,
                )
                candidate_tab = apply_dataset_set_mapping_plan(
                    document=self._project_session.document,
                    tab=tab,
                    plan=plan,
                    mechanism_species=projected_species_names(tab),
                )
            else:
                plan = plan_leave_unmapped(
                    links=tab.dataset_links,
                    dataset_id=intent.dataset_id,
                )
                candidate_tab = apply_dataset_set_mapping_plan(
                    document=self._project_session.document,
                    tab=tab,
                    plan=plan,
                    mechanism_species=projected_species_names(tab),
                )
            self._consume_scientific_document_delta(
                self._workspace_tab_registry.replace_active_tab(
                    candidate_tab
                )
            )
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Dataset Mapping", str(exc))
        self._render_workspace()

    @QtCore.Slot(object)
    def _edit_observation_mapping(
        self,
        intent: DatasetObservationMappingIntent,
    ) -> None:
        if not isinstance(intent, DatasetObservationMappingIntent):
            return
        try:
            if not self._accept_visible_source(
                description="Edit Observation Mapping"
            ):
                return
            tab = self._workspace_tab_registry.active_tab
            dataset = self._project_session.document.dataset(intent.dataset_id)
            species = projected_species_names(tab)
            current = tab.dataset_link(dataset.dataset_id)
            mapping = self._prompt_observation_mapping(
                dataset,
                current.observation_mapping if current is not None else (),
                species,
                title="Observation Mapping",
            )
            if mapping is None:
                return
            candidate = with_confirmed_observation_mapping(
                document=self._project_session.document,
                tab=tab,
                dataset=dataset,
                model_species=species,
                confirmed_mapping=mapping,
            )
            self._consume_scientific_document_delta(
                self._workspace_tab_registry.replace_active_tab(
                    candidate
                )
            )
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Observation Mapping", str(exc))
        self._render_workspace()

    def _prompt_observation_mapping(
        self,
        dataset,
        current_mapping: Iterable[tuple[str, str]],
        model_species: Sequence[str],
        *,
        title: str,
    ) -> tuple[tuple[str, str], ...] | None:
        species = tuple(dict.fromkeys(str(name) for name in model_species))
        columns = tuple(series.column_name for series in dataset.observations)
        current = dict(current_mapping)
        suggestions = {
            item.raw_column: item.model_species
            for item in suggest_observation_mapping(
                raw_columns=columns,
                model_species=species,
            )
        }
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle(title)
        layout = QtWidgets.QVBoxLayout(dialog)
        explanation = QtWidgets.QLabel(
            "Review every suggested mapping. Selecting OK explicitly confirms "
            "the choices; no equal-name column is adopted silently.",
            dialog,
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)
        form = QtWidgets.QFormLayout()
        editors: dict[str, QtWidgets.QComboBox] = {}
        for column in columns:
            combo = QtWidgets.QComboBox(dialog)
            combo.addItem("(unmapped)", None)
            for species_name in species:
                combo.addItem(species_name, species_name)
            selected = current.get(column, suggestions.get(column))
            if selected in species:
                combo.setCurrentIndex(combo.findData(selected))
            form.addRow(column, combo)
            editors[column] = combo
        layout.addLayout(form)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            parent=dialog,
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return None
        mapping = tuple(
            (column, str(combo.currentData()))
            for column, combo in editors.items()
            if combo.currentData() is not None
        )
        if len({target for _column, target in mapping}) != len(mapping):
            QtWidgets.QMessageBox.warning(
                self,
                title,
                "Each mechanism species may be mapped from at most one column.",
            )
            return None
        return mapping

    @QtCore.Slot(object)
    def _set_dataset_fit_inclusion(
        self,
        intent: DatasetFitInclusionIntent,
    ) -> None:
        if not isinstance(intent, DatasetFitInclusionIntent):
            return
        try:
            if not self._accept_visible_source(
                description="Change Fit Dataset Inclusion"
            ):
                return
            candidate = with_validated_fit_inclusion(
                document=self._project_session.document,
                tab=self._workspace_tab_registry.active_tab,
                dataset_id=intent.dataset_id,
                include=intent.include,
                mechanism_species=projected_species_names(
                    self._workspace_tab_registry.active_tab
                ),
            )
            self._consume_scientific_document_delta(
                self._workspace_tab_registry.replace_active_tab(
                    candidate
                )
            )
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Fit Dataset", str(exc))
        self._render_workspace()

    @QtCore.Slot(object)
    def _set_overlay_selection(self, intent: OverlaySelectionIntent) -> None:
        if not isinstance(intent, OverlaySelectionIntent):
            return
        if not self._accept_visible_source(
            description="Change Dataset Overlay Plot Membership"
        ):
            return
        state = self._workspace_tab_registry.active_presentation
        overlays = dict(state.overlay_columns_by_dataset)
        if intent.columns:
            overlays[intent.dataset_id] = tuple(intent.columns)
        else:
            overlays.pop(intent.dataset_id, None)
        self._workspace_tab_registry.set_presentation(
            self._workspace_tab_registry.active_tab_id,
            replace(
                state,
                overlay_columns_by_dataset=tuple(overlays.items()),
            ),
        )
        self._render_workspace()

    @QtCore.Slot(bool)
    def _set_reference_lines_visible(self, visible: bool) -> None:
        state = self._workspace_tab_registry.active_presentation
        self._workspace_tab_registry.set_presentation(
            self._workspace_tab_registry.active_tab_id,
            replace(state, reference_lines_visible=bool(visible)),
        )
        self._refresh_project_state()
        self._render_results()

    @QtCore.Slot(bool)
    def _set_intervention_annotations_visible(self, visible: bool) -> None:
        state = self._workspace_tab_registry.active_presentation
        self._workspace_tab_registry.set_presentation(
            self._workspace_tab_registry.active_tab_id,
            replace(
                state,
                intervention_annotations_visible=bool(visible),
            ),
        )
        self._refresh_project_state()
        self._render_results()

    @QtCore.Slot(str)
    def _set_x_series(self, name: str) -> None:
        state = self._workspace_tab_registry.active_presentation
        self._workspace_tab_registry.set_presentation(
            self._workspace_tab_registry.active_tab_id,
            replace(state, selected_x_series=str(name or "t")),
        )
        self._refresh_project_state()
        self._render_results()

    @QtCore.Slot(object)
    def _set_y_series(self, names: object) -> None:
        selected = tuple(str(name) for name in (names or ()) if str(name))
        state = self._workspace_tab_registry.active_presentation
        self._workspace_tab_registry.set_presentation(
            self._workspace_tab_registry.active_tab_id,
            replace(
                state,
                selected_y_series=selected,
                selected_y_uses_default_all=False,
            ),
        )
        self._refresh_project_state()
        self._render_results()

    @QtCore.Slot(bool)
    def _set_parametric_mode(self, enabled: bool) -> None:
        state = self._workspace_tab_registry.active_presentation
        self._workspace_tab_registry.set_presentation(
            self._workspace_tab_registry.active_tab_id,
            replace(state, parametric_mode=bool(enabled)),
        )
        self._refresh_project_state()
        self._render_results()

    @QtCore.Slot(object)
    def _set_plot_presentation(self, intent: PlotPresentationIntent) -> None:
        if not isinstance(intent, PlotPresentationIntent):
            return
        state = self._workspace_tab_registry.active_presentation
        self._workspace_tab_registry.set_presentation(
            self._workspace_tab_registry.active_tab_id,
            replace(state, **{intent.option: intent.enabled}),
        )
        self._refresh_project_state()
        self._render_results()

    # ------------------------------------------------------------------
    # Menus and preserved shell actions
    # ------------------------------------------------------------------
    def _create_menus(self) -> None:
        bar = self.menuBar()
        file_menu = bar.addMenu("&File")
        self._file_menu = file_menu
        self._add_action(
            file_menu,
            "New Project",
            self.project_controller.new_project,
            QtGui.QKeySequence.StandardKey.New,
            "newProjectAction",
        )
        self._add_action(
            file_menu,
            "Open Project...",
            self.project_controller.load_project,
            QtGui.QKeySequence.StandardKey.Open,
            "loadProjectAction",
        )
        self._recent_menu = file_menu.addMenu("Recent Projects")
        self.config_controller.populate_recent_menu(self._recent_menu)
        file_menu.addSeparator()
        self._add_action(
            file_menu,
            "Save Project",
            self.project_controller.save_project,
            QtGui.QKeySequence.StandardKey.Save,
            "saveProjectAction",
        )
        self._add_action(
            file_menu,
            "Save Project As...",
            self.project_controller.save_project_as,
            QtGui.QKeySequence.StandardKey.SaveAs,
            "saveProjectAsAction",
        )
        self._add_action(
            file_menu,
            "Load Data...",
            self._dataset_import_panel.trigger_load_dialog,
            "Ctrl+Shift+L",
            "loadDataAction",
        )
        self._xlsx_export_action = self._add_action(
            file_menu,
            "Export Results XLSX...",
            self._export_xlsx,
            "Ctrl+E",
            "exportResultsXlsxAction",
        )
        file_menu.addSeparator()
        self._add_action(
            file_menu,
            "Exit",
            self.close,
            QtGui.QKeySequence.StandardKey.Quit,
            "exitAction",
        )

        edit_menu = bar.addMenu("&Edit")
        self._edit_menu = edit_menu
        self._undo_action = self._add_action(
            edit_menu,
            "Undo",
            self._mechanism_editor.undo_focused_text,
            QtGui.QKeySequence.StandardKey.Undo,
            "undoAction",
        )
        self._redo_action = self._add_action(
            edit_menu,
            "Redo",
            self._mechanism_editor.redo_focused_text,
            QtGui.QKeySequence.StandardKey.Redo,
            "redoAction",
        )
        view_menu = bar.addMenu("&View")
        self._view_menu = view_menu
        panels = view_menu.addMenu("Panels")
        self._panels_menu = panels
        self._dock_toggle_actions: dict[str, QtGui.QAction] = {}
        for spec, dock in zip(
            dock_shell_specs(),
            self._shell_docks(),
            strict=True,
        ):
            action = dock.toggleViewAction()
            action.setText(spec.title)
            action.setObjectName(f"{spec.identity_key}DockToggleAction")
            panels.addAction(action)
            self._dock_toggle_actions[spec.identity_key] = action
        analysis_menu = view_menu.addMenu("Analysis Surfaces")
        for surface in self._workspace.available_analysis_surfaces():
            self._add_action(
                analysis_menu,
                surface,
                lambda _checked=False, name=surface: self._show_analysis_surface(name),
                None,
                f"view{surface}SurfaceAction",
            )
        view_menu.addSeparator()
        self._add_action(
            view_menu,
            "Reset Layout",
            self.reset_layout,
            "Ctrl+Shift+R",
            "resetLayoutAction",
        )
        self._dark_mode_action = self._add_action(
            view_menu,
            "Dark Mode",
            self.config_controller.toggle_theme,
            "Ctrl+Shift+D",
            "darkModeAction",
            checkable=True,
        )

        preferences = bar.addMenu("&Preferences")
        self._preferences_menu = preferences
        self._add_action(
            preferences,
            "Run Parallel Maximum...",
            self._configure_run_parallel_maximum,
            None,
            "runParallelMaximumPreferenceAction",
        )
        self._add_action(
            preferences,
            "Parameter Preview Debounce...",
            self._configure_parameter_debounce,
            None,
            "parameterDebouncePreferenceAction",
        )
        self._add_action(
            preferences,
            "Equilibrium Preview Debounce...",
            self._configure_equilibrium_debounce,
            None,
            "equilibriumDebouncePreferenceAction",
        )

        examples = bar.addMenu("&Examples")
        self._examples_menu = examples
        presets = examples.addMenu("Preset Mechanisms")
        self._preset_examples_menu = presets
        for index in range(1, 10):
            preset_id = f"M{index}"
            action = presets.addAction(preset_id)
            action.setData(preset_id)
            action.setObjectName(f"loadPreset{preset_id}Action")
            action.triggered.connect(
                lambda _checked=False, value=preset_id: self._load_preset_mechanism(
                    value
                )
            )
        interventions = examples.addMenu("Intervention Examples")
        self._intervention_examples_menu = interventions
        try:
            intervention_specs = get_all_intervention_example_specs()
        except InterventionExampleCatalogError as exc:
            unavailable = interventions.addAction(str(exc))
            unavailable.setObjectName("interventionExampleCatalogUnavailableAction")
            unavailable.setEnabled(False)
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
            "Template Manager...",
            self._open_template_manager,
            "Ctrl+T",
            "templateManagerAction",
        )

        simulation = bar.addMenu("&Simulation")
        self._simulation_menu = simulation
        self._run_action = self._add_action(
            simulation,
            "Run",
            self._start_selected_run,
            "Ctrl+R",
            "runSimulationAction",
        )
        self._run_action.setShortcuts(["Ctrl+R", "F5"])
        self._stop_action = self._add_action(
            simulation,
            "Stop",
            self._simulation_jobs.cancel_current,
            "Esc",
            "stopSimulationAction",
        )
        self._add_action(
            simulation,
            "Symbolic Calculator...",
            self._open_symbolic_calculator,
            None,
            "openSymbolicCalculatorAction",
        )
        simulation.addSeparator()
        self._add_action(
            simulation,
            "Tab Scientific Settings...",
            self._open_scientific_settings,
            None,
            "tabScientificSettingsAction",
        )

        fitting = bar.addMenu("&Fitting")
        self._fitting_menu = fitting
        self._add_action(
            fitting,
            "Fitting Defaults...",
            self._configure_fitting,
            None,
            "fittingDefaultsAction",
        )
        self._global_fit_action = self._add_action(
            fitting,
            "Global Fit...",
            self._run_global_fit,
            "Ctrl+Shift+F",
            "globalFitAction",
        )

        help_menu = bar.addMenu("&Help")
        self._help_menu = help_menu
        self._add_action(
            help_menu,
            "Documentation",
            self._open_docs,
            QtGui.QKeySequence.StandardKey.HelpContents,
            "documentationAction",
        )
        self._add_action(
            help_menu,
            "Panel Layout Tips...",
            self._show_panel_layout_tips,
            None,
            "panelLayoutTipsAction",
        )
        help_menu.addSeparator()
        self._add_action(
            help_menu,
            "About",
            self._show_about,
            None,
            "aboutAction",
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
    ) -> QtGui.QAction:
        action = menu.addAction(text)
        action.setObjectName(object_name)
        action.setCheckable(checkable)
        if shortcut is not None:
            action.setShortcut(shortcut)
        action.triggered.connect(callback)
        return action

    def _configure_run_parallel_maximum(self) -> None:
        current = resolve_run_parallel_capacity(
            cpu_count=os.cpu_count(),
            user_max=self.config_controller.run_user_maximum_preference(),
            target_count=60,
        ).resolved_user_max
        value, accepted = QtWidgets.QInputDialog.getInt(
            self,
            "Run Parallel Maximum",
            "Maximum concurrent Run targets:",
            current,
            1,
            60,
            1,
        )
        if not accepted:
            return
        saved = self.config_controller.set_run_user_maximum(int(value))
        self.set_status_text(f"Run parallel maximum set to {saved}.")

    def _configure_parameter_debounce(self) -> None:
        value, accepted = QtWidgets.QInputDialog.getInt(
            self,
            "Parameter Preview Debounce",
            "Debounce interval (milliseconds):",
            self.config_controller.mechanism_parameter_debounce_ms(),
            0,
            1000,
            10,
        )
        if not accepted:
            return
        saved = self.config_controller.set_mechanism_parameter_debounce_ms(
            int(value)
        )
        self.set_status_text(f"Parameter preview debounce set to {saved} ms.")

    def _configure_equilibrium_debounce(self) -> None:
        value, accepted = QtWidgets.QInputDialog.getInt(
            self,
            "Equilibrium Preview Debounce",
            "Debounce interval (milliseconds):",
            self.config_controller.equilibrium_k_debounce_ms(),
            0,
            1000,
            10,
        )
        if not accepted:
            return
        saved = self.config_controller.set_equilibrium_k_debounce_ms(int(value))
        self.set_status_text(f"Equilibrium preview debounce set to {saved} ms.")

    def _open_scientific_settings(self) -> None:
        if not self._settle_execution_candidate():
            return
        dialog = TabScientificSettingsDialog(self)
        dialog.set_scientific_settings(
            self._workspace_tab_registry.active_tab.scientific_settings
        )
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        settings = dialog.scientific_settings()
        if self._initial_conditions_actions.set_tab_scientific_settings(settings):
            self.config_controller.persist_scientific_defaults(settings)
        self._render_workspace()

    def _run_global_fit(self) -> bool:
        tab = self._workspace_tab_registry.active_tab
        fit_set_ids = tuple(
            link.set_id
            for _dataset_id, link in tab.dataset_links
            if link.include_in_fit and link.set_id is not None
        )
        if self._initial_conditions_actions.has_invalid_cells(fit_set_ids):
            QtWidgets.QMessageBox.warning(
                self,
                "Global Fit",
                "Fit is blocked by invalid visible Initial Conditions in an included set.",
            )
            return False
        if not self._computation_gate.is_idle:
            self.set_status_text(
                "Global Fit rejected: another computation is active."
            )
            return False
        if not self._confirm_discard_slider_preview("Global Fit"):
            return False
        if not self._settle_execution_candidate():
            return False
        if not self._accept_visible_source(description="Global Fit"):
            return False
        tab = self._workspace_tab_registry.active_tab
        fit_set_ids = tuple(
            link.set_id
            for _dataset_id, link in tab.dataset_links
            if link.include_in_fit and link.set_id is not None
        )
        if self._initial_conditions_actions.has_invalid_cells(fit_set_ids):
            QtWidgets.QMessageBox.warning(
                self,
                "Global Fit",
                "Fit is blocked by invalid visible Initial Conditions in an included set.",
            )
            return False
        if not self._computation_gate.is_idle:
            self.set_status_text(
                "Global Fit rejected: another computation is active."
            )
            return False
        self._slider_binding.cancel_pending_preview_intent(tab.tab_id)
        return self._fit_session_owner.open_or_focus(tab.tab_id) is not None

    def _configure_fitting(self) -> bool:
        defaults = self._fit_session_owner.project_defaults
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Fitting Defaults")
        form = QtWidgets.QFormLayout(dialog)
        scope = QtWidgets.QComboBox(dialog)
        scope.addItem("Current project", "project")
        scope.addItem("New projects (application default)", "application")
        method = QtWidgets.QComboBox(dialog)
        method.addItems(list(FITTING_METHODS))
        method.setCurrentText(defaults.method)
        max_nfev = QtWidgets.QSpinBox(dialog)
        max_nfev.setRange(*FITTING_MAX_NFEV_RANGE)
        max_nfev.setValue(defaults.max_nfev)
        ftol = QtWidgets.QDoubleSpinBox(dialog)
        xtol = QtWidgets.QDoubleSpinBox(dialog)
        for editor, value in ((ftol, defaults.ftol), (xtol, defaults.xtol)):
            editor.setDecimals(15)
            editor.setRange(1e-15, 1.0)
            editor.setValue(value)
        use_seed = QtWidgets.QCheckBox("Use deterministic seed", dialog)
        use_seed.setChecked(defaults.use_seed)
        seed = QtWidgets.QSpinBox(dialog)
        seed.setRange(*FITTING_SEED_RANGE)
        seed.setValue(defaults.seed)

        def load_selected_defaults(_index: int) -> None:
            selected = (
                self._fit_session_owner.project_defaults
                if scope.currentData() == "project"
                else self.config_controller.application_fit_defaults()
            )
            method.setCurrentText(selected.method)
            max_nfev.setValue(selected.max_nfev)
            ftol.setValue(selected.ftol)
            xtol.setValue(selected.xtol)
            use_seed.setChecked(selected.use_seed)
            seed.setValue(selected.seed)

        scope.currentIndexChanged.connect(load_selected_defaults)
        form.addRow("Save to", scope)
        form.addRow("Method", method)
        form.addRow("Maximum evaluations", max_nfev)
        form.addRow("Function tolerance", ftol)
        form.addRow("Step tolerance", xtol)
        form.addRow(use_seed)
        form.addRow("Seed", seed)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            parent=dialog,
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return False
        candidate = FitDefaults(
            method=method.currentText().strip().lower(),
            max_nfev=int(max_nfev.value()),
            ftol=float(ftol.value()),
            xtol=float(xtol.value()),
            use_seed=bool(use_seed.isChecked()),
            seed=int(seed.value()),
        )
        if scope.currentData() == "project":
            changed = self._fit_session_owner.set_project_defaults(candidate)
            if changed:
                self._refresh_project_state()
            return changed
        else:
            self.config_controller.persist_fit_defaults(candidate)
            self.set_status_text(
                "Application Fit defaults updated for newly created projects."
            )
        return True

    def _export_xlsx(self) -> None:
        projection = self._active_current_export_projection()
        if projection is None:
            self.set_status_text("No result data is available to export.")
            return
        path, _selected = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export Results XLSX",
            resolve_start_dir(None),
            "Excel Workbook (*.xlsx)",
        )
        if not path:
            return
        if not path.casefold().endswith(".xlsx"):
            path += ".xlsx"
        if self._xlsx_export_owner.start(projection, path):
            self.set_status_text(f"Exporting XLSX: {path}")
            self._refresh_actions()

    @QtCore.Slot(object)
    def _on_xlsx_export_succeeded(self, outcome: object) -> None:
        if not isinstance(outcome, _XlsxExportSuccess):
            raise TypeError("XLSX success must be _XlsxExportSuccess.")
        completion = outcome.completion
        self.set_status_text(
            f"Exported XLSX: {outcome.path} "
            f"({completion.worksheet_count} worksheets, "
            f"{completion.data_row_count} data rows, "
            f"{completion.bytes_written} bytes)."
        )
        self._refresh_actions()

    @QtCore.Slot(object)
    def _on_xlsx_export_failed(self, outcome: object) -> None:
        if not isinstance(outcome, _XlsxExportFailure):
            raise TypeError("XLSX failure must be _XlsxExportFailure.")
        if outcome.cancelled:
            self.set_status_text(outcome.message)
        else:
            QtWidgets.QMessageBox.critical(
                self,
                "Export Results XLSX",
                outcome.message,
            )
        self._refresh_actions()

    # ------------------------------------------------------------------
    # Symbolic, inspector, and auxiliary UI
    # ------------------------------------------------------------------
    def _refresh_symbolic_calculator(self) -> None:
        if self._workspace_tab_registry.active_runtime.reactions_draft is not None:
            state = self._symbolic_calculator_owner.reset_project_session_state()
        else:
            state = self._symbolic_calculator_owner.refresh()
        self._symbolic_calculator_panel.render_state(state)
        self._mechanism_editor.symbolic_calculator_btn.setEnabled(state.available)

    def _open_symbolic_calculator(self) -> None:
        if not self._accept_visible_source(
            description="Open Symbolic Calculator"
        ):
            return
        self._refresh_symbolic_calculator()
        self._symbolic_calculator_dock.show()
        self._symbolic_calculator_dock.raise_()

    def _evaluate_symbolic(self, query: str):
        if not self._accept_visible_source(
            description="Evaluate Symbolic Expression"
        ):
            return self._symbolic_calculator_owner.reset_project_session_state()
        return self._symbolic_calculator_owner.evaluate(query)

    def _open_mechanism_inspector(self) -> None:
        from kindred.gui.widgets.mechanism_inspector import MechanismInspectorDialog

        if not self._accept_visible_source(
            description="Open Mechanism Inspector"
        ):
            return
        try:
            MechanismInspectorDialog(
                self._workspace_tab_registry.active_tab.prepared_dsl,
                self,
            ).exec()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Mechanism Inspector", str(exc))

    def _open_docs(self) -> None:
        if DOCUMENTATION_URL:
            webbrowser.open(DOCUMENTATION_URL)
            return
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
            "side. Use View → Panels to restore hidden panels and Reset Layout "
            "to restore the default arrangement.",
        )

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
        layout.addWidget(button_box)
        return dialog

    def _show_about(self) -> None:
        self._build_about_dialog().exec()

    # ------------------------------------------------------------------
    # Dock/layout/config helpers
    # ------------------------------------------------------------------
    def _startup_size(self) -> QtCore.QSize:
        screen = self.screen()
        if screen is None:
            return QtCore.QSize(1440, 900)
        available = screen.availableGeometry()
        return QtCore.QSize(
            max(960, int(available.width() * 0.86)),
            max(700, int(available.height() * 0.88)),
        )

    def _shell_docks(self) -> tuple[QtWidgets.QDockWidget, ...]:
        return (
            self._mechanism_dock,
            self._sliders_dock,
            self._batch_dock,
            self._right_dock,
            self._analysis_dock,
            self._symbolic_calculator_dock,
        )

    def _apply_default_dock_arrangement(self) -> None:
        for dock in self._shell_docks():
            dock.setFloating(False)
            self.addDockWidget(dock_default_area(dock), dock)
            dock.show()
        self.splitDockWidget(
            self._mechanism_dock,
            self._sliders_dock,
            Qt.Orientation.Vertical,
        )
        self.splitDockWidget(
            self._batch_dock,
            self._right_dock,
            Qt.Orientation.Vertical,
        )
        self.tabifyDockWidget(self._right_dock, self._analysis_dock)
        self.tabifyDockWidget(
            self._right_dock,
            self._symbolic_calculator_dock,
        )
        self._right_dock.raise_()

    def reset_layout(self) -> None:
        self._apply_default_dock_arrangement()

    def redock_shell_dock(self, dock: QtWidgets.QDockWidget) -> None:
        dock.setFloating(False)
        self.addDockWidget(dock_default_area(dock), dock)
        dock.show()

    def schedule_restored_floating_dock_recovery(self) -> None:
        QtCore.QTimer.singleShot(0, self._recover_restored_floating_docks)

    def _recover_restored_floating_docks(self) -> None:
        screen = self.screen()
        available = screen.availableGeometry() if screen is not None else QtCore.QRect()
        for dock in self._shell_docks():
            if not dock.isFloating():
                continue
            geometry = dock.frameGeometry()
            if available.isValid() and available.intersects(geometry):
                continue
            self.redock_shell_dock(dock)

    def restore_persisted_maximized_state(self) -> None:
        QtCore.QTimer.singleShot(0, self.showMaximized)

    def _show_analysis_surface(self, name: str) -> bool:
        if not self._workspace.focus_analysis_surface(name):
            return False
        self._analysis_dock.show()
        self._analysis_dock.raise_()
        return True

    # ------------------------------------------------------------------
    # Project-controller physical port and native Close
    # ------------------------------------------------------------------
    def set_status_text(self, text: str) -> None:
        self._status_label.setText(str(text))

    def _on_document_replaced(self) -> None:
        self._workspace.reset_plot_session_state()
        self._slider_binding.reset_project_session_state()
        self._symbolic_calculator_owner.reset_project_session_state()
        self._current_result_projection = None
        self._render_workspace()

    def _refresh_project_state(self) -> None:
        self._render_tab_state()
        self._refresh_title()
        self._refresh_actions()
        self.config_controller.update_recent_files_menu()

    @QtCore.Slot(bool)
    def _on_close_pending_changed(self, pending: bool) -> None:
        if pending:
            self.set_status_text("Closing Kindred; waiting for active work to stop.")

    def _refresh_title(self) -> None:
        path = self._project_session.current_path
        name = os.path.basename(path) if path else "Untitled"
        dirty = bool(
            self._project_session.is_dirty(
                self._workspace_tab_registry.workspace_presentation()
            )
            or self._workspace_tab_registry.has_dirty_drafts()
        )
        self.setWindowTitle(f"Kindred — {name}{' *' if dirty else ''}")

    def _refresh_actions(self) -> None:
        self._undo_action.setEnabled(
            self._mechanism_editor.native_undo_available()
        )
        self._redo_action.setEnabled(
            self._mechanism_editor.native_redo_available()
        )
        self._set_shared_run_enabled(self._run_admitted())
        self._stop_action.setEnabled(self._simulation_jobs.is_executing)
        self._global_fit_action.setEnabled(self._computation_gate.is_idle)
        if self._xlsx_export_action is not None:
            self._xlsx_export_action.setEnabled(
                self._active_current_export_projection() is not None
                and not self._xlsx_export_owner.is_exporting
            )

    def _apply_slider_debounce_settings(self) -> None:
        self._slider_binding.set_preview_debounce_intervals(
            parameter_ms=(
                self.config_controller.mechanism_parameter_debounce_ms()
            ),
            equilibrium_K_ms=self.config_controller.equilibrium_k_debounce_ms(),
        )

    def _invalidate_after_result_render_failure(
        self,
        tab_id: str,
        message: str,
    ) -> None:
        target = str(tab_id)
        self._invalidate_result_projection(target)
        logger.error("Result display failed for %s: %s", target, message)

    def authorize_and_schedule_close(self) -> None:
        self._native_close_authorized = True
        QtCore.QTimer.singleShot(0, self.close)

    def _hard_exit(self) -> None:
        self.config_controller.save_settings()
        os._exit(0)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if not self._native_close_authorized:
            event.ignore()
            self.project_controller.request_close()
            return
        self.config_controller.save_settings()
        event.accept()

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
