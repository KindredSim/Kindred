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

from kindred._version import __version__ as KINDRED_VERSION
from kindred.core.dataset_mapping import (
    DatasetMappingError,
    DatasetMappingChoice,
    OccupiedSetChoice,
    SetCopyPlan,
    allocate_initial_condition_set_name,
    apply_dataset_mapping_batch,
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
    FIT_MAX_NFEV_RANGE,
    FIT_SEED_RANGE,
    FitDefaults,
    MechanismTabDocument,
    MechanismSourceCollision,
    MechanismSourceFailure,
    MechanismSourceReconciliation,
    ProjectDocument,
    SimulationExecutionSettings,
    canonical_species_names,
    canonical_initial_values,
    SUPPORTED_FIT_METHODS,
    new_set_id,
    reconcile_tab_mechanism_source,
    retained_tab_mechanism_source,
    validate_fit_defaults,
)
from kindred.core.project_session import ProjectContentToken, ProjectSession
from kindred.core.project_presentation_state import (
    AxisSeriesSelection,
    ExperimentalAxisSelection,
    SimulationSeriesAxisSelection,
)
from kindred.core.result_repository import (
    ExplicitResultCurrentFacts,
    PreviewResultKey,
    ResultRepository,
)
from kindred.core.simulator.dsl_parse import (
    PreparedDslRequestSet,
    prepare_dsl_request_set,
)
from kindred.core.units import UnitsModel
from kindred.gui.app_wiring import (
    ANALYSIS_DOCK_SPEC,
    BATCH_DOCK_SPEC,
    MECHANISM_DOCK_SPEC,
    RIGHT_DOCK_SPEC,
    SLIDERS_DOCK_SPEC,
    SYMBOLIC_CALCULATOR_DOCK_SPEC,
    build_module_dock,
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
    DatasetImportCompletion,
    DatasetImportStaging,
    build_raw_record_batch_proposal,
)
from kindred.gui.dataset_payload_snapshot import (
    build_data_manager_snapshot,
    build_experimental_layer_inputs,
)
from kindred.gui.fit_session_owner import FitSessionOwner
from kindred.gui.fitting.constants import (
    DE_POPULATION_TOLERANCE_LABEL,
    DIFFERENTIAL_EVOLUTION_METHOD,
    FUNCTION_TOLERANCE_LABEL,
)
from kindred.gui.main_window_variable_runtime import (
    ActiveSliderRenderProjection,
    TabSliderBinding,
)
from kindred.gui.plot_config import PlotPresentationIntent
from kindred.gui.symbolic_calculator_owner import SymbolicCalculatorOwner
from kindred.gui.theme_manager import ThemeManager
from kindred.gui.widgets.batch_initial_conditions_table import (
    BatchInitialConditionsTableModel,
    BatchInitialConditionsTableView,
    InitialConditionsTableActions,
)
from kindred.gui.widgets.data_manager import (
    DataManagerPanel,
    DatasetDeleteIntent,
    DatasetFitInclusionIntent,
    DatasetMappingIntent,
    DatasetObservationMappingIntent,
    DatasetPlotOverlayIntent,
    DatasetRenameIntent,
    DatasetExportOriginalIntent,
)
from kindred.gui.widgets.dataset_mapping_batch_dialog import DatasetMappingBatchDialog
from kindred.gui.widgets.dataset_overlay_panel import (
    DatasetOverlayPanel,
    OverlaySelectionIntent,
)
from kindred.gui.widgets.editor_panel import EditorPanel
from kindred.gui.widgets.mechanism_editor import (
    MechanismEditorBinding,
    MechanismEditorTabbed,
)
from kindred.gui.widgets.mechanism_workspace import MechanismWorkspaceView
from kindred.gui.widgets.pyqtgraph_plot_panel_impl import PyQtGraphPlotPanel
from kindred.gui.widgets.right_panel import RightPanelTabbed
from kindred.gui.widgets.simulation_panel import SimulationPanel
from kindred.gui.widgets.solver_settings import TabScientificSettingsDialog
from kindred.gui.widgets.symbolic_calculator_panel import SymbolicCalculatorPanel
from kindred.gui.ui_helpers import setup_scientific_validator
from kindred.gui.workspace_result_presenter import (
    VisibleResultSet,
    WorkspaceResultPresenter,
    WorkspaceResultProjection,
    XlsxExportCompletion,
)
from kindred.gui.workflow_contracts import (
    DetailedBalanceCorrectionCommitResult,
    DetailedBalanceCorrectionOutcome,
    DetailedBalanceCorrectionProposal,
)
from kindred.gui.workspace_tab_registry import (
    CanonicalMutationResult,
    WorkspaceTabRegistry,
)
from kindred.io.kin_archive_store import KinArchiveStore
from kindred.io.paths import resolve_export_start_dir
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
        self._terminal_outcome: str | None = None

    def cancel(self) -> bool:
        """Request cancellation unless this worker already committed."""
        with self._finalize_lock:
            if self._terminal_outcome is not None:
                return False
            self.requestInterruption()
            return True

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
            self._check_cancelled()
            WorkspaceResultPresenter.validate_xlsx_export(self._projection)
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
                if self.isInterruptionRequested():
                    self._terminal_outcome = "cancelled"
                    raise _XlsxExportCancelled()
                os.replace(temporary_path, self._path)
                temporary_path = ""
                self._terminal_outcome = "committed"
        except _XlsxExportCancelled:
            with self._finalize_lock:
                self._terminal_outcome = "cancelled"
            self.failed.emit(
                _XlsxExportFailure(
                    generation=self._generation,
                    path=self._path,
                    message="XLSX export was cancelled.",
                    cancelled=True,
                )
            )
        except Exception as exc:
            with self._finalize_lock:
                cancelled = (
                    self._terminal_outcome is None
                    and self.isInterruptionRequested()
                )
                if cancelled:
                    self._terminal_outcome = "cancelled"
            self.failed.emit(
                _XlsxExportFailure(
                    generation=self._generation,
                    path=self._path,
                    message=(
                        "XLSX export was cancelled."
                        if cancelled
                        else f"{type(exc).__name__}: {exc}"
                    ),
                    cancelled=cancelled,
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
        try:
            worker.succeeded.connect(self._publish_success)
            worker.failed.connect(self._publish_failure)
            worker.finished.connect(self._worker_finished)
            worker.start()
        except Exception as exc:
            try:
                verified_unstarted = bool(
                    not worker.isRunning()
                    and not worker.isFinished()
                )
            except RuntimeError:
                verified_unstarted = False
            if not verified_unstarted:
                logger.exception(
                    "XLSX worker startup raised and unstarted state could not "
                    "be verified"
                )
                self.failed.emit(
                    _XlsxExportFailure(
                        generation=generation,
                        path=absolute_path,
                        message=f"Could not confirm XLSX export startup: {exc}",
                        cancelled=False,
                    )
                )
                self.owned_qthreads_changed.emit()
                return False
            self._active_worker = None
            for signal, slot in (
                (worker.succeeded, self._publish_success),
                (worker.failed, self._publish_failure),
                (worker.finished, self._worker_finished),
            ):
                try:
                    signal.disconnect(slot)
                except (RuntimeError, TypeError):
                    pass
            worker.deleteLater()
            logger.exception("XLSX worker startup failed before execution")
            self.failed.emit(
                _XlsxExportFailure(
                    generation=generation,
                    path=absolute_path,
                    message=f"Could not start XLSX export: {exc}",
                    cancelled=False,
                )
            )
            self.owned_qthreads_changed.emit()
            return False
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
        worker = self._active_worker
        if worker is not None and worker.cancel():
            self._generation += 1
        return worker

    def cancel_and_wait_for_project_replacement(
        self,
        deadline: float,
    ) -> bool:
        generation_before_cancel = self._generation
        worker = self._cancel_active()
        if worker is None:
            return True
        if worker.isRunning():
            remaining_ms = max(
                0,
                int((float(deadline) - time.monotonic()) * 1000.0),
            )
            if not worker.wait(remaining_ms):
                return False
        if worker is self._active_worker:
            # The native worker may already have committed and emitted queued
            # success/finished callbacks that cannot settle while project
            # replacement synchronously waits on the GUI thread. Retire that
            # exact authority before the replacement turn can continue.
            if self._generation == generation_before_cancel:
                self._generation += 1
            self._active_worker = None
            worker.deleteLater()
            self.owned_qthreads_changed.emit()
        return True

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
        self._close_mutation_fenced = False
        self._close_fenced_action_states: dict[QtGui.QAction, bool] = {}
        self._rendering_workspace = False
        self._recent_menu: QtWidgets.QMenu | None = None
        self._tab_surfaces: dict[str, _TabSurfaceBundle] = {}
        self._dark_mode_action: QtGui.QAction | None = None
        self._run_action: QtGui.QAction | None = None
        self._xlsx_export_action: QtGui.QAction | None = None
        self._last_export_directory: str | None = None
        self._slider_apply_refreshing: set[str] = set()
        self._prepared_save_materialized: tuple[
            ProjectDocument,
            tuple[tuple[str, tuple[str, ...]], ...],
        ] | None = None

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
            current_explicit_result_facts=(
                self._capture_current_explicit_result_facts
            ),
            propose_detailed_balance_source=self._propose_detailed_balance_source,
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
            propose_detailed_balance_source=self._propose_detailed_balance_source,
            commit_detailed_balance_source=self._commit_detailed_balance_source,
            color_resolver=self.config_controller.species_color,
            parent_widget=self,
            parent=self,
        )

        self._result_presenter = WorkspaceResultPresenter()
        self._build_physical_shell()
        for tab in self._project_session.document.tabs:
            self._create_tab_surface(tab.tab_id)
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
            settle_execution_candidate=(
                lambda token: self._settle_execution_candidate(
                    expected_token=token
                )
            ),
            prepare_before_save=self._prepare_before_save,
            commit_saved_document=self._commit_saved_document,
            inspect_transient_state=self._inspect_transient_state,
            discard_authoring=self._discard_authoring,
            discard_slider_drafts=self._discard_all_slider_drafts,
            replacement_timeout_seconds=_REPLACEMENT_TIMEOUT_SECONDS,
        )

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
                f"but Kindred can continue.\n\n{exc}",
            )

        self._create_menus()
        self._connect_application_graph()
        self._apply_default_dock_arrangement()
        self.config_controller.load_settings()
        self._refresh_workspace_document()
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
            actual_dark_mode=lambda: self._theme_manager.applied_dark_mode,
            set_status_text=self.set_status_text,
            recent_menu=lambda: self._recent_menu,
            load_recent_project=lambda path: self.project_controller.load_recent_project(
                path
            ),
        )

    def _new_tab_runtime(self, tab_id: str):
        from kindred.gui.workspace_tab_registry import TabRuntimeState

        return TabRuntimeState(
            result_repository=ResultRepository(
                tab_id,
                cache_capacity=self.config_controller.result_cache_capacity(),
            )
        )

    def _build_physical_shell(self) -> None:
        self._mechanism_stack = QtWidgets.QStackedWidget(self)
        self._mechanism_stack.setObjectName("mechanismSurfaceStack")
        mechanism = build_module_dock(
            self,
            spec=MECHANISM_DOCK_SPEC,
            panel=self._mechanism_stack,
        )
        self._mechanism_dock = mechanism.dock
        self._slider_stack = QtWidgets.QStackedWidget(self)
        self._slider_stack.setObjectName("sliderSurfaceStack")
        sliders = build_module_dock(
            self,
            spec=SLIDERS_DOCK_SPEC,
            panel=self._slider_stack,
        )
        self._sliders_dock = sliders.dock

        self._initial_conditions_stack = QtWidgets.QStackedWidget(self)
        self._initial_conditions_stack.setObjectName(
            "initialConditionsSurfaceStack"
        )
        initials = build_module_dock(
            self,
            spec=BATCH_DOCK_SPEC,
            panel=self._initial_conditions_stack,
        )
        self._batch_dock = initials.dock

        self._data_stack = QtWidgets.QStackedWidget(self)
        self._data_stack.setObjectName("dataSurfaceStack")
        data = build_module_dock(
            self,
            spec=RIGHT_DOCK_SPEC,
            panel=self._data_stack,
        )
        self._right_dock = data.dock

        analysis = build_module_dock(
            self,
            spec=ANALYSIS_DOCK_SPEC,
            panel=self._workspace.main_plot_analysis_widget(),
        )
        self._analysis_dock = analysis.dock

        self._calculator_stack = QtWidgets.QStackedWidget(self)
        self._calculator_stack.setObjectName("symbolicCalculatorSurfaceStack")
        symbolic = build_module_dock(
            self,
            spec=SYMBOLIC_CALCULATOR_DOCK_SPEC,
            panel=self._calculator_stack,
        )
        self._symbolic_calculator_dock = symbolic.dock

    def _create_tab_surface(self, tab_id: str) -> _TabSurfaceBundle:
        """Construct and retain one tab's complete non-plot physical surface."""

        target = self._workspace_tab_registry.tab(tab_id).tab_id
        if target in self._tab_surfaces:
            raise ValueError(f"Tab surface already exists: {target!r}.")

        editor_panel = EditorPanel(parent=self._mechanism_stack)
        editor = editor_panel.editor
        slider_panel = editor.detach_slider_pane_for_dock()
        batch_model = BatchInitialConditionsTableModel(parent=self)
        simulation_panel = SimulationPanel(
            batch_model=batch_model,
            parent=self._initial_conditions_stack,
        )
        batch_model.setParent(simulation_panel)
        batch_table = simulation_panel.batch_table
        data_panel = DataManagerPanel(
            self._dataset_import_staging,
            origin_tab_id=target,
            source_resolver=lambda source_id: self._project_session.document.imported_source(source_id),
            parent=self,
        )
        overlay_panel = DatasetOverlayPanel(
            self,
            color_for_species=self.config_controller.species_color,
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
        )
        calculator_panel = SymbolicCalculatorPanel(
            on_evaluate=lambda query, captured=target: self._evaluate_symbolic(
                captured,
                query,
            ),
            on_copy_result=calculator_owner.copy_compact_text,
            on_copy_context=calculator_owner.copy_context_text,
            parent=self._calculator_stack,
        )
        editor_binding = MechanismEditorBinding(
            tab_id=target,
            editor=editor,
            registry=self._workspace_tab_registry,
            parent=editor_panel,
        )
        initial_actions = InitialConditionsTableActions(
            tab_id=target,
            registry=self._workspace_tab_registry,
            model=batch_model,
            table=batch_table,
            simulation_jobs=self._simulation_jobs,
            parent=simulation_panel,
        )
        slider_binding = TabSliderBinding(
            tab_id=target,
            registry=self._workspace_tab_registry,
            table=batch_table,
            variable_sliders=editor.variable_sliders_widget(),
            species_sliders=editor.species_sliders_widget(),
            slider_workspace_owner=editor,
            simulation_jobs=self._simulation_jobs,
            preview_preflight=(
                lambda targets, captured=target: self._preview_preflight(
                    captured,
                    targets,
                )
            ),
            parameter_preview_debounce_ms=(
                self.config_controller.mechanism_parameter_debounce_ms()
            ),
            equilibrium_preview_debounce_ms=(
                self.config_controller.equilibrium_k_debounce_ms()
            ),
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
        self._mechanism_stack.addWidget(editor_panel)
        self._slider_stack.addWidget(slider_panel)
        self._initial_conditions_stack.addWidget(simulation_panel)
        self._data_stack.addWidget(right_panel)
        self._calculator_stack.addWidget(calculator_panel)
        self._connect_tab_surface(bundle)
        return bundle

    def _connect_tab_surface(self, bundle: _TabSurfaceBundle) -> None:
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
        panel.stopRequested.connect(self._simulation_jobs.cancel_current)
        panel.workspaceRunDefaultsRequested.connect(
            lambda settings, captured=tab_id: self._set_workspace_run_defaults(
                captured,
                settings,
            )
        )
        panel.runEnabledChanged.connect(
            lambda enabled, captured=tab_id: self._set_tab_run_enabled(
                captured,
                enabled,
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

        editor.run_btn.clicked.connect(
            lambda _checked=False, captured=tab_id: self._start_selected_run(
                captured
            )
        )
        editor.sliderApplyRequested.connect(
            lambda captured=tab_id: self._apply_slider_draft(captured)
        )
        editor.sliderResetRequested.connect(sliders.reset)
        sliders.sliderDraftStateChanged.connect(
            self._on_slider_draft_state_changed
        )
        editor.symbolic_calculator_btn.clicked.connect(
            lambda _checked=False, captured=tab_id: self._open_symbolic_calculator(
                captured
            )
        )
        editor.mechanismInspectRequested.connect(
            lambda captured=tab_id: self._open_mechanism_inspector(captured)
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
            lambda _ids, captured=tab_id: self._refresh_slider_target_summary(
                captured
            )
        )
        sliders.previewDetailedBalanceCorrectionRequired.connect(
            editor.show_preview_detailed_balance_block
        )
        sliders.previewDetailedBalanceCommitRequested.connect(
            self._commit_preview_detailed_balance_source
        )
        sliders.previewDetailedBalanceBlockCleared.connect(
            self._clear_preview_detailed_balance_block
        )
        editor.detailedBalanceCorrectionCompleted.connect(
            self._simulation_jobs.complete_detailed_balance_correction
        )
        editor.detailedBalanceCorrectionCompleted.connect(
            sliders.complete_preview_detailed_balance_correction
        )
        editor.detailedBalanceCorrectionCompleted.connect(
            self._fit_session_owner.complete_detailed_balance_correction
        )
        actions.errorRaised.connect(self.set_status_text)
        actions.activeSnapshotChanged.connect(
            lambda captured=tab_id: self._on_initial_snapshot_rendered(captured)
        )
        actions.selectedRunSetIdsChanged.connect(
            lambda _ids, captured=tab_id: self._on_run_selection_changed(captured)
        )
        selection = bundle.batch_table.selectionModel()
        selection.selectionChanged.connect(
            lambda _selected, _deselected, captured=tab_id: (
                self._on_table_focus_changed(captured)
            )
        )
        selection.currentChanged.connect(
            lambda _current, _previous, captured=tab_id: (
                self._on_table_focus_changed(captured)
            )
        )
        data = bundle.data_panel
        data.datasetRenameRequested.connect(self._rename_dataset)
        data.datasetDeleteRequested.connect(self._delete_dataset)
        data.datasetExportOriginalRequested.connect(self._export_original_dataset)
        data.datasetMappingRequested.connect(
            lambda intent, captured=tab_id: self._map_dataset(captured, intent)
        )
        data.datasetObservationMappingRequested.connect(
            lambda intent, captured=tab_id: self._edit_observation_mapping(
                captured,
                intent,
            )
        )
        data.datasetPlotOverlayRequested.connect(
            lambda intent, captured=tab_id: self._show_plot_overlay(
                captured,
                intent,
            )
        )
        data.datasetFitInclusionRequested.connect(
            lambda intent, captured=tab_id: self._set_dataset_fit_inclusion(
                captured,
                intent,
            )
        )
        bundle.overlay_panel.overlaySelectionRequested.connect(
            lambda intent, captured=tab_id: self._set_overlay_selection(
                captured,
                intent,
            )
        )
        bundle.overlay_panel.speciesColorRequested.connect(
            self._set_species_color
        )
        bundle.overlay_panel.experimentalPointSizeRequested.connect(
            self._set_experimental_point_size
        )
        bundle.overlay_panel.experimentalPointOpacityRequested.connect(
            self._set_experimental_point_opacity
        )
        sliders.publish_slider_draft_state()

    def _drop_tab_surface(self, tab_id: str) -> None:
        target = str(tab_id)
        bundle = self._tab_surfaces.pop(target, None)
        if bundle is None:
            return
        bundle.slider_binding.dispose()
        for stack, widget in (
            (self._mechanism_stack, bundle.editor_panel),
            (self._slider_stack, bundle.slider_panel),
            (self._initial_conditions_stack, bundle.simulation_panel),
            (self._data_stack, bundle.right_panel),
            (self._calculator_stack, bundle.calculator_panel),
        ):
            stack.removeWidget(widget)
            widget.setParent(None)
            widget.deleteLater()

    def _set_species_color(
        self,
        species: object,
        color: object,
        known_species: object,
    ) -> None:
        if self._mutation_input_admitted("Change Species Color"):
            self.config_controller.set_species_color(
                str(species),
                color,
                tuple(known_species or ()),
            )

    def _set_experimental_point_size(self, request: object) -> None:
        if self._mutation_input_admitted("Change Experimental Point Size"):
            self.config_controller.set_experimental_point_size(request)

    def _set_experimental_point_opacity(self, request: object) -> None:
        if self._mutation_input_admitted("Change Experimental Point Opacity"):
            self.config_controller.set_experimental_point_opacity(request)

    def _selected_bundle(self) -> _TabSurfaceBundle:
        return self._tab_surfaces[self._workspace_tab_registry.active_tab_id]

    def _connect_application_graph(self) -> None:
        self.project_controller.documentReplaced.connect(self._on_document_replaced)
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

        application = QtWidgets.QApplication.instance()
        if isinstance(application, QtWidgets.QApplication):
            application.focusChanged.connect(
                lambda _old, _new: self._refresh_actions()
            )
        self._simulation_jobs.state_changed.connect(
            self._on_simulation_state_changed
        )
        self._simulation_jobs.run_dispatched.connect(
            self._on_run_dispatched
        )
        self._simulation_jobs.results_changed.connect(self._on_results_changed)
        self._simulation_jobs.preview_endpoints_retired.connect(
            self._retire_preview_endpoint_presentations
        )
        self._simulation_jobs.status_changed.connect(self.set_status_text)
        self._simulation_jobs.detailed_balance_correction_requested.connect(
            self._present_detailed_balance_correction
        )
        self._fit_session_owner.window_changed.connect(
            lambda _open: self._refresh_actions()
        )
        self._fit_session_owner.execution_changed.connect(
            lambda _executing: self._refresh_actions()
        )
        self._fit_session_owner.canonical_applied.connect(
            self._on_fit_canonical_applied
        )
        self._fit_session_owner.status_changed.connect(self.set_status_text)
        self._fit_session_owner.detailed_balance_correction_requested.connect(
            self._present_detailed_balance_correction
        )

        self._dataset_import_staging.completionAvailable.connect(
            self._commit_import_completion
        )
        self._dataset_import_staging.busyChanged.connect(
            self._on_import_busy_changed
        )
        self.config_controller.applicationPlotPreferencesChanged.connect(
            self._refresh_application_plot_preferences
        )
        self._workspace.plotCreated.connect(self._connect_workspace_plot)
        self.config_controller.mechanismParameterDebounceChanged.connect(
            lambda _value: self._apply_slider_debounce_settings()
        )
        self.config_controller.equilibriumKDebounceChanged.connect(
            lambda _value: self._apply_slider_debounce_settings()
        )
        self.config_controller.resultCacheCapacityChanged.connect(
            self._resize_result_repositories
        )

    def _mutation_input_admitted(self, action: str | None = None) -> bool:
        """Return whether user mutations may still enter the live shell."""

        controller = getattr(self, "project_controller", None)
        admitted = bool(
            not self._close_mutation_fenced
            and (controller is None or controller.mutation_admitted)
        )
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

    def _set_close_mutation_fence(self, fenced: bool) -> None:
        """Physically disable writable shell surfaces for armed Close."""

        fenced = bool(fenced)
        if fenced:
            self._close_mutation_fenced = True
        elif not self._mutation_input_admitted():
            # Native/hard-exit authorization is terminal.  Do not briefly
            # reopen the shell between closePendingChanged(False) and
            # closeEvent()/hard exit.
            fenced = True
        else:
            self._close_mutation_fenced = False

        enabled = not self._close_mutation_fenced
        for widget in (
            self._workspace,
            self._mechanism_dock,
            self._sliders_dock,
            self._batch_dock,
            self._right_dock,
            self._analysis_dock,
            self._symbolic_calculator_dock,
        ):
            widget.setEnabled(enabled)

        for menu_name in (
            "_file_menu",
            "_edit_menu",
            "_view_menu",
            "_preferences_menu",
            "_examples_menu",
            "_simulation_menu",
            "_fitting_menu",
        ):
            menu = getattr(self, menu_name, None)
            if not isinstance(menu, QtWidgets.QMenu):
                continue
            menu.setEnabled(enabled)
            for action in self._menu_actions(menu):
                if not enabled:
                    self._close_fenced_action_states.setdefault(
                        action,
                        bool(action.isEnabled()),
                    )
                    action.setEnabled(False)
                elif action in self._close_fenced_action_states:
                    action.setEnabled(
                        self._close_fenced_action_states.pop(action)
                    )

        self._refresh_actions()

    # ------------------------------------------------------------------
    # Direct render
    # ------------------------------------------------------------------
    def _select_tab_surfaces(self, tab_id: str) -> None:
        """Raise retained pages only; tab selection never reconstructs state."""

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

        document = self._project_session.document
        active_tab_id = self._workspace_tab_registry.active_tab_id
        self._workspace.render(document, active_tab_id=active_tab_id)
        self._render_tab_state()
        self._select_tab_surfaces(active_tab_id)

    @QtCore.Slot(str, object)
    def _connect_workspace_plot(
        self,
        tab_id: str,
        plot: PyQtGraphPlotPanel,
    ) -> None:
        """Bind one retained plot only to intents from its owning tab."""

        target = str(tab_id)
        required_signals = (
            "referenceLayerVisibilityRequested",
            "interventionAnnotationsVisibilityRequested",
            "xSeriesRequested",
            "ySeriesRequested",
            "plotPresentationIntentRequested",
            "resultDiagnostic",
            "renderFailed",
        )
        if any(
            not callable(getattr(getattr(plot, name, None), "connect", None))
            for name in required_signals
        ):
            raise TypeError("Workspace plot must expose the plot intent signals.")
        required_preview_methods = (
            "capture_preview_transition_origins",
            "authorize_preview_endpoint_presentation",
            "retire_preview_endpoint_presentations",
        )
        if any(
            not callable(getattr(plot, name, None))
            for name in required_preview_methods
        ):
            raise TypeError(
                "Workspace plot must expose exact Preview transition methods."
            )
        bundle = self._tab_surfaces.get(target)
        if bundle is None:
            raise RuntimeError(
                f"Workspace plot {target!r} has no matching tab runtime bundle."
            )

        plot.referenceLayerVisibilityRequested.connect(
            lambda visible: self._set_reference_lines_visible(target, visible)
        )
        plot.interventionAnnotationsVisibilityRequested.connect(
            lambda visible: self._set_intervention_annotations_visible(
                target,
                visible,
            )
        )
        plot.xSeriesRequested.connect(
            lambda selection: self._set_x_series(target, selection)
        )
        plot.ySeriesRequested.connect(
            lambda selections: self._set_y_series(target, selections)
        )
        plot.plotPresentationIntentRequested.connect(
            lambda intent: self._set_plot_presentation(target, intent)
        )
        plot.resultDiagnostic.connect(self._on_workspace_result_diagnostic)
        plot.renderFailed.connect(
            lambda message: self._on_workspace_plot_render_failed(
                target,
                message,
            )
        )
        bundle.slider_binding.previewTransitionOriginCaptureRequested.connect(
            plot.capture_preview_transition_origins
        )
        bundle.slider_binding.previewEndpointPresentationRequested.connect(
            plot.authorize_preview_endpoint_presentation
        )

    @QtCore.Slot(object)
    def _retire_preview_endpoint_presentations(self, keys: object) -> None:
        """Route exact retired Preview keys only to their owning tab panel."""

        try:
            captured = tuple(keys)  # type: ignore[arg-type]
        except TypeError as exc:
            raise TypeError(
                "retired Preview endpoints must be an iterable of keys."
            ) from exc
        if any(not isinstance(key, PreviewResultKey) for key in captured):
            raise TypeError(
                "retired Preview endpoints must contain PreviewResultKey values."
            )
        grouped: dict[str, list[PreviewResultKey]] = {}
        for key in captured:
            grouped.setdefault(key.tab_id, []).append(key)
        for tab_id, tab_keys in grouped.items():
            try:
                plot = self._workspace.plot_for_tab(tab_id)
            except KeyError:
                continue
            plot.retire_preview_endpoint_presentations(tuple(tab_keys))

    def _refresh_workspace_document(self) -> None:
        """Refresh canonical/global projections without changing selected page state."""

        if self._rendering_workspace:
            return
        self._rendering_workspace = True
        try:
            document = self._project_session.document
            document_ids = {tab.tab_id for tab in document.tabs}
            if set(self._tab_surfaces) != document_ids:
                raise RuntimeError(
                    "Retained tab surfaces do not match the canonical document."
                )
            self._workspace.render(
                document,
                active_tab_id=self._workspace_tab_registry.active_tab_id,
            )
            self._apply_persisted_plot_states(document)
            for tab in document.tabs:
                self._refresh_tab_controls(tab.tab_id)
            self._render_tab_state()
            self._select_tab_surfaces(self._workspace_tab_registry.active_tab_id)
            self._refresh_title()
            self._refresh_actions()
        finally:
            self._rendering_workspace = False

    def _refresh_tab_controls(
        self,
        tab_id: str,
        *,
        scientific_fields: frozenset[str] | None = None,
        preserve_reactions_editor: bool = False,
    ) -> None:
        """Refresh a tab, narrowing scientific-settings invalidation when known."""

        target = str(tab_id)
        bundle = self._tab_surfaces[target]
        if scientific_fields:
            bundle.slider_binding.clear_preview_detailed_balance_block()
            bundle.editor.clear_preview_detailed_balance_block()
        if scientific_fields and scientific_fields <= {
            "rtol",
            "atol",
            "analytic_jacobian_policy",
        }:
            try:
                self._render_results_for_tab(target)
            finally:
                self._refresh_project_state()
            return
        document = self._project_session.document
        tab = self._workspace_tab_registry.tab(target)
        presentation = self._workspace_tab_registry.presentation(target)
        failures: list[str] = []

        def refresh(
            label: str,
            operation: Callable[[], None],
            clear: Callable[[], None] | None = None,
        ) -> None:
            try:
                operation()
            except Exception as exc:
                logger.exception(
                    "Tab %s non-result refresh failed: %s",
                    target,
                    label,
                )
                failures.append(f"{label}: {exc}")
                if clear is not None:
                    try:
                        clear()
                    except Exception as clear_exc:
                        logger.exception(
                            "Tab %s failed-panel invalidation failed: %s",
                            target,
                            label,
                        )
                        failures.append(f"{label} clear: {clear_exc}")

        if not preserve_reactions_editor:
            refresh(
                "mechanism editor",
                lambda: bundle.editor_binding.apply_render_projection(
                    bundle.editor_binding.prepare_render()
                ),
            )
        refresh(
            "initial conditions",
            lambda: bundle.initial_actions.apply_render_projection(
                bundle.initial_actions.prepare_render()
            ),
        )
        refresh(
            "simulation settings",
            lambda: bundle.simulation_panel.render_run_defaults(
                tab.run_defaults,
                run_admitted=self._run_admitted(target),
                simulation_job_kind=self._simulation_jobs.current_kind,
            ),
        )
        slider_projection: ActiveSliderRenderProjection | None = None
        try:
            slider_projection = bundle.slider_binding.prepare_render()
            bundle.slider_binding.apply_render_projection(slider_projection)
            bundle.slider_binding.settle_render_projection(slider_projection)
        except Exception as exc:
            logger.exception("Tab %s Slider refresh failed", target)
            failures.append(f"sliders: {exc}")
            try:
                bundle.slider_binding.clear_projection()
            except Exception as clear_exc:
                logger.exception(
                    "Tab %s Slider invalidation failed",
                    target,
                )
                failures.append(f"sliders clear: {clear_exc}")
        finally:
            bundle.slider_binding.publish_slider_draft_state()
        try:
            data_snapshot = build_data_manager_snapshot(
                document=document,
                active_tab=tab,
                presentation=presentation,
            )
        except Exception as exc:
            logger.exception("Tab %s dataset projection failed", target)
            failures.append(f"datasets: {exc}")
            try:
                bundle.data_panel.clear_projection()
            except Exception as clear_exc:
                logger.exception(
                    "Tab %s Data invalidation failed",
                    target,
                )
                failures.append(f"datasets clear: {clear_exc}")
            try:
                bundle.overlay_panel.clear_projection()
            except Exception as clear_exc:
                logger.exception(
                    "Tab %s Overlay invalidation failed",
                    target,
                )
                failures.append(f"dataset overlays clear: {clear_exc}")
        else:
            refresh(
                "datasets",
                lambda: bundle.data_panel.render_snapshot(data_snapshot),
                bundle.data_panel.clear_projection,
            )
            refresh(
                "dataset overlays",
                lambda: bundle.overlay_panel.render_snapshot(
                    records=data_snapshot.datasets,
                    active_links=data_snapshot.active_links,
                    known_species=tuple(canonical_species_names(tab)),
                ),
                bundle.overlay_panel.clear_projection,
            )

        def refresh_slider_summary() -> None:
            targets = (
                ()
                if slider_projection is None
                else slider_projection.targets
            )
            target_names = tuple(
                tab.initial_condition_set(set_id).name
                for set_id in targets
            )
            bundle.editor.set_slider_edit_targets_summary(
                "Slider edit targets: "
                + (", ".join(target_names) if target_names else "none")
            )

        refresh("slider target summary", refresh_slider_summary)
        refresh(
            "symbolic calculator",
            lambda: self._refresh_symbolic_calculator(target),
        )
        try:
            if failures:
                self.set_status_text(
                    f"Mechanism {tab.name!r} refresh failed: "
                    + "; ".join(failures)
                )
        finally:
            try:
                self._render_results_for_tab(target)
            finally:
                self._refresh_project_state()

    def _refresh_application_plot_preferences(self) -> None:
        point_size = self.config_controller.experimental_point_size()
        point_opacity = self.config_controller.experimental_point_opacity()
        for tab in self._project_session.document.tabs:
            bundle = self._tab_surfaces.get(tab.tab_id)
            if bundle is None:
                continue
            bundle.overlay_panel.refresh_application_style_controls()
            bundle.overlay_panel.refresh_color_swatches(
                known_species=tuple(canonical_species_names(tab)),
            )
            try:
                self._workspace.plot_for_tab(
                    tab.tab_id
                ).apply_application_plot_preferences(
                    color_for_species=self.config_controller.species_color,
                    known_species=tuple(canonical_species_names(tab)),
                    experimental_point_size=point_size,
                    experimental_point_opacity=point_opacity,
                )
            except Exception:
                # The panel has already invalidated itself and emitted the sole
                # renderFailed report route; this caller only stops and refreshes.
                continue
        self._refresh_actions()

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
        failures: list[str] = []
        try:
            for tab_id in targets:
                bundle = self._tab_surfaces.get(tab_id)
                if bundle is None:
                    continue
                try:
                    tab = document.tab(tab_id)
                    presentation = self._workspace_tab_registry.presentation(tab_id)
                except Exception as exc:
                    failures.append(f"{tab_id} source: {exc}")
                    continue
                try:
                    snapshot = build_data_manager_snapshot(
                        document=document,
                        active_tab=tab,
                        presentation=presentation,
                    )
                except Exception as exc:
                    logger.exception("Tab %s dataset projection failed", tab_id)
                    failures.append(f"{tab_id} datasets: {exc}")
                    try:
                        bundle.data_panel.clear_projection()
                    except Exception as clear_exc:
                        logger.exception(
                            "Tab %s Data invalidation failed",
                            tab_id,
                        )
                        failures.append(f"{tab_id} datasets clear: {clear_exc}")
                    try:
                        bundle.overlay_panel.clear_projection()
                    except Exception as clear_exc:
                        logger.exception(
                            "Tab %s Overlay invalidation failed",
                            tab_id,
                        )
                        failures.append(
                            f"{tab_id} dataset overlays clear: {clear_exc}"
                        )
                    self._render_results_for_tab(tab_id)
                    continue
                try:
                    bundle.data_panel.render_snapshot(snapshot)
                except Exception as exc:
                    logger.exception("Tab %s Data refresh failed", tab_id)
                    failures.append(f"{tab_id} datasets: {exc}")
                    try:
                        bundle.data_panel.clear_projection()
                    except Exception as clear_exc:
                        logger.exception(
                            "Tab %s Data invalidation failed",
                            tab_id,
                        )
                        failures.append(f"{tab_id} datasets clear: {clear_exc}")
                try:
                    bundle.overlay_panel.render_snapshot(
                        records=snapshot.datasets,
                        active_links=snapshot.active_links,
                        known_species=tuple(canonical_species_names(tab)),
                    )
                except Exception as exc:
                    logger.exception("Tab %s Overlay refresh failed", tab_id)
                    failures.append(f"{tab_id} overlays: {exc}")
                    try:
                        bundle.overlay_panel.clear_projection()
                    except Exception as clear_exc:
                        logger.exception(
                            "Tab %s Overlay invalidation failed",
                            tab_id,
                        )
                        failures.append(f"{tab_id} overlays clear: {clear_exc}")
                self._render_results_for_tab(tab_id)
        finally:
            self._refresh_project_state()
        if failures:
            self.set_status_text(
                "Dataset view refresh failed: " + "; ".join(failures)
            )

    def _apply_persisted_plot_states(self, document: ProjectDocument) -> None:
        """Apply each tab's persisted plot state to its retained panel."""

        for tab in document.tabs:
            self._apply_persisted_plot_state(tab.tab_id)

    def _apply_persisted_plot_state(self, tab_id: str) -> None:
        state = self._workspace_tab_registry.presentation(tab_id)
        self._workspace.plot_for_tab(tab_id).set_persisted_plot_state(
            tab_id,
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
        experimental_layers = build_experimental_layer_inputs(
            document=self._project_session.document,
            active_tab=tab,
            presentation=presentation,
        )
        repository = self._repository_for_tab(tab_id)
        analysis = tab.dsl_analysis
        modeled_species = tuple(canonical_species_names(tab))
        simulation_series_names = list(
            ()
            if analysis is None
            else (*analysis.species_names, *analysis.observable_names)
        )
        for visible in visible_sets:
            entries = (
                repository.preview_entry(visible.set_id),
                repository.explicit_entry(visible.set_id),
                repository.reference_entry(visible.set_id),
            )
            for entry in entries:
                if entry is not None:
                    simulation_series_names.extend(entry.result.series)
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
            color_name = layer.model_species or layer.raw_column
            color_by_axis_selection[
                ExperimentalAxisSelection(
                    layer.dataset_id,
                    layer.raw_observation_id,
                )
            ] = self.config_controller.species_color(
                color_name,
                modeled_species,
            ).name(QtGui.QColor.NameFormat.HexRgb)
        return self._result_presenter.compose(
            tab_id=tab_id,
            current_set_id=presentation.current_set_id,
            visible_sets=visible_sets,
            repository=repository,
            reference_lines_visible=presentation.reference_lines_visible,
            intervention_annotations_visible=(
                presentation.intervention_annotations_visible
            ),
            experimental_layers=experimental_layers,
            color_by_axis_selection=color_by_axis_selection,
            experimental_point_size=(
                self.config_controller.experimental_point_size()
            ),
            experimental_point_opacity=(
                self.config_controller.experimental_point_opacity()
            ),
            selected_x_series=presentation.selected_x_series,
            selected_y_series=presentation.selected_y_series,
            selected_y_uses_default_all=(
                presentation.selected_y_uses_default_all
            ),
        )

    def _render_results_for_tab(self, tab_id: str) -> bool:
        target = str(tab_id)
        try:
            self._project_session.document.tab(target)
        except KeyError:
            return False
        if target not in self._tab_surfaces:
            return False
        try:
            projection = self._compose_result_projection(target)
        except Exception as exc:
            logger.exception("Result render failed for mechanism %s", target)
            self._invalidate_result_projection(target)
            self.set_status_text(
                f"Result display failed for mechanism {target!r}: {exc}"
            )
            self._refresh_actions()
            return False
        try:
            self._workspace.plot_for_tab(
                target
            ).apply_workspace_result_projection(projection)
        except Exception:
            # The physical panel owns invalidation and the renderFailed report;
            # this caller only stops and refreshes action eligibility.
            self._refresh_actions()
            return False
        self._refresh_actions()
        return True

    @QtCore.Slot(str, str)
    def _on_workspace_plot_render_failed(
        self,
        tab_id: str,
        message: str,
    ) -> None:
        """Report one panel-owned physical failure without clearing again."""

        self.set_status_text(
            f"Result display failed for mechanism {str(tab_id)!r}: {message}"
        )
        self._refresh_actions()

    @QtCore.Slot(str)
    def _on_workspace_result_diagnostic(self, message: str) -> None:
        """Surface a successful result projection's exact diagnostic."""

        self.set_status_text(str(message))

    @QtCore.Slot(str)
    def _on_results_changed(self, tab_id: str) -> None:
        """Ignore late results from a replaced/closed semantic tab."""

        target = str(tab_id)
        if target in self._slider_apply_refreshing:
            return
        self._render_results_for_tab(target)

    @QtCore.Slot(str, object)
    def _on_run_dispatched(self, tab_id: str, set_ids: object) -> None:
        target = str(tab_id)
        bundle = self._tab_surfaces.get(target)
        if bundle is None:
            return
        if not bundle.initial_actions.union_plot_membership(set_ids):
            return
        self._render_results_for_tab(target)
        self._refresh_project_state()

    def _invalidate_result_projection(self, tab_id: str) -> None:
        """Remove a failed tab's physical result surface without touching others."""

        target = str(tab_id)
        try:
            self._workspace.plot_for_tab(target).clear_workspace_result_projection()
        except (KeyError, RuntimeError, TypeError, ValueError):
            logger.exception("Could not clear failed workspace result surface")

    def _active_current_export_projection(
        self,
    ) -> WorkspaceResultProjection | None:
        """Return export truth only when the live active panel applied it."""

        try:
            panel = self._workspace.plot_for_tab(
                self._workspace_tab_registry.active_tab_id
            )
        except (KeyError, RuntimeError):
            return None
        return panel.xlsx_export_projection()

    def _repository_for_tab(self, tab_id: str) -> ResultRepository:
        repository = self._workspace_tab_registry.tab_runtime(tab_id).result_repository
        if not isinstance(repository, ResultRepository):
            raise LookupError(f"Tab {tab_id!r} has no ResultRepository.")
        return repository

    def _consume_canonical_mutation(
        self,
        result: CanonicalMutationResult | None,
    ) -> bool:
        """Fence only the existing Run targets affected by one registry mutation."""
        if result is None:
            return False
        if not isinstance(result, CanonicalMutationResult):
            raise TypeError(
                "Registry mutations must return CanonicalMutationResult or None."
            )
        for tab_id, set_ids in result.delta.affected_targets_by_tab:
            if set_ids:
                self._simulation_jobs.deauthorize_targets(tab_id, set_ids)
        return result.changed

    def _capture_current_explicit_result_facts(
        self,
        tab_id: str,
        set_id: str,
    ) -> ExplicitResultCurrentFacts | None:
        """Capture current typed facts for one exact result-key boundary."""

        try:
            tab = self._project_session.document.tab(str(tab_id))
            target = self._build_explicit_target_for_tab(
                tab,
                str(set_id),
                publication_generation=0,
            )
            key = target.key
            intervention_execution = target.execution_request.snapshot.intervention_execution
            facts = ExplicitResultCurrentFacts(
                tab_id=key.tab_id,
                set_id=key.set_id,
                mechanism_source=key.mechanism_source,
                prepared_dsl=target.prepared_dsl,
                analysis_identity=key.analysis_identity,
                initial_values=key.initial_values,
                execution_settings=key.execution_settings,
                scientific_settings=key.scientific_settings,
                intervention_execution=intervention_execution,
            )
            return facts
        except (KeyError, TypeError, ValueError):
            return None

    def _build_explicit_target_for_tab(
        self,
        tab: MechanismTabDocument,
        set_id: str,
        *,
        publication_generation: int,
    ):
        """Build one bound Run target through the sole exact key path."""

        if not isinstance(tab, MechanismTabDocument):
            raise TypeError("tab must be MechanismTabDocument.")
        target = tab.initial_condition_set(str(set_id))
        return build_explicit_target(
            prepared_dsl=tab.prepared_dsl,
            tab_id=tab.tab_id,
            set_id=target.set_id,
            set_name=target.name,
            mechanism_source=tab.mechanism_source,
            initial_values=canonical_initial_values(tab, target.set_id),
            execution_settings=tab.resolved_execution_settings(target.set_id),
            scientific_settings=tab.scientific_settings,
            publication_generation=publication_generation,
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

    def _on_initial_snapshot_rendered(self, tab_id: str) -> None:
        if self._rendering_workspace:
            return
        bundle = self._tab_surfaces.get(str(tab_id))
        if bundle is None:
            return
        bundle.slider_binding.render()
        self._render_results_for_tab(tab_id)
        self._refresh_title()

    def _on_run_selection_changed(self, tab_id: str) -> None:
        self._on_table_focus_changed(tab_id)

    def _on_table_focus_changed(self, tab_id: str) -> None:
        if self._rendering_workspace:
            return
        bundle = self._tab_surfaces.get(str(tab_id))
        if bundle is None:
            return
        bundle.slider_binding.render()
        self._render_results_for_tab(tab_id)
        self._refresh_slider_target_summary(tab_id)
        self._refresh_project_state()

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
        if not self._settle_execution_candidate():
            self._workspace.select_tab_surface(previous)
            return
        token = self._workspace_tab_registry.select_tab(
            target,
            expected_token=self._project_session.capture_content_token(),
        )
        if token is None:
            self._workspace.select_tab_surface(previous)
            return
        try:
            self._select_tab_surfaces(target)
        except Exception as exc:
            self.set_status_text(f"Could not switch mechanism tab: {exc}")
            logger.exception("Mechanism tab control refresh failed")

    @QtCore.Slot()
    def _create_tab(self) -> None:
        if not self._mutation_input_admitted("Create Tab"):
            return
        if not self._settle_execution_candidate():
            QtCore.QTimer.singleShot(0, self._refresh_workspace_tab_structure)
            return
        created = self._workspace_tab_registry.create_tab(
            run_defaults=self.config_controller.application_run_defaults(),
            scientific_settings=(
                self.config_controller.application_scientific_defaults()
            ),
            expected_token=self._project_session.capture_content_token(),
        )
        if created is None:
            return
        tab, result = created
        self._consume_canonical_mutation(result)
        self._create_tab_surface(tab.tab_id)
        self._refresh_workspace_tab_structure()
        self._apply_persisted_plot_state(tab.tab_id)
        self._refresh_tab_controls(tab.tab_id)

    @QtCore.Slot(str, str)
    def _rename_tab(self, tab_id: str, name: str) -> None:
        if not self._mutation_input_admitted("Rename Tab"):
            return
        try:
            result = self._workspace_tab_registry.rename_tab(
                tab_id,
                name,
                expected_token=self._project_session.capture_content_token(),
            )
            if result is None:
                return
            self._consume_canonical_mutation(result)
        except Exception as exc:
            self._workspace.show_rename_error(tab_id, str(exc))
            return
        self._refresh_workspace_tab_structure()

    @QtCore.Slot(str)
    def _close_tab(self, tab_id: str) -> None:
        if not self._mutation_input_admitted("Close Tab"):
            return
        if not self._settle_execution_candidate():
            return
        document = self._project_session.document
        if len(document.tabs) <= 1:
            self.set_status_text("The last mechanism tab cannot be closed.")
            return
        runtime = self._workspace_tab_registry.tab_runtime(tab_id)
        invalid_cells = bool(runtime.invalid_initial_text_by_set_id)
        has_content = self._workspace_tab_registry.tab_has_content_or_drafts(
            tab_id,
            fresh_run_defaults=self.config_controller.application_run_defaults(),
            fresh_scientific_settings=(
                self.config_controller.application_scientific_defaults()
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
            if invalid_cells:
                discard_label = "Discard Invalid Cell Edit(s) and Continue"
            discard = dialog.addButton(
                discard_label,
                QtWidgets.QMessageBox.ButtonRole.DestructiveRole,
            )
            if invalid_cells:
                dialog.addButton(
                    "Return and Fix",
                    QtWidgets.QMessageBox.ButtonRole.RejectRole,
                )
            dialog.addButton(QtWidgets.QMessageBox.StandardButton.Cancel)
            dialog.exec()
            if dialog.clickedButton() is not discard:
                return
        runtime.reactions_draft = None
        runtime.invalid_initial_text_by_set_id.clear()
        result = self._workspace_tab_registry.close_tab(
            tab_id,
            expected_token=self._project_session.capture_content_token(),
        )
        if result is None:
            return
        self._consume_canonical_mutation(result)
        self._drop_tab_surface(tab_id)
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
            self._refresh_slider_target_summary(tab_id)
            self._refresh_dataset_views((tab_id,))

    @QtCore.Slot(int)
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
            self._refresh_slider_target_summary(tab_id)
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
            bundle.slider_binding._evict_preview_targets(targets)
            self._refresh_slider_target_summary(tab_id)
            self._refresh_dataset_views((tab_id,))

    @QtCore.Slot(object)
    def _set_workspace_run_defaults(
        self,
        tab_id: str,
        settings: SimulationExecutionSettings,
        *,
        expected_token: ProjectContentToken | None = None,
    ) -> ProjectContentToken | None:
        if expected_token is None and not self._mutation_input_admitted(
            "Configure Run Defaults"
        ):
            return None
        bundle = self._tab_surfaces[tab_id]
        token = expected_token or self._project_session.capture_content_token()
        result = bundle.initial_actions.set_workspace_run_defaults(
            settings,
            expected_token=token,
        )
        if result is None:
            return None
        if result.changed:
            tab = self._workspace_tab_registry.tab(tab_id)
            bundle.simulation_panel.render_run_defaults(
                tab.run_defaults,
                run_admitted=self._run_admitted(tab_id),
                simulation_job_kind=self._simulation_jobs.current_kind,
            )
            self._refresh_project_state()
        return result.token

    def _activate_materialized_set(
        self,
        tab_id: str,
        set_id: str,
        *,
        expected_token: ProjectContentToken,
    ) -> ProjectContentToken | None:
        """Activate one admitted set through every existing presentation view."""

        target = str(tab_id)
        active_set_id = str(set_id)
        tab = self._workspace_tab_registry.tab(target)
        tab.initial_condition_set(active_set_id)
        state = self._workspace_tab_registry.presentation(target)
        token = self._workspace_tab_registry.set_presentation(
            target,
            replace(state, current_set_id=active_set_id),
            expected_token=expected_token,
        )
        if token is not None:
            self._workspace_tab_registry.tab_runtime(
                target
            ).selected_run_set_ids = (active_set_id,)
        return token

    @QtCore.Slot(str, bool)
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
        if not requested:
            return
        if not changed:
            return
        preview_entry = self._repository_for_tab(tab_id).preview_entry(set_id)
        if (
            preview_entry is not None
            and self._simulation_jobs.preview_key_is_current(preview_entry.key)
        ):
            self.set_status_text("Preview result plotted.")
            return
        tab = self._workspace_tab_registry.tab(tab_id)
        try:
            if not self._simulation_jobs.show_cached(tab.tab_id, set_id):
                # set_plot_requested() already rendered the changed Plot
                # membership through the retained tab surface.
                return
        except Exception as exc:
            self.set_status_text(f"Could not show cached result: {exc}")

    @QtCore.Slot(str, object)
    def _on_fit_canonical_applied(
        self,
        tab_id: str,
        scopes: object,
    ) -> None:
        if not self._mutation_input_admitted("Apply Fit Result"):
            return
        try:
            self._project_session.document.tab(tab_id)
        except KeyError:
            return
        if not isinstance(scopes, (set, frozenset, tuple, list)):
            raise TypeError("Fit Apply scopes must be an iterable of selectors.")
        selected = frozenset(str(item) for item in scopes)
        bundle = self._tab_surfaces[tab_id]
        source_changed = bool(
            selected & {"observable_definitions", "parameters"}
        )
        initials_changed = "initial_concentrations" in selected
        if source_changed:
            bundle.editor_binding.render()
            self._refresh_symbolic_calculator(tab_id)
        if initials_changed:
            bundle.initial_actions.render()
        if source_changed or initials_changed:
            bundle.slider_binding.render()
            self._refresh_slider_target_summary(tab_id)
        self._render_results_for_tab(tab_id)
        self._refresh_project_state()

    def _run_admitted(self, tab_id: str) -> bool:
        bundle = self._tab_surfaces[str(tab_id)]
        selected = bundle.initial_actions.selected_run_set_ids()
        if not selected or not self._computation_gate.is_idle:
            return False
        if bundle.initial_actions.has_invalid_cells(selected):
            return False
        source = self._visible_source_for_tab(tab_id)
        return bool(source.reactions_text.strip())

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

    def _settle_execution_candidate(
        self,
        tab_id: str | None = None,
        *,
        expected_token: ProjectContentToken | None = None,
    ) -> ProjectContentToken | None:
        target = (
            self._workspace_tab_registry.active_tab_id
            if tab_id is None
            else str(tab_id)
        )
        entry_token = (
            expected_token
            if expected_token is not None
            else self._project_session.capture_content_token()
        )
        if not self._project_session.is_content_token_current(entry_token):
            return None
        bundle = self._tab_surfaces[target]
        settled_token = bundle.initial_actions.settle_active_editor(
            expected_token=entry_token,
        )
        if (
            settled_token is None
            or not self._project_session.is_content_token_current(settled_token)
        ):
            return None
        successor_token = settled_token

        def publish_candidate(settings: SimulationExecutionSettings) -> bool:
            nonlocal successor_token
            published = self._set_workspace_run_defaults(
                target,
                settings,
                expected_token=successor_token,
            )
            if published is None:
                return False
            successor_token = published
            return True

        if not bundle.simulation_panel.settle_run_defaults(publish_candidate):
            return None
        if not self._project_session.is_content_token_current(successor_token):
            return None
        return successor_token

    def _preview_preflight(
        self,
        tab_id: str,
        target_ids: tuple[str, ...],
    ) -> MechanismSourceReconciliation | None:
        if not self._settle_execution_candidate(tab_id):
            return None
        proposal = self._accept_visible_source_result(
            tab_id,
            description="Preview",
        )
        if proposal is None:
            return None
        # Source admission may replace the selected row with an inline
        # materialized set.  Validate the post-admission target roster, not
        # the gesture's pre-admission snapshot.
        live_targets = self._tab_surfaces[tab_id].slider_binding.resolve_target_set_ids()
        if self._tab_surfaces[tab_id].initial_actions.has_invalid_cells(
            live_targets
        ):
            self.set_status_text(
                "Preview blocked by invalid visible Initial Conditions."
            )
            return None
        return proposal

    def _confirm_discard_slider_preview(
        self,
        tab_id: str,
        action: str,
    ) -> bool:
        slider_binding = self._tab_surfaces[tab_id].slider_binding
        if not slider_binding.has_slider_draft():
            return True
        transaction = slider_binding.suspend_preview_disposition()
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
            slider_binding.resume_preview_disposition(transaction)
            raise
        if dialog.clickedButton() is not discard:
            slider_binding.resume_preview_disposition(transaction)
            return False
        slider_binding.discard_preview_disposition(transaction)
        slider_binding.discard_slider_draft()
        return True

    def _start_selected_run(self, tab_id: str | None = None) -> bool:
        if not self._mutation_input_admitted("Run"):
            return False
        target = (
            self._workspace_tab_registry.active_tab_id
            if tab_id is None
            else str(tab_id)
        )
        bundle = self._tab_surfaces[target]
        selected = bundle.initial_actions.selected_run_set_ids()
        if not selected:
            self.set_status_text("Select one or more Initial Conditions rows.")
            return False
        if not self._computation_gate.is_idle:
            self.set_status_text(
                "Run rejected: another computation is active."
            )
            return False
        if not self._confirm_discard_slider_preview(target, "Run"):
            return False
        if not self._settle_execution_candidate(target):
            return False
        selected = bundle.initial_actions.selected_run_set_ids()
        if bundle.initial_actions.has_invalid_cells(selected):
            self.set_status_text(
                "Run blocked by invalid visible Initial Conditions in a selected row."
            )
            return False
        if not self._accept_visible_source(target, description="Run"):
            return False
        selected = bundle.initial_actions.selected_run_set_ids()
        if not selected:
            self.set_status_text("Select one or more Initial Conditions rows.")
            return False
        if bundle.initial_actions.has_invalid_cells(selected):
            self.set_status_text(
                "Run blocked by invalid visible Initial Conditions in a selected row."
            )
            return False
        if not self._computation_gate.is_idle:
            self.set_status_text(
                "Run rejected: another computation is active."
            )
            return False
        tab = self._workspace_tab_registry.tab(target)
        try:
            targets = tuple(
                self._build_explicit_target_for_tab(
                    tab,
                    set_id,
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
        bundle.slider_binding.cancel_pending_preview_intent()
        return bool(self._simulation_jobs.start_run(capture))

    def _inspect_transient_state(self) -> ProjectTransientState:
        reactions_draft = False
        notes_draft = False
        invalid_initials = False
        slider_drafts = False
        for tab in self._project_session.document.tabs:
            runtime = self._workspace_tab_registry.tab_runtime(tab.tab_id)
            if runtime.reactions_draft is not None:
                reactions_draft = True
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
            invalid_initial_conditions=invalid_initials,
            slider_drafts=slider_drafts,
            active_run_or_preview=bool(
                self._simulation_jobs.is_executing
                or self._simulation_jobs.pending_preview_tab_id is not None
            ),
            active_fit=bool(self._fit_session_owner.owned_qthreads()),
            active_import=bool(self._dataset_import_staging.busy),
            active_xlsx_export=self._xlsx_export_owner.is_exporting,
        )

    def _prepare_before_save(
        self,
        discard_invalid_authoring: bool = False,
    ) -> ProjectDocument | None:
        """Build the exact archive candidate without disposing visible drafts."""

        self._prepared_save_materialized = None
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
                    report_error=True,
                )
                if proposal is None:
                    return None
                candidate = proposal.proposed_document
                if proposal.materialized_set_ids:
                    materialized_by_tab.append(
                        (tab_id, proposal.materialized_set_ids)
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
        if not discard_invalid_authoring and any(
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
        self._prepared_save_materialized = (
            candidate,
            tuple(materialized_by_tab),
        )
        return candidate

    def _commit_saved_document(
        self,
        document: ProjectDocument,
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
            self._prepared_save_materialized = None
            return None
        self._consume_canonical_mutation(result)
        prepared_materialized = self._prepared_save_materialized
        self._prepared_save_materialized = None
        if (
            prepared_materialized is not None
            and prepared_materialized[0] is document
        ):
            for tab_id, set_ids in prepared_materialized[1]:
                if set_ids:
                    activation_token = self._activate_materialized_set(
                        tab_id,
                        set_ids[0],
                        expected_token=result.token,
                    )
                    if activation_token is None:
                        raise RuntimeError(
                            "Materialized Initial Conditions could not become active."
                        )
                    result = replace(result, token=activation_token)
        for tab_id, visible_draft in captured_reaction_drafts:
            tab = document.tab(tab_id)
            self._tab_surfaces[
                tab_id
            ].editor_binding.apply_accepted_source_rewrite(
                visible_draft,
                tab.mechanism_source,
            )
        for tab_id in refresh_tab_ids:
            self._refresh_tab_controls(
                tab_id,
                preserve_reactions_editor=True,
            )
        self._refresh_project_state()
        return result.token

    def _discard_authoring(self, mode: AuthoringDiscardMode) -> None:
        if mode is not AuthoringDiscardMode.ALL_UNACCEPTED:
            raise ValueError("Only whole-project unaccepted authoring can be discarded.")
        for tab in self._project_session.document.tabs:
            runtime = self._workspace_tab_registry.tab_runtime(tab.tab_id)
            runtime.reactions_draft = None
            runtime.notes_draft = None
            runtime.invalid_initial_text_by_set_id.clear()
        for bundle in self._tab_surfaces.values():
            bundle.editor_binding.render()
            bundle.initial_actions.render()
            tab = self._workspace_tab_registry.tab(bundle.tab_id)
            bundle.simulation_panel.render_run_defaults(
                tab.run_defaults,
                run_admitted=self._run_admitted(bundle.tab_id),
                simulation_job_kind=self._simulation_jobs.current_kind,
            )

    def _discard_all_slider_drafts(self) -> None:
        for tab in self._project_session.document.tabs:
            self._tab_surfaces[tab.tab_id].slider_binding.discard_slider_draft()

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
            self._set_tab_run_enabled(tab_id, self._run_admitted(tab_id))
        self._refresh_title()

    @QtCore.Slot(str, bool)
    def _on_slider_draft_state_changed(
        self,
        tab_id: str,
        dirty: bool,
    ) -> None:
        """Project only one exact tab binding's Slider Draft actions."""

        target = str(tab_id)
        bundle = self._tab_surfaces.get(target)
        if bundle is None or bundle.slider_binding.tab_id != target:
            return
        bundle.editor.set_slider_draft_actions_enabled(bool(dirty))

    def _refresh_slider_target_summary(self, tab_id: str) -> None:
        bundle = self._tab_surfaces[str(tab_id)]
        targets = bundle.slider_binding.resolve_target_set_ids()
        tab = self._workspace_tab_registry.tab(tab_id)
        names = [tab.initial_condition_set(set_id).name for set_id in targets]
        bundle.editor.set_slider_edit_targets_summary(
            "Slider edit targets: " + (", ".join(names) if names else "none")
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
            accepted = bundle.slider_binding.apply()
        finally:
            self._slider_apply_refreshing.discard(target)
        if not accepted:
            return False
        self._refresh_tab_controls(target)
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
        report_error: bool = True,
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
            try:
                return retained_tab_mechanism_source(document, tab_id)
            except (KeyError, TypeError, ValueError) as exc:
                if report_error:
                    QtWidgets.QMessageBox.warning(
                        self,
                        "Accepted Reactions Unavailable",
                        f"{description} cannot continue:\n\n{exc}",
                    )
                return None

        expected_token = self._project_session.capture_content_token()
        visible = source if source is not None else runtime.reactions_draft
        if not isinstance(visible, MechanismAuthoringSource):
            raise TypeError("Mechanism admission requires exact visible source.")
        proposal = self._prepare_source_reconciliation(
            document,
            tab_id,
            visible,
            description=description,
            report_error=report_error,
        )
        if proposal is None:
            return None

        proposed_document = proposal.proposed_document
        proposed_tab = proposed_document.tab(tab_id)
        source_changed = document != proposed_document
        if source_changed:
            result = self._workspace_tab_registry.replace_document(
                proposed_document,
                expected_token=expected_token,
            )
            if result is None:
                return None
        else:
            result = None
        runtime = self._workspace_tab_registry.tab_runtime(tab_id)
        bundle = self._tab_surfaces[tab_id]
        if proposal.materialized_set_ids:
            active_set_id = proposal.materialized_set_ids[0]
            activation_token = self._activate_materialized_set(
                tab_id,
                active_set_id,
                expected_token=(
                    result.token
                    if result is not None
                    else self._project_session.capture_content_token()
                ),
            )
            if activation_token is None:
                raise RuntimeError(
                    "Materialized Initial Conditions could not become active."
                )
            if result is not None:
                result = replace(result, token=activation_token)
        if result is not None:
            self._consume_canonical_mutation(result)
        if source is None:
            bundle.editor_binding.apply_accepted_source_rewrite(
                visible,
                proposed_tab.mechanism_source,
            )
        runtime.reactions_draft = None
        valid_species = set(canonical_species_names(proposed_tab))
        runtime.invalid_initial_text_by_set_id = {
            set_id: {
                species: raw
                for species, raw in values.items()
                if species in valid_species
            }
            for set_id, values in runtime.invalid_initial_text_by_set_id.items()
            if any(species in valid_species for species in values)
        }
        if refresh_after_admission:
            bundle.editor_binding.render()
            if source_changed:
                self._refresh_tab_controls(
                    tab_id,
                    preserve_reactions_editor=True,
                )
            else:
                self._refresh_project_state()
        return proposal

    def _accept_visible_source(
        self,
        tab_id: str | None = None,
        *,
        description: str,
        source: MechanismAuthoringSource | None = None,
        report_error: bool = True,
        refresh_after_admission: bool = True,
    ) -> bool:
        return self._accept_visible_source_result(
            tab_id,
            description=description,
            source=source,
            report_error=report_error,
            refresh_after_admission=refresh_after_admission,
        ) is not None

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
            proposal = proposal.replacement
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

    def _propose_detailed_balance_source(
        self,
        tab_id: str,
        source: MechanismAuthoringSource,
    ) -> DetailedBalanceCorrectionProposal | None:
        """Build one exact-token detailed-balance proposal without mutation."""

        if not self._mutation_input_admitted("Prepare detailed-balance correction"):
            return None
        if not isinstance(source, MechanismAuthoringSource):
            raise TypeError("source must be MechanismAuthoringSource.")
        target = str(tab_id)
        try:
            document = self._project_session.document
            expected_token = self._project_session.capture_content_token()
            reconciliation = self._prepare_source_reconciliation(
                document,
                target,
                source,
                description="Prepare detailed-balance correction",
                report_error=False,
            )
        except (KeyError, TypeError, ValueError):
            return None
        if not isinstance(reconciliation, MechanismSourceReconciliation):
            return None
        prepared_dsl = reconciliation.prepared_dsl
        if not isinstance(prepared_dsl, PreparedDslRequestSet):
            return None
        if not reconciliation.detailed_balance.resolved:
            return None
        try:
            return DetailedBalanceCorrectionProposal(
                tab_id=target,
                source=reconciliation.proposed_document.tab(
                    target
                ).mechanism_source,
                expected_token=expected_token,
                proposed_document=reconciliation.proposed_document,
                prepared_dsl=prepared_dsl,
            )
        except (KeyError, TypeError, ValueError):
            return None

    def _commit_detailed_balance_source(
        self,
        proposal: DetailedBalanceCorrectionProposal,
    ) -> DetailedBalanceCorrectionCommitResult:
        """Publish exactly one previously prepared detailed-balance proposal."""

        if not isinstance(proposal, DetailedBalanceCorrectionProposal):
            raise TypeError("proposal must be DetailedBalanceCorrectionProposal.")
        committed = False
        content_token = None
        try:
            if not self._mutation_input_admitted(
                "Apply detailed-balance correction"
            ):
                return DetailedBalanceCorrectionCommitResult(
                    proposal=proposal,
                    committed=False,
                    error="application Close is in progress",
                )
            result = self._workspace_tab_registry.replace_document(
                proposal.proposed_document,
                expected_token=proposal.expected_token,
            )
            if result is None:
                return DetailedBalanceCorrectionCommitResult(
                    proposal=proposal,
                    committed=False,
                    error="the canonical document changed before publication",
                )
            committed = True
            content_token = result.token
            self._consume_canonical_mutation(result)
            target = proposal.tab_id
            proposed_tab = proposal.proposed_document.tab(target)
            runtime = self._workspace_tab_registry.tab_runtime(target)
            visible = self._visible_source_for_tab(target)
            self._tab_surfaces[target].editor_binding.apply_accepted_source_rewrite(
                visible,
                proposed_tab.mechanism_source,
            )
            runtime.reactions_draft = None
            valid_species = set(canonical_species_names(proposed_tab))
            runtime.invalid_initial_text_by_set_id = {
                set_id: {
                    species: raw
                    for species, raw in values.items()
                    if species in valid_species
                }
                for set_id, values in runtime.invalid_initial_text_by_set_id.items()
                if any(species in valid_species for species in values)
            }
            self._refresh_tab_controls(target)
            return DetailedBalanceCorrectionCommitResult(
                proposal=proposal,
                committed=True,
                content_token=content_token,
            )
        except Exception as exc:
            if not committed:
                try:
                    if self._project_session.document == proposal.proposed_document:
                        committed = True
                        content_token = (
                            self._project_session.capture_content_token()
                        )
                except Exception:
                    # Preserve the precommit result when the live canonical
                    # boundary cannot prove that this exact proposal landed.
                    pass
            return DetailedBalanceCorrectionCommitResult(
                proposal=proposal,
                committed=committed,
                content_token=content_token if committed else None,
                error=f"{type(exc).__name__}: {exc}",
            )

    @QtCore.Slot(object)
    def _commit_preview_detailed_balance_source(
        self,
        outcome: DetailedBalanceCorrectionOutcome,
    ) -> None:
        if not self._mutation_input_admitted("Apply detailed-balance correction"):
            return
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
        proposal = self._propose_detailed_balance_source(
            outcome.tab_id,
            candidate,
        )
        if proposal is None:
            return
        result = self._commit_detailed_balance_source(proposal)
        if result.committed:
            self.set_status_text(
                (
                    "Detailed-balance correction committed, but Preview "
                    f"presentation failed: {result.error}."
                    if result.error
                    else
                    "Detailed-balance correction applied. Preview was not "
                    "retried; adjust a Slider to request a new Preview."
                )
            )
            self._refresh_tab_controls(outcome.tab_id)
        elif result.error:
            self.set_status_text(
                "Detailed-balance correction was not committed; Preview "
                f"remains unchanged: {result.error}"
            )

    @QtCore.Slot(str)
    def _clear_preview_detailed_balance_block(self, tab_id: str) -> None:
        bundle = self._tab_surfaces.get(str(tab_id))
        if bundle is not None:
            bundle.editor.clear_preview_detailed_balance_block()

    def _present_detailed_balance_correction(self, request: object) -> None:
        tab_id = str(getattr(request, "tab_id", ""))
        bundle = self._tab_surfaces.get(tab_id)
        if bundle is not None:
            bundle.editor.present_detailed_balance_correction(request)

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
                "will discard that tab's unapplied Reactions and Slider drafts. "
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
        runtime.reactions_draft = None
        self._tab_surfaces[target].slider_binding.discard_slider_draft(
            render=False
        )
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
        if not self._mutation_input_admitted("Load Intervention Example"):
            return
        try:
            from kindred.io.resources import get_intervention_example_source

            source = get_intervention_example_source(example_id)
            if self._accept_external_source(
                self._workspace_tab_registry.active_tab_id,
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
                description="Apply mechanism template",
            ),
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
    @QtCore.Slot(bool)
    def _on_import_busy_changed(self, _busy: bool) -> None:
        self._refresh_actions()

    @QtCore.Slot(object)
    def _commit_import_completion(
        self,
        completion: DatasetImportCompletion,
    ) -> None:
        if not isinstance(completion, DatasetImportCompletion):
            raise TypeError("Dataset import completion has the wrong type.")
        if not self._mutation_input_admitted("Commit Dataset Import"):
            self.set_status_text(
                "Dataset import completed after Close armed; publication was skipped."
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
            result = self._workspace_tab_registry.replace_document(
                document,
                expected_token=self._project_session.capture_content_token(),
            )
            if result is None:
                self.set_status_text(
                    "Dataset import publication was superseded; the completed "
                    "import was not committed."
                )
                self._refresh_project_state()
                return
            self._consume_canonical_mutation(result)
        except Exception as exc:
            logger.exception("Raw dataset commit failed")
            QtWidgets.QMessageBox.critical(
                self,
                "Dataset Import",
                "The completed import could not be committed. "
                f"{exc}",
            )
            self.set_status_text(
                "Dataset import publication was rejected; staging is idle and "
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
                + " The initiating mechanism no longer exists; raw data "
                "remains committed and optional mapping was skipped."
            )
            return
        proposal = self._accept_visible_source_result(
            origin_tab_id,
            description="Prepare imported dataset mapping",
        )
        if proposal is None:
            self.set_status_text(
                message
                + " Optional mapping was canceled or blocked by visible "
                "Reactions, so it was skipped."
                + " Raw data remains committed and unmapped; fix Reactions, then"
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
            species = canonical_species_names(tab)
            if not species:
                raise DatasetMappingError(
                    "Optional mapping requires a valid mechanism with species."
                )
            review = DatasetMappingBatchDialog(
                records=batch_proposal.records,
                tab=tab,
                mechanism_species=species,
                parent=self,
            )
            if review.exec() != QtWidgets.QDialog.DialogCode.Accepted:
                self.set_status_text(
                    message + " Mapping canceled; raw data remains unmapped."
                )
                return
            batch = review.proposal()
            candidate_tab = apply_dataset_mapping_batch(
                document=self._project_session.document,
                tab=tab,
                batch=batch,
                mechanism_species=species,
            )
            result = self._workspace_tab_registry.replace_tab(
                origin_tab_id,
                candidate_tab,
                expected_token=self._project_session.capture_content_token(),
            )
            if result is None:
                self.set_status_text(
                    message + " Mapping was not published; raw data remains unmapped."
                )
                return
            mapping_changed = self._consume_canonical_mutation(result)
        except Exception as exc:
            logger.exception("Batch dataset mapping failed")
            QtWidgets.QMessageBox.warning(
                self,
                "Dataset Mapping",
                f"The reviewed mapping was not applied:\n\n{exc}",
            )
            self.set_status_text(
                message + " Mapping failed validation; raw data remains unmapped."
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

    @QtCore.Slot(object)
    def _rename_dataset(self, intent: DatasetRenameIntent) -> None:
        if not self._mutation_input_admitted("Rename Dataset"):
            return
        if not isinstance(intent, DatasetRenameIntent):
            return
        changed = False
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
            result = self._workspace_tab_registry.replace_document(
                candidate,
                expected_token=self._project_session.capture_content_token(),
            )
            if result is not None:
                changed = self._consume_canonical_mutation(result)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Rename Dataset", str(exc))
        if changed:
            self._refresh_dataset_views()

    @QtCore.Slot(object)
    def _delete_dataset(self, intent: DatasetDeleteIntent) -> None:
        if not self._mutation_input_admitted("Delete Dataset"):
            return
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
                tab_id: state.overlay_raw_observation_ids_by_dataset
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
        result = self._workspace_tab_registry.replace_document(
            candidate,
            expected_token=self._project_session.capture_content_token(),
        )
        if result is None:
            return
        self._consume_canonical_mutation(result)
        self._refresh_dataset_views()

    @QtCore.Slot(object)
    def _export_original_dataset(
        self,
        intent: DatasetExportOriginalIntent,
    ) -> None:
        if not self._mutation_input_admitted("Export Original Dataset"):
            return
        if not isinstance(intent, DatasetExportOriginalIntent):
            return
        try:
            record = self._project_session.document.dataset(intent.dataset_id)
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
                self._project_session.document.imported_source(record.source_id).original_filename,
            ),
            "All Files (*)",
        )
        if not path:
            return
        self._remember_export_directory(path)
        try:
            exported = export_original_source(self._project_session.document, record, path)
            self.set_status_text(f"Exported original dataset: {exported}")
        except (DatasetMappingError, OSError) as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "Export Original Dataset",
                str(exc),
            )

    def _map_dataset(
        self,
        tab_id: str,
        intent: DatasetMappingIntent,
    ) -> None:
        if not self._mutation_input_admitted("Map Dataset"):
            return
        if not isinstance(intent, DatasetMappingIntent):
            return
        changed = False
        previous_set_ids: tuple[str, ...] = ()
        try:
            choice = DatasetMappingChoice(intent.action)
            description = (
                "Leave Dataset Unmapped"
                if choice is DatasetMappingChoice.LEAVE_UNMAPPED
                else "Map Dataset"
            )
            if not self._accept_visible_source(tab_id, description=description):
                return
            tab = self._workspace_tab_registry.tab(tab_id)
            previous_set_ids = tuple(
                item.set_id for item in tab.initial_condition_sets
            )
            dataset = self._project_session.document.dataset(intent.dataset_id)
            if choice is DatasetMappingChoice.CREATE_SET:
                species = canonical_species_names(tab)
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
                suggested_name = allocate_initial_condition_set_name(
                    dataset.display_name,
                    existing_names=(
                        item.name for item in tab.initial_condition_sets
                    ),
                )
                candidate_tab = create_set_from_exact_zero(
                    tab=tab,
                    dataset=dataset,
                    set_id=new_set_id(),
                    set_name=suggested_name,
                    model_species=species,
                    confirmed_mapping=mapping,
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
                        new_name=allocate_initial_condition_set_name(
                            source_set.name,
                            existing_names=(
                                item.name for item in tab.initial_condition_sets
                            ),
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
                    mechanism_species=canonical_species_names(tab),
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
                    mechanism_species=canonical_species_names(tab),
                )
            result = self._workspace_tab_registry.replace_tab(
                tab_id,
                candidate_tab,
                expected_token=self._project_session.capture_content_token(),
            )
            if result is not None:
                changed = self._consume_canonical_mutation(result)
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
            self._refresh_slider_target_summary(tab_id)
        self._refresh_dataset_views((tab_id,))

    def _edit_observation_mapping(
        self,
        tab_id: str,
        intent: DatasetObservationMappingIntent,
    ) -> None:
        if not self._mutation_input_admitted("Edit Observation Mapping"):
            return
        if not isinstance(intent, DatasetObservationMappingIntent):
            return
        changed = False
        try:
            if not self._accept_visible_source(
                tab_id,
                description="Edit Observation Mapping"
            ):
                return
            tab = self._workspace_tab_registry.tab(tab_id)
            dataset = self._project_session.document.dataset(intent.dataset_id)
            species = canonical_species_names(tab)
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
            result = self._workspace_tab_registry.replace_tab(
                tab_id,
                candidate,
                expected_token=self._project_session.capture_content_token(),
            )
            if result is not None:
                changed = self._consume_canonical_mutation(result)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Observation Mapping", str(exc))
        if changed:
            self._refresh_dataset_views((tab_id,))

    def _show_plot_overlay(
        self,
        tab_id: str,
        intent: DatasetPlotOverlayIntent,
    ) -> None:
        if not isinstance(intent, DatasetPlotOverlayIntent):
            return
        bundle = self._tab_surfaces.get(str(tab_id))
        if bundle is None:
            return
        bundle.right_panel.select_plot_overlays_page()
        bundle.overlay_panel.navigate_to_dataset(intent.dataset_id)

    def _prompt_observation_mapping(
        self,
        dataset,
        current_mapping: Iterable[tuple[str, str]],
        model_species: Sequence[str],
        *,
        title: str,
    ) -> tuple[tuple[str, str], ...] | None:
        species = tuple(dict.fromkeys(str(name) for name in model_species))
        observations = tuple(
            (series.raw_observation_id, series.column_name)
            for series in dataset.observations
        )
        current = dict(current_mapping)
        suggestions = {
            item.raw_observation_id: item.model_species
            for item in suggest_observation_mapping(
                raw_observations=observations,
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
        for raw_observation_id, column in observations:
            combo = QtWidgets.QComboBox(dialog)
            combo.addItem("(unmapped)", None)
            for species_name in species:
                combo.addItem(species_name, species_name)
            selected = current.get(
                raw_observation_id,
                suggestions.get(raw_observation_id),
            )
            if selected in species:
                combo.setCurrentIndex(combo.findData(selected))
            form.addRow(column, combo)
            editors[raw_observation_id] = combo
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
            (raw_observation_id, str(combo.currentData()))
            for raw_observation_id, combo in editors.items()
            if combo.currentData() is not None
        )
        if len({target for _raw_id, target in mapping}) != len(mapping):
            QtWidgets.QMessageBox.warning(
                self,
                title,
                "Each mechanism species may be mapped from at most one column.",
            )
            return None
        return mapping

    def _set_dataset_fit_inclusion(
        self,
        tab_id: str,
        intent: DatasetFitInclusionIntent,
    ) -> None:
        if not self._mutation_input_admitted("Change Fit Dataset Inclusion"):
            return
        if not isinstance(intent, DatasetFitInclusionIntent):
            return
        changed = False
        try:
            if not self._accept_visible_source(
                tab_id,
                description="Change Fit Dataset Inclusion"
            ):
                return
            tab = self._workspace_tab_registry.tab(tab_id)
            candidate = with_validated_fit_inclusion(
                document=self._project_session.document,
                tab=tab,
                dataset_id=intent.dataset_id,
                include=intent.include,
                mechanism_species=canonical_species_names(tab),
            )
            result = self._workspace_tab_registry.replace_tab(
                tab_id,
                candidate,
                expected_token=self._project_session.capture_content_token(),
            )
            if result is not None:
                changed = self._consume_canonical_mutation(result)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Fit Dataset", str(exc))
        if changed:
            self._refresh_dataset_views((tab_id,))

    def _set_overlay_selection(
        self,
        tab_id: str,
        intent: OverlaySelectionIntent,
    ) -> None:
        if not self._mutation_input_admitted("Change Dataset Overlay"):
            return
        if not isinstance(intent, OverlaySelectionIntent):
            return
        state = self._workspace_tab_registry.presentation(tab_id)
        overlays = dict(state.overlay_raw_observation_ids_by_dataset)
        if intent.raw_observation_ids:
            overlays[intent.dataset_id] = tuple(intent.raw_observation_ids)
        else:
            overlays.pop(intent.dataset_id, None)
        token = self._workspace_tab_registry.set_presentation(
            tab_id,
            replace(
                state,
                overlay_raw_observation_ids_by_dataset=tuple(overlays.items()),
            ),
            expected_token=self._project_session.capture_content_token(),
        )
        if token is None:
            return
        self._tab_surfaces[tab_id].overlay_panel.render_overlay_selection(
            dataset_id=intent.dataset_id,
            raw_observation_ids=intent.raw_observation_ids,
        )
        self._render_results_for_tab(tab_id)
        self._refresh_project_state()

    def _set_reference_lines_visible(self, tab_id: str, visible: bool) -> None:
        if not self._mutation_input_admitted("Change Reference Lines"):
            return
        state = self._workspace_tab_registry.presentation(tab_id)
        token = self._workspace_tab_registry.set_presentation(
            tab_id,
            replace(state, reference_lines_visible=bool(visible)),
            expected_token=self._project_session.capture_content_token(),
        )
        if token is None:
            return
        self._refresh_project_state()
        self._render_results_for_tab(tab_id)

    def _set_intervention_annotations_visible(
        self,
        tab_id: str,
        visible: bool,
    ) -> None:
        if not self._mutation_input_admitted(
            "Change Intervention Annotations"
        ):
            return
        state = self._workspace_tab_registry.presentation(tab_id)
        token = self._workspace_tab_registry.set_presentation(
            tab_id,
            replace(
                state,
                intervention_annotations_visible=bool(visible),
            ),
            expected_token=self._project_session.capture_content_token(),
        )
        if token is None:
            return
        self._refresh_project_state()
        self._render_results_for_tab(tab_id)

    def _set_x_series(
        self,
        tab_id: str,
        selection: AxisSeriesSelection,
    ) -> None:
        if not self._mutation_input_admitted("Change X Series"):
            return
        if not isinstance(
            selection,
            (SimulationSeriesAxisSelection, ExperimentalAxisSelection),
        ):
            raise TypeError("X-axis selection must be typed.")
        state = self._workspace_tab_registry.presentation(tab_id)
        token = self._workspace_tab_registry.set_presentation(
            tab_id,
            replace(state, selected_x_series=selection),
            expected_token=self._project_session.capture_content_token(),
        )
        if token is None:
            return
        self._refresh_project_state()
        self._render_results_for_tab(tab_id)

    def _set_y_series(self, tab_id: str, selections: object) -> None:
        if not self._mutation_input_admitted("Change Y Series"):
            return
        selected = tuple(selections or ())
        if any(
            not isinstance(
                selection,
                (SimulationSeriesAxisSelection, ExperimentalAxisSelection),
            )
            for selection in selected
        ):
            raise TypeError("Y-axis selections must be typed.")
        state = self._workspace_tab_registry.presentation(tab_id)
        token = self._workspace_tab_registry.set_presentation(
            tab_id,
            replace(
                state,
                selected_y_series=selected,
                selected_y_uses_default_all=False,
            ),
            expected_token=self._project_session.capture_content_token(),
        )
        if token is None:
            return
        self._refresh_project_state()
        self._render_results_for_tab(tab_id)

    def _set_plot_presentation(
        self,
        tab_id: str,
        intent: PlotPresentationIntent,
    ) -> None:
        if not self._mutation_input_admitted("Change Plot Presentation"):
            return
        if not isinstance(intent, PlotPresentationIntent):
            return
        state = self._workspace_tab_registry.presentation(tab_id)
        next_state = replace(state, **{intent.option: intent.enabled})
        token = self._workspace_tab_registry.set_presentation(
            tab_id,
            next_state,
            expected_token=self._project_session.capture_content_token(),
        )
        if token is None:
            return
        self._workspace.plot_for_tab(tab_id).set_persisted_plot_state(
            tab_id,
            log_x=next_state.log_x,
            log_y=next_state.log_y,
            legend_visible=next_state.legend_visible,
            x_inverted=next_state.x_inverted,
            y_inverted=next_state.y_inverted,
        )
        self._refresh_project_state()
        self._render_results_for_tab(tab_id)

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
            lambda: self._selected_bundle().data_panel.trigger_load_dialog(),
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
            lambda: self._selected_bundle().editor.undo_focused_text(),
            QtGui.QKeySequence.StandardKey.Undo,
            "undoAction",
        )
        self._redo_action = self._add_action(
            edit_menu,
            "Redo",
            lambda: self._selected_bundle().editor.redo_focused_text(),
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
        preferences.addAction(
            self.config_controller.result_cache_capacity_action
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
            lambda: self._start_selected_run(),
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
            lambda: self._open_symbolic_calculator(
                self._workspace_tab_registry.active_tab_id
            ),
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
        if not self._mutation_input_admitted("Configure Run Parallel Maximum"):
            return
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
        self.config_controller.set_run_user_maximum(int(value))

    def _configure_parameter_debounce(self) -> None:
        if not self._mutation_input_admitted("Configure Parameter Preview"):
            return
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
        self.config_controller.set_mechanism_parameter_debounce_ms(int(value))

    def _configure_equilibrium_debounce(self) -> None:
        if not self._mutation_input_admitted("Configure Equilibrium Preview"):
            return
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
        self.config_controller.set_equilibrium_k_debounce_ms(int(value))

    def _open_scientific_settings(self) -> None:
        if not self._mutation_input_admitted("Configure Scientific Settings"):
            return
        tab_id = self._workspace_tab_registry.active_tab_id
        if not self._settle_execution_candidate(tab_id):
            return
        previous_settings = self._workspace_tab_registry.tab(
            tab_id
        ).scientific_settings
        dialog = TabScientificSettingsDialog(self)
        dialog.set_scientific_settings(
            self._workspace_tab_registry.tab(tab_id).scientific_settings
        )
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        settings = dialog.scientific_settings()
        bundle = self._tab_surfaces[tab_id]
        try:
            # Canonical publication may re-admit a temperature-dependent DSL
            # and reject the entire candidate.  The action owner publishes
            # atomically, so a raised validation error leaves the document
            # unchanged.
            changed = bundle.initial_actions.set_tab_scientific_settings(settings)
        except (TypeError, ValueError, KeyError, OverflowError) as exc:
            message = str(exc)
            self.set_status_text(message)
            QtWidgets.QMessageBox.warning(
                self,
                "Scientific Settings Rejected",
                message,
            )
            return
        if not changed and settings != previous_settings:
            message = (
                "Scientific settings rejected: canonical project content "
                "changed before publication."
            )
            self.set_status_text(message)
            QtWidgets.QMessageBox.warning(
                self,
                "Scientific Settings Rejected",
                message,
            )
            return
        if changed:
            changed_fields = frozenset(
                field
                for field in (
                    "temperature",
                    "rtol",
                    "atol",
                    "analytic_jacobian_policy",
                    "wegscheider_policy",
                )
                if getattr(previous_settings, field)
                != getattr(settings, field)
            )
            try:
                self.config_controller.persist_scientific_defaults(settings)
            except Exception as exc:
                logger.exception(
                    "Scientific settings persistence failed for tab %s",
                    tab_id,
                )
                self.set_status_text(
                    f"Scientific settings persistence failed: {exc}"
                )
            finally:
                self._refresh_tab_controls(
                    tab_id,
                    scientific_fields=changed_fields,
                )

    def _run_global_fit(self) -> bool:
        if not self._mutation_input_admitted("Global Fit"):
            return False
        tab_id = self._workspace_tab_registry.active_tab_id
        bundle = self._tab_surfaces[tab_id]
        tab = self._workspace_tab_registry.tab(tab_id)
        if not self._computation_gate.is_idle:
            self.set_status_text(
                "Global Fit rejected: another computation is active."
            )
            return False
        if not self._confirm_discard_slider_preview(tab_id, "Global Fit"):
            return False
        if not self._settle_execution_candidate(tab_id):
            return False
        if not self._accept_visible_source(tab_id, description="Global Fit"):
            return False
        tab = self._workspace_tab_registry.tab(tab_id)
        fit_set_ids = tuple(
            link.set_id
            for _dataset_id, link in tab.dataset_links
            if link.include_in_fit and link.set_id is not None
        )
        if bundle.initial_actions.has_invalid_cells(fit_set_ids):
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
        bundle.slider_binding.cancel_pending_preview_intent()
        return self._fit_session_owner.open_or_focus(tab.tab_id) is not None

    def _configure_fitting(self) -> bool:
        if not self._mutation_input_admitted("Configure Fitting"):
            return False
        defaults = self._fit_session_owner.project_defaults
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Fitting Defaults")
        form = QtWidgets.QFormLayout(dialog)
        scope = QtWidgets.QComboBox(dialog)
        scope.addItem("Current project", "project")
        scope.addItem("New projects (application default)", "application")
        method = QtWidgets.QComboBox(dialog)
        method.addItems(list(SUPPORTED_FIT_METHODS))
        method.setCurrentText(defaults.method)
        max_nfev = QtWidgets.QSpinBox(dialog)
        max_nfev.setRange(*FIT_MAX_NFEV_RANGE)
        max_nfev.setValue(defaults.max_nfev)
        ftol = QtWidgets.QLineEdit(dialog)
        xtol = QtWidgets.QLineEdit(dialog)
        setup_scientific_validator(ftol)
        setup_scientific_validator(xtol)
        ftol.setText(repr(float(defaults.ftol)))
        xtol.setText(repr(float(defaults.xtol)))
        ftol_label = QtWidgets.QLabel(FUNCTION_TOLERANCE_LABEL, dialog)
        xtol_label = QtWidgets.QLabel("Step tolerance", dialog)
        use_seed = QtWidgets.QCheckBox("Use deterministic seed", dialog)
        use_seed.setChecked(defaults.use_seed)
        seed = QtWidgets.QSpinBox(dialog)
        seed.setRange(*FIT_SEED_RANGE)
        seed.setValue(defaults.seed)

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
            method.setCurrentText(selected.method)
            max_nfev.setValue(selected.max_nfev)
            project_tolerance(ftol, selected.ftol)
            project_tolerance(xtol, selected.xtol)
            use_seed.setChecked(selected.use_seed)
            seed.setValue(selected.seed)
            sync_method_controls()

        def sync_method_controls(_method: str = "") -> None:
            is_de = method.currentText() == DIFFERENTIAL_EVOLUTION_METHOD
            ftol_label.setText(
                DE_POPULATION_TOLERANCE_LABEL
                if is_de
                else FUNCTION_TOLERANCE_LABEL
            )
            xtol_label.setVisible(not is_de)
            xtol.setVisible(not is_de)
            xtol.setEnabled(not is_de)

        scope.currentIndexChanged.connect(load_selected_defaults)
        method.currentTextChanged.connect(sync_method_controls)
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

        def accept_candidate() -> None:
            try:
                validate_fit_defaults(
                    FitDefaults(
                        method=method.currentText().strip().lower(),
                        max_nfev=int(max_nfev.value()),
                        ftol=float(ftol.text()),
                        xtol=float(xtol.text()),
                        use_seed=bool(use_seed.isChecked()),
                        seed=int(seed.value()),
                    )
                )
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
        form.addRow(buttons)
        sync_method_controls()
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return False
        candidate = FitDefaults(
            method=method.currentText().strip().lower(),
            max_nfev=int(max_nfev.value()),
            ftol=float(ftol.text()),
            xtol=float(xtol.text()),
            use_seed=bool(use_seed.isChecked()),
            seed=int(seed.value()),
        )
        validate_fit_defaults(candidate)
        if scope.currentData() == "project":
            changed = self._fit_session_owner.set_project_defaults(candidate)
            if changed:
                self._refresh_project_state()
            return changed
        else:
            self.config_controller.persist_fit_defaults(candidate)
        return True

    def _export_xlsx(self) -> None:
        if not self._mutation_input_admitted("Export Results XLSX"):
            return
        projection = self._active_current_export_projection()
        if projection is None:
            self.set_status_text("No result data is available to export.")
            return
        try:
            WorkspaceResultPresenter.validate_xlsx_export(projection)
        except (TypeError, ValueError) as exc:
            self.set_status_text(str(exc))
            QtWidgets.QMessageBox.critical(
                self,
                "Export Results XLSX",
                str(exc),
            )
            return
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
            return
        path, _selected = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export Results XLSX",
            start_directory,
            "Excel Workbook (*.xlsx)",
        )
        if not path:
            return
        if not path.casefold().endswith(".xlsx"):
            path += ".xlsx"
        self._remember_export_directory(path)
        if self._xlsx_export_owner.start(projection, path):
            self.set_status_text(f"Exporting XLSX: {path}")
            self._refresh_actions()

    def _remember_export_directory(self, path: str) -> None:
        directory = os.path.dirname(os.path.abspath(os.fspath(path)))
        if os.path.isdir(directory):
            self._last_export_directory = directory

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
        status = (
            outcome.message
            if outcome.cancelled
            else f"XLSX export failed: {outcome.message}"
        )
        self.set_status_text(status)
        if not outcome.cancelled:
            QtWidgets.QMessageBox.critical(
                self,
                "Export Results XLSX",
                outcome.message,
            )
        self._refresh_actions()

    # ------------------------------------------------------------------
    # Symbolic, inspector, and auxiliary UI
    # ------------------------------------------------------------------
    def _refresh_symbolic_calculator(self, tab_id: str) -> None:
        bundle = self._tab_surfaces[str(tab_id)]
        if self._workspace_tab_registry.tab_runtime(
            tab_id
        ).reactions_draft is not None:
            state = bundle.calculator_owner.reset_project_session_state()
        else:
            state = bundle.calculator_owner.refresh()
        bundle.calculator_panel.render_state(state)
        bundle.editor.symbolic_calculator_btn.setEnabled(state.available)

    def _open_symbolic_calculator(self, tab_id: str) -> None:
        if not self._mutation_input_admitted("Open Symbolic Calculator"):
            return
        if not self._accept_visible_source(
            tab_id,
            description="Open Symbolic Calculator"
        ):
            return
        self._refresh_symbolic_calculator(tab_id)
        if tab_id == self._workspace_tab_registry.active_tab_id:
            self._calculator_stack.setCurrentWidget(
                self._tab_surfaces[tab_id].calculator_panel
            )
        self._symbolic_calculator_dock.show()
        self._symbolic_calculator_dock.raise_()

    def _evaluate_symbolic(self, tab_id: str, query: str):
        bundle = self._tab_surfaces[str(tab_id)]
        if not self._mutation_input_admitted("Evaluate Symbolic Expression"):
            return bundle.calculator_owner.reset_project_session_state()
        if not self._accept_visible_source(
            tab_id,
            description="Evaluate Symbolic Expression"
        ):
            return bundle.calculator_owner.reset_project_session_state()
        return bundle.calculator_owner.evaluate(query)

    def _open_mechanism_inspector(self, tab_id: str) -> None:
        from kindred.gui.widgets.mechanism_inspector import MechanismInspectorDialog

        if not self._mutation_input_admitted("Open Mechanism Inspector"):
            return
        if not self._accept_visible_source(
            tab_id,
            description="Open Mechanism Inspector"
        ):
            return
        try:
            MechanismInspectorDialog(
                self._workspace_tab_registry.tab(tab_id).prepared_dsl,
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
            "Diagnostic logs are written to the platform log directory "
            "(Linux: `~/.cache/kindred/logs`, macOS: "
            "`~/Library/Logs/kindred`, Windows: `%LOCALAPPDATA%\\kindred\\logs`; "
            "override with `KINDRED_LOG_DIR`). If that location is unavailable, "
            "logs are written to the system temporary directory under "
            "`kindred/logs`.\n\n"
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
        if not self._mutation_input_admitted("Reset Layout"):
            return
        self._apply_default_dock_arrangement()

    def redock_shell_dock(self, dock: QtWidgets.QDockWidget) -> None:
        if not self._mutation_input_admitted("Redock Panel"):
            return
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
        if not self._mutation_input_admitted("Open Analysis Surface"):
            return False
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
        for tab_id in tuple(self._tab_surfaces):
            self._drop_tab_surface(tab_id)
        for tab in self._project_session.document.tabs:
            self._create_tab_surface(tab.tab_id)
        self._refresh_workspace_document()

    def _refresh_project_state(self) -> None:
        self._render_tab_state()
        self._refresh_title()
        self._refresh_actions()
        self.config_controller.update_recent_files_menu()

    @QtCore.Slot()
    def _on_simulation_state_changed(self) -> None:
        """Project the live simulation kind into every retained panel."""

        job_kind = self._simulation_jobs.current_kind
        for bundle in self._tab_surfaces.values():
            try:
                bundle.simulation_panel.render_job_state(job_kind)
            except Exception as exc:
                logger.exception(
                    "Simulation job-state render failed for tab %s",
                    bundle.tab_id,
                )
                self.set_status_text(
                    f"Simulation job-state render failed: {exc}"
                )
        self._refresh_actions()

    @QtCore.Slot(bool)
    def _on_close_pending_changed(self, pending: bool) -> None:
        if pending:
            self._set_close_mutation_fence(True)
            self.set_status_text("Closing Kindred; waiting for active work to stop.")
        else:
            self._set_close_mutation_fence(False)

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
        if not self._tab_surfaces:
            return
        tab_id = self._workspace_tab_registry.active_tab_id
        bundle = self._tab_surfaces.get(tab_id)
        if bundle is None:
            return
        mutation_admitted = self._mutation_input_admitted()
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
            mutation_admitted and self._simulation_jobs.is_executing
        )
        self._global_fit_action.setEnabled(
            mutation_admitted and self._computation_gate.is_idle
        )
        if self._xlsx_export_action is not None:
            self._xlsx_export_action.setEnabled(
                mutation_admitted
                and self._active_current_export_projection() is not None
                and not self._xlsx_export_owner.is_exporting
            )

    def _apply_slider_debounce_settings(self) -> None:
        for bundle in self._tab_surfaces.values():
            bundle.slider_binding.set_preview_debounce_intervals(
                parameter_ms=(
                    self.config_controller.mechanism_parameter_debounce_ms()
                ),
                equilibrium_K_ms=(
                    self.config_controller.equilibrium_k_debounce_ms()
                ),
            )

    @QtCore.Slot(int)
    def _resize_result_repositories(self, capacity: int) -> None:
        for tab in self._project_session.document.tabs:
            self._repository_for_tab(tab.tab_id).resize_cache_capacity(
                int(capacity)
            )
            self._render_results_for_tab(tab.tab_id)

    def authorize_and_schedule_close(self) -> None:
        self._set_close_mutation_fence(True)
        self._native_close_authorized = True
        QtCore.QTimer.singleShot(0, self.close)

    def _hard_exit(self) -> None:
        os._exit(0)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if not self._native_close_authorized:
            event.ignore()
            self.project_controller.request_close()
            return
        if not self.project_controller.consume_native_close_authorization():
            self._native_close_authorized = False
            event.ignore()
            return
        self.config_controller.save_settings()
        event.accept()

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
