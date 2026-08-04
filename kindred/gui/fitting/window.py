"""Four-tab application-modal Fit window with local-only state and plots."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import replace

import numpy as np
from PySide6 import QtCore, QtWidgets

from kindred.core.analysis.dataset_sampling import compute_sampled_indices
from kindred.core.project_document import (
    FitDefaults,
    SimulationExecutionSettings,
)
from kindred.gui.fitting.constants import (
    DE_POPULATION_TOLERANCE_LABEL,
    DIFFERENTIAL_EVOLUTION_METHOD,
    FITTING_MAX_NFEV_RANGE,
    FITTING_METHODS,
    FITTING_SEED_RANGE,
    FUNCTION_TOLERANCE_LABEL,
    SIMULATION_SOLVERS,
)
from kindred.gui.fitting.contracts import (
    FitParametricSettings,
    FitRunIdentity,
    FitTargetSelection,
    copy_target_selection,
)
from kindred.gui.fitting.launch import (
    FitDatasetLocalSettings,
    FitDatasetSetupEdit,
    FitDefaultsSetupEdit,
    FitInitialSetupEdit,
    FitInitialVariable,
    FitOriginSnapshot,
    FitParameterSetupEdit,
    FitParameterVariable,
    FitSetupSnapshot,
)
from kindred.gui.fitting.worker import (
    FitCandidate,
    FitCurveDataset,
    FitLiveUpdate,
)
from kindred.gui.widgets.grid_plot_view import GridPlotView
from kindred.gui.widgets.mechanism_inspector import MechanismInspectorDialog

__all__ = ["FittingWindow"]


def _check_item(checked: bool) -> QtWidgets.QTableWidgetItem:
    item = QtWidgets.QTableWidgetItem()
    item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
    item.setCheckState(
        QtCore.Qt.CheckState.Checked
        if checked
        else QtCore.Qt.CheckState.Unchecked
    )
    return item


def _text_item(value: object, *, editable: bool = True) -> QtWidgets.QTableWidgetItem:
    item = QtWidgets.QTableWidgetItem(str(value))
    if not editable:
        item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
    return item


def _float_text(value: float) -> str:
    return f"{float(value):.12g}"


def _read_float(text: object, field: str) -> float:
    value = float(str(text))
    if not math.isfinite(value):
        raise ValueError(f"{field} must be finite.")
    return value


_PARAMETER_SHARED_BASELINE_ROLE = (
    QtCore.Qt.ItemDataRole.UserRole + 20
)


class FittingWindow(QtWidgets.QDialog):
    """Physical editor/view; the FitSessionOwner builds and retains typed requests."""

    executeRequested = QtCore.Signal()
    cancelRequested = QtCore.Signal()
    pauseRequested = QtCore.Signal()
    resumeRequested = QtCore.Signal()
    applyRequested = QtCore.Signal(object)
    addObservableRequested = QtCore.Signal(str, str, str, str, str)
    setupEditsRequested = QtCore.Signal(object)
    restoreSetupRequested = QtCore.Signal(object)
    projectDefaultsRequested = QtCore.Signal(object)

    def __init__(
        self,
        origin: FitOriginSnapshot,
        *,
        close_handler: Callable[[float], bool],
        color_resolver: Callable[[str], object] | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        if not isinstance(origin, FitOriginSnapshot):
            raise TypeError("origin must be FitOriginSnapshot.")
        if not callable(close_handler):
            raise TypeError("close_handler must be callable.")
        if color_resolver is not None and not callable(color_resolver):
            raise TypeError("color_resolver must be callable.")
        self._origin = origin
        self._close_handler = close_handler
        self._color_resolver = color_resolver
        self._closing_after_cleanup = False
        self._dataset_view_by_id: dict[str, GridPlotView] = {}
        self._paused = False
        self._initial_setup: FitSetupSnapshot | None = None
        self._last_fit_setup: FitSetupSnapshot | None = None
        self._snapshot_capture_id: str | None = None
        self._run_identity = None
        self._candidate_curves: tuple[FitCurveDataset, ...] = ()
        self._species_filter_initialized = False
        self._refreshing_dataset_controls = False
        self._syncing_scopes = False
        self._committed_targets: dict[
            tuple[str, str],
            tuple[bool, str, str],
        ] = {}
        self._projected_setup: FitSetupSnapshot | None = None
        self._invalid_setup_edits: set[tuple[str, ...]] = set()
        self._projecting_setup = False
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal)
        self.setModal(True)
        self.setWindowTitle(
            f"Global Fit — Origin: {origin.tab_name} [{origin.tab_id}]"
        )
        self.resize(1080, 760)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        self._origin_label = QtWidgets.QLabel(self)
        layout.addWidget(self._origin_label)
        self._tabs = QtWidgets.QTabWidget(self)
        layout.addWidget(self._tabs, 1)

        self._data_tab = QtWidgets.QWidget(self._tabs)
        self._parameters_tab = QtWidgets.QWidget(self._tabs)
        self._initials_tab = QtWidgets.QWidget(self._tabs)
        self._results_tab = QtWidgets.QWidget(self._tabs)
        self._tabs.addTab(self._data_tab, "Data and Targets")
        self._tabs.addTab(self._parameters_tab, "Parameters")
        self._tabs.addTab(
            self._initials_tab,
            "Initial Conditions and Simulation",
        )
        self._tabs.addTab(self._results_tab, "Results")
        self._build_data_tab()
        self._build_parameters_tab()
        self._build_initials_tab()
        self._build_results_tab()

        self._progress = QtWidgets.QProgressBar(self)
        self._progress.setRange(0, 100)
        layout.addWidget(self._progress)
        self._status = QtWidgets.QLabel("Ready.", self)
        self._status.setWordWrap(True)
        layout.addWidget(self._status)
        buttons = QtWidgets.QHBoxLayout()
        self._run_button = QtWidgets.QPushButton("Run Fit", self)
        self._pause_button = QtWidgets.QPushButton("Pause", self)
        self._cancel_button = QtWidgets.QPushButton("Cancel Fit", self)
        self._close_button = QtWidgets.QPushButton("Close", self)
        self._cancel_button.setEnabled(False)
        self._pause_button.setEnabled(False)
        self._run_button.clicked.connect(self._request_execute)
        self._cancel_button.clicked.connect(self.cancelRequested)
        self._pause_button.clicked.connect(self._toggle_pause)
        self._close_button.clicked.connect(self.close)
        buttons.addWidget(self._run_button)
        buttons.addWidget(self._pause_button)
        buttons.addWidget(self._cancel_button)
        buttons.addStretch(1)
        buttons.addWidget(self._close_button)
        layout.addLayout(buttons)

    def _build_data_tab(self) -> None:
        layout = QtWidgets.QVBoxLayout(self._data_tab)
        note = QtWidgets.QLabel(
            "t min, t max, and Points select existing observation rows only.",
            self._data_tab,
        )
        layout.addWidget(note)
        self._dataset_table = QtWidgets.QTableWidget(0, 11, self._data_tab)
        self._dataset_table.setHorizontalHeaderLabels(
            (
                "Use",
                "Dataset",
                "Mapped set",
                "t min",
                "t max",
                "Points",
                "Used",
                "Weight",
                "X",
                "X Mapping",
                "Resolved X Mapping",
            )
        )
        self._dataset_table.itemChanged.connect(self._on_dataset_item_changed)
        self._dataset_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self._dataset_table, 1)
        targets_header = QtWidgets.QHBoxLayout()
        targets_header.addWidget(QtWidgets.QLabel("Observation targets", self._data_tab))
        self._targets_all = QtWidgets.QPushButton("All", self._data_tab)
        self._targets_none = QtWidgets.QPushButton("None", self._data_tab)
        self._targets_invert = QtWidgets.QPushButton("Invert", self._data_tab)
        self._targets_copy = QtWidgets.QPushButton("Copy Targets…", self._data_tab)
        self._targets_apply = QtWidgets.QPushButton("Apply", self._data_tab)
        self._targets_revert = QtWidgets.QPushButton("Revert", self._data_tab)
        self._targets_all.clicked.connect(lambda: self._bulk_targets("all"))
        self._targets_none.clicked.connect(lambda: self._bulk_targets("none"))
        self._targets_invert.clicked.connect(lambda: self._bulk_targets("invert"))
        self._targets_copy.clicked.connect(self._copy_targets)
        self._targets_apply.clicked.connect(self._apply_target_changes)
        self._targets_revert.clicked.connect(self._revert_target_changes)
        targets_header.addWidget(self._targets_all)
        targets_header.addWidget(self._targets_none)
        targets_header.addWidget(self._targets_invert)
        targets_header.addWidget(self._targets_copy)
        targets_header.addWidget(self._targets_apply)
        targets_header.addWidget(self._targets_revert)
        targets_header.addStretch(1)
        self._add_observable_button = QtWidgets.QPushButton(
            "Add Observable…",
            self._data_tab,
        )
        self._add_observable_button.clicked.connect(
            self._show_add_observable_dialog
        )
        targets_header.addWidget(self._add_observable_button)
        layout.addLayout(targets_header)
        self._target_table = QtWidgets.QTableWidget(0, 5, self._data_tab)
        self._target_table.setHorizontalHeaderLabels(
            (
                "Use",
                "Dataset",
                "Observed column",
                "Model target",
                "Weight",
            )
        )
        self._target_table.itemChanged.connect(self._on_target_item_changed)
        self._target_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self._target_table, 1)

    def _build_parameters_tab(self) -> None:
        layout = QtWidgets.QVBoxLayout(self._parameters_tab)
        view_steps = QtWidgets.QPushButton("View Steps…", self._parameters_tab)
        view_steps.clicked.connect(self._show_current_steps)
        layout.addWidget(view_steps)
        self._parameter_table = QtWidgets.QTableWidget(
            0,
            7,
            self._parameters_tab,
        )
        self._parameter_table.setHorizontalHeaderLabels(
            ("Fit", "Parameter", "Initial", "Minimum", "Maximum", "Log10", "Scope")
        )
        self._parameter_table.itemChanged.connect(
            self._on_parameter_value_item_changed
        )
        self._parameter_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self._parameter_table, 1)
        self._parameter_local_table = QtWidgets.QTableWidget(
            0,
            3,
            self._parameters_tab,
        )
        self._parameter_local_table.setHorizontalHeaderLabels(
            ("Parameter", "Dataset", "Per-dataset value")
        )
        self._parameter_local_table.itemChanged.connect(
            self._on_parameter_local_item_changed
        )
        self._parameter_local_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self._parameter_local_table)
        optimizer = QtWidgets.QGroupBox("Optimizer", self._parameters_tab)
        form = QtWidgets.QFormLayout(optimizer)
        self._method = QtWidgets.QComboBox(optimizer)
        self._method.addItems(FITTING_METHODS)
        self._max_nfev = QtWidgets.QSpinBox(optimizer)
        self._max_nfev.setRange(*FITTING_MAX_NFEV_RANGE)
        self._ftol = QtWidgets.QDoubleSpinBox(optimizer)
        self._xtol = QtWidgets.QDoubleSpinBox(optimizer)
        for editor in (self._ftol, self._xtol):
            editor.setDecimals(15)
            editor.setRange(1e-15, 1.0)
        self._use_seed = QtWidgets.QCheckBox("Use deterministic seed", optimizer)
        self._seed = QtWidgets.QSpinBox(optimizer)
        self._seed.setRange(*FITTING_SEED_RANGE)
        self._save_project_defaults = QtWidgets.QPushButton(
            "Use as Project Defaults",
            optimizer,
        )
        self._save_project_defaults.clicked.connect(
            self._emit_project_defaults
        )
        self._ftol_label = QtWidgets.QLabel("Function tolerance", optimizer)
        self._xtol_label = QtWidgets.QLabel("Step tolerance", optimizer)
        form.addRow("Method", self._method)
        form.addRow("Maximum evaluations", self._max_nfev)
        form.addRow(self._ftol_label, self._ftol)
        form.addRow(self._xtol_label, self._xtol)
        form.addRow(self._use_seed)
        form.addRow("Seed", self._seed)
        form.addRow(self._save_project_defaults)
        layout.addWidget(optimizer)
        self._method.currentTextChanged.connect(self._on_method_changed)
        self._max_nfev.valueChanged.connect(self._emit_defaults_edit)
        self._ftol.valueChanged.connect(self._emit_defaults_edit)
        self._xtol.valueChanged.connect(self._emit_defaults_edit)
        self._use_seed.toggled.connect(self._emit_defaults_edit)
        self._seed.valueChanged.connect(self._emit_defaults_edit)
        self._update_method_controls(self._method.currentText())

    def _emit_project_defaults(self) -> None:
        self.projectDefaultsRequested.emit(
            FitDefaults(
                method=self._method.currentText(),
                max_nfev=self._max_nfev.value(),
                ftol=self._ftol.value(),
                xtol=self._xtol.value(),
                use_seed=self._use_seed.isChecked(),
                seed=self._seed.value(),
            )
        )

    def _show_current_steps(self) -> None:
        MechanismInspectorDialog(
            self._require_projected_setup().prepared_dsl,
            self,
        ).exec()

    def _build_initials_tab(self) -> None:
        layout = QtWidgets.QVBoxLayout(self._initials_tab)
        layout.addWidget(QtWidgets.QLabel("Fit-local initial concentrations", self._initials_tab))
        snapshot_buttons = QtWidgets.QHBoxLayout()
        self._view_initial = QtWidgets.QPushButton(
            "Reset to Initial",
            self._initials_tab,
        )
        self._view_last_fit = QtWidgets.QPushButton(
            "Reset to Last Fit",
            self._initials_tab,
        )
        self._view_initial.clicked.connect(lambda: self._restore_named_snapshot("initial"))
        self._view_last_fit.clicked.connect(lambda: self._restore_named_snapshot("last_fit"))
        self._view_last_fit.setEnabled(False)
        snapshot_buttons.addWidget(self._view_initial)
        snapshot_buttons.addWidget(self._view_last_fit)
        snapshot_buttons.addStretch(1)
        layout.addLayout(snapshot_buttons)
        self._initial_table = QtWidgets.QTableWidget(0, 9, self._initials_tab)
        self._initial_table.setHorizontalHeaderLabels(
            (
                "Fit",
                "Dataset",
                "Mapped set",
                "Species",
                "Initial",
                "Minimum",
                "Maximum",
                "Log10",
                "Scope",
            )
        )
        self._initial_table.itemChanged.connect(
            self._on_initial_value_item_changed
        )
        layout.addWidget(self._initial_table, 1)
        layout.addWidget(
            QtWidgets.QLabel(
                "Complete Fit-local simulation setup (outside optimizer vector)",
                self._initials_tab,
            )
        )
        self._execution_table = QtWidgets.QTableWidget(0, 6, self._initials_tab)
        self._execution_table.setHorizontalHeaderLabels(
            (
                "Dataset",
                "Mapped set",
                "Solver",
                "t start",
                "t end",
                "Integration points",
            )
        )
        self._execution_table.itemChanged.connect(
            self._on_execution_item_changed
        )
        self._execution_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self._execution_table, 1)

    def _build_results_tab(self) -> None:
        layout = QtWidgets.QVBoxLayout(self._results_tab)
        self._tracker_summary = QtWidgets.QLabel(
            "Run a fit to see the best candidate.",
            self._results_tab,
        )
        self._tracker_summary.setWordWrap(True)
        self._tracker_parameters = QtWidgets.QPlainTextEdit(self._results_tab)
        self._tracker_parameters.setReadOnly(True)
        self._tracker_parameters.setMaximumHeight(110)
        layout.addWidget(self._tracker_summary)
        layout.addWidget(self._tracker_parameters)
        self._result_views = QtWidgets.QTabWidget(self._results_tab)
        layout.addWidget(self._result_views, 1)
        self._all_datasets_view = GridPlotView(
            color_resolver=self._color_resolver,
            parent=self._result_views,
        )
        self._result_views.addTab(self._all_datasets_view, "All Datasets")
        self._result_summary = QtWidgets.QPlainTextEdit(self._results_tab)
        self._result_summary.setReadOnly(True)
        self._result_summary.setMaximumHeight(120)
        layout.addWidget(self._result_summary)
        self._species_filter = QtWidgets.QListWidget(self._results_tab)
        self._species_filter.setMaximumHeight(90)
        self._species_filter.itemChanged.connect(self._apply_species_filter)
        layout.addWidget(self._species_filter)
        scopes = QtWidgets.QHBoxLayout()
        scopes.addWidget(QtWidgets.QLabel("Apply:", self._results_tab))
        self._apply_parameters = QtWidgets.QCheckBox("Parameters", self._results_tab)
        self._apply_initials = QtWidgets.QCheckBox(
            "Initial Concentrations",
            self._results_tab,
        )
        self._apply_setup = QtWidgets.QCheckBox(
            "Simulation Setup",
            self._results_tab,
        )
        self._apply_observables = QtWidgets.QCheckBox(
            "Observable Definitions",
            self._results_tab,
        )
        self._apply_button = QtWidgets.QPushButton(
            "Apply to Project",
            self._results_tab,
        )
        self._copy_button = QtWidgets.QPushButton(
            "Copy Fit Curves",
            self._results_tab,
        )
        self._copy_run_json = QtWidgets.QPushButton(
            "Copy Run JSON",
            self._results_tab,
        )
        self._copy_button.setEnabled(False)
        self._copy_run_json.setEnabled(False)
        self._copy_button.clicked.connect(
            self._all_datasets_view.copy_all_to_clipboard
        )
        self._copy_run_json.clicked.connect(self._copy_run_identity_json)
        self._apply_button.clicked.connect(self._emit_apply)
        for widget in (
            self._apply_parameters,
            self._apply_initials,
            self._apply_setup,
            self._apply_observables,
            self._apply_button,
        ):
            widget.setEnabled(False)
        scopes.addWidget(self._apply_parameters)
        scopes.addWidget(self._apply_initials)
        scopes.addWidget(self._apply_setup)
        scopes.addWidget(self._apply_observables)
        scopes.addStretch(1)
        scopes.addWidget(self._copy_button)
        scopes.addWidget(self._copy_run_json)
        scopes.addWidget(self._apply_button)
        layout.addLayout(scopes)

    def rebase(
        self,
        origin: FitOriginSnapshot,
        *,
        setup: FitSetupSnapshot,
        preserve_snapshots: bool = False,
    ) -> None:
        if not isinstance(origin, FitOriginSnapshot):
            raise TypeError("origin must be FitOriginSnapshot.")
        if not isinstance(setup, FitSetupSnapshot):
            raise TypeError("setup must be FitSetupSnapshot.")
        if (
            setup.origin_tab_id != origin.tab_id
            or setup.origin_capture_id != origin.capture_id
        ):
            raise ValueError("Fit setup does not match the rebased origin.")
        if self._snapshot_capture_id != origin.capture_id:
            if not preserve_snapshots:
                self._initial_setup = None
                self._last_fit_setup = None
                self._view_last_fit.setEnabled(False)
            self._snapshot_capture_id = origin.capture_id
        self._origin = origin
        self._origin_label.setText(
            f"Origin: {origin.tab_name} [{origin.tab_id}] "
            f"({len(origin.mapped_datasets)} mapped datasets)"
        )
        self.setWindowTitle(
            f"Global Fit — Origin: {origin.tab_name} [{origin.tab_id}]"
        )
        self._populate_origin()
        self.project_setup_snapshot(setup)

    def install_initial_snapshot(self, snapshot: FitSetupSnapshot) -> None:
        if not isinstance(snapshot, FitSetupSnapshot):
            raise TypeError("snapshot must be FitSetupSnapshot.")
        if (
            snapshot.origin_tab_id != self._origin.tab_id
            or snapshot.origin_capture_id != self._origin.capture_id
        ):
            raise ValueError("Initial Fit snapshot does not match this origin.")
        self._initial_setup = snapshot
        self._snapshot_capture_id = snapshot.origin_capture_id

    def restore_snapshot(
        self,
        snapshot: FitSetupSnapshot,
    ) -> FitSetupSnapshot:
        """Project one owner-retained typed setup into the physical editors."""

        restoring_initial = snapshot is self._initial_setup
        restoring_last_fit = snapshot is self._last_fit_setup
        self.project_setup_snapshot(snapshot)
        if restoring_initial:
            self._initial_setup = snapshot
        if restoring_last_fit:
            self._last_fit_setup = snapshot
        return snapshot

    def project_setup_snapshot(self, snapshot: FitSetupSnapshot) -> None:
        """Render a typed owner snapshot; widgets never reconstruct this truth."""

        if not isinstance(snapshot, FitSetupSnapshot):
            raise TypeError("snapshot must be FitSetupSnapshot.")
        if (
            snapshot.origin_tab_id != self._origin.tab_id
            or snapshot.origin_capture_id != self._origin.capture_id
        ):
            raise ValueError("Fit snapshot does not match this origin.")
        projected = self._setup_projection_values(snapshot)
        tables = (
            self._dataset_table,
            self._target_table,
            self._parameter_table,
            self._parameter_local_table,
            self._initial_table,
            self._execution_table,
        )
        self.setUpdatesEnabled(False)
        self._syncing_scopes = True
        self._projecting_setup = True
        for table in tables:
            table.blockSignals(True)
        try:
            snapshot_parameter_names = {
                item.name for item in snapshot.parameters
            }
            origin_parameter_names = {
                item.name for item in self._origin.parameter_defaults
            }
            for row in reversed(range(self._parameter_table.rowCount())):
                item = self._parameter_table.item(row, 1)
                name = str(
                    item.data(QtCore.Qt.ItemDataRole.UserRole)
                    or item.text()
                )
                if (
                    name in origin_parameter_names
                    or name in snapshot_parameter_names
                ):
                    continue
                for local_row in reversed(
                    self._parameter_local_rows(name)
                ):
                    self._parameter_local_table.removeRow(local_row)
                self._parameter_table.removeRow(row)
            existing_parameter_names = {
                str(
                    self._parameter_table.item(row, 1).data(
                        QtCore.Qt.ItemDataRole.UserRole
                    )
                    or self._parameter_table.item(row, 1).text()
                )
                for row in range(self._parameter_table.rowCount())
            }
            for parameter in snapshot.parameters:
                if parameter.name not in existing_parameter_names:
                    self._append_parameter_row(parameter)
                    existing_parameter_names.add(parameter.name)
            self._render_setup_projection(
                projected,
                canonicalized_scopes=frozenset(),
            )
            self._syncing_scopes = False
            self._reconcile_parameter_local_rows()
            self._syncing_scopes = True
            self._render_parameter_projection(
                projected,
                frozenset(),
            )
            self._committed_targets = self._current_target_state()
            self._projected_setup = snapshot
            self._invalid_setup_edits.clear()
        finally:
            for table in tables:
                table.blockSignals(False)
            self._syncing_scopes = False
            self._projecting_setup = False
            self.setUpdatesEnabled(True)
        self._refresh_dataset_controls()
        self.set_resolved_auto_modes(
            tuple(
                (item.dataset_id, item.parametric.resolved_mode)
                for item in snapshot.datasets
                if item.parametric is not None
            )
        )
        self._update_method_controls(snapshot.defaults.method)

    def accept_setup_edits(
        self,
        snapshot: FitSetupSnapshot,
        intents: tuple[object, ...],
    ) -> None:
        """Acknowledge typed edits accepted by the sole setup owner."""

        if not isinstance(snapshot, FitSetupSnapshot):
            raise TypeError("snapshot must be FitSetupSnapshot.")
        if not isinstance(intents, tuple):
            raise TypeError("Fit setup acknowledgement must carry a tuple.")
        self._projected_setup = snapshot
        self._invalid_setup_edits.difference_update(
            self._intent_key(intent) for intent in intents
        )

    def reject_setup_edits(
        self,
        intents: tuple[object, ...],
        message: str,
    ) -> None:
        """Keep rejected widget text visibly invalid without changing truth."""

        if not isinstance(intents, tuple):
            raise TypeError("Fit setup rejection must carry a tuple.")
        self._invalid_setup_edits.update(
            self._intent_key(intent) for intent in intents
        )
        self.set_status(str(message))

    @property
    def has_invalid_setup_edits(self) -> bool:
        return bool(self._invalid_setup_edits)

    @staticmethod
    def _intent_key(intent: object) -> tuple[str, ...]:
        if isinstance(intent, FitDatasetSetupEdit):
            return ("dataset", intent.dataset.dataset_id)
        if isinstance(intent, FitParameterSetupEdit):
            return ("parameter", intent.parameter.name)
        if isinstance(intent, FitInitialSetupEdit):
            return ("initial", intent.initial.set_id, intent.initial.species)
        if isinstance(intent, FitDefaultsSetupEdit):
            return ("defaults",)
        return ("unknown", type(intent).__name__)

    def _setup_projection_values(
        self,
        snapshot: FitSetupSnapshot,
    ) -> dict[str, object]:
        mapped_by_id = {
            item.dataset_id: item
            for item in self._origin.mapped_datasets
        }
        datasets: list[dict[str, object]] = []
        for local in snapshot.datasets:
            mapped = mapped_by_id.get(local.dataset_id)
            if mapped is None:
                continue
            targets = {
                raw_id: model_target
                for raw_id, model_target in local.observation_targets
            }
            weights = dict(local.target_weights)
            selected = set(local.target_species)
            datasets.append(
                {
                    "dataset_id": local.dataset_id,
                    "included": local.included,
                    "sample_t_min": local.sample_t_min,
                    "sample_t_max": local.sample_t_max,
                    "sample_points": local.sample_points,
                    "weight": local.weight,
                    "execution": {
                        "set_id": mapped.set_id,
                        "solver": local.execution.solver,
                        "t_start": local.execution.t_start,
                        "t_end": local.execution.t_end,
                        "integration_points": (
                            local.execution.integration_points
                        ),
                    },
                    "targets": [
                        {
                            "use": targets.get(
                                observation.raw_observation_id,
                                observation.model_species,
                            ) in selected,
                            "raw_observation_id": (
                                observation.raw_observation_id
                            ),
                            "raw_column": observation.raw_column,
                            "model_target": targets.get(
                                observation.raw_observation_id,
                                observation.model_species,
                            ),
                            "weight": weights.get(
                                targets.get(
                                    observation.raw_observation_id,
                                    observation.model_species,
                                ),
                                1.0,
                            ),
                        }
                        for observation in mapped.observations
                    ],
                    "x_name": (
                        local.parametric.x_name
                        if local.parametric is not None
                        else "t"
                    ),
                    "x_mapping_mode": (
                        {
                            "auto": "Auto",
                            "monotone": "Monotone",
                            "time_guided": "Time-guided",
                        }[local.parametric.requested_mode]
                        if local.parametric is not None
                        else "Auto"
                    ),
                }
            )
        return {
            "datasets": datasets,
            "parameters": [
                {
                    "name": item.name,
                    "fit": item.fit,
                    "initial": item.initial,
                    "minimum": item.minimum,
                    "maximum": item.maximum,
                    "log10": item.log10,
                    "scope": (
                        "Per-dataset"
                        if item.scope == "per_dataset"
                        else "Shared"
                    ),
                    "dataset_values": [
                        {"dataset_id": dataset_id, "value": value}
                        for dataset_id, value in item.dataset_values
                    ],
                }
                for item in snapshot.parameters
            ],
            "initials": [
                {
                    "set_id": item.set_id,
                    "species": item.species,
                    "fit": item.fit,
                    "initial": item.initial,
                    "minimum": item.minimum,
                    "maximum": item.maximum,
                    "log10": item.log10,
                    "scope": (
                        "Local" if item.scope == "local" else "Global"
                    ),
                }
                for item in snapshot.initials
            ],
            "defaults": {
                "method": snapshot.defaults.method,
                "max_nfev": snapshot.defaults.max_nfev,
                "ftol": snapshot.defaults.ftol,
                "xtol": snapshot.defaults.xtol,
                "use_seed": snapshot.defaults.use_seed,
                "seed": snapshot.defaults.seed,
            },
        }

    def _populate_origin(self) -> None:
        self._refreshing_dataset_controls = True
        self._dataset_table.setRowCount(0)
        self._target_table.setRowCount(0)
        self._parameter_table.setRowCount(0)
        self._parameter_local_table.setRowCount(0)
        self._initial_table.setRowCount(0)
        self._execution_table.setRowCount(0)
        while self._result_views.count() > 1:
            widget = self._result_views.widget(1)
            self._result_views.removeTab(1)
            widget.deleteLater()
        self._dataset_view_by_id.clear()
        self._committed_targets.clear()

        for mapped in self._origin.mapped_datasets:
            times = [
                value
                for observation in mapped.observations
                for value in observation.t
            ]
            low = min(times) if times else mapped.execution.t_start
            high = max(times) if times else mapped.execution.t_end
            row = self._dataset_table.rowCount()
            self._dataset_table.insertRow(row)
            use_item = _check_item(mapped.included_at_launch)
            use_item.setData(QtCore.Qt.ItemDataRole.UserRole, mapped.dataset_id)
            self._dataset_table.setItem(row, 0, use_item)
            self._dataset_table.setItem(row, 1, _text_item(mapped.display_name, editable=False))
            self._dataset_table.setItem(row, 2, _text_item(mapped.set_name, editable=False))
            self._dataset_table.setItem(row, 3, _text_item(_float_text(low)))
            self._dataset_table.setItem(row, 4, _text_item(_float_text(high)))
            self._dataset_table.setItem(row, 5, _text_item("All"))
            self._dataset_table.setItem(
                row,
                6,
                _text_item("0", editable=False),
            )
            self._dataset_table.setItem(row, 7, _text_item("1"))
            x_selector = QtWidgets.QComboBox(self._dataset_table)
            x_selector.addItem("Time", "t")
            for model_name in dict.fromkeys(
                item.model_species for item in mapped.observations
            ):
                x_selector.addItem(model_name, model_name)
            x_selector.currentIndexChanged.connect(
                lambda _index, dataset_id=mapped.dataset_id: (
                    self._on_dataset_selector_changed(dataset_id)
                )
            )
            self._dataset_table.setCellWidget(row, 8, x_selector)
            mapping_selector = QtWidgets.QComboBox(self._dataset_table)
            mapping_selector.addItems(("Auto", "Monotone", "Time-guided"))
            mapping_selector.currentTextChanged.connect(
                lambda _text, dataset_id=mapped.dataset_id: (
                    self._on_dataset_selector_changed(dataset_id)
                )
            )
            self._dataset_table.setCellWidget(row, 9, mapping_selector)
            self._dataset_table.setItem(
                row,
                10,
                _text_item("Time", editable=False),
            )

            execution_row = self._execution_table.rowCount()
            self._execution_table.insertRow(execution_row)
            identity = _text_item(mapped.display_name, editable=False)
            identity.setData(QtCore.Qt.ItemDataRole.UserRole, mapped.dataset_id)
            identity.setData(QtCore.Qt.ItemDataRole.UserRole + 1, mapped.set_id)
            self._execution_table.setItem(execution_row, 0, identity)
            self._execution_table.setItem(execution_row, 1, _text_item(mapped.set_name, editable=False))
            solver = QtWidgets.QComboBox(self._execution_table)
            solver.addItems(SIMULATION_SOLVERS)
            solver.setCurrentText(mapped.execution.solver)
            solver.currentTextChanged.connect(
                lambda _text, dataset_id=mapped.dataset_id: (
                    self._emit_dataset_edit(dataset_id)
                )
            )
            self._execution_table.setCellWidget(execution_row, 2, solver)
            self._execution_table.setItem(execution_row, 3, _text_item(_float_text(mapped.execution.t_start)))
            self._execution_table.setItem(execution_row, 4, _text_item(_float_text(mapped.execution.t_end)))
            self._execution_table.setItem(execution_row, 5, _text_item(str(mapped.execution.integration_points)))

            view = GridPlotView(
                color_resolver=self._color_resolver,
                parent=self._result_views,
            )
            self._dataset_view_by_id[mapped.dataset_id] = view
            self._result_views.addTab(view, mapped.display_name)

            for observation in mapped.observations:
                target_row = self._target_table.rowCount()
                self._target_table.insertRow(target_row)
                use_target = _check_item(True)
                use_target.setData(QtCore.Qt.ItemDataRole.UserRole, mapped.dataset_id)
                self._target_table.setItem(target_row, 0, use_target)
                self._target_table.setItem(target_row, 1, _text_item(mapped.display_name, editable=False))
                raw_item = _text_item(observation.raw_column, editable=False)
                raw_item.setData(QtCore.Qt.ItemDataRole.UserRole, mapped.dataset_id)
                raw_item.setData(
                    QtCore.Qt.ItemDataRole.UserRole + 1,
                    observation.raw_observation_id,
                )
                self._target_table.setItem(target_row, 2, raw_item)
                self._target_table.setItem(
                    target_row,
                    3,
                    _text_item(
                        observation.model_species,
                        editable=False,
                    ),
                )
                self._target_table.setItem(target_row, 4, _text_item("1"))
                self._committed_targets[
                    (mapped.dataset_id, observation.raw_observation_id)
                ] = (True, observation.model_species, "1")

            existing_initial_keys = {
                (
                    str(self._initial_table.item(existing, 0).data(
                        QtCore.Qt.ItemDataRole.UserRole + 1
                    ) or ""),
                    self._initial_table.item(existing, 3).text(),
                )
                for existing in range(self._initial_table.rowCount())
            }
            for species, value in mapped.initial_values:
                if (mapped.set_id, species) in existing_initial_keys:
                    continue
                row = self._initial_table.rowCount()
                self._initial_table.insertRow(row)
                fit_item = _check_item(False)
                fit_item.setData(QtCore.Qt.ItemDataRole.UserRole, mapped.dataset_id)
                fit_item.setData(QtCore.Qt.ItemDataRole.UserRole + 1, mapped.set_id)
                self._initial_table.setItem(row, 0, fit_item)
                self._initial_table.setItem(row, 1, _text_item(mapped.display_name, editable=False))
                self._initial_table.setItem(row, 2, _text_item(mapped.set_name, editable=False))
                self._initial_table.setItem(row, 3, _text_item(species, editable=False))
                self._initial_table.setItem(row, 4, _text_item(_float_text(value)))
                self._initial_table.setItem(row, 5, _text_item("0"))
                self._initial_table.setItem(row, 6, _text_item(_float_text(max(1.0, value * 10.0))))
                self._initial_table.setItem(row, 7, _check_item(False))
                scope = QtWidgets.QComboBox(self._initial_table)
                scope.addItems(("Global", "Local"))
                scope.currentTextChanged.connect(
                    lambda text, species_name=species: self._on_initial_scope_changed(
                        species_name,
                        text,
                    )
                )
                self._initial_table.setCellWidget(row, 8, scope)
                existing_initial_keys.add((mapped.set_id, species))

        included_set_ids = self._included_set_ids()
        for row in range(self._initial_table.rowCount()):
            set_id = str(
                self._initial_table.item(row, 0).data(
                    QtCore.Qt.ItemDataRole.UserRole + 1
                )
                or ""
            )
            if set_id in included_set_ids:
                continue
            selector = self._initial_table.cellWidget(row, 8)
            if isinstance(selector, QtWidgets.QComboBox):
                blocker = QtCore.QSignalBlocker(selector)
                selector.setCurrentText("Local")
                del blocker

        for species in {
            self._initial_table.item(row, 3).text()
            for row in range(self._initial_table.rowCount())
        }:
            rows = self._initial_rows_for_species(species)
            active_rows = tuple(
                row
                for row in rows
                if str(
                    self._initial_table.item(row, 0).data(
                        QtCore.Qt.ItemDataRole.UserRole + 1
                    )
                    or ""
                )
                in included_set_ids
            )
            values = {
                _read_float(
                    self._initial_table.item(row, 4).text(),
                    f"{species} initial value",
                )
                for row in active_rows
            }
            if active_rows and len(values) == 1:
                self._on_initial_scope_changed(species, "Global")
                continue
            self._syncing_scopes = True
            try:
                for row in active_rows:
                    selector = self._initial_table.cellWidget(row, 8)
                    if isinstance(selector, QtWidgets.QComboBox):
                        selector.setCurrentText("Local")
            finally:
                self._syncing_scopes = False

        for parameter in self._origin.parameter_defaults:
            self._append_parameter_row(parameter)
        self._refreshing_dataset_controls = False
        self._refresh_dataset_controls()
        self._committed_targets = self._current_target_state()

    def _append_parameter_row(
        self,
        parameter: FitParameterVariable,
    ) -> None:
        row = self._parameter_table.rowCount()
        self._parameter_table.insertRow(row)
        self._parameter_table.setItem(row, 0, _check_item(parameter.fit))
        parameter_item = _text_item(
            parameter.display_name,
            editable=False,
        )
        parameter_item.setData(
            QtCore.Qt.ItemDataRole.UserRole,
            parameter.name,
        )
        self._parameter_table.setItem(row, 1, parameter_item)
        self._parameter_table.setItem(
            row,
            2,
            _text_item(_float_text(parameter.initial)),
        )
        self._parameter_table.setItem(
            row,
            3,
            _text_item(_float_text(parameter.minimum)),
        )
        self._parameter_table.setItem(
            row,
            4,
            _text_item(_float_text(parameter.maximum)),
        )
        self._parameter_table.setItem(row, 5, _check_item(parameter.log10))
        scope = QtWidgets.QComboBox(self._parameter_table)
        scope.addItems(("Shared", "Per-dataset"))
        scope.currentTextChanged.connect(
            lambda text, parameter_name=parameter.name: (
                self._on_parameter_scope_changed(
                    parameter_name,
                    text,
                )
            )
        )
        self._parameter_table.setCellWidget(row, 6, scope)
        scope.setCurrentText(
            "Per-dataset"
            if parameter.scope == "per_dataset"
            else "Shared"
        )

    def _load_defaults(self) -> None:
        defaults = self._origin.fit_defaults
        self._method.setCurrentText(defaults.method)
        self._max_nfev.setValue(defaults.max_nfev)
        self._ftol.setValue(defaults.ftol)
        self._xtol.setValue(defaults.xtol)
        self._use_seed.setChecked(defaults.use_seed)
        self._seed.setValue(defaults.seed)

    def _initial_rows_for_species(self, species: str) -> tuple[int, ...]:
        return tuple(
            row
            for row in range(self._initial_table.rowCount())
            if self._initial_table.item(row, 3).text() == species
        )

    def _parameter_row(self, name: str) -> int:
        for row in range(self._parameter_table.rowCount()):
            item = self._parameter_table.item(row, 1)
            identity = str(
                item.data(QtCore.Qt.ItemDataRole.UserRole)
                or item.text()
            )
            if identity == name:
                return row
        return -1

    def _parameter_local_rows(self, name: str) -> tuple[int, ...]:
        return tuple(
            row
            for row in range(self._parameter_local_table.rowCount())
            if str(
                self._parameter_local_table.item(row, 0).data(
                    QtCore.Qt.ItemDataRole.UserRole
                )
                or self._parameter_local_table.item(row, 0).text()
            )
            == name
        )

    def _included_dataset_ids(self) -> tuple[str, ...]:
        return tuple(
            str(
                self._dataset_table.item(row, 0).data(
                    QtCore.Qt.ItemDataRole.UserRole
                )
                or ""
            )
            for row in range(self._dataset_table.rowCount())
            if (
                self._dataset_table.item(row, 0).checkState()
                == QtCore.Qt.CheckState.Checked
            )
        )

    def _included_set_ids(self) -> frozenset[str]:
        included = set(self._included_dataset_ids())
        return frozenset(
            mapped.set_id
            for mapped in self._origin.mapped_datasets
            if mapped.dataset_id in included
        )

    def _reconcile_parameter_local_rows(
        self,
        names: Iterable[str] | None = None,
    ) -> None:
        """Retain local scalar rows for every still-existing dataset identity."""

        if self._syncing_scopes:
            return
        requested_names = (
            None
            if names is None
            else {str(name) for name in names}
        )
        retained_ids = tuple(
            mapped.dataset_id for mapped in self._origin.mapped_datasets
        )
        display_by_id = {
            mapped.dataset_id: mapped.display_name
            for mapped in self._origin.mapped_datasets
        }
        existing: dict[tuple[str, str], list[str]] = {}
        names_with_rows: set[str] = set()
        for local_row in range(self._parameter_local_table.rowCount()):
            parameter_item = self._parameter_local_table.item(local_row, 0)
            value_item = self._parameter_local_table.item(local_row, 2)
            name = str(
                parameter_item.data(QtCore.Qt.ItemDataRole.UserRole)
                or parameter_item.text()
            )
            if requested_names is not None and name not in requested_names:
                continue
            dataset_id = str(
                value_item.data(QtCore.Qt.ItemDataRole.UserRole)
                or ""
            )
            names_with_rows.add(name)
            existing.setdefault((name, dataset_id), []).append(
                value_item.text()
            )

        parameter_rows: list[tuple[int, str]] = []
        for row in range(self._parameter_table.rowCount()):
            item = self._parameter_table.item(row, 1)
            name = str(
                item.data(QtCore.Qt.ItemDataRole.UserRole)
                or item.text()
            )
            if requested_names is not None and name not in requested_names:
                continue
            selector = self._parameter_table.cellWidget(row, 6)
            if (
                name in names_with_rows
                or (
                    isinstance(selector, QtWidgets.QComboBox)
                    and selector.currentText() == "Per-dataset"
                )
            ):
                parameter_rows.append((row, name))
        target_names = {name for _row, name in parameter_rows}
        if not target_names and requested_names is not None:
            target_names = set(requested_names)

        self._syncing_scopes = True
        try:
            for local_row in reversed(
                range(self._parameter_local_table.rowCount())
            ):
                item = self._parameter_local_table.item(local_row, 0)
                name = str(
                    item.data(QtCore.Qt.ItemDataRole.UserRole)
                    or item.text()
                )
                if name in target_names:
                    self._parameter_local_table.removeRow(local_row)
            for parameter_row, name in parameter_rows:
                shared_item = self._parameter_table.item(
                    parameter_row,
                    2,
                )
                shared_text = shared_item.text()
                try:
                    _read_float(shared_text, f"{name} shared value")
                except ValueError:
                    remembered = shared_item.data(
                        _PARAMETER_SHARED_BASELINE_ROLE
                    )
                    if remembered is not None:
                        shared_text = _float_text(
                            _read_float(
                                remembered,
                                f"{name} remembered shared value",
                            )
                        )
                display = self._parameter_table.item(
                    parameter_row,
                    1,
                ).text()
                for dataset_id in retained_ids:
                    prior_values = existing.get((name, dataset_id), ())
                    value_text = shared_text
                    if prior_values:
                        try:
                            numeric_values = {
                                _read_float(
                                    value,
                                    f"{name} per-dataset value",
                                )
                                for value in prior_values
                            }
                        except ValueError:
                            numeric_values = set()
                        value_text = (
                            _float_text(numeric_values.pop())
                            if len(numeric_values) == 1
                            else "Multiple values"
                        )
                    local_row = self._parameter_local_table.rowCount()
                    self._parameter_local_table.insertRow(local_row)
                    parameter_item = _text_item(
                        display,
                        editable=False,
                    )
                    parameter_item.setData(
                        QtCore.Qt.ItemDataRole.UserRole,
                        name,
                    )
                    self._parameter_local_table.setItem(
                        local_row,
                        0,
                        parameter_item,
                    )
                    self._parameter_local_table.setItem(
                        local_row,
                        1,
                        _text_item(
                            display_by_id.get(dataset_id, dataset_id),
                            editable=False,
                        ),
                    )
                    value_item = _text_item(value_text)
                    value_item.setData(
                        QtCore.Qt.ItemDataRole.UserRole,
                        dataset_id,
                    )
                    self._parameter_local_table.setItem(
                        local_row,
                        2,
                        value_item,
                    )
        finally:
            self._syncing_scopes = False

    def _project_shared_parameter_from_local_rows(
        self,
        name: str,
    ) -> None:
        row = self._parameter_row(name)
        if row < 0:
            return
        selector = self._parameter_table.cellWidget(row, 6)
        if (
            not isinstance(selector, QtWidgets.QComboBox)
            or selector.currentText() != "Shared"
        ):
            return
        local_rows = self._parameter_local_rows(name)
        included_ids = set(self._included_dataset_ids())
        values = [
            self._parameter_local_table.item(local_row, 2).text()
            for local_row in local_rows
            if str(
                self._parameter_local_table.item(local_row, 2).data(
                    QtCore.Qt.ItemDataRole.UserRole
                )
                or ""
            )
            in included_ids
        ]
        try:
            parsed_values = {
                _read_float(
                    value,
                    f"{name} per-dataset value",
                )
                for value in values
            }
        except ValueError:
            parsed_values = set()
        item = self._parameter_table.item(row, 2)
        self._syncing_scopes = True
        try:
            if len(parsed_values) == 1:
                value = parsed_values.pop()
                item.setText(_float_text(value))
                item.setData(
                    _PARAMETER_SHARED_BASELINE_ROLE,
                    value,
                )
            elif values:
                try:
                    prior = _read_float(
                        item.text(),
                        f"{name} shared value",
                    )
                except ValueError:
                    pass
                else:
                    item.setData(
                        _PARAMETER_SHARED_BASELINE_ROLE,
                        prior,
                    )
                item.setText("Multiple values")
        finally:
            self._syncing_scopes = False

    def _on_parameter_scope_changed(self, name: str, text: str) -> None:
        if (
            self._syncing_scopes
            or self._projecting_setup
            or self._refreshing_dataset_controls
        ):
            return
        row = self._parameter_row(name)
        if row < 0:
            return
        shared_item = self._parameter_table.item(row, 2)
        try:
            shared_value = _read_float(
                shared_item.text(),
                f"{name} shared value",
            )
        except ValueError:
            pass
        else:
            shared_item.setData(
                _PARAMETER_SHARED_BASELINE_ROLE,
                shared_value,
            )
        self._reconcile_parameter_local_rows((name,))
        if str(text).lower() != "per-dataset":
            self._project_shared_parameter_from_local_rows(name)
        self._emit_parameter_edit(name)

    def _on_parameter_value_item_changed(
        self,
        item: QtWidgets.QTableWidgetItem,
    ) -> None:
        if (
            self._syncing_scopes
            or self._projecting_setup
            or self._refreshing_dataset_controls
            or item.column() not in {0, 2, 3, 4, 5}
        ):
            return
        name_item = self._parameter_table.item(item.row(), 1)
        name = str(
            name_item.data(QtCore.Qt.ItemDataRole.UserRole)
            or name_item.text()
        )
        selector = self._parameter_table.cellWidget(item.row(), 6)
        if (
            item.column() != 2
            or not isinstance(selector, QtWidgets.QComboBox)
            or selector.currentText() != "Shared"
        ):
            self._emit_parameter_edit(name)
            return
        try:
            value = _read_float(
                item.text(),
                "shared parameter value",
            )
        except ValueError:
            self._emit_parameter_edit(name)
            return
        item.setData(
            _PARAMETER_SHARED_BASELINE_ROLE,
            value,
        )
        self._syncing_scopes = True
        try:
            included_ids = set(self._included_dataset_ids())
            for row in self._parameter_local_rows(name):
                local_item = self._parameter_local_table.item(row, 2)
                if str(
                    local_item.data(QtCore.Qt.ItemDataRole.UserRole)
                    or ""
                ) in included_ids:
                    local_item.setText(item.text())
        finally:
            self._syncing_scopes = False
        self._emit_parameter_edit(name)

    def _on_parameter_local_item_changed(
        self,
        item: QtWidgets.QTableWidgetItem,
    ) -> None:
        if (
            self._syncing_scopes
            or self._projecting_setup
            or self._refreshing_dataset_controls
            or item.column() != 2
        ):
            return
        name_item = self._parameter_local_table.item(item.row(), 0)
        name = str(
            name_item.data(QtCore.Qt.ItemDataRole.UserRole)
            or name_item.text()
        )
        self._project_shared_parameter_from_local_rows(name)
        self._emit_parameter_edit(name)

    def _on_initial_scope_changed(self, species: str, text: str) -> None:
        if (
            self._syncing_scopes
            or self._projecting_setup
            or self._refreshing_dataset_controls
        ):
            return
        rows = self._initial_rows_for_species(species)
        if not rows:
            return
        included_set_ids = self._included_set_ids()
        sender = self.sender()
        if isinstance(sender, QtWidgets.QComboBox):
            source_row = next(
                (
                    row
                    for row in rows
                    if self._initial_table.cellWidget(row, 8) is sender
                ),
                None,
            )
            source_set_id = (
                ""
                if source_row is None
                else str(
                    self._initial_table.item(source_row, 0).data(
                        QtCore.Qt.ItemDataRole.UserRole + 1
                    )
                    or ""
                )
            )
            if source_set_id not in included_set_ids:
                blocker = QtCore.QSignalBlocker(sender)
                sender.setCurrentText("Local")
                del blocker
                if source_row is not None:
                    self._emit_initial_edit(source_row)
                return
        active_rows = [
            row
            for row in rows
            if str(
                self._initial_table.item(row, 0).data(
                    QtCore.Qt.ItemDataRole.UserRole + 1
                )
                or ""
            )
            in included_set_ids
        ]
        if not active_rows:
            return
        self._syncing_scopes = True
        try:
            scope_text = "Local" if str(text).lower() == "local" else "Global"
            for row in active_rows:
                selector = self._initial_table.cellWidget(row, 8)
                if isinstance(selector, QtWidgets.QComboBox):
                    selector.blockSignals(True)
                    selector.setCurrentText(scope_text)
                    selector.blockSignals(False)
            if scope_text == "Local":
                pending = any(
                    self._initial_table.item(row, 4).text() == "Multiple values"
                    for row in active_rows
                )
                if pending:
                    for row in active_rows:
                        item = self._initial_table.item(row, 4)
                        saved = item.data(QtCore.Qt.ItemDataRole.UserRole + 10)
                        if saved is not None:
                            item.setText(str(saved))
                self._emit_initial_edits(tuple(active_rows))
                return
            values = [
                self._initial_table.item(row, 4).text()
                for row in active_rows
            ]
            try:
                parsed_values = {
                    _read_float(
                        value,
                        f"{species} initial value",
                    )
                    for value in values
                }
            except ValueError:
                parsed_values = set()
            if len(parsed_values) == 1:
                value = _float_text(parsed_values.pop())
                for row in active_rows:
                    self._initial_table.item(row, 4).setText(value)
                self._emit_initial_edits(tuple(active_rows))
                return
            for row, value in zip(active_rows, values):
                item = self._initial_table.item(row, 4)
                item.setData(QtCore.Qt.ItemDataRole.UserRole + 10, value)
                item.setText("Multiple values")
        finally:
            self._syncing_scopes = False
        self._emit_initial_edits(tuple(active_rows))

    def _on_initial_value_item_changed(
        self,
        item: QtWidgets.QTableWidgetItem,
    ) -> None:
        if (
            self._syncing_scopes
            or self._projecting_setup
            or self._refreshing_dataset_controls
            or item.column() not in {0, 4, 5, 6, 7}
        ):
            return
        row = item.row()
        selector = self._initial_table.cellWidget(row, 8)
        if (
            not isinstance(selector, QtWidgets.QComboBox)
            or selector.currentText() != "Global"
        ):
            self._emit_initial_edit(row)
            return
        included_set_ids = self._included_set_ids()
        source_set_id = str(
            self._initial_table.item(row, 0).data(
                QtCore.Qt.ItemDataRole.UserRole + 1
            )
            or ""
        )
        if source_set_id not in included_set_ids:
            self._emit_initial_edit(row)
            return
        if item.column() in {4, 5, 6}:
            try:
                float(item.text())
            except ValueError:
                self._emit_initial_edit(row)
                return
        species = self._initial_table.item(row, 3).text()
        self._syncing_scopes = True
        try:
            for target_row in self._initial_rows_for_species(species):
                target_set_id = str(
                    self._initial_table.item(target_row, 0).data(
                        QtCore.Qt.ItemDataRole.UserRole + 1
                    )
                    or ""
                )
                if target_set_id not in included_set_ids:
                    continue
                target = self._initial_table.item(
                    target_row,
                    item.column(),
                )
                if item.column() in {0, 7}:
                    target.setCheckState(item.checkState())
                else:
                    target.setText(item.text())
                if item.column() == 4:
                    target.setData(
                        QtCore.Qt.ItemDataRole.UserRole + 10,
                        None,
                    )
        finally:
            self._syncing_scopes = False
        self._emit_initial_edits(
            tuple(
                target_row
                for target_row in self._initial_rows_for_species(species)
                if str(
                    self._initial_table.item(target_row, 0).data(
                        QtCore.Qt.ItemDataRole.UserRole + 1
                    )
                    or ""
                )
                in included_set_ids
            )
        )

    def _require_projected_setup(self) -> FitSetupSnapshot:
        setup = self._projected_setup
        if setup is None:
            raise RuntimeError("The owner has not installed a typed Fit setup.")
        return setup

    def _dataset_projection(
        self,
        dataset_id: str,
        *,
        include_pending_targets: bool,
    ) -> FitDatasetLocalSettings:
        setup = self._require_projected_setup()
        current = next(
            item for item in setup.datasets if item.dataset_id == dataset_id
        )
        dataset_row = next(
            row
            for row in range(self._dataset_table.rowCount())
            if str(
                self._dataset_table.item(row, 0).data(
                    QtCore.Qt.ItemDataRole.UserRole
                )
                or ""
            )
            == dataset_id
        )
        execution_row = next(
            row
            for row in range(self._execution_table.rowCount())
            if str(
                self._execution_table.item(row, 0).data(
                    QtCore.Qt.ItemDataRole.UserRole
                )
                or ""
            )
            == dataset_id
        )
        solver = self._execution_table.cellWidget(execution_row, 2)
        x_selector = self._dataset_table.cellWidget(dataset_row, 8)
        mapping_selector = self._dataset_table.cellWidget(dataset_row, 9)
        points_text = self._dataset_table.item(dataset_row, 5).text().strip()
        target_species = current.target_species
        target_weights = current.target_weights
        observation_targets = current.observation_targets
        if include_pending_targets:
            selected: list[tuple[str, str, float]] = []
            for row in range(self._target_table.rowCount()):
                use = self._target_table.item(row, 0)
                if (
                    str(use.data(QtCore.Qt.ItemDataRole.UserRole) or "")
                    != dataset_id
                    or use.checkState() != QtCore.Qt.CheckState.Checked
                ):
                    continue
                selected.append(
                    (
                        str(
                            self._target_table.item(row, 2).data(
                                QtCore.Qt.ItemDataRole.UserRole + 1
                            )
                            or ""
                        ),
                        self._target_table.item(row, 3).text().strip(),
                        _read_float(
                            self._target_table.item(row, 4).text(),
                            "target weight",
                        ),
                    )
                )
            target_species = tuple(model for _raw_id, model, _weight in selected)
            target_weights = tuple(
                (model, weight) for _raw_id, model, weight in selected
            )
            observation_targets = tuple(
                (raw_id, model) for raw_id, model, _weight in selected
            )
        x_name = (
            str(x_selector.currentData() or "t")
            if isinstance(x_selector, QtWidgets.QComboBox)
            else "t"
        )
        mode_text = (
            str(mapping_selector.currentText() or "Auto").strip().lower()
            if isinstance(mapping_selector, QtWidgets.QComboBox)
            else "auto"
        )
        requested_mode = {
            "auto": "auto",
            "monotone": "monotone",
            "time-guided": "time_guided",
        }[mode_text]
        return FitDatasetLocalSettings(
            dataset_id=dataset_id,
            included=(
                self._dataset_table.item(dataset_row, 0).checkState()
                == QtCore.Qt.CheckState.Checked
            ),
            sample_t_min=_read_float(
                self._dataset_table.item(dataset_row, 3).text(),
                "sampling t min",
            ),
            sample_t_max=_read_float(
                self._dataset_table.item(dataset_row, 4).text(),
                "sampling t max",
            ),
            sample_points=(
                None if points_text.casefold() == "all" else int(points_text)
            ),
            execution=SimulationExecutionSettings(
                solver=(
                    solver.currentText()
                    if isinstance(solver, QtWidgets.QComboBox)
                    else ""
                ),
                t_start=_read_float(
                    self._execution_table.item(execution_row, 3).text(),
                    "t start",
                ),
                t_end=_read_float(
                    self._execution_table.item(execution_row, 4).text(),
                    "t end",
                ),
                integration_points=int(
                    self._execution_table.item(execution_row, 5).text()
                ),
            ),
            weight=_read_float(
                self._dataset_table.item(dataset_row, 7).text(),
                "dataset weight",
            ),
            target_species=target_species,
            target_weights=target_weights,
            observation_targets=observation_targets,
            parametric=(
                None
                if x_name == "t"
                else FitParametricSettings(
                    dataset_id=dataset_id,
                    x_name=x_name,
                    requested_mode=requested_mode,
                    resolved_mode=(
                        "time_guided"
                        if requested_mode == "time_guided"
                        else "monotone"
                    ),
                )
            ),
        )

    def _parameter_projection(self, name: str) -> FitParameterVariable:
        setup = self._require_projected_setup()
        current = next(item for item in setup.parameters if item.name == name)
        row = self._parameter_row(name)
        if row < 0:
            raise KeyError(f"Parameter {name!r} has no projection row.")
        scope = self._parameter_table.cellWidget(row, 6)
        per_dataset = (
            isinstance(scope, QtWidgets.QComboBox)
            and scope.currentText() == "Per-dataset"
        )
        return replace(
            current,
            initial=(
                current.initial
                if per_dataset
                else _read_float(
                    self._parameter_table.item(row, 2).text(),
                    f"{name} initial",
                )
            ),
            minimum=float(self._parameter_table.item(row, 3).text()),
            maximum=float(self._parameter_table.item(row, 4).text()),
            fit=(
                self._parameter_table.item(row, 0).checkState()
                == QtCore.Qt.CheckState.Checked
            ),
            log10=(
                self._parameter_table.item(row, 5).checkState()
                == QtCore.Qt.CheckState.Checked
            ),
            scope="per_dataset" if per_dataset else "shared",
            dataset_values=tuple(
                (
                    str(
                        self._parameter_local_table.item(local_row, 2).data(
                            QtCore.Qt.ItemDataRole.UserRole
                        )
                        or ""
                    ),
                    _read_float(
                        self._parameter_local_table.item(local_row, 2).text(),
                        f"{name} per-dataset value",
                    ),
                )
                for local_row in self._parameter_local_rows(name)
                if per_dataset
            ),
        )

    def _initial_projection(self, row: int) -> FitInitialVariable:
        setup = self._require_projected_setup()
        set_id = str(
            self._initial_table.item(row, 0).data(
                QtCore.Qt.ItemDataRole.UserRole + 1
            )
            or ""
        )
        species = self._initial_table.item(row, 3).text()
        current = next(
            item
            for item in setup.initials
            if item.set_id == set_id and item.species == species
        )
        scope = self._initial_table.cellWidget(row, 8)
        return replace(
            current,
            initial=_read_float(
                self._initial_table.item(row, 4).text(),
                f"{species} initial",
            ),
            minimum=_read_float(
                self._initial_table.item(row, 5).text(),
                f"{species} minimum",
            ),
            maximum=_read_float(
                self._initial_table.item(row, 6).text(),
                f"{species} maximum",
            ),
            fit=(
                self._initial_table.item(row, 0).checkState()
                == QtCore.Qt.CheckState.Checked
            ),
            log10=(
                self._initial_table.item(row, 7).checkState()
                == QtCore.Qt.CheckState.Checked
            ),
            scope=(
                "local"
                if isinstance(scope, QtWidgets.QComboBox)
                and scope.currentText() == "Local"
                else "global"
            ),
        )

    def _defaults_projection(self) -> FitDefaults:
        return FitDefaults(
            method=self._method.currentText(),
            max_nfev=self._max_nfev.value(),
            ftol=self._ftol.value(),
            xtol=self._xtol.value(),
            use_seed=self._use_seed.isChecked(),
            seed=self._seed.value(),
        )

    def _record_invalid_edit(
        self,
        key: tuple[str, ...],
        error: BaseException,
    ) -> None:
        self._invalid_setup_edits.add(key)
        self.set_status(f"Fit setup edit is invalid: {error}")

    def _emit_dataset_edit(
        self,
        dataset_id: str,
        *,
        include_pending_targets: bool = False,
        raw_observation_id: str | None = None,
    ) -> bool:
        if raw_observation_id is not None:
            if self._projecting_setup or self._projected_setup is None:
                return False
            key = ("dataset", str(dataset_id))
            try:
                intent = FitDatasetSetupEdit(
                    origin_tab_id=self._origin.tab_id,
                    origin_capture_id=self._origin.capture_id,
                    dataset=self._dataset_projection(
                        str(dataset_id),
                        include_pending_targets=include_pending_targets,
                    ),
                    raw_observation_id=raw_observation_id,
                )
            except (
                KeyError,
                StopIteration,
                TypeError,
                ValueError,
                OverflowError,
            ) as exc:
                self._record_invalid_edit(key, exc)
                return False
            self.setupEditsRequested.emit((intent,))
            return key not in self._invalid_setup_edits
        return self._emit_dataset_edits(
            (str(dataset_id),),
            include_pending_targets=include_pending_targets,
        )

    def _emit_dataset_edits(
        self,
        dataset_ids: tuple[str, ...],
        *,
        include_pending_targets: bool = False,
    ) -> bool:
        if self._projecting_setup or self._projected_setup is None:
            return False
        intents: list[object] = []
        keys: list[tuple[str, ...]] = []
        for dataset_id in dataset_ids:
            key = ("dataset", str(dataset_id))
            try:
                intent = FitDatasetSetupEdit(
                    origin_tab_id=self._origin.tab_id,
                    origin_capture_id=self._origin.capture_id,
                    dataset=self._dataset_projection(
                        str(dataset_id),
                        include_pending_targets=include_pending_targets,
                    ),
                )
            except (
                KeyError,
                StopIteration,
                TypeError,
                ValueError,
                OverflowError,
            ) as exc:
                self._record_invalid_edit(key, exc)
                return False
            intents.append(intent)
            keys.append(key)
        self.setupEditsRequested.emit(tuple(intents))
        return not any(key in self._invalid_setup_edits for key in keys)

    def _emit_parameter_edit(self, name: str) -> bool:
        if self._projecting_setup or self._projected_setup is None:
            return False
        key = ("parameter", str(name))
        try:
            intent = FitParameterSetupEdit(
                origin_tab_id=self._origin.tab_id,
                origin_capture_id=self._origin.capture_id,
                parameter=self._parameter_projection(str(name)),
            )
        except (KeyError, StopIteration, TypeError, ValueError, OverflowError) as exc:
            self._record_invalid_edit(key, exc)
            return False
        self.setupEditsRequested.emit((intent,))
        return key not in self._invalid_setup_edits

    def _emit_initial_edit(self, row: int) -> bool:
        return self._emit_initial_edits((row,))

    def _emit_initial_edits(self, rows: tuple[int, ...]) -> bool:
        if self._projecting_setup or self._projected_setup is None:
            return False
        intents: list[object] = []
        keys: list[tuple[str, ...]] = []
        for row in rows:
            set_id = str(
                self._initial_table.item(row, 0).data(
                    QtCore.Qt.ItemDataRole.UserRole + 1
                )
                or ""
            )
            species = self._initial_table.item(row, 3).text()
            key = ("initial", set_id, species)
            try:
                intent = FitInitialSetupEdit(
                    origin_tab_id=self._origin.tab_id,
                    origin_capture_id=self._origin.capture_id,
                    initial=self._initial_projection(row),
                )
            except (
                KeyError,
                StopIteration,
                TypeError,
                ValueError,
                OverflowError,
            ) as exc:
                self._record_invalid_edit(key, exc)
                return False
            intents.append(intent)
            keys.append(key)
        self.setupEditsRequested.emit(tuple(intents))
        return not any(key in self._invalid_setup_edits for key in keys)

    def _emit_defaults_edit(self, *_args: object) -> bool:
        if self._projecting_setup or self._projected_setup is None:
            return False
        key = ("defaults",)
        try:
            intent = FitDefaultsSetupEdit(
                origin_tab_id=self._origin.tab_id,
                origin_capture_id=self._origin.capture_id,
                defaults=self._defaults_projection(),
            )
        except (TypeError, ValueError, OverflowError) as exc:
            self._record_invalid_edit(key, exc)
            return False
        self.setupEditsRequested.emit((intent,))
        return key not in self._invalid_setup_edits

    def _on_method_changed(self, method: str) -> None:
        self._update_method_controls(method)
        self._emit_defaults_edit()

    def _update_method_controls(self, method: str) -> None:
        is_de = str(method) == DIFFERENTIAL_EVOLUTION_METHOD
        self._ftol_label.setText(
            DE_POPULATION_TOLERANCE_LABEL
            if is_de
            else FUNCTION_TOLERANCE_LABEL
        )
        self._ftol.setToolTip(
            "Passed directly to SciPy differential_evolution(tol=...)."
            if is_de
            else "Optimizer function-value convergence tolerance."
        )
        self._xtol_label.setVisible(not is_de)
        self._xtol.setVisible(not is_de)
        self._xtol.setEnabled(not is_de)

    def _render_setup_projection(
        self,
        preserved: Mapping[str, object],
        *,
        canonicalized_scopes: frozenset[str],
    ) -> None:
        datasets = {
            str(item.get("dataset_id")): item
            for item in list(preserved.get("datasets") or ())
            if isinstance(item, Mapping)
        }
        for row in range(self._dataset_table.rowCount()):
            identity = self._dataset_table.item(row, 0)
            dataset_id = str(identity.data(QtCore.Qt.ItemDataRole.UserRole) or "")
            prior = datasets.get(dataset_id)
            if prior is None:
                continue
            identity.setCheckState(
                QtCore.Qt.CheckState.Checked
                if bool(prior.get("included"))
                else QtCore.Qt.CheckState.Unchecked
            )
            for column, key in (
                (3, "sample_t_min"),
                (4, "sample_t_max"),
                (7, "weight"),
            ):
                self._dataset_table.item(row, column).setText(str(prior.get(key)))
            self._dataset_table.item(row, 5).setText(
                "All" if prior.get("sample_points") is None else str(prior.get("sample_points"))
            )
            x_selector = self._dataset_table.cellWidget(row, 8)
            if isinstance(x_selector, QtWidgets.QComboBox):
                index = x_selector.findData(str(prior.get("x_name") or "t"))
                if index >= 0:
                    blocker = QtCore.QSignalBlocker(x_selector)
                    x_selector.setCurrentIndex(index)
                    del blocker
            mapping_selector = self._dataset_table.cellWidget(row, 9)
            if isinstance(mapping_selector, QtWidgets.QComboBox):
                blocker = QtCore.QSignalBlocker(mapping_selector)
                mapping_selector.setCurrentText(
                    str(prior.get("x_mapping_mode") or "Auto")
                )
                del blocker
        target_lookup = {
            (
                str(item.get("dataset_id")),
                str(target.get("raw_observation_id")),
            ): target
            for item in list(preserved.get("datasets") or ())
            if isinstance(item, Mapping)
            for target in list(item.get("targets") or ())
            if isinstance(target, Mapping)
        }
        for row in range(self._target_table.rowCount()):
            use = self._target_table.item(row, 0)
            key = (
                str(use.data(QtCore.Qt.ItemDataRole.UserRole) or ""),
                str(
                    self._target_table.item(row, 2).data(
                        QtCore.Qt.ItemDataRole.UserRole + 1
                    )
                    or ""
                ),
            )
            prior = target_lookup.get(key)
            if prior is None:
                continue
            use.setCheckState(
                QtCore.Qt.CheckState.Checked
                if bool(prior.get("use"))
                else QtCore.Qt.CheckState.Unchecked
            )
            self._target_table.item(row, 3).setText(str(prior.get("model_target")))
            self._target_table.item(row, 4).setText(str(prior.get("weight")))
        self._render_parameter_projection(preserved, canonicalized_scopes)
        self._render_initial_projection(preserved, canonicalized_scopes)
        self._render_execution_projection(datasets, canonicalized_scopes)
        defaults = preserved.get("defaults")
        if isinstance(defaults, Mapping):
            self._method.setCurrentText(str(defaults.get("method")))
            self._max_nfev.setValue(int(defaults.get("max_nfev")))
            self._ftol.setValue(float(defaults.get("ftol")))
            self._xtol.setValue(float(defaults.get("xtol")))
            self._use_seed.setChecked(bool(defaults.get("use_seed")))
            self._seed.setValue(int(defaults.get("seed")))
        self._refresh_dataset_controls()

    def _render_parameter_projection(
        self,
        preserved: Mapping[str, object],
        canonicalized_scopes: frozenset[str],
    ) -> None:
        previous = {
            str(item.get("name")): item
            for item in list(preserved.get("parameters") or ())
            if isinstance(item, Mapping)
        }
        for row in range(self._parameter_table.rowCount()):
            name_item = self._parameter_table.item(row, 1)
            name = str(
                name_item.data(QtCore.Qt.ItemDataRole.UserRole)
                or name_item.text()
            )
            prior = previous.get(name)
            if prior is None:
                continue
            self._parameter_table.item(row, 0).setCheckState(
                QtCore.Qt.CheckState.Checked if bool(prior.get("fit")) else QtCore.Qt.CheckState.Unchecked
            )
            for column, key in ((3, "minimum"), (4, "maximum")):
                self._parameter_table.item(row, column).setText(str(prior.get(key)))
            if "parameters" not in canonicalized_scopes:
                self._parameter_table.item(row, 2).setText(str(prior.get("initial")))
            self._parameter_table.item(row, 5).setCheckState(
                QtCore.Qt.CheckState.Checked if bool(prior.get("log10")) else QtCore.Qt.CheckState.Unchecked
            )
            scope = self._parameter_table.cellWidget(row, 6)
            if isinstance(scope, QtWidgets.QComboBox):
                scope.setCurrentText(str(prior.get("scope") or "Shared"))
            local_values = {
                str(item.get("dataset_id")): str(item.get("value"))
                for item in list(prior.get("dataset_values") or ())
                if isinstance(item, Mapping)
            }
            for local_row in self._parameter_local_rows(name):
                value_item = self._parameter_local_table.item(local_row, 2)
                dataset_id = str(
                    value_item.data(QtCore.Qt.ItemDataRole.UserRole) or ""
                )
                if dataset_id in local_values:
                    value_item.setText(local_values[dataset_id])

    def _render_initial_projection(
        self,
        preserved: Mapping[str, object],
        canonicalized_scopes: frozenset[str],
    ) -> None:
        previous = {
            (str(item.get("set_id")), str(item.get("species"))): item
            for item in list(preserved.get("initials") or ())
            if isinstance(item, Mapping)
        }
        for row in range(self._initial_table.rowCount()):
            identity = self._initial_table.item(row, 0)
            key = (
                str(identity.data(QtCore.Qt.ItemDataRole.UserRole + 1) or ""),
                self._initial_table.item(row, 3).text(),
            )
            prior = previous.get(key)
            if prior is None:
                continue
            identity.setCheckState(
                QtCore.Qt.CheckState.Checked if bool(prior.get("fit")) else QtCore.Qt.CheckState.Unchecked
            )
            for column, field in ((5, "minimum"), (6, "maximum")):
                self._initial_table.item(row, column).setText(str(prior.get(field)))
            if "initial_concentrations" not in canonicalized_scopes:
                self._initial_table.item(row, 4).setText(str(prior.get("initial")))
            self._initial_table.item(row, 7).setCheckState(
                QtCore.Qt.CheckState.Checked if bool(prior.get("log10")) else QtCore.Qt.CheckState.Unchecked
            )
            if "initial_concentrations" not in canonicalized_scopes:
                scope = self._initial_table.cellWidget(row, 8)
                if isinstance(scope, QtWidgets.QComboBox):
                    scope.setCurrentText(
                        str(prior.get("scope") or "Global")
                    )

    def _render_execution_projection(
        self,
        datasets: Mapping[str, object],
        canonicalized_scopes: frozenset[str],
    ) -> None:
        if "simulation_setup" in canonicalized_scopes:
            return
        for row in range(self._execution_table.rowCount()):
            identity = self._execution_table.item(row, 0)
            dataset_id = str(identity.data(QtCore.Qt.ItemDataRole.UserRole) or "")
            prior_dataset = datasets.get(dataset_id)
            if not isinstance(prior_dataset, Mapping):
                continue
            execution = prior_dataset.get("execution")
            if not isinstance(execution, Mapping):
                continue
            solver = self._execution_table.cellWidget(row, 2)
            if isinstance(solver, QtWidgets.QComboBox):
                solver.setCurrentText(str(execution.get("solver")))
            for column, key in ((3, "t_start"), (4, "t_end"), (5, "integration_points")):
                self._execution_table.item(row, column).setText(str(execution.get(key)))

    def _show_add_observable_dialog(self) -> None:
        if not self._origin.mapped_datasets:
            self.set_status("Map a dataset before adding a Fit observable.")
            return
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Add Fit-local Observable")
        form = QtWidgets.QFormLayout(dialog)
        name = QtWidgets.QLineEdit(dialog)
        expression = QtWidgets.QLineEdit(dialog)
        dataset = QtWidgets.QComboBox(dialog)
        for mapped in self._origin.mapped_datasets:
            dataset.addItem(mapped.display_name, mapped.dataset_id)
        column = QtWidgets.QComboBox(dialog)
        scalar_scope = QtWidgets.QComboBox(dialog)
        scalar_scope.addItems(("Shared", "Per-dataset"))

        def refresh_columns() -> None:
            column.clear()
            dataset_id = str(dataset.currentData() or "")
            mapped = next(
                item for item in self._origin.mapped_datasets
                if item.dataset_id == dataset_id
            )
            for observation in mapped.observations:
                column.addItem(
                    observation.raw_column,
                    observation.raw_observation_id,
                )

        dataset.currentIndexChanged.connect(refresh_columns)
        refresh_columns()
        form.addRow("Name", name)
        form.addRow("Expression", expression)
        form.addRow("Dataset", dataset)
        form.addRow("Observed column", column)
        form.addRow("New scalar scope", scalar_scope)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            parent=dialog,
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        self.addObservableRequested.emit(
            name.text(),
            expression.text(),
            str(dataset.currentData() or ""),
            str(column.currentData() or ""),
            scalar_scope.currentText(),
        )

    def _current_target_dataset_id(self) -> str:
        row = self._target_table.currentRow()
        if row < 0:
            return ""
        return str(
            self._target_table.item(row, 0).data(
                QtCore.Qt.ItemDataRole.UserRole
            )
            or ""
        )

    def _dataset_x_name(self, dataset_id: str) -> str:
        for row in range(self._dataset_table.rowCount()):
            identity = self._dataset_table.item(row, 0)
            if str(
                identity.data(QtCore.Qt.ItemDataRole.UserRole) or ""
            ) != dataset_id:
                continue
            selector = self._dataset_table.cellWidget(row, 8)
            if isinstance(selector, QtWidgets.QComboBox):
                return str(selector.currentData() or "t")
        return "t"

    def _target_row_is_eligible(self, row: int) -> bool:
        identity = self._target_table.item(row, 0)
        dataset_id = str(
            identity.data(QtCore.Qt.ItemDataRole.UserRole) or ""
        )
        raw_column = self._target_table.item(row, 2).text()
        raw_observation_id = str(
            self._target_table.item(row, 2).data(
                QtCore.Qt.ItemDataRole.UserRole + 1
            )
            or ""
        )
        model_target = self._target_table.item(row, 3).text().strip()
        try:
            mapped = next(
                item
                for item in self._origin.mapped_datasets
                if item.dataset_id == dataset_id
            )
            observation = next(
                item
                for item in mapped.observations
                if item.raw_observation_id == raw_observation_id
            )
        except StopIteration:
            return False
        duplicate_target = any(
            other_row != row
            and str(
                self._target_table.item(other_row, 0).data(
                    QtCore.Qt.ItemDataRole.UserRole
                )
                or ""
            )
            == dataset_id
            and self._target_table.item(other_row, 3).text().strip()
            == model_target
            for other_row in range(self._target_table.rowCount())
        )
        return bool(
            model_target
            and observation.raw_column == raw_column
            and not duplicate_target
            and len(observation.t) == len(observation.y)
            == len(observation.source_rows)
            and bool(observation.t)
            and all(row_number > 0 for row_number in observation.source_rows)
            and len(set(observation.source_rows))
            == len(observation.source_rows)
            and model_target != self._dataset_x_name(dataset_id)
        )

    def _target_selection(
        self,
        dataset_id: str,
    ) -> FitTargetSelection:
        return FitTargetSelection(
            dataset_id=dataset_id,
            eligible_targets=tuple(
                self._target_table.item(row, 3).text()
                for row in range(self._target_table.rowCount())
                if str(
                    self._target_table.item(row, 0).data(
                        QtCore.Qt.ItemDataRole.UserRole
                    )
                    or ""
                )
                == dataset_id
                and self._target_row_is_eligible(row)
            ),
            selected_targets=tuple(
                self._target_table.item(row, 3).text()
                for row in range(self._target_table.rowCount())
                if str(
                    self._target_table.item(row, 0).data(
                        QtCore.Qt.ItemDataRole.UserRole
                    )
                    or ""
                )
                == dataset_id
                and self._target_row_is_eligible(row)
                and self._target_table.item(row, 0).checkState()
                == QtCore.Qt.CheckState.Checked
            ),
        )

    def _install_target_selection(
        self,
        selection: FitTargetSelection,
    ) -> None:
        selected = set(selection.selected_targets)
        for row in range(self._target_table.rowCount()):
            item = self._target_table.item(row, 0)
            if (
                str(item.data(QtCore.Qt.ItemDataRole.UserRole) or "")
                != selection.dataset_id
                or not self._target_row_is_eligible(row)
            ):
                continue
            item.setCheckState(
                QtCore.Qt.CheckState.Checked
                if self._target_table.item(row, 3).text() in selected
                else QtCore.Qt.CheckState.Unchecked
            )

    def _on_dataset_item_changed(
        self,
        item: QtWidgets.QTableWidgetItem,
    ) -> None:
        if self._projecting_setup or self._refreshing_dataset_controls:
            return
        dataset_id = str(
            self._dataset_table.item(item.row(), 0).data(
                QtCore.Qt.ItemDataRole.UserRole
            )
            or ""
        )
        if (
            item.column() == 0
            and not self._refreshing_dataset_controls
        ):
            self._reconcile_parameter_local_rows()
            for row in range(self._parameter_table.rowCount()):
                name_item = self._parameter_table.item(row, 1)
                self._project_shared_parameter_from_local_rows(
                    str(
                        name_item.data(
                            QtCore.Qt.ItemDataRole.UserRole
                        )
                        or name_item.text()
                    )
                )
        self._refresh_dataset_controls()
        self._emit_dataset_edit(dataset_id)

    def _on_target_item_changed(
        self,
        _item: QtWidgets.QTableWidgetItem,
    ) -> None:
        if self._projecting_setup:
            return
        self._refresh_dataset_controls()

    def _on_execution_item_changed(
        self,
        item: QtWidgets.QTableWidgetItem,
    ) -> None:
        if self._projecting_setup:
            return
        dataset_id = str(
            self._execution_table.item(item.row(), 0).data(
                QtCore.Qt.ItemDataRole.UserRole
            )
            or ""
        )
        self._emit_dataset_edit(dataset_id)

    def _on_dataset_selector_changed(self, dataset_id: str) -> None:
        if self._projecting_setup:
            return
        self._refresh_dataset_controls()
        self._emit_dataset_edit(dataset_id)

    def _refresh_dataset_controls(self) -> None:
        if self._refreshing_dataset_controls:
            return
        self._refreshing_dataset_controls = True
        try:
            for row in range(self._target_table.rowCount()):
                item = self._target_table.item(row, 0)
                eligible = self._target_row_is_eligible(row)
                flags = item.flags()
                if eligible:
                    item.setFlags(
                        flags
                        | QtCore.Qt.ItemFlag.ItemIsEnabled
                        | QtCore.Qt.ItemFlag.ItemIsUserCheckable
                    )
                else:
                    item.setCheckState(QtCore.Qt.CheckState.Unchecked)
                    item.setFlags(
                        flags
                        & ~QtCore.Qt.ItemFlag.ItemIsEnabled
                    )
            mapped_by_id = {
                item.dataset_id: item
                for item in self._origin.mapped_datasets
            }
            for row in range(self._dataset_table.rowCount()):
                identity = self._dataset_table.item(row, 0)
                dataset_id = str(
                    identity.data(QtCore.Qt.ItemDataRole.UserRole) or ""
                )
                x_selector = self._dataset_table.cellWidget(row, 8)
                mapping_selector = self._dataset_table.cellWidget(row, 9)
                x_name = (
                    str(x_selector.currentData() or "t")
                    if isinstance(x_selector, QtWidgets.QComboBox)
                    else "t"
                )
                if isinstance(mapping_selector, QtWidgets.QComboBox):
                    mapping_selector.setEnabled(x_name != "t")
                resolved = self._dataset_table.item(row, 10)
                if x_name == "t":
                    resolved.setText("Time")
                elif (
                    isinstance(mapping_selector, QtWidgets.QComboBox)
                    and mapping_selector.currentText() == "Auto"
                ):
                    resolved.setText("Pending")
                elif isinstance(mapping_selector, QtWidgets.QComboBox):
                    resolved.setText(mapping_selector.currentText())
                used = 0
                valid = True
                try:
                    t_min = _read_float(
                        self._dataset_table.item(row, 3).text(),
                        "t min",
                    )
                    t_max = _read_float(
                        self._dataset_table.item(row, 4).text(),
                        "t max",
                    )
                    points_text = (
                        self._dataset_table.item(row, 5).text().strip()
                    )
                    points = (
                        None
                        if points_text.casefold() == "all"
                        else int(points_text)
                    )
                    mapped = mapped_by_id[dataset_id]
                    observations = {
                        item.raw_observation_id: item
                        for item in mapped.observations
                    }
                    for target_row in range(
                        self._target_table.rowCount()
                    ):
                        target_item = self._target_table.item(
                            target_row,
                            0,
                        )
                        if (
                            str(
                                target_item.data(
                                    QtCore.Qt.ItemDataRole.UserRole
                                )
                                or ""
                            )
                            != dataset_id
                            or target_item.checkState()
                            != QtCore.Qt.CheckState.Checked
                            or not self._target_row_is_eligible(target_row)
                        ):
                            continue
                        observation = observations[
                            str(
                                self._target_table.item(
                                    target_row,
                                    2,
                                ).data(
                                    QtCore.Qt.ItemDataRole.UserRole + 1
                                )
                                or ""
                            )
                        ]
                        used += int(
                            compute_sampled_indices(
                                t=np.asarray(
                                    observation.t,
                                    dtype=float,
                                ),
                                t_min=t_min,
                                t_max=t_max,
                                n_points=points,
                            ).size
                        )
                except (KeyError, TypeError, ValueError, OverflowError):
                    valid = False
                self._dataset_table.item(row, 6).setText(
                    str(used) if valid else "Invalid"
                )
        finally:
            self._refreshing_dataset_controls = False

    def _bulk_targets(self, operation: str) -> None:
        dataset_id = self._current_target_dataset_id()
        if not dataset_id:
            self.set_status("Select a target row first.")
            return
        selection = self._target_selection(dataset_id)
        if operation == "all":
            selection = selection.all()
        elif operation == "none":
            selection = selection.none()
        else:
            selection = selection.invert()
        self._install_target_selection(selection)
        self._refresh_dataset_controls()

    def _copy_targets(self) -> None:
        source_id = self._current_target_dataset_id()
        if not source_id:
            self.set_status("Select a source target row first.")
            return
        source = self._target_selection(source_id)
        if not source.selected_targets:
            self.set_status("Copy Targets: 0 copied, 0 skipped.")
            return
        included_ids = tuple(
            str(
                self._dataset_table.item(row, 0).data(
                    QtCore.Qt.ItemDataRole.UserRole
                )
                or ""
            )
            for row in range(self._dataset_table.rowCount())
            if self._dataset_table.item(row, 0).checkState()
            == QtCore.Qt.CheckState.Checked
            and str(
                self._dataset_table.item(row, 0).data(
                    QtCore.Qt.ItemDataRole.UserRole
                )
                or ""
            )
            != source_id
        )
        destinations = tuple(
            self._target_selection(dataset_id)
            for dataset_id in included_ids
        )
        copied, matched, unmatched = copy_target_selection(
            source,
            destinations,
        )
        copied_by_id = {
            item.dataset_id: set(item.selected_targets)
            for item in copied
        }
        compatible_ids = {
            item.dataset_id
            for original, item in zip(destinations, copied)
            if original != item
            or set(source.selected_targets).issubset(
                original.eligible_targets
            )
        }
        for row in range(self._target_table.rowCount()):
            item = self._target_table.item(row, 0)
            dataset_id = str(
                item.data(QtCore.Qt.ItemDataRole.UserRole) or ""
            )
            if (
                dataset_id not in compatible_ids
                or not self._target_row_is_eligible(row)
            ):
                continue
            item.setCheckState(
                QtCore.Qt.CheckState.Checked
                if self._target_table.item(row, 3).text()
                in copied_by_id[dataset_id]
                else QtCore.Qt.CheckState.Unchecked
            )
        self._refresh_dataset_controls()
        self.set_status(
            f"Copy Targets: {matched} copied, {unmatched} skipped."
        )

    def _apply_target_changes(self) -> None:
        dataset_ids = tuple(
            dict.fromkeys(
                str(
                    self._target_table.item(row, 0).data(
                        QtCore.Qt.ItemDataRole.UserRole
                    )
                    or ""
                )
                for row in range(self._target_table.rowCount())
            )
        )
        if not self._emit_dataset_edits(
            dataset_ids,
            include_pending_targets=True,
        ):
            self.set_status("Pending Fit target selections are invalid.")
            return
        self._committed_targets = self._current_target_state()
        self._refresh_dataset_controls()
        self.set_status("Pending Fit target selections applied locally.")

    def _current_target_state(
        self,
    ) -> dict[tuple[str, str], tuple[bool, str, str]]:
        return {
            (
                str(
                    self._target_table.item(row, 0).data(
                        QtCore.Qt.ItemDataRole.UserRole
                    )
                    or ""
                ),
                str(
                    self._target_table.item(row, 2).data(
                        QtCore.Qt.ItemDataRole.UserRole + 1
                    )
                    or ""
                ),
            ): (
                self._target_table.item(row, 0).checkState()
                == QtCore.Qt.CheckState.Checked,
                self._target_table.item(row, 3).text(),
                self._target_table.item(row, 4).text(),
            )
            for row in range(self._target_table.rowCount())
        }

    @property
    def has_pending_target_changes(self) -> bool:
        return self._current_target_state() != self._committed_targets

    def commit_current_target_selection(self) -> None:
        self._committed_targets = self._current_target_state()

    def _revert_target_changes(self) -> None:
        self.project_setup_snapshot(self._require_projected_setup())
        self.set_status("Pending Fit target selections reverted.")

    def _toggle_pause(self) -> None:
        self._paused = not self._paused
        self._pause_button.setText("Resume" if self._paused else "Pause")
        if self._paused:
            self.pauseRequested.emit()
        else:
            self.resumeRequested.emit()

    def _request_execute(self) -> None:
        if self.has_invalid_setup_edits:
            self.set_status(
                "Fit setup contains an invalid edit; correct it before running."
            )
            return
        self.executeRequested.emit()

    def _restore_named_snapshot(self, name: str) -> None:
        snapshot = self._initial_setup if name == "initial" else self._last_fit_setup
        if snapshot is None:
            self.set_status("That Fit setup snapshot is unavailable.")
            return
        self.restoreSetupRequested.emit(snapshot)

    def set_executing(self, executing: bool) -> None:
        if executing:
            self._revoke_curve_copy()
            self._copy_run_json.setEnabled(False)
        self._run_button.setEnabled(not executing)
        self._cancel_button.setEnabled(executing)
        self._pause_button.setEnabled(executing)
        if not executing:
            self._paused = False
            self._pause_button.setText("Pause")
        for widget in (
            self._data_tab,
            self._parameters_tab,
            self._initials_tab,
            self._add_observable_button,
            self._targets_all,
            self._targets_none,
            self._targets_invert,
            self._targets_copy,
            self._targets_apply,
            self._targets_revert,
        ):
            widget.setEnabled(not executing)

    def set_detailed_balance_correction_pending(self) -> None:
        self.set_executing(True)
        self._pause_button.setEnabled(False)
        self._cancel_button.setEnabled(True)

    def set_progress(self, value: int, text: str) -> None:
        self._progress.setValue(max(0, min(100, int(value))))
        self.set_status(text)

    def set_status(self, text: str) -> None:
        self._status.setText(self._display_text(text))

    def _display_text(self, text: object) -> str:
        rendered = str(text)
        for mapped in sorted(
            self._origin.mapped_datasets,
            key=lambda item: max(
                len(item.dataset_id),
                len(item.set_id),
            ),
            reverse=True,
        ):
            rendered = rendered.replace(
                mapped.dataset_id,
                mapped.display_name,
            )
            rendered = rendered.replace(
                mapped.set_id,
                mapped.set_name,
            )
        return rendered

    def show_origin_conflict(self, requested_tab_name: str) -> None:
        self.set_status(
            f"This modal Fit belongs to {self._origin.tab_name}; "
            f"close it before fitting {requested_tab_name}."
        )
        self.show()
        self.raise_()
        self.activateWindow()

    def set_live_best(self, update: FitLiveUpdate) -> None:
        if not isinstance(update, FitLiveUpdate):
            raise TypeError("update must be FitLiveUpdate.")
        self._tracker_summary.setText(
            f"Best iteration {update.iteration}; objective SSQ={update.cost:.8g}"
        )
        parameter_labels = {
            item.name: item.display_name
            for item in self._origin.parameter_defaults
        }
        set_labels = {
            item.set_id: item.set_name
            for item in self._origin.mapped_datasets
        }
        dataset_labels = {
            item.dataset_id: item.display_name
            for item in self._origin.mapped_datasets
        }
        lines = [
            f"{parameter_labels.get(name, name)} = {value:.12g}"
            for name, value in update.parameters
        ]
        for dataset_id, values in update.scalar_values_by_dataset:
            lines.append(f"[{dataset_labels.get(dataset_id, 'Dataset')}]")
            lines.extend(
                f"{parameter_labels.get(name, name)} = {value:.12g}"
                for name, value in values
            )
        for set_id, values in update.initial_values_by_set:
            lines.append(f"[{set_labels.get(set_id, 'Mapped set')}]")
            lines.extend(f"{name} = {value:.12g}" for name, value in values)
        self._tracker_parameters.setPlainText("\n".join(lines))

    def set_live_curves(self, curves: tuple[FitCurveDataset, ...]) -> None:
        self._all_datasets_view.set_curves(curves)
        by_id = {item.dataset_id: item for item in curves}
        for dataset_id, view in self._dataset_view_by_id.items():
            curve = by_id.get(dataset_id)
            view.set_curves(()) if curve is None else view.set_curves((curve,))

    def set_resolved_auto_modes(
        self,
        modes: tuple[tuple[str, str], ...],
    ) -> None:
        resolved_modes = dict(modes)
        blocker = QtCore.QSignalBlocker(self._dataset_table)
        for row in range(self._dataset_table.rowCount()):
            dataset_id = str(
                self._dataset_table.item(row, 0).data(
                    QtCore.Qt.ItemDataRole.UserRole
                )
                or ""
            )
            resolved = resolved_modes.get(dataset_id)
            if resolved is not None:
                self._dataset_table.item(row, 10).setText(
                    "Time-guided"
                    if resolved == "time_guided"
                    else "Monotone"
                )
        del blocker

    def set_run_identity(self, identity: FitRunIdentity) -> None:
        if not isinstance(identity, FitRunIdentity):
            raise TypeError("identity must be a FitRunIdentity.")
        self._run_identity = identity
        self._copy_run_json.setEnabled(True)

    def set_candidate(
        self,
        candidate: FitCandidate,
        *,
        available_scopes: frozenset[str],
        unavailable_reasons: Mapping[str, str],
    ) -> bool:
        if not isinstance(candidate, FitCandidate):
            raise TypeError("candidate must be FitCandidate.")
        available = set(available_scopes)
        reasons = {
            str(scope): str(reason or "").strip()
            for scope, reason in unavailable_reasons.items()
        }
        known_scopes = {
            "parameters",
            "initial_concentrations",
            "simulation_setup",
            "observable_definitions",
        }
        if (
            not available.issubset(known_scopes)
            or not set(reasons).issubset(known_scopes - available)
            or any(not reason for reason in reasons.values())
        ):
            raise ValueError("Fit Apply feasibility is invalid.")
        self._revoke_curve_copy()
        for checkbox, scope in (
            (self._apply_parameters, "parameters"),
            (self._apply_initials, "initial_concentrations"),
            (self._apply_setup, "simulation_setup"),
            (self._apply_observables, "observable_definitions"),
        ):
            checkbox.setEnabled(scope in available)
            checkbox.setChecked(scope in available)
            checkbox.setToolTip(reasons.get(scope, ""))
        self._apply_button.setEnabled(bool(available))
        result = candidate.result
        dataset_labels = {
            item.dataset_id: item.display_name
            for item in self._origin.mapped_datasets
        }
        set_labels = {
            item.set_id: item.set_name
            for item in self._origin.mapped_datasets
        }
        parameter_labels = {
            item.name: item.display_name
            for item in candidate.setup_snapshot.parameters
        }
        self._result_summary.setPlainText(
            "\n".join(
                (
                    f"Run: {candidate.run_identity.short_hash}",
                    self._display_text(result.message),
                    f"Weighted residual SSE: {float(result.weighted_residual_sse):.8g}",
                    f"Weighted residual MSE: {float(result.weighted_residual_mse):.8g}",
                    f"Function evaluations: {int(result.nfev)}",
                    f"Pending local observables: {len(candidate.local_observables)}",
                    *(reasons[scope] for scope in sorted(reasons)),
                    *(
                        f"{dataset_labels.get(info.dataset_id, 'Dataset')}: "
                        f"weighted SSE={info.weighted_residual_sse:.8g}, "
                        f"weighted MSE={info.weighted_residual_mse:.8g}, "
                        f"weighted RMSE={info.weighted_rmse:.8g}, "
                        f"weighted MAE={info.weighted_mae:.8g}, "
                        f"weight={info.weight:.8g}, points={info.n_points}"
                        for info in result.dataset_info
                    ),
                    *(
                        f"{dataset_labels.get(info.dataset_id, 'Dataset')} "
                        f"{series}: R²={r_squared:.8g}"
                        for info in result.dataset_info
                        for series, r_squared in sorted(
                            info.r_squared_by_series.items()
                        )
                    ),
                    *(
                        f"shared {parameter_labels.get(name, name)}={value:.12g}"
                        for name, value in candidate.parameters
                    ),
                    *(
                        f"{dataset_labels.get(dataset_id, 'Dataset')} "
                        f"{parameter_labels.get(name, name)}={value:.12g}"
                        for dataset_id, values in candidate.scalar_values_by_dataset
                        for name, value in values
                    ),
                    *(
                        f"{set_labels.get(set_id, 'Mapped set')} "
                        f"init {name}={value:.12g}"
                        for set_id, values in candidate.initial_values_by_set
                        for name, value in values
                    ),
                )
            )
        )
        selected_before = {
            self._species_filter.item(index).text()
            for index in range(self._species_filter.count())
            if self._species_filter.item(index).checkState()
            == QtCore.Qt.CheckState.Checked
        }
        available_species = tuple(
            dict.fromkeys(
                species
                for curve in candidate.curves
                for species, _values in curve.model_y
            )
        )
        self._species_filter.blockSignals(True)
        self._species_filter.clear()
        for species in available_species:
            item = QtWidgets.QListWidgetItem(species)
            item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                QtCore.Qt.CheckState.Checked
                if (
                    not self._species_filter_initialized
                    or species in selected_before
                )
                else QtCore.Qt.CheckState.Unchecked
            )
            self._species_filter.addItem(item)
        self._species_filter.blockSignals(False)
        self._species_filter_initialized = True
        self._candidate_curves = candidate.curves
        self._last_fit_setup = candidate.setup_snapshot
        self._view_last_fit.setEnabled(True)
        self.set_run_identity(candidate.run_identity)
        self.set_resolved_auto_modes(candidate.resolved_auto_modes)
        try:
            self._apply_species_filter()
        except Exception:
            self._revoke_curve_copy()
            return False
        self._copy_button.setEnabled(self._all_datasets_view.copy_eligible())
        self._tabs.setCurrentWidget(self._results_tab)
        return True

    def _revoke_curve_copy(self) -> None:
        """Clear every physical and semantic Fit-curve Copy route."""

        self._copy_button.setEnabled(False)
        self._candidate_curves = ()
        try:
            self.set_live_curves(())
        except Exception:
            try:
                self._all_datasets_view.set_curves(())
            except Exception:
                pass
            for view in self._dataset_view_by_id.values():
                try:
                    view.set_curves(())
                except Exception:
                    pass

    def clear_candidate(self) -> None:
        for checkbox in (
            self._apply_parameters,
            self._apply_initials,
            self._apply_setup,
            self._apply_observables,
        ):
            checkbox.setChecked(False)
            checkbox.setEnabled(False)
            checkbox.setToolTip("")
        self._apply_button.setEnabled(False)
        self._run_identity = None
        self._copy_button.setEnabled(False)
        self._copy_run_json.setEnabled(False)
        self._revoke_curve_copy()
        self._tracker_summary.setText(
            "Run a fit to see the best candidate."
        )
        self._tracker_parameters.clear()
        self._result_summary.clear()
        self._refresh_dataset_controls()

    def _copy_run_identity_json(self) -> None:
        identity = self._run_identity
        if identity is None:
            self.set_status("No completed Fit run is available to copy.")
            return
        clipboard = QtWidgets.QApplication.clipboard()
        clipboard.setText(identity.readable_json())
        self.set_status("Copied the complete Fit run JSON.")

    def _emit_apply(self) -> None:
        scopes = frozenset(
            scope
            for checkbox, scope in (
                (self._apply_parameters, "parameters"),
                (self._apply_initials, "initial_concentrations"),
                (self._apply_setup, "simulation_setup"),
                (self._apply_observables, "observable_definitions"),
            )
            if checkbox.isEnabled() and checkbox.isChecked()
        )
        if not scopes:
            self.set_status("Select at least one Apply scope.")
            return
        self.applyRequested.emit(scopes)

    def _apply_species_filter(self, _item: object = None) -> None:
        curves = getattr(self, "_candidate_curves", ())
        selected = {
            self._species_filter.item(index).text()
            for index in range(self._species_filter.count())
            if self._species_filter.item(index).checkState()
            == QtCore.Qt.CheckState.Checked
        }
        filtered = tuple(
            FitCurveDataset(
                dataset_id=curve.dataset_id,
                display_name=curve.display_name,
                model_x=tuple(item for item in curve.model_x if item[0] in selected),
                model_y=tuple(item for item in curve.model_y if item[0] in selected),
                observed_x=tuple(item for item in curve.observed_x if item[0] in selected),
                observed_y=tuple(item for item in curve.observed_y if item[0] in selected),
                model_segment_ids=tuple(
                    item
                    for item in curve.model_segment_ids
                    if item[0] in selected
                ),
                model_segment_keys=tuple(
                    item
                    for item in curve.model_segment_keys
                    if item[0] in selected
                ),
                model_discontinuity_ids=tuple(
                    item
                    for item in curve.model_discontinuity_ids
                    if item[0] in selected
                ),
                observed_source_rows=tuple(
                    item
                    for item in curve.observed_source_rows
                    if item[0] in selected
                ),
                observed_raw_ids=tuple(
                    item
                    for item in curve.observed_raw_ids
                    if item[0] in selected
                ),
                observed_raw_columns=tuple(
                    item
                    for item in curve.observed_raw_columns
                    if item[0] in selected
                ),
                objective_model_x=tuple(
                    item for item in curve.objective_model_x if item[0] in selected
                ),
                objective_model_y=tuple(
                    item for item in curve.objective_model_y if item[0] in selected
                ),
                objective_model_segment_ids=tuple(
                    item
                    for item in curve.objective_model_segment_ids
                    if item[0] in selected
                ),
                objective_model_segment_keys=tuple(
                    item
                    for item in curve.objective_model_segment_keys
                    if item[0] in selected
                ),
                objective_model_discontinuity_ids=tuple(
                    item
                    for item in curve.objective_model_discontinuity_ids
                    if item[0] in selected
                ),
                objective_selection_provenance=tuple(
                    item
                    for item in curve.objective_selection_provenance
                    if item[0] in selected
                ),
                objective_observation_ids=tuple(
                    item
                    for item in curve.objective_observation_ids
                    if item[0] in selected
                ),
                objective_raw_columns=tuple(
                    item
                    for item in curve.objective_raw_columns
                    if item[0] in selected
                ),
                objective_source_rows=tuple(
                    item
                    for item in curve.objective_source_rows
                    if item[0] in selected
                ),
                x_name=curve.x_name,
            )
            for curve in curves
            if any(name in selected for name, _values in curve.model_y)
        )
        self.set_live_curves(filtered)
        self._copy_button.setEnabled(self._all_datasets_view.copy_eligible())

    def force_close_after_cleanup(self) -> None:
        self._closing_after_cleanup = True
        self.close()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt override
        if self._closing_after_cleanup:
            event.accept()
            return
        if self._close_handler(time.monotonic() + 2.0):
            self._closing_after_cleanup = True
            event.accept()
        else:
            self.set_status("Fit is still stopping; close was deferred.")
            event.ignore()
