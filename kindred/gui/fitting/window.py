"""Physical Fit presentation bound to the current typed session owner."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import replace

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from kindred.gui.fitting.contracts import FitParametricSettings, FitRunIdentity
from kindred.gui.fitting.constants import SIMULATION_SOLVERS
from kindred.gui.fitting.data_tab import DataTab
from kindred.gui.fitting.data_targets_tab import DataTargetsTab
from kindred.gui.fitting.launch import (
    FitDatasetLocalSettings,
    FitDatasetSetupEdit,
    FitDatasetWeightModeSetupEdit,
    FitDefaultsSetupEdit,
    FitInitialSetupEdit,
    FitMappedDatasetSnapshot,
    FitOriginSnapshot,
    FitParameterSetupEdit,
    FitSetupSnapshot,
)
from kindred.gui.fitting.parameters_ics_tab import ParametersIcsTab
from kindred.gui.fitting.run_results_tab import RunResultsTab
from kindred.gui.fitting.unified_species_table import UnifiedSpeciesTable
from kindred.gui.fitting.worker import FitCandidate, FitCurveDataset, FitLiveUpdate

__all__ = ["FittingWindow"]


class FittingWindow(QtWidgets.QDialog):
    """Historical executable Fit shell with current owner-facing contracts.

    This object is deliberately a presentation boundary.  The owner retains
    accepted setup, request construction, worker lifetime, result authority,
    and project mutation; the child views only render snapshots and emit
    intents.
    """

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
        self._paused = False
        self._initial_setup: FitSetupSnapshot | None = None
        self._last_fit_setup: FitSetupSnapshot | None = None
        self._snapshot_capture_id: str | None = None
        self._projected_setup: FitSetupSnapshot | None = None
        self._invalid_setup_edits: set[tuple[str, ...]] = set()
        self._execution_invalid_raw: dict[tuple[str, str], str] = {}
        self._execution_invalid_messages: dict[tuple[str, str], str] = {}
        self._pending_target_changes = False
        self._resolved_auto_modes: tuple[tuple[str, str], ...] = ()
        self._run_identity: FitRunIdentity | None = None
        self._candidate: FitCandidate | None = None
        self._candidate_available_scopes: frozenset[str] = frozenset()
        self._candidate_unavailable_reasons: dict[str, str] = {}
        self._executing = False
        self._execution_projecting = False

        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal)
        self.setModal(True)
        self.setWindowTitle("Fitting Window")
        self.resize(1280, 720)
        self.setWindowFlags(
            self.windowFlags()
            | QtCore.Qt.WindowType.WindowMinMaxButtonsHint
        )
        self._build_ui()

    # ------------------------------------------------------------------
    # Historical top-level presentation
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        self._species_table = UnifiedSpeciesTable(
            origin=self._origin,
            mechanism_species=self._mechanism_species(),
            parent=self,
        )
        self._data_tab = DataTab(
            sampling_applied_config_getter=self._sampling_applied_config,
            sampling_default_config_getter=self._sampling_default_config,
            fit_targets_full_t_getter=self._selected_target_times_for_dataset,
            fit_targets_available_getter=self._available_targets_for_dataset,
            fit_targets_full_series_getter=self._full_series_for_dataset,
            fit_targets_selection_applied_getter=self._applied_targets_for_dataset,
            modeled_series_getter=self._modeled_series,
            parent=self,
        )
        self._data_targets_tab = DataTargetsTab(
            sampling_panel=self._data_tab,
            species_table=self._species_table,
            parent=self,
        )
        self._params_ics_tab = ParametersIcsTab(
            origin=self._origin,
            parameter_choices=self._parameter_choices_for_origin(),
            parent=self,
        )
        self._run_results_tab = RunResultsTab(
            parent=self,
            color_resolver=self._color_resolver,
        )
        self._project_result_identity_labels()
        self._run_results_tab.set_dark_mode(self._current_results_dark_mode())
        self._build_execution_presentation()

        self._tabs = QtWidgets.QTabBar(self)
        self._tabs.setObjectName("global_fit_top_tabs")
        self._tabs.setDocumentMode(True)
        self._tabs.setDrawBase(False)
        self._tabs.addTab("Data and Targets")
        self._tabs.addTab("Parameters")
        self._tabs.addTab("Results")
        layout.addWidget(self._tabs, stretch=0)

        self._current_tab_stack = QtWidgets.QStackedWidget(self)
        self._current_tab_stack.setObjectName("global_fit_current_tab_stack")
        self._current_tab_stack.setMinimumWidth(320)
        self._current_tab_stack.addWidget(self._data_targets_tab)
        self._current_tab_stack.addWidget(self._params_ics_tab)
        self._current_tab_stack.addWidget(self._run_results_tab)
        layout.addWidget(self._current_tab_stack, stretch=1)
        self._tabs.currentChanged.connect(self._current_tab_stack.setCurrentIndex)
        self._current_tab_stack.setCurrentIndex(self._tabs.currentIndex())

        layout.addWidget(self._create_footer(), stretch=0)
        self._connect_leaf_signals()

    def _create_footer(self) -> QtWidgets.QWidget:
        footer = QtWidgets.QWidget(self)
        footer.setObjectName("global_fit_footer")
        control_row = QtWidgets.QHBoxLayout(footer)
        control_row.setContentsMargins(0, 0, 0, 0)
        control_row.setSpacing(8)

        status_widget = QtWidgets.QWidget(footer)
        status_column = QtWidgets.QVBoxLayout(status_widget)
        status_column.setContentsMargins(0, 0, 0, 0)
        status_column.setSpacing(2)
        self._status = QtWidgets.QLabel("Ready", footer)
        self._status.setWordWrap(False)
        self._status.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Ignored,
            QtWidgets.QSizePolicy.Policy.Preferred,
        )
        status_column.addWidget(self._status)
        self._run_block_reason_label = QtWidgets.QLabel(status_widget)
        self._run_block_reason_label.setObjectName(
            "global_fit_run_block_reason_label"
        )
        self._run_block_reason_label.setWordWrap(False)
        self._run_block_reason_label.setStyleSheet("font-size: 11px;")
        self._run_block_reason_label.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Ignored,
            QtWidgets.QSizePolicy.Policy.Preferred,
        )
        self._run_block_reason_label.hide()
        status_column.addWidget(self._run_block_reason_label)
        control_row.addWidget(status_widget, stretch=3)

        self._progress = QtWidgets.QProgressBar(footer)
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        control_row.addWidget(self._progress, stretch=2)

        actions = QtWidgets.QHBoxLayout()
        actions.setContentsMargins(0, 0, 0, 0)
        actions.setSpacing(6)
        self._run_button = QtWidgets.QPushButton("Run Fit", footer)
        self._run_button.clicked.connect(self._request_execute)
        self._cancel_button = QtWidgets.QPushButton("Stop", footer)
        self._cancel_button.setEnabled(False)
        self._cancel_button.clicked.connect(self.cancelRequested)
        self._pause_button = QtWidgets.QPushButton("Pause", footer)
        self._pause_button.setEnabled(False)
        self._pause_button.clicked.connect(self._toggle_pause)
        self._resume_button = QtWidgets.QPushButton("Resume", footer)
        self._resume_button.setEnabled(False)
        self._resume_button.clicked.connect(self._resume_from_button)
        for button in (
            self._run_button,
            self._cancel_button,
            self._pause_button,
            self._resume_button,
        ):
            actions.addWidget(button)

        self._results_summary_button = QtWidgets.QPushButton(
            "Results Summary", footer
        )
        self._results_summary_button.setEnabled(False)
        self._results_summary_button.clicked.connect(
            self._run_results_tab.open_results_summary_dialog
        )
        actions.addWidget(self._results_summary_button)

        self._copy_button = QtWidgets.QPushButton("Copy Fit Curves", footer)
        self._copy_button.setEnabled(False)
        self._copy_button.clicked.connect(
            self._run_results_tab.copy_all_to_clipboard
        )
        actions.addWidget(self._copy_button)
        self._copy_run_json = QtWidgets.QPushButton("Copy Run JSON", footer)
        self._copy_run_json.setEnabled(False)
        self._copy_run_json.clicked.connect(self._copy_run_identity_json)
        actions.addWidget(self._copy_run_json)

        self._apply_parameters = QtWidgets.QCheckBox("Parameters", footer)
        self._apply_initials = QtWidgets.QCheckBox(
            "Initial Concentrations", footer
        )
        self._apply_setup = QtWidgets.QCheckBox("Simulation Setup", footer)
        self._apply_observables = QtWidgets.QCheckBox(
            "Observable Definitions", footer
        )
        for checkbox in (
            self._apply_parameters,
            self._apply_initials,
            self._apply_setup,
            self._apply_observables,
        ):
            checkbox.setEnabled(False)
            checkbox.toggled.connect(self._refresh_apply_button)
            actions.addWidget(checkbox)
        self._apply_button = QtWidgets.QPushButton("Apply to Project", footer)
        self._apply_button.setEnabled(False)
        self._apply_button.clicked.connect(self._emit_apply)
        actions.addWidget(self._apply_button)

        self._project_defaults_button = QtWidgets.QPushButton(
            "Use as Project Defaults", footer
        )
        self._project_defaults_button.clicked.connect(
            self._emit_project_defaults
        )
        actions.addWidget(self._project_defaults_button)

        self._close_button = QtWidgets.QPushButton("Close", footer)
        self._close_button.clicked.connect(self.close)
        actions.addWidget(self._close_button)
        control_row.addLayout(actions)
        return footer

    def _connect_leaf_signals(self) -> None:
        self._connect_origin_dependent_leaf_signals()

        self._params_ics_tab.setupEditsRequested.connect(
            self._on_leaf_setup_edits
        )
        self._params_ics_tab.restoreSetupRequested.connect(
            self.restoreSetupRequested
        )
        self._params_ics_tab.addObservableRequested.connect(
            self._on_add_observable_requested
        )
        self._params_ics_tab.viewStepsRequested.connect(
            self._open_mechanism_inspector
        )
        self._params_ics_tab.statusMessage.connect(self.set_status)
        self._params_ics_tab.validityChanged.connect(
            self._refresh_setup_validity
        )

        self._run_results_tab.statusMessage.connect(self.set_status)
        self._run_results_tab.copyAvailableChanged.connect(
            self._set_copy_available
        )

    def _connect_origin_dependent_leaf_signals(self) -> None:
        self._data_tab.samplingApplied.connect(self._on_sampling_applied)
        self._data_tab.statusMessage.connect(self.set_status)
        self._data_tab.validityChanged.connect(self._refresh_setup_validity)
        self._species_table.setupEditsRequested.connect(
            self._on_leaf_setup_edits
        )
        self._species_table.validityChanged.connect(
            self._refresh_setup_validity
        )
        self._species_table.statusMessage.connect(self.set_status)
        dataset_list = self._data_targets_tab.unified_list
        dataset_list.currentDatasetChanged.connect(
            self._on_dataset_selected
        )
        dataset_list.datasetIncludeChanged.connect(
            self._on_dataset_include_changed
        )
        self._species_table.targetsApplied.connect(
            self.commit_current_target_selection
        )

    def _project_result_identity_labels(self) -> None:
        self._run_results_tab.set_identity_labels(
            dataset_labels={
                item.dataset_id: item.display_name
                for item in self._origin.mapped_datasets
            },
            set_labels={
                item.set_id: item.set_name
                for item in self._origin.mapped_datasets
            },
        )

    def _build_execution_presentation(self) -> None:
        self._execution_table = QtWidgets.QTableWidget(self._params_ics_tab)
        self._execution_table.setObjectName("global_fit_execution_table")
        self._execution_table.setColumnCount(6)
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
        self._execution_table.horizontalHeader().setStretchLastSection(True)
        self._execution_table.itemChanged.connect(
            self._on_execution_item_changed
        )
        self._params_ics_tab.set_execution_presentation_widget(
            self._execution_table
        )

    def _parameter_choices_for_origin(self) -> dict[str, object]:
        return {
            "dataset_ids": tuple(
                item.dataset_id for item in self._origin.mapped_datasets
            ),
            "available_species": self._mechanism_species(),
            "available_rates": (),
            "available_scalars": (),
            "available_observables": {},
        }

    def _open_mechanism_inspector(self) -> None:
        from kindred.gui.widgets.mechanism_inspector import MechanismInspectorDialog

        MechanismInspectorDialog(
            self._origin.prepared_dsl,
            self,
        ).exec()

    def _replace_origin_dependent_presentation(
        self,
        origin: FitOriginSnapshot,
    ) -> None:
        """Replace the one visible data page when its captured origin changes."""

        current_index = self._current_tab_stack.currentIndex()
        old_page = self._data_targets_tab
        self._current_tab_stack.removeWidget(old_page)
        old_page.deleteLater()

        self._species_table = UnifiedSpeciesTable(
            origin=origin,
            mechanism_species=tuple(
                dict.fromkeys(
                    species
                    for mapped in origin.mapped_datasets
                    for species, _value in mapped.initial_values
                )
            ),
            parent=self,
        )
        self._data_targets_tab = DataTargetsTab(
            sampling_panel=self._data_tab,
            species_table=self._species_table,
            parent=self,
        )
        self._current_tab_stack.insertWidget(0, self._data_targets_tab)
        self._current_tab_stack.setCurrentIndex(max(0, current_index))
        self._connect_origin_dependent_leaf_signals()

    # ------------------------------------------------------------------
    # Typed projection and setup-intent boundary
    # ------------------------------------------------------------------

    def rebase(
        self,
        origin: FitOriginSnapshot,
        *,
        setup: FitSetupSnapshot,
        preserve_snapshots: bool = False,
    ) -> None:
        if not isinstance(origin, FitOriginSnapshot):
            raise TypeError("origin must be FitOriginSnapshot.")
        self._check_snapshot_origin(setup, origin)
        capture_changed = self._snapshot_capture_id != origin.capture_id
        if origin is not self._origin and not preserve_snapshots:
            self._replace_origin_dependent_presentation(origin)
        elif origin is not self._origin:
            # A preserving rebase retains widget-local pending/raw state.  The
            # species view's origin is presentation identity only; accepted
            # setup and mutation authority remain with FitSessionOwner.
            self._species_table._origin = origin  # noqa: SLF001
        if capture_changed:
            if not preserve_snapshots:
                self._initial_setup = None
                self._last_fit_setup = None
                self._invalid_setup_edits.clear()
                self._execution_invalid_raw.clear()
                self._execution_invalid_messages.clear()
                self._pending_target_changes = False
                self._clear_resolved_auto_modes()
            else:
                self._initial_setup = self._rebase_retained_snapshot(
                    self._initial_setup,
                    origin,
                )
                self._last_fit_setup = self._rebase_retained_snapshot(
                    self._last_fit_setup,
                    origin,
                )
            self._snapshot_capture_id = origin.capture_id
        self._origin = origin
        if preserve_snapshots and capture_changed:
            self._params_ics_tab._origin = origin  # noqa: SLF001
        else:
            self._params_ics_tab.set_origin(origin)
        self._params_ics_tab.set_parameter_choices(
            **self._parameter_choices_for_origin()
        )
        self._project_result_identity_labels()
        self.project_setup_snapshot(setup)

    def install_initial_snapshot(self, snapshot: FitSetupSnapshot) -> None:
        self._check_snapshot(snapshot)
        self._initial_setup = snapshot
        self._snapshot_capture_id = snapshot.origin_capture_id
        self._params_ics_tab.install_initial_snapshot(snapshot)

    def restore_snapshot(self, snapshot: FitSetupSnapshot) -> FitSetupSnapshot:
        self._check_snapshot(snapshot)
        # Initial/Last Fit are whole-setup replacements.  Rebuild the first
        # presentation page so every target, sampling, weight, IC, and raw
        # validation overlay is cleared without altering the retained reset
        # snapshots themselves.
        self._replace_origin_dependent_presentation(self._origin)
        self._invalid_setup_edits.clear()
        self._execution_invalid_raw.clear()
        self._execution_invalid_messages.clear()
        self._pending_target_changes = False
        self._clear_resolved_auto_modes()
        self._params_ics_tab._clear_invalid_numeric_state()  # noqa: SLF001
        self.project_setup_snapshot(snapshot)
        return snapshot

    def project_setup_snapshot(self, snapshot: FitSetupSnapshot) -> None:
        """Project accepted baselines while retaining unrelated local edits."""

        self._check_snapshot(snapshot)
        self._projected_setup = snapshot
        self._execution_projecting = True
        try:
            self._species_table.project_setup_snapshot(snapshot)
            self._params_ics_tab.project_setup_snapshot(snapshot)
            if self._initial_setup is not None:
                self._params_ics_tab.install_initial_snapshot(
                    self._initial_setup
                )
            self._params_ics_tab.set_last_fit_snapshot(self._last_fit_setup)
            self._params_ics_tab.set_selected_dataset_ids(
                self._included_dataset_ids(snapshot)
            )
            self._project_execution_table(snapshot)
            self._populate_dataset_list(snapshot)
        finally:
            self._execution_projecting = False
        self._species_table.refresh_validity_ui()
        self._refresh_setup_validity()

    def accept_setup_edits(
        self,
        snapshot: FitSetupSnapshot,
        intents: tuple[object, ...],
    ) -> None:
        if not isinstance(intents, tuple):
            raise TypeError("Fit setup acknowledgement must carry a tuple.")
        self._projected_setup = snapshot
        self._invalid_setup_edits.difference_update(
            self._intent_key(intent) for intent in intents
        )
        # Leaf projections update accepted baselines.  Their keyed pending/raw
        # overlays survive, and synchronous Apply handlers acknowledge only
        # the represented fields after this owner callback returns.
        self.project_setup_snapshot(snapshot)
        if self._pending_target_changes:
            self._set_species_footer_dirty(True)

    def reject_setup_edits(
        self,
        intents: tuple[object, ...],
        message: str,
    ) -> None:
        if not isinstance(intents, tuple):
            raise TypeError("Fit setup rejection must carry a tuple.")
        self._invalid_setup_edits.update(
            self._intent_key(intent) for intent in intents
        )
        self.set_status(str(message))
        self._species_table.refresh_validity_ui()
        self._refresh_setup_validity()

    @property
    def has_invalid_setup_edits(self) -> bool:
        return bool(
            self._invalid_setup_edits
            or self._execution_invalid_raw
            or self._data_tab.invalid_editor_keys
            or self._data_sampling_validation_errors()
            or self._species_table.invalid_editor_keys
            or self._params_ics_tab.has_invalid_edits
        )

    @staticmethod
    def _intent_key(intent: object) -> tuple[str, ...]:
        if isinstance(intent, FitDatasetSetupEdit):
            return ("dataset", intent.dataset.dataset_id)
        if isinstance(intent, FitDatasetWeightModeSetupEdit):
            return ("dataset_weight_mode",)
        if isinstance(intent, FitParameterSetupEdit):
            return ("parameter", intent.parameter.name)
        if isinstance(intent, FitInitialSetupEdit):
            return ("initial", intent.initial.set_id, intent.initial.species)
        if isinstance(intent, FitDefaultsSetupEdit):
            return ("defaults",)
        return ("unknown", type(intent).__name__)

    def _check_snapshot(self, snapshot: FitSetupSnapshot) -> None:
        self._check_snapshot_origin(snapshot, self._origin)

    @staticmethod
    def _rebase_retained_snapshot(
        snapshot: FitSetupSnapshot | None,
        origin: FitOriginSnapshot,
    ) -> FitSetupSnapshot | None:
        if snapshot is None:
            return None
        return replace(
            snapshot,
            origin_tab_id=origin.tab_id,
            origin_capture_id=origin.capture_id,
        )

    @staticmethod
    def _check_snapshot_origin(
        snapshot: FitSetupSnapshot,
        origin: FitOriginSnapshot,
    ) -> None:
        if not isinstance(snapshot, FitSetupSnapshot):
            raise TypeError("snapshot must be FitSetupSnapshot.")
        if (
            snapshot.origin_tab_id != origin.tab_id
            or snapshot.origin_capture_id != origin.capture_id
        ):
            raise ValueError("Fit snapshot does not match this origin.")

    @staticmethod
    def _target_signature(
        snapshot: FitSetupSnapshot,
    ) -> tuple[tuple[object, ...], ...]:
        return tuple(
            (
                item.dataset_id,
                item.weight,
                item.target_species,
                item.target_weights,
                item.observation_targets,
            )
            for item in snapshot.datasets
        )

    def _on_leaf_setup_edits(self, intents: object) -> None:
        if self._execution_projecting:
            return
        if not isinstance(intents, tuple) or not intents:
            self.set_status("Fit setup edits must be a nonempty tuple.")
            return
        setup = self._projected_setup
        target_change_requested = False
        if setup is not None:
            for intent in intents:
                if isinstance(intent, FitDatasetSetupEdit):
                    current = next(
                        (
                            item
                            for item in setup.datasets
                            if item.dataset_id == intent.dataset.dataset_id
                        ),
                        None,
                    )
                    if current is not None and (
                        self._target_signature_for_dataset(current)
                        != self._target_signature_for_dataset(intent.dataset)
                    ):
                        self._pending_target_changes = True
                        target_change_requested = True
        if target_change_requested:
            self._refresh_setup_validity()
        self.setupEditsRequested.emit(intents)

    @staticmethod
    def _target_signature_for_dataset(
        dataset: FitDatasetLocalSettings,
    ) -> tuple[object, ...]:
        return (
            dataset.weight,
            dataset.target_species,
            dataset.target_weights,
            dataset.observation_targets,
        )

    def _refresh_setup_validity(self) -> None:
        invalid = self.has_invalid_setup_edits
        pending_targets = self.has_pending_target_changes
        ready = (
            self._projected_setup is not None
            and not invalid
            and not pending_targets
        )
        self._run_button.setEnabled(ready and not self._is_executing())
        reason = ""
        if invalid:
            reason = "Correct invalid Fit editor values before running."
        elif pending_targets:
            reason = (
                "Apply or revert pending Fit target and weight edits before running."
            )
        self._run_block_reason_label.setVisible(bool(reason))
        self._run_block_reason_label.setText(reason)
        self._refresh_apply_button()

    def _data_sampling_validation_errors(self) -> tuple[str, ...]:
        pending_by_dataset = getattr(
            self._data_tab,
            "_sampling_pending_by_dataset",
            {},
        )
        if not isinstance(pending_by_dataset, Mapping):
            return ()
        return tuple(
            error
            for dataset_id, config in pending_by_dataset.items()
            for error in (
                self._data_tab.sampling_validation_error(
                    dataset_id=str(dataset_id),
                    config=dict(config),
                ),
            )
            if error
        )

    def _set_species_footer_dirty(self, dirty: bool) -> None:
        footer = getattr(self._species_table, "_footer", None)
        if footer is not None:
            footer.set_dirty(bool(dirty))

    def _populate_dataset_list(self, snapshot: FitSetupSnapshot) -> None:
        by_id = {item.dataset_id: item for item in self._origin.mapped_datasets}
        entries = [
            {
                "id": dataset.dataset_id,
                "label": by_id.get(dataset.dataset_id, None).display_name
                if dataset.dataset_id in by_id
                else dataset.dataset_id,
                "include": dataset.included,
                "selected_species": list(dataset.target_species),
            }
            for dataset in snapshot.datasets
        ]
        self._data_targets_tab.unified_list.populate(entries)
        selected = self._data_targets_tab.unified_list.selected_dataset_id()
        if selected:
            self._species_table.load_for_dataset(selected)
            self._data_tab.select_dataset(selected)

    def _project_execution_table(self, snapshot: FitSetupSnapshot) -> None:
        table = self._execution_table
        table.blockSignals(True)
        try:
            table.setRowCount(len(snapshot.datasets))
            mapped_by_id = {
                item.dataset_id: item for item in self._origin.mapped_datasets
            }
            for row, dataset in enumerate(snapshot.datasets):
                mapped = mapped_by_id.get(dataset.dataset_id)
                identity = QtWidgets.QTableWidgetItem(
                    mapped.display_name if mapped is not None else dataset.dataset_id
                )
                identity.setData(QtCore.Qt.ItemDataRole.UserRole, dataset.dataset_id)
                identity.setFlags(
                    QtCore.Qt.ItemFlag.ItemIsEnabled
                    | QtCore.Qt.ItemFlag.ItemIsSelectable
                )
                table.setItem(row, 0, identity)
                set_item = QtWidgets.QTableWidgetItem(
                    mapped.set_name if mapped is not None else ""
                )
                set_item.setFlags(
                    QtCore.Qt.ItemFlag.ItemIsEnabled
                    | QtCore.Qt.ItemFlag.ItemIsSelectable
                )
                table.setItem(row, 1, set_item)
                solver = QtWidgets.QComboBox(table)
                solver.addItems(list(SIMULATION_SOLVERS))
                solver.setCurrentText(dataset.execution.solver)
                solver.currentTextChanged.connect(
                    lambda _text, r=row: self._emit_execution_edit(r)
                )
                table.setCellWidget(row, 2, solver)
                for column, value in (
                    (3, dataset.execution.t_start),
                    (4, dataset.execution.t_end),
                    (5, dataset.execution.integration_points),
                ):
                    item = QtWidgets.QTableWidgetItem(str(value))
                    item.setData(QtCore.Qt.ItemDataRole.UserRole, dataset.dataset_id)
                    field = self._execution_field(column)
                    key = (dataset.dataset_id, field)
                    if key in self._execution_invalid_raw:
                        item.setText(self._execution_invalid_raw[key])
                        self._set_execution_item_invalid(
                            item,
                            self._execution_invalid_messages.get(
                                key,
                                "Invalid execution value.",
                            ),
                        )
                    table.setItem(row, column, item)
        finally:
            table.blockSignals(False)

    @staticmethod
    def _execution_field(column: int) -> str:
        return {
            3: "t_start",
            4: "t_end",
            5: "integration_points",
        }[int(column)]

    @staticmethod
    def _set_execution_item_invalid(
        item: QtWidgets.QTableWidgetItem,
        message: str,
    ) -> None:
        item.setBackground(QtGui.QColor(255, 200, 200))
        item.setForeground(QtGui.QColor(80, 0, 0))
        item.setToolTip(str(message))

    @staticmethod
    def _clear_execution_item_invalid(
        item: QtWidgets.QTableWidgetItem,
    ) -> None:
        item.setBackground(QtGui.QBrush())
        item.setForeground(QtGui.QBrush())
        item.setToolTip("")

    def _emit_execution_edit(
        self,
        row: int,
        *,
        changed_column: int | None = None,
    ) -> None:
        if self._execution_projecting or self._projected_setup is None:
            return
        table = self._execution_table
        identity = table.item(row, 0)
        if identity is None:
            return
        dataset_id = str(identity.data(QtCore.Qt.ItemDataRole.UserRole) or "")
        current = next(
            item
            for item in self._projected_setup.datasets
            if item.dataset_id == dataset_id
        )
        solver = table.cellWidget(row, 2)
        parsed: dict[str, float | int] = {}
        errors: list[str] = []
        for column in (3, 4, 5):
            item = table.item(row, column)
            if item is None:
                continue
            field = self._execution_field(column)
            key = (dataset_id, field)
            raw = item.text()
            try:
                if field == "integration_points":
                    value: float | int = int(raw)
                    if value <= 0:
                        raise ValueError("Integration points must be positive.")
                else:
                    value = float(raw)
                    if not np.isfinite(value):
                        raise ValueError(f"{field} must be finite.")
            except (TypeError, ValueError, OverflowError) as exc:
                message = str(exc) or f"{field} is invalid."
                self._execution_invalid_raw[key] = raw
                self._execution_invalid_messages[key] = message
                self._set_execution_item_invalid(item, message)
                errors.append(message)
            else:
                parsed[field] = value
                self._execution_invalid_raw.pop(key, None)
                self._execution_invalid_messages.pop(key, None)
                self._clear_execution_item_invalid(item)
        if errors:
            self.set_status(f"Fit execution settings are invalid: {errors[0]}")
            self._refresh_setup_validity()
            return
        try:
            execution = replace(
                current.execution,
                solver=(
                    solver.currentText()
                    if isinstance(solver, QtWidgets.QComboBox)
                    else current.execution.solver
                ),
                t_start=float(parsed["t_start"]),
                t_end=float(parsed["t_end"]),
                integration_points=int(parsed["integration_points"]),
            )
            edited = replace(current, execution=execution)
            intent = FitDatasetSetupEdit(
                origin_tab_id=self._origin.tab_id,
                origin_capture_id=self._origin.capture_id,
                dataset=edited,
            )
        except (TypeError, ValueError, OverflowError) as exc:
            column = changed_column if changed_column in {3, 4, 5} else 3
            item = table.item(row, column)
            field = self._execution_field(column)
            key = (dataset_id, field)
            if item is not None:
                self._execution_invalid_raw[key] = item.text()
                self._execution_invalid_messages[key] = str(exc)
                self._set_execution_item_invalid(item, str(exc))
            self.set_status(f"Fit execution settings are invalid: {exc}")
            self._refresh_setup_validity()
            return
        self._refresh_setup_validity()
        self._on_leaf_setup_edits((intent,))

    def _on_execution_item_changed(
        self,
        item: QtWidgets.QTableWidgetItem,
    ) -> None:
        self._emit_execution_edit(
            int(item.row()),
            changed_column=int(item.column()),
        )

    # ------------------------------------------------------------------
    # Fit-local data and setup presentation adapters
    # ------------------------------------------------------------------

    def _mechanism_species(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                species
                for mapped in self._origin.mapped_datasets
                for species, _value in mapped.initial_values
            )
        )

    @staticmethod
    def _included_dataset_ids(
        snapshot: FitSetupSnapshot,
    ) -> tuple[str, ...]:
        return tuple(
            item.dataset_id for item in snapshot.datasets if item.included
        )

    def _mapped_dataset(self, dataset_id: str):
        return next(
            (
                item
                for item in self._origin.mapped_datasets
                if item.dataset_id == str(dataset_id)
            ),
            None,
        )

    def _selected_target_times_for_dataset(
        self,
        dataset_id: str,
    ) -> dict[str, np.ndarray]:
        mapped = self._mapped_dataset(dataset_id)
        if mapped is None:
            return {}
        captured_by_raw_id = {
            observation.raw_observation_id: observation
            for observation in mapped.observations
        }
        dataset = (
            next(
                (
                    item
                    for item in (self._projected_setup.datasets if self._projected_setup else ())
                    if item.dataset_id == dataset_id
                ),
                None,
            )
        )
        selected = self._observation_targets(mapped, dataset)
        selected_targets = (
            set(dataset.target_species)
            if dataset is not None
            else {target for _raw_id, target in selected}
        )
        target_times: dict[str, np.ndarray] = {}
        for raw_observation_id, model_target in selected:
            target = str(model_target or "").strip()
            observation = captured_by_raw_id.get(str(raw_observation_id))
            if (
                not target
                or target not in selected_targets
                or observation is None
            ):
                continue
            target_times.setdefault(
                target,
                np.asarray(observation.t, dtype=float).reshape(-1),
            )
        return target_times

    def _available_targets_for_dataset(self, dataset_id: str) -> list[str]:
        mapped = self._mapped_dataset(dataset_id)
        if mapped is None:
            return []
        dataset = self._projected_dataset(dataset_id)
        return list(
            dict.fromkeys(
                target
                for _raw_observation_id, target in self._observation_targets(
                    mapped,
                    dataset,
                )
            )
        )

    def _full_series_for_dataset(self, dataset_id: str) -> dict[str, np.ndarray]:
        mapped = self._mapped_dataset(dataset_id)
        if mapped is None:
            return {}
        dataset = self._projected_dataset(dataset_id)
        target_by_raw_id = dict(self._observation_targets(mapped, dataset))
        return {
            target_by_raw_id[observation.raw_observation_id]: np.asarray(
                observation.y,
                dtype=float,
            )
            for observation in mapped.observations
            if observation.raw_observation_id in target_by_raw_id
        }

    def _projected_dataset(
        self,
        dataset_id: str,
    ) -> FitDatasetLocalSettings | None:
        setup = self._projected_setup
        if setup is None:
            return None
        return next(
            (
                item
                for item in setup.datasets
                if item.dataset_id == dataset_id
            ),
            None,
        )

    @staticmethod
    def _observation_targets(
        mapped: FitMappedDatasetSnapshot,
        dataset: FitDatasetLocalSettings | None,
    ) -> tuple[tuple[str, str], ...]:
        overrides = (
            dict(dataset.observation_targets)
            if dataset is not None
            else {}
        )
        return tuple(
            (
                observation.raw_observation_id,
                overrides.get(
                    observation.raw_observation_id,
                    observation.model_species,
                ),
            )
            for observation in mapped.observations
        )

    def _applied_targets_for_dataset(self, dataset_id: str) -> list[str]:
        setup = self._projected_setup
        if setup is None:
            mapped = self._mapped_dataset(dataset_id)
            return [] if mapped is None else [item.model_species for item in mapped.observations]
        dataset = next(
            (item for item in setup.datasets if item.dataset_id == dataset_id),
            None,
        )
        return [] if dataset is None else list(dataset.target_species)

    def _sampling_applied_config(self, dataset_id: str) -> dict[str, float | int | str]:
        setup = self._projected_setup
        dataset = (
            next(
                (item for item in setup.datasets if item.dataset_id == dataset_id),
                None,
            )
            if setup is not None
            else None
        )
        if dataset is None:
            return self._sampling_default_config(
                self._selected_target_times_for_dataset(dataset_id)
            )
        return {
            "t_min": dataset.sample_t_min,
            "t_max": dataset.sample_t_max,
            "n_points": dataset.sample_points or 0,
            "x_name": dataset.parametric.x_name if dataset.parametric else "t",
            "x_mapping_mode": (
                dataset.parametric.requested_mode if dataset.parametric else "auto"
            ),
        }

    @staticmethod
    def _sampling_default_config(
        target_times: Mapping[str, object],
    ) -> dict[str, float | int | str]:
        finite_ranges: list[tuple[float, float]] = []
        for values in target_times.values():
            array = np.asarray(values, dtype=float).reshape(-1)
            finite = array[np.isfinite(array)]
            if finite.size:
                finite_ranges.append((float(np.min(finite)), float(np.max(finite))))
        return {
            "t_min": min(low for low, _high in finite_ranges)
            if finite_ranges
            else 0.0,
            "t_max": max(high for _low, high in finite_ranges)
            if finite_ranges
            else 0.0,
            "n_points": 0,
            "x_name": "t",
            "x_mapping_mode": "auto",
        }

    def _modeled_series(self) -> set[str]:
        names = {
            observation.model_species
            for mapped in self._origin.mapped_datasets
            for observation in mapped.observations
        }
        names.update(
            species
            for mapped in self._origin.mapped_datasets
            for species, _value in mapped.initial_values
        )
        if self._projected_setup is not None:
            names.update(item.name for item in self._projected_setup.local_observables)
        return names

    def _on_dataset_selected(self, dataset_id: str) -> None:
        if not dataset_id:
            return
        self._species_table.load_for_dataset(dataset_id)
        self._data_tab.select_dataset(dataset_id)

    def _on_dataset_include_changed(
        self,
        _row: int,
        dataset_id: str,
        included: bool,
    ) -> None:
        setup = self._projected_setup
        if setup is None:
            return
        current = next(
            (item for item in setup.datasets if item.dataset_id == dataset_id),
            None,
        )
        if current is None:
            return
        self._on_leaf_setup_edits(
            (
                FitDatasetSetupEdit(
                    origin_tab_id=self._origin.tab_id,
                    origin_capture_id=self._origin.capture_id,
                    dataset=replace(current, included=bool(included)),
                ),
            )
        )

    def _on_sampling_applied(self, dataset_id: str, config: dict) -> None:
        setup = self._projected_setup
        if setup is None:
            return
        current = next(
            (item for item in setup.datasets if item.dataset_id == dataset_id),
            None,
        )
        if current is None:
            return
        try:
            x_name = str(config.get("x_name") or "t").strip() or "t"
            requested = str(config.get("x_mapping_mode") or "auto").strip()
            requested = requested.lower().replace("-", "_").replace(" ", "_")
            if requested not in {"auto", "monotone", "time_guided"}:
                requested = "auto"
            parametric = None
            if x_name != "t":
                parametric = FitParametricSettings(
                    dataset_id=dataset_id,
                    x_name=x_name,
                    requested_mode=requested,
                    resolved_mode=(
                        None if requested == "auto" else requested
                    ),
                )
            edited = replace(
                current,
                sample_t_min=float(config.get("t_min", current.sample_t_min)),
                sample_t_max=float(config.get("t_max", current.sample_t_max)),
                sample_points=(
                    None
                    if int(config.get("n_points", 0)) == 0
                    else int(config["n_points"])
                ),
                parametric=parametric,
            )
            intent = FitDatasetSetupEdit(
                origin_tab_id=self._origin.tab_id,
                origin_capture_id=self._origin.capture_id,
                dataset=edited,
            )
        except (TypeError, ValueError, OverflowError) as exc:
            self.set_status(f"Sampling edit is invalid: {exc}")
            return
        self._on_leaf_setup_edits((intent,))

    def _on_add_observable_requested(self, selection: object) -> None:
        if not isinstance(selection, Mapping):
            self.set_status("Fit-local observable selection is invalid.")
            return
        required = (
            "name",
            "expr",
            "dataset_id",
            "raw_observation_id",
            "scalar_scope",
        )
        if any(
            key not in selection
            or not isinstance(selection[key], str)
            or not selection[key].strip()
            for key in required
        ):
            self.set_status("Fit-local observable selection is incomplete.")
            return
        name = selection["name"]
        expression = selection["expr"]
        dataset_id = selection["dataset_id"]
        raw_id = selection["raw_observation_id"]
        scope = selection["scalar_scope"]
        self.addObservableRequested.emit(
            name,
            expression,
            dataset_id,
            raw_id,
            scope,
        )

    # ------------------------------------------------------------------
    # Current public window behavior
    # ------------------------------------------------------------------

    @property
    def has_pending_target_changes(self) -> bool:
        return bool(
            self._pending_target_changes
            or self._species_table.has_pending_target_changes
        )

    def commit_current_target_selection(self) -> None:
        self._pending_target_changes = False
        self._set_species_footer_dirty(False)
        self._refresh_setup_validity()

    def _is_executing(self) -> bool:
        return bool(self._executing)

    def _request_execute(self) -> None:
        if self.has_invalid_setup_edits:
            self.set_status(
                "Fit setup contains an invalid edit; correct it before running."
            )
            return
        if self.has_pending_target_changes:
            self.set_status(
                "Apply or revert pending Fit target and weight edits before running."
            )
            return
        self.executeRequested.emit()

    def _toggle_pause(self) -> None:
        if not self._executing:
            return
        if self._paused:
            self._resume_from_button()
            return
        self._paused = True
        self._pause_button.setEnabled(False)
        self._resume_button.setEnabled(True)
        self.pauseRequested.emit()

    def _resume_from_button(self) -> None:
        if not self._executing:
            return
        self._paused = False
        self._pause_button.setEnabled(True)
        self._resume_button.setEnabled(False)
        self.resumeRequested.emit()

    def set_executing(self, executing: bool) -> None:
        self._executing = bool(executing)
        if self._executing:
            self._paused = False
            self._run_results_tab.set_view_autorange_locked(True)
            self._run_results_tab.set_live_curves(())
            self._copy_button.setEnabled(False)
        else:
            self._paused = False
            self._run_results_tab.set_view_autorange_locked(False)
        self._cancel_button.setEnabled(self._executing)
        self._pause_button.setEnabled(self._executing)
        self._resume_button.setEnabled(False)
        self._species_table.set_running_state(self._executing)
        self._params_ics_tab.set_running_state(self._executing)
        self._params_ics_tab.setEnabled(not self._executing)
        self._data_targets_tab.unified_list.set_running_state(self._executing)
        self._data_tab.setEnabled(not self._executing)
        self._project_defaults_button.setEnabled(not self._executing)
        self._refresh_setup_validity()
        self._refresh_apply_button()

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
            key=lambda item: max(len(item.dataset_id), len(item.set_id)),
            reverse=True,
        ):
            rendered = rendered.replace(mapped.dataset_id, mapped.display_name)
            rendered = rendered.replace(mapped.set_id, mapped.set_name)
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
        self._run_results_tab.set_live_best(update)

    def set_live_curves(self, curves: tuple[FitCurveDataset, ...]) -> None:
        values = tuple(curves)
        if any(not isinstance(item, FitCurveDataset) for item in values):
            raise TypeError("curves must contain FitCurveDataset values.")
        self._run_results_tab.set_live_curves(values)

    def set_resolved_auto_modes(
        self,
        modes: tuple[tuple[str, str], ...],
    ) -> None:
        if any(mode not in {"monotone", "time_guided"} for _id, mode in modes):
            raise ValueError("Resolved Auto modes are invalid.")
        normalized = tuple((str(dataset_id), str(mode)) for dataset_id, mode in modes)
        if len(normalized) != len({dataset_id for dataset_id, _mode in normalized}):
            raise ValueError("Resolved Auto modes repeat a dataset ID.")
        self._resolved_auto_modes = normalized
        self._data_tab.set_resolved_auto_modes(normalized)

    def _clear_resolved_auto_modes(self) -> None:
        self._resolved_auto_modes = ()
        self._data_tab.set_resolved_auto_modes({})

    def set_run_identity(self, identity: FitRunIdentity) -> None:
        if not isinstance(identity, FitRunIdentity):
            raise TypeError("identity must be a FitRunIdentity.")
        self._run_identity = identity
        self._run_results_tab.set_run_identity(identity)
        self._copy_run_json.setEnabled(True)
        self._results_summary_button.setEnabled(True)

    def set_candidate(
        self,
        candidate: FitCandidate,
        *,
        available_scopes: frozenset[str],
        unavailable_reasons: Mapping[str, str],
    ) -> bool:
        if not isinstance(candidate, FitCandidate):
            raise TypeError("candidate must be FitCandidate.")
        known = {
            "parameters",
            "initial_concentrations",
            "simulation_setup",
            "observable_definitions",
        }
        available = frozenset(str(scope) for scope in available_scopes)
        reasons = {
            str(scope): str(reason or "") for scope, reason in unavailable_reasons.items()
        }
        if (
            not available.issubset(known)
            or set(reasons) != known - available
            or any(not reason for reason in reasons.values())
        ):
            raise ValueError("Fit Apply feasibility is invalid.")
        self._candidate = candidate
        self._candidate_available_scopes = available
        self._candidate_unavailable_reasons = reasons
        self._last_fit_setup = candidate.setup_snapshot
        self._params_ics_tab.set_last_fit_snapshot(self._last_fit_setup)
        self._run_identity = candidate.run_identity
        self.set_resolved_auto_modes(candidate.resolved_auto_modes)
        self._copy_run_json.setEnabled(True)
        self._results_summary_button.setEnabled(True)
        rendered = self._run_results_tab.set_candidate(
            candidate,
            available_scopes=available,
            unavailable_reasons=reasons,
        )
        for checkbox, scope in self._apply_controls():
            checkbox.setEnabled(scope in available)
            checkbox.setChecked(scope in available)
            checkbox.setToolTip(reasons.get(scope, ""))
        self._refresh_apply_button()
        self._set_copy_available(self._run_results_tab.copy_eligible())
        self._tabs.setCurrentIndex(2)
        return bool(rendered)

    def clear_candidate(self) -> None:
        self._candidate = None
        self._candidate_available_scopes = frozenset()
        self._candidate_unavailable_reasons = {}
        self._run_results_tab.clear_candidate()
        self._run_identity = None
        self._clear_resolved_auto_modes()
        for checkbox, _scope in self._apply_controls():
            checkbox.setChecked(False)
            checkbox.setEnabled(False)
            checkbox.setToolTip("")
        self._apply_button.setEnabled(False)
        self._copy_button.setEnabled(False)
        self._copy_run_json.setEnabled(False)
        self._results_summary_button.setEnabled(False)

    def _apply_controls(self):
        return (
            (self._apply_parameters, "parameters"),
            (self._apply_initials, "initial_concentrations"),
            (self._apply_setup, "simulation_setup"),
            (self._apply_observables, "observable_definitions"),
        )

    def _emit_apply(self) -> None:
        scopes = frozenset(
            scope
            for checkbox, scope in self._apply_controls()
            if checkbox.isEnabled() and checkbox.isChecked()
        )
        if not scopes:
            self.set_status("Select at least one Apply scope.")
            return
        self.applyRequested.emit(scopes)

    def _refresh_apply_button(self, *_args: object) -> None:
        selected_available = any(
            checkbox.isEnabled() and checkbox.isChecked()
            for checkbox, _scope in self._apply_controls()
        )
        self._apply_button.setEnabled(
            selected_available
            and not self._executing
            and not self.has_pending_target_changes
        )

    def _emit_project_defaults(self) -> None:
        setup = self._projected_setup
        if setup is not None:
            self.projectDefaultsRequested.emit(setup.defaults)

    def _copy_run_identity_json(self) -> None:
        if self._run_identity is None:
            self.set_status("No completed Fit run is available to copy.")
            return
        clipboard = QtWidgets.QApplication.clipboard()
        clipboard.setText(self._run_identity.readable_json())
        self.set_status("Copied the complete Fit run JSON.")

    def _set_copy_available(self, available: bool) -> None:
        self._copy_button.setEnabled(bool(available) and not self._executing)

    def _current_results_dark_mode(self) -> bool:
        for candidate in (self.parentWidget(), self.parent()):
            if candidate is not None and hasattr(candidate, "_dark_mode"):
                return bool(getattr(candidate, "_dark_mode"))
            if candidate is not None:
                window_color = candidate.palette().color(
                    QtGui.QPalette.ColorRole.Window
                )
                return int(window_color.lightness()) < 128
        app = QtWidgets.QApplication.instance()
        if app is None:
            return False
        window_color = app.palette().color(QtGui.QPalette.ColorRole.Window)
        return int(window_color.lightness()) < 128

    def changeEvent(self, event: QtCore.QEvent) -> None:  # noqa: N802
        super().changeEvent(event)
        if hasattr(self, "_run_results_tab") and event.type() in (
            QtCore.QEvent.Type.PaletteChange,
            QtCore.QEvent.Type.ApplicationPaletteChange,
            QtCore.QEvent.Type.StyleChange,
        ):
            self._run_results_tab.set_dark_mode(self._current_results_dark_mode())

    def force_close_after_cleanup(self) -> None:
        self._closing_after_cleanup = True
        self.close()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802
        if self._closing_after_cleanup:
            event.accept()
            return
        if self._close_handler(time.monotonic() + 2.0):
            self._closing_after_cleanup = True
            event.accept()
        else:
            self.set_status("Fit is still stopping; close was deferred.")
            event.ignore()
