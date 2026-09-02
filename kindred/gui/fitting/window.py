"""Physical Fit presentation bound to the current typed session owner."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace

from PySide6 import QtCore, QtGui, QtWidgets

from kindred.gui.fitting.contracts import (
    FitParametricSettings,
)
from kindred.core.simulation_failure import FactualCause
from kindred.core.analysis.fit_evaluation_outcome import ScientificUnavailable
from kindred.gui.fitting.data_tab import DataTab
from kindred.gui.fitting.data_targets_tab import DataTargetsTab
from kindred.gui.fitting.launch import (
    FitOriginSnapshot,
    FitDatasetLocalSettings,
    FitSetupSnapshot,
    fit_setup_readiness,
    selected_observations,
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
    complete setup replacements.
    """

    executeRequested = QtCore.Signal()
    applyRequested = QtCore.Signal()
    sharedSeedRequested = QtCore.Signal(str)
    cancelRequested = QtCore.Signal()
    pauseRequested = QtCore.Signal()
    resumeRequested = QtCore.Signal()
    addObservableRequested = QtCore.Signal(str, str, object, str)
    setupSnapshotRequested = QtCore.Signal(object)
    resetRequested = QtCore.Signal(str)
    projectDefaultsRequested = QtCore.Signal(object)
    resultsSummaryRequested = QtCore.Signal()

    def __init__(
        self,
        origin: FitOriginSnapshot,
        *,
        close_handler: Callable[[], bool],
        color_resolver: Callable[[str], object] | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        if not isinstance(origin, FitOriginSnapshot):
            raise TypeError("origin must be FitOriginSnapshot.")
        self._origin = origin
        self._close_handler = close_handler
        self._color_resolver = color_resolver
        self._closing = False
        self._closing_after_cleanup = False
        self._paused = False
        self._projected_setup: FitSetupSnapshot | None = None
        self._executing = False
        self._setup_projecting = False

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
            settings_getter=self._fit_settings_for_dataset,
            default_settings_getter=self._fit_default_settings_for_dataset,
            retained_observations_getter=self._retained_observations_for_dataset,
            parent=self,
        )
        self._data_targets_tab = DataTargetsTab(
            data_settings_panel=self._data_tab,
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
        self._run_results_tab.set_dark_mode(self._current_results_dark_mode())

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
            self.resultsSummaryRequested
        )
        actions.addWidget(self._results_summary_button)

        self._copy_button = QtWidgets.QPushButton("Copy Fit Curves", footer)
        self._copy_button.setEnabled(False)
        self._copy_button.clicked.connect(
            self._run_results_tab.copy_all_to_clipboard
        )
        actions.addWidget(self._copy_button)
        self._apply_button = QtWidgets.QPushButton("Apply Fit to Project", footer)
        self._apply_button.setEnabled(False)
        self._apply_button.clicked.connect(self.applyRequested)
        actions.addWidget(self._apply_button)
        self._shared_seed_combo = QtWidgets.QComboBox(footer)
        self._shared_seed_combo.addItem("Arithmetic mean", "arithmetic")
        self._shared_seed_combo.addItem("Geometric mean", "geometric")
        self._shared_seed_combo.setEnabled(False)
        actions.addWidget(self._shared_seed_combo)
        self._shared_seed_button = QtWidgets.QPushButton(
            "Configure Shared Fit", footer
        )
        self._shared_seed_button.setEnabled(False)
        self._shared_seed_button.clicked.connect(self._request_shared_seed)
        actions.addWidget(self._shared_seed_button)
        self._project_defaults_button = QtWidgets.QPushButton(
            "Save Optimizer Settings as Project Defaults", footer
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

        self._data_tab.fitSettingsApplied.connect(self._on_data_settings_applied)
        self._data_tab.statusMessage.connect(self.set_status)
        self._data_tab.validityChanged.connect(self._refresh_setup_validity)
        self._params_ics_tab.setupSnapshotRequested.connect(
            self._on_leaf_snapshot
        )
        self._params_ics_tab.resetRequested.connect(
            self.resetRequested
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
        self._species_table.setupSnapshotRequested.connect(
            self._on_leaf_snapshot
        )
        self._species_table.validityChanged.connect(
            self._refresh_setup_validity
        )
        self._species_table.statusMessage.connect(self.set_status)
        dataset_list = self._data_targets_tab.unified_list
        dataset_list.datasetIncludeChanged.connect(
            self._on_dataset_include_changed
        )
    def _parameter_choices_for_origin(self) -> dict[str, object]:
        return {
            "dataset_ids": tuple(
                item.dataset_id for item in self._origin.mapped_datasets
            ),
            "available_species": self._mechanism_species(),
        }

    def _open_mechanism_inspector(self) -> None:
        from kindred.gui.widgets.mechanism_inspector import MechanismInspectorDialog

        MechanismInspectorDialog(
            self._origin.prepared_dsl,
            self,
        ).exec()

    def set_initial_reset_available(self, snapshot: FitSetupSnapshot) -> None:
        self._check_snapshot(snapshot)
        self._params_ics_tab.set_initial_reset_available(snapshot)

    def set_last_fit_reset_available(self, snapshot: FitSetupSnapshot | None) -> None:
        """Project Last Fit reset availability without retaining the snapshot."""

        if snapshot is not None:
            self._check_snapshot(snapshot)
        self._params_ics_tab.set_last_fit_reset_available(snapshot)

    def set_last_fit_presentation(self) -> None:
        """Label retained Last Fit values as ordinary state."""

        self._params_ics_tab.set_last_fit_presentation()

    def clear_pending_setup_edits(self) -> None:
        """Discard every leaf draft and refresh the complete Fit surface."""

        self._params_ics_tab.clear_pending_setup_edits()
        self._species_table.clear_pending_edits()
        self._data_tab.clear_pending_edits()
        self._refresh_setup_validity()

    def project_setup_snapshot(self, snapshot: FitSetupSnapshot) -> None:
        """Project accepted baselines while retaining unrelated local edits."""

        self._check_snapshot(snapshot)
        self._projected_setup = snapshot
        self._populate_dataset_list(snapshot)
        selected_dataset_id = (
            self._data_targets_tab.unified_list.selected_dataset_id()
        )
        self._setup_projecting = True
        try:
            self._params_ics_tab.project_setup_snapshot(snapshot)
            self._params_ics_tab.set_selected_dataset_ids(
                self._included_dataset_ids(snapshot)
            )
            self._species_table.project_setup_snapshot(
                snapshot,
                dataset_id=selected_dataset_id,
            )
            if selected_dataset_id:
                self._data_tab.select_dataset(
                    selected_dataset_id,
                    self._data_targets_tab.unified_list.dataset_display_name(
                        selected_dataset_id
                    ),
                )
        finally:
            self._setup_projecting = False
        self._species_table.refresh_validity_ui()
        self._refresh_setup_validity()

    def _on_leaf_snapshot(self, snapshot: FitSetupSnapshot) -> None:
        """Forward one complete leaf-authored setup snapshot."""

        if self._closing or self._setup_projecting:
            return
        self.setupSnapshotRequested.emit(snapshot)

    def _check_snapshot(self, snapshot: FitSetupSnapshot) -> None:
        self._check_snapshot_origin(snapshot, self._origin)

    @staticmethod
    def _check_snapshot_origin(
        snapshot: FitSetupSnapshot,
        origin: FitOriginSnapshot,
    ) -> None:
        if not isinstance(snapshot, FitSetupSnapshot):
            raise TypeError("snapshot must be FitSetupSnapshot.")
        if snapshot.origin is not origin:
            raise ValueError("Fit snapshot does not match this origin.")

    @property
    def has_invalid_setup_edits(self) -> bool:
        setup = self._projected_setup
        data_errors = (
            ()
            if setup is None
            else self._data_tab.active_validation_errors(
                self._included_dataset_ids(setup)
            )
        )
        return bool(
            data_errors
            or self._species_table.invalid_editor_keys
            or self._species_table.has_pending_initial_changes
            or self._params_ics_tab.has_invalid_edits
        )

    def _refresh_setup_validity(self) -> None:
        invalid = self.has_invalid_setup_edits
        pending_targets = self.has_pending_target_changes
        pending_data = self.has_pending_data_changes
        structural = None
        setup = self._projected_setup
        if (
            setup is not None
            and not invalid
            and not pending_targets
            and not pending_data
        ):
            structural = fit_setup_readiness(
                origin=self._origin,
                datasets=setup.datasets,
                parameters=setup.parameters,
                initials=setup.initials,
                global_initials=setup.global_initials,
                initial_scopes=setup.initial_scopes,
            )
        ready = (
            setup is not None
            and not invalid
            and not pending_targets
            and not pending_data
            and structural is None
        )
        self._run_button.setEnabled(ready and not self._is_executing())
        reason = ""
        if invalid:
            reason = "Correct invalid Fit editor values before running."
        elif pending_targets:
            reason = (
                "Apply or revert pending Fit target and weight edits before running."
            )
        elif pending_data:
            reason = "Apply or revert pending Fit data edits before running."
        elif structural is not None:
            reason = structural
        self._run_block_reason_label.setVisible(bool(reason))
        self._run_block_reason_label.setText(reason)

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
        self._data_targets_tab.unified_list.populate(
            entries,
            emit_selection=False,
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

    def _fit_settings_for_dataset(self, dataset_id: str) -> dict[str, object]:
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
            return self._fit_default_settings_for_dataset(dataset_id)
        return {
            "t_start": dataset.t_start,
            "data_range_min": dataset.data_range_min,
            "data_range_max": dataset.data_range_max,
            "maximum_observations_per_target": dataset.maximum_observations_per_target,
            "parametric_x": (
                dataset.parametric.x_name if dataset.parametric is not None else "t"
            ),
            "parametric_mode": (
                dataset.parametric.requested_mode
                if dataset.parametric is not None
                else "auto"
            ),
            "eligible_parametric_x": self._eligible_parametric_x(dataset_id),
        }

    def _eligible_parametric_x(self, dataset_id: str) -> tuple[str, ...]:
        setup = self._projected_setup
        mapped = self._mapped_dataset(dataset_id)
        if setup is None or mapped is None:
            return ()
        dataset = next(
            item for item in setup.datasets if item.dataset_id == dataset_id
        )
        target_by_raw_id = dict(dataset.observation_targets)
        resolved_observations = tuple(
            (
                observation,
                target_by_raw_id.get(observation.raw_observation_id),
            )
            for observation in mapped.observations
            if observation.raw_observation_id in target_by_raw_id
        )
        counts: dict[str, int] = {}
        selected_raw_ids = {
            observation.raw_observation_id
            for observation, target in resolved_observations
            if target in dataset.target_species
        }
        for _observation, target in resolved_observations:
            if target is not None:
                counts[target] = counts.get(target, 0) + 1
        return tuple(
            dict.fromkeys(
                target
                for observation, target in resolved_observations
                if target is not None
                and counts[target] == 1
                and target not in dataset.target_species
                and observation.raw_observation_id not in selected_raw_ids
            )
        )

    def _fit_default_settings_for_dataset(self, dataset_id: str) -> dict[str, object]:
        mapped = self._mapped_dataset(dataset_id)
        if mapped is None:
            return {
                "t_start": 0.0,
                "data_range_min": 0.0,
                "data_range_max": 0.0,
                "maximum_observations_per_target": None,
                "parametric_x": "t",
                "parametric_mode": "auto",
                "eligible_parametric_x": (),
            }
        times = [float(value) for item in mapped.observations for value in item.t]
        return {
            "t_start": mapped.resolved_t_start,
            "data_range_min": min(times, default=mapped.resolved_t_start),
            "data_range_max": max(times, default=mapped.resolved_t_start),
            "maximum_observations_per_target": None,
            "parametric_x": "t",
            "parametric_mode": "auto",
            "eligible_parametric_x": self._eligible_parametric_x(dataset_id),
        }

    def _retained_observations_for_dataset(
        self,
        dataset_id: str,
        config: Mapping[str, object],
    ) -> tuple[object, ...]:
        setup = self._projected_setup
        mapped = self._mapped_dataset(dataset_id)
        if setup is None or mapped is None:
            return ()
        current = next(
            item for item in setup.datasets if item.dataset_id == dataset_id
        )
        edited = self._project_dataset_settings(current, config)
        return selected_observations(mapped, edited)

    @staticmethod
    def _project_dataset_settings(
        current: FitDatasetLocalSettings,
        config: Mapping[str, object],
    ) -> FitDatasetLocalSettings:
        """Project one Data-tab config onto the authoritative dataset shape.

        Preview and Apply must use the same target, weight, and mapping
        projection.  Parametric X is an additional weighted target, never a
        Y target; switching X or returning to Time retires only obsolete
        X-only weights while leaving observation mappings untouched.
        """

        parametric = FittingWindow._parametric_from_config(
            current.dataset_id,
            config,
        )
        x_name = None if parametric is None else parametric.x_name
        target_species = tuple(
            target for target in current.target_species if target != x_name
        )
        weight_by_target = dict(current.target_weights)
        target_weights = tuple(
            (target, weight_by_target.get(target, 1.0))
            for target in target_species
        )
        if x_name is not None:
            target_weights = (
                *target_weights,
                (x_name, weight_by_target.get(x_name, 1.0)),
            )
        return replace(
            current,
            t_start=float(config.get("t_start", current.t_start)),
            data_range_min=float(
                config.get("data_range_min", current.data_range_min)
            ),
            data_range_max=float(
                config.get("data_range_max", current.data_range_max)
            ),
            maximum_observations_per_target=config.get(
                "maximum_observations_per_target",
                current.maximum_observations_per_target,
            ),
            parametric=parametric,
            target_species=target_species,
            target_weights=target_weights,
            observation_targets=current.observation_targets,
        )

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
        self._on_leaf_snapshot(
            replace(
                setup,
                datasets=tuple(
                    replace(item, included=bool(included))
                    if item.dataset_id == dataset_id else item
                    for item in setup.datasets
                ),
            )
        )

    def _on_data_settings_applied(self, dataset_id: str, config: dict) -> None:
        setup = self._projected_setup
        if setup is None:
            return
        current = next(
            (item for item in setup.datasets if item.dataset_id == dataset_id),
            None,
        )
        if current is None:
            return
        expected_fields = {
            "t_start",
            "data_range_min",
            "data_range_max",
            "maximum_observations_per_target",
            "parametric_x",
            "parametric_mode",
        }
        if set(config) != expected_fields:
            self.set_status("Data settings edit carried invalid fields.")
            return
        try:
            edited = self._project_dataset_settings(current, config)
            snapshot = replace(
                setup,
                datasets=tuple(
                    edited if item.dataset_id == edited.dataset_id else item
                    for item in setup.datasets
                ),
            )
        except (TypeError, ValueError, OverflowError) as exc:
            self.set_status(f"Data settings edit is invalid: {exc}")
            return
        self._on_leaf_snapshot(snapshot)

    @staticmethod
    def _parametric_from_config(
        dataset_id: str,
        config: Mapping[str, object],
    ) -> FitParametricSettings | None:
        x_name = str(config.get("parametric_x", "t") or "t")
        if x_name == "t":
            return None
        mode = str(config.get("parametric_mode", "auto") or "auto")
        return FitParametricSettings(
            dataset_id=dataset_id,
            x_name=x_name,
            requested_mode=mode,
            resolved_mode=None if mode == "auto" else mode,
        )

    def _on_add_observable_requested(
        self,
        name: str,
        expression: str,
        observations: object,
        scalar_scope: str,
    ) -> None:
        self.addObservableRequested.emit(
            str(name),
            str(expression),
            observations,
            str(scalar_scope),
        )

    # ------------------------------------------------------------------
    # Current public window behavior
    # ------------------------------------------------------------------

    @property
    def has_pending_target_changes(self) -> bool:
        return self._species_table.has_pending_target_changes

    @property
    def has_pending_data_changes(self) -> bool:
        setup = self._projected_setup
        if setup is None:
            return False
        return self._data_tab.has_pending_edits(
            self._included_dataset_ids(setup)
        )

    def add_observable_conflict(self, dataset_id: str) -> str | None:
        """Expose the target leaf's same-dataset Add Observable guard."""

        return self._species_table.add_observable_conflict(dataset_id)

    def selected_fit_dataset_ids(self) -> tuple[str, ...]:
        """Return the accepted dataset roster used by the observable dialog."""

        return tuple(self._params_ics_tab._selected_dataset_ids)

    def _is_executing(self) -> bool:
        return bool(self._executing)

    def _request_execute(self) -> None:
        if self._closing:
            return
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
        if self.has_pending_data_changes:
            self.set_status(
                "Apply or revert pending Fit data edits before running."
            )
            return
        self.executeRequested.emit()

    def _toggle_pause(self) -> None:
        if self._closing or not self._executing:
            return
        if self._paused:
            self._resume_from_button()
            return
        self._paused = True
        self._pause_button.setEnabled(False)
        self._resume_button.setEnabled(True)
        self.pauseRequested.emit()

    def _resume_from_button(self) -> None:
        if self._closing or not self._executing:
            return
        self._paused = False
        self._pause_button.setEnabled(True)
        self._resume_button.setEnabled(False)
        self.resumeRequested.emit()

    def set_executing(self, executing: bool) -> None:
        was_executing = self._executing
        self._executing = bool(executing)
        if self._executing:
            if not was_executing:
                self._progress.setValue(0)
            self._paused = False
            self._run_results_tab.set_live_curves(())
            self._run_results_tab.set_live_layout_locked(True)
            self._copy_button.setEnabled(False)
            self._apply_button.setEnabled(False)
            self.set_shared_seed_available(False)
        else:
            self._paused = False
            self._run_results_tab.set_live_layout_locked(False)
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

    def set_progress(self, value: int, text: str) -> None:
        self._progress.setValue(max(0, min(100, int(value))))
        self.set_status(text)

    def set_status(self, text: str) -> None:
        if self._closing and not self._closing_after_cleanup:
            self._status.setText("Stopping…")
            return
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
        self._data_tab.set_resolved_auto_modes(normalized)

    def _clear_resolved_auto_modes(self) -> None:
        self._data_tab.set_resolved_auto_modes(())

    def set_terminal_presentation(
        self,
        candidate: FitCandidate | None,
        *,
        curves: tuple[FitCurveDataset, ...] = (),
        curve_failure: str | None = None,
    ) -> bool:
        """Consume one call-local typed candidate presentation."""

        rendered = self._run_results_tab.set_terminal_presentation(
            candidate,
            curves=curves,
            curve_failure=curve_failure,
        )
        self._results_summary_button.setEnabled(candidate is not None)
        self._apply_button.setEnabled(candidate is not None and not self._executing)
        resolved_auto_modes = () if candidate is None else tuple(
            (str(dataset_id), str(mode))
            for dataset_id, mode in candidate.resolved_auto_modes
        )
        self.set_resolved_auto_modes(resolved_auto_modes)
        self.set_last_fit_presentation()
        self._tabs.setCurrentIndex(2)
        return bool(rendered)

    def set_apply_available(self, available: bool) -> None:
        self._apply_button.setEnabled(bool(available) and not self._executing)

    def set_shared_seed_available(self, available: bool) -> None:
        enabled = bool(available) and not self._executing
        self._shared_seed_combo.setEnabled(enabled)
        self._shared_seed_button.setEnabled(enabled)

    def _request_shared_seed(self) -> None:
        if self._closing or self._executing:
            return
        self.sharedSeedRequested.emit(
            str(self._shared_seed_combo.currentData() or "arithmetic")
        )

    def clear_candidate(self) -> None:
        self._clear_resolved_auto_modes()
        self._copy_button.setEnabled(False)
        self._apply_button.setEnabled(False)
        self.set_shared_seed_available(False)
        self._results_summary_button.setEnabled(False)
        self.set_last_fit_presentation()
        self._run_results_tab.clear_candidate()

    def render_results_summary(
        self,
        summary: str,
    ) -> None:
        self._run_results_tab.render_results_summary(
            summary,
        )

    def set_terminal_failure(self, failure: FactualCause) -> None:
        """Present a worker failure while retaining its factual cause chain."""

        self._run_results_tab.set_terminal_failure(failure)
        self.set_status(f"Fit failed: {failure.message}")
        self._tabs.setCurrentIndex(2)

    def set_terminal_unavailable(
        self,
        cause: ScientificUnavailable,
    ) -> None:
        """Present scientific unavailability without classifying it as failure."""

        self.clear_candidate()
        self._run_results_tab.set_terminal_unavailable(
            cause,
        )
        factual = cause.factual_cause
        self.set_status(
            "Fit result scientifically unavailable: "
            + (factual.message if factual is not None else cause.stage)
        )
        self._tabs.setCurrentIndex(2)

    def _emit_project_defaults(self) -> None:
        if self._closing:
            return
        setup = self._projected_setup
        if setup is not None:
            self.projectDefaultsRequested.emit(setup.defaults)

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

    def reject(self) -> None:
        """Route QDialog dismissal through the owner-controlled close fence."""

        if self._closing:
            return
        self.close()

    def enter_stopping(self) -> None:
        """Make the presentation inert until owner-authorized cleanup closes it."""

        if self._closing_after_cleanup:
            return
        self._closing = True
        self._status.setText("Stopping…")
        self._tabs.setEnabled(False)
        self._current_tab_stack.setEnabled(False)
        self._data_targets_tab.setEnabled(False)
        self._params_ics_tab.setEnabled(False)
        self._run_button.setEnabled(False)
        self._cancel_button.setEnabled(False)
        self._pause_button.setEnabled(False)
        self._resume_button.setEnabled(False)
        self._results_summary_button.setEnabled(False)
        self._copy_button.setEnabled(False)
        self._apply_button.setEnabled(False)
        self.set_shared_seed_available(False)
        self._project_defaults_button.setEnabled(False)
        self._close_button.setEnabled(False)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802
        if self._closing_after_cleanup:
            event.accept()
            return
        if self._closing:
            event.ignore()
            return
        self.enter_stopping()
        if self._close_handler():
            self._closing_after_cleanup = True
            event.accept()
        else:
            event.ignore()
