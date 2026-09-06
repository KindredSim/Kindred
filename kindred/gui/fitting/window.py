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
from kindred.gui.ui_helpers import clamp_initial_size, set_bounded_label_text
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
    stopRequested = QtCore.Signal()
    pauseRequested = QtCore.Signal()
    resumeRequested = QtCore.Signal()
    setupSnapshotRequested = QtCore.Signal(object)
    resetRequested = QtCore.Signal(str)
    resultsSummaryRequested = QtCore.Signal()

    def __init__(
        self,
        origin: FitOriginSnapshot,
        *,
        close_handler: Callable[[], bool],
        observation_style: tuple[str, float, float],
        color_resolver: Callable[[str], object] | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        if not isinstance(origin, FitOriginSnapshot):
            raise TypeError("origin must be FitOriginSnapshot.")
        self._origin = origin
        self._close_handler = close_handler
        self._color_resolver = color_resolver
        self._observation_style = observation_style
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
        clamp_initial_size(self, 1280, 720)

    # ------------------------------------------------------------------
    # Top-level presentation
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
            parent=self,
        )
        self._run_results_tab = RunResultsTab(
            parent=self,
            color_resolver=self._color_resolver,
            observation_style=self._observation_style,
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
        self._current_tab_stack.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored,
                                             QtWidgets.QSizePolicy.Policy.Ignored)
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
        footer.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Preferred)
        column = QtWidgets.QVBoxLayout(footer)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(4)
        self._status = QtWidgets.QLabel("Ready", footer)
        self._status.setWordWrap(True)
        self._status.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Preferred)
        self._status.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        status_widget = self._status_widget = QtWidgets.QWidget(footer)
        status_column = QtWidgets.QVBoxLayout(status_widget)
        status_column.setContentsMargins(0, 0, 0, 0)
        status_column.addWidget(self._status)
        self._run_block_reason_label = QtWidgets.QLabel(footer)
        self._run_block_reason_label.setObjectName("global_fit_run_block_reason_label")
        self._run_block_reason_label.setWordWrap(True)
        self._run_block_reason_label.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Preferred)
        self._run_block_reason_label.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        self._run_block_reason_label.hide()
        review_row = QtWidgets.QHBoxLayout()
        review_row.addWidget(self._run_block_reason_label, 1)
        self._review_data_button = QtWidgets.QPushButton("Review &Data Changes", footer)
        self._review_data_button.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed
        )
        self._review_data_button.setEnabled(False)
        self._review_data_button.clicked.connect(self._review_data_changes)
        review_row.addWidget(self._review_data_button, 0, QtCore.Qt.AlignmentFlag.AlignRight)
        status_column.addLayout(review_row)
        column.addWidget(status_widget)
        self._progress = QtWidgets.QProgressBar(footer)
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setFormat("%p%")
        self._progress.setFixedWidth(self.fontMetrics().horizontalAdvance("100%") + 48)

        actions = QtWidgets.QVBoxLayout()
        self._footer_actions = actions
        column.addLayout(actions)
        self._footer_widgets = [self._progress]
        self._run_button = QtWidgets.QPushButton("&Run Fit", footer)
        self._run_button.clicked.connect(self._request_execute)
        self._cancel_button = QtWidgets.QPushButton("&Stop", footer)
        self._cancel_button.setEnabled(False)
        self._cancel_button.clicked.connect(self.stopRequested)
        self._pause_button = QtWidgets.QPushButton("&Pause", footer)
        self._pause_button.setEnabled(False)
        self._pause_button.clicked.connect(self._toggle_pause)
        for button in (self._run_button, self._pause_button, self._cancel_button):
            self._footer_widgets.append(button)
        self._results_summary_button = QtWidgets.QPushButton("S&ummary", footer)
        self._results_summary_button.clicked.connect(self.resultsSummaryRequested)
        self._footer_widgets.append(self._results_summary_button)
        self._copy_button = QtWidgets.QPushButton("&Copy Fit Curves", footer)
        self._copy_button.setEnabled(False)
        self._copy_button.clicked.connect(self._run_results_tab.copy_active_to_clipboard)
        self._footer_widgets.append(self._copy_button)
        self._apply_button = QtWidgets.QPushButton("&Apply Fit to Project", footer)
        self._apply_button.setEnabled(False)
        self._apply_button.clicked.connect(self.applyRequested)
        self._footer_widgets.append(self._apply_button)
        self._shared_seed_combo = QtWidgets.QComboBox(footer)
        self._shared_seed_combo.addItem("Arithmetic mean", "arithmetic")
        self._shared_seed_combo.addItem("Geometric mean", "geometric")
        self._footer_widgets.append(self._shared_seed_combo)
        self._shared_seed_button = QtWidgets.QPushButton("Configure S&hared Fit", footer)
        self._shared_seed_button.clicked.connect(
            lambda: self.sharedSeedRequested.emit(self._shared_seed_combo.currentData()))
        self._footer_widgets.append(self._shared_seed_button)
        self.set_shared_seed_available(False)
        reason_row = QtWidgets.QHBoxLayout()
        self._apply_reason_label = QtWidgets.QLabel(footer)
        reason_row.addWidget(self._apply_reason_label, 1)
        column.addLayout(reason_row)
        self.set_apply_available(False, "Run a fit to obtain values to apply.")
        self._layout_footer()
        return footer

    def _layout_footer(self) -> None:
        self._status_widget.setVisible(bool(self._status.text() or self._run_block_reason_label.text()))
        self._progress.setFixedWidth(self._progress.fontMetrics().horizontalAdvance("100%") + 48)
        if hasattr(self, "_apply_reason_label"):
            self._apply_reason_label.setFixedHeight(self._apply_reason_label.fontMetrics().lineSpacing())
        available = max(1, self.width() - 32)
        key = (available, tuple((widget.isHidden(), widget.sizeHint().width()) for widget in self._footer_widgets))
        if key == getattr(self, "_footer_layout_key", None):
            return
        self._footer_layout_key = key
        while self._footer_actions.count():
            row = self._footer_actions.takeAt(0).layout()
            while row.count():
                row.takeAt(0)
            row.deleteLater()
        row = QtWidgets.QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addStretch(1)
        self._footer_actions.addLayout(row)
        used = 0
        for widget in self._footer_widgets:
            if widget.isHidden():
                continue
            width = widget.sizeHint().width()
            if used and used + width > available:
                row = QtWidgets.QHBoxLayout()
                row.setContentsMargins(0, 0, 0, 0)
                row.addStretch(1)
                self._footer_actions.addLayout(row)
                used = 0
            if isinstance(widget, (QtWidgets.QPushButton, QtWidgets.QComboBox)):
                widget.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
            row.addWidget(widget)
            used += width + 6

    def event(self, event: QtCore.QEvent) -> bool:
        if event.type() == QtCore.QEvent.Type.LayoutRequest and hasattr(self, "_footer_widgets"):
            self._layout_footer()
        return super().event(event)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        if hasattr(self, "_footer_widgets"):
            self._layout_footer()

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
        self._params_ics_tab.viewStepsRequested.connect(
            self._open_mechanism_inspector
        )
        self._params_ics_tab.statusMessage.connect(self.set_status)
        self._params_ics_tab.validityChanged.connect(
            self._refresh_setup_validity
        )

        self._run_results_tab.copyAvailableChanged.connect(
            self._copy_button.setEnabled
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
    def _open_mechanism_inspector(self) -> None:
        from kindred.gui.widgets.mechanism_inspector import MechanismInspectorDialog

        MechanismInspectorDialog(
            self._origin.prepared_dsl,
            self,
        ).exec()

    def set_initial_reset_available(self, snapshot: FitSetupSnapshot) -> None:
        self._params_ics_tab.set_initial_reset_available(snapshot)

    def set_last_fit_reset_available(self, snapshot: FitSetupSnapshot | None) -> None:
        """Project Last Fit reset availability without retaining the snapshot."""

        self._params_ics_tab.set_last_fit_reset_available(snapshot)

    def clear_pending_setup_edits(self) -> None:
        """Discard every leaf draft and refresh the complete Fit surface."""

        self._params_ics_tab.clear_pending_setup_edits()
        self._species_table.clear_pending_edits()
        self._data_tab.clear_pending_edits()
        self._refresh_setup_validity()

    def project_setup_snapshot(self, snapshot: FitSetupSnapshot) -> None:
        """Project accepted baselines while retaining unrelated local edits."""

        self._origin = snapshot.origin
        self._projected_setup = snapshot
        self._populate_dataset_list(snapshot)
        selected_dataset_id = (
            self._data_targets_tab.unified_list.selected_dataset_id()
        )
        self._setup_projecting = True
        try:
            self._params_ics_tab.project_setup_snapshot(snapshot)
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

    @property
    def has_invalid_setup_edits(self) -> bool:
        setup = self._projected_setup
        data_issues = (
            {}
            if setup is None
            else self._data_tab.data_edit_issues(
                self._included_dataset_ids(setup)
            )
        )
        return any(data_issues.values()) or self._has_invalid_parameter_or_species_edits()

    def _has_invalid_parameter_or_species_edits(self) -> bool:
        return bool(
            self._species_table.invalid_editor_keys
            or self._species_table.has_pending_initial_changes
            or self._params_ics_tab.has_invalid_edits
        )

    def _refresh_setup_validity(self) -> None:
        setup = self._projected_setup
        data_issues = self._data_tab.data_edit_issues(
            dataset.dataset_id for dataset in setup.datasets
        ) if setup is not None else {}
        included = set(self._included_dataset_ids(setup)) if setup is not None else set()
        active_issues = {key: value for key, value in data_issues.items() if key in included}
        invalid = any(active_issues.values()) or self._has_invalid_parameter_or_species_edits()
        pending_targets = self.has_pending_target_changes
        pending_data = self.has_pending_data_changes
        structural = None
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
                shared_initial_groups=setup.shared_initial_groups,
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
        elif structural is not None:
            reason = structural
        listing = self._data_targets_tab.unified_list
        listing.set_data_edit_issues(data_issues)
        self._review_data_button.setEnabled(bool(active_issues) and not self._is_executing())
        accessible_reason = reason
        if active_issues:
            dataset_id, issue = next(iter(active_issues.items()))
            issue = issue or "Pending data changes"
            name = listing.dataset_display_name(dataset_id) or dataset_id
            # Authored names and diagnostics cannot grow the footer unboundedly.
            visible_name = " ".join(name.split())
            visible_name = visible_name if len(visible_name) <= 60 else visible_name[:59] + "…"
            visible_issue = " ".join(issue.split())
            visible_issue = visible_issue if len(visible_issue) <= 160 else visible_issue[:159] + "…"
            more = f" (+{len(active_issues) - 1} more)" if len(active_issues) > 1 else ""
            reason = f"Data settings — {visible_name}: {visible_issue}{more} — Review Data Changes to resolve."
            accessible_reason = "\n".join(
                f"{listing.dataset_display_name(key) or key}: {value or 'Pending data changes'}"
                for key, value in active_issues.items()
            )
        self._run_block_reason_label.setVisible(bool(reason))
        self._run_block_reason_label.setText(reason)
        self._run_block_reason_label.setAccessibleName(accessible_reason)
        self._layout_footer()

    def _review_data_changes(self) -> None:
        setup = self._projected_setup
        if setup is None:
            return
        issues = self._data_tab.data_edit_issues(self._included_dataset_ids(setup))
        if not issues:
            return
        listing = self._data_targets_tab.unified_list
        ids = list(issues)
        current = listing.selected_dataset_id()
        index = (ids.index(current) + 1) % len(ids) if current in ids else 0
        self._tabs.setCurrentIndex(0)
        listing.select_dataset(ids[index])
        self._data_tab.setFocus(QtCore.Qt.FocusReason.ShortcutFocusReason)

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
    # Fit data and setup presentation adapters
    # ------------------------------------------------------------------

    def _mechanism_species(self) -> tuple[str, ...]:
        analysis = self._origin.prepared_dsl.analysis
        return tuple(
            dict.fromkeys(
                (*analysis.species_names, *analysis.observable_names)
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

    def _is_executing(self) -> bool:
        return bool(self._executing)

    def _request_execute(self) -> None:
        if self._closing:
            return
        # Mnemonics do not move focus like a mouse click does. Let the focused
        # editor settle through its normal signals before checking admission.
        focused = self.focusWidget()
        if focused is not None:
            focused.clearFocus()
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
            self._paused = False
            self._pause_button.setText("Pause")
            self._refresh_copy_allowed()
            self.resumeRequested.emit()
            return
        self._paused = True
        self._pause_button.setText("Resume")
        self._refresh_copy_allowed()
        self.pauseRequested.emit()

    def set_executing(self, executing: bool) -> None:
        was_executing = self._executing
        self._executing = bool(executing)
        if self._executing:
            self._status.clear()
            self._status.hide()
            self._layout_footer()
            if not was_executing:
                self._progress.setValue(0)
            self._paused = False
            self._start_results_presentation()
            self._tabs.setCurrentIndex(2)
            self._apply_button.setEnabled(False)
            self.set_shared_seed_available(False)
        else:
            self._paused = False
        self._refresh_copy_allowed()
        self._results_summary_button.setEnabled(True)
        self._cancel_button.setEnabled(self._executing)
        self._pause_button.setText("Pause")
        self._pause_button.setEnabled(self._executing)
        self._species_table.set_running_state(self._executing)
        self._params_ics_tab.set_running_state(self._executing)
        self._params_ics_tab.setEnabled(not self._executing)
        self._data_targets_tab.unified_list.set_running_state(self._executing)
        self._data_tab.setEnabled(not self._executing)
        self._refresh_setup_validity()
        self._layout_footer()

    def set_observation_style(self, shape: str, size: float, opacity: float) -> None:
        self._observation_style = (shape, size, opacity)
        self._run_results_tab.set_observation_style(shape, size, opacity)

    def set_progress(self, value: int) -> None:
        progress = max(0, min(100, int(value)))
        self._progress.setValue(progress)
        self._run_results_tab.set_progress(progress)

    def set_status(self, text: str) -> None:
        if self._closing and not self._closing_after_cleanup:
            self.set_execution_status("Closing", "Stopping Fit before closing…")
            return
        self._status.setText(str(text))
        self._status.setAccessibleName(str(text))
        self._status.setToolTip(str(text))
        self._status.setVisible(bool(text))
        self._layout_footer()

    def set_execution_status(self, phase: str, detail: str) -> None:
        self._status.clear()
        self._status.hide()
        self._layout_footer()
        self._run_results_tab.set_execution_status(phase, detail)

    def set_live_best(self, update: FitLiveUpdate) -> None:
        if not isinstance(update, FitLiveUpdate):
            raise TypeError("update must be FitLiveUpdate.")
        setup = self._projected_setup
        included = set() if setup is None else {
            dataset.dataset_id for dataset in setup.datasets if dataset.included
        }
        labels = {} if setup is None else {
            dataset.dataset_id: dataset.display_name
            for dataset in setup.origin.mapped_datasets if dataset.dataset_id in included
        }
        initial_labels: dict[str, str] = {}
        if setup is not None:
            for mapped in setup.origin.mapped_datasets:
                if mapped.dataset_id in included:
                    previous = initial_labels.get(mapped.set_id)
                    initial_labels[mapped.set_id] = (
                        mapped.display_name if previous is None else previous + ", " + mapped.display_name
                    )
        self._run_results_tab.set_live_best(update, dataset_labels=labels, initial_labels=initial_labels)

    def set_live_curves(self, curves: tuple[FitCurveDataset, ...]) -> None:
        values = tuple(curves)
        if any(not isinstance(item, FitCurveDataset) for item in values):
            raise TypeError("curves must contain FitCurveDataset values.")
        self._run_results_tab.set_live_curves(values)

    def set_resolved_auto_modes(
        self,
        modes: tuple[tuple[str, str], ...],
    ) -> None:
        self._data_tab.set_resolved_auto_modes(modes)

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

        self._status.clear()
        self._status.hide()
        self._layout_footer()
        rendered = self._run_results_tab.set_terminal_presentation(
            candidate,
            curves=curves,
            curve_failure=curve_failure,
        )
        self._results_summary_button.setEnabled(candidate is not None)
        self.set_resolved_auto_modes(
            () if candidate is None else candidate.resolved_auto_modes
        )
        if candidate is not None:
            self._params_ics_tab.set_last_fit_presentation({
                settlement.parameter_name: settlement.overall_decoded_value
                for settlement in candidate.result.centred_fixed_effects_settlements
            })
        self._layout_footer()
        self._tabs.setCurrentIndex(2)
        return bool(rendered)

    def set_apply_available(self, available: bool, reason: str) -> None:
        self._apply_button.setEnabled(available and not self._executing)
        set_bounded_label_text(self._apply_reason_label, reason, max_width=16777215)
        self._run_results_tab.set_apply_reason(reason)
        self._layout_footer()

    def set_shared_seed_available(self, available: bool) -> None:
        for widget in (self._shared_seed_combo, self._shared_seed_button):
            widget.setEnabled(available and not self._executing)
        self._layout_footer()

    def _start_results_presentation(self) -> None:
        setup = self._projected_setup
        self._run_results_tab.set_waiting(
            "Fit" if setup is None else setup.origin.tab_name,
            () if setup is None else setup.parameters,
            0 if setup is None else sum(dataset.included for dataset in setup.datasets),
        )

    def clear_candidate(self) -> None:
        self._clear_resolved_auto_modes()
        self._apply_button.setEnabled(False)
        self._results_summary_button.setEnabled(True)
        if self._executing:
            self._start_results_presentation()
        else:
            self._run_results_tab.clear_candidate()

    def render_results_summary(self) -> None:
        self._run_results_tab.render_results_summary()

    def set_terminal_failure(self, failure: FactualCause) -> None:
        """Present a worker failure while retaining its factual cause chain."""

        self._status.clear()
        self._status.hide()
        self._layout_footer()
        self._run_results_tab.set_terminal_failure(failure)
        self._tabs.setCurrentIndex(2)

    def set_terminal_unavailable(
        self,
        cause: ScientificUnavailable,
    ) -> None:
        """Present scientific unavailability without classifying it as failure."""

        self.clear_candidate()
        self._status.clear()
        self._status.hide()
        self._layout_footer()
        self._run_results_tab.set_terminal_unavailable(
            cause,
        )
        self._tabs.setCurrentIndex(2)

    def _refresh_copy_allowed(self) -> None:
        self._run_results_tab.set_copy_allowed(
            not self._closing and (not self._executing or self._paused)
        )

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
        if event.type() == QtCore.QEvent.Type.FontChange and hasattr(self, "_footer_widgets"):
            self._layout_footer()
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
        self._refresh_copy_allowed()
        self.set_execution_status("Closing", "Stopping Fit before closing…")
        self._tabs.setEnabled(False)
        self._current_tab_stack.setEnabled(False)
        self._data_targets_tab.setEnabled(False)
        self._params_ics_tab.setEnabled(False)
        self._run_button.setEnabled(False)
        self._cancel_button.setEnabled(False)
        self._pause_button.setText("Pause")
        self._pause_button.setEnabled(False)
        self._results_summary_button.setEnabled(False)
        self._apply_button.setEnabled(False)
        self.set_shared_seed_available(False)

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
