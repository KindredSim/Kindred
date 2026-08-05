"""Presentation for the historical unified Fit target/initial table.

The table is deliberately a view/editor surface.  Accepted Fit setup remains
owned by :class:`FitSessionOwner`; this widget only renders the owner supplied
snapshot and emits existing typed setup intents for user edits.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Sequence

from PySide6 import QtWidgets
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFontMetrics, QPalette

from kindred.gui.display_name_policy import (
    DATASET_LIST_LABEL_MAX_CHARS,
    compact_dataset_label,
)
from kindred.gui.fitting.contracts import FitTargetSelection, copy_target_selection
from kindred.gui.fitting.launch import (
    FitDatasetLocalSettings,
    FitDatasetSetupEdit,
    FitDatasetWeightModeSetupEdit,
    FitInitialSetupEdit,
    FitInitialVariable,
    FitMappedDatasetSnapshot,
    FitObservationSnapshot,
    FitOriginSnapshot,
    FitSetupSnapshot,
)
from kindred.gui.ui_helpers import setup_scientific_validator


class _Col:
    """Column indices for the historical unified species table."""

    INCLUDE = 0
    WEIGHT = 1
    SPECIES = 2
    INITIAL = 3
    FIT_IC = 4
    LOG10 = 5
    MIN = 6
    MAX = 7

    COUNT = 8
    HEADERS = [
        "Include in Fit",
        "Weight",
        "Species",
        "Initial Value",
        "Fit IC",
        "Log10",
        "Min",
        "Max",
    ]


_INVALID_BG = QBrush(QColor(255, 200, 200))
_INVALID_FG = QBrush(QColor(80, 0, 0))
_DEFAULT_BG = QBrush()
_DEFAULT_FG = QBrush()

_DATASET_ROLE = Qt.ItemDataRole.UserRole
_SET_ROLE = Qt.ItemDataRole.UserRole + 1
_RAW_ID_ROLE = Qt.ItemDataRole.UserRole + 2
_RAW_COLUMN_ROLE = Qt.ItemDataRole.UserRole + 3
_MODEL_TARGET_ROLE = Qt.ItemDataRole.UserRole + 4
_SPECIES_ROLE = Qt.ItemDataRole.UserRole + 5


class _ValidationCellDelegate(QtWidgets.QStyledItemDelegate):
    """Paint cell text using ForegroundRole, bypassing stylesheet overrides."""

    def paint(self, painter, option, index):  # noqa: N802 - Qt override
        opt = QtWidgets.QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        fg_variant = index.data(Qt.ItemDataRole.ForegroundRole)
        if (
            isinstance(fg_variant, QBrush)
            and fg_variant.style() != Qt.BrushStyle.NoBrush
        ):
            text = opt.text
            opt.text = ""
            style = (
                option.widget.style()
                if option.widget
                else QtWidgets.QApplication.style()
            )
            style.drawControl(
                QtWidgets.QStyle.ControlElement.CE_ItemViewItem,
                opt,
                painter,
                option.widget,
            )
            painter.save()
            if opt.state & QtWidgets.QStyle.StateFlag.State_Selected:
                painter.setPen(opt.palette.color(QPalette.ColorRole.HighlightedText))
            else:
                painter.setPen(fg_variant.color())
            text_rect = style.subElementRect(
                QtWidgets.QStyle.SubElement.SE_ItemViewItemText,
                opt,
                option.widget,
            )
            metrics = QFontMetrics(opt.font)
            painter.drawText(
                text_rect,
                opt.displayAlignment,
                metrics.elidedText(text, Qt.TextElideMode.ElideRight, text_rect.width()),
            )
            painter.restore()
            return
        super().paint(painter, option, index)


class UnifiedSpeciesTable(QtWidgets.QWidget):
    """Historical target/weight/initial presentation with typed edit hooks."""

    # The payload is the tuple already consumed by FittingWindow's
    # setupEditsRequested signal.  targetsApplied only acknowledges the
    # presentation route after that synchronous owner-facing emission; it
    # does not accept or retain Fit state.
    setupEditsRequested = Signal(object)
    targetsApplied = Signal()
    validityChanged = Signal()
    statusMessage = Signal(str)

    def __init__(
        self,
        *,
        origin: FitOriginSnapshot,
        mechanism_species: Sequence[str] = (),
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        if not isinstance(origin, FitOriginSnapshot):
            raise TypeError("origin must be FitOriginSnapshot.")
        self._origin = origin
        self._mechanism_species = tuple(
            dict.fromkeys(
                str(species).strip()
                for species in mechanism_species
                if str(species).strip()
            )
        )
        if not self._mechanism_species:
            self._mechanism_species = tuple(
                dict.fromkeys(
                    species
                    for mapped in origin.mapped_datasets
                    for species, _value in mapped.initial_values
                )
            )

        self._projected_setup: FitSetupSnapshot | None = None
        self._current_dataset_id: str | None = None
        self._pending_target_selections: dict[str, tuple[str, ...]] = {}
        self._pending_target_weights: dict[tuple[str, str], float] = {}
        self._pending_dataset_weights: dict[str, float] = {}
        self._pending_initials: dict[tuple[str, str], FitInitialVariable] = {}
        self._invalid_raw: dict[tuple[str, ...], str] = {}
        self._unresolved_global_species: set[str] = set()
        self._pre_global_local_backups: dict[
            str, dict[tuple[str, str], FitInitialVariable]
        ] = {}
        self._global_draft_raw: dict[tuple[str, str], str] = {}
        self._global_draft_checks: dict[tuple[str, str], Qt.CheckState] = {}
        self._refreshing = False
        self._running = False

        self._build_ui()

    # ------------------------------------------------------------------
    # Historical physical construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        from kindred.gui.widgets.config_panel_footer import ConfigPanelFooter

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        group = QtWidgets.QGroupBox("Species Table", self)
        group.setObjectName("global_fit_unified_species_group")
        group_layout = QtWidgets.QVBoxLayout(group)

        self._footer = ConfigPanelFooter(
            group,
            show_dirty=True,
            show_secondary_error=True,
            show_divider=True,
            apply_requires_no_error=False,
            button_order=("apply", "revert"),
            error_object_name="global_fit_species_table_error",
            secondary_error_object_name="global_fit_species_table_run_blocked",
            apply_object_name="global_fit_species_table_apply",
            revert_object_name="global_fit_species_table_revert",
        )
        group_layout.addWidget(self._footer, stretch=1)
        self._footer.applyRequested.connect(self._apply_presentation_edits)
        self._footer.revertRequested.connect(self._revert_presentation_edits)

        self._context_label = QtWidgets.QLabel("Selected dataset: —", group)
        self._context_label.setStyleSheet("font-weight: bold;")
        self._footer.body_layout.addWidget(self._context_label)

        scope_form = QtWidgets.QFormLayout()
        scope_form.setContentsMargins(0, 0, 0, 0)
        self._initial_scope_species = QtWidgets.QComboBox(group)
        self._initial_scope_species.setObjectName(
            "global_fit_initial_scope_species"
        )
        self._initial_scope_species.addItems(list(self._mechanism_species))
        scope_form.addRow("Initial concentration:", self._initial_scope_species)

        mode_row = QtWidgets.QHBoxLayout()
        self._initial_global_radio = QtWidgets.QRadioButton("Global", group)
        self._initial_local_radio = QtWidgets.QRadioButton("Local", group)
        self._initial_scope_group = QtWidgets.QButtonGroup(group)
        self._initial_scope_group.addButton(self._initial_global_radio)
        self._initial_scope_group.addButton(self._initial_local_radio)
        mode_row.addWidget(self._initial_global_radio)
        mode_row.addWidget(self._initial_local_radio)
        mode_row.addStretch(1)
        mode_widget = QtWidgets.QWidget(group)
        mode_widget.setLayout(mode_row)
        scope_form.addRow("Mode:", mode_widget)
        self._initial_scope_dataset_label = QtWidgets.QLabel(group)
        scope_form.addRow("", self._initial_scope_dataset_label)
        self._footer.body_layout.addLayout(scope_form)

        self._initial_scope_status = QtWidgets.QLabel(group)
        self._initial_scope_status.setObjectName(
            "global_fit_initial_scope_status"
        )
        self._initial_scope_status.setWordWrap(True)
        self._footer.body_layout.addWidget(self._initial_scope_status)
        initial_help = QtWidgets.QLabel(
            "Global: one parameter updates the initial value for all selected "
            "datasets.\nLocal: a separate parameter is created for each "
            "selected dataset.",
            group,
        )
        initial_help.setWordWrap(True)
        self._footer.body_layout.addWidget(initial_help)
        self._initial_scope_species.currentTextChanged.connect(
            self._refresh_initial_scope_controls
        )
        self._initial_global_radio.toggled.connect(
            self._on_initial_scope_toggled
        )
        self._initial_local_radio.toggled.connect(
            self._on_initial_scope_toggled
        )

        weighting_form = QtWidgets.QFormLayout()
        weighting_form.setContentsMargins(0, 0, 0, 0)
        self._weight_mode_combo = QtWidgets.QComboBox(group)
        self._weight_mode_combo.setObjectName("global_fit_weight_mode_combo")
        self._weight_mode_combo.addItem(
            "Implicit weights (1/N per dataset)",
            "implicit",
        )
        self._weight_mode_combo.addItem("User weights only", "user")
        weighting_form.addRow("Weight mode:", self._weight_mode_combo)

        self._dataset_weight_edit = QtWidgets.QLineEdit(group)
        self._dataset_weight_edit.setObjectName("global_fit_dataset_weight_edit")
        setup_scientific_validator(self._dataset_weight_edit)
        weighting_form.addRow("Dataset weight:", self._dataset_weight_edit)
        self._footer.body_layout.addLayout(weighting_form)
        self._weight_mode_combo.currentIndexChanged.connect(
            self._on_weight_mode_changed
        )
        self._dataset_weight_edit.textChanged.connect(
            self._on_dataset_weight_text_changed
        )

        bulk_row = QtWidgets.QHBoxLayout()
        bulk_label = QtWidgets.QLabel("Bulk:", group)
        bulk_label.setStyleSheet("font-size: 11px;")
        bulk_row.addWidget(bulk_label)
        self._bulk_all_button = QtWidgets.QPushButton("All", group)
        self._bulk_all_button.setObjectName("global_fit_fit_targets_bulk_all")
        self._bulk_all_button.clicked.connect(
            lambda: self._apply_bulk_action("all")
        )
        self._bulk_none_button = QtWidgets.QPushButton("None", group)
        self._bulk_none_button.setObjectName("global_fit_fit_targets_bulk_none")
        self._bulk_none_button.clicked.connect(
            lambda: self._apply_bulk_action("none")
        )
        self._bulk_invert_button = QtWidgets.QPushButton("Invert", group)
        self._bulk_invert_button.setObjectName("global_fit_fit_targets_bulk_invert")
        self._bulk_invert_button.clicked.connect(
            lambda: self._apply_bulk_action("invert")
        )
        self._copy_targets_button = QtWidgets.QPushButton(
            "Copy Targets to Compatible",
            group,
        )
        self._copy_targets_button.setObjectName(
            "global_fit_fit_targets_copy_compatible"
        )
        self._copy_targets_button.clicked.connect(
            self._copy_targets_to_compatible_datasets
        )
        for button in (
            self._bulk_all_button,
            self._bulk_none_button,
            self._bulk_invert_button,
            self._copy_targets_button,
        ):
            bulk_row.addWidget(button)
        bulk_row.addStretch(1)
        self._footer.body_layout.addLayout(bulk_row)

        self._table = QtWidgets.QTableWidget(group)
        self._table.setObjectName("global_fit_unified_species_table")
        self._table.setItemDelegate(_ValidationCellDelegate(self._table))
        self._table.setColumnCount(_Col.COUNT)
        self._table.setHorizontalHeaderLabels(_Col.HEADERS)
        header = self._table.horizontalHeader()
        header.setStretchLastSection(False)
        for column in range(_Col.COUNT):
            header.setSectionResizeMode(
                column,
                QtWidgets.QHeaderView.ResizeMode.ResizeToContents,
            )
        header.setSectionResizeMode(
            _Col.SPECIES,
            QtWidgets.QHeaderView.ResizeMode.Interactive,
        )
        self._table.setColumnWidth(_Col.SPECIES, 140)
        self._table.verticalHeader().setVisible(False)
        self._table.setAlternatingRowColors(True)
        self._table.setMinimumHeight(200)
        self._table.itemChanged.connect(self._on_table_item_changed)
        self._footer.body_layout.addWidget(self._table, stretch=1)

        layout.addWidget(group, stretch=1)

    # ------------------------------------------------------------------
    # Typed snapshot projection
    # ------------------------------------------------------------------
    def project_setup_snapshot(
        self,
        snapshot: FitSetupSnapshot,
        *,
        dataset_id: str | None = None,
    ) -> None:
        """Render an owner snapshot without reconstructing Fit state."""

        if not isinstance(snapshot, FitSetupSnapshot):
            raise TypeError("snapshot must be FitSetupSnapshot.")
        if (
            snapshot.origin_tab_id != self._origin.tab_id
            or snapshot.origin_capture_id != self._origin.capture_id
        ):
            raise ValueError("Fit snapshot does not match this origin.")

        available_ids = tuple(item.dataset_id for item in snapshot.datasets)
        requested_id = str(dataset_id or self._current_dataset_id or "").strip()
        if requested_id and requested_id not in available_ids:
            raise ValueError("Fit snapshot does not contain the selected dataset.")

        previous = self._projected_setup
        inclusion_edits = self._global_inclusion_transition_edits(
            previous,
            snapshot,
        )
        self._projected_setup = snapshot
        self._resolve_unresolved_globals_if_identical()
        self._current_dataset_id = requested_id or (
            available_ids[0] if available_ids else None
        )
        self._refreshing = True
        self.setUpdatesEnabled(False)
        self._table.blockSignals(True)
        self._weight_mode_combo.blockSignals(True)
        self._dataset_weight_edit.blockSignals(True)
        try:
            mode_index = self._weight_mode_combo.findData(
                snapshot.dataset_weight_mode
            )
            self._weight_mode_combo.setCurrentIndex(max(0, mode_index))
            self._populate_table()
            self._refresh_dataset_weight_editor()
            self._refresh_initial_scope_controls()
        finally:
            self._dataset_weight_edit.blockSignals(False)
            self._weight_mode_combo.blockSignals(False)
            self._table.blockSignals(False)
            self._refreshing = False
            self.setUpdatesEnabled(True)
        self._refresh_footer_dirty()
        self._refresh_footer_validity()
        if inclusion_edits:
            self.setupEditsRequested.emit(inclusion_edits)

    def load_for_dataset(self, dataset_id: str) -> None:
        """Select one projected dataset and repaint the table."""

        if self._projected_setup is None:
            return
        dataset_id = str(dataset_id or "").strip()
        if dataset_id not in {
            item.dataset_id for item in self._projected_setup.datasets
        }:
            return
        self._current_dataset_id = dataset_id
        self._refresh_visible_presentation()

    def refresh_validity_ui(self) -> None:
        """Refresh presentation validation without becoming a Fit validator."""

        self._refresh_footer_validity()

    def set_running_state(self, running: bool) -> None:
        """Preserve the historical disabled/locked running-state visuals."""

        self._running = bool(running)
        self._table.setEnabled(not self._running)
        self._weight_mode_combo.setEnabled(not self._running)
        if self._running:
            self._dataset_weight_edit.setEnabled(False)
        else:
            self._refresh_dataset_weight_editor()
        self._footer.setEnabled(not self._running)
        for button in (
            self._bulk_all_button,
            self._bulk_none_button,
            self._bulk_invert_button,
            self._copy_targets_button,
        ):
            button.setEnabled(not self._running)

    def set_status(self, text: str) -> None:
        self.statusMessage.emit(str(text))

    # ------------------------------------------------------------------
    # Snapshot lookup and presentation helpers
    # ------------------------------------------------------------------
    def _current_setup(self) -> FitSetupSnapshot | None:
        return self._projected_setup

    def _projected_dataset(self, dataset_id: str) -> FitDatasetLocalSettings | None:
        setup = self._current_setup()
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

    def _pending_dataset(self, dataset_id: str) -> FitDatasetLocalSettings | None:
        del dataset_id
        return None

    def _current_dataset(self) -> FitDatasetLocalSettings | None:
        if self._projected_setup is None or not self._current_dataset_id:
            return None
        return self._projected_dataset(self._current_dataset_id)

    def _mapped_dataset(self, dataset_id: str) -> FitMappedDatasetSnapshot | None:
        return next(
            (
                item
                for item in self._origin.mapped_datasets
                if item.dataset_id == dataset_id
            ),
            None,
        )

    def _dataset_id_for_set(self, set_id: str) -> str:
        return next(
            (
                item.dataset_id
                for item in self._origin.mapped_datasets
                if item.set_id == set_id
            ),
            "",
        )

    def _initial_variable(
        self,
        set_id: str,
        species: str,
    ) -> FitInitialVariable | None:
        pending = self._pending_initials.get((set_id, species))
        if pending is not None:
            return pending
        setup = self._current_setup()
        if setup is None:
            return None
        return next(
            (
                item
                for item in setup.initials
                if item.set_id == set_id and item.species == species
            ),
            None,
        )

    @staticmethod
    def _float_text(value: float) -> str:
        return f"{float(value):.6g}"

    @staticmethod
    def _set_editable(item: QtWidgets.QTableWidgetItem, enabled: bool) -> None:
        flags = Qt.ItemFlag.ItemIsSelectable
        if enabled:
            flags |= Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
            flags |= Qt.ItemFlag.ItemIsEditable
        item.setFlags(flags)

    @staticmethod
    def _set_checkable(
        item: QtWidgets.QTableWidgetItem,
        checked: bool,
        enabled: bool,
    ) -> None:
        flags = Qt.ItemFlag.ItemIsSelectable
        if enabled:
            flags |= Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable
        item.setFlags(flags)
        item.setCheckState(
            Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        )

    @staticmethod
    def _observation_target(
        dataset: FitDatasetLocalSettings,
        observation: FitObservationSnapshot,
    ) -> str:
        mapping = dict(dataset.observation_targets)
        return str(mapping.get(observation.raw_observation_id, observation.model_species))

    def _row_species(self, dataset: FitDatasetLocalSettings) -> list[str]:
        mapped = self._mapped_dataset(dataset.dataset_id)
        observed = (
            [self._observation_target(dataset, observation) for observation in mapped.observations]
            if mapped is not None
            else []
        )
        projected_initials = (
            [
                item.species
                for item in (self._current_setup().initials if self._current_setup() else ())
                if mapped is not None and item.set_id == mapped.set_id
            ]
        )
        return list(
            dict.fromkeys(
                [
                    *self._mechanism_species,
                    *projected_initials,
                    *(name for name in observed if name),
                ]
            )
        )

    def _observations_for_target(
        self,
        dataset: FitDatasetLocalSettings,
        target: str,
    ) -> tuple[FitObservationSnapshot, ...]:
        mapped = self._mapped_dataset(dataset.dataset_id)
        if mapped is None:
            return ()
        return tuple(
            observation
            for observation in mapped.observations
            if self._observation_target(dataset, observation) == target
        )

    def _observation_tooltip(
        self,
        dataset: FitDatasetLocalSettings,
        target: str,
    ) -> str:
        observations = self._observations_for_target(dataset, target)
        if not observations:
            return ""
        lines = ["Raw observations:"]
        lines.extend(
            f"{item.raw_observation_id} — {item.raw_column} → "
            f"{self._observation_target(dataset, item)}"
            for item in observations
        )
        return "\n".join(lines)

    def _populate_table(self) -> None:
        dataset = self._current_dataset()
        self._table.setRowCount(0)
        if dataset is None:
            self._context_label.setText("Selected dataset: —")
            return

        mapped = self._mapped_dataset(dataset.dataset_id)
        if mapped is None:
            return
        selected = set(
            self._pending_target_selections.get(
                dataset.dataset_id,
                dataset.target_species,
            )
        )
        weights = dict(dataset.target_weights)
        set_id = mapped.set_id
        initial_rows = {
            item.species: item
            for item in self._current_setup().initials
            if item.set_id == set_id
        } if self._current_setup() is not None else {}

        compact = compact_dataset_label(
            mapped.display_name,
            max_chars=DATASET_LIST_LABEL_MAX_CHARS,
        )
        self._context_label.setText(f"Selected dataset: {compact.display}")
        self._context_label.setToolTip(
            f"{compact.full}\nDataset ID: {dataset.dataset_id}\nSet ID: {mapped.set_id}"
        )
        row_species = self._row_species(dataset)
        self._table.setRowCount(len(row_species))
        for row, species in enumerate(row_species):
            observations = self._observations_for_target(dataset, species)
            has_data = bool(observations)
            target_item = QtWidgets.QTableWidgetItem()
            self._set_checkable(target_item, species in selected, has_data)
            target_item.setData(_DATASET_ROLE, dataset.dataset_id)
            target_item.setData(_RAW_ID_ROLE, tuple(item.raw_observation_id for item in observations))
            target_item.setData(_RAW_COLUMN_ROLE, tuple(item.raw_column for item in observations))
            target_item.setData(_MODEL_TARGET_ROLE, species)
            target_item.setData(_SPECIES_ROLE, species)
            self._table.setItem(row, _Col.INCLUDE, target_item)

            weight_key = (dataset.dataset_id, species)
            weight_invalid_key = (
                "target_weight",
                dataset.dataset_id,
                species,
            )
            weight_text = (
                self._invalid_raw[weight_invalid_key]
                if weight_invalid_key in self._invalid_raw
                else self._float_text(
                    self._pending_target_weights.get(
                        weight_key,
                        weights.get(species, 1.0),
                    )
                )
            )
            weight_item = QtWidgets.QTableWidgetItem(
                weight_text if has_data else ""
            )
            weight_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._set_editable(weight_item, has_data)
            weight_item.setData(_DATASET_ROLE, dataset.dataset_id)
            weight_item.setData(
                _RAW_ID_ROLE,
                tuple(item.raw_observation_id for item in observations),
            )
            weight_item.setData(
                _RAW_COLUMN_ROLE,
                tuple(item.raw_column for item in observations),
            )
            weight_item.setData(_MODEL_TARGET_ROLE, species)
            weight_item.setData(_SPECIES_ROLE, species)
            if weight_invalid_key in self._invalid_raw:
                weight_item.setBackground(_INVALID_BG)
                weight_item.setForeground(_INVALID_FG)
            self._table.setItem(row, _Col.WEIGHT, weight_item)

            species_item = QtWidgets.QTableWidgetItem(species)
            species_item.setFlags(
                Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
            )
            species_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            species_item.setData(_DATASET_ROLE, dataset.dataset_id)
            species_item.setData(_MODEL_TARGET_ROLE, species)
            species_item.setData(_SPECIES_ROLE, species)
            tooltip = self._observation_tooltip(dataset, species)
            if tooltip:
                species_item.setToolTip(tooltip)
            self._table.setItem(row, _Col.SPECIES, species_item)

            initial = self._pending_initials.get(
                (set_id, species),
                initial_rows.get(species),
            )
            mapped_value = dict(mapped.initial_values).get(species)
            if initial is None and mapped_value is None:
                for column in (
                    _Col.INITIAL,
                    _Col.FIT_IC,
                    _Col.LOG10,
                    _Col.MIN,
                    _Col.MAX,
                ):
                    disabled = QtWidgets.QTableWidgetItem("")
                    disabled.setFlags(Qt.ItemFlag.ItemIsSelectable)
                    self._table.setItem(row, column, disabled)
                continue

            initial_item = QtWidgets.QTableWidgetItem(
                self._float_text(initial.initial if initial else mapped_value)
            )
            initial_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._set_editable(initial_item, initial is not None)
            initial_item.setData(_DATASET_ROLE, dataset.dataset_id)
            initial_item.setData(_SET_ROLE, set_id)
            initial_item.setData(_SPECIES_ROLE, species)
            initial_item.setToolTip(f"Set ID: {set_id}\nSpecies ID: {species}")
            self._table.setItem(row, _Col.INITIAL, initial_item)

            fit_item = QtWidgets.QTableWidgetItem()
            self._set_checkable(fit_item, bool(initial and initial.fit), initial is not None)
            fit_item.setData(_DATASET_ROLE, dataset.dataset_id)
            fit_item.setData(_SET_ROLE, set_id)
            fit_item.setData(_SPECIES_ROLE, species)
            self._table.setItem(row, _Col.FIT_IC, fit_item)

            log_item = QtWidgets.QTableWidgetItem()
            self._set_checkable(
                log_item,
                bool(initial and initial.log10),
                bool(initial and initial.fit),
            )
            log_item.setData(_DATASET_ROLE, dataset.dataset_id)
            log_item.setData(_SET_ROLE, set_id)
            log_item.setData(_SPECIES_ROLE, species)
            self._table.setItem(row, _Col.LOG10, log_item)

            for column, value in (
                (_Col.MIN, initial.minimum if initial else ""),
                (_Col.MAX, initial.maximum if initial else ""),
            ):
                bound_item = QtWidgets.QTableWidgetItem(
                    self._float_text(value) if initial else ""
                )
                bound_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self._set_editable(bound_item, initial is not None)
                bound_item.setData(_DATASET_ROLE, dataset.dataset_id)
                bound_item.setData(_SET_ROLE, set_id)
                bound_item.setData(_SPECIES_ROLE, species)
                bound_item.setToolTip(f"Set ID: {set_id}\nSpecies ID: {species}")
                self._table.setItem(row, column, bound_item)

            self._restore_initial_raw_presentation(
                row=row,
                set_id=set_id,
                species=species,
            )

    def _refresh_visible_presentation(self) -> None:
        self._refreshing = True
        self.setUpdatesEnabled(False)
        self._table.blockSignals(True)
        self._weight_mode_combo.blockSignals(True)
        self._dataset_weight_edit.blockSignals(True)
        try:
            self._populate_table()
            self._refresh_dataset_weight_editor()
            self._refresh_initial_scope_controls()
        finally:
            self._dataset_weight_edit.blockSignals(False)
            self._weight_mode_combo.blockSignals(False)
            self._table.blockSignals(False)
            self._refreshing = False
            self.setUpdatesEnabled(True)
        self._refresh_footer_dirty()
        self._refresh_footer_validity()

    # ------------------------------------------------------------------
    # Presentation edits
    # ------------------------------------------------------------------
    def _on_table_item_changed(self, item: QtWidgets.QTableWidgetItem) -> None:
        if self._refreshing or self._running:
            return
        dataset_id = str(item.data(_DATASET_ROLE) or "")
        species = str(item.data(_SPECIES_ROLE) or item.data(_MODEL_TARGET_ROLE) or "")
        if not dataset_id or not species:
            return
        if item.column() == _Col.WEIGHT:
            key = ("target_weight", dataset_id, species)
            try:
                value = float(item.text())
                if not math.isfinite(value) or value <= 0.0:
                    raise ValueError
            except (TypeError, ValueError, OverflowError):
                self._invalid_raw[key] = item.text()
                self._mark_invalid(
                    item,
                    "Target weights must be finite and positive.",
                )
                return
            self._invalid_raw.pop(key, None)
            self._clear_invalid(item)
            projected = self._projected_dataset(dataset_id)
            baseline = (
                dict(projected.target_weights).get(species, 1.0)
                if projected is not None
                else 1.0
            )
            if math.isclose(value, baseline, rel_tol=1e-12, abs_tol=1e-12):
                self._pending_target_weights.pop((dataset_id, species), None)
            else:
                self._pending_target_weights[(dataset_id, species)] = value
            self._refresh_footer_dirty()
            self._refresh_footer_validity()
            return
        if item.column() == _Col.INCLUDE:
            self._stage_visible_target_selection(dataset_id)
            return
        if item.column() in (
            _Col.INITIAL,
            _Col.FIT_IC,
            _Col.LOG10,
            _Col.MIN,
            _Col.MAX,
        ):
            if item.column() == _Col.FIT_IC:
                log_item = self._table.item(item.row(), _Col.LOG10)
                if log_item is not None and item.checkState() != Qt.CheckState.PartiallyChecked:
                    fit_checked = item.checkState() == Qt.CheckState.Checked
                    self._table.blockSignals(True)
                    try:
                        self._set_checkable(
                            log_item,
                            log_item.checkState() == Qt.CheckState.Checked,
                            fit_checked,
                        )
                        if not fit_checked:
                            log_item.setCheckState(Qt.CheckState.Unchecked)
                    finally:
                        self._table.blockSignals(False)
            self._stage_initial_item_edit(item)

    def _stage_visible_target_selection(self, dataset_id: str) -> None:
        projected = self._projected_dataset(dataset_id)
        if projected is None:
            return
        selected = tuple(
            str(item.data(_MODEL_TARGET_ROLE) or "").strip()
            for row in range(self._table.rowCount())
            for item in (self._table.item(row, _Col.INCLUDE),)
            if item is not None
            and str(item.data(_DATASET_ROLE) or "") == dataset_id
            and item.checkState() == Qt.CheckState.Checked
            and str(item.data(_MODEL_TARGET_ROLE) or "").strip()
        )
        if selected == projected.target_species:
            self._pending_target_selections.pop(dataset_id, None)
        else:
            self._pending_target_selections[dataset_id] = selected
        self._refresh_footer_dirty()
        self._refresh_footer_validity()

    def _on_dataset_weight_text_changed(self, raw: str) -> None:
        if self._refreshing or self._running:
            return
        dataset_id = str(self._current_dataset_id or "").strip()
        projected = self._projected_dataset(dataset_id)
        if not dataset_id or projected is None:
            return
        key = ("dataset_weight", dataset_id)
        try:
            value = float(raw)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError
        except (TypeError, ValueError, OverflowError):
            self._invalid_raw[key] = raw
            self._mark_invalid_line_edit(
                self._dataset_weight_edit,
                "Dataset weight must be finite and positive.",
            )
            self._refresh_footer_dirty()
            return
        self._invalid_raw.pop(key, None)
        self._clear_invalid_line_edit(self._dataset_weight_edit)
        if math.isclose(value, projected.weight, rel_tol=1e-12, abs_tol=1e-12):
            self._pending_dataset_weights.pop(dataset_id, None)
        else:
            self._pending_dataset_weights[dataset_id] = value
        self._refresh_footer_dirty()
        self._refresh_footer_validity()

    def _selected_targets(self, dataset: FitDatasetLocalSettings) -> tuple[str, ...]:
        return self._pending_target_selections.get(
            dataset.dataset_id,
            dataset.target_species,
        )

    def _eligible_targets(self, dataset_id: str) -> tuple[str, ...]:
        dataset = self._projected_dataset(dataset_id)
        mapped = self._mapped_dataset(dataset_id)
        if dataset is None or mapped is None:
            return ()
        return tuple(
            dict.fromkeys(
                target
                for observation in mapped.observations
                for target in (self._observation_target(dataset, observation),)
                if target
            )
        )

    def _dataset_overlay(
        self,
        dataset_id: str,
    ) -> FitDatasetLocalSettings | None:
        """Overlay only this surface's fields on the latest owner row."""

        current = self._projected_dataset(dataset_id)
        mapped = self._mapped_dataset(dataset_id)
        if current is None or mapped is None:
            return None
        selected = self._selected_targets(current)
        invalid_selected = [
            target
            for target in selected
            if ("target_weight", dataset_id, target) in self._invalid_raw
        ]
        if invalid_selected or ("dataset_weight", dataset_id) in self._invalid_raw:
            return None
        current_weights = dict(current.target_weights)
        target_weights = tuple(
            (
                target,
                self._pending_target_weights.get(
                    (dataset_id, target),
                    current_weights.get(target, 1.0),
                ),
            )
            for target in selected
        )
        current_raw_by_target = {
            target: raw_id for raw_id, target in current.observation_targets
        }
        eligible_raw_by_target: dict[str, str] = {}
        for observation in mapped.observations:
            target = self._observation_target(current, observation)
            eligible_raw_by_target.setdefault(
                target,
                observation.raw_observation_id,
            )
        try:
            observation_targets = tuple(
                (
                    current_raw_by_target.get(target)
                    or eligible_raw_by_target[target],
                    target,
                )
                for target in selected
            )
            return replace(
                current,
                weight=self._pending_dataset_weights.get(
                    dataset_id,
                    current.weight,
                ),
                target_species=selected,
                target_weights=target_weights,
                observation_targets=observation_targets,
            )
        except (KeyError, TypeError, ValueError):
            return None

    def _refresh_footer_dirty(self) -> None:
        self._footer.set_dirty(
            self.has_pending_target_changes
            or bool(self._pending_initials)
            or bool(self._invalid_raw)
        )

    @staticmethod
    def _initial_field(column: int) -> str | None:
        return {
            _Col.INITIAL: "initial",
            _Col.FIT_IC: "fit",
            _Col.LOG10: "log10",
            _Col.MIN: "minimum",
            _Col.MAX: "maximum",
        }.get(column)

    def _stage_initial_item_edit(self, item: QtWidgets.QTableWidgetItem) -> None:
        dataset_id = str(item.data(_DATASET_ROLE) or "")
        set_id = str(item.data(_SET_ROLE) or "")
        species = str(item.data(_SPECIES_ROLE) or "")
        current = self._initial_variable(set_id, species)
        field = self._initial_field(item.column())
        if current is None or field is None:
            return
        if species in self._unresolved_global_species:
            self._update_unresolved_global_draft(item, species)
            return
        raw_key = (
            "initial",
            dataset_id,
            set_id,
            species,
            field,
        )
        try:
            if field in {"fit", "log10"}:
                state = item.checkState()
                if state == Qt.CheckState.PartiallyChecked:
                    raise ValueError("Choose an explicit checked or unchecked state.")
                value: object = state == Qt.CheckState.Checked
            else:
                value = float(item.text())
                if not math.isfinite(value):
                    raise ValueError("Initial-condition values must be finite.")
            replacement = replace(current, **{field: value})
            if field == "fit" and not bool(value):
                replacement = replace(replacement, log10=False)
        except (TypeError, ValueError, OverflowError) as exc:
            self._invalid_raw[raw_key] = (
                item.text()
                if field not in {"fit", "log10"}
                else str(int(item.checkState()))
            )
            self._mark_invalid(item, str(exc))
            return
        self._invalid_raw.pop(raw_key, None)
        self._clear_invalid(item)
        if current.scope == "global" and set_id in self._included_set_ids():
            for active_set_id in self._included_set_ids():
                active = self._initial_variable(active_set_id, species)
                if active is None:
                    continue
                self._set_pending_initial(
                    replace(
                        active,
                        initial=replacement.initial,
                        minimum=replacement.minimum,
                        maximum=replacement.maximum,
                        fit=replacement.fit,
                        log10=replacement.log10,
                        scope="global",
                    )
                )
        else:
            self._set_pending_initial(replacement)
        self._refresh_footer_dirty()
        self._refresh_footer_validity()

    def _set_pending_initial(self, initial: FitInitialVariable) -> None:
        projected = next(
            (
                item
                for item in (self._current_setup().initials if self._current_setup() else ())
                if item.set_id == initial.set_id and item.species == initial.species
            ),
            None,
        )
        key = (initial.set_id, initial.species)
        if projected == initial:
            self._pending_initials.pop(key, None)
        else:
            self._pending_initials[key] = initial

    def _apply_presentation_edits(self) -> None:
        """Overlay owned fields on latest rows and emit one atomic edit tuple."""

        if self._invalid_raw:
            self._refresh_footer_validity()
            return
        dataset_ids = {
            *self._pending_target_selections,
            *self._pending_dataset_weights,
            *(dataset_id for dataset_id, _target in self._pending_target_weights),
        }
        overlays = tuple(
            overlay
            for dataset in (self._current_setup().datasets if self._current_setup() else ())
            if dataset.dataset_id in dataset_ids
            for overlay in (self._dataset_overlay(dataset.dataset_id),)
            if overlay is not None
        )
        dataset_intents = tuple(
            FitDatasetSetupEdit(
                origin_tab_id=self._origin.tab_id,
                origin_capture_id=self._origin.capture_id,
                dataset=overlay,
            )
            for overlay in overlays
        )
        initial_intents = tuple(
            FitInitialSetupEdit(
                origin_tab_id=self._origin.tab_id,
                origin_capture_id=self._origin.capture_id,
                initial=initial,
            )
            for initial in self._ordered_pending_initials()
        )
        intents = (*dataset_intents, *initial_intents)
        if not intents:
            self.set_status("No pending Fit target or initial-condition changes.")
            return
        self.setupEditsRequested.emit(intents)
        if self._acknowledge_applied_intents(intents):
            self.targetsApplied.emit()
            self.set_status("Fit target and initial-condition edits applied.")
        self._refresh_visible_presentation()

    def _revert_presentation_edits(self) -> None:
        self._pending_target_selections.clear()
        self._pending_target_weights.clear()
        self._pending_dataset_weights.clear()
        self._pending_initials.clear()
        self._invalid_raw.clear()
        self._unresolved_global_species.clear()
        self._pre_global_local_backups.clear()
        self._global_draft_raw.clear()
        self._global_draft_checks.clear()
        self._refresh_visible_presentation()
        self.set_status("Pending Fit target and initial-condition edits reverted.")

    def _apply_bulk_action(self, action: str) -> None:
        if self._current_dataset() is None:
            return
        self._table.blockSignals(True)
        try:
            for row in range(self._table.rowCount()):
                item = self._table.item(row, _Col.INCLUDE)
                if item is None or not item.flags() & Qt.ItemFlag.ItemIsUserCheckable:
                    continue
                if action == "all":
                    state = Qt.CheckState.Checked
                elif action == "none":
                    state = Qt.CheckState.Unchecked
                elif action == "invert":
                    state = (
                        Qt.CheckState.Unchecked
                        if item.checkState() == Qt.CheckState.Checked
                        else Qt.CheckState.Checked
                    )
                else:
                    return
                item.setCheckState(state)
        finally:
            self._table.blockSignals(False)
        self._stage_visible_target_selection(self._current_dataset_id or "")

    def _copy_targets_to_compatible_datasets(self) -> None:
        setup = self._current_setup()
        source = self._current_dataset()
        if source is None:
            return
        source_selection = FitTargetSelection(
            dataset_id=source.dataset_id,
            eligible_targets=self._eligible_targets(source.dataset_id),
            selected_targets=self._selected_targets(source),
        )
        if not source_selection.selected_targets:
            self.set_status("No targets selected to copy")
            return
        destinations = tuple(
            FitTargetSelection(
                dataset_id=destination.dataset_id,
                eligible_targets=self._eligible_targets(destination.dataset_id),
                selected_targets=self._selected_targets(destination),
            )
            for destination in setup.datasets
            if destination.included and destination.dataset_id != source.dataset_id
        )
        copied_selections, copied, skipped = copy_target_selection(
            source_selection,
            destinations,
        )
        for selection in copied_selections:
            projected = self._projected_dataset(selection.dataset_id)
            if projected is None:
                continue
            if selection.selected_targets == projected.target_species:
                self._pending_target_selections.pop(selection.dataset_id, None)
            else:
                self._pending_target_selections[
                    selection.dataset_id
                ] = selection.selected_targets
        self._refresh_visible_presentation()
        self.set_status(f"Copy Targets: {copied} copied, {skipped} skipped.")

    # ------------------------------------------------------------------
    # Global / Local initial-concentration presentation
    # ------------------------------------------------------------------
    def _included_set_ids(
        self,
        snapshot: FitSetupSnapshot | None = None,
    ) -> tuple[str, ...]:
        setup = snapshot or self._current_setup()
        if setup is None:
            return ()
        included_dataset_ids = {
            dataset.dataset_id for dataset in setup.datasets if dataset.included
        }
        return tuple(
            mapped.set_id
            for mapped in self._origin.mapped_datasets
            if mapped.dataset_id in included_dataset_ids
        )

    @staticmethod
    def _initial_signature(initial: FitInitialVariable) -> tuple[object, ...]:
        return (
            initial.initial,
            initial.minimum,
            initial.maximum,
            initial.fit,
            initial.log10,
        )

    def _active_initials(self, species: str) -> tuple[FitInitialVariable, ...]:
        return tuple(
            initial
            for set_id in self._included_set_ids()
            for initial in (self._initial_variable(set_id, species),)
            if initial is not None
        )

    def _refresh_initial_scope_controls(self) -> None:
        if not hasattr(self, "_initial_scope_species"):
            return
        species = str(self._initial_scope_species.currentText() or "").strip()
        active = self._active_initials(species) if species else ()
        count = len(self._included_set_ids())
        self._initial_scope_dataset_label.setText(
            f"Applies to {count} selected dataset{'s' if count != 1 else ''}."
        )
        enabled = bool(species and active and not self._running)
        self._initial_scope_species.setEnabled(not self._running)
        self._initial_global_radio.setEnabled(enabled)
        self._initial_local_radio.setEnabled(enabled)
        previous_global = self._initial_global_radio.blockSignals(True)
        previous_local = self._initial_local_radio.blockSignals(True)
        try:
            if species in self._unresolved_global_species:
                self._initial_global_radio.setChecked(True)
                self._initial_scope_status.setText(
                    "Global setup: Multiple values. Enter one explicit value, "
                    "bounds, and Fit/Log10 selection in the species row."
                )
                return
            scopes = {item.scope for item in active}
            if scopes == {"global"}:
                self._initial_global_radio.setChecked(True)
                self._initial_scope_status.setText(
                    "Global: one parameter updates the initial value for all "
                    "selected datasets."
                )
            elif scopes == {"local"}:
                self._initial_local_radio.setChecked(True)
                self._initial_scope_status.setText(
                    "Local: a separate parameter is used for each selected dataset."
                )
            else:
                self._initial_global_radio.setChecked(False)
                self._initial_local_radio.setChecked(False)
                self._initial_scope_status.setText(
                    "Initial scope is unavailable until included rows agree."
                )
        finally:
            self._initial_global_radio.blockSignals(previous_global)
            self._initial_local_radio.blockSignals(previous_local)

    def _on_initial_scope_toggled(self, checked: bool) -> None:
        if self._refreshing or self._running or not checked:
            return
        species = str(self._initial_scope_species.currentText() or "").strip()
        if not species:
            return
        if self._initial_global_radio.isChecked():
            self._stage_global_scope(species)
        elif self._initial_local_radio.isChecked():
            self._stage_local_scope(species)
        self._refresh_visible_presentation()

    def _stage_global_scope(self, species: str) -> None:
        active = self._active_initials(species)
        if not active:
            return
        if {item.scope for item in active} == {"global"} and len(
            {self._initial_signature(item) for item in active}
        ) == 1:
            return
        self._pre_global_local_backups.setdefault(
            species,
            {(item.set_id, species): item for item in active},
        )
        signatures = {self._initial_signature(item) for item in active}
        if len(signatures) == 1:
            template = active[0]
            for initial in active:
                self._set_pending_initial(
                    replace(
                        initial,
                        initial=template.initial,
                        minimum=template.minimum,
                        maximum=template.maximum,
                        fit=template.fit,
                        log10=template.log10,
                        scope="global",
                    )
                )
            self._clear_unresolved_global(species)
            self.set_status(
                f"Global {species} initial staged for all included datasets."
            )
            return
        self._unresolved_global_species.add(species)
        self._invalid_raw[("initial_scope", species)] = "Multiple values"
        for field in ("initial", "minimum", "maximum"):
            self._global_draft_raw[(species, field)] = "Multiple values"
        self._global_draft_checks[(species, "fit")] = Qt.CheckState.PartiallyChecked
        self._global_draft_checks[(species, "log10")] = Qt.CheckState.PartiallyChecked
        self.set_status(
            f"Global {species} requires one explicit complete setup."
        )

    def _stage_local_scope(self, species: str) -> None:
        backup = self._pre_global_local_backups.get(species)
        if species in self._unresolved_global_species or backup is not None:
            for initial in (backup or {}).values():
                self._set_pending_initial(initial)
            self._pre_global_local_backups.pop(species, None)
            self._clear_unresolved_global(species)
            self.set_status(
                f"Restored the untouched Local {species} setups."
            )
            return
        active = self._active_initials(species)
        if not active or {item.scope for item in active} == {"local"}:
            return
        for initial in active:
            self._set_pending_initial(replace(initial, scope="local"))
        self.set_status(
            f"Local {species} initials staged from the accepted Global setup."
        )

    def _clear_unresolved_global(self, species: str) -> None:
        self._unresolved_global_species.discard(species)
        self._invalid_raw.pop(("initial_scope", species), None)
        for key in tuple(self._global_draft_raw):
            if key[0] == species:
                self._global_draft_raw.pop(key, None)
        for key in tuple(self._global_draft_checks):
            if key[0] == species:
                self._global_draft_checks.pop(key, None)

    def _restore_initial_raw_presentation(
        self,
        *,
        row: int,
        set_id: str,
        species: str,
    ) -> None:
        field_by_column = {
            _Col.INITIAL: "initial",
            _Col.MIN: "minimum",
            _Col.MAX: "maximum",
        }
        if species in self._unresolved_global_species and set_id in self._included_set_ids():
            for column, field in field_by_column.items():
                item = self._table.item(row, column)
                if item is not None:
                    item.setText(
                        self._global_draft_raw.get(
                            (species, field),
                            "Multiple values",
                        )
                    )
                    item.setBackground(_INVALID_BG)
                    item.setForeground(_INVALID_FG)
            for column, field in (
                (_Col.FIT_IC, "fit"),
                (_Col.LOG10, "log10"),
            ):
                item = self._table.item(row, column)
                if item is not None:
                    item.setFlags(
                        Qt.ItemFlag.ItemIsEnabled
                        | Qt.ItemFlag.ItemIsSelectable
                        | Qt.ItemFlag.ItemIsUserCheckable
                    )
                    item.setCheckState(
                        self._global_draft_checks.get(
                            (species, field),
                            Qt.CheckState.PartiallyChecked,
                        )
                    )
            return
        for column, field in field_by_column.items():
            key = (
                "initial",
                self._dataset_id_for_set(set_id),
                set_id,
                species,
                field,
            )
            item = self._table.item(row, column)
            if item is not None and key in self._invalid_raw:
                item.setText(self._invalid_raw[key])
                item.setBackground(_INVALID_BG)
                item.setForeground(_INVALID_FG)

    def _update_unresolved_global_draft(
        self,
        item: QtWidgets.QTableWidgetItem,
        species: str,
    ) -> None:
        field = self._initial_field(item.column())
        if field is None:
            return
        if field in {"fit", "log10"}:
            self._global_draft_checks[(species, field)] = item.checkState()
        else:
            self._global_draft_raw[(species, field)] = item.text()
        try:
            values = {
                field_name: float(self._global_draft_raw[(species, field_name)])
                for field_name in ("initial", "minimum", "maximum")
            }
            if not all(math.isfinite(value) for value in values.values()):
                raise ValueError
            fit_state = self._global_draft_checks.get(
                (species, "fit"),
                Qt.CheckState.PartiallyChecked,
            )
            log_state = self._global_draft_checks.get(
                (species, "log10"),
                Qt.CheckState.PartiallyChecked,
            )
            if fit_state == Qt.CheckState.PartiallyChecked:
                return
            fit = fit_state == Qt.CheckState.Checked
            if not fit:
                log10 = False
            elif log_state == Qt.CheckState.PartiallyChecked:
                return
            else:
                log10 = log_state == Qt.CheckState.Checked
            active = self._active_initials(species)
            replacements = tuple(
                replace(
                    initial,
                    initial=values["initial"],
                    minimum=values["minimum"],
                    maximum=values["maximum"],
                    fit=fit,
                    log10=log10,
                    scope="global",
                )
                for initial in active
            )
        except (KeyError, TypeError, ValueError, OverflowError):
            self._refresh_footer_validity()
            return
        for replacement in replacements:
            self._set_pending_initial(replacement)
        self._clear_unresolved_global(species)
        self._refresh_visible_presentation()
        self.set_status(
            f"Complete Global {species} initial staged for all included datasets."
        )

    def _ordered_pending_initials(self) -> tuple[FitInitialVariable, ...]:
        setup = self._current_setup()
        if setup is None:
            return ()
        return tuple(
            self._pending_initials[key]
            for item in setup.initials
            for key in ((item.set_id, item.species),)
            if key in self._pending_initials
        )

    def _acknowledge_applied_intents(self, intents: tuple[object, ...]) -> bool:
        setup = self._current_setup()
        if setup is None:
            return False
        datasets = {item.dataset_id: item for item in setup.datasets}
        initials = {(item.set_id, item.species): item for item in setup.initials}
        if any(
            isinstance(intent, FitDatasetSetupEdit)
            and datasets.get(intent.dataset.dataset_id) != intent.dataset
            or isinstance(intent, FitInitialSetupEdit)
            and initials.get((intent.initial.set_id, intent.initial.species))
            != intent.initial
            for intent in intents
        ):
            return False
        for intent in intents:
            if isinstance(intent, FitDatasetSetupEdit):
                dataset_id = intent.dataset.dataset_id
                self._pending_target_selections.pop(dataset_id, None)
                self._pending_dataset_weights.pop(dataset_id, None)
                for key in tuple(self._pending_target_weights):
                    if key[0] == dataset_id:
                        self._pending_target_weights.pop(key, None)
            elif isinstance(intent, FitInitialSetupEdit):
                self._pending_initials.pop(
                    (intent.initial.set_id, intent.initial.species),
                    None,
                )
        for species in tuple(self._pre_global_local_backups):
            active = self._active_initials(species)
            if active and {item.scope for item in active} == {"global"}:
                self._pre_global_local_backups.pop(species, None)
        return True

    def _global_inclusion_transition_edits(
        self,
        previous: FitSetupSnapshot | None,
        current: FitSetupSnapshot,
    ) -> tuple[FitInitialSetupEdit, ...]:
        if previous is None:
            return ()
        old_included = {
            item.dataset_id for item in previous.datasets if item.included
        }
        newly_included = tuple(
            item.dataset_id
            for item in current.datasets
            if item.included and item.dataset_id not in old_included
        )
        if not newly_included:
            return ()
        old_set_ids = {
            mapped.set_id
            for mapped in self._origin.mapped_datasets
            if mapped.dataset_id in old_included
        }
        new_set_ids = {
            mapped.dataset_id: mapped.set_id
            for mapped in self._origin.mapped_datasets
            if mapped.dataset_id in newly_included
        }
        old_by_species: dict[str, list[FitInitialVariable]] = {}
        for initial in previous.initials:
            if (
                initial.set_id in old_set_ids
                or not old_set_ids and initial.scope == "global"
            ):
                old_by_species.setdefault(initial.species, []).append(initial)
        current_by_key = {
            (item.set_id, item.species): item for item in current.initials
        }
        edits: list[FitInitialSetupEdit] = []
        for species, old_rows in old_by_species.items():
            if (
                {item.scope for item in old_rows} != {"global"}
                or len({self._initial_signature(item) for item in old_rows}) != 1
            ):
                continue
            template = old_rows[0]
            for dataset_id in newly_included:
                set_id = new_set_ids.get(dataset_id)
                row = current_by_key.get((str(set_id or ""), species))
                if row is None:
                    continue
                replacement = replace(
                    row,
                    initial=template.initial,
                    minimum=template.minimum,
                    maximum=template.maximum,
                    fit=template.fit,
                    log10=template.log10,
                    scope="global",
                )
                if replacement == row:
                    continue
                edits.append(
                    FitInitialSetupEdit(
                        origin_tab_id=self._origin.tab_id,
                        origin_capture_id=self._origin.capture_id,
                        initial=replacement,
                    )
                )
        return tuple(edits)

    def _resolve_unresolved_globals_if_identical(self) -> None:
        included = set(self._included_set_ids())
        for species in tuple(self._unresolved_global_species):
            backup = self._pre_global_local_backups.get(species, {})
            remaining = tuple(
                initial
                for (set_id, _species), initial in backup.items()
                if set_id in included
            )
            if not remaining or len(
                {self._initial_signature(item) for item in remaining}
            ) != 1:
                continue
            template = remaining[0]
            for initial in remaining:
                self._set_pending_initial(
                    replace(
                        initial,
                        initial=template.initial,
                        minimum=template.minimum,
                        maximum=template.maximum,
                        fit=template.fit,
                        log10=template.log10,
                        scope="global",
                    )
                )
            self._clear_unresolved_global(species)

    # ------------------------------------------------------------------
    # Weight-mode and validation presentation
    # ------------------------------------------------------------------
    def _on_weight_mode_changed(self) -> None:
        if self._refreshing or self._running or self._current_setup() is None:
            return
        mode = str(self._weight_mode_combo.currentData() or "implicit")
        if mode == self._current_setup().dataset_weight_mode:
            self._refresh_dataset_weight_editor()
            return
        intent = FitDatasetWeightModeSetupEdit(
            origin_tab_id=self._origin.tab_id,
            origin_capture_id=self._origin.capture_id,
            dataset_weight_mode=mode,
        )
        self.setupEditsRequested.emit((intent,))
        self._refresh_dataset_weight_editor()

    def _refresh_dataset_weight_editor(self) -> None:
        dataset = self._current_dataset()
        if dataset is None:
            self._context_label.setText("Selected dataset: —")
            self._dataset_weight_edit.clear()
            self._weight_mode_combo.setEnabled(False)
            self._dataset_weight_edit.setEnabled(False)
            return
        mapped = self._mapped_dataset(dataset.dataset_id)
        label = mapped.display_name if mapped is not None else dataset.dataset_id
        compact = compact_dataset_label(
            label,
            max_chars=DATASET_LIST_LABEL_MAX_CHARS,
        )
        self._context_label.setText(f"Selected dataset: {compact.display}")
        tooltip = f"{compact.full}\nDataset ID: {dataset.dataset_id}"
        if mapped is not None:
            tooltip += f"\nSet ID: {mapped.set_id}"
        self._context_label.setToolTip(tooltip)
        self._weight_mode_combo.setEnabled(not self._running)
        invalid_key = ("dataset_weight", dataset.dataset_id)
        if invalid_key in self._invalid_raw:
            self._dataset_weight_edit.setText(self._invalid_raw[invalid_key])
            self._mark_invalid_line_edit(
                self._dataset_weight_edit,
                "Dataset weight must be finite and positive.",
            )
        else:
            self._dataset_weight_edit.setText(
                self._float_text(
                    self._pending_dataset_weights.get(
                        dataset.dataset_id,
                        dataset.weight,
                    )
                )
            )
            self._clear_invalid_line_edit(self._dataset_weight_edit)
        self._dataset_weight_edit.setEnabled(
            not self._running
            and self._weight_mode_combo.currentData() == "user"
        )

    def _mark_invalid(self, item: QtWidgets.QTableWidgetItem, message: str) -> None:
        previous_blocked = self._table.blockSignals(True)
        try:
            item.setBackground(_INVALID_BG)
            item.setForeground(_INVALID_FG)
        finally:
            self._table.blockSignals(previous_blocked)
        self._set_footer_error(message)
        self.validityChanged.emit()

    def _clear_invalid(self, item: QtWidgets.QTableWidgetItem) -> None:
        previous_blocked = self._table.blockSignals(True)
        try:
            item.setBackground(_DEFAULT_BG)
            item.setForeground(_DEFAULT_FG)
        finally:
            self._table.blockSignals(previous_blocked)

    def _mark_invalid_line_edit(self, edit: QtWidgets.QLineEdit, message: str) -> None:
        edit.setStyleSheet("background: rgb(255, 200, 200); color: rgb(80, 0, 0);")
        self._set_footer_error(message)
        self.validityChanged.emit()

    def _clear_invalid_line_edit(self, edit: QtWidgets.QLineEdit) -> None:
        edit.setStyleSheet("")

    def _set_footer_error(self, message: str | None) -> None:
        self._footer.set_error(message)

    def _refresh_footer_validity(self) -> None:
        message = (
            "Correct invalid target or initial-condition values before running."
            if self._invalid_raw
            else None
        )
        self._set_footer_error(message)
        self.validityChanged.emit()

    @property
    def has_pending_target_changes(self) -> bool:
        return bool(
            self._pending_target_selections
            or self._pending_target_weights
            or self._pending_dataset_weights
        )

    @property
    def invalid_editor_keys(self) -> tuple[tuple[str, ...], ...]:
        return tuple(sorted(self._invalid_raw))

    @property
    def invalid_raw_values(self) -> dict[tuple[str, ...], str]:
        return dict(self._invalid_raw)

    @property
    def has_pending_initial_changes(self) -> bool:
        return bool(self._pending_initials)


__all__ = ["UnifiedSpeciesTable"]
