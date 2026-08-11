"""Presentation for the historical unified Fit target/initial table.

The table is deliberately a view/editor surface.  Accepted Fit setup remains
owned by :class:`FitSessionOwner`; this widget only renders the owner supplied
snapshot and emits existing typed setup intents for user edits.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Sequence

from PySide6 import QtCore, QtWidgets
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
    """Data-and-Targets column indices."""

    INCLUDE = 0
    WEIGHT = 1
    SPECIES = 2
    INITIAL = 3
    FIT = 4
    LOG10 = 5
    MINIMUM = 6
    MAXIMUM = 7
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
_EXACT_VALUE_ROLE = Qt.ItemDataRole.UserRole + 6


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
    # setupEditsRequested signal. The table owns pending target/weight drafts
    # locally; accepted-state acknowledgement comes only from the owner.
    setupEditsRequested = Signal(object)
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
        self._pending_initial_rows: dict[tuple[str, str], dict[str, object]] = {}
        self._pending_initial_values: dict[
            tuple[str, str], dict[str, object]
        ] = {}
        self._invalid_raw: dict[tuple[str, ...], str] = {}
        self._invalid_initial_raw: dict[tuple[str, str, str], str] = {}
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
            show_divider=True,
            apply_requires_no_error=False,
            button_order=("apply", "revert"),
            error_object_name="global_fit_species_table_error",
            apply_object_name="global_fit_species_table_apply",
            revert_object_name="global_fit_species_table_revert",
        )
        group_layout.addWidget(self._footer, stretch=1)
        self._footer.applyRequested.connect(self._apply_presentation_edits)
        self._footer.revertRequested.connect(self._revert_presentation_edits)

        self._context_label = QtWidgets.QLabel("Selected dataset: —", group)
        self._context_label.setStyleSheet("font-weight: bold;")
        self._footer.body_layout.addWidget(self._context_label)

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

    def set_origin(
        self,
        origin: FitOriginSnapshot,
        *,
        preserve_drafts: bool = False,
    ) -> None:
        """Install a new captured origin without introducing a second store."""

        if not isinstance(origin, FitOriginSnapshot):
            raise TypeError("origin must be FitOriginSnapshot.")
        if origin.capture_id != self._origin.capture_id and not preserve_drafts:
            self._pending_target_selections.clear()
            self._pending_target_weights.clear()
            self._pending_dataset_weights.clear()
            self._pending_initial_rows.clear()
            self._pending_initial_values.clear()
            self._invalid_raw.clear()
            self._invalid_initial_raw.clear()
        self._origin = origin

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

        self._projected_setup = snapshot
        for dataset in snapshot.datasets:
            self._purge_parametric_x_pending(dataset)
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
        finally:
            self._dataset_weight_edit.blockSignals(False)
            self._weight_mode_combo.blockSignals(False)
            self._table.blockSignals(False)
            self._refreshing = False
            self.setUpdatesEnabled(True)
        self._refresh_footer_dirty()
        self._refresh_footer_validity()

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

    @staticmethod
    def _float_text(value: float) -> str:
        # ``repr`` is the display projection of the accepted binary64 value;
        # it round-trips without silently changing scientific state.
        return repr(float(value))

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

    def _local_initial(
        self,
        dataset: FitDatasetLocalSettings,
        species: str,
    ) -> FitInitialVariable | None:
        setup = self._current_setup()
        if setup is None:
            return None
        mapped = self._mapped_dataset(dataset.dataset_id)
        if mapped is None:
            return None
        return next(
            (
                item
                for item in setup.initials
                if item.set_id == mapped.set_id and item.species == species
            ),
            None,
        )

    @staticmethod
    def _initial_raw(initial: FitInitialVariable) -> dict[str, object]:
        return {
            "initial": UnifiedSpeciesTable._float_text(initial.initial),
            "minimum": UnifiedSpeciesTable._float_text(initial.minimum),
            "maximum": UnifiedSpeciesTable._float_text(initial.maximum),
            "fit": bool(initial.fit),
            "log10": bool(initial.log10),
        }

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
        self._purge_parametric_x_pending(dataset)
        x_name = dataset.parametric.x_name if dataset.parametric is not None else None
        selected = set(
            self._pending_target_selections.get(
                dataset.dataset_id,
                dataset.target_species,
            )
        )
        if x_name is not None:
            selected.discard(x_name)
        weights = dict(dataset.target_weights)
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
            local = self._local_initial(dataset, species)
            local_key = (
                mapped.set_id if mapped is not None else "",
                species,
            )
            raw_initial = self._pending_initial_rows.get(
                local_key,
                self._initial_raw(local) if local is not None else {},
            )
            target_item = QtWidgets.QTableWidgetItem()
            eligible = has_data and species != x_name
            self._set_checkable(target_item, species in selected, eligible)
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
                weight_text if eligible else ""
            )
            weight_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._set_editable(weight_item, eligible)
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

            for column, field in (
                (_Col.INITIAL, "initial"),
                (_Col.MINIMUM, "minimum"),
                (_Col.MAXIMUM, "maximum"),
            ):
                invalid_key = (mapped.set_id, species, field)
                initial_item = QtWidgets.QTableWidgetItem(
                    self._invalid_initial_raw.get(
                        invalid_key,
                        str(raw_initial.get(field, "")),
                    )
                )
                self._set_editable(initial_item, local is not None)
                initial_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                initial_item.setData(_DATASET_ROLE, dataset.dataset_id)
                initial_item.setData(_SET_ROLE, mapped.set_id)
                initial_item.setData(_SPECIES_ROLE, species)
                if local is not None:
                    initial_item.setData(_EXACT_VALUE_ROLE, getattr(local, field))
                if invalid_key in self._invalid_initial_raw:
                    initial_item.setBackground(_INVALID_BG)
                    initial_item.setForeground(_INVALID_FG)
                    initial_item.setToolTip(self._invalid_initial_raw[invalid_key])
                self._table.setItem(row, column, initial_item)

            for column, field in ((_Col.FIT, "fit"), (_Col.LOG10, "log10")):
                check_item = QtWidgets.QTableWidgetItem()
                self._set_checkable(
                    check_item,
                    bool(raw_initial.get(field, False)),
                    local is not None and (field == "fit" or bool(raw_initial.get("fit", False))),
                )
                check_item.setData(_DATASET_ROLE, dataset.dataset_id)
                check_item.setData(_SET_ROLE, mapped.set_id)
                check_item.setData(_SPECIES_ROLE, species)
                if local is not None:
                    check_item.setData(
                        _EXACT_VALUE_ROLE,
                        getattr(local, field),
                    )
                self._table.setItem(row, column, check_item)


    def _refresh_visible_presentation(self) -> None:
        self._refreshing = True
        self.setUpdatesEnabled(False)
        self._table.blockSignals(True)
        self._weight_mode_combo.blockSignals(True)
        self._dataset_weight_edit.blockSignals(True)
        try:
            self._populate_table()
            self._refresh_dataset_weight_editor()
        finally:
            self._dataset_weight_edit.blockSignals(False)
            self._weight_mode_combo.blockSignals(False)
            self._table.blockSignals(False)
            self._refreshing = False
            self.setUpdatesEnabled(True)
        self._refresh_footer_dirty()
        self._refresh_footer_validity()

    def _purge_parametric_x_pending(
        self,
        dataset: FitDatasetLocalSettings,
    ) -> None:
        """Drop target drafts that are invalid for this dataset's current X."""

        parametric = dataset.parametric
        if parametric is None:
            return
        dataset_id = dataset.dataset_id
        x_name = parametric.x_name
        pending = self._pending_target_selections.get(dataset_id)
        if pending is not None:
            filtered = tuple(target for target in pending if target != x_name)
            if filtered == dataset.target_species:
                self._pending_target_selections.pop(dataset_id, None)
            else:
                self._pending_target_selections[dataset_id] = filtered
        self._pending_target_weights.pop((dataset_id, x_name), None)
        self._invalid_raw.pop(("target_weight", dataset_id, x_name), None)

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
        if item.column() in {
            _Col.INITIAL,
            _Col.FIT,
            _Col.LOG10,
            _Col.MINIMUM,
            _Col.MAXIMUM,
        }:
            self._stage_initial_row(item)
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
            if value == baseline:
                self._pending_target_weights.pop((dataset_id, species), None)
            else:
                self._pending_target_weights[(dataset_id, species)] = value
            self._refresh_footer_dirty()
            self._refresh_footer_validity()
            return
        if item.column() == _Col.INCLUDE:
            self._stage_visible_target_selection(dataset_id)
            return

    def _stage_initial_row(self, item: QtWidgets.QTableWidgetItem) -> None:
        """Reconstruct and validate one complete Local-IC row Draft."""

        set_id = str(item.data(_SET_ROLE) or "").strip()
        species = str(item.data(_SPECIES_ROLE) or "").strip()
        if not set_id or not species:
            return
        items = {
            "initial": self._table.item(item.row(), _Col.INITIAL),
            "minimum": self._table.item(item.row(), _Col.MINIMUM),
            "maximum": self._table.item(item.row(), _Col.MAXIMUM),
        }
        fit_item = self._table.item(item.row(), _Col.FIT)
        log_item = self._table.item(item.row(), _Col.LOG10)
        if (
            any(value is None for value in items.values())
            or fit_item is None
            or log_item is None
        ):
            return
        setup = self._current_setup()
        accepted = next(
            (
                row
                for row in (setup.initials if setup is not None else ())
                if row.set_id == set_id and row.species == species
            ),
            None,
        )
        if accepted is None:
            return
        key = (set_id, species)
        fit = fit_item.checkState() == Qt.CheckState.Checked
        log10 = log_item.checkState() == Qt.CheckState.Checked
        if not fit and log10:
            blocker = QtCore.QSignalBlocker(self._table)
            log_item.setCheckState(Qt.CheckState.Unchecked)
            del blocker
            log10 = False
        raw = {
            field: str(value.text())
            for field, value in items.items()
            if value is not None
        }
        raw.update({"fit": fit, "log10": log10})
        self._pending_initial_rows[key] = raw

        typed: dict[str, object] = {"fit": fit, "log10": log10}
        parse_errors: dict[str, str] = {}
        for field, field_item in items.items():
            assert field_item is not None
            try:
                parsed = float(field_item.text())
                if not math.isfinite(parsed):
                    raise ValueError("value must be finite")
            except (TypeError, ValueError, OverflowError) as exc:
                parse_errors[field] = str(exc)
            else:
                typed[field] = parsed
        self._clear_initial_row_errors(set_id, species, item.row())
        if parse_errors:
            self._pending_initial_values.pop(key, None)
            for field, message in parse_errors.items():
                field_key = (set_id, species, field)
                self._invalid_initial_raw[field_key] = str(raw[field])
                self._decorate_initial_field(
                    item.row(),
                    {
                        "initial": _Col.INITIAL,
                        "minimum": _Col.MINIMUM,
                        "maximum": _Col.MAXIMUM,
                    }[field],
                    message,
                )
            self._refresh_footer_dirty()
            self._refresh_footer_validity()
            return
        try:
            initial = FitInitialVariable(
                set_id=set_id,
                species=species,
                initial=float(typed["initial"]),
                minimum=float(typed["minimum"]),
                maximum=float(typed["maximum"]),
                fit=bool(typed["fit"]),
                log10=bool(typed["log10"]),
            )
        except (TypeError, ValueError, OverflowError) as exc:
            self._pending_initial_values.pop(key, None)
            for field, column in (
                ("initial", _Col.INITIAL),
                ("minimum", _Col.MINIMUM),
                ("maximum", _Col.MAXIMUM),
            ):
                self._invalid_initial_raw[(set_id, species, field)] = str(raw[field])
                self._decorate_initial_field(item.row(), column, str(exc))
        else:
            self._clear_initial_row_errors(set_id, species, item.row())
            self._pending_initial_values[key] = {
                "initial": initial.initial,
                "minimum": initial.minimum,
                "maximum": initial.maximum,
                "fit": initial.fit,
                "log10": initial.log10,
            }
            if accepted is not None and initial == accepted:
                self._pending_initial_rows.pop(key, None)
                self._pending_initial_values.pop(key, None)
        self._refresh_footer_dirty()
        self._refresh_footer_validity()

    def _clear_initial_row_errors(
        self,
        set_id: str,
        species: str,
        row: int,
    ) -> None:
        for field, column in (
            ("initial", _Col.INITIAL),
            ("minimum", _Col.MINIMUM),
            ("maximum", _Col.MAXIMUM),
        ):
            self._invalid_initial_raw.pop((set_id, species, field), None)
            self._decorate_initial_field(row, column, None)

    def _decorate_initial_field(
        self,
        row: int,
        column: int,
        message: str | None,
    ) -> None:
        blocker = QtCore.QSignalBlocker(self._table)
        item = self._table.item(row, column)
        if item is not None:
            item.setBackground(_INVALID_BG if message else _DEFAULT_BG)
            item.setForeground(_INVALID_FG if message else _DEFAULT_FG)
            item.setToolTip(message or "")
        del blocker

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
        if value == projected.weight:
            self._pending_dataset_weights.pop(dataset_id, None)
        else:
            self._pending_dataset_weights[dataset_id] = value
        self._refresh_footer_dirty()
        self._refresh_footer_validity()

    def _selected_targets(self, dataset: FitDatasetLocalSettings) -> tuple[str, ...]:
        selected = self._pending_target_selections.get(
            dataset.dataset_id,
            dataset.target_species,
        )
        if dataset.parametric is not None:
            selected = tuple(
                target
                for target in selected
                if target != dataset.parametric.x_name
            )
        return selected

    def _eligible_targets(self, dataset_id: str) -> tuple[str, ...]:
        dataset = self._projected_dataset(dataset_id)
        mapped = self._mapped_dataset(dataset_id)
        if dataset is None or mapped is None:
            return ()
        x_name = dataset.parametric.x_name if dataset.parametric is not None else None
        return tuple(
            dict.fromkeys(
                target
                for observation in mapped.observations
                for target in (self._observation_target(dataset, observation),)
                if target and target != x_name
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
        self._purge_parametric_x_pending(current)
        x_name = current.parametric.x_name if current.parametric is not None else None
        selected = tuple(
            target
            for target in self._selected_targets(current)
            if target != x_name
        )
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
        try:
            return replace(
                current,
                weight=self._pending_dataset_weights.get(
                    dataset_id,
                    current.weight,
                ),
                target_species=selected,
                target_weights=target_weights,
            )
        except (KeyError, TypeError, ValueError):
            return None

    def _refresh_footer_dirty(self) -> None:
        self._footer.set_dirty(
            self.has_pending_target_changes
            or self.has_pending_initial_changes
            or bool(self._invalid_raw)
            or bool(self._invalid_initial_raw)
        )

    def _apply_presentation_edits(self) -> None:
        """Overlay owned fields on latest rows and emit one atomic edit tuple."""

        if self._invalid_raw or self._invalid_initial_raw:
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
        setup = self._current_setup()
        current_initials = {
            (item.set_id, item.species): item
            for item in (setup.initials if setup is not None else ())
        }
        initial_intents = tuple(
            FitInitialSetupEdit(
                origin_tab_id=self._origin.tab_id,
                origin_capture_id=self._origin.capture_id,
                initial=replace(
                    current_initials[key],
                    initial=float(self._pending_initial_values[key]["initial"]),
                    minimum=float(self._pending_initial_values[key]["minimum"]),
                    maximum=float(self._pending_initial_values[key]["maximum"]),
                    fit=bool(self._pending_initial_values[key]["fit"]),
                    log10=bool(self._pending_initial_values[key]["log10"]),
                ),
            )
            for key, raw in self._pending_initial_rows.items()
            if key in current_initials and key in self._pending_initial_values
        )
        intents = (*dataset_intents, *initial_intents)
        if not intents:
            self.set_status("No pending Fit target, weight, or IC changes.")
            return
        self.setupEditsRequested.emit(intents)
        if self._acknowledge_applied_intents(intents):
            self.set_status("Fit target, weight, and IC edits applied.")
        self._refresh_visible_presentation()

    def _revert_presentation_edits(self) -> None:
        self._pending_target_selections.clear()
        self._pending_target_weights.clear()
        self._pending_dataset_weights.clear()
        self._pending_initial_rows.clear()
        self._pending_initial_values.clear()
        self._invalid_raw.clear()
        self._invalid_initial_raw.clear()
        self._refresh_visible_presentation()
        self.set_status("Pending Fit target, weight, and IC edits reverted.")

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
            if destination.dataset_id != source.dataset_id
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

    def _acknowledge_applied_intents(self, intents: tuple[object, ...]) -> bool:
        setup = self._current_setup()
        if setup is None:
            return False
        datasets = {item.dataset_id: item for item in setup.datasets}
        if any(
            isinstance(intent, FitDatasetSetupEdit)
            and datasets.get(intent.dataset.dataset_id) != intent.dataset
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
                key = (intent.initial.set_id, intent.initial.species)
                current = next(
                    (
                        item
                        for item in (self._current_setup().initials if self._current_setup() else ())
                        if (item.set_id, item.species) == key
                    ),
                    None,
                )
                if current == intent.initial:
                    self._pending_initial_rows.pop(key, None)
                    self._pending_initial_values.pop(key, None)
                    for invalid_key in tuple(self._invalid_initial_raw):
                        if invalid_key[:2] == key:
                            self._invalid_initial_raw.pop(invalid_key, None)
        return True

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
            "Correct invalid target, weight, or IC values before running."
            if self._invalid_raw or self._invalid_initial_raw
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
    def has_pending_initial_changes(self) -> bool:
        return bool(self._pending_initial_rows)

    def add_observable_conflict(self, dataset_id: str) -> str | None:
        """Return a same-dataset leaf draft that blocks Add Observable.

        Add Observable replaces a raw observation's accepted target identity.
        Target-selection and per-target-weight drafts for that same dataset
        must be settled first so no pending dictionary can retain a retired
        target key. Dataset-weight drafts and drafts for other datasets are
        independent and do not block this action.
        """

        dataset_id = str(dataset_id or "").strip()
        if not dataset_id:
            return None
        if dataset_id in self._pending_target_selections:
            return (
                "Apply or Revert the pending target selection for this dataset "
                "before adding an observable."
            )
        if any(
            pending_dataset_id == dataset_id
            for pending_dataset_id, _target in self._pending_target_weights
        ):
            return (
                "Apply or Revert the pending per-target weight edits for this "
                "dataset before adding an observable."
            )
        if any(
            len(key) >= 2
            and key[0] == "target_weight"
            and key[1] == dataset_id
            for key in self._invalid_raw
        ):
            return (
                "Apply or Revert the invalid per-target weight edits for this "
                "dataset before adding an observable."
            )
        return None

    @property
    def invalid_editor_keys(self) -> tuple[tuple[str, ...], ...]:
        return tuple(sorted((*self._invalid_raw, *self._invalid_initial_raw)))

    @property
    def invalid_raw_values(self) -> dict[tuple[str, ...], str]:
        return dict(self._invalid_raw)


__all__ = ["UnifiedSpeciesTable"]
