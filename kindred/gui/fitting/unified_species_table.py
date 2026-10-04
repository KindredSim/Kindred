"""Presentation for the historical unified Fit target/initial table.

The table is deliberately a view/editor surface.  Accepted Fit setup remains
owned by :class:`FitSessionOwner`; this widget only renders the owner supplied
snapshot and emits complete setup replacements for user edits.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Sequence

from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFontMetrics, QPalette

from kindred.gui.fitting.constants import FIT_DATASET_WEIGHT_MODE_LABELS
from kindred.gui.fitting.contracts import FitTargetSelection, copy_target_selection
from kindred.gui.fitting.launch import (
    FitDatasetLocalSettings,
    FitInitialVariable,
    FitMappedDatasetSnapshot,
    FitObservationSnapshot,
    FitOriginSnapshot,
    FitSharedInitialGroup,
    FitSetupSnapshot,
)
from kindred.gui.fitting.ic_tolerance import FitInitialTolerance


class _Col:
    """Data-and-Targets column indices."""

    INCLUDE = 0
    WEIGHT = 1
    SPECIES = 2
    INITIAL = 3
    TOLERANCE_MODE = 4
    VARY_GLOBAL = 5
    LOG10 = 6
    LOWER = 7
    UPPER = 8
    COUNT = 9
    HEADERS = [
        "Include",
        "Weight",
        "Species",
        "Starting IC",
        "Vary IC",
        "Vary Global",
        "Log scale",
        "Lower",
        "Upper",
    ]


_VARY_IC_CHOICES: tuple[tuple[str, str, str], ...] = (
    (
        "No",
        "disabled",
        "Keep this initial concentration fixed during fitting.",
    ),
    (
        "± X% (Symmetric)",
        "symmetric_percent",
        "Vary equally below and above the Starting IC. Example: 5 permits "
        "values from 95% to 105% of the Starting IC.",
    ),
    (
        "− X1% / + X2% (Separate)",
        "asymmetric_percent",
        "Enter separate decreases and increases relative to the Starting IC. "
        "Example: 10 and 5 permit values from 90% to 105% of the Starting IC.",
    ),
    (
        "Absolute Minimum / Maximum (in M)",
        "absolute",
        "Enter the actual minimum and maximum concentrations in molar units.",
    ),
)

_ZERO_IC_PERCENT_MESSAGE = (
    "A zero Starting IC cannot use a percentage Vary IC mode. Choose "
    "Absolute Minimum / Maximum (in M)."
)

_COPY_SPECIES_WEIGHTS_VARY_IC_TOOLTIP = (
    "Copies species weights, Vary IC mode and range, and log-scale choices "
    "from the selected dataset to compatible species in other datasets. "
    "Starting IC values and dataset-level settings are not copied."
)


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
    """Target/weight/initial presentation with complete-snapshot hooks."""

    # Every valid edit is submitted as one complete accepted-snapshot
    # replacement. Invalid numeric text remains in the table's local draft.
    setupSnapshotRequested = Signal(object)
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
        self._pending_shared_initial_groups: dict[
            tuple[str, str], FitSharedInitialGroup | None
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
        layout.addWidget(group, stretch=1)

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
        self._footer.hint_label.setText(
            "Apply and Revert affect pending targets, weights and initial conditions "
            "across all datasets. Data settings use their own Apply and Revert."
        )
        self._footer.applyRequested.connect(self._apply_presentation_edits)
        self._footer.revertRequested.connect(self._revert_presentation_edits)

        self._context_label = QtWidgets.QLabel("Targets and initial conditions", self)
        self._context_label.setStyleSheet("font-weight: bold;")
        self._context_label.setWordWrap(True)
        self._footer.body_layout.addWidget(self._context_label)

        weighting_form = QtWidgets.QFormLayout()
        weighting_form.setContentsMargins(0, 0, 0, 0)
        weighting_form.setRowWrapPolicy(QtWidgets.QFormLayout.RowWrapPolicy.WrapLongRows)
        self._weight_mode_combo = QtWidgets.QComboBox(self)
        self._weight_mode_combo.setObjectName("global_fit_weight_mode_combo")
        for mode, label in FIT_DATASET_WEIGHT_MODE_LABELS.items():
            self._weight_mode_combo.addItem(label, mode)
        weighting_form.addRow("Weight mode:", self._weight_mode_combo)

        self._dataset_weight_edit = QtWidgets.QLineEdit(self)
        self._dataset_weight_edit.setObjectName("global_fit_dataset_weight_edit")
        weighting_form.addRow("Dataset weight:", self._dataset_weight_edit)
        self._footer.body_layout.addLayout(weighting_form)
        self._weight_mode_combo.currentIndexChanged.connect(
            self._on_weight_mode_changed
        )
        self._dataset_weight_edit.textChanged.connect(
            self._on_dataset_weight_text_changed
        )

        bulk_row = QtWidgets.QGridLayout()
        self._bulk_layout = bulk_row
        self._bulk_all_button = QtWidgets.QPushButton("All", self)
        self._bulk_all_button.setObjectName("global_fit_fit_targets_bulk_all")
        self._bulk_all_button.clicked.connect(
            lambda: self._apply_bulk_action("all")
        )
        self._bulk_none_button = QtWidgets.QPushButton("None", self)
        self._bulk_none_button.setObjectName("global_fit_fit_targets_bulk_none")
        self._bulk_none_button.clicked.connect(
            lambda: self._apply_bulk_action("none")
        )
        self._bulk_invert_button = QtWidgets.QPushButton("Invert", self)
        self._bulk_invert_button.setObjectName("global_fit_fit_targets_bulk_invert")
        self._bulk_invert_button.clicked.connect(
            lambda: self._apply_bulk_action("invert")
        )
        self._copy_targets_button = QtWidgets.QPushButton(
            "Copy Targets to Compatible",
            self,
        )
        self._copy_targets_button.setObjectName(
            "global_fit_fit_targets_copy_compatible"
        )
        self._copy_targets_button.clicked.connect(
            self._copy_targets_to_compatible_datasets
        )
        self._copy_species_weights_vary_ic_button = QtWidgets.QPushButton(
            "Copy Species Weights and Vary IC to Compatible",
            self,
        )
        self._copy_species_weights_vary_ic_button.setObjectName(
            "global_fit_copy_species_weights_vary_ic_compatible"
        )
        self._copy_species_weights_vary_ic_button.setToolTip(
            _COPY_SPECIES_WEIGHTS_VARY_IC_TOOLTIP
        )
        self._copy_species_weights_vary_ic_button.clicked.connect(
            self._copy_species_weights_and_vary_ic_to_compatible
        )
        for button in (
            self._bulk_all_button,
            self._bulk_none_button,
            self._bulk_invert_button,
            self._copy_targets_button,
            self._copy_species_weights_vary_ic_button,
        ):
            button.setSizePolicy(QtWidgets.QSizePolicy.Policy.Maximum, QtWidgets.QSizePolicy.Policy.Fixed)
        self._relayout_bulk_buttons()
        self._footer.body_layout.addLayout(bulk_row)

        self._table = QtWidgets.QTableWidget(self)
        self._table.setObjectName("global_fit_unified_species_table")
        self._table.setItemDelegate(_ValidationCellDelegate(self._table))
        self._table.setColumnCount(_Col.COUNT)
        self._table.setHorizontalHeaderLabels(_Col.HEADERS)
        header = self._table.horizontalHeader()
        header.setStretchLastSection(False)
        # Keep useful initial widths while allowing larger fonts and user resizing.
        initial_widths = (72, 72, 140, 100, 220, 100, 82, 82, 82)
        metrics = header.fontMetrics()
        for column in range(_Col.COUNT):
            header.setSectionResizeMode(
                column,
                QtWidgets.QHeaderView.ResizeMode.Interactive,
            )
            self._table.setColumnWidth(
                column,
                max(
                    initial_widths[column],
                    metrics.horizontalAdvance(_Col.HEADERS[column]) + 24,
                ),
            )
        self._table.setMinimumHeight(
            header.sizeHint().height()
            + self._table.verticalHeader().defaultSectionSize()
            + self._table.horizontalScrollBar().sizeHint().height() + 4
        )
        self._table.verticalHeader().setVisible(False)
        self._table.setAlternatingRowColors(True)
        self._table.itemChanged.connect(self._on_table_item_changed)
        self._footer.body_layout.addWidget(self._table, stretch=1)

    def _relayout_bulk_buttons(self) -> None:
        buttons = (
            self._bulk_all_button,
            self._bulk_none_button,
            self._bulk_invert_button,
            self._copy_targets_button,
            self._copy_species_weights_vary_ic_button,
        )
        for button in buttons:
            self._bulk_layout.removeWidget(button)
        for index in range(6):
            self._bulk_layout.setColumnStretch(index, 0)
        metrics = self.fontMetrics()
        wide = self.width() >= (
            sum(metrics.horizontalAdvance(text) + 24 for text in (
                "All", "None", "Invert", "Copy Targets to Compatible",
                "Copy Species Weights and Vary IC to Compatible",
            )) + 80
        )
        self._copy_targets_button.setText(
            "Copy Targets to Compatible" if wide else "Copy Targets\nto Compatible"
        )
        self._copy_species_weights_vary_ic_button.setText(
            "Copy Species Weights and Vary IC to Compatible" if wide
            else "Copy Species Weights and Vary IC\nto Compatible"
        )
        if wide:
            for column, button in enumerate(buttons):
                self._bulk_layout.addWidget(button, 0, column)
            self._bulk_layout.setColumnStretch(5, 1)
        else:
            for column, button in enumerate(buttons[:4]):
                self._bulk_layout.addWidget(button, 0, column)
            self._bulk_layout.addWidget(buttons[4], 1, 0, 1, 4)
            self._bulk_layout.setColumnStretch(4, 1)

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().resizeEvent(event)
        self._relayout_bulk_buttons()

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

        self._origin = snapshot.origin
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

    def clear_pending_edits(self) -> None:
        """Discard every target, weight, IC, and raw editor draft."""

        self._pending_target_selections.clear()
        self._pending_target_weights.clear()
        self._pending_dataset_weights.clear()
        self._pending_initial_rows.clear()
        self._pending_initial_values.clear()
        self._pending_shared_initial_groups.clear()
        self._invalid_raw.clear()
        self._invalid_initial_raw.clear()
        self._refresh_visible_presentation()

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
            self._copy_species_weights_vary_ic_button,
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
        return str(mapping.get(observation.raw_observation_id, observation.model_target))

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
            "mode": initial.tolerance.mode,
            "lower": (
                ""
                if initial.tolerance.lower is None
                else UnifiedSpeciesTable._float_text(initial.tolerance.lower)
            ),
            "upper": (
                ""
                if initial.tolerance.upper is None
                else UnifiedSpeciesTable._float_text(initial.tolerance.upper)
            ),
            "log10": bool(initial.log10),
        }

    def _shared_initial_groups(self) -> tuple[FitSharedInitialGroup, ...]:
        """Return accepted groups overlaid with this editor's pending edits."""

        setup = self._current_setup()
        groups = {
            (item.species, item.origin_dataset_id): item
            for item in (setup.shared_initial_groups if setup is not None else ())
        }
        for identity, pending in self._pending_shared_initial_groups.items():
            if pending is None:
                groups.pop(identity, None)
            else:
                groups[identity] = pending
        return tuple(groups.values())

    def _shared_initial_group(
        self,
        dataset_id: str,
        species: str,
    ) -> FitSharedInitialGroup | None:
        return next(
            (
                item
                for item in self._shared_initial_groups()
                if item.species == species and dataset_id in item.member_dataset_ids
            ),
            None,
        )

    def _pending_initial_config(
        self,
        set_id: str,
        species: str,
    ) -> FitInitialVariable | None:
        """Return one valid typed pending local-IC draft, if present."""

        pending = self._pending_initial_values.get((set_id, species))
        if not isinstance(pending, dict):
            return None
        tolerance = pending.get("tolerance")
        if not isinstance(tolerance, FitInitialTolerance):
            return None
        try:
            return FitInitialVariable(
                set_id=set_id,
                species=species,
                initial=pending["initial"],
                tolerance=tolerance,
                log10=pending["log10"],
            )
        except (KeyError, TypeError, ValueError, OverflowError):
            return None

    def _refresh_vary_global_availability(self, row: int) -> None:
        """Refresh one Vary Global cell without rebuilding the table."""

        global_item = self._table.item(row, _Col.VARY_GLOBAL)
        initial_item = self._table.item(row, _Col.INITIAL)
        if global_item is None or initial_item is None:
            return
        dataset_id = str(initial_item.data(_DATASET_ROLE) or "").strip()
        set_id = str(initial_item.data(_SET_ROLE) or "").strip()
        species = str(initial_item.data(_SPECIES_ROLE) or "").strip()
        if not dataset_id or not set_id or not species:
            return
        group = self._shared_initial_group(dataset_id, species)
        if group is not None:
            checked = True
            enabled = True
        else:
            local = self._local_initial(
                self._projected_dataset(dataset_id),
                species,
            )
            if (set_id, species) in self._pending_initial_rows:
                pending = self._pending_initial_config(set_id, species)
                enabled = pending is not None and pending.fit
            else:
                enabled = local is not None and local.fit
            checked = False
        blocker = QtCore.QSignalBlocker(self._table)
        self._set_checkable(global_item, checked, enabled)
        del blocker

    def _compatible_group_members(
        self,
        species: str,
        origin_dataset_id: str,
    ) -> tuple[str, ...]:
        """List compatible datasets not occupied by another species group."""

        occupied = {
            member
            for group in self._shared_initial_groups()
            if group.species == species
            and group.origin_dataset_id != origin_dataset_id
            for member in group.member_dataset_ids
        }
        members: list[str] = []
        for dataset in self._current_setup().datasets if self._current_setup() else ():
            dataset_id = dataset.dataset_id
            if dataset_id == origin_dataset_id or dataset_id in occupied:
                continue
            if self._local_initial(dataset, species) is not None:
                members.append(dataset_id)
        return tuple(members)

    def _group_edit_dialog(
        self,
        group: FitSharedInitialGroup,
    ) -> tuple[str, ...] | None:
        """Show the small membership checklist for one shared IC group."""

        setup = self._current_setup()
        if setup is None:
            return None
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle(f"Vary Global — {group.species}")
        layout = QtWidgets.QVBoxLayout(dialog)
        layout.addWidget(QtWidgets.QLabel("Datasets sharing this initial concentration:"))
        checks: dict[str, QtWidgets.QCheckBox] = {}
        for dataset in setup.datasets:
            dataset_id = dataset.dataset_id
            local = self._local_initial(dataset, group.species)
            if local is None and dataset_id != group.origin_dataset_id:
                continue
            check = QtWidgets.QCheckBox(
                next(
                    (
                        mapped.display_name
                        for mapped in self._origin.mapped_datasets
                        if mapped.dataset_id == dataset_id
                    ),
                    dataset_id,
                ),
                dialog,
            )
            check.setChecked(dataset_id in group.member_dataset_ids)
            if dataset_id == group.origin_dataset_id:
                check.setChecked(True)
                check.setEnabled(False)
            elif any(
                dataset_id in other.member_dataset_ids
                for other in self._shared_initial_groups()
                if other.species == group.species
                and other.origin_dataset_id != group.origin_dataset_id
            ):
                check.setChecked(False)
                check.setEnabled(False)
            layout.addWidget(check)
            checks[dataset_id] = check
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
        members = tuple(
            dataset_id for dataset_id, check in checks.items() if check.isChecked()
        )
        if len(members) < 2:
            self.set_status("Vary Global needs an additional compatible dataset.")
            return ()
        return members

    def _retain_departing_initials(
        self, group: FitSharedInitialGroup, remaining: tuple[str, ...]
    ) -> None:
        for dataset_id in set(group.member_dataset_ids) - set(remaining):
            mapped = self._mapped_dataset(dataset_id)
            key = (mapped.set_id, group.species)
            pending = self._pending_initial_rows.get(key)
            # Only an explicit edit outranks the current group configuration.
            if pending is not None and not pending.get("_from_group"):
                continue
            self._pending_initial_rows[key] = {
                "_from_group": True,
                "initial": self._float_text(group.initial),
                "mode": group.tolerance.mode,
                "lower": "" if group.tolerance.lower is None else self._float_text(group.tolerance.lower),
                "upper": "" if group.tolerance.upper is None else self._float_text(group.tolerance.upper),
                "log10": group.log10,
            }
            self._pending_initial_values[key] = {
                "initial": group.initial, "tolerance": group.tolerance, "log10": group.log10,
            }

    def _stage_shared_initial_group(self, group: FitSharedInitialGroup) -> None:
        identity = (group.species, group.origin_dataset_id)
        previous = self._shared_initial_group(group.origin_dataset_id, group.species)
        if previous is not None:
            self._retain_departing_initials(
                replace(group, member_dataset_ids=previous.member_dataset_ids),
                group.member_dataset_ids,
            )
        setup = self._current_setup()
        accepted = None if setup is None else next(
            (
                item
                for item in setup.shared_initial_groups
                if (item.species, item.origin_dataset_id) == identity
            ),
            None,
        )
        if group == accepted:
            self._pending_shared_initial_groups.pop(identity, None)
        else:
            self._pending_shared_initial_groups[identity] = group
        self._refresh_visible_presentation()

    def _dissolve_shared_initial_group(self, group: FitSharedInitialGroup) -> None:
        self._retain_departing_initials(group, ())
        identity = (group.species, group.origin_dataset_id)
        setup = self._current_setup()
        if setup is not None and any(
            (item.species, item.origin_dataset_id) == identity
            for item in setup.shared_initial_groups
        ):
            self._pending_shared_initial_groups[identity] = None
        else:
            self._pending_shared_initial_groups.pop(identity, None)
        self._refresh_visible_presentation()

    def _on_vary_global_changed(self, row: int, item: QtWidgets.QTableWidgetItem) -> None:
        if self._refreshing or self._running:
            return
        dataset_id = str(item.data(_DATASET_ROLE) or "").strip()
        species = str(item.data(_SPECIES_ROLE) or "").strip()
        if not dataset_id or not species:
            return
        group = self._shared_initial_group(dataset_id, species)
        if group is not None:
            if (
                dataset_id == group.origin_dataset_id
                and item.checkState() != Qt.CheckState.Checked
            ):
                self._dissolve_shared_initial_group(group)
                return
            members = self._group_edit_dialog(group)
            if members is None:
                self._refresh_visible_presentation()
                return
            if not members:
                self._dissolve_shared_initial_group(group)
                return
            if len(members) < 2:
                self._dissolve_shared_initial_group(group)
                return
            self._stage_shared_initial_group(replace(group, member_dataset_ids=members))
            return
        if item.checkState() != Qt.CheckState.Checked:
            return
        current_dataset = self._current_dataset()
        local = self._local_initial(current_dataset, species) if current_dataset else None
        mapped = self._mapped_dataset(dataset_id)
        if mapped is None or local is None:
            self._refresh_visible_presentation()
            return
        key = (mapped.set_id, species)
        if key in self._pending_initial_rows:
            pending = self._pending_initial_config(mapped.set_id, species)
            if pending is None or not pending.fit:
                self._refresh_visible_presentation()
                return
            group_local = pending
        else:
            if not local.fit:
                self._refresh_visible_presentation()
                return
            group_local = local
        members = self._compatible_group_members(species, dataset_id)
        if not members:
            self.set_status("Vary Global needs an additional compatible dataset.")
            self._refresh_visible_presentation()
            return
        group = FitSharedInitialGroup(
            species=species,
            origin_dataset_id=dataset_id,
            member_dataset_ids=(dataset_id, members[0]),
            initial=group_local.initial,
            tolerance=group_local.tolerance,
            log10=group_local.log10,
        )
        selected = self._group_edit_dialog(group)
        if selected:
            if mapped is not None:
                self._pending_initial_rows.pop(key, None)
                self._pending_initial_values.pop(key, None)
            self._stage_shared_initial_group(replace(group, member_dataset_ids=selected))
        else:
            self._refresh_visible_presentation()

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
            self._context_label.setToolTip("")
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
        self._context_label.setToolTip(
            f"{mapped.display_name}\nDataset ID: {dataset.dataset_id}\nSet ID: {mapped.set_id}"
        )
        row_species = self._row_species(dataset)
        self._table.setRowCount(len(row_species))
        for row, species in enumerate(row_species):
            observations = self._observations_for_target(dataset, species)
            has_data = bool(observations)
            local = self._local_initial(dataset, species)
            group_config = self._shared_initial_group(dataset.dataset_id, species)
            group_origin = (
                group_config is not None
                and group_config.origin_dataset_id == dataset.dataset_id
            )
            local_key = (
                mapped.set_id if mapped is not None else "",
                species,
            )
            raw_initial = self._pending_initial_rows.get(
                local_key,
                self._initial_raw(local) if local is not None else {},
            )
            if group_config is not None:
                raw_initial = {
                    "initial": self._float_text(group_config.initial),
                    "mode": group_config.tolerance.mode,
                    "lower": (
                        ""
                        if group_config.tolerance.lower is None
                        else self._float_text(group_config.tolerance.lower)
                    ),
                    "upper": (
                        ""
                        if group_config.tolerance.upper is None
                        else self._float_text(group_config.tolerance.upper)
                    ),
                    "log10": group_config.log10,
                }
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
            weight_eligible = has_data and (species != x_name or x_name is not None)
            weight_item = QtWidgets.QTableWidgetItem(
                weight_text if weight_eligible else ""
            )
            weight_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._set_editable(weight_item, weight_eligible)
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
                (_Col.LOWER, "lower"),
                (_Col.UPPER, "upper"),
            ):
                invalid_key = (mapped.set_id, species, field)
                initial_item = QtWidgets.QTableWidgetItem(
                    self._invalid_initial_raw.get(
                        invalid_key,
                        str(raw_initial.get(field, "")),
                    )
                )
                tolerance_mode = str(raw_initial.get("mode", "disabled"))
                self._set_editable(
                    initial_item,
                    local is not None and (group_config is None or group_origin)
                    and (
                        field == "initial"
                        or (field == "lower" and tolerance_mode != "disabled")
                        or (
                            field == "upper"
                            and tolerance_mode in {"asymmetric_percent", "absolute"}
                        )
                    ),
                )
                initial_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                initial_item.setData(_DATASET_ROLE, dataset.dataset_id)
                initial_item.setData(_SET_ROLE, mapped.set_id)
                initial_item.setData(_SPECIES_ROLE, species)
                if local is not None:
                    initial_item.setData(
                        _EXACT_VALUE_ROLE,
                        (
                            local.initial
                            if field == "initial"
                            else getattr(local.tolerance, field)
                        ),
                    )
                if invalid_key in self._invalid_initial_raw:
                    initial_item.setBackground(_INVALID_BG)
                    initial_item.setForeground(_INVALID_FG)
                    initial_item.setToolTip(self._invalid_initial_raw[invalid_key])
                self._table.setItem(row, column, initial_item)

            mode = QtWidgets.QComboBox(self._table)
            mode.setObjectName("global_fit_vary_ic_combo")
            for label, value, tooltip in _VARY_IC_CHOICES:
                mode.addItem(label, value)
                item_index = mode.model().index(mode.count() - 1, 0)
                mode.model().setData(
                    item_index,
                    tooltip,
                    Qt.ItemDataRole.ToolTipRole,
                )
            mode.setEnabled(local is not None and (group_config is None or group_origin))
            mode.setCurrentIndex(
                max(0, mode.findData(str(raw_initial.get("mode", "disabled"))))
            )
            self._set_vary_ic_combo_tooltip(mode)
            mode.currentIndexChanged.connect(
                lambda _index, table_row=row, combo=mode:
                self._on_vary_ic_mode_changed(table_row, combo)
            )
            self._table.setCellWidget(row, _Col.TOLERANCE_MODE, mode)

            global_item = QtWidgets.QTableWidgetItem()
            self._set_checkable(global_item, False, False)
            global_item.setData(_DATASET_ROLE, dataset.dataset_id)
            global_item.setData(_SET_ROLE, mapped.set_id)
            global_item.setData(_SPECIES_ROLE, species)
            self._table.setItem(row, _Col.VARY_GLOBAL, global_item)

            check_item = QtWidgets.QTableWidgetItem()
            self._set_checkable(
                check_item,
                bool(raw_initial.get("log10", False)),
                local is not None
                and raw_initial.get("mode") != "disabled"
                and (group_config is None or group_origin),
            )
            check_item.setData(_DATASET_ROLE, dataset.dataset_id)
            check_item.setData(_SET_ROLE, mapped.set_id)
            check_item.setData(_SPECIES_ROLE, species)
            if local is not None:
                check_item.setData(_EXACT_VALUE_ROLE, local.log10)
            self._table.setItem(row, _Col.LOG10, check_item)
            self._refresh_vary_global_availability(row)

    @staticmethod
    def _set_vary_ic_combo_tooltip(combo: QtWidgets.QComboBox) -> None:
        """Keep the closed combo tooltip equal to its selected popup item."""

        index = combo.currentIndex()
        tooltip = "" if index < 0 else str(
            combo.itemData(index, Qt.ItemDataRole.ToolTipRole) or ""
        )
        combo.setToolTip(tooltip)

    def _on_vary_ic_mode_changed(
        self,
        row: int,
        combo: QtWidgets.QComboBox,
    ) -> None:
        """Author one complete fresh range for a Vary IC mode transition."""

        self._set_vary_ic_combo_tooltip(combo)
        if self._refreshing or self._running:
            return
        mode = str(combo.currentData() or "disabled")
        initial_item = self._table.item(row, _Col.INITIAL)
        if initial_item is None:
            return
        try:
            initial = float(initial_item.text())
            if not math.isfinite(initial) or initial < 0.0:
                raise ValueError
        except (TypeError, ValueError, OverflowError):
            # A mode transition still authors a fresh range when Starting IC
            # is incomplete.  The row validator retains the invalid Starting
            # IC and Run remains blocked until the user repairs it.
            initial = None

        if mode in {"symmetric_percent", "asymmetric_percent"} and initial == 0.0:
            # A percentage range around zero is undefined.  Reject the
            # transition itself so no invalid percentage draft can be applied.
            identity = self._table.item(row, _Col.INITIAL)
            previous_mode = "disabled"
            if identity is not None:
                set_id = str(identity.data(_SET_ROLE) or "").strip()
                species = str(identity.data(_SPECIES_ROLE) or "").strip()
                accepted = self._accepted_initial(set_id, species)
                previous_mode = (
                    str(self._pending_initial_rows.get((set_id, species), {}).get("mode"))
                    if (set_id, species) in self._pending_initial_rows
                    else (accepted.tolerance.mode if accepted is not None else "disabled")
                )
                if previous_mode not in {value for _label, value, _tip in _VARY_IC_CHOICES}:
                    previous_mode = "disabled"
            combo.setCurrentIndex(max(0, combo.findData(previous_mode)))
            self._set_vary_ic_combo_tooltip(combo)
            self._set_footer_error(_ZERO_IC_PERCENT_MESSAGE)
            self.statusMessage.emit(_ZERO_IC_PERCENT_MESSAGE)
            self.validityChanged.emit()
            return

        lower_item = self._table.item(row, _Col.LOWER)
        upper_item = self._table.item(row, _Col.UPPER)
        if lower_item is None or upper_item is None:
            return
        if mode == "disabled":
            lower, upper, disable_log = "", "", True
        elif mode == "symmetric_percent":
            lower, upper, disable_log = "5", "", False
        elif mode == "asymmetric_percent":
            lower, upper, disable_log = "5", "5", False
        else:
            lower = "0"
            upper = (
                self._float_text(10.0 * initial)
                if initial is not None and initial > 0.0
                else ""
            )
            disable_log = True

        # Signals from the two cleared cells must not create half-written
        # drafts. The single explicit stage below sees the complete row.
        table_blocker = QtCore.QSignalBlocker(self._table)
        lower_item.setText(lower)
        upper_item.setText(upper)
        if disable_log:
            log_item = self._table.item(row, _Col.LOG10)
            if log_item is not None:
                self._set_checkable(log_item, False, mode != "disabled")
        del table_blocker
        identity = self._table.item(row, _Col.INITIAL)
        if identity is not None:
            self._clear_initial_row_errors(
                str(identity.data(_SET_ROLE) or ""),
                str(identity.data(_SPECIES_ROLE) or ""),
                row,
            )
        self._stage_initial_row_at(row)

    def _accepted_initial(
        self,
        set_id: str,
        species: str,
    ) -> FitInitialVariable | None:
        setup = self._current_setup()
        return next(
            (
                item
                for item in (setup.initials if setup is not None else ())
                if item.set_id == set_id and item.species == species
            ),
            None,
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
        """Drop Y-selection drafts that are invalid for this dataset's current X."""

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
            _Col.LOG10,
            _Col.LOWER,
            _Col.UPPER,
        }:
            self._stage_initial_row(item)
            return
        if item.column() == _Col.VARY_GLOBAL:
            self._on_vary_global_changed(item.row(), item)
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

        self._stage_initial_row_at(item.row())

    def _stage_initial_row_at(self, row: int) -> None:
        """Reconstruct one row after either item or tolerance-mode edits."""

        identity_item = self._table.item(row, _Col.INITIAL)
        if identity_item is None:
            return
        set_id = str(identity_item.data(_SET_ROLE) or "").strip()
        species = str(identity_item.data(_SPECIES_ROLE) or "").strip()
        dataset_id = str(identity_item.data(_DATASET_ROLE) or "").strip()
        if not set_id or not species:
            return
        items = {
            "initial": self._table.item(row, _Col.INITIAL),
            "lower": self._table.item(row, _Col.LOWER),
            "upper": self._table.item(row, _Col.UPPER),
        }
        mode_widget = self._table.cellWidget(row, _Col.TOLERANCE_MODE)
        log_item = self._table.item(row, _Col.LOG10)
        if (
            any(value is None for value in items.values())
            or not isinstance(mode_widget, QtWidgets.QComboBox)
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
        group = self._shared_initial_group(dataset_id, species)
        group_origin = group is not None and group.origin_dataset_id == dataset_id
        if group is not None and not group_origin:
            return
        mode = str(mode_widget.currentData() or "disabled")
        log10 = log_item.checkState() == Qt.CheckState.Checked
        blocker = QtCore.QSignalBlocker(self._table)
        self._set_checkable(
            log_item,
            log10 if mode != "disabled" else False,
            mode != "disabled",
        )
        for field, field_item in items.items():
            assert field_item is not None
            self._set_editable(
                field_item,
                field == "initial"
                or (field == "lower" and mode != "disabled")
                or (
                    field == "upper"
                    and mode in {"asymmetric_percent", "absolute"}
                ),
            )
        del blocker
        if mode == "disabled":
            log10 = False
        raw = {
            field: str(value.text())
            for field, value in items.items()
            if value is not None
        }
        raw.update({"mode": mode, "log10": log10})
        if not group_origin:
            self._pending_initial_rows[key] = raw

        typed: dict[str, object] = {"mode": mode, "log10": log10}
        parse_errors: dict[str, str] = {}
        required_fields = {"initial"}
        if mode == "symmetric_percent":
            required_fields.add("lower")
        elif mode in {"asymmetric_percent", "absolute"}:
            required_fields.update({"lower", "upper"})
        for field, field_item in items.items():
            assert field_item is not None
            if field not in required_fields:
                continue
            try:
                parsed = float(field_item.text())
                if not math.isfinite(parsed):
                    raise ValueError("value must be finite")
            except (TypeError, ValueError, OverflowError) as exc:
                parse_errors[field] = str(exc)
            else:
                typed[field] = parsed
        self._clear_initial_row_errors(set_id, species, row)
        if parse_errors:
            self._pending_initial_values.pop(key, None)
            for field, message in parse_errors.items():
                field_key = (set_id, species, field)
                self._invalid_initial_raw[field_key] = str(raw[field])
                self._decorate_initial_field(
                    row,
                    {
                        "initial": _Col.INITIAL,
                        "lower": _Col.LOWER,
                        "upper": _Col.UPPER,
                    }[field],
                    message,
                )
            self._refresh_vary_global_availability(row)
            self._refresh_footer_dirty()
            self._refresh_footer_validity()
            return
        try:
            if mode == "disabled":
                tolerance = FitInitialTolerance()
            elif mode == "symmetric_percent":
                tolerance = FitInitialTolerance.symmetric_percent(typed["lower"])
            elif mode == "asymmetric_percent":
                tolerance = FitInitialTolerance.asymmetric_percent(
                    typed["lower"], typed["upper"]
                )
            else:
                tolerance = FitInitialTolerance.absolute(
                    typed["lower"], typed["upper"]
                )
            initial = FitInitialVariable(
                set_id=set_id,
                species=species,
                initial=float(typed["initial"]),
                tolerance=tolerance,
                log10=bool(typed["log10"]),
            )
        except (TypeError, ValueError, OverflowError) as caught_error:
            self._pending_initial_values.pop(key, None)
            error_message = str(caught_error)
            columns = {
                "initial": _Col.INITIAL,
                "lower": _Col.LOWER,
                "upper": _Col.UPPER,
            }
            if (
                mode in {"symmetric_percent", "asymmetric_percent"}
                and typed.get("initial") == 0.0
            ):
                error_message = _ZERO_IC_PERCENT_MESSAGE
                self._set_footer_error(_ZERO_IC_PERCENT_MESSAGE)
                self.statusMessage.emit(_ZERO_IC_PERCENT_MESSAGE)
            for field in required_fields:
                self._invalid_initial_raw[(set_id, species, field)] = str(raw[field])
                self._decorate_initial_field(row, columns[field], error_message)
        else:
            self._clear_initial_row_errors(set_id, species, row)
            if group_origin and group is not None:
                if initial.tolerance.mode == "disabled":
                    self._refresh_vary_global_availability(row)
                    self._pending_initial_rows[key] = raw
                    self._pending_initial_values[key] = {
                        "initial": initial.initial, "tolerance": initial.tolerance,
                        "log10": initial.log10,
                    }
                    self._dissolve_shared_initial_group(group)
                    return
                self._stage_shared_initial_group(
                    replace(
                        group,
                        initial=initial.initial,
                        tolerance=initial.tolerance,
                        log10=initial.log10,
                    )
                )
            else:
                self._pending_initial_values[key] = {
                    "initial": initial.initial,
                    "tolerance": initial.tolerance,
                    "log10": initial.log10,
                }
                if accepted is not None and initial == accepted:
                    self._pending_initial_rows.pop(key, None)
                    self._pending_initial_values.pop(key, None)
        self._refresh_vary_global_availability(row)
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
            ("lower", _Col.LOWER),
            ("upper", _Col.UPPER),
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
        # A weight has meaning only for a selected target.  Drop drafts for
        # rows just deselected so this surface never retains dormant weights.
        x_name = (
            projected.parametric.x_name
            if projected.parametric is not None
            else None
        )
        for key in tuple(self._pending_target_weights):
            if key[0] == dataset_id and key[1] not in selected and key[1] != x_name:
                self._pending_target_weights.pop(key, None)
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
        weight_targets = (
            (*selected, x_name)
            if x_name is not None
            else selected
        )
        invalid_selected = [
            target
            for target in weight_targets
            if ("target_weight", dataset_id, target) in self._invalid_raw
        ]
        if invalid_selected or ("dataset_weight", dataset_id) in self._invalid_raw:
            return None
        current_weights = dict(current.target_weights)
        weight_roster = (
            (*selected, current.parametric.x_name)
            if current.parametric is not None
            else selected
        )
        target_weights = tuple(
            (
                target,
                self._pending_target_weights.get(
                    (dataset_id, target),
                    current_weights.get(target, 1.0),
                ),
            )
            for target in weight_roster
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
        """Overlay owned fields and emit one complete snapshot replacement."""

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
        setup = self._current_setup()
        current_initials = {
            (item.set_id, item.species): item
            for item in (setup.initials if setup is not None else ())
        }
        updated_initials = {
            key: replace(
                current_initials[key],
                initial=float(self._pending_initial_values[key]["initial"]),
                tolerance=self._pending_initial_values[key]["tolerance"],
                log10=bool(self._pending_initial_values[key]["log10"]),
            )
            for key in self._pending_initial_rows
            if key in current_initials and key in self._pending_initial_values
        }
        updated_groups = {
            identity: group
            for identity, group in self._pending_shared_initial_groups.items()
        }
        if not overlays and not updated_initials and not updated_groups:
            self.set_status("No pending Fit target, weight, IC, or global-group changes.")
            return
        groups = {
            (item.species, item.origin_dataset_id): item
            for item in setup.shared_initial_groups
        }
        for identity, group in updated_groups.items():
            if group is None:
                groups.pop(identity, None)
            else:
                groups[identity] = group
        replacement = replace(
            setup,
            datasets=tuple(
                next(
                    (overlay for overlay in overlays if overlay.dataset_id == item.dataset_id),
                    item,
                )
                for item in setup.datasets
            ),
            initials=tuple(
                updated_initials.get((item.set_id, item.species), item)
                for item in setup.initials
            ),
            shared_initial_groups=tuple(groups.values()),
        )
        self._pending_target_selections.clear()
        self._pending_target_weights.clear()
        self._pending_dataset_weights.clear()
        self._pending_initial_rows.clear()
        self._pending_initial_values.clear()
        self._pending_shared_initial_groups.clear()
        self.setupSnapshotRequested.emit(replacement)
        self.set_status("Fit target, weight, IC, and global-group edits submitted.")
        self._refresh_visible_presentation()

    def _revert_presentation_edits(self) -> None:
        self.clear_pending_edits()
        self.set_status("Pending Fit target, weight, IC, and global-group edits reverted.")

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

    def _copy_species_weights_and_vary_ic_to_compatible(self) -> None:
        """Copy only row-local Vary IC/weight projections across datasets.

        Destination target selection, Starting IC values, dataset settings,
        and accepted owner state remain untouched.  A target weight is
        projected only when that destination already selects the species.
        """

        setup = self._current_setup()
        source = self._current_dataset()
        if setup is None or source is None:
            return

        source_mapped = self._mapped_dataset(source.dataset_id)
        if source_mapped is None:
            return
        source_weights = dict(source.target_weights)
        source_selected = set(self._selected_targets(source))
        copied_ranges = 0
        copied_weights = 0
        skipped = 0
        source_initials = {
            item.species: item
            for item in setup.initials
            if item.set_id == source_mapped.set_id
        }
        for destination in setup.datasets:
            if destination.dataset_id == source.dataset_id:
                continue
            destination_mapped = self._mapped_dataset(destination.dataset_id)
            if destination_mapped is None:
                continue
            destination_initials = {
                item.species: item
                for item in setup.initials
                if item.set_id == destination_mapped.set_id
            }
            destination_selected = set(self._selected_targets(destination))
            for species, source_initial in source_initials.items():
                destination_initial = destination_initials.get(species)
                if destination_initial is None:
                    skipped += 1
                    continue
                destination_grouped = self._shared_initial_group(
                    destination.dataset_id, species
                ) is not None
                if destination_grouped:
                    # Group membership owns the effective IC configuration.
                    # Leave the retained local untouched while still copying
                    # the independent target weight below.
                    skipped += 1
                else:
                    source_key = (source_mapped.set_id, species)
                    pending_source = self._pending_initial_values.get(source_key)
                    source_group = self._shared_initial_group(source.dataset_id, species)
                    if source_group is not None:
                        source_tolerance = source_group.tolerance
                        source_log10 = source_group.log10
                    elif pending_source is None:
                        source_tolerance = source_initial.tolerance
                        source_log10 = source_initial.log10
                    else:
                        source_tolerance = pending_source["tolerance"]
                        source_log10 = bool(pending_source["log10"])
                    if not isinstance(source_tolerance, FitInitialTolerance):
                        skipped += 1
                        continue
                    destination_key = (destination_mapped.set_id, species)
                    destination_raw = self._initial_raw(destination_initial)
                    pending_destination = self._pending_initial_rows.get(destination_key)
                    if pending_destination is not None:
                        destination_raw["initial"] = pending_destination["initial"]
                    destination_raw.update(
                        {
                            "mode": source_tolerance.mode,
                            "lower": (
                                ""
                                if source_tolerance.lower is None
                                else self._float_text(source_tolerance.lower)
                            ),
                            "upper": (
                                ""
                                if source_tolerance.upper is None
                                else self._float_text(source_tolerance.upper)
                            ),
                            "log10": bool(source_log10),
                        }
                    )
                    try:
                        copied_initial = FitInitialVariable(
                            set_id=destination_mapped.set_id,
                            species=species,
                            initial=float(destination_raw["initial"]),
                            tolerance=source_tolerance,
                            log10=source_log10,
                        )
                    except (TypeError, ValueError, OverflowError):
                        # Preserve the copied range as an invalid draft when the
                        # destination Starting IC is outside it (or zero for a
                        # percentage mode).  The destination value remains
                        # authoritative and Apply stays blocked until the user
                        # repairs the row.
                        destination_key = (destination_mapped.set_id, species)
                        self._pending_initial_rows[destination_key] = destination_raw
                        self._pending_initial_values.pop(destination_key, None)
                        for field in ("lower", "upper"):
                            self._invalid_initial_raw[
                                (destination_mapped.set_id, species, field)
                            ] = str(destination_raw[field])
                        skipped += 1
                    else:
                        for field in ("initial", "lower", "upper"):
                            self._invalid_initial_raw.pop(
                                (destination_mapped.set_id, species, field),
                                None,
                            )
                        if copied_initial == destination_initial:
                            self._pending_initial_rows.pop(destination_key, None)
                            self._pending_initial_values.pop(destination_key, None)
                        else:
                            self._pending_initial_rows[destination_key] = destination_raw
                            self._pending_initial_values[destination_key] = {
                                "initial": copied_initial.initial,
                                "tolerance": copied_initial.tolerance,
                                "log10": copied_initial.log10,
                            }
                        copied_ranges += 1

                # Weight is copied independently of the Vary IC projection,
                # and only for an already selected destination target.
                weight_key = (destination.dataset_id, species)
                if species in destination_selected and species in source_selected:
                    source_weight = self._pending_target_weights.get(
                        (source.dataset_id, species),
                        source_weights.get(species, 1.0),
                    )
                    destination_weight = dict(destination.target_weights).get(
                        species,
                        1.0,
                    )
                    if source_weight == destination_weight:
                        self._pending_target_weights.pop(weight_key, None)
                    else:
                        self._pending_target_weights[weight_key] = source_weight
                    copied_weights += 1
                else:
                    self._pending_target_weights.pop(weight_key, None)

        self._refresh_visible_presentation()
        self.set_status(
            "Copy Species Weights and Vary IC: "
            f"{copied_ranges} ranges, {copied_weights} weights copied, "
            f"{skipped} skipped."
        )

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
        self.setupSnapshotRequested.emit(
            replace(
                self._current_setup(),
                dataset_weight_mode=mode,
            )
        )
        self._refresh_dataset_weight_editor()

    def _refresh_dataset_weight_editor(self) -> None:
        dataset = self._current_dataset()
        if dataset is None:
            self._context_label.setToolTip("")
            self._dataset_weight_edit.clear()
            self._weight_mode_combo.setEnabled(False)
            self._dataset_weight_edit.setEnabled(False)
            return
        mapped = self._mapped_dataset(dataset.dataset_id)
        label = mapped.display_name if mapped is not None else dataset.dataset_id
        tooltip = f"{label}\nDataset ID: {dataset.dataset_id}"
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
        return bool(self._pending_initial_rows or self._pending_shared_initial_groups)

    @property
    def invalid_editor_keys(self) -> tuple[tuple[str, ...], ...]:
        return tuple(sorted((*self._invalid_raw, *self._invalid_initial_raw)))

    @property
    def invalid_raw_values(self) -> dict[tuple[str, ...], str]:
        return dict(self._invalid_raw)


__all__ = ["UnifiedSpeciesTable"]
