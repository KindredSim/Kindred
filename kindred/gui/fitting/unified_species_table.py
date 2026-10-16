"""Presentation for the Fit target, weight and initial-condition table.

The table is deliberately a view/editor surface.  Accepted Fit setup remains
owned by :class:`FitSessionOwner`; this widget only renders the owner supplied
snapshot and emits complete setup replacements for user edits.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Sequence

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFontMetrics, QKeySequence, QPalette

from kindred.core.validation import parse_finite_numeric_text, six_digit_text, six_digit_value
from kindred.gui.fitting.constants import FIT_DATASET_WEIGHT_MODE_LABELS
from kindred.gui.ui_helpers import (
    ChoicesComboBox,
    configure_compact_numeric_edit,
    copy_current_cell_as_shown,
    ignore_wheel,
)
from kindred.gui.fitting.contracts import FitTargetSelection, copy_target_selection
from kindred.gui.fitting.launch import (
    FitDatasetLocalSettings,
    FitInitialVariable,
    FitMappedDatasetSnapshot,
    FitObservationSnapshot,
    FitOriginSnapshot,
    FitSharedInitialGroup,
    FitSetupSnapshot,
    shared_initial_group_labels,
)
from kindred.gui.fitting.ic_tolerance import ZERO_IC_PERCENT_MESSAGE, FitInitialTolerance


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
        "Log Scale",
        "Lower",
        "Upper",
    ]


_INITIAL_FIELD_HEADERS = {
    "initial": _Col.HEADERS[_Col.INITIAL],
    "lower": _Col.HEADERS[_Col.LOWER],
    "upper": _Col.HEADERS[_Col.UPPER],
}

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
        "Minimum / Maximum (in M)",
        "absolute",
        "Enter the actual minimum and maximum concentrations in molar units.",
    ),
)

_TARGET_WEIGHT_MESSAGE = "Target weights must be finite and positive."

_COPY_SPECIES_WEIGHTS_VARY_IC_TOOLTIP = (
    "Copies species weights, Vary IC mode and range, and log-scale choices "
    "from the selected dataset to compatible species in other datasets. "
    "Starting IC values and dataset-level settings are not copied."
)


_INVALID_BG = QBrush(QColor(255, 200, 200))
_INVALID_FG = QBrush(QColor(80, 0, 0))

_DATASET_ROLE = Qt.ItemDataRole.UserRole
_SET_ROLE = Qt.ItemDataRole.UserRole + 1
_MODEL_TARGET_ROLE = Qt.ItemDataRole.UserRole + 4
_SPECIES_ROLE = Qt.ItemDataRole.UserRole + 5


_NUMBER_COLUMNS = (_Col.WEIGHT, _Col.INITIAL, _Col.LOWER, _Col.UPPER)


class _ValidationCellDelegate(QtWidgets.QStyledItemDelegate):
    """Paint cell text using ForegroundRole, bypassing stylesheet overrides.

    Valid numbers paint at six significant digits; invalid cells show their
    text as typed, and an open editor keeps the full text.
    """

    def initStyleOption(self, option, index) -> None:  # noqa: N802 - Qt override
        super().initStyleOption(option, index)
        foreground = index.data(Qt.ItemDataRole.ForegroundRole)
        if index.column() not in _NUMBER_COLUMNS or (
            isinstance(foreground, QBrush) and foreground.color() == _INVALID_FG.color()
        ):
            return
        try:
            option.text = six_digit_text(float(option.text))
        except ValueError:
            pass

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


class _SpeciesTableWidget(QtWidgets.QTableWidget):
    """Copy carries a cell exactly as it is shown."""

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt override
        if event.matches(QKeySequence.StandardKey.Copy):
            copy_current_cell_as_shown(self)
            event.accept()
            return
        super().keyPressEvent(event)


class UnifiedSpeciesTable(QtWidgets.QWidget):
    """Target/weight/initial presentation with complete-snapshot hooks.

    Every valid edit is submitted at once as one complete setup replacement.
    An edit that cannot apply keeps its text, blocks Run, and stays until
    corrected or discarded with Escape.
    """

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
        # Weight text that could not apply, keyed ("target_weight", dataset,
        # species) or ("dataset_weight", dataset).
        self._invalid_raw: dict[tuple[str, ...], str] = {}
        # The complete text of each initial-condition row that could not
        # apply, and each of its invalid cells' text and message.
        self._invalid_initial_rows: dict[tuple[str, str], dict[str, object]] = {}
        self._invalid_initial_raw: dict[tuple[str, str, str], tuple[str, str]] = {}
        self._row_keys: tuple[object, ...] = ()
        self._refreshing = False
        self._running = False

        self._build_ui()

    # ------------------------------------------------------------------
    # Physical construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        group = QtWidgets.QGroupBox("Species Table", self)
        group.setObjectName("global_fit_unified_species_group")
        self._group = group
        group_layout = QtWidgets.QVBoxLayout(group)
        layout.addWidget(group, stretch=1)

        # The composing tab places this one "Dataset weights" setting among
        # the Fit Data settings; this table still owns its edits.  The mode
        # and this dataset's weight always share a line.
        weighting = QtWidgets.QWidget(self)
        weighting_row = QtWidgets.QHBoxLayout(weighting)
        weighting_row.setContentsMargins(0, 0, 0, 0)
        self._weight_mode_combo = ChoicesComboBox(weighting)
        self._weight_mode_combo.setObjectName("global_fit_weight_mode_combo")
        self._weight_mode_combo.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self._weight_mode_combo.setMinimumContentsLength(12)
        self._weight_mode_combo.setToolTip("Applies to all datasets.")
        for mode, label in FIT_DATASET_WEIGHT_MODE_LABELS.items():
            self._weight_mode_combo.addItem(label, mode)
        self._dataset_weight_edit = QtWidgets.QLineEdit(weighting)
        self._dataset_weight_edit.setObjectName("global_fit_dataset_weight_edit")
        configure_compact_numeric_edit(self._dataset_weight_edit, characters=8)
        weighting_row.addWidget(self._weight_mode_combo)
        weighting_row.addSpacing(12)
        weighting_row.addWidget(QtWidgets.QLabel("This dataset", weighting))
        weighting_row.addWidget(self._dataset_weight_edit)
        self.weighting_settings = (("Dataset weights", weighting),)
        self._weight_mode_combo.currentIndexChanged.connect(
            self._on_weight_mode_changed
        )
        self._dataset_weight_edit.editingFinished.connect(
            self._on_dataset_weight_edited
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
        group_layout.addLayout(bulk_row)

        self._table = _SpeciesTableWidget(self)
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
        group_layout.addWidget(self._table, stretch=1)

        self._error_label = QtWidgets.QLabel(self)
        self._error_label.setObjectName("global_fit_species_table_error")
        self._error_label.setWordWrap(True)
        self._error_label.setVisible(False)
        group_layout.addWidget(self._error_label)

        self._discard_shortcuts = tuple(
            QtGui.QShortcut(QKeySequence(Qt.Key.Key_Escape), parent)
            for parent in (self, self._dataset_weight_edit)
        )
        for shortcut in self._discard_shortcuts:
            shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            shortcut.setEnabled(False)
            shortcut.activated.connect(self._discard_current_invalid_entries)

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
        requested_id = dataset_id or self._current_dataset_id
        self._projected_setup = snapshot
        self._current_dataset_id = requested_id or (
            available_ids[0] if available_ids else None
        )
        self._refresh_visible_presentation()

    def load_for_dataset(self, dataset_id: str) -> None:
        """Select one projected dataset and repaint the table."""

        if self._projected_setup is None:
            return
        self._current_dataset_id = dataset_id
        self._refresh_visible_presentation()

    def refresh_validity_ui(self) -> None:
        """Refresh presentation validation without becoming a Fit validator."""

        self._refresh_error()

    def clear_pending_edits(self) -> None:
        """Discard every held invalid weight and initial-condition entry."""

        self._invalid_raw.clear()
        self._invalid_initial_rows.clear()
        self._invalid_initial_raw.clear()
        self._refresh_visible_presentation()

    def set_running_state(self, running: bool) -> None:
        """Lock every editor while a Fit runs."""

        self._running = bool(running)
        self._table.setEnabled(not self._running)
        self._weight_mode_combo.setEnabled(not self._running)
        if self._running:
            self._dataset_weight_edit.setEnabled(False)
        else:
            self._refresh_dataset_weight_editor()
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

    def _submit(self, replacement: FitSetupSnapshot) -> None:
        """Submit one complete setup; the owner projects the accepted one back."""

        self.setupSnapshotRequested.emit(replacement)
        # A refused replacement leaves the previous projection to restore.
        self._refresh_visible_presentation()

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
        # Cells keep the exact accepted binary64 value as their item text;
        # tolerance edits re-parse it.  The delegate paints six digits.
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
    def _initial_raw(
        initial: FitInitialVariable | FitSharedInitialGroup,
    ) -> dict[str, object]:
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

    def _shared_initial_group(
        self,
        dataset_id: str,
        species: str,
    ) -> FitSharedInitialGroup | None:
        setup = self._current_setup()
        return next(
            (
                item
                for item in (setup.shared_initial_groups if setup is not None else ())
                if item.species == species and dataset_id in item.member_dataset_ids
            ),
            None,
        )

    def _dataset_name(self, dataset_id: str) -> str:
        mapped = self._mapped_dataset(dataset_id)
        return mapped.display_name if mapped is not None else dataset_id

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
            # Only an applied varied row can start or join a group.
            enabled = (
                (set_id, species) not in self._invalid_initial_rows
                and local is not None
                and local.fit
            )
            checked = False
        blocker = QtCore.QSignalBlocker(self._table)
        self._set_checkable(global_item, checked, enabled)
        del blocker

    def _group_edit_dialog(
        self,
        species: str,
        dataset_id: str,
    ) -> tuple[str | None, tuple[str, ...]] | None:
        """Ask which shared IC group the dataset joins, or which datasets form a new one.

        Returns the joined group's origin (None for a new group) and the other
        datasets added, or None on Cancel.  The dialog only adds: existing
        members stay, and datasets in other groups of the species are greyed.
        Groups are numbered by the table position of their origin dataset, and
        datasets excluded from the fit are marked.  Each dataset shows its own
        Starting IC.
        """

        setup = self._current_setup()
        if setup is None:
            return None
        included = {item.dataset_id for item in setup.datasets if item.included}
        groups = tuple(
            group for group in setup.shared_initial_groups if group.species == species
        )
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle(f"Vary Global — {species}")
        layout = QtWidgets.QVBoxLayout(dialog)
        layout.addWidget(QtWidgets.QLabel("Datasets sharing this initial concentration:"))
        target = QtWidgets.QComboBox(dialog)
        target.addItem("New Group", None)
        for origin_dataset_id, (_number, label) in shared_initial_group_labels(
            groups, species, setup.datasets, self._origin,
        ).items():
            target.addItem(label, origin_dataset_id)
        layout.addWidget(target)
        checks: dict[str, QtWidgets.QCheckBox] = {}
        for dataset in setup.datasets:
            local = self._local_initial(dataset, species)
            if dataset.dataset_id != dataset_id and local is None:
                continue
            name = self._dataset_name(dataset.dataset_id)
            if local is not None:
                name = f"{name} ({six_digit_text(local.initial)} M)"
            check = QtWidgets.QCheckBox(
                name if dataset.dataset_id in included else f"{name} (not in fit)", dialog,
            )
            layout.addWidget(check)
            checks[dataset.dataset_id] = check

        def project() -> None:
            """Tick and lock the clicked dataset and the chosen group's members."""

            joined = target.currentData()
            members = next(
                (group.member_dataset_ids for group in groups if group.origin_dataset_id == joined), (),
            )
            occupied = {member for group in groups for member in group.member_dataset_ids}
            for item, check in checks.items():
                locked = item == dataset_id or item in members
                free = not locked and item not in occupied
                check.setChecked(locked or (free and check.isChecked()))
                check.setEnabled(free)

        target.currentIndexChanged.connect(project)
        project()
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
        return target.currentData(), tuple(
            item for item, check in checks.items() if check.isChecked() and check.isEnabled()
        )

    def _submit_groups(
        self,
        groups: Sequence[FitSharedInitialGroup],
        *,
        initials: Sequence[FitInitialVariable] | None = None,
    ) -> None:
        setup = self._current_setup()
        self._submit(replace(
            setup,
            shared_initial_groups=tuple(groups),
            initials=setup.initials if initials is None else tuple(initials),
        ))

    def _leave_shared_initial_group(self, group: FitSharedInitialGroup, dataset_id: str) -> None:
        """One dataset leaves; a group left with one member dissolves.

        When the origin leaves, the first remaining member in table order
        becomes the origin.  Departing datasets keep their own Starting IC
        and Vary IC, which membership never changed.
        """

        setup = self._current_setup()
        order = tuple(item.dataset_id for item in setup.datasets)
        remaining = tuple(sorted(
            (item for item in group.member_dataset_ids if item != dataset_id), key=order.index,
        ))
        others = [item for item in setup.shared_initial_groups if item != group]
        if len(remaining) >= 2:
            origin = group.origin_dataset_id if group.origin_dataset_id in remaining else remaining[0]
            others.append(replace(group, origin_dataset_id=origin, member_dataset_ids=remaining))
        self._submit_groups(others)

    def _on_vary_global_changed(self, row: int, item: QtWidgets.QTableWidgetItem) -> None:
        """Untick leaves the dataset's group; tick opens the membership dialog."""

        if self._refreshing or self._running:
            return
        dataset_id = str(item.data(_DATASET_ROLE) or "").strip()
        species = str(item.data(_SPECIES_ROLE) or "").strip()
        if not dataset_id or not species:
            return
        group = self._shared_initial_group(dataset_id, species)
        if group is not None:
            if item.checkState() != Qt.CheckState.Checked:
                self._leave_shared_initial_group(group, dataset_id)
            return
        if item.checkState() != Qt.CheckState.Checked:
            return
        current_dataset = self._current_dataset()
        local = self._local_initial(current_dataset, species) if current_dataset else None
        mapped = self._mapped_dataset(dataset_id)
        if (
            mapped is None
            or local is None
            or not local.fit
            or (mapped.set_id, species) in self._invalid_initial_rows
        ):
            self._refresh_visible_presentation()
            return
        choice = self._group_edit_dialog(species, dataset_id)
        if choice is None:
            self._refresh_visible_presentation()
            return
        joined, added = choice
        if joined is None and not added:
            self.set_status("Vary Global needs an additional compatible dataset.")
            self._refresh_visible_presentation()
            return
        # Every joiner takes the group's configuration over its invalid text.
        for member in (dataset_id, *added):
            self._drop_invalid_initial_row(self._mapped_dataset(member).set_id, species)
        setup = self._current_setup()
        if joined is None:
            self._submit_groups((
                *setup.shared_initial_groups,
                FitSharedInitialGroup(
                    species=species,
                    origin_dataset_id=dataset_id,
                    member_dataset_ids=(dataset_id, *added),
                    initial=local.initial,
                    tolerance=local.tolerance,
                    log10=local.log10,
                ),
            ))
            return
        self._submit_groups(tuple(
            replace(group, member_dataset_ids=(*group.member_dataset_ids, dataset_id, *added))
            if group.species == species and group.origin_dataset_id == joined
            else group
            for group in setup.shared_initial_groups
        ))

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
        return "\n".join(
            f"Data column {item.raw_column} → {self._observation_target(dataset, item)}"
            for item in self._observations_for_target(dataset, target)
        )

    def _populate_table(self) -> None:
        dataset = self._current_dataset()
        if dataset is None:
            self._table.setRowCount(0)
            self._row_keys = ()
            self._group.setTitle("Species Table")
            return

        mapped = self._mapped_dataset(dataset.dataset_id)
        if mapped is None:
            return
        self._group.setTitle(f"Species Table — {mapped.display_name}")
        x_name = dataset.parametric.x_name if dataset.parametric is not None else None
        selected = set(dataset.target_species)
        selected.discard(x_name)
        weights = dict(dataset.target_weights)
        row_species = self._row_species(dataset)
        current = self._table.currentIndex()
        current_cell = (current.row(), current.column()) if current.isValid() else None
        keys = (dataset.dataset_id, tuple(row_species))
        if keys != self._row_keys:
            # Only another dataset or roster replaces rows; ordinary edits
            # update cells in place and keep the current cell.
            self._table.setRowCount(0)
            self._table.setRowCount(len(row_species))
            self._row_keys = keys
            current_cell = None
        for row, species in enumerate(row_species):
            observations = self._observations_for_target(dataset, species)
            has_data = bool(observations)
            local = self._local_initial(dataset, species)
            group_config = self._shared_initial_group(dataset.dataset_id, species)
            group_origin = (
                group_config is not None
                and group_config.origin_dataset_id == dataset.dataset_id
            )
            group_member = group_config is not None and not group_origin
            local_key = (mapped.set_id, species)
            if local_key in self._invalid_initial_rows:
                raw_initial = self._invalid_initial_rows[local_key]
            elif group_config is not None:
                raw_initial = self._initial_raw(group_config)
            else:
                raw_initial = self._initial_raw(local) if local is not None else {}
            member_tooltip = (
                f"Shared with {self._dataset_name(group_config.origin_dataset_id)}. Edit it there."
                if group_member else ""
            )
            target_item = QtWidgets.QTableWidgetItem()
            eligible = has_data and species != x_name
            self._set_checkable(target_item, species in selected, eligible)
            target_item.setData(_DATASET_ROLE, dataset.dataset_id)
            target_item.setData(_MODEL_TARGET_ROLE, species)
            target_item.setData(_SPECIES_ROLE, species)
            self._table.setItem(row, _Col.INCLUDE, target_item)

            weight_invalid_key = (
                "target_weight",
                dataset.dataset_id,
                species,
            )
            weight_text = (
                self._invalid_raw[weight_invalid_key]
                if weight_invalid_key in self._invalid_raw
                else self._float_text(weights.get(species, 1.0))
            )
            weight_item = QtWidgets.QTableWidgetItem(
                weight_text if has_data else ""
            )
            weight_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._set_editable(weight_item, has_data)
            weight_item.setData(_DATASET_ROLE, dataset.dataset_id)
            weight_item.setData(_MODEL_TARGET_ROLE, species)
            weight_item.setData(_SPECIES_ROLE, species)
            if weight_invalid_key in self._invalid_raw:
                weight_item.setBackground(_INVALID_BG)
                weight_item.setForeground(_INVALID_FG)
                weight_item.setToolTip(_TARGET_WEIGHT_MESSAGE)
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

            tolerance_mode = str(raw_initial.get("mode", "disabled"))
            for column, field in (
                (_Col.INITIAL, "initial"),
                (_Col.LOWER, "lower"),
                (_Col.UPPER, "upper"),
            ):
                invalid_key = (mapped.set_id, species, field)
                initial_item = QtWidgets.QTableWidgetItem(
                    self._invalid_initial_raw[invalid_key][0]
                    if invalid_key in self._invalid_initial_raw
                    else str(raw_initial.get(field, ""))
                )
                self._set_editable(
                    initial_item,
                    local is not None and not group_member
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
                if invalid_key in self._invalid_initial_raw:
                    initial_item.setBackground(_INVALID_BG)
                    initial_item.setForeground(_INVALID_FG)
                    initial_item.setToolTip(self._invalid_initial_raw[invalid_key][1])
                elif member_tooltip:
                    initial_item.setToolTip(member_tooltip)
                self._table.setItem(row, column, initial_item)

            mode = QtWidgets.QComboBox(self._table)
            ignore_wheel(mode)
            mode.setObjectName("global_fit_vary_ic_combo")
            for label, value, tooltip in _VARY_IC_CHOICES:
                mode.addItem(label, value)
                item_index = mode.model().index(mode.count() - 1, 0)
                mode.model().setData(
                    item_index,
                    tooltip,
                    Qt.ItemDataRole.ToolTipRole,
                )
            if self._shows_zero_initial(raw_initial):
                # A percentage of zero is no range; only Minimum / Maximum applies.
                for value in ("symmetric_percent", "asymmetric_percent"):
                    choice = mode.model().item(mode.findData(value))
                    choice.setEnabled(False)
                    choice.setToolTip(ZERO_IC_PERCENT_MESSAGE)
            mode.setEnabled(local is not None and not group_member)
            mode.setCurrentIndex(
                max(0, mode.findData(tolerance_mode))
            )
            self._set_vary_ic_combo_tooltip(mode)
            if member_tooltip:
                mode.setToolTip(member_tooltip)
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
                and tolerance_mode != "disabled"
                and not group_member,
            )
            check_item.setData(_DATASET_ROLE, dataset.dataset_id)
            check_item.setData(_SET_ROLE, mapped.set_id)
            check_item.setData(_SPECIES_ROLE, species)
            if member_tooltip:
                check_item.setToolTip(member_tooltip)
            self._table.setItem(row, _Col.LOG10, check_item)
            self._refresh_vary_global_availability(row)
        if current_cell is not None and current_cell[0] < self._table.rowCount():
            self._table.setCurrentCell(*current_cell)

    @staticmethod
    def _shows_zero_initial(raw_initial: dict[str, object]) -> bool:
        try:
            return float(str(raw_initial.get("initial", ""))) == 0.0
        except ValueError:
            return False

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
        mode = combo.currentData()
        initial_item = self._table.item(row, _Col.INITIAL)
        if initial_item is None:
            return
        try:
            initial = float(initial_item.text())
            if not math.isfinite(initial) or initial < 0.0:
                raise ValueError
        except ValueError:
            # A mode transition still authors a fresh range when Starting IC
            # is incomplete.  The row validator retains the invalid Starting
            # IC and Run remains blocked until the user repairs it.
            initial = None

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
                self._float_text(six_digit_value(10.0 * initial))
                if initial is not None and initial > 0.0
                else ""
            )
            disable_log = True

        # Signals from the two cleared cells must not stage half-written
        # rows. The single explicit stage below sees the complete row.
        table_blocker = QtCore.QSignalBlocker(self._table)
        lower_item.setText(lower)
        upper_item.setText(upper)
        if disable_log:
            log_item = self._table.item(row, _Col.LOG10)
            if log_item is not None:
                self._set_checkable(log_item, False, mode != "disabled")
        del table_blocker
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
            setup = self._current_setup()
            if setup is not None:
                mode_index = self._weight_mode_combo.findData(setup.dataset_weight_mode)
                self._weight_mode_combo.setCurrentIndex(max(0, mode_index))
            self._populate_table()
            self._refresh_dataset_weight_editor()
        finally:
            self._dataset_weight_edit.blockSignals(False)
            self._weight_mode_combo.blockSignals(False)
            self._table.blockSignals(False)
            self._refreshing = False
            self.setUpdatesEnabled(True)
        self._refresh_error()

    # ------------------------------------------------------------------
    # Edits
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
            self._stage_initial_row_at(item.row())
            return
        if item.column() == _Col.VARY_GLOBAL:
            self._on_vary_global_changed(item.row(), item)
            return
        if item.column() == _Col.WEIGHT:
            self._on_target_weight_edited(item, dataset_id, species)
            return
        if item.column() == _Col.INCLUDE:
            self._stage_visible_target_selection(dataset_id)
            return

    def _on_target_weight_edited(
        self,
        item: QtWidgets.QTableWidgetItem,
        dataset_id: str,
        species: str,
    ) -> None:
        key = ("target_weight", dataset_id, species)
        try:
            value = float(item.text())
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError
        except ValueError:
            self._invalid_raw[key] = item.text()
            self._mark_invalid(item, _TARGET_WEIGHT_MESSAGE)
            return
        self._invalid_raw.pop(key, None)
        dataset = self._projected_dataset(dataset_id)
        weights = dict(dataset.target_weights)
        if weights.get(species, 1.0) == value:
            self._refresh_visible_presentation()
            return
        weights[species] = value
        self._submit_dataset(replace(dataset, target_weights=tuple(weights.items())))

    def _submit_dataset(self, dataset: FitDatasetLocalSettings) -> None:
        setup = self._current_setup()
        self._submit(replace(
            setup,
            datasets=tuple(
                dataset if item.dataset_id == dataset.dataset_id else item
                for item in setup.datasets
            ),
        ))

    def _stage_initial_row_at(self, row: int) -> None:
        """Apply one complete initial-condition row, or hold its text."""

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
        accepted = self._accepted_initial(set_id, species)
        if accepted is None:
            return
        key = (set_id, species)
        group = self._shared_initial_group(dataset_id, species)
        if group is not None and group.origin_dataset_id != dataset_id:
            return
        mode = str(mode_widget.currentData() or "disabled")
        log10 = mode != "disabled" and log_item.checkState() == Qt.CheckState.Checked
        raw: dict[str, object] = {
            field: str(value.text()) for field, value in items.items()
        }
        raw.update({"mode": mode, "log10": log10})

        typed: dict[str, float] = {}
        parse_errors: dict[str, str] = {}
        required_fields = {"initial"}
        if mode == "symmetric_percent":
            required_fields.add("lower")
        elif mode in {"asymmetric_percent", "absolute"}:
            required_fields.update({"lower", "upper"})
        for field in required_fields:
            try:
                typed[field] = parse_finite_numeric_text(
                    str(raw[field]), field=_INITIAL_FIELD_HEADERS[field]
                )
            except ValueError as exc:
                parse_errors[field] = str(exc)
        if not parse_errors:
            # A tolerance error lies in Lower/Upper alone; a row error may
            # involve the Starting IC too.
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
            except ValueError as caught_error:
                parse_errors = dict.fromkeys(required_fields - {"initial"}, str(caught_error))
        if not parse_errors:
            try:
                initial = FitInitialVariable(
                    set_id=set_id,
                    species=species,
                    initial=float(typed["initial"]),
                    tolerance=tolerance,
                    log10=log10,
                )
            except ValueError as caught_error:
                parse_errors = dict.fromkeys(required_fields, str(caught_error))
        if parse_errors:
            self._drop_invalid_initial_row(set_id, species)
            self._invalid_initial_rows[key] = raw
            for field, message in parse_errors.items():
                self._invalid_initial_raw[(set_id, species, field)] = (str(raw[field]), message)
            self._refresh_visible_presentation()
            return
        self._drop_invalid_initial_row(set_id, species)
        setup = self._current_setup()
        if group is not None:
            if initial.tolerance.mode == "disabled":
                # Turning off Vary IC on the origin dissolves its group; every
                # member keeps its own Starting IC, and the origin's Vary IC is
                # off.  The row shows the group's value unless the user typed
                # another Starting IC, which the origin then keeps.
                fixed = replace(
                    accepted,
                    initial=accepted.initial if initial.initial == group.initial else initial.initial,
                    tolerance=initial.tolerance,
                    log10=False,
                )
                self._submit_groups(
                    tuple(item for item in setup.shared_initial_groups if item != group),
                    initials=tuple(
                        fixed if item.set_id == set_id and item.species == species else item
                        for item in setup.initials
                    ),
                )
                return
            self._submit_groups(tuple(
                replace(item, initial=initial.initial, tolerance=initial.tolerance, log10=initial.log10)
                if item == group else item
                for item in setup.shared_initial_groups
            ))
            return
        if initial == accepted:
            self._refresh_visible_presentation()
            return
        self._submit(replace(
            setup,
            initials=tuple(
                initial if item.set_id == set_id and item.species == species else item
                for item in setup.initials
            ),
        ))

    def _drop_invalid_initial_row(self, set_id: str, species: str) -> None:
        self._invalid_initial_rows.pop((set_id, species), None)
        for field in ("initial", "lower", "upper"):
            self._invalid_initial_raw.pop((set_id, species, field), None)

    @staticmethod
    def _with_targets(
        dataset: FitDatasetLocalSettings,
        selected: tuple[str, ...],
    ) -> FitDatasetLocalSettings:
        """Select targets; every species keeps its weight, a new target weighs 1."""

        weights = dict(dataset.target_weights)
        for target in selected:
            weights.setdefault(target, 1.0)
        return replace(dataset, target_species=selected, target_weights=tuple(weights.items()))

    def _stage_visible_target_selection(self, dataset_id: str) -> None:
        projected = self._projected_dataset(dataset_id)
        if projected is None:
            return
        checked = {
            str(item.data(_MODEL_TARGET_ROLE) or "").strip()
            for row in range(self._table.rowCount())
            for item in (self._table.item(row, _Col.INCLUDE),)
            if item is not None
            and str(item.data(_DATASET_ROLE) or "") == dataset_id
            and item.checkState() == Qt.CheckState.Checked
        }
        # Targets keep the dataset's column order, as on open and Copy
        # Targets, so the same ticks are always the same setup.
        selected = tuple(target for target in self._eligible_targets(dataset_id) if target in checked)
        if selected == projected.target_species:
            self._refresh_visible_presentation()
            return
        self._submit_dataset(self._with_targets(projected, selected))

    def _on_dataset_weight_edited(self) -> None:
        if self._refreshing or self._running:
            return
        raw = self._dataset_weight_edit.text()
        dataset_id = str(self._current_dataset_id or "").strip()
        projected = self._projected_dataset(dataset_id)
        if not dataset_id or projected is None:
            return
        key = ("dataset_weight", dataset_id)
        try:
            value = float(raw)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError
        except ValueError:
            self._invalid_raw[key] = raw
            self._mark_invalid_line_edit(
                self._dataset_weight_edit,
                "Dataset weight must be finite and positive.",
            )
            return
        self._invalid_raw.pop(key, None)
        if value == projected.weight:
            self._refresh_visible_presentation()
            return
        self._submit_dataset(replace(projected, weight=value))

    def _selected_targets(self, dataset: FitDatasetLocalSettings) -> tuple[str, ...]:
        if dataset.parametric is None:
            return dataset.target_species
        return tuple(
            target
            for target in dataset.target_species
            if target != dataset.parametric.x_name
        )

    def _weighted_species(self, dataset: FitDatasetLocalSettings) -> set[str]:
        """The targets whose weights enter the Fit: ticked species or
        observables, and measured X."""
        weighted = set(self._selected_targets(dataset))
        if dataset.parametric is not None:
            weighted.add(dataset.parametric.x_name)
        return weighted

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

    def _discard_current_invalid_entries(self) -> None:
        """Escape: discard the shown dataset's invalid weights and rows."""

        dataset_id = self._current_dataset_id
        mapped = self._mapped_dataset(dataset_id) if dataset_id else None
        if mapped is None:
            return
        for key in tuple(self._invalid_raw):
            if key[1] == dataset_id:
                self._invalid_raw.pop(key)
        for set_id, species in tuple(self._invalid_initial_rows):
            if set_id == mapped.set_id:
                self._drop_invalid_initial_row(set_id, species)
        self._refresh_visible_presentation()
        self.set_status(f"Invalid values for {mapped.display_name} discarded.")

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
        selections = {item.dataset_id: item.selected_targets for item in copied_selections}
        self._submit(replace(
            setup,
            datasets=tuple(
                self._with_targets(item, selections[item.dataset_id])
                if item.dataset_id in selections
                and selections[item.dataset_id] != item.target_species
                else item
                for item in setup.datasets
            ),
        ))
        self.set_status(f"Copy Targets: {copied} copied, {skipped} skipped.")

    def _copy_species_weights_and_vary_ic_to_compatible(self) -> None:
        """Copy row-local Vary IC ranges and target weights across datasets.

        Destination target selection, Starting IC values and dataset
        settings remain untouched.  A weight is copied for every target
        weighted in both datasets (a ticked species or observable, or
        measured X, in either role).
        """

        setup = self._current_setup()
        source = self._current_dataset()
        if setup is None or source is None:
            return

        source_mapped = self._mapped_dataset(source.dataset_id)
        if source_mapped is None:
            return
        source_weights = dict(source.target_weights)
        source_weighted = self._weighted_species(source)
        copied_ranges = 0
        copied_weights = 0
        skipped = 0
        not_copied: dict[str, list[str]] = {}
        source_initials = {
            item.species: item
            for item in setup.initials
            if item.set_id == source_mapped.set_id
        }
        initials = {(item.set_id, item.species): item for item in setup.initials}
        datasets = {item.dataset_id: item for item in setup.datasets}
        for destination in setup.datasets:
            if destination.dataset_id == source.dataset_id:
                continue
            destination_mapped = self._mapped_dataset(destination.dataset_id)
            if destination_mapped is None:
                continue
            destination_weighted = self._weighted_species(destination)
            for species, source_initial in source_initials.items():
                destination_key = (destination_mapped.set_id, species)
                destination_initial = initials.get(destination_key)
                if destination_initial is None:
                    skipped += 1
                    continue
                if self._shared_initial_group(destination.dataset_id, species) is not None:
                    # Group membership owns the effective IC configuration;
                    # leave the dataset's own row untouched.
                    skipped += 1
                    continue
                source_group = self._shared_initial_group(source.dataset_id, species)
                source_config = source_group if source_group is not None else source_initial
                held = self._invalid_initial_rows.get(destination_key)
                try:
                    copied_initial = replace(
                        destination_initial,
                        # A typed Starting IC that could not apply yet is kept.
                        initial=(
                            destination_initial.initial if held is None
                            else parse_finite_numeric_text(
                                str(held["initial"]), field=_INITIAL_FIELD_HEADERS["initial"]
                            )
                        ),
                        tolerance=source_config.tolerance,
                        log10=source_config.log10,
                    )
                except ValueError:
                    # The copied range cannot contain this destination's
                    # Starting IC (e.g. a percentage around zero), so this
                    # row is not compatible and stays as is, typed text included.
                    not_copied.setdefault(
                        destination_mapped.display_name, []
                    ).append(species)
                    skipped += 1
                    continue
                self._drop_invalid_initial_row(*destination_key)
                initials[destination_key] = copied_initial
                copied_ranges += 1

            # Weights are copied independently of Vary IC, for every target
            # weighted in both datasets, observables included.
            destination_weights = dict(destination.target_weights)
            for species in source_weighted & destination_weighted:
                self._invalid_raw.pop(("target_weight", destination.dataset_id, species), None)
                destination_weights[species] = source_weights.get(species, 1.0)
                copied_weights += 1
            datasets[destination.dataset_id] = replace(
                destination, target_weights=tuple(destination_weights.items()),
            )

        self._submit(replace(
            setup,
            datasets=tuple(datasets[item.dataset_id] for item in setup.datasets),
            initials=tuple(initials[(item.set_id, item.species)] for item in setup.initials),
        ))
        detail = " ".join(
            f"Not copied to {name}: {', '.join(names)}; its Starting IC must be a number inside the copied range."
            for name, names in not_copied.items()
        )
        self.set_status(
            "Copy Species Weights and Vary IC: "
            f"{copied_ranges} ranges, {copied_weights} weights copied, "
            f"{skipped} skipped."
            + (f" {detail}" if detail else "")
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
        self._submit(
            replace(
                self._current_setup(),
                dataset_weight_mode=mode,
            )
        )

    def _refresh_dataset_weight_editor(self) -> None:
        dataset = self._current_dataset()
        if dataset is None:
            self._dataset_weight_edit.clear()
            self._weight_mode_combo.setEnabled(False)
            self._dataset_weight_edit.setEnabled(False)
            return
        self._weight_mode_combo.setEnabled(not self._running)
        invalid_key = ("dataset_weight", dataset.dataset_id)
        if invalid_key in self._invalid_raw:
            self._dataset_weight_edit.setText(self._invalid_raw[invalid_key])
            self._mark_invalid_line_edit(
                self._dataset_weight_edit,
                "Dataset weight must be finite and positive.",
            )
        else:
            self._dataset_weight_edit.setText(self._float_text(dataset.weight))
            self._clear_invalid_line_edit(self._dataset_weight_edit)
            if self._weight_mode_combo.currentData() != "user":
                self._dataset_weight_edit.setToolTip("Used only with Custom.")
        self._dataset_weight_edit.setEnabled(
            not self._running
            and self._weight_mode_combo.currentData() == "user"
        )

    def _mark_invalid(self, item: QtWidgets.QTableWidgetItem, message: str) -> None:
        previous_blocked = self._table.blockSignals(True)
        try:
            item.setBackground(_INVALID_BG)
            item.setForeground(_INVALID_FG)
            item.setToolTip(message)
        finally:
            self._table.blockSignals(previous_blocked)
        self._refresh_error()

    def _mark_invalid_line_edit(self, edit: QtWidgets.QLineEdit, message: str) -> None:
        edit.setStyleSheet("background: rgb(255, 200, 200); color: rgb(80, 0, 0);")
        edit.setToolTip(message)
        self._refresh_error()

    def _clear_invalid_line_edit(self, edit: QtWidgets.QLineEdit) -> None:
        edit.setStyleSheet("")
        edit.setToolTip("")

    def _blocking_invalid_keys(self) -> tuple[tuple[str, ...], ...]:
        """Held weight text; a dataset weight counts only while Custom uses it."""

        setup = self._current_setup()
        custom = setup is not None and setup.dataset_weight_mode == "user"
        return tuple(key for key in self._invalid_raw if custom or key[0] != "dataset_weight")

    def _current_has_invalid_entries(self) -> bool:
        dataset_id = self._current_dataset_id
        mapped = self._mapped_dataset(dataset_id) if dataset_id else None
        return mapped is not None and (
            any(key[1] == dataset_id for key in self._blocking_invalid_keys())
            or any(set_id == mapped.set_id for set_id, _species in self._invalid_initial_rows)
        )

    def _refresh_error(self) -> None:
        locations = self.invalid_editor_locations
        current = self._current_has_invalid_entries()
        text = ""
        if locations:
            more = f" (+{len(locations) - 1} more)" if len(locations) > 1 else ""
            text = f"Invalid value in {locations[0]}{more}."
            if current:
                text += "\nPress Esc to discard this dataset's invalid values."
        self._error_label.setText(text)
        self._error_label.setToolTip("\n".join(locations))
        self._error_label.setVisible(bool(text))
        for shortcut in self._discard_shortcuts:
            shortcut.setEnabled(current)
        self.validityChanged.emit()

    @property
    def invalid_editor_locations(self) -> tuple[str, ...]:
        """Name each invalid entry by dataset, species, and column in table order."""

        datasets = self._projected_setup.datasets if self._projected_setup else ()
        order = {item.dataset_id: index for index, item in enumerate(datasets)}
        rows = {item.dataset_id: self._row_species(item) for item in datasets}
        by_set = {item.set_id: item.dataset_id for item in self._origin.mapped_datasets}
        columns = {"initial": _Col.INITIAL, "lower": _Col.LOWER, "upper": _Col.UPPER}

        def row(dataset_id: str, species: str) -> int:
            species_rows = rows.get(dataset_id, [])
            return species_rows.index(species) if species in species_rows else len(species_rows)

        located = [
            (key[1], -1, 0, "Dataset weight") if key[0] == "dataset_weight"
            else (key[1], row(key[1], key[2]), _Col.WEIGHT, f"{key[2]} Weight")
            for key in self._blocking_invalid_keys()
        ]
        located += [
            (
                by_set[set_id], row(by_set[set_id], species), columns[field],
                f"{species} {_INITIAL_FIELD_HEADERS[field]}",
            )
            for set_id, species, field in self._invalid_initial_raw
        ]
        located.sort(key=lambda item: (order.get(item[0], len(order)), item[1], item[2]))
        return tuple(
            f"{self._dataset_name(dataset_id)}: {label}"
            for dataset_id, _row, _column, label in located
        )


__all__ = ["UnifiedSpeciesTable"]
