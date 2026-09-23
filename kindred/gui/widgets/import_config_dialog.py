"""Configure incoming dataset units and columns without loading observations."""

from __future__ import annotations

import html
import itertools
from collections import Counter
from dataclasses import dataclass
from typing import Mapping

from PySide6 import QtCore, QtGui, QtWidgets

from kindred.core.quantity_units import format_unit_for_display
from kindred.core.datasets.csv_import import read_csv_rows_from_bytes
from kindred.core.datasets.excel_import import ExcelWorkbookPair
from kindred.core.datasets.foundation import ImportedSource
from kindred.core.datasets.source_rules import PhysicalColumn
from kindred.core.datasets.units import (
    CONCENTRATION_UNIT_DISPLAY,
    TIME_UNIT_DISPLAY,
    parse_unit,
)
from kindred.gui.import_workflow import (
    ImportConfig,
    ImportSettings,
    classify_unit_row,
    resolve_column_unit,
    resolve_sheet_plan,
    sharing_settings,
)
from kindred.gui.theme_manager import semantic_color
from kindred.gui.ui_helpers import (
    SCIENTIFIC_NAME_HTML_ROLE, ScientificNameComboBox,
    clamp_initial_size, set_bounded_label_text,
)

__all__ = ["ImportConfigDialog", "ImportDialogResult"]

_TIME_CANDIDATES = ("time", "time_s", "t", "Time", "T", "x")
_MAX_PREVIEW_ROWS = 20


@dataclass
class ImportDialogResult:
    config: ImportConfig | None = None
    action: str = "cancel"


@dataclass
class _Sheet:
    name: str | None
    columns: tuple[PhysicalColumn, ...]
    preview_rows: list[Mapping[PhysicalColumn, object]]
    unit_row: Mapping[PhysicalColumn, object] | None
    settings: ImportSettings
    included: bool
    load_error: str = ""


def _concentration_default(unit_row: Mapping[PhysicalColumn, object] | None) -> str:
    """Initialize the default from the most common declaration, ties alphabetical."""
    counts: Counter[str] = Counter()
    for value in (unit_row or {}).values():
        text = str(value).strip().replace("µ", "u").replace("μ", "u")
        try:
            category, _ = parse_unit(text)
        except ValueError:
            continue
        if category == "concentration":
            counts[text] += 1
    return min(counts, key=lambda unit: (-counts[unit], unit)) if counts else "M"


class ImportConfigDialog(QtWidgets.QDialog):
    """Widgets edit one authoritative configuration for each physical sheet."""

    def __init__(
        self,
        source: ImportedSource,
        remaining_count: int = 0,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._source = source
        self._sheets: list[_Sheet] = []
        self._current_sheet: _Sheet | None = None
        self._result: ImportDialogResult | None = None
        self.setWindowTitle(f"Import: {source.original_filename}")
        self.setModal(True)
        self._build_ui(remaining_count)
        try:
            self._load_preview()
        except Exception:
            self.deleteLater()
            raise
        clamp_initial_size(self, 760, 620)

    def _build_ui(self, remaining_count: int) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        workspace = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical, self)
        workspace.setChildrenCollapsible(False)
        layout.addWidget(workspace, 1)
        upper = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal, workspace)
        upper.setChildrenCollapsible(False)

        sheets = QtWidgets.QGroupBox("Sheets", upper)
        sheet_layout = QtWidgets.QVBoxLayout(sheets)
        sheet_layout.setSpacing(6)
        bulk = QtWidgets.QHBoxLayout()
        self._btn_select_all = QtWidgets.QPushButton("Select all", sheets)
        self._btn_invert_sheets = QtWidgets.QPushButton("Invert", sheets)
        bulk.addWidget(self._btn_select_all)
        bulk.addWidget(self._btn_invert_sheets)
        sheet_layout.addLayout(bulk)
        self._sheet_list = QtWidgets.QListWidget(sheets)
        self._sheet_list.setMinimumSize(150, 70)
        sheet_layout.addWidget(self._sheet_list, 1)
        self._apply_remaining_cb = QtWidgets.QCheckBox(sheets)
        excel = self._source.original_format == "xlsx"
        scope = "other sheets" if excel else "remaining files"
        if excel and remaining_count:
            scope += "\nand remaining files"
        self._apply_remaining_cb.setText(f"Use for {scope}")
        self._apply_remaining_cb.setVisible(excel or remaining_count > 0)
        sheet_layout.addWidget(self._apply_remaining_cb)
        self._btn_select_compatible = QtWidgets.QPushButton(
            "Select compatible sheets", sheets
        )
        sheet_layout.addWidget(self._btn_select_compatible)
        if not excel:
            sheets.setTitle("Shared settings")
            self._sheet_list.hide()
            self._btn_select_all.hide()
            self._btn_invert_sheets.hide()
            self._btn_select_compatible.hide()
            sheets.setVisible(remaining_count > 0)
            sheet_layout.addStretch(1)

        settings = QtWidgets.QGroupBox(upper)
        settings_layout = QtWidgets.QVBoxLayout(settings)
        settings_layout.setSpacing(6)
        self._settings_label = QtWidgets.QLabel(settings)
        font = self._settings_label.font()
        font.setBold(True)
        self._settings_label.setFont(font)
        settings_layout.addWidget(self._settings_label)
        form = QtWidgets.QFormLayout()
        form.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        self._time_combo = QtWidgets.QComboBox(settings)
        self._time_combo.setSizeAdjustPolicy(
            QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon
        )
        self._time_combo.setMinimumContentsLength(12)
        form.addRow("Time column", self._time_combo)
        self._time_unit_combo = ScientificNameComboBox(settings)
        form.addRow("Time unit", self._time_unit_combo)
        self._conc_unit_combo = ScientificNameComboBox(settings)
        for unit in CONCENTRATION_UNIT_DISPLAY:
            self._conc_unit_combo.addItem(unit, unit)
            self._conc_unit_combo.setItemData(
                self._conc_unit_combo.count() - 1,
                format_unit_for_display(unit),
                SCIENTIFIC_NAME_HTML_ROLE,
            )
        self._conc_unit_combo.setToolTip(
            "Default for columns without a detected concentration unit."
        )
        form.addRow("Default concentration unit", self._conc_unit_combo)
        settings_layout.addLayout(form)
        self._no_unit_row_cb = QtWidgets.QCheckBox(
            "No unit row (assume s and M)", settings
        )
        settings_layout.addWidget(self._no_unit_row_cb)
        self._unit_info_label = QtWidgets.QLabel(settings)
        settings_layout.addWidget(self._unit_info_label)
        species_header = QtWidgets.QHBoxLayout()
        species_header.addWidget(QtWidgets.QLabel("Species columns", settings))
        species_header.addStretch(1)
        self._species_count_label = QtWidgets.QLabel(settings)
        species_header.addWidget(self._species_count_label)
        settings_layout.addLayout(species_header)
        self._species_table = QtWidgets.QTableWidget(0, 3, settings)
        self._species_table.setHorizontalHeaderLabels(["Use", "Column", "Conversion"])
        self._species_table.setMinimumHeight(80)
        self._species_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self._species_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self._species_table.setSelectionMode(
            QtWidgets.QAbstractItemView.SingleSelection
        )
        self._species_table.verticalHeader().hide()
        self._species_table.verticalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.Fixed
        )
        self._species_table.verticalHeader().setDefaultSectionSize(
            self.fontMetrics().height() + 10
        )
        self._species_table.horizontalHeader().setSectionResizeMode(
            0, QtWidgets.QHeaderView.ResizeToContents
        )
        self._species_table.horizontalHeader().setStretchLastSection(True)
        self._species_table.setColumnWidth(1, 155)
        self._species_table.setColumnWidth(2, 145)
        settings_layout.addWidget(self._species_table, 1)
        upper.setStretchFactor(0, 0)
        upper.setStretchFactor(1, 1)
        upper.setSizes([220, 520])

        preview = QtWidgets.QGroupBox(workspace)
        preview_layout = QtWidgets.QVBoxLayout(preview)
        preview_layout.setSpacing(6)
        self._preview_label = QtWidgets.QLabel(preview)
        self._preview_label.setFont(font)
        preview_layout.addWidget(self._preview_label)
        self._preview_table = QtWidgets.QTableWidget(preview)
        self._preview_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self._preview_table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self._preview_table.setMinimumHeight(90)
        self._preview_table.verticalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.Fixed
        )
        self._preview_table.verticalHeader().setDefaultSectionSize(
            self.fontMetrics().height() + 10
        )
        preview_layout.addWidget(self._preview_table, 1)
        self._preview_error_label = QtWidgets.QLabel(preview)
        preview_layout.addWidget(self._preview_error_label)
        workspace.setStretchFactor(0, 3)
        workspace.setStretchFactor(1, 2)
        workspace.setSizes([360, 200])

        self._readiness_label = QtWidgets.QLabel(self)
        layout.addWidget(self._readiness_label)
        footer = QtWidgets.QHBoxLayout()
        footer.addStretch(1)
        self._btn_import = QtWidgets.QPushButton("Import", self)
        self._btn_skip = QtWidgets.QPushButton("Skip", self)
        self._btn_cancel = QtWidgets.QPushButton("Cancel All", self)
        for button in (self._btn_import, self._btn_skip, self._btn_cancel):
            footer.addWidget(button)
            button.setAutoDefault(False)
        self._btn_import.setDefault(True)
        layout.addLayout(footer)
        self._btn_import.clicked.connect(self._on_import)
        self._btn_skip.clicked.connect(self._on_skip)
        self._btn_cancel.clicked.connect(self.reject)
        self._sheet_list.currentRowChanged.connect(self._on_current_sheet_changed)
        self._sheet_list.itemChanged.connect(self._on_sheet_checked)
        self._btn_select_all.clicked.connect(
            lambda: self._set_checked_sheets(invert=False)
        )
        self._btn_invert_sheets.clicked.connect(
            lambda: self._set_checked_sheets(invert=True)
        )
        self._btn_select_compatible.clicked.connect(self._select_compatible_sheets)
        self._apply_remaining_cb.toggled.connect(self._update_import_enabled)
        self._time_combo.currentIndexChanged.connect(self._on_time_column_changed)
        self._time_unit_combo.currentIndexChanged.connect(self._on_time_unit_changed)
        self._conc_unit_combo.currentIndexChanged.connect(
            self._on_concentration_changed
        )
        self._no_unit_row_cb.toggled.connect(self._on_no_unit_row_toggled)
        self._species_table.itemChanged.connect(self._on_species_changed)
        self._apply_feedback_palette()

    def _make_sheet(self, name, columns, rows, load_error="") -> _Sheet:
        columns = tuple(columns)
        unit_row = classify_unit_row(rows[0]) if rows else None
        time_column = next(
            (
                column
                for candidate in _TIME_CANDIDATES
                for column in columns
                if column.assigned_name == candidate
            ),
            None,
        )
        selected = set(columns)
        # The initial CSV draft includes the hidden time column; Excel excludes it.
        if self._source.original_format == "xlsx":
            selected.discard(time_column)
        return _Sheet(
            name,
            columns,
            rows,
            unit_row,
            ImportSettings(
                time_column,
                selected,
                concentration_default=_concentration_default(unit_row),
            ),
            included=bool(columns),
            load_error=load_error,
        )

    def _load_preview(self) -> None:
        if self._source.original_format == "csv":
            columns, rows = read_csv_rows_from_bytes(
                self._source.exact_source_bytes,
                source_label=self._source.original_filename,
            )
            try:
                preview = [
                    row.values for row in itertools.islice(rows, _MAX_PREVIEW_ROWS)
                ]
            finally:
                rows.close()
            self._sheets.append(self._make_sheet(None, columns, preview))
            self._on_current_sheet_changed(0)
            return
        with ExcelWorkbookPair(
            self._source.exact_source_bytes,
            source_label=self._source.original_filename,
        ) as workbooks:
            for name in workbooks.sheet_names:
                rows = workbooks.read_sheet_rows(name)
                try:
                    # Preserve the established Excel preview's one-row lookahead.
                    preview = []
                    columns = ()
                    for index, row in enumerate(rows):
                        if index == 0:
                            columns = tuple(row)
                        if index >= _MAX_PREVIEW_ROWS:
                            break
                        preview.append(row)
                    sheet = self._make_sheet(name, columns, preview)
                except Exception as exc:
                    sheet = self._make_sheet(name, (), [], str(exc))
                finally:
                    rows.close()
                self._sheets.append(sheet)
        with QtCore.QSignalBlocker(self._sheet_list):
            for sheet in self._sheets:
                item = QtWidgets.QListWidgetItem(sheet.name)
                if sheet.columns:
                    item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
                else:
                    item.setFlags(item.flags() & ~QtCore.Qt.ItemIsUserCheckable)
                item.setCheckState(
                    QtCore.Qt.Checked if sheet.included else QtCore.Qt.Unchecked
                )
                item.setToolTip(sheet.load_error or sheet.name)
                self._sheet_list.addItem(item)
        if self._sheets:
            first = next(
                (i for i, sheet in enumerate(self._sheets) if sheet.columns), 0
            )
            self._sheet_list.setCurrentRow(first)
        else:
            self._update_import_enabled()

    def _on_current_sheet_changed(self, row: int) -> None:
        if row < 0:
            return
        self._current_sheet = self._sheets[row]
        sheet = self._current_sheet
        name = sheet.name if sheet.name is not None else self._source.original_filename
        set_bounded_label_text(
            self._settings_label, f"Settings for: {name}", max_width=640
        )
        set_bounded_label_text(self._preview_label, f"Preview: {name}", max_width=1000)
        with QtCore.QSignalBlocker(self._time_combo):
            self._time_combo.clear()
            self._time_combo.addItem("No experimental time", -1)
            for column in sheet.columns:
                self._time_combo.addItem(column.assigned_name, column.source_index)
                self._time_combo.setItemData(
                    self._time_combo.count() - 1,
                    column.assigned_name,
                    QtCore.Qt.ToolTipRole,
                )
            index = (
                sheet.settings.time_column.source_index
                if sheet.settings.time_column
                else -1
            )
            self._time_combo.setCurrentIndex(self._time_combo.findData(index))
        self._render_units()
        self._render_species()
        self._preview_table.clear()
        self._preview_table.setColumnCount(len(sheet.columns))
        self._preview_table.setRowCount(len(sheet.preview_rows))
        self._preview_table.setHorizontalHeaderLabels(
            [col.assigned_name for col in sheet.columns]
        )
        for row_index, row in enumerate(sheet.preview_rows):
            for column_index, column in enumerate(sheet.columns):
                text = str(row.get(column, ""))
                item = QtWidgets.QTableWidgetItem(text)
                item.setToolTip(text)
                if row_index == 0 and sheet.unit_row is not None:
                    font = item.font()
                    font.setItalic(True)
                    item.setFont(font)
                    item.setToolTip(f"Detected unit: {text}")
                self._preview_table.setItem(row_index, column_index, item)
        self._preview_table.resizeColumnsToContents()
        for index in range(len(sheet.columns)):
            self._preview_table.setColumnWidth(
                index, min(300, self._preview_table.columnWidth(index))
            )
        set_bounded_label_text(
            self._preview_error_label, sheet.load_error, max_width=1000
        )
        self._preview_error_label.setVisible(bool(sheet.load_error))
        self._update_import_enabled()

    def _render_units(self) -> None:
        sheet = self._current_sheet
        settings = sheet.settings
        fixed = settings.override_no_unit_row
        with QtCore.QSignalBlocker(self._time_unit_combo):
            self._time_unit_combo.clear()
            if fixed:
                self._time_unit_combo.addItem("s (assumed)", None)
            else:
                declared = str(
                    (sheet.unit_row or {}).get(settings.time_column, "")
                ).strip()
                evidence_raw = declared or settings.time_fallback
                evidence = format_unit_for_display(evidence_raw)
                suffix = "" if declared else "; fallback"
                self._time_unit_combo.addItem(f"From file ({evidence_raw}{suffix})", None)
                self._time_unit_combo.setItemData(
                    self._time_unit_combo.count() - 1,
                    f"From file ({evidence}{html.escape(suffix)})",
                    SCIENTIFIC_NAME_HTML_ROLE,
                )
                for unit in TIME_UNIT_DISPLAY:
                    self._time_unit_combo.addItem(unit, unit)
                    self._time_unit_combo.setItemData(
                        self._time_unit_combo.count() - 1,
                        format_unit_for_display(unit),
                        SCIENTIFIC_NAME_HTML_ROLE,
                    )
                self._time_unit_combo.setCurrentIndex(
                    self._time_unit_combo.findData(settings.time_choice)
                )
        self._time_unit_combo.setEnabled(not fixed and settings.time_column is not None)
        with QtCore.QSignalBlocker(self._conc_unit_combo):
            self._conc_unit_combo.setCurrentIndex(
                self._conc_unit_combo.findData(
                    "M" if fixed else settings.concentration_default
                )
            )
        self._conc_unit_combo.setEnabled(not fixed)
        with QtCore.QSignalBlocker(self._no_unit_row_cb):
            self._no_unit_row_cb.setEnabled(sheet.unit_row is not None)
            self._no_unit_row_cb.setChecked(fixed)
        if fixed:
            info = "All incoming values use s and M; the first row is data."
        elif settings.time_column is not None:
            try:
                unit, _ = resolve_column_unit(
                    settings.time_column, "time", sheet.unit_row, settings
                )
                info = f"Time: {format_unit_for_display(unit)} → s. Concentrations convert to M."
            except ValueError as exc:
                info = str(exc)
        else:
            info = "No recorded time; observations can be fitted against a measured X quantity."
        set_bounded_label_text(self._unit_info_label, info, max_width=640)

    def _render_species(self) -> None:
        sheet = self._current_sheet
        with QtCore.QSignalBlocker(self._species_table):
            columns = [
                col for col in sheet.columns if col != sheet.settings.time_column
            ]
            self._species_table.setRowCount(len(columns))
            for row, column in enumerate(columns):
                check = QtWidgets.QTableWidgetItem()
                check.setFlags(
                    QtCore.Qt.ItemIsEnabled
                    | QtCore.Qt.ItemIsSelectable
                    | QtCore.Qt.ItemIsUserCheckable
                )
                check.setData(QtCore.Qt.UserRole, column.source_index)
                check.setCheckState(
                    QtCore.Qt.Checked
                    if column in sheet.settings.species_columns
                    else QtCore.Qt.Unchecked
                )
                self._species_table.setItem(row, 0, check)
                label = QtWidgets.QTableWidgetItem(column.assigned_name)
                label.setToolTip(column.assigned_name)
                self._species_table.setItem(row, 1, label)
        self._refresh_conversions()

    def _refresh_conversions(self) -> None:
        sheet = self._current_sheet
        with QtCore.QSignalBlocker(self._species_table):
            for row in range(self._species_table.rowCount()):
                source_index = self._species_table.item(row, 0).data(QtCore.Qt.UserRole)
                column = next(
                    col for col in sheet.columns if col.source_index == source_index
                )
                try:
                    unit, _ = resolve_column_unit(
                        column, "concentration", sheet.unit_row, sheet.settings
                    )
                    text = f"{format_unit_for_display(unit)} → M"
                    if (
                        not sheet.settings.override_no_unit_row
                        and not str((sheet.unit_row or {}).get(column, "")).strip()
                    ):
                        text += " (fallback)"
                    tooltip = text
                except ValueError as exc:
                    text, tooltip = "Invalid unit", str(exc)
                item = QtWidgets.QTableWidgetItem(text)
                item.setToolTip(tooltip)
                self._species_table.setItem(row, 2, item)
        count = len(sheet.settings.species_columns - {sheet.settings.time_column})
        self._species_count_label.setText(f"{count} selected")

    def _on_time_column_changed(self) -> None:
        sheet = self._current_sheet
        source_index = self._time_combo.currentData()
        sheet.settings.time_column = next(
            (col for col in sheet.columns if col.source_index == source_index), None
        )
        sheet.settings.time_choice = None
        self._render_units()
        self._render_species()
        self._update_import_enabled()

    def _on_time_unit_changed(self) -> None:
        self._current_sheet.settings.time_choice = self._time_unit_combo.currentData()
        self._render_units()

    def _on_concentration_changed(self) -> None:
        self._current_sheet.settings.concentration_default = (
            self._conc_unit_combo.currentData()
        )
        self._refresh_conversions()

    def _on_no_unit_row_toggled(self, checked: bool) -> None:
        sheet = self._current_sheet
        sheet.settings.override_no_unit_row = checked
        sheet.settings.time_choice = None
        sheet.settings.time_fallback = "s"
        sheet.settings.concentration_default = (
            "M" if checked else _concentration_default(sheet.unit_row)
        )
        self._render_units()
        self._refresh_conversions()

    def _on_species_changed(self, item: QtWidgets.QTableWidgetItem) -> None:
        if item.column() != 0:
            return
        sheet = self._current_sheet
        column = next(
            col
            for col in sheet.columns
            if col.source_index == item.data(QtCore.Qt.UserRole)
        )
        if item.checkState() == QtCore.Qt.Checked:
            sheet.settings.species_columns.add(column)
        else:
            sheet.settings.species_columns.discard(column)
        self._species_count_label.setText(
            f"{len(sheet.settings.species_columns - {sheet.settings.time_column})} selected"
        )
        self._update_import_enabled()

    def _on_sheet_checked(self, item: QtWidgets.QListWidgetItem) -> None:
        self._sheets[self._sheet_list.row(item)].included = (
            item.checkState() == QtCore.Qt.Checked
        )
        self._update_import_enabled()

    def _set_checked_sheets(self, *, invert: bool) -> None:
        with QtCore.QSignalBlocker(self._sheet_list):
            for index, sheet in enumerate(self._sheets):
                sheet.included = bool(sheet.columns) and (
                    not sheet.included if invert else True
                )
                self._sheet_list.item(index).setCheckState(
                    QtCore.Qt.Checked if sheet.included else QtCore.Qt.Unchecked
                )
        self._update_import_enabled()

    def _select_compatible_sheets(self) -> None:
        settings = self._current_sheet.settings
        required = (settings.species_columns | {settings.time_column}) - {None}
        with QtCore.QSignalBlocker(self._sheet_list):
            for index, sheet in enumerate(self._sheets):
                sheet.included = required <= set(sheet.columns)
                self._sheet_list.item(index).setCheckState(
                    QtCore.Qt.Checked if sheet.included else QtCore.Qt.Unchecked
                )
        self._update_import_enabled()

    def _update_import_enabled(self) -> None:
        checked = [sheet for sheet in self._sheets if sheet.included]
        current = self._current_sheet
        shared = self._apply_remaining_cb.isChecked()
        reason = "" if checked else "Select at least one sheet to import."
        for sheet in checked:
            settings = (
                current.settings if shared and current is not None else sheet.settings
            )
            label = sheet.name or self._source.original_filename
            if settings.time_column is not None and settings.time_column not in sheet.columns:
                reason = f"{label}: select a time column present in this sheet."
                break
            species = settings.species_columns - {settings.time_column}
            if not species:
                reason = f"{label}: select at least one species column."
                break
            missing = species - set(sheet.columns)
            if missing:
                names = ", ".join(sorted(col.assigned_name for col in missing))
                reason = f"{label} is missing columns: {names}"
                break
        self._btn_import.setEnabled(not reason)
        self._btn_import.setToolTip(reason)
        set_bounded_label_text(self._readiness_label, reason, max_width=1000)
        self._readiness_label.setVisible(bool(reason))
        source_ready = (
            current is not None
            and (current.settings.time_column is None or current.settings.time_column in current.columns)
            and bool(current.settings.species_columns - {current.settings.time_column})
        )
        self._btn_select_compatible.setEnabled(shared and source_ready)
        self._btn_select_all.setEnabled(bool(self._sheets))
        self._btn_invert_sheets.setEnabled(bool(self._sheets))

    def _on_import(self) -> None:
        try:
            shared = None
            if self._apply_remaining_cb.isChecked():
                shared = sharing_settings(
                    self._current_sheet.settings, self._current_sheet.unit_row
                )
            plans = tuple(
                resolve_sheet_plan(
                    sheet.name, sheet.columns, sheet.unit_row, shared or sheet.settings
                )
                for sheet in self._sheets
                if sheet.included
            )
            if not plans:
                raise ValueError("Select at least one sheet to import.")
        except ValueError as exc:
            QtWidgets.QMessageBox.critical(self, "Import Configuration", str(exc))
            return
        self._result = ImportDialogResult(
            ImportConfig(self._source, plans, shared), "import"
        )
        self.accept()

    def _on_skip(self) -> None:
        self._result = ImportDialogResult(action="skip")
        self.reject()

    def get_result(self) -> ImportDialogResult:
        return self._result or ImportDialogResult()

    def changeEvent(self, event: QtCore.QEvent) -> None:
        super().changeEvent(event)
        if event.type() in (
            QtCore.QEvent.PaletteChange,
            QtCore.QEvent.ApplicationPaletteChange,
        ):
            self._apply_feedback_palette()

    def _apply_feedback_palette(self) -> None:
        dark = self.palette().color(QtGui.QPalette.Text).lightness() > 128
        for name, role in (
            ("_preview_error_label", "error"),
            ("_readiness_label", "warning"),
        ):
            label = getattr(self, name, None)
            if label is not None:
                palette = label.palette()
                palette.setColor(QtGui.QPalette.WindowText, semantic_color(role, dark))
                label.setPalette(palette)
