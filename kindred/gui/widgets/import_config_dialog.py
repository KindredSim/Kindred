"""Configure incoming dataset units and columns without loading observations."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Mapping

from PySide6 import QtCore, QtGui, QtWidgets

from kindred.core.quantity_units import format_unit_for_display, scientific_unit
from kindred.core.datasets.csv_import import read_csv_rows_from_bytes
from kindred.core.datasets.excel_import import ExcelWorkbookPair
from kindred.core.datasets.foundation import ImportedSource
from kindred.core.datasets.source_rules import PhysicalColumn
from kindred.core.datasets.units import (
    MEASUREMENT_UNIT_DISPLAY,
    TIME_UNIT_DISPLAY,
)
from kindred.gui.import_workflow import (
    ImportConfig,
    ImportSettings,
    UnitSource,
    classify_unit_row,
    resolve_column_unit,
    resolve_sheet_plan,
    sharing_settings,
)
from kindred.gui.theme_manager import semantic_color
from kindred.gui.ui_helpers import (
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
    settings: ImportSettings
    included: bool
    load_error: str = ""

    @property
    def first_row(self) -> Mapping[PhysicalColumn, object] | None:
        return self.preview_rows[0] if self.preview_rows else None

    @property
    def unit_row(self) -> Mapping[PhysicalColumn, object] | None:
        return classify_unit_row(self.first_row, contains_units=self.settings.first_row_contains_units)


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
        self._time_unit_combo = QtWidgets.QComboBox(settings)
        form.addRow("Time unit", self._time_unit_combo)
        self._default_unit_combo = QtWidgets.QComboBox(settings)
        self._populate_unit_choices(self._default_unit_combo, MEASUREMENT_UNIT_DISPLAY, "M")
        self._default_unit_combo.setToolTip(
            "Used by columns set to Use default. Choose Dimensionless for ratios and fractions. "
            "Enter other supported units in each column's Unit control."
        )
        form.addRow("Default unit", self._default_unit_combo)
        settings_layout.addLayout(form)
        self._unit_row_cb = QtWidgets.QCheckBox("First data row contains units", settings)
        self._unit_row_cb.setToolTip("Exclude this row from measurements. Column units are selected separately.")
        settings_layout.addWidget(self._unit_row_cb)
        self._unit_info_label = QtWidgets.QLabel(settings)
        settings_layout.addWidget(self._unit_info_label)
        species_header = QtWidgets.QHBoxLayout()
        species_header.addWidget(QtWidgets.QLabel("Species columns", settings))
        species_header.addStretch(1)
        self._species_count_label = QtWidgets.QLabel(settings)
        species_header.addWidget(self._species_count_label)
        settings_layout.addLayout(species_header)
        self._species_table = QtWidgets.QTableWidget(0, 4, settings)
        self._species_table.setHorizontalHeaderLabels(["Use", "Column", "Unit", "Conversion"])
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
            self.fontMetrics().height() + 16
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
        self._time_unit_combo.currentTextChanged.connect(self._on_time_unit_changed)
        self._default_unit_combo.currentTextChanged.connect(self._on_default_unit_changed)
        self._unit_row_cb.toggled.connect(self._on_unit_row_toggled)
        self._species_table.itemChanged.connect(self._on_species_changed)
        self._apply_feedback_palette()

    def _make_sheet(self, name, columns, rows, load_error="") -> _Sheet:
        columns = tuple(columns)
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
            ImportSettings(time_column, selected),
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
        self._render_preview()
        set_bounded_label_text(
            self._preview_error_label, sheet.load_error, max_width=1000
        )
        self._preview_error_label.setVisible(bool(sheet.load_error))
        self._update_import_enabled()

    @staticmethod
    def _populate_unit_choices(combo, choices, choice, *, default=None, declared="") -> None:
        with QtCore.QSignalBlocker(combo):
            combo.clear()
            if default is not None:
                combo.addItem("Use default", UnitSource.DEFAULT)
                if declared:
                    combo.addItem(f"From file ({declared})", UnitSource.FILE)
            for unit in choices:
                combo.addItem("Dimensionless" if unit == "1" else unit, unit)
            index = combo.findData(choice)
            if choice is UnitSource.FILE and not declared:
                index = combo.findData(UnitSource.DEFAULT)
            if index >= 0:
                combo.setCurrentIndex(index)
            else:
                combo.setCurrentIndex(-1)
                combo.setEditText(choice)

    @staticmethod
    def _unit_choice(combo) -> str | UnitSource:
        text = combo.currentText()
        index = combo.findText(text)
        return combo.itemData(index) if index >= 0 else text

    @staticmethod
    def _declared_unit(sheet, column) -> str:
        value = (sheet.unit_row or {}).get(column)
        return "" if value is None else str(value).strip()

    def _render_units(self) -> None:
        sheet = self._current_sheet
        settings = sheet.settings
        self._populate_unit_choices(
            self._time_unit_combo, TIME_UNIT_DISPLAY, settings.time_choice,
            default="s", declared=self._declared_unit(sheet, settings.time_column),
        )
        self._time_unit_combo.setEnabled(settings.time_column is not None)
        self._populate_unit_choices(self._default_unit_combo, MEASUREMENT_UNIT_DISPLAY, settings.default_unit)
        with QtCore.QSignalBlocker(self._unit_row_cb):
            self._unit_row_cb.setEnabled(sheet.first_row is not None)
            self._unit_row_cb.setChecked(sheet.unit_row is not None)
        self._refresh_unit_info()

    def _refresh_unit_info(self) -> None:
        sheet = self._current_sheet
        if sheet.settings.time_column is None:
            info = "No recorded time; observations can be fitted against a measured X quantity."
        else:
            try:
                unit = resolve_column_unit(sheet.settings.time_column, sheet.unit_row, sheet.settings, time=True)
                info = f"Time: {format_unit_for_display(unit)} → s. Each measured column uses its selected unit."
            except ValueError as exc:
                info = str(exc)
        set_bounded_label_text(self._unit_info_label, info, max_width=640)

    def _render_preview(self) -> None:
        sheet = self._current_sheet
        self._preview_table.clear()
        self._preview_table.setColumnCount(len(sheet.columns))
        self._preview_table.setRowCount(len(sheet.preview_rows))
        self._preview_table.setHorizontalHeaderLabels([col.assigned_name for col in sheet.columns])
        excluded = sheet.unit_row is not None
        self._preview_table.setVerticalHeaderLabels([
            "Units (excluded)" if index == 0 and excluded else f"Data {index + 1 - int(excluded)}"
            for index in range(len(sheet.preview_rows))
        ])
        for row_index, row in enumerate(sheet.preview_rows):
            for column_index, column in enumerate(sheet.columns):
                value = row.get(column)
                text = "" if value is None else str(value)
                item = QtWidgets.QTableWidgetItem(text)
                item.setToolTip(text)
                if row_index == 0 and excluded:
                    font = item.font()
                    font.setItalic(True)
                    item.setFont(font)
                    item.setToolTip(f"Units row — excluded from measurements: {text}")
                self._preview_table.setItem(row_index, column_index, item)
        self._preview_table.resizeColumnsToContents()
        for index in range(len(sheet.columns)):
            self._preview_table.setColumnWidth(index, min(300, self._preview_table.columnWidth(index)))

    def _render_species(self) -> None:
        sheet = self._current_sheet
        with QtCore.QSignalBlocker(self._species_table):
            columns = [col for col in sheet.columns if col != sheet.settings.time_column]
            self._species_table.setRowCount(len(columns))
            for row, column in enumerate(columns):
                check = QtWidgets.QTableWidgetItem()
                check.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsUserCheckable)
                check.setData(QtCore.Qt.UserRole, column.source_index)
                check.setCheckState(QtCore.Qt.Checked if column in sheet.settings.species_columns else QtCore.Qt.Unchecked)
                self._species_table.setItem(row, 0, check)
                label = QtWidgets.QTableWidgetItem(column.assigned_name)
                label.setToolTip(column.assigned_name)
                self._species_table.setItem(row, 1, label)
                combo = QtWidgets.QComboBox(self._species_table)
                combo.setEditable(True)
                combo.setInsertPolicy(QtWidgets.QComboBox.NoInsert)
                combo.setAccessibleName(f"Unit for {column.assigned_name}")
                self._populate_unit_choices(
                    combo, MEASUREMENT_UNIT_DISPLAY, sheet.settings.column_units.get(column, UnitSource.FILE),
                    default=sheet.settings.default_unit, declared=self._declared_unit(sheet, column),
                )
                combo.currentTextChanged.connect(
                    lambda _text, col=column, control=combo: self._on_column_unit_changed(col, control)
                )
                self._species_table.setCellWidget(row, 2, combo)
        self._refresh_conversions()

    def _refresh_conversions(self) -> None:
        sheet = self._current_sheet
        for row in range(self._species_table.rowCount()):
            source_index = self._species_table.item(row, 0).data(QtCore.Qt.UserRole)
            column = next(col for col in sheet.columns if col.source_index == source_index)
            label = QtWidgets.QLabel(self._species_table)
            label.setContentsMargins(4, 0, 4, 0)
            try:
                unit = resolve_column_unit(column, sheet.unit_row, sheet.settings)
                label.setTextFormat(QtCore.Qt.RichText)
                label.setText(f"{format_unit_for_display(unit)} → {format_unit_for_display(scientific_unit(unit.dimensions))}")
                label.setToolTip(f"Incoming {unit.symbol}; stored as {scientific_unit(unit.dimensions).symbol}.")
            except ValueError as exc:
                label.setText("Invalid unit")
                label.setToolTip(str(exc))
            self._species_table.setCellWidget(row, 3, label)
        self._species_count_label.setText(f"{len(sheet.settings.species_columns - {sheet.settings.time_column})} selected")

    def _on_time_column_changed(self) -> None:
        sheet = self._current_sheet
        source_index = self._time_combo.currentData()
        sheet.settings.time_column = next((col for col in sheet.columns if col.source_index == source_index), None)
        sheet.settings.time_choice = UnitSource.FILE
        self._render_units()
        self._render_species()
        self._update_import_enabled()

    def _on_time_unit_changed(self) -> None:
        self._current_sheet.settings.time_choice = self._unit_choice(self._time_unit_combo)
        self._refresh_unit_info()
        self._update_import_enabled()

    def _on_default_unit_changed(self) -> None:
        self._current_sheet.settings.default_unit = self._unit_choice(self._default_unit_combo)
        self._refresh_conversions()
        self._update_import_enabled()

    def _on_column_unit_changed(self, column, combo) -> None:
        self._current_sheet.settings.column_units[column] = self._unit_choice(combo)
        self._refresh_conversions()
        self._update_import_enabled()

    def _on_unit_row_toggled(self, checked: bool) -> None:
        self._current_sheet.settings.first_row_contains_units = checked
        self._render_units()
        self._render_species()
        self._render_preview()
        self._update_import_enabled()

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

    def _resolved_plans(self):
        current = self._current_sheet
        shared = sharing_settings(current.settings) if current is not None and self._apply_remaining_cb.isChecked() else None
        return tuple(
            resolve_sheet_plan(
                sheet.name, sheet.columns, sheet.first_row,
                sheet.settings if sheet is current or shared is None else shared,
            )
            for sheet in self._sheets if sheet.included
        )

    def _update_import_enabled(self) -> None:
        current = self._current_sheet
        try:
            plans = self._resolved_plans()
            reason = "" if plans else "Select at least one sheet to import."
        except ValueError as exc:
            reason = str(exc)
        self._btn_import.setEnabled(not reason)
        self._btn_import.setToolTip(reason)
        set_bounded_label_text(self._readiness_label, reason, max_width=1000)
        self._readiness_label.setVisible(bool(reason))
        source_ready = current is not None and bool(current.settings.species_columns - {current.settings.time_column})
        self._btn_select_compatible.setEnabled(self._apply_remaining_cb.isChecked() and source_ready)
        self._btn_select_all.setEnabled(bool(self._sheets))
        self._btn_invert_sheets.setEnabled(bool(self._sheets))

    def _on_import(self) -> None:
        try:
            plans = self._resolved_plans()
            if not plans:
                raise ValueError("Select at least one sheet to import.")
        except ValueError as exc:
            QtWidgets.QMessageBox.critical(self, "Import Configuration", str(exc))
            return
        shared = sharing_settings(self._current_sheet.settings) if self._apply_remaining_cb.isChecked() else None
        self._result = ImportDialogResult(ImportConfig(self._source, plans, shared), "import")
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
