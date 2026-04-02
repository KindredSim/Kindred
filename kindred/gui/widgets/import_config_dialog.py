"""Import configuration dialog for dataset import pipeline.

Previews a data file (CSV or Excel), auto-detects time column and unit row,
lets the user configure column selection and units, and returns a configuration
object.  The dialog does NOT load full datasets or convert units — it returns
configuration only.
"""

from __future__ import annotations

import csv
import os
from contextlib import closing
from dataclasses import dataclass
from typing import Dict, List, Optional

from PySide6 import QtCore, QtGui, QtWidgets

from kindred.core.datasets.excel_import import list_sheets, read_excel_sheet_rows
from kindred.core.datasets.units import (
    CONCENTRATION_UNIT_DISPLAY,
    TIME_UNIT_DISPLAY,
    looks_like_unit_row,
    parse_unit,
)
from kindred.gui.widgets.import_config import (
    ImportConfig,
    UnitDetection,
    UserImportIntent,
    detect_units_from_row_mapping,
    resolve_import_plans,
)

__all__ = ["ImportConfig", "ImportConfigDialog", "ImportDialogResult"]

_TIME_CANDIDATES = ["time", "time_s", "t", "Time", "T", "x"]
_MAX_PREVIEW_ROWS = 20
_UNIT_ROW_BG = QtGui.QColor(80, 65, 20)  # dark muted amber for dark-theme readability


@dataclass
class ImportDialogResult:
    config: Optional[ImportConfig] = None
    action: str = "cancel"


class ImportConfigDialog(QtWidgets.QDialog):
    """Modal dialog that previews a data file and returns an ImportConfig."""

    def __init__(
        self,
        filepath: str,
        remaining_count: int = 0,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._filepath = filepath
        self._remaining_count = remaining_count
        lower_path = filepath.lower()
        if lower_path.endswith(".xls") and not lower_path.endswith(".xlsx"):
            raise ValueError("Legacy .xls format is not supported. Please save as .xlsx.")
        self._file_type = "excel" if lower_path.endswith(".xlsx") else "csv"
        self._result: Optional[ImportDialogResult] = None

        # State populated during preview reading
        self._columns: List[str] = []
        self._preview_rows: List[List[str]] = []
        self._unit_row_detected: bool = False
        self._detected_time_unit: Optional[str] = None
        self._detected_conc_unit: Optional[str] = None
        self._detected_conc_units: List[str] = []
        self._species_checkboxes: List[QtWidgets.QCheckBox] = []

        # Excel-specific
        self._sheet_names: List[str] = []
        self._previewed_sheet_name: Optional[str] = None

        self.setWindowTitle(f"Import: {os.path.basename(filepath)}")
        self.setModal(True)
        self.resize(700, 550)

        self._build_ui()
        self._load_preview()
        self._update_import_enabled()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        # --- Sheet selector (Excel only) ---
        self._sheet_section = QtWidgets.QWidget(self)
        sheet_lay = QtWidgets.QVBoxLayout(self._sheet_section)
        sheet_lay.setContentsMargins(0, 0, 0, 0)
        sheet_header = QtWidgets.QLabel("<b>Sheets</b>", self)
        sheet_lay.addWidget(sheet_header)
        self._sheet_list = QtWidgets.QListWidget(self)
        self._sheet_list.setMaximumHeight(90)
        sheet_lay.addWidget(self._sheet_list)
        layout.addWidget(self._sheet_section)
        self._sheet_section.setVisible(self._file_type == "excel")

        # --- Preview table ---
        preview_header = QtWidgets.QLabel("<b>Preview</b>", self)
        layout.addWidget(preview_header)
        self._preview_table = QtWidgets.QTableWidget(self)
        self._preview_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self._preview_table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        layout.addWidget(self._preview_table, 1)
        self._preview_error_label = QtWidgets.QLabel("", self)
        self._preview_error_label.setStyleSheet("QLabel { color: #c00; font-size: 11px; }")
        self._preview_error_label.hide()
        layout.addWidget(self._preview_error_label)

        # --- Time column ---
        time_row = QtWidgets.QHBoxLayout()
        time_row.addWidget(QtWidgets.QLabel("Time column:", self))
        self._time_combo = QtWidgets.QComboBox(self)
        self._time_combo.setMinimumWidth(120)
        time_row.addWidget(self._time_combo)
        time_row.addStretch(1)
        self._time_hint = QtWidgets.QLabel("", self)
        self._time_hint.setStyleSheet("QLabel { color: #c00; font-size: 11px; }")
        time_row.addWidget(self._time_hint)
        layout.addLayout(time_row)

        # --- Unit controls ---
        self._unit_info_label = QtWidgets.QLabel("", self)
        layout.addWidget(self._unit_info_label)
        self._unit_warning_label = QtWidgets.QLabel("", self)
        self._unit_warning_label.setStyleSheet("QLabel { color: #c60; font-size: 11px; }")
        self._unit_warning_label.hide()
        layout.addWidget(self._unit_warning_label)

        unit_row = QtWidgets.QHBoxLayout()
        unit_row.addWidget(QtWidgets.QLabel("Time unit:", self))
        self._time_unit_combo = QtWidgets.QComboBox(self)
        for u in TIME_UNIT_DISPLAY:
            self._time_unit_combo.addItem(u)
        self._time_unit_combo.setMaximumWidth(90)
        unit_row.addWidget(self._time_unit_combo)
        unit_row.addSpacing(16)
        unit_row.addWidget(QtWidgets.QLabel("Conc unit:", self))
        self._conc_unit_combo = QtWidgets.QComboBox(self)
        for u in CONCENTRATION_UNIT_DISPLAY:
            self._conc_unit_combo.addItem(u)
        self._conc_unit_combo.setMaximumWidth(90)
        unit_row.addWidget(self._conc_unit_combo)
        unit_row.addStretch(1)
        layout.addLayout(unit_row)

        self._no_unit_row_cb = QtWidgets.QCheckBox("No unit row (assume s and M)", self)
        self._no_unit_row_cb.toggled.connect(self._on_no_unit_row_toggled)
        layout.addWidget(self._no_unit_row_cb)

        # --- Species columns ---
        species_header = QtWidgets.QLabel("<b>Species columns</b>", self)
        layout.addWidget(species_header)
        self._species_container = QtWidgets.QWidget(self)
        self._species_layout = QtWidgets.QVBoxLayout(self._species_container)
        self._species_layout.setContentsMargins(0, 0, 0, 0)
        self._species_layout.setSpacing(2)
        species_scroll = QtWidgets.QScrollArea(self)
        species_scroll.setWidgetResizable(True)
        species_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        species_scroll.setWidget(self._species_container)
        species_scroll.setMaximumHeight(120)
        layout.addWidget(species_scroll)
        self._species_hint = QtWidgets.QLabel("", self)
        self._species_hint.setStyleSheet("QLabel { color: #c00; font-size: 11px; }")
        layout.addWidget(self._species_hint)

        # --- Apply to remaining ---
        self._apply_remaining_cb = QtWidgets.QCheckBox("", self)
        if self._remaining_count > 0:
            self._apply_remaining_cb.setText(
                f"Apply these settings to remaining {self._remaining_count} files"
            )
            self._apply_remaining_cb.setVisible(True)
        else:
            self._apply_remaining_cb.setVisible(False)
        layout.addWidget(self._apply_remaining_cb)

        # --- Buttons ---
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addStretch(1)
        self._btn_import = QtWidgets.QPushButton("Import", self)
        self._btn_skip = QtWidgets.QPushButton("Skip", self)
        self._btn_cancel = QtWidgets.QPushButton("Cancel All", self)
        btn_row.addWidget(self._btn_import)
        btn_row.addWidget(self._btn_skip)
        btn_row.addWidget(self._btn_cancel)
        layout.addLayout(btn_row)

        # --- Wiring ---
        self._btn_import.clicked.connect(self._on_import)
        self._btn_skip.clicked.connect(self._on_skip)
        self._btn_cancel.clicked.connect(self._on_cancel)
        self._time_combo.currentTextChanged.connect(self._on_time_column_changed)
        if self._file_type == "excel":
            self._sheet_list.itemClicked.connect(self._on_sheet_clicked)
            self._sheet_list.itemChanged.connect(lambda _: self._update_import_enabled())

    # ------------------------------------------------------------------
    # Preview loading
    # ------------------------------------------------------------------

    def _load_preview(self) -> None:
        if self._file_type == "excel":
            self._load_excel_preview()
        else:
            self._load_csv_preview()

    def _load_csv_preview(self) -> None:
        self._clear_preview_error()
        try:
            with open(self._filepath, "r", encoding="utf-8-sig", newline="") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if header is None:
                    self._columns = []
                    self._preview_rows = []
                    self._detect_and_populate()
                    return
                self._columns = [h.strip() for h in header]
                rows: List[List[str]] = []
                for _, row in zip(range(_MAX_PREVIEW_ROWS), reader):
                    rows.append([c.strip() if c else "" for c in row])
                self._preview_rows = rows
        except UnicodeError:
            self._columns = []
            self._preview_rows = []
            self._detect_and_populate()
            self._set_preview_error("Cannot read file: encoding error. Expected UTF-8.")
            return
        self._detect_and_populate()

    def _load_excel_preview(self) -> None:
        self._sheet_names = list_sheets(self._filepath)
        self._sheet_list.blockSignals(True)
        self._sheet_list.clear()
        for name in self._sheet_names:
            item = QtWidgets.QListWidgetItem(name)
            item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.CheckState.Checked)
            self._sheet_list.addItem(item)
        self._sheet_list.blockSignals(False)
        if self._sheet_names:
            self._sheet_list.setCurrentRow(0)
            self._load_excel_sheet_preview(self._sheet_names[0])

    def _load_excel_sheet_preview(self, sheet_name: str) -> None:
        self._previewed_sheet_name = sheet_name
        rows_raw: List[Dict[str, str]] = []
        columns: List[str] = []
        with closing(read_excel_sheet_rows(self._filepath, sheet_name)) as row_iter:
            for i, row_dict in enumerate(row_iter):
                if i == 0:
                    columns = list(row_dict.keys())
                if i >= _MAX_PREVIEW_ROWS:
                    break
                rows_raw.append(row_dict)
        self._columns = columns
        self._preview_rows = [
            [str(row_dict.get(c, "")) for c in columns] for row_dict in rows_raw
        ]
        self._detect_and_populate()

    # ------------------------------------------------------------------
    # Detection and population
    # ------------------------------------------------------------------

    def _detect_and_populate(self) -> None:
        self._detect_unit_row()
        self._populate_preview_table()
        self._populate_time_combo()
        self._populate_unit_controls()
        self._populate_species_checkboxes()
        self._update_import_enabled()

    def _detect_unit_row(self) -> None:
        self._unit_row_detected = False
        self._detected_time_unit = None
        self._detected_conc_unit = None
        self._detected_conc_units = []
        if not self._preview_rows:
            return
        row_mapping = dict(zip(self._columns, self._preview_rows[0]))
        det = detect_units_from_row_mapping(row_mapping)
        self._unit_row_detected = det.has_unit_row
        self._detected_time_unit = det.detected_time_unit
        self._detected_conc_unit = det.detected_conc_unit
        self._detected_conc_units = [
            self._normalize_unit_for_combo(u) for u in det.detected_conc_units
        ]

    def _detect_units_for_sheet(
        self,
        sheet_name: str,
        relevant_columns: list[str],
    ) -> tuple[UnitDetection, list[str]]:
        """Return (UnitDetection, column_names) for a specific sheet."""
        if sheet_name == self._previewed_sheet_name:
            row_mapping = dict(zip(self._columns, self._preview_rows[0])) if self._preview_rows else {}
            det = detect_units_from_row_mapping(row_mapping, relevant_columns)
            return det, list(self._columns)
        with closing(read_excel_sheet_rows(self._filepath, sheet_name)) as rows:
            first_row = next(iter(rows), None)
        if first_row is None:
            return UnitDetection.empty(), []
        columns = list(first_row.keys())
        det = detect_units_from_row_mapping(dict(first_row), relevant_columns)
        return det, columns

    def _populate_preview_table(self) -> None:
        self._preview_table.clear()
        if not self._columns:
            return
        display_rows = self._preview_rows
        self._preview_table.setColumnCount(len(self._columns))
        self._preview_table.setHorizontalHeaderLabels(self._columns)
        self._preview_table.setRowCount(len(display_rows))
        for r, row in enumerate(display_rows):
            for c, val in enumerate(row):
                item = QtWidgets.QTableWidgetItem(val)
                if self._unit_row_detected and r == 0:
                    item.setBackground(QtGui.QBrush(_UNIT_ROW_BG))
                self._preview_table.setItem(r, c, item)
        self._preview_table.resizeColumnsToContents()

    def _populate_time_combo(self) -> None:
        self._time_combo.blockSignals(True)
        self._time_combo.clear()
        self._time_combo.addItem("")  # empty sentinel for "not selected"
        for col in self._columns:
            self._time_combo.addItem(col)
        # Auto-detect
        detected = None
        for candidate in _TIME_CANDIDATES:
            if candidate in self._columns:
                detected = candidate
                break
        if detected:
            self._time_combo.setCurrentText(detected)
        else:
            self._time_combo.setCurrentIndex(0)  # empty
        self._time_combo.blockSignals(False)

    def _populate_unit_controls(self) -> None:
        if self._unit_row_detected:
            self._unit_info_label.setText(
                f"Unit row detected (row 1): {', '.join(self._preview_rows[0])}"
            )
            if self._detected_time_unit:
                idx = self._time_unit_combo.findText(self._normalize_unit_for_combo(self._detected_time_unit))
                if idx >= 0:
                    self._time_unit_combo.setCurrentIndex(idx)
            if self._detected_conc_unit:
                idx = self._conc_unit_combo.findText(self._normalize_unit_for_combo(self._detected_conc_unit))
                if idx >= 0:
                    self._conc_unit_combo.setCurrentIndex(idx)
            self._no_unit_row_cb.setChecked(False)
            self._no_unit_row_cb.setEnabled(True)
            if len(self._detected_conc_units) > 1:
                self._unit_warning_label.setText(
                    "Multiple concentration units detected "
                    f"({', '.join(self._detected_conc_units)}). Verify unit selection."
                )
                self._unit_warning_label.show()
            else:
                self._unit_warning_label.clear()
                self._unit_warning_label.hide()
        else:
            self._unit_info_label.setText("")
            # Defaults: s, M
            self._time_unit_combo.setCurrentText("s")
            self._conc_unit_combo.setCurrentText("M")
            self._no_unit_row_cb.setChecked(True)
            self._no_unit_row_cb.setEnabled(False)
            self._unit_warning_label.clear()
            self._unit_warning_label.hide()

    def _populate_species_checkboxes(self) -> None:
        # Clear existing
        for cb in self._species_checkboxes:
            self._species_layout.removeWidget(cb)
            cb.deleteLater()
        self._species_checkboxes.clear()

        time_col = self._time_combo.currentText()
        for col in self._columns:
            if col == time_col:
                continue
            label = col
            if self._unit_row_detected and not self._no_unit_row_cb.isChecked():
                col_idx = self._columns.index(col)
                if col_idx < len(self._preview_rows[0]):
                    unit_str = self._preview_rows[0][col_idx].strip()
                    if unit_str:
                        try:
                            category, _f = parse_unit(unit_str)
                            if category == "concentration":
                                label = f"{col} ({unit_str} -> M)"
                        except ValueError:
                            pass
            cb = QtWidgets.QCheckBox(label, self._species_container)
            cb.setChecked(True)
            cb.setProperty("column_name", col)
            cb.toggled.connect(lambda _: self._update_import_enabled())
            self._species_checkboxes.append(cb)
            self._species_layout.addWidget(cb)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_time_column_changed(self, _text: str) -> None:
        self._populate_species_checkboxes()
        self._update_import_enabled()

    def _on_no_unit_row_toggled(self, checked: bool) -> None:
        if checked:
            self._unit_row_detected = False
            self._time_unit_combo.setCurrentText("s")
            self._conc_unit_combo.setCurrentText("M")
            self._unit_info_label.setText("")
            self._unit_warning_label.clear()
            self._unit_warning_label.hide()
        else:
            # Re-detect
            if self._preview_rows:
                first_row = self._preview_rows[0]
                if looks_like_unit_row(first_row):
                    self._unit_row_detected = True
                    self._populate_unit_controls()
        self._populate_species_checkboxes()

    def _on_sheet_clicked(self, item: QtWidgets.QListWidgetItem) -> None:
        sheet_name = item.text()
        self._load_excel_sheet_preview(sheet_name)

    def _on_import(self) -> None:
        result = self._build_result("import")
        if result.config is None:
            return
        self._result = result
        self.accept()

    def _on_skip(self) -> None:
        self._result = self._build_result("skip")
        self.reject()

    def _on_cancel(self) -> None:
        self._result = self._build_result("cancel")
        self.reject()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _update_import_enabled(self) -> None:
        time_ok = bool(self._time_combo.currentText())
        species_ok = any(cb.isChecked() for cb in self._species_checkboxes)
        sheets_ok = True
        if self._file_type == "excel":
            sheets_ok = bool(self._get_checked_sheet_names())

        enabled = time_ok and species_ok and sheets_ok
        self._btn_import.setEnabled(enabled)

        # Hints
        if not time_ok:
            self._time_hint.setText("Select a time column")
        else:
            self._time_hint.setText("")

        if not species_ok:
            self._species_hint.setText("Select at least one species column")
        else:
            self._species_hint.setText("")

    # ------------------------------------------------------------------
    # Result building
    # ------------------------------------------------------------------

    def _build_result(self, action: str) -> ImportDialogResult:
        if action in ("skip", "cancel"):
            return ImportDialogResult(config=None, action=action)

        override_no_units = (
            self._no_unit_row_cb.isChecked() and self._no_unit_row_cb.isEnabled()
        )

        time_col = self._time_combo.currentText()
        species = [
            cb.property("column_name")
            for cb in self._species_checkboxes
            if cb.isChecked()
        ]
        sheet_names: list[str] = []
        if self._file_type == "excel":
            sheet_names = self._get_checked_sheet_names()

        # Full-row detection for ImportConfig.detection (informational)
        if self._preview_rows:
            full_row_mapping = dict(zip(self._columns, self._preview_rows[0]))
            full_detection = detect_units_from_row_mapping(full_row_mapping)
        else:
            full_detection = UnitDetection.empty()

        if override_no_units:
            chosen_time = "s"
            chosen_conc = "M"
        elif self._unit_row_detected:
            chosen_time = self._time_unit_combo.currentText() or "s"
            chosen_conc = self._conc_unit_combo.currentText() or "M"
        else:
            chosen_time = self._time_unit_combo.currentText() or "s"
            chosen_conc = self._conc_unit_combo.currentText() or "M"

        intent = UserImportIntent(
            time_column=time_col,
            species_columns=tuple(species),
            time_unit=chosen_time,
            concentration_unit=chosen_conc,
            override_no_unit_row=override_no_units,
            sheet_names=tuple(sheet_names),
            apply_to_remaining=self._apply_remaining_cb.isChecked(),
        )

        # Build per-sheet detections scoped to relevant columns
        relevant_cols = [time_col] + species
        per_sheet_detections: dict[str | None, UnitDetection] = {}
        per_sheet_columns: dict[str | None, list[str]] = {}

        if self._file_type == "excel":
            for sn in sheet_names:
                det, cols = self._detect_units_for_sheet(sn, relevant_cols)
                per_sheet_detections[sn] = det
                per_sheet_columns[sn] = cols
        else:
            if self._preview_rows:
                row_mapping = dict(zip(self._columns, self._preview_rows[0]))
                det = detect_units_from_row_mapping(row_mapping, relevant_cols)
            else:
                det = UnitDetection.empty()
            per_sheet_detections[None] = det
            per_sheet_columns[None] = list(self._columns)

        try:
            plans = resolve_import_plans(
                self._filepath,
                self._file_type,
                intent,
                per_sheet_detections,
                per_sheet_columns,
            )
        except ValueError as exc:
            QtWidgets.QMessageBox.critical(
                self, "Import Configuration", str(exc),
            )
            return ImportDialogResult(config=None, action=action)

        config = ImportConfig(
            filepath=self._filepath,
            file_type=self._file_type,
            detection=full_detection,
            intent=intent,
            plans=tuple(plans),
        )
        return ImportDialogResult(config=config, action=action)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_checked_sheet_names(self) -> List[str]:
        names: List[str] = []
        for i in range(self._sheet_list.count()):
            item = self._sheet_list.item(i)
            if item.checkState() == QtCore.Qt.CheckState.Checked:
                names.append(item.text())
        return names

    @staticmethod
    def _normalize_unit_for_combo(unit: Optional[str]) -> str:
        return str(unit or "").strip().replace("µ", "u").replace("μ", "u")

    def _set_preview_error(self, message: str) -> None:
        self._preview_error_label.setText(str(message))
        self._preview_error_label.show()

    def _clear_preview_error(self) -> None:
        self._preview_error_label.clear()
        self._preview_error_label.hide()

    def get_result(self) -> ImportDialogResult:
        """Return the dialog result after exec()."""
        if self._result is None:
            return ImportDialogResult(config=None, action="cancel")
        return self._result
