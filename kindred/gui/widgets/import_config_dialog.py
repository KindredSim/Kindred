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
    parse_concentration_unit,
)
from kindred.gui.widgets.import_config import (
    ImportConfig,
    SheetImportIntent,
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
        self._detected_conc_unit_by_column: Dict[str, Optional[str]] = {}
        self._species_checkboxes: List[QtWidgets.QCheckBox] = []
        self._sheet_states: Dict[Optional[str], dict] = {}

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
        unit_row.addWidget(QtWidgets.QLabel("Default conc unit:", self))
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
        if self._file_type == "excel" and self._remaining_count > 0:
            self._apply_remaining_cb.setText(
                "Apply these settings to all other sheets and remaining files"
            )
            self._apply_remaining_cb.setVisible(True)
        elif self._file_type == "excel":
            self._apply_remaining_cb.setText(
                "Apply these settings to all other sheets"
            )
            self._apply_remaining_cb.setVisible(True)
        elif self._remaining_count > 0:
            self._apply_remaining_cb.setText(
                "Apply these settings to remaining files"
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
        self._apply_remaining_cb.toggled.connect(lambda _: self._update_import_enabled())
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
        self._previewed_sheet_name = None
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
        self._save_current_sheet_state(None)

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

    def _read_excel_sheet_preview(self, sheet_name: str) -> tuple[list[str], list[list[str]]]:
        rows_raw: List[Dict[str, str]] = []
        columns: List[str] = []
        with closing(read_excel_sheet_rows(self._filepath, sheet_name)) as row_iter:
            for i, row_dict in enumerate(row_iter):
                if i == 0:
                    columns = list(row_dict.keys())
                if i >= _MAX_PREVIEW_ROWS:
                    break
                rows_raw.append(row_dict)
        preview_rows = [
            [str(row_dict.get(c, "")) for c in columns] for row_dict in rows_raw
        ]
        return columns, preview_rows

    def _build_default_sheet_state(
        self,
        columns: list[str],
        preview_rows: list[list[str]],
    ) -> dict:
        if preview_rows:
            row_mapping = dict(zip(columns, preview_rows[0]))
            det = detect_units_from_row_mapping(row_mapping)
        else:
            det = UnitDetection.empty()
        time_column = ""
        for candidate in _TIME_CANDIDATES:
            if candidate in columns:
                time_column = candidate
                break
        species_checked = {
            column_name: column_name != time_column
            for column_name in columns
        }
        detected_conc_by_col = {
            col: self._normalize_unit_for_combo(unit) if unit else None
            for col, unit in det.detected_conc_unit_by_column.items()
        }
        # Pick the most common detected unit as combo default, or "M"
        detected_units = [u for u in detected_conc_by_col.values() if u]
        combo_default = self._most_common_unit(detected_units) if detected_units else "M"
        return {
            "time_column": time_column,
            "species_checked": species_checked,
            "time_unit": self._normalize_unit_for_combo(det.detected_time_unit) if det.detected_time_unit else "s",
            "concentration_units": detected_conc_by_col,
            "combo_conc_unit": combo_default,
            "override_no_unit_row": False,
            "no_unit_row_cb_enabled": det.has_unit_row,
            "unit_row_detected": det.has_unit_row,
            "detected_time_unit": det.detected_time_unit,
            "detected_conc_unit_by_column": dict(detected_conc_by_col),
            "columns": list(columns),
            "preview_rows": [list(row) for row in preview_rows],
        }

    def _load_sheet_state_from_file(self, sheet_name: str) -> dict:
        columns, preview_rows = self._read_excel_sheet_preview(sheet_name)
        return self._build_default_sheet_state(columns, preview_rows)

    def _clone_sheet_state(self, state: dict) -> dict:
        return {
            "time_column": str(state.get("time_column", "")),
            "species_checked": dict(state.get("species_checked", {})),
            "time_unit": str(state.get("time_unit", "s")),
            "concentration_units": dict(state.get("concentration_units", {})),
            "combo_conc_unit": str(state.get("combo_conc_unit", "M")),
            "override_no_unit_row": bool(state.get("override_no_unit_row", False)),
            "no_unit_row_cb_enabled": bool(state.get("no_unit_row_cb_enabled", False)),
            "unit_row_detected": bool(state.get("unit_row_detected", False)),
            "detected_time_unit": state.get("detected_time_unit"),
            "detected_conc_unit_by_column": dict(state.get("detected_conc_unit_by_column", {})),
            "columns": [str(column) for column in state.get("columns", [])],
            "preview_rows": [list(row) for row in state.get("preview_rows", [])],
        }

    def _current_state_from_widgets(self) -> dict:
        combo_unit = self._conc_unit_combo.currentText()
        # Build per-column concentration_units from detection + combo default
        conc_units: Dict[str, str] = {}
        for cb in self._species_checkboxes:
            col = str(cb.property("column_name"))
            detected = self._detected_conc_unit_by_column.get(col)
            if self._unit_row_detected and not self._no_unit_row_cb.isChecked() and detected:
                conc_units[col] = detected
            else:
                conc_units[col] = combo_unit
        return {
            "time_column": self._time_combo.currentText(),
            "species_checked": {
                str(cb.property("column_name")): cb.isChecked()
                for cb in self._species_checkboxes
            },
            "time_unit": self._time_unit_combo.currentText(),
            "concentration_units": conc_units,
            "combo_conc_unit": combo_unit,
            "override_no_unit_row": self._no_unit_row_cb.isChecked(),
            "no_unit_row_cb_enabled": self._no_unit_row_cb.isEnabled(),
            "unit_row_detected": self._unit_row_detected,
            "detected_time_unit": self._detected_time_unit,
            "detected_conc_unit_by_column": dict(self._detected_conc_unit_by_column),
            "columns": list(self._columns),
            "preview_rows": [list(row) for row in self._preview_rows],
        }

    def _save_current_sheet_state(self, sheet_name: Optional[str] = None) -> Optional[dict]:
        key = sheet_name
        if key is None:
            key = None if self._file_type == "csv" else self._previewed_sheet_name
        if key is None and self._file_type == "excel":
            return None
        state = self._current_state_from_widgets()
        self._sheet_states[key] = self._clone_sheet_state(state)
        return self._sheet_states[key]

    def _merge_editable_sheet_state(self, base_state: dict, override_state: dict) -> dict:
        merged = self._clone_sheet_state(base_state)
        for key in (
            "time_column",
            "species_checked",
            "time_unit",
            "concentration_units",
            "combo_conc_unit",
            "override_no_unit_row",
        ):
            if key in override_state:
                value = override_state[key]
                if key in ("species_checked", "concentration_units"):
                    merged[key] = dict(value)
                else:
                    merged[key] = value
        if not bool(merged.get("no_unit_row_cb_enabled", False)):
            merged["override_no_unit_row"] = False
        return merged

    def _ensure_sheet_state(self, sheet_name: str) -> dict:
        state = self._sheet_states.get(sheet_name)
        if state is None:
            state = self._load_sheet_state_from_file(sheet_name)
            self._sheet_states[sheet_name] = self._clone_sheet_state(state)
            return self._sheet_states[sheet_name]
        if not state.get("columns") or "preview_rows" not in state:
            loaded_state = self._load_sheet_state_from_file(sheet_name)
            state = self._merge_editable_sheet_state(loaded_state, state)
            self._sheet_states[sheet_name] = self._clone_sheet_state(state)
        return self._sheet_states[sheet_name]

    def _load_excel_sheet_preview(self, sheet_name: str) -> None:
        state = self._load_sheet_state_from_file(sheet_name)
        self._sheet_states[sheet_name] = self._clone_sheet_state(state)
        self._restore_sheet_state(sheet_name, state)

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
        self._detected_conc_unit_by_column = {}
        if not self._preview_rows:
            return
        row_mapping = dict(zip(self._columns, self._preview_rows[0]))
        det = detect_units_from_row_mapping(row_mapping)
        self._unit_row_detected = det.has_unit_row
        self._detected_time_unit = det.detected_time_unit
        self._detected_conc_unit_by_column = {
            col: self._normalize_unit_for_combo(unit) if unit else None
            for col, unit in det.detected_conc_unit_by_column.items()
        }

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

    def _restore_sheet_state(self, sheet_name: Optional[str], state: dict) -> None:
        self._previewed_sheet_name = sheet_name
        self._time_combo.blockSignals(True)
        self._no_unit_row_cb.blockSignals(True)
        self._columns = [str(column) for column in state.get("columns", [])]
        self._preview_rows = [list(row) for row in state.get("preview_rows", [])]
        self._unit_row_detected = bool(state.get("unit_row_detected", False))
        self._detected_time_unit = state.get("detected_time_unit")
        self._detected_conc_unit_by_column = dict(state.get("detected_conc_unit_by_column", {}))
        self._populate_preview_table()
        self._populate_time_combo(selected_time=str(state.get("time_column", "")))
        self._time_unit_combo.setCurrentText(str(state.get("time_unit", "s")))
        self._conc_unit_combo.setCurrentText(str(state.get("combo_conc_unit", "M")))
        no_unit_row_enabled = bool(state.get("no_unit_row_cb_enabled", False))
        self._no_unit_row_cb.setEnabled(no_unit_row_enabled)
        self._no_unit_row_cb.setChecked(
            bool(state.get("override_no_unit_row", False)) and no_unit_row_enabled
        )
        self._refresh_unit_controls()
        self._populate_species_checkboxes()
        species_checked = dict(state.get("species_checked", {}))
        for cb in self._species_checkboxes:
            column_name = str(cb.property("column_name"))
            if column_name in species_checked:
                cb.setChecked(bool(species_checked[column_name]))
        self._time_combo.blockSignals(False)
        self._no_unit_row_cb.blockSignals(False)
        self._save_current_sheet_state(sheet_name)
        self._update_import_enabled()

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

    def _populate_time_combo(self, selected_time: Optional[str] = None) -> None:
        self._time_combo.blockSignals(True)
        self._time_combo.clear()
        self._time_combo.addItem("")  # empty sentinel for "not selected"
        for col in self._columns:
            self._time_combo.addItem(col)
        if selected_time is not None:
            self._time_combo.setCurrentText(selected_time)
        else:
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

    def _refresh_unit_controls(self) -> None:
        if self._unit_row_detected:
            self._unit_info_label.setText(
                f"Unit row detected (row 1): {', '.join(self._preview_rows[0])}"
            )
            distinct_units = set(
                u for u in self._detected_conc_unit_by_column.values() if u
            )
            if len(distinct_units) > 1 and not self._no_unit_row_cb.isChecked():
                self._unit_warning_label.setText(
                    "Different concentration units detected "
                    "\u2014 each column will be converted independently."
                )
                self._unit_warning_label.setStyleSheet("QLabel { color: #8a8; font-size: 11px; }")
                self._unit_warning_label.show()
            else:
                self._unit_warning_label.clear()
                self._unit_warning_label.hide()
        else:
            self._unit_info_label.setText("")
            self._unit_warning_label.clear()
            self._unit_warning_label.hide()

    def _populate_unit_controls(self) -> None:
        if self._unit_row_detected:
            if self._detected_time_unit:
                idx = self._time_unit_combo.findText(self._normalize_unit_for_combo(self._detected_time_unit))
                if idx >= 0:
                    self._time_unit_combo.setCurrentIndex(idx)
            else:
                self._time_unit_combo.setCurrentText("s")
            # Set combo to most common detected conc unit, or "M"
            detected_units = [u for u in self._detected_conc_unit_by_column.values() if u]
            if detected_units:
                idx = self._conc_unit_combo.findText(self._most_common_unit(detected_units))
                if idx >= 0:
                    self._conc_unit_combo.setCurrentIndex(idx)
            else:
                self._conc_unit_combo.setCurrentText("M")
            self._no_unit_row_cb.setChecked(False)
            self._no_unit_row_cb.setEnabled(True)
        else:
            self._time_unit_combo.setCurrentText("s")
            self._conc_unit_combo.setCurrentText("M")
            self._no_unit_row_cb.setChecked(False)
            self._no_unit_row_cb.setEnabled(False)
        self._refresh_unit_controls()

    def _populate_species_checkboxes(self) -> None:
        # Clear existing
        for cb in self._species_checkboxes:
            self._species_layout.removeWidget(cb)
            cb.deleteLater()
        self._species_checkboxes.clear()

        combo_unit = self._conc_unit_combo.currentText()
        time_col = self._time_combo.currentText()
        for col in self._columns:
            if col == time_col:
                continue
            label = col
            if self._unit_row_detected and not self._no_unit_row_cb.isChecked():
                detected = self._detected_conc_unit_by_column.get(col)
                unit_str = detected if detected else combo_unit
                try:
                    factor = parse_concentration_unit(unit_str)
                    if factor != 1.0:
                        if len(unit_str) > 1 and unit_str.startswith("u"):
                            display_unit = "\u00b5" + unit_str[1:]
                        else:
                            display_unit = unit_str
                        label = f"{col} ({display_unit} \u2192 M)"
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
            self._time_unit_combo.setCurrentText("s")
            self._conc_unit_combo.setCurrentText("M")
        elif self._unit_row_detected:
            if self._detected_time_unit:
                self._time_unit_combo.setCurrentText(self._normalize_unit_for_combo(self._detected_time_unit))
            detected_units = [u for u in self._detected_conc_unit_by_column.values() if u]
            if detected_units:
                self._conc_unit_combo.setCurrentText(self._most_common_unit(detected_units))
        self._refresh_unit_controls()
        self._populate_species_checkboxes()
        self._update_import_enabled()

    def _on_sheet_clicked(self, item: QtWidgets.QListWidgetItem) -> None:
        self._save_current_sheet_state()
        sheet_name = item.text()
        state = self._sheet_states.get(sheet_name)
        if state is None:
            self._load_excel_sheet_preview(sheet_name)
            return
        if not state.get("columns"):
            state = self._ensure_sheet_state(sheet_name)
        self._restore_sheet_state(sheet_name, state)

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
        self._save_current_sheet_state()
        current_state = self._current_state_from_widgets()
        time_ok = bool(current_state["time_column"]) and str(current_state["time_column"]) in {
            str(column) for column in current_state.get("columns", [])
        }
        species_ok = self._state_has_valid_species_selection(current_state)
        sheets_ok = True
        all_sheet_configs_ok = time_ok and species_ok
        incompatible_hint = ""
        if self._file_type == "excel":
            checked_sheet_names = self._get_checked_sheet_names()
            sheets_ok = bool(checked_sheet_names)
            # When the unified checkbox is checked, _build_result copies
            # the current sheet's intent to all other checked sheets.
            # Validate current sheet normally, then verify that every other
            # checked sheet has the required columns (time + species).
            if self._apply_remaining_cb.isChecked():
                all_sheet_configs_ok = sheets_ok and time_ok and species_ok
                if all_sheet_configs_ok:
                    required_time = str(current_state["time_column"])
                    required_species = {
                        str(col) for col, checked
                        in dict(current_state.get("species_checked", {})).items()
                        if checked
                    }
                    required_columns = {required_time} | required_species
                    for sheet_name in checked_sheet_names:
                        if sheet_name == self._previewed_sheet_name:
                            continue
                        other_state = self._ensure_sheet_state(sheet_name)
                        other_columns = {
                            str(c) for c in other_state.get("columns", [])
                        }
                        missing = required_columns - other_columns
                        if missing:
                            all_sheet_configs_ok = False
                            incompatible_hint = (
                                f"{sheet_name} is missing columns: "
                                f"{', '.join(sorted(missing))}"
                            )
                            break
            else:
                all_sheet_configs_ok = sheets_ok
                for sheet_name in checked_sheet_names:
                    state = current_state if sheet_name == self._previewed_sheet_name else self._ensure_sheet_state(sheet_name)
                    sheet_time_ok = bool(state["time_column"]) and str(state["time_column"]) in {
                        str(column) for column in state.get("columns", [])
                    }
                    sheet_species_ok = self._state_has_valid_species_selection(state)
                    if not (sheet_time_ok and sheet_species_ok):
                        all_sheet_configs_ok = False
                        break

        enabled = sheets_ok and all_sheet_configs_ok
        self._btn_import.setEnabled(enabled)

        # Hints
        if not time_ok:
            self._time_hint.setText("Select a time column")
        else:
            self._time_hint.setText("")

        if not species_ok:
            self._species_hint.setText("Select at least one species column")
        elif incompatible_hint:
            self._species_hint.setText(incompatible_hint)
        else:
            self._species_hint.setText("")

    # ------------------------------------------------------------------
    # Result building
    # ------------------------------------------------------------------

    def _build_result(self, action: str) -> ImportDialogResult:
        if action in ("skip", "cancel"):
            return ImportDialogResult(config=None, action=action)
        self._save_current_sheet_state()

        sheet_names: list[str] = self._get_checked_sheet_names() if self._file_type == "excel" else []

        if self._apply_remaining_cb.isChecked() and self._file_type == "excel":
            current_state = self._sheet_states.get(self._previewed_sheet_name)
            if current_state is not None:
                for sheet_name in sheet_names:
                    if sheet_name == self._previewed_sheet_name:
                        continue
                    target_state = self._ensure_sheet_state(sheet_name)
                    merged = self._merge_editable_sheet_state(target_state, current_state)
                    self._sheet_states[sheet_name] = self._clone_sheet_state(merged)

        file_intent = UserImportIntent(
            sheet_names=tuple(sheet_names),
            apply_to_remaining=self._apply_remaining_cb.isChecked(),
        )

        per_sheet_intents: dict[str | None, SheetImportIntent] = {}
        per_sheet_detections: dict[str | None, UnitDetection] = {}
        per_sheet_columns: dict[str | None, list[str]] = {}

        target_keys: list[Optional[str]] = sheet_names if self._file_type == "excel" else [None]
        for key in target_keys:
            state = (
                self._ensure_sheet_state(key)
                if self._file_type == "excel" and key is not None
                else self._sheet_states.get(None, self._current_state_from_widgets())
            )
            time_col = str(state.get("time_column", ""))
            species = tuple(
                column_name
                for column_name, checked in dict(state.get("species_checked", {})).items()
                if checked
            )
            override_no_units = bool(
                state.get("override_no_unit_row", False)
                and state.get("no_unit_row_cb_enabled", False)
            )
            # Build per-column concentration_units from state
            conc_units_dict = dict(state.get("concentration_units", {}))
            combo_default = str(state.get("combo_conc_unit", "M")) or "M"
            # Ensure every species column has an entry
            intent_conc_units = {}
            for sp in species:
                intent_conc_units[sp] = conc_units_dict.get(sp, combo_default)
            per_sheet_intents[key] = SheetImportIntent(
                time_column=time_col,
                species_columns=species,
                time_unit=str(state.get("time_unit", "s")) or "s",
                concentration_units=intent_conc_units,
                override_no_unit_row=override_no_units,
            )
            columns = [str(column) for column in state.get("columns", [])]
            per_sheet_columns[key] = columns
            preview_rows = [list(row) for row in state.get("preview_rows", [])]
            if preview_rows:
                row_mapping = dict(zip(columns, preview_rows[0]))
                relevant_cols = [time_col, *species]
                per_sheet_detections[key] = detect_units_from_row_mapping(row_mapping, relevant_cols)
            else:
                per_sheet_detections[key] = UnitDetection.empty()

        try:
            plans = resolve_import_plans(
                self._filepath,
                self._file_type,
                per_sheet_intents,
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
            file_intent=file_intent,
            per_sheet_intents=tuple(
                (sheet_name, per_sheet_intents[sheet_name])
                for sheet_name in target_keys
            ),
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

    @staticmethod
    def _most_common_unit(units: list[str]) -> str:
        """Return the most common unit string, breaking ties alphabetically."""
        counts: dict[str, int] = {}
        for u in units:
            counts[u] = counts.get(u, 0) + 1
        return min(counts, key=lambda u: (-counts[u], u))

    @staticmethod
    def _state_has_valid_species_selection(state: dict) -> bool:
        columns = {str(column) for column in state.get("columns", [])}
        time_column = str(state.get("time_column", ""))
        selected_species = [
            str(column_name)
            for column_name, checked in dict(state.get("species_checked", {})).items()
            if checked
        ]
        if not selected_species:
            return False
        return all(
            column_name in columns and column_name != time_column
            for column_name in selected_species
        )

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
