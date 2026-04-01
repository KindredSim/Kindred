"""Tests for the import configuration dialog.

Covers auto-detection (time column, unit row), Excel sheet handling,
ImportConfig return values, dialog actions, validation, unit-row override,
preview content, and the apply-to-remaining checkbox.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List

import pytest
from PySide6 import QtCore, QtWidgets

from kindred.gui.widgets.import_config_dialog import ImportConfigDialog

pytestmark = pytest.mark.gui


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_csv(path, header: List[str], rows: List[List[str]]) -> str:
    """Write a CSV file and return its path as a string."""
    filepath = str(path)
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)
    return filepath


def _write_xlsx(path, sheets: dict) -> str:
    """Write an Excel workbook with {sheet_name: (header, rows)} and return path."""
    from openpyxl import Workbook

    filepath = str(path)
    wb = Workbook()
    first = True
    for sheet_name, (header, rows) in sheets.items():
        if first:
            ws = wb.active
            ws.title = sheet_name
            first = False
        else:
            ws = wb.create_sheet(title=sheet_name)
        ws.append(header)
        for row in rows:
            ws.append(row)
    wb.save(filepath)
    return filepath


# ---------------------------------------------------------------------------
# 1. CSV auto-detection
# ---------------------------------------------------------------------------

class TestCsvAutoDetection:
    """CSV time-column and unit-row auto-detection."""

    def test_time_column_detected(self, qapp, tmp_path):
        fp = _write_csv(tmp_path / "a.csv", ["time", "A", "B"], [
            ["0", "1.0", "2.0"],
            ["1", "1.1", "2.1"],
        ])
        dlg = ImportConfigDialog(fp)
        assert dlg._time_combo.currentText() == "time"

    def test_t_column_detected(self, qapp, tmp_path):
        fp = _write_csv(tmp_path / "b.csv", ["t", "X", "Y"], [
            ["0", "1.0", "2.0"],
        ])
        dlg = ImportConfigDialog(fp)
        assert dlg._time_combo.currentText() == "t"

    def test_no_standard_time_column(self, qapp, tmp_path):
        fp = _write_csv(tmp_path / "c.csv", ["elapsed", "A", "B"], [
            ["0", "1.0", "2.0"],
        ])
        dlg = ImportConfigDialog(fp)
        # No auto-detection: combo should show first column but no pre-selection
        # The import button should be disabled because no valid time column is confirmed
        # Actually per spec: "no column is pre-selected — user must pick manually"
        assert dlg._time_combo.currentText() == ""

    def test_unit_row_detected(self, qapp, tmp_path):
        fp = _write_csv(tmp_path / "d.csv", ["time", "A", "B"], [
            ["s", "uM", "uM"],
            ["0", "1.0", "2.0"],
            ["1", "1.1", "2.1"],
        ])
        dlg = ImportConfigDialog(fp)
        assert dlg._unit_row_detected is True
        assert dlg._time_unit_combo.currentText() == "s"
        assert dlg._conc_unit_combo.currentText() == "uM"

    def test_unicode_micro_units_normalize_to_ascii_combo_entries(self, qapp, tmp_path):
        fp = _write_csv(tmp_path / "unicode_units.csv", ["time", "A", "B"], [
            ["µs", "µM", "μM"],
            ["0", "1.0", "2.0"],
            ["1", "1.1", "2.1"],
        ])
        dlg = ImportConfigDialog(fp)
        assert dlg._time_unit_combo.currentText() == "us"
        assert dlg._conc_unit_combo.currentText() == "uM"

    def test_no_unit_row(self, qapp, tmp_path):
        fp = _write_csv(tmp_path / "e.csv", ["time", "A", "B"], [
            ["0", "1.0", "2.0"],
            ["1", "1.1", "2.1"],
        ])
        dlg = ImportConfigDialog(fp)
        assert dlg._unit_row_detected is False
        # Defaults
        assert dlg._time_unit_combo.currentText() == "s"
        assert dlg._conc_unit_combo.currentText() == "M"


# ---------------------------------------------------------------------------
# 2. Excel sheet handling
# ---------------------------------------------------------------------------

class TestExcelSheetHandling:
    """Excel multi-sheet listing, preview switching, and validation."""

    def test_multi_sheet_listed_all_checked(self, qapp, tmp_path):
        fp = _write_xlsx(tmp_path / "wb.xlsx", {
            "pH7": (["time", "A"], [["0", "1.0"], ["1", "0.9"]]),
            "pH9": (["time", "A"], [["0", "1.5"], ["1", "1.4"]]),
            "meta": (["key", "val"], [["temp", "298"]]),
        })
        dlg = ImportConfigDialog(fp)
        names = dlg._get_checked_sheet_names()
        assert set(names) == {"pH7", "pH9", "meta"}

    def test_sheet_selection_changes_preview(self, qapp, tmp_path):
        fp = _write_xlsx(tmp_path / "wb2.xlsx", {
            "Run1": (["time", "X"], [["0", "10"], ["1", "20"]]),
            "Run2": (["time", "Y"], [["0", "30"], ["1", "40"]]),
        })
        dlg = ImportConfigDialog(fp)
        # Click Run2 to show its preview
        for i in range(dlg._sheet_list.count()):
            item = dlg._sheet_list.item(i)
            if item.text() == "Run2":
                dlg._sheet_list.setCurrentItem(item)
                dlg._on_sheet_clicked(item)
                break
        # Preview table should now show Run2 columns
        headers = [
            dlg._preview_table.horizontalHeaderItem(c).text()
            for c in range(dlg._preview_table.columnCount())
        ]
        assert "Y" in headers

    def test_no_sheets_checked_disables_import(self, qapp, tmp_path):
        fp = _write_xlsx(tmp_path / "wb3.xlsx", {
            "Sheet1": (["time", "A"], [["0", "1"]]),
        })
        dlg = ImportConfigDialog(fp)
        # Uncheck the only sheet
        item = dlg._sheet_list.item(0)
        item.setCheckState(QtCore.Qt.CheckState.Unchecked)
        dlg._update_import_enabled()
        assert not dlg._btn_import.isEnabled()

    def test_incompatible_checked_sheets_are_blocked_on_import(self, qapp, tmp_path, monkeypatch):
        fp = _write_xlsx(tmp_path / "mismatch.xlsx", {
            "Run1": (["time", "A"], [["0", "1.0"]]),
            "Run2": (["time", "B", "C"], [["0", "2.0", "3.0"]]),
        })
        dlg = ImportConfigDialog(fp)
        warnings = []

        def _warning(parent, title, text):
            warnings.append((title, text))
            return QtWidgets.QMessageBox.StandardButton.Ok

        monkeypatch.setattr(QtWidgets.QMessageBox, "warning", _warning)

        dlg._on_import()

        assert warnings
        assert "different column structures" in warnings[0][1]
        assert dlg._result is None

    def test_checked_sheet_compatibility_uses_checked_set_not_previewed_sheet(self, qapp, tmp_path):
        fp = _write_xlsx(tmp_path / "compatible_checked.xlsx", {
            "Run1": (["time", "A"], [["0", "1.0"]]),
            "Run2": (["time", "B"], [["0", "2.0"]]),
            "Run3": (["time", "B"], [["0", "3.0"]]),
        })
        dlg = ImportConfigDialog(fp)

        for i in range(dlg._sheet_list.count()):
            item = dlg._sheet_list.item(i)
            if item.text() == "Run1":
                item.setCheckState(QtCore.Qt.CheckState.Unchecked)

        assert dlg._checked_excel_sheets_are_compatible() is True

    def test_reordered_checked_sheet_headers_are_compatible(self, qapp, tmp_path, monkeypatch):
        fp = _write_xlsx(tmp_path / "reordered.xlsx", {
            "Run1": (["time", "A", "B"], [["0", "1.0", "2.0"]]),
            "Run2": (["B", "time", "A"], [["3.0", "0", "4.0"]]),
        })
        dlg = ImportConfigDialog(fp)
        warnings = []

        def _warning(parent, title, text):
            warnings.append((title, text))
            return QtWidgets.QMessageBox.StandardButton.Ok

        monkeypatch.setattr(QtWidgets.QMessageBox, "warning", _warning)

        assert dlg._checked_excel_sheets_are_compatible() is True

        dlg._on_import()

        assert warnings == []
        assert dlg._result is not None

    def test_excel_preview_stops_after_preview_limit(self, qapp, tmp_path, monkeypatch):
        from kindred.core.datasets import excel_import
        from kindred.gui.widgets import import_config_dialog as dialog_module

        class _PreviewSheet:
            def __init__(self) -> None:
                self.data_rows_yielded = 0

            def iter_rows(self, values_only: bool = True):
                yield ("time", "A")
                for index in range(100):
                    self.data_rows_yielded += 1
                    if self.data_rows_yielded > dialog_module._MAX_PREVIEW_ROWS + 1:
                        raise AssertionError("preview consumed more rows than needed")
                    yield (index, index + 1.0)

        class _FakeWorkbook:
            def __init__(self) -> None:
                self.sheetnames = ["Data"]
                self.closed = 0
                self._sheet = _PreviewSheet()

            def __getitem__(self, name: str):
                assert name == "Data"
                return self._sheet

            def close(self) -> None:
                self.closed += 1

        workbooks: list[_FakeWorkbook] = []

        def _open_workbook(_path: str):
            workbook = _FakeWorkbook()
            workbooks.append(workbook)
            return workbook

        monkeypatch.setattr(excel_import, "_open_workbook", _open_workbook)

        dlg = ImportConfigDialog(str(tmp_path / "preview.xlsx"))

        preview_workbook = workbooks[-1]
        assert dlg._preview_table.rowCount() == dialog_module._MAX_PREVIEW_ROWS
        assert preview_workbook._sheet.data_rows_yielded == dialog_module._MAX_PREVIEW_ROWS + 1
        assert preview_workbook.closed == 1


# ---------------------------------------------------------------------------
# 3. ImportConfig returned values
# ---------------------------------------------------------------------------

class TestImportConfigValues:
    """Verify returned ImportConfig fields are correct."""

    def test_csv_config_fields(self, qapp, tmp_path):
        fp = _write_csv(tmp_path / "exp.csv", ["time", "Conc_A", "Conc_B"], [
            ["us", "uM", "uM"],
            ["0", "1.0", "2.0"],
            ["1", "0.9", "1.9"],
        ])
        dlg = ImportConfigDialog(fp)
        result = dlg._build_result("import")
        cfg = result.config
        assert cfg is not None
        assert cfg.filepath == fp
        assert cfg.file_type == "csv"
        assert cfg.sheet_names == []
        assert cfg.time_column == "time"
        assert cfg.species_columns == ["Conc_A", "Conc_B"]
        assert cfg.time_unit == "us"
        assert cfg.concentration_unit == "uM"
        assert cfg.unit_row_detected is True
        assert cfg.apply_to_remaining is False

    def test_excel_config_sheets_populated(self, qapp, tmp_path):
        fp = _write_xlsx(tmp_path / "k.xlsx", {
            "Run1": (["time", "A"], [["0", "1"]]),
            "Run2": (["time", "B"], [["0", "2"]]),
        })
        dlg = ImportConfigDialog(fp)
        result = dlg._build_result("import")
        cfg = result.config
        assert cfg.file_type == "excel"
        assert set(cfg.sheet_names) == {"Run1", "Run2"}

    def test_unit_row_detected_flag(self, qapp, tmp_path):
        fp = _write_csv(tmp_path / "no_units.csv", ["time", "A"], [
            ["0", "1.0"],
            ["1", "0.9"],
        ])
        dlg = ImportConfigDialog(fp)
        result = dlg._build_result("import")
        assert result.config.unit_row_detected is False

    def test_apply_to_remaining_reflects_checkbox(self, qapp, tmp_path):
        fp = _write_csv(tmp_path / "r.csv", ["time", "A"], [
            ["0", "1.0"],
        ])
        dlg = ImportConfigDialog(fp, remaining_count=3)
        dlg._apply_remaining_cb.setChecked(True)
        result = dlg._build_result("import")
        assert result.config.apply_to_remaining is True


# ---------------------------------------------------------------------------
# 4. Dialog actions
# ---------------------------------------------------------------------------

class TestDialogActions:
    """Import, Skip, Cancel return correct action strings."""

    def test_import_action(self, qapp, tmp_path):
        fp = _write_csv(tmp_path / "f.csv", ["time", "A"], [["0", "1"]])
        dlg = ImportConfigDialog(fp)
        result = dlg._build_result("import")
        assert result.action == "import"
        assert result.config is not None

    def test_skip_action(self, qapp, tmp_path):
        fp = _write_csv(tmp_path / "g.csv", ["time", "A"], [["0", "1"]])
        dlg = ImportConfigDialog(fp)
        result = dlg._build_result("skip")
        assert result.action == "skip"
        assert result.config is None

    def test_cancel_action(self, qapp, tmp_path):
        fp = _write_csv(tmp_path / "h.csv", ["time", "A"], [["0", "1"]])
        dlg = ImportConfigDialog(fp)
        result = dlg._build_result("cancel")
        assert result.action == "cancel"
        assert result.config is None


# ---------------------------------------------------------------------------
# 5. Validation
# ---------------------------------------------------------------------------

class TestValidation:
    """Import button disabled when required selections are missing."""

    def test_no_time_column_disables_import(self, qapp, tmp_path):
        fp = _write_csv(tmp_path / "v1.csv", ["elapsed", "A"], [["0", "1"]])
        dlg = ImportConfigDialog(fp)
        # No time column auto-detected, none selected
        assert not dlg._btn_import.isEnabled()

    def test_no_species_columns_disables_import(self, qapp, tmp_path):
        fp = _write_csv(tmp_path / "v2.csv", ["time", "A"], [["0", "1"]])
        dlg = ImportConfigDialog(fp)
        # Uncheck all species
        for cb in dlg._species_checkboxes:
            cb.setChecked(False)
        dlg._update_import_enabled()
        assert not dlg._btn_import.isEnabled()

    def test_excel_no_sheets_disables_import(self, qapp, tmp_path):
        fp = _write_xlsx(tmp_path / "v3.xlsx", {
            "S1": (["time", "A"], [["0", "1"]]),
        })
        dlg = ImportConfigDialog(fp)
        dlg._sheet_list.item(0).setCheckState(QtCore.Qt.CheckState.Unchecked)
        dlg._update_import_enabled()
        assert not dlg._btn_import.isEnabled()


# ---------------------------------------------------------------------------
# 6. Unit row override
# ---------------------------------------------------------------------------

class TestUnitRowOverride:
    """Checking 'No unit row' clears detection and defaults to s/M."""

    def test_override_clears_units(self, qapp, tmp_path):
        fp = _write_csv(tmp_path / "u.csv", ["time", "A"], [
            ["us", "uM"],
            ["0", "1.0"],
        ])
        dlg = ImportConfigDialog(fp)
        assert dlg._unit_row_detected is True
        # Override
        dlg._no_unit_row_cb.setChecked(True)
        result = dlg._build_result("import")
        assert result.config.time_unit == "s"
        assert result.config.concentration_unit == "M"
        assert result.config.unit_row_detected is False

    def test_mixed_concentration_units_show_warning_and_allow_override(self, qapp, tmp_path):
        fp = _write_csv(tmp_path / "mixed_units.csv", ["time", "A", "B"], [
            ["s", "uM", "nM"],
            ["0", "1.0", "2.0"],
        ])
        dlg = ImportConfigDialog(fp)
        assert "Multiple concentration units detected" in dlg._unit_warning_label.text()
        assert dlg._conc_unit_combo.currentText() == "uM"
        dlg._conc_unit_combo.setCurrentText("nM")
        result = dlg._build_result("import")
        assert result.config.concentration_unit == "nM"


# ---------------------------------------------------------------------------
# 7. Preview content
# ---------------------------------------------------------------------------

class TestPreviewContent:
    """Preview table shows correct rows, columns, and unit-row highlighting."""

    def test_preview_shows_rows(self, qapp, tmp_path):
        rows = [["time", "A"]] + [[str(i), str(float(i))] for i in range(5)]
        fp = _write_csv(tmp_path / "p.csv", rows[0], rows[1:])
        dlg = ImportConfigDialog(fp)
        assert dlg._preview_table.rowCount() == 5

    def test_preview_shows_columns(self, qapp, tmp_path):
        fp = _write_csv(tmp_path / "q.csv", ["time", "X", "Y"], [["0", "1", "2"]])
        dlg = ImportConfigDialog(fp)
        headers = [
            dlg._preview_table.horizontalHeaderItem(c).text()
            for c in range(dlg._preview_table.columnCount())
        ]
        assert headers == ["time", "X", "Y"]

    def test_unit_row_highlighted(self, qapp, tmp_path):
        fp = _write_csv(tmp_path / "hl.csv", ["time", "A"], [
            ["s", "uM"],
            ["0", "1.0"],
        ])
        dlg = ImportConfigDialog(fp)
        # Row 0 in the table is the unit row — should have a distinct background
        item = dlg._preview_table.item(0, 0)
        assert item is not None
        bg = item.background().color()
        # The unit-row background should differ from the default (white/invalid)
        default_bg = QtWidgets.QTableWidgetItem().background().color()
        assert bg != default_bg

    def test_preview_caps_at_20_rows(self, qapp, tmp_path):
        data_rows = [[str(i), str(float(i))] for i in range(30)]
        fp = _write_csv(tmp_path / "big.csv", ["time", "A"], data_rows)
        dlg = ImportConfigDialog(fp)
        # Should show at most ~20 data rows (unit row detection may eat one)
        assert dlg._preview_table.rowCount() <= 20


# ---------------------------------------------------------------------------
# 8. Apply to remaining
# ---------------------------------------------------------------------------

class TestApplyToRemaining:
    """Checkbox visibility and label reflect remaining_count."""

    def test_hidden_when_zero(self, qapp, tmp_path):
        fp = _write_csv(tmp_path / "z.csv", ["time", "A"], [["0", "1"]])
        dlg = ImportConfigDialog(fp, remaining_count=0)
        assert dlg._apply_remaining_cb.isHidden()

    def test_visible_with_count(self, qapp, tmp_path):
        fp = _write_csv(tmp_path / "z2.csv", ["time", "A"], [["0", "1"]])
        dlg = ImportConfigDialog(fp, remaining_count=4)
        assert not dlg._apply_remaining_cb.isHidden()
        assert "4" in dlg._apply_remaining_cb.text()

    def test_checkbox_state_in_config(self, qapp, tmp_path):
        fp = _write_csv(tmp_path / "z3.csv", ["time", "A"], [["0", "1"]])
        dlg = ImportConfigDialog(fp, remaining_count=2)
        dlg._apply_remaining_cb.setChecked(False)
        r1 = dlg._build_result("import")
        assert r1.config.apply_to_remaining is False
        dlg._apply_remaining_cb.setChecked(True)
        r2 = dlg._build_result("import")
        assert r2.config.apply_to_remaining is True


class TestErrorHandling:
    def test_legacy_xls_is_rejected(self, qapp, tmp_path):
        path = tmp_path / "legacy.xls"
        path.write_bytes(b"legacy-xls")

        with pytest.raises(ValueError, match="Legacy \\.xls format is not supported"):
            ImportConfigDialog(str(path))

    def test_latin1_csv_shows_encoding_error_and_allows_skip_cancel(self, qapp, tmp_path):
        path = Path(tmp_path) / "latin1.csv"
        path.write_bytes(b"time,A\n0,\xff\n")

        dlg = ImportConfigDialog(str(path))

        assert "encoding error" in dlg._preview_error_label.text().lower()
        assert "utf-8" in dlg._preview_error_label.text().lower()
        assert not dlg._btn_import.isEnabled()
        assert dlg._build_result("skip").action == "skip"
        assert dlg._build_result("cancel").action == "cancel"


# ---------------------------------------------------------------------------
# 9. Sheet column mismatch validation
# ---------------------------------------------------------------------------


class TestSheetColumnMismatch:
    """Regression: config built from previewed sheet columns, not checked sheets."""

    def test_sheet_column_mismatch_preview_vs_checked_single_sheet_rejects(
        self, qapp, tmp_path, monkeypatch,
    ):
        """Check sheet A, preview sheet B: species from B must be rejected."""
        fp = _write_xlsx(tmp_path / "mismatch.xlsx", {
            "SheetA": (["time", "A", "B"], [["0", "1", "2"]]),
            "SheetB": (["time", "X", "Y"], [["0", "3", "4"]]),
        })
        dlg = ImportConfigDialog(fp)

        # Uncheck SheetB, keep SheetA checked
        for i in range(dlg._sheet_list.count()):
            item = dlg._sheet_list.item(i)
            if item.text() == "SheetB":
                item.setCheckState(QtCore.Qt.CheckState.Unchecked)

        # Click SheetB to preview (species checkboxes switch to X, Y)
        for i in range(dlg._sheet_list.count()):
            item = dlg._sheet_list.item(i)
            if item.text() == "SheetB":
                dlg._sheet_list.setCurrentItem(item)
                dlg._on_sheet_clicked(item)
                break

        criticals: list[tuple] = []
        monkeypatch.setattr(
            QtWidgets.QMessageBox, "critical",
            lambda *args, **kwargs: criticals.append(args),
        )

        dlg._on_import()

        assert dlg._result is None
        assert criticals
        assert any("missing" in str(c).lower() for c in criticals)

    def test_sheet_column_mismatch_preview_vs_all_checked_sheets_rejects(
        self, qapp, tmp_path, monkeypatch,
    ):
        """Both sheets checked but incompatible: import must be rejected."""
        fp = _write_xlsx(tmp_path / "both_checked.xlsx", {
            "SheetA": (["time", "A", "B"], [["0", "1", "2"]]),
            "SheetB": (["time", "X", "Y"], [["0", "3", "4"]]),
        })
        dlg = ImportConfigDialog(fp)

        # Click SheetB to preview
        for i in range(dlg._sheet_list.count()):
            item = dlg._sheet_list.item(i)
            if item.text() == "SheetB":
                dlg._sheet_list.setCurrentItem(item)
                dlg._on_sheet_clicked(item)
                break

        warnings: list[tuple] = []
        criticals: list[tuple] = []
        monkeypatch.setattr(
            QtWidgets.QMessageBox, "warning",
            lambda *args, **kwargs: warnings.append(args),
        )
        monkeypatch.setattr(
            QtWidgets.QMessageBox, "critical",
            lambda *args, **kwargs: criticals.append(args),
        )

        dlg._on_import()

        assert dlg._result is None

    def test_sheet_column_mismatch_absent_when_checked_matches_preview(
        self, qapp, tmp_path, monkeypatch,
    ):
        """Check only sheet B and preview B: species X, Y are valid, import succeeds."""
        fp = _write_xlsx(tmp_path / "match.xlsx", {
            "SheetA": (["time", "A", "B"], [["0", "1", "2"]]),
            "SheetB": (["time", "X", "Y"], [["0", "3", "4"]]),
        })
        dlg = ImportConfigDialog(fp)

        # Uncheck SheetA
        for i in range(dlg._sheet_list.count()):
            item = dlg._sheet_list.item(i)
            if item.text() == "SheetA":
                item.setCheckState(QtCore.Qt.CheckState.Unchecked)

        # Click SheetB to preview
        for i in range(dlg._sheet_list.count()):
            item = dlg._sheet_list.item(i)
            if item.text() == "SheetB":
                dlg._sheet_list.setCurrentItem(item)
                dlg._on_sheet_clicked(item)
                break

        criticals: list[tuple] = []
        monkeypatch.setattr(
            QtWidgets.QMessageBox, "critical",
            lambda *args, **kwargs: criticals.append(args),
        )

        dlg._on_import()

        assert dlg._result is not None
        assert dlg._result.action == "import"
        assert set(dlg._result.config.species_columns) == {"X", "Y"}
        assert dlg._result.config.sheet_names == ["SheetB"]
        assert not criticals
