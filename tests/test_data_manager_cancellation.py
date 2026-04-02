from __future__ import annotations

from pathlib import Path
import time

import pytest
from openpyxl import Workbook
from PySide6 import QtWidgets
from PySide6.QtTest import QSignalSpy

from kindred.core.datasets.csv_import import CsvImportInterrupted
from kindred.gui.widgets.data_manager import DataManagerPanel
from kindred.gui.widgets.import_config import (
    ImportConfig,
    ResolvedSheetPlan,
    SheetImportIntent,
    UserImportIntent,
)
from kindred.gui.widgets.import_config_dialog import ImportDialogResult

pytestmark = pytest.mark.gui


def _write_csv(path: Path, rows: int = 20000) -> None:
    """Create a simple CSV file with predictable numeric data."""
    with path.open("w", encoding="utf-8") as handle:
        handle.write("time,A,B\n")
        for i in range(rows):
            handle.write(f"{i},{i * 0.1},{i * 0.2}\n")


def _write_workbook(path: Path) -> None:
    workbook = Workbook()
    sheet_a = workbook.active
    sheet_a.title = "SheetA"
    sheet_a.append(["time", "A"])
    sheet_a.append([0.0, 1.0])
    sheet_b = workbook.create_sheet(title="SheetB")
    sheet_b.append(["time", "A"])
    sheet_b.append([0.0, 2.0])
    workbook.save(path)


def _make_test_config(
    filepath: str,
    file_type: str = "csv",
    sheet_names: list[str] | None = None,
    time_column: str = "time",
    species_columns: list[str] | None = None,
    time_unit: str = "s",
    concentration_unit: str = "M",
    unit_row_detected: bool = False,
    apply_to_remaining: bool = False,
    override_no_unit_row: bool = False,
) -> ImportConfig:
    """Build a test ImportConfig with file intent, sheet intents, and resolved plans."""
    from kindred.core.datasets.units import parse_concentration_unit, parse_time_unit

    species = tuple(species_columns or [])
    sheets = tuple(sheet_names or [])
    conc_units = {col: concentration_unit for col in species}

    file_intent = UserImportIntent(
        sheet_names=sheets,
        apply_to_remaining=apply_to_remaining,
    )
    sheet_intent = SheetImportIntent(
        time_column=time_column,
        species_columns=species,
        time_unit=time_unit,
        concentration_units=conc_units,
        override_no_unit_row=override_no_unit_row,
    )

    if override_no_unit_row:
        t_factor = 1.0
        c_factors = {col: 1.0 for col in species}
        t_orig = "s"
        c_origs = {col: "M" for col in species}
    else:
        t_factor = parse_time_unit(time_unit) if time_unit else 1.0
        c_factor = parse_concentration_unit(concentration_unit) if concentration_unit else 1.0
        c_factors = {col: c_factor for col in species}
        t_orig = time_unit or "s"
        c_orig = concentration_unit or "M"
        c_origs = {col: c_orig for col in species}

    if file_type == "excel":
        per_sheet_intents = tuple((sheet_name, sheet_intent) for sheet_name in sheets)
        plans = tuple(
            ResolvedSheetPlan(
                filepath=filepath,
                sheet_name=s,
                time_column=time_column,
                species_columns=species,
                skip_unit_row=unit_row_detected,
                time_factor=t_factor,
                conc_factors=dict(c_factors),
                original_time_unit=t_orig,
                original_conc_units=dict(c_origs),
            )
            for s in sheets
        )
    else:
        per_sheet_intents = ((None, sheet_intent),)
        plans = (
            ResolvedSheetPlan(
                filepath=filepath,
                sheet_name=None,
                time_column=time_column,
                species_columns=species,
                skip_unit_row=unit_row_detected,
                time_factor=t_factor,
                conc_factors=dict(c_factors),
                original_time_unit=t_orig,
                original_conc_units=dict(c_origs),
            ),
        )

    return ImportConfig(
        filepath=filepath,
        file_type=file_type,
        file_intent=file_intent,
        per_sheet_intents=per_sheet_intents,
        plans=plans,
    )


def test_multi_file_import_cancel_cleans_workers(tmp_path, monkeypatch, qtbot):
    """Cancel multi-file import and ensure threads/workers clean up properly."""
    panel = DataManagerPanel()
    qtbot.addWidget(panel)

    files = []
    for idx in range(2):
        csv_path = tmp_path / f"dataset_{idx}.csv"
        _write_csv(csv_path, rows=20000)
        files.append(str(csv_path))

    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getOpenFileNames",
        lambda *args, **kwargs: (files, ""),
    )
    queued_results = [
        ImportDialogResult(
            config=_make_test_config(
                filepath=files[0],
                file_type="csv",
                time_column="time",
                species_columns=["A", "B"],
                time_unit="s",
                concentration_unit="M",
                apply_to_remaining=True,
            ),
            action="import",
        )
    ]

    class _FakeDialog:
        def __init__(self, filepath: str, remaining_count: int = 0, parent=None) -> None:
            self._result = queued_results.pop(0)

        def exec(self) -> int:
            return int(QtWidgets.QDialog.DialogCode.Accepted)

        def get_result(self) -> ImportDialogResult:
            return self._result

    monkeypatch.setattr("kindred.gui.widgets.data_manager.ImportConfigDialog", _FakeDialog)

    finished_spy = QSignalSpy(panel.loadFinished)

    panel._load_dataset()
    panel._on_load_canceled()

    qtbot.waitUntil(lambda: finished_spy.count() == 1, timeout=7000)

    assert finished_spy.count() == 1
    assert bool(finished_spy.at(0)[0]) is True
    assert not panel._csv_workers
    assert panel._progress_dialog is None
    assert panel._pending_files_count == 0
    assert panel._completed_files_count == 0


def test_excel_import_cancel_cleans_workers(tmp_path, monkeypatch, qtbot):
    """Cancel Excel import and ensure worker bookkeeping cleans up properly."""
    panel = DataManagerPanel()
    qtbot.addWidget(panel)

    workbook_path = tmp_path / "dataset.xlsx"
    _write_workbook(workbook_path)

    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getOpenFileNames",
        lambda *args, **kwargs: ([str(workbook_path)], ""),
    )
    queued_results = [
        ImportDialogResult(
            config=_make_test_config(
                filepath=str(workbook_path),
                file_type="excel",
                sheet_names=["SheetA", "SheetB"],
                time_column="time",
                species_columns=["A"],
                time_unit="s",
                concentration_unit="M",
            ),
            action="import",
        )
    ]

    class _FakeDialog:
        def __init__(self, filepath: str, remaining_count: int = 0, parent=None) -> None:
            self._result = queued_results.pop(0)

        def exec(self) -> int:
            return int(QtWidgets.QDialog.DialogCode.Accepted)

        def get_result(self) -> ImportDialogResult:
            return self._result

    def _slow_excel_loader(self, plan):
        sheet_name = plan.sheet_name if hasattr(plan, "sheet_name") else str(plan)
        for _ in range(200):
            if self.isInterruptionRequested():
                raise CsvImportInterrupted()
            time.sleep(0.001)
        return f"{Path(self.filepath).name}::{sheet_name}", {"t": [], "species": {}, "metadata": {}}

    monkeypatch.setattr("kindred.gui.widgets.data_manager.ImportConfigDialog", _FakeDialog)
    monkeypatch.setattr("kindred.gui.widgets.data_manager.ExcelLoaderWorker._load_sheet_payload", _slow_excel_loader)

    finished_spy = QSignalSpy(panel.loadFinished)

    panel._load_dataset()
    panel._on_load_canceled()

    qtbot.waitUntil(lambda: finished_spy.count() == 1, timeout=7000)

    assert finished_spy.count() == 1
    assert bool(finished_spy.at(0)[0]) is True
    assert not panel._csv_workers
    assert panel._progress_dialog is None
    assert panel._pending_files_count == 0
    assert panel._completed_files_count == 0


def test_clear_datasets_cancels_inflight_import_and_prevents_late_commit(tmp_path, monkeypatch, qtbot):
    """Clearing datasets should discard active imports instead of letting them repopulate the panel."""
    panel = DataManagerPanel()
    qtbot.addWidget(panel)

    csv_path = tmp_path / "late.csv"
    _write_csv(csv_path, rows=10)

    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getOpenFileNames",
        lambda *args, **kwargs: ([str(csv_path)], ""),
    )
    queued_results = [
        ImportDialogResult(
            config=_make_test_config(
                filepath=str(csv_path),
                file_type="csv",
                time_column="time",
                species_columns=["A", "B"],
                time_unit="s",
                concentration_unit="M",
            ),
            action="import",
        )
    ]

    class _FakeDialog:
        def __init__(self, filepath: str, remaining_count: int = 0, parent=None) -> None:
            self._result = queued_results.pop(0)

        def exec(self) -> int:
            return int(QtWidgets.QDialog.DialogCode.Accepted)

        def get_result(self) -> ImportDialogResult:
            return self._result

    def _slow_payload(self):
        for _ in range(100):
            if self.isInterruptionRequested():
                raise CsvImportInterrupted()
            time.sleep(0.002)
        return {"t": [0.0, 1.0], "species": {"A": [1.0, 2.0], "B": [2.0, 3.0]}, "metadata": {}}

    monkeypatch.setattr("kindred.gui.widgets.data_manager.ImportConfigDialog", _FakeDialog)
    monkeypatch.setattr("kindred.gui.widgets.data_manager.CSVLoaderWorker._load_csv_payload", _slow_payload)

    panel._load_dataset()
    qtbot.wait(20)
    panel.clear_datasets()
    qtbot.wait(400)

    assert panel.get_datasets() == {}
    assert panel._progress_dialog is None
