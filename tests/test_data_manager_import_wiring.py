from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from openpyxl import Workbook
from PySide6 import QtWidgets
from PySide6.QtTest import QSignalSpy

from kindred.gui.widgets.import_config import (
    ImportConfig,
    ResolvedSheetPlan,
    SheetImportIntent,
    UserImportIntent,
)
from kindred.gui.widgets.import_config_dialog import ImportDialogResult

pytestmark = pytest.mark.gui


def _write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    lines = [",".join(header)]
    for row in rows:
        lines.append(",".join(str(value) for value in row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_workbook(path: Path, sheets: dict[str, tuple[list[str], list[list[object]]]]) -> None:
    workbook = Workbook()
    default_sheet = workbook.active
    workbook.remove(default_sheet)
    for sheet_name, (header, rows) in sheets.items():
        sheet = workbook.create_sheet(title=sheet_name)
        sheet.append(header)
        for row in rows:
            sheet.append(list(row))
    workbook.save(path)


def _patch_dialog_sequence(monkeypatch, results: list[ImportDialogResult]):
    from kindred.gui.widgets import data_manager as data_manager_module

    created: list[tuple[str, int]] = []
    queued = list(results)

    class _FakeDialog:
        def __init__(self, filepath: str, remaining_count: int = 0, parent=None) -> None:
            assert queued, "dialog queue exhausted"
            self._result = queued.pop(0)
            created.append((str(filepath), int(remaining_count)))

        def exec(self) -> int:
            if self._result.action == "import":
                return int(QtWidgets.QDialog.DialogCode.Accepted)
            return int(QtWidgets.QDialog.DialogCode.Rejected)

        def get_result(self) -> ImportDialogResult:
            return self._result

    monkeypatch.setattr(data_manager_module, "ImportConfigDialog", _FakeDialog)
    return created


def _wait_for_load(panel, qtbot, expected_count: int) -> list[object]:
    finished_spy = QSignalSpy(panel.loadFinished)
    qtbot.waitUntil(lambda: finished_spy.count() == 1, timeout=7000)
    assert finished_spy.count() == 1
    assert bool(finished_spy.at(0)[0]) is False
    assert len(panel.get_datasets()) == expected_count
    return [panel.get_datasets(), finished_spy]


def _capture_worker_rows(monkeypatch) -> dict[str, list[dict[str, str]]]:
    from kindred.gui.widgets import data_manager as data_manager_module

    captured: dict[str, list[dict[str, str]]] = {}

    def _fake_parse_csv_rows(rows, **_kwargs):
        materialized = [dict(row) for row in rows]
        captured["rows"] = materialized
        return "time", {
            "t": np.arange(len(materialized), dtype=float),
            "species": {"A": np.arange(len(materialized), dtype=float)},
            "metadata": {},
        }

    monkeypatch.setattr(data_manager_module, "parse_csv_rows", _fake_parse_csv_rows)
    return captured


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

    file_intent = UserImportIntent(
        sheet_names=sheets,
        apply_to_remaining=apply_to_remaining,
    )
    sheet_intent = SheetImportIntent(
        time_column=time_column,
        species_columns=species,
        time_unit=time_unit,
        concentration_unit=concentration_unit,
        override_no_unit_row=override_no_unit_row,
    )

    if override_no_unit_row:
        t_factor = 1.0
        c_factor = 1.0
        t_orig = "s"
        c_orig = "M"
    else:
        t_factor = parse_time_unit(time_unit) if time_unit else 1.0
        c_factor = parse_concentration_unit(concentration_unit) if concentration_unit else 1.0
        t_orig = time_unit or "s"
        c_orig = concentration_unit or "M"

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
                conc_factor=c_factor,
                original_time_unit=t_orig,
                original_conc_unit=c_orig,
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
                conc_factor=c_factor,
                original_time_unit=t_orig,
                original_conc_unit=c_orig,
            ),
        )

    return ImportConfig(
        filepath=filepath,
        file_type=file_type,
        file_intent=file_intent,
        per_sheet_intents=per_sheet_intents,
        plans=plans,
    )


def test_file_dialog_filter_includes_xlsx(monkeypatch, qtbot):
    from kindred.gui.widgets.data_manager import DataManagerPanel

    panel = DataManagerPanel()
    qtbot.addWidget(panel)

    captured: dict[str, str] = {}

    def _fake_get_open_file_names(*args, **kwargs):
        captured["filter"] = str(args[3])
        return ([], "")

    monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileNames", _fake_get_open_file_names)

    panel._load_dataset()

    assert "*.xlsx" in captured["filter"]


def test_csv_import_through_config_dialog(tmp_path, monkeypatch, qtbot):
    from kindred.gui.widgets.data_manager import DataManagerPanel

    csv_path = tmp_path / "converted.csv"
    _write_csv(
        csv_path,
        ["time_ms", "A_uM", "B_uM"],
        [[1000, 2.0, 5.0], [2000, 4.0, 6.0]],
    )

    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getOpenFileNames",
        lambda *args, **kwargs: ([str(csv_path)], ""),
    )
    _patch_dialog_sequence(
        monkeypatch,
        [
            ImportDialogResult(
                config=_make_test_config(
                    filepath=str(csv_path),
                    file_type="csv",
                    time_column="time_ms",
                    species_columns=["A_uM", "B_uM"],
                    time_unit="ms",
                    concentration_unit="uM",
                ),
                action="import",
            )
        ],
    )

    panel = DataManagerPanel()
    qtbot.addWidget(panel)
    panel._load_dataset()
    datasets, _finished_spy = _wait_for_load(panel, qtbot, expected_count=1)

    assert "converted.csv" in datasets
    payload = datasets["converted.csv"]
    assert np.allclose(payload["t"], [1.0, 2.0])
    assert np.allclose(payload["species"]["A_uM"], [2.0e-6, 4.0e-6])
    assert payload["metadata"]["original_time_unit"] == "ms"
    assert payload["metadata"]["original_concentration_unit"] == "uM"


def test_csv_import_trims_whitespace_padded_headers(tmp_path, monkeypatch, qtbot):
    from kindred.gui.widgets.data_manager import DataManagerPanel

    csv_path = tmp_path / "padded_headers.csv"
    _write_csv(
        csv_path,
        [" time ", " A "],
        [[0.0, 1.0], [1.0, 2.0]],
    )

    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getOpenFileNames",
        lambda *args, **kwargs: ([str(csv_path)], ""),
    )
    _patch_dialog_sequence(
        monkeypatch,
        [
            ImportDialogResult(
                config=_make_test_config(
                    filepath=str(csv_path),
                    file_type="csv",
                    time_column="time",
                    species_columns=["A"],
                    time_unit="s",
                    concentration_unit="M",
                ),
                action="import",
            )
        ],
    )
    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", lambda *args, **kwargs: QtWidgets.QMessageBox.StandardButton.Ok)

    panel = DataManagerPanel()
    qtbot.addWidget(panel)
    panel._load_dataset()
    datasets, _finished_spy = _wait_for_load(panel, qtbot, expected_count=1)

    payload = datasets["padded_headers.csv"]
    assert np.allclose(payload["t"], [0.0, 1.0])
    assert np.allclose(payload["species"]["A"], [1.0, 2.0])


def test_csv_import_with_detected_unit_row(tmp_path, monkeypatch, qtbot):
    from kindred.gui.widgets.data_manager import DataManagerPanel

    csv_path = tmp_path / "units_row.csv"
    _write_csv(
        csv_path,
        ["time", "A"],
        [["ms", "uM"], [1000, 2.0], [2000, 4.0]],
    )

    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getOpenFileNames",
        lambda *args, **kwargs: ([str(csv_path)], ""),
    )
    _patch_dialog_sequence(
        monkeypatch,
        [
            ImportDialogResult(
                config=_make_test_config(
                    filepath=str(csv_path),
                    file_type="csv",
                    time_column="time",
                    species_columns=["A"],
                    time_unit="ms",
                    concentration_unit="uM",
                    unit_row_detected=True,
                ),
                action="import",
            )
        ],
    )
    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", lambda *args, **kwargs: QtWidgets.QMessageBox.StandardButton.Ok)

    panel = DataManagerPanel()
    qtbot.addWidget(panel)
    panel._load_dataset()
    datasets, _finished_spy = _wait_for_load(panel, qtbot, expected_count=1)

    payload = datasets["units_row.csv"]
    assert np.allclose(payload["t"], [1.0, 2.0])
    assert np.allclose(payload["species"]["A"], [2.0e-6, 4.0e-6])
    assert payload["metadata"]["original_time_unit"] == "ms"
    assert payload["metadata"]["original_concentration_unit"] == "uM"


def test_csv_unit_row_override_false_preserves_first_row(tmp_path, monkeypatch):
    from kindred.gui.widgets.data_manager import CSVLoaderWorker

    csv_path = tmp_path / "override.csv"
    _write_csv(
        csv_path,
        ["time", "A", "B"],
        [["s", "M", "M"], [1.0, 2.0, 3.0], [2.0, 4.0, 6.0]],
    )
    captured = _capture_worker_rows(monkeypatch)

    plan = ResolvedSheetPlan(
        filepath=str(csv_path), sheet_name=None,
        time_column="time", species_columns=("A", "B"),
        skip_unit_row=False,
        time_factor=1.0, conc_factor=1.0,
        original_time_unit="s", original_conc_unit="M",
    )
    worker = CSVLoaderWorker(plan)

    worker._load_csv_payload()

    assert [row["time"] for row in captured["rows"]] == ["s", "1.0", "2.0"]
    assert [row["A"] for row in captured["rows"]] == ["M", "2.0", "4.0"]
    assert [row["B"] for row in captured["rows"]] == ["M", "3.0", "6.0"]


def test_excel_unit_row_override_false_preserves_first_row(tmp_path, monkeypatch):
    from kindred.gui.widgets.data_manager import ExcelLoaderWorker

    workbook_path = tmp_path / "override.xlsx"
    _write_workbook(
        workbook_path,
        {
            "Data": (
                ["time", "A", "B"],
                [["s", "M", "M"], [1.0, 2.0, 3.0], [2.0, 4.0, 6.0]],
            )
        },
    )
    captured = _capture_worker_rows(monkeypatch)
    plan = ResolvedSheetPlan(
        filepath=str(workbook_path), sheet_name="Data",
        time_column="time", species_columns=("A", "B"),
        skip_unit_row=False,
        time_factor=1.0, conc_factor=1.0,
        original_time_unit="s", original_conc_unit="M",
    )

    worker = ExcelLoaderWorker(str(workbook_path), [plan])
    _name, _data = worker._load_sheet_payload(plan)

    assert [row["time"] for row in captured["rows"]] == ["s", "1", "2"]
    assert [row["A"] for row in captured["rows"]] == ["M", "2", "4"]
    assert [row["B"] for row in captured["rows"]] == ["M", "3", "6"]


def test_csv_unit_row_authoritative_strips_with_blank_selected_columns(tmp_path, monkeypatch):
    """Dialog's unit_row_detected=True is authoritative.  Strip the unit row
    even when the *selected* columns have blank unit cells (the dialog used ALL
    columns to detect, including unselected ones with recognizable units)."""
    from kindred.gui.widgets.data_manager import CSVLoaderWorker

    csv_path = tmp_path / "blank_selected_units.csv"
    _write_csv(
        csv_path,
        ["time", "A", "note1", "note2"],
        [["", "", "ms", "uM"], [0.0, 1.0, "", ""], [1.0, 2.0, "", ""]],
    )
    captured = _capture_worker_rows(monkeypatch)

    plan = ResolvedSheetPlan(
        filepath=str(csv_path), sheet_name=None,
        time_column="time", species_columns=("A",),
        skip_unit_row=True,
        time_factor=1.0, conc_factor=1.0,
        original_time_unit="s", original_conc_unit="M",
    )
    worker = CSVLoaderWorker(plan)

    worker._load_csv_payload()

    assert [row["time"] for row in captured["rows"]] == ["0.0", "1.0"]
    assert [row["A"] for row in captured["rows"]] == ["1.0", "2.0"]


def test_excel_unit_row_authoritative_strips_with_blank_selected_columns(tmp_path, monkeypatch):
    """Same as CSV variant but for Excel sheets."""
    from kindred.gui.widgets.data_manager import ExcelLoaderWorker

    workbook_path = tmp_path / "blank_selected_units.xlsx"
    _write_workbook(
        workbook_path,
        {
            "Data": (
                ["time", "A", "note1", "note2"],
                [["", "", "ms", "uM"], [0.0, 1.0, "", ""], [1.0, 2.0, "", ""]],
            )
        },
    )
    captured = _capture_worker_rows(monkeypatch)
    plan = ResolvedSheetPlan(
        filepath=str(workbook_path), sheet_name="Data",
        time_column="time", species_columns=("A",),
        skip_unit_row=True,
        time_factor=1.0, conc_factor=1.0,
        original_time_unit="s", original_conc_unit="M",
    )

    worker = ExcelLoaderWorker(str(workbook_path), [plan])
    _name, _data = worker._load_sheet_payload(plan)

    assert [row["time"] for row in captured["rows"]] == ["0", "1"]
    assert [row["A"] for row in captured["rows"]] == ["1", "2"]


def test_csv_unit_row_override_true_still_strips_first_row(tmp_path, monkeypatch):
    from kindred.gui.widgets.data_manager import CSVLoaderWorker

    csv_path = tmp_path / "detected_units.csv"
    _write_csv(
        csv_path,
        ["time", "A", "B"],
        [["s", "M", "M"], [1.0, 2.0, 3.0], [2.0, 4.0, 6.0]],
    )
    captured = _capture_worker_rows(monkeypatch)

    plan = ResolvedSheetPlan(
        filepath=str(csv_path), sheet_name=None,
        time_column="time", species_columns=("A", "B"),
        skip_unit_row=True,
        time_factor=1.0, conc_factor=1.0,
        original_time_unit="s", original_conc_unit="M",
    )
    worker = CSVLoaderWorker(plan)

    worker._load_csv_payload()

    assert [row["time"] for row in captured["rows"]] == ["1.0", "2.0"]
    assert [row["A"] for row in captured["rows"]] == ["2.0", "4.0"]
    assert [row["B"] for row in captured["rows"]] == ["3.0", "6.0"]


def test_excel_unit_row_override_true_still_strips_first_row(tmp_path, monkeypatch):
    from kindred.gui.widgets.data_manager import ExcelLoaderWorker

    workbook_path = tmp_path / "detected_units.xlsx"
    _write_workbook(
        workbook_path,
        {
            "Data": (
                ["time", "A", "B"],
                [["s", "M", "M"], [1.0, 2.0, 3.0], [2.0, 4.0, 6.0]],
            )
        },
    )
    captured = _capture_worker_rows(monkeypatch)
    plan = ResolvedSheetPlan(
        filepath=str(workbook_path), sheet_name="Data",
        time_column="time", species_columns=("A", "B"),
        skip_unit_row=True,
        time_factor=1.0, conc_factor=1.0,
        original_time_unit="s", original_conc_unit="M",
    )

    worker = ExcelLoaderWorker(str(workbook_path), [plan])
    _name, _data = worker._load_sheet_payload(plan)

    assert [row["time"] for row in captured["rows"]] == ["1", "2"]
    assert [row["A"] for row in captured["rows"]] == ["2", "4"]
    assert [row["B"] for row in captured["rows"]] == ["3", "6"]


def test_csv_import_rejects_mixed_detected_concentration_units(tmp_path, monkeypatch, qtbot):
    from kindred.gui.widgets.data_manager import DataManagerPanel

    csv_path = tmp_path / "mixed_units.csv"
    _write_csv(
        csv_path,
        ["time", "A", "B"],
        [["ms", "uM", "nM"], [1000, 2.0, 5.0], [2000, 4.0, 6.0]],
    )

    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getOpenFileNames",
        lambda *args, **kwargs: ([str(csv_path)], ""),
    )
    _patch_dialog_sequence(
        monkeypatch,
        [
            ImportDialogResult(config=None, action="import"),
        ],
    )

    panel = DataManagerPanel()
    qtbot.addWidget(panel)
    panel._load_dataset()

    assert panel.get_datasets() == {}


def test_csv_import_ignores_unselected_mixed_unit_columns(tmp_path, monkeypatch, qtbot):
    from kindred.gui.widgets.data_manager import DataManagerPanel

    csv_path = tmp_path / "mixed_units_subset.csv"
    _write_csv(
        csv_path,
        ["time", "A", "B"],
        [["ms", "uM", "nM"], [1000, 2.0, 5.0], [2000, 4.0, 6.0]],
    )

    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getOpenFileNames",
        lambda *args, **kwargs: ([str(csv_path)], ""),
    )
    _patch_dialog_sequence(
        monkeypatch,
        [
            ImportDialogResult(
                config=_make_test_config(
                    filepath=str(csv_path),
                    file_type="csv",
                    time_column="time",
                    species_columns=["A"],
                    time_unit="ms",
                    concentration_unit="uM",
                    unit_row_detected=True,
                ),
                action="import",
            )
        ],
    )
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *args, **kwargs: QtWidgets.QMessageBox.StandardButton.Ok)

    panel = DataManagerPanel()
    qtbot.addWidget(panel)
    panel._load_dataset()
    datasets, _finished_spy = _wait_for_load(panel, qtbot, expected_count=1)

    payload = datasets["mixed_units_subset.csv"]
    assert list(payload["species"]) == ["A"]
    assert np.allclose(payload["species"]["A"], [2.0e-6, 4.0e-6])


def test_csv_import_does_not_strip_row_when_only_unselected_columns_look_like_units(tmp_path, monkeypatch, qtbot):
    from kindred.gui.widgets.data_manager import DataManagerPanel

    csv_path = tmp_path / "notes.csv"
    _write_csv(
        csv_path,
        ["time", "A", "note1", "note2"],
        [[0.0, 1.0, "ms", "uM"], [1.0, 2.0, "", ""]],
    )

    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getOpenFileNames",
        lambda *args, **kwargs: ([str(csv_path)], ""),
    )
    _patch_dialog_sequence(
        monkeypatch,
        [
            ImportDialogResult(
                config=_make_test_config(
                    filepath=str(csv_path),
                    file_type="csv",
                    time_column="time",
                    species_columns=["A"],
                    time_unit="s",
                    concentration_unit="M",
                    unit_row_detected=False,
                ),
                action="import",
            )
        ],
    )

    panel = DataManagerPanel()
    qtbot.addWidget(panel)
    panel._load_dataset()
    datasets, _finished_spy = _wait_for_load(panel, qtbot, expected_count=1)

    payload = datasets["notes.csv"]
    assert np.allclose(payload["t"], [0.0, 1.0])
    assert np.allclose(payload["species"]["A"], [1.0, 2.0])


def test_excel_import_through_config_dialog(tmp_path, monkeypatch, qtbot):
    from kindred.gui.widgets.data_manager import DataManagerPanel

    workbook_path = tmp_path / "multi.xlsx"
    _write_workbook(
        workbook_path,
        {
            "SheetA": (["time", "A", "B"], [[0.0, 1.0, 2.0], [1.0, 3.0, 4.0]]),
            "SheetB": (["time", "A", "B"], [[0.0, 5.0, 6.0], [1.0, 7.0, 8.0]]),
        },
    )

    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getOpenFileNames",
        lambda *args, **kwargs: ([str(workbook_path)], ""),
    )
    _patch_dialog_sequence(
        monkeypatch,
        [
            ImportDialogResult(
                config=_make_test_config(
                    filepath=str(workbook_path),
                    file_type="excel",
                    sheet_names=["SheetA", "SheetB"],
                    time_column="time",
                    species_columns=["A", "B"],
                    time_unit="s",
                    concentration_unit="M",
                ),
                action="import",
            )
        ],
    )

    panel = DataManagerPanel()
    qtbot.addWidget(panel)
    panel._load_dataset()
    datasets, _finished_spy = _wait_for_load(panel, qtbot, expected_count=2)

    assert set(datasets) == {"multi.xlsx::SheetA", "multi.xlsx::SheetB"}


def test_excel_import_rejects_multi_sheet_unit_mismatch(tmp_path, monkeypatch, qtbot):
    from kindred.gui.widgets.data_manager import DataManagerPanel

    workbook_path = tmp_path / "mismatch.xlsx"
    _write_workbook(
        workbook_path,
        {
            "SheetA": (["time", "A"], [["ms", "uM"], [0.0, 1.0], [1.0, 2.0]]),
            "SheetB": (["time", "A"], [["s", "M"], [0.0, 3.0], [1.0, 4.0]]),
        },
    )

    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getOpenFileNames",
        lambda *args, **kwargs: ([str(workbook_path)], ""),
    )
    _patch_dialog_sequence(
        monkeypatch,
        [
            ImportDialogResult(config=None, action="import"),
        ],
    )

    panel = DataManagerPanel()
    qtbot.addWidget(panel)
    panel._load_dataset()

    assert panel.get_datasets() == {}


def test_excel_import_rejects_mixed_unit_row_presence(tmp_path, monkeypatch, qtbot):
    from kindred.gui.widgets.data_manager import DataManagerPanel

    workbook_path = tmp_path / "mixed_presence.xlsx"
    _write_workbook(
        workbook_path,
        {
            "SheetA": (["time", "A"], [["ms", "uM"], [0.0, 1.0], [1.0, 2.0]]),
            "SheetB": (["time", "A"], [[0.0, 3.0], [1.0, 4.0]]),
        },
    )

    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getOpenFileNames",
        lambda *args, **kwargs: ([str(workbook_path)], ""),
    )
    _patch_dialog_sequence(
        monkeypatch,
        [
            ImportDialogResult(config=None, action="import"),
        ],
    )

    panel = DataManagerPanel()
    qtbot.addWidget(panel)
    panel._load_dataset()

    assert panel.get_datasets() == {}


def test_excel_import_does_not_strip_row_when_only_unselected_columns_look_like_units(tmp_path, monkeypatch, qtbot):
    from kindred.gui.widgets.data_manager import DataManagerPanel

    workbook_path = tmp_path / "notes.xlsx"
    _write_workbook(
        workbook_path,
        {
            "SheetA": (
                ["time", "A", "note1", "note2"],
                [[0.0, 1.0, "ms", "uM"], [1.0, 2.0, "", ""]],
            ),
        },
    )

    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getOpenFileNames",
        lambda *args, **kwargs: ([str(workbook_path)], ""),
    )
    _patch_dialog_sequence(
        monkeypatch,
        [
            ImportDialogResult(
                config=_make_test_config(
                    filepath=str(workbook_path),
                    file_type="excel",
                    sheet_names=["SheetA"],
                    time_column="time",
                    species_columns=["A"],
                    time_unit="s",
                    concentration_unit="M",
                    unit_row_detected=False,
                ),
                action="import",
            )
        ],
    )

    panel = DataManagerPanel()
    qtbot.addWidget(panel)
    panel._load_dataset()
    datasets, _finished_spy = _wait_for_load(panel, qtbot, expected_count=1)

    payload = datasets["notes.xlsx::SheetA"]
    assert np.allclose(payload["t"], [0.0, 1.0])
    assert np.allclose(payload["species"]["A"], [1.0, 2.0])


def test_unit_conversion_applied_at_import(tmp_path, monkeypatch, qtbot):
    from kindred.gui.widgets.data_manager import DataManagerPanel

    csv_path = tmp_path / "microseconds.csv"
    _write_csv(csv_path, ["time", "A"], [[100, 1.0], [200, 2.0], [300, 3.0]])

    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getOpenFileNames",
        lambda *args, **kwargs: ([str(csv_path)], ""),
    )
    _patch_dialog_sequence(
        monkeypatch,
        [
            ImportDialogResult(
                config=_make_test_config(
                    filepath=str(csv_path),
                    file_type="csv",
                    time_column="time",
                    species_columns=["A"],
                    time_unit="us",
                    concentration_unit="M",
                ),
                action="import",
            )
        ],
    )

    panel = DataManagerPanel()
    qtbot.addWidget(panel)
    panel._load_dataset()
    datasets, _finished_spy = _wait_for_load(panel, qtbot, expected_count=1)

    payload = datasets["microseconds.csv"]
    assert np.allclose(payload["t"], [1e-4, 2e-4, 3e-4])
    assert payload["metadata"]["original_time_unit"] == "us"
    assert payload["metadata"]["original_concentration_unit"] == "M"


def test_apply_to_remaining_stops_after_current_file(tmp_path, monkeypatch, qtbot):
    from kindred.gui.widgets.data_manager import DataManagerPanel

    filepaths = []
    for idx in range(3):
        path = tmp_path / f"clone_{idx}.csv"
        _write_csv(path, ["time", "A", "B"], [[0.0, idx + 1.0, idx + 2.0], [1.0, idx + 3.0, idx + 4.0]])
        filepaths.append(str(path))

    monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileNames", lambda *args, **kwargs: (filepaths, ""))
    created = _patch_dialog_sequence(
        monkeypatch,
        [
            ImportDialogResult(
                config=_make_test_config(
                    filepath=filepaths[0],
                    file_type="csv",
                    time_column="time",
                    species_columns=["A", "B"],
                    time_unit="s",
                    concentration_unit="M",
                    apply_to_remaining=True,
                ),
                action="import",
            )
        ],
    )

    panel = DataManagerPanel()
    qtbot.addWidget(panel)
    panel._load_dataset()
    datasets, _finished_spy = _wait_for_load(panel, qtbot, expected_count=3)

    assert len(created) == 1
    assert set(datasets) == {"clone_0.csv", "clone_1.csv", "clone_2.csv"}

def test_apply_to_remaining_skips_later_csv_files_even_when_units_differ(tmp_path, monkeypatch, qtbot):
    from kindred.gui.widgets.data_manager import DataManagerPanel

    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    _write_csv(first, ["time", "A"], [["ms", "uM"], [1000, 2.0], [2000, 4.0]])
    _write_csv(second, ["time", "A"], [[1.0, 3.0], [2.0, 5.0]])

    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getOpenFileNames",
        lambda *args, **kwargs: ([str(first), str(second)], ""),
    )
    created = _patch_dialog_sequence(
        monkeypatch,
        [
            ImportDialogResult(
                config=_make_test_config(
                    filepath=str(first),
                    file_type="csv",
                    time_column="time",
                    species_columns=["A"],
                    time_unit="ms",
                    concentration_unit="uM",
                    unit_row_detected=True,
                    apply_to_remaining=True,
                ),
                action="import",
            ),
            ImportDialogResult(
                config=_make_test_config(
                    filepath=str(second),
                    file_type="csv",
                    time_column="time",
                    species_columns=["A"],
                    time_unit="s",
                    concentration_unit="M",
                    unit_row_detected=False,
                ),
                action="import",
            ),
        ],
    )

    panel = DataManagerPanel()
    qtbot.addWidget(panel)
    panel._load_dataset()
    datasets, _finished_spy = _wait_for_load(panel, qtbot, expected_count=2)

    assert len(created) == 1
    assert set(datasets) == {"first.csv", "second.csv"}

def test_apply_to_remaining_skips_later_csv_files_even_when_columns_differ(tmp_path, monkeypatch, qtbot):
    from kindred.gui.widgets.data_manager import DataManagerPanel

    first = tmp_path / "same.csv"
    second = tmp_path / "different.csv"
    _write_csv(first, ["time", "A", "B"], [[0.0, 1.0, 2.0], [1.0, 3.0, 4.0]])
    _write_csv(second, ["time", "A", "C"], [[0.0, 5.0, 6.0], [1.0, 7.0, 8.0]])

    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getOpenFileNames",
        lambda *args, **kwargs: ([str(first), str(second)], ""),
    )
    created = _patch_dialog_sequence(
        monkeypatch,
        [
            ImportDialogResult(
                config=_make_test_config(
                    filepath=str(first),
                    file_type="csv",
                    time_column="time",
                    species_columns=["A", "B"],
                    time_unit="s",
                    concentration_unit="M",
                    apply_to_remaining=True,
                ),
                action="import",
            ),
            ImportDialogResult(
                config=_make_test_config(
                    filepath=str(second),
                    file_type="csv",
                    time_column="time",
                    species_columns=["A", "C"],
                    time_unit="s",
                    concentration_unit="M",
                ),
                action="import",
            ),
        ],
    )

    criticals: list[tuple] = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "critical",
        lambda *args, **kwargs: criticals.append(args),
    )

    panel = DataManagerPanel()
    qtbot.addWidget(panel)
    panel._load_dataset()
    datasets, _finished_spy = _wait_for_load(panel, qtbot, expected_count=1)

    assert len(created) == 1
    assert set(datasets) == {"same.csv"}
    assert criticals, "Error dialog must be shown for file with incompatible columns"
    assert "different.csv" in str(criticals[0])

def test_apply_to_remaining_skips_later_excel_files(tmp_path, monkeypatch, qtbot):
    from kindred.gui.widgets.data_manager import DataManagerPanel

    first = tmp_path / "first.xlsx"
    second = tmp_path / "second.xlsx"
    _write_workbook(
        first,
        {
            "SheetA": (["time", "A"], [["ms", "uM"], [0.0, 1.0], [1.0, 2.0]]),
            "SheetB": (["time", "A"], [["ms", "uM"], [0.0, 3.0], [1.0, 4.0]]),
        },
    )
    _write_workbook(
        second,
        {
            "SheetA": (["time", "A"], [["ms", "uM"], [0.0, 5.0], [1.0, 6.0]]),
            "SheetB": (["time", "A"], [["s", "M"], [0.0, 7.0], [1.0, 8.0]]),
        },
    )

    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getOpenFileNames",
        lambda *args, **kwargs: ([str(first), str(second)], ""),
    )
    created = _patch_dialog_sequence(
        monkeypatch,
        [
            ImportDialogResult(
                config=_make_test_config(
                    filepath=str(first),
                    file_type="excel",
                    sheet_names=["SheetA", "SheetB"],
                    time_column="time",
                    species_columns=["A"],
                    time_unit="ms",
                    concentration_unit="uM",
                    unit_row_detected=True,
                    apply_to_remaining=True,
                ),
                action="import",
            ),
            ImportDialogResult(
                config=None,
                action="import",
            ),
        ],
    )

    panel = DataManagerPanel()
    qtbot.addWidget(panel)
    panel._load_dataset()
    datasets, _finished_spy = _wait_for_load(panel, qtbot, expected_count=4)

    assert len(created) == 1
    assert set(datasets) == {
        "first.xlsx::SheetA", "first.xlsx::SheetB",
        "second.xlsx::SheetA", "second.xlsx::SheetB",
    }


def test_skip_action_skips_file(tmp_path, monkeypatch, qtbot):
    from kindred.gui.widgets.data_manager import DataManagerPanel

    first = tmp_path / "skip.csv"
    second = tmp_path / "keep.csv"
    _write_csv(first, ["time", "A"], [[0.0, 1.0], [1.0, 2.0]])
    _write_csv(second, ["time", "A"], [[0.0, 3.0], [1.0, 4.0]])

    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getOpenFileNames",
        lambda *args, **kwargs: ([str(first), str(second)], ""),
    )
    _patch_dialog_sequence(
        monkeypatch,
        [
            ImportDialogResult(config=None, action="skip"),
            ImportDialogResult(
                config=_make_test_config(
                    filepath=str(second),
                    file_type="csv",
                    time_column="time",
                    species_columns=["A"],
                    time_unit="s",
                    concentration_unit="M",
                ),
                action="import",
            ),
        ],
    )

    panel = DataManagerPanel()
    qtbot.addWidget(panel)
    panel._load_dataset()
    datasets, _finished_spy = _wait_for_load(panel, qtbot, expected_count=1)

    assert set(datasets) == {"keep.csv"}


def test_cancel_action_aborts_all(tmp_path, monkeypatch, qtbot):
    from kindred.gui.widgets.data_manager import DataManagerPanel

    files = []
    for idx in range(3):
        path = tmp_path / f"cancel_{idx}.csv"
        _write_csv(path, ["time", "A"], [[0.0, idx + 1.0], [1.0, idx + 2.0]])
        files.append(str(path))

    monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileNames", lambda *args, **kwargs: (files, ""))
    _patch_dialog_sequence(monkeypatch, [ImportDialogResult(config=None, action="cancel")])

    panel = DataManagerPanel()
    qtbot.addWidget(panel)
    panel._load_dataset()

    assert panel.get_datasets() == {}
    assert panel._progress_dialog is None
    assert not panel._csv_workers


def test_legacy_mapping_widget_removed(qtbot):
    from kindred.gui.widgets.data_manager import DataManagerPanel

    panel = DataManagerPanel()
    qtbot.addWidget(panel)

    assert not hasattr(panel, "_mapping_widget")
    assert not hasattr(panel, "_time_col_edit")
    assert not hasattr(panel, "_species_col_edit")


def test_deadcode_allowlist_entries_removed() -> None:
    allowlist_path = Path(__file__).resolve().parent.parent / "tools" / "audit" / "deadcode_test_only_keep_allowlist.txt"
    text = allowlist_path.read_text(encoding="utf-8")

    assert "kindred/core/datasets/units.py" not in text
    assert "kindred/core/datasets/excel_import.py" not in text
    assert "kindred/gui/widgets/import_config_dialog.py" not in text


def test_unsupported_xls_file_shows_error_and_skips_to_next_file(tmp_path, monkeypatch, qtbot):
    from kindred.gui.widgets import data_manager as data_manager_module
    from kindred.gui.widgets.data_manager import DataManagerPanel

    bad_xls = tmp_path / "legacy.xls"
    bad_xls.write_text("legacy", encoding="utf-8")
    good_csv = tmp_path / "good.csv"
    _write_csv(good_csv, ["time", "A"], [[0.0, 1.0], [1.0, 2.0]])

    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getOpenFileNames",
        lambda *args, **kwargs: ([str(bad_xls), str(good_csv)], ""),
    )
    captured_errors: list[str] = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "critical",
        lambda _parent, _title, message: captured_errors.append(str(message)),
    )

    class _DialogFactory:
        def __init__(self, filepath: str, remaining_count: int = 0, parent=None) -> None:
            if str(filepath).endswith(".xls"):
                raise ValueError("Legacy .xls format is not supported. Please save as .xlsx.")
            self._result = ImportDialogResult(
                config=_make_test_config(
                    filepath=str(filepath),
                    file_type="csv",
                    time_column="time",
                    species_columns=["A"],
                    time_unit="s",
                    concentration_unit="M",
                ),
                action="import",
            )

        def exec(self) -> int:
            return int(QtWidgets.QDialog.DialogCode.Accepted)

        def get_result(self) -> ImportDialogResult:
            return self._result

    monkeypatch.setattr(data_manager_module, "ImportConfigDialog", _DialogFactory)

    panel = DataManagerPanel()
    qtbot.addWidget(panel)
    panel._load_dataset()
    datasets, _finished_spy = _wait_for_load(panel, qtbot, expected_count=1)

    assert "good.csv" in datasets
    assert captured_errors
    assert "legacy.xls" in captured_errors[0]


def test_import_load_finished_prompts_mapping_when_batch_store_is_pristine(main_window, monkeypatch):
    payload = {"t": np.array([0.0, 1.0]), "species": {"A": np.array([1.0, 0.5])}}
    data_panel = main_window._right_panel._data_manager
    data_panel._datasets["dataset.csv"] = payload
    main_window._pending_import_batch_mapping_names = ["dataset.csv"]

    called: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        main_window,
        "_maybe_prompt_for_import_batch_mapping",
        lambda *args, **kwargs: called.append(args),
    )

    main_window._on_dataset_load_finished(False)

    assert len(called) == 1
    assert called[0][0] == "dataset.csv"
    assert main_window._pending_import_batch_mapping_names == []


def test_import_load_finished_still_prompts_after_partial_cancel(main_window, monkeypatch):
    payload = {"t": np.array([0.0, 1.0]), "species": {"A": np.array([1.0, 0.5])}}
    data_panel = main_window._right_panel._data_manager
    data_panel._datasets["dataset.csv"] = payload
    main_window._batch_store.ensure_set("existing set")
    main_window._pending_import_batch_mapping_names = ["dataset.csv"]

    called: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        main_window,
        "_maybe_prompt_for_import_batch_mapping",
        lambda *args, **kwargs: called.append(args),
    )

    main_window._on_dataset_load_finished(True)

    assert len(called) == 1
    assert called[0][0] == "dataset.csv"


def test_import_batch_mapping_skip_clears_stale_mapping(main_window, monkeypatch):
    payload = {"t": np.array([0.0, 1.0]), "species": {"A": np.array([1.0, 0.5])}}
    main_window._batch_store.ensure_set("existing set")

    monkeypatch.setattr(
        "kindred.gui.fitting.batch_mapping.prompt_dataset_batch_mapping_choice",
        lambda *args, **kwargs: "skip",
    )

    settings = main_window._dataset_manager.get_fit_settings("dataset.csv")
    settings.batch_set = "stale set"
    settings.batch_set_id = "stale-id"
    main_window._dataset_manager.update_fit_settings("dataset.csv", settings)

    main_window._maybe_prompt_for_import_batch_mapping("dataset.csv", payload, [])

    updated = main_window._dataset_manager.get_fit_settings("dataset.csv")
    assert updated.batch_set is None
    assert updated.batch_set_id is None


def test_import_batch_mapping_create_seeds_and_persists_batch_set_id(main_window, monkeypatch):
    payload = {"t": np.array([0.0, 1.0]), "species": {"A": np.array([1.5, 1.0])}}
    main_window._batch_store.ensure_set("existing set")

    monkeypatch.setattr(
        "kindred.gui.fitting.batch_mapping.prompt_dataset_batch_mapping_choice",
        lambda *args, **kwargs: "create",
    )

    main_window._maybe_prompt_for_import_batch_mapping("dataset.csv", payload, ["A"])

    settings = main_window._dataset_manager.get_fit_settings("dataset.csv")
    created_set = "dataset"
    assert settings.batch_set == created_set
    assert settings.batch_set_id == main_window._batch_store.set_id_for_row(main_window._batch_store.row_for_set(created_set))
    assert main_window._batch_store.get_value(main_window._batch_store.row_for_set(created_set), "A") == "1.5"


def test_import_batch_mapping_create_preserves_excel_sheet_identity_in_set_name(main_window, monkeypatch):
    payload = {"t": np.array([0.0, 1.0]), "species": {"A": np.array([1.5, 1.0])}}
    main_window._batch_store.ensure_set("existing set")

    monkeypatch.setattr(
        "kindred.gui.fitting.batch_mapping.prompt_dataset_batch_mapping_choice",
        lambda *args, **kwargs: "create",
    )

    main_window._maybe_prompt_for_import_batch_mapping("multi.xlsx::SheetA", payload, ["A"])

    settings = main_window._dataset_manager.get_fit_settings("multi.xlsx::SheetA")
    assert settings.batch_set == "multi_SheetA"
    assert settings.batch_set_id == main_window._batch_store.set_id_for_row(main_window._batch_store.row_for_set("multi_SheetA"))


def test_import_batch_mapping_create_unseeded_leaves_dataset_unmapped(main_window, monkeypatch):
    payload = {"t": np.array([1.0, 2.0]), "species": {"A": np.array([3.0, 4.0])}}
    main_window._batch_store.ensure_set("existing set")

    monkeypatch.setattr(
        "kindred.gui.fitting.batch_mapping.prompt_dataset_batch_mapping_choice",
        lambda *args, **kwargs: "create",
    )

    main_window._maybe_prompt_for_import_batch_mapping("dataset.csv", payload, ["A"])

    created_set = "dataset"
    row = main_window._batch_store.row_for_set(created_set)
    assert row is not None
    assert main_window._batch_store.get_value(int(row), "A") in ("", "0", "0.0")

    settings = main_window._dataset_manager.get_fit_settings("dataset.csv")
    assert settings.batch_set is None
    assert settings.batch_set_id is None


def test_import_batch_mapping_create_syncs_visible_species(main_window, monkeypatch):
    payload = {"t": np.array([0.0, 1.0]), "species": {"A": np.array([1.0, 0.5])}}
    main_window._batch_store.ensure_set("existing set")

    monkeypatch.setattr(
        "kindred.gui.fitting.batch_mapping.prompt_dataset_batch_mapping_choice",
        lambda *args, **kwargs: "create",
    )

    main_window._maybe_prompt_for_import_batch_mapping("dataset.csv", payload, ["A"])

    assert "A" in list(main_window._batch_store.visible_species())


def test_import_batch_mapping_map_existing_persists_batch_set_id(main_window, monkeypatch):
    payload = {"t": np.array([0.0, 1.0]), "species": {"A": np.array([1.0, 0.5])}}
    main_window._batch_store.ensure_set("existing set")

    monkeypatch.setattr(
        "kindred.gui.fitting.batch_mapping.prompt_dataset_batch_mapping_choice",
        lambda *args, **kwargs: "map",
    )
    monkeypatch.setattr(
        "kindred.gui.fitting.batch_mapping.pick_existing_batch_set",
        lambda *args, **kwargs: "existing set",
    )

    main_window._maybe_prompt_for_import_batch_mapping("dataset.csv", payload, ["A"])

    settings = main_window._dataset_manager.get_fit_settings("dataset.csv")
    row = main_window._batch_store.row_for_set("existing set")
    assert settings.batch_set == "existing set"
    assert settings.batch_set_id == main_window._batch_store.set_id_for_row(int(row))


def test_apply_batch_mapping_to_settings_clears_stale_id_without_batch_store(main_window):
    from kindred.gui.fitting.batch_mapping import apply_batch_mapping_to_settings

    settings = main_window._dataset_manager.get_fit_settings("dataset.csv")
    settings.batch_set = "stale set"
    settings.batch_set_id = "stale-id"

    resolution = apply_batch_mapping_to_settings(settings, None, "new set")

    assert resolution.batch_set == "new set"
    assert resolution.batch_set_id is None
    assert settings.batch_set == "new set"
    assert settings.batch_set_id is None


# ---------------------------------------------------------------------------
# Batch mapping t≈0 guard regression tests
# ---------------------------------------------------------------------------


class TestBatchMappingT0Guard:
    """Regression: false-positive 'does not start at t≈0' when mechanism_species is empty."""

    def test_batch_mapping_t0_zero_empty_species_no_warning(self, main_window, monkeypatch):
        """t0=0 + empty mechanism_species: the t≈0 warning must NOT fire and
        the dataset must NOT be mapped (empty species guard returns early)."""
        main_window._batch_store.ensure_set("existing set")
        payload = {"t": np.array([0.0, 1.0, 2.0]), "species": {"A": np.array([1.0, 0.9, 0.8])}}

        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
        monkeypatch.setattr(
            "kindred.gui.fitting.batch_mapping.prompt_dataset_batch_mapping_choice",
            lambda *args, **kwargs: "create",
        )
        warnings_fired: list[tuple] = []
        monkeypatch.setattr(
            QtWidgets.QMessageBox,
            "warning",
            lambda *args, **kwargs: (
                warnings_fired.append(args),
                QtWidgets.QMessageBox.StandardButton.Ok,
            )[-1],
        )
        monkeypatch.setattr(QtWidgets.QMessageBox, "information", lambda *args, **kwargs: None)

        main_window._maybe_prompt_for_import_batch_mapping("dataset.csv", payload, [])

        assert not any("does not start" in str(w) for w in warnings_fired), (
            "False-positive t\u22480 warning fired when t0=0.0"
        )
        settings = main_window._dataset_manager.get_fit_settings("dataset.csv")
        assert settings.batch_set is None, (
            "Mapping must not be persisted when mechanism_species is empty"
        )

    def test_batch_mapping_t0_nonzero_empty_species_no_mapping(self, main_window, monkeypatch):
        """t0=5 + empty mechanism_species: the empty-species guard returns
        early before the t≈0 check, so no warning fires and no mapping is persisted."""
        main_window._batch_store.ensure_set("existing set")
        payload = {"t": np.array([5.0, 6.0, 7.0]), "species": {"A": np.array([1.0, 0.9, 0.8])}}

        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
        monkeypatch.setattr(
            "kindred.gui.fitting.batch_mapping.prompt_dataset_batch_mapping_choice",
            lambda *args, **kwargs: "create",
        )
        warnings_fired: list[tuple] = []
        monkeypatch.setattr(
            QtWidgets.QMessageBox,
            "warning",
            lambda *args, **kwargs: (
                warnings_fired.append(args),
                QtWidgets.QMessageBox.StandardButton.Ok,
            )[-1],
        )
        monkeypatch.setattr(QtWidgets.QMessageBox, "information", lambda *args, **kwargs: None)

        main_window._maybe_prompt_for_import_batch_mapping("dataset.csv", payload, [])

        assert not any("does not start" in str(w) for w in warnings_fired), (
            "t\u22480 warning should not fire when empty-species guard returns early"
        )
        settings = main_window._dataset_manager.get_fit_settings("dataset.csv")
        assert settings.batch_set is None, (
            "Mapping must not be persisted when mechanism_species is empty"
        )


# ---------------------------------------------------------------------------
# Empty mechanism_species batch mapping guard
# ---------------------------------------------------------------------------


def test_empty_mechanism_mapping_create_and_seed_returns_unseeded():
    """create_and_seed_batch_set must return seeded=False when mechanism_species
    is empty, even when the dataset starts at t=0 and has species data."""
    from unittest import mock

    from kindred.gui.fitting.batch_mapping import create_and_seed_batch_set

    batch_store = mock.MagicMock()
    batch_store.row_for_set.return_value = None
    batch_store.row_count.return_value = 0
    batch_store.ensure_set.return_value = 0
    batch_model = mock.MagicMock()

    row_idx, created, seeded = create_and_seed_batch_set(
        dataset_name="ds1",
        dataset_payload={"t": [0.0, 1.0], "species": {"A": [1.0, 2.0]}},
        mechanism_species=[],
        batch_store=batch_store,
        batch_model=batch_model,
        set_name="Batch Set 1",
    )

    assert created is True
    assert seeded is False
    batch_store.set_value.assert_not_called()


def test_apply_to_remaining_imports_all_remaining_csv_files(tmp_path, monkeypatch, qtbot):
    """apply_to_remaining must import remaining files, not silently drop them."""
    from kindred.gui.widgets.data_manager import DataManagerPanel

    filepaths = []
    for idx in range(2):
        path = tmp_path / f"data_{idx}.csv"
        _write_csv(path, ["time", "A"], [
            ["s", "uM"],
            [0.0, 1.0 + idx],
            [1.0, 2.0 + idx],
        ])
        filepaths.append(str(path))

    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getOpenFileNames",
        lambda *args, **kwargs: (filepaths, ""),
    )
    created = _patch_dialog_sequence(
        monkeypatch,
        [
            ImportDialogResult(
                config=_make_test_config(
                    filepath=filepaths[0],
                    file_type="csv",
                    time_column="time",
                    species_columns=["A"],
                    time_unit="s",
                    concentration_unit="uM",
                    unit_row_detected=True,
                    apply_to_remaining=True,
                ),
                action="import",
            )
        ],
    )

    panel = DataManagerPanel()
    qtbot.addWidget(panel)
    finished_spy = QSignalSpy(panel.loadFinished)
    panel._load_dataset()
    qtbot.waitUntil(lambda: finished_spy.count() == 1, timeout=7000)

    assert len(created) == 1, "Dialog must only open for the first file"
    assert len(panel.get_datasets()) == 2, (
        "Both files must be imported when apply_to_remaining is True"
    )


def test_apply_to_remaining_error_on_incompatible_remaining_file(tmp_path, monkeypatch, qtbot):
    """When a remaining file has incompatible columns,
    QMessageBox.critical must be shown and that file skipped."""
    from kindred.gui.widgets.data_manager import DataManagerPanel

    source = tmp_path / "source.csv"
    target = tmp_path / "target.csv"
    _write_csv(source, ["time", "A", "B"], [[0.0, 1.0, 2.0], [1.0, 3.0, 4.0]])
    _write_csv(target, ["time", "A", "C"], [[0.0, 5.0, 6.0], [1.0, 7.0, 8.0]])

    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getOpenFileNames",
        lambda *args, **kwargs: ([str(source), str(target)], ""),
    )
    _patch_dialog_sequence(
        monkeypatch,
        [
            ImportDialogResult(
                config=_make_test_config(
                    filepath=str(source),
                    file_type="csv",
                    time_column="time",
                    species_columns=["A", "B"],
                    time_unit="s",
                    concentration_unit="M",
                    apply_to_remaining=True,
                ),
                action="import",
            )
        ],
    )

    criticals: list[tuple] = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "critical",
        lambda *args, **kwargs: criticals.append(args),
    )

    panel = DataManagerPanel()
    qtbot.addWidget(panel)
    finished_spy = QSignalSpy(panel.loadFinished)
    panel._load_dataset()
    qtbot.waitUntil(lambda: finished_spy.count() == 1, timeout=7000)

    assert len(panel.get_datasets()) == 1, "Only the source file should be imported"
    assert "source.csv" in list(panel.get_datasets().keys())[0]
    assert criticals, "QMessageBox.critical must be shown for incompatible file"
    assert "target.csv" in str(criticals[0]), "Error must mention the filename"


def test_empty_mechanism_mapping_not_persisted_at_import(main_window, monkeypatch):
    """When mechanism_species is empty at import time, the dataset must NOT be
    mapped to the created batch set so launch.py can re-seed later."""
    main_window._batch_store.ensure_set("existing set")
    payload = {"t": np.array([0.0, 1.0, 2.0]), "species": {"A": np.array([1.0, 0.9, 0.8])}}

    monkeypatch.setattr(
        "kindred.gui.fitting.batch_mapping.prompt_dataset_batch_mapping_choice",
        lambda *args, **kwargs: "create",
    )
    monkeypatch.setattr(QtWidgets.QMessageBox, "information", lambda *args, **kwargs: None)

    main_window._maybe_prompt_for_import_batch_mapping("dataset.csv", payload, [])

    settings = main_window._dataset_manager.get_fit_settings("dataset.csv")
    assert settings.batch_set is None, (
        "Dataset should NOT be mapped when mechanism_species is empty"
    )
    assert settings.batch_set_id is None, (
        "batch_set_id should be None when mechanism_species is empty"
    )


# ---------------------------------------------------------------------------
# Regression: scoped remaining-file detection
# ---------------------------------------------------------------------------


def test_apply_to_remaining_scoped_detection_ignores_unselected_columns(tmp_path, monkeypatch, qtbot):
    """Regression: remaining-file unit detection must scope to selected
    columns only.  Mixed units in unselected columns must not reject the file."""
    from kindred.gui.widgets.data_manager import DataManagerPanel

    file1 = tmp_path / "source.csv"
    file2 = tmp_path / "target.csv"
    _write_csv(file1, ["time", "A", "B"], [["s", "uM", "uM"], [0.0, 1.0, 2.0], [1.0, 3.0, 4.0]])
    _write_csv(file2, ["time", "A", "B"], [["s", "uM", "nM"], [0.0, 5.0, 6.0], [1.0, 7.0, 8.0]])

    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getOpenFileNames",
        lambda *args, **kwargs: ([str(file1), str(file2)], ""),
    )
    created = _patch_dialog_sequence(
        monkeypatch,
        [
            ImportDialogResult(
                config=_make_test_config(
                    filepath=str(file1),
                    file_type="csv",
                    time_column="time",
                    species_columns=["A"],
                    time_unit="s",
                    concentration_unit="uM",
                    unit_row_detected=True,
                    apply_to_remaining=True,
                ),
                action="import",
            )
        ],
    )

    criticals: list[tuple] = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "critical",
        lambda *args, **kwargs: criticals.append(args),
    )

    panel = DataManagerPanel()
    qtbot.addWidget(panel)
    finished_spy = QSignalSpy(panel.loadFinished)
    panel._load_dataset()
    qtbot.waitUntil(lambda: finished_spy.count() == 1, timeout=7000)

    assert len(created) == 1
    assert not criticals, "No error expected; column B mixed unit must be ignored"
    assert len(panel.get_datasets()) == 2, (
        "Both files must import when mixed units are only in unselected columns"
    )


# ---------------------------------------------------------------------------
# Regression: sheet filtering for remaining Excel files
# ---------------------------------------------------------------------------


def test_apply_to_remaining_filters_sheets_by_source_checked_set(tmp_path, monkeypatch, qtbot):
    """Regression: remaining Excel files must only import sheets that
    were checked in the source file's configuration."""
    from kindred.gui.widgets.data_manager import DataManagerPanel

    source = tmp_path / "source.xlsx"
    target = tmp_path / "target.xlsx"
    _write_workbook(
        source,
        {
            "Data": (["time", "A"], [[0.0, 1.0], [1.0, 2.0]]),
            "Metadata": (["key", "val"], [["temp", "298"]]),
        },
    )
    _write_workbook(
        target,
        {
            "Data": (["time", "A"], [[0.0, 3.0], [1.0, 4.0]]),
            "Metadata": (["key", "val"], [["temp", "310"]]),
            "Extra": (["x", "y"], [[1, 2]]),
        },
    )

    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getOpenFileNames",
        lambda *args, **kwargs: ([str(source), str(target)], ""),
    )
    created = _patch_dialog_sequence(
        monkeypatch,
        [
            ImportDialogResult(
                config=_make_test_config(
                    filepath=str(source),
                    file_type="excel",
                    sheet_names=["Data"],
                    time_column="time",
                    species_columns=["A"],
                    time_unit="s",
                    concentration_unit="M",
                    apply_to_remaining=True,
                ),
                action="import",
            )
        ],
    )

    criticals: list[tuple] = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "critical",
        lambda *args, **kwargs: criticals.append(args),
    )

    panel = DataManagerPanel()
    qtbot.addWidget(panel)
    finished_spy = QSignalSpy(panel.loadFinished)
    panel._load_dataset()
    qtbot.waitUntil(lambda: finished_spy.count() == 1, timeout=7000)

    assert len(created) == 1
    datasets = panel.get_datasets()
    dataset_names = set(datasets.keys())
    assert "target.xlsx::Data" in dataset_names, "Data sheet must be imported from target"
    assert "target.xlsx::Metadata" not in dataset_names, "Metadata must be filtered out"
    assert "target.xlsx::Extra" not in dataset_names, "Extra must be filtered out"
    assert len(datasets) == 2, "source::Data + target::Data only"


# ---------------------------------------------------------------------------
# Regression: UnicodeDecodeError in remaining-file loop
# ---------------------------------------------------------------------------


def test_apply_to_remaining_handles_unicode_decode_error(tmp_path, monkeypatch, qtbot):
    """Regression: UnicodeDecodeError in remaining files must trigger
    an error dialog and not crash the import sequence."""
    from kindred.gui.widgets.data_manager import DataManagerPanel

    file1 = tmp_path / "good.csv"
    file2 = tmp_path / "bad_encoding.csv"
    _write_csv(file1, ["time", "A"], [[0.0, 1.0], [1.0, 2.0]])
    file2.write_bytes(b"time,A\n\xff\xfe,1.0\n")

    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getOpenFileNames",
        lambda *args, **kwargs: ([str(file1), str(file2)], ""),
    )
    created = _patch_dialog_sequence(
        monkeypatch,
        [
            ImportDialogResult(
                config=_make_test_config(
                    filepath=str(file1),
                    file_type="csv",
                    time_column="time",
                    species_columns=["A"],
                    time_unit="s",
                    concentration_unit="M",
                    apply_to_remaining=True,
                ),
                action="import",
            )
        ],
    )

    criticals: list[tuple] = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "critical",
        lambda *args, **kwargs: criticals.append(args),
    )

    panel = DataManagerPanel()
    qtbot.addWidget(panel)
    finished_spy = QSignalSpy(panel.loadFinished)
    panel._load_dataset()
    qtbot.waitUntil(lambda: finished_spy.count() == 1, timeout=7000)

    assert len(created) == 1
    assert len(panel.get_datasets()) == 1, "Only the good file should be imported"
    assert "good.csv" in list(panel.get_datasets().keys())[0]
    assert criticals, "QMessageBox.critical must be shown for encoding error"
    assert "bad_encoding.csv" in str(criticals[0])


# ---------------------------------------------------------------------------
# Remaining-file detection uses full row for has_unit_row
# ---------------------------------------------------------------------------


def test_remaining_file_detects_unit_row_from_unselected_columns(tmp_path, qtbot):
    """has_unit_row is a physical property of the full row.  When unit text
    appears only in an unselected column (B has 'nM', selected columns time
    and A are blank), _build_remaining_file_config must still detect the
    unit row so that skip_unit_row=True in the resolved plan."""
    from kindred.gui.widgets.data_manager import DataManagerPanel

    # CSV where unit text appears ONLY in unselected column B
    csv_path = tmp_path / "sparse_units.csv"
    _write_csv(csv_path, ["time", "A", "B"], [["", "", "nM"], [0.0, 5.0, 6.0], [1.0, 7.0, 8.0]])

    source_intent = SheetImportIntent(
        time_column="time",
        species_columns=("A",),
        time_unit="s",
        concentration_unit="M",
        override_no_unit_row=False,
    )

    panel = DataManagerPanel()
    qtbot.addWidget(panel)
    config = panel._build_remaining_file_config(str(csv_path), source_intent)

    assert config.plans[0].skip_unit_row is True, (
        "Unit row must be detected from full row (column B has 'nM') "
        "even though selected column A has no unit text"
    )


def test_remaining_excel_file_detects_unit_row_from_unselected_columns(tmp_path, qtbot):
    """Excel variant: unit text only in unselected column B must still
    trigger has_unit_row=True for the remaining file."""
    from kindred.gui.widgets.data_manager import DataManagerPanel

    xlsx_path = tmp_path / "sparse_units.xlsx"
    _write_workbook(xlsx_path, {
        "Data": (["time", "A", "B"], [["", "", "nM"], [0.0, 5.0, 6.0], [1.0, 7.0, 8.0]]),
    })

    source_intent = SheetImportIntent(
        time_column="time",
        species_columns=("A",),
        time_unit="s",
        concentration_unit="M",
        override_no_unit_row=False,
    )

    panel = DataManagerPanel()
    qtbot.addWidget(panel)
    config = panel._build_remaining_file_config(
        str(xlsx_path), source_intent, source_sheet_names=("Data",),
    )

    assert config.plans[0].skip_unit_row is True, (
        "Unit row must be detected from full row (column B has 'nM') "
        "even though selected column A has no unit text"
    )
