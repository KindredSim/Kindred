from __future__ import annotations

import datetime as dt
import os
import time
from pathlib import Path

import numpy as np
import pytest
from openpyxl import Workbook

from kindred.core.datasets.csv_import import parse_csv_rows


def _save_workbook(path: Path, sheets: list[tuple[str, list[list[object]]]]) -> Path:
    workbook = Workbook()
    default_sheet = workbook.active
    first_name, first_rows = sheets[0]
    default_sheet.title = first_name
    for row in first_rows:
        default_sheet.append(list(row))

    for sheet_name, rows in sheets[1:]:
        sheet = workbook.create_sheet(title=sheet_name)
        for row in rows:
            sheet.append(list(row))

    workbook.save(path)
    workbook.close()
    return path


def test_list_sheets_returns_single_sheet_name(tmp_path: Path) -> None:
    from kindred.core.datasets.excel_import import list_sheets

    path = _save_workbook(tmp_path / "single.xlsx", [("Sheet1", [["time", "A"], [0, 1.0]])])

    assert list_sheets(str(path)) == ["Sheet1"]


def test_list_sheets_returns_named_sheets_in_order(tmp_path: Path) -> None:
    from kindred.core.datasets.excel_import import list_sheets

    path = _save_workbook(
        tmp_path / "multi.xlsx",
        [
            ("Alpha", [["time", "A"], [0, 1.0]]),
            ("Beta", [["time", "B"], [0, 2.0]]),
            ("Gamma", [["time", "C"], [0, 3.0]]),
        ],
    )

    assert list_sheets(str(path)) == ["Alpha", "Beta", "Gamma"]


@pytest.mark.parametrize("path_name", ["missing.xlsx", "corrupt.xlsx"])
def test_list_sheets_raises_for_missing_or_corrupt_workbook(tmp_path: Path, path_name: str) -> None:
    from kindred.core.datasets.excel_import import list_sheets

    path = tmp_path / path_name
    if path_name == "corrupt.xlsx":
        path.write_text("not an xlsx workbook", encoding="utf-8")

    with pytest.raises(ValueError):
        list_sheets(str(path))


def test_read_excel_sheet_rows_returns_stringified_row_dicts(tmp_path: Path) -> None:
    from kindred.core.datasets.excel_import import read_excel_sheet_rows

    path = _save_workbook(
        tmp_path / "rows.xlsx",
        [
            (
                "Data",
                [
                    ["time", "species_A", "flag"],
                    [0, 1, True],
                    [1.5, 2.25, False],
                    [2, None, None],
                ],
            )
        ],
    )

    rows = list(read_excel_sheet_rows(str(path), "Data"))

    assert rows == [
        {"time": "0", "species_A": "1", "flag": "True"},
        {"time": "1.5", "species_A": "2.25", "flag": "False"},
        {"time": "2", "species_A": "", "flag": ""},
    ]


def test_read_excel_sheet_rows_raises_for_unknown_sheet(tmp_path: Path) -> None:
    from kindred.core.datasets.excel_import import read_excel_sheet_rows

    path = _save_workbook(tmp_path / "unknown_sheet.xlsx", [("Known", [["time", "A"], [0, 1.0]])])

    with pytest.raises(ValueError, match="not found"):
        list(read_excel_sheet_rows(str(path), "Missing"))


def test_read_excel_sheet_rows_header_only_yields_no_rows(tmp_path: Path) -> None:
    from kindred.core.datasets.excel_import import read_excel_sheet_rows

    path = _save_workbook(tmp_path / "header_only.xlsx", [("Data", [["time", "A"]])])

    assert list(read_excel_sheet_rows(str(path), "Data")) == []


def test_read_excel_sheet_rows_is_lazy_and_closes_on_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from kindred.core.datasets import excel_import

    class _FakeSheet:
        def __init__(self) -> None:
            self.data_rows_yielded = 0

        def iter_rows(self, values_only: bool = True):
            yield ("time", "A")
            yield (0, 1.0)
            self.data_rows_yielded += 1
            yield (1, 2.0)
            self.data_rows_yielded += 1

    class _FakeWorkbook:
        def __init__(self) -> None:
            self.sheetnames = ["Data"]
            self.closed = 0
            self._sheet = _FakeSheet()

        def __getitem__(self, name: str):
            assert name == "Data"
            return self._sheet

        def close(self) -> None:
            self.closed += 1

    fake_workbook = _FakeWorkbook()
    monkeypatch.setattr(excel_import, "_open_workbook", lambda _path: fake_workbook)

    rows = excel_import.read_excel_sheet_rows(str(tmp_path / "rows.xlsx"), "Data")
    row_iter = iter(rows)

    assert fake_workbook.closed == 0
    assert fake_workbook._sheet.data_rows_yielded == 0
    assert next(row_iter) == {"time": "0", "A": "1.0"}
    assert fake_workbook.closed == 0
    assert fake_workbook._sheet.data_rows_yielded == 0
    assert next(row_iter) == {"time": "1", "A": "2.0"}
    assert fake_workbook._sheet.data_rows_yielded == 1

    rows.close()

    assert fake_workbook.closed == 1


def test_load_excel_dataset_returns_csv_payload_shape(tmp_path: Path) -> None:
    from kindred.core.datasets.excel_import import load_excel_dataset

    path = _save_workbook(
        tmp_path / "dataset.xlsx",
        [("SheetA", [["time", "A", "B"], [0, 1.0, 2.0], [1, 0.5, 1.5]])],
    )

    name, payload = load_excel_dataset(str(path), "SheetA")

    assert name == "dataset.xlsx::SheetA"
    assert np.allclose(payload["t"], [0.0, 1.0])
    assert np.allclose(payload["species"]["A"], [1.0, 0.5])
    assert np.allclose(payload["species"]["B"], [2.0, 1.5])
    assert payload["metadata"] == {
        "time_column": "time",
        "species_columns": ["A", "B"],
        "mapping_source": "auto",
    }


def test_load_excel_dataset_supports_auto_and_explicit_mapping(tmp_path: Path) -> None:
    from kindred.core.datasets.excel_import import load_excel_dataset

    path = _save_workbook(
        tmp_path / "explicit.xlsx",
        [("SheetB", [["elapsed", "A", "B", "label"], [0, 1.0, 2.0, "x"], [2, 3.0, 4.0, "y"]])],
    )

    name, payload = load_excel_dataset(
        str(path),
        "SheetB",
        time_column="elapsed",
        species_columns=["B"],
    )

    assert name == "explicit.xlsx::SheetB"
    assert np.allclose(payload["t"], [0.0, 2.0])
    assert list(payload["species"]) == ["B"]
    assert np.allclose(payload["species"]["B"], [2.0, 4.0])
    assert payload["metadata"]["mapping_source"] == "explicit"


def test_load_excel_dataset_auto_mode_drops_non_numeric_species_columns(tmp_path: Path) -> None:
    from kindred.core.datasets.excel_import import load_excel_dataset

    path = _save_workbook(
        tmp_path / "auto_drop.xlsx",
        [("Data", [["time", "A", "note"], [0, 1.0, "x"], [1, 2.0, "y"]])],
    )

    _name, payload = load_excel_dataset(str(path), "Data")

    assert list(payload["species"]) == ["A"]
    assert payload["metadata"]["species_columns"] == ["A"]


def test_load_excel_dataset_explicit_mode_errors_on_non_numeric_species_column(tmp_path: Path) -> None:
    from kindred.core.datasets.excel_import import load_excel_dataset

    path = _save_workbook(
        tmp_path / "explicit_error.xlsx",
        [("Data", [["time", "A", "note"], [0, 1.0, "x"], [1, 2.0, "y"]])],
    )

    with pytest.raises(ValueError, match="contains non-numeric values"):
        load_excel_dataset(str(path), "Data", species_columns=["note"])


def test_load_excel_workbook_loads_all_sheets_and_reports_failures(tmp_path: Path) -> None:
    from kindred.core.datasets.excel_import import ExcelWorkbookLoadResult, load_excel_workbook

    path = _save_workbook(
        tmp_path / "workbook.xlsx",
        [
            ("Good1", [["time", "A"], [0, 1.0], [1, 0.5]]),
            ("Bad", [["value", "label"], [1, "x"]]),
            ("Good2", [["t", "B"], [0, 2.0], [1, 3.0]]),
        ],
    )

    result = load_excel_workbook(str(path))

    assert isinstance(result, ExcelWorkbookLoadResult)
    assert [name for name, _payload in result.successes] == [
        "workbook.xlsx::Good1",
        "workbook.xlsx::Good2",
    ]
    assert [sheet_name for sheet_name, _message in result.failures] == ["Bad"]
    assert "Time column not found" in result.failures[0][1]


def test_load_excel_workbook_supports_selective_sheet_loading(tmp_path: Path) -> None:
    from kindred.core.datasets.excel_import import load_excel_workbook

    path = _save_workbook(
        tmp_path / "selective.xlsx",
        [
            ("One", [["time", "A"], [0, 1.0]]),
            ("Two", [["time", "B"], [0, 2.0]]),
            ("Three", [["time", "C"], [0, 3.0]]),
        ],
    )

    result = load_excel_workbook(str(path), sheet_names=["Three", "One"])

    assert [name for name, _payload in result.successes] == [
        "selective.xlsx::Three",
        "selective.xlsx::One",
    ]
    assert result.failures == []


def test_load_excel_workbook_reports_all_failures_when_all_sheets_bad(tmp_path: Path) -> None:
    from kindred.core.datasets.excel_import import load_excel_workbook

    path = _save_workbook(
        tmp_path / "all_bad.xlsx",
        [
            ("Bad1", [["value", "label"], [1, "x"]]),
            ("Bad2", [["elapsed", "note"], [2, "y"]]),
        ],
    )

    result = load_excel_workbook(str(path))

    assert result.successes == []
    assert [sheet_name for sheet_name, _message in result.failures] == ["Bad1", "Bad2"]


def test_load_excel_workbook_opens_workbook_once_for_multi_sheet_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from kindred.core.datasets import excel_import

    path = _save_workbook(
        tmp_path / "open_once.xlsx",
        [
            ("One", [["time", "A"], [0, 1.0]]),
            ("Two", [["time", "B"], [0, 2.0]]),
        ],
    )

    real_load_workbook = excel_import.load_workbook
    call_count = 0

    def _counting_load_workbook(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return real_load_workbook(*args, **kwargs)

    monkeypatch.setattr(excel_import, "load_workbook", _counting_load_workbook)

    result = excel_import.load_excel_workbook(str(path))

    assert len(result.successes) == 2
    assert call_count == 1


def test_load_excel_workbook_propagates_csv_import_interruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from kindred.core.datasets import excel_import
    from kindred.core.datasets.csv_import import CsvImportInterrupted

    path = _save_workbook(tmp_path / "interrupt.xlsx", [("Data", [["time", "A"], [0, 1.0]])])

    def _raise_interrupt(*args, **kwargs):
        raise CsvImportInterrupted()

    monkeypatch.setattr(excel_import, "parse_csv_rows", _raise_interrupt)

    with pytest.raises(CsvImportInterrupted):
        excel_import.load_excel_workbook(str(path))


def test_load_excel_dataset_interrupts_without_materializing_whole_sheet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from kindred.core.datasets import excel_import
    from kindred.core.datasets.csv_import import CsvImportInterrupted

    class _StreamingRows:
        def __init__(self) -> None:
            self.rows_yielded = 0
            self.closed = 0

        def __iter__(self):
            yield {"time": "s", "A": "uM"}
            for index in range(50):
                self.rows_yielded += 1
                yield {"time": str(index), "A": str(index + 1.0)}

        def close(self) -> None:
            self.closed += 1

    rows = _StreamingRows()
    monkeypatch.setattr(excel_import, "read_excel_sheet_rows", lambda *_args, **_kwargs: rows)

    def _interrupt_after_first_data_row() -> bool:
        return rows.rows_yielded > 1

    with pytest.raises(CsvImportInterrupted):
        excel_import.load_excel_dataset(
            str(tmp_path / "interrupt_dataset.xlsx"),
            "Data",
            interruption_checker=_interrupt_after_first_data_row,
        )

    assert rows.rows_yielded == 2
    assert rows.closed == 1


def test_load_excel_workbook_interrupts_without_exhausting_current_sheet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from kindred.core.datasets import excel_import
    from kindred.core.datasets.csv_import import CsvImportInterrupted

    class _FakeSheet:
        def __init__(self, name: str) -> None:
            self.name = name
            self.data_rows_yielded = 0

        def iter_rows(self, values_only: bool = True):
            yield ("time", "A")
            yield ("s", "uM")
            for index in range(50):
                self.data_rows_yielded += 1
                yield (index, index + 1.0)

    class _FakeWorkbook:
        def __init__(self) -> None:
            self.sheetnames = ["Data", "Later"]
            self.closed = 0
            self.sheets = {
                "Data": _FakeSheet("Data"),
                "Later": _FakeSheet("Later"),
            }

        def __getitem__(self, name: str):
            return self.sheets[name]

        def close(self) -> None:
            self.closed += 1

    fake_workbook = _FakeWorkbook()
    monkeypatch.setattr(excel_import, "_open_workbook", lambda _path: fake_workbook)

    def _interrupt_after_first_data_row() -> bool:
        return fake_workbook.sheets["Data"].data_rows_yielded > 1

    with pytest.raises(CsvImportInterrupted):
        excel_import.load_excel_workbook(
            str(tmp_path / "interrupt_workbook.xlsx"),
            interruption_checker=_interrupt_after_first_data_row,
        )

    assert fake_workbook.sheets["Data"].data_rows_yielded == 2
    assert fake_workbook.sheets["Later"].data_rows_yielded == 0
    assert fake_workbook.closed == 1


def test_load_excel_dataset_matches_parse_csv_rows_for_equivalent_content(tmp_path: Path) -> None:
    from kindred.core.datasets.excel_import import load_excel_dataset

    path = _save_workbook(
        tmp_path / "equivalent.xlsx",
        [("Data", [["time", "species_A", "species_B"], [0, 1.0, 2.0], [1, 1.5, 2.5]])],
    )

    _name, payload = load_excel_dataset(str(path), "Data")
    _time_source, expected = parse_csv_rows(
        [
            {"time": "0", "species_A": "1.0", "species_B": "2.0"},
            {"time": "1", "species_A": "1.5", "species_B": "2.5"},
        ]
    )

    assert np.allclose(payload["t"], expected["t"])
    assert np.allclose(payload["species"]["species_A"], expected["species"]["species_A"])
    assert np.allclose(payload["species"]["species_B"], expected["species"]["species_B"])
    assert payload["metadata"] == expected["metadata"]


def test_load_excel_dataset_supports_time_column_override_with_nonstandard_header(tmp_path: Path) -> None:
    from kindred.core.datasets.excel_import import load_excel_dataset

    path = _save_workbook(
        tmp_path / "override.xlsx",
        [("Data", [["elapsed_seconds", "A"], [0, 1.0], [2, 3.0]])],
    )

    _name, payload = load_excel_dataset(str(path), "Data", time_column="elapsed_seconds")

    assert payload["metadata"]["time_column"] == "elapsed_seconds"
    assert np.allclose(payload["t"], [0.0, 2.0])


def test_float_stringification_roundtrips_cleanly_through_parse_csv_rows(tmp_path: Path) -> None:
    from kindred.core.datasets.excel_import import read_excel_sheet_rows

    value = 1.05e-15
    path = _save_workbook(
        tmp_path / "roundtrip.xlsx",
        [("Data", [["time", "A"], [0, value], [1, 2.0]])],
    )

    rows = list(read_excel_sheet_rows(str(path), "Data"))

    assert rows[0]["A"] == repr(value)
    _time_source, payload = parse_csv_rows(rows)
    assert payload["species"]["A"][0] == value


def test_unicode_sheet_names_and_headers_are_supported(tmp_path: Path) -> None:
    from kindred.core.datasets.excel_import import list_sheets, load_excel_dataset

    path = _save_workbook(
        tmp_path / "unicode.xlsx",
        [("Δεδομένα µ", [["time", "κ", "β"], [0, 1.0, 2.0], [1, 3.0, 4.0]])],
    )

    assert list_sheets(str(path)) == ["Δεδομένα µ"]
    name, payload = load_excel_dataset(str(path), "Δεδομένα µ")
    assert name == "unicode.xlsx::Δεδομένα µ"
    assert list(payload["species"]) == ["κ", "β"]


def test_load_excel_dataset_converts_datetime_time_cells_to_epoch_seconds(tmp_path: Path) -> None:
    from kindred.core.datasets.excel_import import load_excel_dataset

    if not hasattr(time, "tzset"):
        pytest.skip("tzset is unavailable on this platform")

    t0 = dt.datetime(2024, 1, 15, 10, 30, 0)
    t1 = dt.datetime(2024, 1, 15, 10, 31, 30)
    path = _save_workbook(
        tmp_path / "datetime.xlsx",
        [("Data", [["time", "A"], [t0, 1.0], [t1, 2.5]])],
    )

    original_tz = os.environ.get("TZ")
    try:
        os.environ["TZ"] = "America/Los_Angeles"
        time.tzset()
        _name, payload = load_excel_dataset(str(path), "Data")
    finally:
        if original_tz is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = original_tz
        time.tzset()

    assert np.allclose(
        payload["t"],
        [
            t0.replace(tzinfo=dt.timezone.utc).timestamp(),
            t1.replace(tzinfo=dt.timezone.utc).timestamp(),
        ],
    )
    assert np.allclose(payload["species"]["A"], [1.0, 2.5])


def test_load_excel_dataset_converts_time_only_cells_to_seconds(tmp_path: Path) -> None:
    from kindred.core.datasets.excel_import import load_excel_dataset

    path = _save_workbook(
        tmp_path / "time_only.xlsx",
        [("Data", [["time", "A"], [dt.time(0, 0, 1), 1.0], [dt.time(0, 1, 30), 2.5]])],
    )

    _name, payload = load_excel_dataset(str(path), "Data")

    assert np.allclose(payload["t"], [1.0, 90.0])
    assert np.allclose(payload["species"]["A"], [1.0, 2.5])
