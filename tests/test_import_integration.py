from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from kindred.core.datasets.excel_import import (
    list_sheets,
    load_excel_dataset,
    load_excel_workbook,
    read_excel_sheet_rows,
)
from kindred.core.datasets.units import (
    looks_like_unit_row,
    parse_concentration_unit,
    parse_time_unit,
)
from kindred.gui.widgets.import_config_dialog import ImportConfigDialog

pytestmark = [pytest.mark.integration]

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "m9_test_datasets.xlsx"


def test_load_ph7_sheet_with_units() -> None:
    _name, payload = load_excel_dataset(str(FIXTURE_PATH), "pH7_run")

    assert payload["metadata"]["time_column"] == "time"
    assert {"A", "B", "C", "Int", "PBMP", "pinBOH"}.issubset(payload["species"])
    assert payload["t"].shape == (50,)
    assert np.all(np.isfinite(payload["t"]))
    assert np.all(payload["t"] >= 0.0)
    assert payload["species"]["A"].shape == (50,)
    assert payload["species"]["A"][0] > payload["species"]["A"][-1]
    assert payload["species"]["PBMP"][-1] >= payload["species"]["PBMP"][0]


def test_unit_row_detected_on_ph7() -> None:
    rows = list(read_excel_sheet_rows(str(FIXTURE_PATH), "pH7_run"))
    first_row = rows[0]

    assert looks_like_unit_row(list(first_row.values())) is True
    assert parse_time_unit(first_row["time"]) == pytest.approx(1e-6)
    assert parse_concentration_unit(first_row["A"]) == pytest.approx(1e-6)
    assert parse_concentration_unit(first_row["PBMP"]) == pytest.approx(1e-6)


def test_unit_conversion_roundtrip() -> None:
    _name, payload = load_excel_dataset(str(FIXTURE_PATH), "pH7_run")

    converted_t = payload["t"] * parse_time_unit("us")
    converted_a = payload["species"]["A"] * parse_concentration_unit("uM")

    assert converted_t[0] == pytest.approx(0.0)
    assert converted_t[-1] == pytest.approx(500e-6)
    assert np.all(converted_a >= 0.0)
    assert converted_a.max() < 1e-3


def test_different_units_sheet() -> None:
    rows = list(read_excel_sheet_rows(str(FIXTURE_PATH), "different_units"))
    first_row = rows[0]
    _name, payload = load_excel_dataset(str(FIXTURE_PATH), "different_units")

    assert looks_like_unit_row(list(first_row.values())) is True
    converted_t = payload["t"] * parse_time_unit(first_row["time"])
    converted_c = payload["species"]["C"] * parse_concentration_unit(first_row["C"])
    assert converted_t[0] == pytest.approx(0.0)
    assert converted_t[-1] == pytest.approx(500e-6)
    assert converted_c.max() < 1e-3


def test_no_unit_row_sheet() -> None:
    rows = list(read_excel_sheet_rows(str(FIXTURE_PATH), "no_unit_row"))
    _name, payload = load_excel_dataset(str(FIXTURE_PATH), "no_unit_row")

    assert looks_like_unit_row(list(rows[0].values())) is False
    assert payload["t"][0] == pytest.approx(0.0)
    assert payload["t"][-1] == pytest.approx(500e-6)
    assert payload["species"]["A"].max() < 1e-3


def test_extra_columns_filtered() -> None:
    _name, payload = load_excel_dataset(str(FIXTURE_PATH), "extra_columns")

    assert "notes" not in payload["species"]
    assert "temperature" not in payload["species"]
    assert {"A", "B", "C"}.issubset(payload["species"])


@pytest.mark.gui
def test_import_config_dialog_with_m9_workbook(qapp) -> None:
    dialog = ImportConfigDialog(str(FIXTURE_PATH))

    assert list_sheets(str(FIXTURE_PATH)) == [
        "pH7_run",
        "pH9_run",
        "different_units",
        "no_unit_row",
        "extra_columns",
        "different_structure",
        "parametric_x",
    ]
    assert dialog._sheet_names == list_sheets(str(FIXTURE_PATH))
    assert dialog._preview_table.rowCount() > 0
    assert dialog._unit_row_detected is True

    for index in range(dialog._sheet_list.count()):
        item = dialog._sheet_list.item(index)
        if item.text() == "no_unit_row":
            dialog._sheet_list.setCurrentItem(item)
            dialog._on_sheet_clicked(item)
            break

    assert dialog._unit_row_detected is False


def test_multi_sheet_load() -> None:
    result = load_excel_workbook(str(FIXTURE_PATH))

    assert len(result.successes) == 7
    assert result.failures == []
    assert {name for name, _ in result.successes} == {
        "m9_test_datasets.xlsx::pH7_run",
        "m9_test_datasets.xlsx::pH9_run",
        "m9_test_datasets.xlsx::different_units",
        "m9_test_datasets.xlsx::no_unit_row",
        "m9_test_datasets.xlsx::extra_columns",
        "m9_test_datasets.xlsx::different_structure",
        "m9_test_datasets.xlsx::parametric_x",
    }


def test_different_structure_sheet_detected() -> None:
    ph7_rows = list(read_excel_sheet_rows(str(FIXTURE_PATH), "pH7_run"))
    different_rows = list(read_excel_sheet_rows(str(FIXTURE_PATH), "different_structure"))

    assert list(ph7_rows[0].keys()) != list(different_rows[0].keys())
    assert list(different_rows[0].keys()) == ["time", "X", "Y"]


def test_parametric_x_sheet() -> None:
    _name, payload = load_excel_dataset(str(FIXTURE_PATH), "parametric_x")

    assert payload["metadata"]["time_column"] == "time"
    assert "observable_X" in payload["species"]
    assert {"A", "B"}.issubset(payload["species"])
    assert payload["species"]["observable_X"].shape == (50,)
    assert not np.allclose(payload["species"]["observable_X"], payload["t"])
