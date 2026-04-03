from __future__ import annotations

from pathlib import Path

import pytest

from kindred.core.datasets.excel_import import (
    list_sheets,
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
def test_unit_row_detected_on_ph7() -> None:
    rows = list(read_excel_sheet_rows(str(FIXTURE_PATH), "pH7_run"))
    first_row = rows[0]

    assert looks_like_unit_row(list(first_row.values())) is True
    assert parse_time_unit(first_row["time"]) == pytest.approx(1e-6)
    assert parse_concentration_unit(first_row["A"]) == pytest.approx(1e-6)
    assert parse_concentration_unit(first_row["PBMP"]) == pytest.approx(1e-6)
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
def test_different_structure_sheet_detected() -> None:
    ph7_rows = list(read_excel_sheet_rows(str(FIXTURE_PATH), "pH7_run"))
    different_rows = list(read_excel_sheet_rows(str(FIXTURE_PATH), "different_structure"))

    assert list(ph7_rows[0].keys()) != list(different_rows[0].keys())
    assert list(different_rows[0].keys()) == ["time", "X", "Y"]
