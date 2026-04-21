from __future__ import annotations

from pathlib import Path

import pytest

from kindred.core.datasets.excel_import import list_sheets
from kindred.gui.widgets.import_config_dialog import ImportConfigDialog

pytestmark = pytest.mark.gui

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "m9_test_datasets.xlsx"


@pytest.mark.integration
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
