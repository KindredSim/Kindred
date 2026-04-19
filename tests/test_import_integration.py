from __future__ import annotations

from pathlib import Path

import pytest

from kindred.core.datasets.excel_import import read_excel_sheet_rows
from kindred.core.datasets.units import (
    looks_like_unit_row,
    parse_concentration_unit,
    parse_time_unit,
)


FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "m9_test_datasets.xlsx"


pytestmark = [pytest.mark.integration]


def test_unit_row_detected_on_ph7() -> None:
    rows = list(read_excel_sheet_rows(str(FIXTURE_PATH), "pH7_run"))
    first_row = rows[0]

    assert looks_like_unit_row(list(first_row.values())) is True
    assert parse_time_unit(first_row["time"]) == pytest.approx(1e-6)
    assert parse_concentration_unit(first_row["A"]) == pytest.approx(1e-6)
    assert parse_concentration_unit(first_row["PBMP"]) == pytest.approx(1e-6)


def test_different_structure_sheet_detected() -> None:
    ph7_rows = list(read_excel_sheet_rows(str(FIXTURE_PATH), "pH7_run"))
    different_rows = list(read_excel_sheet_rows(str(FIXTURE_PATH), "different_structure"))

    assert list(ph7_rows[0].keys()) != list(different_rows[0].keys())
    assert list(different_rows[0].keys()) == ["time", "X", "Y"]
