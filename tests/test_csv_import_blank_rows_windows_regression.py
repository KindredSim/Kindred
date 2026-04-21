from __future__ import annotations

import numpy as np
import pytest

from kindred.core.datasets.csv_import import parse_csv_rows, load_csv_dataset

pytestmark = pytest.mark.unit



def test_parse_csv_skips_fully_empty_rows() -> None:
    rows = [
        {"time": "0", "A": "1.0"},
        {"time": "", "A": ""},
        {"time": "   ", "A": "  "},
        {"time": "1", "A": "2.0"},
    ]

    _, payload = parse_csv_rows(rows)

    assert np.allclose(payload["t"], [0.0, 1.0])
    assert np.allclose(payload["species"]["A"], [1.0, 2.0])


def test_parse_csv_errors_on_missing_time_in_nonempty_row() -> None:
    rows = [
        {"time": "0", "A": "1.0"},
        {"time": "", "A": "2.0"},
    ]

    with pytest.raises(ValueError) as excinfo:
        parse_csv_rows(rows)

    message = str(excinfo.value)
    assert "Row 2" in message
    assert "time" in message
    assert "missing" in message.lower()


def test_utf8_sig_bom_in_header(tmp_path) -> None:
    csv_path = tmp_path / "bom_dataset.csv"
    csv_path.write_text("time,A\n0,1.0\n1,2.0\n", encoding="utf-8-sig")

    name, payload = load_csv_dataset(str(csv_path))

    assert name == "bom_dataset.csv"
    assert payload["metadata"]["time_column"] == "time"
    assert np.allclose(payload["t"], [0.0, 1.0])
    assert np.allclose(payload["species"]["A"], [1.0, 2.0])
