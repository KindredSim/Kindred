from __future__ import annotations

import numpy as np


def test_csv_time_s_alias_is_auto_detected(tmp_path):
    """
    Regression: accept `time_s` as a common time column alias when auto-detecting.
    """
    from kindred.core.datasets.csv_import import load_csv_dataset

    csv_path = tmp_path / "time_s_alias.csv"
    rows = ["time_s,A", "0.0,1.0", "1.0,0.5", "2.0,0.25"]
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    _name, payload = load_csv_dataset(str(csv_path))
    assert payload["metadata"]["time_column"] == "time_s"
    assert np.asarray(payload["t"], dtype=float).shape == (3,)
