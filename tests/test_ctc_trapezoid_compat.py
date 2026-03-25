from __future__ import annotations

import numpy as np
import pytest

from kindred.gui.widgets import species_statistics_table as sst


pytestmark = [pytest.mark.gui]


def _ctc_cell_value(table: sst.SpeciesStatisticsTable) -> float:
    item = table.item(0, 5)
    assert item is not None
    return float(item.text())


def test_species_statistics_uses_numpy_trapezoid_when_trapz_missing(qt_app, monkeypatch):
    monkeypatch.delattr(sst.np, "trapz", raising=False)

    table = sst.SpeciesStatisticsTable()
    t = np.array([0.0, 1.0, 2.0], dtype=float)
    species_data = {"A": np.array([0.0, 1.0, 0.0], dtype=float)}

    table.update_results(t, species_data)

    assert _ctc_cell_value(table) == pytest.approx(1.0)


def test_species_statistics_falls_back_to_scipy_trapezoid(qt_app, monkeypatch):
    monkeypatch.setattr(sst.np, "trapezoid", None, raising=False)
    monkeypatch.setattr(sst.np, "trapz", None, raising=False)

    calls: list[tuple[np.ndarray, np.ndarray]] = []

    def _fake_scipy_trapezoid(y, x):
        calls.append((np.asarray(y, dtype=float), np.asarray(x, dtype=float)))
        return 2.5

    monkeypatch.setattr(sst, "_scipy_trapezoid", _fake_scipy_trapezoid, raising=False)

    table = sst.SpeciesStatisticsTable()
    t = np.array([0.0, 1.0, 2.0], dtype=float)
    species_data = {"A": np.array([0.0, 1.0, 0.0], dtype=float)}

    table.update_results(t, species_data)

    assert calls
    assert _ctc_cell_value(table) == pytest.approx(2.5)
