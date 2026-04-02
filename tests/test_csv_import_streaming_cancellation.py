from __future__ import annotations

import os
from typing import Iterable

import pytest

from kindred.core.datasets.csv_import import CsvImportInterrupted, load_csv_dataset
from kindred.gui.widgets.data_manager import CSVLoaderWorker
from kindred.gui.widgets.import_config import ResolvedSheetPlan


class _StreamingReader:
    def __init__(self, rows: list[dict[str, str]], state: dict[str, int]) -> None:
        self._rows = rows
        self._state = state
        self._index = 0

    def __iter__(self) -> "_StreamingReader":
        return self

    def __next__(self) -> dict[str, str]:
        if self._index >= len(self._rows):
            raise StopIteration
        row = self._rows[self._index]
        self._index += 1
        self._state["yielded"] += 1
        return row


def _patch_reader(monkeypatch: pytest.MonkeyPatch, rows: Iterable[dict[str, str]], state: dict[str, int]) -> None:
    row_list = list(rows)

    def _factory(_handle):  # noqa: ANN001
        return _StreamingReader(list(row_list), state)

    monkeypatch.setattr("kindred.core.datasets.csv_import.csv.DictReader", _factory)


class _StreamingCsvReader:
    def __init__(self, rows: list[list[str]], state: dict[str, int]) -> None:
        self._rows = rows
        self._state = state
        self._index = 0

    def __iter__(self) -> "_StreamingCsvReader":
        return self

    def __next__(self) -> list[str]:
        if self._index >= len(self._rows):
            raise StopIteration
        row = self._rows[self._index]
        self._index += 1
        if self._index > 1:
            self._state["yielded"] += 1
        return row


def _patch_worker_reader(monkeypatch: pytest.MonkeyPatch, rows: Iterable[dict[str, str]], state: dict[str, int]) -> None:
    row_list = list(rows)
    header = list(row_list[0].keys()) if row_list else []
    csv_rows = [header, *[[row.get(column, "") for column in header] for row in row_list]]

    def _factory(_handle):  # noqa: ANN001
        return _StreamingCsvReader(list(csv_rows), state)

    monkeypatch.setattr("kindred.gui.widgets.data_manager.csv.reader", _factory)


def test_load_csv_dataset_cancellation_interrupts_during_row_iteration(tmp_path, monkeypatch):
    csv_path = tmp_path / "streaming.csv"
    csv_path.write_text("time,A\n0,1.0\n", encoding="utf-8")

    rows = [{"time": str(i), "A": str(float(i))} for i in range(5)]
    state = {"yielded": 0}
    _patch_reader(monkeypatch, rows, state)

    with pytest.raises(CsvImportInterrupted):
        load_csv_dataset(
            str(csv_path),
            interruption_checker=lambda: state["yielded"] >= 2,
        )

    assert state["yielded"] < len(rows)


def test_csv_loader_worker_cancellation_interrupts_before_full_read(tmp_path, monkeypatch):
    csv_path = tmp_path / "worker_streaming.csv"
    csv_path.write_text("time,A\n0,1.0\n", encoding="utf-8")

    rows = [{"time": str(i), "A": str(float(i))} for i in range(6)]
    state = {"yielded": 0}
    _patch_worker_reader(monkeypatch, rows, state)

    plan = ResolvedSheetPlan(
        filepath=str(csv_path), sheet_name=None,
        time_column="time", species_columns=("A",),
        skip_unit_row=False, time_factor=1.0, conc_factor=1.0,
        original_time_unit="s", original_conc_unit="M",
    )
    worker = CSVLoaderWorker(plan)
    cancelled: list[str] = []
    errors: list[str] = []
    loaded: list[tuple[str, dict]] = []
    worker.cancelled.connect(lambda name: cancelled.append(str(name)))
    worker.error.connect(lambda msg: errors.append(str(msg)))
    worker.finished.connect(lambda name, data: loaded.append((str(name), data)))
    monkeypatch.setattr(worker, "isInterruptionRequested", lambda: state["yielded"] >= 2)

    worker.run()

    assert cancelled == [os.path.basename(str(csv_path))]
    assert not errors
    assert not loaded
    assert state["yielded"] < len(rows)
