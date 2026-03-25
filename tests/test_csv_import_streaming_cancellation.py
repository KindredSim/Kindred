from __future__ import annotations

import os
from typing import Iterable

import pytest

from kindred.core.datasets.csv_import import CsvImportInterrupted, load_csv_dataset
from kindred.gui.widgets.data_manager import CSVLoaderWorker


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
    _patch_reader(monkeypatch, rows, state)

    worker = CSVLoaderWorker(str(csv_path))
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
