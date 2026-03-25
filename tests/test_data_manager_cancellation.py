from __future__ import annotations

from pathlib import Path

import pytest
from PySide6 import QtWidgets
from PySide6.QtTest import QSignalSpy

from kindred.gui.widgets.data_manager import DataManagerPanel

pytestmark = pytest.mark.gui


def _write_csv(path: Path, rows: int = 20000) -> None:
    """Create a simple CSV file with predictable numeric data."""
    with path.open("w", encoding="utf-8") as handle:
        handle.write("time,A,B\n")
        for i in range(rows):
            handle.write(f"{i},{i * 0.1},{i * 0.2}\n")


def test_multi_file_import_cancel_cleans_workers(tmp_path, monkeypatch, qtbot):
    """Cancel multi-file import and ensure threads/workers clean up properly."""
    panel = DataManagerPanel()
    qtbot.addWidget(panel)

    files = []
    for idx in range(2):
        csv_path = tmp_path / f"dataset_{idx}.csv"
        _write_csv(csv_path, rows=20000)
        files.append(str(csv_path))

    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getOpenFileNames",
        lambda *args, **kwargs: (files, ""),
    )

    finished_spy = QSignalSpy(panel.loadFinished)

    panel._load_dataset()
    panel._on_load_canceled()

    qtbot.waitUntil(lambda: finished_spy.count() == 1, timeout=7000)

    assert finished_spy.count() == 1
    assert bool(finished_spy.at(0)[0]) is True
    assert not panel._csv_workers
    assert panel._progress_dialog is None
    assert panel._pending_files_count == 0
    assert panel._completed_files_count == 0
