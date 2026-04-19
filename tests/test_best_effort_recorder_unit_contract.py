from __future__ import annotations

import logging

import pytest

from kindred.gui.main_window import MainWindow

pytestmark = [pytest.mark.unit]


class _MainWindowRecorderHost:
    pass


def test_main_window_best_effort_wrapper_matches_shared_helper_contract(caplog) -> None:
    host = _MainWindowRecorderHost()
    caplog.set_level(logging.DEBUG, logger="kindred.gui.main_window")

    for _ in range(4):
        MainWindow._record_best_effort_failure(
            host,
            "main.key",
            message="MainWindow best effort",
        )

    assert host._best_effort_failures == {"main.key"}
    assert host._best_effort_failure_counts == {"main.key": 4}
    messages = [record.getMessage() for record in caplog.records if record.name == "kindred.gui.main_window"]
    assert messages == [
        "MainWindow best effort (key=main.key count=1)",
        "MainWindow best effort (key=main.key count=2)",
        "MainWindow best effort (key=main.key count=3)",
    ]
