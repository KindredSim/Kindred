"""Standalone extraction tests for RunResultsTab."""
from __future__ import annotations

import pytest
from PySide6 import QtWidgets


pytestmark = [pytest.mark.gui]


def _make_tab():
    from kindred.gui.fitting.run_results_tab import RunResultsTab

    return RunResultsTab(parent=None)


def test_construction(qt_app):
    """RunResultsTab builds expected widget hierarchy."""
    tab = _make_tab()
    try:
        stamp_label = tab.findChild(QtWidgets.QLabel, "global_fit_run_stamp_label")
        assert stamp_label is not None

        copy_btn = tab.findChild(QtWidgets.QPushButton, "global_fit_copy_stamp_button")
        assert copy_btn is not None

        copy_json_btn = tab.findChild(QtWidgets.QPushButton, "global_fit_copy_stamp_json_button")
        assert copy_json_btn is not None

        assert len(tab._stats_labels) == 8
    finally:
        tab.close()
        qt_app.processEvents()


def test_signals_defined(qt_app):
    """RunResultsTab exposes statusMessage signal."""
    tab = _make_tab()
    try:
        received = []
        tab.statusMessage.connect(lambda msg: received.append(msg))
        tab.statusMessage.emit("test")
        assert received == ["test"]
    finally:
        tab.close()
        qt_app.processEvents()


def test_set_run_stamp(qt_app):
    """set_run_stamp populates label and enables copy buttons."""
    tab = _make_tab()
    try:
        assert not tab._copy_stamp_button.isEnabled()
        assert not tab._copy_stamp_json_button.isEnabled()

        tab.set_run_stamp({"solver": "LSODA"}, "abc123hash", "abc123")
        qt_app.processEvents()

        assert tab._run_stamp_label.text() == "Stamp: abc123"
        assert tab._copy_stamp_button.isEnabled()
        assert tab._copy_stamp_json_button.isEnabled()
    finally:
        tab.close()
        qt_app.processEvents()


def test_update_statistics(qt_app):
    """update_statistics writes values into stat labels."""
    tab = _make_tab()
    try:
        tab.update_statistics({"Datasets": 3, "SSQ": 1.5, "Points": 100})
        qt_app.processEvents()

        assert tab._stats_labels["Datasets"].text() == "3"
        assert tab._stats_labels["SSQ"].text() == "1.5"
        assert tab._stats_labels["Points"].text() == "100"
        assert tab._stats_labels["DF"].text() == "\u2014"
    finally:
        tab.close()
        qt_app.processEvents()
