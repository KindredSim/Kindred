"""Standalone extraction tests for RunResultsTab."""
from __future__ import annotations

import pytest
from PySide6 import QtWidgets


pytestmark = [pytest.mark.gui]


def _make_tab():
    from kindred.gui.fitting.run_results_tab import RunResultsTab

    return RunResultsTab(parent=None)


def test_construction(qt_app):
    """RunResultsTab builds expected widget hierarchy (no Run Stamp GroupBox)."""
    tab = _make_tab()
    try:
        groups = tab.findChildren(QtWidgets.QGroupBox)
        stamp_groups = [g for g in groups if g.title() == "Run Stamp"]
        assert len(stamp_groups) == 0

        assert hasattr(tab, "open_results_summary_dialog")
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
    """set_run_stamp stores stamp data internally."""
    tab = _make_tab()
    try:
        tab.set_run_stamp({"solver": "LSODA"}, "abc123hash", "abc123")
        qt_app.processEvents()

        assert tab._last_run_stamp == {"solver": "LSODA"}
        assert tab._last_run_stamp_hash == "abc123hash"
        assert tab._last_run_stamp_short == "abc123"
    finally:
        tab.close()
        qt_app.processEvents()


def test_update_statistics(qt_app):
    """update_statistics stores values internally."""
    tab = _make_tab()
    try:
        tab.update_statistics({"Datasets": 3, "SSQ": 1.5, "Points": 100})
        qt_app.processEvents()

        assert tab._last_stats["Datasets"] == 3
        assert tab._last_stats["SSQ"] == 1.5
        assert tab._last_stats["Points"] == 100
    finally:
        tab.close()
        qt_app.processEvents()
