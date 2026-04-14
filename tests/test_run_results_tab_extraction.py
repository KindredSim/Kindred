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
        tab.set_run_stamp({"solver": "BDF"}, "abc123hash", "abc123")
        qt_app.processEvents()

        assert tab._last_run_stamp == {"solver": "BDF"}
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


def test_stamp_stats_sync_across_runs(qt_app):
    """set_run_stamp clears stale stats; dialog reflects new state."""
    tab = _make_tab()
    try:
        # Run A: stamp + stats
        tab.set_run_stamp({"run": "A"}, "hashA", "shortA")
        tab.update_statistics({"Datasets": 5, "SSQ": 2.0})
        assert tab._last_stats == {"Datasets": 5, "SSQ": 2.0}

        # Run B starts: stamp overwritten, stats must be cleared
        tab.set_run_stamp({"run": "B"}, "hashB", "shortB")
        assert tab._last_stats == {}  # old stats cleared

        # Open dialog during run B (before stats arrive)
        tab.open_results_summary_dialog()
        qt_app.processEvents()
        dialog = tab._stamp_dialog
        assert dialog is not None
        assert dialog._stamp_short == "shortB"
        # All stats show dash
        for label in dialog._stats_labels.values():
            assert label.text() == "\u2014"

        # Run B completes with new stats
        tab.update_statistics({"Datasets": 10, "SSQ": 0.5})
        qt_app.processEvents()
        assert dialog._stamp_short == "shortB"
        assert dialog._stats_labels["Datasets"].text() == "10"
        assert dialog._stats_labels["SSQ"].text() == "0.5"
    finally:
        if tab._stamp_dialog is not None:
            tab._stamp_dialog.close()
        tab.close()
        qt_app.processEvents()
