"""Standalone extraction tests for DataTab."""
from __future__ import annotations

import numpy as np
import pytest
from PySide6 import QtWidgets


pytestmark = [pytest.mark.gui]


def _make_tab(*, worker_running=False):
    from kindred.gui.fitting.data_tab import DataTab

    empty_cfg = {
        "t_min": 0.0, "t_max": 1.0, "n_points": 10,
        "x_name": "t", "x_mapping_mode": "auto",
    }
    tab = DataTab(
        sampling_applied_config_getter=lambda ds_id: dict(empty_cfg),
        sampling_default_config_getter=lambda t: dict(empty_cfg),
        fit_targets_full_t_getter=lambda ds_id: np.linspace(0.0, 1.0, 6),
        fit_targets_available_getter=lambda ds_id: ["A", "B"],
        fit_targets_full_series_getter=lambda ds_id: {},
        fit_targets_selection_applied_getter=lambda ds_id: [],
        modeled_series_getter=lambda: set(),
        worker_running_getter=lambda: worker_running,
    )
    return tab


def test_construction(qt_app):
    """DataTab builds expected widget hierarchy."""
    tab = _make_tab()
    try:
        assert tab._dataset_table is not None
        assert isinstance(tab._dataset_table, QtWidgets.QTableWidget)

        add_btn = tab.findChild(QtWidgets.QPushButton, "global_fit_datasets_add")
        assert add_btn is not None

        remove_btn = tab.findChild(QtWidgets.QPushButton, "global_fit_datasets_remove")
        assert remove_btn is not None

        sampling_panel = tab.findChild(QtWidgets.QWidget, "global_fit_sampling_panel")
        assert sampling_panel is not None
    finally:
        tab.close()
        qt_app.processEvents()


def test_signals_defined(qt_app):
    """DataTab exposes all expected signals."""
    tab = _make_tab()
    try:
        for signal_name in [
            "datasetIncludeChanged",
            "addDatasetsRequested",
            "removeDatasetsRequested",
            "samplingApplied",
            "statusMessage",
        ]:
            sig = getattr(tab, signal_name, None)
            assert sig is not None, f"Signal {signal_name} not found"
    finally:
        tab.close()
        qt_app.processEvents()


def test_selected_dataset_id(qt_app):
    """selected_dataset_id returns correct ID for the selected row."""
    tab = _make_tab()
    try:
        entries = [
            {"id": "ds1", "label": "DS 1", "selected_species": ["A"], "include": True},
            {"id": "ds2", "label": "DS 2", "selected_species": ["B"], "include": True},
        ]
        tab.populate_table(entries)
        qt_app.processEvents()

        tab._dataset_table.selectRow(1)
        qt_app.processEvents()

        assert tab.selected_dataset_id() == "ds2"
    finally:
        tab.close()
        qt_app.processEvents()


def test_included_dataset_ids(qt_app):
    """included_dataset_ids returns IDs of checked rows only."""
    tab = _make_tab()
    try:
        entries = [
            {"id": "ds1", "label": "DS 1", "selected_species": ["A"], "include": True},
            {"id": "ds2", "label": "DS 2", "selected_species": ["B"], "include": False},
        ]
        tab.populate_table(entries)
        qt_app.processEvents()

        ids = tab.included_dataset_ids()
        assert "ds1" in ids
        assert "ds2" not in ids
    finally:
        tab.close()
        qt_app.processEvents()


def test_populate_table(qt_app):
    """populate_table sets the correct row count."""
    tab = _make_tab()
    try:
        entries = [
            {"id": "ds1", "label": "DS 1", "selected_species": ["A"], "include": True},
            {"id": "ds2", "label": "DS 2", "selected_species": ["B"], "include": True},
            {"id": "ds3", "label": "DS 3", "selected_species": ["A", "B"], "include": True},
        ]
        tab.populate_table(entries)
        qt_app.processEvents()

        assert tab._dataset_table.rowCount() == 3
    finally:
        tab.close()
        qt_app.processEvents()
