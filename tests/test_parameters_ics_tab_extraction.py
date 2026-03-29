"""Standalone extraction tests for ParametersIcsTab."""
from __future__ import annotations

import pytest
from PySide6 import QtWidgets


pytestmark = [pytest.mark.gui]


def _make_tab(*, entries=None, species=None, integration_defaults=("LSODA", 1e-6, 1e-12)):
    from kindred.gui.fitting.parameters_ics_tab import ParametersIcsTab

    if entries is None:
        entries = [{"id": "ds1", "label": "DS 1"}]
    if species is None:
        species = ["A", "B"]

    tab = ParametersIcsTab(
        parameter_state=[],
        initial_parameter_snapshot=[],
        global_dataset_params={},
        global_dataset_variable_params={},
        fixed_shared_params={},
        shared_param_definitions={},
        mechanism_species=list(species),
        dataset_entries=list(entries),
        prepared_param_names=[],
        selected_dataset_ids_getter=lambda: [str(e["id"]) for e in entries],
        dataset_entries_getter=lambda: list(entries),
        worker_running_getter=lambda: False,
        dataset_manager_getter=lambda: None,
        reactions_text_getter=lambda: "",
        integration_defaults=integration_defaults,
        config_defaults={},
    )
    return tab


def test_construction(qt_app):
    """ParametersIcsTab builds expected widget hierarchy."""
    tab = _make_tab()
    try:
        assert tab._param_table is not None
        assert isinstance(tab._param_table, QtWidgets.QTableWidget)

        ic_table = tab.findChild(QtWidgets.QTableWidget, "global_fit_initial_conditions_table")
        assert ic_table is not None

        ic_combo = tab.findChild(QtWidgets.QComboBox, "global_fit_initial_conditions_dataset_combo")
        assert ic_combo is not None
    finally:
        tab.close()
        qt_app.processEvents()


def test_signals_defined(qt_app):
    """ParametersIcsTab exposes expected signals."""
    tab = _make_tab()
    try:
        received = []
        tab.addAlgebraicObservableRequested.connect(lambda payload: received.append(payload))
        tab.statusMessage.connect(lambda msg: received.append(msg))
        tab.statusMessage.emit("test")
        assert "test" in received
    finally:
        tab.close()
        qt_app.processEvents()


def test_ic_dataset_combo_population(qt_app):
    """IC dataset combo has one item per dataset entry."""
    entries = [
        {"id": "ds1", "label": "DS 1"},
        {"id": "ds2", "label": "DS 2"},
    ]
    tab = _make_tab(entries=entries)
    try:
        assert tab._ic_dataset_combo.count() == 2
    finally:
        tab.close()
        qt_app.processEvents()


def test_parameter_table_initially_empty(qt_app):
    """With empty parameter_state, param table has zero rows."""
    tab = _make_tab()
    try:
        assert tab._param_table.rowCount() == 0
    finally:
        tab.close()
        qt_app.processEvents()


def test_state_getters_return_initial_values(qt_app):
    """State getter API returns values matching constructor inputs."""
    tab = _make_tab(species=["X", "Y"])
    try:
        assert tab.get_parameter_state() == []
        assert tab.get_initial_parameter_snapshot() == []
        assert tab.get_global_dataset_params() == {}
        assert tab.get_fixed_shared_params() == {}
        assert tab.get_mechanism_species() == ["X", "Y"]
        assert tab.get_prepared_param_names() == []
    finally:
        tab.close()
        qt_app.processEvents()
