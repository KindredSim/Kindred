"""Standalone extraction tests for ParametersIcsTab."""
from __future__ import annotations

import pytest
from PySide6 import QtWidgets


pytestmark = [pytest.mark.gui]


def _make_tab(*, entries=None, species=None, integration_defaults=("LSODA", 1e-6, 1e-12)):
    from kindred.gui.fitting.initial_conditions_panel import InitialConditionsPanel
    from kindred.gui.fitting.parameters_ics_tab import ParametersIcsTab

    if entries is None:
        entries = [{"id": "ds1", "label": "DS 1"}]
    if species is None:
        species = ["A", "B"]

    ic_panel = InitialConditionsPanel(
        dataset_entries=list(entries),
        mechanism_species=list(species),
        dataset_manager_getter=lambda: None,
        worker_running_getter=lambda: False,
    )
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
        ic_panel=ic_panel,
    )
    return tab


def test_construction(qt_app):
    """ParametersIcsTab builds expected widget hierarchy."""
    tab = _make_tab()
    try:
        assert tab._param_table is not None
        assert isinstance(tab._param_table, QtWidgets.QTableWidget)

        # IC widgets are accessible via _ic_panel reference (no longer children of tab)
        assert tab._ic_panel is not None
        assert tab._ic_table is not None
        assert tab._ic_dataset_combo is not None
    finally:
        tab._ic_panel.close()
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


def test_ic_panel_is_standalone_widget(qt_app):
    """InitialConditionsPanel is a standalone widget referenced by ParametersIcsTab."""
    from kindred.gui.fitting.initial_conditions_panel import InitialConditionsPanel

    tab = _make_tab()
    try:
        assert hasattr(tab, "_ic_panel")
        assert isinstance(tab._ic_panel, InitialConditionsPanel)
        # IC panel is no longer a child of the tab — it is hosted externally
        assert not tab.isAncestorOf(tab._ic_panel)
    finally:
        tab._ic_panel.close()
        tab.close()
        qt_app.processEvents()


def test_ic_panel_apply_signal_round_trip(qt_app):
    """IC edit -> Apply -> icApplied signal -> handler -> parameter state updated."""
    from types import SimpleNamespace

    ds_id = "ds1"
    species = ["A", "B"]

    # Fake fit_settings with initial conditions
    fake_settings = SimpleNamespace(
        initial_conditions={"A": 1.0, "B": 0.0},
        fit_flags={"A": False, "B": False},
        log10_flags={},
        bounds={},
    )

    class FakeManager:
        def get_fit_settings(self, _ds_id):
            return fake_settings

        def update_fit_settings(self, _ds_id, _settings):
            pass

    tab = _make_tab(
        entries=[{"id": ds_id, "label": "DS 1"}],
        species=species,
    )
    # Wire icApplied -> _on_ic_applied (done by FittingWindow in production)
    tab._ic_panel.icApplied.connect(tab._on_ic_applied)
    # Replace dataset_manager_getter to return our fake manager
    tab._dataset_manager_getter = lambda: FakeManager()
    tab._ic_panel._dataset_manager_getter = lambda: FakeManager()
    # Reload IC table so it picks up the fake settings
    tab._ic_panel._load_initial_conditions_for_current_dataset()
    try:
        from PySide6.QtCore import Qt
        from kindred.gui.fitting.initial_conditions_panel import _ICCol

        ic_table = tab._ic_panel._ic_table
        assert ic_table.rowCount() == 2

        # Edit species A: set fit=True, initial=5.0, min=0.1, max=100
        fit_item = ic_table.item(0, _ICCol.FIT)
        fit_item.setCheckState(Qt.Checked)
        ic_table.item(0, _ICCol.INITIAL).setText("5.0")
        ic_table.item(0, _ICCol.MIN).setText("0.1")
        ic_table.item(0, _ICCol.MAX).setText("100")

        # Track signal emissions
        applied_args = []
        tab._ic_panel.icApplied.connect(lambda *args: applied_args.append(args))

        # Click Apply
        tab._ic_panel._apply_initial_conditions_changes()

        # Verify icApplied signal fired
        assert len(applied_args) == 1
        sig_ds_id, sig_updates, sig_fit_flags = applied_args[0]
        assert sig_ds_id == ds_id
        assert "A" in sig_updates
        assert sig_updates["A"]["initial"] == 5.0
        assert sig_updates["A"]["min"] == 0.1
        assert sig_updates["A"]["max"] == 100.0
        assert sig_fit_flags["A"] is True
        assert sig_fit_flags["B"] is False

        # Verify parameter state was updated via _on_ic_applied handler
        init_a_rows = [
            r for r in tab._parameter_state
            if r.get("param_name") == "init:A" and r.get("dataset_id") == ds_id
        ]
        assert len(init_a_rows) == 1
        assert init_a_rows[0]["fit"] is True
        assert init_a_rows[0]["value"] == 5.0

        # Verify dirty flag cleared
        assert tab._ic_panel._ic_editor_dirty is False
    finally:
        tab.close()
        qt_app.processEvents()
