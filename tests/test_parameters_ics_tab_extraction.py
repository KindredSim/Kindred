"""Standalone extraction tests for ParametersIcsTab."""
from __future__ import annotations

import pytest
from PySide6 import QtWidgets


pytestmark = [pytest.mark.gui]


def _make_tab(*, entries=None, species=None, integration_defaults=("BDF", 1e-6, 1e-12)):
    from kindred.gui.fitting.unified_species_table import UnifiedSpeciesTable
    from kindred.gui.fitting.parameters_ics_tab import ParametersIcsTab

    if entries is None:
        entries = [{"id": "ds1", "label": "DS 1"}]
    if species is None:
        species = ["A", "B"]

    species_table = UnifiedSpeciesTable(
        dataset_entries=list(entries),
        mechanism_species=list(species),
        dataset_entries_getter=lambda: list(entries),
        included_dataset_ids_getter=lambda: [str(e["id"]) for e in entries],
        dataset_label_getter=lambda ds_id: str(ds_id),
        dataset_weight_getter=lambda ds_id: 1.0,
        persist_dataset_weight_callback=lambda ds_id, w: None,
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
        ic_panel=species_table,
    )
    return tab


def test_construction(qt_app):
    """ParametersIcsTab builds expected widget hierarchy."""
    tab = _make_tab()
    try:
        assert tab._param_table is not None
        assert isinstance(tab._param_table, QtWidgets.QTableWidget)

        assert tab._ic_panel is not None
        assert getattr(tab._ic_panel, "_table", None) is not None
        # _ic_dataset_combo is None for UnifiedSpeciesTable (no combo)
        assert tab._ic_dataset_combo is None
    finally:
        tab._ic_panel.close()
        tab.close()
        qt_app.processEvents()


def test_signals_defined(qt_app):
    """ParametersIcsTab exposes expected signals."""
    tab = _make_tab()
    try:
        received = []
        tab.statusMessage.connect(lambda msg: received.append(msg))
        tab.statusMessage.emit("hello")
        assert received == ["hello"]
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
    """UnifiedSpeciesTable is a standalone widget referenced by ParametersIcsTab."""
    from kindred.gui.fitting.unified_species_table import UnifiedSpeciesTable

    tab = _make_tab()
    try:
        assert hasattr(tab, "_ic_panel")
        assert isinstance(tab._ic_panel, UnifiedSpeciesTable)
        # Species table is no longer a child of the tab
        assert not tab.isAncestorOf(tab._ic_panel)
    finally:
        tab._ic_panel.close()
        tab.close()
        qt_app.processEvents()


def test_ic_panel_apply_signal_round_trip(qt_app):
    """IC edit -> Apply -> icApplied signal -> handler -> parameter state updated."""
    from types import SimpleNamespace
    from kindred.gui.fitting.unified_species_table import _Col

    ds_id = "ds1"
    species = ["A", "B"]

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
    # Load dataset so the table is populated
    tab._ic_panel.load_for_dataset(ds_id)
    try:
        from PySide6.QtCore import Qt

        ic_table = tab._ic_panel._table
        assert ic_table.rowCount() == 2

        # Edit species A: set fit=True, initial=5.0, min=0.1, max=100
        fit_item = ic_table.item(0, _Col.FIT_IC)
        fit_item.setCheckState(Qt.Checked)
        ic_table.item(0, _Col.INITIAL).setText("5.0")
        ic_table.item(0, _Col.MIN).setText("0.1")
        ic_table.item(0, _Col.MAX).setText("100")

        # Track signal emissions
        applied_args = []
        tab._ic_panel.icApplied.connect(lambda *args: applied_args.append(args))

        # Click Apply (combined apply — only IC dirty, targets not dirty)
        tab._ic_panel._apply_changes()

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


def test_restore_failed_fit_state_restores_pre_run_values_and_clears_last_fit(qt_app):
    tab = _make_tab(
        entries=[{"id": "ds1", "label": "DS 1"}],
        species=["A"],
    )
    try:
        pre_run_state = [
            {
                "name": "k1",
                "param_name": "k1",
                "scope": "shared",
                "dataset_id": "",
                "fit": True,
                "value": 1.25,
                "last_fit": None,
                "min": 0.0,
                "max": 10.0,
                "log10": False,
            },
            {
                "name": "init:A (DS 1)",
                "param_name": "init:A",
                "scope": "dataset",
                "dataset_id": "ds1",
                "fit": True,
                "value": 2.0,
                "last_fit": None,
                "min": 0.0,
                "max": 10.0,
                "log10": False,
            },
        ]
        tab.set_parameter_state(pre_run_state)
        tab.set_last_fit_params({"k1": 9.0})
        tab.set_staged_dataset_params({"ds1": {"init:A": 9.0}})
        tab.push_best_update({"k1": 4.0}, {"ds1": {"init:A": 5.0}})

        tab._restore_failed_fit_state(pre_run_state, {"ds1": {"init:A": 2.0}})

        restored_state = tab.get_parameter_state()
        assert [row["value"] for row in restored_state] == [1.25, 2.0]
        assert all(row["last_fit"] is None for row in restored_state)
        assert tab.get_last_fit_params() == {}
        assert tab.get_staged_dataset_params() == {"ds1": {"init:A": 2.0}}
    finally:
        tab.close()
        qt_app.processEvents()


def test_restore_failed_fit_state_without_baseline_clears_live_fit_markers_only(qt_app):
    tab = _make_tab(
        entries=[{"id": "ds1", "label": "DS 1"}],
        species=["A"],
    )
    try:
        current_state = [
            {
                "name": "k1",
                "param_name": "k1",
                "scope": "shared",
                "dataset_id": "",
                "fit": True,
                "value": 4.0,
                "last_fit": 4.0,
                "min": 0.0,
                "max": 10.0,
                "log10": False,
            },
            {
                "name": "init:A (DS 1)",
                "param_name": "init:A",
                "scope": "dataset",
                "dataset_id": "ds1",
                "fit": True,
                "value": 5.0,
                "last_fit": 5.0,
                "min": 0.0,
                "max": 10.0,
                "log10": False,
            },
        ]
        tab.set_parameter_state(current_state)
        tab.set_last_fit_params({"k1": 4.0})
        tab.set_staged_dataset_params({"ds1": {"init:A": 5.0}})

        tab._restore_failed_fit_state(None, None)

        restored_state = tab.get_parameter_state()
        assert [row["value"] for row in restored_state] == [4.0, 5.0]
        assert all(row["last_fit"] is None for row in restored_state)
        assert tab.get_last_fit_params() == {}
        assert tab.get_staged_dataset_params() == {}
    finally:
        tab.close()
        qt_app.processEvents()


def test_restore_failed_fit_state_ignores_partial_baseline_and_avoids_split_restore(qt_app):
    tab = _make_tab(
        entries=[{"id": "ds1", "label": "DS 1"}],
        species=["A"],
    )
    try:
        current_state = [
            {
                "name": "k1",
                "param_name": "k1",
                "scope": "shared",
                "dataset_id": "",
                "fit": True,
                "value": 4.0,
                "last_fit": 4.0,
                "min": 0.0,
                "max": 10.0,
                "log10": False,
            },
            {
                "name": "init:A (DS 1)",
                "param_name": "init:A",
                "scope": "dataset",
                "dataset_id": "ds1",
                "fit": True,
                "value": 5.0,
                "last_fit": 5.0,
                "min": 0.0,
                "max": 10.0,
                "log10": False,
            },
        ]
        tab.set_parameter_state(current_state)
        tab.set_last_fit_params({"k1": 4.0})
        tab.set_staged_dataset_params({"ds1": {"init:A": 5.0}})

        tab._restore_failed_fit_state(None, {"ds1": {"init:A": 2.0}})

        restored_state = tab.get_parameter_state()
        assert [row["value"] for row in restored_state] == [4.0, 5.0]
        assert all(row["last_fit"] is None for row in restored_state)
        assert tab.get_last_fit_params() == {}
        assert tab.get_staged_dataset_params() == {}
    finally:
        tab.close()
        qt_app.processEvents()


def test_restore_failed_fit_state_ignores_parameter_only_baseline_when_staged_state_is_live(qt_app):
    tab = _make_tab(
        entries=[{"id": "ds1", "label": "DS 1"}],
        species=["A"],
    )
    try:
        pre_run_state = [
            {
                "name": "k1",
                "param_name": "k1",
                "scope": "shared",
                "dataset_id": "",
                "fit": True,
                "value": 1.25,
                "last_fit": None,
                "min": 0.0,
                "max": 10.0,
                "log10": False,
            },
            {
                "name": "init:A (DS 1)",
                "param_name": "init:A",
                "scope": "dataset",
                "dataset_id": "ds1",
                "fit": True,
                "value": 2.0,
                "last_fit": None,
                "min": 0.0,
                "max": 10.0,
                "log10": False,
            },
        ]
        current_state = [
            dict(pre_run_state[0], value=4.0, last_fit=4.0),
            dict(pre_run_state[1], value=5.0, last_fit=5.0),
        ]
        tab.set_parameter_state(current_state)
        tab.set_last_fit_params({"k1": 4.0})
        tab.set_staged_dataset_params({"ds1": {"init:A": 5.0}})

        tab._restore_failed_fit_state(pre_run_state, None)

        restored_state = tab.get_parameter_state()
        assert [row["value"] for row in restored_state] == [4.0, 5.0]
        assert all(row["last_fit"] is None for row in restored_state)
        assert tab.get_last_fit_params() == {}
        assert tab.get_staged_dataset_params() == {}
    finally:
        tab.close()
        qt_app.processEvents()
