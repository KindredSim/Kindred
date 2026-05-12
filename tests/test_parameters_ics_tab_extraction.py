"""Standalone extraction tests for ParametersIcsTab."""
from __future__ import annotations

import pytest
from PySide6 import QtWidgets


pytestmark = [pytest.mark.gui]


def _make_tab(
    *,
    entries=None,
    species=None,
    integration_defaults=("BDF", 1e-6, 1e-12),
    reactions_text="",
):
    from kindred.gui.fitting.parameters_ics_tab import ParametersIcsTab

    if entries is None:
        entries = [{"id": "ds1", "label": "DS 1"}]
    if species is None:
        species = ["A", "B"]

    def initial_parameter_defaults_getter(_dataset_id, _species):
        return False, {"initial": 0.0, "min": 0.0, "max": 10.0, "log10": False}

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
        reactions_text_getter=lambda: str(reactions_text),
        integration_defaults=integration_defaults,
        config_defaults={},
        initial_parameter_defaults_getter=initial_parameter_defaults_getter,
    )
    return tab


def test_construction(qt_app):
    """ParametersIcsTab builds expected widget hierarchy."""
    tab = _make_tab()
    try:
        assert tab._param_table is not None
        assert isinstance(tab._param_table, QtWidgets.QTableWidget)
        assert not hasattr(tab, "_ic_panel")
        assert not hasattr(tab, "_ic_dataset_combo")
    finally:
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


def testcollect_parameter_config_snapshot_for_readiness_reads_table_without_mutating_state(qt_app):
    tab = _make_tab()
    try:
        tab.set_parameter_state([
            {
                "scope": "shared",
                "name": "k",
                "param_name": "k",
                "value": 1.0,
                "min": 0.1,
                "max": 10.0,
                "fit": True,
                "log10": False,
                "last_fit": None,
            }
        ])
        tab._populate_parameter_table()
        tab._param_table.item(0, 3).setText("2.5")
        before_state = tab.get_parameter_state()

        snapshot = tab.collect_parameter_config_snapshot_for_readiness()

        assert snapshot is not None
        config, dataset_params, dataset_variable_params = snapshot
        assert config["parameters"] == {"k": 2.5}
        assert config["bounds"] == {"k": (0.1, 10.0)}
        assert config["fixed_params"] == {}
        assert dataset_params == {}
        assert dataset_variable_params == {}
        assert tab.get_parameter_state() == before_state
    finally:
        tab.close()
        qt_app.processEvents()


def test_rebuild_uses_initial_parameter_defaults_provider(qt_app):
    """Parameter rows use the bounded IC default provider instead of a widget reach-through."""
    from kindred.gui.fitting.parameters_ics_tab import ParametersIcsTab

    entries = [{"id": "ds1", "label": "DS 1"}]

    def initial_parameter_defaults_getter(dataset_id, species):
        assert dataset_id == "ds1"
        assert species == "A"
        return True, {"initial": 2.5, "min": 0.1, "max": 5.0, "log10": False}

    tab = ParametersIcsTab(
        parameter_state=[],
        initial_parameter_snapshot=[],
        global_dataset_params={},
        global_dataset_variable_params={},
        fixed_shared_params={},
        shared_param_definitions={},
        mechanism_species=["A"],
        dataset_entries=list(entries),
        prepared_param_names=[],
        selected_dataset_ids_getter=lambda: ["ds1"],
        dataset_entries_getter=lambda: list(entries),
        worker_running_getter=lambda: False,
        dataset_manager_getter=lambda: None,
        reactions_text_getter=lambda: "reaction: A -> B; k=1\ninitial: A=1\ninitial: B=0",
        integration_defaults=("BDF", 1e-6, 1e-12),
        config_defaults={},
        initial_parameter_defaults_getter=initial_parameter_defaults_getter,
    )
    try:
        tab.rebuild_for_mechanism("reaction: A -> B; k=1\ninitial: A=1\ninitial: B=0", entries)
        variable = tab.get_global_dataset_variable_params()
        assert variable["ds1"]["init:A"] == {"initial": 2.5, "min": 0.1, "max": 5.0, "log10": False}
    finally:
        tab.close()
        qt_app.processEvents()


def test_ic_applied_handler_updates_parameter_state(qt_app):
    """IC apply payload -> ParametersIcsTab handler -> parameter state updated."""
    ds_id = "ds1"
    species = ["A", "B"]

    tab = _make_tab(
        entries=[{"id": ds_id, "label": "DS 1"}],
        species=species,
    )
    try:
        tab._on_ic_applied(
            ds_id,
            {"A": {"initial": 5.0, "min": 0.1, "max": 100.0, "log10": False}},
            {"A": True, "B": False},
        )
        init_a_rows = [
            r for r in tab._parameter_state
            if r.get("param_name") == "init:A" and r.get("dataset_id") == ds_id
        ]
        assert len(init_a_rows) == 1
        assert init_a_rows[0]["fit"] is True
        assert init_a_rows[0]["value"] == 5.0
    finally:
        tab.close()
        qt_app.processEvents()


def test_view_steps_dialog_lists_mechanism_steps_read_only(qt_app, qtbot):
    reactions_text = "\n".join(
        [
            "reaction: A + OH -> AO; k=1.0",
            "equilibrium: C + O <-> CO; kf=2.0; kr=0.5",
            "reaction: P -> Q; k=0.2",
            "init: A=1, OH=1, AO=0, C=1, O=1, CO=0, P=1, Q=0",
        ]
    )
    tab = _make_tab(
        species=["A", "OH", "AO", "C", "O", "CO", "P", "Q"],
        reactions_text=reactions_text,
    )
    try:
        before_state = tab.get_parameter_state()
        tab._show_mechanism_steps_dialog()
        dialog = tab._last_steps_dialog
        qtbot.addWidget(dialog)

        assert dialog.windowTitle() == "Mechanism Steps"
        text = dialog.findChild(QtWidgets.QPlainTextEdit)
        assert text is not None
        assert text.isReadOnly() is True
        body = text.toPlainText()
        assert "Step 1    A + OH -> AO" in body
        assert "Step 2    C + O <-> CO" in body
        assert "Step 3    P -> Q" in body
        assert "kN, kfN, krN, and KeqN refer to Step N" in body
        assert tab.get_parameter_state() == before_state
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
