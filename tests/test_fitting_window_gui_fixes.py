"""Regression tests for fitting window GUI layout fixes."""
from __future__ import annotations

import hashlib
import inspect

import numpy as np
import pytest
from PySide6 import QtWidgets
from PySide6.QtCore import Qt


pytestmark = [pytest.mark.gui]


# ---- F3: add_dataset_state delegates to _recompute_fit_universe ----

def test_add_dataset_state_delegates_to_recompute_fit_universe(qt_app, monkeypatch):
    """add_dataset_state must delegate fit-universe computation to
    _recompute_fit_universe (ARCH_RULES F3), not write
    _fit_targets_available_by_dataset inline."""
    from kindred.gui.fitting.unified_species_table import UnifiedSpeciesTable

    t = np.linspace(0, 1, 5)
    entries = [{
        "id": "ds1", "label": "DS 1", "t": t,
        "species_data": {"A": np.ones(5), "B": np.ones(5)},
        "selected_species": ["A", "B"], "weight": 1.0, "include": True,
    }]
    species = ["A", "B", "C"]
    modeled = {"A", "B", "C"}

    kwargs = dict(
        dataset_entries=entries,
        mechanism_species=species,
        dataset_entries_getter=lambda: entries,
        included_dataset_ids_getter=lambda: ["ds1"],
        dataset_label_getter=lambda ds_id: str(ds_id),
        dataset_weight_getter=lambda ds_id: 1.0,
        persist_dataset_weight_callback=lambda ds_id, w: None,
        dataset_manager_getter=lambda: None,
        worker_running_getter=lambda: False,
    )
    if "modeled_series_getter" in inspect.signature(UnifiedSpeciesTable).parameters:
        kwargs["modeled_series_getter"] = lambda: modeled

    tbl = UnifiedSpeciesTable(**kwargs)

    calls: list[str] = []
    original = tbl._recompute_fit_universe

    def tracking_recompute():
        calls.append("called")
        original()

    monkeypatch.setattr(tbl, "_recompute_fit_universe", tracking_recompute)

    new_t = np.linspace(0, 2, 10)
    tbl.add_dataset_state(
        "ds2",
        full_series={"A": np.ones(10), "C": np.ones(10)},
        full_t=new_t,
        available=["A", "C"],
    )

    assert len(calls) >= 1, (
        "add_dataset_state must call _recompute_fit_universe (F3 rule)"
    )
    assert sorted(tbl._fit_targets_available_by_dataset["ds2"]) == ["A", "C"]
    tbl.close()
    qt_app.processEvents()


def _make_window():
    from kindred.gui.fitting.window import FittingWindow
    from kindred.core.simulation_preparation import PreparedSimulationMetadata

    t = np.linspace(0.0, 1.0, 6)
    y_a = np.linspace(1.0, 0.5, t.size)
    y_b = np.linspace(0.2, 0.9, t.size)

    mechanism_text = "rxn: A -> B; k1=0.2"
    reactions_text = "rxn: A -> B; k1=0.2"

    def simulation_func(_params):
        return {"t": t.copy(), "species": {"A": y_a.copy(), "B": y_b.copy()}}

    prepared_meta = PreparedSimulationMetadata(
        version=1,
        mechanism_text_sha256=hashlib.sha256(mechanism_text.encode("utf-8")).hexdigest(),
        mechanism_text_len=len(mechanism_text),
        param_names=["k1"],
        t_end=1.0,
        num_points=6,
        temperature_K=298.15,
        solver_requested="BDF",
        solver_normalized="BDF",
        solver_warning=None,
        rtol=1e-6,
        atol=1e-12,
        use_sparse_jacobian=False,
        wegscheider_cyclicity_enabled=False,
        initial_prefix="init:",
    )
    simulation_func._kindred_prepared_simulation_meta = prepared_meta  # type: ignore[attr-defined]

    dataset_entries = [
        {
            "id": "ds1",
            "label": "ds1",
            "t": t.copy(),
            "species_data": {"A": y_a.copy(), "B": y_b.copy()},
            "selected_species": ["A"],
            "weight": 1.0,
            "include": True,
        }
    ]
    dataset_payloads = [
        {
            "id": "ds1",
            "t": t.copy(),
            "y": np.vstack([y_a.copy()]),
            "species": ["A"],
        }
    ]
    return FittingWindow(
        mode="global",
        parameter_defs=[{"name": "k1", "value": 0.2, "min": 0.01, "max": 1.0}],
        dataset_entries=dataset_entries,
        simulation_func=simulation_func,
        mechanism_text_getter=lambda: mechanism_text,
        reactions_text_getter=lambda: reactions_text,
        dataset_params={"ds1": {"init:A": 1.0}},
        dataset_variable_params={
            "ds1": {"init:B": {"initial": 0.2, "min": 0.0, "max": 10.0, "log10": False}}
        },
        dataset_payloads=dataset_payloads,
        dataset_weights={"ds1": 1.0},
    )


# ---- FittingWindow has minimize/maximize buttons ----

def test_fitting_window_has_min_max_buttons(qt_app):
    window = _make_window()
    try:
        flags = window.windowFlags()
        assert not (flags & Qt.CustomizeWindowHint), "CustomizeWindowHint breaks taskbar grouping"
        assert flags & Qt.WindowMinMaxButtonsHint
        assert flags & Qt.WindowCloseButtonHint
    finally:
        window.close()
        qt_app.processEvents()


# ---- Parameters/ICs splitter is horizontal ----

def test_parameters_tab_has_no_splitter_after_ic_extraction(qt_app):
    from kindred.gui.fitting.parameters_ics_tab import ParametersIcsTab

    tab = ParametersIcsTab(
        parameter_state=[],
        initial_parameter_snapshot=[],
        global_dataset_params={},
        global_dataset_variable_params={},
        fixed_shared_params={},
        shared_param_definitions={},
        mechanism_species=["A", "B"],
        dataset_entries=[{"id": "ds1", "label": "DS 1"}],
        prepared_param_names=[],
        selected_dataset_ids_getter=lambda: ["ds1"],
        dataset_entries_getter=lambda: [{"id": "ds1", "label": "DS 1"}],
        worker_running_getter=lambda: False,
        dataset_manager_getter=lambda: None,
        reactions_text_getter=lambda: "",
        integration_defaults=("BDF", 1e-6, 1e-12),
        config_defaults={},
    )
    try:
        splitter = tab.findChild(QtWidgets.QSplitter)
        assert splitter is None  # IC panel extracted, no splitter needed
    finally:
        tab.close()
        qt_app.processEvents()


# ---- Run Stamp moved to footer popup ----

def test_run_results_tab_has_no_run_stamp_groupbox(qt_app):
    from kindred.gui.fitting.run_results_tab import RunResultsTab

    tab = RunResultsTab(parent=None)
    try:
        groups = tab.findChildren(QtWidgets.QGroupBox)
        stamp_groups = [g for g in groups if g.title() == "Run Stamp"]
        assert len(stamp_groups) == 0
    finally:
        tab.close()
        qt_app.processEvents()


def test_footer_has_results_summary_button(qt_app):
    window = _make_window()
    try:
        btn = window.findChild(QtWidgets.QPushButton, "global_fit_results_summary_footer_button")
        assert btn is not None
        assert not btn.isEnabled()
    finally:
        window.close()
        qt_app.processEvents()


# ---- Regex: _validate_observable_name_rules rejects rate-constant patterns ----

def test_validate_observable_name_rejects_rate_constant_patterns(qt_app, monkeypatch):
    from kindred.core.algebra.symbols import SymbolTable

    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *a, **kw: None)

    window = _make_window()
    try:
        st = SymbolTable()
        must_reject = ["k1", "kf3", "kr10", "K5", "Keq5", "KF3", "KR5", "KEQ2", "K2"]
        for name in must_reject:
            result = window._validate_observable_name_rules(
                name, mechanism_species=set(), symbol_table=st,
            )
            assert result is False, f"Should reject rate-constant-like name: {name!r}"

        must_accept = ["kcat", "myK1", "species_k1", "K"]
        for name in must_accept:
            result = window._validate_observable_name_rules(
                name, mechanism_species=set(), symbol_table=st,
            )
            assert result is True, f"Should accept non-rate-constant name: {name!r}"
    finally:
        window.close()
        qt_app.processEvents()


# ---- Regex: _reactions_text_has_param_decl detects existing param declarations ----

def test_reactions_text_has_param_decl_detects_existing_param():
    from kindred.gui.fitting.window import FittingWindow

    check = FittingWindow._reactions_text_has_param_decl

    assert check("param myP = 1.0", "myP") is True
    assert check("  param myP = 1.0", "myP") is True
    assert check("PARAM myP = 1.0", "myP") is True
    assert check("line1\n  param myP = 1.0\nline3", "myP") is True

    assert check("param otherName = 1.0", "myP") is False
    assert check("", "myP") is False
    assert check("something else", "myP") is False


# ---- Stale-applied-targets bug: empty apply must disable Run Fit ----

def test_apply_empty_targets_disables_run_fit(qt_app, qtbot):
    window = _make_window()
    try:
        st = window._species_table

        # Initially ds1 has applied target ["A"], Run Fit should be enabled
        window._prepare_fit_runtime_for_current_state()
        qtbot.waitUntil(lambda: window._run_button.isEnabled(), timeout=2000)
        assert window._run_button.isEnabled(), "Run Fit should start enabled with applied targets"

        # Uncheck all targets for ds1
        st._fit_targets_selection_pending["ds1"] = set()
        st._fit_targets_dirty = True

        # Apply — this emits targetsApplied which triggers _on_targets_validity_changed
        st._apply_changes()
        qt_app.processEvents()

        # Run Fit must now be disabled
        assert not window._run_button.isEnabled(), (
            "Run Fit must be disabled after applying empty targets"
        )
    finally:
        window.close()
        qt_app.processEvents()
