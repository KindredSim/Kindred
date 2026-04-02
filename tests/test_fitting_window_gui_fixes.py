"""Regression tests for fitting window GUI layout fixes."""
from __future__ import annotations

import hashlib

import numpy as np
import pytest
from PySide6 import QtWidgets
from PySide6.QtCore import Qt


pytestmark = [pytest.mark.gui]


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
        solver_requested="LSODA",
        solver_normalized="LSODA",
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
        integration_defaults=("LSODA", 1e-6, 1e-12),
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
