"""Phase-0 extraction regression tests.

Each test validates a specific behavior contract that was at risk
during the FittingWindow -> tab extraction refactor.
"""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
from PySide6 import QtWidgets


pytestmark = [pytest.mark.gui]


# ---------------------------------------------------------------------------
# Helpers — TargetsWeightsTab
# ---------------------------------------------------------------------------

def _make_tw_tab(*, entries=None, included_ids=None):
    from kindred.gui.fitting.targets_weights_tab import TargetsWeightsTab

    if entries is None:
        t = np.linspace(0.0, 1.0, 6)
        entries = [
            {
                "id": "ds1",
                "label": "Dataset 1",
                "t": t.copy(),
                "species_data": {"A": np.linspace(0, 1, t.size)},
                "selected_species": ["A"],
                "weight": 1.0,
                "include": True,
            }
        ]

    _entries = list(entries)

    if included_ids is None:
        included_ids = [str(e["id"]) for e in _entries if e.get("include", True)]

    tab = TargetsWeightsTab(
        dataset_entries=list(_entries),
        dataset_entries_getter=lambda: list(_entries),
        included_dataset_ids_getter=lambda: list(included_ids),
        dataset_label_getter=lambda ds_id: next(
            (str(e.get("label") or ds_id) for e in _entries if str(e.get("id")) == ds_id),
            str(ds_id),
        ),
        dataset_weight_getter=lambda ds_id: 1.0,
        persist_dataset_weight_callback=lambda ds_id, w: None,
        worker_running_getter=lambda: False,
    )
    return tab


# ---------------------------------------------------------------------------
# Helpers — DataTab
# ---------------------------------------------------------------------------

def _make_data_tab(*, worker_running=False):
    from kindred.gui.fitting.data_tab import DataTab

    empty_cfg = {
        "t_min": 0.0, "t_max": 1.0, "n_points": 10,
        "x_name": "t", "x_mapping_mode": "auto",
    }
    tab = DataTab(
        sampling_applied_config_getter=lambda ds_id: dict(empty_cfg),
        sampling_default_config_getter=lambda t: dict(empty_cfg),
        fit_targets_full_t_getter=lambda ds_id: np.asarray([]),
        fit_targets_available_getter=lambda ds_id: [],
        fit_targets_full_series_getter=lambda ds_id: {},
        fit_targets_selection_applied_getter=lambda ds_id: [],
        modeled_series_getter=lambda: set(),
        worker_running_getter=lambda: worker_running,
    )
    return tab


# ---------------------------------------------------------------------------
# Helpers — ParametersIcsTab
# ---------------------------------------------------------------------------

def _make_params_tab(*, integration_defaults=("LSODA", 1e-6, 1e-12),
                     entries=None, species=None):
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


# ---------------------------------------------------------------------------
# Helpers — FittingWindow (minimal, for ITEM 19)
# ---------------------------------------------------------------------------

def _make_fitting_window():
    from kindred.gui.fitting.window import FittingWindow

    t = np.linspace(0.0, 1.0, 5)
    y_a = np.linspace(1.0, 0.5, t.size)
    return FittingWindow(
        mode="global",
        parameter_defs=[{"name": "k1", "value": 0.2, "min": 0.01, "max": 1.0}],
        dataset_entries=[
            {
                "id": "ds1",
                "label": "ds1",
                "t": t.copy(),
                "species_data": {"A": y_a.copy()},
                "selected_species": ["A"],
                "weight": 1.0,
                "include": True,
            }
        ],
        simulation_func=lambda _params: {"t": t.copy(), "species": {"A": y_a.copy()}},
        dataset_payloads=[{
            "id": "ds1", "t": t.copy(),
            "y": np.vstack([y_a.copy()]), "species": ["A"],
        }],
        dataset_weights={"ds1": 1.0},
    )


# ===================================================================
# ITEM 19 — Subset stale suffix on apply failure
# ===================================================================

def test_on_targets_applied_subset_failure(qt_app):
    """When subset widget update fails, status shows '(subset view stale)' suffix."""
    window = _make_fitting_window()
    try:
        window._subset_widget.set_dataset_entries = lambda *a, **kw: (_ for _ in ()).throw(
            RuntimeError("test subset failure")
        )
        window._on_targets_applied()
        assert window._subset_view_stale is True

        window._on_targets_status_message("Fit targets applied")
        assert window._status_label.text().endswith("(subset view stale)")
        assert window._subset_view_stale is False
    finally:
        window.close()
        qt_app.processEvents()


# ===================================================================
# ITEM 20 — Footer cleared after last dataset removal
# ===================================================================

def test_refresh_dataset_list_clears_footer_on_empty(qt_app):
    """refresh_dataset_list clears footer errors when dataset list becomes empty."""
    entries = []

    def entries_getter():
        return list(entries)

    t = np.linspace(0.0, 1.0, 6)
    initial_entries = [
        {
            "id": "ds1",
            "label": "Dataset 1",
            "t": t.copy(),
            "species_data": {"A": np.linspace(0, 1, t.size)},
            "selected_species": ["A"],
            "weight": 1.0,
            "include": True,
        }
    ]
    entries.extend(initial_entries)

    from kindred.gui.fitting.targets_weights_tab import TargetsWeightsTab
    tab = TargetsWeightsTab(
        dataset_entries=list(entries),
        dataset_entries_getter=entries_getter,
        included_dataset_ids_getter=lambda: [e["id"] for e in entries if e.get("include", True)],
        dataset_label_getter=lambda ds_id: next(
            (str(e.get("label") or ds_id) for e in entries if str(e.get("id")) == ds_id),
            str(ds_id),
        ),
        dataset_weight_getter=lambda ds_id: 1.0,
        persist_dataset_weight_callback=lambda ds_id, w: None,
        worker_running_getter=lambda: False,
    )
    try:
        tab._fit_targets_footer.set_error("Stale error from previous dataset")
        assert tab._fit_targets_footer._error_text == "Stale error from previous dataset"

        entries.clear()
        tab.remove_dataset_state("ds1")
        tab.refresh_dataset_list()
        qt_app.processEvents()

        assert tab._fit_targets_footer._error_text == ""
    finally:
        tab.close()
        qt_app.processEvents()


# ===================================================================
# ITEM 6 — Remove button disabled during active fit
# ===================================================================

def test_remove_button_disabled_during_active_fit(qt_app):
    """DataTab remove button stays disabled when worker is running."""
    tab = _make_data_tab(worker_running=True)
    try:
        entries = [
            {"id": "ds1", "label": "DS 1", "selected_species": ["A"], "include": True},
        ]
        tab.populate_table(entries)
        tab._dataset_table.selectRow(0)
        qt_app.processEvents()

        assert not tab._dataset_remove_button.isEnabled()
    finally:
        tab.close()
        qt_app.processEvents()


# ===================================================================
# ITEM 15 — Integration defaults from active method
# ===================================================================

def test_integration_defaults_from_active_method(qt_app):
    """ParametersIcsTab populates rtol/atol fields from integration_defaults tuple."""
    tab = _make_params_tab(integration_defaults=("LSODA", 1e-3, 1e-9))
    try:
        assert tab._integration_rtol_edit.text() == "0.001"
        assert tab._integration_atol_edit.text() == "1e-9"
        assert tab._integration_solver_combo.currentText() == "LSODA"
    finally:
        tab.close()
        qt_app.processEvents()


# ===================================================================
# ITEM 16 — Algebraic observable add preserves metadata
# ===================================================================

def test_algebraic_observable_add_preserves_metadata(qt_app):
    """addAlgebraicObservableRequested payload includes dataset_ids and persist."""
    tab = _make_params_tab()
    try:
        received = []
        tab.addAlgebraicObservableRequested.connect(lambda payload: received.append(dict(payload)))

        with patch(
            "kindred.gui.fitting.parameters_ics_tab._AddFittableParameterDialog"
        ) as MockDialog:
            instance = MockDialog.return_value
            instance.exec.return_value = QtWidgets.QDialog.Accepted
            instance.selection.return_value = {
                "type": "observable_new",
                "name": "obs1",
                "expr": "A+B",
            }
            tab._add_parameter()

        assert len(received) == 1
        payload = received[0]
        assert "dataset_ids" in payload
        assert isinstance(payload["dataset_ids"], list)
        assert "persist" in payload
        assert payload["persist"] is True
    finally:
        tab.close()
        qt_app.processEvents()


# ===================================================================
# ITEM 17 — push_best_update with empty dataset_params clears staged
# ===================================================================

def test_push_best_update_empty_dataset_params_clears_staged(qt_app):
    """push_best_update with empty dict clears _staged_dataset_params."""
    tab = _make_params_tab()
    try:
        tab._staged_dataset_params = {"ds1": {"k1": 1.0}}
        assert tab._staged_dataset_params

        tab.push_best_update(shared_params={}, dataset_params={})
        assert tab._staged_dataset_params == {}
    finally:
        tab.close()
        qt_app.processEvents()


# ===================================================================
# ITEM 18 — rebuild_for_mechanism updates entries before combo
# ===================================================================

def test_rebuild_for_mechanism_updates_dataset_entries_before_combo(qt_app):
    """rebuild_for_mechanism updates _dataset_entries before refreshing IC combo."""
    initial_entries = [{"id": "ds1", "label": "DS 1"}]
    tab = _make_params_tab(entries=initial_entries, species=["A"])
    try:
        new_entries = [
            {"id": "ds1", "label": "DS 1"},
            {"id": "ds2", "label": "DS 2"},
        ]
        tab.rebuild_for_mechanism("", new_entries)

        combo = tab._ic_dataset_combo
        combo_ids = [str(combo.itemData(i) or "") for i in range(combo.count())]
        assert "ds1" in combo_ids
        assert "ds2" in combo_ids
    finally:
        tab.close()
        qt_app.processEvents()
