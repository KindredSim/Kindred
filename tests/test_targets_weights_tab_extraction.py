"""Tests for TargetsWeightsTab standalone extraction."""
from __future__ import annotations

import numpy as np
import pytest
from PySide6 import QtWidgets
from PySide6.QtCore import Qt


pytestmark = [pytest.mark.gui]


def _make_tab(
    *,
    species: list[str] | None = None,
    selected: list[str] | None = None,
    target_weights: dict[str, float] | None = None,
    include: bool = True,
    extra_datasets: list[dict] | None = None,
):
    from kindred.gui.fitting.targets_weights_tab import TargetsWeightsTab

    if species is None:
        species = ["A", "B"]
    if selected is None:
        selected = list(species)

    t = np.linspace(0.0, 1.0, 6)
    entry = {
        "id": "ds1",
        "label": "Dataset 1",
        "t": t.copy(),
        "species_data": {name: np.linspace(0, 1, t.size) for name in species},
        "selected_species": list(selected),
        "weight": 1.0,
        "include": include,
    }
    if target_weights:
        entry["target_weights"] = dict(target_weights)
    entries = [entry]
    if extra_datasets:
        entries.extend(extra_datasets)

    tab = TargetsWeightsTab(
        dataset_entries=list(entries),
        dataset_entries_getter=lambda: list(entries),
        included_dataset_ids_getter=lambda: [
            str(e["id"]) for e in entries if e.get("include", True)
        ],
        dataset_label_getter=lambda ds_id: next(
            (str(e.get("label") or ds_id) for e in entries if str(e.get("id")) == ds_id),
            str(ds_id),
        ),
        dataset_weight_getter=lambda ds_id: 1.0,
        persist_dataset_weight_callback=lambda ds_id, w: None,
        worker_running_getter=lambda: False,
    )
    return tab


def test_construction(qt_app):
    """TargetsWeightsTab builds expected widget hierarchy."""
    tab = _make_tab()
    try:
        dataset_list = tab.findChild(QtWidgets.QListWidget, "global_fit_fit_targets_dataset_list")
        assert dataset_list is not None

        group = tab.findChild(QtWidgets.QGroupBox, "global_fit_fit_targets_panel")
        assert group is not None

        from kindred.gui.widgets.config_panel_footer import ConfigPanelFooter
        footer = tab.findChild(ConfigPanelFooter)
        assert footer is not None
    finally:
        tab.close()
        qt_app.processEvents()


def test_three_tier_state_init(qt_app):
    """Constructor seeds applied, pending, and available state correctly."""
    tab = _make_tab(species=["A", "B", "C"], selected=["A", "C"], target_weights={"A": 2.0, "C": 3.5})
    try:
        assert tab._fit_targets_selection_applied["ds1"] == ["A", "C"]
        assert set(tab._fit_targets_selection_pending["ds1"]) == {"A", "C"}
        assert tab._fit_target_weights_applied["ds1"]["A"] == 2.0
        assert tab._fit_target_weights_applied["ds1"]["C"] == 3.5
        assert sorted(tab.available_by_dataset["ds1"]) == ["A", "B", "C"]
    finally:
        tab.close()
        qt_app.processEvents()


def test_apply_revert_cycle(qt_app):
    """Toggle pending state, apply, verify; then dirty up and revert."""
    tab = _make_tab(species=["A", "B"], selected=["A"])
    try:
        # Initial state: applied = ["A"], pending = {"A"}
        assert tab._fit_targets_selection_applied["ds1"] == ["A"]
        assert set(tab._fit_targets_selection_pending["ds1"]) == {"A"}

        # Toggle B on in pending
        tab._fit_targets_selection_pending["ds1"].add("B")
        tab._update_fit_targets_dirty_state()
        assert tab._fit_targets_dirty is True

        # Apply
        tab._apply_fit_targets_changes()
        assert set(tab._fit_targets_selection_applied["ds1"]) == {"A", "B"}

        # Dirty up again: remove A from pending
        tab._fit_targets_selection_pending["ds1"].discard("A")
        tab._update_fit_targets_dirty_state()
        assert tab._fit_targets_dirty is True

        # Revert
        tab._revert_fit_targets_changes()
        assert set(tab._fit_targets_selection_pending["ds1"]) == {"A", "B"}
        assert tab._fit_targets_dirty is False
    finally:
        tab.close()
        qt_app.processEvents()


def test_flush_weight_edits(qt_app):
    """flush_visible_weight_edits updates pending weight state from UI."""
    tab = _make_tab(species=["A", "B"], selected=["A", "B"])
    try:
        # Activate the tab to populate the checklist
        tab.on_tab_activated(seed_dataset_id="ds1")
        qt_app.processEvents()

        # Flush should not raise
        tab.flush_visible_weight_edits()
    finally:
        tab.close()
        qt_app.processEvents()


def test_on_tab_activated_seeding(qt_app):
    """First activation seeds dataset selection; second call does not re-seed."""
    t = np.linspace(0.0, 1.0, 6)
    extra = {
        "id": "ds2",
        "label": "Dataset 2",
        "t": t.copy(),
        "species_data": {"X": np.linspace(0, 1, t.size)},
        "selected_species": ["X"],
        "weight": 1.0,
        "include": True,
    }
    tab = _make_tab(extra_datasets=[extra])
    try:
        tab.on_tab_activated(seed_dataset_id="ds2")
        qt_app.processEvents()

        dataset_list = tab.findChild(QtWidgets.QListWidget, "global_fit_fit_targets_dataset_list")
        assert dataset_list is not None
        selected_items = dataset_list.selectedItems()
        if selected_items:
            first_selected = str(selected_items[0].data(Qt.UserRole) or "")
            assert first_selected == "ds2"

        # Second call should NOT re-seed (write-once gate)
        tab.on_tab_activated(seed_dataset_id="ds1")
        qt_app.processEvents()
        selected_items = dataset_list.selectedItems()
        if selected_items:
            still_selected = str(selected_items[0].data(Qt.UserRole) or "")
            assert still_selected == "ds2"
    finally:
        tab.close()
        qt_app.processEvents()


def test_validity_changed_signal(qt_app):
    """validityChanged signal is emitted when validity state changes."""
    tab = _make_tab(species=["A"], selected=[])
    try:
        emissions = []
        tab.validityChanged.connect(lambda: emissions.append(True))
        tab.refresh_validity_ui()
        qt_app.processEvents()
        assert len(emissions) >= 1
    finally:
        tab.close()
        qt_app.processEvents()


def test_add_remove_dataset_state(qt_app):
    """add_dataset_state and remove_dataset_state manage per-dataset dicts."""
    tab = _make_tab()
    try:
        t2 = np.linspace(0, 2, 10)
        tab.add_dataset_state(
            "ds_new",
            full_series={"X": np.linspace(0, 1, 10), "Y": np.linspace(1, 0, 10)},
            full_t=t2,
            available=["X", "Y"],
        )
        assert "ds_new" in tab.full_series_by_dataset
        assert "ds_new" in tab.full_t_by_dataset
        assert sorted(tab.available_by_dataset["ds_new"]) == ["X", "Y"]
        assert tab._fit_targets_selection_applied["ds_new"] == []

        tab.remove_dataset_state({"ds_new"})
        assert "ds_new" not in tab.full_series_by_dataset
        assert "ds_new" not in tab.full_t_by_dataset
        assert "ds_new" not in tab.available_by_dataset
    finally:
        tab.close()
        qt_app.processEvents()
