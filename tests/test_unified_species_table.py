"""Tests for the UnifiedSpeciesTable widget."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtCore import Qt

from kindred.gui.fitting.unified_species_table import UnifiedSpeciesTable, _Col


pytestmark = [pytest.mark.gui]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_entries(species_data=None, ds_id="ds1", label="DS 1", selected=None):
    if species_data is None:
        t = np.linspace(0, 1, 5)
        species_data = {"A": np.linspace(1, 0.5, t.size), "B": np.linspace(0.2, 0.9, t.size)}
    else:
        t = np.linspace(0, 1, len(next(iter(species_data.values()))))
    if selected is None:
        selected = list(species_data.keys())
    return [{
        "id": ds_id,
        "label": label,
        "t": t,
        "species_data": species_data,
        "selected_species": selected,
        "weight": 1.0,
        "include": True,
    }]


def _make_manager(species, ds_id="ds1"):
    """Return a fake dataset manager with default fit settings."""
    settings = SimpleNamespace(
        initial_conditions={s: 1.0 for s in species},
        fit_flags={s: False for s in species},
        log10_flags={},
        bounds={},
    )

    class FakeManager:
        def get_fit_settings(self, _ds_id):
            return settings

        def update_fit_settings(self, _ds_id, _settings):
            pass

    return FakeManager()


def _make_table(*, entries=None, species=None, manager=None, included_ids=None):
    if entries is None:
        entries = _make_entries()
    if species is None:
        species = sorted({s for e in entries for s in (e.get("species_data") or {}).keys()})
    ds_ids = [str(e["id"]) for e in entries]
    if included_ids is None:
        included_ids = list(ds_ids)
    weights = {str(e["id"]): float(e.get("weight", 1.0)) for e in entries}
    persisted_weights: dict[str, float] = {}

    tbl = UnifiedSpeciesTable(
        dataset_entries=list(entries),
        mechanism_species=list(species),
        dataset_entries_getter=lambda: list(entries),
        included_dataset_ids_getter=lambda: list(included_ids),
        dataset_label_getter=lambda ds_id: str(ds_id),
        dataset_weight_getter=lambda ds_id: weights.get(str(ds_id), 1.0),
        persist_dataset_weight_callback=lambda ds_id, w: persisted_weights.update({ds_id: w}),
        dataset_manager_getter=lambda: manager,
        worker_running_getter=lambda: False,
    )
    tbl._persisted_weights = persisted_weights  # for test introspection
    return tbl


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_table_has_correct_columns(qt_app):
    """The table has exactly 8 columns with the expected headers."""
    tbl = _make_table()
    try:
        tbl.load_for_dataset("ds1")
        assert tbl._table.columnCount() == _Col.COUNT
        headers = [
            tbl._table.horizontalHeaderItem(c).text() for c in range(_Col.COUNT)
        ]
        assert headers == _Col.HEADERS
    finally:
        tbl.close()
        qt_app.processEvents()


def test_include_checkbox_updates_pending(qt_app):
    """Toggling Include in Fit updates the pending selection dict."""
    tbl = _make_table()
    try:
        tbl.load_for_dataset("ds1")
        # Initially both A and B are selected
        assert "A" in tbl._fit_targets_selection_pending["ds1"]
        assert "B" in tbl._fit_targets_selection_pending["ds1"]

        # Uncheck A
        for row in range(tbl._table.rowCount()):
            if tbl._table.item(row, _Col.SPECIES).text() == "A":
                tbl._table.item(row, _Col.INCLUDE).setCheckState(Qt.Unchecked)
                break
        qt_app.processEvents()

        assert "A" not in tbl._fit_targets_selection_pending["ds1"]
        assert "B" in tbl._fit_targets_selection_pending["ds1"]
    finally:
        tbl.close()
        qt_app.processEvents()


def test_weight_edit_updates_pending(qt_app):
    """Editing the Weight column updates pending weight state."""
    tbl = _make_table()
    try:
        tbl.load_for_dataset("ds1")
        for row in range(tbl._table.rowCount()):
            if tbl._table.item(row, _Col.SPECIES).text() == "A":
                tbl._table.item(row, _Col.WEIGHT).setText("3.5")
                break
        qt_app.processEvents()

        assert tbl._fit_target_weights_pending["ds1"]["A"] == pytest.approx(3.5)
    finally:
        tbl.close()
        qt_app.processEvents()


def test_ic_edit_sets_dirty(qt_app):
    """Editing an IC column (Initial, Min, Max, Fit IC) sets IC dirty flag."""
    mgr = _make_manager(["A", "B"])
    tbl = _make_table(manager=mgr)
    try:
        tbl.load_for_dataset("ds1")
        assert tbl._ic_editor_dirty is False

        for row in range(tbl._table.rowCount()):
            if tbl._table.item(row, _Col.SPECIES).text() == "A":
                tbl._table.item(row, _Col.INITIAL).setText("9.9")
                break
        qt_app.processEvents()

        assert tbl._ic_editor_dirty is True
    finally:
        tbl.close()
        qt_app.processEvents()


def test_log10_greyed_unless_fit_ic_checked(qt_app):
    """Log10 column is non-interactive unless Fit IC is checked."""
    mgr = _make_manager(["A", "B"])
    tbl = _make_table(manager=mgr)
    try:
        tbl.load_for_dataset("ds1")

        for row in range(tbl._table.rowCount()):
            if tbl._table.item(row, _Col.SPECIES).text() == "A":
                log_item = tbl._table.item(row, _Col.LOG10)
                fit_item = tbl._table.item(row, _Col.FIT_IC)

                # Initially Fit IC unchecked -> Log10 greyed
                assert not (log_item.flags() & Qt.ItemIsEnabled)

                # Check Fit IC -> Log10 enabled
                fit_item.setCheckState(Qt.Checked)
                qt_app.processEvents()
                log_item = tbl._table.item(row, _Col.LOG10)
                assert log_item.flags() & Qt.ItemIsEnabled

                # Uncheck Fit IC -> Log10 greyed again, check cleared
                fit_item.setCheckState(Qt.Unchecked)
                qt_app.processEvents()
                log_item = tbl._table.item(row, _Col.LOG10)
                assert not (log_item.flags() & Qt.ItemIsEnabled)
                assert log_item.checkState() == Qt.Unchecked
                break
    finally:
        tbl.close()
        qt_app.processEvents()


def test_bulk_all_skips_greyed_rows(qt_app):
    """Bulk All selects only species with available data (not greyed rows)."""
    # Create entries with only species A having data, but mechanism has A and C
    entries = _make_entries(species_data={"A": np.linspace(1, 0, 5)}, selected=["A"])
    tbl = _make_table(entries=entries, species=["A", "C"])
    try:
        tbl.load_for_dataset("ds1")

        tbl._apply_bulk_action("all")
        qt_app.processEvents()

        pending = tbl._fit_targets_selection_pending.get("ds1", set())
        assert "A" in pending
        assert "C" not in pending  # C has no data, row greyed
    finally:
        tbl.close()
        qt_app.processEvents()


def test_apply_commits_targets_and_ic(qt_app):
    """Apply persists both target selection and IC changes."""
    mgr = _make_manager(["A", "B"])
    tbl = _make_table(manager=mgr)
    try:
        tbl.load_for_dataset("ds1")

        # Uncheck B from targets
        for row in range(tbl._table.rowCount()):
            if tbl._table.item(row, _Col.SPECIES).text() == "B":
                tbl._table.item(row, _Col.INCLUDE).setCheckState(Qt.Unchecked)
                break

        # Edit IC for A
        for row in range(tbl._table.rowCount()):
            if tbl._table.item(row, _Col.SPECIES).text() == "A":
                tbl._table.item(row, _Col.INITIAL).setText("5.0")
                tbl._table.item(row, _Col.FIT_IC).setCheckState(Qt.Checked)
                break
        qt_app.processEvents()

        applied_targets = []
        applied_ic = []
        tbl.targetsApplied.connect(lambda: applied_targets.append(True))
        tbl.icApplied.connect(lambda *a: applied_ic.append(a))

        tbl._apply_changes()
        qt_app.processEvents()

        assert applied_targets == [True]
        assert tbl.fit_targets_selection_applied["ds1"] == ["A"]

        assert len(applied_ic) == 1
        ic_ds, ic_updates, ic_fit = applied_ic[0]
        assert ic_ds == "ds1"
        assert ic_updates["A"]["initial"] == pytest.approx(5.0)
        assert ic_fit["A"] is True
    finally:
        tbl.close()
        qt_app.processEvents()


def test_revert_restores_state(qt_app):
    """Revert restores pending selection to applied state and clears IC dirty."""
    mgr = _make_manager(["A", "B"])
    tbl = _make_table(manager=mgr)
    try:
        tbl.load_for_dataset("ds1")

        # Mutate pending state
        for row in range(tbl._table.rowCount()):
            if tbl._table.item(row, _Col.SPECIES).text() == "A":
                tbl._table.item(row, _Col.INCLUDE).setCheckState(Qt.Unchecked)
                tbl._table.item(row, _Col.INITIAL).setText("99.0")
                break
        qt_app.processEvents()

        assert "A" not in tbl._fit_targets_selection_pending["ds1"]
        assert tbl._ic_editor_dirty is True

        tbl._revert_changes()
        qt_app.processEvents()

        # Pending should match applied (both A and B selected)
        assert set(tbl._fit_targets_selection_pending["ds1"]) == set(tbl._fit_targets_selection_applied["ds1"])
        assert tbl._ic_editor_dirty is False
    finally:
        tbl.close()
        qt_app.processEvents()


def test_dataset_switch_flushes_targets(qt_app):
    """Switching datasets flushes pending weight edits and loads new dataset."""
    t = np.linspace(0, 1, 5)
    entries = [
        {
            "id": "ds1", "label": "DS 1", "t": t,
            "species_data": {"A": np.linspace(1, 0, 5)},
            "selected_species": ["A"], "weight": 1.0, "include": True,
        },
        {
            "id": "ds2", "label": "DS 2", "t": t,
            "species_data": {"A": np.linspace(0.5, 0.1, 5)},
            "selected_species": ["A"], "weight": 2.0, "include": True,
        },
    ]
    tbl = _make_table(entries=entries, species=["A"])
    try:
        tbl.load_for_dataset("ds1")
        assert tbl._current_dataset_id == "ds1"

        tbl.load_for_dataset("ds2")
        assert tbl._current_dataset_id == "ds2"
        assert tbl._ic_editor_dirty is False
    finally:
        tbl.close()
        qt_app.processEvents()


def test_add_remove_dataset_state(qt_app):
    """add_dataset_state / remove_dataset_state manage internal dictionaries."""
    tbl = _make_table()
    try:
        assert "ds1" in tbl.available_by_dataset

        new_series = {"X": np.linspace(0, 1, 5)}
        tbl.add_dataset_state("ds_new", full_series=new_series,
                              full_t=np.linspace(0, 1, 5), available=["X"])
        assert "ds_new" in tbl.available_by_dataset
        assert tbl.fit_targets_selection_applied["ds_new"] == []

        tbl.load_for_dataset("ds_new")
        assert tbl._current_dataset_id == "ds_new"

        tbl.remove_dataset_state({"ds_new"})
        assert "ds_new" not in tbl.available_by_dataset
        assert tbl._current_dataset_id is None
    finally:
        tbl.close()
        qt_app.processEvents()


def test_validity_empty_selection_invalid(qt_app):
    """An included dataset with empty pending selection is reported invalid."""
    tbl = _make_table()
    try:
        tbl.load_for_dataset("ds1")

        # Deselect all species
        for row in range(tbl._table.rowCount()):
            include_item = tbl._table.item(row, _Col.INCLUDE)
            if include_item is not None and (include_item.flags() & Qt.ItemIsEnabled):
                include_item.setCheckState(Qt.Unchecked)
        qt_app.processEvents()

        invalid = tbl.invalid_pending_used_dataset_ids()
        assert "ds1" in invalid
    finally:
        tbl.close()
        qt_app.processEvents()


def test_weight_mode_implicit(qt_app):
    """weight_mode_is_implicit returns True when combo is at index 0."""
    tbl = _make_table()
    try:
        assert tbl.weight_mode_is_implicit() is True
        tbl._weight_mode_combo.setCurrentIndex(1)
        assert tbl.weight_mode_is_implicit() is False
        tbl._weight_mode_combo.setCurrentIndex(0)
        assert tbl.weight_mode_is_implicit() is True
    finally:
        tbl.close()
        qt_app.processEvents()


def test_set_mechanism_species_updates_rows(qt_app):
    """set_mechanism_species changes _mechanism_species and repopulates the table."""
    mgr = _make_manager(["A", "B", "C"])
    tbl = _make_table(manager=mgr, species=["A", "B"])
    try:
        tbl.load_for_dataset("ds1")
        assert tbl._table.rowCount() == 2

        tbl.set_mechanism_species(["A", "B", "C"])
        assert tbl._table.rowCount() == 3
        species_col = [tbl._table.item(r, _Col.SPECIES).text() for r in range(3)]
        assert species_col == ["A", "B", "C"]
    finally:
        tbl.close()
        qt_app.processEvents()


def test_public_api_surface(qt_app):
    """All expected public methods/properties exist on UnifiedSpeciesTable."""
    tbl = _make_table()
    try:
        # Properties
        assert hasattr(tbl, "fit_targets_selection_applied")
        assert hasattr(tbl, "full_series_by_dataset")
        assert hasattr(tbl, "full_t_by_dataset")
        assert hasattr(tbl, "available_by_dataset")

        # Methods
        assert callable(getattr(tbl, "add_dataset_state", None))
        assert callable(getattr(tbl, "remove_dataset_state", None))
        assert callable(getattr(tbl, "refresh_dataset_list", None))
        assert callable(getattr(tbl, "on_tab_activated", None))
        assert callable(getattr(tbl, "flush_visible_weight_edits", None))
        assert callable(getattr(tbl, "flush_dataset_weight_editor", None))
        assert callable(getattr(tbl, "weight_mode_is_implicit", None))
        assert callable(getattr(tbl, "applied_target_weights_for_dataset", None))
        assert callable(getattr(tbl, "invalid_pending_used_dataset_ids", None))
        assert callable(getattr(tbl, "invalid_pending_target_weight_dataset_ids", None))
        assert callable(getattr(tbl, "invalid_applied_used_dataset_ids", None))
        assert callable(getattr(tbl, "refresh_validity_ui", None))
        assert callable(getattr(tbl, "load_for_dataset", None))
        assert callable(getattr(tbl, "set_mechanism_species", None))
        assert callable(getattr(tbl, "refresh_dataset_combo", None))
        assert callable(getattr(tbl, "initial_parameter_defaults_for_species", None))
        assert callable(getattr(tbl, "set_running_state", None))

        # Signals
        assert hasattr(tbl, "targetsApplied")
        assert hasattr(tbl, "validityChanged")
        assert hasattr(tbl, "icApplied")
        assert hasattr(tbl, "statusMessage")

        # Compatibility aliases
        assert hasattr(tbl, "_ic_editor_current_dataset_id")
        assert hasattr(tbl, "_table")
    finally:
        tbl.close()
        qt_app.processEvents()
