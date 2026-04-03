"""Tests for the UnifiedSpeciesTable widget."""
from __future__ import annotations

from copy import deepcopy
import inspect
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


def _make_manager(
    species=None,
    ds_id="ds1",
    *,
    initials=None,
    fit_flags=None,
    log10_flags=None,
    bounds=None,
    by_dataset=None,
):
    """Return a fake dataset manager with per-dataset fit settings."""
    if by_dataset is None:
        species = list(species or [])
        by_dataset = {
            str(ds_id): {
                "initial_conditions": dict(initials or {s: 1.0 for s in species}),
                "fit_flags": dict(fit_flags or {s: False for s in species}),
                "log10_flags": dict(log10_flags or {}),
                "bounds": dict(bounds or {}),
            }
        }

    def _normalize_settings(raw):
        return SimpleNamespace(
            initial_conditions=dict((raw or {}).get("initial_conditions") or {}),
            fit_flags=dict((raw or {}).get("fit_flags") or {}),
            log10_flags=dict((raw or {}).get("log10_flags") or {}),
            bounds=dict((raw or {}).get("bounds") or {}),
        )

    class FakeManager:
        def __init__(self):
            self.settings_by_dataset = {
                str(name): _normalize_settings(raw) for name, raw in dict(by_dataset or {}).items()
            }
            self.update_calls: list[tuple[str, SimpleNamespace]] = []

        def get_fit_settings(self, dataset_id):
            return self.settings_by_dataset[str(dataset_id)]

        def update_fit_settings(self, dataset_id, settings):
            normalized = _normalize_settings(
                {
                    "initial_conditions": deepcopy(getattr(settings, "initial_conditions", {}) or {}),
                    "fit_flags": deepcopy(getattr(settings, "fit_flags", {}) or {}),
                    "log10_flags": deepcopy(getattr(settings, "log10_flags", {}) or {}),
                    "bounds": deepcopy(getattr(settings, "bounds", {}) or {}),
                }
            )
            ds_key = str(dataset_id)
            self.settings_by_dataset[ds_key] = normalized
            self.update_calls.append((ds_key, normalized))

    return FakeManager()


def _make_table(*, entries=None, species=None, manager=None, included_ids=None, modeled_series=None):
    if entries is None:
        entries = _make_entries()
    if species is None:
        species = sorted({s for e in entries for s in (e.get("species_data") or {}).keys()})
    ds_ids = [str(e["id"]) for e in entries]
    if included_ids is None:
        included_ids = list(ds_ids)
    if modeled_series is None:
        modeled_series = {str(name) for name in (species or []) if str(name).strip()}
    weights = {str(e["id"]): float(e.get("weight", 1.0)) for e in entries}
    persisted_weights: dict[str, float] = {}

    kwargs = dict(
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
    if "modeled_series_getter" in inspect.signature(UnifiedSpeciesTable).parameters:
        kwargs["modeled_series_getter"] = lambda: set(modeled_series)
    tbl = UnifiedSpeciesTable(**kwargs)
    tbl._persisted_weights = persisted_weights  # for test introspection
    return tbl


def _row_for_species(tbl: UnifiedSpeciesTable, species: str) -> int:
    for row in range(tbl._table.rowCount()):
        item = tbl._table.item(row, _Col.SPECIES)
        if item is not None and item.text() == str(species):
            return row
    raise AssertionError(f"Species row not found: {species}")


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


def test_ic_pending_seeded_from_dataset_manager(qt_app):
    mgr = _make_manager(
        by_dataset={
            "ds1": {
                "initial_conditions": {"A": 2.5, "B": 3.5},
                "fit_flags": {"A": True, "B": False},
                "log10_flags": {"A": True},
                "bounds": {"A": (0.5, 5.0), "B": (0.0, 9.0)},
            }
        }
    )
    tbl = _make_table(manager=mgr, species=["A", "B"])
    try:
        assert tbl._ic_pending["ds1"]["A"] == {
            "initial": pytest.approx(2.5),
            "fit": True,
            "log10": True,
            "min": pytest.approx(0.5),
            "max": pytest.approx(5.0),
        }
        assert tbl._ic_applied["ds1"]["B"] == {
            "initial": pytest.approx(3.5),
            "fit": False,
            "log10": False,
            "min": pytest.approx(0.0),
            "max": pytest.approx(9.0),
        }
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


def test_ic_state_survives_bulk_action(qt_app):
    mgr = _make_manager(["A", "B"])
    tbl = _make_table(manager=mgr)
    try:
        tbl.load_for_dataset("ds1")
        row = _row_for_species(tbl, "A")
        tbl._table.item(row, _Col.INITIAL).setText("9.9")
        qt_app.processEvents()

        tbl._apply_bulk_action("all")
        qt_app.processEvents()

        assert tbl._table.item(row, _Col.INITIAL).text() == "9.9"
    finally:
        tbl.close()
        qt_app.processEvents()


def test_ic_state_survives_tab_revisit(qt_app):
    mgr = _make_manager(["A", "B"])
    tbl = _make_table(manager=mgr)
    try:
        tbl.load_for_dataset("ds1")
        row = _row_for_species(tbl, "A")
        tbl._table.item(row, _Col.INITIAL).setText("8.8")
        qt_app.processEvents()

        tbl.on_tab_activated()
        qt_app.processEvents()

        row = _row_for_species(tbl, "A")
        assert tbl._table.item(row, _Col.INITIAL).text() == "8.8"
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


def test_apply_ic_failure_preserves_edits_and_error(qt_app):
    mgr = _make_manager(["A", "B"])
    tbl = _make_table(manager=mgr)
    try:
        tbl.load_for_dataset("ds1")
        row = _row_for_species(tbl, "A")
        tbl._table.item(row, _Col.MIN).setText("abc")
        qt_app.processEvents()

        tbl._apply_changes()
        qt_app.processEvents()

        assert "numeric bounds" in tbl._footer.error_label.text().lower()
        assert tbl._table.item(row, _Col.MIN).text() == "abc"
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


def test_apply_ic_success_updates_manager(qt_app):
    mgr = _make_manager(["A", "B"])
    tbl = _make_table(manager=mgr)
    try:
        tbl.load_for_dataset("ds1")
        row = _row_for_species(tbl, "A")
        tbl._table.item(row, _Col.INITIAL).setText("5.0")
        qt_app.processEvents()

        tbl._apply_changes()
        qt_app.processEvents()

        assert mgr.settings_by_dataset["ds1"].initial_conditions["A"] == pytest.approx(5.0)
        assert tbl._ic_applied["ds1"]["A"]["initial"] == pytest.approx(5.0)
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


def test_revert_reads_live_dataset_manager(qt_app):
    mgr = _make_manager(
        by_dataset={
            "ds1": {
                "initial_conditions": {"A": 1.0},
                "fit_flags": {"A": False},
                "log10_flags": {},
                "bounds": {"A": (0.0, 10.0)},
            }
        }
    )
    tbl = _make_table(manager=mgr, species=["A"])
    try:
        tbl.load_for_dataset("ds1")
        row = _row_for_species(tbl, "A")
        tbl._table.item(row, _Col.INITIAL).setText("1.0")
        qt_app.processEvents()

        tbl._apply_changes()
        qt_app.processEvents()

        settings = SimpleNamespace(
            initial_conditions={"A": 9.9},
            fit_flags={"A": False},
            log10_flags={},
            bounds={"A": (0.0, 10.0)},
        )
        mgr.update_fit_settings("ds1", settings)

        row = _row_for_species(tbl, "A")
        tbl._table.item(row, _Col.INITIAL).setText("4.4")
        qt_app.processEvents()

        tbl._revert_changes()
        qt_app.processEvents()

        row = _row_for_species(tbl, "A")
        assert tbl._table.item(row, _Col.INITIAL).text() == "9.9"
    finally:
        tbl.close()
        qt_app.processEvents()


def test_ic_error_clears_on_cell_edit(qt_app):
    mgr = _make_manager(["A", "B"])
    tbl = _make_table(manager=mgr)
    try:
        tbl.load_for_dataset("ds1")
        row = _row_for_species(tbl, "A")
        tbl._table.item(row, _Col.MIN).setText("abc")
        qt_app.processEvents()

        tbl._apply_changes()
        qt_app.processEvents()

        assert "numeric bounds" in tbl._footer.error_label.text().lower()

        tbl._table.item(row, _Col.INITIAL).setText("5.0")
        qt_app.processEvents()

        assert tbl._footer.error_label.text() == ""
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


def test_dataset_switch_discards_ic_pending(qt_app):
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
    mgr = _make_manager(
        by_dataset={
            "ds1": {
                "initial_conditions": {"A": 1.0},
                "fit_flags": {"A": False},
                "log10_flags": {},
                "bounds": {"A": (0.0, 10.0)},
            },
            "ds2": {
                "initial_conditions": {"A": 4.0},
                "fit_flags": {"A": False},
                "log10_flags": {},
                "bounds": {"A": (0.0, 10.0)},
            },
        }
    )
    tbl = _make_table(entries=entries, species=["A"], manager=mgr)
    try:
        tbl.load_for_dataset("ds1")
        row = _row_for_species(tbl, "A")
        tbl._table.item(row, _Col.INITIAL).setText("9.0")
        qt_app.processEvents()

        tbl.load_for_dataset("ds2")
        tbl.load_for_dataset("ds1")
        qt_app.processEvents()

        row = _row_for_species(tbl, "A")
        assert tbl._table.item(row, _Col.INITIAL).text() == "1"
        assert tbl._ic_pending["ds1"]["A"]["initial"] == pytest.approx(1.0)
    finally:
        tbl.close()
        qt_app.processEvents()


def test_apply_partial_targets_ok_ic_fail(qt_app):
    entries = _make_entries(selected=["A"])
    mgr = _make_manager(["A", "B"])
    tbl = _make_table(entries=entries, manager=mgr)
    try:
        tbl.load_for_dataset("ds1")
        row_b = _row_for_species(tbl, "B")
        tbl._table.item(row_b, _Col.INCLUDE).setCheckState(Qt.Checked)
        row_a = _row_for_species(tbl, "A")
        tbl._table.item(row_a, _Col.MIN).setText("abc")
        qt_app.processEvents()

        applied_targets = []
        applied_ic = []
        tbl.targetsApplied.connect(lambda: applied_targets.append(True))
        tbl.icApplied.connect(lambda *args: applied_ic.append(args))

        tbl._apply_changes()
        qt_app.processEvents()

        assert applied_targets == [True]
        assert applied_ic == []
        assert "numeric bounds" in tbl._footer.error_label.text().lower()
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


def test_non_modeled_column_excluded_from_rows(qt_app):
    entries = _make_entries(
        species_data={"A": np.linspace(1, 0, 5), "B": np.linspace(0.2, 0.9, 5), "pH": np.linspace(6.8, 7.2, 5)},
        selected=["A", "B"],
    )
    mgr = _make_manager(["A", "B"])
    tbl = _make_table(entries=entries, species=["A", "B"], manager=mgr, modeled_series={"A", "B"})
    try:
        tbl.load_for_dataset("ds1")

        assert tbl._table.rowCount() == 2
        species_names = [tbl._table.item(row, _Col.SPECIES).text() for row in range(tbl._table.rowCount())]
        assert species_names == ["A", "B"]
        assert "pH" not in species_names
    finally:
        tbl.close()
        qt_app.processEvents()


def test_row_order_mechanism_then_observed(qt_app):
    entries = _make_entries(species_data={"A": np.linspace(1, 0, 5), "B": np.linspace(0.2, 0.9, 5), "C": np.linspace(0.3, 0.7, 5), "D": np.linspace(0.8, 0.1, 5)})
    mgr = _make_manager(["A", "B"])
    tbl = _make_table(entries=entries, species=["B", "A"], manager=mgr, modeled_series={"A", "B", "C", "D"})
    try:
        tbl.load_for_dataset("ds1")
        assert [tbl._table.item(row, _Col.SPECIES).text() for row in range(tbl._table.rowCount())] == ["B", "A", "C", "D"]
    finally:
        tbl.close()
        qt_app.processEvents()


def test_bulk_all_includes_observed_only(qt_app):
    entries = _make_entries(species_data={"A": np.linspace(1, 0, 5), "B": np.linspace(0.2, 0.9, 5)}, selected=["A"])
    mgr = _make_manager(["A"])
    tbl = _make_table(entries=entries, species=["A"], manager=mgr, modeled_series={"A", "B"})
    try:
        tbl.load_for_dataset("ds1")
        tbl._apply_bulk_action("all")
        qt_app.processEvents()

        row_a = _row_for_species(tbl, "A")
        row_b = _row_for_species(tbl, "B")
        assert tbl._table.item(row_a, _Col.INCLUDE).checkState() == Qt.Checked
        assert tbl._table.item(row_b, _Col.INCLUDE).checkState() == Qt.Checked
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


def test_weight_editor_disabled_after_run_implicit_mode(qt_app):
    tbl = _make_table()
    try:
        tbl.load_for_dataset("ds1")
        tbl._weight_mode_combo.setCurrentIndex(0)
        qt_app.processEvents()

        tbl.set_running_state(True)
        tbl.set_running_state(False)
        qt_app.processEvents()

        assert tbl._dataset_weight_edit.isEnabled() is False
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


def test_parameter_defaults_read_live_settings(qt_app):
    mgr = _make_manager(
        by_dataset={
            "ds1": {
                "initial_conditions": {"A": 0.5, "B": 1.0},
                "fit_flags": {"A": False, "B": False},
                "log10_flags": {},
                "bounds": {"A": (0.0, 5.0), "B": (0.0, 10.0)},
            }
        }
    )
    tbl = _make_table(manager=mgr, species=["A", "B"])
    try:
        mgr.settings_by_dataset["ds1"].initial_conditions["A"] = 9.9
        fit_flag, defaults = tbl.initial_parameter_defaults_for_species("ds1", "A")

        assert fit_flag is False
        assert defaults["initial"] == pytest.approx(9.9)
    finally:
        tbl.close()
        qt_app.processEvents()


def test_tab_reentry_picks_up_external_changes(qt_app):
    mgr = _make_manager(["A", "B"])
    tbl = _make_table(manager=mgr)
    try:
        tbl.load_for_dataset("ds1")
        row = _row_for_species(tbl, "A")
        assert tbl._table.item(row, _Col.INITIAL).text() == "1"

        mgr.settings_by_dataset["ds1"].initial_conditions["A"] = 7.7
        tbl.on_tab_activated()
        qt_app.processEvents()

        row = _row_for_species(tbl, "A")
        assert tbl._table.item(row, _Col.INITIAL).text() == "7.7"
    finally:
        tbl.close()
        qt_app.processEvents()


def test_tab_reentry_preserves_dirty_edits(qt_app):
    mgr = _make_manager(["A", "B"])
    tbl = _make_table(manager=mgr)
    try:
        tbl.load_for_dataset("ds1")
        row = _row_for_species(tbl, "A")
        tbl._table.item(row, _Col.INITIAL).setText("8.8")
        qt_app.processEvents()

        tbl.on_tab_activated()
        qt_app.processEvents()

        row = _row_for_species(tbl, "A")
        assert tbl._table.item(row, _Col.INITIAL).text() == "8.8"
    finally:
        tbl.close()
        qt_app.processEvents()


def test_empty_session_clears_table(qt_app):
    tbl = _make_table()
    try:
        tbl.load_for_dataset("ds1")
        assert tbl._table.rowCount() == 2

        tbl.remove_dataset_state({"ds1"})
        qt_app.processEvents()

        assert tbl._table.rowCount() == 0
    finally:
        tbl.close()
        qt_app.processEvents()


def test_empty_session_clears_weight_controls(qt_app):
    tbl = _make_table()
    try:
        tbl.load_for_dataset("ds1")
        tbl._weight_mode_combo.setCurrentIndex(1)
        qt_app.processEvents()
        assert tbl._dataset_weight_edit.isEnabled() is True
        assert "ds1" in tbl._context_label.text().lower()

        tbl.remove_dataset_state({"ds1"})
        qt_app.processEvents()

        assert tbl._dataset_weight_edit.isEnabled() is False
        assert str(tbl._dataset_weight_edit.text() or "").strip() in {"", "1", "1.0"}
        assert "ds1" not in tbl._context_label.text().lower()
    finally:
        tbl.close()
        qt_app.processEvents()


def test_algebra_observable_still_shown(qt_app):
    entries = _make_entries(
        species_data={
            "A": np.linspace(1, 0, 5),
            "B": np.linspace(0.2, 0.9, 5),
            "selectivity": np.linspace(0.1, 0.5, 5),
        },
        selected=["A", "B", "selectivity"],
    )
    mgr = _make_manager(["A", "B"])
    tbl = _make_table(
        entries=entries,
        species=["A", "B"],
        manager=mgr,
        modeled_series={"A", "B", "selectivity"},
    )
    try:
        tbl.load_for_dataset("ds1")

        assert tbl._table.rowCount() == 3
        row = _row_for_species(tbl, "selectivity")
        assert tbl._table.item(row, _Col.INCLUDE).flags() & Qt.ItemIsEnabled
        assert tbl._table.item(row, _Col.WEIGHT).flags() & Qt.ItemIsEnabled
        assert not (tbl._table.item(row, _Col.INITIAL).flags() & Qt.ItemIsEnabled)
        assert not (tbl._table.item(row, _Col.FIT_IC).flags() & Qt.ItemIsEnabled)
        assert not (tbl._table.item(row, _Col.LOG10).flags() & Qt.ItemIsEnabled)
        assert not (tbl._table.item(row, _Col.MIN).flags() & Qt.ItemIsEnabled)
        assert not (tbl._table.item(row, _Col.MAX).flags() & Qt.ItemIsEnabled)
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


# ---------------------------------------------------------------------------
# Hidden-species pruning tests
# ---------------------------------------------------------------------------

def test_apply_prunes_hidden_species_from_applied(qt_app):
    """Hidden non-modeled species in pending must not reach applied state."""
    entries = _make_entries(
        species_data={
            "A": np.linspace(1, 0, 5),
            "B": np.linspace(0.2, 0.9, 5),
            "pH": np.linspace(6.8, 7.2, 5),
        },
        selected=["A", "B"],
    )
    mgr = _make_manager(["A", "B"])
    tbl = _make_table(entries=entries, species=["A", "B"], manager=mgr, modeled_series={"A", "B"})
    try:
        tbl.load_for_dataset("ds1")
        # Inject hidden species into pending
        tbl._fit_targets_selection_pending["ds1"].add("pH")
        tbl._fit_targets_dirty = True
        tbl._apply_changes()
        qt_app.processEvents()

        assert "pH" not in tbl._fit_targets_selection_applied.get("ds1", [])
        assert "A" in tbl._fit_targets_selection_applied["ds1"]
        assert "B" in tbl._fit_targets_selection_applied["ds1"]
    finally:
        tbl.close()
        qt_app.processEvents()


def test_bulk_all_excludes_hidden_species(qt_app):
    """Bulk All must not select hidden non-modeled species."""
    entries = _make_entries(
        species_data={
            "A": np.linspace(1, 0, 5),
            "B": np.linspace(0.2, 0.9, 5),
            "pH": np.linspace(6.8, 7.2, 5),
        },
        selected=["A"],
    )
    mgr = _make_manager(["A", "B"])
    tbl = _make_table(entries=entries, species=["A", "B"], manager=mgr, modeled_series={"A", "B"})
    try:
        tbl.load_for_dataset("ds1")
        tbl._apply_bulk_action("all")
        qt_app.processEvents()

        pending = tbl._fit_targets_selection_pending.get("ds1", set())
        assert "A" in pending
        assert "B" in pending
        assert "pH" not in pending
    finally:
        tbl.close()
        qt_app.processEvents()


def test_bulk_invert_excludes_hidden_species(qt_app):
    """Bulk Invert must not toggle hidden non-modeled species."""
    entries = _make_entries(
        species_data={
            "A": np.linspace(1, 0, 5),
            "B": np.linspace(0.2, 0.9, 5),
            "pH": np.linspace(6.8, 7.2, 5),
        },
        selected=["A"],
    )
    mgr = _make_manager(["A", "B"])
    tbl = _make_table(entries=entries, species=["A", "B"], manager=mgr, modeled_series={"A", "B"})
    try:
        tbl.load_for_dataset("ds1")
        tbl._apply_bulk_action("invert")
        qt_app.processEvents()

        pending = tbl._fit_targets_selection_pending.get("ds1", set())
        assert "B" in pending
        assert "A" not in pending
        assert "pH" not in pending
    finally:
        tbl.close()
        qt_app.processEvents()


def test_non_modeled_species_excluded_from_available_at_source(qt_app):
    """Non-modeled species are excluded from available_by_dataset at construction."""
    entries = _make_entries(
        species_data={
            "A": np.linspace(1, 0, 5),
            "B": np.linspace(0.2, 0.9, 5),
            "pH": np.linspace(6.8, 7.2, 5),
        },
        selected=["A", "B"],
    )
    mgr = _make_manager(["A", "B"])
    tbl = _make_table(entries=entries, species=["A", "B"], manager=mgr, modeled_series={"A", "B"})
    try:
        tbl.load_for_dataset("ds1")
        avail = set(tbl._fit_targets_available_by_dataset.get("ds1", []))
        assert "pH" not in avail, "pH should be excluded from fit-universe at source"
        assert "A" in avail
        assert "B" in avail
        # pH data is still in full_series for plotting
        assert "pH" in tbl._fit_targets_full_series_by_dataset.get("ds1", {})
    finally:
        tbl.close()
        qt_app.processEvents()


def test_set_mechanism_species_detects_modeled_series_change(qt_app):
    """set_mechanism_species must repopulate when modeled_series_getter output changes."""
    entries = _make_entries(
        species_data={
            "A": np.linspace(1, 0, 5),
            "B": np.linspace(0.2, 0.9, 5),
            "selectivity": np.linspace(0.1, 0.5, 5),
        },
        selected=["A", "B"],
    )
    mgr = _make_manager(["A", "B"])
    # Use mutable container so the getter result can change
    modeled = [{"A", "B"}]
    tbl = _make_table(entries=entries, species=["A", "B"], manager=mgr, modeled_series={"A", "B"})
    try:
        tbl._modeled_series_getter = lambda: modeled[0]
        tbl.load_for_dataset("ds1")
        assert tbl._table.rowCount() == 2

        # Change modeled set to include selectivity (simulates algebra observable added)
        modeled[0] = {"A", "B", "selectivity"}
        # Call with same species list -- should still repopulate
        tbl.set_mechanism_species(["A", "B"])

        assert tbl._table.rowCount() == 3
        species_names = [tbl._table.item(r, _Col.SPECIES).text() for r in range(3)]
        assert "selectivity" in species_names
    finally:
        tbl.close()
        qt_app.processEvents()


# ---------------------------------------------------------------------------
# Fit-universe invariant tests
# ---------------------------------------------------------------------------


def test_available_by_dataset_is_fit_universe(qt_app):
    """available_by_dataset must contain only observed AND modeled species."""
    entries = _make_entries(
        species_data={
            "A": np.linspace(1, 0, 5),
            "B": np.linspace(0.2, 0.9, 5),
            "pH": np.linspace(7.0, 7.5, 5),
        },
        selected=["A", "B"],
    )
    mgr = _make_manager(["A", "B"])
    tbl = _make_table(
        entries=entries,
        species=["A", "B"],
        manager=mgr,
        modeled_series={"A", "B"},
    )
    try:
        tbl.load_for_dataset("ds1")
        avail = tbl._fit_targets_available_by_dataset["ds1"]
        assert sorted(avail) == ["A", "B"], (
            f"pH should be excluded at source; got {avail}"
        )
    finally:
        tbl.close()
        qt_app.processEvents()


def test_initial_selection_cannot_contain_non_modeled(qt_app):
    """After first load, pending and applied selections contain only fit-universe species."""
    entries = _make_entries(
        species_data={
            "A": np.linspace(1, 0, 5),
            "pH": np.linspace(7.0, 7.5, 5),
        },
        selected=["A", "pH"],
    )
    mgr = _make_manager(["A"])
    tbl = _make_table(
        entries=entries,
        species=["A"],
        manager=mgr,
        modeled_series={"A"},
    )
    try:
        # Fit-universe is computed on first _populate_table (deferred from init)
        tbl.load_for_dataset("ds1")
        pending = tbl._fit_targets_selection_pending.get("ds1", set())
        applied = tbl._fit_targets_selection_applied.get("ds1", [])
        assert pending == {"A"}, f"Pending should be {{A}} only; got {pending}"
        assert applied == ["A"], f"Applied should be [A] only; got {applied}"
    finally:
        tbl.close()
        qt_app.processEvents()


def test_add_dataset_state_filters_to_fit_universe(qt_app):
    """add_dataset_state must filter available to fit-universe."""
    entries = _make_entries()
    mgr = _make_manager(["A", "B", "C"])
    tbl = _make_table(
        entries=entries,
        species=["A", "B", "C"],
        manager=mgr,
        modeled_series={"A", "B", "C"},
    )
    try:
        tbl.add_dataset_state(
            "ds2",
            full_series={
                "A": np.linspace(1, 0, 5),
                "C": np.linspace(0.1, 0.5, 5),
                "pH": np.linspace(7.0, 7.5, 5),
            },
            full_t=np.linspace(0, 1, 5),
            available=["A", "C", "pH"],
        )
        avail = tbl._fit_targets_available_by_dataset["ds2"]
        assert sorted(avail) == ["A", "C"], (
            f"pH should be excluded from fit-universe; got {avail}"
        )
    finally:
        tbl.close()
        qt_app.processEvents()


def test_set_mechanism_species_preserves_ic_when_only_modeled_changes(qt_app):
    """IC edits must survive when only modeled_series changes (not mechanism)."""
    entries = _make_entries(
        species_data={
            "A": np.linspace(1, 0, 5),
            "B": np.linspace(0.2, 0.9, 5),
        },
        selected=["A", "B"],
    )
    mgr = _make_manager(["A", "B"])
    modeled = [{"A", "B"}]
    tbl = _make_table(
        entries=entries, species=["A", "B"], manager=mgr, modeled_series={"A", "B"},
    )
    try:
        tbl._modeled_series_getter = lambda: modeled[0]
        tbl.load_for_dataset("ds1")

        # Manually edit IC pending for species A
        pending_ic = tbl._pending_ic_state_for_dataset("ds1")
        pending_ic["A"]["initial"] = 9.9
        tbl._ic_editor_dirty = True

        # Change modeled set (add observable), but keep mechanism species the same
        modeled[0] = {"A", "B", "selectivity"}
        tbl.set_mechanism_species(["A", "B"])

        # IC edits must survive
        refreshed = tbl._pending_ic_state_for_dataset("ds1")
        assert refreshed["A"]["initial"] == 9.9, (
            f"IC edit lost; got {refreshed['A']['initial']}"
        )
        assert tbl._ic_editor_dirty is True, "IC dirty flag was reset"
    finally:
        tbl.close()
        qt_app.processEvents()


def test_set_mechanism_species_reseeds_ic_when_mechanism_changes(qt_app):
    """IC must be reseeded from dataset_manager when mechanism species change."""
    entries = _make_entries(
        species_data={
            "A": np.linspace(1, 0, 5),
            "B": np.linspace(0.2, 0.9, 5),
        },
        selected=["A", "B"],
    )
    mgr = _make_manager(["A", "B", "C"])
    tbl = _make_table(
        entries=entries, species=["A", "B"], manager=mgr, modeled_series={"A", "B"},
    )
    try:
        tbl.load_for_dataset("ds1")

        # Edit IC pending
        pending_ic = tbl._pending_ic_state_for_dataset("ds1")
        pending_ic["A"]["initial"] = 9.9
        tbl._ic_editor_dirty = True

        # Change mechanism species (add C)
        tbl.set_mechanism_species(["A", "B", "C"])

        refreshed = tbl._pending_ic_state_for_dataset("ds1")
        assert refreshed["A"]["initial"] != 9.9, (
            "IC edit should have been reseeded from manager"
        )
        assert tbl._ic_editor_dirty is False, "IC dirty flag should be reset"
    finally:
        tbl.close()
        qt_app.processEvents()


def test_recompute_fit_universe_prunes_all_datasets(qt_app):
    """When modeled_series shrinks, ALL datasets must have stale selections pruned."""
    t = np.linspace(0, 1, 5)
    entries = [
        {
            "id": "ds1", "label": "DS 1", "t": t,
            "species_data": {
                "A": np.linspace(1, 0, 5),
                "B": np.linspace(0.2, 0.9, 5),
                "X": np.linspace(0.5, 0.3, 5),
            },
            "selected_species": ["A", "B", "X"], "weight": 1.0, "include": True,
        },
        {
            "id": "ds2", "label": "DS 2", "t": t,
            "species_data": {
                "A": np.linspace(0.5, 1, 5),
                "X": np.linspace(0.1, 0.2, 5),
            },
            "selected_species": ["A", "X"], "weight": 1.0, "include": True,
        },
    ]
    modeled = [{"A", "B", "X"}]
    mgr = _make_manager(
        ["A", "B"],
        by_dataset={
            "ds1": {"initial_conditions": {"A": 1.0, "B": 0.5}, "fit_flags": {}, "log10_flags": {}, "bounds": {}},
            "ds2": {"initial_conditions": {"A": 1.0, "B": 0.5}, "fit_flags": {}, "log10_flags": {}, "bounds": {}},
        },
    )
    tbl = _make_table(
        entries=entries, species=["A", "B"], manager=mgr,
        modeled_series={"A", "B", "X"},
    )
    try:
        tbl._modeled_series_getter = lambda: modeled[0]
        tbl.load_for_dataset("ds1")

        # Apply targets so that X is in applied selection for both datasets
        tbl._apply_changes()

        # Verify X is in applied for both
        assert "X" in tbl._fit_targets_selection_applied.get("ds1", [])
        assert "X" in tbl._fit_targets_selection_applied.get("ds2", [])

        # Now shrink modeled (remove X)
        modeled[0] = {"A", "B"}
        tbl.set_mechanism_species(["A", "B"])

        # X must be pruned from ALL datasets' pending and applied
        for ds_id in ["ds1", "ds2"]:
            pending = tbl._fit_targets_selection_pending.get(ds_id, set())
            applied = tbl._fit_targets_selection_applied.get(ds_id, [])
            assert "X" not in pending, f"X still in pending for {ds_id}: {pending}"
            assert "X" not in applied, f"X still in applied for {ds_id}: {applied}"
    finally:
        tbl.close()
        qt_app.processEvents()


def test_first_set_mechanism_species_noop_after_cache_seeded(qt_app):
    """After cache is seeded, set_mechanism_species with same args is a no-op."""
    entries = _make_entries()
    mgr = _make_manager(["A", "B"])
    tbl = _make_table(
        entries=entries, species=["A", "B"], manager=mgr, modeled_series={"A", "B"},
    )
    try:
        tbl.load_for_dataset("ds1")

        # Edit IC
        pending_ic = tbl._pending_ic_state_for_dataset("ds1")
        pending_ic["A"]["initial"] = 7.7
        tbl._ic_editor_dirty = True

        # Call with identical species + identical modeled -- should be a true no-op
        tbl.set_mechanism_species(["A", "B"])

        refreshed = tbl._pending_ic_state_for_dataset("ds1")
        assert refreshed["A"]["initial"] == 7.7, (
            f"IC edit was destroyed by spurious reset; got {refreshed['A']['initial']}"
        )
        assert tbl._ic_editor_dirty is True, "IC dirty flag was reset by no-op call"
    finally:
        tbl.close()
        qt_app.processEvents()


# ---------------------------------------------------------------------------
# Validation foreground delegate
# ---------------------------------------------------------------------------


def test_species_table_has_validation_delegate(qt_app):
    """UnifiedSpeciesTable must install _ValidationCellDelegate on its QTableWidget."""
    from kindred.gui.fitting.unified_species_table import _ValidationCellDelegate

    tbl = _make_table()
    try:
        assert isinstance(tbl._table.itemDelegate(), _ValidationCellDelegate)
    finally:
        tbl.close()
        qt_app.processEvents()


def test_species_table_delegate_reads_foreground_role(qt_app):
    """Invalid weight cells store _INVALID_FG via ForegroundRole."""
    from PySide6.QtGui import QBrush, QColor

    mgr = _make_manager(species=["A"])
    tbl = _make_table(manager=mgr, species=["A"])
    try:
        tbl.load_for_dataset("ds1")
        row = _row_for_species(tbl, "A")

        # Set an invalid weight value — _on_table_item_changed triggers
        # validation and calls setForeground(_INVALID_FG) on the weight cell.
        weight_item = tbl._table.item(row, _Col.WEIGHT)
        assert weight_item is not None
        weight_item.setText("abc")
        qt_app.processEvents()

        weight_item = tbl._table.item(row, _Col.WEIGHT)
        fg = weight_item.data(Qt.ForegroundRole)
        assert isinstance(fg, QBrush), f"Expected QBrush, got {type(fg)}"
        assert fg.color() == QColor(80, 0, 0), f"Expected dark red (80,0,0), got {fg.color().getRgb()}"
    finally:
        tbl.close()
        qt_app.processEvents()


# ---------------------------------------------------------------------------
# Stale-applied-targets bug — empty pending must promote into applied
# ---------------------------------------------------------------------------

def test_apply_empty_targets_clears_applied_state(qt_app):
    """Unchecking all targets and applying must clear applied state, not keep stale targets."""
    entries = _make_entries(selected=["A", "B"])
    mgr = _make_manager(species=["A", "B"])
    tbl = _make_table(entries=entries, manager=mgr)
    try:
        # Initial applied has A, B
        assert sorted(tbl._fit_targets_selection_applied.get("ds1", [])) == ["A", "B"]

        # Uncheck all targets — set pending to empty
        tbl._fit_targets_selection_pending["ds1"] = set()
        tbl._fit_targets_dirty = True

        result = tbl._apply_targets()
        assert result is True

        # Applied must now be empty — not stale ["A", "B"]
        assert tbl._fit_targets_selection_applied.get("ds1", []) == []
        assert tbl._fit_target_weights_applied.get("ds1", {}) == {}
        assert "ds1" in tbl.invalid_applied_used_dataset_ids()

        # Dirty state must be clean: pending=empty matches applied=empty
        tbl._update_fit_targets_dirty_state()
        assert tbl._fit_targets_dirty is False

        # Round-trip recovery: re-select one target and apply again
        tbl._fit_targets_selection_pending["ds1"] = {"A"}
        tbl._fit_targets_dirty = True
        tbl._apply_targets()
        assert tbl._fit_targets_selection_applied.get("ds1", []) == ["A"]
        assert "ds1" not in tbl.invalid_applied_used_dataset_ids()
    finally:
        tbl.close()
        qt_app.processEvents()


def test_apply_empty_targets_one_of_two_datasets(qt_app):
    """Empty apply on one dataset must not affect the other dataset's applied state."""
    t = np.linspace(0, 1, 5)
    entries = [
        {
            "id": "ds1", "label": "DS 1", "t": t,
            "species_data": {"A": np.linspace(1, 0.5, t.size), "B": np.linspace(0.2, 0.9, t.size)},
            "selected_species": ["A", "B"], "weight": 1.0, "include": True,
        },
        {
            "id": "ds2", "label": "DS 2", "t": t,
            "species_data": {"A": np.linspace(0.8, 0.3, t.size), "B": np.linspace(0.1, 0.7, t.size)},
            "selected_species": ["A", "B"], "weight": 1.0, "include": True,
        },
    ]
    mgr = _make_manager(
        by_dataset={
            "ds1": {"initial_conditions": {"A": 1.0, "B": 0.2}, "fit_flags": {}, "log10_flags": {}, "bounds": {}},
            "ds2": {"initial_conditions": {"A": 0.8, "B": 0.1}, "fit_flags": {}, "log10_flags": {}, "bounds": {}},
        }
    )
    tbl = _make_table(entries=entries, manager=mgr, included_ids=["ds1", "ds2"])
    try:
        assert sorted(tbl._fit_targets_selection_applied.get("ds1", [])) == ["A", "B"]
        assert sorted(tbl._fit_targets_selection_applied.get("ds2", [])) == ["A", "B"]

        # Uncheck all for ds1 only
        tbl._fit_targets_selection_pending["ds1"] = set()
        tbl._fit_targets_dirty = True

        tbl._apply_targets()

        # ds1 applied must be empty; ds2 preserved
        assert tbl._fit_targets_selection_applied.get("ds1", []) == []
        assert sorted(tbl._fit_targets_selection_applied.get("ds2", [])) == ["A", "B"]

        invalid = tbl.invalid_applied_used_dataset_ids()
        assert "ds1" in invalid
        assert "ds2" not in invalid
    finally:
        tbl.close()
        qt_app.processEvents()
