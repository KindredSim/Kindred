"""Regression tests for the layout & visual polish slice.

Covers:
  A) Sampling panel relocated to left panel
  B) Validation highlighting sets explicit foreground
  C) Blank dataset label normalization
  D) _recompute_fit_universe ordering in _populate_table
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from PySide6 import QtCore, QtGui, QtWidgets

from kindred.gui.fitting.unified_dataset_list import UnifiedDatasetList
from kindred.gui.fitting.unified_species_table import UnifiedSpeciesTable, _Col


pytestmark = [pytest.mark.gui]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sample_entries():
    return [
        {"id": "ds1", "label": "Dataset 1", "include": True},
        {"id": "ds2", "label": "Dataset 2", "include": False},
    ]


def _make_fitting_window():
    from kindred.gui.fitting.window import FittingWindow

    t = np.linspace(0.0, 1.0, 5)
    y_a = np.linspace(1.0, 0.5, t.size)
    y_b = np.linspace(0.2, 0.9, t.size)
    return FittingWindow(
        mode="global",
        parameter_defs=[{"name": "k1", "value": 1.23, "min": 0.01, "max": 10.0}],
        dataset_entries=[
            {
                "id": "ds1",
                "label": "Dataset 1",
                "t": t.copy(),
                "species_data": {"A": y_a.copy()},
                "selected_species": ["A"],
                "weight": 1.0,
                "include": True,
            },
            {
                "id": "ds2",
                "label": "Dataset 2",
                "t": t.copy(),
                "species_data": {"B": y_b.copy()},
                "selected_species": ["B"],
                "weight": 0.5,
                "include": True,
            },
        ],
        simulation_func=lambda _params: {"t": t.copy(), "species": {"A": y_a.copy(), "B": y_b.copy()}},
        dataset_payloads=[
            {"id": "ds1", "t": t.copy(), "y": np.vstack([y_a.copy()]), "species": ["A"]},
            {"id": "ds2", "t": t.copy(), "y": np.vstack([y_b.copy()]), "species": ["B"]},
        ],
        dataset_weights={"ds1": 1.0, "ds2": 0.5},
    )


def _make_species_entries(species_data=None, ds_id="ds1"):
    if species_data is None:
        t = np.linspace(0, 1, 5)
        species_data = {"A": np.linspace(1, 0.5, t.size), "B": np.linspace(0.2, 0.9, t.size)}
    else:
        t = np.linspace(0, 1, len(next(iter(species_data.values()))))
    return [{
        "id": ds_id,
        "label": "DS 1",
        "t": t,
        "species_data": species_data,
        "selected_species": list(species_data.keys()),
        "weight": 1.0,
        "include": True,
    }]


def _make_species_table(*, entries=None, species=None, modeled_series=None, manager=None):
    import inspect

    if entries is None:
        entries = _make_species_entries()
    if species is None:
        species = sorted({s for e in entries for s in (e.get("species_data") or {}).keys()})
    ds_ids = [str(e["id"]) for e in entries]
    if modeled_series is None:
        modeled_series = {str(name) for name in (species or []) if str(name).strip()}
    weights = {str(e["id"]): float(e.get("weight", 1.0)) for e in entries}
    persisted_weights: dict[str, float] = {}

    if manager is None:
        class FakeManager:
            def get_fit_settings(self, dataset_id):
                return SimpleNamespace(
                    initial_conditions={s: 1.0 for s in species},
                    fit_flags={s: False for s in species},
                    log10_flags={},
                    bounds={},
                )
            def update_fit_settings(self, dataset_id, settings):
                pass
        manager = FakeManager()

    kwargs = dict(
        dataset_entries=list(entries),
        mechanism_species=list(species),
        dataset_entries_getter=lambda: list(entries),
        included_dataset_ids_getter=lambda: list(ds_ids),
        dataset_label_getter=lambda ds_id: str(ds_id),
        dataset_weight_getter=lambda ds_id: weights.get(str(ds_id), 1.0),
        persist_dataset_weight_callback=lambda ds_id, w: persisted_weights.update({ds_id: w}),
        dataset_manager_getter=lambda: manager,
        worker_running_getter=lambda: False,
    )
    if "modeled_series_getter" in inspect.signature(UnifiedSpeciesTable).parameters:
        kwargs["modeled_series_getter"] = lambda: set(modeled_series)
    return UnifiedSpeciesTable(**kwargs)


# ---------------------------------------------------------------------------
# A) Sampling panel in left panel
# ---------------------------------------------------------------------------


def test_sampling_panel_in_left_panel(qt_app):
    """Sampling panel should be parented under the left panel, not the right scroll area."""
    window = _make_fitting_window()
    try:
        window.show()
        qt_app.processEvents()

        dt_tab = window._data_targets_tab
        sampling_panel = window._data_tab._sampling_panel_widget

        # Sampling panel must be visible.
        assert sampling_panel.isVisible(), "Sampling panel should be visible"

        # Sampling panel must be in the left side of the splitter (not the right scroll area).
        splitter = dt_tab.findChild(QtWidgets.QSplitter)
        assert splitter is not None
        left_widget = splitter.widget(0)
        right_widget = splitter.widget(1)

        # The sampling panel should be an ancestor-descendant of the left widget.
        assert left_widget.isAncestorOf(sampling_panel), (
            "Sampling panel should be in the left panel"
        )
        assert not right_widget.isAncestorOf(sampling_panel), (
            "Sampling panel should NOT be in the right scroll area"
        )
    finally:
        window.close()


# ---------------------------------------------------------------------------
# B) Validation highlighting sets foreground
# ---------------------------------------------------------------------------


def test_validation_highlight_sets_foreground(qt_app):
    """set_validation_state must set foreground alongside background for dark-theme safety."""
    widget = UnifiedDatasetList()
    widget.populate(_sample_entries())

    widget.set_validation_state("ds1", "invalid_applied")
    item = widget._list.item(0)
    assert item is not None
    fg_color = item.foreground().color()
    assert fg_color == QtGui.QColor(80, 0, 0), (
        f"Expected dark foreground QColor(80,0,0), got {fg_color.name()}"
    )

    widget.set_validation_state("ds2", "invalid_pending")
    item2 = widget._list.item(1)
    assert item2 is not None
    fg_color2 = item2.foreground().color()
    assert fg_color2 == QtGui.QColor(80, 60, 0), (
        f"Expected dark foreground QColor(80,60,0), got {fg_color2.name()}"
    )

    # Clearing should restore default foreground.
    widget.set_validation_state("ds1", "")
    assert item.foreground() == QtGui.QBrush(), "Foreground should be restored to default"
    widget.close()


def test_species_table_invalid_weight_sets_foreground(qt_app):
    """Invalid weight cells must set explicit foreground for dark-theme safety."""
    tbl = _make_species_table()
    try:
        tbl.load_for_dataset("ds1")
        qt_app.processEvents()

        # Find a species row that has data (can have weight edited).
        available = set(tbl._fit_targets_available_by_dataset.get("ds1", []))
        target_species = None
        target_row = None
        for row, species in enumerate(tbl._current_row_species):
            if species in available:
                target_species = species
                target_row = row
                break
        assert target_species is not None, "No available species found for weight test"

        # Set invalid weight text.
        weight_item = tbl._table.item(target_row, _Col.WEIGHT)
        assert weight_item is not None
        tbl._table.blockSignals(False)
        weight_item.setText("abc")
        qt_app.processEvents()

        fg = weight_item.foreground().color()
        assert fg == QtGui.QColor(80, 0, 0), (
            f"Expected dark foreground on invalid weight, got {fg.name()}"
        )

        # Set valid weight — foreground should be restored.
        weight_item.setText("1.5")
        qt_app.processEvents()

        assert weight_item.foreground() == QtGui.QBrush(), (
            "Foreground should be restored to default for valid weight"
        )
    finally:
        tbl.close()
        qt_app.processEvents()


# ---------------------------------------------------------------------------
# C) Blank dataset label normalization
# ---------------------------------------------------------------------------


def test_blank_label_normalization(qt_app):
    """Dataset entries with whitespace-only labels should fall back to dataset ID."""
    from kindred.gui.fitting.window import FittingWindow

    t = np.linspace(0.0, 1.0, 5)
    y_a = np.linspace(1.0, 0.5, t.size)
    window = FittingWindow(
        mode="global",
        parameter_defs=[{"name": "k1", "value": 1.0, "min": 0.0, "max": 10.0}],
        dataset_entries=[
            {
                "id": "ds_whitespace",
                "label": "   ",
                "t": t.copy(),
                "species_data": {"A": y_a.copy()},
                "selected_species": ["A"],
                "weight": 1.0,
                "include": True,
            },
        ],
        simulation_func=lambda _params: {"t": t.copy(), "species": {"A": y_a.copy()}},
        dataset_payloads=[
            {"id": "ds_whitespace", "t": t.copy(), "y": np.vstack([y_a.copy()]), "species": ["A"]},
        ],
        dataset_weights={"ds_whitespace": 1.0},
    )
    try:
        # After normalization, label must not be whitespace.
        for entry in window._dataset_entries:
            if entry["id"] == "ds_whitespace":
                label = str(entry.get("label", "")).strip()
                assert label, "Label should not be blank after normalization"
                assert label == "ds_whitespace", f"Expected fallback to ID, got {label!r}"
                break
        else:
            pytest.fail("ds_whitespace entry not found")
    finally:
        window.close()


# ---------------------------------------------------------------------------
# D) _recompute_fit_universe before row building
# ---------------------------------------------------------------------------


def test_recompute_fit_universe_before_row_building(qt_app):
    """_recompute_fit_universe should run before row building so that
    _fit_targets_available_by_dataset is pruned to modeled-only BEFORE
    the Include checkboxes read it. With the old ordering (finally block),
    available_by_dataset contains unmodeled species during row building."""
    t = np.linspace(0, 1, 5)
    entries = [{
        "id": "ds1",
        "label": "DS 1",
        "t": t,
        "species_data": {
            "A": np.linspace(1, 0.5, t.size),
            "B": np.linspace(0.2, 0.9, t.size),
            "pH": np.linspace(7.0, 6.5, t.size),
        },
        "selected_species": ["A", "B"],
        "weight": 1.0,
        "include": True,
    }]
    # mechanism has A, B, and pH; modeled is only A and B.
    # Before _recompute_fit_universe, available = ["A","B","pH"] (all observed).
    # After _recompute_fit_universe, available = ["A","B"] (observed AND modeled).
    # With old ordering, pH row would get has_data=True on first populate
    # (because available hasn't been pruned yet).
    tbl = _make_species_table(
        entries=entries,
        species=["A", "B", "pH"],
        modeled_series={"A", "B"},
    )
    try:
        tbl.load_for_dataset("ds1")
        qt_app.processEvents()

        # After first load, _fit_targets_available_by_dataset should be pruned
        # to modeled-only (the intersection of observed and modeled).
        available = tbl._fit_targets_available_by_dataset.get("ds1", [])
        assert "pH" not in available, (
            f"pH should not be in available_by_dataset after fit-universe prune; got {available}"
        )

        # Verify pH row exists (it's a mechanism species) but is NOT checkable
        # for Include (has_data should be False because pH is not in available).
        ph_row = None
        for row, species in enumerate(tbl._current_row_species):
            if species == "pH":
                ph_row = row
                break
        assert ph_row is not None, "pH should appear as a mechanism species row"
        include_item = tbl._table.item(ph_row, _Col.INCLUDE)
        assert include_item is not None
        # If _recompute ran before row building, pH is not in available,
        # so has_data=False and Include should not be user-checkable.
        is_checkable = bool(include_item.flags() & QtCore.Qt.ItemIsUserCheckable)
        assert not is_checkable, (
            "pH Include checkbox should NOT be checkable (not in fit universe)"
        )
    finally:
        tbl.close()
        qt_app.processEvents()
