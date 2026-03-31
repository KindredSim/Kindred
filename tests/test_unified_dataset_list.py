"""Tests for UnifiedDatasetList and the unified master/detail layout."""

import numpy as np
import pytest
from unittest import mock

from PySide6 import QtCore, QtGui, QtWidgets

from kindred.gui.fitting.unified_dataset_list import UnifiedDatasetList
from kindred.gui.fitting.window import FittingWindow


pytestmark = [pytest.mark.gui]


def _sample_entries():
    return [
        {"id": "ds1", "label": "Dataset 1", "include": True},
        {"id": "ds2", "label": "Dataset 2", "include": False},
        {"id": "ds3", "label": "Dataset 3", "include": True},
    ]


def _make_fitting_window():
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


# ------------------------------------------------------------------
# (a) Population
# ------------------------------------------------------------------


def test_populate_builds_items_with_correct_data(qt_app):
    widget = UnifiedDatasetList()
    entries = _sample_entries()
    widget.populate(entries)

    assert widget._list.count() == 3
    for i, entry in enumerate(entries):
        item = widget._list.item(i)
        assert item is not None
        assert item.text() == entry["label"]
        assert str(item.data(QtCore.Qt.UserRole) or "") == entry["id"]
        expected_state = QtCore.Qt.Checked if entry["include"] else QtCore.Qt.Unchecked
        assert item.checkState() == expected_state
    widget.close()


# ------------------------------------------------------------------
# (b) Selection signal
# ------------------------------------------------------------------


def test_selection_emits_current_dataset_changed(qt_app):
    widget = UnifiedDatasetList()
    widget.populate(_sample_entries())
    received = []
    widget.currentDatasetChanged.connect(lambda ds_id: received.append(ds_id))

    widget._list.setCurrentRow(2)
    qt_app.processEvents()

    assert "ds3" in received
    widget.close()


# ------------------------------------------------------------------
# (c) Include toggle
# ------------------------------------------------------------------


def test_include_toggle_emits_dataset_include_changed(qt_app):
    widget = UnifiedDatasetList()
    widget.populate(_sample_entries())
    received = []
    widget.datasetIncludeChanged.connect(lambda row, ds_id, inc: received.append((row, ds_id, inc)))

    item = widget._list.item(0)
    assert item is not None
    item.setCheckState(QtCore.Qt.Unchecked)
    qt_app.processEvents()

    assert len(received) == 1
    assert received[0] == (0, "ds1", False)
    widget.close()


# ------------------------------------------------------------------
# (d) Validation coloring
# ------------------------------------------------------------------


def test_validation_state_coloring(qt_app):
    widget = UnifiedDatasetList()
    widget.populate(_sample_entries())

    widget.set_validation_state("ds1", "invalid_applied")
    item = widget._list.item(0)
    assert item is not None
    assert item.background().color() == QtGui.QColor(255, 225, 225)

    widget.set_validation_state("ds2", "invalid_pending")
    item2 = widget._list.item(1)
    assert item2 is not None
    assert item2.background().color() == QtGui.QColor(255, 245, 210)

    widget.set_validation_state("ds1", "")
    assert item.background() == QtGui.QBrush()
    widget.close()


# ------------------------------------------------------------------
# (e) DataTargetsTab layout
# ------------------------------------------------------------------


def test_data_targets_tab_unified_layout(qt_app):
    window = _make_fitting_window()
    try:
        window.show()
        qt_app.processEvents()

        dt_tab = window._data_targets_tab

        # No QTabWidget subtabs.
        assert dt_tab.findChild(QtWidgets.QTabWidget) is None

        # QSplitter present.
        splitter = dt_tab.findChild(QtWidgets.QSplitter)
        assert splitter is not None

        # UnifiedDatasetList present.
        assert dt_tab.unified_list is not None
        assert dt_tab.isAncestorOf(dt_tab.unified_list)

        # Species table is a child of the right panel.
        assert dt_tab.isAncestorOf(dt_tab.species_table)

        # Sampling panel is reparented into the left panel.
        sampling_panel = dt_tab.data_tab._sampling_panel_widget
        assert dt_tab.isAncestorOf(sampling_panel)
    finally:
        window.close()


# ------------------------------------------------------------------
# (f) Dataset switching integration
# ------------------------------------------------------------------


def test_unified_list_drives_all_panels(qt_app):
    window = _make_fitting_window()
    try:
        window.show()
        qt_app.processEvents()

        unified_list = window._data_targets_tab.unified_list

        # Select ds2 via unified list.
        unified_list.select_dataset("ds2")
        qt_app.processEvents()

        # DataTab sampling reflects ds2.
        assert window._data_tab._sampling_current_dataset_id == "ds2"

        # TargetsWeightsTab reflects ds2.
        assert window._species_table._current_dataset_id == "ds2"

        # InitialConditionsPanel reflects ds2.
        assert window._species_table._ic_editor_current_dataset_id == "ds2"
    finally:
        window.close()


# ------------------------------------------------------------------
# (g) IC desync guard
# ------------------------------------------------------------------


def test_ic_load_for_dataset_rejects_unknown_id(qt_app):
    window = _make_fitting_window()
    try:
        window.show()
        qt_app.processEvents()

        ic = window._species_table
        original_id = ic._ic_editor_current_dataset_id

        ic.load_for_dataset("nonexistent_dataset")

        assert ic._ic_editor_current_dataset_id == original_id
    finally:
        window.close()


# ------------------------------------------------------------------
# (h) Add/Remove signal wiring
# ------------------------------------------------------------------


def test_add_remove_signals_connected(qt_app):
    window = _make_fitting_window()
    try:
        unified_list = window._data_targets_tab.unified_list

        # Verify addRequested is connected by checking isSignalConnected.
        add_meta = unified_list.metaObject().method(
            unified_list.metaObject().indexOfSignal("addRequested()")
        )
        assert unified_list.isSignalConnected(add_meta)

        remove_meta = unified_list.metaObject().method(
            unified_list.metaObject().indexOfSignal("removeRequested(QVariantList)")
        )
        assert unified_list.isSignalConnected(remove_meta)
    finally:
        window.close()


# ------------------------------------------------------------------
# select_dataset syncs hidden table selection
# ------------------------------------------------------------------


def test_select_dataset_syncs_hidden_table_selection(qt_app):
    window = _make_fitting_window()
    try:
        window.show()
        qt_app.processEvents()

        window._data_tab.select_dataset("ds2")
        qt_app.processEvents()

        assert window._data_tab.selected_dataset_id() == "ds2"
        header = window._data_tab._sampling_header_label.text()
        assert "ds2" in header.lower() or "Dataset 2" in header
    finally:
        window.close()


# ------------------------------------------------------------------
# Remove button disabled during fit run
# ------------------------------------------------------------------


def test_remove_button_disabled_during_fit_run(qt_app):
    widget = UnifiedDatasetList()
    widget.populate(_sample_entries())
    widget._list.setCurrentRow(0)
    qt_app.processEvents()

    assert widget._remove_button.isEnabled()
    assert widget._add_button.isEnabled()

    widget.set_running_state(True)
    assert not widget._remove_button.isEnabled()
    assert not widget._add_button.isEnabled()

    widget.set_running_state(False)
    assert widget._remove_button.isEnabled()
    assert widget._add_button.isEnabled()
    widget.close()


# ------------------------------------------------------------------
# IC dirty state cleared on external switch
# ------------------------------------------------------------------


def test_ic_dirty_state_cleared_on_external_switch(qt_app):
    window = _make_fitting_window()
    try:
        window.show()
        qt_app.processEvents()

        unified_list = window._data_targets_tab.unified_list
        ic = window._species_table

        unified_list.select_dataset("ds1")
        qt_app.processEvents()

        ic._ic_editor_dirty = True
        assert ic._ic_editor_dirty is True

        unified_list.select_dataset("ds2")
        qt_app.processEvents()

        assert ic._ic_editor_dirty is False
    finally:
        window.close()


# ------------------------------------------------------------------
# SingleSelection mode
# ------------------------------------------------------------------


def test_single_selection_mode(qt_app):
    widget = UnifiedDatasetList()
    widget.populate(_sample_entries())
    qt_app.processEvents()

    assert widget._list.selectionMode() == QtWidgets.QAbstractItemView.SelectionMode.SingleSelection

    # Select A, then click B — only B should be selected.
    widget._list.setCurrentRow(0)
    qt_app.processEvents()
    widget._list.setCurrentRow(1)
    qt_app.processEvents()

    selected = widget._list.selectedItems()
    assert len(selected) == 1
    assert str(selected[0].data(QtCore.Qt.UserRole) or "") == "ds2"
    widget.close()


def test_remove_emits_single_dataset(qt_app):
    widget = UnifiedDatasetList()
    widget.populate(_sample_entries())
    widget._list.setCurrentRow(1)  # select ds2
    qt_app.processEvents()

    received = []
    widget.removeRequested.connect(lambda ids: received.append(ids))

    widget._on_remove_clicked()
    qt_app.processEvents()

    assert len(received) == 1
    assert received[0] == ["ds2"]
    widget.close()


def test_selected_dataset_id_reads_current_item(qt_app):
    widget = UnifiedDatasetList()
    widget.populate(_sample_entries())
    qt_app.processEvents()

    widget._list.setCurrentRow(1)
    qt_app.processEvents()
    assert widget.selected_dataset_id() == "ds2"

    widget._list.setCurrentRow(2)
    qt_app.processEvents()
    assert widget.selected_dataset_id() == "ds3"
    widget.close()


# ------------------------------------------------------------------
# Sampling survives dataset add/remove
# ------------------------------------------------------------------


def test_sampling_survives_dataset_add_remove(qt_app):
    window = _make_fitting_window()
    try:
        window.show()
        qt_app.processEvents()

        unified_list = window._data_targets_tab.unified_list
        unified_list.select_dataset("ds1")
        qt_app.processEvents()

        assert window._data_tab.selected_dataset_id() == "ds1"
        original_header = window._data_tab._sampling_header_label.text()

        # Add a third dataset via the internal session method.
        t3 = np.linspace(0.0, 1.0, 5)
        y_c = np.linspace(0.3, 0.7, t3.size)
        window._loaded_dataset_pool["ds3"] = {
            "id": "ds3",
            "label": "Dataset 3",
            "t": t3.copy(),
            "species_data": {"C": y_c.copy()},
        }
        window._add_datasets_to_session(["ds3"])
        qt_app.processEvents()

        # After add, ds1 should still be selected and sampling should not be blank.
        assert window._data_tab.selected_dataset_id() == "ds1"
        assert window._data_tab._sampling_header_label.text() == original_header
    finally:
        window.close()


# ------------------------------------------------------------------
# Sampling clears when last dataset removed
# ------------------------------------------------------------------


def test_sampling_clears_when_last_dataset_removed(qt_app):
    window = _make_fitting_window()
    try:
        window.show()
        qt_app.processEvents()

        # Remove both datasets.
        window._remove_datasets_from_session(["ds1", "ds2"])
        qt_app.processEvents()

        # Unified list should be empty.
        assert window._data_targets_tab.unified_list._list.count() == 0

        # Sampling header should show the blank state.
        header = window._data_tab._sampling_header_label.text()
        assert "\u2014" in header, f"Expected blank dash in sampling header, got: {header!r}"

        # Sampling controls should be disabled.
        assert not window._data_tab._sampling_t_min_spin.isEnabled()

        # IC table should be empty / blank.
        assert window._species_table._ic_editor_current_dataset_id is None
    finally:
        window.close()


# ------------------------------------------------------------------
# Active dataset updates on plain click
# ------------------------------------------------------------------


def test_active_dataset_updates_on_plain_click(qt_app):
    widget = UnifiedDatasetList()
    widget.populate(_sample_entries())
    widget._list.setCurrentRow(0)
    qt_app.processEvents()

    assert widget.selected_dataset_id() == "ds1"

    received = []
    widget.currentDatasetChanged.connect(lambda ds_id: received.append(ds_id))

    widget._list.setCurrentRow(1)
    qt_app.processEvents()

    assert widget.selected_dataset_id() == "ds2"
    assert len(received) == 1
    assert received[0] == "ds2"
    widget.close()


# ------------------------------------------------------------------
# Active dataset follows populate when removed
# ------------------------------------------------------------------


def test_active_dataset_follows_populate_when_removed(qt_app):
    widget = UnifiedDatasetList()
    entries = _sample_entries()
    widget.populate(entries)
    widget._list.setCurrentRow(1)  # select ds2
    qt_app.processEvents()

    assert widget.selected_dataset_id() == "ds2"

    received = []
    widget.currentDatasetChanged.connect(lambda ds_id: received.append(ds_id))

    # Remove ds2.
    widget.populate([e for e in entries if e["id"] != "ds2"])
    qt_app.processEvents()

    assert widget.selected_dataset_id() == "ds1"
    assert len(received) == 1
    assert received[0] == "ds1"
    widget.close()


# ------------------------------------------------------------------
# Active dataset stable across populate with same list
# ------------------------------------------------------------------


def test_active_dataset_stable_across_populate_same_list(qt_app):
    widget = UnifiedDatasetList()
    widget.populate(_sample_entries())
    widget._list.setCurrentRow(1)  # select ds2
    qt_app.processEvents()

    assert widget.selected_dataset_id() == "ds2"

    received = []
    widget.currentDatasetChanged.connect(lambda ds_id: received.append(ds_id))

    widget.populate(_sample_entries())
    qt_app.processEvents()

    assert widget.selected_dataset_id() == "ds2"
    assert len(received) == 0
    widget.close()


# ------------------------------------------------------------------
# Regression: select_dataset dedup guard
# ------------------------------------------------------------------


def test_select_dataset_no_emit_when_already_active(qt_app):
    widget = UnifiedDatasetList()
    widget.populate(_sample_entries())

    # First select_dataset("ds1") — sets active and emits.
    widget.select_dataset("ds1")
    qt_app.processEvents()
    assert widget.selected_dataset_id() == "ds1"

    received = []
    widget.currentDatasetChanged.connect(lambda ds_id: received.append(ds_id))

    # Second select_dataset("ds1") — should NOT emit.
    widget.select_dataset("ds1")
    qt_app.processEvents()

    assert len(received) == 0
    assert widget.selected_dataset_id() == "ds1"
    widget.close()


# ------------------------------------------------------------------
# Regression: _on_unified_dataset_selected dedup guard
# ------------------------------------------------------------------


def test_detail_panels_not_reloaded_on_duplicate_dataset_signal(qt_app):
    window = _make_fitting_window()
    try:
        window.show()
        qt_app.processEvents()

        dt_tab = window._data_targets_tab
        unified_list = dt_tab.unified_list

        # Select ds1 (already selected from populate, but force it).
        unified_list.select_dataset("ds1")
        qt_app.processEvents()

        # Patch the three panel methods to track calls.
        with mock.patch.object(dt_tab.data_tab, "select_dataset", wraps=dt_tab.data_tab.select_dataset) as m_data, \
             mock.patch.object(dt_tab.species_table, "load_for_dataset",
                               wraps=dt_tab.species_table.load_for_dataset) as m_species:

            # Directly emit the signal with the already-active dataset id.
            unified_list.currentDatasetChanged.emit("ds1")
            qt_app.processEvents()

            assert m_data.call_count == 0, f"data_tab.select_dataset called {m_data.call_count} times on duplicate"
            assert m_species.call_count == 0, f"species_table.load_for_dataset called {m_species.call_count} times on duplicate"
    finally:
        window.close()


# ------------------------------------------------------------------
# Regression: sampling validity refresh on targets apply
# ------------------------------------------------------------------


def test_sampling_validity_refreshed_on_targets_apply(qt_app):
    window = _make_fitting_window()
    try:
        window.show()
        qt_app.processEvents()

        # Ensure mechanism species includes "A" so x_name="A" is modeled.
        window._params_ics_tab._mechanism_species = ["A", "B"]

        # Apply sampling with x_name = "A" for ds1.
        full_t = window._species_table.full_t_by_dataset.get("ds1", np.linspace(0.0, 1.0, 5))
        t_min = float(full_t[0]) if full_t.size else 0.0
        t_max = float(full_t[-1]) if full_t.size else 1.0
        window._sampling_applied["ds1"] = {
            "t_min": t_min,
            "t_max": t_max,
            "n_points": 0,
            "x_name": "A",
            "x_mapping_mode": "auto",
        }

        # Apply fit targets that include "A" for ds1.
        window._species_table._fit_targets_selection_applied["ds1"] = ["A"]

        # Trigger _on_targets_applied — this should refresh sampling validity.
        window._on_targets_applied()
        qt_app.processEvents()

        # After fix: sampling validity detects x_name="A" conflicts with fit target "A".
        assert not window._run_button.isEnabled(), (
            "Run button should be disabled when x_name conflicts with a fit target"
        )
    finally:
        window.close()


# ------------------------------------------------------------------
# Validation foreground delegate
# ------------------------------------------------------------------


def test_dataset_list_has_validation_delegate(qt_app):
    """UnifiedDatasetList must install _ValidationForegroundDelegate on its QListWidget."""
    from kindred.gui.fitting.unified_dataset_list import _ValidationForegroundDelegate

    widget = UnifiedDatasetList()
    try:
        assert isinstance(widget._list.itemDelegate(), _ValidationForegroundDelegate)
    finally:
        widget.close()


def test_dataset_list_delegate_reads_foreground_role(qt_app):
    """After set_validation_state, the item's ForegroundRole stores the correct brush."""
    widget = UnifiedDatasetList()
    widget.populate(_sample_entries())

    widget.set_validation_state("ds1", "invalid_applied")
    item = widget._list.item(0)
    assert item is not None
    fg = item.data(QtCore.Qt.ForegroundRole)
    assert isinstance(fg, QtGui.QBrush)
    assert fg.color() == QtGui.QColor(80, 0, 0)
    widget.close()
