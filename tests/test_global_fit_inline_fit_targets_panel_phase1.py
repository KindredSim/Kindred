import numpy as np
import pytest


pytestmark = [pytest.mark.gui]

def _unified_list_item_for(window, dataset_id: str):
    """Return the QListWidgetItem in the unified list matching *dataset_id*."""
    from PySide6 import QtCore
    ulist = window._data_targets_tab.unified_list._list
    for i in range(ulist.count()):
        item = ulist.item(i)
        if item is not None and str(item.data(QtCore.Qt.UserRole) or "") == str(dataset_id):
            return item
    raise AssertionError(f"Unified list item not found: {dataset_id!r}")


def _make_window(
    *,
    selected_species: list[str],
    entry_target_weights: dict[str, float] | None = None,
    payload_target_weights: dict[str, float] | None = None,
):
    from kindred.gui.fitting.window import FittingWindow

    t = np.linspace(0.0, 1.0, 5)
    y_a = np.linspace(1.0, 0.5, t.size)
    y_b = np.linspace(0.2, 0.9, t.size)
    dataset_entries = [
        {
            "id": "ds1",
            "label": "ds1",
            "t": t.copy(),
            # Provide full observed series; selected_species is the applied fit-target snapshot.
            "species_data": {"A": y_a.copy(), "B": y_b.copy()},
            "selected_species": list(selected_species),
            "weight": 1.0,
            "include": True,
        }
    ]
    if entry_target_weights is not None:
        dataset_entries[0]["target_weights"] = dict(entry_target_weights)
    selected_rows = [np.asarray(dataset_entries[0]["species_data"][name]) for name in selected_species]
    dataset_payloads = [
        {
            "id": "ds1",
            "t": t.copy(),
            "y": np.vstack(selected_rows),
            "species": list(selected_species),
        }
    ]
    if payload_target_weights is not None:
        dataset_payloads[0]["target_weights"] = dict(payload_target_weights)
    return FittingWindow(
        mode="global",
        parameter_defs=[{"name": "k1", "value": 0.2, "min": 0.01, "max": 1.0}],
        dataset_entries=dataset_entries,
        simulation_func=lambda _params: {"t": t.copy(), "species": {"A": y_a.copy(), "B": y_b.copy()}},
        dataset_payloads=dataset_payloads,
        dataset_weights={"ds1": 1.0},
    )

def _make_two_dataset_window(*, ds1_selected: list[str], ds2_selected: list[str]):
    from kindred.gui.fitting.window import FittingWindow

    t = np.linspace(0.0, 1.0, 5)
    y_a = np.linspace(1.0, 0.5, t.size)
    y_b = np.linspace(0.2, 0.9, t.size)
    y_c = np.linspace(0.8, 0.1, t.size)
    y_d = np.linspace(0.0, 0.4, t.size)

    ds1_series = {"A": y_a.copy(), "B": y_b.copy()}
    ds2_series = {"C": y_c.copy(), "D": y_d.copy()}

    dataset_entries = [
        {
            "id": "ds1",
            "label": "ds1",
            "t": t.copy(),
            "species_data": dict(ds1_series),
            "selected_species": list(ds1_selected),
            "weight": 1.0,
            "include": True,
        },
        {
            "id": "ds2",
            "label": "ds2",
            "t": t.copy(),
            "species_data": dict(ds2_series),
            "selected_species": list(ds2_selected),
            "weight": 1.0,
            "include": True,
        },
    ]

    dataset_payloads = []
    for dataset_id, series_map, selection in (
        ("ds1", ds1_series, ds1_selected),
        ("ds2", ds2_series, ds2_selected),
    ):
        rows = [np.asarray(series_map[name]) for name in selection]
        dataset_payloads.append({"id": dataset_id, "t": t.copy(), "y": np.vstack(rows), "species": list(selection)})

    return FittingWindow(
        mode="global",
        parameter_defs=[{"name": "k1", "value": 0.2, "min": 0.01, "max": 1.0}],
        dataset_entries=dataset_entries,
        simulation_func=lambda _params: {"t": t.copy(), "species": {**ds1_series, **ds2_series}},
        dataset_payloads=dataset_payloads,
        dataset_weights={"ds1": 1.0, "ds2": 1.0},
    )


def _set_fit_targets_dataset(widget, *, dataset_id: str) -> None:
    from PySide6 import QtCore
    window = widget if hasattr(widget, '_data_targets_tab') else widget.window()
    ulist = window._data_targets_tab.unified_list._list
    for i in range(ulist.count()):
        item = ulist.item(i)
        if item is not None and str(item.data(QtCore.Qt.UserRole) or "") == str(dataset_id):
            ulist.setCurrentRow(i)
            return
    raise AssertionError(f"Dataset id not in list: {dataset_id!r}")


def _set_unified_list_include(window, dataset_id: str, checked: bool) -> None:
    from PySide6 import QtCore
    ulist = window._data_targets_tab.unified_list._list
    for row in range(ulist.count()):
        item = ulist.item(row)
        if item is not None and str(item.data(QtCore.Qt.UserRole) or "") == str(dataset_id):
            item.setCheckState(QtCore.Qt.Checked if checked else QtCore.Qt.Unchecked)
            return
    raise AssertionError(f"Unified list item not found: {dataset_id!r}")


def _target_weight_edit(widget, *, target_name: str):
    from kindred.gui.fitting.unified_species_table import _Col
    window = widget if hasattr(widget, '_species_table') else widget.window()
    table = window._species_table._table
    for row in range(table.rowCount()):
        species_item = table.item(row, _Col.SPECIES)
        if species_item is not None and species_item.text() == target_name:
            return table.item(row, _Col.WEIGHT)
    raise AssertionError(f"Target weight item not found: {target_name!r}")


def test_global_fit_window_shows_inline_fit_targets_panel(qt_app):
    from PySide6 import QtWidgets

    window = _make_window(selected_species=["A"])
    try:
        panel = window.findChild(QtWidgets.QGroupBox, "global_fit_unified_species_group")
        assert panel is not None
    finally:
        window.close()
        qt_app.processEvents()


def test_fit_targets_apply_required_to_update_payload(qt_app, monkeypatch):
    from PySide6 import QtCore, QtWidgets

    captured = {"datasets": [], "starts": 0}

    class _FakeWorker(QtCore.QObject):
        progress = QtCore.Signal(int, str)
        bestUpdated = QtCore.Signal(dict)
        finished = QtCore.Signal(dict)
        error = QtCore.Signal(str)

        def __init__(self, datasets, shared_params, **_kwargs):
            super().__init__()
            captured["datasets"].append([dict(ds) for ds in datasets])

        def start(self):
            captured["starts"] += 1

        def isRunning(self):
            return False

    monkeypatch.setattr("kindred.gui.fitting.worker_launch.GlobalFitWorker", _FakeWorker)

    window = _make_window(selected_species=["A"])
    try:
        panel = window.findChild(QtWidgets.QGroupBox, "global_fit_unified_species_group")
        assert panel is not None
        apply_btn = panel.findChild(QtWidgets.QPushButton, "global_fit_species_table_apply")
        assert apply_btn is not None

        assert window._global_payload_lookup["ds1"]["species"] == ["A"]

        # Toggle "B" in the pending selection; applied payload must remain unchanged until Apply.
        from kindred.gui.fitting.unified_species_table import _Col
        table = window._species_table._table
        for row in range(table.rowCount()):
            if table.item(row, _Col.SPECIES).text() == "B":
                table.item(row, _Col.INCLUDE).setCheckState(QtCore.Qt.Checked)
                break
        qt_app.processEvents()
        assert window._global_payload_lookup["ds1"]["species"] == ["A"]

        config = window._params_ics_tab.collect_parameter_config()
        assert config is not None
        window.run_fit()
        assert captured["starts"] == 1
        assert captured["datasets"][-1][0]["species"] == ["A"]

        # Apply pending changes; payload should update, but must not auto-run.
        apply_btn.click()
        qt_app.processEvents()
        assert captured["starts"] == 1
        assert set(window._global_payload_lookup["ds1"]["species"]) == {"A", "B"}

        window.run_fit()
        assert captured["starts"] == 2
        assert set(captured["datasets"][-1][0]["species"]) == {"A", "B"}
    finally:
        window.close()
        qt_app.processEvents()


def test_fit_target_weights_apply_required_to_update_payload(qt_app, monkeypatch):
    from PySide6 import QtCore, QtWidgets

    captured = {"datasets": [], "starts": 0}

    class _FakeWorker(QtCore.QObject):
        progress = QtCore.Signal(int, str)
        bestUpdated = QtCore.Signal(dict)
        finished = QtCore.Signal(dict)
        error = QtCore.Signal(str)

        def __init__(self, datasets, shared_params, **_kwargs):
            super().__init__()
            captured["datasets"].append([dict(ds) for ds in datasets])

        def start(self):
            captured["starts"] += 1

        def isRunning(self):
            return False

    monkeypatch.setattr("kindred.gui.fitting.worker_launch.GlobalFitWorker", _FakeWorker)

    window = _make_window(selected_species=["A"])
    try:
        panel = window.findChild(QtWidgets.QGroupBox, "global_fit_unified_species_group")
        assert panel is not None
        apply_btn = panel.findChild(QtWidgets.QPushButton, "global_fit_species_table_apply")
        assert apply_btn is not None

        assert window._global_payload_lookup["ds1"]["target_weights"] == {"A": 1.0}

        edit_a = _target_weight_edit(panel, target_name="A")
        edit_b = _target_weight_edit(panel, target_name="B")
        edit_a.setText("2.5")
        edit_b.setText("9.0")
        qt_app.processEvents()

        assert window._global_payload_lookup["ds1"]["target_weights"] == {"A": 1.0}

        config = window._params_ics_tab.collect_parameter_config()
        assert config is not None
        window.run_fit()
        assert captured["starts"] == 1
        assert captured["datasets"][-1][0]["target_weights"] == {"A": 1.0}

        apply_btn.click()
        qt_app.processEvents()
        assert captured["starts"] == 1
        assert window._global_payload_lookup["ds1"]["target_weights"] == {"A": 2.5}

        window.run_fit()
        assert captured["starts"] == 2
        assert captured["datasets"][-1][0]["target_weights"] == {"A": 2.5}
    finally:
        window.close()
        qt_app.processEvents()


def test_preloaded_target_weights_seed_applied_state_from_dataset_entry_before_payload(qt_app):
    window = _make_window(
        selected_species=["A"],
        entry_target_weights={"A": 2.5},
        payload_target_weights={"A": 9.0},
    )
    try:
        assert window._species_table._fit_target_weights_applied["ds1"] == {"A": 2.5}
        assert window._species_table._fit_target_weights_pending["ds1"]["A"] == 2.5
        assert window._dataset_entries[0]["target_weights"] == {"A": 2.5}
        assert window._global_payload_lookup["ds1"]["target_weights"] == {"A": 2.5}
    finally:
        window.close()
        qt_app.processEvents()


def test_preloaded_target_weights_fall_back_to_seeded_payload_when_entry_omits_them(qt_app):
    window = _make_window(
        selected_species=["A"],
        payload_target_weights={"A": 3.5},
    )
    try:
        assert window._species_table._fit_target_weights_applied["ds1"] == {"A": 3.5}
        assert window._species_table._fit_target_weights_pending["ds1"]["A"] == 3.5
        assert window._dataset_entries[0]["target_weights"] == {"A": 3.5}
        assert window._global_payload_lookup["ds1"]["target_weights"] == {"A": 3.5}
    finally:
        window.close()
        qt_app.processEvents()


def test_fit_targets_apply_promotes_empty_selection_and_highlights_row(qt_app, monkeypatch):
    from PySide6 import QtCore, QtGui, QtWidgets
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *_a, **_k: QtWidgets.QMessageBox.StandardButton.Ok)

    window = _make_two_dataset_window(ds1_selected=["A"], ds2_selected=["C"])
    try:
        panel = window.findChild(QtWidgets.QGroupBox, "global_fit_unified_species_group")
        assert panel is not None
        apply_btn = panel.findChild(QtWidgets.QPushButton, "global_fit_species_table_apply")
        assert apply_btn is not None

        _set_fit_targets_dataset(window, dataset_id="ds1")
        from kindred.gui.fitting.unified_species_table import _Col
        table = window._species_table._table
        for row in range(table.rowCount()):
            name = table.item(row, _Col.SPECIES).text()
            include_item = table.item(row, _Col.INCLUDE)
            if name in {"A", "B"} and include_item is not None:
                include_item.setCheckState(QtCore.Qt.Unchecked)
        qt_app.processEvents()
        assert window._species_table._fit_targets_selection_pending["ds1"] == set()

        apply_btn.click()
        qt_app.processEvents()

        # Empty pending selection is promoted truthfully into applied state.
        assert window._species_table.fit_targets_selection_applied["ds1"] == []
        assert window._species_table._fit_targets_selection_pending["ds1"] == set()
        # Pending now matches applied (both empty), so no unsaved changes remain.
        assert not apply_btn.isEnabled(), "Apply should be disabled when pending matches applied."

        error_label = panel.findChild(QtWidgets.QLabel, "global_fit_species_table_error")
        assert error_label is not None
        assert "ds1" in error_label.text()
        assert "no fit targets" in error_label.text().lower()

        unified_item = _unified_list_item_for(window, "ds1")
        bg = unified_item.background()
        assert bg is not None and bg.color().isValid()
        assert bg.color() != QtGui.QColor(), "Expected a non-default highlight brush for invalid dataset row."
    finally:
        window.close()
        qt_app.processEvents()


def test_run_fit_disabled_when_used_dataset_has_zero_applied_targets(qt_app, monkeypatch):
    from PySide6 import QtCore, QtWidgets
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *_a, **_k: QtWidgets.QMessageBox.StandardButton.Ok)

    window = _make_two_dataset_window(ds1_selected=["A"], ds2_selected=["C"])
    try:
        panel = window.findChild(QtWidgets.QGroupBox, "global_fit_unified_species_group")
        assert panel is not None
        apply_btn = panel.findChild(QtWidgets.QPushButton, "global_fit_species_table_apply")
        assert apply_btn is not None

        # Make ds1 pending empty and uncheck Use for ds1 so Apply is allowed to commit empty selection.
        _set_fit_targets_dataset(window, dataset_id="ds1")
        from kindred.gui.fitting.unified_species_table import _Col
        table = window._species_table._table
        for row in range(table.rowCount()):
            name = table.item(row, _Col.SPECIES).text()
            include_item = table.item(row, _Col.INCLUDE)
            if name in {"A", "B"} and include_item is not None:
                include_item.setCheckState(QtCore.Qt.Unchecked)
        qt_app.processEvents()

        _set_unified_list_include(window, "ds1", False)
        qt_app.processEvents()

        apply_btn.click()
        qt_app.processEvents()
        assert window._species_table.fit_targets_selection_applied["ds1"] == []

        # Re-enable Use; Run Fit must disable immediately due to invalid applied targets.
        _set_unified_list_include(window, "ds1", True)
        qt_app.processEvents()

        assert not window._run_button.isEnabled()
        status = panel.findChild(QtWidgets.QLabel, "global_fit_species_table_run_blocked")
        assert status is not None
        assert not status.isHidden()
        assert "ds1" in status.text()

        # Fix: select one series and Apply -> Run Fit enabled.
        _set_fit_targets_dataset(window, dataset_id="ds1")
        for row in range(table.rowCount()):
            if table.item(row, _Col.SPECIES).text() == "A":
                table.item(row, _Col.INCLUDE).setCheckState(QtCore.Qt.Checked)
                break
        qt_app.processEvents()
        apply_btn.click()
        qt_app.processEvents()

        assert window._species_table.fit_targets_selection_applied["ds1"] == ["A"]
        assert window._run_button.isEnabled()
        assert status.isHidden()
    finally:
        window.close()
        qt_app.processEvents()


def test_excluded_dataset_invalid_pending_target_weight_is_non_actionable_until_reincluded(qt_app, monkeypatch):
    from PySide6 import QtGui, QtWidgets

    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *_a, **_k: QtWidgets.QMessageBox.StandardButton.Ok)

    window = _make_two_dataset_window(ds1_selected=["A"], ds2_selected=["C"])
    try:
        panel = window.findChild(QtWidgets.QGroupBox, "global_fit_unified_species_group")
        assert panel is not None
        apply_btn = panel.findChild(QtWidgets.QPushButton, "global_fit_species_table_apply")
        error_label = panel.findChild(QtWidgets.QLabel, "global_fit_species_table_error")
        assert apply_btn is not None
        assert error_label is not None

        _set_fit_targets_dataset(panel, dataset_id="ds1")
        edit_a = _target_weight_edit(panel, target_name="A")
        edit_a.setText("0")
        qt_app.processEvents()

        assert "ds1" in window._species_table.invalid_pending_target_weight_dataset_ids()
        assert "invalid target weights" in error_label.text().lower()

        unified_item = _unified_list_item_for(window, "ds1")

        _set_unified_list_include(window, "ds1", False)
        qt_app.processEvents()

        assert "ds1" not in window._species_table.invalid_pending_target_weight_dataset_ids()
        assert "invalid target weights" not in error_label.text().lower()
        assert unified_item.background() == QtGui.QBrush()
        assert window._run_button.isEnabled()

        apply_btn.click()
        qt_app.processEvents()

        assert window._status_label.text() in {"Fit targets applied", "Fitting runtime ready"}
        assert window._species_table._fit_target_weights_pending_invalid["ds1"] == {"A": "0"}
        assert _target_weight_edit(panel, target_name="A").text() == "0"

        _set_unified_list_include(window, "ds1", True)
        qt_app.processEvents()

        assert "ds1" in window._species_table.invalid_pending_target_weight_dataset_ids()
        assert "invalid target weights" in error_label.text().lower()
        unified_item = _unified_list_item_for(window, "ds1")
        assert unified_item.background().color() != QtGui.QColor()
        assert window._run_button.isEnabled()
    finally:
        window.close()
        qt_app.processEvents()
