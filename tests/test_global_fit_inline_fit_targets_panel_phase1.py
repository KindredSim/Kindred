import numpy as np
import pytest


pytestmark = [pytest.mark.gui]

_DATASET_INVALID_MARK_ROLE = 0x10FF  # Qt.UserRole + offset, internal-only for tests


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


def _set_fit_targets_dataset(panel, *, dataset_id: str) -> None:
    from PySide6 import QtCore, QtWidgets
    dataset_list = panel.window().findChild(QtWidgets.QListWidget, "global_fit_fit_targets_dataset_list")
    assert dataset_list is not None
    for i in range(dataset_list.count()):
        item = dataset_list.item(i)
        if item is not None and str(item.data(QtCore.Qt.UserRole) or "") == str(dataset_id):
            dataset_list.setCurrentRow(i)
            return
    raise AssertionError(f"Dataset id not in list: {dataset_id!r}")


def _dataset_table_row_for(window, dataset_id: str) -> int:
    from PySide6 import QtCore
    table = window._dataset_table
    for row in range(table.rowCount()):
        item = table.item(row, 0)
        if item is None:
            continue
        if str(item.data(QtCore.Qt.UserRole) or "") == str(dataset_id):
            return row
    raise AssertionError(f"Dataset row not found: {dataset_id!r}")


def _target_weight_edit(panel, *, target_name: str):
    from PySide6 import QtWidgets

    for edit in reversed(panel.findChildren(QtWidgets.QLineEdit)):
        if str(edit.property("fitTargetName") or "") == str(target_name):
            return edit
    raise AssertionError(f"Target weight edit not found: {target_name!r}")


def test_global_fit_window_shows_inline_fit_targets_panel(qt_app):
    from PySide6 import QtWidgets

    window = _make_window(selected_species=["A"])
    try:
        panel = window.findChild(QtWidgets.QGroupBox, "global_fit_fit_targets_panel")
        assert panel is not None
    finally:
        window.close()
        qt_app.processEvents()


def test_fit_targets_apply_required_to_update_payload(qt_app, monkeypatch):
    from PySide6 import QtCore, QtWidgets

    captured = {"datasets": [], "starts": 0}

    class _FakeWorker(QtCore.QObject):
        progress = QtCore.Signal(int, str)
        finished = QtCore.Signal(dict)
        error = QtCore.Signal(str)

        def __init__(self, datasets, shared_params, **_kwargs):
            super().__init__()
            captured["datasets"].append([dict(ds) for ds in datasets])

        def start(self):
            captured["starts"] += 1

        def isRunning(self):
            return False

    monkeypatch.setattr("kindred.gui.fitting.window.GlobalFitWorker", _FakeWorker)

    window = _make_window(selected_species=["A"])
    try:
        panel = window.findChild(QtWidgets.QGroupBox, "global_fit_fit_targets_panel")
        assert panel is not None
        apply_btn = panel.findChild(QtWidgets.QPushButton, "global_fit_fit_targets_apply")
        assert apply_btn is not None

        assert window._global_payload_lookup["ds1"]["species"] == ["A"]

        # Toggle "B" in the pending selection; applied payload must remain unchanged until Apply.
        checkbox_b = next(
            cb for cb in panel.findChildren(QtWidgets.QCheckBox) if cb.text().strip() == "B"
        )
        checkbox_b.setChecked(True)
        qt_app.processEvents()
        assert window._global_payload_lookup["ds1"]["species"] == ["A"]

        config = window._collect_parameter_config()
        assert config is not None
        selection = window._collect_dataset_selection()
        window._start_global_fit(config, selection)
        assert captured["starts"] == 1
        assert captured["datasets"][-1][0]["species"] == ["A"]

        # Apply pending changes; payload should update, but must not auto-run.
        apply_btn.click()
        qt_app.processEvents()
        assert captured["starts"] == 1
        assert set(window._global_payload_lookup["ds1"]["species"]) == {"A", "B"}

        window._start_global_fit(config, selection)
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
        finished = QtCore.Signal(dict)
        error = QtCore.Signal(str)

        def __init__(self, datasets, shared_params, **_kwargs):
            super().__init__()
            captured["datasets"].append([dict(ds) for ds in datasets])

        def start(self):
            captured["starts"] += 1

        def isRunning(self):
            return False

    monkeypatch.setattr("kindred.gui.fitting.window.GlobalFitWorker", _FakeWorker)

    window = _make_window(selected_species=["A"])
    try:
        panel = window.findChild(QtWidgets.QGroupBox, "global_fit_fit_targets_panel")
        assert panel is not None
        apply_btn = panel.findChild(QtWidgets.QPushButton, "global_fit_fit_targets_apply")
        assert apply_btn is not None

        assert window._global_payload_lookup["ds1"]["target_weights"] == {"A": 1.0}

        edit_a = _target_weight_edit(panel, target_name="A")
        edit_b = _target_weight_edit(panel, target_name="B")
        edit_a.setText("2.5")
        edit_b.setText("9.0")
        qt_app.processEvents()

        assert window._global_payload_lookup["ds1"]["target_weights"] == {"A": 1.0}

        config = window._collect_parameter_config()
        assert config is not None
        selection = window._collect_dataset_selection()
        window._start_global_fit(config, selection)
        assert captured["starts"] == 1
        assert captured["datasets"][-1][0]["target_weights"] == {"A": 1.0}

        apply_btn.click()
        qt_app.processEvents()
        assert captured["starts"] == 1
        assert window._global_payload_lookup["ds1"]["target_weights"] == {"A": 2.5}

        window._start_global_fit(config, selection)
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
        assert window._fit_target_weights_applied["ds1"] == {"A": 2.5}
        assert window._fit_target_weights_pending["ds1"]["A"] == 2.5
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
        assert window._fit_target_weights_applied["ds1"] == {"A": 3.5}
        assert window._fit_target_weights_pending["ds1"]["A"] == 3.5
        assert window._dataset_entries[0]["target_weights"] == {"A": 3.5}
        assert window._global_payload_lookup["ds1"]["target_weights"] == {"A": 3.5}
    finally:
        window.close()
        qt_app.processEvents()


def test_fit_targets_apply_keeps_invalid_used_dataset_pending_and_highlights_row(qt_app, monkeypatch):
    from PySide6 import QtGui, QtWidgets
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *_a, **_k: QtWidgets.QMessageBox.StandardButton.Ok)

    window = _make_two_dataset_window(ds1_selected=["A"], ds2_selected=["C"])
    try:
        panel = window.findChild(QtWidgets.QGroupBox, "global_fit_fit_targets_panel")
        assert panel is not None
        apply_btn = panel.findChild(QtWidgets.QPushButton, "global_fit_fit_targets_apply")
        assert apply_btn is not None

        applied_before = list(window._fit_targets_selection_applied["ds1"])

        _set_fit_targets_dataset(panel, dataset_id="ds1")
        for cb in [cb for cb in panel.findChildren(QtWidgets.QCheckBox) if cb.text().strip() in {"A", "B"}]:
            cb.setChecked(False)
        qt_app.processEvents()
        assert window._fit_targets_selection_pending["ds1"] == set()

        apply_btn.click()
        qt_app.processEvents()

        # Used dataset with empty pending selection must not be applied.
        assert window._fit_targets_selection_applied["ds1"] == applied_before
        assert window._fit_targets_selection_pending["ds1"] == set()
        assert apply_btn.isEnabled(), "Apply should remain enabled while invalid pending changes exist."

        error_label = panel.findChild(QtWidgets.QLabel, "global_fit_fit_targets_error")
        assert error_label is not None
        assert "ds1" in error_label.text()
        assert "no fit targets" in error_label.text().lower()

        row = _dataset_table_row_for(window, "ds1")
        mark_item = window._dataset_table.item(row, 1)
        assert mark_item is not None
        bg = mark_item.background()
        assert bg is not None and bg.color().isValid()
        assert bg.color() != QtGui.QColor(), "Expected a non-default highlight brush for invalid dataset row."
        assert bool(mark_item.data(_DATASET_INVALID_MARK_ROLE)), "Expected invalid-row mark role set for tests."
    finally:
        window.close()
        qt_app.processEvents()


def test_run_fit_disabled_when_used_dataset_has_zero_applied_targets(qt_app, monkeypatch):
    from PySide6 import QtCore, QtWidgets
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *_a, **_k: QtWidgets.QMessageBox.StandardButton.Ok)

    window = _make_two_dataset_window(ds1_selected=["A"], ds2_selected=["C"])
    try:
        panel = window.findChild(QtWidgets.QGroupBox, "global_fit_fit_targets_panel")
        assert panel is not None
        apply_btn = panel.findChild(QtWidgets.QPushButton, "global_fit_fit_targets_apply")
        assert apply_btn is not None

        # Make ds1 pending empty and uncheck Use for ds1 so Apply is allowed to commit empty selection.
        _set_fit_targets_dataset(panel, dataset_id="ds1")
        for cb in [cb for cb in panel.findChildren(QtWidgets.QCheckBox) if cb.text().strip() in {"A", "B"}]:
            cb.setChecked(False)
        qt_app.processEvents()

        row = _dataset_table_row_for(window, "ds1")
        use_item = window._dataset_table.item(row, 0)
        assert use_item is not None
        use_item.setCheckState(QtCore.Qt.Unchecked)
        qt_app.processEvents()

        apply_btn.click()
        qt_app.processEvents()
        assert window._fit_targets_selection_applied["ds1"] == []

        # Re-enable Use; Run Fit must disable immediately due to invalid applied targets.
        row = _dataset_table_row_for(window, "ds1")
        use_item = window._dataset_table.item(row, 0)
        assert use_item is not None
        use_item.setCheckState(QtCore.Qt.Checked)
        qt_app.processEvents()

        assert not window._run_button.isEnabled()
        status = panel.findChild(QtWidgets.QLabel, "global_fit_fit_targets_run_blocked")
        assert status is not None
        assert not status.isHidden()
        assert "ds1" in status.text()

        # Fix: select one series and Apply -> Run Fit enabled.
        _set_fit_targets_dataset(panel, dataset_id="ds1")
        checkbox_a = next(cb for cb in panel.findChildren(QtWidgets.QCheckBox) if cb.text().strip() == "A")
        checkbox_a.setChecked(True)
        qt_app.processEvents()
        apply_btn.click()
        qt_app.processEvents()

        assert window._fit_targets_selection_applied["ds1"] == ["A"]
        assert window._run_button.isEnabled()
        assert status.isHidden()
    finally:
        window.close()
        qt_app.processEvents()


def test_excluded_dataset_invalid_pending_target_weight_is_non_actionable_until_reincluded(qt_app, monkeypatch):
    from PySide6 import QtCore, QtWidgets

    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *_a, **_k: QtWidgets.QMessageBox.StandardButton.Ok)

    window = _make_two_dataset_window(ds1_selected=["A"], ds2_selected=["C"])
    try:
        panel = window.findChild(QtWidgets.QGroupBox, "global_fit_fit_targets_panel")
        assert panel is not None
        apply_btn = panel.findChild(QtWidgets.QPushButton, "global_fit_fit_targets_apply")
        error_label = panel.findChild(QtWidgets.QLabel, "global_fit_fit_targets_error")
        assert apply_btn is not None
        assert error_label is not None

        _set_fit_targets_dataset(panel, dataset_id="ds1")
        edit_a = _target_weight_edit(panel, target_name="A")
        edit_a.setText("0")
        qt_app.processEvents()

        assert "ds1" in window._invalid_pending_target_weight_dataset_ids()
        assert "invalid target weights" in error_label.text().lower()

        row = _dataset_table_row_for(window, "ds1")
        use_item = window._dataset_table.item(row, 0)
        mark_item = window._dataset_table.item(row, 1)
        assert use_item is not None
        assert mark_item is not None

        use_item.setCheckState(QtCore.Qt.Unchecked)
        qt_app.processEvents()

        assert "ds1" not in window._invalid_pending_target_weight_dataset_ids()
        assert "invalid target weights" not in error_label.text().lower()
        assert not bool(mark_item.data(_DATASET_INVALID_MARK_ROLE))
        assert window._run_button.isEnabled()

        apply_btn.click()
        qt_app.processEvents()

        assert window._status_label.text() == "Fit targets applied"
        assert window._fit_target_weights_pending_invalid["ds1"] == {"A": "0"}
        assert _target_weight_edit(panel, target_name="A").text() == "0"

        row = _dataset_table_row_for(window, "ds1")
        use_item = window._dataset_table.item(row, 0)
        mark_item = window._dataset_table.item(row, 1)
        assert use_item is not None
        assert mark_item is not None
        use_item.setCheckState(QtCore.Qt.Checked)
        qt_app.processEvents()

        assert "ds1" in window._invalid_pending_target_weight_dataset_ids()
        assert "invalid target weights" in error_label.text().lower()
        assert bool(mark_item.data(_DATASET_INVALID_MARK_ROLE))
        assert window._run_button.isEnabled()
    finally:
        window.close()
        qt_app.processEvents()
