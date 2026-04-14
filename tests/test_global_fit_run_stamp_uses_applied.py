from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest


pytestmark = [pytest.mark.gui]


def _make_window(*, selected_species: list[str]):
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
            "selected_species": list(selected_species),
            "weight": 1.0,
            "include": True,
        }
    ]
    selected_rows = [np.asarray(dataset_entries[0]["species_data"][name]) for name in selected_species]
    dataset_payloads = [
        {
            "id": "ds1",
            "t": t.copy(),
            "y": np.vstack(selected_rows),
            "species": list(selected_species),
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


def _toggle_fit_targets_pending(window) -> None:
    from PySide6.QtCore import Qt
    from kindred.gui.fitting.unified_species_table import _Col
    table = window._species_table._table
    for row in range(table.rowCount()):
        species_item = table.item(row, _Col.SPECIES)
        include_item = table.item(row, _Col.INCLUDE)
        if species_item is None or include_item is None:
            continue
        name = species_item.text()
        if name == "A":
            include_item.setCheckState(Qt.Unchecked)
        elif name == "B":
            include_item.setCheckState(Qt.Checked)


def _target_weight_edit(widget, *, target_name: str):
    from kindred.gui.fitting.unified_species_table import _Col
    window = widget if hasattr(widget, '_species_table') else widget.window()
    table = window._species_table._table
    for row in range(table.rowCount()):
        species_item = table.item(row, _Col.SPECIES)
        if species_item is not None and species_item.text() == target_name:
            return table.item(row, _Col.WEIGHT)
    raise AssertionError(f"Target weight item not found: {target_name!r}")


def test_run_stamp_uses_applied_fit_targets_not_pending(qt_app, monkeypatch):
    from PySide6 import QtCore, QtWidgets
    from kindred import __version__ as kindred_version

    captured = {"payloads": []}

    class _FakeWorker(QtCore.QObject):
        progress = QtCore.Signal(int, str)
        finished = QtCore.Signal(dict)
        error = QtCore.Signal(str)

        def __init__(self, *args, **kwargs):
            super().__init__()
            captured["payloads"].append({"args": args, "kwargs": dict(kwargs)})

        def start(self):
            return

        def isRunning(self):
            return False

    monkeypatch.setattr("kindred.gui.fitting.window.GlobalFitWorker", _FakeWorker)

    window = _make_window(selected_species=["A"])
    try:
        panel = window.findChild(QtWidgets.QGroupBox, "global_fit_unified_species_group")
        assert panel is not None

        _toggle_fit_targets_pending(window)
        qt_app.processEvents()
        assert window._species_table.fit_targets_selection_applied["ds1"] == ["A"]
        assert window._species_table._fit_targets_selection_pending["ds1"] == {"B"}

        config = window._params_ics_tab._collect_parameter_config()
        assert config is not None
        selection = window._collect_dataset_selection()
        window._start_global_fit(config, selection)

        rrt = window._run_results_tab
        assert hasattr(rrt, "_last_run_stamp")
        assert rrt._last_run_stamp["fit_targets_applied"]["ds1"] == ["A"]
        assert rrt._last_run_stamp.get("prepared_simulation") is not None
        assert rrt._last_run_stamp["prepared_simulation"]["solver_requested"] == "BDF"
        assert rrt._last_run_stamp["prepared_simulation"]["solver_normalized"] == "BDF"
        assert "solver" not in rrt._last_run_stamp["prepared_simulation"]
        assert rrt._last_run_stamp.get("kindred_version") == kindred_version
        assert rrt._last_run_stamp.get("dataset_params") is not None
        assert rrt._last_run_stamp["dataset_params"]["ds1"]["init:A"] == "1"
        assert rrt._last_run_stamp.get("dataset_variable_params") is not None
        assert rrt._last_run_stamp["dataset_variable_params"]["ds1"]["init:B"]["max"] == "10"
        assert rrt._last_run_stamp.get("shared_params") is not None
        assert rrt._last_run_stamp["shared_params"]["fit_initial"]["k1"] == "0.2"
        json.dumps(rrt._last_run_stamp, sort_keys=True)
        assert rrt._last_run_stamp_hash

        assert rrt._last_run_stamp_short
        assert rrt._last_run_stamp_hash
    finally:
        window.close()
        qt_app.processEvents()


def test_run_stamp_uses_applied_target_weights_not_pending(qt_app, monkeypatch):
    from PySide6 import QtCore, QtWidgets

    class _FakeWorker(QtCore.QObject):
        progress = QtCore.Signal(int, str)
        finished = QtCore.Signal(dict)
        error = QtCore.Signal(str)

        def __init__(self, *args, **kwargs):
            super().__init__()

        def start(self):
            return

        def isRunning(self):
            return False

    monkeypatch.setattr("kindred.gui.fitting.window.GlobalFitWorker", _FakeWorker)

    window = _make_window(selected_species=["A"])
    try:
        panel = window.findChild(QtWidgets.QGroupBox, "global_fit_unified_species_group")
        assert panel is not None

        edit_a = _target_weight_edit(panel, target_name="A")
        edit_b = _target_weight_edit(panel, target_name="B")
        edit_a.setText("2.5")
        edit_b.setText("9.0")
        qt_app.processEvents()

        config = window._params_ics_tab._collect_parameter_config()
        assert config is not None
        selection = window._collect_dataset_selection()
        window._start_global_fit(config, selection)

        rrt = window._run_results_tab
        assert rrt._last_run_stamp["version"] == 3
        assert rrt._last_run_stamp["target_weights_applied"]["ds1"] == {"A": "1"}
        assert rrt._last_run_stamp["datasets"][0]["target_weights_applied"] == {"A": "1"}
    finally:
        window.close()
        qt_app.processEvents()


def test_run_stamp_hash_stable_across_dataset_orderings():
    from kindred.core.simulation_preparation import PreparedSimulationMetadata
    from kindred.gui.fitting.run_stamp import (
        build_global_fit_run_stamp,
        hash_global_fit_run_stamp,
    )

    applied = {"ds1": ["B", "A"]}
    config = {
        "parameters": {"k1": 0.2},
        "bounds": {"k1": (0.01, 1.0)},
        "log10_params": {"k1": False},
        "fixed_params": {},
        "method": "trf",
        "max_nfev": 10,
        "seed": 42,
        "use_parallel": False,
        "parallel_starts": 4,
    }

    rows_a = [{"id": "ds1", "label": "ds1", "include": True, "weight": 1.0}]
    rows_b = list(reversed(rows_a))
    prepared_meta_a = PreparedSimulationMetadata(
        version=1,
        mechanism_text_sha256="abc",
        mechanism_text_len=1,
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
    prepared_meta_b = PreparedSimulationMetadata(
        version=1,
        mechanism_text_sha256="abc",
        mechanism_text_len=1,
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

    stamp_a = build_global_fit_run_stamp(
        dataset_rows=rows_a,
        included_ids=["ds1"],
        applied_fit_targets=applied,
        weights_used={"ds1": 1.0},
        weight_mode="custom",
        fit_config=config,
        mechanism_text="rxn: A -> B; k=0.2",
        reactions_text="rxn: A -> B; k=0.2",
        prepared_simulation=prepared_meta_a,
        dataset_params={"ds1": {"init:A": 1.0}},
        dataset_variable_params={"ds1": {"init:B": {"initial": 0.2, "min": 0.0, "max": 10.0, "log10": False}}},
    )
    stamp_b = build_global_fit_run_stamp(
        dataset_rows=rows_b,
        included_ids=["ds1"],
        applied_fit_targets={"ds1": ["A", "B"]},
        weights_used={"ds1": 1.0},
        weight_mode="custom",
        fit_config=dict(config),
        mechanism_text="rxn: A -> B; k=0.2",
        reactions_text="rxn: A -> B; k=0.2",
        prepared_simulation=prepared_meta_b,
        dataset_params={"ds1": {"init:A": 1.0}},
        dataset_variable_params={
            "ds1": {"init:B": {"max": 10.0, "min": 0.0, "initial": 0.2, "log10": False}}
        },
    )
    assert hash_global_fit_run_stamp(stamp_a) == hash_global_fit_run_stamp(stamp_b)
