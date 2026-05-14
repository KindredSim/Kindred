import numpy as np
import pytest


pytestmark = [pytest.mark.gui]


def test_local_initial_fit_toggle_controls_solver_payload(qt_app, monkeypatch):
    from PySide6 import QtCore

    from kindred.core.analysis.dataset_parameter_overrides import split_fit_dataset_parameter_overrides
    from kindred.gui.fitting.window import FittingWindow

    t = np.linspace(0.0, 1.0, 5)
    dataset_entries = [
        {
            "id": "ds1",
            "label": "ds1",
            "t": t.copy(),
            "species_data": {"A": np.ones_like(t)},
            "selected_species": ["A"],
            "weight": 1.0,
            "include": True,
        }
    ]
    dataset_payloads = [{"id": "ds1", "t": t.copy(), "y": np.vstack([np.ones_like(t)]), "species": ["A"]}]

    captured = {}

    class _FakeWorker(QtCore.QObject):
        progress = QtCore.Signal(int, str)
        bestUpdated = QtCore.Signal(dict)
        finished = QtCore.Signal(dict)
        error = QtCore.Signal(str)

        def __init__(self, datasets, shared_params, *, dataset_overrides=None, dataset_params=None, dataset_variable_params=None, **kwargs):
            super().__init__()
            captured["dataset_overrides"] = dataset_overrides
            captured["dataset_params"] = dataset_params
            captured["dataset_variable_params"] = dataset_variable_params

        def start(self):
            pass

        def isRunning(self):
            return False

        def cancel(self):
            pass

    monkeypatch.setattr("kindred.gui.fitting.worker_launch.GlobalFitWorker", _FakeWorker)

    window = FittingWindow(
        mode="global",
        parameter_defs=[{"name": "k1", "value": 0.2, "min": 0.01, "max": 1.0}],
        dataset_entries=dataset_entries,
        simulation_func=lambda _params: {"t": t.copy(), "species": {"A": np.ones_like(t)}},
        dataset_variable_params={"ds1": {"init:A": {"initial": 1.0, "min": 0.1, "max": 2.0, "log10": False}}},
        dataset_payloads=dataset_payloads,
        dataset_weights={"ds1": 1.0},
    )
    try:
        ds_rows = [
            idx
            for idx, entry in enumerate(window._params_ics_tab.get_parameter_state())
            if entry.get("scope") == "dataset" and entry.get("dataset_id") == "ds1" and entry.get("param_name") == "init:A"
        ]
        assert len(ds_rows) == 1
        row = ds_rows[0]

        # Uncheck "Fit" for the local initial condition row.
        window._params_ics_tab._param_table.item(row, 0).setCheckState(QtCore.Qt.Unchecked)

        config = window._params_ics_tab.collect_parameter_config()
        assert config is not None
        window.run_fit()

        assert captured["dataset_params"] is None
        assert captured["dataset_variable_params"] is None
        ds_params, dataset_variable_params = split_fit_dataset_parameter_overrides(captured["dataset_overrides"])
        ds_params = ds_params["ds1"]
        assert ds_params["init:A"] == pytest.approx(1.0)
        assert "init:A" not in (dataset_variable_params or {}).get("ds1", {})
    finally:
        window.close()
