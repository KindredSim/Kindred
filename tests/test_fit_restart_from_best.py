import numpy as np
import pytest

pytestmark = pytest.mark.gui



def test_global_fit_restart_uses_staged_dataset_variable_initials(qt_app, monkeypatch):
    """A second global run should seed dataset-variable initials from best-so-far without applying."""
    from PySide6 import QtCore

    from kindred.core.analysis.dataset_parameter_overrides import split_fit_dataset_parameter_overrides
    from kindred.gui.fitting.window import FittingWindow

    t = np.linspace(0.0, 1.0, 5)
    dataset_entries = [
        {
            "id": "ds1",
            "label": "ds1",
            "t": t,
            "species_data": {"A": np.ones_like(t)},
            "selected_species": ["A"],
            "weight": 1.0,
            "include": True,
        }
    ]
    dataset_payloads = [{"id": "ds1", "t": t, "y": np.vstack([np.ones_like(t)]), "species": ["A"]}]

    captured = {}

    class _FakeWorker(QtCore.QObject):
        progress = QtCore.Signal(int, str)
        finished = QtCore.Signal(dict)
        error = QtCore.Signal(str)

        def __init__(self, datasets, shared_params, *, dataset_overrides=None, dataset_variable_params=None, **kwargs):
            super().__init__()
            captured["dataset_overrides"] = dataset_overrides
            captured["dataset_variable_params"] = dataset_variable_params

        def start(self):
            pass

        def isRunning(self):
            return False

        def cancel(self):
            pass

    monkeypatch.setattr("kindred.gui.fitting.window.GlobalFitWorker", _FakeWorker)

    window = FittingWindow(
        mode="global",
        parameter_defs=[{"name": "k1", "value": 0.2, "min": 0.01, "max": 1.0}],
        dataset_entries=dataset_entries,
        simulation_func=lambda params: {"t": t, "species": {"A": np.ones_like(t)}},
        dataset_variable_params={"ds1": {"init:A": {"initial": 1.0, "min": 0.1, "max": 2.0}}},
        dataset_payloads=dataset_payloads,
        dataset_weights={"ds1": 1.0},
    )
    try:
        window._params_ics_tab.set_staged_dataset_params({"ds1": {"init:A": 0.6}})
        window._start_fit()
        assert captured["dataset_variable_params"] is None
        _, dataset_variable_params = split_fit_dataset_parameter_overrides(captured["dataset_overrides"])
        assert dataset_variable_params["ds1"]["init:A"]["initial"] == pytest.approx(0.6)
    finally:
        window.close()
