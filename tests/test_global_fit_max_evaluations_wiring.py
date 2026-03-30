import numpy as np
import pytest


pytestmark = [pytest.mark.gui]

def test_global_fit_window_default_max_evaluations_is_1000(qt_app):
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

    window = FittingWindow(
        mode="global",
        parameter_defs=[{"name": "k1", "value": 0.2, "min": 0.01, "max": 1.0}],
        dataset_entries=dataset_entries,
        simulation_func=lambda _params: {"t": t.copy(), "species": {"A": np.ones_like(t)}},
        dataset_payloads=dataset_payloads,
        dataset_weights={"ds1": 1.0},
    )
    try:
        assert window._params_ics_tab._max_eval_spin.value() == 1000
        config = window._params_ics_tab._collect_parameter_config()
        assert config is not None
        assert config["max_nfev"] == 1000
    finally:
        window.close()


def test_configure_fitting_persists_default_max_nfev_1000(qt_app, monkeypatch):
    from PySide6 import QtCore, QtWidgets

    from kindred.gui.mixins.fitting_mixin import FittingMixin

    settings = QtCore.QSettings("KindredTest", "KindredTestFittingDefaults")
    settings.clear()

    class _Host(QtWidgets.QWidget, FittingMixin):
        def __init__(self):
            super().__init__()
            self._settings = settings
            self._status_label = QtWidgets.QLabel("")

    monkeypatch.setattr(QtWidgets.QDialog, "exec", lambda _self: QtWidgets.QDialog.DialogCode.Accepted)

    host = _Host()
    try:
        host._configure_fitting()
        assert settings.value("fitting/max_nfev", type=int) == 1000
    finally:
        settings.clear()
        host.close()


def test_max_evaluations_spinbox_is_passed_to_worker(qt_app, monkeypatch):
    from PySide6 import QtCore

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
        finished = QtCore.Signal(dict)
        error = QtCore.Signal(str)

        def __init__(self, datasets, shared_params, *, max_nfev=None, **kwargs):
            super().__init__()
            captured["max_nfev"] = max_nfev

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
        simulation_func=lambda _params: {"t": t.copy(), "species": {"A": np.ones_like(t)}},
        dataset_payloads=dataset_payloads,
        dataset_weights={"ds1": 1.0},
    )
    try:
        window._params_ics_tab._max_eval_spin.setValue(123)
        config = window._params_ics_tab._collect_parameter_config()
        assert config is not None
        dataset_selection = window._collect_dataset_selection()
        window._start_global_fit(config, dataset_selection)
        assert captured["max_nfev"] == 123
    finally:
        window.close()
