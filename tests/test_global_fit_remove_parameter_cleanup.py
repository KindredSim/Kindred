import numpy as np
import pytest

from PySide6 import QtWidgets

from kindred.gui.fitting.window import FittingWindow


pytestmark = [pytest.mark.gui]


def test_remove_global_initial_parameter_allows_readd(qt_app, monkeypatch):
    t = np.linspace(0.0, 1.0, 5)
    window = FittingWindow(
        mode="global",
        parameter_defs=[{"name": "k1", "value": 0.2, "min": 0.01, "max": 1.0}],
        dataset_entries=[
            {
                "id": "ds1",
                "label": "ds1",
                "t": t.copy(),
                "species_data": {"A": np.ones_like(t)},
                "selected_species": ["A"],
                "weight": 1.0,
                "include": True,
            }
        ],
        simulation_func=lambda _params: {"t": t.copy(), "species": {"A": np.ones_like(t)}},
        dataset_payloads=[{"id": "ds1", "t": t.copy(), "y": np.vstack([np.ones_like(t)]), "species": ["A"]}],
        dataset_weights={"ds1": 1.0},
        dataset_params={"ds1": {"init:A": 1.0}},
    )
    try:
        window._add_global_initial_parameter("A", ["ds1"])
        rows = [
            idx
            for idx, entry in enumerate(window._parameter_state)
            if entry.get("scope") == "shared" and entry.get("param_name") == "init:A"
        ]
        assert len(rows) == 1
        window._remove_parameter_rows(rows)

        calls = []

        def _fake_information(parent, title, text, *args, **kwargs):
            calls.append((str(title), str(text)))
            return QtWidgets.QMessageBox.StandardButton.Ok

        monkeypatch.setattr(QtWidgets.QMessageBox, "information", _fake_information)
        window._add_global_initial_parameter("A", ["ds1"])

        assert not calls, "Re-adding a removed global initial parameter should not claim it already exists."
        rows2 = [
            idx
            for idx, entry in enumerate(window._parameter_state)
            if entry.get("scope") == "shared" and entry.get("param_name") == "init:A"
        ]
        assert len(rows2) == 1
    finally:
        window.close()
