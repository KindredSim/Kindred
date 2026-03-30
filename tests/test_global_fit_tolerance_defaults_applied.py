import numpy as np
import pytest


pytestmark = [pytest.mark.gui]


def test_global_fit_window_applies_ftol_xtol_defaults_to_ui(qt_app):
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
        config_defaults={"ftol": 1e-8, "xtol": 1e-9},
    )
    try:
        assert window._params_ics_tab._ftol_edit.text() == "1e-8"
        assert window._params_ics_tab._xtol_edit.text() == "1e-9"
        config = window._params_ics_tab._collect_parameter_config()
        assert config is not None
        assert config["ftol"] == pytest.approx(1e-8)
        assert config["xtol"] == pytest.approx(1e-9)
    finally:
        window.close()
