from __future__ import annotations

import numpy as np
import pytest


pytestmark = [pytest.mark.gui]


def test_global_fit_does_not_require_initials_for_derived_series(main_window, monkeypatch):
    """
    Regression: Global Fit must only require initial concentrations for mechanism
    species (state variables), not for selected observed series names such as
    algebraic observables present in the dataset columns.
    """
    from PySide6 import QtWidgets

    # Seed a dataset with both mechanism species and a derived/observable column.
    t = np.linspace(0.0, 1.0, 6)
    dataset = {
        "t": t,
        "species": {
            "A": np.ones_like(t),
            "B": np.zeros_like(t),
            "selectivity": np.linspace(0.2, 0.8, t.size),
        },
    }
    data_panel = main_window._right_panel._data_manager
    data_panel._datasets.clear()
    data_panel._datasets["ds1"] = dataset

    main_window._mechanism_editor._reactions_text.setPlainText(
        "\n".join(
            [
                "reaction: A -> B; k=0.2",
                "initial: A=1.0",
                "initial: B=0.0",
                "# Algebra",
                "let selectivity = [B] / max([A] + [B], 1e-18)",
            ]
        )
    )

    # Keep setup deterministic and avoid reliance on slider variable metadata.
    monkeypatch.setattr(main_window, "_apply_parameter_overrides_to_dsl", lambda mech, _params: mech)
    monkeypatch.setattr(
        type(main_window),
        "_extract_mechanism_initials",
        lambda _self, _dsl: {"A": 1.0, "B": 0.0},
        raising=False,
    )
    monkeypatch.setattr(
        main_window._dataset_manager,
        "scan_mechanism_parameters",
        lambda _dsl: [{"name": "k1", "value": 0.2, "min": 0.01, "max": 1.0}],
    )

    # Avoid building a real simulation function; reaching the fit window is sufficient.
    monkeypatch.setattr("kindred.core.simulation_preparation.build_prepared_simulation_func", lambda **_kwargs: (lambda _p: {"t": t, "species": {"selectivity": np.ones_like(t)}}))

    captured = {"warnings": []}

    def _capture_warning(_parent, _title, text, *args, **kwargs):
        captured["warnings"].append(str(text))
        return QtWidgets.QMessageBox.StandardButton.Ok

    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", _capture_warning)

    class _FakeFitWindow(QtWidgets.QDialog):
        def __init__(self, *args, **kwargs):
            super().__init__()
            captured["fit_window"] = dict(kwargs)

        def setWindowTitle(self, *_):
            pass

        def show(self):
            pass

        def raise_(self):
            pass

        def activateWindow(self):
            pass

    monkeypatch.setattr("kindred.gui.fitting.window.FittingWindow", _FakeFitWindow)

    main_window._run_global_fit()

    assert not any("requires initial concentrations for" in msg for msg in captured["warnings"])
    assert "fit_window" in captured
