from __future__ import annotations

import numpy as np
import pytest


pytestmark = [pytest.mark.gui]


def test_global_fit_observable_dropdown_reads_from_reactions_dsl_declarations(qt_app, monkeypatch):
    """
    Regression: Global Fit must detect algebraic observables from the current
    Reactions DSL declarations, not from the GUI Notes tab.
    """
    from PySide6 import QtWidgets

    from kindred.gui.fitting.window import FittingWindow

    reactions_dsl = "\n".join(
        [
            "reaction: A -> B; k=0.2",
            "initial: A=1.0",
            "initial: B=0.0",
            "# Algebraic observables",
            "let total_PBMP = [PBMPBPIN] + [PBMP]",
            "let selectivity = [PBMP] / max([PBMP] + [pinBOH], 1e-18)",
        ]
    )

    t = np.linspace(0.0, 1.0, 6)
    dataset_entries = [
        {
            "id": "ds1",
            "label": "ds1",
            "t": t.copy(),
            "species_data": {"selectivity": np.linspace(0.2, 0.8, t.size)},
            "selected_species": ["selectivity"],
            "weight": 1.0,
            "include": True,
        }
    ]

    captured = {}

    class _FakeDialog:
        def __init__(self, *, available_observables=None, **kwargs):
            captured["available_observables"] = dict(available_observables or {})

        def exec(self):
            return QtWidgets.QDialog.Rejected

        def selection(self):
            return None

    monkeypatch.setattr("kindred.gui.fitting.parameters_ics_tab._AddFittableParameterDialog", _FakeDialog)

    window = FittingWindow(
        mode="global",
        parameter_defs=[{"name": "k1", "value": 0.2, "min": 0.01, "max": 1.0}],
        dataset_entries=dataset_entries,
        simulation_func=lambda _params: {"t": t.copy(), "species": {"selectivity": np.linspace(0.2, 0.8, t.size)}},
        mechanism_species=["A", "B"],
        mechanism_text_getter=lambda: reactions_dsl,
        reactions_text_getter=lambda: reactions_dsl,
        reactions_text_setter=lambda _s: None,
        simulation_builder=lambda _dsl, _names: (lambda _p: {"t": t.copy(), "species": {}}),
        dataset_payloads=[{"id": "ds1", "t": t.copy(), "y": np.vstack([np.linspace(0.2, 0.8, t.size)]), "species": ["selectivity"]}],
        dataset_weights={"ds1": 1.0},
        runtime_lane_budget=lambda dataset_count: max(1, int(dataset_count)),
)
    try:
        monkeypatch.setattr(window, "_selected_dataset_ids", lambda: ["ds1"])
        window._params_ics_tab._add_parameter()
        obs = captured.get("available_observables") or {}
        assert "total_PBMP" in obs
        assert "selectivity" in obs
    finally:
        window.close()
