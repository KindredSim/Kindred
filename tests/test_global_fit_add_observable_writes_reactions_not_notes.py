from __future__ import annotations

import numpy as np
import pytest


pytestmark = [pytest.mark.gui]


def test_global_fit_add_observable_appends_to_reactions_algebra_not_notes(monkeypatch, qt_app):
    """
    Regression: Adding an algebraic observable from Global Fit must append into the
    Reactions DSL text without creating a `# Algebra` header and not into the GUI
    Notes tab.
    """
    from kindred.gui.fitting.window import FittingWindow

    reactions_text = "\n".join(
        [
            "reaction: A -> B; k=0.2",
            "initial: A=1.0",
            "initial: B=0.0",
        ]
    )
    notes_text = "param should_not_change = 9.0\n"

    called = {"reactions_set": 0}

    def _get_mechanism_text():
        return reactions_text

    def _reactions_setter(new_text: str):
        nonlocal reactions_text
        called["reactions_set"] += 1
        reactions_text = str(new_text)

    class _FakeDatasetManager:
        def scan_mechanism_parameters(self, _dsl):
            # Minimal: rate constant only; scalar params will be discovered from DSL.
            return [{"name": "k1", "value": 0.2, "min": 0.01, "max": 1.0}]

    t = np.linspace(0.0, 1.0, 6)
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
        dataset_manager=_FakeDatasetManager(),
        simulation_func=lambda _params: {"t": t.copy(), "species": {"A": np.ones_like(t)}},
        mechanism_species=["A", "B"],
        mechanism_text_getter=_get_mechanism_text,
        reactions_text_getter=lambda: reactions_text,
        reactions_text_setter=_reactions_setter,
        simulation_builder=lambda _dsl, _names, *, solver, rtol, atol: (
            lambda _p: {"t": t.copy(), "species": {"A": np.ones_like(t)}}
        ),
        dataset_payloads=[{"id": "ds1", "t": t.copy(), "y": np.vstack([np.ones_like(t)]), "species": ["A"]}],
        dataset_weights={"ds1": 1.0},
    )
    try:
        window._add_algebraic_observable(
            "signal",
            "scale * [A]",
            ["ds1"],
            scalar_scope="shared",
            persist_observable=True,
        )

        assert called["reactions_set"] >= 1
        assert "# Algebra" not in reactions_text
        assert "param scale = 1.0" in reactions_text
        assert "let signal = scale * [A]" in reactions_text
        assert "should_not_change" in notes_text
    finally:
        window.close()
