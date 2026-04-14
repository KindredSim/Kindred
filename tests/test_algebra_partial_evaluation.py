from __future__ import annotations

import numpy as np
import pytest

from kindred.core.simulator.dsl import parse_dsl_to_mechanism


def _dsl_with_one_bad_observable() -> str:
    return "\n".join(
        [
            "reaction: PBMPBPIN -> PBMP ; k=1.0",
            "reaction: PBMP -> pinBOH ; k=1.0",
            "initial: PBMPBPIN=1.0",
            "initial: PBMP=0.0",
            "initial: pinBOH=0.0",
            "# Algebra",
            "let ok1 = [PBMPBPIN] + [PBMP]",
            "let bad = A0 + 1",
            "let ok2 = [PBMP] / max([PBMP] + [pinBOH], 1e-18)",
        ]
    )


def test_algebra_series_partial_eval_keeps_valid_observables():
    """
    Regression: one broken algebra line must not remove other valid observables.

    Pre-fix behavior: evaluate_algebra_series_for_simulation raises (unknown symbol),
    causing callers to drop the entire algebra block.
    """
    from kindred.core.algebra.simulation_series import evaluate_algebra_series_for_simulation

    dsl = _dsl_with_one_bad_observable()
    mechanism = parse_dsl_to_mechanism(dsl, initials={})

    t = np.asarray([0.0, 1.0, 2.0], dtype=float)
    species_series = {
        "PBMPBPIN": np.asarray([1.0, 0.5, 0.25], dtype=float),
        "PBMP": np.asarray([0.0, 0.5, 0.75], dtype=float),
        "pinBOH": np.asarray([0.0, 0.0, 0.0], dtype=float),
    }
    initials = {"PBMPBPIN": 1.0, "PBMP": 0.0, "pinBOH": 0.0}

    series, scalars = evaluate_algebra_series_for_simulation(
        mechanism,
        t=t,
        species_series=species_series,
        initials=initials,
    )

    assert "ok1" in series
    assert "ok2" in series
    assert "bad" not in series
    assert isinstance(scalars, dict)


@pytest.mark.gui
def test_simulation_worker_keeps_valid_algebra_series_when_one_fails(monkeypatch, qtbot):
    """
    Regression: the GUI SimulationWorker must keep successful algebra series in results
    even if one observable fails, so plot candidates retain them.
    """
    from kindred.core.simulator.solvers import SimulationOutput
    from kindred.gui.simulation_worker import SimulationWorker

    dsl = _dsl_with_one_bad_observable()
    t = np.asarray([0.0, 1.0, 2.0], dtype=float)

    # Order should match mechanism.species_names() for the stubbed output.
    species_names = parse_dsl_to_mechanism(dsl, initials={}).species_names()
    series_map = {
        "PBMPBPIN": np.asarray([1.0, 0.5, 0.25], dtype=float),
        "PBMP": np.asarray([0.0, 0.5, 0.75], dtype=float),
        "pinBOH": np.asarray([0.0, 0.0, 0.0], dtype=float),
    }
    Y = np.vstack([series_map[name] for name in species_names])

    def _fake_solve(_request):
        return SimulationOutput(t=t, Y=Y, provenance={})

    monkeypatch.setattr("kindred.core.simulator.solvers.solve_ode", _fake_solve)

    worker = SimulationWorker(
        dsl,
        {"PBMPBPIN": 1.0, "PBMP": 0.0, "pinBOH": 0.0},
        (0.0, 2.0),
        {"solver": "BDF", "rtol": 1e-6, "atol": 1e-12, "grid": {"N": 3}},
    )

    with qtbot.waitSignal(worker.result_ready, timeout=3000) as blocker:
        worker.start()

    worker.wait(1000)
    payload = blocker.args[0]

    names = list(payload.get("species_names") or [])
    assert "ok1" in names
    assert "ok2" in names
    assert "bad" not in names

    # GUI pipeline should also surface per-observable errors for minimal inline feedback.
    errors = payload.get("algebra_errors")
    assert isinstance(errors, list)
    assert any(e.get("name") == "bad" for e in errors if isinstance(e, dict))
