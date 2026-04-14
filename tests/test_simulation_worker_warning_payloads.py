from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest


pytestmark = [pytest.mark.gui]


@dataclass
class _DummySpecies:
    initial_conc: float = 1.0


class _DummyMechanism:
    def __init__(self, *, algebra_text: str | None = None) -> None:
        self.metadata = {}
        if algebra_text is not None:
            self.metadata["algebra_text"] = algebra_text
        self.species = {"A": _DummySpecies(initial_conc=1.0)}


def _prepared_payload(*, algebra_text: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        mechanism=_DummyMechanism(algebra_text=algebra_text),
        species_names=["A"],
        initials_for_algebra={"A": 1.0},
        request=SimpleNamespace(solver="BDF"),
        solver_warning=None,
        solver_input="BDF",
        temperature_schedule=None,
    )


def test_simulation_worker_emits_nonfatal_algebra_warning_payload(monkeypatch, qtbot):
    from kindred.core.simulator.solvers import SimulationOutput
    from kindred.gui.simulation_worker import SimulationWorker

    monkeypatch.setattr(
        "kindred.core.simulation_preparation.prepare_simulation_worker_run",
        lambda **_kwargs: _prepared_payload(algebra_text="obs = A"),
    )
    monkeypatch.setattr(
        "kindred.core.simulator.solvers.solve_ode",
        lambda _request: SimulationOutput(
            t=np.asarray([0.0, 1.0], dtype=float),
            Y=np.asarray([[1.0, 0.5]], dtype=float),
            provenance={},
        ),
    )

    def _boom(*_args, **_kwargs):
        raise RuntimeError("algebra boom")

    monkeypatch.setattr("kindred.core.algebra.simulation_series.evaluate_algebra_series_for_simulation_with_errors", _boom)

    worker = SimulationWorker(
        "reaction: A -> B; k=0.2\ninitial: A=1.0\ninitial: B=0.0",
        {"A": 1.0, "B": 0.0},
        (0.0, 1.0),
        {"solver": "BDF", "grid": {"N": 5}},
    )

    with qtbot.waitSignal(worker.result_ready, timeout=3000) as blocker:
        worker.start()

    worker.wait(1000)
    payload = blocker.args[0]
    warnings = payload.get("warnings")
    assert isinstance(warnings, list) and len(warnings) == 1
    assert warnings[0]["kind"] == "algebra_warning"
    assert warnings[0]["details"]["stage"] == "algebra_evaluation"
    assert warnings[0]["exc_type"] == "RuntimeError"
    assert warnings[0]["message"] == "algebra boom"
    assert payload["algebra_errors"][0]["message"] == "algebra boom"
    assert payload["success"] is True


def test_simulation_worker_unexpected_internal_failure_reports_stage(monkeypatch, qtbot):
    from kindred.gui.simulation_worker import SimulationWorker

    def _boom(**_kwargs):
        raise RuntimeError("prepare plumbing broke")

    monkeypatch.setattr("kindred.core.simulation_preparation.prepare_simulation_worker_run", _boom)

    worker = SimulationWorker(
        "reaction: A -> B; k=0.2\ninitial: A=1.0\ninitial: B=0.0",
        {"A": 1.0, "B": 0.0},
        (0.0, 1.0),
        {"solver": "BDF", "grid": {"N": 5}},
    )

    with qtbot.waitSignal(worker.error, timeout=3000) as blocker:
        worker.start()

    worker.wait(1000)
    payload = blocker.args[0]
    assert payload["message"] == "prepare plumbing broke"
    assert payload["details"]["stage"] == "prepare_simulation"
