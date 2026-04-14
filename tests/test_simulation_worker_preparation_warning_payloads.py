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
    def __init__(self) -> None:
        self.metadata = {}
        self.species = {"A": _DummySpecies(initial_conc=1.0)}


def test_simulation_worker_emits_preparation_warning_payload(monkeypatch, qtbot):
    from kindred.core.simulator.solvers import SimulationOutput
    from kindred.gui.simulation_worker import SimulationWorker

    monkeypatch.setattr(
        "kindred.core.simulation_preparation.prepare_simulation_worker_run",
        lambda **_kwargs: SimpleNamespace(
            mechanism=_DummyMechanism(),
            species_names=["A"],
            initials_for_algebra={"A": 1.0},
            warnings=["Sparse Jacobian unavailable; falling back to dense Jacobian: sparsity blew up"],
            request=SimpleNamespace(solver="BDF"),
            solver_warning=None,
            solver_input="BDF",
            temperature_schedule=None,
        ),
    )
    monkeypatch.setattr(
        "kindred.core.simulator.solvers.solve_ode",
        lambda _request: SimulationOutput(
            t=np.asarray([0.0, 1.0], dtype=float),
            Y=np.asarray([[1.0, 0.5]], dtype=float),
            provenance={},
        ),
    )

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
    assert warnings[0]["kind"] == "preparation_warning"
    assert warnings[0]["details"]["stage"] == "prepare_run_context"
    assert "Sparse Jacobian unavailable" in warnings[0]["message"]
