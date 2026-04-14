from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest


pytestmark = [pytest.mark.gui]


@dataclass
class _DummySpecies:
    initial_conc: float = 0.0


class _DummyMechanism:
    def __init__(self) -> None:
        self.metadata = {}
        self.species = {"A": _DummySpecies(initial_conc=1.0)}

    def species_names(self):
        return ["A"]


def test_simulation_worker_result_payload_includes_mechanism_for_prepared_completion_bookkeeping(
    monkeypatch, qt_app
):
    _ = qt_app

    from kindred.core.simulator.solvers import SimulationOutput
    from kindred.gui.simulation_worker import SimulationWorker

    def _fake_solve(_request):
        return SimulationOutput(
            t=np.asarray([0.0, 1.0], dtype=float),
            Y=np.asarray([[1.0, 1.0]], dtype=float),
            provenance={},
        )

    monkeypatch.setattr("kindred.core.simulator.solvers.solve_ode", _fake_solve)
    monkeypatch.setattr(
        "kindred.core.simulator.parameter_algebra.apply_parameter_algebra_to_mechanism",
        lambda *_args, **_kwargs: None,
    )

    prepared = {
        "mechanism": _DummyMechanism(),
        "rhs": (lambda _t, y: y),
        "y0": np.asarray([1.0], dtype=float),
        "species_names": ["A"],
    }

    worker = SimulationWorker(
        mechanism_text="reaction: A -> A ; k=0.0",
        initials={"A": 1.0},
        t_span=(0.0, 1.0),
        solver_config={"solver": "BDF", "grid": {"N": 2}},
        prepared=prepared,
    )

    captured: list[dict] = []
    worker.result_ready.connect(lambda payload: captured.append(payload))

    worker.run()  # run synchronously for inspection (no thread start)
    assert captured
    assert captured[0]["mechanism"] is prepared["mechanism"]
    assert captured[0]["mechanism_text"] == "reaction: A -> A ; k=0.0"


def test_simulation_worker_result_payload_omits_mechanism_for_secondary_completion_payloads(
    monkeypatch, qt_app
):
    _ = qt_app

    from kindred.core.simulator.solvers import SimulationOutput
    from kindred.gui.simulation_worker import SimulationWorker

    def _fake_solve(_request):
        return SimulationOutput(
            t=np.asarray([0.0, 1.0], dtype=float),
            Y=np.asarray([[1.0, 1.0]], dtype=float),
            provenance={},
        )

    monkeypatch.setattr("kindred.core.simulator.solvers.solve_ode", _fake_solve)
    monkeypatch.setattr(
        "kindred.core.simulator.parameter_algebra.apply_parameter_algebra_to_mechanism",
        lambda *_args, **_kwargs: None,
    )

    prepared = {
        "mechanism": _DummyMechanism(),
        "rhs": (lambda _t, y: y),
        "y0": np.asarray([1.0], dtype=float),
        "species_names": ["A"],
    }

    worker = SimulationWorker(
        mechanism_text="reaction: A -> A ; k=0.0",
        initials={"A": 1.0},
        t_span=(0.0, 1.0),
        solver_config={"solver": "BDF", "grid": {"N": 2}},
        prepared=prepared,
        include_mechanism_in_result_payload=False,
    )

    captured: list[dict] = []
    worker.result_ready.connect(lambda payload: captured.append(payload))

    worker.run()  # run synchronously for inspection (no thread start)
    assert captured
    assert "mechanism" not in captured[0]
    assert captured[0]["mechanism_text"] == "reaction: A -> A ; k=0.0"


def test_simulation_worker_secondary_payload_reports_base_species_count_for_algebra_status(
    monkeypatch, qt_app
):
    _ = qt_app

    from kindred.core.simulator.solvers import SimulationOutput
    from kindred.gui.simulation_worker import SimulationWorker

    def _fake_solve(_request):
        return SimulationOutput(
            t=np.asarray([0.0, 1.0], dtype=float),
            Y=np.asarray([[1.0, 1.0]], dtype=float),
            provenance={},
        )

    monkeypatch.setattr("kindred.core.simulator.solvers.solve_ode", _fake_solve)
    monkeypatch.setattr(
        "kindred.core.simulator.parameter_algebra.apply_parameter_algebra_to_mechanism",
        lambda *_args, **_kwargs: None,
    )

    prepared = {
        "mechanism": _DummyMechanism(),
        "rhs": (lambda _t, y: y),
        "y0": np.asarray([1.0], dtype=float),
        "species_names": ["A"],
    }

    worker = SimulationWorker(
        mechanism_text="reaction: A -> A ; k=0.0",
        initials={"A": 1.0},
        t_span=(0.0, 1.0),
        solver_config={"solver": "BDF", "grid": {"N": 2}},
        prepared=prepared,
        include_mechanism_in_result_payload=False,
    )
    monkeypatch.setattr(
        worker,
        "_evaluate_algebra_outputs",
        lambda **_kwargs: (
            np.asarray([[1.0, 1.0], [0.2, 0.3]], dtype=float),
            ["A", "Alg"],
            {},
            [{"kind": "algebra_error", "message": "bad algebra"}],
            [],
        ),
    )

    captured: list[dict] = []
    worker.result_ready.connect(lambda payload: captured.append(payload))

    worker.run()

    assert captured
    assert "mechanism" not in captured[0]
    assert captured[0]["base_species_count"] == 1
