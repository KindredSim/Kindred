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
    from kindred.core.simulation_plan import SimulationAlgebraPolicy, SimulationPlan
    from kindred.core.simulation_preparation import SimulationExecutionRequest
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
    worker._simulation_plan = SimulationPlan.from_execution_request(  # type: ignore[attr-defined]
        SimulationExecutionRequest(
            prepared_payload=None,
            initials={"A": 1.0},
            t_span=(0.0, 1.0),
            solver_config={"solver": "BDF", "grid": {"N": 5}},
            mechanism_text="reaction: A -> B; k=0.2",
        ),
        execution_mode="explicit",
        algebra_policy=SimulationAlgebraPolicy.GUI_BEST_EFFORT,
    ).to_payload()

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


def test_simulation_worker_reads_algebra_policy_from_attached_plan(monkeypatch, qtbot):
    from kindred.core.simulation_algebra_policy import SimulationAlgebraEvaluation
    from kindred.core.simulator.solvers import SimulationOutput
    from kindred.core.simulation_plan import SimulationAlgebraPolicy, SimulationPlan
    from kindred.core.simulation_preparation import SimulationExecutionRequest
    from kindred.gui.simulation_worker import SimulationWorker

    captured = {}
    captured_prepare_kwargs = {}

    def _capture_prepare_kwargs(**kwargs):
        captured_prepare_kwargs.update(kwargs)
        return _prepared_payload(algebra_text="obs = A")

    monkeypatch.setattr(
        "kindred.core.simulation_preparation.prepare_simulation_worker_run",
        _capture_prepare_kwargs,
    )
    monkeypatch.setattr(
        "kindred.core.simulator.solvers.solve_ode",
        lambda _request: SimulationOutput(
            t=np.asarray([0.0, 1.0], dtype=float),
            Y=np.asarray([[1.0, 0.5]], dtype=float),
            provenance={},
        ),
    )

    def _capture_policy(policy, *_args, **_kwargs):
        captured["policy"] = policy
        return SimulationAlgebraEvaluation(series={}, scalars={}, errors=[], warning=None)

    monkeypatch.setattr("kindred.core.simulation_algebra_policy.evaluate_simulation_algebra", _capture_policy)

    worker = SimulationWorker(
        "reaction: A -> B; k=0.2\ninitial: A=1.0\ninitial: B=0.0",
        {"A": 1.0, "B": 0.0},
        (0.0, 1.0),
        {"solver": "BDF", "grid": {"N": 5}},
    )
    worker._simulation_plan = SimulationPlan.from_execution_request(  # type: ignore[attr-defined]
        SimulationExecutionRequest(
            prepared_payload=None,
            initials={"A": 1.0},
            t_span=(0.0, 1.0),
            solver_config={"solver": "BDF", "grid": {"N": 5}},
            mechanism_text="reaction: A -> B; k=0.2",
        ),
        execution_mode="explicit",
        algebra_policy=SimulationAlgebraPolicy.GUI_BEST_EFFORT,
    ).to_payload()

    with qtbot.waitSignal(worker.result_ready, timeout=3000):
        worker.start()

    worker.wait(1000)
    assert captured["policy"] is SimulationAlgebraPolicy.GUI_BEST_EFFORT
    assert captured_prepare_kwargs["execution_request"]["mechanism_text"] == "reaction: A -> B; k=0.2"
    assert captured_prepare_kwargs["execution_request"]["initials"] == {"A": 1.0}


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
