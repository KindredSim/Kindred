from __future__ import annotations

from pathlib import Path
import threading

import numpy as np
import pytest

from kindred.gui.fitting.worker import GlobalFitWorker
from kindred.gui.simulation_worker import SimulationWorker

pytestmark = pytest.mark.gui


ROOT = Path(__file__).resolve().parents[1]
REGRESSION_DIR = ROOT / "benchmarks" / "regression_suite"
NONSTIFF_DSL = REGRESSION_DIR / "nonstiff_first_order.dsl"
NONSTIFF_DATA = REGRESSION_DIR / "nonstiff_first_order.csv"


def _load_nonstiff_dataset(limit: int = 12) -> dict:
    """Return a small slice of the non-stiff benchmark dataset."""
    data = np.genfromtxt(NONSTIFF_DATA, delimiter=",", names=True)
    t = np.asarray(data["time"][:limit], dtype=float)
    species = {
        "A": np.asarray(data["A"][:limit], dtype=float),
        "B": np.asarray(data["B"][:limit], dtype=float),
    }
    return {"t": t, "species": species}


def test_simulation_worker_completes_small_decay(qtbot):
    """Run the real SimulationWorker against a tiny benchmark mechanism."""
    mechanism = NONSTIFF_DSL.read_text()
    initials = {"A": 1.0, "B": 0.0}
    worker = SimulationWorker(
        mechanism,
        initials,
        (0.0, 1.5),
        {"solver": "LSODA", "rtol": 1e-5, "atol": 1e-8, "grid": {"N": 40}},
    )

    with qtbot.waitSignal(worker.result_ready, timeout=5000) as blocker:
        worker.start()

    worker.wait(2000)
    result = blocker.args[0]
    assert result["success"] is True
    assert result["t"].size > 4
    assert result["Y"].shape == (len(result["species_names"]), result["t"].size)
    assert np.all(np.isfinite(result["Y"]))


def test_simulation_worker_surfaces_solver_failure(monkeypatch, qtbot):
    """SimulationWorker should emit error when the solver raises."""
    mechanism = "reaction: A -> B; k=0.2\ninitial: A=1.0\ninitial: B=0.0"

    def _boom(_request):
        raise RuntimeError("solver blew up")

    monkeypatch.setattr("kindred.core.simulator.solvers.solve_ode", _boom)

    worker = SimulationWorker(
        mechanism,
        {"A": 1.0, "B": 0.0},
        (0.0, 0.5),
        {"solver": "LSODA", "grid": {"N": 10}},
    )

    with qtbot.waitSignal(worker.error, timeout=3000) as blocker:
        worker.start()

    worker.wait(1000)
    error_payload = blocker.args[0]
    assert isinstance(error_payload, dict)
    assert error_payload["kind"] == "simulation_error"
    assert error_payload["message"] == "solver blew up"


def test_simulation_worker_cancellation_wires_scipy_terminal_event(monkeypatch, qtbot):
    """Regression: cancellation must not raise from progress callbacks (Fortran-unsafe for LSODA)."""
    mechanism = "reaction: A -> B; k=0.2\ninitial: A=1.0\ninitial: B=0.0"

    def _fake_solve_ode(request):
        cancel_event = list(request.events or [])
        assert len(cancel_event) == 1
        cancel_event = cancel_event[0]
        assert getattr(cancel_event, "terminal", False) is True

        progress_callback = request.progress_callback
        assert progress_callback is not None

        assert float(cancel_event(0.0, np.zeros(1))) > 0.0
        worker._cancelled = True
        assert float(cancel_event(0.0, np.zeros(1))) < 0.0
        progress_callback(0.0, 0.0, 1.0)
        worker._cancelled = False

        from kindred.core.simulator.solvers import SimulationOutput

        return SimulationOutput(t=np.asarray([0.0, 1.0]), Y=np.asarray([[1.0, 0.5]]), provenance={})

    monkeypatch.setattr("kindred.core.simulator.solvers.solve_ode", _fake_solve_ode)

    worker = SimulationWorker(
        mechanism,
        {"A": 1.0, "B": 0.0},
        (0.0, 1.0),
        {"solver": "LSODA", "grid": {"N": 5}},
    )

    with qtbot.waitSignal(worker.result_ready, timeout=3000) as blocker:
        worker.start()

    worker.wait(1000)
    result = blocker.args[0]
    assert result["success"] is True


def test_simulation_worker_treats_brentq_valueerror_during_cancellation_as_clean_exit(
    monkeypatch,
    qtbot,
):
    mechanism = "reaction: A -> B; k=0.2\ninitial: A=1.0\ninitial: B=0.0"

    def _boom_solve_ivp(*, fun, t_span, y0, **kwargs):  # noqa: ANN001
        # Simulate the race condition where the cancellation flag flips just before SciPy's root-finder runs.
        worker._cancelled = True
        raise ValueError("f(a) and f(b) must have different signs")

    monkeypatch.setattr("kindred.core.simulator.solvers._solve_ivp", _boom_solve_ivp)

    worker = SimulationWorker(
        mechanism,
        {"A": 1.0, "B": 0.0},
        (0.0, 1.0),
        {"solver": "LSODA", "grid": {"N": 5}},
    )

    with qtbot.waitSignal(worker.error, timeout=3000) as blocker:
        worker.start()

    worker.wait(1000)
    assert blocker.args[0]["kind"] == "cancelled"
    assert blocker.args[0]["message"] == "Simulation cancelled by user"


def test_simulation_worker_treats_brentq_valueerror_during_cancellation_as_clean_exit_non_lsoda(
    monkeypatch,
    qtbot,
):
    mechanism = "reaction: A -> B; k=0.2\ninitial: A=1.0\ninitial: B=0.0"

    def _boom_solve_ivp(*, fun, t_span, y0, **kwargs):  # noqa: ANN001
        # Simulate the race condition where the cancellation flag flips just before SciPy's root-finder runs.
        worker._cancelled = True
        raise ValueError("f(a) and f(b) must have different signs")

    monkeypatch.setattr("kindred.core.simulator.solvers._solve_ivp", _boom_solve_ivp)

    worker = SimulationWorker(
        mechanism,
        {"A": 1.0, "B": 0.0},
        (0.0, 1.0),
        {"solver": "Radau", "grid": {"N": 5}},
    )

    with qtbot.waitSignal(worker.error, timeout=3000) as blocker:
        worker.start()

    worker.wait(1000)
    assert blocker.args[0]["kind"] == "cancelled"
    assert blocker.args[0]["message"] == "Simulation cancelled by user"


def test_simulation_worker_cancel_immediately_after_start_prevents_solve_entry(monkeypatch, qtbot):
    mechanism = "reaction: A -> B; k=0.2\ninitial: A=1.0\ninitial: B=0.0"
    thread_entered = threading.Event()
    allow_run = threading.Event()
    solve_calls: list[object] = []
    errors: list[dict] = []
    results: list[dict] = []

    class _GateWorker(SimulationWorker):
        def run(self):
            thread_entered.set()
            assert allow_run.wait(timeout=2.0)
            super().run()

    def _fake_solve_ode(request):
        solve_calls.append(request)

        from kindred.core.simulator.solvers import SimulationOutput

        return SimulationOutput(
            t=np.asarray([0.0, 1.0]),
            Y=np.asarray([[1.0, 0.5], [0.0, 0.5]]),
            provenance={},
        )

    monkeypatch.setattr("kindred.core.simulator.solvers.solve_ode", _fake_solve_ode)

    worker = _GateWorker(
        mechanism,
        {"A": 1.0, "B": 0.0},
        (0.0, 1.0),
        {"solver": "LSODA", "grid": {"N": 5}},
    )
    worker.error.connect(lambda payload: errors.append(payload))
    worker.result_ready.connect(lambda payload: results.append(payload))

    worker.start()
    qtbot.waitUntil(thread_entered.is_set, timeout=2000)
    worker.cancel()
    allow_run.set()

    qtbot.waitUntil(lambda: bool(errors or results), timeout=3000)
    worker.wait(1000)

    assert solve_calls == []
    assert results == []
    assert len(errors) == 1
    assert errors[0]["kind"] == "cancelled"
    assert errors[0]["message"] == "Simulation cancelled by user"


def test_global_fit_worker_smoke(monkeypatch, qtbot):
    """GlobalFitWorker should converge quickly on a two-dataset toy problem."""
    t_axis = np.linspace(0.0, 2.5, 10)
    datasets = [
        {"id": "ds1", "t": t_axis, "y": np.exp(-0.3 * t_axis), "species": "A"},
        {"id": "ds2", "t": t_axis, "y": 0.85 * np.exp(-0.28 * t_axis), "species": "A"},
    ]
    shared_params = {"k": 0.2}
    dataset_params = {"ds1": {"scale": 1.0}, "ds2": {"scale": 0.85}}

    def _simulate(params):
        k = float(params.get("k", 0.2))
        scale = float(params.get("scale", 1.0))
        return {"t": t_axis, "species": {"A": scale * np.exp(-k * t_axis)}}

    worker = GlobalFitWorker(
        datasets,
        shared_params,
        dataset_params=dataset_params,
        bounds={"k": (0.05, 0.6)},
        method="trf",
        max_nfev=24,
        simulation_func=_simulate,
    )

    with qtbot.waitSignal(worker.finished, timeout=7000) as blocker:
        worker.start()

    worker.wait(2000)
    payload = blocker.args[0]
    result = payload["result"]
    assert result.success is True
    assert np.isfinite(result.global_chi_squared)
    assert "k" in result.shared_params
    assert 0.05 < result.shared_params["k"] < 0.6
