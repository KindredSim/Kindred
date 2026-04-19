from __future__ import annotations

import numpy as np
import pytest

pytestmark = [pytest.mark.gui]


def test_global_fit_worker_best_payload_accepts_typed_series_payload(qt_app) -> None:
    from kindred.core.fitting_evaluation import CallableFittingEvaluator
    from kindred.core.simulation_series_payload import SimulationSeriesPayload
    from kindred.gui.fitting.worker import GlobalFitWorker

    t = np.array([0.0, 1.0], dtype=float)
    y = np.array([1.0, 0.5], dtype=float)

    worker = GlobalFitWorker(
        datasets=[{"id": "ds1", "t": t.copy(), "y": y.copy(), "species": "A"}],
        shared_params={"k1": 0.5},
        fit_evaluator=CallableFittingEvaluator(
            lambda _params: SimulationSeriesPayload(
                t=t.copy(),
                species={"A": y.copy()},
                algebra_scalars={},
            )
        ),
    )

    sim_time, sim_species = worker._simulate_best_payload_result({"k1": 0.5})

    assert np.allclose(sim_time, t)
    assert np.allclose(sim_species["A"], y)
