from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.unit
def test_prepared_simulation_returns_typed_series_payload() -> None:
    from kindred.core.simulation_preparation import build_prepared_simulation_func
    from kindred.core.simulation_series_payload import SimulationSeriesPayload

    prepared = build_prepared_simulation_func(
        mechanism_text="\n".join(
            [
                "reaction: A -> B; k=0.5",
                "initial: A=1.0",
                "initial: B=0.0",
            ]
        ),
        param_names=["k1"],
        t_end=1.0,
        num_points=5,
        solver="BDF",
    )

    result = prepared({"k1": 0.5})

    assert isinstance(result, SimulationSeriesPayload)
    assert np.asarray(result["t"], dtype=float).shape == (5,)
    assert "A" in result["species"]
    assert result.to_legacy_dict()["species"].keys() == result["species"].keys()


@pytest.mark.unit
def test_global_fit_accepts_typed_simulation_series_payload() -> None:
    from kindred.core.analysis.global_fitting import fit_global
    from kindred.core.simulation_series_payload import SimulationSeriesPayload

    t = np.array([0.0, 0.5, 1.0], dtype=float)
    y = np.array([1.0, 0.75, 0.5], dtype=float)

    def _simulation(_params):
        return SimulationSeriesPayload(
            t=t.copy(),
            species={"A": y.copy()},
            algebra_scalars={},
        )

    result = fit_global(
        _simulation,
        datasets=[{"id": "ds1", "t": t.copy(), "y": y.copy(), "species": "A"}],
        shared_params={"k1": 0.5},
        method="trf",
        max_nfev=2,
    )

    assert result.dataset_info
    assert result.dataset_info[0].dataset_id == "ds1"


@pytest.mark.gui
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
