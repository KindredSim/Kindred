import numpy as np
import pytest

from kindred.core.analysis.global_fitting import DatasetFitInfo, GlobalFitResult
from kindred.gui.fitting.worker import GlobalFitWorker


def test_global_fit_worker_emits_best_updated_only_on_improvement(qt_app):
    emitted = []

    def fake_fit_global(*_args, **kwargs):
        progress = kwargs.get("progress_callback")
        assert progress is not None
        progress(1, 10.0, {"k": 1.0})
        progress(2, 12.0, {"k": 1.1})
        progress(3, 9.0, {"k": 0.9})
        progress(4, 9.0, {"k": 0.9})
        progress(5, 8.0, {"k": 0.8})
        return GlobalFitResult(
            success=True,
            shared_params={"k": 0.8},
            dataset_params={"ds": {}},
            uncertainties=None,
            global_chi_squared=1.0,
            global_r_squared=0.0,
            dataset_info=[
                DatasetFitInfo(
                    dataset_id="ds",
                    r_squared=0.0,
                    chi_squared=1.0,
                    rmse=1.0,
                    mae=1.0,
                    residuals=np.array([0.0]),
                    n_points=1,
                    weight=1.0,
                )
            ],
            nfev=5,
            message="ok",
            covariance=None,
            objective_residuals=np.array([0.0]),
            model_series={"ds": {}},
            residual_series={"ds": {}},
        )

    datasets = [{"id": "ds", "t": np.array([0.0]), "y": np.array([0.0]), "species": "A"}]

    def simulation(_params):
        return {"t": np.array([0.0]), "species": {"A": np.array([0.0])}}

    worker = GlobalFitWorker(
        datasets,
        {"k": 1.0},
        fit_evaluator=simulation,
        fit_func=fake_fit_global,
        best_update_interval_s=0.0,
        plot_update_interval_s=0.0,
    )
    worker.bestUpdated.connect(lambda payload: emitted.append(payload))

    worker._execute()

    assert [p["cost"] for p in emitted] == [10.0, 9.0, 8.0]


def test_global_fit_worker_best_payload_stats_remain_raw_under_target_weighting(qt_app):
    from kindred.core.fitting_evaluation import CallableFittingEvaluator

    t_obs = np.linspace(0.0, 1.0, 5, dtype=float)

    def simulation(_params):
        return {
            "t": t_obs.copy(),
            "species": {
                "A": np.zeros_like(t_obs),
                "B": np.zeros_like(t_obs),
            },
        }

    worker = GlobalFitWorker(
        [
            {
                "id": "ds1",
                "t": t_obs.copy(),
                "y": np.vstack([np.ones_like(t_obs), 2.0 * np.ones_like(t_obs)]),
                "species": ["A", "B"],
                "target_weights": {"A": 5.0, "B": 1.0},
            }
        ],
        {"k": 0.2},
        fit_evaluator=CallableFittingEvaluator(simulation),
        best_update_interval_s=0.0,
    )

    model_series, residual_series, plot_model_series, plot_model_x, dataset_stats = worker._build_best_payload_series(
        shared_params={"k": 0.2},
        dataset_params={},
    )

    np.testing.assert_allclose(model_series["ds1"]["A"], np.zeros_like(t_obs))
    np.testing.assert_allclose(plot_model_series["ds1"]["A"], np.zeros_like(t_obs))
    np.testing.assert_allclose(plot_model_x["ds1"], t_obs)
    np.testing.assert_allclose(
        residual_series["ds1"]["A"],
        -np.ones_like(t_obs),
    )
    np.testing.assert_allclose(
        residual_series["ds1"]["B"],
        np.full_like(t_obs, -2.0),
    )
    assert dataset_stats["ds1"]["chi_squared"] == pytest.approx(2.5)
