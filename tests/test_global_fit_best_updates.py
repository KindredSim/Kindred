import numpy as np

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
        simulation_func=simulation,
        fit_func=fake_fit_global,
        best_update_interval_s=0.0,
    )
    worker.bestUpdated.connect(lambda payload: emitted.append(payload))

    worker._execute()

    assert [p["cost"] for p in emitted] == [10.0, 9.0, 8.0]
