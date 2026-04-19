from __future__ import annotations

import numpy as np
import pytest

from kindred.core.fitting_completion import FitDiagnostic, GlobalFitCompletion
from kindred.core.simulation_failure import build_simulation_failure


pytestmark = [pytest.mark.gui]


def _make_window():
    from kindred.gui.fitting.window import FittingWindow

    t = np.arange(0, 10, dtype=float)
    dataset_entries = [
        {
            "id": "ds1",
            "label": "Dataset 1",
            "t": t.copy(),
            "species_data": {"A": t.copy()},
            "selected_species": ["A"],
            "weight": 1.0,
            "include": True,
        }
    ]

    def simulation_func(_params):
        return {"t": t.copy(), "species": {"A": t.copy()}}

    return FittingWindow(
        mode="global",
        parameter_defs=[{"name": "k1", "value": 0.2, "min": 0.01, "max": 1.0}],
        dataset_entries=dataset_entries,
        simulation_func=simulation_func,
        mechanism_species=["A"],
    )


def test_completion_dialog_spec_reports_failure_when_chi2_nonfinite_or_errors(qt_app):
    from kindred.core.analysis.global_fitting import DatasetFitInfo, GlobalFitResult

    window = _make_window()
    try:
        qt_app.processEvents()

        result = GlobalFitResult(
            shared_params={"k1": 0.2},
            dataset_params={},
            uncertainties=None,
            global_chi_squared=float("inf"),
            global_r_squared=0.0,
            dataset_info=[
                DatasetFitInfo(
                    dataset_id="ds1",
                    r_squared=0.0,
                    chi_squared=float("inf"),
                    rmse=float("inf"),
                    mae=float("inf"),
                    residuals=np.asarray([], dtype=float),
                    n_points=10,
                    weight=1.0,
                )
            ],
            nfev=1,
            message="fake",
            completion=GlobalFitCompletion(
                status="fail",
                optimizer_converged=True,
                nonfinite_metrics=True,
                dataset_failures={
                    "ds1": FitDiagnostic(
                        phase="final_replay",
                        dataset_id="ds1",
                        failure=build_simulation_failure(kind="simulation_error", message="outside model range"),
                    )
                },
            ),
        )

        severity, title, text = window._global_fit_completion_dialog_spec(result)
        assert severity == "fail"
        assert "failed" in title.lower()
        assert "dataset 1" in text.lower()
        assert "outside model range" in text.lower()

        result2 = GlobalFitResult(
            shared_params={"k1": 0.2},
            dataset_params={},
            uncertainties=None,
            global_chi_squared=1.234,
            global_r_squared=0.9,
            dataset_info=[],
            nfev=1,
            message="fake",
            completion=GlobalFitCompletion(
                status="ok",
                optimizer_converged=True,
                nonfinite_metrics=False,
            ),
        )
        severity2, title2, text2 = window._global_fit_completion_dialog_spec(result2)
        assert severity2 == "ok"
        assert "complete" in title2.lower()
        assert "chi" in text2.lower()
    finally:
        window.close()
        qt_app.processEvents()
