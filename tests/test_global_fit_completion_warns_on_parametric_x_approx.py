from __future__ import annotations

import numpy as np
import pytest


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


def test_completion_dialog_spec_warns_when_alignment_has_approximations(qt_app):
    from kindred.core.analysis.global_fitting import DatasetFitInfo, GlobalFitResult

    window = _make_window()
    try:
        qt_app.processEvents()

        result = GlobalFitResult(
            success=True,
            shared_params={"k1": 0.2},
            dataset_params={},
            uncertainties=None,
            global_chi_squared=1.234,
            global_r_squared=0.9,
            dataset_info=[
                DatasetFitInfo(
                    dataset_id="ds1",
                    r_squared=0.9,
                    chi_squared=1.234,
                    rmse=0.1,
                    mae=0.1,
                    residuals=np.asarray([], dtype=float),
                    n_points=10,
                    weight=1.0,
                )
            ],
            nfev=1,
            message="fake",
        )
        setattr(result, "dataset_errors", {})
        setattr(result, "dataset_warnings", {"ds1": "X alignment required approximations"})

        severity, title, text = window._global_fit_completion_dialog_spec(result)
        assert severity == "warn"
        assert "warning" in title.lower() or "warnings" in title.lower()
        assert "dataset 1" in text.lower()
        assert "approxim" in text.lower()
    finally:
        window.close()
        qt_app.processEvents()
