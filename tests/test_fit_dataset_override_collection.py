from __future__ import annotations

import numpy as np

from kindred.core.analysis.dataset_parameter_overrides import (
    FitDatasetParameterOverrides,
    FitDatasetVariableParamSpec,
)
from kindred.core.analysis.global_fitting import GlobalFitResult
from kindred.gui.fitting.worker import GlobalFitWorker
from kindred.gui.fitting.run_stamp import build_global_fit_run_stamp


def test_build_global_fit_run_stamp_accepts_typed_dataset_overrides():
    overrides = [
        FitDatasetParameterOverrides(
            dataset_id="ds1",
            fixed_params={"init:A": 1.0},
            variable_params={"init:B": FitDatasetVariableParamSpec(initial=0.2, minimum=0.0, maximum=10.0, log10=False)},
        )
    ]

    stamp = build_global_fit_run_stamp(
        dataset_rows=[{"id": "ds1", "label": "Dataset 1", "weight": 1.0, "include": True}],
        included_ids=["ds1"],
        applied_fit_targets={"ds1": ["A"]},
        weights_used={"ds1": 1.0},
        weight_mode="custom",
        fit_config={
            "parameters": {"k1": 0.2},
            "fixed_params": {},
            "bounds": {"k1": (0.01, 1.0)},
            "log10_params": {"k1": False},
            "method": "trf",
            "max_nfev": 10,
            "seed": 42,
            "use_parallel": False,
            "parallel_starts": 1,
        },
        mechanism_text="rxn: A -> B; k1=0.2",
        reactions_text="rxn: A -> B; k1=0.2",
        dataset_overrides=overrides,
    )

    assert stamp["dataset_params"]["ds1"]["init:A"] == "1"
    assert stamp["dataset_variable_params"]["ds1"]["init:B"]["max"] == "10"


def test_global_fit_worker_accepts_typed_dataset_overrides():
    captured: dict[str, object] = {}

    def fake_fit_global(*_args, **kwargs):
        captured["dataset_overrides"] = kwargs.get("dataset_overrides")
        captured["dataset_params"] = kwargs.get("dataset_params")
        captured["dataset_variable_params"] = kwargs.get("dataset_variable_params")
        return GlobalFitResult(
            success=True,
            shared_params={"k": 0.2},
            dataset_params={"ds1": {"init:A": 1.0, "init:B": 0.2}},
            uncertainties=None,
            global_chi_squared=0.0,
            global_r_squared=1.0,
            dataset_info=[],
            nfev=1,
            message="ok",
            model_series={"ds1": {}},
            residual_series={"ds1": {}},
        )

    def simulation(_params):
        t = np.array([0.0, 1.0], dtype=float)
        return {"t": t, "species": {"A": np.array([1.0, 0.8], dtype=float)}}

    overrides = [
        FitDatasetParameterOverrides(
            dataset_id="ds1",
            fixed_params={"init:A": 1.0},
            variable_params={"init:B": FitDatasetVariableParamSpec(initial=0.2, minimum=0.0, maximum=10.0, log10=False)},
        )
    ]
    worker = GlobalFitWorker(
        [{"id": "ds1", "t": np.array([0.0, 1.0], dtype=float), "y": np.array([1.0, 0.8], dtype=float), "species": "A"}],
        {"k": 0.2},
        dataset_overrides=overrides,
        fit_evaluator=simulation,
        fit_func=fake_fit_global,
    )

    payload = worker._execute()

    assert payload is not None
    assert captured["dataset_overrides"] == overrides
    assert captured["dataset_params"] is None
    assert captured["dataset_variable_params"] is None
