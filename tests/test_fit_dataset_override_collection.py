from __future__ import annotations

from kindred.core.analysis.dataset_parameter_overrides import (
    FitDatasetParameterOverrides,
    FitDatasetVariableParamSpec,
)
from kindred.gui.fitting.run_stamp import build_global_fit_run_stamp
import pytest

pytestmark = pytest.mark.unit



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
            "parallel_starts": 1,
        },
        mechanism_text="rxn: A -> B; k1=0.2",
        reactions_text="rxn: A -> B; k1=0.2",
        dataset_overrides=overrides,
    )

    assert stamp["dataset_params"]["ds1"]["init:A"] == "1"
    assert stamp["dataset_variable_params"]["ds1"]["init:B"]["max"] == "10"
