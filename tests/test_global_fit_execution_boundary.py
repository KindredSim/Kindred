from __future__ import annotations

import inspect

import numpy as np

from kindred.core.fitting_completion import GlobalFitCompletion
from kindred.core.fitting_optimization import FitResult


def _raw_dataset(dataset_id: str, y_values) -> dict[str, object]:
    y = np.asarray(y_values, dtype=float).reshape(-1)
    return {
        "id": str(dataset_id),
        "t": np.linspace(0.0, 1.0, y.size),
        "species": "A",
        "y": y,
    }


def test_global_fit_execution_module_owns_candidate_replay_and_completion_policy() -> None:
    import kindred.core.analysis.global_fit_execution as execution
    import kindred.core.analysis.global_fitting as global_fitting

    assert execution.ObjectiveDatasetInput.__module__ == "kindred.core.analysis.global_fit_execution"
    assert execution.DatasetSimulationEvaluation.__module__ == "kindred.core.analysis.global_fit_execution"
    assert execution.GlobalFitObjective.__module__ == "kindred.core.analysis.global_fit_execution"
    assert execution.assemble_global_fit_result.__module__ == "kindred.core.analysis.global_fit_execution"
    assert execution.evaluate_dataset_simulations.__module__ == "kindred.core.analysis.global_fit_execution"

    public_source = inspect.getsource(global_fitting)
    assert "class _GlobalFitObjective" not in public_source
    assert "def _assemble_global_fit_result" not in public_source
    assert "def _evaluate_dataset_simulations" not in public_source


def test_fit_global_composes_global_fit_execution_boundary(monkeypatch) -> None:
    import kindred.core.analysis.global_fitting as global_fitting

    calls: list[str] = []

    class _RecordingObjective:
        def __init__(self, **kwargs):
            calls.append("objective")
            self.kwargs = dict(kwargs)
            assert self.kwargs["payloads"][0].dataset_id == "ds1"
            assert self.kwargs["weights"]["ds1"] > 0.0

        def __call__(self, _params: np.ndarray) -> np.ndarray:
            return np.zeros(2, dtype=float)

    def _fit_parameters(_objective, _initial_params, **_kwargs):
        calls.append("optimizer")
        return FitResult(
            success=True,
            parameters={"k": 1.0},
            uncertainties=None,
            chi_squared=0.0,
            r_squared=1.0,
            residuals=np.zeros(2, dtype=float),
            nfev=1,
            message="forced",
            covariance=None,
        )

    def _assemble_global_fit_result(**kwargs):
        calls.append("assembler")
        assert kwargs["fitted_params"] == {"k": 1.0}
        assert kwargs["payloads"][0].dataset_id == "ds1"
        return global_fitting.GlobalFitResult(
            shared_params=kwargs["fitted_params"],
            dataset_params=kwargs["combined_dataset_params"],
            uncertainties=kwargs["uncertainties"],
            global_chi_squared=0.0,
            global_r_squared=1.0,
            dataset_info=[],
            nfev=kwargs["nfev"],
            message="assembled",
            completion=GlobalFitCompletion(
                status="ok",
                optimizer_converged=True,
                nonfinite_metrics=False,
            ),
            covariance=kwargs["covariance"],
            objective_residuals=kwargs["objective_residuals"],
        )

    monkeypatch.setattr(global_fitting, "GlobalFitObjective", _RecordingObjective, raising=False)
    monkeypatch.setattr(global_fitting, "fit_parameters", _fit_parameters)
    monkeypatch.setattr(global_fitting, "assemble_global_fit_result", _assemble_global_fit_result, raising=False)

    result = global_fitting.fit_global(
        lambda params: {
            "t": np.asarray([0.0, 1.0], dtype=float),
            "A": np.full(2, float(dict(params).get("k", 1.0)), dtype=float),
        },
        [_raw_dataset("ds1", [1.0, 1.0])],
        shared_params={"k": 1.0},
        max_nfev=1,
    )

    assert result.message == "assembled"
    assert calls == ["objective", "optimizer", "assembler"]
