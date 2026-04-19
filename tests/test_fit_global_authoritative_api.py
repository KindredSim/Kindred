from __future__ import annotations

import importlib
import importlib.resources
import inspect

import numpy as np
import pytest


@pytest.mark.unit
def test_fit_global_authoritative_api_module_exports_core_contract() -> None:
    api = importlib.import_module("kindred.core.api.fitting")

    assert hasattr(api, "fit_global")
    assert hasattr(api, "GlobalFitResult")
    assert hasattr(api, "DatasetFitInfo")


@pytest.mark.unit
def test_fit_global_authoritative_api_signature_matches_current_core_surface() -> None:
    from kindred.core.api.fitting import fit_global

    signature = inspect.signature(fit_global)

    assert "process_pool_callback" not in signature.parameters
    assert "dataset_overrides" in signature.parameters
    assert all(param.kind is not inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values())


@pytest.mark.unit
def test_global_fit_result_requires_explicit_completion_contract() -> None:
    from kindred.core.api.fitting import GlobalFitResult

    with pytest.raises(TypeError):
        GlobalFitResult(
            shared_params={},
            dataset_params={},
            uncertainties=None,
            global_chi_squared=0.0,
            global_r_squared=1.0,
            dataset_info=[],
            nfev=1,
            message="ok",
        )


@pytest.mark.unit
def test_gui_global_fit_code_imports_from_core_api_not_gui_shim() -> None:
    targets = (
        ("kindred.gui.fitting", "window.py"),
        ("kindred.gui.fitting", "worker.py"),
    )

    for package, filename in targets:
        source = importlib.resources.files(package).joinpath(filename).read_text(encoding="utf-8")
        assert "from kindred.gui.compat.shims import fit_global" not in source
        assert "from kindred.core.api.fitting import fit_global" in source


@pytest.mark.unit
def test_authoritative_fit_global_api_accepts_shared_fitting_evaluator_contract() -> None:
    from kindred.core.api.fitting import fit_global
    from kindred.core.fitting_evaluation import CallableFittingEvaluator

    datasets = [
        {
            "id": "ds1",
            "t": np.array([0.0, 1.0], dtype=float),
            "y": np.array([1.0, np.exp(-0.2)], dtype=float),
            "species": "A",
        }
    ]

    def simulate(params):
        t = np.array([0.0, 1.0], dtype=float)
        return {"t": t, "species": {"A": np.exp(-float(params["k"]) * t)}}

    result = fit_global(
        CallableFittingEvaluator(simulate),
        datasets,
        {"k": 0.2},
        method="trf",
        max_nfev=2,
    )

    assert result.completion.status == "ok"


@pytest.mark.unit
def test_authoritative_fit_global_api_accepts_raw_callable_evaluator() -> None:
    from kindred.core.api.fitting import fit_global

    datasets = [
        {
            "id": "ds1",
            "t": np.array([0.0, 1.0], dtype=float),
            "y": np.array([1.0, np.exp(-0.2)], dtype=float),
            "species": "A",
        }
    ]

    def simulate(params):
        t = np.array([0.0, 1.0], dtype=float)
        return {"t": t, "species": {"A": np.exp(-float(params["k"]) * t)}}

    result = fit_global(
        simulate,
        datasets,
        {"k": 0.2},
        method="trf",
        max_nfev=2,
    )

    assert result.completion.status == "ok"


@pytest.mark.unit
def test_coerced_non_callable_evaluator_preserves_origin_aware_method() -> None:
    from kindred.core.fitting_evaluation import coerce_fitting_series_evaluator

    calls = []

    class _OriginAwareEvaluateOnly:
        def evaluate_series(self, params):
            calls.append(("plain", dict(params)))
            return {"t": [0.0], "species": {"A": [1.0]}}

        def evaluate_series_with_parameter_origins(self, params, origins=None, *, failed_params=None):
            calls.append(("origin", dict(params), dict(origins or {}), dict(failed_params or {})))
            return {"t": [0.0], "species": {"A": [2.0]}}

    evaluator = coerce_fitting_series_evaluator(_OriginAwareEvaluateOnly())

    result = evaluator.evaluate_series_with_parameter_origins(
        {"k": 0.2},
        {"k": "optimizer_shared"},
        failed_params={"k": 0.2},
    )

    assert calls == [("origin", {"k": 0.2}, {"k": "optimizer_shared"}, {"k": 0.2})]
    assert np.asarray(result.species["A"], dtype=float).tolist() == [2.0]


@pytest.mark.unit
def test_evaluate_fitting_series_normalizes_callable_origin_aware_result() -> None:
    from kindred.core.fitting_evaluation import evaluate_fitting_series
    from kindred.core.simulation_series_payload import SimulationSeriesPayload

    calls = []

    class _CallableOriginAware:
        def __call__(self, params):
            calls.append(("call", dict(params)))
            return {"t": [0.0], "species": {"A": [0.0]}}

        def evaluate_series(self, params):
            calls.append(("plain", dict(params)))
            return {"t": [0.0], "species": {"A": [1.0]}}

        def evaluate_series_with_parameter_origins(self, params, origins=None, *, failed_params=None):
            calls.append(("origin", dict(params), dict(origins or {}), dict(failed_params or {})))
            return {"t": [0.0], "species": {"A": [2.0]}}

    result = evaluate_fitting_series(
        _CallableOriginAware(),
        {"k": 0.2},
        origins={"k": "optimizer_shared"},
        failed_params={"k": 0.2},
    )

    assert calls == [("origin", {"k": 0.2}, {"k": "optimizer_shared"}, {"k": 0.2})]
    assert isinstance(result, SimulationSeriesPayload)
    assert np.asarray(result.species["A"], dtype=float).tolist() == [2.0]
