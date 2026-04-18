import types

import numpy as np
import pytest

from kindred.core.analysis.fit_dataset_payload import FitDatasetSpec
from kindred.core.analysis.dataset_parameter_overrides import FitDatasetParameterOverrides
from kindred.core.analysis.global_fitting import (
    _FitParameterLayout,
    _GlobalFitObjective,
    fit_global,
)
from kindred.core.exceptions import FitSimulationError
from kindred.core.objective import ObjectiveContext
from kindred.core.fitting_evaluation import (
    CallableFittingEvaluator,
    SerialFittingEvaluator,
    coerce_fitting_series_evaluator,
    prepare_fitting_execution_context,
)
import kindred.core.fitting_optimization as fitting_optimization
from kindred.core.simulation_preparation import build_prepared_simulation_func
from kindred.core.simulator.solvers import SimulationOutput


def _build_serial_fit_components():
    mechanism_text = "\n".join(
        [
            "reaction: A -> B; k=0.5",
            "initial: A=1.0",
            "initial: B=0.0",
        ]
    )
    evaluator = SerialFittingEvaluator(
        prepare_fitting_execution_context(
            mechanism_text=mechanism_text,
            param_names=["k1"],
            t_end=1.0,
            num_points=2,
            solver="BDF",
            rtol=1e-6,
            atol=1e-12,
            initial_prefix="init:",
        )
    )
    payload = FitDatasetSpec(
        dataset_id="ds1",
        t_exp=np.asarray([0.0, 1.0], dtype=float),
        species_list=["B"],
        y_matrix=np.zeros((1, 2), dtype=float),
        point_count=2,
        x_name="t",
        x_obs=None,
        x_mode="auto",
    )
    layout = _FitParameterLayout(
        param_names=["k1"],
        shared_log10={},
        dataset_var_order=[],
        dataset_var_index={},
        dataset_var_log10={},
        x0=np.asarray([0.5], dtype=float),
        lower=np.asarray([0.0], dtype=float),
        upper=np.asarray([1.0], dtype=float),
    )
    dataset = {
        "id": "ds1",
        "t": np.asarray([0.0, 1.0], dtype=float),
        "y": np.zeros(2, dtype=float),
        "species": "B",
    }
    return evaluator, payload, layout, dataset


class _OriginAwareEvaluateOnly:
    def __init__(self, evaluator):
        self.evaluator = evaluator
        self.origin_calls = []
        self.plain_calls = []

    def evaluate_series(self, params):
        self.plain_calls.append(dict(params))
        return self.evaluator.evaluate_series(params)

    def evaluate_series_with_parameter_origins(self, params, origins=None, *, failed_params=None):
        self.origin_calls.append((dict(params), dict(origins or {}), dict(failed_params or {})))
        return self.evaluator.evaluate_series_with_parameter_origins(
            params,
            origins,
            failed_params=failed_params,
        )


def test_global_fit_penalty_on_nonfinite_dataset():
    t = np.linspace(0.0, 1.0, 4)
    datasets = [
        {"id": "ok", "t": t, "y": np.zeros_like(t), "species": "A"},
        {"id": "bad", "t": t, "y": np.zeros_like(t), "species": "A"},
    ]

    def _sim(_params):
        return {"t": t, "A": np.array([0.0, np.nan, 0.0, 0.0])}

    result = fit_global(
        _sim,
        datasets,
        shared_params={"k": 0.1},
        method="trf",
        max_nfev=10,
    )

    expected_len = sum(len(ds["t"]) for ds in datasets)
    assert result.objective_residuals is not None
    assert result.objective_residuals.shape == (expected_len,)
    assert np.all(np.isfinite(result.objective_residuals))
    assert result.error_diagnostics is not None
    details = result.error_diagnostics.get("details") if isinstance(result.error_diagnostics, dict) else None
    assert isinstance(details, dict)
    assert details.get("last_error_dataset") == "bad"
    assert "last_error_dataset" not in result.message.lower()


@pytest.mark.unit
def test_global_fit_objective_penalizes_nonfinite_param_without_stale_binding_reuse(monkeypatch):
    mechanism_text = "\n".join(
        [
            "reaction: A -> B; k=0.5",
            "reaction: B -> C; k=0.1",
            "initial: A=1.0",
            "initial: B=0.0",
            "initial: C=0.0",
        ]
    )

    def _binding_sensitive_solver(request):
        dy0 = np.asarray(request.rhs(0.0, np.asarray(request.y0, dtype=float)), dtype=float).reshape(-1)
        b_rate = float(dy0[1])
        t = np.asarray([0.0, float(request.t_span[1])], dtype=float)
        y = np.vstack(
            [
                np.ones_like(t),
                np.full_like(t, b_rate, dtype=float),
                np.zeros_like(t),
            ]
        )
        return SimulationOutput(t=t, Y=y, provenance={"solver": "binding_stub"})

    monkeypatch.setattr("kindred.core.simulator.solvers.solve_ode", _binding_sensitive_solver)

    simulation_func = build_prepared_simulation_func(
        mechanism_text=mechanism_text,
        param_names=["k1", "k2"],
        t_end=1.0,
        num_points=2,
        solver="BDF",
        rtol=1e-6,
        atol=1e-12,
    )
    payload = FitDatasetSpec(
        dataset_id="ds1",
        t_exp=np.asarray([0.0, 1.0], dtype=float),
        species_list=["B"],
        y_matrix=np.zeros((1, 2), dtype=float),
        point_count=2,
        x_name="t",
        x_obs=None,
        x_mode="auto",
    )
    layout = _FitParameterLayout(
        param_names=["k1", "k2"],
        shared_log10={},
        dataset_var_order=[],
        dataset_var_index={},
        dataset_var_log10={},
        x0=np.asarray([0.25, 0.5], dtype=float),
        lower=np.asarray([0.0, 0.0], dtype=float),
        upper=np.asarray([1.0, 1.0], dtype=float),
    )
    objective = _GlobalFitObjective(
        fit_evaluator=CallableFittingEvaluator(simulation_func),
        payloads=[payload],
        shared_params={"k1": 0.25, "k2": 0.5},
        dataset_params={},
        weights={"ds1": 1.0},
        layout=layout,
        penalty_value=1e6,
        ctx=ObjectiveContext(),
        progress_callback=None,
        cancellation_check=None,
    )

    finite_residuals = objective(np.asarray([0.25, 0.5], dtype=float))
    assert np.allclose(finite_residuals, np.asarray([0.25, 0.25], dtype=float))

    nonfinite_residuals = objective(np.asarray([np.nan, 0.5], dtype=float))
    assert np.all(np.isfinite(nonfinite_residuals))
    assert np.allclose(nonfinite_residuals, np.full(2, 1e6, dtype=float))


@pytest.mark.unit
def test_global_fit_objective_penalizes_nonfinite_probe_on_shared_serial_evaluator():
    evaluator, payload, layout, _dataset = _build_serial_fit_components()
    objective = _GlobalFitObjective(
        fit_evaluator=evaluator,
        payloads=[payload],
        shared_params={"k1": 0.5},
        dataset_params={},
        weights={"ds1": 1.0},
        layout=layout,
        penalty_value=1e6,
        ctx=ObjectiveContext(),
        progress_callback=None,
        cancellation_check=None,
    )

    finite_residuals = objective(np.asarray([0.5], dtype=float))
    assert np.all(np.isfinite(finite_residuals))

    nonfinite_residuals = objective(np.asarray([np.nan], dtype=float))
    assert np.all(np.isfinite(nonfinite_residuals))
    assert np.allclose(nonfinite_residuals, np.full(2, 1e6, dtype=float))


@pytest.mark.unit
def test_global_fit_objective_penalizes_nonfinite_probe_through_callable_wrapper():
    evaluator, payload, layout, _dataset = _build_serial_fit_components()

    def _simulate(params):
        return evaluator.evaluate_series(params)

    objective = _GlobalFitObjective(
        fit_evaluator=CallableFittingEvaluator(_simulate),
        payloads=[payload],
        shared_params={"k1": 0.5},
        dataset_params={},
        weights={"ds1": 1.0},
        layout=layout,
        penalty_value=1e6,
        ctx=ObjectiveContext(),
        progress_callback=None,
        cancellation_check=None,
    )

    nonfinite_residuals = objective(np.asarray([np.nan], dtype=float))

    assert np.all(np.isfinite(nonfinite_residuals))
    assert np.allclose(nonfinite_residuals, np.full(2, 1e6, dtype=float))


@pytest.mark.unit
def test_global_fit_objective_penalizes_nonfinite_probe_through_evaluate_series_adapter():
    evaluator, payload, layout, _dataset = _build_serial_fit_components()

    class _EvaluateOnly:
        def evaluate_series(self, params):
            return evaluator.evaluate_series(params)

    objective = _GlobalFitObjective(
        fit_evaluator=coerce_fitting_series_evaluator(_EvaluateOnly()),
        payloads=[payload],
        shared_params={"k1": 0.5},
        dataset_params={},
        weights={"ds1": 1.0},
        layout=layout,
        penalty_value=1e6,
        ctx=ObjectiveContext(),
        progress_callback=None,
        cancellation_check=None,
    )

    nonfinite_residuals = objective(np.asarray([np.nan], dtype=float))

    assert np.all(np.isfinite(nonfinite_residuals))
    assert np.allclose(nonfinite_residuals, np.full(2, 1e6, dtype=float))


@pytest.mark.unit
def test_global_fit_objective_uses_origin_aware_method_on_wrapped_non_callable_optimizer_probe():
    evaluator, payload, layout, _dataset = _build_serial_fit_components()
    wrapped = _OriginAwareEvaluateOnly(evaluator)
    objective = _GlobalFitObjective(
        fit_evaluator=coerce_fitting_series_evaluator(wrapped),
        payloads=[payload],
        shared_params={"k1": 0.5},
        dataset_params={},
        weights={"ds1": 1.0},
        layout=layout,
        penalty_value=1e6,
        ctx=ObjectiveContext(),
        progress_callback=None,
        cancellation_check=None,
    )

    nonfinite_residuals = objective(np.asarray([np.nan], dtype=float))

    assert np.all(np.isfinite(nonfinite_residuals))
    assert np.allclose(nonfinite_residuals, np.full(2, 1e6, dtype=float))
    assert len(wrapped.origin_calls) == 1
    _params, origins, failed_params = wrapped.origin_calls[0]
    assert origins["k1"] == "optimizer_shared"
    assert np.isnan(failed_params["k1"])


@pytest.mark.unit
def test_global_fit_objective_uses_origin_aware_method_on_wrapped_non_callable_configured_param():
    evaluator, payload, layout, _dataset = _build_serial_fit_components()
    wrapped = _OriginAwareEvaluateOnly(evaluator)
    objective = _GlobalFitObjective(
        fit_evaluator=coerce_fitting_series_evaluator(wrapped),
        payloads=[payload],
        shared_params={"k1": 0.5},
        dataset_params={"ds1": {"k1": float("nan")}},
        weights={"ds1": 1.0},
        layout=layout,
        penalty_value=1e6,
        ctx=ObjectiveContext(),
        progress_callback=None,
        cancellation_check=None,
    )

    with pytest.raises(FitSimulationError, match="Non-finite parameter value") as exc_info:
        objective(np.asarray([0.5], dtype=float))

    assert getattr(exc_info.value, "details", {}).get("fatal") is True
    assert len(wrapped.origin_calls) == 1
    _params, origins, failed_params = wrapped.origin_calls[0]
    assert origins["k1"] == "configured_dataset"
    assert np.isnan(failed_params["k1"])


@pytest.mark.unit
def test_global_fit_objective_fails_when_fixed_override_shadows_optimizer_name_with_nonfinite():
    evaluator, payload, layout, _dataset = _build_serial_fit_components()
    objective = _GlobalFitObjective(
        fit_evaluator=evaluator,
        payloads=[payload],
        shared_params={"k1": 0.5},
        dataset_params={"ds1": {"k1": float("nan")}},
        weights={"ds1": 1.0},
        layout=layout,
        penalty_value=1e6,
        ctx=ObjectiveContext(),
        progress_callback=None,
        cancellation_check=None,
    )

    with pytest.raises(FitSimulationError, match="Non-finite parameter value") as exc_info:
        objective(np.asarray([0.5], dtype=float))

    assert getattr(exc_info.value, "details", {}).get("fatal") is True


@pytest.mark.unit
def test_fit_global_shared_serial_evaluator_survives_nonfinite_least_squares_probe(monkeypatch):
    evaluator, _payload, _layout, dataset = _build_serial_fit_components()
    seen: dict[str, np.ndarray] = {}

    def fake_least_squares(fun, x0, **_kwargs):
        seen["nonfinite_residuals"] = np.asarray(fun(np.asarray([np.nan], dtype=float)), dtype=float).reshape(-1)
        x = np.asarray(x0, dtype=float)
        seen["finite_residuals"] = np.asarray(fun(x), dtype=float).reshape(-1)
        return types.SimpleNamespace(
            x=x,
            success=True,
            message="ok",
            nfev=2,
            jac=np.eye(x.size, dtype=float),
            fun=seen["finite_residuals"],
        )

    monkeypatch.setattr(
        fitting_optimization,
        "load_scipy_optimize",
        lambda: (fake_least_squares, lambda *_a, **_k: None),
    )

    result = fit_global(
        evaluator,
        [dataset],
        {"k1": 0.5},
        method="trf",
        max_nfev=5,
    )

    assert np.allclose(seen["nonfinite_residuals"], np.full(2, 1e6, dtype=float))
    assert np.all(np.isfinite(seen["finite_residuals"]))
    assert result.success is True
    assert result.message == "ok"
    assert result.objective_residuals is not None
    assert np.all(np.isfinite(result.objective_residuals))


@pytest.mark.unit
def test_fit_global_shared_serial_evaluator_de_penalizes_nonfinite_probe(monkeypatch):
    evaluator, _payload, _layout, dataset = _build_serial_fit_components()
    seen: dict[str, float] = {}

    def fake_differential_evolution(func, *, bounds, **_kwargs):
        seen["nonfinite_penalty"] = float(func(np.asarray([np.nan], dtype=float)))
        x = np.asarray([(float(lo) + float(hi)) / 2.0 for lo, hi in bounds], dtype=float)
        seen["finite_cost"] = float(func(x))
        return types.SimpleNamespace(x=x, success=True, message="ok", nfev=2)

    monkeypatch.setattr(
        fitting_optimization,
        "load_scipy_optimize",
        lambda: (lambda *_a, **_k: None, fake_differential_evolution),
    )

    result = fit_global(
        evaluator,
        [dataset],
        {"k1": 0.5},
        method="de",
        bounds={"k1": (0.0, 1.0)},
        max_nfev=5,
    )

    assert np.isfinite(seen["nonfinite_penalty"])
    assert seen["nonfinite_penalty"] > seen["finite_cost"]
    assert result.objective_residuals is not None
    assert np.all(np.isfinite(result.objective_residuals))


@pytest.mark.unit
def test_fit_global_shared_serial_evaluator_fails_on_nonfinite_dataset_override(monkeypatch):
    evaluator, _payload, _layout, dataset = _build_serial_fit_components()

    def fake_least_squares(fun, x0, **_kwargs):
        x = np.asarray(x0, dtype=float)
        fun_val = np.asarray(fun(x), dtype=float).reshape(-1)
        return types.SimpleNamespace(
            x=x,
            success=True,
            message="ok",
            nfev=1,
            jac=np.eye(x.size, dtype=float),
            fun=fun_val,
        )

    monkeypatch.setattr(
        fitting_optimization,
        "load_scipy_optimize",
        lambda: (fake_least_squares, lambda *_a, **_k: None),
    )

    result = fit_global(
        evaluator,
        [dataset],
        {"k1": 0.5},
        dataset_overrides=[
            FitDatasetParameterOverrides(dataset_id="ds1", fixed_params={"init:A": float("nan")})
        ],
        method="trf",
        max_nfev=5,
    )

    assert result.success is False
    assert "non-finite parameter value" in result.message.lower()
    assert result.objective_residuals is not None
    assert result.objective_residuals.size == 0


@pytest.mark.unit
def test_fit_global_shared_serial_evaluator_ignores_nonfinite_unconsumed_dataset_overrides(monkeypatch):
    evaluator, _payload, _layout, dataset = _build_serial_fit_components()

    def fake_least_squares(fun, x0, **_kwargs):
        x = np.asarray(x0, dtype=float)
        fun_val = np.asarray(fun(x), dtype=float).reshape(-1)
        return types.SimpleNamespace(
            x=x,
            success=True,
            message="ok",
            nfev=1,
            jac=np.eye(x.size, dtype=float),
            fun=fun_val,
        )

    monkeypatch.setattr(
        fitting_optimization,
        "load_scipy_optimize",
        lambda: (fake_least_squares, lambda *_a, **_k: None),
    )

    result = fit_global(
        evaluator,
        [dataset],
        {"k1": 0.5},
        dataset_overrides=[
            FitDatasetParameterOverrides(
                dataset_id="ds1",
                fixed_params={
                    "init:Removed": float("nan"),
                    "unknown_rate": float("inf"),
                    "arbitrary_extra": float("-inf"),
                },
            )
        ],
        method="trf",
        max_nfev=5,
    )

    assert result.success is True
    assert result.message == "ok"
    assert result.objective_residuals is not None
    assert np.all(np.isfinite(result.objective_residuals))


@pytest.mark.unit
def test_global_fit_raw_callable_fails_on_configured_nonfinite_forwarded_key():
    _evaluator, payload, layout, _dataset = _build_serial_fit_components()

    def _simulate(params):
        t_axis = np.asarray([0.0, 1.0], dtype=float)
        return {"t": t_axis, "B": np.zeros_like(t_axis)}

    objective = _GlobalFitObjective(
        fit_evaluator=CallableFittingEvaluator(_simulate),
        payloads=[payload],
        shared_params={"k1": 0.5},
        dataset_params={"ds1": {"ignored_by_callable": float("nan")}},
        weights={"ds1": 1.0},
        layout=layout,
        penalty_value=1e6,
        ctx=ObjectiveContext(),
        progress_callback=None,
        cancellation_check=None,
    )

    with pytest.raises(FitSimulationError, match="Non-finite parameter value") as exc_info:
        objective(np.asarray([0.5], dtype=float))

    assert getattr(exc_info.value, "details", {}).get("fatal") is True


@pytest.mark.unit
def test_global_fit_evaluate_series_adapter_fails_on_configured_nonfinite_forwarded_key():
    _evaluator, payload, layout, _dataset = _build_serial_fit_components()

    class _EvaluateOnly:
        def evaluate_series(self, params):
            t_axis = np.asarray([0.0, 1.0], dtype=float)
            return {"t": t_axis, "B": np.zeros_like(t_axis)}

    objective = _GlobalFitObjective(
        fit_evaluator=coerce_fitting_series_evaluator(_EvaluateOnly()),
        payloads=[payload],
        shared_params={"k1": 0.5},
        dataset_params={"ds1": {"ignored_by_adapter": float("nan")}},
        weights={"ds1": 1.0},
        layout=layout,
        penalty_value=1e6,
        ctx=ObjectiveContext(),
        progress_callback=None,
        cancellation_check=None,
    )

    with pytest.raises(FitSimulationError, match="Non-finite parameter value") as exc_info:
        objective(np.asarray([0.5], dtype=float))

    assert getattr(exc_info.value, "details", {}).get("fatal") is True
