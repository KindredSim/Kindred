from __future__ import annotations

import numpy as np
import pytest

from kindred.core.analysis.fit_dataset_payload import FitDatasetSpec
from kindred.core.analysis.global_fit_execution import (
    GlobalFitObjective,
    build_completion_detail_sections,
    build_parameter_layout,
    completion_result_message,
    normalize_weights,
)
from kindred.core.analysis.global_fitting import fit_global
from kindred.core.fitting_completion import FitDiagnostic, GlobalFitCompletion
from kindred.core.fitting_evaluation import CallableFittingEvaluator
from kindred.core.objective import ObjectiveContext
from kindred.core.exceptions import FitSimulationError
from kindred.core.simulation_failure import build_simulation_failure
from kindred.core.simulation_preparation import (
    SimulationPreparationError,
    _fit_simulation_error_from_preparation_error,
)


pytestmark = [pytest.mark.unit]


def _make_payload(dataset_id: str, *, point_count: int = 3) -> FitDatasetSpec:
    t_exp = np.linspace(0.0, 1.0, point_count, dtype=float)
    return FitDatasetSpec(
        dataset_id=dataset_id,
        t_exp=t_exp,
        species_list=["A"],
        y_matrix=np.zeros((1, point_count), dtype=float),
        point_count=point_count,
        x_name="t",
        x_obs=None,
        x_mode="auto",
    )


def testnormalize_weights_rejects_unknown_dataset_ids() -> None:
    payloads = [_make_payload("ds1"), _make_payload("ds2")]

    with pytest.raises(ValueError, match="ghost"):
        normalize_weights(payloads, {"ds1": 2.0, "ghost": 1.0})


def test_failed_param_snapshot_namespaces_dataset_specific_values() -> None:
    snapshot = GlobalFitObjective.build_failed_param_snapshot(
        ds_id="ds2",
        shared_params={"k": 0.5},
        full_params={"k": 0.5, "init:A": 2.25, "offset": 7.0},
    )

    assert snapshot == {
        "k": pytest.approx(0.5),
        "ds2::init:A": pytest.approx(2.25),
        "ds2::offset": pytest.approx(7.0),
    }


def test_failed_result_preserves_namespaced_dataset_variable_params(monkeypatch) -> None:
    from kindred.core.analysis import global_fitting

    def _fake_fit_parameters(*_args, **_kwargs):
        raise FitSimulationError(
            "fatal failure",
            failed_params={
                "k": 0.75,
                "ds1::init:A": 1.25,
                "ds2::init:A": 2.5,
            },
        )

    monkeypatch.setattr(global_fitting, "fit_parameters", _fake_fit_parameters)

    result = global_fitting.fit_global(
        lambda _params: {"t": np.asarray([0.0, 1.0], dtype=float), "A": np.asarray([1.0, 1.0], dtype=float)},
        datasets=[
            {"id": "ds1", "t": np.asarray([0.0, 1.0], dtype=float), "y": np.asarray([1.0, 0.8], dtype=float), "species": "A"},
            {"id": "ds2", "t": np.asarray([0.0, 1.0], dtype=float), "y": np.asarray([1.0, 0.6], dtype=float), "species": "A"},
        ],
        shared_params={"k": 0.2},
        dataset_variable_params={
            "ds1": {"init:A": {"initial": 0.3, "min": 0.1, "max": 3.0}},
            "ds2": {"init:A": {"initial": 0.4, "min": 0.1, "max": 4.0}},
        },
        max_nfev=1,
    )

    assert result.completion.status == "fail"
    assert result.shared_params["k"] == pytest.approx(0.75)
    assert result.dataset_params["ds1"]["init:A"] == pytest.approx(1.25)
    assert result.dataset_params["ds2"]["init:A"] == pytest.approx(2.5)


def test_failed_result_preserves_preparation_failure_semantics_and_remediation(monkeypatch) -> None:
    from kindred.core.analysis import global_fitting

    def _fake_fit_parameters(*_args, **_kwargs):
        raise _fit_simulation_error_from_preparation_error(
            SimulationPreparationError("parameter_algebra", "undefined symbol k_total")
        )

    monkeypatch.setattr(global_fitting, "fit_parameters", _fake_fit_parameters)

    result = global_fitting.fit_global(
        lambda _params: {"t": np.asarray([0.0, 1.0], dtype=float), "A": np.asarray([1.0, 1.0], dtype=float)},
        datasets=[
            {"id": "ds1", "t": np.asarray([0.0, 1.0], dtype=float), "y": np.asarray([1.0, 0.8], dtype=float), "species": "A"},
        ],
        shared_params={"k": 0.2},
        max_nfev=1,
    )

    assert result.completion.status == "fail"
    assert result.completion.nonfinite_metrics is False
    assert result.completion.optimizer_diagnostic is not None
    assert result.completion.optimizer_diagnostic.failure["kind"] == "preparation_error"
    assert result.completion.optimizer_diagnostic.failure["details"]["stage"] == "parameter_algebra"
    assert result.completion.optimizer_diagnostic.remediation == "preparation"


def test_global_fit_fail_message_overrides_optimizer_success_text_when_final_replay_fails() -> None:
    t = np.linspace(0.0, 1.0, 4, dtype=float)

    def _sim(_params):
        return {"t": t, "A": np.asarray([0.0, np.nan, 0.0, 0.0], dtype=float)}

    result = fit_global(
        _sim,
        datasets=[
            {"id": "ds1", "t": t.copy(), "y": np.zeros_like(t), "species": "A"},
        ],
        shared_params={"k": 0.1},
        method="trf",
        max_nfev=10,
    )

    assert result.completion.status == "fail"
    assert result.message != "Optimization terminated successfully."
    assert "failed" in result.message.lower()


def test_transient_optimizer_error_does_not_leave_false_warn_completion(monkeypatch) -> None:
    from kindred.core.analysis import global_fitting
    from kindred.core.fitting_optimization import FitResult

    def _fake_fit_parameters(objective_func, initial_params, **_kwargs):
        objective_func(np.asarray([float("nan")], dtype=float))
        return FitResult(
            success=True,
            parameters=dict(initial_params),
            uncertainties=None,
            chi_squared=0.0,
            r_squared=1.0,
            residuals=np.zeros(4, dtype=float),
            nfev=1,
            message="Optimization terminated successfully.",
            covariance=None,
        )

    def _sim(params):
        t = np.asarray([0.0, 1.0, 2.0, 3.0], dtype=float)
        k = float(params["k"])
        if not np.isfinite(k):
            return {"t": t, "A": np.asarray([0.0, np.nan, 0.0, 0.0], dtype=float)}
        return {"t": t, "A": np.zeros_like(t)}

    monkeypatch.setattr(global_fitting, "fit_parameters", _fake_fit_parameters)

    result = global_fitting.fit_global(
        _sim,
        datasets=[
            {"id": "ds1", "t": np.asarray([0.0, 1.0, 2.0, 3.0], dtype=float), "y": np.zeros(4, dtype=float), "species": "A"},
        ],
        shared_params={"k": 0.1},
        method="trf",
        max_nfev=3,
    )

    assert result.completion.status == "ok"
    assert result.completion.optimizer_diagnostic is None
    assert result.message == "Optimization terminated successfully."


def test_warn_message_is_not_success_like_when_warning_source_is_real(monkeypatch) -> None:
    from kindred.core.analysis import global_fitting
    from kindred.core.fitting_optimization import FitResult

    def _fake_fit_parameters(objective_func, initial_params, **_kwargs):
        objective_func(np.asarray([float("nan")], dtype=float))
        return FitResult(
            success=False,
            parameters=dict(initial_params),
            uncertainties=None,
            chi_squared=0.0,
            r_squared=1.0,
            residuals=np.zeros(4, dtype=float),
            nfev=1,
            message="Optimization terminated successfully.",
            covariance=None,
        )

    def _sim(params):
        t = np.asarray([0.0, 1.0, 2.0, 3.0], dtype=float)
        k = float(params["k"])
        if not np.isfinite(k):
            return {"t": t, "A": np.asarray([0.0, np.nan, 0.0, 0.0], dtype=float)}
        return {"t": t, "A": np.zeros_like(t)}

    monkeypatch.setattr(global_fitting, "fit_parameters", _fake_fit_parameters)

    result = global_fitting.fit_global(
        _sim,
        datasets=[
            {"id": "ds1", "t": np.asarray([0.0, 1.0, 2.0, 3.0], dtype=float), "y": np.zeros(4, dtype=float), "species": "A"},
        ],
        shared_params={"k": 0.1},
        method="trf",
        max_nfev=3,
    )

    assert result.completion.status == "warn"
    assert result.completion.optimizer_diagnostic is not None
    assert result.message != "Optimization terminated successfully."
    assert "warning" in result.message.lower()


def test_transient_optimizer_error_does_not_leak_into_real_warning_completion(monkeypatch) -> None:
    from kindred.core.analysis import global_fitting
    from kindred.core.fitting_optimization import FitResult

    def _fake_fit_parameters(objective_func, initial_params, **_kwargs):
        objective_func(np.asarray([float("nan")], dtype=float))
        return FitResult(
            success=True,
            parameters=dict(initial_params),
            uncertainties=None,
            chi_squared=0.0,
            r_squared=1.0,
            residuals=np.zeros(6, dtype=float),
            nfev=1,
            message="Optimization terminated successfully.",
            covariance=None,
        )

    def _sim(params):
        t_sim = np.linspace(0.0, 1.0, 101, dtype=float)
        k = float(params["k"])
        if not np.isfinite(k):
            return {"t": t_sim, "species": {"X": t_sim.copy(), "Y": np.full_like(t_sim, np.nan)}}
        return {"t": t_sim, "species": {"X": t_sim.copy(), "Y": np.zeros_like(t_sim)}}

    monkeypatch.setattr(global_fitting, "fit_parameters", _fake_fit_parameters)

    t_obs = np.linspace(0.0, 1.0, 6, dtype=float)
    x_obs = np.full_like(t_obs, 10.0, dtype=float)
    result = global_fitting.fit_global(
        _sim,
        datasets=[
            {
                "id": "ds1",
                "t": t_obs.copy(),
                "y": np.zeros((1, t_obs.size), dtype=float),
                "species": ["Y"],
                "x_name": "X",
                "x_obs": x_obs.copy(),
                "x_mapping_mode": "auto",
            }
        ],
        shared_params={"k": 0.1},
        method="trf",
        max_nfev=3,
    )

    assert result.completion.status == "warn"
    assert result.completion.optimizer_diagnostic is None
    assert result.completion.dataset_warnings
    assert "warning" in result.message.lower()
    assert "non-finite parameter value" not in result.message.lower()


def testbuild_completion_detail_sections_keeps_top_level_failure_without_stack_trace_when_dataset_failures_exist() -> None:
    sections = build_completion_detail_sections(
        status="fail",
        optimizer_diagnostic=FitDiagnostic(
            phase="fatal",
            dataset_id=None,
            failure=build_simulation_failure(kind="simulation_error", message="top-level process-pool failure"),
            remediation="generic_retry",
        ),
        dataset_failures={
            "ds1": FitDiagnostic(
                phase="final_replay",
                dataset_id="ds1",
                failure=build_simulation_failure(kind="simulation_error", message="dataset replay failed"),
                remediation="generic_retry",
            )
        },
    )

    assert len(sections) == 1
    assert sections[0].dataset_id is None
    assert sections[0].failure["message"] == "top-level process-pool failure"


def test_nonfinite_metrics_message_takes_precedence_over_stale_optimizer_diagnostic() -> None:
    diagnostic = FitDiagnostic(
        phase="optimizer",
        dataset_id="ds1",
        failure=build_simulation_failure(kind="simulation_error", message="stale alignment warning"),
    )
    completion = GlobalFitCompletion(
        status="fail",
        optimizer_converged=True,
        nonfinite_metrics=True,
        optimizer_diagnostic=diagnostic,
        dataset_failures={},
        dataset_warnings={},
        detail_sections=[],
    )

    message = completion_result_message(
        base_message="Optimization terminated successfully.",
        completion=completion,
        optimizer_diagnostic=diagnostic,
        dataset_failures={},
        dataset_warnings={},
    )

    assert message == "Final χ² is non-finite; results are invalid."


def test_warn_message_preserves_non_success_optimizer_termination_reason() -> None:
    completion = GlobalFitCompletion(
        status="warn",
        optimizer_converged=False,
        nonfinite_metrics=False,
        optimizer_diagnostic=None,
        dataset_failures={},
        dataset_warnings={},
        detail_sections=[],
    )

    message = completion_result_message(
        base_message="Maximum number of function evaluations exceeded.",
        completion=completion,
        optimizer_diagnostic=None,
        dataset_failures={},
        dataset_warnings={},
    )

    assert message == "Global fit completed with warnings: Maximum number of function evaluations exceeded."


def test_global_chi_squared_uses_full_objective_residuals_for_parametric_x_penalties(monkeypatch) -> None:
    from kindred.core.analysis import global_fitting
    import kindred.core.fitting_optimization as fitting_optimization

    t_obs = np.linspace(0.0, 1.0, 6, dtype=float)
    x_obs = np.full_like(t_obs, 10.0, dtype=float)
    y_obs = np.linspace(0.0, 2.0, t_obs.size, dtype=float).reshape(1, -1)

    datasets = [
        {
            "id": "ds1",
            "t": t_obs.copy(),
            "y": y_obs.copy(),
            "species": ["Y"],
            "x_name": "X",
            "x_obs": x_obs.copy(),
            "x_mapping_mode": "auto",
        }
    ]

    t_sim = np.linspace(0.0, 1.0, 101, dtype=float)

    def simulation_func(_params):
        return {"t": t_sim.copy(), "species": {"X": t_sim.copy(), "Y": np.zeros_like(t_sim)}}

    def _fake_least_squares(fun, x0, **_kwargs):
        residuals = np.asarray(fun(np.asarray(x0, dtype=float)), dtype=float).reshape(-1)

        class _Result:
            pass

        result = _Result()
        result.x = np.asarray(x0, dtype=float)
        result.success = True
        result.message = "fake"
        result.nfev = 1
        result.fun = residuals
        result.jac = np.zeros((residuals.size, result.x.size), dtype=float)
        return result

    monkeypatch.setattr(
        fitting_optimization,
        "load_scipy_optimize",
        lambda: (_fake_least_squares, lambda *_a, **_k: None),
    )

    result = global_fitting.fit_global(
        simulation_func,
        datasets=datasets,
        shared_params={"k1": 0.2},
        weights=None,
        method="trf",
        max_nfev=1,
    )

    residuals = np.asarray(result.objective_residuals, dtype=float).reshape(-1)
    y_flat = y_obs.reshape(-1)
    expected_chi_squared = float(np.mean(residuals**2))
    expected_r_squared = 1.0 - float(np.sum(residuals**2)) / float(np.sum((y_flat - float(np.mean(y_flat))) ** 2))

    assert residuals.size == 2 * t_obs.size
    assert result.global_chi_squared == pytest.approx(expected_chi_squared)
    assert result.global_r_squared == pytest.approx(expected_r_squared)


def test_global_fit_result_dataset_diagnostics_remain_raw_under_target_weighting() -> None:
    t_obs = np.linspace(0.0, 1.0, 5, dtype=float)

    def simulation_func(_params):
        return {
            "t": t_obs.copy(),
            "species": {
                "A": np.zeros_like(t_obs),
                "B": np.zeros_like(t_obs),
            },
        }

    result = fit_global(
        simulation_func,
        datasets=[
            {
                "id": "ds1",
                "t": t_obs.copy(),
                "y": np.vstack([np.ones_like(t_obs), 2.0 * np.ones_like(t_obs)]),
                "species": ["A", "B"],
                "target_weights": {"A": 5.0, "B": 1.0},
            }
        ],
        shared_params={"k": 0.2},
        max_nfev=1,
    )

    info = result.dataset_info[0]
    np.testing.assert_allclose(
        result.residual_series["ds1"]["A"],
        -np.ones_like(t_obs),
    )
    np.testing.assert_allclose(
        result.residual_series["ds1"]["B"],
        np.full_like(t_obs, -2.0),
    )
    np.testing.assert_allclose(info.residuals[: t_obs.size], -np.ones_like(t_obs))
    np.testing.assert_allclose(info.residuals[t_obs.size :], np.full_like(t_obs, -2.0))
    assert float(np.sum(info.residuals**2)) == pytest.approx(25.0)
    assert info.chi_squared == pytest.approx(2.5)
    assert info.rmse == pytest.approx(np.sqrt(2.5))
    assert info.mae == pytest.approx(1.5)
    assert result.objective_residuals is not None
    assert float(np.sum(result.objective_residuals**2)) == pytest.approx(15.0)
    assert result.global_chi_squared == pytest.approx(1.5)


def test_global_fit_result_missing_species_failure_diagnostics_remain_raw_under_target_weighting() -> None:
    t_obs = np.linspace(0.0, 1.0, 5, dtype=float)
    penalty_value = 1e6
    weight_a = np.sqrt(2.0 * 5.0 / 6.0)
    weight_b = np.sqrt(2.0 * 1.0 / 6.0)

    def simulation_func(_params):
        return {
            "t": t_obs.copy(),
            "species": {
                "A": np.zeros_like(t_obs),
            },
        }

    result = fit_global(
        simulation_func,
        datasets=[
            {
                "id": "ds1",
                "t": t_obs.copy(),
                "y": np.vstack([np.ones_like(t_obs), 2.0 * np.ones_like(t_obs)]),
                "species": ["A", "B"],
                "target_weights": {"A": 5.0, "B": 1.0},
            }
        ],
        shared_params={"k": 0.2},
        weights={"ds1": 1.0},
        max_nfev=1,
    )

    info = result.dataset_info[0]
    expected_raw_penalty = np.full_like(t_obs, penalty_value)

    np.testing.assert_allclose(result.residual_series["ds1"]["A"], -np.ones_like(t_obs))
    np.testing.assert_allclose(result.residual_series["ds1"]["B"], expected_raw_penalty)
    np.testing.assert_allclose(info.residuals[: t_obs.size], -np.ones_like(t_obs))
    np.testing.assert_allclose(info.residuals[t_obs.size :], expected_raw_penalty)
    assert info.chi_squared == pytest.approx((5.0 + 5.0 * penalty_value**2) / 10.0)
    assert info.rmse == pytest.approx(np.sqrt((5.0 + 5.0 * penalty_value**2) / 10.0))
    assert info.mae == pytest.approx((5.0 + 5.0 * penalty_value) / 10.0)

    assert result.objective_residuals is not None
    np.testing.assert_allclose(result.objective_residuals[: t_obs.size], -weight_a * np.ones_like(t_obs))
    np.testing.assert_allclose(
        result.objective_residuals[t_obs.size :],
        np.full_like(t_obs, weight_b * penalty_value),
    )
    assert result.global_chi_squared == pytest.approx(
        float(np.mean(np.asarray(result.objective_residuals, dtype=float) ** 2))
    )


def test_global_fit_result_nonfinite_series_failure_diagnostics_remain_raw_under_target_weighting() -> None:
    t_obs = np.linspace(0.0, 1.0, 5, dtype=float)
    penalty_value = 1e6
    weight_a = np.sqrt(2.0 * 5.0 / 6.0)
    weight_b = np.sqrt(2.0 * 1.0 / 6.0)

    def simulation_func(_params):
        return {
            "t": t_obs.copy(),
            "species": {
                "A": np.zeros_like(t_obs),
                "B": np.full_like(t_obs, np.nan),
            },
        }

    result = fit_global(
        simulation_func,
        datasets=[
            {
                "id": "ds1",
                "t": t_obs.copy(),
                "y": np.vstack([np.ones_like(t_obs), 2.0 * np.ones_like(t_obs)]),
                "species": ["A", "B"],
                "target_weights": {"A": 5.0, "B": 1.0},
            }
        ],
        shared_params={"k": 0.2},
        weights={"ds1": 1.0},
        max_nfev=1,
    )

    info = result.dataset_info[0]
    expected_raw_penalty = np.full_like(t_obs, penalty_value)

    np.testing.assert_allclose(result.residual_series["ds1"]["A"], -np.ones_like(t_obs))
    np.testing.assert_allclose(result.residual_series["ds1"]["B"], expected_raw_penalty)
    np.testing.assert_allclose(info.residuals[: t_obs.size], -np.ones_like(t_obs))
    np.testing.assert_allclose(info.residuals[t_obs.size :], expected_raw_penalty)
    assert info.chi_squared == pytest.approx((5.0 + 5.0 * penalty_value**2) / 10.0)
    assert info.rmse == pytest.approx(np.sqrt((5.0 + 5.0 * penalty_value**2) / 10.0))
    assert info.mae == pytest.approx((5.0 + 5.0 * penalty_value) / 10.0)

    assert result.objective_residuals is not None
    np.testing.assert_allclose(result.objective_residuals[: t_obs.size], -weight_a * np.ones_like(t_obs))
    np.testing.assert_allclose(
        result.objective_residuals[t_obs.size :],
        np.full_like(t_obs, weight_b * penalty_value),
    )
    assert result.global_chi_squared == pytest.approx(
        float(np.mean(np.asarray(result.objective_residuals, dtype=float) ** 2))
    )


def test_global_fit_objective_normalizes_missing_target_penalty_within_dataset_weight_scale() -> None:
    t_obs = np.linspace(0.0, 1.0, 4, dtype=float)
    payload = FitDatasetSpec(
        dataset_id="ds1",
        t_exp=t_obs,
        species_list=["A", "B"],
        y_matrix=np.zeros((2, t_obs.size), dtype=float),
        point_count=int(2 * t_obs.size),
        x_name="t",
        x_obs=None,
        x_mode="auto",
        target_weights={"A": 1.0, "B": 3.0},
    )
    layout = build_parameter_layout(
        payloads=[payload],
        shared_params={"k": 0.5},
        dataset_variable_params={},
        bounds=None,
        log10_params=None,
    )
    objective = GlobalFitObjective(
        fit_evaluator=CallableFittingEvaluator(lambda _params: {"t": t_obs.copy(), "species": {}}),
        payloads=[payload],
        shared_params={"k": 0.5},
        dataset_params={},
        weights={"ds1": 2.0},
        layout=layout,
        penalty_value=10.0,
        ctx=ObjectiveContext(),
        progress_callback=None,
        cancellation_check=None,
    )

    residuals = objective(layout.x0.copy())

    expected_scales = np.sqrt(np.asarray([2.0 * 1.0 / 4.0, 2.0 * 3.0 / 4.0], dtype=float))
    np.testing.assert_allclose(
        residuals[: t_obs.size],
        np.full_like(t_obs, 2.0 * expected_scales[0] * 10.0),
    )
    np.testing.assert_allclose(
        residuals[t_obs.size :],
        np.full_like(t_obs, 2.0 * expected_scales[1] * 10.0),
    )


def test_global_fit_objective_timeout_fit_simulation_error_uses_penalty_residuals() -> None:
    t_obs = np.linspace(0.0, 1.0, 4, dtype=float)
    payload = FitDatasetSpec(
        dataset_id="ds-timeout",
        t_exp=t_obs,
        species_list=["A"],
        y_matrix=np.zeros((1, t_obs.size), dtype=float),
        point_count=int(t_obs.size),
        x_name="t",
        x_obs=None,
        x_mode="auto",
    )
    layout = build_parameter_layout(
        payloads=[payload],
        shared_params={"k": 0.5},
        dataset_variable_params={},
        bounds=None,
        log10_params=None,
    )

    def _timeout(_params):
        raise FitSimulationError(
            "Fitting simulation failed: Simulation timed out after 0.2 seconds.",
            details={
                "fatal": False,
                "failure": build_simulation_failure(
                    "timeout",
                    "Simulation timed out after 0.2 seconds.",
                    details={"walltime_s": 0.2},
                ),
            },
        )

    objective = GlobalFitObjective(
        fit_evaluator=CallableFittingEvaluator(_timeout),
        payloads=[payload],
        shared_params={"k": 0.5},
        dataset_params={},
        weights={"ds-timeout": 1.0},
        layout=layout,
        penalty_value=10.0,
        ctx=ObjectiveContext(),
        progress_callback=None,
        cancellation_check=None,
    )

    residuals = objective(layout.x0.copy())

    np.testing.assert_allclose(residuals, np.full_like(t_obs, 10.0))
    assert isinstance(objective._ctx.last_error, FitSimulationError)
    assert objective._ctx.last_error.details["fatal"] is False


def test_final_replay_timeout_failure_is_scoped_to_failed_dataset(monkeypatch) -> None:
    import kindred.core.analysis.global_fit_execution as global_fit_execution
    from kindred.core.analysis import global_fitting
    from kindred.core.fitting_optimization import FitResult

    t_obs = np.linspace(0.0, 1.0, 4, dtype=float)

    def _fake_fit_parameters(*_args, **_kwargs):
        return FitResult(
            success=True,
            parameters={"k": 0.5},
            uncertainties=None,
            chi_squared=0.0,
            r_squared=1.0,
            residuals=np.zeros(t_obs.size * 2, dtype=float),
            nfev=1,
            message="Optimization terminated successfully.",
            covariance=None,
        )

    calls = {"n": 0}

    def _evaluate_final_replay(_evaluator, _params, **_kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise FitSimulationError(
                "Fitting simulation failed: Simulation timed out after 0.2 seconds.",
                details={
                    "fatal": False,
                    "failure": build_simulation_failure(
                        "timeout",
                        "Simulation timed out after 0.2 seconds.",
                        details={"walltime_s": 0.2},
                    ),
                },
            )
        return {"t": t_obs.copy(), "species": {"A": np.zeros_like(t_obs)}}

    monkeypatch.setattr(global_fitting, "fit_parameters", _fake_fit_parameters)
    monkeypatch.setattr(global_fit_execution, "evaluate_fitting_series", _evaluate_final_replay)

    result = global_fitting.fit_global(
        lambda _params: {"t": t_obs.copy(), "A": np.zeros_like(t_obs)},
        datasets=[
            {"id": "ds-ok", "t": t_obs.copy(), "y": np.zeros_like(t_obs), "species": "A"},
            {"id": "ds-timeout", "t": t_obs.copy(), "y": np.zeros_like(t_obs), "species": "A"},
        ],
        shared_params={"k": 0.2},
        method="trf",
        max_nfev=1,
    )

    assert result.completion.status == "fail"
    assert set(result.completion.dataset_failures) == {"ds-timeout"}
    assert result.completion.dataset_failures["ds-timeout"].failure["kind"] == "timeout"
    assert "ds-ok" in result.model_series
    assert "ds-timeout" not in result.model_series


def test_global_fit_objective_rebalances_targets_without_raw_cross_dataset_inflation_under_equal_baseline_residuals() -> None:
    t_obs = np.linspace(0.0, 1.0, 4, dtype=float)
    ds1_y = np.linspace(1.0, 2.0, t_obs.size, dtype=float)
    ds2_y = np.linspace(0.5, 1.5, t_obs.size, dtype=float)

    def _make_ds1_payload(target_weights: dict[str, float]) -> FitDatasetSpec:
        return FitDatasetSpec(
            dataset_id="ds1",
            t_exp=t_obs,
            species_list=["A", "B"],
            y_matrix=np.vstack([ds1_y, ds1_y]),
            point_count=int(2 * t_obs.size),
            x_name="t",
            x_obs=None,
            x_mode="auto",
            target_weights=dict(target_weights),
        )

    ds2_payload = FitDatasetSpec(
        dataset_id="ds2",
        t_exp=t_obs,
        species_list=["C"],
        y_matrix=ds2_y.reshape(1, -1),
        point_count=int(t_obs.size),
        x_name="t",
        x_obs=None,
        x_mode="auto",
        target_weights={"C": 1.0},
    )

    def simulation_func(_params):
        return {
            "t": t_obs.copy(),
            "species": {
                "A": np.zeros_like(t_obs),
                "B": np.zeros_like(t_obs),
                "C": np.zeros_like(t_obs),
            },
        }

    residual_vectors = []
    for target_weights in ({"A": 1.0, "B": 1.0}, {"A": 5.0, "B": 1.0}):
        payloads = [_make_ds1_payload(target_weights), ds2_payload]
        layout = build_parameter_layout(
            payloads=payloads,
            shared_params={"k": 0.5},
            dataset_variable_params={},
            bounds=None,
            log10_params=None,
        )
        objective = GlobalFitObjective(
            fit_evaluator=CallableFittingEvaluator(simulation_func),
            payloads=payloads,
            shared_params={"k": 0.5},
            dataset_params={},
            weights={"ds1": 2.0, "ds2": 3.0},
            layout=layout,
            penalty_value=10.0,
            ctx=ObjectiveContext(),
            progress_callback=None,
            cancellation_check=None,
        )
        residual_vectors.append(objective(layout.x0.copy()))

    residuals_equal, residuals_reweighted = residual_vectors
    n_obs = t_obs.size
    ds1_a_equal = residuals_equal[:n_obs]
    ds1_b_equal = residuals_equal[n_obs : 2 * n_obs]
    ds2_equal = residuals_equal[2 * n_obs :]
    ds1_a_reweighted = residuals_reweighted[:n_obs]
    ds1_b_reweighted = residuals_reweighted[n_obs : 2 * n_obs]
    ds2_reweighted = residuals_reweighted[2 * n_obs :]

    assert float(np.sum(ds1_a_reweighted**2)) > float(np.sum(ds1_a_equal**2))
    assert float(np.sum(ds1_b_reweighted**2)) < float(np.sum(ds1_b_equal**2))
    assert float(np.sum(residuals_reweighted[: 2 * n_obs] ** 2)) == pytest.approx(
        float(np.sum(residuals_equal[: 2 * n_obs] ** 2))
    )
    assert float(np.sum(ds2_reweighted**2)) == pytest.approx(float(np.sum(ds2_equal**2)))


def test_global_fit_objective_keeps_dx_penalty_dataset_weighted_only_when_target_weight_changes() -> None:
    t_obs = np.linspace(0.0, 1.0, 6, dtype=float)
    x_obs = np.full_like(t_obs, 10.0, dtype=float)
    y_obs = np.linspace(0.0, 2.0, t_obs.size, dtype=float)

    def _make_payload(target_weight: float) -> FitDatasetSpec:
        return FitDatasetSpec(
            dataset_id="ds1",
            t_exp=t_obs,
            species_list=["Y"],
            y_matrix=y_obs.reshape(1, -1),
            point_count=int(t_obs.size),
            x_name="X",
            x_obs=x_obs.copy(),
            x_mode="time_guided",
            target_weights={"Y": target_weight},
        )

    def simulation_func(_params):
        t_sim = np.linspace(0.0, 1.0, 101, dtype=float)
        return {"t": t_sim.copy(), "species": {"X": t_sim.copy(), "Y": np.zeros_like(t_sim)}}

    payload_1 = _make_payload(1.0)
    payload_5 = _make_payload(5.0)
    residuals = []
    for payload in (payload_1, payload_5):
        layout = build_parameter_layout(
            payloads=[payload],
            shared_params={"k": 0.5},
            dataset_variable_params={},
            bounds=None,
            log10_params=None,
        )
        objective = GlobalFitObjective(
            fit_evaluator=CallableFittingEvaluator(simulation_func),
            payloads=[payload],
            shared_params={"k": 0.5},
            dataset_params={},
            weights={"ds1": 2.0},
            layout=layout,
            penalty_value=10.0,
            ctx=ObjectiveContext(),
            progress_callback=None,
            cancellation_check=None,
        )
        residuals.append(objective(layout.x0.copy()))

    residuals_1, residuals_5 = residuals
    np.testing.assert_allclose(residuals_5[: t_obs.size], residuals_1[: t_obs.size])
    np.testing.assert_allclose(residuals_5[t_obs.size :], residuals_1[t_obs.size :])
