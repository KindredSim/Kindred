from __future__ import annotations

import inspect

import numpy as np
import pytest

from kindred.core.exceptions import ErrorContext, FitSimulationError
from kindred.core.fitting_completion import FitDetailSection, FitDiagnostic, GlobalFitCompletion
from kindred.core.fitting_optimization import FitResult
from kindred.core.simulation_failure import build_simulation_failure

pytestmark = pytest.mark.unit



def _payload(dataset_id: str, y_values) -> object:
    from kindred.core.analysis.fit_dataset_payload import FitDatasetSpec

    y = np.asarray(y_values, dtype=float).reshape(1, -1)
    return FitDatasetSpec(
        dataset_id=str(dataset_id),
        t_exp=np.linspace(0.0, 1.0, y.shape[1]),
        species_list=["A"],
        y_matrix=y,
        point_count=int(y.size),
        x_name="t",
        x_obs=None,
        x_mode="auto",
        target_weights={},
    )


def _raw_dataset(dataset_id: str, y_values) -> dict[str, object]:
    y = np.asarray(y_values, dtype=float).reshape(-1)
    return {
        "id": str(dataset_id),
        "t": np.linspace(0.0, 1.0, y.size),
        "species": "A",
        "y": y,
    }


def _dataset_input(index: int, dataset_id: str, init_a: float):
    from kindred.core.analysis.global_fitting import _ObjectiveDatasetInput

    return _ObjectiveDatasetInput(
        index=int(index),
        payload=_payload(dataset_id, [0.0, 0.0]),
        full_params={"init:A": float(init_a)},
        parameter_origins={},
        failed_param_snapshot={"init:A": float(init_a)},
    )


def test_serial_only_public_api_contract() -> None:
    from kindred.core.api.fitting import fit_global

    api_signature = inspect.signature(fit_global)
    removed_names = {"parallel_enabled", "max_parallel_workers", "limit_blas_threads"}
    assert removed_names.isdisjoint(api_signature.parameters)

    t = np.linspace(0.0, 1.0, 5)

    def _sim(params):
        value = float(dict(params).get("k", 1.0))
        return {"t": t.copy(), "A": np.full_like(t, value, dtype=float)}

    result = fit_global(
        _sim,
        [_raw_dataset("ds1", np.ones_like(t)), _raw_dataset("ds2", np.ones_like(t))],
        shared_params={"k": 1.0},
        bounds={"k": (0.1, 2.0)},
        method="trf",
        max_nfev=5,
    )

    assert result.completion.status == "ok"
    assert {info.dataset_id for info in result.dataset_info} == {"ds1", "ds2"}

    for removed_name, removed_values in (
        ("parallel_enabled", (True, False)),
        ("max_parallel_workers", (0, 1, None)),
        ("limit_blas_threads", (True, False)),
    ):
        for removed_value in removed_values:
            with pytest.raises(TypeError, match=removed_name):
                fit_global(
                    _sim,
                    [_raw_dataset("ds1", np.ones_like(t))],
                    shared_params={"k": 1.0},
                    **{removed_name: removed_value},
                )


def test_fit_global_final_replay_executes_parameterized_intervention_schedule(monkeypatch) -> None:
    import kindred.core.analysis.global_fitting as global_fitting
    from kindred.core.fitting_evaluation import SerialFittingEvaluator, prepare_fitting_execution_context

    mechanism_text = "\n".join(
        [
            "reaction: A -> B; k=0",
            "initial: A=1.0",
            "initial: B=0.0",
            "intervention: op=add; species=A; time=1.0; amount_param=dose",
        ]
    )
    context = prepare_fitting_execution_context(
        mechanism_text=mechanism_text,
        param_names=["dose"],
        t_end=2.0,
        num_points=3,
        solver="BDF",
        rtol=1e-6,
        atol=1e-12,
        initial_prefix="init:",
    )
    evaluator = SerialFittingEvaluator(context)

    def _fit_parameters(_objective, _initial_params, **_kwargs):
        return FitResult(
            success=True,
            parameters={"dose": 3.0},
            uncertainties=None,
            chi_squared=0.0,
            r_squared=1.0,
            residuals=np.zeros(3, dtype=float),
            nfev=1,
            message="forced optimum",
        )

    monkeypatch.setattr(global_fitting, "fit_parameters", _fit_parameters)

    result = global_fitting.fit_global(
        evaluator,
        [
            {
                "id": "ds1",
                "t": np.asarray([0.0, 1.0, 2.0], dtype=float),
                "species": "A",
                "y": np.asarray([1.0, 4.0, 4.0], dtype=float),
            }
        ],
        shared_params={"dose": 1.0},
        bounds={"dose": (0.0, 10.0)},
        method="trf",
        max_nfev=1,
        max_runtime_lanes=1,
    )

    assert result.completion.status == "ok"
    assert result.shared_params["dose"] == pytest.approx(3.0)
    assert float(np.asarray(result.model_series["ds1"]["A"], dtype=float)[-1]) == pytest.approx(4.0, abs=1e-6)


def test_dataset_simulation_generic_wrap_prefers_inner_snapshot_when_sources_disagree_and_preserves_context() -> None:
    import kindred.core.analysis.global_fitting as global_fitting

    class _ContextCarrierError(RuntimeError):
        def __init__(self):
            super().__init__("generic simulation failure")
            self.context = ErrorContext(line=7, col=3, line_text="A -> B", stack_trace="sim traceback")
            self.details = {"origin": "generic-sim", "parameters": {"ds1::init:A": 9.99}}
            self.failed_params = {"ds1::init:A": 1.25}

    item = _dataset_input(0, "ds1", 2.0)

    def _raise_generic(*_args, **_kwargs):
        raise _ContextCarrierError()

    original = global_fitting.evaluate_fitting_series
    global_fitting.evaluate_fitting_series = _raise_generic
    try:
        evaluation = global_fitting._evaluate_dataset_simulation(object(), item)
    finally:
        global_fitting.evaluate_fitting_series = original

    assert isinstance(evaluation.error, FitSimulationError)
    assert evaluation.error.context is not None
    assert evaluation.error.context.line == 7
    assert evaluation.error.context.col == 3
    assert evaluation.error.context.line_text == "A -> B"
    assert evaluation.error.context.stack_trace == "sim traceback"
    assert evaluation.error.details["origin"] == "generic-sim"
    assert item.failed_param_snapshot == {"init:A": 2.0}
    assert evaluation.error.failed_params == {"ds1::init:A": 1.25}
    assert "parameters" not in evaluation.error.details


def test_fit_global_serial_generic_wrap_uses_outer_candidate_snapshot_when_inner_snapshot_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import kindred.core.analysis.global_fitting as global_fitting

    def _fake_fit_parameters(*_args, **_kwargs) -> FitResult:
        return FitResult(
            success=True,
            parameters={"k": 0.75, "ds1::init:A": 1.25},
            uncertainties=None,
            chi_squared=1.0,
            r_squared=0.0,
            residuals=np.asarray([0.0], dtype=float),
            nfev=1,
            message="ok",
            covariance=None,
        )

    def _raise_runtime_error(*_args, **_kwargs):
        raise RuntimeError("generic simulation failure without snapshot")

    monkeypatch.setattr(global_fitting, "fit_parameters", _fake_fit_parameters)
    monkeypatch.setattr(global_fitting, "evaluate_fitting_series", _raise_runtime_error)

    result = global_fitting.fit_global(
        lambda _params: {"t": np.asarray([0.0, 1.0], dtype=float), "A": np.asarray([1.0, 1.0], dtype=float)},
        datasets=[_raw_dataset("ds1", [1.0, 0.8])],
        shared_params={"k": 0.2},
        dataset_variable_params={
            "ds1": {"init:A": {"initial": 0.3, "min": 0.1, "max": 3.0}},
        },
        max_nfev=1,
    )

    assert result.completion.status == "fail"
    assert result.completion.dataset_failures["ds1"].parameter_snapshot == {
        "k": pytest.approx(0.75),
        "ds1::init:A": pytest.approx(1.25),
    }
    assert "parameters" not in result.completion.dataset_failures["ds1"].failure.get("details", {})


def test_objective_simulation_error_rewrap_preserves_context() -> None:
    import kindred.core.analysis.global_fitting as global_fitting
    from kindred.core.objective import ObjectiveContext

    class _ObjectiveCarrierError(RuntimeError):
        def __init__(self):
            super().__init__("evaluation error")
            self.context = ErrorContext(line=9, col=4, line_text="k1=bad", stack_trace="objective traceback")
            self.details = {"origin": "objective-sim"}

    payloads = [_payload("ds1", [0.0, 0.0])]
    layout = global_fitting._build_parameter_layout(
        payloads=payloads,
        shared_params={"k1": 1.0},
        dataset_variable_params={},
        bounds=None,
        log10_params=None,
    )
    ctx = ObjectiveContext()
    objective = global_fitting._GlobalFitObjective(
        fit_evaluator=object(),
        payloads=payloads,
        shared_params={"k1": 1.0},
        dataset_params={"ds1": {"init:A": 1.25}},
        weights={"ds1": 1.0},
        layout=layout,
        penalty_value=1e6,
        ctx=ctx,
        progress_callback=None,
        cancellation_check=None,
    )
    evaluation = global_fitting._DatasetSimulationEvaluation(
        index=0,
        sim_time=None,
        sim_species={},
        error=_ObjectiveCarrierError(),
        error_provenance={"dataset": "ds1"},
        final_error_message="evaluation error",
    )

    original = global_fitting._evaluate_dataset_simulations
    global_fitting._evaluate_dataset_simulations = lambda *_args, **_kwargs: [evaluation]
    try:
        residuals = objective(layout.x0.copy())
    finally:
        global_fitting._evaluate_dataset_simulations = original

    assert residuals.size
    assert isinstance(ctx.last_error, FitSimulationError)
    assert ctx.last_error.context is not None
    assert ctx.last_error.context.line == 9
    assert ctx.last_error.context.col == 4
    assert ctx.last_error.context.line_text == "k1=bad"
    assert ctx.last_error.context.stack_trace == "objective traceback"
    assert ctx.last_error.details["origin"] == "objective-sim"


def test_objective_alignment_rewrap_preserves_context() -> None:
    import kindred.core.analysis.global_fitting as global_fitting
    from kindred.core.objective import ObjectiveContext

    payloads = [_payload("ds1", [0.0, 0.0])]
    layout = global_fitting._build_parameter_layout(
        payloads=payloads,
        shared_params={"k1": 1.0},
        dataset_variable_params={},
        bounds=None,
        log10_params=None,
    )
    ctx = ObjectiveContext()
    objective = global_fitting._GlobalFitObjective(
        fit_evaluator=object(),
        payloads=payloads,
        shared_params={"k1": 1.0},
        dataset_params={"ds1": {"init:A": 1.25}},
        weights={"ds1": 1.0},
        layout=layout,
        penalty_value=1e6,
        ctx=ctx,
        progress_callback=None,
        cancellation_check=None,
    )
    evaluation = global_fitting._DatasetSimulationEvaluation(
        index=0,
        sim_time=np.asarray([0.0, 1.0], dtype=float),
        sim_species={"A": np.asarray([0.0, 0.0], dtype=float)},
    )
    exc = FitSimulationError(
        "alignment failure",
        context=ErrorContext(line=11, col=2, line_text="align", stack_trace="alignment traceback"),
        details={"origin": "alignment"},
    )

    original_evaluate = global_fitting._evaluate_dataset_simulations
    original_align = global_fitting._align_series
    global_fitting._evaluate_dataset_simulations = lambda *_args, **_kwargs: [evaluation]
    global_fitting._align_series = lambda *_args, **_kwargs: (_ for _ in ()).throw(exc)
    try:
        residuals = objective(layout.x0.copy())
    finally:
        global_fitting._evaluate_dataset_simulations = original_evaluate
        global_fitting._align_series = original_align

    assert residuals.size
    assert isinstance(ctx.last_error, FitSimulationError)
    assert ctx.last_error.context is not None
    assert ctx.last_error.context.line == 11
    assert ctx.last_error.context.col == 2
    assert ctx.last_error.context.line_text == "align"
    assert ctx.last_error.context.stack_trace == "alignment traceback"
    assert ctx.last_error.details["origin"] == "alignment"


def test_objective_alignment_generic_wrap_preserves_context() -> None:
    import kindred.core.analysis.global_fitting as global_fitting
    from kindred.core.objective import ObjectiveContext

    class _AlignmentGenericError(RuntimeError):
        def __init__(self):
            super().__init__("alignment runtime failure")
            self.context = ErrorContext(line=13, col=6, line_text="align generic", stack_trace="generic traceback")
            self.details = {"origin": "alignment-generic"}

    payloads = [_payload("ds1", [0.0, 0.0])]
    layout = global_fitting._build_parameter_layout(
        payloads=payloads,
        shared_params={"k1": 1.0},
        dataset_variable_params={},
        bounds=None,
        log10_params=None,
    )
    ctx = ObjectiveContext()
    objective = global_fitting._GlobalFitObjective(
        fit_evaluator=object(),
        payloads=payloads,
        shared_params={"k1": 1.0},
        dataset_params={"ds1": {"init:A": 1.25}},
        weights={"ds1": 1.0},
        layout=layout,
        penalty_value=1e6,
        ctx=ctx,
        progress_callback=None,
        cancellation_check=None,
    )
    evaluation = global_fitting._DatasetSimulationEvaluation(
        index=0,
        sim_time=np.asarray([0.0, 1.0], dtype=float),
        sim_species={"A": np.asarray([0.0, 0.0], dtype=float)},
    )

    original_evaluate = global_fitting._evaluate_dataset_simulations
    original_align = global_fitting._align_series
    global_fitting._evaluate_dataset_simulations = lambda *_args, **_kwargs: [evaluation]
    global_fitting._align_series = lambda *_args, **_kwargs: (_ for _ in ()).throw(_AlignmentGenericError())
    try:
        residuals = objective(layout.x0.copy())
    finally:
        global_fitting._evaluate_dataset_simulations = original_evaluate
        global_fitting._align_series = original_align

    assert residuals.size
    assert isinstance(ctx.last_error, FitSimulationError)
    assert ctx.last_error.context is not None
    assert ctx.last_error.context.line == 13
    assert ctx.last_error.context.col == 6
    assert ctx.last_error.context.line_text == "align generic"
    assert ctx.last_error.context.stack_trace == "generic traceback"
    assert ctx.last_error.details["origin"] == "alignment-generic"


def test_global_fit_result_completion_keeps_dataset_failures_without_message_view_duplication() -> None:
    from kindred.core.analysis.global_fitting import GlobalFitResult

    result = GlobalFitResult(
        shared_params={"k1": 1.0},
        dataset_params={"ds1": {}},
        uncertainties=None,
        global_chi_squared=1.0,
        global_r_squared=0.0,
        dataset_info=[],
        nfev=1,
        message="failed",
        completion=GlobalFitCompletion(
            status="fail",
            optimizer_converged=True,
            nonfinite_metrics=False,
            dataset_failures={
                "ds1": FitDiagnostic(
                    phase="final_replay",
                    dataset_id="ds1",
                    failure=build_simulation_failure(kind="simulation_error", message="first message"),
                ),
                "ds2": FitDiagnostic(
                    phase="final_replay",
                    dataset_id="ds2",
                    failure=build_simulation_failure(kind="preparation_error", message="second message"),
                ),
            },
            detail_sections=[
                FitDetailSection(
                    dataset_id="ds1",
                    failure=build_simulation_failure(kind="simulation_error", message="first message"),
                ),
                FitDetailSection(
                    dataset_id="ds2",
                    failure=build_simulation_failure(kind="preparation_error", message="second message"),
                ),
            ],
        ),
    )

    assert result.completion.dataset_failures["ds1"].failure["message"] == "first message"
    assert result.completion.dataset_failures["ds2"].failure["message"] == "second message"
    assert not hasattr(result, "dataset_error_messages")


def test_fit_global_routes_last_error_to_optimizer_diagnostic_without_message_suffix() -> None:
    from kindred.core.api.fitting import fit_global

    datasets = [_raw_dataset("ds1", [0.0, 0.0, 0.0, 0.0])]
    t = np.asarray(datasets[0]["t"], dtype=float)

    def _sim(_params):
        return {"t": t, "A": np.array([0.0, np.nan, 0.0, 0.0])}

    result = fit_global(
        _sim,
        datasets,
        shared_params={"k": 0.1},
        method="trf",
        max_nfev=10,
    )

    assert result.completion.optimizer_diagnostic is not None
    assert result.completion.optimizer_diagnostic.dataset_id == "ds1"
    assert result.completion.optimizer_diagnostic.remediation == "generic_retry"
    assert "parameters" not in result.completion.optimizer_diagnostic.failure.get("details", {})
