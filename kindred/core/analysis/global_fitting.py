"""Strict typed global-fit composition boundary."""

from __future__ import annotations

import logging
from concurrent.futures import Executor
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from typing import Optional

import numpy as np

from kindred.core.analysis.dataset_parameter_overrides import (
    FitCentredFixedEffectsParamSpec,
    FitCoupledVariableParamSpec,
    FitDatasetParameterOverrides,
    FitSharedVariableParamSpec,
)
from kindred.core.analysis.fit_dataset_payload import FitDatasetSpec
from kindred.core.analysis.fit_uncertainty import (
    bootstrap_de_seed,
    ProfileLikelihoodConfig,
    ProfileLikelihoodResult,
    ResidualBootstrapConfig,
    ResidualBootstrapResult,
    profile_likelihood_settlement_result,
    profile_likelihood_result,
    residual_bootstrap_settlement_result,
    residual_bootstrap_result,
)
from kindred.core.analysis import global_fit_execution as _fit_execution
from kindred.core.exceptions import (
    FitSimulationError,
    FittingCancelled,
    SimulationCancelled,
)
from kindred.core.fitting_optimization import fit_parameters
from kindred.core.fitting_completion import FitExtensionProgress
from kindred.core.objective import ObjectiveContext
from kindred.core.fitting_method_policy import validate_fit_method
from kindred.core.species import validate_evaluation_count
from kindred.core.simulation_failure import (
    FactualCause,
    factual_cause_from_exception,
)
from kindred.core.analysis.fit_evaluation_outcome import ScientificUnavailable
from kindred.core.simulator.solver_types import SimulationCancellationControl

logger = logging.getLogger(__name__)

__all__ = [
    "fit_global",
]


def _validate_run_inputs(
    *,
    fit_evaluators: Mapping[str, object],
    datasets: Sequence[FitDatasetSpec],
    shared_variables: Sequence[FitSharedVariableParamSpec],
    dataset_overrides: Sequence[FitDatasetParameterOverrides],
) -> tuple[
    list[FitDatasetSpec],
    dict[str, object],
    tuple[FitSharedVariableParamSpec, ...],
    tuple[FitDatasetParameterOverrides, ...],
]:
    payloads = sorted(datasets, key=lambda item: str(getattr(item, "dataset_id", "")))
    if not payloads:
        raise ValueError("At least one Fit dataset is required.")
    if any(not isinstance(item, FitDatasetSpec) for item in payloads):
        raise TypeError("fit_global accepts only FitDatasetSpec dataset captures.")
    dataset_ids = [item.dataset_id for item in payloads]
    if len(dataset_ids) != len(set(dataset_ids)):
        raise ValueError("Fit dataset IDs must be unique.")

    evaluators = dict(fit_evaluators)
    if set(evaluators) != set(dataset_ids):
        missing = sorted(set(dataset_ids) - set(evaluators))
        extra = sorted(set(evaluators) - set(dataset_ids))
        details = []
        if missing:
            details.append(f"missing={','.join(missing)}")
        if extra:
            details.append(f"extra={','.join(extra)}")
        raise ValueError(
            "Fit evaluator keys must exactly match captured dataset IDs"
            + (f" ({'; '.join(details)})" if details else "")
            + "."
        )

    shared = tuple(sorted(shared_variables, key=lambda item: item.name))
    shared_names = [item.name for item in shared]
    if len(shared_names) != len(set(shared_names)):
        raise ValueError("Shared Fit parameter names must be unique.")

    overrides = tuple(sorted(dataset_overrides, key=lambda item: item.dataset_id))
    override_ids = [item.dataset_id for item in overrides]
    if len(override_ids) != len(set(override_ids)):
        raise ValueError("Each Fit dataset may have only one override record.")
    unknown_overrides = sorted(set(override_ids) - set(dataset_ids))
    if unknown_overrides:
        raise ValueError(
            "Fit overrides reference unknown datasets: "
            + ", ".join(unknown_overrides)
            + "."
        )
    return payloads, evaluators, shared, overrides
def _mark_late_stop(
    result: _fit_execution.GlobalFitResult,
    *,
    cause: FactualCause,
) -> _fit_execution.GlobalFitResult:
    """Mark an already replayed result diagnostic-only without replaying it."""

    if not result.optimizer_converged and result.diagnostic_only:
        return result
    return replace(
        result,
        message=_fit_execution.fit_result_message(
            base_message="Stopped by user after primary Fit settlement.",
            status="warn",
            optimizer_converged=False,
            optimizer_diagnostic=cause,
        ),
        status="warn",
        optimizer_converged=False,
        optimizer_diagnostic=cause,
        diagnostic_only=True,
    )


def fit_global(
    *,
    fit_evaluators: Mapping[str, object],
    datasets: Sequence[FitDatasetSpec],
    shared_variables: Sequence[FitSharedVariableParamSpec],
    dataset_overrides: Sequence[FitDatasetParameterOverrides] = (),
    coupled_variables: Sequence[FitCoupledVariableParamSpec] = (),
    centred_variables: Sequence[FitCentredFixedEffectsParamSpec] = (),
    weights: Optional[Mapping[str, float]] = None,
    method: str,
    max_nfev: int = 1000,
    ftol: float = 1e-10,
    xtol: float = 1e-10,
    seed: Optional[int] = None,
    progress_callback: Optional[
        Callable[[_fit_execution.FitProgress], None]
    ] = None,
    cancellation_check: Optional[Callable[[], bool]] = None,
    cancellation_control: SimulationCancellationControl | None = None,
    stop_best_requested: Callable[[], bool] | None = None,
    stopped_replay_control: SimulationCancellationControl | None = None,
    profile_likelihood: ProfileLikelihoodConfig | None = None,
    residual_bootstrap: ResidualBootstrapConfig | None = None,
    extension_progress_callback: Optional[
        Callable[[FitExtensionProgress], None]
    ] = None,
    optimizer_layout: _fit_execution.FitParameterLayout,
    executor: Executor,
) -> _fit_execution.GlobalFitResult | ScientificUnavailable:
    """Fit immutable dataset captures through one evaluator per dataset."""

    if executor is None:
        raise TypeError("Global Fit requires a borrowed executor.")
    method = validate_fit_method(method)
    max_nfev = validate_evaluation_count(
        max_nfev,
        field="Fit evaluation budget",
    )
    if max_nfev <= 0:
        raise ValueError("Fit evaluation budget must be positive.")
    payloads, evaluators, shared, overrides = _validate_run_inputs(
        fit_evaluators=fit_evaluators,
        datasets=datasets,
        shared_variables=shared_variables,
        dataset_overrides=dataset_overrides,
    )
    coupled = tuple(coupled_variables)
    centred = tuple(centred_variables)
    if any(not isinstance(item, FitCentredFixedEffectsParamSpec) for item in centred):
        raise TypeError("Centred Fit requests must contain typed parameter specifications.")
    shared_params = {item.name: item.initial for item in shared}
    shared_bounds = {
        item.name: (item.minimum, item.maximum)
        for item in shared
    }
    shared_log10 = {item.name: item.log10 for item in shared}
    fixed_dataset_params = {
        item.dataset_id: dict(item.fixed_params)
        for item in overrides
    }
    weights_norm = _fit_execution.normalize_weights(
        payloads,
        None if weights is None else dict(weights),
    )
    if not isinstance(optimizer_layout, _fit_execution.FitParameterLayout):
        raise TypeError("Global Fit requires its pre-dispatch optimizer layout.")
    layout = optimizer_layout
    expected_layout = _fit_execution.build_parameter_layout_for_dataset_ids(
        dataset_ids=tuple(item.dataset_id for item in payloads),
        shared_params=shared_params,
        dataset_overrides=overrides,
        coupled_variables=coupled,
        bounds=shared_bounds,
        log10_params=shared_log10,
        centred_variables=centred,
    )
    if (
        layout.opt_param_keys() != expected_layout.opt_param_keys()
        or layout.initial_params() != expected_layout.initial_params()
        or layout.bounds_dict() != expected_layout.bounds_dict()
    ):
        raise ValueError("Pre-dispatch Fit optimizer layout no longer matches execution.")

    logger.info(
        "Global fitting: %d datasets, %d optimizer coordinates",
        len(payloads),
        len(layout.opt_param_keys()),
    )
    penalty_value = 1e6
    objective_total_points = int(sum(item.point_count for item in payloads))
    context = ObjectiveContext()

    try:
        objective = _fit_execution.GlobalFitObjective(
            fit_evaluators=evaluators,
            payloads=payloads,
            dataset_params=fixed_dataset_params,
            weights=weights_norm,
            layout=layout,
            penalty_value=penalty_value,
            ctx=context,
            progress_callback=progress_callback,
            cancellation_check=cancellation_check,
            cancellation_control=cancellation_control,
            executor=executor,
        )
        de_penalty = (
            float((penalty_value**2) * max(1, objective_total_points))
            if method == "differential_evolution"
            else None
        )
        optimizer_result = fit_parameters(
            objective,
            layout.initial_params(),
            bounds=layout.bounds_dict(),
            method=method,
            max_nfev=max_nfev,
            seed=seed,
            ftol=ftol,
            xtol=xtol,
            cancellation_check=cancellation_check,
            de_penalty=de_penalty,
        )
        success = bool(optimizer_result.success)
        message = str(optimizer_result.message)
        optimizer_diagnostic: FactualCause | None = None
        identifiability = str(
            optimizer_result.identifiability_warning or ""
        ).strip()
        if identifiability:
            optimizer_diagnostic = FactualCause(
                domain="fit",
                origin="identifiability",
                code=None,
                type="FittingIdentifiability",
                message=identifiability,
                details={"final_report_warning": True},
            )
        if context.last_error is not None and optimizer_diagnostic is None:
            provenance = context.last_error_provenance
            dataset_id = (
                str(provenance.get("dataset"))
                if isinstance(provenance, Mapping) and provenance.get("dataset") is not None
                else None
            )
            optimizer_diagnostic = factual_cause_from_exception(
                context.last_error,
                domain="fit",
                origin="optimizer",
            )
            if dataset_id is not None:
                optimizer_diagnostic = replace(
                    optimizer_diagnostic,
                    details={
                        **dict(optimizer_diagnostic.details),
                        "dataset": dataset_id,
                    },
                )
        keys = layout.opt_param_keys()
        optimizer_endpoint_vector = np.asarray(
            [optimizer_result.parameters[key] for key in keys],
            dtype=float,
        )
        vector = optimizer_endpoint_vector.copy()
        retained_best = objective.best_complete_vector
        if retained_best is not None:
            retained_best_vector = np.asarray(retained_best, dtype=float).reshape(-1)
            if not np.array_equal(optimizer_endpoint_vector, retained_best_vector):
                success = False
                replacement_cause = FactualCause(
                    domain="fit",
                    origin="optimizer_endpoint",
                    code=None,
                    type="FittingUnavailableEndpointReplaced",
                    message=(
                        "The unavailable optimizer endpoint was replaced by the "
                        "best valid evaluated candidate."
                    ),
                    details={
                        "endpoint_replaced": True,
                        "settlement": "best_valid_evaluated_candidate",
                    },
                    cause=optimizer_diagnostic,
                )
                optimizer_diagnostic = replacement_cause
            vector = retained_best_vector
        layout.validate_optimizer_vector(vector)
        replay_cancellation_check = cancellation_check
        replay_cancellation_control = cancellation_control
        if callable(stop_best_requested):
            def stopped_replay_check() -> bool:
                # A user Stop is settled by this one terminal replay.  The
                # ordinary control still owns Close and dataset deadlines.
                return False
            replay_cancellation_check = stopped_replay_check
            if stopped_replay_control is not None:
                replay_cancellation_control = stopped_replay_control
        try:
            replay_state = _fit_execution._assemble_fit_replay_state(
                fit_evaluators=evaluators,
                payloads=payloads,
                layout=layout,
                candidate_vector=vector,
                fixed_dataset_params=fixed_dataset_params,
                weights=weights_norm,
                cancellation_check=replay_cancellation_check,
                cancellation_control=replay_cancellation_control,
                success=success,
                message=message,
                scipy_nfev=int(optimizer_result.scipy_nfev),
                objective_call_count=int(objective.objective_calls),
                optimizer_diagnostic=optimizer_diagnostic,
                executor=executor,
            )
        except FitSimulationError as exc:
            details = dict(getattr(exc, "details", None) or {})
            if not details.get("scientific_unavailable"):
                raise
            return ScientificUnavailable(
                producer=str(details.get("producer", "global_fit_replay")),
                stage=str(details.get("stage", "final_replay")),
                details=details,
                factual_cause=getattr(exc, "factual_cause", None),
            )
        stop_during_primary_replay = bool(
            callable(stop_best_requested) and stop_best_requested()
        )
        # Post-fit methods receive a fresh objective without progress emission.
        # It retains the same evaluators, candidate/fatal taxonomy, coordinates,
        # normalized weights, cancellation owner, and immutable captures.
        terminal_objective = _fit_execution.GlobalFitObjective(
            fit_evaluators=evaluators,
            payloads=payloads,
            dataset_params=fixed_dataset_params,
            weights=weights_norm,
            layout=layout,
            penalty_value=penalty_value,
            ctx=ObjectiveContext(),
            progress_callback=None,
            cancellation_check=cancellation_check,
            cancellation_control=cancellation_control,
            executor=executor,
        )
        profile_result: ProfileLikelihoodResult | None = None
        bootstrap_result: ResidualBootstrapResult | None = None
        optional_analysis_stopped = False
        terminal_parameters = {
            key: float(vector[index])
            for index, key in enumerate(keys)
        }
        if profile_likelihood is not None:
            profile_config = replace(
                profile_likelihood,
                parameter_names=tuple(
                    layout.coordinate_identity(name)
                    for name in profile_likelihood.parameter_names
                ),
            )
            try:
                profile_result = profile_likelihood_result(
                    objective=terminal_objective,
                    initial_parameters=terminal_parameters,
                    bounds=layout.bounds_dict(),
                    config=profile_config,
                    decode_coordinate=layout.decode_coordinate,
                    base_residuals=replay_state.weighted_residuals,
                    cancellation_check=cancellation_check,
                    terminal_converged=replay_state.optimizer_converged,
                    progress_callback=extension_progress_callback,
                )
            except (FittingCancelled, SimulationCancelled) as exc:
                optional_analysis_stopped = True
                stop_cause = factual_cause_from_exception(
                    exc,
                    domain="fit",
                    origin="profile_likelihood",
                )
                profile_result = profile_likelihood_settlement_result(
                    config=profile_config,
                    status="stopped",
                    cause=stop_cause,
                    observation_count=int(replay_state.weighted_residuals.size),
                    terminal_weighted_residual_sse=float(
                        replay_state.weighted_residual_sse.value
                    ),
                )
            except Exception as exc:
                profile_result = profile_likelihood_settlement_result(
                    config=profile_config,
                    status="failed",
                    cause=factual_cause_from_exception(
                        exc,
                        domain="fit",
                        origin="profile_likelihood",
                    ),
                    observation_count=int(replay_state.weighted_residuals.size),
                    terminal_weighted_residual_sse=float(
                        replay_state.weighted_residual_sse.value
                    ),
                )

        if residual_bootstrap is not None:
            def refit_bootstrap(
                synthetic_payloads: Sequence[FitDatasetSpec],
                replicate_index: int,
            ) -> _fit_execution.GlobalFitResult | ScientificUnavailable:
                replicate_seed = (
                    bootstrap_de_seed(
                        residual_bootstrap.seed,
                        replicate_index,
                    )
                    if (
                        residual_bootstrap.optimizer_method == "differential_evolution"
                    )
                    else None
                )
                return fit_global(
                    fit_evaluators=evaluators,
                    datasets=synthetic_payloads,
                    shared_variables=shared,
                    dataset_overrides=overrides,
                    coupled_variables=coupled,
                    centred_variables=centred,
                    weights=weights_norm,
                    method=residual_bootstrap.optimizer_method,
                    max_nfev=max_nfev,
                    ftol=ftol,
                    xtol=xtol,
                    seed=replicate_seed,
                    progress_callback=None,
                    extension_progress_callback=None,
                    cancellation_check=cancellation_check,
                    cancellation_control=cancellation_control,
                    optimizer_layout=layout,
                    executor=executor,
                )

            if optional_analysis_stopped:
                bootstrap_result = residual_bootstrap_settlement_result(
                    config=residual_bootstrap,
                    status="stopped",
                    settlement="not_started_after_stop",
                    cause=factual_cause_from_exception(
                        FittingCancelled(
                            "Bootstrap was not started after post-fit processing stopped."
                        ),
                        domain="fit",
                        origin="bootstrap",
                    ),
                )
            else:
                try:
                    bootstrap_result = residual_bootstrap_result(
                        terminal_result=replay_state,
                        payloads=payloads,
                        config=residual_bootstrap,
                        refit=refit_bootstrap,
                        cancellation_check=cancellation_check,
                        progress_callback=extension_progress_callback,
                    )
                    optional_analysis_stopped = bootstrap_result.status == "stopped"
                except (FittingCancelled, SimulationCancelled) as exc:
                    optional_analysis_stopped = True
                    bootstrap_result = residual_bootstrap_settlement_result(
                        config=residual_bootstrap,
                        status="stopped",
                        settlement="stopped",
                        cause=factual_cause_from_exception(
                            exc, domain="fit", origin="bootstrap"
                        ),
                    )
                except Exception as exc:
                    bootstrap_result = residual_bootstrap_settlement_result(
                        config=residual_bootstrap,
                        status="failed",
                        settlement="failed",
                        cause=factual_cause_from_exception(
                            exc,
                            domain="fit",
                            origin="bootstrap",
                        ),
                    )

        replay_state = replace(
            replay_state,
            profile_likelihood_result=profile_result,
            residual_bootstrap_result=bootstrap_result,
        )
        result = _fit_execution._construct_global_fit_result(replay_state)
        if (
            callable(stop_best_requested)
            and stop_best_requested()
            and (
                stop_during_primary_replay
                or not optional_analysis_stopped
            )
        ):
            result = _mark_late_stop(
                result,
                cause=factual_cause_from_exception(
                    FittingCancelled(
                        "Stopped by user after primary Fit settlement."
                    ),
                    domain="fit",
                    origin="user_stop",
                ),
            )
        return result

    except FittingCancelled as exc:
        if not hasattr(exc, "scipy_nfev"):
            optimizer = locals().get("optimizer_result")
            if optimizer is not None:
                setattr(exc, "scipy_nfev", int(optimizer.scipy_nfev))
        if not hasattr(exc, "objective_call_count"):
            active_objective = locals().get("objective")
            calls = getattr(active_objective, "objective_calls", None)
            if calls is not None:
                setattr(exc, "objective_call_count", int(calls))
        existing_replay_state = locals().get("replay_state")
        if (
            callable(stop_best_requested)
            and stop_best_requested()
            and existing_replay_state is not None
            and hasattr(existing_replay_state, "weighted_residuals")
        ):
            stop_cause = factual_cause_from_exception(
                exc,
                domain="fit",
                origin="user_stop",
            )
            settled_profile = locals().get("profile_result")
            if profile_likelihood is not None and settled_profile is None:
                settled_profile_config = replace(
                    profile_likelihood,
                    parameter_names=tuple(
                        layout.coordinate_identity(name)
                        for name in profile_likelihood.parameter_names
                    ),
                )
                settled_profile = profile_likelihood_settlement_result(
                    config=settled_profile_config,
                    status="stopped",
                    cause=stop_cause,
                    observation_count=int(existing_replay_state.weighted_residuals.size),
                    terminal_weighted_residual_sse=float(
                        existing_replay_state.weighted_residual_sse.value
                    ),
                )
            settled_bootstrap = locals().get("bootstrap_result")
            if residual_bootstrap is not None and settled_bootstrap is None:
                settled_bootstrap = residual_bootstrap_settlement_result(
                    config=residual_bootstrap,
                    cause=stop_cause,
                    settlement="not_started_after_stop",
                    status="stopped",
                )
            settled_state = replace(
                existing_replay_state,
                profile_likelihood_result=settled_profile,
                residual_bootstrap_result=settled_bootstrap,
            )
            return _fit_execution._construct_global_fit_result(settled_state)
        active_objective = locals().get("objective")
        if (
            callable(stop_best_requested)
            and stop_best_requested()
            and stopped_replay_control is not None
            and active_objective is not None
        ):
            retained_best = getattr(active_objective, "best_complete_vector", None)
            if callable(retained_best):
                retained_best = retained_best()
            if retained_best is not None:
                scipy_nfev = getattr(exc, "scipy_nfev", None)
                objective_call_count = getattr(exc, "objective_call_count", None)
                if scipy_nfev is None or objective_call_count is None:
                    raise RuntimeError(
                        "Stopped Fit lost its exact optimizer/objective counters."
                    )
                stop_cause = factual_cause_from_exception(
                    exc,
                    domain="fit",
                    origin="user_stop",
                )
                try:
                    stopped_state = _fit_execution._assemble_fit_replay_state(
                        fit_evaluators=evaluators,
                        payloads=payloads,
                        layout=layout,
                        candidate_vector=np.asarray(retained_best, dtype=float),
                        fixed_dataset_params=fixed_dataset_params,
                        weights=weights_norm,
                        cancellation_check=None,
                        cancellation_control=stopped_replay_control,
                        success=False,
                        message=(
                            "Stopped by user; published the best fully evaluated "
                            "valid candidate."
                        ),
                        scipy_nfev=int(scipy_nfev),
                        objective_call_count=int(objective_call_count),
                        optimizer_diagnostic=stop_cause,
                        executor=executor,
                    )
                except FitSimulationError as replay_error:
                    replay_details = dict(
                        getattr(replay_error, "details", None) or {}
                    )
                    if not replay_details.get("scientific_unavailable"):
                        raise
                    return ScientificUnavailable(
                        producer=str(
                            replay_details.get("producer", "fit_stopped_replay")
                        ),
                        stage=str(
                            replay_details.get("stage", "stopped_replay")
                        ),
                        details=replay_details,
                        factual_cause=getattr(
                            replay_error, "factual_cause", None
                        ),
                    )
                if not stop_best_requested():
                    raise FittingCancelled()
                stopped_profile = None
                if profile_likelihood is not None:
                    stopped_profile = replace(
                        profile_likelihood,
                        parameter_names=tuple(
                            layout.coordinate_identity(name)
                            for name in profile_likelihood.parameter_names
                        ),
                    )
                stopped_bootstrap = residual_bootstrap
                stopped_state = replace(
                    stopped_state,
                    profile_likelihood_result=(
                        None
                        if stopped_profile is None
                        else profile_likelihood_settlement_result(
                            config=stopped_profile,
                            status="stopped",
                            cause=stop_cause,
                            observation_count=int(stopped_state.weighted_residuals.size),
                            terminal_weighted_residual_sse=float(
                                stopped_state.weighted_residual_sse.value
                            ),
                        )
                    ),
                    residual_bootstrap_result=(
                        None
                        if stopped_bootstrap is None
                        else residual_bootstrap_settlement_result(
                            config=stopped_bootstrap,
                            cause=stop_cause,
                            settlement="not_started_after_stop",
                            status="stopped",
                        )
                    ),
                )
                return _fit_execution._construct_global_fit_result(stopped_state)
        raise
    except SimulationCancelled as exc:
        cancellation = FittingCancelled()
        optimizer = locals().get("optimizer_result")
        if optimizer is not None:
            setattr(cancellation, "scipy_nfev", int(optimizer.scipy_nfev))
        active_objective = locals().get("objective")
        calls = getattr(active_objective, "objective_calls", None)
        if calls is not None:
            setattr(cancellation, "objective_call_count", int(calls))
        raise cancellation from exc
