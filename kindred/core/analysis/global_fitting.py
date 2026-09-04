"""Strict typed global-fit composition boundary."""

from __future__ import annotations

import logging
from concurrent.futures import Executor
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from typing import Optional

import numpy as np

from kindred.core.analysis.dataset_parameter_overrides import (
    FitDatasetParameterOverrides,
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
    dataset_overrides: Sequence[FitDatasetParameterOverrides],
) -> tuple[
    list[FitDatasetSpec],
    dict[str, object],
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
    return payloads, evaluators, overrides


def fit_global(
    *,
    fit_evaluators: Mapping[str, object],
    datasets: Sequence[FitDatasetSpec],
    dataset_overrides: Sequence[FitDatasetParameterOverrides] = (),
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
    payloads, evaluators, overrides = _validate_run_inputs(
        fit_evaluators=fit_evaluators,
        datasets=datasets,
        dataset_overrides=dataset_overrides,
    )
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
        keys = layout.opt_param_keys()
        optimizer_endpoint_vector = np.asarray(
            [optimizer_result.parameters[key] for key in keys],
            dtype=float,
        )
        layout.validate_optimizer_vector(optimizer_endpoint_vector)

        def replay(
            candidate_vector: np.ndarray,
            *,
            converged: bool,
        ) -> _fit_execution._FitReplayState:
            return _fit_execution._assemble_fit_replay_state(
                fit_evaluators=evaluators,
                payloads=payloads,
                layout=layout,
                candidate_vector=candidate_vector,
                fixed_dataset_params=fixed_dataset_params,
                weights=weights_norm,
                cancellation_check=cancellation_check,
                cancellation_control=cancellation_control,
                success=converged,
                message=message,
                optimizer_diagnostic=optimizer_diagnostic,
                executor=executor,
            )

        try:
            # Scientific availability belongs to the actual optimizer endpoint
            # replay.  A better fully evaluated probe must not replace it.
            replay_state = replay(optimizer_endpoint_vector, converged=success)
        except FitSimulationError as exc:
            details = dict(getattr(exc, "details", None) or {})
            if not details.get("scientific_unavailable"):
                raise
            retained_best = objective.best_complete_vector
            if retained_best is None:
                return ScientificUnavailable(
                    producer=str(details.get("producer", "global_fit_replay")),
                    stage=str(details.get("stage", "final_replay")),
                    details=details,
                    factual_cause=getattr(exc, "factual_cause", None),
                )
            retained_best_vector = np.asarray(retained_best, dtype=float).reshape(-1)
            layout.validate_optimizer_vector(retained_best_vector)
            success = False
            optimizer_diagnostic = FactualCause(
                domain="fit",
                origin="optimizer_endpoint",
                code=None,
                type="FittingUnavailableEndpointReplaced",
                message=(
                    "The unavailable optimizer endpoint was replaced by the "
                    "best scientifically available evaluated candidate."
                ),
                details={
                    "endpoint_replaced": True,
                    "settlement": "best_scientifically_available_evaluated_candidate",
                },
                cause=optimizer_diagnostic,
            )
            try:
                replay_state = replay(retained_best_vector, converged=False)
            except FitSimulationError as fallback_exc:
                fallback_details = dict(
                    getattr(fallback_exc, "details", None) or {}
                )
                if not fallback_details.get("scientific_unavailable"):
                    raise
                return ScientificUnavailable(
                    producer=str(
                        fallback_details.get("producer", "global_fit_replay")
                    ),
                    stage=str(fallback_details.get("stage", "final_replay")),
                    details=fallback_details,
                    factual_cause=getattr(fallback_exc, "factual_cause", None),
                )
        vector = np.asarray(replay_state.optimizer_vector, dtype=float)
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
                    dataset_overrides=overrides,
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
        return result

    except FittingCancelled:
        raise
    except SimulationCancelled as exc:
        cancellation = FittingCancelled()
        raise cancellation from exc
