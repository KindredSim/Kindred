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
    FitCandidateDomainError,
    FittingCancelled,
    SimulationCancelled,
)
from kindred.core.fitting_optimization import fit_parameters
from kindred.core.fitting_completion import FitExtensionProgress
from kindred.core.fitting_method_policy import LEAST_SQUARES_TOLERANCE
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


def fit_global(
    *,
    fit_evaluators: Mapping[str, object],
    datasets: Sequence[FitDatasetSpec],
    dataset_overrides: Sequence[FitDatasetParameterOverrides] = (),
    weights: Optional[Mapping[str, float]] = None,
    method: str,
    max_nfev: int = 1000,
    ftol: float,
    xtol: float,
    seed: Optional[int] = None,
    progress_callback: Optional[
        Callable[[_fit_execution.FitProgress], None]
    ] = None,
    cancellation_check: Optional[Callable[[], bool]] = None,
    cancellation_control: SimulationCancellationControl | None = None,
    stop_check: Optional[Callable[[], bool]] = None,
    starting_facts: tuple[_fit_execution.DatasetFitInfo, ...] = (),
    profile_likelihood: ProfileLikelihoodConfig | None = None,
    residual_bootstrap: ResidualBootstrapConfig | None = None,
    extension_progress_callback: Optional[
        Callable[[FitExtensionProgress], None]
    ] = None,
    optimizer_layout: _fit_execution.FitParameterLayout,
    executor: Executor,
    optimizer_start: np.ndarray | None = None,
) -> _fit_execution.GlobalFitResult | ScientificUnavailable:
    """Fit immutable dataset captures through one evaluator per dataset.

    ``optimizer_start`` overrides the layout's starting vector (in optimizer
    coordinates) without changing the layout used to decode results.
    """

    # The Fit setup admits the method and evaluation budget (Fit defaults), and
    # its dispatch compiles one evaluator, dataset capture and override record
    # for each unique included dataset, in roster order.
    payloads = list(datasets)
    evaluators = dict(fit_evaluators)
    overrides = dataset_overrides
    fixed_dataset_params = {
        item.dataset_id: dict(item.fixed_params)
        for item in overrides
    }
    weights_norm = _fit_execution.normalize_weights(
        payloads,
        None if weights is None else dict(weights),
    )
    layout = optimizer_layout

    logger.info(
        "Global fitting: %d datasets, %d optimizer coordinates",
        len(payloads),
        len(layout.opt_param_keys()),
    )
    penalty_value = 1e6
    objective_total_points = int(sum(item.point_count for item in payloads))

    def hard_cancelled() -> bool:
        return bool(
            (cancellation_check is not None and cancellation_check())
            or (cancellation_control is not None and cancellation_control.cancel_requested())
        )

    def active_cancelled() -> bool:
        return hard_cancelled() or bool(stop_check is not None and stop_check())

    active_control = SimulationCancellationControl(
        lambda: bool(
            (cancellation_control is not None and cancellation_control.cancel_requested())
            or (stop_check is not None and stop_check())
        )
    )
    try:
        objective = _fit_execution.GlobalFitObjective(
            fit_evaluators=evaluators,
            payloads=payloads,
            dataset_params=fixed_dataset_params,
            weights=weights_norm,
            layout=layout,
            penalty_value=penalty_value,
            progress_callback=progress_callback,
            cancellation_check=active_cancelled,
            cancellation_control=active_control,
            executor=executor,
        )
        if starting_facts:
            objective.retain_complete_candidate(layout.x0, starting_facts)
        keys = layout.opt_param_keys()
        primary_stopped = False
        try:
            de_penalty = (
                float((penalty_value**2) * max(1, objective_total_points))
                if method == "differential_evolution"
                else None
            )
            optimizer_result = fit_parameters(
                objective,
                (
                    layout.initial_params()
                    if optimizer_start is None
                    else {
                        key: float(optimizer_start[index])
                        for index, key in enumerate(keys)
                    }
                ),
                bounds=layout.bounds_dict(),
                method=method,
                max_nfev=max_nfev,
                seed=seed,
                ftol=ftol,
                xtol=xtol,
                cancellation_check=active_cancelled,
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
            optimizer_endpoint_vector = np.asarray(
                [optimizer_result.parameters[key] for key in keys],
                dtype=float,
            )

            def replay(
                candidate_vector: np.ndarray,
                *,
                converged: bool,
            ) -> _fit_execution.GlobalFitResult | ScientificUnavailable:
                try:
                    return _fit_execution._replay_fit_result(
                        fit_evaluators=evaluators,
                        payloads=payloads,
                        layout=layout,
                        candidate_vector=candidate_vector,
                        fixed_dataset_params=fixed_dataset_params,
                        weights=weights_norm,
                        cancellation_check=active_cancelled,
                        cancellation_control=active_control,
                        success=converged,
                        message=message,
                        optimizer_diagnostic=optimizer_diagnostic,
                        executor=executor,
                    )
                except FitCandidateDomainError as exc:
                    return ScientificUnavailable(
                        producer="global_fit_replay",
                        stage=exc.details["stage"],
                        details=exc.details,
                        factual_cause=getattr(exc, "factual_cause", None),
                    )

            # Scientific availability belongs to the actual optimizer endpoint
            # replay. A better fully evaluated probe must not replace it.
            result = replay(optimizer_endpoint_vector, converged=success)
            if isinstance(result, ScientificUnavailable):
                retained_best = objective.best_complete_vector
                if retained_best is None:
                    return result
                retained_best_vector = np.asarray(retained_best, dtype=float).reshape(-1)
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
                        # Whether the optimizer itself converged before its endpoint failed.
                        "optimizer_converged": success,
                    },
                    cause=optimizer_diagnostic,
                )
                result = replay(retained_best_vector, converged=False)
                if isinstance(result, ScientificUnavailable):
                    return result
        except (FittingCancelled, SimulationCancelled):
            if hard_cancelled() or stop_check is None or not stop_check():
                raise
            retained_vector = objective.best_complete_vector
            if retained_vector is None:
                raise FittingCancelled()
            primary_stopped = True
            message = "Fit stopped. Best complete candidate retained."
            result = _fit_execution._assemble_fit_result_facts(
                payloads=payloads,
                layout=layout,
                candidate_vector=retained_vector,
                facts=objective.best_complete_facts,
                cancellation_check=hard_cancelled,
                cancellation_control=cancellation_control,
                success=False,
                message=message,
                optimizer_diagnostic=FactualCause(
                    domain="fit", origin="optimizer", code=None,
                    type="FittingStopped", message=message, details={},
                ),
            )
        vector = np.asarray(result.optimizer_vector, dtype=float)
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
            progress_callback=None,
            cancellation_check=active_cancelled,
            cancellation_control=active_control,
            executor=executor,
        )
        profile_result: ProfileLikelihoodResult | None = None
        bootstrap_result: ResidualBootstrapResult | None = None
        optional_analysis_stopped = primary_stopped
        terminal_parameters = {
            key: float(vector[index])
            for index, key in enumerate(keys)
        }
        # A parameter group's public uncertainty target is moved by the
        # optimizer as its coupled variable; analyses see the group identity.
        group_targets = {
            _fit_execution.FitCoordinate(
                "coupled",
                _fit_execution.parameter_group_variable_id(coordinate.name, coordinate.dataset_id),
            ): coordinate
            for analysis in (profile_likelihood, residual_bootstrap)
            if analysis is not None
            for coordinate in analysis.parameter_names
            if coordinate.scope == "group"
        }
        optimizer_keys = {target: key for key, target in group_targets.items()}
        # Profile and Bootstrap refit by least squares.  After Differential
        # Evolution, whose ftol is its population tolerance, they use the
        # least-squares default tolerances.
        refit_ftol, refit_xtol = (
            (LEAST_SQUARES_TOLERANCE, LEAST_SQUARES_TOLERANCE)
            if method == "differential_evolution" else (ftol, xtol)
        )
        if profile_likelihood is not None:
            base_residuals = np.concatenate(
                [info.weighted_residuals for info in result.dataset_info]
            )
            base_residuals.setflags(write=False)
            fitted_value_count = len(keys)
            try:
                if primary_stopped:
                    raise FittingCancelled("Profile likelihood was not started after Fit stopped.")
                profile_result = profile_likelihood_result(
                    objective=terminal_objective,
                    initial_parameters={
                        group_targets.get(key, key): value
                        for key, value in terminal_parameters.items()
                    },
                    bounds={
                        group_targets.get(key, key): value
                        for key, value in layout.bounds_dict().items()
                    },
                    config=profile_likelihood,
                    decode_coordinate=lambda coordinate, value: layout.decode_coordinate(
                        optimizer_keys.get(coordinate, coordinate), value,
                    ),
                    ftol=refit_ftol,
                    xtol=refit_xtol,
                    base_residuals=base_residuals,
                    cancellation_check=active_cancelled,
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
                    config=profile_likelihood,
                    status="stopped",
                    cause=stop_cause,
                    observation_count=int(base_residuals.size),
                    fitted_value_count=fitted_value_count,
                )
            except Exception as exc:
                profile_result = profile_likelihood_settlement_result(
                    config=profile_likelihood,
                    status="failed",
                    cause=factual_cause_from_exception(
                        exc,
                        domain="fit",
                        origin="profile_likelihood",
                    ),
                    observation_count=int(base_residuals.size),
                    fitted_value_count=fitted_value_count,
                )

            if primary_stopped:
                profile_result = replace(
                    profile_result,
                    provenance={**profile_result.provenance, "settlement": "not_started_after_stop"},
                )

        if residual_bootstrap is not None:
            def refit_bootstrap(
                synthetic_payloads: Sequence[FitDatasetSpec],
            ) -> _fit_execution.GlobalFitResult | ScientificUnavailable:
                return fit_global(
                    fit_evaluators=evaluators,
                    datasets=synthetic_payloads,
                    dataset_overrides=overrides,
                    weights=weights_norm,
                    method=residual_bootstrap.optimizer_method,
                    max_nfev=max_nfev,
                    ftol=refit_ftol,
                    xtol=refit_xtol,
                    progress_callback=None,
                    extension_progress_callback=None,
                    cancellation_check=active_cancelled,
                    cancellation_control=active_control,
                    optimizer_layout=layout,
                    executor=executor,
                    # Bootstrap refits start from the converged primary fit.
                    optimizer_start=vector,
                )

            if optional_analysis_stopped:
                bootstrap_result = residual_bootstrap_settlement_result(
                    config=residual_bootstrap,
                    payloads=payloads,
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
                        terminal_result=result,
                        payloads=payloads,
                        config=residual_bootstrap,
                        refit=refit_bootstrap,
                        group_member_ids={
                            target: next(
                                spec.dataset_ids[0]
                                for spec in layout.coupled_var_order
                                if spec.variable_id == key.name
                            )
                            for key, target in group_targets.items()
                        },
                        cancellation_check=active_cancelled,
                        progress_callback=extension_progress_callback,
                    )
                except (FittingCancelled, SimulationCancelled) as exc:
                    bootstrap_result = residual_bootstrap_settlement_result(
                        config=residual_bootstrap,
                        payloads=payloads,
                        status="stopped",
                        settlement="stopped",
                        cause=factual_cause_from_exception(
                            exc, domain="fit", origin="bootstrap"
                        ),
                    )
                except Exception as exc:
                    bootstrap_result = residual_bootstrap_settlement_result(
                        config=residual_bootstrap,
                        payloads=payloads,
                        status="failed",
                        settlement="failed",
                        cause=factual_cause_from_exception(
                            exc,
                            domain="fit",
                            origin="bootstrap",
                        ),
                    )

        analysis_messages = []
        for label, analysis in (
            ("Profile likelihood", profile_result), ("Residual bootstrap", bootstrap_result)
        ):
            if analysis is not None and analysis.status == "stopped":
                analysis_messages.append(
                    f"{label} was not started after Stop."
                    if analysis.provenance.get("settlement") == "not_started_after_stop"
                    else f"{label} stopped."
                )
        result = replace(
            result,
            message=" ".join((result.message, *analysis_messages)),
            profile_likelihood_result=profile_result,
            residual_bootstrap_result=bootstrap_result,
        )
        if hard_cancelled():
            raise FittingCancelled()
        return result

    except FittingCancelled:
        raise
    except SimulationCancelled as exc:
        cancellation = FittingCancelled()
        raise cancellation from exc
