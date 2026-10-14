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
    ProfileNoiseModel,
    profile_likelihood_settlement_result,
    profile_likelihood_result,
    profile_noise_model,
    rate_scale_undetermined,
    undetermined_coordinates,
)
from kindred.core.analysis import global_fit_execution as _fit_execution
from kindred.core.exceptions import (
    FittingCancelled,
    SimulationCancelled,
)
from kindred.core.fitting_optimization import fit_parameters
from kindred.core.fitting_completion import FitExtensionProgress
from kindred.core.simulation_failure import FactualCause, factual_cause_from_exception
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
    progress_callback: Callable[[_fit_execution.FitProgress], None],
    cancellation_check: Optional[Callable[[], bool]] = None,
    cancellation_control: SimulationCancellationControl | None = None,
    stop_check: Callable[[], bool],
    starting_facts: tuple[_fit_execution.DatasetFitInfo, ...],
    profile_likelihood: ProfileLikelihoodConfig | None = None,
    extension_progress_callback: Callable[[FitExtensionProgress], None],
    optimizer_layout: _fit_execution.FitParameterLayout,
    executor: Executor,
    prepared_dsl: object,
) -> _fit_execution.GlobalFitResult | ScientificUnavailable:
    """Fit immutable dataset captures through one evaluator per dataset.

    After the fit, it checks for fitted values the data do not determine
    within their bounds, and for a rate scale they do not determine, and runs
    the requested Profile likelihood.
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

    def hard_cancelled() -> bool:
        return bool(
            (cancellation_check is not None and cancellation_check())
            or (cancellation_control is not None and cancellation_control.cancel_requested())
        )

    def active_cancelled() -> bool:
        return hard_cancelled() or stop_check()

    active_control = SimulationCancellationControl(
        lambda: bool(
            (cancellation_control is not None and cancellation_control.cancel_requested())
            or stop_check()
        )
    )
    def new_objective(
        progress: Optional[Callable[[_fit_execution.FitProgress], None]] = None,
    ) -> _fit_execution.GlobalFitObjective:
        return _fit_execution.GlobalFitObjective(
            fit_evaluators=evaluators,
            payloads=payloads,
            dataset_params=fixed_dataset_params,
            weights=weights_norm,
            layout=layout,
            penalty_value=penalty_value,
            progress_callback=progress,
            cancellation_check=active_cancelled,
            cancellation_control=active_control,
            executor=executor,
        )

    objective = new_objective(progress_callback)
    if starting_facts:
        objective.retain_complete_candidate(layout.x0, starting_facts)
    keys = layout.opt_param_keys()
    primary_stopped = False
    try:
        optimizer_result = fit_parameters(
            objective,
            layout.initial_params(),
            bounds=layout.bounds_dict(),
            method=method,
            max_nfev=max_nfev,
            seed=seed,
            ftol=ftol,
            xtol=xtol,
            cancellation_check=active_cancelled,
        )
        ending = "converged" if optimizer_result.success else "not_converged"
        message = optimizer_result.message
        optimizer_endpoint_vector = np.asarray(
            [optimizer_result.parameters[key] for key in keys],
            dtype=float,
        )

        def replay(
            candidate_vector: np.ndarray,
            *,
            endpoint_replaced: bool,
            identifiability_warning: str | None,
        ) -> _fit_execution.GlobalFitResult | ScientificUnavailable:
            return _fit_execution._replay_fit_result(
                fit_evaluators=evaluators,
                payloads=payloads,
                layout=layout,
                candidate_vector=candidate_vector,
                fixed_dataset_params=fixed_dataset_params,
                weights=weights_norm,
                cancellation_check=active_cancelled,
                cancellation_control=active_control,
                ending=ending,
                message=message,
                endpoint_replaced=endpoint_replaced,
                identifiability_warning=identifiability_warning,
                executor=executor,
            )

        # Scientific availability belongs to the actual optimizer endpoint
        # replay. A better fully evaluated probe must not replace it.
        result = replay(
            optimizer_endpoint_vector,
            endpoint_replaced=False,
            identifiability_warning=optimizer_result.identifiability_warning,
        )
        if isinstance(result, ScientificUnavailable):
            retained_best = objective.best_complete_vector
            if retained_best is None:
                return result
            retained_best_vector = np.asarray(retained_best, dtype=float).reshape(-1)
            # The replaced endpoint's rank warning does not describe the
            # retained candidate, so it is not reported.
            result = replay(retained_best_vector, endpoint_replaced=True, identifiability_warning=None)
            if isinstance(result, ScientificUnavailable):
                return result
    except (FittingCancelled, SimulationCancelled):
        if hard_cancelled() or not stop_check():
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
            ending="stopped",
            message=message,
        )
    vector = np.asarray(result.optimizer_vector, dtype=float)
    # Post-fit methods receive a fresh objective without progress emission.
    # It retains the same evaluators, candidate/fatal taxonomy, coordinates,
    # normalized weights, cancellation owner, and immutable captures.
    terminal_objective = new_objective()
    profile_result: ProfileLikelihoodResult | None = None
    optional_analysis_stopped = primary_stopped
    terminal_parameters = {
        key: float(vector[index])
        for index, key in enumerate(keys)
    }
    base_residuals = np.concatenate(
        [info.weighted_residuals for info in result.dataset_info]
    )
    base_residuals.setflags(write=False)
    # The noise model and the bound check evaluate on their own objective,
    # so their probes never change the penalty scale Profile refits use.
    noise = None
    noise_failure: Exception | None = None
    undetermined: tuple[_fit_execution.FitCoordinate, ...] = ()
    if not primary_stopped:
        try:
            # The post-fit stage starts here, so the progress says so
            # before the noise model and the bound check run.
            extension_progress_callback(FitExtensionProgress(status="running", stage="primary_complete"))
            analysis_objective = new_objective()
            noise = profile_noise_model(
                objective=analysis_objective, vector=vector, keys=keys,
                bounds=layout.bounds_dict(), dataset_info=result.dataset_info,
            )
            if noise is not None and noise.cutoffs:
                undetermined = undetermined_coordinates(
                    objective=analysis_objective, vector=vector, keys=keys,
                    bounds=layout.bounds_dict(),
                    base_residuals=base_residuals, cutoffs=noise.cutoffs,
                )
        except (FittingCancelled, SimulationCancelled):
            if hard_cancelled():
                raise
            optional_analysis_stopped = True
            noise = None
            undetermined = ()
        except Exception as exc:
            # A fatal evaluation next to the estimate leaves the fit itself
            # intact; Profile then reports the actual cause.
            logger.warning("Fit noise model could not be estimated.", exc_info=True)
            noise_failure = exc
            noise = None
            undetermined = ()
    # A parameter group's public uncertainty target is moved by the
    # optimizer as its coupled variable; analyses see the group identity.
    group_targets = {
        _fit_execution.FitCoordinate(
            "coupled",
            _fit_execution.parameter_group_variable_id(coordinate.name, coordinate.dataset_id),
        ): coordinate
        for coordinate in (
            () if profile_likelihood is None else profile_likelihood.parameter_names
        )
        if coordinate.scope == "group"
    }
    optimizer_keys = {target: key for key, target in group_targets.items()}
    if profile_likelihood is not None and optional_analysis_stopped:
        stopped_profile = profile_likelihood_settlement_result(
            config=profile_likelihood, status="stopped", cause=None,
        )
        profile_result = replace(
            stopped_profile,
            provenance={**stopped_profile.provenance, "settlement": "not_started_after_stop"},
        )
    elif profile_likelihood is not None and noise is None:
        # Profile needs the noise model; its own failure is Profile's cause.
        profile_result = profile_likelihood_settlement_result(
            config=profile_likelihood,
            status="failed",
            cause=(
                factual_cause_from_exception(noise_failure, domain="fit", origin="profile_likelihood")
                if noise_failure is not None
                else FactualCause(
                    domain="fit", origin="profile_likelihood", code=None,
                    message=(
                        "The noise of the data could not be estimated, because a candidate "
                        "next to the estimate is scientifically unavailable."
                    ),
                )
            ),
        )
    elif profile_likelihood is not None:
        try:
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
                ftol=ftol,
                xtol=xtol,
                base_residuals=base_residuals,
                noise=ProfileNoiseModel(noise.blocks, {
                    group_targets.get(key, key): cutoff for key, cutoff in noise.cutoffs.items()
                }),
                undetermined=frozenset(group_targets.get(key, key) for key in undetermined),
                cancellation_check=active_cancelled,
                progress_callback=extension_progress_callback,
            )
        except (FittingCancelled, SimulationCancelled):
            profile_result = profile_likelihood_settlement_result(
                config=profile_likelihood,
                status="stopped",
                cause=None,
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
            )

        profile_result = replace(profile_result, noise_blocks=noise.blocks)

    message = result.message
    if profile_result is not None and profile_result.status == "stopped":
        message += (
            " Profile likelihood was not started after Stop."
            if profile_result.provenance.get("settlement") == "not_started_after_stop"
            else " Profile likelihood stopped."
        )
    result = replace(
        result,
        message=message,
        profile_likelihood_result=profile_result,
        undetermined_values=undetermined,
        # Structural, so it holds after a stopped Fit too.
        rate_scale_undetermined=rate_scale_undetermined(
            prepared_dsl=prepared_dsl, fit_evaluators=evaluators, datasets=payloads,
            fixed_dataset_params=fixed_dataset_params, layout=layout, vector=vector,
        ),
    )
    if hard_cancelled():
        raise FittingCancelled()
    return result

