"""Strict typed global-fit composition boundary."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Optional

import numpy as np

from kindred.core.analysis.dataset_parameter_overrides import (
    FitCoupledVariableParamSpec,
    FitDatasetParameterOverrides,
    FitSharedVariableParamSpec,
)
from kindred.core.analysis.fit_dataset_payload import FitDatasetSpec
from kindred.core.analysis.fit_uncertainty import (
    ProfileLikelihoodConfig,
    ResidualBootstrapConfig,
    profile_likelihood_extension,
    residual_bootstrap_extension,
)
from kindred.core.analysis.hierarchical_fit import (
    HierarchicalFitConfig,
    HierarchicalNuisanceParameter,
    hierarchical_fit_extension,
)
from kindred.core.analysis.fit_evaluation_outcome import FitEvaluationOutcome
from kindred.core.analysis import global_fit_execution as _fit_execution
from kindred.core.exceptions import (
    FittingCancelled,
    SimulationCancelled,
)
from kindred.core.fitting_optimization import fit_parameters
from kindred.core.fitting_completion import FitTerminalExtension
from kindred.core.objective import ObjectiveContext
from kindred.core.project_document import validate_fit_method
from kindred.core.simulation_failure import build_simulation_failure
from kindred.core.simulator.solver_types import SimulationCancellationControl

logger = logging.getLogger(__name__)

__all__ = [
    "SharedRefitPromotion",
    "fit_global",
    "fit_global_shared_refit",
]


@dataclass(frozen=True, slots=True)
class SharedRefitPromotion:
    """One Per-dataset coordinate promoted into a new Shared refit."""

    parameter_name: str
    minimum: float
    maximum: float
    log10: bool = False
    seed_statistic: Literal["arithmetic_mean", "geometric_mean"] = "arithmetic_mean"

    def __post_init__(self) -> None:
        name = str(self.parameter_name or "").strip()
        minimum = float(self.minimum)
        maximum = float(self.maximum)
        if not name:
            raise ValueError("Shared-refit parameter name must be nonblank.")
        if not np.isfinite(minimum) or not np.isfinite(maximum) or minimum >= maximum:
            raise ValueError("Shared-refit promotion requires finite minimum < maximum.")
        if self.log10 and minimum <= 0.0:
            raise ValueError("Log10 Shared-refit promotion requires positive bounds.")
        if self.seed_statistic not in {"arithmetic_mean", "geometric_mean"}:
            raise ValueError("Shared-refit seed statistic must be arithmetic or geometric.")
        object.__setattr__(self, "parameter_name", name)
        object.__setattr__(self, "minimum", minimum)
        object.__setattr__(self, "maximum", maximum)
        object.__setattr__(self, "log10", bool(self.log10))


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
    payloads = list(datasets)
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

    shared = tuple(shared_variables)
    shared_names = [item.name for item in shared]
    if len(shared_names) != len(set(shared_names)):
        raise ValueError("Shared Fit parameter names must be unique.")

    overrides = tuple(dataset_overrides)
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


def fit_global(
    *,
    fit_evaluators: Mapping[str, object],
    datasets: Sequence[FitDatasetSpec],
    shared_variables: Sequence[FitSharedVariableParamSpec],
    dataset_overrides: Sequence[FitDatasetParameterOverrides] = (),
    coupled_variables: Sequence[FitCoupledVariableParamSpec] = (),
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
    hierarchical: Sequence[HierarchicalFitConfig] = (),
) -> _fit_execution.GlobalFitResult:
    """Fit immutable dataset captures through one evaluator per dataset."""

    method = validate_fit_method(method)
    payloads, evaluators, shared, overrides = _validate_run_inputs(
        fit_evaluators=fit_evaluators,
        datasets=datasets,
        shared_variables=shared_variables,
        dataset_overrides=dataset_overrides,
    )
    coupled = tuple(coupled_variables)
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
    layout = _fit_execution.build_parameter_layout(
        payloads=payloads,
        shared_params=shared_params,
        dataset_overrides=overrides,
        coupled_variables=coupled,
        bounds=shared_bounds,
        log10_params=shared_log10,
    )

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
            shared_params=shared_params,
            dataset_params=fixed_dataset_params,
            weights=weights_norm,
            layout=layout,
            penalty_value=penalty_value,
            ctx=context,
            progress_callback=progress_callback,
            cancellation_check=cancellation_check,
            cancellation_control=cancellation_control,
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
        optimizer_evaluation = optimizer_result.evaluation
        if not isinstance(optimizer_evaluation, FitEvaluationOutcome):
            raise TypeError("Fit optimizer omitted its typed terminal evaluation outcome.")
        if optimizer_evaluation.status == "cancelled":
            cause = optimizer_evaluation.cause
            if isinstance(cause, (FittingCancelled, SimulationCancelled)):
                raise FittingCancelled() from cause
            raise FittingCancelled()
        if optimizer_evaluation.status == "fatal":
            cause = optimizer_evaluation.cause
            if isinstance(cause, BaseException):
                raise cause
            raise RuntimeError("Fatal Fit evaluation omitted its typed cause.")
        success = bool(
            optimizer_result.success
            and optimizer_evaluation.status == "converged"
        )
        message = str(optimizer_result.message)
        diagnostic = None
        identifiability = str(
            optimizer_result.identifiability_warning or ""
        ).strip()
        if identifiability:
            diagnostic = _fit_execution.build_fit_diagnostic_from_failure(
                build_simulation_failure(
                    kind="fitting_identifiability",
                    message=identifiability,
                    details={"final_report_warning": True},
                ),
                phase="optimizer",
            )
        if context.last_error is not None and diagnostic is None:
            provenance = context.last_error_provenance
            dataset_id = (
                str(provenance.get("dataset"))
                if isinstance(provenance, dict) and provenance.get("dataset") is not None
                else None
            )
            diagnostic = _fit_execution.build_fit_diagnostic_from_exception(
                context.last_error,
                phase="optimizer",
                dataset_id=dataset_id,
            )
        keys = layout.opt_param_keys()
        vector = np.asarray(
            [optimizer_result.parameters[key] for key in keys],
            dtype=float,
        )
        fitted_shared = layout.shared_param_dict_from_vector(vector)
        combined_dataset_params = {
            item.dataset_id: {}
            for item in payloads
        }
        for dataset_id, parameter_name in layout.dataset_var_order:
            combined_dataset_params[dataset_id].update(
                layout.dataset_var_params_for_dataset(dataset_id, vector)
            )
        for item in layout.coupled_var_order:
            for dataset_id in item.dataset_ids:
                combined_dataset_params[dataset_id].update(
                    layout.coupled_params_for_dataset(dataset_id, vector)
                )

        result = _fit_execution.assemble_global_fit_result(
            fit_evaluators=evaluators,
            payloads=payloads,
            layout=layout,
            fitted_params=fitted_shared,
            fixed_dataset_params=fixed_dataset_params,
            combined_dataset_params=combined_dataset_params,
            weights=weights_norm,
            cancellation_check=cancellation_check,
            cancellation_control=cancellation_control,
            success=success,
            message=message,
            nfev=int(optimizer_result.nfev),
            optimizer_diagnostic=diagnostic,
        )
        result.scipy_nfev = int(optimizer_result.scipy_nfev or optimizer_result.nfev)
        result.terminal_replay_count = int(optimizer_result.terminal_replay_count)
        result.objective_call_count = int(objective.objective_calls)
        hierarchical_configs = tuple(hierarchical)
        if any(
            not isinstance(item, HierarchicalFitConfig)
            for item in hierarchical_configs
        ):
            raise TypeError(
                "Hierarchical Fit requests must contain HierarchicalFitConfig values."
            )
        if (
            profile_likelihood is None
            and residual_bootstrap is None
            and not hierarchical_configs
        ):
            return result

        # Post-fit methods receive a fresh objective without progress emission.
        # It retains the same evaluators, candidate/fatal taxonomy, coordinates,
        # normalized weights, cancellation owner, and immutable captures.
        terminal_objective = _fit_execution.GlobalFitObjective(
            fit_evaluators=evaluators,
            payloads=payloads,
            shared_params=shared_params,
            dataset_params=fixed_dataset_params,
            weights=weights_norm,
            layout=layout,
            penalty_value=penalty_value,
            ctx=ObjectiveContext(),
            progress_callback=None,
            cancellation_check=cancellation_check,
            cancellation_control=cancellation_control,
        )
        extensions: list[FitTerminalExtension] = []
        terminal_parameters = {
            key: float(optimizer_result.parameters[key])
            for key in keys
        }
        if profile_likelihood is not None:
            extensions.append(
                profile_likelihood_extension(
                    objective=terminal_objective,
                    initial_parameters=terminal_parameters,
                    bounds=layout.bounds_dict(),
                    config=profile_likelihood,
                    decode_coordinate=layout.decode_coordinate,
                    cancellation_check=cancellation_check,
                    terminal_converged=(
                        optimizer_evaluation.status == "converged"
                    ),
                )
            )

        if residual_bootstrap is not None:
            def refit_bootstrap(
                synthetic_payloads: Sequence[FitDatasetSpec],
            ) -> _fit_execution.GlobalFitResult:
                return fit_global(
                    fit_evaluators=evaluators,
                    datasets=synthetic_payloads,
                    shared_variables=shared,
                    dataset_overrides=overrides,
                    coupled_variables=coupled,
                    weights=weights_norm,
                    method=method,
                    max_nfev=max_nfev,
                    ftol=ftol,
                    xtol=xtol,
                    seed=seed,
                    progress_callback=None,
                    cancellation_check=cancellation_check,
                    cancellation_control=cancellation_control,
                )

            extensions.append(
                residual_bootstrap_extension(
                    terminal_result=result,
                    payloads=payloads,
                    config=residual_bootstrap,
                    refit=refit_bootstrap,
                    cancellation_check=cancellation_check,
                )
            )

        payloads_by_id = {
            payload.dataset_id: payload
            for payload in payloads
        }
        base_vector = np.asarray(
            [terminal_parameters[key] for key in keys],
            dtype=float,
        )
        for hierarchy in hierarchical_configs:
            coordinate_names = {
                dataset_id: f"{dataset_id}::{hierarchy.parameter_name}"
                for dataset_id in payloads_by_id
            }
            missing_coordinates = tuple(
                coordinate
                for coordinate in coordinate_names.values()
                if coordinate not in terminal_parameters
            )
            hierarchy_provenance: dict[str, object] = {
                "likelihood_model": "weighted Gaussian observation residuals",
                "random_effects": "independent Gaussian dataset effects",
                "marginalization": "first-order Laplace approximation at conditional modes",
                "optimizer_scale": "log10" if hierarchy.log10 else "linear",
                "optimizer_method": hierarchy.optimizer_method,
                "max_nfev": hierarchy.max_nfev,
                "inner_max_nfev": hierarchy.inner_max_nfev,
                "joint_nuisance_reoptimization": True,
                "dataset_specific_evaluation": True,
                "fit_local_only": True,
                "publication_requires_shared_refit": True,
                "primary_evaluation_status": optimizer_evaluation.status,
            }
            if missing_coordinates:
                extensions.append(
                    FitTerminalExtension(
                        kind="hierarchical",
                        status="unavailable",
                        method=(
                            "nonlinear mixed-effects weighted-Gaussian likelihood "
                            "with Gaussian random effects and Laplace marginalization"
                        ),
                        provenance=hierarchy_provenance,
                        details={
                            "parameter_name": hierarchy.parameter_name,
                            "missing_optimizer_coordinates": missing_coordinates,
                        },
                        unavailable_reason=(
                            "Hierarchical scope requires one active Per-dataset "
                            "optimizer coordinate for every included dataset."
                        ),
                    )
                )
                continue
            if any(
                layout.coordinate_log10(coordinate) != hierarchy.log10
                for coordinate in coordinate_names.values()
            ):
                raise ValueError(
                    f"Hierarchical scale for {hierarchy.parameter_name!r} does not "
                    "match its configured dataset coordinates."
                )

            random_coordinates = frozenset(coordinate_names.values())
            nuisance_parameters = tuple(
                HierarchicalNuisanceParameter(
                    name=coordinate,
                    initial=layout.decode_coordinate(
                        coordinate,
                        terminal_parameters[coordinate],
                    ),
                    minimum=layout.decode_coordinate(
                        coordinate,
                        layout.bounds_dict()[coordinate][0],
                    ),
                    maximum=layout.decode_coordinate(
                        coordinate,
                        layout.bounds_dict()[coordinate][1],
                    ),
                    log10=layout.coordinate_log10(coordinate),
                )
                for coordinate in keys
                if coordinate not in random_coordinates
            )
            nuisance_names = frozenset(
                item.name
                for item in nuisance_parameters
            )
            dataset_objectives = {
                dataset_id: _fit_execution.GlobalFitObjective(
                    fit_evaluators={dataset_id: evaluators[dataset_id]},
                    payloads=[payloads_by_id[dataset_id]],
                    shared_params=shared_params,
                    dataset_params=fixed_dataset_params,
                    weights={dataset_id: weights_norm[dataset_id]},
                    layout=layout,
                    penalty_value=penalty_value,
                    ctx=ObjectiveContext(),
                    progress_callback=None,
                    cancellation_check=cancellation_check,
                    cancellation_control=cancellation_control,
                )
                for dataset_id in payloads_by_id
            }

            def hierarchical_dataset_evaluation(
                dataset_id: str,
                physical_parameters: Mapping[str, float],
                *,
                _coordinate_names: Mapping[str, str] = coordinate_names,
                _nuisance_names: frozenset[str] = nuisance_names,
                _dataset_objectives: Mapping[
                    str,
                    _fit_execution.GlobalFitObjective,
                ] = dataset_objectives,
            ) -> FitEvaluationOutcome:
                if dataset_id not in _dataset_objectives:
                    raise KeyError(dataset_id)
                coordinate = _coordinate_names[dataset_id]
                expected_coordinates = _nuisance_names | {coordinate}
                if set(physical_parameters) != expected_coordinates:
                    raise ValueError(
                        "Hierarchical dataset evaluation received a mismatched "
                        "optimizer coordinate roster."
                    )
                candidate = base_vector.copy()
                for name, physical_value in physical_parameters.items():
                    candidate[layout.coordinate_index(name)] = (
                        layout.encode_coordinate(name, physical_value)
                    )
                return _dataset_objectives[dataset_id].evaluate_outcome(candidate)

            extensions.append(
                hierarchical_fit_extension(
                    dataset_ids=tuple(payloads_by_id),
                    config=hierarchy,
                    dataset_evaluator=hierarchical_dataset_evaluation,
                    primary_evaluation=optimizer_evaluation,
                    nuisance_parameters=nuisance_parameters,
                    initial_effects={
                        dataset_id: result.dataset_params[dataset_id][
                            hierarchy.parameter_name
                        ]
                        for dataset_id in payloads_by_id
                    },
                    cancellation_check=cancellation_check,
                )
            )
        return _fit_execution.attach_terminal_extensions(result, extensions)
    except FittingCancelled:
        raise
    except SimulationCancelled as exc:
        raise FittingCancelled() from exc


def fit_global_shared_refit(
    *,
    previous_result: _fit_execution.GlobalFitResult,
    promotions: Sequence[SharedRefitPromotion],
    fit_evaluators: Mapping[str, object],
    datasets: Sequence[FitDatasetSpec],
    existing_shared_variables: Sequence[FitSharedVariableParamSpec] = (),
    dataset_overrides: Sequence[FitDatasetParameterOverrides] = (),
    coupled_variables: Sequence[FitCoupledVariableParamSpec] = (),
    weights: Optional[Mapping[str, float]] = None,
    method: str,
    max_nfev: int = 1000,
    ftol: float = 1e-10,
    xtol: float = 1e-10,
    seed: Optional[int] = None,
    progress_callback: Optional[Callable[[_fit_execution.FitProgress], None]] = None,
    cancellation_check: Optional[Callable[[], bool]] = None,
    cancellation_control: SimulationCancellationControl | None = None,
    profile_likelihood: ProfileLikelihoodConfig | None = None,
    residual_bootstrap: ResidualBootstrapConfig | None = None,
) -> _fit_execution.GlobalFitResult:
    """Promote Per-dataset coordinates and execute a new exact Shared refit.

    The labelled central tendency is used only as the new optimizer starting
    value.  This function never returns or publishes that seed: its canonical
    parameter values come exclusively from the completed Shared refit.
    """

    if not isinstance(previous_result, _fit_execution.GlobalFitResult):
        raise TypeError("Shared refit requires a completed previous GlobalFitResult.")
    requested = tuple(promotions)
    if not requested or any(
        not isinstance(item, SharedRefitPromotion) for item in requested
    ):
        raise ValueError("Shared refit requires typed promotion requests.")
    names = tuple(item.parameter_name for item in requested)
    if len(names) != len(set(names)):
        raise ValueError("Shared refit repeats a promoted parameter.")
    existing = tuple(existing_shared_variables)
    overlap = set(names).intersection(item.name for item in existing)
    if overlap:
        raise ValueError(
            "Shared refit promotions duplicate existing Shared coordinates: "
            + ", ".join(sorted(overlap))
        )

    payload_ids = tuple(item.dataset_id for item in datasets)
    if set(payload_ids) != set(previous_result.dataset_params):
        raise ValueError(
            "Shared refit requires the same executed included dataset roster as "
            "the Per-dataset terminal candidate."
        )
    configured_dataset_names = {
        name
        for override in dataset_overrides
        for name in (
            *(item.name for item in override.variable_params),
            *(item[0] for item in override.fixed_params),
        )
    }
    coupled_names = {item.parameter_name for item in coupled_variables}
    stale_routes = set(names).intersection(configured_dataset_names | coupled_names)
    if stale_routes:
        raise ValueError(
            "Promoted Shared parameters still have a Per-dataset/coupled route: "
            + ", ".join(sorted(stale_routes))
        )

    promoted_variables: list[FitSharedVariableParamSpec] = []
    seed_provenance: dict[str, object] = {}
    for promotion in requested:
        try:
            values = np.asarray(
                [
                    float(previous_result.dataset_params[dataset_id][promotion.parameter_name])
                    for dataset_id in payload_ids
                ],
                dtype=float,
            )
        except KeyError as exc:
            raise ValueError(
                f"Previous Fit omitted Per-dataset estimate {promotion.parameter_name!r}."
            ) from exc
        if not np.all(np.isfinite(values)):
            raise ValueError("Shared-refit seeds require finite Per-dataset estimates.")
        if promotion.seed_statistic == "geometric_mean":
            if np.any(values <= 0.0):
                raise ValueError(
                    f"Geometric Shared-refit seed for {promotion.parameter_name!r} "
                    "requires positive estimates."
                )
            seed_value = float(np.exp(np.mean(np.log(values))))
        else:
            seed_value = float(np.mean(values))
        if not promotion.minimum <= seed_value <= promotion.maximum:
            raise ValueError(
                f"{promotion.parameter_name!r} central-tendency seed lies outside "
                "the new Shared bounds."
            )
        promoted_variables.append(
            FitSharedVariableParamSpec(
                name=promotion.parameter_name,
                initial=seed_value,
                minimum=promotion.minimum,
                maximum=promotion.maximum,
                log10=promotion.log10,
            )
        )
        seed_provenance[promotion.parameter_name] = {
            "seed_statistic": promotion.seed_statistic,
            "seed_value": seed_value,
            "source_values": tuple(float(value) for value in values),
            "source_scope": "per_dataset",
            "terminal_authority": "completed Shared refit only",
        }

    result = fit_global(
        fit_evaluators=fit_evaluators,
        datasets=datasets,
        shared_variables=existing + tuple(promoted_variables),
        dataset_overrides=dataset_overrides,
        coupled_variables=coupled_variables,
        weights=weights,
        method=method,
        max_nfev=max_nfev,
        ftol=ftol,
        xtol=xtol,
        seed=seed,
        progress_callback=progress_callback,
        cancellation_check=cancellation_check,
        cancellation_control=cancellation_control,
        profile_likelihood=profile_likelihood,
        residual_bootstrap=residual_bootstrap,
    )
    result.shared_refit_provenance = MappingProxyType(
        {
            "execution": "Promote to Shared and Refit",
            "previous_candidate_invalidated": True,
            "promotions": seed_provenance,
            "publication_authority": "result.shared_params from this completed refit",
        }
    )
    return result
