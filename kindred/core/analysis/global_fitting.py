"""Strict typed global-fit composition boundary."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import InitVar, dataclass, field, replace
from types import MappingProxyType
from typing import Literal, Optional

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
    ResidualBootstrapConfig,
    profile_likelihood_extension,
    residual_bootstrap_extension,
)
from kindred.core.analysis.fit_evaluation_outcome import FitEvaluationOutcome
from kindred.core.analysis import global_fit_execution as _fit_execution
from kindred.core.exceptions import (
    FitSimulationError,
    FittingCancelled,
    SimulationCancelled,
)
from kindred.core.fitting_optimization import fit_parameters
from kindred.core.fitting_completion import (
    FitExtensionProgress,
    FitTerminalExtension,
)
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
    "SharedRefitExecution",
    "SharedRefitPromotionAdmission",
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


@dataclass(frozen=True, slots=True)
class SharedRefitPromotionAdmission:
    """One pre-dispatch promotion with its exact admitted seed and provenance."""

    promotion: InitVar[SharedRefitPromotion]
    source_values: InitVar[Sequence[tuple[str, float]]]
    shared_variable: FitSharedVariableParamSpec = field(init=False)
    provenance: Mapping[str, object] = field(init=False)

    def __post_init__(
        self,
        promotion: SharedRefitPromotion,
        source_values: Sequence[tuple[str, float]],
    ) -> None:
        if not isinstance(promotion, SharedRefitPromotion):
            raise TypeError("Shared-refit admission requires a typed promotion.")
        source_values = tuple(
            (str(dataset_id or "").strip(), float(value))
            for dataset_id, value in source_values
        )
        dataset_ids = tuple(dataset_id for dataset_id, _value in source_values)
        values = np.asarray(
            [value for _dataset_id, value in source_values],
            dtype=float,
        )
        if (
            not source_values
            or any(not dataset_id for dataset_id in dataset_ids)
            or len(dataset_ids) != len(set(dataset_ids))
        ):
            raise ValueError(
                "Shared-refit admission requires distinct source dataset values."
            )
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
        shared_variable = FitSharedVariableParamSpec(
            name=promotion.parameter_name,
            initial=seed_value,
            minimum=promotion.minimum,
            maximum=promotion.maximum,
            log10=promotion.log10,
        )
        provenance = MappingProxyType(
            {
                "seed_statistic": promotion.seed_statistic,
                "seed_value": seed_value,
                "source_values": tuple(float(value) for value in values),
                "source_dataset_ids": dataset_ids,
                "source_scope": "per_dataset",
                "seed_authority": "pre-dispatch Shared-refit admission",
            }
        )
        object.__setattr__(self, "shared_variable", shared_variable)
        object.__setattr__(self, "provenance", provenance)


@dataclass(frozen=True, slots=True, init=False)
class SharedRefitExecution:
    """One admitted, immutable Shared-refit parameter execution.

    Promoted specifications are derived exclusively from their admissions and
    the optimizer layout is built exclusively from the resulting final
    parameter routes.  Callers therefore cannot supply either fact twice.
    """

    dataset_ids: tuple[str, ...]
    promotion_admissions: tuple[SharedRefitPromotionAdmission, ...]
    shared_variables: tuple[FitSharedVariableParamSpec, ...]
    dataset_overrides: tuple[FitDatasetParameterOverrides, ...]
    coupled_variables: tuple[FitCoupledVariableParamSpec, ...]
    optimizer_layout: _fit_execution.FitParameterLayout

    def __init__(
        self,
        *,
        dataset_ids: Sequence[str],
        promotion_admissions: Sequence[SharedRefitPromotionAdmission],
        existing_shared_variables: Sequence[FitSharedVariableParamSpec] = (),
        dataset_overrides: Sequence[FitDatasetParameterOverrides] = (),
        coupled_variables: Sequence[FitCoupledVariableParamSpec] = (),
    ) -> None:
        roster = tuple(str(value or "").strip() for value in dataset_ids)
        if (
            not roster
            or any(not value for value in roster)
            or len(roster) != len(set(roster))
        ):
            raise ValueError(
                "Shared-refit execution requires a distinct fitted dataset roster."
            )
        admitted = tuple(promotion_admissions)
        if not admitted or any(
            not isinstance(item, SharedRefitPromotionAdmission)
            for item in admitted
        ):
            raise ValueError(
                "Shared-refit execution requires typed promotion admissions."
            )
        promoted_names = tuple(item.shared_variable.name for item in admitted)
        if len(promoted_names) != len(set(promoted_names)):
            raise ValueError("Shared-refit execution repeats a promoted parameter.")
        for item in admitted:
            source_ids = tuple(item.provenance.get("source_dataset_ids", ()))
            if source_ids != roster:
                raise ValueError(
                    "Shared-refit admission source datasets must exactly match "
                    "the fitted dataset roster."
                )

        existing = tuple(existing_shared_variables)
        if any(
            not isinstance(item, FitSharedVariableParamSpec) for item in existing
        ):
            raise TypeError(
                "Shared-refit existing Shared parameters must use typed specs."
            )
        existing_names = tuple(item.name for item in existing)
        if (
            len(existing_names) != len(set(existing_names))
            or set(existing_names).intersection(promoted_names)
        ):
            raise ValueError(
                "Shared-refit promoted and existing Shared parameter routes overlap."
            )
        shared = (*existing, *(item.shared_variable for item in admitted))
        supplied_overrides = tuple(dataset_overrides)
        if any(
            not isinstance(item, FitDatasetParameterOverrides)
            for item in supplied_overrides
        ):
            raise TypeError(
                "Shared-refit dataset routes must use typed override records."
            )
        override_ids = tuple(item.dataset_id for item in supplied_overrides)
        if (
            len(override_ids) != len(set(override_ids))
            or set(override_ids) - set(roster)
        ):
            raise ValueError(
                "Shared-refit dataset routes must belong uniquely to its roster."
            )
        overrides_by_id = {
            item.dataset_id: item for item in supplied_overrides
        }
        overrides = tuple(
            overrides_by_id.get(
                dataset_id,
                FitDatasetParameterOverrides(dataset_id=dataset_id),
            )
            for dataset_id in roster
        )
        coupled = tuple(coupled_variables)
        if any(
            not isinstance(item, FitCoupledVariableParamSpec) for item in coupled
        ):
            raise TypeError(
                "Shared-refit coupled routes must use typed specifications."
            )
        stale_dataset_routes = set(promoted_names).intersection(
            name
            for override in overrides
            for name in (
                *(item.name for item in override.variable_params),
                *(item[0] for item in override.fixed_params),
            )
        )
        stale_coupled_routes = set(promoted_names).intersection(
            item.parameter_name for item in coupled
        )
        if stale_dataset_routes or stale_coupled_routes:
            raise ValueError(
                "Shared-refit execution retained an obsolete promoted parameter route."
            )
        shared_params = {item.name: item.initial for item in shared}
        shared_bounds = {
            item.name: (item.minimum, item.maximum) for item in shared
        }
        shared_log10 = {item.name: item.log10 for item in shared}
        layout = _fit_execution.build_parameter_layout_for_dataset_ids(
            dataset_ids=roster,
            shared_params=shared_params,
            dataset_overrides=overrides,
            coupled_variables=coupled,
            bounds=shared_bounds,
            log10_params=shared_log10,
        )
        object.__setattr__(self, "dataset_ids", roster)
        object.__setattr__(self, "promotion_admissions", admitted)
        object.__setattr__(self, "shared_variables", tuple(shared))
        object.__setattr__(self, "dataset_overrides", overrides)
        object.__setattr__(self, "coupled_variables", coupled)
        object.__setattr__(self, "optimizer_layout", layout)


def _shared_refit_provenance(
    execution: SharedRefitExecution,
    *,
    terminal_route: Literal[
        "optimizer_terminal",
        "user_stop_best_valid_bounded_replay",
    ],
) -> Mapping[str, object]:
    """Return admitted Shared-refit provenance for sole result construction."""

    if not isinstance(execution, SharedRefitExecution):
        raise TypeError("Shared-refit terminal requires its admitted execution.")
    admitted = execution.promotion_admissions
    names = tuple(item.shared_variable.name for item in admitted)
    if len(names) != len(set(names)):
        raise ValueError("Shared-refit terminal admissions repeat a parameter.")
    if terminal_route not in {
        "optimizer_terminal",
        "user_stop_best_valid_bounded_replay",
    }:
        raise ValueError("Shared-refit terminal route is invalid.")
    return {
        "execution": "Promote to Shared and Refit",
        "terminal_route": terminal_route,
        "promotions": {
            item.shared_variable.name: item.provenance
            for item in admitted
        },
        "starting_value_authority": "shared_refit_execution",
        "terminal_value_authority": "result.shared_params",
    }


def _validate_shared_refit_replay(
    *,
    shared_params: Mapping[str, float],
    dataset_params: Mapping[str, Mapping[str, float]],
    terminal_replay_count: int,
    completion: object,
    execution: SharedRefitExecution,
    terminal_route: Literal[
        "optimizer_terminal",
        "user_stop_best_valid_bounded_replay",
    ],
) -> None:
    """Validate replay custody before the sole result construction."""

    if tuple(dataset_params) != execution.dataset_ids:
        raise ValueError("Shared-refit replay dataset roster differs from admission.")
    expected_shared = set(execution.optimizer_layout.param_names)
    if set(shared_params) != expected_shared:
        raise ValueError("Shared-refit replay Shared routes differ from admission.")
    expected_dataset = {
        dataset_id: {
            *(item.name for item in override.variable_params),
            *(
                item.parameter_name
                for item in execution.coupled_variables
                if dataset_id in item.dataset_ids
            ),
        }
        for dataset_id, override in zip(
            execution.dataset_ids,
            execution.dataset_overrides,
            strict=True,
        )
    }
    if any(
        set(dataset_params[dataset_id]) != expected_dataset[dataset_id]
        for dataset_id in execution.dataset_ids
    ):
        raise ValueError("Shared-refit replay dataset routes differ from admission.")
    if int(terminal_replay_count) != 1:
        raise ValueError("A publishable Shared-refit replay requires one replay.")
    if terminal_route == "user_stop_best_valid_bounded_replay" and (
        getattr(completion, "status", None) != "warn"
        or bool(getattr(completion, "optimizer_converged", False))
    ):
        raise ValueError("User-Stop Shared-refit replay semantics are invalid.")




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
    profile_likelihood: ProfileLikelihoodConfig | None = None,
    residual_bootstrap: ResidualBootstrapConfig | None = None,
    extension_progress_callback: Optional[
        Callable[[FitExtensionProgress], None]
    ] = None,
    shared_refit_provenance: Mapping[str, object] | None = None,
    optimizer_layout: _fit_execution.FitParameterLayout,
 ) -> tuple[_fit_execution.GlobalFitResult | None, FitEvaluationOutcome]:
    """Fit immutable dataset captures through one evaluator per dataset."""

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
        if optimizer_evaluation.status == "scientific_unavailable":
            unavailability = optimizer_evaluation.unavailability
            if unavailability is None:
                raise RuntimeError(
                    "Scientific-unavailability terminal replay omitted its result."
                )
            return None, optimizer_evaluation
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
                FactualCause(
                    domain="fit",
                    origin="identifiability",
                    code=None,
                    type="FittingIdentifiability",
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
        layout.validate_optimizer_vector(vector)
        try:
            replay_state = _fit_execution._assemble_fit_replay_state(
                fit_evaluators=evaluators,
                payloads=payloads,
                layout=layout,
                candidate_vector=vector,
                fixed_dataset_params=fixed_dataset_params,
                weights=weights_norm,
                cancellation_check=cancellation_check,
                cancellation_control=cancellation_control,
                success=success,
                message=message,
                scipy_nfev=int(optimizer_result.scipy_nfev),
                terminal_replay_count=int(optimizer_result.terminal_replay_count),
                objective_call_count=int(objective.objective_calls),
                optimizer_diagnostic=diagnostic,
            )
        except FitSimulationError as exc:
            details = dict(getattr(exc, "details", None) or {})
            if not details.get("scientific_unavailable"):
                raise
            outcome = FitEvaluationOutcome.scientific_unavailable(
                ScientificUnavailable(
                    producer=str(details.get("producer", "global_fit_replay")),
                    stage=str(details.get("stage", "final_replay")),
                    details=details,
                    factual_cause=getattr(exc, "factual_cause", None),
                ),
                provenance={"stage": "optimizer_terminal_replay"},
            )
            return None, outcome
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
        extension_pipeline_stopped = False
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
                extension = profile_likelihood_extension(
                    objective=terminal_objective,
                    initial_parameters=terminal_parameters,
                    bounds=layout.bounds_dict(),
                    config=profile_config,
                    decode_coordinate=layout.decode_coordinate,
                    cancellation_check=cancellation_check,
                    terminal_converged=(
                        optimizer_evaluation.status == "converged"
                    ),
                    progress_callback=extension_progress_callback,
                )
            except (FittingCancelled, SimulationCancelled) as exc:
                extension_pipeline_stopped = True
                stop_cause = factual_cause_from_exception(
                    exc,
                    domain="fit",
                    origin="profile_likelihood",
                )
                extension = FitTerminalExtension(
                    kind="profile_likelihood",
                    status="stopped",
                    method="profile likelihood with nuisance reoptimization",
                    provenance={
                        "settlement": "stopped",
                        "primary_fit_preserved": True,
                    },
                    details={"primary_fit_preserved": True},
                    cause=stop_cause,
                )
            except Exception as exc:
                extension = FitTerminalExtension(
                    kind="profile_likelihood",
                    status="failed",
                    method="weighted-Gaussian profile likelihood",
                    provenance={
                        "settlement": "failed",
                        "primary_fit_preserved": True,
                    },
                    details={"primary_fit_preserved": True},
                    cause=factual_cause_from_exception(
                        exc,
                        domain="fit",
                        origin="profile_likelihood",
                    ),
                )
            extensions.append(extension)

        if residual_bootstrap is not None:
            def refit_bootstrap(
                synthetic_payloads: Sequence[FitDatasetSpec],
                replicate_index: int,
            ) -> tuple[_fit_execution.GlobalFitResult | None, FitEvaluationOutcome]:
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
                )

            if extension_pipeline_stopped:
                extension = FitTerminalExtension(
                    kind="bootstrap",
                    status="stopped",
                    method="stratified residual bootstrap with full refits",
                    provenance={
                        "settlement": "not_started_after_stop",
                        "primary_fit_preserved": True,
                    },
                    details={"primary_fit_preserved": True},
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
                    extension = residual_bootstrap_extension(
                        terminal_result=replay_state,
                        terminal_outcome=optimizer_evaluation,
                        payloads=payloads,
                        config=residual_bootstrap,
                        refit=refit_bootstrap,
                        cancellation_check=cancellation_check,
                        progress_callback=extension_progress_callback,
                    )
                except (FittingCancelled, SimulationCancelled) as exc:
                    extension_pipeline_stopped = True
                    extension = FitTerminalExtension(
                        kind="bootstrap",
                        status="stopped",
                        method="stratified residual bootstrap with full refits",
                        provenance={
                            "settlement": "stopped",
                            "primary_fit_preserved": True,
                        },
                        details={"primary_fit_preserved": True},
                        cause=factual_cause_from_exception(
                            exc, domain="fit", origin="bootstrap"
                        ),
                    )
                except Exception as exc:
                    extension = FitTerminalExtension(
                        kind="bootstrap",
                        status="failed",
                        method="stratified residual bootstrap with full refits",
                        provenance={
                            "settlement": "failed",
                            "primary_fit_preserved": True,
                        },
                        details={"primary_fit_preserved": True},
                        cause=factual_cause_from_exception(
                            exc,
                            domain="fit",
                            origin="bootstrap",
                        ),
                    )
            extensions.append(extension)

        result = _fit_execution._construct_global_fit_result(
            replay_state,
            terminal_extensions=extensions,
            shared_refit_provenance=shared_refit_provenance,
        )
        return result, optimizer_evaluation

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


def fit_global_shared_refit(
    *,
    execution: SharedRefitExecution,
    fit_evaluators: Mapping[str, object],
    datasets: Sequence[FitDatasetSpec],
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
    extension_progress_callback: Optional[
        Callable[[FitExtensionProgress], None]
    ] = None,
) -> tuple[_fit_execution.GlobalFitResult | None, FitEvaluationOutcome]:
    """Execute one pre-dispatch-admitted exact Shared refit.

    Promotion seeds, source provenance, final parameter routes, and optimizer
    layout have one owner in ``execution``.  Canonical terminal values still
    come exclusively from the completed Shared refit.
    """

    if not isinstance(execution, SharedRefitExecution):
        raise TypeError("Shared refit requires its admitted execution.")
    dataset_ids = tuple(item.dataset_id for item in datasets)
    if dataset_ids != execution.dataset_ids:
        raise ValueError(
            "Shared-refit fitted datasets must exactly match its admitted roster."
        )
    result = fit_global(
        fit_evaluators=fit_evaluators,
        datasets=datasets,
        shared_variables=execution.shared_variables,
        dataset_overrides=execution.dataset_overrides,
        coupled_variables=execution.coupled_variables,
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
        extension_progress_callback=extension_progress_callback,
        shared_refit_provenance=_shared_refit_provenance(
            execution,
            terminal_route="optimizer_terminal",
        ),
        optimizer_layout=execution.optimizer_layout,
    )
    return result
