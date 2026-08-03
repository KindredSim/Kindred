"""Strict typed global-fit composition boundary."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from typing import Optional

import numpy as np

from kindred.core.analysis.dataset_parameter_overrides import (
    FitCoupledVariableParamSpec,
    FitDatasetParameterOverrides,
    FitSharedVariableParamSpec,
)
from kindred.core.analysis.fit_dataset_payload import FitDatasetSpec
from kindred.core.analysis.global_fit_execution import (
    DatasetFitInfo,
    GlobalFitObjective,
    GlobalFitResult,
    assemble_global_fit_result,
    build_completion_detail_sections,
    build_fit_diagnostic_from_exception,
    build_fit_diagnostic_from_failure,
    build_parameter_layout,
    ensure_failure_diagnostic,
    normalize_weights,
)
from kindred.core.exceptions import (
    FitSimulationError,
    FittingCancelled,
    SimulationCancelled,
)
from kindred.core.fitting_completion import (
    FitDetailSection,
    FitDiagnostic,
    GlobalFitCompletion,
)
from kindred.core.fitting_optimization import fit_parameters
from kindred.core.objective import ObjectiveContext
from kindred.core.simulation_failure import build_simulation_failure

logger = logging.getLogger(__name__)

__all__ = [
    "DatasetFitInfo",
    "FitDetailSection",
    "FitDiagnostic",
    "GlobalFitCompletion",
    "GlobalFitResult",
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
    method: str = "trf",
    max_nfev: int = 1000,
    ftol: float = 1e-10,
    xtol: float = 1e-10,
    seed: Optional[int] = None,
    progress_callback: Optional[
        Callable[[int, float, dict[str, float]], None]
    ] = None,
    cancellation_check: Optional[Callable[[], bool]] = None,
) -> GlobalFitResult:
    """Fit immutable dataset captures through one evaluator per dataset."""

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
    weights_norm = normalize_weights(
        payloads,
        None if weights is None else dict(weights),
    )
    layout = build_parameter_layout(
        payloads=payloads,
        shared_params=shared_params,
        dataset_overrides=overrides,
        coupled_variables=coupled,
        bounds=shared_bounds,
        log10_params=shared_log10,
    )

    method_key = str(method or "trf").strip().lower()
    if method_key in {"differential_evolution", "de"}:
        method_key = "de"
    elif method_key not in {"lm", "trf", "dogbox"}:
        raise ValueError(f"Unknown optimization method: {method}")

    logger.info(
        "Global fitting: %d datasets, %d optimizer coordinates",
        len(payloads),
        len(layout.opt_param_keys()),
    )
    penalty_value = 1e6
    objective_total_points = int(sum(item.point_count for item in payloads))
    context = ObjectiveContext()
    objective: Optional[GlobalFitObjective] = None
    optimizer_completed = False

    def failed_result(
        message: str,
        *,
        diagnostic: Optional[FitDiagnostic] = None,
    ) -> GlobalFitResult:
        chosen = ensure_failure_diagnostic(
            diagnostic,
            message=str(message or "Global fit failed before replay completed."),
        )
        return GlobalFitResult(
            shared_params=dict(shared_params),
            dataset_params={
                item.dataset_id: dict(fixed_dataset_params.get(item.dataset_id, {}))
                for item in payloads
            },
            uncertainties=None,
            weighted_residual_sse=np.inf,
            weighted_residual_mse=np.inf,
            dataset_info=[],
            nfev=int(getattr(objective, "_iteration", 0)),
            message=str(message),
            completion=GlobalFitCompletion(
                status="fail",
                optimizer_converged=False,
                nonfinite_metrics=False,
                optimizer_diagnostic=chosen,
                dataset_failures={},
                dataset_warnings={},
                detail_sections=build_completion_detail_sections(
                    status="fail",
                    optimizer_diagnostic=chosen,
                    dataset_failures={},
                ),
            ),
        )

    try:
        objective = GlobalFitObjective(
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
        )
        de_penalty = (
            float((penalty_value**2) * max(1, objective_total_points))
            if method_key == "de"
            else None
        )
        optimizer_result = fit_parameters(
            objective,
            layout.initial_params(),
            bounds=layout.bounds_dict(),
            method=method_key,
            progress_callback=None,
            max_nfev=max_nfev,
            seed=seed,
            ftol=ftol,
            xtol=xtol,
            cancellation_check=cancellation_check,
            de_penalty=de_penalty,
        )
        optimizer_completed = True
        success = bool(optimizer_result.success)
        message = str(optimizer_result.message)
        diagnostic: Optional[FitDiagnostic] = None
        identifiability = str(
            optimizer_result.identifiability_warning or ""
        ).strip()
        if identifiability:
            diagnostic = build_fit_diagnostic_from_failure(
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
            diagnostic = build_fit_diagnostic_from_exception(
                context.last_error,
                phase="optimizer",
                dataset_id=dataset_id,
            )
        if (
            not success
            and str(optimizer_result.parameter_space) == "physical"
        ):
            return failed_result(message, diagnostic=diagnostic)

        keys = layout.opt_param_keys()
        vector = np.asarray(
            [optimizer_result.parameters[key] for key in keys],
            dtype=float,
        )
        fitted_shared = layout.shared_param_dict_from_vector(vector)
        combined_dataset_params = {
            item.dataset_id: dict(fixed_dataset_params.get(item.dataset_id, {}))
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

        covariance = optimizer_result.covariance
        uncertainties = None
        if covariance is not None:
            try:
                deviations = np.sqrt(np.diag(covariance))
                uncertainties = {}
                for index, name in enumerate(layout.param_names):
                    if index >= deviations.size:
                        break
                    deviation = float(deviations[index])
                    if layout.shared_log10.get(name):
                        uncertainties[name] = float(
                            np.log(10.0) * fitted_shared[name] * deviation
                        )
                    else:
                        uncertainties[name] = deviation
            except (IndexError, TypeError, ValueError):
                logger.debug("Shared uncertainty projection failed.", exc_info=True)

        return assemble_global_fit_result(
            fit_evaluators=evaluators,
            payloads=payloads,
            layout=layout,
            fitted_params=fitted_shared,
            combined_dataset_params=combined_dataset_params,
            weights=weights_norm,
            penalty_value=penalty_value,
            cancellation_check=cancellation_check,
            success=success,
            message=message,
            nfev=int(optimizer_result.nfev),
            covariance=covariance,
            objective_residuals=np.asarray(
                optimizer_result.residuals,
                dtype=float,
            ).reshape(-1),
            uncertainties=uncertainties,
            optimizer_diagnostic=diagnostic,
        )
    except FittingCancelled:
        raise
    except FitSimulationError as exc:
        if optimizer_completed:
            raise
        logger.error("Global fitting failed: %s", exc, exc_info=False)
        return failed_result(
            str(exc),
            diagnostic=build_fit_diagnostic_from_exception(
                exc,
                phase="fatal",
                parameter_snapshot=exc.failed_params,
            ),
        )
    except Exception as exc:
        if isinstance(exc, (FittingCancelled, SimulationCancelled)):
            raise FittingCancelled() from exc
        if optimizer_completed:
            raise
        logger.error("Global fitting failed: %s", exc, exc_info=True)
        return failed_result(
            f"Fitting failed: {exc}",
            diagnostic=build_fit_diagnostic_from_exception(exc, phase="fatal"),
        )
