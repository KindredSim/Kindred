"""
Global fitting for chemical kinetics - fit single mechanism to multiple datasets.

This module provides functionality to fit a single reaction mechanism to multiple
experimental datasets simultaneously. Supports:
- Shared parameters (common across all datasets)
- Dataset-specific parameters (e.g., different initial conditions)
- Weighted residuals across datasets
- Comprehensive per-dataset and global statistics
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from kindred.core.analysis.dataset_parameter_overrides import (
    coerce_fit_dataset_parameter_overrides,
    split_fit_dataset_parameter_overrides,
)
from kindred.core.analysis.global_fit_execution import (
    DatasetFitInfo,
    GlobalFitObjective,
    GlobalFitResult,
    assemble_global_fit_result,
    build_dataset_payloads,
    build_completion_detail_sections,
    build_fit_diagnostic_from_exception,
    build_parameter_layout,
    ensure_failure_diagnostic,
    normalize_input_datasets,
    normalize_weights,
)
from kindred.core.exceptions import FittingCancelled, FitSimulationError, SimulationCancelled
from kindred.core.fitting_completion import FitDiagnostic, FitDetailSection, GlobalFitCompletion
from kindred.core.fitting_optimization import fit_parameters
from kindred.core.fitting_evaluation import SerialFittingEvaluator, coerce_fitting_series_evaluator
from kindred.core.fitting_runtime_session import FittingRuntimeSession
from kindred.core.objective import ObjectiveContext, ObjectiveWrapper

logger = logging.getLogger(__name__)

__all__ = [
    "GlobalFitResult",
    "DatasetFitInfo",
    "GlobalFitCompletion",
    "FitDiagnostic",
    "FitDetailSection",
    "fit_global",
]


def fit_global(
    fit_evaluator,
    datasets: List[object],
    shared_params: Dict[str, float],
    dataset_params: Optional[Dict[str, Dict[str, float]]] = None,
    dataset_variable_params: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None,
    bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    weights: Optional[Dict[str, float]] = None,
    method: str = "trf",
    max_nfev: int = 1000,
    ftol: float = 1e-10,
    xtol: float = 1e-10,
    seed: Optional[int] = None,
    log10_params: Optional[Dict[str, bool]] = None,
    progress_callback: Optional[Callable[[int, float, Dict[str, float]], None]] = None,
    cancellation_check: Optional[Callable[[], bool]] = None,
    dataset_overrides: Optional[List[object]] = None,
    runtime_session: Optional[FittingRuntimeSession] = None,
    max_runtime_lanes: Optional[int] = None,
    runtime_ledger: Optional[object] = None,
) -> GlobalFitResult:
    """
    Fit single mechanism to multiple experimental datasets simultaneously.

    This function optimizes a single set of shared parameters (e.g., rate constants)
    to fit multiple datasets, while allowing dataset-specific parameters (e.g.,
    initial conditions). The objective is a weighted sum of residuals across all
    datasets.

    Parameters
    ----------
    fit_evaluator : object
        Shared fitting evaluation boundary. Must expose
        `evaluate_series(params: Dict[str, float]) -> SimulationSeriesPayload`,
        or be a callable that returns a compatible simulation-series payload.
    datasets : list of dict
        List of experimental datasets. Each dict must contain:
        - 't': np.ndarray of time points
        - 'y': np.ndarray of experimental values
        - 'species': str, target species name
        - 'id': str, unique identifier (optional, defaults to index)
    shared_params : dict
        Initial values for shared parameters {name: value}
    dataset_params : dict of dict, optional
        Fixed dataset-specific parameters {dataset_id: {name: value}}
        (e.g., required initial concentrations)
    dataset_variable_params : dict of dict, optional
        Dataset parameters to be optimized {dataset_id: {name: {"initial": float, "min": float, "max": float}}}
        Useful for fitting per-dataset initial conditions with bounds.
    bounds : dict, optional
        Parameter bounds {name: (min, max)} for shared parameters only
    weights : dict, optional
        Dataset weights {dataset_id: weight}. Defaults to 1/n_points for each dataset
    method : str
        Optimization method ('trf', 'dogbox', 'lm', 'de')
    max_nfev : int
        Maximum number of function evaluations
    seed : int, optional
        Random seed for differential evolution (reproducibility).
    log10_params : dict, optional
        Per-parameter toggle for log10-space fitting of shared parameters.
        Keys are shared parameter names; values are True/False. When enabled,
        bounds/initial values must be strictly positive and are transformed to
        log10-space internally.
    progress_callback : callable, optional
        Called with (iteration, cost, params_dict) periodically and on best-so-far
        improvements. When dataset-specific variable parameters are being optimized,
        they are included in `params_dict` with a unique key of the form
        `"{dataset_id}::{param_name}"` to avoid collisions across datasets.
    cancellation_check : callable, optional
        Function returning True when fitting should abort early.
    runtime_session : FittingRuntimeSession, optional
        Internal runtime-controller hook for callers that already own reusable
        fitting evaluator lanes for this prepared runtime identity.
    max_runtime_lanes : int, optional
        Internal lane-budget hint used when this function creates its own
        runtime session for an exact SerialFittingEvaluator.
    runtime_ledger : object, optional
        Internal deterministic ledger object used by runtime-session tests and
        GUI controller diagnostics.

    Returns
    -------
    GlobalFitResult
        Comprehensive fitting results including per-dataset statistics

    Examples
    --------
    >>> # Fit rate constant to two datasets with different initial conditions
    >>> def sim(params):
    ...     k = params['k']
    ...     # Run simulation with k...
    ...     return {'A': concentrations}
    >>>
    >>> datasets = [
    ...     {'id': 'exp1', 't': t1, 'y': y1, 'species': 'A'},
    ...     {'id': 'exp2', 't': t2, 'y': y2, 'species': 'A'},
    ... ]
    >>> shared = {'k': 0.1}  # Initial guess
    >>> result = fit_global(sim, datasets, shared)
    >>> print(f"Optimal k: {result.shared_params['k']:.3f}")
    >>> print(f"Global R²: {result.global_r_squared:.3f}")

    Notes
    -----
    - Datasets may be single-species (`species: str`, `y: (N,)`) or multi-species
      (`species: list[str]`, `y: (S,N)` where rows align to `species` order).
    - Weights default to inverse of number of points (equal contribution per dataset).
    - Dataset-specific params are fixed (not optimized) - useful for IC variations.
    - Returns comprehensive per-dataset statistics for validation.
    """
    datasets = normalize_input_datasets(datasets)
    payloads = build_dataset_payloads(datasets)

    overrides = coerce_fit_dataset_parameter_overrides(
        dataset_ids=[payload.dataset_id for payload in payloads],
        dataset_overrides=dataset_overrides,
        dataset_params=dataset_params,
        dataset_variable_params=dataset_variable_params,
    )
    dataset_params_map_raw, dataset_variable_params_map_raw = split_fit_dataset_parameter_overrides(overrides)
    dataset_params_map: Dict[str, Dict[str, float]] = {
        ds_id: {name: float(value) for name, value in param_map.items()}
        for ds_id, param_map in dataset_params_map_raw.items()
    }
    dataset_variable_params_map: Dict[str, Dict[str, Dict[str, float]]] = {
        ds_id: {
            name: {
                "initial": float(spec["initial"]),
                "min": float(spec["min"]),
                "max": float(spec["max"]),
                "log10": bool(spec.get("log10", False)),
            }
            for name, spec in param_map.items()
        }
        for ds_id, param_map in dataset_variable_params_map_raw.items()
    }

    weights_norm = normalize_weights(payloads, weights)

    logger.info("Global fitting: %d datasets, %d shared parameters", len(payloads), len(shared_params))
    for payload in payloads:
        species_label = ", ".join(payload.species_list)
        logger.debug(
            "  %s: %d points, weight=%.3f, species=%s",
            payload.dataset_id,
            payload.point_count,
            weights_norm.get(payload.dataset_id, 1.0),
            species_label,
        )

    penalty_value = 1e6
    objective_total_points = int(sum(p.point_count for p in payloads))

    layout = build_parameter_layout(
        payloads=payloads,
        shared_params=shared_params,
        dataset_variable_params=dataset_variable_params_map,
        bounds=bounds,
        log10_params=log10_params,
    )

    method_key = str(method or "trf").strip().lower()
    if method_key in {"differential_evolution", "de"}:
        method_key = "de"
    elif method_key not in {"lm", "trf", "dogbox"}:
        raise ValueError(f"Unknown optimization method: {method}")

    ctx = ObjectiveContext()
    fit_evaluator = coerce_fitting_series_evaluator(fit_evaluator)

    objective_impl: Optional[GlobalFitObjective] = None
    fit_parameters_completed = False
    result_to_return: Optional[GlobalFitResult] = None
    contained_fit_evaluator = None

    def _failed_result(
        message: str,
        *,
        failed_params: Optional[Dict[str, float]] = None,
        optimizer_diagnostic: Optional[FitDiagnostic] = None,
    ) -> GlobalFitResult:
        shared_snapshot = dict(shared_params)
        dataset_snapshot: Dict[str, Dict[str, float]] = {
            payload.dataset_id: dict(dataset_params_map.get(payload.dataset_id, {})) for payload in payloads
        }
        if failed_params:
            formatted_failed_params = GlobalFitObjective.format_params(failed_params)
            dataset_param_owners: Dict[str, set[str]] = {}
            for ds_id, params_map in dataset_snapshot.items():
                for param_name in params_map:
                    dataset_param_owners.setdefault(param_name, set()).add(ds_id)
            for ds_id, var_map in layout.dataset_var_index.items():
                for var_name in var_map:
                    dataset_param_owners.setdefault(var_name, set()).add(ds_id)

            for name, value in formatted_failed_params.items():
                if "::" in name:
                    failed_ds_id, param_name = name.split("::", 1)
                    dataset_snapshot.setdefault(failed_ds_id, {})
                    dataset_snapshot[failed_ds_id][param_name] = float(value)
                    continue
                if name in shared_snapshot or name in layout.param_names:
                    shared_snapshot[name] = float(value)
                    continue
                owners = dataset_param_owners.get(name, set())
                if len(owners) == 1:
                    owner_ds_id = next(iter(owners))
                    dataset_snapshot.setdefault(owner_ds_id, {})
                    dataset_snapshot[owner_ds_id][name] = float(value)

        completion_optimizer_diagnostic = ensure_failure_diagnostic(
            optimizer_diagnostic,
            message=str(message or "Global fit failed before replay completed."),
        )
        return GlobalFitResult(
            shared_params=shared_snapshot,
            dataset_params=dataset_snapshot,
            uncertainties=None,
            global_chi_squared=np.inf,
            global_r_squared=0.0,
            dataset_info=[],
            nfev=int(getattr(objective_impl, "_iteration", 0)),
            message=message,
            completion=GlobalFitCompletion(
                status="fail",
                optimizer_converged=False,
                nonfinite_metrics=False,
                optimizer_diagnostic=completion_optimizer_diagnostic,
                dataset_failures={},
                dataset_warnings={},
                detail_sections=build_completion_detail_sections(
                    status="fail",
                    optimizer_diagnostic=completion_optimizer_diagnostic,
                    dataset_failures={},
                ),
            ),
            covariance=None,
            objective_residuals=None,
            model_series={},
            residual_series={},
        )

    try:
        if runtime_session is not None:
            fit_evaluator_for_run = runtime_session.evaluator(cancellation_check=cancellation_check)
        elif type(fit_evaluator) is SerialFittingEvaluator:
            lane_count = 1 if max_runtime_lanes is None else max(1, int(max_runtime_lanes))
            contained_fit_evaluator = FittingRuntimeSession.from_serial_evaluator(
                fit_evaluator,
                max_lanes=lane_count,
                ledger=runtime_ledger,
            )
            fit_evaluator_for_run = contained_fit_evaluator.evaluator(cancellation_check=cancellation_check)
        else:
            fit_evaluator_for_run = fit_evaluator

        objective_impl = GlobalFitObjective(
            fit_evaluator=fit_evaluator_for_run,
            payloads=payloads,
            shared_params=shared_params,
            dataset_params=dataset_params_map,
            weights=weights_norm,
            layout=layout,
            penalty_value=penalty_value,
            ctx=ctx,
            progress_callback=progress_callback,
            cancellation_check=cancellation_check,
        )
        objective_wrapper = ObjectiveWrapper(objective_impl, ctx)

        de_penalty = None
        if method_key == "de":
            de_penalty = float((float(penalty_value) ** 2) * float(max(1, objective_total_points)))

        fit_result = fit_parameters(
            objective_wrapper,
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
        fit_parameters_completed = True

        opt_param_keys = layout.opt_param_keys()
        x_opt = np.asarray([fit_result.parameters[key] for key in opt_param_keys], dtype=float)
        success = bool(fit_result.success)
        message = str(fit_result.message)
        nfev = int(fit_result.nfev)
        covariance = fit_result.covariance
        objective_residuals = np.asarray(fit_result.residuals, dtype=float).reshape(-1)
        optimizer_diagnostic: Optional[FitDiagnostic] = None

        if getattr(objective_wrapper, "last_error", None) is not None:
            ds_tag = None
            prov = getattr(objective_wrapper, "last_error_provenance", None)
            if isinstance(prov, dict):
                ds_tag = prov.get("dataset")
            if optimizer_diagnostic is None:
                optimizer_diagnostic = build_fit_diagnostic_from_exception(
                    objective_wrapper.last_error,
                    phase="optimizer",
                    dataset_id=str(ds_tag) if ds_tag is not None else None,
                )

        fitted_params = layout.shared_param_dict_from_vector(x_opt)

        combined_dataset_params: Dict[str, Dict[str, float]] = {
            payload.dataset_id: dict(dataset_params_map.get(payload.dataset_id, {})) for payload in payloads
        }
        for (ds_id, param_name) in layout.dataset_var_order:
            combined_dataset_params.setdefault(ds_id, {})
            idx = layout.dataset_var_index[ds_id][param_name]
            val = float(x_opt[idx])
            if layout.dataset_var_log10.get((ds_id, param_name)):
                combined_dataset_params[ds_id][param_name] = float(10.0 ** val)
            else:
                combined_dataset_params[ds_id][param_name] = val

        uncertainties = None
        if covariance is not None:
            try:
                std_devs = np.sqrt(np.diag(covariance))
                uncertainties = {}
                for i, name in enumerate(layout.param_names):
                    if i >= int(std_devs.size):
                        break
                    std = float(std_devs[i])
                    if layout.shared_log10.get(name):
                        x_val = float(fitted_params.get(name, float("nan")))
                        uncertainties[name] = float(np.log(10.0) * x_val * std)
                    else:
                        uncertainties[name] = std
            except Exception as exc:
                logger.debug("Failed to calculate uncertainties: %s", exc)

        result_to_return = assemble_global_fit_result(
            fit_evaluator=fit_evaluator_for_run,
            payloads=payloads,
            layout=layout,
            fitted_params=fitted_params,
            combined_dataset_params=combined_dataset_params,
            weights=weights_norm,
            penalty_value=penalty_value,
            cancellation_check=cancellation_check,
            success=success,
            message=message,
            nfev=nfev,
            covariance=covariance,
            objective_residuals=objective_residuals,
            uncertainties=uncertainties,
            optimizer_diagnostic=optimizer_diagnostic,
        )
        if result_to_return is None:
            raise RuntimeError("Global fitting did not produce a result.")
        return result_to_return
    except FittingCancelled:
        raise
    except FitSimulationError as exc:
        if fit_parameters_completed:
            raise
        logger.error("Global fitting failed: %s", exc, exc_info=False)
        return _failed_result(
            str(exc),
            failed_params=exc.failed_params,
            optimizer_diagnostic=build_fit_diagnostic_from_exception(
                exc,
                phase="fatal",
                parameter_snapshot=exc.failed_params,
            ),
        )
    except Exception as exc:
        if isinstance(exc, (FittingCancelled, SimulationCancelled)) or "cancelled" in str(exc).lower():
            raise FittingCancelled() from exc
        if fit_parameters_completed:
            raise
        logger.error("Global fitting failed: %s", exc, exc_info=True)
        return _failed_result(
            f"Fitting failed: {exc}",
            optimizer_diagnostic=build_fit_diagnostic_from_exception(
                exc,
                phase="fatal",
            ),
        )
    finally:
        if contained_fit_evaluator is not None:
            contained_fit_evaluator.close(kill=False)
