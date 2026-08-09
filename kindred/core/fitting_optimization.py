"""
Optimization primitives for parameter fitting.

This module intentionally does *not* own simulation preparation or DSL parsing; it only
operates on an objective callable that returns residual vectors.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from kindred.core.exceptions import FittingCancelled, FitSimulationError, OptimizationError, SimulationCancelled
from kindred.core.optimization_de import (
    DifferentialEvolutionBudgetExhausted,
    DifferentialEvolutionEvaluationBudget,
    compute_de_popsize_maxiter,
)
from kindred.core.optimization_least_squares import build_least_squares_kwargs
from kindred.core.scipy_optimize import load_scipy_optimize

logger = logging.getLogger(__name__)

__all__ = [
    "FitResult",
    "fit_parameters",
]


def _is_cancelled_error(exc: BaseException) -> bool:
    # Cancellation is an explicit typed outcome. Never infer it from an
    # arbitrary exception message, which could hide a real solver/programming
    # failure containing the word "cancel".
    return isinstance(exc, (KeyboardInterrupt, FittingCancelled, SimulationCancelled))


def _is_expected_candidate_domain_error(exc: BaseException) -> bool:
    """Return true only for an evaluator's explicit candidate-domain outcome."""

    return bool(
        isinstance(exc, FitSimulationError)
        and isinstance(exc.details, dict)
        and exc.details.get("candidate_domain") is True
        and exc.details.get("fatal") is False
    )


@dataclass
class FitResult:
    """Results from parameter fitting."""

    success: bool
    parameters: Dict[str, float]
    residuals: np.ndarray
    nfev: int
    message: str
    identifiability_warning: Optional[str] = None
    parameter_space: str = "optimizer"


def fit_parameters(
    objective_func: Callable[[np.ndarray], np.ndarray],
    initial_params: Dict[str, float],
    bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    max_nfev: int = 1000,
    seed: Optional[int] = None,
    ftol: float = 1e-10,
    xtol: float = 1e-10,
    *,
    method: str,
    cancellation_check: Callable[[], bool] | None = None,
    de_penalty: float | None = None,
    scipy_loader: Callable[[], tuple[Callable[..., Any], Callable[..., Any]]] | None = None,
) -> FitResult:
    """Fit parameters, interpreting ``ftol`` as native DE ``tol`` for DE runs."""

    scipy_loader = load_scipy_optimize if scipy_loader is None else scipy_loader
    param_names = list(initial_params.keys())
    x0 = np.array([initial_params[name] for name in param_names])

    if bounds is not None:
        lower = np.array([bounds.get(name, (-np.inf, np.inf))[0] for name in param_names])
        upper = np.array([bounds.get(name, (-np.inf, np.inf))[1] for name in param_names])
        scipy_bounds = (lower, upper)
    else:
        scipy_bounds = (-np.inf, np.inf)

    iteration_counter = [0]
    de_evaluation_budget: DifferentialEvolutionEvaluationBudget | None = None
    identifiability_warning: Optional[str] = None

    def _vector_to_param_dict(vector: np.ndarray) -> Dict[str, float]:
        return {name: float(vector[i]) for i, name in enumerate(param_names)}

    def _build_failed_result(
        message: str,
        *,
        failed_params: Optional[Dict[str, float]] = None,
        parameter_space: str = "optimizer",
    ) -> FitResult:
        params = failed_params or _vector_to_param_dict(x0)
        space = str(parameter_space or "optimizer").strip().lower()
        if space not in {"optimizer", "physical"}:
            raise ValueError(f"Unknown failed parameter space: {parameter_space!r}")
        return FitResult(
            success=False,
            parameters=params,
            residuals=np.array([]),
            nfev=(
                de_evaluation_budget.nfev
                if de_evaluation_budget is not None
                else iteration_counter[0]
            ),
            message=message,
            identifiability_warning=None,
            parameter_space=space,
        )

    def wrapped_objective(params):
        if cancellation_check is not None and cancellation_check():
            raise FittingCancelled()
        residuals = objective_func(params)
        iteration_counter[0] += 1
        return residuals

    try:
        if method == "differential_evolution":
            _least_squares, differential_evolution = scipy_loader()
            if bounds is None:
                raise ValueError("Differential Evolution requires bounds")

            invalid_params: list[str] = []
            for idx, name in enumerate(param_names):
                low = float(scipy_bounds[0][idx])
                high = float(scipy_bounds[1][idx])
                if not (np.isfinite(low) and np.isfinite(high)):
                    invalid_params.append(name)
                elif low >= high:
                    invalid_params.append(name)
            if invalid_params:
                raise ValueError(
                    "Differential Evolution requires finite bounds with min < max. "
                    f"Parameters with invalid bounds: {', '.join(sorted(set(invalid_params)))}"
                )

            effective_penalty = float(de_penalty) if de_penalty is not None else 1e12
            de_evaluation_budget = DifferentialEvolutionEvaluationBudget(max_nfev)
            best_de_params: np.ndarray | None = None
            best_de_score = np.inf

            def evaluate_de_residuals(
                params: np.ndarray,
                *,
                reserve: int,
            ) -> np.ndarray:
                if cancellation_check is not None and cancellation_check():
                    raise FittingCancelled()
                de_evaluation_budget.claim(reserve=reserve)
                residuals = np.asarray(objective_func(params), dtype=float).reshape(-1)
                return residuals

            def de_objective(params):
                nonlocal best_de_params, best_de_score
                try:
                    residuals = evaluate_de_residuals(params, reserve=1)
                except FittingCancelled:
                    raise
                except FitSimulationError as exc:
                    if not _is_expected_candidate_domain_error(exc):
                        raise
                    score = float(effective_penalty)
                else:
                    if residuals.size == 0 or not np.all(np.isfinite(residuals)):
                        raise OptimizationError(
                            "Fit objective returned empty or non-finite residuals."
                        )
                    score = float(np.sum(residuals**2))
                if score < best_de_score:
                    best_de_score = score
                    best_de_params = np.asarray(params, dtype=float).copy()
                return score

            dim = max(1, len(x0))
            result = None
            if de_evaluation_budget.limit > 1:
                popsize, maxiter = compute_de_popsize_maxiter(
                    budget=de_evaluation_budget.limit - 1,
                    dim=dim,
                )
                try:
                    result = differential_evolution(
                        de_objective,
                        bounds=list(zip(scipy_bounds[0], scipy_bounds[1])),
                        maxiter=maxiter,
                        popsize=popsize,
                        tol=float(ftol),
                        seed=seed,
                        disp=False,
                        polish=False,
                        workers=1,
                    )
                except DifferentialEvolutionBudgetExhausted:
                    pass

            if result is not None:
                x_opt = np.asarray(result.x, dtype=float)
                success = bool(result.success)
                message = str(result.message)
            else:
                x_opt = (
                    best_de_params
                    if best_de_params is not None
                    else np.asarray(x0, dtype=float).copy()
                )
                success = False
                message = (
                    "Differential Evolution reached its maximum objective-evaluation "
                    f"budget ({de_evaluation_budget.limit})."
                )

            try:
                residuals = evaluate_de_residuals(x_opt, reserve=0)
            except FittingCancelled:
                raise
            except FitSimulationError as exc:
                if not _is_expected_candidate_domain_error(exc):
                    raise
                logger.error("Simulation failed when evaluating DE optimum: %s", exc)
                if exc.failed_params:
                    return _build_failed_result(
                        str(exc),
                        failed_params=exc.failed_params,
                        parameter_space="physical",
                    )
                return _build_failed_result(
                    str(exc),
                    failed_params=_vector_to_param_dict(x_opt),
                    parameter_space="optimizer",
                )
            nfev = de_evaluation_budget.nfev
        else:
            least_squares, _de = scipy_loader()

            if bounds is not None:
                _lb = np.array(scipy_bounds[0])
                _ub = np.array(scipy_bounds[1])
                _eps = np.minimum(1e-12, (_ub - _lb) / 3.0)
                x0 = np.clip(x0, _lb + _eps, _ub - _eps)

            # scipy interprets diff_step as relative to each coordinate.  Passing
            # an already scaled absolute step applies that scaling a second time.
            diff_step = 1e-4
            lsq_kwargs = build_least_squares_kwargs(
                ftol=ftol,
                xtol=xtol,
                max_nfev=max_nfev,
                method=method,
                bounds=scipy_bounds,
                diff_step=diff_step,
                verbose=0,
            )
            result = least_squares(wrapped_objective, x0, **lsq_kwargs)

            x_opt = result.x
            success = bool(result.success)
            message = str(result.message)
            nfev = int(result.nfev)
            residuals = np.asarray(result.fun, dtype=float).reshape(-1)
            if residuals.size == 0 or not np.all(np.isfinite(residuals)):
                raise OptimizationError(
                    "Least-squares optimizer returned empty or non-finite residuals."
                )

            try:
                J = result.jac
                param_count = int(len(x_opt))
                rank = int(np.linalg.matrix_rank(J))
                if rank < param_count:
                    identifiability_warning = (
                        "Fit Jacobian is rank deficient; parameter identifiability "
                        f"is limited (rank {rank} < {param_count})."
                    )
                    message = f"{message}; {identifiability_warning}"
            except Exception as exc:
                logger.debug("Failed to assess Fit Jacobian rank: %s", exc)
    except FittingCancelled:
        raise
    except FitSimulationError:
        raise
    except Exception as exc:
        if _is_cancelled_error(exc):
            raise FittingCancelled() from exc
        logger.error("Fitting failed: %s", exc, exc_info=True)
        raise OptimizationError(f"Fitting failed: {exc}") from exc

    fitted_params = {name: val for name, val in zip(param_names, x_opt)}

    return FitResult(
        success=success,
        parameters=fitted_params,
        residuals=residuals,
        nfev=nfev,
        message=message,
        identifiability_warning=identifiability_warning,
        parameter_space="optimizer",
    )
