"""
Optimization primitives for parameter fitting.

This module intentionally does *not* own simulation preparation or DSL parsing; it only
operates on an objective callable that returns residual vectors.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import numpy as np

from kindred.core.analysis.fit_statistics import finite_sum_of_squares
from kindred.core.exceptions import (
    FittingCancelled,
    FitSimulationError,
    KineticRuntimeDomainError,
    OptimizationError,
    SimulationCancelled,
)
from kindred.core.optimization_de import (
    DifferentialEvolutionBudgetExhausted,
    DifferentialEvolutionEvaluationBudget,
    compute_de_popsize_maxiter,
)
from kindred.core.optimization_least_squares import build_least_squares_kwargs
from kindred.core.scipy_optimize import load_scipy_optimize
from kindred.core.fitting_method_policy import validate_fit_method
from kindred.core.species import validate_evaluation_count

logger = logging.getLogger(__name__)

__all__ = [
    "FitResult",
    "fit_parameters",
    "fit_parameters_with_fixed",
]


def _is_cancelled_error(exc: BaseException) -> bool:
    # Cancellation is an explicit typed outcome. Never infer it from an
    # arbitrary exception message, which could hide a real solver/programming
    # failure containing the word "cancel".
    return isinstance(exc, (KeyboardInterrupt, FittingCancelled, SimulationCancelled))


@dataclass
class FitResult:
    """Results from parameter fitting."""

    success: bool
    parameters: Dict[object, float]
    message: str
    identifiability_warning: Optional[str] = None

    def __post_init__(self) -> None:
        self.parameters = {
            name: float(value) for name, value in self.parameters.items()
        }
        self.success = bool(self.success)


def fit_parameters_with_fixed(
    objective_func: Callable[[np.ndarray], np.ndarray],
    initial_params: Dict[object, float],
    *,
    fixed_parameters: Dict[object, float],
    bounds: Optional[Dict[object, Tuple[float, float]]] = None,
    method: str,
    max_nfev: int = 1000,
    seed: Optional[int] = None,
    ftol: float = 1e-10,
    xtol: float = 1e-10,
    cancellation_check: Callable[[], bool] | None = None,
    de_penalty: float | None = None,
    scipy_loader: Callable[[], tuple[Callable[..., Any], Callable[..., Any]]] | None = None,
) -> FitResult:
    """Optimize one canonical objective while holding named coordinates fixed.

    The reduced-coordinate wrapper delegates to :func:`fit_parameters`; it is
    intentionally not a second optimizer implementation.  Returned parameters
    retain the original full coordinate order and include the fixed values.
    """

    initial = {name: float(value) for name, value in initial_params.items()}
    fixed = {name: float(value) for name, value in fixed_parameters.items()}
    unknown = sorted(set(fixed) - set(initial))
    if unknown:
        raise ValueError(
            "Fixed Fit coordinates are absent from the optimizer layout: "
            + ", ".join(str(item) for item in unknown)
        )
    if not fixed:
        return fit_parameters(
            objective_func,
            initial,
            bounds=bounds,
            method=method,
            max_nfev=max_nfev,
            seed=seed,
            ftol=ftol,
            xtol=xtol,
            cancellation_check=cancellation_check,
            de_penalty=de_penalty,
            scipy_loader=scipy_loader,
        )
    bound_map = dict(bounds or {})
    for name, value in fixed.items():
        if not np.isfinite(value):
            raise ValueError(f"Fixed Fit coordinate {name!r} must be finite.")
        if name not in bound_map:
            raise ValueError(f"Fixed Fit coordinate {name!r} requires finite bounds.")
        low, high = bound_map[name]
        if (
            not np.isfinite(float(low))
            or not np.isfinite(float(high))
            or float(low) >= float(high)
            or not float(low) <= value <= float(high)
        ):
            raise ValueError(f"Fixed Fit coordinate {name!r} lies outside its bounds.")

    names = tuple(initial)
    free = {name: value for name, value in initial.items() if name not in fixed}
    free_bounds = {name: bound_map[name] for name in free if name in bound_map}

    def full_vector(values: Mapping[str, float]) -> np.ndarray:
        return np.asarray(
            [fixed[name] if name in fixed else values[name] for name in names],
            dtype=float,
        )

    if not free:
        if cancellation_check is not None and cancellation_check():
            raise FittingCancelled()
        return FitResult(
            success=True,
            parameters={name: fixed[name] for name in names},
            message="All optimizer coordinates were fixed.",
        )

    def reduced_objective(vector: np.ndarray) -> np.ndarray:
        values = {
            name: float(vector[index])
            for index, name in enumerate(free)
        }
        return objective_func(full_vector(values))

    reduced = fit_parameters(
        reduced_objective,
        free,
        bounds=free_bounds,
        method=method,
        max_nfev=max_nfev,
        seed=seed,
        ftol=ftol,
        xtol=xtol,
        cancellation_check=cancellation_check,
        de_penalty=de_penalty,
        scipy_loader=scipy_loader,
    )
    return FitResult(
        success=reduced.success,
        parameters={
            name: fixed[name] if name in fixed else reduced.parameters[name]
            for name in names
        },
        message=reduced.message,
        identifiability_warning=reduced.identifiability_warning,
    )


def fit_parameters(
    objective_func: Callable[[np.ndarray], np.ndarray],
    initial_params: Dict[object, float],
    bounds: Optional[Dict[object, Tuple[float, float]]] = None,
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

    method = validate_fit_method(method)
    max_nfev = validate_evaluation_count(
        max_nfev,
        field="Fit evaluation budget",
    )
    if max_nfev <= 0:
        raise ValueError("Fit evaluation budget must be positive.")
    scipy_loader = load_scipy_optimize if scipy_loader is None else scipy_loader
    param_names = list(initial_params.keys())
    if not param_names:
        raise ValueError("Fit optimization requires at least one coordinate.")
    if bounds is None:
        raise ValueError("Fit optimization requires finite ordered bounds for every coordinate.")
    missing_bounds = sorted(set(param_names) - set(bounds))
    if missing_bounds:
        raise ValueError(
            "Fit optimization requires bounds for every coordinate: "
            + ", ".join(missing_bounds)
        )
    x0 = np.asarray([float(initial_params[name]) for name in param_names], dtype=float)
    lower = np.asarray([float(bounds[name][0]) for name in param_names], dtype=float)
    upper = np.asarray([float(bounds[name][1]) for name in param_names], dtype=float)
    if (
        not np.all(np.isfinite(x0))
        or not np.all(np.isfinite(lower))
        or not np.all(np.isfinite(upper))
        or np.any(lower >= upper)
        or np.any(x0 < lower)
        or np.any(x0 > upper)
    ):
        raise ValueError(
            "Fit optimization requires finite ordered bounds and initial values "
            "inside those bounds."
        )
    scipy_bounds = (lower, upper)

    de_evaluation_budget: DifferentialEvolutionEvaluationBudget | None = None
    identifiability_warning: Optional[str] = None

    def wrapped_objective(params):
        if cancellation_check is not None and cancellation_check():
            raise FittingCancelled()
        return objective_func(params)

    try:
        if method == "differential_evolution":
            _least_squares, differential_evolution = scipy_loader()
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
            best_de_score: float | None = None
            best_de_params: np.ndarray | None = None

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
                nonlocal best_de_score, best_de_params
                try:
                    residuals = evaluate_de_residuals(params, reserve=1)
                except FittingCancelled:
                    raise
                except (FitSimulationError, KineticRuntimeDomainError):
                    raise
                else:
                    if residuals.size == 0 or not np.all(np.isfinite(residuals)):
                        raise OptimizationError(
                            "Fit objective returned empty or non-finite residuals."
                        )
                    finite_score = finite_sum_of_squares(residuals)
                    score = (
                        float(effective_penalty)
                        if finite_score is None
                        else float(finite_score)
                    )
                    if best_de_score is None or score < best_de_score:
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
                    np.asarray(best_de_params, dtype=float).copy()
                    if best_de_params is not None
                    else np.asarray(x0, dtype=float).copy()
                )
                success = False
                message = (
                    "Differential Evolution reached its maximum objective-evaluation "
                    f"budget ({de_evaluation_budget.limit})."
                )

        else:
            least_squares, _de = scipy_loader()

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
            with warnings.catch_warnings():
                # The zero-only gradient stop deliberately sits below epsilon.
                warnings.filterwarnings(
                    "ignore",
                    message=(
                        r"^Setting `gtol` below the machine epsilon .* effectively "
                        r"disables the corresponding termination condition\.$"
                    ),
                    category=UserWarning,
                    module=r"^scipy\.optimize\._lsq\.least_squares$",
                )
                result = least_squares(wrapped_objective, x0, **lsq_kwargs)

            x_opt = result.x
            success = bool(result.success)
            message = str(result.message)
            if result.status == 1:
                message = "Fit stopped at zero computed first-order optimality."
            try:
                J = result.jac
                param_count = len(x_opt)
                rank = int(np.linalg.matrix_rank(np.asarray(J)))
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
    except (FitSimulationError, KineticRuntimeDomainError):
        raise
    except Exception as exc:
        if _is_cancelled_error(exc):
            raise FittingCancelled() from exc
        logger.error("Fitting failed: %s", exc, exc_info=True)
        raise OptimizationError(f"Fitting failed: {exc}") from exc

    fitted_params = {name: val for name, val in zip(param_names, x_opt)}

    terminal_success = bool(success)

    return FitResult(
        success=terminal_success,
        parameters=fitted_params,
        message=message,
        identifiability_warning=identifiability_warning,
    )
