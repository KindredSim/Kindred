"""
Optimization primitives for parameter fitting.

This module intentionally does *not* own simulation preparation or DSL parsing; it only
operates on an objective callable that returns residual vectors.
"""

from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass
from typing import Callable, Dict, Literal, Mapping, Optional, Tuple

import numpy as np

from kindred.core.exceptions import (
    FittingCancelled,
    FitSimulationError,
    KineticRuntimeDomainError,
    OptimizationError,
)
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
    "fit_bound_position",
    "fit_parameters",
    "fit_parameters_with_fixed",
]

FitBoundPosition = Literal["lower", "upper", "interior"]

# Finite-difference Jacobians resolve exactly dependent fitted values only to
# singular-value ratios of roughly 1e-8 to 1e-5, far above numpy's default
# rank tolerance, so the rank uses a tolerance above that level.
_JACOBIAN_RANK_RTOL = 1e-4


def fit_bound_position(
    value: float,
    minimum: float,
    maximum: float,
    *,
    log10: bool = False,
) -> FitBoundPosition:
    """Classify a fitted value as at its lower or upper bound, or interior.

    A value is at a bound when its distance from it, in the coordinate the
    optimizer moved (log10 under Log Scale), is at most one billionth of that
    bound's own scale (its magnitude, at least 1, and at most the bounds'
    width), and it is nearer that bound than the other.  A distant opposite
    bound therefore never widens the tolerance.  The result is an
    informational label; values are always published as fitted.
    """

    def encode(raw: float) -> float:
        return math.log10(raw) if log10 else float(raw)

    encoded, low, high = encode(value), encode(minimum), encode(maximum)
    lower_distance = encoded - low
    upper_distance = high - encoded

    def tolerance(bound: float) -> float:
        return 1e-9 * min(high - low, max(1.0, abs(bound)))

    position: FitBoundPosition = "interior"
    if lower_distance <= min(upper_distance, tolerance(low)):
        position = "lower"
    if upper_distance <= min(lower_distance, tolerance(high)):
        position = "upper"
    return position


@dataclass
class FitResult:
    """Results from parameter fitting."""

    success: bool
    parameters: Dict[object, float]
    message: str
    identifiability_warning: Optional[str] = None


def fit_parameters_with_fixed(
    objective_func: Callable[[np.ndarray], np.ndarray],
    initial_params: Dict[object, float],
    *,
    fixed_parameters: Dict[object, float],
    bounds: Dict[object, Tuple[float, float]],
    method: str,
    max_nfev: int = 1000,
    ftol: float,
    xtol: float,
    cancellation_check: Callable[[], bool] | None = None,
) -> FitResult:
    """Optimize one canonical objective while holding named coordinates fixed.

    The reduced-coordinate wrapper delegates to :func:`fit_parameters`; it is
    intentionally not a second optimizer implementation.  Returned parameters
    retain the original full coordinate order and include the fixed values.
    """

    # Profile fixes one layout coordinate at a point between its fitted value
    # and one of its admitted bounds.
    initial = {name: float(value) for name, value in initial_params.items()}
    fixed = {name: float(value) for name, value in fixed_parameters.items()}
    names = tuple(initial)
    free = {name: value for name, value in initial.items() if name not in fixed}
    free_bounds = {name: bounds[name] for name in free}

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
        ftol=ftol,
        xtol=xtol,
        cancellation_check=cancellation_check,
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
    bounds: Dict[object, Tuple[float, float]],
    max_nfev: int = 1000,
    seed: Optional[int] = None,
    *,
    ftol: float,
    xtol: float,
    method: str,
    cancellation_check: Callable[[], bool] | None = None,
) -> FitResult:
    """Fit parameters, interpreting ``ftol`` as native DE ``tol`` for DE runs."""

    # The method, evaluation budget, coordinates and their bounds arrive
    # admitted by the Fit setup or its Profile likelihood configuration.
    param_names = list(initial_params.keys())
    # Starts are admitted values or optimizer results inside their admitted
    # finite ordered bounds.
    x0 = np.asarray([initial_params[name] for name in param_names], dtype=float)
    lower = np.asarray([bounds[name][0] for name in param_names], dtype=float)
    upper = np.asarray([bounds[name][1] for name in param_names], dtype=float)
    scipy_bounds = (lower, upper)

    de_evaluation_budget: DifferentialEvolutionEvaluationBudget | None = None
    identifiability_warning: Optional[str] = None

    def wrapped_objective(params):
        if cancellation_check is not None and cancellation_check():
            raise FittingCancelled()
        return objective_func(params)

    try:
        if method == "differential_evolution":
            _least_squares, differential_evolution = load_scipy_optimize()
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
                # The Fit objective returns finite residuals with a finite
                # sum of squares.
                residuals = evaluate_de_residuals(params, reserve=1)
                score = float(np.dot(residuals, residuals))
                if best_de_score is None or score < best_de_score:
                    best_de_score = score
                    best_de_params = np.asarray(params, dtype=float).copy()
                return score

            dim = len(x0)
            result = None
            if de_evaluation_budget.limit > 1:
                popsize, maxiter = compute_de_popsize_maxiter(
                    budget=de_evaluation_budget.limit - 1,
                    dim=dim,
                )
                # SciPy rescales x0 to [0, 1] and refuses a start that rounds
                # just outside, as a start on its bound can; such a start moves
                # inside by a negligible fraction of the range.
                margin = (upper - lower) * 1e-12
                try:
                    result = differential_evolution(
                        de_objective,
                        bounds=list(zip(scipy_bounds[0], scipy_bounds[1])),
                        x0=np.clip(x0, lower + margin, upper - margin),
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
                message = result.message
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
            least_squares, _de = load_scipy_optimize()

            # SciPy's default finite-difference step keeps an absolute floor, so
            # coordinates near zero still get usable Jacobian columns.
            lsq_kwargs = build_least_squares_kwargs(
                ftol=ftol,
                xtol=xtol,
                max_nfev=max_nfev,
                method=method,
                bounds=scipy_bounds,
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
            message = result.message
            if result.status == 1:
                message = "Fit stopped at zero computed first-order optimality."
            J = np.asarray(result.jac, dtype=float)
            param_count = len(x_opt)
            # Rank of the column-scaled Jacobian, so parameters on very
            # different scales are not mistaken for dependent ones.
            # All-zero columns stay zero and count as deficient.
            norms = np.linalg.norm(J, axis=0)
            scaled = np.divide(J, norms, out=np.zeros_like(J), where=norms > 0.0)
            rank = int(np.linalg.matrix_rank(scaled, rtol=_JACOBIAN_RANK_RTOL))
            if rank < param_count:
                identifiability_warning = (
                    "These data cannot determine every fitted value independently. "
                    f"The fit Jacobian has rank {rank} for {param_count} fitted value{'' if param_count == 1 else 's'}."
                )
    except FittingCancelled:
        raise
    except (FitSimulationError, KineticRuntimeDomainError):
        raise
    except Exception as exc:
        logger.error("Fitting failed: %s", exc, exc_info=True)
        raise OptimizationError(str(exc)) from exc

    fitted_params = {name: float(val) for name, val in zip(param_names, x_opt)}

    return FitResult(
        success=success,
        parameters=fitted_params,
        message=message,
        identifiability_warning=identifiability_warning,
    )
