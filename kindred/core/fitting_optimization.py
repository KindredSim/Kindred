"""
Optimization primitives for parameter fitting.

This module intentionally does *not* own simulation preparation or DSL parsing; it only
operates on an objective callable that returns residual vectors.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Callable, Dict, Literal, Mapping, Optional, Tuple

import numpy as np

from kindred.core.exceptions import FittingCancelled
from kindred.core.optimization_least_squares import build_least_squares_kwargs
from kindred.core.scipy_optimize import load_scipy_optimize

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
    """Fit parameters by least squares, or by Differential Evolution polished
    with Trust Region Reflective at the same tolerances."""

    # The method, evaluation budget, coordinates and their bounds arrive
    # admitted by the Fit setup or its Profile likelihood configuration.
    param_names = list(initial_params.keys())
    # Starts are admitted values or optimizer results inside their admitted
    # finite ordered bounds.
    x0 = np.asarray([initial_params[name] for name in param_names], dtype=float)
    lower = np.asarray([bounds[name][0] for name in param_names], dtype=float)
    upper = np.asarray([bounds[name][1] for name in param_names], dtype=float)
    scipy_bounds = (lower, upper)

    identifiability_warning: Optional[str] = None

    def wrapped_objective(params):
        if cancellation_check is not None and cancellation_check():
            raise FittingCancelled()
        return objective_func(params)

    def run_least_squares(start: np.ndarray, least_squares_method: str):
        least_squares, _de = load_scipy_optimize()
        # SciPy's default finite-difference step keeps an absolute floor, so
        # coordinates near zero still get usable Jacobian columns.
        lsq_kwargs = build_least_squares_kwargs(
            ftol=ftol,
            xtol=xtol,
            max_nfev=max_nfev,
            method=least_squares_method,
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
            return least_squares(wrapped_objective, start, **lsq_kwargs)

    if method == "differential_evolution":
        _least_squares, differential_evolution = load_scipy_optimize()

        def de_objective(params):
            # The Fit objective returns finite residuals with a finite
            # sum of squares.
            residuals = np.asarray(wrapped_objective(params), dtype=float).reshape(-1)
            return float(np.dot(residuals, residuals))

        def polish(_scalar_objective, start, **_scipy_arguments):
            # SciPy keeps the polish only when it succeeds, scores lower and
            # stays inside the bounds. Its score is the sum of squares, twice
            # the least-squares cost.
            polished = run_least_squares(np.asarray(start, dtype=float), "trf")
            polished.fun = 2.0 * polished.cost
            return polished

        # SciPy's default population size. Its first population is always
        # scored in full; the generations after it fit in Maximum
        # evaluations, (maxiter + 1) * popsize * len(x0) at most.
        popsize = 15
        maxiter = max(0, max_nfev // (popsize * len(x0)) - 1)
        # SciPy rescales x0 to [0, 1] and refuses a start that rounds
        # just outside, as a start on its bound can; such a start moves
        # inside by a negligible fraction of the range.
        margin = (upper - lower) * 1e-12
        error: Exception | None = None
        try:
            result = differential_evolution(
                de_objective,
                bounds=list(zip(scipy_bounds[0], scipy_bounds[1])),
                x0=np.clip(x0, lower + margin, upper - margin),
                maxiter=maxiter,
                popsize=popsize,
                seed=seed,
                disp=False,
                polish=polish,
                workers=1,
            )
        except RuntimeError as exc:
            # SciPy reports a TypeError or ValueError raised while it scores
            # its first population as a RuntimeError caused by it.
            if not isinstance(exc.__cause__, (TypeError, ValueError)):
                raise
            error = exc.__cause__
        if error is not None:
            raise error
        x_opt = np.asarray(result.x, dtype=float)
        success = bool(result.success)
        message = result.message

    else:
        result = run_least_squares(x0, method)
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

    fitted_params = {name: float(val) for name, val in zip(param_names, x_opt)}

    return FitResult(
        success=success,
        parameters=fitted_params,
        message=message,
        identifiability_warning=identifiability_warning,
    )
