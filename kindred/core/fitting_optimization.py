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
from kindred.core.optimization_de import compute_de_popsize_maxiter
from kindred.core.optimization_least_squares import build_least_squares_kwargs
from kindred.core.scipy_optimize import load_scipy_optimize

logger = logging.getLogger(__name__)

__all__ = [
    "FitProgress",
    "FitResult",
    "fit_parameters",
]


def _is_cancelled_error(exc: BaseException) -> bool:
    if isinstance(exc, (KeyboardInterrupt, FittingCancelled, SimulationCancelled)):
        return True
    message = str(exc).lower()
    return "cancelled" in message or "canceled" in message or "cancel" in message


@dataclass
class FitProgress:
    """Progress information during fitting."""

    iteration: int
    nfev: int
    cost: float
    parameters: Dict[str, float]
    message: str = ""


@dataclass
class FitResult:
    """Results from parameter fitting."""

    success: bool
    parameters: Dict[str, float]
    uncertainties: Optional[Dict[str, float]]
    residual_mse: float
    r_squared: float
    residuals: np.ndarray
    nfev: int
    message: str
    covariance: Optional[np.ndarray] = None
    identifiability_warning: Optional[str] = None
    parameter_space: str = "optimizer"


def fit_parameters(
    objective_func: Callable[[np.ndarray], np.ndarray],
    initial_params: Dict[str, float],
    bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    method: str = "lm",
    progress_callback: Optional[Callable[[FitProgress], None]] = None,
    max_nfev: int = 1000,
    seed: Optional[int] = None,
    ftol: float = 1e-10,
    xtol: float = 1e-10,
    *,
    cancellation_check: Callable[[], bool] | None = None,
    de_penalty: float | None = None,
    scipy_loader: Callable[[], tuple[Callable[..., Any], Callable[..., Any]]] | None = None,
) -> FitResult:
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
            uncertainties=None,
            residual_mse=np.inf,
            r_squared=float("nan"),
            residuals=np.array([]),
            nfev=iteration_counter[0],
            message=message,
            covariance=None,
            identifiability_warning=None,
            parameter_space=space,
        )

    def wrapped_objective(params):
        if cancellation_check is not None and cancellation_check():
            raise FittingCancelled()
        residuals = objective_func(params)
        iteration_counter[0] += 1
        if progress_callback is not None and iteration_counter[0] % 10 == 0:
            param_dict = {name: val for name, val in zip(param_names, params)}
            cost = np.sum(residuals**2)
            progress_callback(
                FitProgress(
                    iteration=iteration_counter[0],
                    nfev=iteration_counter[0],
                    cost=cost,
                    parameters=param_dict,
                    message=f"Iteration {iteration_counter[0]}",
                )
            )
        return residuals

    try:
        if method.lower() == "de":
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

            def de_objective(params):
                if cancellation_check is not None and cancellation_check():
                    raise FittingCancelled()
                try:
                    residuals = np.asarray(wrapped_objective(params), dtype=float).reshape(-1)
                except FittingCancelled:
                    raise
                except FitSimulationError:
                    return float(effective_penalty)
                except RuntimeError as exc:
                    if _is_cancelled_error(exc):
                        raise FittingCancelled() from exc
                    return float(effective_penalty)
                except Exception as exc:
                    if _is_cancelled_error(exc):
                        raise FittingCancelled() from exc
                    return float(effective_penalty)

                if residuals.size == 0 or not np.all(np.isfinite(residuals)):
                    return float(effective_penalty)
                return float(np.sum(residuals**2))

            dim = max(1, len(x0))
            budget = max(1, int(max_nfev))
            popsize, maxiter = compute_de_popsize_maxiter(budget=budget, dim=dim)

            result = differential_evolution(
                de_objective,
                bounds=list(zip(scipy_bounds[0], scipy_bounds[1])),
                maxiter=maxiter,
                popsize=popsize,
                seed=seed,
                disp=False,
                polish=False,
                workers=1,
            )

            x_opt = result.x
            success = bool(result.success)
            message = str(result.message)
            nfev = result.nfev

            try:
                residuals = objective_func(x_opt)
            except FittingCancelled:
                raise
            except FitSimulationError as exc:
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
            covariance = None
        else:
            least_squares, _de = scipy_loader()
            if method.lower() == "lm":
                scipy_method = "lm"
                if bounds is not None:
                    lower_arr = np.asarray(scipy_bounds[0], dtype=float).reshape(-1)
                    upper_arr = np.asarray(scipy_bounds[1], dtype=float).reshape(-1)
                    bounded_params = [
                        name
                        for idx, name in enumerate(param_names)
                        if idx < lower_arr.size
                        and idx < upper_arr.size
                        and (np.isfinite(lower_arr[idx]) or np.isfinite(upper_arr[idx]))
                    ]
                    if bounded_params:
                        raise ValueError(
                            "Levenberg-Marquardt fitting does not support finite bounds. "
                            "Use trf or dogbox, or remove bounds for: "
                            + ", ".join(sorted(set(bounded_params)))
                        )
                scipy_bounds = (-np.inf, np.inf)
            elif method.lower() == "trf":
                scipy_method = "trf"
            elif method.lower() == "dogbox":
                scipy_method = "dogbox"
            else:
                raise ValueError(f"Unknown method: {method}")

            if scipy_method in ("trf", "dogbox") and bounds is not None:
                _lb = np.array(scipy_bounds[0])
                _ub = np.array(scipy_bounds[1])
                _eps = np.minimum(1e-12, (_ub - _lb) / 3.0)
                x0 = np.clip(x0, _lb + _eps, _ub - _eps)

            if scipy_method == "lm":
                diff_step = float(np.nanmedian(np.maximum(np.abs(x0) * 1e-4, 1e-6)))
                if not np.isfinite(diff_step) or diff_step <= 0.0:
                    diff_step = 1e-6
            else:
                diff_step = np.maximum(np.abs(x0) * 1e-4, 1e-6)
            lsq_kwargs = build_least_squares_kwargs(
                ftol=ftol,
                xtol=xtol,
                max_nfev=max_nfev,
                method=scipy_method,
                bounds=scipy_bounds,
                diff_step=diff_step,
                verbose=0,
            )
            result = least_squares(wrapped_objective, x0, **lsq_kwargs)

            x_opt = result.x
            success = result.success
            message = result.message
            nfev = result.nfev
            try:
                residuals = objective_func(x_opt)
            except FittingCancelled:
                raise
            except FitSimulationError as exc:
                logger.error("Simulation failed when evaluating optimal parameters: %s", exc)
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
            except Exception:
                residuals = result.fun

            try:
                J = result.jac
                param_count = int(len(x_opt))
                rank = int(np.linalg.matrix_rank(J))
                if rank < param_count:
                    identifiability_warning = (
                        "Covariance and uncertainties were not reported because the Jacobian is "
                        f"rank deficient (rank {rank} < {param_count})."
                    )
                    message = f"{message}; {identifiability_warning}"
                    covariance = None
                else:
                    JtJ = J.T @ J
                    residual_sum = np.sum(residuals**2)
                    dof = len(residuals) - len(x_opt)
                    if dof <= 0:
                        dof = max(1, len(residuals))
                    residual_var = residual_sum / dof
                    covariance = np.linalg.pinv(JtJ) * residual_var
            except Exception as exc:
                logger.debug("Failed to calculate covariance: %s", exc)
                covariance = None
    except FittingCancelled:
        raise
    except FitSimulationError as exc:
        logger.error("Simulation failed during fitting: %s", exc)
        if exc.failed_params:
            return _build_failed_result(
                str(exc),
                failed_params=exc.failed_params,
                parameter_space="physical",
            )
        return _build_failed_result(str(exc), parameter_space="optimizer")
    except RuntimeError as exc:
        if _is_cancelled_error(exc):
            raise FittingCancelled() from exc
        logger.error("Fitting failed: %s", exc, exc_info=True)
        raise OptimizationError(f"Fitting failed: {exc}") from exc
    except Exception as exc:
        if _is_cancelled_error(exc):
            raise FittingCancelled() from exc
        logger.error("Fitting failed: %s", exc, exc_info=True)
        raise OptimizationError(f"Fitting failed: {exc}") from exc

    fitted_params = {name: val for name, val in zip(param_names, x_opt)}

    uncertainties = None
    if covariance is not None:
        try:
            std_devs = np.sqrt(np.diag(covariance))
            uncertainties = {name: std for name, std in zip(param_names, std_devs)}
        except Exception as exc:
            logger.debug("Failed to calculate uncertainties: %s", exc)

    residual_mse = np.sum(residuals**2) / len(residuals) if len(residuals) > 0 else np.inf

    return FitResult(
        success=success,
        parameters=fitted_params,
        uncertainties=uncertainties,
        residual_mse=residual_mse,
        r_squared=float("nan"),
        residuals=residuals,
        nfev=nfev,
        message=message,
        covariance=covariance,
        identifiability_warning=identifiability_warning,
        parameter_space="optimizer",
    )
