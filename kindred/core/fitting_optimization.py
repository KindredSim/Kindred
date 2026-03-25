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


def _extract_failure_time(exc: BaseException) -> Optional[float]:
    """Best-effort extraction of a solver failure time from exception causes."""
    seen: set[int] = set()
    cur: Optional[BaseException] = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        details = getattr(cur, "details", None)
        if isinstance(details, dict):
            for key in ("time", "t_fail"):
                raw = details.get(key)
                if isinstance(raw, (int, float, np.integer, np.floating)):
                    return float(raw)
        cur = getattr(cur, "__cause__", None) or getattr(cur, "__context__", None)
    return None


def _objective_time_span(objective: object) -> Optional[Tuple[float, float]]:
    span = getattr(objective, "_kindred_t_span", None) or getattr(objective, "t_span", None)
    if span is None:
        return None
    try:
        t0 = float(span[0])
        t1 = float(span[1])
    except Exception:
        return None
    return (t0, t1)


def _scaled_de_penalty(*, base_penalty: float, exc: BaseException, objective: object) -> float:
    t_fail = _extract_failure_time(exc)
    if t_fail is None:
        return float(base_penalty)
    span = _objective_time_span(objective)
    if span is None:
        return float(base_penalty)
    t0, t1 = span
    denom = float(t1 - t0)
    if not np.isfinite(denom) or denom <= 0.0:
        return float(base_penalty)
    frac = (float(t_fail) - float(t0)) / denom
    if frac < 0.0:
        frac = 0.0
    elif frac > 1.0:
        frac = 1.0
    scaled = float(base_penalty) * (2.0 - frac)
    if not np.isfinite(scaled):
        return float(base_penalty)
    return float(scaled)


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
    chi_squared: float
    r_squared: float
    residuals: np.ndarray
    nfev: int
    message: str
    covariance: Optional[np.ndarray] = None


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

    def _vector_to_param_dict(vector: np.ndarray) -> Dict[str, float]:
        return {name: float(vector[i]) for i, name in enumerate(param_names)}

    def _build_failed_result(message: str, *, failed_params: Optional[Dict[str, float]] = None) -> FitResult:
        params = failed_params or _vector_to_param_dict(x0)
        return FitResult(
            success=False,
            parameters=params,
            uncertainties=None,
            chi_squared=np.inf,
            r_squared=float("nan"),
            residuals=np.array([]),
            nfev=iteration_counter[0],
            message=message,
            covariance=None,
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
            if de_penalty is None and getattr(objective_func, "_kindred_vector_objective", False):
                try:
                    penalty_len = int(np.asarray(getattr(objective_func, "y_exp")).size)
                    effective_penalty = float((1e6**2) * max(1, penalty_len))
                except Exception:
                    effective_penalty = 1e12

            def de_objective(params):
                if cancellation_check is not None and cancellation_check():
                    raise FittingCancelled()
                try:
                    residuals = np.asarray(wrapped_objective(params), dtype=float).reshape(-1)
                except FittingCancelled:
                    raise
                except FitSimulationError as exc:
                    de_objective.last_error = exc  # type: ignore[attr-defined]
                    t_fail = _extract_failure_time(exc)
                    de_objective.last_error_provenance = (  # type: ignore[attr-defined]
                        {"time": t_fail} if t_fail is not None else None
                    )
                    return _scaled_de_penalty(
                        base_penalty=effective_penalty, exc=exc, objective=objective_func
                    )
                except RuntimeError as exc:
                    if _is_cancelled_error(exc):
                        raise FittingCancelled() from exc
                    de_objective.last_error = FitSimulationError(str(exc))  # type: ignore[attr-defined]
                    t_fail = _extract_failure_time(exc)
                    de_objective.last_error_provenance = (  # type: ignore[attr-defined]
                        {"time": t_fail} if t_fail is not None else None
                    )
                    return _scaled_de_penalty(
                        base_penalty=effective_penalty, exc=exc, objective=objective_func
                    )
                except Exception as exc:
                    if _is_cancelled_error(exc):
                        raise FittingCancelled() from exc
                    de_objective.last_error = FitSimulationError(str(exc))  # type: ignore[attr-defined]
                    t_fail = _extract_failure_time(exc)
                    de_objective.last_error_provenance = (  # type: ignore[attr-defined]
                        {"time": t_fail} if t_fail is not None else None
                    )
                    return _scaled_de_penalty(
                        base_penalty=effective_penalty, exc=exc, objective=objective_func
                    )

                if residuals.size == 0 or not np.all(np.isfinite(residuals)):
                    err = FitSimulationError("Non-finite residuals encountered in DE objective.")
                    de_objective.last_error = err  # type: ignore[attr-defined]
                    de_objective.last_error_provenance = None  # type: ignore[attr-defined]
                    return float(effective_penalty)
                return float(np.sum(residuals**2))

            de_objective.last_error = None  # type: ignore[attr-defined]
            de_objective.last_error_provenance = None  # type: ignore[attr-defined]

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
            if not success and "maximum number of iterations has been exceeded" in message.lower():
                success = True
            nfev = result.nfev

            try:
                residuals = objective_func(x_opt)
            except FittingCancelled:
                raise
            except FitSimulationError as exc:
                logger.error("Simulation failed when evaluating DE optimum: %s", exc)
                return _build_failed_result(str(exc), failed_params=exc.failed_params or _vector_to_param_dict(x_opt))
            covariance = None
        else:
            least_squares, _de = scipy_loader()
            if method.lower() == "lm":
                scipy_method = "lm"
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
                return _build_failed_result(str(exc), failed_params=exc.failed_params or _vector_to_param_dict(x_opt))
            except Exception:
                residuals = result.fun

            try:
                J = result.jac
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
        return _build_failed_result(str(exc), failed_params=exc.failed_params)
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

    y_model_opt = None
    y_exp_data = getattr(objective_func, "y_exp", None)
    y_exp_arr = None
    if y_exp_data is not None:
        try:
            y_exp_arr = np.asarray(y_exp_data, dtype=float).reshape(-1)
        except Exception:
            y_exp_arr = None
    try:
        residuals_arr = np.asarray(residuals, dtype=float).reshape(-1)
    except Exception:
        residuals_arr = None
    if y_exp_arr is not None and residuals_arr is not None and y_exp_arr.shape == residuals_arr.shape:
        y_model_opt = y_exp_arr + residuals_arr

    uncertainties = None
    if covariance is not None:
        try:
            std_devs = np.sqrt(np.diag(covariance))
            uncertainties = {name: std for name, std in zip(param_names, std_devs)}
        except Exception as exc:
            logger.debug("Failed to calculate uncertainties: %s", exc)

    chi_squared = np.sum(residuals**2) / len(residuals) if len(residuals) > 0 else np.inf
    if y_model_opt is not None and y_exp_arr is not None and y_exp_arr.size > 0:
        ss_res = np.sum((y_exp_arr - y_model_opt) ** 2)
        ss_tot = np.sum((y_exp_arr - np.mean(y_exp_arr)) ** 2)
        r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    else:
        r_squared = float("nan")

    return FitResult(
        success=success,
        parameters=fitted_params,
        uncertainties=uncertainties,
        chi_squared=chi_squared,
        r_squared=r_squared,
        residuals=residuals,
        nfev=nfev,
        message=message,
        covariance=covariance,
    )
