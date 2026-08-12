"""
Optimization primitives for parameter fitting.

This module intentionally does *not* own simulation preparation or DSL parsing; it only
operates on an objective callable that returns residual vectors.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import numpy as np

from kindred.core.analysis.fit_statistics import finite_sum_of_squares
from kindred.core.analysis.fit_evaluation_outcome import FitEvaluationOutcome
from kindred.core.exceptions import (
    FitFinalReplayError,
    FittingCancelled,
    FitSimulationError,
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
from kindred.core.project_document import validate_fit_method

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


def _is_expected_candidate_domain_error(exc: BaseException) -> bool:
    """Return true only for an evaluator's explicit candidate-domain outcome."""

    return bool(
        isinstance(exc, FitSimulationError)
        and isinstance(exc.details, Mapping)
        and exc.details.get("candidate_domain") is True
        and exc.details.get("fatal") is False
    )


def _canonical_terminal_evaluation(
    objective: object,
    parameters: np.ndarray,
    *,
    optimizer_success: bool,
    optimizer_nfev: int,
) -> tuple[np.ndarray, FitEvaluationOutcome, int]:
    """Run exactly one terminal evaluation at the optimizer's returned point."""

    vector = np.asarray(parameters, dtype=float).reshape(-1)
    evaluator = getattr(objective, "evaluate_outcome", None)
    if callable(evaluator):
        outcome = evaluator(vector)
        if not isinstance(outcome, FitEvaluationOutcome):
            raise TypeError("Fit objective evaluate_outcome() returned an invalid DTO.")
        transport = outcome.transport_residuals
        if transport is None:
            transport = outcome.residuals
        if transport is None:
            raise OptimizationError(
                "Canonical terminal Fit evaluation omitted optimizer transport residuals."
            )
        residuals = np.asarray(transport, dtype=float).reshape(-1)
        terminal_status = (
            "nonconverged"
            if outcome.status == "converged" and not optimizer_success
            else outcome.status
        )
        cause = outcome.cause
        if outcome.status == "candidate_domain_rejected":
            typed_cause = outcome.cause
            if not isinstance(typed_cause, FitSimulationError):
                raise OptimizationError(
                    "Candidate-domain terminal replay omitted its typed cause."
                )
            cause = FitFinalReplayError(
                "Final Fit replay rejected the optimizer's terminal candidate.",
                typed_cause=typed_cause,
                stage="optimizer_terminal_replay",
            )
            terminal_status = "fatal"
        outcome = FitEvaluationOutcome(
            terminal_status,
            residuals=outcome.residuals,
            transport_residuals=residuals,
            cause=cause,
            provenance={
                **dict(outcome.provenance),
                "stage": "optimizer_terminal_replay",
                "canonical_replay": True,
                "optimizer_reported_convergence": bool(optimizer_success),
                "optimizer_nfev": int(optimizer_nfev),
                "terminal_replay_count": 1,
                "scipy_nfev": int(optimizer_nfev),
            },
        )
        return residuals, outcome, 1

    residuals = np.asarray(objective(vector), dtype=float).reshape(-1)
    if residuals.size == 0 or not np.all(np.isfinite(residuals)):
        raise OptimizationError(
            "Canonical terminal Fit evaluation returned empty or non-finite residuals."
        )
    outcome = (
        FitEvaluationOutcome.converged(
            residuals,
            provenance={
                "stage": "optimizer_terminal_replay",
                "canonical_replay": True,
                "optimizer_reported_convergence": bool(optimizer_success),
            },
        )
        if optimizer_success
        else FitEvaluationOutcome.nonconverged(
            residuals,
            provenance={
                "stage": "optimizer_terminal_replay",
                "canonical_replay": True,
                "optimizer_reported_convergence": False,
            },
        )
    )
    return residuals, outcome, 1


@dataclass
class FitResult:
    """Results from parameter fitting."""

    success: bool
    parameters: Dict[str, float]
    residuals: np.ndarray
    nfev: int
    message: str
    scipy_nfev: int | None = None
    terminal_replay_count: int = 0
    objective_call_count: int | None = None
    identifiability_warning: Optional[str] = None
    evaluation: FitEvaluationOutcome | None = None

    def __post_init__(self) -> None:
        self.parameters = {
            str(name): float(value) for name, value in self.parameters.items()
        }
        self.residuals = np.asarray(self.residuals, dtype=float).reshape(-1).copy()
        self.nfev = int(self.nfev)
        if self.scipy_nfev is None:
            self.scipy_nfev = self.nfev
        self.scipy_nfev = int(self.scipy_nfev)
        self.terminal_replay_count = int(self.terminal_replay_count)
        if self.terminal_replay_count < 0:
            raise ValueError("FitResult terminal replay count cannot be negative.")
        if self.objective_call_count is not None:
            self.objective_call_count = int(self.objective_call_count)
            if self.objective_call_count < 0:
                raise ValueError("FitResult objective-call count cannot be negative.")
        self.success = bool(self.success)
        if self.evaluation is None:
            if self.residuals.size == 0 or not np.all(np.isfinite(self.residuals)):
                raise ValueError("FitResult requires finite optimizer residuals.")
            self.evaluation = (
                FitEvaluationOutcome.converged(self.residuals)
                if self.success
                else FitEvaluationOutcome.nonconverged(self.residuals)
            )
        elif not isinstance(self.evaluation, FitEvaluationOutcome):
            raise TypeError("FitResult.evaluation requires FitEvaluationOutcome.")


def fit_parameters_with_fixed(
    objective_func: Callable[[np.ndarray], np.ndarray],
    initial_params: Dict[str, float],
    *,
    fixed_parameters: Dict[str, float],
    bounds: Optional[Dict[str, Tuple[float, float]]] = None,
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

    initial = {str(name): float(value) for name, value in initial_params.items()}
    fixed = {str(name): float(value) for name, value in fixed_parameters.items()}
    unknown = sorted(set(fixed) - set(initial))
    if unknown:
        raise ValueError(
            "Fixed Fit coordinates are absent from the optimizer layout: "
            + ", ".join(unknown)
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
        residuals, evaluation, _nfev = _canonical_terminal_evaluation(
            objective_func,
            full_vector({}),
            optimizer_success=True,
            optimizer_nfev=0,
        )
        return FitResult(
            success=evaluation.status == "converged",
            parameters={name: fixed[name] for name in names},
            residuals=residuals,
            nfev=0,
            message="All optimizer coordinates were fixed.",
            scipy_nfev=0,
            terminal_replay_count=1,
            evaluation=evaluation,
        )

    def reduced_objective(vector: np.ndarray) -> np.ndarray:
        values = {
            name: float(vector[index])
            for index, name in enumerate(free)
        }
        return objective_func(full_vector(values))

    def reduced_evaluate_outcome(vector: np.ndarray) -> FitEvaluationOutcome:
        values = {
            name: float(vector[index])
            for index, name in enumerate(free)
        }
        full = full_vector(values)
        evaluator = getattr(objective_func, "evaluate_outcome", None)
        if callable(evaluator):
            outcome = evaluator(full)
            if not isinstance(outcome, FitEvaluationOutcome):
                raise TypeError("Fit objective evaluate_outcome() returned an invalid DTO.")
            return outcome
        residuals = np.asarray(objective_func(full), dtype=float).reshape(-1)
        return FitEvaluationOutcome.converged(residuals)

    reduced_objective.evaluate_outcome = reduced_evaluate_outcome  # type: ignore[attr-defined]

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
        residuals=reduced.residuals,
        nfev=reduced.nfev,
        message=reduced.message,
        identifiability_warning=reduced.identifiability_warning,
        evaluation=reduced.evaluation,
    )


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

    method = validate_fit_method(method)
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
    terminal_outcome: FitEvaluationOutcome | None = None
    terminal_replay_count = 0

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
                    finite_score = finite_sum_of_squares(residuals)
                    score = (
                        float(effective_penalty)
                        if finite_score is None
                        else float(finite_score)
                    )
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

            residuals, terminal_outcome, nfev = _canonical_terminal_evaluation(
                objective_func,
                x_opt,
                optimizer_success=success,
                optimizer_nfev=de_evaluation_budget.nfev,
            )
            terminal_replay_count = nfev
            scipy_nfev = int(de_evaluation_budget.nfev)
        else:
            least_squares, _de = scipy_loader()

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
            residuals, terminal_outcome, nfev = _canonical_terminal_evaluation(
                objective_func,
                x_opt,
                optimizer_success=success,
                optimizer_nfev=int(result.nfev),
            )
            terminal_replay_count = nfev
            scipy_nfev = int(result.nfev)

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

    terminal_success = bool(success and terminal_outcome.status == "converged")
    if terminal_outcome.status == "candidate_domain_rejected":
        message = (
            f"{message}; terminal candidate was rejected by the admitted domain"
        )

    objective_call_count = getattr(objective_func, "objective_calls", None)
    if objective_call_count is not None:
        objective_call_count = int(objective_call_count)
    return FitResult(
        success=terminal_success,
        parameters=fitted_params,
        residuals=residuals,
        nfev=scipy_nfev,
        message=message,
        scipy_nfev=scipy_nfev,
        terminal_replay_count=terminal_replay_count,
        objective_call_count=objective_call_count,
        identifiability_warning=identifiability_warning,
        evaluation=terminal_outcome,
    )
