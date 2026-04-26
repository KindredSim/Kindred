from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Dict, Mapping, Optional

import numpy as np

from kindred.core.exceptions import ErrorContext, FitSimulationError
from kindred.core.simulation_failure import serialize_algebra_error, simulation_failure_from_exception
from kindred.core.simulation_plan import SimulationAlgebraPolicy, SimulationPlan

logger = logging.getLogger(__name__)


def _copy_evaluation_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return np.array(value, copy=True)
    if isinstance(value, Mapping):
        return {str(key): _copy_evaluation_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_evaluation_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy_evaluation_value(item) for item in value)
    return value


@dataclass(frozen=True)
class SimulationAlgebraEvaluation:
    series: Dict[str, np.ndarray]
    scalars: Dict[str, float]
    errors: list[dict[str, Any]]
    warning: Optional[dict[str, Any]] = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "series",
            {str(name): np.array(values, copy=True) for name, values in dict(self.series or {}).items()},
        )
        object.__setattr__(
            self,
            "scalars",
            {str(name): float(value) for name, value in dict(self.scalars or {}).items()},
        )
        object.__setattr__(
            self,
            "errors",
            [_copy_evaluation_value(error) for error in list(self.errors or [])],
        )
        object.__setattr__(
            self,
            "warning",
            _copy_evaluation_value(self.warning) if self.warning is not None else None,
        )


def coerce_simulation_algebra_policy(
    value: SimulationAlgebraPolicy | str | None,
    *,
    default: SimulationAlgebraPolicy,
) -> SimulationAlgebraPolicy:
    if value is None:
        return default
    if isinstance(value, SimulationAlgebraPolicy):
        return value
    return SimulationAlgebraPolicy(str(value))


def algebra_policy_from_simulation_plan(
    value: SimulationPlan | Mapping[str, Any] | None,
    *,
    default: SimulationAlgebraPolicy,
) -> SimulationAlgebraPolicy:
    if value is None:
        return default
    plan = value if isinstance(value, SimulationPlan) else SimulationPlan.from_payload(value)
    return coerce_simulation_algebra_policy(plan.algebra_policy, default=default)


def evaluate_simulation_algebra(
    policy: SimulationAlgebraPolicy | str,
    mechanism: object,
    *,
    t: np.ndarray,
    species_series: Mapping[str, np.ndarray],
    initials: Mapping[str, float],
    gui_evaluator: Optional[Any] = None,
    batch_evaluator: Optional[Any] = None,
) -> SimulationAlgebraEvaluation:
    algebra_policy = coerce_simulation_algebra_policy(
        policy,
        default=SimulationAlgebraPolicy.GUI_BEST_EFFORT,
    )
    if algebra_policy is SimulationAlgebraPolicy.FITTING_STRICT:
        raise ValueError("fitting_strict algebra policy is not a best-effort simulation policy.")
    if algebra_policy is SimulationAlgebraPolicy.GUI_BEST_EFFORT:
        return _evaluate_gui_best_effort_algebra(
            mechanism,
            t=t,
            species_series=species_series,
            initials=initials,
            evaluator=gui_evaluator,
        )
    if algebra_policy is SimulationAlgebraPolicy.BATCH_BEST_EFFORT:
        return _evaluate_batch_best_effort_algebra(
            mechanism,
            t=t,
            species_series=species_series,
            initials=initials,
            evaluator=batch_evaluator,
        )
    raise ValueError(f"Unsupported simulation algebra policy: {algebra_policy!r}")


def _evaluate_gui_best_effort_algebra(
    mechanism: object,
    *,
    t: np.ndarray,
    species_series: Mapping[str, np.ndarray],
    initials: Mapping[str, float],
    evaluator: Optional[Any],
) -> SimulationAlgebraEvaluation:
    if evaluator is None:
        from kindred.core.algebra.simulation_series import (
            evaluate_algebra_series_for_simulation_with_errors,
        )

        evaluator = evaluate_algebra_series_for_simulation_with_errors

    try:
        algebra_series, algebra_scalars, errors = evaluator(
            mechanism,
            t=np.asarray(t, dtype=float).reshape(-1),
            species_series={str(name): np.asarray(values, dtype=float).reshape(-1) for name, values in species_series.items()},
            initials={str(name): float(value) for name, value in dict(initials or {}).items()},
        )
    except Exception as exc:
        logger.warning("Algebra evaluation failed: %s", exc, exc_info=True)
        return SimulationAlgebraEvaluation(
            series={},
            scalars={},
            errors=[serialize_algebra_error(exc, name="__algebra__")],
            warning=simulation_failure_from_exception(
                exc,
                kind="algebra_warning",
                details={"stage": "algebra_evaluation"},
            ),
        )

    algebra_errors: list[dict[str, Any]] = []
    for error_entry in (errors or []):
        try:
            algebra_errors.append(serialize_algebra_error(error_entry))
        except Exception as exc:
            logger.debug("Failed to serialize algebra error entry: %s", exc, exc_info=True)
    return SimulationAlgebraEvaluation(
        series={str(name): np.asarray(values, dtype=float).reshape(-1) for name, values in (algebra_series or {}).items()},
        scalars={str(name): float(value) for name, value in dict(algebra_scalars or {}).items()},
        errors=algebra_errors,
        warning=None,
    )


def _evaluate_batch_best_effort_algebra(
    mechanism: object,
    *,
    t: np.ndarray,
    species_series: Mapping[str, np.ndarray],
    initials: Mapping[str, float],
    evaluator: Optional[Any],
) -> SimulationAlgebraEvaluation:
    if evaluator is None:
        from kindred.core.algebra.simulation_series import evaluate_algebra_series_for_simulation

        evaluator = evaluate_algebra_series_for_simulation

    try:
        algebra_series, algebra_scalars = evaluator(
            mechanism,
            t=np.asarray(t, dtype=float).reshape(-1),
            species_series={str(name): np.asarray(values, dtype=float).reshape(-1) for name, values in species_series.items()},
            initials={str(name): float(value) for name, value in dict(initials or {}).items()},
        )
    except Exception as exc:
        logger.warning("Algebra evaluation failed in batch worker: %s", exc)
        return SimulationAlgebraEvaluation(
            series={},
            scalars={},
            errors=[serialize_algebra_error(exc, name="__algebra__")],
            warning=None,
        )
    return SimulationAlgebraEvaluation(
        series={str(name): np.asarray(values, dtype=float).reshape(-1) for name, values in (algebra_series or {}).items()},
        scalars={str(name): float(value) for name, value in dict(algebra_scalars or {}).items()},
        errors=[],
        warning=None,
    )


def ensure_fitting_strict_algebra_policy(policy: SimulationAlgebraPolicy | str) -> SimulationAlgebraPolicy:
    algebra_policy = coerce_simulation_algebra_policy(
        policy,
        default=SimulationAlgebraPolicy.FITTING_STRICT,
    )
    if algebra_policy is not SimulationAlgebraPolicy.FITTING_STRICT:
        raise ValueError("Fitting algebra handling requires a fitting_strict algebra policy.")
    return algebra_policy


def fitting_algebra_error_is_fatal(exc: BaseException) -> bool:
    from kindred.core.algebra.errors import (
        AlgebraNameError,
        AlgebraShadowError,
        AlgebraSyntaxError,
    )

    return isinstance(exc, (AlgebraNameError, AlgebraShadowError, AlgebraSyntaxError))


def fitting_strict_parse_error(exc: BaseException, *, message_prefix: str) -> FitSimulationError:
    return FitSimulationError(
        f"{message_prefix}: {exc}",
        details={"fatal": True},
    )


def fitting_strict_time_ref_error(stmt: object) -> FitSimulationError:
    return FitSimulationError(
        "Algebra baseline references like [A](T0) are not supported for fitting (v1).",
        details={"fatal": True},
        context=ErrorContext(
            line=getattr(stmt, "line", None),
            col=getattr(stmt, "col", None),
            line_text=getattr(stmt, "line_text", None),
        ),
    )


def fitting_strict_evaluation_error(
    exc: BaseException,
    *,
    message_prefix: str,
    failed_params: Optional[Dict[str, float]] = None,
) -> FitSimulationError:
    from kindred.core.algebra.errors import AlgebraError

    if isinstance(exc, AlgebraError):
        return FitSimulationError(
            f"{message_prefix}: {exc}",
            failed_params=failed_params,
            details={"fatal": bool(fitting_algebra_error_is_fatal(exc))},
            context=ErrorContext(line=exc.line, col=exc.col, line_text=exc.line_text),
        )
    return FitSimulationError(
        f"{message_prefix}: {exc}",
        failed_params=failed_params,
        details={"fatal": False},
    )
