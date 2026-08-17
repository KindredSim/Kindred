from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import logging
from typing import Any, Mapping, Optional

import numpy as np

from kindred.core.simulation_failure import (
    FactualCause,
    factual_cause_from_exception,
)

logger = logging.getLogger(__name__)


class SimulationAlgebraPolicy(str, Enum):
    """Caller-owned algebra handling policy for simulation finalization."""

    GUI_BEST_EFFORT = "gui_best_effort"
    FITTING_STRICT = "fitting_strict"


def _algebra_cause(error: object, *, name: str | None = None) -> FactualCause:
    """Project an algebra error entry directly into the factual cause tree."""

    details: dict[str, object] = {}
    if name is not None:
        details["name"] = str(name)
    else:
        raw_name = getattr(error, "name", None)
        if raw_name is not None:
            details["name"] = str(raw_name)
    for source_name, target_name in (
        ("line", "line"),
        ("col", "col"),
        ("line_text", "line_text"),
    ):
        value = getattr(error, source_name, None)
        if value is not None:
            details[target_name] = value
    code = getattr(error, "code", None)
    message = str(getattr(error, "message", None) or str(error))
    return FactualCause(
        domain="simulation",
        origin="algebra",
        code=str(code) if code is not None else None,
        type=str(getattr(error, "exc_type", None) or type(error).__name__),
        message=message,
        details=details,
    )


@dataclass(frozen=True)
class SimulationAlgebraEvaluation:
    series: dict[str, np.ndarray]
    scalars: dict[str, float]
    errors: tuple[FactualCause, ...]
    warning: FactualCause | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "series",
            {
                str(name): np.array(values, copy=True)
                for name, values in dict(self.series or {}).items()
            },
        )
        object.__setattr__(
            self,
            "scalars",
            {str(name): float(value) for name, value in dict(self.scalars or {}).items()},
        )
        object.__setattr__(
            self,
            "errors",
            tuple(self.errors or ()),
        )
        object.__setattr__(
            self,
            "warning",
            self.warning,
        )
        if any(not isinstance(error, FactualCause) for error in self.errors):
            raise TypeError("Algebra evaluation errors must be FactualCause values.")
        if self.warning is not None and not isinstance(self.warning, FactualCause):
            raise TypeError("Algebra evaluation warning must be a FactualCause.")


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


def evaluate_simulation_algebra(
    policy: SimulationAlgebraPolicy | str,
    mechanism: object,
    *,
    t: np.ndarray,
    species_series: Mapping[str, np.ndarray],
    initials: Mapping[str, float],
    gui_evaluator: Optional[Any] = None,
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
        cause = factual_cause_from_exception(
            exc,
            domain="simulation",
            origin="algebra_evaluation",
        )
        warning = FactualCause(
            domain=cause.domain,
            origin=cause.origin,
            code=cause.code,
            type=cause.type,
            message=cause.message,
            context=cause.context,
            details={**cause.details, "stage": "algebra_evaluation"},
            suggestion=cause.suggestion,
            examples=cause.examples,
            cause=cause.cause,
        )
        return SimulationAlgebraEvaluation(
            series={},
            scalars={},
            # The warning is the sole factual cause for this failure.  Keeping
            # a second serialized error here would duplicate its message.
            errors=(),
            warning=warning,
        )

    algebra_errors: list[FactualCause] = []
    for error_entry in (errors or []):
        try:
            algebra_errors.append(_algebra_cause(error_entry))
        except Exception as exc:
            logger.debug("Failed to serialize algebra error entry: %s", exc, exc_info=True)
    return SimulationAlgebraEvaluation(
        series={str(name): np.asarray(values, dtype=float).reshape(-1) for name, values in (algebra_series or {}).items()},
        scalars={str(name): float(value) for name, value in dict(algebra_scalars or {}).items()},
        errors=tuple(algebra_errors),
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
