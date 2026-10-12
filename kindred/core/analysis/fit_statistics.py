"""Finite, explicitly available Fit statistics and weight scaling."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping, Optional

import numpy as np

__all__ = [
    "FitMetric",
    "finite_positive_mean_normalize",
    "finite_sum_of_squares",
    "fit_error_metrics",
    "r_squared_metric",
    "root_mean_square",
]


@dataclass(frozen=True, slots=True)
class FitMetric:
    """One terminal metric that is finite or explicitly unavailable."""

    value: Optional[float]
    unavailable_reason: Optional[str] = None

    # Its producers check finiteness first and name each unavailable reason.

    @property
    def available(self) -> bool:
        return self.value is not None

    @classmethod
    def finite(cls, value: object) -> "FitMetric":
        return cls(value=float(value))

    @classmethod
    def unavailable(cls, reason: str) -> "FitMetric":
        return cls(value=None, unavailable_reason=reason)

    def to_payload(self) -> dict[str, object]:
        return {
            "value": self.value,
            "available": self.available,
            "unavailable_reason": self.unavailable_reason,
        }


@dataclass(frozen=True, slots=True)
class FitErrorMetrics:
    """Weighted terminal metrics calculated from one residual vector."""

    weighted_sse: FitMetric
    weighted_mse: FitMetric
    weighted_rmse: FitMetric
    weighted_mae: FitMetric


def finite_positive_mean_normalize(
    values: Mapping[str, object],
) -> dict[str, float]:
    """Normalize positive finite values to arithmetic mean one.

    Division by the largest input occurs before summation and multiplication,
    so finite inputs cannot overflow merely because their sum does.  Ratios are
    preserved in floating-point space; an unrepresentable underflow is rejected
    explicitly rather than silently replaced with equal weights.
    """

    # Callers pass a nonempty map of admitted finite positive weights.
    cleaned = {name: float(value) for name, value in values.items()}
    maximum = max(cleaned.values())
    scaled = {name: value / maximum for name, value in cleaned.items()}
    scaled_total = math.fsum(scaled.values())
    count = float(len(scaled))
    return {
        name: (value / scaled_total) * count
        for name, value in scaled.items()
    }


def finite_sum_of_squares(values: Iterable[object]) -> float:
    """Return the SSE, or ``inf`` when it is not a finite float."""

    array = np.asarray(tuple(values), dtype=float).reshape(-1)
    if array.size == 0:
        return 0.0
    if not np.all(np.isfinite(array)):
        return math.inf
    maximum = float(np.max(np.abs(array)))
    if maximum == 0.0:
        return 0.0
    scaled_square_sum = math.fsum(
        float(value / maximum) ** 2 for value in array
    )
    if maximum > math.sqrt(np.finfo(float).max / scaled_square_sum):
        return math.inf
    return float((maximum * maximum) * scaled_square_sum)


def root_mean_square(residuals: object) -> FitMetric:
    """Root-mean-square of one admitted, finite, nonempty residual vector."""

    values = np.asarray(residuals, dtype=float).reshape(-1)
    return FitMetric.finite(math.sqrt(finite_sum_of_squares(values) / float(values.size)))


def fit_error_metrics(weighted_residuals: object) -> FitErrorMetrics:
    residuals = np.asarray(weighted_residuals, dtype=float).reshape(-1)
    sse = finite_sum_of_squares(residuals)
    mse = sse / float(residuals.size)
    mae = math.fsum(float(abs(value)) for value in residuals) / float(residuals.size)
    return FitErrorMetrics(
        weighted_sse=FitMetric.finite(sse),
        weighted_mse=FitMetric.finite(mse),
        weighted_rmse=FitMetric.finite(math.sqrt(mse)),
        weighted_mae=FitMetric.finite(mae),
    )


def r_squared_metric(observed: object, residuals: object) -> FitMetric:
    observations = np.asarray(observed, dtype=float).reshape(-1)
    residual_array = np.asarray(residuals, dtype=float).reshape(-1)
    if bool(np.all(observations == observations[0])):
        return FitMetric.unavailable("observations are constant")
    centered = observations - float(np.mean(observations))
    return FitMetric.finite(
        1.0 - finite_sum_of_squares(residual_array) / finite_sum_of_squares(centered)
    )
