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

    def __post_init__(self) -> None:
        if self.value is None:
            reason = str(self.unavailable_reason or "").strip()
            if not reason:
                raise ValueError("An unavailable Fit metric requires a reason.")
            object.__setattr__(self, "unavailable_reason", reason)
            return
        value = float(self.value)
        if not math.isfinite(value):
            raise ValueError("An available Fit metric must be finite.")
        if self.unavailable_reason is not None:
            raise ValueError("An available Fit metric cannot have an unavailable reason.")
        object.__setattr__(self, "value", value)

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

    cleaned: dict[str, float] = {}
    for raw_name, raw_value in values.items():
        name = str(raw_name)
        value = float(raw_value)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"Weight for {name!r} must be finite and positive.")
        cleaned[name] = value
    if not cleaned:
        return {}
    maximum = max(cleaned.values())
    scaled = {name: value / maximum for name, value in cleaned.items()}
    if any(value <= 0.0 for value in scaled.values()):
        raise ValueError(
            "Finite positive weights span a ratio too wide to preserve during normalization."
        )
    scaled_total = math.fsum(scaled.values())
    if not math.isfinite(scaled_total) or scaled_total <= 0.0:
        raise ValueError("Scaled weight sum must be finite and positive.")
    count = float(len(scaled))
    normalized = {
        name: (value / scaled_total) * count
        for name, value in scaled.items()
    }
    if any(not math.isfinite(value) or value <= 0.0 for value in normalized.values()):
        raise ValueError("Normalized weights must remain finite and positive.")
    return normalized


def finite_sum_of_squares(values: Iterable[object]) -> Optional[float]:
    """Return a finite SSE, or ``None`` when the exact float result overflows."""

    array = np.asarray(tuple(values), dtype=float).reshape(-1)
    if array.size == 0:
        return 0.0
    if not np.all(np.isfinite(array)):
        return None
    maximum = float(np.max(np.abs(array)))
    if maximum == 0.0:
        return 0.0
    scaled_square_sum = math.fsum(
        float(value / maximum) ** 2 for value in array
    )
    if maximum > math.sqrt(np.finfo(float).max / scaled_square_sum):
        return None
    result = (maximum * maximum) * scaled_square_sum
    return float(result) if math.isfinite(result) else None


def root_mean_square(residuals: object) -> FitMetric:
    """Root-mean-square of one admitted, finite, nonempty residual vector."""

    values = np.asarray(residuals, dtype=float).reshape(-1)
    sse = finite_sum_of_squares(values)
    if sse is None:
        return FitMetric.unavailable("residual sum of squares overflowed")
    return FitMetric.finite(math.sqrt(sse / float(values.size)))


def fit_error_metrics(weighted_residuals: object) -> FitErrorMetrics:
    residuals = np.asarray(weighted_residuals, dtype=float).reshape(-1)
    if residuals.size == 0:
        unavailable = FitMetric.unavailable("no residual observations")
        return FitErrorMetrics(unavailable, unavailable, unavailable, unavailable)
    if not np.all(np.isfinite(residuals)):
        unavailable = FitMetric.unavailable("residuals are nonfinite")
        return FitErrorMetrics(unavailable, unavailable, unavailable, unavailable)
    sse = finite_sum_of_squares(residuals)
    if sse is None:
        unavailable = FitMetric.unavailable("residual sum of squares overflowed")
        return FitErrorMetrics(unavailable, unavailable, unavailable, unavailable)
    mse = sse / float(residuals.size)
    mae = math.fsum(float(abs(value)) for value in residuals) / float(residuals.size)
    if not (math.isfinite(mse) and math.isfinite(mae)):
        unavailable = FitMetric.unavailable("weighted metric arithmetic overflowed")
        return FitErrorMetrics(unavailable, unavailable, unavailable, unavailable)
    return FitErrorMetrics(
        weighted_sse=FitMetric.finite(sse),
        weighted_mse=FitMetric.finite(mse),
        weighted_rmse=FitMetric.finite(math.sqrt(mse)),
        weighted_mae=FitMetric.finite(mae),
    )


def r_squared_metric(observed: object, residuals: object) -> FitMetric:
    observations = np.asarray(observed, dtype=float).reshape(-1)
    residual_array = np.asarray(residuals, dtype=float).reshape(-1)
    if observations.size == 0 or observations.size != residual_array.size:
        return FitMetric.unavailable("observations and residuals are not aligned")
    if not (
        np.all(np.isfinite(observations))
        and np.all(np.isfinite(residual_array))
    ):
        return FitMetric.unavailable("observations or residuals are nonfinite")
    if bool(np.all(observations == observations[0])):
        return FitMetric.unavailable("observations are constant")
    centered = observations - float(np.mean(observations))
    ss_res = finite_sum_of_squares(residual_array)
    ss_tot = finite_sum_of_squares(centered)
    if ss_res is None or ss_tot is None or ss_tot <= 0.0:
        return FitMetric.unavailable("R-squared arithmetic overflowed")
    value = 1.0 - (ss_res / ss_tot)
    if not math.isfinite(value):
        return FitMetric.unavailable("R-squared arithmetic was nonfinite")
    return FitMetric.finite(value)
