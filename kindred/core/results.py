"""Compact fixed-policy concentration-time statistics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kindred.core.time_grid import is_uniform_time_grid

__all__ = ["CtcResult", "integrate_ctc"]


@dataclass(frozen=True, slots=True)
class CtcResult:
    """Typed scalar CTC outcome.

    ``value`` is present only for an available finite result.  Numerical input
    or accumulation overflow is an ordinary unavailable projection, not an
    exception that callers must classify heuristically.
    """

    value: float | None
    reason: str | None = None

    @property
    def available(self) -> bool:
        return self.value is not None

    @classmethod
    def unavailable(cls, reason: str = "nonfinite") -> "CtcResult":
        return cls(None, str(reason))


def _composite_trapezoid(time: np.ndarray, values: np.ndarray) -> float:
    with np.errstate(over="ignore", invalid="ignore"):
        return float(np.sum(np.diff(time) * 0.5 * (values[:-1] + values[1:])))


def _simpson13_uniform(time: np.ndarray, values: np.ndarray) -> float:
    # Callers pass an even number of intervals.
    intervals = time.size - 1
    h = (time[-1] - time[0]) / intervals
    with np.errstate(over="ignore", invalid="ignore"):
        weighted = (
            values[0]
            + values[-1]
            + 4.0 * np.sum(values[1:-1:2])
            + 2.0 * np.sum(values[2:-2:2])
        )
        return float(weighted * h / 3.0)


def integrate_ctc(t: np.ndarray, y: np.ndarray) -> CtcResult:
    """Integrate one baseline-corrected curve using fixed internal policy.

    The policy is intentionally not caller-configurable: uniform grids use
    Simpson 1/3 with a hard-coded relative tolerance of ``1e-6`` and a 3/8
    tail for an odd interval count; nonuniform grids use composite trapezoid.
    The canonical result supplies finite, strictly increasing times and finite
    species values of equal length; finite-shape overflow produces a typed
    unavailable result.
    """

    time = np.asarray(t, dtype=float)
    values = np.asarray(y, dtype=float)
    if time.size < 2:
        return CtcResult(0.0)
    uniform = is_uniform_time_grid(time, 1e-6)
    if not uniform:
        value = _composite_trapezoid(time, values)
    else:
        intervals = time.size - 1
        if intervals % 2 == 0:
            value = _simpson13_uniform(time, values)
        elif time.size >= 4:
            head_points = time.size - 3
            head_value = (
                _simpson13_uniform(time[:head_points], values[:head_points])
                if head_points >= 3
                else 0.0
            )
            h = float(np.median(np.diff(time)))
            with np.errstate(over="ignore", invalid="ignore"):
                tail_value = (
                    3.0 * h * (
                        values[-4]
                        + 3.0 * values[-3]
                        + 3.0 * values[-2]
                        + values[-1]
                    ) / 8.0
                )
            value = float(head_value + tail_value)
        else:
            value = _composite_trapezoid(time, values)
    if not np.isfinite(value):
        return CtcResult.unavailable("nonfinite result")
    return CtcResult(value)
