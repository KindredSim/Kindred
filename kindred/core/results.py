"""Native-grid concentration-time integration used by result statistics."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from kindred.core.time_grid import is_uniform_time_grid

__all__ = ["integrate_ctc"]


def _trapezoid_segment(h: float, y0: float, y1: float) -> float:
    return 0.5 * h * (y0 + y1)


def _composite_trapezoid(t: np.ndarray, y: np.ndarray) -> float:
    return float(np.sum(np.diff(t) * 0.5 * (y[:-1] + y[1:])))


def _simpson13_uniform(t: np.ndarray, y: np.ndarray) -> float:
    intervals = t.size - 1
    if intervals % 2:
        raise ValueError("Simpson 1/3 requires an even number of intervals.")
    h = (t[-1] - t[0]) / intervals
    weighted = (
        y[0]
        + y[-1]
        + 4.0 * np.sum(y[1:-1:2])
        + 2.0 * np.sum(y[2:-2:2])
    )
    return float(weighted * h / 3.0)


def _tail_strategy(value: str | None) -> str:
    normalized = str(value or "").strip().casefold()
    if (
        not normalized
        or normalized in {"38", "3/8", "three-eighths", "three_eighths"}
        or normalized.startswith("38")
    ):
        return "38"
    return "trapezoid"


def integrate_ctc(
    t: np.ndarray,
    y: np.ndarray,
    *,
    eps_uniform: float | None = None,
    uniformity_eps: float | None = None,
    tail_strategy: str | None = None,
    policy: Mapping[str, Any] | None = None,
) -> tuple[float, str, bool, float, str]:
    """Preserve clean-baseline Simpson/trapezoid selection and metadata."""

    time = np.asarray(t, dtype=float)
    values = np.asarray(y, dtype=float)
    if time.ndim != 1 or values.ndim != 1 or time.size != values.size:
        raise ValueError("t and y must be 1D arrays of equal length.")
    if np.any(~np.isfinite(time)) or np.any(~np.isfinite(values)):
        raise ValueError("t and y must contain finite values.")

    if policy:
        uniformity_eps = policy.get("uniformity_eps", uniformity_eps)
        tail_strategy = policy.get("tail_strategy", tail_strategy)
    if eps_uniform is not None:
        uniformity_eps = eps_uniform
    used_eps = float(1e-6 if uniformity_eps is None else uniformity_eps)
    used_tail = _tail_strategy(tail_strategy)

    if time.size < 2:
        return 0.0, "Trapezoidal", True, used_eps, used_tail

    monotonic = bool(np.all(np.diff(time) > 0.0))
    uniform = monotonic and is_uniform_time_grid(time, used_eps)
    if not uniform:
        return (
            _composite_trapezoid(time, values),
            "Trapezoidal",
            False,
            used_eps,
            used_tail,
        )

    intervals = time.size - 1
    if intervals % 2 == 0:
        return (
            _simpson13_uniform(time, values),
            "Simpson13",
            True,
            used_eps,
            used_tail,
        )

    h = float(np.median(np.diff(time)))
    if used_tail == "38" and time.size >= 4:
        head_points = time.size - 3
        head_value = (
            _simpson13_uniform(time[:head_points], values[:head_points])
            if head_points >= 3
            else 0.0
        )
        tail_value = (
            3.0
            * h
            * (
                values[-4]
                + 3.0 * values[-3]
                + 3.0 * values[-2]
                + values[-1]
            )
            / 8.0
        )
        return (
            float(head_value + tail_value),
            "Simpson13+38",
            True,
            used_eps,
            used_tail,
        )

    head_value = (
        _simpson13_uniform(time[:-1], values[:-1])
        if time.size - 1 >= 3
        else 0.0
    )
    tail_value = _trapezoid_segment(
        time[-1] - time[-2],
        values[-2],
        values[-1],
    )
    return (
        float(head_value + tail_value),
        "Simpson13+TrapezoidTail",
        True,
        used_eps,
        used_tail,
    )
