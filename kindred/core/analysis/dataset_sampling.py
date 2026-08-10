"""Deterministic selection of existing experimental observation rows."""

from __future__ import annotations

from typing import Optional

import numpy as np

__all__ = ["compute_sampled_indices", "compute_windowed_indices"]


def compute_windowed_indices(
    *,
    t: np.ndarray,
    t_min: float,
    t_max: float,
) -> np.ndarray:
    """Return original-array indices in chronological order inside the window."""
    values = np.asarray(t, dtype=float).reshape(-1)
    if values.size == 0:
        return np.asarray([], dtype=int)
    mask = (values >= float(t_min)) & (values <= float(t_max))
    eligible = np.nonzero(mask)[0].astype(int, copy=False)
    if eligible.size < 2:
        return eligible
    chronological = np.argsort(values[eligible], kind="stable")
    return eligible[chronological]


def compute_sampled_indices(
    *,
    t: np.ndarray,
    t_min: float,
    t_max: float,
    n_points: Optional[int],
) -> np.ndarray:
    """Select existing rows evenly; never interpolate, average, or fabricate."""
    if float(t_min) > float(t_max):
        raise ValueError("t_min must be <= t_max.")
    windowed = compute_windowed_indices(
        t=t,
        t_min=float(t_min),
        t_max=float(t_max),
    )
    if windowed.size == 0:
        raise ValueError("Sampling window contains no points.")
    if n_points is None:
        return windowed
    try:
        requested = int(n_points)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("n_points must be an int or None.") from exc
    if requested <= 0:
        return windowed
    if requested < 2:
        raise ValueError("n_points must be All/None or >= 2.")
    if requested >= int(windowed.size):
        return windowed

    count = int(windowed.size)
    denominator = requested - 1
    last = count - 1
    positions = np.fromiter(
        (
            (2 * index * last + denominator) // (2 * denominator)
            for index in range(requested)
        ),
        dtype=int,
        count=requested,
    )
    return windowed[positions]
