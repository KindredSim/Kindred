from __future__ import annotations

from typing import Optional

import numpy as np

__all__ = ["compute_sampled_indices", "compute_windowed_indices"]


def compute_windowed_indices(*, t: np.ndarray, t_min: float, t_max: float) -> np.ndarray:
    """
    Return indices of points within the inclusive time window [t_min, t_max].

    The returned indices are strictly increasing and refer to the original `t` array.
    """
    t_values = np.asarray(t, dtype=float).reshape(-1)
    if t_values.size == 0:
        return np.asarray([], dtype=int)
    mask = (t_values >= float(t_min)) & (t_values <= float(t_max))
    return np.nonzero(mask)[0].astype(int, copy=False)


def compute_sampled_indices(
    *,
    t: np.ndarray,
    t_min: float,
    t_max: float,
    n_points: Optional[int],
) -> np.ndarray:
    """
    Compute deterministic sampling indices for a time axis.

    Rules:
    - Window: keep indices where t_min <= t <= t_max (inclusive).
    - Downsample: after windowing, if n_points is None/0/"All" keep all windowed points.
      Otherwise select exactly n_points *existing* points evenly across the windowed
      index list, always including the window endpoints. No averaging, no fabrication.
    """
    if float(t_min) > float(t_max):
        raise ValueError("t_min must be <= t_max.")

    windowed = compute_windowed_indices(t=t, t_min=float(t_min), t_max=float(t_max))
    if windowed.size == 0:
        raise ValueError("Sampling window contains no points.")

    if n_points is None:
        return windowed
    try:
        n_int = int(n_points)
    except Exception as exc:  # pragma: no cover
        raise ValueError("n_points must be an int or None.") from exc

    if n_int <= 0 or n_int >= int(windowed.size):
        return windowed
    if n_int < 2:
        raise ValueError("n_points must be All/None or >= 2.")
    if n_int > int(windowed.size):
        raise ValueError("n_points must be <= the number of windowed points.")

    m = int(windowed.size)
    denom = n_int - 1
    max_pos = m - 1
    positions = np.fromiter(
        ((2 * k * max_pos + denom) // (2 * denom) for k in range(n_int)),
        dtype=int,
        count=n_int,
    )
    return windowed[positions]

