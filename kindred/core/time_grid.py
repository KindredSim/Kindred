"""
Canonical time-grid policies for Kindred.

This module centralizes:
- Time grid construction (dt / N policies)
- Uniform-grid detection using a median-spacing relative-deviation check
"""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np

__all__ = ["build_time_grid", "is_uniform_time_grid"]

_EPS_T = 1e-15  # Guard the relative-uniformity denominator on tiny time steps.


def build_time_grid(t0: float, t1: float, grid: Mapping[str, float | int]) -> np.ndarray:
    """
    Build a deterministic, strictly increasing evaluation grid on [t0, t1].

    Supported grid policies:
      - {"dt": float}: use ceil((t1 - t0)/dt) intervals and adjust spacing to land exactly on t1.
      - {"N": int}: use N points via linspace (N >= 2).
    """
    t0 = float(t0)
    t1 = float(t1)
    if not (math.isfinite(t0) and math.isfinite(t1)) or not (t1 > t0):
        raise ValueError("t_span must be finite with t1 > t0")

    if "dt" in grid:
        h = float(grid["dt"])
        if h <= 0 or not math.isfinite(h):
            raise ValueError("grid['dt'] must be positive and finite")
        n_int = max(1, int(math.ceil((t1 - t0) / h)))
        h_adj = (t1 - t0) / n_int
        t = t0 + h_adj * np.arange(n_int + 1, dtype=float)
        t[-1] = t1
        return t

    if "N" in grid:
        n = int(grid["N"])
        if n < 2:
            raise ValueError("grid['N'] must be >= 2")
        return np.linspace(t0, t1, n, dtype=float)

    raise ValueError("grid must be {'dt': float} or {'N': int}")


def is_uniform_time_grid(t: np.ndarray, eps: float) -> bool:
    """
    Return True if t is a strictly increasing (or size<3) grid with near-constant spacing.

    Uniformity uses median(dt) and relative deviations against that median spacing.
    """
    if t.size < 3:
        return True
    dt = np.diff(t)
    if not np.all(np.isfinite(dt)) or np.any(dt <= 0):
        return False
    med = float(np.median(dt))
    denom = max(med, _EPS_T)
    r = np.abs(dt - med) / denom
    return bool(np.max(r) <= eps)
