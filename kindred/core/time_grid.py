"""
Canonical time-grid policies for Kindred.

This module centralizes:
- Time grid construction (dt / N policies)
- Uniform-grid detection using a median-spacing relative-deviation check
"""

from __future__ import annotations

import math
import numbers
from typing import Any, Mapping

import numpy as np

__all__ = ["build_time_grid", "is_uniform_time_grid", "normalize_time_grid_policy"]

_EPS_T = 1e-15  # Guard the relative-uniformity denominator on tiny time steps.


def normalize_time_grid_policy(grid: Mapping[str, Any]) -> dict[str, float | int]:
    """
    Return the canonical requested time-grid policy.

    A requested grid policy is either exactly {"dt": positive finite float} or
    exactly {"N": int >= 2}. The two forms are intentionally not collapsed into
    one another because they mean different things to callers and cache identity.
    """
    if not isinstance(grid, Mapping) or not grid:
        raise ValueError("grid must be {'dt': float} or {'N': int}")
    keys = set(grid.keys())
    policy_keys = {key for key in ("dt", "N") if key in keys}
    if len(policy_keys) != 1 or keys != policy_keys:
        raise ValueError("grid must specify exactly one of 'dt' or 'N'")
    if "dt" in policy_keys:
        raw_dt = grid["dt"]
        if isinstance(raw_dt, bool) or not isinstance(raw_dt, numbers.Real):
            raise ValueError("grid['dt'] must be positive and finite")
        value = float(raw_dt)
        if value <= 0.0 or not math.isfinite(value):
            raise ValueError("grid['dt'] must be positive and finite")
        return {"dt": float(value)}
    raw_n = grid["N"]
    if isinstance(raw_n, bool) or not isinstance(raw_n, numbers.Integral):
        raise ValueError("grid['N'] must be an integer >= 2")
    n = int(raw_n)
    if n < 2:
        raise ValueError("grid['N'] must be >= 2")
    return {"N": int(n)}


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

    grid_policy = normalize_time_grid_policy(grid)

    if "dt" in grid_policy:
        h = float(grid_policy["dt"])
        n_int = max(1, int(math.ceil((t1 - t0) / h)))
        h_adj = (t1 - t0) / n_int
        t = t0 + h_adj * np.arange(n_int + 1, dtype=float)
        t[-1] = t1
        return t

    if "N" in grid_policy:
        n = int(grid_policy["N"])
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
