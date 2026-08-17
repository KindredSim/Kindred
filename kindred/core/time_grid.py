"""
Canonical time-grid policies for Kindred.

This module centralizes:
- N-only generated time-grid construction
- Uniform-grid detection using a median-spacing relative-deviation check
"""

from __future__ import annotations

import numbers
from typing import Any, Mapping

import numpy as np
from kindred.core.species import (
    validate_row_count,
    validate_time_seconds,
)

__all__ = ["build_time_grid", "is_uniform_time_grid", "normalize_time_grid_policy"]

_EPS_T = 1e-15  # Guard the relative-uniformity denominator on tiny time steps.


def normalize_time_grid_policy(grid: Mapping[str, Any]) -> dict[str, float | int]:
    """
    Return the canonical requested time-grid policy.

    A requested generated grid policy is exactly {"N": int >= 2}.

    The generated-row ceiling applies only to this N policy.  Explicit t_eval
    arrays are admitted and validated at their request/output boundaries.
    """
    if not isinstance(grid, Mapping) or not grid:
        raise ValueError("grid must be {'N': int}")
    keys = set(grid.keys())
    if keys != {"N"}:
        raise ValueError("grid must specify exactly {'N': int}")
    raw_n = grid["N"]
    if isinstance(raw_n, bool) or not isinstance(raw_n, numbers.Integral):
        raise ValueError("grid['N'] must be an integer >= 2")
    n = int(raw_n)
    if n < 2:
        raise ValueError("grid['N'] must be >= 2")
    validate_row_count(n, field="grid['N']")
    return {"N": int(n)}


def build_time_grid(t0: float, t1: float, grid: Mapping[str, float | int]) -> np.ndarray:
    """
    Build a deterministic, strictly increasing evaluation grid on [t0, t1].

    The sole supported generated-grid policy is {"N": int}, which realizes
    exactly N finite binary64 values on [t0, t1].  A request is rejected when
    the span is too small to represent all N points strictly increasingly.
    """
    t0 = validate_time_seconds(t0, field="t_span start")
    t1 = validate_time_seconds(t1, field="t_span end")
    if not t1 > t0:
        raise ValueError("t_span must have end greater than start")

    grid_policy = normalize_time_grid_policy(grid)

    n = int(grid_policy["N"])
    validate_row_count(n, field="generated time-grid rows")
    t = np.linspace(t0, t1, n, dtype=float)
    if (
        t.size != n
        or np.any(~np.isfinite(t))
        or float(t[0]) != t0
        or float(t[-1]) != t1
        or np.any(np.diff(t) <= 0.0)
    ):
        raise ValueError(
            "generated N time grid cannot realize exactly N finite, strictly "
            "increasing binary64 values including both endpoints"
        )
    return t


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
