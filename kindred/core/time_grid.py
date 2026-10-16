"""
Canonical time-grid policies for Kindred.

This module centralizes:
- N-only generated time-grid construction
- Uniform-grid detection using a median-spacing relative-deviation check
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

__all__ = ["build_time_grid", "is_uniform_time_grid"]

_EPS_T = 1e-15  # Guard the relative-uniformity denominator on tiny time steps.


def build_time_grid(t0: float, t1: float, grid: Mapping[str, int]) -> np.ndarray:
    """
    Build a deterministic, strictly increasing evaluation grid on [t0, t1].

    The span and the generated-grid policy {"N": int} arrive admitted with
    their request.  A request is rejected when the span is too small to
    represent all N points strictly increasingly in binary64.
    """
    n = grid["N"]
    t = np.linspace(t0, t1, n, dtype=float)
    if (
        float(t[0]) != t0
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
