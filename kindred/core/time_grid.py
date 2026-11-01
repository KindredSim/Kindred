"""Canonical N-only generated time-grid construction for Kindred."""

from __future__ import annotations

from typing import Mapping

import numpy as np

__all__ = ["build_time_grid"]


def build_time_grid(t0: float, t1: float, grid: Mapping[str, int]) -> np.ndarray:
    """
    Build a deterministic, strictly increasing evaluation grid on [t0, t1].

    The span and the generated-grid policy {"N": int} arrive admitted with
    their request.  A request is rejected when the span is too small to
    represent all N points strictly increasingly in binary64.
    """
    n = grid["N"]
    t = np.linspace(t0, t1, n, dtype=float)
    # At least 10 points; linspace keeps both endpoints exactly.
    if np.any(np.diff(t) <= 0.0):
        raise ValueError(
            "generated N time grid cannot realize exactly N finite, strictly "
            "increasing binary64 values including both endpoints"
        )
    return t
