"""
Pure helpers for Species mode slider math.

This module intentionally contains no Qt imports so it can be unit-tested in
headless environments.
"""

from __future__ import annotations

import math
import sys
from kindred.core.validation import try_parse_nonneg_finite_float

__all__ = [
    "sanitize_nonneg_finite",
    "compute_slider_max_option_c",
]


def sanitize_nonneg_finite(value: object) -> float:
    """
    Convert `value` to a non-negative finite float.

    Any non-finite / non-numeric / negative value is treated as 0.0.
    For callers that need to distinguish invalid inputs, use
    `try_parse_nonneg_finite_float` directly.
    """
    v, ok = try_parse_nonneg_finite_float(value)
    return float(v) if ok else 0.0


def compute_slider_max_option_c(*, v: object, row_max: object) -> float:
    """
    Option C slider max:
    - `row_max` is the row-wide max (after sanitization)
    - `v` is the current species value (after sanitization)
    - slider_max = max(1.0, 2*row_max, 5*v)

    Both growth terms saturate at the largest finite float so a valid finite
    species value never turns the GUI range into infinity.
    """
    v_f = sanitize_nonneg_finite(v)
    row_max_f = sanitize_nonneg_finite(row_max)
    return float(
        max(
            1.0,
            _scale_finite_gui_range(row_max_f, 2.0),
            _scale_finite_gui_range(v_f, 5.0),
        )
    )


def _scale_finite_gui_range(candidate: float, scale: float) -> float:
    """Scale a finite nonnegative GUI range candidate without overflowing."""

    candidate_f = float(candidate)
    scale_f = float(scale)
    if (
        not math.isfinite(candidate_f)
        or candidate_f < 0.0
        or not math.isfinite(scale_f)
        or scale_f <= 0.0
    ):
        raise ValueError("GUI range candidate and scale must be finite and valid.")
    largest_finite = sys.float_info.max
    if candidate_f >= largest_finite / scale_f:
        return largest_finite
    return candidate_f * scale_f
