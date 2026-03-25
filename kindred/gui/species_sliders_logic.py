"""
Pure helpers for Species mode slider math.

This module intentionally contains no Qt imports so it can be unit-tested in
headless environments.
"""

from __future__ import annotations

from typing import Iterable

from kindred.core.validation import try_parse_nonneg_finite_float

__all__ = [
    "try_nonneg_finite",
    "sanitize_nonneg_finite",
    "compute_row_max",
    "compute_slider_max_option_c",
]


def try_nonneg_finite(value: object) -> tuple[float, bool]:
    """
    Parse a non-negative finite float with an explicit validity flag.

    Returns (value, ok). When ok is False, value is 0.0 and callers can decide
    whether to log, surface UI feedback, or block downstream actions.
    """
    return try_parse_nonneg_finite_float(value)


def sanitize_nonneg_finite(value: object) -> float:
    """
    Convert `value` to a non-negative finite float.

    Any non-finite / non-numeric / negative value is treated as 0.0.
    For callers that need to distinguish invalid inputs, use `try_nonneg_finite`.
    """
    v, ok = try_nonneg_finite(value)
    return float(v) if ok else 0.0


def compute_row_max(values: Iterable[object]) -> float:
    """
    Return the maximum sanitized value across the row.

    Values are sanitized via `sanitize_nonneg_finite`.
    """
    row_max = 0.0
    for raw in values:
        v = sanitize_nonneg_finite(raw)
        if v > row_max:
            row_max = v
    return float(row_max)


def compute_slider_max_option_c(*, v: object, row_max: object) -> float:
    """
    Option C slider max:
    - `row_max` is the row-wide max (after sanitization)
    - `v` is the current species value (after sanitization)
    - slider_max = max(1.0, 2*row_max, 5*v)
    """
    v_f = sanitize_nonneg_finite(v)
    row_max_f = sanitize_nonneg_finite(row_max)
    return float(max(1.0, 2.0 * row_max_f, 5.0 * v_f))
