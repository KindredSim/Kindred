"""
Pure helpers for Species mode slider math.

This module intentionally contains no Qt imports so it can be unit-tested in
headless environments.
"""

from __future__ import annotations

import sys

__all__ = [
    "compute_slider_max_option_c",
]


def compute_slider_max_option_c(*, v: object) -> float:
    """
    Return max(1.0, 5*v) for the current species value after sanitization.

    The growth term saturates at the largest finite float so a valid finite
    species value never turns the GUI range into infinity.
    """
    # Admitted concentrations, staged drafts and slider positions are
    # nonnegative finite floats.
    v_f = v
    return float(
        max(
            1.0,
            _scale_finite_gui_range(v_f, 5.0),
        )
    )


def _scale_finite_gui_range(candidate: float, scale: float) -> float:
    """Scale a finite nonnegative GUI range candidate without overflowing."""

    # Candidates are sanitized finite values; scales are positive literals.
    candidate_f = candidate
    scale_f = scale
    largest_finite = sys.float_info.max
    if candidate_f >= largest_finite / scale_f:
        return largest_finite
    return candidate_f * scale_f
