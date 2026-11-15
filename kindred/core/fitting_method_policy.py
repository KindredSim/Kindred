"""Stable optimizer identifiers shared by fitting persistence and execution."""

from __future__ import annotations

SUPPORTED_FIT_METHODS = ("trf", "dogbox", "differential_evolution")
# Profile likelihood refits from the completed Fit with a local optimizer.
UNCERTAINTY_FIT_METHODS = ("trf", "dogbox")
# Default objective and step tolerance of a least-squares fit.
LEAST_SQUARES_TOLERANCE = 1e-10


def validate_fit_method(value: str) -> None:
    """Reject a method that is not one of the supported optimizer identifiers.

    The archive decoder, settings reader and method combos hand over text.
    """

    if value not in SUPPORTED_FIT_METHODS:
        raise ValueError(
            "Fit optimization method must be one of "
            f"{SUPPORTED_FIT_METHODS!r}."
        )
