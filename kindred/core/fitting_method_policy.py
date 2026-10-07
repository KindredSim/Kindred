"""Stable optimizer identifiers shared by fitting persistence and execution."""

from __future__ import annotations

__all__ = [
    "LEAST_SQUARES_TOLERANCE",
    "SUPPORTED_FIT_METHODS",
    "UNCERTAINTY_FIT_METHODS",
    "validate_fit_method",
]

SUPPORTED_FIT_METHODS = ("trf", "dogbox", "differential_evolution")
# Profile and Bootstrap refit from the completed Fit with a local optimizer.
UNCERTAINTY_FIT_METHODS = ("trf", "dogbox")
# Default objective and step tolerance of a least-squares fit, also used by the
# Profile and Bootstrap refits after Differential Evolution.
LEAST_SQUARES_TOLERANCE = 1e-10


def validate_fit_method(value: object) -> str:
    """Return one stable optimizer identifier or reject the value."""

    if not isinstance(value, str):
        raise TypeError("Fit optimization method must be a string identifier.")
    normalized = value.strip()
    if normalized not in SUPPORTED_FIT_METHODS:
        raise ValueError(
            "Fit optimization method must be one of "
            f"{SUPPORTED_FIT_METHODS!r}."
        )
    return normalized
