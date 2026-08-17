"""Stable optimizer identifiers shared by fitting persistence and execution."""

from __future__ import annotations

__all__ = [
    "SUPPORTED_FIT_METHODS",
    "validate_fit_method",
]

SUPPORTED_FIT_METHODS = ("trf", "dogbox", "differential_evolution")


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
