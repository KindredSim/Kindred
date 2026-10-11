"""
Species model and strict initial-concentration coercion.

Current contract
----------------
- Species has one field: `name: str`.
- Mechanism owns its ordered species mapping directly.

Design notes
------------
- Initial concentrations are finite non-negative floats. The core model rejects
  negative values before simulation, fitting, or export can consume them.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

from .validation import parse_finite_numeric_text

__all__ = [
    "CONCENTRATION_MIN_M",
    "CONCENTRATION_MAX_M",
    "C0_MIN_M",
    "C0_MAX_M",
    "TEMPERATURE_MIN_K",
    "TEMPERATURE_MAX_K",
    "TIME_MIN_S",
    "TIME_INPUT_MIN_S",
    "TIME_MAX_S",
    "IMPORTED_TIME_MIN_S",
    "IMPORTED_TIME_MAX_S",
    "MAX_GENERATED_TIME_ROWS",
    "MAX_FIT_EVALUATIONS",
    "Species",
    "concentration_from_text",
    "validate_concentration",
    "validate_c0",
    "validate_temperature",
    "validate_time_seconds",
    "validate_input_time_seconds",
    "validate_imported_time_seconds",
    "validate_observation",
]


# Numerical admission is category-specific.  Keep these values in one small,
# dependency-free owner so parsing, project/archive admission, runtime request
# construction, and Fit setup can share the exact same boundary semantics.
CONCENTRATION_MIN_M = 1e-50
CONCENTRATION_MAX_M = 1e30
C0_MIN_M = CONCENTRATION_MIN_M
C0_MAX_M = CONCENTRATION_MAX_M
TEMPERATURE_MIN_K = 0.1
TEMPERATURE_MAX_K = 10_000.0
TIME_MIN_S = 0.0
TIME_INPUT_MIN_S = 1e-30
TIME_MAX_S = 1e10
IMPORTED_TIME_MIN_S = -1e10
IMPORTED_TIME_MAX_S = 1e10
MAX_GENERATED_TIME_ROWS = 100_000
MAX_FIT_EVALUATIONS = 100_000


def _numeric(value: object, *, field: str) -> float:
    if type(value) is bool or not isinstance(value, Real):
        raise TypeError(f"{field} must be numeric.")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{field} must be finite.")
    return parsed


def validate_concentration(
    value: object,
    *,
    field: str = "concentration",
    allow_zero: bool = True,
) -> float:
    """Admit one authored concentration in molar units.

    Exact zero is the explicit off/absent state.  Every nonzero authored
    concentration must stay within the supported finite interval; values below
    the floor are never silently rounded to zero.
    """

    parsed = _numeric(value, field=field)
    if parsed < 0.0 or (not allow_zero and parsed == 0.0):
        qualifier = "nonnegative" if allow_zero else "strictly positive"
        raise ValueError(f"{field} must be {qualifier}.")
    if parsed == 0.0:
        return 0.0
    if parsed < CONCENTRATION_MIN_M:
        raise ValueError(
            f"{field} is below the supported concentration floor "
            f"{CONCENTRATION_MIN_M:g} M. Enter exact 0 if zero is intended."
        )
    if parsed > CONCENTRATION_MAX_M:
        raise ValueError(
            f"{field} must be at most {CONCENTRATION_MAX_M:g} M."
        )
    return parsed


def concentration_from_text(text: str, *, field: str = "concentration") -> float:
    """Admit authored concentration text; nonzero text never becomes zero."""

    return validate_concentration(
        parse_finite_numeric_text(text, field=field),
        field=field,
    )


def validate_c0(value: object, *, field: str = "C0") -> float:
    """Admit the strictly-positive standard concentration in molar units."""

    parsed = _numeric(value, field=field)
    if not C0_MIN_M <= parsed <= C0_MAX_M:
        raise ValueError(
            f"{field} must be within {C0_MIN_M:g}–{C0_MAX_M:g} M."
        )
    return parsed


def validate_temperature(value: object, *, field: str = "temperature") -> float:
    parsed = _numeric(value, field=field)
    if not TEMPERATURE_MIN_K <= parsed <= TEMPERATURE_MAX_K:
        raise ValueError(
            f"{field} must be within {TEMPERATURE_MIN_K:g}–{TEMPERATURE_MAX_K:g} K."
        )
    return parsed


def validate_time_seconds(
    value: object,
    *,
    field: str = "time",
    allow_zero: bool = True,
) -> float:
    """Check execution times, including generated samples below the input floor."""
    parsed = _numeric(value, field=field)
    if parsed < TIME_MIN_S or (not allow_zero and parsed == TIME_MIN_S):
        qualifier = "nonnegative" if allow_zero else "strictly positive"
        raise ValueError(f"{field} must be {qualifier}.")
    if parsed > TIME_MAX_S:
        raise ValueError(f"{field} must be at most {TIME_MAX_S:g} seconds.")
    return parsed


def validate_input_time_seconds(
    value: object,
    *,
    field: str = "time",
    allow_zero: bool = True,
) -> float:
    """Admit model timing inputs without restricting generated execution times."""
    parsed = validate_time_seconds(value, field=field, allow_zero=allow_zero)
    if 0.0 < parsed < TIME_INPUT_MIN_S:
        raise ValueError(
            f"{field} must be at least {TIME_INPUT_MIN_S:g} seconds when nonzero."
        )
    return parsed


def validate_imported_time_seconds(
    value: object,
    *,
    field: str = "imported time",
) -> float:
    parsed = _numeric(value, field=field)
    if not IMPORTED_TIME_MIN_S <= parsed <= IMPORTED_TIME_MAX_S:
        raise ValueError(
            f"{field} must be within {IMPORTED_TIME_MIN_S:g}–"
            f"{IMPORTED_TIME_MAX_S:g} seconds after unit conversion."
        )
    return parsed


def validate_observation(value: object, *, field: str = "observation") -> float:
    """Admit an arbitrary finite observation, including signed values."""

    return _numeric(value, field=field)


@dataclass(frozen=True)
class Species:
    """Immutable species record; DSL parsing admits its name."""
    name: str


# --- helpers -----------------------------------------------------------------

