"""
Species model and strict initial-concentration coercion.

Current contract
----------------
- Species has fields: `name: str`, `initial_conc: float` (standard state basis).
- Mechanism owns its ordered species mapping directly.

Design notes
------------
- Initial concentrations are finite non-negative floats. The core model rejects
  negative values before simulation, fitting, or export can consume them.
"""

from __future__ import annotations

from dataclasses import dataclass

from .validation import validate_name

__all__ = [
    "Species",
    "coerce_float",
]


@dataclass(frozen=True)
class Species:
    """Immutable species record."""
    name: str
    initial_conc: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", validate_name(self.name))
        object.__setattr__(self, "initial_conc", coerce_float(self.initial_conc))


# --- helpers -----------------------------------------------------------------

def coerce_float(value) -> float:
    """
    Coerce a number-like to float with strict finite check.

    Raises
    ------
    TypeError, ValueError
        If the value is not convertible to a finite float.
    """
    try:
        f = float(value)
    except (ValueError, TypeError, OverflowError) as e:
        raise TypeError(f"initial concentration must be a real number, got {value!r}") from e
    if not (f == f):  # NaN check without importing math
        raise ValueError("initial concentration cannot be NaN")
    if f in (float("inf"), float("-inf")):
        raise ValueError("initial concentration must be finite")
    if f < 0.0:
        raise ValueError("initial concentration must be non-negative")
    return f
