"""Exact directional stoichiometric-order helpers.

Stoichiometric coefficients have already been admitted at the mechanism
boundary by the time they reach this module.  Keeping the sum here exact
prevents each scientific consumer from independently reinterpreting a
fractional order through binary floating-point summation.
"""

from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction

__all__ = ["directional_order_fraction"]


def directional_order_fraction(
    stoichiometry: Mapping[str, float],
) -> Fraction:
    """Return the exact order of an admitted stoichiometric map.

    Values are converted through their textual representation so decimal
    authored values such as ``0.1`` and ``0.2`` retain their authored decimal
    meaning instead of being summed as binary floats.
    """

    return sum(
        (Fraction(str(value)) for value in stoichiometry.values()),
        start=Fraction(0),
    )
