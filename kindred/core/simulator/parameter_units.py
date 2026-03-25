"""
Unit helpers for solver parameters.

These are deliberately lightweight string helpers used by GUI tables.
"""

from __future__ import annotations


__all__ = ["rate_constant_unit"]


def rate_constant_unit(order: int, *, conc_unit: str = "M", time_unit: str = "s") -> str:
    """
    Return a mass-action rate-constant unit string for a reaction of given order.

    For mass action, rate has units conc/time (e.g. M/s). If a reaction has
    molecularity/order n, the rate law uses ∏[reactants]^n, so:

        [k] = conc^(1-n) / time
    """
    try:
        n = int(order)
    except Exception:
        n = 1

    if n == 1:
        return f"1/{time_unit}"
    if n == 0:
        return f"{conc_unit}/{time_unit}"
    power = n - 1
    if power == 1:
        return f"1/({conc_unit} {time_unit})"
    return f"1/({conc_unit}^{power} {time_unit})"

