"""
Unit helpers for solver parameters.

These are deliberately lightweight string helpers used by GUI tables.
"""

from __future__ import annotations

import math


__all__ = ["rate_constant_unit"]


def _validated_order(order: object) -> float:
    if type(order) is bool:
        raise TypeError("reaction order must be numeric")
    try:
        value = float(order)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("reaction order must be numeric") from exc
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("reaction order must be a finite non-negative real")
    return value


def _format_exponent(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    return format(value, ".15g")


def rate_constant_unit(
    order: float,
    *,
    conc_unit: str = "M",
    time_unit: str = "s",
) -> str:
    """
    Return a mass-action rate-constant unit string for a reaction of given order.

    For mass action, rate has units conc/time (e.g. M/s). If a reaction has
    molecularity/order n, the rate law uses ∏[reactants]^n, so:

        [k] = conc^(1-n) / time
    """
    n = _validated_order(order)

    if n == 1:
        return f"1/{time_unit}"
    if n == 0:
        return f"{conc_unit}/{time_unit}"
    if n > 1:
        power = n - 1.0
        if power == 1:
            return f"1/({conc_unit} {time_unit})"
        return f"1/({conc_unit}^{_format_exponent(power)} {time_unit})"
    return f"{conc_unit}^{_format_exponent(1.0 - n)}/{time_unit}"
