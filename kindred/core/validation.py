"""
Small core validation helpers shared across multiple core modules.
"""

from __future__ import annotations

import math
from decimal import Decimal, InvalidOperation


def parse_finite_numeric_text(text: str, *, field: str = "Value") -> float:
    """Admit numeric text without silently replacing a nonzero input by zero."""
    try:
        value = float(text)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"{field} must be a number.") from exc
    if not math.isfinite(value):
        raise ValueError(f"{field} must be finite.")
    if value == 0.0:
        try:
            nonzero = not Decimal(text).is_zero()
        except InvalidOperation as exc:
            raise ValueError(f"{field} cannot be represented as a number.") from exc
        if nonzero:
            raise ValueError(f"{field} is nonzero but too small to represent.")
    return value


def six_digit_value(value: float) -> float:
    """The value rounded to six significant digits."""
    return float(format(float(value), ".6g"))


def six_digit_text(value: float) -> str:
    """Display text for one value at six significant digits.

    Plain notation for 0.01 <= |v| < 10000 and scientific notation otherwise,
    chosen from the rounded value; exact zero stays 0 and a nonzero value never
    becomes 0.
    """
    rounded = six_digit_value(value)
    if rounded == 0.0 or 0.01 <= abs(rounded) < 10000.0:
        return format(rounded, ".6g")
    mantissa, exponent = format(rounded, ".5e").split("e")
    return f"{mantissa.rstrip('0').rstrip('.')}e{exponent}"


def six_digit_source_text(value: float) -> str:
    """Writer spelling: the six-digit text when it parses back exactly, else repr."""
    text = six_digit_text(value)
    return text if float(text) == float(value) else repr(float(value))


def try_parse_finite_float(value: object) -> tuple[float, bool]:
    """
    Best-effort float parsing with an explicit validity flag.

    Returns (parsed_value, ok). When ok is False, parsed_value is 0.0 and callers
    must decide how to handle the invalid input at their boundary layer.
    """
    try:
        v = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return 0.0, False
    if not math.isfinite(v):
        return 0.0, False
    return float(v), True


def try_parse_nonneg_finite_float(value: object) -> tuple[float, bool]:
    """
    Best-effort float parsing for non-negative finite values.

    Returns (parsed_value, ok). When ok is False, parsed_value is 0.0.
    """
    v, ok = try_parse_finite_float(value)
    if not ok or v < 0.0:
        return 0.0, False
    return float(v), True


def try_parse_int(value: object) -> tuple[int, bool]:
    """
    Best-effort int parsing with an explicit validity flag.

    Returns (parsed_value, ok). When ok is False, parsed_value is 0.
    """
    if isinstance(value, bool):
        return 0, False
    try:
        v = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return 0, False
    return int(v), True
