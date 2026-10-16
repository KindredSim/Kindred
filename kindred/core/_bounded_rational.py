"""Checked exact arithmetic for preparation, never numerical evaluation."""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
from fractions import Fraction

EXACT_WORK_BITS = 4096


class ExactWorkLimit(ValueError):
    """An exact preparation result would require excessive expansion."""


def _size(*bits: int) -> None:
    if max(bits, default=0) > EXACT_WORK_BITS:
        raise ExactWorkLimit("Exact unit preparation exceeds its work limit.")


def checked_fraction(value: Fraction) -> Fraction:
    _size(value.numerator.bit_length(), value.denominator.bit_length())
    return value


def literal_fraction(text: str) -> Fraction:
    """Inspect decimal size before constructing its integer ratio."""
    try:
        decimal = Decimal(text)
    except InvalidOperation as exc:
        raise ValueError("Invalid exact numeric literal.") from exc
    if not decimal.is_finite():
        raise ValueError("Exact numeric literal must be finite.")
    if decimal.is_zero():
        return Fraction(0)
    _sign, digits, exponent = decimal.as_tuple()
    # Four bits per decimal digit is a conservative allocation estimate.
    _size(4 * (len(digits) + max(exponent, 0)), 4 * max(-exponent, 0))
    return checked_fraction(Fraction(decimal))


def rational_binary(operation: str, left: Fraction, right: Fraction) -> Fraction:
    checked_fraction(left)
    checked_fraction(right)
    ln, ld = left.numerator.bit_length(), left.denominator.bit_length()
    rn, rd = right.numerator.bit_length(), right.denominator.bit_length()
    if operation in {"+", "-"}:
        _size(max(ln + rd, rn + ld) + 1, ld + rd)
        value = left + right if operation == "+" else left - right
    elif operation == "*":
        if not left or not right:
            return Fraction(0)
        _size(ln + rn, ld + rd)
        value = left * right
    elif operation == "/":
        if not right:
            raise ZeroDivisionError("division by zero")
        _size(ln + rd, ld + rn)
        value = left / right
    elif operation in {"^", "**"}:
        if right.denominator != 1:
            raise ValueError("An exact constant power requires an integer exponent.")
        exponent = right.numerator
        if not exponent:
            return Fraction(1)
        if left in (0, 1, -1):
            return left ** exponent
        _size(ln * abs(exponent), ld * abs(exponent))
        value = left ** exponent
    else:
        raise ValueError(f"Unsupported exact arithmetic: {operation}.")
    return checked_fraction(value)
