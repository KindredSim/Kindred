"""Dimensions, exact unit scales and bounded unit-string admission.

Numerical values are never wrapped. A unit describes the scale of an ordinary
number relative to J, mol, m, s and K; ``None`` means unspecified units.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from decimal import Decimal, localcontext, DecimalException
from functools import lru_cache
from html import escape
import math
import re

from ._bounded_rational import (
    EXACT_WORK_BITS, ExactWorkLimit, checked_fraction, literal_fraction, rational_binary,
)

@dataclass(frozen=True)
class UnitScale:
    """Positive scale as prime factors with rational exponents."""

    powers: tuple[tuple[int, Fraction], ...] = ()

    def __mul__(self, other: UnitScale) -> UnitScale:
        powers = dict(self.powers)
        for prime, exponent in other.powers:
            powers[prime] = powers.get(prime, Fraction(0)) + exponent
        return UnitScale(tuple(sorted((p, e) for p, e in powers.items() if e)))

    def __truediv__(self, other: UnitScale) -> UnitScale:
        return self * other.power(Fraction(-1))

    def power(self, exponent: Fraction) -> UnitScale:
        return UnitScale(tuple((p, e * exponent) for p, e in self.powers if e * exponent))

    def rational(self) -> Fraction | None:
        self.check_exact_work()
        if any(exponent.denominator != 1 for _prime, exponent in self.powers):
            return None
        value = Fraction(1)
        for prime, exponent in self.powers:
            value *= Fraction(prime) ** exponent.numerator
        return checked_fraction(value)

    def check_exact_work(self) -> None:
        """Bound expansion of prime powers before rational or CAS conversion."""
        positive = negative = 1
        for prime, exponent in self.powers:
            checked_fraction(exponent)
            magnitude = (abs(exponent.numerator) + exponent.denominator - 1) // exponent.denominator
            if exponent > 0:
                positive += magnitude * prime.bit_length()
            else:
                negative += magnitude * prime.bit_length()
        if max(positive, negative) > EXACT_WORK_BITS:
            raise ExactWorkLimit("Exact unit-scale expansion exceeds its work limit.")

    def number(self) -> float:
        return self.apply(1.0)

    def apply(self, value: float) -> float:
        """Scale a number before rounding the complete result to a float."""
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("Unit conversion requires a finite value.")
        if value == 0.0:
            return value
        try:
            # Numerical conversion can use logarithms when exact expansion is
            # unavailable. Its value domain is independent of preparation work.
            try:
                rational = self.rational()
            except ExactWorkLimit:
                rational = None
            if rational is not None:
                converted = float(Fraction.from_float(value) * rational)
            else:
                with localcontext() as context:
                    context.prec = 50
                    logarithm = sum((Decimal(p).ln() * Decimal(e.numerator) / Decimal(e.denominator)
                                     for p, e in self.powers), Decimal(abs(value)).ln())
                    converted = math.copysign(float(logarithm.exp()), value)
        except (OverflowError, DecimalException) as exc:
            raise ValueError("Unit conversion exceeds the numerical range.") from exc
        if not math.isfinite(converted) or converted == 0.0:
            raise ValueError("Unit conversion exceeds the numerical range.")
        return converted

    def symbolic(self):
        from kindred.core.symbolic.backend import require_sympy

        self.check_exact_work()
        sp = require_sympy()
        return sp.Mul(*(sp.Integer(p) ** sp.Rational(e.numerator, e.denominator) for p, e in self.powers))


@lru_cache(maxsize=128)
def rational_scale(value: Fraction) -> UnitScale:
    powers: dict[int, Fraction] = {}
    for integer, sign in ((value.numerator, 1), (value.denominator, -1)):
        divisor = 2
        while divisor * divisor <= integer:
            while integer % divisor == 0:
                powers[divisor] = powers.get(divisor, Fraction(0)) + sign
                integer //= divisor
            divisor += 1
        if integer > 1:
            powers[integer] = powers.get(integer, Fraction(0)) + sign
    return UnitScale(tuple(sorted((p, e) for p, e in powers.items() if e)))


ONE_SCALE = UnitScale()
Dimensions = tuple[Fraction, Fraction, Fraction, Fraction, Fraction]
_ZERO: Dimensions = (Fraction(0),) * 5


@dataclass(frozen=True)
class Unit:
    dimensions: Dimensions = _ZERO
    scale: UnitScale = ONE_SCALE
    symbol: str = field(default="1", compare=False)

    def compatible(self, other: Unit) -> bool:
        return self.dimensions == other.dimensions

    def multiply(self, other: Unit) -> Unit:
        return Unit(tuple(a + b for a, b in zip(self.dimensions, other.dimensions)),
                    self.scale * other.scale, _product_label(self.symbol, other.symbol))

    def divide(self, other: Unit) -> Unit:
        return Unit(tuple(a - b for a, b in zip(self.dimensions, other.dimensions)),
                    self.scale / other.scale, f"{self.symbol}/({other.symbol})")

    def power(self, exponent: Fraction) -> Unit:
        return Unit(tuple(a * exponent for a in self.dimensions), self.scale.power(exponent),
                    _power_label(self.symbol, exponent))


def _product_label(left: str, right: str) -> str:
    return right if left == "1" else left if right == "1" else f"{left}*{right}"


def _power_label(symbol: str, exponent: Fraction) -> str:
    if not exponent:
        return "1"
    if exponent == 1:
        return symbol
    power = str(exponent.numerator) if exponent.denominator == 1 else f"({exponent})"
    base = symbol if re.fullmatch(r"[A-Za-zµ]+", symbol) else f"({symbol})"
    return f"{base}^{power}"


DIMENSIONLESS = Unit()
JOULE = Unit((Fraction(1), Fraction(0), Fraction(0), Fraction(0), Fraction(0)), symbol="J")
MOLE = Unit((Fraction(0), Fraction(1), Fraction(0), Fraction(0), Fraction(0)), symbol="mol")
METRE = Unit((Fraction(0), Fraction(0), Fraction(1), Fraction(0), Fraction(0)), symbol="m")
SECOND = Unit((Fraction(0), Fraction(0), Fraction(0), Fraction(1), Fraction(0)), symbol="s")
KELVIN = Unit((Fraction(0), Fraction(0), Fraction(0), Fraction(0), Fraction(1)), symbol="K")
CUBIC_METRE = METRE.power(Fraction(3))
MOLAR = Unit(MOLE.divide(CUBIC_METRE).dimensions, rational_scale(Fraction(1000)), "M")
ENERGY_PER_MOLE = Unit(JOULE.divide(MOLE).dimensions, symbol="J/mol")


@lru_cache(maxsize=512)
def scientific_unit(dimensions: Dimensions) -> Unit:
    energy, amount, length, time, temperature = dimensions
    if energy == 0 and length == -3 * amount:
        factors = ((MOLAR, amount), (SECOND, time), (KELVIN, temperature))
    else:
        factors = tuple(zip((JOULE, MOLE, METRE, SECOND, KELVIN), dimensions))
    result = DIMENSIONLESS
    for base, exponent in factors:
        if exponent:
            result = result.multiply(base.power(exponent))
    return result


def _named_units() -> dict[str, Unit]:
    units = {u.symbol: u for u in (DIMENSIONLESS, JOULE, MOLE, METRE, SECOND, KELVIN, MOLAR)}
    units["L"] = Unit(CUBIC_METRE.dimensions, rational_scale(Fraction(1, 1000)), "L")
    units["kcal"] = Unit(JOULE.dimensions, rational_scale(Fraction(4184)), "kcal")
    for base in (MOLAR, MOLE, METRE, SECOND, JOULE, units["L"]):
        for prefix, exponent in (("f", -15), ("p", -12), ("n", -9), ("µ", -6), ("m", -3), ("c", -2), ("k", 3)):
            # Centi is useful for length, not a second concentration/time menu.
            if prefix == "c" and base is not METRE:
                continue
            symbol = prefix + base.symbol
            units[symbol] = Unit(base.dimensions, base.scale * rational_scale(Fraction(10) ** exponent), symbol)
    for symbol, factor in (("min", 60), ("h", 3600), ("d", 86400), ("yr", 31557600)):
        units[symbol] = Unit(SECOND.dimensions, rational_scale(Fraction(factor)), symbol)
    return units


NAMED_UNITS = _named_units()
_TOKEN = re.compile(r"\s*(\*\*|[A-Za-zµμ]+|(?:\d+(?:\.\d*)?|\.\d+)|[*/^()+-])")


@lru_cache(maxsize=512)
def parse_unit(source: str) -> Unit:
    """Admit a bounded unit expression, independently of scientific formulas."""
    source = source.strip().replace("μ", "µ")
    tokens = []
    position = 0
    while position < len(source):
        match = _TOKEN.match(source, position)
        if match is None:
            raise ValueError(f"Malformed unit declaration: {source!r}.")
        tokens.append(match.group(1))
        position = match.end()
    index = 0

    def take(expected=None):
        nonlocal index
        token = tokens[index] if index < len(tokens) else ""
        if not token or expected is not None and token != expected:
            raise ValueError(f"Malformed unit declaration: {source!r}.")
        index += 1
        return token

    def exponent():
        parenthesized = index < len(tokens) and tokens[index] == "("
        if parenthesized:
            take("(")
        sign = 1
        if index < len(tokens) and tokens[index] in {"+", "-"}:
            sign = -1 if take() == "-" else 1
        try:
            value = literal_fraction(take()) * sign
            if parenthesized and index < len(tokens) and tokens[index] == "/":
                take("/")
                value = rational_binary("/", value, literal_fraction(take()))
        except (ValueError, ZeroDivisionError) as exc:
            raise ValueError(f"Invalid unit exponent in {source!r}.") from exc
        if parenthesized:
            take(")")
        return value

    def atom():
        name = take()
        if name == "(":
            result = product()
            take(")")
        else:
            name = "µ" + name[1:] if name.startswith("u") else name
            try:
                result = NAMED_UNITS[name]
            except KeyError as exc:
                raise ValueError(f"Unsupported unit {name!r} in {source!r}.") from exc
        if index < len(tokens) and tokens[index] in {"^", "**"}:
            take()
            power = exponent()
            # Declaration parsing owns expansion of authored powers; derived
            # dimensions retain their separate inference/operation authority.
            for value in (*result.dimensions, *(e for _p, e in result.scale.powers)):
                rational_binary("*", value, power)
            result = result.power(power)
        return result

    def product():
        result = atom()
        while index < len(tokens) and tokens[index] in {"*", "/"}:
            op = take()
            rhs = atom()
            for left, right in zip(result.dimensions, rhs.dimensions):
                rational_binary("+" if op == "*" else "-", left, right)
            left_scales, right_scales = dict(result.scale.powers), dict(rhs.scale.powers)
            for prime in left_scales.keys() | right_scales.keys():
                rational_binary("+" if op == "*" else "-", left_scales.get(prime, Fraction(0)),
                                right_scales.get(prime, Fraction(0)))
            result = result.multiply(rhs) if op == "*" else result.divide(rhs)
        return result

    try:
        result = product()
    except RecursionError:
        raise ValueError(f"Unit declaration is nested too deeply: {source!r}.") from None
    if index != len(tokens):
        raise ValueError(f"Malformed unit declaration: {source!r}.")
    return Unit(result.dimensions, result.scale, source.replace("μ", "µ"))


def conversion_scale(source: Unit, target: Unit) -> UnitScale:
    if not source.compatible(target):
        raise ValueError(f"Incompatible units: {source.symbol} and {target.symbol}.")
    return source.scale / target.scale


def convert_value(value: float, source: Unit, target: Unit) -> float:
    scale = conversion_scale(source, target)
    try:
        return scale.apply(value)
    except ValueError as exc:
        raise ValueError(f"Conversion from {source.symbol} to {target.symbol}: {exc}") from exc


def parse_declared_unit(source: str) -> Unit:
    """Admit an authored unit within the supported scientific conversion range."""
    unit = parse_unit(source)
    diagnostic = (
        f"Unsupported unit declaration {unit.symbol!r}: its conversion scale "
        "relative to scientific units must be between 1e-30 and 1e30, inclusive."
    )
    try:
        factor = conversion_scale(unit, scientific_unit(unit.dimensions)).number()
    except ValueError as exc:
        raise ValueError(diagnostic) from exc
    if not 1e-30 <= factor <= 1e30:
        raise ValueError(diagnostic)
    return unit


def presentation_unit(unit: Unit | None, declared: Unit | None = None) -> Unit | None:
    if unit is None:
        return None
    if declared is not None and unit.compatible(declared):
        return declared
    return NAMED_UNITS["L"] if unit.dimensions == CUBIC_METRE.dimensions else unit


def analysis_display_unit(
    unit: Unit, *, concentration: Unit, time: Unit, volume: Unit,
) -> Unit:
    """Express concentration/time quantities and vessel volume in display units.

    Other dimensions keep their retained unit. These choices affect readouts,
    never the scientific or editable coordinates of a parameter.
    """
    energy, amount, length, time_power, temperature = unit.dimensions
    if energy == 0 and length == -3 * amount:
        factors = ((concentration, amount), (time, time_power), (KELVIN, temperature))
    elif energy == amount == temperature == 0 and length == 3:
        factors = ((volume, Fraction(1)), (time, time_power))
    else:
        return unit
    result = DIMENSIONLESS
    for base, exponent in factors:
        if exponent:
            result = result.multiply(base.power(exponent))
    return result


_UNIT_POWER = re.compile(
    r"(?:\^|\*\*)\s*(\(\s*[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:\s*/\s*[+-]?(?:\d+(?:\.\d*)?|\.\d+))?\s*\)"
    r"|[+-]?(?:\d+(?:\.\d*)?|\.\d+))"
)
_UNIT_WORD = re.compile(r"[A-Za-zµ]+")


def _display_unit_spelling(text: str) -> str:
    text = text.replace("μ", "µ")
    return _UNIT_WORD.sub(
        lambda match: "µ" + match.group(0)[1:]
        if match.group(0).startswith("u")
        and "µ" + match.group(0)[1:] in NAMED_UNITS
        else match.group(0),
        text,
    )


def format_unit_for_display(unit: Unit | str | None) -> str | None:
    """Format an admitted unit spelling for human-facing text.

    The result is a Qt-independent HTML fragment for rich-text consumers.
    This is a presentation-only lexical projection.  It does not admit or
    interpret units, alter their identity, or participate in conversion.
    """
    if unit is None:
        return None
    text = unit.symbol if isinstance(unit, Unit) else unit
    if text == "1":
        return text
    text = _display_unit_spelling(text)
    escaped = escape(text)
    escaped = _UNIT_POWER.sub(
        lambda match: f"<sup>{match.group(1)}</sup>", escaped,
    )
    return escaped.replace("*", "·")


def reconcile_unit_choices(
    candidates: tuple[Unit, ...], selected: Unit | None = None,
) -> tuple[Unit, ...]:
    """Collapse spelling aliases in order, retaining the selected identity."""
    unique = {}
    if selected is not None:
        candidates = (*candidates, selected)
    for item in candidates:
        key = (item.dimensions, item.scale, _display_unit_spelling(item.symbol))
        if key not in unique or item is selected:
            unique[key] = item
    return tuple(unique.values())


def compatible_units(unit: Unit, preferred: tuple[Unit, ...] = ()) -> tuple[Unit, ...]:
    return reconcile_unit_choices(tuple(
        item for item in (unit, *preferred, *NAMED_UNITS.values())
        if unit.compatible(item)
    ))


CONSTANT_UNITS = {
    "k_B": JOULE.divide(KELVIN), "h": JOULE.multiply(SECOND),
    "h_bar": JOULE.multiply(SECOND), "N_A": MOLE.power(Fraction(-1)),
    "R": ENERGY_PER_MOLE.divide(KELVIN), "T": KELVIN, "t": SECOND,
}
