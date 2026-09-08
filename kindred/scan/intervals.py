"""Interval arithmetic for user-selected scan ranges, without chemical rules."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math


@dataclass(frozen=True)
class Interval:
    lower: float
    upper: float
    include_lower: bool = True
    include_upper: bool = True

    def contains(self, value: float) -> bool:
        return ((value > self.lower or self.include_lower and value == self.lower)
                and (value < self.upper or self.include_upper and value == self.upper))


def normalize(intervals: tuple[Interval, ...]) -> tuple[Interval, ...]:
    ordered = sorted(intervals, key=lambda i: (i.lower, not i.include_lower, i.upper))
    result: list[Interval] = []
    for item in ordered:
        if not math.isfinite(item.lower) or not math.isfinite(item.upper):
            raise ValueError("Scan bounds must be finite numbers.")
        if item.lower > item.upper or (item.lower == item.upper and not (
            item.include_lower and item.include_upper
        )):
            raise ValueError("An interval is empty; check its bounds and endpoint choices.")
        if result and (item.lower < result[-1].upper or (
            item.lower == result[-1].upper and (item.include_lower or result[-1].include_upper)
        )):
            previous = result.pop()
            upper = max(previous.upper, item.upper)
            closed = ((previous.upper == upper and previous.include_upper)
                      or (item.upper == upper and item.include_upper))
            result.append(Interval(previous.lower, upper, previous.include_lower, closed))
        else:
            result.append(item)
    if not result:
        raise ValueError("At least one allowed interval is required.")
    return tuple(result)


def intersect(left: tuple[Interval, ...], right: tuple[Interval, ...]) -> tuple[Interval, ...]:
    out = []
    for a in left:
        for b in right:
            low, high = max(a.lower, b.lower), min(a.upper, b.upper)
            lc = a.contains(low) and b.contains(low)
            uc = a.contains(high) and b.contains(high)
            if low < high or low == high and lc and uc:
                out.append(Interval(low, high, lc, uc))
    return normalize(tuple(out)) if out else ()


def representable(low: Fraction, high: Fraction, lc: bool, uc: bool) -> Interval:
    a, b = float(low), float(high)
    if Fraction(a) < low or (not lc and Fraction(a) == low):
        a = math.nextafter(a, math.inf)
    if Fraction(b) > high or (not uc and Fraction(b) == high):
        b = math.nextafter(b, -math.inf)
    if not math.isfinite(a) or not math.isfinite(b) or a > b:
        raise ValueError("The feasible interval contains no representable input value.")
    return Interval(a, b)


def sample(intervals: tuple[Interval, ...], u: float, allocation: str) -> float:
    """Map one unit sample directly into a union of feasible float intervals."""
    intervals = normalize(intervals)
    positive = tuple(i for i in intervals if i.lower < i.upper)
    if not positive:
        return intervals[min(int(u * len(intervals)), len(intervals) - 1)].lower
    # Isolated values are supplied through explicit endpoint faces.
    intervals = positive
    widths = [Fraction(i.upper) - Fraction(i.lower) for i in intervals]
    weights = widths if allocation == "width" else [Fraction(1)] * len(intervals)
    position = Fraction(u) * sum(weights)
    for item, weight in zip(intervals, weights):
        if position < weight or item is intervals[-1]:
            local = min(1.0, max(0.0, float(position / weight)))
            bounds = representable(Fraction(item.lower), Fraction(item.upper),
                                   item.include_lower, item.include_upper)
            value = (1.0 - local) * bounds.lower + local * bounds.upper
            return min(bounds.upper, max(bounds.lower, value))
        position -= weight
    raise AssertionError("Nonempty interval union had no sample.")
