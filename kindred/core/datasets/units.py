from __future__ import annotations

from typing import Sequence
from kindred.core.quantity_units import parse_unit as scientific_parse_unit, conversion_scale, MOLAR, SECOND

__all__ = [
    "CONCENTRATION_UNIT_DISPLAY",
    "TIME_UNIT_DISPLAY",
    "looks_like_unit_row",
    "parse_concentration_unit",
    "parse_time_unit",
    "parse_unit",
]

_TIME_ALIASES = {
    "us": "µs",
}
_TIME_DISPLAY_UNITS = ("fs", "ps", "ns", "µs", "us", "μs", "ms", "s", "min", "h")
TIME_UNIT_DISPLAY = ("fs", "ps", "ns", "us", "ms", "s", "min", "h")

_CONCENTRATION_ALIASES = {
    "uM": "µM",
}
_CONCENTRATION_DISPLAY_UNITS = ("fM", "pM", "nM", "µM", "uM", "μM", "mM", "M")
CONCENTRATION_UNIT_DISPLAY = ("fM", "pM", "nM", "uM", "mM", "M")


def parse_time_unit(unit: str) -> float:
    """Return the conversion factor from a supported dataset time unit to seconds."""
    normalized = _normalize_unit_text(unit)
    if not normalized:
        raise ValueError(
            f"Unsupported time unit {unit!r}. Supported time units: {', '.join(_TIME_DISPLAY_UNITS)}"
        )
    canonical = _normalize_micro_prefix(_TIME_ALIASES.get(normalized, normalized))
    if canonical.lower() == "min":
        canonical = "min"
    if canonical in _TIME_DISPLAY_UNITS:
        return conversion_scale(scientific_parse_unit(canonical), SECOND).number()

    raise ValueError(
        f"Unsupported time unit {unit!r}. Supported time units: {', '.join(_TIME_DISPLAY_UNITS)}"
    )


def parse_concentration_unit(unit: str) -> float:
    """Return the conversion factor from a supported dataset concentration unit to molar."""
    normalized = _normalize_unit_text(unit)
    if not normalized:
        raise ValueError(
            "Unsupported concentration unit "
            f"{unit!r}. Supported concentration units: {', '.join(_CONCENTRATION_DISPLAY_UNITS)}"
        )
    canonical = _normalize_micro_prefix(_CONCENTRATION_ALIASES.get(normalized, normalized))
    if canonical in _CONCENTRATION_DISPLAY_UNITS:
        return conversion_scale(scientific_parse_unit(canonical), MOLAR).number()

    raise ValueError(
        "Unsupported concentration unit "
        f"{unit!r}. Supported concentration units: {', '.join(_CONCENTRATION_DISPLAY_UNITS)}"
    )


def looks_like_unit_row(values: Sequence[str]) -> bool:
    """Heuristically detect whether a row of dataset cells looks like a units row."""
    non_empty_count = 0
    recognized_count = 0
    for value in values:
        text = str(value).strip()
        if not text:
            continue
        non_empty_count += 1
        try:
            parse_unit(text)
        except ValueError:
            continue
        recognized_count += 1
    if non_empty_count == 0:
        return False
    return (recognized_count / non_empty_count) >= 0.5


def parse_unit(unit: str) -> tuple[str, float]:
    """Return the dataset unit category and factor for a supported time or concentration unit."""
    try:
        return "time", parse_time_unit(unit)
    except ValueError:
        pass

    try:
        return "concentration", parse_concentration_unit(unit)
    except ValueError:
        pass

    raise ValueError(
        f"Unsupported unit {unit!r}. "
        f"Supported time units: {', '.join(_TIME_DISPLAY_UNITS)}. "
        f"Supported concentration units: {', '.join(_CONCENTRATION_DISPLAY_UNITS)}."
    )


def _normalize_unit_text(unit: str) -> str:
    return str(unit).strip()


def _normalize_micro_prefix(unit: str) -> str:
    return unit.replace("μ", "µ")
