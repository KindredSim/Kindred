"""Dataset declaration admission and conservative units-row recognition."""

from __future__ import annotations

import re
from typing import Sequence

from kindred.core.quantity_units import Unit, SECOND, parse_declared_unit

TIME_UNIT_DISPLAY = ("fs", "ps", "ns", "us", "ms", "s", "min", "h", "d")
MEASUREMENT_UNIT_DISPLAY = (
    "1", "M", "mM", "uM", "nM", "pM", "fM", "M/s", "s", "K", "J/mol", "L",
)


def parse_measurement_unit(text: str) -> Unit:
    """Admit existing unit syntax, preserving dataset minute normalization."""
    normalized = re.sub(r"(?<![A-Za-zµμ])min(?![A-Za-zµμ])", "min", text.strip(), flags=re.IGNORECASE)
    return parse_declared_unit(normalized)


def parse_time_unit(text: str) -> Unit:
    unit = parse_measurement_unit(text)
    if not unit.compatible(SECOND):
        raise ValueError(f"Time requires a time unit, not {text!r}.")
    return unit


def looks_like_unit_row(values: Sequence[object]) -> bool:
    """Require recognized textual evidence; numeric rows remain measurements."""
    nonempty = recognized = 0
    textual_unit = False
    for value in values:
        text = "" if value is None else str(value).strip()
        if not text:
            continue
        nonempty += 1
        try:
            parse_measurement_unit(text)
        except ValueError:
            continue
        recognized += 1
        try:
            float(text)
        except ValueError:
            textual_unit = True
    return textual_unit and recognized * 2 >= nonempty
