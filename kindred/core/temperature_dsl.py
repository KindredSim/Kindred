"""
Temperature schedule DSL parsing for Kindred.

Lowers one temperature schedule line to the temperature changes the
intervention schedule executor applies:
- temp_step: t=[0,50,100], T=[298,320]  (steps)
- temp_response: t=[0,50,100], T=[298,320], tau=10  (first-order response)

Example DSL:
    temp_step: t=[0,50,100,150], T=[298,320,310]
    # 298 K from t=0, 320 K from t=50 and 310 K from t=100; the final
    # 310 K continues after t=150.
"""

from __future__ import annotations

import re
from typing import List

from .authored_source_errors import AuthoredSourceError
from .species import validate_input_time_seconds, validate_temperature
from .temperature import TemperatureChange
from .validation import parse_finite_numeric_text

__all__ = ["parse_temperature_schedule", "TemperatureDSLError"]


class TemperatureDSLError(AuthoredSourceError):
    """Temperature schedule DSL parsing error."""


def _parse_finite_number(text: str, *, field: str) -> float:
    token = text.strip()
    if not token:
        raise TemperatureDSLError(f"{field} must not be empty.")
    try:
        return parse_finite_numeric_text(token, field=field)
    except ValueError as exc:
        raise TemperatureDSLError(str(exc)) from exc


def _parse_number_list(text: str, *, field: str) -> List[float]:
    """Parse one completely consumed, nonempty bracketed numeric list."""

    token = text.strip()
    if not (token.startswith("[") and token.endswith("]")):
        raise TemperatureDSLError(f"{field} must be a bracketed numeric list.")
    contents = token[1:-1]
    parts = contents.split(",")
    if not contents.strip() or any(not part.strip() for part in parts):
        raise TemperatureDSLError(
            f"{field} must not contain empty list elements."
        )
    return [
        _parse_finite_number(part, field=f"{field} element")
        for part in parts
    ]


def _split_directive_fields(payload: str, *, syntax_name: str) -> list[str]:
    fields: list[str] = []
    start = 0
    bracket_depth = 0
    for index, character in enumerate(payload):
        if character == "[":
            if bracket_depth:
                raise TemperatureDSLError(
                    f"Invalid {syntax_name} syntax: nested lists are not supported."
                )
            bracket_depth = 1
        elif character == "]":
            if not bracket_depth:
                raise TemperatureDSLError(
                    f"Invalid {syntax_name} syntax: unmatched closing bracket."
                )
            bracket_depth = 0
        elif character == "," and not bracket_depth:
            fields.append(payload[start:index].strip())
            start = index + 1
    if bracket_depth:
        raise TemperatureDSLError(
            f"Invalid {syntax_name} syntax: unclosed list."
        )
    fields.append(payload[start:].strip())
    if any(not field for field in fields):
        raise TemperatureDSLError(
            f"Invalid {syntax_name} syntax: fields must not be empty."
        )
    return fields


def _parse_directive_fields(
    line: str,
    *,
    syntax_name: str,
    expected_fields: frozenset[str],
) -> dict[str, str]:
    source, _comment_marker, _comment = line.partition("#")
    match = re.fullmatch(
        rf"(?is){re.escape(syntax_name)}:(?P<payload>.*)",
        source.strip(),
    )
    # The caller passes a line that starts with this directive.
    payload = match.group("payload").strip()
    if not payload:
        raise TemperatureDSLError(
            f"Invalid {syntax_name} syntax: directive fields are required."
        )

    parsed: dict[str, str] = {}
    for field in _split_directive_fields(payload, syntax_name=syntax_name):
        if field.count("=") != 1:
            raise TemperatureDSLError(
                f"Invalid {syntax_name} field {field!r}. Expected name=value."
            )
        name, value = (part.strip() for part in field.split("=", 1))
        if not name or not value:
            raise TemperatureDSLError(
                f"Invalid {syntax_name} field {field!r}. Name and value are required."
            )
        if name in parsed:
            raise TemperatureDSLError(
                f"Duplicate {syntax_name} field {name!r}."
            )
        parsed[name] = value

    missing = sorted(expected_fields - set(parsed))
    unknown = sorted(set(parsed) - expected_fields)
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append("Missing: " + ", ".join(missing) + ".")
        if unknown:
            details.append("Unknown: " + ", ".join(unknown) + ".")
        raise TemperatureDSLError(
            f"Invalid {syntax_name} fields. " + " ".join(details)
        )
    return parsed


def parse_temperature_schedule(
    directive_line: str,
    *,
    nominal_temperature: float,
) -> tuple[TemperatureChange, ...]:
    """
    Lower one admitted temperature schedule directive to its changes.

    Each temperature applies from its time until the next one; the final one
    continues after the last time.  Before the first time the nominal
    temperature applies.  A change to the temperature already in force is no
    change and is left out.
    """
    # Parsing hands over one line that starts with temp_step: or temp_response:.
    line = directive_line.strip()
    syntax_name = line.partition(":")[0].strip().lower()
    response = syntax_name == "temp_response"
    fields = _parse_directive_fields(
        line,
        syntax_name=syntax_name,
        expected_fields=frozenset({"t", "T", "tau"} if response else {"t", "T"}),
    )
    times = _parse_number_list(fields["t"], field=f"{syntax_name} t")
    temperatures = _parse_number_list(fields["T"], field=f"{syntax_name} T")
    if len(times) < 2:
        raise TemperatureDSLError("t needs at least two time points.")
    if len(temperatures) != len(times) - 1:
        raise TemperatureDSLError("T must have one temperature fewer than t has time points.")
    try:
        times = [validate_input_time_seconds(time, field="t") for time in times]
        temperatures = [validate_temperature(value, field="T") for value in temperatures]
        tau = (
            validate_input_time_seconds(
                _parse_finite_number(fields["tau"], field="temp_response tau"),
                field="tau", allow_zero=False,
            )
            if response else None
        )
    except ValueError as exc:
        raise TemperatureDSLError(str(exc)) from exc
    for earlier, later in zip(times, times[1:]):
        if later <= earlier:
            raise TemperatureDSLError(
                f"Each time in t must be greater than the one before it ({later:g} s follows {earlier:g} s)."
            )
    changes: list[TemperatureChange] = []
    current = nominal_temperature
    for time, temperature in zip(times, temperatures):
        if temperature != current:
            changes.append(TemperatureChange(time, temperature, tau))
            current = temperature
    return tuple(changes)
