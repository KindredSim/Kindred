"""
Temperature schedule DSL parsing for Kindred.

Supports parsing temperature schedule definitions from DSL text:
- temp_step: t=[0,50,100], T=[298,320]  (piecewise steps)
- temp_response: t=[0,50,100], T=[298,320], tau=10  (first-order response)

Integrates with TemperatureSchedule class from kindred.core.temperature.

Example DSL:
    # Piecewise temperature schedule
    temp_step: t=[0,50,100,150], T=[298,320,310]
    # Creates intervals: [0,50)→298K, [50,100)→320K, [100,150)→310K;
    # the final 310K temperature continues after t=150.
"""

from __future__ import annotations

import logging
import re
from typing import List

from .authored_source_errors import AuthoredSourceError
from .temperature import TemperatureSchedule, TemperatureScheduleError
from .validation import parse_finite_numeric_text

logger = logging.getLogger(__name__)

__all__ = ["parse_temperature_schedule", "TemperatureDSLError"]


class TemperatureDSLError(AuthoredSourceError):
    """
    Enhanced error for temperature schedule DSL parsing.

    Provides helpful context and examples for common mistakes.
    """
    def __init__(self, message: str, *, suggestion: str = None, examples: list = None):
        self.suggestion = suggestion
        self.examples = examples or []

        # Build comprehensive error message
        parts = [message]
        if suggestion:
            parts.append(f"\nSuggestion: {suggestion}")
        if examples:
            parts.append("\nValid examples:")
            for ex in examples:
                parts.append(f"  • {ex}")

        super().__init__("\n".join(parts))


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
    if match is None:
        raise TemperatureDSLError(f"Invalid {syntax_name} syntax.")
    payload = match.group("payload").strip()
    if not payload:
        raise TemperatureDSLError(
            f"Invalid {syntax_name} syntax: directive fields are required."
        )

    parsed: dict[str, str] = {}
    for field in _split_directive_fields(payload, syntax_name=syntax_name):
        if field.count("=") != 1:
            raise TemperatureDSLError(
                f"Invalid {syntax_name} field {field!r}; expected name=value."
            )
        name, value = (part.strip() for part in field.split("=", 1))
        if not name or not value:
            raise TemperatureDSLError(
                f"Invalid {syntax_name} field {field!r}; name and value are required."
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
            details.append("missing " + ", ".join(missing))
        if unknown:
            details.append("unknown " + ", ".join(unknown))
        raise TemperatureDSLError(
            f"Invalid {syntax_name} fields: " + "; ".join(details) + "."
        )
    return parsed


def parse_temperature_schedule(
    directive_line: str,
    *,
    nominal_temperature: float = 298.15,
) -> TemperatureSchedule:
    """
    Parse one admitted temperature schedule directive.

    Recognizes:
    - temp_step: t=[0,50,100], T=[298,320]  (piecewise)
    - temp_response: t=[0,50,100], T=[298,320], tau=10

    Parameters
    ----------
    directive_line : str
        One top-level temperature schedule directive already admitted by the
        simulator DSL owner.

    Returns
    -------
    TemperatureSchedule
        Parsed schedule

    Raises
    ------
    TemperatureDSLError
        If temperature schedule syntax is invalid

    Examples
    --------
    >>> dsl = "temp_step: t=[0,50,100], T=[298,320]"
    >>> schedule = parse_temperature_schedule(dsl)
    >>> schedule(25.0)
    298.0
    >>> schedule(75.0)
    320.0
    """
    lines = directive_line.splitlines()
    if len(lines) != 1:
        raise TemperatureDSLError(
            "Temperature schedule parsing requires exactly one admitted directive."
        )
    line = lines[0].strip()
    lower = line.lower()
    if lower.startswith('temp_step:'):
        return _parse_piecewise_temperature(
            line,
            nominal_temperature=nominal_temperature,
        )
    if lower.startswith('temp_response:'):
        return _parse_response_temperature(
            line,
            nominal_temperature=nominal_temperature,
        )
    raise TemperatureDSLError(
        "Expected one admitted temp_step: or temp_response: directive."
    )

def _parse_piecewise_temperature(
    line: str,
    *,
    nominal_temperature: float,
) -> TemperatureSchedule:
    """
    Parse piecewise temperature line.

    Format: temp_step: t=[0,50,100], T=[298,320]

    Creates intervals:
    - [0, 50) → 298 K
    - [50, 100) → 320 K
    - [100, inf) → 320 K (the final temperature continues)

    Parameters
    ----------
    line : str
        DSL line

    Returns
    -------
    TemperatureSchedule
        Piecewise temperature schedule
    """
    fields = _parse_directive_fields(
        line,
        syntax_name="temp_step",
        expected_fields=frozenset({"t", "T"}),
    )
    times = _parse_number_list(fields["t"], field="temp_step t")
    temperatures = _parse_number_list(fields["T"], field="temp_step T")

    # Validate
    if len(times) < 2:
        raise TemperatureDSLError(
            f"At least 2 time points required, got {len(times)}",
            suggestion="Provide at least 2 time points to define an interval",
            examples=[
                "t=[0,100], T=[298]  (one interval)",
                "t=[0,50,100], T=[298,350]  (two intervals)"
            ]
        )

    if len(temperatures) != len(times) - 1:
        raise TemperatureDSLError(
            f"Temperature count mismatch: {len(temperatures)} temperatures for {len(times)} time points",
            suggestion=f"Need exactly {len(times)-1} temperature(s) for {len(times)} time points",
            examples=[
                "t=[0,50,100], T=[298,350]  (2 intervals: [0,50)→298K, [50,100)→350K)",
                "t=[0,25,50,75], T=[298,310,320]  (3 intervals)"
            ]
        )

    # Build intervals: [(t_start, t_end, T), ...]
    intervals = []
    for i in range(len(temperatures)):
        t_start = times[i]
        t_end = times[i + 1]
        temp = temperatures[i]
        intervals.append((t_start, t_end, temp))

    logger.info(f"Parsed piecewise temperature schedule with {len(intervals)} intervals")
    for t_start, t_end, temp in intervals:
        logger.debug(f"  t ∈ [{t_start}, {t_end}): T = {temp} K")

    try:
        return TemperatureSchedule.piecewise(
            intervals,
            baseline_temperature=nominal_temperature,
        )
    except TemperatureScheduleError as e:
        raise TemperatureDSLError(f"Invalid piecewise schedule: {e}")


def _parse_response_temperature(
    line: str,
    *,
    nominal_temperature: float,
) -> TemperatureSchedule:
    """
    Parse first-order response temperature line.

    Format: temp_response: t=[0,50,100], T=[298,320], tau=10
    """
    fields = _parse_directive_fields(
        line,
        syntax_name="temp_response",
        expected_fields=frozenset({"t", "T", "tau"}),
    )
    times = _parse_number_list(fields["t"], field="temp_response t")
    temperatures = _parse_number_list(
        fields["T"], field="temp_response T"
    )
    tau = _parse_finite_number(fields["tau"], field="temp_response tau")
    if tau <= 0:
        raise TemperatureDSLError(
            f"tau must be positive, got {tau}",
            suggestion="Use a strictly positive time constant",
        )
    if len(times) < 2:
        raise TemperatureDSLError(
            f"At least 2 time points required, got {len(times)}",
            suggestion="Provide at least 2 time points to define an interval",
        )
    if len(temperatures) != len(times) - 1:
        raise TemperatureDSLError(
            f"Temperature count mismatch: {len(temperatures)} temperatures for {len(times)} time points",
            suggestion=f"Need exactly {len(times)-1} temperature(s) for {len(times)} time points",
        )

    try:
        return TemperatureSchedule.response(
            times,
            temperatures,
            tau=tau,
            baseline_temperature=nominal_temperature,
        )
    except TemperatureScheduleError as e:
        raise TemperatureDSLError(f"Invalid response schedule: {e}")
