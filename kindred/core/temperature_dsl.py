"""
Temperature schedule DSL parsing for Kindred.

Supports parsing temperature schedule definitions from DSL text:
- temp_step: t=[0,50,100], T=[298,320,310]  (piecewise steps)
- temp_response: t=[0,50,100], T=[298,320], tau=10  (first-order response)

Integrates with TemperatureSchedule class from kindred.core.temperature.

Example DSL:
    # Piecewise temperature schedule
    temp_step: t=[0,50,100,150], T=[298,320,320,310]
    # Creates intervals: [0,50)→298K, [50,100)→320K, [100,150]→310K
"""

from __future__ import annotations

import logging
import re
from typing import List

from .temperature import TemperatureSchedule, TemperatureScheduleError

logger = logging.getLogger(__name__)

__all__ = ["parse_temperature_schedule", "TemperatureDSLError"]


class TemperatureDSLError(ValueError):
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


def _parse_number_list(text: str) -> List[float]:
    """
    Parse comma-separated list of numbers from string like '[1,2,3]' or '1,2,3'.

    Parameters
    ----------
    text : str
        Text containing numbers

    Returns
    -------
    list of float
        Parsed numbers

    Raises
    ------
    TemperatureDSLError
        If parsing fails
    """
    # Remove brackets if present
    text = text.strip()
    if text.startswith('[') and text.endswith(']'):
        text = text[1:-1]

    # Split by comma and parse
    parts = [p.strip() for p in text.split(',')]
    try:
        return [float(p) for p in parts if p]
    except ValueError as e:
        raise TemperatureDSLError(f"Failed to parse number list '{text}': {e}")


def parse_temperature_schedule(
    directive_line: str,
    *,
    nominal_temperature: float = 298.15,
) -> TemperatureSchedule:
    """
    Parse one admitted temperature schedule directive.

    Recognizes:
    - temp_step: t=[0,50,100], T=[298,320,310]  (piecewise)
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
    >>> dsl = "temp_step: t=[0,50,100], T=[298,320,310]"
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

    Format: temp_step: t=[0,50,100], T=[298,320,310]

    Creates intervals:
    - [0, 50) → 298 K
    - [50, 100) → 320 K
    - [100, inf) → 310 K (last interval extends to infinity)

    Parameters
    ----------
    line : str
        DSL line

    Returns
    -------
    TemperatureSchedule
        Piecewise temperature schedule
    """
    times, temperatures = _parse_time_temperature_lists(
        line,
        syntax_name="temp_step",
        examples=[
            "temp_step: t=[0,50,100], T=[298,350]",
            "temp_step: t=[0,25,50,75], T=[298,310,320]",
        ],
    )

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


def _parse_time_temperature_lists(
    line: str,
    *,
    syntax_name: str,
    examples: list[str],
) -> tuple[List[float], List[float]]:
    t_match = re.search(r'\bt\s*=\s*\[([^\]]+)\]', line)
    T_match = re.search(r'\bT\s*=\s*\[([^\]]+)\]', line)

    if not t_match or not T_match:
        raise TemperatureDSLError(
            f"Invalid {syntax_name} syntax",
            suggestion=f"Format: {syntax_name}: t=[time_points], T=[temperatures]",
            examples=examples,
        )

    times = _parse_number_list(t_match.group(1))
    temperatures = _parse_number_list(T_match.group(1))
    return times, temperatures


def _parse_response_temperature(
    line: str,
    *,
    nominal_temperature: float,
) -> TemperatureSchedule:
    """
    Parse first-order response temperature line.

    Format: temp_response: t=[0,50,100], T=[298,320], tau=10
    """
    times, temperatures = _parse_time_temperature_lists(
        line,
        syntax_name="temp_response",
        examples=[
            "temp_response: t=[0,50,100], T=[298,350], tau=10",
            "temp_response: t=[0,25,50,75], T=[298,310,320], tau=5",
        ],
    )
    tau_match = re.search(r'\btau\s*=\s*([0-9.eE+-]+)', line, re.IGNORECASE)
    if not tau_match:
        raise TemperatureDSLError(
            "Invalid temp_response syntax",
            suggestion="Format: temp_response: t=[time_points], T=[temperatures], tau=<value>",
            examples=[
                "temp_response: t=[0,50,100], T=[298,350], tau=10",
                "temp_response: t=[0,25,50,75], T=[298,310,320], tau=5",
            ],
        )

    try:
        tau = float(tau_match.group(1))
    except ValueError:
        raise TemperatureDSLError(
            f"Invalid tau value: {tau_match.group(1)}",
            suggestion="tau must be a valid positive number",
        )
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
