"""
Temperature schedule DSL parsing for Kindred.

Supports parsing temperature schedule definitions from DSL text:
- temp_const: T = 298.15 K  (constant temperature)
- temp_step: t=[0,50,100], T=[298,320,310]  (piecewise steps)
- temp_response: t=[0,50,100], T=[298,320], tau=10  (first-order response)

Integrates with TemperatureSchedule class from kindred.core.temperature.

Example DSL:
    # Constant temperature (default)
    temp_const: T=298.15

    # Piecewise temperature schedule
    temp_step: t=[0,50,100,150], T=[298,320,320,310]
    # Creates intervals: [0,50)→298K, [50,100)→320K, [100,150]→310K
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional

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


def parse_temperature_schedule(dsl_text: str) -> Optional[TemperatureSchedule]:
    """
    Parse temperature schedule from DSL text.

    Recognizes:
    - temp_const: T=298.15  (constant temperature)
    - temp_step: t=[0,50,100], T=[298,320,310]  (piecewise)
    - temp_response: t=[0,50,100], T=[298,320], tau=10

    Parameters
    ----------
    dsl_text : str
        DSL text containing temperature schedule definition

    Returns
    -------
    TemperatureSchedule or None
        Parsed schedule, or None if no temperature schedule found

    Raises
    ------
    TemperatureDSLError
        If temperature schedule syntax is invalid

    Examples
    --------
    >>> dsl = "temp_const: T=310.15"
    >>> schedule = parse_temperature_schedule(dsl)
    >>> schedule(0.0)
    310.15

    >>> dsl = "temp_step: t=[0,50,100], T=[298,320,310]"
    >>> schedule = parse_temperature_schedule(dsl)
    >>> schedule(25.0)
    298.0
    >>> schedule(75.0)
    320.0
    """
    lines = [line.strip() for line in dsl_text.splitlines()]

    for line in lines:
        if not line or line.startswith('#'):
            continue

        # Constant temperature: temp_const: T=298.15
        if line.lower().startswith('temp_const:'):
            return _parse_const_temperature(line)

        # Piecewise temperature: temp_step: t=[...], T=[...]
        if line.lower().startswith('temp_step:'):
            return _parse_piecewise_temperature(line)

        if line.lower().startswith('temp_response:'):
            return _parse_response_temperature(line)

    # No temperature schedule found
    return None


def _parse_const_temperature(line: str) -> TemperatureSchedule:
    """
    Parse constant temperature line.

    Format: temp_const: T=298.15

    Parameters
    ----------
    line : str
        DSL line

    Returns
    -------
    TemperatureSchedule
        Constant temperature schedule
    """
    # Extract T=<value>
    match = re.search(r'T\s*=\s*([0-9.eE+-]+)', line, re.IGNORECASE)
    if not match:
        raise TemperatureDSLError(
            "Invalid temp_const syntax",
            suggestion="Use format: temp_const: T=<value>",
            examples=[
                "temp_const: T=298.15",
                "temp_const: T=310.0"
            ]
        )

    try:
        temperature = float(match.group(1))
    except ValueError:
        raise TemperatureDSLError(
            f"Invalid temperature value: {match.group(1)}",
            suggestion="Temperature must be a valid number",
            examples=["T=298.15", "T=3.1015e+02"]
        )

    if temperature <= 0:
        raise TemperatureDSLError(
            f"Temperature must be positive, got {temperature} K",
            suggestion="Temperature must be in Kelvin (absolute temperature)",
            examples=[
                "T=298.15  (room temperature)",
                "T=273.15  (ice point)",
                "T=373.15  (boiling point of water)"
            ]
        )

    logger.info(f"Parsed constant temperature: T = {temperature} K")
    return TemperatureSchedule.constant(temperature)


def _parse_piecewise_temperature(line: str) -> TemperatureSchedule:
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
        return TemperatureSchedule.piecewise(intervals)
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


def _parse_response_temperature(line: str) -> TemperatureSchedule:
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
        return TemperatureSchedule.response(times, temperatures, tau=tau)
    except TemperatureScheduleError as e:
        raise TemperatureDSLError(f"Invalid response schedule: {e}")
