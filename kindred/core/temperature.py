"""
Temperature schedule support for Kindred simulations.

Supports constant and piecewise temperature profiles:
- Constant: T = 298.15 K
- Piecewise: T(t) defined over time intervals

Usage:
    from kindred.core.temperature import TemperatureSchedule

    # Constant temperature
    T_const = TemperatureSchedule.constant(298.15)

    # Piecewise temperature
    intervals = [(0, 50, 298), (50, 100, 320)]
    T_piece = TemperatureSchedule.piecewise(intervals)

    # Evaluate
    temperature = T_const(25.0)  # Returns 298.15
    temperature = T_piece(75.0)  # Returns 320
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from typing import Callable, List, Mapping, Optional, Protocol, Tuple, runtime_checkable

logger = logging.getLogger(__name__)

__all__ = [
    "TemperatureSchedule",
    "TemperatureScheduleError",
    "TemperatureScheduleProtocol",
    "TemperatureScheduleDictProtocol",
    "coerce_temperature_schedule",
]


@runtime_checkable
class TemperatureScheduleProtocol(Protocol):
    def __call__(self, t: float) -> float: ...


@runtime_checkable
class TemperatureScheduleDictProtocol(TemperatureScheduleProtocol, Protocol):
    def to_dict(self) -> Mapping[str, object]: ...


def coerce_temperature_schedule(value: object) -> TemperatureScheduleProtocol | None:
    """
    Normalize a temperature schedule value coming from module boundaries.

    Accepted inputs:
    - None
    - A callable schedule object (e.g., TemperatureSchedule instance)
    - A mapping produced by TemperatureSchedule.to_dict() (converted via from_dict)

    Raises TypeError for non-callable, non-mapping values to avoid silent
    mismatches across parsing/metadata/solver boundaries.
    """
    if value is None:
        return None
    if isinstance(value, Mapping):
        if "type" not in value:
            raise TypeError(
                "temperature_schedule mapping must contain a 'type' key (expected TemperatureSchedule.to_dict() format)"
            )
        return TemperatureSchedule.from_dict(dict(value))
    if callable(value):
        return value  # type: ignore[return-value]
    raise TypeError(
        f"temperature_schedule must be callable or a mapping (TemperatureSchedule.to_dict()), got {type(value)!r}"
    )


class TemperatureScheduleError(Exception):
    """Exception raised for temperature schedule errors."""
    pass


@dataclass
class TimeInterval:
    """
    Time interval with associated temperature.

    Attributes
    ----------
    t_start : float
        Start time (inclusive)
    t_end : float
        End time (exclusive, except for last interval)
    temperature : float
        Temperature in Kelvin
    """
    t_start: float
    t_end: float
    temperature: float

    def __post_init__(self):
        """Validate interval."""
        if self.t_start < 0:
            raise TemperatureScheduleError(f"Interval start time must be non-negative, got {self.t_start}")
        if self.t_end <= self.t_start:
            raise TemperatureScheduleError(
                f"Interval end time ({self.t_end}) must be greater than start time ({self.t_start})"
            )
        if self.temperature <= 0:
            raise TemperatureScheduleError(f"Temperature must be positive, got {self.temperature} K")

    def contains(self, t: float) -> bool:
        """Check if time is in this interval."""
        return self.t_start <= t < self.t_end

    def contains_inclusive(self, t: float) -> bool:
        """Check if time is in this interval (inclusive end)."""
        return self.t_start <= t <= self.t_end


class TemperatureSchedule:
    """
    Temperature schedule T(t) for simulations.

    Supports:
    - Constant temperature (default)
    - Piecewise constant temperature over time intervals

    Examples
    --------
    Constant temperature:
    >>> T = TemperatureSchedule.constant(298.15)
    >>> T(10.0)
    298.15

    Piecewise temperature:
    >>> intervals = [(0, 50, 298), (50, 100, 320)]
    >>> T = TemperatureSchedule.piecewise(intervals)
    >>> T(25.0)
    298.0
    >>> T(75.0)
    320.0
    """

    def __init__(self, T_func: Callable[[float], float], schedule_type: str = "constant"):
        """
        Initialize temperature schedule.

        Parameters
        ----------
        T_func : callable
            Function T(t) that returns temperature in Kelvin
        schedule_type : str
            Type of schedule: "constant" or "piecewise"
        """
        self._T_func = T_func
        self.schedule_type = schedule_type
        self._intervals: List[TimeInterval] = []
        self._response_times: List[float] = []
        self._response_temperatures: List[float] = []
        self._response_tau: Optional[float] = None

    @staticmethod
    def _validated_intervals(intervals: List[Tuple[float, float, float]]) -> List[TimeInterval]:
        if not intervals:
            raise TemperatureScheduleError("At least one interval required for piecewise schedule")

        sorted_intervals = sorted(intervals, key=lambda x: x[0])
        time_intervals: List[TimeInterval] = []
        for i, (t_start, t_end, temp) in enumerate(sorted_intervals):
            try:
                interval = TimeInterval(t_start, t_end, temp)
                time_intervals.append(interval)
            except TemperatureScheduleError as e:
                raise TemperatureScheduleError(f"Invalid interval {i+1} ({t_start}, {t_end}, {temp}): {e}")

        for i in range(len(time_intervals) - 1):
            current = time_intervals[i]
            next_interval = time_intervals[i + 1]

            if current.t_end != next_interval.t_start:
                if current.t_end < next_interval.t_start:
                    raise TemperatureScheduleError(
                        f"Gap in intervals: [{current.t_start}, {current.t_end}) and "
                        f"[{next_interval.t_start}, {next_interval.t_end}). "
                        f"Missing coverage for t ∈ [{current.t_end}, {next_interval.t_start})"
                    )
                raise TemperatureScheduleError(
                    f"Overlapping intervals: [{current.t_start}, {current.t_end}) and "
                    f"[{next_interval.t_start}, {next_interval.t_end})"
                )

        if time_intervals[0].t_start != 0:
            raise TemperatureScheduleError(
                f"First interval must start at t=0, got t={time_intervals[0].t_start}"
            )

        return time_intervals

    def __call__(self, t: float) -> float:
        """
        Evaluate temperature at time t.

        Parameters
        ----------
        t : float
            Time in seconds

        Returns
        -------
        float
            Temperature in Kelvin
        """
        return self._T_func(t)

    @property
    def fingerprint(self) -> str:
        """
        Deterministic, serialization-friendly fingerprint for caching/metadata.

        Prefer this over `repr(self)` when building cache keys.
        """
        try:
            payload = self.to_dict()
            return json.dumps(payload, sort_keys=True, separators=(",", ":"))
        except Exception:
            return repr(self)

    def __reduce__(self):
        """
        Make TemperatureSchedule picklable by round-tripping through `to_dict`.

        This avoids pickling closures captured by constant/piecewise schedules.
        """
        return (type(self).from_dict, (self.to_dict(),))

    @classmethod
    def constant(cls, temperature: float) -> TemperatureSchedule:
        """
        Create constant temperature schedule.

        Parameters
        ----------
        temperature : float
            Constant temperature in Kelvin

        Returns
        -------
        TemperatureSchedule
            Constant temperature schedule

        Raises
        ------
        TemperatureScheduleError
            If temperature is non-positive
        """
        if temperature <= 0:
            raise TemperatureScheduleError(f"Temperature must be positive, got {temperature} K")

        def T_const(t: float) -> float:
            return temperature

        schedule = cls(T_const, schedule_type="constant")
        logger.debug(f"Created constant temperature schedule: T = {temperature} K")
        return schedule

    @classmethod
    def piecewise(cls, intervals: List[Tuple[float, float, float]]) -> TemperatureSchedule:
        """
        Create piecewise constant temperature schedule.

        Parameters
        ----------
        intervals : list of tuple
            List of (t_start, t_end, temperature) tuples
            - t_start: interval start time (inclusive)
            - t_end: interval end time (exclusive, except last)
            - temperature: temperature in Kelvin

        Returns
        -------
        TemperatureSchedule
            Piecewise temperature schedule

        Raises
        ------
        TemperatureScheduleError
            If intervals are invalid (overlapping, gaps, negative times, etc.)

        Examples
        --------
        >>> intervals = [(0, 50, 298), (50, 100, 320), (100, 150, 310)]
        >>> T = TemperatureSchedule.piecewise(intervals)
        >>> T(25.0)
        298.0
        >>> T(75.0)
        320.0
        >>> T(125.0)
        310.0
        """
        time_intervals = cls._validated_intervals(intervals)

        # Create lookup function
        def T_piecewise(t: float) -> float:
            if t < 0:
                raise ValueError(f"Time must be non-negative, got t={t}")

            # Find matching interval
            for i, interval in enumerate(time_intervals):
                # Last interval includes endpoint
                if i == len(time_intervals) - 1:
                    if interval.contains_inclusive(t):
                        return interval.temperature
                else:
                    if interval.contains(t):
                        return interval.temperature

            # If t is beyond last interval, extrapolate with last temperature
            last_interval = time_intervals[-1]
            if t >= last_interval.t_end:
                logger.warning(
                    f"Time t={t} exceeds piecewise schedule range "
                    f"[{time_intervals[0].t_start}, {last_interval.t_end}]. "
                    f"Using last temperature: {last_interval.temperature} K"
                )
                return last_interval.temperature

            # Should never reach here if validation is correct
            raise TemperatureScheduleError(f"No interval found for t={t}")

        schedule = cls(T_piecewise, schedule_type="piecewise")
        schedule._intervals = time_intervals

        logger.debug(f"Created piecewise temperature schedule with {len(intervals)} intervals:")
        for interval in time_intervals:
            logger.debug(f"  t ∈ [{interval.t_start}, {interval.t_end}): T = {interval.temperature} K")

        return schedule

    @classmethod
    def response(
        cls,
        times: List[float],
        temperatures: List[float],
        *,
        tau: float,
    ) -> TemperatureSchedule:
        """
        Create first-order response schedule driven by a step setpoint schedule.

        The setpoint is piecewise constant over [times[i], times[i+1]) with value
        temperatures[i]. The actual runtime temperature evolves as:

            dT/dt = (T_set(t) - T(t)) / tau
        """
        if not math.isfinite(float(tau)) or float(tau) <= 0.0:
            raise TemperatureScheduleError(f"tau must be positive and finite, got {tau}")

        if len(times) < 2:
            raise TemperatureScheduleError(f"At least 2 time points required, got {len(times)}")
        if len(temperatures) != len(times) - 1:
            raise TemperatureScheduleError(
                f"Temperature count mismatch: {len(temperatures)} temperatures for {len(times)} time points"
            )

        time_points = [float(t) for t in times]
        setpoints = [float(T) for T in temperatures]
        time_intervals = cls._validated_intervals(
            [
                (time_points[i], time_points[i + 1], setpoints[i])
                for i in range(len(setpoints))
            ]
        )
        tau_f = float(tau)

        start_actuals = [float(setpoints[0])]
        for interval in time_intervals:
            start_actual = float(start_actuals[-1])
            duration = float(interval.t_end - interval.t_start)
            decay = math.exp(-duration / tau_f)
            end_actual = float(interval.temperature) + (start_actual - float(interval.temperature)) * decay
            start_actuals.append(float(end_actual))

        def T_response(t: float) -> float:
            if t < 0:
                raise ValueError(f"Time must be non-negative, got t={t}")

            for i, interval in enumerate(time_intervals):
                if i == len(time_intervals) - 1:
                    in_interval = interval.contains_inclusive(t)
                else:
                    in_interval = interval.contains(t)
                if in_interval:
                    dt = float(t - interval.t_start)
                    return float(interval.temperature) + (
                        float(start_actuals[i]) - float(interval.temperature)
                    ) * math.exp(-dt / tau_f)

            last_interval = time_intervals[-1]
            if t >= last_interval.t_end:
                dt = float(t - last_interval.t_end)
                last_setpoint = float(last_interval.temperature)
                start_actual = float(start_actuals[-1])
                return last_setpoint + (start_actual - last_setpoint) * math.exp(-dt / tau_f)

            raise TemperatureScheduleError(f"No interval found for t={t}")

        schedule = cls(T_response, schedule_type="response")
        schedule._intervals = time_intervals
        schedule._response_times = list(time_points)
        schedule._response_temperatures = list(setpoints)
        schedule._response_tau = tau_f

        logger.debug(
            "Created response temperature schedule with %d intervals and tau=%s",
            len(time_intervals),
            tau_f,
        )
        return schedule

    def get_intervals(self) -> List[TimeInterval]:
        """
        Get list of time intervals (for piecewise schedules).

        Returns
        -------
        list of TimeInterval
            List of intervals, empty for constant schedules
        """
        return self._intervals.copy()

    def get_time_range(self) -> Optional[Tuple[float, float]]:
        """
        Get time range covered by schedule.

        Returns
        -------
        tuple of (float, float) or None
            (t_min, t_max) for piecewise schedules, None for constant
        """
        if self.schedule_type in {"piecewise", "response"} and self._intervals:
            return (self._intervals[0].t_start, self._intervals[-1].t_end)
        return None

    def to_dict(self) -> dict:
        """
        Convert schedule to dictionary (for serialization).

        Returns
        -------
        dict
            Dictionary representation
        """
        if self.schedule_type == "constant":
            # Get constant temperature by evaluating at t=0
            T = self._T_func(0.0)
            return {
                "type": "constant",
                "temperature": T
            }
        if self.schedule_type == "piecewise":
            return {
                "type": "piecewise",
                "intervals": [
                    {
                        "t_start": interval.t_start,
                        "t_end": interval.t_end,
                        "temperature": interval.temperature
                    }
                    for interval in self._intervals
                ]
            }
        if self.schedule_type == "response":
            return {
                "type": "response",
                "times": list(self._response_times),
                "temperatures": list(self._response_temperatures),
                "tau": float(self._response_tau if self._response_tau is not None else 0.0),
            }
        raise TemperatureScheduleError(f"Unknown schedule type: {self.schedule_type}")

    @classmethod
    def from_dict(cls, data: dict) -> TemperatureSchedule:
        """
        Create schedule from dictionary.

        Parameters
        ----------
        data : dict
            Dictionary with schedule data

        Returns
        -------
        TemperatureSchedule
            Temperature schedule
        """
        schedule_type = data.get("type", "constant")

        if schedule_type == "constant":
            temperature = data.get("temperature", 298.15)
            return cls.constant(temperature)

        elif schedule_type == "piecewise":
            intervals_data = data.get("intervals", [])
            intervals = [
                (item["t_start"], item["t_end"], item["temperature"])
                for item in intervals_data
            ]
            return cls.piecewise(intervals)
        elif schedule_type == "response":
            times = data.get("times", [])
            temperatures = data.get("temperatures", [])
            tau = data.get("tau", None)
            return cls.response(times, temperatures, tau=float(tau))

        else:
            raise TemperatureScheduleError(f"Unknown schedule type: {schedule_type}")

    def __repr__(self) -> str:
        """String representation."""
        if self.schedule_type == "constant":
            T = self._T_func(0.0)
            return f"TemperatureSchedule.constant({T} K)"
        if self.schedule_type == "piecewise":
            n = len(self._intervals)
            return f"TemperatureSchedule.piecewise({n} intervals)"
        if self.schedule_type == "response":
            n = len(self._intervals)
            tau = self._response_tau
            return f"TemperatureSchedule.response({n} intervals, tau={tau})"
        return f"TemperatureSchedule({self.schedule_type})"
