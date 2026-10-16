"""
Temperature schedule support for Kindred simulations.

Supports the two authored temperature profiles:
- Piecewise: T(t) constant over time intervals
- Response: first-order approach to a piecewise setpoint

Usage:
    from kindred.core.temperature import TemperatureSchedule

    intervals = [(0, 50, 298), (50, 100, 320)]
    T_piece = TemperatureSchedule.piecewise(intervals)
    temperature = T_piece(75.0)  # Returns 320

The temperature DSL admits authored times and temperatures through these
constructors; the nominal temperature arrives admitted by DSL parsing.
Evaluation receives execution times inside the admitted span.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Callable, List, Literal, Mapping, Optional, Tuple

from kindred.core.species import (
    validate_input_time_seconds,
    validate_temperature,
)

logger = logging.getLogger(__name__)

__all__ = [
    "TemperatureSchedule",
    "TemperatureScheduleError",
    "declared_temperature_step_boundaries",
    "temperature_at_execution_time",
]


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
        try:
            self.t_start = validate_input_time_seconds(self.t_start, field="interval start time")
            self.t_end = validate_input_time_seconds(self.t_end, field="interval end time")
            self.temperature = validate_temperature(self.temperature)
        except (TypeError, ValueError) as exc:
            raise TemperatureScheduleError(str(exc)) from exc
        if self.t_end <= self.t_start:
            raise TemperatureScheduleError(
                f"The end time ({self.t_end:g} s) must be greater than the start time ({self.t_start:g} s)."
            )

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
    - Piecewise constant temperature over time intervals
    - First-order response to a piecewise setpoint

    Examples
    --------
    >>> intervals = [(0, 50, 298), (50, 100, 320)]
    >>> T = TemperatureSchedule.piecewise(intervals)
    >>> T(25.0)
    298.0
    >>> T(75.0)
    320.0
    """

    def __init__(self, T_func: Callable[[float], float], schedule_type: str):
        """
        Initialize temperature schedule.

        Parameters
        ----------
        T_func : callable
            Function T(t) that returns temperature in Kelvin
        schedule_type : str
            Type of schedule: "piecewise" or "response"
        """
        self._T_func = T_func
        self.schedule_type = schedule_type
        self._intervals: List[TimeInterval] = []
        self._response_times: List[float] = []
        self._response_temperatures: List[float] = []
        self._response_tau: Optional[float] = None
        self._baseline_temperature: Optional[float] = None

    @staticmethod
    def _validated_intervals(intervals: List[Tuple[float, float, float]]) -> List[TimeInterval]:
        # The DSL builds at least one contiguous interval from its time list.
        sorted_intervals = sorted(intervals, key=lambda x: x[0])
        time_intervals: List[TimeInterval] = []
        for i, (t_start, t_end, temp) in enumerate(sorted_intervals):
            try:
                interval = TimeInterval(t_start, t_end, temp)
                time_intervals.append(interval)
            except TemperatureScheduleError as e:
                raise TemperatureScheduleError(f"Interval {i+1}: {e}")
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

    def __reduce__(self):
        """
        Make TemperatureSchedule picklable by round-tripping through `to_dict`.

        This avoids pickling closures captured by constant/piecewise schedules.
        """
        return (type(self).from_dict, (self.to_dict(),))

    @classmethod
    def piecewise(
        cls,
        intervals: List[Tuple[float, float, float]],
        *,
        baseline_temperature: float = 298.15,
    ) -> TemperatureSchedule:
        """
        Create piecewise constant temperature schedule.

        Parameters
        ----------
        intervals : list of tuple
            List of (t_start, t_end, temperature) tuples
            - t_start: interval start time (inclusive)
            - t_end: interval end time (exclusive, except last)
            - temperature: temperature in Kelvin
        baseline_temperature : float
            Nominal tab temperature used before the first authored interval.

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
        baseline = baseline_temperature

        # Create lookup function
        def T_piecewise(t: float) -> float:
            if t < time_intervals[0].t_start:
                return baseline

            # Find matching interval
            for i, interval in enumerate(time_intervals):
                # Last interval includes endpoint
                if i == len(time_intervals) - 1:
                    if interval.contains_inclusive(t):
                        return interval.temperature
                else:
                    if interval.contains(t):
                        return interval.temperature

            # Contiguous intervals leave only the tail, where the final
            # authored temperature continues.
            return time_intervals[-1].temperature

        schedule = cls(T_piecewise, schedule_type="piecewise")
        schedule._intervals = time_intervals
        schedule._baseline_temperature = baseline

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
        baseline_temperature: float = 298.15,
    ) -> TemperatureSchedule:
        """
        Create first-order response schedule driven by a step setpoint schedule.

        The setpoint is piecewise constant over [times[i], times[i+1]) with value
        temperatures[i]. The actual runtime temperature evolves as:

            dT/dt = (T_set(t) - T(t)) / tau

        The response begins from ``baseline_temperature`` at the first authored
        setpoint; it never extrapolates an authored setpoint backward in time.
        """
        try:
            tau_f = validate_input_time_seconds(tau, field="tau", allow_zero=False)
        except (TypeError, ValueError) as exc:
            raise TemperatureScheduleError(str(exc)) from exc

        # The temperature DSL admits the time and temperature counts.
        time_points = [float(t) for t in times]
        setpoints = [float(T) for T in temperatures]
        time_intervals = cls._validated_intervals(
            [
                (time_points[i], time_points[i + 1], setpoints[i])
                for i in range(len(setpoints))
            ]
        )
        baseline = baseline_temperature

        start_actuals = [baseline]
        for interval in time_intervals:
            start_actual = float(start_actuals[-1])
            duration = float(interval.t_end - interval.t_start)
            decay = math.exp(-duration / tau_f)
            end_actual = float(interval.temperature) + (start_actual - float(interval.temperature)) * decay
            start_actuals.append(float(end_actual))

        def T_response(t: float) -> float:
            if t < time_intervals[0].t_start:
                return baseline

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

            # Contiguous intervals leave only the tail.
            last_interval = time_intervals[-1]
            dt = float(t - last_interval.t_end)
            last_setpoint = float(last_interval.temperature)
            start_actual = float(start_actuals[-1])
            return last_setpoint + (start_actual - last_setpoint) * math.exp(-dt / tau_f)

        schedule = cls(T_response, schedule_type="response")
        schedule._intervals = time_intervals
        schedule._response_times = list(time_points)
        schedule._response_temperatures = list(setpoints)
        schedule._response_tau = tau_f
        schedule._baseline_temperature = baseline

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

    def to_dict(self) -> dict:
        """
        Convert schedule to dictionary (for serialization).

        Returns
        -------
        dict
            Dictionary representation
        """
        if self.schedule_type == "piecewise":
            return {
                "type": "piecewise",
                "baseline_temperature": float(self._baseline_temperature),
                "intervals": [
                    {
                        "t_start": interval.t_start,
                        "t_end": interval.t_end,
                        "temperature": interval.temperature
                    }
                    for interval in self._intervals
                ]
            }
        return {
            "type": "response",
            "times": list(self._response_times),
            "temperatures": list(self._response_temperatures),
            "tau": self._response_tau,
            "baseline_temperature": float(self._baseline_temperature),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> TemperatureSchedule:
        """Rebuild a schedule from its own ``to_dict`` payload (pickling)."""
        if data["type"] == "piecewise":
            return cls.piecewise(
                [
                    (item["t_start"], item["t_end"], item["temperature"])
                    for item in data["intervals"]
                ],
                baseline_temperature=data["baseline_temperature"],
            )
        return cls.response(
            data["times"],
            data["temperatures"],
            tau=data["tau"],
            baseline_temperature=data["baseline_temperature"],
        )

    def __repr__(self) -> str:
        """String representation."""
        n = len(self._intervals)
        if self.schedule_type == "piecewise":
            return f"TemperatureSchedule.piecewise({n} intervals)"
        return f"TemperatureSchedule.response({n} intervals, tau={self._response_tau})"


def temperature_at_execution_time(
    schedule: TemperatureSchedule | None,
    reference_temperature_K: float,
    time: float,
    *,
    side: Literal["preceding", "following"] = "following",
) -> float:
    """Resolve an execution-selected side without changing the scientific clock.

    Only declared piecewise steps have distinct sides. Response schedules
    retain their ordinary evaluation at the actual time.
    """
    if schedule is None:
        return reference_temperature_K
    if side == "preceding":
        identity = temperature_schedule_execution_identity(schedule)
        if identity["type"] == "piecewise_steps":
            temperature = float(identity["baseline_temperature"])
            for step in identity["steps"]:
                if float(step["time"]) >= time:
                    break
                temperature = float(step["temperature"])
            return temperature
    return float(schedule(time))


def declared_temperature_step_boundaries(
    schedule: TemperatureSchedule | None,
    *,
    t_start: float,
    t_end: float,
) -> tuple[float, ...]:
    """Return only authored finite piecewise-temperature discontinuities."""

    # The execution span arrives admitted with its simulation request.
    if schedule is None:
        return ()
    identity = temperature_schedule_execution_identity(schedule)
    if identity["type"] != "piecewise_steps":
        return ()
    boundaries = {
        float(item["time"])
        for item in identity["steps"]
        if t_start < float(item["time"]) <= t_end
    }
    return tuple(sorted(boundaries))


def temperature_schedule_execution_identity(
    schedule: TemperatureSchedule,
) -> Mapping[str, object]:
    """Return declared piecewise steps with authored semantic no-ops removed.

    Other schedules report only their type; callers act on piecewise steps.
    """

    if schedule.schedule_type != "piecewise":
        return {"type": schedule.schedule_type}
    baseline = float(schedule._baseline_temperature)
    steps: list[dict[str, float]] = []
    current = baseline
    for interval in schedule.get_intervals():
        temperature = float(interval.temperature)
        time = float(interval.t_start)
        if time == 0.0 and not steps:
            baseline = temperature
            current = temperature
            continue
        if temperature == current:
            continue
        steps.append({"time": time, "temperature": temperature})
        current = temperature
    if not steps:
        return {"type": "constant"}
    return {
        "type": "piecewise_steps",
        "baseline_temperature": baseline,
        "steps": tuple(steps),
    }
