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

import logging
import math
from dataclasses import dataclass
from typing import Callable, List, Literal, Mapping, Optional, Tuple

from kindred.core.species import (
    validate_input_time_seconds,
    validate_temperature,
    validate_time_seconds,
)

logger = logging.getLogger(__name__)

__all__ = [
    "TemperatureSchedule",
    "TemperatureScheduleError",
    "coerce_temperature_kelvin",
    "declared_temperature_step_boundaries",
    "temperature_at_execution_time",
    "temperature_schedule_execution_identity",
]


def coerce_temperature_kelvin(value: object, *, field: str = "temperature") -> float:
    try:
        return validate_temperature(value, field=field)
    except (TypeError, ValueError) as exc:
        raise TemperatureScheduleError(str(exc)) from exc


def _coerce_nonnegative_time(value: object, *, field: str) -> float:
    try:
        return validate_time_seconds(value, field=field)
    except (TypeError, ValueError) as exc:
        raise TemperatureScheduleError(str(exc)) from exc


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
        except (TypeError, ValueError) as exc:
            raise TemperatureScheduleError(str(exc)) from exc
        self.temperature = coerce_temperature_kelvin(self.temperature)
        if self.t_end <= self.t_start:
            raise TemperatureScheduleError(
                f"Interval end time ({self.t_end}) must be greater than start time ({self.t_start})"
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
        self._baseline_temperature: Optional[float] = None

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
        time_value = _coerce_nonnegative_time(t, field="temperature schedule time")
        return coerce_temperature_kelvin(self._T_func(time_value), field="temperature schedule output")

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
        temperature_f = coerce_temperature_kelvin(temperature)

        def T_const(t: float) -> float:
            return temperature_f

        schedule = cls(T_const, schedule_type="constant")
        logger.debug(f"Created constant temperature schedule: T = {temperature_f} K")
        return schedule

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
        baseline = coerce_temperature_kelvin(
            baseline_temperature,
            field="baseline temperature",
        )

        # Create lookup function
        def T_piecewise(t: float) -> float:
            if t < 0:
                raise ValueError(f"Time must be non-negative, got t={t}")
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

            # The final authored temperature defines the schedule's tail.
            last_interval = time_intervals[-1]
            if t >= last_interval.t_end:
                return last_interval.temperature

            # Should never reach here if validation is correct
            raise TemperatureScheduleError(f"No interval found for t={t}")

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
        baseline = coerce_temperature_kelvin(
            baseline_temperature,
            field="baseline temperature",
        )

        start_actuals = [baseline]
        for interval in time_intervals:
            start_actual = float(start_actuals[-1])
            duration = float(interval.t_end - interval.t_start)
            decay = math.exp(-duration / tau_f)
            end_actual = float(interval.temperature) + (start_actual - float(interval.temperature)) * decay
            start_actuals.append(float(end_actual))

        def T_response(t: float) -> float:
            if t < 0:
                raise ValueError(f"Time must be non-negative, got t={t}")
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
        if self.schedule_type == "response":
            return {
                "type": "response",
                "times": list(self._response_times),
                "temperatures": list(self._response_temperatures),
                "tau": float(self._response_tau if self._response_tau is not None else 0.0),
                "baseline_temperature": float(self._baseline_temperature),
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
        if not isinstance(data, Mapping):
            raise TemperatureScheduleError("temperature schedule must be a mapping.")
        if "type" not in data:
            raise TemperatureScheduleError("temperature schedule must include a type.")
        schedule_type = str(data["type"])

        if schedule_type == "constant":
            if "temperature" not in data:
                raise TemperatureScheduleError("constant temperature schedule must include temperature.")
            return cls.constant(data["temperature"])

        elif schedule_type == "piecewise":
            missing = [
                key
                for key in ("intervals", "baseline_temperature")
                if key not in data
            ]
            if missing:
                raise TemperatureScheduleError(
                    "piecewise temperature schedule missing required field(s): "
                    + ", ".join(missing)
                )
            intervals_data = data["intervals"]
            intervals = [
                (item["t_start"], item["t_end"], item["temperature"])
                for item in intervals_data
            ]
            return cls.piecewise(
                intervals,
                baseline_temperature=data["baseline_temperature"],
            )
        elif schedule_type == "response":
            missing = [
                key
                for key in (
                    "times",
                    "temperatures",
                    "tau",
                    "baseline_temperature",
                )
                if key not in data
            ]
            if missing:
                raise TemperatureScheduleError(
                    f"response temperature schedule missing required field(s): {', '.join(missing)}"
                )
            times = data["times"]
            temperatures = data["temperatures"]
            tau = data["tau"]
            return cls.response(
                times,
                temperatures,
                tau=float(tau),
                baseline_temperature=data["baseline_temperature"],
            )

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

    try:
        start = validate_time_seconds(t_start, field="temperature boundary start")
        end = validate_time_seconds(t_end, field="temperature boundary end")
    except (TypeError, ValueError) as exc:
        raise TemperatureScheduleError(str(exc)) from exc
    if end <= start:
        raise TemperatureScheduleError(
            "temperature boundary end must be greater than start."
        )
    if schedule is None:
        return ()
    identity = temperature_schedule_execution_identity(schedule)
    if identity["type"] != "piecewise_steps":
        return ()
    boundaries = {
        float(item["time"])
        for item in identity["steps"]
        if start < float(item["time"]) <= end
    }
    return tuple(sorted(boundaries))


def temperature_schedule_execution_identity(
    schedule: TemperatureSchedule | None,
) -> Mapping[str, object] | None:
    """Return exact runtime identity with authored semantic no-ops removed."""

    if schedule is None:
        return None
    decoded = schedule
    if decoded.schedule_type == "constant":
        return {
            "type": "constant",
            "temperature": float(decoded(0.0)),
        }
    if decoded.schedule_type == "piecewise":
        intervals = decoded.get_intervals()
        baseline = float(decoded._baseline_temperature)
        steps: list[dict[str, float]] = []
        current = baseline
        for interval in intervals:
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
            return {
                "type": "constant",
                "temperature": baseline,
            }
        return {
            "type": "piecewise_steps",
            "baseline_temperature": baseline,
            "steps": tuple(steps),
        }
    if decoded.schedule_type == "response":
        baseline = float(decoded._baseline_temperature)
        times = tuple(float(value) for value in decoded._response_times)
        temperatures = tuple(
            float(value) for value in decoded._response_temperatures
        )
        changes: list[dict[str, float]] = []
        current = baseline
        for index, temperature in enumerate(temperatures):
            if temperature == current:
                continue
            changes.append(
                {
                    "time": times[index],
                    "temperature": temperature,
                }
            )
            current = temperature
        if not changes:
            return {
                "type": "constant",
                "temperature": baseline,
            }
        return {
            "type": "response_steps",
            "baseline_temperature": baseline,
            "changes": tuple(changes),
            "tau": float(decoded._response_tau),
        }
