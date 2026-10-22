"""
Temperature changes during a Run.

A ``temp_step`` or ``temp_response`` line lowers to a sequence of
``TemperatureChange`` values.  The intervention schedule executor applies each
change at its time, like any other intervention, and carries the temperature
between changes as a ``TemperatureSegment``: constant after a step, or a
first-order approach toward the change's target,

    T(t) = target + (T(start) - target) * exp(-(t - start) / tau)

Each solver segment therefore uses exactly the temperature curve it started
with; a change takes effect at the boundary that executes it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

__all__ = [
    "TemperatureChange",
    "TemperatureSegment",
    "starting_temperature_segment",
]


@dataclass(frozen=True, slots=True)
class TemperatureChange:
    """A step to ``temperature`` at ``time``, or, with ``tau``, a first-order
    approach toward it from the temperature at ``time``."""

    time: float
    temperature: float
    tau: float | None = None


@dataclass(frozen=True, slots=True)
class TemperatureSegment:
    """The temperature from ``start`` until the next change."""

    start: float
    temperature: float
    target: float
    tau: float | None = None

    def __call__(self, time: float) -> float:
        if self.tau is None:
            return self.temperature
        return self.target + (self.temperature - self.target) * math.exp(
            -(time - self.start) / self.tau
        )

    def values(self, times: np.ndarray) -> np.ndarray:
        """The temperature at each of ``times``, each worked out as one call."""
        if self.tau is None:
            return np.full(len(times), self.temperature)
        return np.fromiter(map(self, times.tolist()), float, len(times))

    def after(self, change: TemperatureChange) -> TemperatureSegment:
        """The segment that a change executed at its time starts."""
        temperature = change.temperature if change.tau is None else self(change.time)
        return TemperatureSegment(change.time, temperature, change.temperature, change.tau)


def starting_temperature_segment(
    changes: Sequence[TemperatureChange],
    nominal_temperature_K: float,
    t_start: float,
) -> TemperatureSegment:
    """The nominal temperature until the first change, with every change at or
    before ``t_start`` already executed."""
    segment = TemperatureSegment(t_start, nominal_temperature_K, nominal_temperature_K)
    for change in changes:
        if change.time > t_start:
            break
        segment = segment.after(change)
    return segment
