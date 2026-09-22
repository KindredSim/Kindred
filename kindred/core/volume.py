"""Scientific vessel-volume facts, distinct from modeled species and units."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from kindred.core.exceptions import KindredError


VOLUME_INPUT_MIN = 1e-30
VOLUME_INPUT_MAX = 1e30


def volume_input_in_range(value: float, *, allow_zero: bool = False) -> bool:
    """Admit a supplied volume or inflow rate, not the evolving vessel state."""
    return (allow_zero and value == 0.0) or VOLUME_INPUT_MIN <= value <= VOLUME_INPUT_MAX


class PhysicalDomainError(KindredError):
    """A successfully evaluated quantity violates an executed physical predicate."""

    def __init__(self, message: str, *, time: float, declaration: int, **operands):
        super().__init__(message, details={"time": time, "declaration_ordinal": declaration,
                                         **operands})


@dataclass(frozen=True, slots=True)
class SimulationBoundaryState:
    concentrations: tuple[tuple[str, float], ...]
    volume: float | None
    temperature_K: float

    def __post_init__(self):
        if self.volume is not None and (not math.isfinite(self.volume) or self.volume <= 0):
            raise ValueError("Boundary vessel volume must be positive and finite.")
        if not math.isfinite(self.temperature_K) or self.temperature_K <= 0:
            raise ValueError("Executed boundary temperature must be positive and finite.")


@dataclass(frozen=True, slots=True)
class VolumeTrajectory:
    """Defined suffix of canonical simulation rows, in scientific cubic metres."""

    first_index: int
    values: np.ndarray

    def __post_init__(self):
        values = np.asarray(self.values, dtype=float)
        if self.first_index < 0 or values.ndim != 1 or not values.size:
            raise ValueError("Volume trajectory requires a nonempty defined suffix.")
        if np.any(~np.isfinite(values)) or np.any(values <= 0):
            raise ValueError("Tracked volume must be positive and finite.")
        values.setflags(write=False)
        object.__setattr__(self, "values", values)
