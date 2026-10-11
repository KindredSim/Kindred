"""Scientific vessel-volume facts, distinct from modeled species and units."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kindred.core.exceptions import ErrorContext, KindredError


VOLUME_INPUT_MIN = 1e-30
VOLUME_INPUT_MAX = 1e30


def volume_input_in_range(value: float, *, allow_zero: bool = False) -> bool:
    """Admit a supplied volume or inflow rate, not the evolving vessel state."""
    return (allow_zero and value == 0.0) or VOLUME_INPUT_MIN <= value <= VOLUME_INPUT_MAX


class PhysicalDomainError(KindredError):
    """A successfully evaluated quantity violates an executed physical predicate."""

    def __init__(self, message: str, *, time: float, declaration: int, line: tuple[int, str], **operands):
        # The text states the refusal at its time on the declaration line; operands stay structured.
        super().__init__(f"{message} at t = {time:g} s.", context=ErrorContext(*line),
                         details={"time": time, "declaration_ordinal": declaration, **operands})


@dataclass(frozen=True, slots=True)
class SimulationBoundaryState:
    concentrations: tuple[tuple[str, float], ...]
    volume: float | None
    temperature_K: float


@dataclass(frozen=True, slots=True)
class VolumeTrajectory:
    """Defined suffix of canonical simulation rows, in scientific cubic metres."""

    first_index: int
    values: np.ndarray

    def __post_init__(self):
        # The volume executor range-checks every recorded volume.
        values = np.asarray(self.values, dtype=float)
        values.setflags(write=False)
        object.__setattr__(self, "values", values)
