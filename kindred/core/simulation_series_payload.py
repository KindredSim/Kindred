from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True, slots=True)
class SimulationSeriesPayload:
    """Typed payload for fitting-oriented simulation series output."""

    t: np.ndarray
    species: dict[str, np.ndarray]
    algebra_scalars: dict[str, float] = field(default_factory=dict)
