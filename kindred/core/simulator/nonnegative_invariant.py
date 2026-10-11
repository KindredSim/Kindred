from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from kindred.core.simulator.solver_types import (
    SimulationScientificUnavailability,
    _SimulationScientificUnavailabilitySignal,
)


@dataclass(frozen=True, slots=True)
class NonnegativeProjection:
    projected: bool = False
    count: int = 0
    max_negative_magnitude: float = 0.0

    @classmethod
    def aggregate(
        cls,
        projections: Iterable["NonnegativeProjection"],
    ) -> "NonnegativeProjection":
        values = tuple(projections)
        return cls(
            projected=any(value.projected for value in values),
            count=sum(value.count for value in values),
            max_negative_magnitude=max(
                (value.max_negative_magnitude for value in values),
                default=0.0,
            ),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "projected": self.projected,
            "count": self.count,
            "max_negative_magnitude": self.max_negative_magnitude,
        }


def project_nonnegative_trajectory(
    Y: np.ndarray,
    *,
    context: str,
) -> tuple[np.ndarray, NonnegativeProjection]:
    Y2 = np.asarray(Y, float).copy()
    if Y2.size == 0:
        return Y2, NonnegativeProjection()
    if not np.all(np.isfinite(Y2)):
        raise _SimulationScientificUnavailabilitySignal(
            SimulationScientificUnavailability(
                producer="nonnegative_invariant",
                stage="trajectory_arithmetic",
                details={"context": context},
            )
        )
    negative_mask = Y2 < 0.0
    count = int(np.count_nonzero(negative_mask))
    max_negative_magnitude = 0.0
    if count:
        max_negative_magnitude = float(np.max(np.abs(Y2[negative_mask])))
        Y2[negative_mask] = 0.0
    return Y2, NonnegativeProjection(
        projected=bool(count),
        count=count,
        max_negative_magnitude=max_negative_magnitude,
    )
