from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from kindred.core.exceptions import SolverError


@dataclass(frozen=True, slots=True)
class NonnegativeProjection:
    projected: bool = False
    count: int = 0
    max_negative_magnitude: float = 0.0

    def __post_init__(self) -> None:
        count = int(self.count)
        magnitude = float(self.max_negative_magnitude)
        if count < 0:
            raise ValueError("Nonnegative projection count must be nonnegative.")
        if not math.isfinite(magnitude) or magnitude < 0.0:
            raise ValueError(
                "Nonnegative projection magnitude must be finite and nonnegative."
            )
        if bool(self.projected) != bool(count):
            raise ValueError(
                "Nonnegative projection flag must agree with its projected count."
            )
        if count == 0 and magnitude != 0.0:
            raise ValueError(
                "An empty nonnegative projection cannot report a magnitude."
            )
        object.__setattr__(self, "projected", bool(count))
        object.__setattr__(self, "count", count)
        object.__setattr__(self, "max_negative_magnitude", magnitude)

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
        raise SolverError(f"{context} trajectory contains non-finite concentration values.")
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
