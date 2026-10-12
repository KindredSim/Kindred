from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


FitExtensionProgressStatus = Literal[
    "running",
    "available",
    "partial",
    "unavailable",
    "failed",
]
FitExtensionProgressStage = Literal[
    "primary_complete",
    "discovery",
    "refinement",
    "settlement",
]


@dataclass(frozen=True, slots=True)
class FitExtensionProgress:
    """Typed progress for the post-fit stage and the optional Profile analysis.

    This projection carries only work counters and the current Profile
    coordinate/side. It is transient callback data and deliberately has no
    optimizer objective payload or timing estimate.  Global Fit composition
    sends "primary_complete" when the post-fit stage starts; the Profile loop
    passes literal statuses and stages, the admitted coordinate name and its
    own work counters.
    """

    status: FitExtensionProgressStatus
    stage: FitExtensionProgressStage
    coordinate: str | None = None
    side: Literal["lower", "upper"] | None = None
    discovery_step: int | None = None
    refinement_step: int | None = None
    completed_work: int = 0
    planned_work: int = 0
