from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


FitExtensionProgressPhase = Literal["profile_likelihood", "bootstrap"]
FitExtensionProgressStatus = Literal[
    "running",
    "available",
    "partial",
    "unavailable",
    "stopped",
    "failed",
]
FitExtensionProgressStage = Literal[
    "primary_complete",
    "discovery",
    "refinement",
    "replicate",
    "settlement",
]


@dataclass(frozen=True, slots=True)
class FitExtensionProgress:
    """Typed, phase-local progress for optional Fit analyses.

    This projection carries only work counters and the current Profile
    coordinate/side. It is transient callback data and deliberately has no
    optimizer objective payload or timing estimate.  Its two producers, the
    Profile and Bootstrap loops, pass literal phases, statuses and stages,
    the admitted coordinate name and their own work counters.
    """

    phase: FitExtensionProgressPhase
    status: FitExtensionProgressStatus
    stage: FitExtensionProgressStage
    coordinate: str | None = None
    side: Literal["lower", "upper"] | None = None
    discovery_step: int | None = None
    refinement_step: int | None = None
    completed_work: int = 0
    planned_work: int = 0
    completed_replicates: int | None = None
    planned_replicates: int | None = None
