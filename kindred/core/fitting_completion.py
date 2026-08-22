from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast


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
    optimizer objective payload or timing estimate.
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

    def __post_init__(self) -> None:
        phase = str(self.phase or "")
        status = str(self.status or "")
        stage = str(self.stage or "")
        if phase not in {"profile_likelihood", "bootstrap"}:
            raise ValueError(f"Unsupported Fit extension progress phase: {phase!r}")
        if status not in {
            "running",
            "available",
            "partial",
            "unavailable",
            "stopped",
            "failed",
        }:
            raise ValueError(f"Unsupported Fit extension progress status: {status!r}")
        if stage not in {
            "primary_complete",
            "discovery",
            "refinement",
            "replicate",
            "settlement",
        }:
            raise ValueError(f"Unsupported Fit extension progress stage: {stage!r}")
        if self.side not in {None, "lower", "upper"}:
            raise ValueError("Fit Profile progress side must be lower, upper, or None.")
        coordinate = None if self.coordinate is None else str(self.coordinate).strip()
        if coordinate == "":
            coordinate = None
        completed = int(self.completed_work)
        planned = int(self.planned_work)
        if completed < 0 or planned < 0 or completed > planned:
            raise ValueError("Fit extension progress work counters are invalid.")
        discovery = (
            None if self.discovery_step is None else int(self.discovery_step)
        )
        refinement = (
            None if self.refinement_step is None else int(self.refinement_step)
        )
        if discovery is not None and discovery <= 0:
            raise ValueError("Profile discovery steps are one-based positive integers.")
        if refinement is not None and refinement <= 0:
            raise ValueError("Profile refinement steps are one-based positive integers.")
        completed_replicates = (
            None
            if self.completed_replicates is None
            else int(self.completed_replicates)
        )
        planned_replicates = (
            None
            if self.planned_replicates is None
            else int(self.planned_replicates)
        )
        if (
            completed_replicates is not None
            and completed_replicates < 0
            or planned_replicates is not None
            and planned_replicates < 0
            or completed_replicates is not None
            and planned_replicates is not None
            and completed_replicates > planned_replicates
        ):
            raise ValueError("Bootstrap replicate progress counters are invalid.")
        if phase == "profile_likelihood" and (
            completed_replicates is not None or planned_replicates is not None
        ):
            raise ValueError("Profile progress cannot carry Bootstrap replicate counters.")
        if phase == "bootstrap" and (
            discovery is not None or refinement is not None or self.side is not None
        ):
            raise ValueError("Bootstrap progress cannot carry Profile step fields.")
        object.__setattr__(self, "phase", cast(FitExtensionProgressPhase, phase))
        object.__setattr__(self, "status", cast(FitExtensionProgressStatus, status))
        object.__setattr__(self, "stage", cast(FitExtensionProgressStage, stage))
        object.__setattr__(self, "coordinate", coordinate)
        object.__setattr__(self, "completed_work", completed)
        object.__setattr__(self, "planned_work", planned)
        object.__setattr__(self, "discovery_step", discovery)
        object.__setattr__(self, "refinement_step", refinement)
        object.__setattr__(self, "completed_replicates", completed_replicates)
        object.__setattr__(self, "planned_replicates", planned_replicates)
