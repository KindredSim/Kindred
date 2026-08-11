from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, cast

from kindred.core.simulation_failure import SimulationFailure, coerce_simulation_failure


FitDiagnosticPhase = Literal["optimizer"]
FitCompletionStatus = Literal["ok", "warn"]

_VALID_COMPLETION_STATUSES = frozenset({"ok", "warn"})
_VALID_DIAGNOSTIC_PHASES = frozenset({"optimizer"})


def _coerce_fit_diagnostic_phase(value: object, *, default_phase: FitDiagnosticPhase) -> FitDiagnosticPhase:
    phase = str(value or default_phase)
    if phase not in _VALID_DIAGNOSTIC_PHASES:
        raise ValueError(f"Unsupported fit diagnostic phase: {phase!r}")
    return cast(FitDiagnosticPhase, phase)


@dataclass
class FitDiagnostic:
    phase: FitDiagnosticPhase
    dataset_id: Optional[str]
    failure: SimulationFailure

    def __post_init__(self) -> None:
        self.phase = _coerce_fit_diagnostic_phase(self.phase, default_phase="optimizer")
        self.dataset_id = str(self.dataset_id) if self.dataset_id is not None else None
        self.failure = coerce_simulation_failure(self.failure)


@dataclass
class GlobalFitCompletion:
    status: FitCompletionStatus
    optimizer_converged: bool
    optimizer_diagnostic: Optional[FitDiagnostic] = None

    def __post_init__(self) -> None:
        status = str(self.status or "")
        if status not in _VALID_COMPLETION_STATUSES:
            raise ValueError(f"Unsupported global fit completion status: {status!r}")
        self.status = cast(FitCompletionStatus, status)
        self.optimizer_converged = bool(self.optimizer_converged)
        if (
            self.optimizer_diagnostic is not None
            and not isinstance(self.optimizer_diagnostic, FitDiagnostic)
        ):
            raise TypeError(
                "optimizer_diagnostic must be FitDiagnostic or None."
            )
        if self.status == "ok":
            if not self.optimizer_converged:
                raise ValueError("GlobalFitCompletion status='ok' requires optimizer_converged=True.")
            if self.optimizer_diagnostic is not None:
                raise ValueError("GlobalFitCompletion status='ok' forbids optimizer_diagnostic.")
            return
        if self.optimizer_converged and self.optimizer_diagnostic is None:
            raise ValueError("GlobalFitCompletion status='warn' requires a real warning source.")
