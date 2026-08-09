from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Literal, Optional, cast

from kindred.core.simulation_failure import SimulationFailure, coerce_simulation_failure


FitDiagnosticPhase = Literal["optimizer", "final_replay", "fatal"]
FitCompletionStatus = Literal["ok", "warn", "fail"]

_VALID_COMPLETION_STATUSES = frozenset({"ok", "warn", "fail"})
_VALID_DIAGNOSTIC_PHASES = frozenset({"optimizer", "final_replay", "fatal"})


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
    nonfinite_metrics: bool
    optimizer_diagnostic: Optional[FitDiagnostic] = None
    dataset_failures: Dict[str, FitDiagnostic] = field(default_factory=dict)

    def __post_init__(self) -> None:
        status = str(self.status or "")
        if status not in _VALID_COMPLETION_STATUSES:
            raise ValueError(f"Unsupported global fit completion status: {status!r}")
        self.status = cast(FitCompletionStatus, status)
        self.optimizer_converged = bool(self.optimizer_converged)
        self.nonfinite_metrics = bool(self.nonfinite_metrics)
        if (
            self.optimizer_diagnostic is not None
            and not isinstance(self.optimizer_diagnostic, FitDiagnostic)
        ):
            raise TypeError(
                "optimizer_diagnostic must be FitDiagnostic or None."
            )
        typed_failures: Dict[str, FitDiagnostic] = {}
        for ds_id, diagnostic in dict(self.dataset_failures).items():
            if not isinstance(diagnostic, FitDiagnostic):
                raise TypeError(
                    "dataset_failures must contain FitDiagnostic values."
                )
            dataset_id = str(ds_id)
            if (
                diagnostic.dataset_id is not None
                and diagnostic.dataset_id != dataset_id
            ):
                raise ValueError(
                    "Dataset failure key conflicts with diagnostic identity."
                )
            typed_failures[dataset_id] = diagnostic
        self.dataset_failures = typed_failures
        if self.nonfinite_metrics and self.status != "fail":
            raise ValueError("GlobalFitCompletion.nonfinite_metrics=True requires status='fail'.")
        if self.status == "ok":
            if not self.optimizer_converged:
                raise ValueError("GlobalFitCompletion status='ok' requires optimizer_converged=True.")
            if self.nonfinite_metrics:
                raise ValueError("GlobalFitCompletion status='ok' requires nonfinite_metrics=False.")
            if self.dataset_failures:
                raise ValueError("GlobalFitCompletion status='ok' forbids dataset_failures.")
            if self.optimizer_diagnostic is not None:
                raise ValueError("GlobalFitCompletion status='ok' forbids optimizer_diagnostic.")
            return
        if self.status == "warn":
            if self.nonfinite_metrics:
                raise ValueError("GlobalFitCompletion status='warn' requires nonfinite_metrics=False.")
            if self.dataset_failures:
                raise ValueError("GlobalFitCompletion status='warn' forbids dataset_failures.")
            if self.optimizer_converged and self.optimizer_diagnostic is None:
                raise ValueError("GlobalFitCompletion status='warn' requires a real warning source.")
            return
        if not self.dataset_failures and self.optimizer_diagnostic is None:
            raise ValueError("GlobalFitCompletion status='fail' requires a real failure source.")
