from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Mapping, Optional, cast

from kindred.core.simulation_failure import SimulationFailure, coerce_simulation_failure


FitDiagnosticPhase = Literal["optimizer"]
FitCompletionStatus = Literal["ok", "warn"]
FitTerminalExtensionKind = Literal["profile_likelihood", "bootstrap", "hierarchical"]
FitTerminalExtensionStatus = Literal["available", "partial", "unavailable"]

_VALID_COMPLETION_STATUSES = frozenset({"ok", "warn"})
_VALID_DIAGNOSTIC_PHASES = frozenset({"optimizer"})
_VALID_EXTENSION_KINDS = frozenset(
    {"profile_likelihood", "bootstrap", "hierarchical"}
)
_VALID_EXTENSION_STATUSES = frozenset({"available", "partial", "unavailable"})


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


@dataclass(frozen=True, slots=True)
class FitTerminalExtension:
    """Typed attachment point for named uncertainty/hierarchical truth.

    The numerical engines own the immutable ``details`` schema for their
    method.  This common envelope prevents a report consumer from presenting
    an unlabelled or provenance-free result.
    """

    kind: FitTerminalExtensionKind
    status: FitTerminalExtensionStatus
    method: str
    provenance: Mapping[str, object]
    details: Mapping[str, object]
    unavailable_reason: Optional[str] = None

    def __post_init__(self) -> None:
        kind = str(self.kind or "")
        status = str(self.status or "")
        method = str(self.method or "").strip()
        if kind not in _VALID_EXTENSION_KINDS:
            raise ValueError(f"Unsupported Fit terminal extension kind: {kind!r}")
        if status not in _VALID_EXTENSION_STATUSES:
            raise ValueError(f"Unsupported Fit terminal extension status: {status!r}")
        if not method:
            raise ValueError("A Fit terminal extension requires a named method.")
        provenance = dict(self.provenance or {})
        if not provenance:
            raise ValueError("A Fit terminal extension requires provenance.")
        reason = (
            str(self.unavailable_reason or "").strip()
            if self.unavailable_reason is not None
            else ""
        )
        if status == "unavailable" and not reason:
            raise ValueError("An unavailable Fit terminal extension requires a reason.")
        if status != "unavailable" and reason:
            raise ValueError(
                "Only an unavailable Fit terminal extension may carry an unavailable reason."
            )
        object.__setattr__(self, "kind", cast(FitTerminalExtensionKind, kind))
        object.__setattr__(self, "status", cast(FitTerminalExtensionStatus, status))
        object.__setattr__(self, "method", method)
        object.__setattr__(self, "provenance", MappingProxyType(provenance))
        object.__setattr__(self, "details", MappingProxyType(dict(self.details or {})))
        object.__setattr__(self, "unavailable_reason", reason or None)


@dataclass
class GlobalFitCompletion:
    status: FitCompletionStatus
    optimizer_converged: bool
    optimizer_diagnostic: Optional[FitDiagnostic] = None
    terminal_extensions: tuple[FitTerminalExtension, ...] = ()

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
        extensions = tuple(self.terminal_extensions or ())
        if any(not isinstance(item, FitTerminalExtension) for item in extensions):
            raise TypeError(
                "terminal_extensions must contain FitTerminalExtension values."
            )
        self.terminal_extensions = extensions
        if self.status == "ok":
            if not self.optimizer_converged:
                raise ValueError("GlobalFitCompletion status='ok' requires optimizer_converged=True.")
            if self.optimizer_diagnostic is not None:
                raise ValueError("GlobalFitCompletion status='ok' forbids optimizer_diagnostic.")
            return
        if self.optimizer_converged and self.optimizer_diagnostic is None:
            raise ValueError("GlobalFitCompletion status='warn' requires a real warning source.")
