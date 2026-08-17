from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Mapping, Optional, cast

from kindred.core.simulation_failure import (
    FactualCause,
)


FitDiagnosticPhase = Literal["optimizer"]
FitCompletionStatus = Literal["ok", "warn"]
FitPrimaryModel = Literal["ordinary", "centred_fixed_effects"]
FitPrimaryClassification = Literal[
    "converged",
    "nonconverged",
    "cancelled",
    "fatal",
]
FitPolicyPublication = Literal["allowed", "withheld"]
FitPolicyApply = Literal["eligible", "ineligible"]
FitTerminalExtensionKind = Literal["profile_likelihood", "bootstrap"]
FitTerminalExtensionStatus = Literal[
    "available",
    "partial",
    "unavailable",
    "stopped",
    "failed",
]
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

_VALID_COMPLETION_STATUSES = frozenset({"ok", "warn"})
_VALID_DIAGNOSTIC_PHASES = frozenset({"optimizer"})
_VALID_EXTENSION_KINDS = frozenset({"profile_likelihood", "bootstrap"})
_VALID_EXTENSION_STATUSES = frozenset(
    {"available", "partial", "unavailable", "stopped", "failed"}
)
_VALID_PRIMARY_CLASSIFICATIONS = frozenset(
    {
        "converged",
        "nonconverged",
        "cancelled",
        "fatal",
    }
)
_VALID_PRIMARY_MODELS = frozenset({"ordinary", "centred_fixed_effects"})


def _freeze_value(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {name: _freeze_value(item) for name, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(_freeze_value(item) for item in sorted(value, key=repr))
    return value


def _coerce_fit_diagnostic_phase(value: object, *, default_phase: FitDiagnosticPhase) -> FitDiagnosticPhase:
    phase = str(value or default_phase)
    if phase not in _VALID_DIAGNOSTIC_PHASES:
        raise ValueError(f"Unsupported fit diagnostic phase: {phase!r}")
    return cast(FitDiagnosticPhase, phase)


@dataclass(frozen=True, slots=True)
class FitDiagnostic:
    phase: FitDiagnosticPhase
    dataset_id: Optional[str]
    failure: FactualCause

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "phase",
            _coerce_fit_diagnostic_phase(self.phase, default_phase="optimizer"),
        )
        object.__setattr__(
            self,
            "dataset_id",
            str(self.dataset_id) if self.dataset_id is not None else None,
        )
        if not isinstance(self.failure, FactualCause):
            raise TypeError("Fit diagnostic failure must be a FactualCause.")


@dataclass(frozen=True, slots=True)
class FitTerminalExtension:
    """Typed attachment point for named optional-analysis truth.

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
    cause: FactualCause | None = None

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
        raw_cause = self.cause
        if status in {"stopped", "failed"} and raw_cause is None:
            raise ValueError(
                f"A {status} Fit terminal extension requires a typed cause."
            )
        if raw_cause is not None and not isinstance(raw_cause, FactualCause):
            raise TypeError("Fit terminal extension cause must be a FactualCause.")
        object.__setattr__(self, "kind", cast(FitTerminalExtensionKind, kind))
        object.__setattr__(self, "status", cast(FitTerminalExtensionStatus, status))
        object.__setattr__(self, "method", method)
        object.__setattr__(self, "provenance", _freeze_value(provenance))
        object.__setattr__(self, "details", _freeze_value(dict(self.details or {})))
        object.__setattr__(self, "unavailable_reason", reason or None)
        object.__setattr__(
            self,
            "cause",
            raw_cause,
        )

    @property
    def classification(self) -> FitTerminalExtensionStatus:
        """Expose optional-analysis classification without policy inference elsewhere."""

        return self.status

    @property
    def publication_allowed(self) -> bool:
        """Only completed optional evidence is publishable as that optional."""

        return self.status in {"available", "partial"}

    @property
    def apply_allowed(self) -> bool:
        """Optional analyses never grant primary Fit Apply permission."""

        return False


@dataclass(frozen=True, slots=True)
class FitExtensionProgress:
    """Typed, phase-local progress for bounded terminal Fit analyses.

    This projection carries only work counters and the current Profile
    coordinate/side.  It deliberately has no optimizer objective payload or
    timing estimate, so terminal analysis cannot masquerade as primary Fit
    progress or trigger nested objective presentation.
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


@dataclass(frozen=True, slots=True)
class FitPrimaryOutcome:
    """Immutable primary Fit classification plus its publication/Apply policy."""

    classification: FitPrimaryClassification
    publication: FitPolicyPublication
    apply: FitPolicyApply
    details: Mapping[str, object] = MappingProxyType({})
    cause: FactualCause | None = None

    def __post_init__(self) -> None:
        classification = str(self.classification or "")
        publication = str(self.publication or "")
        apply = str(self.apply or "")
        if classification not in _VALID_PRIMARY_CLASSIFICATIONS:
            raise ValueError(f"Unsupported primary Fit classification: {classification!r}")
        if publication not in {"allowed", "withheld"}:
            raise ValueError("Primary Fit publication policy is invalid.")
        if apply not in {"eligible", "ineligible"}:
            raise ValueError("Primary Fit Apply policy is invalid.")
        if apply == "eligible" and publication != "allowed":
            raise ValueError("Primary Fit Apply cannot be eligible when publication is withheld.")
        raw_cause = self.cause
        if (
            classification in {"cancelled", "fatal"}
            and raw_cause is None
        ):
            raise ValueError(
                f"Primary Fit classification {classification!r} requires a factual cause."
            )
        if raw_cause is not None and not isinstance(raw_cause, FactualCause):
            raise TypeError("Primary Fit outcome cause must be a FactualCause.")
        object.__setattr__(self, "classification", cast(FitPrimaryClassification, classification))
        object.__setattr__(self, "publication", cast(FitPolicyPublication, publication))
        object.__setattr__(self, "apply", cast(FitPolicyApply, apply))
        object.__setattr__(self, "details", _freeze_value(dict(self.details or {})))
        object.__setattr__(self, "cause", raw_cause)

    @property
    def publication_allowed(self) -> bool:
        return self.publication == "allowed"

    @property
    def apply_allowed(self) -> bool:
        return self.apply == "eligible"


@dataclass(frozen=True, slots=True)
class GlobalFitCompletion:
    status: FitCompletionStatus
    optimizer_converged: bool
    optimizer_diagnostic: Optional[FitDiagnostic] = None
    terminal_extensions: tuple[FitTerminalExtension, ...] = ()
    primary_model: FitPrimaryModel = "ordinary"
    primary_model_details: Mapping[str, object] = MappingProxyType({})
    primary_outcome: FitPrimaryOutcome | None = None

    def __post_init__(self) -> None:
        status = str(self.status or "")
        if status not in _VALID_COMPLETION_STATUSES:
            raise ValueError(f"Unsupported global fit completion status: {status!r}")
        object.__setattr__(self, "status", cast(FitCompletionStatus, status))
        object.__setattr__(self, "optimizer_converged", bool(self.optimizer_converged))
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
        object.__setattr__(self, "terminal_extensions", extensions)
        primary_model = str(self.primary_model or "")
        if primary_model not in _VALID_PRIMARY_MODELS:
            raise ValueError(f"Unsupported primary Fit model: {primary_model!r}")
        primary_model_details = dict(self.primary_model_details or {})
        if primary_model == "centred_fixed_effects":
            if primary_model_details.get("scope") != "centred_fixed_effects":
                raise ValueError(
                    "Centred primary Fit completion requires centred scope details."
                )
        object.__setattr__(self, "primary_model", cast(FitPrimaryModel, primary_model))
        object.__setattr__(
            self,
            "primary_model_details",
            _freeze_value(primary_model_details),
        )
        primary_outcome = self.primary_outcome
        if primary_outcome is not None and not isinstance(primary_outcome, FitPrimaryOutcome):
            raise TypeError("primary_outcome must be FitPrimaryOutcome or None.")
        if self.status == "ok":
            if not self.optimizer_converged:
                raise ValueError("GlobalFitCompletion status='ok' requires optimizer_converged=True.")
            if self.optimizer_diagnostic is not None:
                raise ValueError("GlobalFitCompletion status='ok' forbids optimizer_diagnostic.")
        elif self.optimizer_converged and self.optimizer_diagnostic is None:
            raise ValueError("GlobalFitCompletion status='warn' requires a real warning source.")
        if primary_outcome is None:
            diagnostic_cause = (
                None
                if self.optimizer_diagnostic is None
                else self.optimizer_diagnostic.failure
            )
            primary_outcome = FitPrimaryOutcome(
                classification=(
                    "converged" if self.optimizer_converged else "nonconverged"
                ),
                publication="allowed",
                apply=("eligible" if self.optimizer_converged else "ineligible"),
                details={"completion_status": status},
                cause=diagnostic_cause,
            )
        object.__setattr__(self, "primary_outcome", primary_outcome)
