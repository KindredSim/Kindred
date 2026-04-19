from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Mapping, Optional, cast

from kindred.core.simulation_failure import SimulationFailure, coerce_simulation_failure


FitDiagnosticPhase = Literal["optimizer", "final_replay", "fatal"]
FitDiagnosticRemediation = Literal["x_axis_mapping", "nonfinite_metrics", "preparation", "generic_retry"]
FitCompletionStatus = Literal["ok", "warn", "fail"]

_VALID_COMPLETION_STATUSES = frozenset({"ok", "warn", "fail"})
_VALID_DIAGNOSTIC_PHASES = frozenset({"optimizer", "final_replay", "fatal"})
_VALID_DIAGNOSTIC_REMEDIATIONS = frozenset(
    {"x_axis_mapping", "nonfinite_metrics", "preparation", "generic_retry"}
)


def _copy_parameter_snapshot(snapshot: Optional[Dict[str, float]]) -> Optional[Dict[str, float]]:
    if not snapshot:
        return None
    return {str(name): float(value) for name, value in dict(snapshot).items()}


def infer_fit_diagnostic_remediation(failure: object) -> FitDiagnosticRemediation:
    payload = coerce_simulation_failure(failure)
    kind = str(payload.get("kind") or "").strip().lower()
    details = payload.get("details") if isinstance(payload.get("details"), Mapping) else {}
    stage = str(details.get("stage") or "").strip().lower()
    message = str(payload.get("message") or "").strip().lower()

    if kind == "preparation_error" or stage in {
        "parse",
        "prepared_payload",
        "solver_config",
        "parameter_algebra",
        "ode_build",
        "temperature_schedule",
        "initials_for_algebra",
        "execution_request",
    }:
        return "preparation"
    if any(
        token in message
        for token in (
            "adjust t_min/t_max",
            "sampled window",
            "outside model range",
            "strictly monotone",
            "x(t)",
            "x_obs",
            "x series",
            "x axis",
            "x-axis",
            "x mapping",
        )
    ):
        return "x_axis_mapping"
    if (
        "non-finite" in message or "nonfinite" in message
    ) and any(token in message for token in ("chi", "χ²", "metrics", "results are invalid")):
        return "nonfinite_metrics"
    return "generic_retry"


def _coerce_fit_diagnostic_phase(value: object, *, default_phase: FitDiagnosticPhase) -> FitDiagnosticPhase:
    phase = str(value or default_phase)
    if phase not in _VALID_DIAGNOSTIC_PHASES:
        raise ValueError(f"Unsupported fit diagnostic phase: {phase!r}")
    return cast(FitDiagnosticPhase, phase)


def _coerce_fit_diagnostic_remediation(
    value: object,
    *,
    failure: SimulationFailure,
) -> FitDiagnosticRemediation:
    if value is None or str(value).strip() == "":
        return infer_fit_diagnostic_remediation(failure)
    remediation = str(value).strip()
    if remediation not in _VALID_DIAGNOSTIC_REMEDIATIONS:
        raise ValueError(f"Unsupported fit diagnostic remediation: {remediation!r}")
    return cast(FitDiagnosticRemediation, remediation)


def _coerce_fit_diagnostic(
    value: object,
    *,
    default_phase: FitDiagnosticPhase,
    default_dataset_id: Optional[str] = None,
) -> FitDiagnostic:
    if isinstance(value, FitDiagnostic):
        return value
    if isinstance(value, Mapping) and (
        "failure" in value or "phase" in value or "dataset_id" in value or "remediation" in value
    ):
        failure = coerce_simulation_failure(value.get("failure"))
        return FitDiagnostic(
            phase=_coerce_fit_diagnostic_phase(value.get("phase"), default_phase=default_phase),
            dataset_id=value.get("dataset_id", default_dataset_id),
            failure=failure,
            remediation=_coerce_fit_diagnostic_remediation(value.get("remediation"), failure=failure),
            parameter_snapshot=value.get("parameter_snapshot"),  # type: ignore[arg-type]
        )
    failure = coerce_simulation_failure(value)
    return FitDiagnostic(
        phase=default_phase,
        dataset_id=default_dataset_id,
        failure=failure,
        remediation=_coerce_fit_diagnostic_remediation(None, failure=failure),
    )


def _coerce_fit_detail_section(value: object) -> FitDetailSection:
    if isinstance(value, FitDetailSection):
        return value
    if isinstance(value, Mapping) and ("failure" in value or "dataset_id" in value):
        return FitDetailSection(
            dataset_id=value.get("dataset_id"),
            failure=value.get("failure"),
        )
    return FitDetailSection(
        dataset_id=getattr(value, "dataset_id", None),
        failure=getattr(value, "failure", value),
    )


@dataclass
class FitDiagnostic:
    phase: FitDiagnosticPhase
    dataset_id: Optional[str]
    failure: SimulationFailure
    remediation: FitDiagnosticRemediation | str | None = None
    parameter_snapshot: Optional[Dict[str, float]] = None

    def __post_init__(self) -> None:
        self.phase = _coerce_fit_diagnostic_phase(self.phase, default_phase="optimizer")
        self.dataset_id = str(self.dataset_id) if self.dataset_id is not None else None
        self.failure = coerce_simulation_failure(self.failure)
        self.remediation = _coerce_fit_diagnostic_remediation(self.remediation, failure=self.failure)
        self.parameter_snapshot = _copy_parameter_snapshot(self.parameter_snapshot)


@dataclass
class FitDetailSection:
    dataset_id: Optional[str]
    failure: SimulationFailure

    def __post_init__(self) -> None:
        self.dataset_id = str(self.dataset_id) if self.dataset_id is not None else None
        self.failure = coerce_simulation_failure(self.failure)


@dataclass
class GlobalFitCompletion:
    status: FitCompletionStatus
    optimizer_converged: bool
    nonfinite_metrics: bool
    optimizer_diagnostic: Optional[FitDiagnostic] = None
    dataset_failures: Dict[str, FitDiagnostic] = field(default_factory=dict)
    dataset_warnings: Dict[str, str] = field(default_factory=dict)
    detail_sections: List[FitDetailSection] = field(default_factory=list)

    def __post_init__(self) -> None:
        status = str(self.status or "")
        if status not in _VALID_COMPLETION_STATUSES:
            raise ValueError(f"Unsupported global fit completion status: {status!r}")
        self.status = cast(FitCompletionStatus, status)
        self.optimizer_converged = bool(self.optimizer_converged)
        self.nonfinite_metrics = bool(self.nonfinite_metrics)
        if self.optimizer_diagnostic is not None:
            self.optimizer_diagnostic = _coerce_fit_diagnostic(
                self.optimizer_diagnostic,
                default_phase="optimizer",
            )
        self.dataset_failures = {
            str(ds_id): _coerce_fit_diagnostic(
                diagnostic,
                default_phase="final_replay",
                default_dataset_id=str(ds_id),
            )
            for ds_id, diagnostic in dict(self.dataset_failures).items()
        }
        self.dataset_warnings = {
            str(ds_id): str(message)
            for ds_id, message in dict(self.dataset_warnings).items()
        }
        self.detail_sections = [_coerce_fit_detail_section(section) for section in list(self.detail_sections)]
        if self.nonfinite_metrics and self.status != "fail":
            raise ValueError("GlobalFitCompletion.nonfinite_metrics=True requires status='fail'.")
        if self.status == "ok":
            if not self.optimizer_converged:
                raise ValueError("GlobalFitCompletion status='ok' requires optimizer_converged=True.")
            if self.nonfinite_metrics:
                raise ValueError("GlobalFitCompletion status='ok' requires nonfinite_metrics=False.")
            if self.dataset_failures:
                raise ValueError("GlobalFitCompletion status='ok' forbids dataset_failures.")
            if self.dataset_warnings:
                raise ValueError("GlobalFitCompletion status='ok' forbids dataset_warnings.")
            if self.optimizer_diagnostic is not None:
                raise ValueError("GlobalFitCompletion status='ok' forbids optimizer_diagnostic.")
            if self.detail_sections:
                raise ValueError("GlobalFitCompletion status='ok' forbids detail_sections.")
            return
        if self.status == "warn":
            if self.nonfinite_metrics:
                raise ValueError("GlobalFitCompletion status='warn' requires nonfinite_metrics=False.")
            if self.dataset_failures:
                raise ValueError("GlobalFitCompletion status='warn' forbids dataset_failures.")
            if (
                self.optimizer_converged
                and not self.dataset_warnings
                and self.optimizer_diagnostic is None
            ):
                raise ValueError("GlobalFitCompletion status='warn' requires a real warning source.")
            return
        if not self.dataset_failures and self.optimizer_diagnostic is None:
            raise ValueError("GlobalFitCompletion status='fail' requires a real failure source.")
